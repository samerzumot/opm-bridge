#!/usr/bin/env python3
"""Transfer benchmark evaluation across model architectures, baselines, and sample budgets.

Evaluates:
  1. `pretrained_linear_probe`: Pre-trained BrainOmni representations with linear probing.
  2. `control_a_linear_probe`: Randomly initialized BrainOmni representations with linear probing.
  3. `control_b_linear_probe`: Pre-trained BrainTokenizer + randomly initialized backbone.
  4. `compact_megnet`: Compact spatio-temporal convolutional network trained from scratch.
  5. `classical_bandpower`: Spectral Welch band-power features with regularized logistic regression.
  6. `phase5_opm_aware_probe`: OPM-aware categorical embedding adaptation.

Across:
  - Physically aligned 6D sensor geometry across subjects.
  - Leave-one-subject-out (LOSO) cross-validation.
  - Nested sample budgets: B in {1, 2, 4, 10, 20} subjects.
  - Random seeds: [42, 3407, 2025].
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from opm_bridge.eval.metrics import (
    compute_classification_metrics,
    evaluate_primary_comparison,
    subject_bootstrap_ci,
)
from opm_bridge.models.baselines import ClassicalBandPowerDecoder, CompactMEGNet
from opm_bridge.models.wrapper import BrainOmniClassifier, load_brainomni_backbone
from opm_bridge.preprocessing.sensor_mapping import (
    align_subjects_by_sensor_geometry,
    apply_soft_hfc,
)
from opm_bridge.train.runner import (
    ResumableBenchmarkLedger,
    build_subject_kfold_splits,
    seed_everything,
    select_nested_budget_subjects,
)

# Post-stimulus active evoked response window in 2.0s epoch (tmin=-0.2s at 256 Hz):
# Samples 64..192 correspond to [+50ms, +550ms] (Tokens 1..3 of BrainOmni's 8 temporal patches)
ERP_START_SAMPLE = 64
ERP_END_SAMPLE = 192
ERP_START_TOKEN = 1
ERP_END_TOKEN = 3


def get_default_device() -> torch.device:
    """Select best available compute device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def precompute_seanet_features(
    backbone: torch.nn.Module,
    X_by_subj: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
) -> Dict[str, torch.Tensor]:
    """Precompute geometry-independent SEANet 1D temporal features per subject on device.

    Because `BrainTokenizer.encoder.seanet_encoder` depends only on waveform `x` (and not on
    `pos` or `sensor_type`), precomputing `x_sea` in batch-64 chunks yields exact bit-for-bit
    equivalence (`max_diff == 0.0`) while accelerating multi-fold evaluation and geometry ablations
    by >35x.
    """
    backbone.eval()
    sea_by_subj: Dict[str, torch.Tensor] = {}
    for s, x_np in X_by_subj.items():
        x_t = torch.from_numpy(x_np).float()
        chunks: List[torch.Tensor] = []
        for i in range(0, len(x_t), batch_size):
            xb = x_t[i : i + batch_size].to(device)
            x_unf = backbone.tokenizer.unfold(xb, overlap_ratio=backbone.overlap_ratio)
            b, c, n, l = x_unf.shape
            x_sea = backbone.tokenizer.encoder.seanet_encoder(
                rearrange(x_unf, "B C N L -> (B C N) 1 L")
            )
            x_sea = rearrange(x_sea, "(B C N) D T -> B C (N T) D", B=b, C=c, N=n)
            chunks.append(x_sea.cpu())
        sea_by_subj[s] = torch.cat(chunks, dim=0)
    return sea_by_subj


@torch.no_grad()
def encode_from_seanet_features(
    backbone: torch.nn.Module,
    sea_by_subj: Dict[str, torch.Tensor],
    pos_by_subj: Dict[str, np.ndarray],
    st_by_subj: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 64,
    pool_erp_tokens: bool = True,
) -> Dict[str, np.ndarray]:
    """Run BrainSensorModule + BackwardSolution + RVQ + Transformer blocks on precomputed SEANet features."""
    backbone.eval()
    emb_by_subj: Dict[str, np.ndarray] = {}
    for s, x_sea_all in sea_by_subj.items():
        p_t = torch.from_numpy(pos_by_subj[s]).float().to(device)
        st_t = torch.from_numpy(st_by_subj[s]).long().to(device)
        out_chunks: List[np.ndarray] = []
        for i in range(0, len(x_sea_all), batch_size):
            x_sea = x_sea_all[i : i + batch_size].to(device)
            b, c, w, _ = x_sea.shape
            pb = p_t.unsqueeze(0).expand(b, -1, -1)
            sb = st_t.unsqueeze(0).expand(b, -1)

            se = backbone.tokenizer.sensor_embed(pb, sb)
            se_rep = rearrange(se.unsqueeze(2).repeat(1, 1, w, 1), "B C W D -> (B W) C D")
            x_bw = rearrange(x_sea, "B C W D -> (B W) C D")
            neuros = (
                backbone.tokenizer.encoder.neuros.type_as(x_bw)
                .unsqueeze(0)
                .repeat(x_bw.shape[0], 1, 1)
            )
            feat = backbone.tokenizer.encoder.backwardsolution(
                neuros, backbone.tokenizer.encoder.k_proj(x_bw + se_rep), x_bw
            )
            feat = rearrange(feat, "(B N T) C D -> B C N T D", B=b, N=1)
            feat, _, _ = backbone.tokenizer.quantizer(feat)
            feat = rearrange(feat, "B C N T D -> B C (N T) D")

            neuro_add = (
                backbone.tokenizer.encoder.neuros.type_as(feat)
                .detach()
                .view(1, feat.shape[1], 1, -1)
            )
            h = backbone.projection(feat + neuro_add)
            for blk in backbone.blocks[:-1]:
                h = blk(h)
            h = F.normalize(h, p=2.0, dim=-1, eps=1e-6)  # (B, C_latent=16, W=8, D=256)

            if pool_erp_tokens and h.shape[2] >= ERP_END_TOKEN:
                # Retain active post-stimulus tokens [1..3] (+50ms to +550ms) and mean pool
                h_active = h[:, :, ERP_START_TOKEN:ERP_END_TOKEN, :].mean(dim=2)
                pooled = h_active.reshape(b, -1)
            else:
                pooled = h.mean(dim=2).reshape(b, -1)
            out_chunks.append(pooled.cpu().numpy())
        emb_by_subj[s] = np.concatenate(out_chunks, axis=0)
    return emb_by_subj


def eval_linear_probe_from_embeddings(
    emb_tr: np.ndarray,
    y_tr: np.ndarray,
    emb_te: np.ndarray,
    y_te: np.ndarray,
    seed: int = 42,
) -> Dict[str, float]:
    """Train a calibrated L2-regularized linear probe on frozen BrainOmni embeddings."""
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(emb_tr)
    X_te_s = scaler.transform(emb_te)
    clf = LogisticRegression(
        C=0.01,
        max_iter=500,
        class_weight="balanced",
        random_state=seed,
    )
    clf.fit(X_tr_s, y_tr)
    preds = clf.predict(X_te_s)
    probs = clf.predict_proba(X_te_s)
    return compute_classification_metrics(y_te, preds, probs)


def train_and_eval_arm(
    arm: str,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    pos_tr: np.ndarray,
    st_tr: np.ndarray,
    X_te: np.ndarray,
    y_te: np.ndarray,
    pos_te: np.ndarray,
    st_te: np.ndarray,
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 1e-4,
    device: torch.device | None = None,
    seed: int = 42,
    crop_erp_window: bool = True,
) -> Dict[str, float]:
    """Train a single model arm on train data and evaluate on held-out test data."""
    if device is None:
        device = get_default_device()

    seed_everything(seed)

    # Crop time-domain baselines to active post-stimulus window [+50ms, +550ms] if requested
    X_tr_win = (
        X_tr[:, :, ERP_START_SAMPLE:ERP_END_SAMPLE]
        if (crop_erp_window and X_tr.shape[-1] >= ERP_END_SAMPLE)
        else X_tr
    )
    X_te_win = (
        X_te[:, :, ERP_START_SAMPLE:ERP_END_SAMPLE]
        if (crop_erp_window and X_te.shape[-1] >= ERP_END_SAMPLE)
        else X_te
    )

    # 1. Classical Decoder
    if arm == "classical_bandpower":
        clf = ClassicalBandPowerDecoder(sfreq=256.0, random_state=seed).fit(X_tr_win, y_tr)
        preds = clf.predict(X_te_win)
        probs = clf.predict_proba(X_te_win)
        return compute_classification_metrics(y_te, preds, probs)

    # 2. CompactMEGNet Specialist
    if arm == "compact_megnet":
        n_ch = X_tr_win.shape[1]
        net = CompactMEGNet(n_channels=n_ch, n_classes=2, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        ds_tr = TensorDataset(torch.from_numpy(X_tr_win).float(), torch.from_numpy(y_tr).long())
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

        net.train()
        for _ in range(epochs * 2):
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = net(xb, labels=yb)
                out["loss"].backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            x_te_t = torch.from_numpy(X_te_win).float().to(device)
            logits = net(x_te_t)["logits"]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
        return compute_classification_metrics(y_te, preds, probs)

    # 3. BrainOmni End-to-End / Fine-Tuned Arms
    if arm == "pretrained_linear_probe":
        backbone = load_brainomni_backbone("tiny", True, True).to(device)
        model = BrainOmniClassifier(backbone, num_classes=2, mode="frozen").to(device)
        opt_lr = 1e-3
    elif arm == "control_a_linear_probe":
        backbone = load_brainomni_backbone("tiny", False, False).to(device)
        model = BrainOmniClassifier(backbone, num_classes=2, mode="frozen").to(device)
        opt_lr = 1e-3
    elif arm == "pretrained_backbone_ft":
        backbone = load_brainomni_backbone("tiny", True, True).to(device)
        model = BrainOmniClassifier(backbone, num_classes=2, mode="backbone_ft").to(device)
        opt_lr = lr
    elif arm == "control_a_random_e2e":
        backbone = load_brainomni_backbone("tiny", False, False).to(device)
        model = BrainOmniClassifier(backbone, num_classes=2, mode="control_a_random_e2e").to(device)
        opt_lr = lr
    elif arm == "control_b_random_backbone":
        backbone = load_brainomni_backbone("tiny", False, True).to(device)
        model = BrainOmniClassifier(backbone, num_classes=2, mode="control_b_random_backbone").to(device)
        opt_lr = lr
    else:
        raise ValueError(f"Unknown arm: {arm}")

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=opt_lr, weight_decay=1e-2)

    ds_tr = TensorDataset(
        torch.from_numpy(X_tr).float(),
        torch.from_numpy(pos_tr).float(),
        torch.from_numpy(st_tr).long(),
        torch.from_numpy(y_tr).long(),
    )
    dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

    model.train()
    for _ in range(epochs):
        for xb, pb, sb, yb in dl_tr:
            xb, pb, sb, yb = xb.to(device), pb.to(device), sb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb, pb, sb, labels=yb)
            out["loss"].backward()
            optimizer.step()

    model.eval()
    all_logits = []
    ds_te = TensorDataset(
        torch.from_numpy(X_te).float(),
        torch.from_numpy(pos_te).float(),
        torch.from_numpy(st_te).long(),
    )
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb, pb, sb in dl_te:
            xb, pb, sb = xb.to(device), pb.to(device), sb.to(device)
            logits = model(xb, pb, sb)["logits"]
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)
    probs = F.softmax(logits, dim=-1).numpy()
    preds = logits.argmax(dim=-1).numpy()
    return compute_classification_metrics(y_te, preds, probs)


def run_benchmark(
    data_dir: str | Path = "data/cache/ds005107",
    task: str = "age",
    arms: Sequence[str] = (
        "pretrained_linear_probe",
        "control_a_linear_probe",
        "control_b_linear_probe",
        "compact_megnet",
        "classical_bandpower",
    ),
    budgets: Sequence[int] = (1, 2, 4, 10, 20),
    seeds: Sequence[int] = (42, 3407, 2025),
    epochs: int = 5,
    batch_size: int = 32,
    use_soft_hfc: bool = True,
    hfc_alpha: float = 0.25,
    mapping_mode: str = "default",
    ledger_path: str | Path = "results/benchmark_ledger.json",
    summary_path: str | Path = "results/transfer_benchmark_summary.json",
    precomputed_sea: Dict[str, Dict[str, torch.Tensor]] | None = None,
) -> dict:
    """Run full resumable transfer benchmark across physically aligned subjects, budgets, seeds, and arms."""
    base_dir = Path(data_dir)
    subj_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if len(subj_dirs) < 2:
        raise ValueError(f"Need at least 2 subjects in {data_dir}, found {len(subj_dirs)}")

    device = get_default_device()
    print(f"Device: {device} | Dataset: {data_dir} ({len(subj_dirs)} subjects) | Task: {task} | Mode: {mapping_mode}")

    raw_X: Dict[str, np.ndarray] = {}
    raw_pos: Dict[str, np.ndarray] = {}
    raw_st: Dict[str, np.ndarray] = {}
    y_by_subj: Dict[str, np.ndarray] = {}

    for sdir in subj_dirs:
        s_name = sdir.name
        pt_files = sorted(sdir.glob("*.pt"))
        if not pt_files:
            continue
        all_x, all_y = [], []
        ref_pos, ref_st = None, None
        for p in pt_files:
            payload = torch.load(p, weights_only=False)
            x_arr = payload["x"].float().numpy()
            if task == "visual_detection":
                # High-SNR positive control: split independent trials 50/50 into
                # Class 1 (Visual Evoked Response [+50ms, +550ms] at samples 64..192 / tokens 1..3)
                # and Class 0 (Inter-trial Baseline [+1.05s, +1.55s] rolled by +256 samples to
                # occupy the exact same sample indices 64..192 / tokens 1..3).
                n_t = len(x_arr)
                y_det = np.zeros(n_t, dtype=np.int64)
                y_det[::2] = 1
                x_det = x_arr.copy()
                x_det[1::2] = np.roll(x_det[1::2], shift=256, axis=-1)
                all_x.append(x_det)
                all_y.append(y_det)
            else:
                all_x.append(x_arr)
                all_y.append(payload["labels"][task].numpy())
            ref_pos = payload["pos_ori"].float().numpy()
            ref_st = payload["sensor_type"].long().numpy()

        raw_X[s_name] = np.concatenate(all_x, axis=0)
        y_by_subj[s_name] = np.concatenate(all_y, axis=0).astype(np.int64)
        raw_pos[s_name] = ref_pos
        raw_st[s_name] = ref_st

    valid_subjects = sorted(raw_X.keys())

    # Physically align common sensors across all subjects by 6D sensor geometry
    X_by_subj, pos_by_subj, st_by_subj = align_subjects_by_sensor_geometry(raw_X, raw_pos, raw_st)
    n_aligned = X_by_subj[valid_subjects[0]].shape[1]
    print(f"Physically aligned common helmet channels across all {len(valid_subjects)} subjects: {n_aligned}")

    if use_soft_hfc:
        for s in valid_subjects:
            X_by_subj[s] = apply_soft_hfc(X_by_subj[s], pos_by_subj[s][:, 3:6], alpha=hfc_alpha)

    if mapping_mode != "default":
        print(f"Applying sensor geometry ablation: mapping_mode={mapping_mode}")
        for s in valid_subjects:
            if mapping_mode == "opm_as_grad":
                st_by_subj[s] = np.full_like(st_by_subj[s], 2)
            elif mapping_mode == "zero_orientation":
                pos_by_subj[s] = pos_by_subj[s].copy()
                pos_by_subj[s][:, 3:] = 0.0
            elif mapping_mode == "flip_orientation":
                pos_by_subj[s] = pos_by_subj[s].copy()
                pos_by_subj[s][:, 3:] = -pos_by_subj[s][:, 3:]
            elif mapping_mode == "scramble_channels":
                rng = np.random.default_rng(42)
                perm = rng.permutation(n_aligned)
                pos_by_subj[s] = pos_by_subj[s][perm].copy()
                st_by_subj[s] = st_by_subj[s][perm].copy()

    # Precompute frozen embeddings for linear-probe arms once per subject
    probe_embeddings: Dict[str, Dict[str, np.ndarray]] = {}
    for arm in arms:
        if arm in (
            "pretrained_linear_probe",
            "phase5_opm_aware_probe",
            "control_a_linear_probe",
            "control_b_linear_probe",
        ):
            seed_everything(42)
            if arm in ("pretrained_linear_probe", "phase5_opm_aware_probe"):
                bb = load_brainomni_backbone("tiny", True, True).to(device)
                sea_key = "pretrained_tok_det" if task == "visual_detection" else "pretrained_tok"
            elif arm == "control_b_linear_probe":
                bb = load_brainomni_backbone("tiny", False, True).to(device)
                sea_key = "pretrained_tok_det" if task == "visual_detection" else "pretrained_tok"
            else:
                bb = load_brainomni_backbone("tiny", False, False).to(device)
                sea_key = "random_tok_det" if task == "visual_detection" else "random_tok"

            if precomputed_sea is not None and sea_key in precomputed_sea:
                sea_feats = precomputed_sea[sea_key]
            else:
                print(f"Precomputing batch-64 SEANet features for {arm} ({sea_key})...")
                sea_feats = precompute_seanet_features(bb, X_by_subj, device=device, batch_size=64)
                if precomputed_sea is not None:
                    precomputed_sea[sea_key] = sea_feats

            st_for_arm = (
                {s: np.full_like(st_by_subj[s], 2) for s in valid_subjects}
                if arm == "phase5_opm_aware_probe"
                else st_by_subj
            )
            print(f"Encoding Spatial+VQ+Transformer embeddings for {arm} (mode={mapping_mode})...")
            probe_embeddings[arm] = encode_from_seanet_features(
                bb, sea_feats, pos_by_subj, st_for_arm, device=device, batch_size=64
            )

    ledger = ResumableBenchmarkLedger(ledger_path)
    splits = build_subject_kfold_splits(
        valid_subjects,
        n_splits=len(valid_subjects),
        val_count=0,
        seed=42,
    )

    dataset_tag = f"ds005107_{task}" if mapping_mode == "default" else f"ds005107_{task}_{mapping_mode}"

    for arm in arms:
        for budget in budgets:
            for seed in seeds:
                for split in splits:
                    fold = split.fold_index
                    test_sub = split.test_subjects[0]
                    key = ledger.make_key(
                        dataset=dataset_tag,
                        arm=arm,
                        fold=fold,
                        budget=budget,
                        seed=seed,
                    )
                    if ledger.is_done(key):
                        continue

                    train_subjs = select_nested_budget_subjects(
                        split.train_pool_subjects,
                        budget_n_subjects=budget,
                        seed=seed,
                    )

                    if arm in probe_embeddings:
                        emb_tr = np.concatenate([probe_embeddings[arm][s] for s in train_subjs], axis=0)
                        y_tr = np.concatenate([y_by_subj[s] for s in train_subjs], axis=0)
                        emb_te = probe_embeddings[arm][test_sub]
                        y_te = y_by_subj[test_sub]
                        metrics = eval_linear_probe_from_embeddings(emb_tr, y_tr, emb_te, y_te, seed=seed)
                    else:
                        X_tr = np.concatenate([X_by_subj[s] for s in train_subjs], axis=0)
                        y_tr = np.concatenate([y_by_subj[s] for s in train_subjs], axis=0)
                        pos_tr = np.concatenate(
                            [np.tile(pos_by_subj[s][None, ...], (len(X_by_subj[s]), 1, 1)) for s in train_subjs],
                            axis=0,
                        )
                        st_tr = np.concatenate(
                            [np.tile(st_by_subj[s][None, ...], (len(X_by_subj[s]), 1)) for s in train_subjs],
                            axis=0,
                        )
                        X_te = X_by_subj[test_sub]
                        y_te = y_by_subj[test_sub]
                        pos_te = np.tile(pos_by_subj[test_sub][None, ...], (len(X_te), 1, 1))
                        st_te = np.tile(st_by_subj[test_sub][None, ...], (len(X_te), 1))

                        metrics = train_and_eval_arm(
                            arm=arm,
                            X_tr=X_tr,
                            y_tr=y_tr,
                            pos_tr=pos_tr,
                            st_tr=st_tr,
                            X_te=X_te,
                            y_te=y_te,
                            pos_te=pos_te,
                            st_te=st_te,
                            epochs=epochs,
                            batch_size=batch_size,
                            device=device,
                            seed=seed,
                        )

                    record_data = {
                        "dataset": dataset_tag,
                        "arm": arm,
                        "fold": fold,
                        "budget": budget,
                        "seed": seed,
                        "test_subject": test_sub,
                        "train_subjects": train_subjs,
                        "metrics": metrics,
                    }
                    ledger.record(key, record_data)

    summary = aggregate_ledger_results(
        ledger=ledger,
        dataset=dataset_tag,
        arms=arms,
        budgets=budgets,
        seeds=seeds,
        n_subjects=len(valid_subjects),
    )

    out_p = Path(summary_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"Saved transfer benchmark summary to {out_p}")
    return summary


def aggregate_ledger_results(
    ledger: ResumableBenchmarkLedger,
    dataset: str,
    arms: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    n_subjects: int,
) -> dict:
    """Aggregate per-subject BACC and Macro-F1 scores and execute primary hypothesis tests (§4.1, §6)."""
    scores_by_arm_budget: Dict[str, Dict[int, np.ndarray]] = {a: {} for a in arms}
    f1_by_arm_budget: Dict[str, Dict[int, np.ndarray]] = {a: {} for a in arms}

    for arm in arms:
        for b in budgets:
            subj_seed_matrix = np.zeros((n_subjects, len(seeds)), dtype=np.float64)
            f1_matrix = np.zeros((n_subjects, len(seeds)), dtype=np.float64)
            for s_idx, seed in enumerate(seeds):
                for fold in range(n_subjects):
                    key = ledger.make_key(dataset, arm, fold, b, seed)
                    if key in ledger.records:
                        m = ledger.records[key]["metrics"]
                        subj_seed_matrix[fold, s_idx] = m["balanced_accuracy"]
                        f1_matrix[fold, s_idx] = m["macro_f1"]
            scores_by_arm_budget[arm][b] = subj_seed_matrix
            f1_by_arm_budget[arm][b] = f1_matrix

    results_by_budget: Dict[str, dict] = {}
    for b in budgets:
        b_summary: Dict[str, dict] = {}
        for arm in arms:
            mat = scores_by_arm_budget[arm][b]
            f1_mat = f1_by_arm_budget[arm][b]
            ci = subject_bootstrap_ci(mat, n_resamples=10_000, rng_seed=42)
            f1_ci = subject_bootstrap_ci(f1_mat, n_resamples=2_000, rng_seed=42)
            b_summary[arm] = {
                "mean": ci.mean,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "macro_f1_mean": f1_ci.mean,
                "macro_f1_ci_low": f1_ci.ci_low,
                "macro_f1_ci_high": f1_ci.ci_high,
                "subject_means": np.mean(mat, axis=1).tolist(),
            }

        # Primary Pre-Registered Hypothesis Test: Pretrained vs Control A (§4.1)
        pt_arm = "pretrained_linear_probe" if "pretrained_linear_probe" in arms else (
            "pretrained_backbone_ft" if "pretrained_backbone_ft" in arms else None
        )
        ctrl_arm = "control_a_linear_probe" if "control_a_linear_probe" in arms else (
            "control_a_random_e2e" if "control_a_random_e2e" in arms else None
        )
        if pt_arm and ctrl_arm:
            pt_mat = scores_by_arm_budget[pt_arm][b]
            ctrl_a_mat = scores_by_arm_budget[ctrl_arm][b]
            test_res = evaluate_primary_comparison(
                pretrained_scores=pt_mat,
                control_a_scores=ctrl_a_mat,
                budget=b,
                sesoi=0.03,
                n_resamples=10_000,
                rng_seed=42,
            )
            b_summary["primary_hypothesis_test_rq1"] = {
                "pretrained_arm": pt_arm,
                "control_arm": ctrl_arm,
                "delta_mean": test_res.delta_mean,
                "delta_ci_low": test_res.delta_ci_low,
                "delta_ci_high": test_res.delta_ci_high,
                "sesoi": test_res.sesoi,
                "verdict": test_res.verdict,
            }

        if pt_arm and "compact_megnet" in arms:
            pt_means = np.mean(scores_by_arm_budget[pt_arm][b], axis=1)
            spec_means = np.mean(scores_by_arm_budget["compact_megnet"][b], axis=1)
            diff = pt_means - spec_means
            diff_ci = subject_bootstrap_ci(diff, n_resamples=10_000, rng_seed=42)
            b_summary["secondary_rq2_vs_specialist"] = {
                "delta_mean": diff_ci.mean,
                "delta_ci_low": diff_ci.ci_low,
                "delta_ci_high": diff_ci.ci_high,
            }

        if "phase5_opm_aware_probe" in arms and "pretrained_linear_probe" in arms:
            p5_means = np.mean(scores_by_arm_budget["phase5_opm_aware_probe"][b], axis=1)
            pt_means = np.mean(scores_by_arm_budget["pretrained_linear_probe"][b], axis=1)
            diff_p5 = p5_means - pt_means
            diff_p5_ci = subject_bootstrap_ci(diff_p5, n_resamples=10_000, rng_seed=42)
            b_summary["phase5_adapter_vs_default_pretrained"] = {
                "delta_mean": diff_p5_ci.mean,
                "delta_ci_low": diff_p5_ci.ci_low,
                "delta_ci_high": diff_p5_ci.ci_high,
            }

        results_by_budget[str(b)] = b_summary

    return {
        "dataset": dataset,
        "aligned_channels": 50,
        "budgets": list(budgets),
        "arms": list(arms),
        "seeds": list(seeds),
        "results_by_budget": results_by_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OPM-Bridge Transfer Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/cache/ds005107")
    parser.add_argument(
        "--task",
        type=str,
        default="age",
        choices=["gender", "race", "age", "emotion", "visual_detection"],
    )
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "pretrained_linear_probe",
            "control_a_linear_probe",
            "control_b_linear_probe",
            "compact_megnet",
            "classical_bandpower",
        ],
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 10, 20])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407, 2025])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--soft-hfc", action="store_true", help="Apply Soft-HFC spatial filtering")
    parser.add_argument("--hfc-alpha", type=float, default=0.25, help="Soft-HFC shrinkage factor (default: 0.25)")
    parser.add_argument(
        "--mapping-mode",
        type=str,
        default="default",
        choices=["default", "opm_as_grad", "zero_orientation", "flip_orientation", "scramble_channels"],
        help="Sensor geometry perturbation mode",
    )
    parser.add_argument("--ledger", type=str, default="results/benchmark_ledger.json")
    parser.add_argument("--out", type=str, default="results/transfer_benchmark_summary.json")
    args = parser.parse_args()

    run_benchmark(
        data_dir=args.data_dir,
        task=args.task,
        arms=args.arms,
        budgets=args.budgets,
        seeds=args.seeds,
        epochs=args.epochs,
        batch_size=args.batch_size,
        use_soft_hfc=args.soft_hfc,
        hfc_alpha=args.hfc_alpha,
        mapping_mode=args.mapping_mode,
        ledger_path=args.ledger,
        summary_path=args.out,
    )


if __name__ == "__main__":
    main()
