#!/usr/bin/env python3
"""OPM-Bridge Phase 2 Transfer Benchmark CLI (§4, §6, §8, §9).

Evaluates:
  1. Pretrained BrainOmni (`backbone_ft`): Official pre-trained weights, frozen tokenizer, fine-tuned backbone.
  2. Control A (`control_a_random_e2e`): Untrained/random weights, trained from scratch end-to-end.
  3. Control B (`control_b_random_backbone`): Pre-trained tokenizer (frozen), random backbone trained from scratch.
  4. CompactMEGNet: Simple Specialist CNN trained from scratch.
  5. ClassicalBandPowerDecoder: Welch log band-power + Logistic Regression.

Across:
  - Subject-wise K-folds (Leave-One-Subject-Out).
  - Nested data budgets: B in {1, 2, ...} subjects.
  - Seeds: [42, 3407].
  - Tasks: face classification tasks (default: gender, race).

Results are incrementally recorded to `results/benchmark_ledger.json` via ResumableBenchmarkLedger,
and statistical hypothesis tests (primary paired bootstrap, SESOI=0.03) are computed and exported.
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
from torch.utils.data import DataLoader, TensorDataset

from opm_bridge.eval.metrics import (
    compute_classification_metrics,
    evaluate_primary_comparison,
    subject_bootstrap_ci,
)
from opm_bridge.models.baselines import ClassicalBandPowerDecoder, CompactMEGNet
from opm_bridge.models.wrapper import BrainOmniClassifier, load_brainomni_backbone
from opm_bridge.preprocessing.sensor_mapping import apply_soft_hfc
from opm_bridge.train.runner import (
    ResumableBenchmarkLedger,
    build_subject_kfold_splits,
    seed_everything,
    select_nested_budget_subjects,
)


def get_default_device() -> torch.device:
    """Select best available compute device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


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
) -> Dict[str, float]:
    """Train a single model arm on train data and evaluate on held-out test data."""
    if device is None:
        device = get_default_device()

    seed_everything(seed)

    # 1. Classical Decoder
    if arm == "classical_bandpower":
        clf = ClassicalBandPowerDecoder(sfreq=256.0, random_state=seed).fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        probs = clf.predict_proba(X_te)
        return compute_classification_metrics(y_te, preds, probs)

    # 2. CompactMEGNet Specialist
    if arm == "compact_megnet":
        n_ch = X_tr.shape[1]
        net = CompactMEGNet(n_channels=n_ch, n_classes=2, dropout=0.25).to(device)
        optimizer = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        ds_tr = TensorDataset(torch.from_numpy(X_tr).float(), torch.from_numpy(y_tr).long())
        dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

        net.train()
        for _ in range(epochs * 2):  # CompactMEGNet trains for 2x epochs since it's tiny
            for xb, yb in dl_tr:
                xb, yb = xb.to(device), yb.to(device)
                optimizer.zero_grad()
                out = net(xb, labels=yb)
                out["loss"].backward()
                optimizer.step()

        net.eval()
        with torch.no_grad():
            x_te_t = torch.from_numpy(X_te).float().to(device)
            logits = net(x_te_t)["logits"]
            probs = F.softmax(logits, dim=-1).cpu().numpy()
            preds = logits.argmax(dim=-1).cpu().numpy()
        return compute_classification_metrics(y_te, preds, probs)

    # 3. BrainOmni Arms
    if arm == "pretrained_backbone_ft":
        backbone = load_brainomni_backbone(
            size="tiny",
            load_pretrained_backbone=True,
            load_pretrained_tokenizer=True,
        ).to(device)
        model = BrainOmniClassifier(backbone, num_classes=2, mode="backbone_ft").to(device)
        opt_lr = lr
    elif arm == "control_a_random_e2e":
        backbone = load_brainomni_backbone(
            size="tiny",
            load_pretrained_backbone=False,
            load_pretrained_tokenizer=False,
        ).to(device)
        model = BrainOmniClassifier(backbone, num_classes=2, mode="control_a_random_e2e").to(device)
        opt_lr = lr
    elif arm == "control_b_random_backbone":
        backbone = load_brainomni_backbone(
            size="tiny",
            load_pretrained_backbone=False,
            load_pretrained_tokenizer=True,
        ).to(device)
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
    task: str = "gender",
    arms: Sequence[str] = (
        "pretrained_backbone_ft",
        "control_a_random_e2e",
        "control_b_random_backbone",
        "compact_megnet",
        "classical_bandpower",
    ),
    budgets: Sequence[int] = (1, 2),
    seeds: Sequence[int] = (42, 3407),
    epochs: int = 5,
    batch_size: int = 16,
    use_soft_hfc: bool = False,
    hfc_alpha: float = 0.25,
    mapping_mode: str = "default",
    ledger_path: str | Path = "results/benchmark_ledger.json",
    summary_path: str | Path = "results/transfer_benchmark_summary.json",
) -> dict:
    """Run full resumable transfer benchmark across folds, budgets, seeds, and arms."""
    base_dir = Path(data_dir)
    subj_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if len(subj_dirs) < 2:
        raise ValueError(f"Need at least 2 subjects in {data_dir}, found {len(subj_dirs)}")

    device = get_default_device()
    print(f"Device: {device}")
    print(f"Dataset: {data_dir} ({len(subj_dirs)} subjects)")
    print(f"Task: {task}")
    print(f"Arms: {list(arms)}")
    print(f"Budgets: {list(budgets)}")
    print(f"Seeds: {list(seeds)}")

    # Load all cached runs per subject
    subj_X: Dict[str, torch.Tensor] = {}
    subj_pos: Dict[str, torch.Tensor] = {}
    subj_st: Dict[str, torch.Tensor] = {}
    subj_labels: Dict[str, torch.Tensor] = {}

    for sdir in subj_dirs:
        s_name = sdir.name
        pt_files = sorted(sdir.glob("*.pt"))
        if not pt_files:
            continue
        all_x, all_y = [], []
        ref_pos, ref_st = None, None
        for p in pt_files:
            payload = torch.load(p, weights_only=False)
            all_x.append(payload["x"].float())
            all_y.append(payload["labels"][task])
            ref_pos = payload["pos_ori"].float()
            ref_st = payload["sensor_type"].long()

        subj_X[s_name] = torch.cat(all_x, dim=0)
        subj_labels[s_name] = torch.cat(all_y, dim=0)
        subj_pos[s_name] = ref_pos
        subj_st[s_name] = ref_st

    valid_subjects = sorted(subj_X.keys())
    min_ch = min(subj_X[s].shape[1] for s in valid_subjects)
    print(f"Harmonized channels across subjects: {min_ch}")

    # Standardize channel count and prepare arrays
    X_by_subj: Dict[str, np.ndarray] = {
        s: subj_X[s][:, :min_ch, :].numpy().astype(np.float32) for s in valid_subjects
    }
    pos_by_subj: Dict[str, np.ndarray] = {
        s: subj_pos[s][:min_ch].numpy().astype(np.float32) for s in valid_subjects
    }

    if use_soft_hfc:
        print(f"Applying Soft-HFC spatial filtering (alpha={hfc_alpha}) to all subjects...")
        for s in valid_subjects:
            X_by_subj[s] = apply_soft_hfc(X_by_subj[s], pos_by_subj[s][:, 3:6], alpha=hfc_alpha)

    y_by_subj: Dict[str, np.ndarray] = {
        s: subj_labels[s].numpy().astype(np.int64) for s in valid_subjects
    }
    st_by_subj: Dict[str, np.ndarray] = {
        s: subj_st[s][:min_ch].numpy().astype(np.int64) for s in valid_subjects
    }

    if mapping_mode != "default":
        print(f"Applying sensor geometry ablation: mapping_mode={mapping_mode}")
        for s in valid_subjects:
            if mapping_mode == "opm_as_grad":
                st_by_subj[s][:] = 2
            elif mapping_mode == "zero_orientation":
                pos_by_subj[s][:, 3:] = 0.0
            elif mapping_mode == "flip_orientation":
                pos_by_subj[s][:, 3:] = -pos_by_subj[s][:, 3:]
            elif mapping_mode == "scramble_channels":
                rng = np.random.default_rng(42)
                perm = rng.permutation(min_ch)
                pos_by_subj[s] = pos_by_subj[s][perm]
                st_by_subj[s] = st_by_subj[s][perm]


    ledger = ResumableBenchmarkLedger(ledger_path)
    splits = build_subject_kfold_splits(
        valid_subjects,
        n_splits=len(valid_subjects),
        val_count=0,
        seed=42,
    )

    # Execution loop
    for arm in arms:
        for budget in budgets:
            for seed in seeds:
                for split in splits:
                    fold = split.fold_index
                    test_sub = split.test_subjects[0]
                    key = ledger.make_key(
                        dataset=f"ds005107_{task}",
                        arm=arm,
                        fold=fold,
                        budget=budget,
                        seed=seed,
                    )
                    if ledger.is_done(key):
                        continue

                    # Select nested budget training subjects
                    train_subjs = select_nested_budget_subjects(
                        split.train_pool_subjects,
                        budget_n_subjects=budget,
                        seed=seed,
                    )

                    X_tr = np.concatenate([X_by_subj[s] for s in train_subjs], axis=0)
                    y_tr = np.concatenate([y_by_subj[s] for s in train_subjs], axis=0)
                    # Broadcast sensor positions & types across trials
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
                        "dataset": f"ds005107_{task}",
                        "arm": arm,
                        "fold": fold,
                        "budget": budget,
                        "seed": seed,
                        "test_subject": test_sub,
                        "train_subjects": train_subjs,
                        "metrics": metrics,
                    }
                    ledger.record(key, record_data)
                    print(
                        f"[{arm:25s}] Budget={budget} | Seed={seed} | Fold={fold} (Test: {test_sub}) | "
                        f"BACC: {metrics['balanced_accuracy'] * 100:.2f}% | "
                        f"Macro-F1: {metrics['macro_f1'] * 100:.2f}%"
                    )
                    sys.stdout.flush()

    # Aggregate results and compute statistical hypothesis tests
    summary = aggregate_ledger_results(
        ledger=ledger,
        dataset=f"ds005107_{task}",
        arms=arms,
        budgets=budgets,
        seeds=seeds,
        n_subjects=len(valid_subjects),
    )

    out_p = Path(summary_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"\nSaved transfer benchmark summary to {out_p}")
    return summary


def aggregate_ledger_results(
    ledger: ResumableBenchmarkLedger,
    dataset: str,
    arms: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    n_subjects: int,
) -> dict:
    """Aggregate per-subject scores and execute primary hypothesis tests (§4.1, §6)."""
    scores_by_arm_budget: Dict[str, Dict[int, np.ndarray]] = {a: {} for a in arms}

    for arm in arms:
        for b in budgets:
            subj_seed_matrix = np.zeros((n_subjects, len(seeds)), dtype=np.float64)
            for s_idx, seed in enumerate(seeds):
                for fold in range(n_subjects):
                    key = ledger.make_key(dataset, arm, fold, b, seed)
                    if key in ledger.records:
                        bacc = ledger.records[key]["metrics"]["balanced_accuracy"]
                        subj_seed_matrix[fold, s_idx] = bacc
            scores_by_arm_budget[arm][b] = subj_seed_matrix

    # Hypothesis tests and CI tables
    results_by_budget: Dict[str, dict] = {}
    for b in budgets:
        b_summary: Dict[str, dict] = {}
        for arm in arms:
            mat = scores_by_arm_budget[arm][b]
            ci = subject_bootstrap_ci(mat, n_resamples=10_000, rng_seed=42)
            b_summary[arm] = {
                "mean": ci.mean,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "subject_means": np.mean(mat, axis=1).tolist(),
            }

        # Primary Pre-Registered Hypothesis Test: Pretrained vs Control A (§4.1)
        if "pretrained_backbone_ft" in arms and "control_a_random_e2e" in arms:
            pt_mat = scores_by_arm_budget["pretrained_backbone_ft"][b]
            ctrl_a_mat = scores_by_arm_budget["control_a_random_e2e"][b]
            test_res = evaluate_primary_comparison(
                pretrained_scores=pt_mat,
                control_a_scores=ctrl_a_mat,
                budget=b,
                sesoi=0.03,
                n_resamples=10_000,
                rng_seed=42,
            )
            b_summary["primary_hypothesis_test_rq1"] = {
                "delta_mean": test_res.delta_mean,
                "delta_ci_low": test_res.delta_ci_low,
                "delta_ci_high": test_res.delta_ci_high,
                "sesoi": test_res.sesoi,
                "verdict": test_res.verdict,
            }

        # Secondary RQ2: Pretrained vs CompactMEGNet Specialist
        if "pretrained_backbone_ft" in arms and "compact_megnet" in arms:
            pt_means = np.mean(scores_by_arm_budget["pretrained_backbone_ft"][b], axis=1)
            spec_means = np.mean(scores_by_arm_budget["compact_megnet"][b], axis=1)
            diff = pt_means - spec_means
            diff_ci = subject_bootstrap_ci(diff, n_resamples=10_000, rng_seed=42)
            b_summary["secondary_rq2_vs_specialist"] = {
                "delta_mean": diff_ci.mean,
                "delta_ci_low": diff_ci.ci_low,
                "delta_ci_high": diff_ci.ci_high,
            }

        results_by_budget[str(b)] = b_summary

    return {
        "dataset": dataset,
        "budgets": list(budgets),
        "arms": list(arms),
        "seeds": list(seeds),
        "results_by_budget": results_by_budget,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OPM-Bridge Transfer Benchmark")
    parser.add_argument("--data-dir", type=str, default="data/cache/ds005107")
    parser.add_argument("--task", type=str, default="gender", choices=["gender", "race", "age", "emotion"])
    parser.add_argument(
        "--arms",
        nargs="+",
        default=[
            "pretrained_backbone_ft",
            "control_a_random_e2e",
            "control_b_random_backbone",
            "compact_megnet",
            "classical_bandpower",
        ],
    )
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 3407])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--soft-hfc", action="store_true", help="Apply Soft-HFC spatial filtering")
    parser.add_argument("--hfc-alpha", type=float, default=0.25, help="Soft-HFC shrinkage factor (default: 0.25)")
    parser.add_argument(
        "--mapping-mode",
        type=str,
        default="default",
        choices=["default", "opm_as_grad", "zero_orientation", "flip_orientation", "scramble_channels"],
        help="Sensor geometry ablation mode for Phase 3 (FR7 / RQ3)",
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
