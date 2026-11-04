#!/usr/bin/env python3
"""In-domain SQUID benchmark on OpenNeuro ds000117 (Elekta Neuromag-306).

Evaluates:
  1. Zero-shot BrainTokenizer reconstruction PCC across sensor configurations:
     - Full 306 SQUID channels (102 MAG + 204 GRAD)
     - 102 Magnetometers only
     - 204 Gradiometers only
     - Inverse control: Magnetometers remapped to GRAD (testing sensor-type embedding bias)
  2. Evoked Visual Detection (Pre-stimulus baseline vs. post-stimulus active response):
     - Pretrained Linear Probe vs Control A (Random Init) vs Control B (Tokenizer Only)
  3. Face vs. Scrambled Classification:
     - Pretrained Linear Probe vs Control A vs Control B

Saves results to results/summary_ds000117_squid_baseline.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from opm_bridge.eval.metrics import (
    compute_paired_permutation_p_value,
    subject_bootstrap_ci,
)
from opm_bridge.models.wrapper import load_brainomni_backbone
from opm_bridge.preprocessing.sensor_mapping import sensortype_wise_normalize

ERP_START_SAMPLE = 64
ERP_END_SAMPLE = 192
ERP_START_TOKEN = 1
ERP_END_TOKEN = 3


def get_default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_pcc(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    af = a.reshape(a.shape[0], -1)
    bf = b.reshape(b.shape[0], -1)
    am = af - af.mean(dim=1, keepdim=True)
    bm = bf - bf.mean(dim=1, keepdim=True)
    cov = (am * bm).sum(dim=1)
    den = torch.sqrt((am**2).sum(dim=1) * (bm**2).sum(dim=1) + 1e-8)
    return (cov / den).mean()


@torch.no_grad()
def evaluate_reconstruction_pcc(
    tokenizer: torch.nn.Module,
    x: torch.Tensor,
    pos_ori: torch.Tensor,
    sensor_type: torch.Tensor,
    device: torch.device,
    batch_size: int = 16,
) -> float:
    """Run forward reconstruction pass through BrainTokenizer and compute Pearson r."""
    tokenizer.eval()
    tokenizer.to(device)
    pccs: List[float] = []

    pos_dev = pos_ori.unsqueeze(0).to(device)
    st_dev = sensor_type.unsqueeze(0).to(device)

    for i in range(0, len(x), batch_size):
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{len(x)} epochs...", flush=True)
        xb = x[i : i + batch_size].to(device)
        b = len(xb)
        pos_b = pos_dev.expand(b, -1, -1)
        st_b = st_dev.expand(b, -1)

        x_unf = tokenizer.unfold(xb, overlap_ratio=0.0)
        se = tokenizer.sensor_embed(pos_b, st_b)
        feat = tokenizer.encoder(x_unf, se)
        q_feat, _, _ = tokenizer.quantizer(feat)
        x_rec = tokenizer.decoder(q_feat, se).float()
        pcc = float(get_pcc(tokenizer.norm_target(x_unf), x_rec).cpu().item())
        pccs.append(pcc)

    return float(np.mean(pccs))


@torch.no_grad()
def extract_representations(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    pos_ori: torch.Tensor,
    sensor_type: torch.Tensor,
    device: torch.device,
    batch_size: int = 4,
    pool_erp_tokens: bool = True,
) -> np.ndarray:
    """Extract representations through full forward pass."""
    backbone.eval()
    backbone.to(device)
    feats: List[np.ndarray] = []

    pos_dev = pos_ori.unsqueeze(0).to(device)
    st_dev = sensor_type.unsqueeze(0).to(device)

    for i in range(0, len(x), batch_size):
        if (i // batch_size) % 10 == 0:
            print(f"  Processed {i}/{len(x)} epochs...", flush=True)
        xb = x[i : i + batch_size].to(device)
        b = len(xb)
        pb = pos_dev.expand(b, -1, -1)
        sb = st_dev.expand(b, -1)

        x_unf = backbone.tokenizer.unfold(xb, overlap_ratio=0.0)
        b_u, c_u, n_u, l_u = x_unf.shape
        x_sea = backbone.tokenizer.encoder.seanet_encoder(
            rearrange(x_unf, "B C N L -> (B C N) 1 L")
        )
        x_sea = rearrange(x_sea, "(B C N) D T -> B C (N T) D", B=b_u, C=c_u, N=n_u)

        w = x_sea.shape[2]
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
            h_active = h[:, :, ERP_START_TOKEN:ERP_END_TOKEN, :].mean(dim=2)
            pooled = h_active.reshape(b, -1)
        else:
            pooled = h.mean(dim=2).reshape(b, -1)
        feats.append(pooled.cpu().numpy())
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    return np.concatenate(feats, axis=0)


def run_squid_benchmark(cache_path: Path, out_path: Path) -> None:
    device = get_default_device()
    print(f"Loading SQUID dataset from {cache_path}...")
    d = torch.load(cache_path, map_location="cpu", weights_only=False)

    x = d["x"]  # (N, 306, 512)
    pos_ori = d["pos_ori"]  # (306, 6)
    sensor_type = d["sensor_type"]  # (306,)
    is_face = d["is_face"].numpy()  # (N,)

    n_trials, n_channels, n_times = x.shape
    print(f"Loaded {n_trials} trials, {n_channels} channels, {n_times} timepoints. Device: {device}")

    # Identify channel groups:
    mag_mask = (sensor_type == 1)
    grad_mask = (sensor_type == 2)
    print(f"Channels: {int(mag_mask.sum())} Magnetometers, {int(grad_mask.sum())} Gradiometers")

    # Load BrainOmni models
    print("Loading Pretrained BrainOmni tiny...")
    pretrained_bb = load_brainomni_backbone(size="tiny", load_pretrained_backbone=True, load_pretrained_tokenizer=True)
    pretrained_bb.to(device)

    print("Loading Control A (Random Init) BrainOmni tiny...")
    control_a_bb = load_brainomni_backbone(size="tiny", load_pretrained_backbone=False, load_pretrained_tokenizer=False)
    control_a_bb.to(device)

    print("Loading Control B (Tokenizer Only) BrainOmni tiny...")
    control_b_bb = load_brainomni_backbone(size="tiny", load_pretrained_backbone=False, load_pretrained_tokenizer=True)
    control_b_bb.to(device)

    results: Dict[str, dict] = {}

    # =========================================================================
    # Part 1: Zero-Shot Reconstruction Pearson Correlation (PCC)
    # =========================================================================
    print("\n--- Part 1: Zero-Shot Reconstruction PCC ---")
    pcc_results = {}

    # 1. Full 306 SQUID channels
    print("Evaluating Full 306 SQUID channels...")
    pcc_full = evaluate_reconstruction_pcc(
        pretrained_bb.tokenizer, x, pos_ori, sensor_type, device
    )
    print(f"  Full 306 ch PCC: {pcc_full:.4f}")
    pcc_results["full_306ch"] = pcc_full

    # 2. 102 Magnetometers only
    print("Evaluating 102 Magnetometers only...")
    x_mag = x[:, mag_mask]
    pos_mag = pos_ori[mag_mask]
    st_mag = sensor_type[mag_mask]
    x_mag_norm = torch.stack([torch.from_numpy(sensortype_wise_normalize(x_mag[i].numpy(), st_mag.numpy())) for i in range(len(x_mag))])
    pcc_mag = evaluate_reconstruction_pcc(
        pretrained_bb.tokenizer, x_mag_norm, pos_mag, st_mag, device
    )
    print(f"  102 Mag only PCC: {pcc_mag:.4f}")
    pcc_results["mag_102ch"] = pcc_mag

    # 3. 204 Gradiometers only
    print("Evaluating 204 Gradiometers only...")
    x_grad = x[:, grad_mask]
    pos_grad = pos_ori[grad_mask]
    st_grad = sensor_type[grad_mask]
    x_grad_norm = torch.stack([torch.from_numpy(sensortype_wise_normalize(x_grad[i].numpy(), st_grad.numpy())) for i in range(len(x_grad))])
    pcc_grad = evaluate_reconstruction_pcc(
        pretrained_bb.tokenizer, x_grad_norm, pos_grad, st_grad, device
    )
    print(f"  204 Grad only PCC: {pcc_grad:.4f}")
    pcc_results["grad_204ch"] = pcc_grad

    # 4. Inverse control: Remap Magnetometers to GRAD (sensor_type = 2)
    print("Evaluating Inverse Control: SQUID Magnetometers remapped to GRAD...")
    st_mag_as_grad = torch.full_like(st_mag, 2)
    pcc_mag_as_grad = evaluate_reconstruction_pcc(
        pretrained_bb.tokenizer, x_mag_norm, pos_mag, st_mag_as_grad, device
    )
    print(f"  102 Mag as GRAD PCC: {pcc_mag_as_grad:.4f}")
    pcc_results["mag_as_grad"] = pcc_mag_as_grad
    pcc_results["delta_mag_remapped_to_grad"] = pcc_mag_as_grad - pcc_mag

    results["reconstruction_pcc"] = pcc_results

    # =========================================================================
    # Part 2: Evoked Visual Detection (Baseline vs. Post-stimulus)
    # =========================================================================
    print("\n--- Part 2: Evoked Visual Detection on SQUID ---")
    x_np = x.numpy()
    baseline_slices = x_np[:, :, :51]  # 51 samples = 200 ms
    evoked_slices = x_np[:, :, ERP_START_SAMPLE:ERP_END_SAMPLE]  # 128 samples = 500 ms

    tiled_baseline = np.tile(baseline_slices, (1, 1, int(np.ceil(128 / baseline_slices.shape[-1]))))[:, :, :128]

    container_base = np.zeros_like(x_np)
    container_base[:, :, ERP_START_SAMPLE:ERP_END_SAMPLE] = tiled_baseline

    container_evoked = np.zeros_like(x_np)
    container_evoked[:, :, ERP_START_SAMPLE:ERP_END_SAMPLE] = evoked_slices

    x_det = np.concatenate([container_base, container_evoked], axis=0)
    y_det = np.concatenate([np.zeros(n_trials, dtype=int), np.ones(n_trials, dtype=int)], axis=0)
    det_tensor = torch.from_numpy(x_det).float()

    print(f"Detection dataset: {len(x_det)} epochs (50% baseline, 50% evoked). Extracting features...")
    feats_pre = extract_representations(pretrained_bb, det_tensor, pos_ori, sensor_type, device)
    feats_ctrl_a = extract_representations(control_a_bb, det_tensor, pos_ori, sensor_type, device)
    feats_ctrl_b = extract_representations(control_b_bb, det_tensor, pos_ori, sensor_type, device)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def evaluate_probe(feats: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
        accs, aucs = [], []
        for train_idx, test_idx in cv.split(feats, y):
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(feats[train_idx])
            X_te = scaler.transform(feats[test_idx])
            clf = LogisticRegression(C=0.01, max_iter=500, random_state=42)
            clf.fit(X_tr, y[train_idx])
            probs = clf.predict_proba(X_te)[:, 1]
            preds = (probs >= 0.5).astype(int)
            accs.append(float(np.mean(preds == y[test_idx])))
            aucs.append(float(roc_auc_score(y[test_idx], probs)))
        return float(np.mean(accs)), float(np.mean(aucs))

    acc_pre, auc_pre = evaluate_probe(feats_pre, y_det)
    acc_ctrl_a, auc_ctrl_a = evaluate_probe(feats_ctrl_a, y_det)
    acc_ctrl_b, auc_ctrl_b = evaluate_probe(feats_ctrl_b, y_det)

    print(f"  Pretrained: Acc = {acc_pre*100:.2f}%, AUC = {auc_pre:.4f}")
    print(f"  Control A (Random Init): Acc = {acc_ctrl_a*100:.2f}%, AUC = {auc_ctrl_a:.4f}")
    print(f"  Control B (Tokenizer Only): Acc = {acc_ctrl_b*100:.2f}%, AUC = {auc_ctrl_b:.4f}")

    results["visual_detection"] = {
        "pretrained": {"accuracy": acc_pre, "auc": auc_pre},
        "control_a": {"accuracy": acc_ctrl_a, "auc": auc_ctrl_a},
        "control_b": {"accuracy": acc_ctrl_b, "auc": auc_ctrl_b},
        "delta_vs_ctrl_a": acc_pre - acc_ctrl_a,
    }

    # =========================================================================
    # Part 3: Face vs. Scrambled Classification
    # =========================================================================
    print("\n--- Part 3: Face vs. Scrambled Classification on SQUID ---")
    feats_pre_stim = extract_representations(pretrained_bb, x, pos_ori, sensor_type, device)
    feats_ctrl_a_stim = extract_representations(control_a_bb, x, pos_ori, sensor_type, device)
    feats_ctrl_b_stim = extract_representations(control_b_bb, x, pos_ori, sensor_type, device)

    face_pre_acc, face_pre_auc = evaluate_probe(feats_pre_stim, is_face)
    face_ctrl_a_acc, face_ctrl_a_auc = evaluate_probe(feats_ctrl_a_stim, is_face)
    face_ctrl_b_acc, face_ctrl_b_auc = evaluate_probe(feats_ctrl_b_stim, is_face)

    print(f"  Pretrained: Acc = {face_pre_acc*100:.2f}%, AUC = {face_pre_auc:.4f}")
    print(f"  Control A (Random Init): Acc = {face_ctrl_a_acc*100:.2f}%, AUC = {face_ctrl_a_auc:.4f}")
    print(f"  Control B (Tokenizer Only): Acc = {face_ctrl_b_acc*100:.2f}%, AUC = {face_ctrl_b_auc:.4f}")

    results["face_decoding"] = {
        "pretrained": {"accuracy": face_pre_acc, "auc": face_pre_auc},
        "control_a": {"accuracy": face_ctrl_a_acc, "auc": face_ctrl_a_auc},
        "control_b": {"accuracy": face_ctrl_b_acc, "auc": face_ctrl_b_auc},
        "delta_vs_ctrl_a": face_pre_acc - face_ctrl_a_acc,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll in-domain SQUID results saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SQUID in-domain benchmark")
    parser.add_argument("--cache", type=str, default="data/cache/ds000117/sub-01/sub-01_ses-meg_run-01_256hz.pt")
    parser.add_argument("--out", type=str, default="results/summary_ds000117_squid_baseline.json")
    args = parser.parse_args()
    run_squid_benchmark(Path(args.cache), Path(args.out))
