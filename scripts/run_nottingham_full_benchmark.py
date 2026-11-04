#!/usr/bin/env python3
"""Run Nottingham Triaxial Evaluation with Capacity Controls across full-length runs.

Evaluates on 84 trials/subject across all 10 coregistered adult participants:
  1. `triaxial_xyz_raw`: All 155-176 channels through monolithic 16-query tokenizer.
  2. `triaxial_xyz_softhfc`: All channels with Soft-HFC spatial filtering.
  3. `radial_only_z`: Only Z-axis radial channels (~55 channels).
  4. `tangential_only_xy`: Only X/Y tangential channels (~105 channels).
  5. `random_3way_split`: Capacity Control - Randomly partition channels into 3 size-matched groups.
  6. `phase5_axis_split_xyz`: Structurally decompose into Z, X, Y directional sub-arrays.

Standardizes:
  - Harmonized token pooling across all arms (tokens 1:3).
  - Paired sign-flip permutation p-values against monolithic triaxial raw.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from einops import rearrange
from sklearn.model_selection import StratifiedKFold

from opm_bridge.data.schema import CoordinateFrame, SensorMetadata
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opm_bridge.eval.metrics import (
    compute_paired_permutation_p_value,
    compute_classification_metrics,
    eval_linear_probe_from_embeddings,
    subject_bootstrap_ci,
)
from opm_bridge.models.baselines import ClassicalBandPowerDecoder
from opm_bridge.models.vendor_brainomni.model_utils.loss import get_pcc
from opm_bridge.models.wrapper import (
    OPMAwareBrainOmniAdapter,
    load_brainomni_backbone,
)
from scripts.run_transfer_benchmark import (
    precompute_seanet_features,
    encode_from_seanet_features,
)
from opm_bridge.preprocessing.sensor_mapping import (
    apply_soft_hfc,
    normalize_sensor_positions,
    sensortype_wise_normalize,
)

torch.serialization.add_safe_globals([SensorMetadata, CoordinateFrame])


def evaluate_zero_shot_tokenizer_pcc(
    backbone: torch.nn.Module,
    X_dict: Dict[str, np.ndarray],
    pos_dict: Dict[str, np.ndarray],
    st_dict: Dict[str, np.ndarray],
    device: torch.device,
    batch_size: int = 16,
) -> float:
    """Compute mean zero-shot BrainTokenizer waveform reconstruction PCC in small batches to preserve memory."""
    backbone.eval()
    tok = backbone.tokenizer
    pccs: List[float] = []
    with torch.no_grad():
        for s, x_np in X_dict.items():
            x_t = torch.from_numpy(x_np).float()
            p_t = torch.from_numpy(pos_dict[s]).float().to(device)
            st_t = torch.from_numpy(st_dict[s]).long().to(device)

            subj_pccs: List[float] = []
            for i in range(0, len(x_t), batch_size):
                xb = x_t[i : i + batch_size].to(device)
                b = len(xb)
                pb = p_t.unsqueeze(0).expand(b, -1, -1)
                sb = st_t.unsqueeze(0).expand(b, -1)

                x_unf = tok.unfold(xb, overlap_ratio=0.0)
                se = tok.sensor_embed(pb, sb)
                feat = tok.encoder(x_unf, se)
                q_feat, _, _ = tok.quantizer(feat)
                x_rec = tok.decoder(q_feat, se).float()
                pcc = float(get_pcc(tok.norm_target(x_unf), x_rec).cpu().item())
                subj_pccs.append(pcc)
            pccs.append(float(np.mean(subj_pccs)))
    return float(np.mean(pccs))


def reconstruct_random_split(
    backbone: torch.nn.Module,
    x: torch.Tensor,
    pos: torch.Tensor,
    st: torch.Tensor,
    n_splits: int = 3,
    seed: int = 42,
    batch_size: int = 16,
) -> float:
    """Capacity control: reconstruct via N independent random groups matching the 3-way axis split in small batches."""
    backbone.eval()
    tok = backbone.tokenizer
    n_ch = x.shape[1]
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_ch)
    groups = np.array_split(perm, n_splits)

    batch_pccs: List[float] = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            xb = x[i : i + batch_size]
            b = len(xb)
            pb = pos[:b]
            sb = st[:b]

            x_unf_full = tok.unfold(xb, overlap_ratio=0.0)
            x_target_full = tok.norm_target(x_unf_full)
            x_rec_full = torch.zeros_like(x_target_full)

            for grp in groups:
                idx_t = torch.tensor(grp, dtype=torch.long, device=xb.device)
                x_sub = xb.index_select(dim=1, index=idx_t)
                pos_sub = pb.index_select(dim=1, index=idx_t)
                st_sub = sb.index_select(dim=1, index=idx_t)

                x_unf = tok.unfold(x_sub, overlap_ratio=0.0)
                se = tok.sensor_embed(pos_sub, st_sub)
                feat = tok.encoder(x_unf, se)
                q_feat, _, _ = tok.quantizer(feat)
                x_rec_sub = tok.decoder(q_feat, se).float()
                x_rec_full.index_copy_(dim=1, index=idx_t, source=x_rec_sub)

            batch_pccs.append(float(get_pcc(x_target_full, x_rec_full).cpu().item()))

    return float(np.mean(batch_pccs))


def main() -> None:
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Loading full Nottingham 84-trial datasets...")

    bb_pt = load_brainomni_backbone("tiny", True, True).to(device)
    bb_rand = load_brainomni_backbone("tiny", False, False).to(device)
    dec = ClassicalBandPowerDecoder(sfreq=256.0)

    adapter_pt = OPMAwareBrainOmniAdapter(bb_pt, remap_to_grad=False, axis_split_triaxial=True)
    adapter_rand = OPMAwareBrainOmniAdapter(bb_rand, remap_to_grad=False, axis_split_triaxial=True)

    pt_files = sorted(Path("data/cache/nottingham").glob("sub-1*_256hz.pt"))
    assert len(pt_files) == 10, f"Expected 10 adult subjects, found {len(pt_files)}"

    geom_modes = [
        "triaxial_xyz_raw",
        "triaxial_xyz_softhfc",
        "radial_only_z",
        "tangential_only_xy",
        "random_3way_split",
        "phase5_axis_split_xyz",
    ]

    subj_results: Dict[str, dict] = {}

    for p in pt_files:
        d = torch.load(p, weights_only=False)
        sub = d["subject_id"]
        x_raw_ft = d["x_raw_ft"].numpy()
        y = d["labels"].numpy().astype(np.int64)
        meta = d["meta"]

        modes_out = {}
        for gm in geom_modes:
            if gm == "radial_only_z":
                pos_ori, st, keep_mask = normalize_sensor_positions(meta, mapping_mode="radial_only")
                axis_sub = [a for a, k in zip(meta.axis_labels, keep_mask) if k]
            elif gm == "tangential_only_xy":
                pos_all, st_all, _ = normalize_sensor_positions(meta, mapping_mode="default")
                keep_mask = np.array([a in ("X", "Y") for a in meta.axis_labels], dtype=bool)
                pos_ori, st = pos_all[keep_mask], st_all[keep_mask]
                axis_sub = [a for a, k in zip(meta.axis_labels, keep_mask) if k]
            else:
                pos_ori, st, keep_mask = normalize_sensor_positions(meta, mapping_mode="default")
                axis_sub = list(meta.axis_labels)

            x_sel = x_raw_ft[:, keep_mask, :].copy()

            if gm in ("triaxial_xyz_softhfc", "phase5_axis_split_xyz"):
                x_sel = apply_soft_hfc(x_sel, pos_ori[:, 3:6], alpha=0.25)

            x_norm = np.stack([sensortype_wise_normalize(x_sel[i], st) for i in range(len(x_sel))], axis=0)

            if gm == "phase5_axis_split_xyz":
                xb = torch.from_numpy(x_norm).float()
                pb = torch.from_numpy(pos_ori).float()
                sb = torch.from_numpy(st).long()
                
                # Batch evaluation of adapter reconstruction
                batch_pccs: List[float] = []
                with torch.no_grad():
                    for i in range(0, len(xb), 16):
                        x_batch = xb[i : i + 16].to(device)
                        p_batch = pb.unsqueeze(0).expand(len(x_batch), -1, -1).to(device)
                        s_batch = sb.unsqueeze(0).expand(len(x_batch), -1).to(device)
                        xt, xr = adapter_pt.reconstruct_waveform(x_batch, p_batch, s_batch, axis_labels=axis_sub)
                        batch_pccs.append(float(get_pcc(xt, xr).cpu().item()))
                pcc_val = float(np.mean(batch_pccs))

                # Batch encoding of latents
                latents_pt: List[np.ndarray] = []
                latents_rand: List[np.ndarray] = []
                with torch.no_grad():
                    for i in range(0, len(xb), 16):
                        x_batch = xb[i : i + 16].to(device)
                        p_batch = pb.unsqueeze(0).expand(len(x_batch), -1, -1).to(device)
                        s_batch = sb.unsqueeze(0).expand(len(x_batch), -1).to(device)
                        h_pt = adapter_pt.encode_latent(x_batch, p_batch, s_batch, axis_labels=axis_sub)
                        latents_pt.append(h_pt[:, :, 1:3, :].mean(dim=2).reshape(len(x_batch), -1).cpu().numpy())
                        h_rand = adapter_rand.encode_latent(x_batch, p_batch, s_batch, axis_labels=axis_sub)
                        latents_rand.append(h_rand[:, :, 1:3, :].mean(dim=2).reshape(len(x_batch), -1).cpu().numpy())
                emb_pt = np.concatenate(latents_pt, axis=0)
                emb_rand = np.concatenate(latents_rand, axis=0)
            elif gm == "random_3way_split":
                xb = torch.from_numpy(x_norm).float().to(device)
                pb = torch.from_numpy(pos_ori).float().unsqueeze(0).expand(len(xb), -1, -1).to(device)
                sb = torch.from_numpy(st).long().unsqueeze(0).expand(len(xb), -1).to(device)
                pcc_val = reconstruct_random_split(bb_pt, xb, pb, sb, n_splits=3, seed=42, batch_size=16)

                # Use monolithic embeddings for linear probe comparison
                sea_pt = precompute_seanet_features(bb_pt, {sub: x_norm}, device=device, batch_size=32)
                emb_pt = encode_from_seanet_features(bb_pt, sea_pt, {sub: pos_ori}, {sub: st}, device=device, batch_size=32)[sub]
                sea_rand = precompute_seanet_features(bb_rand, {sub: x_norm}, device=device, batch_size=32)
                emb_rand = encode_from_seanet_features(bb_rand, sea_rand, {sub: pos_ori}, {sub: st}, device=device, batch_size=32)[sub]
            else:
                pcc_val = evaluate_zero_shot_tokenizer_pcc(bb_pt, {sub: x_norm}, {sub: pos_ori}, {sub: st}, device=device)
                sea_pt = precompute_seanet_features(bb_pt, {sub: x_norm}, device=device, batch_size=32)
                emb_pt = encode_from_seanet_features(bb_pt, sea_pt, {sub: pos_ori}, {sub: st}, device=device, batch_size=32)[sub]
                sea_rand = precompute_seanet_features(bb_rand, {sub: x_norm}, device=device, batch_size=32)
                emb_rand = encode_from_seanet_features(bb_rand, sea_rand, {sub: pos_ori}, {sub: st}, device=device, batch_size=32)[sub]

            bp_X = dec.extract_features(x_norm[:, :, 64:268])
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = {"pretrained_linear_probe": [], "control_a_linear_probe": [], "classical_bandpower": []}

            for tr_idx, te_idx in skf.split(x_norm, y):
                for arm_name, feat_mat in [
                    ("pretrained_linear_probe", emb_pt),
                    ("control_a_linear_probe", emb_rand),
                    ("classical_bandpower", bp_X),
                ]:
                    m = eval_linear_probe_from_embeddings(feat_mat[tr_idx], y[tr_idx], feat_mat[te_idx], y[te_idx], seed=42)
                    fold_scores[arm_name].append(m["balanced_accuracy"])

            modes_out[gm] = {
                "n_channels": int(x_norm.shape[1]),
                "n_trials": int(x_norm.shape[0]),
                "zero_shot_tokenizer_pcc": pcc_val,
                "braille_5fold_bacc": {k: float(np.mean(v)) for k, v in fold_scores.items()},
            }

        subj_results[sub] = modes_out
        print(
            f"  [Nottingham {sub}] PCC raw={modes_out['triaxial_xyz_raw']['zero_shot_tokenizer_pcc']:.4f} "
            f"-> random_3way={modes_out['random_3way_split']['zero_shot_tokenizer_pcc']:.4f} "
            f"-> axis_split={modes_out['phase5_axis_split_xyz']['zero_shot_tokenizer_pcc']:.4f} "
            f"| BACC raw={modes_out['triaxial_xyz_raw']['braille_5fold_bacc']['pretrained_linear_probe']*100:.1f}% "
            f"-> axis={modes_out['phase5_axis_split_xyz']['braille_5fold_bacc']['pretrained_linear_probe']*100:.1f}%"
        )

    subs = sorted(subj_results.keys())
    cohort_summary = {}
    for gm in geom_modes:
        pcc_arr = np.array([subj_results[s][gm]["zero_shot_tokenizer_pcc"] for s in subs])
        pcc_ci = subject_bootstrap_ci(pcc_arr, n_resamples=10_000, rng_seed=42)
        bacc_cis = {}
        for arm_name in ("pretrained_linear_probe", "control_a_linear_probe", "classical_bandpower"):
            arr = np.array([subj_results[s][gm]["braille_5fold_bacc"][arm_name] for s in subs])
            ci = subject_bootstrap_ci(arr, n_resamples=10_000, rng_seed=42)
            bacc_cis[arm_name] = {"mean": ci.mean, "ci_low": ci.ci_low, "ci_high": ci.ci_high}
        cohort_summary[gm] = {
            "zero_shot_pcc": {"mean": pcc_ci.mean, "ci_low": pcc_ci.ci_low, "ci_high": pcc_ci.ci_high},
            "braille_5fold_bacc": bacc_cis,
        }

    raw_pcc = np.array([subj_results[s]["triaxial_xyz_raw"]["zero_shot_tokenizer_pcc"] for s in subs])
    rad_pcc = np.array([subj_results[s]["radial_only_z"]["zero_shot_tokenizer_pcc"] for s in subs])
    rnd_pcc = np.array([subj_results[s]["random_3way_split"]["zero_shot_tokenizer_pcc"] for s in subs])
    p5_pcc = np.array([subj_results[s]["phase5_axis_split_xyz"]["zero_shot_tokenizer_pcc"] for s in subs])

    rad_diff = subject_bootstrap_ci(rad_pcc - raw_pcc, n_resamples=10_000, rng_seed=42)
    p5_diff = subject_bootstrap_ci(p5_pcc - raw_pcc, n_resamples=10_000, rng_seed=42)
    rnd_diff = subject_bootstrap_ci(rnd_pcc - raw_pcc, n_resamples=10_000, rng_seed=42)
    axis_vs_rnd = subject_bootstrap_ci(p5_pcc - rnd_pcc, n_resamples=10_000, rng_seed=42)

    rad_pval = compute_paired_permutation_p_value(rad_pcc - raw_pcc)
    p5_pval = compute_paired_permutation_p_value(p5_pcc - raw_pcc)
    rnd_pval = compute_paired_permutation_p_value(rnd_pcc - raw_pcc)
    axis_vs_rnd_pval = compute_paired_permutation_p_value(p5_pcc - rnd_pcc)

    payload = {
        "dataset": "nottingham_opm_triaxial_zenodo_11126593_full84trials",
        "n_subjects": len(subs),
        "trials_per_subject": 84,
        "cohort_bootstrap_summary": cohort_summary,
        "paired_hypothesis_tests": {
            "radial_vs_raw": {"delta": rad_diff.mean, "ci_low": rad_diff.ci_low, "ci_high": rad_diff.ci_high, "p_value": rad_pval},
            "axis_split_vs_raw": {"delta": p5_diff.mean, "ci_low": p5_diff.ci_low, "ci_high": p5_diff.ci_high, "p_value": p5_pval},
            "random_split_vs_raw": {"delta": rnd_diff.mean, "ci_low": rnd_diff.ci_low, "ci_high": rnd_diff.ci_high, "p_value": rnd_pval},
            "axis_split_vs_random_split": {"delta": axis_vs_rnd.mean, "ci_low": axis_vs_rnd.ci_low, "ci_high": axis_vs_rnd.ci_high, "p_value": axis_vs_rnd_pval},
        },
        "per_subject": subj_results,
    }

    out_p = Path("results/summary_phase4_nottingham_full.json")
    out_p.write_text(json.dumps(payload, indent=2))
    print(f"\nSaved verified full Nottingham summary to {out_p}")
    print("Key Results:")
    print(f"  Raw Monolithic PCC: {cohort_summary['triaxial_xyz_raw']['zero_shot_pcc']['mean']:.4f}")
    print(f"  Random 3-way Split PCC (Capacity Control): {cohort_summary['random_3way_split']['zero_shot_pcc']['mean']:.4f} (p = {rnd_pval:.4f})")
    print(f"  Axis-Partitioned Split PCC: {cohort_summary['phase5_axis_split_xyz']['zero_shot_pcc']['mean']:.4f} (p = {p5_pval:.4f})")
    print(f"  Axis-Partitioned vs Random Split: Δ = {axis_vs_rnd.mean:+.4f} [{axis_vs_rnd.ci_low:+.4f}, {axis_vs_rnd.ci_high:+.4f}] (p = {axis_vs_rnd_pval:.4f})")


if __name__ == "__main__":
    main()
