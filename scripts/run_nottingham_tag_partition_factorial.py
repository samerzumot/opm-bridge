#!/usr/bin/env python3
"""Run Nottingham Triaxial Tag x Partition Factorial and Multi-Seed Controls.

This script executes:
1. Exact per-trial PCC logging alongside trial-weighted and batch-weighted aggregations.
2. The full factorial grid across:
   - Tag: MAG (type=1) vs GRAD (type=2)
   - Filter: Raw vs Soft-HFC (alpha=0.25)
   - Partition: Monolithic, Random 3-way split, Axis-partitioned (Z, X, Y),
                Radial only (Z), Tangential only (X, Y)
3. 10 Random partition seeds with group sizes strictly matched to per-subject (|Z|, |X|, |Y|) counts.
4. Downstream 5-fold cross-validated linear probe decoding (Pretrained, Control A, Classical).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold

from opm_bridge.data.schema import CoordinateFrame, SensorMetadata
from opm_bridge.eval.metrics import (
    compute_paired_permutation_p_value,
    eval_linear_probe_from_embeddings,
    subject_bootstrap_ci,
)
from opm_bridge.models.baselines import ClassicalBandPowerDecoder
from opm_bridge.models.wrapper import load_brainomni_backbone
from opm_bridge.preprocessing.sensor_mapping import (
    apply_soft_hfc,
    normalize_sensor_positions,
    sensortype_wise_normalize,
)

torch.serialization.add_safe_globals([SensorMetadata, CoordinateFrame])


def compute_per_trial_pcc(rec: torch.Tensor, raw: torch.Tensor) -> np.ndarray:
    """Compute Pearson correlation coefficient per trial.
    
    Args:
        rec: Reconstructed tensor of shape (B, C, W, D)
        raw: Normalized target tensor of shape (B, C, W, D)
        
    Returns:
        1D numpy array of length B with mean PCC across channels and windows for each trial.
    """
    B, C, W, D = rec.shape
    x = rec.float() - rec.float().mean(dim=-1, keepdim=True)
    y = raw.float() - raw.float().mean(dim=-1, keepdim=True)
    cov = (x * y).sum(dim=-1) * (1.0 / (D - 1))
    std_x = torch.std(rec.float(), dim=-1)
    std_y = torch.std(raw.float(), dim=-1)
    pcc_bcw = cov / (std_x * std_y + 1e-6)
    # Average across channels and temporal windows per trial: (B,)
    trial_pcc = pcc_bcw.mean(dim=(1, 2)).detach().cpu().numpy()
    return trial_pcc


def reconstruct_and_encode_partition(
    backbone: torch.nn.Module,
    x_sub: torch.Tensor,  # (B, C_grp, L)
    pos_sub: torch.Tensor,  # (B, C_grp, 6)
    st_sub: torch.Tensor,  # (B, C_grp)
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pass a channel subset through BrainTokenizer and return (reconstructed_sub, latents_sub)."""
    tok = backbone.tokenizer
    x_unf = tok.unfold(x_sub, overlap_ratio=0.0)
    se = tok.sensor_embed(pos_sub, st_sub)
    feat = tok.encoder(x_unf, se)
    q_feat, _, _ = tok.quantizer(feat)
    x_rec = tok.decoder(q_feat, se).float()

    # Latent encoding for linear probe
    feat_tok, _ = tok.tokenize(x_sub, pos_sub, st_sub, backbone.overlap_ratio)
    c_lat = feat_tok.shape[1]
    neuro = tok.encoder.neuros.type_as(feat_tok).detach().view(1, c_lat, 1, -1)
    h = backbone.projection(feat_tok + neuro)
    for block in backbone.blocks[:-1]:
        h = block(h)
    h = F.normalize(h, p=2.0, dim=-1, eps=1e-6)
    return x_rec, h


def run_benchmark():
    device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    print(f"Device: {device} | Loading BrainOmni backbones...")

    bb_pt = load_brainomni_backbone("tiny", True, True).to(device)
    bb_rand = load_brainomni_backbone("tiny", False, False).to(device)
    bb_pt.eval()
    bb_rand.eval()
    dec = ClassicalBandPowerDecoder(sfreq=256.0)

    pt_files = sorted(Path("data/cache/nottingham").glob("sub-1*_256hz.pt"))
    assert len(pt_files) == 10, f"Expected 10 adult subjects, found {len(pt_files)}"

    SEEDS = [42, 101, 102, 103, 104, 105, 106, 107, 108, 109]
    BATCH_SIZE = 16

    # Results container
    # Structure: [subject][filter][tag][mode] -> metrics
    all_subject_data: Dict[str, dict] = {}

    for p in pt_files:
        d = torch.load(p, weights_only=False)
        sub = d["subject_id"]
        x_raw_ft = d["x_raw_ft"].numpy()  # (84, C, 512)
        y = d["labels"].numpy().astype(np.int64)
        meta: SensorMetadata = d["meta"]
        n_trials = len(y)

        print(f"\n--- Processing Subject: {sub} ({n_trials} trials, {meta.pos_3d.shape[0]} channels) ---")
        sub_results = {}

        # Determine exact axis group indices
        axis_groups = {}
        for ax in ("Z", "X", "Y"):
            axis_groups[ax] = [i for i, a in enumerate(meta.axis_labels) if a == ax]
        group_sizes = [len(axis_groups["Z"]), len(axis_groups["X"]), len(axis_groups["Y"])]
        print(f"  Axis channel counts: Z={group_sizes[0]}, X={group_sizes[1]}, Y={group_sizes[2]} (Total: {sum(group_sizes)})")

        for filter_name in ("raw", "softhfc"):
            sub_results[filter_name] = {}
            # Apply filter if softhfc
            pos_ori_base, st_base, _ = normalize_sensor_positions(meta, mapping_mode="default")
            if filter_name == "softhfc":
                x_filtered = apply_soft_hfc(x_raw_ft, pos_ori_base[:, 3:6], alpha=0.25)
            else:
                x_filtered = x_raw_ft.copy()

            for tag_name in ("mag", "grad"):
                sub_results[filter_name][tag_name] = {}
                
                # Setup sensor type
                st_current = st_base.copy()
                if tag_name == "grad":
                    st_current[:] = 2  # Set all to GRAD=2
                else:
                    st_current[:] = 1  # Set all to MAG=1

                # Normalize per sensor type
                x_norm = np.stack([sensortype_wise_normalize(x_filtered[i], st_current) for i in range(n_trials)], axis=0)

                # Precompute Classical Bandpower features
                bp_X = dec.extract_features(x_norm[:, :, 64:268])
                skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

                # Define arms to run for this (filter, tag) pair:
                # 1. monolithic
                # 2. axis_partitioned
                # 3. radial_only_z
                # 4. tangential_only_xy
                # 5. random_split_seeds (10 seeds)
                modes = ["monolithic", "axis_partitioned", "radial_only_z", "tangential_only_xy"]
                
                for mode in modes:
                    if mode == "radial_only_z":
                        sel_idx = axis_groups["Z"]
                    elif mode == "tangential_only_xy":
                        sel_idx = axis_groups["X"] + axis_groups["Y"]
                    else:
                        sel_idx = list(range(x_norm.shape[1]))

                    x_mode = x_norm[:, sel_idx, :]
                    pos_mode = pos_ori_base[sel_idx]
                    st_mode = st_current[sel_idx]
                    
                    # Compute PCC & latents in batches
                    all_trial_pccs: List[float] = []
                    all_batch_pccs: List[float] = []
                    latents_pt_list: List[np.ndarray] = []
                    latents_rand_list: List[np.ndarray] = []

                    with torch.no_grad():
                        for b_start in range(0, n_trials, BATCH_SIZE):
                            xb = torch.from_numpy(x_mode[b_start : b_start + BATCH_SIZE]).float().to(device)
                            b_curr = len(xb)
                            pb = torch.from_numpy(pos_mode).float().unsqueeze(0).expand(b_curr, -1, -1).to(device)
                            sb = torch.from_numpy(st_mode).long().unsqueeze(0).expand(b_curr, -1).to(device)

                            if mode == "axis_partitioned":
                                # Process each axis group independently
                                x_unf_full = bb_pt.tokenizer.unfold(xb, overlap_ratio=0.0)
                                x_tgt_full = bb_pt.tokenizer.norm_target(x_unf_full)
                                x_rec_full = torch.zeros_like(x_tgt_full)

                                h_pt_groups = []
                                h_rand_groups = []

                                for ax in ("Z", "X", "Y"):
                                    ax_idx = axis_groups[ax]
                                    idx_t = torch.tensor(ax_idx, dtype=torch.long, device=device)
                                    x_grp = xb.index_select(dim=1, index=idx_t)
                                    p_grp = pb.index_select(dim=1, index=idx_t)
                                    s_grp = sb.index_select(dim=1, index=idx_t)

                                    rec_sub, h_pt_sub = reconstruct_and_encode_partition(bb_pt, x_grp, p_grp, s_grp, device)
                                    x_rec_full.index_copy_(dim=1, index=idx_t, source=rec_sub)
                                    h_pt_groups.append(h_pt_sub)

                                    _, h_rand_sub = reconstruct_and_encode_partition(bb_rand, x_grp, p_grp, s_grp, device)
                                    h_rand_groups.append(h_rand_sub)

                                trial_pccs = compute_per_trial_pcc(x_rec_full, x_tgt_full)
                                all_trial_pccs.extend(trial_pccs.tolist())
                                all_batch_pccs.append(float(np.mean(trial_pccs)))

                                h_pt_cat = torch.cat(h_pt_groups, dim=1)
                                latents_pt_list.append(h_pt_cat[:, :, 1:3, :].mean(dim=2).reshape(b_curr, -1).cpu().numpy())

                                h_rand_cat = torch.cat(h_rand_groups, dim=1)
                                latents_rand_list.append(h_rand_cat[:, :, 1:3, :].mean(dim=2).reshape(b_curr, -1).cpu().numpy())

                            else:
                                # Monolithic / Radial / Tangential
                                x_unf_full = bb_pt.tokenizer.unfold(xb, overlap_ratio=0.0)
                                x_tgt_full = bb_pt.tokenizer.norm_target(x_unf_full)
                                se_pt = bb_pt.tokenizer.sensor_embed(pb, sb)
                                feat_pt = bb_pt.tokenizer.encoder(x_unf_full, se_pt)
                                q_pt, _, _ = bb_pt.tokenizer.quantizer(feat_pt)
                                x_rec_full = bb_pt.tokenizer.decoder(q_pt, se_pt).float()

                                trial_pccs = compute_per_trial_pcc(x_rec_full, x_tgt_full)
                                all_trial_pccs.extend(trial_pccs.tolist())
                                all_batch_pccs.append(float(np.mean(trial_pccs)))

                                # Latent encoding
                                f_tok, _ = bb_pt.tokenizer.tokenize(xb, pb, sb, bb_pt.overlap_ratio)
                                c_lat = f_tok.shape[1]
                                neuro = bb_pt.tokenizer.encoder.neuros.type_as(f_tok).detach().view(1, c_lat, 1, -1)
                                h_pt = bb_pt.projection(f_tok + neuro)
                                for blk in bb_pt.blocks[:-1]:
                                    h_pt = blk(h_pt)
                                h_pt = F.normalize(h_pt, p=2.0, dim=-1, eps=1e-6)
                                latents_pt_list.append(h_pt[:, :, 1:3, :].mean(dim=2).reshape(b_curr, -1).cpu().numpy())

                                f_tok_r, _ = bb_rand.tokenizer.tokenize(xb, pb, sb, bb_rand.overlap_ratio)
                                c_lat_r = f_tok_r.shape[1]
                                neuro_r = bb_rand.tokenizer.encoder.neuros.type_as(f_tok_r).detach().view(1, c_lat_r, 1, -1)
                                h_rand = bb_rand.projection(f_tok_r + neuro_r)
                                for blk in bb_rand.blocks[:-1]:
                                    h_rand = blk(h_rand)
                                h_rand = F.normalize(h_rand, p=2.0, dim=-1, eps=1e-6)
                                latents_rand_list.append(h_rand[:, :, 1:3, :].mean(dim=2).reshape(b_curr, -1).cpu().numpy())

                    emb_pt = np.concatenate(latents_pt_list, axis=0)
                    emb_rand = np.concatenate(latents_rand_list, axis=0)

                    # 5-fold cross-validation for Braille decoding
                    fold_scores = {"pretrained_linear_probe": [], "control_a_linear_probe": [], "classical_bandpower": []}
                    for tr_idx, te_idx in skf.split(x_mode, y):
                        for arm_name, feat_mat in [
                            ("pretrained_linear_probe", emb_pt),
                            ("control_a_linear_probe", emb_rand),
                            ("classical_bandpower", bp_X),
                        ]:
                            m = eval_linear_probe_from_embeddings(feat_mat[tr_idx], y[tr_idx], feat_mat[te_idx], y[te_idx], seed=42)
                            fold_scores[arm_name].append(m["balanced_accuracy"])

                    sub_results[filter_name][tag_name][mode] = {
                        "n_channels": int(x_mode.shape[1]),
                        "n_trials": n_trials,
                        "trial_pccs": all_trial_pccs,
                        "pcc_trial_weighted": float(np.mean(all_trial_pccs)),
                        "pcc_batch_weighted": float(np.mean(all_batch_pccs)),
                        "braille_5fold_bacc": {k: float(np.mean(v)) for k, v in fold_scores.items()},
                    }

                # Now evaluate the 10 random partition seeds (size-matched to group_sizes [|Z|, |X|, |Y|])
                sub_results[filter_name][tag_name]["random_partitions"] = {}
                n_total_ch = x_norm.shape[1]
                pos_full = pos_ori_base
                st_full = st_current

                for seed in SEEDS:
                    rng = np.random.default_rng(seed)
                    perm = rng.permutation(n_total_ch)
                    # Split exactly by group_sizes:
                    s0, s1, s2 = group_sizes[0], group_sizes[1], group_sizes[2]
                    rand_groups = [
                        perm[:s0].tolist(),
                        perm[s0 : s0 + s1].tolist(),
                        perm[s0 + s1 :].tolist(),
                    ]

                    all_trial_pccs = []
                    all_batch_pccs = []
                    latents_pt_list = []
                    latents_rand_list = []

                    with torch.no_grad():
                        for b_start in range(0, n_trials, BATCH_SIZE):
                            xb = torch.from_numpy(x_norm[b_start : b_start + BATCH_SIZE]).float().to(device)
                            b_curr = len(xb)
                            pb = torch.from_numpy(pos_full).float().unsqueeze(0).expand(b_curr, -1, -1).to(device)
                            sb = torch.from_numpy(st_full).long().unsqueeze(0).expand(b_curr, -1).to(device)

                            x_unf_full = bb_pt.tokenizer.unfold(xb, overlap_ratio=0.0)
                            x_tgt_full = bb_pt.tokenizer.norm_target(x_unf_full)
                            x_rec_full = torch.zeros_like(x_tgt_full)

                            h_pt_groups = []
                            h_rand_groups = []

                            for grp_idx in rand_groups:
                                idx_t = torch.tensor(grp_idx, dtype=torch.long, device=device)
                                x_grp = xb.index_select(dim=1, index=idx_t)
                                p_grp = pb.index_select(dim=1, index=idx_t)
                                s_grp = sb.index_select(dim=1, index=idx_t)

                                rec_sub, h_pt_sub = reconstruct_and_encode_partition(bb_pt, x_grp, p_grp, s_grp, device)
                                x_rec_full.index_copy_(dim=1, index=idx_t, source=rec_sub)
                                h_pt_groups.append(h_pt_sub)

                                _, h_rand_sub = reconstruct_and_encode_partition(bb_rand, x_grp, p_grp, s_grp, device)
                                h_rand_groups.append(h_rand_sub)

                            trial_pccs = compute_per_trial_pcc(x_rec_full, x_tgt_full)
                            all_trial_pccs.extend(trial_pccs.tolist())
                            all_batch_pccs.append(float(np.mean(trial_pccs)))

                            h_pt_cat = torch.cat(h_pt_groups, dim=1)
                            latents_pt_list.append(h_pt_cat[:, :, 1:3, :].mean(dim=2).reshape(b_curr, -1).cpu().numpy())

                            h_rand_cat = torch.cat(h_rand_groups, dim=1)
                            latents_rand_list.append(h_rand_cat[:, :, 1:3, :].mean(dim=2).reshape(b_curr, -1).cpu().numpy())

                    emb_pt = np.concatenate(latents_pt_list, axis=0)
                    emb_rand = np.concatenate(latents_rand_list, axis=0)

                    # Only compute linear probe on seed 42 to match existing setup, or on all
                    fold_scores = {"pretrained_linear_probe": [], "control_a_linear_probe": [], "classical_bandpower": []}
                    if seed == 42:
                        for tr_idx, te_idx in skf.split(x_norm, y):
                            for arm_name, feat_mat in [
                                ("pretrained_linear_probe", emb_pt),
                                ("control_a_linear_probe", emb_rand),
                                ("classical_bandpower", bp_X),
                            ]:
                                m = eval_linear_probe_from_embeddings(feat_mat[tr_idx], y[tr_idx], feat_mat[te_idx], y[te_idx], seed=42)
                                fold_scores[arm_name].append(m["balanced_accuracy"])

                    sub_results[filter_name][tag_name]["random_partitions"][str(seed)] = {
                        "trial_pccs": all_trial_pccs,
                        "pcc_trial_weighted": float(np.mean(all_trial_pccs)),
                        "pcc_batch_weighted": float(np.mean(all_batch_pccs)),
                        "braille_5fold_bacc": {k: float(np.mean(v)) for k, v in fold_scores.items()} if seed == 42 else {},
                    }

                mono_p = sub_results[filter_name][tag_name]["monolithic"]["pcc_trial_weighted"]
                axis_p = sub_results[filter_name][tag_name]["axis_partitioned"]["pcc_trial_weighted"]
                rnd_p = sub_results[filter_name][tag_name]["random_partitions"]["42"]["pcc_trial_weighted"]
                print(f"  [{filter_name.upper()} | {tag_name.upper()}] Mono: {mono_p:.4f} -> Rand(s42): {rnd_p:.4f} -> Axis: {axis_p:.4f} | Diff(Axis-Rand): {axis_p - rnd_p:+.4f}")

        all_subject_data[sub] = sub_results

    # Cohort Analysis and Bootstrap CIs
    subjects = sorted(all_subject_data.keys())
    cohort_summary = {}

    for filter_name in ("raw", "softhfc"):
        cohort_summary[filter_name] = {}
        for tag_name in ("mag", "grad"):
            cohort_summary[filter_name][tag_name] = {}
            for mode in ("monolithic", "axis_partitioned", "radial_only_z", "tangential_only_xy"):
                pcc_arr = np.array([all_subject_data[s][filter_name][tag_name][mode]["pcc_trial_weighted"] for s in subjects])
                ci = subject_bootstrap_ci(pcc_arr, n_resamples=10_000, rng_seed=42)
                cohort_summary[filter_name][tag_name][mode] = {
                    "mean": ci.mean,
                    "ci_low": ci.ci_low,
                    "ci_high": ci.ci_high,
                }

            # Random partitions cohort summary across seeds
            rnd_summary = {}
            for seed in SEEDS:
                pcc_arr = np.array([all_subject_data[s][filter_name][tag_name]["random_partitions"][str(seed)]["pcc_trial_weighted"] for s in subjects])
                ci = subject_bootstrap_ci(pcc_arr, n_resamples=10_000, rng_seed=42)
                rnd_summary[str(seed)] = {
                    "mean": ci.mean,
                    "ci_low": ci.ci_low,
                    "ci_high": ci.ci_high,
                }
            cohort_summary[filter_name][tag_name]["random_partitions"] = rnd_summary

    # Hypothesis tests and interactions
    hypothesis_tests = {}
    for filter_name in ("raw", "softhfc"):
        hypothesis_tests[filter_name] = {}
        for tag_name in ("mag", "grad"):
            axis_arr = np.array([all_subject_data[s][filter_name][tag_name]["axis_partitioned"]["pcc_trial_weighted"] for s in subjects])
            mono_arr = np.array([all_subject_data[s][filter_name][tag_name]["monolithic"]["pcc_trial_weighted"] for s in subjects])
            rnd42_arr = np.array([all_subject_data[s][filter_name][tag_name]["random_partitions"]["42"]["pcc_trial_weighted"] for s in subjects])

            # Axis vs Random (Seed 42)
            diff_axis_rnd = axis_arr - rnd42_arr
            ci_axis_rnd = subject_bootstrap_ci(diff_axis_rnd, n_resamples=10_000, rng_seed=42)
            p_axis_rnd = compute_paired_permutation_p_value(diff_axis_rnd)

            # Random (Seed 42) vs Mono
            diff_rnd_mono = rnd42_arr - mono_arr
            ci_rnd_mono = subject_bootstrap_ci(diff_rnd_mono, n_resamples=10_000, rng_seed=42)
            p_rnd_mono = compute_paired_permutation_p_value(diff_rnd_mono)

            # Axis vs Mono
            diff_axis_mono = axis_arr - mono_arr
            ci_axis_mono = subject_bootstrap_ci(diff_axis_mono, n_resamples=10_000, rng_seed=42)
            p_axis_mono = compute_paired_permutation_p_value(diff_axis_mono)

            # Across all 10 seeds: distribution of (Axis - Random)
            seed_diffs_mean = []
            for seed in SEEDS:
                r_arr = np.array([all_subject_data[s][filter_name][tag_name]["random_partitions"][str(seed)]["pcc_trial_weighted"] for s in subjects])
                seed_diffs_mean.append(float(np.mean(axis_arr - r_arr)))

            hypothesis_tests[filter_name][tag_name] = {
                "axis_vs_random_s42": {
                    "delta": ci_axis_rnd.mean,
                    "ci_low": ci_axis_rnd.ci_low,
                    "ci_high": ci_axis_rnd.ci_high,
                    "p_value": p_axis_rnd,
                },
                "random_vs_mono": {
                    "delta": ci_rnd_mono.mean,
                    "ci_low": ci_rnd_mono.ci_low,
                    "ci_high": ci_rnd_mono.ci_high,
                    "p_value": p_rnd_mono,
                },
                "axis_vs_mono": {
                    "delta": ci_axis_mono.mean,
                    "ci_low": ci_axis_mono.ci_low,
                    "ci_high": ci_axis_mono.ci_high,
                    "p_value": p_axis_mono,
                },
                "axis_minus_random_10seeds_distribution": {
                    "mean_across_seeds": float(np.mean(seed_diffs_mean)),
                    "std_across_seeds": float(np.std(seed_diffs_mean)),
                    "min_across_seeds": float(np.min(seed_diffs_mean)),
                    "max_across_seeds": float(np.max(seed_diffs_mean)),
                    "all_seed_means": seed_diffs_mean,
                },
            }

    # Tag Effect Tests: GRAD vs MAG for Monolithic, Random, and Axis
    tag_effects = {}
    for filter_name in ("raw", "softhfc"):
        tag_effects[filter_name] = {}
        for mode in ("monolithic", "axis_partitioned"):
            grad_arr = np.array([all_subject_data[s][filter_name]["grad"][mode]["pcc_trial_weighted"] for s in subjects])
            mag_arr = np.array([all_subject_data[s][filter_name]["mag"][mode]["pcc_trial_weighted"] for s in subjects])
            diff = grad_arr - mag_arr
            ci = subject_bootstrap_ci(diff, n_resamples=10_000, rng_seed=42)
            pval = compute_paired_permutation_p_value(diff)
            tag_effects[filter_name][mode] = {
                "delta_grad_minus_mag": ci.mean,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "p_value": pval,
            }

    output_payload = {
        "dataset": "nottingham_opm_triaxial_zenodo_11126593_full84trials",
        "description": "Tag (MAG vs GRAD) x Filter (Raw vs Soft-HFC) x Partition (Mono, Random 10 seeds, Axis) Factorial",
        "n_subjects": len(subjects),
        "trials_per_subject": 84,
        "cohort_summary": cohort_summary,
        "hypothesis_tests": hypothesis_tests,
        "tag_effects": tag_effects,
        "per_subject": all_subject_data,
    }

    out_file = Path("results/summary_phase4_nottingham_factorial_and_seeds.json")
    out_file.write_text(json.dumps(output_payload, indent=2))
    print(f"\n=======================================================")
    print(f"Saved complete factorial and seed audit to {out_file}")
    print(f"=======================================================")


if __name__ == "__main__":
    run_benchmark()

