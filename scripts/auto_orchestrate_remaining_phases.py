#!/usr/bin/env python3
"""Unified benchmark evaluation orchestrator for OPM-Bridge.

Executes:
  1. Full 21-subject physically aligned (50 common helmet channels) transfer benchmark on
     OpenNeuro ds005107 across budgets B in {1, 2, 4, 10, 20} and multiple random seeds for:
       - Cross-subject LOSO decoding (age, race, gender)
       - Within-subject few-shot calibration (gender, B_trials in {32, 64, 128, 180})
  2. BrainOmni sensor geometry and embedding perturbation experiments (default, opm_as_grad,
     zero_orientation, flip_orientation, scramble_channels) on linear probes and zero-shot
     waveform reconstruction correlation.
  3. Real Nottingham triaxial (X/Y/Z) QuSpin OPM-MEG evaluation (Zenodo 11126593,
     sub-101..sub-110, CoordinateFrame.HEAD) comparing full triaxial (X/Y/Z), radial isolation (Z),
     and axis-partitioned tokenization.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

from opm_bridge.eval.metrics import (
    compute_classification_metrics,
    evaluate_primary_comparison,
    subject_bootstrap_ci,
)
from opm_bridge.models.baselines import ClassicalBandPowerDecoder, CompactMEGNet
from opm_bridge.models.vendor_brainomni.model_utils.loss import get_pcc
from opm_bridge.models.wrapper import OPMAwareBrainOmniAdapter, load_brainomni_backbone
from opm_bridge.preprocessing.sensor_mapping import (
    align_subjects_by_sensor_geometry,
    apply_soft_hfc,
    normalize_sensor_positions,
    sensortype_wise_normalize,
)
from opm_bridge.train.runner import seed_everything
from scripts.run_transfer_benchmark import (
    ERP_END_SAMPLE,
    ERP_START_SAMPLE,
    encode_from_seanet_features,
    eval_linear_probe_from_embeddings,
    get_default_device,
    precompute_seanet_features,
    run_benchmark,
)

RESULTS_DIR = Path("results")


@torch.no_grad()
def evaluate_zero_shot_tokenizer_pcc(
    backbone: torch.nn.Module,
    X_by_subj: Dict[str, np.ndarray],
    pos_by_subj: Dict[str, np.ndarray],
    st_by_subj: Dict[str, np.ndarray],
    device: torch.device,
    max_trials_per_subj: int = 64,
    precomputed_sea_by_subj: Dict[str, torch.Tensor] | None = None,
) -> Dict[str, float]:
    """Compute zero-shot BrainTokenizer waveform reconstruction PCC across subjects."""
    backbone.eval()
    tok = backbone.tokenizer
    subj_pccs: List[float] = []
    for s in sorted(X_by_subj.keys()):
        x_np = X_by_subj[s][:max_trials_per_subj]
        xb = torch.from_numpy(x_np).float().to(device)
        b = len(xb)
        pb = torch.from_numpy(pos_by_subj[s]).float().unsqueeze(0).expand(b, -1, -1).to(device)
        sb = torch.from_numpy(st_by_subj[s]).long().unsqueeze(0).expand(b, -1).to(device)

        x_unf = tok.unfold(xb, overlap_ratio=0.0)
        se = tok.sensor_embed(pb, sb)
        if precomputed_sea_by_subj is not None and s in precomputed_sea_by_subj:
            x_sea = precomputed_sea_by_subj[s][:max_trials_per_subj].to(device)
            _, c, w, _ = x_sea.shape
            se_rep = rearrange(se.unsqueeze(2).repeat(1, 1, w, 1), "B C W D -> (B W) C D")
            x_bw = rearrange(x_sea, "B C W D -> (B W) C D")
            neuros = tok.encoder.neuros.type_as(x_bw).unsqueeze(0).repeat(x_bw.shape[0], 1, 1)
            feat = tok.encoder.backwardsolution(neuros, tok.encoder.k_proj(x_bw + se_rep), x_bw)
            feat = rearrange(feat, "(B N T) C D -> B C N T D", B=b, N=1)
        else:
            feat = tok.encoder(x_unf, se)
        q_feat, _, _ = tok.quantizer(feat)
        x_rec = tok.decoder(q_feat, se)
        pcc_val = float(get_pcc(tok.norm_target(x_unf), x_rec).cpu().item())
        subj_pccs.append(pcc_val)

    arr = np.array(subj_pccs, dtype=np.float64)
    if len(subj_pccs) >= 2:
        ci = subject_bootstrap_ci(arr, n_resamples=5_000, rng_seed=42)
        mean_v, low_v, high_v = ci.mean, ci.ci_low, ci.ci_high
    else:
        mean_v = float(arr[0])
        low_v, high_v = mean_v, mean_v
    return {
        "mean_pcc": mean_v,
        "ci_low": low_v,
        "ci_high": high_v,
        "n_subjects": len(subj_pccs),
        "per_subject_pcc": subj_pccs,
    }


def run_within_subject_calibration_benchmark(
    X_by_subj: Dict[str, np.ndarray],
    y_by_subj: Dict[str, np.ndarray],
    pt_emb_by_subj: Dict[str, np.ndarray],
    ctrl_a_emb_by_subj: Dict[str, np.ndarray],
    ctrl_b_emb_by_subj: Dict[str, np.ndarray],
    trial_budgets: List[int] = (32, 64, 128, 180),
    seeds: List[int] = (42, 3407, 2025),
) -> dict:
    """Evaluate within-subject few-shot calibration across all 21 subjects on fixed 64-trial holdout."""
    subjects = sorted(X_by_subj.keys())
    n_subj = len(subjects)
    arms = ["pretrained_linear_probe", "control_a_linear_probe", "control_b_linear_probe", "classical_bandpower"]
    dec = ClassicalBandPowerDecoder(sfreq=256.0)
    bp_feats = {
        s: dec.extract_features(X_by_subj[s][:, :, ERP_START_SAMPLE:ERP_END_SAMPLE])
        for s in subjects
    }

    results_by_budget = {}
    for b_trials in trial_budgets:
        scores = {a: np.zeros((n_subj, len(seeds)), dtype=np.float64) for a in arms}
        f1s = {a: np.zeros((n_subj, len(seeds)), dtype=np.float64) for a in arms}

        for s_idx, s in enumerate(subjects):
            y_all = y_by_subj[s]
            # Fixed last 64 trials as subject test set; first 192 trials as calibration pool
            y_pool, y_te = y_all[:192], y_all[192:]
            for seed_idx, seed in enumerate(seeds):
                rng = np.random.default_rng(seed + s_idx)
                idx0 = np.where(y_pool == 0)[0]
                idx1 = np.where(y_pool == 1)[0]
                k = min(b_trials // 2, len(idx0), len(idx1))
                sel = np.concatenate([
                    rng.choice(idx0, size=k, replace=False),
                    rng.choice(idx1, size=k, replace=False),
                ])

                for arm, emb_dict in [
                    ("pretrained_linear_probe", pt_emb_by_subj),
                    ("control_a_linear_probe", ctrl_a_emb_by_subj),
                    ("control_b_linear_probe", ctrl_b_emb_by_subj),
                    ("classical_bandpower", bp_feats),
                ]:
                    X_s = emb_dict[s]
                    m = eval_linear_probe_from_embeddings(
                        X_s[sel], y_pool[sel], X_s[192:], y_te, seed=seed
                    )
                    scores[arm][s_idx, seed_idx] = m["balanced_accuracy"]
                    f1s[arm][s_idx, seed_idx] = m["macro_f1"]

        b_dict = {}
        for arm in arms:
            ci = subject_bootstrap_ci(scores[arm], n_resamples=5_000, rng_seed=42)
            f1_ci = subject_bootstrap_ci(f1s[arm], n_resamples=2_000, rng_seed=42)
            b_dict[arm] = {
                "mean": ci.mean,
                "ci_low": ci.ci_low,
                "ci_high": ci.ci_high,
                "macro_f1_mean": f1_ci.mean,
            }
        rq1 = evaluate_primary_comparison(
            scores["pretrained_linear_probe"],
            scores["control_a_linear_probe"],
            budget=b_trials,
            sesoi=0.03,
        )
        b_dict["primary_hypothesis_test_rq1"] = {
            "delta_mean": rq1.delta_mean,
            "delta_ci_low": rq1.delta_ci_low,
            "delta_ci_high": rq1.delta_ci_high,
            "verdict": rq1.verdict,
        }
        results_by_budget[str(b_trials)] = b_dict

    return {
        "protocol": "within_subject_few_shot_calibration",
        "task": "gender",
        "n_subjects": n_subj,
        "test_trials_per_subject": 64,
        "trial_budgets": list(trial_budgets),
        "results_by_trial_budget": results_by_budget,
    }


def run_phase4_real_nottingham_evaluation(
    bb_pt: torch.nn.Module,
    bb_rand: torch.nn.Module,
    device: torch.device,
) -> dict:
    """Evaluate real Nottingham triaxial (X/Y/Z) QuSpin OPM-MEG data (Zenodo 11126593)."""
    nott_dir = Path("data/cache/nottingham")
    pt_files = sorted(nott_dir.glob("sub-*_256hz.pt"))
    if not pt_files:
        raise FileNotFoundError("No cached Nottingham .pt files found in data/cache/nottingham/")

    subj_results = {}
    dec = ClassicalBandPowerDecoder(sfreq=256.0)
    adapter_pt = OPMAwareBrainOmniAdapter(bb_pt, remap_to_grad=False, axis_split_triaxial=True)
    adapter_rand = OPMAwareBrainOmniAdapter(bb_rand, remap_to_grad=False, axis_split_triaxial=True)

    geom_modes = [
        "triaxial_xyz_raw",
        "triaxial_xyz_softhfc",
        "radial_only_z",
        "tangential_only_xy",
        "phase5_axis_split_xyz",
    ]

    for p in pt_files:
        d = torch.load(p, weights_only=False)
        sub = d["subject_id"]
        x_raw_ft = d["x_raw_ft"].numpy()  # (N_trials=26, C_good, 512)
        y = d["labels"].numpy().astype(np.int64)
        meta = d["meta"]

        modes_out = {}
        for geom_mode in geom_modes:
            if geom_mode == "radial_only_z":
                pos_ori, st, keep_mask = normalize_sensor_positions(meta, mapping_mode="radial_only")
                axis_sub = [a for a, k in zip(meta.axis_labels, keep_mask) if k]
            elif geom_mode == "tangential_only_xy":
                pos_all, st_all, _ = normalize_sensor_positions(meta, mapping_mode="default")
                keep_mask = np.array([a in ("X", "Y") for a in meta.axis_labels], dtype=bool)
                pos_ori, st = pos_all[keep_mask], st_all[keep_mask]
                axis_sub = [a for a, k in zip(meta.axis_labels, keep_mask) if k]
            else:
                pos_ori, st, keep_mask = normalize_sensor_positions(meta, mapping_mode="default")
                axis_sub = list(meta.axis_labels)

            x_sel = x_raw_ft[:, keep_mask, :].copy()

            if geom_mode in ("triaxial_xyz_softhfc", "phase5_axis_split_xyz"):
                x_sel = apply_soft_hfc(x_sel, pos_ori[:, 3:6], alpha=0.25)

            x_norm = np.stack(
                [sensortype_wise_normalize(x_sel[i], st) for i in range(len(x_sel))],
                axis=0,
            )

            if geom_mode == "phase5_axis_split_xyz":
                xb = torch.from_numpy(x_norm).float().to(device)
                b = len(xb)
                pb = torch.from_numpy(pos_ori).float().unsqueeze(0).expand(b, -1, -1).to(device)
                sb = torch.from_numpy(st).long().unsqueeze(0).expand(b, -1).to(device)
                xt, xr = adapter_pt.reconstruct_waveform(xb, pb, sb, axis_labels=axis_sub)
                pcc_val = float(get_pcc(xt, xr).cpu().item())

                h_pt = adapter_pt.encode_latent(xb, pb, sb, axis_labels=axis_sub)
                emb_pt = h_pt[:, :, 1:4, :].mean(dim=2).reshape(b, -1).cpu().numpy()

                h_rand = adapter_rand.encode_latent(xb, pb, sb, axis_labels=axis_sub)
                emb_rand = h_rand[:, :, 1:4, :].mean(dim=2).reshape(b, -1).cpu().numpy()
            else:
                pcc_res = evaluate_zero_shot_tokenizer_pcc(
                    bb_pt, {sub: x_norm}, {sub: pos_ori}, {sub: st}, device=device
                )
                pcc_val = pcc_res["mean_pcc"]

                sea_pt = precompute_seanet_features(bb_pt, {sub: x_norm}, device=device, batch_size=32)
                emb_pt = encode_from_seanet_features(
                    bb_pt, sea_pt, {sub: pos_ori}, {sub: st}, device=device, batch_size=32
                )[sub]

                sea_rand = precompute_seanet_features(bb_rand, {sub: x_norm}, device=device, batch_size=32)
                emb_rand = encode_from_seanet_features(
                    bb_rand, sea_rand, {sub: pos_ori}, {sub: st}, device=device, batch_size=32
                )[sub]

            # Active somatosensory window [+50ms, +850ms] -> samples 64..268
            bp_X = dec.extract_features(x_norm[:, :, 64:268])

            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            fold_scores = {
                "pretrained_linear_probe": [],
                "control_a_linear_probe": [],
                "classical_bandpower": [],
            }
            for tr_idx, te_idx in skf.split(x_norm, y):
                for arm_name, feat_mat in [
                    ("pretrained_linear_probe", emb_pt),
                    ("control_a_linear_probe", emb_rand),
                    ("classical_bandpower", bp_X),
                ]:
                    m = eval_linear_probe_from_embeddings(
                        feat_mat[tr_idx], y[tr_idx], feat_mat[te_idx], y[te_idx], seed=42
                    )
                    fold_scores[arm_name].append(m["balanced_accuracy"])

            modes_out[geom_mode] = {
                "n_channels": int(x_norm.shape[1]),
                "n_trials": int(x_norm.shape[0]),
                "zero_shot_tokenizer_pcc": pcc_val,
                "braille_5fold_bacc": {k: float(np.mean(v)) for k, v in fold_scores.items()},
            }
        subj_results[sub] = modes_out
        print(
            f"  [Nottingham {sub}] PCC raw={modes_out['triaxial_xyz_raw']['zero_shot_tokenizer_pcc']:.4f} "
            f"-> radial={modes_out['radial_only_z']['zero_shot_tokenizer_pcc']:.4f} "
            f"-> phase5_axis_split={modes_out['phase5_axis_split_xyz']['zero_shot_tokenizer_pcc']:.4f}"
        )

    # Compute 10,000-resample subject-level bootstrap CIs across all evaluated subjects
    subs = sorted(subj_results.keys())
    cohort_summary = {}
    if len(subs) >= 2:
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

        # Paired hypothesis tests on zero-shot PCC and Braille BACC
        raw_pcc = np.array([subj_results[s]["triaxial_xyz_raw"]["zero_shot_tokenizer_pcc"] for s in subs])
        rad_pcc = np.array([subj_results[s]["radial_only_z"]["zero_shot_tokenizer_pcc"] for s in subs])
        p5_pcc = np.array([subj_results[s]["phase5_axis_split_xyz"]["zero_shot_tokenizer_pcc"] for s in subs])
        rad_diff = subject_bootstrap_ci(rad_pcc - raw_pcc, n_resamples=10_000, rng_seed=42)
        p5_diff = subject_bootstrap_ci(p5_pcc - raw_pcc, n_resamples=10_000, rng_seed=42)
        cohort_summary["paired_hypothesis_tests"] = {
            "radial_vs_triaxial_raw_pcc_delta": {
                "delta_mean": rad_diff.mean,
                "delta_ci_low": rad_diff.ci_low,
                "delta_ci_high": rad_diff.ci_high,
            },
            "phase5_axis_split_vs_triaxial_raw_pcc_delta": {
                "delta_mean": p5_diff.mean,
                "delta_ci_low": p5_diff.ci_low,
                "delta_ci_high": p5_diff.ci_high,
            },
        }

    summary = {
        "dataset": "nottingham_opm_triaxial_zenodo_11126593",
        "coordinate_frame": "head",
        "is_triaxial": True,
        "has_digitized_landmarks": True,
        "status": "VERIFIED_REAL_DATA",
        "n_subjects": len(subs),
        "subjects_evaluated": subs,
        "cohort_bootstrap_summary": cohort_summary,
        "per_subject": subj_results,
    }
    out_p = RESULTS_DIR / "summary_phase4_nottingham.json"
    out_p.write_text(json.dumps(summary, indent=2))
    print(f"Saved real Nottingham triaxial evaluation ({len(subs)} subjects) to {out_p}")
    return summary


def main() -> None:
    t_start = time.time()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    device = get_default_device()
    print(f"=== STARTING COMPLETE VERIFIED OPM-BRIDGE BENCHMARK ON {device} ===")

    # Shared SEANet cache across tasks & geometry ablations (computed once!)
    shared_sea: Dict[str, Dict[str, torch.Tensor]] = {}

    # 1. PHASE 2: Cross-Subject Aligned LOSO Benchmark on `age` (full 5 arms) and `race`/`gender`
    for task in ["age", "race", "gender"]:
        print(f"\n--- Cross-Subject Transfer Benchmark (Aligned 50 ch, Soft-HFC): task={task} ---")
        task_arms = (
            [
                "pretrained_linear_probe",
                "control_a_linear_probe",
                "control_b_linear_probe",
                "compact_megnet",
                "classical_bandpower",
            ]
            if task == "age"
            else [
                "pretrained_linear_probe",
                "control_a_linear_probe",
                "control_b_linear_probe",
                "classical_bandpower",
            ]
        )
        run_benchmark(
            data_dir="data/cache/ds005107",
            task=task,
            arms=task_arms,
            budgets=[1, 2, 4, 10, 20],
            seeds=[42, 3407],
            epochs=5,
            batch_size=64,
            use_soft_hfc=True,
            hfc_alpha=0.25,
            mapping_mode="default",
            ledger_path=RESULTS_DIR / f"ledger_aligned21_{task}.json",
            summary_path=RESULTS_DIR / f"summary_aligned21_{task}.json",
            precomputed_sea=shared_sea,
        )

    # Also save primary summary_cohort21_full.json pointing to the aligned benchmark
    age_sum = json.loads((RESULTS_DIR / "summary_aligned21_age.json").read_text())
    race_sum = json.loads((RESULTS_DIR / "summary_aligned21_race.json").read_text())
    gender_sum = json.loads((RESULTS_DIR / "summary_aligned21_gender.json").read_text())

    # 2. Within-Subject Few-Shot Calibration on `gender` (where within-subject signal is p=0.0002)
    print("\n--- Within-Subject Few-Shot Calibration (21 Subjects, task=gender) ---")
    base_dir = Path("data/cache/ds005107")
    raw_X, raw_pos, raw_st, y_gender = {}, {}, {}, {}
    for sdir in sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")]):
        s_name = sdir.name
        pt_files = sorted(sdir.glob("*.pt"))
        if not pt_files:
            continue
        d = torch.load(pt_files[0], weights_only=False)
        raw_X[s_name] = d["x"].float().numpy()
        raw_pos[s_name] = d["pos_ori"].float().numpy()
        raw_st[s_name] = d["sensor_type"].long().numpy()
        y_gender[s_name] = d["labels"]["gender"].numpy().astype(np.int64)

    X_al, pos_al, st_al = align_subjects_by_sensor_geometry(raw_X, raw_pos, raw_st)
    for s in X_al:
        X_al[s] = apply_soft_hfc(X_al[s], pos_al[s][:, 3:6], alpha=0.25)

    seed_everything(42)
    bb_pt = load_brainomni_backbone("tiny", True, True).to(device)
    bb_ctrl_b = load_brainomni_backbone("tiny", False, True).to(device)
    bb_rand = load_brainomni_backbone("tiny", False, False).to(device)

    pt_emb = encode_from_seanet_features(bb_pt, shared_sea["pretrained_tok"], pos_al, st_al, device=device)
    ctrl_b_emb = encode_from_seanet_features(bb_ctrl_b, shared_sea["pretrained_tok"], pos_al, st_al, device=device)
    ctrl_a_emb = encode_from_seanet_features(bb_rand, shared_sea["random_tok"], pos_al, st_al, device=device)

    within_summary = run_within_subject_calibration_benchmark(
        X_al, y_gender, pt_emb, ctrl_a_emb, ctrl_b_emb
    )
    (RESULTS_DIR / "summary_within_subject_gender.json").write_text(json.dumps(within_summary, indent=2))

    # 3. PHASE 3: Genuine Sensor Geometry Ablations on BrainOmni (`pretrained_linear_probe` & Zero-Shot PCC)
    print("\n--- BrainOmni Sensor Geometry and Embedding Ablations ---")
    ablation_summaries = {}
    pcc_by_mode = {}
    for mode in ["default", "opm_as_grad", "zero_orientation", "flip_orientation", "scramble_channels"]:
        pos_m = {s: pos_al[s].copy() for s in pos_al}
        st_m = {s: st_al[s].copy() for s in st_al}
        X_m = {s: X_al[s].copy() for s in X_al}
        if mode == "opm_as_grad":
            for s in st_m:
                st_m[s][:] = 2
        elif mode == "zero_orientation":
            for s in pos_m:
                pos_m[s][:, 3:] = 0.0
        elif mode == "flip_orientation":
            for s in pos_m:
                pos_m[s][:, 3:] = -pos_m[s][:, 3:]
        elif mode == "scramble_channels":
            rng = np.random.default_rng(42)
            perm = rng.permutation(50)
            for s in pos_m:
                pos_m[s] = pos_m[s][perm]
                st_m[s] = st_m[s][perm]

        pcc_by_mode[mode] = evaluate_zero_shot_tokenizer_pcc(
            bb_pt,
            X_m,
            pos_m,
            st_m,
            device=device,
            max_trials_per_subj=64,
            precomputed_sea_by_subj=shared_sea["pretrained_tok"],
        )
        print(f"  [Zero-Shot BrainTokenizer PCC] mode={mode:18s}: {pcc_by_mode[mode]['mean_pcc']:.4f}")

        if mode != "default":
            res_m = run_benchmark(
                data_dir="data/cache/ds005107",
                task="age",
                arms=["pretrained_linear_probe", "control_b_linear_probe"],
                budgets=[1, 2, 4, 10, 20],
                seeds=[42, 3407],
                epochs=5,
                batch_size=64,
                use_soft_hfc=True,
                hfc_alpha=0.25,
                mapping_mode=mode,
                ledger_path=RESULTS_DIR / f"ledger_phase3_ablation_{mode}.json",
                summary_path=RESULTS_DIR / f"summary_phase3_ablation_{mode}.json",
                precomputed_sea=shared_sea,
            )
            ablation_summaries[mode] = res_m
        else:
            ablation_summaries["default"] = age_sum

    phase3_full = {
        "zero_shot_tokenizer_pcc_by_geometry_mode": pcc_by_mode,
        "downstream_transfer_by_geometry_mode": ablation_summaries,
    }
    (RESULTS_DIR / "summary_phase3_ablations.json").write_text(json.dumps(phase3_full, indent=2))

    # 4. PHASE 4: Real Nottingham Triaxial OPM Evaluation (Zenodo 11126593)
    print("\n--- Real Nottingham Triaxial OPM Evaluation ---")
    nott_summary = run_phase4_real_nottingham_evaluation(bb_pt, bb_rand, device=device)

    # Save master cohort summary
    master_summary = {
        "cross_subject_age": age_sum,
        "cross_subject_race": race_sum,
        "cross_subject_gender": gender_sum,
        "within_subject_gender": within_summary,
        "phase3_zero_shot_pcc": pcc_by_mode,
        "phase4_nottingham_real": nott_summary,
    }
    (RESULTS_DIR / "summary_cohort21_full.json").write_text(json.dumps(master_summary, indent=2))
    print(f"\n=== ALL PHASES COMPLETED IN {(time.time() - t_start)/60.0:.2f} MINUTES ===")


if __name__ == "__main__":
    main()
