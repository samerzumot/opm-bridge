#!/usr/bin/env python3
"""Phase 1.5 Signal Gate CLI (§5.2).

Evaluates `CompactMEGNet` (Simple Specialist) and `ClassicalBandPowerDecoder` at the largest
available subject budget across subject-wise cross-validation folds. A task passes the signal gate
only if at least one baseline achieves a 95% subject-level bootstrap CI lower bound strictly above
chance (0.50).

Usage:
  # Synthetic smoke test:
  python scripts/run_phase15_signal_gate.py --smoke

  # Real cached OPM epochs (ds005107):
  python scripts/run_phase15_signal_gate.py --data-dir data/cache/ds005107 --tasks all --out results/phase15_ds005107_pilot.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from opm_bridge.eval.metrics import check_phase15_signal_gate, compute_classification_metrics
from opm_bridge.models.baselines import ClassicalBandPowerDecoder, CompactMEGNet
from opm_bridge.train.runner import build_subject_kfold_splits, seed_everything


def run_synthetic_smoke_gate() -> dict:
    """Run an end-to-end Phase 1.5 signal gate smoke check on synthetic multi-subject data."""
    torch.set_num_threads(1)
    seed_everything(42)
    n_subjects = 10
    n_trials_per_subj = 24
    n_channels = 32
    n_times = 512
    subjects = [f"sub-{i:02d}" for i in range(1, n_subjects + 1)]

    # Generate synthetic class-modulated beta/ERP signal across subjects
    t = np.linspace(0, 2.0, n_times, dtype=np.float32)
    X_by_subj, y_by_subj = {}, {}
    rng = np.random.default_rng(42)
    for s in subjects:
        y = np.array([0, 1] * (n_trials_per_subj // 2), dtype=np.int64)
        X = rng.normal(0.0, 1.0, size=(n_trials_per_subj, n_channels, n_times)).astype(np.float32)
        for i in range(n_trials_per_subj):
            freq = 18.0 if y[i] == 1 else 10.0
            X[i, :8, :] += 0.8 * np.sin(2 * np.pi * freq * t)
        X_by_subj[s] = X
        y_by_subj[s] = y

    splits = build_subject_kfold_splits(subjects, n_splits=5, val_count=1, seed=42)
    spec_subj_scores = []
    clas_subj_scores = []

    for split in splits:
        X_tr = np.concatenate([X_by_subj[s] for s in split.train_pool_subjects], axis=0)
        y_tr = np.concatenate([y_by_subj[s] for s in split.train_pool_subjects], axis=0)

        # 1. Classical decoder
        clas = ClassicalBandPowerDecoder(sfreq=256.0).fit(X_tr, y_tr)

        # 2. Simple specialist (CompactMEGNet)
        seed_everything(42 + split.fold_index)
        net = CompactMEGNet(n_channels=n_channels, n_classes=2)
        opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-2)
        ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True)
        net.train()
        for _ in range(8):
            for xb, yb in dl_tr:
                opt.zero_grad()
                out = net(xb, labels=yb)
                out["loss"].backward()
                opt.step()

        net.eval()
        for ts in split.test_subjects:
            X_te, y_te = X_by_subj[ts], y_by_subj[ts]
            c_pred = clas.predict(X_te)
            clas_subj_scores.append(compute_classification_metrics(y_te, c_pred)["balanced_accuracy"])

            with torch.no_grad():
                logits = net(torch.from_numpy(X_te))["logits"]
                s_pred = logits.argmax(dim=-1).numpy()
            spec_subj_scores.append(compute_classification_metrics(y_te, s_pred)["balanced_accuracy"])

    passed, summary = check_phase15_signal_gate(
        np.array(spec_subj_scores),
        np.array(clas_subj_scores),
        chance_level=0.50,
    )
    return {
        "gate_passed": passed,
        "simple_specialist": vars(summary["simple_specialist"]),
        "classical_decoder": vars(summary["classical_decoder"]),
    }


def run_real_signal_gate(
    data_dir: str | Path = "data/cache/ds005107",
    tasks: Sequence[str] = ("race", "gender", "age", "emotion"),
    epochs_per_fit: int = 15,
    batch_size: int = 32,
    lr: float = 3e-3,
    seed: int = 42,
) -> dict[str, dict]:
    """Execute Phase 1.5 Signal Gate on real cached OPM epochs (§5.2)."""
    torch.set_num_threads(1)
    base_dir = Path(data_dir)
    subj_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("sub-")])
    if len(subj_dirs) < 2:
        raise ValueError(f"Need at least 2 subjects in {data_dir}, found {len(subj_dirs)}")

    print(f"Discovered {len(subj_dirs)} subjects in {data_dir}: {[s.name for s in subj_dirs]}")

    # Load all cached runs per subject
    subj_raw_X: Dict[str, torch.Tensor] = {}
    subj_raw_labels: Dict[str, Dict[str, torch.Tensor]] = {}

    for sdir in subj_dirs:
        s_name = sdir.name
        pt_files = sorted(sdir.glob("*.pt"))
        if not pt_files:
            continue
        all_x, all_labels = [], {t: [] for t in tasks}
        for p in pt_files:
            payload = torch.load(p, weights_only=False)
            all_x.append(payload["x"].float())
            for t in tasks:
                all_labels[t].append(payload["labels"][t])

        subj_raw_X[s_name] = torch.cat(all_x, dim=0)  # (N_total, C_s, 512)
        subj_raw_labels[s_name] = {t: torch.cat(all_labels[t], dim=0) for t in tasks}
        print(f"  {s_name}: {subj_raw_X[s_name].shape[0]} trials, {subj_raw_X[s_name].shape[1]} channels")

    valid_subjects = sorted(subj_raw_X.keys())
    # Harmonize channel dimension across subjects by taking min channels
    min_ch = min(subj_raw_X[s].shape[1] for s in valid_subjects)
    print(f"Harmonizing channel count to min across subjects: {min_ch} channels")

    X_by_subj: Dict[str, np.ndarray] = {
        s: subj_raw_X[s][:, :min_ch, :].numpy().astype(np.float32) for s in valid_subjects
    }
    labels_by_subj: Dict[str, Dict[str, np.ndarray]] = {
        t: {s: subj_raw_labels[s][t].numpy().astype(np.int64) for s in valid_subjects}
        for t in tasks
    }

    # Subject-wise K-fold (Leave-One-Subject-Out when val_count=0)
    splits = build_subject_kfold_splits(
        valid_subjects,
        n_splits=len(valid_subjects),
        val_count=0,
        seed=seed,
    )

    all_task_results = {}

    for task in tasks:
        print(f"\n==================================================")
        print(f"Executing Phase 1.5 Signal Gate on Task: {task.upper()}")
        print(f"==================================================")

        clas_subj_scores: List[float] = []
        spec_subj_scores: List[float] = []
        tested_subjects: List[str] = []

        for split in splits:
            assert len(split.test_subjects) == 1, "Expected leave-one-subject-out"
            test_sub = split.test_subjects[0]
            tested_subjects.append(test_sub)

            # Assemble train pool from remaining subjects
            X_tr = np.concatenate([X_by_subj[s] for s in split.train_pool_subjects], axis=0)
            y_tr = np.concatenate([labels_by_subj[task][s] for s in split.train_pool_subjects], axis=0)

            X_te = X_by_subj[test_sub]
            y_te = labels_by_subj[task][test_sub]

            # 1. Classical Decoder (Log Band Power + Welch + Logistic Regression)
            clas = ClassicalBandPowerDecoder(sfreq=256.0, random_state=seed + split.fold_index).fit(X_tr, y_tr)
            c_pred = clas.predict(X_te)
            c_bacc = compute_classification_metrics(y_te, c_pred)["balanced_accuracy"]
            clas_subj_scores.append(float(c_bacc))

            # 2. Simple Specialist (CompactMEGNet temporal-spatial depthwise CNN)
            seed_everything(seed + split.fold_index * 100)
            net = CompactMEGNet(n_channels=min_ch, n_classes=2, dropout=0.25)
            optimizer = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=1e-2)
            ds_tr = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
            dl_tr = DataLoader(ds_tr, batch_size=batch_size, shuffle=True)

            net.train()
            for ep in range(epochs_per_fit):
                for xb, yb in dl_tr:
                    optimizer.zero_grad()
                    out = net(xb, labels=yb)
                    out["loss"].backward()
                    optimizer.step()

            net.eval()
            with torch.no_grad():
                logits = net(torch.from_numpy(X_te))["logits"]
                s_pred = logits.argmax(dim=-1).numpy()
            s_bacc = compute_classification_metrics(y_te, s_pred)["balanced_accuracy"]
            spec_subj_scores.append(float(s_bacc))

            print(
                f"  Fold {split.fold_index} (Test: {test_sub:6s}) | "
                f"Classical BACC: {c_bacc * 100:.2f}% | "
                f"CompactMEGNet BACC: {s_bacc * 100:.2f}%"
            )

        passed, summary = check_phase15_signal_gate(
            np.array(spec_subj_scores),
            np.array(clas_subj_scores),
            chance_level=0.50,
            n_resamples=10_000,
            rng_seed=seed,
        )

        spec_ci = summary["simple_specialist"]
        clas_ci = summary["classical_decoder"]

        print(f"\n--- Signal Gate Summary for {task.upper()} ---")
        print(f"  Simple Specialist: Mean={spec_ci.mean * 100:.2f}%, 95% CI=[{spec_ci.ci_low * 100:.2f}%, {spec_ci.ci_high * 100:.2f}%]")
        print(f"  Classical Decoder: Mean={clas_ci.mean * 100:.2f}%, 95% CI=[{clas_ci.ci_low * 100:.2f}%, {clas_ci.ci_high * 100:.2f}%]")
        print(f"  Gate Status:       {'PASSED (Proceed to Phase 2)' if passed else 'FAILED / INCONCLUSIVE (Halt/Refine)'}")

        all_task_results[task] = {
            "gate_passed": bool(passed),
            "subjects": tested_subjects,
            "simple_specialist": {
                "scores": spec_subj_scores,
                "mean": spec_ci.mean,
                "ci_low": spec_ci.ci_low,
                "ci_high": spec_ci.ci_high,
            },
            "classical_decoder": {
                "scores": clas_subj_scores,
                "mean": clas_ci.mean,
                "ci_low": clas_ci.ci_low,
                "ci_high": clas_ci.ci_high,
            },
        }

    return all_task_results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1.5 Signal Gate (§5.2)")
    parser.add_argument("--smoke", action="store_true", help="Run synthetic multi-subject smoke gate")
    parser.add_argument("--data-dir", type=str, default="data/cache/ds005107", help="Path to cached subject data")
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=["race", "gender", "age", "emotion"],
        help="List of task dimensions to evaluate (default: all 4)",
    )
    parser.add_argument("--epochs", type=int, default=15, help="Training epochs for CompactMEGNet")
    parser.add_argument("--out", type=str, default="results/phase15_signal_gate.json")
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        res = run_synthetic_smoke_gate()
        out_path.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))
        return

    res = run_real_signal_gate(
        data_dir=args.data_dir,
        tasks=args.tasks,
        epochs_per_fit=args.epochs,
    )
    out_path.write_text(json.dumps(res, indent=2))
    print(f"\nWrote full Phase 1.5 Signal Gate report to {out_path}")


if __name__ == "__main__":
    main()
