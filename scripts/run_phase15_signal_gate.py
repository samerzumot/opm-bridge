#!/usr/bin/env python3
"""Phase 1.5 Signal Gate CLI (§5.2).

Evaluates `CompactMEGNet` (Simple Specialist) and `ClassicalBandPowerDecoder` at the largest
subject budget across subject-wise folds. A task passes the signal gate only if at least one
baseline achieves a 95% subject-level bootstrap CI lower bound strictly above chance (0.50).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1.5 Signal Gate (§5.2)")
    parser.add_argument("--smoke", action="store_true", help="Run synthetic multi-subject smoke gate")
    parser.add_argument("--out", type=str, default="results/phase15_signal_gate.json")
    args = parser.parse_args()

    if args.smoke:
        res = run_synthetic_smoke_gate()
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
