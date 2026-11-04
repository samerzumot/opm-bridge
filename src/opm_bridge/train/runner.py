"""Subject-wise K-fold splitting, nested data-budget sampling, and resumable trainer (§8.2, §9)."""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset

from opm_bridge.data.schema import HarmonizedSample
from opm_bridge.eval.metrics import compute_classification_metrics


def seed_everything(seed: int) -> None:
    """Set deterministic seeds across Python, NumPy, and PyTorch (§8.2)."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class SubjectFoldSplit:
    """Strict subject-wise split with nested training budgets (§9)."""

    fold_index: int
    train_pool_subjects: List[str]
    val_subjects: List[str]
    test_subjects: List[str]


def build_subject_kfold_splits(
    subjects: Sequence[str],
    n_splits: int = 5,
    val_count: int = 3,
    seed: int = 42,
) -> List[SubjectFoldSplit]:
    """Build subject-wise K-fold splits where no subject ever spans train, val, and test (§9)."""
    unique_subjs = np.array(sorted(set(subjects)))
    if len(unique_subjs) < n_splits:
        raise ValueError(f"Need >= {n_splits} subjects, got {len(unique_subjs)}")

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits: List[SubjectFoldSplit] = []

    for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(unique_subjs)):
        test_subjs = unique_subjs[test_idx].tolist()
        train_val_subjs = unique_subjs[train_val_idx].tolist()

        rng = np.random.default_rng(seed + fold_idx * 100)
        perm = rng.permutation(len(train_val_subjs))
        n_val = min(val_count, max(1, len(train_val_subjs) // 5))
        val_subjs = [train_val_subjs[i] for i in perm[:n_val]]
        train_pool = [train_val_subjs[i] for i in perm[n_val:]]

        assert set(train_pool).isdisjoint(val_subjs)
        assert set(train_pool).isdisjoint(test_subjs)
        assert set(val_subjs).isdisjoint(test_subjs)

        splits.append(
            SubjectFoldSplit(
                fold_index=fold_idx,
                train_pool_subjects=train_pool,
                val_subjects=val_subjs,
                test_subjects=test_subjs,
            )
        )
    return splits


def select_nested_budget_subjects(
    train_pool_subjects: Sequence[str],
    budget_n_subjects: Optional[int],
    seed: int = 42,
) -> List[str]:
    """Select nested subset of training subjects so B=1 c B=2 c B=5 c B=10 c All (§9)."""
    rng = np.random.default_rng(seed)
    ordered = [train_pool_subjects[i] for i in rng.permutation(len(train_pool_subjects))]
    if budget_n_subjects is None or budget_n_subjects >= len(ordered):
        return ordered
    return ordered[:budget_n_subjects]


class HarmonizedTensorDataset(Dataset):
    """PyTorch Dataset wrapping a list of HarmonizedSample objects."""

    def __init__(self, samples: Sequence[HarmonizedSample]) -> None:
        self.samples = list(samples)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        s = self.samples[idx]
        return {
            "x": s.x,
            "pos": s.pos_ori,
            "sensor_type": s.sensor_type,
            "label": torch.tensor(s.label, dtype=torch.long),
            "subject_id": s.subject_id,
        }


class ResumableBenchmarkLedger:
    """JSON-backed ledger so interrupted 12h Kaggle/Colab sessions resume seamlessly (§8.2)."""

    def __init__(self, ledger_path: str | Path) -> None:
        self.ledger_path = Path(ledger_path)
        self.ledger_path.parent.mkdir(parents=True, exist_ok=True)
        self.records: Dict[str, dict] = {}
        if self.ledger_path.exists():
            self.records = json.loads(self.ledger_path.read_text())

    @staticmethod
    def make_key(
        dataset: str,
        arm: str,
        fold: int,
        budget: int | str,
        seed: int,
    ) -> str:
        return f"{dataset}__{arm}__fold{fold}__budget{budget}__seed{seed}"

    def is_done(self, key: str) -> bool:
        return key in self.records

    def record(self, key: str, payload: dict) -> None:
        self.records[key] = payload
        tmp_path = self.ledger_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(self.records, indent=2))
        tmp_path.replace(self.ledger_path)

