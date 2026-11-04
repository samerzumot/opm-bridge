"""Simple specialist (EEGNet-style compact CNN) and classical baselines (§5.2, §6, FR5)."""

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import welch
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


class CompactMEGNet(nn.Module):
    """EEGNet-style depthwise-separable temporal-spatial CNN specialist for OPM/MEG epochs.

    Used in:
      - Phase 1.5 Signal Gate (§5.2): must beat chance (95% subject-level bootstrap CI)
        at the largest budget before any transfer experiment runs on a task.
      - RQ2 Baseline (§6): guards against simple models trained from scratch matching
        or beating foundation models at small OPM budgets.
    """

    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        f1: int = 16,
        d: int = 2,
        f2: int = 32,
        kernel_length: int = 64,
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.temporal_conv = nn.Sequential(
            nn.Conv2d(1, f1, kernel_size=(1, kernel_length), padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(f1),
        )
        self.spatial_depthwise = nn.Sequential(
            nn.Conv2d(f1, f1 * d, kernel_size=(n_channels, 1), groups=f1, bias=False),
            nn.BatchNorm2d(f1 * d),
            nn.ELU(),
            nn.AvgPool2d(kernel_size=(1, 4)),
            nn.Dropout(dropout),
        )
        self.separable_conv = nn.Sequential(
            nn.Conv2d(f1 * d, f1 * d, kernel_size=(1, 16), padding=(0, 8), groups=f1 * d, bias=False),
            nn.Conv2d(f1 * d, f2, kernel_size=(1, 1), bias=False),
            nn.BatchNorm2d(f2),
            nn.ELU(),
            nn.AdaptiveAvgPool2d((1, 8)),
            nn.Dropout(dropout),
        )
        self.head = nn.Linear(f2 * 8, n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(
        self,
        x: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        # x: (B, C, T) -> (B, 1, C, T)
        h = x.unsqueeze(1)
        h = self.temporal_conv(h)
        h = self.spatial_depthwise(h)
        h = self.separable_conv(h)
        feat = h.flatten(start_dim=1)
        logits = self.head(feat)
        out = {"logits": logits, "embeddings": feat}
        if labels is not None:
            out["loss"] = self.criterion(logits, labels.long())
        return out


class ClassicalBandPowerDecoder:
    """CPU-fast classical decoder using log band-power + temporal summary features + L2 Logistic Regression."""

    BANDS = (
        (1.0, 4.0),    # delta
        (4.0, 8.0),    # theta
        (8.0, 13.0),   # alpha
        (13.0, 30.0),  # beta
        (30.0, 60.0),  # low-gamma
        (60.0, 96.0),  # high-gamma
    )

    def __init__(self, sfreq: float = 256.0, c_reg: float = 1.0, random_state: int = 42) -> None:
        self.sfreq = sfreq
        self.c_reg = c_reg
        self.random_state = random_state
        self.pipe = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        C=c_reg,
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=random_state,
                    ),
                ),
            ]
        )

    def extract_features(self, X: np.ndarray) -> np.ndarray:
        """Extract per-channel log band-powers + decimated temporal means from X of shape (N, C, T)."""
        n_samples, n_ch, n_times = X.shape
        nperseg = min(256, n_times)
        freqs, psd = welch(X, fs=self.sfreq, nperseg=nperseg, axis=-1)
        band_feats = []
        for f_lo, f_hi in self.BANDS:
            idx = (freqs >= f_lo) & (freqs < f_hi)
            if np.any(idx):
                bp = np.log10(np.mean(psd[:, :, idx], axis=-1) + 1e-30)
            else:
                bp = np.zeros((n_samples, n_ch), dtype=np.float32)
            band_feats.append(bp)

        # Add 4 temporal window averages for ERP/ERF phase sensitivity
        chunks = np.array_split(X, 4, axis=-1)
        temp_feats = [np.mean(c, axis=-1) for c in chunks]

        all_feats = np.concatenate(band_feats + temp_feats, axis=-1)
        return all_feats.astype(np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ClassicalBandPowerDecoder":
        feats = self.extract_features(X)
        self.pipe.fit(feats, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        feats = self.extract_features(X)
        return self.pipe.predict(feats)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        feats = self.extract_features(X)
        return self.pipe.predict_proba(feats)

