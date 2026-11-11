"""Auditable preprocessing diagnostics. Power reduction is not a signal-quality verdict."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from scipy.signal import welch

TUTORIAL = "https://mne.tools/1.11/auto_tutorials/preprocessing/80_opm_processing.html"
PIPELINE_VERSION = "2-svd-hfc-qc"
FILTER_KWARGS = dict(method="fir", phase="zero", fir_design="firwin", fir_window="hamming")
SETTINGS = {
    "ds005107": dict(bandpass_hz=[1., 96.], notch_hz=[44., 50., 100., 150., 200.],
                     epoch_window_sec=[-.2, 1.8], sfreq_target=256.),
    "nottingham_adults": dict(bandpass_hz=[1., 45.], notch_hz=[50., 100.],
                             epoch_window_sec=[-.2, 1.8], sfreq_target=256.),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def provenance(dataset: str, sources: dict, **details) -> dict:
    return dict(pipeline_version=PIPELINE_VERSION, dataset=dataset,
                settings=SETTINGS[dataset], filter=FILTER_KWARGS,
                versions={p: importlib.metadata.version(p) for p in ("mne", "numpy", "scipy")},
                sources=sources, **details)


def spectral_summary(data, sfreq: float, ch_names, unit: str) -> dict:
    """Welch spectra at ~1 Hz resolution, per channel, in physical units.

    Epochs are analyzed independently then averaged, never concatenated.
    One channel is processed at a time to bound working memory.
    """
    x = np.asarray(data)
    if x.ndim not in (2, 3) or any(n == 0 for n in x.shape) or x.shape[-1] < 2:
        raise ValueError("Expected nonempty [channels, time] or [epochs, channels, time]")
    if not np.isfinite(sfreq) or sfreq <= 0 or not np.isfinite(x).all():
        raise ValueError("Data and positive sampling frequency must be finite")
    if unit not in ("T", "fT", "pT"):
        raise ValueError("QC requires physical units T, fT or pT before model normalization")
    names = list(ch_names)
    if len(names) != x.shape[-2] or len(set(names)) != len(names):
        raise ValueError("Unique channel names must match the data")
    psds, stds, peaks = [], [], []
    for c in range(len(names)):
        signal = x[..., c, :]
        freq, power = welch(signal, fs=sfreq, nperseg=min(round(sfreq), x.shape[-1]), axis=-1)
        if power.ndim == 2:
            power = power.mean(axis=0)
        psds.append(power.tolist())
        stds.append(float(signal.std()))
        peaks.append(float(np.ptp(signal, axis=-1).max()))
    return dict(sfreq=float(sfreq), unit=unit, ch_names=names, shape=list(x.shape),
                frequencies_hz=freq.tolist(), psd=psds, std=stds, peak_to_peak=peaks,
                flat_channels=[names[i] for i, v in enumerate(stds) if v == 0])


def compare_spectra(before: dict, after: dict) -> dict:
    for key in ("sfreq", "unit", "ch_names", "shape", "frequencies_hz"):
        if before[key] != after[key]:
            raise ValueError(f"Before/after {key} must match; compare identical samples and channels")
    pre, post = np.asarray(before["psd"]), np.asarray(after["psd"])
    valid = (pre > 0) & (post > 0)
    ratio = np.zeros_like(pre)
    np.divide(pre, post, out=ratio, where=valid)
    shield = np.zeros_like(pre)
    np.log10(ratio, out=shield, where=valid)
    # Zero-power bins are undefined, not infinite successes.
    values = [[float(10 * v) if ok else None for v, ok in zip(row, mask)]
              for row, mask in zip(shield, valid)]
    return dict(shielding_db=values, undefined_bins=int((~valid).sum()),
                interpretation="Positive means less power, not proof of neural-signal preservation")


def epoch_checks(epochs, labels, sfreq=256., tmin=-.2, expected_samples=512) -> dict:
    x = np.asarray(epochs)
    if x.ndim != 3 or min(x.shape) == 0 or x.shape[-1] != expected_samples:
        raise ValueError("Empty or incorrectly sized epochs")
    if len(labels) != len(x) or not np.isfinite(x).all():
        raise ValueError("Labels must align with finite epochs")
    times = tmin + np.arange(x.shape[-1]) / sfreq
    baseline = times < 0
    return dict(n_epochs=len(x), n_channels=x.shape[1], samples=x.shape[2],
                tmin=tmin, last_sample_sec=float(times[-1]),
                event_offset_samples=float(-tmin * sfreq),
                baseline_abs_mean_max=float(np.abs(x[..., baseline].mean(axis=-1)).max()) if baseline.any() else None,
                baseline_note="Residual after resampling is diagnostic, not an exact-zero gate",
                labels={str(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))})


def geometry_checks(orientations) -> dict:
    q = np.asarray(orientations, dtype=float)
    if q.ndim != 2 or q.shape[1] != 3 or not np.isfinite(q).all():
        raise ValueError("Finite [channels, 3] orientations required")
    norms = np.linalg.norm(q, axis=1)
    valid = norms > 1e-8
    normalized = q[valid] / norms[valid, None]
    singular = np.linalg.svd(normalized, compute_uv=False) if valid.any() else np.array([])
    tol = max(normalized.shape) * np.finfo(float).eps * singular[0] if singular.size else 0.
    return dict(rank=int((singular > tol).sum()), singular_values=singular.tolist(),
                rank_tolerance=float(tol), zero_orientation_indices=np.flatnonzero(~valid).tolist())


def write_report(path, report) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
