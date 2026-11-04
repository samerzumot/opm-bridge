#!/usr/bin/env python3
"""Streaming cache builder for OpenNeuro ds005107.

Streams one `.fif` + `beh/*_events.tsv` run at a time directly from OpenNeuro's S3 archive,
applies harmonized notch + bandpass filtering and 256 Hz decimation, extracts latency-corrected
(+33 ms) epochs and 2^4 factorial face labels, and saves a compact `.pt` cache.
"""

from __future__ import annotations

import argparse
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import List

import mne
import numpy as np
import torch

from opm_bridge.data.ds005107 import (
    DS005107_SUBJECTS,
    align_ds005107_events,
    stimulus_id_to_face_labels,
)
from opm_bridge.preprocessing.sensor_mapping import (
    apply_soft_hfc,
    extract_sensor_metadata_from_mne,
    normalize_sensor_positions,
    sensortype_wise_normalize,
)

S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds005107"


def _download_bytes(url: str, timeout: int = 180, max_retries: int = 4, backoff: float = 2.0) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "opm-bridge/0.1"})
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                chunks: List[bytes] = []
                while True:
                    chunk = r.read(2 * 1024 * 1024)  # 2 MB chunks
                    if not chunk:
                        break
                    chunks.append(chunk)
                return b"".join(chunks)
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            print(f"Download warning on {url} (attempt {attempt + 1}/{max_retries}): {e}. Retrying in {backoff:.1f}s...")
            time.sleep(backoff)
            backoff *= 2
    raise RuntimeError(f"Failed to download {url} after {max_retries} attempts.")



def stream_and_cache_ds005107_run(
    subject: str,
    session: str = "ses-01",
    run: str = "run-01",
    out_dir: str | Path = "data/cache/ds005107",
    reject_outlier_channels: bool = True,
    use_soft_hfc: bool = False,
    hfc_alpha: float = 0.25,
) -> Path:
    """Stream a single ds005107 run from S3, preprocess to 256 Hz epochs, and save `.pt` cache."""
    tag = "_softhfc" if use_soft_hfc else ""
    out_path = Path(out_dir) / subject / f"{subject}_{session}_{run}{tag}_256hz.pt"
    if out_path.exists():
        return out_path

    fif_url = f"{S3_BASE}/{subject}/{session}/meg/{subject}_{session}_task-face_{run}_meg.fif"
    beh_url = f"{S3_BASE}/{subject}/{session}/beh/{subject}_{session}_task-face_{run}_events.tsv"

    fif_bytes = _download_bytes(fif_url)
    beh_bytes = _download_bytes(beh_url)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_fif = Path(tmpdir) / f"{subject}_{session}_task-face_{run}_meg.fif"
        tmp_beh = Path(tmpdir) / f"{subject}_{session}_task-face_{run}_events.tsv"
        tmp_fif.write_bytes(fif_bytes)
        tmp_beh.write_bytes(beh_bytes)

        raw = mne.io.read_raw_fif(str(tmp_fif), preload=True, verbose=False)
        events = mne.find_events(raw, stim_channel="STIM", shortest_event=1, verbose=False)
        valid_events = align_ds005107_events(
            events,
            tmp_beh,
            sfreq=float(raw.info["sfreq"]),
            proj_latency_sec=0.033,
        )

        # Pick MEG channels only
        meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, ref_meg=False)
        raw.pick(meg_picks)

        # Notch filter (44 Hz OPM artifact + 50 Hz line harmonics) and bandpass 1.0-96.0 Hz
        raw.notch_filter(freqs=[44.0, 50.0, 100.0, 150.0, 200.0], verbose=False)
        raw.filter(l_freq=1.0, h_freq=96.0, verbose=False)

        # Epoch around latency-corrected stimulus onsets (-0.2s to +1.8s = 2.0s window)
        epochs = mne.Epochs(
            raw,
            valid_events,
            tmin=-0.2,
            tmax=1.8,
            baseline=(-0.2, 0.0),
            detrend=1,
            preload=True,
            verbose=False,
        )
        epochs.resample(256.0, verbose=False)
        # Trim or pad to exact 512 samples (2.0s at 256 Hz)
        ep_data = epochs.get_data(copy=True)[:, :, :512]  # (N_trials, C, 512)

        meta = extract_sensor_metadata_from_mne(epochs.info, dataset_name="ds005107")
        pos_ori, sensor_type, _ = normalize_sensor_positions(meta, mapping_mode="default")

        if reject_outlier_channels:
            ch_stds = ep_data.std(axis=(0, 2))
            med = float(np.median(ch_stds))
            good_ch = (ch_stds < 3.5 * med) & (ch_stds > 0.1 * med)
            if int(good_ch.sum()) >= 16:
                ep_data = ep_data[:, good_ch, :]
                pos_ori = pos_ori[good_ch]
                sensor_type = sensor_type[good_ch]

        if use_soft_hfc:
            ep_data = apply_soft_hfc(ep_data, pos_ori[:, 3:6], alpha=hfc_alpha)

        # Normalize each epoch using BrainOmni sensor-type-wise group normalization
        norm_epochs = np.stack(
            [sensortype_wise_normalize(ep_data[i], sensor_type) for i in range(len(ep_data))],
            axis=0,
        )

        stim_ids = epochs.events[:, 2].tolist()
        labels = {dim: [] for dim in ("race", "gender", "age", "emotion")}
        for sid in stim_ids:
            fl = stimulus_id_to_face_labels(int(sid))
            for dim, val in fl.items():
                labels[dim].append(val)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "x": torch.from_numpy(norm_epochs),  # (N_trials, C_good, 512)
            "pos_ori": torch.from_numpy(pos_ori),  # (C_good, 6)
            "sensor_type": torch.from_numpy(sensor_type).long(),  # (C_good,)
            "stim_ids": torch.tensor(stim_ids, dtype=torch.long),
            "labels": {k: torch.tensor(v, dtype=torch.long) for k, v in labels.items()},
            "subject_id": subject,
            "session": session,
            "run": run,
            "frame": meta.frame.value,
        }
        torch.save(payload, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream and cache ds005107 runs at 256 Hz")
    parser.add_argument("--subjects", nargs="+", default=list(DS005107_SUBJECTS))
    parser.add_argument("--runs", nargs="+", default=["run-01"])
    parser.add_argument("--out-dir", type=str, default="data/cache/ds005107")
    parser.add_argument("--soft-hfc", action="store_true", help="Apply Soft-HFC spatial filtering")
    parser.add_argument("--hfc-alpha", type=float, default=0.25, help="Soft-HFC shrinkage factor (default: 0.25)")
    args = parser.parse_args()

    saved: List[str] = []
    for subj in args.subjects:
        for run in args.runs:
            p = stream_and_cache_ds005107_run(
                subj,
                session="ses-01",
                run=run,
                out_dir=args.out_dir,
                use_soft_hfc=args.soft_hfc,
                hfc_alpha=args.hfc_alpha,
            )
            saved.append(str(p))
            print(f"Cached {subj} {run} -> {p}")


if __name__ == "__main__":
    main()

