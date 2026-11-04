#!/usr/bin/env python3
"""Streaming cache builder for OpenNeuro ds000117 (SQUID Elekta Neuromag-306).

Downloads one run of ds000117 from OpenNeuro S3, applies standard filtering (1-96 Hz, 50 Hz notch),
decimates to 256 Hz, extracts stimulus epochs (face vs. scrambled, plus prestimulus baseline vs. poststimulus evoked),
extracts Elekta SQUID 3D coordinates and sensor types (102 magnetometers + 204 planar gradiometers),
and saves a compact .pt cache for in-domain foundation model validation.
"""

from __future__ import annotations

import argparse
import io
import os
import shutil
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd
import torch

from opm_bridge.preprocessing.sensor_mapping import (
    extract_sensor_metadata_from_mne,
    normalize_sensor_positions,
    sensortype_wise_normalize,
)

S3_BASE = "https://s3.amazonaws.com/openneuro.org/ds000117"


def download_file_with_progress(url: str, dest_path: Path, chunk_size: int = 4 * 1024 * 1024) -> None:
    """Download a large file with streaming chunks and progress display."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": "opm-bridge/0.1"})
    print(f"Connecting to {url} ...")
    with urllib.request.urlopen(req, timeout=300) as response:
        total_size = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        t0 = time.time()
        with open(dest_path, "wb") as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                elapsed = time.time() - t0
                speed = (downloaded / (1024 * 1024)) / max(elapsed, 0.1)
                pct = (downloaded / total_size * 100) if total_size > 0 else 0
                print(
                    f"\rDownloaded {downloaded / (1024 * 1024):.1f}/{total_size / (1024 * 1024):.1f} MB "
                    f"({pct:.1f}%) at {speed:.1f} MB/s",
                    end="",
                    flush=True,
                )
    print(f"\nFinished download in {time.time() - t0:.1f}s -> {dest_path}")


def stream_and_cache_ds000117_run(
    subject: str = "sub-01",
    session: str = "ses-meg",
    run: str = "run-01",
    out_dir: str | Path = "data/cache/ds000117",
    keep_temp_raw: bool = False,
) -> Path:
    """Stream a single ds000117 run from S3, preprocess to 256 Hz epochs, and save compact .pt cache."""
    out_dir = Path(out_dir)
    out_path = out_dir / subject / f"{subject}_{session}_{run}_256hz.pt"
    if out_path.exists():
        print(f"Cache already exists at {out_path}")
        return out_path

    fif_url = f"{S3_BASE}/{subject}/{session}/meg/{subject}_{session}_task-facerecognition_{run}_meg.fif"
    events_url = f"{S3_BASE}/{subject}/{session}/meg/{subject}_{session}_task-facerecognition_{run}_events.tsv"

    tmp_dir = Path("data/cache/tmp_squid")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_fif = tmp_dir / f"{subject}_{session}_{run}_meg.fif"
    tmp_events = tmp_dir / f"{subject}_{session}_{run}_events.tsv"

    try:
        # Download events TSV
        print(f"Fetching events from {events_url}...")
        with urllib.request.urlopen(events_url) as r:
            tmp_events.write_bytes(r.read())

        # Download raw FIF
        if not tmp_fif.exists() or tmp_fif.stat().st_size < 100 * 1024 * 1024:
            download_file_with_progress(fif_url, tmp_fif)

        # Read raw MNE FIF (preloading MEG channels only to minimize RAM)
        print("Reading raw FIF with MNE...")
        raw = mne.io.read_raw_fif(str(tmp_fif), preload=True, verbose=False)

        # Pick MEG channels (306 channels: 102 MAG + 204 GRAD)
        meg_picks = mne.pick_types(raw.info, meg=True, eeg=False, stim=False, ref_meg=False)
        raw.pick(meg_picks)

        # Filtering: 50 Hz power line notch + 1-96 Hz bandpass
        print("Filtering (50 Hz notch + 1.0-96.0 Hz bandpass)...")
        raw.notch_filter(freqs=[50.0, 100.0, 150.0, 200.0], verbose=False)
        raw.filter(l_freq=1.0, h_freq=96.0, verbose=False)

        # Parse events
        events_df = pd.read_csv(tmp_events, sep="\t")
        # Exclude button presses
        stim_df = events_df[~events_df["trial_type"].isin(["btnL", "btnR"])].copy()

        # Build MNE events array from sample column
        # In ds000117, 'sample' is the sample index at sfreq=1100 Hz
        sample_indices = stim_df["sample"].values.astype(int)
        
        # Binary label: 1 = Face (Initial/Immediate/Delayed Famous/Unfamiliar), 0 = Scrambled
        is_face = stim_df["trial_type"].str.contains("F") & ~stim_df["trial_type"].str.contains("SF")
        condition_ids = is_face.astype(int).values  # 1 = face, 0 = scrambled

        events_mne = np.column_stack([
            sample_indices,
            np.zeros_like(sample_indices),
            condition_ids,
        ])

        print(f"Epoching {len(events_mne)} stimulus trials (tmin=-0.2s, tmax=1.8s)...")
        epochs = mne.Epochs(
            raw,
            events_mne,
            tmin=-0.2,
            tmax=1.8,
            baseline=(-0.2, 0.0),
            detrend=1,
            preload=True,
            verbose=False,
        )

        print("Resampling epochs to 256 Hz...")
        epochs.resample(256.0, verbose=False)

        # Trim to exact 512 samples (2.0s at 256 Hz)
        ep_data = epochs.get_data(copy=True)[:, :, :512]  # (N_trials, 306, 512)
        print(f"Epoched data shape: {ep_data.shape}")

        # Extract sensor metadata
        meta = extract_sensor_metadata_from_mne(epochs.info, dataset_name="ds000117")
        pos_ori, sensor_type, _ = normalize_sensor_positions(meta, mapping_mode="default")

        # Normalize using BrainOmni sensortype-wise normalization
        print("Normalizing across sensor types (MAG vs GRAD)...")
        norm_epochs = np.stack(
            [sensortype_wise_normalize(ep_data[i], sensor_type) for i in range(len(ep_data))],
            axis=0,
        )

        n_ep = len(ep_data)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "x": torch.from_numpy(norm_epochs).float(),  # (N_trials, 306, 512)
            "pos_ori": torch.from_numpy(pos_ori).float(),  # (306, 6)
            "sensor_type": torch.from_numpy(sensor_type).long(),  # (306,)
            "is_face": torch.from_numpy(condition_ids[:n_ep]).long(),  # (N_trials,)
            "trial_type": list(stim_df["trial_type"].iloc[:n_ep].values),
            "subject_id": subject,
            "session": session,
            "run": run,
            "frame": meta.frame.value,
        }
        torch.save(payload, out_path)
        print(f"Successfully saved SQUID cache to {out_path} ({out_path.stat().st_size / (1024*1024):.1f} MB)")

    finally:
        if not keep_temp_raw:
            if tmp_fif.exists():
                tmp_fif.unlink()
                print(f"Cleaned up temporary raw FIF: {tmp_fif}")
            if tmp_events.exists():
                tmp_events.unlink()

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream and cache ds000117 SQUID run")
    parser.add_argument("--subject", type=str, default="sub-01")
    parser.add_argument("--run", type=str, default="run-01")
    parser.add_argument("--out-dir", type=str, default="data/cache/ds000117")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    stream_and_cache_ds000117_run(
        subject=args.subject,
        session="ses-meg",
        run=args.run,
        out_dir=args.out_dir,
        keep_temp_raw=args.keep_temp,
    )

