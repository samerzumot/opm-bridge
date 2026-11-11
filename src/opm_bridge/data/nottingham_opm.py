"""Dataset adapter for Nottingham 192-ch triaxial OPM-MEG somatosensory dataset (Zenodo 11126593).

Dataset specifications:
  - DOI: `10.5281/zenodo.11126593` (Rier & Rhodes et al., eLife 2024), License: `CC-BY-4.0`.
  - Archives: `Adults.zip` (12.24 GB, 26 adults `sub-101`..`sub-126`) and
    `Children.zip` (12.67 GB, 27 children `sub-001`..`sub-027`).
  - Evaluating adult cohorts eliminates pediatric age and head-size confounds in the main transfer
    experiments.
  - Per `index_pinky_Tstat_func.m`:
      * Flat binary `*_task-braille_run-001_meg.dat` accompanied by `*_meg_read_info.txt`
        (giving matrix `Size`, `Precision`, `Ordering`), `*_meg.json` (`SamplingFrequency`),
        `*_channels.tsv` (with `name` ending in `X`/`Y`/`Z`, `Px,Py,Pz` in cm, and `Ox,Oy,Oz`
        unit orientation vectors in subject head/MRI space), and `*_events.tsv`
        (`Start_index` vs `Start_pinky` tactile Braille stimulation).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from opm_bridge.data.schema import CoordinateFrame, SensorMetadata, SensorType

NOTTINGHAM_ADULT_SUBJECTS: Tuple[str, ...] = tuple(f"sub-{i}" for i in range(101, 127))
NOTTINGHAM_CHILD_SUBJECTS: Tuple[str, ...] = tuple(f"sub-{i:03d}" for i in range(1, 28))


def parse_meg_read_info(read_info_path: str | Path) -> Tuple[Tuple[int, int], np.dtype, str]:
    """Parse Nottingham `*_meg_read_info.txt` (lines 68-74 of `index_pinky_Tstat_func.m`)."""
    lines = Path(read_info_path).read_text().strip().splitlines()
    size_parts = lines[0].split()
    n_rows, n_cols = int(size_parts[1]), int(size_parts[3])
    prec_str = lines[1].split()[1].lower()
    order_str = lines[2].split()[1].lower()

    dtype_map = {
        "double": np.float64,
        "float64": np.float64,
        "single": np.float32,
        "float32": np.float32,
    }
    base_dtype = dtype_map.get(prec_str, np.float64)
    endian = "<" if ("ieee-le" in order_str or "l" in order_str) else ">"
    dt = np.dtype(base_dtype).newbyteorder(endian)
    return (n_rows, n_cols), dt, order_str


def read_nottingham_opm_binary(
    dat_path: str | Path,
    read_info_path: str | Path,
) -> np.ndarray:
    """Read flat binary `*_meg.dat` into shape (n_channels, n_samples) in pure Python."""
    (n_rows, n_cols), dt, _ = parse_meg_read_info(read_info_path)
    raw_1d = np.fromfile(dat_path, dtype=dt)
    # MATLAB fread(FileID, [n_rows, n_cols]) fills column-major (Fortran order), then transposed `'`
    mat = raw_1d.reshape((n_rows, n_cols), order="F")
    # In `index_pinky_Tstat_func.m` line 77: `data = fread(...)'` -> (n_samples, n_channels),
    # and line 116: `data_f = [filtfilt(...)]'` -> (n_channels, n_samples)
    if mat.shape[0] < mat.shape[1]:
        return mat.astype(np.float64)
    return mat.T.astype(np.float64)


def extract_nottingham_sensor_metadata(channels_tsv_path: str | Path) -> Tuple[SensorMetadata, np.ndarray]:
    """Extract true 3D triaxial sensor positions (`Px,Py,Pz`) and orientations (`Ox,Oy,Oz`).

    Returns:
        meta: SensorMetadata for all good OPM channels (with `pos_3d` converted from cm to meters).
        good_indices: Integer indices of good channels into the raw binary matrix.
    """
    df = pd.read_csv(channels_tsv_path, sep="\t")
    good_mask = ~df["status"].astype(str).str.lower().str.startswith("bad")
    good_indices = np.where(good_mask.to_numpy())[0]
    df_good = df.iloc[good_indices].reset_index(drop=True)

    ch_names = df_good["name"].astype(str).tolist()
    # Note line 149 of `index_pinky_Tstat_func.m`: `pos / 100` converts cm to meters
    pos_cm = df_good[["Px", "Py", "Pz"]].to_numpy(dtype=np.float32)
    pos_m = pos_cm / 100.0 if np.max(np.abs(pos_cm)) > 1.0 else pos_cm

    ori = df_good[["Ox", "Oy", "Oz"]].to_numpy(dtype=np.float32)
    norms = np.linalg.norm(ori, axis=1, keepdims=True)
    norms = np.where(norms > 1e-8, norms, 1.0)
    ori = ori / norms

    axis_labels: List[str] = []
    for nm in ch_names:
        if nm.endswith("X"):
            axis_labels.append("X")
        elif nm.endswith("Y"):
            axis_labels.append("Y")
        elif nm.endswith("Z"):
            axis_labels.append("Z")
        else:
            axis_labels.append("radial")

    sensor_type = np.full(len(ch_names), int(SensorType.MAG), dtype=np.int32)
    meta = SensorMetadata(
        ch_names=ch_names,
        pos_3d=pos_m,
        ori_3d=ori,
        sensor_type=sensor_type,
        frame=CoordinateFrame.HEAD,
        is_triaxial=True,
        axis_labels=axis_labels,
        dev_head_t=np.eye(4, dtype=np.float32),
        has_digitized_landmarks=True,
    )
    return meta, good_indices


def parse_nottingham_braille_events(
    events_tsv_path: str | Path, sfreq: float | None = None,
) -> Dict[str, np.ndarray]:
    """Return zero-based stimulus indices from this release's one-based samples.

    When sfreq is supplied, verify the conversion against the onset column.
    This convention is specific to Zenodo 11126593, not generic BIDS events.
    """
    df = pd.read_csv(events_tsv_path, sep="\t")
    type_col = "type" if "type" in df.columns else "trial_type"
    df = df[df[type_col].astype(str).str.startswith(("Start_index", "Start_pinky"))]
    types = df[type_col].astype(str)
    source_samples = df["sample"].to_numpy(dtype=np.float64)
    if (not np.isfinite(source_samples).all() or (source_samples < 1).any()
            or not np.equal(source_samples, np.rint(source_samples)).all()):
        raise ValueError("Nottingham stimulus samples must be positive one-based integers")
    samps = source_samples.astype(np.int64) - 1
    if sfreq is not None:
        if not np.isfinite(sfreq) or sfreq <= 0:
            raise ValueError("sfreq must be positive and finite")
        onsets = df["onset"].to_numpy(dtype=np.float64)
        if (not np.isfinite(onsets).all() or (onsets < 0).any()
                or not np.array_equal(np.rint(onsets * sfreq), samps)):
            raise ValueError("Nottingham sample-minus-one indices disagree with onset times")

    idx_mask = types.str.startswith("Start_index").to_numpy()
    pnk_mask = types.str.startswith("Start_pinky").to_numpy()

    return {
        "index_samples": samps[idx_mask],
        "pinky_samples": samps[pnk_mask],
    }

