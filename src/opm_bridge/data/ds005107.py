"""Dataset adapter for OpenNeuro ds005107 ('FACE-DEC', 64-ch Quanmag OPM-MEG).

Dataset specifications:
  - 21 participants (`sub-01`..`sub-23`, with `sub-05` and `sub-12` absent).
  - 64 MEG channels typed `MEGMAG` (`coil_type=3024`, unit `T`), `sfreq=1000 Hz`, 50 Hz line.
  - Per `ds005107/README`: sensor coordinates in the FIF files were approximated from an
    Elekta layout for 2D topomap visualization (`CoordinateFrame.APPROX_ELEKTA`) and
    `dev_head_t` is the 4x4 identity matrix (`DigitizedLandmarks: false`).
  - Stimulus identities (`1.bmp`..`64.bmp`) are stored in `beh/*_events.tsv` (`stim_file`),
    while `meg/*_events.tsv` only holds trigger codes (`1`=Stimulus, `99`=Catch, `999`=Button).
  - Per `code/analysis/face_1_prep.py` and `face_3_rsa.py`:
      * Projector latency offset: +33 ms (`epoch_proj_latency = 0.033`).
      * 64 face stimuli follow a balanced 2x2x2x2 factorial design across 4 binary dimensions:
        - `race`: stimuli 1..32 (1) vs 33..64 (0)
        - `gender`: ((stim_id - 1) // 16) % 2 == 0 -> 1 else 0
        - `age`: ((stim_id - 1) // 8) % 2 == 0 -> 1 else 0
        - `emotion`: ((stim_id - 1) // 4) % 2 == 0 -> 1 else 0
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Literal, Tuple

import mne
import numpy as np
import pandas as pd

DS005107_SUBJECTS: Tuple[str, ...] = (
    "sub-01", "sub-02", "sub-03", "sub-04", "sub-06", "sub-07", "sub-08",
    "sub-09", "sub-10", "sub-11", "sub-13", "sub-14", "sub-15", "sub-16",
    "sub-17", "sub-18", "sub-19", "sub-20", "sub-21", "sub-22", "sub-23",
)

FaceDimension = Literal["race", "gender", "age", "emotion"]


def stimulus_id_to_face_labels(stim_id: int) -> Dict[FaceDimension, int]:
    """Map stimulus number (1..64) to its 4 binary factorial labels (from `face_3_rsa.py`)."""
    if not (1 <= stim_id <= 64):
        raise ValueError(f"stim_id must be in 1..64, got {stim_id}")
    idx0 = stim_id - 1
    return {
        "race": 1 if idx0 < 32 else 0,
        "gender": 1 if (idx0 // 16) % 2 == 0 else 0,
        "age": 1 if (idx0 // 8) % 2 == 0 else 0,
        "emotion": 1 if (idx0 // 4) % 2 == 0 else 0,
    }


def align_ds005107_events(
    meg_events: np.ndarray,
    beh_tsv_path: str | Path,
    sfreq: float = 1000.0,
    proj_latency_sec: float = 0.033,
) -> np.ndarray:
    """Align MNE trigger events with behavioral stimulus IDs (`face_1_prep.py` lines 240-255).

    Returns:
        valid_events: MNE event array of shape (N_valid, 3) where column 0 is latency-corrected
        sample index and column 2 is stimulus ID in 1..64 (excluding one-back catch trials 99
        and false-alarm button presses 999).
    """
    stim_events = meg_events[meg_events[:, 2] == 1].copy()
    df_beh = pd.read_csv(beh_tsv_path, sep="\t")

    stim_list: List[int] = []
    for _, row in df_beh.iterrows():
        stim_str = str(row["stim_file"]).strip()
        # e.g. 'meg/18.bmp' -> 18
        fname = stim_str.split("/")[-1].split(".")[0]
        stim_num = int(fname)
        if int(row["is_oneback"]) == 0 and int(row["is_responsed"]) == 1:
            stim_num = 999  # False alarm
        elif int(row["is_oneback"]) == 1:
            stim_num = 99  # One-back catch trial (already coded as 99 on STIM channel)
        stim_list.append(stim_num)

    stim_arr = np.asarray(stim_list, dtype=np.int64)
    non_catch = stim_arr[stim_arr != 99]
    if len(non_catch) != len(stim_events):
        raise ValueError(
            f"Event count mismatch: {len(stim_events)} trigger-1 events vs "
            f"{len(non_catch)} non-catch behavioral rows in {beh_tsv_path}"
        )

    stim_events[:, 2] = non_catch
    valid_events = stim_events[stim_events[:, 2] != 999].copy()
    valid_events[:, 0] += int(round(proj_latency_sec * sfreq))
    return valid_events

