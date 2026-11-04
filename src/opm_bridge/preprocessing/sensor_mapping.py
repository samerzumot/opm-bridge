"""Harmonized OPM and SQUID sensor metadata extraction, normalization, and ablations (FR1, FR2, FR3, FR7)."""

from __future__ import annotations

from typing import Literal, Tuple

import mne
import numpy as np

from opm_bridge.data.schema import CoordinateFrame, SensorMetadata, SensorType

# MNE FIFF coil types known to represent magnetometers (SQUID and OPM)
MAG_COIL_TYPES = {
    2000,  # FIFFV_COIL_POINT_MAGNETOMETER (used by ds005346 Quanmag OPM)
    3022,  # FIFFV_COIL_VV_MAG_T1
    3023,  # FIFFV_COIL_VV_MAG_T2
    3024,  # FIFFV_COIL_VV_MAG_T3 (used by Elekta and ds005107 proxy layout)
    4001,  # FIFFV_COIL_MAGNES_MAG
    8001,  # FIFFV_COIL_QUSPIN_ZFOPM_MAG
    8002,  # FIFFV_COIL_QUSPIN_ZFOPM_MAG2
    8003,  # FIFFV_COIL_FIELDLINE_OPM_MAG_GEN1
    8004,  # FIFFV_COIL_KERNEL_OPM_MAG_GEN1
}


def detect_coordinate_frame(
    info: mne.Info,
    dataset_name: str = "",
) -> Tuple[CoordinateFrame, bool, np.ndarray]:
    """Identify the coordinate frame, digitization status, and dev_head_t of an MNE Info object."""
    dev_head_t = np.eye(4, dtype=np.float32)
    if info.get("dev_head_t") is not None and "trans" in info["dev_head_t"]:
        dev_head_t = np.array(info["dev_head_t"]["trans"], dtype=np.float32)

    has_dig = bool(info.get("dig") is not None and len(info["dig"]) > 0)

    if "ds005107" in dataset_name.lower():
        # ds005107 README explicitly states Elekta layout approximation without optical scanning
        return CoordinateFrame.APPROX_ELEKTA, False, dev_head_t

    chs = info.get("chs", [])
    if not chs:
        return CoordinateFrame.UNKNOWN, has_dig, dev_head_t

    coord_frame_id = int(chs[0].get("coord_frame", 0))
    # FIFFV_COORD_HEAD == 4, FIFFV_COORD_DEVICE == 1, FIFFV_COORD_MRI == 5
    if coord_frame_id == 4:
        return CoordinateFrame.HEAD, has_dig, dev_head_t
    if coord_frame_id == 5:
        return CoordinateFrame.MRI, has_dig, dev_head_t
    if coord_frame_id == 1:
        return CoordinateFrame.DEVICE, has_dig, dev_head_t
    return CoordinateFrame.UNKNOWN, has_dig, dev_head_t


def extract_sensor_metadata_from_mne(
    info: mne.Info,
    dataset_name: str = "",
    apply_dev_head_transform: bool = False,
) -> SensorMetadata:
    """Extract SensorMetadata from an MNE Info object, fixing upstream int coil_type bug.

    Upstream BrainOmni checks `if "MAG" in str(i["coil_type"])`, which fails when
    `coil_type` is stored as a plain integer (e.g. 2000 or 3024) rather than an MNE NamedInt.
    Here we check both integer FIFF constants and string names.
    """
    frame, has_dig, dev_head_t = detect_coordinate_frame(info, dataset_name=dataset_name)
    ch_names: list[str] = []
    pos_list: list[np.ndarray] = []
    ori_list: list[np.ndarray] = []
    st_list: list[int] = []
    axis_labels: list[str] = []

    rot = dev_head_t[:3, :3]
    trans = dev_head_t[:3, 3]

    for ch in info["chs"]:
        kind = int(ch["kind"])
        if kind not in (1, 2):  # 1 = MEG, 2 = EEG
            continue
        ch_name = str(ch["ch_name"])
        ch_names.append(ch_name)
        loc = np.asarray(ch["loc"], dtype=np.float32)
        xyz = loc[:3].copy()

        if kind == 2:  # EEG
            ori = np.zeros(3, dtype=np.float32)
            st = int(SensorType.EEG)
            axis_labels.append("eeg")
        else:  # MEG (SQUID or OPM)
            coil_int = int(ch["coil_type"])
            coil_str = str(ch["coil_type"]).upper()
            dir_idx = 1 if "PLANAR" in coil_str else 3
            ori = loc[3 * dir_idx : 3 * (dir_idx + 1)].copy()
            norm = np.linalg.norm(ori)
            if norm > 1e-8:
                ori = ori / norm

            if coil_int in MAG_COIL_TYPES or "MAG" in coil_str or "OPM" in coil_str:
                st = int(SensorType.MAG)
            else:
                st = int(SensorType.GRAD)

            if ch_name.endswith(("X", "_x", "-x")):
                axis_labels.append("X")
            elif ch_name.endswith(("Y", "_y", "-y")):
                axis_labels.append("Y")
            elif ch_name.endswith(("Z", "_z", "-z")):
                axis_labels.append("Z")
            else:
                axis_labels.append("radial")

        if apply_dev_head_transform and frame == CoordinateFrame.DEVICE:
            xyz = rot @ xyz + trans
            if kind == 1:
                ori = rot @ ori

        pos_list.append(xyz)
        ori_list.append(ori)
        st_list.append(st)

    if apply_dev_head_transform and frame == CoordinateFrame.DEVICE:
        frame = CoordinateFrame.HEAD

    is_triaxial = any(a in ("X", "Y") for a in axis_labels) and any(a == "Z" for a in axis_labels)

    return SensorMetadata(
        ch_names=ch_names,
        pos_3d=np.stack(pos_list, axis=0).astype(np.float32),
        ori_3d=np.stack(ori_list, axis=0).astype(np.float32),
        sensor_type=np.array(st_list, dtype=np.int32),
        frame=frame,
        is_triaxial=is_triaxial,
        axis_labels=axis_labels,
        dev_head_t=dev_head_t,
        has_digitized_landmarks=has_dig,
    )


def normalize_sensor_positions(
    meta: SensorMetadata,
    mapping_mode: Literal[
        "default",
        "opm_as_grad",
        "zero_orientation",
        "flip_orientation",
        "radial_only",
        "scramble_channels",
    ] = "default",
    rng_seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Produce BrainOmni-compatible (pos_ori [C', 6], sensor_type [C'], channel_keep_mask [C]).

    Upstream BrainOmni centers `pos[:, :3]` per modality (EEG vs MEG) and scales by
    `sqrt(3 * mean(sum(pos^2, axis=1)))`, leaving `pos[:, 3:6]` (orientation) as a unit vector.

    Supports diagnostics and ablations required by FR7 (RQ3):
      - "default": OPM mapped to MAG (1), unit orientation vector preserved.
      - "opm_as_grad": OPM mapped to GRAD (2) to test sensor-type embedding sensitivity.
      - "zero_orientation": Orientation vector set to [0, 0, 0] (position-only ablation).
      - "flip_orientation": Orientation vector negated (-ori) to test sign convention sensitivity.
      - "radial_only": If triaxial (X/Y/Z), keep only radial (Z) channels.
      - "scramble_channels": Permute sensor metadata across channels (sanity control).
    """
    c = len(meta.ch_names)
    keep_mask = np.ones(c, dtype=bool)
    if mapping_mode == "radial_only" and meta.is_triaxial and meta.axis_labels is not None:
        keep_mask = np.array([a == "Z" for a in meta.axis_labels], dtype=bool)

    pos = meta.pos_3d[keep_mask].copy().astype(np.float32)
    ori = meta.ori_3d[keep_mask].copy().astype(np.float32)
    st = meta.sensor_type[keep_mask].copy().astype(np.int32)

    eeg_mask = st == int(SensorType.EEG)
    meg_mask = (st == int(SensorType.MAG)) | (st == int(SensorType.GRAD))

    if eeg_mask.any():
        eeg_mean = np.mean(pos[eeg_mask], axis=0, keepdims=True)
        pos[eeg_mask] -= eeg_mean
        eeg_scale = np.sqrt(3.0 * np.mean(np.sum(pos[eeg_mask] ** 2, axis=1)))
        if eeg_scale > 1e-8:
            pos[eeg_mask] /= eeg_scale

    if meg_mask.any():
        meg_mean = np.mean(pos[meg_mask], axis=0, keepdims=True)
        pos[meg_mask] -= meg_mean
        meg_scale = np.sqrt(3.0 * np.mean(np.sum(pos[meg_mask] ** 2, axis=1)))
        if meg_scale > 1e-8:
            pos[meg_mask] /= meg_scale

    if mapping_mode == "opm_as_grad":
        st[meg_mask] = int(SensorType.GRAD)
    elif mapping_mode == "zero_orientation":
        ori[:] = 0.0
    elif mapping_mode == "flip_orientation":
        ori = -ori
    elif mapping_mode == "scramble_channels":
        rng = np.random.default_rng(rng_seed)
        perm = rng.permutation(pos.shape[0])
        pos = pos[perm]
        ori = ori[perm]
        st = st[perm]

    pos_ori = np.hstack([pos, ori]).astype(np.float32)
    return pos_ori, st, keep_mask


def sensortype_wise_normalize(
    data: np.ndarray,
    sensor_type: np.ndarray,
) -> np.ndarray:
    """Apply BrainOmni's sensor-type-wise common-average reference and group z-score.

    Preserves relative amplitude ratios across channels of the same sensor type.
    """
    out = data.copy().astype(np.float64)
    for st_val, eps in (
        (int(SensorType.EEG), 1e-5),
        (int(SensorType.MAG), 1e-13),
        (int(SensorType.GRAD), 1e-13),
    ):
        mask = sensor_type == st_val
        if mask.any():
            sub = out[mask, :]
            sub = sub - np.mean(sub, axis=0, keepdims=True)
            std = np.std(sub) + eps
            out[mask, :] = sub / std
    return out.astype(np.float32)

