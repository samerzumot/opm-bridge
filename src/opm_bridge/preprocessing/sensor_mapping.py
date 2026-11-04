"""Harmonized OPM and SQUID sensor metadata extraction, normalization, and geometric ablations."""

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

    Supports experimental diagnostics and ablations:
      - "default": OPM mapped to MAG (1), unit orientation vector preserved.
      - "opm_as_grad": OPM mapped to GRAD (2) to assess sensor-type embedding alignment.
      - "zero_orientation": Orientation vector set to [0, 0, 0] (position-only ablation).
      - "flip_orientation": Orientation vector negated (-ori) to evaluate sign sensitivity.
      - "radial_only": In triaxial arrays, retain radial (Z) channels only.
      - "scramble_channels": Permute sensor metadata across channels (topology sanity check).
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


def compute_soft_hfc_projector(
    orientations: np.ndarray,
    alpha: float = 0.25,
) -> np.ndarray:
    """Compute the Soft Homogeneous Field Correction (Soft-HFC) spatial projection matrix.

    Homogeneous external interference in OPM-MEG manifests as a uniform 3D magnetic field:
        B_ext = [Bx, By, Bz]^T
    Sensor responses are given by the projection onto sensitive axes Q in R^{C x 3}.

    Hard HFC projects orthogonally away from col(Q):
        P_hard = I - Q (Q^T Q)^{-1} Q^T
    which removes 3 degrees of freedom and creates a rank-deficient covariance matrix.

    Soft-HFC (Tierney et al. 2021 / Mellor et al. 2022) introduces shrinkage alpha in [0, 1]:
        P_soft(alpha) = I - alpha * Q (Q^T Q)^{-1} Q^T

    When alpha in [0, 1), all eigenvalues of P_soft are strictly in (0, 1], guaranteeing:
      1. Full numerical rank rank(P_soft) == C (no singular covariance matrices or zero eigenvalues).
      2. Partial attenuation of homogeneous environmental interference by (1 - (1 - alpha)^2) in power
         (43.75% power attenuation / 25% amplitude shrinkage at alpha=0.25).
      3. Partial shrinkage of any neural signals projecting onto col(Q), with zero distortion of
         neural signals in the orthogonal complement subspace.

    Uses QR decomposition Q = Q_R R where Q_R has orthonormal columns (Q_R^T Q_R = I),
    yielding P_soft = I - alpha * Q_R Q_R^T without explicit matrix inversion.

    Args:
        orientations: Sensor orientation matrix [C, 3] or [C, 6] (if pos_ori is passed).
        alpha: Shrinkage parameter in [0.0, 1.0]. alpha=0.0 is identity, alpha=1.0 is hard HFC,
            default alpha=0.25.

    Returns:
        P_soft: Spatial projection matrix [C, C] (float32).
    """
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0.0, 1.0], got {alpha}")

    q = np.asarray(orientations, dtype=np.float64)
    if q.ndim != 2:
        raise ValueError(f"orientations must be 2D [C, 3], got shape {q.shape}")
    if q.shape[1] > 3:
        # If pos_ori [C, 6] was passed, extract orientation slice [:, 3:6]
        q = q[:, 3:6]
    if q.shape[1] != 3:
        raise ValueError(f"orientations must have 3 columns, got {q.shape[1]}")

    c = q.shape[0]
    if alpha == 0.0 or c == 0:
        return np.eye(c, dtype=np.float32)

    # Check for near-zero orientation vectors and normalize non-zero rows
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    valid_mask = (norms > 1e-8).flatten()
    if not valid_mask.any():
        return np.eye(c, dtype=np.float32)

    # QR decomposition on valid orientations to get orthonormal basis Q_R
    # If some sensors have zero orientation (e.g. EEG in hybrid setup), project only MEG channels
    q_valid = q[valid_mask] / np.maximum(norms[valid_mask], 1e-8)
    qr_basis, _ = np.linalg.qr(q_valid, mode="reduced")  # [C_valid, k <= 3]

    p_valid = np.eye(q_valid.shape[0], dtype=np.float64) - alpha * (qr_basis @ qr_basis.T)

    p_full = np.eye(c, dtype=np.float64)
    valid_indices = np.where(valid_mask)[0]
    p_full[np.ix_(valid_indices, valid_indices)] = p_valid

    return p_full.astype(np.float32)


def apply_soft_hfc(
    data: np.ndarray,
    orientations: np.ndarray,
    alpha: float = 0.25,
) -> np.ndarray:
    """Apply Soft-HFC spatial filtering to multichannel OPM data.

    Args:
        data: Multichannel continuous or epoched array of shape [C, T] or [B, C, T].
        orientations: Sensor orientation matrix [C, 3] or [C, 6].
        alpha: Shrinkage parameter in [0.0, 1.0], default 0.25.

    Returns:
        Filtered array of same shape and float32 dtype.
    """
    p_soft = compute_soft_hfc_projector(orientations=orientations, alpha=alpha)

    if data.ndim == 2:
        # [C, T] -> P @ X
        return (p_soft @ data.astype(np.float32)).astype(np.float32)
    elif data.ndim == 3:
        # [B, C, T] -> einsum or matmul
        b, c, t = data.shape
        reshaped = data.astype(np.float32).reshape(b, c, t)
        filtered = np.matmul(p_soft, reshaped)
        return filtered.astype(np.float32)
    else:
        raise ValueError(f"data must be 2D [C, T] or 3D [B, C, T], got shape {data.shape}")


def align_subjects_by_sensor_geometry(
    subj_X: dict[str, np.ndarray],
    subj_pos: dict[str, np.ndarray],
    subj_st: dict[str, np.ndarray],
    decimals: int = 4,
) -> Tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Align channels across subjects by matching physical 6D sensor geometry (pos_ori).

    When per-subject bad-channel rejection drops different noisy sensors across subjects,
    naive index slicing (`[:, :min_ch, :]`) misaligns physical helmet channels across subjects.
    This function finds the intersection of physical sensors present in all subjects
    (matched by rounded 6D `[x, y, z, ox, oy, oz]` coordinates) and orders every subject's
    channels identically so channel index `k` refers to the exact same physical sensor.
    """
    subjects = sorted(subj_X.keys())
    if not subjects:
        return {}, {}, {}

    pos_tuples_by_subj: dict[str, list[tuple[float, ...]]] = {}
    for s in subjects:
        rounded = np.round(np.asarray(subj_pos[s], dtype=np.float32), decimals=decimals)
        pos_tuples_by_subj[s] = [tuple(float(v) for v in row) for row in rounded]

    common_sensors = set(pos_tuples_by_subj[subjects[0]])
    for s in subjects[1:]:
        common_sensors.intersection_update(pos_tuples_by_subj[s])

    # Preserve canonical sensor order from the first subject
    canonical_order = [pt for pt in pos_tuples_by_subj[subjects[0]] if pt in common_sensors]
    if not canonical_order:
        # Fallback if subjects have subject-specific coregistrations without shared slots
        min_ch = min(subj_X[s].shape[1] for s in subjects)
        return (
            {s: np.asarray(subj_X[s], dtype=np.float32)[:, :min_ch, :] for s in subjects},
            {s: np.asarray(subj_pos[s], dtype=np.float32)[:min_ch] for s in subjects},
            {s: np.asarray(subj_st[s], dtype=np.int64)[:min_ch] for s in subjects},
        )

    aligned_X: dict[str, np.ndarray] = {}
    aligned_pos: dict[str, np.ndarray] = {}
    aligned_st: dict[str, np.ndarray] = {}

    for s in subjects:
        idx_map = {pt: idx for idx, pt in enumerate(pos_tuples_by_subj[s])}
        sel = [idx_map[pt] for pt in canonical_order]
        aligned_X[s] = np.asarray(subj_X[s], dtype=np.float32)[:, sel, :]
        aligned_pos[s] = np.asarray(subj_pos[s], dtype=np.float32)[sel]
        aligned_st[s] = np.asarray(subj_st[s], dtype=np.int64)[sel]

    return aligned_X, aligned_pos, aligned_st



