"""Harmonized OPM/SQUID sample and metadata schema with explicit coordinate frames."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

import numpy as np
import torch


class CoordinateFrame(str, Enum):
    """Explicit coordinate frame tag required by FR1 and Gate 5b."""

    DEVICE = "device"  # FIFFV_COORD_DEVICE (1): Helmet / sensor array frame
    HEAD = "head"  # FIFFV_COORD_HEAD (4): Digitized landmark head frame (Nasion/LPA/RPA)
    MRI = "mri"  # Subject anatomical MRI / MNI frame
    APPROX_ELEKTA = "approx_elekta"  # Proxy SQUID layout (e.g., ds005107 README caveat)
    UNKNOWN = "unknown"


class SensorType(int, Enum):
    """BrainOmni sensor type encoding (matches upstream SENSOR_TYPE_DICT)."""

    EEG = 0
    MAG = 1
    GRAD = 2


@dataclass
class SensorMetadata:
    """Per-recording sensor geometry and frame metadata.

    Attributes:
        ch_names: List of channel names (length C).
        pos_3d: Raw 3D Cartesian positions in meters, shape (C, 3).
        ori_3d: Unit 3D sensitive-axis orientation vectors, shape (C, 3).
        sensor_type: Integer sensor type per channel (0=EEG, 1=MAG, 2=GRAD), shape (C,).
        frame: Explicit CoordinateFrame of pos_3d and ori_3d.
        is_triaxial: Whether sensors include radial/tangential triaxial channels (X/Y/Z).
        axis_labels: Optional per-channel axis labels ('X', 'Y', 'Z', or 'radial').
        dev_head_t: Optional 4x4 homogeneous transform from device to head frame.
        has_digitized_landmarks: Whether cardinal landmarks / head points were digitized.
    """

    ch_names: list[str]
    pos_3d: np.ndarray  # (C, 3), float32
    ori_3d: np.ndarray  # (C, 3), float32
    sensor_type: np.ndarray  # (C,), int32
    frame: CoordinateFrame
    is_triaxial: bool = False
    axis_labels: Optional[list[str]] = None
    dev_head_t: Optional[np.ndarray] = None  # (4, 4)
    has_digitized_landmarks: bool = False

    def __post_init__(self) -> None:
        c = len(self.ch_names)
        assert self.pos_3d.shape == (c, 3), f"Expected pos_3d ({c}, 3), got {self.pos_3d.shape}"
        assert self.ori_3d.shape == (c, 3), f"Expected ori_3d ({c}, 3), got {self.ori_3d.shape}"
        assert self.sensor_type.shape == (c,), (
            f"Expected sensor_type ({c},), got {self.sensor_type.shape}"
        )


@dataclass
class HarmonizedSample:
    """Harmonized single-epoch or segment sample ready for BrainOmni and baselines.

    Attributes:
        x: Normalized neural time series, shape (C, T), float32, at 256 Hz.
        pos_ori: Normalized position + orientation tensor, shape (C, 6), float32.
        sensor_type: Sensor type tensor, shape (C,), int64.
        label: Downstream task class label (int).
        subject_id: Subject identifier string (for strict subject-wise splitting).
        dataset_name: Dataset identifier string.
        frame: Explicit CoordinateFrame tag carried from the loader.
        metadata: Additional trial/run metadata (e.g., session, run, stim_id, snr_db).
    """

    x: torch.Tensor  # (C, T)
    pos_ori: torch.Tensor  # (C, 6)
    sensor_type: torch.Tensor  # (C,)
    label: int
    subject_id: str
    dataset_name: str
    frame: CoordinateFrame
    metadata: Dict[str, Any] = field(default_factory=dict)

