"""Preprocessing and sensor-metadata mapping for OPM-Bridge."""

from opm_bridge.preprocessing.sensor_mapping import (
    apply_soft_hfc,
    compute_soft_hfc_projector,
    detect_coordinate_frame,
    extract_sensor_metadata_from_mne,
    normalize_sensor_positions,
    sensortype_wise_normalize,
)

__all__ = [
    "apply_soft_hfc",
    "compute_soft_hfc_projector",
    "detect_coordinate_frame",
    "extract_sensor_metadata_from_mne",
    "normalize_sensor_positions",
    "sensortype_wise_normalize",
]
