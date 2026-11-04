"""Unit and smoke tests for OPM-Bridge Phase 0 / Phase 1 components."""

from __future__ import annotations

from pathlib import Path

import mne
import numpy as np
import pandas as pd
import pytest
import torch

from opm_bridge.data.ds005107 import align_ds005107_events, stimulus_id_to_face_labels
from opm_bridge.data.nottingham_opm import (
    extract_nottingham_sensor_metadata,
    parse_nottingham_braille_events,
    read_nottingham_opm_binary,
)
from opm_bridge.data.schema import CoordinateFrame, SensorMetadata, SensorType
from opm_bridge.eval.metrics import (
    check_phase15_signal_gate,
    compute_curve_summaries,
    evaluate_primary_comparison,
    subject_bootstrap_ci,
)
from opm_bridge.models.baselines import ClassicalBandPowerDecoder, CompactMEGNet
from opm_bridge.models.vendor_brainomni.brainomni.model import BrainOmni
from opm_bridge.models.wrapper import BrainOmniClassifier
from opm_bridge.preprocessing.sensor_mapping import (
    align_subjects_by_sensor_geometry,
    apply_soft_hfc,
    compute_soft_hfc_projector,
    extract_sensor_metadata_from_mne,
    normalize_sensor_positions,
    sensortype_wise_normalize,
)


def _tiny_test_config() -> dict:
    return {
        "window_length": 512,
        "n_filters": 16,
        "ratios": [8, 4, 2],
        "kernel_size": 5,
        "last_kernel_size": 5,
        "n_dim": 64,
        "n_head": 4,
        "n_neuro": 8,
        "dropout": 0.0,
        "codebook_dim": 64,
        "codebook_size": 64,
        "num_quantizers": 2,
        "rotation_trick": True,
        "quantize_optimize_method": "ema",
        "overlap_ratio": 0.25,
        "lm_dim": 64,
        "lm_head": 4,
        "lm_depth": 3,
        "lm_dropout": 0.0,
        "mask_ratio": 0.5,
        "num_quantizers_used": 2,
    }


def test_ds005107_factorial_labels_and_events(tmp_path: Path) -> None:
    # Verify all 4 binary dimensions are strictly balanced (32 vs 32 across stimuli 1..64)
    counts = {k: 0 for k in ("race", "gender", "age", "emotion")}
    for s in range(1, 65):
        lbls = stimulus_id_to_face_labels(s)
        for k, v in lbls.items():
            counts[k] += v
    assert counts == {"race": 32, "gender": 32, "age": 32, "emotion": 32}

    # Test event alignment (+33ms offset, catch removal, false-alarm removal)
    beh_path = tmp_path / "events.tsv"
    df = pd.DataFrame(
        {
            "onset": ["n/a"] * 4,
            "duration": ["n/a"] * 4,
            "is_oneback": [0, 1, 0, 0],
            "is_responsed": [0, 1, 1, 0],  # 3rd row is a false alarm (999)
            "response_time": [0.0, 0.45, 0.50, 0.0],
            "stim_file": ["meg/18.bmp", "meg/18.bmp", "meg/7.bmp", "meg/64.bmp"],
        }
    )
    df.to_csv(beh_path, sep="\t", index=False)

    # MNE trigger events: 3 normal stimuli (code 1) and 1 catch trial (code 99)
    meg_events = np.array(
        [
            [1000, 0, 1],
            [2000, 0, 99],
            [3000, 0, 1],
            [4000, 0, 1],
        ],
        dtype=np.int64,
    )
    aligned = align_ds005107_events(meg_events, beh_path, sfreq=1000.0, proj_latency_sec=0.033)
    # Catch (2000) and false-alarm (3000) excluded; remaining two shifted by +33 samples
    assert aligned.shape == (2, 3)
    assert aligned[0, 0] == 1033 and aligned[0, 2] == 18
    assert aligned[1, 0] == 4033 and aligned[1, 2] == 64


def test_nottingham_triaxial_loader_and_ablations(tmp_path: Path) -> None:
    # Create synthetic Nottingham BIDS files
    ch_path = tmp_path / "sub-101_channels.tsv"
    df_ch = pd.DataFrame(
        {
            "name": ["OPM01X", "OPM01Y", "OPM01Z", "OPM02X", "OPM02Y", "OPM02Z", "BAD01Z"],
            "Px": [5.0, 5.0, 5.0, -5.0, -5.0, -5.0, 0.0],
            "Py": [2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 0.0],
            "Pz": [6.0, 6.0, 6.0, 6.0, 6.0, 6.0, 0.0],
            "Ox": [1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "Oy": [0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
            "Oz": [0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            "status": ["good", "good", "good", "good", "good", "good", "bad"],
        }
    )
    df_ch.to_csv(ch_path, sep="\t", index=False)

    meta, good_idx = extract_nottingham_sensor_metadata(ch_path)
    assert len(good_idx) == 6
    assert meta.is_triaxial is True
    assert meta.frame == CoordinateFrame.HEAD
    # Positions converted from cm to m (< 0.2m)
    assert np.all(np.abs(meta.pos_3d) < 0.2)

    # Default vs radial_only vs opm_as_grad vs zero_orientation ablations
    pos_def, st_def, mask_def = normalize_sensor_positions(meta, "default")
    assert pos_def.shape == (6, 6)
    assert np.all(st_def == int(SensorType.MAG))
    assert mask_def.sum() == 6

    pos_rad, st_rad, mask_rad = normalize_sensor_positions(meta, "radial_only")
    assert pos_rad.shape == (2, 6)
    assert mask_rad.sum() == 2

    pos_grad, st_grad, _ = normalize_sensor_positions(meta, "opm_as_grad")
    assert np.all(st_grad == int(SensorType.GRAD))

    pos_zori, _, _ = normalize_sensor_positions(meta, "zero_orientation")
    assert np.allclose(pos_zori[:, 3:], 0.0)


def test_mne_integer_coil_type_extraction() -> None:
    info = mne.create_info(["MEG01", "MEG02", "STIM"], sfreq=1000.0, ch_types=["mag", "mag", "stim"])
    for idx, ch in enumerate(info["chs"][:2]):
        ch["loc"][:3] = np.array([0.08 * (1 if idx == 0 else -1), 0.02, 0.05])
        ch["loc"][9:12] = np.array([0.0, 0.0, 1.0])
        ch["coil_type"] = 2000  # Point magnetometer integer (ds005346)

    meta_5346 = extract_sensor_metadata_from_mne(info, dataset_name="ds005346")
    assert len(meta_5346.ch_names) == 2
    assert np.all(meta_5346.sensor_type == int(SensorType.MAG))

    meta_5107 = extract_sensor_metadata_from_mne(info, dataset_name="ds005107")
    assert meta_5107.frame == CoordinateFrame.APPROX_ELEKTA


def test_brainomni_wrapper_gradient_flow_across_arms() -> None:
    cfg = _tiny_test_config()
    x = torch.randn(2, 12, 512)
    pos = torch.randn(2, 12, 6)
    pos[:, :, 3:] = torch.nn.functional.normalize(pos[:, :, 3:], dim=-1)
    st = torch.ones(2, 12, dtype=torch.long)
    y = torch.tensor([0, 1], dtype=torch.long)

    # 1. `backbone_ft`: tokenizer gradients must be None/zero, backbone gradients non-zero
    bb1 = BrainOmni(**cfg)
    clf_bb = BrainOmniClassifier(bb1, num_classes=2, mode="backbone_ft")
    out1 = clf_bb(x, pos, st, labels=y)
    out1["loss"].backward()
    tok_grad_1 = clf_bb.backbone.tokenizer.sensor_embed.sensor_embedding_layer.weight.grad
    blk_grad_1 = clf_bb.backbone.blocks[0].time_attn.qkv.weight.grad
    assert tok_grad_1 is None
    assert blk_grad_1 is not None and torch.norm(blk_grad_1) > 0

    # 2. `control_a_random_e2e`: BOTH tokenizer (`BrainSensorModule`) and backbone receive gradients!
    bb2 = BrainOmni(**cfg)
    clf_ctrl_a = BrainOmniClassifier(bb2, num_classes=2, mode="control_a_random_e2e")
    out2 = clf_ctrl_a(x, pos, st, labels=y)
    out2["loss"].backward()
    tok_grad_2 = clf_ctrl_a.backbone.tokenizer.sensor_embed.pos_embedding_layer[0].weight.grad
    blk_grad_2 = clf_ctrl_a.backbone.blocks[0].time_attn.qkv.weight.grad
    assert tok_grad_2 is not None and torch.norm(tok_grad_2) > 0
    assert blk_grad_2 is not None and torch.norm(blk_grad_2) > 0


def test_baselines_and_statistical_protocol() -> None:
    # 1. CompactMEGNet forward/backward
    net = CompactMEGNet(n_channels=16, n_classes=2)
    x = torch.randn(4, 16, 256)
    y = torch.tensor([0, 1, 0, 1])
    out = net(x, labels=y)
    assert out["logits"].shape == (4, 2)
    out["loss"].backward()

    # 2. ClassicalBandPowerDecoder fit/predict
    X_np = np.random.randn(20, 8, 256).astype(np.float32)
    y_np = np.array([0, 1] * 10)
    clf = ClassicalBandPowerDecoder(sfreq=256.0).fit(X_np, y_np)
    preds = clf.predict(X_np)
    assert preds.shape == (20,)

    # 3. Primary comparison decision rules (§4.1)
    rng = np.random.default_rng(0)
    ctrl_scores = rng.normal(0.55, 0.02, size=(20, 3))
    pt_helped = ctrl_scores + 0.06
    res_helped = evaluate_primary_comparison(pt_helped, ctrl_scores, budget=2, sesoi=0.03)
    assert res_helped.verdict == "pre-training helped"

    pt_equiv = ctrl_scores + rng.normal(0.0, 0.005, size=(20, 3))
    res_equiv = evaluate_primary_comparison(pt_equiv, ctrl_scores, budget=2, sesoi=0.03)
    assert res_equiv.verdict == "equivalent"

    # 4. Phase 1.5 signal gate
    passed, _ = check_phase15_signal_gate(pt_helped, ctrl_scores, chance_level=0.50)
    assert passed is True


def test_soft_hfc_properties() -> None:
    rng = np.random.default_rng(42)
    n_ch = 12
    n_time = 100

    # Create synthetic 3D orientations on unit sphere
    ori = rng.normal(size=(n_ch, 3))
    ori /= np.linalg.norm(ori, axis=1, keepdims=True)

    # 1. Rank preservation check:
    # Hard HFC (alpha=1.0) must drop rank from 12 to 9
    p_hard = compute_soft_hfc_projector(ori, alpha=1.0)
    assert np.linalg.matrix_rank(p_hard) == n_ch - 3

    # Soft HFC (alpha=0.25) must retain FULL rank (12)
    p_soft = compute_soft_hfc_projector(ori, alpha=0.25)
    assert np.linalg.matrix_rank(p_soft) == n_ch
    eigenvals = np.linalg.eigvalsh(p_soft)
    assert np.all(eigenvals > 0.0)
    # The 3 lowest eigenvalues should be exactly (1 - 0.25) = 0.75
    assert np.isclose(eigenvals[0], 0.75, atol=1e-5)
    assert np.isclose(eigenvals[1], 0.75, atol=1e-5)
    assert np.isclose(eigenvals[2], 0.75, atol=1e-5)
    # The remaining eigenvalues must be 1.0
    assert np.allclose(eigenvals[3:], 1.0, atol=1e-5)

    # 2. Pure homogeneous interference attenuation
    # Field B_ext = [1.0, -0.5, 2.0]^T across time
    b_ext = np.array([[1.0], [-0.5], [2.0]])  # [3, 1]
    time_series = np.sin(np.linspace(0, 4 * np.pi, n_time))[None, :]  # [1, T]
    x_homo = (ori @ b_ext) @ time_series  # [C, T]

    # Hard HFC should completely eliminate homogeneous field
    x_hard_clean = apply_soft_hfc(x_homo, ori, alpha=1.0)
    assert np.allclose(x_hard_clean, 0.0, atol=1e-5)

    # Soft HFC (alpha=0.25) should attenuate by exactly 25% (leaving 75% amplitude)
    x_soft_clean = apply_soft_hfc(x_homo, ori, alpha=0.25)
    assert np.allclose(x_soft_clean, 0.75 * x_homo, atol=1e-5)

    # 3. Orthogonal signal preservation
    # Construct signal orthogonal to ori: X_ortho in nullspace of ori.T
    u, s, vh = np.linalg.svd(ori)
    null_basis = u[:, 3:]  # [C, C - 3]
    x_ortho = null_basis @ rng.normal(size=(n_ch - 3, n_time))  # [C, T]

    x_ortho_clean = apply_soft_hfc(x_ortho, ori, alpha=0.25)
    assert np.allclose(x_ortho_clean, x_ortho, atol=1e-5)

    # 4. 3D batch shape support [B, C, T]
    batch_data = np.stack([x_homo, x_ortho], axis=0)  # [2, C, T]
    batch_cleaned = apply_soft_hfc(batch_data, ori, alpha=0.25)
    assert batch_cleaned.shape == (2, n_ch, n_time)
    assert np.allclose(batch_cleaned[0], 0.75 * x_homo, atol=1e-5)
    assert np.allclose(batch_cleaned[1], x_ortho, atol=1e-5)

    # 5. Invalid alpha check
    with pytest.raises(ValueError):
        compute_soft_hfc_projector(ori, alpha=1.5)


def test_cross_subject_sensor_alignment() -> None:
    """Verify that physical 6D sensor geometry alignment fixes misaligned bad-channel drops."""
    rng = np.random.default_rng(123)
    # 10 canonical physical helmet slots
    canonical_pos = rng.normal(size=(10, 6)).astype(np.float32)
    canonical_st = np.ones(10, dtype=np.int64)

    # Subject 1 drops slot #2 (keeps 9 slots)
    keep_s1 = [0, 1, 3, 4, 5, 6, 7, 8, 9]
    # Subject 2 drops slots #0 and #5 (keeps 8 slots)
    keep_s2 = [1, 2, 3, 4, 6, 7, 8, 9]

    # Put each slot's true index into the signal so we can verify exact channel alignment
    x_full = np.broadcast_to(np.arange(10, dtype=np.float32)[None, :, None], (4, 10, 16)).copy()
    subj_X = {"sub-01": x_full[:, keep_s1, :], "sub-02": x_full[:, keep_s2, :]}
    subj_pos = {"sub-01": canonical_pos[keep_s1], "sub-02": canonical_pos[keep_s2]}
    subj_st = {"sub-01": canonical_st[keep_s1], "sub-02": canonical_st[keep_s2]}

    al_X, al_pos, al_st = align_subjects_by_sensor_geometry(subj_X, subj_pos, subj_st)
    # Common slots between keep_s1 and keep_s2: [1, 3, 4, 6, 7, 8, 9] -> 7 channels
    assert al_X["sub-01"].shape == (4, 7, 16)
    assert al_X["sub-02"].shape == (4, 7, 16)
    assert np.allclose(al_pos["sub-01"], al_pos["sub-02"])
    assert np.allclose(al_X["sub-01"], al_X["sub-02"])
    assert al_X["sub-01"][0, :, 0].tolist() == [1.0, 3.0, 4.0, 6.0, 7.0, 8.0, 9.0]



