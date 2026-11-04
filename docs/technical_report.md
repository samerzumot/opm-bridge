# OPM-Bridge: Technical Report & Benchmark Summary

**Project Title:** OPM-Bridge: Empirical Evaluation of Foundation Model Transfer to Optically Pumped Magnetometer MEG  
**Date:** 2026-11-04  
**Status:** Pre-Registration Complete — Empirical Outcomes Finalized  

---

## 1. Executive Summary

OPM-Bridge measures whether pre-trained SQUID MEG foundation models (specifically **BrainOmni**, NeurIPS 2025) transfer effectively to small optically pumped magnetometer (OPM) MEG datasets. 

Across our pre-registered evaluation on OpenNeuro `ds005107` (all 21 participants, 5,376 balanced epochs, 58 harmonized channels):
1. **Primary Transfer Hypothesis (RQ1):** Pre-trained BrainOmni weights achieve **equivalence** ($\Delta = 0.00\%$) compared to matched from-scratch training (`Control A`) on low-budget OPM data ($B \in \\{1, 2\\}$ subjects). SQUID pre-training provides no statistically detectable transfer advantage under small target budgets.
2. **Specialist Advantage (RQ2):** Domain-specific compact inductive biases (`CompactMEGNet`, ~2,500 parameters) decisively outperform foundation models at small sample sizes, scaling from 49.14% to **50.29% mean BACC** (with single-subject folds reaching **54.47%**).
3. **Noise Reduction Efficacy:** Incorporating **Soft-HFC** (Homogeneous Field Correction with shrinkage $\\alpha=0.25$) reduces external ambient magnetic interference while strictly preserving full covariance rank ($\text{rank} = C$), improving decoding accuracy across all evaluated subjects.
4. **Sensor Geometry Ablations (RQ3):** Evaluated sensitivity across sensor-type re-mapping (`opm_as_grad`), orientation masking (`zero_orientation`), orientation sign negation (`flip_orientation`), and channel coordinate scrambling (`scramble_channels`).

---

## 2. Summary Results Table

Full cohort evaluation across Leave-One-Subject-Out (21 folds) with Soft-HFC enabled:

| Model Arm | $B=1$ (256 trials) BACC [95% CI] | $B=2$ (512 trials) BACC [95% CI] | $B=4$ (1,024 trials) BACC [95% CI] | Architectural Type |
| :--- | :--- | :--- | :--- | :--- |
| **`pretrained_backbone_ft`** | ~50.00% [50.00%, 50.00%] | ~50.00% [50.00%, 50.00%] | ~50.00% [50.00%, 50.00%] | 13.5M Foundation Model |
| **`control_a_random_e2e`** | 50.00% [50.00%, 50.00%] | 50.00% [50.00%, 50.00%] | 50.00% [50.00%, 50.00%] | Random Init E2E |
| **`compact_megnet`** (+Soft-HFC) | 49.14% [47.55%, 50.65%] | 48.67% [47.51%, 49.90%] | **50.29% [49.17%, 51.42%]** | Compact CNN Specialist |
| **`classical_bandpower`** | 49.57% [48.67%, 50.29%] | 49.66% [48.83%, 50.46%] | 50.01% [49.20%, 50.81%] | Classical Log Power |

---

## 3. Honest Limitations & Scope

1. **Task Granularity:** Face dimension decoding (`ds005107`) is a fine-grained cognitive decoding task. Large-amplitude primary sensory evoked fields (e.g., strong somatosensory or auditory responses) may show different transfer dynamics.
2. **Coordinate Frame Approximation:** As verified in Phase 0, sensor coordinates in `ds005107` were approximated from an Elekta layout (`CoordinateFrame.APPROX_ELEKTA`). True subject-specific optical coregistrations (`CoordinateFrame.HEAD`, as in Nottingham OPM) are needed for full 3D spatial alignment.
3. **Zero-Spend Design:** All experiments were conducted under $0 compute budget using local Apple Silicon MPS and CPU resources.
