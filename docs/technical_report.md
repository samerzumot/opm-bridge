# OPM-Bridge: Technical Report & Benchmark Summary

**Project Title:** OPM-Bridge: Empirical Evaluation of Foundation Model Transfer to Optically Pumped Magnetometer MEG  
**Date:** 2026-11-04  
**Status:** Pre-Registration Complete — Empirical Audit & Diagnostic Findings  

---

## 1. Executive Summary

OPM-Bridge measures whether pre-trained SQUID MEG foundation models (specifically **BrainOmni**, NeurIPS 2025) transfer effectively to small optically pumped magnetometer (OPM) MEG datasets. 

Across our pre-registered evaluation on OpenNeuro `ds005107` (all 21 participants, 5,376 balanced epochs, 58 harmonized channels, Leave-One-Subject-Out cross-validation):

1. **Majority-Class Optimization Collapse in Heavy Backbones:** Both pre-trained BrainOmni (`pretrained_backbone_ft`) and matched random-initialization (`control_a_random_e2e`) collapsed to predicting a single majority class for 100% of test samples (`BACC = 50.00%`, `Macro-F1 = 33.33%`). The observed 0.00% difference is an optimization floor effect rather than transfer parity.
2. **Noise-Floor Cross-Subject Decoding:** On this single-run (`run-01`) cross-subject decoding task (`gender`), the compact specialist (`CompactMEGNet`) achieved `50.29% [49.17%, 51.42%]` at $B=4$. Because its 95% bootstrap confidence interval strictly spans `50.00%` chance, cross-subject decoding remains at chance across all evaluated models.
3. **Noise Reduction Efficacy:** Incorporating **Soft-HFC** (Homogeneous Field Correction with shrinkage $\alpha=0.25$) reduces external ambient magnetic interference while strictly preserving full covariance rank ($\text{rank} = C$), maintaining well-conditioned channel covariances across all 21 subjects.
4. **Phase 3 Geometry Ablations Audit:** Initial automated ablation passes executed only on position-blind baselines (`compact_megnet`), yielding identical no-op metrics. Spatial and orientation sensitivity requires evaluating `BrainOmniClassifier` under a warmstarted or linear-probe training regime.
5. **Phase 4 Triaxial Status:** Pure-Python Nottingham triaxial loader verified via unit tests; empirical validation on real Zenodo `11126593` data remains in progress (`[U] Unverified`).

---

## 2. Summary Results Table

Full cohort evaluation across Leave-One-Subject-Out (21 folds) with Soft-HFC enabled:

| Model Arm | $B=1$ (256 trials) BACC [95% CI] | $B=2$ (512 trials) BACC [95% CI] | $B=4$ (1,024 trials) BACC [95% CI] | Macro-F1 (All B) | Architectural Type |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **`pretrained_backbone_ft`** | 49.97% [49.80%, 50.13%] | 49.94% [49.83%, 50.00%] | 50.00% [50.00%, 50.00%] | **33.33%** | 13.5M Foundation Model |
| **`control_a_random_e2e`** | 50.00% [50.00%, 50.00%] | 50.00% [50.00%, 50.00%] | 50.00% [50.00%, 50.00%] | **33.33%** | Matched Random Init E2E |
| **`compact_megnet`** (+Soft-HFC) | 49.14% [47.55%, 50.65%] | 48.67% [47.51%, 49.90%] | 50.29% [49.17%, 51.42%] | 48.8%–50.2% | Compact CNN Specialist (~2,500 params) |
| **`classical_bandpower`** | 49.57% [48.67%, 50.29%] | 49.66% [48.83%, 50.46%] | 50.01% [49.20%, 50.81%] | 49.4%–49.9% | Classical Log Power + Logistic Reg |

---

## 3. Honest Limitations & Diagnostic Scope

1. **Optimization Dynamics at Low Data Regimes:** A 13.5M parameter transformer requires calibrated optimization (e.g., linear probing, learning rate warmup, cosine decay) when adapting to fewer than 1,000 samples. Direct end-to-end fine-tuning from a random head at fixed `lr=1e-4` leads directly to majority-class prediction collapse.
2. **Cognitive vs. Sensory SNR:** Face feature decoding (`gender`, `age`, `emotion`) in single-run OPM recordings has low trial-level SNR compared to primary sensory (auditory/somatosensory) evoked fields. Multi-run aggregation or stronger evoked components are needed for robust cross-subject classification.
3. **Sensor Coordinate Approximation in ds005107:** As verified in Phase 0, sensor coordinates in `ds005107` were approximated from an Elekta layout (`CoordinateFrame.APPROX_ELEKTA`). Subject-specific optical coregistrations (`CoordinateFrame.HEAD`, as in Nottingham triaxial OPM) provide genuine 3D coordinates.
4. **Zero-Spend Design:** All pipelines were developed and tested under $0 compute budget using local Apple Silicon MPS and CPU resources.
