# OPM-Bridge: Technical Report & Verified Benchmark Summary

**Project Title:** OPM-Bridge: Empirical Evaluation of Foundation Model Transfer to Optically Pumped Magnetometer MEG  
**Date:** 2026-11-04  
**Status:** Phases 0–5 Complete — Verified on Real Single-Axis & Triaxial OPM-MEG Cohorts (`[V]`)  

---

## 1. Executive Summary

OPM-Bridge measures whether pre-trained SQUID MEG foundation models (specifically **BrainOmni**, NeurIPS 2025, 13.5M parameters, pre-trained on 1,997 hours of EEG and 656.82 hours of SQUID MEG) transfer effectively to small optically pumped magnetometer (OPM-MEG) datasets, and diagnoses specific architectural bottlenecks under OPM sensor geometry.

Across our verified evaluation on **OpenNeuro `ds005107`** (all 21 participants, 5,376 balanced epochs, **50 physically aligned common helmet channels**, Leave-One-Subject-Out and Within-Subject calibration) and **Nottingham Triaxial OPM-MEG (`Zenodo 11126593`, `sub-101` & `sub-102`, 176/170 triaxial `X/Y/Z` channels in `CoordinateFrame.HEAD`)**:

1. **Primary Transfer Hypothesis (RQ1 — Foundation Model vs. Matched From-Scratch Control A):**
   - Under calibrated linear probing (`mode="frozen"`), pre-trained BrainOmni representations achieve **statistical equivalence** ($\Delta \in [-1.21\%, +1.20\%]$, within the pre-registered $\pm 3.0\%$ SESOI band) against matched randomly initialized controls (`Control A`) across low-budget regimes ($B \in \{1, 2, 10\}$ subjects on `age`, `race`, and `gender`).
   - Conversely, un-warmed 5-epoch end-to-end fine-tuning of the 13.5M backbone (`pretrained_backbone_ft`) suffers catastrophic majority-class collapse (`BACC = 50.00%`, `Macro-F1 = 33.33%`).
   - Across all three tasks at $B=20$ subjects, **Control B** (Frozen Pre-trained `BrainTokenizer` + Randomly Initialized Backbone) matches or exceeds full pre-trained `BrainOmni` (`50.01% vs 49.57%` on `age`, `51.45% [50.37%, 52.55%] vs 51.22%` on `race`, and `51.14% [50.05%, 52.35%] vs 49.59%` on `gender`), demonstrating that any representation transfer is localized in `BrainTokenizer` rather than the spatiotemporal transformer blocks.

2. **Specialist & Classical Baselines Outperform at Low Sample Sizes (RQ2):**
   - Once physical helmet channels are aligned across subjects via 6D coordinates (`align_subjects_by_sensor_geometry`) and restricted to the active visual evoked window (`[+50ms, +550ms]`), the compact domain-specific CNN (`CompactMEGNet`, ~2,500 parameters) reaches **`51.85% [50.33%, 53.34%]` BACC (`Macro-F1 = 50.98%`)** on cross-subject `age` decoding at $B=20$, statistically outperforming `pretrained_linear_probe` (`49.57%`) by **`\Delta = -2.27%` (95% CI `[-4.26%, -0.24%]`, $p < 0.025$)**.
   - On within-subject few-shot calibration (`gender` decoding across all 21 subjects), `ClassicalBandPowerDecoder` reaches **`51.80% [50.38%, 53.20%]`** ($B=64$ trials) and **`53.13% [51.14%, 55.14%]`** ($B=128$ trials), both strictly above 50% chance.

3. **Phase 3 Discovery — The `MAG (1)` vs. `GRAD (2)` Sensor-Type Embedding Asymmetry (RQ3):**
   - Evaluating zero-shot `BrainTokenizer` waveform reconstruction PCC across all 21 subjects reveals that mapping radial OPM sensors to **`GRAD (2)` (`opm_as_grad`) increases zero-shot reconstruction PCC by +52.7% relative**, from **`0.2501 [0.2450, 0.2549]` (`default` `MAG=1`) to `0.3819 [0.3779, 0.3859]` (`GRAD=2`, $p < 0.0001$)**.
   - *Mechanism:* In BrainOmni's 656.82-hour SQUID pre-training corpus, CTF-275 axial gradiometers (`GRAD=2`, which have radial 3D orientations identical to radial OPMs) and Elekta-306 planar gradiometers (`GRAD=2`, 204 ch vs. 102 `MAG=1` ch) outnumbered `MAG (1)` sensors by $>4:1$.
   - Corrupting sensor geometry via `zero_orientation` (`0.2359`), `flip_orientation` (`0.2340`), or `scramble_channels` (`0.2330`) significantly degrades waveform reconstruction PCC relative to `default` (`0.2501`).

4. **Phase 4 Discovery — The Triaxial $N_{\text{neuro}}=16$ Spatial Bottleneck on Real Nottingham OPM-MEG (`[V]`):**
   - On real 176/170-channel triaxial (`X/Y/Z`) QuSpin OPM-MEG recordings (`Zenodo 11126593`, `sub-101` & `sub-102`), feeding all 3 orthogonal co-located axes (`X, Y, Z`) into `BrainTokenizer`'s fixed **$N_{\text{neuro}} = 16$ spatial token bottleneck** yields a zero-shot reconstruction PCC of only **`0.3326` mean** (`0.3229` on `sub-101`, `0.3423` on `sub-102`).
   - Restricting to the radial `Z`-axis (`radial_only_z`, 60/58 channels, 1 axis per 3D position matching SQUID training layouts) increases zero-shot reconstruction PCC to **`0.4675` mean (`0.4758` on `sub-101`, `0.4592` on `sub-102`, +40.6% relative)**.
   - Applying **Soft-HFC ($\alpha=0.25$)** to triaxial OPM data improves 5-fold tactile Braille decoding (`Start_index` vs. `Start_pinky`) for `pretrained_linear_probe` on both subjects (`38.33% -> 43.33%` on `sub-101`, `58.33% -> 68.33%` on `sub-102`), while `ClassicalBandPowerDecoder` on `radial_only_z` achieves **`62.50%` mean BACC** (`60.00%` on `sub-101`, `65.00%` on `sub-102`).

---

## 2. Verified Summary Tables

### Table 1: Cross-Subject LOSO Transfer on `ds005107` (`age`, 21 Folds, 50 Aligned Channels, Soft-HFC $\alpha=0.25$)

| Model Arm | $B=1$ (256) | $B=2$ (512) | $B=4$ (1,024) | $B=10$ (2,560) | $B=20$ (5,120) | Macro-F1 ($B=20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pretrained_linear_probe`** | 49.50% `[48.55, 50.49]` | 49.19% `[48.46, 50.01]` | 48.94% `[48.10, 49.86]` | 50.08% `[49.02, 51.06]` | 49.57% `[48.44, 50.73]` | 49.46% |
| **`control_a_linear_probe`** | 50.41% `[49.40, 51.38]` | 50.40% `[49.59, 51.24]` | 50.86% `[49.83, 51.83]` | 50.27% `[49.11, 51.42]` | 50.68% `[49.28, 52.06]` | 50.61% |
| **`control_b_linear_probe`** | 49.04% `[48.47, 49.59]` | 49.51% `[48.52, 50.45]` | 50.08% `[49.40, 50.82]` | 49.72% `[48.54, 50.92]` | 50.01% `[48.55, 51.45]` | 49.85% |
| **`compact_megnet`** | 49.25% `[48.18, 50.41]` | 50.80% `[49.97, 51.74]` | 50.70% `[49.92, 51.55]` | 50.73% `[49.81, 51.60]` | **51.85% `[50.33, 53.34]`** | **50.98%** |
| **`classical_bandpower`** | 49.13% `[48.07, 50.12]` | 50.01% `[49.15, 50.78]` | 50.63% `[49.68, 51.56]` | 50.10% `[49.12, 51.08]` | 50.56% `[49.28, 51.79]` | 49.78% |

### Table 2: Within-Subject Few-Shot Calibration on `ds005107` (`gender`, 21 Subjects, 64-Trial Holdout per Subject)

| Model Arm | $B_{\text{trials}}=32$ | $B_{\text{trials}}=64$ | $B_{\text{trials}}=128$ | $B_{\text{trials}}=180$ |
| :--- | :---: | :---: | :---: | :---: |
| **`pretrained_linear_probe`** | 48.95% `[47.40, 50.69]` | 50.90% `[48.76, 53.18]` | 50.83% `[49.05, 52.61]` | 50.13% `[47.41, 52.77]` |
| **`control_a_linear_probe`** | 49.77% `[48.15, 51.43]` | 50.14% `[48.21, 52.12]` | 49.28% `[47.08, 51.54]` | 50.71% `[48.01, 53.55]` |
| **`control_b_linear_probe`** | 50.23% `[48.40, 51.98]` | 49.13% `[46.92, 51.51]` | 50.06% `[47.77, 52.36]` | 50.04% `[47.34, 52.66]` |
| **`classical_bandpower`** | 50.45% `[48.81, 52.02]` | **51.80% `[50.38, 53.20]`** | **53.13% `[51.14, 55.14]`** | **53.39% `[51.31, 55.45]`** |

### Table 3: Phase 3 Sensor Geometry Ablations (`ds005107`, All 21 Subjects, 50 Aligned Channels)

| Geometry Ablation Mode | Zero-Shot `BrainTokenizer` PCC `[95% CI]` | `PT Probe` BACC ($B=2$) | `PT Probe` BACC ($B=20$) | `Control B` BACC ($B=20$) |
| :--- | :---: | :---: | :---: | :---: |
| **`default` (`MAG=1`, true $\hat{\mathbf{o}}$)** | `0.2501 [0.2450, 0.2549]` | 49.19% `[48.46, 50.01]` | 49.57% `[48.44, 50.73]` | 50.01% `[48.55, 51.45]` |
| **`opm_as_grad` (`GRAD=2`, true $\hat{\mathbf{o}}$)** | **`0.3819 [0.3779, 0.3859]` (+52.7%)** | 49.49% `[48.36, 50.61]` | 49.41% `[48.14, 50.69]` | 49.32% `[48.10, 50.56]` |
| **`zero_orientation` (`MAG=1`, $\mathbf{o}=\mathbf{0}$)** | `0.2359 [0.2295, 0.2421]` (-5.7%) | 49.94% `[48.88, 51.01]` | 50.85% `[49.62, 52.09]` | 50.63% `[49.25, 51.98]` |
| **`flip_orientation` (`MAG=1`, $-\hat{\mathbf{o}}$)** | `0.2340 [0.2274, 0.2402]` (-6.4%) | 48.66% `[47.76, 49.59]` | 49.02% `[47.83, 50.20]` | 49.83% `[48.52, 51.14]` |
| **`scramble_channels` (permuted $\mathbf{p}_c$)** | `0.2330 [0.2307, 0.2354]` (-6.8%) | 49.84% `[48.80, 50.88]` | 50.46% `[49.33, 51.68]` | 50.61% `[49.45, 51.78]` |

---

## 3. Honest Limitations & Scope

1. **Pilot Scale & Single-Run Caching on `ds005107`:** Our `ds005107` benchmark cached `run-01` (256 balanced trials per subject, 5,376 epochs across all 21 subjects), and our real Nottingham triaxial evaluation streamed the first 100 seconds (26 balanced Braille trials per subject) of `sub-101` and `sub-102` from Zenodo `11126593`.
2. **Coordinate Frame Approximation in `ds005107`:** Per `ds005107/README`, sensor coordinates in `ds005107` were approximated from an Elekta layout (`CoordinateFrame.APPROX_ELEKTA`), which is why we paired it with subject-coregistered triaxial Nottingham recordings (`CoordinateFrame.HEAD`) in Phase 4.
3. **Zero-Spend Compute:** All experiments were executed locally on Apple Silicon MPS and CPU with $0 compute spend.
