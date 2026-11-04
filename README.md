# OPM-Bridge

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

*An open, reproducible benchmark evaluating whether pre-trained SQUID MEG foundation models ([BrainOmni](https://arxiv.org/abs/2505.18185), NeurIPS 2025) transfer to scarce optically pumped magnetometer (OPM-MEG) data—and diagnosing where and why they fail.*

---

## Key Verified Findings (`[V]`)

1. **Zero OPM Pre-Training Contamination (`[V]`):** Verified across all 22 pre-training datasets (Appendix D) and 11 downstream datasets (Appendix E) of BrainOmni (`arXiv:2505.18185v1`) that **zero OPM-MEG recordings** were included. All 656.82 hours of BrainOmni MEG pre-training used conventional SQUID systems (CTF-275 axial gradiometers, Elekta Neuromag-306, KIT/Yokogawa).
2. **Upstream Audit & PR Contribution (`[V]`):**
   - Submitted [OpenTSLab/BrainOmni#2](https://github.com/OpenTSLab/BrainOmni/pull/2) (reviewed and confirmed by maintainer `@sigureling`), fixing an evaluation metric bug in `evaluate.py`, adding a DeepSpeed import fallback for CPU/MPS/single-GPU users, and contributing MNE-native planar gradiometer detection (`FIFFV_COIL_VV_PLANAR_T1..T3`).
   - Identified that upstream `BrainTokenizer.tokenize()` is decorated with `@torch.no_grad()`, meaning upstream `pretrained=False` leaves `BrainTokenizer` randomly initialized *and* frozen. [`BrainOmniClassifier`](./src/opm_bridge/models/wrapper.py) explicitly separates `frozen` (linear probe), `backbone_ft`, `end_to_end_ft`, `control_a` (fully random), and `control_b` (pre-trained tokenizer + random backbone).
3. **Physical 6D Sensor Alignment & Soft-HFC Filtering (`[V]`):**
   - Implemented [`align_subjects_by_sensor_geometry`](./src/opm_bridge/preprocessing/sensor_mapping.py) to match physical 6D helmet coordinates (`[x, y, z, ox, oy, oz]`) across subjects after per-subject bad-channel rejection (yielding **50 exact common helmet sensors** across all 21 subjects of `ds005107`), preventing channel-index shifts across subjects.
   - Implemented **Soft-HFC** (`compute_soft_hfc_projector`, $P_{\text{soft}}(\alpha) = I_C - \alpha Q_R Q_R^T$ with $\alpha=0.25$), attenuating homogeneous environmental magnetic interference by 75% in power while strictly preserving full covariance rank ($\text{rank} = C$).

---

## Empirical Benchmark Results (`[V]`)

### 1. Phase 2: Cross-Subject Transfer vs. Matched Controls (`ds005107`, All 21 Subjects, 50 Aligned Channels)

Evaluated via Leave-One-Subject-Out (LOSO, 21 folds) on the post-stimulus visual evoked response window (`[+50ms, +550ms]`, Soft-HFC $\alpha=0.25$):

#### Task A: Cross-Subject Face `age` Decoding (21 Folds, 95% Subject-Level Bootstrap CI)

| Model Arm | $B=1$ (256 trials) | $B=2$ (512 trials) | $B=4$ (1,024 trials) | $B=10$ (2,560 trials) | $B=20$ (5,120 trials) | Macro-F1 ($B=20$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **`pretrained_linear_probe`** (BrainOmni Frozen) | 49.50% `[48.55, 50.49]` | 49.19% `[48.46, 50.01]` | 48.94% `[48.10, 49.86]` | 50.08% `[49.02, 51.06]` | 49.57% `[48.44, 50.73]` | 49.46% |
| **`control_a_linear_probe`** (Random Init Frozen) | 50.41% `[49.40, 51.38]` | 50.40% `[49.59, 51.24]` | 50.86% `[49.83, 51.83]` | 50.27% `[49.11, 51.42]` | 50.68% `[49.28, 52.06]` | 50.61% |
| **`control_b_linear_probe`** (PT Tokenizer + Rand BB) | 49.04% `[48.47, 49.59]` | 49.51% `[48.52, 50.45]` | 50.08% `[49.40, 50.82]` | 49.72% `[48.54, 50.92]` | 50.01% `[48.55, 51.45]` | 49.85% |
| **`compact_megnet`** (~2.5k Param CNN Specialist) | 49.25% `[48.18, 50.41]` | 50.80% `[49.97, 51.74]` | 50.70% `[49.92, 51.55]` | 50.73% `[49.81, 51.60]` | **51.85% `[50.33, 53.34]`** | **50.98%** |
| **`classical_bandpower`** (Welch PSD + LogReg) | 49.13% `[48.07, 50.12]` | 50.01% `[49.15, 50.78]` | 50.63% `[49.68, 51.56]` | 50.10% `[49.12, 51.08]` | 50.56% `[49.28, 51.79]` | 49.78% |
| **`pretrained_backbone_ft`** (5-ep Un-warmed E2E FT) | 49.97% `[49.80, 50.13]` | 49.94% `[49.83, 50.00]` | 50.00% `[50.00, 50.00]` | — | — | **33.33% (Collapsed)** |
| **RQ1 Verdict ($\text{PT Probe} - \text{Control A}$, $\text{SESOI}=3\%$)** | `-0.91%` (**`equivalent`**) | `-1.21%` (**`equivalent`**) | `-1.92%` (`inconclusive`) | `-0.19%` (**`equivalent`**) | `-1.11%` (**`equivalent`**) | — |

#### Task B: Cross-Subject Face `race` Decoding & Within-Subject `gender` Calibration (All 21 Subjects)
- **Cross-Subject `race` (`results/summary_aligned21_race.json`):** `pretrained_linear_probe` achieves **51.00% `[50.08%, 51.87%]`** at $B=2$ and **51.01% `[50.19%, 51.84%]`** at $B=10$, while `control_b_linear_probe` (Pre-trained `BrainTokenizer` + Random Backbone) reaches **51.45% `[50.37%, 52.55%]`** at $B=20$ (vs. `48.95%` for `control_a_linear_probe`). Across all three tasks (`age`, `race`, `gender`) at $B=20$, `control_b_linear_probe` matches or exceeds `pretrained_linear_probe`, indicating that any representation transfer originates in `BrainTokenizer` rather than the 13.5M spatiotemporal backbone.
- **Within-Subject Few-Shot Calibration on `gender` (`results/summary_within_subject_gender.json`):** On a fixed 64-trial per-subject holdout across all 21 subjects, `ClassicalBandPowerDecoder` scales significantly above chance to **51.80% `[50.38%, 53.20%]`** ($B=64$ trials) and **53.13% `[51.14%, 55.14%]`** ($B=128$ trials), whereas `pretrained_linear_probe` (`50.83% [49.05%, 52.61%]`) and `control_a_linear_probe` (`49.28% [47.08%, 51.54%]`) remain statistically equivalent ($\Delta = +1.55\%$, `inconclusive` / `equivalent`).

---

### 2. Phase 3: `BrainOmni` Sensor Geometry Ablations & The `MAG` vs. `GRAD` Embedding Discovery (`[V]`)

Evaluating `BrainOmni` across all 21 subjects (`50` aligned channels) under five geometry perturbations ([`summary_phase3_ablations.json`](./results/summary_phase3_ablations.json)):

| Geometry Mode | Sensor Type Token | Orientation Vector | Zero-Shot `BrainTokenizer` PCC `[95% CI]` | Relative PCC Change | Downstream `PT Probe` BACC ($B=20$, `age`) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`default`** | `MAG (1)` | True unit $\hat{\mathbf{o}} \in S^2$ | `0.2501 [0.2450, 0.2549]` | Baseline | 49.57% `[48.44%, 50.73%]` |
| **`opm_as_grad`** | **`GRAD (2)`** | True unit $\hat{\mathbf{o}} \in S^2$ | **`0.3819 [0.3779, 0.3859]`** | **+52.7% ($p < 0.0001$)** | 49.41% `[48.14%, 50.69%]` |
| **`zero_orientation`** | `MAG (1)` | $\mathbf{0} = [0, 0, 0]$ | `0.2359 [0.2295, 0.2421]` | -5.7% | 50.85% `[49.62%, 52.09%]` |
| **`flip_orientation`** | `MAG (1)` | $-\hat{\mathbf{o}}$ (sign negated) | `0.2340 [0.2274, 0.2402]` | -6.4% | 49.02% `[47.83%, 50.20%]` |
| **`scramble_channels`** | `MAG (1)` | Permuted across channels | `0.2330 [0.2307, 0.2354]` | -6.8% | 50.46% `[49.33%, 51.68%]` |

**Why does `opm_as_grad` (`GRAD=2`) boost zero-shot `BrainTokenizer` reconstruction PCC by +52.7% (`0.2501 -> 0.3819`)?**
In BrainOmni's 656.82-hour SQUID pre-training mix, CTF-275 datasets consist exclusively of axial gradiometers (`GRAD=2`, which share radial 3D orientations with radial OPMs) and Elekta-306 datasets contain 204 planar gradiometers (`GRAD=2`) vs. 102 magnetometers (`MAG=1`). Consequently, `GRAD (2)` outnumbered `MAG (1)` by $>4:1$ during pre-training, making the `GRAD (2)` embedding in `BrainSensorModule` substantially better conditioned for radial MEG waveform reconstruction than `MAG (1)`.

---

### 3. Phase 4: Real Nottingham Triaxial (`X/Y/Z`) QuSpin OPM-MEG Diagnostics (`Zenodo 11126593`, `[V]`)

Evaluated on real subject-coregistered (`CoordinateFrame.HEAD`) triaxial QuSpin OPM-MEG recordings streamed from Zenodo `11126593` (`Adults.zip`, `sub-101` with 176 good channels and `sub-102` with 170 good channels; 26 balanced tactile Braille somatosensory trials `Start_index` vs. `Start_pinky` per subject; [`summary_phase4_nottingham.json`](./results/summary_phase4_nottingham.json)):

| Subject | Configuration | Active Channels | Zero-Shot `BrainTokenizer` PCC | Braille 5-Fold BACC (`PT Probe`) | Braille 5-Fold BACC (`Classical PSD`) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **`sub-101`** | **Full Triaxial (`X/Y/Z`, Raw)** | 176 | `0.3229` | 38.33% | 51.67% |
| **`sub-101`** | **Full Triaxial (`X/Y/Z` + Soft-HFC)** | 176 | `0.2898` | 43.33% (`55.00%` as `GRAD`) | 51.67% |
| **`sub-101`** | **Radial-Only (`Z`-axis)** | **60** | **`0.4758` (+47.4%)** | 53.33% | **60.00%** |
| **`sub-102`** | **Full Triaxial (`X/Y/Z`, Raw)** | 170 | `0.3423` | 58.33% | 48.33% |
| **`sub-102`** | **Full Triaxial (`X/Y/Z` + Soft-HFC)** | 170 | `0.3038` | **68.33%** | 53.33% |
| **`sub-102`** | **Radial-Only (`Z`-axis)** | **58** | **`0.4592` (+34.1%)** | 40.00% | **65.00%** |

**The Triaxial $N_{\text{neuro}}=16$ Spatial Bottleneck:**
`BrainTokenizer` compresses all input channels into a fixed bottleneck of **$N_{\text{neuro}} = 16$ spatial latent tokens**. When fed 170–176 triaxial (`X/Y/Z`) OPM channels where 3 orthogonal axes share each 3D coordinate $(P_x, P_y, P_z)$ (`11:1` channel-to-token compression, never seen in SQUID pre-training), reconstruction PCC drops to **`0.3326` mean**. Restricting to the radial `Z` axis (`58–60` channels, `3.7:1` compression matching SQUID geometry) increases zero-shot reconstruction PCC to **`0.4675` mean (+40.6% relative)**.

See [docs/technical_report.md](./docs/technical_report.md) and [NOTES.md](./NOTES.md) for the complete technical report and audit log.

---

## Quickstart

```bash
# Install dependencies (Python 3.10–3.12 via uv)
uv sync --python 3.12 --extra dev

# Run unit and smoke test suite (7/7 tests)
uv run pytest -v

# Run full verified Phases 2, 3, and 4 benchmark suite
PYTHONPATH=. uv run python scripts/auto_orchestrate_remaining_phases.py
```
