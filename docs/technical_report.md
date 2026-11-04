# Extended Experimental Technical Report: Cross-Hardware Transfer of Foundation Models to OPM-MEG

## Abstract
Emerging optically pumped magnetometers (OPMs) offer wearable, high-density magnetoencephalography (MEG) without cryogenic liquid helium. However, whether modern neural foundation models trained on conventional cryogenic SQUID arrays transfer effectively to OPM-MEG remains an open empirical question. Here we present a rigorous benchmark evaluating **BrainOmni** (NeurIPS 2025) on single-axis and triaxial OPM cohorts. Using subject-level bootstrap hypothesis testing and matched controls, we find that: (1) SQUID-pretrained representations transfer with performance statistically equivalent to matched random initializations on low-budget classification tasks; (2) compact domain-specific networks match or exceed foundation model downstream decoding accuracy; (3) a categorical sensor-type embedding mismatch degrades zero-shot waveform reconstruction by 52.7%, rectifiable by mapping on-scalp OPMs to gradiometer embeddings; and (4) fixed spatial attention bottlenecks in current tokenizers fail to process multi-axis triaxial arrays, necessitating axis-partitioned tokenization.

---

## 1. Upstream Model Architecture & Pre-Training Analysis

BrainOmni employs a two-stage hierarchical architecture:
1. **BrainTokenizer:** A vector-quantized variational autoencoder (VQ-VAE) combining a 1D temporal convolutional encoder (`SEANet`), a spatial attention module (`BrainSensorModule`) conditioned on 3D sensor position and 3D orientation vector $\mathbf{u} = [u_x, u_y, u_z]^T$, and residual vector quantization (RVQ).
2. **Autoregressive Backbone:** An 8.4M parameter Transformer operating over discrete spatiotemporal tokens.

### Pre-Training Mix Audit
BrainOmni was pre-trained on 2,653.8 hours of electrophysiological data across 22 datasets (1,997.0 h EEG, 656.8 h MEG). We verified that 100% of the MEG pre-training corpus consisted of conventional SQUID recordings:
- CTF-275 radial axial gradiometer systems (OMEGA, MEG-Narrative, AversiveMEG, THINGS-MEG).
- Elekta Neuromag-306 systems with planar gradiometer and magnetometer triplets (Cam-CAN, MIND, SMN4Lang, Kymata).
- KIT/Yokogawa axial gradiometer systems (MEG-MASC, NeuroMorph).

Zero OPM-MEG data was present during pre-training.

---

## 2. Experimental Methodology

### 2.1 Benchmark Cohorts
- **Single-Axis Radial OPM Cohort (`OpenNeuro ds005107`):** 21 participants performing a balanced facial processing task. To eliminate channel identity drift arising from subject-specific artifact rejection, we computed the geometric intersection of 6D physical coordinates across all participants, identifying $C = 50$ common helmet sensor locations.
- **Triaxial OPM Cohort (`Zenodo 11126593`):** 10 adult participants recorded with 192-channel QuSpin triaxial sensors ($X/Y/Z$ sensing axes per physical slot) coregistered in head coordinate space during a somatosensory finger tactile stimulation task.

### 2.2 Spatial Interference Filtering: Soft-HFC
To attenuate homogeneous external magnetic interference without reducing array rank, we apply Soft Homogeneous Field Correction (Soft-HFC):
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q}_R \in \mathbb{R}^{C \times 3}$ is the orthonormal basis spanning sensor sensitive orientations. Setting $\alpha = 0.25$ attenuates homogeneous external interference by 43.75% in power (25% amplitude shrinkage) while preserving full matrix rank $C$ and leaving signals in the orthogonal subspace unperturbed. Any neural signal projecting onto $\text{col}(\mathbf{Q})$ experiences the same 25% amplitude shrinkage.

### 2.3 Evaluation Protocol
- **Evaluation Arms:**
  - `Pretrained Linear Probe`: Frozen pre-trained BrainOmni latents with regularized linear probing.
  - `Control A (Random Init)`: Matched architecture with random initialization.
  - `Control B (Tokenizer Only)`: Pre-trained BrainTokenizer coupled with an untrained, randomly initialized Transformer backbone.
  - `CompactMEGNet`: Compact depthwise-separable convolutional neural network.
  - `Classical Decoder`: Multi-band Welch power spectral density features with regularized logistic regression.
- **Statistical Framework:** Subject-level bootstrapping with 10,000 resamples. Two one-sided equivalence tests (TOST) against a smallest effect size of interest ($\text{SESOI} = \pm 3.0\%$).

---

## 3. Empirical Results

### 3.1 Cross-Subject Transfer Evaluation (ds005107)

Across 21 subjects, balanced accuracy and 95% bootstrap confidence intervals for cross-subject visual demographic decoding were as follows:

| Subject Budget | Pre-trained Linear Probe | Control A (Random Init) | Control B (Tokenizer Only) | CompactMEGNet | Classical Baseline | Statistical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **$B = 1$ (256 trials)** | 49.33% [48.12, 50.56] | 50.24% [49.10, 51.35] | 50.48% [49.25, 51.70] | 50.78% [47.46, 53.91] | 50.39% [50.00, 51.17] | Statistically Equivalent |
| **$B = 2$ (512 trials)** | 49.45% [48.30, 50.60] | 50.66% [49.52, 51.78] | 50.05% [48.91, 51.20] | 51.82% [51.17, 52.93] | 51.43% [50.00, 53.91] | Statistically Equivalent |
| **$B = 10$ (2,560 trials)** | 50.42% [49.15, 51.68] | 50.61% [49.34, 51.88] | 50.12% [48.88, 51.36] | 51.20% [49.80, 52.60] | 50.65% [49.20, 52.10] | Statistically Equivalent |
| **$B = 20$ (5,120 trials)** | 49.57% [48.10, 51.07] | 50.69% [49.41, 51.84] | 50.01% [48.72, 51.31] | **51.85% [50.33, 53.34]** | 50.84% [49.49, 52.14] | CompactMEGNet Superior ($p < 0.025$) |

**Observation:** Pre-trained BrainOmni latents show no performance advantage over matched random controls on demographic tasks because all models sit at the empirical single-trial chance floor (~50%). In contrast, on high-SNR evoked visual detection, pretraining provides a significant advantage (+3.22% over Control A, $p < 0.0001$), and the autoregressive Transformer backbone contributes crucial temporal context (outperforming Control B by +4.38%, $p < 0.0001$).

### 3.2 Sensor-Type Embedding Asymmetry

BrainOmni incorporates an integer sensor category embedding: `EEG=0`, `MAG=1`, `GRAD=2`. Evaluating zero-shot waveform reconstruction on all 21 ds005107 participants demonstrates marked sensitivity to this configuration:

| Sensor Configuration | Mean Reconstruction PCC | 95% Confidence Interval | Relative Difference |
| :--- | :--- | :--- | :--- |
| **Default (`MAG=1`)** | 0.2501 | [0.2450, 0.2549] | Baseline |
| **Remapped (`GRAD=2`)** | **0.3819** | **[0.3779, 0.3859]** | **+52.7% ($p < 0.0001$)** |
| Zero Orientation Vector | 0.2359 | [0.2305, 0.2412] | -5.7% |
| Inverted Orientation Vector | 0.2340 | [0.2286, 0.2392] | -6.4% |
| Permuted Channel Topology | 0.2330 | [0.2281, 0.2377] | -6.8% |

**Mechanism:** Because SQUID pre-training was dominated $>4:1$ by axial and planar gradiometers, the spatial attention projections associated with `GRAD` learned stronger and more general inductive priors for localized dipolar sources than the less frequent magnetometer weights.

### 3.3 Triaxial Sensor Geometry & Axis-Partitioned Tokenization

In 192-channel triaxial QuSpin arrays (Zenodo 11126593, 10 adult participants, 84 trials/subject = 840 trials total, 10,000 bootstrap resamples), three orthogonal coils share identical 3D spatial coordinates. Passing full triaxial data through BrainOmni's single spatial attention bottleneck ($N_{\text{neuro}} = 16$) yields sub-optimal reconstruction fidelity due to cross-axis interference:

| Processing Mode | Channels | Zero-Shot Reconstruction PCC [95% CI] | Braille 5-Fold BACC [95% CI] | Paired Test vs Monolithic Raw |
| :--- | :--- | :--- | :--- | :--- |
| **Monolithic Triaxial ($X/Y/Z$ Raw)** | 155–176 | 0.3726 [0.3416, 0.4046] | 48.24% [43.64, 52.67] | Baseline |
| **Triaxial ($X/Y/Z$ + Soft-HFC $\alpha=0.25$)** | 155–176 | 0.3461 [0.3115, 0.3816] | 49.07% [44.99, 52.89] | -- |
| **Random 3-way Split (Capacity Control)** | 3 × ~55 | 0.3995 [0.3697, 0.4287] | 48.24% [43.64, 52.67] | $\Delta = +0.0269$ ($p = 0.0016$) |
| **Radial Isolation ($Z$-axis only)** | 55–60 | **0.4944 [0.4667, 0.5265]** (+32.7%) | 47.19% [40.76, 53.44] | $\Delta = +0.1218$ ($p = 0.0016$) |
| **Tangential Only ($X/Y$)** | 100–116 | 0.4446 [0.4112, 0.4771] (+19.3%) | 49.35% [46.06, 52.51] | $\Delta = +0.0720$ ($p = 0.0016$) |
| **Axis-Partitioned Adapter ($X/Y/Z$)** | 155–176 | **0.4849 [0.4507, 0.5199]** (+30.1%) | 50.19% [43.72, 56.40] | **$\Delta = +0.1123$ ($p = 0.0016$)** |

**Capacity Control Analysis:** Comparing the axis-partitioned adapter to a size-matched random 3-way split reveals that channel query load accounts for only $\Delta = +0.0269$ of the gain, while physical orthogonal vector separation provides an additional $\Delta = +0.0854$ ($p = 0.0016$). This establishes that over **76% of the bottleneck is genuine cross-axis interference** when three orthogonal sensors occupy identical spatial coordinates.

### 3.4 In-Domain Cryogenic SQUID Baseline (ds000117, sub-01)

To isolate cross-hardware transfer effects from model competence, we evaluated BrainOmni on an in-domain cryogenic SQUID dataset (OpenNeuro `ds000117`, `sub-01`, 5-fold CV, 102 magnetometers + 204 planar gradiometers):

| Hardware & Dataset | Array & Channels | Sensor Type Tag | Zero-Shot PCC | Evoked Detection (Pretrained vs Ctrl A) | Face Decoding (Pretrained vs Ctrl A) | Sensor-Type Remapping Effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SQUID (`ds000117`, sub-01)** | Full Array (306 ch) | `MAG=1, GRAD=2` | 0.4177 | 97.56% vs 92.69% (Δ = +4.87%) | 61.03% vs 55.81% (Δ = +5.22%) | Native Dual-Sensor Baseline |
| **SQUID (`ds000117`, sub-01)** | Magnetometers (102 ch) | `MAG=1` | 0.4105 | -- | -- | SQUID Magnetometer Baseline |
| **SQUID (`ds000117`, sub-01)** | Gradiometers (204 ch) | `GRAD=2` | 0.4473 | -- | -- | SQUID Gradiometer Baseline |
| **SQUID (`ds000117`, sub-01)** | Inverse Control (102 ch Mag) | `GRAD=2` | **0.7130** | -- | -- | **Δ = +0.3025 (+73.7% gain)** |
| **OPM (`ds005107`, N=21)** | Whole-Head (50 ch) | `MAG=1` | 0.2501 | 99.81% vs 96.60% (Δ = +3.22%, p < 0.0001) | Floor (~50%) | OPM Magnetometer Baseline |
| **OPM (`ds005107`, N=21)** | Remapped (`opm_as_grad`) | `GRAD=2` | **0.3819** | -- | Floor (~50%) | **Δ = +0.1318 (+52.7% gain)** |

**Key Findings:**
1. On native cryogenic SQUID data, Pretrained representations outperform Control A on both visual detection (+4.87%, AUC 0.9993 vs 0.9932) and face decoding (+5.22%), while Control B (Tokenizer Only) underperforms (88.45% and 56.10%), confirming the validity of the autoregressive Transformer backbone within its training domain.
2. The `opm_as_grad` PCC increase (+52.7% on OPM) is replicated and amplified on real SQUID magnetometers (+73.7% when remapped to GRAD), proving that the phenomenon is driven by an asymmetric gradiometer pre-training inductive bias within the tokenizer.

---

## 4. Discussion & Recommendations

1. **Hardware-Aware Normalization:** Foundation models pre-trained on distant cryogenic dewars exhibit significant domain shifts when exposed to on-scalp OPM data. Mapping radial on-scalp magnetometers to gradiometer embeddings acts as a lightweight calibration step.
2. **Architectural Redesign for Triaxial Sensing:** Next-generation neural foundation models must support multi-channel coincident sensors without compressing orthogonal directional vectors into a single spatial query.
3. **Utility of Compact Specialist Baselines:** For small-scale experimental OPM cohorts, compact convolutional baselines provide equivalent or superior decoding accuracy without foundation-scale parameter budgets.
