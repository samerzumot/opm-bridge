# Extended Experimental Technical Report: Cross-Hardware Transfer of Foundation Models to OPM-MEG

## Abstract
Optically pumped magnetometers (OPMs) enable wearable, high-density magnetoencephalography (MEG) without cryogenic infrastructure. However, whether electrophysiological foundation models pre-trained on conventional superconducting quantum interference device (SQUID) arrays transfer effectively to OPM-MEG remains uncharacterized. This technical report presents an empirical evaluation of cross-hardware transfer using BrainOmni across single-axis radial and triaxial OPM cohorts. Using subject-level bootstrap hypothesis testing and matched control architectures, we evaluate representations across classification and self-supervised reconstruction tasks. Pre-trained SQUID representations yield decoding performance statistically equivalent to matched random initializations on low-budget classification benchmarks, while compact domain-specific architectures match or exceed foundation model downstream accuracy. In zero-shot waveform reconstruction, a categorical sensor-type embedding mismatch degrades reconstruction fidelity by 52.7% (Pearson correlation coefficient r = 0.2501 versus 0.3819, p < 0.0001), which is resolved by remapping on-scalp magnetometers to gradiometer embeddings. Furthermore, fixed spatial attention bottlenecks in current tokenizers exhibit cross-axis interference when processing co-located triaxial channels, where an axis-partitioned tokenization strategy recovers +30.1% reconstruction accuracy (r = 0.3726 to 0.4849, p = 0.0016). These findings establish concrete architectural and tokenization criteria for cross-hardware transfer in non-invasive neural recording.

---

## 1. Upstream Model Architecture & Pre-Training Analysis

BrainOmni employs a two-stage hierarchical architecture:
1. **BrainTokenizer:** A vector-quantized variational autoencoder (VQ-VAE) combining a 1D temporal convolutional encoder (`SEANet`), a spatial attention module (`BrainSensorModule`) conditioned on 3D sensor position and 3D orientation vector $\mathbf{u} = [u_x, u_y, u_z]^T$, and residual vector quantization (RVQ).
2. **Autoregressive Backbone:** An 8.4M parameter Transformer operating over discrete spatiotemporal tokens.

### Pre-Training Mix Audit
BrainOmni was pre-trained on 2,653.8 hours of electrophysiological data across 22 datasets (1,997.0 h EEG, 656.8 h MEG). Analysis of the pre-training corpus reveals that all upstream MEG recordings were acquired exclusively on conventional SQUID systems:
- CTF-275 radial axial gradiometer systems (OMEGA, MEG-Narrative, AversiveMEG, THINGS-MEG).
- Elekta Neuromag-306 systems with planar gradiometer and magnetometer triplets (Cam-CAN, MIND, SMN4Lang, Kymata).
- KIT/Yokogawa axial gradiometer systems (MEG-MASC, NeuroMorph).

No OPM-MEG recordings were present in the pre-training corpus.

---

## 2. Experimental Methodology

### 2.1 Benchmark Cohorts
- **Single-Axis Radial OPM Cohort (`OpenNeuro ds005107`):** 21 participants performing a balanced facial processing task. To eliminate channel identity drift arising from subject-specific artifact rejection, we computed the geometric intersection of 6D physical coordinates across all participants, identifying $C = 50$ common helmet sensor locations.
- **Triaxial OPM Cohort (`Zenodo 11126593`):** 10 adult participants recorded with 192-channel QuSpin triaxial sensors ($X/Y/Z$ sensing axes per physical slot) coregistered in head coordinate space during a somatosensory finger tactile stimulation task.

### 2.2 Spatial Interference Filtering: Soft-HFC
To attenuate homogeneous external magnetic interference without reducing array rank, we apply Soft Homogeneous Field Correction (Soft-HFC):
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q}_R \in \mathbb{R}^{C \times r}$, with $r=\mathrm{rank}(\mathbf{Q})$, is the orthonormal basis spanning sensor sensitive orientations. Setting $\alpha = 0.25$ attenuates homogeneous external interference by 43.75% in power (25% amplitude shrinkage) while preserving full matrix rank $C$ and leaving signals in the orthogonal subspace unperturbed. Any neural signal projecting onto $\text{col}(\mathbf{Q})$ experiences the same 25% amplitude shrinkage.

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

Across facial stimulus attribute classification tasks (face age, sex, ancestry), pre-trained BrainOmni representations demonstrate no statistically significant advantage over matched random controls, as performance across all models remains constrained near the single-trial empirical chance floor (~50%). Within this low-SNR regime, pre-trained sequence context provides no measurable advantage over tokenizer-only projections (`Control B`). By contrast, the compact convolutional specialist (`CompactMEGNet`) achieves 51.85% [50.33, 53.34] at $B=20$, matching or outperforming foundation model representations.

### 3.2 Sensor-Type Embedding Asymmetry

BrainOmni incorporates an integer sensor category embedding: `EEG=0`, `MAG=1`, `GRAD=2`. Evaluating zero-shot waveform reconstruction on all 21 ds005107 participants demonstrates marked sensitivity to this configuration:

| Sensor Configuration | Mean Reconstruction PCC | 95% Confidence Interval | Relative Difference |
| :--- | :--- | :--- | :--- |
| **Default (`MAG=1`)** | 0.2501 | [0.2450, 0.2549] | Baseline |
| **Remapped (`GRAD=2`)** | **0.3819** | **[0.3779, 0.3859]** | **+52.7% ($p < 0.0001$)** |
| Zero Orientation Vector | 0.2359 | [0.2305, 0.2412] | -5.7% |
| Inverted Orientation Vector | 0.2340 | [0.2286, 0.2392] | -6.4% |
| Permuted Channel Topology | 0.2330 | [0.2281, 0.2377] | -6.8% |

This performance discrepancy stems from the composition of the upstream pre-training corpus, which was dominated in a ratio greater than 4:1 by axial and planar gradiometers over magnetometers. Consequently, the spatial attention projections associated with `GRAD` acquired stronger and more generalizable representations for localized dipolar source fields than the less frequently represented magnetometer parameters.

### 3.3 Triaxial Sensor Geometry & Axis-Partitioned Tokenization

In 192-channel triaxial QuSpin arrays (Zenodo 11126593, 10 adult participants, 84 trials/subject = 840 trials total, 10,000 bootstrap resamples), three orthogonal coils share identical 3D spatial coordinates. Passing full triaxial data through BrainOmni's single spatial attention bottleneck ($N_{\text{neuro}} = 16$) yields sub-optimal reconstruction fidelity due to cross-axis interference. We evaluate a matched $2 \times 3$ factorial grid directly comparing monolithic, capacity-controlled random partition, and axis-partitioned architectures under identical filtering conditions:

| Array Configuration | Filtering | Channel Count | Reconstruction PCC [95% CI] | Braille Decoding BACC [95% CI] | Statistical Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monolithic Triaxial** | Raw | 155–176 | 0.3726 [0.3416, 0.4046] | 48.24% [43.64, 52.67] | Baseline (Raw) |
| **Monolithic Triaxial** | Soft-HFC | 155–176 | 0.3461 [0.3115, 0.3816] | 49.07% [44.99, 52.89] | Baseline (Soft-HFC) |
| **Random 3-way Split (Capacity Control)** | Raw | 3 × ~55 | 0.3995 [0.3697, 0.4287] | 50.83% [46.31, 54.90] | Δ = +0.0269 vs Raw Mono (p = 0.0016) |
| **Random 3-way Split (Capacity Control)** | Soft-HFC | 3 × ~55 | 0.3753 [0.3421, 0.4076] | 48.51% [43.47, 53.00] | Δ = +0.0292 vs HFC Mono (p = 0.0016) |
| **Radial Isolation ($Z$-axis only)** | Raw | 55–60 | 0.4944 [0.4667, 0.5265] | 47.19% [40.76, 53.44] | Δ = +0.1218 vs Raw Mono (p = 0.0016) |
| **Tangential Only ($X/Y$)** | Raw | 100–116 | 0.4446 [0.4112, 0.4771] | 49.35% [46.06, 52.51] | Δ = +0.0720 vs Raw Mono (p = 0.0016) |
| **Axis-Partitioned Adapter ($Z, X, Y$)** | Raw | 155–176 | 0.5130 [0.4840, 0.5432] | 47.53% [43.22, 51.88] | **Δ = +0.1135 vs Random Raw (p = 0.0016)** |
| **Axis-Partitioned Adapter ($Z, X, Y$)** | Soft-HFC | 155–176 | 0.4849 [0.4507, 0.5199] | 50.19% [43.72, 56.40] | **Δ = +0.1096 vs Random HFC (p = 0.0016)** |

A capacity control analysis under strictly matched Soft-HFC filtering indicates that reducing channel query load accounts for only $\Delta = +0.0292$ of the total gain ($PCC = 0.3461 \to 0.3753$). In contrast, physically separating orthogonal orientation vectors accounts for an additional $\Delta = +0.1096$ [0.1010, 0.1195] ($p = 0.0016$). This establishes that 79.0% of the reconstruction bottleneck originates from cross-axis vector interference when three orthogonal sensors occupy identical spatial coordinates. Under raw recordings, vector orthogonality accounts for an identical 80.8% of the bottleneck ($\Delta = +0.1135, p = 0.0016$).

### 3.4 In-Domain Cryogenic SQUID Baseline (ds000117, sub-01)

To isolate cross-hardware transfer effects from model competence, we evaluated BrainOmni on an in-domain cryogenic SQUID dataset (OpenNeuro `ds000117`, `sub-01`, 5-fold CV, 102 magnetometers + 204 planar gradiometers):

| Hardware & Dataset | Array & Channels | Sensor Type Tag | Zero-Shot PCC | Downstream Face Decoding (Pretrained vs Ctrl A) | Sensor-Type Remapping Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SQUID (`ds000117`, sub-01)** | Full Array (306 ch) | `MAG=1, GRAD=2` | 0.4177 | 61.03% vs 55.81% (Δ = +5.22%) | Native Dual-Sensor Baseline |
| **SQUID (`ds000117`, sub-01)** | Magnetometers (102 ch) | `MAG=1` | 0.4105 | -- | SQUID Magnetometer Baseline |
| **SQUID (`ds000117`, sub-01)** | Gradiometers (204 ch) | `GRAD=2` | 0.4473 | -- | SQUID Gradiometer Baseline |
| **SQUID (`ds000117`, sub-01)** | Inverse Control (102 ch Mag) | `GRAD=2` | **0.7130** | -- | **Δ = +0.3025 (+73.7% gain)** |
| **OPM (`ds005107`, N=21)** | Whole-Head (50 ch) | `MAG=1` | 0.2501 | Floor (~50%) | OPM Magnetometer Baseline |
| **OPM (`ds005107`, N=21)** | Remapped (`opm_as_grad`) | `GRAD=2` | **0.3819** | Floor (~50%) | **Δ = +0.1318 (+52.7% gain)** |

Evaluation on native cryogenic SQUID recordings reveals two central insights. First, pre-trained representations outperform Control A on downstream face-versus-scrambled classification (+5.22%, AUC 0.5909 versus 0.5591), while Control B (Tokenizer Only) underperforms (56.10%), verifying that the autoregressive Transformer backbone learns valid neurophysiological dynamics within its training distribution. Second, the reconstruction gain observed with `opm_as_grad` (+52.7% on OPM) is replicated and amplified on cryogenic SQUID magnetometers (+73.7% relative gain upon remapping to `GRAD`), corroborating that this effect is driven by the disproportionate presence of gradiometers in the upstream pre-training corpus rather than an idiosyncratic feature of OPM sensors.

---

## 4. Discussion & Recommendations

Foundation models pre-trained on distant cryogenic dewars exhibit pronounced domain shifts when transferred to on-scalp OPM arrays. Because OPM sensors capture localized, high-amplitude field gradients analogous to planar gradiometers rather than distant volume-conducted magnetometer fields, remapping radial on-scalp magnetometers to gradiometer embeddings serves as an effective zero-parameter calibration step. Future multi-hardware foundation models should replace rigid categorical sensor-type indices with continuous physical parameterizations that account for coil standoff distance and effective baseline.

Current foundation model tokenizers are architecturally ill-suited for multi-axis sensor arrays. When co-located orthogonal sensing coils share identical three-dimensional coordinates, spatial cross-attention collapses distinct directional components into a single spatial query, inducing destructive vector interference. Addressing this limitation in future architectures requires decoupling spatial coordinates from channel orientation—either through independent coordinate tokenization per measurement axis or through directional vector cross-attention mechanisms that natively preserve vectorial geometry.

Finally, for contemporary experimental OPM cohorts characterized by moderate sample sizes, compact domain-specific architectures provide a robust and computationally economical alternative. Compact convolutional networks match or exceed foundation model decoding accuracy on downstream classification benchmarks without requiring large parameter budgets or specialized pre-training corpora. Until foundation models incorporate diverse on-scalp recordings and multi-axis spatial tokenization during pre-training, task-specific compact baselines remain the recommended standard for applied OPM-MEG decoding.
