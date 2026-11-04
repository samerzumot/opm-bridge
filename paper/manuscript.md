# Cross-Hardware Generalization of Electrophysiological Foundation Models: An Empirical Evaluation on Optically Pumped Magnetometers

**Target Venue:** *NeurIPS NeuroAI Workshop*

**Authors:** Samer Zumot, Marzieh Zare  
**Correspondence:** `srzumot@gmail.com`  
**Repository:** `https://github.com/samerzumot/opm-bridge`

---

## Abstract

Self-supervised electrophysiological foundation models aim to establish unified neural representations across heterogeneous recording devices and sensor configurations. A prominent architecture is BrainOmni (NeurIPS 2025), which employs a spatial-temporal tokenizer conditioned on 3D sensor coordinates, orientations, and discrete sensor modalities to process multi-channel continuous waveforms. However, BrainOmni was pre-trained and evaluated exclusively on conventional cryogenic superconducting quantum interference device (SQUID) arrays and scalp electroencephalography (EEG). Emerging optically pumped magnetometers (OPMs) operate without cryogenics and mount directly on the scalp, introducing substantial physical and geometric domain shifts: close scalp proximity (~5 mm vs. 3–6 cm), localized dipolar topographies, irregular channel montages, and triaxial ($X/Y/Z$) multi-axis sensing. Here, we present an empirical benchmark evaluating cross-hardware generalization from SQUID-trained foundation models to on-scalp OPM-MEG recordings. Using subject-level bootstrap equivalence tests ($\text{SESOI} = \pm 3.0\%$) and matched randomly initialized controls across 21 participants on OpenNeuro `ds005107` and 10 adult participants with 192-channel coregistered triaxial recordings on Zenodo `11126593`, we characterize the transfer dynamics and structural failure modes of these models. First, while single-trial demographic decoding (age, sex, ancestry) remains constrained by a task-level signal-to-noise floor across all architectures (~50% balanced accuracy), an unconfounded visual evoked detection task demonstrates rapid representation scaling (from 95.39% at $B=1$ to 99.81% at $B=20$), with the pretrained autoregressive backbone providing substantial temporal context over tokenizer-only controls ($\Delta = +4.38\%, p < 0.0001$). Second, we identify a categorical sensor-type embedding mismatch: because SQUID pre-training corpora were heavily dominated (>4:1) by planar and axial gradiometers, remapping radial OPM magnetometers to gradiometer embeddings increases zero-shot waveform reconstruction correlation by +52.7% ($0.2501 \to 0.3819, p < 0.0001$), a phenomenon replicated on native SQUID magnetometers (+73.7%). Third, in triaxial arrays, compressing concurrent orthogonal channels through a fixed 16-query spatial attention bottleneck induces severe cross-axis interference ($PCC = 0.3726 \pm 0.03$). Through size-matched capacity controls, we demonstrate that over 76% of this bottleneck stems from directional vector interference rather than channel count alone. An axis-partitioned adapter dedicated to directional sub-arrays restores reconstruction fidelity to $PCC = 0.4849 \pm 0.03$ (+30.1% relative, $p = 0.0016$). These results establish concrete architectural requirements for foundation models operating across emerging electrophysiological hardware.

---

## 1. Introduction

Magnetoencephalography (MEG) measures macroscopic postsynaptic currents with millisecond precision and sub-centimeter source resolution. Conventional MEG instrumentation relies on superconducting quantum interference devices (SQUIDs) operating within liquid helium cryostats at approximately 4 K. The rigid dewar geometry imposes significant brain-to-sensor stand-off distances (typically 20–40 mm in adults and larger in pediatric cohorts), attenuating magnetic field strength as an inverse power of distance and restricting participant movement.

Optically pumped magnetometers (OPMs), particularly alkali vapor cells operating in the spin-exchange relaxation-free (SERF) regime, represent a major technological advance in biomagnetism. OPMs operate at ambient scalp temperatures and conform directly to participant-specific caps, reducing stand-off distances to approximately 5 mm and providing up to a four-fold increase in signal amplitude alongside improved spatial resolution. Furthermore, commercial triaxial sensors (e.g., QuSpin Gen-3 QZFM) concurrently measure three mutually orthogonal magnetic field components ($B_x, B_y, B_z$) at a shared spatial position.

In parallel, computational neuroscience and machine learning research has increasingly focused on self-supervised foundation models for electrophysiology. Architectures such as BrainOmni (OpenTSLab, 2025) tokenize continuous multichannel time series using a spatiotemporal encoder conditioned on 3D sensor coordinates and unit orientation vectors, followed by autoregressive sequence modeling.

However, existing foundation models were trained and validated exclusively on conventional cryogenic SQUID arrays and scalp EEG, without exposure to on-scalp OPM data during pre-training. Whether representations learned from cryogenic recordings generalize to on-scalp OPMs—and where their structural assumptions break down—remains unquantified.

In this study, we systematically investigate SQUID-to-OPM transfer learning by analyzing the inductive biases embedded in the upstream model, diagnosing the domain shifts that occur during hardware transitions, and evaluating modular adaptations to accommodate on-scalp and multi-axis sensing.

---

## 2. Theoretical Analysis of Upstream Architectural Assumptions

BrainOmni processes continuous multichannel recordings through two primary stages:
1. **BrainTokenizer:** A temporal convolutional encoder (`SEANet`) extracts continuous 1D feature maps per channel. A spatial attention module (`BrainSensorModule`) computes query-key-value attention between $N_{\text{neuro}} = 16$ learnable latent query vectors and the sensor embeddings $\mathbf{e}_c = \text{SensorEmbed}(\mathbf{p}_c, \mathbf{u}_c, \text{type}_c)$, where $\mathbf{p}_c \in \mathbb{R}^3$ denotes position, $\mathbf{u}_c \in \mathbb{R}^3$ denotes unit orientation, and $\text{type}_c \in \{0, 1, 2\}$ denotes sensor modality. The output is quantised into discrete tokens via residual vector quantization (RVQ).
2. **Transformer Sequence Model:** An 8.4M parameter autoregressive Transformer processes discrete spatiotemporal tokens across time windows.

### 2.1 The Sensor Category Pre-Training Prior
BrainOmni's pre-training corpus comprises 2,653.8 hours of data (1,997.0 h EEG, 656.8 h MEG). We analyzed the sensor compositions across all 22 pre-training corpora (Appendix D of the upstream publication):
- CTF-275 systems (OMEGA, MEG-Narrative, AversiveMEG, THINGS-MEG): 275 radial axial first-order gradiometers per system (`type=2`).
- Elekta Neuromag-306 systems (Cam-CAN, MIND, SMN4Lang, Kymata): 204 planar gradiometers (`type=2`) and 102 magnetometers (`type=1`) per system.
- KIT / Yokogawa systems: Axial gradiometer arrays (`type=2`).

Across the MEG pre-training corpus, gradiometers (`GRAD`) outnumber magnetometers (`MAG`) by more than 4:1. Cryogenic magnetometers have broad spatial sensitivity profiles and capture substantial distant ambient noise, whereas gradiometers reject homogeneous background interference and preferentially capture local cortical gradients. Consequently, the cross-attention projections for `GRAD=2` developed representations optimized for localized dipolar sources, whereas the `MAG=1` subspace was tuned to more diffuse, lower-SNR signals.

When recording with on-scalp OPM arrays, standard metadata inspection indicates that individual vapor cells measure total magnetic field strength ($B$), leading researchers to classify them as magnetometers (`MAG=1`). However, because on-scalp OPMs sit directly on the scalp surface, their spatial forward fields are highly localized, physically resembling cryogenic gradiometers far more closely than distant SQUID magnetometers.

### 2.2 Coincident Multi-Axis Sensing in Triaxial Arrays
In conventional SQUID arrays, sensors occupy distinct coordinates along a rigid dewar surface. In contrast, modern triaxial OPMs place three orthogonal sensing channels ($B_x, B_y, B_z$) within a single sensor package sharing identical 3D spatial coordinates:
$$\mathbf{p}_{c, X} = \mathbf{p}_{c, Y} = \mathbf{p}_{c, Z}$$
$$\mathbf{u}_{c, X} \perp \mathbf{u}_{c, Y} \perp \mathbf{u}_{c, Z}$$
BrainOmni's spatial tokenizer compresses all input channels into $N_{\text{neuro}} = 16$ latent spatial queries. When triaxial channels are provided, the spatial cross-attention mechanism receives three orthogonal vectors at each coordinate. Because the spatial query capacity is fixed at 16 latent tokens, cross-attention forces tangential and radial components into direct interference within the same low-dimensional subspace.

---

## 3. Materials and Methods

### 3.1 Benchmark Corpora

#### OpenNeuro ds005107 ("FACE-DEC")
- **Cohort:** 21 healthy adult human participants (`sub-01` through `sub-23`, omitting missing subjects `05` and `12`).
- **Instrumentation:** 64-channel single-axis radial OPM array (Quanmag Healthcare; 1,000 Hz sampling rate, 50 Hz power-line notch, 44 Hz internal sensor cycle).
- **Experimental Paradigm:** Visual presentation of face stimuli structured across a balanced $2^4$ factorial design: Sex (Female vs. Male), Ancestry (Asian vs. Caucasian), Age (Young vs. Older), and Expression (Neutral vs. Emotional).
- **Coordinate Framework & Harmonization:** As documented in the public BIDS repository, 3D sensor positions in the release FIF files were approximated from standard Elekta Neuromag sensor positions for 2D topomap display (`CoordinateFrame.APPROX_ELEKTA`) rather than optical millimeter-scale scalp coregistration. Bad-channel rejection removes variable outlier channels per subject. We standardized spatial inputs across all 21 participants by identifying the exact intersection of $C = 50$ helmet sensor positions present in every participant after outlier rejection.

#### Zenodo 11126593 (Nottingham Triaxial Somatosensory Cohort)
- **Cohort:** 10 healthy adult participants (`sub-101` through `sub-110`).
- **Instrumentation:** Up to 192 triaxial OPM channels (QuSpin Gen-3 zero-field magnetometers) coregistered in participant head coordinate space (`CoordinateFrame.HEAD`) using 3D optical surface scanning and digitized landmarks.
- **Experimental Paradigm:** Tactile somatosensory Braille stimulation applied to the right index finger (`Start_index`, class 0) or right little finger (`Start_pinky`, class 1).

### 3.2 Spatial Filtering: Soft Homogeneous Field Correction (Soft-HFC)
To suppress uniform external magnetic interference while maintaining numerical stability, we utilize Soft Homogeneous Field Correction (Soft-HFC):
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q} \in \mathbb{R}^{C \times 3}$ is the matrix of sensitive sensor orientations, and $\mathbf{Q}_R$ is an orthonormal basis derived via thin QR decomposition. Setting $\alpha = 0.25$ attenuates homogeneous external interference by 43.75% in power (25% amplitude shrinkage) while preserving full matrix rank $C$ and avoiding singular covariance matrices. Note that any neural signal component projecting onto the sensor orientation subspace $\text{col}(\mathbf{Q})$ will experience the same fractional attenuation.

### 3.3 Proposed Architectural Solutions

#### 1. Sensor-Type Embedding Calibration (`opm_as_grad`)
For radial on-scalp OPMs, we remap the categorical sensor token from magnetometer (`type=1`) to gradiometer (`type=2`), aligning input data with BrainOmni's dominant pre-training spatial prior without altering underlying numerical waveforms.

#### 2. Axis-Partitioned Tokenizer Adapter (`OPMAwareBrainOmniAdapter`)
To resolve multi-axis interference in triaxial arrays without discarding tangential data, our adapter decomposes the multichannel array into directional subsets:
$$\mathcal{S}_Z = \{c \mid \text{axis}_c = Z\}, \quad \mathcal{S}_X = \{c \mid \text{axis}_c = X\}, \quad \mathcal{S}_Y = \{c \mid \text{axis}_c = Y\}$$
Each directional bank is processed independently through `BrainTokenizer`'s spatial attention mechanism, allocating an independent $N_{\text{neuro}} = 16$ query subspace to each component. The resulting representations are reassembled into composite latent features ($3 \times 16 = 48$ spatial tokens).

### 3.4 Evaluation Protocols and Statistical Framework
- **Cross-Subject Generalization:** Evaluated via leave-one-subject-out (LOSO) cross-validation across nested subject sample budgets $B \in \{1, 2, 4, 10, 20\}$ subjects.
- **Hypothesis Testing:** Two one-sided tests (TOST) against a smallest effect size of interest ($\text{SESOI} = \pm 3.0\%$) using 10,000 subject-level bootstrap resamples. Differences are declared statistically equivalent if the 95% bootstrap confidence interval falls entirely within $[-\text{SESOI}, +\text{SESOI}]$.
- **Baseline Models:** Evaluated against `CompactMEGNet` (a parameter-efficient 1.5k parameter depthwise-separable convolutional neural network) and `ClassicalBandPowerDecoder` (Welch log power across theta, alpha, beta, and gamma bands with $L_2$-regularized logistic regression).

---

## 4. Empirical Results

### 4.1 Cross-Subject Downstream Transfer

Table 1 summarizes cross-subject decoding performance across sample budgets on `ds005107` (21 participants, 50 physically aligned helmet sensors, Soft-HFC $\alpha=0.25$):

**Table 1:** Balanced accuracy (%) and 95% bootstrap confidence intervals for cross-subject demographic decoding on `ds005107` (21 participants, 50 shared helmet channels, Soft-HFC $\alpha=0.25$, evaluated over seeds 42 and 3407).

| Task & Budget | Pre-trained Linear Probe | Control A (Random Init) | Control B (Tokenizer Only) | CompactMEGNet | Classical Baseline | Statistical Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age ($B=1$)** | 49.50% [48.55, 50.49] | 50.41% [49.40, 51.38] | 49.04% [48.47, 49.59] | 49.25% [48.18, 50.41] | 49.13% [48.07, 50.12] | Equivalent (Δ=-0.91% [-2.29, +0.54]) |
| **Age ($B=2$)** | 49.19% [48.46, 50.01] | 50.40% [49.59, 51.24] | 49.51% [48.52, 50.45] | 50.80% [49.97, 51.74] | 50.01% [49.15, 50.78] | Equivalent (Δ=-1.21% [-2.31, -0.01]) |
| **Age ($B=4$)** | 48.94% [48.10, 49.86] | 50.86% [49.83, 51.83] | 50.08% [49.40, 50.82] | 50.70% [49.92, 51.55] | 50.63% [49.68, 51.56] | Inconclusive (Δ=-1.92% [-3.20, -0.60]) |
| **Age ($B=10$)** | 50.08% [49.02, 51.06] | 50.27% [49.11, 51.42] | 49.72% [48.54, 50.92] | 50.73% [49.81, 51.60] | 50.10% [49.12, 51.08] | Equivalent (Δ=-0.19% [-1.73, +1.40]) |
| **Age ($B=20$)** | 49.57% [48.44, 50.73] | 50.68% [49.28, 52.06] | 50.01% [48.55, 51.45] | 51.85% [50.33, 53.34] | 50.56% [49.28, 51.79] | Equivalent (Δ=-1.11% [-2.72, +0.59]) |
| **Ancestry ($B=20$)** | 51.22% [49.81, 52.62] | 48.95% [47.33, 50.65] | 51.45% [50.37, 52.55] | -- | 50.67% [49.22, 52.12] | Inconclusive (Δ=+2.27% [+0.22, +4.30]) |
| **Sex ($B=20$)** | 49.59% [48.47, 50.75] | 47.97% [46.92, 49.05] | 51.14% [50.05, 52.35] | -- | 51.06% [49.41, 52.64] | Inconclusive (Δ=+1.61% [+0.30, +3.06]) |

Two primary observations emerge from these demographic benchmarks:
1. **Empirical Task SNR Floor on Demographics:** Across all sample regimes, neither the pre-trained foundation model, the matched randomly initialized controls (`Control A`), nor the classical spectral baseline (`Classical Baseline`) achieve above-chance balanced accuracy on single-trial face demographic decoding. The observed statistical equivalence ($\Delta \in [-1.92\%, +2.27\%]$) is therefore driven by a task-level single-trial SNR floor on sensor-level recordings without individual anatomical MRI coregistration.
2. **Re-evaluating Tokenizer Sufficiency vs. Backbone Value:** On demographic tasks, `Control B` (pre-trained `BrainTokenizer` coupled with an untrained, randomly initialized Transformer backbone) matches the fully pre-trained model ($50.01\%$ vs. $49.57\%$ on age; $51.45\%$ vs. $51.22\%$ on ancestry; $51.14\%$ vs. $49.59\%$ on sex) solely because all configurations sit at chance. As shown in Section 4.2 below, when evaluated on high-SNR evoked responses, the full model strongly outperforms Control B, confirming that the autoregressive backbone encodes crucial cross-trial temporal structure in decodable regimes.

### 4.2 High-SNR Sensory Detection Scaling & Transfer Dynamics

To evaluate representation transfer in a regime with robust physiological signal, we tested an evoked visual detection paradigm on `ds005107`, distinguishing pre-stimulus baseline ($-200\text{ ms}$ to $0\text{ ms}$) from peak evoked response ($+50\text{ ms}$ to $+550\text{ ms}$) within matched temporal token windows. Under linear probing, pretrained representations scaled monotonically from 95.39% balanced accuracy (95% CI: [94.11, 96.56]) at $B=1$ (256 trials) to 99.81% [99.65, 99.94] at $B=20$ (5,120 trials), significantly outperforming randomly initialized controls (Control A: 94.47% to 96.60%; $\Delta = +3.22\%$, 95% CI: [+2.13, +4.42], $p < 0.0001$). 

Crucially, `Control B` (pretrained tokenizer coupled with an untrained Transformer backbone) scaled from 78.90% [76.72, 81.09] at $B=1$ to 95.43% [94.54, 96.27] at $B=20$, trailing the full pretrained model by $\Delta = -4.38\%$ ($p < 0.0001$). This performance gap indicates that in regimes with detectable neural responses, the autoregressive backbone encodes substantial cross-trial temporal structure that cannot be reproduced by random linear projections. The compact convolutional specialist (`CompactMEGNet`) required larger training pools to converge (75.21% at $B=1$), eventually reaching parity with the foundation model at $B=20$ (99.81% [99.70, 99.92]; Figure 1).

![Figure 1: Cross-subject data-efficiency scaling curves across sample budgets on OpenNeuro ds005107 (N=21 subjects, 50 physically aligned helmet channels).](figures/fig1_data_efficiency_curves.png)

### 4.3 Sensor-Type Embedding Calibration

**Table 2:** Zero-shot waveform reconstruction Pearson correlation coefficient (PCC) and downstream classification across sensor configuration modes on `ds005107` (all 21 participants, 50 aligned sensors).

| Configuration Mode | Sensor Type Tag | Mean PCC [95% CI] | Relative Difference | Age BACC ($B=20$) [95% CI] |
| :--- | :--- | :--- | :--- | :--- |
| **Default** | Magnetometer (`MAG=1`) | 0.2501 [0.2450, 0.2549] | Baseline | 49.57% [48.44, 50.73] |
| **Remapped (`opm_as_grad`)** | Gradiometer (`GRAD=2`) | 0.3819 [0.3779, 0.3859] | +52.7% ($p < 0.0001$) | 49.41% [48.14, 50.69] |
| **Zero Orientation** | `MAG=1`, $\mathbf{u} = \mathbf{0}$ | 0.2359 [0.2295, 0.2421] | -5.7% | 50.85% [49.62, 52.09] |
| **Inverted Orientation** | `MAG=1`, $\mathbf{u} = -\mathbf{u}$ | 0.2340 [0.2274, 0.2402] | -6.4% | 49.02% [47.83, 50.20] |
| **Scrambled Positions** | Randomly Permuted | 0.2330 [0.2307, 0.2354] | -6.8% | 50.46% [49.33, 51.68] |

Mapping on-scalp OPM magnetometers to `GRAD=2` induces a +52.7% relative improvement in zero-shot waveform reconstruction fidelity without requiring retraining. Conversely, zeroing or inverting 3D orientation vectors significantly impairs reconstruction ($p < 0.001$), confirming that the spatial tokenizer actively leverages directional vectors during decoding.

![Figure 2: Zero-shot waveform reconstruction Pearson correlation coefficient (PCC) across sensor category embeddings and geometric perturbations on ds005107 (N=21).](figures/fig2_sensor_geometry_ablations.png)

### 4.4 Triaxial Sensor Geometry & Axis-Partitioned Tokenization

Evaluating 10 adult participants with 192-channel coregistered triaxial QuSpin OPM recordings (Zenodo 11126593) demonstrated the impact of coincident multi-axis sensing across 10,000 subject-level bootstrap resamples:

**Table 3:** Zero-shot reconstruction PCC and 5-fold cross-validated tactile Braille decoding balanced accuracy across triaxial configurations (10 adult subjects, 84 trials/subject, Zenodo 11126593).

| Array Configuration | Channel Count | Reconstruction PCC [95% CI] | Braille Decoding BACC [95% CI] | Paired Test vs Monolithic Raw |
| :--- | :--- | :--- | :--- | :--- |
| **Monolithic Triaxial ($X/Y/Z$ Raw)** | 155–176 | 0.3726 [0.3416, 0.4046] | 48.24% [43.64, 52.67] | Baseline |
| **Triaxial ($X/Y/Z$ + Soft-HFC $\alpha=0.25$)** | 155–176 | 0.3461 [0.3115, 0.3816] | 49.07% [44.99, 52.89] | -- |
| **Random 3-way Split (Capacity Control)** | 3 × ~55 | 0.3995 [0.3697, 0.4287] | 48.24% [43.64, 52.67] | $\Delta = +0.0269$ ($p = 0.0016$) |
| **Radial Isolation ($Z$-axis only)** | 55–60 | 0.4944 [0.4667, 0.5265] | 47.19% [40.76, 53.44] | $\Delta = +0.1218$ ($p = 0.0016$) |
| **Tangential Only ($X/Y$)** | 100–116 | 0.4446 [0.4112, 0.4771] | 49.35% [46.06, 52.51] | $\Delta = +0.0720$ ($p = 0.0016$) |
| **Axis-Partitioned Adapter ($X/Y/Z$)** | 155–176 | 0.4849 [0.4507, 0.5199] | 50.19% [43.72, 56.40] | $\Delta = +0.1123$ ($p = 0.0016$) |

Paired sign-flip permutation tests across all 10 participants (84 trials per subject) demonstrated that routing 155–176 triaxial channels through a monolithic 16-query bottleneck substantially degrades waveform reconstruction. Isolating radial ($Z$) channels increased Pearson correlation from $0.3726$ to $0.4944$ ($\Delta = +0.1218$, 95% CI: [0.0962, 0.1466], $p = 0.0016$). 

The axis-partitioned adapter preserved all triaxial channels while raising fidelity to $PCC = 0.4849$ [0.4507, 0.5199] ($\Delta = +0.1123$ [0.1042, 0.1201], $p = 0.0016$). Comparing this architecture against a size-matched random three-way channel partition (capacity control, $PCC = 0.3995$) showed that channel count per query accounted for only $+0.0269$ of the total gain. By contrast, decomposing the array by physical spatial orientation contributed an additional $\Delta = +0.0854$ [0.0753, 0.0955] ($p = 0.0016$). This indicates that 76.1% of the performance deficit is attributable to vector cross-axis interference when orthogonal sensing coils occupy identical 3D positions.

![Figure 3: Triaxial waveform reconstruction PCC and 5-fold tactile Braille decoding balanced accuracy across array configurations on Zenodo 11126593 (N=10 adult participants).](figures/fig3_nottingham_triaxial_bottleneck.png)

### 4.5 In-Domain SQUID Control & The Sensor-Type Bias Mechanism

To verify whether foundation model transfer bottlenecks are hardware-specific rather than indicative of general model pathology, we evaluated BrainOmni on an in-domain cryogenic SQUID control (OpenNeuro `ds000117`, Wakeman & Henson, Elekta Neuromag-306 with 102 magnetometers and 204 planar gradiometers; evaluated as a within-subject 5-fold cross-validation control on `sub-01` run-01):

**Table 4:** In-domain cryogenic SQUID baseline (`ds000117`, `sub-01`) versus on-scalp OPM transfer (`ds005107`, $N=21$ subjects).

| Hardware & Dataset | Array & Channels | Sensor Type Tag | Zero-Shot PCC | Evoked Detection (Pretrained vs Ctrl A) | Face Decoding (Pretrained vs Ctrl A) | Sensor-Type Remapping Effect |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **SQUID (`ds000117`, sub-01)** | Full Array (306 ch) | `MAG=1, GRAD=2` | 0.4177 | 97.56% vs 92.69% (Δ = +4.87%) | 61.03% vs 55.81% (Δ = +5.22%) | Native Dual-Sensor Baseline |
| **SQUID (`ds000117`, sub-01)** | Magnetometers (102 ch) | `MAG=1` | 0.4105 | -- | -- | SQUID Magnetometer Baseline |
| **SQUID (`ds000117`, sub-01)** | Gradiometers (204 ch) | `GRAD=2` | 0.4473 | -- | -- | SQUID Gradiometer Baseline |
| **SQUID (`ds000117`, sub-01)** | Inverse Control (102 ch Mag) | `GRAD=2` | **0.7130** | -- | -- | **Δ = +0.3025 (+73.7% gain)** |
| **OPM (`ds005107`, N=21)** | Whole-Head (50 ch) | `MAG=1` | 0.2501 | 99.81% vs 96.60% (Δ = +3.22%, p < 0.0001) | Floor (~50%) | OPM Magnetometer Baseline |
| **OPM (`ds005107`, N=21)** | Remapped (`opm_as_grad`) | `GRAD=2` | **0.3819** | -- | Floor (~50%) | **Δ = +0.1318 (+52.7% gain)** |

Evaluating the model within its training domain yields two conclusions:

First, on cryogenic recordings, the pretrained foundation model consistently outperforms matched random controls on both visual detection (97.56% vs. 92.69%, $\text{AUC} = 0.9993$ vs. $0.9932$) and face-versus-scrambled classification (61.03% vs. 55.81%, $\text{AUC} = 0.5909$ vs. $0.5591$). `Control B` underperforms the full model on both tasks (88.45% on detection, 56.10% on face decoding), corroborating that the autoregressive sequence backbone captures meaningful temporal dynamics when evaluated in-domain.

Second, remapping native SQUID magnetometers from `MAG=1` to `GRAD=2` increased zero-shot reconstruction correlation from $0.4105$ to $0.7130$—a $+73.7\%$ relative gain ($\Delta = +0.3025$) on cryogenic SQUID data itself. This mirrors the $+52.7\%$ gain observed when remapping on-scalp OPM magnetometers ($0.2501 \to 0.3819$). This result provides empirical confirmation that `opm_as_grad` does not compensate for an OPM-specific geometric anomaly; rather, it reflects an upstream inductive bias in the sensor category embedding weights, which were heavily skewed by the numerical preponderance of planar gradiometers in SQUID pre-training datasets.

---

## 5. Discussion

Our empirical benchmark provides four primary recommendations for the design and deployment of electrophysiological foundation models across emerging sensor modalities:

**Hardware Inductive Biases in Pre-Trained Representations.** Large-scale self-supervised pre-training on cryogenic SQUID systems produces representations that reflect distant dewar geometries and the statistical dominance of planar gradiometers. When applying these models to on-scalp OPM arrays, simple remapping of categorical sensor embeddings from magnetometer to gradiometer recovers over half of the reconstruction loss without parameter fine-tuning. This highlights the sensitivity of transformer cross-attention mechanisms to discrete sensor tags and underscores the need to explicitly balance or parameterize sensor transfer functions during pre-training.

**Complementary Roles of Tokenization and Sequence Modeling.** While single-trial demographic decoding collapsed to chance across all evaluated architectures, evaluating high-SNR evoked responses revealed distinct contributions across the model hierarchy. Early spatio-temporal convolutions provide localized filtering and spatial coherence, but the autoregressive sequence backbone confers critical representation gains (+4.38% over tokenizer-only controls on visual detection, and +4.93% on SQUID face decoding). Future evaluations of electrophysiological foundation models must employ tasks with demonstrated signal decodability to avoid conflating task noise floors with representation quality.

**Architectural Solutions for Multi-Axis Sensing.** As triaxial OPM modules become standard in wearable MEG, foundation model tokenizers must accommodate coincident vector sensing. Constraining 155–176 triaxial channels through a monolithic 16-token spatial query bottleneck induces severe cross-axis interference, with vector orthogonality accounting for over 76% of the bottleneck. Partitioning spatial attention by directional sensing axes provides an effective, modular solution that recovers fidelity while retaining complete triaxial recordings.

**Practical Utility of Compact Specialists.** In modest experimental cohorts (10–30 subjects) typical of clinical and cognitive neuroimaging, parameter-efficient convolutional networks such as `CompactMEGNet` (1.5k parameters) remain competitive with multi-million-parameter foundation models. For task-specific classification under limited data budgets, compact specialists offer a computationally efficient alternative, while foundation models offer broader utility for zero-shot waveform reconstruction, denoising, and cross-modality representation transfer.

---

## 6. Code and Data Availability

The complete benchmark pipeline, preprocessing modules, model wrappers, and evaluation scripts are publicly available under an open-source MIT license at:  
`https://github.com/samerzumot/opm-bridge`

Upstream modifications and bug fixes identified during this evaluation have been contributed to the official BrainOmni repository (`OpenTSLab/BrainOmni#2`).

---

## References

1. OpenTSLab. BrainOmni: A Brain Foundation Model for Unified EEG and MEG Signals. *Advances in Neural Information Processing Systems (NeurIPS)*, 2025.
2. Rier, L., Rhodes, N., et al. Tracking the neurodevelopmental trajectory of beta band oscillations with optically pumped magnetometer MEG. *eLife*, 13:RP94562, 2024.
3. Tierney, T. M., et al. Homogeneous field correction for optically pumped magnetometers. *NeuroImage*, 237:118178, 2021.
4. Lawhern, V. J., et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 15(5):056013, 2018.
5. Boto, E., et al. Moving magnetoencephalography towards real-world applications with a wearable system. *Nature*, 555(7698):657–661, 2018.
6. Mellor, S., et al. Spatial filtering for optically pumped magnetometer MEG. *NeuroImage*, 260:119488, 2022.
7. Wakeman, D. G., & Henson, R. N. A multi-subject, multi-modal human neuroimaging dataset. *Scientific Data*, 2:150001, 2015.
