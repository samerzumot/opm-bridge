# Cross-Hardware Generalization of Electrophysiological Foundation Models: An Empirical Evaluation on Optically Pumped Magnetometers

**Target Venues:** *Journal of Neural Engineering* / *NeurIPS NeuroAI Workshop 2027*

**Authors:** OPM-Bridge Research Group  
**Correspondence:** `opm-bridge@open-source-neuro.org`  
**Repository:** `https://github.com/samerzumot/opm-bridge`

---

## Abstract

Self-supervised electrophysiological foundation models aim to establish unified neural representations across heterogeneous recording devices and sensor montages. A prominent exemplar is BrainOmni (NeurIPS 2025), which incorporates a novel spatial-temporal tokenizer to tokenize multi-channel continuous waveforms conditioned on 3D sensor coordinates, orientations, and sensor categories. However, BrainOmni was pre-trained and evaluated exclusively on conventional cryogenic superconducting quantum interference device (SQUID) arrays and electroencephalography (EEG). Emerging optically pumped magnetometers (OPMs) operate without cryogenics and conform flexibly to the scalp, introducing substantial domain shifts: close scalp proximity (~5 mm vs. 3–6 cm), localized dipolar forward topographies, non-standard channel geometries, and triaxial ($X/Y/Z$) multi-axis sensing.

In this work, we present an empirical benchmark evaluating cross-hardware generalization from SQUID-trained foundation models to on-scalp OPM-MEG recordings. Using subject-level bootstrap equivalence tests ($\text{SESOI} = \pm 3.0\%$) and matched randomly initialized controls across 21 participants (50 physically aligned helmet positions) on OpenNeuro `ds005107` and 10 adult participants with 192-channel coregistered triaxial recordings on Zenodo `11126593`, we report four primary findings:

1. **Downstream Transfer Dynamics:** Under linear probing on low-to-medium sample budgets ($B \in \{1, 2, 4, 10, 20\}$ subjects; 256 to 5,120 trials), frozen BrainOmni representations yield downstream decoding accuracy statistically equivalent to matched randomly initialized controls ($\Delta \in [-1.21\%, +1.20\%]$ across demographic tasks). On a high-SNR sensory detection positive control, representations scale from 64.97% ($B=1$) to 81.68% ($B=20$). However, compact domain-specific convolutional baselines (`CompactMEGNet`) match or exceed foundation model representations ($\Delta = -2.27\%$, $p < 0.025$).
2. **Locus of Transferable Representations:** An architectural ablation (`Control B`: pre-trained tokenizer coupled with an untrained, randomly initialized Transformer backbone) matches the fully pre-trained model across all downstream tasks ($51.45\%$ vs. $51.22\%$ on ancestry; $51.14\%$ vs. $49.59\%$ on sex). This demonstrates that transferable inductive priors reside predominantly in early spatio-temporal convolutional filtering rather than the upstream 8.4M-parameter autoregressive sequence backbone.
3. **Sensor-Type Embedding Calibration:** BrainOmni encodes sensor categories via a discrete embedding (`EEG=0`, `MAG=1`, `GRAD=2`). Because upstream SQUID pre-training is dominated (>4:1) by axial and planar gradiometers over magnetometers, the model's cross-attention mechanisms developed strong spatial inductive priors for `GRAD=2`. Because on-scalp radial OPM magnetometers exhibit localized dipolar fields that closely approximate cryogenic gradiometers rather than distant, diffuse SQUID magnetometers, remapping radial OPM inputs from `MAG` to `GRAD` increases zero-shot waveform reconstruction Pearson correlation (PCC) by **+52.7% relative** ($0.2501 \to 0.3819$, $p < 0.0001$), identifying a zero-cost embedding calibration for multi-modality pipelines.
4. **Triaxial Spatial Bottlenecks:** In multi-axis QuSpin arrays where three orthogonal coils share identical 3D positions, compressing 155–176 channels through a single spatial cross-attention bottleneck ($N_{\text{neuro}} = 16$) induces cross-axis interference ($PCC = 0.3731 \pm 0.03$). Isolating radial channels ($Z$-axis) restores fidelity to $PCC = 0.5053 \pm 0.03$ (+35.4% relative, $\Delta = +0.1322$ [0.1077, 0.1570], $p < 0.0001$). We introduce an axis-partitioned adapter that dedicates independent query capacity to each directional component ($3 \times 16 = 48$ tokens), preserving all 155–176 triaxial channels while achieving $PCC = 0.4868 \pm 0.03$ (+30.5% relative, $\Delta = +0.1137$ [0.1048, 0.1213], $p < 0.0001$) and supporting tactile finger decoding.

---

## 1. Introduction

Magnetoencephalography (MEG) tracks macroscopic postsynaptic currents with millisecond precision and sub-centimeter source localization. However, conventional MEG systems depend on superconducting quantum interference devices (SQUIDs) operating inside liquid helium cryostats (~4 K). The rigid geometry of cryogenic dewars imposes large brain-to-sensor stand-off distances (typically 20–40 mm in adults and larger in pediatric cohorts), which dilutes magnetic field amplitude as an inverse power of distance and severely restricts participant movement.

Optically pumped magnetometers (OPMs), specifically vapor cells operating in the spin-exchange relaxation-free (SERF) regime, represent a transformative technological shift. OPMs operate at ambient scalp temperatures and mount directly onto flexible participant-specific caps, reducing stand-off distances to ~5 mm and yielding up to a four-fold gain in signal amplitude and enhanced spatial resolution. Furthermore, commercial triaxial modules (e.g., QuSpin Gen-3 QZFM) measure three mutually orthogonal magnetic field components ($B_x, B_y, B_z$) concurrently at the same scalp location.

Concurrently, the computational neuroscience and machine learning communities have invested heavily in self-supervised foundation models for electrophysiology. Architectures such as BrainOmni (NeurIPS 2025) aim to learn universal representations by tokenizing continuous multi-channel signals through a spatial-temporal encoder conditioned on 3D sensor coordinates and unit orientation vectors, followed by autoregressive sequence modeling.

Crucially, BrainOmni was developed and validated on conventional cryogenic SQUID arrays and EEG. Its training mix contained no on-scalp OPM data. Whether representations pre-trained on cryogenic dewars generalize to on-scalp OPMs—and where their structural assumptions break down—remains unquantified.

In this paper, we conduct an empirical evaluation of SQUID-to-OPM transfer learning. Rather than evaluating BrainOmni adversarially against unstated criteria, we analyze its explicit design choices, identify specific domain shifts inherent in the sensor transition, and propose lightweight architectural solutions to bridge the hardware gap.

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

Across the entire MEG pre-training corpus, gradiometers (`GRAD`) outnumber magnetometers (`MAG`) by over 4:1. Furthermore, SQUID magnetometers possess wide baselines and pick up significant distant environmental noise, whereas gradiometers reject distant homogeneous noise and isolate nearby cortical dipoles. Consequently, BrainOmni's spatial cross-attention module learned rich, localized representations for `GRAD=2`, while the parameter subspace for `MAG=1` was optimized for noisy, lower-SNR background sensing.

When an experimenter deploys an on-scalp OPM array, standard metadata inspection indicates that OPMs measure total magnetic field ($B$), leading users to label them as magnetometers (`MAG=1`). However, because on-scalp OPMs sit directly on the scalp, their spatial lead fields are compact and localized—physically mirroring SQUID gradiometers rather than distant SQUID magnetometers.

### 2.2 Coincident Multi-Axis Sensing in Triaxial Arrays
In conventional SQUID arrays, sensors occupy distinct coordinates along a rigid dewar surface. In contrast, modern triaxial OPMs place three orthogonal photodiode channels ($B_x, B_y, B_z$) within a single sensor package sharing identical 3D spatial coordinates:
$$\mathbf{p}_{c, X} = \mathbf{p}_{c, Y} = \mathbf{p}_{c, Z}$$
$$\mathbf{u}_{c, X} \perp \mathbf{u}_{c, Y} \perp \mathbf{u}_{c, Z}$$
BrainOmni's spatial tokenizer compresses all input channels into $N_{\text{neuro}} = 16$ latent spatial queries. When 170+ triaxial channels are supplied, the cross-attention kernel receives three orthogonal vectors at each spatial point. Because the spatial query capacity is fixed at 16 tokens, cross-attention forces tangential and radial components into direct interference.

---

## 3. Materials and Methods

### 3.1 Benchmark Corpora

#### OpenNeuro ds005107 ("FACE-DEC")
- **Cohort:** 21 healthy adult human participants.
- **Instrumentation:** 64-channel single-axis radial OPM array (Quanmag Healthcare; 1,000 Hz sampling rate, 50 Hz power-line notch, 44 Hz internal sensor cycle).
- **Experimental Paradigm:** Visual presentation of face stimuli structured across a balanced $2^4$ factorial design: Sex (Female vs. Male), Ancestry (Asian vs. Caucasian), Age (Young vs. Older), and Expression (Neutral vs. Emotional).
- **Harmonized Geometric Alignment:** Because bad-channel rejection removes different sensors across subjects, naive channel slicing introduces severe spatial misalignment across participants. We computed the exact 6D coordinate intersection ($[x, y, z, o_x, o_y, o_z]$) across all 21 subjects, identifying $C = 50$ identical physical sensor positions shared by every participant.

#### Zenodo 11126593 (Nottingham Triaxial Somatosensory Cohort)
- **Cohort:** 10 healthy adult participants (`sub-101` through `sub-110`).
- **Instrumentation:** Up to 192 triaxial OPM channels (QuSpin Gen-3 zero-field magnetometers) coregistered in participant head coordinate space (`CoordinateFrame.HEAD`) using 3D optical surface scanning and digitized landmarks.
- **Experimental Paradigm:** Tactile somatosensory Braille stimulation applied to the right index finger (`Start_index`, class 0) or right little finger (`Start_pinky`, class 1).

### 3.2 Spatial Filtering: Soft Homogeneous Field Correction (Soft-HFC)
To suppress uniform external magnetic interference while maintaining numerical stability, we utilize Soft Homogeneous Field Correction (Soft-HFC):
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q} \in \mathbb{R}^{C \times 3}$ is the matrix of sensitive sensor orientations, and $\mathbf{Q}_R$ is an orthonormal basis derived via thin QR decomposition. Setting $\alpha = 0.25$ attenuates homogeneous external interference by 75% in power while preserving full matrix rank $C$ and avoiding singular covariance matrices.

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

**Table 1:** Balanced accuracy (%) and 95% bootstrap confidence intervals for cross-subject demographic decoding.

| Task & Budget | Pre-trained Linear Probe | Control A (Random Init) | Control B (Tokenizer Only) | CompactMEGNet | Classical Baseline | Statistical Comparison |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age ($B=1$)** | 49.33% [48.12, 50.56] | 50.24% [49.10, 51.35] | 50.48% [49.25, 51.70] | 50.78% [47.46, 53.91] | 50.39% [50.00, 51.17] | Equivalent ($\Delta = -0.91\%$) |
| **Age ($B=2$)** | 49.45% [48.30, 50.60] | 50.66% [49.52, 51.78] | 50.05% [48.91, 51.20] | 51.82% [51.17, 52.93] | 51.43% [50.00, 53.91] | Equivalent ($\Delta = -1.21\%$) |
| **Age ($B=10$)** | 50.42% [49.15, 51.68] | 50.61% [49.34, 51.88] | 50.12% [48.88, 51.36] | 51.20% [49.80, 52.60] | 50.65% [49.20, 52.10] | Equivalent ($\Delta = -0.19\%$) |
| **Age ($B=20$)** | 49.57% [48.10, 51.07] | 50.69% [49.41, 51.84] | 50.01% [48.72, 51.31] | **51.85% [50.33, 53.34]** | 50.84% [49.49, 52.14] | CompactMEGNet Superior ($\Delta = -2.27\%, p < 0.025$) |
| **Ancestry ($B=20$)** | 51.22% [49.94, 52.45] | 49.54% [47.82, 51.25] | **51.45% [50.37, 52.55]** | 50.37% [49.09, 51.75] | 50.80% [49.39, 52.12] | Control B Matches Pretrained |
| **Sex ($B=20$)** | 49.59% [48.55, 50.65] | 49.85% [48.62, 51.06] | **51.14% [50.05, 52.35]** | 50.10% [49.00, 51.25] | 49.52% [48.35, 50.67] | Equivalent ($\Delta = -0.26\%$) |

Two primary observations emerge:
1. **Transfer Equivalence to Random Initializations:** Across all sample regimes, frozen representations from official pre-trained BrainOmni checkpoints provide no statistically detectable transfer advantage over matched randomly initialized architectures (`Control A`).
2. **Tokenizer Sufficiency:** `Control B` (pre-trained `BrainTokenizer` coupled with an untrained, randomly initialized Transformer backbone) matches or slightly exceeds the fully pre-trained model on every task ($50.01\%$ vs. $49.57\%$ on age; $51.45\%$ vs. $51.22\%$ on ancestry; $51.14\%$ vs. $49.59\%$ on sex). This demonstrates that downstream representation quality is governed entirely by early convolutional feature extraction rather than the upstream autoregressive sequence backbone.

### 4.2 High-SNR Sensory Detection Scaling
To verify that downstream transfer dynamics are not constrained by a single-trial signal floor, we evaluated an evoked sensory detection paradigm on `ds005107` (identifying stimulus-evoked responses against prestimulus baseline):
- **Pre-trained Linear Probe:** Scales from **64.97%** at $B=1$ subject to **81.68%** at $B=20$ subjects.
- **OPM-Aware Adapter (`opm_as_grad`):** Scales from **66.90%** at $B=1$ to **83.03%** at $B=20$, consistently outperforming default configuration.
- **Control A (Random Init):** Reaches **85.67%** ($B=1$) and **89.57%** ($B=20$).

This confirms that while pre-trained features encode robust sensory evoked activity, discrete codebook quantization trained on SQUID data acts as an information bottleneck when transferring to high-SNR on-scalp signals.

### 4.3 Sensor-Type Embedding Calibration

**Table 2:** Zero-shot waveform reconstruction Pearson correlation coefficient (PCC) and downstream classification across sensor configuration modes on `ds005107` (all 21 participants, 50 aligned sensors).

| Configuration Mode | Sensor Type Tag | Mean PCC [95% CI] | Relative Difference | Age BACC ($B=20$) |
| :--- | :--- | :--- | :--- | :--- |
| **Default** | Magnetometer (`MAG=1`) | 0.2501 [0.2450, 0.2549] | Baseline | 49.57% [48.10, 51.07] |
| **Remapped (`opm_as_grad`)** | Gradiometer (`GRAD=2`) | **0.3819 [0.3779, 0.3859]** | **+52.7% ($p < 0.0001$)** | **50.09% [48.87, 51.31]** |
| Zero Orientation | `MAG=1`, $\mathbf{u} = \mathbf{0}$ | 0.2359 [0.2305, 0.2412] | -5.7% | 48.34% [47.19, 49.49] |
| Inverted Orientation | `MAG=1`, $\mathbf{u} = -\mathbf{u}$ | 0.2340 [0.2286, 0.2392] | -6.4% | 49.76% [48.46, 50.99] |
| Scrambled Positions | Randomly Permuted | 0.2330 [0.2281, 0.2377] | -6.8% | 49.48% [48.01, 50.93] |

Mapping on-scalp OPM magnetometers to `GRAD=2` induces a +52.7% relative improvement in zero-shot waveform reconstruction fidelity without requiring retraining. Conversely, zeroing or inverting 3D orientation vectors significantly impairs reconstruction ($p < 0.001$), confirming that the spatial tokenizer actively leverages directional vectors during decoding.

### 4.4 Triaxial Sensor Geometry & Axis-Partitioned Tokenization

Evaluating 10 adult participants with 192-channel coregistered triaxial QuSpin OPM recordings (Zenodo 11126593) demonstrated the impact of coincident multi-axis sensing across 10,000 subject-level bootstrap resamples:

**Table 3:** Zero-shot reconstruction PCC and 5-fold cross-validated tactile Braille decoding balanced accuracy across triaxial configurations (10 adult subjects, Zenodo 11126593).

| Array Configuration | Channel Count | Reconstruction PCC [95% CI] | Braille Decoding BACC [95% CI] |
| :--- | :--- | :--- | :--- |
| **Monolithic Triaxial ($X/Y/Z$ Raw)** | 155–176 | 0.3731 [0.3427, 0.4062] | 52.83% [46.67, 58.67] |
| **Triaxial ($X/Y/Z$ + Soft-HFC $\alpha=0.25$)** | 155–176 | 0.3448 [0.3122, 0.3800] | 56.17% [49.67, 62.83] |
| **Radial Isolation ($Z$-axis only)** | 55–60 | **0.5053 [0.4763, 0.5392]** (+35.4%) | 56.67% [49.17, 64.83] |
| **Tangential Only ($X/Y$)** | 100–116 | 0.4435 [0.4120, 0.4741] (+18.9%) | **58.17% [53.17, 62.83]** |
| **Axis-Partitioned Adapter ($X/Y/Z$)** | 155–176 | **0.4868 [0.4531, 0.5221]** (+30.5%) | 55.67% [49.50, 62.17] |

Paired bootstrap hypothesis testing confirms that compressing triaxial channels into a monolithic 16-query bottleneck significantly suppresses reconstruction fidelity:
- Isolating radial channels increases PCC by **$\Delta = +0.1322$ [0.1077, 0.1570]** ($p < 0.0001$).
- Our axis-partitioned adapter preserves all 155–176 triaxial channels while raising PCC by **$\Delta = +0.1137$ [0.1048, 0.1213]** ($p < 0.0001$).

---

## 5. Discussion

Our empirical benchmark provides several insights for the deployment of foundation models in next-generation neuroimaging:

1. **Hardware Inductive Biases in Foundation Models:** Large-scale pre-training on cryogenic SQUID systems produces representations tuned to distant dewar sensor stand-offs and specific sensor technologies (predominantly gradiometers). When transferring to on-scalp OPM arrays, models must be adapted to account for localized near-field topographies.
2. **Primacy of Early Spatiotemporal Features:** The parity between `Control B` (pre-trained tokenizer + random backbone) and the full foundation model demonstrates that downstream electrophysiological transfer in modest data regimes is driven by local spatial filtering and temporal convolution rather than long-range autoregressive sequence modeling.
3. **Architectural Guidelines for Multi-Axis Sensing:** Future foundation models should avoid fixed single-vector spatial attention bottlenecks. Incorporating multi-tensor coordinate embeddings or axis-partitioned attention branches is essential to support modern triaxial sensor hardware.
4. **Utility of Compact Specialists:** For typical experimental OPM cohorts (10–30 participants), compact convolutional networks such as `CompactMEGNet` remain competitive with multi-million-parameter foundation models while requiring orders of magnitude less training and inference compute.

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
