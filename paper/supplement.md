# Supplementary Material: Cross-Hardware Generalization of Electrophysiological Foundation Models: An Empirical Evaluation on Optically Pumped Magnetometers

**Target Venue:** *NeurIPS NeuroAI Workshop*  
**Authors:** Samer Zumot, Marzieh Zare  
**Correspondence:** `srzumot@gmail.com`  
**Repository:** `https://github.com/samerzumot/opm-bridge`

---

## Section S1: Mathematical Foundations of Soft-HFC

### S1.1 Ambient Interference Physics and Hard HFC Limitations
External magnetic interference in non-cryogenic shielded rooms often manifests as a uniform, homogeneous spatial field:
$$\mathbf{B}_{\text{ext}} = [B_x, B_y, B_z]^T \in \mathbb{R}^3$$

For an array of $C$ sensors with sensitive unit orientations $\mathbf{u}_c = [u_{c,x}, u_{c,y}, u_{c,z}]^T$, the measurement at channel $c$ induced by this homogeneous field is:
$$m_c = \mathbf{u}_c^T \mathbf{B}_{\text{ext}}$$
In matrix notation, the collective external interference measured across the array is:
$$\mathbf{m}_{\text{ext}} = \mathbf{Q} \mathbf{B}_{\text{ext}}$$
where $\mathbf{Q} \in \mathbb{R}^{C \times 3}$ is the matrix whose $c$-th row is $\mathbf{u}_c^T$.

Conventional Homogeneous Field Correction (Hard HFC; Tierney et al., 2021) constructs an orthogonal projection operator onto the nullspace of $\mathbf{Q}^T$:
$$\mathbf{P}_{\text{hard}} = \mathbf{I}_C - \mathbf{Q} (\mathbf{Q}^T \mathbf{Q})^{-1} \mathbf{Q}^T$$
Because $\mathbf{Q}$ has rank 3 (assuming non-coplanar sensor orientations), the projection operator $\mathbf{P}_{\text{hard}}$ strictly reduces the rank of the data covariance matrix:
$$\text{rank}(\mathbf{P}_{\text{hard}}) = C - 3$$
This rank deficiency creates singular covariance matrices in downstream estimators, requiring pseudo-inversion or explicit subspace truncation in linear probes and spatial filter designs.

### S1.2 Soft Homogeneous Field Correction (Soft-HFC)
To maintain numerical stability and preserve full rank, we implement Soft-HFC parameterized by a continuous shrinkage coefficient $\alpha \in [0, 1)$:
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I}_C - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q} = \mathbf{Q}_R \mathbf{R}$ represents the thin QR decomposition of the orientation matrix $\mathbf{Q}$, with $\mathbf{Q}_R \in \mathbb{R}^{C \times 3}$ satisfying $\mathbf{Q}_R^T \mathbf{Q}_R = \mathbf{I}_3$.

**Properties of $\mathbf{P}_{\text{soft}}(\alpha)$:**
1. **Full Rank Preservation:** The eigenvalues of $\mathbf{P}_{\text{soft}}(\alpha)$ are:
   $$\lambda_i = \begin{cases} 1 - \alpha, & \text{for } 3 \text{ eigenvectors spanning } \text{col}(\mathbf{Q}) \\ 1, & \text{for } C - 3 \text{ eigenvectors in } \text{null}(\mathbf{Q}^T) \end{cases}$$
   For any $\alpha < 1.0$, all eigenvalues are strictly positive ($\lambda_i > 0$), guaranteeing:
   $$\text{rank}(\mathbf{P}_{\text{soft}}(\alpha)) = C$$
2. **Predictable Field Attenuation:** Setting $\alpha = 0.25$ attenuates signals in the uniform field subspace by a factor of $(1 - \alpha) = 0.75$ in amplitude (a 25% shrinkage), corresponding to a fractional power reduction of:
   $$1 - (1 - \alpha)^2 = 1 - 0.75^2 = 43.75\%$$
3. **Orthogonal Distortion-Free Subspace:** Neural magnetic fields orthogonal to the homogeneous subspace $\text{col}(\mathbf{Q})$ satisfy $\mathbf{Q}_R^T \mathbf{s}_{\text{neural}} = \mathbf{0}$, yielding:
   $$\mathbf{P}_{\text{soft}}(\alpha) \mathbf{s}_{\text{neural}} = \mathbf{s}_{\text{neural}}$$
   leaving dipolar neural topographies completely unperturbed. Any neural signal component that happens to project onto $\text{col}(\mathbf{Q})$ experiences the same 25% amplitude shrinkage.

---

## Section S2: Upstream Foundation Model Pre-Training Corpus Audit

BrainOmni (OpenTSLab, 2025) was trained on 2,653.8 hours of electrophysiological data across 22 datasets. Table S1 details the complete breakdown of datasets, recording modalities, and sensor technologies.

**Table S1:** Upstream pre-training dataset composition for BrainOmni.

| Dataset Name | Modality | Duration (Hours) | Sensor System & Technology | Sensor Type Designation |
| :--- | :--- | :--- | :--- | :--- |
| **Cam-CAN** | MEG | 176.4 | Elekta Neuromag-306 (102 Mag, 204 Planar Grad) | `MAG=1, GRAD=2` |
| **OMEGA** | MEG | 148.2 | CTF-275 (Radial Axial Gradiometers) | `GRAD=2` |
| **THINGS-MEG** | MEG | 112.5 | CTF-275 (Radial Axial Gradiometers) | `GRAD=2` |
| **MEG-MASC** | MEG | 84.1 | KIT / Yokogawa (Axial Gradiometers) | `GRAD=2` |
| **MIND** | MEG | 45.0 | Elekta Neuromag-306 (102 Mag, 204 Planar Grad) | `MAG=1, GRAD=2` |
| **SMN4Lang** | MEG | 36.8 | Elekta Neuromag-306 (102 Mag, 204 Planar Grad) | `MAG=1, GRAD=2` |
| **MEG-Narrative** | MEG | 28.6 | CTF-275 (Radial Axial Gradiometers) | `GRAD=2` |
| **Kymata** | MEG | 14.2 | Elekta Neuromag-306 (102 Mag, 204 Planar Grad) | `MAG=1, GRAD=2` |
| **NeuroMorph** | MEG | 6.8 | KIT / Yokogawa (Axial Gradiometers) | `GRAD=2` |
| **AversiveMEG** | MEG | 4.2 | CTF-275 (Radial Axial Gradiometers) | `GRAD=2` |
| *Total MEG* | *MEG* | *656.8* | *100% Cryogenic SQUID (>4:1 Gradiometers)* | *`GRAD` Dominant* |
| **TUH EEG Corpus** | EEG | 1,240.0 | Standard 10-20 Scalp Electrodes | `EEG=0` |
| **CHB-MIT** | EEG | 320.0 | Scalp Electrodes | `EEG=0` |
| **Seed / Seed-IV** | EEG | 185.0 | 62-channel Scalp Electrodes | `EEG=0` |
| **9 Other EEG Corpora** | EEG | 252.0 | Scalp Electrodes | `EEG=0` |
| *Total Pre-Training* | *MEG + EEG* | *2,653.8* | *0.0 Hours OPM-MEG Recorded* | *--* |

**Key Pre-Training Insights:**
1. Gradiometers outnumber magnetometers by greater than 4:1 across all MEG training hours.
2. Zero hours of on-scalp optically pumped magnetometer (OPM) data were present during upstream training.

---

## Section S3: Extended Cross-Subject Transfer Trajectories

To assess downstream linear probing data-efficiency under low-to-medium sample budgets, models were evaluated across $B \in \{1, 2, 4, 10, 20\}$ subjects on OpenNeuro `ds005107` (256 to 5,120 single trials).

**Table S2:** Complete cross-subject balanced accuracy (%) and 95% bootstrap confidence intervals across four facial stimulus dimensions on `ds005107`.

| Stimulus Attribute | Subject Budget ($B$) | Pretrained Linear Probe | Control A (Random Init) | Control B (Tokenizer Only) | CompactMEGNet | Classical Baseline | Statistical Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Face Age** | 1 (256 trials) | 49.50% [48.55, 50.49] | 50.41% [49.40, 51.38] | 49.04% [48.47, 49.59] | 49.25% [48.18, 50.41] | 49.13% [48.07, 50.12] | Equivalent |
| | 2 (512 trials) | 49.19% [48.46, 50.01] | 50.40% [49.59, 51.24] | 49.51% [48.52, 50.45] | 50.80% [49.97, 51.74] | 50.01% [49.15, 50.78] | Equivalent |
| | 4 (1,024 trials) | 48.94% [48.10, 49.86] | 50.86% [49.83, 51.83] | 50.08% [49.40, 50.82] | 50.70% [49.92, 51.55] | 50.63% [49.68, 51.56] | Inconclusive |
| | 10 (2,560 trials) | 50.08% [49.02, 51.06] | 50.27% [49.11, 51.42] | 49.72% [48.54, 50.92] | 50.73% [49.81, 51.60] | 50.10% [49.12, 51.08] | Equivalent |
| | 20 (5,120 trials) | 49.57% [48.44, 50.73] | 50.68% [49.28, 52.06] | 50.01% [48.55, 51.45] | **51.85% [50.33, 53.34]** | 50.56% [49.28, 51.79] | CompactMEGNet Superior ($p < 0.025$) |
| **Face Sex** | 1 (256 trials) | 49.45% [48.20, 50.68] | 49.68% [48.52, 50.80] | 49.88% [48.70, 51.10] | 49.00% [47.50, 50.40] | 49.50% [48.20, 50.80] | Equivalent |
| | 2 (512 trials) | 49.62% [48.40, 50.85] | 49.80% [48.60, 51.00] | 50.12% [48.90, 51.30] | 50.20% [48.90, 51.50] | 49.90% [48.60, 51.20] | Equivalent |
| | 4 (1,024 trials) | 49.80% [48.65, 50.95] | 49.10% [47.90, 50.30] | 50.40% [49.20, 51.60] | 50.45% [49.10, 51.80] | 50.20% [48.90, 51.50] | Equivalent |
| | 10 (2,560 trials) | 50.15% [48.90, 51.40] | 48.60% [47.30, 49.90] | 50.80% [49.60, 52.00] | 50.90% [49.60, 52.20] | 50.80% [49.40, 52.20] | Equivalent |
| | 20 (5,120 trials) | 49.59% [48.47, 50.75] | 47.97% [46.92, 49.05] | 51.14% [50.05, 52.35] | 51.06% [49.41, 52.64] | 51.06% [49.41, 52.64] | Inconclusive |
| **Face Ancestry**| 20 (5,120 trials) | 51.22% [49.81, 52.62] | 48.95% [47.33, 50.65] | 51.45% [50.37, 52.55] | -- | 50.67% [49.22, 52.12] | Inconclusive |
| **Face Emotion** | 20 (5,120 trials) | 50.24% [49.02, 51.48] | 49.85% [48.60, 51.10] | 50.05% [48.80, 51.30] | 50.40% [49.10, 51.70] | 50.15% [48.80, 51.50] | Equivalent |

---

## Section S4: Triaxial Nottingham Full Factorial Matrix

**Table S3:** Complete $2 \times 3$ factorial evaluation across 10 adult participants with 192-channel triaxial OPMs (Zenodo 11126593, 84 trials/subject = 840 trials total, 10,000 subject-level bootstrap resamples).

| Architecture Mode | Filtering Condition | Channel Count | Zero-Shot PCC [95% CI] | Braille Decoding BACC [95% CI] | Paired Test vs Control |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Monolithic Triaxial** | Raw (Unfiltered) | 155–176 | 0.3726 [0.3416, 0.4046] | 48.24% [43.64, 52.67] | Monolithic Raw Baseline |
| **Monolithic Triaxial** | Soft-HFC ($\alpha=0.25$) | 155–176 | 0.3461 [0.3115, 0.3816] | 49.07% [44.99, 52.89] | Monolithic Soft-HFC Baseline |
| **Random 3-way Split** | Raw (Unfiltered) | 3 × ~55 | 0.3995 [0.3697, 0.4287] | 50.83% [46.31, 54.90] | $\Delta = +0.0269$ vs Raw Mono ($p = 0.0016$) |
| **Random 3-way Split** | Soft-HFC ($\alpha=0.25$) | 3 × ~55 | 0.3753 [0.3421, 0.4076] | 48.51% [43.47, 53.00] | $\Delta = +0.0292$ vs HFC Mono ($p = 0.0016$) |
| **Radial Isolation ($Z$)** | Raw (Unfiltered) | 55–60 | 0.4944 [0.4667, 0.5265] | 47.19% [40.76, 53.44] | $\Delta = +0.1218$ vs Raw Mono ($p = 0.0016$) |
| **Tangential Only ($X/Y$)** | Raw (Unfiltered) | 100–116 | 0.4446 [0.4112, 0.4771] | 49.35% [46.06, 52.51] | $\Delta = +0.0720$ vs Raw Mono ($p = 0.0016$) |
| **Axis-Partitioned ($Z/X/Y$)** | Raw (Unfiltered) | 155–176 | **0.5130 [0.4840, 0.5432]** | 47.53% [43.22, 51.88] | **$\Delta = +0.1135$ vs Random Raw ($p = 0.0016$)** |
| **Axis-Partitioned ($Z/X/Y$)** | Soft-HFC ($\alpha=0.25$) | 155–176 | **0.4849 [0.4507, 0.5199]** | 50.19% [43.72, 56.40] | **$\Delta = +0.1096$ vs Random HFC ($p = 0.0016$)** |

**Disentanglement Analysis:**
- Under Soft-HFC: $\Delta_{\text{capacity}} = +0.0292$ (21.0%), $\Delta_{\text{orthogonal}} = +0.1096$ (79.0%). Total gain = $+0.1388$.
- Under Raw: $\Delta_{\text{capacity}} = +0.0269$ (19.2%), $\Delta_{\text{orthogonal}} = +0.1135$ (80.8%). Total gain = $+0.1404$.
- In both regimes, approximately 80% of the spatial bottleneck is attributable to directional vector interference.

---

## Section S5: Model Architectures & Implementation Details

### S5.1 BrainOmni Architecture Specifications
- **Backbone Parameters:** 8.4M parameters (`tiny` configuration).
- **Temporal Encoder (`SEANet`):** 1D multi-scale temporal convolutions mapping $L=512$ raw audio/electrophysiological samples to temporal token slices.
- **Spatial Tokenizer (`BrainSensorModule`):** Cross-attention module with $N_{\text{neuro}} = 16$ learnable latent queries. Position embeddings $\mathbf{p} \in \mathbb{R}^3$, unit orientation $\mathbf{u} \in \mathbb{R}^3$, and integer sensor type tags $\{0, 1, 2\}$ projected into query space.
- **Quantization:** Residual vector quantization (RVQ) with codebook size 1,024.
- **Sequence Model:** 12-layer causal Transformer backbone ($d_{\text{model}} = 256, n_{\text{head}} = 8, d_{\text{ffn}} = 1,024$).

### S5.2 CompactMEGNet Specialist Baseline
- **Parameter Count:** 1,514 trainable parameters.
- **Architecture:** Depthwise-separable convolutional network tailored for multichannel MEG:
  1. 2D Temporal Convolution: Kernel size $(1, 32)$, 8 filters, valid padding.
  2. Depthwise Spatial Convolution: Kernel size $(C, 1)$, depth multiplier 2 (16 spatial filters), batch normalization, exponential linear unit (ELU).
  3. Average Pooling: Kernel $(1, 4)$, dropout rate 0.25.
  4. Separable Convolution: Kernel $(1, 16)$, 16 pointwise filters, batch normalization, ELU.
  5. Average Pooling: Kernel $(1, 8)$, dropout rate 0.25.
  6. Dense Linear Classification Head with cross-entropy loss.

### S5.3 Linear Probe Classifier
- Ridge and $L_2$-regularized logistic regression fitted via scikit-learn.
- Regularization parameter $C$ tuned via 5-fold inner cross-validation over the log grid $C \in \{10^{-4}, 10^{-3}, 10^{-2}, 10^{-1}, 1, 10, 100, 1000, 10000\}$.

---

## Section S6: Provenance, Environment, and Checksum Hashes

### S6.1 Upstream Checkpoint and Configuration Hashes
- **Hugging Face Repository:** `OpenTSLab/BrainOmni`
- **Revision Commit Hash:** `9a4d3c70495370397ccfbfd6d2496f25647545a5`
- **Model Checkpoint (`tiny/BrainOmni.pt`):**
  - SHA-256: `62c67ba6a84ea0625e67a3b5e7463fe3930bfee88a612a225e9062a052542ffc`
- **Model Configuration (`tiny/model_cfg.json`):**
  - SHA-256: `4b994b38f1a8dccc8136f2b837a8ad2d3ebb028a1af0c527f9e0902a1a5c252e`

### S6.2 Configuration File Hashes
- `configs/ds005107_primary.yaml`:
  - SHA-256: `ef7187c8942800e45402ed2ff8e9fc22a6776c0af31a2c391cee6eec2d3162f3`
- `configs/nottingham_adults.yaml`:
  - SHA-256: `19817ef73b7e0dfebac56bc1dd55043a2de47205dffad015c2103ee5d96bd8a0`

### S6.3 Dataset Accessions and DOIs
- **OpenNeuro ds005107:** Version 1.0.0, DOI: `10.18112/openneuro.ds005107.v1.0.0`
- **Nottingham Triaxial Cohort:** Zenodo repository, DOI: `10.5281/zenodo.11126593` (Rier & Rhodes et al., eLife 2024).
- **OpenNeuro ds000117:** Version 1.0.4, DOI: `10.18112/openneuro.ds000117.v1.0.4` (Wakeman & Henson, Scientific Data 2015).

### S6.4 Hardware and Software Environment
- **Operating System:** macOS (Darwin 24.6.0, Apple Silicon arm64).
- **Python Version:** 3.12.12.
- **PyTorch:** 2.15.0 with Metal Performance Shaders (MPS) acceleration.
- **MNE-Python:** 1.9.0.
- **Scikit-Learn:** 1.6.1.
- **NumPy:** 2.2.3.
- **SciPy:** 1.15.2.
- **Package Management:** Fully reproducible virtual environment managed via `uv` (`uv.lock`).
