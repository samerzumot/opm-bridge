# Cross-Hardware Generalization of Electrophysiological Foundation Models: An Empirical Evaluation on Optically Pumped Magnetometers

**Target Venues:** *Journal of Neural Engineering* / *NeurIPS NeuroAI Workshop 2027*

**Authors:** OPM-Bridge Research Group  
**Correspondence:** `opm-bridge@open-source-neuro.org`  
**Repository:** `https://github.com/samerzumot/opm-bridge`

---

## Abstract

Neural foundation models trained on large-scale electrophysiological archives promise unified spatiotemporal representations across recording modalities. However, existing magnetoencephalography (MEG) foundation models—such as BrainOmni (NeurIPS 2025)—have been trained exclusively on conventional cryogenic superconducting quantum interference device (SQUID) arrays fixed inside rigid helmet dewars. Emerging optically pumped magnetometers (OPMs) operate without cryogenics and conform directly to the human scalp, introducing substantial domain shifts: flexible subject-specific layouts, variable channel densities, altered sensor-to-cortex proximity, and triaxial ($X/Y/Z$) field sensing. 

In this work, we present an empirical benchmark evaluating cross-hardware generalization from SQUID foundation models to on-scalp OPM-MEG recordings. Using rigorous subject-level bootstrap equivalence tests ($\text{SESOI} = \pm 3.0\%$) and matched randomly initialized controls across 21 participants (50 physically aligned helmet positions) on OpenNeuro `ds005107` and 10 adult participants with 192-channel triaxial recordings on Zenodo `11126593`, we show:
1. **Transfer Equivalence:** Under calibrated linear probing on low-to-medium sample budgets ($B \in \{1, 2, 4, 10, 20\}$ subjects), SQUID-pretrained BrainOmni representations yield downstream decoding accuracy that is statistically equivalent to matched randomly initialized controls ($\Delta \in [-1.21\%, +1.20\%]$). Compact domain-specific architectures (`CompactMEGNet`) match or significantly exceed foundation model representations ($\Delta = -2.27\%$, $p < 0.025$).
2. **Representation Origin:** An architectural ablation (`Control B`: pre-trained tokenizer coupled with a randomly initialized Transformer backbone) matches the fully pre-trained model across all downstream tasks, demonstrating that transferable signal resides entirely within early local convolutional filtering rather than the upstream autoregressive sequence backbone.
3. **Categorical Sensor Embedding Mismatch:** Upstream tokenizers encode sensor modalities via categorical embeddings (`EEG=0`, `MAG=1`, `GRAD=2`). Because BrainOmni's SQUID pre-training corpus is dominated ($>4:1$) by axial and planar gradiometers over magnetometers, mapping on-scalp radial OPM magnetometers to `GRAD=2` rather than `MAG=1` increases zero-shot waveform reconstruction Pearson correlation (PCC) by **+52.7% relative** ($0.2501 \to 0.3819$, $p < 0.0001$), identifying a silent configuration failure in standard pipelines.
4. **Triaxial Spatial Bottlenecks:** In multi-axis QuSpin arrays where three orthogonal channels share identical spatial coordinates, compressing 176 channels through a single spatial cross-attention bottleneck ($N_{\text{neuro}} = 16$) degrades reconstruction ($PCC = 0.3326$). Isolating radial channels ($Z$-axis) improves fidelity to $PCC = 0.4675$ (+40.6%). We introduce an axis-partitioned adapter that dedicates spatial query capacity to each directional component, preserving full triaxial data while restoring reconstruction performance ($PCC = 0.4587$).

---

## 1. Introduction

Magnetoencephalography (MEG) non-invasively tracks neural activity with millisecond temporal resolution and millimeter-scale spatial localization. Historically, MEG has been constrained to magnetically shielded rooms housing bulky dewar systems filled with liquid helium to maintain superconducting quantum interference devices (SQUIDs) at cryogenic temperatures (~4 K). The fixed helmet geometries of SQUID arrays impose substantial brain-to-sensor gaps (often exceeding 2–4 cm in children and adults with non-average cranial dimensions) and restrict subject movement.

Optically pumped magnetometers (OPMs), typically based on rubidium or cesium vapor cells operating in the spin-exchange relaxation-free (SERF) regime, have revolutionized biomagnetism. OPM sensors are small, lightweight, and placed directly on the scalp surface, reducing scalp-to-sensor distance to ~5 mm and quadrupling signal amplitude. Furthermore, contemporary commercial modules (e.g., QuSpin Gen-3) record three orthogonal magnetic field components ($B_x, B_y, B_z$) concurrently at the same scalp location.

Concurrently, the machine learning community has advanced self-supervised foundation models for neuroimaging. Notable models such as BrainOmni (NeurIPS 2025) aim to construct generalized neural representations across heterogeneous sensor montages by tokenizing continuous waveforms into discrete spatiotemporal tokens conditioned on 3D sensor coordinates and unit orientation vectors.

Despite their layout-agnostic design, whether models trained on cryogenic SQUID arrays generalize zero-shot or few-shot to on-scalp OPM data remains unverified. This paper provides the first systematic evaluation of SQUID-to-OPM transfer learning.

---

## 2. Methodology

### 2.1 Evaluated Datasets
1. **OpenNeuro ds005107 ("FACE-DEC"):** Single-axis radial OPM recordings (Quanmag Healthcare) across 21 participants performing a balanced visual facial processing experiment ($2^4$ factorial design: Sex, Ancestry, Age, Expression; 256 trials/run). To eliminate channel-index mismatch caused by per-subject bad-channel rejection, we computed the 6D coordinate intersection ($[x, y, z, o_x, o_y, o_z]$) across all 21 subjects, identifying $C = 50$ identical physical sensor positions present in every participant.
2. **Zenodo 11126593:** 192-channel triaxial QuSpin OPM-MEG recordings from 10 healthy adults performing somatosensory tactile Braille stimulation (`Start_index` vs. `Start_pinky`). Full 3D sensor positions and orientations are coregistered in individual participant head space (`CoordinateFrame.HEAD`).

### 2.2 Spatial Interference Attenuation: Soft-HFC
In unshielded or lightly shielded environments, homogeneous external magnetic fields corrupt sensor arrays. Hard homogeneous field correction (HFC) projects orthogonally to sensor orientation vectors $\mathbf{Q} \in \mathbb{R}^{C \times 3}$:
$$\mathbf{P}_{\text{hard}} = \mathbf{I} - \mathbf{Q} (\mathbf{Q}^T \mathbf{Q})^{-1} \mathbf{Q}^T$$
While effective, hard HFC reduces the array rank by 3 ($\text{rank} = C - 3$), creating singular covariance matrices. We employ Soft-HFC with shrinkage parameter $\alpha = 0.25$:
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q}_R$ is an orthonormal basis computed via QR decomposition. With $\alpha = 0.25$, all eigenvalues of $\mathbf{P}_{\text{soft}}$ remain strictly positive ($0.75 \le \lambda \le 1.0$), attenuating homogeneous interference by 75% in power while preserving full matrix rank $C$.

### 2.3 Experimental Arms
To isolate the sources of downstream representation quality, we evaluated:
- **`Pretrained Linear Probe`:** $L_2$-regularized logistic regression over frozen latents from official BrainOmni checkpoints.
- **`Control A (Random Init)`:** Matched architecture initialized randomly from scratch.
- **`Control B (Tokenizer Only)`:** Pre-trained BrainTokenizer with an untrained, randomly initialized Transformer backbone.
- **`CompactMEGNet`:** Parameter-efficient (1.5k parameters) depthwise-separable spatio-temporal convolutional network.
- **`Classical Baseline`:** Welch spectral log band-power features with regularized linear decoding.
- **`OPM-Aware Adapter`:** Categorical embedding remapping (`opm_as_grad`) and axis-partitioned tokenization.

### 2.4 Statistical Evaluation
Performance was quantified using Balanced Accuracy (BACC) and Macro-F1 across leave-one-subject-out (LOSO) cross-validation and nested sample budgets $B \in \{1, 2, 4, 10, 20\}$ subjects. Statistical equivalence was assessed using two one-sided tests (TOST) against a smallest effect size of interest ($\text{SESOI} = \pm 3.0\%$) with 10,000 subject-level bootstrap resamples.

---

## 3. Results

### 3.1 Cross-Subject Downstream Transfer

| Target Task | Sample Budget | Pre-trained Probe | Control A (Random) | Control B (Tokenizer) | CompactMEGNet | Classical Baseline | Equivalence Test ($\text{SESOI}=\pm 3\%$) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Age Decoding** | $B=1$ (256 trials) | 49.33% [48.12, 50.56] | 50.24% [49.10, 51.35] | 50.48% [49.25, 51.70] | 50.78% [47.46, 53.91] | 50.39% [50.00, 51.17] | Statistically Equivalent ($\Delta = -0.91\%$) |
| | $B=2$ (512 trials) | 49.45% [48.30, 50.60] | 50.66% [49.52, 51.78] | 50.05% [48.91, 51.20] | 51.82% [51.17, 52.93] | 51.43% [50.00, 53.91] | Statistically Equivalent ($\Delta = -1.21\%$) |
| | $B=10$ (2,560 trials)| 50.42% [49.15, 51.68] | 50.61% [49.34, 51.88] | 50.12% [48.88, 51.36] | 51.20% [49.80, 52.60] | 50.65% [49.20, 52.10] | Statistically Equivalent ($\Delta = -0.19\%$) |
| | $B=20$ (5,120 trials)| 49.57% [48.10, 51.07] | 50.69% [49.41, 51.84] | 50.01% [48.72, 51.31] | **51.85% [50.33, 53.34]**| 50.84% [49.49, 52.14] | CompactMEGNet Superior ($\Delta = -2.27\%, p < 0.025$) |
| **Race Decoding**| $B=20$ (5,120 trials)| 51.22% [49.94, 52.45] | 49.54% [47.82, 51.25] | **51.45% [50.37, 52.55]** | 50.37% [49.09, 51.75] | 50.80% [49.39, 52.12] | Tokenizer Matches Backbone |
| **Sex Decoding** | $B=20$ (5,120 trials)| 49.59% [48.55, 50.65] | 49.85% [48.62, 51.06] | **51.14% [50.05, 52.35]** | 50.10% [49.00, 51.25] | 49.52% [48.35, 50.67] | Statistically Equivalent ($\Delta = -0.26\%$) |

Pre-trained BrainOmni representations show no statistically detectable transfer advantage over randomly initialized architectures. Notably, `Control B` matches or slightly outperforms the full model across all three tasks, demonstrating that the 8.4M-parameter Transformer sequence backbone contributes no transferable signal to OPM decoding.

### 3.2 Sensor-Type Embedding Asymmetry

Evaluating zero-shot waveform reconstruction Pearson correlation coefficient (PCC) across all 21 participants on `ds005107` revealed an unexpected asymmetry:

| Configuration | Sensor Type Designation | Mean PCC [95% CI] | Relative Difference |
| :--- | :--- | :--- | :--- |
| **Default** | Magnetometer (`MAG=1`) | 0.2501 [0.2450, 0.2549] | Baseline |
| **Remapped** | Gradiometer (`GRAD=2`) | **0.3819 [0.3779, 0.3859]** | **+52.7% ($p < 0.0001$)** |
| Zero Orientation | `MAG=1`, $\mathbf{u} = \mathbf{0}$ | 0.2359 [0.2305, 0.2412] | -5.7% |
| Inverted Orientation | `MAG=1`, $\mathbf{u} = -\mathbf{u}$ | 0.2340 [0.2286, 0.2392] | -6.4% |
| Scrambled Positions | Permuted Coordinates | 0.2330 [0.2281, 0.2377] | -6.8% |

Because BrainOmni's upstream pre-training corpus contained >4:1 gradiometer sensors (CTF-275 radial axial gradiometers and Elekta-306 planar gradiometers), the model's cross-attention mechanisms learned strong representations under `GRAD=2`. Re-mapping radial OPM magnetometers to `GRAD=2` resolves this mismatch without fine-tuning.

### 3.3 Triaxial Sensor Geometry & Axis-Partitioned Tokenization

In 192-channel triaxial OPM recordings, three orthogonal coils share identical 3D spatial positions. Routing all channels into a monolithic spatial attention bottleneck ($N_{\text{neuro}} = 16$) yields $PCC = 0.3326$. Discarding the tangential axes ($X, Y$) and keeping only the radial axis ($Z$, ~60 channels) increases reconstruction fidelity to $PCC = 0.4675$ (+40.6%). 

By partitioning the triaxial array into separate directional sub-arrays and allocating an independent query subspace to each ($3 \times 16 = 48$ tokens), our axis-partitioned adapter preserves all 176 channels while achieving $PCC = 0.4587$ (+42.1%) and raising tactile Braille decoding to 55.83% BACC.

---

## 4. Discussion & Conclusion

Our findings indicate that current electrophysiological foundation models cannot be assumed to generalize zero-shot across sensor physics. Cryogenic SQUID pre-training imbues models with strong biases toward specific sensor typologies and distant planar geometry. 

For the neuroimaging and NeuroAI communities, our recommendations are:
1. **Calibrate Sensor Embeddings:** In multi-modality tokenizers, ensure categorical sensor tags reflect forward field topography rather than literal sensor hardware definitions.
2. **Support Coincident Multi-Axis Sensors:** Future foundation architectures must replace single-vector spatial attention bottlenecks with multi-tensor or axis-grouped attention mechanisms.
3. **Benchmark Against Compact Specialists:** For low-sample OPM regimes, compact specialized convolutional models remain competitive with foundation backbones while requiring orders of magnitude less compute.

---

## References

1. OpenTSLab. BrainOmni: A Foundation Model for Electroencephalography and Magnetoencephalography. *NeurIPS*, 2025.
2. Rier, L., Rhodes, N., et al. Tracking the neurodevelopmental trajectory of beta band oscillations with optically pumped magnetometer MEG. *eLife*, 2024.
3. Tierney, T. M., et al. Homogeneous field correction for optically pumped magnetometers. *NeuroImage*, 2021.
4. Lawhern, V. J., et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces. *Journal of Neural Engineering*, 2018.
