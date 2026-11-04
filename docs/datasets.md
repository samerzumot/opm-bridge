# Dataset Specifications and Harmonization Protocols

This document specifies the neuroimaging datasets evaluated in this benchmark, detailing sensor geometry, coordinate frames, experimental paradigms, preprocessing pipelines, and spatial harmonization procedures.

---

## 1. Benchmark Datasets

### 1.1 OpenNeuro ds005107: Radial Single-Axis OPM-MEG
- **Accession:** OpenNeuro `ds005107` (Visual Facial Processing).
- **Cohort:** 21 adult human participants (`sub-01` through `sub-23`; subjects `05` and `12` omitted in the source repository).
- **Instrumentation:** 64-channel radial optically pumped magnetometer array (Quanmag Healthcare), sampled natively at 1,000 Hz.
- **Sensor Geometry:** Rigid helmet mounting. Sensor positions in the public BIDS release correspond to Elekta Neuromag fiducial approximation coordinates (`CoordinateFrame.APPROX_ELEKTA`).
- **Experimental Task:** Rapid visual presentation of facial stimuli across a balanced $2^4$ factorial design spanning demographic and affective dimensions:
  - Sex: Female versus Male
  - Ancestry: Asian versus Caucasian
  - Age: Young versus Older
  - Expression: Neutral versus Emotional
- **Trial Structure:** 256 stimulus trials per subject run, incorporating a +33 ms display projector latency correction applied to stimulus trigger timestamps.
- **Preprocessing:** Fourth-order zero-phase Butterworth bandpass filter (1.0–96.0 Hz), notch filters at 44 Hz (OPM internal modulation frequency) and 50 Hz power-line harmonics, followed by decimation to 256 Hz.

### 1.2 Zenodo 11126593: Triaxial OPM-MEG
- **Accession:** Zenodo repository `10.5281/zenodo.11126593` (Rier & Rhodes et al., eLife 2024).
- **Cohort:** 10 adult human participants (`sub-101` through `sub-110`) evaluated in this benchmark.
- **Instrumentation:** Up to 192 triaxial OPM channels utilizing Gen-3 QuSpin zero-field magnetometers (QZFMs). Each triaxial sensor module simultaneously measures three orthogonal field components ($B_x, B_y, B_z$) at a shared spatial position.
- **Coordinate Framework:** Three-dimensional sensor positions ($P_x, P_y, P_z$) and individual unit orientation vectors ($O_x, O_y, O_z$) coregistered in individual subject head space (`CoordinateFrame.HEAD`) via optical 3D surface scanning and anatomical fiducial digitization.
- **Experimental Task:** Somatosensory tactile Braille stimulation delivered to either the index finger (`Start_index`, class 0) or the little finger (`Start_pinky`, class 1) of the right hand.
- **Preprocessing:** Notch filtering at 50 Hz and 100 Hz, bandpass filtering from 1.0 to 45.0 Hz, epoching over $[-0.2\text{ s}, +1.8\text{ s}]$ with baseline correction relative to $[-0.2\text{ s}, 0.0\text{ s}]$, and decimation to 256 Hz.

### 1.3 OpenNeuro ds000117: In-Domain Cryogenic SQUID Baseline
- **Accession:** OpenNeuro `ds000117` (Wakeman & Henson, Scientific Data 2015).
- **Cohort:** Evaluated on participant `sub-01` across 5-fold cross-validation.
- **Instrumentation:** 306-channel cryogenic SQUID system (Elekta Neuromag-306), consisting of 102 magnetometers and 204 planar gradiometers.
- **Experimental Task:** Visual presentation of familiar faces, unfamiliar faces, and scrambled control images. Serves as the matched in-domain positive control for BrainOmni pre-trained representations.
- **Preprocessing:** Native continuous MEG epoched to match the visual evaluation window ($[-0.2\text{ s}, +0.6\text{ s}]$), bandpass filtered (1.0–96.0 Hz), and resampled to 256 Hz.

---

## 2. Harmonization and Sensor Alignment

### 2.1 Geometric Sensor Matching
In clinical and experimental OPM recordings, channel rejection procedures discard differing subsets of noisy or unlocked sensors across recording sessions. Selecting arbitrary channel indices (e.g., retaining the first $K$ channels) causes coordinate misalignment across participants.

To guarantee topological consistency, our harmonization pipeline computes the exact coordinate intersection across all participants in a cohort:
```python
aligned_X, aligned_pos, aligned_st = align_subjects_by_sensor_geometry(
    raw_X, raw_pos, raw_st, decimals=4
)
```
For `ds005107`, this procedure identifies $C = 50$ identical physical sensor coordinates present across all 21 participants, ensuring that channel index $k$ corresponds strictly to the same physical sensor across all subjects.

### 2.2 Soft Homogeneous Field Correction (Soft-HFC)
External magnetic interference in non-cryogenic magnetically shielded environments frequently presents as a spatially homogeneous ambient field $B_{\text{ext}} = [B_x, B_y, B_z]^T$. 

Conventional hard Homogeneous Field Correction (HFC) projects the recorded data orthogonal to the sensor orientation matrix $\mathbf{Q} \in \mathbb{R}^{C \times 3}$:
$$\mathbf{P}_{\text{hard}} = \mathbf{I} - \mathbf{Q} (\mathbf{Q}^T \mathbf{Q})^{-1} \mathbf{Q}^T$$
Hard HFC strictly reduces array rank by 3 ($\text{rank} = C - 3$), which produces singular covariance matrices in downstream estimators.

To preserve numerical stability, we implement Soft-HFC with a continuous shrinkage parameter $\alpha = 0.25$:
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q}_R$ is an orthonormal basis obtained via thin QR decomposition ($\mathbf{Q} = \mathbf{Q}_R \mathbf{R}$). This formulation provides three properties:
1. Full numerical rank is preserved ($\text{rank} = C$).
2. Homogeneous external interference is attenuated by 43.75% in power (25% amplitude shrinkage at $\alpha = 0.25$), with an identical fractional reduction applying only to neural components collinear with $\text{col}(\mathbf{Q})$.
3. Signal components orthogonal to the uniform interference subspace remain unperturbed.
