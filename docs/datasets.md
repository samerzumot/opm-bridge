# Dataset Catalog and Harmonization Protocol

This document details the public optically pumped magnetometer (OPM) datasets evaluated in this benchmark, including sensor specifications, coordinate systems, task structures, and data handling protocols.

---

## 1. Primary Benchmark Datasets

### 1.1 OpenNeuro ds005107 ("FACE-DEC")
- **Source:** OpenNeuro accession `ds005107`.
- **Participants:** 21 adult human participants (`sub-01` through `sub-23`; subjects `05` and `12` omitted in original repository).
- **Instrumentation:** 64-channel radial optically pumped magnetometer array (Quanmag Healthcare). Sampling rate: 1,000 Hz.
- **Hardware Layout:** Sensors housed in a rigid helmet. In the public BIDS release, sensor locations were approximated using standard Elekta Neuromag sensor positions (`CoordinateFrame.APPROX_ELEKTA`).
- **Experimental Task:** Rapid visual presentation of facial stimuli across a balanced $2^4$ factorial design spanning demographic and expressive dimensions:
  - Sex / Gender: Female vs. Male
  - Race / Ancestry: Asian vs. Caucasian
  - Age: Young vs. Older
  - Expression: Neutral vs. Emotional
- **Trial Structure:** 256 stimulus trials per run, with a +33 ms display projector latency correction applied to stimulus trigger events.
- **Preprocessing:** 1.0–96.0 Hz fourth-order zero-phase Butterworth bandpass filter, notch filters at 44 Hz (OPM internal cycle) and 50 Hz power-line harmonics, followed by decimation to 256 Hz.

### 1.2 Zenodo 11126593 (Nottingham Triaxial OPM-MEG)
- **Source:** Zenodo repository `10.5281/zenodo.11126593` (Rier & Rhodes et al., eLife 2024).
- **Participants:** 10 adult human participants (`sub-101` through `sub-110`) evaluated in this benchmark.
- **Instrumentation:** Up to 192 triaxial OPM channels utilizing Gen-3 QuSpin zero-field magnetometers (QZFMs). Each sensor module measures three orthogonal magnetic field components ($B_x, B_y, B_z$) at a shared spatial position.
- **Coordinate Framework:** Full 3D sensor positions ($P_x, P_y, P_z$) and individual 3D unit orientation vectors ($O_x, O_y, O_z$) coregistered in individual participant head space (`CoordinateFrame.HEAD`) via optical scalp scanning and digitized fiducial landmarks.
- **Experimental Task:** Somatosensory tactile Braille stimulation applied to the index finger (`Start_index`, class 0) or the little finger (`Start_pinky`, class 1) of the right hand.
- **Preprocessing:** Notch filtered at 50 Hz and 100 Hz, bandpass filtered at 1.0–45.0 Hz, epoched over $[-0.2\text{ s}, +1.8\text{ s}]$, baseline corrected relative to $[-0.2\text{ s}, 0.0\text{ s}]$, and resampled to 256 Hz.

---

## 2. Harmonization and Sensor Alignment

### 2.1 Geometric Sensor Matching
In clinical and experimental OPM recordings, bad-channel rejection typically discards differing subsets of noisy or unlocked sensors across recording sessions. Performing naive channel slicing (e.g., keeping the first $K$ channels) causes arbitrary channel shuffling across participants.

To resolve this, our harmonization pipeline computes the exact coordinate intersection across all participants in a cohort:
```python
aligned_X, aligned_pos, aligned_st = align_subjects_by_sensor_geometry(
    raw_X, raw_pos, raw_st, decimals=4
)
```
For `ds005107`, this procedure identifies $C = 50$ identical physical sensor coordinates present in all 21 participants, ensuring that channel index $k$ corresponds strictly to the same physical sensor across all subjects.

### 2.2 Soft Homogeneous Field Correction (Soft-HFC)
External magnetic interference in non-cryogenic shielded rooms often manifests as a uniform, homogeneous spatial field $B_{\text{ext}} = [B_x, B_y, B_z]^T$. 

Standard hard HFC projects orthogonal to the sensor orientation matrix $\mathbf{Q} \in \mathbb{R}^{C \times 3}$:
$$\mathbf{P}_{\text{hard}} = \mathbf{I} - \mathbf{Q} (\mathbf{Q}^T \mathbf{Q})^{-1} \mathbf{Q}^T$$
Hard HFC reduces the array rank by 3 ($\text{rank} = C - 3$), creating singular covariance matrices. 

We apply Soft-HFC with shrinkage parameter $\alpha = 0.25$:
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q}_R$ is obtained via thin QR decomposition ($\mathbf{Q} = \mathbf{Q}_R \mathbf{R}$). This formulation guarantees:
1. Full numerical rank ($\text{rank} = C$).
2. 43.75% power attenuation (25% amplitude shrinkage at $\alpha = 0.25$) of uniform external magnetic interference. Any neural signal projecting onto $\text{col}(\mathbf{Q})$ experiences identical fractional shrinkage.
3. Zero distortion of neural signals orthogonal to the uniform field subspace.
