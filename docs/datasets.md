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
- **Preprocessing:** Zero-phase FIR bandpass filter (1.0–96.0 Hz), using MNE firwin design with a Hamming window, notch filters at 44 Hz (OPM internal modulation frequency) and 50 Hz power-line harmonics, followed by epochs from -0.2 to 1.8 seconds, baseline correction from -0.2 to 0 seconds with linear detrending, and anti-aliased resampling to 256 Hz.

### 1.2 Zenodo 11126593: Triaxial OPM-MEG
- **Accession:** Zenodo repository `10.5281/zenodo.11126593` (Rier & Rhodes et al., eLife 2024).
- **Cohort:** 10 adult human participants (`sub-101` through `sub-110`) evaluated in this benchmark.
- **Instrumentation:** Up to 192 triaxial OPM channels utilizing Gen-3 QuSpin zero-field magnetometers (QZFMs). Each triaxial sensor module simultaneously measures three orthogonal field components ($B_x, B_y, B_z$) at a shared spatial position.
- **Coordinate Framework:** Three-dimensional sensor positions ($P_x, P_y, P_z$) and individual unit orientation vectors ($O_x, O_y, O_z$) coregistered in individual subject head space (`CoordinateFrame.HEAD`) via optical 3D surface scanning and anatomical fiducial digitization.
- **Experimental Task:** Somatosensory tactile Braille stimulation delivered to either the index finger (`Start_index`, class 0) or the little finger (`Start_pinky`, class 1) of the right hand.
- **Preprocessing:** Zero-phase FIR notch filtering at 50 Hz and 100 Hz and bandpass filtering from 1.0 to 45.0 Hz (MNE firwin design, Hamming window), epoching over $[-0.2\text{ s}, +1.8\text{ s}]$ with baseline correction relative to $[-0.2\text{ s}, 0.0\text{ s}]$, and anti-aliased resampling to 256 Hz.

### 1.3 OpenNeuro ds000117: In-Domain Cryogenic SQUID Baseline
- **Accession:** OpenNeuro `ds000117` (Wakeman & Henson, Scientific Data 2015).
- **Cohort:** Evaluated on participant `sub-01` across 5-fold cross-validation.
- **Instrumentation:** 306-channel cryogenic SQUID system (Elekta Neuromag-306), consisting of 102 magnetometers and 204 planar gradiometers.
- **Experimental Task:** Visual presentation of familiar faces, unfamiliar faces, and scrambled control images. Serves as the matched in-domain positive control for BrainOmni pre-trained representations.
- **Preprocessing:** Native continuous MEG filtered with zero-phase FIR notch filters at 50, 100, 150 and 200 Hz and a 1.0–96.0 Hz bandpass, then epoched to match the visual evaluation window ($[-0.2\text{ s}, +1.8\text{ s}]$), baseline corrected over $[-0.2, 0.0]$ seconds with linear detrending, and resampled to 256 Hz.

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
Hard HFC reduces operator rank by $r = \mathrm{rank}(\mathbf{Q})$ ($\text{rank} = C - r$, at most three removed directions), which produces singular covariance matrices in downstream estimators.

To preserve numerical stability, we implement Soft-HFC with a continuous shrinkage parameter $\alpha = 0.25$:
$$\mathbf{P}_{\text{soft}}(\alpha) = \mathbf{I} - \alpha \mathbf{Q}_R \mathbf{Q}_R^T$$
where $\mathbf{Q}_R$ is a rank-aware orthonormal SVD basis for the normalized orientation matrix. Singular values below or equal to $\max(C,3)\epsilon_{64} s_{\max}$ are discarded. The inverse expression above applies only to full column rank; otherwise use the pseudoinverse. This formulation provides three properties:
1. Full numerical rank is preserved ($\text{rank} = C$).
2. Homogeneous external interference is attenuated by 43.75% in power (25% amplitude shrinkage at $\alpha = 0.25$), with an identical fractional reduction applying only to neural components collinear with $\text{col}(\mathbf{Q})$.
3. Signal components orthogonal to the uniform interference subspace remain unperturbed.


## 3. Reproducible preprocessing validation

The reference is the [MNE 1.11 OPM tutorial](https://mne.tools/1.11/auto_tutorials/preprocessing/80_opm_processing.html). Its reference regression and order-2 HFC are examples for a different recording system. Our optional Soft-HFC is order-1 orientation-subspace shrinkage, with alpha 0.25. It is not equivalent to the tutorial's full projection with gradients. Its attenuation can affect neural signals too.

Both OPM cache builders now use settings from `src/opm_bridge/preprocessing/validation.py`; tests require the YAML filter, sampling and epoch settings to match. FIR method, phase, design and window are explicit. Other MNE defaults and the installed MNE, NumPy and SciPy versions are recorded through the versioned pipeline specification. Nottingham's sampling rate is read from its MEG JSON, and binary byte counts use the declared dtype.

Each newly built cache has a `.qc.json` sidecar containing physical-unit spectra before and after temporal filtering, per-channel standard deviation and peak-to-peak values, flat-channel flags, shielding in dB, geometry rank, epoch sizes and label counts, retained/dropped event records, and source SHA-256 hashes. FACE-DEC also records outlier decisions, MNE epoch drop logs, available reference channels excluded from processing, and spectra before/after optional Soft-HFC. Nottingham records source-status channel exclusions and boundary-related trial exclusions; it does not implement artifact-based trial rejection. Reports explicitly remain `requires_scientific_review`.

Existing caches without matching versioned QC require `--overwrite`; old results are not silently certified. Rebuild into a separate directory when preserving historical inputs:

```bash
PYTHONPATH=src python scripts/stream_and_cache.py --subjects sub-01 --runs run-01 --soft-hfc --out-dir data/cache/validation/ds005107
PYTHONPATH=src python scripts/stream_nottingham_zenodo.py --subjects sub-101 --out-dir data/cache/validation/nottingham
```

For an independent stage comparison, export an NPZ with `before` and `after` arrays of identical shape, channel order and time alignment, scalar `sfreq`, string array `ch_names`, and scalar `unit` (`T`, `fT` or `pT`). Use continuous `[channels, samples]` arrays or matched `[trials, channels, samples]` epochs before model normalization. Optional `orientations` adds a Soft-HFC comparison. Optional `labels` and `tmin` add 512-sample epoch checks and per-condition evoked traces.

```bash
PYTHONPATH=src python scripts/validate_preprocessing.py stages.npz --out results/preprocessing/stages.qc.json
python -m pytest tests/test_preprocessing_validation.py
```

Review these outputs before accepting preprocessing:

1. Inspect raw traces, flat/noisy channels and bad intervals. Check rejected event counts and class balance. Review why FACE-DEC's outlier rule was not applied if fewer than 16 channels survive.
2. Inspect channel-level spectra and shielding, especially low-frequency drift, line frequencies and the task band. Undefined zero-power bins are recorded as null. Positive shielding measures power reduction, not neural-signal preservation.
3. Verify physical units, geometry, coordinate frames and sensor orientations against source metadata. FACE-DEC uses approximate Elekta geometry. Nottingham source metadata confirms fT for the representative recording. Its one-based event samples are now converted to zero-based indices and checked against onset times using the declared sampling rate (see Section 5).
4. Compare matched evoked waveforms, amplitude and latency with and without denoising, separately by condition. Include task-relevant decoding under identical splits. The CLI saves evoked traces for inspection, without declaring a universal pass threshold.
5. If valid environmental reference sensors exist, evaluate reference regression before filtering as in the tutorial. The current cache builders record that regression was not applied. Investigate high-amplitude interference before changing the historical processing order. Keep hard HFC, gradient correction and Soft-HFC as distinct sensitivity analyses with appropriate geometry and rank handling.
6. Archive the inspected reports and rerun affected benchmark conditions. Source hashes identify the bytes read; Nottingham partial streams are labeled as read-prefix hashes. Full-cohort replay and biological validity require further review. Representative source replay and sample-index checks are recorded below.

The rank-aware SVD correction replaces historical unpivoted QR. Full-rank operators are mathematically equivalent within floating-point precision. Rank-deficient arrays can change; full-cohort numerical results have not been recomputed. The representative checks below isolate the effect of this correction.


## 4. Representative raw replay on 2026-11-11

The [evidence summary](../reviews/preprocessing-2026-11-11/findings.json) records source hashes, the pinned BrainOmni checkpoint, versions, limitations and historical comparisons. Replay caches are separate from historical benchmark outputs under `data/cache/validation/`.

- FACE-DEC `sub-01/run-01`: 256 trials, 60 retained channels, rank-three geometry. The old QR and new SVD operators and model inputs are byte-identical. MAG and GRAD reconstruction PCC changes are exactly zero. This does not reproduce the historical 21-subject, 50-channel intersection analysis.
- Nottingham `sub-101`: 84 trials (42 index, 42 pinky), 176 channels, rank-three geometry. The raw member's original ZIP CRC and size were verified. Both operators and model inputs are byte-identical. MAG monolithic PCC is 0.2828227674 and axis-partitioned PCC is 0.4109716458, differing from the saved participant results by approximately 1.8e-8 and -1.1e-9. HFC-fix deltas are exactly zero. Identical deterministic inputs were inferred once rather than redundantly rerun.
- All ten Nottingham source channel tables were checked. Retained-channel counts match the historical factorial, and every old/new float32 operator is identical. This supports unchanged HFC behavior for matching historical inputs; historical cache hashes are unavailable.
- FACE-DEC whole-record spectra show filtering transients at the recording edges. Removing five seconds at each end removes the anomalous power increases; none of the retained epochs overlaps those edge regions. Interior attenuation is about 34 dB at 50 Hz and 9.6 dB at 44 Hz.
- The initial replay found a separate Nottingham timing discrepancy: all 84 event sample numbers exceed rounded onset times 1200 by one. Direct zero-based slicing places epochs 0.833 ms late relative to the onset column. That replay retained the historical convention to isolate HFC. The subsequent correction and paired analysis are described in Section 5.
- The Nottingham source JSON reports `DigitizedLandmarks=false`, whereas the adapter currently sets `has_digitized_landmarks=True`. This provenance flag needs correction or justification, although it is not used in the HFC calculation.

The full test suite passed (23 tests). [Spectral diagnostics](../reviews/preprocessing-2026-11-11/raw_replay_spectra.png) and [condition-average response diagnostics](../reviews/preprocessing-2026-11-11/evoked_hfc_comparison.png) are saved. The evoked plots show before/after HFC effects, not evidence of guaranteed neural preservation. The preprocessing remains subject to scientific review. Section 5 records the subsequent timing correction; the dated artifacts in this section preserve the original audit.

To reproduce the old/new comparisons after preparing the separate caches and checkpoint:

```bash
PYTHONPATH=src:. HF_HOME="$PWD/data/validation/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python scripts/replay_hfc_validation.py --face data/cache/validation/ds005107_nohfc/sub-01/sub-01_ses-01_run-01_256hz.pt --out reviews/preprocessing-2026-11-11/face_hfc_comparison.json
PYTHONPATH=src:. HF_HOME="$PWD/data/validation/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python scripts/replay_hfc_validation.py --nottingham data/cache/validation/nottingham/sub-101_ses-001_task-braille_run-001_256hz.pt --tags 1 --modes monolithic axis_partitioned --out reviews/preprocessing-2026-11-11/nottingham_hfc_comparison.json
```

The dated review directory also contains the bounded-download helper, the local-archive replay wrapper, the geometry audit, plotting scripts and synthesis script. Run them from the repository root with `PYTHONPATH=src:.`. Model inference used CPU and the pinned checkpoint revision recorded in `findings.json`.


## 5. Nottingham event-index correction

The source `sample` column in Zenodo 11126593 is one-based. `parse_nottingham_braille_events` now subtracts one before NumPy slicing. The cache builder passes the recording sampling frequency and checks every selected event against `round(onset * sfreq)`. Invalid samples or inconsistent onsets raise an error. This rule is dataset-specific and is not applied to FACE-DEC or generic BIDS events.

New Nottingham caches record `event_sample_offset=-1` and `event_onset_validation=passed`. Cache reuse requires the corrected offset, so a previously built cache cannot bypass the fix. Historical caches and benchmark summaries are preserved. The representative corrected cache is in `data/cache/validation/nottingham_timing_corrected/`.

The paired sub-101 comparison uses the same raw-byte hashes, filter settings, channel geometry, 84 trial labels and five-fold splits for both conventions. It evaluates MAG-tag Soft-HFC monolithic and axis-partitioned reconstruction and pretrained linear probing, plus classical decoding with and without Soft-HFC. It does not recompute ten-participant inference, random-partition controls, GRAD conditions or confidence intervals.

```bash
PYTHONPATH=src:. .venv/bin/python reviews/nottingham-timing-2026-11-11/replay.py
PYTHONPATH=src:. HF_HOME="$PWD/data/validation/huggingface" HF_HUB_OFFLINE=1 .venv/bin/python scripts/compare_nottingham_timing.py --old data/cache/validation/nottingham/sub-101_ses-001_task-braille_run-001_256hz.pt --corrected data/cache/validation/nottingham_timing_corrected/sub-101_ses-001_task-braille_run-001_256hz.pt --out reviews/nottingham-timing-2026-11-11/comparison.json
```

The replay helper uses the previously checksum-verified local raw archive. [Comparison results](../reviews/nottingham-timing-2026-11-11/comparison.json) retain per-trial PCCs, per-fold balanced accuracies and checkpoint provenance. [Evoked response diagnostics](../reviews/nottingham-timing-2026-11-11/evoked_timing_effect.png) compare matched physical-unit waveforms. Full-cohort historical results retain their original timing until the corresponding cohort rerun is completed.


### Representative timing sensitivity results

For sub-101, with MAG tags and Soft-HFC, the completed paired comparison found:

| Metric | Historical timing | Corrected timing | Change |
|---|---:|---:|---:|
| Monolithic reconstruction PCC | 0.282823 | 0.283460 | +0.000637 |
| Axis-partitioned reconstruction PCC | 0.410972 | 0.408747 | -0.002225 |
| Monolithic probe balanced accuracy | 40.69% | 44.17% | +3.47 percentage points |
| Axis-partitioned probe balanced accuracy | 35.56% | 35.83% | +0.28 percentage points |
| Classical balanced accuracy without Soft-HFC | 49.31% | 49.31% | 0 |
| Classical balanced accuracy with Soft-HFC | 50.69% | 50.69% | 0 |

The axis-minus-monolithic reconstruction advantage remains positive for this participant, changing from 0.128149 to 0.125287. Both historical-timing probe accuracies reproduce the saved results exactly; both historical-timing reconstruction values reproduce to within 2e-8 PCC. All five classical fold accuracies are unchanged. No hyperparameters were retuned.

The correction moves all 84 events earlier by one native sample (0.833 ms), while preserving 176 channels and class counts of 42/42. All 30 tests pass, including onset agreement, invalid indexing, correct retained-event positions and rejection of stale timing caches. See the [final timing findings](../reviews/nottingham-timing-2026-11-11/findings.json).

These are numerical changes in one participant. They do not establish that cohort confidence intervals, p-values, decoding conclusions or axis-versus-random contrasts are unchanged. Historical cohort tables have not been overwritten; full-cohort evaluation with the corrected timing remains required before publishing updated numbers.
