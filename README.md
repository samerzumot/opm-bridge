# OPM-Bridge: Cross-Hardware Generalization of MEG Foundation Models to Optically Pumped Magnetometers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository contains the empirical benchmark and evaluation codebase investigating cross-hardware transfer of electrophysiological foundation models—specifically BrainOmni (NeurIPS 2025)—to on-scalp optically pumped magnetometer (OPM-MEG) recordings.

Conventional magnetoencephalography (MEG) relies on cryogenic superconducting quantum interference device (SQUID) arrays fixed inside rigid dewars. Emerging optically pumped magnetometers (OPMs) operate without cryogenics and conform flexibly to the scalp, introducing marked shifts in spatial layout, channel counts, local signal-to-noise ratios, and sensor orientations (including triaxial $X/Y/Z$ sensing). OPM-Bridge quantifies cross-hardware transfer efficiency, provides matched baselines and statistical equivalence tests, diagnoses failure modes, and evaluates layout adaptation modules.

---

## Key Experimental Findings

1. **Pre-Training Corpus Composition:** Auditing the BrainOmni pre-training mixture (2,653.8 hours; 1,997.0 h EEG, 656.8 h MEG across 22 datasets) confirms that all upstream MEG data originated from conventional SQUID systems (CTF-275, Elekta Neuromag-306, and KIT/Yokogawa). No OPM-MEG data was present in the pre-training corpus.
2. **In-Domain SQUID Positive Control (`ds000117`):** On native cryogenic recordings (Elekta-306, 102 magnetometers + 204 planar gradiometers), pre-trained representations outperform random initializations on face-versus-scrambled decoding (61.03% vs 55.81%, $\text{AUC} = 0.5909$ vs $0.5591$, $\Delta = +5.22\%$). This verifies model competence on in-distribution SQUID data and isolates performance drops on OPM-MEG as cross-hardware transfer effects.
3. **Sensor-Type Categorical Embedding Sensitivity:** Remapping cryogenic SQUID magnetometers from `MAG=1` to `GRAD=2` improves zero-shot reconstruction PCC from 0.4105 to 0.7130 (+73.7% relative gain, $\Delta = +0.3025$). This replicates the +52.7% gain observed when remapping OPM magnetometers to `GRAD=2` on `ds005107` (0.2501 to 0.3819, $p < 0.0001$), indicating that the effect is driven by an asymmetric gradiometer pre-training inductive bias within the tokenizer.
4. **Disentangled Triaxial Bottleneck (Matched $2 \times 3$ Grid):** In coregistered triaxial ($X/Y/Z$) OPM recordings (Zenodo 11126593, 10 participants, 84 trials/subject), monolithic triaxial encoding yields $PCC = 0.3461$ (Soft-HFC) and $0.3726$ (Raw). An axis-partitioned adapter restores reconstruction accuracy to $PCC = 0.4849$ (Soft-HFC) and $0.5130$ (Raw). Capacity control testing under matched filtering demonstrates that query channel count accounts for only ~20% of the gain, while ~80% ($\Delta = +0.1096, p = 0.0016$ under Soft-HFC; $\Delta = +0.1135, p = 0.0016$ raw) is attributable to resolving cross-axis directional vector interference.
5. **Single-Trial Downstream Decoding Boundaries:** On single-trial facial stimulus attribute classification (face age, sex, ancestry), downstream linear probing remains bounded near the empirical chance floor (~50%) across all subject budgets ($B \in \{1, 2, 4, 10, 20\}$), where a parameter-efficient convolutional specialist (`CompactMEGNet`, 1.5k parameters) matches or outperforms foundation-scale representations (51.85% vs 49.57% at $B=20$).

---

## Datasets

OPM-Bridge benchmarks across three public neuroimaging corpora:

1. **OpenNeuro ds005107 ("FACE-DEC"):**
   - 21 adult subjects, single-axis radial OPM sensors (Quanmag Healthcare).
   - Balanced factorial design: face perception across demographic and affective dimensions.
   - Standardized to 50 physically aligned helmet positions (`CoordinateFrame.APPROX_ELEKTA`).
2. **Zenodo 11126593 (Rier & Rhodes et al., eLife 2024):**
   - 192-channel triaxial ($X/Y/Z$) QuSpin zero-field magnetometers coregistered in head space (`CoordinateFrame.HEAD`).
   - Somatosensory tactile stimulation (Braille stimulator; index finger vs. little finger; 84 trials/subject across 10 adult participants).
3. **OpenNeuro ds000117 (Wakeman & Henson, Scientific Data 2015):**
   - Cryogenic SQUID Elekta Neuromag-306 array (102 magnetometers + 204 planar gradiometers).
   - Visual face perception paradigm serving as the exact in-domain foundation model baseline.

---

## Installation

```bash
git clone https://github.com/samerzumot/opm-bridge.git
cd opm-bridge
pip install -e .
```

To run unit tests:

```bash
pytest -v
```

---

## Reproducing Empirical Benchmarks

### 1. Cross-Subject Transfer Evaluation
To evaluate linear probing, end-to-end models, and compact baselines across varying subject sample budgets:

```bash
python scripts/run_transfer_benchmark.py \
    --data-dir data/cache/ds005107 \
    --task age \
    --arms pretrained_linear_probe control_a_linear_probe control_b_linear_probe compact_megnet classical_bandpower \
    --budgets 1 2 4 10 20 \
    --seeds 42 3407 2025 \
    --soft-hfc \
    --out results/summary_aligned21_age.json
```

### 2. Sensor Geometry and Embedding Diagnostics
To assess sensitivity to coordinate perturbations and sensor type designations:

```bash
for mode in default opm_as_grad zero_orientation flip_orientation scramble_channels; do
    python scripts/run_transfer_benchmark.py \
        --data-dir data/cache/ds005107 \
        --task age \
        --mapping-mode $mode \
        --arms pretrained_linear_probe \
        --out results/summary_phase3_ablation_${mode}.json
done
```

### 3. Nottingham Triaxial Benchmark & Capacity Control
To run the full 84-trial/subject benchmark and capacity control ablation:

```bash
python scripts/run_nottingham_full_benchmark.py \
    --data-dir data/cache/nottingham \
    --out results/summary_phase4_nottingham_full.json
```

### 4. In-Domain SQUID Positive Control
To stream and benchmark the paired Elekta Neuromag-306 SQUID dataset (`ds000117`):

```bash
python scripts/stream_ds000117_squid.py
python scripts/run_squid_in_domain_benchmark.py \
    --cache data/cache/ds000117/sub-01/sub-01_ses-meg_run-01_256hz.pt \
    --out results/summary_ds000117_squid_baseline.json
```

### 5. Render Tables and Figures
To generate the exact publication figures and Markdown tables from JSON ledgers:

```bash
python scripts/generate_paper_tables.py
python scripts/generate_paper_figures.py
```

---

## Citation

If you use this benchmark, code, or findings in your research, please cite:

```bibtex
@article{opm_bridge_2026,
  title={Cross-Hardware Generalization of Electrophysiological Foundation Models: An Empirical Evaluation on Optically Pumped Magnetometers},
  author={Zumot, Samer and Zare, Marzieh},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License. Upstream model architectures adapt code from OpenTSLab/BrainOmni under the MIT License.

Preprocessing validation, cache QC reports, and historical-result limitations are documented in [Dataset Specifications and Harmonization Protocols](docs/datasets.md#3-reproducible-preprocessing-validation).
