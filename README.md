# OPM-Bridge: Cross-Hardware Generalization of MEG Foundation Models to Optically Pumped Magnetometers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository provides an empirical benchmark evaluating whether large electrophysiological foundation models—specifically **BrainOmni** (NeurIPS 2025)—transfer zero-shot or few-shot representations to on-scalp optically pumped magnetometer (OPM-MEG) recordings.

Conventional magnetoencephalography (MEG) relies on cryogenic superconducting quantum interference device (SQUID) arrays fixed inside rigid dewars. Emerging optically pumped magnetometers (OPMs) operate without cryogenics and conform flexibly to the scalp, introducing marked shifts in spatial layout, channel counts, local signal-to-noise ratios, and sensor orientations (including triaxial $X/Y/Z$ sensing). OPM-Bridge quantifies cross-hardware transfer efficiency, provides matched baselines and statistical equivalence tests, diagnoses failure modes, and introduces layout adaptation modules.

---

## Key Experimental Findings

1. **Pre-Training Corpus Domain:** An audit of BrainOmni's pre-training corpus (2,653.8 hours total; 1,997.0 h EEG, 656.8 h MEG across 22 datasets) confirms that zero OPM-MEG recordings were present during pre-training. All upstream MEG data originated from conventional cryogenic SQUID systems (CTF-275 axial gradiometers, Elekta Neuromag-306, and KIT/Yokogawa systems).
2. **Transfer Efficiency vs. Matched Random Controls:** Across cross-subject evaluations on visual decoding tasks (21 subjects, 50 physically aligned helmet channels), linear probing on frozen pre-trained BrainOmni latents yields classification performance statistically equivalent to matched randomly initialized controls ($\Delta \in [-1.21\%, +1.20\%]$ across varying subject budgets $B \in \{1, 2, 4, 10, 20\}$). Compact domain-specific architectures (`CompactMEGNet`) match or exceed foundation model performance at lower sample regimes.
3. **Sensor-Type Embedding Asymmetry:** Upstream tokenizers utilize a categorical sensor embedding (`EEG=0`, `MAG=1`, `GRAD=2`). In BrainOmni's pre-training corpus, gradiometers (`GRAD`) outnumber magnetometers (`MAG`) by over 4:1. Mapping radial OPM sensors to `GRAD` rather than `MAG` increases zero-shot waveform reconstruction Pearson correlation coefficient (PCC) by **+52.7% relative** (from $0.2501 \pm 0.005$ to $0.3819 \pm 0.004$, $p < 0.0001$), identifying a silent domain mismatch in standard tokenizer configurations.
4. **Spatial Bottlenecks in Triaxial OPMs:** On coregistered triaxial ($X/Y/Z$) OPM recordings (Zenodo 11126593), feeding all orthogonal axes into a fixed spatial bottleneck ($N_{\text{neuro}} = 16$) induces cross-axis interference ($PCC = 0.3326$). Isolating radial channels ($Z$-axis) improves reconstruction to $PCC = 0.4675$ (+40.6%). An axis-partitioned adapter preserves all channels while restoring reconstruction fidelity ($PCC = 0.4587$).

---

## Repository Structure

```text
├── configs/               # Reproducible experimental configurations (YAML)
├── docs/                  # Dataset documentation and extended experimental analyses
├── paper/                 # Manuscript draft and figures for academic submission
│   ├── manuscript.md      # Full academic manuscript
│   └── figures/           # Rendered empirical plots
├── results/               # Full JSON ledgers and bootstrapped summary metrics
├── scripts/               # Automated data streaming and evaluation runners
│   ├── run_transfer_benchmark.py
│   └── stream_nottingham_zenodo.py
├── src/opm_bridge/        # Core library
│   ├── data/              # BIDS loaders and metadata parsers
│   ├── eval/              # Bootstrapped metrics and equivalence testing
│   ├── models/            # BrainOmni wrappers, baseline models, and adapters
│   ├── preprocessing/     # Sensor geometry mapping and Soft-HFC spatial filtering
│   └── train/             # Deterministic training routines and ledger tracking
└── tests/                 # Unit and regression test suite
```

---

## Datasets

OPM-Bridge benchmarks on two primary public OPM corpora:

1. **OpenNeuro ds005107 ("FACE-DEC"):**
   - 21 adult subjects, single-axis radial OPM sensors (Quanmag Healthcare).
   - Balanced factorial design: face perception across demographic and affective dimensions (gender, race, age, expression).
   - Harmonized to 50 physically aligned coregistered helmet positions across all participants.
2. **Zenodo 11126593 (Rier & Rhodes et al., eLife 2024):**
   - 192-channel triaxial ($X/Y/Z$) QuSpin zero-field magnetometers coregistered in head space (`CoordinateFrame.HEAD`).
   - Somatosensory tactile stimulation (Braille stimulator; index finger vs. little finger).

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

---

## Citation

If you use this benchmark, code, or findings in your research, please cite:

```bibtex
@article{opm_bridge_2026,
  title={OPM-Bridge: Evaluating Cross-Hardware Transfer of Electrophysiological Foundation Models to Optically Pumped Magnetometers},
  author={OPM-Bridge Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License. Upstream model architectures adapt code from OpenTSLab/BrainOmni under the MIT License.
