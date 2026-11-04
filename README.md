# OPM-Bridge: Cross-Hardware Generalization of MEG Foundation Models to Optically Pumped Magnetometers

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

This repository provides an empirical benchmark evaluating whether large electrophysiological foundation models—specifically **BrainOmni** (NeurIPS 2025)—transfer zero-shot or few-shot representations to on-scalp optically pumped magnetometer (OPM-MEG) recordings.

Conventional magnetoencephalography (MEG) relies on cryogenic superconducting quantum interference device (SQUID) arrays fixed inside rigid dewars. Emerging optically pumped magnetometers (OPMs) operate without cryogenics and conform flexibly to the scalp, introducing marked shifts in spatial layout, channel counts, local signal-to-noise ratios, and sensor orientations (including triaxial $X/Y/Z$ sensing). OPM-Bridge quantifies cross-hardware transfer efficiency, provides matched baselines and statistical equivalence tests, diagnoses failure modes, and introduces layout adaptation modules.

---

## Key Experimental Findings

1. **Pre-Training Corpus Domain:** An audit of BrainOmni's pre-training corpus (2,653.8 hours total; 1,997.0 h EEG, 656.8 h MEG across 22 datasets) confirms that zero OPM-MEG recordings were present during pre-training. All upstream MEG data originated from conventional cryogenic SQUID systems (CTF-275 axial gradiometers, Elekta Neuromag-306, and KIT/Yokogawa systems).
2. **In-Domain SQUID Baseline (`ds000117`):** On cryogenic SQUID recordings (Elekta-306, 102 magnetometers + 204 planar gradiometers), Pretrained BrainOmni consistently outperforms Control A (Random Init) on both visual detection (97.56% vs 92.69%, $\text{AUC} = 0.9993$ vs $0.9932$, $\Delta = +4.87\%$) and fine-grained face decoding (61.03% vs 55.81%, $\Delta = +5.22\%$). This validates that the foundation model encodes valid neurophysiological representations in its native cryogenic domain, and confirms that OPM performance bottlenecks reflect true **cross-hardware transfer deficits**.
3. **Sensor-Type Embedding Asymmetry:** Remapping native SQUID magnetometers from `MAG=1` to `GRAD=2` increases zero-shot reconstruction PCC from **0.4105** to **0.7130** (**+73.7% relative gain**, $\Delta = +0.3025$). This mirrors the +52.7% gain observed when remapping OPM magnetometers to `GRAD=2` on `ds005107` (0.2501 to 0.3819, $p < 0.0001$). This proves that `opm_as_grad` is not an OPM-specific geometric compensation, but an architectural inductive bias in BrainOmni's tokenizer weights, which were heavily shaped by planar gradiometer pre-training.
4. **Disentangled Triaxial Bottleneck (Capacity vs. Orthogonal Vectors):** In coregistered triaxial ($X/Y/Z$) OPM recordings (Zenodo 11126593, 10 adult subjects, 84 trials/subject = 840 trials total), monolithic triaxial encoding yields $PCC = 0.3726 \pm 0.03$. A size-matched random 3-way split (capacity control) yields $PCC = 0.3995 \pm 0.03$ ($\Delta = +0.0269, p = 0.0016$). Our axis-partitioned adapter restores reconstruction fidelity to **$PCC = 0.4849 \pm 0.03$** ($\Delta = +0.1123, p = 0.0016$). This establishes that channel load per query explains only 24% of the bottleneck; over **76% is genuine cross-axis vector interference** when 3 orthogonal coils share identical 3D positions.
5. **Evoked Representation Transfer Dynamics:** On clean, unconfounded visual evoked detection ($[-200\text{ ms}, 0\text{ ms}]$ baseline vs $[+50\text{ ms}, +550\text{ ms}]$ evoked response), the Pretrained linear probe scales monotonically from **95.39%** ($B=1$) to **99.81%** ($B=20$), significantly exceeding Control A (94.47% to 96.60%, $\Delta = +3.22\%, p < 0.0001$). In contrast, demographic decoding (age, sex, ancestry) remains near chance (~50%) across all architectures, establishing an empirical single-trial SNR floor.

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
  title={OPM-Bridge: Evaluating Cross-Hardware Transfer of Electrophysiological Foundation Models to Optically Pumped Magnetometers},
  author={OPM-Bridge Contributors},
  journal={arXiv preprint},
  year={2026}
}
```

## License

This project is licensed under the MIT License. Upstream model architectures adapt code from OpenTSLab/BrainOmni under the MIT License.
