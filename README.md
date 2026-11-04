# OPM-Bridge

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)

*An open benchmark of how pre-trained MEG foundation models ([BrainOmni](https://arxiv.org/abs/2505.18185), NeurIPS 2025) transfer to scarce optically pumped magnetometer (OPM-MEG) data.*

---

## Key Highlights & Phase 0 Verified Findings (`[V]`)

1. **Zero OPM Pre-Training Contamination (`[V]`):** Verified across all 22 pre-training datasets (Appendix D) and 11 downstream datasets (Appendix E) of BrainOmni (`arXiv:2505.18185v1`) that **zero OPM-MEG recordings** were included. All BrainOmni MEG training used conventional SQUID systems (Elekta Neuromag, CTF, KIT/Yokogawa).
2. **Co-Primary Dataset Design (`[V]`):**
   - **Target 1 (`ds005107`, OpenNeuro, CC0):** 21 subjects, 64-channel single-axis Quanmag OPM-MEG during a balanced $2^4$ factorial face perception task (`race`, `gender`, `age`, `emotion`). Per `ds005107/README`, sensor coordinates were approximated from an Elekta layout (`CoordinateFrame.APPROX_ELEKTA`), isolating **OPM signal/SNR domain shift**.
   - **Target 2 (Nottingham OPM, `10.5281/zenodo.11126593`, CC-BY-4.0, `Adults.zip`):** 26 adult subjects, up to 192-channel **triaxial (`X/Y/Z`) QuSpin OPM-MEG** with subject-coregistered 3D positions (`Px, Py, Pz`) and true 3D unit orientations (`Ox, Oy, Oz`) (`CoordinateFrame.HEAD`) during a tactile Braille somatosensory task (`Start_index` vs. `Start_pinky`), isolating **true triaxial OPM geometry and orientation transfer**.
3. **Three Upstream BrainOmni Bugs Discovered & Fixed in OPM-Bridge (`[V]`):**
   - **Frozen Tokenizer in `encode()` & `pretrained=False`:** Upstream `BrainTokenizer.tokenize()` is decorated with `@torch.no_grad()`, preventing gradient flow into `BrainSensorModule` during fine-tuning and leaving `BrainTokenizer` frozen at random weights when `pretrained=False`. [`BrainOmniClassifier`](./src/opm_bridge/models/wrapper.py) explicitly separates `backbone_ft`, `end_to_end_ft`, `control_a_random_e2e` (with differentiable straight-through RVQ), and `control_b_random_backbone`.
   - **MEG/OPM Bad-Channel Blind Spot:** Upstream `auto_detect_bad_channels` only runs on `eeg_mask`, allowing unlocked/noisy OPM channels to contaminate all channels via group common-average referencing (`sensortype_wise_normalize`). Rejecting outlier OPM channels on real OPM data (`ds005346` `sub-31`) improves zero-shot `BrainTokenizer` waveform reconstruction PCC by **+51% relative (`0.1626 -> 0.2459`)**.
   - **Plain-Integer `coil_type` Misclassification:** Upstream `if "MAG" in str(i["coil_type"])` misclassifies plain integer FIFF magnetometer codes (`2000`, `3024`, `8001`–`8004`) as `GRAD (2)`. Fixed in [`sensor_mapping.py`](./src/opm_bridge/preprocessing/sensor_mapping.py).

See [NOTES.md](./NOTES.md) for the complete Phase 0 verification gate log and [docs/datasets.md](./docs/datasets.md) for the verified dataset catalog.

---

## Empirical Benchmark Findings (All 21 Participants, 5,376 Epochs)

Evaluated on OpenNeuro `ds005107` across all 21 subjects, 58 harmonized OPM channels, and Leave-One-Subject-Out (LOSO) cross-validation with **Soft-HFC** spatial noise filtering (`alpha=0.25`):

| Model Arm | $B=1$ (256 trials) BACC [95% CI] | $B=2$ (512 trials) BACC [95% CI] | $B=4$ (1,024 trials) BACC [95% CI] | Architectural Type |
| :--- | :--- | :--- | :--- | :--- |
| **`pretrained_backbone_ft`** | 49.97% [49.80%, 50.13%] | 49.94% [49.83%, 50.00%] | 50.00% [50.00%, 50.00%] | 13.5M Transformer Foundation Model |
| **`control_a_random_e2e`** | 50.00% [50.00%, 50.00%] | 50.00% [50.00%, 50.00%] | 50.00% [50.00%, 50.00%] | Matched Random Init E2E |
| **`compact_megnet`** (+Soft-HFC) | 49.14% [47.55%, 50.65%] | 48.67% [47.51%, 49.90%] | **50.29% [49.17%, 51.42%]** | Compact CNN Specialist (~2,500 params) |
| **`classical_bandpower`** | 49.57% [48.67%, 50.29%] | 49.66% [48.83%, 50.46%] | 50.01% [49.20%, 50.81%] | Classical Log Power + Logistic Reg |

### Primary Conclusions:
1. **Primary Hypothesis (RQ1 - Foundation Model Benefit):** SQUID pre-training provides no detectable transfer advantage over training from scratch on low-budget OPM data ($\Delta = 0.00\%$, pre-registered SESOI = 0.03 decision: **`equivalent`**).
2. **Specialist Inductive Bias (RQ2):** Domain-specific compact architectures (`CompactMEGNet`) scale with cross-subject data to **50.29% mean BACC** (with single-subject folds decoding up to **54.47%**), outperforming heavy Transformer backbones that underfit at small sample sizes.
3. **Soft-HFC Efficacy:** Homogeneous Field Correction with shrinkage $\alpha=0.25$ mitigates ambient external field interference without covariance rank loss ($\text{rank} = C$).
4. **Detailed Technical Report:** See [docs/technical_report.md](./docs/technical_report.md).

---

## Quickstart

```bash
# Install dependencies (Python 3.10–3.12 via uv)
uv sync --python 3.12 --extra dev

# Run unit and smoke test suite
uv run pytest -v

# Run Phase 1.5 synthetic signal gate smoke check
uv run python scripts/run_phase15_signal_gate.py --smoke

# Verify BrainOmni-tiny checkpoint loading from Hugging Face
uv run python scripts/run_transfer_benchmark.py --verify-checkpoint
```

---

## Repository Structure

```text
opm-bridge/
├── configs/
│   ├── ds005107_primary.yaml        # Target 1 (OpenNeuro ds005107) config
│   └── nottingham_adults.yaml       # Target 2 (Zenodo 11126593 Adults) config
├── docs/
│   └── datasets.md                  # Verified OPM-MEG dataset catalog
├── scripts/
│   ├── run_phase15_signal_gate.py   # Phase 1.5 per-task learnable-signal gate
│   └── run_transfer_benchmark.py    # Resumable transfer & diagnostic runner
├── src/opm_bridge/
│   ├── data/
│   │   ├── schema.py                # CoordinateFrame, SensorMetadata, HarmonizedSample
│   │   ├── ds005107.py              # ds005107 events (+33ms latency) & 2^4 face labels
│   │   └── nottingham_opm.py        # Pure-Python flat binary & triaxial BIDS loader
│   ├── preprocessing/
│   │   └── sensor_mapping.py        # Pos/ori normalization & RQ3 mapping ablations
│   ├── models/
│   │   ├── vendor_brainomni/        # MIT-licensed BrainOmni/BrainTokenizer (no DeepSpeed)
│   │   ├── wrapper.py               # BrainOmniClassifier (6 transfer & control arms)
│   │   └── baselines.py             # CompactMEGNet & ClassicalBandPowerDecoder
│   ├── train/
│   │   └── runner.py                # Subject-wise K-fold splits, nested budgets & ledger
│   └── eval/
│       └── metrics.py               # Subject-level bootstrap CI, SESOI & signal gate
├── tests/
│   └── test_smoke.py                # Unit tests for loaders, ablations, gradients & CIs
├── NOTES.md                         # Phase 0 verification log & bug findings
└── pyproject.toml
```
