# OPM-Bridge: Phase 0 Verification Log & Pre-Registration Notes

**Date:** 2026-11-03  
**Status:** Phase 0 Verification Gate **COMPLETE** (Primary Track Confirmed)

Evidence labels:
- **[V]** Verified directly by a command, file inspection, or execution that ran on 2026-11-03.
- **[E]** Estimate or analytical derivation.
- **[U]** Unverified.

---

## 1. Phase 0 Verification Gate Results (§5.1)

| # | Check | Status | Verified Facts & Method (2026-11-03) |
|---|---|---|---|
| **1** | **Code license** | **PASS [V]** | Inspected `OpenTSLab/BrainOmni/LICENSE` and scanned all `.py` files via GitHub API. Exactly 3 files (`model_utils/conv.py`, `model_utils/lstm.py`, `model_utils/seanet.py` — derived from Meta's EnCodec SEANet) carry the `Copyright (c) Meta Platforms, Inc.` header; all other files are `Copyright (c) 2025 OpenTSLab`. Both notices in `LICENSE` are standard permissive **MIT licenses**. |
| **2** | **Weight license & terms** | **PASS [V]** | Inspected `README.md` and Hugging Face metadata for `OpenTSLab/BrainOmni` (rev `9a4d3c70495370397ccfbfd6d2496f25647545a5`), `braindecode/brainomni-tiny-pretrained` (rev `d111525e4d`), and `braindecode/braintokenizer-pretrained` (rev `945aa4892e`). All explicitly declare `license: mit` with no restrictions on fine-tuning or research redistribution. |
| **3** | **Checkpoints load & run** | **PASS [V]** | Vendored the 9 pure-PyTorch model modules from `OpenTSLab/BrainOmni` into `src/opm_bridge/models/vendor_brainomni/`, replacing the single `import deepspeed.comm as dist` in `model_utils/vq.py` with a standard `torch.distributed as dist` fallback. Downloaded `tiny/BrainOmni.pt` (48.82 MB, sha256 `62c67ba6a84ea0625e67a3b5e7463fe3930bfee88a612a225e9062a052542ffc`) and ran `BrainOmni-tiny` on CPU (Python 3.12.12, PyTorch 2.15.0): `Missing: [] Unexpected: []`, forward pass completed in `2.399s` on CPU. Total parameters: `13,475,042` (`5,050,338` in `BrainTokenizer` + `8,424,704` in `BrainOmni` backbone). |
| **3b** | **Braindecode port equivalence** | **PASS [V]** | Inspected `convert_brainomni_checkpoints.py` and `README.md` in `braindecode/brainomni-tiny-pretrained`: the conversion is an exact lossless key-rename (`max-abs difference 0.0` in float32 and bfloat16) dropping only pre-training `mask_token` and `predict_head`. Because our 1-line `torch.distributed` shim lets the official `OpenTSLab/BrainOmni` code run natively on Python 3.12 CPU/MPS/CUDA without `deepspeed`, we use the exact upstream architecture directly while retaining `braindecode` compatibility. |
| **4** | **Pre-training data & OPM overlap** | **PASS [V]** | Extracted Appendix D (Pre-training datasets, 22 corpora) and Appendix E (Downstream datasets, 11 corpora) from `arXiv:2505.18185v1` HTML. All 9 pre-training MEG datasets (`MEG-MASC` [KIT], `MEG-Narrative` [CTF], `OMEGA` [CTF], `CC700/Cam-CAN` [Elekta], `AversiveMEG` [CTF], `MIND` [Elekta], `SMN4Lang` [Elekta], `THINGS-MEG` [CTF], `ASWR-MEG` [Elekta], `ImageLine` [Elekta], `NeuroMorph` [KIT], `Kymata-SOTO` [Elekta]) and all downstream MEG datasets (`Gloups-MEG`, `ASD74` [`ds005234`], `SomatoMotor` [`ds006035`]) are **conventional SQUID MEG**. **Zero OPM-MEG recordings were used in BrainOmni pre-training or evaluation.** |
| **5** | **Input requirements & OPM compatibility** | **PASS [V]** | Inspected `constant.py` and `factory/utils.py`: `SAMPLE_RATE = 256 Hz`, `LOW = 0.1 Hz`, `HIGH = 96 Hz`, `TOKENIZER_SEGMENT_TIME = 2.0 s` (`window_length = 512` samples, `overlap_ratio = 0.25`), `SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}`, `pos` shape `(B, C, 6)` (3D position centered and scaled by $\sqrt{3 \cdot \text{mean}(\|p\|^2)}$ + 3D unit orientation vector). Tested live on a real downloaded Quanmag OPM `.fif` recording (`ds005346` `sub-31_task-question_meg.fif`, 4.4 MB, 64 channels). |
| **5b** | **Coordinate frames** | **CRITICAL FINDING [V]** | 1. **Mathematical invariance check**: Because `factory/utils.py:normalize_pos` centers `pos[:, :3]` to zero mean and normalizes root-mean-square radius per recording, **pure translation and uniform scaling (e.g., cm vs. m or head-size scaling) are invariant**, whereas **3D rotation between device and head frames (`dev_head_t`) is NOT invariant**.<br>2. **`ds005107` ("FACE-DEC") frame caveat**: Inspected `ds005107/README`, `code/analysis/face_1_prep.py`, and the FIF header of `sub-01_ses-01_task-face_run-01_meg.fif` via S3 HTTP Range request. `dev_head_t` is the $4\times 4$ identity matrix (`coord_frame = 1 FIFFV_COORD_DEVICE`), and `ds005107/README` explicitly states: *"the sensor coordinates used were approximated by selecting corresponding locations from the Elekta layout and do not reflect the actual sensor positions (only for visualizing topographic maps)."* Tagged as `CoordinateFrame.APPROX_ELEKTA`.<br>3. **Nottingham OPM (`10.5281/zenodo.11126593`) & `ds005346`**: Both carry **true subject-coregistered OPM sensor positions and orientations** (`CoordinateFrame.HEAD`). |
| **6** | **Reproduce a reported / demo output** | **PASS [V]** | Ran `BrainOmni-tiny` on upstream `assets/example_data.pt` (64 channels, 1024 samples): achieved **zero-shot waveform reconstruction `PCC = 0.8043`** on CPU in `2.4s`, matching Table 4 of the BrainOmni paper (`PCC = 0.802` on unseen devices). |
| **7** | **Prior OPM evaluations** | **PASS [V]** | Full-text search of `arXiv:2505.18185v1` (`OPM`, `optically`, `QuSpin`, `Cerca`, `FieldLine`, `5107`) and literature search confirmed zero prior evaluations of BrainOmni (or MEG-GPT / MEG-Mamba) on OPM-MEG transfer. |
| **8** | **OPM dataset terms & metadata** | **PASS [V]** | Verified primary sources for `ds005107` (OpenNeuro, CC0), Nottingham OPM (`10.5281/zenodo.11126593`, CC-BY-4.0, separate `Adults.zip` [N=26, 12.24 GB] and `Children.zip` [N=27, 12.67 GB]), Rier et al. test-retest (`10.5281/zenodo.7525341`, CC-BY-4.0, N=10, 8.24 GB), and `ds005346` (OpenNeuro, CC0, N=30 OPM subjects with digitized landmarks). See `docs/datasets.md`. |

---

## 2. Three Technical Bugs Uncovered in Upstream BrainOmni During Phase 0 [V]

1. **Frozen Tokenizer in `encode()` & `pretrained=False` (`braintokenizer/model.py:1006`, `brainomni/model.py:165`):**
   - Upstream `BrainTokenizer.tokenize()` is decorated with `@torch.no_grad()`, and `self.tokenizer.encoder.neuros` is `.detach()`ed inside `BrainOmni.encode()`.
   - As a result, upstream "full fine-tuning" **never updates `BrainTokenizer` or `BrainSensorModule`**, and running upstream `pretrained=False` (Control A) passes data through a **frozen randomly initialized VQ-VAE tokenizer**!
   - **Fix in OPM-Bridge (`src/opm_bridge/models/wrapper.py`):** Explicitly separate `backbone_ft` (frozen pre-trained tokenizer + fine-tuned backbone, matching upstream), `end_to_end_ft` (gradients enabled through `BrainTokenizer` + `BrainSensorModule`), `control_a_random_e2e` (fully random init with end-to-end gradients through `BrainTokenizer`), and `control_b_random_backbone` (frozen pre-trained tokenizer + random backbone).

2. **Bad-Channel Detection Blind Spot on MEG/OPM (`factory/utils.py:621`):**
   - Upstream `auto_detect_bad_channels(raw, eeg_mask, mag_mask, grad_mask)` only checks `if eeg_mask.any():` and **completely ignores `mag_mask` and `grad_mask`**.
   - Combined with `sensortype_wise_normalize` (which subtracts the common average across all magnetometers and divides by the global `std(mag_data)`), a single unlocked/noisy OPM channel contaminates all channels via the common average and crushes signal amplitude across the entire array.
   - **Empirical proof on `ds005346` `sub-31` [V]:** Zero-shot OPM waveform reconstruction with `BrainTokenizer` is `PCC = 0.1626` on all 64 channels, and jumps by **+51% relative (`PCC = 0.2459`)** simply by filtering out 14 outlier OPM channels before group normalization.

3. **Plain-Integer `coil_type` Misclassification (`factory/utils.py:578`):**
   - Upstream checks `if "MAG" in str(i["coil_type"])`, which evaluates to `False` (misclassifying magnetometers as `GRAD = 2`) whenever `coil_type` is stored as a plain integer (e.g., `2000` or `3024`) rather than an MNE `NamedInt`. Fixed in `src/opm_bridge/preprocessing/sensor_mapping.py`.
