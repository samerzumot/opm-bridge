# OPM-Bridge: Product Requirements Document

*An open benchmark of how pre-trained MEG foundation models transfer to scarce OPM data*

Version 4.1 (draft) · November 3, 2026 · Status: pending verification gate (Phase 0)

**Evidence labels used in this document:** **[V]** verified by a command or file read that actually ran (date and method in NOTES.md) · **[S]** from search snippets or secondary sources, not yet checked against the primary source · **[U]** unverified · **[E]** estimate or reasoning.

## Changes from v4.0

- **Primary metric redefined** (§4): the "convergence budget" is now descriptive only. The pre-registered primary is Δ balanced accuracy at the smallest budget.
- **Random-init control defined** (§6): fully random is the primary control; pre-trained tokenizer with a random backbone is a diagnostic.
- **Tasks specified** (§9): decoding regime (cross-subject or within-subject) fixed per dataset. The naturalistic "held-out-segment prediction" task is replaced with labelled tasks.
- **New Phase 1.5 signal gate** (§5.2): no transfer runs on a task where the simple specialist cannot beat chance.
- **Gate updated with checks already run** (§5): code license, weight tags, braindecode ports, ds005107 facts. **New gate items:** coordinate frames and equivalence of the braindecode port.
- **Gate item 6 made feasible:** any public-data result with the tiny model, under a GPU-hour cap.
- **Compute budget added** (§8.2). The tiny model is primary; the base model is limited.
- **Scope tightened** (§10): two OPM datasets are required; the third dataset and the OSE probe are stretch goals.
- **Credibility-step note** added to §12. The scope is otherwise unchanged.

## 1. Summary

OPM-Bridge is an open-source benchmark that measures whether pre-trained MEG foundation models help on small optically pumped magnetometer (OPM) datasets, and where they fail. The primary model under test is BrainOmni, which already encodes sensor position, orientation and type and has public checkpoints. Our contribution is the evaluation: harmonized OPM loading, data-efficiency curves against matched from-scratch and simple baselines, failure diagnostics, and (stretch) a cross-hardware consistency check.

The layout-invariant tokenizer is a baseline component, not a contribution. Our own pre-training happens only if the verification gate in §5 fails.

## 2. Problem and opportunity

- OPM-MEG is growing quickly, but public OPM datasets are small and heterogeneous: different vendors, channel counts, single-axis versus triaxial sensors, tasks and shielding setups.
- BrainOmni (NeurIPS 2025) is an EEG and MEG foundation model whose sensor encoder represents layout, orientation and sensor type [S]. It was pre-trained on 1,997 hours of EEG and 656 hours of MEG from public sources and reports generalization to unseen devices [S]. Its code is MIT-licensed [V], and checkpoints are on Hugging Face under MIT tags, including braindecode ports [V, tags only]. No OPM evaluation was found in its abstract or results tables [S]; Phase 0 confirms this.
- Other MEG foundation models (MEG-GPT, MEG-GPT-2, MEG-Mamba) are pre-trained on conventional MEG. As far as was found, their transfer to OPM is also untested [S].
- Evidence that large pre-trained neural models help is mixed. EEG benchmarks report that large models do not reliably beat specialists trained from scratch [S]. OPM data is a distribution shift: different sensor distance, noise, triaxial sensors and preprocessing.
- Small OPM labs need to decide whether to adopt pre-trained models. A careful benchmark answers that question whether the result is positive or negative.
- Related efforts may overlap (Brain4FMs, EEG-Bench, the Stanford Digital Brain Project, and braindecode's own BrainOmni integration). Phase 0 checks them.

## 3. Goals and non-goals

**Goals**

- G1. Verify BrainOmni's license, checkpoints, pre-training data, input requirements and OPM compatibility (gate in §5) and record the results.
- G2. Harmonized loading and preprocessing across OPM and SQUID datasets, matched to the model's confirmed input requirements.
- G3. A transfer benchmark: BrainOmni-tiny (primary) versus matched from-scratch and simple baselines on **at least two** OPM datasets, reported as data-efficiency curves. BrainOmni-base on a reduced grid.
- G4. Failure diagnostics: where and why transfer fails (sensor-type mapping, triaxial handling, sensor count, SNR, preprocessing, coordinate frames).
- G5 (stretch). A cross-hardware consistency test on the OSE dataset.
- G6. A reproducible open-source release (code, configs, results, model and data cards).
- G7 (conditional). An OPM-aware adaptation, built only if the diagnostics identify a specific fixable failure.

**Non-goals**

- A new foundation model, or large-scale pre-training, unless the gate fails.
- Claiming a novel tokenizer.
- 7T MRI alignment or contrastive MEG-to-fMRI/MRI bridging.
- A real-time or Triton-accelerated runtime.
- A commercial product, clinical claims or diagnostic use.

## 4. Research questions, hypotheses and success criteria

| ID | Research question | Hypothesis |
| --- | --- | --- |
| RQ1 | Does BrainOmni pre-training improve OPM task performance versus the identical architecture trained from scratch, as OPM training data varies? | H1: gains are largest at small budgets and shrink as data grows; on some tasks or datasets they may be zero or negative. |
| RQ2 | Does a pre-trained model beat a simple specialist at small OPM budgets? | H2: not always; simple baselines remain competitive on some tasks. |
| RQ3 | Which properties of OPM data predict transfer failure? | H3: sensor-type mapping, triaxial handling, channel count, SNR, preprocessing and coordinate frames explain much of the variation. |
| RQ4 (stretch) | Do embeddings of the same person agree across SQUID and OPM hardware? | H4 (OSE): agreement beats a between-subject baseline but is weaker than within-hardware agreement. |

### 4.1 Pre-registered primary comparison

Written into NOTES.md, with a commit hash, **before any transfer run**:

- **Primary dataset:** ds005107. Exact label set chosen in Phase 1 from `events.tsv` and passed through the Phase 1.5 signal gate.
- **Primary model and mode:** BrainOmni-tiny, full fine-tuning.
- **Primary control:** the same architecture, fully randomly initialized (§6, control A), with an equal hyperparameter-search budget.
- **Primary metric:** Δ = balanced accuracy (pre-trained) − balanced accuracy (from scratch), evaluated on held-out test subjects at the **smallest pre-registered training budget** (2 training subjects, or equivalent minutes if the dataset is within-subject; §9).
- **Uncertainty:** 95% bootstrap CI over test subjects (10,000 resamples), with seeds averaged within subject first.
- **Smallest effect size of interest:** to be fixed in NOTES.md before running. Proposed: Δ = 3 percentage points [E].
- **Decision rule:** a CI excluding 0 and Δ ≥ SESOI means "pre-training helped"; a CI within ±SESOI means "equivalent"; anything else is "inconclusive". All three are reported.
- **Everything else is exploratory:** other datasets, budgets, modes, the base model, diagnostics. It is reported with CIs but without confirmatory claims.

### 4.2 Success criteria

- Primary: the pre-registered comparison completed and reproducible on ds005107, plus the same protocol run on a second OPM dataset. Each dataset reports Δ at every budget with 95% subject-level bootstrap CIs. Success does not depend on the direction of the result.
- Descriptive secondary: area under the Δ-curve and the budget at which the Δ CI first includes 0 ("convergence budget").
- Secondary: diagnostics (RQ3) completed and reported. Stretch: the OSE test (RQ4).
- A null, equivalent or negative result is a valid outcome and is reported in full.
- If the verification gate fails, the project is rescoped as described in §5.

## 5. Verification gates

### 5.1 Phase 0 verification gate

No experiment starts until these checks are done and recorded in NOTES.md, each with the exact command or file used.

| # | Check | Known so far (2026-11-03) | Pass criterion | If it fails |
| --- | --- | --- | --- | --- |
| 1 | Code license | GitHub `LICENSE` contains two MIT notices (OpenTSLab 2025; Meta Platforms) [V]. GitHub reports "Other" only because the two are concatenated. | Confirm which files carry the Meta code, and that MIT covers everything used | Ask the authors; otherwise fallback track |
| 2 | Weight license and terms | HF `OpenTSLab/BrainOmni`, `braindecode/brainomni-tiny-pretrained`, `brainomni-base-pretrained` and `braintokenizer-pretrained` all tagged `license:mit` [V, tags only] | Model-card text permits research use and fine-tuning; note any rules on redistributing derived weights | Do not redistribute fine-tuned weights, or fallback track |
| 3 | Checkpoints load and run | README pins Python 3.10.14 and CUDA 12.4 and provides demo.ipynb [S] | demo.ipynb (or the braindecode equivalent) runs end to end on CPU and on a free GPU | Pin the environment or adapt; otherwise fallback track |
| 3b | Braindecode port equivalence | Ports exist [V] and were created 2026-10-07 [V]; equivalence [U] | Upstream and braindecode outputs match on one fixed input within a stated tolerance | Use upstream vendored code only |
| 4 | Pre-training data | "Curated from public sources" [S] | Dataset list with devices and licenses; no overlap with our OPM subjects; note whether **any OPM data** was used | Exclude overlaps or report as contaminated; adjust RQ1 interpretation |
| 5 | Input requirements and OPM compatibility | Encoder uses layout, orientation and type; stated MEG types are gradiometer and magnetometer [S] | Documented sampling rate, window length, normalization and units; a tested mapping of one OPM recording (sensor type, triaxial channels, units, filtering) | Treat OPM as magnetometer and state the limitation, or fallback track |
| 5b | Coordinate frames | ds005107 has `DigitizedLandmarks: false`, `DigitizedHeadPoints: false` [V]. BrainOmni's expected frame [U] | Frame of BrainOmni's training positions identified; frame of each OPM dataset's positions and orientations identified; a documented transform (or a documented limitation) | Report results as frame-mismatched; add a frame ablation to diagnostics |
| 6 | Reproduce a reported result | Paper reports EEG, MEG (ASD74) and EEG-MEG tasks [S]; ASD74 availability [U] | **Either** one reported result on public data reproduced with the tiny model within its stated spread, **or** demo outputs matching documented outputs. Cap: 6 GPU-h [E] | Record the discrepancy and weight conclusions accordingly |
| 7 | Prior OPM evaluations | None found so far; search not exhaustive [S] | Check citing papers, braindecode docs and benchmarks, Brain4FMs, EEG-Bench and the Stanford Digital Brain Project | If one exists, reposition as replication and extension |
| 8 | OPM dataset terms | ds005107 partially verified (§7) | License, BIDS completeness, sensor positions and orientations, frames, size and T1 availability confirmed for each dataset | Drop or substitute the dataset |

**Gate decision.** If items 1, 2, 3 and 5 pass, take the primary track. If any of them fails, take the fallback track: the v3.0 plan scaled down (own simple layout-invariant tokenizer, small pre-training on an open SQUID subset), or another open MEG model such as MEG-GPT or MEG-Mamba if its code and weights are public and licensed. If item 4 or 5b fails, continue with contamination or frame-mismatch analysis. If item 7 finds prior work, reposition as replication and extension.

### 5.2 Phase 1.5 signal gate (per task)

Before any transfer run on a task: the simple specialist (§6) and the classical decoder are trained at the **largest** budget. The task proceeds only if at least one of them beats chance, with a 95% subject-level bootstrap CI excluding chance. Tasks that fail are reported as "no learnable signal under our pipeline" and replaced or dropped. This prevents a benchmark where every arm sits at chance.

## 6. Models and baselines

| Arm | Description | Role |
| --- | --- | --- |
| BrainOmni-tiny, frozen | Linear probe on frozen embeddings | Primary: transfer without adaptation |
| BrainOmni-tiny, fine-tuned | Partial and full fine-tuning (full = primary mode) | Primary |
| **Control A: fully random init** | BrainOmni-tiny architecture, tokenizer **and** backbone randomly initialized, trained end to end; equal hyperparameter-search budget. If the VQ tokenizer collapses when trained from scratch, a documented recipe (e.g., codebook reset) is used and reported | **Primary control** |
| Control B: pre-trained tokenizer, random backbone | Pre-trained BrainTokenizer, randomly initialized backbone | Diagnostic: separates tokenizer transfer from backbone transfer |
| BrainOmni-base | Frozen and full fine-tuning only, at 2 budgets on the primary dataset | Exploratory size comparison |
| Simple specialist | Small CNN (EEGNet-style) or compact transformer from scratch | Guards against simple models being just as good; used in the signal gate |
| Classical decoder | e.g., ridge/logistic regression on band-power or time-domain features; CSP where appropriate | Same; runs on CPU |
| Other open MEG models | MEG-GPT or MEG-Mamba if code and weights are public and licensed | Optional comparison |
| Own layout-invariant tokenizer | Position only, then plus orientation, then plus sensor type | Fallback track and ablation only; not a contribution |
| OPM-aware adaptation | e.g., adapters or sensor-type tokens for triaxial sensors | Conditional (G7); compared against fine-tuned BrainOmni |

## 7. Data

| Dataset | Role | Facts |
| --- | --- | --- |
| BrainOmni checkpoints (HF) | Pre-trained models | See gate items 1–5 |
| ds005107 "FACE-DEC" (OpenNeuro) | **OPM target 1 (primary)** | License CC0 [V]; BIDS 1.7.0 [V]; 21 participants (IDs 05 and 12 absent) [V]; Quanmag Healthcare system [V]; 64 MEG channels, all typed `MEGMAG`, unit T [V, sub-01]; 1000 Hz, 50 Hz line [V, sub-01]; 10 face-task runs per subject, ~85 MB each [V, sub-01]; no digitized landmarks or head points [V]; total size ≈ 18 GB [E]; sensor orientations and frames [U] |
| Nottingham developmental OPM (Zenodo) | OPM target 2 | 192-channel system, somatosensory task, across ages [S]; Zenodo ID, license, subject count [U]. Age and head-size confounds noted |
| Rier et al. test-retest OPM (Zenodo) | OPM target 2 alternate / target 3 (stretch) | 10 subjects, naturalistic movie viewing, triaxial (~168 channels), T1s, sensor locations and orientations [S]; Zenodo ID, license [U] |
| OSE dataset (ATR) | Cross-hardware probe (stretch) | 5 subjects; OPM, SQUID, EEG and T1; motor, auditory, somatosensory and rest [S]; license and MNE readability of the VBMEG/MATLAB format [U] |
| FLUX example datasets | Loader tests and layout diversity | MEGIN, Cerca/QuSpin and FieldLine systems [S] |
| Cam-CAN | Fallback-track pre-training only | ~600 subjects [S]; request access only if the fallback looks likely |

All datasets must be checked for license and redistribution terms. No raw data is redistributed. Preprocessed caches are stored downsampled (to the model's input rate) to fit free-tier storage; the limits are checked in Phase 0 [U].

## 8. Requirements

### 8.1 Functional requirements

| ID | Requirement |
| --- | --- |
| FR1 | Dataset adapters built on MNE-Python and MNE-BIDS, with fallbacks for FIF, Zenodo archives and FieldTrip-format files. Every position and orientation tensor carries an explicit coordinate-frame tag. |
| FR2 | Harmonized, configurable and logged preprocessing matched to the input requirements confirmed in the gate (sampling rate, filtering, normalization, units), with device-appropriate noise reduction (e.g., SSS for SQUID, homogeneous field correction for OPM). Normalization statistics are fitted on training subjects only. Preprocessing choices are ablated. |
| FR3 | BrainOmni wrapper: load checkpoints via braindecode **if gate 3b passes**, otherwise via upstream code vendored at a pinned commit. A thin adapter maps OPM sensor metadata (position, orientation, type, triaxial channels, frame) to model inputs. Upstream code is not modified beyond that adapter. The mapping is unit-tested on one OPM recording. |
| FR4 | Transfer modes: linear probe, partial fine-tuning and full fine-tuning, plus controls A and B with an equal hyperparameter-search budget. |
| FR5 | Baselines as listed in §6. |
| FR6 | Evaluation harness: subject-wise splits, data-budget sweeps (§9), 3 seeds (5 if compute allows), bootstrap CIs over subjects, the pre-registered primary comparison, and auto-generated results tables. |
| FR7 | Diagnostics: sensor-type mapping ablation (OPM as magnetometer vs. orientation-aware), channel and axis ablations, sensor-count sweeps, SNR and preprocessing sensitivity, coordinate-frame ablation, control B, and (stretch) SQUID-versus-OPM embedding shift. |
| FR8 (conditional) | OPM-aware adaptation, built only if the diagnostics identify a specific fixable failure. |
| FR9 | Reproducibility: pinned environments, one-command experiment scripts, configs in YAML, model and data cards, CI. Release only what the licenses allow. |
| FR10 (stretch) | Static results page showing the data-efficiency curves and diagnostics. No live-inference or real-time claims. |

### 8.2 Non-functional requirements

- Zero spend: free-tier GPUs (Kaggle, Colab), GitHub, Hugging Face Hub and free logging (CSV/TensorBoard) only.
- All runs checkpoint and resume to persistent storage, because free sessions disconnect. Develop and test on CPU first.
- No large-scale pre-training on the primary track.
- Deterministic seeds, unit tests for the loaders and the sensor-metadata mapping, CI.
- Our code under a permissive license (MIT, matching upstream), subject to upstream constraints.

**Compute budget [E, to be re-estimated after gate 3 timing runs]**

Free-tier reference: Kaggle about 30 GPU-h/week with 12 h sessions; Colab free has no fixed quota [S]. Confirm in account settings in Phase 0.

| Block | Runs | GPU-h [E] |
| --- | --- | --- |
| Gate 3 / 3b / 6 checks | few | ≤ 8 |
| HP search (tiny, pre-trained and control A, equal budget) | ~2 × 10 trials × 1 dataset | 5–8 |
| Primary grid: 2 datasets × 5 budgets × 3 seeds × {frozen, partial, full, control A} | 120 | 15–25 |
| Control B + simple specialist (GPU part) | ~60 | 5–8 |
| BrainOmni-base: 2 budgets × 3 seeds × {frozen, full} × 1 dataset | 12 | 5–10 |
| Diagnostics (reduced budgets) | ~40 | 5–10 |
| **Total** | | **~45–70 GPU-h ≈ 2–3 weeks of Kaggle quota** |

Classical decoders and linear probes on cached embeddings run on CPU. If timing runs show the budget is exceeded, cut in this order: base model, then partial fine-tuning, then the number of diagnostics.

## 9. Evaluation protocol

- **Splits:** always by subject. Outer loop: subject-wise K-fold with fixed test subjects per fold (for ds005107, proposed 5 folds of about 4 test subjects [E]). Inner: validation subjects drawn from the training pool only. No subject appears in both training and test at any stage. Overlap with BrainOmni pre-training data is checked (gate 4).
- **Budgets:** nested subsets of the training pool: 1, 2, 5, 10 and all available training subjects (cross-subject), or equivalent minutes per subject (within-subject). The subset order is fixed per seed and logged.
- **Decoding regime and tasks (fixed in Phase 1, before the signal gate):**
  - ds005107: **cross-subject** decoding of face-task stimulus classes defined from `events.tsv` (exact classes [U] until the events are inspected).
  - Nottingham developmental: **cross-subject** classification of somatosensory stimulus vs. baseline epochs (or stimulus condition, if available).
  - Rier et al. (stretch): **labelled** naturalistic tasks only, e.g., movie-segment identification across subjects, or test-retest subject fingerprinting. Self-supervised objectives are never the headline metric.
- **Metric:** balanced accuracy (primary); macro-F1 and AUROC as secondary. Chance level is stated per task.
- **Main figure:** OPM task performance versus OPM training budget for each arm in §6, per dataset, with CIs; plus the Δ-curve.
- **Fairness:** equal hyperparameter-search budget (same number of trials, same search space size) and identical preprocessing for pre-trained and control arms.
- **Statistics:** 3+ seeds; bootstrap CIs over test subjects; one pre-registered primary test (§4.1); everything else labelled exploratory; effect sizes reported. Negative results are reported in full.
- **Diagnostics (RQ3):** which OPM properties predict transfer failure (§8.1 FR7).
- **Cross-hardware test (OSE, stretch):** agreement of embeddings for the same subject and task across SQUID and OPM sessions, against a between-subject baseline. Sessions are separate recordings, not simultaneous.
- **Confounds:** document differences in SNR, shielding, sensor distance, task, coordinate frames and preprocessing across hardware, and report whether gains survive harmonization.
- **Too-good-to-be-true rule:** any result clearly above prior expectations is treated as a bug (leakage, label error, unit mismatch) and investigated before reporting.

## 10. Milestones (about 10 weeks, plus stretch)

| Weeks | Milestone | Exit criterion |
| --- | --- | --- |
| 0 (first 3–5 days) | Phase 0 verification gate (§5.1) | All checks recorded; primary or fallback track chosen; compute budget re-estimated |
| 1–2 | Loaders and harmonized preprocessing (ds005107, then target 2) | Both datasets load into the common tensor and metadata format with frame tags; preprocessing report generated |
| 2.5 | Phase 1.5 signal gate | Tasks fixed; specialist beats chance on each retained task; **primary comparison pre-registered and committed** |
| 3–4 | BrainOmni wrapper, controls and baselines | Tiny checkpoint runs on one OPM recording through the tested mapping; controls A and B and the baselines train |
| 5–6 | Transfer experiments on ds005107 | Primary comparison and data-efficiency curves with CIs; **first outreach gate** |
| 7–8 | Second OPM dataset and diagnostics | Curves on two OPM datasets; diagnostics reported |
| 9 | Optional adaptation and write-up | Adaptation only if justified; draft technical report |
| 10 | Release | Public repository, results tables, model and data cards, short preprint |
| Stretch | Third OPM dataset (Rier), OSE probe (RQ4), base-model extension, results page | As time and quota allow |

## 11. Risks and mitigations

| Risk | Mitigation |
| --- | --- |
| BrainOmni's license or checkpoints are unusable. | Gate decision sends the project to the fallback track or another open model. |
| An OPM transfer benchmark of BrainOmni already exists. | Gate 7 checks first; if found, reposition as replication and extension. |
| Pre-training does not help, or hurts, on OPM. | Pre-registered comparison with an equivalence band; the negative result is reported with the conditions under which it holds. |
| OPM metadata cannot be mapped faithfully (sensor type, triaxial channels, orientation conventions). | Document the limitation, restrict claims to the mapped configuration, ablate mapping choices. |
| **Coordinate-frame mismatch** (e.g., ds005107 lacks digitized landmarks). | Gate 5b; frame tags in every tensor; frame ablation; results labelled accordingly. |
| **Tasks have no learnable signal.** | Phase 1.5 signal gate; replace or drop the task. |
| Primary-metric noise with ~21 subjects. | Smallest-budget Δ with a subject-level bootstrap, a pre-registered SESOI, and an "inconclusive" category. |
| BrainOmni's pre-training data overlaps our OPM targets. | Contamination analysis; exclude overlapping subjects or datasets. |
| Labels and tasks differ across datasets. | Dataset-specific tasks and budget curves instead of one shared label set. |
| Preprocessing differences confound comparisons. | One harmonized pipeline with preprocessing ablations. |
| Free-tier compute or storage limits; pinned Python/CUDA environment. | Tiny model primary; braindecode path if equivalent; downsampled caches; checkpoint and resume; budget table with a cut order. |
| Overlap with larger programs (e.g., Stanford Digital Brain Project). | Release early and small; consider collaboration. |

## 12. Deferred ideas (not in scope)

- A new tokenizer or foundation model as the contribution. Revisit only if the benchmark reveals a specific failure that an adaptation fixes.
- Pre-training with OPM data included, as a v2 of the benchmark.
- MEG-to-7T MRI or fMRI alignment, which first needs a precise definition of the positive pairs and the downstream payoff.
- A real-time streaming runtime and edge deployment.
- Commercial and fundraising plans, which need a validated result and an identified customer or partner such as an OPM hardware vendor.
- Source-space tokenization as a full alternative path.

**Positioning note.** This benchmark is meant as the credibility step toward any later OPM + MRI collaboration. It produces a public result, reusable tooling and relationships with groups that hold OPM and MRI data. It does not change the current scope.

## 13. Outreach and collaboration plan

- **Gate:** reach out only after the first transfer result (end of week 6), with a short message, one figure, a repository link and one specific ask (e.g., running the harness on one of their datasets), not a vision document.
- **Candidate groups:** the BrainOmni authors (they can say how the model should be used on OPM), the Oxford group behind MEG-GPT, the Nottingham OPM-MEG group, the Stanford Digital Brain Project, the Jerbi lab (Université de Montréal) and, once the porting authors are verified, the braindecode maintainers.
- **Offer:** a hardware-agnostic benchmark and diagnostics they can run on their own data, with co-authorship on extensions where appropriate.
- **Claims discipline:** describe only what has been run and measured. Confirm author lists, affiliations and citations before naming anyone.

## 14. Open questions

- Do OPM recordings need sensor-type handling that BrainOmni's encoder does not offer, and how large is the cost of ignoring it?
- Which coordinate frame did BrainOmni's pre-training use, and can OPM datasets without digitized landmarks be mapped into it?
- Which ds005107 stimulus classes give a learnable cross-subject signal?
- Can additional public OPM datasets be added before release?
- Should a v2 include OPM data in pre-training?

## Sources to verify

Items marked [S] come from search snippets of these pages, not full reads. §5 exists to check them.

- [BrainOmni paper (arXiv 2505.18185)](https://arxiv.org/abs/2505.18185)
- [BrainOmni code, README and LICENSE](https://github.com/OpenTSLab/BrainOmni)
- [BrainOmni on Hugging Face (OpenTSLab)](https://huggingface.co/OpenTSLab/BrainOmni)
- [braindecode BrainOmni-tiny port](https://huggingface.co/braindecode/brainomni-tiny-pretrained) · [base](https://huggingface.co/braindecode/brainomni-base-pretrained) · [tokenizer](https://huggingface.co/braindecode/braintokenizer-pretrained)
- [BrainOmni at NeurIPS 2025](https://neurips.cc/virtual/2025/poster/117066)
- [ds005107 FACE-DEC (OpenNeuro)](https://openneuro.org/datasets/ds005107)
- [A Roadmap for MEG Foundation Models (arXiv 2609.04461)](https://arxiv.org/pdf/2609.04461)
- [MEG-Mamba (arXiv 2610.00746)](https://arxiv.org/pdf/2610.00746)
- [MEG-GPT (arXiv 2510.18080)](https://arxiv.org/abs/2510.18080)
