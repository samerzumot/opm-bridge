# Methods and statistics review

Role: methods and statistics reviewer. Review date: 2026-11-10.

## Initial independent report

Verdict: **minor revision for arXiv**, with no demonstrated blocker to the narrowed reconstruction contribution.

Reviewed `paper/neuroai2027/short_paper.tex`, SHA-256 `8ffc8e42b6ff3d8a781f7183be7c6e36db4668163f7bce88682fc6ab9422b717`, and `paper/neuroai2027/supplement.tex`, SHA-256 `2c54efe39049f68c80c3f76a645f7c6d2324da399e774da1acc6d06f19dd6f32`, with generated values, saved factorial records and relevant implementation. Independent first pass without consulting previous reviews.

### Supported strengths

- The matched factorial design holds filtering constant, compares axis and random partitions with matching sizes and pass counts, and averages random seeds within participants. Participants remain the inference unit (`short_paper.tex:47`, `factorial_results.py:55-70`).
- Recalculation from saved trial records confirms Soft-HFC axis-minus-random PCC gains of 0.107142 under MAG and 0.074536 under GRAD, with interaction -0.032607. Each contrast has consistent direction across ten participants and exact two-sided sign-flip p=0.001953125. This was a read-only call to `factorial_results.load()`, not model inference.
- The manuscript distinguishes reconstruction from neural-signal recovery, treats decoding as inconclusive, acknowledges exploratory multiplicity, and declines to isolate capacity without a matched-query monolithic control (`short_paper.tex:43`, `:50`, `:111`, `:118`, `:125`).

### Findings

**M1. Minor, high confidence: incorrect decoder regularization.**

Location: `paper/neuroai2027/supplement.tex:87`. The text reports C=1.0. The factorial runner calls `eval_linear_probe_from_embeddings` for all three feature arms at `scripts/run_nottingham_tag_partition_factorial.py:264` and `:347`; its actual default is C=0.01 at `src/opm_bridge/eval/metrics.py:103`. The classical decoder object's C=1.0 pipeline is not used for fitting in this experiment. This discrepancy impairs replication of the reported decoding setup without undermining the primary reconstruction comparisons.

Remedy: report C=0.01, training-fold StandardScaler, balanced class weights, and max_iter=500 for all arms. Replace the assertion that no selection leakage is possible with the narrower fact that regularization was fixed rather than selected on held-out folds. Closure criterion: arXiv methods agree with this execution path. Confidence is high from the inspected caller and callee; the saved scores do not independently establish which code version originally executed.

**M2. Minor, high confidence: raw preprocessing is underspecified.**

Locations: `paper/neuroai2027/short_paper.tex:47` and `supplement.tex:79`. The text contrasts raw and Soft-HFC without explaining that cached triaxial inputs already receive 50/100 Hz notch filtering, 1-45 Hz bandpass filtering, baseline subtraction, and resampling to 256 Hz (`scripts/stream_nottingham_zenodo.py:155-178`). The nominal six-band classical feature set extends to 96 Hz, above the low-pass cutoff. This can mislead readers about the experimental contrast and the information available to the decoder.

Remedy: explicitly define raw as without additional Soft-HFC, document cached preprocessing and epoch bounds, and note attenuation of upper feature bands. Describe these settings as code-derived where raw-data provenance remains unverified. Closure criterion: readers can distinguish acquisition data, cached preprocessing, and the experimental Soft-HFC manipulation. Confidence is high regarding supplied code and missing description, not raw-data execution provenance.

**M3. Minor, high confidence: compute claim exceeds measurements.**

Location: `paper/neuroai2027/short_paper.tex:123`. The text says partitioning triples tokenizer compute and proposed axis-conditioned queries operate at neither cost. Three passes and three 16-query banks are supported; threefold total runtime or FLOPs are not. The encoder applies temporal convolution separately per channel (`src/opm_bridge/models/vendor_brainomni/model_utils/module.py:124-134`), so dividing channels across passes does not triple every component. No benchmark establishes the efficiency of the proposed architecture.

Remedy: describe three passes and 48 aggregate query slots, with unmeasured runtime and memory costs. Present efficiency of proposed axis-conditioned queries as untested. Closure criterion: remove unsupported exact compute and cost-free claims. Confidence is high from the architecture and absence of a reported computational benchmark.

### Optional clarification

Participant-bootstrap uncertainty conditions on existing predictions, fixed five-fold splits, and the selected random-partition seeds. It does not include checkpoint variation, refitting across new splits, or new acquisition sessions.

### Verification limits

Inspected scripts and regenerated factorial statistics in memory. No model inference or raw-data replay. No independent confirmation of cached trial provenance, checkpoint hashes, saved partition assignments, or participant independence beyond supplied metadata. Those limits are already substantially acknowledged. Continuous metadata and axis-conditioned architecture extensions are future work, not prerequisites for this exploratory preprint.

## Closure addendum for the arXiv version

Reviewed `paper/arxiv/main_text.tex` and `paper/arxiv/supplement_text.tex` on 2026-11-10 after the focused revisions. No manuscript files were edited by this reviewer.

- **M1 closed.** `supplement_text.tex:60` now gives C=0.01, balanced class weights, max_iter=500, training-fold scaling and stratified five-fold splitting. It confines the no-selection claim to fixed regularization and distinguishes code inspection from independent execution provenance.
- **M2 closed.** `main_text.tex:28` now defines raw as cached inputs without additional Soft-HFC and records notch frequencies, bandpass, epochs, baseline correction, sampling rate and 512-sample trials. `supplement_text.tex:65` describes the nominal feature bands and their attenuation by preprocessing. The chair subsequently replaced the remaining without filtering and unfiltered interval wording with without additional Soft-HFC and rendered raw with emphasis. These changes were verified in the current source and close the residual editorial consistency points.
- **M3 closed.** `main_text.tex:104` reports three passes over smaller channel sets and aggregate query count 16 to 48; it explicitly states runtime and FLOPs were not benchmarked and labels the proposed architecture's performance and cost as untested.
- The optional uncertainty clarification appears in `main_text.tex:31`. The added unmasked-reconstruction definition at `main_text.tex:22` agrees with the inspected evaluation code. The narrowed interaction wording in the abstract and conclusion accurately describes the paired factorial contrast.

No new material methodological error was identified. Within the verified scope, the arXiv version is scientifically ready; M1-M3 and their associated editorial consistency points are closed. This assessment does not establish raw-data reproducibility or predict arXiv moderation.
