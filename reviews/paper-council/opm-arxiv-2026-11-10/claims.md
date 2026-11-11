# Claims, contribution, and editorial review

Role: claims, contribution, and editorial reviewer. Review date: November 10, 2026. This is a simulated scientific reviewer role, not an actual arXiv moderation decision.

## Initial independent report

Recommendation: **minor revision for arXiv preparation**. The narrowed paper has a defensible contribution. No demonstrated blocker to its central reconstruction claim was identified within this review's scope.

### Version and scope

- `paper/neuroai2027/short_paper.tex`, SHA-256 `8ffc8e42b6ff3d8a781f7183be7c6e36db4668163f7bce88682fc6ab9422b717`.
- `paper/neuroai2027/supplement.tex`, SHA-256 `2c54efe39049f68c80c3f76a645f7c6d2324da399e774da1acc6d06f19dd6f32`.
- Bibliography and selected analysis, preprocessing, tokenizer, and decoder code. No prior audit reports were consulted before this first-pass judgment.

### Supported contribution

The strongest contribution is a controlled characterization of a frozen model's sensor interface: paired categorical-tag interventions, axis grouping against random partitions with matched group sizes and pass counts, and a crossed design showing configuration dependence. The paper explicitly distinguishes reconstruction from neural fidelity and decoding utility (`short_paper.tex:43,109-128`). Its negative decoding result is appropriately inconclusive, with no equivalence claim (`short_paper.tex:114-118`). The approximate FACE-DEC geometry limitation is supported directly by the [dataset documentation](https://raw.githubusercontent.com/OpenNeuroDatasets/ds005107/master/README), inspected during this review.

### Initial findings

**CE-1. Incorrect decoder regularization. Minor; high confidence.**

Location: `paper/neuroai2027/supplement.tex:87`. The text states `C=1.0`. The factorial script calls `eval_linear_probe_from_embeddings` without overriding `c_reg` (`scripts/run_nottingham_tag_partition_factorial.py:264,347`). That helper defaults to `c_reg=0.01`, used as `C` (`src/opm_bridge/eval/metrics.py:103,112`). It also fits `StandardScaler` only on training folds, uses balanced class weights, and allows 500 iterations (`metrics.py:108-115`). The mismatch misdescribes the supplied analysis implementation. Report the code accurately while distinguishing code inspection from verification of the historical execution configuration. Remove the blanket statement that no selection leakage is possible: fixed regularization excludes this particular tuning pathway, not every selection pathway. Closure criterion: correct the parameter and scope wording and preserve a provenance caveat if original run settings cannot be confirmed.

**CE-2. Computational claim exceeds evidence. Minor; high confidence.**

Location: `paper/neuroai2027/short_paper.tex:123`. The text says axis partitioning triples tokenizer compute and predicts a new query design avoids both costs. Code establishes three calls over smaller channel groups, not a measured threefold runtime or FLOP cost (`scripts/run_nottingham_tag_partition_factorial.py:197-208`). Channelwise convolution processes the same total channels, while query-related work increases (`src/opm_bridge/models/vendor_brainomni/model_utils/module.py:124-136`). The computational burden is therefore inaccurately quantified. State that partitioning uses three tokenizer passes and triples the aggregate latent query count; computational cost was not benchmarked. Describe prospective efficiency benefits as hypotheses. Closure criterion: replace the numerical compute claim and untested guarantee.

**CE-3. Clarify the reconstruction endpoint. Minor; high confidence.**

Location: `paper/neuroai2027/short_paper.tex:41-43`. Frozen reconstruction is described without explicitly stating that the full input is supplied without masking. Evaluation directly invokes encoder, quantizer, and decoder (`scripts/run_nottingham_tag_partition_factorial.py:74-78,223-228`). A reader could conflate zero-shot reconstruction with prediction of withheld samples. Define zero-shot as inference without OPM-specific weight updates and identify the endpoint as unmasked reconstruction. Closure criterion: an explicit methods sentence makes this distinction.

**CE-4. Raw is relative to Soft-HFC. Minor; high confidence.**

Location: repeated raw-condition descriptions in `paper/neuroai2027/short_paper.tex`, especially lines 47 and 109. The stream script applies 50/100 Hz notch filtering, 1-45 Hz band-pass filtering, baseline correction, epochs from -0.2 to 1.8 seconds, and resampling to 256 Hz (`scripts/stream_nottingham_zenodo.py:155-178`). The terminology could imply unprocessed recordings. Add these implementation details with the raw-data replay caveat and define raw as without additional Soft-HFC. Closure criterion: methods state the processing history and avoid implying untouched recordings.

**CE-5. Author-controlled disclosures. Information question; high confidence that fields are absent.**

The initial files are anonymous and contain no affiliations, code availability section, funding/conflict statement, secondary-use ethics statement, or significant AI-assistance disclosure. Missing statements do not establish an ethics defect. Confirm author-controlled facts before public release; do not invent approvals or conflicts. arXiv's [current policy](https://info.arxiv.org/help/moderation/index.html#policy-for-authors-use-of-generative-ai-language-tools), inspected during this review, requires reporting significant generative-AI use and assigns responsibility to human authors. Closure criterion: factual, author-reviewed front matter and disclosures appropriate to the actual work.

**CE-6. Optional interaction wording. Suggestion; high confidence.**

Replace the abstract's statement that tag and grouping effects partly overlap with the more direct statement that the grouping advantage depends on sensor tag. The interaction supports nonadditivity; the supplement correctly says it does not identify a shared mechanism (`paper/neuroai2027/supplement.tex:84`). No new experiment is needed for this editorial clarification.

### Initial verification limits

No models were rerun. Cached data were not verified against raw recordings. Reported statistics were not independently recalculated by this reviewer. Final PDF layout, all citations, and comprehensive novelty were not audited. The BrainOmni primary PDF fetch timed out. These are evidence limits, not evidence that the study is wrong. The verdict concerns presentation and claim scope, not guaranteed arXiv moderation approval.

## Closure addendum

Re-reviewed November 10, 2026: `paper/arxiv/main_text.tex`, `supplement_text.tex`, `main.tex`, and `README.md`. The arXiv text was examined separately from the original workshop sources, which were not changed by this reviewer.

Hashes at closure-pass inspection:

- `main_text.tex`: `92087b6cc3c26c5c2bdd1d32dab1e85ce1a10aab15018a16dc166c2cd891d79c`.
- `supplement_text.tex`: `1501def9700b3aa3c71065accdfa5a27fbe8006518c2828cabd9c32234279441`.
- `main.tex`: `69b5ce6b3ba2fc05c95a4761b1bf30ef3f0969ac93297e45f57f5d6369123134`.

| Finding | Disposition | Evidence |
| --- | --- | --- |
| CE-1 | Resolved for supplied-code reporting | `supplement_text.tex:60` reports C=0.01, training-fold scaling, balanced weights, 500 iterations, fixed regularization, and split seed. It explicitly distinguishes current code from historical fitted settings. Historical execution provenance remains an acknowledged evidence limit. |
| CE-2 | Resolved | `main_text.tex:104` reports three passes over smaller channel sets and 16 to 48 aggregate queries. It explicitly says runtime and FLOPs were not benchmarked and future performance/cost are untested. |
| CE-3 | Resolved | `main_text.tex:22` defines zero-shot and states that reconstruction is unmasked with the complete input supplied. |
| CE-4 | Substantively resolved; small wording cleanup remains | `main_text.tex:28` documents prior filtering, epochs, baseline correction, sample rate and cache length, with a replay caveat. `supplement_text.tex:65` qualifies nominal high-frequency bands. The residual word unfiltered should become without additional Soft-HFC. |
| CE-5 | Partially resolved; author-controlled facts remain open | `main.tex` restores author names, `main_text.tex:111-115` adds availability and factual AI-assistance sections, and `README.md` explicitly identifies author order/affiliations, funding/conflicts, secondary-use statements, public repository accessibility, and license as author-controlled items. These are properly left unasserted rather than invented. |
| CE-6 | Resolved in abstract and conclusion | `main_text.tex:2,109` expresses the interaction as dependence of grouping advantage on tag. The older phrase remains in the results at line 90, but the supplement preserves the no-shared-mechanism caveat, so this is not a scientific blocker. |

One newly introduced editorial defect was found at `main_text.tex:28`: the TeX opening quotation before raw lacks its closing quotation. Remove the opening quotation or balance it. This was reported to the chair. No newly introduced substantive scientific error was identified in this pass.

Updated recommendation: the scientific wording issues identified by this reviewer are resolved, subject to the two small terminology/quotation cleanups above. The draft is suitable for author review as an arXiv preprint. Public-release information and author agreement remain unresolved as stated in the README. This closure pass does not certify numeric regeneration, compilation, final PDF layout, original model execution, or publication eligibility.

### Final closure check

The two small cleanups were verified directly in the files: `main_text.tex:28` now uses `\emph{raw}`, while `main_text.tex:97` and `supplement_text.tex:65` use without additional Soft-HFC. CE-4 is fully resolved for reporting the supplied implementation. The introduced quotation defect is resolved.

The chair relayed the user's confirmation of Samer Zumot first and Marzieh Zare second, both at NeuroGenis Inc., with Marzieh also at Universite Laval. The front matter at `main.tex:18-20` matches those details and includes the accented university spelling. This closes the author-order and affiliation portion of CE-5. Funding, competing interests, and secondary-use statements remain pending author input; no unsupported statements about them have been inserted.

Final inspected hashes:

- `main_text.tex`: `5f5a3a23f5a950d15d1e528e4ed88570e307cb4f2957029b6a828696b005cc6d`.
- `supplement_text.tex`: `e691a13c8c612912b7a358fc59a5a105d3d5ef65f2e46246667ed0df23ecb571`.
- `main.tex`: `2136d901685d441b1808b8dd58eb452e4788c3666f611afe4d0ddfaaa6f086ec`.

Final role verdict: **ready for scientific author review as an arXiv preprint within the verified claims/editorial scope**. No identified scientific wording correction remains required from this role. The remaining author-controlled release information and the verification limits above still apply.
