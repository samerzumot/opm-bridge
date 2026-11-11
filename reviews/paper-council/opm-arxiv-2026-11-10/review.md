# arXiv readiness assessment

Role: scientific editor and council chair. Date: November 10, 2026.

**Verdict: ready for submission within the verified scientific scope after the localized corrections made in this arXiv draft. Public release still requires author-controlled disclosure and submission decisions.** The paper is a defensible exploratory empirical study of a frozen model's sensor interface. This is a readiness assessment, not a forecast of arXiv moderation or journal acceptance.

## Scope and versions

The current revised workshop paper and supplement were reviewed, rather than the older broad manuscript or repository README. Version hashes are recorded in `packet-sha256.json`. The final preprint is `paper/arxiv/output/pdf/opm_bridge_arxiv.pdf` and its minimal upload archive is `paper/arxiv/opm_bridge_arxiv_source.tar.gz`.

Two independent agents performed focused methods/statistics and claims/contribution/editorial reviews. Their original first-pass findings and closure addenda are preserved in `methods.md` and `claims.md`. The chair inspected the numerical verifier, relevant implementation paths, arXiv guidance, source conversion, archive compilation and PDF layout. This reduced-scope council did not perform a comprehensive novelty or literature audit.

## Supported contribution

- The single-axis categorical intervention changes reconstruction from 0.2501 to 0.3819 PCC, with the paired direction consistent in 21 of 21 participants. Approximate geometry limits interpretation of the spatial comparisons.
- The full triaxial design crosses tag, filtering and partition. Axis grouping is compared with size-matched random partitions averaged over ten seeds within each of ten participants.
- Independent recalculation gives Soft-HFC axis-minus-random gains of 0.107142 under MAG and 0.074536 under GRAD, with interaction -0.032607. Directions are consistent across all ten participants; exact two-sided sign-flip p is 0.001953125.
- Reconstruction, neural fidelity and decoding remain distinct endpoints. The paper does not claim equivalence or established decoding improvement from the null probe comparisons.

## Adjudicated issue ledger

| ID | Severity | Roles | Evidence and consequence | Disposition and closure |
| --- | --- | --- | --- | --- |
| A1 | Minor | M1, CE1 | Workshop supplement reports C=1.0; the factorial runner uses a common helper whose default is C=0.01. | Confirmed and corrected in arXiv supplement_text.tex:60. Added training-fold scaling, class weights, iterations, split seed and a historical-execution caveat. |
| A2 | Minor | M2, CE4 | Raw cached trials already have notch/bandpass filtering, baseline correction and resampling. | Confirmed and corrected in main_text.tex:28 and supplement_text.tex:65. Raw means no additional Soft-HFC. Nominal high-frequency features are qualified by the 45 Hz low-pass. |
| A3 | Minor | M3, CE2 | Three passes and 48 aggregate queries do not establish threefold runtime or FLOPs. | Confirmed and corrected in the Discussion. Proposed design efficiency remains untested. |
| A4 | Minor | CE3 | Zero-shot could be mistaken for held-out waveform prediction. | Clarified in main_text.tex:22: frozen, unmasked reconstruction of supplied input. |
| A5 | Minor | Claims | Interaction supports dependence on tag, without identifying a shared mechanism. | Abstract and conclusion now state the observed tag dependence. |
| A6 | Author information | CE5 | Front matter and disclosure facts were incomplete. | User confirmed author order and affiliations; included. Added factual code/data and AI-assistance sections. Funding, competing interests, secondary-use ethics/license coverage and final author review remain pending. |

Confidence in A1-A5 is high because each finding is tied to inspectable source and implementation. A6 records absent information; it is not a finding of improper conduct. No scientific blocker or unresolved major issue was established within this scope.

## Disagreements and limits

The reviewers agreed on the required corrections. Agreement is not treated as additional empirical evidence. Both reviews distinguish code inspection from verification of original execution. The chair retains that caveat explicitly in the preprint. The older broad hardware-generalization and capacity-mechanism claims were not restored.

One model, two small OPM cohorts, one SQUID participant, approximate FACE-DEC geometry, exploratory multiplicity and artifact-sensitive reconstruction bound the inference. A matched-query monolithic control, artifact-robust neural endpoint, larger decoding sample and new model family would strengthen a later paper, but are not prerequisites to publishing the current bounded findings.

## Verification performed

- Executed `paper/neuroai2027/audit/verify.py`: passed, recalculating results from saved participant and trial records and checking macros, figure values, tables, citations and the existing builds.
- Independent methods reviewer recomputed the central factorial contrasts in memory from saved trial records.
- Executed `paper/arxiv/build.py`: passed. Final PDF has 13 pages, 8 references and 8 supplementary tables. No unresolved references, duplicate reference warnings or overfull boxes.
- Extracted the source archive into a fresh temporary directory and compiled it successfully with only the packaged files. PDF text matches the local build.
- Numerical macros and figure are byte-identical to the verified workshop assets. Supplementary table changes are limited to float environment syntax; numerical contents are unchanged.
- Rendered all final pages and inspected the layout, author block, figure, references and tables.
- Checked current official arXiv submission, TeX, moderation and category guidance. Suggested primary category is cs.LG; q-bio.NC is a possible cross-list, subject to author choice and moderator classification.

Not performed: raw-data replay, new model inference, checkpoint/environment certification, comprehensive citation correction/retraction checks, comprehensive novelty assessment, arXiv server compilation or actual submission. The cited repository's anonymous public accessibility was not established by the attempted web fetch.

## Minimum remaining release actions

1. Authors review and agree to the final scientific text and AI-assistance disclosure.
2. Supply applicable funding, competing-interest and secondary-use ethics/consent/license statements. No unsupported declarations have been inserted.
3. Confirm public availability of the intended code/results, choose an arXiv distribution license, and check the workshop's applicable preprint policy.
4. Upload the checked source archive as pdfLaTeX, inspect the arXiv-generated PDF, confirm metadata, and submit through the author's account. Nothing has been uploaded or submitted here.

## Policy sources

Checked November 10, 2026: [submission guidance](https://info.arxiv.org/help/submit/index.html), [TeX packaging](https://info.arxiv.org/help/submit_tex.html), [moderation and AI assistance](https://info.arxiv.org/help/moderation/index.html), [category taxonomy](https://arxiv.org/category_taxonomy), and [workshop guidance](https://neuroai-multimodal-workshop.github.io/).

## Author-block update

The confirmed author order remains Samer Zumot (1), followed by Marzieh Zare (1,2). The user supplied updated affiliations only: (1) NeuroGenis Inc. Toronto, Canada; (2) Université Laval, School of Psychology, Quebec City, QC, Canada. An editing error briefly reversed the authors; this was corrected in the preprint, PDF metadata and submission metadata. The PDF and source archive were rebuilt. Scientific content is unchanged.

## Bibliography update

The introduction now adds three targeted sources: Boto et al. (2017) for near-scalp OPM measurements, Boto et al. (2018) for wearable MEG during head movement, and Yang et al. (2023, BIOT) for heterogeneous biosignal representation. The bibliography now contains 11 cited entries. Source locators and verification limits are recorded in `paper/arxiv/reference_additions.json`. Scientific results, Samer-first author order, and full affiliations are unchanged. The updated PDF remains 13 pages and the standalone archive build passed. The earlier verification count of eight references describes the initial draft.
