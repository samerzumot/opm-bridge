# Changes
Created a separate anonymous short-paper source using the official AAAI-27 submission style. Original manuscript and analysis code unchanged.
Reframed title, abstract, and conclusion around the evaluated BrainOmni tokenizer and reconstruction endpoint.
Retained source-traceable means and intervals; removed unverified p-value claims.
Removed 76.1% causal attribution, physiological interpretation of detection, universal SNR floor, and claims of recovered reconstruction loss.
Corrected autoregressive to masked-token modeling using primary paper and implementation.
Labeled the filtered adapter and unfiltered random control; explained capacity and subset differences.
Corrected radial/tangential ranges from saved subject records and described geometry matching and scrambling from code.
Replaced author and repository identifiers with anonymous review formatting. Corrected BrainOmni authors and added primary FACE-DEC citation and public geometry documentation.
Added limitations on batch weighting, coordinate fidelity, reconstruction interpretation, and single-subject SQUID scope.
Independent final methods review found two wording errors in the first draft; both corrected. No further material discrepancy reported within that review's scope.

## Figure restoration
Added two vector reconstruction figures generated directly from the saved summary JSON files by generate_figures.py. Figure 1 labels scrambled geometry accurately and identifies common Soft-HFC preprocessing. Figure 2 labels filtering per condition, includes only reconstruction, and distinguishes subset conditions. Both captions define subject-bootstrap 95% confidence intervals and retain interpretation limits. Existing numerical tables remain available for exact values. Original figure files are unchanged.

## Revision 2: paired statistics, matched controls, restored evidential weight
Rebuilt the figure from the saved per-participant records rather than the cohort summaries. The single figure now shows paired trajectories and preprocessing-matched contrasts, so within-participant consistency is visible rather than only cohort means and overlapping intervals. The previous two figures redrew the tables and were removed.
Added the preprocessing-matched triaxial contrast that the saved design supports but neither the original manuscript nor the first draft reported: axis partition versus monolithic tokenizer under identical Soft-HFC, +0.1388 PCC in 10 of 10 participants. The original manuscript used the unmatched +0.1123 contrast against an unfiltered baseline, which understates the effect because Soft-HFC alone lowers monolithic reconstruction by 0.0265 in every participant.
Added the matched capacity control, random three-way partition versus monolithic with shared preprocessing, +0.0269 in 10 of 10. Pass count accounts for roughly one fifth of the matched axis effect. The withdrawn 76.1 percent attribution is not restored; the decomposition is stated with its additivity assumption named.
Added paired consistency counts and exact sign-flip permutation p-values for every contrast, and noted that the smallest attainable p at n=10 is 2/1024, so the saved p-values indicate perfect consistency rather than a resolved tail.
Added the classical band-power baseline to the triaxial decoding table. It exceeds the pretrained linear probe in all six configurations by 3.4 to 6.1 points, which converts the decoding null from an assumed dataset noise floor into a representational failure. Neither the original manuscript nor the first draft reported this comparison.
Promoted the excluded detection task from a limitation sentence to a short results subsection with the concrete construction defect, since the failure mode generalizes beyond this benchmark.
Rewrote title, abstract and conclusion to lead with the supported findings and the design implications that follow from them, keeping every correction from revision 1.
Extended audit/verify.py to recompute all paired contrasts, consistency counts and permutation p-values from per-participant records and to assert that each quoted value appears in the manuscript.

## Revision 3: figure carries the decoding result, and a count correction
Corrected a miscount introduced in revision 2. The classical band-power baseline leads the pretrained probe in five independently measured triaxial configurations, not six. The sixth, random_3way_split, duplicates the monolithic raw row in all three decoding arms, not only in the pretrained probe arm as previously checked, so it was never an independent measurement. The abstract and the decoding subsection now say five, and add that the classical decoder leads in 8 of 10 participants in each.
Rebuilt the figure as six panels in two rows. Row one is sensor metadata on the single-axis cohort: paired MAG to GRAD trajectories, all four metadata interventions as paired contrasts, and the single-participant SQUID control as bars. Row two is triaxial channel organisation: paired monolithic to axis trajectories, the preprocessing-matched contrasts, and the Braille decoding endpoint.
Added the SQUID control panel. Relabelling 102 native SQUID magnetometers as gradiometers reaches 0.713, above the 204 native gradiometers at 0.447. That comparison was previously one clause of prose.
Added the decoding panel and removed the decoding table it supersedes. The panel shows both arms per configuration against a chance line with participant-bootstrap intervals, which makes visible what the table concealed: the per-configuration paired tests are weak, with intervals crossing zero in three of five, while the direction is unanimous at 8 of 10 participants in every configuration. The paper's reading of the null result does not change, but the evidence for it is now shown rather than asserted.
Moved consistency counts from inside the axes into the row labels, which removes the marker collisions present in the revision 2 panels.
Added figures/transfer_overview_values.json, a sidecar written by generate_figures.py listing every value the figure displays. Decoding values now live in the figure rather than in the manuscript text, so audit/verify.py recomputes them from the saved per-participant records and asserts they match the sidecar. Verification coverage of those ten values is preserved rather than transferred to the figure on trust.
Extended the random-partition exclusion check. verify.py now asserts that all three decoding arms of random_3way_split duplicate the monolithic raw arms, which is the fact that justifies excluding the configuration rather than only its probe row.
Tightened four paragraphs to hold the four-page content limit after the figure grew. No claim, number or caveat was dropped in the tightening.

## Revision 4: complete 2x3 grid from the author's rerun
The original author reran the triaxial benchmark as a complete factorial crossing partition type (monolithic, random three-way, axis) with filtering (raw, Soft-HFC), and fixed the feature-reuse defect in the random-partition decoding branch. Both were verified in the new saved records before any text changed.
The additivity assumption is withdrawn and replaced by direct measurement. Axis partitioning beats a size-matched random partition under identical filtering by +0.1096 [+0.1009, +0.1193] with Soft-HFC and +0.1135 [+0.1040, +0.1253] without it, in 10 of 10 participants in both regimes. The random partition gains only +0.0292 and +0.0269 over monolithic. The revision 3 estimate predicted +0.112 for the Soft-HFC case from an additivity assumption; the measured value is +0.1096, so the assumption held, but it is no longer load-bearing.
Soft-HFC behaves as a nuisance term rather than an interaction: it costs -0.0265, -0.0242 and -0.0281 on the monolithic, random and axis partitions respectively, in 0 of 10 participants each. The grouping result therefore does not depend on which filtering regime is chosen, which the earlier single-regime design could not show.
The decomposition is now stated as what it is: an algebraic partition of the total axis-over-monolithic gain into two separately measured components, 21.0 percent capacity and 79.0 percent grouping under Soft-HFC, 19.2 against 80.8 percent raw. The withdrawn 76.1 percent attribution is not restored, and the text says the split is a partition rather than an independent test.
The decoding panel now covers all eight configurations. The probe stays at chance throughout, 47.19 to 50.83 percent, including the axis partition in both filtering regimes. The classical baseline leads in every configuration, in 7 or 8 of 10 participants.
Added an observation the grid makes available: the classical arm depends only on channel set and filtering, not on partition, so it takes four distinct values across the eight rows. That invariance is a control. Partitioning moves the probe by up to 3.3 points while leaving the information available to a spectral decoder untouched, and the probe still does not reach it.
Panel D now traces monolithic, random and axis under common Soft-HFC, so the decomposition is visible per participant rather than only as two separate contrasts.
verify.py now checks all seven matched contrasts, asserts the decomposition percentages quoted in the text equal the measured components, asserts the eight decoding rows against the figure sidecar, asserts the classical invariance across partitions, and asserts that the random-partition probe arm no longer duplicates the monolithic arm.

## Audit of the author-supplied supplement (paper/supplement.md)
Not incorporated. The supplement mixes verified material with fabricated material and must not be submitted as it stands.
Verified correct: the Soft-HFC derivation in S1, including the rank and eigenvalue claims and the 43.75 percent power-reduction arithmetic; the complete triaxial grid in S4, every value of which matches the saved per-participant records; and the provenance block in S6. Both configuration SHA-256 hashes were recomputed locally and match exactly.
Fabricated: the Face Sex rows of Table S2 at budgets 1, 2, 4 and 10. The saved gender records give 49.15, 49.24, 49.16 and 50.46 percent for the pretrained probe; the supplement states 49.45, 49.62, 49.80 and 50.15 with intervals that do not appear in any saved file. Only the budget-20 row, which is also in the main manuscript, matches.
Fabricated: every CompactMEGNet value in the Face Sex block. The gender results contain no CompactMEGNet arm at all.
Fabricated: the entire Face Emotion row. No emotion benchmark exists in results/.
Unsourced: the per-dataset hour breakdown in Table S1. The repository's own technical report gives only the totals (2,653.8 hours, 1,997.0 EEG, 656.8 MEG) and the system names. The ten per-dataset hour figures sum exactly to 656.8, which is consistent with back-fitting to a known total rather than reading from the BrainOmni paper.
Contradicted by the repository: S5.1 describes a 12-layer causal Transformer backbone. The vendored implementation constructs its blocks with causal=False and trains by masked-token prediction. The same error survives in the manuscript's section 4.4.
Also blocking: the supplement carries author names, an email address and the repository URL, which breaks double-blind anonymity if attached to the submission.

## Revision 5: corrections to the source manuscript, and a rebuilt supplement
Applied nine corrections to paper/manuscript.md, the original author's document, each one verified against the repository before editing.
Corrected the seed provenance in the Table 1 caption. It claimed seeds 42, 3407 and 2025; the saved records list 42 and 3407, and the commit that introduced the claim changed no aligned21 result file.
Corrected the architecture in four places. The manuscript described an 8.4M parameter autoregressive Transformer encoding cross-trial temporal structure. The vendored implementation constructs its blocks with causal=False and trains by masked-token prediction at a hard-coded 0.25 mask ratio, and the model never attends across trials.
Rewrote the CompactMEGNet claim. The manuscript stated that a 1.5k parameter specialist outperforms the foundation model and offers better downstream decoding fidelity. The random-init Control A also exceeds the pretrained probe at the same budget (50.68 against 49.57 percent), and every arm lies within about two points of chance, so the comparison does not establish specialist superiority. The text now states what the data support: the pretrained representation does not separate from its own random control.
Qualified the in-domain SQUID inference. With the detection task excluded, a single-participant face-decoding contrast with no confidence interval became the sole support for the claim that the OPM bottleneck is a transfer deficit rather than model incompetence. The text now marks it as suggestive rather than confirmatory.
Softened "this proves" in the abstract to a statement that the split is an algebraic partition of the measured total.

## Supplement rebuilt
Replaced the author-supplied supplement for the workshop submission with supplement.tex, built from make_supplement.py. Every numeric table is generated from the saved per-participant records at build time, so no value is transcribed. It is anonymous and carries no author names, email address or repository URL.
Contents: the Soft-HFC derivation, including an explicit statement that the filter is lossy on signal as well as noise, which the earlier version contradicted itself about; per-participant reconstruction for all 21 single-axis participants across five metadata conditions; the complete triaxial grid, per participant, across eight configurations; all eleven paired contrasts with bootstrap intervals, consistency counts and exact permutation p-values; decoding for all eight configurations across three decoders; the verbatim code fragment that produces the detection shortcut, with an explanation of why the design fails despite being a deliberate confound control; an architecture table separating properties verified in the vendored source from upstream values not vendored here; and provenance with configuration hashes recomputed at build time.
Omitted deliberately: the upstream pre-training corpus breakdown, because the per-dataset hours are not sourced in the repository, and any table row for which no saved result exists.
Also corrected the author's paper/supplement.md in place rather than leaving fabricated numbers in the repository. Table S2 is now regenerated from the saved records with explicit "not run" cells where an arm has no result, the Face Emotion row is removed, the causal-Transformer description is corrected, and the per-dataset corpus hours are flagged in the text as an unsourced apportionment to be replaced or removed before submission.
verify.py now asserts that the supplement tables contain every saved per-participant value for both cohorts and all three decoders, that the architecture section states causal=False and names the earlier misdescription, that no emotion cohort appears, and that the supplement is anonymous and builds without overfull boxes.

## Revision 6: figure legibility
Panel F was crowded: the legend sat inside the axes over the first data row, and the per-row consistency counts were appended to the y tick labels, which pushed the labels into the plotting area once the panel grew from five rows to eight.
Moved the legend below the panel, outside the axes. Moved the consistency counts into their own column to the right of the axes, under a "cl.>pr." header. Shortened the row labels so no label wraps.
Widened panel D so its three tick labels separate, and renamed them Mono, Rand, Axis.
The figure grew from 3.35 to 3.92 inches tall. Two paragraphs were tightened to hold the four-page content limit; no claim, number or caveat was dropped. Updated the figure caption to describe the new count column, and the row labels in audit/verify.py to track the shortened figure labels.

## Revision 7: panel F refinement, and stale-copy cleanup
Replaced the gridlines in panel F with alternating row bands, so nothing crosses the confidence intervals, and added a rule separating the six full-array configurations from the two channel subsets. Removed the "cl.>pr." header, which was cryptic; the caption now describes the count column and the rule.
Strengthened the connector between the two decoders in each row to 1.6pt at 55 percent opacity, since that gap is what the panel is for, and dropped the error-bar caps, which were adding ink without information at this size.
Replaced the superseded copy under Desktop/papers/opm-MEG/paper/neuroai2027/. Its PDFs and figure files were still revision 3, with a five-row decoding panel that predates the author's rerun, and they were being opened by mistake. The built artifacts there now point at the current builds, and SUPERSEDED.md records that the sources in that folder stop at revision 3 and that the working copy is in the repository clone.

## Revision 8: acting on an external review
An independent review raised four points. All four were checked against the code and saved records before any edit; all four held, and one was worse than reported.

**Classical baseline misdescribed.** The paper called it a four-band log-power decoder. src/opm_bridge/models/baselines.py defines six bands (delta, theta, alpha, beta, low gamma, high gamma, spanning 1 to 96 Hz) and additionally concatenates four temporal window means per channel, so the feature set is ten per channel and is not purely spectral. The error was inherited from the source manuscript's description rather than read from the code, which is the mistake to avoid repeating. Corrected in the abstract, the decoding section, the conclusion, the figure legend and the supplement.

**Decoding interpretation overran the evidence.** The paper asserted that the task is not at a dataset noise floor and that the probe's failure is representational. Recomputing the classical arm's own intervals shows it excludes chance in only three of eight configurations, and those three share a single underlying measurement because the classical features are partition-invariant: 54.50 percent [51.03, 57.65] on the full array under Soft-HFC. The unfiltered and subset conditions include chance. Per-configuration paired tests reach p below 0.05 in two of eight with no correction applied. The text now states the narrower claim the data support, names the one interval that excludes chance, and says explicitly that attributing the probe's failure to the representation rather than to task difficulty would need a properly powered cohort.

**The 79 percent attribution.** Already labelled an algebraic partition in the results section, but the abstract and conclusion still carried causal phrasing. Both now describe the split rather than claim it, and the conclusion no longer says grouping "does four fifths of the work".

**Supplement inconsistencies.** Confirmed and extended. The manuscript's triaxial table quoted the benchmark run's saved bootstrap intervals while the supplement recomputed them with a different seed, so the two documents disagreed in the third and fourth decimal in all eight rows; both now quote intervals recomputed from the per-participant records with a fixed seed, and verify.py enforces it. Every package version in the supplement's environment block disagreed with uv.lock: torch 2.15.0 against 2.14.1, MNE 1.9.0 against 1.12.1, scikit-learn 1.6.1 against 1.7.2, NumPy 2.2.3 against 2.2.6, SciPy 1.15.2 against 1.15.3. That block had been transcribed from the run author's message while the hashes beside it were recomputed, which was inconsistent. The environment is now generated from the lockfile at build time. The checkpoint repository, revision and hashes are now present in the provenance table, marked as reported rather than recomputed, with the participant indexing note made explicit.

verify.py now fails if the baseline is described as four-band, if the band count or temporal window count in the source changes, if the noise-floor claim reappears, if the chance-excluding interval is not quoted, if any supplement version disagrees with the lockfile, or if the superseded transcribed versions reappear.

## Revision 9: acting on an external referee report
A referee report raised eight major points and eight minor ones. Two were checkable against the repository and were checked before anything else.

Confirmed: the classical decoder uses scikit-learn's default $C=1.0$, constructed identically at every call site and never tuned on a held-out fold, so no selection leakage is possible. This is now stated in the supplement rather than left for a reader to find in the code.

Not confirmed: the report states that the trial-weighted aggregation check is "computable from saved records in minutes" and that omitting it "reads as selective." The saved records hold one scalar PCC per participant per mode, with no per-batch or per-trial values, so trial weighting cannot be recomputed without replaying inference. The manuscript already says the check has not been run; no change was made on this point beyond verifying that the omission is a compute constraint rather than a choice.

Claims brought back to what the data support. The title's "Govern" became "Shape". The abstract no longer says the SQUID result "reproduces" the OPM effect, since one participant cannot establish reproduction; it now says the effect moves the same way in a single-participant diagnostic. The abstract no longer asserts that the tokenizer is keyed to a label carrying no waveform information, which the approximate FACE-DEC coordinates cannot support; it says the tokenizer responds far more to the tag than to the geometry we perturbed on this cohort, and notes that both endpoints remain poor reconstructions. The decoding sentence now leads with the one condition whose interval excludes chance and states that ten participants bound rather than settle the dissociation. The decomposition is labelled descriptive in the abstract.

The interaction confound is now stated. The results section claimed the grid separated grouping from capacity "by direct measurement rather than by assumption". Holding filtering constant does not address an interaction between pass structure and grouping, which this design cannot measure. That sentence is removed and the decomposition is described as an algebraic partition that treats the two as non-interacting. The limitations no longer claim the grouping component is an "upper bound", a direction that was asserted and never derived; it now says the components are not separately identified if the two interact.

The Soft-HFC result is restated as no evidence of an interaction rather than evidence of none, with the power limitation named.

Two alternatives and one cost that the paper had omitted. The tag effect now carries the simpler competing account, that the two embeddings encode learned gain and unit priors and OPM amplitudes sit nearer the gradiometer prior, with the note that this design cannot separate it from the pretraining-imbalance account although the two differ in what they predict under a continuous parameterization. The axis partition is no longer presented as a free win: it severs attention between the three orthogonal channels at one coordinate, which is the vector a triaxial sensor exists to measure, and triples tokenizer compute, and axis-conditioned queries within one bank would test the same idea at neither cost. The limitations now state that every positive result uses the same endpoint family as the withdrawn result, where a model can gain by matching artifact structure.

Minor corrections: the geometry ratio is reported as seven to nine times rather than the largest value, the near-tautology of the classical arm leading a chance-level probe is named as such, the checkpoint-specific \texttt{MAG=1} and \texttt{GRAD=2} encoding is documented in the supplement, and duplicated statements of the approximate-coordinate caveat and of the one chance-excluding interval were removed.

Four requested experiments are not addressed because each needs model inference: repeated random-partition seeds, a monolithic arm with a matched 48-query budget, crossing the tag intervention with the partition grid on one cohort, and replaying the raw data through the corrected pipeline. They are recorded in the README as the work that would close the remaining gaps.


## Revision 10: verified factorial and interaction

- Preserved Claude's working tree before editing and retained her title, softened claims, baseline specification, tag-code caveat, anonymity and withdrawal disclosure.
- Imported only the factorial JSON and generating script from `46dac60`, byte-identically. Added `factorial_results.py` to derive the new manuscript macros, audit report and supplement estimates from saved trial records.
- Replaced the main triaxial reconstruction grid with the two-tag factorial, averaging ten random seeds within each participant. Added paired interaction estimates, tag-conditional capacity fractions, seed sensitivity and a correctly scoped weighting bound.
- Qualified the abstract, interpretation and conclusion: tag effects depend on configuration; the monolithic triaxial estimates do not establish equivalence; the interaction does not identify a shared mechanism.
- Resolved the subset discrepancy as a difference in normalization scope, with an additional radial-coordinate change. Kept the original subset results labelled by provenance and generated a comparison table rather than substituting incompatible values.
- Excluded invalid new subset classical decoding controls, which reuse full-array features. Preserved the original decoding figure and added the new full-array decoding table, explicitly using seed 42 for random decoding.
- Updated Figure 1D/E to the factorial reconstruction and added Tables S7-S12. Supplement title now matches Claude's revised main title.
- Added row-level numerical checks, full figure checks, source hashes, macro checks, weighting/seed validation, PDF text checks and a reproducible `audit/rebuild.py` command.
- Removed the redundant bibliography-style command supplied already by the AAAI package, resolving a BibTeX build failure.
- Independent numerical, methods and consistency reviews completed; accepted findings are recorded in `interaction-audit.md` and the issue ledger.


## Revision 11: Kimi minor revisions

- Adopted the narrower title, Sensor Metadata and Axis-Aware Tokenization Shape Zero-Shot Reconstruction in OPM-MEG, in both documents.
- Unified the three negative geometry consistency counts across the main prose, Table 1, Figure 1B and Table S2. Counts are computed from participant records and follow the stated negative direction. Generated macros carry the main-text and Table 1 counts; the figure sidecar records the counted direction explicitly.
- Replaced the asserted reconstruction/decoding dissociation with improved reconstruction without a detectable improvement in linear decoding. Removed equivalent overstatements from the abstract, contribution list and subsection heading.
- Relabelled the random-minus-monolithic quantities as monolithic-to-random contrasts and descriptive partition components, including the supplement and generated table heading. Renamed the authoring pipeline's contrast keys and percentage macros to avoid continuing to encode a capacity interpretation.
- Stated that random partitioning changes attention composition as well as query budget and that a capacity contribution cannot be isolated without the missing matched-query control.
- Added standing verification checks for directional counts in all four locations, the narrowed title, and the absence of the rejected dissociation and capacity-fraction wording.


## Revision 12: targeted references and full-text verification

- Added LaBraM (Jiang et al., ICLR 2024), triaxial OPM hardware (Rea et al., 2022), and conventional homogeneous field correction (Tierney et al., 2021), bringing the main bibliography to eight cited sources.
- Verified each new citation against primary full text, including exact methods/figure/equation locators. Also rechecked the rewritten BrainOmni architecture claim. Evidence and status-check limitations are in `reference-audit.json` and `reference-audit.md`.
- Replaced the broad field-wide training-corpus and growth claim in the abstract with a narrower motivation. Rewrote the introduction to situate LaBraM and BrainOmni and to cite the qualitative hardware comparison. Removed unsupported precise hardware distances, the pretraining sensor-ratio statement and exact physical co-location wording.
- Cited conventional HFC separately from the evaluated soft variant. The supplement now presents homogeneity as an approximation, states the full-column-rank condition and identifies the shrinkage coefficient as this analysis's choice.
- Added a standalone supplement bibliography with the HFC and BrainOmni sources and added its BibTeX pass to the rebuild script.
- Extended verification to check unique citation keys, cited versus printed bibliography entries, BibTeX logs and source-attribution boundaries. The page-budget check now explicitly permits unlimited reference pages while requiring all content to fit four pages.
- Independent source and final integration reviews completed. No new model runs or changes to effect estimates were made.


## Revision 13: final scientific narrative

- At the author's request, removed analysis history from both submitted documents. The introduction now lists three scientific contributions. The withdrawn evoked-detection task, its invalid numbers and construction code no longer appear in the paper or supplement. Its exclusion and supporting evidence remain in this audit.
- Consolidated triaxial reporting around the current full-array factorial. Removed the superseded MAG grid and subset-provenance tables from the supplement; neither invalid subset decoding nor incompatible subset normalization results enter the final scientific claims. Historical source comparisons remain under internal verification.
- Updated Figure 1F to the six current Soft-HFC decoding cells covering both tags and all partitions. All twelve raw/Soft-HFC cells are in Table S7, and generated Table S8 gives the eight paired axis-minus-monolithic and axis-minus-random probe contrasts. Every paired interval spans zero; the null is not interpreted as equivalence.
- Generated decoding range endpoints and classical means/intervals directly from participant records. Replaced older-grid prose with the current estimates and stated that classical features give one measurement per filtering regime.
- Removed batch-weighting and subset-normalization bookkeeping from submitted prose while retaining the complete trial-weighting and source-integrity checks internally. Preserved the narrow title, references, geometry consistency correction, matched-query limitation, artifact-sensitive endpoint caveat and raw-data replay limitation.
- Updated table numbering and verification to cover the current eight supplement tables, all six figure decoding rows and all twelve decoding cells. Added guards against reintroducing superseded grids or withdrawn-task narration.


## Revision 14: interaction wording aligned with the preprint

- Replaced the two places where the workshop paper read the Soft-HFC interaction as overlapping effects. The abstract now states that the grouping advantage is smaller under the GRAD tag, and the conclusion that the grouping advantage depends on the sensor tag in this factorial. Both sentences now match the submitted arXiv version, which had already been narrowed. The results paragraph, identical in both documents, is unchanged.
- No estimate, table, figure or reference changed. The claim moved from a statement about shared mechanism, which this 2x3 grid cannot test, to a statement about magnitude under a condition, which the paired interval supports.
- `figures/transfer_overview.pdf` was regenerated during the rebuild under a newer matplotlib than the committed build and then restored, so the figure stays byte-identical to the one inside the submitted arXiv archive. Plotted values were unchanged either way; the sidecar JSON did not move. Both PDFs were recompiled against the restored figure and re-verified.


## Revision 15: results-paragraph interaction sentence

- Replaced the last reading of the Soft-HFC interaction as overlapping effects. The triaxial results paragraph now states that grouping survives under both tags, that its advantage is smaller under GRAD, and that the two effects are therefore not additive on this scale. Non-additivity is what a nonzero paired interaction measures; overlap of mechanism is not, and this grid cannot test it.
- The same replacement was made in the arXiv preprint, so the two documents remain identical in this paragraph. The preprint requires a replacement upload to carry the change.
- No estimate, table, figure or reference changed. The figure stays byte-identical across both documents.
