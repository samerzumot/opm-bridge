# Verification record

- Official AAAI-27 style downloaded with curl from https://aaai.org/authorkit27/; exit 0. Files copied unmodified.
- `pdflatex -interaction=nonstopmode -halt-on-error -output-directory=build short_paper.tex`, `BIBINPUTS=.: bibtex build/short_paper`, then two pdflatex passes: exit 0. Logs: build/compile-1.txt, build/bibtex.txt, build/compile-2.txt, build/compile-3.txt.
- `python3 audit/verify.py`: exit 0. Output: audit/verification.txt. Checks all 18 triaxial table values against saved summaries and selected format/anonymity conditions. It does not verify raw data or all manuscript claims.
- `python3 /Users/marziehzare/.codex/skills/orchestrate-scientific-manuscripts/scripts/manuscript_lint.py short_paper.tex --bib references.bib --json`: exit 0, zero leads. Output: audit/lint.json. Static lint is not scientific validation.
- `pdfinfo short_paper.pdf`: exit 0, 3 pages including references, US Letter 612 x 792 points.
- `pdftoppm -scale-to 1100 -png build/short_paper.pdf build/final-page`: exit 0. All three final page images inspected visually. No clipped tables, overlapping text, or unresolved citations observed.
- Independent clean-room methods review confirmed bounded claims; its two corrections to channel matching and geometry scrambling were applied.
- Not checked: raw replay, all citation correction/retraction records, participant consent/license applicability to this secondary analysis, exact OpenReview submission fields, completed anonymized code supplement.

## Updated PDF with figures
- `MPLCONFIGDIR=/tmp/opm-mpl python3 paper/neuroai2027/generate_figures.py` from repository root: exit 0. Figures read saved JSON means and confidence limits directly.
- Two pdflatex passes: exit 0. Logs: build/figures-compile-1.txt and build/figures-compile-2.txt. No overfull boxes or unresolved references.
- `python3 audit/verify.py`: exit 0; saved numeric checks still pass.
- `pdfinfo build/short_paper.pdf`: 4 pages including references, US Letter.
- `pdftoppm -scale-to 1300 -png build/short_paper.pdf build/with-figures`: exit 0. All four pages inspected visually; figure labels, confidence intervals, captions and tables are legible without clipping.
- Final build copied to short_paper.pdf. Earlier three-page check refers to the superseded draft without figures.

## Revision 2
- `MPLCONFIGDIR=/tmp/opm-mpl python3 paper/neuroai2027/generate_figures.py`: exit 0. The script prints the four headline contrasts it plots.
- Three pdflatex passes plus bibtex: exit 0. `build/short_paper.log` contains no overfull box and no undefined reference.
- `pdfinfo build/short_paper.pdf`: 4 pages, US Letter, references on the final page and excluded from the workshop's four-page content limit.
- `python3 audit/verify.py`: exit 0. Recomputes every paired contrast from per-participant records, checks consistency counts and exact permutation p-values, checks the decoding table direction, the SQUID control, anonymity, em dashes and the build log.
- All four rendered pages inspected visually. Figure panels, tick labels, annotations and captions are legible without clipping.
- Not checked, unchanged from revision 1: raw-data replay, citation status audit, consent and licence applicability, OpenReview form fields.

## Revision 3
- `MPLCONFIGDIR=/tmp/opm-mpl python3 generate_figures.py`: exit 0. Prints the four headline reconstruction contrasts and the five decoding rows it plots, and writes figures/transfer_overview_values.json.
- Three pdflatex passes plus bibtex: exit 0. `build/short_paper.log` contains no overfull box and no undefined reference.
- `pdfinfo short_paper.pdf`: 4 pages, 612 x 792 points, references on the final page and excluded from the workshop's four-page content limit.
- `python3 audit/verify.py`: exit 0. In addition to the revision 2 checks, it now recomputes the ten decoding values from per-participant records and asserts they match the figure sidecar, asserts the classical lead is 8 of 10 participants in each of the five configurations, and asserts that all three decoding arms of random_3way_split duplicate the monolithic raw arms.
- Rendered figure inspected visually at full resolution after each of four layout iterations. Final version has no overlapping annotation, no clipped label and a legend distinguishing the two decoding arms.
- Verified directly in the saved results that random_3way_split duplicates triaxial_xyz_raw for pretrained_linear_probe, classical_bandpower and control_a_linear_probe. This is what corrected the six-configuration count to five.
- Not checked, unchanged from revision 1: raw-data replay, citation status audit, consent and licence applicability, OpenReview form fields.

## Revision 4
- Cloned github.com/samerzumot/opm-bridge at commit d70f4dd and confirmed that the only results file changed by that commit is summary_phase4_nottingham_full.json.
- Confirmed the new modes random_3way_split_softhfc and axis_split_raw complete the 2x3 grid, and recomputed every contrast from the per-participant records.
- Confirmed the random-partition decoding fix: the pretrained probe and random-init control arms now differ from the monolithic arms, while the classical arm is identical by construction.
- Confirmed the detection shortcut remains in scripts/run_transfer_benchmark.py at lines 351-354. The task is excluded from the manuscript rather than repaired, so the exclusion subsection stands as written.
- `MPLCONFIGDIR=/tmp/opm-mpl python3 generate_figures.py`: exit 0, prints nine reconstruction contrasts and eight decoding rows.
- Three pdflatex passes plus bibtex: exit 0, no overfull box, no undefined reference.
- `pdfinfo short_paper.pdf`: 5 pages, of which page 5 is references only. Content is within the four-page limit; references are excluded from it.
- `python3 audit/verify.py`: exit 0.
- Supplement audit: recomputed both config SHA-256 hashes (match), compared every Table S2 and Table S3 value against the saved records, and checked the causal flag in the vendored model. Findings recorded in change-log.md.
- Not checked: the per-dataset hour breakdown in supplement Table S1 against the BrainOmni publication, raw-data replay, citation status audit, consent and licence applicability, OpenReview form fields.

## Revision 5
- `MPLCONFIGDIR=/tmp/opm-mpl python3 make_supplement.py`: exit 0, writes supplement_tables.tex from results/.
- Three pdflatex passes for each document: exit 0. short_paper.log and supplement.log contain no overfull box and no undefined reference.
- `pdfinfo short_paper.pdf`: 5 pages, page 5 references only, content within the four-page limit. `pdfinfo supplement.pdf`: 6 pages, no limit applies.
- `python3 audit/verify.py`: exit 0, now covering the supplement as well as the main paper.
- Each manuscript.md correction was checked against its source before editing: seeds against summary_aligned21_age.json, the architecture against the vendored model.py and braintokenizer/model.py, the CompactMEGNet comparison against summary_aligned21_age.json at B=20, and the SQUID inference against summary_ds000117_squid_baseline.json.
- Supplement page proofs rendered and inspected. Title block shows "Anonymous submission"; no author identifiers present.
- Not checked: the upstream pre-training corpus figures against the BrainOmni publication (omitted from the new supplement for that reason), raw-data replay, citation status audit, consent and licence applicability, OpenReview form fields.

## Revision 6
- Figure regenerated and inspected at full resolution after each of three layout passes. No overlapping text, no clipped label, no legend over data in any panel.
- Three pdflatex passes: exit 0, no overfull box, no undefined reference.
- `pdfinfo short_paper.pdf`: 5 pages, page 5 references only, content within the four-page limit.
- `python3 audit/verify.py`: exit 0.

## Revision 7
- Figure regenerated and inspected at full resolution. Panel F has no gridline crossing an interval, no label or legend inside the data area, and a visible subset rule.
- Three pdflatex passes: exit 0, no overfull box, no undefined reference. 5 pages, page 5 references only.
- `python3 audit/verify.py`: exit 0.
- Confirmed by text extraction that the superseded copy's PDF lacked the "Random, HFC" row before replacement and carries it after, so the old paths now serve the current build.

## Revision 8
- Read src/opm_bridge/models/baselines.py directly: BANDS has six entries and extract_features concatenates np.array_split(X, 4) temporal means. The four-band description was wrong.
- Recomputed the classical arm's bootstrap interval for all eight configurations: three exclude chance, all three being the Soft-HFC full-array rows that share one measurement.
- Recomputed paired classical-minus-probe tests: p below 0.05 in two of eight, uncorrected.
- Compared the manuscript's triaxial intervals against recomputation: all eight rows differed from the saved intervals in the third or fourth decimal. Both documents now use the recomputed values.
- Compared every supplement environment version against uv.lock: all five disagreed. The table is now generated from the lockfile.
- Three pdflatex passes for each document: exit 0, no overfull box, no undefined reference.
- `python3 audit/verify.py`: exit 0, with the new standing checks.
- Not rerun: raw data and model inference, consistent with every prior revision.

## Revision 9
- Read src/opm_bridge/models/baselines.py and every call site: ClassicalBandPowerDecoder is always constructed with the default c_reg=1.0 and no fold-wise tuning.
- Inspected the per-subject records: each mode stores a single scalar zero_shot_tokenizer_pcc with no per-batch or per-trial array, so trial-weighted aggregation is not computable from saved outputs.
- Recomputed the classical arm's bootstrap intervals: exactly three of eight configurations exclude chance, and all three are the Soft-HFC full-array rows that share one underlying measurement.
- Three pdflatex passes for each document: exit 0, no overfull box, no undefined reference. Paper 5 pages with page 5 references only; supplement 4 pages.
- python3 audit/verify.py: exit 0.
- Not addressed, each requiring model inference: random-seed distribution for the random partition, a matched 48-query monolithic arm, the tag by partition cross, and a raw-data replay.


## Revision 10

- `python3 paper/neuroai2027/audit/rebuild.py`: exit 0. Figure and supplement regeneration, three main LaTeX passes, BibTeX, three supplement LaTeX passes and final verification all completed successfully. Exact commands, outputs and exit statuses are saved in `rebuild-log.txt`.
- `python3 audit/verify.py`: exit 0. Independently recomputed all 560 records from their 84 trial scores, the seed-averaged factorial, paired contrasts and interactions, seed robustness, weighting bound, all labelled new table rows, and all figure numerical panels. Source hashes, original-grid replication and subset provenance checks pass.
- Paper: 5 pages, page 5 references only, hence 4 content pages. Supplement: 8 pages. Neither final LaTeX log contains an overfull box or undefined reference.
- Anonymous PDF text and source checks pass. No em dashes in either document or their generated inputs. Both titles use Shape.
- `manuscript_lint.py short_paper.tex supplement.tex supplement_tables.tex --bib references.bib --json`: exit 0, zero findings; result saved as `interaction-lint.json`. Generated table inputs were included in the scan so references could be resolved.
- Rendered all 13 PDF pages with `pdftoppm -r 100 -png` and visually inspected every page, plus the full-resolution generated figure. No clipped tables, overlapping labels or unreadable plot annotations found. Proofs are in `/tmp/opm-proof/`.
- After replacing repeated literal factorial counts and p-values with generated macros, rebuilt the main PDF and reran verification. Extracted PDF text was asserted identical to the inspected proof.
- The result JSON and generating script match commit `46dac60` byte-for-byte. Manuscript numbers are newly analysed from saved outputs; neither raw data nor model inference was replayed. Group assignments are unavailable for metadata-level verification. The raw monolithic decoding difference between runs remains unexplained and is not described as exact replication.
- Current citation-status, author disclosure, consent/licence and portal checks were outside this scoped revision and remain as previously recorded.


## Revision 11

- `python3 paper/neuroai2027/audit/rebuild.py`: exit 0. Regenerated figure, numeric macros and supplement tables, rebuilt both PDFs, and ran verification. Two further LaTeX passes per document and final verification after independent-review wording corrections also exited 0; exact outputs are in `rebuild-log.txt`.
- `audit/verify.py`: exit 0. The three geometry counts are independently recomputed as negative differences and checked in main prose, Table 1, Figure 1B sidecar and Table S2. The verifier also enforces the narrowed title and guards against the rejected dissociation and capacity-fraction claims.
- Independent focused reviewer confirmed the four requested corrections and raw geometry counts. Accepted follow-up: replace two residual absence-of-effect phrases with chance-overlapping intervals and no detectable improvement.
- `manuscript_lint.py` over both documents and generated tables with `references.bib`: exit 0, zero findings (`kimi-lint.json`). `git diff --check`: exit 0.
- Paper remains 5 pages, with page 5 references only; supplement remains 8 pages. Final logs have no overfull boxes or undefined references. Anonymous source/PDF and no-em-dash checks pass.
- Rendered all 13 pages at 100 dpi in `/tmp/opm-kimi-proof/`. Inspected all changed pages and the full figure. Main page 5 and supplement pages 4, 6, 7 and 8 are pixel-identical to the previously inspected proofs. No new clipping, overlaps or unreadable labels found.
- Raw result files and reported effect estimates are unchanged. The changes correct the direction counted and narrow the interpretation; they do not represent new model runs or new statistical evidence.


## Revision 12

- Primary full-text source checks completed for the three added references and the rewritten BrainOmni architecture claim. Independent methods, hardware and final-integration reviewers found the final citation wording supported. Exact URLs, sections/equations, claim boundaries and correction-status limitations are in `reference-audit.json` and `reference-audit.md`.
- `python3 paper/neuroai2027/audit/rebuild.py`: exit 0. Figure/macro/table regeneration, both BibTeX runs, three LaTeX passes per document, and final verification succeeded. Exact command/output/exit records are in `rebuild-log.txt`.
- `audit/verify.py`: exit 0. Eight unique main-paper references; every cited key resolves and every printed bibliography entry matches a citation. Both BibTeX logs are clean. Full numerical, directional-count, figure, subset provenance, anonymity and source-attribution checks also pass.
- Paper: 5 total pages; content ends on page 4, where references begin, and page 5 contains references only. Supplement: 9 pages, including its new standalone reference page. No overfull boxes or undefined references in either final log.
- `manuscript_lint.py` over the two documents plus generated tables, with `references.bib`: exit 0, zero findings (`reference-lint.json`). `git diff --check`: exit 0.
- Rendered all 14 pages at 100 dpi in `/tmp/opm-reference-proof/`. Inspected every changed page, including both bibliographies. Supplement pages 2, 4, 5, 6, 7 and 8 are pixel-identical to the previously inspected revision. No clipped text, overlapping labels or malformed citations found.
- Published PDF files are byte-identical to their verified build outputs. No new model inference or changed result estimates. Existing dataset citation audits were retained; no comprehensive new correction-status audit of those references was claimed.


## Revision 13

- `python3 paper/neuroai2027/audit/rebuild.py`: exit 0 after the final control-label clarification. Regenerated the figure, macros, numerical report and all eight supplement tables; ran both BibTeX passes, three LaTeX passes per document and full verification. Exact outputs are in `rebuild-log.txt`.
- `audit/verify.py`: exit 0. All 560 saved trial records, factorial estimates, paired interactions, seed sensitivity, weighting integrity, current decoding summaries, all eight paired decoding contrasts, every figure panel, table rows, macros, citations, source/PDF anonymity and layout logs pass. Historical source comparisons remain internal; submitted-text guards reject superseded grids and withdrawn-task narration.
- Independent focused reviewer found no material issues and independently recomputed the eight paired decoding intervals. All span zero. Accepted the minor request to expand Ctrl A as the random-initialization control in Table S7.
- Paper: 4 total pages, with references beginning on page 4. Supplement: 6 pages, including its bibliography. Neither final log has overfull boxes or undefined references.
- `manuscript_lint.py` across both documents and generated tables: exit 0, zero findings (`final-narrative-lint.json`). `git diff --check`: exit 0 after normalizing audit CSV line endings.
- Rendered and visually inspected all ten pages and the full-resolution figure. Re-rendered both final PDFs after the minor caption clarification: all main pages and five supplement pages are pixel-identical to the inspected proofs; visually inspected the changed supplement page 5. No clipped text, overlapping labels or malformed references. Proofs: `/tmp/opm-final-narrative-proof/`.
- Both delivered PDFs are byte-identical to their verified build outputs. No new inference or raw-data replay was performed. Earlier versions and exclusion evidence remain preserved in the internal audit and the pre-edit snapshot `/tmp/opm-before-final-scientific-narrative.tar.gz`.

- Push preflight: `git diff --cached --check -- paper/neuroai2027` passes. The imported factorial script has whitespace warnings already present in Samer's commit `46dac60`; it remains byte-identical to that source for provenance. The imported result JSON also matches that commit exactly.
