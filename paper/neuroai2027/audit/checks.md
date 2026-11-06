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
