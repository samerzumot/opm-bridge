# Interaction integration audit

Scope: verify the saved factorial results, resolve subset provenance, integrate the interaction into the anonymous short paper, and rebuild. Evidence is restricted to the local saved JSON, generating scripts and preprocessing code. No raw-data or model-inference replay, new citation review, or external communication was performed.

## Preserved starting state

Claude's working tree was preserved before edits at `/tmp/opm-before-interaction/neuroai2027.tar.gz`, with a binary Git diff at `/tmp/opm-before-interaction/claude-working-tree.patch`. Her title, softened interpretation, classical baseline specification, checkpoint-specific sensor codes, withdrawal disclosure, anonymity and noise-floor qualifications remain. Existing audit history is retained.

The result JSON and generating script were imported byte-for-byte from commit `46dac60`. The rest of `origin/main` was not merged into the manuscript branch.

## Verified analysis

The manuscript uses trial-weighted participant scores and averages the ten random seeds within each participant before inference. Bootstrap confidence intervals use 20,000 participant resamples with seed 0. Exact two-sided sign-flip tests enumerate 1,024 sign assignments for the ten participants. Tests are exploratory and unadjusted. See `factorial-results.json` for all estimates and source SHA-256.

The numerical reviewer independently recomputed the contrasts from trial arrays. Both grouping interactions are negative for every participant. The interaction demonstrates nonadditive effects of the interventions; it does not establish a shared biological or model mechanism. The lack of a large monolithic tag effect does not establish equivalence or isolate axis count as the reason for cohort differences.

All 560 records contain 84 finite trial scores. Trial averages exactly reproduce saved trial-weighted values. Reconstructed batch means agree within floating-point tolerance. Four common MAG monolithic/axis conditions replicate original participant batch scores within 5e-8. The weighting bound applies only to the twelve seed-averaged full-array cohort means, not individual participants, individual random seeds or subset conditions.

## Resolved subset discrepancy

The old script `scripts/run_nottingham_full_benchmark.py:219-236` selects Z or XY channels before sensor-type waveform normalization. The new script `scripts/run_nottingham_tag_partition_factorial.py:132-173` normalizes the full array before selecting channels. The normalizer subtracts the channel common average at each time point and then rescales the group (`src/opm_bridge/preprocessing/sensor_mapping.py:224`). These operations do not commute. Radial metadata also changes from subset-based to full-array centering and scaling; tangential order changes to X then Y. Participant IDs and channel counts match throughout.

Thus the old and new subset values describe different preprocessing conditions. Supplement Table S12 labels them separately; they are not treated as replications. Code inspection identifies changed inputs but does not isolate each operation's numerical contribution without inference replay.

A separate defect in the new script computes classical features on the full array at line 152 and reuses them for subset arms at lines 258-264. All saved new subset classical scores equal their full-array scores. Those subset decoding rows are excluded. The original subset decoding uses features computed after subset selection and remains the provenance of Figure 1F and Table S6.

## Reconstruction and decoding scopes

The original random control uses approximately equal thirds at seed 42. The new script matches each participant's axis group sizes exactly and runs ten reconstruction seeds. Actual group assignments and X/Y counts are not saved, so exact matching is supported by code inspection, not an independent metadata replay.

The new run saves random decoding only for seed 42; the other seeds have empty decoding dictionaries. Table S11 explicitly uses seed 42 for random decoding. All twelve new full-array pretrained-probe intervals include chance, and the classical cohort mean leads in all twelve. This does not establish equivalence. Original and new random decoding are not interchangeable because the groups differ. Raw monolithic decoding also differs by 0.125 percentage points at cohort level, localized to one participant; its cause is not identifiable from saved aggregate decoding scores. No exact decoding replication is claimed.

## Independent review and corrections

Three read-only reviewers covered numeric integrity, preprocessing provenance, and final consistency. Accepted review findings: remove the stale historical S4 caption, label original and new grids explicitly, change signed reduction wording, remove causal-separation wording from the contribution list, and verify complete labelled rows for Tables S7-S12. All were implemented.

A strict BibTeX run exposed a duplicate bibliography-style command already supplied by the AAAI style. The redundant manuscript command was removed, preserving the same bibliography style.

## Reproduction

From the repository root: `python3 paper/neuroai2027/audit/rebuild.py`.

This regenerates figure, macros, audit JSON and supplement tables, runs LaTeX and BibTeX, runs `audit/verify.py` last, and copies verified build PDFs to the stable submission paths. The full command/output/exit record is `rebuild-log.txt`. The checker independently rebuilds factorial statistics from trial arrays, checks labelled table rows and figure values, validates subset provenance assertions, checks anonymity and punctuation, and confirms the paper content ends by page four. Visual page-proof and lint results are recorded in `checks.md` after completion.


## Follow-up: Kimi minor revisions

Revision 11 narrows the title to zero-shot reconstruction, reports negative geometry consistency counts consistently, removes asserted decoding dissociation, and labels monolithic-to-random contrasts as descriptive partition components. The internal report keys now use `mono_to_random` instead of `capacity`. Underlying estimates are unchanged. See revision 11 in `checks.md` and `change-log.md` for the final verified package.
