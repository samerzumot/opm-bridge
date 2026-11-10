> Historical handoff. The interaction has since been integrated; see [interaction-audit.md](interaction-audit.md) and revisions 10 through 13 in [checks.md](checks.md) for current verified state.

# Handoff: AAAI-27 NeuroAI workshop paper

Written 2026-11-09. Deadline **2026-11-20**, 4-page non-archival short paper, double-blind,
AAAI-27 format. References and supplement are unlimited and excluded from the page count.

## Where things are

Working copy, and the only one:

    /Users/marziehzare/Desktop/papers/opm-bridge-latest/paper/neuroai2027/

It sits inside a clone of `github.com/samerzumot/opm-bridge`, so its scripts read the live
`results/` two directories up. An older copy under `Desktop/papers/opm-MEG` was deleted on
5 November after confirming it held nothing unique.

Branch `paper/neuroai2027-workshop-draft`, pushed, three commits ahead of `origin/main`.
`main` is untouched. Authors: Samer Zumot (original analysis, all model runs) and
Marzieh Zare. The paper is anonymized; restore authorship at camera-ready.

## The rule that matters most

**No number enters either document by being typed.** This is not a style preference. The
author-supplied supplement that was replaced contained fabricated table rows: Face Sex values
at budgets 1, 2, 4 and 10 that appear in no saved file, CompactMEGNet values for an arm that
was never run, and an entire Face Emotion row for a benchmark that does not exist. They were
plausible, internally consistent, and anchored on the one real row.

So:

- `generate_figures.py` builds the figure from `results/` and writes
  `figures/transfer_overview_values.json`, a sidecar of every plotted value.
- `make_supplement.py` generates every supplement table at build time, including the
  environment table, which is read from `uv.lock` rather than transcribed. An earlier
  transcribed environment block disagreed with the lockfile in all five entries.
- `audit/verify.py` recomputes every quoted contrast from the per-participant records,
  asserts each appears in the manuscript, checks the figure sidecar, the classical-baseline
  description, anonymity, em dashes and the build logs.

Build order, from the paper directory. `verify.py` reads the build logs, so it runs last:

```sh
MPLCONFIGDIR=/tmp/opm-mpl python3 generate_figures.py
MPLCONFIGDIR=/tmp/opm-mpl python3 make_supplement.py
pdflatex -interaction=nonstopmode -output-directory=build short_paper.tex
BIBINPUTS=.: BSTINPUTS=.: bibtex build/short_paper
pdflatex -interaction=nonstopmode -output-directory=build short_paper.tex
pdflatex -interaction=nonstopmode -output-directory=build supplement.tex
python3 audit/verify.py
```

Current state: paper 5 pages with page 5 references only, so 4 pages of content. Supplement
4 pages. No overfull boxes. `verify.py` exits 0. The paper is submittable as it stands.

## What Samer just delivered, and why it changes the paper

Commit `46dac60` on `origin/main`, 9 November, adds
`results/summary_phase4_nottingham_factorial_and_seeds.json`: a Tag (MAG/GRAD) x Filter
(Raw/Soft-HFC) x Partition (Mono / Random with 10 seeds / Axis) factorial, with `trial_pccs`,
`pcc_trial_weighted` and `pcc_batch_weighted` saved per participant.

**None of this is in the paper yet.** I computed it but ran out of budget before revising.
All values below are trial-weighted, paired, 10 participants, exact sign-flip p.

### 1. The grouping effect survives the gradiometer tag, but loses about a third

| contrast | MAG tag | GRAD tag |
| :-- | :-- | :-- |
| axis minus random, raw | +0.1120 [+0.1048, +0.1208] | +0.0688 [+0.0567, +0.0805] |
| axis minus random, Soft-HFC | +0.1071 [+0.1002, +0.1143] | +0.0745 [+0.0656, +0.0833] |

All four are 10/10 participants, p = 0.002. The grouping story does not collapse, which was
the risk. But roughly a third of the advantage disappears once the tag is already
"correct", so the two headline findings are **not independent**. The old design could not
see this.

### 2. The tag effect is null on the triaxial monolithic array

| contrast | effect | consistency | p |
| :-- | :-- | :-- | :-- |
| GRAD minus MAG, monolithic, raw | +0.0002 | 5/10 | 0.97 |
| GRAD minus MAG, monolithic, Soft-HFC | +0.0014 | 6/10 | 0.82 |
| GRAD minus MAG, random partition, Soft-HFC | +0.0193 | 9/10 | 0.012 |
| GRAD minus MAG, axis partition, raw | -0.0277 | 1/10 | 0.006 |

This is the finding that most needs writing in. On the single-axis FACE-DEC cohort the same
relabelling gives +0.1318 in 21/21 participants. On the triaxial cohort it does **nothing**
to the monolithic path, helps a random partition slightly, and *hurts* an axis partition.
The abstract currently generalizes the tag effect; it cannot, as written. The honest
statement is that the tag effect is configuration-dependent, large on a single-axis array
and absent on a triaxial one under monolithic tokenization.

This also weakens the "load-bearing free parameter at inference time" line, and it bears on
the competing mechanisms already named in the Results: a learned gain/unit prior would not
obviously vanish on a triaxial array, so this is evidence worth weighing explicitly.

### 3. Capacity grows under the gradiometer tag

random minus monolithic: +0.0292 to +0.0446 raw, +0.0328 to +0.0507 Soft-HFC. So the
decomposition percentages in the Results (21.0 / 79.0 and 19.2 / 80.8) are tag-conditional
and should be labelled as such.

### 4. Two caveats can now be retired

- **Seed sensitivity**: across 10 seeds, axis minus random under Soft-HFC/MAG ranges
  +0.0969 to +0.1178, sd 0.005. The single-seed result was not a fluke. The limitation
  sentence about "one random-partition seed" can go.
- **Trial vs batch weighting**: differences are +/-0.0006. The sentence "Trial-weighted
  aggregation is a reproducibility check we have not run" can be replaced with the result.

### 5. One discrepancy to resolve before citing the new file

The four full-array conditions replicate the existing summary exactly (monolithic raw .3726,
monolithic Soft-HFC .3461, axis raw .5130, axis Soft-HFC .4849, batch-weighted). The two
channel-subset conditions do **not**: radial .4944 old against .5100 new, tangential .4446
old against .4368 new. Subject ids match. Likely a different subset definition in
`scripts/run_nottingham_tag_partition_factorial.py`. Check before quoting either number;
the subset rows are not load-bearing, but the paper currently quotes the old values.

## The decision this forces

The paper's two findings interact, and one of them does not generalize across cohorts. Three
options, in my order of preference:

1. **Report the interaction.** Add the tag x partition result as a qualification: the tag
   effect is configuration-dependent, and grouping and tag partly overlap. This is the most
   honest version and it strengthens the paper, because the interaction is itself the
   finding. Costs roughly 120 words, which means cutting 120 elsewhere. The paper has been
   cut to the bone twice already, so expect to lose a limitation sentence or the
   radial/tangential paragraph.
2. **Qualify the tag claim only.** State that the tag result is measured on the single-axis
   cohort and does not transfer to triaxial monolithic tokenization, and leave the
   interaction to the journal version. Cheaper, still honest.
3. **Submit as is.** Defensible only because the new data is not yet in any public version.
   I would not: the claim in the abstract is now known to be narrower than stated, and we
   have already withdrawn one overreaching result in this same paper.

## Uncommitted local state

`git status` shows modified, all of it the revision responding to the second referee report:

    paper/neuroai2027/short_paper.tex       title Govern -> Shape, abstract and claims softened
    paper/neuroai2027/supplement.tex        classical C=1.0 and MAG/GRAD encoding notes
    paper/neuroai2027/audit/change-log.md   revision 9
    paper/neuroai2027/audit/checks.md       revision 9
    plus the rebuilt PDFs and figure

It builds and verifies. Commit it before starting anything new, or it will be hard to tell
your changes from mine.

## Still open

- The two experiments deferred to the journal version: a monolithic arm with a matched
  48-query budget, and a replay from raw data through the corrected pipeline.
- The multiple-comparison and endpoint-robustness points from the referee report. Both are
  stated as limitations; neither is fixable by a rerun.
- Camera-ready: restore authorship, the repository availability statement, and the upstream
  contribution note, then relax the anonymity assertions in `verify.py`.
- `paper/manuscript.md` is the original long-form manuscript. It has had nine corrections
  applied but is not the submission.

## Conventions

No em dashes anywhere, in any file. No emojis. `verify.py` fails the build on an em dash.
Keep the paper anonymous until camera-ready; `verify.py` checks for author strings.
