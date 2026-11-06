# Reproducibility and numerical integrity review

Role: independent numerical reviewer. Date: 2026-11-05, America/Toronto.

Reviewed current revision 5 in `/Users/marziehzare/Desktop/papers/opm-bridge-latest/`, specifically main and supplement LaTeX, generated tables, figure sidecar, verification/generation scripts, saved result records, Nottingham evaluation code, metrics, configurations and lockfile. No prior review was read before this report. No manuscript edits or model inference were performed.

Version hashes (SHA-256):

- `paper/neuroai2027/short_paper.tex`: `40283d8b9d97a207ad1e93f35031dcba13f4311d974d8a8f94bd495224ab1084`
- `paper/neuroai2027/supplement.tex`: `2aab749768d37a445eba15b59add1cb0fb5f93df8bf75a07ee691fac4ddcbb1e`
- `results/summary_phase4_nottingham_full.json`: `b69bd137acdbd4e8cdc9c8e77fbb042591bd9b92f3f192fb6944e19ea3dbe09f`

## Verdict

Major revision for provenance and complete reporting, with the central saved reconstruction arithmetic supported. This is not evidence that the underlying runs are invalid. A definitive end-to-end reproducibility claim remains unavailable without run artifacts and checkpoint verification. Other roles may identify scientific problems beyond this numerical scope.

## Supported strengths

The revised triaxial JSON contains all eight configurations, including all six factorial cells. The primary axis-versus-random differences are supported by the saved participant records. I executed the supplied read-only `audit/verify.py`; it passed. It recalculates paired contrasts, bootstrap intervals for contrasts, exact sign-flip probabilities, direction counts, decomposition percentages, decoder sidecar values, and checks build logs. The main means and supplementary participant reconstruction values are consistent with the saved results at the displayed precision. The random-partition probe no longer duplicates the monolithic mean. The report correctly states that model inference was not replayed and acknowledges batch-mean weighting. Both configuration SHA-256 values match the local files exactly.

## Findings

### NUM-01. Major: environment provenance conflicts with the supplied lockfile

Location: `paper/neuroai2027/supplement.tex:208-211`, `uv.lock`; corresponding older prose in `paper/supplement.md:175-184`.

Observed: supplement specifies MNE 1.9.0, scikit-learn 1.6.1, NumPy 2.2.3 and SciPy 1.15.2 while asserting the dependency graph is locked in the code supplement. Parsing the supplied lockfile finds MNE 1.12.1/1.13.2, sklearn 1.7.2/1.9.1, NumPy 2.2.6/2.4.6/2.5.4, and SciPy 1.15.3/1.17.1/1.18.1 across conditional package branches. None of the four claimed versions is present. PyTorch 2.15.0 is present.

Consequence: the supplied lockfile cannot reconstruct the stated environment. It is possible the runs used a different environment, but that needs evidence.

Remedy and closure: supply the run-time environment export with a run identifier and matching lockfile, or explicitly label the listed versions as author-reported and the lockfile as a later environment. Do not describe the present lock as reproducing the reported runs. Confidence: high for mismatch, unknown for actual run environment.

### NUM-02. Minor: main and supplement report different confidence intervals for the same cohort means

Location: `short_paper.tex:48-49, 104-113`; `supplement_tables.tex:52-59`; `scripts/run_nottingham_full_benchmark.py:324-335`; `make_supplement.py:47-50,138-142`.

Observed: main Table 2 gives monolithic raw CI [0.3416,0.4046], supplement cohort table gives [0.3405,0.4048]. The other cohort CIs differ similarly. Main table retains saved 10,000-resample seed-42 summaries, whereas supplement recomputes 20,000-resample seed-0 intervals. Main statistics paragraph reports 20,000 resamples without this distinction.

Consequence: small differences do not change conclusions, but the common analysis is inconsistently specified and displayed. `verify.py:82-84` trusts upstream cohort summary fields for these checks instead of recalculating them consistently.

Remedy and closure: generate both tables from the same bootstrap settings or explicitly document the two conventions. Require named table-cell equality in verification. Confidence: high.

### NUM-03. Minor: main-table consistency labels reverse the intended count

Location: `short_paper.tex:76-82` and caption, contrasted with `supplement_tables.tex:38-41`.

Observed: main Table 1 lists negative differences with 0/21, 0/21, and 2/21, while its caption says consistency counts participants moving in the stated direction. Those counts measure improvements, not the negative stated direction. The correct negative-direction counts are 21/21, 21/21, and 19/21, as supplement reports. Main text line 93 similarly says filtering costs are "all in 0/10 participants" despite all ten showing a cost.

Consequence: literal reading contradicts the consistency claim.

Remedy and closure: label counts explicitly as number improved throughout, or consistently count the stated sign and change the main counts. State filtering lowers PCC in 10/10 participants. Confidence: high.

### NUM-04. Minor: checkpoint identifiers promised in the supplement are absent

Location: `supplement.tex:213-217`; `supplement_tables.tex:140`; compare `paper/supplement.md:154-160`.

Observed: supplement says the checkpoint hash is supplied by the run author and table caption says it is reported in the text. No checkpoint SHA-256, model-configuration hash, or upstream revision appears in this LaTeX supplement. They are present in the Markdown source, but not transferred into the submitted supplement.

Remedy and closure: include upstream repository/revision, checkpoint and model-config hashes as author-reported and unverified, with retrieval instructions. This does not require claiming local verification. Confidence: high.

### NUM-05. Minor reporting defect: decoding uncertainty promised but omitted, with an incorrect summary count

Location: `supplement.tex:25-29,125-126`; `short_paper.tex:123`; `supplement_tables.tex:112-128`.

Observed: opening supplement promises complete paired tests, but decoding table has means and sign counts only, no paired CIs/p-values. Both texts say "most intervals cross zero." Independent participant bootstrap using the declared 20,000 resamples and seed 0 shows exactly four of eight classical-minus-probe intervals crossing zero, not most.

Independent recalculation, in percentage points:

| Configuration | Difference | 95% CI | Exact unadjusted p |
| --- | ---: | --- | ---: |
| Monolithic raw | 5.3056 | [-0.7778,10.2778] | .115234 |
| Monolithic Soft | 5.4306 | [0.1944,10.3194] | .083984 |
| Random raw | 2.7083 | [-2.1389,6.8611] | .292969 |
| Random Soft | 5.9861 | [1.0833,10.8611] | .056641 |
| Axis raw | 6.0139 | [1.1528,10.4722] | .048828 |
| Axis Soft | 4.3056 | [-3.3476,11.2222] | .298828 |
| Radial raw | 6.1389 | [1.8472,10.0417] | .031250 |
| Tangential raw | 3.4167 | [-2.2642,8.3472] | .265625 |

Consequence: readers cannot inspect the tests on which the cautious inferential text relies. The eight conditions reuse the same participants and four classical endpoints, so eight favorable mean signs are not eight independent replications. Broader interpretation is deferred to methods review.

Remedy and closure: publish the paired uncertainty table, mark tests exploratory/unadjusted as appropriate, and replace "most" with "four of eight" or "several". Confidence: high.

## Verification limits and follow-up

No raw data, model checkpoint or cached inputs are present for end-to-end replay. No expensive inference was run. Provenance and exact reproduction from raw recordings remain unverified. The verifier passing demonstrates substantial saved-output consistency, but its number-substring membership tests do not establish that every number is in the correct row or that all prose claims are accurate. The checkpoint sensor-type semantic mapping is being independently investigated by the claims reviewer and is not adjudicated in this initial report.
