# Numerical reproducibility review

Role: independent numerical reproducibility reviewer. Diagnosis only. Original manuscript unchanged.

Scope: paper/manuscript.md, results/summary_phase4_nottingham_full.json, results/summary_phase3_ablations.json, results/summary_clean_visual_detection.json, results/summary_ds000117_squid_baseline.json; relevant Nottingham benchmark, wrapper, metric and SQUID benchmark code. Saved outputs were inspected with Python stdlib JSON and numbered source listings. Commands returned exit 0. Models and raw data were not rerun.

## Findings

| ID | Severity / confidence | Manuscript location | Evidence and impact | Smallest remedy |
|---|---|---|---|---|
| N1 | blocker / high | paper/manuscript.md:155,190 | Attribution of 76.1% of deficit to vector interference is not identified by the comparison. scripts/run_nottingham_full_benchmark.py:188-189 applies Soft-HFC to axis split but not random split or raw. The ratio 0.0854391/0.1123336 is arithmetically about 76.1%, but numerator includes preprocessing differences and is not a causal variance decomposition. | Label axis split as axis split plus Soft-HFC. Remove mechanism percentage pending preprocessing-matched controls and direct mechanism evidence. |
| N2 | major / high | paper/manuscript.md:148 | Random-split BACC is presented as that architecture's decoding score. scripts/run_nottingham_full_benchmark.py:229-233 explicitly computes monolithic embeddings for this row. It is not a random-split feature decoding test. | Show decoding as not evaluated for random split, or clearly label reused monolithic baseline. |
| N3 | minor / high | paper/manuscript.md:149-150 | Per-subject JSON channels yield Z 53-60 and XY 102-116. Manuscript gives 55-60 and 100-116. Example sub-105 Z count 53 is at results/summary_phase4_nottingham_full.json:448. | Correct ranges to 53-60 and 102-116. Full-array range 155-176 is correct. |
| N4 | major / high | paper/manuscript.md:118 | Near-perfect classical baseline omitted from visual-detection narrative. results/summary_clean_visual_detection.json:832-835 gives 99.9814% [99.9442,100] at B=20; pretrained is 99.8137%. This limits performance-superiority framing. | Include classical comparator. No inferential ranking is established by point estimates alone. |
| N5 | major / high | paper/manuscript.md:13,116,118,129 | Several p<0.0001 claims are absent from corresponding summary JSON. metrics.py:42-74 uses 10000 random sign flips with +1 correction, whose minimum is 1/10001, slightly below 0.0001. That threshold is numerically possible but cannot be verified from the reviewed saved summary. | Report available confidence intervals until test provenance is supplied or explicitly recomputed and logged. |
| N6 | major / high | paper/manuscript.md:186 | A relative PCC improvement of 52.7% is described as recovering over half of reconstruction loss. PCC ratio does not establish loss recovery. If deficit were 1-PCC, recovery would instead be about 17.6%, but that estimand is not established here. | State relative PCC improvement, without loss-recovery interpretation. |
| N7 | minor / high | paper/manuscript.md:167-176 | SQUID values are ordinary accuracy, from scripts/run_squid_in_domain_benchmark.py:293, not balanced accuracy. Tables do not always distinguish metrics when comparing OPM and SQUID. | Label SQUID accuracy explicitly and avoid treating it as directly comparable to OPM BACC. |
| N8 | major / high | paper/manuscript.md:155 | Random partitions use equal thirds (np.array_split, lines106-108), one seed 42, not exact axis-bank sizes. This matches number of banks but not exact group sizes, orientations or preprocessing. | Describe as equal-size three-way random partition, one seed, and limit capacity-control interpretation. |

## Values traced with no numerical mismatch

Triaxial saved summary, results/summary_phase4_nottingham_full.json:6-174:

- Raw PCC 0.3726098 [0.3416135,0.4046306].
- Soft-HFC PCC 0.3461372 [0.3114862,0.3815673].
- Random split raw PCC 0.3995043 [0.3696539,0.4286862].
- Radial raw PCC 0.4944299 [0.4666712,0.5265443].
- Tangential raw PCC 0.4446406 [0.4111549,0.4770760].
- Axis split plus Soft-HFC PCC 0.4849434 [0.4507357,0.5198654].
- Axis-plus-HFC minus raw difference 0.1123336 [0.1041877,0.1201013]; saved Monte Carlo p=0.00159984.
- Axis-plus-HFC minus random raw difference 0.0854391 [0.0753106,0.0955358]; saved Monte Carlo p=0.00159984.
- Ten subject entries, all 84 trials; full arrays 155-176 channels.

OPM geometry summary: default PCC 0.2500615 [0.2450055,0.2548928], remapped 0.3818502 [0.3779064,0.3858641]. Manuscript rounding and 52.7% relative increase agree.

SQUID summary: native magnetometers 0.4104952, remapped 0.7129925. Manuscript 73.7% relative increase agrees. Single subject control only.

Visual detection summary: B=1 pretrained 95.3865%, B=20 pretrained 99.8137%, random 96.5984%, pretrained-tokenizer/random-backbone 95.4310%, compact network 99.8135%. Rounded narrative values agree. Pretrained minus random difference 3.2153 percentage points [2.1256,4.4189] agrees.

## Further limitations

- Checks establish consistency with saved outputs, not validity of preprocessing, dataset claims, raw data or executable provenance.
- Nottingham PCC averages batch means equally (scripts/run_nottingham_full_benchmark.py:75-90,198-207), so the final 4-trial batch receives the same weight as a 16-trial batch. This should be stated or corrected through a rerun if trial-weighted PCC is intended.
- All four saved Nottingham p-values come from the same Monte Carlo sign-flip procedure. They should not be described as exhaustive exact p-values.
- Bibliography, pretraining corpus proportions, checkpoints, model parameter counts, raw figures and corrected/retracted source status were not independently verified.
