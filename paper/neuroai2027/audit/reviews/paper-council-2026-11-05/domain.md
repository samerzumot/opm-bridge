Role: domain and contribution reviewer. Review date: 2026-11-05.

Verdict: Major revision. The observed reconstruction results support a useful diagnostic study, but the strongest mechanism and architectural prescriptions exceed the interventions tested. These issues are repairable by narrowing the claims; a new experimental programme is not a prerequisite for that narrower paper.

Version: current workshop sources under `/Users/marziehzare/Desktop/papers/opm-bridge-latest/paper/neuroai2027/`, reviewed `short_paper.tex`, `supplement.tex`, and relevant axis/random partition implementation. Manuscript SHA-256: `40283d8b9d97a207ad1e93f35031dcba13f4311d974d8a8f94bd495224ab1084`.

Supported strengths

- The strongest contribution is the within-model, frozen-weight audit of sensitivity to sensor metadata and inference-time grouping. A categorical intervention holding waveforms constant is a clear diagnostic, while the complete partition-by-filter grid removes the previous filtering ambiguity. Main lines 26, 44-46, 61, 87-89.
- The axis-minus-random contrasts retain the full channel target and match three tokenizer passes. This is more informative than comparing radial-only reconstructions to a full-array reconstruction. The implementation reassembles the full array in `models/wrapper.py:309-327`, and the random control does likewise in `scripts/run_nottingham_full_benchmark.py:120-138`.
- The manuscript correctly distinguishes processed-waveform reconstruction from neural fidelity, states approximate FACE-DEC geometry, limits the adult cohort and single-participant SQUID control, and explicitly withdraws the compromised detection endpoint. Main lines 33-42 and 128-130.

D1. Major, high confidence. Mechanistic partition is stronger than the design identifies.

Location: main lines 20, 28, 87-89, 135-140; supplement lines 98-109.

Evidence: the manipulation compares monolithic processing with one random partition and an axis-defined partition. It does not vary query capacity independently of channel membership, and the supplement explicitly concedes that groups are not exactly size matched. The 21/79 percent calculation is a valid decomposition of these observed means, conditional on this random partition. It is not a general separation of spatial capacity from orientation mechanism. Axis grouping can also change within-group waveform covariance, channel composition, and how the pretrained tokenizer distributes attention. These are plausible alternatives, not demonstrated causes. Neither a single comparator nor lack of a matched monolithic token budget proves that the observed remainder is an upper bound on an orientation effect.

Consequence: the paper risks turning a useful intervention result into an unsupported causal mechanism, especially in the abstract and conclusion.

Remedy: report the two measured contrasts directly. If retaining percentages, call them the relative sizes of those contrasts under the tested partition, not percentages of a causal effect attributable to capacity and orientation. Replace 'size-matched' with 'approximately equal-size, pass-count-matched' unless exact group counts are established. Remove the upper-bound assertion.

Closure: abstract, results, supplement and conclusion consistently describe conditional intervention effects. Repeated size-matched random partitions are an optional strengthening experiment for the narrower claim, not an obligatory new study.

D2. Major, high confidence. Continuous sensor encoding is an untested proposal, not an established practical conclusion.

Location: main lines 20, 133, 140.

Evidence: the study varies a learned categorical label and metadata supplied to one pretrained model. It never trains or evaluates the proposed continuous sensing parameterization. The existing categorical intervention establishes sensitivity of this implementation; it does not establish that replacing the category is superior to recalibration, balanced pretraining, richer categorical metadata, or a different checkpoint. Likewise no waveform intervention supplies a common effect-size comparison for the conclusion that transfer is governed 'less by the waveforms' than by interface choices.

Consequence: a justified diagnosis is presented as a validated architecture prescription and general hierarchy of causal importance.

Remedy: describe continuous parameterization as a motivated hypothesis for future evaluation. Restrict the conclusion to the observed model, cohorts, and frozen reconstruction endpoint. Suggested framing: 'These results motivate testing continuous sensor descriptions and axis-aware query allocation, with decoding retained as a separate endpoint.'

Closure: no sentence claims comparative superiority of an untested continuous architecture, or ranks waveform importance without a waveform comparison.

D3. Major, high confidence. Decoding interpretation must remain specific to the tested readout.

Location: main lines 20, 119-123; supplement lines 116-126.

Evidence: the results give low mean accuracy for one frozen feature/readout construction and modestly higher classical means; the text itself states weak paired tests and repeats four classical outcomes across eight configurations. Consistent directions across dependent configurations do not establish a broad absence of useful information in the pretrained representation or rule out readout, temporal aggregation, regularization, finite sample, or task limitations. The concrete result is no demonstrated decoding benefit accompanying the reconstruction gain in this evaluation.

Consequence: the stated representation failure and rejection of a dataset noise-floor explanation exceed the evidence shown.

Remedy: use 'no demonstrated benefit for the evaluated linear probe' and report paired uncertainty. Retain the classical comparison as evidence that reconstruction ranking and the observed decoder ranking differ. Do not require a broad readout search merely to support this narrower statement.

Closure: claims remain conditional on the tested readout and no dependent row count substitutes for statistical evidence.

Limits

This was a focused domain pass with source inspection, not a model rerun or independent numerical replication. No exhaustive novelty search was performed, so priority over prior work is unverified. No new external scientific assertion is needed for the findings above. Other role reports were not opened; inherited parent context included preliminary findings, so this is a separately authored role pass rather than a fully blinded review.
