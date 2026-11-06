# Independent methods and statistics review

Role: methods, statistics, and claims reviewer. Diagnosis only. No manuscript changes.

Corpus: paper/manuscript.md; scripts/run_transfer_benchmark.py; scripts/run_nottingham_full_benchmark.py; src/opm_bridge/models/wrapper.py; src/opm_bridge/eval/metrics.py; results/summary_clean_visual_detection.json; results/summary_phase4_nottingham_full.json.

## Findings

1. M01 | blocker | high confidence | manuscript abstract and lines 116-118. The visual-detection claim treats the contrast as physiological and unconfounded. In scripts/run_transfer_benchmark.py:349-354, evoked samples occupy a 128-sample interval, whereas the identical 64-sample baseline segment is repeated twice only for the negative class. This creates label-dependent periodicity and a boundary feature. Code establishes the confound; its actual contribution to accuracy is unknown without rerunning. Minimum remedy: exclude the detection result from central physiological claims, report the exact construction as exploratory if retained, and rerun with equal-duration genuine intervals and identical processing. Cross-subject evaluation does not eliminate this class-construction cue.

2. M02 | blocker | high confidence | paper/manuscript.md:155. The 76.1% attribution to vector interference is not isolated. scripts/run_nottingham_full_benchmark.py:188-189 applies Soft-HFC only to triaxial_xyz_softhfc and phase5_axis_split_xyz. The random split and raw monolithic comparator receive no HFC. Thus axis-versus-random also changes preprocessing. Random groups are created with np.array_split using one seed (lines 106-108, 227), not matched to each actual axis-bank count. Minimum remedy: remove causal fraction attribution, explicitly call the reported adapter condition axis partitioning plus Soft-HFC, retain descriptive PCC with limitation, and run both partitions under matching HFC settings with repeated partitions before mechanism claims.

3. M03 | major | high confidence | paper/manuscript.md:100,112,118. Control B is called tokenizer only, but scripts/run_transfer_benchmark.py:410-412 loads a pretrained tokenizer plus random backbone and lines 430-433 encode through Transformer. It is not a tokenizer-only ablation. Minimum remedy: consistently name it pretrained tokenizer plus random backbone. The performance difference supports a comparison with that control, not the necessity of a backbone over no backbone.

4. M04 | major | high confidence | paper/manuscript.md:112,118. Cross-trial temporal structure is asserted, but samples are processed as independent trials in scripts/run_transfer_benchmark.py:349-356 and in batch feature extraction. No evidence reviewed shows temporal modeling across trials. Minimum remedy: remove cross-trial mechanism wording and describe the actual learned representation comparison; within-trial temporal modeling may be investigated separately.

5. M05 | major | high confidence | paper/manuscript.md section 3.4. The code in src/opm_bridge/eval/metrics.py:173-199 implements paired percentile bootstrap intervals with a bespoke decision rule, not a pair of reported one-sided tests. The positive-benefit branch precedes equivalence. The manuscript should call this the implemented 95% bootstrap equivalence criterion, give the 3 percentage-point bound, and describe sign-flip tests separately. Do not label it conventional alpha=0.05 TOST without a corresponding test implementation. Saved summary_clean_visual_detection.json lacks p-values in its primary comparison objects; manuscript p-values therefore need independently traced generation or recomputation from saved subject scores.

6. M06 | major | high confidence | paper/manuscript.md:148. Braille accuracy shown for random three-way partition is not derived from partitioned representations. scripts/run_nottingham_full_benchmark.py:229-233 explicitly substitutes monolithic embeddings for this row. Minimum remedy: leave random-partition decoding blank or mark it as monolithic reference; retain its actual reconstruction result.

7. M07 | major | high confidence | paper/manuscript.md:111-112. A universal chance floor is contradicted by the same table: CompactMEGNet age B=20 CI is [50.33,53.34], and pretrained-tokenizer/random-backbone ancestry CI is [50.37,52.55]. This does not establish broad robust decodability or correct multiplicity, but rules out the literal statement all configurations sit at chance. Near-chance decoding also does not identify SNR or absent anatomical MRI as its cause. Minimum remedy: report generally weak decoding and uncertainty without claiming an established physical floor.

## Checks with no finding

The Nottingham raw, random partition, and adapter PCC means and bootstrap intervals in Table 3 agree to reported precision with results/summary_phase4_nottingham_full.json. The displayed p=0.0016 matches saved Monte Carlo sign-flip outputs. metrics.py:73 uses (count+1)/(10000+1); this finite-sampling value is not an exact enumeration p-value. No rerun was performed.

## Limitations

No raw data, training run, checkpoint integrity, citation support, ethics permissions, or dataset metadata was verified. These are code-to-manuscript observations, not a complete reproducibility validation. The confound mechanisms and interpretation risks are reviewer inferences grounded in the specified code. Numerical effects of corrections remain unknown.

## Checks performed

Read line-numbered source excerpts with rg and nl/sed. Parsed the two JSON summaries using Python json. No new statistical test or experiment was performed. All tool commands completed with exit code 0.
