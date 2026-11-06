# Role: Claims, citations, and disclosure reviewer

Verdict: Major revision of interpretation, without evidence here that the saved reconstruction differences are invalid.

Reviewed independently on 5 November 2026. Current packet: `/Users/marziehzare/Desktop/papers/opm-bridge-latest/paper/neuroai2027/short_paper.tex`, `supplement.tex`, `supplement_tables.tex`, `references.bib`, and selected analysis source. Main source SHA-256: `40283d8b9d97a207ad1e93f35031dcba13f4311d974d8a8f94bd495224ab1084`. No prior reviews read before this report. No manuscript edits or model inference.

## Supported strengths and claim map

- The sensor-type intervention is clearly separated from waveform changes and the text explicitly avoids concluding that pretraining imbalance caused the effect (main lines 61-65). This is a defensible sensitivity result, conditional on provenance.
- Approximate FACE-DEC coordinates are disclosed (main line 33). The [dataset README](https://github.com/OpenNeuroDatasets/ds005107) explicitly confirms that these are Elekta-layout approximations for plotting, not measured OPM locations.
- The adult triaxial subset is distinguished from the developmental dataset (main line 35). The [Zenodo record](https://zenodo.org/records/11126593) confirms the 192-channel device and developmental population. It does not independently verify the 84-trial cache or retained-channel counts.
- The distinction between reconstruction and physiological validity is explicit (main line 42), and exclusion of the window-construction shortcut is clear (lines 127-130).
- The [BrainOmni paper](https://papers.neurips.cc/paper_files/paper/2025/file/3aef4a18c2646b9d11d9ab4d9bec72c8-Paper-Conference.pdf), sections 2 and 5.3, supports sensor-conditioned tokenization, masked-token modeling, and the selection of 16 latent queries. Its pretraining appendix lists conventional MEG systems. The exact greater-than-four-to-one channel imbalance is not independently derived in this pass.

## C1: Mechanistic attribution exceeds the completed grid

Severity: Major. Confidence: High. Locations: main lines 20, 87-89, 133-140; supplement lines 98-109; `scripts/run_nottingham_full_benchmark.py` lines 107-109.

Observed: the completed grid holds filtering constant and estimates contrasts between one seeded random partition, an axis partition, and the monolithic condition. Random groups use `np.array_split` on a permutation, not the actual axis group sizes. The manuscript nevertheless calls this size-matched and attributes 79-81% to orientation and 19-21% to pass count. It also calls the residual an upper bound on orientation's contribution.

Consequence: an algebraic decomposition of a contrast is not a uniquely identified causal decomposition. The random-versus-monolithic intervention also changes grouping, channel competition and normalization behavior; one partition does not estimate the distribution of random group effects. Uncontrolled differences can bias either direction, so a formal upper bound has not been established. A complete filtering grid resolves filtering confounding but not all capacity versus grouping confounding.

Remedy: report these percentages as descriptive fractions of the observed gain relative to the selected random control. Use approximately equal group sizes, and replace upper bound with an explicit statement that orientation-specific attribution remains uncertain. Present continuous sensor parameterization and per-axis capacity as hypotheses motivated by these results, not designs proved superior to all categorical alternatives.

Closure: consistent qualification in abstract, results, discussion, conclusion and supplement. Repeated exactly size-matched random partitions would strengthen interpretation but are not necessary if claims are narrowed.

## C2: The decoding null is interpreted too strongly

Severity: Major. Confidence: High for wording issue; inference about actual above-chance evidence requires the methods review. Locations: main lines 20, 119-123, 140; supplement lines 116-126.

Observed: classical cohort means are 52.76-54.50%, probe means 47.19-50.83%. Text concedes that most paired intervals cross zero, then relies on all eight configurations pointing the same way to say the dataset is not at a noise floor and the representation fails. These configurations reuse participants, trials, and four classical feature conditions; they are not eight independent replications. No above-chance classical test or equivalence assessment for the probe is presented in the supplement table.

Consequence: positive sample means are insufficient by themselves to establish population above-chance decoding, and nonsignificant probe effects are not demonstrated absence of useful information. The frozen representation, pooling, linear readout and finite-sample evaluation form one evaluated pipeline.

Remedy: retain the descriptive result, state that no reliable improvement was demonstrated with this probe, report uncertainty for classical accuracy relative to chance if supported, and avoid assigning failure uniquely to the representation. Explicitly identify repeated configurations as dependent. Do not require a new research project to keep a negative result.

Closure: claims match reported inferential evidence throughout, or stronger claims are supported by appropriate grouped permutation/uncertainty analyses.

## C3: Resolve a primary-paper versus implementation convention discrepancy

Severity: Minor. Confidence: High. Locations: main line 44 and supplement architecture/provenance sections.

Observed: the official BrainOmni paper, printed page 3, defines 0=EEG, 1=GRAD, 2=MAG. In contrast, the official [preprocessing constants](https://raw.githubusercontent.com/OpenTSLab/BrainOmni/main/factory/brain_constant.py), line 19, define `SENSOR_TYPE_DICT = {"EEG": 0, "MAG": 1, "GRAD": 2}`, matching this benchmark and its local `src/opm_bridge/data/schema.py` lines 27-28.

Consequence: a reviewer relying on the cited paper could reasonably think the headline label intervention has been reversed. Checking upstream implementation resolves the immediate concern; this is not evidence of an erroneous benchmark mapping.

Remedy: cite a pinned upstream implementation revision for the encoding convention and explicitly note that it follows the released code. If necessary, disclose the paper/code discrepancy in a supplement footnote. Preserve checkpoint-version uncertainty unless that link is established.

Closure: stable implementation reference and explicit convention in provenance. No reversal of reported results warranted by the evidence reviewed here.

## C4: Provenance text promises a checkpoint hash that is absent

Severity: Minor, potentially major reproducibility importance. Confidence: High. Locations: supplement lines 213-217 and `supplement_tables.tex` lines 130-141.

Observed: supplement says a checkpoint hash is supplied by the run author; the hash table caption says it is reported in text. The supplement and hash table contain only the two configuration hashes and no actual checkpoint digest.

Remedy and closure: include the exact author-supplied digest, model release identifier and source with explicit unverified status, or remove the promise and state it is unavailable. Numerical integrity review should adjudicate the larger provenance scope.

## C5: Disclosure information requires author confirmation

Severity: Evidence request, not a demonstrated ethics defect. Confidence: High that the submitted text lacks these statements.

Observed: main paper and supplement identify public datasets but do not include a dedicated secondary-use ethics/license statement, conflicts/funding statement or AI-assistance disclosure. Public availability alone does not establish every relevant condition. No claim of misconduct or definite venue violation follows.

Remedy: authors confirm applicable secondary-use conditions and required disclosures in the appropriate anonymous submission fields or text. Closure depends on actual venue requirements and author information. This role did not verify the current submission form or impose main-conference rules on a workshop.

## Limits

Read source and cited primary texts, not raw data or checkpoint. No numerical rerun, exhaustive novelty survey, retraction audit, ethics determination, or PDF layout inspection in this role. No external upload of draft material. The main revision can remain a useful sensitivity and reconstruction study after narrowing the mechanistic and decoding claims.
