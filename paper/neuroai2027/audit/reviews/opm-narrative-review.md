# Independent narrative and citation review

Scope: paper/manuscript.md, local model implementation, official BrainOmni NeurIPS PDF, Zenodo 11126593, and the primary ds005107 article. Diagnosis only. No manuscript edits. Sources retrieved November 4, 2026, user timezone.

## Findings

- N1, major, high confidence. manuscript.md:13,23,35,112,118,176,188 repeatedly calls BrainOmni autoregressive. This is contradicted by the official paper, Section 2.3, page 5, explicitly describing non-autoregressive masked prediction, and local vendor_brainomni/brainomni/model.py:135-169. Replace with masked-token-pretrained spatiotemporal Transformer. Cross-trial temporal inference at lines 112/118 is unsupported by a model processing epochs independently; describe the measured pretrained-backbone benefit without assigning a mechanism.
- N2, major, high confidence. manuscript.md:43,178 attributes remapping gains causally to gradiometer prevalence. The experiments manipulate tags, not the pretraining distribution. The SQUID control shows that the effect is not unique to OPMs, but does not identify training imbalance as its cause. Keep the sensitivity result and label imbalance as a hypothesis. The claimed >4:1 ratio needs a weighting definition and reproducible calculation. The cited upstream appendix is also wrong: published PDF Appendix H contains pretraining information, not D.
- N3, major, high confidence. manuscript.md:155,190 says 76.1% is attributable to vector interference. Axis partitions versus random partitions distinguish grouping strategies, not a unique physical mechanism. Preserve the measured +0.0854 advantage over random partition, but do not interpret the quotient as a causal decomposition. Verify numerical comparison independently.
- N4, major, high confidence. manuscript.md:111-112 infers an SNR floor from near-chance decoding. No SNR manipulation or bound establishes this. Table 1 itself includes confidence intervals above 50% for some controls. Report near-chance performance under the tested cross-subject protocol. The primary ds005107 publication reports within-subject face-dimension decoding under a different analysis, so avoid suggesting the dataset has no decodable information.
- N5, major, high confidence. manuscript.md:13 and 62 uses demographic terms that can imply participant traits. These are labels on synthetic face stimuli, not participant demographics. The primary article explicitly describes StyleGAN2 stimuli with manipulated apparent race, gender, age, expression. Use stimulus-attribute decoding, preserve source label definitions, and avoid substituting genetic ancestry for the stimulus category called race by the source. Source article is missing from bibliography.
- N6, major, high confidence. manuscript.md:186 says remapping recovers over half of reconstruction loss. A 52.7% relative PCC increase is not half of a loss recovered. Use PCC change 0.2501 to 0.3819, pending numerical verification.
- N7, minor, high confidence. manuscript.md:207 lists OpenTSLab as author. Official paper authors: Qinfan Xiao, Ziyun Cui, Chi Zhang, Siqi Chen, Wen Wu, Andrew Thwaites, Alexandra Woolgar, Bowen Zhou, Chao Zhang. Cite Xiao et al. 2025. manuscript.md:208 gives RP94562, but official Zenodo links DOI 10.7554/eLife.94561.1. Resolve final eLife version before final bibliography.
- N8, major, high confidence. manuscript.md:1,25,184,192 generalizes from one evaluated foundation model to the model class. Bound title and conclusions to BrainOmni and the evaluated cohorts. Denoising and cross-modality transfer in line 192 are not established by these experiments.
- N9, major, high confidence. manuscript.md:5-7,198-201 reveal authors and repository provenance. Remove from anonymous review copy and provide anonymous code supplement when ready. Target venue at line 3 must change to AAAI-27 NeuroAI.
- N10, minor, high confidence. Four-page submission has too many competing narratives: demographic decoding, detection scaling, sensor remapping, axis partition, SQUID evaluation. Center on reconstruction sensitivity to sensor tags and axis grouping. Put detailed downstream negative results and exploratory single-subject SQUID control in supplementary material, retaining limitations and one concise downstream-results paragraph in the paper.

## Recommended narrow narrative

Title direction: Evaluating BrainOmni Transfer to OPM-MEG: Sensor Tags and Axis-Partitioned Tokenization.

Question: How sensitive is pretrained BrainOmni reconstruction to sensor metadata and channel grouping on two OPM cohorts?

Main results: tag substitution changes reconstruction PCC on radial OPM data; axis grouping improves PCC relative to joint and random-group processing on triaxial data. These results diagnose deployment sensitivity and motivate broader hardware-aware evaluation. They do not establish a physical interference mechanism, universal failure of foundation models, or decoding improvement from reconstruction improvement.

Possible four-page layout: introduction 0.6 page; model/data/protocol 1 page; reconstruction results and two compact panels/table 1.5 pages; downstream boundary and limitations 0.6 page; conclusion 0.3 page. References additional pages.

## Sources and checks

1. BrainOmni, official NeurIPS full text: https://papers.neurips.cc/paper_files/paper/2025/file/3aef4a18c2646b9d11d9ab4d9bec72c8-Paper-Conference.pdf . Identity and masked training verified in title and Section 2.3. No correction/retraction status audit performed.
2. Zenodo dataset: https://zenodo.org/records/11126593 . Dataset identity, 192-channel hardware description, adult/child archives, and eLife source DOI verified. Exact adult subset not verified from downloaded raw files.
3. Xu et al. 2024: https://pmc.ncbi.nlm.nih.gov/articles/PMC11580774/ . Title: Decoding the Temporal Structures and Interactions of Multiple Face Dimensions Using Optically Pumped Magnetometer Magnetoencephalography (OPM-MEG). Journal of Neuroscience 44(47):e2237232024. DOI 10.1523/JNEUROSCI.2237-23.2024. Participant cohort of 21, artificial face stimulus attributes, and ds005107 provenance verified. No correction/retraction audit performed.
4. Local commands: nl -ba paper/manuscript.md; rg -n 'MAG|GRAD|sensor_type|type.*[12]|autoregress|mask' on wrapper/model/schema/sensor_mapping files. All exited 0. Checks were read-only.

No finding: manuscript clearly distinguishes the two OPM cohort sizes in methods; the use of 50 shared channels is explained; random-partition and single-axis controls are reported, rather than omitted.

Not checked: raw data, saved result arithmetic, permutation implementation, clean visual detection task construction, reference entries 3-7, complete training-corpus composition, final PDF page count. These require other review passes.
