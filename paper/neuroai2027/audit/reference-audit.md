# Reference verification, revision 12

The bibliography now has eight references. Three sources were added to address methods, hardware and related-model coverage. Existing result estimates and dataset references are retained. The rewritten BrainOmni claim was also checked against its author full text.

Full-text support is an editorial finding documented below; the automated verifier checks citation keys, compiled bibliographies and attribution boundaries. It does not determine scientific support by itself.

## jiang2024labram

**Claim:** LaBraM learns reusable EEG representations through channel patches and masked neural-code prediction.

**Source:** [Source 1](https://proceedings.iclr.cc/paper_files/paper/2024/file/47393e8594c82ce8fd83adc672cf9872-Paper-Conference.pdf) [Source 2](https://proceedings.iclr.cc/paper_files/paper/2024/hash/47393e8594c82ce8fd83adc672cf9872-Abstract-Conference.html)

**Locators:** Section 2.1, printed pages 3-4: channel patch architecture; Section 2.2, printed pages 5-6: neural tokenizer; Section 2.3, printed page 6: masked neural-code pretraining.

**Scope:** EEG method context only; no claim of OPM evaluation or superiority over BrainOmni.

**Status checks:** No correction or retraction notice found on official proceedings entry or author arXiv record; targeted title search found none. No comprehensive Crossmark check.

## rea2022triaxial

**Claim:** Unlike fixed cryogenic SQUID arrays, OPMs permit flexible near-scalp placement; triaxial sensors measure three orthogonal field components.

**Source:** [Source 1](https://nyaspubs.onlinelibrary.wiley.com/doi/full/10.1111/nyas.14890)

**Locators:** Introduction, first paragraph: cryogenics and adaptable near-scalp placement; Introduction, triaxial discussion and Figure 1A: orthogonal components; Methods, System overview: wearable sensor array.

**Scope:** Does not establish our tokenizer grouping benefits, exact physical co-location, or the removed sensor-distance values.

**Status checks:** No correction or retraction notice on publisher full-text page or targeted DOI search; Crossmark not independently obtained.

## tierney2021hfc

**Claim:** Conventional HFC approximates distant interference as spatially homogeneous and removes its orientation-defined subspace.

**Source:** [Source 1](https://discovery.ucl.ac.uk/id/eprint/10134967/1/1-s2.0-S1053811921007576-main.pdf) [Source 2](https://doi.org/10.1016/j.neuroimage.2021.118484)

**Locators:** PDF page 2, Section 2.1, equations 4-7: orientation matrix and homogeneous approximation; PDF page 2, Section 2.2, equation 8: hard projection I-NN+; PDF page 3, Section 3.2: possible attenuation of neural signals.

**Scope:** Our inverse expression assumes full column rank. Soft-HFC and alpha=0.25 are evaluated analysis choices, not established in this source.

**Status checks:** No correction or retraction notice found in UCL/PubMed records or targeted DOI search. Publisher access blocked for reviewer; comprehensive Crossmark status not verified.

## xiao2025brainomni

**Claim:** BrainOmni combines sensor-conditioned tokenization, residual vector quantization and masked-token learning for EEG and MEG.

**Source:** [Source 1](https://arxiv.org/html/2505.18185v2) [Source 2](https://proceedings.neurips.cc/paper_files/paper/2025/hash/3aef4a18c2646b9d11d9ab4d9bec72c8-Abstract-Conference.html)

**Locators:** Section 2.2, Sensor Encoder, Channel Compression and Quantisation; Section 2.3, masked-token model.

**Scope:** Cited only for shared architecture. Checkpoint-specific sensor integers and mask ratio are grounded in local code, not copied from the publication. No field-wide corpus composition or sensor-distance claim remains.

**Status checks:** No correction notice visible in consulted official metadata/author version; no full renewed correction-status audit of all existing dataset references.

## Review and scope

Independent methods and hardware reviewers inspected the Tierney and Rea full texts. A separate final reviewer checked the integrated citation wording and independently inspected LaBraM and BrainOmni; no actionable integration findings remained.

The hardware source supports qualitative placement and orthogonal components, not the precise distances or exact co-location previously stated. Those details were removed. The field-wide training-corpus claim and growth claim were also replaced with a narrower motivation, without asserting prevalence across all foundation models.

The HFC source describes hard projection and a homogeneous-field approximation. The supplement now distinguishes that method from the evaluated soft variant, states the full-rank assumption, and attributes the shrinkage coefficient to this analysis. Checkpoint-specific BrainOmni settings remain local-code claims.

No comprehensive new correction-status review of the existing dataset references was performed. The recorded status checks establish only that no notice was found in the consulted sources.
