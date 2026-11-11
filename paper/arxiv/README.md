# arXiv preprint preparation

Role: scientific editor. Prepared November 10, 2026.

This is a standalone, named-author preprint derived from the latest revised NeuroAI manuscript and its current factorial supplement. It includes the complete supplementary material in one PDF. It does not use the older, broader claims in `paper/manuscript.md` or the repository README as its scientific narrative. No submission has been made.

## Deliverables

- `output/pdf/opm_bridge_arxiv.pdf`: readable preprint with integrated supplement.
- `opm_bridge_arxiv_source.tar.gz`: minimal source upload archive.
- `main.tex`: top-level source; choose pdfLaTeX when uploading.
- `main_text.tex`, `supplement_text.tex`: editable scientific text.
- `build.py`: reproducible build and archive checks, requiring Python 3.12+, pdfLaTeX, BibTeX and pdftotext.
- `build/validation.json`: source hashes and local verification results.
- `submission_metadata.txt`: title, author order, abstract and category suggestion.

Run `python3 build.py` from this directory. The script checks numerical assets against the reviewed workshop version, compiles the PDF, packages only its dependencies, extracts the archive into a fresh temporary directory, and compiles again. The upload includes the bibliography source and compiled `main.bbl`, figures, and required TeX files. It excludes review reports, intermediate files, and the final manuscript PDF.

## Changes from the reviewed workshop source

1. Neutral single-column article format, author names from the existing manuscript, fixed date, linked references, and integrated supplementary sections and tables.
2. Decoder regularization corrected to C=0.01 for the supplied factorial implementation. Added training-fold standardization, balanced class weights, iteration limit and split seed. Historical estimator settings remain unverified.
3. Defined zero-shot as no OPM-specific weight updates, and reconstruction as unmasked reconstruction of the supplied input.
4. Defined raw as without additional Soft-HFC and documented the cache preprocessing implemented by the supplied script. The high nominal band-power frequencies are qualified by the earlier 45 Hz low-pass.
5. Replaced the unmeasured threefold compute claim with the verified three passes and 48 aggregate queries. Proposed architectural benefits remain hypotheses.
6. Clarified the interaction and conditional uncertainty; added data/code availability and a factual AI-assistance disclosure.
7. Expanded the introduction with Boto et al. (2017), Boto et al. (2018), and BIOT (Yang et al., 2023), bringing the cited bibliography to 11 entries. Source support is recorded in `reference_additions.json`.
8. Preserved all numerical macros, the figure, and every supplementary table value. The workshop files are unchanged.

## Author-controlled items before public release

- The confirmed author order is Samer Zumot first (1), Marzieh Zare second (2). Affiliation 1: Independent NeuroTech Researcher, Toronto, ON, Canada. Affiliation 2: Université Laval, School of Psychology, Quebec City, QC, Canada. The affiliation update does not change the author order. Obtain agreement to the final scientific text from both authors.
- Supply applicable funding, competing-interest, and secondary-use ethics/consent/license statements. Missing statements are unresolved information, not a finding of misconduct or a claim that new approval is required.
- Review the AI-assistance statement against the full history of the work. It currently describes assistance documented in this repository and this preparation.
- Confirm the repository is publicly accessible with the intended code and saved results. Its URL is inherited from the manuscript; anonymous public access was not established here.
- Select the arXiv distribution license and confirm rights to all included material. The repository's code license does not automatically choose the paper's license.
- Check any applicable workshop preprint/anonymity policy before public posting. The workshop page specifies double-blind review but does not itself settle all preprint timing questions.

These items do not prevent preparing or reviewing this draft. The source archive is technically checked, but no final author approval or arXiv server compilation is claimed.

## Current arXiv guidance checked November 10, 2026

- [Submission guidance](https://info.arxiv.org/help/submit/index.html): topical, refereeable scientific contributions; registration and possible endorsement.
- [TeX packaging](https://info.arxiv.org/help/submit_tex.html): include required sources, figures and bibliography; remove extraneous files; inspect the server-generated PDF.
- [Moderation and AI disclosure](https://info.arxiv.org/help/moderation/index.html): moderation is not peer review; significant generative-AI assistance should be reported.
- [Category taxonomy](https://arxiv.org/category_taxonomy): suggested primary category cs.LG, with q-bio.NC as a possible cross-list. This is an editorial recommendation, subject to author choice and moderation.
- [NeuroAI workshop](https://neuroai-multimodal-workshop.github.io/): current short-paper track is non-archival and double-blind.

## Verification limits

The existing manuscript verification script passed. It recalculates claims from saved records, including 560 trial-record arrays, paired contrasts, seed sensitivity, figure values and supplementary tables. Independent methods review also recalculated the central factorial contrasts. No model inference, raw-data replay, full novelty search, comprehensive citation-status audit, or checkpoint-to-output provenance certification was performed.
