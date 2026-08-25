# PhasePair ICASSP technical-content skeleton

**State:** `DRAFT / AUTHORITY0 / NO_RESULT / NOT_FOR_SUBMISSION`

This directory contains a compact, four-page-target technical-content skeleton. It is not an official ICASSP manuscript and must not be submitted, described as submission-ready, or compiled and circulated as a final paper.

## Files

- `main.tex` — anonymous two-column content skeleton with exactly three core display-equation blocks: score-symmetric MIME residual, six-band 13D relation tokens, and symmetric multi-positive InfoNCE.
- `references.bib` — local-evidence-only bibliography. Every incomplete entry is visibly marked `BIB_METADATA_HOLD`; no missing author, title, venue, page, or URL was invented.

The method-figure area is an editable-figure replacement box that points to the repository source deck:

`../slides/PHASEPAIR_EDITABLE_PAPER_FIGURES_DATA_FREE_DRAFT.pptx`

The LaTeX source embeds no third-party or private image.

## Data-free build check

From this directory, the current technical skeleton can be checked with:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

The verified 2026-08-25 build produced four pages with no undefined citation or
reference and no overfull horizontal box. The generated PDF remains a
`NO_RESULT / NOT_FOR_SUBMISSION` engineering check, not a final paper artifact.

## Why this is not a final or currently submit-ready paper

The following gates are open:

- the [official ICASSP 2027 call, dates, and page boundary](../docs/PHASEPAIR_ICASSP_2027_OFFICIAL_REQUIREMENTS_20260825.md) have been verified, but no usable 2027 paper-kit link has yet been bound from the official author pages;
- author names, affiliations, acknowledgments, and submission metadata are absent;
- incomplete closest-work records remain `BIB_METADATA_HOLD`;
- the B0 annotation-caption technical lineage, required-code negative tests, strict archive boundaries, and public allowlist receipts are closed, but owner-license attestation, the one-archive topology owner signature/transfer disposition, and qualified public-release clearance remain human gates; consequently `training_authorized=false`;
- an exact private research CLIP snapshot has been assessed through repeated offline local-CPU preflight, and a stopped `AUTHORITY0` live text bridge exists; no target-server CUDA runtime, production runtime lease, redistribution right, or production text-initialization receipt exists;
- data-free trainer, checkpoint, job-planner, residual-head, caption-processing,
  sampler-manifest, validation-gallery, base-qualification, and
  statistical-reporting modules exist, but caption/token mappings still await
  a qualified execution adapter and no authorized live AdamW execution,
  energy-floor receipt, execution grant, private-data run, GPU run, reportable
  checkpoint, score matrix, or bootstrap output exists;
- the 9 base qualification runs, 18 residual runs, 21 score rows, H1--H6 tests, and Holm H2--H6 family have not run;
- every empirical table cell is deliberately rendered by `\ResultHold` as `HOLD--NOT_RUN`;
- the public data-free implementation milestone exists, while the final
  empirical release, paper-rights review, and submission authority remain open.

## Permitted use at this state

Read and revise prose, inspect equation-to-contract consistency, run the public
data-free/synthetic checks, repeat already-scoped authority-zero local preflight,
replace `BIB_METADATA_HOLD` entries only after independent metadata
verification, and edit the source PPTX. Do not replace result HOLDs, add
empirical claims, access private data, begin an optimizer/training session, use
a GPU, generate a final-looking PDF, or bind an official conference template
without the corresponding authority and evidence.

## Local evidence basis

This skeleton began from the frozen `20260824_165840` exact7 scientific family,
the historical `20260824_220209` data-free whole-core handoff, and the editable
no-result figure deck. Its current engineering evidence also includes the
[B0-v4 closure](../docs/PHASEPAIR_B0_ANNOTATION_CAPTION_CLOSURE_SUCCESSOR_PUBLIC_V4_20260825_145013.json),
[schema-corrected local CPU runtime receipt](../docs/PHASEPAIR_LOCAL_PY312_CPU_WHEELHOUSE_PUBLIC_SUCCESSOR_V3_20260825_020011.md),
and [live bridge report](../docs/PHASEPAIR_LIVE_CLIP_TEXT_BRIDGE_REPORT_20260825_031124.md),
plus the current data-free implementation and tests. These establish only
scoped engineering contracts; they are not evidence of authorized training, a
server run, or a scientific result. The repository-wide current state is
recorded in [STATUS.md](../STATUS.md).
