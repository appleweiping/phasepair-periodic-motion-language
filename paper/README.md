# PhaseSet NO_RESULT technical draft

**Title:** *PhaseSet: Permutation-Invariant Periodic Relation Tokens for
Multi-Person Motion–Language Retrieval*
**State:** `DRAFT / AUTHORITY0 / NO_RESULT / NOT_FOR_SUBMISSION`

This directory contains a compact four-content-page-target manuscript plus a
references-only fifth-page target. It is a protocol-bearing technical draft,
not evidence of a completed experiment or an authorized conference submission.

## Files

- `main.tex` — PhaseSet multi-person method, frozen 572-capture protocol,
  400/96/76 participant-component split, unseen-cardinality `K=3` test,
  33-attempt ledger, H1–H8 family, one sealed test, and 100,000-draw analysis.
- `references.bib` — primary-source-checked bibliography; no placeholder author,
  title, venue, or result metadata.
- `../figures/phaseset_motivation.pdf` (PNG twin available) — data-free
  pair-bag/incidence counterexample.
- `../figures/phaseset_architecture.pdf` (PNG twin available) — data-free
  PhaseSet method diagram.
- `../docs/PHASESET_LITERATURE_REVIEW.md` — 72-page local-PDF audit and
  primary-source review of adjacent multi-person and periodic-motion work.

Every cell in the manuscript result matrix is visibly `HOLD`. Do not replace a
cell until the rights/runtime/qualification/validation-freeze chain permits the
single sealed-test execution and frozen aggregate.

## Template status

The [official ICASSP 2027 publishing page](https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/)
confirms four technical pages plus an optional references-only fifth page, but
its current author section does not expose a downloadable 2027 paper kit in a
form available to this repository. The source therefore deliberately retains
the engineering `article` template and labels that fact in `main.tex`. Page
count in this template is only a drafting constraint. Re-template and re-audit
page limits, fonts, figures, bibliography, and anonymization against the
official kit before any submission claim. The official full-paper deadline is
[September 16, 2026](https://2027.ieeeicassp.org/call-for-papers/).

## Local build

From this directory:

```text
pdflatex -interaction=nonstopmode -halt-on-error main.tex
bibtex main
pdflatex -interaction=nonstopmode -halt-on-error main.tex
pdflatex -interaction=nonstopmode -halt-on-error main.tex
```

Verified locally on 2026-08-25 with MiKTeX pdfTeX/BibTeX: the build succeeds as
a five-page PDF, pages 1–4 contain manuscript content, and page 5 contains only
references. The final LaTeX log has no undefined citation/reference, multiply
defined label, overfull box, underfull box, or package warning. The command-line
tools emit the environment-level advisory “MiKTeX updates have not been
checked”; this is a toolchain-maintenance notice, not a manuscript warning.
A generated PDF remains a `NO_RESULT / NOT_FOR_SUBMISSION` engineering artifact.

## Non-negotiable language

- The official Embody 3D scale and the project’s 572 eligible captures are
  different quantities.
- All 27 `K=3` captures are test-only; never imply that training saw `K=3`.
- Multi-TPC is cross-domain, M3Act3D/AIOZ-GDANCE are probes, and dyadic corpora
  are backward-transfer checks—not confirmatory primary data.
- No in-house base is an official MIME, WaMo, InterEdit, or other reproduction.
- No “first,” improvement, generalization, topology, or phase-effectiveness
  statement is authorized by this draft.
