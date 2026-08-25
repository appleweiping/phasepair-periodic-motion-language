# PhasePair ICASSP 2027 official-requirements checkpoint

Verified: 2026-08-25 (Asia/Shanghai)
Target: regular conference paper, ICASSP 2027
Status: `OFFICIAL_CALL_VERIFIED / PAPER_KIT_LINK_NOT_YET_BOUND`

This checkpoint records only requirements visible on official IEEE ICASSP 2027
pages at the verification time. It does not substitute an older conference kit
for the eventual 2027 author package.

## Verified target and dates

- ICASSP 2027 is scheduled for 16--21 May 2027 in Toronto, Canada.
- The full-paper submission deadline is 16 September 2026.
- Acceptance notification is scheduled for 13 January 2027.
- The final-paper deadline is 27 January 2027.
- Author registration is due 10 February 2027.

Authoritative sources:

- [ICASSP 2027 Call for Papers](https://2027.ieeeicassp.org/call-for-papers/)
- [ICASSP 2027 official home page](https://2027.ieeeicassp.org/)

## Verified regular-paper length boundary

The regular conference route permits at most four pages of technical content,
including figures and references. The official call-for-papers PDF additionally
describes an optional fifth page restricted to references, funding
acknowledgements, and a Compliance with Ethical Standards statement. PhasePair
will therefore target four technical pages and use page five only for those
explicitly permitted elements.

Authoritative sources:

- [Publishing and paper-presentation options](https://2027.ieeeicassp.org/publishing-and-paper-presentation-options/)
- [Official ICASSP 2027 call-for-papers PDF](https://2027.ieeeicassp.org/wp-content/uploads/sites/13/2019/08/ICASSP2027-CallForPapers-draft-Jun7.pdf)

## Compliance and reproducibility implications

The official editorial-policy page states that submissions undergo template
compliance checks covering such items as paper length, structure, topics,
language, and author metadata, followed by plagiarism/self-plagiarism checks.
It also encourages associated code, data, and presentation artifacts for
reproducibility. PhasePair's public code, release manifest, result provenance,
and editable figures are therefore submission deliverables rather than optional
appendices.

Authoritative source:

- [ICASSP 2027 Editorial Policies](https://2027.ieeeicassp.org/about/editorial-policies/)

## Template status and binding rule

At this checkpoint, the official publishing-options page referred readers to
submission instructions and templates, but the page exposed no usable 2027
paper-kit link in its visible author section. Targeted searches of the official
2027 conference domain likewise did not identify a downloadable 2027 LaTeX or
Word paper kit. This is a time-scoped observation, not a claim that the kit will
remain unavailable.

Consequently:

1. `paper/main.tex` remains a content skeleton and must not be called an
   official-format submission.
2. No 2026 or generic IEEE package may be relabeled as the 2027 conference kit.
3. The official 2027 author page and kit must be rechecked before submission.
4. Once published, the kit will be downloaded from an official IEEE/ICASSP
   origin, hashed, recorded, and used to build and visually inspect the final
   PDF.
5. Author identities, affiliations, conflicts, acknowledgements, ethical
   statement, EDICS selection, and submission-system metadata remain separate
   final-owner inputs and may not be invented.

## PhasePair paper acceptance gate

The manuscript cannot leave `NOT_FOR_SUBMISSION` until all of the following are
true:

- the official 2027 paper kit is bound and its source identity recorded;
- the registered 9 base and 18 residual runs have terminal receipts;
- the 21 validation rows and registered H1--H6/Holm inference are frozen;
- every table, figure, and claim maps to released evidence;
- the PDF satisfies the four-plus-optional-reference-page rule;
- citation metadata and closest-work records have no unresolved holds;
- author/submission metadata is supplied by its owner;
- a clean public release and anonymous reproduction check have completed.
