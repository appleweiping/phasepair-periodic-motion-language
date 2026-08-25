# Contributing

PhaseSet accepts narrowly scoped, reviewable changes through topic branches.

1. Create a topic branch; do not force-push shared history or move release tags.
2. Keep `phasepair_core` legacy behavior stable unless the change fixes a
   demonstrated defect with a regression test.
3. Add deterministic tests for group permutation, padding, masks, chunking,
   and fail-closed authority boundaries affected by the change.
4. Run `pytest`, Ruff, local-link checking, and the public-release audit.
5. Never commit third-party data, captions, identifiers, weights, endpoints,
   credentials, private receipts, raw results, or reference-paper PDFs.

Semantic model review is advisory/provisional. Merge gates rely on code review
plus deterministic tests and public-release checks.
