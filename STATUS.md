# PhaseSet project status

**Evidence snapshot:** 2026-08-25 (Asia/Shanghai)
**Research state:** `ACTIVE / AUTHORITY0 / NO_REAL_DATA_RESULT / NO_CLAIM`
**Release target:** `v0.2.0-alpha.1 / PUBLIC_DATA_FREE_MIGRATION`

PhaseSet is the active multi-person successor to PhasePair. The public source
is being migrated on `codex/phaseset-migration`; the immutable annotated
PhasePair `v0.1.0` tag remains the dyadic compatibility baseline. Implemented
code, passing synthetic tests, and frozen plans are software evidence, not
training results or scientific claims.

## Completed in this migration

- Full encrypted snapshots of the public repository and private research
  workspace, plus a verified all-refs bundle and SHA-256 receipts.
- Clean migration branch from exact public commit
  `123bf9d2017f09f11c812c6450f854595d2e29aa`.
- New `phaseset-core` distribution identity and `phaseset_core` namespace;
  legacy `phasepair_core` remains present.
- Frozen public and private ARIS research/experiment contracts for dynamic-K,
  permutation-invariant, streamed multi-person computation.
- Public/private boundary, migration policy, contribution policy, ownership,
  and security rules.
- Multi-person implementation, data adapters, runner, statistics, and tests
  are in active local qualification. Their final pass counts will be recorded
  only after the integrated suite finishes.

## Registered scientific census

- Native Embody 3D census: 572 eligible K>=3 captures, 69 participants,
  20.673 scene hours, 27 K=3 and 545 K=4.
- Participant-disjoint split: 400 train / 96 validation / 76 test captures.
  Private membership and participant identifiers are not public artifacts.
- All K=3 captures are sealed test cases. Training does not observe K=3.
- Base qualification: 3 architectures x 3 seeds = 9 attempts.
- Periodic residuals: 8 systems x 3 seeds = 24 attempts.
- Formal training census: 33 attempts; final score matrix: 9 systems x 3
  seeds.
- Test runs once after validation and aggregation code are frozen. Inference
  uses 100,000 paired capture-cluster bootstrap draws and Holm correction for
  H2-H8.

No real-data attempt, score row, confidence interval, corrected decision, or
model-quality claim currently exists.

## External gates

Embody 3D requires a real applicant to submit the official release form with
true identity, institution, and email. No automation may invent those facts.
Private download URLs and licensed assets stay outside Git.

The registered experiment server most recently refused the registered
non-default SSH port before authentication. The next attempt may probe only
that port, must verify the host key, and must begin with a read-only system/GPU
inventory. No default-port scan or substitute host is authorized by the
frozen contract.

The GitHub repository was renamed in place to
`appleweiping/phaseset-multiperson-motion-language`. Its numeric repository ID,
main ref, immutable v0.1.0 tag object, release, and latest successful Actions
run were unchanged. The old web URL returns a permanent redirect, and old/new
anonymous git and codeload endpoints resolve to the same refs and post-rename
archive bytes. The separate `periodic-motion-language` repository contains one
deprecation notice linking PhaseSet and is archived without history rewrite.

## Remaining completion gates

1. Finish integrated data-free model, data, runner, bootstrap, recovery, and
   release tests on the migration branch.
2. Commit and push independently reversible migration/core/runner milestones
   to the already-renamed public repository and preserve rename receipts.
3. Obtain legitimate Embody access and close private split/caption provenance.
4. Qualify the Linux/CUDA runtime, precision, canonical reductions, synthetic
   lifecycle, and disposable overfit.
5. Execute all 33 attempts with immutable terminals, then perform exactly one
   sealed-test evaluation.
6. Generate statistics, figures, tables, paper, and slides from one frozen
   aggregate; retain negative or inconclusive outcomes.
7. Publish safe releases through `v1.0.0` and verify a clean anonymous codeload
   can install, test, build the paper, and validate release hashes.

Until these gates have real evidence, PhaseSet is a data-free implementation
and execution-contract milestone, not a completed empirical paper.
