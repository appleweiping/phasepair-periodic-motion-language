# PhaseSet project status

**Evidence snapshot:** 2026-09-08 (UTC)
**Research state:** `ACTIVE / AUTHORITY0 / NO_REAL_DATA_RESULT / NO_CLAIM`
**Published release:** `v0.2.1 / PUBLIC_DATA_FREE_STABLE_CORE_PORTABLE_TEST_HOTFIX`
**Final target:** `v1.0.0 / REAL_EXPERIMENTS_AND_PAPER / NOT_COMPLETE`

PhaseSet is the active multi-person successor to PhasePair. The in-place public
migration and `v0.2.0-alpha.1` pre-release are complete; the immutable
annotated PhasePair `v0.1.0` tag remains the dyadic compatibility baseline.
Implemented code, passing synthetic tests, and frozen plans are software
evidence, not training results or scientific claims.

## Server continuation

- Strict known-host SSH login to the registered host now succeeds. The initial
  read-only inventory is complete. Existing workloads were not interrupted.
- A separate data-volume Python 3.12.12 / NumPy 2.4.6 / Torch 2.12.0+cu126
  environment is installed with a clean dependency check. The complete Linux
  CPU regression including capture-level training selection passed:
  1012 tests, 2 skipped, and 39 subtests, in 286.46 s. Its focused integration
  passed 84 tests and 6 subtests. Source bytes were unchanged before and after
  both suites. The prior prepared-data v2 host integration passed 94 focused /
  964 complete tests. The prior residual-resume integration
  passed 948 tests and 39 subtests.
  The earlier frozen-CLIP integration passed 932 tests and 39 subtests.
  The earlier host-only suite passed 917 tests and its 25 focused checks.
  This includes synthetic model/lifecycle checks, not main-data training.
- A separately bounded shared-GPU functionality probe passed FP64 forward and
  FP32 forward/backward using 8x8 tensors. It did not stop existing workloads
  and does not qualify BF16, model execution, performance, or the full runtime.
- A subsequent actual-width PhaseSet periodic-core CUDA observation passed
  with 512-dimensional tokens and a 256-dimensional hidden layer: K2/K3
  forward/backward, actor permutations, K2 topology output/gradients as exact
  positive zero, and K13 (78 edges) bitwise equality at chunks 64/128/256.
  Maximum observed CPU/CUDA token difference was 1.431e-6 within the frozen
  FP32 tolerance. The engineering fixture peaked at 80,722,944 allocated bytes
  under a 2 GiB allocator cap; this is not a performance benchmark. Its first
  attempt stopped before model construction because physical FB total and
  CUDA context-usable total were incorrectly equated; that failed receipt is
  retained. The corrected memory instrumentation did not relax a model gate.
  This does not qualify BF16, the learned group base, or the full training runtime.
- Separate actual-width B0/B1/B2 FP32 CUDA forward/backward lanes passed the
  registered parameter census, physical permutation/padding equality, finite
  gradients, frozen-text preservation, and CPU/CUDA tolerance checks. Their
  small analytic fixtures are not training or performance results. A subsequent
  residual retrieval lane failed its frozen-base CPU/CUDA embedding comparison
  after zero-residual scoring checks. That failed receipt is preserved. A
  same-weight diagnostic isolated an observed native-MHA-fastpath dependency.
  The frozen runtime now disables that fastpath on CPU and CUDA and restores
  caller flags. With unchanged tolerance, the new CUDA retrieval-head seam
  passed: frozen-base maximum difference 7.153e-7, score difference 4.471e-8,
  and finite nonzero exercised head gradients. This did not run the periodic
  core or an optimizer. A subsequent B0 repeat stopped before model creation
  for insufficient free memory. After memory recovered, a distinct retry chain
  passed retrieval/B0/B1/B2 with the same reviewed source and tolerance; the
  original resource failure remains preserved. Full-training and BF16
  qualification remain pending. See [runtime policy](docs/FROZEN_MHA_RUNTIME.md).
- The native Embody loader and concrete licensed SMPL-X body-22 evaluator are
  implemented. Seven adapter contract tests passed on Linux/Python 3.12.12/
  NumPy 2.4.6; these use explicit fixtures, not licensed assets or main data.
- The immutable prepared-data v2 writer/reader now connects canonical body-22
  windows and frozen variable-count text features to the private-host factory.
  It verifies the exact consumed index/manifest/payload bytes, rejects split
  overlap, and applies reproducible shared-group yaw from original bytes each
  training epoch. Validation is unrotated. Window-positive families support the
  auxiliary task; the private-host holistic capture source is not yet wired.
  Actor-set commitments must not be used as unique capture/window row IDs.
  See [prepared-data contract](docs/PREPARED_DATA_V2.md).
- The complete-window capture evaluator now pools unnormalized embeddings
  before one full-gallery scoring call and preserves exact capture-macro R@1
  fractions. The training API uses this concrete val-only source for checkpoint
  selection and binds its full census on resume. The standalone caller passed
  42 focused / 997 complete server tests; the training integration passed 54
  focused tests including actual optimizer/checkpoint/resume equality, followed
  by complete 1002- and 1012-test regressions and actual CLIP requalifications. Two
  legacy fixture failures from the first full attempt are retained and explained
  in [test stability](docs/LEGACY_TEST_STABILITY.md). These are analytic lifecycle
  checks, not formal runs. Actual scored-census consistency and scoped live
  capture-function drift checks were added after provisional composition review.
  Private-host disk composition
  and licensed source/caption provenance remain incomplete. See
  [capture validation](docs/CAPTURE_VALIDATION.md).
- Neither Embody approval nor a licensed neutral model has been observed.
  Open-license Multi-TPC acquisition completed with its official byte count and
  MD5 verified. Its archive inventory has 322 files. The bounded audit confirms
  19 Euler rotation triplets plus three translation anchors, not body22 joint
  positions, and unresolved modality-alignment/group-identity details. It is
  supplementary conversation data, never replacement
  confirmatory data. The pinned CLIP snapshot's eight files also match the
  previously acquired official SHA-256 values. Actual frozen CLIP text CPU
  loading and two public-caption forwards passed, with bitwise repeat outputs
  and unchanged RNG. The production-shaped frozen text adapter subsequently
  passed 15 server tests and three real-model forwards: the unchanged repeat
  golden plus a separate 226-to-77-token long-caption observation. It verifies
  the exact retained prefix and EOS, restores caller RNG, and emits text-free
  receipts. See [frozen CLIP text adapter](docs/FROZEN_CLIP_TEXT_ADAPTER.md).
  This does not qualify GPU text execution or real-data training.
  The MHA-policy successor separately passed 33 focused/949 complete server
  tests and three official CLIP forwards, preserving the original output
  golden and RNG under correctly changed code/runtime/receipt identities.
  The capture-training successor also passed three official CLIP forwards with
  both prior output digests and RNG unchanged under its new source identity.
- The earlier text-adapter commit was downloaded from anonymous codeload,
  manifest/content/link checked, built offline into a wheel, and installed into
  a separate server target. All 56 wheel Python modules match the public Git
  index. Three real CLIP forwards from that installed wheel produced a complete
  observation byte-identical to the previously verified source copy.
- Native Linux last-bit Morlet formula drift is repaired using strictly
  verified canonical coefficient bytes, while preserving both original hashes.
  Windows still requires its original exact formula digest; Linux requires its
  measured exact native formula fingerprint plus a bounded tap comparison.
  See [Morlet portability](docs/MORLET_PORTABILITY.md). POSIX checkpoint cleanup
  also now pins its inode so a concurrently replaced file is not deleted.
- The current no-result manuscript now builds using the hash-pinned official
  ICASSP 2027 template: five pages, technical content through page four,
  references-only page five, all fonts embedded, no unresolved citations or
  overfull boxes. All result cells and pending author metadata remain explicit.
- The concrete private-host bridge now connects base/residual training to
  live checkpoint receipts, a periodic heartbeat, and a whole-attempt OS lease.
  Base and residual resume verify the predecessor checkpoint chain, including
  process-loss recovery and authenticated latest/older-best materialization.
  Fifty-one focused server tests passed, including actual tiny residual
  optimization/checkpoint/resume equality and zero execution on invalid prior
  checkpoint bytes. See [resume details](docs/RESIDUAL_RESUME.md).
  The remaining preparation/qualification/evaluation host commands are not yet wired.
  Four actual tiny server fits verified observer behavior without changing
  checkpoint/model/loss/validation outputs. These are engineering fixtures,
  not formal training attempts. GitHub CI passed both Windows runtimes and the
  Ubuntu public-tree/wheel audit for the integration commit.

This continuation is not a completed scientific release. Private receipts,
data, model assets, connection details, and download material are not public.

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
- Executable capacity-matched systems 00--08. System 01 is strictly
  motion/Morlet-independent fixed band-ID input; system 03 uses only symmetric
  endpoint self-power statistics and requires bilateral support, so it has no
  cross-endpoint phase or lag. Marginal-power, pair-only, coverage-only,
  incidence-shuffled, phase-stripped, and full paths are also implemented.
- Licensed-format-independent numeric capture preparation with body-22
  extraction, anti-aliased 30-to-20 Hz conversion, shared group transforms,
  exact masks, dynamic padding, salted lineage, and all-edge resource limits.
- Deterministic base/residual optimization, variable-positive symmetric
  InfoNCE, frozen-base residuals, edge-budget gradient caching, validation-only
  selection, immutable checkpoints, exact resume tests, and strict loading of
  each winning seed checkpoint with its frozen base logit-scale.
- A complete nine-row base-qualification receipt schema and closed validator:
  exact score fractions,
  parameter counts, integer per-run latency, nine terminal digests, nine
  validation-selected checkpoint digests, validation-manifest/query-census/
  evaluator digests, nine unique score-artifact digests, and three seed-ordered
  winner checkpoint digests. The production bridge recomputes the canonical
  artifact and requires a separately authenticated nine-row authorization.
  The resource tie-break uses the sorted middle of each
  base's three integer latency measurements. Every formal loader requires an
  externally trusted expected qualification digest; residual construction
  additionally requires the externally trusted complete-cohort capacity-audit
  digest before constructing any system. A checkpoint proves best-at-write;
  final-run best status remains bound to the immutable terminal/qualification
  evidence.
- Default 256-edge runtime chunks with fixed 64-edge canonical microblocks;
  system 06 uses a group-global, per-band half-edge bijection backed by only
  `O(E)` integer routing metadata, while B2 never materializes a complete
  `K x K` attention-score map.
- A frozen numerical runtime with deterministic-algorithm errors, highest
  float32 matmul precision, TF32/reduced-precision reductions disabled,
  deterministic cuDNN, a mandatory deterministic CUDA workspace setting, and
  disabled native MHA fastpath on CPU/CUDA,
  an environment-v2 hardware/runtime/build/thread inventory. Live drift is
  rejected and ambient host flags are restored on exit.
- A differentiable small-batch Torch CPU skeleton/Morlet oracle for input-
  gradient permutation qualification. Production periodic preparation retains
  an explicit non-differentiable NumPy descriptor-stream boundary.
- Digest-only auxiliary caption fusion with a frozen backend manifest, a
  trusted-host verify-and-consume gate for test admission, and an explicitly
  provisional same-family semantic-review status.
- Digest-only host-injected production adapter. The public CLI remains
  authority zero unless an authenticated private host injects that adapter.
  Qualification, evaluation, bootstrap, and render-paper completions are
  semantically recomputed. Rendering re-verifies the sealed-test ledger, exact
  27-row score census, checkpoint/terminal/environment-bound resource census,
  deterministic claim ladder, and exact publication artifact set.

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
  independently evaluates all 27 system-by-seed score tables. It never averages
  logits across seeds. Each capture's three paired seed effects are averaged in
  fixed seed order; 100,000 bootstrap draws resample captures only, never seeds.
  H1--H8 share one index stream and Holm correction applies only to H2--H8.

No real-data attempt, score row, confidence interval, corrected decision, or
model-quality claim currently exists.

## External gates

Embody 3D requires a real applicant to submit the official release form with
true identity, institution, and email. No automation may invent those facts.
Private download URLs and licensed assets stay outside Git.

The earlier connection-refused state was superseded by successful strict
known-host login on 2026-09-08 UTC. Only the registered port and host remain
in scope. Resource availability, numerical qualification, and actual data
access are separate observed requirements; SSH success alone is not a passed
training environment.

The GitHub repository was renamed in place to
`appleweiping/phaseset-multiperson-motion-language`. Its numeric repository ID,
main ref, immutable v0.1.0 tag object, release, and latest successful Actions
run were unchanged. The old web URL returns a permanent redirect, and old/new
anonymous git and codeload endpoints resolve to the same refs and post-rename
archive bytes. The separate `periodic-motion-language` repository contains one
deprecation notice linking PhaseSet and is archived without history rewrite.

## Release provenance boundary

This source tree deliberately does not self-attest whether its declared
version has been published. A `v0.2.1` release exists only when the external
annotated tag, GitHub release, Actions results, repository identity, complete
asset inventory, and anonymous codeload receipts agree on the same commit and
tree while the immutable `v0.1.0` object remains unchanged. A branch name,
package version, changelog date, or statement inside the candidate tree is not
such evidence.

## Remaining empirical completion gates

1. Obtain legitimate Embody access and close private split/caption provenance.
2. Qualify the Linux/CUDA runtime, precision, canonical reductions, synthetic
   lifecycle, and disposable overfit.
3. Execute all 33 attempts with immutable terminals, then perform exactly one
   sealed-test evaluation.
4. Generate statistics, figures, tables, paper, and slides from one frozen
   aggregate; retain negative or inconclusive outcomes.
5. Publish the safe `v1.0.0` empirical release and verify a clean anonymous
   codeload can install, test, build the paper, and validate release hashes.

Until these gates have real evidence, PhaseSet is a data-free implementation
and execution-contract milestone, not a completed empirical paper.
