# Changelog

All notable public changes to PhaseSet and its PhasePair ancestor are recorded
here. The source version may precede its external publication: only the
immutable annotated tag and corresponding GitHub release establish a released
version. No empirical result or final paper release has been issued.

## Unreleased

## 0.2.0 - 2026-08-26

### Stable data-free multi-person core

- Added executable, capacity-matched PhaseSet systems 00--08 rather than
  configuration-only control rows.
- Added a licensed-format-independent body-22 preparation pipeline with salted
  actor lineage, anti-aliased 30-to-20 Hz conversion, synchronized group yaw,
  dynamic padding, and explicit all-edge resource limits.
- Added deterministic base and frozen-base residual training with
  variable-positive symmetric InfoNCE, edge-budget gradient caching,
  validation-only checkpoint selection, atomic checkpoints, and exact resume.
- Bound base qualification to all nine validation rows and all nine selected
  checkpoint digests. Resource ties use the sorted middle of the three integer
  seed latencies; residual construction strictly reloads the matching winner
  checkpoint and freezes its group encoder and learned base logit scale.
  Formal loaders require an externally trusted qualification digest, and the
  residual cohort also requires an externally trusted capacity-audit digest;
  local self-hashing cannot substitute for either trust anchor.
- Made system 01 a genuinely motion-independent fixed six-band-ID control,
  with no activity, Morlet-support, or energy path. Made system 03 a bilateral
  endpoint-self-power control with symmetric marginal features and no
  cross-endpoint product, phase, or lag.
- Added the exact frozen-base-plus-bounded-periodic score path, with the base
  cosine score detached and the learned periodic residual scaled by `tanh`.
- Made the production system-06 incidence shuffle a group-global, per-band
  bijection over valid directed half-edge slots using `O(E)` integer metadata
  without retaining `O(E*D)` activations.
- Streamed B2 social attention without a complete `K x K` score map. Periodic
  execution defaults to 256-edge runtime chunks while retaining canonical
  64-edge reduction microblocks.
- Froze formal numerical execution to deterministic-algorithm errors, highest
  float32 matmul precision, disabled TF32/reduced-precision reductions,
  deterministic cuDNN, and a deterministic CUDA workspace configuration. The
  environment-v2 digest now binds hardware, runtime, build, threads, and that
  policy; live drift fails closed and ambient flags are restored afterward.
- Added a differentiable Torch CPU skeleton/Morlet oracle for input-gradient
  qualification while preserving the explicit non-differentiable NumPy
  descriptor boundary in the production periodic path.
- Added digest-only auxiliary caption fusion with a frozen model/backend/runtime
  manifest, trusted-host verify-and-consume test admission, three physical
  actor-order calls, and provisional rather than gate-closing semantic review.
- Added a digest-only production runtime adapter while preserving the public
  CLI's authority-zero, fail-closed default.
- Added a fail-closed wheel verifier that compares every packaged Python module
  with the staged Git-index blob and binds wheel version/entry points to staged
  project metadata. Added a canonical post-release asset receipt whose verifier
  cross-checks trusted repository/tag expectations, independent GitHub REST
  responses, annotated tag/commit/tree identity, and fresh downloaded bytes.
- Generalized preprocessing to the public `K>=2` contract; the registered
  Embody confirmatory task independently remains native `K>=3` only.

No real-data training result or scientific performance claim is part of
`v0.2.0`.

### PhaseSet migration

- Renamed the active project from dyadic PhasePair to multi-person PhaseSet
  while preserving the immutable PhasePair `v0.1.0` tag and namespace.
- Added the `phaseset-core` distribution metadata, `phaseset_core` namespace,
  migration boundary, contribution policy, security policy, and ownership
  rules.
- Replaced the old two-person scientific target with a dynamically padded,
  permutation-invariant, streamed `O(K^2)` group relation contract.
- Registered a true `K>=3` Embody 3D protocol, 33-run experiment census,
  fail-closed runner surface, and no-result ICASSP manuscript path.

### Added

- Data-free PhasePair batching, signal, motion-tower, objective, evaluation,
  bootstrap, initialization, dropout, lineage, and execution-schema modules.
- Stopped authority-zero CLIP resolution/text-runtime bridge with public-safe
  offline runtime receipts; no model weights are redistributed.
- Data-free trainer transaction, checkpoint lifecycle, and deterministic
  9-base-plus-24-residual job-plan primitives.
- Registered residual heads, strict three-caption processing, complete
  30-epoch sampler manifests, motion/caption training-batch cross-binding,
  full-gallery validation selection, base qualification, and statistical
  reporting primitives.
- Windows Python 3.12/3.14 exact-runtime CI, an Ubuntu portability audit/wheel
  lane, a fail-closed public-tree audit, and a deterministic tracked-file
  checksum manifest.
- Authority-consistent release tooling that audits Git-index blobs in a
  checkout and the complete manifest-bound file set in a GitHub codeload ZIP,
  with regression coverage for tampering, extra files, parent-repository
  attachment, and unstaged worktree masking.
- B0 annotation/caption technical closure receipts with the remaining human
  license/release decisions stated explicitly.
- No-result paper source, editable method/protocol slide source, project status,
  release-candidate manifest, and software citation metadata.

### Pending for v1.0.0

- Production original-caption-to-tokenizer execution-adapter qualification.
- Target-server runtime qualification and the registered 9 + 24 experiments.
- Real-data aggregate statistics, empirical table cells, final ICASSP author-kit
  validation, and the `v1.0.0` GitHub release with anonymous-clone verification.

### Release boundary

- Restricted datasets, captions, identifiers, pretrained weights, private
  receipts, credentials, server locators, checkpoints, and raw per-sample
  outputs are not part of the public repository.
