# PhasePair public data-free release manifest

**Manifest snapshot:** 2026-08-25 (Asia/Shanghai)

**Scope:** sanitized, data-free public implementation milestone

**Scientific status:** `AUTHORITY0 / NO_SERVER_RESULT / NO_CLAIM`

This document inventories the eligible public tree. It is not a dataset-rights
grant, training receipt, empirical result, final-paper release, or submission
authorization.

## Repository material

- `README.md`, `STATUS.md`, `CHANGELOG.md`, `CITATION.cff`, and `LICENSE`;
- `pyproject.toml` with Python 3.12--3.14 package metadata and separate exact
  3.14.5 signal-oracle dependencies;
- `.github/workflows/ci.yml` for 3.12.0 and 3.14.5 data-free CI;
- `scripts/release_tree.py`, `scripts/public_release_audit.py`, and
  `scripts/check_local_links.py` for one authority-consistent view of paths,
  bytes, forbidden content, and repository-local link targets;
- `scripts/release_manifest.py` plus `RELEASE_FILES.sha256` for deterministic
  Git-index blob hashes and byte counts, independent of checkout line-ending
  conversion, and exact file-set/hash verification in GitHub codeload archives;
- `src/phasepair_core/` and matching synthetic/data-free tests;
- sanitized specifications and aggregate no-result evidence under `docs/`;
- the anonymous no-result manuscript skeleton under `paper/`;
- the sanitized editable no-result figure deck under `slides/`.

## Implemented source families

The source tree contains:

- contracts, lineage, batching, deterministic sampler/dropout, and runtime
  readiness diagnostics;
- fixed signal descriptors, initialization, MIME/early/late motion towers,
  symmetric objectives, retrieval evaluation, and paired bootstrap primitives;
- CLIP resolution and text-runtime contracts plus a stopped authority-zero
  live bridge;
- execution schemas, exact-identity registries, trainer transactions,
  checkpoint lifecycle validation, and deterministic job planning;
- the generic, WaMo-style, InterEdit-style, no-relation, phase-stripped, and
  PhasePair residual-head family;
- strict caption parsing/normalization, sampler caption ordinals, complete
  30-epoch sampler manifests, and motion/caption training-batch cross-binding;
- full-gallery validation selection, base-system qualification, and registered
  statistical-report rendering.

Caption/token arrays remain explicitly unverified caller assertions until the
production tokenizer execution adapter binds the original payloads to an exact
tokenizer invocation. The public code fails closed on that boundary.

## Verification scope

- Windows x86-64, CPython 3.14.5 public tree: `571 passed, 3 skipped, 36
  subtests passed`.
- Windows x86-64, CPython 3.12.0 public tree: `570 passed, 3 skipped, 1
  deselected, 36 subtests passed`; the deselection is the exact 3.14.5
  signal-runtime identity test.
- Ubuntu portability lane: public audit, checksum manifest, local links, Ruff,
  and no-dependency wheel build; it does not qualify the fixed signal oracle.
- Ruff: clean across `src`, `tests`, and `scripts`.
- Release-tool regression tests reject parent-repository attachment, empty
  passes, archive tampering/extra files, and Git-index/worktree divergence.
- The no-result manuscript builds to four pages with BibTeX, no undefined
  citation/reference, and no overfull horizontal box.
- Independent caption/training/sampler review: PASS for bare-CR handling,
  receipt privacy, exact-type issuance, rebind/forgery rejection, ordinal
  binding, and cross-epoch/cross-seed replay rejection.

These checks establish scoped software behavior only. They do not establish
data rights, server compatibility, model quality, statistical significance, or
paper claims.

## Intentionally excluded

The public release must not contain:

- third-party dataset samples, captions, identifiers, split membership, or
  reconstructable private commitments;
- CLIP weights, runtime wheel archives, or other restricted third-party bytes;
- credentials, access endpoints, usernames, keys, tokens, or private paths;
- private receipts, raw logs, checkpoints, unpublished results, or per-sample
  outputs;
- local agent state, ARIS private traces, debug/compile products, caches, or
  licensed reference-paper PDFs.

## Not yet produced

- qualified original-caption-to-tokenizer execution-adapter receipt;
- target-server runtime receipt and the authorized 9 base + 18 residual runs;
- 21-row score matrix, registered bootstrap output, H1--H6 decisions, and Holm
  correction evidence;
- final empirical figures, quantitative tables, official-format manuscript,
  final presentation, and conference submission bundle;
- final empirical release tag and post-result anonymous-clone audit.

The tracked-file checksum manifest certifies only the bytes in this data-free
snapshot. It must be regenerated and reverified whenever any tracked file
changes.
