# PhasePair project status

**Evidence snapshot:** 2026-08-25 (Asia/Shanghai)

**Research state:** `ACTIVE / AUTHORITY0 / NO_SERVER_RESULT / NO_CLAIM`

**Release state:** `PUBLIC_DATA_FREE_CORE / EMPIRICAL_RELEASE_HOLD`

This page separates implemented code from experiments that have actually run.
A module, test, receipt, or plan is not a scientific result and does not grant
data, optimizer, GPU, publication, or submission authority.

## Current implementation

The public source tree contains:

- strict lineage, batching, deterministic dropout, initialization, motion
  towers, signal descriptors, objectives, evaluation, and bootstrap inference;
- CLIP snapshot assessment and a stopped, offline, authority-zero live text
  bridge that does not redistribute model bytes;
- exact-identity registries, execution schemas, trainer transactions,
  checkpoint lifecycle validation, and deterministic job planning;
- the registered generic/WaMo-style/InterEdit-style/no-relation/
  phase-stripped/PhasePair residual-head family;
- strict three-caption parsing and normalization, sampler caption ordinals, a
  complete 30-epoch sampler manifest, and motion/caption batch cross-binding;
- full-gallery validation selection, base-system qualification, and registered
  statistical-reporting primitives;
- synthetic/data-free tests, a no-result paper skeleton, and an editable
  no-result figure deck.

The caption batch receipt deliberately says
`tokenizer_mapping_verified=false`. Current arrays are caller-frozen assertions,
not proof that one exact tokenizer execution consumed the original caption
payloads. A production execution adapter must close that boundary before any
training admission.

## Verified software evidence

| Scope | Result | Boundary |
|---|---|---|
| Public tree, CPython 3.14.5 | `568 passed, 3 skipped, 36 subtests passed`; Ruff clean | Synthetic/data-free behavior only. |
| Public tree, CPython 3.12.0 | `567 passed, 3 skipped, 1 deselected, 36 subtests passed` | The deselected test intentionally asserts the exact 3.14.5 signal-oracle runtime. |
| Wider private engineering workspace | `679 passed, 22 skipped, 36 subtests passed`; Ruff clean | Includes non-public platform/process tests and is not a substitute for public-tree QA. |
| Caption/training/sampler independent review | PASS on bare-CR rejection, receipt privacy, rebind/forgery rejection, ordinal binding, and cross-epoch/cross-seed replay rejection | Still authority zero; the real tokenizer adapter remains open. |

The Windows skips are limited to symlink privilege and the explicitly private
CLIP-runtime lane. The wider private-suite skips additionally cover POSIX-only,
Linux syscall, CUDA, or private-runtime fixtures.

## Experiment state

The registered experiment census remains unexecuted:

- base qualification: 3 systems × 3 seeds = 9 runs;
- residual comparison: 6 systems × 3 seeds = 18 runs;
- final validation table: 7 systems × 3 seeds = 21 score rows;
- paired source-cluster bootstrap, H1--H6, and Holm correction for H2--H6.

There is no authorized server training run, GPU result, score matrix,
confidence interval, corrected p-value, final quantitative table, or
evidence-backed scientific claim in this repository.

## Public/private boundary

The public repository may contain sanitized source, tests, specifications,
aggregate no-result evidence, manuscript source, and editable figures. It must
not contain dataset samples or captions, identifiers or split membership,
private commitments, endpoints or credentials, model weights, runtime wheels,
private receipts, raw logs, checkpoints, or per-sample outputs.

`RELEASE_FILES.sha256` binds the tracked public snapshot while
`scripts/public_release_audit.py` rejects forbidden paths, local locators,
unexpected binaries, common credential patterns, and endpoint-shaped strings.
Neither mechanism certifies scientific correctness or third-party rights.

## Remaining completion gates

1. Implement and independently qualify the original-caption-to-tokenizer
   execution adapter without weakening the fail-closed boundary.
2. Close the remaining owner-license, topology-signature/transfer, and
   qualified-release decisions for the private dataset lineage.
3. Establish the target-server CUDA/runtime receipt and compatible production
   environment.
4. Execute the registered 9 base and 18 residual runs with complete receipts.
5. Produce the 21-row validation matrix, paired bootstrap output, H1--H6
   decisions, and Holm-corrected H2--H6 family.
6. Convert only verified results into final figures, tables, paper, and slides.
7. Re-run release, rights, anonymous-clone, and manuscript/submission audits on
   the final empirical artifact.

Until those gates are evidenced, PhasePair is a published data-free
implementation milestone, not a completed empirical paper or submission.
