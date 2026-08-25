# PhasePair experiment-bridge whole-core handoff

**Status:** `WHOLE_CORE_DATA_FREE_SUBSET_CLEAN / EXPERIMENT_BRIDGE_RUNNING / AUTHORITY0 / NO_DATA / NO_GPU / NO_RESULT / NO_CLAIM`

**Historical scope:** timestamped implementation checkpoint retained for provenance; later repository records supersede its live gate status.

**Current status pointer:** use the repository [STATUS](../STATUS.md) for the
current implementation, experiment, and publication state. The HOLD list below
is preserved as a historical observation and must not be read as a current
inventory.

## Outcome

The local data-free PhasePair subset now includes contract-shaped Torch motion
towers, external runtime dropout, deterministic motion initialization,
batching, objectives, and sampler logic in addition to the earlier numerical
and audit primitives. The exact 27-file set passed author and fresh local QA;
its ledger SHA-256 is
`bebecced16ed2c10d8d05481b47ef81660d5b82a89aa1c9cf077da4980b85d9a`.
Exact identities and scoped evidence are recorded in the companion
[historical code review](./EXPERIMENT_CODE_REVIEW_20260824_220209.md).

This is an implementation checkpoint, not an experiment result or an
end-to-end executable training system. The frozen `20260824_165840` contract
files retain their author-time statements and bytes; this later report does
not promote their fixed aliases or grant execution authority.

## Implemented locally

| Component | Implemented data-free scope |
|---|---|
| Package and lineage | Authority-zero package boundary, source-lineage messages, actor/pair commitments, uniqueness, canonical manifests, and crop validation. |
| Signal | Pure-DCT control, exact NumPy Morlet oracle, relation algebra, swap invariants, masks, and diagnostic descriptor math. The canonical descriptor still stops before an absent energy-floor receipt. |
| Batching | Owned strict float32 snapshots for 262D actor streams and the 799D relation stream, with exact ordered `[AB, BA]` pass construction. |
| CLIP preflight | Eight-file pin contract, root/census/file stability, hashing, wheel/runtime/rights evidence schemas, canonical assessment, and sealed lease behavior. No actual snapshot was resolved; even synthetic complete evidence remains `HOLD_FRESH_REVIEW_REQUIRED`. |
| Static contracts | MIME/early/late inventories, planned optimizer partitions, frozen-row exclusion rules, and fail-closed resolved-text requirements. These are not a live optimizer receipt. |
| Torch motion towers | Contract-shaped MIME, TMR-style early-fusion, and TMR-style late-fusion motion encoders producing 512D motion embeddings from prepared passes. No text tower is connected. |
| Runtime dropout | External deterministic dropout masks, exact site inventories (`49/16/32`), immutable bundle identity, and same-thread single-use lease semantics. |
| Motion initializer | Closed model/parameter inventory checks, deterministic raw parameter derivation, transactional commit, state hash, and receipt verification. Public status remains `HOLD_RUNTIME_LIBM_ORACLE_UNPINNED_AUTHORITY0`. |
| Objectives | Ordered-pair score averaging and symmetric multi-positive InfoNCE over validated normalized 512D synthetic tensors. |
| Sampler | Deterministic 30-epoch anchor, caption, hardness-window, and batch planning over caller-supplied immutable synthetic inputs. |
| Evaluation | Full-gallery T2M/M2T ranks and R@1/3/5/10/MedR with exact tie ordering. |
| Bootstrap | Nested source-cluster effects, canonical immutable 100,000-draw counter matrix, bootstrap-t, and percentile interval arithmetic. |
| Readiness | Canonical authority-zero diagnostic JSON that cannot treat opaque hashes as authorization. |

The whole-core verification used CPython `3.14.5`, NumPy `2.4.6`, and
PyTorch `2.12.0+cpu` with CUDA build `None`. Author discovery reported
`259 passed, 1 skipped, 36 subtests passed in 251.39s`; fresh discovery
reported `259 passed, 1 skipped, 36 subtests passed in 232.42s`. Ruff was
clean in both reviews. These synthetic tests establish only the scoped local
contracts exercised by the tests.

## Hard HOLDs before any real run

1. **Actual CLIP package and runtime:** no pinned snapshot bytes, tokenizer,
   offline reload, exact compatible wheels, model-rights disposition, or live
   fresh assessment lease has been produced.
2. **Resolved text runtime bridge:** no sealed binding joins a live CLIP
   assessment to the resolved trainable text rows, frozen projection row,
   exact `nn.Parameter` identities, state keys, checkpoint identity, and active
   text-dropout-zero setting.
3. **Optimizer and training execution:** no real AdamW instance, schedule,
   training-step bridge, training loop, checkpoint writer, resume path, or
   step-0/first-step optimizer receipt exists. AdamW must not be constructed
   from motion parameters alone while the text bridge is absent.
4. **Data, rights, and private lineage:** no authorized InterHuman rows or
   captions have been opened by this implementation, and the final owner,
   rights, server-copy, source-lineage, pair, caption, crop, and allowlist
   receipts remain absent.
5. **Training energy floors:** the six-band training-final energy-floor values
   and their allowlist/adapter/crop/seed-bound receipt are not available. The
   canonical PhasePair descriptor therefore remains held.
6. **Runtime locks:** the initializer's exact runtime/libm oracle is not pinned,
   and the local CPU QA runtime is not the target Torch/CUDA/CLIP training
   environment. The known remote runtime is an older, mismatched environment.
7. **Execution authority:** the existing single-use grant binds older hashes
   and excludes PhasePair training. A new grant must bind the current contract,
   plan, code ledger, data scope, target runtime/server, and allowed actions.
8. **Downstream evidence:** no authorized training run, checkpoint, score
   matrix, bootstrap result, statistical claim, paper figure, editable slide
   deck, manuscript PDF, or public Git publication exists.

## Execution decision

- Do not construct a motion-only optimizer, load private data, use a GPU, run
  training, create a checkpoint, compute reportable retrieval statistics, or
  start the paper/slide publication pipeline.
- Keep `experiment-bridge` running and the machine state at
  `NO_DATA / NO_GPU / NO_CHECKPOINT / NO_SCORE / NO_RESULT / NO_CLAIM / AUTHORITY0`.
- Treat every locally callable path as a scoped contract primitive. Local
  callability is not end-to-end execution readiness.

## Safe next transition

First resolve and independently review the actual CLIP snapshot/runtime/rights
package. Then implement a sealed resolved-text runtime binding that holds
before any optimizer constructor when the live assessment or parameter
identity is missing. Only after that bridge, the energy-floor/data receipts,
the target runtime, and a new exact-hash grant are all frozen may the project
add AdamW and the training-step bridge and begin the plan's non-GPU preflight.
