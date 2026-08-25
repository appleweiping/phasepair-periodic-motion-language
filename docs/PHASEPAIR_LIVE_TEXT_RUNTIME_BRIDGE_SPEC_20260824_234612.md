# PhasePair sealed live text-runtime bridge implementation specification

**Generated:** 2026-08-24 23:46:12 +08:00  
**Schema:** `phasepair-live-text-runtime-bridge-spec-v1`  
**Contract family:** `PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840`  
**Status:** `DESIGN_ONLY / AUTHORITY0 / HOLD / NO_ISSUER / NO_RUNTIME / NO_OPTIMIZER / NO_TRAINING / NO_RESULT`  
**Authority:** `0`  
**Production:** `false`  
**Training authorized:** `false`

This document is an implementation specification, not a runtime receipt, authority
grant, rights verdict, optimizer artifact, or training result. It does not assert
that the pinned CLIP snapshot, wheelhouse, private-use rights review, resolution
authority, training authority, loader runtime, live model, optimizer, checkpoint,
or server exists. No API described below may be invoked in production until its
external authority and content prerequisites have been issued and independently
verified.

## 1. Binding inputs and non-authority boundary

This specification implements the inherited requirements of:

- `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260823_151744.md`, exact
  bytes/SHA-256 `51558 / 9fca7d4dc307864d3e5dba75c14e1ff6569a7e805ae75e5f4273e2efead3b394`;
- `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260824_165840.md`, exact
  bytes/SHA-256 `21568 / 02403b1bb5b3aae18853964be8cef5ade994fba13bc69d58a3b321e7ae15de1c`;
- `idea-stage/docs/phasepair_research_contract_20260824_165840.md`, exact
  bytes/SHA-256 `10613 / 1e8b90184acb04e5af298aa2be28585500a529b1ca32594952634e6b1fa06db4`.

The following existing objects remain descriptive or static-only and MUST NOT be
accepted as construction authority:

- `clip_resolution.ClipResolutionAssessment` and its canonical bytes;
- `clip_resolution.ClipResolutionEvidence`, including caller-supplied wheel,
  loader, trace, rights, or golden fields;
- `contracts.ParameterRow`, any tuple of rows, `OptimizerPartition`, or
  `OptimizerReceiptDraft`;
- `readiness.TrainingGateReferences` or a complete-reference readiness
  assessment;
- any caller-supplied tokenizer, text model, motion model, projection, logit
  scale, optimizer, parameter group, state dictionary, digest, or receipt.

An opaque hash proves neither object identity nor authority. A standalone receipt
or its SHA-256 can never be exchanged for a live lease or optimizer.

## 2. Minimal implementation surface

The implementation SHALL add `src/phasepair_core/text_runtime.py`. It MAY add
private snapshot-capability callables to `clip_resolution.py`, but those callables
must not enter `clip_resolution.__all__`. No production function may gain a test
injection keyword.

The only production API exported by `text_runtime.py` SHALL be:

```python
def issue_phasepair_runtime_lease(
    snapshot_root: pathlib.Path,
    wheelhouse_root: pathlib.Path,
    *,
    architecture: str,
    seed: int,
    resolution_authority: ResolutionAuthorityLease,
) -> ResolvedPhasePairRuntimeLease:
    ...

def canonical_runtime_receipt_bytes(
    lease: ResolvedPhasePairRuntimeLease,
) -> bytes:
    ...

def consume_phasepair_runtime_lease(
    lease: ResolvedPhasePairRuntimeLease,
    *,
    training_authority: TrainingAuthorityLease,
) -> PhasePairTrainingRuntime:
    ...
```

`ResolutionAuthorityLease`, `TrainingAuthorityLease`,
`ResolvedPhasePairRuntimeLease`, and `PhasePairTrainingRuntime` SHALL be exact,
closed-construction types. Their public constructors must raise `TypeError`.
Instances are valid only while present in their issuer's closure-owned registry.
Subclass instances, equal-leaf copies, `object.__new__` forgeries, and leases from
another factory or process are invalid.

The production signatures MUST NOT accept any of the following:

```text
assessment, evidence, rows, resolved_text_rows, frozen_rows, model,
text_model, motion_model, tokenizer, projection, logit_scale, state_dict,
parameter_groups, optimizer_class, optimizer_factory, loader, loader_factory,
hashes, receipt_bytes, receipt_sha256, runtime_evidence
```

`snapshot_root` and `wheelhouse_root` must be exact platform `pathlib.Path`
instances. `architecture` must be exact built-in `str` in `mime/early/late`.
`seed` must be an exact built-in integer in `[0,2^64-1]`. Both authority inputs
must be live opaque leases whose scope, path commitments, stage, expiry, single-use
state, and contract-family commitment are verified before any loader import or
file open beyond read-only preflight.

## 3. Private pinned-snapshot capability

`clip_resolution.py` MAY extend its existing sealed factory with exactly these
private callables:

```python
_issue_complete_pinned_snapshot_lease(root: Path) -> _PinnedSnapshotLease
_consume_complete_pinned_snapshot_lease(
    lease: _PinnedSnapshotLease,
) -> _PinnedSnapshotHandle
```

The issuer shall sign only a complete, eight-file, exact-pin, non-reparse,
single-link, local-path snapshot that passes the existing two equal full scans.
It shall not convert a `ClipResolutionAssessment` into a lease. The snapshot
lease is one-shot and carries no rights or training authority. `text_runtime.py`
captures the two private callables at module initialization; later global
rebinding is inert.

The text bridge shall perform complete snapshot scans at these boundaries:

1. before offline loader import/use;
2. after tokenizer and text model load;
3. immediately before emitting canonical receipt bytes;
4. inside `consume_phasepair_runtime_lease`, immediately before AdamW.

Every scan must agree in root identity, census, path metadata, per-file bytes and
SHA-256, and total bytes. Any difference is `HOLD_SNAPSHOT_TOCTOU` and burns the
live lease. This is a best-effort stable-snapshot seal, not a claim of an
operating-system atomic transaction.

## 4. Bridge-owned offline load

Only the bridge may import and call the pinned runtime loader. It shall validate
actual retained wheel/archive bytes before importing. At minimum the runtime
manifest binds PyTorch, Transformers, huggingface-hub, and tokenizers by
distribution, version, filename, bytes, and SHA-256. It also binds Python
implementation/version/platform, Torch/CUDA identity, imported source manifest,
loader source, training-forward source, AdamW source, constructor signature, and
their canonical SHA-256 digests.

The only accepted loader/tokenizer classes are the reviewed exact runtime types
equivalent to:

```text
CLIPTextModelWithProjection
CLIPTokenizerFast
```

The loader invocation is fixed to `local_files_only=True` and
`trust_remote_code=False`. Network calls are forbidden. The trace-open census
must be a nonempty subset of the reviewed local allowlist and must contain no
outside path. A mutable cache alias, floating revision, unreviewed shard/index,
added-token asset, or undeclared read is a HOLD.

The loaded object must expose final-LayerNorm pooled EOS and pretrained
`text_projection`. It must not instantiate a vision tower. If the pinned runtime
can only instantiate a full CLIP model, the status remains HOLD pending a fresh
contract amendment; the implementation may not freeze-and-ignore vision on its
own initiative.

The bridge internally owns and strongly retains:

```text
motion_model
tokenizer
CLIP text model
text.project = exact bridge-owned Linear(D_clip_hidden,512,bias=True)
text.logit_scale = exact nn.Parameter with shape [1]
```

It internally builds the selected motion encoder and applies the sealed motion
initializer. No caller model is accepted. The training text path is exactly:

```text
resolved text transformer final-LN pooled EOS
-> bridge-owned text.project
-> L2 normalization
```

The pretrained CLIP projection is used only by the separately traced anchor and
Hard-32 path. It is immediately frozen and never enters the training text path,
trainable count, optimizer, optimizer state, or training FLOPs.

## 5. Exact direct-registry validation

The implementation must follow the direct-registry pattern used by
`initialization.py`; it must not trust `named_parameters`, `named_buffers`,
`state_dict`, `_named_members`, an instance override, or a caller mapping.

For every module, recursively inspect `object.__getattribute__(module,
"__dict__")` and require:

- exact reviewed module type at each exact path;
- exact built-in `dict` for `__dict__`, `_parameters`, `_buffers`, and `_modules`;
- exact reviewed key order for parameter, buffer, and child registries;
- exact reviewed instance-field name/order and literal fields;
- exact empty hook registries and reviewed non-persistent buffer set;
- no extra hidden field, child, parameter, buffer, container, slot, cycle, or
  module alias;
- exact built-in string keys, never subclasses.

Every trainable parameter must be exact `nn.Parameter`, never a subclass. Every
buffer must be exact `torch.Tensor`, never a Tensor subclass. Both must satisfy
the reviewed device/dtype/layout/contiguity/storage-offset/storage-extent rules.
Parameter objects must be unique. Parameter and buffer storage intervals must be
pairwise disjoint. Views, shared storage, null pointers, storage over-allocation,
and registry replacements are invalid.

The first content-resolution run may produce a candidate module-tree manifest,
but production must not accept a tree merely because it was observed. A fresh
review must promote exact module paths/types/fields/registry order and manifest
SHA into the closure-captured production contract. Until then the only possible
status is `HOLD_FRESH_RUNTIME_REVIEW_REQUIRED`.

## 6. Live-derived parameter inventories

Resolved text rows are derived only from the validated live registry. Each
trainable text row is:

```text
canonical_name = "text.clip." + reviewed_resolved_state_key
component = "CLIP_TEXT_TRANSFORMER"
requires_grad = true
source_checkpoint_sha256 = pinned checkpoint raw32 SHA-256
```

Canonical name, resolved state key, and live parameter must be a bijection.
Semantic class is resolved by a reviewed owner-module/local-slot table, not by
rank, suffix, or dimension heuristics:

- exact embedding weight -> `EMBEDDING_WEIGHT`;
- exact Linear weight -> `MATRIX_WEIGHT`;
- exact Linear bias -> `BIAS`;
- exact LayerNorm weight/bias -> `LAYERNORM_GAMMA/LAYERNORM_BETA`;
- explicitly reviewed other parameter slot -> `QUERY_OR_OTHER_WEIGHT`;
- any unknown owner/type/slot -> `HOLD_RESOLVED_TEXT_INVENTORY`.

The frozen inventory must contain exactly one canonical row:

```text
text.pretrained_projection.weight
component = CLIP_PRETRAINED_TEXT_PROJECTION
semantic_class = FROZEN
requires_grad = false
optimizer_group = NOT_APPLICABLE
```

The trainable bridge-owned head adds exactly:

```text
text.project.weight [512,D] MATRIX_WEIGHT decay
text.project.bias   [512]   BIAS          no_decay
text.logit_scale    [1]     LOGIT_SCALE   no_decay
```

`text.project` uses the inherited `phasepair-init-v1` byte derivation under the
same run seed. The logit scale must use a fresh-reviewed pinned float32 raw-byte
golden for `log(1/0.07)`; a runtime `math.log` call is not authority. Until that
golden is promoted, issuance must HOLD.

After deriving live rows, the bridge may call existing static contract validators
as an independent cross-check. Those validators do not become the source of live
identity or authority. The full inventory remains exactly motion plus resolved
trainable CLIP text plus project weight/bias plus logit scale. Vision and the
pretrained projection remain optimizer-invisible.

## 7. Active text-dropout and actual-forward proof

Resolution must prove that active train-mode text dropout count is zero. A
configuration field or module census alone is insufficient. The bridge shall:

1. enumerate every reviewed dropout operator and probability by direct registry;
2. run the exact fixed ASCII synthetic caption fixture in train mode;
3. trace the actual forward for dropout, RNG, Bernoulli/native-dropout, fused
   stochastic operations, checkpointing, and hidden recomputation;
4. require all `0 < p < 1` active sites to be absent;
5. label reviewed `p == 0` sites `ZERO_PROBABILITY_NONCONSUMING`;
6. prove the RNG state is unchanged and no framework RNG call occurred;
7. use autograd with `allow_unused=True` and non-materialized missing gradients
   to match all and only training-forward-read text parameters to the live
   trainable inventory;
8. separately trace the frozen pretrained projection in the anchor golden path;
9. transactionally restore train/eval state, hooks, registries, gradients, and
   RNG state, then revalidate them.

Any nonzero or hidden stochastic text site, fused RNG, gradient checkpointing,
or hidden recomputation is `HOLD_ACTIVE_TEXT_DROPOUT_OR_RECOMPUTATION`; it may not
fall back to framework RNG or silently change the contract.

The synthetic golden binds exact caption bytes, token IDs, attention mask,
tokenizer max length 77, final-LN pooled EOS, pretrained projection output, and
float64 L2-normalized output bytes/SHA-256. Empty or nonfinite output is fatal.

## 8. Runtime receipt schema

`canonical_runtime_receipt_bytes` revalidates the live lease and the current
snapshot before serializing. It never consumes the lease and never authorizes
AdamW. Schema is `phasepair-resolved-text-runtime-receipt-v1`; top-level keys are
closed and exactly:

```text
schema,status,authority,production,training_authorized,contract_family,
model_id,revision,architecture,seed,snapshot_manifest,runtime_manifest,
rights_manifest,config_manifest,tokenizer_manifest,loader_manifest,
dropout_trace,registry_manifest,trainable_rows,frozen_rows,
text_head_manifest,optimizer_candidate,motion_initializer_receipt_sha256,
motion_initial_state_sha256,full_initial_state_sha256,hold_reasons
```

Every parameter row has exactly:

```text
canonical_name,resolved_state_key,component,shape,numel,dtype,
semantic_class,requires_grad,source_checkpoint_sha256,tensor_bytes_sha256,
optimizer_group,object_identity_ordinal,storage_identity_ordinal,storage_nbytes
```

The manifests bind actual, bridge-observed values including:

- complete snapshot and per-file path/bytes/SHA;
- checkpoint, config raw bytes, canonical parsed config, and hidden size;
- tokenizer asset set, vocab, merges, tokenizer JSON/config, class, and max77;
- retained wheels, imported source set, loader/forward/AdamW source and runtime;
- loader open trace and synthetic golden;
- direct module registry, buffer rows, object identities, and storage ranges;
- per-loaded-tensor raw bytes SHA and checkpoint state-key mapping;
- trainable/frozen inventories and ordered decay/no-decay name lists;
- motion initializer receipt/state and bridge text-head/full initial state.

Canonical JSON uses exact built-in scalar/container types, closed keys, UTF-8
name-byte ordering, lowercase SHA-256, `sort_keys=True`, compact separators,
`ensure_ascii=False`, `allow_nan=False`, UTF-8 encoding, and exactly one trailing
LF. JSON floats are forbidden; numeric runtime settings use frozen decimal or
IEEE hexadecimal strings.

The receipt's literals remain `authority=0`, `production=false`, and
`training_authorized=false` until a different, separately reviewed contract
defines a production receipt. A receipt alone is never consumable.

## 9. Opaque live-lease state machine

Each runtime lease has a closure-captured lock, strong references to every owned
runtime object, and an exact integer lifecycle:

```text
0 ISSUED
1 CONSUMING
2 CONSUMED
3 BURNED
```

Allowed transitions are:

```text
ISSUED -> CONSUMING -> CONSUMED
ISSUED -> CONSUMING -> BURNED
ISSUED ----------------> BURNED
```

No other transition is legal. The issuer registry is weak-keyed by the lease.
Its values contain only primitive provenance/seals/identity ordinals and never a
lease, model, tokenizer, parameter, or other back-reference. The lease itself is
the only strong owner of its live object set before consumption. Garbage
collection must return the weak registry to baseline and release those objects.

Consumption performs an atomic locked `ISSUED -> CONSUMING` transition. A second
thread, recursive consume, or repeated consume cannot enter the constructor. Any
`BaseException` during authority checks, revalidation, snapshot scan, constructor,
or post-constructor verification changes the state permanently to `BURNED` and
clears owned references. Failure can never reset a lease to `ISSUED`. Successful
consumption changes it permanently to `CONSUMED`.

Immediately before optimizer construction, the consumer rechecks exact registry
objects, object IDs, storage ranges, `_version` observations, raw tensor bytes,
requires-grad flags, training modes, config/tokenizer/runtime/source hashes,
dropout trace, motion receipt/state, text-head state, full inventory, group
membership, and the final snapshot scan. Direct registry rebinding, `.data`
replacement, same-shape parameter substitution, Q/K swap, or byte mutation burns
the lease.

## 10. AdamW zero-call gate and consumption

No code path outside `consume_phasepair_runtime_lease` may instantiate AdamW.
Within the consumer, the following must finish successfully before the captured
constructor is called:

1. resolution-authority provenance remains valid;
2. training authority is a valid, unexpired, single-use opaque lease for the
   exact contract/runtime/snapshot/architecture/seed;
3. rights verdict permits only the authorized private noncommercial use and
   forbids redistribution;
4. all live registry/object/storage and raw-byte checks pass;
5. snapshot/config/tokenizer/runtime/source/golden/dropout checks pass;
6. motion initializer receipt and state pass;
7. project/logit initial bytes pass;
8. live-derived trainable/frozen rows and exact two-group partition pass;
9. final full snapshot double scan passes.

If any item is missing or invalid, the observable test invariant is:

```text
adamw_constructor_call_count == 0
```

Only then may the bridge call the closure-captured exact AdamW type and original
constructor once, with exactly:

```python
AdamW(
    [
        {"params": decay_parameters, "weight_decay": 1e-4},
        {"params": no_decay_parameters, "weight_decay": 0.0},
    ],
    lr=1e-4,
    betas=(0.9, 0.999),
    eps=1e-8,
    amsgrad=False,
    maximize=False,
    foreach=False,
    capturable=False,
    differentiable=False,
    fused=False,
)
```

Group order is decay then no-decay; parameter objects within each group follow
canonical UTF-8 name order. Classification reads only semantic class. Frozen
rows never enter either group. The constructor is called at most once per lease.

Immediately after construction, verify exact optimizer type, exactly two groups,
parameter identity and order by `is`, weight decays, shared kwargs, and the
reviewed step-zero state schema. A mismatch burns the lease and returns nothing.
The returned `PhasePairTrainingRuntime` owns the model, tokenizer, optimizer, and
their sealed methods; it does not expose an interchangeable caller model or a
receipt-to-runtime reconstruction path.

## 11. Synthetic seam

Production signatures remain closed. Tests use only an isolated private factory:

```python
def _seal_text_runtime_api_for_tests(
    contract_snapshot: _SyntheticContractSnapshot,
    filesystem_ops: _SyntheticFilesystemOps,
    loader_ops: _SyntheticLoaderOps,
    optimizer_ops: _SyntheticOptimizerOps,
) -> _SyntheticRuntimeApi:
    ...
```

Every factory invocation owns fresh capability types, issuer registries, locks,
and captured callables. Synthetic leases have an exact type distinct from the
production lease and are rejected by the production consumer. Synthetic receipts
are permanently `SYNTHETIC_AUTHORITY0_NONPRODUCTION`. Tiny checkpoint/config/
tokenizer fixtures and an optimizer spy may exercise success, failure, lifecycle,
TOCTOU, GC, and concurrency without importing Transformers, accessing a network,
or constructing a production optimizer.

No test seam may accept production authority, emit a production receipt, or
promote a synthetic outcome.

## 12. Must-kill matrix

The implementation is not reviewable until a dedicated
`tests/test_phasepair_core_text_runtime.py` rejects every class below.

| ID | Accepted-invalid mutant | Required outcome before AdamW |
|---|---|---|
| LTR01 | assessment + convincing bare rows + caller text model | signature/type rejection; constructor count 0 |
| LTR02 | caller tokenizer/model/projection/logit/hash/group/optimizer injection | signature rejection; count 0 |
| LTR03 | forged, copied, subclassed, foreign-factory, or already-consumed lease | HOLD/burn; count 0 |
| LTR04 | standalone receipt/hash used in place of live lease | rejection; count 0 |
| LTR05 | missing/expired/wrong-scope/replayed authority or rights evidence | HOLD/burn; count 0 |
| LTR06 | global rebind of constants/helpers/hash/json/Torch/AdamW/loader | captured semantics unchanged |
| LTR07 | rebound `named_parameters`, `named_buffers`, `state_dict`, `_named_members` | ignored; direct registry remains authoritative |
| LTR08 | ndarray, Tensor, Parameter, dict, string, or module subclass | reject before content/callback read |
| LTR09 | registry extra/omission/reorder, hidden field/container/slot, hook, cycle, alias | HOLD/burn; count 0 |
| LTR10 | object alias, overlapping/view storage, offset/layout/extent/dtype/device drift | HOLD/burn; count 0 |
| LTR11 | Q/K or equal-shape swap; canonical/state-key mismatch | HOLD/burn; count 0 |
| LTR12 | text row missing/extra/frozen, unknown semantic owner, requires-grad drift | HOLD/burn; count 0 |
| LTR13 | vision appears; pretrained projection trainable/in optimizer/aliased with project | HOLD/burn; count 0 |
| LTR14 | project/logit missing, wrong shape/object/storage/bytes/classification | HOLD/burn; count 0 |
| LTR15 | group swap/third group/duplicate/missing; LN gamma, embedding, or pool.query misgrouped | HOLD/burn; count 0 |
| LTR16 | nonzero/hidden/fused text dropout, RNG call, checkpointing, recomputation | HOLD/burn; count 0 |
| LTR17 | config hidden-size, tokenizer max77/assets/token IDs/golden drift | HOLD/burn; count 0 |
| LTR18 | wheel/runtime/source/constructor/signature hash drift | HOLD/burn; count 0 |
| LTR19 | snapshot changes during load, after load, after receipt, or before consume | TOCTOU burn; count 0 |
| LTR20 | parameter bytes, `.data`, registry, mode, grad flag, or buffer mutate after issue | burn; count 0 |
| LTR21 | receipt unknown/missing key, float, uppercase hash, noncanonical JSON, row reorder | receipt rejection; count 0 |
| LTR22 | two threads or recursive consume | at most one constructor call and one success |
| LTR23 | constructor raises or post-constructor audit fails | permanent burn; no retry |
| LTR24 | successful consume followed by repeat consume | rejection; no second constructor call |
| LTR25 | lease/runtime GC | weak registries return to baseline; no retained model graph |
| LTR26 | complete readiness digest references without live leases | still HOLD; count 0 |
| LTR27 | synthetic lease presented to production consumer | exact-type/issuer rejection; count 0 |

Tests must additionally prove that a valid synthetic lifecycle derives every row
from the same retained live objects, that ordered optimizer parameter objects are
identical to those rows, and that every pre-constructor failure leaves the
optimizer spy at zero calls.

## 13. Current STOP

At the time of this specification:

- no resolution-authority issuer exists;
- no training-authority issuer exists;
- no reviewed live-registry manifest exists;
- no pinned loader/wheel runtime has been loaded;
- no actual CLIP snapshot has been opened by this work;
- no model, tokenizer, live lease, runtime receipt, AdamW instance, optimizer
  state, checkpoint, training step, score, result, or claim has been produced;
- this specification grants no data, network, SSH, server, GPU, training,
  publication, or Git authority.

Therefore the only truthful state is:

```text
AUTHORITY0 / HOLD / NO_ISSUER / NO_RUNTIME / ADAMW_CONSTRUCTOR_CALL_COUNT=0 /
NO_TRAINING / NO_RESULT
```

Implementation must remain fail-closed until separate ARIS authority, rights,
content, runtime, fresh-review, and training gates are satisfied. STOP.
