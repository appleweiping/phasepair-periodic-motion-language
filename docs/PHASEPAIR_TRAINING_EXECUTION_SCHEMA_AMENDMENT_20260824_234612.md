# PhasePair training execution schema amendment v2

**Amendment ID:** `phasepair-training-execution-schema-amendment-v2/20260824_234612`  
**Public artifact ID:** `artifact://phasepair/PHASEPAIR_TRAINING_EXECUTION_SCHEMA_AMENDMENT_20260824_234612.md`
**Status:** `AUTHORING_ONLY / AUTHORITY0 / HOLD / NOT_RUN / NO_CHECKPOINT / NO_RESULT / NO_CLAIM`  

## 0. Scope and authority boundary

This document closes the training-execution identity and state-machine gap found
between the current `165840` scientific family and the inherited `151744`
execution-receipt vocabulary.  It is a schema amendment only.  It does not
authorize CLIP or dataset acquisition, private-data access, SSH, GPU use,
training, checkpoint creation, result generation, publication, or Git activity.
No run identity, receipt, checkpoint, cache, score, result, or claim exists at
authoring time.

Bound read-only sources:

| source | bytes | SHA-256 |
|---|---:|---|
| `idea-stage/docs/phasepair_research_contract_20260824_165840.md` | 10,613 | `1e8b90184acb04e5af298aa2be28585500a529b1ca32594952634e6b1fa06db4` |
| `artifact://phasepair/PHASEPAIR_EXPERIMENT_PLAN_20260824_165840.md` | 10,042 | `335d48ec91609364509a7bb922995d1a60a8b4013f8c689e74afc4394ade8ac3` |
| `artifact://phasepair/PHASEPAIR_MIME_REIMPLEMENTATION_SPEC_20260823_151744.md` | 51,558 | `9fca7d4dc307864d3e5dba75c14e1ff6569a7e805ae75e5f4273e2efead3b394` |

The `165840` contract and plan remain the scientific authority.  The `151744`
MIME specification remains the detailed inherited source for optimizer,
training-step, checkpoint-selection, residual-training, and execution-receipt
semantics only where this amendment does not replace them.  This amendment must
receive fresh byte-level review and explicit binding before any implementation
may call it production authority.

## 1. Conflict closure and v2 execution census

The inherited `151744` receipt matrix used base TMR system `06`, residual systems
`01..05`, five residual heads, and `phasepair-run-v1`.  Those identifiers are
not current.  For this amendment and every successor that binds it:

```text
seeds = [1729,2718,31415]

base systems = [00 MIME,07 TMR_STYLE_EARLY_FUSION,
                08 TMR_STYLE_LATE_FUSION]
base run count = 3*3 = 9

residual systems = [01 GENERIC,02 WAMO_MARGINAL_WAVELET,
                    03 INTEREDIT_MEAN_DIFFERENCE_DCT,04 NO_RELATION,
                    05 PHASE_STRIPPED,06 PHASEPAIR_FULL]
residual run count = 3*6 = 18
validation score rows = 3*7 = 21
```

The only base run IDs are:

```text
phasepair-run-v2/BASE_TRAIN/<1729|2718|31415>/<00|07|08>
```

The only residual run IDs are:

```text
phasepair-run-v2/RESIDUAL_HEAD_TRAIN/<1729|2718|31415>/<01|02|03|04|05|06>
```

Legacy system `06=TMR`, a five-head census, a 15-run residual census, or any
`phasepair-run-v1` identity is rejected with
`TRAINING_EXECUTION_IDENTITY_VERSION_FAIL`.  Systems `02` and `03` are separate
matched controls and separate train-from-scratch heads; neither may alias,
share optimizer state, or be merged into the inherited five-head
`marginal+S/D` row.

## 2. Six-head parameter ledger

Each residual run owns one and only one shared-six-slot projector:

```text
residual.fc1.weight [256,13]     3,328
residual.fc1.bias   [256]          256
residual.fc2.weight [512,256]  131,072
residual.fc2.bias   [512]          512
                                  -----
per-head total                 135,168
```

Therefore:

```text
six independent head artifacts per seed = 6*135,168 = 811,008 parameters
eighteen run-local head artifacts total  = 18*135,168 = 2,433,024 parameters
```

These are artifact-local totals, not the parameter count of one run and not a
claim about simultaneous residency.  All six heads for one seed must begin from
parameter-wise byte-identical canonical initial state, but their `nn.Parameter`
objects, storage ranges, optimizer objects, optimizer states, checkpoint
namespaces, and output namespaces must be pairwise disjoint.  A six-slot
per-band projector mutant has `6*135,168` parameters in one run and must fail.

## 3. Closed base lifecycle

The base lifecycle is intentionally two-phase so the current gate order and the
inherited cache-finalization rule are both satisfiable:

```text
9 committed run inputs
  -> 9 BASE_TRAIN executions
  -> 9 immutable BASE_TRAIN_COMPLETION objects
  -> one three-model/three-seed BASE_QUALIFICATION object
  -> if qualified: three selected-MIME frozen-base caches
  -> exactly 9 terminal BASE_TRAIN success/failure receipts
  -> only then may a QualifiedMimeBaseLease be minted
```

`BASE_TRAIN_COMPLETION` is not a terminal run receipt and cannot be used by a
residual job.  It records the selected checkpoint and validation metric artifact
for the completed execution, but no MIME cache commitment.  Its exact top-level
keys are:

```text
schema,run_id,run_input_sha256,system_id,seed,architecture,
optimizer_step_count,example_count,selected_epoch_index,
selected_completed_epoch,selected_checkpoint_sha256,
validation_metric_sha256,evaluator_sha256,log_sha256,
model_final_state_sha256,optimizer_final_state_sha256,status
```

Literals are `schema=phasepair-base-train-completion-v2` and
`status=EXECUTION_COMPLETE_NONTERMINAL_AWAITING_QUALIFICATION`.
`selected_epoch_index` is in `12..29`, and
`selected_completed_epoch=selected_epoch_index+1`.

The qualification object consumes all nine completion digests in exact system
outer, seed inner order.  It applies the current MIME gate: MIME mean strictly
exceeds both early and late means; at least two of three MIME seeds exceed the
stronger same-seed comparator; minimum same-seed delta is at least `-0.005`.
Its scientific verdict is `QUALIFIED` or `NOT_QUALIFIED`; either verdict may
follow nine operationally successful executions and is not itself an
operational failure.

If and only if verdict=`QUALIFIED`, the three selected MIME checkpoints may each
build one train/validation frozen-base cache.  MIME terminal success receipts
then bind the corresponding cache digest.  Early and late terminal success
receipts use literal `NOT_APPLICABLE` for that field.  If verdict is
`NOT_QUALIFIED`, no cache is built and all nine terminal execution receipts may
still be `SUCCESS`, but every cache field is `NOT_APPLICABLE` and no
`QualifiedMimeBaseLease` is minted.  A scientific failure must not be rewritten
as an operational run failure.

Each run identity has exactly one terminal success or failure after its unique
run input.  A completion object, qualification object, or cache receipt is not a
second terminal.  Missing, duplicate, or mixed terminal objects fail with
`TRAINING_EXECUTION_TERMINAL_CARDINALITY_FAIL`.

## 4. Training admission and live-state ownership

No public API may construct an optimizer from a caller-supplied
`OptimizerPartition`, parameter row list, model object plus claimed hashes, or
an authority-zero assessment.  The minimum admission chain is:

```python
admit_base_run(
    verified_run_input_lease,
    initialized_motion_lease,
    sealed_text_training_lease,
    sealed_base_sampler_lease,
    sealed_base_dropout_schedule_lease,
    first_prepared_training_batch_lease,
) -> BaseRunAdmission

start_base_training(admission: BaseRunAdmission) -> BaseTrainingSession
```

Every input is an opaque, issued-registry-backed, strong-reference lease.
Ordinary copies, subclasses, reconstructed dataclasses, bare receipt bytes, and
expired, replayed, or previously consumed leases are rejected.  The admission
is single-consume, so one admitted live parameter inventory can construct one
and only one optimizer.

The sealed text lease must be minted only by a bridge-owned offline loader.  The
bridge, not the caller, owns tokenizer/model construction, live registry
traversal, resolved state-key mapping, project/logit initialization, and forward
trace.  It must prove:

- exact resolved snapshot and reviewed runtime/right receipts;
- text-only live model; vision instantiated count `0`;
- pretrained CLIP `text_projection` frozen and optimizer-invisible;
- active train-mode text dropout site count `0`;
- identical resolved text inventory semantics for base systems `00/07/08`;
- live parameter identity/storage/state-key mapping and initial-state digests;
- new `text.project.weight/bias` and `text.logit_scale` are present and live.

The motion lease must be minted by verifying a production initializer receipt
against the same live model, seed, architecture, parameter identities, storage
ranges, and canonical state digest in the admission critical section.  The
current `HOLD_RUNTIME_LIBM_ORACLE_UNPINNED_AUTHORITY0` receipt is never eligible.

The sampler lease owns the full 30-epoch base manifest set, pair commitments,
caption orders, anchor receipt, and global-step mapping.  The first and every
subsequent prepared batch must match its planned batch indices and pair/caption
commitments exactly.  `PreparedMotionBatch` validation alone is insufficient.

The dropout lease binds one reviewed `phasepair-base-dropout-schedule-receipt-v1`
and outer pointer for the exact architecture/seed.  The training session mints
the single-use per-forward bundle internally.  A caller-created
`OneForwardCanonicalDropoutBundle`, even of the exact public class, is rejected.

## 5. Exact AdamW construction

`start_base_training` derives live parameters from the sealed motion and text
registries and compares them with the canonical inventory internally.  It does
not accept caller-supplied groups.  It constructs one AdamW with exactly two
groups:

```python
torch.optim.AdamW(
    [
        {"params": decay_params, "weight_decay": 1e-4},
        {"params": no_decay_params, "weight_decay": 0.0},
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

Group order is `decay,no_decay`; names within each group are strictly ascending
by UTF-8 bytes.  Matrix, embedding, query, and other weights are decay.
Biases, LayerNorm gamma/beta, and `text.logit_scale` are no-decay.  The union is
the full live trainable inventory, the intersection is empty, and every live
parameter identity occurs exactly once.  Vision and pretrained projection
optimizer/state rows are zero.  Backend flags, constructor signature, source,
runtime, group-name lists, step-0 state, and first-step state all receive
content digests.

Residual sessions use the same backend flags and exact grouping law, with
`lr=3e-4`, and only the four run-local head tensors are trainable.  The selected
MIME motion/text/logit state and cache are read-only; base parameters remain
`eval()` and `requires_grad=False` even if an enclosing head module enters train
mode.

Canonical post-first-step optimizer state is serialized group0 then group1,
name order inner.  Each parameter row contains
`step_u64be`, C-order little-endian float32 `exp_avg`, and `exp_avg_sq`.
For `R` tensors and `P` elements: moment elements=`2P`, step scalars=`R`, and
logical bytes=`8P+8R`.  Framework pickle/ZIP bytes are separately identified
and can never substitute for canonical logical bytes.

## 6. Exact base training step

The only state-mutating base API is:

```python
train_base_step(
    session: BaseTrainingSession,
    batch: PreparedTrainingBatchLease,
) -> TrainingStepReceipt
```

Its order is exact:

1. Revalidate session ownership, plan cursor, batch digest, epoch/batch indices,
   and dropout global step.
2. `optimizer.zero_grad(set_to_none=True)` exactly once.
3. Run the motion encoder once with the internally minted canonical dropout
   bundle; its ordered pass axis is `[AB,BA]`.
4. Run the text transformer once; final-LN pooled EOS passes through the new
   project and L2 normalization.  Active text dropout remains zero.
5. Internally derive the exact `[B,3B]` positive mask from pair/caption
   commitments; callers cannot supply a mask.
6. Call the canonical PhasePair objective once.  Base score is exactly
   `0.5*(AB@text.T+BA@text.T)`; every source has three positives and every
   caption one source.
7. Require finite scalar loss, then call `loss.backward()` exactly once.
8. Require a finite float32 gradient for every trainable parameter and no
   gradient for any frozen parameter; record the canonical pre-clip gradient
   digest.
9. Apply one global clip over the canonical full trainable order:

   ```python
   torch.nn.utils.clip_grad_norm_(
       canonical_params,
       max_norm=1.0,
       norm_type=2.0,
       error_if_nonfinite=True,
       foreach=False,
   )
   ```

10. Record pre-clip norm and post-clip gradient digest, then call
    `optimizer.step()` exactly once.
11. Require finite model and optimizer state, equal per-row Adam step, and
    global optimizer step transition `g -> g+1`.
12. Call `zero_grad(set_to_none=True)` so a checkpoint never owns pending grads.
13. Assert both group LRs are unchanged.  The only LR policy is
    `CONSTANT`; scheduler=`NOT_APPLICABLE`; scheduler step count=`0`.
14. Commit the immutable private step receipt, then advance the session cursor.

The step receipt schema is `phasepair-training-step-receipt-v1` with exact keys:

```text
schema,run_input_sha256,run_id,role,system_id,seed,architecture,
epoch_index,batch_index,global_optimizer_step_before,
global_optimizer_step_after,examples_seen_after,
prepared_motion_batch_sha256,prepared_training_batch_sha256,
sampler_epoch_plan_sha256,dropout_row_set_sha256,
dropout_application_trace_sha256,text_batch_sha256,
positive_mask_sha256,logits_sha256,loss_f32le_sha256,
gradient_preclip_sha256,preclip_global_norm_f32le_sha256,
gradient_postclip_sha256,model_state_before_sha256,
model_state_after_sha256,optimizer_state_before_sha256,
optimizer_state_after_sha256,lr_policy,scheduler,
objective_source_sha256,trainer_source_sha256,runtime_sha256,status
```

This is a private integrity receipt, not a public per-sample log or result.

## 7. Atomic checkpoint and deterministic resume

Canonical checkpoint state uses content-addressed immutable blobs plus one
atomic manifest commit marker.  The writer must:

1. create sibling temporary files exclusively;
2. write model, optimizer, trainer-cursor, and RNG/config blobs;
3. flush, fsync/FlushFileBuffers, close, reopen, and rehash each blob;
4. publish blobs under their content digests without overwrite;
5. write and verify a canonical temporary manifest;
6. atomically replace the final manifest path;
7. treat only the final manifest as checkpoint commitment.  Orphan blobs or
   temporary files are never resumable checkpoints.

The exact manifest keys are:

```text
schema,contract_family_sha256,run_input_sha256,run_id,role,
system_id,seed,architecture,completed_epoch,epoch_index,
next_epoch_index,next_batch_index,global_optimizer_step,examples_seen,
model_inventory_sha256,model_state_sha256,optimizer_schema_sha256,
optimizer_state_sha256,sampler_manifest_set_sha256,
dropout_schedule_sha256,rng_state_sha256,runtime_sha256,
trainer_source_sha256,previous_checkpoint_sha256,
lr_policy,scheduler,status
```

Literals are `schema=phasepair-training-checkpoint-v1`,
`lr_policy=CONSTANT`, `scheduler=NOT_APPLICABLE`, and `status=COMMITTED`.
Checkpointing occurs only after a complete optimizer step and post-step
zero-grad.  Candidate base checkpoints exist only for
`completed_epoch=13..30`, equivalent to `epoch_index=12..29`; index `30` is
invalid.

The RNG/config blob binds Python, NumPy, Torch CPU, every visible CUDA generator,
deterministic-algorithm state, TF32 flags, and the required CUDA deterministic
environment.  Canonical sampler, initialization, and dropout must not consume
framework RNG, but the states are still bound to detect hidden consumption.

The only resume API is:

```python
resume_base_training(
    admission: BaseRunAdmission,
    checkpoint_receipt: bytes,
) -> BaseTrainingSession
```

It reconstructs fresh live motion/text/project/logit objects, consumes a fresh
admission, constructs the same one-AdamW/two-group layout, strictly loads named
canonical model and optimizer state, restores cursor and RNG/config state,
rehashes every artifact, and verifies the next sampler/dropout tuple.  It does
not accept a caller `state_dict`.  A required golden compares uninterrupted
steps with checkpoint/resume at the same step boundary; model, optimizer,
cursor, next-step, and receipt digests must be identical in the same bound
runtime/hardware.  Cross-runtime or cross-hardware equality is not inferred.

## 8. Job separation and promotion gate

The job census must be generated mechanically in canonical system-outer,
seed-inner order and must prove exactly 9 base plus 18 residual identities.
Every run has a unique run-input digest, optimizer object, parameter identity
set, checkpoint namespace, log namespace, and terminal namespace.

Same-seed base runs may share only immutable input artifacts: batch/caption/crop
and normalization manifests, resolved CLIP initial bytes, optimizer schema, and
global-step mapping.  They may not share live text or motion parameters,
optimizer state, gradients, checkpoints, or output files.

Residual jobs are not constructible until a `QualifiedMimeBaseLease` has been
minted from all nine terminal base receipts, the qualified strength artifact,
and all three verified MIME caches.  Same-seed residual runs may share only the
read-only selected MIME cache, the 20 frozen head batch manifests, and the
reviewed head-dropout schedule.  They may not share head parameter objects,
optimizer states, checkpoints, or output files.

## 9. Mandatory rejection tests

At minimum, the production closure must reject:

1. forged, copied, subclassed, expired, replayed, or already-consumed leases;
2. assessment plus bare rows/model/hashes in place of a live text lease;
3. text or motion parameter replacement, storage alias, state-key drift, or
   state mutation after verification;
4. an AUTH0/HOLD initializer or CLIP assessment entering admission;
5. hidden text dropout, vision instantiation, or pretrained projection in the
   optimizer;
6. a caller-created dropout bundle or a schedule/outer-pointer mismatch;
7. prepared-batch pair/caption order differing from the sampler plan;
8. second AdamW construction, group swap, omit/duplicate/extra parameter,
   incorrect WD/LR/backend flag, or scheduler creation;
9. missing/double backward, missing/double/per-group clip, nonfinite gradient,
   frozen gradient, or optimizer step after failure;
10. missing/extra optimizer state row, unequal Adam steps, wrong endian/order,
    max-moment state under `amsgrad=False`, or frozen optimizer state;
11. partial checkpoint publication, manifest/blob corruption, wrong
    run/seed/architecture/plan/dropout resume, or mid-step checkpoint;
12. uninterrupted/resume digest divergence in the same bound runtime;
13. eight-or-fewer base jobs, legacy `06=TMR`, fifteen residual jobs, merged
    systems `02/03`, or any residual launch before qualification;
14. shared live parameter, optimizer, checkpoint, or output identity across
    distinct jobs;
15. a completion or qualification object accepted as a terminal receipt.

## 10. Implementation boundary

The following work is data-free and may be implemented before external
execution authority, provided every public production path remains HOLD:

- v2 job, completion, qualification, run, step, optimizer-state, and checkpoint
  schemas plus strict validators;
- opaque admission/session state machines with private tiny-fixture test seams;
- canonical live-name grouping and AdamW logical-state serialization;
- six-head projector topology, parameter ledger, and head-dropout-v2 primitives;
- sampler-manifest serialization and prepared-batch-to-plan join;
- base dropout schedule/outer receipt verification and internal bundle minting;
- exact step ordering, constant-LR guard, atomic checkpoint writer, and CPU tiny
  uninterrupted/resume tests;
- exact 9/18 census, namespace isolation, and all must-kill mutants above.

The following remain mandatory HOLDs:

- live production text lease until exact CLIP files, pinned wheels, offline
  loader/open trace, model rights, synthetic goldens, and fresh review exist;
- production motion initialization until the current libm and hash-selected-16
  HOLDs are resolved by a reviewed successor receipt;
- real sampler, captions, batches, normalization, lineage, and energy floors
  until the owner-authorized InterHuman audit and private receipts exist;
- actual optimizer source/runtime/state goldens until the target server runtime
  is pinned and reviewed;
- CUDA determinism/resume, memory, overfit, 9 base, 18 residual, validation, and
  sealed test until their distinct authority gates are granted and passed.

## 11. Truthful terminal state

This amendment records design closure only.  Its terminal state is:

```text
AUTHORITY0
HOLD_EXTERNAL_RIGHTS_ASSET_CLIP_SERVER_GPU_GRANTS
NOT_RUN
NO_OPTIMIZER_CONSTRUCTED
NO_TRAINING_STEP
NO_CHECKPOINT
NO_CACHE
NO_SCORE
NO_RESULT
NO_CLAIM
```

It must not be cited as experimental evidence, runtime readiness, a successful
receipt, or permission to execute any external action.
