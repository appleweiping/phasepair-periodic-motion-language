# PhaseSet private-host runtime integration

The composition root is `src/phaseset_core/host.py`. It contains no server endpoint,
credential, dataset root, or receipt payload.

## Implemented production chain

`host.RuntimeAdapterFactory(private_host_config)` is passed directly to
`phaseset_core.cli.main(..., runtime_adapter_factory=factory)`.  The factory:

1. loads an existing private receipt record;
2. hashes every receipt artifact named by that record, including the rights,
   runtime, execution, source/prepared-data, split, caption, scoring-census,
   and hard-gallery records required by `PrivateReceiptAssertions`;
3. reads every prepared NPZ once into a bounded buffer, authenticates that exact
   buffer, and parses the same bytes without loading pickle data;
4. constructs the repository's `ProductionRuntimeAdapter`.

It never creates authority, rights, runtime, or execution receipts.  If an
artifact is absent or changed, the factory raises the existing
`ExecutionHold` codes and the public CLI remains authority zero.

`PrivatePreparedDataSource` maps one immutable split manifest to the public
`TrainingDataSource` protocol.  Each NPZ contains exactly:

- `skeletons`: float32 `[B,K,200,22,3]`
- `actor_mask`: bool `[B,K]`
- `frame_mask`: bool `[B,200]`
- `track_mask`: bool `[B,K,200,22]`
- `actor_commitments`: uint8 `[B,K,32]` (ignored where actor_mask is false)
- `group_commitments`: uint8 `[B,32]`
- `text_embeddings`: float32 `[Q,D]`
- `motion_positive_ids`: uint8 `[B,32]`
- `text_positive_ids`: uint8 `[Q,32]`
- `text_commitments`: uint8 `[Q,32]`

`run-base` uses the formal seed-bound B0--B2 constructor and
`PhaseSetTrainingRuntime`.  Its checkpoint directory must be
`<attempt-root>/<attempt-id>/model-checkpoints`, allowing the harness to build
the public write-once attempt/checkpoint/terminal ledger around the runtime
checkpoint.  A controlled interruption is terminalized as
`CONTROLLED_INTERRUPTION`, retaining a resumable checkpoint.

The optional `checkpoint_observer` adds a post-atomic-write
observer to the public runtime without changing model, optimizer, sampler, or
numerical behavior. The host uses it to write live checkpoint receipts and a
30-second heartbeat chain while `fit()` is active. The final update checkpoint
of an epoch is deliberately deferred because the runtime immediately writes a
validation checkpoint at the same global step, while `AttemptStore` requires
strictly increasing receipt steps. Python failures and controlled interrupts
terminalize against the latest live receipt. A report checkpoint restored from
the predecessor is explicitly receipted into the new attempt even when `fit()`
makes no progress and emits no observer event.

Each active attempt holds an OS-released exclusive lock at
`<attempt>/.lifetime.lock` from attempt creation through successful or failed
terminal persistence. Heartbeats are immutable files under
`<attempt>/heartbeats/`; checkpoint receipts are immutable files under
`<attempt>/checkpoints/`; runtime payloads remain under
`<attempt>/model-checkpoints/`. If a process is lost before it writes a
terminal, `resume` may close the orphan only after nonblocking acquisition of
the lifetime lock and a second complete attempt-chain verification while that
lock remains held. It never infers death from PID identity or heartbeat age.

An ordinary runtime or checkpoint-observer exception writes `failure.json` and
a `HOST_EXECUTION_FAILED` terminal, then returns a `FAILED BackendExecution`
whose artifacts are those actual files. `BaseException` paths attempt the same
terminalization but are re-raised. If the failure or terminal cannot be
persisted (for example, disk failure), the host propagates the error and does
not manufacture a backend artifact.

`resume` accepts base and residual attempts. The predecessor checkpoint
must be under the same attempt root; the harness verifies its terminal and
checkpoint receipt, creates a distinct resume-linked attempt with
`AttemptStore.create_resumed`, and passes that exact `ResumeRecord` to the
training runtime. Residual resume also requires `--base-checkpoint` and
`--periodic-cache`; base resume rejects those arguments. The host verifies and
copies the latest and any older best checkpoint into the new attempt before
decoding owned authenticated bytes. Failed partial copies are retained, not
deleted by a path-based cleanup race. See [resume details](RESIDUAL_RESUME.md).

## Executable invocation

Install the package first. The
private-host module accepts its private host configuration before `--` and
passes everything after `--` through the public parser:

```bash
python -m phaseset_core.host --host-config private-run/host.json -- \
  preflight --config configs/phaseset/training.json

python -m phaseset_core.host --host-config private-run/host.json -- \
  run-base \
  --run-id phaseset-run-v1/BASE_QUALIFICATION/1729/B0 \
  --checkpoint-dir private-run/attempts/base-1729-b0-a/model-checkpoints \
  --config configs/phaseset/training.json

python -m phaseset_core.host --host-config private-run/host.json -- \
  run-residual \
  --run-id phaseset-run-v1/RESIDUAL_TRAIN/1729/08 \
  --base-checkpoint private-run/base/1729/selected-validation.pt \
  --periodic-cache private-run/cache/1729 \
  --checkpoint-dir private-run/attempts/residual-1729-08-a/model-checkpoints \
  --config configs/phaseset/training.json
```

A trusted Python composition root can equivalently call:

```python
from phaseset_core.host import main

raise SystemExit(main(PUBLIC_CLI_ARGUMENTS, config_path=PRIVATE_HOST_CONFIG))
```

The private host config schema is `phaseset-private-host-v1`.  It points to a
prepared index, an existing receipt record plus the exact artifact file for
each receipt digest, the installed source-tree digest, and runtime facts
(device, BF16 qualification decision, edge budget, checkpoint cadence, and
resume metadata).  Paths may be absolute or relative to the private config.
They are never serialized by the public CLI.

For residual runs, the optional `residual` object points to the one canonical
base-qualification artifact and one canonical capacity-audit artifact per
registered seed. The host process derives and verifies their SHA-256 values;
the operator does not transcribe individual hashes. `--periodic-cache` names a
seed directory containing canonical `record.json` (`PeriodicCacheRecord`) and
the digest-bound `cache-content.npz`. The payload must expose float64
`energy_floors[6]`. The host reads the payload once into a bounded buffer and
hashes and parses those same bytes. It verifies seed, winning-base run, base
terminal, qualification, and cache-content bindings before the public residual
factory is called.

The execution controller is responsible for creating the single private
receipt record from the user's already granted execution scope and the actual
rights/runtime/data measurements. This harness consumes and verifies that
record; it does not ask the user to sign each digest and does not create a
rights assertion when Embody approval is unknown.

## Deliberate gaps (no placeholder completion)

- Embody rights are not known. Until a real rights artifact participates in
  the receipt record, preflight/run-base/run-residual/resume remain held.
- The Embody-format conversion is outside this harness.  It must produce the
  exact prepared index/split/NPZ seam above and bind it to the authenticated
  prepared-data receipt.
- `run-residual` is wired through the strict qualified-base loader, canonical
  periodic-cache record, energy floors, canonical all-system capacity audit,
  closed residual constructor, and the same attempt ledger. Residual resume
  reconstructs those same dependencies and passes the verified predecessor
  record to the unchanged runtime. Historical cache-file SHA provenance is not
  retroactively claimed when only energy floors were consumed.
- `prepare-data`, `audit-split`, `qualify-base`, `build-periodic-cache`,
  `evaluate`, `bootstrap`, and `render-paper` are not
  implemented here.  Calling one raises before a `BackendExecution` exists.
- The public `resume` parser has no stop-after-step option. It accepts B0/B1/B2
  and registered final system IDs; the host implements base and residual resume
  and recovers identity from the verified attempt ledger.
- The current public training runtime does not accept a precomputed periodic
  descriptor cache. The host authenticates the complete cache payload and
  consumes its energy floors, but periodic descriptors are recomputed by the
  model path. Wiring cached descriptors into training is a lower-level public
  runtime gap and must not be represented as cache consumption in reports.
- A hard power loss before the first atomic runtime checkpoint has no resumable
  payload. The host refuses such a resume instead of manufacturing a checkpoint
  receipt. SIGKILL/power loss cannot write a terminal synchronously; terminal
  recovery occurs on the authorized resume operation.
- Runtime checkpoint fields do not expose separate sampler, dataloader, and
  dropout blobs.  The host receipt records the exact cursor/manifests and RNG
  payload under those ledger identities; a future public checkpoint-to-ledger
  helper would remove this local encoding convention.

`score_execution_census_sha256` is treated as the preregistered census of score
executions required by admission. It is never interpreted as a post-training
score table and therefore does not make base training depend on future scores.

## Observed tests and limits

On the Linux server, 12 host contract tests passed, including a genuine child
process lifetime-lock test and injected persistence failures. Four separate
tiny CPU fits passed checkpoint-observer integration: durable update/validation
callbacks, propagation of observer failure with a resumable checkpoint,
unchanged model/checkpoint/loss/validation values with an attached observer,
and a real checkpoint payload receipted by the attempt ledger. These use
`synthetic_contract=True` and do not assert production admission or results.

Report/failure JSON is file-fsynced and, on POSIX, parent-directory-fsynced
before a terminal is persisted. A failed fsync cannot create a success terminal.
Actual Embody rights, full model/runtime qualification, all registered runs,
and sealed test remain separate requirements.
