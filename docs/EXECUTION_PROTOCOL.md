# PhaseSet execution protocol

Status: `AUTHORITY0 / DATA-FREE / NO SERVER ACCESS / NO RESULT`

This protocol fixes experiment identities, dependencies, selection, recovery,
and statistical rendering. It does not grant dataset, server, GPU, sealed-test,
publication, or scientific authority. The public configuration contains no
endpoint, dataset root, private digest, credential, or external grant.

## Frozen training census and DAG

The only seeds are `1729`, `2718`, and `31415`.

Base qualification trains exactly these three architectures for every seed:

| Base ID | Registered name | Runs |
| --- | --- | ---: |
| B0 | ActorMean | 3 |
| B1 | SetPMA | 3 |
| B2 | SocialTemporal | 3 |

All nine base rows are mandatory. Each row contains its canonical run ID, exact
integer numerator/denominator for validation bidirectional R@1, positive
parameter count, positive integer frozen-runtime latency in nanoseconds,
terminal SHA-256, validation-selected checkpoint SHA-256, validation-manifest
SHA-256, query-census SHA-256, evaluator/selector-code SHA-256, and unique
score-artifact SHA-256. All nine terminal, checkpoint, and score-artifact
digests must be unique, while the validation manifest, query census, and
evaluator identity must agree across all rows. Parameter count must
agree across the three seeds of one architecture. For each architecture,
resource latency is the integer median: sort its three integer nanosecond
measurements and take the middle value, never their mean.

The winner maximizes the exact three-seed mean of validation bidirectional
R@1. An exact tie is resolved by, in order, fewer parameters, the lower
three-seed integer-median frozen-runtime latency, and smaller base system ID. A
complete valid nine-row table therefore always selects exactly one winner;
there is no seed-win threshold, minimum-delta gate, or negative qualification
branch. The qualification artifact binds the full nine-terminal census, full
nine-checkpoint census, and the three winner terminal/checkpoint digests in
seed order `1729,2718,31415`. Its canonical schema carries the nine original
score rows and rebuilds every derived winner field before serialization. The
production adapter accepts a completed qualification only after it recomputes
that canonical artifact from the nine rows, confirms their evaluator digest
matches the installed validation/selector sources, and receives an external
authorization binding the manifest, query census, evaluator, and canonical
score-row digest. Every formal consumer must also receive an expected
qualification SHA-256 from an external trusted manifest and compare it before
cohort construction; a digest recomputed solely from the caller-supplied
qualification object is not a trust anchor.

For each residual seed, the strict checkpoint loader verifies the corresponding
winner checkpoint digest against that qualification artifact, reloads the exact
registered base architecture, and freezes both the group encoder and the
checkpoint's learned base logit-scale (inverse-temperature) scalar. The same
seed-specific frozen score state is used by system 00 and systems 01--08; it is
not reinitialized or retrained. The selected checkpoint must be a validation
checkpoint whose payload identifies itself, its finite validation metric, and
its own artifact digest as best at the moment it was written. That payload
cannot prove that no later validation checkpoint improved; final best status
therefore also requires the immutable completed-run terminal and trusted
qualification receipt.

The final score matrix has nine systems by three seeds:

| ID | Registered system |
| ---: | --- |
| 00 | qualified group base, scored from its frozen checkpoint without retraining |
| 01 | motion-independent fixed six band-ID tokens |
| 02 | marginal Morlet power |
| 03 | bilateral endpoint self-power DCT mean/difference |
| 04 | PhasePair pair-bag |
| 05 | coverage/missing-only |
| 06 | incidence-shuffled |
| 07 | phase-stripped full |
| 08 | PhaseSet full |

Systems 01--08 each train for all three seeds, giving 24 residual runs. The
formal training census is 9 base plus 24 residual runs, or 33 total. System 00
is a final evaluation row derived from the winning base checkpoint and is not
an additional training run.

Every base run requires the prepared-data manifest, split audit, and frozen
runtime preflight. Every residual run depends on all nine successful base
terminals and additionally requires the immutable base-qualification artifact,
its seed's winning-base checkpoint, its seed's cache, the prepared-data
manifest, split audit, and frozen runtime preflight. A run-plan row is only an
identity and cannot satisfy an artifact dependency.

## Frozen optimization contract

Base runs use AdamW, learning rate `2e-4`, weight decay `0.01`, 30 epochs, 5%
linear warmup followed by cosine decay, gradient clipping at `1.0`, and an
effective global batch of 128 windows. Residual runs freeze the selected base
and use AdamW, learning rate `3e-4`, weight decay `0.01`, 20 epochs, and the
same warmup, decay, clipping, and effective batch.

FP32 is the default. BF16 is admissible only after target-runtime
qualification, and all comparable systems must use the same qualified
precision. Trainable parameter counts for controls 01--07 must each remain
within the inclusive plus-or-minus 1% bound relative to full system 08.

Formal training always enters a frozen numerical context. Deterministic
algorithms are enabled with `warn_only=false`; float32 matmul precision is
`highest`; TF32 and reduced-precision reduction flags are disabled; cuDNN
benchmarking is disabled and deterministic mode is enabled. CUDA formal runs
also require `CUBLAS_WORKSPACE_CONFIG` to be `:4096:8` or `:16:8`. The
environment-v2 digest binds CPU thread counts, interop threads, OS/Python,
Torch and relevant build/runtime versions, CUDA device properties, cuDNN,
cuBLAS workspace selection, and the frozen flag policy. The runtime checks for
drift while active and restores the host's ambient Torch flags on exit.

System 01 enumerates every valid unordered actor pair and exposes all six band
IDs with valid support, independent of activity values, masks beyond actor
validity, Morlet support, or energy. System 03 transforms each endpoint
independently with registered cosine/sine quadratures, uses only symmetric
mean, absolute-difference, and normalized-difference endpoint powers, and has
no cross-endpoint product, phase, or lag. A DCT relation band is supported only
when both endpoints strictly exceed the same registered per-band floor.

The exact residual score is

```text
base_score = detach(
  exp(clamp(theta_base, 0, ln(100)))
  * cosine(frozen_base_motion, frozen_global_text)
)
periodic_score = masked_mean_valid_band(
  cosine(group_band_token, TextBandMLP(frozen_global_text, band_id))
)
score = base_score + tanh(lambda_residual) * periodic_score
```

`theta_base` comes from the selected base checkpoint and remains frozen.
`lambda_residual` and the shared text-band projection are residual parameters.

## Command order and fail-closed boundary

The public CLI exposes all eleven registered commands:

1. `preflight`
2. `prepare-data`
3. `audit-split`
4. `run-base`
5. `qualify-base`
6. `build-periodic-cache`
7. `run-residual`
8. `evaluate`
9. `bootstrap`
10. `render-paper`
11. `resume`

`python -m phaseset_core.cli preflight` prints every unresolved hold and exits
nonzero. Every other command validates its registered identity, then stops at
the same holds before external side effects. The public CLI never accepts a
literal server endpoint and never echoes caller-supplied paths. A future
private host must inject the supplied production adapter, authenticate external
receipts, and supply an unforgeable capability; changing a JSON field is not
authorization.

### Host-injected runtime adapters

The default command-line entry point never imports an adapter from a path,
environment variable, URL, or configuration field. A trusted host process may
inject a `RuntimeAdapter` object programmatically. The adapter must declare the
exact eleven-command census and a matching handler-manifest digest. Before any
handler is called, its immutable admission must bind the plan, matrix, training
configuration, adapter, handler manifest, data/prepared-data manifests, split
audit, rights assertion, runtime assertion, and execution assertion by SHA-256.
Each handler receives only a closed command intent and returns a closed result
containing artifact digests rather than paths or endpoints.

`PRIVATE_AUTHORIZED` means only that the injected adapter asserts it performed
external authentication. The public package validates schema, digest shape,
and binding consistency; it explicitly records
`public_verification_performed=false` and never claims that it authenticated a
real private receipt.

`SYNTHETIC_DATA_FREE` is an authority-zero execution mode for offline contract
tests. Its output is always marked `SYNTHETIC / NO SCIENTIFIC RESULT`. The
end-to-end fixture exercises tiny parameter updates, write-once checkpoints, a
failed terminal, new-attempt resume, gallery evaluation, 100,000 paired
bootstrap draws, and table/paper rendering. Synthetic rendering writes only to
its disposable test directory and may never replace `paper/main.tex` or any
registered paper artifact.

Completed production `qualify-base`, `evaluate`, `bootstrap`, and
`render-paper` commands cannot close with an arbitrary nonempty artifact.
Qualification rebuilds the nine-row selector output; evaluation rebuilds all
27 seed reports; bootstrap re-verifies the persistent one-use evaluation
ledger; and rendering rebuilds the resource report, claim decisions, CSV/JSON,
and LaTeX tables. The renderer accepts only an exact artifact-content set and
an external authorization bound to its evaluation, statistics, resource,
claim, renderer-code, and publication-manifest digests. Resource rows for all
27 systems/seeds must match the evaluated checkpoint, terminal, and environment
identities; the fixed K-scaling curve reuses system 08 seed 1729.

The stable public core also exposes `PhaseSetTrainingRuntime` for a trusted
host that already holds validated `PreparedGroupBatch` objects and frozen text
embeddings. It implements the registered AdamW schedules, variable-positive
symmetric InfoNCE, complete-effective-batch gradient caching over edge-budget
microbatches, validation-only selection, and immutable model/optimizer/
scheduler/RNG/cursor checkpoints. Base systems accept B0--B2; residual systems
accept every registered periodic encoder 01--08 and keep the qualified base in
evaluation mode with `requires_grad=false`. The runtime report remains
`authority=0`, `production=false`, and `result_claimed=false`; only the
authenticated host adapter can bind its artifact digest into an external
attempt receipt.

Formal construction is also fail-closed. Arbitrary callable factories are
accepted only by the synthetic contract. Formal runs must use the closed
`construct_registered_base_seed_bound_system()` or
`construct_registered_residual_seed_bound_system()` registry entry. They
install the registered seed before any trainable module is created, restore
the host's ambient Python/NumPy/Torch RNG state afterward, and emit a binding
over the internal registry factory, initial optimizable state, exact module
behavior and trainability graph, and system ID. Expected behavior is rebuilt
from an independent canonical registered factory; it is never trusted from a
mutable marker on the submitted model. Residual bindings additionally include
the qualified base checkpoint and loaded frozen-base state digests plus an
immutable all-system capacity-audit digest covering complete systems 01--08,
including the text head and logit scale. A non-synthetic runtime revalidates
the live behavior, initial state, frozen base, and cohort receipt before data
materialization or optimizer creation; checkpoints and resume preserve and
revalidate the same bindings. Building the audit requires the externally
trusted expected qualification digest. Consuming it requires both that digest
and an externally trusted expected capacity-audit digest, then freshly rebuilds
all systems 01--08 and compares every registered field and canonical byte.

### Streaming and control memory contract

Unordered periodic edges use a default runtime chunk of 256 edges. Every
runtime chunk size must be a multiple of the fixed 64-edge canonical
microblock, and reductions are defined by those 64-edge microblocks rather than
by the runtime chunk boundary. The edge stream never constructs a dense
`[B,K,K,...]` relation tensor.

Production system 06 leaves symmetric pair tokens unchanged. For each group
and each of the six bands independently, it first censuses all valid directed
half-edge endpoint slots, then applies one deterministic bijection across that
entire group-band census. Target endpoint slots and support masks remain fixed,
so degree, coverage, and the valid half-edge multiset are preserved while
actor--edge ownership changes. The routing plan stores only `O(E)` integer
actor/cursor metadata; it does not retain `O(E*D)` half-edge activations.

B2 SocialTemporal implements the exact registered self-attention algebra with
query chunks of at most 64 actors and small frame-row chunks. It materializes
only each bounded query-by-all-keys score block, never a complete `K x K`
attention-score map for the full actor set.

### Skeleton-gradient qualification boundary

The production periodic input boundary is deliberately the NumPy
skeleton-to-activity/Morlet descriptor streamer, so it does not expose
autograd to source skeleton arrays. A separate small-batch Torch CPU oracle
retains skeleton gradients through activity, Morlet, edge, topology, and
postprocess operations and is used for input-gradient permutation-equivariance
qualification. It shares registered learned parameters but is not the
production preparation path and is not evidence of CUDA/GPU qualification.

### Auxiliary caption-fusion admission

Caption fusion requires a frozen backend manifest binding the exact model
revision, backend implementation, inference runtime, deterministic decoding
fields, and strict response schema. For test provenance, sealed manifest and
caption digests are necessary but insufficient: before a backend request, a
trusted host gate must verify and consume an external grant and return an
admission bound to the exact provenance and consumption receipt. The public
serializer records only digests.

Three distinct actor orders are physically sent to the backend and normalized
outputs must match. This qualifies surface stability for those calls only.
Because generation and review use the same model family, semantic review stays
provisional and cannot close an execution, sealed-test, or scientific gate.

## Attempt, checkpoint, terminal, and resume contracts

An attempt has a unique canonical identifier. Attempt directories are never
reused. Metadata uses canonical JSON with one terminal LF and exclusive
write-once creation.

- `attempt.json` binds the run, plan, matrix, source tree, and training config.
- `heartbeats/heartbeat-NNNNNNNN.json` forms a contiguous predecessor-hash
  chain and cannot move `global_step` backward.
- `checkpoints/checkpoint-SSSSSSSSSSSS.json` strictly increases global step and
  binds model, optimizer, CPU/CUDA RNG, sampler, dataloader, dropout, and
  validation-selection state.
- `terminal.json` binds the attempt and latest heartbeat/checkpoint. A
  successful terminal requires a checkpoint; held and failed terminals require
  a closed failure code.

Once terminal, an attempt cannot receive another heartbeat, checkpoint, or
terminal. Resume creates a new attempt linked to a verified predecessor
terminal and checkpoint; it never edits or reopens the predecessor. Automatic
resume is limited to registered infrastructure, resource, implementation, and
data failures. Scientific and rights failures require review and cannot be
silently retried.

Checkpoint cadence is measured during the disposable overfit and frozen before
the nine base runs. It remains identical for comparable systems.

## Evaluation and registered inference

H1 compares system 08 with system 00. H2--H8 compare 08 with systems 01--07,
respectively. H1 is the sole primary hypothesis; exactly H2--H8 form the Holm
family.

Caption counts may differ between captures. PhaseSet first averages captions
inside each capture, then averages captures, so captures with more captions do
not gain hidden weight. Treatment and control rows must contain the same
caption census within each capture.

All 27 system-by-seed score tables are evaluated independently, and no
cross-seed logit average or ensemble is permitted. For every capture and
hypothesis, the paired treatment-minus-control contribution is computed once
per fixed seed and averaged in the exact order `1729,2718,31415`. Inference then
uses exactly 100,000 paired capture-level bootstrap draws. Only captures are
resampled; training seeds remain fixed repeated blocks and are neither
resampled nor treated as independent observations. All captions belonging to a
capture travel together. H1--H8 share the exact hashed resampling index stream,
and the 2.5% and 97.5% order statistics are fixed. Reports bind the full 27-row
checkpoint/score/evaluation digest census; per-system summaries are the
three-seed arithmetic mean and population standard deviation. Reports expose
only aggregate values and digests, never capture identifiers or raw caption
rows.

Group-Hard-32 independence is established only by an externally authenticated,
pre-score freeze binding over the scorer, exact similarity matrix, excluded
27-checkpoint census, manifests, and gallery collection. The internal ranking
comparison is a fail-fast heuristic, not proof of noncircularity.

## Sealed test

The test split has a maximum consumption count of one. The local sealed-test
ledger uses exclusive creation so a second consumption fails even in another
process. This ledger remains authority zero: it records local use but cannot
mint or verify an external test grant. Test evaluation is admitted only after
all validation decisions and aggregate code are frozen, an authenticated grant
exists, and the ledger remains unconsumed.

## External holds

Execution requires evidence this repository cannot self-issue:

- server inventory and target-runtime/precision qualification;
- dataset location, immutable source and prepared-data manifests, and rights;
- split/caption digests plus production tokenizer and adapter receipts;
- execution grant and GPU qualification;
- all nine base terminals and one complete base-qualification artifact;
- each winning seed checkpoint and its seed-specific cache;
- the sole sealed-test grant for final test access.

Missing, stale, contradictory, or unverifiable evidence is a `HOLD`. It is
never permission to fabricate a receipt, substitute synthetic data for a
registered run, or silently use an unregistered local path.
