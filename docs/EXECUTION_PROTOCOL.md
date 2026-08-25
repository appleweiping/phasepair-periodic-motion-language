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

All nine base terminal rows are mandatory. The winner maximizes the exact
three-seed mean of validation bidirectional R@1. An exact tie is resolved by,
in order, fewer parameters, lower frozen-runtime latency, and smaller base
system ID. A complete valid nine-row table therefore always selects exactly
one winner; there is no seed-win threshold, minimum-delta gate, or negative
qualification branch. The winner's three seed-specific checkpoints are frozen.

The final score matrix has nine systems by three seeds:

| ID | Registered system |
| ---: | --- |
| 00 | qualified group base, scored from its frozen checkpoint without retraining |
| 01 | generic six tokens |
| 02 | marginal Morlet power |
| 03 | mean/difference DCT |
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
private adapter must authenticate external receipts and supply an unforgeable
capability; changing a JSON field is not authorization.

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

Inference uses exactly 100,000 paired capture-level bootstrap draws. All
captions belonging to a capture travel together. The deterministic resampling
stream is hashed, and the 2.5% and 97.5% order statistics are fixed. Reports
bind every hypothesis to its treatment and baseline system IDs and expose only
aggregate values and digests, never capture identifiers or raw caption rows.

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
