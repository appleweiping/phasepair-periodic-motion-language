# PhaseSet: Permutation-Invariant Periodic Relation Tokens for Multi-Person Motion–Language Retrieval

PhaseSet is a data-free research implementation of permutation-invariant
periodic relation tokens for multi-person motion–language retrieval. It is the
successor to the dyadic **PhasePair** prototype, whose audited source and
annotated `v0.1.0` tag remain preserved for reproducibility.

```text
multi-person skeleton sequence
  -> shared actor motion encoder
  -> six-band Morlet decomposition
  -> all unordered actor-pair phase relations
  -> actor–relation incidence topology
  -> six group periodic tokens
  -> text encoder and band-semantic projections
  -> group motion–language retrieval representation
```

## What PhaseSet changes

- Accepts dynamically padded groups with any valid `K >= 2`; the public schema
  does not impose a semantic maximum group size.
- Uses shared actor encoders, no actor ordinal embeddings, no actor-specific
  towers, and no actor-0 coordinate reference.
- Streams unordered edges in canonical microblocks and never materializes a
  dense `[K,K,...]` relation tensor.
- Aggregates incident half-edge means, population second moments, coverage,
  and missingness into a permutation-invariant topology residual.
- Hard-gates topology after the final topology MLP. At `K=2`, its value and
  parameter gradients are exact positive zero, and the full model equals the
  new pair-only control bit-for-bit under the frozen runtime.
- Retains the audited six physical Morlet center frequencies, 13D directed
  dyadic descriptor, swap algebra, and all golden tests. The legacy
  `phasepair_core` namespace keeps its immutable 30-Hz tap bytes; PhaseSet uses
  an independently hashed 20-Hz formula bank after resampling. This is a
  compatibility boundary, not a claim that the two kernel banks or full models
  are bitwise identical. See the
  [20-Hz rate contract](docs/PHASESET_20HZ_MORLET_CONTRACT.md).

## Current milestone

`v0.2.0-alpha.1` is the identity, multi-person contract, and migration
pre-release. Its authority is intentionally **data-free**: it contains public
code, synthetic tests, manifests, and execution contracts, but no real result,
private dataset, model weight, participant identifier, credential, or private
endpoint.

The registered study uses native `K>=3` Embody 3D captures, a strict
participant-disjoint `400/96/76` capture split, three base qualifications, and
eight periodic residual systems across three fixed seeds. Test membership and
restricted captions stay private; only irreversible digests and license-safe
aggregates may be released. Test evaluation is sealed and statistical claims
are generated from one frozen aggregate with capture-cluster bootstrap and
Holm correction.

See:

- [migration and compatibility boundary](MIGRATION.md)
- [frozen research contract](docs/PHASESET_RESEARCH_CONTRACT.md)
- [experiment matrix](docs/PHASESET_EXPERIMENT_PLAN.md)
- [execution and recovery protocol](docs/EXECUTION_PROTOCOL.md)
- [data preparation and licensing protocol](docs/DATA_PROTOCOL.md)

## Install and verify

```bash
python -m pip install -e ".[test]"
python -m pytest
phaseset preflight
```

Real-data and accelerator commands fail closed unless the required private
manifest, access receipt, environment digest, and frozen predecessor artifacts
are present. The runner never fabricates an external receipt or silently
substitutes synthetic data for a registered experiment.

## Public/private boundary

This repository may contain source, synthetic fixtures, public-safe metadata,
paper sources, editable diagrams, aggregate results, and cryptographic
receipts. It must not contain raw or reversible human data, SMPL-X assets,
download URLs, machine-fused sealed-test captions, private endpoints,
credentials, restricted checkpoints, or participant identifiers. Refer to
[SECURITY.md](SECURITY.md) before contributing or publishing artifacts.

PhaseSet is research software and does not currently assert a real-data
performance result. Negative or inconclusive registered outcomes remain valid
deliverables and will not be replaced by unregistered seed or test selection.
