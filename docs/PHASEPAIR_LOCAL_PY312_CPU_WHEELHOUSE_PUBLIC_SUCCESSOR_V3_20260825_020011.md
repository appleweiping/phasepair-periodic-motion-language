# PhasePair local CPython 3.12 CPU wheelhouse — schema-corrected v3 receipt

Date: 2026-08-25 (Asia/Shanghai)  
Objective: `phasepair-end-to-end-execution-objective-v3/20260825_002858`  
Objective bytes / SHA-256: `78,288` / `4e8977ec114de8657f607dc3be0fb2027485cc6d22b3a323e88114bb12c29f29`  
Status: `WHEELHOUSE_PASS / TEXT_PREFLIGHT_CANDIDATE_PASS / PRODUCTION_RUNTIME_HOLD`  
Authority: `0`  
Production: `false`  
Training authorized: `false`

## Exact correction and lineage

Fresh review of the stopped runtime successor found one schema-identity defect:
the materially expanded 7,925-byte preflight layout still self-identified as
`phasepair-local-py312-cpu-text-preflight-candidate-v1`, the same schema used by
the earlier 5,588-byte layout. That violated the project's strict exact-key /
schema rule even though every wheel and runtime check passed.

The correction is deliberately narrow:

```text
old schema: phasepair-local-py312-cpu-text-preflight-candidate-v1
new schema: phasepair-local-py312-cpu-text-preflight-candidate-v2
```

No layout, runtime value, model byte, wheel byte, parameter manifest, storage
result, or golden value changed. Two new isolated processes with two new empty
network caches reproduced the schema-corrected canonical JSON exactly.

Artifact lineage:

- the complete 25-wheel census, official PyPI sources, and 32-edge dependency
  closure remain in the
  [exact public base receipt](./PHASEPAIR_LOCAL_PY312_CPU_WHEELHOUSE_PUBLIC_20260825_012601.md)
  (10,866 bytes, SHA-256
  `2b2f62c473f3eecca1c08e9a992db7649ca275bd0b080e911b7b23aa7bc9baf7`);
- the detailed empty-cache and 197-to-149-to-197 storage analysis remains in the
  [runtime successor](./PHASEPAIR_LOCAL_PY312_CPU_WHEELHOUSE_PUBLIC_SUCCESSOR_V2_20260825_014253.md)
  (8,163 bytes, SHA-256
  `5317078b8dbd38e34ad0ac2a7e2f0d76a03deb98e3b8ceaca145e8053fde055c`),
  whose runtime facts remain valid but whose referenced preflight schema label
  is superseded by this receipt;
- this receipt is the active public schema correction.

## Schema-corrected repeated result

Both new processes exited 0 and independently observed:

- schema exactly `phasepair-local-py312-cpu-text-preflight-candidate-v2`;
- all 39 candidate checks true and no hold reason;
- a new empty cache before load and zero cache entries after load;
- zero network attempts under offline flags and explicit network blockers;
- all 8 exact snapshot files opened and unchanged, total 608,863,580 bytes;
- exact text loader/tokenizer types, exact reviewed config, 197 state keys;
- pre-materialization: 197 parameters, 149 storages, 24 shared Q/K/V groups,
  48 nonzero offsets;
- post-materialization: 197 parameters, 197 exact-extent storages, zero shared
  groups, zero offsets, contiguous tensors, zero pairwise overlap, and preserved
  tensor bytes/state keys;
- one `position_ids` buffer, `int64 [1,77]`, exact 616-byte storage, offset zero,
  and no parameter overlap;
- zero vision modules/parameters, zero active dropout modules/calls, 12 attention
  calls at dropout 0, and unchanged RNG state;
- identical token, pooled-EOS, pretrained-projection, and float64-normalized
  golden hashes.

The complete canonical JSON without its final LF is 7,924 bytes with SHA-256
`624b9434f7a1b525df940d0d1cbe3040f3b0cd4d9361c459d9b115052d37a89b`.

Current private identities are disclosed without locators:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Schema-corrected candidate script | 28,150 | `3feab11d5f51e2e1a82001e2800ebcff5481cce35bf6add0db3785ef986b4216` |
| Schema-corrected candidate receipt | 7,925 | `23e56043ef67fea7d1e3cd766028730fa6f7716289be572203a2c1fe2b59f7f9` |
| Private wheelhouse successor v3 | 5,447 | `5272b80449d37195aa6ad96aedf65e9cf0b25e801f5daafd2ff5d1294cd83d0e` |

The complete retained attempt history now has nine attempts. Attempts 8 and 9
are the two schema-v2 passes. The earlier signature failure, both storage HOLDs,
and the fresh-review schema finding remain preserved and are not relabeled.

## Capability and rights boundary

The exact local CPython 3.12 Windows CPU wheelhouse is `PASS`. The text-only
runtime is `PASS_CANDIDATE_ONLY`.

Production remains `HOLD`: the exact disjoint-storage materialization and
direct-registry validation must be promoted into the sealed live bridge and
fresh-reviewed before a production lease, optimizer, or training session can be
issued. The server CUDA wheelhouse remains `DEFERRED_FAIL_CLOSED` until actual
remote GPU/driver/OS/Python/CUDA evidence exists; no CUDA family was guessed.

The conservative CLIP rights verdict in the
[fresh-reviewed acquisition receipt](./PHASEPAIR_CLIP_RIGHTS_RESOLUTION_PUBLIC_20260825_004016.md)
is unchanged: private non-deployed research is conditional `GO`; deployment and
weight redistribution/mirroring are `NO_GO`. No model or wheel archive is
present in the public repository.

No optimizer was constructed. No training, GPU use, server access, Git push,
deployment, redistribution, or commercial authorization occurred. This is an
engineering and operational integrity record, not legal advice.
