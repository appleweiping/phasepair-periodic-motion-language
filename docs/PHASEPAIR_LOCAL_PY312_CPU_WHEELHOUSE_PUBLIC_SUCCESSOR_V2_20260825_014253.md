# PhasePair local CPython 3.12 CPU wheelhouse — v3 successor receipt

Date: 2026-08-25 (Asia/Shanghai)
Objective: `phasepair-end-to-end-execution-objective-v3/20260825_002858`
Objective bytes / SHA-256: `78,288` / `4e8977ec114de8657f607dc3be0fb2027485cc6d22b3a323e88114bb12c29f29`
Status: `WHEELHOUSE_PASS / TEXT_PREFLIGHT_CANDIDATE_PASS / PRODUCTION_RUNTIME_HOLD`
Authority: `0`
Production: `false`
Training authorized: `false`

## Successor scope

This receipt applies the objective-v3 requirements in Sections 26 and 35. It
supersedes only the runtime/preflight portion of the earlier
[exact wheelhouse receipt](./PHASEPAIR_LOCAL_PY312_CPU_WHEELHOUSE_PUBLIC_20260825_012601.md)
(10,866 bytes, SHA-256
`2b2f62c473f3eecca1c08e9a992db7649ca275bd0b080e911b7b23aa7bc9baf7`).
The predecessor's complete 25-row wheel table, official PyPI links, filename /
tag / bytes / SHA-256 values, and 32-edge dependency closure remain the exact
wheel-census base.

The wheelhouse was freshly revalidated without changing its bytes:

- target: CPython 3.12.0, Windows `win_amd64`, CPU only;
- 25 wheels, 153,244,336 bytes;
- 25 cached official PyPI release JSON documents, 924,608 bytes;
- 5 exact roots and 32 active dependency edges;
- zero missing/extra wheels, missing active dependencies, or unreferenced
  non-root distributions;
- all ZIP CRC, archive path, target-tag, and top-level distribution-metadata
  checks pass;
- all 25 wheels still match one official non-yanked PyPI `bdist_wheel` row by
  filename, bytes, and SHA-256;
- the 2,404-byte complete `--require-hashes` lock remains
  `e72315fc4f998273646553afe8f22574d81a8fca2472ebb1022fc571c87bd852`;
- isolated `--no-index --require-hashes` installation exited 0 and `pip check`
  reported no broken requirements; the existing Python environment was not
  modified.

## Why the runtime candidate changed

Objective v3 requires an empty-network-cache repeat and a live parameter/storage
inventory. Those checks exposed a real issue that the earlier aggregate
preflight did not test:

1. the default text-only load produced 197 distinct parameter objects but only
   149 underlying storages;
2. 24 Q/K/V storage groups were shared and 48 parameters had nonzero storage
   offsets;
3. the intervals did not overlap, but the live-runtime contract forbids shared
   parameter storage and nonzero offsets;
4. explicitly setting `low_cpu_mem_usage=False` did not change that outcome;
5. the candidate therefore materialized every registered parameter as a
   detached contiguous clone, preserving registry name order and
   `requires_grad`;
6. the resulting registry had 197 parameter objects, 197 storages, zero shared
   groups, and zero nonzero offsets, while all tensor bytes and all state keys
   remained identical.

This materialization is an observed, reproducible candidate solution. It is not
yet production authority: the exact rewrite and direct-registry validation must
be promoted into and fresh-reviewed with the sealed live bridge before a
production lease can be issued.

## Attempt history retained

| Attempt | Terminal observation |
|---:|---|
| 1 | Checkpoint load passed; the candidate stopped before forward because the internal 4.57.3 text-transformer signature rejects a caller-supplied `return_dict`. The argument was removed. |
| 2 | Legacy aggregate candidate passed. |
| 3 | Legacy aggregate candidate passed again with identical golden values. |
| 4 | v3 live-storage check correctly held: 197 parameters / 149 storages / 24 shared groups / 48 nonzero offsets. |
| 5 | `low_cpu_mem_usage=False` produced the same storage hold. |
| 6 | Explicit disjoint-storage materialization passed all v3 candidate checks. |
| 7 | A new process and a new empty network cache reproduced the entire receipt and all golden hashes exactly. |

No failed or held attempt was relabeled as a pass.

## Repeated empty-cache text preflight

The final two runs were separate processes. Each created a new empty Hugging Face
network-cache root, set `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, and
`HF_HUB_DISABLE_TELEMETRY=1`, loaded the exact private snapshot with
`local_files_only=True`, `trust_remote_code=False`, and
`low_cpu_mem_usage=False`, and ended with the cache still empty. Both runs
recorded zero network attempts.

Both full canonical outputs were identical: 7,924 bytes before the terminating
LF, SHA-256
`6d075b8450288d81a8f2a7b0f03f145e9ed28db84624ec597ac478e74c600a8b`.

Observed runtime facts:

- exact classes:
  `transformers.models.clip.modeling_clip.CLIPTextModelWithProjection` and
  `transformers.models.clip.tokenization_clip_fast.CLIPTokenizerFast`;
- exact runtime: CPython 3.12.0, Torch 2.10.0+cpu, Transformers 4.57.3,
  huggingface-hub 0.36.0, tokenizers 0.22.2, NumPy 2.4.6;
- all 8 frozen snapshot files were opened; before/after full scans stayed equal
  at 608,863,580 bytes;
- 0 missing, mismatched, or error state keys; all 202 unused checkpoint keys
  were limited to `vision_model.*`, `visual_projection.*`, or `logit_scale`;
- 0 vision modules, 0 vision parameters, and no `vision_model` attribute;
- exact reviewed text config: hidden size 512, intermediate size 2,048,
  12 layers, 8 heads, 77 positions, vocabulary 49,408, projection width 512,
  `quick_gelu`, attention dropout 0;
- 197 state keys with SHA-256
  `0c135012e876b7ebc7e36287434cb5f24cbcd1c2dd9e507b3135a33f5789dacc`;
- post-materialization 197-row parameter manifest: 66,645 bytes, SHA-256
  `4a2d1dde9f92f2eee6c17019b58fcc9636529f677fd9bc8b580992bdba3136eb`;
- one-row buffer manifest: 268 bytes, SHA-256
  `267c15fa50192124509fea524557e418f97e6cf877f5819a4c0483dab9c86fdd`;
- 63,165,952 trainable text parameters; the 262,144-parameter pretrained text
  projection was immediately frozen;
- 12 actual train-mode scaled-dot-product attention calls, all at dropout 0;
  active dropout modules/calls were 0 and the RNG state did not change;
- no optimizer was constructed and no training occurred.

The three-caption candidate golden remains byte-identical:

| Evidence | SHA-256 |
|---|---|
| Caption bytes | `dd2202c7fd1c1e7e49ecd1f77060e526f6c12d6eb9e991d83460273c541784bb` |
| Input IDs (`int64`, `[3,77]`) | `2c1bc6af4fe563c4b2dee0bc78bcd8bf2868ad3bd8f74d4126329f43a280cc8a` |
| Attention mask (`int64`, `[3,77]`) | `764c9a7807146b3dd2c20deae73cffe92db80069861a45d3ec3f0863c6b8f34e` |
| Final-LN pooled EOS (`float32`, `[3,512]`) | `07acbb2f6d1c803f12dfd937913d4f2165ca0ed08e248baef2a725a6e3c99a1a` |
| Frozen pretrained projection (`float32`, `[3,512]`) | `a4ec38be61e2db328652bd1119d4e8aba3f218a594562d0eb0060f07cf6e0f61` |
| L2-normalized projection (`float64`, `[3,512]`) | `26df05351d13d0e66048bb6f63c3fc84b62aced5d4c765cc6b18d4606201ee97` |

Private successor identities are published without their locators:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| v3 candidate preflight script | 28,150 | `d560ce44115269a3b1525f611976453c073058a03998672c47cf238ea1640eeb` |
| v3 candidate preflight receipt | 7,925 | `2713ae36ae3d4811354505a1a7932c931df893d57de5a7bbf9b5c61cf2261bbd` |
| v3 private wheelhouse successor receipt | 6,626 | `11e5cf50dc10c9d94fc851bda1fef44db24ab124ab11f43541a8a485f07b7ca5` |

## Rights and remaining HOLDs

PyPI integrity metadata does not grant rights in separately hosted model bytes.
The conservative CLIP verdict in the
[fresh-reviewed acquisition receipt](./PHASEPAIR_CLIP_RIGHTS_RESOLUTION_PUBLIC_20260825_004016.md)
is unchanged: private non-deployed research is conditional `GO`; deployment and
weight redistribution/mirroring are `NO_GO`. No weight or wheel bytes are in the
public repository.

Remaining boundaries:

- `PRODUCTION_LIVE_RUNTIME`: `HOLD` until the exact storage-materialization and
  direct-registry rules are promoted into the sealed bridge and fresh-reviewed;
- `SERVER_CUDA_WHEELHOUSE`: `DEFERRED_FAIL_CLOSED` until the actual remote GPU,
  driver, OS, Python ABI, and compatible CUDA requirements are observed;
- this receipt does not authorize an optimizer, training, deployment,
  redistribution, commercial use, server mutation, or GPU execution.

This is an engineering and operational integrity record, not legal advice.
