# PhasePair CLIP acquisition and rights receipt — public sanitized

**Receipt schema:** `phasepair-clip-acquisition-public-redacted-v1`  
**Observed:** `2026-08-25 00:40:16 +08:00` / `2026-08-24T16:40:16.034Z`  
**Model:** `openai/clip-vit-base-patch32`  
**Immutable revision:** `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`  
**Scope:** P1 public acquisition, byte integrity, official-source rights review, and pre-install local runtime-archive inventory.  
**Private locator disclosure:** intentionally omitted. The snapshot remains in a Git-ignored private content-addressed store and is not a public artifact.

Private exact receipt witness: `12,015` bytes, SHA-256 `565d5110e7ee96c8b0c340d9c767be7185a606633fb4b22dc6723d61ffbfb5a3`. Its absolute locator is intentionally absent here.

## Outcome

- `PASS` — the exact public, non-gated revision was acquired without authentication.
- `PASS` — all eight runtime allowlist files match the frozen byte counts and SHA-256 values; the selected runtime snapshot has exactly 8 regular files, no child directory, no symlink/reparse point, no extra file, and no missing file.
- `PASS` — the exact official remote tree was inventoried as 12 files totaling `1,819,546,255` bytes.
- `CONDITIONAL_GO_WITH_MODEL_CARD_LIMITS` — local, non-deployed research use is the project operating policy.
- `NO_GO_ABSENT_EXPLICIT_HF_WEIGHT_LICENSE` — CLIP weights must not be committed, uploaded, mirrored, or redistributed.
- `HOLD_EXACT_RUNTIME_WHEEL_SET_ABSENT` — snapshot acquisition is complete, but the production text runtime is not yet authorized because no reusable exact archives were found for the four required runtime distributions.

The exact model API reports `private=false`, `gated=false`, commit `3d74...0268`, last modification `2024-02-29T09:45:55Z`, and omits `cardData.license`; the exact 12-file tree also contains no `LICENSE` file. These facts come from the [immutable Hugging Face API response](https://huggingface.co/api/models/openai/clip-vit-base-patch32/revision/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268?blobs=true) and [immutable revision tree](https://huggingface.co/openai/clip-vit-base-patch32/tree/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268).

## Selected runtime snapshot: 8/8 exact match

| file | bytes | SHA-256 | result |
|---|---:|---|---|
| `config.json` | 4,186 | `b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770` | MATCH |
| `merges.txt` | 524,657 | `f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85` | MATCH |
| `preprocessor_config.json` | 316 | `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd` | MATCH |
| `pytorch_model.bin` | 605,247,071 | `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` | MATCH |
| `special_tokens_map.json` | 389 | `f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf` | MATCH |
| `tokenizer.json` | 2,224,041 | `b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842` | MATCH |
| `tokenizer_config.json` | 592 | `34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad` | MATCH |
| `vocab.json` | 862,328 | `5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363` | MATCH |

Selected total: `608,863,580` bytes.  
Canonical selected-snapshot manifest: `1,134` bytes, SHA-256 `73594c210b9b7ae67090956f766720238db680057c29c47ca15f86d52441c2f7`.

The project-local sealed assessor independently performed its two complete scans. Its canonical AUTHORITY0 preflight is `1,352` bytes, SHA-256 `339b98e00a1b3b777b14add0ed1fc2e9bce37fcc4e8084dd1c09acdeab5848e1`, with `extra_files=[]`, `missing_files=[]`, and truthful status `HOLD_RUNTIME_WHEELS_ABSENT`.

## Exact official remote tree

The canonical 12-file remote-tree manifest is `2,262` bytes, SHA-256 `3919ef4931a77b1b0870ec11a4a00a25cee76b1ec7eaa81e1759cfa88de17839`.

| path | bytes | SHA-256 | evidence basis |
|---|---:|---|---|
| `.gitattributes` | 690 | `98cf30ae2568ea1d18641cc0d1d9a2f9041cc43aea81f5d719654abd9e290e0d` | exact resolve + local rehash |
| `README.md` | 7,942 | `8f0e8c678e0afe5f384f3bf8a6694c1cb53c41e52f86277967b6c5354cb0f9f4` | exact resolve + local rehash |
| `config.json` | 4,186 | `b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770` | exact resolve + local rehash |
| `flax_model.msgpack` | 605,123,003 | `2a2994d89bebd77abba5a554789dd9152a7e25467b79d88f6bd237d2dec5051c` | official LFS oid; excluded, not downloaded |
| `merges.txt` | 524,657 | `f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85` | exact resolve + local rehash |
| `preprocessor_config.json` | 316 | `910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd` | exact resolve + local rehash |
| `pytorch_model.bin` | 605,247,071 | `a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f` | exact resolve + local rehash |
| `special_tokens_map.json` | 389 | `f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf` | exact resolve + local rehash |
| `tf_model.h5` | 605,551,040 | `0d7e64ea2c496306a4bc0a6a7ab21e4755fd8c7beb01e6d9f5dc8d6662f1bfd2` | official LFS oid; excluded, not downloaded |
| `tokenizer.json` | 2,224,041 | `b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842` | exact resolve + local rehash |
| `tokenizer_config.json` | 592 | `34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad` | exact resolve + local rehash |
| `vocab.json` | 862,328 | `5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363` | exact resolve + local rehash |

Remote-only/excluded from the runtime snapshot: `.gitattributes`, `README.md`, `flax_model.msgpack`, and `tf_model.h5`, totaling `1,210,682,675` bytes. The exact remote tree has no added-token file, no SentencePiece/tokenizer model, no safetensors file or index, and no PyTorch shard/index file. The runtime snapshot intentionally contains only the eight frozen PyTorch/text/tokenizer assets, consistent with the [Hugging Face exact-revision download mechanism](https://huggingface.co/docs/huggingface_hub/en/guides/download).

## Rights and use review

The [exact Hugging Face model card](https://huggingface.co/openai/clip-vit-base-patch32/blob/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268/README.md) describes CLIP as a research output for AI researchers, treats deployed use as out of scope, excludes surveillance and facial recognition, and limits the evaluated language scope to English. PhasePair is therefore restricted to private, non-deployed research with English captions and no surveillance/facial-recognition use.

The [Hugging Face model-card documentation](https://huggingface.co/docs/hub/en/model-cards) identifies `license` as repository metadata, but this exact repository supplies neither that metadata nor a license file. The [upstream OpenAI CLIP repository license](https://github.com/openai/CLIP/blob/main/LICENSE) is MIT and speaks of the software and associated documentation. This receipt does not infer that the upstream software grant automatically licenses the separately hosted Hugging Face weight bytes. Public availability, a research-oriented model card, and an upstream software license are not treated as a weight-redistribution grant.

Operational verdict, not legal advice:

- private, non-deployed research: `CONDITIONAL_GO_WITH_MODEL_CARD_LIMITS`;
- model deployment: `NO_GO` under the current project scope;
- CLIP weight/checkpoint redistribution: `NO_GO_ABSENT_EXPLICIT_HF_WEIGHT_LICENSE`;
- public release: source URL, immutable revision, hashes, code, and this sanitized receipt only—no CLIP bytes.

## Pre-install runtime archive inventory

The local pip cache contains `1,351` HTTP body objects; `428` parse as wheel/ZIP or source archives. A metadata scan found zero archive for `torch`, `transformers`, `huggingface-hub`, or `tokenizers`; explicit `.whl` search also found zero target file. Installed distributions are recorded only as environment observations, not as archive evidence:

| Python | torch | transformers | huggingface-hub | tokenizers |
|---|---|---|---|---|
| 3.14.5 | 2.12.0+cpu | absent | 1.16.1 | 0.22.2 |
| 3.12.0 | 2.10.0 | 5.5.3 | 1.7.1 | 0.22.2 |
| 3.11.2 | absent | absent | absent | absent |

Consequently, the next runtime step must select and download a mutually compatible exact wheel set, record every archive byte count/SHA-256 and tag, and only then install into an isolated environment. No installation, Transformers model load, inference, vision construction, training, checkpoint creation, Git operation, or upload occurred in this P1 acquisition step.

## Source-byte witnesses

| official source | saved bytes | saved SHA-256 |
|---|---:|---|
| exact API response with blobs | 6,119 | `52e94841d1156a668f493a0aaf189e76053964cd012c0effc95259bdc476ce01` |
| exact Hugging Face model card | 7,942 | `8f0e8c678e0afe5f384f3bf8a6694c1cb53c41e52f86277967b6c5354cb0f9f4` |
| current Hugging Face download guide | 169,073 | `b5c127ed940c23e1f7f44a6fb60a116dd6c9db7d6471d654a348ac492d3f1b24` |
| current Hugging Face model-card documentation | 260,069 | `32fc4ad5bbf38c99c0438b8f68628081708da1547e76f25ad98c9c672c500fb0` |
| current upstream OpenAI CLIP license | 1,064 | `987e63b32f6c89ff5160e429458a872ff048e6860b590a3912e938f9da8f14db` |
| current upstream OpenAI CLIP model card | 7,733 | `7baf04f60c6234b301ec2c9ca39e67a3ca54b47c05e9509bddf732cbcbec8b7f` |

The upstream `main` sources are explicitly labeled current/mutable and are byte-hashed at observation time. They do not replace the immutable Hugging Face revision evidence.
