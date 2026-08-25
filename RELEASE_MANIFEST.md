# PhaseSet public data-free release manifest

**Manifest snapshot:** 2026-08-25 (Asia/Shanghai)
**Candidate:** `v0.2.0-alpha.1`
**Scientific authority:** `AUTHORITY0 / NO_REAL_DATA_RESULT / NO_CLAIM`

This manifest defines the public-safe PhaseSet migration artifact. It is not a
dataset-rights grant, server receipt, training result, completed paper, or
submission authorization. `RELEASE_FILES.sha256` is regenerated only from the
final staged Git index and is the byte/file-set authority for the candidate.

## Included families

- project identity: README, status, changelog, citation, license, migration,
  contribution, security, ownership, and package metadata;
- `src/phaseset_core/`: multi-person contracts, preprocessing, streamed
  periodic relations, shared group models, objectives, experiment registry,
  execution state machine, CLI, split auditing, and statistics;
- `src/phasepair_core/`: retained dyadic v0.1.0 compatibility code and goldens;
- synthetic/data-free tests for K=2 degeneration, actor permutations, dynamic
  padding, chunk invariance, gradient equivariance, topology witnesses,
  failure injection, resume, bootstrap, and public-release safety;
- public-safe configs for the fixed split census, three base candidates, eight
  residual systems, three seeds, optimization, sealed evaluation, and
  statistics;
- frozen scientific, data, execution, and experiment protocol documents;
- no-result paper sources and editable, explicitly non-empirical figures where
  present;
- release-audit, manifest, link-checking, and anonymous-codeload verification
  tools.

## Required verification before tagging

1. Legacy and PhaseSet unit/integration tests pass on the recorded runtime.
2. Ruff and package/wheel builds pass from a clean Git-index snapshot.
3. K=2 topology is exact positive zero in value and gradient, and new full
   equals new pair-only bitwise.
4. Exhaustive K=3..6 and stress K=8/32/128/256 permutation, padding, and chunk
   checks pass without dense actor-pair allocations.
5. CLI and runner fail closed when rights, private manifests, server/runtime
   receipts, or predecessors are missing.
6. Staged blobs and all refs pass credential, endpoint, private-path,
   restricted-binary, data, participant-identifier, and licensing scans.
7. `RELEASE_FILES.sha256` verifies the exact Git index, wheel source set,
   GitHub codeload, and release asset inventory.
8. The original PhasePair `v0.1.0` tag object and peeled commit are unchanged.

Exact pass counts, runtime identity, commit/tree/tag hashes, GitHub repository
ID, Actions run IDs, and codeload hashes are receipts and must be inserted only
after observation. Empty placeholders are not considered verification.

## Explicitly excluded

- raw or reversible human motion, images, video, audio, captions, participant
  identities, capture membership, or private commitments;
- Embody 3D/Multi-TPC/M3Act3D/AIOZ-GDANCE/InterHuman/Inter-X samples,
  download URLs, access forms, or restricted metadata;
- SMPL-X files, CLIP or other third-party weights, private checkpoints, caches,
  runtime wheelhouses, and licensed reference PDFs;
- machine-fused sealed-test captions, prompts containing private payloads, raw
  model responses, per-sample test predictions, or recoverable identifiers;
- endpoints, usernames, host keys, credentials, tokens, secrets, private
  receipts/logs, local absolute paths, agent state, or encrypted-backup keys.

## Version boundary

- `v0.1.0`: immutable PhasePair dyadic data-free release.
- `v0.2.0-alpha.1`: PhaseSet identity, contracts, and migration pre-release.
- `v0.2.0`: stable data-free multi-person core and runner, after qualification.
- `v1.0.0`: real experimental aggregate, compliant paper, and complete public
  provenance, only if every external and scientific gate is genuinely closed.

Negative results do not block a release when accurately represented. Missing
rights, fabricated evidence, leaked restricted material, or an unsealed test
does block it.
