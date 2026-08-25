# PhasePair: Periodic Relational Motion-Language Understanding

PhasePair is a research project for fine-grained understanding and retrieval of
two-person skeleton motion with natural-language descriptions. The central
question is whether explicitly motion-derived periodic relation tokens provide
repeatable retrieval value beyond a strong interaction-aware motion-language
encoder.

The intended high-level path is:

```text
skeleton sequence
  -> motion encoder
  -> periodic decomposition
  -> N periodic relation tokens
  -> text-semantic alignment
  -> semantic periodic representation
  -> motion-language retrieval
```

## Current public milestone

This branch contains the data-free PhasePair core, its tests, scientific
contracts, a provisional paper source, and an editable draft figure deck. The
core covers strict batching and lineage, deterministic motion dropout, three
motion-tower comparators, initialization, fixed signal descriptors, objectives,
evaluation, bootstrap inference, CLIP snapshot assessment, a sealed
authority-zero live CLIP text bridge, exact-identity runtime registries,
execution artifact schemas, a data-free trainer state machine, checkpoint
lifecycle validation, deterministic job planning, the registered residual-head
family, strict three-caption processing, a complete 30-epoch sampler manifest,
motion/caption training-batch cross-binding, full-gallery validation selection,
base-system qualification, and registered statistical-reporting primitives.

The presence of trainer and checkpoint code is not evidence that training has
run: no authorized dataset row has entered this repository, no production
optimizer or GPU execution has occurred, and no reportable checkpoint or result
has been produced. Caption/token arrays remain explicitly marked as
caller-frozen and unverified until the production tokenizer execution adapter
binds the original payloads to the exact tokenizer invocation.

This milestone is **not an empirical result**. It provides no dataset, CLIP
weights, runtime wheels, private server configuration, training checkpoint, or
paper claim. The paper tables remain explicitly held until the registered
server experiments finish.

## Public execution and review records

- [Current project status](STATUS.md) separates implemented public code,
  unresolved execution and human gates, and experiments that have not run.
- [Release manifest](RELEASE_MANIFEST.md) inventories this data-free release
  candidate and explicitly lists excluded or not-yet-produced artifacts.
- [ICASSP 2027 official-requirements checkpoint](docs/PHASEPAIR_ICASSP_2027_OFFICIAL_REQUIREMENTS_20260825.md)
  records the verified deadline and page boundary while refusing to relabel an
  older template as the not-yet-bound 2027 paper kit.

- [Detailed end-to-end execution objective](docs/PHASEPAIR_END_TO_END_EXECUTION_OBJECTIVE_20260825_002858.md)
  defines the complete delivery path, authorization boundary, run census,
  statistical family, visual requirements, release rules, and terminal
  completion checklist.
- [Public objective publication amendment](docs/PHASEPAIR_PUBLIC_OBJECTIVE_GITHUB_AMENDMENT_20260825.json)
  binds the added rule that local commits are not "uploaded" until the remote
  commit/tree and anonymous fresh-clone checks agree.
- [InterHuman B0 refresh](docs/PHASEPAIR_INTERHUMAN_B0_REFRESH_PUBLIC_20260825_012846.md)
  records the earlier public-safe split and duplicate census; it is retained as
  historical evidence and is superseded by the closure records below.
- [B0 annotation-caption closure v4](docs/PHASEPAIR_B0_ANNOTATION_CAPTION_CLOSURE_SUCCESSOR_PUBLIC_V4_20260825_145013.json)
  binds the independently rebuilt lineage, exact-code negative tests, strict
  DEFLATE boundaries, and withdrawn intermediate claims. Its technical gate
  count is zero, but `training_authorized=false`: owner-license attestation,
  topology owner signature, and qualified public-release clearance remain
  human gates. The bound [public lineage](docs/PHASEPAIR_ANNOTATION_CAPTION_LINEAGE_V2_PUBLIC_20260825_144456.json)
  and [required-code receipt](docs/PHASEPAIR_ANNOTATION_CAPTION_LINEAGE_V2_REQUIRED_CODE_SUCCESSOR_PUBLIC_20260825_144222.json)
  are published separately for machine verification.
- [Five-reference deep review](docs/PHASEPAIR_REFERENCE_PAPERS_DEEP_REVIEW_20260825_014557.md)
  records 72/72-page textual review, rendered figure inspection, the narrow
  PhasePair boundary, matched-control obligations, and the visual-language
  rules for the paper and editable deck.
- [Current closest-work refresh](docs/PHASEPAIR_CURRENT_CLOSEST_WORK_REFRESH_20260825_015706.md)
  verifies 11 primary-source neighbors and records which broad novelty claims
  are already unavailable, why MIME/WaMo/InterEdit/phase-stripped controls must
  remain separate, and how H1-H6 can falsify the remaining narrow claim.
- [CLIP rights and acquisition review](docs/PHASEPAIR_CLIP_RIGHTS_RESOLUTION_PUBLIC_20260825_004016.md)
  binds the exact eight-file private research snapshot and its conservative
  rights disposition without publishing model bytes or private locations.
- [Local CPython 3.12 CPU wheelhouse](docs/PHASEPAIR_LOCAL_PY312_CPU_WHEELHOUSE_PUBLIC_20260825_012601.md)
  records the 25-wheel, hash-locked offline dependency closure and clean
  installation audit.
- [Text-only runtime successor](docs/PHASEPAIR_LOCAL_PY312_CPU_WHEELHOUSE_PUBLIC_SUCCESSOR_V3_20260825_020011.md)
  records two independent empty-cache CLIP text preflights, including explicit
  parameter-storage materialization and zero-network/zero-vision checks. Its
  status is `PASS_CANDIDATE_ONLY` and does not authorize training.
- [Live CLIP text bridge report](docs/PHASEPAIR_LIVE_CLIP_TEXT_BRIDGE_REPORT_20260825_031124.md)
  publishes the public-safe implementation identity, stopped lifecycle, offline
  local-CPU evidence, and zero-AdamW boundary. It remains `AUTHORITY0`; server
  CUDA validation and every production training gate remain pending.

These records are evidence of planning, review, and gate status only. They do
not authorize training or imply a positive scientific result.

## Registered experiment census

- Base qualification: MIME-style, early-fusion, and late-fusion comparators.
- Seeds: `1729`, `2718`, and `31415`.
- Base runs: `3 systems x 3 seeds = 9`.
- Residual systems: generic, WaMo-style marginal-wavelet, InterEdit-style
  mean/difference-DCT, no-relation, phase-stripped, and PhasePair-full.
- Residual runs: `6 systems x 3 seeds = 18`.
- Final validation table: `7 systems x 3 seeds = 21` score rows.
- Confirmatory inference: paired source-cluster bootstrap with the registered
  H1-H6 family and Holm correction for H2-H6.

## Local data-free checks

The canonical authoring environment is documented in `docs/`. A compatible
development environment can run:

```powershell
python -B -m pytest -p no:cacheprovider -q
python -B -m ruff check --no-cache src tests
python -B scripts/release_manifest.py verify
python -B scripts/public_release_audit.py
python -B scripts/check_local_links.py
```

The three release checks use Git-index paths and blob bytes in an exact Git
checkout. In a GitHub codeload ZIP, they instead require
`RELEASE_FILES.sha256` to match the complete regular-file set, byte counts,
and SHA-256 digests before auditing content or links. They reject an empty,
tampered, incomplete, or extra-file archive instead of reporting a vacuous
pass.

The current public working tree was verified on 2026-08-25 on Windows x86-64
with CPython 3.14.5:
`571 passed, 3 skipped, 36 subtests passed`, followed by a clean Ruff run. Its
CPython 3.12 lane passed `570` tests with `3` platform/private-runtime skips and
one deliberate deselection of the test that asserts the exact 3.14.5 signal
oracle runtime. The wider private engineering workspace separately passed
`679` tests with `22` documented platform/runtime skips and `36` subtests.
These are scoped software checks; they neither authorize nor substitute for
real training. Ubuntu CI verifies the public tree, links, lint, manifest, and
wheel build only; the fixed Morlet byte oracle has not yet been qualified for
the target Linux server runtime.

## Public/private boundary

The following are intentionally absent and must remain absent:

- InterHuman samples, captions, identifiers, split membership, or reconstructable
  commitments;
- SSH endpoints, usernames, identity keys, tokens, or private paths;
- CLIP weights, runtime wheel archives, or other restricted third-party bytes;
- private receipts, checkpoints, raw logs, or unreviewed result files.

Only sanitized source, tests, contracts, aggregate evidence, paper sources,
editable figures, and reproducibility instructions are eligible for this
repository. Third-party data and model assets retain their own licenses and are
not relicensed by this project.

## Status

Active research and implementation. The publishable data-free core is present
at authority zero, but the tokenizer execution adapter, owner/license gates,
target-server runtime, authorized training, evaluation, statistical results,
final figures, and final ICASSP paper remain open. See [STATUS.md](STATUS.md)
for the time-scoped release state.
