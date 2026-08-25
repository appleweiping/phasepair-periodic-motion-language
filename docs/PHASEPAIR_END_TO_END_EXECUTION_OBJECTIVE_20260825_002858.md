# PhasePair end-to-end execution objective

**Objective ID:** `phasepair-end-to-end-execution-objective-v3/20260825_002858`  
**Project root:** repository root
**Target venue:** ICASSP 2027  
**User authorization:** On 2026-08-25 the user explicitly approved all previously requested execution authority and instructed the team to complete the project without repeatedly requesting the same permission.  
**Current truth state:** implementation and document work exists; real PhasePair training results do not yet exist.  

**Superseding user objective:** The user subsequently confirmed that the required
outcome is the complete project and complete delivery, not a plan, prototype,
partial experiment, draft-only paper, or collection of HOLD documents. The
architecture and figure requirements below are therefore hard delivery
requirements. Private server coordinates and account details are consumed only
from the private execution configuration and must never be copied into public
documents, logs, paper sources, slides, receipts, or the GitHub repository.

This file is the persistent, detailed goal for every current and future agent. It
is an operational handoff contract, not a scientific result, license grant,
dataset redistribution permission, or substitute for evidence. Every agent must
continue from the latest verified artifacts and must not reduce the goal to a
prototype, a synthetic demo, a paper skeleton, or an unexecuted plan.

## 1. Mission

Complete a reproducible, honestly evidenced ICASSP submission project for
periodic two-person skeleton-motion/language understanding. The final project
must include all of the following, not a subset:

1. a novelty position that survives comparison with the closest motion-language,
   dyadic-motion, frequency/wavelet, phase-aligned retrieval, and repetition
   understanding work;
2. a fully specified and implemented PhasePair method with deterministic signal,
   model, optimizer, sampling, dropout, evaluation, bootstrap, checkpoint, and
   resume behavior;
3. a rights-aware, commitment-bound InterHuman data lineage and split audit;
4. an exact, offline, text-only CLIP runtime at the frozen revision, with live
   parameter provenance and no vision or hidden text dropout in training;
5. a verified remote GPU environment and documented invocation that a fresh
   agent can execute verbatim;
6. all preregistered sanity, base, residual, validation, statistical, and sealed
   test runs on the authorized server;
7. immutable logs, configs, checkpoints, receipts, metrics, and failure evidence;
8. editable top-conference-quality PowerPoint figures based on actual results;
9. a complete ICASSP-format paper, compiled PDF, verified citations, and no
   placeholder or unsupported result claim;
10. a sanitized public GitHub repository that contains reproducible code,
    contracts, manifests, reports, paper sources, and public figures, while
    excluding private data, credentials, proprietary assets, CLIP weights,
    checkpoints that cannot be redistributed, and other non-public material.

The work is complete only when the final completion checklist in Section 18 is
fully satisfied and independently reviewed. A green unit test suite alone, a
same-family review, a successful server login, one training run, a draft PDF, or
a public push alone is not completion.

## 2. Scientific hypothesis and method identity

The project tests whether explicit symmetric phase-pair relations improve
fine-grained retrieval between dyadic skeleton motion and language beyond a
strong MIME-style base and matched controls.

The canonical method family contains:

- two ordered motion passes, `AB` and `BA`, evaluated symmetrically;
- a paper-faithful MIME-style two-actor motion baseline;
- a fully specified early-fusion comparator;
- a fully specified late-fusion comparator;
- six periodic relation slots derived from symmetric and directional signals;
- a shared `13 -> 256 -> 512` residual projector over six relation slots;
- masked attention over relation tokens, followed by one residual coefficient
  `0.2` applied to the aggregated residual representation, not per slot;
- a multi-positive motion-language objective with exactly three captions per
  source and full-gallery evaluation;
- deterministic, commitment-derived ordering and dropout schedules;
- source-cluster paired bootstrap inference over three fixed seeds.

The method must not be renamed, weakened, or silently replaced after seeing
validation or test results. Any scientifically necessary amendment must be
timestamped, justified, reviewed, and applied before the affected run begins.

### 2.1 Literal architecture narrative

The user-specified high-level architecture is a non-optional narrative anchor:

```text
skeleton sequence
    -> motion encoder
    -> periodic decomposition module
    -> N periodic tokens
    -> text encoder / semantic alignment path
    -> semantic periodic representation
    -> motion-language understanding and retrieval outputs
```

Implementation diagrams and paper prose may clarify that motion and text are
encoded on aligned branches rather than claim an impossible serial tensor
operation, but they must preserve the user's intended relations:

1. the raw visual-side input is a temporal skeleton sequence, not RGB video;
2. the motion encoder produces a learnable motion representation;
3. an explicit periodic decomposition stage extracts several periodic tokens,
   rather than hiding all periodic reasoning inside an unnamed backbone;
4. the text path provides semantic supervision and a comparable embedding;
5. the periodic tokens are aggregated into a semantic periodic representation;
6. the final task is content understanding and fine-grained motion-language
   retrieval, not only period estimation or repetition counting;
7. the two-person ordered `AB`/`BA` relation and six PhasePair relation slots
   remain visible in the full method, even if the Introduction figure abstracts
   them to one clean periodic-reasoning module;
8. any change to module ordering, token semantics, number of slots, supervision,
   or output task requires an explicit scientific amendment before training.

The paper must explain the distinction between a period token, a relation token,
a motion embedding, a text embedding, and the final semantic periodic
representation. These terms cannot be used interchangeably merely to simplify a
figure. Every equation and diagram label must map to an implemented tensor and a
tested operation.

### 2.2 Literature and experimental quality bar

Before freezing claims, every relevant primary paper under the project's
`recommend paper` collection must be read in full, not only its abstract,
figures, or related-work paragraph. For each close paper, preserve a review row
covering:

- exact research question and task definition;
- input modality, dataset, split, sample unit, and leakage controls;
- encoder, fusion, temporal/frequency/phase representation, and objective;
- trainable/frozen components, initialization, optimizer, schedule, and seeds;
- baselines and whether they are truly capacity- and training-matched;
- evaluation protocol, metrics, statistical treatment, and failure analysis;
- claimed novelty and the exact boundary where PhasePair differs;
- figure language: layout, color semantics, connectors, density, hierarchy,
  caption burden, and what remains legible at paper size;
- reproducibility assets and any missing detail that PhasePair must improve.

The nearest-neighbor novelty review must include motion-language retrieval,
two-person/dyadic motion, frequency or wavelet motion representations,
mean/difference or DCT interaction models, repetition counting, phase-aligned
retrieval, and any newer work found before submission. A novelty statement is
accepted only when it is narrow, falsifiable, source-backed, and survives the
strong matched controls in the frozen experiment matrix.

"Match or exceed the reference papers" means the final project must provide at
least comparable rigor in data splits, baseline strength, run census,
reproducibility, metric definition, ablation design, uncertainty reporting,
qualitative evidence, figure quality, and disclosure of limitations. It does
not authorize inventing favorable results or silently changing the protocol.

## 3. Frozen system and run census

Seeds are exactly:

```text
1729, 2718, 31415
```

Base qualification systems are exactly:

```text
00 MIME
07 TMR_STYLE_EARLY_FUSION
08 TMR_STYLE_LATE_FUSION
```

Base run IDs are exactly the Cartesian product:

```text
phasepair-run-v2/BASE_TRAIN/<1729|2718|31415>/<00|07|08>
```

Therefore base run count is exactly `9`.

Residual systems are exactly:

```text
01 GENERIC
02 WAMO_MARGINAL_WAVELET
03 INTEREDIT_MEAN_DIFFERENCE_DCT
04 NO_RELATION
05 PHASE_STRIPPED
06 PHASEPAIR_FULL
```

Residual run IDs are exactly:

```text
phasepair-run-v2/RESIDUAL_HEAD_TRAIN/<1729|2718|31415>/<01|02|03|04|05|06>
```

Therefore residual run count is exactly `18`. Validation score rows are exactly
`3 seeds * 7 systems = 21`. Legacy `phasepair-run-v1`, base `06=TMR`, a
five-head/15-run residual census, or a merged WaMo/InterEdit head is invalid.

## 4. Hypotheses and statistical family

The hypotheses remain the preregistered family:

- `H1`: overall PhasePair-full vs the MIME base comparison;
- `H2`: PhasePair-full vs generic relation residual;
- `H3`: PhasePair-full vs WaMo-style marginal-wavelet residual;
- `H4`: PhasePair-full vs InterEdit-style mean/difference-DCT residual;
- `H5`: PhasePair-full vs no-relation residual;
- `H6`: PhasePair-full vs phase-stripped residual.

Holm correction applies to `H2-H6` exactly. The source-cluster paired bootstrap
uses exactly 100,000 canonical replicates and the frozen commitment-derived
index generator. Both retrieval directions and all preregistered metrics must be
reported. Negative or inconclusive findings remain valid project outcomes and
must not be hidden or rewritten as operational failures.

## 5. Evidence and truthfulness rules

All claims must be downstream of actual bytes and receipts. In particular:

- never report a score before the corresponding evaluator artifact exists;
- never report a run as successful merely because its process exited zero;
- never call an AUTHORITY0/HOLD object a production lease or grant;
- never substitute a static parameter row list for live parameter identity;
- never substitute a model name/revision string for verified local weight bytes;
- never infer dataset rights from file possession;
- never infer experimental support from a synthetic/data-free test;
- never turn scientific non-qualification into an operational training failure;
- never select a method, seed, checkpoint, caption, split, or hypothesis after
  reading sealed test results;
- never put private data, credentials, raw captions, raw motions, CLIP weights,
  or restricted checkpoints in the public repository;
- never leave placeholder metrics in a document presented as submission-ready.

Every stage records exact commands, software/runtime identity, input digests,
output digests, timestamps, and failure disposition. When a stage fails, read
the primary log, correct the actual cause, rerun the smallest valid gate, and
preserve evidence. Do not blindly relaunch an unchanged command.

## 6. User-authorized execution scope

The user has authorized the team to proceed without another permission round
for the previously gated work, including:

- public web research and model-rights verification;
- downloading the frozen public CLIP snapshot and required public wheels;
- reading the minimum necessary private InterHuman material for the B0 audit;
- producing private and sanitized public receipts;
- connecting to the configured private SSH server;
- creating or updating an isolated PhasePair environment;
- syncing only required code and public model assets;
- running CPU/GPU witnesses, synthetic tests, disposable overfit, base,
  residual, validation, bootstrap, and sealed test jobs;
- writing logs, checkpoints, results, figures, paper sources, and a PDF;
- committing and pushing the sanitized project to the configured public GitHub
  repository when the public tree passes the release audit.

This authorization does not waive third-party rights or confidentiality. The
team must still obey dataset/model licenses, keep credentials and restricted
assets private, avoid destructive overwrite of unrelated remote/local state,
and publish only material whose redistribution is permitted.

## 7. Phase A — core implementation closure

### Required work

- Preserve the latest clean PhasePair core modules and tests.
- Finish and independently review the live-text runtime boundary.
- Finish and independently review the v2 execution schema.
- Implement any remaining production-shaped but data-free components required
  by later phases: live admission, sampler-plan join, schedule verification,
  exact optimizer serialization, training state machine, checkpoint/resume,
  residual head, head dropout, job planner, and receipts.
- Keep production entry points fail-closed until their external evidence is
  present; private test seams may never mint production-compatible types.

### Done definition

- all target tests pass with bytecode/cache generation disabled where specified;
- full PhasePair core test discovery passes;
- Ruff passes on all current source and test files;
- fresh reviewers find no reproducible accepted-invalid path;
- current source/test bytes and aggregate ledger are recorded;
- all public statuses remain truthful.

## 8. Phase B — rights and private data B0

### CLIP/model rights

- Read the official model card, repository metadata, upstream license, and any
  use/restriction notices from authoritative sources.
- Freeze source URLs, access timestamps, page/file bytes when appropriate, and
  a conservative verdict for private research use and redistribution.
- A missing or ambiguous redistribution license must prevent publishing weights,
  but need not prevent authorized private research use when that use is lawful.

### InterHuman B0

- Bind the owner authorization and applicable research-use license/terms.
- Recompute archive/source/annotation/processed-motion censuses.
- Apply the frozen quarantine and final allowlist rules.
- Prove exact train/validation/test split membership and disjointness.
- Run the frozen cross-split near-duplicate screen.
- Build unconditional unique actor commitments and pair commitments.
- Build caption/source/lineage commitments without exposing raw private text.
- Record only aggregate counts, digests, and permitted metadata publicly.

### Done definition

- exact public/private receipts exist and reverify;
- every accepted sample is in the final allowlist with one lineage;
- quarantine, collision, missing-file, and cross-split leak counts are zero or
  explicitly dispositioned according to the frozen contract;
- the training-final census is frozen;
- no private raw data appears in logs or the public tree.

## 9. Phase C — frozen CLIP snapshot and live text runtime

### Snapshot

- Model: `openai/clip-vit-base-patch32`.
- Revision: `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`.
- Download into a private/gitignored content-addressed location.
- Verify complete tree, per-file bytes/SHA, expected eight pinned files,
  additional files, total bytes, symlink/reparse status, and offline reload.
- Never commit or upload CLIP weights.

### Runtime

- Pin exact Python, Torch, Transformers, huggingface-hub, tokenizers, and wheel
  archive bytes.
- Use a text-only loader equivalent to `CLIPTextModelWithProjection` and
  `CLIPTokenizerFast`, `local_files_only=True`, `trust_remote_code=False`.
- Prove vision instantiated count is zero.
- Expose final-LN pooled EOS and a frozen pretrained text projection for anchor
  construction only.
- Derive resolved trainable text rows from the retained live registry.
- Create the project-owned `text.project` and `text.logit_scale` with frozen
  initialization bytes.
- Prove active train-mode text dropout count is zero by actual forward trace.
- Freeze token IDs, attention masks, pooled EOS, pretrained projection, and
  normalized synthetic-caption golden bytes.
- Issue an opaque one-shot live lease only after fresh review of the observed
  registry/runtime manifest.

### Done definition

- no caller-supplied row/model/hash can substitute for live identity;
- snapshot load is repeatably offline and trace-closed;
- all live parameter/storage/state-key mappings revalidate;
- rights, runtime, golden, registry, and dropout receipts are fresh-reviewed;
- precondition failure produces zero AdamW constructor calls.

## 10. Phase D — remote environment

### Declarative spec

- Write a PhasePair-specific environment spec before building it.
- Record Python/NumPy/Torch/CUDA/Transformers versions, ordered installation
  phases, environment variables, cache roots, source snapshot, and GPU witness.
- Keep it distinct from the historical `cyclic-audit-py310` environment.

### Validation

1. imports and exact versions;
2. a seeded CUDA kernel-dispatch witness on a free GPU;
3. a fresh agent executes the documented invocation verbatim.

The environment ledger is keyed by the canonical spec hash. A changed spec is a
cache miss and requires rebuild/revalidation; an unchanged validated spec may be
warm-reused.

### Done definition

- remote GPU, storage, and environment capacity are recorded;
- source tree on the server exactly matches the local run manifest;
- CLIP reload works with network disabled from the intended private cache;
- tier 1, tier 2, and tier 3 validation pass;
- no job has yet consumed sealed test data.

## 11. Phase E — training admission and optimizer

Admission requires opaque, live, mutually bound leases for:

- verified run input;
- initialized motion model;
- resolved live text runtime;
- base sampler manifest set;
- base dropout schedule;
- first prepared training batch;
- current execution authority.

The session derives all live parameters internally and constructs one exact
AdamW with two groups, `decay` then `no_decay`, names in UTF-8 byte order.
Frozen vision/pretrained projection rows are absent. Motion, resolved text,
project, and logit rows each appear exactly once. No scheduler exists; LR is
constant. Step-zero and first-step optimizer states receive canonical and
framework-specific digests.

### Done definition

- current AUTH0/HOLD initializer is replaced by a reviewed production receipt;
- every pre-admission failure proves AdamW constructor count zero;
- one admitted inventory can construct at most one optimizer;
- live object/storage identity exactly matches the derived inventory;
- checkpoint/resume state schema is validated before real training.

## 12. Phase F — sanity and disposable overfit

Run the smallest valid gates first:

1. CPU synthetic end-to-end step and checkpoint/resume equivalence;
2. seeded GPU forward/backward/AdamW/checkpoint witness;
3. tiny real-data metadata/loader screen without optimization;
4. the preregistered disposable 64-cluster MIME overfit run, seed 1729,
   system 00, maximum 500 optimizer steps.

Overfit passes only when both train T2M R@1 and M2T R@1 reach at least `0.95`.
The disposable state is permanently retired and cannot seed full training.

### Done definition

- no NaN/Inf, divergence, missing gradient, hidden RNG, split leak, or receipt
  mismatch;
- checkpoint/resume equals uninterrupted execution in the same runtime;
- overfit criterion is met or an evidence-backed scientific/implementation
  failure is reported and fixed before full runs;
- GPU memory and throughput estimates update the full-run budget.

## 13. Phase G — nine base runs

Run exactly nine independent jobs in the frozen census. Same-seed systems share
only immutable inputs and initial resolved text bytes, never live parameters,
optimizers, gradients, checkpoints, or output namespaces.

Each run must record:

- immutable run input/config/source/runtime digests;
- epoch/batch/global-step mapping;
- training loss, LR, gradient norm, throughput, and memory;
- validation full-gallery metrics per eligible epoch;
- selected checkpoint under the preregistered selection rule;
- model/optimizer/checkpoint/log/evaluator receipts;
- terminal process and scientific status.

Training health is checked from W&B when configured, otherwise from preserved
logs. Sustained NaN/divergence or an unrecoverable contract failure stops the
affected job with evidence; ordinary metric noise does not.

### Qualification

After all nine immutable `BASE_TRAIN_COMPLETION` objects exist, compute the
preregistered three-model/three-seed strength gate. MIME must:

- have mean strictly above early and late means;
- beat the stronger same-seed comparator in at least two of three seeds;
- have minimum same-seed delta at least `-0.005`.

If not qualified, report the negative result honestly and do not launch
residual jobs. If qualified, build exactly one frozen MIME base cache per seed,
finalize nine terminal receipts, and mint the non-replayable qualified base
lease.

## 14. Phase H — eighteen residual runs

For each seed, train six independent `135,168`-parameter residual heads from
byte-identical initial parameter values but distinct objects/storage/optimizers.
The selected MIME base remains permanently frozen, in eval mode, and is read
only through the seed-owned cache. All six systems for a seed share the exact
same 20 epoch manifests and head-dropout schedule.

The run-local residual heads must not share parameters, optimizer state,
checkpoint namespace, log path, or output path. WaMo and InterEdit remain
separate controls. InterEdit reads no Morlet mask. System 06 is PhasePair-full,
not the legacy TMR identifier.

### Done definition

- exactly 18 terminal run receipts exist;
- all selection uses validation only and the frozen rule;
- no test access occurs;
- each run has complete model/optimizer/runtime/evaluator evidence;
- failed runs are rerun only after an evidence-backed fix, never silently
  replaced or omitted.

## 15. Phase I — validation, bootstrap, and sealed test

- Evaluate all 21 seed/system rows using the same full-gallery evaluator.
- Preserve commitment-derived tie order and the exact positive mask.
- Freeze per-source/per-cluster atoms before inference.
- Run the exact paired source-cluster bootstrap with 100,000 replicates.
- Report per-seed values, aggregate means/dispersion, effect sizes, confidence
  intervals, raw p-values, and Holm-adjusted decisions.
- Apply the preregistered terminal priority and gate logic.
- Open the sealed test split exactly once, only after method/checkpoints,
  hypotheses, evaluator, and analysis code are frozen.
- Record test-access count and prove no earlier test-derived selection.

### Done definition

- evaluator and inference inputs rehash to the frozen artifacts;
- all 21 rows and H1-H6 outputs are present;
- sealed test access is exactly one authorized event;
- conclusions match actual confidence intervals and adjusted tests;
- null or negative results are retained.

## 16. Phase J — figures, paper, and PDF

### Figures

- Use the Codex embedded PowerPoint authoring workflow and deliver native,
  editable `.pptx` sources. Text, equations, modules, boundaries, arrows,
  curves, and section frames remain editable vector objects wherever practical.
  Photographs, video frames, and complex textures may be raster assets, but the
  complete paper figure must not be flattened into one AI-generated bitmap.
- Read the current paper, frozen contracts, existing figures, the preceding
  paper's visual system, and the relevant figures in every close reference paper
  before drawing. A reference figure can constrain layout, palette, connector
  language, information density, or general style, but cannot silently alter
  PhasePair's module relations, hierarchy, or narrative.
- When editing an existing figure, record three regions before modification:
  `PRESERVE`, `EDIT_ALLOWED`, and `DO_NOT_TOUCH`. A request to change only a
  curve, arrow, border, gap, label, or alignment authorizes only that local
  change. It does not authorize a redesign of the whole figure.

#### Introduction and motivation figure

- Establish the problem, limitation, comparison, or core intuition within a few
  seconds at the intended paper scale.
- Prefer a simple narrative such as shared skeleton input -> existing approach
  limitation -> PhasePair periodic-relation insight -> correct semantic match.
- Show the essential contrast between generic motion encoding and explicit
  periodic semantic reasoning.
- Do not insert detailed backbones, every fusion layer, optimizer, loss family,
  checkpoint state, or low-level network block.
- Do not make this figure a miniature architecture diagram.

#### Method architecture figure

- Show the complete data flow from skeleton sequence through motion encoding,
  periodic decomposition, `N` periodic tokens, text semantics, relation
  aggregation, objective, and output.
- Distinguish the motion path, text path, PhasePair relation path, training-only
  supervision, and inference output through stable visual semantics.
- Use an overview plus a focused expansion when necessary; keep the main flow
  visually dominant and auxiliary supervision secondary.
- Preserve ordered `AB`/`BA` logic, the six relation slots, masking, attention,
  residual aggregation, and final embedding alignment at the correct level.
- Do not reduce this figure to a few slogan boxes.

#### Academic visual language

- Use a white or near-white background, dark-gray text, low-saturation accents,
  light group boundaries, and flat rational styling typical of strong CV/ML/AI
  papers.
- Limit the palette to one neutral family plus approximately three or four
  stable accent colors. Once a color denotes an input, feature family, module
  type, phase, or state, it retains that meaning throughout all figures.
- Outer group boundaries remain lighter than inner modules. Avoid equal heavy
  borders that make the figure look like a table.
- Prohibit large gradients, glow, glassmorphism, heavy shadows, neon saturation,
  cartoon people, cheap generic icons, button-like components, pill badges,
  dashboard card grids, excessive corner rounding, and decorative UI chrome.
- Add no meaningless labels, corner tags, repeated legends, ornamental lines,
  or colors whose only purpose is to make the page appear busy.

#### Layout and reading direction

- Establish one primary reading direction, normally left-to-right or
  top-to-bottom. Do not create competing directions.
- Place the shared input at the start. Parallel methods or branches use the same
  visual level, width, height, baseline, and internal spacing.
- Build a visible alignment grid shared by titles, modules, images, equations,
  outputs, and section boundaries.
- Leave enough space for the complete connector, including start, shaft, bend if
  any, and arrowhead. Increase spacing instead of compressing arrows into boxes.
- Upper/lower or overview/detail tiers must be connected by explicit arrows,
  labels, or structural mapping; large modules cannot sit as unrelated islands.
- Maintain a stable safe area around every page edge. No label, arrow, module,
  or output may touch or cross the canvas boundary.

#### Connectors

- Draw connectors before entity nodes when using native PowerPoint shapes so
  that edges remain visually behind nodes.
- Main-flow arrows share one dark-gray color, thickness, and arrowhead and are
  visibly stronger than ordinary borders.
- Auxiliary relations use a lighter gray; training-only or optional paths may
  use one other stable line style. One style has exactly one semantic meaning.
- Prefer straight lines. Use at most one natural orthogonal bend when needed.
- For shared input/output, use a short, clean bus, one expert-bank entry, or a
  unified fusion node rather than one page-spanning line per branch.
- Prohibit saw-tooth paths, page-wide T junctions, U-shaped returns, diagonal
  crossings, arrows through modules, arrows over text, arrowheads beyond box
  boundaries, detached arrowheads, and endpoints floating arbitrarily inside a
  module.

#### Typography and mathematics

- Keep visible text short: ordinarily one to five words per module. Put
  explanatory sentences in the caption or paper body.
- All peer modules use the same font size, weight, capitalization, alignment,
  dimensions, and padding. Titles, group labels, module names, auxiliary labels,
  and equations form a clear but restrained hierarchy.
- Shorten content or expand the layout before reducing font size. A title meant
  for one line must never wrap accidentally.
- Render subscripts and superscripts as actual typographic positions. Do not
  expose underscores, parenthesized pseudo-subscripts, or misplaced labels.
- Use consistent notation for variables, vectors, hats, transposes, indices, and
  ellipses. Every displayed formula sits beside the data flow or output it
  defines and is not decorative filler.

#### Curves, temporal signals, and media

- Periodic response, velocity, phase, and local-frequency curves must be smooth,
  continuous, and scientifically plausible, including the intended local
  changes. Do not substitute repeated vertical bars, a default sine wave, or an
  arbitrary decorative squiggle.
- Concept curves omit axes, grid lines, default chart frames, legends, and ticks
  unless those elements carry necessary meaning.
- Motion frames and experiment images retain their aspect ratio and show the
  complete relevant content. Do not crop away the subject or distort frames to
  fit a box.
- Consecutive frames maintain the same subject identity, clothing, viewpoint,
  and background unless the scientific comparison explicitly requires a change.
- Do not replace a requested scientific visual with an empty placeholder box or
  a toy icon.

#### Result plots and evidence

- Rebuild every result plot only from frozen aggregate result tables; never
  transcribe values manually from console output or a moving tracker.
- Display uncertainty and the exact metric definition where needed. Preserve
  negative, null, and mixed results.
- Ensure colors and line encodings remain distinguishable in grayscale and at
  the final single- or double-column print size.
- Put source artifact paths and hashes in each slide's speaker-note `[Sources]`
  block, including external claims and visual assets.

#### Mandatory visual QA

- Render every final slide and inspect every slide individually at full size;
  use a montage only for cross-slide consistency, not as the sole inspection.
- Check text overflow, accidental wrapping, image cropping, object overlap,
  connector direction, arrow intrusion, border crossings, curve continuity,
  color consistency, subscript placement, unresolved placeholders, and canvas
  overflow.
- Compare the final version side-by-side with the prior version and confirm that
  every `PRESERVE` and `DO_NOT_TOUCH` region remains unchanged.
- Recheck legibility at the paper's actual single- or double-column size and in
  grayscale.
- Run automated overflow tests, but never treat a clean object list or zero
  automated warnings as a replacement for rendered visual inspection.
- Any unauthorized structural rearrangement, arrow crossing, malformed symbol,
  cropped image, excessive prose, or obvious AI-template aesthetic is a failed
  deliverable even if the `.pptx` opens successfully.

The final visual priority order is fixed: information relations before
decoration; reading order before local beautification; consistency before color
variety; whitespace before forced density; minimal requested change before
unrequested redesign.

### Paper

- Move from the generic draft to the official ICASSP 2027 template.
- Technical content fits the official page limit; references/allowed material
  follow the current call.
- Replace every HOLD placeholder with an actual value or remove the unsupported
  sentence/table/claim.
- Verify equations, system names, seed/run counts, metrics, statistical family,
  citations, and figure references against frozen artifacts.
- Compile a clean PDF with embedded fonts, no missing references, no overflow,
  and no private path/secret/data leakage.
- Run independent scientific, citation, typography, and claim audits.

### Done definition

- editable PPTX and rendered visual QA are clean;
- paper source uses the official template;
- PDF builds reproducibly from the documented public command;
- every quantitative claim traces to a result artifact;
- title/abstract/conclusion state the actual evidence without overclaiming.

## 17. Phase K — public GitHub release

Before any push, construct an explicit public allowlist and scan the staged tree
for:

- credentials, SSH endpoints, usernames, key paths, tokens, private paths;
- raw/private motions, captions, annotations, commitments that reveal content;
- CLIP/model weights and restricted third-party assets;
- checkpoints or logs with non-public data;
- cache, bytecode, temporary, PDF reference-library, or large unintended files;
- stale manifests, old claims, placeholder results, and broken reproduction
  commands.

Commit only the sanitized allowlist. Push to a `codex/` branch first unless the
repository policy explicitly requires another branch. Verify the remote commit
and public file census after push. Continue pushing coherent reviewed milestones,
not moving or failing snapshots. The final public release contains source,
tests, configs, public contracts/receipts, paper source/PDF, editable figures,
and reproducibility instructions, but no prohibited asset.

## 18. Final completion checklist

The root agent must not mark the project complete until all items below are
true and independently evidenced:

- [ ] novelty and closest-work boundary is current and citation-complete;
- [ ] scientific contract and execution amendment are frozen and fresh-reviewed;
- [ ] full production code path exists for data, CLIP, model, optimizer,
      sampler, dropout, trainer, checkpoint/resume, evaluator, and statistics;
- [ ] all unit, integration, adversarial, GPU witness, and fresh-agent tests pass;
- [ ] P1 model/data rights receipts are complete and public-safe;
- [ ] P2 InterHuman B0 lineage/split/quarantine/commitment audit passes;
- [ ] exact CLIP snapshot and offline live text runtime receipt pass;
- [ ] PhasePair server environment passes tier 1, tier 2, and tier 3 validation;
- [ ] disposable overfit qualification passes;
- [ ] all 9 base runs have immutable terminal evidence;
- [ ] MIME strength gate is evaluated honestly;
- [ ] if qualified, all 3 frozen MIME caches and all 18 residual runs exist;
- [ ] all 21 validation rows and H1-H6 bootstrap/Holm outputs exist;
- [ ] sealed test was accessed exactly once after freeze;
- [ ] result tables, editable PPT figures, and rendered visual QA match;
- [ ] official-template paper source and final PDF pass independent audits;
- [ ] public release allowlist passes secret/private/rights/size scans;
- [ ] sanitized public GitHub push is verified at the intended remote commit;
- [ ] README, STATUS, MANIFEST, tracker, paper, figures, and receipts all agree;
- [ ] no remaining HOLD affects a claim presented as final;
- [ ] any negative or inconclusive scientific outcome is reported faithfully.

## 19. Agent handoff protocol

Every agent continuation must begin by reading this file, `AGENTS.md`, the
latest STATUS/MANIFEST, the current scientific contract/plan, and the exact
artifacts in its assigned scope. Each checkpoint must state:

```text
scope
files read or changed
commands actually run
entry and exit hashes for frozen artifacts
current pass/fail/HOLD status
new evidence or blocker
next concrete action
forbidden actions not taken
```

Agents must not restart completed work, trust an author summary over bytes,
promote a same-family review to final acceptance, or silently change a frozen
artifact. Moving files are never review candidates. Author and fresh reviewer
roles remain separate.

## 20. Immediate queue at this objective revision

The immediate order is updated to reflect work already completed and work that
is actively moving. No future agent may restart an earlier item merely because
it did not read the current status artifacts.

1. finish the private Windows CPU wheelhouse and exact offline CLIP text-runtime
   preflight, then freeze private and sanitized public receipts and obtain fresh
   independent QA;
2. close the InterHuman B0 `HOLD`: restore the receipt-bound live mount, complete
   the exhaustive 9,870,181-pair near-duplicate screen, mint the final allowlist
   and source/actor-v2/pair/caption/composite lineage receipts, and close the
   owner-license and one-archive-topology gates before any server staging;
3. read every page of every PDF in `recommend paper`, render and visually inspect
   the method/experiment/figure pages, and produce a page-grounded scientific
   and visual-language review plus a closest-work novelty matrix;
4. finish the production training admission, trainer, exact AdamW construction,
   checkpoint/resume, residual-head, job-planner, and ARIS execution-lifecycle
   path without weakening the clean data-free core;
5. keep retrying the authorized private SSH endpoint while other work proceeds;
   once reachable, inventory the actual GPU/driver/runtime before choosing any
   CUDA wheel or environment version;
6. build and fresh-review the PhasePair-specific server environment, then run
   CPU synthetic, GPU synthetic, checkpoint/resume, private-data loader, and
   disposable overfit gates in that order;
7. launch, monitor, and close exactly nine base runs, then evaluate the frozen
   MIME qualification rule;
8. if and only if the frozen qualification branch permits it, launch and close
   exactly eighteen residual runs, freeze all twenty-one validation rows, run
   the paired bootstrap/Holm analysis, and perform the one authorized sealed-test
   opening;
9. generate all tables and plots from frozen result artifacts, create the native
   editable Introduction/Motivation/Architecture/Results PowerPoint, render and
   inspect it, and insert the verified exports into the official ICASSP paper;
10. compile and independently audit the final PDF, then push every stable,
    sanitized milestone and the final reproducibility release to the public
    PhasePair GitHub repository.

This objective remains active until Section 18 and the expanded delivery ledger
in Section 37 are complete.

## 21. Non-negotiable interpretation of the user's instruction

The phrase "finish everything and deliver everything" has the following exact
operational meaning:

- continue performing authorized in-scope work rather than returning a plan;
- do not repeatedly ask the user to approve downloads, private evidence reads,
  SSH connection, isolated environment creation, server-side PhasePair jobs,
  paper compilation, figure creation, or sanitized GitHub pushes that are already
  covered by Section 6;
- when one dependency is temporarily unavailable, work on every independent
  item that can still move and retry the dependency at bounded intervals;
- prefer an evidence-backed implementation decision over a vague clarification
  request when the frozen contracts, primary sources, repository, private
  locator, or runtime observation already determine the answer;
- fix ordinary implementation, environment, test, formatting, and integration
  failures directly, preserve their evidence, and rerun the smallest meaningful
  gate before expanding;
- never use "authorization", "fresh review", "HOLD", or a self-created ceremony
  as a way to avoid doing ordinary work;
- never treat the user's broad operational approval as permission to violate a
  third-party license, expose credentials/private data, overwrite unrelated
  state, fabricate results, bypass a scientifically frozen branch, or claim
  evidence that does not exist;
- ask the user only when a genuinely new choice is both undiscoverable and
  materially outcome-changing, or when the only remaining action would require
  new legal authority outside the already granted project scope.

"Directly make it work" does not mean hiding a failure. It means diagnose the
failure, repair what is repairable, use an in-scope alternative when sound,
continue other useful work in parallel, and report the exact remaining external
dependency without padding the report with process theater.

## 22. Workspace, source-of-truth, and artifact-lifecycle map

### 22.1 Workspace boundaries

- Canonical working project: private development workspace outside this repository.
- Sanitized public mirror: this repository.
- Public repository:
  `https://github.com/appleweiping/phasepair-periodic-motion-language`.
- The current shell may start in another project. That does not change the
  PhasePair root. Every write must resolve to an explicitly named PhasePair path
  or the sanitized mirror; unrelated workspaces must not be modified.
- Private assets, credentials, server locators, wheels, model bytes, raw data,
  restricted receipts, and non-public results live only under the established
  gitignored private boundary. No public document may reveal its absolute
  internal paths or access details.

### 22.2 Source-of-truth precedence

When two artifacts disagree, resolve in this order and record the resolution:

1. the user's latest objective and explicit scientific intent;
2. the latest fresh-reviewed scientific-contract family;
3. exact primary-source/runtime/data bytes and their receipts;
4. the latest fresh-reviewed implementation and tests;
5. current STATUS/MANIFEST/tracker artifacts;
6. author summaries, commentary messages, and older drafts.

No summary overrides bytes. No old fixed alias overrides a newer explicitly
named clean family. No paper sentence overrides the implemented/frozen contract.

### 22.3 Lifecycle vocabulary

- `MOVING`: author is still changing bytes; not reviewable or publishable.
- `CANDIDATE_STOPPED`: author has stopped and supplied exact size/SHA identities.
- `FRESH_QA_CLEAN`: an independent reader checked the stopped bytes and found no
  reproducible blocker in the assigned scope.
- `REVISE`: a reproducible discrepancy or accepted-invalid exists; the stopped
  family remains immutable and a successor or narrow code revision is required.
- `SUPERSEDED`: retained for lineage but not active authority.
- `AUTHORITY0`, `HOLD`, `NO_RESULT`, `DATA_FREE_NONPRODUCTION`: truthful negative
  capability labels; none authorizes real training or a scientific result.
- `TERMINAL_SUCCESS`: process and scientific completion both satisfy the exact
  stage contract; process exit zero alone is insufficient.
- `TERMINAL_FAILURE`: immutable failed attempt with cause and disposition; it is
  never silently deleted or relabeled as success.

Frozen evidence is never edited in place. A material correction receives a new
timestamp or a new exact code STOP identity, an explicit predecessor relation,
and fresh QA.

## 23. Exact reference-paper review obligation

The local reference library currently contains the following five required
primary PDFs. Every page must be read, and visually important pages must also be
rendered and inspected rather than inferred from extracted text:

1. `reference-artifact://phasepair/2503.17690v2.pdf`, 3,283,577 bytes,
   SHA-256 `073e8bd9edda27926612026578318cda2c28141ea4a0473ada55dce8705119f2`;
2. `reference-artifact://phasepair/Jiang_MotionMaster_Generalizable_Text-Driven_Motion_Generation_and_Editing_CVPR_2026_paper.pdf`,
   4,484,508 bytes,
   SHA-256 `18cae7971aebb66c6d3ecac163c1962b9da842923fb426c8af5df3385b739e97`;
3. `reference-artifact://phasepair/Liao_LangPose_Language-Aligned_Motion_for_Robust_3D_Human_Pose_Estimation_WACV_2026_paper.pdf`,
   3,213,424 bytes,
   SHA-256 `9d54395f4dea71223074b5b56ac95530ff93aefa11fc897b30a1fc60809e19c7`;
4. `reference-artifact://phasepair/NeurIPS-2025-hmvlmhuman-motion-vision-language-model-via-moe-lora-Paper-Conference.pdf`,
   5,530,658 bytes,
   SHA-256 `74be15eef3a5dbc0567ff21de8492e876d94860f6e21c3988f5db0c101ece758`;
5. `reference-artifact://phasepair/Rongali_Pose2Lang3D_Distilling_3D_Reasoning_from_2D_Skeletons_via_Language_Supervision_CVPRW_2026_paper.pdf`,
   675,267 bytes,
   SHA-256 `286923f6dd06479768cd1dc1483faf62f99335e38e107ea98ed4b38c7a1612ea`.

For each paper, the permanent review must include page-specific evidence for:

- problem, task unit, input/output, datasets, splits, preprocessing, and rights;
- architecture blocks, dimensionalities, representation choice, loss, and
  training regime;
- trainable/frozen components, initialization, optimizer, schedule, batch size,
  epochs/steps, hardware, seeds, and model-selection rule;
- baselines, parameter/capacity matching, ablations, metrics, uncertainty,
  statistical testing, qualitative analysis, and limitations;
- the strongest result, the most important negative result or omitted control,
  and the exact reproduction ambiguity;
- Introduction and Method figure composition, reading order, palette, connector
  semantics, label density, caption dependence, and print-scale legibility;
- exact novelty overlap with PhasePair and one falsifiable sentence explaining
  why PhasePair is not merely a renamed combination of that paper's components.

The review deliverables are:

- one byte/hash-bound page-by-page review report;
- one closest-work comparison matrix with claims, datasets, modules, controls,
  and evidence gaps;
- one experimental-rigor checklist that maps every reference strength to a
  PhasePair experiment or explicit non-applicability reason;
- one visual-language report containing rendered page references and a list of
  visual rules to adopt or reject;
- one citation ledger using verified primary metadata, with uncertain metadata
  marked for verification rather than guessed.

Before submission, browse authoritative primary sources for work published after
the frozen local review and for current venue rules. New nearest neighbors are
added to the novelty matrix; they do not silently change the preregistered
experiment family after results are seen.

## 24. Implementation closure and cross-module contract

The production path must join, not merely colocate, all of these concerns:

```text
private lineage + split allowlist
    -> prepared two-actor motion batch
    -> deterministic AB/BA motion passes
    -> initialized MIME/early/late motion tower
    -> exact runtime dropout schedule
    -> periodic and directional signal descriptors
    -> six relation tokens / matched-control tokens
    -> frozen or trainable text runtime according to the exact row inventory
    -> multi-positive objective
    -> exact two-group AdamW
    -> trainer state machine
    -> atomic checkpoint/resume
    -> full-gallery evaluation
    -> paired source-cluster inference
    -> immutable execution completion
```

The following module families must remain mutually consistent: batching,
signal, lineage, contracts, dropout, runtime dropout, torch models,
initialization, objectives, sampler, evaluation, bootstrap, readiness, CLIP
resolution, sealed text runtime, execution schema, training admission, trainer,
checkpoint/resume, residual heads, job planning, ARIS lifecycle integration, and
public receipts.

For every public or production-shaped API:

- exact input types, shapes, dtypes, layouts, finite constraints, value domains,
  key order/census, and ownership rules are validated before dynamic reads;
- caller arrays/tensors are rejected or snapshotted before validation so no
  validate-then-copy race can change accepted bytes;
- output dataclasses cannot be forged into canonical evidence by direct
  construction, equality tricks, subclasses, mutable globals, helper rebinding,
  or post-construction mutation;
- module-private semantics used by canonical entry points are captured or
  independently revalidated so ordinary global rebinding cannot broaden
  authority;
- state transitions burn correctly on every admitted failure and cannot be
  replayed, reset by ordinary mutation, or consumed concurrently twice;
- canonical receipts are strict, domain-separated, exact-key JSON with linked
  source/runtime/input/output digests and explicit authority/status fields;
- synthetic seams and diagnostics emit types/statuses that can never be accepted
  by production admission;
- tests cover success, each failure cut, stale/reordered/cross-bound evidence,
  type-forgery, subclass, aliasing, mutation, concurrency, partial-write,
  rollback, helper/global rebind, and canonical-byte stability.

No agent may rewrite a clean module merely to make it look more "production".
Only a concrete integration need or reproducible defect justifies modification,
and the affected stopped bytes must be re-reviewed.

## 25. InterHuman private-data B0 execution detail

The prior safe aggregate baseline is approximately:

```text
official split union                 7,779 sources
paired motion                        7,777 sources
strict usable                        7,280 sources
conservative duplicate quarantine      157 sources removed
final motion allowlist               7,123 sources
final caption rows                  21,369 captions
final train/validation/test       5,564 / 517 / 1,042 sources
```

These numbers are an expected baseline, not permission to skip the live refresh.
The current B0 audit must independently recompute or exact-reverify:

1. owner/license evidence and the allowed private, noncommercial research scope;
2. archive and extracted-tree identity, source and annotation census, processed
   motion identity, and official split definitions;
3. source-to-pair-to-actor lineage with no ambiguous or multiply claimed item;
4. strict tensor/schema/length/finiteness validation before any model access;
5. quarantine reasons and exact membership, including missing pairs, malformed
   samples, actor collisions, duplicates, near-duplicates, and split leakage;
6. unconditional globally unique actor commitments, pair commitments, caption
   commitments, and source-cluster commitments;
7. three captions per accepted source or the exact disposition of a source that
   cannot satisfy that invariant;
8. train/validation/test disjointness after quarantine;
9. private full receipt and sanitized public aggregate receipt;
10. staging manifest whose paths are private but whose allowed counts and digests
    can be compared on the server without leaking content.

Dataset bytes, captions, motions, identifiers, and restricted annotations are
never placed in GitHub, paper supplementary material, ordinary chat output, or
public logs. The current rights expectation permits private noncommercial
research under the applicable terms; it does not permit raw redistribution,
commercial use, public weights that memorize restricted content, or publication
of private samples without a separate rights basis.

## 26. CLIP acquisition, wheelhouse, and live-runtime detail

The exact CLIP identity is fixed by Section 9. The acquisition stage additionally
requires:

- a private content-addressed snapshot with the exact eight runtime files, all
  expected official-tree entries dispositioned, no symlink/reparse alias, and an
  offline rehash after installation/preflight;
- a private wheelhouse selected from official non-yanked distribution metadata,
  with filename, tag, bytes, SHA-256, dependency edges, and archive integrity;
- a `--require-hashes` lock and a fresh isolated environment installed with
  `--no-index`, followed by package import/version evidence and `pip check`;
- separate local-CPU and server-CUDA runtime receipts. The CUDA wheel family is
  never guessed from the local machine; it is selected only after observing the
  actual server driver/GPU/runtime compatibility;
- an actual text-only offline load, tokenizer call, synthetic-caption forward,
  EOS pooling, pretrained projection, normalization, live registry inventory,
  dropout trace, and repeated empty-network-cache reload;
- a fresh review that checks model/tokenizer class, config fields, state keys,
  live parameter/storage identity, output bytes, and absence of vision
  construction or remote code;
- private provenance and public-safe aggregate receipts. Model weight bytes and
  wheels remain private and gitignored.

The current rights stance is deliberately conservative: private non-deployed
research use may proceed under the recorded evidence; redistribution of hosted
weight bytes is `NO_GO` unless a later authoritative license grants it. The
public repository may contain download/verification code and hashes, not the
weights.

## 27. Remote server and environment operating procedure

The authorized server endpoint, account, key, remote roots, and environment
locator are read only from the private configuration. They are never repeated in
public artifacts or normal handoff prose.

### 27.1 Connection handling

- Use the exact configured endpoint and host-key policy.
- Do not substitute a different open SSH port, username, key, or host when the
  configured service is temporarily refused.
- A TCP refusal before authentication is an endpoint/service availability fact,
  not an authentication failure and not a reason to rewrite credentials.
- Retry at bounded intervals while local literature, code, rights, paper, and
  figure work continues. Preserve concise attempt timestamps and outcomes.
- On first successful connection, verify host identity before any mutation and
  perform a read-only inventory first.

### 27.2 Read-only inventory before build

Record OS/kernel, shell, disk quotas/free space, CPU/RAM, GPU models/count,
driver, CUDA capability, current utilization, scheduler availability, Python and
conda/mamba tools, existing PhasePair roots, caches, and permissions. Do not
reuse the historical cyclic-audit environment as evidence for PhasePair.

### 27.3 Isolated build

- Create a uniquely named PhasePair environment and code snapshot; do not
  overwrite unrelated environments or user projects.
- Transfer only the sanitized code snapshot plus private assets that are required
  and licensed for this execution.
- Verify local/server source manifests before running tests.
- Install from exact private wheelhouses with hashes and no opportunistic
  dependency upgrades.
- Store model/data caches and run outputs in explicit private roots with enough
  free-space headroom.
- Record environment variables, library versions, build commands, wheel hashes,
  import paths, GPU witness, and resulting environment digest.

### 27.4 Scheduling

GPU selection follows measured free memory and scheduler policy. Never
oversubscribe a device, kill another user's process, seize an unrelated job, or
change system-wide drivers. Parallel job count is derived from measured peak
memory plus a safety margin, not from the number of visible GPUs alone.

## 28. ARIS execution and run-directory contract

Every real experiment must pass through the current validated ARIS lifecycle and
produce evidence that binds authority, inputs, runtime, model, execution,
cleanup, and publication. The existing `AUTHORITY0` near-production integration
families are verification references, not real execution grants.

Each real attempt receives an immutable execution ID and a unique attempt
directory conceptually equivalent to:

```text
<private-run-root>/<phase>/<seed>/<system>/<attempt-id>/
    run_input.canonical.json
    source_manifest.json
    environment_manifest.json
    data_manifest.json
    model_manifest.json
    optimizer_manifest.json
    command.txt
    stdout.log
    stderr.log
    heartbeat.jsonl
    metrics.jsonl
    checkpoints/
    validation/
    terminal_status.json
    execution_receipt.json
    failure_receipt.json            # only when applicable
```

Rules:

- directories are never reused or overwritten by a rerun;
- moving files use a temporary name and atomic final rename where supported;
- every canonical JSON file is strict UTF-8, deterministic, exact-key, and
  terminated consistently;
- `command.txt` contains the exact invocable command with secrets redacted;
- stdout/stderr are preserved even on failure and never treated as the sole
  metric source;
- heartbeat records wall time, global step, epoch, last checkpoint, GPU memory,
  throughput, and health without containing private samples or captions;
- checkpoint filenames include immutable step/epoch identity and digest;
- terminal status is written exactly once after cleanup and cannot be mutated by
  a later attempt;
- a rerun links to the failed predecessor attempt and a concrete corrective
  change; unchanged blind reruns are prohibited;
- ARIS publication/cleanup timing is not misdescribed as proof that a result was
  generated after some unrelated event unless the evidence actually establishes
  that chronology.

## 29. Trainer, optimizer, checkpoint, and resume contract

Before the first real optimizer constructor call, all admission evidence must be
present, live, mutually bound, unconsumed, and applicable to the exact run ID.
The trainer must derive rather than trust caller-supplied parameter rows.

### 29.1 Optimizer

- exactly one AdamW instance per admitted training session;
- exactly two ordered groups: `decay`, then `no_decay`;
- exact live parameter object/storage identities, names, shapes, dtypes,
  trainability, and coverage/disjointness;
- frozen vision and pretrained projection excluded;
- motion, eligible text, project-owned text projection, and logit scale included
  according to the frozen resolved inventory;
- constant learning rate and all other hyperparameters exactly as frozen; no
  hidden scheduler or framework default that changes the contract;
- pre-admission failures demonstrate optimizer constructor count zero.

### 29.2 Step semantics

For each optimizer step record the exact batch-manifest position, epoch, global
step, dropout schedule position, loss components, finite-gradient checks,
gradient norm, optimizer-state transition, and checkpoint eligibility. A step
that partially updates parameters but fails before terminal verification is a
failed attempt and must be rolled back or retired according to the frozen
transaction rule; it is not silently continued.

### 29.3 Checkpoint contents

Each checkpoint binds at minimum:

- run ID/attempt ID/system/seed/epoch/global step;
- source, data, split, environment, code, model, text-runtime, and optimizer
  manifests;
- all trainable model tensors and exact storage-independent bytes;
- optimizer state and ordered group membership;
- deterministic sampler/dropout schedule position and every relevant RNG/counter
  state;
- validation-selection state without sealed-test information;
- schema/version/status, parent checkpoint when any, bytes, and SHA-256.

Checkpoint writes are atomic. Resume first validates every dependency and the
entire checkpoint before allocating an optimizer or modifying a model. A resumed
synthetic witness must match the uninterrupted trajectory under the frozen
tolerance/bitwise rule. Incompatible or stale checkpoints fail closed and do not
fall back to partial loading.

## 30. Experiment execution, monitoring, retry, and stop rules

### 30.1 Preflight order

The mandatory order is:

```text
static and unit tests
    -> local CPU synthetic integration
    -> server environment/import tests
    -> seeded GPU kernel witness
    -> GPU synthetic forward/backward/optimizer/checkpoint
    -> private-data read-only loader screen
    -> disposable 64-cluster overfit
    -> nine base runs
    -> base qualification
    -> eighteen residual runs if qualified
    -> validation and inference freeze
    -> one sealed-test opening
```

A later phase cannot be used to compensate for a failed earlier gate.

### 30.2 Monitoring

For running jobs, monitor process liveness, heartbeat freshness, global-step
progress, loss finiteness, gradient finiteness/norm, device utilization, memory,
throughput, checkpoint cadence, validation cadence, disk growth, and log errors.
No single noisy batch or ordinary metric fluctuation is sufficient to kill a
job. Sustained non-finite values, no-progress timeout, repeated OOM, corrupted
checkpoint, split/receipt mismatch, hidden test access, or an ARIS lifecycle
violation requires an evidence-preserving stop.

### 30.3 Retry classification

- `INFRA_TRANSIENT`: network interruption, scheduler delay, or external service
  outage; resume/retry only from a validated checkpoint or new attempt.
- `RESOURCE`: OOM/disk exhaustion; update measured resource plan, preserve the
  failed attempt, and rerun only with a contract-compatible setting.
- `IMPLEMENTATION`: reproducible bug; fix locally, rerun narrow tests and fresh
  QA, sync a new source manifest, then start a new attempt.
- `DATA`: malformed or lineage-violating input; quarantine/disposition only under
  the frozen data rule, never silently skip an inconvenient sample.
- `SCIENTIFIC`: stable but weak performance or qualification failure; preserve as
  the actual result. Do not relabel it as infrastructure failure or tune on the
  sealed test.
- `RIGHTS`: evidence does not permit the attempted use/publication; keep private
  work stopped at that boundary while completing all lawful deliverables.

### 30.4 Negative-result branch

If MIME does not satisfy the preregistered qualification gate, the residual
phase is not launched merely to satisfy a numerical run-count ambition. The
project still completes the scientifically valid branch by preserving all nine
base results, explaining the failed gate, analyzing why the hypothesis could not
be tested as planned, updating figures/tables honestly, and delivering a paper
whose claims match that outcome. Any decision to revise and rerun the scientific
protocol must be an explicit pre-new-run amendment, never a retroactive edit.

## 31. Result artifact and statistical reporting contract

Every reported number must be generated from frozen machine-readable artifacts.
The canonical result table contains, at minimum:

```text
run_id, attempt_id, system_id, seed, split, direction, metric,
value, selected_checkpoint_sha256, evaluator_sha256,
source_cluster_atoms_sha256, status
```

The final validation ledger contains exactly twenty-one eligible system/seed
rows. Per-direction T2M and M2T metrics include the frozen R@1/3/5/10 and MedR
definitions, plus every other preregistered value. Ties use the exact
commitment-derived order; positives come only from the explicit source-ID mask.

For H1-H6, preserve:

- exact comparison definition and direction;
- three seed-level values and the source-cluster atom input;
- bootstrap index commitment and 100,000-replicate matrix/generator identity;
- effect estimate, confidence interval, raw p-value, and applicability gates;
- Holm family ordering, adjusted thresholds/decisions for H2-H6;
- terminal interpretation including null, negative, or non-applicable outcomes.

Tables, plots, paper prose, abstract numbers, slide callouts, README results, and
release notes must all be rendered from the same frozen aggregate tables. Manual
copying from a console, spreadsheet cell, screenshot, or moving tracker is not an
acceptable result path.

## 32. Paper construction blueprint

The final paper is written after the scientific result state is frozen enough to
support each claim. A useful provisional structure is:

1. **Title** — concise, task-accurate, and not broader than skeleton-based
   motion-language evidence.
2. **Abstract** — problem, precise gap, PhasePair mechanism, datasets/protocol,
   actual strongest result or honest null outcome, and limitations; no placeholder
   numbers.
3. **Introduction** — task importance; why generic motion-language embeddings
   miss dyadic periodic semantics; why existing frequency/counting/dyadic work is
   insufficient; concise PhasePair intuition; evidence-backed contributions.
4. **Related work** — motion-language retrieval, dyadic motion understanding,
   skeleton-language supervision, frequency/wavelet/DCT motion modeling,
   repetition/phase reasoning, and the exact closest-work boundary.
5. **Method** — input/notation; AB/BA motion encoding; periodic decomposition;
   six slots and N-token semantic representation; text encoder/alignment;
   residual aggregation; objective; deterministic training/evaluation details.
6. **Experimental setup** — data/rights/splits/quarantine; systems; seeds; runtime;
   optimizer/training; metrics; checkpoint selection; statistics; sealed-test
   policy.
7. **Results** — base qualification, main comparison, H1-H6, ablations, efficiency,
   robustness/failure analysis, and qualitative retrieval cases where rights
   permit visualization.
8. **Limitations and responsible use** — skeleton-only scope, dataset/license
   constraints, two-person focus, compute, privacy, uncertainty, and negative
   evidence.
9. **Conclusion** — only the claims supported by the final artifacts.

Paper construction rules:

- verify the current official ICASSP call, template, page limit, anonymity,
  supplementary, ethics, and deadline rules from authoritative sources before
  final formatting;
- keep a claim-to-artifact ledger for every numeric or comparative sentence;
- verify every BibTeX entry against a primary source; never invent missing year,
  venue, DOI, pages, or author spelling;
- use stable notation shared with code, contracts, captions, and figures;
- state whether results are validation or sealed test, per-seed or aggregate,
  and which direction/metric they use;
- remove every `HOLD`, `TBD`, dummy number, generic article-template marker, and
  unsupported superlative before calling the paper submission-ready;
- compile from a clean environment, inspect logs, render every PDF page, and
  verify fonts, equations, references, tables, figures, margins, accessibility,
  metadata, and absence of private leakage.

## 33. Required editable figure and PowerPoint deliverables

The final editable PowerPoint is a paper-figure source deck, not a presentation
full of prose. It must contain at least the following native editable canvases,
unless the final paper evidence makes one explicitly non-applicable:

1. **Introduction/Motivation** — shared skeleton input; limitation of generic
   motion encoding; PhasePair periodic-semantic insight; improved/contrasting
   retrieval interpretation. It must be understandable in seconds.
2. **Method Overview** — skeleton sequence, motion encoder, periodic
   decomposition, N periodic/relation tokens, text branch, semantic alignment,
   semantic periodic representation, objective, and output.
3. **PhasePair Detail** — AB/BA ordering, symmetric/directional signals, six
   slots, masks, token projector, attention, and the single post-aggregation 0.2
   residual coefficient.
4. **Training/Evaluation Protocol** — multi-positive three-caption objective,
   deterministic run pairing, validation selection, full-gallery metrics, paired
   bootstrap, and one sealed-test boundary, at a density appropriate for a
   supplementary method/protocol figure.
5. **Main Results** — generated from the frozen aggregate table with uncertainty,
   baselines, and negative/null evidence intact.
6. **Ablation or Qualitative Evidence** — only if supported and publishable under
   rights; otherwise replace with an evidence-backed diagnostic plot, never a
   fabricated example.

For each canvas deliver:

- editable `.pptx` source built with the Codex embedded presentation workflow;
- speaker-note `[Sources]` block with exact local/public paths and hashes;
- a rendered high-resolution inspection image;
- paper-ready vector or lossless export as supported by the toolchain;
- a caption draft mapping every visible label to the paper notation;
- a preserve/edit/do-not-touch record when revising an earlier version;
- automated overflow result and human visual-QA checklist;
- grayscale and paper-scale legibility result.

The final deck is compared against the preceding clean editable deck and the
visual-language review of the five required PDFs. It may learn their restraint,
hierarchy, spacing, and connector quality, but must not copy a copyrighted figure
or misrepresent PhasePair's actual graph.

## 34. Continuous public GitHub delivery contract

The public repository already exists at the URL in Section 22. Its release rule
is continuous but evidence-gated:

- push a coherent milestone when source/tests and public documentation are
  internally consistent and the public allowlist scan passes;
- do not wait until the end to publish months of unreviewed work;
- do not push a moving/failing snapshot simply to increase commit frequency;
- preferred temporary working branch names use the `codex/` prefix, while every
  reviewed public milestone is published from the canonical `main` branch;
- each commit message states the milestone, such as `core:`, `clip:`, `data:`,
  `trainer:`, `experiments:`, `figures:`, `paper:`, or `release:`;
- before each push, run the relevant tests/Ruff/build plus a secret/private/path/
  restricted-asset/large-file/cache scan over the exact staged tree;
- after each push, verify the remote branch head, recursive tree census, and blob
  identities rather than trusting the local push command alone;
- if ordinary Git HTTPS is unavailable but authenticated official GitHub API
  access works, the Git Data API may be used to create blobs/trees/commits/refs;
  the resulting remote tree must exactly match the sanitized local tree;
- never place the private `.aris` tree, SSH locator, dataset, captions, motions,
  weights, wheels, restricted checkpoints, private receipts, reference-library
  PDFs, caches, or server logs in the public repository;
- public code must include reproducible download/verification instructions for
  third-party assets instead of redistributing prohibited bytes;
- update README/STATUS/MANIFEST and reproduction commands in the same milestone
  that changes the public behavior they describe;
- final release includes a tagged commit, license/citation metadata, exact public
  environment instructions, tests, paper source/PDF, editable figures, public
  receipts, and a truthful results summary.

## 35. Agent continuity and anti-forgetting protocol

Every new root continuation or sub-agent must begin by reading, in order:

1. this objective from byte 1 to EOF;
2. the current goal/status supplied by the root agent;
3. the latest public README/STATUS/MANIFEST and private execution status relevant
   to its scope;
4. the latest fresh-reviewed scientific contract/plan;
5. exact source/test or evidence bytes in scope;
6. predecessor review findings that caused the current revision.

Every assignment must name:

```text
objective and why the subtask matters
exact allowed files/systems/data boundary
required source documents and hashes
required implementation or evidence output
commands and independent checks expected
truthful capability/status boundary
forbidden writes/publications
handoff fields and completion condition
```

Every natural checkpoint reports concrete new evidence, not an ETA-only status.
Every final handoff reports exact paths, bytes, SHA-256, commands, exit codes,
test counts, skips, warnings, private/public boundary, remaining HOLDs, and the
next dependency. If an agent disappears, the next agent resumes from durable
artifacts and this ledger rather than reconstructing intent from memory.

Known failed approaches are retained in the status ledger so they are not
repeated blindly. For example, an alternate SSH port that does not authenticate
is not a substitute for the configured endpoint, and a wrapper that failed to
render reference PDFs must be diagnosed or replaced with the bundled underlying
renderer rather than treated as completed visual review.

## 36. Current verified state at v3 publication

This is a dated progress snapshot, not a substitute for live revalidation:

- the current PhasePair data-free core and its targeted integrations have passed
  the latest full local test and Ruff gates recorded by the root agent;
- sealed `text_runtime.py` and `execution_schema.py` implementations exist and
  their current targeted/full local suites pass;
- the exact CLIP revision has been acquired into private storage, its eight-file
  runtime snapshot has been rehashed, a public rights/acquisition receipt has
  fresh QA, and redistribution of weight bytes remains prohibited;
- a private Windows CPU wheelhouse and offline text-runtime preflight are moving
  and must be frozen/reviewed before they count as final runtime evidence;
- the InterHuman B0 refresh has completed as `HOLD / NOT_READY_FOR_SERVER_STAGING`;
  its public-safe report is
  `artifact://phasepair/PHASEPAIR_INTERHUMAN_B0_REFRESH_PUBLIC_20260825_012846.md`
  (8,067 bytes, SHA-256
  `b0a7cdd37fb24aee16cafe877ed1fb53a1b9ac23c554c7c0de9d238a4a919dbb`);
  all seven receipt-bound raw objects rehash, but the live mount is 0/6 and the
  exhaustive near-duplicate, final allowlist, lineage, owner-license, and
  one-archive-topology gates are still absent;
- the configured private SSH service is temporarily refusing connections before
  authentication; the host is reachable, and independent local work continues;
- no alternate unauthorized SSH endpoint has been adopted;
- no real PhasePair training, base result, residual result, statistical result,
  or sealed-test result exists yet;
- the existing paper and editable figure deck are truthful data-free drafts, not
  submission-ready result artifacts;
- the public repository has been created and an exact sanitized source tree has
  been pushed; subsequent stable milestones must continue to be pushed;
- the five required reference PDFs have an identity/page inventory, but full
  page-by-page scientific and rendered visual review is still required.

No future agent may convert any moving item in this list into a PASS merely by
quoting this snapshot.

## 37. Expanded final delivery ledger

In addition to Section 18, final delivery must include the following explicit
artifacts or a documented, independently accepted non-applicability reason.

### 37.1 Scientific and literature

- [ ] exact current scientific contract and experiment plan;
- [ ] full five-PDF page-grounded review and visual-language report;
- [ ] current closest-work/novelty matrix with verified citations;
- [ ] final H1-H6 definitions and protocol amendment history;
- [ ] claim-to-evidence ledger used by paper, slides, README, and release.

### 37.2 Code and runtime

- [ ] production data loader and lineage admission;
- [ ] live text-runtime bridge and exact CLIP resolver;
- [ ] motion encoders, periodic decomposition, matched controls, objective;
- [ ] optimizer admission, trainer, checkpoint/resume, residual trainer;
- [ ] evaluation, bootstrap, sealed-test controller, ARIS lifecycle publication;
- [ ] local CPU and server CUDA environment locks/receipts;
- [ ] complete unit/integration/adversarial/GPU test evidence.

### 37.3 Private evidence and experiments

- [ ] CLIP snapshot/wheel/runtime/rights private receipt and public redaction;
- [ ] InterHuman owner/rights/lineage/quarantine/split private receipt and public
      redaction;
- [ ] server inventory and environment receipt;
- [ ] synthetic CPU/GPU and resume-equivalence receipts;
- [ ] private-data loader screen and disposable overfit receipt;
- [ ] nine base attempt directories and terminal receipts;
- [ ] base qualification object and three frozen MIME caches if qualified;
- [ ] eighteen residual attempt directories and terminal receipts if applicable;
- [ ] twenty-one validation rows, bootstrap/Holm outputs, and sealed-test receipt.

### 37.4 Results, visuals, and paper

- [ ] frozen canonical result tables and plot-generation inputs;
- [ ] editable Introduction/Motivation/Method/Protocol/Results PPTX sources;
- [ ] rendered slide/page QA images and reports;
- [ ] paper-ready figure exports and verified captions;
- [ ] official ICASSP LaTeX source, verified BibTeX, and reproducible build script;
- [ ] final PDF with all pages rendered/inspected and no placeholder/private leak;
- [ ] final scientific/citation/claim/typography/reproducibility review reports.

### 37.5 Public release and handoff

- [ ] sanitized public allowlist and release scan report;
- [ ] public repository remote branch/tag/commit/tree verification;
- [ ] README quickstart and exact reproduction levels: data-free, private-asset
      preparation, environment, training, evaluation, figures, and paper build;
- [ ] public license, citation metadata, status, manifest, receipts, paper/PDF,
      editable figures, and result summary;
- [ ] private operator handoff containing only the necessary restricted pointers,
      never copied to the public repository;
- [ ] final root report listing what was delivered, exact remote URL/commit/tag,
      actual experiment outcomes, known limitations, and any lawful boundary.

## 38. Final decision and reporting rules

- The project is not complete while real results are missing and the paper still
  contains result placeholders.
- The project is not complete merely because code QA is clean, CLIP loads, SSH
  works, or the repository is public.
- A scientifically negative outcome can still be a complete project when every
  frozen gate was executed honestly and the final paper reports that outcome.
- An operationally unreachable server is not by itself project completion or a
  reason to abandon independent work; it remains an external dependency that is
  retried while the rest advances.
- A third-party redistribution `NO_GO` changes the public packaging strategy, not
  the obligation to complete lawful private research and reproducible download
  instructions.
- The final response must lead with actual deliverables and evidence, not a list
  of intentions. It must distinguish completed, negative, non-applicable, and
  legally non-redistributable items.
- The root agent may mark this objective complete only when every applicable box
  in Sections 18 and 37 is closed, the public remote has been verified, and an
  independent final audit finds no unresolved contradiction that affects a
  delivered claim.

## 39. Mandatory GitHub publication and remote-verification contract

GitHub publication is a required project deliverable, not an optional cleanup
step and not something that may be inferred from a local commit. The designated
public repository is:

- repository: `https://github.com/appleweiping/phasepair-periodic-motion-language`;
- visibility required at handoff: public;
- final default branch: an explicitly verified release branch, with its exact
  name, commit SHA, and tree SHA recorded in the final report;
- release identity: a signed or otherwise immutable annotated release tag when
  the final release authority and local tooling permit it.

The final public upload must contain every lawful, sanitized, reproducibility-
relevant project artifact, including source code, tests, configuration,
environment specifications, data-free fixtures, public lineage/rights receipts,
experiment schemas, aggregate result tables, plot inputs and scripts, paper
sources, final paper PDF, editable figure/slide sources, rendered figures,
documentation, citation metadata, license, changelog/status, release manifest,
and public QA evidence. An item may be omitted only when it is restricted,
private, credential-bearing, license-prohibited, personally identifying, or
otherwise unsafe to publish; each such omission must have a public explanation
and a private handoff entry without exposing the restricted value itself.

The phrase "uploaded to GitHub" is valid only after all of the following checks
pass against the remote service itself:

1. every intended local release commit is present on the remote;
2. the reported remote branch SHA exactly equals the intended local release SHA;
3. the reported remote tree SHA exactly equals the intended local release tree;
4. the repository visibility and default branch are queried and recorded;
5. all intended release files are enumerated by a deterministic manifest with
   sizes and SHA-256 digests;
6. the remote repository and every retained Git ref/history object pass the
   secret, credential, endpoint, local-path, private-lineage, oversized-blob,
   weight/checkpoint, dataset, and forbidden-artifact scans;
7. an anonymous fresh clone or archive download succeeds without relying on the
   developer machine's untracked files or local caches;
8. installation, import, test, lint, documentation-link, paper-build, and
   artifact-integrity checks are rerun from that fresh checkout in every
   environment that is available and relevant;
9. README/status text agrees with the actual uploaded implementation, evidence,
   remaining legal gates, server execution state, and scientific results;
10. the final public release/tag page is reachable and the final report records
    its exact URL, commit SHA, tag, timestamp, test summary, and known limits.

Local commits, local working-tree files, a successful `git commit`, a configured
remote, or an earlier partial push do not satisfy this gate. If GitHub networking
is temporarily unavailable, all safe local release work continues, the exact
unpushed commit/file ledger is preserved, bounded push retries continue, and the
project remains incomplete until remote equality and fresh-clone verification
succeed. No status report may describe unpushed work as public.

Before the final push, previously published Git history must also be sanitized.
Adding a later deletion is insufficient when a forbidden value remains in an old
commit. The release procedure must preserve a recoverable local backup, rewrite
or replace affected public history in a controlled manner, rescan every retained
ref/object, update the public refs only after the sanitized tree is complete, and
verify that no obsolete remote branch or tag still exposes superseded content.

Historical checkpoint at the start of the 2026-08-25 publication pass: the
repository existed but its remote lagged the local public worktree and its old
history still required sanitation. That checkpoint was **repository exists,
partial upload only** and may be superseded only by remote branch/tree equality,
retained-ref scans, and a fresh anonymous checkout. The mandatory final project
end state remains **all lawful public deliverables uploaded, remote-equal,
history-clean, anonymously reproducible, and independently audited**.
