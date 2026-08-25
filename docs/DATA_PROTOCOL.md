# PhaseSet data and split protocol

This protocol freezes the public-metadata layer for the PhaseSet migration. It
does not grant dataset access, download motion or language assets, or authorize
publication of participant-level material.

## Source and access boundary

The only automatically readable sources are the `acting/dataset.json` and
`daylife/dataset.json` indexes in the official
[Embody 3D repository](https://github.com/facebookresearch/embody-3d). The
official repository states that these indexes describe 30 fps sequence length,
text availability, and multi-person membership. It also states that full data
access requires the Meta release form and that the dataset itself is governed
by the XRCIA dataset license. The repository toolbox license is not a substitute
for that dataset license.

`load_official_public_metadata()` reads both public JSON documents into memory.
It has an exact HTTPS host/source allowlist, a response-size limit, a timeout,
strict UTF-8/JSON parsing, and duplicate-key rejection. It never follows an
asset path from an index and never writes source bytes to disk. Local index
copies can instead be supplied with `load_public_metadata_files()`.

The snapshot audited on 2026-08-25 had these public index byte digests:

| Subset | SHA-256 |
| --- | --- |
| acting | `94be382d6279db9cb51241705a3eb532e607e53f4f6c477a932bb07114ddecd0` |
| daylife | `9c1a56914e6f389d05f2100e2f7e619e30a61eba10b7cf62182bba10580139ef` |

These digests are provenance anchors, not dataset redistribution. If the
upstream indexes change, the component audit fails until a reviewed manifest
revision is created.

## Ingestion and capture identity

The public index repeats a multi-person capture once under each participant.
Ingestion collapses those rows to one record with canonical identity
`(subset, capture_name)`. For every collapsed capture, all repeated rows must
agree on frame length and the full participant set, and every declared
participant must have exactly one row. A disagreement, missing row, duplicate
row, or cross-document identity collision fails closed.

No annotation payload is read. A nonempty string in the index's text field is
reduced to a Boolean language-text availability bit per actor. The officially
documented integer class-label form remains a valid index row but does not
count as natural-language text. An eligible PhaseSet record must satisfy all of
the following:

- it belongs to `acting` or `daylife`;
- it has at least three co-present participants;
- every declared participant has a public actor row;
- every actor row has a nonempty string text entry;
- frame length and participant membership agree across the repeated rows.

This `K>=3` condition belongs to the confirmatory Embody study, not to the
software API. The public preparation and model contracts accept every valid
`K>=2` group so dyadic backward-transfer evaluation remains possible; such
dyadic samples cannot enter the native multi-person main table.

This establishes index-level actor-text availability only. It does not prove
that an annotation file is readable, semantically correct, or holistic. Asset
existence, rights, hashes, and annotation quality remain execution-time gates
after authorized dataset access.

### Auxiliary machine-fused caption contract

`caption_fusion.py` exposes a text-only, host-injected contract for the
auxiliary ten-second group-caption task. It is fixed to `gpt-5.6-sol`, one
versioned prompt, a three-attempt transient-only retry policy, and three
distinct deterministic actor orders that are actually sent to the backend.
All three normalized outputs must be identical or fusion fails closed. This is
evidence of normalized surface stability under the three tested orders, not a
semantic-equivalence proof. Same-model-family semantic review therefore stays
`PROVISIONAL` and cannot close an execution or scientific gate. The
backend request contains only the K official human actor descriptions and an
optional official human holistic context from the same window; commitments,
split labels, skeletons, phase/coherence fields, model outputs, and paths are
absent from that request schema.

Before any request, the host-injected backend must expose a frozen manifest
binding the exact model revision, backend implementation, inference runtime,
`temperature=0`, `top_p=1`, generation seed `1729`, maximum 512 output tokens,
and the strict two-field JSON response format. Each request binds that manifest
and model revision; a caller-supplied model name alone is insufficient.

The raw input and generated text remain private in-memory values with no
serializer. The only public serializer emits a digest-only authority-zero
receipt that binds prompt/model/backend/input/output/provenance/retries and
states `machine_fused=true`, `human_annotation=false`, and
same-family review `PROVISIONAL`. Test provenance includes both a sealed
test-manifest digest and a sealed test-caption digest, but digest strings alone
cannot admit fusion. A trusted host gate must verify and consume an external
grant, return a `CaptionFusionTestAdmission` bound to that exact provenance,
and record external-grant and consumption digests. The public receipt exposes
only the resulting admission digest. These local schemas cannot authenticate
dataset rights or close the semantic-review or sealed-evaluation gates by
themselves.

The audited eligible pool contains 572 unique captures, 69 participants,
2,232,649 source frames, and 20.672675926 hours at 30 fps. It consists of 296
acting captures and 276 daylife captures; 27 captures have K=3 and 545 have
K=4.

## Participant-disjoint component split

Participants are vertices in a hypergraph and each eligible capture connects
all of its participants. Connected components are indivisible split units. This
is stronger than capture-level random splitting: any transitive co-occurrence
chain stays in one split.

The current graph has 16 connected components: one with 10 participants,
fourteen with 4 participants, and one with 3 participants. Public labels
`C00` through `C15` are opaque manifest labels, not participant codes.

Each component is bound by a domain-separated SHA-256 commitment over canonical
JSON containing its complete runtime preimage: sorted participants, sorted
capture identities, frame counts, and actor-text availability. Only the digest
and aggregate statistics are public. The commitment is an integrity mechanism,
not a general anonymization guarantee; participant/capture preimages must still
remain inside the authorized data environment.

The fixed component allocation is:

- train: `C01,C02,C03,C04,C05,C06,C07,C08,C10,C12,C13,C14`;
- validation: `C00`;
- test: `C15,C09,C11`.

Its exact aggregate contract is:

| Split | Components | Participants | Captures | Frames | Hours | Acting | Daylife | K=3 | K=4 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| train | 12 | 48 | 400 | 1,511,750 | 13.997685185 | 208 | 192 | 0 | 400 |
| validation | 1 | 10 | 96 | 394,975 | 3.657175926 | 40 | 56 | 0 | 96 |
| test | 3 | 11 | 76 | 325,924 | 3.017814815 | 48 | 28 | 27 | 49 |

`audit_split()` re-derives every component, matches every commitment, verifies
component and split aggregates, assigns every eligible capture exactly once,
and checks that participant sets have pairwise-empty intersections. The audit
report contains aggregate statistics only.

### Exact K=3 claim

All 27 eligible real K=3 captures are in test; train and validation contain no
real K=3 capture. Therefore this split supports the narrow statement "K=3 is a
real-data zero-shot group-size holdout under the eligible Embody 3D K=3/K=4
index." It does not establish generalization to arbitrary K, a new interaction
domain, unseen annotation styles, or unseen recording conditions.

## Missing-track window rule

After authorized motion access, construct a Boolean mask `[T,K]` by requiring
the official tracking-quality indicator to be good for each participant at each
source frame. Each production window is exactly 300 source frames (10 seconds
at 30 fps). It is accepted only when all of the following hold:

- every actor has at least 285 valid source frames, exactly 95% of 300;
- every contiguous missing run has at most seven source frames;
- every missing run is internal and bounded by valid frames on both sides;
- the first and last source frame are valid for every actor.

Missing duration is defined as `run_length / 30`, not timestamp endpoint
distance. Thus seven frames are 0.2333 seconds and satisfy the 0.25-second
limit, while eight frames are 0.2667 seconds and are rejected. This definition
removes the usual off-by-one ambiguity.

For an accepted internal gap, each motion feature is linearly interpolated
between its two observed boundary samples. There is no leading/trailing
extrapolation, zero fill, or forward fill. Non-finite values are allowed only at
masked actor-frames; a non-finite observed value rejects the window.

`assess_missing_window()` reports total missing actor-frames, minimum
per-participant validity, maximum run length, and boundary-gap status.
`interpolate_accepted_gaps()` returns the filled kinematics together with the
unchanged source mask. Interpolation never changes a false mask bit to true.

The numeric seam expands the official actor-frame indicator across the 22 body
joints. It validates that all 22 bits are identical within each actor-frame;
a partial-joint mask is rejected before interpolation. This prevents one
missing joint from causing every joint to be interpolated while other channels
remain falsely labelled observed. Each accepted sample retains private
source-file and source-window SHA-256 lineage metadata. Those digests bind the
preparation manifest but are not model inputs or public participant labels.

## 30 to 20 fps resampling

Resampling maps exactly 300 frames at 30 fps to exactly 200 frames at 20 fps and
is offline and deterministic:

1. validate the 300-frame observation mask and linearly fill only accepted
   internal gaps;
2. apply a symmetric, zero-phase, 31-tap Blackman-windowed sinc low-pass filter
   at 30 fps;
3. use a 9 Hz cutoff, below the 20 fps output Nyquist frequency of 10 Hz;
4. use constant endpoint extension for the finite-window filter boundary;
5. sample the filtered trajectory at exact 20 fps timestamps with linear
   interpolation between filtered 30 fps samples.

The general timestamp formula gives
`floor((300 - 1) × 20 / 30) + 1 = 200`, beginning at time zero and ending at
9.95 seconds. The result retains the immutable 300-frame source mask and a
200-frame derived mask. At an integer source position, the derived bit equals
that source bit; at a half-frame position, both bracketing source bits must be
true. The filled motion is never presented as fully observed. FIR coefficients,
timestamps, masks, and resampled values are returned read-only. Raw subsampling
without low-pass filtering is outside the protocol.

All downstream PhaseSet periodic responses therefore use the independent
20-Hz bank frozen in
[`PHASESET_20HZ_MORLET_CONTRACT.md`](PHASESET_20HZ_MORLET_CONTRACT.md). The
30-Hz PhasePair v0.1.0 tap bytes are not applied to the resampled sequence and
are not relabelled as 20-Hz filters.

### Production and gradient-qualification boundary

The production periodic path validates a `PreparedGroupBatch`, derives the
five-channel activity representation in NumPy, and streams NumPy Morlet/pair
descriptors into the learned Torch edge heads. It is intentionally an explicit
non-differentiable boundary: production execution does not claim gradients to
the original skeleton array.

`PhaseSetEncoder.forward_skeleton_autograd_oracle()` is a separate,
small-batch CPU qualification path. It keeps skeleton-to-activity, Morlet, and
edge summaries in Torch autograd, accepts arbitrary valid-actor positions, and
shares the registered edge/topology/postprocess parameters with the production
encoder. It exists to test skeleton-input gradient permutation equivariance;
it is not a production data adapter, accelerator result, or assertion that the
NumPy descriptor stream is differentiable.

## Group coordinate canonicalization

Input positions have shape `[T,K,J,3]` and root yaw has shape `[T,K]`. The
default reference is the first frame that observes all participants.

- The group anchor is the mean 3D root position across all K participants at
  the reference frame. One constant anchor is subtracted from every joint,
  participant, and frame.
- Yaw is never inferred from an indexed actor or from a circular mean of actor
  orientations. The caller may supply an audited scene-level `shared_yaw` to
  remove; its default is zero.
- Training may separately supply one sampled `augmentation_yaw`. The applied
  angle is `augmentation_yaw - shared_yaw` and the exact same rotation/addition
  is used for every participant and frame.
- The public training seam derives this yaw only from the registered seed,
  epoch, and a globally unique window ordinal supplied by the frozen manifest.
  A capture-local default is forbidden. Actor and group commitments never key
  an augmentation and therefore remain lineage/reduction-order metadata rather
  than neural inputs. Distinct windows receive distinct deterministic draws.
- Per-participant centering or per-participant yaw normalization is forbidden:
  it would erase relative group position and facing.

The shared transform is participant-order invariant and preserves pairwise
distances and relative yaw.

## Reproducible audit

The repository tests are fully offline and use synthetic identities. An
authorized live metadata audit can be run without downloading any asset:

```python
from phaseset_core.data import load_official_public_metadata
from phaseset_core.split_audit import audit_split, load_split_manifest

metadata = load_official_public_metadata()
manifest = load_split_manifest("configs/phaseset/embody3d_split_v1.json")
report = audit_split(metadata, manifest)
assert report.participant_overlap_count == 0
assert report.cross_split_capture_count == 0
assert report.k3_zero_shot_verified
```

Do not serialize runtime component objects: they intentionally retain the
preimage in memory for verification. Publish only the aggregate report and the
reviewed commitment manifest.
