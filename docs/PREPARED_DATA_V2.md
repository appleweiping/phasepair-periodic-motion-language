# Prepared motion-text batches and epoch augmentation

`phaseset_core.prepared_data_v2` connects already prepared body-22 motion
windows to already frozen CLIP text features. It writes immutable numeric
training artifacts and reads them as `TrainingDataSource` objects. It does not
download or authorize data, decode SMPL-X, generate captions, or run CLIP.

## Interface

`PreparedMotionTextBlock` contains canonical `PreparedGroupSample` values,
one `FrozenClipTextBatch`, variable per-motion `text_counts`, and independent
uint64 `window_ordinals`. Every motion receives at least one text row; no fixed
three-caption or two-caption assumption is made. The supplied frozen text
receipt must bind the actual embedding bytes, row order, and commitments.

`write_prepared_training_tree_v2(...)` takes separate train/val blocks, an edge
budget, optional microbatch-size bound, and explicit artifact-size bounds. It
creates a new tree and returns its index and census digests. Existing output
roots cannot be reused. Failed partial builds are preserved, not admitted as
completed datasets.

`load_prepared_training_sources_v2(...)` requires the expected index digest
and returns train and val sources. Each split constructor verifies the exact
manifest bytes it consumes against the digest in that index. Test cannot be
inserted into this training-only index.

## Numeric artifact contract

Every uncompressed NPZ has exactly ten arrays:

| Name | Type and shape |
|---|---|
| `skeletons` | float32 `[B,K_pad,200,22,3]` |
| `actor_mask` | bool `[B,K_pad]` |
| `frame_mask` | bool `[B,200]` |
| `track_mask` | bool `[B,K_pad,200,22]` |
| `actor_commitments` | uint8 `[B,K_pad,32]` |
| `group_commitments` | uint8 `[B,32]` |
| `text_embeddings` | float32 `[Q,512]` |
| `motion_positive_ids` | uint8 `[B,32]` |
| `text_positive_ids` | uint8 `[Q,32]` |
| `text_commitments` | uint8 `[Q,32]` |

Padding is exact positive zero. Output creation is write-once with read-only
file permissions; this is not operating-system immutability. Authenticated
digests bind the bytes actually consumed. Storage uses fixed-order ZIP_STORED members
and NPY v1 headers. Loading checks member identity, shape/type, declared byte
length, CRC, and decoded-size bounds before array allocation; pickle is not
accepted. The writer rejects train/val overlap in window, ordinal, caption,
and actor commitments. Upstream participant/capture split verification remains
necessary; these numeric checks do not establish participant identity.

## Epoch behavior

Stored motion always has exact positive-zero augmentation yaw. Each training
epoch rereads the original authenticated bytes and applies the registered
`deterministic_group_yaw(seed, epoch, window_ordinal)` once, synchronously to
all actors. The angle is not derived from any lineage commitment. Rotation is
computed in float64 and cast to float32, then all invalid positions are reset
to exact positive zero. No augmented sample becomes the next epoch's input.

Validation is unrotated. Masks, lineage, positive families, caption identity,
and frozen text values do not change. This storage-based path is not claimed
bitwise equal to rotating earlier, before preprocessing's final float32 cast.

## Provenance and integration boundary

The private build census binds index/batch/window/source/text digests and the
independent ordinals. It contains no raw captions, source paths, or participant
raw identifiers and is not a data-rights certificate. Stable commitments may
still be linkable; they are not an anonymity guarantee. Keep the entire
prepared tree, its commitments, and its census private.

The private-host factory recognizes the v2 index and passes the actual
receipt-bound index digest into this loader. A replaced index or manifest, or
invalid decoded training batch, is rejected before backend construction. The
legacy v1 loader remains supported and also binds the manifest bytes consumed.

Positive families in this storage format identify individual windows. They
support the auxiliary window-caption task, not the main holistic capture task.
Actor-set group commitments are lineage, not unique capture/window row IDs.
Main validation must separately encode every admitted window in a capture,
aggregate embeddings with the fixed capture mean, and score one capture gallery
against official holistic text. The [capture validation API](CAPTURE_VALIDATION.md)
now implements that operation and training checkpoint selection, but the
private-host capture-source disk route is not yet connected to this storage.

This module alone is not the complete `prepare-data` command. Native capture
loading, licensed body-model conversion, preprocessing rejection/tail census,
caption provenance, and the preparation command composition must also be wired
and verified before real training. See [the frozen text adapter](FROZEN_CLIP_TEXT_ADAPTER.md)
for text feature semantics and [host integration](HOST_RUNTIME_INTEGRATION.md)
for the execution lifecycle.

Server tests exercise analytic numeric artifacts, variable caption counts,
epoch replay, padding zeros, validation invariance, split isolation, immutable
creation, and invalid/replaced manifests and ZIP payloads. They are regression
evidence, not real-data experiments or scientific results. The final combined
host integration passed 94 focused tests and a separate complete Linux server
suite of 964 tests, 2 skipped, and 39 subtests, with unchanged source bytes.
