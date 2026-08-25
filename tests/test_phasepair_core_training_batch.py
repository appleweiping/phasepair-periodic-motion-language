"""Synthetic-only tests for motion/caption training-batch cross-binding."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from phasepair_core import batching, caption_processing, training_batch


def _raw32(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _motion(pairs: tuple[bytes, ...]) -> batching.PreparedMotionBatch:
    batch_size = len(pairs)
    time_steps = 2
    return batching.PreparedMotionBatch(
        actor_a=np.zeros((batch_size, time_steps, 262), dtype=np.float32),
        actor_b=np.zeros((batch_size, time_steps, 262), dtype=np.float32),
        relation_ab=np.zeros((batch_size, time_steps, 799), dtype=np.float32),
        relation_ba=np.zeros((batch_size, time_steps, 799), dtype=np.float32),
        valid_mask=np.ones((batch_size, time_steps), dtype=np.uint8),
        valid_lengths=(time_steps,) * batch_size,
        pair_commitments=pairs,
    )


def _captions(pairs: tuple[bytes, ...]) -> caption_processing.PreparedCaptionBatch:
    batch_size = len(pairs)
    caption_commitments = tuple(
        tuple(_raw32(1000 + 3 * row + ordinal) for ordinal in range(3))
        for row in range(batch_size)
    )
    payloads = tuple(
        tuple(f"caption-{row}-{ordinal}".encode() for ordinal in range(3))
        for row in range(batch_size)
    )
    ids = np.full((3 * batch_size, 77), 1, dtype=np.int64)
    mask = np.zeros((3 * batch_size, 77), dtype=np.int64)
    for row in range(3 * batch_size):
        ids[row, :3] = [0, 10 + row, 2]
        mask[row, :3] = 1
    return caption_processing.PreparedCaptionBatch(
        pair_commitments=pairs,
        caption_commitments=caption_commitments,
        caption_ordinals=tuple((0, 1, 2) for _ in pairs),
        caption_payloads=payloads,
        input_ids=ids,
        attention_mask=mask,
        tokenizer_manifest_sha256=_digest("tokenizer"),
        caption_text_manifest_sha256=_digest("caption-text"),
        caption_lineage_manifest_sha256=_digest("caption-lineage"),
        normalizer_manifest_sha256=_digest("normalizer"),
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        vocab_size=100,
    )


def _training(
    motion: object,
    caption_batch: object,
    *,
    manifest_sha256: str | None = None,
    seed: int = 1729,
    epoch: int = 0,
    batch_index: int = 0,
    global_step: int = 0,
) -> training_batch.PreparedTrainingBatch:
    return training_batch.PreparedTrainingBatch(
        motion_batch=motion,
        caption_batch=caption_batch,
        sampler_manifest_sha256=(
            _digest("sampler-manifest")
            if manifest_sha256 is None
            else manifest_sha256
        ),
        sampler_seed=seed,
        epoch_index=epoch,
        batch_index=batch_index,
        global_step=global_step,
    )


def test_training_batch_derives_only_valid_three_caption_positive_mask() -> None:
    pairs = (_raw32(1), _raw32(2))
    motion = _motion(pairs)
    captions = _captions(pairs)
    batch = _training(motion, captions)
    np.testing.assert_array_equal(
        batch.positive_mask,
        np.array(
            [
                [True, True, True, False, False, False],
                [False, False, False, True, True, True],
            ],
            dtype=np.bool_,
        ),
    )
    assert not batch.positive_mask.flags.writeable
    assert batching.prepared_batch_sha256(batch.motion_batch) == (
        batching.prepared_batch_sha256(motion)
    )
    assert batch.caption_batch is captions


def test_receipt_is_canonical_public_safe_and_binds_both_batches() -> None:
    pairs = (_raw32(1), _raw32(2))
    motion = _motion(pairs)
    caption_batch = _captions(pairs)
    batch = _training(motion, caption_batch)
    raw = training_batch.prepared_training_batch_bytes(batch)
    payload = json.loads(raw)
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert payload["schema"] == "phasepair-prepared-training-batch-v1"
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["execution_authorized"] is False
    assert payload["result_claimed"] is False
    assert payload["status"] == training_batch.STATUS
    assert payload["batch_size"] == 2
    assert payload["caption_count"] == 6
    assert payload["positive_mask_shape"] == [2, 6]
    assert payload["sampler_manifest_sha256"] == _digest("sampler-manifest")
    assert payload["sampler_seed"] == 1729
    assert payload["epoch_index"] == 0
    assert payload["batch_index"] == 0
    assert payload["global_step"] == 0
    assert payload["motion_prepared_batch_sha256"] == batching.prepared_batch_sha256(
        motion
    )
    assert payload["caption_prepared_batch_sha256"] == (
        caption_processing.prepared_caption_batch_sha256(caption_batch)
    )
    assert b"caption-" not in raw
    assert raw == (
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        + b"\n"
    )
    assert training_batch.prepared_training_batch_sha256(batch) == hashlib.sha256(
        raw
    ).hexdigest()


def test_pair_order_mismatch_fails_even_when_the_pair_set_is_equal() -> None:
    pairs = (_raw32(1), _raw32(2))
    with pytest.raises(training_batch.TrainingBatchError, match="order must be identical"):
        _training(_motion(pairs), _captions(tuple(reversed(pairs))))


def test_receipt_changes_with_pair_order_and_remains_deterministic() -> None:
    first_pairs = (_raw32(1), _raw32(2))
    second_pairs = tuple(reversed(first_pairs))
    first = _training(_motion(first_pairs), _captions(first_pairs))
    first_again = _training(_motion(first_pairs), _captions(first_pairs))
    second = _training(_motion(second_pairs), _captions(second_pairs))
    assert training_batch.prepared_training_batch_bytes(first) == (
        training_batch.prepared_training_batch_bytes(first_again)
    )
    assert training_batch.prepared_training_batch_bytes(first) != (
        training_batch.prepared_training_batch_bytes(second)
    )


def test_types_and_immutability_fail_closed() -> None:
    pairs = (_raw32(1),)
    motion = _motion(pairs)
    caption_batch = _captions(pairs)
    with pytest.raises(TypeError, match="exactly PreparedMotionBatch"):
        _training(object(), caption_batch)
    with pytest.raises(TypeError, match="exactly PreparedCaptionBatch"):
        _training(motion, object())
    batch = _training(motion, caption_batch)
    with pytest.raises(AttributeError, match="immutable"):
        batch.positive_mask = np.zeros((1, 3), dtype=np.bool_)
    with pytest.raises(TypeError, match="exactly PreparedTrainingBatch"):
        training_batch.prepared_training_batch_bytes(object())


def test_authority_zero_constants_are_unambiguous() -> None:
    assert training_batch.AUTHORITY == 0
    assert training_batch.PRODUCTION is False
    assert training_batch.EXECUTION_AUTHORIZED is False
    assert training_batch.RESULT_CLAIMED is False
    assert training_batch.STATUS.endswith("AUTHORITY0_NO_RESULT")


def test_training_batch_rejects_forgery_nested_rebinding_and_bad_provenance() -> None:
    pairs = (_raw32(1),)
    motion = _motion(pairs)
    caption_batch = _captions(pairs)
    issued = _training(motion, caption_batch)
    object.__setattr__(issued, "canonical_receipt", b"FORGED\n")
    with pytest.raises(training_batch.TrainingBatchError, match="fields were rebound"):
        training_batch.prepared_training_batch_bytes(issued)

    forged = object.__new__(training_batch.PreparedTrainingBatch)
    object.__setattr__(forged, "canonical_receipt", b"FORGED\n")
    with pytest.raises(training_batch.TrainingBatchError, match="not issued"):
        training_batch.prepared_training_batch_bytes(forged)

    rebound_caption = _captions(pairs)
    bound = _training(_motion(pairs), rebound_caption)
    object.__setattr__(rebound_caption, "caption_commitments", ((_raw32(999),) * 3,))
    with pytest.raises(caption_processing.CaptionProcessingError, match="fields were rebound"):
        training_batch.prepared_training_batch_bytes(bound)

    with pytest.raises(TypeError, match="exact built-in int"):
        _training(_motion(pairs), _captions(pairs), seed=True)
    with pytest.raises(training_batch.TrainingBatchError, match="outside the frozen"):
        _training(_motion(pairs), _captions(pairs), seed=1)
    with pytest.raises(training_batch.TrainingBatchError, match="lowercase SHA-256"):
        _training(_motion(pairs), _captions(pairs), manifest_sha256="A" * 64)
