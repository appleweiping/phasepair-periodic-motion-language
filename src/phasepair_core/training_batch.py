"""Cross-bind prepared PhasePair motion and three-caption batches.

The builder requires identical pair order on both sides, derives the only
valid ``[B,3B]`` source-positive mask internally, and emits a public-safe
Authority-0 receipt.  It performs no model forward, optimizer step, file I/O,
or experiment authorization.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading

import numpy as np

from phasepair_core import batching, caption_processing
from phasepair_core._identity_registry import make_identity_weak_registry


AUTHORITY = 0
PRODUCTION = False
EXECUTION_AUTHORIZED = False
RESULT_CLAIMED = False
STATUS = "DATA_FREE_PREPARED_TRAINING_BATCH_AUTHORITY0_NO_RESULT"


class TrainingBatchError(ValueError):
    """Prepared motion and caption batches do not form one exact batch."""


class PreparedTrainingBatch:
    """Immutable cross-bound motion, caption, and positive-mask snapshot."""

    __slots__ = (
        "__weakref__",
        "batch_index",
        "canonical_receipt",
        "caption_batch",
        "epoch_index",
        "global_step",
        "motion_batch",
        "positive_mask",
        "sampler_manifest_sha256",
        "sampler_seed",
    )

    def __init__(
        self,
        *,
        motion_batch: object,
        caption_batch: object,
        sampler_manifest_sha256: str,
        sampler_seed: int,
        epoch_index: int,
        batch_index: int,
        global_step: int,
    ) -> None:
        if type(motion_batch) is not batching.PreparedMotionBatch:
            raise TypeError("motion_batch must be exactly PreparedMotionBatch")
        checked_motion = batching.validate_prepared_motion_batch(motion_batch)
        caption_record = caption_processing.prepared_caption_batch_components(
            caption_batch
        )
        caption_pairs = caption_record[0]
        caption_ids = caption_record[5]
        if checked_motion.pair_commitments != caption_pairs:
            raise TrainingBatchError(
                "motion and caption pair commitments/order must be identical"
            )
        batch_size = len(checked_motion.pair_commitments)
        if caption_ids.shape[0] != 3 * batch_size:
            raise TrainingBatchError("caption row count must equal three times batch size")
        if type(sampler_manifest_sha256) is not str:
            raise TypeError("sampler_manifest_sha256 must be an exact built-in str")
        if len(sampler_manifest_sha256) != 64 or any(
            character not in "0123456789abcdef"
            for character in sampler_manifest_sha256
        ):
            raise TrainingBatchError(
                "sampler_manifest_sha256 must be lowercase SHA-256 hex"
            )

        def cursor_int(
            value: object,
            label: str,
            *,
            minimum: int,
            maximum: int,
        ) -> int:
            if type(value) is not int:
                raise TypeError(f"{label} must be an exact built-in int")
            if value < minimum or value > maximum:
                raise TrainingBatchError(
                    f"{label} must be inside [{minimum},{maximum}]"
                )
            return value

        seed = cursor_int(
            sampler_seed,
            "sampler_seed",
            minimum=0,
            maximum=2**64 - 1,
        )
        if seed not in (1729, 2718, 31415):
            raise TrainingBatchError("sampler_seed is outside the frozen census")
        epoch = cursor_int(epoch_index, "epoch_index", minimum=0, maximum=29)
        batch = cursor_int(batch_index, "batch_index", minimum=0, maximum=2**32 - 1)
        step = cursor_int(global_step, "global_step", minimum=0, maximum=2**64 - 1)

        mask = np.zeros((batch_size, 3 * batch_size), dtype=np.bool_)
        for row in range(batch_size):
            mask[row, 3 * row : 3 * row + 3] = True
        frozen_mask = np.frombuffer(mask.tobytes(order="C"), dtype=np.bool_).reshape(
            mask.shape
        )
        if not bool(np.all(frozen_mask.sum(axis=1) == 3)) or not bool(
            np.all(frozen_mask.sum(axis=0) == 1)
        ):
            raise AssertionError("internal three-caption positive mask is malformed")

        motion_sha256 = batching.prepared_batch_sha256(checked_motion)
        caption_sha256 = caption_processing.prepared_caption_batch_sha256(
            caption_batch
        )
        positive_mask_sha256 = hashlib.sha256(
            b"phasepair-training-positive-mask-v1\x00"
            + struct.pack(">II", *frozen_mask.shape)
            + frozen_mask.astype(np.uint8, copy=False).tobytes(order="C")
        ).hexdigest()
        pair_order_sha256 = hashlib.sha256(
            b"phasepair-training-pair-order-v1\x00"
            + b"".join(checked_motion.pair_commitments)
        ).hexdigest()
        payload = {
            "authority": AUTHORITY,
            "batch_index": batch,
            "batch_size": batch_size,
            "caption_count": 3 * batch_size,
            "caption_prepared_batch_sha256": caption_sha256,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "epoch_index": epoch,
            "global_step": step,
            "motion_prepared_batch_sha256": motion_sha256,
            "pair_order_sha256": pair_order_sha256,
            "positive_mask_rule": "row_i_positive_columns_3i_through_3i_plus_2",
            "positive_mask_sha256": positive_mask_sha256,
            "positive_mask_shape": [batch_size, 3 * batch_size],
            "production": PRODUCTION,
            "result_claimed": RESULT_CLAIMED,
            "sampler_manifest_sha256": sampler_manifest_sha256,
            "sampler_seed": seed,
            "schema": "phasepair-prepared-training-batch-v1",
            "status": STATUS,
        }
        raw = (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            ).encode("ascii")
            + b"\n"
        )
        object.__setattr__(self, "motion_batch", checked_motion)
        object.__setattr__(self, "caption_batch", caption_batch)
        object.__setattr__(self, "positive_mask", frozen_mask)
        object.__setattr__(self, "canonical_receipt", raw)
        object.__setattr__(self, "sampler_manifest_sha256", sampler_manifest_sha256)
        object.__setattr__(self, "sampler_seed", seed)
        object.__setattr__(self, "epoch_index", epoch)
        object.__setattr__(self, "batch_index", batch)
        object.__setattr__(self, "global_step", step)
        record = (
            checked_motion,
            caption_batch,
            caption_record,
            frozen_mask,
            raw,
            sampler_manifest_sha256,
            seed,
            epoch,
            batch,
            step,
            motion_sha256,
        )
        with _training_batch_lock:
            _training_batch_set(self, record)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PreparedTrainingBatch is immutable")


def prepared_training_batch_bytes(value: object) -> bytes:
    """Return the canonical public-safe Authority-0 cross-binding receipt."""

    return _validated_training_batch_record(value)[4]


def prepared_training_batch_sha256(value: object) -> str:
    """Return the lowercase SHA-256 identity of the cross-binding receipt."""

    return hashlib.sha256(prepared_training_batch_bytes(value)).hexdigest()


_training_batch_lock = threading.RLock()
_training_batch_set, _training_batch_get, _, _, _ = make_identity_weak_registry(
    _training_batch_lock
)


def _validated_training_batch_record(value: object) -> tuple[object, ...]:
    if type(value) is not PreparedTrainingBatch:
        raise TypeError("batch must be exactly PreparedTrainingBatch")
    with _training_batch_lock:
        record = _training_batch_get(value)
        if type(record) is not tuple or len(record) != 11:
            raise TrainingBatchError("training batch was not issued by this module")
        try:
            if (
                object.__getattribute__(value, "motion_batch") is not record[0]
                or object.__getattribute__(value, "caption_batch") is not record[1]
                or object.__getattribute__(value, "positive_mask") is not record[3]
                or object.__getattribute__(value, "canonical_receipt") is not record[4]
                or object.__getattribute__(value, "sampler_manifest_sha256")
                != record[5]
                or object.__getattribute__(value, "sampler_seed") != record[6]
                or object.__getattribute__(value, "epoch_index") != record[7]
                or object.__getattribute__(value, "batch_index") != record[8]
                or object.__getattribute__(value, "global_step") != record[9]
            ):
                raise TrainingBatchError("training batch fields were rebound")
        except AttributeError as exc:
            raise TrainingBatchError("training batch fields were rebound") from exc
        checked_motion = batching.validate_prepared_motion_batch(record[0])
        if batching.prepared_batch_sha256(checked_motion) != record[10]:
            raise TrainingBatchError("nested motion batch changed after issuance")
        current_caption_record = (
            caption_processing.prepared_caption_batch_components(record[1])
        )
        if current_caption_record is not record[2]:
            raise TrainingBatchError("nested caption batch changed after issuance")
        return record


def validate_prepared_training_batch(value: object) -> PreparedTrainingBatch:
    """Prove module issuance and unchanged nested immutable identities."""

    _validated_training_batch_record(value)
    return value


def prepared_training_batch_components(value: object) -> tuple[object, ...]:
    """Return immutable issued components for package-internal admission joins."""

    return _validated_training_batch_record(value)


__all__ = [
    "AUTHORITY",
    "EXECUTION_AUTHORIZED",
    "PRODUCTION",
    "PreparedTrainingBatch",
    "RESULT_CLAIMED",
    "STATUS",
    "TrainingBatchError",
    "prepared_training_batch_bytes",
    "prepared_training_batch_components",
    "prepared_training_batch_sha256",
    "validate_prepared_training_batch",
]
