"""Freeze and join the complete 30-epoch PhasePair base sampler plan.

The existing sampler derives row-index plans.  This module binds all thirty
epochs to pair and original-caption commitments, records each in-batch caption
permutation, and verifies a prepared training batch against one exact plan
position.  It remains Authority 0 and performs no training or filesystem I/O.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import struct
import threading

import numpy as np

from phasepair_core import sampler, training_batch
from phasepair_core._identity_registry import make_identity_weak_registry


AUTHORITY = 0
PRODUCTION = False
EXECUTION_AUTHORIZED = False
RESULT_CLAIMED = False
STATUS = "DATA_FREE_BASE_SAMPLER_MANIFEST_AUTHORITY0_NO_RESULT"
SEEDS = (1729, 2718, 31415)
EPOCH_COUNT = 30


class SamplerManifestError(ValueError):
    """A full sampler manifest or prepared-batch join is not exact."""


class BaseSamplerManifest:
    """Opaque module-issued full 30-epoch sampler manifest."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise SamplerManifestError(
            "base sampler manifests are minted only by build_base_sampler_manifest"
        )


class SamplerBatchJoin:
    """Opaque module-issued receipt for one batch-to-plan join."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise SamplerManifestError(
            "sampler batch joins are minted only by join_prepared_batch_to_sampler"
        )


@dataclass(frozen=True, slots=True)
class PlannedTrainingBatch:
    """Private commitment identities expected at one sampler cursor."""

    seed: int
    manifest_sha256: str
    epoch_index: int
    batch_index: int
    global_step: int
    pair_row_indices: tuple[int, ...]
    pair_commitments: tuple[bytes, ...]
    caption_ordinal_orders: tuple[tuple[int, int, int], ...]
    ordered_caption_commitments: tuple[tuple[bytes, bytes, bytes], ...]


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be exact built-in bytes[32]")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < minimum or value > maximum:
        raise SamplerManifestError(
            f"{label} must be inside the closed interval [{minimum},{maximum}]"
        )
    return value


def _pair_commitments(value: object) -> tuple[bytes, ...]:
    if type(value) is not tuple or not value:
        raise TypeError("pair_commitments must be an exact nonempty tuple")
    checked = tuple(
        _raw32(item, f"pair_commitments[{index}]")
        for index, item in enumerate(value)
    )
    if any(left >= right for left, right in zip(checked, checked[1:])):
        raise SamplerManifestError(
            "pair_commitments must be strictly raw32 ascending and distinct"
        )
    return checked


def _caption_commitments(
    value: object,
    row_count: int,
) -> tuple[tuple[bytes, bytes, bytes], ...]:
    if type(value) is not tuple or len(value) != row_count:
        raise TypeError(
            "caption_commitments must be an exact tuple with one row per pair"
        )
    rows: list[tuple[bytes, bytes, bytes]] = []
    flattened: list[bytes] = []
    for row_index, row in enumerate(value):
        if type(row) is not tuple or len(row) != 3:
            raise TypeError(
                f"caption_commitments[{row_index}] must be an exact tuple of length 3"
            )
        checked = tuple(
            _raw32(item, f"caption_commitments[{row_index}][{ordinal}]")
            for ordinal, item in enumerate(row)
        )
        rows.append((checked[0], checked[1], checked[2]))
        flattened.extend(checked)
    if len(set(flattened)) != 3 * row_count:
        raise SamplerManifestError("caption commitments must be globally distinct")
    return tuple(rows)


def _anchor_snapshot(value: object, row_count: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("caption_mean_anchors must be an exact numpy.ndarray")
    if value.dtype != np.dtype(np.float64) or value.ndim != 2:
        raise TypeError("caption_mean_anchors must be a rank-2 float64 array")
    if not value.flags.c_contiguous:
        raise SamplerManifestError("caption_mean_anchors must be C-contiguous")
    copied = np.array(value, dtype=np.float64, copy=True, order="C", subok=False)
    if copied.shape[0] != row_count or copied.shape[1] < 1:
        raise SamplerManifestError("caption_mean_anchors shape mismatch")
    if not bool(np.isfinite(copied).all()):
        raise SamplerManifestError("caption_mean_anchors must be finite")
    norms = np.linalg.norm(copied, ord=2, axis=1)
    if not bool(np.all(np.abs(norms - 1.0) <= 1e-12)):
        raise SamplerManifestError("caption_mean_anchors must be L2 normalized")
    return np.frombuffer(copied.astype("<f8", copy=False).tobytes(order="C"), dtype="<f8").reshape(
        copied.shape
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


_manifest_lock = threading.RLock()
_manifest_set, _manifest_get, _, _, _ = make_identity_weak_registry(_manifest_lock)
_join_lock = threading.RLock()
_join_set, _join_get, _, _, _ = make_identity_weak_registry(_join_lock)
_object_new = object.__new__


def build_base_sampler_manifest(
    *,
    pair_commitments: tuple[bytes, ...],
    caption_commitments: tuple[tuple[bytes, bytes, bytes], ...],
    caption_mean_anchors: np.ndarray,
    seed: int,
) -> BaseSamplerManifest:
    """Derive and freeze every base batch and caption order for one seed."""

    pairs = _pair_commitments(pair_commitments)
    captions = _caption_commitments(caption_commitments, len(pairs))
    anchors = _anchor_snapshot(caption_mean_anchors, len(pairs))
    checked_seed = _exact_int(seed, "seed", minimum=0, maximum=2**64 - 1)
    if checked_seed not in SEEDS:
        raise SamplerManifestError("seed is outside the frozen PhasePair census")

    epoch_records: list[
        tuple[
            tuple[tuple[int, ...], ...],
            tuple[tuple[tuple[int, int, int], ...], ...],
            tuple[int, ...],
        ]
    ] = []
    epoch_payloads: list[dict[str, object]] = []
    global_step = 0
    for epoch_index in range(EPOCH_COUNT):
        plan = sampler.build_epoch_batch_plan(
            pairs,
            anchors,
            seed=checked_seed,
            epoch_index=epoch_index,
        )
        batch_caption_orders: list[tuple[tuple[int, int, int], ...]] = []
        batch_global_steps: list[int] = []
        batch_payloads: list[dict[str, object]] = []
        for batch_index, indices in enumerate(plan.batches):
            orders = tuple(
                sampler.caption_permutation(
                    pairs[row_index],
                    captions[row_index],
                    seed=checked_seed,
                    epoch_index=epoch_index,
                )
                for row_index in indices
            )
            batch_caption_orders.append(orders)
            batch_global_steps.append(global_step)
            batch_payloads.append(
                {
                    "batch_index": batch_index,
                    "caption_ordinal_orders": [list(order) for order in orders],
                    "global_step": global_step,
                    "pair_row_indices": list(indices),
                }
            )
            global_step += 1
        epoch_payload = {
            "batches": batch_payloads,
            "epoch_index": epoch_index,
            "eta_hex": sampler.curriculum_eta(epoch_index).hex(),
        }
        epoch_raw = _canonical_json_bytes(epoch_payload)
        epoch_payloads.append(
            {
                **epoch_payload,
                "epoch_plan_sha256": hashlib.sha256(
                    b"phasepair-base-sampler-epoch-v1\x00" + epoch_raw
                ).hexdigest(),
            }
        )
        epoch_records.append(
            (plan.batches, tuple(batch_caption_orders), tuple(batch_global_steps))
        )

    pair_order_sha256 = hashlib.sha256(
        b"phasepair-base-sampler-pair-order-v1\x00" + b"".join(pairs)
    ).hexdigest()
    caption_order_sha256 = hashlib.sha256(
        b"phasepair-base-sampler-caption-original-order-v1\x00"
        + b"".join(item for row in captions for item in row)
    ).hexdigest()
    anchors_sha256 = hashlib.sha256(
        b"phasepair-base-sampler-anchor-f64-v1\x00"
        + struct.pack(">II", *anchors.shape)
        + anchors.astype("<f8", copy=False).tobytes(order="C")
    ).hexdigest()
    private_payload = {
        "anchor_rows_sha256": anchors_sha256,
        "caption_original_order_sha256": caption_order_sha256,
        "epoch_count": EPOCH_COUNT,
        "epochs": epoch_payloads,
        "pair_count": len(pairs),
        "pair_order_sha256": pair_order_sha256,
        "schema": "phasepair-base-sampler-manifest-v1",
        "seed": checked_seed,
        "total_optimizer_steps": global_step,
    }
    private_raw = _canonical_json_bytes(private_payload)
    manifest_sha256 = hashlib.sha256(
        b"phasepair-base-sampler-manifest-v1\x00" + private_raw
    ).hexdigest()
    public_payload = {
        "anchor_rows_sha256": anchors_sha256,
        "authority": AUTHORITY,
        "caption_original_order_sha256": caption_order_sha256,
        "epoch_count": EPOCH_COUNT,
        "execution_authorized": EXECUTION_AUTHORIZED,
        "manifest_sha256": manifest_sha256,
        "pair_count": len(pairs),
        "pair_order_sha256": pair_order_sha256,
        "production": PRODUCTION,
        "result_claimed": RESULT_CLAIMED,
        "schema": "phasepair-base-sampler-manifest-receipt-v1",
        "seed": checked_seed,
        "status": STATUS,
        "total_optimizer_steps": global_step,
    }
    public_raw = _canonical_json_bytes(public_payload)
    record = (
        public_raw,
        private_raw,
        manifest_sha256,
        checked_seed,
        pairs,
        captions,
        tuple(epoch_records),
        tuple(payload["epoch_plan_sha256"] for payload in epoch_payloads),
    )
    value = _object_new(BaseSamplerManifest)
    with _manifest_lock:
        _manifest_set(value, record)
    return value


def _manifest_record(value: object) -> tuple[object, ...]:
    if type(value) is not BaseSamplerManifest:
        raise TypeError("manifest must be exactly BaseSamplerManifest")
    with _manifest_lock:
        record = _manifest_get(value)
    if type(record) is not tuple or len(record) != 8:
        raise SamplerManifestError("manifest was not issued by this module")
    return record


def base_sampler_manifest_receipt_bytes(value: BaseSamplerManifest) -> bytes:
    """Return the canonical public-safe Authority-0 manifest receipt."""

    raw = _manifest_record(value)[0]
    if type(raw) is not bytes:
        raise AssertionError("internal sampler receipt is malformed")
    return raw


def base_sampler_manifest_private_bytes(value: BaseSamplerManifest) -> bytes:
    """Return the canonical private plan bytes for the execution workspace."""

    raw = _manifest_record(value)[1]
    if type(raw) is not bytes:
        raise AssertionError("internal private sampler manifest is malformed")
    return raw


def base_sampler_manifest_sha256(value: BaseSamplerManifest) -> str:
    """Return the domain-separated private manifest identity."""

    digest = _manifest_record(value)[2]
    if type(digest) is not str:
        raise AssertionError("internal sampler manifest digest is malformed")
    return digest


def planned_training_batch(
    manifest: BaseSamplerManifest,
    *,
    epoch_index: int,
    batch_index: int,
) -> PlannedTrainingBatch:
    """Return private expected commitments at one exact plan cursor."""

    record = _manifest_record(manifest)
    epoch = _exact_int(epoch_index, "epoch_index", minimum=0, maximum=29)
    records = record[6]
    if type(records) is not tuple or len(records) != EPOCH_COUNT:
        raise AssertionError("internal epoch record census is malformed")
    batches, orders_by_batch, global_steps = records[epoch]
    if (
        type(batches) is not tuple
        or type(orders_by_batch) is not tuple
        or type(global_steps) is not tuple
    ):
        raise AssertionError("internal batch records are malformed")
    batch = _exact_int(
        batch_index,
        "batch_index",
        minimum=0,
        maximum=len(batches) - 1,
    )
    indices = batches[batch]
    orders = orders_by_batch[batch]
    pairs = record[4]
    captions = record[5]
    expected_pairs = tuple(pairs[index] for index in indices)
    expected_captions = tuple(
        tuple(captions[index][ordinal] for ordinal in order)
        for index, order in zip(indices, orders, strict=True)
    )
    return PlannedTrainingBatch(
        seed=int(record[3]),
        manifest_sha256=str(record[2]),
        epoch_index=epoch,
        batch_index=batch,
        global_step=global_steps[batch],
        pair_row_indices=indices,
        pair_commitments=expected_pairs,
        caption_ordinal_orders=orders,
        ordered_caption_commitments=expected_captions,
    )


def join_prepared_batch_to_sampler(
    manifest: BaseSamplerManifest,
    prepared_batch: training_batch.PreparedTrainingBatch,
    *,
    epoch_index: int,
    batch_index: int,
) -> SamplerBatchJoin:
    """Verify one prepared batch against its pair and caption plan position."""

    prepared_record = training_batch.prepared_training_batch_components(
        prepared_batch
    )
    expected = planned_training_batch(
        manifest,
        epoch_index=epoch_index,
        batch_index=batch_index,
    )
    prepared_motion = prepared_record[0]
    prepared_caption = prepared_record[2]
    if prepared_record[5] != expected.manifest_sha256:
        raise SamplerManifestError("prepared batch manifest identity differs from plan")
    if prepared_record[6] != expected.seed:
        raise SamplerManifestError("prepared batch seed differs from sampler plan")
    if prepared_record[7] != expected.epoch_index:
        raise SamplerManifestError("prepared batch epoch differs from sampler plan")
    if prepared_record[8] != expected.batch_index:
        raise SamplerManifestError("prepared batch index differs from sampler plan")
    if prepared_record[9] != expected.global_step:
        raise SamplerManifestError("prepared batch global step differs from sampler plan")
    if prepared_motion.pair_commitments != expected.pair_commitments:
        raise SamplerManifestError("prepared motion pair order differs from sampler plan")
    if prepared_caption[0] != expected.pair_commitments:
        raise SamplerManifestError("prepared caption pair order differs from sampler plan")
    if prepared_caption[1] != expected.ordered_caption_commitments:
        raise SamplerManifestError(
            "prepared caption commitment order differs from sampler plan"
        )
    if prepared_caption[2] != expected.caption_ordinal_orders:
        raise SamplerManifestError("prepared caption ordinals differ from sampler plan")
    record = _manifest_record(manifest)
    epoch_digests = record[7]
    payload = {
        "authority": AUTHORITY,
        "batch_index": expected.batch_index,
        "epoch_index": expected.epoch_index,
        "epoch_plan_sha256": epoch_digests[expected.epoch_index],
        "execution_authorized": EXECUTION_AUTHORIZED,
        "manifest_sha256": record[2],
        "prepared_training_batch_sha256": (
            training_batch.prepared_training_batch_sha256(prepared_batch)
        ),
        "production": PRODUCTION,
        "result_claimed": RESULT_CLAIMED,
        "schema": "phasepair-sampler-prepared-batch-join-v1",
        "seed": expected.seed,
        "status": STATUS,
    }
    raw = _canonical_json_bytes(payload)
    join = _object_new(SamplerBatchJoin)
    with _join_lock:
        _join_set(join, raw)
    return join


def sampler_batch_join_bytes(value: SamplerBatchJoin) -> bytes:
    """Return the canonical Authority-0 batch-to-plan join receipt."""

    if type(value) is not SamplerBatchJoin:
        raise TypeError("join must be exactly SamplerBatchJoin")
    with _join_lock:
        raw = _join_get(value)
    if type(raw) is not bytes:
        raise SamplerManifestError("join was not issued by this module")
    return raw


def sampler_batch_join_sha256(value: SamplerBatchJoin) -> str:
    """Return the lowercase SHA-256 identity of one batch join receipt."""

    return hashlib.sha256(sampler_batch_join_bytes(value)).hexdigest()


__all__ = [
    "AUTHORITY",
    "BaseSamplerManifest",
    "EPOCH_COUNT",
    "EXECUTION_AUTHORIZED",
    "PRODUCTION",
    "PlannedTrainingBatch",
    "RESULT_CLAIMED",
    "SEEDS",
    "STATUS",
    "SamplerBatchJoin",
    "SamplerManifestError",
    "base_sampler_manifest_private_bytes",
    "base_sampler_manifest_receipt_bytes",
    "base_sampler_manifest_sha256",
    "build_base_sampler_manifest",
    "join_prepared_batch_to_sampler",
    "planned_training_batch",
    "sampler_batch_join_bytes",
    "sampler_batch_join_sha256",
]
