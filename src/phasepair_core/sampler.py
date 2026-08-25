"""Data-free deterministic curriculum planning for PhasePair base training.

The module returns row-index plans only.  It does not read captions or assets,
does not serialize a production batch manifest, and does not authorize a run.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass

import numpy as np


STATUS = "DATA_FREE_SAMPLER_PLAN_NONPRODUCTION_NO_RECEIPT"
MAX_BATCH = 128
EPOCH_COUNT = 30
ANCHOR_ORDER_DOMAIN = b"mime-anchor-order-v1"
CAPTION_ORDER_DOMAIN = b"mime-caption-order-v1"


class SamplerContractError(ValueError):
    """Raised when the frozen sampler preconditions are violated."""


def _exact_uint(value: object, label: str, *, bits: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < 0 or value >= 1 << bits:
        raise SamplerContractError(f"{label} is outside uint{bits}")
    return value


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be exact built-in bytes[32]")
    return value


def _commitments(value: object) -> tuple[bytes, ...]:
    if type(value) is not tuple or not value:
        raise TypeError("pair_commitments must be a nonempty built-in tuple")
    checked = tuple(_raw32(item, f"pair_commitments[{index}]") for index, item in enumerate(value))
    if any(left >= right for left, right in zip(checked, checked[1:])):
        raise SamplerContractError(
            "pair_commitments must be distinct and strictly raw32 ascending"
        )
    return checked


def _anchor_snapshot(value: object, rows: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("caption_mean_anchors must be an exact base numpy.ndarray")
    if not value.flags.c_contiguous:
        raise SamplerContractError("caption_mean_anchors must be C-contiguous")
    snapshot = np.array(value, copy=True, order="C", subok=False)
    if snapshot.dtype != np.dtype(np.float64) or snapshot.ndim != 2:
        raise TypeError("caption_mean_anchors must be a rank-2 native float64 array")
    if snapshot.shape[0] != rows or snapshot.shape[1] < 1:
        raise SamplerContractError("caption_mean_anchors shape mismatch")
    if not bool(np.isfinite(snapshot).all()):
        raise SamplerContractError("caption_mean_anchors must be finite")
    # ``numpy.linalg.vector_norm`` was added after the retained CPython 3.12
    # compatibility probe's NumPy 1.26.4.  ``norm(..., ord=2, axis=1)`` is the
    # same Euclidean row-vector operation and is available throughout every
    # runtime lane recorded by this project.  Keep ``ord=2`` explicit so this
    # cannot silently become a matrix norm if the snapshot contract changes.
    norms = np.linalg.norm(snapshot, ord=2, axis=1)
    if not bool(np.all(np.abs(norms - 1.0) <= 1e-12)):
        raise SamplerContractError("caption_mean_anchors must already be L2 normalized")
    raw = snapshot.tobytes(order="C")
    frozen = np.frombuffer(raw, dtype=np.dtype("<f8")).reshape(snapshot.shape)
    if frozen.flags.writeable:
        raise RuntimeError("anchor snapshot unexpectedly writeable")
    return frozen


def curriculum_eta(epoch_index: int) -> float:
    epoch = _exact_uint(epoch_index, "epoch_index", bits=32)
    if epoch >= 30:
        raise SamplerContractError("epoch_index must be in [0,29]")
    if epoch < 3:
        return 0.0
    if epoch >= 13:
        return 0.25
    tau = (epoch - 2) / 10.0
    return 0.25 * (1.0 - math.cos(math.pi * tau)) / 2.0


def hardness_target(epoch_index: int, remaining_count: int) -> int:
    epoch = _exact_uint(epoch_index, "epoch_index", bits=32)
    if epoch < 3 or epoch >= 30:
        raise SamplerContractError("hardness target requires epoch_index in [3,29]")
    remaining = _exact_uint(remaining_count, "remaining_count", bits=32)
    if remaining < 1:
        raise SamplerContractError("remaining_count must be positive")
    return math.floor((1.0 - curriculum_eta(epoch)) * (remaining - 1))


def hardness_window(epoch_index: int, remaining_count: int) -> tuple[int, int, int]:
    remaining = _exact_uint(remaining_count, "remaining_count", bits=32)
    if remaining < 1:
        raise SamplerContractError("remaining_count must be positive")
    width = min(127, remaining)
    target = hardness_target(epoch_index, remaining)
    start = min(max(target - math.floor((width - 1) / 2), 0), remaining - width)
    return target, start, width


def anchor_permutation(
    pair_commitments: tuple[bytes, ...],
    *,
    seed: int,
    epoch_index: int,
) -> tuple[int, ...]:
    commitments = _commitments(pair_commitments)
    checked_seed = _exact_uint(seed, "seed", bits=64)
    epoch = _exact_uint(epoch_index, "epoch_index", bits=32)
    if epoch >= 30:
        raise SamplerContractError("epoch_index must be in [0,29]")
    prefix = b"mime-anchor-order-v1" + struct.pack(">QI", checked_seed, epoch)
    return tuple(
        sorted(
            range(len(commitments)),
            key=lambda index: (hashlib.sha256(prefix + commitments[index]).digest(), commitments[index]),
        )
    )


def caption_permutation(
    pair_commitment: bytes,
    caption_commitments: tuple[bytes, bytes, bytes],
    *,
    seed: int,
    epoch_index: int,
) -> tuple[int, int, int]:
    pair = _raw32(pair_commitment, "pair_commitment")
    if type(caption_commitments) is not tuple or len(caption_commitments) != 3:
        raise TypeError("caption_commitments must be a built-in tuple of length 3")
    captions = tuple(
        _raw32(item, f"caption_commitments[{index}]")
        for index, item in enumerate(caption_commitments)
    )
    if len(set(captions)) != 3:
        raise SamplerContractError("caption commitments must be distinct")
    checked_seed = _exact_uint(seed, "seed", bits=64)
    epoch = _exact_uint(epoch_index, "epoch_index", bits=32)
    if epoch >= 30:
        raise SamplerContractError("epoch_index must be in [0,29]")
    prefix = b"mime-caption-order-v1" + struct.pack(">QI", checked_seed, epoch) + pair
    ordered = sorted(
        range(3),
        key=lambda index: (
            hashlib.sha256(prefix + captions[index]).digest(),
            captions[index],
        ),
    )
    return (ordered[0], ordered[1], ordered[2])


def _derive_batches(
    commitments: tuple[bytes, ...],
    anchors: np.ndarray,
    *,
    seed: int,
    epoch: int,
) -> tuple[tuple[int, ...], ...]:
    permutation = anchor_permutation(commitments, seed=seed, epoch_index=epoch)
    if epoch < 3:
        return tuple(
            tuple(permutation[start : start + 128])
            for start in range(0, len(permutation), 128)
        )

    remaining = set(range(len(commitments)))
    batches_list: list[tuple[int, ...]] = []
    cursor = 0
    while remaining:
        while permutation[cursor] not in remaining:
            cursor += 1
        anchor_index = permutation[cursor]
        remaining.remove(anchor_index)
        if not remaining:
            batches_list.append((anchor_index,))
            break
        ranked = sorted(
            remaining,
            key=lambda index: (
                -float(np.dot(anchors[anchor_index], anchors[index])),
                commitments[index],
            ),
        )
        _, start, width = hardness_window(epoch, len(ranked))
        selected = tuple(ranked[start : start + width])
        for index in selected:
            remaining.remove(index)
        batches_list.append((anchor_index, *selected))
    return tuple(batches_list)


def _validate_batches(batches: tuple[tuple[int, ...], ...]) -> None:
    if type(batches) is not tuple or not batches:
        raise TypeError("batches must be a nonempty built-in tuple")
    flattened: list[int] = []
    for batch_index, batch in enumerate(batches):
        if type(batch) is not tuple or not batch or len(batch) > 128:
            raise SamplerContractError(f"invalid batch at index {batch_index}")
        for row_index, item in enumerate(batch):
            flattened.append(
                _exact_uint(item, f"batches[{batch_index}][{row_index}]", bits=32)
            )
    if len(flattened) != len(set(flattened)):
        raise SamplerContractError("batch plan repeats a row")
    if set(flattened) != set(range(len(flattened))):
        raise SamplerContractError("batch plan must cover each row exactly once")
    expected_sizes = [128] * (len(flattened) // 128)
    if len(flattened) % 128:
        expected_sizes.append(len(flattened) % 128)
    if [len(batch) for batch in batches] != expected_sizes:
        raise SamplerContractError("batch-size sequence mismatch")


@dataclass(frozen=True, slots=True, init=False)
class EpochBatchPlan:
    seed: int
    epoch_index: int
    batches: tuple[tuple[int, ...], ...]
    status: str

    def __init__(
        self,
        pair_commitments: tuple[bytes, ...],
        caption_mean_anchors: np.ndarray,
        *,
        seed: int,
        epoch_index: int,
    ) -> None:
        """Derive the plan from frozen inputs; precomputed batches are not accepted."""

        commitments = _commitments(pair_commitments)
        anchors = _anchor_snapshot(caption_mean_anchors, len(commitments))
        checked_seed = _exact_uint(seed, "seed", bits=64)
        epoch = _exact_uint(epoch_index, "epoch_index", bits=32)
        if epoch >= 30:
            raise SamplerContractError("epoch_index must be in [0,29]")
        batches = _derive_batches(
            commitments,
            anchors,
            seed=checked_seed,
            epoch=epoch,
        )
        _validate_batches(batches)
        object.__setattr__(self, "seed", checked_seed)
        object.__setattr__(self, "epoch_index", epoch)
        object.__setattr__(self, "batches", batches)
        object.__setattr__(
            self,
            "status",
            "DATA_FREE_SAMPLER_PLAN_NONPRODUCTION_NO_RECEIPT",
        )


def build_epoch_batch_plan(
    pair_commitments: tuple[bytes, ...],
    caption_mean_anchors: np.ndarray,
    *,
    seed: int,
    epoch_index: int,
) -> EpochBatchPlan:
    return EpochBatchPlan(
        pair_commitments,
        caption_mean_anchors,
        seed=seed,
        epoch_index=epoch_index,
    )


__all__ = [
    "EPOCH_COUNT",
    "EpochBatchPlan",
    "SamplerContractError",
    "STATUS",
    "anchor_permutation",
    "build_epoch_batch_plan",
    "caption_permutation",
    "curriculum_eta",
    "hardness_target",
    "hardness_window",
]
