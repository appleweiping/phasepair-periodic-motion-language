"""Strict, data-free batching for PhasePair motion towers.

This module accepts already processed 262D actor features and 799D relation
features.  It never opens a dataset, resolves captions, or infers lineage.  Its
only job is to snapshot a verified in-memory batch and construct the two
ordered actor passes required by the frozen scientific contract.
"""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np


STATUS = "DATA_FREE_BATCH_CONTRACT_VALIDATED_NONPRODUCTION"
IDENTITY_DOMAIN = b"phasepair-prepared-motion-batch-v1"
MAX_BATCH = 128
MAX_PADDED_TIME = 300
MAX_VALID_TIME = 299
ACTOR_DIM = 262
RELATION_DIM = 799
EARLY_DIM = 3 * ACTOR_DIM


class BatchContractError(ValueError):
    """Raised when a prepared batch violates the closed local contract."""


def _require_exact_int(value: object, label: str, *, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < minimum or value > maximum:
        raise BatchContractError(f"{label} is outside [{minimum},{maximum}]")
    return value


def _require_raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be exact built-in bytes[32]")
    return value


def _immutable_array(
    value: np.ndarray,
    _array: object = np.array,
    _frombuffer: object = np.frombuffer,
) -> np.ndarray:
    snapshot = _array(value, copy=True, order="C", subok=False)
    raw = snapshot.tobytes(order="C")
    frozen = _frombuffer(raw, dtype=snapshot.dtype).reshape(snapshot.shape)
    if frozen.flags.writeable:
        raise RuntimeError("bytes-backed snapshot unexpectedly remained writeable")
    return frozen


def _require_float32_tensor(
    value: object,
    label: str,
    *,
    batch_size: int | None = None,
    padded_time: int | None = None,
    feature_dim: int,
    _snapshot: object = _immutable_array,
    _ndarray_type: type[np.ndarray] = np.ndarray,
    _float32_dtype: np.dtype[np.float32] = np.dtype(np.float32),
    _isfinite: object = np.isfinite,
    _require_int: object = _require_exact_int,
) -> np.ndarray:
    if type(value) is not _ndarray_type:
        raise TypeError(f"{label} must be an exact base numpy.ndarray")
    if not value.flags.c_contiguous:
        raise BatchContractError(f"{label} must be C-contiguous")
    snapshot = _snapshot(value)
    if snapshot.dtype != _float32_dtype:
        raise TypeError(f"{label} must have native float32 dtype")
    if snapshot.ndim != 3:
        raise BatchContractError(f"{label} must have rank 3")
    b, t, d = snapshot.shape
    if batch_size is not None and b != batch_size:
        raise BatchContractError(f"{label} batch dimension mismatch")
    if padded_time is not None and t != padded_time:
        raise BatchContractError(f"{label} time dimension mismatch")
    _require_int(int(b), f"{label} batch size", minimum=1, maximum=128)
    _require_int(int(t), f"{label} padded time", minimum=1, maximum=300)
    if d != feature_dim:
        raise BatchContractError(f"{label} feature dimension must be {feature_dim}")
    if not bool(_isfinite(snapshot).all()):
        raise BatchContractError(f"{label} must contain only finite values")
    return snapshot


def _require_lengths(
    value: object,
    batch_size: int,
    padded_time: int,
    _require_int: object = _require_exact_int,
) -> tuple[int, ...]:
    if type(value) is not tuple or len(value) != batch_size:
        raise TypeError("valid_lengths must be a built-in tuple with one item per row")
    return tuple(
        _require_int(
            length,
            f"valid_lengths[{index}]",
            minimum=1,
            maximum=min(299, padded_time),
        )
        for index, length in enumerate(value)
    )


def _require_mask(
    value: object,
    *,
    batch_size: int,
    padded_time: int,
    lengths: tuple[int, ...],
    _snapshot: object = _immutable_array,
    _ndarray_type: type[np.ndarray] = np.ndarray,
    _uint8_dtype: np.dtype[np.uint8] = np.dtype(np.uint8),
    _logical_or: object = np.logical_or,
    _zeros: object = np.zeros,
    _array_equal: object = np.array_equal,
) -> np.ndarray:
    if type(value) is not _ndarray_type:
        raise TypeError("valid_mask must be an exact base numpy.ndarray")
    if not value.flags.c_contiguous:
        raise BatchContractError("valid_mask shape/layout mismatch")
    snapshot = _snapshot(value)
    if snapshot.dtype != _uint8_dtype:
        raise TypeError("valid_mask must have uint8 dtype")
    if snapshot.shape != (batch_size, padded_time):
        raise BatchContractError("valid_mask shape/layout mismatch")
    if not bool(_logical_or(snapshot == 0, snapshot == 1).all()):
        raise BatchContractError("valid_mask must be binary")
    expected = _zeros((batch_size, padded_time), dtype=_uint8_dtype)
    for row, length in enumerate(lengths):
        expected[row, :length] = 1
    if not bool(_array_equal(snapshot, expected)):
        raise BatchContractError("valid_mask must be the exact valid-prefix mask")
    return snapshot


def _require_commitments(
    value: object,
    batch_size: int,
    _require_commitment: object = _require_raw32,
) -> tuple[bytes, ...]:
    if type(value) is not tuple or len(value) != batch_size:
        raise TypeError("pair_commitments must be a built-in tuple with one item per row")
    commitments = tuple(
        _require_commitment(item, f"pair_commitments[{index}]")
        for index, item in enumerate(value)
    )
    if len(set(commitments)) != len(commitments):
        raise BatchContractError("pair commitments must be distinct")
    return commitments


def _require_positive_zero_padding(
    value: np.ndarray,
    lengths: tuple[int, ...],
    label: str,
    _uint32_dtype: type[np.uint32] = np.uint32,
    _all: object = np.all,
) -> None:
    words = value.view(_uint32_dtype)
    for row, length in enumerate(lengths):
        if length < value.shape[1] and not bool(_all(words[row, length:, :] == 0)):
            raise BatchContractError(f"{label} invalid padding must be positive float32 zero")


@dataclass(frozen=True, slots=True)
class PreparedMotionBatch:
    actor_a: np.ndarray
    actor_b: np.ndarray
    relation_ab: np.ndarray
    relation_ba: np.ndarray
    valid_mask: np.ndarray
    valid_lengths: tuple[int, ...]
    pair_commitments: tuple[bytes, ...]

    def __post_init__(
        self,
        _require_tensor: object = _require_float32_tensor,
        _require_batch_lengths: object = _require_lengths,
        _require_batch_mask: object = _require_mask,
        _require_batch_commitments: object = _require_commitments,
        _require_padding: object = _require_positive_zero_padding,
    ) -> None:
        actor_a = _require_tensor(self.actor_a, "actor_a", feature_dim=262)
        b, t, _ = actor_a.shape
        actor_b = _require_tensor(
            self.actor_b,
            "actor_b",
            batch_size=b,
            padded_time=t,
            feature_dim=262,
        )
        relation_ab = _require_tensor(
            self.relation_ab,
            "relation_ab",
            batch_size=b,
            padded_time=t,
            feature_dim=799,
        )
        relation_ba = _require_tensor(
            self.relation_ba,
            "relation_ba",
            batch_size=b,
            padded_time=t,
            feature_dim=799,
        )
        lengths = _require_batch_lengths(self.valid_lengths, b, t)
        mask = _require_batch_mask(
            self.valid_mask,
            batch_size=b,
            padded_time=t,
            lengths=lengths,
        )
        commitments = _require_batch_commitments(self.pair_commitments, b)
        for label, tensor in (
            ("actor_a", actor_a),
            ("actor_b", actor_b),
            ("relation_ab", relation_ab),
            ("relation_ba", relation_ba),
        ):
            _require_padding(tensor, lengths, label)
        object.__setattr__(self, "actor_a", actor_a)
        object.__setattr__(self, "actor_b", actor_b)
        object.__setattr__(self, "relation_ab", relation_ab)
        object.__setattr__(self, "relation_ba", relation_ba)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "valid_lengths", lengths)
        object.__setattr__(self, "pair_commitments", commitments)


def _close_post_init(implementation: object) -> object:
    def __post_init__(self: object) -> None:
        implementation(self)

    return __post_init__


PreparedMotionBatch.__post_init__ = _close_post_init(PreparedMotionBatch.__post_init__)


def _make_batch_validator(batch_type: type[PreparedMotionBatch]) -> object:
    def validate_prepared_motion_batch(value: object) -> PreparedMotionBatch:
        """Rebuild a prepared batch so forged dataclass instances cannot bypass checks."""

        if type(value) is not batch_type:
            raise TypeError("batch must be exactly PreparedMotionBatch")
        return batch_type(
            value.actor_a,
            value.actor_b,
            value.relation_ab,
            value.relation_ba,
            value.valid_mask,
            value.valid_lengths,
            value.pair_commitments,
        )

    return validate_prepared_motion_batch


validate_prepared_motion_batch = _make_batch_validator(PreparedMotionBatch)
del _make_batch_validator


@dataclass(frozen=True, slots=True)
class OrderedMotionPasses:
    actor_role_a: np.ndarray
    actor_role_b: np.ndarray
    relation: np.ndarray
    early_fusion: np.ndarray
    valid_mask: np.ndarray
    pair_commitments: tuple[bytes, ...]

    def __post_init__(
        self,
        _snapshot: object = _immutable_array,
        _ndarray_type: type[np.ndarray] = np.ndarray,
        _float32_dtype: np.dtype[np.float32] = np.dtype(np.float32),
        _uint8_dtype: np.dtype[np.uint8] = np.dtype(np.uint8),
        _uint32_dtype: type[np.uint32] = np.uint32,
        _uint64_dtype: np.dtype[np.uint64] = np.dtype(np.uint64),
        _isfinite: object = np.isfinite,
        _array_equal: object = np.array_equal,
        _concatenate: object = np.concatenate,
        _subtract: object = np.subtract,
        _errstate: object = np.errstate,
        _all: object = np.all,
        _zeros: object = np.zeros,
        _require_int: object = _require_exact_int,
        _require_batch_commitments: object = _require_commitments,
    ) -> None:
        arrays = (
            ("actor_role_a", self.actor_role_a, 262),
            ("actor_role_b", self.actor_role_b, 262),
            ("relation", self.relation, 799),
            ("early_fusion", self.early_fusion, 786),
        )
        snapshots: dict[str, np.ndarray] = {}
        common_shape: tuple[int, int, int] | None = None
        for label, value, width in arrays:
            if type(value) is not _ndarray_type:
                raise TypeError(f"{label} must be an exact base numpy.ndarray")
            if not value.flags.c_contiguous:
                raise BatchContractError(f"{label} must be C-contiguous")
            checked = _snapshot(value)
            if checked.dtype != _float32_dtype or checked.ndim != 4:
                raise BatchContractError(f"{label} must be rank-4 native float32")
            if checked.shape[0] != 2 or checked.shape[3] != width:
                raise BatchContractError(f"{label} shape mismatch")
            shape = checked.shape[:3]
            if common_shape is None:
                common_shape = shape
            elif shape != common_shape:
                raise BatchContractError("ordered pass tensor shapes disagree")
            if not bool(_isfinite(checked).all()):
                raise BatchContractError(f"{label} must contain only finite values")
            snapshots[label] = checked

        if common_shape is None:
            raise RuntimeError("ordered pass shape validation did not run")
        _, batch_size, padded_time = common_shape
        _require_int(batch_size, "ordered pass batch size", minimum=1, maximum=128)
        _require_int(padded_time, "ordered pass padded time", minimum=1, maximum=300)

        if type(self.valid_mask) is not _ndarray_type:
            raise TypeError("valid_mask must be an exact base numpy.ndarray")
        if not self.valid_mask.flags.c_contiguous:
            raise BatchContractError("valid_mask must be C-contiguous")
        mask = _snapshot(self.valid_mask)
        if mask.dtype != _uint8_dtype or mask.shape != (batch_size, padded_time):
            raise BatchContractError("valid_mask shape/dtype mismatch")
        if not bool(_all((mask == 0) | (mask == 1))):
            raise BatchContractError("valid_mask must be binary")
        for row in range(batch_size):
            length = int(mask[row].sum(dtype=_uint64_dtype))
            _require_int(
                length,
                f"valid_mask[{row}] length",
                minimum=1,
                maximum=min(299, padded_time),
            )
            expected_row = _zeros((padded_time,), dtype=_uint8_dtype)
            expected_row[:length] = 1
            if not bool(_array_equal(mask[row], expected_row)):
                raise BatchContractError("valid_mask must be the exact valid-prefix mask")

        role_a = snapshots["actor_role_a"]
        role_b = snapshots["actor_role_b"]
        if not bool(_array_equal(role_a[0], role_b[1])) or not bool(
            _array_equal(role_a[1], role_b[0])
        ):
            raise BatchContractError("ordered actor passes must be exact AB/BA swaps")
        with _errstate(over="ignore", invalid="ignore"):
            difference = _subtract(role_b, role_a, dtype=_float32_dtype)
        if not bool(_isfinite(difference).all()):
            raise BatchContractError("ordered actor difference must remain finite")
        expected_early = _concatenate((role_a, role_b, difference), axis=-1)
        if snapshots["early_fusion"].tobytes(order="C") != expected_early.tobytes(order="C"):
            raise BatchContractError("early_fusion must exactly equal [a;b;b-a]")

        invalid = mask == 0
        for label, checked in snapshots.items():
            words = checked.view(_uint32_dtype)
            if not bool(_all(words[:, invalid, :] == 0)):
                raise BatchContractError(f"{label} padding must be positive float32 zero")

        commitments = _require_batch_commitments(self.pair_commitments, batch_size)
        for label, checked in snapshots.items():
            object.__setattr__(self, label, checked)
        object.__setattr__(self, "valid_mask", mask)
        object.__setattr__(self, "pair_commitments", commitments)


OrderedMotionPasses.__post_init__ = _close_post_init(OrderedMotionPasses.__post_init__)
del _close_post_init


def _make_ordered_pass_mint(output_type: type[OrderedMotionPasses]) -> object:
    def mint(
        actor_role_a: np.ndarray,
        actor_role_b: np.ndarray,
        relation: np.ndarray,
        early_fusion: np.ndarray,
        valid_mask: np.ndarray,
        pair_commitments: tuple[bytes, ...],
    ) -> OrderedMotionPasses:
        result = object.__new__(output_type)
        object.__setattr__(result, "actor_role_a", actor_role_a)
        object.__setattr__(result, "actor_role_b", actor_role_b)
        object.__setattr__(result, "relation", relation)
        object.__setattr__(result, "early_fusion", early_fusion)
        object.__setattr__(result, "valid_mask", valid_mask)
        object.__setattr__(result, "pair_commitments", pair_commitments)
        return result

    return mint


_mint_ordered_motion_passes = _make_ordered_pass_mint(OrderedMotionPasses)
del _make_ordered_pass_mint


def _freeze_float32(
    value: np.ndarray,
    _ascontiguousarray: object = np.ascontiguousarray,
    _float32_dtype: np.dtype[np.float32] = np.dtype(np.float32),
    _snapshot: object = _immutable_array,
) -> np.ndarray:
    contiguous = _ascontiguousarray(value, dtype=_float32_dtype)
    return _snapshot(contiguous)


def _make_ordered_pass_builder(
    validate: object,
    mint: object,
    stack: object,
    concatenate: object,
    subtract: object,
    isfinite: object,
    errstate: object,
    float32_dtype: np.dtype[np.float32],
    positive_zero: np.float32,
    freeze: object,
) -> object:
    def build_ordered_motion_passes(batch: PreparedMotionBatch) -> OrderedMotionPasses:
        """Construct the exact `[AB, BA]` leading pass axis without reading external state."""

        checked = validate(batch)
        role_a = stack((checked.actor_a, checked.actor_b), axis=0)
        role_b = stack((checked.actor_b, checked.actor_a), axis=0)
        relation = stack((checked.relation_ab, checked.relation_ba), axis=0)
        with errstate(over="ignore", invalid="ignore"):
            difference = subtract(role_b, role_a, dtype=float32_dtype)
        if not bool(isfinite(difference).all()):
            raise BatchContractError("ordered actor difference must remain finite")
        early = concatenate((role_a, role_b, difference), axis=-1)
        invalid = checked.valid_mask == 0
        for tensor in (role_a, role_b, relation, early):
            tensor[:, invalid, :] = positive_zero
        return mint(
            freeze(role_a),
            freeze(role_b),
            freeze(relation),
            freeze(early),
            checked.valid_mask,
            checked.pair_commitments,
        )

    return build_ordered_motion_passes


build_ordered_motion_passes = _make_ordered_pass_builder(
    validate_prepared_motion_batch,
    _mint_ordered_motion_passes,
    np.stack,
    np.concatenate,
    np.subtract,
    np.isfinite,
    np.errstate,
    np.dtype(np.float32),
    np.float32(0.0),
    _freeze_float32,
)
del _make_ordered_pass_builder


def _make_batch_identity(validate: object, domain: bytes, pack: object) -> object:
    def prepared_batch_identity_bytes(batch: PreparedMotionBatch) -> bytes:
        """Bind the caller's frozen row order in a validated in-memory batch identity."""

        checked = validate(batch)
        b, t, _ = checked.actor_a.shape
        payload = bytearray(domain + b"\x00")
        payload.extend(pack(">HH", b, t))
        for length in checked.valid_lengths:
            payload.extend(pack(">H", length))
        payload.extend(b"".join(checked.pair_commitments))
        payload.extend(checked.valid_mask.tobytes(order="C"))
        for tensor in (
            checked.actor_a,
            checked.actor_b,
            checked.relation_ab,
            checked.relation_ba,
        ):
            payload.extend(tensor.astype("<f4", copy=False).tobytes(order="C"))
        return bytes(payload)

    return prepared_batch_identity_bytes


prepared_batch_identity_bytes = _make_batch_identity(
    validate_prepared_motion_batch,
    b"phasepair-prepared-motion-batch-v1",
    struct.pack,
)
del _make_batch_identity


def _make_batch_digest(identity_bytes: object, sha256: object) -> object:
    def prepared_batch_sha256(batch: PreparedMotionBatch) -> str:
        return sha256(identity_bytes(batch)).hexdigest()

    return prepared_batch_sha256


prepared_batch_sha256 = _make_batch_digest(prepared_batch_identity_bytes, hashlib.sha256)
del _make_batch_digest


__all__ = [
    "ACTOR_DIM",
    "BatchContractError",
    "EARLY_DIM",
    "IDENTITY_DOMAIN",
    "MAX_BATCH",
    "MAX_PADDED_TIME",
    "MAX_VALID_TIME",
    "OrderedMotionPasses",
    "PreparedMotionBatch",
    "RELATION_DIM",
    "STATUS",
    "build_ordered_motion_passes",
    "prepared_batch_identity_bytes",
    "prepared_batch_sha256",
    "validate_prepared_motion_batch",
]
