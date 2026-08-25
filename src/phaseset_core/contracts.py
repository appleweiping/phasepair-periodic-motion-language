"""Strict public skeleton and internal activity contracts for PhaseSet.

The public boundary is :class:`PreparedGroupBatch`: synchronized 22-joint
multi-person skeletons with dynamic actor padding and joint-level tracking
masks.  Actor commitments canonicalize numerical reduction order but are never
neural features.  :class:`PreparedActivityBatch` is a downstream compatibility
seam for the inherited five-channel PhasePair descriptor.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import struct
from typing import Final

import numpy as np


STATUS: Final = "DATA_FREE_GROUP_SKELETON_BATCH_NONPRODUCTION_AUTHORITY0"
GROUP_COMMITMENT_DOMAIN: Final = b"phaseset-group-commitment-v1"
JOINT_COUNT: Final = 22
COORDINATE_DIM: Final = 3
ACTIVITY_DIM: Final = 5
ACTIVITY_SAMPLE_RATE_HZ: Final = 20.0
MAX_PADDED_TIME: Final = 300
MAX_VALID_TIME: Final = 299

# Five deterministic body groups.  The order is part of the inherited 5D
# activity seam and must not be inferred from actor identity.
ACTIVITY_JOINT_GROUPS: Final[tuple[tuple[int, ...], ...]] = (
    (0, 3, 6, 9, 12, 15),
    (13, 16, 18, 20),
    (14, 17, 19, 21),
    (1, 4, 7, 10),
    (2, 5, 8, 11),
)


class GroupContractError(ValueError):
    """Raised when a prepared multi-person batch violates its closed contract."""


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be exact built-in bytes[32]")
    return value


def group_commitment(actor_commitments: tuple[bytes, ...]) -> bytes:
    """Return an actor-order-invariant commitment for one group."""

    if type(actor_commitments) is not tuple or len(actor_commitments) < 2:
        raise TypeError("actor_commitments must be a tuple with at least two actors")
    actors = tuple(
        _raw32(value, f"actor_commitments[{index}]")
        for index, value in enumerate(actor_commitments)
    )
    if len(set(actors)) != len(actors):
        raise GroupContractError("actor commitments must be distinct within a group")
    ordered = tuple(sorted(actors))
    payload = bytearray(GROUP_COMMITMENT_DOMAIN + b"\x00")
    payload.extend(struct.pack(">I", len(ordered)))
    payload.extend(b"".join(ordered))
    return hashlib.sha256(bytes(payload)).digest()


def _immutable_array(value: np.ndarray) -> np.ndarray:
    snapshot = np.array(value, copy=True, order="C", subok=False)
    frozen = np.frombuffer(snapshot.tobytes(order="C"), dtype=snapshot.dtype).reshape(
        snapshot.shape
    )
    if frozen.flags.writeable:
        raise RuntimeError("bytes-backed snapshot unexpectedly remained writeable")
    return frozen


def _require_positive_zero(value: np.ndarray, label: str) -> None:
    if value.dtype != np.dtype(np.float32):
        raise TypeError("positive-zero validation requires float32")
    if value.size and not bool(np.all(value.view(np.uint32) == 0)):
        raise GroupContractError(f"{label} must be exact positive float32 zero")


def _validate_masks_and_commitments(
    *,
    actor_mask: np.ndarray,
    frame_mask: np.ndarray,
    actor_commitments: tuple[tuple[bytes | None, ...], ...],
    group_commitments: tuple[bytes, ...],
    batch_size: int,
    padded_actors: int,
    padded_time: int,
) -> tuple[
    np.ndarray,
    tuple[tuple[tuple[bytes, int], ...], ...],
    tuple[tuple[bytes | None, ...], ...],
    tuple[bytes, ...],
]:
    if type(actor_mask) is not np.ndarray:
        raise TypeError("actor_mask must be an exact base numpy.ndarray")
    if (
        actor_mask.dtype != np.dtype(np.bool_)
        or actor_mask.shape != (batch_size, padded_actors)
        or not actor_mask.flags.c_contiguous
    ):
        raise GroupContractError("actor_mask must be C-contiguous bool [B,K_pad]")
    if type(frame_mask) is not np.ndarray:
        raise TypeError("frame_mask must be an exact base numpy.ndarray")
    if (
        frame_mask.dtype != np.dtype(np.bool_)
        or frame_mask.shape != (batch_size, padded_time)
        or not frame_mask.flags.c_contiguous
    ):
        raise GroupContractError("frame_mask must be C-contiguous bool [B,T]")
    if type(actor_commitments) is not tuple or len(actor_commitments) != batch_size:
        raise TypeError("actor_commitments must contain one exact tuple per batch row")
    if type(group_commitments) is not tuple or len(group_commitments) != batch_size:
        raise TypeError("group_commitments must contain one bytes[32] per batch row")

    canonical_frame_mask = np.array(frame_mask, copy=True, order="C", subok=False)
    orders: list[tuple[tuple[bytes, int], ...]] = []
    canonical_commitments: list[tuple[bytes | None, ...]] = []
    normalized_groups: list[bytes] = []
    for row in range(batch_size):
        valid_length = int(np.count_nonzero(canonical_frame_mask[row]))
        if not 1 <= valid_length <= min(MAX_VALID_TIME, padded_time):
            raise GroupContractError(
                "each frame_mask row must contain 1..min(T,299) valid frames"
            )
        expected_time = np.zeros((padded_time,), dtype=np.bool_)
        expected_time[:valid_length] = True
        if not bool(np.array_equal(canonical_frame_mask[row], expected_time)):
            raise GroupContractError("frame_mask must be an exact contiguous prefix")

        commitments = actor_commitments[row]
        if type(commitments) is not tuple or len(commitments) != padded_actors:
            raise TypeError(
                "each actor_commitments row must be an exact tuple of length K_pad"
            )
        valid_actors: list[tuple[bytes, int]] = []
        for actor in range(padded_actors):
            commitment = commitments[actor]
            if bool(actor_mask[row, actor]):
                valid_actors.append(
                    (_raw32(commitment, f"actor_commitments[{row}][{actor}]"), actor)
                )
            elif commitment is not None:
                raise GroupContractError(
                    "invalid actor positions must carry commitment None"
                )
        if len(valid_actors) < 2:
            raise GroupContractError("every group must contain at least two valid actors")
        actor_keys = tuple(value for value, _ in valid_actors)
        if len(set(actor_keys)) != len(actor_keys):
            raise GroupContractError("actor commitments must be distinct within a group")
        expected_group = group_commitment(actor_keys)
        supplied_group = _raw32(
            group_commitments[row], f"group_commitments[{row}]"
        )
        if supplied_group != expected_group:
            raise GroupContractError("group commitment does not match its valid actor set")
        ordered = tuple(sorted(valid_actors, key=lambda item: item[0]))
        orders.append(ordered)
        canonical_commitments.append(
            tuple(value for value, _ in ordered)
            + (None,) * (padded_actors - len(ordered))
        )
        normalized_groups.append(expected_group)
    return (
        canonical_frame_mask,
        tuple(orders),
        tuple(canonical_commitments),
        tuple(normalized_groups),
    )


@dataclass(frozen=True, slots=True)
class PreparedGroupBatch:
    """Immutable public 22-joint skeleton batch with dynamic actor padding."""

    skeletons: np.ndarray
    actor_mask: np.ndarray
    frame_mask: np.ndarray
    track_mask: np.ndarray
    actor_commitments: tuple[tuple[bytes | None, ...], ...]
    group_commitments: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.skeletons) is not np.ndarray:
            raise TypeError("skeletons must be an exact base numpy.ndarray")
        if (
            self.skeletons.dtype != np.dtype(np.float32)
            or self.skeletons.ndim != 5
            or not self.skeletons.flags.c_contiguous
        ):
            raise GroupContractError(
                "skeletons must be C-contiguous float32 [B,K_pad,T,22,3]"
            )
        batch_size, padded_actors, padded_time, joints, coordinates = (
            self.skeletons.shape
        )
        if batch_size < 1 or padded_actors < 2:
            raise GroupContractError("skeletons must contain B>=1 and K_pad>=2")
        if (
            not 1 <= padded_time <= MAX_PADDED_TIME
            or joints != JOINT_COUNT
            or coordinates != COORDINATE_DIM
        ):
            raise GroupContractError(
                "skeletons must have shape [B,K_pad,T,22,3] with T<=300"
            )
        if not bool(np.isfinite(self.skeletons).all()):
            raise GroupContractError("skeletons must contain only finite values")
        if type(self.track_mask) is not np.ndarray:
            raise TypeError("track_mask must be an exact base numpy.ndarray")
        if (
            self.track_mask.dtype != np.dtype(np.bool_)
            or self.track_mask.shape
            != (batch_size, padded_actors, padded_time, JOINT_COUNT)
            or not self.track_mask.flags.c_contiguous
        ):
            raise GroupContractError(
                "track_mask must be C-contiguous bool [B,K_pad,T,22]"
            )

        frame_mask, orders, commitments, groups = _validate_masks_and_commitments(
            actor_mask=self.actor_mask,
            frame_mask=self.frame_mask,
            actor_commitments=self.actor_commitments,
            group_commitments=self.group_commitments,
            batch_size=batch_size,
            padded_actors=padded_actors,
            padded_time=padded_time,
        )
        source = np.array(self.skeletons, copy=True, order="C", subok=False)
        source_track = np.array(self.track_mask, copy=True, order="C", subok=False)
        canonical = np.zeros_like(source)
        canonical_track = np.zeros_like(source_track)
        canonical_actor_mask = np.zeros((batch_size, padded_actors), dtype=np.bool_)

        for row, ordered in enumerate(orders):
            invalid_frames = ~frame_mask[row]
            for actor in range(padded_actors):
                actor_valid = bool(self.actor_mask[row, actor])
                if not actor_valid:
                    _require_positive_zero(
                        source[row, actor], "invalid actor skeleton padding"
                    )
                    if bool(source_track[row, actor].any()):
                        raise GroupContractError(
                            "invalid actor track_mask padding must be false"
                        )
                    continue
                _require_positive_zero(
                    source[row, actor, invalid_frames],
                    "invalid frame skeleton padding",
                )
                if bool(source_track[row, actor, invalid_frames].any()):
                    raise GroupContractError(
                        "invalid frame track_mask padding must be false"
                    )
                missing = ~source_track[row, actor]
                _require_positive_zero(
                    source[row, actor][missing],
                    "untracked joint skeleton values",
                )
                if not bool(source_track[row, actor, frame_mask[row]].any()):
                    raise GroupContractError(
                        "every valid actor must contain at least one tracked joint"
                    )
            for destination, (_, source_actor) in enumerate(ordered):
                canonical[row, destination] = source[row, source_actor]
                canonical_track[row, destination] = source_track[row, source_actor]
            canonical_actor_mask[row, : len(ordered)] = True

        object.__setattr__(self, "skeletons", _immutable_array(canonical))
        object.__setattr__(self, "actor_mask", _immutable_array(canonical_actor_mask))
        object.__setattr__(self, "frame_mask", _immutable_array(frame_mask))
        object.__setattr__(self, "track_mask", _immutable_array(canonical_track))
        object.__setattr__(self, "actor_commitments", commitments)
        object.__setattr__(self, "group_commitments", groups)

    @property
    def batch_size(self) -> int:
        return int(self.skeletons.shape[0])

    @property
    def padded_actor_count(self) -> int:
        return int(self.skeletons.shape[1])

    @property
    def padded_time(self) -> int:
        return int(self.skeletons.shape[2])

    @property
    def actor_counts(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(row)) for row in self.actor_mask)

    @property
    def valid_lengths(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(row)) for row in self.frame_mask)


@dataclass(frozen=True, slots=True)
class PreparedActivityBatch:
    """Internal 20-Hz five-channel seam for PhaseSet relation descriptors."""

    activities: np.ndarray
    actor_mask: np.ndarray
    frame_mask: np.ndarray
    activity_mask: np.ndarray
    actor_commitments: tuple[tuple[bytes | None, ...], ...]
    group_commitments: tuple[bytes, ...]

    def __post_init__(self) -> None:
        if type(self.activities) is not np.ndarray:
            raise TypeError("activities must be an exact base numpy.ndarray")
        if (
            self.activities.dtype != np.dtype(np.float32)
            or self.activities.ndim != 4
            or not self.activities.flags.c_contiguous
        ):
            raise GroupContractError(
                "activities must be C-contiguous float32 [B,K_pad,T,5]"
            )
        batch_size, padded_actors, padded_time, feature_dim = self.activities.shape
        if batch_size < 1 or padded_actors < 2 or feature_dim != ACTIVITY_DIM:
            raise GroupContractError("activities must contain B>=1, K_pad>=2, D=5")
        if not 1 <= padded_time <= MAX_PADDED_TIME:
            raise GroupContractError("activity T must satisfy 1<=T<=300")
        if not bool(np.isfinite(self.activities).all()):
            raise GroupContractError("activities must contain only finite values")
        if type(self.activity_mask) is not np.ndarray:
            raise TypeError("activity_mask must be an exact base numpy.ndarray")
        if (
            self.activity_mask.dtype != np.dtype(np.bool_)
            or self.activity_mask.shape != self.activities.shape
            or not self.activity_mask.flags.c_contiguous
        ):
            raise GroupContractError(
                "activity_mask must be C-contiguous bool [B,K_pad,T,5]"
            )
        frame_mask, orders, commitments, groups = _validate_masks_and_commitments(
            actor_mask=self.actor_mask,
            frame_mask=self.frame_mask,
            actor_commitments=self.actor_commitments,
            group_commitments=self.group_commitments,
            batch_size=batch_size,
            padded_actors=padded_actors,
            padded_time=padded_time,
        )
        source = np.array(self.activities, copy=True, order="C", subok=False)
        source_mask = np.array(self.activity_mask, copy=True, order="C", subok=False)
        canonical = np.zeros_like(source)
        canonical_mask = np.zeros_like(source_mask)
        canonical_actor_mask = np.zeros((batch_size, padded_actors), dtype=np.bool_)
        for row, ordered in enumerate(orders):
            invalid_frames = ~frame_mask[row]
            for actor in range(padded_actors):
                if not bool(self.actor_mask[row, actor]):
                    _require_positive_zero(source[row, actor], "invalid actor padding")
                    if bool(source_mask[row, actor].any()):
                        raise GroupContractError(
                            "invalid actor activity_mask padding must be false"
                        )
                    continue
                _require_positive_zero(
                    source[row, actor, invalid_frames], "invalid frame padding"
                )
                if bool(source_mask[row, actor, invalid_frames].any()):
                    raise GroupContractError(
                        "invalid frame activity_mask padding must be false"
                    )
                _require_positive_zero(
                    source[row, actor][~source_mask[row, actor]],
                    "masked activity values",
                )
            for destination, (_, source_actor) in enumerate(ordered):
                canonical[row, destination] = source[row, source_actor]
                canonical_mask[row, destination] = source_mask[row, source_actor]
            canonical_actor_mask[row, : len(ordered)] = True
        object.__setattr__(self, "activities", _immutable_array(canonical))
        object.__setattr__(self, "actor_mask", _immutable_array(canonical_actor_mask))
        object.__setattr__(self, "frame_mask", _immutable_array(frame_mask))
        object.__setattr__(self, "activity_mask", _immutable_array(canonical_mask))
        object.__setattr__(self, "actor_commitments", commitments)
        object.__setattr__(self, "group_commitments", groups)

    @property
    def valid_mask(self) -> np.ndarray:
        """Legacy read-only spelling retained only for diagnostic callers."""

        return self.frame_mask

    @property
    def batch_size(self) -> int:
        return int(self.activities.shape[0])

    @property
    def padded_actor_count(self) -> int:
        return int(self.activities.shape[1])

    @property
    def padded_time(self) -> int:
        return int(self.activities.shape[2])

    @property
    def actor_counts(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(row)) for row in self.actor_mask)

    @property
    def valid_lengths(self) -> tuple[int, ...]:
        return tuple(int(np.count_nonzero(row)) for row in self.frame_mask)


def validate_prepared_group_batch(value: object) -> PreparedGroupBatch:
    """Rebuild a public batch so forged dataclass instances fail closed."""

    if type(value) is not PreparedGroupBatch:
        raise TypeError("batch must be exactly PreparedGroupBatch")
    return PreparedGroupBatch(
        value.skeletons,
        value.actor_mask,
        value.frame_mask,
        value.track_mask,
        value.actor_commitments,
        value.group_commitments,
    )


def validate_prepared_activity_batch(value: object) -> PreparedActivityBatch:
    """Rebuild an internal activity batch so forged instances fail closed."""

    if type(value) is not PreparedActivityBatch:
        raise TypeError("batch must be exactly PreparedActivityBatch")
    return PreparedActivityBatch(
        value.activities,
        value.actor_mask,
        value.frame_mask,
        value.activity_mask,
        value.actor_commitments,
        value.group_commitments,
    )


def skeleton_to_activity(batch: PreparedGroupBatch) -> PreparedActivityBatch:
    """Convert 20-fps skeletons to five masked body-group speed channels.

    For each time ``t>0`` and body group, joint speeds are available only when
    the same joint is tracked at both ``t-1`` and ``t``.  The channel is their
    masked mean.  Frame zero and every unavailable channel are exact +0, and the
    derived mask records the missingness rather than treating zero as observed.
    """

    checked = validate_prepared_group_batch(batch)
    values = checked.skeletons
    batch_size, padded_actors, padded_time, _, _ = values.shape
    activities = np.zeros(
        (batch_size, padded_actors, padded_time, ACTIVITY_DIM),
        dtype=np.float32,
    )
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    for row, actor_count in enumerate(checked.actor_counts):
        for actor in range(actor_count):
            for time_index in range(1, checked.valid_lengths[row]):
                if not (
                    bool(checked.frame_mask[row, time_index - 1])
                    and bool(checked.frame_mask[row, time_index])
                ):
                    continue
                for channel, joints in enumerate(ACTIVITY_JOINT_GROUPS):
                    joint_indices = np.asarray(joints, dtype=np.int64)
                    observed = (
                        checked.track_mask[row, actor, time_index - 1, joint_indices]
                        & checked.track_mask[row, actor, time_index, joint_indices]
                    )
                    if not bool(observed.any()):
                        continue
                    selected = joint_indices[observed]
                    displacement = (
                        values[row, actor, time_index, selected]
                        - values[row, actor, time_index - 1, selected]
                    )
                    speeds = np.sqrt(
                        np.sum(
                            displacement.astype(np.float64) ** 2,
                            axis=1,
                            dtype=np.float64,
                        )
                    ) * ACTIVITY_SAMPLE_RATE_HZ
                    activities[row, actor, time_index, channel] = np.float32(
                        speeds.mean(dtype=np.float64)
                    )
                    activity_mask[row, actor, time_index, channel] = True
    return PreparedActivityBatch(
        activities,
        checked.actor_mask,
        checked.frame_mask,
        activity_mask,
        checked.actor_commitments,
        checked.group_commitments,
    )


__all__ = [
    "ACTIVITY_DIM",
    "ACTIVITY_JOINT_GROUPS",
    "ACTIVITY_SAMPLE_RATE_HZ",
    "COORDINATE_DIM",
    "GROUP_COMMITMENT_DOMAIN",
    "GroupContractError",
    "JOINT_COUNT",
    "MAX_PADDED_TIME",
    "MAX_VALID_TIME",
    "PreparedActivityBatch",
    "PreparedGroupBatch",
    "STATUS",
    "group_commitment",
    "skeleton_to_activity",
    "validate_prepared_activity_batch",
    "validate_prepared_group_batch",
]
