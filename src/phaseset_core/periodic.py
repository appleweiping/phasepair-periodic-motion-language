"""Streaming multi-person periodic relations at the PhaseSet 20-Hz rate."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Iterator

import numpy as np

from phasepair_core import signal as legacy_signal

from .contracts import (
    ACTIVITY_SAMPLE_RATE_HZ,
    PreparedActivityBatch,
    validate_prepared_activity_batch,
)
from . import morlet as phaseset_morlet


STATUS: Final = "DATA_FREE_STREAMED_PAIR_WRAPPER_NONPRODUCTION_AUTHORITY0"
BAND_COUNT: Final = 6
TOKEN_WIDTH: Final = 13
DEFAULT_EDGE_CHUNK_SIZE: Final = 256
CANONICAL_MICROBLOCK_SIZE: Final = 64
MORLET_FREQUENCIES_HZ: Final = phaseset_morlet.MORLET_FREQUENCIES_HZ

if ACTIVITY_SAMPLE_RATE_HZ != phaseset_morlet.SAMPLE_RATE_HZ:
    raise RuntimeError("PHASESET_ACTIVITY_MORLET_RATE_MISMATCH")


class PeriodicContractError(ValueError):
    """Raised when streamed pair construction receives an invalid runtime input."""


class ResourceLimitError(RuntimeError):
    """Fail-closed signal that exact all-edge execution exceeds its budget."""

    code = "RESOURCE_LIMIT"

    def __init__(self, *, required_edges: int, edge_budget: int) -> None:
        self.required_edges = required_edges
        self.edge_budget = edge_budget
        super().__init__("RESOURCE_LIMIT")


def _immutable_array(value: np.ndarray) -> np.ndarray:
    snapshot = np.array(value, copy=True, order="C", subok=False)
    return np.frombuffer(snapshot.tobytes(order="C"), dtype=snapshot.dtype).reshape(
        snapshot.shape
    )


def validate_energy_floors(value: object) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("energy_floors must be an exact base numpy.ndarray")
    if (
        value.dtype != np.dtype(np.float64)
        or value.shape != (BAND_COUNT,)
        or not value.flags.c_contiguous
    ):
        raise PeriodicContractError("energy_floors must be C-contiguous float64[6]")
    if not bool(np.isfinite(value).all()) or bool(np.any(value < 0.0)):
        raise PeriodicContractError("energy_floors must be finite and nonnegative")
    zero = value == 0.0
    if bool(np.any(np.signbit(value[zero]))):
        raise PeriodicContractError("zero energy floors must be exact positive zero")
    return _immutable_array(value)


def validate_edge_chunk_size(value: object) -> int:
    """Return a runtime chunk that preserves 64-edge canonical boundaries."""

    if type(value) is not int:
        raise TypeError("edge_chunk_size must be an exact built-in int")
    if value < CANONICAL_MICROBLOCK_SIZE or value % CANONICAL_MICROBLOCK_SIZE != 0:
        raise PeriodicContractError(
            "edge_chunk_size must be a positive multiple of the 64-edge "
            "canonical microblock"
        )
    return value


def validate_edge_budget(value: object) -> int:
    if type(value) is not int:
        raise TypeError("edge_budget must be an exact built-in int")
    if value < 1:
        raise PeriodicContractError("edge_budget must be positive")
    return value


@dataclass(frozen=True, slots=True)
class _ActorResponse:
    """One actor's fixed-bank response, computed once before edge traversal."""

    transformed: tuple[np.ndarray | None, ...]
    response_mask: tuple[np.ndarray | None, ...]


def _masked_self_power(response: np.ndarray, response_mask: np.ndarray) -> float:
    """Pool one actor's Morlet power without consulting another actor."""

    if (
        response.ndim != 2
        or response.shape[1] != 5
        or response_mask.shape != response.shape
    ):
        raise PeriodicContractError("actor-local Morlet response has invalid shape")
    sample_count, channels = response.shape
    total = 0.0
    count = 0
    for channel in range(channels):
        for time_index in range(sample_count):
            if not bool(response_mask[time_index, channel]):
                continue
            value = complex(response[time_index, channel])
            total = total + value.real * value.real + value.imag * value.imag
            count += 1
    if count == 0:
        return 0.0
    power = total / count
    if not math.isfinite(power) or power < 0.0:
        raise PeriodicContractError("actor-local Morlet power became nonfinite")
    return power


def _masked_cross_fields(
    left: np.ndarray,
    right: np.ndarray,
    left_mask: np.ndarray,
    right_mask: np.ndarray,
) -> tuple[float, float, complex]:
    """Pool only response samples observed for both edge endpoints."""

    if (
        left.shape != right.shape
        or left.ndim != 2
        or left.shape[1] != 5
        or left_mask.shape != left.shape
        or right_mask.shape != right.shape
    ):
        raise PeriodicContractError("paired Morlet responses have inconsistent shapes")
    sample_count, channels = left.shape
    total_left = 0.0
    total_right = 0.0
    total_cross = 0.0 + 0.0j
    count = 0
    for channel in range(channels):
        for time_index in range(sample_count):
            if not (
                bool(left_mask[time_index, channel])
                and bool(right_mask[time_index, channel])
            ):
                continue
            value_left = complex(left[time_index, channel])
            value_right = complex(right[time_index, channel])
            total_left = (
                total_left
                + value_left.real * value_left.real
                + value_left.imag * value_left.imag
            )
            total_right = (
                total_right
                + value_right.real * value_right.real
                + value_right.imag * value_right.imag
            )
            total_cross = total_cross + value_left.conjugate() * value_right
            count += 1
    if count == 0:
        return 0.0, 0.0, 0.0 + 0.0j
    return total_left / count, total_right / count, total_cross / count


def _response_masks_for_actor(
    valid_activity_mask: np.ndarray,
    bands: tuple[phaseset_morlet.PhaseSetMorletBand, ...],
    length_mask: np.ndarray,
) -> tuple[np.ndarray | None, ...]:
    """Derive Morlet-window observation masks without reading signal values."""

    response_masks: list[np.ndarray | None] = []
    for band_index, band in enumerate(bands):
        if not bool(length_mask[band_index]):
            response_masks.append(None)
            continue
        mask_windows = np.lib.stride_tricks.sliding_window_view(
            valid_activity_mask,
            band.length,
            axis=0,
        )
        response_masks.append(
            np.ascontiguousarray(mask_windows.all(axis=-1), dtype=np.bool_)
        )
    return tuple(response_masks)


def _precompute_actor_responses(
    activities: np.ndarray,
    activity_mask: np.ndarray,
    valid_length: int,
) -> tuple[
    tuple[_ActorResponse, ...],
    tuple[phaseset_morlet.PhaseSetMorletBand, ...],
    np.ndarray,
]:
    """Compute each valid actor/band Morlet response exactly once."""

    bands = phaseset_morlet.morlet_kernel_bank()
    length_mask = phaseset_morlet.morlet_length_mask(valid_length)
    actor_rows: list[_ActorResponse] = []
    for actor in range(activities.shape[0]):
        valid_activity = np.asarray(
            activities[actor, :valid_length], dtype=np.float64, order="C"
        )
        transformed_rows: list[np.ndarray | None] = []
        valid_activity_mask = np.asarray(
            activity_mask[actor, :valid_length],
            dtype=np.bool_,
            order="C",
        )
        response_masks = _response_masks_for_actor(
            valid_activity_mask,
            bands,
            length_mask,
        )
        for band_index, band in enumerate(bands):
            if not bool(length_mask[band_index]):
                transformed_rows.append(None)
                continue
            transformed = legacy_signal._sliding_complex_dot(  # noqa: SLF001
                valid_activity,
                band.kernel,
            )
            transformed = np.ascontiguousarray(transformed, dtype=np.complex128)
            transformed_rows.append(transformed)
        actor_rows.append(_ActorResponse(tuple(transformed_rows), response_masks))
    return tuple(actor_rows), bands, length_mask


def _actor_marginal_powers(
    response: _ActorResponse,
    length_mask: np.ndarray,
) -> np.ndarray:
    """Return six actor-local powers from only this actor's response and mask."""

    powers = np.zeros((BAND_COUNT,), dtype=np.float64)
    for band_index in range(BAND_COUNT):
        if not bool(length_mask[band_index]):
            continue
        transformed = response.transformed[band_index]
        response_mask = response.response_mask[band_index]
        if transformed is None or response_mask is None:
            raise AssertionError("length-valid actor-local Morlet response is missing")
        powers[band_index] = _masked_self_power(transformed, response_mask)
    return np.ascontiguousarray(powers)


def _relation_from_cached_responses(
    left: _ActorResponse,
    right: _ActorResponse,
    bands: tuple[phaseset_morlet.PhaseSetMorletBand, ...],
    length_mask: np.ndarray,
    floors: np.ndarray,
) -> legacy_signal.RelationDescriptor:
    """Build the audited 13D descriptor from cached per-actor responses."""

    s_aa = np.zeros((BAND_COUNT,), dtype=np.float64)
    s_bb = np.zeros((BAND_COUNT,), dtype=np.float64)
    s_ab = np.zeros((BAND_COUNT,), dtype=np.complex128)
    coherence = np.zeros((BAND_COUNT,), dtype=np.float64)
    phase = np.zeros((BAND_COUNT,), dtype=np.float64)
    cosine = np.ones((BAND_COUNT,), dtype=np.float64)
    sine = np.zeros((BAND_COUNT,), dtype=np.float64)
    delay = np.zeros((BAND_COUNT,), dtype=np.float64)
    energy_mask = np.zeros((BAND_COUNT,), dtype=np.uint8)
    relation_valid = np.zeros((BAND_COUNT,), dtype=np.uint8)
    signed_phase_valid = np.zeros((BAND_COUNT,), dtype=np.uint8)

    for band_index, band in enumerate(bands):
        if not bool(length_mask[band_index]):
            continue
        transformed_left = left.transformed[band_index]
        transformed_right = right.transformed[band_index]
        mask_left = left.response_mask[band_index]
        mask_right = right.response_mask[band_index]
        if (
            transformed_left is None
            or transformed_right is None
            or mask_left is None
            or mask_right is None
        ):
            raise AssertionError("length-valid Morlet response is missing")
        auto_left, auto_right, cross = _masked_cross_fields(
            transformed_left,
            transformed_right,
            mask_left,
            mask_right,
        )
        if not (
            math.isfinite(auto_left)
            and math.isfinite(auto_right)
            and math.isfinite(cross.real)
            and math.isfinite(cross.imag)
            and auto_left >= 0.0
            and auto_right >= 0.0
        ):
            raise PeriodicContractError("cached Morlet cross fields became nonfinite")
        s_aa[band_index] = auto_left
        s_bb[band_index] = auto_right
        s_ab[band_index] = cross
        # Strict inequality is intentional: an exact-zero response is never a
        # valid edge, including when the diagnostic floor itself is zero.
        energy_mask[band_index] = np.uint8(
            auto_left > floors[band_index] and auto_right > floors[band_index]
        )
        raw_coherence = abs(cross) ** 2 / (
            (auto_left + legacy_signal.TOKEN_EPSILON)
            * (auto_right + legacy_signal.TOKEN_EPSILON)
        )
        coherence[band_index] = legacy_signal.clamp_coherence(raw_coherence)
        threshold = legacy_signal.TOKEN_EPSILON * math.sqrt(auto_left * auto_right)
        relation_valid[band_index] = np.uint8(abs(cross) > threshold)
        if not bool(relation_valid[band_index]):
            continue
        band_phase = legacy_signal.principal_phase(cross)
        phase[band_index] = band_phase
        cosine[band_index] = math.cos(band_phase)
        branch_valid = abs(math.pi - abs(band_phase)) > 1e-6
        signed_phase_valid[band_index] = np.uint8(branch_valid)
        if branch_valid:
            sine[band_index] = math.sin(band_phase)
            delay[band_index] = legacy_signal.wrapped_delay_seconds(
                band_phase,
                band.frequency_hz,
            )

    descriptor_valid = (
        length_mask.astype(np.uint8)
        * energy_mask.astype(np.uint8)
        * relation_valid.astype(np.uint8)
    )
    tokens = np.zeros((BAND_COUNT, TOKEN_WIDTH), dtype=np.float64)
    for band_index, band in enumerate(bands):
        tokens[band_index, :7] = (
            math.log(float(s_aa[band_index]) + legacy_signal.TOKEN_EPSILON),
            math.log(float(s_bb[band_index]) + legacy_signal.TOKEN_EPSILON),
            float(coherence[band_index]),
            float(cosine[band_index]),
            float(sine[band_index]),
            2.0 * band.frequency_hz * float(delay[band_index]),
            float(signed_phase_valid[band_index]),
        )
        tokens[band_index, 7 + band_index] = 1.0
    return legacy_signal.RelationDescriptor(
        s_aa,
        s_bb,
        s_ab,
        coherence,
        phase,
        cosine,
        sine,
        delay,
        np.ascontiguousarray(length_mask, dtype=np.uint8),
        energy_mask,
        relation_valid,
        signed_phase_valid,
        descriptor_valid,
        tokens,
    )


def _analytic_swapped_tokens(
    descriptor: legacy_signal.RelationDescriptor,
    bands: tuple[phaseset_morlet.PhaseSetMorletBand, ...],
) -> np.ndarray:
    """Apply the frozen endpoint-swap algebra without spectral recomputation.

    The legacy public swap helper revalidates a negated wrapped delay against a
    freshly wrapped conjugate phase.  At rare floating-point branch points those
    two analytically equivalent routes can differ by more than its two-ULP
    diagnostic tolerance.  PhaseSet needs the registered algebra itself:
    exchange marginal powers, preserve coherence/cosine, and negate the signed
    phase fields.  This path never changes edge validity or recomputes Morlet.
    """

    tokens = np.zeros((BAND_COUNT, TOKEN_WIDTH), dtype=np.float64)
    for band_index, band in enumerate(bands):
        signed_valid = bool(descriptor.signed_phase_valid[band_index])
        swapped_sine = (
            -float(descriptor.sine[band_index]) if signed_valid else 0.0
        )
        swapped_delay = (
            -float(descriptor.delay_seconds[band_index]) if signed_valid else 0.0
        )
        tokens[band_index, :7] = (
            math.log(float(descriptor.s_bb[band_index]) + legacy_signal.TOKEN_EPSILON),
            math.log(float(descriptor.s_aa[band_index]) + legacy_signal.TOKEN_EPSILON),
            float(descriptor.coherence[band_index]),
            float(descriptor.cosine[band_index]),
            swapped_sine,
            2.0 * band.frequency_hz * swapped_delay,
            float(descriptor.signed_phase_valid[band_index]),
        )
        tokens[band_index, 7 + band_index] = 1.0
    return tokens


@dataclass(frozen=True, slots=True)
class PairChunk:
    """One bounded slice of the canonical unordered-edge stream."""

    batch_indices: np.ndarray
    actor_i: np.ndarray
    actor_j: np.ndarray
    tokens_ij: np.ndarray
    tokens_ji: np.ndarray
    support_mask: np.ndarray

    def __post_init__(self) -> None:
        edge_count = int(self.batch_indices.shape[0])
        for name, value in (
            ("batch_indices", self.batch_indices),
            ("actor_i", self.actor_i),
            ("actor_j", self.actor_j),
        ):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.int64)
                or value.shape != (edge_count,)
                or not value.flags.c_contiguous
            ):
                raise PeriodicContractError(f"{name} must be C-contiguous int64[E]")
        for name, value in (("tokens_ij", self.tokens_ij), ("tokens_ji", self.tokens_ji)):
            if (
                type(value) is not np.ndarray
                or value.dtype != np.dtype(np.float32)
                or value.shape != (edge_count, BAND_COUNT, TOKEN_WIDTH)
                or not value.flags.c_contiguous
                or not bool(np.isfinite(value).all())
            ):
                raise PeriodicContractError(f"{name} must be finite float32[E,6,13]")
        if (
            type(self.support_mask) is not np.ndarray
            or self.support_mask.dtype != np.dtype(np.bool_)
            or self.support_mask.shape != (edge_count, BAND_COUNT)
            or not self.support_mask.flags.c_contiguous
        ):
            raise PeriodicContractError("support_mask must be C-contiguous bool[E,6]")
        if edge_count < 1:
            raise PeriodicContractError("PairChunk cannot be empty")
        if bool(np.any(self.actor_i >= self.actor_j)):
            raise PeriodicContractError("PairChunk must contain canonical i<j edges")
        for name in (
            "batch_indices",
            "actor_i",
            "actor_j",
            "tokens_ij",
            "tokens_ji",
            "support_mask",
        ):
            object.__setattr__(self, name, _immutable_array(getattr(self, name)))


def _flush_chunk(
    batch_indices: list[int],
    actor_i: list[int],
    actor_j: list[int],
    tokens_ij: list[np.ndarray],
    tokens_ji: list[np.ndarray],
    support_masks: list[np.ndarray],
) -> PairChunk:
    return PairChunk(
        np.asarray(batch_indices, dtype=np.int64),
        np.asarray(actor_i, dtype=np.int64),
        np.asarray(actor_j, dtype=np.int64),
        np.ascontiguousarray(np.stack(tokens_ij).astype(np.float32, copy=False)),
        np.ascontiguousarray(np.stack(tokens_ji).astype(np.float32, copy=False)),
        np.ascontiguousarray(np.stack(support_masks).astype(np.bool_, copy=False)),
    )


def _band_id_descriptor() -> np.ndarray:
    tokens = np.zeros((BAND_COUNT, TOKEN_WIDTH), dtype=np.float32)
    tokens[:, 7:] = np.eye(BAND_COUNT, dtype=np.float32)
    return np.ascontiguousarray(tokens)


def _marginal_power_descriptor(
    source_power: np.ndarray,
    target_power: np.ndarray,
) -> np.ndarray:
    """Combine two independently pooled endpoint marginals into one direction."""

    if source_power.shape != (BAND_COUNT,) or target_power.shape != (BAND_COUNT,):
        raise PeriodicContractError("actor marginal powers must have shape [6]")
    tokens = _band_id_descriptor()
    for band_index in range(BAND_COUNT):
        tokens[band_index, 0] = np.float32(
            math.log(float(source_power[band_index]) + legacy_signal.TOKEN_EPSILON)
        )
        tokens[band_index, 1] = np.float32(
            math.log(float(target_power[band_index]) + legacy_signal.TOKEN_EPSILON)
        )
    return tokens


def iter_marginal_power_pair_chunks(
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    edge_budget: int | None = None,
) -> Iterator[PairChunk]:
    """Yield System-02 edges from independently pooled actor marginals.

    Each actor/band power is computed once from only that actor's Morlet
    response and observation mask.  Pair construction merely places the two
    cached self-marginals into the directed endpoint slots.  Support is based
    only on group-window length, so partner observation, energy, phase,
    coherence, and cross-person sample availability cannot enter the stream.
    """

    chunk_size = validate_edge_chunk_size(edge_chunk_size)
    checked = validate_prepared_activity_batch(batch)
    required_edges = sum(
        count * (count - 1) // 2 for count in checked.actor_counts
    )
    if edge_budget is not None:
        budget = validate_edge_budget(edge_budget)
        if required_edges > budget:
            raise ResourceLimitError(
                required_edges=required_edges,
                edge_budget=budget,
            )

    batch_indices: list[int] = []
    actor_i: list[int] = []
    actor_j: list[int] = []
    tokens_ij: list[np.ndarray] = []
    tokens_ji: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []

    for batch_index, actor_count in enumerate(checked.actor_counts):
        valid_length = checked.valid_lengths[batch_index]
        responses, _, length_mask = _precompute_actor_responses(
            checked.activities[batch_index, :actor_count],
            checked.activity_mask[batch_index, :actor_count],
            valid_length,
        )
        marginal_powers = tuple(
            _actor_marginal_powers(response, length_mask) for response in responses
        )
        support = np.ascontiguousarray(length_mask, dtype=np.bool_)
        for left in range(actor_count):
            for right in range(left + 1, actor_count):
                batch_indices.append(batch_index)
                actor_i.append(left)
                actor_j.append(right)
                tokens_ij.append(
                    _marginal_power_descriptor(
                        marginal_powers[left],
                        marginal_powers[right],
                    )
                )
                tokens_ji.append(
                    _marginal_power_descriptor(
                        marginal_powers[right],
                        marginal_powers[left],
                    )
                )
                support_masks.append(support)

                if len(batch_indices) == chunk_size:
                    yield _flush_chunk(
                        batch_indices,
                        actor_i,
                        actor_j,
                        tokens_ij,
                        tokens_ji,
                        support_masks,
                    )
                    batch_indices = []
                    actor_i = []
                    actor_j = []
                    tokens_ij = []
                    tokens_ji = []
                    support_masks = []

    if batch_indices:
        yield _flush_chunk(
            batch_indices,
            actor_i,
            actor_j,
            tokens_ij,
            tokens_ji,
            support_masks,
        )


def iter_observation_availability_pair_chunks(
    batch: PreparedActivityBatch,
    *,
    edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    edge_budget: int | None = None,
) -> Iterator[PairChunk]:
    """Yield System-05 support from observation masks and length alone.

    A pair/band is supported when the band fits the group window and at least
    one Morlet-response position/channel is jointly observed by both endpoints.
    Signal values are never read while constructing either support or tokens.
    """

    chunk_size = validate_edge_chunk_size(edge_chunk_size)
    checked = validate_prepared_activity_batch(batch)
    required_edges = sum(
        count * (count - 1) // 2 for count in checked.actor_counts
    )
    if edge_budget is not None:
        budget = validate_edge_budget(edge_budget)
        if required_edges > budget:
            raise ResourceLimitError(
                required_edges=required_edges,
                edge_budget=budget,
            )

    bands = phaseset_morlet.morlet_kernel_bank()
    generic = _band_id_descriptor()
    batch_indices: list[int] = []
    actor_i: list[int] = []
    actor_j: list[int] = []
    tokens_ij: list[np.ndarray] = []
    tokens_ji: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []

    for batch_index, actor_count in enumerate(checked.actor_counts):
        valid_length = checked.valid_lengths[batch_index]
        length_mask = phaseset_morlet.morlet_length_mask(valid_length)
        response_masks = tuple(
            _response_masks_for_actor(
                np.asarray(
                    checked.activity_mask[batch_index, actor, :valid_length],
                    dtype=np.bool_,
                    order="C",
                ),
                bands,
                length_mask,
            )
            for actor in range(actor_count)
        )
        for left in range(actor_count):
            for right in range(left + 1, actor_count):
                support = np.zeros((BAND_COUNT,), dtype=np.bool_)
                for band_index in range(BAND_COUNT):
                    if not bool(length_mask[band_index]):
                        continue
                    left_mask = response_masks[left][band_index]
                    right_mask = response_masks[right][band_index]
                    if left_mask is None or right_mask is None:
                        raise AssertionError(
                            "length-valid observation response mask is missing"
                        )
                    support[band_index] = bool(
                        np.logical_and(left_mask, right_mask).any()
                    )

                batch_indices.append(batch_index)
                actor_i.append(left)
                actor_j.append(right)
                tokens_ij.append(generic)
                tokens_ji.append(generic)
                support_masks.append(np.ascontiguousarray(support))

                if len(batch_indices) == chunk_size:
                    yield _flush_chunk(
                        batch_indices,
                        actor_i,
                        actor_j,
                        tokens_ij,
                        tokens_ji,
                        support_masks,
                    )
                    batch_indices = []
                    actor_i = []
                    actor_j = []
                    tokens_ij = []
                    tokens_ji = []
                    support_masks = []

    if batch_indices:
        yield _flush_chunk(
            batch_indices,
            actor_i,
            actor_j,
            tokens_ij,
            tokens_ji,
            support_masks,
        )


def iter_unordered_pair_chunks(
    batch: PreparedActivityBatch,
    *,
    energy_floors: np.ndarray,
    edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
    edge_budget: int | None = None,
) -> Iterator[PairChunk]:
    """Yield all valid unordered actor pairs without allocating a dense KxK tensor.

    Actors have already been sorted by commitment by
    :class:`PreparedActivityBatch`.
    Edges therefore appear in exact ``(batch,i,j)`` lexicographic order regardless
    of the caller's original actor order or padding positions.

    The 13D relation algebra remains inherited from PhasePair, but Morlet
    responses use the independent PhaseSet 20-Hz bank.  The PhasePair 30-Hz tap
    bytes and oracle are never modified or relabelled.  This API remains
    data-free and carries no production or scientific-result authority.
    """

    chunk_size = validate_edge_chunk_size(edge_chunk_size)
    checked = validate_prepared_activity_batch(batch)
    floors = validate_energy_floors(energy_floors)
    required_edges = sum(
        count * (count - 1) // 2 for count in checked.actor_counts
    )
    if edge_budget is not None:
        budget = validate_edge_budget(edge_budget)
        if required_edges > budget:
            raise ResourceLimitError(
                required_edges=required_edges,
                edge_budget=budget,
            )

    batch_indices: list[int] = []
    actor_i: list[int] = []
    actor_j: list[int] = []
    tokens_ij: list[np.ndarray] = []
    tokens_ji: list[np.ndarray] = []
    support_masks: list[np.ndarray] = []

    for batch_index, actor_count in enumerate(checked.actor_counts):
        valid_length = checked.valid_lengths[batch_index]
        responses, bands, length_mask = _precompute_actor_responses(
            checked.activities[batch_index, :actor_count],
            checked.activity_mask[batch_index, :actor_count],
            valid_length,
        )
        for left in range(actor_count):
            for right in range(left + 1, actor_count):
                forward = _relation_from_cached_responses(
                    responses[left],
                    responses[right],
                    bands,
                    length_mask,
                    floors,
                )
                reverse_tokens = _analytic_swapped_tokens(forward, bands)
                # Support is deliberately length/energy based.  Low coherence is
                # represented by descriptor fields rather than hidden in degree.
                support = np.logical_and(forward.length_mask != 0, forward.energy_mask != 0)

                batch_indices.append(batch_index)
                actor_i.append(left)
                actor_j.append(right)
                tokens_ij.append(np.ascontiguousarray(forward.tokens, dtype=np.float32))
                tokens_ji.append(np.ascontiguousarray(reverse_tokens, dtype=np.float32))
                support_masks.append(np.ascontiguousarray(support, dtype=np.bool_))

                if len(batch_indices) == chunk_size:
                    yield _flush_chunk(
                        batch_indices,
                        actor_i,
                        actor_j,
                        tokens_ij,
                        tokens_ji,
                        support_masks,
                    )
                    batch_indices = []
                    actor_i = []
                    actor_j = []
                    tokens_ij = []
                    tokens_ji = []
                    support_masks = []

    if batch_indices:
        yield _flush_chunk(
            batch_indices,
            actor_i,
            actor_j,
            tokens_ij,
            tokens_ji,
            support_masks,
        )


def unordered_pair_count(batch: PreparedActivityBatch) -> int:
    checked = validate_prepared_activity_batch(batch)
    return sum(count * (count - 1) // 2 for count in checked.actor_counts)


__all__ = [
    "BAND_COUNT",
    "CANONICAL_MICROBLOCK_SIZE",
    "DEFAULT_EDGE_CHUNK_SIZE",
    "MORLET_FREQUENCIES_HZ",
    "PairChunk",
    "PeriodicContractError",
    "ResourceLimitError",
    "STATUS",
    "TOKEN_WIDTH",
    "iter_marginal_power_pair_chunks",
    "iter_observation_availability_pair_chunks",
    "iter_unordered_pair_chunks",
    "unordered_pair_count",
    "validate_edge_chunk_size",
    "validate_edge_budget",
    "validate_energy_floors",
]
