"""Tests for the streamed all-pairs wrapper."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from phasepair_core import signal as legacy_signal
from phaseset_core.contracts import PreparedActivityBatch, group_commitment
from phaseset_core.periodic import (
    DEFAULT_EDGE_CHUNK_SIZE,
    PeriodicContractError,
    ResourceLimitError,
    iter_unordered_pair_chunks,
    unordered_pair_count,
)


def _key(index: int) -> bytes:
    return hashlib.sha256(f"actor-{index}".encode("ascii")).digest()


def _batch(actor_count: int = 4) -> PreparedActivityBatch:
    time = 17
    activities = np.zeros((1, actor_count, time, 5), dtype=np.float32)
    grid = np.arange(16, dtype=np.float32)
    for actor in range(actor_count):
        for channel in range(5):
            activities[0, actor, :16, channel] = np.sin(
                np.float32(0.17 + 0.01 * actor) * grid
                + np.float32(0.2 * channel + 0.1 * actor)
            )
    actor_mask = np.ones((1, actor_count), dtype=np.bool_)
    valid_mask = np.zeros((1, time), dtype=np.bool_)
    valid_mask[:, :16] = True
    keys = tuple(_key(index) for index in range(actor_count))
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    activity_mask[:, :, :16] = True
    return PreparedActivityBatch(
        activities,
        actor_mask,
        valid_mask,
        activity_mask,
        (keys,),
        (group_commitment(keys),),
    )


def _flatten(batch: PreparedActivityBatch, chunk_size: int) -> tuple[np.ndarray, ...]:
    chunks = list(
        iter_unordered_pair_chunks(
            batch,
            energy_floors=np.zeros((6,), dtype=np.float64),
            edge_chunk_size=chunk_size,
        )
    )
    assert all(chunk.batch_indices.shape[0] <= chunk_size for chunk in chunks)
    return tuple(
        np.concatenate([getattr(chunk, field) for chunk in chunks], axis=0)
        for field in (
            "batch_indices",
            "actor_i",
            "actor_j",
            "tokens_ij",
            "tokens_ji",
            "support_mask",
        )
    )


def test_stream_enumerates_every_unordered_pair_once_without_dense_materialization() -> None:
    batch = _batch(4)
    flattened = _flatten(batch, 64)
    _, left, right, *_ = flattened
    assert unordered_pair_count(batch) == 6
    assert list(zip(left.tolist(), right.tolist(), strict=True)) == [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 2),
        (1, 3),
        (2, 3),
    ]


def test_stream_is_chunk_size_invariant_and_obeys_directional_swap_laws() -> None:
    batch = _batch(4)
    one = _flatten(batch, 64)
    four = _flatten(batch, 128)
    for left, right in zip(one, four, strict=True):
        assert np.array_equal(left, right)

    tokens_ij = one[3]
    tokens_ji = one[4]
    assert np.array_equal(tokens_ij[..., 0], tokens_ji[..., 1])
    assert np.array_equal(tokens_ij[..., 1], tokens_ji[..., 0])
    assert np.array_equal(tokens_ij[..., 2:4], tokens_ji[..., 2:4])
    assert np.array_equal(tokens_ij[..., 4], -tokens_ji[..., 4])
    assert np.array_equal(tokens_ij[..., 5], -tokens_ji[..., 5])
    assert np.array_equal(tokens_ij[..., 6:], tokens_ji[..., 6:])


def test_known_delay_changes_only_edges_incident_to_the_delayed_actor() -> None:
    actor_count = 4
    valid_length = 200
    padded_time = valid_length + 1
    grid = np.arange(valid_length, dtype=np.float64)
    activities = np.zeros((1, actor_count, padded_time, 5), dtype=np.float32)
    for actor in range(actor_count):
        for channel in range(5):
            activities[0, actor, :valid_length, channel] = np.sin(
                2.0
                * np.pi
                * (1.6875 + 0.03125 * channel)
                * grid
                / 20.0
                + 0.23 * actor
                + 0.07 * channel
            ).astype(np.float32)
    actor_mask = np.ones((1, actor_count), dtype=np.bool_)
    frame_mask = np.zeros((1, padded_time), dtype=np.bool_)
    frame_mask[:, :valid_length] = True
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    activity_mask[:, :, :valid_length] = True
    keys = tuple(_key(index) for index in range(actor_count))

    original = PreparedActivityBatch(
        np.ascontiguousarray(activities),
        actor_mask,
        frame_mask,
        activity_mask,
        (keys,),
        (group_commitment(keys),),
    )
    delayed_values = np.array(activities, copy=True, order="C")
    delayed_values[0, 1, :valid_length] = np.roll(
        delayed_values[0, 1, :valid_length],
        shift=3,
        axis=0,
    )
    delayed = PreparedActivityBatch(
        delayed_values,
        actor_mask,
        frame_mask,
        activity_mask,
        (keys,),
        (group_commitment(keys),),
    )

    before = _flatten(original, 64)
    after = _flatten(delayed, 64)
    assert np.array_equal(before[1], after[1])
    assert np.array_equal(before[2], after[2])
    delayed_canonical_actor = sorted(keys).index(keys[1])
    for edge_index, (left, right) in enumerate(
        zip(before[1].tolist(), before[2].tolist(), strict=True)
    ):
        incident = delayed_canonical_actor in (left, right)
        if incident:
            assert not np.array_equal(before[3][edge_index], after[3][edge_index])
            assert not np.array_equal(before[4][edge_index], after[4][edge_index])
        else:
            assert np.array_equal(before[3][edge_index], after[3][edge_index])
            assert np.array_equal(before[4][edge_index], after[4][edge_index])
            assert np.array_equal(before[5][edge_index], after[5][edge_index])


def test_default_runtime_chunk_is_exactly_explicit_256() -> None:
    assert DEFAULT_EDGE_CHUNK_SIZE == 256
    batch = _batch(24)
    default_chunks = list(
        iter_unordered_pair_chunks(
            batch,
            energy_floors=np.zeros((6,), dtype=np.float64),
        )
    )
    explicit_chunks = list(
        iter_unordered_pair_chunks(
            batch,
            energy_floors=np.zeros((6,), dtype=np.float64),
            edge_chunk_size=256,
        )
    )
    assert [chunk.batch_indices.shape[0] for chunk in default_chunks] == [256, 20]
    for default, explicit in zip(default_chunks, explicit_chunks, strict=True):
        for field in (
            "batch_indices",
            "actor_i",
            "actor_j",
            "tokens_ij",
            "tokens_ji",
            "support_mask",
        ):
            assert np.array_equal(getattr(default, field), getattr(explicit, field))


def test_two_actor_wrapper_uses_phaseset_20hz_support_and_legacy_13d_layout() -> None:
    batch = _batch(2)
    chunk = next(
        iter_unordered_pair_chunks(
            batch,
            energy_floors=np.zeros((6,), dtype=np.float64),
            edge_chunk_size=64,
        )
    )
    assert chunk.tokens_ij.shape == (1, 6, 13)
    assert np.array_equal(
        chunk.support_mask[0],
        np.array([False, False, False, False, True, True], dtype=np.bool_),
    )


@pytest.mark.parametrize("chunk_size", (1, 63, 65, 127, True))
def test_runtime_chunk_must_be_an_exact_positive_multiple_of_64(
    chunk_size: object,
) -> None:
    with pytest.raises((TypeError, PeriodicContractError), match="64-edge|exact"):
        list(
            iter_unordered_pair_chunks(
                _batch(2),
                energy_floors=np.zeros((6,), dtype=np.float64),
                edge_chunk_size=chunk_size,  # type: ignore[arg-type]
            )
        )


def test_each_actor_band_morlet_response_is_computed_once(monkeypatch: pytest.MonkeyPatch) -> None:
    original = legacy_signal._sliding_complex_dot
    calls = 0

    def counted(activity: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        nonlocal calls
        calls += 1
        return original(activity, kernel)

    monkeypatch.setattr(legacy_signal, "_sliding_complex_dot", counted)
    batch = _batch(4)
    _flatten(batch, 64)
    # At 20 Hz, N=16 enables the 16- and 11-tap registered bands.
    assert calls == 8


def test_explicit_edge_budget_fails_closed_without_sampling() -> None:
    batch = _batch(4)
    with pytest.raises(ResourceLimitError, match="RESOURCE_LIMIT") as raised:
        list(
            iter_unordered_pair_chunks(
                batch,
                energy_floors=np.zeros((6,), dtype=np.float64),
                edge_chunk_size=64,
                edge_budget=5,
            )
        )
    assert raised.value.required_edges == 6
    assert raised.value.edge_budget == 5


def test_activity_missing_mask_propagates_into_pair_support() -> None:
    batch = _batch(2)
    activities = np.array(batch.activities, copy=True)
    activity_mask = np.array(batch.activity_mask, copy=True)
    activities[0, 0, :16] = np.float32(0.0)
    activity_mask[0, 0, :16] = False
    masked = PreparedActivityBatch(
        activities,
        np.array(batch.actor_mask, copy=True),
        np.array(batch.frame_mask, copy=True),
        activity_mask,
        batch.actor_commitments,
        batch.group_commitments,
    )
    chunk = next(
        iter_unordered_pair_chunks(
            masked,
            energy_floors=np.full((6,), 1e-12, dtype=np.float64),
            edge_chunk_size=64,
        )
    )
    # Missing activity invalidates both 20-Hz response bands instead of treating
    # the exact-zero payload as observed motion.
    assert not chunk.support_mask.any()


@pytest.mark.parametrize("observed", (False, True))
def test_zero_energy_is_invalid_even_when_default_floor_is_zero(observed: bool) -> None:
    padded_time = 81
    valid_length = 80
    activities = np.zeros((1, 2, padded_time, 5), dtype=np.float32)
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    if observed:
        activity_mask[:, :, :valid_length] = True
    actor_mask = np.ones((1, 2), dtype=np.bool_)
    frame_mask = np.zeros((1, padded_time), dtype=np.bool_)
    frame_mask[:, :valid_length] = True
    keys = (_key(0), _key(1))
    batch = PreparedActivityBatch(
        activities,
        actor_mask,
        frame_mask,
        activity_mask,
        (keys,),
        (group_commitment(keys),),
    )
    chunk = next(
        iter_unordered_pair_chunks(
            batch,
            energy_floors=np.zeros((6,), dtype=np.float64),
            edge_chunk_size=64,
        )
    )
    assert not chunk.support_mask.any()


def test_positive_energy_zero_coherence_edge_is_not_deleted() -> None:
    padded_time = 201
    valid_length = 200
    frequency = 1.6875
    time = np.arange(valid_length, dtype=np.float64)
    tone = np.sin(2.0 * np.pi * frequency * time / 20.0).astype(np.float32)
    activities = np.zeros((1, 2, padded_time, 5), dtype=np.float32)
    activities[0, 0, :valid_length, 0] = tone
    activities[0, 1, :valid_length, 1] = tone
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    activity_mask[:, :, :valid_length] = True
    actor_mask = np.ones((1, 2), dtype=np.bool_)
    frame_mask = np.zeros((1, padded_time), dtype=np.bool_)
    frame_mask[:, :valid_length] = True
    keys = (_key(0), _key(1))
    batch = PreparedActivityBatch(
        activities,
        actor_mask,
        frame_mask,
        activity_mask,
        (keys,),
        (group_commitment(keys),),
    )
    chunk = next(
        iter_unordered_pair_chunks(
            batch,
            energy_floors=np.zeros((6,), dtype=np.float64),
            edge_chunk_size=64,
        )
    )
    assert bool(chunk.support_mask[0, 2])
    assert chunk.tokens_ij[0, 2, 2] == 0.0
