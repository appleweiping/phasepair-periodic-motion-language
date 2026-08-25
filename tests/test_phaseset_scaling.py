"""High-coverage invariance and streaming-scaling tests for PhaseSet.

The K<=27 cases exercise the real activity-to-token path. The K=8..256 stress
tests exercise the complete neural edge stream while replacing only the costly
Morlet convolution with shape-correct cached responses. K=256 therefore passes
through the real half-edge, pair, incident-moment, topology, and postprocess
modules without constructing a dense actor-pair tensor.
"""

from __future__ import annotations

import hashlib
from itertools import combinations, permutations

import numpy as np
import pytest
import torch

from phaseset_core import morlet, periodic
from phaseset_core.contracts import PreparedActivityBatch, group_commitment
from phaseset_core.models import PhaseSetEncoder
from phaseset_core.periodic import ResourceLimitError, iter_unordered_pair_chunks


_TIME = 12
_VALID_TIME = 11
_OUTPUT_FIELDS = (
    "tokens",
    "band_mask",
    "pair_component",
    "topology_delta",
    "valid_pair_count",
    "topology_node_count",
)


def _actor_key(domain: str, actor: int) -> bytes:
    return hashlib.sha256(f"scaling:{domain}:{actor}".encode("ascii")).digest()


def _physical_activity(domain: str, actor: int) -> np.ndarray:
    domain_phase = (sum(domain.encode("ascii")) % 31) / 100.0
    grid = np.arange(_VALID_TIME, dtype=np.float32)
    result = np.zeros((_TIME, 5), dtype=np.float32)
    for channel in range(5):
        result[:_VALID_TIME, channel] = np.sin(
            np.float32(0.19 + 0.013 * actor) * grid
            + np.float32(domain_phase + 0.11 * channel + 0.07 * actor)
        )
    return result


def _single_activity_batch(
    actor_count: int,
    *,
    domain: str = "single",
    padded_actors: int | None = None,
    positions: tuple[int, ...] | None = None,
    physical_order: tuple[int, ...] | None = None,
) -> PreparedActivityBatch:
    padded = actor_count if padded_actors is None else padded_actors
    positions = tuple(range(actor_count)) if positions is None else positions
    physical_order = tuple(range(actor_count)) if physical_order is None else physical_order
    assert len(positions) == actor_count == len(physical_order)
    activities = np.zeros((1, padded, _TIME, 5), dtype=np.float32)
    actor_mask = np.zeros((1, padded), dtype=np.bool_)
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    commitments: list[bytes | None] = [None] * padded
    keys = tuple(_actor_key(domain, actor) for actor in range(actor_count))
    for position, physical_actor in zip(positions, physical_order, strict=True):
        activities[0, position] = _physical_activity(domain, physical_actor)
        actor_mask[0, position] = True
        activity_mask[0, position, :_VALID_TIME] = True
        commitments[position] = keys[physical_actor]
    frame_mask = np.zeros((1, _TIME), dtype=np.bool_)
    frame_mask[:, :_VALID_TIME] = True
    return PreparedActivityBatch(
        activities,
        actor_mask,
        frame_mask,
        activity_mask,
        (tuple(commitments),),
        (group_commitment(keys),),
    )


def _mixed_activity_batch(
    actor_counts: tuple[int, ...],
    *,
    padded_actors: int,
) -> PreparedActivityBatch:
    batch_size = len(actor_counts)
    activities = np.zeros(
        (batch_size, padded_actors, _TIME, 5),
        dtype=np.float32,
    )
    actor_mask = np.zeros((batch_size, padded_actors), dtype=np.bool_)
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    frame_mask = np.zeros((batch_size, _TIME), dtype=np.bool_)
    frame_mask[:, :_VALID_TIME] = True
    commitment_rows: list[tuple[bytes | None, ...]] = []
    group_rows: list[bytes] = []
    for row, actor_count in enumerate(actor_counts):
        domain = f"mixed-{row}"
        keys = tuple(_actor_key(domain, actor) for actor in range(actor_count))
        # Deliberately scatter valid actors and reverse their physical order.
        positions = tuple(range(padded_actors - 1, padded_actors - actor_count - 1, -1))
        physical_order = tuple(reversed(range(actor_count)))
        commitments: list[bytes | None] = [None] * padded_actors
        for position, physical_actor in zip(positions, physical_order, strict=True):
            activities[row, position] = _physical_activity(domain, physical_actor)
            actor_mask[row, position] = True
            activity_mask[row, position, :_VALID_TIME] = True
            commitments[position] = keys[physical_actor]
        commitment_rows.append(tuple(commitments))
        group_rows.append(group_commitment(keys))
    return PreparedActivityBatch(
        activities,
        actor_mask,
        frame_mask,
        activity_mask,
        tuple(commitment_rows),
        tuple(group_rows),
    )


def _small_model() -> PhaseSetEncoder:
    torch.manual_seed(20260825)
    return PhaseSetEncoder(embedding_dim=4, hidden_dim=4).eval()


def _assert_output_equal(expected: object, actual: object) -> None:
    for field in _OUTPUT_FIELDS:
        assert torch.equal(getattr(expected, field), getattr(actual, field)), field


def test_every_actor_permutation_k3_through_k6_is_bitwise_invariant() -> None:
    model = _small_model()
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with torch.no_grad():
            for actor_count in range(3, 7):
                baseline = model.forward_activity(
                    _single_activity_batch(actor_count),
                    edge_chunk_size=64,
                )
                expected_pairs = actor_count * (actor_count - 1) // 2
                assert int(baseline.valid_pair_count.max().item()) == expected_pairs
                assert int(baseline.topology_node_count.max().item()) == actor_count
                for physical_order in permutations(range(actor_count)):
                    actual = model.forward_activity(
                        _single_activity_batch(
                            actor_count,
                            physical_order=physical_order,
                        ),
                        edge_chunk_size=64,
                    )
                    _assert_output_equal(baseline, actual)
    finally:
        torch.set_num_threads(previous_threads)


def test_kpad_k_kplus5_and_2k_are_bitwise_identical() -> None:
    actor_count = 4
    model = _small_model()
    cases = (
        _single_activity_batch(actor_count),
        _single_activity_batch(
            actor_count,
            padded_actors=actor_count + 5,
            positions=(8, 1, 6, 3),
            physical_order=(2, 0, 3, 1),
        ),
        _single_activity_batch(
            actor_count,
            padded_actors=2 * actor_count,
            positions=(7, 0, 5, 2),
            physical_order=(3, 1, 0, 2),
        ),
    )
    with torch.no_grad():
        baseline = model.forward_activity(cases[0], edge_chunk_size=64)
        for batch in cases[1:]:
            _assert_output_equal(
                baseline,
                model.forward_activity(batch, edge_chunk_size=64),
            )


def test_dynamic_mixed_batch_matches_each_isolated_sample_bitwise() -> None:
    actor_counts = (4, 12, 5)
    model = _small_model()
    mixed_batch = _mixed_activity_batch(actor_counts, padded_actors=17)
    with torch.no_grad():
        mixed = model.forward_activity(mixed_batch, edge_chunk_size=64)
        for row, actor_count in enumerate(actor_counts):
            isolated = model.forward_activity(
                _single_activity_batch(actor_count, domain=f"mixed-{row}"),
                edge_chunk_size=64,
            )
            for field in _OUTPUT_FIELDS:
                assert torch.equal(
                    getattr(isolated, field),
                    getattr(mixed, field)[row : row + 1],
                ), (row, field)


def test_chunks_64_128_256_320_and_full_equivalent_are_bitwise_identical() -> None:
    # K=27 has 351 edges, so 384 is the smallest valid chunk that is
    # equivalent to a single all-edge runtime chunk.
    batch = _single_activity_batch(27)
    model = _small_model()
    with torch.no_grad():
        baseline = model.forward_activity(batch, edge_chunk_size=64)
        for chunk_size in (128, 256, 320, 384):
            _assert_output_equal(
                baseline,
                model.forward_activity(batch, edge_chunk_size=chunk_size),
            )


def _install_synthetic_morlet_cache(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    """Replace convolution only; relation construction and edge streaming stay real."""

    bands = morlet.morlet_kernel_bank()
    length_mask = np.zeros((6,), dtype=np.bool_)
    length_mask[-1] = True
    calls = [0]

    def cached_responses(
        activities: np.ndarray,
        activity_mask: np.ndarray,
        valid_length: int,
    ) -> tuple[tuple[periodic._ActorResponse, ...], tuple[object, ...], np.ndarray]:
        del activity_mask
        assert valid_length == _VALID_TIME
        calls[0] += 1
        actors: list[periodic._ActorResponse] = []
        for actor in range(activities.shape[0]):
            real = float(activities[actor, 0, 0]) + 2.0
            imaginary = float(activities[actor, 0, 1]) + 0.5
            transformed = np.full((1, 5), complex(real, imaginary), dtype=np.complex128)
            response_mask = np.ones((1, 5), dtype=np.bool_)
            actors.append(
                periodic._ActorResponse(
                    (None, None, None, None, None, transformed),
                    (None, None, None, None, None, response_mask),
                )
            )
        return tuple(actors), bands, length_mask

    monkeypatch.setattr(periodic, "_precompute_actor_responses", cached_responses)
    return calls


def _track_numpy_shapes(
    monkeypatch: pytest.MonkeyPatch,
) -> set[tuple[int, ...]]:
    """Record explicit NumPy allocation/result shapes used by the iterator."""

    observed: set[tuple[int, ...]] = set()
    for name in (
        "array",
        "asarray",
        "ascontiguousarray",
        "concatenate",
        "empty",
        "frombuffer",
        "full",
        "ones",
        "stack",
        "zeros",
    ):
        original = getattr(periodic.np, name)

        def recorder(*args: object, _original: object = original, **kwargs: object) -> object:
            result = _original(*args, **kwargs)  # type: ignore[operator]
            if type(result) is np.ndarray:
                observed.add(tuple(int(value) for value in result.shape))
            return result

        monkeypatch.setattr(periodic.np, name, recorder)
    return observed


@pytest.mark.parametrize("actor_count", (8, 32, 128, 256))
def test_large_k_stream_is_complete_budgeted_and_never_allocates_k_by_k(
    actor_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """This is an iterator/allocation stress test, not a neural-forward claim."""

    batch = _single_activity_batch(actor_count, domain=f"large-{actor_count}")
    expected_edges = actor_count * (actor_count - 1) // 2
    cache_calls = _install_synthetic_morlet_cache(monkeypatch)

    with pytest.raises(ResourceLimitError, match="RESOURCE_LIMIT") as raised:
        list(
            iter_unordered_pair_chunks(
                batch,
                energy_floors=np.zeros((6,), dtype=np.float64),
                edge_chunk_size=64,
                edge_budget=expected_edges - 1,
            )
        )
    assert raised.value.required_edges == expected_edges
    assert raised.value.edge_budget == expected_edges - 1
    assert cache_calls == [0]

    observed_shapes = _track_numpy_shapes(monkeypatch)
    edge_count = 0
    previous_pair: tuple[int, int] | None = None
    maximum_resident_edges = 0
    for chunk in iter_unordered_pair_chunks(
        batch,
        energy_floors=np.zeros((6,), dtype=np.float64),
        edge_chunk_size=64,
        edge_budget=expected_edges,
    ):
        maximum_resident_edges = max(maximum_resident_edges, len(chunk.actor_i))
        for left, right in zip(chunk.actor_i, chunk.actor_j, strict=True):
            pair = (int(left), int(right))
            assert pair[0] < pair[1]
            if previous_pair is not None:
                assert previous_pair < pair
            previous_pair = pair
            edge_count += 1
    assert cache_calls == [1]
    assert edge_count == expected_edges
    assert previous_pair == (actor_count - 2, actor_count - 1)
    assert maximum_resident_edges <= 64
    assert not any(
        len(shape) >= 2 and shape[0] == actor_count and shape[1] == actor_count
        for shape in observed_shapes
    ), observed_shapes


@pytest.mark.parametrize("actor_count", (32, 128, 256))
def test_large_k_complete_neural_forward_is_permutation_invariant(
    actor_count: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cache_calls = _install_synthetic_morlet_cache(monkeypatch)
    model = _small_model()
    expected_edges = actor_count * (actor_count - 1) // 2
    original = _single_activity_batch(actor_count, domain=f"neural-{actor_count}")
    permuted = _single_activity_batch(
        actor_count,
        domain=f"neural-{actor_count}",
        physical_order=tuple(reversed(range(actor_count))),
    )
    with torch.no_grad():
        baseline = model.forward_activity(
            original,
            edge_chunk_size=256,
        )
        reordered = model.forward_activity(
            permuted,
            edge_chunk_size=256,
        )
    _assert_output_equal(baseline, reordered)
    assert int(baseline.valid_pair_count.max().item()) == expected_edges
    assert cache_calls == [2]


def test_torch_descriptor_seam_gradient_is_actor_permutation_equivariant() -> None:
    """Check differentiable descriptor MLPs, not the NumPy skeleton/Morlet boundary."""

    actor_count = 5
    edges = tuple(combinations(range(actor_count), 2))
    edge_lookup = {edge: index for index, edge in enumerate(edges)}
    torch.manual_seed(991)
    model = _small_model()
    relation_ij = torch.randn((len(edges), 6, 13), requires_grad=True)
    relation_ji = torch.randn((len(edges), 6, 13), requires_grad=True)
    upstream = torch.randn((len(edges), 6, model.embedding_dim))
    pair_tokens = model._edge_forward(relation_ij, relation_ji)[2]
    original_grad_ij, original_grad_ji = torch.autograd.grad(
        pair_tokens,
        (relation_ij, relation_ji),
        grad_outputs=upstream,
    )

    # New actor position -> original physical actor.  Some canonical endpoints
    # reverse, so equivariance also covers endpoint-swap gradients.
    actor_permutation = (3, 0, 4, 1, 2)
    permuted_left: list[torch.Tensor] = []
    permuted_right: list[torch.Tensor] = []
    permuted_upstream: list[torch.Tensor] = []
    mappings: list[tuple[int, bool]] = []
    for new_left, new_right in edges:
        physical_left = actor_permutation[new_left]
        physical_right = actor_permutation[new_right]
        physical_edge = tuple(sorted((physical_left, physical_right)))
        physical_index = edge_lookup[physical_edge]
        same_orientation = physical_left < physical_right
        mappings.append((physical_index, same_orientation))
        if same_orientation:
            permuted_left.append(relation_ij.detach()[physical_index])
            permuted_right.append(relation_ji.detach()[physical_index])
        else:
            permuted_left.append(relation_ji.detach()[physical_index])
            permuted_right.append(relation_ij.detach()[physical_index])
        permuted_upstream.append(upstream[physical_index])
    permuted_ij = torch.stack(permuted_left).requires_grad_()
    permuted_ji = torch.stack(permuted_right).requires_grad_()
    permuted_tokens = model._edge_forward(permuted_ij, permuted_ji)[2]
    permuted_grad_ij, permuted_grad_ji = torch.autograd.grad(
        permuted_tokens,
        (permuted_ij, permuted_ji),
        grad_outputs=torch.stack(permuted_upstream),
    )
    for new_index, (physical_index, same_orientation) in enumerate(mappings):
        expected_left = (
            original_grad_ij[physical_index]
            if same_orientation
            else original_grad_ji[physical_index]
        )
        expected_right = (
            original_grad_ji[physical_index]
            if same_orientation
            else original_grad_ij[physical_index]
        )
        assert torch.equal(permuted_grad_ij[new_index], expected_left)
        assert torch.equal(permuted_grad_ji[new_index], expected_right)
