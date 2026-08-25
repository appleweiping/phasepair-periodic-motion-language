from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest
import torch

from phaseset_core import streaming_autograd as streaming_module
from phaseset_core.contracts import PreparedActivityBatch, group_commitment
from phaseset_core.controls import (
    RESIDUAL_SYSTEM_IDS,
    SYSTEM_SPECS,
    PhaseSetControlError,
    audit_control_parameter_counts,
    build_phaseset_system,
    gate_relation_descriptors,
    iter_mean_difference_dct_chunks,
    shuffle_half_edge_incidence,
    system_spec,
    trainable_parameter_count,
)
from phaseset_core.periodic import (
    BAND_COUNT,
    TOKEN_WIDTH,
    iter_marginal_power_pair_chunks,
    iter_observation_availability_pair_chunks,
)


def _key(index: int) -> bytes:
    return hashlib.sha256(f"phaseset-control-actor-{index}".encode()).digest()


def _activity_batch(
    actor_count: int,
    *,
    padded_actors: int | None = None,
    positions: tuple[int, ...] | None = None,
    physical_order: tuple[int, ...] | None = None,
    phase_offset: float = 0.0,
    amplitude_scale: float = 1.0,
) -> PreparedActivityBatch:
    padded = actor_count if padded_actors is None else padded_actors
    target_positions = tuple(range(actor_count)) if positions is None else positions
    order = tuple(range(actor_count)) if physical_order is None else physical_order
    assert len(target_positions) == actor_count
    assert sorted(order) == list(range(actor_count))
    time_steps = 100
    time = np.arange(time_steps, dtype=np.float64) / 20.0
    physical = np.zeros((actor_count, time_steps, 5), dtype=np.float32)
    for actor in range(actor_count):
        for channel in range(5):
            frequency = (0.75, 1.125, 1.6875)[(actor + channel) % 3]
            phase = 0.31 * actor + 0.17 * channel + phase_offset * actor
            physical[actor, :, channel] = np.asarray(
                amplitude_scale
                * (1.0 + 0.13 * actor)
                * np.sin(2.0 * math.pi * frequency * time + phase)
                + 0.19 * np.cos(2.0 * math.pi * 2.53125 * time + 0.11 * channel),
                dtype=np.float32,
            )
    activities = np.zeros((1, padded, time_steps, 5), dtype=np.float32)
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    actor_mask = np.zeros((1, padded), dtype=np.bool_)
    commitments: list[bytes | None] = [None] * padded
    keys = tuple(_key(index) for index in range(actor_count))
    for destination, physical_actor in zip(target_positions, order, strict=True):
        activities[0, destination] = physical[physical_actor]
        activity_mask[0, destination] = True
        actor_mask[0, destination] = True
        commitments[destination] = keys[physical_actor]
    return PreparedActivityBatch(
        np.ascontiguousarray(activities),
        np.ascontiguousarray(actor_mask),
        np.ones((1, time_steps), dtype=np.bool_),
        np.ascontiguousarray(activity_mask),
        (tuple(commitments),),
        (group_commitment(keys),),
    )


def _small_system(system_id: str, *, incidence_seed: int = 1729):
    torch.manual_seed(20260825)
    return build_phaseset_system(
        system_id,
        embedding_dim=8,
        hidden_dim=12,
        incidence_seed=incidence_seed,
    )


def _stack_activity_batches(
    batches: tuple[PreparedActivityBatch, ...],
) -> PreparedActivityBatch:
    assert batches
    shape = batches[0].activities.shape[1:]
    assert all(batch.activities.shape[1:] == shape for batch in batches)
    return PreparedActivityBatch(
        np.ascontiguousarray(
            np.concatenate([batch.activities for batch in batches], axis=0)
        ),
        np.ascontiguousarray(
            np.concatenate([batch.actor_mask for batch in batches], axis=0)
        ),
        np.ascontiguousarray(
            np.concatenate([batch.frame_mask for batch in batches], axis=0)
        ),
        np.ascontiguousarray(
            np.concatenate([batch.activity_mask for batch in batches], axis=0)
        ),
        tuple(batch.actor_commitments[0] for batch in batches),
        tuple(batch.group_commitments[0] for batch in batches),
    )


def _stationary_activity_batch(actor_count: int) -> PreparedActivityBatch:
    batch = _activity_batch(actor_count)
    return PreparedActivityBatch(
        np.zeros_like(batch.activities),
        batch.actor_mask.copy(),
        batch.frame_mask.copy(),
        batch.activity_mask.copy(),
        batch.actor_commitments,
        batch.group_commitments,
    )


def _replace_canonical_actor(
    batch: PreparedActivityBatch,
    actor: int,
    *,
    values: np.ndarray,
    observed: np.ndarray,
) -> PreparedActivityBatch:
    activities = batch.activities.copy()
    activity_mask = batch.activity_mask.copy()
    assert values.shape == activities[0, actor].shape
    assert observed.shape == activity_mask[0, actor].shape
    activities[0, actor] = values
    activity_mask[0, actor] = observed
    activities[0, actor][~observed] = np.float32(0.0)
    return PreparedActivityBatch(
        np.ascontiguousarray(activities),
        batch.actor_mask.copy(),
        batch.frame_mask.copy(),
        np.ascontiguousarray(activity_mask),
        batch.actor_commitments,
        batch.group_commitments,
    )


def _signed_pair_activity_batch(sign: int) -> PreparedActivityBatch:
    assert sign in (-1, 1)
    time_steps = 100
    time = np.arange(time_steps, dtype=np.float64) / 20.0
    signal = np.asarray(
        np.sin(2.0 * math.pi * 1.125 * time)
        + 0.23 * np.cos(2.0 * math.pi * 2.53125 * time),
        dtype=np.float32,
    )
    activities = np.zeros((1, 2, time_steps, 5), dtype=np.float32)
    activities[0, 0] = signal[:, None]
    activities[0, 1] = np.float32(sign) * signal[:, None]
    keys = (_key(0), _key(1))
    return PreparedActivityBatch(
        np.ascontiguousarray(activities),
        np.ones((1, 2), dtype=np.bool_),
        np.ones((1, time_steps), dtype=np.bool_),
        np.ones_like(activities, dtype=np.bool_),
        (keys,),
        (group_commitment(keys),),
    )


def _unilateral_pair_activity_batch() -> PreparedActivityBatch:
    batch = _signed_pair_activity_batch(1)
    activities = batch.activities.copy()
    activities[0, 1] = np.float32(0.0)
    return PreparedActivityBatch(
        np.ascontiguousarray(activities),
        batch.actor_mask.copy(),
        batch.frame_mask.copy(),
        batch.activity_mask.copy(),
        batch.actor_commitments,
        batch.group_commitments,
    )


def test_frozen_system_census_factory_and_real_parameter_audit() -> None:
    assert tuple(spec.system_id for spec in SYSTEM_SPECS) == tuple(
        f"{index:02d}" for index in range(9)
    )
    assert system_spec("00").residual_enabled is False
    assert system_spec("04").include_topology is False
    assert system_spec("08").include_topology is True
    with pytest.raises(PhaseSetControlError, match="00--08"):
        system_spec("09")

    systems = {system_id: _small_system(system_id) for system_id in RESIDUAL_SYSTEM_IDS}
    rows = audit_control_parameter_counts(systems)
    assert tuple(system_id for system_id, _ in rows) == RESIDUAL_SYSTEM_IDS
    assert len({count for _, count in rows}) == 1
    assert rows[0][1] == trainable_parameter_count(systems["08"])
    assert trainable_parameter_count(_small_system("00")) == 0


def test_system02_uses_the_same_capacity_head_as_full() -> None:
    marginal = _small_system("02")
    full = _small_system("08")
    assert marginal.encoder is not None
    assert full.encoder is not None
    assert tuple(marginal.state_dict()) == tuple(full.state_dict())
    assert tuple(
        (name, type(module)) for name, module in marginal.encoder.named_modules()
    ) == tuple((name, type(module)) for name, module in full.encoder.named_modules())
    assert trainable_parameter_count(marginal) == trainable_parameter_count(full)
    full.load_state_dict(marginal.state_dict(), strict=True)


@pytest.mark.parametrize("system_id", ("02", "05"))
def test_marginal_and_coverage_controls_do_not_inherit_energy_floor_support(
    system_id: str,
) -> None:
    reference = _small_system(system_id).eval()
    changed_floor = build_phaseset_system(
        system_id,
        embedding_dim=8,
        hidden_dim=12,
        energy_floors=np.full((BAND_COUNT,), 1.0e100, dtype=np.float64),
    ).eval()
    changed_floor.load_state_dict(reference.state_dict(), strict=True)
    with torch.no_grad():
        expected = reference.forward_activity(_activity_batch(3))
        actual = changed_floor.forward_activity(_activity_batch(3))
    for field in (
        "tokens",
        "band_mask",
        "pair_component",
        "topology_delta",
        "valid_pair_count",
        "topology_node_count",
    ):
        assert torch.equal(getattr(actual, field), getattr(expected, field))


def test_descriptor_information_gates_are_exact_and_phase_stripped_retains_coherence() -> None:
    tokens = torch.arange(2 * BAND_COUNT * TOKEN_WIDTH, dtype=torch.float32).reshape(
        2, BAND_COUNT, TOKEN_WIDTH
    )
    generic = gate_relation_descriptors(tokens, "BAND_ID_ONLY")
    marginal = gate_relation_descriptors(tokens, "MARGINAL_POWER")
    stripped = gate_relation_descriptors(tokens, "PHASE_STRIPPED")
    assert torch.equal(generic[..., :7], torch.zeros_like(generic[..., :7]))
    assert torch.equal(generic[..., 7:], tokens[..., 7:])
    assert torch.equal(marginal[..., :2], tokens[..., :2])
    assert torch.equal(marginal[..., 2:7], torch.zeros_like(marginal[..., 2:7]))
    assert torch.equal(stripped[..., :3], tokens[..., :3])
    assert torch.equal(stripped[..., 3:7], torch.zeros_like(stripped[..., 3:7]))
    assert torch.equal(stripped[..., 7:], tokens[..., 7:])


def test_incidence_shuffle_is_seeded_deterministic_and_preserves_valid_multisets() -> None:
    half_ij = torch.arange(4 * BAND_COUNT * 3, dtype=torch.float32).reshape(4, BAND_COUNT, 3)
    half_ji = half_ij + 10_000.0
    support = torch.tensor(
        [
            [True, True, False, True, True, False],
            [True, False, True, True, False, True],
            [True, True, True, False, True, True],
            [False, True, True, True, True, True],
        ],
        dtype=torch.bool,
    )
    metadata = {
        "batch_indices": torch.zeros((4,), dtype=torch.int64),
        "actor_i": torch.tensor((0, 0, 0, 1), dtype=torch.int64),
        "actor_j": torch.tensor((1, 2, 3, 2), dtype=torch.int64),
    }
    first = shuffle_half_edge_incidence(
        half_ij, half_ji, support, seed=2718, **metadata
    )
    repeated = shuffle_half_edge_incidence(
        half_ij, half_ji, support, seed=2718, **metadata
    )
    changed_seed = shuffle_half_edge_incidence(
        half_ij, half_ji, support, seed=31415, **metadata
    )
    assert all(torch.equal(left, right) for left, right in zip(first, repeated, strict=True))
    assert any(
        not torch.equal(left, right) for left, right in zip(first, changed_seed, strict=True)
    )
    original = torch.stack((half_ij, half_ji), dim=1)
    shuffled = torch.stack(first, dim=1)
    for band in range(BAND_COUNT):
        valid = support[:, band].repeat_interleave(2)
        original_rows = original[:, :, band].reshape(-1, 3)[valid]
        shuffled_rows = shuffled[:, :, band].reshape(-1, 3)[valid]
        assert torch.equal(
            torch.sort(original_rows[:, 0]).values,
            torch.sort(shuffled_rows[:, 0]).values,
        )
    assert torch.equal(support, support.clone())


def test_system06_is_group_local_batch_isolated_rechunked_and_permutation_invariant() -> None:
    """Regression: no A half-edge may be shuffled with any co-batched B row."""

    canonical_a = _activity_batch(
        4,
        padded_actors=7,
        phase_offset=0.19,
        amplitude_scale=0.8,
    )
    permuted_a = _activity_batch(
        4,
        padded_actors=7,
        positions=(6, 1, 4, 2),
        physical_order=(2, 0, 3, 1),
        phase_offset=0.19,
        amplitude_scale=0.8,
    )
    b_rows = tuple(
        _activity_batch(
            7,
            phase_offset=0.41 + 0.03 * row,
            amplitude_scale=2.0 + row,
        )
        for row in range(7)
    )
    mixed = _stack_activity_batches((canonical_a, *b_rows))
    mixed_with_a_last = _stack_activity_batches((*b_rows, canonical_a))
    model = _small_system("06", incidence_seed=31415).eval()
    fields = (
        "tokens",
        "band_mask",
        "pair_component",
        "topology_delta",
        "valid_pair_count",
        "topology_node_count",
    )
    with torch.no_grad():
        baseline = model.forward_activity(canonical_a, edge_chunk_size=64)
        for chunk_size in (64, 128, 256):
            alone = model.forward_activity(canonical_a, edge_chunk_size=chunk_size)
            permuted = model.forward_activity(permuted_a, edge_chunk_size=chunk_size)
            co_batched = model.forward_activity(mixed, edge_chunk_size=chunk_size)
            co_batched_last = model.forward_activity(
                mixed_with_a_last,
                edge_chunk_size=chunk_size,
            )
            for field in fields:
                expected = getattr(baseline, field)
                assert torch.equal(getattr(alone, field), expected), (chunk_size, field)
                assert torch.equal(getattr(permuted, field), expected), (
                    chunk_size,
                    field,
                )
                assert torch.equal(getattr(co_batched, field)[0:1], expected), (
                    chunk_size,
                    field,
                )
                assert torch.equal(getattr(co_batched_last, field)[-1:], expected), (
                    chunk_size,
                    field,
                )


def test_system06_incidence_bijection_is_global_across_canonical_microblocks() -> None:
    batch = _activity_batch(12)
    system = _small_system("06", incidence_seed=31415).eval()
    assert system.encoder is not None
    plan = streaming_module._build_global_incidence_routing_plan(
        system.encoder,
        batch,
        edge_chunk_size=64,
    )
    assert plan is not None
    witnessed_cross_block = False
    for band in range(BAND_COUNT):
        actors = plan.target_actors[0][band]
        length = int(actors.shape[0])
        assert length > 128
        target_ranks = [plan.target_rank(0, band, rank) for rank in range(length)]
        assert sorted(target_ranks) == list(range(length))
        routed_actors = np.asarray(
            [plan.target_actor(0, band, rank) for rank in range(length)],
            dtype=np.int64,
        )
        assert np.array_equal(np.sort(routed_actors), np.sort(actors))
        witnessed_cross_block |= any(
            source_rank // 128 != target_rank // 128
            for source_rank, target_rank in enumerate(target_ranks)
        )
    assert witnessed_cross_block


@pytest.mark.parametrize("system_id", tuple(f"{index:02d}" for index in range(9)))
def test_every_system_has_uniform_shapes_and_residual_systems_backpropagate(
    system_id: str,
) -> None:
    model = _small_system(system_id)
    output = model.forward_activity(_activity_batch(3), edge_chunk_size=64)
    assert output.tokens.shape == (1, BAND_COUNT, 8)
    assert output.band_mask.shape == (1, BAND_COUNT)
    assert output.pair_component.shape == output.tokens.shape
    assert output.topology_delta.shape == output.tokens.shape
    if system_id == "00":
        assert torch.equal(output.tokens, torch.zeros_like(output.tokens))
        assert not output.band_mask.any()
        return
    output.tokens.square().mean().backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert any(gradient is not None for gradient in gradients)
    assert all(
        bool(torch.isfinite(gradient).all().item())
        for gradient in gradients
        if gradient is not None
    )


@pytest.mark.parametrize("system_id", RESIDUAL_SYSTEM_IDS)
def test_all_residual_controls_are_actor_permutation_and_padding_invariant(
    system_id: str,
) -> None:
    model = _small_system(system_id).eval()
    canonical = _activity_batch(3)
    permuted = _activity_batch(
        3,
        padded_actors=8,
        positions=(7, 1, 4),
        physical_order=(2, 0, 1),
    )
    with torch.no_grad():
        expected = model.forward_activity(canonical, edge_chunk_size=64)
        actual = model.forward_activity(permuted, edge_chunk_size=64)
    for field in (
        "tokens",
        "band_mask",
        "pair_component",
        "topology_delta",
        "valid_pair_count",
        "topology_node_count",
    ):
        assert torch.equal(getattr(expected, field), getattr(actual, field))


def test_k2_full_pair_only_identity_and_all_topology_controls_are_exact_zero() -> None:
    pair = _small_system("04").eval()
    full = _small_system("08").eval()
    full.load_state_dict(pair.state_dict())
    batch = _activity_batch(2)
    with torch.no_grad():
        pair_output = pair.forward_activity(batch)
        full_output = full.forward_activity(batch)
    assert torch.equal(pair_output.tokens, full_output.tokens)
    assert torch.equal(full_output.topology_delta, torch.zeros_like(full_output.topology_delta))
    assert not torch.signbit(full_output.topology_delta).any()
    for system_id in ("05", "06", "07", "08"):
        output = _small_system(system_id).forward_activity(batch)
        assert torch.equal(output.topology_delta, torch.zeros_like(output.topology_delta))
        assert not output.topology_node_count.any()


def test_incidence_shuffle_preserves_pair_bag_but_changes_topology_witness() -> None:
    full = _small_system("08").eval()
    shuffled = _small_system("06", incidence_seed=31415).eval()
    shuffled.load_state_dict(full.state_dict())
    with torch.no_grad():
        full_output = full.forward_activity(_activity_batch(4))
        shuffled_output = shuffled.forward_activity(_activity_batch(4))
    assert torch.equal(full_output.pair_component, shuffled_output.pair_component)
    assert torch.equal(full_output.valid_pair_count, shuffled_output.valid_pair_count)
    assert torch.equal(full_output.topology_node_count, shuffled_output.topology_node_count)
    assert not torch.equal(full_output.topology_delta, shuffled_output.topology_delta)
    assert not torch.equal(full_output.tokens, shuffled_output.tokens)


def test_generic_tokens_are_fixed_six_bands_without_motion_or_support_leakage() -> None:
    model = _small_system("01").eval()
    with torch.no_grad():
        moving = model.forward_activity(
            _activity_batch(3, phase_offset=0.7, amplitude_scale=1.8)
        )
        stationary = model.forward_activity(_stationary_activity_batch(3))
    assert moving.band_mask.all()
    assert stationary.band_mask.all()
    assert torch.equal(moving.valid_pair_count, torch.full_like(moving.valid_pair_count, 3))
    assert torch.equal(stationary.valid_pair_count, moving.valid_pair_count)
    assert torch.equal(stationary.pair_component, moving.pair_component)
    assert torch.equal(stationary.tokens, moving.tokens)


def test_system02_actor_marginal_is_partner_mask_and_signal_invariant() -> None:
    baseline_batch = _activity_batch(3)
    partner = 2
    stationary_partner = _replace_canonical_actor(
        baseline_batch,
        partner,
        values=np.zeros_like(baseline_batch.activities[0, partner]),
        observed=np.ones_like(baseline_batch.activity_mask[0, partner]),
    )
    missing_partner = _replace_canonical_actor(
        baseline_batch,
        partner,
        values=np.zeros_like(baseline_batch.activities[0, partner]),
        observed=np.zeros_like(baseline_batch.activity_mask[0, partner]),
    )

    def one_chunk(batch: PreparedActivityBatch):
        chunks = tuple(
            iter_marginal_power_pair_chunks(batch, edge_chunk_size=64)
        )
        assert len(chunks) == 1
        return chunks[0]

    baseline = one_chunk(baseline_batch)
    for changed in (one_chunk(stationary_partner), one_chunk(missing_partner)):
        assert np.array_equal(changed.support_mask, baseline.support_mask)
        assert changed.support_mask.all()
        # Actor 2 is not incident to edge {0,1}; that complete edge is unchanged.
        assert np.array_equal(changed.tokens_ij[0], baseline.tokens_ij[0])
        assert np.array_equal(changed.tokens_ji[0], baseline.tokens_ji[0])
        # On edge {0,2}, actor 0's directed self-marginal remains byte-identical.
        assert np.array_equal(changed.tokens_ij[1, :, 0], baseline.tokens_ij[1, :, 0])
        assert np.array_equal(changed.tokens_ji[1, :, 1], baseline.tokens_ji[1, :, 1])
        assert not np.array_equal(
            changed.tokens_ij[1, :, 1],
            baseline.tokens_ij[1, :, 1],
        )
        assert np.array_equal(
            changed.tokens_ij[..., 2:7],
            np.zeros_like(changed.tokens_ij[..., 2:7]),
        )
        assert np.array_equal(changed.tokens_ij[..., 7:], baseline.tokens_ij[..., 7:])


def test_system02_raw_stream_is_actor_permutation_and_padding_invariant() -> None:
    canonical = tuple(
        iter_marginal_power_pair_chunks(_activity_batch(3), edge_chunk_size=64)
    )
    permuted = tuple(
        iter_marginal_power_pair_chunks(
            _activity_batch(
                3,
                padded_actors=8,
                positions=(7, 1, 4),
                physical_order=(2, 0, 1),
            ),
            edge_chunk_size=64,
        )
    )
    assert len(canonical) == len(permuted) == 1
    for field in (
        "batch_indices",
        "actor_i",
        "actor_j",
        "tokens_ij",
        "tokens_ji",
        "support_mask",
    ):
        assert np.array_equal(getattr(canonical[0], field), getattr(permuted[0], field))


def test_system05_is_bitwise_motion_content_invariant_for_fixed_observation() -> None:
    batches = (
        _activity_batch(3),
        _activity_batch(3, phase_offset=0.73, amplitude_scale=9.0),
        _stationary_activity_batch(3),
    )
    raw = tuple(
        tuple(
            iter_observation_availability_pair_chunks(batch, edge_chunk_size=64)
        )[0]
        for batch in batches
    )
    for changed in raw[1:]:
        assert np.array_equal(changed.tokens_ij, raw[0].tokens_ij)
        assert np.array_equal(changed.tokens_ji, raw[0].tokens_ji)
        assert np.array_equal(changed.support_mask, raw[0].support_mask)
    assert raw[0].support_mask.all()

    model = _small_system("05").eval()
    with torch.no_grad():
        outputs = tuple(model.forward_activity(batch) for batch in batches)
    for changed in outputs[1:]:
        for field in (
            "tokens",
            "band_mask",
            "pair_component",
            "topology_delta",
            "valid_pair_count",
            "topology_node_count",
        ):
            assert torch.equal(getattr(changed, field), getattr(outputs[0], field))
    assert outputs[0].band_mask.all()
    assert torch.equal(
        outputs[0].valid_pair_count,
        torch.full_like(outputs[0].valid_pair_count, 3),
    )
    assert torch.equal(
        outputs[0].pair_component,
        torch.zeros_like(outputs[0].pair_component),
    )
    assert not torch.signbit(outputs[0].pair_component).any()


def test_system05_unobserved_pair_has_clear_band_and_exact_zero_semantics() -> None:
    observed = _activity_batch(2)
    unobserved = _replace_canonical_actor(
        observed,
        1,
        values=np.zeros_like(observed.activities[0, 1]),
        observed=np.zeros_like(observed.activity_mask[0, 1]),
    )
    chunks = tuple(
        iter_observation_availability_pair_chunks(unobserved, edge_chunk_size=64)
    )
    assert len(chunks) == 1
    assert not chunks[0].support_mask.any()
    with torch.no_grad():
        output = _small_system("05").eval().forward_activity(unobserved)
    assert not output.band_mask.any()
    assert not output.valid_pair_count.any()
    assert not output.topology_node_count.any()
    for value in (output.tokens, output.pair_component, output.topology_delta):
        assert torch.equal(value, torch.zeros_like(value))
        assert not torch.signbit(value).any()


def test_dct_stream_is_endpoint_symmetric_and_has_no_cross_endpoint_phase() -> None:
    floors = np.zeros((BAND_COUNT,), dtype=np.float64)
    same_phase = tuple(
        iter_mean_difference_dct_chunks(
            _signed_pair_activity_batch(1),
            energy_floors=floors,
            edge_chunk_size=64,
        )
    )
    opposite_phase = tuple(
        iter_mean_difference_dct_chunks(
            _signed_pair_activity_batch(-1),
            energy_floors=floors,
            edge_chunk_size=64,
        )
    )
    assert len(same_phase) == len(opposite_phase) == 1
    assert np.array_equal(same_phase[0].tokens_ij, same_phase[0].tokens_ji)
    assert np.array_equal(same_phase[0].tokens_ij, opposite_phase[0].tokens_ij)
    assert np.array_equal(
        same_phase[0].tokens_ij[..., 3:7],
        np.zeros_like(same_phase[0].tokens_ij[..., 3:7]),
    )
    model = _small_system("03").eval()
    with torch.no_grad():
        same_output = model.forward_activity(_signed_pair_activity_batch(1))
        opposite_output = model.forward_activity(_signed_pair_activity_batch(-1))
    assert torch.equal(same_output.pair_component, opposite_output.pair_component)
    assert torch.equal(same_output.tokens, opposite_output.tokens)


def test_dct_edge_requires_both_endpoint_powers_to_pass_the_common_floor() -> None:
    floors = np.zeros((BAND_COUNT,), dtype=np.float64)
    chunk = tuple(
        iter_mean_difference_dct_chunks(
            _unilateral_pair_activity_batch(),
            energy_floors=floors,
            edge_chunk_size=64,
        )
    )
    assert len(chunk) == 1
    assert not chunk[0].support_mask.any()
    with torch.no_grad():
        output = _small_system("03").eval().forward_activity(
            _unilateral_pair_activity_batch()
        )
    assert not output.band_mask.any()
    assert not output.valid_pair_count.any()
