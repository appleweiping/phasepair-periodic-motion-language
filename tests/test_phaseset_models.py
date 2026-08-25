"""Permutation, padding, chunking, and topology tests for PhaseSet."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest
import torch

from phaseset_core.contracts import (
    PreparedActivityBatch,
    PreparedGroupBatch,
    group_commitment,
    skeleton_to_activity,
)
from phaseset_core.models import (
    ActorMeanBase,
    GroupTokenOutput,
    PhaseSetEncoder,
    PhaseSetOutput,
    SetPMABase,
    SocialTemporalBase,
    build_group_base,
)
from phaseset_core.periodic import ResourceLimitError


def _key(index: int) -> bytes:
    return hashlib.sha256(f"model-actor-{index}".encode("ascii")).digest()


def _physical_activity(actor_count: int) -> np.ndarray:
    values = np.zeros((actor_count, 17, 5), dtype=np.float32)
    grid = np.arange(16, dtype=np.float32)
    for actor in range(actor_count):
        for channel in range(5):
            values[actor, :16, channel] = np.cos(
                np.float32(0.23 + 0.015 * actor) * grid
                + np.float32(0.19 * channel + 0.07 * actor)
            )
    return values


def _activity_batch(
    actor_count: int,
    *,
    padded_actors: int | None = None,
    positions: tuple[int, ...] | None = None,
    physical_order: tuple[int, ...] | None = None,
) -> PreparedActivityBatch:
    padded = actor_count if padded_actors is None else padded_actors
    positions = tuple(range(actor_count)) if positions is None else positions
    physical_order = tuple(range(actor_count)) if physical_order is None else physical_order
    assert len(positions) == actor_count == len(physical_order)
    activities = np.zeros((1, padded, 17, 5), dtype=np.float32)
    actor_mask = np.zeros((1, padded), dtype=np.bool_)
    commitments: list[bytes | None] = [None] * padded
    physical = _physical_activity(actor_count)
    keys = tuple(_key(index) for index in range(actor_count))
    for position, physical_actor in zip(positions, physical_order, strict=True):
        activities[0, position] = physical[physical_actor]
        actor_mask[0, position] = True
        commitments[position] = keys[physical_actor]
    valid_mask = np.zeros((1, 17), dtype=np.bool_)
    valid_mask[:, :16] = True
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    for position in positions:
        activity_mask[0, position, :16] = True
    return PreparedActivityBatch(
        activities,
        actor_mask,
        valid_mask,
        activity_mask,
        (tuple(commitments),),
        (group_commitment(keys),),
    )


def _skeleton_batch(
    actor_count: int,
    *,
    padded_actors: int | None = None,
    positions: tuple[int, ...] | None = None,
    physical_order: tuple[int, ...] | None = None,
) -> PreparedGroupBatch:
    padded = actor_count if padded_actors is None else padded_actors
    positions = tuple(range(actor_count)) if positions is None else positions
    physical_order = (
        tuple(range(actor_count)) if physical_order is None else physical_order
    )
    skeletons = np.zeros((1, padded, 18, 22, 3), dtype=np.float32)
    actor_mask = np.zeros((1, padded), dtype=np.bool_)
    track_mask = np.zeros((1, padded, 18, 22), dtype=np.bool_)
    commitments: list[bytes | None] = [None] * padded
    keys = tuple(_key(index) for index in range(actor_count))
    grid = np.arange(17, dtype=np.float32)
    for position, physical_actor in zip(positions, physical_order, strict=True):
        actor_mask[0, position] = True
        commitments[position] = keys[physical_actor]
        track_mask[0, position, :17] = True
        for joint in range(22):
            skeletons[0, position, :17, joint, 0] = np.sin(
                np.float32(0.19 + 0.01 * physical_actor) * grid
                + np.float32(0.03 * joint)
            )
            skeletons[0, position, :16, joint, 1] = np.float32(
                0.01 * physical_actor + 0.001 * joint
            )
    frame_mask = np.zeros((1, 18), dtype=np.bool_)
    frame_mask[:, :17] = True
    return PreparedGroupBatch(
        skeletons,
        actor_mask,
        frame_mask,
        track_mask,
        (tuple(commitments),),
        (group_commitment(keys),),
    )


def _model() -> PhaseSetEncoder:
    torch.manual_seed(20260825)
    return PhaseSetEncoder(embedding_dim=8, hidden_dim=12)


def test_k2_topology_is_exact_positive_zero_and_has_zero_parameter_gradients() -> None:
    model = _model()
    batch = _activity_batch(2)
    full = model.forward_activity(batch, edge_chunk_size=64, include_topology=True)
    pair_only = model.forward_activity(
        batch, edge_chunk_size=64, include_topology=False
    )
    assert isinstance(full, GroupTokenOutput)
    assert PhaseSetOutput is GroupTokenOutput
    assert torch.equal(full.topology_delta, torch.zeros_like(full.topology_delta))
    assert not torch.signbit(full.topology_delta).any()
    assert torch.equal(full.topology_node_count, torch.zeros_like(full.topology_node_count))
    assert torch.equal(full.tokens, pair_only.tokens)

    full.tokens.square().sum().backward()
    for parameter in (model.topology_lambda, *model.topology_encoder.parameters()):
        assert parameter.grad is not None
        assert torch.equal(parameter.grad, torch.zeros_like(parameter.grad))
        assert not torch.signbit(parameter.grad).any()


def test_actor_permutation_padding_and_edge_chunk_size_do_not_change_output() -> None:
    model = _model().eval()
    canonical = _activity_batch(3)
    permuted_and_padded = _activity_batch(
        3,
        padded_actors=7,
        positions=(6, 1, 4),
        physical_order=(2, 0, 1),
    )
    with torch.no_grad():
        expected = model.forward_activity(canonical, edge_chunk_size=64)
        permuted = model.forward_activity(permuted_and_padded, edge_chunk_size=64)
        rechunked = model.forward_activity(canonical, edge_chunk_size=128)
    for field in ("tokens", "pair_component", "topology_delta"):
        assert torch.equal(getattr(expected, field), getattr(permuted, field))
        assert torch.equal(getattr(expected, field), getattr(rechunked, field))
    assert torch.equal(expected.band_mask, permuted.band_mask)
    assert torch.equal(expected.valid_pair_count, rechunked.valid_pair_count)
    assert int(expected.topology_node_count.max().item()) == 3


def test_pair_only_and_full_share_identical_postprocessing_when_delta_is_zero() -> None:
    model = _model().eval()
    batch = _activity_batch(
        2, padded_actors=5, positions=(4, 1), physical_order=(1, 0)
    )
    with torch.no_grad():
        full = model.forward_activity(
            batch, edge_chunk_size=64, include_topology=True
        )
        pair = model.forward_activity(
            batch, edge_chunk_size=64, include_topology=False
        )
    assert torch.equal(full.topology_delta, torch.zeros_like(full.topology_delta))
    assert torch.equal(full.tokens, pair.tokens)


def test_canonical_microblocks_make_cross_boundary_chunks_bitwise_identical() -> None:
    model = _model().eval()
    batch = _activity_batch(12)  # 66 unordered edges, crossing the boundary.
    with torch.no_grad():
        sixty_four = model.forward_activity(batch, edge_chunk_size=64)
        one_twenty_eight = model.forward_activity(batch, edge_chunk_size=128)
    for field in (
        "tokens",
        "band_mask",
        "pair_component",
        "topology_delta",
        "valid_pair_count",
        "topology_node_count",
    ):
        assert torch.equal(getattr(sixty_four, field), getattr(one_twenty_eight, field))


def test_pair_only_reports_exact_zero_topology_for_k3_and_uses_common_postprocess() -> None:
    model = _model().eval()
    batch = _activity_batch(3)
    with torch.no_grad():
        pair = model.forward_activity(
            batch, edge_chunk_size=64, include_topology=False
        )
        expected = model.common_postprocess(pair.pair_component)
        expected = torch.where(
            pair.band_mask[..., None],
            expected,
            torch.zeros_like(expected),
        )
    assert torch.equal(pair.topology_delta, torch.zeros_like(pair.topology_delta))
    assert not torch.signbit(pair.topology_delta).any()
    assert torch.equal(pair.tokens, expected)


def test_zero_learned_lambda_makes_full_and_pair_only_bitwise_identical() -> None:
    model = _model().eval()
    assert tuple(model.topology_lambda.shape) == (6,)
    assert model.topology_lambda.requires_grad
    with torch.no_grad():
        model.topology_lambda.zero_()
        full = model.forward_activity(
            _activity_batch(3), edge_chunk_size=64, include_topology=True
        )
        pair = model.forward_activity(
            _activity_batch(3), edge_chunk_size=64, include_topology=False
        )
    assert torch.equal(full.tokens, pair.tokens)


def test_symmetric_pair_token_is_invariant_to_endpoint_exchange() -> None:
    model = _model().eval()
    torch.manual_seed(19)
    left = torch.randn((7, 6, 13), dtype=torch.float32)
    right = torch.randn((7, 6, 13), dtype=torch.float32)
    with torch.no_grad():
        _, _, forward = model._edge_forward(left, right)
        _, _, reverse = model._edge_forward(right, left)
    assert torch.equal(forward, reverse)


def test_public_skeleton_forward_matches_internal_activity_seam() -> None:
    model = _model().eval()
    skeletons = _skeleton_batch(
        3,
        padded_actors=7,
        positions=(6, 1, 4),
        physical_order=(2, 0, 1),
    )
    activity = skeleton_to_activity(skeletons)
    with torch.no_grad():
        public = model(skeletons, edge_chunk_size=64)
        internal = model.forward_activity(activity, edge_chunk_size=64)
    for field in (
        "tokens",
        "band_mask",
        "pair_component",
        "topology_delta",
        "valid_pair_count",
        "topology_node_count",
    ):
        assert torch.equal(getattr(public, field), getattr(internal, field))


def test_public_k2_path_has_exact_zero_topology_and_rejects_activity_at_forward() -> None:
    model = _model().eval()
    skeletons = _skeleton_batch(2, padded_actors=5, positions=(4, 1))
    with torch.no_grad():
        full = model(skeletons, edge_chunk_size=64, include_topology=True)
        pair = model(skeletons, edge_chunk_size=64, include_topology=False)
    assert torch.equal(full.topology_delta, torch.zeros_like(full.topology_delta))
    assert not torch.signbit(full.topology_delta).any()
    assert torch.equal(full.tokens, pair.tokens)
    with pytest.raises(TypeError, match="PreparedGroupBatch"):
        model(skeleton_to_activity(skeletons), edge_chunk_size=64)  # type: ignore[arg-type]


def test_model_edge_budget_returns_resource_limit_before_edge_sampling() -> None:
    model = PhaseSetEncoder(embedding_dim=8, hidden_dim=12, edge_budget=5)
    with pytest.raises(ResourceLimitError, match="RESOURCE_LIMIT") as raised:
        model.forward_activity(_activity_batch(4), edge_chunk_size=64)
    assert raised.value.required_edges == 6


@pytest.mark.parametrize("observed_zero", (False, True))
def test_default_model_all_invalid_or_zero_energy_bands_are_exact_zero(
    observed_zero: bool,
) -> None:
    padded_time = 81
    valid_length = 80
    activities = np.zeros((1, 3, padded_time, 5), dtype=np.float32)
    activity_mask = np.zeros_like(activities, dtype=np.bool_)
    if observed_zero:
        activity_mask[:, :, :valid_length] = True
    actor_mask = np.ones((1, 3), dtype=np.bool_)
    frame_mask = np.zeros((1, padded_time), dtype=np.bool_)
    frame_mask[:, :valid_length] = True
    keys = tuple(_key(index) for index in range(3))
    batch = PreparedActivityBatch(
        activities,
        actor_mask,
        frame_mask,
        activity_mask,
        (keys,),
        (group_commitment(keys),),
    )
    with torch.no_grad():
        output = _model().eval().forward_activity(batch, edge_chunk_size=64)
    assert not output.band_mask.any()
    assert not output.valid_pair_count.any()
    assert not output.topology_node_count.any()
    for value in (output.tokens, output.pair_component, output.topology_delta):
        assert torch.equal(value, torch.zeros_like(value))
        assert not torch.signbit(value).any()


@pytest.mark.parametrize(
    "base_type,system_id",
    (
        (ActorMeanBase, "B0"),
        (SetPMABase, "B1"),
        (SocialTemporalBase, "B2"),
    ),
)
def test_registered_bases_are_permutation_and_padding_invariant(
    base_type: type[torch.nn.Module],
    system_id: str,
) -> None:
    torch.manual_seed(1234)
    model = base_type(hidden_dim=8, heads=2, ffn_dim=16, dropout=0.0).eval()
    assert getattr(model, "system_id") == system_id
    canonical = _skeleton_batch(3)
    scattered = _skeleton_batch(
        3,
        padded_actors=7,
        positions=(6, 1, 4),
        physical_order=(2, 0, 1),
    )
    with torch.no_grad():
        expected = model(canonical)
        actual = model(scattered)
    assert torch.equal(expected.group_embedding, actual.group_embedding)
    assert torch.equal(expected.actor_embeddings, actual.actor_embeddings[:, :3])
    assert torch.equal(
        actual.actor_embeddings[:, 3:],
        torch.zeros_like(actual.actor_embeddings[:, 3:]),
    )
    assert not torch.signbit(actual.actor_embeddings[:, 3:]).any()
    parameter_names = tuple(name for name, _ in model.named_parameters())
    assert not any("ordinal" in name or "actor_id" in name for name in parameter_names)


def test_base_factory_and_fixed_social_depth_match_registered_specification() -> None:
    assert type(
        build_group_base("B0", hidden_dim=8, heads=2, ffn_dim=16, dropout=0.0)
    ) is ActorMeanBase
    assert type(
        build_group_base("B1", hidden_dim=8, heads=2, ffn_dim=16, dropout=0.0)
    ) is SetPMABase
    social = build_group_base(
        "B2", hidden_dim=8, heads=2, ffn_dim=16, dropout=0.0
    )
    assert type(social) is SocialTemporalBase
    assert len(social.layers) == 4


def test_social_temporal_actor_attention_never_materializes_k_by_k(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    actor_count = 65
    observed: list[tuple[int, ...]] = []
    original_matmul = torch.matmul

    def tracked_matmul(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
        result = original_matmul(left, right)
        if result.ndim == 4 and result.shape[-1] == actor_count:
            observed.append(tuple(int(axis) for axis in result.shape))
        return result

    monkeypatch.setattr(torch, "matmul", tracked_matmul)
    torch.manual_seed(987)
    model = SocialTemporalBase(
        hidden_dim=8,
        heads=2,
        ffn_dim=16,
        dropout=0.0,
    ).eval()
    with torch.no_grad():
        output = model(_skeleton_batch(actor_count))
    assert output.group_embedding.shape == (1, 8)
    assert observed
    assert max(shape[-2] for shape in observed) <= 64
    assert not any(shape[-2:] == (actor_count, actor_count) for shape in observed)
