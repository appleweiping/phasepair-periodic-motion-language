"""Qualification tests for the explicit differentiable periodic CPU oracle."""

from __future__ import annotations

import numpy as np
import torch

from phaseset_core.contracts import PreparedGroupBatch, group_commitment, skeleton_to_activity
from phaseset_core.models import PhaseSetEncoder
from phaseset_core.torch_periodic import torch_skeleton_to_activity


_ACTORS = 4
_TIME = 96


def _key(index: int) -> bytes:
    return index.to_bytes(32, "big")


def _torch_skeleton_batch(
    *,
    actor_order: tuple[int, ...] = tuple(range(_ACTORS)),
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    skeletons = torch.zeros((1, _ACTORS, _TIME, 22, 3), dtype=torch.float32)
    actor_mask = torch.ones((1, _ACTORS), dtype=torch.bool)
    frame_mask = torch.ones((1, _TIME), dtype=torch.bool)
    track_mask = torch.ones((1, _ACTORS, _TIME, 22), dtype=torch.bool)
    time = torch.arange(_TIME, dtype=torch.float32)
    for position, physical_actor in enumerate(actor_order):
        for joint in range(22):
            phase = 0.17 * physical_actor + 0.013 * joint
            skeletons[0, position, :, joint, 0] = torch.sin(
                (0.055 + 0.003 * physical_actor) * time + phase
            )
            skeletons[0, position, :, joint, 1] = torch.cos(
                (0.083 + 0.002 * joint) * time - phase
            )
            skeletons[0, position, :, joint, 2] = (
                0.001 * (physical_actor + 1) * time.square()
                + 0.01 * joint
            )
    return skeletons, actor_mask, frame_mask, track_mask


def _numpy_contract() -> PreparedGroupBatch:
    skeletons, actor_mask, frame_mask, track_mask = _torch_skeleton_batch()
    commitments = tuple(_key(index) for index in range(_ACTORS))
    return PreparedGroupBatch(
        np.ascontiguousarray(skeletons.numpy()),
        np.ascontiguousarray(actor_mask.numpy()),
        np.ascontiguousarray(frame_mask.numpy()),
        np.ascontiguousarray(track_mask.numpy()),
        (commitments,),
        (group_commitment(commitments),),
    )


def _model() -> PhaseSetEncoder:
    torch.manual_seed(20260826)
    return PhaseSetEncoder(embedding_dim=8, hidden_dim=12).eval()


def test_torch_activity_oracle_matches_numpy_contract() -> None:
    batch = _numpy_contract()
    expected = skeleton_to_activity(batch)
    actual = torch_skeleton_to_activity(
        torch.from_numpy(batch.skeletons.copy()),
        torch.from_numpy(batch.actor_mask.copy()),
        torch.from_numpy(batch.frame_mask.copy()),
        torch.from_numpy(batch.track_mask.copy()),
    )
    np.testing.assert_allclose(
        actual.activities.detach().numpy(),
        expected.activities,
        rtol=2e-6,
        atol=2e-6,
    )
    assert torch.equal(
        actual.activity_mask,
        torch.from_numpy(expected.activity_mask.copy()),
    )


def test_formal_encoder_can_call_differentiable_skeleton_oracle() -> None:
    batch = _numpy_contract()
    model = _model()
    skeletons = torch.from_numpy(batch.skeletons.copy()).requires_grad_()
    output = model.forward_skeleton_autograd_oracle(
        skeletons,
        torch.from_numpy(batch.actor_mask.copy()),
        torch.from_numpy(batch.frame_mask.copy()),
        torch.from_numpy(batch.track_mask.copy()),
    )
    production = model(batch, edge_chunk_size=64)
    assert torch.equal(output.band_mask, production.band_mask)
    assert torch.equal(output.valid_pair_count, production.valid_pair_count)
    torch.testing.assert_close(output.tokens, production.tokens, rtol=2e-5, atol=2e-5)
    output.tokens.square().sum().backward()
    assert skeletons.grad is not None
    assert bool(torch.isfinite(skeletons.grad).all().item())
    assert bool(torch.any(skeletons.grad != 0).item())


def test_raw_skeleton_input_gradient_is_actor_permutation_equivariant() -> None:
    """Permuting raw actors permutes skeleton gradients under a fixed CPU oracle."""

    actor_permutation = (2, 0, 3, 1)
    canonical, actor_mask, frame_mask, track_mask = _torch_skeleton_batch()
    permuted, permuted_actor_mask, permuted_frame_mask, permuted_track_mask = (
        _torch_skeleton_batch(actor_order=actor_permutation)
    )
    canonical.requires_grad_()
    permuted.requires_grad_()
    model = _model()

    canonical_output = model.forward_skeleton_autograd_oracle(
        canonical,
        actor_mask,
        frame_mask,
        track_mask,
    )
    canonical_loss = (
        canonical_output.tokens.square().sum()
        + 0.25 * canonical_output.pair_component.square().sum()
        + 0.125 * canonical_output.topology_delta.square().sum()
    )
    canonical_gradient = torch.autograd.grad(canonical_loss, canonical)[0]

    permuted_output = model.forward_skeleton_autograd_oracle(
        permuted,
        permuted_actor_mask,
        permuted_frame_mask,
        permuted_track_mask,
    )
    permuted_loss = (
        permuted_output.tokens.square().sum()
        + 0.25 * permuted_output.pair_component.square().sum()
        + 0.125 * permuted_output.topology_delta.square().sum()
    )
    permuted_gradient = torch.autograd.grad(permuted_loss, permuted)[0]

    # Actor permutation changes only the fixed summation order.  The frozen
    # CPU oracle therefore uses the registered cross-hardware tolerance rather
    # than claiming bitwise identity for this diagnostic path.
    torch.testing.assert_close(
        canonical_output.tokens,
        permuted_output.tokens,
        rtol=3e-5,
        atol=3e-5,
    )
    torch.testing.assert_close(
        permuted_gradient,
        canonical_gradient[:, actor_permutation],
        rtol=2e-4,
        atol=2e-4,
    )
    assert bool(torch.any(canonical_gradient != 0).item())


def test_differentiable_oracle_honors_scattered_actor_mask_and_padding() -> None:
    canonical, actor_mask, frame_mask, track_mask = _torch_skeleton_batch()
    padded = torch.zeros((1, 7, _TIME, 22, 3), dtype=torch.float32)
    padded_track = torch.zeros((1, 7, _TIME, 22), dtype=torch.bool)
    padded_actor_mask = torch.zeros((1, 7), dtype=torch.bool)
    positions = (6, 1, 4, 2)
    for source, destination in enumerate(positions):
        padded[:, destination] = canonical[:, source]
        padded_track[:, destination] = track_mask[:, source]
        padded_actor_mask[:, destination] = True
    canonical.requires_grad_()
    padded.requires_grad_()
    model = _model()
    expected = model.forward_skeleton_autograd_oracle(
        canonical,
        actor_mask,
        frame_mask,
        track_mask,
    )
    actual = model.forward_skeleton_autograd_oracle(
        padded,
        padded_actor_mask,
        frame_mask,
        padded_track,
    )
    torch.testing.assert_close(actual.tokens, expected.tokens, rtol=3e-5, atol=3e-5)
    expected_gradient = torch.autograd.grad(expected.tokens.square().sum(), canonical)[0]
    actual_gradient = torch.autograd.grad(actual.tokens.square().sum(), padded)[0]
    for source, destination in enumerate(positions):
        torch.testing.assert_close(
            actual_gradient[:, destination],
            expected_gradient[:, source],
            rtol=2e-4,
            atol=2e-4,
        )
    invalid_positions = (0, 3, 5)
    assert torch.equal(
        actual_gradient[:, invalid_positions],
        torch.zeros_like(actual_gradient[:, invalid_positions]),
    )
