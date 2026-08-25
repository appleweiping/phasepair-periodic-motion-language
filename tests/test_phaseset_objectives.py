"""Tests for variable-positive PhaseSet retrieval objectives."""

from __future__ import annotations

import pytest
import torch

from phasepair_core.objectives import ObjectiveContractError
from phaseset_core.objectives import (
    PhaseSetRetrievalHead,
    TextBandMLP,
    band_text_scores,
    phaseset_retrieval_objective,
    variable_positive_symmetric_infonce,
)


def _unequal_positive_mask() -> torch.Tensor:
    # Row counts are deliberately 2, 1, and 3; every text column remains bound.
    return torch.tensor(
        [
            [True, True, False, False, False, False],
            [False, False, True, False, False, False],
            [False, False, False, True, True, True],
        ],
        dtype=torch.bool,
    ).contiguous()


def test_variable_positive_infonce_accepts_unequal_positive_counts() -> None:
    logits = torch.tensor(
        [
            [3.0, 2.5, -1.0, 0.0, 0.5, -0.5],
            [-1.0, 0.0, 2.0, 0.5, -0.5, 0.2],
            [0.2, -0.3, -0.5, 1.0, 1.5, 2.0],
        ],
        dtype=torch.float32,
        requires_grad=True,
    ).contiguous()
    motion_to_text, text_to_motion, loss = variable_positive_symmetric_infonce(
        logits,
        _unequal_positive_mask(),
    )
    assert torch.isfinite(torch.stack((motion_to_text, text_to_motion, loss))).all()
    assert torch.equal(loss, 0.5 * (motion_to_text + text_to_motion))
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize(
    "mask",
    (
        torch.tensor([[True, False], [False, False]], dtype=torch.bool),
        torch.tensor([[True, False], [True, False]], dtype=torch.bool),
    ),
)
def test_variable_positive_infonce_rejects_unbound_rows_or_columns(
    mask: torch.Tensor,
) -> None:
    logits = torch.zeros((2, 2), dtype=torch.float32)
    with pytest.raises(ObjectiveContractError, match="positive"):
        variable_positive_symmetric_infonce(logits, mask.contiguous())


def test_band_text_scores_are_band_permutation_invariant_and_zero_without_bands() -> None:
    torch.manual_seed(20260825)
    group_tokens = torch.randn((3, 6, 8), dtype=torch.float32).contiguous()
    band_mask = torch.tensor(
        [
            [True, True, False, True, False, True],
            [False, True, True, False, True, True],
            [False, False, False, False, False, False],
        ],
        dtype=torch.bool,
    ).contiguous()
    text = torch.randn((6, 8), dtype=torch.float32).contiguous()
    expected = band_text_scores(group_tokens, band_mask, text)
    permutation = torch.tensor([5, 1, 3, 0, 4, 2], dtype=torch.long)
    actual = band_text_scores(
        group_tokens[:, permutation].contiguous(),
        band_mask[:, permutation].contiguous(),
        text,
    )
    assert torch.allclose(actual, expected, atol=1e-7, rtol=1e-7)
    assert torch.equal(expected[2], torch.zeros_like(expected[2]))


def test_retrieval_objective_backpropagates_with_variable_positive_mask() -> None:
    torch.manual_seed(20260825)
    group_tokens = torch.randn(
        (3, 6, 8),
        dtype=torch.float32,
        requires_grad=True,
    ).contiguous()
    band_mask = torch.tensor(
        [
            [True, True, True, True, True, True],
            [True, False, True, False, True, False],
            [False, True, False, True, False, True],
        ],
        dtype=torch.bool,
    ).contiguous()
    text = torch.randn(
        (6, 8),
        dtype=torch.float32,
        requires_grad=True,
    ).contiguous()
    logit_scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
    mask = _unequal_positive_mask()
    output = phaseset_retrieval_objective(
        group_tokens,
        band_mask,
        text,
        mask,
        logit_scale,
    )
    assert tuple(output.logits.shape) == (3, 6)
    assert torch.equal(output.positive_mask, mask)
    assert torch.isfinite(output.loss)
    output.loss.backward()
    assert group_tokens.grad is not None
    assert text.grad is not None
    assert logit_scale.grad is not None
    assert torch.isfinite(group_tokens.grad).all()
    assert torch.isfinite(text.grad).all()
    assert torch.isfinite(logit_scale.grad).all()


def test_band_text_scores_use_fixed_masked_mean_with_band_specific_text() -> None:
    group = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]],
        dtype=torch.float32,
    ).contiguous()
    mask = torch.tensor([[True, True, False, False, False, False]], dtype=torch.bool)
    text_bands = torch.tensor(
        [
            [[1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0]],
            [[0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0]],
        ],
        dtype=torch.float32,
    ).contiguous()
    scores = band_text_scores(group, mask, text_bands)
    assert torch.equal(scores, torch.tensor([[0.5, 0.5]], dtype=torch.float32))


def test_text_band_mlp_is_shared_and_emits_six_semantic_embeddings() -> None:
    torch.manual_seed(7)
    projector = TextBandMLP(embedding_dim=8, hidden_dim=12)
    text = torch.randn((5, 8), dtype=torch.float32).contiguous()
    output = projector(text)
    assert tuple(output.shape) == (5, 6, 8)
    assert output.is_contiguous()
    assert len(tuple(projector.shared_projection.parameters())) == 4
    output.square().mean().backward()
    assert projector.band_embedding.grad is not None


def test_retrieval_head_adds_only_a_bounded_residual_to_frozen_base() -> None:
    torch.manual_seed(11)
    head = PhaseSetRetrievalHead(embedding_dim=8, residual_lambda_init=0.0)
    group = torch.randn((3, 6, 8), dtype=torch.float32).contiguous()
    band_mask = torch.ones((3, 6), dtype=torch.bool).contiguous()
    text = torch.randn((4, 8), dtype=torch.float32).contiguous()
    base = torch.randn((3, 4), dtype=torch.float32).contiguous()
    zero = head(group, band_mask, text, base)
    assert torch.equal(zero.scores, base)
    assert torch.equal(zero.bounded_residual_scale, torch.zeros_like(zero.bounded_residual_scale))
    with torch.no_grad():
        head.residual_lambda.fill_(100.0)
    saturated = head(group, band_mask, text, base)
    assert torch.equal(saturated.scores, base + saturated.periodic_scores)
    assert bool(torch.all(torch.abs(saturated.bounded_residual_scale) <= 1.0))
