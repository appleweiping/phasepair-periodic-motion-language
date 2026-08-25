"""Variable-positive retrieval objectives for six PhaseSet group tokens."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phasepair_core.objectives import (
    symmetric_multi_positive_infonce as _legacy_multi_positive_infonce,
)

from .periodic import BAND_COUNT


STATUS: Final = "DATA_FREE_PHASESET_OBJECTIVE_NONPRODUCTION_AUTHORITY0"


class PhaseSetObjectiveError(ValueError):
    """Raised when a group retrieval objective input is malformed."""


@dataclass(frozen=True, slots=True)
class PhaseSetObjectiveOutput:
    logits: Tensor
    motion_to_text_loss: Tensor
    text_to_motion_loss: Tensor
    loss: Tensor
    positive_mask: Tensor


@dataclass(frozen=True, slots=True)
class RetrievalScoreOutput:
    scores: Tensor
    frozen_base_scores: Tensor
    periodic_scores: Tensor
    text_band_embeddings: Tensor
    bounded_residual_scale: Tensor


class TextBandMLP(nn.Module):
    """Shared text projection conditioned only on a learned six-band identity."""

    def __init__(self, embedding_dim: int, hidden_dim: int | None = None) -> None:
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim < 2:
            raise TypeError("embedding_dim must be an exact int >=2")
        width = embedding_dim if hidden_dim is None else hidden_dim
        if type(width) is not int or width < 2:
            raise TypeError("hidden_dim must be an exact int >=2")
        self.embedding_dim = embedding_dim
        self.band_embedding = nn.Parameter(
            torch.empty((BAND_COUNT, embedding_dim), dtype=torch.float32)
        )
        nn.init.normal_(self.band_embedding, mean=0.0, std=embedding_dim**-0.5)
        self.shared_projection = nn.Sequential(
            nn.Linear(2 * embedding_dim, width),
            nn.GELU(),
            nn.Linear(width, embedding_dim),
        )

    def forward(self, text_embeddings: Tensor) -> Tensor:
        if type(text_embeddings) is not Tensor:
            raise TypeError("text_embeddings must be an exact torch.Tensor")
        if (
            text_embeddings.dtype != torch.float32
            or text_embeddings.ndim != 2
            or text_embeddings.shape[0] < 1
            or text_embeddings.shape[1] != self.embedding_dim
            or text_embeddings.device != self.band_embedding.device
            or not text_embeddings.is_contiguous()
            or not bool(torch.isfinite(text_embeddings).all().item())
        ):
            raise PhaseSetObjectiveError(
                "text_embeddings must be finite contiguous float32 [Q,D]"
            )
        query_count = int(text_embeddings.shape[0])
        text = text_embeddings[:, None, :].expand(-1, BAND_COUNT, -1)
        bands = self.band_embedding[None, :, :].expand(query_count, -1, -1)
        return self.shared_projection(torch.cat((text, bands), dim=-1)).contiguous()


class PhaseSetRetrievalHead(nn.Module):
    """Add a bounded periodic residual to an externally frozen base score."""

    def __init__(
        self,
        embedding_dim: int,
        *,
        text_hidden_dim: int | None = None,
        residual_lambda_init: float = 0.0,
    ) -> None:
        super().__init__()
        if type(residual_lambda_init) not in (float, int) or not math.isfinite(
            residual_lambda_init
        ):
            raise TypeError("residual_lambda_init must be a finite built-in number")
        self.text_band_mlp = TextBandMLP(embedding_dim, text_hidden_dim)
        self.residual_lambda = nn.Parameter(
            torch.tensor([float(residual_lambda_init)], dtype=torch.float32)
        )

    def forward(
        self,
        group_tokens: Tensor,
        band_mask: Tensor,
        text_embeddings: Tensor,
        frozen_base_scores: Tensor,
    ) -> RetrievalScoreOutput:
        text_bands = self.text_band_mlp(text_embeddings)
        periodic_scores = band_text_scores(group_tokens, band_mask, text_bands)
        if type(frozen_base_scores) is not Tensor:
            raise TypeError("frozen_base_scores must be an exact torch.Tensor")
        if (
            frozen_base_scores.dtype != torch.float32
            or tuple(frozen_base_scores.shape) != tuple(periodic_scores.shape)
            or frozen_base_scores.device != periodic_scores.device
            or not frozen_base_scores.is_contiguous()
            or frozen_base_scores.requires_grad
            or not bool(torch.isfinite(frozen_base_scores).all().item())
        ):
            raise PhaseSetObjectiveError(
                "frozen_base_scores must be finite non-grad contiguous float32 [B,Q]"
            )
        bounded = torch.tanh(self.residual_lambda)
        scores = frozen_base_scores + bounded[0] * periodic_scores
        return RetrievalScoreOutput(
            scores=scores.contiguous(),
            frozen_base_scores=frozen_base_scores,
            periodic_scores=periodic_scores,
            text_band_embeddings=text_bands,
            bounded_residual_scale=bounded,
        )


def variable_positive_symmetric_infonce(
    logits: Tensor,
    positive_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Wrap the audited low-level oracle without its legacy three-caption wrapper.

    Every motion row and every text column must have at least one positive, but
    row positive counts may differ and ``Q`` need not be a multiple of ``B``.
    """

    return _legacy_multi_positive_infonce(logits, positive_mask)


def band_text_scores(
    group_tokens: Tensor,
    band_mask: Tensor,
    text_embeddings: Tensor,
) -> Tensor:
    """Return the registered masked-mean band cosine scores ``[B,Q]``.

    ``text_embeddings`` may be projected band semantics ``[Q,6,D]``.  A
    two-dimensional ``[Q,D]`` input is retained as a data-free compatibility
    seam and is broadcast to all bands; production scoring uses
    :class:`TextBandMLP`.
    """

    if type(group_tokens) is not Tensor or type(band_mask) is not Tensor:
        raise TypeError("group_tokens and band_mask must be exact torch.Tensor objects")
    if type(text_embeddings) is not Tensor:
        raise TypeError("text_embeddings must be an exact torch.Tensor")
    if (
        group_tokens.dtype != torch.float32
        or group_tokens.ndim != 3
        or group_tokens.shape[1] != BAND_COUNT
        or not group_tokens.is_contiguous()
    ):
        raise PhaseSetObjectiveError("group_tokens must be contiguous float32 [B,6,D]")
    batch_size, _, width = group_tokens.shape
    if (
        band_mask.dtype != torch.bool
        or tuple(band_mask.shape) != (batch_size, BAND_COUNT)
        or band_mask.device != group_tokens.device
        or not band_mask.is_contiguous()
        or band_mask.requires_grad
    ):
        raise PhaseSetObjectiveError("band_mask must be contiguous bool [B,6]")
    valid_text_shape = (
        text_embeddings.ndim == 2
        and text_embeddings.shape[0] >= 1
        and text_embeddings.shape[1] == width
    ) or (
        text_embeddings.ndim == 3
        and text_embeddings.shape[0] >= 1
        and tuple(text_embeddings.shape[1:]) == (BAND_COUNT, width)
    )
    if (
        text_embeddings.dtype != torch.float32
        or not valid_text_shape
        or text_embeddings.device != group_tokens.device
        or not text_embeddings.is_contiguous()
    ):
        raise PhaseSetObjectiveError(
            "text_embeddings must be contiguous float32 [Q,D] or [Q,6,D]"
        )
    if not bool(torch.isfinite(group_tokens).all().item()) or not bool(
        torch.isfinite(text_embeddings).all().item()
    ):
        raise PhaseSetObjectiveError("token and text inputs must be finite")

    tokens = F.normalize(group_tokens, dim=-1, eps=1e-12)
    if text_embeddings.ndim == 2:
        text_bands = text_embeddings[:, None, :].expand(-1, BAND_COUNT, -1)
    else:
        text_bands = text_embeddings
    text = F.normalize(text_bands, dim=-1, eps=1e-12)
    similarity = torch.einsum("bkd,qkd->bkq", tokens, text)
    mask = band_mask.unsqueeze(-1)
    numerator = torch.where(mask, similarity, torch.zeros_like(similarity)).sum(dim=1)
    denominator = band_mask.sum(dim=1, dtype=similarity.dtype)[:, None]
    scores = torch.where(
        denominator > 0,
        numerator / denominator.clamp_min(1.0),
        torch.zeros_like(numerator),
    )
    return scores.contiguous()


def phaseset_retrieval_objective(
    group_tokens: Tensor,
    band_mask: Tensor,
    text_embeddings: Tensor,
    positive_mask: Tensor,
    logit_scale: Tensor | nn.Parameter,
) -> PhaseSetObjectiveOutput:
    """Compute variable-positive symmetric InfoNCE for group/text retrieval."""

    scores = band_text_scores(group_tokens, band_mask, text_embeddings)
    if type(logit_scale) not in (Tensor, nn.Parameter):
        raise TypeError("logit_scale must be an exact Tensor or Parameter")
    if (
        logit_scale.dtype != torch.float32
        or tuple(logit_scale.shape) != (1,)
        or logit_scale.device != scores.device
        or not logit_scale.is_contiguous()
        or not bool(torch.isfinite(logit_scale).all().item())
    ):
        raise PhaseSetObjectiveError("logit_scale must be finite contiguous float32[1]")
    if type(positive_mask) is not Tensor:
        raise TypeError("positive_mask must be an exact torch.Tensor")
    if (
        positive_mask.dtype != torch.bool
        or tuple(positive_mask.shape) != tuple(scores.shape)
        or positive_mask.device != scores.device
        or not positive_mask.is_contiguous()
        or positive_mask.requires_grad
    ):
        raise PhaseSetObjectiveError("positive_mask must be contiguous bool [B,Q]")
    scale = torch.exp(torch.clamp(logit_scale[0], min=0.0, max=math.log(100.0)))
    logits = (scale * scores).contiguous()
    motion_to_text, text_to_motion, loss = variable_positive_symmetric_infonce(
        logits,
        positive_mask,
    )
    return PhaseSetObjectiveOutput(
        logits=logits,
        motion_to_text_loss=motion_to_text,
        text_to_motion_loss=text_to_motion,
        loss=loss,
        positive_mask=positive_mask.clone(memory_format=torch.contiguous_format),
    )


__all__ = [
    "PhaseSetObjectiveError",
    "PhaseSetObjectiveOutput",
    "PhaseSetRetrievalHead",
    "RetrievalScoreOutput",
    "STATUS",
    "TextBandMLP",
    "band_text_scores",
    "phaseset_retrieval_objective",
    "variable_positive_symmetric_infonce",
]
