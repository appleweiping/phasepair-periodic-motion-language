"""Strict, data-free PhasePair full-gallery retrieval evaluation.

The evaluator consumes only a frozen score matrix and explicit ground-truth
identities.  It never derives positives from model outputs or from text
equality, and it never compares one model with another.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


RANK_CUTOFFS = (1, 3, 5, 10)
CAPTIONS_PER_SOURCE = 3


class EvaluationError(ValueError):
    """Raised when a purported full-gallery evaluation input is not exact."""


def _raw32_sequence(name: str, values: Sequence[bytes]) -> tuple[bytes, ...]:
    if isinstance(values, (bytes, bytearray, memoryview, str)):
        raise EvaluationError(f"{name} must be a sequence of raw 32-byte values")
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise EvaluationError(f"{name} must be a finite sequence") from exc
    for ordinal, value in enumerate(normalized):
        if type(value) is not bytes or len(value) != 32:
            raise EvaluationError(f"{name}[{ordinal}] must be exactly bytes[32]")
    return normalized


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    return result


@dataclass(frozen=True)
class RetrievalMetrics:
    """One direction's exact ranks and descriptive retrieval statistics."""

    ranks: np.ndarray
    r_at_1: float
    r_at_3: float
    r_at_5: float
    r_at_10: float
    median_rank: float

    def recall_at(self, cutoff: int) -> float:
        if type(cutoff) is not int:
            raise EvaluationError("retrieval cutoff must be an exact built-in int")
        if cutoff == 1:
            return self.r_at_1
        if cutoff == 3:
            return self.r_at_3
        if cutoff == 5:
            return self.r_at_5
        if cutoff == 10:
            return self.r_at_10
        raise EvaluationError(f"unsupported retrieval cutoff: {cutoff}")


@dataclass(frozen=True)
class FullGalleryEvaluation:
    """Contract-complete result for one seed's frozen full gallery."""

    t2m: RetrievalMetrics
    m2t: RetrievalMetrics
    t2m_top1_by_source: np.ndarray
    m2t_any_caption_top1_by_source: np.ndarray
    t2m_cluster_macro_r_at_1: float
    m2t_any_caption_r_at_1: float
    bidirectional_mean_r_at_1: float


def _metrics(ranks: np.ndarray) -> RetrievalMetrics:
    if ranks.ndim != 1 or ranks.size == 0:
        raise AssertionError("internal rank vector must be nonempty and one-dimensional")
    readonly_ranks = _readonly_copy(ranks.astype(np.int64, copy=False))
    recalls = {
        cutoff: float(np.count_nonzero(readonly_ranks <= cutoff) / readonly_ranks.size)
        for cutoff in RANK_CUTOFFS
    }
    return RetrievalMetrics(
        ranks=readonly_ranks,
        r_at_1=recalls[1],
        r_at_3=recalls[3],
        r_at_5=recalls[5],
        r_at_10=recalls[10],
        median_rank=float(np.median(readonly_ranks)),
    )


def evaluate_full_gallery(
    scores: np.ndarray,
    *,
    positive_mask: np.ndarray,
    motion_source_cluster_ids: Sequence[bytes],
    caption_source_cluster_ids: Sequence[bytes],
    motion_pair_commitments: Sequence[bytes],
    caption_commitments: Sequence[bytes],
) -> FullGalleryEvaluation:
    """Evaluate one frozen motion-by-caption full-gallery score matrix.

    ``scores`` is shaped ``[C, 3*C]`` and must be finite, C-contiguous
    ``float32``.  The positive mask is supplied explicitly and must equal the
    source-cluster relation.  Ties are resolved by raw commitment bytes:
    motion pair commitment for text-to-motion, and private caption commitment
    for motion-to-text.
    """

    if type(scores) is not np.ndarray:
        raise EvaluationError("scores must be an exact numpy.ndarray, not a subclass")
    if scores.dtype != np.dtype(np.float32):
        raise EvaluationError("scores must have exact dtype float32")
    if scores.ndim != 2:
        raise EvaluationError("scores must be a two-dimensional full-gallery matrix")
    if not scores.flags.c_contiguous:
        raise EvaluationError("scores must be C-contiguous")
    scores = np.array(scores, dtype=np.float32, copy=True, order="C", subok=False)
    if not np.isfinite(scores).all():
        raise EvaluationError("scores must contain only finite values")

    motion_sources = _raw32_sequence("motion_source_cluster_ids", motion_source_cluster_ids)
    caption_sources = _raw32_sequence("caption_source_cluster_ids", caption_source_cluster_ids)
    motion_ties = _raw32_sequence("motion_pair_commitments", motion_pair_commitments)
    caption_ties = _raw32_sequence("caption_commitments", caption_commitments)

    cluster_count = len(motion_sources)
    caption_count = CAPTIONS_PER_SOURCE * cluster_count
    if cluster_count == 0:
        raise EvaluationError("the full gallery must contain at least one source cluster")
    if scores.shape != (cluster_count, caption_count):
        raise EvaluationError(
            f"scores shape must be exactly ({cluster_count}, {caption_count})"
        )
    if len(caption_sources) != caption_count:
        raise EvaluationError("caption_source_cluster_ids must contain exactly 3*C rows")
    if len(motion_ties) != cluster_count:
        raise EvaluationError("motion_pair_commitments must contain exactly C rows")
    if len(caption_ties) != caption_count:
        raise EvaluationError("caption_commitments must contain exactly 3*C rows")
    if len(set(motion_sources)) != cluster_count:
        raise EvaluationError("motion source-cluster IDs must be unique")
    if len(set(motion_ties)) != cluster_count:
        raise EvaluationError("motion pair commitments must be unique")
    if motion_ties != tuple(sorted(motion_ties)):
        raise EvaluationError("motion rows must be in raw32-ascending pair-commitment order")
    if len(set(caption_ties)) != caption_count:
        raise EvaluationError("caption commitments must be unique")

    motion_source_set = set(motion_sources)
    if any(source not in motion_source_set for source in caption_sources):
        raise EvaluationError("every caption source cluster must resolve to one motion")
    source_caption_indices: list[list[int]] = []
    for source in motion_sources:
        indices = [i for i, caption_source in enumerate(caption_sources) if caption_source == source]
        if len(indices) != CAPTIONS_PER_SOURCE:
            raise EvaluationError("every source cluster must have exactly three captions")
        indices.sort(key=lambda i: caption_ties[i])
        source_caption_indices.append(indices)

    if type(positive_mask) is not np.ndarray:
        raise EvaluationError("positive_mask must be an exact numpy.ndarray, not a subclass")
    if positive_mask.dtype != np.dtype(np.bool_):
        raise EvaluationError("positive_mask must have exact dtype bool")
    if not positive_mask.flags.c_contiguous:
        raise EvaluationError("positive_mask must be C-contiguous")
    positive_mask = np.array(
        positive_mask,
        dtype=np.bool_,
        copy=True,
        order="C",
        subok=False,
    )
    if positive_mask.shape != scores.shape:
        raise EvaluationError("positive_mask shape must exactly equal scores shape")
    expected_mask = np.fromiter(
        (
            motion_source == caption_source
            for motion_source in motion_sources
            for caption_source in caption_sources
        ),
        dtype=np.bool_,
        count=cluster_count * caption_count,
    ).reshape(scores.shape)
    if not np.array_equal(positive_mask, expected_mask):
        raise EvaluationError(
            "positive_mask must exactly equal source-cluster identity; "
            "cross-cluster text equality is not a positive"
        )

    # Each caption has exactly one positive motion.  Stable Python sorting plus
    # the complete raw commitment tie key makes the ordering total.
    t2m_ranks = np.empty(caption_count, dtype=np.int64)
    for caption_ordinal in range(caption_count):
        order = sorted(
            range(cluster_count),
            key=lambda motion_ordinal: (
                -float(scores[motion_ordinal, caption_ordinal]),
                motion_ties[motion_ordinal],
            ),
        )
        t2m_ranks[caption_ordinal] = next(
            rank
            for rank, motion_ordinal in enumerate(order, start=1)
            if expected_mask[motion_ordinal, caption_ordinal]
        )

    # Each motion has exactly three positive captions; its rank is the first
    # positive in the total score/commitment order ("any-caption" retrieval).
    m2t_ranks = np.empty(cluster_count, dtype=np.int64)
    for motion_ordinal in range(cluster_count):
        order = sorted(
            range(caption_count),
            key=lambda caption_ordinal: (
                -float(scores[motion_ordinal, caption_ordinal]),
                caption_ties[caption_ordinal],
            ),
        )
        m2t_ranks[motion_ordinal] = next(
            rank
            for rank, caption_ordinal in enumerate(order, start=1)
            if expected_mask[motion_ordinal, caption_ordinal]
        )

    t2m_top1 = np.empty((cluster_count, CAPTIONS_PER_SOURCE), dtype=np.bool_)
    for motion_ordinal, caption_indices in enumerate(source_caption_indices):
        for within_source_ordinal, caption_ordinal in enumerate(caption_indices):
            t2m_top1[motion_ordinal, within_source_ordinal] = t2m_ranks[caption_ordinal] == 1
    m2t_top1 = m2t_ranks == 1

    t2m_cluster_macro = float(np.mean(t2m_top1, axis=1, dtype=np.float64).mean(dtype=np.float64))
    m2t_any_caption = float(np.mean(m2t_top1, dtype=np.float64))

    return FullGalleryEvaluation(
        t2m=_metrics(t2m_ranks),
        m2t=_metrics(m2t_ranks),
        t2m_top1_by_source=_readonly_copy(t2m_top1),
        m2t_any_caption_top1_by_source=_readonly_copy(m2t_top1),
        t2m_cluster_macro_r_at_1=t2m_cluster_macro,
        m2t_any_caption_r_at_1=m2t_any_caption,
        bidirectional_mean_r_at_1=0.5 * (t2m_cluster_macro + m2t_any_caption),
    )


__all__ = [
    "CAPTIONS_PER_SOURCE",
    "EvaluationError",
    "FullGalleryEvaluation",
    "RANK_CUTOFFS",
    "RetrievalMetrics",
    "evaluate_full_gallery",
]
