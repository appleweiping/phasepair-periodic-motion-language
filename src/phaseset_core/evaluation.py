"""Deterministic variable-positive retrieval and Group-Hard-32 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import re
import numpy as np


class EvaluationContractError(ValueError):
    """An evaluation input or frozen hard-gallery contract is invalid."""


@dataclass(frozen=True, slots=True)
class RetrievalDataset:
    """One frozen motion-by-caption score table and its positive relation."""

    scores: np.ndarray
    motion_commitments: tuple[bytes, ...]
    caption_commitments: tuple[bytes, ...]
    positive_motion_indices: tuple[tuple[int, ...], ...]
    group_sizes: np.ndarray
    component_labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DirectionMetrics:
    query_count: int
    aggregation_unit_count: int
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    median_rank: float


@dataclass(frozen=True, slots=True)
class BidirectionalMetrics:
    text_to_motion: DirectionMetrics
    motion_to_text: DirectionMetrics
    primary: float


@dataclass(frozen=True, slots=True)
class CaptureR1Contribution:
    """One capture-cluster's two directional R@1 contribution.

    A capture cluster is one connected component of the frozen caption-to-
    positive-motion relation. Commitments, rather than table positions, define
    canonical query order and the cluster identity.
    """

    capture_commitment: bytes
    caption_commitments: tuple[bytes, ...]
    motion_commitments: tuple[bytes, ...]
    text_to_motion_hits: tuple[int, ...]
    motion_to_text_hits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class EvaluationReport:
    full_gallery: BidirectionalMetrics
    by_group_size: tuple[tuple[int, BidirectionalMetrics], ...]
    macro_group_size: BidirectionalMetrics
    leave_one_component_out: tuple[tuple[str, BidirectionalMetrics], ...]
    capture_contributions: tuple[CaptureR1Contribution, ...]
    dataset_sha256: str
    evaluation_type: str = "UNBOUND_SYNTHETIC"
    census_provenance_sha256: str | None = None
    score_provenance_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class HardNegativeScorerProvenance:
    """Externally frozen text-only scorer lineage for hard-negative selection."""

    scorer_id: str
    scorer_code_sha256: str
    scorer_checkpoint_sha256: str
    freeze_receipt_sha256: str
    caption_manifest_sha256: str
    motion_manifest_sha256: str
    frozen_at_utc: str
    evaluated_checkpoint_sha256s: tuple[str, ...]
    text_only: bool = True
    frozen_before_evaluation: bool = True


@dataclass(frozen=True, slots=True)
class HardGalleryFreezeBinding:
    """Trusted expected digests fixed before hard-gallery construction."""

    scorer_provenance_sha256: str
    similarity_sha256: str
    freeze_receipt_sha256: str
    caption_manifest_sha256: str
    motion_manifest_sha256: str
    external_authorization_sha256: str


@dataclass(frozen=True, slots=True)
class HardNegativeFeatures:
    """Motion-side matching fields and frozen text-to-motion similarity."""

    duration_seconds: np.ndarray
    per_actor_band_power: np.ndarray
    actor_mask: np.ndarray
    total_motion_energy: np.ndarray
    root_speed: np.ndarray
    frozen_text_similarity: np.ndarray
    scorer_provenance: HardNegativeScorerProvenance


@dataclass(frozen=True, slots=True)
class HardGalleryConfig:
    """Caller-frozen calipers and maximum gallery size.

    ``candidate_count`` is a cap, not a promise to manufacture 32 candidates.
    A native K stratum with only 27 captures therefore yields a 27-item
    Group-Hard-32 gallery when all 26 negatives pass the frozen calipers.
    Failure never widens a caliper or borrows a motion from another K.
    """

    candidate_count: int = 32
    duration_absolute_max: float = 0.0
    band_power_log_linf_max: float = 0.10
    total_energy_log_max: float = 0.10
    root_speed_log_max: float = 0.10


@dataclass(frozen=True, slots=True)
class HardGallery:
    caption_index: int
    motion_indices: tuple[int, ...]
    maximum_candidate_count: int
    gallery_sha256: str


@dataclass(frozen=True, slots=True)
class HardGalleryCollection:
    galleries: tuple[HardGallery, ...]
    scorer_provenance: HardNegativeScorerProvenance
    freeze_binding: HardGalleryFreezeBinding
    selection_binding_sha256: str
    retrieval_census_sha256: str
    collection_sha256: str


@dataclass(frozen=True, slots=True)
class _CaptureCluster:
    commitment: bytes
    motion_indices: tuple[int, ...]
    caption_indices: tuple[int, ...]


def _commitment(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise EvaluationContractError(f"{label} must be exact bytes32")
    return value


def _readonly_array(
    value: object,
    *,
    dtype: np.dtype[object] | type[np.generic],
    ndim: int,
    label: str,
) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != ndim:
        raise EvaluationContractError(f"{label} must have rank {ndim}")
    if array.dtype != np.dtype(dtype):
        raise EvaluationContractError(f"{label} must have dtype {np.dtype(dtype)}")
    if not array.flags.c_contiguous:
        raise EvaluationContractError(f"{label} must be C-contiguous")
    detached = array.copy(order="C")
    detached.setflags(write=False)
    return detached


def validate_retrieval_dataset(value: object) -> RetrievalDataset:
    if type(value) is not RetrievalDataset:
        raise TypeError("dataset must be exact RetrievalDataset")
    scores = _readonly_array(value.scores, dtype=np.float64, ndim=2, label="scores")
    group_sizes = _readonly_array(
        value.group_sizes,
        dtype=np.int64,
        ndim=1,
        label="group_sizes",
    )
    motion_count, caption_count = scores.shape
    if motion_count < 2 or caption_count < 2 or not np.isfinite(scores).all():
        raise EvaluationContractError("scores must be finite with at least two rows/columns")
    if group_sizes.shape != (motion_count,) or np.any(group_sizes < 2):
        raise EvaluationContractError("group_sizes must contain one K>=2 per motion")
    if (
        type(value.motion_commitments) is not tuple
        or len(value.motion_commitments) != motion_count
        or type(value.caption_commitments) is not tuple
        or len(value.caption_commitments) != caption_count
    ):
        raise EvaluationContractError("commitment census does not match score shape")
    motions = tuple(
        _commitment(item, f"motion_commitments[{index}]")
        for index, item in enumerate(value.motion_commitments)
    )
    captions = tuple(
        _commitment(item, f"caption_commitments[{index}]")
        for index, item in enumerate(value.caption_commitments)
    )
    if len(set(motions)) != motion_count or len(set(captions)) != caption_count:
        raise EvaluationContractError("motion and caption commitments must be unique")
    if (
        type(value.positive_motion_indices) is not tuple
        or len(value.positive_motion_indices) != caption_count
    ):
        raise EvaluationContractError("positive relation must contain one row per caption")
    positives: list[tuple[int, ...]] = []
    motion_positive_count = np.zeros(motion_count, dtype=np.int64)
    for caption_index, row in enumerate(value.positive_motion_indices):
        if type(row) is not tuple or not row:
            raise EvaluationContractError(f"caption {caption_index} has no positive motion")
        if any(type(item) is not int for item in row):
            raise EvaluationContractError("positive indices must be exact integers")
        canonical = tuple(sorted(set(row)))
        if canonical != row or canonical[0] < 0 or canonical[-1] >= motion_count:
            raise EvaluationContractError("positive indices must be sorted, unique, and in range")
        positives.append(canonical)
        motion_positive_count[np.asarray(canonical, dtype=np.int64)] += 1
    if np.any(motion_positive_count == 0):
        raise EvaluationContractError("every motion must have at least one positive caption")
    if type(value.component_labels) is not tuple or len(value.component_labels) != motion_count:
        raise EvaluationContractError("component_labels must contain one row per motion")
    if any(type(item) is not str or not item for item in value.component_labels):
        raise EvaluationContractError("component labels must be nonempty exact strings")
    components = tuple(value.component_labels)
    for caption_index, row in enumerate(positives):
        if len({int(group_sizes[index]) for index in row}) != 1:
            raise EvaluationContractError(
                f"caption {caption_index} positive motions span multiple K values"
            )
        if len({components[index] for index in row}) != 1:
            raise EvaluationContractError(
                f"caption {caption_index} positive motions span multiple components"
            )
    return RetrievalDataset(
        scores=scores,
        motion_commitments=motions,
        caption_commitments=captions,
        positive_motion_indices=tuple(positives),
        group_sizes=group_sizes,
        component_labels=components,
    )


def _ranks(
    scores_by_query: np.ndarray,
    target_commitments: tuple[bytes, ...],
    positives_by_query: tuple[tuple[int, ...], ...],
) -> np.ndarray:
    ranks = np.empty(scores_by_query.shape[0], dtype=np.int64)
    for query_index, row in enumerate(scores_by_query):
        order = sorted(
            range(row.shape[0]),
            key=lambda target: (-float(row[target]), target_commitments[target]),
        )
        locations = {target: rank for rank, target in enumerate(order, start=1)}
        ranks[query_index] = min(locations[target] for target in positives_by_query[query_index])
    return ranks


def _capture_clusters(dataset: RetrievalDataset) -> tuple[_CaptureCluster, ...]:
    """Return canonical capture units from the frozen positive-relation graph."""

    motion_count, caption_count = dataset.scores.shape
    parent = list(range(motion_count + caption_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left == root_right:
            return
        if root_left < root_right:
            parent[root_right] = root_left
        else:
            parent[root_left] = root_right

    for caption_index, positives in enumerate(dataset.positive_motion_indices):
        caption_node = motion_count + caption_index
        for motion_index in positives:
            union(caption_node, motion_index)

    motions_by_root: dict[int, list[int]] = {}
    captions_by_root: dict[int, list[int]] = {}
    for motion_index in range(motion_count):
        motions_by_root.setdefault(find(motion_index), []).append(motion_index)
    for caption_index in range(caption_count):
        captions_by_root.setdefault(find(motion_count + caption_index), []).append(caption_index)
    if set(motions_by_root) != set(captions_by_root):
        raise EvaluationContractError("positive relation produced an incomplete capture cluster")

    clusters: list[_CaptureCluster] = []
    for root in motions_by_root:
        motion_indices = tuple(
            sorted(
                motions_by_root[root],
                key=lambda index: dataset.motion_commitments[index],
            )
        )
        caption_indices = tuple(
            sorted(
                captions_by_root[root],
                key=lambda index: dataset.caption_commitments[index],
            )
        )
        motion_set = set(motion_indices)
        payload = {
            "caption_positive_motion_edges": [
                {
                    "caption": dataset.caption_commitments[caption_index].hex(),
                    "motions": [
                        dataset.motion_commitments[motion_index].hex()
                        for motion_index in dataset.positive_motion_indices[caption_index]
                    ],
                }
                for caption_index in caption_indices
            ],
            "domain": "phaseset-capture-cluster-v1",
            "motions": [
                dataset.motion_commitments[motion_index].hex()
                for motion_index in motion_indices
            ],
        }
        if any(
            not set(dataset.positive_motion_indices[caption_index]).issubset(motion_set)
            for caption_index in caption_indices
        ):
            raise EvaluationContractError("capture cluster split a positive relation")
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
        clusters.append(
            _CaptureCluster(
                commitment=hashlib.sha256(raw).digest(),
                motion_indices=motion_indices,
                caption_indices=caption_indices,
            )
        )
    ordered = tuple(sorted(clusters, key=lambda value: value.commitment))
    if len({cluster.commitment for cluster in ordered}) != len(ordered):
        raise EvaluationContractError("capture-cluster commitments are not unique")
    return ordered


def _metrics(
    ranks: np.ndarray,
    query_groups: tuple[tuple[int, ...], ...],
) -> DirectionMetrics:
    """Aggregate queries inside capture first, then macro-average captures."""

    if ranks.ndim != 1 or ranks.size == 0 or not query_groups:
        raise EvaluationContractError("direction metrics require queries and capture units")
    flattened = [index for group in query_groups for index in group]
    if (
        any(type(group) is not tuple or not group for group in query_groups)
        or sorted(flattened) != list(range(int(ranks.size)))
        or len(flattened) != len(set(flattened))
    ):
        raise EvaluationContractError("direction query groups must partition all queries")

    def capture_macro_recall(limit: int) -> float:
        return float(
            np.mean(
                [
                    np.mean(ranks[np.asarray(group, dtype=np.int64)] <= limit)
                    for group in query_groups
                ]
            )
        )

    return DirectionMetrics(
        query_count=int(ranks.size),
        aggregation_unit_count=len(query_groups),
        recall_at_1=capture_macro_recall(1),
        recall_at_3=capture_macro_recall(3),
        recall_at_5=capture_macro_recall(5),
        recall_at_10=capture_macro_recall(10),
        median_rank=float(
            np.mean(
                [
                    np.median(ranks[np.asarray(group, dtype=np.int64)])
                    for group in query_groups
                ]
            )
        ),
    )


def _bidirectional_ranks(dataset: RetrievalDataset) -> tuple[np.ndarray, np.ndarray]:
    text_ranks = _ranks(
        dataset.scores.T,
        dataset.motion_commitments,
        dataset.positive_motion_indices,
    )
    captions_by_motion: list[list[int]] = [[] for _ in dataset.motion_commitments]
    for caption_index, positives in enumerate(dataset.positive_motion_indices):
        for motion_index in positives:
            captions_by_motion[motion_index].append(caption_index)
    motion_ranks = _ranks(
        dataset.scores,
        dataset.caption_commitments,
        tuple(tuple(row) for row in captions_by_motion),
    )
    return text_ranks, motion_ranks


def _evaluate(dataset: RetrievalDataset) -> BidirectionalMetrics:
    text_ranks, motion_ranks = _bidirectional_ranks(dataset)
    clusters = _capture_clusters(dataset)
    text = _metrics(
        text_ranks,
        tuple(cluster.caption_indices for cluster in clusters),
    )
    motion = _metrics(
        motion_ranks,
        tuple(cluster.motion_indices for cluster in clusters),
    )
    return BidirectionalMetrics(
        text_to_motion=text,
        motion_to_text=motion,
        primary=0.5 * (text.recall_at_1 + motion.recall_at_1),
    )


def capture_r1_contributions(
    value: object,
) -> tuple[CaptureR1Contribution, ...]:
    """Expose the exact capture-macro primary endpoint contribution rows."""

    dataset = validate_retrieval_dataset(value)
    text_ranks, motion_ranks = _bidirectional_ranks(dataset)
    output: list[CaptureR1Contribution] = []
    for cluster in _capture_clusters(dataset):
        output.append(
            CaptureR1Contribution(
                capture_commitment=cluster.commitment,
                caption_commitments=tuple(
                    dataset.caption_commitments[index]
                    for index in cluster.caption_indices
                ),
                motion_commitments=tuple(
                    dataset.motion_commitments[index]
                    for index in cluster.motion_indices
                ),
                text_to_motion_hits=tuple(
                    int(text_ranks[index] <= 1)
                    for index in cluster.caption_indices
                ),
                motion_to_text_hits=tuple(
                    int(motion_ranks[index] <= 1)
                    for index in cluster.motion_indices
                ),
            )
        )
    return tuple(output)


def capture_cluster_commitments(value: object) -> tuple[bytes, ...]:
    """Return canonical capture identities without exposing table positions."""

    dataset = validate_retrieval_dataset(value)
    return tuple(cluster.commitment for cluster in _capture_clusters(dataset))


def _subset(dataset: RetrievalDataset, motion_indices: tuple[int, ...]) -> RetrievalDataset:
    selected = set(motion_indices)
    captions = tuple(
        index
        for index, positives in enumerate(dataset.positive_motion_indices)
        if set(positives).issubset(selected)
    )
    if len(motion_indices) < 2 or len(captions) < 2:
        raise EvaluationContractError("evaluation subset has fewer than two motions/captions")
    remap = {old: new for new, old in enumerate(motion_indices)}
    positives = tuple(
        tuple(remap[index] for index in dataset.positive_motion_indices[caption])
        for caption in captions
    )
    return validate_retrieval_dataset(
        RetrievalDataset(
            scores=np.ascontiguousarray(
                dataset.scores[np.ix_(np.asarray(motion_indices), np.asarray(captions))],
                dtype=np.float64,
            ),
            motion_commitments=tuple(dataset.motion_commitments[index] for index in motion_indices),
            caption_commitments=tuple(dataset.caption_commitments[index] for index in captions),
            positive_motion_indices=positives,
            group_sizes=np.ascontiguousarray(dataset.group_sizes[list(motion_indices)]),
            component_labels=tuple(dataset.component_labels[index] for index in motion_indices),
        )
    )


def _macro(rows: tuple[BidirectionalMetrics, ...]) -> BidirectionalMetrics:
    if not rows:
        raise EvaluationContractError("macro metrics require at least one stratum")

    def direction(values: tuple[DirectionMetrics, ...]) -> DirectionMetrics:
        count = len(values)
        return DirectionMetrics(
            query_count=sum(item.query_count for item in values),
            aggregation_unit_count=sum(item.aggregation_unit_count for item in values),
            recall_at_1=sum(item.recall_at_1 for item in values) / count,
            recall_at_3=sum(item.recall_at_3 for item in values) / count,
            recall_at_5=sum(item.recall_at_5 for item in values) / count,
            recall_at_10=sum(item.recall_at_10 for item in values) / count,
            median_rank=sum(item.median_rank for item in values) / count,
        )

    text = direction(tuple(item.text_to_motion for item in rows))
    motion = direction(tuple(item.motion_to_text for item in rows))
    return BidirectionalMetrics(
        text_to_motion=text,
        motion_to_text=motion,
        primary=sum(item.primary for item in rows) / len(rows),
    )


def _dataset_digest(dataset: RetrievalDataset) -> str:
    payload = {
        "caption_commitments": [value.hex() for value in dataset.caption_commitments],
        "component_labels": list(dataset.component_labels),
        "group_sizes": dataset.group_sizes.tolist(),
        "motion_commitments": [value.hex() for value in dataset.motion_commitments],
        "positive_motion_indices": [list(row) for row in dataset.positive_motion_indices],
        "score_sha256": hashlib.sha256(dataset.scores.astype("<f8").tobytes()).hexdigest(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def retrieval_census_sha256(dataset_value: object) -> str:
    """Digest query identities/positives/strata without binding model scores."""

    dataset = validate_retrieval_dataset(dataset_value)
    payload = {
        "caption_commitments": [value.hex() for value in dataset.caption_commitments],
        "component_labels": list(dataset.component_labels),
        "group_sizes": dataset.group_sizes.tolist(),
        "motion_commitments": [value.hex() for value in dataset.motion_commitments],
        "positive_motion_indices": [list(row) for row in dataset.positive_motion_indices],
        "schema": "phaseset-retrieval-query-census-v1",
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def canonical_evaluation_report_bytes(value: object) -> bytes:
    """Serialize one computed evaluation report with exact private lineage.

    This format is intended for private execution evidence. Commitments are
    cryptographic identities, never model inputs, and are retained here so a
    later paired bootstrap can prove an identical query census.
    """

    if type(value) is not EvaluationReport:
        raise TypeError("value must be exact EvaluationReport")

    def direction(row: object) -> dict[str, object]:
        if type(row) is not DirectionMetrics:
            raise TypeError("direction row must be exact DirectionMetrics")
        if (
            type(row.query_count) is not int
            or row.query_count < 1
            or type(row.aggregation_unit_count) is not int
            or row.aggregation_unit_count < 1
        ):
            raise EvaluationContractError("direction metric counts are invalid")
        values = {
            name: float(getattr(row, name))
            for name in (
                "recall_at_1",
                "recall_at_3",
                "recall_at_5",
                "recall_at_10",
                "median_rank",
            )
        }
        if any(not math.isfinite(item) for item in values.values()):
            raise EvaluationContractError("direction metrics must be finite")
        if any(
            not 0.0 <= values[name] <= 1.0
            for name in ("recall_at_1", "recall_at_3", "recall_at_5", "recall_at_10")
        ):
            raise EvaluationContractError("recall metrics must lie in [0,1]")
        return {
            "aggregation_unit_count": row.aggregation_unit_count,
            "median_rank_hex": values["median_rank"].hex(),
            "query_count": row.query_count,
            "recall_at_10_hex": values["recall_at_10"].hex(),
            "recall_at_1_hex": values["recall_at_1"].hex(),
            "recall_at_3_hex": values["recall_at_3"].hex(),
            "recall_at_5_hex": values["recall_at_5"].hex(),
        }

    def bidirectional(row: object) -> dict[str, object]:
        if type(row) is not BidirectionalMetrics:
            raise TypeError("metric row must be exact BidirectionalMetrics")
        primary = float(row.primary)
        if not math.isfinite(primary) or not 0.0 <= primary <= 1.0:
            raise EvaluationContractError("primary metric must lie in [0,1]")
        return {
            "motion_to_text": direction(row.motion_to_text),
            "primary_hex": primary.hex(),
            "text_to_motion": direction(row.text_to_motion),
        }

    if (
        type(value.dataset_sha256) is not str
        or len(value.dataset_sha256) != 64
        or any(character not in "0123456789abcdef" for character in value.dataset_sha256)
    ):
        raise EvaluationContractError("evaluation dataset digest is invalid")
    contributions: list[dict[str, object]] = []
    for row in value.capture_contributions:
        if type(row) is not CaptureR1Contribution:
            raise TypeError("capture contribution must be exact CaptureR1Contribution")
        if (
            type(row.capture_commitment) is not bytes
            or len(row.capture_commitment) != 32
            or not row.caption_commitments
            or not row.motion_commitments
            or row.caption_commitments != tuple(sorted(row.caption_commitments))
            or row.motion_commitments != tuple(sorted(row.motion_commitments))
            or len(row.text_to_motion_hits) != len(row.caption_commitments)
            or len(row.motion_to_text_hits) != len(row.motion_commitments)
            or any(item not in {0, 1} for item in row.text_to_motion_hits)
            or any(item not in {0, 1} for item in row.motion_to_text_hits)
        ):
            raise EvaluationContractError("capture contribution lineage is invalid")
        contributions.append(
            {
                "caption_commitments": [item.hex() for item in row.caption_commitments],
                "capture_commitment": row.capture_commitment.hex(),
                "motion_commitments": [item.hex() for item in row.motion_commitments],
                "motion_to_text_hits": list(row.motion_to_text_hits),
                "text_to_motion_hits": list(row.text_to_motion_hits),
            }
        )
    payload = {
        "by_group_size": [
            {"group_size": group_size, "metrics": bidirectional(metrics)}
            for group_size, metrics in value.by_group_size
        ],
        "capture_contributions": contributions,
        "dataset_sha256": value.dataset_sha256,
        "evaluation_type": value.evaluation_type,
        "full_gallery": bidirectional(value.full_gallery),
        "leave_one_component_out": [
            {"component": label, "metrics": bidirectional(metrics)}
            for label, metrics in value.leave_one_component_out
        ],
        "macro_group_size": bidirectional(value.macro_group_size),
        "census_provenance_sha256": value.census_provenance_sha256,
        "schema": "phaseset-private-evaluation-report-v1",
        "score_provenance_sha256": value.score_provenance_sha256,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii") + b"\n"


def evaluate_retrieval(value: object) -> EvaluationReport:
    """Evaluate full gallery, K strata, macro-K, and component sensitivity."""

    dataset = validate_retrieval_dataset(value)
    by_k: list[tuple[int, BidirectionalMetrics]] = []
    for group_size in sorted(set(int(item) for item in dataset.group_sizes)):
        indices = tuple(
            index for index, value in enumerate(dataset.group_sizes) if int(value) == group_size
        )
        if len(indices) < 2:
            raise EvaluationContractError(f"K={group_size} stratum contains fewer than two motions")
        by_k.append((group_size, _evaluate(_subset(dataset, indices))))
    component_rows: list[tuple[str, BidirectionalMetrics]] = []
    component_labels = sorted(set(dataset.component_labels))
    if len(component_labels) < 2:
        raise EvaluationContractError("leave-one-component-out requires at least two components")
    for label in component_labels:
        indices = tuple(
            index for index, value in enumerate(dataset.component_labels) if value != label
        )
        if len(indices) < 2:
            raise EvaluationContractError(
                f"leaving component {label!r} yields fewer than two motions"
            )
        try:
            metrics = _evaluate(_subset(dataset, indices))
        except EvaluationContractError as exc:
            raise EvaluationContractError(
                f"leaving component {label!r} is not evaluable: {exc}"
            ) from exc
        component_rows.append((label, metrics))
    return EvaluationReport(
        full_gallery=_evaluate(dataset),
        by_group_size=tuple(by_k),
        macro_group_size=_macro(tuple(row for _, row in by_k)),
        leave_one_component_out=tuple(component_rows),
        capture_contributions=capture_r1_contributions(dataset),
        dataset_sha256=_dataset_digest(dataset),
    )


def _finite_vector(
    value: object,
    shape: tuple[int, ...],
    label: str,
    *,
    allow_negative: bool = False,
) -> np.ndarray:
    array = _readonly_array(value, dtype=np.float64, ndim=len(shape), label=label)
    if (
        array.shape != shape
        or not np.isfinite(array).all()
        or (not allow_negative and np.any(array < 0.0))
    ):
        raise EvaluationContractError(f"{label} has invalid shape or values")
    return array


def _array_sha256(value: np.ndarray, dtype: str) -> str:
    canonical = np.asarray(value, dtype=dtype, order="C")
    header = json.dumps(
        {"dtype": canonical.dtype.str, "shape": list(canonical.shape)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(header + b"\n" + canonical.tobytes(order="C")).hexdigest()


_LOWER_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_SECOND_UTC = re.compile(
    r"(?:19|20)\d\d-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01])"
    r"T(?:[01]\d|2[0-3]):[0-5]\d:[0-5]\dZ\Z"
)


def _hard_scorer_provenance_bytes(value: object) -> bytes:
    if type(value) is not HardNegativeScorerProvenance:
        raise TypeError(
            "scorer_provenance must be exact HardNegativeScorerProvenance"
        )
    if (
        type(value.scorer_id) is not str
        or not value.scorer_id
        or not value.scorer_id.isascii()
        or len(value.scorer_id) > 128
    ):
        raise EvaluationContractError("hard scorer id must be bounded nonempty ASCII")
    digest_fields = (
        "scorer_code_sha256",
        "scorer_checkpoint_sha256",
        "freeze_receipt_sha256",
        "caption_manifest_sha256",
        "motion_manifest_sha256",
    )
    for name in digest_fields:
        item = getattr(value, name)
        if type(item) is not str or _LOWER_SHA256.fullmatch(item) is None:
            raise EvaluationContractError(f"hard scorer {name} is not lowercase SHA-256")
    if type(value.frozen_at_utc) is not str or _SECOND_UTC.fullmatch(
        value.frozen_at_utc
    ) is None:
        raise EvaluationContractError("hard scorer freeze time is not exact UTC")
    evaluated = value.evaluated_checkpoint_sha256s
    if (
        type(evaluated) is not tuple
        or len(evaluated) != 27
        or any(type(item) is not str or _LOWER_SHA256.fullmatch(item) is None for item in evaluated)
        or evaluated != tuple(sorted(evaluated))
        or len(evaluated) != len(set(evaluated))
    ):
        raise EvaluationContractError(
            "hard scorer must bind the canonical 27-checkpoint evaluation census"
        )
    if value.scorer_checkpoint_sha256 in evaluated:
        raise EvaluationContractError(
            "hard scorer checkpoint cannot be an evaluated system checkpoint"
        )
    if value.text_only is not True or value.frozen_before_evaluation is not True:
        raise EvaluationContractError(
            "hard-negative scorer must be independently frozen and text-only"
        )
    return (
        json.dumps(
            {
                "caption_manifest_sha256": value.caption_manifest_sha256,
                "evaluated_checkpoint_sha256s": list(evaluated),
                "freeze_receipt_sha256": value.freeze_receipt_sha256,
                "frozen_at_utc": value.frozen_at_utc,
                "frozen_before_evaluation": value.frozen_before_evaluation,
                "motion_manifest_sha256": value.motion_manifest_sha256,
                "schema": "phaseset-hard-negative-scorer-provenance-v1",
                "scorer_checkpoint_sha256": value.scorer_checkpoint_sha256,
                "scorer_code_sha256": value.scorer_code_sha256,
                "scorer_id": value.scorer_id,
                "text_only": value.text_only,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def hard_negative_scorer_provenance_sha256(value: object) -> str:
    return hashlib.sha256(_hard_scorer_provenance_bytes(value)).hexdigest()


def hard_negative_similarity_sha256(value: object) -> str:
    if type(value) is not np.ndarray:
        raise TypeError("similarity must be exact numpy.ndarray")
    if value.dtype != np.float64 or value.ndim != 2 or not np.isfinite(value).all():
        raise EvaluationContractError("similarity must be finite float64 rank-2")
    return _array_sha256(value, "<f8")


def _hard_freeze_binding_bytes(value: object) -> bytes:
    if type(value) is not HardGalleryFreezeBinding:
        raise TypeError("freeze_binding must be exact HardGalleryFreezeBinding")
    for name in (
        "scorer_provenance_sha256",
        "similarity_sha256",
        "freeze_receipt_sha256",
        "caption_manifest_sha256",
        "motion_manifest_sha256",
        "external_authorization_sha256",
    ):
        item = getattr(value, name)
        if type(item) is not str or _LOWER_SHA256.fullmatch(item) is None:
            raise EvaluationContractError(f"hard freeze binding {name} is invalid")
    return (
        json.dumps(
            {
                "caption_manifest_sha256": value.caption_manifest_sha256,
                "external_authorization_sha256": value.external_authorization_sha256,
                "freeze_receipt_sha256": value.freeze_receipt_sha256,
                "motion_manifest_sha256": value.motion_manifest_sha256,
                "schema": "phaseset-hard-gallery-freeze-binding-v1",
                "scorer_provenance_sha256": value.scorer_provenance_sha256,
                "similarity_sha256": value.similarity_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def hard_gallery_freeze_binding_sha256(value: object) -> str:
    return hashlib.sha256(_hard_freeze_binding_bytes(value)).hexdigest()


def _validate_hard_features(
    dataset: RetrievalDataset,
    value: object,
) -> HardNegativeFeatures:
    if type(value) is not HardNegativeFeatures:
        raise TypeError("features must be exact HardNegativeFeatures")
    motion_count, caption_count = dataset.scores.shape
    power = _readonly_array(
        value.per_actor_band_power,
        dtype=np.float64,
        ndim=3,
        label="per_actor_band_power",
    )
    actor_mask = _readonly_array(
        value.actor_mask,
        dtype=np.bool_,
        ndim=2,
        label="actor_mask",
    )
    if (
        power.shape[0] != motion_count
        or power.shape[2] != 6
        or actor_mask.shape != power.shape[:2]
        or power.shape[1] < int(np.max(dataset.group_sizes))
        or not np.isfinite(power).all()
        or np.any(power < 0.0)
    ):
        raise EvaluationContractError(
            "per_actor_band_power must be finite nonnegative [motion,K_pad,6]"
        )
    if not np.array_equal(actor_mask.sum(axis=1, dtype=np.int64), dataset.group_sizes):
        raise EvaluationContractError("actor_mask census must equal each motion group size")
    invalid_power = power[~actor_mask]
    if invalid_power.size and (np.any(invalid_power != 0.0) or np.any(np.signbit(invalid_power))):
        raise EvaluationContractError("invalid actor band power must be exact +0.0")
    similarity = _finite_vector(
        value.frozen_text_similarity,
        (caption_count, motion_count),
        "frozen_text_similarity",
        allow_negative=True,
    )
    evaluated_similarity = dataset.scores.T
    same_full_rank = all(
        tuple(
            sorted(
                range(motion_count),
                key=lambda index: (
                    -float(similarity[caption_index, index]),
                    dataset.motion_commitments[index],
                ),
            )
        )
        == tuple(
            sorted(
                range(motion_count),
                key=lambda index: (
                    -float(evaluated_similarity[caption_index, index]),
                    dataset.motion_commitments[index],
                ),
            )
        )
        for caption_index in range(caption_count)
    )
    if same_full_rank:
        raise EvaluationContractError(
            "hard-negative similarity cannot preserve the evaluated system's full ranking"
        )
    provenance = value.scorer_provenance
    _hard_scorer_provenance_bytes(provenance)
    return HardNegativeFeatures(
        duration_seconds=_finite_vector(
            value.duration_seconds, (motion_count,), "duration_seconds"
        ),
        per_actor_band_power=power,
        actor_mask=actor_mask,
        total_motion_energy=_finite_vector(
            value.total_motion_energy,
            (motion_count,),
            "total_motion_energy",
        ),
        root_speed=_finite_vector(value.root_speed, (motion_count,), "root_speed"),
        frozen_text_similarity=similarity,
        scorer_provenance=provenance,
    )


def _actor_power_signatures(features: HardNegativeFeatures) -> tuple[np.ndarray, ...]:
    """Return permutation-invariant, per-band sorted actor-power multisets."""

    signatures: list[np.ndarray] = []
    for motion_index in range(features.per_actor_band_power.shape[0]):
        valid = features.per_actor_band_power[motion_index, features.actor_mask[motion_index]]
        # Sorting each band independently represents the actor-marginal power
        # distribution without creating an actor-order shortcut.
        signatures.append(np.sort(np.log1p(valid), axis=0))
    return tuple(signatures)


def _validate_hard_config(value: object) -> HardGalleryConfig:
    if type(value) is not HardGalleryConfig:
        raise TypeError("config must be exact HardGalleryConfig")
    if type(value.candidate_count) is not int or value.candidate_count < 2:
        raise EvaluationContractError("candidate_count must be an exact integer >=2")
    for label in (
        "duration_absolute_max",
        "band_power_log_linf_max",
        "total_energy_log_max",
        "root_speed_log_max",
    ):
        item = getattr(value, label)
        if type(item) is not float or not math.isfinite(item) or item < 0.0:
            raise EvaluationContractError(f"{label} must be a finite nonnegative float")
    return value


def _hard_gallery_digest(
    dataset: RetrievalDataset,
    *,
    caption_index: int,
    motion_indices: tuple[int, ...],
    maximum_candidate_count: int,
    selection_binding_sha256: str,
) -> str:
    positives = dataset.positive_motion_indices[caption_index]
    target_k = int(dataset.group_sizes[positives[0]])
    payload = {
        "caption_commitment": dataset.caption_commitments[caption_index].hex(),
        "domain": "phaseset-group-hard-v1",
        "maximum_candidate_count": maximum_candidate_count,
        "motion_commitments": [dataset.motion_commitments[index].hex() for index in motion_indices],
        "positive_motion_commitments": sorted(
            dataset.motion_commitments[index].hex() for index in positives
        ),
        "selection_binding_sha256": selection_binding_sha256,
        "target_k": target_k,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _selection_binding_sha256(
    dataset: RetrievalDataset,
    features: HardNegativeFeatures,
    config: HardGalleryConfig,
    freeze_binding: HardGalleryFreezeBinding,
) -> str:
    scorer_raw = _hard_scorer_provenance_bytes(features.scorer_provenance)
    payload = {
        "actor_power_signature_sha256s": [
            _array_sha256(signature, "<f8")
            for signature in _actor_power_signatures(features)
        ],
        "band_power_log_linf_max_hex": config.band_power_log_linf_max.hex(),
        "candidate_count": config.candidate_count,
        "freeze_binding_sha256": hard_gallery_freeze_binding_sha256(freeze_binding),
        "retrieval_census_sha256": retrieval_census_sha256(dataset),
        "duration_absolute_max_hex": config.duration_absolute_max.hex(),
        "duration_sha256": _array_sha256(features.duration_seconds, "<f8"),
        "motion_energy_sha256": _array_sha256(features.total_motion_energy, "<f8"),
        "root_speed_log_max_hex": config.root_speed_log_max.hex(),
        "root_speed_sha256": _array_sha256(features.root_speed, "<f8"),
        "schema": "phaseset-hard-gallery-selection-binding-v1",
        "scorer_provenance_sha256": hashlib.sha256(scorer_raw).hexdigest(),
        "similarity_sha256": hard_negative_similarity_sha256(
            features.frozen_text_similarity
        ),
        "total_energy_log_max_hex": config.total_energy_log_max.hex(),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def _hard_collection_digest(
    retrieval_census_sha256_value: str,
    scorer_provenance_sha256: str,
    freeze_binding_sha256: str,
    selection_binding_sha256: str,
    galleries: tuple[HardGallery, ...],
) -> str:
    raw = json.dumps(
        {
            "freeze_binding_sha256": freeze_binding_sha256,
            "domain": "phaseset-group-hard-collection-v2",
            "gallery_sha256": [gallery.gallery_sha256 for gallery in galleries],
            "scorer_provenance_sha256": scorer_provenance_sha256,
            "retrieval_census_sha256": retrieval_census_sha256_value,
            "selection_binding_sha256": selection_binding_sha256,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("ascii")
    return hashlib.sha256(raw).hexdigest()


def build_group_hard_galleries(
    dataset_value: object,
    features_value: object,
    config_value: object,
    freeze_binding_value: object,
) -> HardGalleryCollection:
    """Build exact-K, caliper-matched hard galleries without fallback."""

    dataset = validate_retrieval_dataset(dataset_value)
    features = _validate_hard_features(dataset, features_value)
    config = _validate_hard_config(config_value)
    if type(freeze_binding_value) is not HardGalleryFreezeBinding:
        raise TypeError("freeze_binding must be exact HardGalleryFreezeBinding")
    freeze_binding = freeze_binding_value
    _hard_freeze_binding_bytes(freeze_binding)
    scorer_provenance = features.scorer_provenance
    if (
        freeze_binding.scorer_provenance_sha256
        != hard_negative_scorer_provenance_sha256(scorer_provenance)
        or freeze_binding.similarity_sha256
        != hard_negative_similarity_sha256(features.frozen_text_similarity)
        or freeze_binding.freeze_receipt_sha256
        != scorer_provenance.freeze_receipt_sha256
        or freeze_binding.caption_manifest_sha256
        != scorer_provenance.caption_manifest_sha256
        or freeze_binding.motion_manifest_sha256
        != scorer_provenance.motion_manifest_sha256
    ):
        raise EvaluationContractError(
            "hard-gallery inputs differ from the externally frozen expected binding"
        )
    output: list[HardGallery] = []
    selection_binding_sha256 = _selection_binding_sha256(
        dataset,
        features,
        config,
        freeze_binding,
    )
    power_signatures = _actor_power_signatures(features)
    log_energy = np.log1p(features.total_motion_energy)
    log_speed = np.log1p(features.root_speed)
    for caption_index, positives in enumerate(dataset.positive_motion_indices):
        positive_sizes = {int(dataset.group_sizes[index]) for index in positives}
        if len(positive_sizes) != 1:
            raise EvaluationContractError("one hard-gallery query cannot span multiple K values")
        if len(positives) >= config.candidate_count:
            raise EvaluationContractError("positive set fills the entire hard gallery")
        positive_array = np.asarray(positives, dtype=np.int64)
        target_k = next(iter(positive_sizes))
        reference_duration = float(np.mean(features.duration_seconds[positive_array]))
        reference_power = np.mean(
            np.stack([power_signatures[index] for index in positives]),
            axis=0,
        )
        reference_energy = float(np.mean(log_energy[positive_array]))
        reference_speed = float(np.mean(log_speed[positive_array]))
        candidates: list[tuple[float, float, bytes, int]] = []
        for motion_index in range(dataset.scores.shape[0]):
            if motion_index in positives or int(dataset.group_sizes[motion_index]) != target_k:
                continue
            duration_delta = abs(
                float(features.duration_seconds[motion_index]) - reference_duration
            )
            power_delta = float(np.max(np.abs(power_signatures[motion_index] - reference_power)))
            energy_delta = abs(float(log_energy[motion_index]) - reference_energy)
            speed_delta = abs(float(log_speed[motion_index]) - reference_speed)
            if (
                duration_delta > config.duration_absolute_max
                or power_delta > config.band_power_log_linf_max
                or energy_delta > config.total_energy_log_max
                or speed_delta > config.root_speed_log_max
            ):
                continue
            feature_distance = duration_delta + power_delta + energy_delta + speed_delta
            candidates.append(
                (
                    -float(features.frozen_text_similarity[caption_index, motion_index]),
                    feature_distance,
                    dataset.motion_commitments[motion_index],
                    motion_index,
                )
            )
        maximum_negatives = config.candidate_count - len(positives)
        if maximum_negatives <= 0:
            raise EvaluationContractError(
                f"positive set fills the Group-Hard-{config.candidate_count} cap "
                f"for caption={caption_index}"
            )
        if not candidates:
            raise EvaluationContractError(
                f"HARD_GALLERY_NO_MATCHED_NEGATIVE caption={caption_index}"
            )
        negative_count = min(maximum_negatives, len(candidates))
        negatives = tuple(item[3] for item in sorted(candidates)[:negative_count])
        selected = tuple(
            sorted(
                (*positives, *negatives),
                key=lambda index: dataset.motion_commitments[index],
            )
        )
        output.append(
            HardGallery(
                caption_index=caption_index,
                motion_indices=selected,
                maximum_candidate_count=config.candidate_count,
                gallery_sha256=_hard_gallery_digest(
                    dataset,
                    caption_index=caption_index,
                    motion_indices=selected,
                    maximum_candidate_count=config.candidate_count,
                    selection_binding_sha256=selection_binding_sha256,
                ),
            )
        )
    galleries = tuple(output)
    scorer_provenance_sha256 = hashlib.sha256(
        _hard_scorer_provenance_bytes(features.scorer_provenance)
    ).hexdigest()
    census_sha256 = retrieval_census_sha256(dataset)
    freeze_binding_sha256 = hard_gallery_freeze_binding_sha256(freeze_binding)
    return HardGalleryCollection(
        galleries=galleries,
        scorer_provenance=features.scorer_provenance,
        freeze_binding=freeze_binding,
        selection_binding_sha256=selection_binding_sha256,
        retrieval_census_sha256=census_sha256,
        collection_sha256=_hard_collection_digest(
            census_sha256,
            scorer_provenance_sha256,
            freeze_binding_sha256,
            selection_binding_sha256,
            galleries,
        ),
    )


def evaluate_hard_text_to_motion(
    dataset_value: object,
    collection_value: object,
) -> DirectionMetrics:
    dataset = validate_retrieval_dataset(dataset_value)
    galleries = _validated_hard_collection(dataset, collection_value).galleries
    ranks: list[int] = []
    for caption_index, gallery in enumerate(galleries):
        positives = set(dataset.positive_motion_indices[caption_index])
        ordered = sorted(
            gallery.motion_indices,
            key=lambda index: (
                -float(dataset.scores[index, caption_index]),
                dataset.motion_commitments[index],
            ),
        )
        ranks.append(min(ordered.index(index) + 1 for index in positives))
    clusters = _capture_clusters(dataset)
    return _metrics(
        np.asarray(ranks, dtype=np.int64),
        tuple(cluster.caption_indices for cluster in clusters),
    )


def _validated_hard_galleries(
    dataset: RetrievalDataset,
    value: object,
    *,
    selection_binding_sha256: str,
) -> tuple[HardGallery, ...]:
    if type(value) is not tuple or len(value) != dataset.scores.shape[1]:
        raise EvaluationContractError("hard galleries must contain one row per caption")
    maximum_candidate_count: int | None = None
    for caption_index, gallery in enumerate(value):
        if type(gallery) is not HardGallery or gallery.caption_index != caption_index:
            raise EvaluationContractError("hard gallery identity/order is invalid")
        candidates = gallery.motion_indices
        if (
            type(candidates) is not tuple
            or any(type(index) is not int for index in candidates)
            or len(candidates) != len(set(candidates))
            or len(candidates) < 2
        ):
            raise EvaluationContractError(
                "hard gallery candidates must be exact, unique, and include a negative"
            )
        if min(candidates) < 0 or max(candidates) >= dataset.scores.shape[0]:
            raise EvaluationContractError("hard gallery candidate index is out of range")
        if (
            type(gallery.maximum_candidate_count) is not int
            or gallery.maximum_candidate_count < 2
            or len(candidates) > gallery.maximum_candidate_count
        ):
            raise EvaluationContractError("hard gallery maximum candidate count is invalid")
        if maximum_candidate_count is None:
            maximum_candidate_count = gallery.maximum_candidate_count
        elif gallery.maximum_candidate_count != maximum_candidate_count:
            raise EvaluationContractError("hard gallery maximum must be frozen across queries")
        positives = set(dataset.positive_motion_indices[caption_index])
        if not positives.issubset(candidates):
            raise EvaluationContractError("hard gallery omitted a positive motion")
        target_k = int(dataset.group_sizes[next(iter(positives))])
        if any(int(dataset.group_sizes[index]) != target_k for index in candidates):
            raise EvaluationContractError("hard gallery violates strict K matching")
        canonical = tuple(
            sorted(
                candidates,
                key=lambda index: dataset.motion_commitments[index],
            )
        )
        if candidates != canonical:
            raise EvaluationContractError("hard gallery candidate order is not canonical")
        expected_digest = _hard_gallery_digest(
            dataset,
            caption_index=caption_index,
            motion_indices=candidates,
            maximum_candidate_count=gallery.maximum_candidate_count,
            selection_binding_sha256=selection_binding_sha256,
        )
        if type(gallery.gallery_sha256) is not str or gallery.gallery_sha256 != expected_digest:
            raise EvaluationContractError("hard gallery digest does not match its lineage")
    return value


def _validated_hard_collection(
    dataset: RetrievalDataset,
    value: object,
) -> HardGalleryCollection:
    if type(value) is not HardGalleryCollection:
        raise TypeError("hard gallery collection must be exact HardGalleryCollection")
    census_sha256 = retrieval_census_sha256(dataset)
    if value.retrieval_census_sha256 != census_sha256:
        raise EvaluationContractError("hard gallery collection query census differs")
    if (
        type(value.selection_binding_sha256) is not str
        or _LOWER_SHA256.fullmatch(value.selection_binding_sha256) is None
    ):
        raise EvaluationContractError("hard gallery selection binding is invalid")
    scorer_raw = _hard_scorer_provenance_bytes(value.scorer_provenance)
    scorer_provenance_sha256 = hashlib.sha256(scorer_raw).hexdigest()
    freeze_binding_raw = _hard_freeze_binding_bytes(value.freeze_binding)
    freeze_binding_sha256 = hashlib.sha256(freeze_binding_raw).hexdigest()
    if (
        value.freeze_binding.scorer_provenance_sha256 != scorer_provenance_sha256
        or value.freeze_binding.freeze_receipt_sha256
        != value.scorer_provenance.freeze_receipt_sha256
        or value.freeze_binding.caption_manifest_sha256
        != value.scorer_provenance.caption_manifest_sha256
        or value.freeze_binding.motion_manifest_sha256
        != value.scorer_provenance.motion_manifest_sha256
    ):
        raise EvaluationContractError(
            "hard gallery collection freeze binding differs from scorer provenance"
        )
    galleries = _validated_hard_galleries(
        dataset,
        value.galleries,
        selection_binding_sha256=value.selection_binding_sha256,
    )
    expected_collection_sha256 = _hard_collection_digest(
        census_sha256,
        scorer_provenance_sha256,
        freeze_binding_sha256,
        value.selection_binding_sha256,
        galleries,
    )
    if value.collection_sha256 != expected_collection_sha256:
        raise EvaluationContractError("hard gallery collection digest does not match lineage")
    return value


def hard_gallery_collection_sha256(
    dataset_value: object,
    collection_value: object,
) -> str:
    """Return a public-safe digest without exposing indices or commitments."""

    dataset = validate_retrieval_dataset(dataset_value)
    collection = _validated_hard_collection(dataset, collection_value)
    return collection.collection_sha256


def count_only_chance_recall_at_1(
    dataset_value: object,
    collection_value: object,
) -> float:
    """Analytical random-tie R@1 for a K-matched count-only diagnostic."""

    dataset = validate_retrieval_dataset(dataset_value)
    galleries = _validated_hard_collection(dataset, collection_value).galleries
    probabilities = [
        len(dataset.positive_motion_indices[caption_index]) / len(gallery.motion_indices)
        for caption_index, gallery in enumerate(galleries)
    ]
    return float(np.mean(probabilities))


def count_only_scores(dataset_value: object) -> np.ndarray:
    """Return the deterministic K-only diagnostic score matrix."""

    dataset = validate_retrieval_dataset(dataset_value)
    result = np.empty_like(dataset.scores)
    for caption_index, positives in enumerate(dataset.positive_motion_indices):
        sizes = {int(dataset.group_sizes[index]) for index in positives}
        if len(sizes) != 1:
            raise EvaluationContractError("count-only query cannot span multiple K values")
        result[:, caption_index] = dataset.group_sizes == next(iter(sizes))
    result.setflags(write=False)
    return result


def assert_count_only_non_discriminative(
    dataset_value: object,
    collection_value: object,
) -> None:
    dataset = validate_retrieval_dataset(dataset_value)
    scores = count_only_scores(dataset)
    frozen_galleries = _validated_hard_collection(
        dataset,
        collection_value,
    ).galleries
    for gallery in frozen_galleries:
        values = scores[list(gallery.motion_indices), gallery.caption_index]
        if not np.all(values == values[0]):
            raise EvaluationContractError("count-only diagnostic discriminates a hard gallery")


__all__ = [
    "BidirectionalMetrics",
    "CaptureR1Contribution",
    "DirectionMetrics",
    "EvaluationContractError",
    "EvaluationReport",
    "HardGallery",
    "HardGalleryCollection",
    "HardGalleryConfig",
    "HardGalleryFreezeBinding",
    "HardNegativeFeatures",
    "HardNegativeScorerProvenance",
    "RetrievalDataset",
    "assert_count_only_non_discriminative",
    "build_group_hard_galleries",
    "canonical_evaluation_report_bytes",
    "capture_r1_contributions",
    "capture_cluster_commitments",
    "count_only_chance_recall_at_1",
    "count_only_scores",
    "evaluate_hard_text_to_motion",
    "evaluate_retrieval",
    "hard_gallery_collection_sha256",
    "hard_gallery_freeze_binding_sha256",
    "hard_negative_scorer_provenance_sha256",
    "hard_negative_similarity_sha256",
    "retrieval_census_sha256",
    "validate_retrieval_dataset",
]
