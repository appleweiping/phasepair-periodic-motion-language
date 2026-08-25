"""Frozen PhasePair validation-gallery construction and base selection.

The module snapshots already-produced AB/BA motion embeddings and three
caption embeddings per source, constructs the complete ``C x 3C`` score
matrix, delegates ranking to the strict full-gallery evaluator, and selects a
base checkpoint only from completed epochs 13--30.  All public artifacts are
Authority-0 receipts: this module neither reads project data nor authorizes or
claims an experiment result.
"""

from __future__ import annotations

import hashlib
import json
import math
import struct
import threading
from typing import Sequence

import numpy as np

from phasepair_core._identity_registry import make_identity_weak_registry
from phasepair_core.evaluation import (
    CAPTIONS_PER_SOURCE,
    EvaluationError,
    evaluate_full_gallery,
)


AUTHORITY = 0
PRODUCTION = False
EXECUTION_AUTHORIZED = False
RESULT_CLAIMED = False
STATUS = "DATA_FREE_VALIDATION_GALLERY_AUTHORITY0_NO_RESULT"
BASE_COMPLETED_EPOCH_ORDER = tuple(range(13, 31))
BASE_EPOCH_INDEX_ORDER = tuple(range(12, 30))
BASE_SYSTEMS = ("00", "07", "08")
SEEDS = (1729, 2718, 31415)
SCORE_SWAP_TOLERANCE = 1e-6
SELECTION_TIE_TOLERANCE = 1e-12


class ValidationGalleryError(ValueError):
    """A gallery snapshot or checkpoint selection violated the contract."""


class ValidationGallerySnapshot:
    """Opaque, module-issued receipt for one completed-epoch full gallery."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise ValidationGalleryError(
            "validation snapshots are minted only by build_validation_gallery_snapshot"
        )


class BaseValidationSelection:
    """Opaque, module-issued receipt selecting one base checkpoint."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise ValidationGalleryError(
            "base selections are minted only by select_base_validation_checkpoint"
        )


def _exact_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    return value


def _lower_sha256(value: object, label: str) -> str:
    checked = _exact_text(value, label)
    if len(checked) != 64 or any(
        character not in "0123456789abcdef" for character in checked
    ):
        raise ValidationGalleryError(f"{label} must be lowercase SHA-256 hex")
    return checked


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < minimum or value > maximum:
        raise ValidationGalleryError(
            f"{label} must be inside the closed interval [{minimum},{maximum}]"
        )
    return value


def _parse_base_run_id(run_id: object) -> tuple[str, int, str]:
    checked = _exact_text(run_id, "run_id")
    parts = checked.split("/")
    if len(parts) != 4 or parts[0] != "phasepair-run-v2":
        raise ValidationGalleryError("TRAINING_EXECUTION_IDENTITY_VERSION_FAIL")
    if parts[1] != "BASE_TRAIN":
        raise ValidationGalleryError("run_id must identify a BASE_TRAIN run")
    if parts[2] not in {str(seed) for seed in SEEDS}:
        raise ValidationGalleryError("run_id seed is outside the frozen census")
    if parts[3] not in BASE_SYSTEMS:
        raise ValidationGalleryError("run_id system is outside the base census")
    return checked, int(parts[2]), parts[3]


def _raw32_sequence(name: str, values: Sequence[bytes]) -> tuple[bytes, ...]:
    if isinstance(values, (bytes, bytearray, memoryview, str)):
        raise ValidationGalleryError(f"{name} must be a sequence of raw 32-byte values")
    try:
        normalized = tuple(values)
    except TypeError as exc:
        raise ValidationGalleryError(f"{name} must be a finite sequence") from exc
    for ordinal, value in enumerate(normalized):
        if type(value) is not bytes or len(value) != 32:
            raise ValidationGalleryError(f"{name}[{ordinal}] must be exactly bytes[32]")
    return normalized


def _frozen_float32_matrix(value: object, label: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray, not a subclass")
    if value.dtype != np.dtype(np.float32):
        raise ValidationGalleryError(f"{label} must have exact dtype float32")
    if value.ndim != 2:
        raise ValidationGalleryError(f"{label} must be two-dimensional")
    if not value.flags.c_contiguous:
        raise ValidationGalleryError(f"{label} must be C-contiguous")
    copied = np.array(value, dtype=np.float32, copy=True, order="C", subok=False)
    if copied.shape[0] == 0 or copied.shape[1] == 0:
        raise ValidationGalleryError(f"{label} must have nonzero rows and columns")
    if not bool(np.isfinite(copied).all()):
        raise ValidationGalleryError(f"{label} must contain only finite values")
    readonly = np.frombuffer(copied.tobytes(order="C"), dtype=np.float32).reshape(
        copied.shape
    )
    return readonly


def _require_l2_rows(value: np.ndarray, label: str) -> None:
    norms = np.linalg.norm(value.astype(np.float64), ord=2, axis=1)
    if not bool(np.isfinite(norms).all()) or bool(
        np.any(np.abs(norms - 1.0) > 1e-5)
    ):
        raise ValidationGalleryError(
            f"{label} rows must be finite unit-L2 embeddings within 1e-5"
        )


def _array_sha256(tag: bytes, value: np.ndarray) -> str:
    canonical = value.astype("<f4", copy=False).tobytes(order="C")
    header = struct.pack(">II", value.shape[0], value.shape[1])
    return hashlib.sha256(tag + b"\x00" + header + canonical).hexdigest()


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _identity_sha256(tag: bytes, rows: tuple[bytes, ...]) -> str:
    return hashlib.sha256(tag + b"\x00" + b"".join(rows)).hexdigest()


def _metric_payload(
    *,
    score_matrix_sha256: str,
    gallery_manifest_sha256: str,
    motion_pair_commitments_sha256: str,
    caption_commitments_sha256: str,
    evaluation: object,
) -> dict[str, object]:
    t2m = evaluation.t2m
    m2t = evaluation.m2t
    return {
        "bidirectional_mean_r_at_1_hex": evaluation.bidirectional_mean_r_at_1.hex(),
        "caption_commitments_sha256": caption_commitments_sha256,
        "gallery_manifest_sha256": gallery_manifest_sha256,
        "m2t_any_caption_r_at_1_hex": evaluation.m2t_any_caption_r_at_1.hex(),
        "m2t_median_rank_hex": m2t.median_rank.hex(),
        "m2t_r_at_10_hex": m2t.r_at_10.hex(),
        "m2t_r_at_1_hex": m2t.r_at_1.hex(),
        "m2t_r_at_3_hex": m2t.r_at_3.hex(),
        "m2t_r_at_5_hex": m2t.r_at_5.hex(),
        "m2t_ranks": [int(value) for value in m2t.ranks],
        "motion_pair_commitments_sha256": motion_pair_commitments_sha256,
        "schema": "phasepair-validation-metric-v1",
        "score_matrix_sha256": score_matrix_sha256,
        "t2m_cluster_macro_r_at_1_hex": evaluation.t2m_cluster_macro_r_at_1.hex(),
        "t2m_median_rank_hex": t2m.median_rank.hex(),
        "t2m_r_at_10_hex": t2m.r_at_10.hex(),
        "t2m_r_at_1_hex": t2m.r_at_1.hex(),
        "t2m_r_at_3_hex": t2m.r_at_3.hex(),
        "t2m_r_at_5_hex": t2m.r_at_5.hex(),
        "t2m_ranks": [int(value) for value in t2m.ranks],
    }


_snapshot_lock = threading.RLock()
_snapshot_set, _snapshot_get, _, _, _ = make_identity_weak_registry(_snapshot_lock)
_selection_lock = threading.RLock()
_selection_set, _selection_get, _, _, _ = make_identity_weak_registry(_selection_lock)
_object_new = object.__new__


def build_validation_gallery_snapshot(
    *,
    run_id: str,
    run_input_sha256: str,
    completed_epoch: int,
    checkpoint_sha256: str,
    evaluator_sha256: str,
    gallery_manifest_sha256: str,
    motion_ab: np.ndarray,
    motion_ba: np.ndarray,
    captions: np.ndarray,
    motion_source_cluster_ids: Sequence[bytes],
    caption_source_cluster_ids: Sequence[bytes],
    motion_pair_commitments: Sequence[bytes],
    caption_commitments: Sequence[bytes],
) -> ValidationGallerySnapshot:
    """Snapshot one complete base-validation gallery without filesystem I/O."""

    checked_run_id, seed, system_id = _parse_base_run_id(run_id)
    epoch = _exact_int(completed_epoch, "completed_epoch", minimum=13, maximum=30)
    run_input = _lower_sha256(run_input_sha256, "run_input_sha256")
    checkpoint = _lower_sha256(checkpoint_sha256, "checkpoint_sha256")
    evaluator = _lower_sha256(evaluator_sha256, "evaluator_sha256")
    gallery_manifest = _lower_sha256(
        gallery_manifest_sha256, "gallery_manifest_sha256"
    )

    frozen_ab = _frozen_float32_matrix(motion_ab, "motion_ab")
    frozen_ba = _frozen_float32_matrix(motion_ba, "motion_ba")
    frozen_captions = _frozen_float32_matrix(captions, "captions")
    if frozen_ab.shape != frozen_ba.shape:
        raise ValidationGalleryError("motion_ab and motion_ba shapes must be identical")
    cluster_count, vector_dim = frozen_ab.shape
    if frozen_captions.shape != (CAPTIONS_PER_SOURCE * cluster_count, vector_dim):
        raise ValidationGalleryError(
            "captions shape must be exactly [3*C,D] for motion shape [C,D]"
        )
    _require_l2_rows(frozen_ab, "motion_ab")
    _require_l2_rows(frozen_ba, "motion_ba")
    _require_l2_rows(frozen_captions, "captions")

    motion_sources = _raw32_sequence(
        "motion_source_cluster_ids", motion_source_cluster_ids
    )
    caption_sources = _raw32_sequence(
        "caption_source_cluster_ids", caption_source_cluster_ids
    )
    motion_ties = _raw32_sequence(
        "motion_pair_commitments", motion_pair_commitments
    )
    caption_ties = _raw32_sequence("caption_commitments", caption_commitments)
    if len(motion_sources) != cluster_count or len(motion_ties) != cluster_count:
        raise ValidationGalleryError("motion identity rows must equal C")
    if (
        len(caption_sources) != CAPTIONS_PER_SOURCE * cluster_count
        or len(caption_ties) != CAPTIONS_PER_SOURCE * cluster_count
    ):
        raise ValidationGalleryError("caption identity rows must equal 3*C")

    score_ab = np.matmul(frozen_ab, frozen_captions.T)
    score_ba = np.matmul(frozen_ba, frozen_captions.T)
    scores = np.ascontiguousarray(
        (score_ab + score_ba) * np.float32(0.5), dtype=np.float32
    )
    swapped = np.ascontiguousarray(
        (score_ba + score_ab) * np.float32(0.5), dtype=np.float32
    )
    swap_error = float(
        np.max(np.abs(scores.astype(np.float64) - swapped.astype(np.float64)))
    )
    if not math.isfinite(swap_error) or swap_error > SCORE_SWAP_TOLERANCE:
        raise ValidationGalleryError("AB/BA score swap error exceeds 1e-6")

    positive_mask = np.fromiter(
        (
            motion_source == caption_source
            for motion_source in motion_sources
            for caption_source in caption_sources
        ),
        dtype=np.bool_,
        count=cluster_count * CAPTIONS_PER_SOURCE * cluster_count,
    ).reshape(scores.shape)
    positive_mask = np.ascontiguousarray(positive_mask, dtype=np.bool_)
    try:
        evaluation = evaluate_full_gallery(
            scores,
            positive_mask=positive_mask,
            motion_source_cluster_ids=motion_sources,
            caption_source_cluster_ids=caption_sources,
            motion_pair_commitments=motion_ties,
            caption_commitments=caption_ties,
        )
    except EvaluationError as exc:
        raise ValidationGalleryError(str(exc)) from exc

    motion_ab_sha256 = _array_sha256(b"phasepair-motion-ab-f32-v1", frozen_ab)
    motion_ba_sha256 = _array_sha256(b"phasepair-motion-ba-f32-v1", frozen_ba)
    captions_sha256 = _array_sha256(
        b"phasepair-caption-embeddings-f32-v1", frozen_captions
    )
    score_matrix_sha256 = _array_sha256(
        b"phasepair-full-gallery-score-f32-v1", scores
    )
    motion_ties_sha256 = _identity_sha256(
        b"phasepair-motion-pair-order-v1", motion_ties
    )
    caption_ties_sha256 = _identity_sha256(
        b"phasepair-caption-order-v1", caption_ties
    )
    metric_raw = _canonical_json_bytes(
        _metric_payload(
            score_matrix_sha256=score_matrix_sha256,
            gallery_manifest_sha256=gallery_manifest,
            motion_pair_commitments_sha256=motion_ties_sha256,
            caption_commitments_sha256=caption_ties_sha256,
            evaluation=evaluation,
        )
    )
    metric_sha256 = hashlib.sha256(
        b"phasepair-validation-metric-v1\x00" + metric_raw
    ).hexdigest()
    payload = {
        "authority": AUTHORITY,
        "caption_count": CAPTIONS_PER_SOURCE * cluster_count,
        "caption_embeddings_sha256": captions_sha256,
        "checkpoint_sha256": checkpoint,
        "cluster_count": cluster_count,
        "completed_epoch": epoch,
        "epoch_index": epoch - 1,
        "evaluator_sha256": evaluator,
        "execution_authorized": EXECUTION_AUTHORIZED,
        "gallery_manifest_sha256": gallery_manifest,
        "metric_sha256": metric_sha256,
        "motion_ab_sha256": motion_ab_sha256,
        "motion_ba_sha256": motion_ba_sha256,
        "production": PRODUCTION,
        "result_claimed": RESULT_CLAIMED,
        "run_id": checked_run_id,
        "run_input_sha256": run_input,
        "schema": "phasepair-validation-gallery-snapshot-v1",
        "score_formula": "0.5*(AB@TEXT_T+BA@TEXT_T)",
        "score_matrix_sha256": score_matrix_sha256,
        "score_shape": [cluster_count, CAPTIONS_PER_SOURCE * cluster_count],
        "score_swap_max_abs_error_hex": swap_error.hex(),
        "seed": seed,
        "status": STATUS,
        "system_id": system_id,
        "t2m_cluster_macro_r_at_1_hex": evaluation.t2m_cluster_macro_r_at_1.hex(),
        "m2t_any_caption_r_at_1_hex": evaluation.m2t_any_caption_r_at_1.hex(),
        "bidirectional_mean_r_at_1_hex": evaluation.bidirectional_mean_r_at_1.hex(),
        "vector_dim": vector_dim,
    }
    raw = _canonical_json_bytes(payload)
    record = (
        raw,
        checked_run_id,
        run_input,
        seed,
        system_id,
        epoch,
        checkpoint,
        evaluator,
        gallery_manifest,
        score_matrix_sha256,
        metric_sha256,
        evaluation.bidirectional_mean_r_at_1,
    )
    snapshot = _object_new(ValidationGallerySnapshot)
    with _snapshot_lock:
        _snapshot_set(snapshot, record)
    return snapshot


def _snapshot_record(value: object) -> tuple[object, ...]:
    if type(value) is not ValidationGallerySnapshot:
        raise TypeError("snapshot must be exactly ValidationGallerySnapshot")
    with _snapshot_lock:
        record = _snapshot_get(value)
    if type(record) is not tuple or len(record) != 12:
        raise ValidationGalleryError("snapshot was not issued by this module")
    return record


def validation_gallery_snapshot_bytes(value: ValidationGallerySnapshot) -> bytes:
    """Return the immutable canonical Authority-0 snapshot receipt."""

    raw = _snapshot_record(value)[0]
    if type(raw) is not bytes:
        raise AssertionError("internal snapshot bytes are malformed")
    return raw


def validation_gallery_snapshot_sha256(value: ValidationGallerySnapshot) -> str:
    """Return the lowercase SHA-256 identity of one snapshot receipt."""

    return hashlib.sha256(validation_gallery_snapshot_bytes(value)).hexdigest()


def select_base_validation_checkpoint(
    snapshots: object,
) -> BaseValidationSelection:
    """Select the best epoch 13--30 metric; ties within 1e-12 choose earlier."""

    if type(snapshots) is not tuple:
        raise TypeError("snapshots must be an exact built-in tuple")
    if len(snapshots) != len(BASE_COMPLETED_EPOCH_ORDER):
        raise ValidationGalleryError("base selection requires exactly 18 snapshots")
    records = tuple(_snapshot_record(snapshot) for snapshot in snapshots)
    by_epoch: dict[int, tuple[object, ...]] = {}
    for record in records:
        epoch = record[5]
        if type(epoch) is not int:
            raise AssertionError("internal snapshot epoch is malformed")
        if epoch in by_epoch:
            raise ValidationGalleryError(f"duplicate completed epoch: {epoch}")
        by_epoch[epoch] = record
    if tuple(sorted(by_epoch)) != BASE_COMPLETED_EPOCH_ORDER:
        raise ValidationGalleryError("snapshots must cover every completed epoch 13--30")
    ordered = tuple(by_epoch[epoch] for epoch in BASE_COMPLETED_EPOCH_ORDER)

    binding_columns = (1, 2, 3, 4, 7, 8)
    for column in binding_columns:
        if len({record[column] for record in ordered}) != 1:
            raise ValidationGalleryError(
                "all selection snapshots must bind the same run/input/evaluator/gallery"
            )
    checkpoints = tuple(record[6] for record in ordered)
    if len(set(checkpoints)) != len(checkpoints):
        raise ValidationGalleryError("each completed epoch must bind a distinct checkpoint")

    best = ordered[0]
    best_metric = float(best[11])
    for candidate in ordered[1:]:
        candidate_metric = float(candidate[11])
        if candidate_metric > best_metric + SELECTION_TIE_TOLERANCE:
            best = candidate
            best_metric = candidate_metric

    candidate_rows = [
        {
            "bidirectional_mean_r_at_1_hex": float(record[11]).hex(),
            "checkpoint_sha256": record[6],
            "completed_epoch": record[5],
            "epoch_index": int(record[5]) - 1,
            "metric_sha256": record[10],
            "score_matrix_sha256": record[9],
            "snapshot_sha256": hashlib.sha256(record[0]).hexdigest(),
        }
        for record in ordered
    ]
    payload = {
        "authority": AUTHORITY,
        "candidate_count": len(candidate_rows),
        "candidates": candidate_rows,
        "completed_epoch_order": list(BASE_COMPLETED_EPOCH_ORDER),
        "evaluator_sha256": best[7],
        "execution_authorized": EXECUTION_AUTHORIZED,
        "gallery_manifest_sha256": best[8],
        "production": PRODUCTION,
        "result_claimed": RESULT_CLAIMED,
        "run_id": best[1],
        "run_input_sha256": best[2],
        "schema": "phasepair-base-validation-selection-v1",
        "selected_bidirectional_mean_r_at_1_hex": best_metric.hex(),
        "selected_checkpoint_sha256": best[6],
        "selected_completed_epoch": best[5],
        "selected_epoch_index": int(best[5]) - 1,
        "selected_metric_sha256": best[10],
        "selected_score_matrix_sha256": best[9],
        "seed": best[3],
        "selection_endpoint": "validation_full_gallery_cluster_macro_bidirectional_mean_R1",
        "status": STATUS,
        "system_id": best[4],
        "tie_policy": "metric_delta_le_1e-12_choose_earlier_completed_epoch",
    }
    raw = _canonical_json_bytes(payload)
    selection = _object_new(BaseValidationSelection)
    with _selection_lock:
        _selection_set(selection, raw)
    return selection


def base_validation_selection_bytes(value: BaseValidationSelection) -> bytes:
    """Return the immutable canonical Authority-0 selection receipt."""

    if type(value) is not BaseValidationSelection:
        raise TypeError("selection must be exactly BaseValidationSelection")
    with _selection_lock:
        raw = _selection_get(value)
    if type(raw) is not bytes:
        raise ValidationGalleryError("selection was not issued by this module")
    return raw


def base_validation_selection_sha256(value: BaseValidationSelection) -> str:
    """Return the lowercase SHA-256 identity of one selection receipt."""

    return hashlib.sha256(base_validation_selection_bytes(value)).hexdigest()


__all__ = [
    "AUTHORITY",
    "BASE_COMPLETED_EPOCH_ORDER",
    "BASE_EPOCH_INDEX_ORDER",
    "BASE_SYSTEMS",
    "BaseValidationSelection",
    "EXECUTION_AUTHORIZED",
    "PRODUCTION",
    "RESULT_CLAIMED",
    "SCORE_SWAP_TOLERANCE",
    "SEEDS",
    "SELECTION_TIE_TOLERANCE",
    "STATUS",
    "ValidationGalleryError",
    "ValidationGallerySnapshot",
    "base_validation_selection_bytes",
    "base_validation_selection_sha256",
    "build_validation_gallery_snapshot",
    "select_base_validation_checkpoint",
    "validation_gallery_snapshot_bytes",
    "validation_gallery_snapshot_sha256",
]
