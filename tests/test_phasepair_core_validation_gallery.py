"""Synthetic-only tests for frozen PhasePair validation galleries."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from phasepair_core import validation_gallery as gallery


def _raw32(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _fixture(*, good: bool = True) -> dict[str, object]:
    motion_sources = (_raw32(100), _raw32(200))
    motion = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    first = [1.0, 0.0] if good else [0.0, 1.0]
    second = [0.0, 1.0] if good else [1.0, 0.0]
    captions = np.array([first] * 3 + [second] * 3, dtype=np.float32)
    return {
        "run_id": "phasepair-run-v2/BASE_TRAIN/1729/00",
        "run_input_sha256": _digest("run-input"),
        "completed_epoch": 13,
        "checkpoint_sha256": _digest("checkpoint/13"),
        "evaluator_sha256": _digest("evaluator"),
        "gallery_manifest_sha256": _digest("gallery"),
        "motion_ab": motion.copy(),
        "motion_ba": motion.copy(),
        "captions": captions,
        "motion_source_cluster_ids": motion_sources,
        "caption_source_cluster_ids": (motion_sources[0],) * 3
        + (motion_sources[1],) * 3,
        "motion_pair_commitments": (_raw32(1), _raw32(2)),
        "caption_commitments": tuple(_raw32(10 + index) for index in range(6)),
    }


def _snapshot(*, epoch: int = 13, good: bool = True, system: str = "00"):
    fixture = _fixture(good=good)
    fixture["completed_epoch"] = epoch
    fixture["checkpoint_sha256"] = _digest(f"checkpoint/{epoch}/{system}")
    fixture["run_id"] = f"phasepair-run-v2/BASE_TRAIN/1729/{system}"
    return gallery.build_validation_gallery_snapshot(**fixture)


def _selection_snapshots() -> tuple[gallery.ValidationGallerySnapshot, ...]:
    return tuple(
        _snapshot(epoch=epoch, good=epoch >= 14)
        for epoch in gallery.BASE_COMPLETED_EPOCH_ORDER
    )


def test_snapshot_constructs_complete_ab_ba_three_caption_gallery() -> None:
    snapshot = _snapshot()
    raw = gallery.validation_gallery_snapshot_bytes(snapshot)
    payload = json.loads(raw)

    assert raw.endswith(b"\n") and b"\r" not in raw
    assert payload["schema"] == "phasepair-validation-gallery-snapshot-v1"
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["execution_authorized"] is False
    assert payload["result_claimed"] is False
    assert payload["status"] == gallery.STATUS
    assert payload["cluster_count"] == 2
    assert payload["caption_count"] == 6
    assert payload["score_shape"] == [2, 6]
    assert payload["vector_dim"] == 2
    assert payload["score_formula"] == "0.5*(AB@TEXT_T+BA@TEXT_T)"
    assert float.fromhex(payload["score_swap_max_abs_error_hex"]) == 0.0
    assert float.fromhex(payload["t2m_cluster_macro_r_at_1_hex"]) == 1.0
    assert float.fromhex(payload["m2t_any_caption_r_at_1_hex"]) == 1.0
    assert float.fromhex(payload["bidirectional_mean_r_at_1_hex"]) == 1.0
    assert payload["completed_epoch"] == 13
    assert payload["epoch_index"] == 12
    assert gallery.validation_gallery_snapshot_sha256(snapshot) == hashlib.sha256(
        raw
    ).hexdigest()
    assert raw == (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("ascii")
        + b"\n"
    )


def test_snapshot_is_deterministic_and_immune_to_caller_array_mutation() -> None:
    fixture = _fixture()
    first = gallery.build_validation_gallery_snapshot(**fixture)
    first_raw = gallery.validation_gallery_snapshot_bytes(first)

    fixture["motion_ab"][:] = 0
    fixture["motion_ba"][:] = np.nan
    fixture["captions"][:] = -1
    assert gallery.validation_gallery_snapshot_bytes(first) == first_raw

    second = gallery.build_validation_gallery_snapshot(**_fixture())
    assert gallery.validation_gallery_snapshot_bytes(second) == first_raw


def test_base_selection_requires_all_epochs_and_chooses_earliest_tied_best() -> None:
    selection = gallery.select_base_validation_checkpoint(_selection_snapshots())
    raw = gallery.base_validation_selection_bytes(selection)
    payload = json.loads(raw)

    assert payload["schema"] == "phasepair-base-validation-selection-v1"
    assert payload["candidate_count"] == 18
    assert payload["completed_epoch_order"] == list(range(13, 31))
    assert [row["completed_epoch"] for row in payload["candidates"]] == list(
        range(13, 31)
    )
    assert payload["selected_completed_epoch"] == 14
    assert payload["selected_epoch_index"] == 13
    assert payload["selected_checkpoint_sha256"] == _digest("checkpoint/14/00")
    assert float.fromhex(payload["selected_bidirectional_mean_r_at_1_hex"]) == 1.0
    assert payload["tie_policy"] == (
        "metric_delta_le_1e-12_choose_earlier_completed_epoch"
    )
    assert payload["selection_endpoint"] == (
        "validation_full_gallery_cluster_macro_bidirectional_mean_R1"
    )
    assert payload["authority"] == 0
    assert payload["result_claimed"] is False
    assert gallery.base_validation_selection_sha256(selection) == hashlib.sha256(
        raw
    ).hexdigest()


def test_selection_canonicalizes_input_order_but_rejects_epoch_gaps_or_duplicates() -> None:
    snapshots = _selection_snapshots()
    forward = gallery.select_base_validation_checkpoint(snapshots)
    reverse = gallery.select_base_validation_checkpoint(tuple(reversed(snapshots)))
    assert gallery.base_validation_selection_bytes(forward) == (
        gallery.base_validation_selection_bytes(reverse)
    )

    with pytest.raises(gallery.ValidationGalleryError, match="exactly 18"):
        gallery.select_base_validation_checkpoint(snapshots[:-1])
    with pytest.raises(gallery.ValidationGalleryError, match="duplicate completed epoch"):
        gallery.select_base_validation_checkpoint(snapshots[:-1] + (snapshots[0],))
    with pytest.raises(TypeError, match="exact built-in tuple"):
        gallery.select_base_validation_checkpoint(list(snapshots))


def test_selection_rejects_mixed_run_or_binding_and_reused_checkpoint() -> None:
    snapshots = _selection_snapshots()
    mixed_run = snapshots[:-1] + (_snapshot(epoch=30, good=True, system="07"),)
    with pytest.raises(gallery.ValidationGalleryError, match="same run/input"):
        gallery.select_base_validation_checkpoint(mixed_run)

    fixture = _fixture()
    fixture["completed_epoch"] = 30
    fixture["checkpoint_sha256"] = _digest("checkpoint/29/00")
    repeated_checkpoint = gallery.build_validation_gallery_snapshot(**fixture)
    with pytest.raises(gallery.ValidationGalleryError, match="distinct checkpoint"):
        gallery.select_base_validation_checkpoint(
            snapshots[:-1] + (repeated_checkpoint,)
        )


class _ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda value: value.update(motion_ab=value["motion_ab"].astype(np.float64)), "dtype float32"),
        (lambda value: value.update(motion_ab=value["motion_ab"][:, ::-1]), "C-contiguous"),
        (lambda value: value["motion_ab"].__setitem__((0, 0), np.nan), "finite"),
        (lambda value: value.update(motion_ba=np.ones((3, 2), dtype=np.float32)), "shapes must be identical"),
        (
            lambda value: value.update(captions=value["captions"][:-1].copy()),
            r"exactly \[3\*C,D\]",
        ),
        (lambda value: value["captions"].__setitem__((0, slice(None)), 0), "unit-L2"),
        (lambda value: value.update(completed_epoch=12), "closed interval"),
        (lambda value: value.update(completed_epoch=True), "exact built-in int"),
        (lambda value: value.update(checkpoint_sha256="A" * 64), "lowercase SHA-256"),
        (lambda value: value.update(run_id="phasepair-run-v1/BASE_TRAIN/1729/00"), "IDENTITY_VERSION"),
        (lambda value: value.update(run_id="phasepair-run-v2/RESIDUAL_HEAD_TRAIN/1729/01"), "BASE_TRAIN"),
    ],
)
def test_snapshot_rejects_invalid_numeric_shape_identity_or_epoch(
    mutation, message: str
) -> None:
    fixture = _fixture()
    mutation(fixture)
    with pytest.raises((TypeError, gallery.ValidationGalleryError), match=message):
        gallery.build_validation_gallery_snapshot(**fixture)


def test_snapshot_rejects_array_subclasses_and_gallery_identity_mutants() -> None:
    fixture = _fixture()
    fixture["motion_ab"] = fixture["motion_ab"].view(_ArraySubclass)
    with pytest.raises(TypeError, match="not a subclass"):
        gallery.build_validation_gallery_snapshot(**fixture)

    fixture = _fixture()
    fixture["motion_pair_commitments"] = (_raw32(2), _raw32(1))
    with pytest.raises(gallery.ValidationGalleryError, match="raw32-ascending"):
        gallery.build_validation_gallery_snapshot(**fixture)

    fixture = _fixture()
    fixture["caption_source_cluster_ids"] = fixture[
        "caption_source_cluster_ids"
    ][:-1]
    with pytest.raises(gallery.ValidationGalleryError, match="caption identity rows"):
        gallery.build_validation_gallery_snapshot(**fixture)


def test_opaque_receipts_cannot_be_forged_or_cross_used() -> None:
    with pytest.raises(gallery.ValidationGalleryError, match="minted only"):
        gallery.ValidationGallerySnapshot()
    with pytest.raises(gallery.ValidationGalleryError, match="minted only"):
        gallery.BaseValidationSelection()
    with pytest.raises(TypeError, match="exactly ValidationGallerySnapshot"):
        gallery.validation_gallery_snapshot_bytes(object())
    with pytest.raises(TypeError, match="exactly BaseValidationSelection"):
        gallery.base_validation_selection_bytes(object())


def test_authority_zero_constants_and_epoch_windows_are_unambiguous() -> None:
    assert gallery.AUTHORITY == 0
    assert gallery.PRODUCTION is False
    assert gallery.EXECUTION_AUTHORIZED is False
    assert gallery.RESULT_CLAIMED is False
    assert gallery.STATUS.endswith("AUTHORITY0_NO_RESULT")
    assert gallery.BASE_COMPLETED_EPOCH_ORDER == tuple(range(13, 31))
    assert gallery.BASE_EPOCH_INDEX_ORDER == tuple(range(12, 30))
    assert gallery.SCORE_SWAP_TOLERANCE == 1e-6
    assert gallery.SELECTION_TIE_TOLERANCE == 1e-12
