from __future__ import annotations

import hashlib
import math

import numpy as np
import pytest

from phaseset_core import pipeline


def _commitment(index: int) -> bytes:
    return hashlib.sha256(f"actor-{index}".encode("ascii")).digest()


def _capture(
    *,
    actors: int = 3,
    frames: int = 300,
    joints: int = 30,
) -> pipeline.PrivateCaptureArrays:
    time = np.arange(frames, dtype=np.float32) / 30.0
    values = np.zeros((frames, actors, joints, 3), dtype=np.float32)
    for actor in range(actors):
        values[:, actor, :, 0] = actor * 2.0 + 0.1 * np.sin(time * (actor + 1))[:, None]
        values[:, actor, :, 1] = np.arange(joints, dtype=np.float32)[None, :] * 0.01
        values[:, actor, :, 2] = actor * -0.5 + 0.05 * np.cos(time * 2.0)[:, None]
    values[:, :, 22:] = 1.0e6
    mask = np.ones((frames, actors, joints), dtype=np.bool_)
    return pipeline.PrivateCaptureArrays(
        values,
        mask,
        tuple(_commitment(index) for index in range(actors)),
    )


def _permuted(
    capture: pipeline.PrivateCaptureArrays,
    order: tuple[int, ...],
) -> pipeline.PrivateCaptureArrays:
    index = np.asarray(order, dtype=np.int64)
    return pipeline.PrivateCaptureArrays(
        capture.joints[:, index],
        capture.track_mask[:, index],
        tuple(capture.actor_commitments[item] for item in order),
    )


def test_private_actor_commitments_are_salted_and_never_key_group_yaw() -> None:
    salt = bytes(range(32))
    first = pipeline.derive_actor_commitment(salt, "private-participant-a")
    again = pipeline.derive_actor_commitment(salt, "private-participant-a")
    second = pipeline.derive_actor_commitment(salt, "private-participant-b")
    assert first == again
    assert first != second
    assert len(first) == 32

    capture = pipeline.PrivateCaptureArrays(
        _capture().joints,
        _capture().track_mask,
        (first, second, _commitment(7)),
    )
    permuted = _permuted(capture, (2, 0, 1))
    assert capture.group_commitment == permuted.group_commitment
    yaw = pipeline.deterministic_group_yaw(seed=1729, epoch=3, window_ordinal=7)
    assert yaw == pipeline.deterministic_group_yaw(
        seed=1729,
        epoch=3,
        window_ordinal=7,
    )
    replacement_commitments = tuple(_commitment(index + 100) for index in range(3))
    replacement = pipeline.PrivateCaptureArrays(
        capture.joints,
        capture.track_mask,
        replacement_commitments,
    )
    assert replacement.group_commitment != capture.group_commitment
    assert yaw == pipeline.deterministic_group_yaw(
        seed=1729,
        epoch=3,
        window_ordinal=7,
    )
    assert -math.pi <= yaw < math.pi

    with pytest.raises(pipeline.PhaseSetPipelineError, match="at least 32 bytes"):
        pipeline.derive_actor_commitment(b"short", "private-participant-a")


def test_prepare_window_extracts_only_body22_and_centers_the_whole_group() -> None:
    capture = _capture()
    prepared = pipeline.prepare_window(capture, source_start_frame=0)
    assert prepared.skeletons.shape == (3, 200, 22, 3)
    assert prepared.track_mask.all()
    assert prepared.decision.accepted is True
    assert prepared.edge_count == 3
    assert len(prepared.source_sha256) == 64
    assert len(prepared.window_sha256) == 64
    np.testing.assert_allclose(
        prepared.skeletons[:, 0, 0].mean(axis=0, dtype=np.float64),
        np.zeros(3),
        atol=2.0e-7,
        rtol=0.0,
    )
    assert float(np.max(np.abs(prepared.skeletons))) < 10.0


def test_pipeline_accepts_k2_for_dyadic_backward_transfer() -> None:
    prepared = pipeline.prepare_window(_capture(actors=2), source_start_frame=0)
    assert prepared.skeletons.shape == (2, 200, 22, 3)
    assert prepared.actor_count == 2
    assert prepared.edge_count == 1
    batch = pipeline.collate_group_samples((prepared,))
    assert batch.actor_counts == (2,)


def test_preparation_is_bitwise_invariant_to_private_actor_permutation() -> None:
    capture = _capture(actors=4)
    permuted = _permuted(capture, (3, 1, 0, 2))
    original = pipeline.prepare_window(capture, source_start_frame=0, augmentation_yaw=0.37)
    reordered = pipeline.prepare_window(permuted, source_start_frame=0, augmentation_yaw=0.37)
    assert original.actor_commitments == reordered.actor_commitments
    assert original.group_commitment == reordered.group_commitment
    assert original.skeletons.tobytes() == reordered.skeletons.tobytes()
    assert original.track_mask.tobytes() == reordered.track_mask.tobytes()


def test_short_internal_missing_span_is_masked_exact_zero_after_interpolation() -> None:
    source = _capture()
    joints = np.array(source.joints, copy=True)
    mask = np.array(source.track_mask, copy=True)
    mask[100:107, 1, :22] = False
    joints[100:107, 1, :22] = np.nan
    capture = pipeline.PrivateCaptureArrays(joints, mask, source.actor_commitments)
    prepared = pipeline.prepare_window(capture, source_start_frame=0)
    assert np.count_nonzero(~prepared.track_mask) > 0
    raw = prepared.skeletons.view(np.uint32)
    missing = ~prepared.track_mask[..., None].repeat(3, axis=-1)
    assert np.count_nonzero(raw[missing]) == 0
    assert prepared.decision.longest_missing_run == 7


def test_partial_joint_tracking_mask_is_rejected_before_actor_interpolation() -> None:
    source = _capture()
    joints = np.array(source.joints, copy=True)
    mask = np.array(source.track_mask, copy=True)
    mask[100:107, 1, 0] = False
    joints[100:107, 1, 0] = np.nan
    capture = pipeline.PrivateCaptureArrays(joints, mask, source.actor_commitments)
    with pytest.raises(pipeline.PhaseSetPipelineError, match="uniform within each actor-frame"):
        pipeline.prepare_window(capture, source_start_frame=0)


def test_long_missing_span_is_rejected_without_dropping_the_actor() -> None:
    source = _capture(frames=620)
    joints = np.array(source.joints, copy=True)
    mask = np.array(source.track_mask, copy=True)
    mask[100:108, 2, :22] = False
    joints[100:108, 2, :22] = np.nan
    capture = pipeline.PrivateCaptureArrays(joints, mask, source.actor_commitments)
    report = pipeline.prepare_capture(capture, training=False)
    assert len(report.accepted) == 1
    assert report.accepted[0].source_start_frame == 300
    assert report.rejected == (
        pipeline.RejectedWindow(0, "MISSING_RUN_EXCEEDS_DURATION_BOUND"),
    )
    assert report.trailing_source_frames == 20


def test_training_yaw_is_shared_and_frozen_by_seed_epoch() -> None:
    capture = _capture(frames=600)
    first = pipeline.prepare_capture(
        capture,
        training=True,
        seed=1729,
        epoch=4,
        manifest_window_ordinal_start=100,
    )
    again = pipeline.prepare_capture(
        capture,
        training=True,
        seed=1729,
        epoch=4,
        manifest_window_ordinal_start=100,
    )
    other = pipeline.prepare_capture(
        capture,
        training=True,
        seed=1729,
        epoch=5,
        manifest_window_ordinal_start=100,
    )
    assert len(first.accepted) == 2
    assert first.accepted[0].augmentation_yaw == again.accepted[0].augmentation_yaw
    assert first.accepted[0].skeletons.tobytes() == again.accepted[0].skeletons.tobytes()
    assert first.accepted[0].augmentation_yaw != other.accepted[0].augmentation_yaw
    assert first.accepted[0].augmentation_yaw != first.accepted[1].augmentation_yaw
    next_capture = pipeline.prepare_capture(
        _capture(),
        training=True,
        seed=1729,
        epoch=4,
        manifest_window_ordinal_start=200,
    )
    assert first.accepted[0].augmentation_yaw != next_capture.accepted[0].augmentation_yaw
    with pytest.raises(pipeline.PhaseSetPipelineError, match="global uint64"):
        pipeline.prepare_capture(capture, training=True, seed=1729, epoch=4)
    evaluation = pipeline.prepare_capture(capture, training=False)
    assert all(sample.augmentation_yaw == 0.0 for sample in evaluation.accepted)


def test_dynamic_collation_and_edge_budget_never_sample_actors_or_edges() -> None:
    sample3 = pipeline.prepare_window(_capture(actors=3), source_start_frame=0)
    sample5 = pipeline.prepare_window(_capture(actors=5), source_start_frame=0)
    batch = pipeline.collate_group_samples((sample5, sample3))
    assert batch.skeletons.shape == (2, 5, 200, 22, 3)
    assert batch.actor_counts == (5, 3)
    assert np.count_nonzero(batch.skeletons[1, 3:]) == 0
    assert not batch.track_mask[1, 3:].any()
    assert pipeline.total_edge_count((sample3, sample5)) == 13

    buckets = pipeline.edge_budget_batches(
        (sample5, sample3),
        max_total_edges=10,
    )
    assert tuple(tuple(item.actor_count for item in group) for group in buckets) == (
        (3,),
        (5,),
    )
    with pytest.raises(pipeline.PreparationResourceLimit, match="RESOURCE_LIMIT"):
        pipeline.edge_budget_batches((sample5,), max_total_edges=9)


def test_numeric_npz_seam_rejects_extra_keys_and_pickle(tmp_path) -> None:
    capture = _capture()
    commitments = np.asarray(
        [list(value) for value in capture.actor_commitments],
        dtype=np.uint8,
    )
    good = tmp_path / "capture.npz"
    np.savez(
        good,
        joints=capture.joints,
        track_mask=capture.track_mask,
        actor_commitments=commitments,
    )
    loaded = pipeline.load_private_capture_npz(good)
    assert loaded.actor_commitments == capture.actor_commitments
    assert loaded.source_sha256 == hashlib.sha256(good.read_bytes()).hexdigest()

    extra = tmp_path / "capture-extra.npz"
    np.savez(
        extra,
        joints=capture.joints,
        track_mask=capture.track_mask,
        actor_commitments=commitments,
        participant_names=np.asarray(["private-a", "private-b", "private-c"]),
    )
    with pytest.raises(pipeline.PhaseSetPipelineError, match="unregistered key"):
        pipeline.load_private_capture_npz(extra)

    unsafe = tmp_path / "capture-pickle.npz"
    np.savez(
        unsafe,
        joints=np.asarray([object()], dtype=object),
        track_mask=capture.track_mask,
        actor_commitments=commitments,
    )
    with pytest.raises(pipeline.PhaseSetPipelineError, match="safe numeric NPZ"):
        pipeline.load_private_capture_npz(unsafe)
