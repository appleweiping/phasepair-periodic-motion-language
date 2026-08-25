"""Public skeleton and internal activity contract tests for PhaseSet."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from phaseset_core.contracts import (
    ACTIVITY_JOINT_GROUPS,
    GroupContractError,
    PreparedActivityBatch,
    PreparedGroupBatch,
    group_commitment,
    skeleton_to_activity,
    validate_prepared_group_batch,
)


def _key(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _batch(
    *,
    actor_count: int = 3,
    padded_actors: int | None = None,
    positions: tuple[int, ...] | None = None,
    physical_order: tuple[int, ...] | None = None,
    time_steps: int = 6,
    valid_length: int = 5,
) -> PreparedGroupBatch:
    padded = actor_count if padded_actors is None else padded_actors
    positions = tuple(range(actor_count)) if positions is None else positions
    physical_order = (
        tuple(range(actor_count)) if physical_order is None else physical_order
    )
    skeletons = np.zeros((1, padded, time_steps, 22, 3), dtype=np.float32)
    actor_mask = np.zeros((1, padded), dtype=np.bool_)
    track_mask = np.zeros((1, padded, time_steps, 22), dtype=np.bool_)
    commitments: list[bytes | None] = [None] * padded
    keys = tuple(_key(f"actor-{index}") for index in range(actor_count))
    for position, physical_actor in zip(positions, physical_order, strict=True):
        actor_mask[0, position] = True
        commitments[position] = keys[physical_actor]
        track_mask[0, position, :valid_length] = True
        for time_index in range(valid_length):
            skeletons[0, position, time_index, :, 0] = np.float32(
                0.01 * physical_actor + 0.1 * time_index
            )
            skeletons[0, position, time_index, :, 1] = np.float32(
                0.02 * physical_actor
            )
    frame_mask = np.zeros((1, time_steps), dtype=np.bool_)
    frame_mask[:, :valid_length] = True
    return PreparedGroupBatch(
        skeletons,
        actor_mask,
        frame_mask,
        track_mask,
        (tuple(commitments),),
        (group_commitment(keys),),
    )


def test_group_commitment_is_order_invariant_and_rejects_duplicates() -> None:
    keys = (_key("a"), _key("b"), _key("c"))
    assert group_commitment(keys) == group_commitment((keys[2], keys[0], keys[1]))
    with pytest.raises(GroupContractError, match="distinct"):
        group_commitment((keys[0], keys[0]))


def test_public_batch_accepts_nonprefix_actors_canonicalizes_and_freezes() -> None:
    canonical = _batch(actor_count=3, padded_actors=7, positions=(0, 1, 2))
    scattered = _batch(
        actor_count=3,
        padded_actors=7,
        positions=(6, 1, 4),
        physical_order=(2, 0, 1),
    )
    assert canonical.actor_counts == (3,)
    assert canonical.valid_lengths == (5,)
    assert np.array_equal(canonical.skeletons, scattered.skeletons)
    assert np.array_equal(canonical.track_mask, scattered.track_mask)
    assert np.array_equal(
        scattered.actor_mask,
        np.array([[1, 1, 1, 0, 0, 0, 0]], dtype=np.bool_),
    )
    assert canonical.actor_commitments == scattered.actor_commitments
    assert not scattered.skeletons.flags.writeable
    assert not scattered.track_mask.flags.writeable
    rebuilt = validate_prepared_group_batch(scattered)
    assert np.array_equal(rebuilt.skeletons, scattered.skeletons)


def test_public_batch_requires_exact_zero_for_actor_frame_and_joint_padding() -> None:
    good = _batch(actor_count=2, padded_actors=3, positions=(0, 1))
    arrays = {
        "skeletons": np.array(good.skeletons, copy=True),
        "actor_mask": np.array(good.actor_mask, copy=True),
        "frame_mask": np.array(good.frame_mask, copy=True),
        "track_mask": np.array(good.track_mask, copy=True),
    }

    bad_actor = arrays["skeletons"].copy()
    bad_actor[0, 2, 0, 0, 0] = 1.0
    with pytest.raises(GroupContractError, match="invalid actor skeleton"):
        PreparedGroupBatch(
            bad_actor,
            arrays["actor_mask"],
            arrays["frame_mask"],
            arrays["track_mask"],
            good.actor_commitments,
            good.group_commitments,
        )

    bad_frame = arrays["skeletons"].copy()
    bad_frame[0, 0, -1, 0, 0] = np.float32(-0.0)
    with pytest.raises(GroupContractError, match="invalid frame skeleton"):
        PreparedGroupBatch(
            bad_frame,
            arrays["actor_mask"],
            arrays["frame_mask"],
            arrays["track_mask"],
            good.actor_commitments,
            good.group_commitments,
        )

    missing_track = arrays["track_mask"].copy()
    missing_track[0, 0, 2, 0] = False
    with pytest.raises(GroupContractError, match="untracked joint"):
        PreparedGroupBatch(
            arrays["skeletons"],
            arrays["actor_mask"],
            arrays["frame_mask"],
            missing_track,
            good.actor_commitments,
            good.group_commitments,
        )


def test_skeleton_to_activity_is_deterministic_20hz_speed_and_propagates_mask() -> None:
    batch = _batch(actor_count=2, time_steps=5, valid_length=4)
    activity = skeleton_to_activity(batch)
    assert type(activity) is PreparedActivityBatch
    assert activity.activities.shape == (1, 2, 5, 5)
    assert np.array_equal(activity.activities[:, :, 0], np.zeros((1, 2, 5), np.float32))
    assert not np.signbit(activity.activities[:, :, 0]).any()
    assert not activity.activity_mask[:, :, 0].any()
    # Every joint moves 0.1 m per 20-Hz step, hence every body-group speed is 2 m/s.
    assert np.allclose(activity.activities[:, :, 1:4], 2.0, atol=1e-6, rtol=0.0)
    assert activity.activity_mask[:, :, 1:4].all()
    assert not activity.activity_mask[:, :, 4].any()

    skeletons = np.array(batch.skeletons, copy=True)
    tracks = np.array(batch.track_mask, copy=True)
    left_arm = np.asarray(ACTIVITY_JOINT_GROUPS[1], dtype=np.int64)
    tracks[0, 0, 2, left_arm] = False
    skeletons[0, 0, 2, left_arm] = np.float32(0.0)
    missing = PreparedGroupBatch(
        skeletons,
        np.array(batch.actor_mask, copy=True),
        np.array(batch.frame_mask, copy=True),
        tracks,
        batch.actor_commitments,
        batch.group_commitments,
    )
    derived = skeleton_to_activity(missing)
    assert not bool(derived.activity_mask[0, 0, 2, 1])
    assert not bool(derived.activity_mask[0, 0, 3, 1])
    assert derived.activities[0, 0, 2, 1].view(np.uint32) == 0
    assert derived.activities[0, 0, 3, 1].view(np.uint32) == 0
