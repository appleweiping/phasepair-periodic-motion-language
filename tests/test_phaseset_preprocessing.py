from __future__ import annotations

import math

import numpy as np
import pytest

from phaseset_core import preprocessing


def _complete_mask() -> np.ndarray:
    return np.ones((300, 3), dtype=np.bool_)


def test_missing_rule_freezes_300_source_frames_and_285_valid_per_actor() -> None:
    complete = preprocessing.assess_missing_window(_complete_mask())
    assert complete.accepted is True
    assert complete.reason == "ACCEPT"
    assert complete.minimum_participant_valid_fraction == 1.0

    exactly_95 = _complete_mask()
    exactly_95[50:55, 0] = False
    exactly_95[100:105, 0] = False
    exactly_95[150:155, 0] = False
    accepted = preprocessing.assess_missing_window(exactly_95)
    assert accepted.accepted is True
    assert accepted.minimum_participant_valid_fraction == pytest.approx(285 / 300)
    assert accepted.longest_missing_run == 5

    below_95 = exactly_95.copy()
    below_95[200, 0] = False
    rejected = preprocessing.assess_missing_window(below_95)
    assert rejected.accepted is False
    assert rejected.reason == "ACTOR_VALID_FRACTION_BELOW_THRESHOLD"

    wrong_length = preprocessing.assess_missing_window(np.ones((299, 3), dtype=np.bool_))
    assert wrong_length.accepted is False
    assert wrong_length.reason == "WRONG_SOURCE_FRAME_COUNT"


def test_quarter_second_rounding_is_seven_source_frames_not_eight() -> None:
    assert preprocessing.MAX_MISSING_RUN_SOURCE_FRAMES == 7
    assert 7 / preprocessing.SOURCE_FPS <= preprocessing.MAX_MISSING_SECONDS
    assert 8 / preprocessing.SOURCE_FPS > preprocessing.MAX_MISSING_SECONDS

    seven = _complete_mask()
    seven[100:107, 1] = False
    accepted = preprocessing.assess_missing_window(seven)
    assert accepted.accepted is True
    assert accepted.longest_missing_run == 7

    eight = _complete_mask()
    eight[100:108, 1] = False
    rejected = preprocessing.assess_missing_window(eight)
    assert rejected.accepted is False
    assert rejected.reason == "MISSING_RUN_EXCEEDS_DURATION_BOUND"


def test_boundary_gaps_are_rejected_instead_of_extrapolated() -> None:
    leading = _complete_mask()
    leading[:2, 0] = False
    decision = preprocessing.assess_missing_window(leading)
    assert decision.accepted is False
    assert decision.reason == "UNBOUNDED_BOUNDARY_GAP"
    assert decision.has_boundary_missing is True

    trailing = _complete_mask()
    trailing[-1, 2] = False
    with pytest.raises(preprocessing.PreprocessingError, match="UNBOUNDED_BOUNDARY_GAP"):
        values = np.zeros((300, 3, 1), dtype=np.float32)
        preprocessing.interpolate_accepted_gaps(values, observed_mask=trailing)


def test_internal_gaps_are_linearly_filled_while_source_mask_is_retained() -> None:
    time = np.arange(300, dtype=np.float64)
    values = np.repeat(time[:, None, None], 3, axis=1)
    mask = _complete_mask()
    mask[100:107, 1] = False
    values[100:107, 1, 0] = np.nan
    interpolated = preprocessing.interpolate_accepted_gaps(values, observed_mask=mask)
    np.testing.assert_allclose(interpolated.values[99:108, 1, 0], np.arange(99, 108))
    assert interpolated.observed_mask[100:107, 1].tolist() == [False] * 7
    assert interpolated.observed_mask.flags.writeable is False
    assert interpolated.values.flags.writeable is False
    assert interpolated.decision.missing_actor_frames == 7


def test_nonfinite_observed_values_fail_closed() -> None:
    values = np.zeros((300, 3, 1), dtype=np.float32)
    values[100, 0, 0] = np.nan
    with pytest.raises(preprocessing.PreprocessingError, match="observed actor-frame"):
        preprocessing.interpolate_accepted_gaps(values, observed_mask=_complete_mask())


def test_antialias_fir_is_symmetric_and_has_unit_dc_gain() -> None:
    kernel = preprocessing.design_antialias_fir()
    assert kernel.shape == (31,)
    assert kernel.flags.writeable is False
    np.testing.assert_allclose(kernel, kernel[::-1], atol=0.0, rtol=0.0)
    assert float(kernel.sum()) == pytest.approx(1.0, abs=1.0e-14)


def test_30_to_20_resampling_emits_200_frames_and_preserves_masks() -> None:
    time = np.arange(300, dtype=np.float64) / 30.0
    values = np.repeat(np.sin(2.0 * np.pi * 3.0 * time)[:, None, None], 3, axis=1)
    mask = _complete_mask()
    mask[100:107, 1] = False
    values[100:107, 1, 0] = np.nan
    output = preprocessing.resample_30_to_20(values, observed_mask=mask)
    assert output.values.shape == (200, 3, 1)
    assert output.source_observed_mask.shape == (300, 3)
    assert output.target_observed_mask.shape == (200, 3)
    assert output.interpolated_actor_frames == 7
    assert output.source_observed_mask[100:107, 1].tolist() == [False] * 7
    assert np.count_nonzero(~output.target_observed_mask[:, 1]) > 0
    assert output.timestamps_seconds[0] == 0.0
    assert output.timestamps_seconds[-1] == pytest.approx(199 / 20.0)
    assert output.values.flags.writeable is False
    assert output.source_observed_mask.flags.writeable is False
    assert output.target_observed_mask.flags.writeable is False


def test_antialiasing_preserves_low_tone_and_attenuates_above_output_nyquist() -> None:
    time = np.arange(300, dtype=np.float64) / 30.0
    low = np.repeat(np.sin(2.0 * np.pi * 3.0 * time)[:, None, None], 3, axis=1)
    high = np.repeat(np.sin(2.0 * np.pi * 12.0 * time)[:, None, None], 3, axis=1)
    mask = _complete_mask()
    low_output = preprocessing.resample_30_to_20(low, observed_mask=mask)
    high_output = preprocessing.resample_30_to_20(high, observed_mask=mask)
    low_rms = float(np.sqrt(np.mean(low_output.values[20:-20, 0, 0] ** 2)))
    high_rms = float(np.sqrt(np.mean(high_output.values[20:-20, 0, 0] ** 2)))
    assert low_rms > 0.65
    assert high_rms < 0.08


def test_group_origin_uses_first_all_actor_valid_frame() -> None:
    positions = np.array(
        [
            [[[100.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]], [[100.0, 0.0, 0.0]]],
            [[[1.0, 0.0, 0.0]], [[3.0, 0.0, 0.0]], [[2.0, 0.0, 3.0]]],
            [[[2.0, 0.0, 0.0]], [[4.0, 0.0, 0.0]], [[3.0, 0.0, 3.0]]],
        ],
        dtype=np.float64,
    )
    yaw = np.zeros((3, 3), dtype=np.float64)
    mask = np.ones((3, 3), dtype=np.bool_)
    mask[0, 0] = False
    result = preprocessing.canonicalize_group(positions, yaw, observed_mask=mask)
    assert result.reference_frame == 1
    np.testing.assert_allclose(result.group_center, [2.0, 0.0, 1.0])
    np.testing.assert_allclose(result.positions[1, :, 0, :].mean(axis=0), 0.0)


def test_scene_and_augmentation_yaw_are_explicit_shared_group_transforms() -> None:
    positions = np.array(
        [[[[1.0, 0.0, 0.0]], [[-1.0, 0.0, 0.0]], [[0.0, 0.0, 2.0]]]] * 2,
        dtype=np.float64,
    )
    yaw = np.array([[0.0, 2.0 * math.pi / 3.0, -2.0 * math.pi / 3.0]] * 2)
    mask = np.ones((2, 3), dtype=np.bool_)
    result = preprocessing.canonicalize_group(
        positions,
        yaw,
        observed_mask=mask,
        shared_yaw=math.pi / 2,
        augmentation_yaw=math.pi / 4,
    )
    assert result.shared_yaw == pytest.approx(math.pi / 2)
    assert result.augmentation_yaw == pytest.approx(math.pi / 4)
    assert result.applied_yaw == pytest.approx(-math.pi / 4)
    np.testing.assert_allclose(
        result.root_yaw[:, 0],
        -math.pi / 4,
    )
    original_distance = np.linalg.norm(positions[:, 0] - positions[:, 1], axis=-1)
    canonical_distance = np.linalg.norm(result.positions[:, 0] - result.positions[:, 1], axis=-1)
    np.testing.assert_allclose(canonical_distance, original_distance)


def test_group_transform_is_participant_permutation_invariant() -> None:
    generator = np.random.default_rng(42)
    positions = generator.normal(size=(4, 3, 2, 3)).astype(np.float64)
    yaw = generator.normal(size=(4, 3)).astype(np.float64)
    mask = np.ones((4, 3), dtype=np.bool_)
    original = preprocessing.canonicalize_group(
        positions,
        yaw,
        observed_mask=mask,
        shared_yaw=0.3,
        augmentation_yaw=-0.2,
    )
    permutation = np.array([2, 0, 1])
    permuted = preprocessing.canonicalize_group(
        positions[:, permutation],
        yaw[:, permutation],
        observed_mask=mask[:, permutation],
        shared_yaw=0.3,
        augmentation_yaw=-0.2,
    )
    inverse = np.argsort(permutation)
    np.testing.assert_allclose(permuted.positions[:, inverse], original.positions)
    np.testing.assert_allclose(permuted.root_yaw[:, inverse], original.root_yaw)
    np.testing.assert_allclose(permuted.group_center, original.group_center)
