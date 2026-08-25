"""Mask-preserving group preprocessing for PhaseSet motion windows.

The frozen production path consumes 300 source frames at 30 Hz and emits 200
frames at 20 Hz.  Short, bounded missing spans may be linearly interpolated for
kinematics, but their original observation mask is retained.  Translation and
yaw transforms are shared by the complete group.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


SOURCE_FPS = 30
TARGET_FPS = 20
SOURCE_WINDOW_FRAMES = 300
TARGET_WINDOW_FRAMES = 200
MINIMUM_GROUP_SIZE = 3
MINIMUM_VALID_FRACTION = 0.95
MAX_MISSING_SECONDS = 0.25
# Missing duration is defined as run_length / SOURCE_FPS.  Seven frames are
# 0.233... seconds; eight frames are 0.266... seconds and therefore exceed 0.25.
MAX_MISSING_RUN_SOURCE_FRAMES = 7
DEFAULT_CUTOFF_HZ = 9.0
DEFAULT_FIR_TAPS = 31


class PreprocessingError(ValueError):
    """A motion window violates the frozen preprocessing contract."""


@dataclass(frozen=True, slots=True)
class WindowDecision:
    accepted: bool
    reason: str
    frame_count: int
    participant_count: int
    missing_actor_frames: int
    minimum_participant_valid_fraction: float
    longest_missing_run: int
    has_boundary_missing: bool

    @property
    def worst_participant_missing_fraction(self) -> float:
        return 1.0 - self.minimum_participant_valid_fraction


@dataclass(frozen=True, slots=True)
class InterpolatedWindow:
    values: np.ndarray
    observed_mask: np.ndarray
    decision: WindowDecision


@dataclass(frozen=True, slots=True)
class ResampledMotion:
    values: np.ndarray
    timestamps_seconds: np.ndarray
    source_observed_mask: np.ndarray
    target_observed_mask: np.ndarray
    interpolated_actor_frames: int
    source_fps: int
    target_fps: int
    anti_alias_fir: np.ndarray


@dataclass(frozen=True, slots=True)
class CanonicalGroupMotion:
    positions: np.ndarray
    root_yaw: np.ndarray
    group_center: np.ndarray
    shared_yaw: float
    augmentation_yaw: float
    applied_yaw: float
    reference_frame: int


def _float_array(
    value: object,
    label: str,
    *,
    dimensions: int | None = None,
    require_finite: bool = True,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray")
    if value.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise PreprocessingError(f"{label} must have dtype float32 or float64")
    if dimensions is not None and value.ndim != dimensions:
        raise PreprocessingError(f"{label} must have {dimensions} dimensions")
    if value.size == 0:
        raise PreprocessingError(f"{label} must be nonempty")
    if require_finite and not bool(np.isfinite(value).all()):
        raise PreprocessingError(f"{label} must be finite")
    return value


def _observed_mask(
    value: object,
    *,
    frame_count: int | None = None,
    participant_count: int | None = None,
) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError("observed_mask must be an exact numpy.ndarray")
    if value.dtype != np.dtype(np.bool_) or value.ndim != 2:
        raise PreprocessingError("observed_mask must be a two-dimensional bool array")
    if value.shape[0] == 0 or value.shape[1] < MINIMUM_GROUP_SIZE:
        raise PreprocessingError("observed_mask must describe T>0 and K>=3")
    if frame_count is not None and value.shape[0] != frame_count:
        raise PreprocessingError("observed_mask frame count differs from motion")
    if participant_count is not None and value.shape[1] != participant_count:
        raise PreprocessingError("observed_mask participant count differs from motion")
    return value


def _longest_false_run(mask: np.ndarray) -> int:
    longest = 0
    for participant in range(mask.shape[1]):
        current = 0
        for observed in mask[:, participant]:
            if bool(observed):
                current = 0
            else:
                current += 1
                longest = max(longest, current)
    return longest


def assess_missing_window(
    observed_mask: np.ndarray,
    *,
    source_window_frames: int = SOURCE_WINDOW_FRAMES,
    minimum_valid_fraction: float = MINIMUM_VALID_FRACTION,
    max_consecutive_missing: int = MAX_MISSING_RUN_SOURCE_FRAMES,
    reject_boundary_missing: bool = True,
) -> WindowDecision:
    """Apply the fixed source-window acceptance and interpolation boundary rule.

    At the defaults, each actor must have at least 285 observed frames out of
    300, every missing run must contain at most seven frames, and every run must
    be bounded by observations so that no temporal extrapolation is needed.
    """

    mask = _observed_mask(observed_mask)
    if type(source_window_frames) is not int or source_window_frames < 2:
        raise PreprocessingError("source_window_frames must be an exact int >= 2")
    if isinstance(minimum_valid_fraction, (bool, np.bool_)) or not isinstance(
        minimum_valid_fraction, (int, float, np.number)
    ):
        raise TypeError("minimum_valid_fraction must be numeric")
    valid_limit = float(minimum_valid_fraction)
    if not math.isfinite(valid_limit) or not 0.0 < valid_limit <= 1.0:
        raise PreprocessingError("minimum_valid_fraction must be finite and in (0,1]")
    if type(max_consecutive_missing) is not int or max_consecutive_missing < 0:
        raise PreprocessingError("max_consecutive_missing must be an exact int >= 0")
    if type(reject_boundary_missing) is not bool:
        raise TypeError("reject_boundary_missing must be an exact bool")

    frame_count, participant_count = mask.shape
    valid_by_participant = np.count_nonzero(mask, axis=0)
    missing_actor_frames = int(frame_count * participant_count - valid_by_participant.sum())
    minimum_fraction = float(valid_by_participant.min(initial=frame_count) / frame_count)
    longest_run = _longest_false_run(mask)
    boundary_missing = bool((~mask[0]).any() or (~mask[-1]).any())
    if frame_count != source_window_frames:
        reason = "WRONG_SOURCE_FRAME_COUNT"
    elif minimum_fraction < valid_limit:
        reason = "ACTOR_VALID_FRACTION_BELOW_THRESHOLD"
    elif longest_run > max_consecutive_missing:
        reason = "MISSING_RUN_EXCEEDS_DURATION_BOUND"
    elif reject_boundary_missing and boundary_missing:
        reason = "UNBOUNDED_BOUNDARY_GAP"
    else:
        reason = "ACCEPT"
    return WindowDecision(
        accepted=reason == "ACCEPT",
        reason=reason,
        frame_count=frame_count,
        participant_count=participant_count,
        missing_actor_frames=missing_actor_frames,
        minimum_participant_valid_fraction=minimum_fraction,
        longest_missing_run=longest_run,
        has_boundary_missing=boundary_missing,
    )


def interpolate_accepted_gaps(
    values: np.ndarray,
    *,
    observed_mask: np.ndarray,
) -> InterpolatedWindow:
    """Linearly fill only accepted internal actor gaps and retain the source mask.

    ``values`` must be shaped ``[300,K,...]``.  Non-finite values are permitted
    only where the corresponding actor-frame mask is false.
    """

    motion = _float_array(values, "values", require_finite=False)
    if motion.ndim < 3:
        raise PreprocessingError("values must have shape [T,K,...]")
    mask = _observed_mask(
        observed_mask,
        frame_count=motion.shape[0],
        participant_count=motion.shape[1],
    )
    decision = assess_missing_window(mask)
    if not decision.accepted:
        raise PreprocessingError(f"window is not interpolation-eligible: {decision.reason}")

    actor_frame_finite = np.isfinite(motion.reshape(motion.shape[0], motion.shape[1], -1)).all(
        axis=2
    )
    if not bool(actor_frame_finite[mask].all()):
        raise PreprocessingError("an observed actor-frame contains a non-finite value")

    work = np.asarray(motion, dtype=np.float64).copy()
    flat = work.reshape(work.shape[0], work.shape[1], -1)
    time = np.arange(work.shape[0], dtype=np.float64)
    for participant in range(work.shape[1]):
        valid = np.flatnonzero(mask[:, participant])
        missing = np.flatnonzero(~mask[:, participant])
        if missing.size == 0:
            continue
        if valid[0] != 0 or valid[-1] != work.shape[0] - 1:
            raise AssertionError("accepted interpolation window has an unbounded gap")
        for feature in range(flat.shape[2]):
            flat[missing, participant, feature] = np.interp(
                time[missing],
                time[valid],
                flat[valid, participant, feature],
            )
    if not bool(np.isfinite(work).all()):
        raise PreprocessingError("gap interpolation did not produce finite values")

    output_dtype = np.float32 if motion.dtype == np.dtype(np.float32) else np.float64
    output_values = np.asarray(work, dtype=output_dtype)
    output_mask = np.array(mask, dtype=np.bool_, copy=True)
    output_values.setflags(write=False)
    output_mask.setflags(write=False)
    return InterpolatedWindow(output_values, output_mask, decision)


def design_antialias_fir(
    *,
    source_fps: int = SOURCE_FPS,
    target_fps: int = TARGET_FPS,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    taps: int = DEFAULT_FIR_TAPS,
) -> np.ndarray:
    """Return a symmetric Blackman-windowed low-pass FIR with unit DC gain."""

    if type(source_fps) is not int or type(target_fps) is not int:
        raise TypeError("source_fps and target_fps must be exact ints")
    if source_fps <= 0 or target_fps <= 0 or target_fps >= source_fps:
        raise PreprocessingError("sampling rates must satisfy 0 < target_fps < source_fps")
    if isinstance(cutoff_hz, bool) or not isinstance(cutoff_hz, (int, float, np.number)):
        raise TypeError("cutoff_hz must be numeric")
    cutoff = float(cutoff_hz)
    if not math.isfinite(cutoff) or not 0.0 < cutoff <= target_fps / 2.0:
        raise PreprocessingError("cutoff_hz must be finite and in (0,target Nyquist]")
    if type(taps) is not int or taps < 3 or taps % 2 != 1:
        raise PreprocessingError("taps must be an odd exact int >= 3")
    offsets = np.arange(taps, dtype=np.float64) - taps // 2
    normalized_cutoff = cutoff / source_fps
    kernel = 2.0 * normalized_cutoff * np.sinc(2.0 * normalized_cutoff * offsets)
    kernel *= np.blackman(taps)
    kernel /= kernel.sum(dtype=np.float64)
    output = np.asarray(kernel, dtype=np.float64)
    output.setflags(write=False)
    return output


def _zero_phase_filter(values: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    half = kernel.size // 2
    flattened = np.asarray(values, dtype=np.float64).reshape(values.shape[0], -1)
    padded = np.pad(flattened, ((half, half), (0, 0)), mode="edge")
    windows = np.lib.stride_tricks.sliding_window_view(padded, kernel.size, axis=0)
    filtered = np.tensordot(windows, kernel, axes=([-1], [0]))
    return np.asarray(filtered, dtype=np.float64).reshape(values.shape)


def _target_mask(mask: np.ndarray, source_positions: np.ndarray) -> np.ndarray:
    lower = np.floor(source_positions).astype(np.int64)
    upper = np.minimum(lower + 1, mask.shape[0] - 1)
    output = np.array(mask[lower], dtype=np.bool_, copy=True)
    fractional = source_positions != lower
    output[fractional] &= mask[upper[fractional]]
    output.setflags(write=False)
    return output


def resample_30_to_20(
    values: np.ndarray,
    *,
    observed_mask: np.ndarray,
    cutoff_hz: float = DEFAULT_CUTOFF_HZ,
    taps: int = DEFAULT_FIR_TAPS,
) -> ResampledMotion:
    """Interpolate accepted gaps, low-pass at 30 Hz, then sample 200 frames at 20 Hz."""

    interpolated = interpolate_accepted_gaps(values, observed_mask=observed_mask)
    kernel = design_antialias_fir(cutoff_hz=cutoff_hz, taps=taps)
    filtered = _zero_phase_filter(interpolated.values, kernel)

    output_count = ((SOURCE_WINDOW_FRAMES - 1) * TARGET_FPS) // SOURCE_FPS + 1
    if output_count != TARGET_WINDOW_FRAMES:
        raise AssertionError("frozen 300-to-200 frame arithmetic changed")
    timestamps = np.arange(output_count, dtype=np.float64) / TARGET_FPS
    source_positions = timestamps * SOURCE_FPS
    lower = np.floor(source_positions).astype(np.int64)
    upper = np.minimum(lower + 1, SOURCE_WINDOW_FRAMES - 1)
    fraction_shape = (output_count,) + (1,) * (filtered.ndim - 1)
    fraction = (source_positions - lower).reshape(fraction_shape)
    resampled = filtered[lower] * (1.0 - fraction) + filtered[upper] * fraction

    output_dtype = np.float32 if interpolated.values.dtype == np.dtype(np.float32) else np.float64
    output_values = np.asarray(resampled, dtype=output_dtype)
    source_mask = np.array(interpolated.observed_mask, dtype=np.bool_, copy=True)
    target_mask = _target_mask(interpolated.observed_mask, source_positions)
    output_values.setflags(write=False)
    source_mask.setflags(write=False)
    timestamps.setflags(write=False)
    return ResampledMotion(
        values=output_values,
        timestamps_seconds=timestamps,
        source_observed_mask=source_mask,
        target_observed_mask=target_mask,
        interpolated_actor_frames=interpolated.decision.missing_actor_frames,
        source_fps=SOURCE_FPS,
        target_fps=TARGET_FPS,
        anti_alias_fir=kernel,
    )


def _wrapped_angle(value: np.ndarray | float) -> np.ndarray | float:
    return (value + np.pi) % (2.0 * np.pi) - np.pi


def _finite_angle(value: object, label: str) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, float, np.number)):
        raise TypeError(f"{label} must be numeric")
    checked = float(value)
    if not math.isfinite(checked):
        raise PreprocessingError(f"{label} must be finite")
    return float(_wrapped_angle(checked))


def canonicalize_group(
    positions: np.ndarray,
    root_yaw: np.ndarray,
    *,
    observed_mask: np.ndarray,
    root_joint_index: int = 0,
    reference_frame: int | None = None,
    shared_yaw: float = 0.0,
    augmentation_yaw: float = 0.0,
) -> CanonicalGroupMotion:
    """Center on the group pelvis and apply only explicit group-wide yaw angles.

    ``positions`` is ``[T,K,J,3]`` and ``root_yaw`` is ``[T,K]`` in radians.
    ``shared_yaw`` is an audited scene yaw to remove; ``augmentation_yaw`` is a
    training angle to add.  Neither value is inferred from an indexed actor or
    from an average of actor orientations.
    """

    motion = _float_array(positions, "positions", dimensions=4)
    yaw = _float_array(root_yaw, "root_yaw", dimensions=2)
    if motion.shape[-1] != 3:
        raise PreprocessingError("positions must end in xyz coordinates")
    frame_count, participant_count, joint_count, _ = motion.shape
    if participant_count < MINIMUM_GROUP_SIZE:
        raise PreprocessingError("positions must contain K>=3 participants")
    if yaw.shape != (frame_count, participant_count):
        raise PreprocessingError("root_yaw shape must equal [T,K]")
    mask = _observed_mask(
        observed_mask,
        frame_count=frame_count,
        participant_count=participant_count,
    )
    if type(root_joint_index) is not int or not 0 <= root_joint_index < joint_count:
        raise PreprocessingError("root_joint_index is outside the joint axis")
    if reference_frame is None:
        complete = np.flatnonzero(mask.all(axis=1))
        if complete.size == 0:
            raise PreprocessingError("no frame observes the full group")
        reference = int(complete[0])
    else:
        if type(reference_frame) is not int or not 0 <= reference_frame < frame_count:
            raise PreprocessingError("reference_frame is outside the time axis")
        reference = reference_frame
        if not bool(mask[reference].all()):
            raise PreprocessingError("reference_frame must observe every participant")

    scene_yaw = _finite_angle(shared_yaw, "shared_yaw")
    train_yaw = _finite_angle(augmentation_yaw, "augmentation_yaw")
    applied_yaw = float(_wrapped_angle(train_yaw - scene_yaw))
    center = np.asarray(
        motion[reference, :, root_joint_index, :].mean(axis=0, dtype=np.float64),
        dtype=np.float64,
    )
    translated = np.asarray(motion, dtype=np.float64) - center.reshape(1, 1, 1, 3)
    cosine = math.cos(applied_yaw)
    sine = math.sin(applied_yaw)
    rotated = np.empty_like(translated)
    rotated[..., 0] = cosine * translated[..., 0] + sine * translated[..., 2]
    rotated[..., 1] = translated[..., 1]
    rotated[..., 2] = -sine * translated[..., 0] + cosine * translated[..., 2]
    rotated_yaw = _wrapped_angle(np.asarray(yaw, dtype=np.float64) + applied_yaw)

    output_dtype = np.float32 if motion.dtype == np.dtype(np.float32) else np.float64
    output_positions = np.asarray(rotated, dtype=output_dtype)
    yaw_dtype = np.float32 if yaw.dtype == np.dtype(np.float32) else np.float64
    output_yaw = np.asarray(rotated_yaw, dtype=yaw_dtype)
    output_center = np.asarray(center, dtype=output_dtype)
    output_positions.setflags(write=False)
    output_yaw.setflags(write=False)
    output_center.setflags(write=False)
    return CanonicalGroupMotion(
        positions=output_positions,
        root_yaw=output_yaw,
        group_center=output_center,
        shared_yaw=scene_yaw,
        augmentation_yaw=train_yaw,
        applied_yaw=applied_yaw,
        reference_frame=reference,
    )


__all__ = [
    "CanonicalGroupMotion",
    "DEFAULT_CUTOFF_HZ",
    "DEFAULT_FIR_TAPS",
    "InterpolatedWindow",
    "MAX_MISSING_RUN_SOURCE_FRAMES",
    "MAX_MISSING_SECONDS",
    "MINIMUM_GROUP_SIZE",
    "MINIMUM_VALID_FRACTION",
    "PreprocessingError",
    "ResampledMotion",
    "SOURCE_FPS",
    "SOURCE_WINDOW_FRAMES",
    "TARGET_FPS",
    "TARGET_WINDOW_FRAMES",
    "WindowDecision",
    "assess_missing_window",
    "canonicalize_group",
    "design_antialias_fir",
    "interpolate_accepted_gaps",
    "resample_30_to_20",
]
