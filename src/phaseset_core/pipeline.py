"""Concrete, lineage-preserving preparation seam for private group captures.

The public package never knows participant names or licensed dataset paths. A
trusted host converts one capture to a numeric ``.npz`` seam containing only
joint positions, tracking flags, and salted 32-byte actor commitments. This
module then performs the frozen 30-to-20 Hz, 22-body-joint preparation without
silently dropping actors or edges.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import hmac
import math
from pathlib import Path
from typing import Iterable

import numpy as np

from phaseset_core.contracts import PreparedGroupBatch, group_commitment
from phaseset_core.preprocessing import (
    SOURCE_FPS,
    SOURCE_WINDOW_FRAMES,
    TARGET_WINDOW_FRAMES,
    PreprocessingError,
    WindowDecision,
    canonicalize_group,
    resample_30_to_20,
)


SMPLX_BODY22_INDICES = tuple(range(22))
PRIVATE_CAPTURE_KEYS = frozenset({"joints", "track_mask", "actor_commitments"})
MAX_PRIVATE_CAPTURE_BYTES = 64 * 1024 * 1024 * 1024


class PhaseSetPipelineError(ValueError):
    """A private numeric seam or prepared sample violates the frozen contract."""


class PreparationResourceLimit(RuntimeError):
    """The requested all-edge computation exceeds an explicit caller budget."""

    code = "RESOURCE_LIMIT"


def _readonly(value: np.ndarray) -> np.ndarray:
    output = np.ascontiguousarray(value)
    output.setflags(write=False)
    return output


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _bytes32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise PhaseSetPipelineError(f"{label} must be exact bytes32")
    return value


def derive_actor_commitment(private_salt: object, private_actor_id: object) -> bytes:
    """Derive a domain-separated, non-public actor lineage commitment.

    The salt is intentionally required to be at least 256 bits. Raw actor IDs
    are never stored on a prepared sample or emitted by any serializer here.
    """

    if type(private_salt) is not bytes or len(private_salt) < 32:
        raise PhaseSetPipelineError("private_salt must contain at least 32 bytes")
    if type(private_actor_id) is not str or not private_actor_id or "\x00" in private_actor_id:
        raise PhaseSetPipelineError("private_actor_id must be a nonempty NUL-free string")
    payload = b"phaseset-private-actor-v1\x00" + private_actor_id.encode("utf-8")
    return hmac.new(private_salt, payload, hashlib.sha256).digest()


def deterministic_group_yaw(
    *,
    seed: object,
    epoch: object,
    window_ordinal: object,
) -> float:
    """Return one deterministic group-wide training yaw in ``[-pi, pi)``.

    Augmentation is bound only to the registered RNG stream and the manifest
    window ordinal. Actor/group commitments remain lineage and reduction-order
    metadata and therefore cannot influence a neural input through this seam.
    """

    if type(seed) is not int or not 0 <= seed < 2**64:
        raise PhaseSetPipelineError("seed must be an exact uint64")
    if type(epoch) is not int or not 0 <= epoch < 2**64:
        raise PhaseSetPipelineError("epoch must be an exact uint64")
    if type(window_ordinal) is not int or not 0 <= window_ordinal < 2**64:
        raise PhaseSetPipelineError("window_ordinal must be an exact uint64")
    raw = hashlib.sha256(
        b"phaseset-group-yaw-v2\x00"
        + seed.to_bytes(8, "big")
        + epoch.to_bytes(8, "big")
        + window_ordinal.to_bytes(8, "big")
    ).digest()
    unit = (int.from_bytes(raw[:8], "big") + 0.5) / 2**64
    return float(unit * (2.0 * math.pi) - math.pi)


def _capture_lineage_sha256(capture: PrivateCaptureArrays) -> str:
    if capture.source_sha256 is not None:
        return capture.source_sha256
    digest = hashlib.sha256(b"phaseset-in-memory-capture-lineage-v1\x00")
    for array in (capture.joints, capture.track_mask):
        contiguous = np.ascontiguousarray(array)
        digest.update(contiguous.dtype.str.encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=">u8").tobytes())
        digest.update(contiguous.tobytes(order="C"))
    for commitment in capture.actor_commitments:
        digest.update(commitment)
    return digest.hexdigest()


def _window_lineage_sha256(source_sha256: str, source_start_frame: int) -> str:
    return hashlib.sha256(
        b"phaseset-source-window-lineage-v1\x00"
        + source_sha256.encode("ascii")
        + source_start_frame.to_bytes(8, "big")
        + SOURCE_WINDOW_FRAMES.to_bytes(8, "big")
    ).hexdigest()


@dataclass(frozen=True, slots=True)
class PrivateCaptureArrays:
    """One private numeric capture after the licensed-format conversion seam.

    ``joints`` is ``[T,K,J,3]`` at 30 Hz with ``J>=22``. ``track_mask`` is
    ``[T,K,J]``. Non-finite positions are allowed only at untracked joints.
    """

    joints: np.ndarray
    track_mask: np.ndarray
    actor_commitments: tuple[bytes, ...]
    source_sha256: str | None = None

    def __post_init__(self) -> None:
        if type(self.joints) is not np.ndarray:
            raise TypeError("joints must be an exact numpy.ndarray")
        if self.joints.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
            raise PhaseSetPipelineError("joints must be float32 or float64")
        if self.joints.ndim != 4 or self.joints.shape[-1] != 3:
            raise PhaseSetPipelineError("joints must have shape [T,K,J,3]")
        frame_count, actor_count, joint_count, _ = self.joints.shape
        if frame_count < 1 or actor_count < 2 or joint_count < len(SMPLX_BODY22_INDICES):
            raise PhaseSetPipelineError("capture must contain T>=1, K>=2, and J>=22")
        if type(self.track_mask) is not np.ndarray:
            raise TypeError("track_mask must be an exact numpy.ndarray")
        if self.track_mask.dtype != np.dtype(np.bool_) or self.track_mask.shape != (
            frame_count,
            actor_count,
            joint_count,
        ):
            raise PhaseSetPipelineError("track_mask must be bool [T,K,J]")
        if type(self.actor_commitments) is not tuple or len(self.actor_commitments) != actor_count:
            raise PhaseSetPipelineError("actor commitments must cover exactly K actors")
        commitments = tuple(
            _bytes32(value, f"actor commitment {index}")
            for index, value in enumerate(self.actor_commitments)
        )
        if len(set(commitments)) != actor_count:
            raise PhaseSetPipelineError("actor commitments must be unique")
        finite = np.isfinite(self.joints)
        if not bool(finite[self.track_mask[..., None].repeat(3, axis=-1)].all()):
            raise PhaseSetPipelineError("a tracked joint contains a non-finite coordinate")
        if self.source_sha256 is not None:
            if (
                type(self.source_sha256) is not str
                or len(self.source_sha256) != 64
                or any(character not in "0123456789abcdef" for character in self.source_sha256)
            ):
                raise PhaseSetPipelineError("source_sha256 must be lowercase SHA-256 hex")
        object.__setattr__(self, "joints", _readonly(np.array(self.joints, copy=True)))
        object.__setattr__(self, "track_mask", _readonly(np.array(self.track_mask, copy=True)))
        object.__setattr__(self, "actor_commitments", commitments)

    @property
    def actor_count(self) -> int:
        return int(self.joints.shape[1])

    @property
    def frame_count(self) -> int:
        return int(self.joints.shape[0])

    @property
    def group_commitment(self) -> bytes:
        return group_commitment(self.actor_commitments)


@dataclass(frozen=True, slots=True)
class PreparedGroupSample:
    """One accepted 20 Hz window before dynamic batch padding."""

    skeletons: np.ndarray
    track_mask: np.ndarray
    actor_commitments: tuple[bytes, ...]
    group_commitment: bytes
    source_sha256: str
    window_sha256: str
    source_start_frame: int
    augmentation_yaw: float
    decision: WindowDecision

    def __post_init__(self) -> None:
        if type(self.skeletons) is not np.ndarray:
            raise TypeError("skeletons must be an exact numpy.ndarray")
        if (
            self.skeletons.dtype != np.dtype(np.float32)
            or self.skeletons.ndim != 4
            or self.skeletons.shape[1:] != (TARGET_WINDOW_FRAMES, 22, 3)
            or self.skeletons.shape[0] < 2
        ):
            raise PhaseSetPipelineError("skeletons must be float32 [K,200,22,3], K>=2")
        if type(self.track_mask) is not np.ndarray:
            raise TypeError("track_mask must be an exact numpy.ndarray")
        if self.track_mask.dtype != np.dtype(np.bool_) or self.track_mask.shape != (
            self.skeletons.shape[0],
            TARGET_WINDOW_FRAMES,
            22,
        ):
            raise PhaseSetPipelineError("track_mask must be bool [K,200,22]")
        if not bool(np.isfinite(self.skeletons).all()):
            raise PhaseSetPipelineError("prepared skeletons must be finite")
        raw = self.skeletons.view(np.uint32)
        if bool((raw[~self.track_mask[..., None].repeat(3, axis=-1)] != 0).any()):
            raise PhaseSetPipelineError("untracked prepared values must be exact +0")
        if type(self.actor_commitments) is not tuple or len(self.actor_commitments) != len(
            self.skeletons
        ):
            raise PhaseSetPipelineError("prepared actor commitments must cover K actors")
        commitments = tuple(
            _bytes32(value, f"prepared actor commitment {index}")
            for index, value in enumerate(self.actor_commitments)
        )
        if group_commitment(commitments) != _bytes32(
            self.group_commitment, "prepared group commitment"
        ):
            raise PhaseSetPipelineError("prepared group commitment mismatch")
        for name in ("source_sha256", "window_sha256"):
            value = getattr(self, name)
            if (
                type(value) is not str
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise PhaseSetPipelineError(f"{name} must be lowercase SHA-256 hex")
        if type(self.source_start_frame) is not int or self.source_start_frame < 0:
            raise PhaseSetPipelineError("source_start_frame must be an exact nonnegative int")
        if isinstance(self.augmentation_yaw, bool) or not isinstance(
            self.augmentation_yaw, (int, float)
        ):
            raise PhaseSetPipelineError("augmentation_yaw must be numeric")
        if not math.isfinite(float(self.augmentation_yaw)):
            raise PhaseSetPipelineError("augmentation_yaw must be finite")
        if type(self.decision) is not WindowDecision or not self.decision.accepted:
            raise PhaseSetPipelineError("prepared sample requires an accepted decision")
        object.__setattr__(self, "skeletons", _readonly(np.array(self.skeletons, copy=True)))
        object.__setattr__(self, "track_mask", _readonly(np.array(self.track_mask, copy=True)))
        object.__setattr__(self, "actor_commitments", commitments)

    @property
    def actor_count(self) -> int:
        return int(self.skeletons.shape[0])

    @property
    def edge_count(self) -> int:
        return self.actor_count * (self.actor_count - 1) // 2


@dataclass(frozen=True, slots=True)
class RejectedWindow:
    source_start_frame: int
    reason: str

    def __post_init__(self) -> None:
        if type(self.source_start_frame) is not int or self.source_start_frame < 0:
            raise PhaseSetPipelineError("rejected window start must be nonnegative")
        if type(self.reason) is not str or not self.reason:
            raise PhaseSetPipelineError("rejected window reason must be nonempty")


@dataclass(frozen=True, slots=True)
class CapturePreparation:
    accepted: tuple[PreparedGroupSample, ...]
    rejected: tuple[RejectedWindow, ...]
    trailing_source_frames: int

    def __post_init__(self) -> None:
        if type(self.accepted) is not tuple or type(self.rejected) is not tuple:
            raise TypeError("accepted and rejected must be exact tuples")
        if type(self.trailing_source_frames) is not int or not (
            0 <= self.trailing_source_frames < SOURCE_WINDOW_FRAMES
        ):
            raise PhaseSetPipelineError("trailing_source_frames is outside [0,300)")


def load_private_capture_npz(
    path: str | Path,
    *,
    max_bytes: int = MAX_PRIVATE_CAPTURE_BYTES,
) -> PrivateCaptureArrays:
    """Load the exact numeric private seam with ``allow_pickle=False``.

    Actor commitments are stored as ``uint8[K,32]``. The archive may not carry
    names, captions, paths, endpoints, or additional unregistered arrays.
    """

    if type(max_bytes) is not int or max_bytes <= 0:
        raise TypeError("max_bytes must be a positive exact int")
    source = Path(path)
    if source.suffix.lower() != ".npz" or not source.is_file():
        raise PhaseSetPipelineError("private capture seam must be a regular .npz file")
    size = source.stat().st_size
    if size <= 0 or size > max_bytes:
        raise PhaseSetPipelineError("private capture seam is empty or exceeds max_bytes")
    raw_digest = _sha256_file(source)
    try:
        with np.load(source, allow_pickle=False) as archive:
            if set(archive.files) != PRIVATE_CAPTURE_KEYS:
                raise PhaseSetPipelineError("private capture seam has an unregistered key")
            joints = np.array(archive["joints"], copy=True)
            track_mask = np.array(archive["track_mask"], copy=True)
            packed = np.array(archive["actor_commitments"], copy=True)
    except PhaseSetPipelineError:
        raise
    except (OSError, ValueError) as exc:
        raise PhaseSetPipelineError("private capture seam is not a safe numeric NPZ") from exc
    if packed.dtype != np.dtype(np.uint8) or packed.ndim != 2 or packed.shape[1] != 32:
        raise PhaseSetPipelineError("actor_commitments must be uint8 [K,32]")
    commitments = tuple(bytes(row.tolist()) for row in packed)
    return PrivateCaptureArrays(joints, track_mask, commitments, raw_digest)


def _target_joint_mask(source_mask: np.ndarray, target_timestamps: np.ndarray) -> np.ndarray:
    source_positions = target_timestamps * SOURCE_FPS
    lower = np.floor(source_positions).astype(np.int64)
    upper = np.minimum(lower + 1, source_mask.shape[0] - 1)
    output = np.array(source_mask[lower], dtype=np.bool_, copy=True)
    fractional = source_positions != lower
    output[fractional] &= source_mask[upper[fractional]]
    return output


def prepare_window(
    capture: PrivateCaptureArrays,
    *,
    source_start_frame: int,
    augmentation_yaw: float = 0.0,
) -> PreparedGroupSample:
    """Prepare one exact non-overlapping 300-frame source window."""

    if type(capture) is not PrivateCaptureArrays:
        raise TypeError("capture must be exactly PrivateCaptureArrays")
    if type(source_start_frame) is not int or source_start_frame < 0:
        raise PhaseSetPipelineError("source_start_frame must be nonnegative")
    stop = source_start_frame + SOURCE_WINDOW_FRAMES
    if stop > capture.frame_count:
        raise PhaseSetPipelineError("source window exceeds capture frame count")
    if isinstance(augmentation_yaw, bool) or not isinstance(augmentation_yaw, (int, float)):
        raise TypeError("augmentation_yaw must be numeric")
    train_yaw = float(augmentation_yaw)
    if not math.isfinite(train_yaw):
        raise PhaseSetPipelineError("augmentation_yaw must be finite")

    # Canonicalize private actor order before every floating-point reduction so
    # a caller permutation cannot perturb the group-centroid summation order.
    actor_order = tuple(
        sorted(range(capture.actor_count), key=lambda index: capture.actor_commitments[index])
    )
    actor_order_array = np.asarray(actor_order, dtype=np.int64)
    joints = capture.joints[source_start_frame:stop][:, actor_order_array][
        :, :, SMPLX_BODY22_INDICES, :
    ]
    joint_mask = capture.track_mask[source_start_frame:stop][:, actor_order_array][
        :, :, SMPLX_BODY22_INDICES
    ]
    ordered_commitments = tuple(capture.actor_commitments[index] for index in actor_order)
    actor_observed = joint_mask[..., 0]
    if not np.array_equal(joint_mask, np.broadcast_to(actor_observed[..., None], joint_mask.shape)):
        raise PhaseSetPipelineError(
            "body-22 track mask must be uniform within each actor-frame"
        )
    resampled = resample_30_to_20(joints, observed_mask=actor_observed)
    target_joint_mask = _target_joint_mask(joint_mask, resampled.timestamps_seconds)
    target_actor_mask = target_joint_mask.all(axis=2)
    if not np.array_equal(target_actor_mask, resampled.target_observed_mask):
        raise AssertionError("joint and actor target masks diverged")
    canonical = canonicalize_group(
        resampled.values,
        np.zeros((TARGET_WINDOW_FRAMES, capture.actor_count), dtype=np.float32),
        observed_mask=target_actor_mask,
        augmentation_yaw=train_yaw,
    )
    masked = np.where(target_joint_mask[..., None], canonical.positions, 0.0).astype(
        np.float32,
        copy=False,
    )
    skeletons = np.ascontiguousarray(masked.transpose(1, 0, 2, 3))
    prepared_mask = np.ascontiguousarray(target_joint_mask.transpose(1, 0, 2))
    source_sha256 = _capture_lineage_sha256(capture)
    return PreparedGroupSample(
        skeletons=skeletons,
        track_mask=prepared_mask,
        actor_commitments=ordered_commitments,
        group_commitment=capture.group_commitment,
        source_sha256=source_sha256,
        window_sha256=_window_lineage_sha256(source_sha256, source_start_frame),
        source_start_frame=source_start_frame,
        augmentation_yaw=train_yaw,
        decision=resampled_to_decision(resampled),
    )


def resampled_to_decision(resampled: object) -> WindowDecision:
    """Recover the immutable source decision carried by a resampling result."""

    # ``resample_30_to_20`` exposes the retained mask and missing count rather
    # than the original decision object. Re-evaluating the immutable source
    # mask is deterministic and keeps this seam independent of private fields.
    from phaseset_core.preprocessing import ResampledMotion, assess_missing_window

    if type(resampled) is not ResampledMotion:
        raise TypeError("resampled must be exactly ResampledMotion")
    decision = assess_missing_window(resampled.source_observed_mask)
    if not decision.accepted or decision.missing_actor_frames != resampled.interpolated_actor_frames:
        raise AssertionError("resampling decision provenance diverged")
    return decision


def prepare_capture(
    capture: PrivateCaptureArrays,
    *,
    training: bool,
    seed: int = 0,
    epoch: int = 0,
    manifest_window_ordinal_start: int | None = None,
) -> CapturePreparation:
    """Prepare every full non-overlapping ten-second capture window."""

    if type(capture) is not PrivateCaptureArrays:
        raise TypeError("capture must be exactly PrivateCaptureArrays")
    if type(training) is not bool:
        raise TypeError("training must be an exact bool")
    if training:
        if (
            type(manifest_window_ordinal_start) is not int
            or manifest_window_ordinal_start < 0
            or manifest_window_ordinal_start >= 2**64
        ):
            raise PhaseSetPipelineError(
                "training requires a global uint64 manifest_window_ordinal_start"
            )
    elif manifest_window_ordinal_start is not None:
        raise PhaseSetPipelineError(
            "evaluation preparation cannot consume a training window ordinal"
        )
    accepted: list[PreparedGroupSample] = []
    rejected: list[RejectedWindow] = []
    full_windows = capture.frame_count // SOURCE_WINDOW_FRAMES
    for window_index in range(full_windows):
        start = window_index * SOURCE_WINDOW_FRAMES
        yaw = (
            deterministic_group_yaw(
                seed=seed,
                epoch=epoch,
                window_ordinal=manifest_window_ordinal_start + window_index,
            )
            if training
            else 0.0
        )
        try:
            accepted.append(
                prepare_window(capture, source_start_frame=start, augmentation_yaw=yaw)
            )
        except PreprocessingError as exc:
            reason = str(exc).rsplit(": ", maxsplit=1)[-1]
            rejected.append(RejectedWindow(start, reason))
    return CapturePreparation(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        trailing_source_frames=capture.frame_count % SOURCE_WINDOW_FRAMES,
    )


def collate_group_samples(samples: object) -> PreparedGroupBatch:
    """Dynamically pad one nonempty sample tuple and preserve exact +0 padding."""

    if type(samples) is not tuple or not samples:
        raise TypeError("samples must be a nonempty exact tuple")
    if any(type(sample) is not PreparedGroupSample for sample in samples):
        raise TypeError("every sample must be exactly PreparedGroupSample")
    batch_size = len(samples)
    padded_actors = max(sample.actor_count for sample in samples)
    skeletons = np.zeros(
        (batch_size, padded_actors, TARGET_WINDOW_FRAMES, 22, 3),
        dtype=np.float32,
    )
    track_mask = np.zeros(
        (batch_size, padded_actors, TARGET_WINDOW_FRAMES, 22),
        dtype=np.bool_,
    )
    actor_mask = np.zeros((batch_size, padded_actors), dtype=np.bool_)
    frame_mask = np.ones((batch_size, TARGET_WINDOW_FRAMES), dtype=np.bool_)
    commitments: list[tuple[bytes | None, ...]] = []
    groups: list[bytes] = []
    for row, sample in enumerate(samples):
        count = sample.actor_count
        skeletons[row, :count] = sample.skeletons
        track_mask[row, :count] = sample.track_mask
        actor_mask[row, :count] = True
        commitments.append(sample.actor_commitments + (None,) * (padded_actors - count))
        groups.append(sample.group_commitment)
    return PreparedGroupBatch(
        skeletons=np.ascontiguousarray(skeletons),
        actor_mask=np.ascontiguousarray(actor_mask),
        frame_mask=np.ascontiguousarray(frame_mask),
        track_mask=np.ascontiguousarray(track_mask),
        actor_commitments=tuple(commitments),
        group_commitments=tuple(groups),
    )


def edge_budget_batches(
    samples: object,
    *,
    max_total_edges: int,
    max_batch_size: int | None = None,
) -> tuple[tuple[PreparedGroupSample, ...], ...]:
    """Return deterministic K-bucketed batches without sampling any edge."""

    if type(samples) is not tuple:
        raise TypeError("samples must be an exact tuple")
    if any(type(sample) is not PreparedGroupSample for sample in samples):
        raise TypeError("every sample must be exactly PreparedGroupSample")
    if type(max_total_edges) is not int or max_total_edges < 1:
        raise PhaseSetPipelineError("max_total_edges must be a positive exact int")
    if max_batch_size is not None and (
        type(max_batch_size) is not int or max_batch_size < 1
    ):
        raise PhaseSetPipelineError("max_batch_size must be None or a positive exact int")
    ordered = tuple(
        sorted(
            samples,
            key=lambda sample: (
                sample.actor_count,
                sample.group_commitment,
                sample.source_start_frame,
            ),
        )
    )
    batches: list[tuple[PreparedGroupSample, ...]] = []
    current: list[PreparedGroupSample] = []
    current_edges = 0
    for sample in ordered:
        if sample.edge_count > max_total_edges:
            raise PreparationResourceLimit(
                f"RESOURCE_LIMIT: sample requires {sample.edge_count} edges, "
                f"budget is {max_total_edges}"
            )
        batch_full = max_batch_size is not None and len(current) >= max_batch_size
        edge_full = current_edges + sample.edge_count > max_total_edges
        if current and (batch_full or edge_full):
            batches.append(tuple(current))
            current = []
            current_edges = 0
        current.append(sample)
        current_edges += sample.edge_count
    if current:
        batches.append(tuple(current))
    return tuple(batches)


def total_edge_count(samples: Iterable[PreparedGroupSample]) -> int:
    """Count exact unordered edges without materializing an actor-pair tensor."""

    total = 0
    for sample in samples:
        if type(sample) is not PreparedGroupSample:
            raise TypeError("every sample must be exactly PreparedGroupSample")
        total += sample.edge_count
    return total


__all__ = [
    "CapturePreparation",
    "MAX_PRIVATE_CAPTURE_BYTES",
    "PRIVATE_CAPTURE_KEYS",
    "PhaseSetPipelineError",
    "PreparationResourceLimit",
    "PreparedGroupSample",
    "PrivateCaptureArrays",
    "RejectedWindow",
    "SMPLX_BODY22_INDICES",
    "collate_group_samples",
    "derive_actor_commitment",
    "deterministic_group_yaw",
    "edge_budget_batches",
    "load_private_capture_npz",
    "prepare_capture",
    "prepare_window",
    "resampled_to_decision",
    "total_edge_count",
]
