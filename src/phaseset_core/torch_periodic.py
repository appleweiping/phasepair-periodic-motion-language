"""Differentiable CPU oracle for the skeleton-to-periodic PhaseSet seam.

The production edge streamer deliberately consumes audited NumPy descriptors
and recomputes only neural edge activations in backward.  This module is a
separate, explicit oracle: it keeps skeleton motion in Torch autograd while
reusing the frozen 20-Hz Morlet taps and the inherited 13D relation algebra.
It is intended for gradient qualification and small diagnostic batches, not as
a claim that the NumPy production descriptor path is differentiable.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final, Protocol

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phasepair_core import signal as legacy_signal

from .contracts import (
    ACTIVITY_JOINT_GROUPS,
    ACTIVITY_SAMPLE_RATE_HZ,
    COORDINATE_DIM,
    JOINT_COUNT,
    MAX_PADDED_TIME,
    MAX_VALID_TIME,
)
from .morlet import morlet_kernel_bank
from .periodic import BAND_COUNT, TOKEN_WIDTH, validate_energy_floors
from .streaming_autograd import StreamingEdgeSummaries


STATUS: Final = "DATA_FREE_DIFFERENTIABLE_CPU_ORACLE_AUTHORITY0"


class TorchPeriodicOracleError(ValueError):
    """Raised when the differentiable oracle receives an invalid tensor batch."""


class _EdgeModel(Protocol):
    embedding_dim: int
    half_edge_encoder: nn.Module
    pair_encoder: nn.Module

    def _edge_forward(
        self,
        relation_ij: Tensor,
        relation_ji: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]: ...


@dataclass(frozen=True, slots=True)
class TorchActivityBatch:
    """Validated differentiable activity plus structural masks."""

    activities: Tensor
    actor_mask: Tensor
    frame_mask: Tensor
    activity_mask: Tensor


def _require_cpu_tensor(
    value: object,
    *,
    name: str,
    dtype: torch.dtype,
    shape: tuple[int, ...] | None = None,
) -> Tensor:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a torch.Tensor")
    if value.device.type != "cpu" or value.dtype != dtype or not value.is_contiguous():
        raise TorchPeriodicOracleError(
            f"{name} must be a contiguous CPU {dtype} tensor"
        )
    if shape is not None and tuple(value.shape) != shape:
        raise TorchPeriodicOracleError(f"{name} has an invalid shape")
    return value


def _require_positive_zero(value: Tensor, *, name: str) -> None:
    if value.numel() == 0:
        return
    bits = value.detach().view(torch.int32)
    if bool(torch.any(bits != 0).item()):
        raise TorchPeriodicOracleError(f"{name} must be exact positive float32 zero")


def torch_skeleton_to_activity(
    skeletons: Tensor,
    actor_mask: Tensor,
    frame_mask: Tensor,
    track_mask: Tensor,
) -> TorchActivityBatch:
    """Derive the inherited five speed channels without leaving autograd.

    Actor commitments are intentionally absent: they are lineage/order keys,
    never neural inputs.  The caller may place valid actors in arbitrary tensor
    positions; invalid actors, invalid frames, and untracked coordinates must
    be exact positive zero just as in :class:`PreparedGroupBatch`.
    """

    values = _require_cpu_tensor(
        skeletons,
        name="skeletons",
        dtype=torch.float32,
    )
    if values.ndim != 5:
        raise TorchPeriodicOracleError(
            "skeletons must have shape [B,K_pad,T,22,3]"
        )
    batch_size, padded_actors, padded_time, joints, coordinates = values.shape
    if (
        batch_size < 1
        or padded_actors < 2
        or not 1 <= padded_time <= MAX_PADDED_TIME
        or joints != JOINT_COUNT
        or coordinates != COORDINATE_DIM
    ):
        raise TorchPeriodicOracleError(
            "skeletons must have B>=1, K_pad>=2, 1<=T<=300, J=22, C=3"
        )
    actors = _require_cpu_tensor(
        actor_mask,
        name="actor_mask",
        dtype=torch.bool,
        shape=(batch_size, padded_actors),
    )
    frames = _require_cpu_tensor(
        frame_mask,
        name="frame_mask",
        dtype=torch.bool,
        shape=(batch_size, padded_time),
    )
    tracks = _require_cpu_tensor(
        track_mask,
        name="track_mask",
        dtype=torch.bool,
        shape=(batch_size, padded_actors, padded_time, JOINT_COUNT),
    )
    if not bool(torch.isfinite(values.detach()).all().item()):
        raise TorchPeriodicOracleError("skeletons must be finite")

    for row in range(batch_size):
        actor_count = int(actors[row].sum().item())
        valid_length = int(frames[row].sum().item())
        if actor_count < 2:
            raise TorchPeriodicOracleError("every group must contain at least two actors")
        if not 1 <= valid_length <= min(MAX_VALID_TIME, padded_time):
            raise TorchPeriodicOracleError("each group must contain 1..299 valid frames")
        expected_frames = torch.arange(padded_time) < valid_length
        if not torch.equal(frames[row], expected_frames):
            raise TorchPeriodicOracleError("frame_mask must be a contiguous prefix")
        for actor in range(padded_actors):
            if not bool(actors[row, actor].item()):
                _require_positive_zero(
                    values[row, actor], name="invalid actor skeleton padding"
                )
                if bool(tracks[row, actor].any().item()):
                    raise TorchPeriodicOracleError(
                        "invalid actor track_mask padding must be false"
                    )
                continue
            _require_positive_zero(
                values[row, actor, valid_length:],
                name="invalid frame skeleton padding",
            )
            if bool(tracks[row, actor, valid_length:].any().item()):
                raise TorchPeriodicOracleError(
                    "invalid frame track_mask padding must be false"
                )
            missing = ~tracks[row, actor]
            _require_positive_zero(
                values[row, actor][missing],
                name="untracked joint skeleton values",
            )
            if not bool(tracks[row, actor, :valid_length].any().item()):
                raise TorchPeriodicOracleError(
                    "every valid actor must contain at least one tracked joint"
                )

    activity_rows: list[Tensor] = []
    activity_mask_rows: list[Tensor] = []
    consecutive_frames = frames[:, 1:] & frames[:, :-1]
    for joints_in_channel in ACTIVITY_JOINT_GROUPS:
        joint_indices = torch.tensor(joints_in_channel, dtype=torch.int64)
        observed = (
            tracks[:, :, 1:, joint_indices]
            & tracks[:, :, :-1, joint_indices]
            & actors[:, :, None, None]
            & consecutive_frames[:, None, :, None]
        )
        displacement = (
            values[:, :, 1:, joint_indices]
            - values[:, :, :-1, joint_indices]
        ).to(dtype=torch.float64)
        speed = torch.linalg.vector_norm(displacement, dim=-1)
        observed_count = observed.sum(dim=-1)
        safe_count = observed_count.clamp_min(1).to(dtype=torch.float64)
        channel = (
            torch.where(observed, speed, torch.zeros_like(speed)).sum(dim=-1)
            / safe_count
            * ACTIVITY_SAMPLE_RATE_HZ
        )
        channel_valid = observed_count > 0
        channel = torch.where(channel_valid, channel, torch.zeros_like(channel))
        # The production contract stores float32 activity before applying its
        # float64 Morlet oracle.  Preserve that quantization while retaining the
        # straight-through cast in autograd.
        channel = channel.to(dtype=torch.float32)
        activity_rows.append(
            F.pad(channel, (1, 0), mode="constant", value=0.0)
        )
        activity_mask_rows.append(
            F.pad(channel_valid, (1, 0), mode="constant", value=False)
        )
    activities = torch.stack(activity_rows, dim=-1).contiguous()
    activity_masks = torch.stack(activity_mask_rows, dim=-1).contiguous()
    return TorchActivityBatch(activities, actors, frames, activity_masks)


def _morlet_response(activity: Tensor, kernel: np.ndarray) -> Tensor:
    """Return the same sliding complex dot as the NumPy oracle."""

    source = activity.to(dtype=torch.float64).transpose(0, 1)[:, None, :]
    real_kernel = torch.from_numpy(np.ascontiguousarray(kernel.real))[None, None]
    imag_kernel = torch.from_numpy(np.ascontiguousarray(kernel.imag))[None, None]
    real = F.conv1d(source, real_kernel)
    imaginary = F.conv1d(source, imag_kernel)
    return torch.complex(real[:, 0], imaginary[:, 0]).transpose(0, 1)


def _response_mask(mask: Tensor, kernel_length: int) -> Tensor:
    return mask.unfold(0, kernel_length, 1).all(dim=-1)


def _directed_tokens(
    left: Tensor,
    right: Tensor,
    left_mask: Tensor,
    right_mask: Tensor,
    *,
    frequency_hz: float,
    energy_floor: float,
    band_index: int,
) -> tuple[Tensor, Tensor, Tensor]:
    joint_mask = left_mask & right_mask
    count = int(joint_mask.sum().item())
    zero = left.real.sum() * 0.0 + right.real.sum() * 0.0
    if count:
        selected_left = torch.where(joint_mask, left, torch.zeros_like(left))
        selected_right = torch.where(joint_mask, right, torch.zeros_like(right))
        auto_left = (selected_left.abs().square().sum() / count).real
        auto_right = (selected_right.abs().square().sum() / count).real
        cross = (selected_left.conj() * selected_right).sum() / count
    else:
        auto_left = zero
        auto_right = zero
        cross = torch.complex(zero, zero)

    epsilon = float(legacy_signal.TOKEN_EPSILON)
    cross_abs = cross.abs()
    energy_valid = (auto_left > energy_floor) & (auto_right > energy_floor)
    relation_valid = cross_abs > epsilon * torch.sqrt(auto_left * auto_right)
    safe_real = torch.where(relation_valid, cross.real, torch.ones_like(cross.real))
    safe_imag = torch.where(relation_valid, cross.imag, torch.zeros_like(cross.imag))
    phase = torch.atan2(safe_imag, safe_real)
    branch_valid = relation_valid & ((math.pi - phase.abs()).abs() > 1e-6)
    coherence = (
        cross_abs.square()
        / ((auto_left + epsilon) * (auto_right + epsilon))
    ).clamp(0.0, 1.0)
    cosine = torch.where(relation_valid, torch.cos(phase), torch.ones_like(phase))
    sine = torch.where(branch_valid, torch.sin(phase), torch.zeros_like(phase))
    delay_feature = torch.where(
        branch_valid,
        -phase / math.pi,
        torch.zeros_like(phase),
    )
    signed_valid = branch_valid.to(dtype=torch.float64)
    one_hot = torch.zeros((BAND_COUNT,), dtype=torch.float64)
    one_hot[band_index] = 1.0
    forward = torch.cat(
        (
            torch.stack(
                (
                    torch.log(auto_left + epsilon),
                    torch.log(auto_right + epsilon),
                    coherence,
                    cosine,
                    sine,
                    delay_feature,
                    signed_valid,
                )
            ),
            one_hot,
        )
    )
    reverse = torch.cat(
        (
            torch.stack(
                (
                    torch.log(auto_right + epsilon),
                    torch.log(auto_left + epsilon),
                    coherence,
                    cosine,
                    -sine,
                    -delay_feature,
                    signed_valid,
                )
            ),
            one_hot,
        )
    )
    if forward.shape != (TOKEN_WIDTH,) or reverse.shape != (TOKEN_WIDTH,):
        raise AssertionError("differentiable descriptor width drift")
    return forward.to(dtype=torch.float32), reverse.to(dtype=torch.float32), energy_valid


def _fixed_sum(values: list[Tensor]) -> Tensor:
    if not values:
        raise ValueError("fixed sum requires at least one tensor")
    level = values
    while len(level) > 1:
        next_level = [
            level[index] + level[index + 1]
            for index in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def differentiable_edge_summaries(
    model: _EdgeModel,
    skeletons: Tensor,
    actor_mask: Tensor,
    frame_mask: Tensor,
    track_mask: Tensor,
    *,
    energy_floors: np.ndarray,
    edge_budget: int,
) -> StreamingEdgeSummaries:
    """Compute small-batch periodic summaries with skeleton gradients."""

    batch = torch_skeleton_to_activity(
        skeletons,
        actor_mask,
        frame_mask,
        track_mask,
    )
    floors = validate_energy_floors(energy_floors)
    actor_counts = [int(row.sum().item()) for row in batch.actor_mask]
    required_edges = sum(count * (count - 1) // 2 for count in actor_counts)
    if required_edges > edge_budget:
        raise TorchPeriodicOracleError("RESOURCE_LIMIT")

    batch_size, padded_actors, _, _ = batch.activities.shape
    width = model.embedding_dim
    pair_rows: list[Tensor] = []
    statistic_rows: list[Tensor] = []
    pair_count_rows: list[Tensor] = []
    topology_mask_rows: list[Tensor] = []
    bands = morlet_kernel_bank()
    for group in range(batch_size):
        valid_positions = torch.nonzero(batch.actor_mask[group], as_tuple=False)[:, 0].tolist()
        valid_length = int(batch.frame_mask[group].sum().item())
        responses: list[list[Tensor | None]] = [[] for _ in valid_positions]
        response_masks: list[list[Tensor | None]] = [[] for _ in valid_positions]
        for actor_row, actor in enumerate(valid_positions):
            for band in bands:
                if valid_length < band.length:
                    responses[actor_row].append(None)
                    response_masks[actor_row].append(None)
                else:
                    responses[actor_row].append(
                        _morlet_response(
                            batch.activities[group, actor, :valid_length],
                            band.kernel,
                        )
                    )
                    response_masks[actor_row].append(
                        _response_mask(
                            batch.activity_mask[group, actor, :valid_length],
                            band.length,
                        )
                    )

        pair_by_band: list[list[Tensor]] = [[] for _ in range(BAND_COUNT)]
        half_by_actor_band: list[list[list[Tensor]]] = [
            [[] for _ in range(BAND_COUNT)] for _ in valid_positions
        ]
        linked_zero: Tensor | None = None
        for left in range(len(valid_positions)):
            for right in range(left + 1, len(valid_positions)):
                directed_left: list[Tensor] = []
                directed_right: list[Tensor] = []
                supports: list[Tensor] = []
                for band_index, band in enumerate(bands):
                    left_response = responses[left][band_index]
                    right_response = responses[right][band_index]
                    left_response_mask = response_masks[left][band_index]
                    right_response_mask = response_masks[right][band_index]
                    if (
                        left_response is None
                        or right_response is None
                        or left_response_mask is None
                        or right_response_mask is None
                    ):
                        zero_token = skeletons.sum() * 0.0
                        token_left = zero_token.expand(TOKEN_WIDTH)
                        token_right = zero_token.expand(TOKEN_WIDTH)
                        support = torch.tensor(False)
                    else:
                        token_left, token_right, support = _directed_tokens(
                            left_response,
                            right_response,
                            left_response_mask,
                            right_response_mask,
                            frequency_hz=band.frequency_hz,
                            energy_floor=float(floors[band_index]),
                            band_index=band_index,
                        )
                    directed_left.append(token_left)
                    directed_right.append(token_right)
                    supports.append(support)
                relation_left = torch.stack(directed_left)
                relation_right = torch.stack(directed_right)
                half_left, half_right, pair_tokens = model._edge_forward(
                    relation_left,
                    relation_right,
                )
                linked_zero = pair_tokens.sum() * 0.0
                for band_index, support in enumerate(supports):
                    if bool(support.item()):
                        pair_by_band[band_index].append(pair_tokens[band_index])
                        half_by_actor_band[left][band_index].append(half_left[band_index])
                        half_by_actor_band[right][band_index].append(half_right[band_index])

        if linked_zero is None:
            raise AssertionError("K>=2 must create at least one edge")
        pair_bands: list[Tensor] = []
        pair_counts: list[int] = []
        for values in pair_by_band:
            pair_counts.append(len(values))
            pair_bands.append(
                (_fixed_sum([value.to(torch.float64) for value in values]) / len(values)).to(torch.float32)
                if values
                else linked_zero.expand(width)
            )
        pair_rows.append(torch.stack(pair_bands))
        pair_count_rows.append(torch.tensor(pair_counts, dtype=torch.int64))

        global_half: list[list[Tensor]] = [[] for _ in range(BAND_COUNT)]
        for actor_values in half_by_actor_band:
            for band_index, values in enumerate(actor_values):
                global_half[band_index].extend(values)
        global_means = [
            _fixed_sum([value.to(torch.float64) for value in values]) / len(values)
            if values
            else linked_zero.to(torch.float64).expand(width)
            for values in global_half
        ]
        actor_statistics: list[Tensor] = []
        actor_topology_masks: list[Tensor] = []
        actor_count = len(valid_positions)
        valid_actor_lookup = {position: index for index, position in enumerate(valid_positions)}
        for padded_actor in range(padded_actors):
            physical_actor = valid_actor_lookup.get(padded_actor)
            band_statistics: list[Tensor] = []
            band_topology_masks: list[bool] = []
            for band_index in range(BAND_COUNT):
                values = (
                    half_by_actor_band[physical_actor][band_index]
                    if physical_actor is not None
                    else []
                )
                degree = len(values)
                if values:
                    values64 = [value.to(torch.float64) for value in values]
                    mean = _fixed_sum(values64) / degree
                    centered = [(value - mean).square() for value in values64]
                    moment = _fixed_sum(centered) / degree
                else:
                    mean = linked_zero.to(torch.float64).expand(width)
                    moment = linked_zero.to(torch.float64).expand(width)
                coverage = degree / (actor_count - 1) if physical_actor is not None else 0.0
                scalars = torch.tensor((coverage, 1.0 - coverage), dtype=torch.float64)
                band_statistics.append(
                    torch.cat((mean - global_means[band_index], moment, scalars)).to(torch.float32)
                )
                band_topology_masks.append(physical_actor is not None and degree >= 2)
            actor_statistics.append(torch.stack(band_statistics))
            actor_topology_masks.append(torch.tensor(band_topology_masks, dtype=torch.bool))
        statistic_rows.append(torch.stack(actor_statistics))
        topology_mask_rows.append(torch.stack(actor_topology_masks))

    return StreamingEdgeSummaries(
        pair_component=torch.stack(pair_rows),
        node_statistics=torch.stack(statistic_rows),
        valid_pair_count=torch.stack(pair_count_rows),
        topology_node_mask=torch.stack(topology_mask_rows),
    )


__all__ = [
    "STATUS",
    "TorchActivityBatch",
    "TorchPeriodicOracleError",
    "differentiable_edge_summaries",
    "torch_skeleton_to_activity",
]
