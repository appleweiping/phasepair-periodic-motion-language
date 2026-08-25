"""Permutation-invariant, streamed PhaseSet group-token encoder."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

from .contracts import (
    ACTIVITY_DIM,
    COORDINATE_DIM,
    JOINT_COUNT,
    PreparedActivityBatch,
    PreparedGroupBatch,
    skeleton_to_activity,
    validate_prepared_activity_batch,
    validate_prepared_group_batch,
)
from .periodic import (
    BAND_COUNT,
    CANONICAL_MICROBLOCK_SIZE,
    DEFAULT_EDGE_CHUNK_SIZE,
    ResourceLimitError,
    TOKEN_WIDTH,
    iter_unordered_pair_chunks,
    validate_energy_floors,
)


STATUS: Final = "DATA_FREE_PHASESET_MODEL_NONPRODUCTION_AUTHORITY0"
DEFAULT_EDGE_BUDGET: Final = 32_768


class PhaseSetModelError(ValueError):
    """Raised when model inputs violate the multi-person contract."""


@dataclass(frozen=True, slots=True)
class GroupTokenOutput:
    """Band-indexed group tokens and auditable decomposition."""

    tokens: Tensor
    band_mask: Tensor
    pair_component: Tensor
    topology_delta: Tensor
    valid_pair_count: Tensor
    topology_node_count: Tensor


# Compatibility alias retained for the alpha namespace.  The public contract
# name is GroupTokenOutput.
PhaseSetOutput = GroupTokenOutput


def _fixed_binary_tree_sum(values: list[Tensor]) -> Tensor:
    """Reduce one canonical microblock with a fixed adjacent binary tree."""

    if not values:
        raise ValueError("fixed-tree reduction requires at least one value")
    level = values
    while len(level) > 1:
        next_level: list[Tensor] = []
        for offset in range(0, len(level) - 1, 2):
            next_level.append(level[offset] + level[offset + 1])
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def _compensated_add(total: Tensor, correction: Tensor, value: Tensor) -> tuple[Tensor, Tensor]:
    """Neumaier-accumulate canonical microblock sums componentwise."""

    updated = total + value
    use_total = torch.abs(total) >= torch.abs(value)
    increment = torch.where(
        use_total,
        (total - updated) + value,
        (value - updated) + total,
    )
    return updated, correction + increment


def _chan_merge(
    left_count: int,
    left_mean: Tensor,
    left_m2: Tensor,
    right_count: int,
    right_mean: Tensor,
    right_m2: Tensor,
) -> tuple[int, Tensor, Tensor]:
    """Merge two population-moment states in a fixed order."""

    if left_count == 0:
        return right_count, right_mean, right_m2
    if right_count == 0:
        return left_count, left_mean, left_m2
    count = left_count + right_count
    delta = right_mean - left_mean
    mean = left_mean + delta * (right_count / count)
    m2 = (
        left_m2
        + right_m2
        + delta * delta * (left_count * right_count / count)
    )
    return count, mean, m2


def _fixed_tree_chan(values: list[Tensor]) -> tuple[int, Tensor, Tensor]:
    """Create one Chan state using the same fixed adjacent tree as pair sums."""

    if not values:
        raise ValueError("Chan reduction requires at least one value")
    states = [(1, value, torch.zeros_like(value)) for value in values]
    while len(states) > 1:
        next_states: list[tuple[int, Tensor, Tensor]] = []
        for offset in range(0, len(states) - 1, 2):
            next_states.append(_chan_merge(*states[offset], *states[offset + 1]))
        if len(states) % 2:
            next_states.append(states[-1])
        states = next_states
    return states[0]


class SharedActorEncoder(nn.Module):
    """One temporal encoder shared by every actor and every group size."""

    def __init__(self, embedding_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(ACTIVITY_DIM, embedding_dim)
        self.output_projection = nn.Linear(embedding_dim, embedding_dim)

    def forward(self, activities: Tensor, frame_mask: Tensor, actor_mask: Tensor) -> Tensor:
        if activities.ndim != 4 or activities.shape[-1] != ACTIVITY_DIM:
            raise PhaseSetModelError("activities must have shape [B,K,T,5]")
        hidden = self.output_projection(F.gelu(self.input_projection(activities)))
        valid = frame_mask[:, None, :, None] & actor_mask[:, :, None, None]
        hidden = torch.where(valid, hidden, torch.zeros_like(hidden))
        denominator = frame_mask.sum(dim=1, dtype=hidden.dtype).clamp_min(1.0)
        pooled = hidden.sum(dim=2) / denominator[:, None, None]
        return torch.where(actor_mask[..., None], pooled, torch.zeros_like(pooled))


@dataclass(frozen=True, slots=True)
class GroupBaseOutput:
    """Permutation-invariant base embedding plus auditable actor embeddings."""

    group_embedding: Tensor
    actor_embeddings: Tensor
    actor_mask: Tensor


def _validate_base_widths(
    hidden_dim: int,
    heads: int,
    ffn_dim: int,
    dropout: float,
) -> None:
    if type(hidden_dim) is not int or hidden_dim < 2:
        raise TypeError("hidden_dim must be an exact int >=2")
    if type(heads) is not int or heads < 1 or hidden_dim % heads:
        raise TypeError("heads must be an exact positive divisor of hidden_dim")
    if type(ffn_dim) is not int or ffn_dim < hidden_dim:
        raise TypeError("ffn_dim must be an exact int >=hidden_dim")
    if type(dropout) not in (float, int) or not 0.0 <= float(dropout) < 1.0:
        raise TypeError("dropout must be a finite built-in number in [0,1)")


def _sinusoidal_time_encoding(
    time_steps: int,
    width: int,
    *,
    device: torch.device,
) -> Tensor:
    positions = torch.arange(time_steps, dtype=torch.float32, device=device)[:, None]
    even_count = (width + 1) // 2
    exponents = torch.arange(even_count, dtype=torch.float32, device=device)
    scales = torch.exp(-math.log(10_000.0) * (2.0 * exponents / width))
    encoding = torch.zeros((time_steps, width), dtype=torch.float32, device=device)
    encoding[:, 0::2] = torch.sin(positions * scales)
    if width > 1:
        encoding[:, 1::2] = torch.cos(positions * scales[: width // 2])
    return encoding


def _public_skeleton_tensors(
    batch: PreparedGroupBatch,
    device: torch.device,
) -> tuple[PreparedGroupBatch, Tensor, Tensor, Tensor, Tensor]:
    checked = validate_prepared_group_batch(batch)
    skeletons = torch.tensor(
        checked.skeletons,
        dtype=torch.float32,
        device=device,
    ).contiguous()
    actor_mask = torch.tensor(
        checked.actor_mask,
        dtype=torch.bool,
        device=device,
    ).contiguous()
    frame_mask = torch.tensor(
        checked.frame_mask,
        dtype=torch.bool,
        device=device,
    ).contiguous()
    track_mask = torch.tensor(
        checked.track_mask,
        dtype=torch.bool,
        device=device,
    ).contiguous()
    return checked, skeletons, actor_mask, frame_mask, track_mask


def _skeleton_features(skeletons: Tensor, track_mask: Tensor) -> Tensor:
    batch_size, actors, time_steps, _, _ = skeletons.shape
    coordinates = skeletons.reshape(
        batch_size,
        actors,
        time_steps,
        JOINT_COUNT * COORDINATE_DIM,
    )
    return torch.cat((coordinates, track_mask.to(dtype=torch.float32)), dim=-1)


class SharedSkeletonTemporalEncoder(nn.Module):
    """One identity-free temporal Transformer shared by every valid actor."""

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_base_widths(hidden_dim, heads, ffn_dim, dropout)
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(
            JOINT_COUNT * COORDINATE_DIM + JOINT_COUNT,
            hidden_dim,
        )
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=heads,
            dim_feedforward=ffn_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal = nn.TransformerEncoder(
            layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        skeletons: Tensor,
        actor_mask: Tensor,
        frame_mask: Tensor,
        track_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        features = _skeleton_features(skeletons, track_mask)
        batch_size, actors, time_steps, _ = features.shape
        time_mask = (
            actor_mask[:, :, None]
            & frame_mask[:, None, :]
            & track_mask.any(dim=-1)
        )
        actor_embeddings = torch.zeros(
            (batch_size, actors, self.hidden_dim),
            dtype=features.dtype,
            device=features.device,
        )
        actor_sequences = torch.zeros(
            (batch_size, actors, time_steps, self.hidden_dim),
            dtype=features.dtype,
            device=features.device,
        )
        # Each actor is evaluated as a fixed-shape singleton.  This makes the
        # frozen runtime independent of K_pad and of unrelated batch members.
        for group in range(batch_size):
            valid_length = int(frame_mask[group].sum().item())
            for actor in range(actors):
                if not bool(actor_mask[group, actor].item()):
                    continue
                selected_features = features[
                    group : group + 1,
                    actor,
                    :valid_length,
                ]
                selected_mask = time_mask[
                    group : group + 1,
                    actor,
                    :valid_length,
                ]
                hidden = self.input_projection(selected_features)
                hidden = hidden + _sinusoidal_time_encoding(
                    valid_length,
                    self.hidden_dim,
                    device=hidden.device,
                )[None, :, :]
                hidden = self.temporal(hidden, src_key_padding_mask=~selected_mask)
                hidden = self.output_norm(hidden)
                hidden = torch.where(
                    selected_mask[..., None],
                    hidden,
                    torch.zeros_like(hidden),
                )
                denominator = selected_mask.sum(
                    dim=1, dtype=hidden.dtype
                ).clamp_min(1.0)
                actor_embeddings[group, actor] = hidden.sum(dim=1)[0] / denominator[0]
                actor_sequences[group, actor, :valid_length] = hidden[0]
        return actor_embeddings, actor_sequences, time_mask


class PMAPool(nn.Module):
    """One learned seed pools an unordered set with no actor-position feature."""

    def __init__(
        self,
        hidden_dim: int,
        heads: int,
        ffn_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.seed = nn.Parameter(torch.empty((1, 1, hidden_dim), dtype=torch.float32))
        nn.init.normal_(self.seed, mean=0.0, std=hidden_dim**-0.5)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, actors: Tensor, actor_mask: Tensor) -> Tensor:
        outputs: list[Tensor] = []
        for group in range(actors.shape[0]):
            actor_count = int(actor_mask[group].sum().item())
            values = actors[group : group + 1, :actor_count]
            attended, _ = self.attention(
                self.seed,
                values,
                values,
                need_weights=False,
            )
            hidden = self.norm1(self.seed + attended)
            outputs.append(self.norm2(hidden + self.ffn(hidden))[0, 0])
        return torch.stack(outputs)


class ActorMeanBase(nn.Module):
    """B0: shared temporal actor encoder followed by a masked actor mean."""

    system_id: Final = "B0"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor_encoder = SharedSkeletonTemporalEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )

    def forward(self, batch: PreparedGroupBatch) -> GroupBaseOutput:
        device = next(self.parameters()).device
        _, skeletons, actor_mask, frame_mask, track_mask = _public_skeleton_tensors(
            batch, device
        )
        actor_embeddings, _, _ = self.actor_encoder(
            skeletons,
            actor_mask,
            frame_mask,
            track_mask,
        )
        group = torch.stack(
            [
                actor_embeddings[row, : int(actor_mask[row].sum().item())].mean(dim=0)
                for row in range(actor_embeddings.shape[0])
            ]
        )
        return GroupBaseOutput(
            group.contiguous(),
            actor_embeddings.contiguous(),
            actor_mask,
        )


class SetPMABase(nn.Module):
    """B1: shared temporal actor encoder followed by set PMA pooling."""

    system_id: Final = "B1"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.actor_encoder = SharedSkeletonTemporalEncoder(
            hidden_dim=hidden_dim,
            heads=heads,
            ffn_dim=ffn_dim,
            dropout=dropout,
        )
        self.pool = PMAPool(hidden_dim, heads, ffn_dim, dropout)

    def forward(self, batch: PreparedGroupBatch) -> GroupBaseOutput:
        device = next(self.parameters()).device
        _, skeletons, actor_mask, frame_mask, track_mask = _public_skeleton_tensors(
            batch, device
        )
        actor_embeddings, _, _ = self.actor_encoder(
            skeletons,
            actor_mask,
            frame_mask,
            track_mask,
        )
        group = self.pool(actor_embeddings, actor_mask)
        return GroupBaseOutput(
            group.contiguous(),
            actor_embeddings.contiguous(),
            actor_mask,
        )


class SocialTemporalLayer(nn.Module):
    """One temporal-attention then per-frame actor-set-attention layer."""

    def __init__(self, hidden_dim: int, heads: int, ffn_dim: int, dropout: float) -> None:
        super().__init__()
        self.temporal_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.social_attention = nn.MultiheadAttention(
            hidden_dim,
            heads,
            dropout=float(dropout),
            batch_first=True,
        )
        self.temporal_norm = nn.LayerNorm(hidden_dim)
        self.social_norm = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(ffn_dim, hidden_dim),
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, value: Tensor, time_mask: Tensor, actor_mask: Tensor) -> Tensor:
        batch_size, actors, time_steps, width = value.shape
        flat = value.reshape(batch_size * actors, time_steps, width)
        flat_mask = time_mask.reshape(batch_size * actors, time_steps)
        valid_actor_rows = actor_mask.reshape(batch_size * actors)
        selected = flat[valid_actor_rows]
        selected_mask = flat_mask[valid_actor_rows]
        attended, _ = self.temporal_attention(
            selected,
            selected,
            selected,
            key_padding_mask=~selected_mask,
            need_weights=False,
        )
        selected = self.temporal_norm(selected + attended)
        selected = torch.where(
            selected_mask[..., None],
            selected,
            torch.zeros_like(selected),
        )
        temporal = torch.zeros_like(flat)
        temporal[valid_actor_rows] = selected
        temporal = temporal.reshape(batch_size, actors, time_steps, width)

        by_frame = temporal.permute(0, 2, 1, 3).reshape(
            batch_size * time_steps,
            actors,
            width,
        )
        frame_actor_mask = time_mask.permute(0, 2, 1).reshape(
            batch_size * time_steps,
            actors,
        )
        valid_frame_rows = frame_actor_mask.any(dim=1)
        selected_frames = by_frame[valid_frame_rows]
        selected_frame_mask = frame_actor_mask[valid_frame_rows]
        social, _ = self.social_attention(
            selected_frames,
            selected_frames,
            selected_frames,
            key_padding_mask=~selected_frame_mask,
            need_weights=False,
        )
        selected_frames = self.social_norm(selected_frames + social)
        selected_frames = torch.where(
            selected_frame_mask[..., None],
            selected_frames,
            torch.zeros_like(selected_frames),
        )
        social_rows = torch.zeros_like(by_frame)
        social_rows[valid_frame_rows] = selected_frames
        output = social_rows.reshape(batch_size, time_steps, actors, width).permute(
            0, 2, 1, 3
        )
        output = self.output_norm(output + self.ffn(output))
        return torch.where(time_mask[..., None], output, torch.zeros_like(output))


class SocialTemporalBase(nn.Module):
    """B2: four alternating temporal/social layers and final PMA pooling."""

    system_id: Final = "B2"

    def __init__(
        self,
        *,
        hidden_dim: int = 512,
        heads: int = 8,
        ffn_dim: int = 2048,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        _validate_base_widths(hidden_dim, heads, ffn_dim, dropout)
        self.hidden_dim = hidden_dim
        self.input_projection = nn.Linear(
            JOINT_COUNT * COORDINATE_DIM + JOINT_COUNT,
            hidden_dim,
        )
        self.layers = nn.ModuleList(
            SocialTemporalLayer(hidden_dim, heads, ffn_dim, dropout)
            for _ in range(4)
        )
        self.pool = PMAPool(hidden_dim, heads, ffn_dim, dropout)

    def forward(self, batch: PreparedGroupBatch) -> GroupBaseOutput:
        device = next(self.parameters()).device
        _, skeletons, actor_mask, frame_mask, track_mask = _public_skeleton_tensors(
            batch, device
        )
        all_features = _skeleton_features(skeletons, track_mask)
        actor_embeddings = torch.zeros(
            (skeletons.shape[0], skeletons.shape[1], self.hidden_dim),
            dtype=torch.float32,
            device=device,
        )
        group_rows: list[Tensor] = []
        for group_index in range(skeletons.shape[0]):
            actor_count = int(actor_mask[group_index].sum().item())
            valid_length = int(frame_mask[group_index].sum().item())
            local_actor_mask = actor_mask[
                group_index : group_index + 1,
                :actor_count,
            ]
            local_track = track_mask[
                group_index : group_index + 1,
                :actor_count,
                :valid_length,
            ]
            local_time_mask = (
                local_actor_mask[:, :, None]
                & frame_mask[
                    group_index : group_index + 1,
                    None,
                    :valid_length,
                ]
                & local_track.any(dim=-1)
            )
            local_features = all_features[
                group_index : group_index + 1,
                :actor_count,
                :valid_length,
            ]
            hidden = self.input_projection(local_features)
            hidden = hidden + _sinusoidal_time_encoding(
                valid_length,
                self.hidden_dim,
                device=device,
            )[None, None, :, :]
            hidden = torch.where(
                local_time_mask[..., None], hidden, torch.zeros_like(hidden)
            )
            for layer in self.layers:
                hidden = layer(hidden, local_time_mask, local_actor_mask)
            denominator = local_time_mask.sum(
                dim=2, dtype=hidden.dtype
            ).clamp_min(1.0)
            local_actors = hidden.sum(dim=2) / denominator[..., None]
            actor_embeddings[group_index, :actor_count] = local_actors[0]
            group_rows.append(self.pool(local_actors, local_actor_mask)[0])
        group = torch.stack(group_rows)
        return GroupBaseOutput(
            group.contiguous(),
            actor_embeddings.contiguous(),
            actor_mask,
        )


def build_group_base(
    system_id: str,
    *,
    hidden_dim: int = 512,
    heads: int = 8,
    ffn_dim: int = 2048,
    dropout: float = 0.1,
) -> nn.Module:
    """Construct exactly one registered base-qualification candidate."""

    candidates: dict[str, type[nn.Module]] = {
        "B0": ActorMeanBase,
        "B1": SetPMABase,
        "B2": SocialTemporalBase,
    }
    if system_id not in candidates:
        raise PhaseSetModelError("base system_id must be one of B0, B1, B2")
    return candidates[system_id](
        hidden_dim=hidden_dim,
        heads=heads,
        ffn_dim=ffn_dim,
        dropout=dropout,
    )


class PhaseSetEncoder(nn.Module):
    """Six-band group encoder with O(K^2) time and O(K) resident state.

    The fixed PhasePair descriptor is streamed edge-by-edge.  No dense KxK
    tensor is constructed.  Actor commitments canonicalize the stream before
    this module sees it, while commitments themselves are never model features.
    """

    def __init__(
        self,
        *,
        embedding_dim: int = 512,
        hidden_dim: int = 256,
        energy_floors: np.ndarray | None = None,
        topology_scale: float = 1.0,
        edge_budget: int = DEFAULT_EDGE_BUDGET,
    ) -> None:
        super().__init__()
        if type(embedding_dim) is not int or embedding_dim < 2:
            raise TypeError("embedding_dim must be an exact int >=2")
        if type(hidden_dim) is not int or hidden_dim < 2:
            raise TypeError("hidden_dim must be an exact int >=2")
        if type(topology_scale) not in (float, int) or not np.isfinite(topology_scale):
            raise TypeError("topology_scale must be a finite built-in number")
        if type(edge_budget) is not int or edge_budget < 1:
            raise TypeError("edge_budget must be an exact positive int")
        floors = (
            np.zeros((BAND_COUNT,), dtype=np.float64)
            if energy_floors is None
            else energy_floors
        )
        self._energy_floors = validate_energy_floors(floors)
        self.embedding_dim = embedding_dim
        self.edge_budget = edge_budget
        # ``topology_scale`` is the initialization of the registered per-band
        # lambda.  The forward path always applies tanh, so the residual scale
        # remains learned and bounded.
        self.topology_lambda = nn.Parameter(
            torch.full((BAND_COUNT,), float(topology_scale), dtype=torch.float32)
        )
        self.half_edge_encoder = nn.Sequential(
            nn.Linear(TOKEN_WIDTH, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.pair_encoder = nn.Sequential(
            nn.Linear(3 * embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        # Raw degree is deliberately absent.  The last two scalars are support
        # coverage d/(K-1) and its complementary missingness.
        self.topology_encoder = nn.Sequential(
            nn.Linear(2 * embedding_dim + 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.common_postprocess = nn.LayerNorm(embedding_dim)

    def _device(self) -> torch.device:
        return next(self.parameters()).device

    def _edge_forward(
        self,
        relation_ij: Tensor,
        relation_ji: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Apply the registered directed-half-edge and symmetric-pair MLPs."""

        half_ij = self.half_edge_encoder(relation_ij)
        half_ji = self.half_edge_encoder(relation_ji)
        pair_tokens = self.pair_encoder(
            torch.cat(
                (
                    half_ij + half_ji,
                    torch.abs(half_ij - half_ji),
                    half_ij * half_ji,
                ),
                dim=-1,
            )
        )
        return half_ij, half_ji, pair_tokens

    def forward(
        self,
        batch: PreparedGroupBatch,
        *,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
        include_topology: bool = True,
    ) -> GroupTokenOutput:
        """Encode the public 22-joint skeleton contract."""

        checked = validate_prepared_group_batch(batch)
        return self.forward_activity(
            skeleton_to_activity(checked),
            edge_chunk_size=edge_chunk_size,
            include_topology=include_topology,
        )

    def forward_activity(
        self,
        batch: PreparedActivityBatch,
        *,
        edge_chunk_size: int = DEFAULT_EDGE_CHUNK_SIZE,
        include_topology: bool = True,
    ) -> GroupTokenOutput:
        """Internal/legacy test seam for already derived five-channel activity."""

        if type(include_topology) is not bool:
            raise TypeError("include_topology must be an exact built-in bool")
        checked = validate_prepared_activity_batch(batch)
        device = self._device()
        batch_size, padded_actors = checked.actor_mask.shape
        width = self.embedding_dim

        # Python-ordered accumulators preserve the canonical edge order across
        # runtime chunk sizes and avoid GPU atomic scatter reductions.
        pair_sums = [
            [torch.zeros((width,), dtype=torch.float64, device=device) for _ in range(BAND_COUNT)]
            for _ in range(batch_size)
        ]
        pair_corrections = [
            [torch.zeros((width,), dtype=torch.float64, device=device) for _ in range(BAND_COUNT)]
            for _ in range(batch_size)
        ]
        pair_counts = [[0 for _ in range(BAND_COUNT)] for _ in range(batch_size)]
        node_means = [
            [
                [
                    torch.zeros((width,), dtype=torch.float64, device=device)
                    for _ in range(BAND_COUNT)
                ]
                for _ in range(padded_actors)
            ]
            for _ in range(batch_size)
        ]
        node_m2 = [
            [
                [
                    torch.zeros((width,), dtype=torch.float64, device=device)
                    for _ in range(BAND_COUNT)
                ]
                for _ in range(padded_actors)
            ]
            for _ in range(batch_size)
        ]
        node_degrees = [
            [[0 for _ in range(BAND_COUNT)] for _ in range(padded_actors)]
            for _ in range(batch_size)
        ]

        for chunk in iter_unordered_pair_chunks(
            checked,
            energy_floors=self._energy_floors,
            edge_chunk_size=edge_chunk_size,
            edge_budget=self.edge_budget,
        ):
            edge_count = int(chunk.batch_indices.shape[0])
            for micro_start in range(0, edge_count, CANONICAL_MICROBLOCK_SIZE):
                micro_stop = min(
                    micro_start + CANONICAL_MICROBLOCK_SIZE,
                    edge_count,
                )
                relation_ij = torch.tensor(
                    chunk.tokens_ij[micro_start:micro_stop],
                    dtype=torch.float32,
                    device=device,
                )
                relation_ji = torch.tensor(
                    chunk.tokens_ji[micro_start:micro_stop],
                    dtype=torch.float32,
                    device=device,
                )
                if self.training:
                    half_ij, half_ji, pair_tokens = checkpoint(
                        self._edge_forward,
                        relation_ij,
                        relation_ji,
                        use_reentrant=False,
                    )
                else:
                    half_ij, half_ji, pair_tokens = self._edge_forward(
                        relation_ij,
                        relation_ji,
                    )

                pair_block: dict[tuple[int, int], list[Tensor]] = {}
                node_block: dict[tuple[int, int, int], list[Tensor]] = {}
                for local_edge, edge in enumerate(range(micro_start, micro_stop)):
                    group = int(chunk.batch_indices[edge])
                    left = int(chunk.actor_i[edge])
                    right = int(chunk.actor_j[edge])
                    for band in range(BAND_COUNT):
                        if not bool(chunk.support_mask[edge, band]):
                            continue
                        pair_block.setdefault((group, band), []).append(
                            pair_tokens[local_edge, band].to(dtype=torch.float64)
                        )
                        node_block.setdefault((group, left, band), []).append(
                            half_ij[local_edge, band].to(dtype=torch.float64)
                        )
                        node_block.setdefault((group, right, band), []).append(
                            half_ji[local_edge, band].to(dtype=torch.float64)
                        )

                for group, band in sorted(pair_block):
                    values = pair_block[(group, band)]
                    block_sum = _fixed_binary_tree_sum(values)
                    pair_sums[group][band], pair_corrections[group][band] = (
                        _compensated_add(
                            pair_sums[group][band],
                            pair_corrections[group][band],
                            block_sum,
                        )
                    )
                    pair_counts[group][band] += len(values)

                for group, actor, band in sorted(node_block):
                    block_state = _fixed_tree_chan(node_block[(group, actor, band)])
                    merged = _chan_merge(
                        node_degrees[group][actor][band],
                        node_means[group][actor][band],
                        node_m2[group][actor][band],
                        *block_state,
                    )
                    node_degrees[group][actor][band] = merged[0]
                    node_means[group][actor][band] = merged[1]
                    node_m2[group][actor][band] = merged[2]

        pair_rows: list[Tensor] = []
        pair_count_rows: list[list[int]] = []
        for group in range(batch_size):
            bands: list[Tensor] = []
            for band in range(BAND_COUNT):
                count = pair_counts[group][band]
                value = (
                    (pair_sums[group][band] + pair_corrections[group][band]) / count
                    if count
                    else torch.zeros((width,), dtype=torch.float64, device=device)
                )
                bands.append(value.to(dtype=torch.float32))
            pair_rows.append(torch.stack(bands))
            pair_count_rows.append(pair_counts[group])
        pair_component = torch.stack(pair_rows)
        pair_count_tensor = torch.tensor(pair_count_rows, dtype=torch.int64, device=device)
        band_mask = pair_count_tensor > 0

        global_halfedge_means: list[list[Tensor]] = []
        for group, actor_count in enumerate(checked.actor_counts):
            group_means: list[Tensor] = []
            for band in range(BAND_COUNT):
                count = 0
                mean = torch.zeros((width,), dtype=torch.float64, device=device)
                m2 = torch.zeros((width,), dtype=torch.float64, device=device)
                for actor in range(actor_count):
                    count, mean, m2 = _chan_merge(
                        count,
                        mean,
                        m2,
                        node_degrees[group][actor][band],
                        node_means[group][actor][band],
                        node_m2[group][actor][band],
                    )
                group_means.append(mean)
            global_halfedge_means.append(group_means)

        statistic_groups: list[Tensor] = []
        topology_masks: list[list[list[bool]]] = []
        for group, actor_count in enumerate(checked.actor_counts):
            actor_statistics: list[Tensor] = []
            actor_topology_masks: list[list[bool]] = []
            possible_neighbors = actor_count - 1
            for actor in range(padded_actors):
                band_statistics: list[Tensor] = []
                band_topology_masks: list[bool] = []
                for band in range(BAND_COUNT):
                    degree = node_degrees[group][actor][band] if actor < actor_count else 0
                    if degree:
                        mean = node_means[group][actor][band]
                        moment = torch.clamp(
                            node_m2[group][actor][band] / degree,
                            min=0.0,
                        )
                    else:
                        mean = torch.zeros((width,), dtype=torch.float64, device=device)
                        moment = torch.zeros((width,), dtype=torch.float64, device=device)
                    delta_mean = mean - global_halfedge_means[group][band]
                    coverage = degree / possible_neighbors if actor < actor_count else 0.0
                    scalars = torch.tensor(
                        (coverage, 1.0 - coverage),
                        dtype=torch.float64,
                        device=device,
                    )
                    band_statistics.append(torch.cat((delta_mean, moment, scalars)))
                    band_topology_masks.append(actor < actor_count and degree >= 2)
                actor_statistics.append(torch.stack(band_statistics))
                actor_topology_masks.append(band_topology_masks)
            statistic_groups.append(torch.stack(actor_statistics))
            topology_masks.append(actor_topology_masks)
        statistics = torch.stack(statistic_groups).to(dtype=torch.float32)
        topology_mask = torch.tensor(topology_masks, dtype=torch.bool, device=device)
        raw_node_delta = self.topology_encoder(statistics)
        # The false branch is materialized +0.  The topology encoder remains in
        # the graph, so an all-K=2 batch receives allocated, exactly zero grads.
        node_delta = torch.where(
            topology_mask[..., None],
            raw_node_delta,
            torch.zeros_like(raw_node_delta),
        )

        topology_rows: list[Tensor] = []
        topology_count_rows: list[list[int]] = []
        for group, actor_count in enumerate(checked.actor_counts):
            bands = []
            counts = []
            for band in range(BAND_COUNT):
                values = [
                    node_delta[group, actor, band].to(dtype=torch.float64)
                    for actor in range(actor_count)
                    if topology_masks[group][actor][band]
                ]
                count = len(values)
                if count:
                    value = (_fixed_binary_tree_sum(values) / count).to(
                        dtype=torch.float32
                    )
                else:
                    # Preserve an autograd path through every hard-gated node so
                    # K=2 produces allocated, bitwise-positive zero gradients.
                    linked_zeros = [
                        node_delta[group, actor, band].to(dtype=torch.float64)
                        for actor in range(actor_count)
                    ]
                    value = _fixed_binary_tree_sum(linked_zeros).to(dtype=torch.float32)
                bands.append(value)
                counts.append(count)
            topology_rows.append(torch.stack(bands))
            topology_count_rows.append(counts)
        topology_delta = torch.stack(topology_rows)
        topology_count_tensor = torch.tensor(
            topology_count_rows,
            dtype=torch.int64,
            device=device,
        )

        effective_topology = (
            topology_delta if include_topology else torch.zeros_like(topology_delta)
        )
        bounded_scale = torch.tanh(self.topology_lambda)[None, :, None]
        combined = pair_component + bounded_scale * effective_topology
        processed = self.common_postprocess(combined)
        tokens = torch.where(band_mask[..., None], processed, torch.zeros_like(processed))
        if not bool(torch.isfinite(tokens).all().item()):
            raise PhaseSetModelError("group tokens became nonfinite")
        return GroupTokenOutput(
            tokens=tokens.contiguous(),
            band_mask=band_mask.contiguous(),
            pair_component=pair_component.contiguous(),
            topology_delta=effective_topology.contiguous(),
            valid_pair_count=pair_count_tensor.contiguous(),
            topology_node_count=topology_count_tensor.contiguous(),
        )


__all__ = [
    "ActorMeanBase",
    "DEFAULT_EDGE_BUDGET",
    "GroupBaseOutput",
    "GroupTokenOutput",
    "PMAPool",
    "PhaseSetEncoder",
    "PhaseSetModelError",
    "PhaseSetOutput",
    "ResourceLimitError",
    "SetPMABase",
    "SharedSkeletonTemporalEncoder",
    "SocialTemporalBase",
    "SocialTemporalLayer",
    "STATUS",
    "SharedActorEncoder",
    "build_group_base",
]
