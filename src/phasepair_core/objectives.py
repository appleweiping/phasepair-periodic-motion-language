"""Stable, data-free PhasePair score and multi-positive loss primitives.

The canonical entry point implements the frozen three-caption-per-source batch
contract.  It consumes already normalized motion/text embeddings and never
loads captions, assets, checkpoints, or experiment results.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


STATUS = "DATA_FREE_OBJECTIVE_COMPUTED_NONPRODUCTION_NO_RESULT"
VECTOR_WIDTH = 512
MAX_BATCH = 128
CAPTIONS_PER_SOURCE = 3


class ObjectiveContractError(ValueError):
    """Raised when an objective input violates the closed tensor contract."""


def _make_snapshot_helpers(
    *,
    tensor_type: type[Tensor] = Tensor,
    parameter_type: type[nn.Parameter] = nn.Parameter,
    tensor_detach=Tensor.detach,
    tensor_clone=Tensor.clone,
    tensor_is_contiguous=Tensor.is_contiguous,
    contiguous_format=torch.contiguous_format,
    float32=torch.float32,
    bool_dtype=torch.bool,
    isfinite=torch.isfinite,
):
    """Capture exact Tensor primitives before any caller can shadow methods."""

    def snapshot_float32(
        value: object,
        label: str,
        *,
        rank: int,
        device: torch.device | None = None,
    ) -> Tensor:
        if type(value) is not tensor_type:
            raise TypeError(f"{label} must be an exact torch.Tensor")
        if value.dtype != float32 or not tensor_is_contiguous(value):
            raise ObjectiveContractError(f"{label} must be contiguous float32")
        if value.requires_grad is False and label == "logits":
            # Diagnostic logits are allowed to be detached; this branch documents
            # that no gradient requirement is inferred from the low-level oracle.
            pass
        snapshot = tensor_clone(value, memory_format=contiguous_format)
        if snapshot.ndim != rank:
            raise ObjectiveContractError(f"{label} rank mismatch")
        if device is not None and snapshot.device != device:
            raise ObjectiveContractError(f"{label} device mismatch")
        if not bool(isfinite(snapshot).all().item()):
            raise ObjectiveContractError(f"{label} contains nonfinite values")
        return snapshot

    def snapshot_logit_scale(value: object, device: torch.device) -> Tensor:
        if type(value) not in {tensor_type, parameter_type}:
            raise TypeError("logit_scale must be an exact Tensor or Parameter")
        if (
            value.dtype != float32
            or value.device != device
            or tuple(value.shape) != (1,)
            or not tensor_is_contiguous(value)
        ):
            raise ObjectiveContractError(
                "logit_scale must be contiguous float32[1] on-device"
            )
        snapshot = tensor_clone(value, memory_format=contiguous_format)
        if not bool(isfinite(snapshot).all().item()):
            raise ObjectiveContractError("logit_scale must be finite")
        return snapshot

    def snapshot_positive_mask(
        value: object,
        shape: tuple[int, int],
        device: torch.device,
    ) -> Tensor:
        if type(value) is not tensor_type:
            raise TypeError("positive_mask must be an exact torch.Tensor")
        if (
            value.dtype != bool_dtype
            or value.device != device
            or not tensor_is_contiguous(value)
            or value.requires_grad
        ):
            raise ObjectiveContractError("positive_mask dtype/device/layout mismatch")
        snapshot = tensor_clone(
            tensor_detach(value),
            memory_format=contiguous_format,
        )
        if tuple(snapshot.shape) != shape:
            raise ObjectiveContractError("positive_mask shape mismatch")
        return snapshot

    return snapshot_float32, snapshot_logit_scale, snapshot_positive_mask


(
    _snapshot_float32,
    _snapshot_logit_scale,
    _snapshot_positive_mask,
) = _make_snapshot_helpers()


def _require_l2_rows(value: Tensor, label: str) -> None:
    norms = torch.linalg.vector_norm(value, dim=-1)
    if not bool(torch.isfinite(norms).all().item()):
        raise ObjectiveContractError(f"{label} norms are nonfinite")
    if not bool(torch.all(torch.abs(norms - 1.0) <= 1e-5).item()):
        raise ObjectiveContractError(f"{label} rows must already be L2 normalized")


def ordered_pair_average_scores(
    motion_ordered: Tensor,
    text_embeddings: Tensor,
) -> Tensor:
    """Return ``0.5*(AB@text.T + BA@text.T)`` from normalized inputs."""

    motion = _snapshot_float32(motion_ordered, "motion_ordered", rank=3)
    text = _snapshot_float32(
        text_embeddings,
        "text_embeddings",
        rank=2,
        device=motion.device,
    )
    if motion.shape[0] != 2 or motion.shape[2] != 512:
        raise ObjectiveContractError("motion_ordered must have shape [2,B,512]")
    batch_size = int(motion.shape[1])
    if batch_size < 1 or batch_size > 128:
        raise ObjectiveContractError("motion batch dimension is outside [1,128]")
    if text.shape[0] < 1 or text.shape[1] != 512:
        raise ObjectiveContractError("text_embeddings must have shape [Q,512]")
    _require_l2_rows(motion, "motion_ordered")
    _require_l2_rows(text, "text_embeddings")
    return (0.5 * (motion[0] @ text.transpose(0, 1) + motion[1] @ text.transpose(0, 1))).contiguous()


def symmetric_multi_positive_infonce(
    logits: Tensor,
    positive_mask: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Stable low-level loss oracle; every row and column needs a positive."""

    checked_logits = _snapshot_float32(logits, "logits", rank=2)
    if checked_logits.shape[0] < 1 or checked_logits.shape[1] < 1:
        raise ObjectiveContractError("logits dimensions must be nonempty")
    checked_mask = _snapshot_positive_mask(
        positive_mask,
        (int(checked_logits.shape[0]), int(checked_logits.shape[1])),
        checked_logits.device,
    )
    if not bool(checked_mask.any(dim=1).all().item()):
        raise ObjectiveContractError("every motion row must have a positive caption")
    if not bool(checked_mask.any(dim=0).all().item()):
        raise ObjectiveContractError("every caption column must have a positive motion")
    negative_infinity = torch.tensor(
        float("-inf"),
        dtype=checked_logits.dtype,
        device=checked_logits.device,
    )
    positive_logits = torch.where(checked_mask, checked_logits, negative_infinity)
    motion_to_text = -torch.mean(
        torch.logsumexp(positive_logits, dim=1)
        - torch.logsumexp(checked_logits, dim=1)
    )
    text_to_motion = -torch.mean(
        torch.logsumexp(positive_logits, dim=0)
        - torch.logsumexp(checked_logits, dim=0)
    )
    total = 0.5 * (motion_to_text + text_to_motion)
    if not bool(
        torch.isfinite(torch.stack((motion_to_text, text_to_motion, total))).all().item()
    ):
        raise ObjectiveContractError("multi-positive loss is nonfinite")
    return motion_to_text, text_to_motion, total


class PhasePairObjectiveOutput:
    """Read-only canonical objective view; direct public construction is forbidden."""

    __slots__ = (
        "_logits",
        "_motion_to_text_loss",
        "_text_to_motion_loss",
        "_loss",
        "_positive_mask",
    )

    def __init__(self, *_: object, **__: object) -> None:
        raise ObjectiveContractError(
            "PhasePairObjectiveOutput is minted only by the canonical objective entry point"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable: {name}")

    @property
    def logits(self) -> Tensor:
        return self._logits.clone(memory_format=torch.contiguous_format)

    @property
    def motion_to_text_loss(self) -> Tensor:
        return self._motion_to_text_loss.clone()

    @property
    def text_to_motion_loss(self) -> Tensor:
        return self._text_to_motion_loss.clone()

    @property
    def loss(self) -> Tensor:
        return self._loss.clone()

    @property
    def positive_mask(self) -> Tensor:
        return self._positive_mask.clone(memory_format=torch.contiguous_format)

    @property
    def status(self) -> str:
        return "DATA_FREE_OBJECTIVE_COMPUTED_NONPRODUCTION_NO_RESULT"


def _mint_objective_output(logits: Tensor, positive_mask: Tensor) -> PhasePairObjectiveOutput:
    checked_logits = _snapshot_float32(logits, "logits", rank=2)
    if (
        checked_logits.shape[0] < 1
        or checked_logits.shape[0] > 128
        or checked_logits.shape[1] != 3 * checked_logits.shape[0]
    ):
        raise ObjectiveContractError("logits output contract mismatch")
    checked_mask = _snapshot_positive_mask(
        positive_mask,
        (int(checked_logits.shape[0]), int(checked_logits.shape[1])),
        checked_logits.device,
    )
    if not bool((checked_mask.sum(dim=1) == 3).all().item()):
        raise ObjectiveContractError("each output row must bind exactly three positives")
    if not bool((checked_mask.sum(dim=0) == 1).all().item()):
        raise ObjectiveContractError("each output column must bind exactly one source")
    motion_to_text, text_to_motion, total = symmetric_multi_positive_infonce(
        checked_logits,
        checked_mask,
    )
    output = object.__new__(PhasePairObjectiveOutput)
    object.__setattr__(output, "_logits", checked_logits)
    object.__setattr__(output, "_motion_to_text_loss", motion_to_text)
    object.__setattr__(output, "_text_to_motion_loss", text_to_motion)
    object.__setattr__(output, "_loss", total)
    object.__setattr__(output, "_positive_mask", checked_mask)
    return output


def phasepair_multi_positive_objective(
    motion_ordered: Tensor,
    text_embeddings: Tensor,
    logit_scale: Tensor | nn.Parameter,
    positive_mask: Tensor,
    *,
    ordered_residual_scores: Tensor | None = None,
) -> PhasePairObjectiveOutput:
    """Compute the canonical three-caption/source PhasePair training loss."""

    base_scores = ordered_pair_average_scores(motion_ordered, text_embeddings)
    batch_size, caption_count = (int(item) for item in base_scores.shape)
    if caption_count != 3 * batch_size:
        raise ObjectiveContractError("canonical batches require exactly three captions per source")
    checked_mask = _snapshot_positive_mask(
        positive_mask,
        (batch_size, caption_count),
        base_scores.device,
    )
    row_counts = checked_mask.sum(dim=1)
    column_counts = checked_mask.sum(dim=0)
    if not bool((row_counts == 3).all().item()):
        raise ObjectiveContractError("each motion row must have exactly three positives")
    if not bool((column_counts == 1).all().item()):
        raise ObjectiveContractError("each caption must have exactly one source motion")

    combined = base_scores
    if ordered_residual_scores is not None:
        residual = _snapshot_float32(
            ordered_residual_scores,
            "ordered_residual_scores",
            rank=3,
            device=base_scores.device,
        )
        if tuple(residual.shape) != (2, batch_size, caption_count):
            raise ObjectiveContractError(
                "ordered_residual_scores must have shape [2,B,3B]"
            )
        if not bool(torch.logical_and(residual >= -1.0, residual <= 1.0).all().item()):
            raise ObjectiveContractError("ordered residual scores must remain in [-1,1]")
        combined = combined + 0.2 * (0.5 * (residual[0] + residual[1]))

    checked_scale = _snapshot_logit_scale(logit_scale, base_scores.device)
    scale = torch.clamp(
        torch.exp(torch.clamp(checked_scale[0], min=0.0, max=math.log(100.0))),
        min=1.0,
        max=100.0,
    )
    logits = (scale * combined).contiguous()
    return _mint_objective_output(logits, checked_mask)


__all__ = [
    "CAPTIONS_PER_SOURCE",
    "ObjectiveContractError",
    "PhasePairObjectiveOutput",
    "STATUS",
    "ordered_pair_average_scores",
    "phasepair_multi_positive_objective",
    "symmetric_multi_positive_infonce",
]
