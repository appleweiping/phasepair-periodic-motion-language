"""Data-free PyTorch motion topologies for the PhasePair contract.

This module contains no dataset reader, CLIP resolver, optimizer, training
loop, checkpoint writer, or result path.  Parameters are allocated without
calling framework ``reset_parameters`` methods and intentionally remain
uninitialized.  A caller must provide a separately reviewed initializer before
using a model for anything beyond the synthetic topology checks in this
repository.

Training-mode motion dropout is deliberately external.  The model consumes the
49/16/32 canonical semantic sites in architecture order, but it cannot create
or bless a production schedule.  Missing external evidence therefore HOLDs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn
from torch.nn import functional as torch_functional

from . import batching, contracts, dropout
from .runtime_dropout import OneForwardCanonicalDropoutBundle


STATUS = "DATA_FREE_NONPRODUCTION_AUTHORITY0"
DROPOUT_HOLD_CODE = "HOLD_EXTERNAL_CANONICAL_DROPOUT_BUNDLE"
VECTOR_WIDTH = 512
HEAD_COUNT = 4
HEAD_WIDTH = 128
FFN_WIDTH = 2048
LAYER_NORM_EPS = 1e-5


class TorchModelContractError(ValueError):
    """A local motion-topology invariant was violated."""


class TorchModelContractHold(TorchModelContractError):
    """A required external artifact is absent, so execution must HOLD."""

    code = DROPOUT_HOLD_CODE

    def __init__(self, message: str = DROPOUT_HOLD_CODE) -> None:
        super().__init__(message)


def _make_motion_embedding_validator(
    *,
    tensor_type: type[Tensor] = Tensor,
    tensor_detach=Tensor.detach,
    tensor_is_contiguous=Tensor.is_contiguous,
    float32=torch.float32,
    isfinite=torch.isfinite,
):
    """Capture the exact Tensor boundary before instance methods can be shadowed."""

    def validate(ordered: object) -> None:
        if type(ordered) is not tensor_type:
            raise TypeError("ordered must be an exact torch.Tensor")
        if ordered.ndim != 3 or ordered.shape[0] != 2:
            raise TorchModelContractError("ordered must have shape [2,B,512]")
        if ordered.shape[1] < 1 or ordered.shape[1] > 128:
            raise TorchModelContractError(
                "ordered batch dimension is outside [1,128]"
            )
        if ordered.shape[2] != 512:
            raise TorchModelContractError("ordered feature dimension must be 512")
        if ordered.dtype != float32 or not tensor_is_contiguous(ordered):
            raise TorchModelContractError("ordered must be contiguous float32")
        if not bool(isfinite(tensor_detach(ordered)).all().item()):
            raise TorchModelContractError("ordered contains a nonfinite value")

    return validate


_validate_motion_embedding = _make_motion_embedding_validator()


@dataclass(frozen=True, slots=True)
class MotionEmbeddings:
    """The two ordered, L2-normalized motion embeddings ``[AB, BA]``."""

    ordered: Tensor

    def __post_init__(self) -> None:
        _validate_motion_embedding(self.ordered)

    @property
    def ab(self) -> Tensor:
        return self.ordered[0]

    @property
    def ba(self) -> Tensor:
        return self.ordered[1]


@dataclass(frozen=True, slots=True)
class MotionInventoryAudit:
    architecture: str
    tensor_count: int
    numel: int
    canonical_names: tuple[str, ...]
    meta_device: bool
    status: str = "META_PARAMETER_INVENTORY_VALIDATED_NONPRODUCTION"


class _ContractLinear(nn.Module):
    """A Linear topology with no framework initializer invocation."""

    def __init__(self, in_features: int, out_features: int, *, device: object) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), dtype=torch.float32, device=device)
        )
        self.bias = nn.Parameter(
            torch.empty((out_features,), dtype=torch.float32, device=device)
        )

    def forward(self, value: Tensor) -> Tensor:
        return torch_functional.linear(value, self.weight, self.bias)


class _ContractLayerNorm(nn.Module):
    """An affine LayerNorm topology with no framework initializer invocation."""

    def __init__(self, width: int = 512, *, device: object) -> None:
        super().__init__()
        self.width = width
        self.weight = nn.Parameter(
            torch.empty((width,), dtype=torch.float32, device=device)
        )
        self.bias = nn.Parameter(
            torch.empty((width,), dtype=torch.float32, device=device)
        )

    def forward(self, value: Tensor) -> Tensor:
        return torch_functional.layer_norm(
            value,
            (self.width,),
            self.weight,
            self.bias,
            1e-5,
        )


def _position_encoding(*, device: object) -> Tensor:
    target = torch.device("cpu" if device is None else device)
    if target.type == "meta":
        return torch.empty(
            (300, 512),
            dtype=torch.float32,
            device=target,
        )
    positions = torch.arange(
        300, dtype=torch.float64, device="cpu"
    ).unsqueeze(1)
    dimensions = torch.arange(256, dtype=torch.float64, device="cpu")
    denominator = torch.pow(
        torch.tensor(10000.0, dtype=torch.float64),
        (2.0 * dimensions) / 512.0,
    )
    angles = positions / denominator.unsqueeze(0)
    table = torch.empty(
        (300, 512), dtype=torch.float64, device="cpu"
    )
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table.to(dtype=torch.float32, device=target).contiguous()


def _positive_zero_invalid(value: Tensor, valid_mask: Tensor) -> Tensor:
    return value.masked_fill(~valid_mask.unsqueeze(0).unsqueeze(-1), 0.0)


def _l2(value: Tensor) -> Tensor:
    denominator = torch.clamp_min(
        torch.linalg.vector_norm(value, dim=-1, keepdim=True), 1e-12
    )
    return value / denominator


def _masked_query_pool(
    value: Tensor,
    query: Tensor,
    valid_mask: Tensor,
    *,
    scale_by_width: bool,
) -> Tensor:
    scores = torch.sum(value * query, dim=-1)
    if scale_by_width:
        scores = scores / math.sqrt(512.0)
    scores = scores.masked_fill(~valid_mask.unsqueeze(0), float("-inf"))
    weights = torch.softmax(scores, dim=-1)
    pooled = torch.sum(weights.unsqueeze(-1) * value, dim=-2)
    return _l2(pooled)


def _make_dropout_runtime_class(
    *,
    reviewed_bundle_type=OneForwardCanonicalDropoutBundle,
    reviewed_take=OneForwardCanonicalDropoutBundle.take,
    reviewed_assert_complete=OneForwardCanonicalDropoutBundle.assert_complete,
    reviewed_architecture_getter=OneForwardCanonicalDropoutBundle.architecture.fget,
    reviewed_batch_getter=OneForwardCanonicalDropoutBundle.batch_size.fget,
    reviewed_time_getter=OneForwardCanonicalDropoutBundle.time_steps.fget,
    tensor_type: type[Tensor] = Tensor,
    tensor_detach=Tensor.detach,
    tensor_clone=Tensor.clone,
    tensor_is_contiguous=Tensor.is_contiguous,
    contiguous_format=torch.contiguous_format,
):
    """Capture the sole reviewed bundle type, then discard this factory."""

    class _DropoutRuntime:
        """Validate the consumer-visible portion of the reviewed mask bundle."""

        def __init__(
            self,
            architecture: str,
            batch_size: int,
            time_steps: int,
            bundle: OneForwardCanonicalDropoutBundle | None,
        ) -> None:
            if bundle is None:
                raise TorchModelContractHold()
            if type(bundle) is not reviewed_bundle_type:
                raise TorchModelContractError(
                    "training dropout requires the exact reviewed one-forward bundle"
                )
            supplied_architecture = reviewed_architecture_getter(bundle)
            supplied_batch = reviewed_batch_getter(bundle)
            supplied_time = reviewed_time_getter(bundle)
            if (
                type(supplied_architecture) is not str
                or supplied_architecture != architecture
            ):
                raise TorchModelContractError("dropout bundle architecture mismatch")
            if type(supplied_batch) is not int or supplied_batch != batch_size:
                raise TorchModelContractError("dropout bundle batch-size mismatch")
            if type(supplied_time) is not int or supplied_time != time_steps:
                raise TorchModelContractError("dropout bundle time-step mismatch")
            self.architecture = architecture
            self.batch_size = batch_size
            self.time_steps = time_steps
            self.bundle = bundle
            self.sites = dropout.site_inventory(architecture)
            self.next_ordinal = 0

        def apply(self, ordinal: int, value: Tensor) -> Tensor:
            if type(ordinal) is not int or ordinal != self.next_ordinal:
                raise TorchModelContractError(
                    "dropout sites were not consumed in traversal order"
                )
            if ordinal >= len(self.sites) or self.sites[ordinal].ordinal != ordinal:
                raise TorchModelContractError(
                    "dropout site ordinal is outside the inventory"
                )
            expected_shape = dropout.site_shape(
                self.sites[ordinal],
                batch_size=self.batch_size,
                time_steps=self.time_steps,
            )
            if tuple(value.shape) != expected_shape:
                raise TorchModelContractError(
                    "model tensor does not match dropout site shape"
                )
            mask = reviewed_take(self.bundle, ordinal, like=value)
            if type(mask) is not tensor_type:
                raise TypeError("dropout bundle must return an exact torch.Tensor")
            if (
                mask.dtype != torch.float32
                or mask.device != value.device
                or not tensor_is_contiguous(mask)
                or mask.requires_grad
            ):
                raise TorchModelContractError(
                    "dropout mask dtype/device/layout mismatch"
                )
            snapshot = tensor_clone(
                tensor_detach(mask),
                memory_format=contiguous_format,
            )
            if tuple(snapshot.shape) != expected_shape:
                raise TorchModelContractError("dropout mask shape mismatch")
            if (
                snapshot.dtype != torch.float32
                or snapshot.device != value.device
                or not tensor_is_contiguous(snapshot)
                or snapshot.requires_grad
            ):
                raise TorchModelContractError("dropout mask snapshot mismatch")
            words = snapshot.view(torch.int32)
            allowed = torch.logical_or(words == 0, words == 0x3F8E38E4)
            if not bool(allowed.all().item()):
                raise TorchModelContractError(
                    "dropout mask contains noncanonical float32 bits"
                )
            self.next_ordinal += 1
            return value * snapshot

        def close(self) -> None:
            if self.next_ordinal != len(self.sites):
                raise TorchModelContractError(
                    "dropout bundle consumption is incomplete"
                )
            outcome = reviewed_assert_complete(self.bundle)
            if outcome is not None:
                raise TorchModelContractError(
                    "dropout assert_complete must return None"
                )

    return _DropoutRuntime


_DropoutRuntime = _make_dropout_runtime_class()
del _make_dropout_runtime_class


def _make_motion_runtime_builder(runtime_type):
    """Bind model construction to the reviewed runtime class by closure."""

    def _build_runtime(
        self,
        *,
        batch_size: int,
        time_steps: int,
        bundle: OneForwardCanonicalDropoutBundle | None,
    ):
        if not self.training:
            return None
        return runtime_type(
            self.architecture,
            batch_size,
            time_steps,
            bundle,
        )

    return _build_runtime


class _ContractAttention(nn.Module):
    def __init__(self, *, device: object) -> None:
        super().__init__()
        self.q = _ContractLinear(512, 512, device=device)
        self.k = _ContractLinear(512, 512, device=device)
        self.v = _ContractLinear(512, 512, device=device)
        self.o = _ContractLinear(512, 512, device=device)

    def forward(
        self,
        query_input: Tensor,
        key_value_input: Tensor,
        valid_mask: Tensor,
        runtime: _DropoutRuntime | None,
        *,
        attention_ordinal: int,
        projection_ordinal: int,
    ) -> Tensor:
        pass_count, batch_size, time_steps, _ = query_input.shape
        query = self.q(query_input).reshape(
            pass_count, batch_size, time_steps, 4, 128
        )
        key = self.k(key_value_input).reshape(
            pass_count, batch_size, time_steps, 4, 128
        )
        value = self.v(key_value_input).reshape(
            pass_count, batch_size, time_steps, 4, 128
        )
        query = query.permute(0, 1, 3, 2, 4)
        key = key.permute(0, 1, 3, 2, 4)
        value = value.permute(0, 1, 3, 2, 4)
        logits = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(
            128.0
        )
        logits = logits.masked_fill(
            ~valid_mask.unsqueeze(0).unsqueeze(2).unsqueeze(2),
            float("-inf"),
        )
        probabilities = torch.softmax(logits, dim=-1)
        if runtime is not None:
            probabilities = runtime.apply(attention_ordinal, probabilities)
        attended = torch.matmul(probabilities, value)
        attended = attended.permute(0, 1, 3, 2, 4).contiguous().reshape(
            pass_count, batch_size, time_steps, 512
        )
        output = self.o(attended)
        if runtime is not None:
            output = runtime.apply(projection_ordinal, output)
        return output


class _ContractFeedForward(nn.Module):
    def __init__(self, *, device: object) -> None:
        super().__init__()
        self.in_projection = _ContractLinear(512, 2048, device=device)
        self.out_projection = _ContractLinear(2048, 512, device=device)

    def forward(
        self,
        value: Tensor,
        runtime: _DropoutRuntime | None,
        *,
        inner_ordinal: int,
        output_ordinal: int,
    ) -> Tensor:
        hidden = torch_functional.gelu(
            self.in_projection(value), approximate="none"
        )
        if runtime is not None:
            hidden = runtime.apply(inner_ordinal, hidden)
        output = self.out_projection(hidden)
        if runtime is not None:
            output = runtime.apply(output_ordinal, output)
        return output


def _linear_pairs(
    prefix: str, layer: _ContractLinear
) -> tuple[tuple[str, nn.Parameter], tuple[str, nn.Parameter]]:
    return (
        (f"{prefix}.weight", layer.weight),
        (f"{prefix}.bias", layer.bias),
    )


def _layer_norm_pairs(
    prefix: str, layer: _ContractLayerNorm
) -> tuple[tuple[str, nn.Parameter], tuple[str, nn.Parameter]]:
    return (
        (f"{prefix}.weight", layer.weight),
        (f"{prefix}.bias", layer.bias),
    )


def _attention_pairs(
    prefix: str,
    attention: _ContractAttention,
    projection_names: tuple[str, str, str, str],
) -> tuple[tuple[str, nn.Parameter], ...]:
    rows: list[tuple[str, nn.Parameter]] = []
    for name, layer in zip(
        projection_names,
        (attention.q, attention.k, attention.v, attention.o),
        strict=True,
    ):
        rows.extend(_linear_pairs(f"{prefix}.{name}", layer))
    return tuple(rows)


def _make_inventory_parameter_checker(
    *,
    parameter_type: type[nn.Parameter] = nn.Parameter,
    tensor_is_contiguous=Tensor.is_contiguous,
    tensor_untyped_storage=Tensor.untyped_storage,
    tensor_storage_offset=Tensor.storage_offset,
    tensor_numel=Tensor.numel,
    tensor_element_size=Tensor.element_size,
    storage_type: type[torch.UntypedStorage] = torch.UntypedStorage,
    storage_data_ptr=torch.UntypedStorage.data_ptr,
    storage_nbytes=torch.UntypedStorage.nbytes,
    float32=torch.float32,
):
    """Return an exact registered-Parameter provenance checker."""

    def check(
        parameter: object,
        *,
        name: str,
        shape: tuple[int, ...],
        require_meta: bool,
        storage_identities: set[tuple[str, int | None, int, int]] | None = None,
        storage_intervals: list[tuple[str, int | None, int, int]] | None = None,
    ) -> int:
        if (storage_identities is None) != (storage_intervals is None):
            raise TypeError("inventory storage accumulators must be supplied together")
        if type(parameter) is not parameter_type:
            raise TorchModelContractError(f"parameter type mismatch: {name}")
        if tuple(parameter.shape) != shape:
            raise TorchModelContractError(f"parameter shape mismatch: {name}")
        if (
            parameter.dtype != float32
            or not parameter.requires_grad
            or not parameter.is_leaf
            or parameter.grad is not None
            or not tensor_is_contiguous(parameter)
        ):
            raise TorchModelContractError(
                f"parameter dtype/gradient/layout mismatch: {name}"
            )
        if require_meta and parameter.device.type != "meta":
            raise TorchModelContractError("meta inventory audit requires meta parameters")
        count = int(tensor_numel(parameter))
        if parameter.device.type != "meta":
            storage = tensor_untyped_storage(parameter)
            if type(storage) is not storage_type:
                raise TorchModelContractError(f"parameter storage type mismatch: {name}")
            element_size = int(tensor_element_size(parameter))
            offset = int(tensor_storage_offset(parameter))
            capacity = int(storage_nbytes(storage))
            pointer = int(storage_data_ptr(storage))
            if (
                offset != 0
                or capacity != count * element_size
                or pointer == 0
            ):
                raise TorchModelContractError(
                    f"parameter storage provenance mismatch: {name}"
                )
            if storage_identities is not None and storage_intervals is not None:
                identity = (
                    parameter.device.type,
                    parameter.device.index,
                    pointer,
                    capacity,
                )
                if identity in storage_identities:
                    raise TorchModelContractError(
                        f"parameter storage identity repeated: {name}"
                    )
                start = pointer + offset * element_size
                stop = start + count * element_size
                if any(
                    device_type == identity[0]
                    and device_index == identity[1]
                    and start < other_stop
                    and other_start < stop
                    for device_type, device_index, other_start, other_stop in storage_intervals
                ):
                    raise TorchModelContractError(
                        f"parameter storage ranges overlap: {name}"
                    )
                storage_identities.add(identity)
                storage_intervals.append(
                    (identity[0], identity[1], start, stop)
                )
        return count

    return check


_check_inventory_parameter = _make_inventory_parameter_checker()


class _MotionEncoderBase(nn.Module):
    architecture: str
    _build_runtime = _make_motion_runtime_builder(_DropoutRuntime)

    def canonical_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        raise NotImplementedError

    def _validate_parameter_inventory(self, *, require_meta: bool) -> MotionInventoryAudit:
        pairs = self.canonical_named_parameters()
        expected = contracts.canonical_motion_rows(self.architecture)
        names = tuple(name for name, _ in pairs)
        if names != tuple(row.name for row in expected):
            raise TorchModelContractError("canonical parameter traversal mismatch")
        if len({id(parameter) for _, parameter in pairs}) != len(pairs):
            raise TorchModelContractError("motion parameters share an object")
        registered = tuple(self.parameters())
        if len(registered) != len(pairs) or {id(item) for item in registered} != {
            id(parameter) for _, parameter in pairs
        }:
            raise TorchModelContractError("registered parameters differ from inventory")
        storage_identities: set[tuple[str, int | None, int, int]] = set()
        storage_intervals: list[tuple[str, int | None, int, int]] = []
        count = sum(
            _check_inventory_parameter(
                parameter,
                name=name,
                shape=row.shape,
                require_meta=require_meta,
                storage_identities=storage_identities,
                storage_intervals=storage_intervals,
            )
            for (name, parameter), row in zip(pairs, expected, strict=True)
        )
        expected_count = contracts.MOTION_PARTITION_CONSTANTS[self.architecture].total
        if len(pairs) != expected_count.tensor_count or count != expected_count.numel:
            raise TorchModelContractError("motion parameter ledger mismatch")
        return MotionInventoryAudit(
            self.architecture,
            len(pairs),
            count,
            names,
            all(parameter.device.type == "meta" for _, parameter in pairs),
        )

    def audit_meta_inventory(self) -> MotionInventoryAudit:
        return self._validate_parameter_inventory(require_meta=True)

    def _device(self) -> torch.device:
        parameters = tuple(self.parameters())
        if not parameters:
            raise TorchModelContractError("motion model has no parameters")
        devices = {parameter.device for parameter in parameters}
        if len(devices) != 1:
            raise TorchModelContractError("motion parameters span multiple devices")
        device = parameters[0].device
        if device.type == "meta":
            raise TorchModelContractError("meta topology cannot execute a forward pass")
        return device

    def _ordered_inputs(
        self,
        batch: batching.PreparedMotionBatch,
        *,
        dropout_bundle: OneForwardCanonicalDropoutBundle | None,
    ) -> tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        _DropoutRuntime | None,
    ]:
        passes = batching.build_ordered_motion_passes(batch)
        _, batch_size, time_steps, _ = passes.actor_role_a.shape
        runtime = self._build_runtime(
            batch_size=batch_size,
            time_steps=time_steps,
            bundle=dropout_bundle,
        )
        device = self._device()
        role_a = torch.tensor(
            passes.actor_role_a, dtype=torch.float32, device=device
        ).contiguous()
        role_b = torch.tensor(
            passes.actor_role_b, dtype=torch.float32, device=device
        ).contiguous()
        relation = torch.tensor(
            passes.relation, dtype=torch.float32, device=device
        ).contiguous()
        early = torch.tensor(
            passes.early_fusion, dtype=torch.float32, device=device
        ).contiguous()
        valid = torch.tensor(
            passes.valid_mask, dtype=torch.bool, device=device
        ).contiguous()
        return role_a, role_b, relation, early, valid, runtime


del _make_motion_runtime_builder


class _MimeBlock(nn.Module):
    def __init__(self, block_index: int, *, device: object) -> None:
        super().__init__()
        self.block_index = block_index
        self.self_a_attention = _ContractAttention(device=device)
        self.self_a_norm = _ContractLayerNorm(device=device)
        self.self_b_attention = _ContractAttention(device=device)
        self.self_b_norm = _ContractLayerNorm(device=device)
        self.cross_a_attention = _ContractAttention(device=device)
        self.cross_a_query_norm = _ContractLayerNorm(device=device)
        self.cross_a_kv_norm = _ContractLayerNorm(device=device)
        self.cross_b_attention = _ContractAttention(device=device)
        self.cross_b_query_norm = _ContractLayerNorm(device=device)
        self.cross_b_kv_norm = _ContractLayerNorm(device=device)
        self.ffn_a = _ContractFeedForward(device=device)
        self.ffn_a_norm = _ContractLayerNorm(device=device)
        self.ffn_b = _ContractFeedForward(device=device)
        self.ffn_b_norm = _ContractLayerNorm(device=device)

    def forward(
        self,
        actor_a: Tensor,
        actor_b: Tensor,
        valid_mask: Tensor,
        runtime: _DropoutRuntime | None,
    ) -> tuple[Tensor, Tensor]:
        base = 12 * self.block_index
        self_a = self.self_a_attention(
            self.self_a_norm(actor_a),
            self.self_a_norm(actor_a),
            valid_mask,
            runtime,
            attention_ordinal=base,
            projection_ordinal=base + 1,
        )
        next_a = _positive_zero_invalid(actor_a + self_a, valid_mask)
        self_b = self.self_b_attention(
            self.self_b_norm(actor_b),
            self.self_b_norm(actor_b),
            valid_mask,
            runtime,
            attention_ordinal=base + 2,
            projection_ordinal=base + 3,
        )
        next_b = _positive_zero_invalid(actor_b + self_b, valid_mask)
        cross_a = self.cross_a_attention(
            self.cross_a_query_norm(next_a),
            self.cross_a_kv_norm(next_b),
            valid_mask,
            runtime,
            attention_ordinal=base + 4,
            projection_ordinal=base + 5,
        )
        cross_b = self.cross_b_attention(
            self.cross_b_query_norm(next_b),
            self.cross_b_kv_norm(next_a),
            valid_mask,
            runtime,
            attention_ordinal=base + 6,
            projection_ordinal=base + 7,
        )
        next_a = _positive_zero_invalid(next_a + cross_a, valid_mask)
        next_b = _positive_zero_invalid(next_b + cross_b, valid_mask)
        ffn_a = self.ffn_a(
            self.ffn_a_norm(next_a),
            runtime,
            inner_ordinal=base + 8,
            output_ordinal=base + 9,
        )
        actor_a = _positive_zero_invalid(next_a + ffn_a, valid_mask)
        ffn_b = self.ffn_b(
            self.ffn_b_norm(next_b),
            runtime,
            inner_ordinal=base + 10,
            output_ordinal=base + 11,
        )
        actor_b = _positive_zero_invalid(next_b + ffn_b, valid_mask)
        return actor_a, actor_b

    def canonical_pairs(self) -> tuple[tuple[str, nn.Parameter], ...]:
        bb = f"{self.block_index:02d}"
        rows: list[tuple[str, nn.Parameter]] = []
        for actor, attention, norm in (
            ("a", self.self_a_attention, self.self_a_norm),
            ("b", self.self_b_attention, self.self_b_norm),
        ):
            prefix = f"mime.block.{bb}.self.{actor}"
            rows.extend(
                _attention_pairs(prefix + ".attn", attention, ("q", "k", "v", "o"))
            )
            rows.extend(_layer_norm_pairs(prefix + ".ln", norm))
        for direction, attention, query_norm, kv_norm in (
            (
                "a_from_b",
                self.cross_a_attention,
                self.cross_a_query_norm,
                self.cross_a_kv_norm,
            ),
            (
                "b_from_a",
                self.cross_b_attention,
                self.cross_b_query_norm,
                self.cross_b_kv_norm,
            ),
        ):
            prefix = f"mime.block.{bb}.cross.{direction}"
            rows.extend(
                _attention_pairs(prefix + ".attn", attention, ("q", "k", "v", "o"))
            )
            rows.extend(_layer_norm_pairs(prefix + ".ln_q", query_norm))
            rows.extend(_layer_norm_pairs(prefix + ".ln_kv", kv_norm))
        for actor, feed_forward, norm in (
            ("a", self.ffn_a, self.ffn_a_norm),
            ("b", self.ffn_b, self.ffn_b_norm),
        ):
            prefix = f"mime.block.{bb}.ffn.{actor}"
            rows.extend(_linear_pairs(prefix + ".in", feed_forward.in_projection))
            rows.extend(_linear_pairs(prefix + ".out", feed_forward.out_projection))
            rows.extend(_layer_norm_pairs(prefix + ".ln", norm))
        return tuple(rows)


class MimeMotionEncoder(_MotionEncoderBase):
    architecture = "mime"

    def __init__(self, *, device: object = None) -> None:
        super().__init__()
        self.input_a_linear = _ContractLinear(262, 512, device=device)
        self.input_a_norm = _ContractLayerNorm(device=device)
        self.actor_embedding_a = nn.Parameter(
            torch.empty((512,), dtype=torch.float32, device=device)
        )
        self.input_b_linear = _ContractLinear(262, 512, device=device)
        self.input_b_norm = _ContractLayerNorm(device=device)
        self.actor_embedding_b = nn.Parameter(
            torch.empty((512,), dtype=torch.float32, device=device)
        )
        self.relation_linear = _ContractLinear(799, 512, device=device)
        self.relation_norm = _ContractLayerNorm(device=device)
        self.relation_type_embedding = nn.Parameter(
            torch.empty((512,), dtype=torch.float32, device=device)
        )
        self.blocks = nn.ModuleList(
            _MimeBlock(block, device=device) for block in range(4)
        )
        self.fusion_in = _ContractLinear(1024, 512, device=device)
        self.fusion_out = _ContractLinear(512, 512, device=device)
        self.pool_query = nn.Parameter(
            torch.empty((512,), dtype=torch.float32, device=device)
        )
        self.register_buffer("position_encoding", _position_encoding(device=device))
        self._validate_parameter_inventory(require_meta=False)

    def canonical_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        rows: list[tuple[str, nn.Parameter]] = []
        rows.extend(_linear_pairs("mime.input.a.linear", self.input_a_linear))
        rows.extend(_layer_norm_pairs("mime.input.a.ln", self.input_a_norm))
        rows.append(("mime.input.actor_embedding.a", self.actor_embedding_a))
        rows.extend(_linear_pairs("mime.input.b.linear", self.input_b_linear))
        rows.extend(_layer_norm_pairs("mime.input.b.ln", self.input_b_norm))
        rows.append(("mime.input.actor_embedding.b", self.actor_embedding_b))
        rows.extend(_linear_pairs("mime.relation.linear", self.relation_linear))
        rows.extend(_layer_norm_pairs("mime.relation.ln", self.relation_norm))
        rows.append(("mime.relation.type_embedding", self.relation_type_embedding))
        for block in self.blocks:
            if type(block) is not _MimeBlock:
                raise TorchModelContractError("MIME block type drift")
            rows.extend(block.canonical_pairs())
        rows.extend(_linear_pairs("mime.fusion.in", self.fusion_in))
        rows.extend(_linear_pairs("mime.fusion.out", self.fusion_out))
        rows.append(("mime.pool.query", self.pool_query))
        return tuple(rows)

    def forward(
        self,
        batch: batching.PreparedMotionBatch,
        *,
        dropout_bundle: OneForwardCanonicalDropoutBundle | None = None,
    ) -> MotionEmbeddings:
        role_a, role_b, relation, _, valid, runtime = self._ordered_inputs(
            batch, dropout_bundle=dropout_bundle
        )
        time_steps = role_a.shape[2]
        position = self.position_encoding[:time_steps].unsqueeze(0).unsqueeze(0)
        actor_a = (
            self.input_a_norm(self.input_a_linear(role_a))
            + self.actor_embedding_a
            + position
        )
        actor_b = (
            self.input_b_norm(self.input_b_linear(role_b))
            + self.actor_embedding_b
            + position
        )
        relation_embedding = (
            self.relation_norm(self.relation_linear(relation))
            + self.relation_type_embedding
        )
        actor_a = _positive_zero_invalid(actor_a + relation_embedding, valid)
        actor_b = _positive_zero_invalid(actor_b + relation_embedding, valid)
        for block in self.blocks:
            actor_a, actor_b = block(actor_a, actor_b, valid, runtime)
        fused = torch_functional.gelu(
            self.fusion_in(torch.cat((actor_a, actor_b), dim=-1)),
            approximate="none",
        )
        if runtime is not None:
            fused = runtime.apply(48, fused)
        fused = _positive_zero_invalid(self.fusion_out(fused), valid)
        ordered = _masked_query_pool(
            fused, self.pool_query, valid, scale_by_width=False
        ).contiguous()
        if runtime is not None:
            runtime.close()
        return MotionEmbeddings(ordered)


class _EncoderBlock(nn.Module):
    def __init__(self, block_index: int, *, device: object) -> None:
        super().__init__()
        self.block_index = block_index
        self.attention_norm = _ContractLayerNorm(device=device)
        self.attention = _ContractAttention(device=device)
        self.ffn_norm = _ContractLayerNorm(device=device)
        self.ffn = _ContractFeedForward(device=device)

    def forward(
        self,
        value: Tensor,
        valid_mask: Tensor,
        runtime: _DropoutRuntime | None,
        *,
        ordinal_base: int,
    ) -> Tensor:
        normalized = self.attention_norm(value)
        attended = self.attention(
            normalized,
            normalized,
            valid_mask,
            runtime,
            attention_ordinal=ordinal_base,
            projection_ordinal=ordinal_base + 1,
        )
        value = _positive_zero_invalid(value + attended, valid_mask)
        feed_forward = self.ffn(
            self.ffn_norm(value),
            runtime,
            inner_ordinal=ordinal_base + 2,
            output_ordinal=ordinal_base + 3,
        )
        return _positive_zero_invalid(value + feed_forward, valid_mask)

    def early_pairs(self) -> tuple[tuple[str, nn.Parameter], ...]:
        bb = f"{self.block_index:02d}"
        prefix = f"tmr.block.{bb}"
        rows: list[tuple[str, nn.Parameter]] = []
        rows.extend(
            _attention_pairs(
                prefix + ".self.attn", self.attention, ("q", "k", "v", "o")
            )
        )
        rows.extend(_layer_norm_pairs(prefix + ".self.ln", self.attention_norm))
        rows.extend(_linear_pairs(prefix + ".ffn.in", self.ffn.in_projection))
        rows.extend(_linear_pairs(prefix + ".ffn.out", self.ffn.out_projection))
        rows.extend(_layer_norm_pairs(prefix + ".ffn.ln", self.ffn_norm))
        return tuple(rows)

    def late_pairs(self, actor: str) -> tuple[tuple[str, nn.Parameter], ...]:
        bb = f"{self.block_index:02d}"
        prefix = f"{actor}.blocks.{bb}"
        rows: list[tuple[str, nn.Parameter]] = []
        rows.extend(_layer_norm_pairs(prefix + ".ln_attn", self.attention_norm))
        rows.extend(
            _attention_pairs(
                prefix + ".self_attn",
                self.attention,
                ("q_proj", "k_proj", "v_proj", "out_proj"),
            )
        )
        rows.extend(_layer_norm_pairs(prefix + ".ln_ffn", self.ffn_norm))
        rows.extend(_linear_pairs(prefix + ".ffn.fc1", self.ffn.in_projection))
        rows.extend(_linear_pairs(prefix + ".ffn.fc2", self.ffn.out_projection))
        return tuple(rows)


class EarlyFusionMotionEncoder(_MotionEncoderBase):
    architecture = "early"

    def __init__(self, *, device: object = None) -> None:
        super().__init__()
        self.input_linear = _ContractLinear(786, 512, device=device)
        self.input_norm = _ContractLayerNorm(device=device)
        self.blocks = nn.ModuleList(
            _EncoderBlock(block, device=device) for block in range(4)
        )
        self.pool_query = nn.Parameter(
            torch.empty((512,), dtype=torch.float32, device=device)
        )
        self.register_buffer("position_encoding", _position_encoding(device=device))
        self._validate_parameter_inventory(require_meta=False)

    def canonical_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        rows: list[tuple[str, nn.Parameter]] = []
        rows.extend(_linear_pairs("tmr.input.linear", self.input_linear))
        rows.extend(_layer_norm_pairs("tmr.input.ln", self.input_norm))
        for block in self.blocks:
            if type(block) is not _EncoderBlock:
                raise TorchModelContractError("early block type drift")
            rows.extend(block.early_pairs())
        rows.append(("tmr.pool.query", self.pool_query))
        return tuple(rows)

    def forward(
        self,
        batch: batching.PreparedMotionBatch,
        *,
        dropout_bundle: OneForwardCanonicalDropoutBundle | None = None,
    ) -> MotionEmbeddings:
        _, _, _, early, valid, runtime = self._ordered_inputs(
            batch, dropout_bundle=dropout_bundle
        )
        time_steps = early.shape[2]
        position = self.position_encoding[:time_steps].unsqueeze(0).unsqueeze(0)
        value = self.input_norm(self.input_linear(early)) + position
        value = _positive_zero_invalid(value, valid)
        for block_index, block in enumerate(self.blocks):
            value = block(
                value,
                valid,
                runtime,
                ordinal_base=4 * block_index,
            )
        ordered = _masked_query_pool(
            value, self.pool_query, valid, scale_by_width=False
        ).contiguous()
        if runtime is not None:
            runtime.close()
        return MotionEmbeddings(ordered)


class _LateActorTower(nn.Module):
    def __init__(self, actor: str, actor_index: int, *, device: object) -> None:
        super().__init__()
        self.actor = actor
        self.actor_index = actor_index
        self.input_projection = _ContractLinear(262, 512, device=device)
        self.input_norm = _ContractLayerNorm(device=device)
        self.blocks = nn.ModuleList(
            _EncoderBlock(block, device=device) for block in range(4)
        )
        self.pool_query = nn.Parameter(
            torch.empty((512,), dtype=torch.float32, device=device)
        )

    def forward(
        self,
        value: Tensor,
        position_encoding: Tensor,
        valid_mask: Tensor,
        runtime: _DropoutRuntime | None,
    ) -> Tensor:
        value = self.input_norm(self.input_projection(value)) + position_encoding
        value = _positive_zero_invalid(value, valid_mask)
        for block_index, block in enumerate(self.blocks):
            value = block(
                value,
                valid_mask,
                runtime,
                ordinal_base=16 * self.actor_index + 4 * block_index,
            )
        return _masked_query_pool(
            value,
            self.pool_query,
            valid_mask,
            scale_by_width=True,
        )

    def canonical_pairs(self) -> tuple[tuple[str, nn.Parameter], ...]:
        rows: list[tuple[str, nn.Parameter]] = []
        rows.extend(_linear_pairs(f"{self.actor}.input_proj", self.input_projection))
        rows.extend(_layer_norm_pairs(f"{self.actor}.input_ln", self.input_norm))
        for block in self.blocks:
            if type(block) is not _EncoderBlock:
                raise TorchModelContractError("late block type drift")
            rows.extend(block.late_pairs(self.actor))
        rows.append((f"{self.actor}.pool.query", self.pool_query))
        return tuple(rows)


class LateFusionMotionEncoder(_MotionEncoderBase):
    architecture = "late"

    def __init__(self, *, device: object = None) -> None:
        super().__init__()
        self.actor_a = _LateActorTower("actor_a", 0, device=device)
        self.actor_b = _LateActorTower("actor_b", 1, device=device)
        self.fusion_projection = _ContractLinear(
            1024, 512, device=device
        )
        self.fusion_norm = _ContractLayerNorm(device=device)
        self.register_buffer("position_encoding", _position_encoding(device=device))
        self._validate_parameter_inventory(require_meta=False)

    def canonical_named_parameters(self) -> tuple[tuple[str, nn.Parameter], ...]:
        rows: list[tuple[str, nn.Parameter]] = []
        rows.extend(self.actor_a.canonical_pairs())
        rows.extend(self.actor_b.canonical_pairs())
        rows.extend(_linear_pairs("fusion.proj", self.fusion_projection))
        rows.extend(_layer_norm_pairs("fusion.ln", self.fusion_norm))
        return tuple(rows)

    def forward(
        self,
        batch: batching.PreparedMotionBatch,
        *,
        dropout_bundle: OneForwardCanonicalDropoutBundle | None = None,
    ) -> MotionEmbeddings:
        role_a, role_b, _, _, valid, runtime = self._ordered_inputs(
            batch, dropout_bundle=dropout_bundle
        )
        time_steps = role_a.shape[2]
        position = self.position_encoding[:time_steps].unsqueeze(0).unsqueeze(0)
        pooled_a = self.actor_a(role_a, position, valid, runtime)
        pooled_b = self.actor_b(role_b, position, valid, runtime)
        fused = self.fusion_norm(
            self.fusion_projection(torch.cat((pooled_a, pooled_b), dim=-1))
        )
        ordered = _l2(fused).contiguous()
        if runtime is not None:
            runtime.close()
        return MotionEmbeddings(ordered)


def build_motion_encoder(
    architecture: str, *, device: object = "meta"
) -> MimeMotionEncoder | EarlyFusionMotionEncoder | LateFusionMotionEncoder:
    if type(architecture) is not str:
        raise TypeError("architecture must be an exact built-in str")
    if architecture == "mime":
        return MimeMotionEncoder(device=device)
    if architecture == "early":
        return EarlyFusionMotionEncoder(device=device)
    if architecture == "late":
        return LateFusionMotionEncoder(device=device)
    raise TorchModelContractError(f"unknown motion architecture: {architecture!r}")


__all__ = [
    "DROPOUT_HOLD_CODE",
    "EarlyFusionMotionEncoder",
    "LateFusionMotionEncoder",
    "MimeMotionEncoder",
    "MotionEmbeddings",
    "MotionInventoryAudit",
    "STATUS",
    "TorchModelContractError",
    "TorchModelContractHold",
    "build_motion_encoder",
]
