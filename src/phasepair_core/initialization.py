"""Byte-exact, authority-zero initializers for PhasePair motion encoders.

The public module constants are descriptive only. Every executable contract
value is captured once in an immutable snapshot and then closed over by the
public callables. Rebinding a module global therefore cannot change parameter
bytes, accepted model types, state serialization, or receipt semantics.

This module remains data-free and non-production. Its receipt is deliberately
``AUTHORITY=0`` and ``HOLD`` because the 151744 runtime/libm oracle and its
hash-selected sixteen-index evidence are not content-pinned by a data-free
initializer alone.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import json
import math
import struct
import sys
from typing import Callable, Iterable

import torch
from torch import Tensor, nn

from . import batching, contracts, torch_models

_BATCHING_MODULE_AT_IMPORT = batching


STATUS = "HOLD_RUNTIME_LIBM_ORACLE_UNPINNED_AUTHORITY0"
AUTHORITY = 0
RECEIPT_SCHEMA = "phasepair-motion-initializer-receipt-v2"
MIME_EARLY_DOMAIN = "phasepair-init-v1"
LATE_DOMAIN = "phasepair-late-fusion-init-v1"
STATE_DOMAIN = b"phasepair-motion-initial-state-v1\x00"
INVENTORY_SCHEMA = "phasepair-motion-initializer-inventory-v1"

_UINT64_MAX = (1 << 64) - 1
_FLOAT32_BYTES = 4
_CANONICAL_PE_SHA256 = (
    "69401066bb295d8d3c66cea0b4c839cad3e967756f242791fac6defc00141ae0"
)


class InitializationContractError(ValueError):
    """A model, seed, initializer, transaction, or receipt invariant failed."""


@dataclass(frozen=True, slots=True)
class _FrozenRow:
    name: str
    shape: tuple[int, ...]
    semantic_class: str

    @property
    def numel(self) -> int:
        result = 1
        for dimension in self.shape:
            result *= dimension
        return result


@dataclass(frozen=True, slots=True)
class _ValidatedModel:
    architecture: str
    pairs: tuple[tuple[str, nn.Parameter], ...]
    rows: tuple[_FrozenRow, ...]
    position_encoding: Tensor
    parameter_intervals: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class _PreparedTensor:
    name: str
    shape: tuple[int, ...]
    parameter: nn.Parameter
    raw: bytearray
    sha256: str


@dataclass(frozen=True, slots=True)
class _OriginalTensor:
    name: str
    shape: tuple[int, ...]
    parameter: nn.Parameter
    raw: bytearray


@dataclass(frozen=True, slots=True)
class _RegistryModuleSpec:
    path: str
    module_type: type[nn.Module]
    parameter_names: tuple[str, ...]
    buffer_names: tuple[str, ...]
    child_names: tuple[str, ...]
    instance_field_names: tuple[str, ...]
    literal_instance_fields: tuple[tuple[str, object], ...]


@dataclass(frozen=True, slots=True)
class _ContractSnapshot:
    rows_by_architecture: tuple[tuple[str, tuple[_FrozenRow, ...]], ...]
    expected_ledgers: tuple[tuple[str, int, int], ...]
    model_specs: tuple[
        tuple[
            type[nn.Module],
            str,
            tuple[tuple[str, str], ...],
            tuple[_RegistryModuleSpec, ...],
        ],
        ...,
    ]
    parameter_type: type[nn.Parameter]
    tensor_type: type[Tensor]
    float32_dtype: object
    strided_layout: object
    copy_primitive: object
    tensor_detach: object
    tensor_numpy: object
    tensor_is_contiguous: object
    tensor_untyped_storage: object
    tensor_storage_offset: object
    tensor_numel: object
    tensor_element_size: object
    storage_data_ptr: object
    storage_nbytes: object
    frombuffer: object
    no_grad: object
    object_getattribute: object
    ordered_dict_type: type
    hook_map_fields: tuple[str, ...]
    sha256_factory: object
    json_dumps: object
    json_loads: object
    sqrt: object
    struct_pack: object
    struct_pack_into: object
    byteorder: str
    python_implementation: str
    python_version: str
    torch_version: str
    float_radix: int
    float_mant_dig: int
    float_rounds: int
    canonical_pe_sha256: bytes
    validated_type: type[_ValidatedModel]
    prepared_type: type[_PreparedTensor]
    original_type: type[_OriginalTensor]
    registry_spec_type: type[_RegistryModuleSpec]
    error_type: type[InitializationContractError]


_MODULE_BASE_INSTANCE_FIELDS = (
    "training",
    "_parameters",
    "_buffers",
    "_non_persistent_buffers_set",
    "_backward_pre_hooks",
    "_backward_hooks",
    "_is_full_backward_hook",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
    "_modules",
)

_MODULE_HOOK_MAP_FIELDS = (
    "_backward_pre_hooks",
    "_backward_hooks",
    "_forward_hooks",
    "_forward_hooks_with_kwargs",
    "_forward_hooks_always_called",
    "_forward_pre_hooks",
    "_forward_pre_hooks_with_kwargs",
    "_state_dict_hooks",
    "_state_dict_pre_hooks",
    "_load_state_dict_pre_hooks",
    "_load_state_dict_post_hooks",
)


def _freeze_rows(architecture: str) -> tuple[_FrozenRow, ...]:
    return tuple(
        _FrozenRow(row.name, tuple(row.shape), row.semantic_class)
        for row in contracts.canonical_motion_rows(architecture)
    )


def _mime_registered_mapping() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = [
        ("actor_embedding_a", "mime.input.actor_embedding.a"),
        ("actor_embedding_b", "mime.input.actor_embedding.b"),
        ("relation_type_embedding", "mime.relation.type_embedding"),
        ("pool_query", "mime.pool.query"),
    ]
    for registered, canonical in (
        ("input_a_linear", "mime.input.a.linear"),
        ("input_a_norm", "mime.input.a.ln"),
        ("input_b_linear", "mime.input.b.linear"),
        ("input_b_norm", "mime.input.b.ln"),
        ("relation_linear", "mime.relation.linear"),
        ("relation_norm", "mime.relation.ln"),
    ):
        pairs.extend(
            (f"{registered}.{suffix}", f"{canonical}.{suffix}")
            for suffix in ("weight", "bias")
        )
    for block in range(4):
        bb = f"{block:02d}"
        for registered_actor, canonical_actor in (("a", "a"), ("b", "b")):
            for projection in ("q", "k", "v", "o"):
                for suffix in ("weight", "bias"):
                    pairs.append(
                        (
                            f"blocks.{block}.self_{registered_actor}_attention."
                            f"{projection}.{suffix}",
                            f"mime.block.{bb}.self.{canonical_actor}.attn."
                            f"{projection}.{suffix}",
                        )
                    )
            for suffix in ("weight", "bias"):
                pairs.append(
                    (
                        f"blocks.{block}.self_{registered_actor}_norm.{suffix}",
                        f"mime.block.{bb}.self.{canonical_actor}.ln.{suffix}",
                    )
                )
        for registered_direction, canonical_direction in (
            ("a", "a_from_b"),
            ("b", "b_from_a"),
        ):
            for projection in ("q", "k", "v", "o"):
                for suffix in ("weight", "bias"):
                    pairs.append(
                        (
                            f"blocks.{block}.cross_{registered_direction}_attention."
                            f"{projection}.{suffix}",
                            f"mime.block.{bb}.cross.{canonical_direction}.attn."
                            f"{projection}.{suffix}",
                        )
                    )
            for registered_norm, canonical_norm in (("query", "q"), ("kv", "kv")):
                for suffix in ("weight", "bias"):
                    pairs.append(
                        (
                            f"blocks.{block}.cross_{registered_direction}_"
                            f"{registered_norm}_norm.{suffix}",
                            f"mime.block.{bb}.cross.{canonical_direction}."
                            f"ln_{canonical_norm}.{suffix}",
                        )
                    )
        for actor in ("a", "b"):
            for registered_projection, canonical_projection in (
                ("in_projection", "in"),
                ("out_projection", "out"),
            ):
                for suffix in ("weight", "bias"):
                    pairs.append(
                        (
                            f"blocks.{block}.ffn_{actor}.{registered_projection}."
                            f"{suffix}",
                            f"mime.block.{bb}.ffn.{actor}.{canonical_projection}."
                            f"{suffix}",
                        )
                    )
            for suffix in ("weight", "bias"):
                pairs.append(
                    (
                        f"blocks.{block}.ffn_{actor}_norm.{suffix}",
                        f"mime.block.{bb}.ffn.{actor}.ln.{suffix}",
                    )
                )
    for registered, canonical in (
        ("fusion_in", "mime.fusion.in"),
        ("fusion_out", "mime.fusion.out"),
    ):
        pairs.extend(
            (f"{registered}.{suffix}", f"{canonical}.{suffix}")
            for suffix in ("weight", "bias")
        )
    return tuple(pairs)


def _early_registered_mapping() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = [("pool_query", "tmr.pool.query")]
    for registered, canonical in (
        ("input_linear", "tmr.input.linear"),
        ("input_norm", "tmr.input.ln"),
    ):
        pairs.extend(
            (f"{registered}.{suffix}", f"{canonical}.{suffix}")
            for suffix in ("weight", "bias")
        )
    for block in range(4):
        bb = f"{block:02d}"
        for suffix in ("weight", "bias"):
            pairs.append(
                (
                    f"blocks.{block}.attention_norm.{suffix}",
                    f"tmr.block.{bb}.self.ln.{suffix}",
                )
            )
        for projection in ("q", "k", "v", "o"):
            for suffix in ("weight", "bias"):
                pairs.append(
                    (
                        f"blocks.{block}.attention.{projection}.{suffix}",
                        f"tmr.block.{bb}.self.attn.{projection}.{suffix}",
                    )
                )
        for suffix in ("weight", "bias"):
            pairs.append(
                (
                    f"blocks.{block}.ffn_norm.{suffix}",
                    f"tmr.block.{bb}.ffn.ln.{suffix}",
                )
            )
        for registered_projection, canonical_projection in (
            ("in_projection", "in"),
            ("out_projection", "out"),
        ):
            for suffix in ("weight", "bias"):
                pairs.append(
                    (
                        f"blocks.{block}.ffn.{registered_projection}.{suffix}",
                        f"tmr.block.{bb}.ffn.{canonical_projection}.{suffix}",
                    )
                )
    return tuple(pairs)


def _late_registered_mapping() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for actor in ("actor_a", "actor_b"):
        pairs.append((f"{actor}.pool_query", f"{actor}.pool.query"))
        for registered, canonical in (
            ("input_projection", "input_proj"),
            ("input_norm", "input_ln"),
        ):
            pairs.extend(
                (
                    f"{actor}.{registered}.{suffix}",
                    f"{actor}.{canonical}.{suffix}",
                )
                for suffix in ("weight", "bias")
            )
        for block in range(4):
            bb = f"{block:02d}"
            for suffix in ("weight", "bias"):
                pairs.append(
                    (
                        f"{actor}.blocks.{block}.attention_norm.{suffix}",
                        f"{actor}.blocks.{bb}.ln_attn.{suffix}",
                    )
                )
            for registered_projection, canonical_projection in (
                ("q", "q_proj"),
                ("k", "k_proj"),
                ("v", "v_proj"),
                ("o", "out_proj"),
            ):
                for suffix in ("weight", "bias"):
                    pairs.append(
                        (
                            f"{actor}.blocks.{block}.attention."
                            f"{registered_projection}.{suffix}",
                            f"{actor}.blocks.{bb}.self_attn."
                            f"{canonical_projection}.{suffix}",
                        )
                    )
            for suffix in ("weight", "bias"):
                pairs.append(
                    (
                        f"{actor}.blocks.{block}.ffn_norm.{suffix}",
                        f"{actor}.blocks.{bb}.ln_ffn.{suffix}",
                    )
                )
            for registered_projection, canonical_projection in (
                ("in_projection", "fc1"),
                ("out_projection", "fc2"),
            ):
                for suffix in ("weight", "bias"):
                    pairs.append(
                        (
                            f"{actor}.blocks.{block}.ffn."
                            f"{registered_projection}.{suffix}",
                            f"{actor}.blocks.{bb}.ffn."
                            f"{canonical_projection}.{suffix}",
                        )
                    )
    for registered, canonical in (
        ("fusion_projection", "fusion.proj"),
        ("fusion_norm", "fusion.ln"),
    ):
        pairs.extend(
            (f"{registered}.{suffix}", f"{canonical}.{suffix}")
            for suffix in ("weight", "bias")
        )
    return tuple(pairs)


def _checked_registered_mapping(
    architecture: str, mapping: tuple[tuple[str, str], ...]
) -> tuple[tuple[str, str], ...]:
    rows = _freeze_rows(architecture)
    if (
        type(mapping) is not tuple
        or any(
            type(pair) is not tuple
            or len(pair) != 2
            or type(pair[0]) is not str
            or type(pair[1]) is not str
            for pair in mapping
        )
    ):
        raise AssertionError("initializer registered-name mapping type mismatch")
    registered = tuple(pair[0] for pair in mapping)
    canonical = tuple(pair[1] for pair in mapping)
    if len(registered) != len(set(registered)) or len(canonical) != len(set(canonical)):
        raise AssertionError("initializer registered-name mapping is not bijective")
    if set(canonical) != {row.name for row in rows} or len(mapping) != len(rows):
        raise AssertionError("initializer registered-name mapping inventory mismatch")
    return mapping


def _module_type_for_registered_path(
    architecture: str,
    path: str,
    root_type: type[nn.Module],
) -> type[nn.Module]:
    if path == "":
        return root_type
    components = path.split(".")
    local_name = components[-1]
    if local_name == "blocks":
        return nn.ModuleList
    if local_name.isdigit() and len(components) >= 2 and components[-2] == "blocks":
        return (
            torch_models._MimeBlock
            if architecture == "mime"
            else torch_models._EncoderBlock
        )
    if architecture == "late" and path in ("actor_a", "actor_b"):
        return torch_models._LateActorTower
    if local_name in {
        "input_a_linear",
        "input_b_linear",
        "relation_linear",
        "fusion_in",
        "fusion_out",
        "input_linear",
        "input_projection",
        "fusion_projection",
        "q",
        "k",
        "v",
        "o",
        "in_projection",
        "out_projection",
    }:
        return torch_models._ContractLinear
    if local_name == "attention" or local_name.endswith("_attention"):
        return torch_models._ContractAttention
    if local_name in {"ffn", "ffn_a", "ffn_b"}:
        return torch_models._ContractFeedForward
    if local_name.endswith("norm"):
        return torch_models._ContractLayerNorm
    raise AssertionError(f"unresolved registered module type: {architecture}:{path}")


def _module_instance_field_schema(
    architecture: str,
    path: str,
    module_type: type[nn.Module],
    mapping_by_registered_name: dict[str, str],
    rows_by_name: dict[str, _FrozenRow],
) -> tuple[tuple[str, ...], tuple[tuple[str, object], ...]]:
    literal_fields: tuple[tuple[str, object], ...]
    if module_type is torch_models._ContractLinear:
        registered_weight = f"{path}.weight" if path else "weight"
        canonical_weight = mapping_by_registered_name.get(registered_weight)
        if canonical_weight is None:
            raise AssertionError(f"linear module has no mapped weight: {path}")
        shape = rows_by_name[canonical_weight].shape
        if len(shape) != 2:
            raise AssertionError(f"linear weight is not rank two: {canonical_weight}")
        literal_fields = (
            ("in_features", shape[1]),
            ("out_features", shape[0]),
        )
    elif module_type is torch_models._ContractLayerNorm:
        registered_weight = f"{path}.weight" if path else "weight"
        canonical_weight = mapping_by_registered_name.get(registered_weight)
        if canonical_weight is None:
            raise AssertionError(f"layer norm has no mapped weight: {path}")
        shape = rows_by_name[canonical_weight].shape
        if len(shape) != 1:
            raise AssertionError(
                f"layer norm weight is not rank one: {canonical_weight}"
            )
        literal_fields = (("width", shape[0]),)
    elif module_type in (torch_models._MimeBlock, torch_models._EncoderBlock):
        block_component = path.rsplit(".", 1)[-1]
        if not block_component.isdigit():
            raise AssertionError(f"block path has no numeric index: {path}")
        literal_fields = (("block_index", int(block_component)),)
    elif module_type is torch_models._LateActorTower:
        if architecture != "late" or path not in ("actor_a", "actor_b"):
            raise AssertionError(f"late actor tower path mismatch: {architecture}:{path}")
        literal_fields = (
            ("actor", path),
            ("actor_index", 0 if path == "actor_a" else 1),
        )
    elif module_type in (
        torch_models.MimeMotionEncoder,
        torch_models.EarlyFusionMotionEncoder,
        torch_models.LateFusionMotionEncoder,
        nn.ModuleList,
        torch_models._ContractAttention,
        torch_models._ContractFeedForward,
    ):
        literal_fields = ()
    else:
        raise AssertionError(f"unresolved module instance schema: {architecture}:{path}")
    return (
        _MODULE_BASE_INSTANCE_FIELDS
        + tuple(field_name for field_name, _ in literal_fields),
        literal_fields,
    )


def _registered_module_specs(
    architecture: str,
    root_type: type[nn.Module],
    mapping: tuple[tuple[str, str], ...],
) -> tuple[_RegistryModuleSpec, ...]:
    mapping_by_registered_name = dict(mapping)
    rows_by_name = {row.name: row for row in _freeze_rows(architecture)}
    parameters_by_path: dict[str, list[str]] = {"": []}
    children_by_path: dict[str, list[str]] = {"": []}
    for registered_name, _ in mapping:
        components = registered_name.split(".")
        if any(not component for component in components):
            raise AssertionError("registered name contains an empty component")
        module_components = components[:-1]
        parent_path = ""
        for component in module_components:
            child_path = component if not parent_path else f"{parent_path}.{component}"
            parameters_by_path.setdefault(child_path, [])
            children_by_path.setdefault(child_path, [])
            if component not in children_by_path[parent_path]:
                children_by_path[parent_path].append(component)
            parent_path = child_path
        parameters_by_path[parent_path].append(components[-1])

    ordered_paths: list[str] = []

    def visit(path: str) -> None:
        ordered_paths.append(path)
        for child_name in children_by_path[path]:
            child_path = child_name if not path else f"{path}.{child_name}"
            visit(child_path)

    visit("")
    specs: list[_RegistryModuleSpec] = []
    for path in ordered_paths:
        module_type = _module_type_for_registered_path(architecture, path, root_type)
        instance_field_names, literal_instance_fields = (
            _module_instance_field_schema(
                architecture,
                path,
                module_type,
                mapping_by_registered_name,
                rows_by_name,
            )
        )
        specs.append(
            _RegistryModuleSpec(
                path,
                module_type,
                tuple(parameters_by_path[path]),
                ("position_encoding",) if path == "" else (),
                tuple(children_by_path[path]),
                instance_field_names,
                literal_instance_fields,
            )
        )
    return tuple(specs)


def _model_spec(
    model_type: type[nn.Module],
    architecture: str,
    mapping: tuple[tuple[str, str], ...],
) -> tuple[
    type[nn.Module],
    str,
    tuple[tuple[str, str], ...],
    tuple[_RegistryModuleSpec, ...],
]:
    checked_mapping = _checked_registered_mapping(architecture, mapping)
    return (
        model_type,
        architecture,
        checked_mapping,
        _registered_module_specs(architecture, model_type, checked_mapping),
    )


_CAPTURED = _ContractSnapshot(
    rows_by_architecture=tuple(
        (architecture, _freeze_rows(architecture))
        for architecture in ("mime", "early", "late")
    ),
    expected_ledgers=(
        ("mime", 244, 35_111_936),
        ("early", 69, 13_014_016),
        ("late", 142, 26_017_280),
    ),
    model_specs=(
        _model_spec(
            torch_models.MimeMotionEncoder,
            "mime",
            _mime_registered_mapping(),
        ),
        _model_spec(
            torch_models.EarlyFusionMotionEncoder,
            "early",
            _early_registered_mapping(),
        ),
        _model_spec(
            torch_models.LateFusionMotionEncoder,
            "late",
            _late_registered_mapping(),
        ),
    ),
    parameter_type=nn.Parameter,
    tensor_type=Tensor,
    float32_dtype=torch.float32,
    strided_layout=torch.strided,
    copy_primitive=Tensor.copy_,
    tensor_detach=Tensor.detach,
    tensor_numpy=Tensor.numpy,
    tensor_is_contiguous=Tensor.is_contiguous,
    tensor_untyped_storage=Tensor.untyped_storage,
    tensor_storage_offset=Tensor.storage_offset,
    tensor_numel=Tensor.numel,
    tensor_element_size=Tensor.element_size,
    storage_data_ptr=torch.UntypedStorage.data_ptr,
    storage_nbytes=torch.UntypedStorage.nbytes,
    frombuffer=torch.frombuffer,
    no_grad=torch.no_grad,
    object_getattribute=object.__getattribute__,
    ordered_dict_type=OrderedDict,
    hook_map_fields=_MODULE_HOOK_MAP_FIELDS,
    sha256_factory=hashlib.sha256,
    json_dumps=json.dumps,
    json_loads=json.loads,
    sqrt=math.sqrt,
    struct_pack=struct.pack,
    struct_pack_into=struct.pack_into,
    byteorder=sys.byteorder,
    python_implementation=sys.implementation.name,
    python_version=(
        f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    ),
    torch_version=str(torch.__version__),
    float_radix=sys.float_info.radix,
    float_mant_dig=sys.float_info.mant_dig,
    float_rounds=sys.float_info.rounds,
    canonical_pe_sha256=bytes.fromhex(_CANONICAL_PE_SHA256),
    validated_type=_ValidatedModel,
    prepared_type=_PreparedTensor,
    original_type=_OriginalTensor,
    registry_spec_type=_RegistryModuleSpec,
    error_type=InitializationContractError,
)

# Compatibility-only diagnostic globals. Executable paths never read them.
_MODEL_TYPES: dict[type[nn.Module], str] = {
    torch_models.MimeMotionEncoder: "mime",
    torch_models.EarlyFusionMotionEncoder: "early",
    torch_models.LateFusionMotionEncoder: "late",
}


def _rows_for(
    architecture: str, snapshot: _ContractSnapshot
) -> tuple[_FrozenRow, ...]:
    for name, rows in snapshot.rows_by_architecture:
        if architecture == name:
            return rows
    raise snapshot.error_type(f"unknown motion architecture: {architecture!r}")


def _ledger_for(architecture: str, snapshot: _ContractSnapshot) -> tuple[int, int]:
    for name, tensor_count, numel in snapshot.expected_ledgers:
        if architecture == name:
            return tensor_count, numel
    raise snapshot.error_type(f"unknown motion architecture: {architecture!r}")


def _require_seed(seed: object) -> int:
    if type(seed) is not int:
        raise TypeError("seed must be an exact built-in int")
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise ValueError("seed must be in [0, 2^64-1]")
    return seed


def _require_architecture(
    architecture: object,
    error_type: type[InitializationContractError] = InitializationContractError,
) -> str:
    if type(architecture) is not str:
        raise TypeError("architecture must be an exact built-in str")
    if architecture not in ("mime", "early", "late"):
        raise error_type(
            f"unknown motion architecture: {architecture!r}"
        )
    return architecture


def _initializer_domain(architecture: str) -> str:
    if architecture == "late":
        return "phasepair-late-fusion-init-v1"
    return "phasepair-init-v1"


def _serialization_rows(
    architecture: str, rows: tuple[_FrozenRow, ...]
) -> tuple[_FrozenRow, ...]:
    if architecture == "late":
        return rows
    return tuple(sorted(rows, key=lambda row: row.name.encode("utf-8")))


def _canonical_json_bytes(value: object, snapshot: _ContractSnapshot) -> bytes:
    return (
        snapshot.json_dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _tensor_raw_bytes(tensor: Tensor, snapshot: _ContractSnapshot) -> bytes:
    if (
        tensor.device.type != "cpu"
        or tensor.dtype != snapshot.float32_dtype
        or not snapshot.tensor_is_contiguous(tensor)
    ):
        raise snapshot.error_type("state tensor must be contiguous CPU float32")
    detached = snapshot.tensor_detach(tensor)
    return snapshot.tensor_numpy(detached).tobytes(order="C")


def _parameter_interval(
    parameter: nn.Parameter, snapshot: _ContractSnapshot
) -> tuple[int, int]:
    if (
        parameter.layout != snapshot.strided_layout
        or not snapshot.tensor_is_contiguous(parameter)
        or snapshot.tensor_storage_offset(parameter) != 0
    ):
        raise snapshot.error_type(
            "parameter must have contiguous exact-offset strided storage"
        )
    storage = snapshot.tensor_untyped_storage(parameter)
    required_bytes = (
        snapshot.tensor_numel(parameter) * snapshot.tensor_element_size(parameter)
    )
    if snapshot.storage_nbytes(storage) != required_bytes:
        raise snapshot.error_type("parameter storage extent is not exact")
    start = snapshot.storage_data_ptr(storage)
    if start == 0:
        raise snapshot.error_type("parameter storage has a null data pointer")
    return start, start + required_bytes


def _assert_disjoint(
    intervals: Iterable[tuple[int, int]], label: str, snapshot: _ContractSnapshot
) -> None:
    ordered = sorted(intervals)
    for previous, current in zip(ordered, ordered[1:]):
        if current[0] < previous[1]:
            raise snapshot.error_type(f"{label} storage ranges overlap")


def _validate_model_impl(
    model: object,
    snapshot: _ContractSnapshot,
    rows_for: object = _rows_for,
    ledger_for: object = _ledger_for,
    parameter_interval: object = _parameter_interval,
    assert_disjoint: object = _assert_disjoint,
    tensor_raw_bytes: object = _tensor_raw_bytes,
) -> _ValidatedModel:
    architecture: str | None = None
    registered_mapping: tuple[tuple[str, str], ...] | None = None
    registry_specs: tuple[_RegistryModuleSpec, ...] | None = None
    for model_type, label, mapping, module_specs in snapshot.model_specs:
        if type(model) is model_type:
            architecture = label
            registered_mapping = mapping
            registry_specs = module_specs
            break
    if (
        architecture is None
        or registered_mapping is None
        or registry_specs is None
    ):
        raise TypeError("model must be an exact PhasePair motion encoder type")
    object_getattribute = snapshot.object_getattribute
    if (
        type(object_getattribute(model, "architecture")) is not str
        or object_getattribute(model, "architecture") != architecture
    ):
        raise snapshot.error_type("model architecture label mismatch")
    if snapshot.byteorder != "little":
        raise snapshot.error_type("initializer requires a little-endian runtime")

    rows = rows_for(architecture, snapshot)
    registered_names = tuple(pair[0] for pair in registered_mapping)
    mapped_canonical_names = tuple(pair[1] for pair in registered_mapping)
    row_names = tuple(row.name for row in rows)
    if (
        len(registered_names) != len(set(registered_names))
        or len(mapped_canonical_names) != len(set(mapped_canonical_names))
        or set(mapped_canonical_names) != set(row_names)
        or len(registered_mapping) != len(rows)
    ):
        raise snapshot.error_type("captured registered-name mapping is not bijective")
    if (
        not registry_specs
        or registry_specs[0].path != ""
        or len({spec.path for spec in registry_specs}) != len(registry_specs)
        or any(type(spec) is not snapshot.registry_spec_type for spec in registry_specs)
    ):
        raise snapshot.error_type("captured module registry specification is malformed")
    specs_by_path = {spec.path: spec for spec in registry_specs}
    mapping_by_registered_name = dict(registered_mapping)
    observed_registered_names: list[str] = []
    parameters_by_canonical_name: dict[str, nn.Parameter] = {}
    object_ids: set[int] = set()
    interval_by_canonical_name: dict[str, tuple[int, int]] = {}
    rows_by_name = {row.name: row for row in rows}
    seen_module_ids: set[int] = set()
    position_encoding: Tensor | None = None

    def walk(module: object, path: str) -> None:
        nonlocal position_encoding
        spec = specs_by_path.get(path)
        if spec is None or type(module) is not spec.module_type:
            raise snapshot.error_type(f"registered module type/path mismatch: {path}")
        if id(module) in seen_module_ids:
            raise snapshot.error_type("registered module cycle or alias detected")
        seen_module_ids.add(id(module))
        instance_dict = object_getattribute(module, "__dict__")
        if type(instance_dict) is not dict:
            raise snapshot.error_type("module instance dictionary must be exact dict")
        instance_field_names = tuple(instance_dict.keys())
        if (
            any(type(name) is not str for name in instance_field_names)
            or instance_field_names != spec.instance_field_names
        ):
            raise snapshot.error_type(
                f"module instance field name/order mismatch at path {path!r}"
            )
        if type(instance_dict["training"]) is not bool:
            raise snapshot.error_type(
                f"module training flag must be an exact bool at path {path!r}"
            )
        parameters = instance_dict["_parameters"]
        buffers = instance_dict["_buffers"]
        modules = instance_dict["_modules"]
        nonpersistent = instance_dict["_non_persistent_buffers_set"]
        if (
            type(parameters) is not dict
            or type(buffers) is not dict
            or type(modules) is not dict
        ):
            raise snapshot.error_type("module registries must be exact built-in dicts")
        if type(nonpersistent) is not set or nonpersistent:
            raise snapshot.error_type(
                "non-persistent buffer registry must be an exact empty set"
            )
        if instance_dict["_is_full_backward_hook"] is not None:
            raise snapshot.error_type(
                f"module backward-hook flag must be None at path {path!r}"
            )
        for field_name in snapshot.hook_map_fields:
            hook_map = instance_dict[field_name]
            if type(hook_map) is not snapshot.ordered_dict_type or len(hook_map) != 0:
                raise snapshot.error_type(
                    f"module hook registry must be an exact empty OrderedDict: "
                    f"{path}:{field_name}"
                )
        for field_name, expected_value in spec.literal_instance_fields:
            observed_value = instance_dict[field_name]
            if (
                type(observed_value) is not type(expected_value)
                or observed_value != expected_value
            ):
                raise snapshot.error_type(
                    f"module literal instance field mismatch: {path}:{field_name}"
                )
        for registry, expected_names, label in (
            (parameters, spec.parameter_names, "parameter"),
            (buffers, spec.buffer_names, "buffer"),
            (modules, spec.child_names, "child-module"),
        ):
            keys = tuple(registry.keys())
            if any(type(name) is not str for name in keys) or keys != expected_names:
                raise snapshot.error_type(
                    f"{label} registry name/order mismatch at module {path!r}"
                )

        for local_name, parameter in parameters.items():
            registered_name = local_name if not path else f"{path}.{local_name}"
            canonical_name = mapping_by_registered_name.get(registered_name)
            if canonical_name is None:
                raise snapshot.error_type(
                    f"registered parameter is absent from mapping: {registered_name}"
                )
            if type(parameter) is not snapshot.parameter_type:
                raise TypeError(
                    f"registered parameter {registered_name} must be exactly nn.Parameter"
                )
            if id(parameter) in object_ids:
                raise snapshot.error_type("registered parameters share an object")
            object_ids.add(id(parameter))
            row = rows_by_name[canonical_name]
            if tuple(parameter.shape) != row.shape:
                raise snapshot.error_type(f"parameter shape mismatch: {canonical_name}")
            if (
                parameter.dtype != snapshot.float32_dtype
                or parameter.device.type != "cpu"
            ):
                raise snapshot.error_type(
                    "parameter must be CPU float32 before initialization: "
                    f"{canonical_name}"
                )
            if not parameter.requires_grad or not parameter.is_leaf:
                raise snapshot.error_type(
                    f"parameter gradient/leaf invariant mismatch: {canonical_name}"
                )
            if parameter.grad is not None:
                raise snapshot.error_type(
                    f"parameter has a stale gradient before initialization: {canonical_name}"
                )
            observed_registered_names.append(registered_name)
            parameters_by_canonical_name[canonical_name] = parameter
            interval_by_canonical_name[canonical_name] = parameter_interval(
                parameter, snapshot
            )

        for local_name, buffer in buffers.items():
            registered_name = local_name if not path else f"{path}.{local_name}"
            if registered_name != "position_encoding":
                raise snapshot.error_type(f"unknown registered buffer: {registered_name}")
            if type(buffer) is not snapshot.tensor_type:
                raise TypeError("position_encoding must be an exact torch.Tensor")
            if position_encoding is not None:
                raise snapshot.error_type("position_encoding buffer is duplicated")
            position_encoding = buffer

        for local_name, child in modules.items():
            child_path = local_name if not path else f"{path}.{local_name}"
            walk(child, child_path)

    walk(model, "")
    if len(seen_module_ids) != len(registry_specs):
        raise snapshot.error_type("registered module census mismatch")
    if tuple(observed_registered_names) != registered_names:
        raise snapshot.error_type("registered parameter traversal/order mismatch")
    if position_encoding is None:
        raise snapshot.error_type("canonical position_encoding buffer is absent")
    checked_pairs = tuple(
        (row.name, parameters_by_canonical_name[row.name]) for row in rows
    )
    intervals = tuple(interval_by_canonical_name[row.name] for row in rows)
    assert_disjoint(intervals, "parameter", snapshot)

    tensor_count, numel = ledger_for(architecture, snapshot)
    if len(checked_pairs) != tensor_count or sum(
        snapshot.tensor_numel(parameter) for _, parameter in checked_pairs
    ) != numel:
        raise snapshot.error_type("canonical parameter ledger mismatch")

    if (
        tuple(position_encoding.shape) != (300, 512)
        or position_encoding.dtype != snapshot.float32_dtype
        or position_encoding.device.type != "cpu"
        or position_encoding.layout != snapshot.strided_layout
        or not snapshot.tensor_is_contiguous(position_encoding)
        or position_encoding.requires_grad
        or snapshot.tensor_storage_offset(position_encoding) != 0
    ):
        raise snapshot.error_type("position_encoding buffer invariant mismatch")
    pe_storage = snapshot.tensor_untyped_storage(position_encoding)
    pe_required_bytes = (
        snapshot.tensor_numel(position_encoding)
        * snapshot.tensor_element_size(position_encoding)
    )
    pe_nbytes = snapshot.storage_nbytes(pe_storage)
    pe_start = snapshot.storage_data_ptr(pe_storage)
    if pe_nbytes != pe_required_bytes or pe_start == 0:
        raise snapshot.error_type("position_encoding storage extent is not exact")
    pe_interval = (pe_start, pe_start + pe_required_bytes)
    assert_disjoint((*intervals, pe_interval), "parameter/buffer", snapshot)

    pe_raw = tensor_raw_bytes(position_encoding, snapshot)
    if (
        len(pe_raw) != 300 * 512 * 4
        or snapshot.sha256_factory(pe_raw).digest() != snapshot.canonical_pe_sha256
    ):
        raise snapshot.error_type(
            "position_encoding bytes do not match the frozen [300,512] oracle"
        )
    return snapshot.validated_type(
        architecture,
        checked_pairs,
        rows,
        position_encoding,
        intervals,
    )


def _constant_bytes(
    numel: int, value: float, snapshot: _ContractSnapshot
) -> bytearray:
    raw_word = snapshot.struct_pack("<f", value)
    return bytearray(raw_word * numel)


def _counter_bytes(
    *,
    domain_prefix: bytes,
    seed: int,
    name: str,
    numel: int,
    bound: float,
    late: bool,
    post_scale: float | None,
    snapshot: _ContractSnapshot,
) -> bytearray:
    name_raw = name.encode("ascii" if late else "utf-8", "strict")
    if len(name_raw) > 65535:
        raise snapshot.error_type("parameter name exceeds uint16 length")
    prefix = (
        domain_prefix
        + seed.to_bytes(8, "big")
        + len(name_raw).to_bytes(2, "big")
        + name_raw
    )
    base = snapshot.sha256_factory(prefix)
    raw = bytearray(numel * 4)
    denominator_65 = float(1 << 65)
    denominator_53 = float(1 << 53)
    for index in range(numel):
        digest = base.copy()
        digest.update(index.to_bytes(8, "big"))
        value64 = int.from_bytes(digest.digest()[:8], "big")
        if late:
            unit = ((value64 >> 11) + 0.5) / denominator_53
        else:
            unit = ((value64 << 1) + 1) / denominator_65
        initialized = (2.0 * unit - 1.0) * bound
        if post_scale is not None:
            initialized *= post_scale
        snapshot.struct_pack_into("<f", raw, index * 4, initialized)
    return raw


def _late_fan_in(
    row: _FrozenRow,
    rows_by_name: dict[str, _FrozenRow],
    snapshot: _ContractSnapshot,
) -> int:
    if row.semantic_class == "MATRIX_WEIGHT":
        if len(row.shape) != 2:
            raise snapshot.error_type("late Linear weight must be rank two")
        return row.shape[1]
    if row.semantic_class == "BIAS":
        if not row.name.endswith(".bias"):
            raise snapshot.error_type("late Linear bias name is malformed")
        weight_name = row.name.removesuffix("bias") + "weight"
        weight = rows_by_name.get(weight_name)
        if weight is None or len(weight.shape) != 2 or weight.shape[0] != row.shape[0]:
            raise snapshot.error_type("late Linear bias has no canonical weight")
        return weight.shape[1]
    if row.semantic_class == "QUERY_OR_OTHER_WEIGHT":
        if row.shape != (512,):
            raise snapshot.error_type("late pool query shape mismatch")
        return 512
    raise snapshot.error_type("late parameter has no fan-in rule")


def _derive_parameter_raw(
    architecture: str,
    row: _FrozenRow,
    seed: int,
    rows_by_name: dict[str, _FrozenRow],
    snapshot: _ContractSnapshot,
    constant_bytes: object = _constant_bytes,
    counter_bytes: object = _counter_bytes,
    late_fan_in: object = _late_fan_in,
) -> bytearray:
    if architecture == "late":
        if row.semantic_class == "LAYERNORM_GAMMA":
            return constant_bytes(row.numel, 1.0, snapshot)
        if row.semantic_class == "LAYERNORM_BETA":
            return constant_bytes(row.numel, 0.0, snapshot)
        fan_in = late_fan_in(row, rows_by_name, snapshot)
        return counter_bytes(
            domain_prefix=b"phasepair-late-fusion-init-v1\x00",
            seed=seed,
            name=row.name,
            numel=row.numel,
            bound=1.0 / snapshot.sqrt(float(fan_in)),
            late=True,
            post_scale=None,
            snapshot=snapshot,
        )

    if row.semantic_class == "LAYERNORM_GAMMA":
        return constant_bytes(row.numel, 1.0, snapshot)
    if row.semantic_class in ("LAYERNORM_BETA", "BIAS"):
        return constant_bytes(row.numel, 0.0, snapshot)
    if row.semantic_class == "MATRIX_WEIGHT":
        if len(row.shape) != 2:
            raise snapshot.error_type("Xavier weight must be rank two")
        fan_out, fan_in = row.shape
        bound = snapshot.sqrt(6.0 / float(fan_in + fan_out))
        post_scale = None
    elif row.semantic_class in ("EMBEDDING_WEIGHT", "QUERY_OR_OTHER_WEIGHT"):
        bound = snapshot.sqrt(3.0)
        post_scale = 0.02
    else:
        raise snapshot.error_type(
            f"unsupported phasepair-init-v1 semantic class: {row.semantic_class}"
        )
    return counter_bytes(
        domain_prefix=b"phasepair-init-v1",
        seed=seed,
        name=row.name,
        numel=row.numel,
        bound=bound,
        late=False,
        post_scale=post_scale,
        snapshot=snapshot,
    )


def _canonical_parameter_bytes_impl(
    architecture: object,
    name: object,
    seed: object,
    snapshot: _ContractSnapshot,
    require_architecture: object = _require_architecture,
    require_seed: object = _require_seed,
    rows_for: object = _rows_for,
    derive_parameter_raw: object = _derive_parameter_raw,
) -> bytes:
    checked_architecture = require_architecture(architecture)
    checked_seed = require_seed(seed)
    if type(name) is not str:
        raise TypeError("name must be an exact built-in str")
    rows = rows_for(checked_architecture, snapshot)
    rows_by_name = {row.name: row for row in rows}
    row = rows_by_name.get(name)
    if row is None:
        raise snapshot.error_type(
            f"name is not canonical for {checked_architecture}: {name!r}"
        )
    return bytes(
        derive_parameter_raw(
            checked_architecture, row, checked_seed, rows_by_name, snapshot
        )
    )


def _canonical_initializer_inventory_bytes_impl(
    architecture: object,
    snapshot: _ContractSnapshot,
    require_architecture: object = _require_architecture,
    rows_for: object = _rows_for,
    serialization_rows: object = _serialization_rows,
    initializer_domain: object = _initializer_domain,
    canonical_json_bytes: object = _canonical_json_bytes,
) -> bytes:
    checked_architecture = require_architecture(architecture)
    rows = serialization_rows(
        checked_architecture, rows_for(checked_architecture, snapshot)
    )
    return canonical_json_bytes(
        {
            "architecture": checked_architecture,
            "initializer_domain": initializer_domain(checked_architecture),
            "rows": [
                {
                    "dtype": "float32",
                    "name": row.name,
                    "semantic_class": row.semantic_class,
                    "shape": list(row.shape),
                }
                for row in rows
            ],
            "schema": "phasepair-motion-initializer-inventory-v1",
        },
        snapshot,
    )


def _update_state_hash_record(
    digest: object,
    name: str,
    shape: tuple[int, ...],
    raw: bytes | bytearray,
    snapshot: _ContractSnapshot,
) -> None:
    name_raw = name.encode("ascii", "strict")
    if len(name_raw) > 65535 or len(shape) > 255:
        raise snapshot.error_type("state record name/rank is out of range")
    digest.update(len(name_raw).to_bytes(2, "big"))
    digest.update(name_raw)
    digest.update(len(shape).to_bytes(1, "big"))
    for dimension in shape:
        digest.update(dimension.to_bytes(8, "big"))
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def _state_sha256_from_raw(
    architecture: str,
    parameter_records: Iterable[tuple[str, tuple[int, ...], bytes | bytearray]],
    position_encoding_raw: bytes,
    snapshot: _ContractSnapshot,
    update_record: object = _update_state_hash_record,
) -> bytes:
    digest = snapshot.sha256_factory()
    digest.update(b"phasepair-motion-initial-state-v1\x00")
    architecture_raw = architecture.encode("ascii")
    digest.update(len(architecture_raw).to_bytes(1, "big"))
    digest.update(architecture_raw)
    records = tuple(parameter_records)
    digest.update(len(records).to_bytes(2, "big"))
    for name, shape, raw in records:
        update_record(digest, name, shape, raw, snapshot)
    update_record(
        digest,
        "position_encoding",
        (300, 512),
        position_encoding_raw,
        snapshot,
    )
    return digest.digest()


def _canonical_motion_state_sha256_impl(
    model: object,
    snapshot: _ContractSnapshot,
    validate_model: object = _validate_model_impl,
    serialization_rows: object = _serialization_rows,
    tensor_raw_bytes: object = _tensor_raw_bytes,
    state_sha256_from_raw: object = _state_sha256_from_raw,
) -> bytes:
    validated = validate_model(model, snapshot)
    parameters_by_name = dict(validated.pairs)
    records = tuple(
        (
            row.name,
            row.shape,
            tensor_raw_bytes(parameters_by_name[row.name], snapshot),
        )
        for row in serialization_rows(validated.architecture, validated.rows)
    )
    return state_sha256_from_raw(
        validated.architecture,
        records,
        tensor_raw_bytes(validated.position_encoding, snapshot),
        snapshot,
    )


def _snapshot_originals(
    validated: _ValidatedModel,
    snapshot: _ContractSnapshot,
    tensor_raw_bytes: object = _tensor_raw_bytes,
) -> tuple[_OriginalTensor, ...]:
    return tuple(
        snapshot.original_type(
            name,
            tuple(parameter.shape),
            parameter,
            bytearray(tensor_raw_bytes(parameter, snapshot)),
        )
        for name, parameter in validated.pairs
    )


def _prepare_tensors(
    validated: _ValidatedModel,
    seed: int,
    snapshot: _ContractSnapshot,
    derive_parameter_raw: object = _derive_parameter_raw,
) -> tuple[_PreparedTensor, ...]:
    rows_by_name = {row.name: row for row in validated.rows}
    prepared: list[_PreparedTensor] = []
    for pair_and_row in zip(validated.pairs, validated.rows):
        (name, parameter), row = pair_and_row
        raw = derive_parameter_raw(
            validated.architecture, row, seed, rows_by_name, snapshot
        )
        if len(raw) != snapshot.tensor_numel(parameter) * 4:
            raise AssertionError("internal initializer byte length mismatch")
        prepared.append(
            snapshot.prepared_type(
                name,
                row.shape,
                parameter,
                raw,
                snapshot.sha256_factory(raw).hexdigest(),
            )
        )
    return tuple(prepared)


def _normalized_identity_evidence(
    validated: _ValidatedModel,
    snapshot: _ContractSnapshot,
    canonical_json_bytes: object = _canonical_json_bytes,
) -> dict[str, object]:
    object_ordinals: dict[int, int] = {}
    object_rows: list[dict[str, object]] = []
    for name, parameter in validated.pairs:
        identity = id(parameter)
        if identity not in object_ordinals:
            object_ordinals[identity] = len(object_ordinals)
        object_rows.append(
            {"identity_ordinal": object_ordinals[identity], "name": name}
        )

    storage_ordinals: dict[tuple[int, int], int] = {}
    storage_rows: list[dict[str, object]] = []
    for (name, _), interval in zip(validated.pairs, validated.parameter_intervals):
        if interval not in storage_ordinals:
            storage_ordinals[interval] = len(storage_ordinals)
        storage_rows.append(
            {
                "identity_ordinal": storage_ordinals[interval],
                "name": name,
                "nbytes": interval[1] - interval[0],
            }
        )

    actor_a_indices = [
        index
        for index, (name, _) in enumerate(validated.pairs)
        if name.startswith("actor_a.")
    ]
    actor_b_indices = [
        index
        for index, (name, _) in enumerate(validated.pairs)
        if name.startswith("actor_b.")
    ]
    actor_a_objects = {id(validated.pairs[index][1]) for index in actor_a_indices}
    actor_b_objects = {id(validated.pairs[index][1]) for index in actor_b_indices}
    actor_storage_overlap_count = 0
    for left_index in actor_a_indices:
        left = validated.parameter_intervals[left_index]
        for right_index in actor_b_indices:
            right = validated.parameter_intervals[right_index]
            if left[0] < right[1] and right[0] < left[1]:
                actor_storage_overlap_count += 1

    return {
        "actor_shared_parameter_object_count": len(
            actor_a_objects & actor_b_objects
        ),
        "actor_shared_storage_range_count": actor_storage_overlap_count,
        "identity_evidence_schema": (
            "phasepair-normalized-runtime-identity-evidence-v1"
        ),
        "parameter_object_count": len(object_ordinals),
        "parameter_object_identity_sha256": snapshot.sha256_factory(
            canonical_json_bytes(
                {
                    "rows": object_rows,
                    "schema": "phasepair-normalized-object-identity-v1",
                },
                snapshot,
            )
        ).hexdigest(),
        "parameter_storage_identity_sha256": snapshot.sha256_factory(
            canonical_json_bytes(
                {
                    "rows": storage_rows,
                    "schema": "phasepair-normalized-storage-identity-v1",
                },
                snapshot,
            )
        ).hexdigest(),
        "parameter_storage_range_count": len(storage_ordinals),
    }


_LATE_GOLDENS = (
    ("actor_a.input_proj.weight", 262, "60da6b3dca74adbcee9415bdfb70153d"),
    (
        "actor_b.blocks.00.self_attn.q_proj.weight",
        512,
        "d95d1c3dbe0309bdced321bd41801cbc",
    ),
    ("actor_a.pool.query", 512, "93abd03bbf45b0b9e20f0bbd2ceed43c"),
    ("fusion.proj.weight", 1024, "eeada13aaa597fbc4abdb83bd9f3c5bc"),
)


def _late_golden_evidence(
    architecture: str,
    snapshot: _ContractSnapshot,
    goldens: tuple[tuple[str, int, str], ...] = _LATE_GOLDENS,
    counter_bytes: object = _counter_bytes,
) -> dict[str, object]:
    if architecture != "late":
        return {
            "applicable": False,
            "expected_first_four_le_hex": {},
            "observed_first_four_le_hex": {},
            "schema": "phasepair-late-seed1729-goldens-v1",
            "status": "NOT_APPLICABLE",
        }
    expected: dict[str, str] = {}
    observed: dict[str, str] = {}
    for name, fan_in, frozen_hex in goldens:
        expected[name] = frozen_hex
        observed[name] = counter_bytes(
            domain_prefix=b"phasepair-late-fusion-init-v1\x00",
            seed=1729,
            name=name,
            numel=4,
            bound=1.0 / snapshot.sqrt(float(fan_in)),
            late=True,
            post_scale=None,
            snapshot=snapshot,
        ).hex()
    if observed != expected:
        raise snapshot.error_type("late seed1729 initializer golden mismatch")
    return {
        "applicable": True,
        "expected_first_four_le_hex": expected,
        "observed_first_four_le_hex": observed,
        "schema": "phasepair-late-seed1729-goldens-v1",
        "status": "PASS",
    }


def _late_counterpart_evidence(
    architecture: str,
    prepared_by_name: dict[str, _PreparedTensor],
    rows_by_name: dict[str, _FrozenRow],
    snapshot: _ContractSnapshot,
) -> tuple[int, int]:
    if architecture != "late":
        return 0, 0
    count = 0
    distinct = 0
    for name, item in prepared_by_name.items():
        if not name.startswith("actor_a."):
            continue
        counterpart_name = "actor_b." + name.removeprefix("actor_a.")
        counterpart = prepared_by_name.get(counterpart_name)
        if counterpart is None:
            raise snapshot.error_type("late actor counterpart is absent")
        semantic = rows_by_name[name].semantic_class
        if semantic in ("LAYERNORM_GAMMA", "LAYERNORM_BETA"):
            continue
        count += 1
        if item.sha256 == counterpart.sha256:
            raise snapshot.error_type(
                f"late nonconstant actor counterpart SHA collision: {name}"
            )
        distinct += 1
    return count, distinct


def _runtime_evidence(snapshot: _ContractSnapshot) -> dict[str, object]:
    return {
        "byteorder": snapshot.byteorder,
        "float32_pack_oracle": "python-struct.pack-<f-runtime-ieee754",
        "float_mant_dig": snapshot.float_mant_dig,
        "float_radix": snapshot.float_radix,
        "float_rounds": snapshot.float_rounds,
        "libm_oracle": "python-math.sqrt-runtime-libm",
        "libm_oracle_status": "UNPINNED_RUNTIME_DEPENDENT_HOLD",
        "python_implementation": snapshot.python_implementation,
        "python_version": snapshot.python_version,
        "torch_version": snapshot.torch_version,
    }


def _build_receipt(
    validated: _ValidatedModel,
    prepared: tuple[_PreparedTensor, ...],
    seed: int,
    position_encoding_raw: bytes,
    snapshot: _ContractSnapshot,
    serialization_rows: object = _serialization_rows,
    inventory_bytes_impl: object = _canonical_initializer_inventory_bytes_impl,
    state_sha256_from_raw: object = _state_sha256_from_raw,
    runtime_evidence: object = _runtime_evidence,
    canonical_json_bytes: object = _canonical_json_bytes,
    identity_evidence: object = _normalized_identity_evidence,
    counterpart_evidence: object = _late_counterpart_evidence,
    initializer_domain: object = _initializer_domain,
    late_golden_evidence: object = _late_golden_evidence,
) -> tuple[bytes, bytes]:
    prepared_by_name = {item.name: item for item in prepared}
    rows_by_name = {row.name: row for row in validated.rows}
    serialized_prepared = tuple(
        prepared_by_name[row.name]
        for row in serialization_rows(validated.architecture, validated.rows)
    )
    inventory_sha256 = snapshot.sha256_factory(
        inventory_bytes_impl(validated.architecture, snapshot)
    ).hexdigest()
    full_state_sha256 = state_sha256_from_raw(
        validated.architecture,
        ((item.name, item.shape, item.raw) for item in serialized_prepared),
        position_encoding_raw,
        snapshot,
    )
    runtime = runtime_evidence(snapshot)
    runtime_sha256 = snapshot.sha256_factory(
        canonical_json_bytes(runtime, snapshot)
    ).hexdigest()
    identity = identity_evidence(validated, snapshot)
    counterpart_count, counterpart_distinct = counterpart_evidence(
        validated.architecture, prepared_by_name, rows_by_name, snapshot
    )
    hold_reasons = ["HOLD_RUNTIME_LIBM_ORACLE_NOT_CONTENT_PINNED"]
    if validated.architecture in ("mime", "early"):
        hold_reasons.append(
            "HOLD_PHASEPAIR_INIT_V1_HASH_SELECTED_16_INDEX_RULE_UNRESOLVED"
        )
        phasepair_v1_status = "HOLD_HASH_SELECTED_16_INDEX_RULE_UNRESOLVED"
    else:
        phasepair_v1_status = "NOT_APPLICABLE"

    payload = {
        "actor_shared_parameter_object_count": identity[
            "actor_shared_parameter_object_count"
        ],
        "actor_shared_storage_range_count": identity[
            "actor_shared_storage_range_count"
        ],
        "architecture": validated.architecture,
        "authority": 0,
        "evidence_hold_reasons": hold_reasons,
        "full_state_sha256": full_state_sha256.hex(),
        "identity_evidence_schema": identity["identity_evidence_schema"],
        "initializer_domain": initializer_domain(validated.architecture),
        "inventory_sha256": inventory_sha256,
        "late_nonconstant_counterpart_count": counterpart_count,
        "late_nonconstant_counterpart_distinct_sha256_count": (
            counterpart_distinct
        ),
        "late_seed_1729_golden_evidence": late_golden_evidence(
            validated.architecture, snapshot
        ),
        "oracle_runtime": runtime,
        "oracle_runtime_sha256": runtime_sha256,
        "parameter_numel": sum(
            snapshot.tensor_numel(item.parameter) for item in prepared
        ),
        "parameter_object_count": identity["parameter_object_count"],
        "parameter_object_identity_sha256": identity[
            "parameter_object_identity_sha256"
        ],
        "parameter_storage_identity_sha256": identity[
            "parameter_storage_identity_sha256"
        ],
        "parameter_storage_range_count": identity[
            "parameter_storage_range_count"
        ],
        "parameter_tensor_count": len(prepared),
        "parameter_tensors": [
            {
                "name": item.name,
                "sha256": item.sha256,
                "shape": list(item.shape),
            }
            for item in serialized_prepared
        ],
        "phasepair_v1_golden_status": phasepair_v1_status,
        "position_encoding_sha256": snapshot.sha256_factory(
            position_encoding_raw
        ).hexdigest(),
        "production": False,
        "schema": "phasepair-motion-initializer-receipt-v2",
        "seed": seed,
        "status": "HOLD_RUNTIME_LIBM_ORACLE_UNPINNED_AUTHORITY0",
        "training_authorized": False,
    }
    return canonical_json_bytes(payload, snapshot), full_state_sha256


def _source_tensor(
    raw: bytearray, shape: tuple[int, ...], snapshot: _ContractSnapshot
) -> Tensor:
    return snapshot.frombuffer(raw, dtype=snapshot.float32_dtype).reshape(shape)


def _transactional_commit(
    prepared: tuple[_PreparedTensor, ...],
    originals: tuple[_OriginalTensor, ...],
    position_encoding: Tensor | None,
    position_encoding_raw: bytes | None,
    primary_copy: Callable[[Tensor, Tensor], Tensor],
    rollback_copy: Callable[[Tensor, Tensor], Tensor],
    postcondition: Callable[[], None],
    snapshot: _ContractSnapshot,
    source_tensor: object = _source_tensor,
    tensor_raw_bytes: object = _tensor_raw_bytes,
) -> None:
    if len(prepared) != len(originals):
        raise snapshot.error_type("transaction prepared/original count mismatch")
    try:
        with snapshot.no_grad():
            for item in prepared:
                primary_copy(item.parameter, source_tensor(item.raw, item.shape, snapshot))
        for item in prepared:
            if tensor_raw_bytes(item.parameter, snapshot) != item.raw:
                raise snapshot.error_type(
                    f"parameter post-write byte mismatch: {item.name}"
                )
        postcondition()
    except BaseException as primary_error:
        cleanup_errors: list[BaseException] = []
        with snapshot.no_grad():
            for item in originals:
                try:
                    rollback_copy(
                        item.parameter, source_tensor(item.raw, item.shape, snapshot)
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
        for item in originals:
            try:
                if tensor_raw_bytes(item.parameter, snapshot) != item.raw:
                    cleanup_errors.append(
                        snapshot.error_type(
                            f"rollback byte mismatch: {item.name}"
                        )
                    )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if position_encoding is not None and position_encoding_raw is not None:
            try:
                if (
                    tensor_raw_bytes(position_encoding, snapshot)
                    != position_encoding_raw
                ):
                    cleanup_errors.append(
                        snapshot.error_type(
                            "position_encoding changed during failed transaction"
                        )
                    )
            except BaseException as cleanup_error:
                cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            message = (
                "initializer rollback failed; cleanup failure takes priority over "
                f"primary {type(primary_error).__name__}"
            )
            raise snapshot.error_type(message) from cleanup_errors[0]
        raise


def _initialize_motion_model_impl(
    model: object,
    seed: object,
    primary_copy: Callable[[Tensor, Tensor], Tensor],
    rollback_copy: Callable[[Tensor, Tensor], Tensor],
    snapshot: _ContractSnapshot,
    require_seed: object = _require_seed,
    validate_model: object = _validate_model_impl,
    snapshot_originals: object = _snapshot_originals,
    prepare_tensors: object = _prepare_tensors,
    tensor_raw_bytes: object = _tensor_raw_bytes,
    build_receipt: object = _build_receipt,
    state_sha_impl: object = _canonical_motion_state_sha256_impl,
    transactional_commit: object = _transactional_commit,
) -> bytes:
    checked_seed = require_seed(seed)
    validated = validate_model(model, snapshot)
    originals = snapshot_originals(validated, snapshot)
    prepared = prepare_tensors(validated, checked_seed, snapshot)
    position_encoding_raw = tensor_raw_bytes(
        validated.position_encoding, snapshot
    )
    receipt, expected_state_sha256 = build_receipt(
        validated, prepared, checked_seed, position_encoding_raw, snapshot
    )

    def postcondition() -> None:
        if (
            tensor_raw_bytes(validated.position_encoding, snapshot)
            != position_encoding_raw
        ):
            raise snapshot.error_type("frozen position_encoding changed")
        actual = state_sha_impl(model, snapshot)
        if actual != expected_state_sha256:
            raise snapshot.error_type(
                "post-write full state SHA does not match the prepared state"
            )

    transactional_commit(
        prepared,
        originals,
        validated.position_encoding,
        position_encoding_raw,
        primary_copy,
        rollback_copy,
        postcondition,
        snapshot,
    )
    return receipt


_RECEIPT_KEYS = frozenset(
    {
        "actor_shared_parameter_object_count",
        "actor_shared_storage_range_count",
        "architecture",
        "authority",
        "evidence_hold_reasons",
        "full_state_sha256",
        "identity_evidence_schema",
        "initializer_domain",
        "inventory_sha256",
        "late_nonconstant_counterpart_count",
        "late_nonconstant_counterpart_distinct_sha256_count",
        "late_seed_1729_golden_evidence",
        "oracle_runtime",
        "oracle_runtime_sha256",
        "parameter_numel",
        "parameter_object_count",
        "parameter_object_identity_sha256",
        "parameter_storage_identity_sha256",
        "parameter_storage_range_count",
        "parameter_tensor_count",
        "parameter_tensors",
        "phasepair_v1_golden_status",
        "position_encoding_sha256",
        "production",
        "schema",
        "seed",
        "status",
        "training_authorized",
    }
)


def _is_lower_hex(value: object, length: int) -> bool:
    return (
        type(value) is str
        and len(value) == length
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_json_number(_value: str) -> object:
    raise ValueError("initializer receipt must not contain JSON floats/constants")


def _parse_receipt_static_semantics(
    receipt: object,
    snapshot: _ContractSnapshot,
    receipt_keys: frozenset[str] = _RECEIPT_KEYS,
    reject_json_number: object = _reject_json_number,
    canonical_json_bytes: object = _canonical_json_bytes,
    lower_hex: object = _is_lower_hex,
    rows_for: object = _rows_for,
    ledger_for: object = _ledger_for,
    serialization_rows: object = _serialization_rows,
    inventory_bytes_impl: object = _canonical_initializer_inventory_bytes_impl,
    runtime_evidence: object = _runtime_evidence,
    late_golden_evidence: object = _late_golden_evidence,
) -> dict[str, object]:
    if type(receipt) is not bytes:
        raise TypeError("receipt must be exact built-in bytes")
    try:
        text = receipt.decode("utf-8", "strict")
        parsed = snapshot.json_loads(
            text,
            parse_float=reject_json_number,
            parse_constant=reject_json_number,
        )
    except (UnicodeDecodeError, ValueError) as exc:
        raise snapshot.error_type("receipt is not strict canonical JSON") from exc
    if type(parsed) is not dict or set(parsed) != receipt_keys:
        raise snapshot.error_type("receipt top-level keys are not closed")
    if canonical_json_bytes(parsed, snapshot) != receipt:
        raise snapshot.error_type("receipt bytes are not canonical JSON")

    integer_keys = (
        "actor_shared_parameter_object_count",
        "actor_shared_storage_range_count",
        "authority",
        "late_nonconstant_counterpart_count",
        "late_nonconstant_counterpart_distinct_sha256_count",
        "parameter_numel",
        "parameter_object_count",
        "parameter_storage_range_count",
        "parameter_tensor_count",
        "seed",
    )
    if any(type(parsed[key]) is not int for key in integer_keys):
        raise snapshot.error_type("receipt integer field has a non-exact type")
    string_keys = (
        "architecture",
        "full_state_sha256",
        "identity_evidence_schema",
        "initializer_domain",
        "inventory_sha256",
        "oracle_runtime_sha256",
        "parameter_object_identity_sha256",
        "parameter_storage_identity_sha256",
        "phasepair_v1_golden_status",
        "position_encoding_sha256",
        "schema",
        "status",
    )
    if any(type(parsed[key]) is not str for key in string_keys):
        raise snapshot.error_type("receipt string field has a non-exact type")
    if type(parsed["production"]) is not bool or type(
        parsed["training_authorized"]
    ) is not bool:
        raise snapshot.error_type("receipt boolean field has a non-exact type")
    if (
        parsed["authority"] != 0
        or parsed["production"] is not False
        or parsed["training_authorized"] is not False
        or parsed["schema"] != "phasepair-motion-initializer-receipt-v2"
        or parsed["status"]
        != "HOLD_RUNTIME_LIBM_ORACLE_UNPINNED_AUTHORITY0"
        or parsed["architecture"] not in ("mime", "early", "late")
    ):
        raise snapshot.error_type("receipt closed authority/status literals mismatch")
    expected_domain = (
        "phasepair-late-fusion-init-v1"
        if parsed["architecture"] == "late"
        else "phasepair-init-v1"
    )
    if parsed["initializer_domain"] != expected_domain:
        raise snapshot.error_type("receipt initializer domain mismatch")
    seed = parsed["seed"]
    if seed < 0 or seed > 0xFFFFFFFFFFFFFFFF:
        raise snapshot.error_type("receipt seed is outside uint64")
    for key in (
        "full_state_sha256",
        "inventory_sha256",
        "oracle_runtime_sha256",
        "parameter_object_identity_sha256",
        "parameter_storage_identity_sha256",
        "position_encoding_sha256",
    ):
        if not lower_hex(parsed[key], 64):
            raise snapshot.error_type(f"receipt {key} is not lowercase SHA-256")

    architecture = parsed["architecture"]
    rows = rows_for(architecture, snapshot)
    tensor_count, parameter_numel = ledger_for(architecture, snapshot)
    if (
        parsed["parameter_tensor_count"] != tensor_count
        or parsed["parameter_object_count"] != tensor_count
        or parsed["parameter_storage_range_count"] != tensor_count
        or parsed["parameter_numel"] != parameter_numel
        or parsed["actor_shared_parameter_object_count"] != 0
        or parsed["actor_shared_storage_range_count"] != 0
    ):
        raise snapshot.error_type("receipt parameter ledger/count semantics mismatch")
    if (
        parsed["identity_evidence_schema"]
        != "phasepair-normalized-runtime-identity-evidence-v1"
    ):
        raise snapshot.error_type("receipt identity evidence schema mismatch")

    object_rows = [
        {"identity_ordinal": index, "name": row.name}
        for index, row in enumerate(rows)
    ]
    storage_rows = [
        {
            "identity_ordinal": index,
            "name": row.name,
            "nbytes": row.numel * 4,
        }
        for index, row in enumerate(rows)
    ]
    expected_object_sha256 = snapshot.sha256_factory(
        canonical_json_bytes(
            {
                "rows": object_rows,
                "schema": "phasepair-normalized-object-identity-v1",
            },
            snapshot,
        )
    ).hexdigest()
    expected_storage_sha256 = snapshot.sha256_factory(
        canonical_json_bytes(
            {
                "rows": storage_rows,
                "schema": "phasepair-normalized-storage-identity-v1",
            },
            snapshot,
        )
    ).hexdigest()
    if (
        parsed["parameter_object_identity_sha256"] != expected_object_sha256
        or parsed["parameter_storage_identity_sha256"]
        != expected_storage_sha256
    ):
        raise snapshot.error_type("receipt normalized identity digest mismatch")

    expected_inventory_sha256 = snapshot.sha256_factory(
        inventory_bytes_impl(architecture, snapshot)
    ).hexdigest()
    if parsed["inventory_sha256"] != expected_inventory_sha256:
        raise snapshot.error_type("receipt inventory SHA mismatch")
    if parsed["position_encoding_sha256"] != snapshot.canonical_pe_sha256.hex():
        raise snapshot.error_type("receipt position_encoding SHA mismatch")

    reasons = parsed["evidence_hold_reasons"]
    expected_reasons = ["HOLD_RUNTIME_LIBM_ORACLE_NOT_CONTENT_PINNED"]
    if architecture in ("mime", "early"):
        expected_reasons.append(
            "HOLD_PHASEPAIR_INIT_V1_HASH_SELECTED_16_INDEX_RULE_UNRESOLVED"
        )
        expected_phasepair_status = (
            "HOLD_HASH_SELECTED_16_INDEX_RULE_UNRESOLVED"
        )
    else:
        expected_phasepair_status = "NOT_APPLICABLE"
    if reasons != expected_reasons or parsed["phasepair_v1_golden_status"] != (
        expected_phasepair_status
    ):
        raise snapshot.error_type("receipt HOLD/golden-status semantics mismatch")

    tensors = parsed["parameter_tensors"]
    if type(tensors) is not list:
        raise snapshot.error_type("receipt parameter_tensors must be an exact list")
    serialized_rows = serialization_rows(architecture, rows)
    if len(tensors) != len(serialized_rows):
        raise snapshot.error_type("receipt tensor row count mismatch")
    tensor_hashes: dict[str, str] = {}
    for index, item_and_row in enumerate(zip(tensors, serialized_rows)):
        item, row = item_and_row
        if type(item) is not dict or set(item) != {"name", "sha256", "shape"}:
            raise snapshot.error_type("receipt tensor row keys are not closed")
        if (
            type(item["name"]) is not str
            or item["name"] != row.name
            or not lower_hex(item["sha256"], 64)
        ):
            raise snapshot.error_type("receipt tensor row name/SHA is malformed")
        if (
            type(item["shape"]) is not list
            or any(type(dimension) is not int for dimension in item["shape"])
            or item["shape"] != list(row.shape)
        ):
            raise snapshot.error_type(
                f"receipt tensor row shape/order mismatch at index {index}"
            )
        tensor_hashes[row.name] = item["sha256"]

    if architecture == "late":
        counterpart_names = tuple(
            row.name
            for row in rows
            if row.name.startswith("actor_a.")
            and row.semantic_class not in ("LAYERNORM_GAMMA", "LAYERNORM_BETA")
        )
        for name in counterpart_names:
            counterpart = "actor_b." + name.removeprefix("actor_a.")
            if counterpart not in tensor_hashes or tensor_hashes[name] == tensor_hashes[
                counterpart
            ]:
                raise snapshot.error_type(
                    "receipt late nonconstant counterpart SHA semantics mismatch"
                )
        expected_counterpart_count = len(counterpart_names)
    else:
        expected_counterpart_count = 0
    if (
        parsed["late_nonconstant_counterpart_count"]
        != expected_counterpart_count
        or parsed["late_nonconstant_counterpart_distinct_sha256_count"]
        != expected_counterpart_count
    ):
        raise snapshot.error_type("receipt late counterpart counts mismatch")

    runtime = parsed["oracle_runtime"]
    runtime_keys = {
        "byteorder",
        "float32_pack_oracle",
        "float_mant_dig",
        "float_radix",
        "float_rounds",
        "libm_oracle",
        "libm_oracle_status",
        "python_implementation",
        "python_version",
        "torch_version",
    }
    if type(runtime) is not dict or set(runtime) != runtime_keys:
        raise snapshot.error_type("receipt runtime evidence keys are not closed")
    for key in runtime_keys - {"float_mant_dig", "float_radix", "float_rounds"}:
        if type(runtime[key]) is not str:
            raise snapshot.error_type("receipt runtime string type mismatch")
    for key in ("float_mant_dig", "float_radix", "float_rounds"):
        if type(runtime[key]) is not int:
            raise snapshot.error_type("receipt runtime integer type mismatch")
    expected_runtime = runtime_evidence(snapshot)
    if runtime != expected_runtime:
        raise snapshot.error_type("receipt runtime evidence literal mismatch")
    expected_runtime_sha256 = snapshot.sha256_factory(
        canonical_json_bytes(expected_runtime, snapshot)
    ).hexdigest()
    if parsed["oracle_runtime_sha256"] != expected_runtime_sha256:
        raise snapshot.error_type("receipt runtime evidence SHA mismatch")

    golden = parsed["late_seed_1729_golden_evidence"]
    golden_keys = {
        "applicable",
        "expected_first_four_le_hex",
        "observed_first_four_le_hex",
        "schema",
        "status",
    }
    if type(golden) is not dict or set(golden) != golden_keys:
        raise snapshot.error_type("receipt late-golden keys are not closed")
    if type(golden["applicable"]) is not bool or type(golden["schema"]) is not str:
        raise snapshot.error_type("receipt late-golden scalar type mismatch")
    if type(golden["status"]) is not str:
        raise snapshot.error_type("receipt late-golden status type mismatch")
    for map_key in ("expected_first_four_le_hex", "observed_first_four_le_hex"):
        mapping = golden[map_key]
        if type(mapping) is not dict or any(
            type(name) is not str or not lower_hex(value, 32)
            for name, value in mapping.items()
        ):
            raise snapshot.error_type("receipt late-golden map is malformed")
    expected_golden = late_golden_evidence(architecture, snapshot)
    if golden != expected_golden:
        raise snapshot.error_type("receipt late-golden evidence mismatch")
    return parsed


def _parse_receipt(
    receipt: object,
    model: object,
    seed: object,
    snapshot: _ContractSnapshot,
    parse_static: object = _parse_receipt_static_semantics,
    require_seed: object = _require_seed,
    validate_model: object = _validate_model_impl,
    prepare_tensors: object = _prepare_tensors,
    tensor_raw_bytes: object = _tensor_raw_bytes,
    build_receipt: object = _build_receipt,
    state_sha_impl: object = _canonical_motion_state_sha256_impl,
) -> dict[str, object]:
    parsed = parse_static(receipt, snapshot)
    checked_seed = require_seed(seed)
    if parsed["seed"] != checked_seed:
        raise snapshot.error_type("receipt seed does not match verifier seed")
    validated = validate_model(model, snapshot)
    prepared = prepare_tensors(validated, checked_seed, snapshot)
    for item in prepared:
        if tensor_raw_bytes(item.parameter, snapshot) != item.raw:
            raise snapshot.error_type(
                f"model parameter does not match canonical initializer: {item.name}"
            )
    position_encoding_raw = tensor_raw_bytes(
        validated.position_encoding, snapshot
    )
    expected, expected_state_sha256 = build_receipt(
        validated, prepared, checked_seed, position_encoding_raw, snapshot
    )
    if receipt != expected:
        raise snapshot.error_type(
            "receipt cryptographic fields do not match independent reconstruction"
        )
    if state_sha_impl(model, snapshot) != expected_state_sha256:
        raise snapshot.error_type("verified receipt state hash mismatch")
    return parsed


def _verify_motion_initializer_receipt_impl(
    receipt: object,
    model: object,
    seed: object,
    snapshot: _ContractSnapshot,
    parse_receipt: object = _parse_receipt,
) -> bytes:
    parse_receipt(receipt, model, seed, snapshot)
    return receipt


def _captured_contract_snapshot_bytes_impl(
    snapshot: _ContractSnapshot,
    canonical_json_bytes: object = _canonical_json_bytes,
) -> bytes:
    return canonical_json_bytes(
        {
            "architectures": ["mime", "early", "late"],
            "authority": 0,
            "dimensions": [300, 512],
            "initializer_domains": [
                "phasepair-init-v1",
                "phasepair-late-fusion-init-v1",
            ],
            "inventory_schema": "phasepair-motion-initializer-inventory-v1",
            "ledgers": [list(item) for item in snapshot.expected_ledgers],
            "model_types": [
                model_type.__module__ + "." + model_type.__qualname__
                for model_type, _, _, _ in snapshot.model_specs
            ],
            "position_encoding_sha256": snapshot.canonical_pe_sha256.hex(),
            "receipt_schema": "phasepair-motion-initializer-receipt-v2",
            "registered_parameter_mappings": [
                {
                    "architecture": architecture,
                    "rows": [
                        {
                            "canonical_name": canonical_name,
                            "registered_name": registered_name,
                        }
                        for registered_name, canonical_name in mapping
                    ],
                }
                for _, architecture, mapping, _ in snapshot.model_specs
            ],
            "registered_module_trees": [
                {
                    "architecture": architecture,
                    "rows": [
                        {
                            "buffer_names": list(spec.buffer_names),
                            "child_names": list(spec.child_names),
                            "instance_field_names": list(
                                spec.instance_field_names
                            ),
                            "literal_instance_fields": [
                                [field_name, value]
                                for field_name, value in spec.literal_instance_fields
                            ],
                            "module_type": (
                                spec.module_type.__module__
                                + "."
                                + spec.module_type.__qualname__
                            ),
                            "parameter_names": list(spec.parameter_names),
                            "path": spec.path,
                        }
                        for spec in module_specs
                    ],
                }
                for _, architecture, _, module_specs in snapshot.model_specs
            ],
            "semantic_classes": [
                "BIAS",
                "EMBEDDING_WEIGHT",
                "LAYERNORM_BETA",
                "LAYERNORM_GAMMA",
                "MATRIX_WEIGHT",
                "QUERY_OR_OTHER_WEIGHT",
            ],
            "state_domain_hex": b"phasepair-motion-initial-state-v1\x00".hex(),
            "status": "HOLD_RUNTIME_LIBM_ORACLE_UNPINNED_AUTHORITY0",
            "uint64_max": 0xFFFFFFFFFFFFFFFF,
        },
        snapshot,
    )


def _transactional_copy_for_test_impl(
    parameters: object,
    target_raws: object,
    primary_copy: object,
    rollback_copy: object,
    snapshot: _ContractSnapshot,
    tensor_raw_bytes: object = _tensor_raw_bytes,
    transactional_commit: object = _transactional_commit,
) -> None:
    if type(parameters) is not tuple or type(target_raws) is not tuple:
        raise TypeError("test transaction inputs must be exact tuples")
    if len(parameters) != len(target_raws) or not parameters:
        raise ValueError("test transaction inputs must be nonempty and aligned")
    prepared: list[_PreparedTensor] = []
    originals: list[_OriginalTensor] = []
    for index, pair in enumerate(zip(parameters, target_raws)):
        parameter, raw_value = pair
        if type(parameter) is not snapshot.parameter_type:
            raise TypeError("test transaction parameter must be exactly nn.Parameter")
        if type(raw_value) is not bytes:
            raise TypeError("test transaction target must be exact bytes")
        shape = tuple(parameter.shape)
        if len(raw_value) != snapshot.tensor_numel(parameter) * 4:
            raise ValueError("test transaction target byte length mismatch")
        originals.append(
            snapshot.original_type(
                f"test.{index}",
                shape,
                parameter,
                bytearray(tensor_raw_bytes(parameter, snapshot)),
            )
        )
        raw = bytearray(raw_value)
        prepared.append(
            snapshot.prepared_type(
                f"test.{index}",
                shape,
                parameter,
                raw,
                snapshot.sha256_factory(raw).hexdigest(),
            )
        )
    transactional_commit(
        tuple(prepared),
        tuple(originals),
        None,
        None,
        primary_copy,
        rollback_copy,
        lambda: None,
        snapshot,
    )


def _bind_public(snapshot: _ContractSnapshot) -> tuple[object, ...]:
    parameter_bytes_impl = _canonical_parameter_bytes_impl
    inventory_bytes_impl = _canonical_initializer_inventory_bytes_impl
    state_sha_impl = _canonical_motion_state_sha256_impl
    initialize_impl = _initialize_motion_model_impl
    verify_impl = _verify_motion_initializer_receipt_impl
    validate_impl = _validate_model_impl
    snapshot_impl = _captured_contract_snapshot_bytes_impl
    transaction_test_impl = _transactional_copy_for_test_impl
    captured_copy = snapshot.copy_primitive

    def parameter_bytes(architecture: str, name: str, *, seed: int) -> bytes:
        return parameter_bytes_impl(architecture, name, seed, snapshot)

    def inventory_bytes(architecture: str) -> bytes:
        return inventory_bytes_impl(architecture, snapshot)

    def state_sha(model: object) -> bytes:
        return state_sha_impl(model, snapshot)

    def initialize(model: object, *, seed: int) -> bytes:
        return initialize_impl(model, seed, captured_copy, captured_copy, snapshot)

    def verify(receipt: bytes, model: object, *, seed: int) -> bytes:
        return verify_impl(receipt, model, seed, snapshot)

    def validate(model: object) -> _ValidatedModel:
        return validate_impl(model, snapshot)

    def contract_snapshot() -> bytes:
        return snapshot_impl(snapshot)

    def transaction_test(
        parameters: tuple[nn.Parameter, ...],
        target_raws: tuple[bytes, ...],
        *,
        primary_copy: Callable[[Tensor, Tensor], Tensor],
        rollback_copy: Callable[[Tensor, Tensor], Tensor] = captured_copy,
    ) -> None:
        transaction_test_impl(
            parameters,
            target_raws,
            primary_copy,
            rollback_copy,
            snapshot,
        )

    return (
        parameter_bytes,
        inventory_bytes,
        state_sha,
        initialize,
        verify,
        validate,
        contract_snapshot,
        transaction_test,
    )


(
    canonical_parameter_bytes,
    canonical_initializer_inventory_bytes,
    canonical_motion_state_sha256,
    initialize_motion_model,
    verify_motion_initializer_receipt,
    _validate_model,
    _captured_contract_snapshot_bytes,
    _transactional_copy_for_test,
) = _bind_public(_CAPTURED)


__all__ = [
    "AUTHORITY",
    "INVENTORY_SCHEMA",
    "InitializationContractError",
    "LATE_DOMAIN",
    "MIME_EARLY_DOMAIN",
    "RECEIPT_SCHEMA",
    "STATUS",
    "canonical_initializer_inventory_bytes",
    "canonical_motion_state_sha256",
    "canonical_parameter_bytes",
    "initialize_motion_model",
    "verify_motion_initializer_receipt",
]
