"""Contract-bound PhasePair residual head and head-dropout primitives.

This module is deliberately data-free and non-production.  It implements the
shared 13 -> 256 -> 512 residual projector, the exact
``phasepair-head-dropout-v2`` counter schedule, structural token laws for the
six matched heads, and the ordered residual score used by the frozen objective.
It does not load motion, captions, checkpoints, CLIP, or private lineage data.
"""

from __future__ import annotations

import builtins
from collections import OrderedDict
import hashlib
import json
import math
import struct
import threading
from typing import Final

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from phasepair_core._identity_registry import make_identity_weak_registry
from phasepair_core.objectives import (
    PhasePairObjectiveOutput,
    phasepair_multi_positive_objective,
)


STATUS: Final = "DATA_FREE_RESIDUAL_HEAD_NONPRODUCTION_AUTHORITY0"
AUTHORITY: Final = 0
INITIALIZER_DOMAIN: Final = "phasepair-init-v1"
DROPOUT_DOMAIN: Final = "phasepair-head-dropout-v2"
INITIALIZER_RECEIPT_SCHEMA: Final = "phasepair-residual-initializer-receipt-v1"
DROPOUT_RECEIPT_SCHEMA: Final = "phasepair-head-dropout-receipt-v2"

SYSTEMS: Final = (
    "01_GENERIC",
    "02_WAMO_MARGINAL_WAVELET",
    "03_INTEREDIT_MEAN_DIFFERENCE_DCT",
    "04_NO_RELATION",
    "05_PHASE_STRIPPED",
    "06_PHASEPAIR_FULL",
)

_PARAMETER_ROWS: Final = (
    ("residual.fc1.bias", (256,), "BIAS"),
    ("residual.fc1.weight", (256, 13), "MATRIX_WEIGHT"),
    ("residual.fc2.bias", (512,), "BIAS"),
    ("residual.fc2.weight", (512, 256), "MATRIX_WEIGHT"),
)
_REGISTERED_TO_CANONICAL: Final = (
    ("fc1_weight", "residual.fc1.weight"),
    ("fc1_bias", "residual.fc1.bias"),
    ("fc2_weight", "residual.fc2.weight"),
    ("fc2_bias", "residual.fc2.bias"),
)
_EXPECTED_NUMEL: Final = 135_168
_MASK_KEEP_BITS: Final = 0x3F8E38E4
_DROP_THRESHOLD: Final = 0x1999999999999999
_U64_MASK: Final = (1 << 64) - 1


class ResidualHeadContractError(ValueError):
    """Raised when a residual-head contract is not satisfied."""


class ResidualHeadHold(RuntimeError):
    """Raised when a required canonical runtime input is not yet supplied."""

    code = "HOLD_CANONICAL_RESIDUAL_RUNTIME_INPUT"


def _make_public_api(
    _builtins_module: object = builtins,
    _contract_error_type: type[ResidualHeadContractError] = ResidualHeadContractError,
    _hold_error_type: type[ResidualHeadHold] = ResidualHeadHold,
    _ordered_dict_type: type[OrderedDict[object, object]] = OrderedDict,
) -> tuple[object, ...]:
    AssertionError = _builtins_module.AssertionError
    BaseException = _builtins_module.BaseException
    TypeError = _builtins_module.TypeError
    UnicodeDecodeError = _builtins_module.UnicodeDecodeError
    any = _builtins_module.any
    bool = _builtins_module.bool
    bytearray = _builtins_module.bytearray
    bytes = _builtins_module.bytes
    dict = _builtins_module.dict
    float = _builtins_module.float
    int = _builtins_module.int
    len = _builtins_module.len
    list = _builtins_module.list
    min = _builtins_module.min
    next = _builtins_module.next
    range = _builtins_module.range
    set = _builtins_module.set
    sorted = _builtins_module.sorted
    str = _builtins_module.str
    tuple = _builtins_module.tuple
    type = _builtins_module.type
    zip = _builtins_module.zip
    ResidualHeadContractError = _contract_error_type
    ResidualHeadHold = _hold_error_type
    tensor_type = Tensor
    parameter_type = nn.Parameter
    module_type = nn.Module
    module_init = nn.Module.__init__
    module_register_parameter = nn.Module.register_parameter
    module_setattr = nn.Module.__setattr__
    module_delattr = nn.Module.__delattr__
    parameter_ctor = nn.Parameter
    torch_empty = torch.empty
    torch_frombuffer = torch.frombuffer
    torch_float32 = torch.float32
    torch_float64 = torch.float64
    torch_bool = torch.bool
    torch_int32 = torch.int32
    torch_contiguous_format = torch.contiguous_format
    torch_no_grad = torch.no_grad
    torch_isfinite = torch.isfinite
    torch_equal = torch.equal
    torch_eye = torch.eye
    torch_exp = torch.exp
    torch_log = torch.log
    torch_atan2 = torch.atan2
    torch_maximum = torch.maximum
    torch_clamp = torch.clamp
    torch_where = torch.where
    torch_zeros_like = torch.zeros_like
    torch_abs = torch.abs
    torch_einsum = torch.einsum
    torch_full_like = torch.full_like
    torch_logical_and = torch.logical_and
    vector_norm = torch.linalg.vector_norm
    functional_linear = F.linear
    functional_gelu = F.gelu
    objective_impl = phasepair_multi_positive_objective
    objective_output_type = PhasePairObjectiveOutput
    sha256 = hashlib.sha256
    json_dumps = json.dumps
    json_loads = json.loads
    json_decode_error = json.JSONDecodeError
    int_type = int
    pack = struct.pack
    product = math.prod
    square_root = math.sqrt
    natural_log = math.log
    phase_pi = math.pi
    unpack = struct.unpack
    builtin_sum = sum
    builtin_len = len
    type_of = type
    object_getattribute = object.__getattribute__
    object_setattr = object.__setattr__
    type_getattribute = type.__getattribute__
    type_setattr = type.__setattr__
    type_delattr = type.__delattr__
    object_id = id
    copy_primitive = Tensor.copy_
    tensor_detach_primitive = Tensor.detach
    tensor_clone_primitive = Tensor.clone
    tensor_to_primitive = Tensor.to
    tensor_contiguous_primitive = Tensor.contiguous
    tensor_numpy_primitive = Tensor.numpy
    tensor_untyped_storage_primitive = Tensor.untyped_storage
    tensor_storage_offset_primitive = Tensor.storage_offset
    tensor_numel_primitive = Tensor.numel
    tensor_is_contiguous_primitive = Tensor.is_contiguous
    storage_data_ptr_primitive = torch.UntypedStorage.data_ptr
    schema_probe = module_type()
    schema_probe_dict = object_getattribute(schema_probe, "__dict__")
    expected_model_dict_keys = tuple(schema_probe_dict)
    expected_model_value_types = tuple(
        (name, type(value)) for name, value in schema_probe_dict.items()
    )
    registry_lock = threading.RLock()
    ordered_dict_type = _ordered_dict_type
    weak_set, weak_get, weak_items, _, weak_pop = make_identity_weak_registry(
        registry_lock
    )
    head_class_sealed = False
    minimum_log_energy = unpack(
        "<f", pack("<f", natural_log(1.0e-12))
    )[0]

    class ResidualHeadMeta(type(module_type)):
        def __setattr__(cls, name: str, value: object) -> None:
            if head_class_sealed and cls is ResidualHead:
                raise ResidualHeadContractError(
                    "canonical residual head class is sealed"
                )
            type_setattr(cls, name, value)

        def __delattr__(cls, name: str) -> None:
            if head_class_sealed and cls is ResidualHead:
                raise ResidualHeadContractError(
                    "canonical residual head class is sealed"
                )
            type_delattr(cls, name)

    parameter_rows = tuple(
        (name, tuple(shape), semantic)
        for name, shape, semantic in _PARAMETER_ROWS
    )
    registered_mapping = tuple(_REGISTERED_TO_CANONICAL)
    systems = tuple(SYSTEMS)
    u64_mask = (1 << 64) - 1
    drop_threshold = 0x1999999999999999

    def require_int(value: object, label: str, upper: int) -> int:
        if type(value) is not int or value < 0 or value >= upper:
            raise ResidualHeadContractError(f"{label} is outside its exact range")
        return value

    def canonical_json(value: object) -> bytes:
        return (
            json_dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    def tensor_raw(value: Tensor) -> bytes:
        detached = tensor_detach_primitive(value)
        on_cpu = tensor_to_primitive(
            detached,
            device="cpu",
            dtype=torch_float32,
        )
        checked = tensor_contiguous_primitive(on_cpu)
        return tensor_numpy_primitive(checked).tobytes(order="C")

    def parameter_raw(name: str, seed: int) -> bytes:
        if type(name) is not str:
            raise ResidualHeadContractError("parameter name must be exact str")
        checked_seed = require_int(seed, "seed", 1 << 64)
        row = next((item for item in parameter_rows if item[0] == name), None)
        if row is None:
            raise ResidualHeadContractError("unknown canonical residual parameter")
        _, shape, semantic = row
        count = product(shape)
        if semantic == "BIAS":
            return b"\x00\x00\x00\x00" * count
        fan_out, fan_in = shape
        scale = square_root(6.0 / float(fan_in + fan_out))
        encoded_name = name.encode("utf-8")
        prefix = (
            b"phasepair-init-v1"
            + pack(">Q", checked_seed)
            + pack(">H", len(encoded_name))
            + encoded_name
        )
        output = bytearray(4 * count)
        offset = 0
        denominator = float(1 << 65)
        for flat_index in range(count):
            digest = sha256(prefix + pack(">Q", flat_index)).digest()
            integer = int.from_bytes(digest[:8], "big", signed=False)
            uniform = (2 * integer + 1) / denominator
            value = (2.0 * uniform - 1.0) * scale
            output[offset : offset + 4] = pack("<f", value)
            offset += 4
        return bytes(output)

    def inventory_bytes() -> bytes:
        return canonical_json(
            {
                "authority": 0,
                "initializer_domain": "phasepair-init-v1",
                "parameter_count": 4,
                "parameter_numel": 135168,
                "rows": [
                    {
                        "name": name,
                        "semantic_class": semantic,
                        "shape": list(shape),
                    }
                    for name, shape, semantic in parameter_rows
                ],
                "status": "DATA_FREE_NONPRODUCTION_AUTHORITY0",
            }
        )

    def exact_model_dict(model: object) -> dict[str, object]:
        if type(model) is not ResidualHead:
            raise ResidualHeadContractError("model must be an exact ResidualHead")
        class_dict = type_getattribute(ResidualHead, "__dict__")
        if (
            tuple(class_dict) != expected_head_class_keys
            or class_dict.get("__init__") is not canonical_head_init
            or class_dict.get("__call__") is not canonical_head_call
            or class_dict.get("forward") is not canonical_head_forward
            or class_dict.get("__setattr__") is not canonical_head_setattr
            or class_dict.get("__delattr__") is not canonical_head_delattr
        ):
            raise ResidualHeadContractError("residual head class schema mismatch")
        model_dict = object_getattribute(model, "__dict__")
        if (
            type(model_dict) is not dict
            or tuple(model_dict) != expected_model_dict_keys
        ):
            raise ResidualHeadContractError("residual model object schema mismatch")
        for name, expected_type in expected_model_value_types:
            value = model_dict[name]
            if type(value) is not expected_type:
                raise ResidualHeadContractError(
                    f"residual model field type mismatch: {name}"
                )
            if name == "training":
                continue
            if name == "_parameters":
                continue
            if name == "_is_full_backward_hook":
                if value is not None:
                    raise ResidualHeadContractError(
                        "residual model backward-hook state is forbidden"
                    )
                continue
            if len(value) != 0:
                raise ResidualHeadContractError(
                    f"residual model registry must be empty: {name}"
                )
        return model_dict

    def direct_parameters(model: object) -> tuple[tuple[str, nn.Parameter], ...]:
        model_dict = exact_model_dict(model)
        registry = model_dict.get("_parameters")
        if type(registry) is not dict or tuple(registry) != tuple(
            item[0] for item in registered_mapping
        ):
            raise ResidualHeadContractError("residual parameter registry mismatch")
        rows_by_name = {name: (shape, semantic) for name, shape, semantic in parameter_rows}
        output: list[tuple[str, nn.Parameter]] = []
        object_ids: set[int] = set()
        intervals: list[tuple[int, int]] = []
        total = 0
        for registered, canonical in registered_mapping:
            value = registry.get(registered)
            if type(value) is not parameter_type:
                raise ResidualHeadContractError("residual parameter type mismatch")
            shape, _ = rows_by_name[canonical]
            if (
                tuple(value.shape) != shape
                or value.dtype is not torch_float32
                or not tensor_is_contiguous_primitive(value)
                or not value.requires_grad
                or not value.is_leaf
            ):
                raise ResidualHeadContractError(
                    f"residual parameter contract mismatch: {canonical}"
                )
            backward_hooks = object_getattribute(value, "_backward_hooks")
            post_accumulate_hooks = object_getattribute(
                value, "_post_accumulate_grad_hooks"
            )
            if backward_hooks is not None or (
                post_accumulate_hooks is not None
                and (
                    type_of(post_accumulate_hooks) is not ordered_dict_type
                    or builtin_len(post_accumulate_hooks) != 0
                )
            ):
                raise ResidualHeadContractError(
                    f"residual parameter hooks are forbidden: {canonical}"
                )
            if object_id(value) in object_ids:
                raise ResidualHeadContractError("residual parameters alias by object")
            object_ids.add(object_id(value))
            if value.device.type != "meta":
                storage = tensor_untyped_storage_primitive(value)
                start = int(storage_data_ptr_primitive(storage)) + 4 * int(
                    tensor_storage_offset_primitive(value)
                )
                stop = start + 4 * int(tensor_numel_primitive(value))
                if any(start < other_stop and other_start < stop for other_start, other_stop in intervals):
                    raise ResidualHeadContractError("residual parameter storages overlap")
                intervals.append((start, stop))
            total += int(tensor_numel_primitive(value))
            output.append((canonical, value))
        if total != 135168:
            raise ResidualHeadContractError("residual parameter ledger mismatch")
        return tuple(output)

    def parameter_binding(
        rows: tuple[tuple[str, nn.Parameter], ...],
    ) -> tuple[
        tuple[int, ...],
        tuple[tuple[str, int | None, int, int, int], ...],
    ]:
        object_ids = tuple(object_id(parameter) for _, parameter in rows)
        intervals: list[tuple[str, int | None, int, int, int]] = []
        for _, parameter in rows:
            if parameter.device.type == "meta":
                raise ResidualHeadHold("meta residual head has no live storage binding")
            storage = tensor_untyped_storage_primitive(parameter)
            start = int(storage_data_ptr_primitive(storage)) + 4 * int(
                tensor_storage_offset_primitive(parameter)
            )
            intervals.append(
                (
                    parameter.device.type,
                    parameter.device.index,
                    int(storage._cdata),
                    start,
                    start + 4 * int(tensor_numel_primitive(parameter)),
                )
            )
        return object_ids, tuple(intervals)

    def require_unowned_parameters(
        model: object,
        rows: tuple[tuple[str, nn.Parameter], ...],
    ) -> tuple[
        tuple[int, ...],
        tuple[tuple[str, int | None, int, int, int], ...],
    ]:
        candidate_ids, candidate_intervals = parameter_binding(rows)
        for owner, record in weak_items():
            if owner is model:
                continue
            current_rows = direct_parameters(owner)
            current_ids, current_intervals = parameter_binding(current_rows)
            owned_ids = record[2] + current_ids
            owned_intervals = record[3] + current_intervals
            if any(candidate in owned_ids for candidate in candidate_ids):
                raise ResidualHeadContractError(
                    "residual parameters are already owned by another live head"
                )
            for (
                device_type,
                device_index,
                _,
                start,
                stop,
            ) in candidate_intervals:
                for (
                    owned_device_type,
                    owned_device_index,
                    _,
                    owned_start,
                    owned_stop,
                ) in owned_intervals:
                    if (
                        device_type == owned_device_type
                        and device_index == owned_device_index
                        and start < owned_stop
                        and owned_start < stop
                    ):
                        raise ResidualHeadContractError(
                            "residual storage is already owned by another live head"
                        )
        return candidate_ids, candidate_intervals

    def live_parameters(
        model: object,
    ) -> tuple[tuple[tuple[str, nn.Parameter], ...], bool]:
        rows = direct_parameters(model)
        record = weak_get(model)
        if record is None:
            raise ResidualHeadHold("residual head lacks canonical initializer receipt")
        object_ids, intervals = parameter_binding(rows)
        hook_registries = tuple(
            object_getattribute(parameter, "_post_accumulate_grad_hooks")
            for _, parameter in rows
        )
        if (
            object_ids != record[2]
            or intervals != record[3]
            or builtin_len(record) != 5
            or builtin_len(record[4]) != builtin_len(hook_registries)
            or any(
                current is not expected
                for current, expected in zip(
                    hook_registries, record[4], strict=True
                )
            )
        ):
            raise ResidualHeadContractError(
                "live residual parameters or hook registries no longer match initialized ownership"
            )
        training = object_getattribute(model, "__dict__")["training"]
        return rows, training

    def state_sha(model: object) -> bytes:
        rows = direct_parameters(model)
        digest = sha256()
        digest.update(b"phasepair-residual-state-v1\x00")
        for name, parameter in sorted(rows, key=lambda item: item[0].encode("utf-8")):
            encoded = name.encode("utf-8")
            raw = tensor_raw(parameter)
            digest.update(pack(">H", len(encoded)))
            digest.update(encoded)
            digest.update(pack(">Q", len(raw)))
            digest.update(raw)
        return digest.digest()

    def build_initializer_receipt(model: object, seed: int) -> bytes:
        checked_seed = require_int(seed, "seed", 1 << 64)
        rows = direct_parameters(model)
        expected = {name: parameter_raw(name, checked_seed) for name, _ in rows}
        for name, parameter in rows:
            if tensor_raw(parameter) != expected[name]:
                raise ResidualHeadContractError(
                    f"residual parameter is not canonically initialized: {name}"
                )
        inventory = inventory_bytes()
        return canonical_json(
            {
                "authority": 0,
                "initializer_domain": "phasepair-init-v1",
                "inventory_sha256": sha256(inventory).hexdigest(),
                "parameter_count": 4,
                "parameter_numel": 135168,
                "parameter_sha256": {
                    name: sha256(expected[name]).hexdigest()
                    for name, _ in sorted(rows, key=lambda item: item[0].encode("utf-8"))
                },
                "schema": "phasepair-residual-initializer-receipt-v1",
                "seed": checked_seed,
                "state_sha256": state_sha(model).hex(),
                "status": "DATA_FREE_NONPRODUCTION_AUTHORITY0",
            }
        )

    def initialize_impl(model: object, seed: int) -> bytes:
        checked_seed = require_int(seed, "seed", 1 << 64)
        with registry_lock:
            rows = direct_parameters(model)
            if any(parameter.device.type == "meta" for _, parameter in rows):
                raise ResidualHeadHold("meta residual head cannot be initialized")
            if weak_get(model) is not None:
                raise ResidualHeadContractError("residual head is already initialized")
            parameter_ids, storage_intervals = require_unowned_parameters(model, rows)
            targets = tuple((name, parameter_raw(name, checked_seed)) for name, _ in rows)
            originals = tuple((name, tensor_raw(parameter)) for name, parameter in rows)
            original_hook_registries = tuple(
                object_getattribute(parameter, "_post_accumulate_grad_hooks")
                for _, parameter in rows
            )
            canonical_hook_registries = tuple(
                ordered_dict_type() for _ in rows
            )
            target_tensors: list[Tensor] = []
            for (name, parameter), (target_name, raw) in zip(rows, targets, strict=True):
                if name != target_name:
                    raise AssertionError("residual target traversal mismatch")
                target = torch_frombuffer(bytearray(raw), dtype=torch_float32).reshape(
                    tuple(parameter.shape)
                )
                target_tensors.append(target.to(device=parameter.device))

            committed = False
            registry_inserted = False
            try:
                for (_, parameter), hook_registry in zip(
                    rows, canonical_hook_registries, strict=True
                ):
                    object_setattr(
                        parameter, "_post_accumulate_grad_hooks", hook_registry
                    )
                with torch_no_grad():
                    for (_, parameter), target in zip(rows, target_tensors, strict=True):
                        copy_primitive(parameter, target)
                for (name, parameter), (_, raw) in zip(rows, targets, strict=True):
                    if tensor_raw(parameter) != raw:
                        raise ResidualHeadContractError(
                            f"residual initialization postcondition failed: {name}"
                        )

                receipt = build_initializer_receipt(model, checked_seed)
                binding = (
                    checked_seed,
                    sha256(receipt).digest(),
                    parameter_ids,
                    storage_intervals,
                    canonical_hook_registries,
                )
                if weak_get(model) is not None:
                    raise ResidualHeadContractError(
                        "residual initializer registry collision"
                    )
                weak_set(model, binding)
                registry_inserted = True
                if (
                    weak_get(model) is not binding
                    or parameter_binding(direct_parameters(model))
                    != (parameter_ids, storage_intervals)
                    or any(
                        object_getattribute(
                            parameter, "_post_accumulate_grad_hooks"
                        )
                        is not expected
                        for (_, parameter), expected in zip(
                            rows, canonical_hook_registries, strict=True
                        )
                    )
                ):
                    raise ResidualHeadContractError(
                        "residual initializer registry postcondition failed"
                    )
                committed = True
                return receipt
            finally:
                if not committed:
                    cleanup_failure: BaseException | None = None
                    try:
                        if registry_inserted:
                            weak_pop(model)
                        with torch_no_grad():
                            for (_, parameter), (_, raw) in zip(
                                rows, originals, strict=True
                            ):
                                source = torch_frombuffer(
                                    bytearray(raw), dtype=torch_float32
                                ).reshape(tuple(parameter.shape))
                                copy_primitive(
                                    parameter, source.to(device=parameter.device)
                                )
                        for (_, parameter), original_hook_registry in zip(
                            rows, original_hook_registries, strict=True
                        ):
                            object_setattr(
                                parameter,
                                "_post_accumulate_grad_hooks",
                                original_hook_registry,
                            )
                        if any(
                            tensor_raw(parameter) != raw
                            for (_, parameter), (_, raw) in zip(
                                rows, originals, strict=True
                            )
                        ):
                            raise ResidualHeadContractError(
                                "residual initialization rollback verification failed"
                            )
                        if weak_get(model) is not None:
                            raise ResidualHeadContractError(
                                "residual initializer registry rollback failed"
                            )
                        if any(
                            object_getattribute(
                                parameter, "_post_accumulate_grad_hooks"
                            )
                            is not original
                            for (_, parameter), original in zip(
                                rows, original_hook_registries, strict=True
                            )
                        ):
                            raise ResidualHeadContractError(
                                "residual hook-registry rollback failed"
                            )
                    except BaseException as exc:
                        cleanup_failure = exc
                    if cleanup_failure is not None:
                        raise ResidualHeadContractError(
                            "residual initialization cleanup failed"
                        ) from cleanup_failure

    def verify_initializer_impl(receipt: object, model: object, seed: int) -> bytes:
        if type(receipt) is not bytes:
            raise ResidualHeadContractError("initializer receipt must be exact bytes")
        try:
            parsed = json_loads(
                receipt.decode("utf-8"),
                parse_int=int_type,
                parse_float=lambda value: (_ for _ in ()).throw(
                    ResidualHeadContractError("JSON numbers are forbidden")
                ),
            )
        except (UnicodeDecodeError, json_decode_error) as exc:
            raise ResidualHeadContractError("initializer receipt is invalid JSON") from exc
        if canonical_json(parsed) != receipt:
            raise ResidualHeadContractError("initializer receipt is not canonical JSON")
        checked_seed = require_int(seed, "seed", 1 << 64)
        with registry_lock:
            live_parameters(model)
            expected = build_initializer_receipt(model, checked_seed)
            digest = sha256(receipt).digest()
            record = weak_get(model)
            if (
                receipt != expected
                or record is None
                or record[0] != checked_seed
                or record[1] != digest
            ):
                raise ResidualHeadContractError(
                    "initializer receipt does not bind live model"
                )
            return digest

    def dropout_material(seed: int, step: int, batch_size: int) -> tuple[bytes, bytes, bytes]:
        checked_seed = require_int(seed, "seed", 1 << 64)
        checked_step = require_int(step, "global_optimizer_step", 1 << 32)
        if type(batch_size) is not int or batch_size < 1 or batch_size > 128:
            raise ResidualHeadContractError("batch_size is outside [1,128]")
        shape = (2, batch_size, 6, 256)
        preimage = (
            b"phasepair-head-dropout-v2\x00"
            + pack(">Q", checked_seed)
            + pack(">I", checked_step)
            + pack(">H", 0)
            + pack(">H", 0)
            + pack(">B", 4)
            + b"".join(pack(">I", dim) for dim in shape)
        )
        h0 = sha256(preimage).digest()
        base = int.from_bytes(h0[:8], "big", signed=False)
        count = product(shape)
        packed_mask = bytearray((count + 7) // 8)
        expanded = bytearray(4 * count)
        for flat_index in range(count):
            z = (base + 0x9E3779B97F4A7C15 * (flat_index + 1)) & u64_mask
            z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & u64_mask
            z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & u64_mask
            z ^= z >> 31
            keep = z >= drop_threshold
            if keep:
                packed_mask[flat_index // 8] |= 1 << (flat_index % 8)
                expanded[4 * flat_index : 4 * flat_index + 4] = b"\xe4\x38\x8e\x3f"
        return h0, bytes(packed_mask), bytes(expanded)

    def dropout_receipt_impl(seed: int, step: int, batch_size: int) -> bytes:
        h0, packed_mask, expanded = dropout_material(seed, step, batch_size)
        count = 2 * batch_size * 6 * 256
        first16 = "".join(
            "1" if (packed_mask[index // 8] >> (index % 8)) & 1 else "0"
            for index in range(min(16, count))
        )
        keep_count = builtin_sum(byte.bit_count() for byte in packed_mask)
        return canonical_json(
            {
                "authority": 0,
                "batch_size": batch_size,
                "drop_count": count - keep_count,
                "expanded_sha256": sha256(expanded).hexdigest(),
                "first16_keep": first16,
                "global_optimizer_step": step,
                "h0_sha256": h0.hex(),
                "packed_sha256": sha256(packed_mask).hexdigest(),
                "schema": "phasepair-head-dropout-receipt-v2",
                "seed": seed,
                "shape": [2, batch_size, 6, 256],
                "status": "DATA_FREE_NONPRODUCTION_AUTHORITY0",
            }
        )

    def dropout_mask_impl(
        seed: int,
        step: int,
        batch_size: int,
        device: torch.device,
    ) -> Tensor:
        _, _, expanded = dropout_material(seed, step, batch_size)
        mask = torch_frombuffer(bytearray(expanded), dtype=torch_float32).reshape(
            2, batch_size, 6, 256
        )
        mask = mask.to(device=device).contiguous()
        raw = tensor_raw(mask)
        if raw != expanded:
            raise ResidualHeadContractError("device materialization changed dropout bytes")
        return mask

    def snapshot_tokens(
        tokens: object,
        active_mask: object,
        system: object,
    ) -> tuple[Tensor, Tensor, str]:
        if type(system) is not str or system not in systems:
            raise ResidualHeadContractError("system must be one exact residual system ID")
        if type(tokens) is not tensor_type or type(active_mask) is not tensor_type:
            raise TypeError("tokens and active_mask must be exact torch.Tensor objects")
        if (
            tokens.dtype is not torch_float32
            or tokens.ndim != 4
            or tokens.shape[0] != 2
            or tokens.shape[2:] != (6, 13)
            or tokens.shape[1] < 1
            or tokens.shape[1] > 128
            or not tensor_is_contiguous_primitive(tokens)
            or tokens.requires_grad
        ):
            raise ResidualHeadContractError("tokens must be float32 C [2,B,6,13]")
        if (
            active_mask.dtype is not torch_bool
            or tuple(active_mask.shape) != tuple(tokens.shape[:3])
            or active_mask.device != tokens.device
            or not tensor_is_contiguous_primitive(active_mask)
            or active_mask.requires_grad
        ):
            raise ResidualHeadContractError("active_mask must be bool C [2,B,6]")
        checked_tokens = tensor_clone_primitive(
            tensor_detach_primitive(tokens),
            memory_format=torch_contiguous_format,
        )
        checked_mask = tensor_clone_primitive(
            tensor_detach_primitive(active_mask),
            memory_format=torch_contiguous_format,
        )
        if not bool(torch_isfinite(checked_tokens).all().item()):
            raise ResidualHeadContractError("tokens must be finite")
        one_hot = torch_eye(6, dtype=torch_float32, device=tokens.device).reshape(1, 1, 6, 6)
        expected_one_hot = one_hot.expand(2, tokens.shape[1], 6, 6).contiguous()
        if not torch_equal(
            checked_tokens[..., 7:].contiguous().view(torch_int32),
            expected_one_hot.view(torch_int32),
        ):
            raise ResidualHeadContractError("token slot one-hot suffix mismatch")
        if not torch_equal(checked_mask[0], checked_mask[1]):
            raise ResidualHeadContractError("AB/BA active masks must be bitwise equal")

        first = checked_tokens[..., :7]

        def require_positive_zero(value: Tensor, label: str) -> None:
            bits = value.contiguous().view(torch_int32)
            if not bool((bits == 0).all().item()):
                raise ResidualHeadContractError(f"{label} must be exact +0")

        def bit_equal(left: Tensor, right: Tensor) -> bool:
            return torch_equal(
                left.contiguous().view(torch_int32),
                right.contiguous().view(torch_int32),
            )

        def derived_close(actual: Tensor, expected: Tensor) -> bool:
            actual64 = actual.to(dtype=torch_float64)
            expected64 = expected.to(dtype=torch_float64)
            if not bool(
                torch_logical_and(
                    torch_isfinite(actual64), torch_isfinite(expected64)
                )
                .all()
                .item()
            ):
                return False
            scale = torch_clamp(
                torch_abs(actual64) + torch_abs(expected64), min=1.0
            )
            return bool(
                (
                    torch_abs(actual64 - expected64)
                    <= (8.0 * (2.0**-23)) * scale
                )
                .all()
                .item()
            )

        def combined_log_energy(
            left: Tensor,
            right: Tensor,
            label: str,
        ) -> Tensor:
            if not bool(
                torch_logical_and(
                    left >= minimum_log_energy,
                    right >= minimum_log_energy,
                )
                .all()
                .item()
            ):
                raise ResidualHeadContractError(
                    f"{label} marginal log energy is below log(epsilon)"
                )
            left64 = left.to(dtype=torch_float64)
            right64 = right.to(dtype=torch_float64)
            maximum = torch_maximum(left64, right64)
            scaled_sum = (
                torch_exp(left64 - maximum)
                + torch_exp(right64 - maximum)
                - 1.0e-12 * torch_exp(-maximum)
            )
            if not bool(
                torch_logical_and(
                    torch_isfinite(scaled_sum), scaled_sum > 0.0
                )
                .all()
                .item()
            ):
                raise ResidualHeadContractError(
                    f"{label} combined energy is outside its numeric domain"
                )
            combined = maximum + torch_log(scaled_sum)
            if not bool(torch_isfinite(combined).all().item()):
                raise ResidualHeadContractError(
                    f"{label} combined log energy is nonfinite"
                )
            return combined.to(dtype=torch_float32)

        def require_nonnegative_positive_zero(value: Tensor, label: str) -> None:
            if not bool((value >= 0.0).all().item()):
                raise ResidualHeadContractError(f"{label} must be nonnegative")
            zero = value == 0.0
            bits = value.contiguous().view(torch_int32)
            if not bool((bits[zero] == 0).all().item()):
                raise ResidualHeadContractError(
                    f"{label} zero values must be exact +0"
                )

        require_positive_zero(first[~checked_mask], "masked token fields")
        ab = first[0]
        ba = first[1]
        if system != "01_GENERIC" and not bool(
            (first[..., :2][checked_mask] >= minimum_log_energy).all().item()
        ):
            raise ResidualHeadContractError(
                "active marginal log energy is below log(epsilon)"
            )
        if system == "01_GENERIC":
            require_positive_zero(first, "generic token fields")
            if not bit_equal(ab, ba):
                raise ResidualHeadContractError("generic AB/BA law mismatch")
        elif system == "02_WAMO_MARGINAL_WAVELET":
            require_positive_zero(first[..., 5:7], "WaMo forbidden fields")
            if not (
                bit_equal(ab[..., 0], ba[..., 1])
                and bit_equal(ab[..., 1], ba[..., 0])
                and bit_equal(ab[..., 2:5], ba[..., 2:5])
            ):
                raise ResidualHeadContractError("WaMo AB/BA law mismatch")
            require_nonnegative_positive_zero(
                first[..., 3], "WaMo absolute marginal difference"
            )
            wamo_active = checked_mask
            if not derived_close(
                first[..., 3][wamo_active],
                torch_abs(
                    first[..., 0][wamo_active].to(dtype=torch_float64)
                    - first[..., 1][wamo_active].to(dtype=torch_float64)
                ),
            ):
                raise ResidualHeadContractError(
                    "WaMo absolute marginal difference law mismatch"
                )
            expected_mean = 0.5 * (
                first[..., 0][wamo_active].to(dtype=torch_float64)
                + first[..., 1][wamo_active].to(dtype=torch_float64)
            )
            if not derived_close(first[..., 4][wamo_active], expected_mean):
                raise ResidualHeadContractError("WaMo marginal mean law mismatch")
            require_positive_zero(
                first[..., 4][wamo_active][expected_mean == 0.0],
                "WaMo zero marginal mean",
            )
            expected_combined = combined_log_energy(
                first[..., 0][wamo_active],
                first[..., 1][wamo_active],
                "WaMo",
            )
            if not derived_close(first[..., 2][wamo_active], expected_combined):
                raise ResidualHeadContractError(
                    "WaMo combined marginal energy law mismatch"
                )
        elif system == "03_INTEREDIT_MEAN_DIFFERENCE_DCT":
            require_positive_zero(first[..., 5:7], "InterEdit forbidden fields")
            if not bit_equal(ab, ba):
                raise ResidualHeadContractError("InterEdit AB/BA law mismatch")
            if not (
                torch_equal(checked_mask[..., 0], checked_mask[..., 1])
                and torch_equal(checked_mask[..., 2], checked_mask[..., 3])
                and torch_equal(checked_mask[..., 4], checked_mask[..., 5])
            ):
                raise ResidualHeadContractError(
                    "InterEdit S/D availability masks must be paired"
                )
            for sum_slot, difference_slot in ((0, 1), (2, 3), (4, 5)):
                sum_token = ab[:, sum_slot]
                difference_token = ab[:, difference_slot]
                if not (
                    bit_equal(sum_token[..., 0], sum_token[..., 1])
                    and bit_equal(difference_token[..., 0], difference_token[..., 2])
                    and bit_equal(sum_token[..., 1:4], difference_token[..., 1:4])
                ):
                    raise ResidualHeadContractError(
                        "InterEdit S/D token field law mismatch"
                    )
                pair_active = checked_mask[0, :, sum_slot]
                expected_total = combined_log_energy(
                    sum_token[..., 1][pair_active],
                    sum_token[..., 2][pair_active],
                    "InterEdit",
                )
                if not derived_close(
                    sum_token[..., 3][pair_active], expected_total
                ):
                    raise ResidualHeadContractError(
                        "InterEdit total-energy law mismatch"
                    )
                if not (
                    derived_close(
                        sum_token[..., 4][pair_active],
                        sum_token[..., 0][pair_active].to(dtype=torch_float64)
                        - sum_token[..., 3][pair_active].to(dtype=torch_float64),
                    )
                    and derived_close(
                        difference_token[..., 4][pair_active],
                        difference_token[..., 0][pair_active].to(
                            dtype=torch_float64
                        )
                        - difference_token[..., 3][pair_active].to(
                            dtype=torch_float64
                        ),
                    )
                ):
                    raise ResidualHeadContractError(
                        "InterEdit selected-to-total law mismatch"
                    )
        elif system == "04_NO_RELATION":
            require_positive_zero(first[..., 2:7], "no-relation forbidden fields")
            if not (
                bit_equal(ab[..., 0], ba[..., 1])
                and bit_equal(ab[..., 1], ba[..., 0])
            ):
                raise ResidualHeadContractError("no-relation AB/BA law mismatch")
        elif system == "05_PHASE_STRIPPED":
            require_positive_zero(first[..., 3:7], "phase-stripped forbidden fields")
            require_nonnegative_positive_zero(
                first[..., 2], "phase-stripped coherence"
            )
            if not bool((first[..., 2] <= 1.0).all().item()):
                raise ResidualHeadContractError(
                    "phase-stripped coherence must not exceed one"
                )
            if not (
                bit_equal(ab[..., 0], ba[..., 1])
                and bit_equal(ab[..., 1], ba[..., 0])
                and bit_equal(ab[..., 2], ba[..., 2])
            ):
                raise ResidualHeadContractError("phase-stripped AB/BA law mismatch")
        else:
            active = checked_mask[0]
            if not (
                bit_equal(ab[..., 0][active], ba[..., 1][active])
                and bit_equal(ab[..., 1][active], ba[..., 0][active])
                and bit_equal(ab[..., 2][active], ba[..., 2][active])
                and bit_equal(ab[..., 3][active], ba[..., 3][active])
                and bit_equal(ab[..., 6][active], ba[..., 6][active])
            ):
                raise ResidualHeadContractError("PhasePair-full AB/BA law mismatch")
            signed_active = torch_logical_and(active, ab[..., 6] == 1.0)
            if not (
                bit_equal(
                    ab[..., 4][signed_active], -ba[..., 4][signed_active]
                )
                and bit_equal(
                    ab[..., 5][signed_active], -ba[..., 5][signed_active]
                )
            ):
                raise ResidualHeadContractError(
                    "PhasePair-full signed AB/BA law mismatch"
                )
            require_nonnegative_positive_zero(first[..., 2], "PhasePair coherence")
            if not bool((first[..., 2] <= 1.0).all().item()):
                raise ResidualHeadContractError(
                    "PhasePair coherence must not exceed one"
                )
            if not bool(
                torch_logical_and(
                    first[..., 3] >= -1.0, first[..., 3] <= 1.0
                )
                .all()
                .item()
            ) or not bool(
                torch_logical_and(
                    first[..., 4] >= -1.0, first[..., 4] <= 1.0
                )
                .all()
                .item()
            ):
                raise ResidualHeadContractError(
                    "PhasePair cosine/sine fields must stay in [-1,1]"
                )
            if not bool(
                torch_logical_and(
                    first[..., 5] > -1.0, first[..., 5] <= 1.0
                )
                .all()
                .item()
            ):
                raise ResidualHeadContractError(
                    "PhasePair normalized delay must stay in (-1,1]"
                )
            signed_valid = first[..., 6]
            if not bool(
                torch_logical_and(signed_valid >= 0.0, signed_valid <= 1.0)
                .all()
                .item()
            ) or not bool(
                ((signed_valid == 0.0) | (signed_valid == 1.0)).all().item()
            ):
                raise ResidualHeadContractError(
                    "PhasePair signed-phase validity must be binary"
                )
            require_nonnegative_positive_zero(
                signed_valid, "PhasePair signed-phase validity"
            )
            active_cosine = ab[..., 3][active].to(dtype=torch_float64)
            active_sine = ab[..., 4][active].to(dtype=torch_float64)
            unit_norm = active_cosine * active_cosine + active_sine * active_sine
            if not derived_close(
                unit_norm,
                torch_full_like(unit_norm, 1.0),
            ):
                raise ResidualHeadContractError(
                    "PhasePair active cosine/sine must lie on the unit circle"
                )
            signed_phase = torch_atan2(
                ab[..., 4][signed_active].to(dtype=torch_float64),
                ab[..., 3][signed_active].to(dtype=torch_float64),
            )
            if not bool(
                (phase_pi - torch_abs(signed_phase) > 1.0e-6)
                .all()
                .item()
            ):
                raise ResidualHeadContractError(
                    "PhasePair signed phase is inside the anti-phase branch cut"
                )
            if not derived_close(
                ab[..., 5][signed_active],
                -signed_phase / phase_pi,
            ):
                raise ResidualHeadContractError(
                    "PhasePair normalized delay is inconsistent with phase"
                )
            unsigned = signed_valid == 0.0
            require_positive_zero(
                first[..., 4][unsigned],
                "PhasePair unsigned sine fields",
            )
            require_positive_zero(
                first[..., 5][unsigned],
                "PhasePair unsigned delay fields",
            )
            unsigned_active = torch_logical_and(active, ab[..., 6] == 0.0)
            if not derived_close(
                ab[..., 3][unsigned_active],
                torch_full_like(ab[..., 3][unsigned_active], -1.0),
            ):
                raise ResidualHeadContractError(
                    "PhasePair unsigned active phase must be anti-phase"
                )
        return checked_tokens, checked_mask, system

    class ResidualHead(module_type, metaclass=ResidualHeadMeta):
        """The single shared 135,168-parameter residual projector."""

        def __setattr__(self, name: str, value: object) -> None:
            if name in {"__call__", "forward"}:
                raise ResidualHeadContractError(
                    "canonical residual call methods cannot be shadowed"
                )
            module_setattr(self, name, value)

        def __delattr__(self, name: str) -> None:
            if name in {"__call__", "forward"}:
                raise ResidualHeadContractError(
                    "canonical residual call methods cannot be deleted"
                )
            module_delattr(self, name)

        def __init__(self, *, device: torch.device | str | None = None) -> None:
            module_init(self)
            module_register_parameter(
                self,
                "fc1_weight",
                parameter_ctor(torch_empty((256, 13), dtype=torch_float32, device=device)),
            )
            module_register_parameter(
                self,
                "fc1_bias",
                parameter_ctor(torch_empty((256,), dtype=torch_float32, device=device)),
            )
            module_register_parameter(
                self,
                "fc2_weight",
                parameter_ctor(torch_empty((512, 256), dtype=torch_float32, device=device)),
            )
            module_register_parameter(
                self,
                "fc2_bias",
                parameter_ctor(torch_empty((512,), dtype=torch_float32, device=device)),
            )

        def __call__(self, *args: object, **kwargs: object) -> Tensor:
            return canonical_head_forward(self, *args, **kwargs)

        def forward(
            self,
            tokens: Tensor,
            *,
            system: str,
            active_mask: Tensor,
            seed: int | None = None,
            global_optimizer_step: int | None = None,
        ) -> Tensor:
            with registry_lock:
                bound_rows, training = live_parameters(self)
            checked_tokens, _, _ = snapshot_tokens(tokens, active_mask, system)
            rows = dict(bound_rows)
            hidden = functional_linear(
                checked_tokens,
                rows["residual.fc1.weight"],
                rows["residual.fc1.bias"],
            )
            hidden = functional_gelu(hidden, approximate="none")
            if training:
                if seed is None or global_optimizer_step is None:
                    raise ResidualHeadHold(
                        "training requires seed and global_optimizer_step"
                    )
                mask = dropout_mask_impl(
                    seed,
                    global_optimizer_step,
                    int(checked_tokens.shape[1]),
                    checked_tokens.device,
                )
                hidden = hidden * mask
            elif seed is not None or global_optimizer_step is not None:
                raise ResidualHeadContractError(
                    "eval forward must not consume head dropout counters"
                )
            projected = functional_linear(
                hidden,
                rows["residual.fc2.weight"],
                rows["residual.fc2.bias"],
            ).contiguous()
            if not bool(torch_isfinite(projected).all().item()):
                raise ResidualHeadContractError("residual projection became nonfinite")
            return projected

    canonical_head_init = ResidualHead.__init__
    canonical_head_call = ResidualHead.__call__
    canonical_head_forward = ResidualHead.forward
    canonical_head_setattr = ResidualHead.__setattr__
    canonical_head_delattr = ResidualHead.__delattr__
    expected_head_class_keys = tuple(
        type_getattribute(ResidualHead, "__dict__")
    )
    head_class_sealed = True

    def ordered_scores_impl(
        model: object,
        tokens: object,
        active_mask: object,
        text_embeddings: object,
        system: object,
        seed: int | None,
        step: int | None,
    ) -> Tensor:
        checked_tokens, checked_mask, checked_system = snapshot_tokens(
            tokens, active_mask, system
        )
        if type(text_embeddings) is not tensor_type:
            raise TypeError("text_embeddings must be an exact torch.Tensor")
        if (
            text_embeddings.dtype is not torch_float32
            or text_embeddings.ndim != 2
            or text_embeddings.shape[0] < 1
            or text_embeddings.shape[1] != 512
            or text_embeddings.device != checked_tokens.device
            or not tensor_is_contiguous_primitive(text_embeddings)
            or text_embeddings.requires_grad
        ):
            raise ResidualHeadContractError("text embeddings must be frozen float32 [Q,512]")
        text = tensor_clone_primitive(
            tensor_detach_primitive(text_embeddings),
            memory_format=torch_contiguous_format,
        )
        if not bool(torch_isfinite(text).all().item()):
            raise ResidualHeadContractError("text embeddings must be finite")
        text_norm = vector_norm(text, dim=-1)
        if not bool((torch_abs(text_norm - 1.0) <= 1e-5).all().item()):
            raise ResidualHeadContractError("text embeddings must already be L2 normalized")
        if type(model) is not ResidualHead:
            raise ResidualHeadContractError("model must be exact ResidualHead")
        projected = canonical_head_forward(
            model,
            checked_tokens,
            system=checked_system,
            active_mask=checked_mask,
            seed=seed,
            global_optimizer_step=step,
        )
        norms = vector_norm(projected, dim=-1, keepdim=True)
        normalized = projected / torch_clamp(norms, min=1e-12)
        similarity = torch_einsum("pbkd,qd->pbkq", normalized, text)
        mask4 = checked_mask.unsqueeze(-1)
        any_active = checked_mask.any(dim=2, keepdim=True).unsqueeze(-1)
        negative = torch_full_like(similarity, float("-inf"))
        attention_logits = torch_where(mask4, 4.0 * similarity, negative)
        safe_logits = torch_where(any_active, attention_logits, torch_zeros_like(attention_logits))
        maximum = safe_logits.max(dim=2, keepdim=True).values
        numerator = torch_where(
            mask4,
            torch_exp(safe_logits - maximum),
            torch_zeros_like(safe_logits),
        )
        denominator = numerator.sum(dim=2, keepdim=True)
        weights = torch_where(
            any_active,
            numerator / torch_clamp(denominator, min=1e-30),
            torch_zeros_like(numerator),
        )
        scores = (weights * similarity).sum(dim=2).contiguous()
        scores = torch_where(any_active.squeeze(2), scores, torch_zeros_like(scores))
        if not bool(torch_isfinite(scores).all().item()):
            raise ResidualHeadContractError("ordered residual scores became nonfinite")
        if not bool(torch_logical_and(scores >= -1.0, scores <= 1.0).all().item()):
            raise ResidualHeadContractError("ordered residual scores left [-1,1]")
        return scores

    def objective_wrapper(
        model: object,
        tokens: object,
        active_mask: object,
        base_motion_ordered: object,
        text_embeddings: object,
        logit_scale: object,
        positive_mask: object,
        system: object,
        seed: int | None,
        step: int | None,
    ) -> PhasePairObjectiveOutput:
        if type(base_motion_ordered) is not tensor_type:
            raise TypeError("base_motion_ordered must be exact torch.Tensor")
        if base_motion_ordered.requires_grad:
            raise ResidualHeadContractError("frozen base motion must not require gradients")
        if not tensor_is_contiguous_primitive(base_motion_ordered):
            raise ResidualHeadContractError("frozen base motion must be contiguous")
        base_snapshot = tensor_clone_primitive(
            tensor_detach_primitive(base_motion_ordered),
            memory_format=torch_contiguous_format,
        )
        if type(text_embeddings) is not tensor_type:
            raise TypeError("text_embeddings must be an exact torch.Tensor")
        if text_embeddings.requires_grad:
            raise ResidualHeadContractError("frozen text embeddings must not require gradients")
        if not tensor_is_contiguous_primitive(text_embeddings):
            raise ResidualHeadContractError("frozen text embeddings must be contiguous")
        text_snapshot = tensor_clone_primitive(
            tensor_detach_primitive(text_embeddings),
            memory_format=torch_contiguous_format,
        )
        if type(logit_scale) not in (tensor_type, parameter_type):
            raise TypeError("logit_scale must be an exact Tensor or Parameter")
        if logit_scale.requires_grad:
            raise ResidualHeadContractError("frozen base logit_scale must not require gradients")
        if not tensor_is_contiguous_primitive(logit_scale):
            raise ResidualHeadContractError("frozen base logit_scale must be contiguous")
        scale_snapshot = tensor_clone_primitive(
            tensor_detach_primitive(logit_scale),
            memory_format=torch_contiguous_format,
        )
        if type(positive_mask) is not tensor_type:
            raise TypeError("positive_mask must be an exact torch.Tensor")
        if positive_mask.requires_grad:
            raise ResidualHeadContractError("positive_mask must not require gradients")
        if not tensor_is_contiguous_primitive(positive_mask):
            raise ResidualHeadContractError("positive_mask must be contiguous")
        positive_snapshot = tensor_clone_primitive(
            tensor_detach_primitive(positive_mask),
            memory_format=torch_contiguous_format,
        )
        scores = ordered_scores_impl(
            model,
            tokens,
            active_mask,
            text_snapshot,
            system,
            seed,
            step,
        )
        output = objective_impl(
            base_snapshot,
            text_snapshot,
            scale_snapshot,
            positive_snapshot,
            ordered_residual_scores=scores,
        )
        if type(output) is not objective_output_type:
            raise ResidualHeadContractError("objective returned an unexpected output type")
        return output

    return (
        ResidualHead,
        parameter_raw,
        inventory_bytes,
        initialize_impl,
        verify_initializer_impl,
        dropout_receipt_impl,
        dropout_mask_impl,
        ordered_scores_impl,
        objective_wrapper,
    )


(
    PhasePairResidualHead,
    canonical_residual_parameter_bytes,
    canonical_residual_inventory_bytes,
    initialize_residual_head,
    verify_residual_head_initializer_receipt,
    canonical_head_dropout_receipt,
    canonical_head_dropout_mask,
    residual_ordered_scores,
    phasepair_residual_objective,
) = _make_public_api()


__all__ = [
    "AUTHORITY",
    "DROPOUT_DOMAIN",
    "DROPOUT_RECEIPT_SCHEMA",
    "INITIALIZER_DOMAIN",
    "INITIALIZER_RECEIPT_SCHEMA",
    "PhasePairResidualHead",
    "ResidualHeadContractError",
    "ResidualHeadHold",
    "STATUS",
    "SYSTEMS",
    "canonical_head_dropout_mask",
    "canonical_head_dropout_receipt",
    "canonical_residual_inventory_bytes",
    "canonical_residual_parameter_bytes",
    "initialize_residual_head",
    "phasepair_residual_objective",
    "residual_ordered_scores",
    "verify_residual_head_initializer_receipt",
]
