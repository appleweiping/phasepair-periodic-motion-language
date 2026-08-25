"""Fail-closed PhasePair optimizer and step transaction primitives.

This module deliberately starts at the CPU-synthetic boundary.  It executes a
real ``torch.optim.AdamW`` and real autograd, but it cannot mint production
authority and it cannot consume the still-separate live CLIP lease.  The
synthetic admission is opaque, single-use, and bound to the exact live
parameter objects and storage observed at issuance.

The production bridge will reuse the transaction machinery only after a live
text-runtime lease and the private data/run authority are mutually bound.
Nothing in this module is a training result.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import struct
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Final
from weakref import WeakKeyDictionary

import torch
from torch import Tensor, nn

from phasepair_core import checkpoint as _checkpoint_module
from phasepair_core._identity_registry import make_identity_weak_registry


STATUS: Final = "CPU_SYNTHETIC_TRAINER_VALIDATED_NONPRODUCTION_AUTHORITY0_NO_RESULT"
ADMISSION_STATUS: Final = "CPU_SYNTHETIC_ADMISSION_ISSUED_AUTHORITY0"
STEP_STATUS: Final = "CPU_SYNTHETIC_STEP_COMMITTED_AUTHORITY0_NO_RESULT"
LEARNING_RATE: Final = 1.0e-4
BETAS: Final = (0.9, 0.999)
EPSILON: Final = 1.0e-8
DECAY_WEIGHT_DECAY: Final = 1.0e-4
NO_DECAY_WEIGHT_DECAY: Final = 0.0
GRAD_CLIP_NORM: Final = 1.0


class TrainerContractError(ValueError):
    """Raised before mutation when a trainer contract is violated."""


class TrainingAdmissionBurnedError(TrainerContractError):
    """Raised when an admission or session is reused after a terminal edge."""


class TrainingTransactionError(RuntimeError):
    """Raised after a failed step has been rolled back and retired."""


def _exact_uint(value: object, label: str, *, bits: int) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < 0 or value >= 1 << bits:
        raise TrainerContractError(f"{label} is outside uint{bits}")
    return value


def _exact_ascii(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a nonempty exact built-in str")
    try:
        encoded = value.encode("ascii")
    except UnicodeEncodeError as exc:
        raise TrainerContractError(f"{label} must be ASCII") from exc
    if encoded.decode("ascii") != value or any(byte < 0x21 or byte > 0x7E for byte in encoded):
        raise TrainerContractError(f"{label} contains forbidden characters")
    return value


def _exact_name_tuple(
    value: object,
    label: str,
    _ascii: Callable[[object, str], str] = _exact_ascii,
) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact built-in tuple")
    checked: list[str] = []
    for index, item in enumerate(value):
        checked.append(_ascii(item, f"{label}[{index}]"))
    result = tuple(checked)
    if not result:
        raise TrainerContractError(f"{label} cannot be empty")
    if result != tuple(sorted(result, key=lambda item: item.encode("utf-8"))):
        raise TrainerContractError(f"{label} must use UTF-8 byte order")
    if len(result) != len(set(result)):
        raise TrainerContractError(f"{label} contains duplicate names")
    return result


def _tensor_bytes(
    value: Tensor,
    _tensor_type: type[Tensor] = Tensor,
    _parameter_type: type[nn.Parameter] = nn.Parameter,
    _float32: torch.dtype = torch.float32,
    _detach: Callable[..., Tensor] = Tensor.detach,
    _contiguous: Callable[..., Tensor] = Tensor.contiguous,
    _numpy: Callable[..., object] = Tensor.numpy,
    _is_contiguous: Callable[..., bool] = Tensor.is_contiguous,
    _storage_offset: Callable[..., int] = Tensor.storage_offset,
) -> bytes:
    if type(value) not in {_tensor_type, _parameter_type}:
        raise TypeError("tensor must be an exact Tensor or Parameter")
    if value.device.type != "cpu" or value.dtype != _float32:
        raise TrainerContractError("synthetic trainer tensors must be CPU float32")
    if not _is_contiguous(value) or _storage_offset(value) != 0:
        raise TrainerContractError("synthetic trainer tensors must be zero-offset contiguous")
    detached = _contiguous(_detach(value))
    array = _numpy(detached)
    return array.tobytes(order="C")  # type: ignore[union-attr]


def _write_field(output: bytearray, raw: bytes) -> None:
    output.extend(len(raw).to_bytes(8, "big"))
    output.extend(raw)


def _tensor_record_bytes(
    name: str,
    value: Tensor,
    _write: Callable[[bytearray, bytes], None] = _write_field,
    _raw: Callable[[Tensor], bytes] = _tensor_bytes,
) -> bytes:
    output = bytearray(b"phasepair-live-tensor-v1\x00")
    _write(output, name.encode("utf-8"))
    _write(output, str(value.dtype).encode("ascii"))
    output.extend(value.ndim.to_bytes(2, "big"))
    for dimension in value.shape:
        output.extend(int(dimension).to_bytes(8, "big"))
    _write(output, _raw(value))
    return bytes(output)


def _named_parameters(
    model: nn.Module,
    *,
    allow_grad: bool = False,
    _module_type: type[nn.Module] = nn.Module,
    _named: Callable[..., object] = nn.Module.named_parameters,
    _parameter_type: type[nn.Parameter] = nn.Parameter,
    _ascii: Callable[[object, str], str] = _exact_ascii,
    _float32: torch.dtype = torch.float32,
    _isfinite: Callable[[Tensor], Tensor] = torch.isfinite,
    _tensor_detach: Callable[..., Tensor] = Tensor.detach,
    _tensor_is_contiguous: Callable[..., bool] = Tensor.is_contiguous,
    _tensor_storage_offset: Callable[..., int] = Tensor.storage_offset,
    _tensor_numel: Callable[..., int] = Tensor.numel,
    _tensor_element_size: Callable[..., int] = Tensor.element_size,
    _tensor_untyped_storage: Callable[..., torch.UntypedStorage] = (
        Tensor.untyped_storage
    ),
    _storage_nbytes: Callable[..., int] = torch.UntypedStorage.nbytes,
    _storage_data_ptr: Callable[..., int] = torch.UntypedStorage.data_ptr,
) -> tuple[tuple[str, nn.Parameter], ...]:
    if type(allow_grad) is not bool:
        raise TypeError("allow_grad must be an exact built-in bool")
    if not isinstance(model, _module_type):
        raise TypeError("model must be an nn.Module")
    rows = tuple(
        _named(
            model,
            prefix="",
            recurse=True,
            remove_duplicate=False,
        )
    )
    if not rows:
        raise TrainerContractError("model has no parameters")
    names: list[str] = []
    seen_objects: set[int] = set()
    seen_storage: set[tuple[int, int]] = set()
    for index, (name, parameter) in enumerate(rows):
        checked_name = _ascii(name, f"parameter name {index}")
        if type(parameter) is not _parameter_type:
            raise TypeError(f"parameter {checked_name} must be exactly nn.Parameter")
        if (
            parameter.dtype != _float32
            or parameter.device.type != "cpu"
            or not parameter.requires_grad
            or not parameter.is_leaf
            or not _tensor_is_contiguous(parameter)
            or _tensor_storage_offset(parameter) != 0
            or _tensor_numel(parameter) < 1
        ):
            raise TrainerContractError(f"parameter invariant mismatch: {checked_name}")
        if not allow_grad and parameter.grad is not None:
            raise TrainerContractError(f"stale gradient before admission: {checked_name}")
        if not bool(_isfinite(_tensor_detach(parameter)).all().item()):
            raise TrainerContractError(f"nonfinite parameter before admission: {checked_name}")
        storage = _tensor_untyped_storage(parameter)
        extent = _tensor_numel(parameter) * _tensor_element_size(parameter)
        storage_nbytes = _storage_nbytes(storage)
        if storage_nbytes != extent:
            raise TrainerContractError(f"parameter storage extent mismatch: {checked_name}")
        object_id = id(parameter)
        storage_id = (_storage_data_ptr(storage), storage_nbytes)
        if object_id in seen_objects or storage_id in seen_storage:
            raise TrainerContractError("parameter object/storage alias detected")
        seen_objects.add(object_id)
        seen_storage.add(storage_id)
        names.append(checked_name)
    if len(names) != len(set(names)):
        raise TrainerContractError("duplicate registered parameter name")
    return rows


def _model_digest(
    rows: tuple[tuple[str, nn.Parameter], ...],
    _write: Callable[[bytearray, bytes], None] = _write_field,
    _record: Callable[[str, Tensor], bytes] = _tensor_record_bytes,
    _sha256: Callable[..., object] = hashlib.sha256,
) -> bytes:
    output = bytearray(b"phasepair-live-model-state-v1\x00")
    output.extend(len(rows).to_bytes(4, "big"))
    for name, parameter in rows:
        _write(output, _record(name, parameter))
    return _sha256(bytes(output)).digest()  # type: ignore[attr-defined]


def _parameter_binding(
    rows: tuple[tuple[str, nn.Parameter], ...],
    _tensor_untyped_storage: Callable[..., torch.UntypedStorage] = (
        Tensor.untyped_storage
    ),
    _storage_data_ptr: Callable[..., int] = torch.UntypedStorage.data_ptr,
    _storage_nbytes: Callable[..., int] = torch.UntypedStorage.nbytes,
) -> tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]]:
    object_ids = tuple(id(parameter) for _, parameter in rows)
    storage_intervals: list[tuple[int, int, int]] = []
    for _, parameter in rows:
        storage = _tensor_untyped_storage(parameter)
        start = _storage_data_ptr(storage)
        storage_intervals.append(
            (int(storage._cdata), start, start + _storage_nbytes(storage))
        )
    return object_ids, tuple(storage_intervals)


def _optimizer_digest(
    optimizer: torch.optim.AdamW,
    ordered_rows: tuple[tuple[str, nn.Parameter], ...],
    decay_names: tuple[str, ...],
    no_decay_names: tuple[str, ...],
    _write: Callable[[bytearray, bytes], None] = _write_field,
    _record: Callable[[str, Tensor], bytes] = _tensor_record_bytes,
    _sha256: Callable[..., object] = hashlib.sha256,
    _pack: Callable[..., bytes] = struct.pack,
    _tensor_type: type[Tensor] = Tensor,
) -> bytes:
    output = bytearray(b"phasepair-adamw-state-v1\x00")
    output.extend(_pack(">d", 1.0e-4))
    output.extend(_pack(">dd", 0.9, 0.999))
    output.extend(_pack(">d", 1.0e-8))
    by_name = dict(ordered_rows)
    for ordinal, (group_names, weight_decay) in enumerate(
        ((decay_names, 1.0e-4), (no_decay_names, 0.0))
    ):
        output.extend(ordinal.to_bytes(1, "big"))
        output.extend(_pack(">d", weight_decay))
        output.extend(len(group_names).to_bytes(4, "big"))
        for name in group_names:
            parameter = by_name[name]
            _write(output, name.encode("utf-8"))
            state = optimizer.state.get(parameter)
            if not state:
                output.extend((0).to_bytes(1, "big"))
                continue
            output.extend((1).to_bytes(1, "big"))
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise TrainerContractError(f"unexpected AdamW state keys: {name}")
            for state_name in ("step", "exp_avg", "exp_avg_sq"):
                state_tensor = state[state_name]
                if type(state_tensor) is not _tensor_type:
                    raise TypeError(f"AdamW state {state_name} must be exactly Tensor")
                _write(output, _record(f"{name}/{state_name}", state_tensor))
    return _sha256(bytes(output)).digest()  # type: ignore[attr-defined]


@dataclass(frozen=True, slots=True)
class _AdmissionRecord:
    model: nn.Module
    model_type: type[nn.Module]
    rows: tuple[tuple[str, nn.Parameter], ...]
    decay_names: tuple[str, ...]
    no_decay_names: tuple[str, ...]
    run_id: str
    attempt_id: str
    system_id: str
    seed: int
    issued_model_sha256: bytes
    parameter_object_ids: tuple[int, ...]
    parameter_storage_intervals: tuple[tuple[int, int, int], ...]


@dataclass(slots=True)
class _SessionRecord:
    admission: _AdmissionRecord
    optimizer: torch.optim.AdamW
    state: str
    next_global_step: int
    lock: threading.Lock
    expected_model_sha256: bytes
    expected_optimizer_sha256: bytes


class SyntheticOptimizerAdmission:
    """Opaque, single-use CPU-synthetic admission; direct construction is forbidden."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise TrainerContractError("admissions are minted only by the issuer")


class SyntheticTrainingSession:
    """Opaque optimizer session; the optimizer is never exposed publicly."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise TrainerContractError("sessions are minted only from a live admission")


class StepReceipt:
    """Opaque committed-step receipt with canonical, revalidated bytes."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise TrainerContractError("step receipts are minted only after commit")


_REGISTRY_LOCK = threading.RLock()
(
    _ADMISSION_SET,
    _ADMISSION_GET,
    _ADMISSION_ITEMS,
    _,
    _,
) = make_identity_weak_registry(_REGISTRY_LOCK)
(
    _ADMISSION_STATE_SET,
    _ADMISSION_STATE_GET,
    _,
    _,
    _,
) = make_identity_weak_registry(_REGISTRY_LOCK)
(
    _SESSION_SET,
    _SESSION_GET,
    _,
    _SESSION_VALUES,
    _,
) = make_identity_weak_registry(_REGISTRY_LOCK)
(
    _RECEIPT_SET,
    _RECEIPT_GET,
    _,
    _,
    _,
) = make_identity_weak_registry(_REGISTRY_LOCK)

# These module-visible maps are decoys retained for rebinding regression tests.
# Canonical operations capture the identity-only closures above.
_ADMISSIONS: WeakKeyDictionary[SyntheticOptimizerAdmission, _AdmissionRecord] = WeakKeyDictionary()
_ADMISSION_STATES: WeakKeyDictionary[SyntheticOptimizerAdmission, str] = WeakKeyDictionary()
_SESSIONS: WeakKeyDictionary[SyntheticTrainingSession, _SessionRecord] = WeakKeyDictionary()
_RECEIPTS: WeakKeyDictionary[StepReceipt, bytes] = WeakKeyDictionary()


def _admission_records_overlap(
    left: _AdmissionRecord,
    right: _AdmissionRecord,
    _binding: Callable[
        [tuple[tuple[str, nn.Parameter], ...]],
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    ] = _parameter_binding,
    _named: Callable[[nn.Module], tuple[tuple[str, nn.Parameter], ...]] = (
        _named_parameters
    ),
) -> bool:
    if left.model is right.model:
        return True
    current_left_ids, current_left_intervals = _binding(_named(left.model))
    frozen_right_ids, frozen_right_intervals = _binding(right.rows)
    current_right_ids, current_right_intervals = _binding(_named(right.model))
    left_object_ids = left.parameter_object_ids + current_left_ids
    left_storage_intervals = (
        left.parameter_storage_intervals + current_left_intervals
    )
    right_object_ids = (
        right.parameter_object_ids + frozen_right_ids + current_right_ids
    )
    right_storage_intervals = (
        right.parameter_storage_intervals
        + frozen_right_intervals
        + current_right_intervals
    )
    if any(
        object_id in right_object_ids
        for object_id in left_object_ids
    ):
        return True
    return any(
        left_start < right_stop and right_start < left_stop
        for _, left_start, left_stop in left_storage_intervals
        for _, right_start, right_stop in right_storage_intervals
    )


def _require_unowned_admission(
    candidate: _AdmissionRecord,
    admission_items: Callable[[], tuple[tuple[object, object], ...]],
    state_get: Callable[[object], object | None],
    session_values: Callable[[], tuple[object, ...]],
    _overlap: Callable[[_AdmissionRecord, _AdmissionRecord], bool] = (
        _admission_records_overlap
    ),
) -> None:
    for admission, existing in admission_items():
        state = state_get(admission)
        if state in {"ISSUED", "VALIDATING", "RESUME_VALIDATING"} and _overlap(
            candidate, existing
        ):
            raise TrainerContractError(
                "model, parameter, or storage is already bound to a live admission"
            )
    for session_record in session_values():
        if session_record.state == "ACTIVE" and _overlap(
            candidate, session_record.admission
        ):
            raise TrainerContractError(
                "model, parameter, or storage is already bound to an active session"
            )


def _retire_session_record(
    record: _SessionRecord,
    _lock: threading.RLock = _REGISTRY_LOCK,
) -> None:
    with _lock:
        record.state = "RETIRED"


def _make_constructor_counter() -> tuple[Callable[[], None], Callable[[], int]]:
    counter_lock = threading.Lock()
    count = 0

    def increment() -> None:
        nonlocal count
        with counter_lock:
            count += 1

    def read() -> int:
        with counter_lock:
            return count

    return increment, read


_INCREMENT_ADAMW_CONSTRUCTOR, _READ_ADAMW_CONSTRUCTOR_COUNT = (
    _make_constructor_counter()
)
del _make_constructor_counter


def _issue_cpu_synthetic_admission_impl(
    model: nn.Module,
    *,
    decay_names: tuple[str, ...],
    no_decay_names: tuple[str, ...],
    run_id: str,
    attempt_id: str,
    system_id: str,
    seed: int,
    _name_tuple: Callable[[object, str], tuple[str, ...]] = _exact_name_tuple,
    _uint: Callable[..., int] = _exact_uint,
    _ascii: Callable[[object, str], str] = _exact_ascii,
    _named: Callable[[nn.Module], tuple[tuple[str, nn.Parameter], ...]] = _named_parameters,
    _digest: Callable[[tuple[tuple[str, nn.Parameter], ...]], bytes] = _model_digest,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _admission_set: Callable[[object, object], None] = _ADMISSION_SET,
    _admission_items: Callable[[], tuple[tuple[object, object], ...]] = (
        _ADMISSION_ITEMS
    ),
    _state_set: Callable[[object, object], None] = _ADMISSION_STATE_SET,
    _state_get: Callable[[object], object | None] = _ADMISSION_STATE_GET,
    _session_values: Callable[[], tuple[object, ...]] = _SESSION_VALUES,
    _binding: Callable[
        [tuple[tuple[str, nn.Parameter], ...]],
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    ] = _parameter_binding,
    _require_unowned: Callable[..., None] = _require_unowned_admission,
    _admission_type: type[SyntheticOptimizerAdmission] = SyntheticOptimizerAdmission,
    _record_type: type[_AdmissionRecord] = _AdmissionRecord,
    _object_new: Callable[[type[object]], object] = object.__new__,
) -> SyntheticOptimizerAdmission:
    """Bind an exact CPU model and canonical two-group partition without AdamW."""

    decay = _name_tuple(decay_names, "decay_names")
    no_decay = _name_tuple(no_decay_names, "no_decay_names")
    if set(decay) & set(no_decay):
        raise TrainerContractError("optimizer group intersection is nonempty")
    checked_seed = _uint(seed, "seed", bits=64)
    checked_run_id = _ascii(run_id, "run_id")
    checked_attempt_id = _ascii(attempt_id, "attempt_id")
    checked_system_id = _ascii(system_id, "system_id")
    with _lock:
        rows = _named(model)
        names = tuple(name for name, _ in rows)
        if set(decay) | set(no_decay) != set(names):
            raise TrainerContractError(
                "optimizer group union differs from live parameters"
            )
        parameter_object_ids, parameter_storage_intervals = _binding(rows)
        record = _record_type(
            model=model,
            model_type=type(model),
            rows=rows,
            decay_names=decay,
            no_decay_names=no_decay,
            run_id=checked_run_id,
            attempt_id=checked_attempt_id,
            system_id=checked_system_id,
            seed=checked_seed,
            issued_model_sha256=_digest(rows),
            parameter_object_ids=parameter_object_ids,
            parameter_storage_intervals=parameter_storage_intervals,
        )
        _require_unowned(
            record,
            _admission_items,
            _state_get,
            _session_values,
        )
        final_rows = _named(model)
        final_object_ids, final_storage_intervals = _binding(final_rows)
        if (
            tuple(name for name, _ in final_rows) != names
            or final_object_ids != parameter_object_ids
            or final_storage_intervals != parameter_storage_intervals
            or _digest(final_rows) != record.issued_model_sha256
        ):
            raise TrainerContractError(
                "candidate parameter binding changed during admission issuance"
            )
        admission = _object_new(_admission_type)
        _admission_set(admission, record)
        _state_set(admission, "ISSUED")
    return admission


def _bind_issue_admission(
    implementation: Callable[..., SyntheticOptimizerAdmission],
) -> Callable[..., SyntheticOptimizerAdmission]:
    def issue_cpu_synthetic_admission(
        model: nn.Module,
        *,
        decay_names: tuple[str, ...],
        no_decay_names: tuple[str, ...],
        run_id: str,
        attempt_id: str,
        system_id: str,
        seed: int,
    ) -> SyntheticOptimizerAdmission:
        return implementation(
            model,
            decay_names=decay_names,
            no_decay_names=no_decay_names,
            run_id=run_id,
            attempt_id=attempt_id,
            system_id=system_id,
            seed=seed,
        )

    return issue_cpu_synthetic_admission


issue_cpu_synthetic_admission = _bind_issue_admission(
    _issue_cpu_synthetic_admission_impl
)
del _bind_issue_admission


def _cpu_synthetic_admission_status_impl(
    admission: SyntheticOptimizerAdmission,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _state_get: Callable[[object], object | None] = _ADMISSION_STATE_GET,
    _admission_type: type[SyntheticOptimizerAdmission] = SyntheticOptimizerAdmission,
) -> str:
    if type(admission) is not _admission_type:
        raise TypeError("admission must be exactly SyntheticOptimizerAdmission")
    with _lock:
        state = _state_get(admission)
    if state is None:
        raise TrainerContractError("admission was not issued by this module")
    return f"CPU_SYNTHETIC_ADMISSION_ISSUED_AUTHORITY0:{state}"


def _bind_admission_status(
    implementation: Callable[[SyntheticOptimizerAdmission], str],
) -> Callable[[SyntheticOptimizerAdmission], str]:
    def cpu_synthetic_admission_status(
        admission: SyntheticOptimizerAdmission,
    ) -> str:
        return implementation(admission)

    return cpu_synthetic_admission_status


cpu_synthetic_admission_status = _bind_admission_status(
    _cpu_synthetic_admission_status_impl
)
del _bind_admission_status


def _open_cpu_synthetic_training_session_impl(
    admission: SyntheticOptimizerAdmission,
    _adamw: type[torch.optim.AdamW] = torch.optim.AdamW,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _admission_get: Callable[[object], object | None] = _ADMISSION_GET,
    _state_get: Callable[[object], object | None] = _ADMISSION_STATE_GET,
    _state_set: Callable[[object, object], None] = _ADMISSION_STATE_SET,
    _session_set: Callable[[object, object], None] = _SESSION_SET,
    _named: Callable[[nn.Module], tuple[tuple[str, nn.Parameter], ...]] = _named_parameters,
    _digest: Callable[[tuple[tuple[str, nn.Parameter], ...]], bytes] = _model_digest,
    _binding: Callable[
        [tuple[tuple[str, nn.Parameter], ...]],
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    ] = _parameter_binding,
    _optimizer_hash: Callable[..., bytes] = _optimizer_digest,
    _increment_constructor: Callable[[], None] = _INCREMENT_ADAMW_CONSTRUCTOR,
    _admission_type: type[SyntheticOptimizerAdmission] = SyntheticOptimizerAdmission,
    _session_type: type[SyntheticTrainingSession] = SyntheticTrainingSession,
    _session_record_type: type[_SessionRecord] = _SessionRecord,
    _object_new: Callable[[type[object]], object] = object.__new__,
    _new_lock: Callable[[], threading.Lock] = threading.Lock,
) -> SyntheticTrainingSession:
    """Consume one admission and construct exactly one canonical AdamW."""

    if type(admission) is not _admission_type:
        raise TypeError("admission must be exactly SyntheticOptimizerAdmission")
    with _lock:
        record = _admission_get(admission)
        state = _state_get(admission)
        if record is None or state is None:
            raise TrainerContractError("admission was not issued by this module")
        if state != "ISSUED":
            raise TrainingAdmissionBurnedError("admission is terminal")
        _state_set(admission, "VALIDATING")
    try:
        if type(record.model) is not record.model_type:
            raise TrainerContractError("live model type changed")
        current_rows = _named(record.model)
        current_object_ids, current_storage_intervals = _binding(current_rows)
        if (
            tuple(name for name, _ in current_rows)
            != tuple(name for name, _ in record.rows)
            or current_object_ids != record.parameter_object_ids
            or current_storage_intervals != record.parameter_storage_intervals
        ):
            raise TrainerContractError(
                "live parameter object/name/storage binding changed"
            )
        if _digest(current_rows) != record.issued_model_sha256:
            raise TrainerContractError("live parameter bytes changed after admission")
        by_name = dict(current_rows)
        groups = [
            {
                "params": [by_name[name] for name in record.decay_names],
                "lr": 1.0e-4,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
                "weight_decay": 1.0e-4,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            },
            {
                "params": [by_name[name] for name in record.no_decay_names],
                "lr": 1.0e-4,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
                "weight_decay": 0.0,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            },
        ]
        _increment_constructor()
        optimizer = _adamw(groups)
        if len(optimizer.param_groups) != 2 or optimizer.state:
            raise TrainerContractError("AdamW step-zero state/group mismatch")
        for ordinal, expected_names in enumerate(
            (record.decay_names, record.no_decay_names)
        ):
            group = optimizer.param_groups[ordinal]
            expected_parameters = [by_name[name] for name in expected_names]
            if tuple(id(item) for item in group["params"]) != tuple(
                id(item) for item in expected_parameters
            ):
                raise TrainerContractError("AdamW parameter group order drift")
            expected_wd = 1.0e-4 if ordinal == 0 else 0.0
            expected = {
                "lr": 1.0e-4,
                "betas": (0.9, 0.999),
                "eps": 1.0e-8,
                "weight_decay": expected_wd,
                "amsgrad": False,
                "maximize": False,
                "foreach": False,
                "capturable": False,
                "differentiable": False,
                "fused": False,
            }
            for key, value in expected.items():
                if group[key] != value:
                    raise TrainerContractError(f"AdamW hyperparameter drift: {key}")
        session = _object_new(_session_type)
        post_object_ids, post_storage_intervals = _binding(current_rows)
        if (
            post_object_ids != record.parameter_object_ids
            or post_storage_intervals != record.parameter_storage_intervals
            or _digest(current_rows) != record.issued_model_sha256
        ):
            raise TrainerContractError(
                "live parameter binding or bytes changed during AdamW construction"
            )
        optimizer_sha256 = _optimizer_hash(
            optimizer,
            current_rows,
            record.decay_names,
            record.no_decay_names,
        )
        with _lock:
            _session_set(session, _session_record_type(
                admission=record,
                optimizer=optimizer,
                state="ACTIVE",
                next_global_step=0,
                lock=_new_lock(),
                expected_model_sha256=record.issued_model_sha256,
                expected_optimizer_sha256=optimizer_sha256,
            ))
            _state_set(admission, "CONSUMED")
        return session
    except BaseException:
        with _lock:
            _state_set(admission, "BURNED")
        raise


def _bind_open_session(
    implementation: Callable[
        [SyntheticOptimizerAdmission], SyntheticTrainingSession
    ],
) -> Callable[[SyntheticOptimizerAdmission], SyntheticTrainingSession]:
    def open_cpu_synthetic_training_session(
        admission: SyntheticOptimizerAdmission,
    ) -> SyntheticTrainingSession:
        return implementation(admission)

    return open_cpu_synthetic_training_session


open_cpu_synthetic_training_session = _bind_open_session(
    _open_cpu_synthetic_training_session_impl
)
del _bind_open_session


def _bind_constructor_count(reader: Callable[[], int]) -> Callable[[], int]:
    def cpu_synthetic_optimizer_constructor_count() -> int:
        return reader()

    return cpu_synthetic_optimizer_constructor_count


cpu_synthetic_optimizer_constructor_count = _bind_constructor_count(
    _READ_ADAMW_CONSTRUCTOR_COUNT
)
del _bind_constructor_count


def _receipt_bytes(
    payload: dict[str, object],
    _dumps: Callable[..., str] = json.dumps,
) -> bytes:
    return (
        _dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _canonical_json_bytes(
    payload: dict[str, object],
    _dumps: Callable[..., str] = json.dumps,
) -> bytes:
    return (
        _dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _tensor_json(
    name: str,
    value: Tensor,
    _raw: Callable[[Tensor], bytes] = _tensor_bytes,
) -> dict[str, object]:
    return {
        "dtype": "float32",
        "name": name,
        "raw_hex": _raw(value).hex(),
        "shape": [int(item) for item in value.shape],
    }


def _model_state_bytes(
    record: _SessionRecord,
    _canonical: Callable[[dict[str, object]], bytes] = _canonical_json_bytes,
    _tensor_row: Callable[[str, Tensor], dict[str, object]] = _tensor_json,
) -> bytes:
    return _canonical(
        {
            "next_global_step": record.next_global_step,
            "parameters": [
                _tensor_row(name, parameter)
                for name, parameter in record.admission.rows
            ],
            "schema": "phasepair-cpu-synthetic-model-state-v1",
        }
    )


def _group_payloads(admission: _AdmissionRecord) -> list[dict[str, object]]:
    return [
        {
            "amsgrad": False,
            "betas_hex": [float(item).hex() for item in (0.9, 0.999)],
            "capturable": False,
            "differentiable": False,
            "eps_hex": float(1.0e-8).hex(),
            "foreach": False,
            "fused": False,
            "lr_hex": float(1.0e-4).hex(),
            "maximize": False,
            "name": "decay",
            "parameter_names": list(admission.decay_names),
            "weight_decay_hex": float(1.0e-4).hex(),
        },
        {
            "amsgrad": False,
            "betas_hex": [float(item).hex() for item in (0.9, 0.999)],
            "capturable": False,
            "differentiable": False,
            "eps_hex": float(1.0e-8).hex(),
            "foreach": False,
            "fused": False,
            "lr_hex": float(1.0e-4).hex(),
            "maximize": False,
            "name": "no_decay",
            "parameter_names": list(admission.no_decay_names),
            "weight_decay_hex": float(0.0).hex(),
        },
    ]


def _optimizer_state_bytes(
    record: _SessionRecord,
    _canonical: Callable[[dict[str, object]], bytes] = _canonical_json_bytes,
    _tensor_row: Callable[[str, Tensor], dict[str, object]] = _tensor_json,
    _groups: Callable[[_AdmissionRecord], list[dict[str, object]]] = _group_payloads,
) -> bytes:
    rows: list[dict[str, object]] = []
    for name, parameter in record.admission.rows:
        state = record.optimizer.state.get(parameter)
        if not state:
            state_payload: dict[str, object] | None = None
        else:
            if set(state) != {"step", "exp_avg", "exp_avg_sq"}:
                raise TrainerContractError(f"unexpected AdamW state keys: {name}")
            state_payload = {
                key: _tensor_row(f"{name}/{key}", state[key])
                for key in ("step", "exp_avg", "exp_avg_sq")
            }
        rows.append({"name": name, "state": state_payload})
    return _canonical(
        {
            "groups": _groups(record.admission),
            "parameter_states": rows,
            "schema": "phasepair-cpu-synthetic-adamw-state-v1",
        }
    )


def _rng_state_bytes(
    _get_rng_state: Callable[[], Tensor] = torch.get_rng_state,
    _tensor_type: type[Tensor] = Tensor,
    _uint8: torch.dtype = torch.uint8,
) -> bytes:
    state = _get_rng_state()
    if type(state) is not _tensor_type or state.dtype != _uint8 or state.device.type != "cpu":
        raise TrainerContractError("torch CPU RNG state invariant mismatch")
    return bytes(state.tolist())


def _export_cpu_synthetic_checkpoint_sections_impl(
    session: SyntheticTrainingSession,
    *,
    sampler_state: bytes,
    dropout_state: bytes,
    validation_state: bytes,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _session_get: Callable[[object], object | None] = _SESSION_GET,
    _model_state: Callable[[_SessionRecord], bytes] = _model_state_bytes,
    _optimizer_state: Callable[[_SessionRecord], bytes] = _optimizer_state_bytes,
    _rng_state: Callable[[], bytes] = _rng_state_bytes,
    _named: Callable[[nn.Module], tuple[tuple[str, nn.Parameter], ...]] = _named_parameters,
    _binding: Callable[
        [tuple[tuple[str, nn.Parameter], ...]],
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    ] = _parameter_binding,
    _model_hash: Callable[[tuple[tuple[str, nn.Parameter], ...]], bytes] = _model_digest,
    _optimizer_hash: Callable[..., bytes] = _optimizer_digest,
    _retire: Callable[[_SessionRecord], None] = _retire_session_record,
    _session_type: type[SyntheticTrainingSession] = SyntheticTrainingSession,
) -> tuple[tuple[str, bytes], ...]:
    """Derive all checkpoint sections from one quiescent synthetic session."""

    if type(session) is not _session_type:
        raise TypeError("session must be exactly SyntheticTrainingSession")
    for label, raw in (
        ("sampler_state", sampler_state),
        ("dropout_state", dropout_state),
        ("validation_state", validation_state),
    ):
        if type(raw) is not bytes:
            raise TypeError(f"{label} must be exact bytes")
    with _lock:
        record = _session_get(session)
    if record is None:
        raise TrainerContractError("session was not issued by this module")
    if not record.lock.acquire(blocking=False):
        raise TrainingAdmissionBurnedError("session is executing")
    try:
        if record.state != "ACTIVE":
            raise TrainingAdmissionBurnedError("session is terminal")
        current_rows = _named(record.admission.model, allow_grad=True)
        current_object_ids, current_storage_intervals = _binding(current_rows)
        if (
            tuple(name for name, _ in current_rows)
            != tuple(name for name, _ in record.admission.rows)
            or current_object_ids != record.admission.parameter_object_ids
            or current_storage_intervals
            != record.admission.parameter_storage_intervals
        ):
            _retire(record)
            raise TrainerContractError(
                "live parameter object/name/storage binding drift before checkpoint export"
            )
        if _model_hash(current_rows) != record.expected_model_sha256:
            _retire(record)
            raise TrainerContractError("live model bytes drift before checkpoint export")
        if (
            _optimizer_hash(
                record.optimizer,
                current_rows,
                record.admission.decay_names,
                record.admission.no_decay_names,
            )
            != record.expected_optimizer_sha256
        ):
            _retire(record)
            raise TrainerContractError("live optimizer drift before checkpoint export")
        return (
            ("dropout_state", dropout_state),
            ("model_state", _model_state(record)),
            ("optimizer_state", _optimizer_state(record)),
            ("rng_state", _rng_state()),
            ("sampler_state", sampler_state),
            ("validation_state", validation_state),
        )
    except BaseException:
        if record.state == "ACTIVE":
            _retire(record)
        raise
    finally:
        record.lock.release()


def _bind_export_sections(
    implementation: Callable[..., tuple[tuple[str, bytes], ...]],
) -> Callable[..., tuple[tuple[str, bytes], ...]]:
    def export_cpu_synthetic_checkpoint_sections(
        session: SyntheticTrainingSession,
        *,
        sampler_state: bytes,
        dropout_state: bytes,
        validation_state: bytes,
    ) -> tuple[tuple[str, bytes], ...]:
        return implementation(
            session,
            sampler_state=sampler_state,
            dropout_state=dropout_state,
            validation_state=validation_state,
        )

    return export_cpu_synthetic_checkpoint_sections


export_cpu_synthetic_checkpoint_sections = _bind_export_sections(
    _export_cpu_synthetic_checkpoint_sections_impl
)
del _bind_export_sections


def _parse_json(
    raw: bytes,
    label: str,
    _loads: Callable[[str], object] = json.loads,
    _canonical: Callable[[dict[str, object]], bytes] = _canonical_json_bytes,
    _json_error: type[Exception] = json.JSONDecodeError,
) -> dict[str, object]:
    if type(raw) is not bytes or not raw.endswith(b"\n") or b"\r" in raw:
        raise TrainerContractError(f"{label} must be canonical LF JSON bytes")
    try:
        payload = _loads(raw.decode("ascii"))
    except (UnicodeDecodeError, _json_error) as exc:
        raise TrainerContractError(f"{label} JSON parse failure") from exc
    if type(payload) is not dict or _canonical(payload) != raw:
        raise TrainerContractError(f"{label} is not exact canonical JSON")
    return payload


def _parse_tensor_json(
    value: object,
    *,
    expected_name: str,
    expected_shape: tuple[int, ...],
    _frombuffer: Callable[..., Tensor] = torch.frombuffer,
    _float32: torch.dtype = torch.float32,
    _isfinite: Callable[[Tensor], Tensor] = torch.isfinite,
    _prod: Callable[[tuple[int, ...]], int] = math.prod,
) -> bytes:
    if type(value) is not dict or set(value) != {"dtype", "name", "raw_hex", "shape"}:
        raise TrainerContractError("checkpoint tensor row key census mismatch")
    if value["dtype"] != "float32" or value["name"] != expected_name:
        raise TrainerContractError("checkpoint tensor dtype/name mismatch")
    shape = value["shape"]
    if type(shape) is not list or any(type(item) is not int for item in shape):
        raise TrainerContractError("checkpoint tensor shape type mismatch")
    if tuple(shape) != expected_shape:
        raise TrainerContractError("checkpoint tensor shape mismatch")
    raw_hex = value["raw_hex"]
    expected_bytes = _prod(expected_shape) * 4 if expected_shape else 4
    if type(raw_hex) is not str or len(raw_hex) != 2 * expected_bytes:
        raise TrainerContractError("checkpoint tensor raw length mismatch")
    try:
        raw = bytes.fromhex(raw_hex)
    except ValueError as exc:
        raise TrainerContractError("checkpoint tensor raw is not hex") from exc
    if raw.hex() != raw_hex:
        raise TrainerContractError("checkpoint tensor raw hex is not lowercase canonical")
    probe = _frombuffer(bytearray(raw), dtype=_float32)
    if not bool(_isfinite(probe).all().item()):
        raise TrainerContractError("checkpoint tensor contains nonfinite values")
    return raw


def _require_nonnegative_float32_bytes(
    raw: bytes,
    label: str,
    _frombuffer: Callable[..., Tensor] = torch.frombuffer,
    _float32: torch.dtype = torch.float32,
    _signbit: Callable[[Tensor], Tensor] = torch.signbit,
) -> None:
    if type(raw) is not bytes or len(raw) % 4 != 0:
        raise TrainerContractError(f"{label} has an invalid float32 byte extent")
    probe = _frombuffer(bytearray(raw), dtype=_float32)
    if bool(_signbit(probe).any().item()):
        raise TrainerContractError(
            f"{label} must be nonnegative and encode zero as exact +0"
        )


@dataclass(frozen=True, slots=True)
class _ParsedResumeState:
    next_global_step: int
    parameter_raw: tuple[bytes, ...]
    optimizer_raw: tuple[tuple[bytes, bytes, bytes] | None, ...]
    rng_state: bytes


def _parse_resume_sections(
    sections: tuple[tuple[str, bytes], ...],
    admission: _AdmissionRecord,
    _parse: Callable[[bytes, str], dict[str, object]] = _parse_json,
    _uint: Callable[..., int] = _exact_uint,
    _parse_tensor: Callable[..., bytes] = _parse_tensor_json,
    _require_nonnegative: Callable[[bytes, str], None] = (
        _require_nonnegative_float32_bytes
    ),
    _groups: Callable[[_AdmissionRecord], list[dict[str, object]]] = _group_payloads,
    _frombuffer: Callable[..., Tensor] = torch.frombuffer,
    _float32: torch.dtype = torch.float32,
    _uint8: torch.dtype = torch.uint8,
    _rng_bytes: Callable[[], bytes] = _rng_state_bytes,
    _generator_ctor: Callable[..., torch.Generator] = torch.Generator,
    _generator_set_state: Callable[[torch.Generator, Tensor], torch.Generator] = (
        torch.Generator.set_state
    ),
    _generator_get_state: Callable[[torch.Generator], Tensor] = torch.Generator.get_state,
    _tensor_clone: Callable[..., Tensor] = Tensor.clone,
    _tensor_equal: Callable[[Tensor, Tensor], bool] = torch.equal,
) -> _ParsedResumeState:
    if type(sections) is not tuple or tuple(name for name, _ in sections) != (
        "dropout_state",
        "model_state",
        "optimizer_state",
        "rng_state",
        "sampler_state",
        "validation_state",
    ):
        raise TrainerContractError("checkpoint section census/order mismatch")
    by_name = dict(sections)
    model_payload = _parse(by_name["model_state"], "model_state")
    if set(model_payload) != {"next_global_step", "parameters", "schema"} or model_payload[
        "schema"
    ] != "phasepair-cpu-synthetic-model-state-v1":
        raise TrainerContractError("model_state schema/key mismatch")
    next_step = _uint(model_payload["next_global_step"], "next_global_step", bits=32)
    parameter_rows = model_payload["parameters"]
    if type(parameter_rows) is not list or len(parameter_rows) != len(admission.rows):
        raise TrainerContractError("model_state parameter census mismatch")
    parameter_raw = tuple(
        _parse_tensor(
            row,
            expected_name=name,
            expected_shape=tuple(parameter.shape),
        )
        for row, (name, parameter) in zip(parameter_rows, admission.rows, strict=True)
    )

    optimizer_payload = _parse(by_name["optimizer_state"], "optimizer_state")
    if set(optimizer_payload) != {"groups", "parameter_states", "schema"} or optimizer_payload[
        "schema"
    ] != "phasepair-cpu-synthetic-adamw-state-v1":
        raise TrainerContractError("optimizer_state schema/key mismatch")
    expected_groups = _groups(admission)
    if optimizer_payload["groups"] != expected_groups:
        raise TrainerContractError("optimizer_state exact group/hyperparameter mismatch")
    state_rows = optimizer_payload["parameter_states"]
    if type(state_rows) is not list or len(state_rows) != len(admission.rows):
        raise TrainerContractError("optimizer_state parameter census mismatch")
    optimizer_raw: list[tuple[bytes, bytes, bytes] | None] = []
    for row, (name, parameter) in zip(state_rows, admission.rows, strict=True):
        if type(row) is not dict or set(row) != {"name", "state"} or row["name"] != name:
            raise TrainerContractError("optimizer parameter state name/key mismatch")
        state = row["state"]
        if next_step == 0:
            if state is not None:
                raise TrainerContractError("step-zero checkpoint must have empty AdamW state")
            optimizer_raw.append(None)
            continue
        if type(state) is not dict or set(state) != {"step", "exp_avg", "exp_avg_sq"}:
            raise TrainerContractError("nonzero checkpoint AdamW state census mismatch")
        step_raw = _parse_tensor(
            state["step"],
            expected_name=f"{name}/step",
            expected_shape=(),
        )
        step_value = float(_frombuffer(bytearray(step_raw), dtype=_float32)[0].item())
        if step_value != float(next_step):
            raise TrainerContractError("AdamW per-parameter step mismatch")
        exp_avg = _parse_tensor(
            state["exp_avg"],
            expected_name=f"{name}/exp_avg",
            expected_shape=tuple(parameter.shape),
        )
        exp_avg_sq = _parse_tensor(
            state["exp_avg_sq"],
            expected_name=f"{name}/exp_avg_sq",
            expected_shape=tuple(parameter.shape),
        )
        _require_nonnegative(exp_avg_sq, f"{name}/exp_avg_sq")
        optimizer_raw.append((step_raw, exp_avg, exp_avg_sq))
    rng_state = by_name["rng_state"]
    current_rng = _rng_bytes()
    if type(rng_state) is not bytes or len(rng_state) != len(current_rng):
        raise TrainerContractError("CPU RNG checkpoint length mismatch")
    rng_tensor = _tensor_clone(_frombuffer(bytearray(rng_state), dtype=_uint8))
    try:
        probe_generator = _generator_ctor(device="cpu")
        _generator_set_state(probe_generator, rng_tensor)
    except (RuntimeError, ValueError) as exc:
        raise TrainerContractError("CPU RNG checkpoint state is invalid") from exc
    if not _tensor_equal(_generator_get_state(probe_generator), rng_tensor):
        raise TrainerContractError("CPU RNG checkpoint state roundtrip mismatch")
    return _ParsedResumeState(next_step, parameter_raw, tuple(optimizer_raw), rng_state)


def _resume_cpu_synthetic_training_session_impl(
    admission: SyntheticOptimizerAdmission,
    validated_checkpoint: object,
    _open_session: Callable[
        [SyntheticOptimizerAdmission], SyntheticTrainingSession
    ] = open_cpu_synthetic_training_session,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _admission_get: Callable[[object], object | None] = _ADMISSION_GET,
    _state_get: Callable[[object], object | None] = _ADMISSION_STATE_GET,
    _state_set: Callable[[object, object], None] = _ADMISSION_STATE_SET,
    _session_get: Callable[[object], object | None] = _SESSION_GET,
    _parse_sections: Callable[
        [tuple[tuple[str, bytes], ...], _AdmissionRecord], _ParsedResumeState
    ] = _parse_resume_sections,
    _named: Callable[[nn.Module], tuple[tuple[str, nn.Parameter], ...]] = _named_parameters,
    _binding: Callable[
        [tuple[tuple[str, nn.Parameter], ...]],
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    ] = _parameter_binding,
    _model_hash: Callable[[tuple[tuple[str, nn.Parameter], ...]], bytes] = _model_digest,
    _tensor_raw: Callable[[Tensor], bytes] = _tensor_bytes,
    _get_rng_state: Callable[[], Tensor] = torch.get_rng_state,
    _set_rng_state: Callable[[Tensor], None] = torch.set_rng_state,
    _no_grad: Callable[..., object] = torch.no_grad,
    _frombuffer: Callable[..., Tensor] = torch.frombuffer,
    _float32: torch.dtype = torch.float32,
    _uint8: torch.dtype = torch.uint8,
    _tensor_copy: Callable[..., Tensor] = Tensor.copy_,
    _tensor_clone: Callable[..., Tensor] = Tensor.clone,
    _model_state: Callable[[_SessionRecord], bytes] = _model_state_bytes,
    _optimizer_state: Callable[[_SessionRecord], bytes] = _optimizer_state_bytes,
    _rng_bytes: Callable[[], bytes] = _rng_state_bytes,
    _optimizer_hash: Callable[..., bytes] = _optimizer_digest,
    _checkpoint_bindings: Callable[[object], object] = _checkpoint_module.checkpoint_bindings,
    _checkpoint_sections: Callable[[object], tuple[tuple[str, bytes], ...]] = (
        _checkpoint_module.checkpoint_sections
    ),
    _retire: Callable[[_SessionRecord], None] = _retire_session_record,
    _admission_type: type[SyntheticOptimizerAdmission] = SyntheticOptimizerAdmission,
) -> SyntheticTrainingSession:
    """Validate the complete envelope before AdamW allocation, then restore exactly."""

    if type(admission) is not _admission_type:
        raise TypeError("admission must be exactly SyntheticOptimizerAdmission")

    with _lock:
        admission_record = _admission_get(admission)
        state = _state_get(admission)
        if admission_record is None or state is None:
            raise TrainerContractError("admission was not issued by this module")
        if state != "ISSUED":
            raise TrainingAdmissionBurnedError("admission is terminal")
        _state_set(admission, "RESUME_VALIDATING")
    try:
        bindings = _checkpoint_bindings(validated_checkpoint)
        sections = _checkpoint_sections(validated_checkpoint)
        if (
            bindings.run_id != admission_record.run_id
            or bindings.attempt_id != admission_record.attempt_id
            or bindings.system_id != admission_record.system_id
            or bindings.seed != admission_record.seed
        ):
            raise TrainerContractError("checkpoint run/admission binding mismatch")
        parsed = _parse_sections(sections, admission_record)
        if parsed.next_global_step != bindings.global_step:
            raise TrainerContractError("checkpoint next/global step mismatch")
        current_rows = _named(admission_record.model)
        current_object_ids, current_storage_intervals = _binding(current_rows)
        if (
            tuple(name for name, _ in current_rows)
            != tuple(name for name, _ in admission_record.rows)
            or current_object_ids != admission_record.parameter_object_ids
            or current_storage_intervals
            != admission_record.parameter_storage_intervals
            or _model_hash(current_rows) != admission_record.issued_model_sha256
        ):
            raise TrainerContractError("resume target changed after admission")
    except BaseException:
        with _lock:
            _state_set(admission, "BURNED")
        raise

    before_parameter_bytes = tuple(
        _tensor_raw(parameter) for _, parameter in admission_record.rows
    )
    before_rng = _tensor_clone(_get_rng_state())
    session: SyntheticTrainingSession | None = None
    try:
        with _lock:
            _state_set(admission, "ISSUED")
            session = _open_session(admission)
        session_record = _session_get(session)
        if session_record is None:
            raise TrainerContractError("resumed session registry entry is absent")
        with _no_grad():
            for (_, parameter), raw in zip(
                admission_record.rows,
                parsed.parameter_raw,
                strict=True,
            ):
                restored = _frombuffer(bytearray(raw), dtype=_float32).reshape(
                    tuple(parameter.shape)
                )
                _tensor_copy(parameter, restored)
        if parsed.next_global_step > 0:
            for (_, parameter), state_raw in zip(
                admission_record.rows,
                parsed.optimizer_raw,
                strict=True,
            ):
                if state_raw is None:
                    raise TrainerContractError("nonzero resume state unexpectedly absent")
                step_raw, exp_avg_raw, exp_avg_sq_raw = state_raw
                session_record.optimizer.state[parameter] = {
                    "step": _tensor_clone(
                        _frombuffer(bytearray(step_raw), dtype=_float32).reshape(())
                    ),
                    "exp_avg": _tensor_clone(
                        _frombuffer(bytearray(exp_avg_raw), dtype=_float32).reshape(
                            tuple(parameter.shape)
                        )
                    ),
                    "exp_avg_sq": _tensor_clone(
                        _frombuffer(bytearray(exp_avg_sq_raw), dtype=_float32).reshape(
                            tuple(parameter.shape)
                        )
                    ),
                }
        session_record.next_global_step = parsed.next_global_step
        rng_tensor = _tensor_clone(
            _frombuffer(bytearray(parsed.rng_state), dtype=_uint8)
        )
        _set_rng_state(rng_tensor)
        if (
            _model_state(session_record) != dict(sections)["model_state"]
            or _optimizer_state(session_record) != dict(sections)["optimizer_state"]
            or _rng_bytes() != parsed.rng_state
        ):
            raise TrainerContractError("restored session postcondition mismatch")
        restored_rows = _named(admission_record.model)
        restored_object_ids, restored_storage_intervals = _binding(restored_rows)
        if (
            tuple(name for name, _ in restored_rows)
            != tuple(name for name, _ in admission_record.rows)
            or restored_object_ids != admission_record.parameter_object_ids
            or restored_storage_intervals
            != admission_record.parameter_storage_intervals
        ):
            raise TrainerContractError(
                "resume target object/name/storage binding changed during restore"
            )
        session_record.expected_model_sha256 = _model_hash(restored_rows)
        session_record.expected_optimizer_sha256 = _optimizer_hash(
            session_record.optimizer,
            restored_rows,
            admission_record.decay_names,
            admission_record.no_decay_names,
        )
        return session
    except BaseException as primary:
        cleanup_failure: BaseException | None = None
        try:
            with _no_grad():
                for (_, parameter), raw in zip(
                    admission_record.rows,
                    before_parameter_bytes,
                    strict=True,
                ):
                    restored = _frombuffer(
                        bytearray(raw), dtype=_float32
                    ).reshape(tuple(parameter.shape))
                    _tensor_copy(parameter, restored)
            _set_rng_state(before_rng)
            if tuple(
                _tensor_raw(parameter) for _, parameter in admission_record.rows
            ) != before_parameter_bytes:
                raise TrainerContractError("resume rollback parameter verification failed")
        except BaseException as cleanup:
            cleanup_failure = cleanup
        if session is not None:
            failed_record = _session_get(session)
            if failed_record is not None:
                _retire(failed_record)
        if cleanup_failure is not None:
            raise TrainingTransactionError("resume failed and rollback verification failed") from cleanup_failure
        raise TrainingTransactionError("resume failed; target restored and attempt retired") from primary


def _bind_resume_session(
    implementation: Callable[
        [SyntheticOptimizerAdmission, object], SyntheticTrainingSession
    ],
) -> Callable[[SyntheticOptimizerAdmission, object], SyntheticTrainingSession]:
    def resume_cpu_synthetic_training_session(
        admission: SyntheticOptimizerAdmission,
        validated_checkpoint: object,
    ) -> SyntheticTrainingSession:
        return implementation(admission, validated_checkpoint)

    return resume_cpu_synthetic_training_session


resume_cpu_synthetic_training_session = _bind_resume_session(
    _resume_cpu_synthetic_training_session_impl
)
del _bind_resume_session


def _run_cpu_synthetic_optimizer_step_impl(
    session: SyntheticTrainingSession,
    loss: Tensor,
    *,
    epoch_index: int,
    global_step: int,
    batch_position: int,
    dropout_position: int,
    _uint: Callable[..., int] = _exact_uint,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _session_get: Callable[[object], object | None] = _SESSION_GET,
    _tensor_raw: Callable[[Tensor], bytes] = _tensor_bytes,
    _model_hash: Callable[[tuple[tuple[str, nn.Parameter], ...]], bytes] = _model_digest,
    _optimizer_hash: Callable[..., bytes] = _optimizer_digest,
    _deepcopy: Callable[[object], object] = copy.deepcopy,
    _optimizer_state_dict: Callable[..., dict[object, object]] = torch.optim.Optimizer.state_dict,
    _optimizer_zero_grad: Callable[..., None] = torch.optim.Optimizer.zero_grad,
    _loss_backward: Callable[..., None] = Tensor.backward,
    _clip_grad_norm: Callable[..., Tensor] = torch.nn.utils.clip_grad_norm_,
    _adamw_step: Callable[..., object] = torch.optim.AdamW.step,
    _no_grad: Callable[..., object] = torch.no_grad,
    _frombuffer: Callable[..., Tensor] = torch.frombuffer,
    _tensor_copy: Callable[..., Tensor] = Tensor.copy_,
    _optimizer_load_state: Callable[..., None] = torch.optim.Optimizer.load_state_dict,
    _get_cpu_rng_state: Callable[[], Tensor] = torch.get_rng_state,
    _set_cpu_rng_state: Callable[[Tensor], None] = torch.set_rng_state,
    _tensor_clone: Callable[..., Tensor] = Tensor.clone,
    _tensor_equal: Callable[[Tensor, Tensor], bool] = torch.equal,
    _receipt_encoder: Callable[[dict[str, object]], bytes] = _receipt_bytes,
    _receipt_set: Callable[[object, object], None] = _RECEIPT_SET,
    _named: Callable[[nn.Module], tuple[tuple[str, nn.Parameter], ...]] = _named_parameters,
    _binding: Callable[
        [tuple[tuple[str, nn.Parameter], ...]],
        tuple[tuple[int, ...], tuple[tuple[int, int, int], ...]],
    ] = _parameter_binding,
    _retire: Callable[[_SessionRecord], None] = _retire_session_record,
    _session_type: type[SyntheticTrainingSession] = SyntheticTrainingSession,
    _receipt_type: type[StepReceipt] = StepReceipt,
    _tensor_type: type[Tensor] = Tensor,
    _float32: torch.dtype = torch.float32,
    _float64: torch.dtype = torch.float64,
    _isfinite: Callable[[Tensor], Tensor] = torch.isfinite,
    _all: Callable[..., Tensor] = torch.all,
    _sum: Callable[..., Tensor] = torch.sum,
    _tensor_detach: Callable[..., Tensor] = Tensor.detach,
    _tensor_reshape: Callable[..., Tensor] = Tensor.reshape,
    _tensor_is_contiguous: Callable[..., bool] = Tensor.is_contiguous,
    _tensor_to: Callable[..., Tensor] = Tensor.to,
    _tensor_item: Callable[..., object] = Tensor.item,
    _sqrt: Callable[[float], float] = math.sqrt,
    _math_isfinite: Callable[[float], bool] = math.isfinite,
    _object_new: Callable[[type[object]], object] = object.__new__,
) -> StepReceipt:
    """Commit one real autograd/AdamW step or roll back and retire exactly."""

    if type(session) is not _session_type:
        raise TypeError("session must be exactly SyntheticTrainingSession")
    if type(loss) is not _tensor_type:
        raise TypeError("loss must be an exact base torch.Tensor")
    epoch = _uint(epoch_index, "epoch_index", bits=32)
    step = _uint(global_step, "global_step", bits=32)
    batch = _uint(batch_position, "batch_position", bits=32)
    dropout = _uint(dropout_position, "dropout_position", bits=32)
    with _lock:
        record = _session_get(session)
    if record is None:
        raise TrainerContractError("session was not issued by this module")
    if not record.lock.acquire(blocking=False):
        raise TrainingAdmissionBurnedError("session is already executing")
    try:
        if record.state != "ACTIVE":
            raise TrainingAdmissionBurnedError("session is terminal")
        if step != record.next_global_step:
            _retire(record)
            raise TrainerContractError("global_step is not the exact next step")
        if epoch >= 30:
            _retire(record)
            raise TrainerContractError("epoch_index must be in [0,29]")
        detached_loss = _tensor_detach(loss)
        if (
            loss.ndim != 0
            or loss.dtype != _float32
            or loss.device.type != "cpu"
            or not loss.requires_grad
            or not bool(_tensor_item(_isfinite(detached_loss)))
        ):
            _retire(record)
            raise TrainerContractError("loss must be a finite differentiable CPU float32 scalar")
        loss_for_backward = _tensor_reshape(loss, ())

        current_rows = _named(record.admission.model, allow_grad=True)
        current_object_ids, current_storage_intervals = _binding(current_rows)
        if (
            tuple(name for name, _ in current_rows)
            != tuple(name for name, _ in record.admission.rows)
            or current_object_ids != record.admission.parameter_object_ids
            or current_storage_intervals
            != record.admission.parameter_storage_intervals
        ):
            _retire(record)
            raise TrainerContractError(
                "live parameter object/name/storage binding drift before optimizer step"
            )
        current_model_sha256 = _model_hash(current_rows)
        current_optimizer_sha256 = _optimizer_hash(
            record.optimizer,
            current_rows,
            record.admission.decay_names,
            record.admission.no_decay_names,
        )
        if (
            current_model_sha256 != record.expected_model_sha256
            or current_optimizer_sha256 != record.expected_optimizer_sha256
        ):
            _retire(record)
            raise TrainerContractError("live model/optimizer state drift before optimizer step")
        for name, parameter in current_rows:
            if parameter.grad is not None:
                _retire(record)
                raise TrainerContractError(
                    f"stale gradient at optimizer-step boundary: {name}"
                )
        parameters = tuple(parameter for _, parameter in current_rows)
        before_parameter_bytes = tuple(_tensor_raw(parameter) for parameter in parameters)
        before_model_sha256 = current_model_sha256
        before_optimizer_state = _deepcopy(_optimizer_state_dict(record.optimizer))
        before_cpu_rng_state = _tensor_clone(_get_cpu_rng_state())
        before_optimizer_sha256 = _optimizer_hash(
            record.optimizer,
            record.admission.rows,
            record.admission.decay_names,
            record.admission.no_decay_names,
        )
        try:
            _optimizer_zero_grad(record.optimizer, set_to_none=True)
            _loss_backward(loss_for_backward)
            squared_norm = 0.0
            for name, parameter in record.admission.rows:
                gradient = parameter.grad
                if type(gradient) is not _tensor_type:
                    raise TrainerContractError(f"missing exact gradient: {name}")
                if (
                    gradient.dtype != _float32
                    or gradient.device.type != "cpu"
                    or not _tensor_is_contiguous(gradient)
                    or not bool(_tensor_item(_all(_isfinite(gradient))))
                ):
                    raise TrainerContractError(f"gradient invariant mismatch: {name}")
                gradient64 = _tensor_to(
                    _tensor_detach(gradient),
                    dtype=_float64,
                )
                squared_norm += float(_tensor_item(_sum(gradient64 ** 2)))
            raw_norm = _sqrt(squared_norm)
            if not _math_isfinite(raw_norm):
                raise TrainerContractError("global gradient norm is nonfinite")
            reported_norm = _clip_grad_norm(
                parameters,
                max_norm=1.0,
                norm_type=2.0,
                error_if_nonfinite=True,
                foreach=False,
            )
            if not _math_isfinite(float(_tensor_item(reported_norm))):
                raise TrainerContractError("clip_grad_norm returned nonfinite")
            _adamw_step(record.optimizer)
            after_parameter_bytes = tuple(_tensor_raw(parameter) for parameter in parameters)
            if before_parameter_bytes == after_parameter_bytes:
                raise TrainerContractError("AdamW step made no parameter-byte progress")
            post_rows = _named(record.admission.model, allow_grad=True)
            post_object_ids, post_storage_intervals = _binding(post_rows)
            if (
                tuple(name for name, _ in post_rows)
                != tuple(name for name, _ in record.admission.rows)
                or post_object_ids != record.admission.parameter_object_ids
                or post_storage_intervals
                != record.admission.parameter_storage_intervals
            ):
                raise TrainerContractError(
                    "live parameter object/name/storage binding drift during optimizer step"
                )
            for name, parameter in post_rows:
                if not bool(
                    _tensor_item(_all(_isfinite(_tensor_detach(parameter))))
                ):
                    raise TrainerContractError(f"nonfinite parameter after AdamW: {name}")
            after_model_sha256 = _model_hash(post_rows)
            after_optimizer_sha256 = _optimizer_hash(
                record.optimizer,
                post_rows,
                record.admission.decay_names,
                record.admission.no_decay_names,
            )
            if before_optimizer_sha256 == after_optimizer_sha256:
                raise TrainerContractError("AdamW state digest did not change")
            _optimizer_zero_grad(record.optimizer, set_to_none=True)
            if any(parameter.grad is not None for parameter in parameters):
                raise TrainerContractError(
                    "successful optimizer step did not restore the grad=None boundary"
                )
            payload = {
                "attempt_id": record.admission.attempt_id,
                "authority": 0,
                "batch_position": batch,
                "dropout_position": dropout,
                "epoch_index": epoch,
                "global_step": step,
                "gradient_clip_norm": "1.0",
                "gradient_norm_hex": float(raw_norm).hex(),
                "loss_hex": float(_tensor_item(detached_loss)).hex(),
                "model_after_sha256": after_model_sha256.hex(),
                "model_before_sha256": before_model_sha256.hex(),
                "optimizer_after_sha256": after_optimizer_sha256.hex(),
                "optimizer_before_sha256": before_optimizer_sha256.hex(),
                "production": False,
                "result_claimed": False,
                "run_id": record.admission.run_id,
                "schema": "phasepair-cpu-synthetic-step-receipt-v1",
                "seed": record.admission.seed,
                "status": "CPU_SYNTHETIC_STEP_COMMITTED_AUTHORITY0_NO_RESULT",
                "system_id": record.admission.system_id,
            }
            raw = _receipt_encoder(payload)
            receipt = _object_new(_receipt_type)
            with _lock:
                _receipt_set(receipt, raw)
            record.next_global_step += 1
            record.expected_model_sha256 = after_model_sha256
            record.expected_optimizer_sha256 = after_optimizer_sha256
            return receipt
        except BaseException as primary:
            rollback_failure: BaseException | None = None
            try:
                with _no_grad():
                    for parameter, raw in zip(parameters, before_parameter_bytes, strict=True):
                        restored = _frombuffer(bytearray(raw), dtype=_float32).reshape(
                            tuple(parameter.shape)
                        )
                        _tensor_copy(parameter, restored)
                _optimizer_load_state(record.optimizer, before_optimizer_state)
                _optimizer_zero_grad(record.optimizer, set_to_none=True)
                _set_cpu_rng_state(before_cpu_rng_state)
                if any(parameter.grad is not None for parameter in parameters):
                    raise TrainerContractError(
                        "gradient rollback did not restore the grad=None boundary"
                    )
                if tuple(_tensor_raw(parameter) for parameter in parameters) != before_parameter_bytes:
                    raise TrainerContractError("parameter rollback byte verification failed")
                if _optimizer_hash(
                    record.optimizer,
                    record.admission.rows,
                    record.admission.decay_names,
                    record.admission.no_decay_names,
                ) != before_optimizer_sha256:
                    raise TrainerContractError("optimizer rollback digest verification failed")
                if not _tensor_equal(_get_cpu_rng_state(), before_cpu_rng_state):
                    raise TrainerContractError("CPU RNG rollback verification failed")
            except BaseException as cleanup:
                rollback_failure = cleanup
            _retire(record)
            if rollback_failure is not None:
                raise TrainingTransactionError("step failed and rollback verification failed") from rollback_failure
            raise TrainingTransactionError("step failed; exact pre-step state restored; session retired") from primary
    except BaseException:
        if record.state == "ACTIVE":
            _retire(record)
        raise
    finally:
        record.lock.release()


def _bind_run_step(
    implementation: Callable[..., StepReceipt],
) -> Callable[..., StepReceipt]:
    def run_cpu_synthetic_optimizer_step(
        session: SyntheticTrainingSession,
        loss: Tensor,
        *,
        epoch_index: int,
        global_step: int,
        batch_position: int,
        dropout_position: int,
    ) -> StepReceipt:
        return implementation(
            session,
            loss,
            epoch_index=epoch_index,
            global_step=global_step,
            batch_position=batch_position,
            dropout_position=dropout_position,
        )

    return run_cpu_synthetic_optimizer_step


run_cpu_synthetic_optimizer_step = _bind_run_step(
    _run_cpu_synthetic_optimizer_step_impl
)
del _bind_run_step


def _canonical_step_receipt_bytes_impl(
    receipt: StepReceipt,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _receipt_get: Callable[[object], object | None] = _RECEIPT_GET,
    _loads: Callable[[str], object] = json.loads,
    _encoder: Callable[[dict[str, object]], bytes] = _receipt_bytes,
    _receipt_type: type[StepReceipt] = StepReceipt,
) -> bytes:
    if type(receipt) is not _receipt_type:
        raise TypeError("receipt must be exactly StepReceipt")
    with _lock:
        raw = _receipt_get(receipt)
    if raw is None:
        raise TrainerContractError("receipt was not issued by this module")
    parsed = _loads(raw.decode("ascii"))
    if type(parsed) is not dict:
        raise TrainerContractError("stored receipt JSON root is not an object")
    rebuilt = _encoder(parsed)
    if rebuilt != raw:
        raise TrainerContractError("stored receipt is not canonical")
    return raw


def _bind_receipt_reader(
    implementation: Callable[[StepReceipt], bytes],
) -> Callable[[StepReceipt], bytes]:
    def canonical_step_receipt_bytes(receipt: StepReceipt) -> bytes:
        return implementation(receipt)

    return canonical_step_receipt_bytes


canonical_step_receipt_bytes = _bind_receipt_reader(
    _canonical_step_receipt_bytes_impl
)
del _bind_receipt_reader


def _bind_receipt_digest(
    reader: Callable[[StepReceipt], bytes],
    sha256: Callable[..., object],
) -> Callable[[StepReceipt], bytes]:
    def step_receipt_sha256(receipt: StepReceipt) -> bytes:
        return sha256(reader(receipt)).digest()  # type: ignore[attr-defined]

    return step_receipt_sha256


step_receipt_sha256 = _bind_receipt_digest(
    canonical_step_receipt_bytes,
    hashlib.sha256,
)
del _bind_receipt_digest


def _cpu_synthetic_session_state_impl(
    session: SyntheticTrainingSession,
    _lock: threading.RLock = _REGISTRY_LOCK,
    _session_get: Callable[[object], object | None] = _SESSION_GET,
    _session_type: type[SyntheticTrainingSession] = SyntheticTrainingSession,
) -> str:
    if type(session) is not _session_type:
        raise TypeError("session must be exactly SyntheticTrainingSession")
    with _lock:
        record = _session_get(session)
    if record is None:
        raise TrainerContractError("session was not issued by this module")
    return record.state


def _bind_session_state(
    implementation: Callable[[SyntheticTrainingSession], str],
) -> Callable[[SyntheticTrainingSession], str]:
    def cpu_synthetic_session_state(session: SyntheticTrainingSession) -> str:
        return implementation(session)

    return cpu_synthetic_session_state


cpu_synthetic_session_state = _bind_session_state(
    _cpu_synthetic_session_state_impl
)
del _bind_session_state

# Retire the module-visible registry objects after every public closure has
# captured the actual stores.  Rebinding or mutating these decoys cannot mint
# admissions, sessions, or receipts.
_ADMISSIONS = WeakKeyDictionary()
_ADMISSION_STATES = WeakKeyDictionary()
_SESSIONS = WeakKeyDictionary()
_RECEIPTS = WeakKeyDictionary()
_REGISTRY_LOCK = threading.RLock()


__all__ = [
    "ADMISSION_STATUS",
    "BETAS",
    "DECAY_WEIGHT_DECAY",
    "EPSILON",
    "GRAD_CLIP_NORM",
    "LEARNING_RATE",
    "NO_DECAY_WEIGHT_DECAY",
    "STATUS",
    "STEP_STATUS",
    "StepReceipt",
    "SyntheticOptimizerAdmission",
    "SyntheticTrainingSession",
    "TrainerContractError",
    "TrainingAdmissionBurnedError",
    "TrainingTransactionError",
    "canonical_step_receipt_bytes",
    "cpu_synthetic_admission_status",
    "cpu_synthetic_optimizer_constructor_count",
    "cpu_synthetic_session_state",
    "export_cpu_synthetic_checkpoint_sections",
    "issue_cpu_synthetic_admission",
    "open_cpu_synthetic_training_session",
    "resume_cpu_synthetic_training_session",
    "run_cpu_synthetic_optimizer_step",
    "step_receipt_sha256",
]
