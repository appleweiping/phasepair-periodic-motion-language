"""Sealed local-CPU CLIP text bridge for PhasePair.

This module promotes the exact candidate-v2 storage materialization and live
registry checks into an executable, offline bridge.  It deliberately stops at
an authority-zero optimizer *preflight*: no training-authority issuer exists,
and this module never calls :class:`torch.optim.AdamW`.

The public resolver accepts paths, never a caller-supplied model, state dict,
parameter row, or parameter mapping.  The loaded tokenizer, text model, and
live Parameter provenance remain reachable only through a weak-key registry
owned by an opaque one-shot lease.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import http.client
import inspect
import json
import os
from pathlib import Path
import platform
import socket
import stat
import sys
import tempfile
import threading
from typing import Any, Iterator
import urllib.request
import warnings
import weakref

from phasepair_core import clip_resolution, contracts, initialization, readiness


STATUS = "VERIFIED_LOCAL_CPU_CANDIDATE_AUTHORITY0"
OPTIMIZER_STATUS = "HOLD_NO_TRAINING_AUTHORITY_ISSUER"
AUTHORITY = 0
PRODUCTION = False
TRAINING_AUTHORIZED = False

ISSUED = 0
VALIDATING = 1
BURNED = 3

_MODEL_ID = "openai/clip-vit-base-patch32"
_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
_SOURCE_CHECKPOINT_SHA256 = bytes.fromhex(
    "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"
)
_EXPECTED_SNAPSHOT_ASSESSMENT_SHA256 = (
    "339b98e00a1b3b777b14add0ed1fc2e9bce37fcc4e8084dd1c09acdeab5848e1"
)
_EXPECTED_PREFLIGHT_NAME = (
    "local_py312_cpu_text_preflight_candidate_v2_20260825_020011.json"
)
_EXPECTED_PREFLIGHT_BYTES = 7_925
_EXPECTED_PREFLIGHT_SHA256 = (
    "23e56043ef67fea7d1e3cd766028730fa6f7716289be572203a2c1fe2b59f7f9"
)
_EXPECTED_PREFLIGHT_CANONICAL_SHA256 = (
    "624b9434f7a1b525df940d0d1cbe3040f3b0cd4d9361c459d9b115052d37a89b"
)
_EXPECTED_PRIVATE_RUNTIME_NAME = (
    "phasepair_local_py312_cpu_wheelhouse_private_successor_v3_"
    "20260825_020011.json"
)
_EXPECTED_PRIVATE_RUNTIME_BYTES = 5_447
_EXPECTED_PRIVATE_RUNTIME_SHA256 = (
    "5272b80449d37195aa6ad96aedf65e9cf0b25e801f5daafd2ff5d1294cd83d0e"
)
_EXPECTED_PREFLIGHT_SCRIPT_BYTES = 28_150
_EXPECTED_PREFLIGHT_SCRIPT_SHA256 = (
    "3feab11d5f51e2e1a82001e2800ebcff5481cce35bf6add0db3785ef986b4216"
)
_EXPECTED_RIGHTS_BYTES = 9_521
_EXPECTED_RIGHTS_SHA256 = (
    "504e4be2d8125990a96549730f55e178537babfa22f7be47518530698dab433d"
)
_EXPECTED_PARAMETER_MANIFEST_SHA256 = (
    "4a2d1dde9f92f2eee6c17019b58fcc9636529f677fd9bc8b580992bdba3136eb"
)
_EXPECTED_BUFFER_MANIFEST_SHA256 = (
    "267c15fa50192124509fea524557e418f97e6cf877f5819a4c0483dab9c86fdd"
)
_EXPECTED_STATE_KEYS_SHA256 = (
    "0c135012e876b7ebc7e36287434cb5f24cbcd1c2dd9e507b3135a33f5789dacc"
)
_EXPECTED_TEXT_ROWS = 196
_EXPECTED_TEXT_NUMEL = 63_165_952
_EXPECTED_TEXT_DECAY = (74, 63_085_056)
_EXPECTED_TEXT_NO_DECAY = (122, 80_896)

_CAPTIONS = (
    "A person walks in a circle.",
    "Two people clap in rhythm.",
    "A hand moves left and then right.",
)
_EXPECTED_CONFIG = {
    "attention_dropout": 0.0,
    "bos_token_id": 0,
    "eos_token_id": 2,
    "hidden_act": "quick_gelu",
    "hidden_size": 512,
    "intermediate_size": 2048,
    "layer_norm_eps": 1e-05,
    "max_position_embeddings": 77,
    "model_type": "clip_text_model",
    "num_attention_heads": 8,
    "num_hidden_layers": 12,
    "pad_token_id": 1,
    "projection_dim": 512,
    "vocab_size": 49_408,
}
_EXPECTED_GOLDENS = {
    "attention_mask_sha256": (
        "764c9a7807146b3dd2c20deae73cffe92db80069861a45d3ec3f0863c6b8f34e"
    ),
    "input_ids_sha256": (
        "2c1bc6af4fe563c4b2dee0bc78bcd8bf2868ad3bd8f74d4126329f43a280cc8a"
    ),
    "normalized_float64_l2_sha256": (
        "26df05351d13d0e66048bb6f63c3fc84b62aced5d4c765cc6b18d4606201ee97"
    ),
    "pooled_eos_sha256": (
        "07acbb2f6d1c803f12dfd937913d4f2165ca0ed08e248baef2a725a6e3c99a1a"
    ),
    "pretrained_projection_sha256": (
        "a4ec38be61e2db328652bd1119d4e8aba3f218a594562d0eb0060f07cf6e0f61"
    ),
}
_EXPECTED_RUNTIME = {
    "huggingface_hub": "0.36.0",
    "numpy": "2.4.6",
    "python": "3.12.0",
    "python_implementation": "CPython",
    "tokenizers": "0.22.2",
    "torch": "2.10.0+cpu",
    "transformers": "4.57.3",
}
_EXPECTED_SOURCE = {
    "modeling_source_bytes": 50_315,
    "modeling_source_sha256": (
        "3e779a99827618d6e6d57583ec4f5c0d765321dfa5054a4ffca6a56bb004d499"
    ),
    "tokenizer_source_bytes": 6_766,
    "tokenizer_source_sha256": (
        "67dff0f21a56d0dc18a45f8764f3fafa02ee2b0aeafeb70761c9623f9114dd53"
    ),
}


class LiveClipBridgeError(ValueError):
    """A closed live-bridge invariant was violated."""


class LiveClipBridgeHold(LiveClipBridgeError):
    """A live candidate cannot be promoted or consumed."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("hold code must be an exact nonempty built-in str")
        self.code = code
        super().__init__(code)


class VerifiedLocalClipTextLease:
    """Opaque one-shot owner of a verified local text runtime."""

    __slots__ = ("_lock", "_nonce", "_seal", "__weakref__")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("VerifiedLocalClipTextLease construction is internal")


class OptimizerCandidateAssessment:
    """Authority-zero result after complete live two-group validation."""

    __slots__ = (
        "status",
        "authority",
        "production",
        "training_authorized",
        "architecture",
        "full_tensor_count",
        "full_numel",
        "decay_tensor_count",
        "decay_numel",
        "no_decay_tensor_count",
        "no_decay_numel",
        "inventory_sha256",
        "decay_names_sha256",
        "no_decay_names_sha256",
        "parameter_binding_sha256",
        "missing_gates",
        "adamw_constructor_count",
        "__weakref__",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("OptimizerCandidateAssessment construction is internal")


@dataclass(slots=True)
class _RuntimePayload:
    snapshot_root: Path
    snapshot_rows: tuple[tuple[str, int, str], ...]
    snapshot_manifest_sha256: str
    preflight_sha256: str
    private_runtime_sha256: str
    rights_sha256: str
    runtime_identity: tuple[tuple[str, str], ...]
    text_rows: tuple[contracts.ParameterRow, ...]
    frozen_rows: tuple[contracts.ParameterRow, ...]
    text_parameters: tuple[tuple[str, object], ...]
    parameter_manifest_sha256: str
    buffer_manifest_sha256: str
    text_inventory_sha256: str
    text_binding_sha256: str
    model: object
    tokenizer: object


@dataclass(slots=True)
class _LeaseRecord:
    owner_id: int
    state: int
    nonce: int
    lock: object
    seal: bytes
    payload: _RuntimePayload | None


_LOAD_LOCK = threading.RLock()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json(value: object, *, final_lf: bool = True) -> bytes:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return encoded + (b"\n" if final_lf else b"")


def _safe_regular_file(path: Path, *, expected_bytes: int, expected_sha256: str) -> bytes:
    try:
        resolved = path.resolve(strict=True)
        metadata = path.lstat()
    except (OSError, RuntimeError) as exc:
        raise LiveClipBridgeHold("HOLD_REQUIRED_RUNTIME_ARTIFACT_ABSENT") from exc
    attributes = getattr(metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        path != resolved
        or not stat.S_ISREG(metadata.st_mode)
        or bool(attributes & reparse_flag)
        or metadata.st_nlink != 1
        or metadata.st_size != expected_bytes
    ):
        raise LiveClipBridgeHold("HOLD_REQUIRED_RUNTIME_ARTIFACT_IDENTITY")
    raw = resolved.read_bytes()
    if len(raw) != expected_bytes or _sha256_bytes(raw) != expected_sha256:
        raise LiveClipBridgeHold("HOLD_REQUIRED_RUNTIME_ARTIFACT_IDENTITY")
    return raw


def _scan_snapshot(root: Path) -> tuple[tuple[str, int, str], ...]:
    try:
        resolved = root.resolve(strict=True)
        root_metadata = root.lstat()
    except (OSError, RuntimeError) as exc:
        raise LiveClipBridgeHold("HOLD_CLIP_SNAPSHOT_ABSENT") from exc
    attributes = getattr(root_metadata, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    if (
        root != resolved
        or not stat.S_ISDIR(root_metadata.st_mode)
        or bool(attributes & reparse_flag)
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_SNAPSHOT_ROOT_IDENTITY")
    expected = {name: (size, digest) for name, size, digest in clip_resolution.PINNED_FILES}
    try:
        children = tuple(sorted(resolved.iterdir(), key=lambda value: value.name))
    except OSError as exc:
        raise LiveClipBridgeHold("HOLD_CLIP_SNAPSHOT_CENSUS") from exc
    if tuple(child.name for child in children) != tuple(sorted(expected)):
        raise LiveClipBridgeHold("HOLD_CLIP_SNAPSHOT_CENSUS")
    rows: list[tuple[str, int, str]] = []
    for child in children:
        metadata = child.lstat()
        child_attributes = getattr(metadata, "st_file_attributes", 0)
        expected_size, expected_digest = expected[child.name]
        if (
            not stat.S_ISREG(metadata.st_mode)
            or bool(child_attributes & reparse_flag)
            or metadata.st_nlink != 1
            or metadata.st_size != expected_size
            or child.resolve(strict=True).parent != resolved
        ):
            raise LiveClipBridgeHold("HOLD_CLIP_SNAPSHOT_FILE_IDENTITY")
        digest = _sha256_file(child)
        if digest != expected_digest:
            raise LiveClipBridgeHold("HOLD_CLIP_SNAPSHOT_FILE_BYTES")
        rows.append((child.name, expected_size, digest))
    return tuple(rows)


def _snapshot_manifest_sha256(rows: tuple[tuple[str, int, str], ...]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "files": [
                    {"bytes": size, "path": name, "sha256": digest}
                    for name, size, digest in rows
                ],
                "model_id": _MODEL_ID,
                "revision": _REVISION,
                "schema": "phasepair-live-clip-snapshot-v1",
            },
            final_lf=False,
        )
    )


def _validate_candidate_receipt(raw: bytes) -> None:
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_SCHEMA") from exc
    if type(parsed) is not dict:
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_SCHEMA")
    if (
        _canonical_json(parsed, final_lf=False) != raw[:-1]
        or not raw.endswith(b"\n")
        or _sha256_bytes(raw[:-1]) != _EXPECTED_PREFLIGHT_CANONICAL_SHA256
    ):
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_CANONICAL_BYTES")
    checks = parsed.get("checks")
    if (
        parsed.get("schema")
        != "phasepair-local-py312-cpu-text-preflight-candidate-v2"
        or parsed.get("status") != "PASS"
        or parsed.get("authority") != 0
        or parsed.get("production") is not False
        or parsed.get("training_authorized") is not False
        or parsed.get("hold_reasons") != []
        or type(checks) is not dict
        or len(checks) != 39
        or any(value is not True for value in checks.values())
    ):
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_SEMANTICS")
    model = parsed.get("model")
    registry = parsed.get("registry")
    golden = parsed.get("synthetic_golden_candidate")
    runtime = parsed.get("runtime")
    if type(model) is not dict or type(registry) is not dict:
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_SEMANTICS")
    if type(golden) is not dict or type(runtime) is not dict:
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_SEMANTICS")
    expected_model = {
        "loader_class": (
            "transformers.models.clip.modeling_clip."
            "CLIPTextModelWithProjection"
        ),
        "model_id": _MODEL_ID,
        "revision": _REVISION,
        "snapshot_files": 8,
        "snapshot_total_bytes": clip_resolution.TOTAL_BYTES,
        "tokenizer_class": (
            "transformers.models.clip.tokenization_clip_fast.CLIPTokenizerFast"
        ),
    }
    if model != expected_model:
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_MODEL_IDENTITY")
    if (
        registry.get("parameter_row_count") != 197
        or registry.get("parameter_object_identity_count") != 197
        or registry.get("parameter_storage_identity_count") != 197
        or registry.get("parameter_manifest_sha256")
        != _EXPECTED_PARAMETER_MANIFEST_SHA256
        or registry.get("buffer_manifest_sha256")
        != _EXPECTED_BUFFER_MANIFEST_SHA256
        or registry.get("state_keys_sha256") != _EXPECTED_STATE_KEYS_SHA256
        or registry.get("config_fields") != _EXPECTED_CONFIG
        or registry.get("text_trainable_numel") != _EXPECTED_TEXT_NUMEL
    ):
        raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_REGISTRY")
    for key, expected in _EXPECTED_GOLDENS.items():
        if golden.get(key) != expected:
            raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_GOLDEN")
    for key, expected in _EXPECTED_RUNTIME.items():
        if runtime.get(key) != expected:
            raise LiveClipBridgeHold("HOLD_PREFLIGHT_RECEIPT_RUNTIME")


def _validate_private_runtime_receipt(raw: bytes) -> None:
    try:
        parsed = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LiveClipBridgeHold("HOLD_PRIVATE_RUNTIME_RECEIPT_SCHEMA") from exc
    if type(parsed) is not dict:
        raise LiveClipBridgeHold("HOLD_PRIVATE_RUNTIME_RECEIPT_SCHEMA")
    candidate = parsed.get("preflight_candidate_v2")
    boundary = parsed.get("capability_boundary")
    if type(candidate) is not dict or type(boundary) is not dict:
        raise LiveClipBridgeHold("HOLD_PRIVATE_RUNTIME_RECEIPT_SEMANTICS")
    receipt = candidate.get("receipt")
    script = candidate.get("script")
    if type(receipt) is not dict or type(script) is not dict:
        raise LiveClipBridgeHold("HOLD_PRIVATE_RUNTIME_RECEIPT_SEMANTICS")
    if (
        parsed.get("authority") != 0
        or parsed.get("production") is not False
        or parsed.get("training_authorized") is not False
        or candidate.get("schema")
        != "phasepair-local-py312-cpu-text-preflight-candidate-v2"
        or candidate.get("all_39_checks_pass") is not True
        or candidate.get("parameter_row_count") != 197
        or candidate.get("postmaterialization_parameter_manifest_sha256")
        != _EXPECTED_PARAMETER_MANIFEST_SHA256
        or receipt.get("bytes") != _EXPECTED_PREFLIGHT_BYTES
        or receipt.get("sha256") != _EXPECTED_PREFLIGHT_SHA256
        or script.get("bytes") != _EXPECTED_PREFLIGHT_SCRIPT_BYTES
        or script.get("sha256") != _EXPECTED_PREFLIGHT_SCRIPT_SHA256
        or boundary.get("production_live_runtime")
        != (
            "HOLD_FRESH_BRIDGE_PROMOTION_OF_STORAGE_MATERIALIZATION_AND_"
            "DIRECT_REGISTRY_CONTRACT"
        )
        or boundary.get("optimizer_constructed") is not False
        or boundary.get("training_performed") is not False
    ):
        raise LiveClipBridgeHold("HOLD_PRIVATE_RUNTIME_RECEIPT_SEMANTICS")


def _validate_runtime_environment(wheelhouse_root: Path, modules: dict[str, object]) -> None:
    expected_python = (
        wheelhouse_root / "preflight-venv" / "Scripts" / "python.exe"
    ).resolve(strict=True)
    if Path(sys.executable).resolve(strict=True) != expected_python:
        raise LiveClipBridgeHold("HOLD_EXACT_LOCAL_RUNTIME_INTERPRETER")
    observed = {
        "huggingface_hub": str(modules["huggingface_hub"].__version__),
        "numpy": str(modules["numpy"].__version__),
        "python": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "tokenizers": str(modules["tokenizers"].__version__),
        "torch": str(modules["torch"].__version__),
        "transformers": str(modules["transformers"].__version__),
    }
    torch = modules["torch"]
    if (
        observed != _EXPECTED_RUNTIME
        or torch.version.cuda is not None
        or torch.cuda.is_available() is not False
    ):
        raise LiveClipBridgeHold("HOLD_EXACT_LOCAL_RUNTIME_IDENTITY")


@dataclass(frozen=True, slots=True)
class _TensorBoundaryFacts:
    storage: object
    device_type: str
    device_index: int | None
    pointer: int
    byte_capacity: int
    offset: int
    element_count: int
    item_bytes: int
    is_c_contiguous: bool
    shape: tuple[int, ...]
    stride: tuple[int, ...]


def _make_tensor_boundary_helpers(
    *,
    _facts_type: type[_TensorBoundaryFacts] = _TensorBoundaryFacts,
    _hold_type: type[LiveClipBridgeHold] = LiveClipBridgeHold,
    _sha256: object = _sha256_bytes,
    _canonical: object = _canonical_json,
):
    """Capture exact Tensor and UntypedStorage primitives in one sealed factory."""

    import torch

    tensor_type = torch.Tensor
    parameter_type = torch.nn.Parameter
    storage_type = torch.UntypedStorage
    tensor_detach = tensor_type.detach
    tensor_clone = tensor_type.clone
    tensor_to = tensor_type.to
    tensor_contiguous = tensor_type.contiguous
    tensor_numpy = tensor_type.numpy
    tensor_is_contiguous = tensor_type.is_contiguous
    tensor_untyped_storage = tensor_type.untyped_storage
    tensor_storage_offset = tensor_type.storage_offset
    tensor_numel = tensor_type.numel
    tensor_element_size = tensor_type.element_size
    tensor_stride = tensor_type.stride
    tensor_all = tensor_type.all
    tensor_item = tensor_type.item
    storage_data_ptr = storage_type.data_ptr
    storage_nbytes = storage_type.nbytes
    contiguous_format = torch.contiguous_format
    float32 = torch.float32
    isfinite = torch.isfinite

    def boundary_types() -> tuple[type[object], type[object], type[object]]:
        return parameter_type, tensor_type, storage_type

    def tensor_bytes(tensor: object) -> bytes:
        if type(tensor) not in (tensor_type, parameter_type):
            raise _hold_type("HOLD_TENSOR_BYTES_EXACT_TYPE")
        detached = tensor_detach(tensor)
        on_cpu = tensor_to(detached, device="cpu")
        checked = tensor_contiguous(on_cpu)
        return tensor_numpy(checked).tobytes(order="C")

    def clone_parameter(parameter: object) -> object:
        if type(parameter) is not parameter_type:
            raise _hold_type("HOLD_CLIP_PARAMETER_OBJECT_IDENTITY")
        copied = tensor_clone(
            tensor_detach(parameter),
            memory_format=contiguous_format,
        )
        return parameter_type(copied, requires_grad=parameter.requires_grad)

    def storage_facts(tensor: object) -> _TensorBoundaryFacts:
        if type(tensor) not in (tensor_type, parameter_type):
            raise _hold_type("HOLD_TENSOR_STORAGE_EXACT_TYPE")
        storage = tensor_untyped_storage(tensor)
        if type(storage) is not storage_type:
            raise _hold_type("HOLD_TENSOR_STORAGE_EXACT_TYPE")
        return _facts_type(
            storage=storage,
            device_type=tensor.device.type,
            device_index=tensor.device.index,
            pointer=int(storage_data_ptr(storage)),
            byte_capacity=int(storage_nbytes(storage)),
            offset=int(tensor_storage_offset(tensor)),
            element_count=int(tensor_numel(tensor)),
            item_bytes=int(tensor_element_size(tensor)),
            is_c_contiguous=bool(tensor_is_contiguous(tensor)),
            shape=tuple(tensor.shape),
            stride=tuple(int(item) for item in tensor_stride(tensor)),
        )

    def storage_key(tensor: object) -> tuple[str, int | None, int, int]:
        facts = storage_facts(tensor)
        return (
            facts.device_type,
            facts.device_index,
            facts.pointer,
            facts.byte_capacity,
        )

    def finite_tensor(tensor: object) -> bool:
        if type(tensor) not in (tensor_type, parameter_type):
            raise _hold_type("HOLD_TENSOR_FINITE_EXACT_TYPE")
        finite = isfinite(tensor_detach(tensor))
        return bool(tensor_item(tensor_all(finite)))

    def exact_parameter(
        value: object,
        *,
        shape: tuple[int, ...],
        torch: object,
        label: str,
    ) -> object:
        if (
            torch.Tensor is not tensor_type
            or torch.nn.Parameter is not parameter_type
            or torch.UntypedStorage is not storage_type
            or type(value) is not parameter_type
        ):
            raise _hold_type(f"HOLD_{label}_PARAMETER_INVARIANT")
        facts = storage_facts(value)
        if (
            facts.shape != shape
            or facts.device_type != "cpu"
            or value.dtype != float32
            or not value.requires_grad
            or not value.is_leaf
            or value.grad is not None
            or not facts.is_c_contiguous
            or facts.offset != 0
            or facts.byte_capacity != facts.element_count * facts.item_bytes
            or facts.pointer == 0
            or not finite_tensor(value)
        ):
            raise _hold_type(f"HOLD_{label}_PARAMETER_INVARIANT")
        return value

    def parameter_row_facts(
        value: object,
        *,
        shape: tuple[int, ...],
        numel: int,
    ) -> _TensorBoundaryFacts:
        if type(value) is not parameter_type:
            raise _hold_type("HOLD_PARAMETER_ROW_PROVENANCE")
        facts = storage_facts(value)
        if (
            facts.shape != shape
            or facts.element_count != numel
            or facts.device_type != "cpu"
            or value.dtype != float32
            or value.requires_grad is not True
            or not value.is_leaf
            or value.grad is not None
            or not facts.is_c_contiguous
            or facts.offset != 0
            or facts.byte_capacity != facts.element_count * facts.item_bytes
            or facts.pointer == 0
            or not finite_tensor(value)
        ):
            raise _hold_type("HOLD_PARAMETER_ROW_PROVENANCE")
        return facts

    def binding_digest(ordered: tuple[tuple[str, object], ...]) -> str:
        object_ordinals: dict[int, int] = {}
        storage_ordinals: dict[tuple[str, int | None, int, int], int] = {}
        rows: list[dict[str, object]] = []
        intervals: list[tuple[int, int]] = []
        for name, parameter in ordered:
            if type(name) is not str or not name or type(parameter) is not parameter_type:
                raise _hold_type("HOLD_PARAMETER_ROW_PROVENANCE")
            facts = storage_facts(parameter)
            if (
                facts.device_type != "cpu"
                or parameter.dtype != float32
                or parameter.requires_grad is not True
                or not parameter.is_leaf
                or parameter.grad is not None
                or not facts.is_c_contiguous
                or facts.offset != 0
                or facts.byte_capacity != facts.element_count * facts.item_bytes
                or facts.pointer == 0
                or not finite_tensor(parameter)
            ):
                raise _hold_type("HOLD_PARAMETER_ROW_PROVENANCE")
            object_ordinal = object_ordinals.setdefault(
                id(parameter), len(object_ordinals)
            )
            key = (
                facts.device_type,
                facts.device_index,
                facts.pointer,
                facts.byte_capacity,
            )
            storage_ordinal = storage_ordinals.setdefault(key, len(storage_ordinals))
            extent = facts.element_count * facts.item_bytes
            start = facts.pointer + facts.offset * facts.item_bytes
            intervals.append((start, start + extent))
            rows.append(
                {
                    "name": name,
                    "numel": facts.element_count,
                    "object_ordinal": object_ordinal,
                    "shape": list(facts.shape),
                    "storage_identity_ordinal": storage_ordinal,
                    "storage_nbytes": facts.byte_capacity,
                    "storage_offset": facts.offset,
                }
            )
        if len(object_ordinals) != len(ordered) or len(storage_ordinals) != len(ordered):
            raise _hold_type("HOLD_FULL_PARAMETER_ALIAS")
        sorted_intervals = sorted(intervals)
        if any(
            right[0] < left[1]
            for left, right in zip(sorted_intervals, sorted_intervals[1:])
        ):
            raise _hold_type("HOLD_FULL_PARAMETER_STORAGE_OVERLAP")
        return _sha256(
            _canonical(
                {"rows": rows, "schema": "phasepair-live-optimizer-binding-v1"},
                final_lf=False,
            )
        )

    return (
        boundary_types,
        tensor_bytes,
        clone_parameter,
        storage_facts,
        storage_key,
        exact_parameter,
        parameter_row_facts,
        binding_digest,
    )


(
    _tensor_boundary_types,
    _tensor_bytes,
    _clone_exact_parameter,
    _tensor_storage_facts,
    _storage_key,
    _exact_parameter,
    _parameter_row_facts,
    _binding_digest,
) = _make_tensor_boundary_helpers()


def _direct_registry(
    model: object,
    *,
    parameter_type: type[object],
    tensor_type: type[object],
) -> tuple[
    tuple[tuple[str, object, object], ...],
    tuple[tuple[str, object, object], ...],
    tuple[tuple[str, object], ...],
]:
    parameter_rows: list[tuple[str, object, object]] = []
    buffer_rows: list[tuple[str, object, object]] = []
    module_rows: list[tuple[str, object]] = []
    module_ids: set[int] = set()
    parameter_ids: set[int] = set()
    buffer_ids: set[int] = set()

    def walk(module: object, path: str) -> None:
        if id(module) in module_ids:
            raise LiveClipBridgeHold("HOLD_CLIP_MODULE_ALIAS_OR_CYCLE")
        module_ids.add(id(module))
        module_rows.append((path, module))
        instance = object.__getattribute__(module, "__dict__")
        if type(instance) is not dict:
            raise LiveClipBridgeHold("HOLD_CLIP_DIRECT_REGISTRY_TYPE")
        parameters = instance.get("_parameters")
        buffers = instance.get("_buffers")
        children = instance.get("_modules")
        if not all(type(value) is dict for value in (parameters, buffers, children)):
            raise LiveClipBridgeHold("HOLD_CLIP_DIRECT_REGISTRY_TYPE")
        for local_name, parameter in parameters.items():
            if type(local_name) is not str or not local_name:
                raise LiveClipBridgeHold("HOLD_CLIP_DIRECT_REGISTRY_NAME")
            if parameter is None:
                continue
            if type(parameter) is not parameter_type or id(parameter) in parameter_ids:
                raise LiveClipBridgeHold("HOLD_CLIP_PARAMETER_OBJECT_IDENTITY")
            parameter_ids.add(id(parameter))
            name = local_name if not path else f"{path}.{local_name}"
            parameter_rows.append((name, parameter, module))
        for local_name, buffer in buffers.items():
            if type(local_name) is not str or not local_name:
                raise LiveClipBridgeHold("HOLD_CLIP_DIRECT_REGISTRY_NAME")
            if buffer is None:
                continue
            if type(buffer) is not tensor_type or id(buffer) in buffer_ids:
                raise LiveClipBridgeHold("HOLD_CLIP_BUFFER_OBJECT_IDENTITY")
            buffer_ids.add(id(buffer))
            name = local_name if not path else f"{path}.{local_name}"
            buffer_rows.append((name, buffer, module))
        for local_name, child in children.items():
            if type(local_name) is not str or not local_name or child is None:
                raise LiveClipBridgeHold("HOLD_CLIP_DIRECT_REGISTRY_CHILD")
            child_path = local_name if not path else f"{path}.{local_name}"
            walk(child, child_path)

    walk(model, "")
    return tuple(parameter_rows), tuple(buffer_rows), tuple(module_rows)


def _materialize_registered_parameters(model: object, torch: object) -> None:
    parameter_type, tensor_type, storage_type = _tensor_boundary_types()
    if (
        torch.nn.Parameter is not parameter_type
        or torch.Tensor is not tensor_type
        or torch.UntypedStorage is not storage_type
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_TENSOR_RUNTIME_REBOUND")
    parameter_rows, _buffer_rows, module_rows = _direct_registry(
        model,
        parameter_type=parameter_type,
        tensor_type=tensor_type,
    )
    before_names = tuple(name for name, _parameter, _module in parameter_rows)
    before_hashes = tuple(
        (name, _sha256_bytes(_tensor_bytes(parameter)))
        for name, parameter, _module in parameter_rows
    )
    for _path, module in module_rows:
        registry = object.__getattribute__(module, "__dict__")["_parameters"]
        for local_name, parameter in tuple(registry.items()):
            if parameter is None:
                continue
            registry[local_name] = _clone_exact_parameter(parameter)
    after_rows, _buffers, _modules = _direct_registry(
        model,
        parameter_type=parameter_type,
        tensor_type=tensor_type,
    )
    after_names = tuple(name for name, _parameter, _module in after_rows)
    after_hashes = tuple(
        (name, _sha256_bytes(_tensor_bytes(parameter)))
        for name, parameter, _module in after_rows
    )
    if before_names != after_names or before_hashes != after_hashes:
        raise LiveClipBridgeHold("HOLD_CLIP_MATERIALIZATION_CONTENT_DRIFT")


def _live_manifests(
    model: object, torch: object
) -> tuple[
    tuple[tuple[str, object, object], ...],
    tuple[tuple[str, object, object], ...],
    bytes,
    bytes,
]:
    parameter_type, tensor_type, storage_type = _tensor_boundary_types()
    if (
        torch.nn.Parameter is not parameter_type
        or torch.Tensor is not tensor_type
        or torch.UntypedStorage is not storage_type
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_TENSOR_RUNTIME_REBOUND")
    parameters, buffers, _modules = _direct_registry(
        model,
        parameter_type=parameter_type,
        tensor_type=tensor_type,
    )
    storage_ordinals: dict[tuple[str, int | None, int, int], int] = {}
    parameter_json: list[dict[str, object]] = []
    intervals: list[tuple[int, int]] = []
    for index, (name, parameter, _module) in enumerate(parameters):
        facts = _tensor_storage_facts(parameter)
        key = _storage_key(parameter)
        ordinal = storage_ordinals.setdefault(key, len(storage_ordinals))
        extent = facts.element_count * facts.item_bytes
        start = facts.pointer + facts.offset * facts.item_bytes
        intervals.append((start, start + extent))
        parameter_json.append(
            {
                "contiguous": facts.is_c_contiguous,
                "dtype": str(parameter.dtype),
                "index": index,
                "name": name,
                "numel": facts.element_count,
                "requires_grad": parameter.requires_grad,
                "shape": list(facts.shape),
                "storage_identity_ordinal": ordinal,
                "storage_nbytes": facts.byte_capacity,
                "storage_offset": facts.offset,
                "stride": list(facts.stride),
                "tensor_bytes_sha256": _sha256_bytes(_tensor_bytes(parameter)),
            }
        )
        if (
            type(parameter) is not parameter_type
            or facts.device_type != "cpu"
            or parameter.dtype != torch.float32
            or not parameter.is_leaf
            or parameter.grad is not None
            or not facts.is_c_contiguous
            or facts.offset != 0
            or facts.byte_capacity != extent
            or facts.pointer == 0
        ):
            raise LiveClipBridgeHold("HOLD_CLIP_PARAMETER_STORAGE_INVARIANT")
    if len(parameters) != 197 or len(storage_ordinals) != 197:
        raise LiveClipBridgeHold("HOLD_CLIP_PARAMETER_STORAGE_IDENTITY")
    ordered_intervals = sorted(intervals)
    if any(right[0] < left[1] for left, right in zip(ordered_intervals, ordered_intervals[1:])):
        raise LiveClipBridgeHold("HOLD_CLIP_PARAMETER_STORAGE_OVERLAP")

    buffer_json: list[dict[str, object]] = []
    if len(buffers) != 1 or buffers[0][0] != "text_model.embeddings.position_ids":
        raise LiveClipBridgeHold("HOLD_CLIP_BUFFER_CENSUS")
    for index, (name, buffer, _module) in enumerate(buffers):
        facts = _tensor_storage_facts(buffer)
        extent = facts.element_count * facts.item_bytes
        start = facts.pointer + facts.offset * facts.item_bytes
        if any(start < end and begin < start + extent for begin, end in intervals):
            raise LiveClipBridgeHold("HOLD_CLIP_BUFFER_PARAMETER_OVERLAP")
        if (
            facts.shape != (1, 77)
            or buffer.dtype != torch.int64
            or not facts.is_c_contiguous
            or facts.offset != 0
            or facts.byte_capacity != 616
        ):
            raise LiveClipBridgeHold("HOLD_CLIP_BUFFER_STORAGE_INVARIANT")
        buffer_json.append(
            {
                "contiguous": facts.is_c_contiguous,
                "dtype": str(buffer.dtype),
                "index": index,
                "name": name,
                "numel": facts.element_count,
                "shape": list(facts.shape),
                "storage_nbytes": facts.byte_capacity,
                "storage_offset": facts.offset,
                "stride": list(facts.stride),
                "tensor_bytes_sha256": _sha256_bytes(_tensor_bytes(buffer)),
            }
        )
    parameter_manifest = _canonical_json(parameter_json, final_lf=False)
    buffer_manifest = _canonical_json(buffer_json, final_lf=False)
    if (
        len(parameter_manifest) != 66_645
        or _sha256_bytes(parameter_manifest) != _EXPECTED_PARAMETER_MANIFEST_SHA256
        or len(buffer_manifest) != 268
        or _sha256_bytes(buffer_manifest) != _EXPECTED_BUFFER_MANIFEST_SHA256
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_LIVE_MANIFEST_MISMATCH")
    return parameters, buffers, parameter_manifest, buffer_manifest


def _semantic_class(module: object, local_name: str, torch: object) -> str:
    if isinstance(module, torch.nn.LayerNorm):
        return (
            contracts.LAYERNORM_GAMMA
            if local_name == "weight"
            else contracts.LAYERNORM_BETA
        )
    if isinstance(module, torch.nn.Embedding):
        return contracts.EMBEDDING_WEIGHT
    if local_name == "bias":
        return contracts.BIAS
    if local_name == "weight" and any(
        token in type(module).__qualname__.casefold()
        for token in ("attention", "attn")
    ):
        return contracts.QUERY_OR_OTHER_WEIGHT
    return contracts.MATRIX_WEIGHT


def _build_text_rows_and_bindings(
    parameters: tuple[tuple[str, object, object], ...], torch: object
) -> tuple[
    tuple[contracts.ParameterRow, ...],
    tuple[contracts.ParameterRow, ...],
    tuple[tuple[str, object], ...],
    str,
]:
    trainable_rows: list[contracts.ParameterRow] = []
    bindings: list[tuple[str, object]] = []
    frozen_rows: list[contracts.ParameterRow] = []
    for resolved_name, parameter, module in parameters:
        if resolved_name == "text_projection.weight":
            frozen_rows.append(
                contracts.ParameterRow(
                    "text.pretrained_projection.weight",
                    tuple(parameter.shape),
                    contracts.FROZEN,
                    contracts.PRETRAINED_PROJECTION_COMPONENT,
                    False,
                    resolved_name,
                    _SOURCE_CHECKPOINT_SHA256,
                )
            )
            continue
        if not resolved_name.startswith("text_model.") or not parameter.requires_grad:
            raise LiveClipBridgeHold("HOLD_CLIP_TRAINABLE_PARAMETER_CENSUS")
        local_name = resolved_name.rsplit(".", 1)[-1]
        canonical_name = f"text.clip.{resolved_name}"
        row = contracts.ParameterRow(
            canonical_name,
            tuple(parameter.shape),
            _semantic_class(module, local_name, torch),
            contracts.CLIP_TEXT_COMPONENT,
            True,
            resolved_name,
            _SOURCE_CHECKPOINT_SHA256,
        )
        trainable_rows.append(row)
        bindings.append((canonical_name, parameter))
    rows = tuple(sorted(trainable_rows, key=lambda row: row.name.encode("utf-8")))
    binding_map = dict(bindings)
    ordered_bindings = tuple((row.name, binding_map[row.name]) for row in rows)
    frozen = tuple(sorted(frozen_rows, key=lambda row: row.name.encode("utf-8")))
    checked = contracts.validate_resolved_text_rows(rows, 512)
    contracts.validate_frozen_rows(frozen)
    decay = tuple(row for row in checked if row.semantic_class in contracts.DECAY_CLASSES)
    no_decay = tuple(
        row for row in checked if row.semantic_class in contracts.NO_DECAY_CLASSES
    )
    if (
        len(checked) != _EXPECTED_TEXT_ROWS
        or sum(row.numel for row in checked) != _EXPECTED_TEXT_NUMEL
        or (len(decay), sum(row.numel for row in decay)) != _EXPECTED_TEXT_DECAY
        or (len(no_decay), sum(row.numel for row in no_decay))
        != _EXPECTED_TEXT_NO_DECAY
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_TEXT_PARTITION_ARITHMETIC")
    binding_payload = [
        {
            "canonical_name": row.name,
            "object_ordinal": index,
            "resolved_state_key": row.resolved_state_key,
            "shape": list(row.shape),
        }
        for index, row in enumerate(rows)
    ]
    return rows, frozen, ordered_bindings, _sha256_bytes(
        _canonical_json(
            {
                "rows": binding_payload,
                "schema": "phasepair-live-clip-parameter-binding-v1",
            },
            final_lf=False,
        )
    )


@contextmanager
def _offline_environment(cache_root: Path) -> Iterator[list[str]]:
    keys = (
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "HF_HUB_DISABLE_TELEMETRY",
        "TOKENIZERS_PARALLELISM",
        "HF_HOME",
        "HF_HUB_CACHE",
    )
    old_values = {key: os.environ.get(key) for key in keys}
    os.environ.update(
        {
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "TOKENIZERS_PARALLELISM": "false",
            "HF_HOME": str(cache_root / "hf-home"),
            "HF_HUB_CACHE": str(cache_root / "hub"),
        }
    )
    attempts: list[str] = []

    def blocked(label: str) -> Any:
        def reject(*_args: object, **_kwargs: object) -> object:
            attempts.append(label)
            raise RuntimeError(f"network disabled during sealed CLIP load: {label}")

        return reject

    import requests

    patches = (
        (requests.sessions.Session, "request", requests.sessions.Session.request),
        (urllib.request, "urlopen", urllib.request.urlopen),
        (http.client.HTTPConnection, "connect", http.client.HTTPConnection.connect),
        (http.client.HTTPSConnection, "connect", http.client.HTTPSConnection.connect),
        (socket, "create_connection", socket.create_connection),
        (socket.socket, "connect", socket.socket.connect),
    )
    try:
        for owner, name, _original in patches:
            setattr(owner, name, blocked(f"{owner!r}.{name}"))
        yield attempts
    finally:
        for owner, name, original in reversed(patches):
            setattr(owner, name, original)
        for key, value in old_values.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _load_and_verify_impl(
    snapshot_root: Path, wheelhouse_root: Path
) -> _RuntimePayload:
    import huggingface_hub
    import numpy
    import tokenizers
    import torch
    import torch.nn.functional as functional
    import transformers
    from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

    modules = {
        "huggingface_hub": huggingface_hub,
        "numpy": numpy,
        "tokenizers": tokenizers,
        "torch": torch,
        "transformers": transformers,
    }
    _validate_runtime_environment(wheelhouse_root, modules)
    rng_entry = torch.get_rng_state().clone()
    with tempfile.TemporaryDirectory(
        prefix="phasepair-live-clip-empty-cache-", dir=wheelhouse_root
    ) as cache_text:
        cache_root = Path(cache_text)
        if tuple(cache_root.rglob("*")):
            raise LiveClipBridgeHold("HOLD_EMPTY_NETWORK_CACHE_NOT_EMPTY")
        with _offline_environment(cache_root) as network_attempts:
            with torch.random.fork_rng(devices=[], enabled=True):
                torch.manual_seed(0)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    tokenizer = CLIPTokenizerFast.from_pretrained(
                        str(snapshot_root),
                        local_files_only=True,
                        trust_remote_code=False,
                    )
                    model, loading_info = (
                        CLIPTextModelWithProjection.from_pretrained(
                            str(snapshot_root),
                            local_files_only=True,
                            trust_remote_code=False,
                            low_cpu_mem_usage=False,
                            output_loading_info=True,
                        )
                    )
                if caught:
                    raise LiveClipBridgeHold("HOLD_CLIP_LOADER_WARNING")
                if type(tokenizer) is not CLIPTokenizerFast:
                    raise LiveClipBridgeHold("HOLD_CLIP_TOKENIZER_EXACT_TYPE")
                if type(model) is not CLIPTextModelWithProjection:
                    raise LiveClipBridgeHold("HOLD_CLIP_MODEL_EXACT_TYPE")
                _materialize_registered_parameters(model, torch)
                for parameter in object.__getattribute__(
                    object.__getattribute__(model, "__dict__")["_modules"][
                        "text_projection"
                    ],
                    "__dict__",
                )["_parameters"].values():
                    if parameter is not None:
                        parameter.requires_grad_(False)
                model.train()
                encoded = tokenizer(
                    list(_CAPTIONS),
                    max_length=77,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                )
                dropout_calls: list[tuple[str, float, bool]] = []
                original_dropout = functional.dropout
                original_sdpa = functional.scaled_dot_product_attention

                def traced_dropout(
                    input_tensor: object,
                    p: float = 0.5,
                    training: bool = True,
                    inplace: bool = False,
                ) -> object:
                    dropout_calls.append(("functional.dropout", float(p), bool(training)))
                    return original_dropout(
                        input_tensor, p=p, training=training, inplace=inplace
                    )

                def traced_sdpa(*args: object, **kwargs: object) -> object:
                    if "dropout_p" in kwargs:
                        probability = float(kwargs["dropout_p"])
                    elif len(args) >= 5:
                        probability = float(args[4])
                    else:
                        probability = 0.0
                    dropout_calls.append(("scaled_dot_product_attention", probability, True))
                    return original_sdpa(*args, **kwargs)

                functional.dropout = traced_dropout
                functional.scaled_dot_product_attention = traced_sdpa
                forward_rng_before = torch.get_rng_state().clone()
                try:
                    output = model.text_model(
                        input_ids=encoded["input_ids"],
                        attention_mask=encoded["attention_mask"],
                    )
                    pooled = output.pooler_output
                    projected = model.text_projection(pooled)
                finally:
                    functional.dropout = original_dropout
                    functional.scaled_dot_product_attention = original_sdpa
                forward_rng_after = torch.get_rng_state().clone()
                normalized = projected.detach().to(dtype=torch.float64)
                normalized = normalized / torch.linalg.vector_norm(
                    normalized, ord=2, dim=-1, keepdim=True
                )
        if network_attempts or tuple(cache_root.rglob("*")):
            raise LiveClipBridgeHold("HOLD_NETWORK_OR_CACHE_ACTIVITY")
    if not torch.equal(rng_entry, torch.get_rng_state()):
        raise LiveClipBridgeHold("HOLD_CALLER_RNG_CHANGED")
    if not torch.equal(forward_rng_before, forward_rng_after):
        raise LiveClipBridgeHold("HOLD_CLIP_FORWARD_RNG_CHANGED")
    if (
        len(dropout_calls) != 12
        or any(name != "scaled_dot_product_attention" for name, _p, _t in dropout_calls)
        or any(probability != 0.0 for _name, probability, _training in dropout_calls)
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_ACTIVE_DROPOUT")
    unexpected = tuple(sorted(loading_info.get("unexpected_keys", ())))
    missing = tuple(sorted(loading_info.get("missing_keys", ())))
    mismatched = tuple(sorted(str(value) for value in loading_info.get("mismatched_keys", ())))
    errors = tuple(sorted(str(value) for value in loading_info.get("error_msgs", ())))
    if (
        missing
        or mismatched
        or errors
        or len(unexpected) != 202
        or _sha256_bytes(
            "".join(key + "\n" for key in unexpected).encode("ascii")
        )
        != "354d27b4b81e1f6e9fc9e908b04fefd182847eeb35c980176b2260a8b572f597"
        or not all(
            key == "logit_scale"
            or key.startswith("vision_model.")
            or key.startswith("visual_projection.")
            for key in unexpected
        )
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_LOADING_INFO")
    config = {
        "attention_dropout": float(model.config.attention_dropout),
        "bos_token_id": int(model.config.bos_token_id),
        "eos_token_id": int(model.config.eos_token_id),
        "hidden_act": str(model.config.hidden_act),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(model.config.intermediate_size),
        "layer_norm_eps": float(model.config.layer_norm_eps),
        "max_position_embeddings": int(model.config.max_position_embeddings),
        "model_type": str(model.config.model_type),
        "num_attention_heads": int(model.config.num_attention_heads),
        "num_hidden_layers": int(model.config.num_hidden_layers),
        "pad_token_id": int(model.config.pad_token_id),
        "projection_dim": int(model.config.projection_dim),
        "vocab_size": int(model.config.vocab_size),
    }
    if config != _EXPECTED_CONFIG:
        raise LiveClipBridgeHold("HOLD_CLIP_CONFIG_IDENTITY")
    modeling_source = Path(inspect.getsourcefile(CLIPTextModelWithProjection) or "")
    tokenizer_source = Path(inspect.getsourcefile(CLIPTokenizerFast) or "")
    for path, bytes_key, digest_key in (
        (modeling_source, "modeling_source_bytes", "modeling_source_sha256"),
        (tokenizer_source, "tokenizer_source_bytes", "tokenizer_source_sha256"),
    ):
        if (
            path.stat().st_size != _EXPECTED_SOURCE[bytes_key]
            or _sha256_file(path) != _EXPECTED_SOURCE[digest_key]
        ):
            raise LiveClipBridgeHold("HOLD_CLIP_RUNTIME_SOURCE_IDENTITY")
    parameters, _buffers, parameter_manifest, buffer_manifest = _live_manifests(
        model, torch
    )
    parameter_type, tensor_type, _storage_type = _tensor_boundary_types()
    module_rows = _direct_registry(
        model,
        parameter_type=parameter_type,
        tensor_type=tensor_type,
    )[2]
    if any("vision" in name.casefold() for name, _module in module_rows):
        raise LiveClipBridgeHold("HOLD_CLIP_VISION_MODULE_PRESENT")
    if any(
        isinstance(module, torch.nn.modules.dropout._DropoutNd)
        for _name, module in module_rows
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_DROPOUT_MODULE_PRESENT")
    state_names = tuple(name for name, _parameter, _module in parameters)
    if (
        _sha256_bytes("".join(key + "\n" for key in state_names).encode("ascii"))
        != _EXPECTED_STATE_KEYS_SHA256
    ):
        raise LiveClipBridgeHold("HOLD_CLIP_STATE_KEY_ORDER")
    observed_goldens = {
        "attention_mask_sha256": _sha256_bytes(_tensor_bytes(encoded["attention_mask"])),
        "input_ids_sha256": _sha256_bytes(_tensor_bytes(encoded["input_ids"])),
        "normalized_float64_l2_sha256": _sha256_bytes(_tensor_bytes(normalized)),
        "pooled_eos_sha256": _sha256_bytes(_tensor_bytes(pooled)),
        "pretrained_projection_sha256": _sha256_bytes(_tensor_bytes(projected)),
    }
    if observed_goldens != _EXPECTED_GOLDENS:
        raise LiveClipBridgeHold("HOLD_CLIP_GOLDEN_MISMATCH")
    text_rows, frozen_rows, text_bindings, text_binding_sha256 = (
        _build_text_rows_and_bindings(parameters, torch)
    )
    inventory_sha256 = _sha256_bytes(
        contracts.canonical_inventory_json_bytes(text_rows)
    )
    snapshot_rows = _scan_snapshot(snapshot_root)
    runtime_identity = tuple(sorted(_EXPECTED_RUNTIME.items()))
    return _RuntimePayload(
        snapshot_root,
        snapshot_rows,
        _snapshot_manifest_sha256(snapshot_rows),
        _EXPECTED_PREFLIGHT_SHA256,
        _EXPECTED_PRIVATE_RUNTIME_SHA256,
        _EXPECTED_RIGHTS_SHA256,
        runtime_identity,
        text_rows,
        frozen_rows,
        text_bindings,
        _sha256_bytes(parameter_manifest),
        _sha256_bytes(buffer_manifest),
        inventory_sha256,
        text_binding_sha256,
        model,
        tokenizer,
    )


def _load_and_verify(snapshot_root: Path, wheelhouse_root: Path) -> _RuntimePayload:
    import torch

    original_thread_count = torch.get_num_threads()
    try:
        if original_thread_count != 1:
            torch.set_num_threads(1)
        return _load_and_verify_impl(snapshot_root, wheelhouse_root)
    finally:
        if torch.get_num_threads() != original_thread_count:
            torch.set_num_threads(original_thread_count)


def _payload_seal(payload: _RuntimePayload) -> bytes:
    return hashlib.sha256(
        _canonical_json(
            {
                "buffer_manifest_sha256": payload.buffer_manifest_sha256,
                "parameter_manifest_sha256": payload.parameter_manifest_sha256,
                "preflight_sha256": payload.preflight_sha256,
                "private_runtime_sha256": payload.private_runtime_sha256,
                "rights_sha256": payload.rights_sha256,
                "runtime_identity": dict(payload.runtime_identity),
                "snapshot_manifest_sha256": payload.snapshot_manifest_sha256,
                "text_binding_sha256": payload.text_binding_sha256,
                "text_inventory_sha256": payload.text_inventory_sha256,
            },
            final_lf=False,
        )
    ).digest()


def _resolve_runtime_payload(
    snapshot_root: Path,
    wheelhouse_root: Path,
    rights_receipt: Path,
) -> _RuntimePayload:
    """Resolve a payload; only the sealed lifecycle closure may issue it."""

    for value, label in (
        (snapshot_root, "snapshot_root"),
        (wheelhouse_root, "wheelhouse_root"),
        (rights_receipt, "rights_receipt"),
    ):
        if not isinstance(value, Path):
            raise TypeError(f"{label} must be a concrete pathlib.Path")
    with _LOAD_LOCK:
        try:
            wheelhouse = wheelhouse_root.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise LiveClipBridgeHold("HOLD_WHEELHOUSE_ROOT_ABSENT") from exc
        if wheelhouse_root != wheelhouse or not wheelhouse.is_dir():
            raise LiveClipBridgeHold("HOLD_WHEELHOUSE_ROOT_IDENTITY")
        receipts = wheelhouse / "receipts"
        preflight_path = receipts / _EXPECTED_PREFLIGHT_NAME
        private_runtime_path = receipts / _EXPECTED_PRIVATE_RUNTIME_NAME
        script_path = wheelhouse / "preflight_text_only.py"
        preflight_raw = _safe_regular_file(
            preflight_path,
            expected_bytes=_EXPECTED_PREFLIGHT_BYTES,
            expected_sha256=_EXPECTED_PREFLIGHT_SHA256,
        )
        private_runtime_raw = _safe_regular_file(
            private_runtime_path,
            expected_bytes=_EXPECTED_PRIVATE_RUNTIME_BYTES,
            expected_sha256=_EXPECTED_PRIVATE_RUNTIME_SHA256,
        )
        _safe_regular_file(
            script_path,
            expected_bytes=_EXPECTED_PREFLIGHT_SCRIPT_BYTES,
            expected_sha256=_EXPECTED_PREFLIGHT_SCRIPT_SHA256,
        )
        _safe_regular_file(
            rights_receipt,
            expected_bytes=_EXPECTED_RIGHTS_BYTES,
            expected_sha256=_EXPECTED_RIGHTS_SHA256,
        )
        _validate_candidate_receipt(preflight_raw)
        _validate_private_runtime_receipt(private_runtime_raw)
        assessment = clip_resolution.assess_local_clip_snapshot(snapshot_root)
        assessment_bytes = clip_resolution.canonical_assessment_bytes(assessment)
        if _sha256_bytes(assessment_bytes) != _EXPECTED_SNAPSHOT_ASSESSMENT_SHA256:
            raise LiveClipBridgeHold("HOLD_SEALED_SNAPSHOT_RESOLVER_MISMATCH")
        payload = _load_and_verify(snapshot_root, wheelhouse)
        if payload.snapshot_rows != _scan_snapshot(snapshot_root):
            raise LiveClipBridgeHold("HOLD_CLIP_SNAPSHOT_TOCTOU")
        return payload


def _candidate_receipt_payload(payload: _RuntimePayload) -> dict[str, object]:
    return {
        "adamw_constructor_count": 0,
        "authority": 0,
        "buffer_manifest_sha256": payload.buffer_manifest_sha256,
        "hold_reasons": [
            "HOLD_NO_TRAINING_AUTHORITY_ISSUER",
            "HOLD_DATA_RIGHTS_LINEAGE_OWNER_AND_SERVER_GATES",
        ],
        "lifecycle": "ISSUED",
        "model_id": _MODEL_ID,
        "parameter_manifest_sha256": payload.parameter_manifest_sha256,
        "preflight_candidate_sha256": payload.preflight_sha256,
        "private_runtime_receipt_sha256": payload.private_runtime_sha256,
        "production": False,
        "revision": _REVISION,
        "rights_receipt_sha256": payload.rights_sha256,
        "runtime": dict(payload.runtime_identity),
        "schema": "phasepair-live-clip-text-candidate-receipt-v1",
        "snapshot_manifest_sha256": payload.snapshot_manifest_sha256,
        "status": STATUS,
        "text_binding_sha256": payload.text_binding_sha256,
        "text_decay_numel": _EXPECTED_TEXT_DECAY[1],
        "text_decay_tensor_count": _EXPECTED_TEXT_DECAY[0],
        "text_inventory_sha256": payload.text_inventory_sha256,
        "text_no_decay_numel": _EXPECTED_TEXT_NO_DECAY[1],
        "text_no_decay_tensor_count": _EXPECTED_TEXT_NO_DECAY[0],
        "text_trainable_numel": _EXPECTED_TEXT_NUMEL,
        "text_trainable_tensor_count": _EXPECTED_TEXT_ROWS,
        "training_authorized": False,
        "vision_module_count": 0,
        "vision_parameter_count": 0,
    }


def _make_optimizer_assessment(
    partition: contracts.OptimizerPartition,
    binding_sha256: str,
    missing_gates: tuple[str, ...],
) -> OptimizerCandidateAssessment:
    decay, no_decay = partition.groups
    value = object.__new__(OptimizerCandidateAssessment)
    fields = {
        "status": OPTIMIZER_STATUS,
        "authority": 0,
        "production": False,
        "training_authorized": False,
        "architecture": partition.architecture,
        "full_tensor_count": len(partition.full_trainable_rows),
        "full_numel": partition.total_numel,
        "decay_tensor_count": decay.tensor_count,
        "decay_numel": decay.numel,
        "no_decay_tensor_count": no_decay.tensor_count,
        "no_decay_numel": no_decay.numel,
        "inventory_sha256": _sha256_bytes(
            contracts.canonical_inventory_json_bytes(partition.full_trainable_rows)
        ),
        "decay_names_sha256": _sha256_bytes(
            contracts.canonical_name_list_bytes(decay.parameter_names)
        ),
        "no_decay_names_sha256": _sha256_bytes(
            contracts.canonical_name_list_bytes(no_decay.parameter_names)
        ),
        "parameter_binding_sha256": binding_sha256,
        "missing_gates": missing_gates,
        "adamw_constructor_count": 0,
    }
    for name, item in fields.items():
        object.__setattr__(value, name, item)
    return value


def _unsealed_optimizer_candidate_bytes(
    assessment: OptimizerCandidateAssessment,
) -> bytes:
    if type(assessment) is not OptimizerCandidateAssessment:
        raise TypeError("assessment must be exactly OptimizerCandidateAssessment")
    payload = {
        "adamw_constructor_count": assessment.adamw_constructor_count,
        "architecture": assessment.architecture,
        "authority": assessment.authority,
        "decay_names_sha256": assessment.decay_names_sha256,
        "decay_numel": assessment.decay_numel,
        "decay_tensor_count": assessment.decay_tensor_count,
        "full_numel": assessment.full_numel,
        "full_tensor_count": assessment.full_tensor_count,
        "inventory_sha256": assessment.inventory_sha256,
        "missing_gates": list(assessment.missing_gates),
        "no_decay_names_sha256": assessment.no_decay_names_sha256,
        "no_decay_numel": assessment.no_decay_numel,
        "no_decay_tensor_count": assessment.no_decay_tensor_count,
        "parameter_binding_sha256": assessment.parameter_binding_sha256,
        "production": assessment.production,
        "schema": "phasepair-local-optimizer-candidate-receipt-v1",
        "status": assessment.status,
        "training_authorized": assessment.training_authorized,
    }
    return _canonical_json(payload)


def _seal_live_clip_lifecycle(
    *,
    _lease_type: type[VerifiedLocalClipTextLease] = VerifiedLocalClipTextLease,
    _record_type: type[_LeaseRecord] = _LeaseRecord,
    _payload_type: type[_RuntimePayload] = _RuntimePayload,
    _assessment_type: type[OptimizerCandidateAssessment] = (
        OptimizerCandidateAssessment
    ),
    _resolver: object = _resolve_runtime_payload,
    _scan: object = _scan_snapshot,
    _seal: object = _payload_seal,
    _candidate_payload: object = _candidate_receipt_payload,
    _canonical: object = _canonical_json,
    _binding: object = _binding_digest,
    _make_assessment: object = _make_optimizer_assessment,
    _assessment_bytes: object = _unsealed_optimizer_candidate_bytes,
    _exact_parameter_impl: object = _exact_parameter,
    _row_facts: object = _parameter_row_facts,
    _validate_motion: object = initialization._validate_model,
    _expected_rows: object = contracts.expected_full_trainable_rows,
    _validate_partition: object = contracts.validate_optimizer_partition,
    _readiness_assess: object = readiness.assess_training_readiness,
    _rlock_factory: object = threading.RLock,
    _weak_registry_type: object = weakref.WeakKeyDictionary,
    _sha256_factory: object = hashlib.sha256,
    _hold_type: type[LiveClipBridgeHold] = LiveClipBridgeHold,
) -> tuple[object, ...]:
    """Capture the issuer registry and every lifecycle dependency in closures."""

    issued: weakref.WeakKeyDictionary[
        VerifiedLocalClipTextLease, _LeaseRecord
    ] = _weak_registry_type()
    assessments: weakref.WeakKeyDictionary[
        OptimizerCandidateAssessment, tuple[object, ...]
    ] = _weak_registry_type()
    registry_lock = _rlock_factory()
    load_lock = _rlock_factory()
    next_nonce = 0
    adamw_count = 0
    issued_state = 0
    validating_state = 1
    burned_state = 3
    expected_preflight = _EXPECTED_PREFLIGHT_SHA256
    expected_private_runtime = _EXPECTED_PRIVATE_RUNTIME_SHA256
    expected_rights = _EXPECTED_RIGHTS_SHA256
    expected_parameter_manifest = _EXPECTED_PARAMETER_MANIFEST_SHA256
    expected_buffer_manifest = _EXPECTED_BUFFER_MANIFEST_SHA256
    expected_text_count = 196
    expected_text_numel = 63_165_952
    expected_candidate_status = "VERIFIED_LOCAL_CPU_CANDIDATE_AUTHORITY0"
    expected_optimizer_status = "HOLD_NO_TRAINING_AUTHORITY_ISSUER"
    module_globals = globals()
    guarded_module_names = (
        "AUTHORITY",
        "BURNED",
        "ISSUED",
        "OPTIMIZER_STATUS",
        "PRODUCTION",
        "STATUS",
        "TRAINING_AUTHORIZED",
        "VALIDATING",
        "LiveClipBridgeHold",
        "OptimizerCandidateAssessment",
        "VerifiedLocalClipTextLease",
        "_CAPTIONS",
        "_EXPECTED_BUFFER_MANIFEST_SHA256",
        "_EXPECTED_GOLDENS",
        "_EXPECTED_PARAMETER_MANIFEST_SHA256",
        "_EXPECTED_PREFLIGHT_BYTES",
        "_EXPECTED_PREFLIGHT_CANONICAL_SHA256",
        "_EXPECTED_PREFLIGHT_NAME",
        "_EXPECTED_PREFLIGHT_SCRIPT_BYTES",
        "_EXPECTED_PREFLIGHT_SCRIPT_SHA256",
        "_EXPECTED_PREFLIGHT_SHA256",
        "_EXPECTED_PRIVATE_RUNTIME_BYTES",
        "_EXPECTED_PRIVATE_RUNTIME_NAME",
        "_EXPECTED_PRIVATE_RUNTIME_SHA256",
        "_EXPECTED_RIGHTS_BYTES",
        "_EXPECTED_RIGHTS_SHA256",
        "_EXPECTED_RUNTIME",
        "_EXPECTED_SNAPSHOT_ASSESSMENT_SHA256",
        "_EXPECTED_SOURCE",
        "_EXPECTED_STATE_KEYS_SHA256",
        "_EXPECTED_TEXT_DECAY",
        "_EXPECTED_TEXT_NO_DECAY",
        "_EXPECTED_TEXT_NUMEL",
        "_EXPECTED_TEXT_ROWS",
        "_LOAD_LOCK",
        "_MODEL_ID",
        "_REVISION",
        "_SOURCE_CHECKPOINT_SHA256",
        "_binding_digest",
        "_build_text_rows_and_bindings",
        "_candidate_receipt_payload",
        "_canonical_json",
        "_direct_registry",
        "_exact_parameter",
        "_clone_exact_parameter",
        "_live_manifests",
        "_load_and_verify",
        "_load_and_verify_impl",
        "_make_optimizer_assessment",
        "_materialize_registered_parameters",
        "_make_tensor_boundary_helpers",
        "_offline_environment",
        "_parameter_row_facts",
        "_payload_seal",
        "_resolve_runtime_payload",
        "_safe_regular_file",
        "_scan_snapshot",
        "_semantic_class",
        "_sha256_bytes",
        "_sha256_file",
        "_snapshot_manifest_sha256",
        "_storage_key",
        "_tensor_boundary_types",
        "_tensor_storage_facts",
        "_tensor_bytes",
        "_unsealed_optimizer_candidate_bytes",
        "_validate_candidate_receipt",
        "_validate_private_runtime_receipt",
        "_validate_runtime_environment",
        "clip_resolution",
        "contracts",
        "hashlib",
        "http",
        "initialization",
        "inspect",
        "json",
        "os",
        "platform",
        "readiness",
        "socket",
        "stat",
        "sys",
        "tempfile",
        "threading",
        "urllib",
        "warnings",
        "weakref",
    )
    guarded_module_bindings = tuple(
        (name, module_globals[name]) for name in guarded_module_names
    )
    guarded_mutable_values = tuple(
        (
            name,
            _canonical(module_globals[name], final_lf=False),
        )
        for name in (
            "_EXPECTED_CONFIG",
            "_EXPECTED_GOLDENS",
            "_EXPECTED_RUNTIME",
            "_EXPECTED_SOURCE",
        )
    )
    guarded_external_bindings = (
        (
            clip_resolution,
            "PINNED_FILES",
            clip_resolution.PINNED_FILES,
        ),
        (clip_resolution, "TOTAL_BYTES", clip_resolution.TOTAL_BYTES),
        (
            clip_resolution,
            "assess_local_clip_snapshot",
            clip_resolution.assess_local_clip_snapshot,
        ),
        (
            clip_resolution,
            "canonical_assessment_bytes",
            clip_resolution.canonical_assessment_bytes,
        ),
        (contracts, "ParameterRow", contracts.ParameterRow),
        (contracts, "FROZEN", contracts.FROZEN),
        (
            contracts,
            "PRETRAINED_PROJECTION_COMPONENT",
            contracts.PRETRAINED_PROJECTION_COMPONENT,
        ),
        (contracts, "CLIP_TEXT_COMPONENT", contracts.CLIP_TEXT_COMPONENT),
        (contracts, "MATRIX_WEIGHT", contracts.MATRIX_WEIGHT),
        (contracts, "EMBEDDING_WEIGHT", contracts.EMBEDDING_WEIGHT),
        (
            contracts,
            "QUERY_OR_OTHER_WEIGHT",
            contracts.QUERY_OR_OTHER_WEIGHT,
        ),
        (contracts, "BIAS", contracts.BIAS),
        (contracts, "LAYERNORM_GAMMA", contracts.LAYERNORM_GAMMA),
        (contracts, "LAYERNORM_BETA", contracts.LAYERNORM_BETA),
        (contracts, "DECAY_CLASSES", contracts.DECAY_CLASSES),
        (contracts, "NO_DECAY_CLASSES", contracts.NO_DECAY_CLASSES),
        (
            contracts,
            "validate_resolved_text_rows",
            contracts.validate_resolved_text_rows,
        ),
        (contracts, "validate_frozen_rows", contracts.validate_frozen_rows),
        (
            contracts,
            "expected_full_trainable_rows",
            contracts.expected_full_trainable_rows,
        ),
        (
            contracts,
            "validate_optimizer_partition",
            contracts.validate_optimizer_partition,
        ),
        (
            contracts,
            "canonical_inventory_json_bytes",
            contracts.canonical_inventory_json_bytes,
        ),
        (
            contracts,
            "canonical_name_list_bytes",
            contracts.canonical_name_list_bytes,
        ),
        (
            initialization,
            "_validate_model",
            initialization._validate_model,
        ),
        (
            readiness,
            "assess_training_readiness",
            readiness.assess_training_readiness,
        ),
    )
    assessment_fields = (
        "status",
        "authority",
        "production",
        "training_authorized",
        "architecture",
        "full_tensor_count",
        "full_numel",
        "decay_tensor_count",
        "decay_numel",
        "no_decay_tensor_count",
        "no_decay_numel",
        "inventory_sha256",
        "decay_names_sha256",
        "no_decay_names_sha256",
        "parameter_binding_sha256",
        "missing_gates",
        "adamw_constructor_count",
    )

    def module_invariants() -> None:
        for name, expected in guarded_module_bindings:
            if module_globals.get(name) is not expected:
                raise _hold_type("HOLD_LIVE_CLIP_MODULE_REBOUND")
        for name, expected_bytes in guarded_mutable_values:
            current = module_globals.get(name)
            if (
                type(current) is not dict
                or _canonical(current, final_lf=False) != expected_bytes
            ):
                raise _hold_type("HOLD_LIVE_CLIP_MODULE_REBOUND")
        for owner, name, expected in guarded_external_bindings:
            if getattr(owner, name, None) is not expected:
                raise _hold_type("HOLD_LIVE_CLIP_DEPENDENCY_REBOUND")

    def payload_invariants(payload: object) -> _RuntimePayload:
        if type(payload) is not _payload_type:
            raise _hold_type("HOLD_LOCAL_CLIP_PAYLOAD_TYPE")
        if (
            payload.preflight_sha256 != expected_preflight
            or payload.private_runtime_sha256 != expected_private_runtime
            or payload.rights_sha256 != expected_rights
            or payload.parameter_manifest_sha256 != expected_parameter_manifest
            or payload.buffer_manifest_sha256 != expected_buffer_manifest
            or len(payload.text_rows) != expected_text_count
            or sum(row.numel for row in payload.text_rows) != expected_text_numel
            or len(payload.text_parameters) != expected_text_count
            or len(payload.frozen_rows) != 1
        ):
            raise _hold_type("HOLD_LOCAL_CLIP_PAYLOAD_INVARIANT")
        return payload

    def record_for(lease: object) -> _LeaseRecord:
        if type(lease) is not _lease_type:
            raise _hold_type("HOLD_LOCAL_CLIP_LEASE_NOT_ISSUED")
        with registry_lock:
            record = issued.get(lease)
        if record is None:
            raise _hold_type("HOLD_LOCAL_CLIP_LEASE_NOT_ISSUED")
        return record

    def assessment_values(
        assessment: OptimizerCandidateAssessment,
    ) -> tuple[object, ...]:
        try:
            return tuple(
                object.__getattribute__(assessment, name)
                for name in assessment_fields
            )
        except AttributeError as exc:
            raise _hold_type("HOLD_OPTIMIZER_ASSESSMENT_MUTATED") from exc

    def burn_locked(
        lease: VerifiedLocalClipTextLease, record: _LeaseRecord
    ) -> None:
        record.state = burned_state
        record.payload = None
        try:
            object.__setattr__(lease, "_seal", b"")
        except (AttributeError, TypeError):
            pass

    def validate_locked(
        lease: VerifiedLocalClipTextLease, record: _LeaseRecord
    ) -> _RuntimePayload:
        try:
            observed_lock = object.__getattribute__(lease, "_lock")
            observed_nonce = object.__getattribute__(lease, "_nonce")
            observed_seal = object.__getattribute__(lease, "_seal")
        except AttributeError as exc:
            raise _hold_type("HOLD_LOCAL_CLIP_LEASE_MUTATED") from exc
        payload = record.payload
        if (
            record.owner_id != id(lease)
            or record.lock is not observed_lock
            or record.nonce != observed_nonce
            or type(observed_seal) is not bytes
            or record.seal != observed_seal
            or payload is None
            or _seal(payload) != record.seal
        ):
            raise _hold_type("HOLD_LOCAL_CLIP_LEASE_MUTATED")
        return payload_invariants(payload)

    def resolve(
        snapshot_root: Path,
        wheelhouse_root: Path,
        rights_receipt: Path,
    ) -> VerifiedLocalClipTextLease:
        nonlocal next_nonce
        with load_lock:
            module_invariants()
            payload = payload_invariants(
                _resolver(snapshot_root, wheelhouse_root, rights_receipt)
            )
            module_invariants()
            seal = _seal(payload)
            lease = object.__new__(_lease_type)
            lock = _rlock_factory()
            with registry_lock:
                nonce = next_nonce
                next_nonce += 1
                object.__setattr__(lease, "_lock", lock)
                object.__setattr__(lease, "_nonce", nonce)
                object.__setattr__(lease, "_seal", seal)
                issued[lease] = _record_type(
                    id(lease), issued_state, nonce, lock, seal, payload
                )
            return lease

    def candidate_receipt(lease: VerifiedLocalClipTextLease) -> bytes:
        record = record_for(lease)
        trusted_lock = record.lock
        try:
            with trusted_lock:
                if record.state != issued_state:
                    raise _hold_type(
                        "HOLD_LOCAL_CLIP_LEASE_NOT_ISSUED_STATE"
                    )
                try:
                    module_invariants()
                    payload = validate_locked(lease, record)
                    if _scan(payload.snapshot_root) != payload.snapshot_rows:
                        raise _hold_type("HOLD_CLIP_SNAPSHOT_TOCTOU")
                    receipt_payload = _candidate_payload(payload)
                    if (
                        type(receipt_payload) is not dict
                        or receipt_payload.get("authority") != 0
                        or receipt_payload.get("production") is not False
                        or receipt_payload.get("training_authorized") is not False
                        or receipt_payload.get("status")
                        != expected_candidate_status
                        or receipt_payload.get("text_trainable_tensor_count")
                        != expected_text_count
                        or receipt_payload.get("text_trainable_numel")
                        != expected_text_numel
                        or receipt_payload.get("adamw_constructor_count") != 0
                    ):
                        raise _hold_type("HOLD_CANDIDATE_RECEIPT_SEMANTICS")
                    receipt = _canonical(receipt_payload)
                    validate_locked(lease, record)
                    module_invariants()
                    return receipt
                except BaseException:
                    burn_locked(lease, record)
                    raise
        except BaseException:
            if record.state in (issued_state, validating_state):
                burn_locked(lease, record)
            raise

    def prepare(
        lease: VerifiedLocalClipTextLease,
        *,
        architecture: str,
        motion_model: object,
        project_weight: object,
        project_bias: object,
        logit_scale: object,
        gate_references: readiness.TrainingGateReferences,
    ) -> OptimizerCandidateAssessment:
        record = record_for(lease)
        trusted_lock = record.lock
        try:
            with trusted_lock:
                if record.state != issued_state:
                    raise _hold_type(
                        "HOLD_LOCAL_CLIP_LEASE_NOT_ISSUED_STATE"
                    )
                record.state = validating_state
                try:
                    module_invariants()
                    payload = validate_locked(lease, record)
                    if _scan(payload.snapshot_root) != payload.snapshot_rows:
                        raise _hold_type("HOLD_CLIP_SNAPSHOT_TOCTOU")
                    import torch

                    validated_motion = _validate_motion(motion_model)
                    if validated_motion.architecture != architecture:
                        raise _hold_type(
                            "HOLD_MOTION_ARCHITECTURE_MISMATCH"
                        )
                    motion_bindings = tuple(validated_motion.pairs)
                    project_weight_checked = _exact_parameter_impl(
                        project_weight,
                        shape=(512, 512),
                        torch=torch,
                        label="TEXT_PROJECT_WEIGHT",
                    )
                    project_bias_checked = _exact_parameter_impl(
                        project_bias,
                        shape=(512,),
                        torch=torch,
                        label="TEXT_PROJECT_BIAS",
                    )
                    logit_scale_checked = _exact_parameter_impl(
                        logit_scale,
                        shape=(1,),
                        torch=torch,
                        label="TEXT_LOGIT_SCALE",
                    )
                    full_rows = _expected_rows(
                        architecture, payload.text_rows, 512
                    )
                    partition = _validate_partition(
                        architecture,
                        full_trainable_rows=full_rows,
                        resolved_text_rows=payload.text_rows,
                        hidden_size=512,
                        frozen_rows=payload.frozen_rows,
                    )
                    parameter_map = dict(motion_bindings)
                    if len(parameter_map) != len(motion_bindings):
                        raise _hold_type(
                            "HOLD_MOTION_PARAMETER_BINDING_DUPLICATE"
                        )
                    parameter_map.update(dict(payload.text_parameters))
                    parameter_map.update(
                        {
                            "text.logit_scale": logit_scale_checked,
                            "text.project.bias": project_bias_checked,
                            "text.project.weight": project_weight_checked,
                        }
                    )
                    expected_names = tuple(
                        row.name for row in partition.full_trainable_rows
                    )
                    if set(parameter_map) != set(expected_names):
                        raise _hold_type(
                            "HOLD_FULL_PARAMETER_BINDING_COVERAGE"
                        )
                    ordered = tuple(
                        (name, parameter_map[name]) for name in expected_names
                    )
                    for row, (name, parameter) in zip(
                        partition.full_trainable_rows, ordered
                    ):
                        _row_facts(
                            parameter,
                            shape=row.shape,
                            numel=row.numel,
                        )
                        if (
                            row.name != name
                            or tuple(parameter.shape) != row.shape
                            or parameter.requires_grad is not True
                        ):
                            raise _hold_type("HOLD_PARAMETER_ROW_PROVENANCE")
                    binding_sha256 = _binding(ordered)
                    second_motion = _validate_motion(motion_model)
                    if tuple(
                        (name, id(parameter))
                        for name, parameter in second_motion.pairs
                    ) != tuple(
                        (name, id(parameter))
                        for name, parameter in motion_bindings
                    ):
                        raise _hold_type("HOLD_MOTION_PARAMETER_TOCTOU")
                    readiness_assessment = _readiness_assess(gate_references)
                    missing = readiness_assessment.missing_gates
                    if not missing:
                        missing = (
                            "UNVERIFIED_REFERENCES_NO_AUTHORITY_ISSUER",
                        )
                    result = _make_assessment(
                        partition, binding_sha256, missing
                    )
                    if type(result) is not _assessment_type:
                        raise _hold_type("HOLD_OPTIMIZER_ASSESSMENT_TYPE")
                    if (
                        result.status != expected_optimizer_status
                        or result.authority != 0
                        or result.production is not False
                        or result.training_authorized is not False
                        or result.adamw_constructor_count != 0
                    ):
                        raise _hold_type("HOLD_OPTIMIZER_ASSESSMENT_SEMANTICS")
                    result_bytes = _assessment_bytes(result)
                    result_values = assessment_values(result)
                    with registry_lock:
                        assessments[result] = (
                            id(result),
                            result_values,
                            result_bytes,
                            _sha256_factory(result_bytes).digest(),
                        )
                    validate_locked(lease, record)
                    module_invariants()
                    burn_locked(lease, record)
                    return result
                except BaseException:
                    burn_locked(lease, record)
                    raise
        except BaseException:
            if record.state in (issued_state, validating_state):
                burn_locked(lease, record)
            raise

    def optimizer_receipt(
        assessment: OptimizerCandidateAssessment,
    ) -> bytes:
        if type(assessment) is not _assessment_type:
            raise _hold_type("HOLD_OPTIMIZER_ASSESSMENT_NOT_ISSUED")
        with registry_lock:
            record = assessments.get(assessment)
        if record is None:
            raise _hold_type("HOLD_OPTIMIZER_ASSESSMENT_NOT_ISSUED")
        try:
            module_invariants()
            values = assessment_values(assessment)
            if (
                record[0] != id(assessment)
                or record[1] != values
                or type(record[2]) is not bytes
                or _sha256_factory(record[2]).digest() != record[3]
            ):
                raise _hold_type("HOLD_OPTIMIZER_ASSESSMENT_MUTATED")
            module_invariants()
            return record[2]
        except BaseException:
            with registry_lock:
                assessments.pop(assessment, None)
            raise

    def state(lease: VerifiedLocalClipTextLease) -> int:
        record = record_for(lease)
        trusted_lock = record.lock
        try:
            with trusted_lock:
                module_invariants()
                observed = record.state
                if type(observed) is not int or observed not in (
                    issued_state,
                    validating_state,
                    burned_state,
                ):
                    raise AssertionError("live CLIP lease registry is malformed")
                if observed in (issued_state, validating_state):
                    validate_locked(lease, record)
                else:
                    try:
                        observed_lock = object.__getattribute__(lease, "_lock")
                        observed_nonce = object.__getattribute__(lease, "_nonce")
                        observed_seal = object.__getattribute__(lease, "_seal")
                    except AttributeError as exc:
                        raise _hold_type("HOLD_LOCAL_CLIP_LEASE_MUTATED") from exc
                    if (
                        record.owner_id != id(lease)
                        or record.lock is not observed_lock
                        or record.nonce != observed_nonce
                        or observed_seal != b""
                        or record.payload is not None
                    ):
                        raise _hold_type("HOLD_LOCAL_CLIP_LEASE_MUTATED")
                module_invariants()
                return observed
        except BaseException:
            burn_locked(lease, record)
            raise

    def lease_count() -> int:
        with registry_lock:
            return len(issued)

    def constructor_count() -> int:
        return adamw_count

    return (
        resolve,
        candidate_receipt,
        prepare,
        optimizer_receipt,
        state,
        lease_count,
        constructor_count,
    )


(
    resolve_local_clip_text_candidate,
    canonical_local_clip_candidate_receipt,
    prepare_local_optimizer_candidate,
    canonical_optimizer_candidate_bytes,
    local_clip_lease_state,
    _live_lease_count_for_tests,
    _adamw_constructor_count_for_tests,
) = _seal_live_clip_lifecycle()
del _seal_live_clip_lifecycle


__all__ = [
    "OptimizerCandidateAssessment",
    "VerifiedLocalClipTextLease",
    "canonical_local_clip_candidate_receipt",
    "canonical_optimizer_candidate_bytes",
    "local_clip_lease_state",
    "prepare_local_optimizer_candidate",
    "resolve_local_clip_text_candidate",
]
