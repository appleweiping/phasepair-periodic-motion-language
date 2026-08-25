"""DATA_FREE_NONPRODUCTION PhasePair static contract primitives.

This module validates inventories and produces static optimizer manifests only.
It does not instantiate an optimizer, load a checkpoint, or claim that runtime
receipt evidence exists.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from types import MappingProxyType
from typing import Iterable, Mapping, Optional, Tuple


STATIC_STATUS = "STATIC_CONTRACT_VALIDATED_NONPRODUCTION"
HOLD_STATUS = "HOLD_RESOLVED_TEXT_INVENTORY"

MATRIX_WEIGHT = "MATRIX_WEIGHT"
EMBEDDING_WEIGHT = "EMBEDDING_WEIGHT"
QUERY_OR_OTHER_WEIGHT = "QUERY_OR_OTHER_WEIGHT"
BIAS = "BIAS"
LAYERNORM_GAMMA = "LAYERNORM_GAMMA"
LAYERNORM_BETA = "LAYERNORM_BETA"
LOGIT_SCALE = "LOGIT_SCALE"
FROZEN = "FROZEN"

SEMANTIC_CLASSES = frozenset(
    {
        MATRIX_WEIGHT,
        EMBEDDING_WEIGHT,
        QUERY_OR_OTHER_WEIGHT,
        BIAS,
        LAYERNORM_GAMMA,
        LAYERNORM_BETA,
        LOGIT_SCALE,
        FROZEN,
    }
)
DECAY_CLASSES = frozenset(
    {MATRIX_WEIGHT, EMBEDDING_WEIGHT, QUERY_OR_OTHER_WEIGHT}
)
NO_DECAY_CLASSES = frozenset(
    {BIAS, LAYERNORM_GAMMA, LAYERNORM_BETA, LOGIT_SCALE}
)

MOTION_COMPONENT = "MOTION"
CLIP_TEXT_COMPONENT = "CLIP_TEXT_TRANSFORMER"
PROJECT_COMPONENT = "TEXT_PROJECT"
LOGIT_COMPONENT = "TEXT_LOGIT_SCALE"
PRETRAINED_PROJECTION_COMPONENT = "CLIP_PRETRAINED_TEXT_PROJECTION"
VISION_COMPONENT = "CLIP_VISION"

ARCHITECTURES = ("mime", "early", "late")
DECAY_WEIGHT_DECAY = "1e-4"
NO_DECAY_WEIGHT_DECAY = "0"
OPTIMIZER_GROUP_ORDER = ("decay", "no_decay")


class PhasePairContractError(ValueError):
    """Base class for fail-closed PhasePair contract errors."""


class HoldResolvedTextInventory(PhasePairContractError):
    """The resolved text receipt is absent, so optimizer closure must HOLD."""

    status = HOLD_STATUS


class DuplicateNameError(PhasePairContractError):
    pass


class MissingNameError(PhasePairContractError):
    pass


class UnknownNameError(PhasePairContractError):
    pass


class RowMismatchError(PhasePairContractError):
    pass


class OrderingError(PhasePairContractError):
    pass


class FrozenLeakError(PhasePairContractError):
    pass


def _require_exact_int(
    value: object, label: str, *, minimum: int = 0, maximum: int = (1 << 64) - 1
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be a built-in int")
    if value < minimum or value > maximum:
        raise ValueError(f"{label} must be in [{minimum}, {maximum}]")
    return value


def _require_exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be a built-in bool")
    return value


def _require_name(value: object, label: str = "name") -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be a built-in str")
    if not value or "\x00" in value or any(ord(ch) < 0x20 for ch in value):
        raise ValueError(f"{label} must be nonempty UTF-8 text without controls")
    encoded = value.encode("utf-8", "strict")
    if len(encoded) > 65535:
        raise ValueError(f"{label} UTF-8 encoding exceeds uint16 length")
    return value


def _require_ascii_name(value: object, label: str = "name") -> str:
    result = _require_name(value, label)
    try:
        result.encode("ascii", "strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{label} must be ASCII") from exc
    return result


def _require_shape(value: object, label: str = "shape") -> Tuple[int, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a built-in tuple")
    if not value:
        raise ValueError(f"{label} must not be empty")
    for index, dimension in enumerate(value):
        _require_exact_int(
            dimension,
            f"{label}[{index}]",
            minimum=1,
            maximum=(1 << 64) - 1,
        )
    return value


def require_raw_sha256(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{label} must be built-in bytes")
    if len(value) != 32:
        raise ValueError(f"{label} must contain exactly 32 bytes")
    return value


def _optional_raw_sha256(value: object, label: str) -> Optional[bytes]:
    if value is None:
        return None
    return require_raw_sha256(value, label)


def _numel(shape: Tuple[int, ...]) -> int:
    result = 1
    for dimension in shape:
        result *= dimension
        if result >= (1 << 64):
            raise ValueError("shape numel must be less than 2^64")
    return result


def utf8_name_key(name: str) -> bytes:
    return _require_name(name).encode("utf-8")


@dataclass(frozen=True, slots=True)
class ParameterRow:
    """One canonical parameter row; numel is derived, never trusted as input."""

    name: str
    shape: Tuple[int, ...]
    semantic_class: str
    component: str
    requires_grad: bool
    resolved_state_key: Optional[str] = None
    source_checkpoint_sha256: Optional[bytes] = None

    def __post_init__(self) -> None:
        _require_name(self.name)
        _require_shape(self.shape)
        _numel(self.shape)
        if type(self.semantic_class) is not str or self.semantic_class not in SEMANTIC_CLASSES:
            raise ValueError("semantic_class is not a closed PhasePair value")
        _require_ascii_name(self.component, "component")
        _require_exact_bool(self.requires_grad, "requires_grad")
        if self.resolved_state_key is not None:
            _require_name(self.resolved_state_key, "resolved_state_key")
        _optional_raw_sha256(
            self.source_checkpoint_sha256, "source_checkpoint_sha256"
        )
        if self.requires_grad and self.semantic_class == FROZEN:
            raise ValueError("FROZEN rows cannot require gradients")
        if not self.requires_grad and self.semantic_class != FROZEN:
            raise ValueError("non-trainable rows must use semantic_class FROZEN")

    @property
    def numel(self) -> int:
        return _numel(self.shape)


def _row(
    name: str,
    shape: Tuple[int, ...],
    semantic_class: str,
    *,
    component: str = MOTION_COMPONENT,
) -> ParameterRow:
    return ParameterRow(name, shape, semantic_class, component, True)


def _linear(rows: list[ParameterRow], prefix: str, out_dim: int, in_dim: int) -> None:
    rows.append(_row(f"{prefix}.weight", (out_dim, in_dim), MATRIX_WEIGHT))
    rows.append(_row(f"{prefix}.bias", (out_dim,), BIAS))


def _layer_norm(rows: list[ParameterRow], prefix: str, width: int = 512) -> None:
    rows.append(_row(f"{prefix}.weight", (width,), LAYERNORM_GAMMA))
    rows.append(_row(f"{prefix}.bias", (width,), LAYERNORM_BETA))


def _mime_motion_rows() -> Tuple[ParameterRow, ...]:
    rows: list[ParameterRow] = []
    for actor in ("a", "b"):
        _linear(rows, f"mime.input.{actor}.linear", 512, 262)
        _layer_norm(rows, f"mime.input.{actor}.ln")
        rows.append(
            _row(
                f"mime.input.actor_embedding.{actor}",
                (512,),
                EMBEDDING_WEIGHT,
            )
        )
    _linear(rows, "mime.relation.linear", 512, 799)
    _layer_norm(rows, "mime.relation.ln")
    rows.append(_row("mime.relation.type_embedding", (512,), EMBEDDING_WEIGHT))

    for block in range(4):
        bb = f"{block:02d}"
        for actor in ("a", "b"):
            for projection in ("q", "k", "v", "o"):
                _linear(
                    rows,
                    f"mime.block.{bb}.self.{actor}.attn.{projection}",
                    512,
                    512,
                )
            _layer_norm(rows, f"mime.block.{bb}.self.{actor}.ln")
        for direction in ("a_from_b", "b_from_a"):
            for projection in ("q", "k", "v", "o"):
                _linear(
                    rows,
                    f"mime.block.{bb}.cross.{direction}.attn.{projection}",
                    512,
                    512,
                )
            for suffix in ("q", "kv"):
                _layer_norm(
                    rows, f"mime.block.{bb}.cross.{direction}.ln_{suffix}"
                )
        for actor in ("a", "b"):
            _linear(rows, f"mime.block.{bb}.ffn.{actor}.in", 2048, 512)
            _linear(rows, f"mime.block.{bb}.ffn.{actor}.out", 512, 2048)
            _layer_norm(rows, f"mime.block.{bb}.ffn.{actor}.ln")

    _linear(rows, "mime.fusion.in", 512, 1024)
    _linear(rows, "mime.fusion.out", 512, 512)
    rows.append(_row("mime.pool.query", (512,), QUERY_OR_OTHER_WEIGHT))
    return tuple(rows)


def _early_motion_rows() -> Tuple[ParameterRow, ...]:
    rows: list[ParameterRow] = []
    _linear(rows, "tmr.input.linear", 512, 786)
    _layer_norm(rows, "tmr.input.ln")
    for block in range(4):
        bb = f"{block:02d}"
        for projection in ("q", "k", "v", "o"):
            _linear(
                rows,
                f"tmr.block.{bb}.self.attn.{projection}",
                512,
                512,
            )
        _layer_norm(rows, f"tmr.block.{bb}.self.ln")
        _linear(rows, f"tmr.block.{bb}.ffn.in", 2048, 512)
        _linear(rows, f"tmr.block.{bb}.ffn.out", 512, 2048)
        _layer_norm(rows, f"tmr.block.{bb}.ffn.ln")
    rows.append(_row("tmr.pool.query", (512,), QUERY_OR_OTHER_WEIGHT))
    return tuple(rows)


def _late_motion_rows() -> Tuple[ParameterRow, ...]:
    rows: list[ParameterRow] = []
    for actor in ("actor_a", "actor_b"):
        _linear(rows, f"{actor}.input_proj", 512, 262)
        _layer_norm(rows, f"{actor}.input_ln")
        for block in range(4):
            bb = f"{block:02d}"
            prefix = f"{actor}.blocks.{bb}"
            _layer_norm(rows, f"{prefix}.ln_attn")
            for projection in ("q_proj", "k_proj", "v_proj", "out_proj"):
                _linear(rows, f"{prefix}.self_attn.{projection}", 512, 512)
            _layer_norm(rows, f"{prefix}.ln_ffn")
            _linear(rows, f"{prefix}.ffn.fc1", 2048, 512)
            _linear(rows, f"{prefix}.ffn.fc2", 512, 2048)
        rows.append(_row(f"{actor}.pool.query", (512,), QUERY_OR_OTHER_WEIGHT))
    _linear(rows, "fusion.proj", 512, 1024)
    _layer_norm(rows, "fusion.ln")
    return tuple(rows)


_MOTION_ROWS: Mapping[str, Tuple[ParameterRow, ...]] = MappingProxyType(
    {
        "mime": _mime_motion_rows(),
        "early": _early_motion_rows(),
        "late": _late_motion_rows(),
    }
)


@dataclass(frozen=True, slots=True)
class PartitionCount:
    tensor_count: int
    numel: int

    def __post_init__(self) -> None:
        _require_exact_int(self.tensor_count, "tensor_count", minimum=0)
        _require_exact_int(self.numel, "numel", minimum=0)


@dataclass(frozen=True, slots=True)
class MotionPartitionConstant:
    decay: PartitionCount
    no_decay: PartitionCount
    total: PartitionCount


MOTION_PARTITION_CONSTANTS: Mapping[str, MotionPartitionConstant] = MappingProxyType(
    {
        "mime": MotionPartitionConstant(
            PartitionCount(89, 35_020_288),
            PartitionCount(155, 91_648),
            PartitionCount(244, 35_111_936),
        ),
        "early": MotionPartitionConstant(
            PartitionCount(26, 12_985_856),
            PartitionCount(43, 28_160),
            PartitionCount(69, 13_014_016),
        ),
        "late": MotionPartitionConstant(
            PartitionCount(53, 25_959_424),
            PartitionCount(89, 57_856),
            PartitionCount(142, 26_017_280),
        ),
    }
)


def _require_architecture(value: object) -> str:
    if type(value) is not str:
        raise TypeError("architecture must be a built-in str")
    if value not in ARCHITECTURES:
        raise ValueError(f"unknown architecture: {value!r}")
    return value


def canonical_motion_rows(architecture: str) -> Tuple[ParameterRow, ...]:
    """Return the immutable architecture-traversal motion inventory."""

    return _MOTION_ROWS[_require_architecture(architecture)]


def _partition_counts(rows: Iterable[ParameterRow]) -> MotionPartitionConstant:
    materialized = tuple(rows)
    decay_rows = [row for row in materialized if row.semantic_class in DECAY_CLASSES]
    no_decay_rows = [row for row in materialized if row.semantic_class in NO_DECAY_CLASSES]
    if len(decay_rows) + len(no_decay_rows) != len(materialized):
        raise RowMismatchError("trainable row has an unclassifiable semantic class")
    return MotionPartitionConstant(
        PartitionCount(len(decay_rows), sum(row.numel for row in decay_rows)),
        PartitionCount(len(no_decay_rows), sum(row.numel for row in no_decay_rows)),
        PartitionCount(
            len(decay_rows) + len(no_decay_rows),
            sum(row.numel for row in decay_rows) + sum(row.numel for row in no_decay_rows),
        ),
    )


def audit_motion_partition_constants() -> Mapping[str, MotionPartitionConstant]:
    """Mechanically derive every count and compare it with the frozen ledger."""

    derived = {name: _partition_counts(rows) for name, rows in _MOTION_ROWS.items()}
    for architecture, expected in MOTION_PARTITION_CONSTANTS.items():
        if derived[architecture] != expected:
            raise RowMismatchError(
                f"{architecture} motion partition does not match its frozen ledger"
            )
    return MappingProxyType(derived)


def _require_row_tuple(value: object, label: str) -> Tuple[ParameterRow, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be a built-in tuple")
    rebuilt: list[ParameterRow] = []
    for index, row in enumerate(value):
        if type(row) is not ParameterRow:
            raise TypeError(f"{label}[{index}] must be exactly ParameterRow")
        # A frozen dataclass can still be assembled through object.__new__.
        # Reconstruct every public leaf so bool/int equality and forged slots
        # cannot bypass the exact-type checks in ParameterRow.__post_init__.
        rebuilt.append(
            ParameterRow(
                row.name,
                row.shape,
                row.semantic_class,
                row.component,
                row.requires_grad,
                row.resolved_state_key,
                row.source_checkpoint_sha256,
            )
        )
    return tuple(rebuilt)


def _reject_duplicate_names(rows: Tuple[ParameterRow, ...], label: str) -> None:
    seen: set[str] = set()
    duplicates: list[str] = []
    for row in rows:
        if row.name in seen:
            duplicates.append(row.name)
        seen.add(row.name)
    if duplicates:
        raise DuplicateNameError(f"{label} duplicate names: {sorted(set(duplicates))!r}")


def _require_utf8_order(rows: Tuple[ParameterRow, ...], label: str) -> None:
    names = tuple(row.name for row in rows)
    ordered = tuple(sorted(names, key=utf8_name_key))
    if names != ordered:
        raise OrderingError(f"{label} must be strictly ordered by UTF-8 name bytes")


def validate_motion_inventory(
    architecture: str, rows: Tuple[ParameterRow, ...]
) -> MotionPartitionConstant:
    """Reject any motion omission, insertion, duplicate, or row mutation."""

    expected = canonical_motion_rows(architecture)
    supplied = _require_row_tuple(rows, "motion_rows")
    _reject_duplicate_names(supplied, "motion_rows")
    expected_by_name = {row.name: row for row in expected}
    supplied_by_name = {row.name: row for row in supplied}
    missing = sorted(set(expected_by_name) - set(supplied_by_name), key=utf8_name_key)
    unknown = sorted(set(supplied_by_name) - set(expected_by_name), key=utf8_name_key)
    if missing:
        raise MissingNameError(f"motion inventory missing names: {missing!r}")
    if unknown:
        raise UnknownNameError(f"motion inventory unknown names: {unknown!r}")
    for name, expected_row in expected_by_name.items():
        if supplied_by_name[name] != expected_row:
            raise RowMismatchError(f"motion row mismatch: {name}")
    if supplied != expected:
        raise OrderingError("motion inventory must preserve canonical architecture traversal")
    derived = _partition_counts(supplied)
    if derived != MOTION_PARTITION_CONSTANTS[architecture]:
        raise RowMismatchError("motion inventory arithmetic mismatch")
    return derived


def validate_resolved_text_rows(
    rows: Optional[Tuple[ParameterRow, ...]], hidden_size: int
) -> Tuple[ParameterRow, ...]:
    """Validate the resolved CLIP text rows or raise an explicit HOLD."""

    _require_exact_int(hidden_size, "hidden_size", minimum=1, maximum=(1 << 32) - 1)
    if rows is None:
        raise HoldResolvedTextInventory("resolved CLIP text receipt is absent")
    checked = _require_row_tuple(rows, "resolved_text_rows")
    if not checked:
        raise HoldResolvedTextInventory("resolved CLIP text receipt has no trainable rows")
    _reject_duplicate_names(checked, "resolved_text_rows")
    _require_utf8_order(checked, "resolved_text_rows")
    for row in checked:
        if row.component != CLIP_TEXT_COMPONENT or not row.requires_grad:
            raise RowMismatchError("resolved text row component/requires_grad mismatch")
        if row.semantic_class not in DECAY_CLASSES | NO_DECAY_CLASSES:
            raise RowMismatchError("resolved text row semantic class is not trainable")
        if row.resolved_state_key is None:
            raise RowMismatchError("resolved text row lacks resolved_state_key")
        if row.name != f"text.clip.{row.resolved_state_key}":
            raise RowMismatchError("resolved text canonical-name mapping mismatch")
        if row.source_checkpoint_sha256 is None:
            raise HoldResolvedTextInventory("resolved text source checkpoint hash is absent")
    return checked


def validate_frozen_rows(rows: Tuple[ParameterRow, ...]) -> Tuple[ParameterRow, ...]:
    checked = _require_row_tuple(rows, "frozen_rows")
    _reject_duplicate_names(checked, "frozen_rows")
    _require_utf8_order(checked, "frozen_rows")
    projection_count = 0
    for row in checked:
        if row.requires_grad or row.semantic_class != FROZEN:
            raise FrozenLeakError("frozen row is marked trainable")
        if row.source_checkpoint_sha256 is None:
            raise HoldResolvedTextInventory("frozen checkpoint evidence hash is absent")
        if row.name == "text.pretrained_projection.weight":
            if row.component != PRETRAINED_PROJECTION_COMPONENT:
                raise RowMismatchError("pretrained projection component mismatch")
            projection_count += 1
        elif row.name.startswith("vision."):
            if row.component != VISION_COMPONENT:
                raise RowMismatchError("vision frozen component mismatch")
        else:
            raise UnknownNameError(f"unknown frozen row: {row.name}")
    if projection_count != 1:
        raise MissingNameError("exactly one frozen text.pretrained_projection.weight is required")
    return checked


def expected_full_trainable_rows(
    architecture: str,
    resolved_text_rows: Optional[Tuple[ParameterRow, ...]],
    hidden_size: int,
) -> Tuple[ParameterRow, ...]:
    architecture = _require_architecture(architecture)
    resolved = validate_resolved_text_rows(resolved_text_rows, hidden_size)
    project_rows = (
        ParameterRow(
            "text.project.weight",
            (512, hidden_size),
            MATRIX_WEIGHT,
            PROJECT_COMPONENT,
            True,
        ),
        ParameterRow(
            "text.project.bias", (512,), BIAS, PROJECT_COMPONENT, True
        ),
        ParameterRow("text.logit_scale", (1,), LOGIT_SCALE, LOGIT_COMPONENT, True),
    )
    rows = canonical_motion_rows(architecture) + resolved + project_rows
    return tuple(sorted(rows, key=lambda row: utf8_name_key(row.name)))


@dataclass(frozen=True, slots=True)
class OptimizerGroup:
    ordinal: int
    name: str
    weight_decay: str
    parameter_names: Tuple[str, ...]
    tensor_count: int
    numel: int

    def __post_init__(self) -> None:
        _require_exact_int(self.ordinal, "group ordinal", maximum=1)
        _require_ascii_name(self.name, "group name")
        if type(self.weight_decay) is not str:
            raise TypeError("weight_decay must be an exact decimal string")
        expected_name = OPTIMIZER_GROUP_ORDER[self.ordinal]
        expected_weight_decay = (
            DECAY_WEIGHT_DECAY
            if self.ordinal == 0
            else NO_DECAY_WEIGHT_DECAY
        )
        if self.name != expected_name:
            raise RowMismatchError(
                f"optimizer group {self.ordinal} must be named {expected_name!r}"
            )
        if self.weight_decay != expected_weight_decay:
            raise RowMismatchError(
                f"optimizer group {self.ordinal} has the wrong weight decay"
            )
        if type(self.parameter_names) is not tuple:
            raise TypeError("parameter_names must be a built-in tuple")
        for index, name in enumerate(self.parameter_names):
            _require_name(name, f"parameter_names[{index}]")
        if tuple(sorted(self.parameter_names, key=utf8_name_key)) != self.parameter_names:
            raise OrderingError("optimizer group names must use UTF-8 byte order")
        if len(self.parameter_names) != len(set(self.parameter_names)):
            raise DuplicateNameError("optimizer group contains a duplicate name")
        _require_exact_int(self.tensor_count, "tensor_count", minimum=0)
        _require_exact_int(self.numel, "numel", minimum=0)
        if self.tensor_count != len(self.parameter_names):
            raise RowMismatchError("tensor_count does not match parameter_names")


@dataclass(frozen=True, slots=True)
class OptimizerPartition:
    architecture: str
    status: str
    full_trainable_rows: Tuple[ParameterRow, ...]
    groups: Tuple[OptimizerGroup, OptimizerGroup]

    def __post_init__(self) -> None:
        _validate_optimizer_partition_object(self)

    @property
    def total_numel(self) -> int:
        canonical_rows, _ = _validate_optimizer_partition_object(self)
        return sum(row.numel for row in canonical_rows)


def _canonical_optimizer_groups(
    rows: Tuple[ParameterRow, ...],
) -> Tuple[OptimizerGroup, OptimizerGroup]:
    decay_rows = tuple(row for row in rows if row.semantic_class in DECAY_CLASSES)
    no_decay_rows = tuple(
        row for row in rows if row.semantic_class in NO_DECAY_CLASSES
    )
    if len(decay_rows) + len(no_decay_rows) != len(rows):
        raise RowMismatchError("full trainable inventory has an unclassified row")
    decay_names = tuple(row.name for row in decay_rows)
    no_decay_names = tuple(row.name for row in no_decay_rows)
    if set(decay_names) & set(no_decay_names):
        raise DuplicateNameError("optimizer group intersection is nonempty")
    if set(decay_names) | set(no_decay_names) != {row.name for row in rows}:
        raise MissingNameError("optimizer group union is incomplete")
    return (
        OptimizerGroup(
            0,
            "decay",
            DECAY_WEIGHT_DECAY,
            decay_names,
            len(decay_rows),
            sum(row.numel for row in decay_rows),
        ),
        OptimizerGroup(
            1,
            "no_decay",
            NO_DECAY_WEIGHT_DECAY,
            no_decay_names,
            len(no_decay_rows),
            sum(row.numel for row in no_decay_rows),
        ),
    )


def _validate_optimizer_partition_object(
    partition: OptimizerPartition,
) -> Tuple[Tuple[ParameterRow, ...], Tuple[OptimizerGroup, OptimizerGroup]]:
    """Re-derive a public partition from its rows; trust no stored summary."""

    if type(partition) is not OptimizerPartition:
        raise TypeError("partition must be exactly OptimizerPartition")
    architecture = _require_architecture(partition.architecture)
    if type(partition.status) is not str or partition.status != STATIC_STATUS:
        raise RowMismatchError("optimizer partition status is not the static status")
    supplied = _require_row_tuple(
        partition.full_trainable_rows, "full_trainable_rows"
    )
    _reject_duplicate_names(supplied, "full_trainable_rows")
    _require_utf8_order(supplied, "full_trainable_rows")

    project_weight = tuple(
        row for row in supplied if row.name == "text.project.weight"
    )
    if len(project_weight) != 1:
        raise MissingNameError("exactly one text.project.weight row is required")
    if (
        project_weight[0].component != PROJECT_COMPONENT
        or project_weight[0].semantic_class != MATRIX_WEIGHT
        or not project_weight[0].requires_grad
        or len(project_weight[0].shape) != 2
        or project_weight[0].shape[0] != 512
    ):
        raise RowMismatchError("text.project.weight cannot determine hidden_size")
    hidden_size = project_weight[0].shape[1]

    resolved_candidates = tuple(
        row
        for row in supplied
        if row.component == CLIP_TEXT_COMPONENT or row.name.startswith("text.clip.")
    )
    resolved = validate_resolved_text_rows(resolved_candidates, hidden_size)
    expected = expected_full_trainable_rows(architecture, resolved, hidden_size)
    if supplied != expected:
        supplied_by_name = {row.name: row for row in supplied}
        expected_by_name = {row.name: row for row in expected}
        missing = sorted(
            set(expected_by_name) - set(supplied_by_name), key=utf8_name_key
        )
        unknown = sorted(
            set(supplied_by_name) - set(expected_by_name), key=utf8_name_key
        )
        if missing:
            raise MissingNameError(f"full trainable inventory missing names: {missing!r}")
        if unknown:
            raise UnknownNameError(f"full trainable inventory unknown names: {unknown!r}")
        for name, expected_row in expected_by_name.items():
            if supplied_by_name[name] != expected_row:
                raise RowMismatchError(f"full trainable row mismatch: {name}")
        raise OrderingError("full trainable inventory is not in UTF-8 byte order")

    if type(partition.groups) is not tuple or len(partition.groups) != 2:
        raise TypeError("groups must be a built-in tuple containing exactly two groups")
    rebuilt_groups: list[OptimizerGroup] = []
    for index, group in enumerate(partition.groups):
        if type(group) is not OptimizerGroup:
            raise TypeError(f"groups[{index}] must be exactly OptimizerGroup")
        rebuilt_groups.append(
            OptimizerGroup(
                group.ordinal,
                group.name,
                group.weight_decay,
                group.parameter_names,
                group.tensor_count,
                group.numel,
            )
        )
    canonical_groups = _canonical_optimizer_groups(supplied)
    if tuple(rebuilt_groups) != canonical_groups:
        raise RowMismatchError("partition groups are not canonically derived from rows")
    return supplied, canonical_groups


def validate_optimizer_partition(
    architecture: str,
    *,
    full_trainable_rows: Tuple[ParameterRow, ...],
    resolved_text_rows: Optional[Tuple[ParameterRow, ...]],
    hidden_size: int,
    frozen_rows: Tuple[ParameterRow, ...],
) -> OptimizerPartition:
    """Validate the exact full union and return its only legal two groups.

    The API deliberately requires the caller's full trainable inventory so that
    missing, duplicate, unknown, and frozen-leak mutants cannot be hidden by an
    internally generated list.
    """

    architecture = _require_architecture(architecture)
    resolved = validate_resolved_text_rows(resolved_text_rows, hidden_size)
    frozen = validate_frozen_rows(frozen_rows)
    supplied = _require_row_tuple(full_trainable_rows, "full_trainable_rows")
    _reject_duplicate_names(supplied, "full_trainable_rows")
    _require_utf8_order(supplied, "full_trainable_rows")
    expected = expected_full_trainable_rows(architecture, resolved, hidden_size)

    supplied_by_name = {row.name: row for row in supplied}
    expected_by_name = {row.name: row for row in expected}
    frozen_names = {row.name for row in frozen}
    leaked = sorted(set(supplied_by_name) & frozen_names, key=utf8_name_key)
    if leaked:
        raise FrozenLeakError(f"frozen rows entered optimizer inventory: {leaked!r}")
    missing = sorted(set(expected_by_name) - set(supplied_by_name), key=utf8_name_key)
    unknown = sorted(set(supplied_by_name) - set(expected_by_name), key=utf8_name_key)
    if missing:
        raise MissingNameError(f"full trainable inventory missing names: {missing!r}")
    if unknown:
        raise UnknownNameError(f"full trainable inventory unknown names: {unknown!r}")
    for name, expected_row in expected_by_name.items():
        if supplied_by_name[name] != expected_row:
            raise RowMismatchError(f"full trainable row mismatch: {name}")

    groups = _canonical_optimizer_groups(supplied)
    return OptimizerPartition(architecture, STATIC_STATUS, supplied, groups)


def validate_optimizer_groups(
    partition: OptimizerPartition, supplied_groups: Tuple[OptimizerGroup, ...]
) -> None:
    """Reject group swap, wrong WD, wrong membership, or any extra group."""

    if type(partition) is not OptimizerPartition:
        raise TypeError("partition must be exactly OptimizerPartition")
    _, canonical_groups = _validate_optimizer_partition_object(partition)
    if type(supplied_groups) is not tuple:
        raise TypeError("supplied_groups must be a built-in tuple")
    rebuilt_groups: list[OptimizerGroup] = []
    for index, group in enumerate(supplied_groups):
        if type(group) is not OptimizerGroup:
            raise TypeError(f"supplied_groups[{index}] must be exactly OptimizerGroup")
        rebuilt_groups.append(
            OptimizerGroup(
                group.ordinal,
                group.name,
                group.weight_decay,
                group.parameter_names,
                group.tensor_count,
                group.numel,
            )
        )
    if tuple(rebuilt_groups) != canonical_groups:
        raise RowMismatchError("optimizer groups differ from the canonical two-group layout")


def _row_json_object(row: ParameterRow) -> dict[str, object]:
    return {
        "component": row.component,
        "name": row.name,
        "numel": row.numel,
        "requires_grad": row.requires_grad,
        "resolved_state_key": row.resolved_state_key,
        "semantic_class": row.semantic_class,
        "shape": list(row.shape),
        "source_checkpoint_sha256": (
            None
            if row.source_checkpoint_sha256 is None
            else row.source_checkpoint_sha256.hex()
        ),
    }


def canonical_inventory_json_bytes(rows: Tuple[ParameterRow, ...]) -> bytes:
    checked = _require_row_tuple(rows, "rows")
    _reject_duplicate_names(checked, "rows")
    _require_utf8_order(checked, "rows")
    return (
        json.dumps(
            [_row_json_object(row) for row in checked],
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_name_list_bytes(names: Tuple[str, ...]) -> bytes:
    if type(names) is not tuple:
        raise TypeError("names must be a built-in tuple")
    for index, name in enumerate(names):
        _require_name(name, f"names[{index}]")
    if names != tuple(sorted(names, key=utf8_name_key)):
        raise OrderingError("names must be in UTF-8 byte order")
    if len(names) != len(set(names)):
        raise DuplicateNameError("names contain duplicates")
    output = bytearray(b"phasepair-optimizer-name-list-v1\x00")
    output.extend(len(names).to_bytes(4, "big"))
    for name in names:
        raw = name.encode("utf-8")
        output.extend(len(raw).to_bytes(2, "big"))
        output.extend(raw)
    return bytes(output)


@dataclass(frozen=True, slots=True)
class OptimizerRuntimeEvidence:
    """Optional real-runtime digests; defaults are deliberately all absent."""

    optimizer_source_sha256: Optional[bytes] = None
    runtime_sha256: Optional[bytes] = None
    step0_state_sha256: Optional[bytes] = None
    first_step_state_sha256: Optional[bytes] = None

    def __post_init__(self) -> None:
        for label in (
            "optimizer_source_sha256",
            "runtime_sha256",
            "step0_state_sha256",
            "first_step_state_sha256",
        ):
            _optional_raw_sha256(getattr(self, label), label)

    @property
    def capture_status(self) -> str:
        values = (
            self.optimizer_source_sha256,
            self.runtime_sha256,
            self.step0_state_sha256,
            self.first_step_state_sha256,
        )
        if all(value is None for value in values):
            return "RUNTIME_EVIDENCE_NOT_CAPTURED"
        if all(value is not None for value in values):
            return "RUNTIME_EVIDENCE_CAPTURED_UNVERIFIED"
        return "RUNTIME_EVIDENCE_PARTIAL_UNVERIFIED"


@dataclass(frozen=True, slots=True, init=False)
class OptimizerReceiptDraft:
    """Static hashes plus nullable runtime evidence; never emits runtime PASS."""

    architecture: str
    static_status: str
    full_inventory_sha256: bytes
    decay_name_list_sha256: bytes
    no_decay_name_list_sha256: bytes
    runtime: OptimizerRuntimeEvidence

    def __init__(
        self,
        partition: OptimizerPartition,
        *,
        runtime: OptimizerRuntimeEvidence | None = None,
    ) -> None:
        """Derive every static digest; callers cannot inject summary hashes."""

        canonical_rows, canonical_groups = _validate_optimizer_partition_object(
            partition
        )
        if runtime is None:
            checked_runtime = OptimizerRuntimeEvidence()
        elif type(runtime) is OptimizerRuntimeEvidence:
            checked_runtime = OptimizerRuntimeEvidence(
                runtime.optimizer_source_sha256,
                runtime.runtime_sha256,
                runtime.step0_state_sha256,
                runtime.first_step_state_sha256,
            )
        else:
            raise TypeError("runtime must be exactly OptimizerRuntimeEvidence")
        object.__setattr__(self, "architecture", partition.architecture)
        object.__setattr__(self, "static_status", STATIC_STATUS)
        object.__setattr__(
            self,
            "full_inventory_sha256",
            hashlib.sha256(canonical_inventory_json_bytes(canonical_rows)).digest(),
        )
        object.__setattr__(
            self,
            "decay_name_list_sha256",
            hashlib.sha256(
                canonical_name_list_bytes(canonical_groups[0].parameter_names)
            ).digest(),
        )
        object.__setattr__(
            self,
            "no_decay_name_list_sha256",
            hashlib.sha256(
                canonical_name_list_bytes(canonical_groups[1].parameter_names)
            ).digest(),
        )
        object.__setattr__(self, "runtime", checked_runtime)

    @classmethod
    def from_partition(cls, partition: OptimizerPartition) -> "OptimizerReceiptDraft":
        return cls(partition)


__all__ = [
    "ARCHITECTURES",
    "BIAS",
    "CLIP_TEXT_COMPONENT",
    "DECAY_CLASSES",
    "DECAY_WEIGHT_DECAY",
    "DuplicateNameError",
    "EMBEDDING_WEIGHT",
    "FROZEN",
    "FrozenLeakError",
    "HOLD_STATUS",
    "HoldResolvedTextInventory",
    "LAYERNORM_BETA",
    "LAYERNORM_GAMMA",
    "LOGIT_SCALE",
    "MATRIX_WEIGHT",
    "MOTION_COMPONENT",
    "MOTION_PARTITION_CONSTANTS",
    "MissingNameError",
    "NO_DECAY_CLASSES",
    "NO_DECAY_WEIGHT_DECAY",
    "OPTIMIZER_GROUP_ORDER",
    "OptimizerGroup",
    "OptimizerPartition",
    "OptimizerReceiptDraft",
    "OptimizerRuntimeEvidence",
    "PRETRAINED_PROJECTION_COMPONENT",
    "PROJECT_COMPONENT",
    "ParameterRow",
    "PartitionCount",
    "PhasePairContractError",
    "QUERY_OR_OTHER_WEIGHT",
    "RowMismatchError",
    "STATIC_STATUS",
    "UnknownNameError",
    "VISION_COMPONENT",
    "audit_motion_partition_constants",
    "canonical_inventory_json_bytes",
    "canonical_motion_rows",
    "canonical_name_list_bytes",
    "expected_full_trainable_rows",
    "require_raw_sha256",
    "utf8_name_key",
    "validate_frozen_rows",
    "validate_motion_inventory",
    "validate_optimizer_groups",
    "validate_optimizer_partition",
    "validate_resolved_text_rows",
]
