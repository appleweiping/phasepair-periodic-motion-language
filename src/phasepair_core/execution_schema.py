"""Data-free PhasePair v2 execution schemas and qualification state machine.

This module is deliberately standard-library-only and ``AUTHORITY0``.  It
validates canonical execution identities and immutable JSON artifacts; it does
not construct an optimizer, run a training step, write a checkpoint, mint a
production capability, or make a result claim.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from types import MappingProxyType
from typing import Any
from weakref import WeakKeyDictionary


AUTHORITY = 0
PRODUCTION = False
EXECUTION_AUTHORIZED = False
RESULT_CLAIMED = False
STATUS = "DATA_FREE_EXECUTION_SCHEMA_AUTHORITY0_NO_RESULT"
_IDENTITY_VERSION_FAIL = "TRAINING_EXECUTION_IDENTITY_VERSION_FAIL"

SEEDS = (1729, 2718, 31415)
BASE_SYSTEMS = ("00", "07", "08")
RESIDUAL_SYSTEMS = ("01", "02", "03", "04", "05", "06")
BASE_ARCHITECTURES = MappingProxyType(
    {"00": "mime", "07": "early", "08": "late"}
)
RESIDUAL_NAMES = MappingProxyType(
    {
        "01": "generic",
        "02": "wamo_marginal_wavelet",
        "03": "interedit_mean_difference_dct",
        "04": "no_relation",
        "05": "phase_stripped",
        "06": "phasepair_full",
    }
)

NOT_APPLICABLE = "NOT_APPLICABLE"
QUALIFIED = "QUALIFIED"
NOT_QUALIFIED = "NOT_QUALIFIED"

_COMPLETION_KEYS = frozenset(
    {
        "architecture",
        "evaluator_sha256",
        "example_count",
        "log_sha256",
        "model_final_state_sha256",
        "optimizer_final_state_sha256",
        "optimizer_step_count",
        "run_id",
        "run_input_sha256",
        "schema",
        "seed",
        "selected_checkpoint_sha256",
        "selected_completed_epoch",
        "selected_epoch_index",
        "status",
        "system_id",
        "validation_metric_sha256",
    }
)
_QUALIFICATION_KEYS = frozenset(
    {
        "authority",
        "completion_sha256s",
        "early_mean_denominator",
        "early_mean_numerator",
        "late_mean_denominator",
        "late_mean_numerator",
        "mime_mean_denominator",
        "mime_mean_numerator",
        "minimum_delta_denominator",
        "minimum_delta_numerator",
        "production",
        "result_claimed",
        "same_seed_win_count",
        "schema",
        "score_rows_sha256",
        "status",
        "verdict",
    }
)
_TERMINAL_KEYS = frozenset(
    {
        "authority",
        "completion_sha256",
        "production",
        "qualification_sha256",
        "result_claimed",
        "run_id",
        "run_input_sha256",
        "schema",
        "seed",
        "selected_checkpoint_sha256",
        "status",
        "system_id",
        "terminal_outcome",
        "training_frozen_base_cache_sha256",
    }
)
_CACHE_KEYS = frozenset(
    {
        "authority",
        "base_terminal_sha256",
        "cache_sha256",
        "production",
        "result_claimed",
        "run_id",
        "schema",
        "seed",
        "status",
        "system_id",
    }
)


class ExecutionSchemaError(ValueError):
    """A closed v2 execution-schema invariant was violated."""


def _exact_str(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    maximum: int = (1 << 63) - 1,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < minimum or value > maximum:
        raise ExecutionSchemaError(f"{label} is outside [{minimum},{maximum}]")
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact built-in bool")
    return value


def _sha256(value: object, label: str) -> str:
    checked = _exact_str(value, label)
    if len(checked) != 64 or any(character not in "0123456789abcdef" for character in checked):
        raise ExecutionSchemaError(f"{label} must be lowercase SHA-256 hex")
    return checked


def _json_tree(value: object, label: str = "value") -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is list:
        return [_json_tree(item, f"{label}[{index}]") for index, item in enumerate(value)]
    if type(value) is dict:
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be exact built-in str")
            output[key] = _json_tree(item, f"{label}.{key}")
        return output
    raise TypeError(f"{label} contains a non-canonical JSON type")


def canonical_json_bytes(
    value: object,
    _dumps: Any = json.dumps,
) -> bytes:
    """Return strict sorted-key UTF-8 JSON with exactly one terminal LF."""

    checked = _json_tree(value)
    return (
        _dumps(
            checked,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _reject_float(value: str) -> object:
    raise ExecutionSchemaError(f"JSON floating-point value is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ExecutionSchemaError(f"JSON constant is forbidden: {value}")


def _pairs_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise ExecutionSchemaError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _parse_canonical_object(
    raw: object,
    label: str,
    _loads: Any = json.loads,
    _canonical: Any = canonical_json_bytes,
) -> dict[str, object]:
    if type(raw) is not bytes:
        raise TypeError(f"{label} must be exact built-in bytes")
    if raw.startswith(b"\xef\xbb\xbf"):
        raise ExecutionSchemaError(f"{label} must not contain a UTF-8 BOM")
    try:
        text = raw.decode("utf-8", "strict")
        parsed = _loads(
            text,
            object_pairs_hook=_pairs_object,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExecutionSchemaError(f"{label} is not strict UTF-8 JSON") from exc
    if type(parsed) is not dict:
        raise ExecutionSchemaError(f"{label} must contain one JSON object")
    if _canonical(parsed) != raw:
        raise ExecutionSchemaError(f"{label} bytes are not canonical sorted JSON plus LF")
    return parsed


def _require_keys(
    payload: dict[str, object], expected: frozenset[str], label: str
) -> None:
    if set(payload) != expected:
        missing = sorted(expected - set(payload))
        extra = sorted(set(payload) - expected)
        raise ExecutionSchemaError(
            f"{label} keys are not closed; missing={missing!r}, extra={extra!r}"
        )


def base_run_ids(
    _systems: tuple[str, ...] = ("00", "07", "08"),
    _seeds: tuple[int, ...] = (1729, 2718, 31415),
) -> tuple[str, ...]:
    return tuple(
        f"phasepair-run-v2/BASE_TRAIN/{seed}/{system}"
        for system in _systems
        for seed in _seeds
    )


def residual_run_ids(
    _systems: tuple[str, ...] = ("01", "02", "03", "04", "05", "06"),
    _seeds: tuple[int, ...] = (1729, 2718, 31415),
) -> tuple[str, ...]:
    return tuple(
        f"phasepair-run-v2/RESIDUAL_HEAD_TRAIN/{seed}/{system}"
        for system in _systems
        for seed in _seeds
    )


def parse_run_id(
    run_id: object,
    _seeds: tuple[int, ...] = (1729, 2718, 31415),
    _base_systems: tuple[str, ...] = ("00", "07", "08"),
    _residual_systems: tuple[str, ...] = ("01", "02", "03", "04", "05", "06"),
    _identity_failure: str = _IDENTITY_VERSION_FAIL,
) -> tuple[str, int, str]:
    checked = _exact_str(run_id, "run_id")
    parts = checked.split("/")
    if len(parts) != 4 or parts[0] != "phasepair-run-v2":
        raise ExecutionSchemaError(
            f"{_identity_failure}: run_id must use the exact phasepair-run-v2 form"
        )
    role = parts[1]
    if parts[2] not in {str(seed) for seed in _seeds}:
        raise ExecutionSchemaError(
            f"{_identity_failure}: run_id seed is outside the frozen census"
        )
    seed = int(parts[2])
    system = parts[3]
    if role == "BASE_TRAIN":
        allowed = _base_systems
    elif role == "RESIDUAL_HEAD_TRAIN":
        allowed = _residual_systems
    else:
        raise ExecutionSchemaError(
            f"{_identity_failure}: run_id role is outside the v2 execution census"
        )
    if system not in allowed:
        raise ExecutionSchemaError(
            f"{_identity_failure}: run_id system is outside its v2 role census"
        )
    expected = f"phasepair-run-v2/{role}/{seed}/{system}"
    if checked != expected:
        raise ExecutionSchemaError(
            f"{_identity_failure}: run_id is not in exact canonical form"
        )
    return role, seed, system


def canonical_execution_census_bytes(
    _base_ids: Any = base_run_ids,
    _residual_ids: Any = residual_run_ids,
    _canonical: Any = canonical_json_bytes,
    _seeds: tuple[int, ...] = (1729, 2718, 31415),
    _base_systems: tuple[str, ...] = ("00", "07", "08"),
    _residual_systems: tuple[str, ...] = ("01", "02", "03", "04", "05", "06"),
) -> bytes:
    payload = {
        "authority": 0,
        "base_run_count": 9,
        "base_run_ids": list(_base_ids()),
        "base_systems": list(_base_systems),
        "execution_authorized": False,
        "production": False,
        "residual_run_count": 18,
        "residual_run_ids": list(_residual_ids()),
        "residual_systems": list(_residual_systems),
        "result_claimed": False,
        "schema": "phasepair-training-execution-census-v2",
        "seeds": list(_seeds),
        "status": "DATA_FREE_EXECUTION_SCHEMA_AUTHORITY0_NO_RESULT",
    }
    return _canonical(payload)


def validate_execution_census(
    raw: object,
    _parse: Any = _parse_canonical_object,
    _expected_builder: Any = canonical_execution_census_bytes,
    _identity_failure: str = _IDENTITY_VERSION_FAIL,
) -> bytes:
    """Accept only the one canonical 9-base/18-residual AUTH0 census."""

    _parse(raw, "execution census")
    expected = _expected_builder()
    if raw != expected:
        raise ExecutionSchemaError(
            f"{_identity_failure}: execution census differs from the canonical "
            "9-base/18-residual v2 census"
        )
    return expected


class _ClosedArtifact:
    __slots__ = ("_canonical", "__weakref__")

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise ExecutionSchemaError(
            f"{type(self).__name__} is minted only by its closed validator"
        )

    def __setattr__(self, name: str, value: object) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable: {name}")

    def __delattr__(self, name: str) -> None:
        raise AttributeError(f"{type(self).__name__} is immutable: {name}")


class BaseTrainCompletion(_ClosedArtifact):
    """Issued, immutable view of one canonical base completion object."""


class BaseQualification(_ClosedArtifact):
    """Authority-zero evaluation of the preregistered nine-completion gate."""


class BaseTerminalReceipt(_ClosedArtifact):
    """Issued, authority-zero terminal-schema view."""


class MimeCacheReceipt(_ClosedArtifact):
    """Issued, authority-zero MIME cache-schema view."""


class QualifiedMimeBaseLease(_ClosedArtifact):
    """Non-production proof bundle; never authorizes execution."""


_COMPLETIONS: WeakKeyDictionary[BaseTrainCompletion, bytes] = WeakKeyDictionary()
_QUALIFICATIONS: WeakKeyDictionary[BaseQualification, bytes] = WeakKeyDictionary()
_TERMINALS: WeakKeyDictionary[BaseTerminalReceipt, bytes] = WeakKeyDictionary()
_CACHES: WeakKeyDictionary[MimeCacheReceipt, bytes] = WeakKeyDictionary()
_LEASES: WeakKeyDictionary[QualifiedMimeBaseLease, bytes] = WeakKeyDictionary()


def _mint(
    artifact_type: type[Any],
    raw: bytes,
    issued: WeakKeyDictionary[Any, bytes],
    _new: Any = object.__new__,
    _setattr: Any = object.__setattr__,
) -> Any:
    value = _new(artifact_type)
    _setattr(value, "_canonical", raw)
    issued[value] = raw
    return value


def _issued_payload(
    value: object,
    artifact_type: type[Any],
    issued: WeakKeyDictionary[Any, bytes],
    label: str,
    _exact_type: Any = type,
    _getattribute: Any = object.__getattribute__,
    _parse: Any = _parse_canonical_object,
) -> dict[str, object]:
    if _exact_type(value) is not artifact_type:
        raise TypeError(f"{label} has the wrong exact type")
    expected = issued.get(value)
    if expected is None:
        raise ExecutionSchemaError(f"{label} was not issued by the closed validator")
    try:
        current = _getattribute(value, "_canonical")
    except AttributeError as exc:
        raise ExecutionSchemaError(f"{label} has no canonical bytes") from exc
    if _exact_type(current) is not bytes or current != expected:
        raise ExecutionSchemaError(f"{label} canonical bytes changed after issuance")
    return _parse(current, label)


def _completion_payload(
    value: object,
    _artifact_type: type[BaseTrainCompletion] = BaseTrainCompletion,
    _issued: WeakKeyDictionary[BaseTrainCompletion, bytes] = _COMPLETIONS,
    _read: Any = _issued_payload,
) -> dict[str, object]:
    return _read(value, _artifact_type, _issued, "completion")


def _qualification_payload(
    value: object,
    _artifact_type: type[BaseQualification] = BaseQualification,
    _issued: WeakKeyDictionary[BaseQualification, bytes] = _QUALIFICATIONS,
    _read: Any = _issued_payload,
) -> dict[str, object]:
    return _read(value, _artifact_type, _issued, "qualification")


def _terminal_payload(
    value: object,
    _artifact_type: type[BaseTerminalReceipt] = BaseTerminalReceipt,
    _issued: WeakKeyDictionary[BaseTerminalReceipt, bytes] = _TERMINALS,
    _read: Any = _issued_payload,
) -> dict[str, object]:
    return _read(value, _artifact_type, _issued, "terminal")


def _cache_payload(
    value: object,
    _artifact_type: type[MimeCacheReceipt] = MimeCacheReceipt,
    _issued: WeakKeyDictionary[MimeCacheReceipt, bytes] = _CACHES,
    _read: Any = _issued_payload,
) -> dict[str, object]:
    return _read(value, _artifact_type, _issued, "cache receipt")


def _lease_payload(
    value: object,
    _artifact_type: type[QualifiedMimeBaseLease] = QualifiedMimeBaseLease,
    _issued: WeakKeyDictionary[QualifiedMimeBaseLease, bytes] = _LEASES,
    _read: Any = _issued_payload,
) -> dict[str, object]:
    return _read(value, _artifact_type, _issued, "qualified lease")


def _bind_closed_artifact_properties(
    _completion_type: type[BaseTrainCompletion] = BaseTrainCompletion,
    _qualification_type: type[BaseQualification] = BaseQualification,
    _terminal_type: type[BaseTerminalReceipt] = BaseTerminalReceipt,
    _cache_type: type[MimeCacheReceipt] = MimeCacheReceipt,
    _lease_type: type[QualifiedMimeBaseLease] = QualifiedMimeBaseLease,
    _completion_reader: Any = _completion_payload,
    _qualification_reader: Any = _qualification_payload,
    _terminal_reader: Any = _terminal_payload,
    _cache_reader: Any = _cache_payload,
    _lease_reader: Any = _lease_payload,
    _property: Any = property,
) -> None:
    def completion_run_id(value: object) -> str:
        return _completion_reader(value)["run_id"]  # type: ignore[return-value]

    def completion_system_id(value: object) -> str:
        return _completion_reader(value)["system_id"]  # type: ignore[return-value]

    def completion_seed(value: object) -> int:
        return _completion_reader(value)["seed"]  # type: ignore[return-value]

    def qualification_verdict(value: object) -> str:
        return _qualification_reader(value)["verdict"]  # type: ignore[return-value]

    def terminal_run_id(value: object) -> str:
        return _terminal_reader(value)["run_id"]  # type: ignore[return-value]

    def cache_seed(value: object) -> int:
        return _cache_reader(value)["seed"]  # type: ignore[return-value]

    def lease_authority(value: object) -> int:
        _lease_reader(value)
        return 0

    def lease_execution_authorized(value: object) -> bool:
        _lease_reader(value)
        return False

    _completion_type.run_id = _property(completion_run_id)  # type: ignore[attr-defined]
    _completion_type.system_id = _property(completion_system_id)  # type: ignore[attr-defined]
    _completion_type.seed = _property(completion_seed)  # type: ignore[attr-defined]
    _qualification_type.verdict = _property(qualification_verdict)  # type: ignore[attr-defined]
    _terminal_type.run_id = _property(terminal_run_id)  # type: ignore[attr-defined]
    _cache_type.seed = _property(cache_seed)  # type: ignore[attr-defined]
    _lease_type.authority = _property(lease_authority)  # type: ignore[attr-defined]
    _lease_type.execution_authorized = _property(  # type: ignore[attr-defined]
        lease_execution_authorized
    )


_bind_closed_artifact_properties()
del _bind_closed_artifact_properties


def validate_base_train_completion(
    raw: object,
    _parse: Any = _parse_canonical_object,
    _require_closed_keys: Any = _require_keys,
    _parse_id: Any = parse_run_id,
    _architectures: MappingProxyType[str, str] = BASE_ARCHITECTURES,
    _artifact_type: type[BaseTrainCompletion] = BaseTrainCompletion,
    _issued: WeakKeyDictionary[BaseTrainCompletion, bytes] = _COMPLETIONS,
    _mint_issued: Any = _mint,
) -> BaseTrainCompletion:
    payload = _parse(raw, "base completion")
    _require_closed_keys(payload, _COMPLETION_KEYS, "base completion")
    if payload["schema"] != "phasepair-base-train-completion-v2":
        raise ExecutionSchemaError("base completion schema mismatch")
    if payload["status"] != "EXECUTION_COMPLETE_NONTERMINAL_AWAITING_QUALIFICATION":
        raise ExecutionSchemaError("base completion status mismatch")
    role, seed, system = _parse_id(payload["run_id"])
    if role != "BASE_TRAIN":
        raise ExecutionSchemaError("base completion must bind a BASE_TRAIN run")
    if _exact_int(payload["seed"], "completion.seed") != seed:
        raise ExecutionSchemaError("completion seed/run_id mismatch")
    if _exact_str(payload["system_id"], "completion.system_id") != system:
        raise ExecutionSchemaError("completion system/run_id mismatch")
    architecture = _exact_str(payload["architecture"], "completion.architecture")
    if architecture != _architectures[system]:
        raise ExecutionSchemaError("completion architecture/system mismatch")
    _exact_int(payload["optimizer_step_count"], "optimizer_step_count", minimum=1)
    _exact_int(payload["example_count"], "example_count", minimum=1)
    selected = _exact_int(
        payload["selected_epoch_index"],
        "selected_epoch_index",
        minimum=12,
        maximum=29,
    )
    completed = _exact_int(
        payload["selected_completed_epoch"],
        "selected_completed_epoch",
        minimum=13,
        maximum=30,
    )
    if completed != selected + 1:
        raise ExecutionSchemaError("selected completed/index epoch mismatch")
    for key in (
        "run_input_sha256",
        "selected_checkpoint_sha256",
        "validation_metric_sha256",
        "evaluator_sha256",
        "log_sha256",
        "model_final_state_sha256",
        "optimizer_final_state_sha256",
    ):
        _sha256(payload[key], key)
    return _mint_issued(_artifact_type, raw, _issued)


def base_train_completion_bytes(
    value: BaseTrainCompletion,
    _read: Any = _completion_payload,
    _getattribute: Any = object.__getattribute__,
) -> bytes:
    _read(value)
    return _getattribute(value, "_canonical")


def base_train_completion_sha256(
    value: BaseTrainCompletion,
    _sha256_factory: Any = hashlib.sha256,
    _serialize: Any = base_train_completion_bytes,
) -> str:
    return _sha256_factory(_serialize(value)).hexdigest()


@dataclass(frozen=True, slots=True)
class BaseValidationScore:
    system_id: str
    seed: int
    numerator: int
    denominator: int
    completion_sha256: str

    def __post_init__(
        self,
        _base_systems: tuple[str, ...] = ("00", "07", "08"),
        _seeds: tuple[int, ...] = (1729, 2718, 31415),
    ) -> None:
        system = _exact_str(self.system_id, "score.system_id")
        if system not in _base_systems:
            raise ExecutionSchemaError("score system is outside the base census")
        seed = _exact_int(self.seed, "score.seed")
        if seed not in _seeds:
            raise ExecutionSchemaError("score seed is outside the frozen census")
        numerator = _exact_int(self.numerator, "score.numerator")
        denominator = _exact_int(self.denominator, "score.denominator", minimum=1)
        if numerator > denominator:
            raise ExecutionSchemaError("validation score must be in [0,1]")
        _sha256(self.completion_sha256, "score.completion_sha256")


def _fraction_fields(prefix: str, value: Fraction) -> dict[str, int]:
    return {
        f"{prefix}_numerator": value.numerator,
        f"{prefix}_denominator": value.denominator,
    }


def qualify_nine_base_completions(
    completions: object,
    scores: object,
    _expected_ids: tuple[str, ...] = base_run_ids(),
    _seeds: tuple[int, ...] = (1729, 2718, 31415),
    _systems: tuple[str, ...] = ("00", "07", "08"),
    _completion_digest: Any = base_train_completion_sha256,
    _canonical: Any = canonical_json_bytes,
    _sha256_factory: Any = hashlib.sha256,
    _fraction_type: Any = Fraction,
    _score_type: type[BaseValidationScore] = BaseValidationScore,
    _completion_reader: Any = _completion_payload,
    _artifact_type: type[BaseQualification] = BaseQualification,
    _issued: WeakKeyDictionary[BaseQualification, bytes] = _QUALIFICATIONS,
    _mint_issued: Any = _mint,
) -> BaseQualification:
    if type(completions) is not tuple or len(completions) != 9:
        raise ExecutionSchemaError("qualification requires exactly nine completions")
    if type(scores) is not tuple or len(scores) != 9:
        raise ExecutionSchemaError("qualification requires exactly nine score rows")

    completion_payloads = tuple(_completion_reader(item) for item in completions)
    actual_ids = tuple(payload["run_id"] for payload in completion_payloads)
    if actual_ids != _expected_ids:
        raise ExecutionSchemaError(
            "completions must be unique and in canonical system-outer seed-inner order"
        )
    evaluators = tuple(payload["evaluator_sha256"] for payload in completion_payloads)
    if len(set(evaluators)) != 1:
        raise ExecutionSchemaError("all nine completions must bind one evaluator")

    checked_scores: list[BaseValidationScore] = []
    for index, score in enumerate(scores):
        if type(score) is not _score_type:
            raise TypeError(f"scores[{index}] must be exactly BaseValidationScore")
        rebuilt = _score_type(
            score.system_id,
            score.seed,
            score.numerator,
            score.denominator,
            score.completion_sha256,
        )
        expected_payload = completion_payloads[index]
        if (rebuilt.system_id, rebuilt.seed) != (
            expected_payload["system_id"],
            expected_payload["seed"],
        ):
            raise ExecutionSchemaError("score row order/identity mismatches completion")
        if rebuilt.completion_sha256 != _completion_digest(completions[index]):
            raise ExecutionSchemaError("score row completion digest mismatch")
        checked_scores.append(rebuilt)
    if len({score.denominator for score in checked_scores}) != 1:
        raise ExecutionSchemaError("all base validation scores must use one denominator")

    values = {
        (score.system_id, score.seed): _fraction_type(
            score.numerator, score.denominator
        )
        for score in checked_scores
    }
    means = {
        system: sum(
            (values[(system, seed)] for seed in _seeds), _fraction_type()
        )
        / 3
        for system in _systems
    }
    deltas = tuple(
        values[("00", seed)]
        - max(values[("07", seed)], values[("08", seed)])
        for seed in _seeds
    )
    same_seed_win_count = sum(delta > 0 for delta in deltas)
    minimum_delta = min(deltas)
    passed = (
        means["00"] > means["07"]
        and means["00"] > means["08"]
        and same_seed_win_count >= 2
        and minimum_delta >= _fraction_type(-1, 200)
    )
    verdict = "QUALIFIED" if passed else "NOT_QUALIFIED"

    score_rows = [
        {
            "completion_sha256": score.completion_sha256,
            "denominator": score.denominator,
            "numerator": score.numerator,
            "seed": score.seed,
            "system_id": score.system_id,
        }
        for score in checked_scores
    ]
    payload: dict[str, object] = {
        "authority": 0,
        "completion_sha256s": [
            _completion_digest(item) for item in completions
        ],
        "production": False,
        "result_claimed": False,
        "same_seed_win_count": same_seed_win_count,
        "schema": "phasepair-base-qualification-v2",
        "score_rows_sha256": _sha256_factory(_canonical(score_rows)).hexdigest(),
        "status": "DATA_FREE_QUALIFICATION_SCHEMA_EVALUATED_NO_RESULT",
        "verdict": verdict,
    }
    payload.update(_fraction_fields("mime_mean", means["00"]))
    payload.update(_fraction_fields("early_mean", means["07"]))
    payload.update(_fraction_fields("late_mean", means["08"]))
    payload.update(_fraction_fields("minimum_delta", minimum_delta))
    if set(payload) != _QUALIFICATION_KEYS:
        raise AssertionError("internal qualification key drift")
    raw = _canonical(payload)
    return _mint_issued(_artifact_type, raw, _issued)


def base_qualification_bytes(
    value: BaseQualification,
    _read: Any = _qualification_payload,
    _getattribute: Any = object.__getattribute__,
) -> bytes:
    _read(value)
    return _getattribute(value, "_canonical")


def base_qualification_sha256(
    value: BaseQualification,
    _sha256_factory: Any = hashlib.sha256,
    _serialize: Any = base_qualification_bytes,
) -> str:
    return _sha256_factory(_serialize(value)).hexdigest()


def validate_base_terminal_receipt(
    raw: object,
    completion: BaseTrainCompletion,
    qualification: BaseQualification,
    _parse: Any = _parse_canonical_object,
    _require_closed_keys: Any = _require_keys,
    _parse_id: Any = parse_run_id,
    _base_ids: tuple[str, ...] = base_run_ids(),
    _completion_digest: Any = base_train_completion_sha256,
    _qualification_digest: Any = base_qualification_sha256,
    _qualified_literal: str = "QUALIFIED",
    _not_applicable_literal: str = "NOT_APPLICABLE",
    _completion_reader: Any = _completion_payload,
    _qualification_reader: Any = _qualification_payload,
    _artifact_type: type[BaseTerminalReceipt] = BaseTerminalReceipt,
    _issued: WeakKeyDictionary[BaseTerminalReceipt, bytes] = _TERMINALS,
    _mint_issued: Any = _mint,
) -> BaseTerminalReceipt:
    completion_payload = _completion_reader(completion)
    qualification_payload = _qualification_reader(qualification)
    payload = _parse(raw, "base terminal receipt")
    _require_closed_keys(payload, _TERMINAL_KEYS, "base terminal receipt")
    if payload["schema"] != "phasepair-base-terminal-receipt-v2":
        raise ExecutionSchemaError("base terminal schema mismatch")
    if payload["status"] != "DATA_FREE_TERMINAL_SCHEMA_VALIDATED_NO_RESULT":
        raise ExecutionSchemaError("base terminal status mismatch")
    if _exact_int(payload["authority"], "terminal.authority") != 0:
        raise ExecutionSchemaError("terminal authority must remain zero")
    if _exact_bool(payload["production"], "terminal.production"):
        raise ExecutionSchemaError("terminal production must remain false")
    if _exact_bool(payload["result_claimed"], "terminal.result_claimed"):
        raise ExecutionSchemaError("terminal result_claimed must remain false")
    role, seed, system = _parse_id(payload["run_id"])
    if role != "BASE_TRAIN":
        raise ExecutionSchemaError("terminal must bind a BASE_TRAIN run")
    if payload["run_id"] != completion_payload["run_id"]:
        raise ExecutionSchemaError("terminal/completion run mismatch")
    if _exact_int(payload["seed"], "terminal.seed") != seed:
        raise ExecutionSchemaError("terminal seed/run mismatch")
    if _exact_str(payload["system_id"], "terminal.system_id") != system:
        raise ExecutionSchemaError("terminal system/run mismatch")
    if payload["run_input_sha256"] != completion_payload["run_input_sha256"]:
        raise ExecutionSchemaError("terminal run-input mismatch")
    _sha256(payload["run_input_sha256"], "terminal.run_input_sha256")
    completion_digest = _completion_digest(completion)
    if payload["completion_sha256"] != completion_digest:
        raise ExecutionSchemaError("terminal completion digest mismatch")
    bound_completions = qualification_payload["completion_sha256s"]
    if type(bound_completions) is not list or len(bound_completions) != 9:
        raise ExecutionSchemaError("qualification completion census is malformed")
    canonical_index = _base_ids.index(completion_payload["run_id"])
    if bound_completions[canonical_index] != completion_digest:
        raise ExecutionSchemaError(
            "terminal completion is not the qualification-bound original"
        )
    if payload["qualification_sha256"] != _qualification_digest(qualification):
        raise ExecutionSchemaError("terminal qualification digest mismatch")

    outcome = _exact_str(payload["terminal_outcome"], "terminal_outcome")
    if outcome not in {"SUCCESS", "FAILURE"}:
        raise ExecutionSchemaError("terminal_outcome is outside the closed vocabulary")
    selected = _exact_str(
        payload["selected_checkpoint_sha256"], "selected_checkpoint_sha256"
    )
    cache = _exact_str(
        payload["training_frozen_base_cache_sha256"],
        "training_frozen_base_cache_sha256",
    )
    verdict = qualification_payload["verdict"]
    if outcome == "SUCCESS":
        if selected != completion_payload["selected_checkpoint_sha256"]:
            raise ExecutionSchemaError("successful terminal checkpoint mismatch")
        _sha256(selected, "selected_checkpoint_sha256")
        cache_required = verdict == _qualified_literal and system == "00"
        if cache_required:
            _sha256(cache, "training_frozen_base_cache_sha256")
        elif cache != _not_applicable_literal:
            raise ExecutionSchemaError("cache is not applicable to this terminal")
    else:
        if selected != _not_applicable_literal or cache != _not_applicable_literal:
            raise ExecutionSchemaError("failed terminal cannot commit checkpoint/cache")
    return _mint_issued(_artifact_type, raw, _issued)


def base_terminal_receipt_bytes(
    value: BaseTerminalReceipt,
    _read: Any = _terminal_payload,
    _getattribute: Any = object.__getattribute__,
) -> bytes:
    _read(value)
    return _getattribute(value, "_canonical")


def base_terminal_receipt_sha256(
    value: BaseTerminalReceipt,
    _sha256_factory: Any = hashlib.sha256,
    _serialize: Any = base_terminal_receipt_bytes,
) -> str:
    return _sha256_factory(_serialize(value)).hexdigest()


def validate_mime_cache_receipt(
    raw: object,
    terminal: BaseTerminalReceipt,
    _parse: Any = _parse_canonical_object,
    _require_closed_keys: Any = _require_keys,
    _terminal_digest: Any = base_terminal_receipt_sha256,
    _not_applicable_literal: str = "NOT_APPLICABLE",
    _terminal_reader: Any = _terminal_payload,
    _artifact_type: type[MimeCacheReceipt] = MimeCacheReceipt,
    _issued: WeakKeyDictionary[MimeCacheReceipt, bytes] = _CACHES,
    _mint_issued: Any = _mint,
) -> MimeCacheReceipt:
    terminal_payload = _terminal_reader(terminal)
    payload = _parse(raw, "MIME cache receipt")
    _require_closed_keys(payload, _CACHE_KEYS, "MIME cache receipt")
    if payload["schema"] != "phasepair-mime-cache-receipt-v2":
        raise ExecutionSchemaError("MIME cache receipt schema mismatch")
    if payload["status"] != "DATA_FREE_CACHE_SCHEMA_VALIDATED_NO_RESULT":
        raise ExecutionSchemaError("MIME cache receipt status mismatch")
    if _exact_int(payload["authority"], "cache.authority") != 0:
        raise ExecutionSchemaError("cache authority must remain zero")
    if _exact_bool(payload["production"], "cache.production"):
        raise ExecutionSchemaError("cache production must remain false")
    if _exact_bool(payload["result_claimed"], "cache.result_claimed"):
        raise ExecutionSchemaError("cache result_claimed must remain false")
    if terminal_payload["terminal_outcome"] != "SUCCESS":
        raise ExecutionSchemaError("cache requires a successful base terminal")
    if terminal_payload["system_id"] != "00":
        raise ExecutionSchemaError("only MIME base terminals may own a cache")
    if terminal_payload["training_frozen_base_cache_sha256"] == _not_applicable_literal:
        raise ExecutionSchemaError("terminal has no applicable MIME cache")
    if _exact_str(payload["run_id"], "cache.run_id") != terminal_payload["run_id"]:
        raise ExecutionSchemaError("cache/terminal run mismatch")
    if _exact_str(payload["system_id"], "cache.system_id") != "00":
        raise ExecutionSchemaError("cache system must be 00")
    if _exact_int(payload["seed"], "cache.seed") != terminal_payload["seed"]:
        raise ExecutionSchemaError("cache/terminal seed mismatch")
    if payload["base_terminal_sha256"] != _terminal_digest(terminal):
        raise ExecutionSchemaError("cache terminal digest mismatch")
    if payload["cache_sha256"] != terminal_payload["training_frozen_base_cache_sha256"]:
        raise ExecutionSchemaError("cache content digest mismatch")
    _sha256(payload["base_terminal_sha256"], "base_terminal_sha256")
    _sha256(payload["cache_sha256"], "cache_sha256")
    return _mint_issued(_artifact_type, raw, _issued)


def mime_cache_receipt_bytes(
    value: MimeCacheReceipt,
    _read: Any = _cache_payload,
    _getattribute: Any = object.__getattribute__,
) -> bytes:
    _read(value)
    return _getattribute(value, "_canonical")


def mime_cache_receipt_sha256(
    value: MimeCacheReceipt,
    _sha256_factory: Any = hashlib.sha256,
    _serialize: Any = mime_cache_receipt_bytes,
) -> str:
    return _sha256_factory(_serialize(value)).hexdigest()


def _issue_qualified_mime_base_lease(
    qualification: BaseQualification,
    cache_receipts: object,
    terminals: object,
    _qualified_literal: str = "QUALIFIED",
    _base_ids: tuple[str, ...] = base_run_ids(),
    _seeds: tuple[int, ...] = (1729, 2718, 31415),
    _qualification_digest: Any = base_qualification_sha256,
    _terminal_digest: Any = base_terminal_receipt_sha256,
    _canonical: Any = canonical_json_bytes,
    _qualification_reader: Any = _qualification_payload,
    _terminal_reader: Any = _terminal_payload,
    _cache_reader: Any = _cache_payload,
    _artifact_type: type[QualifiedMimeBaseLease] = QualifiedMimeBaseLease,
    _issued: WeakKeyDictionary[QualifiedMimeBaseLease, bytes] = _LEASES,
    _mint_issued: Any = _mint,
) -> QualifiedMimeBaseLease:
    """Private AUTH0 issuer used only after closed schema validation."""

    qualification_payload = _qualification_reader(qualification)
    if qualification_payload["verdict"] != _qualified_literal:
        raise ExecutionSchemaError("a qualified lease requires verdict QUALIFIED")
    if type(terminals) is not tuple or len(terminals) != 9:
        raise ExecutionSchemaError("qualified lease requires exactly nine terminals")
    terminal_payloads = tuple(_terminal_reader(item) for item in terminals)
    if tuple(payload["run_id"] for payload in terminal_payloads) != _base_ids:
        raise ExecutionSchemaError("terminals are not the canonical nine-run census")
    qualification_digest = _qualification_digest(qualification)
    bound_completions = qualification_payload["completion_sha256s"]
    if type(bound_completions) is not list or len(bound_completions) != 9:
        raise ExecutionSchemaError("qualification completion census is malformed")
    for index, payload in enumerate(terminal_payloads):
        if payload["terminal_outcome"] != "SUCCESS":
            raise ExecutionSchemaError("qualified lease requires nine successful terminals")
        if payload["qualification_sha256"] != qualification_digest:
            raise ExecutionSchemaError("terminal qualification digest mismatch")
        if payload["completion_sha256"] != bound_completions[index]:
            raise ExecutionSchemaError(
                "terminal completion is not the qualification-bound original"
            )

    if type(cache_receipts) is not tuple or len(cache_receipts) != 3:
        raise ExecutionSchemaError("qualified lease requires exactly three MIME caches")
    cache_payloads = tuple(_cache_reader(item) for item in cache_receipts)
    if tuple(payload["seed"] for payload in cache_payloads) != _seeds:
        raise ExecutionSchemaError("MIME caches must be unique and seed-ascending")
    mime_terminals = {
        payload["seed"]: terminal
        for payload, terminal in zip(terminal_payloads, terminals, strict=True)
        if payload["system_id"] == "00"
    }
    for payload in cache_payloads:
        terminal = mime_terminals[payload["seed"]]
        if payload["base_terminal_sha256"] != _terminal_digest(terminal):
            raise ExecutionSchemaError("cache does not bind its canonical MIME terminal")
    cache_hashes = tuple(payload["cache_sha256"] for payload in cache_payloads)
    if len(set(cache_hashes)) != 3:
        raise ExecutionSchemaError("three MIME cache content digests must be distinct")

    payload = {
        "authority": 0,
        "cache_sha256s": list(cache_hashes),
        "execution_authorized": False,
        "production": False,
        "qualification_sha256": qualification_digest,
        "result_claimed": False,
        "schema": "phasepair-qualified-mime-base-lease-v1",
        "status": "DATA_FREE_CAPABILITY_NONPRODUCTION_NO_RESULT",
        "terminal_sha256s": [
            _terminal_digest(item) for item in terminals
        ],
    }
    raw = _canonical(payload)
    return _mint_issued(_artifact_type, raw, _issued)


def qualified_mime_base_lease_bytes(
    value: QualifiedMimeBaseLease,
    _read: Any = _lease_payload,
    _getattribute: Any = object.__getattribute__,
) -> bytes:
    _read(value)
    return _getattribute(value, "_canonical")


__all__ = [
    "AUTHORITY",
    "BASE_ARCHITECTURES",
    "BASE_SYSTEMS",
    "BaseQualification",
    "BaseTerminalReceipt",
    "BaseTrainCompletion",
    "BaseValidationScore",
    "EXECUTION_AUTHORIZED",
    "ExecutionSchemaError",
    "MimeCacheReceipt",
    "NOT_APPLICABLE",
    "NOT_QUALIFIED",
    "PRODUCTION",
    "QUALIFIED",
    "QualifiedMimeBaseLease",
    "RESIDUAL_NAMES",
    "RESIDUAL_SYSTEMS",
    "RESULT_CLAIMED",
    "SEEDS",
    "STATUS",
    "base_qualification_bytes",
    "base_qualification_sha256",
    "base_run_ids",
    "base_terminal_receipt_bytes",
    "base_terminal_receipt_sha256",
    "base_train_completion_bytes",
    "base_train_completion_sha256",
    "canonical_execution_census_bytes",
    "canonical_json_bytes",
    "mime_cache_receipt_bytes",
    "mime_cache_receipt_sha256",
    "parse_run_id",
    "qualified_mime_base_lease_bytes",
    "qualify_nine_base_completions",
    "residual_run_ids",
    "validate_base_terminal_receipt",
    "validate_base_train_completion",
    "validate_execution_census",
    "validate_mime_cache_receipt",
]
