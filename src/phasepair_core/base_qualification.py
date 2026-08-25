"""Authority-zero semantic binding for the nine PhasePair base results.

``execution_schema`` already freezes the base-census identities and implements
the preregistered MIME arithmetic.  This module adds the missing scientific
binding: every exact full-gallery primary score is committed together with the
successful terminal, selected checkpoint, score matrix, evaluator, and all
prerequisite receipt digests that make that score eligible for qualification.

The module is data-free and standard-library-only.  It consumes caller-frozen
identities and exact rational scores; ``terminal_success`` and every digest are
opaque caller assertions committed by the output, not a claim that this module
parsed an ``execution_schema.BaseTerminalReceipt``.  It does not read an
artifact, execute a run, authorize later work, or turn a scientific
non-qualification into an operational failure.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from types import MappingProxyType
from typing import Any


_SEED_ORDER = (1729, 2718, 31415)
_SYSTEM_ORDER = ("00", "07", "08")
_SYSTEM_NAMES = MappingProxyType({"00": "mime", "07": "early", "08": "late"})
_PRIMARY_METRIC_ID = "validation_full_gallery_cluster_macro_bidirectional_mean_R1"
_STATUS = "DATA_FREE_BASE_QUALIFICATION_AUTHORITY0_NO_RESULT"
_QUALIFIED = "QUALIFIED"
_NOT_QUALIFIED = "MIME_ADAPTATION_NOT_QUALIFIED"
_NOT_APPLICABLE = "NOT_APPLICABLE"

AUTHORITY = 0
PRODUCTION = False
EXECUTION_AUTHORIZED = False
RESULT_CLAIMED = False
STATUS = _STATUS
SEED_ORDER = _SEED_ORDER
SYSTEM_ORDER = _SYSTEM_ORDER
SYSTEM_NAMES = _SYSTEM_NAMES
PRIMARY_METRIC_ID = _PRIMARY_METRIC_ID
QUALIFIED = _QUALIFIED
NOT_QUALIFIED = _NOT_QUALIFIED
NOT_APPLICABLE = _NOT_APPLICABLE

_EXPECTED_IDENTITIES = tuple(
    (system_id, seed) for system_id in _SYSTEM_ORDER for seed in _SEED_ORDER
)
_SHA_FIELDS = (
    "terminal_execution_receipt_sha256",
    "selected_checkpoint_sha256",
    "full_score_matrix_sha256",
    "evaluator_sha256",
    "optimizer_opt_receipt_sha256",
    "dropout_bd_receipt_sha256",
    "overfit_receipt_sha256",
    "numerical_receipt_sha256",
)


class BaseQualificationError(ValueError):
    """A frozen base result violates the qualification contract."""


@dataclass(frozen=True, slots=True)
class FrozenBaseQualificationRow:
    """One completed run and its exact full-gallery qualification evidence."""

    system_id: str
    seed: int
    run_id: str
    terminal_success: bool
    terminal_execution_receipt_sha256: str
    selected_checkpoint_sha256: str
    full_score_matrix_sha256: str
    evaluator_sha256: str
    optimizer_opt_receipt_sha256: str
    dropout_bd_receipt_sha256: str
    overfit_receipt_sha256: str
    numerical_receipt_sha256: str
    metric_id: str
    primary_numerator: int
    primary_denominator: int


@dataclass(frozen=True, slots=True)
class AuthorityZeroBaseQualification:
    """Canonical nine-run strength-gate result; never an execution grant."""

    rows: tuple[FrozenBaseQualificationRow, ...]
    rows_sha256: str
    mime_mean: Fraction
    early_mean: Fraction
    late_mean: Fraction
    same_seed_stronger_comparator_win_count: int
    minimum_same_seed_delta: Fraction
    mean_strictly_above_early: bool
    mean_strictly_above_late: bool
    at_least_two_of_three_stronger_wins: bool
    minimum_delta_at_least_minus_0_005: bool
    qualified: bool


def _exact_text(value: object, label: str) -> str:
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
        raise TypeError(f"{label} must be an exact built-in int, not bool or float")
    if value < minimum or value > maximum:
        raise BaseQualificationError(
            f"{label} must be inside the closed interval [{minimum},{maximum}]"
        )
    return value


def _exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact built-in bool")
    return value


def _lower_sha256(value: object, label: str) -> str:
    checked = _exact_text(value, label)
    if len(checked) != 64 or any(
        character not in "0123456789abcdef" for character in checked
    ):
        raise BaseQualificationError(f"{label} must be lowercase SHA-256 hex")
    return checked


def _json_tree(value: object, label: str = "value") -> object:
    if value is None or type(value) in {str, bool, int}:
        return value
    if type(value) is list:
        return [
            _json_tree(item, f"{label}[{index}]") for index, item in enumerate(value)
        ]
    if type(value) is dict:
        output: dict[str, object] = {}
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError(f"{label} keys must be exact built-in str")
            output[key] = _json_tree(item, f"{label}.{key}")
        return output
    raise TypeError(f"{label} contains a non-canonical JSON type")


def _canonical_json_bytes(value: object, _dumps: Any = json.dumps) -> bytes:
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


def _expected_run_id(system_id: str, seed: int) -> str:
    return f"phasepair-run-v2/BASE_TRAIN/{seed}/{system_id}"


def _normalize_row(
    value: object,
    ordinal: int,
    _row_type: type[FrozenBaseQualificationRow] = FrozenBaseQualificationRow,
    _systems: tuple[str, ...] = _SYSTEM_ORDER,
    _seeds: tuple[int, ...] = _SEED_ORDER,
    _metric_id: str = _PRIMARY_METRIC_ID,
    _sha_fields: tuple[str, ...] = _SHA_FIELDS,
) -> FrozenBaseQualificationRow:
    label = f"rows[{ordinal}]"
    if type(value) is not _row_type:
        raise TypeError(f"{label} must be exact FrozenBaseQualificationRow")

    system_id = _exact_text(value.system_id, f"{label}.system_id")
    if system_id not in _systems:
        raise BaseQualificationError(
            f"{label}.system_id is outside the 00/07/08 base census"
        )
    seed = _exact_int(value.seed, f"{label}.seed")
    if seed not in _seeds:
        raise BaseQualificationError(
            f"{label}.seed is outside the frozen three-seed census"
        )
    run_id = _exact_text(value.run_id, f"{label}.run_id")
    if run_id != _expected_run_id(system_id, seed):
        raise BaseQualificationError(
            f"{label}.run_id does not bind its exact BASE_TRAIN system/seed identity"
        )
    if not _exact_bool(value.terminal_success, f"{label}.terminal_success"):
        raise BaseQualificationError(
            f"{label} is not a successfully completed terminal base run"
        )

    digests = {
        field: _lower_sha256(getattr(value, field), f"{label}.{field}")
        for field in _sha_fields
    }
    metric_id = _exact_text(value.metric_id, f"{label}.metric_id")
    if metric_id != _metric_id:
        raise BaseQualificationError(
            f"{label}.metric_id is not the exact full-gallery primary metric"
        )
    numerator = _exact_int(value.primary_numerator, f"{label}.primary_numerator")
    denominator = _exact_int(
        value.primary_denominator,
        f"{label}.primary_denominator",
        minimum=1,
    )
    if numerator > denominator:
        raise BaseQualificationError(f"{label} primary metric must be inside [0,1]")

    return _row_type(
        system_id=system_id,
        seed=seed,
        run_id=run_id,
        terminal_success=True,
        terminal_execution_receipt_sha256=digests["terminal_execution_receipt_sha256"],
        selected_checkpoint_sha256=digests["selected_checkpoint_sha256"],
        full_score_matrix_sha256=digests["full_score_matrix_sha256"],
        evaluator_sha256=digests["evaluator_sha256"],
        optimizer_opt_receipt_sha256=digests["optimizer_opt_receipt_sha256"],
        dropout_bd_receipt_sha256=digests["dropout_bd_receipt_sha256"],
        overfit_receipt_sha256=digests["overfit_receipt_sha256"],
        numerical_receipt_sha256=digests["numerical_receipt_sha256"],
        metric_id=metric_id,
        primary_numerator=numerator,
        primary_denominator=denominator,
    )


def collect_base_qualification_rows(
    rows: object,
    _normalize: Any = _normalize_row,
    _expected_identities: tuple[tuple[str, int], ...] = _EXPECTED_IDENTITIES,
) -> tuple[FrozenBaseQualificationRow, ...]:
    """Validate nine rows and restore system-outer/seed-inner order."""

    if type(rows) is not tuple:
        raise TypeError("rows must be an exact built-in tuple")

    by_identity: dict[tuple[str, int], FrozenBaseQualificationRow] = {}
    for ordinal, value in enumerate(rows):
        row = _normalize(value, ordinal)
        identity = (row.system_id, row.seed)
        if identity in by_identity:
            raise BaseQualificationError(
                f"duplicate base qualification system/seed identity: {identity!r}"
            )
        by_identity[identity] = row

    missing = [
        identity for identity in _expected_identities if identity not in by_identity
    ]
    if missing:
        raise BaseQualificationError(
            f"missing base qualification system/seed identities: {missing!r}"
        )
    if len(by_identity) != len(_expected_identities):
        raise BaseQualificationError("base qualification census must contain nine rows")

    canonical = tuple(by_identity[identity] for identity in _expected_identities)
    if len({row.primary_denominator for row in canonical}) != 1:
        raise BaseQualificationError(
            "all nine full-gallery primary metrics must use one denominator"
        )
    if len({row.evaluator_sha256 for row in canonical}) != 1:
        raise BaseQualificationError("all nine rows must bind one frozen evaluator")
    return canonical


def _row_payload(
    row: FrozenBaseQualificationRow,
    _names: MappingProxyType[str, str] = _SYSTEM_NAMES,
) -> dict[str, object]:
    return {
        "dropout_bd_receipt_sha256": row.dropout_bd_receipt_sha256,
        "evaluator_sha256": row.evaluator_sha256,
        "full_score_matrix_sha256": row.full_score_matrix_sha256,
        "metric_id": row.metric_id,
        "numerical_receipt_sha256": row.numerical_receipt_sha256,
        "optimizer_opt_receipt_sha256": row.optimizer_opt_receipt_sha256,
        "overfit_receipt_sha256": row.overfit_receipt_sha256,
        "primary_denominator": row.primary_denominator,
        "primary_numerator": row.primary_numerator,
        "run_id": row.run_id,
        "seed": row.seed,
        "selected_checkpoint_sha256": row.selected_checkpoint_sha256,
        "system_id": row.system_id,
        "system_name": _names[row.system_id],
        "terminal_execution_receipt_sha256": (row.terminal_execution_receipt_sha256),
        "terminal_success": row.terminal_success,
    }


def _fraction_payload(value: Fraction) -> dict[str, int]:
    return {"denominator": value.denominator, "numerator": value.numerator}


def evaluate_base_qualification(
    rows: object,
    _fraction_type: Any = Fraction,
    _sha256_factory: Any = hashlib.sha256,
    _collect: Any = collect_base_qualification_rows,
    _seeds: tuple[int, ...] = _SEED_ORDER,
    _systems: tuple[str, ...] = _SYSTEM_ORDER,
    _row_encoder: Any = _row_payload,
    _canonical: Any = _canonical_json_bytes,
    _result_type: type[AuthorityZeroBaseQualification] = (
        AuthorityZeroBaseQualification
    ),
) -> AuthorityZeroBaseQualification:
    """Evaluate the exact preregistered MIME gate over nine completed runs."""

    canonical = _collect(rows)
    values = {
        (row.system_id, row.seed): _fraction_type(
            row.primary_numerator, row.primary_denominator
        )
        for row in canonical
    }
    means = {
        system_id: sum(
            (values[(system_id, seed)] for seed in _seeds),
            _fraction_type(),
        )
        / 3
        for system_id in _systems
    }
    deltas = tuple(
        values[("00", seed)] - max(values[("07", seed)], values[("08", seed)])
        for seed in _seeds
    )
    win_count = sum(delta > 0 for delta in deltas)
    minimum_delta = min(deltas)
    above_early = means["00"] > means["07"]
    above_late = means["00"] > means["08"]
    enough_wins = win_count >= 2
    minimum_pass = minimum_delta >= _fraction_type(-1, 200)
    qualified = above_early and above_late and enough_wins and minimum_pass
    rows_sha256 = _sha256_factory(
        _canonical([_row_encoder(row) for row in canonical])
    ).hexdigest()

    return _result_type(
        rows=canonical,
        rows_sha256=rows_sha256,
        mime_mean=means["00"],
        early_mean=means["07"],
        late_mean=means["08"],
        same_seed_stronger_comparator_win_count=win_count,
        minimum_same_seed_delta=minimum_delta,
        mean_strictly_above_early=above_early,
        mean_strictly_above_late=above_late,
        at_least_two_of_three_stronger_wins=enough_wins,
        minimum_delta_at_least_minus_0_005=minimum_pass,
        qualified=qualified,
    )


def _validated_qualification(
    value: object,
    _result_type: type[AuthorityZeroBaseQualification] = (
        AuthorityZeroBaseQualification
    ),
    _evaluate: Any = evaluate_base_qualification,
) -> AuthorityZeroBaseQualification:
    if type(value) is not _result_type:
        raise TypeError("qualification must be exact AuthorityZeroBaseQualification")
    rebuilt = _evaluate(value.rows)
    if rebuilt != value:
        raise BaseQualificationError(
            "qualification row commitment or derived gate changed after evaluation"
        )
    return rebuilt


def canonical_base_qualification_bytes(
    qualification: object,
    _validate: Any = _validated_qualification,
    _canonical: Any = _canonical_json_bytes,
    _row_encoder: Any = _row_payload,
    _fraction_encoder: Any = _fraction_payload,
    _seed_order: tuple[int, ...] = _SEED_ORDER,
    _system_order: tuple[str, ...] = _SYSTEM_ORDER,
    _metric_id: str = _PRIMARY_METRIC_ID,
    _status: str = _STATUS,
    _qualified_literal: str = _QUALIFIED,
    _not_qualified_literal: str = _NOT_QUALIFIED,
    _not_applicable_literal: str = _NOT_APPLICABLE,
) -> bytes:
    """Return deterministic Authority0 JSON bytes with one terminal LF."""

    checked = _validate(qualification)
    outcome = _qualified_literal if checked.qualified else _not_qualified_literal
    return _canonical(
        {
            "all_terminals_successful": True,
            "authority": 0,
            "completed_run_count": 9,
            "evidence_scope": "CALLER_FROZEN_RECEIPT_AND_SCORE_BINDINGS",
            "execution_authorized": False,
            "gates": {
                "at_least_two_of_three_stronger_wins": (
                    checked.at_least_two_of_three_stronger_wins
                ),
                "mean_strictly_above_early": checked.mean_strictly_above_early,
                "mean_strictly_above_late": checked.mean_strictly_above_late,
                "minimum_delta_at_least_minus_0_005": (
                    checked.minimum_delta_at_least_minus_0_005
                ),
            },
            "means": {
                "00": _fraction_encoder(checked.mime_mean),
                "07": _fraction_encoder(checked.early_mean),
                "08": _fraction_encoder(checked.late_mean),
            },
            "metric_id": _metric_id,
            "minimum_same_seed_delta": _fraction_encoder(
                checked.minimum_same_seed_delta
            ),
            "production": False,
            "qualification_failure_kind": (
                _not_applicable_literal if checked.qualified else "SCIENTIFIC"
            ),
            "qualification_outcome": outcome,
            "qualified": checked.qualified,
            "result_claimed": False,
            "row_count": len(checked.rows),
            "row_order": "system_outer_seed_inner",
            "rows": [_row_encoder(row) for row in checked.rows],
            "rows_sha256": checked.rows_sha256,
            "same_seed_stronger_comparator_win_count": (
                checked.same_seed_stronger_comparator_win_count
            ),
            "schema": "phasepair-base-qualification-semantic-v1",
            "seed_order": list(_seed_order),
            "status": _status,
            "system_order": list(_system_order),
        }
    )


def base_qualification_sha256(
    qualification: object,
    _sha256_factory: Any = hashlib.sha256,
    _serialize: Any = canonical_base_qualification_bytes,
) -> str:
    """Hash the canonical semantic qualification bytes."""

    return _sha256_factory(_serialize(qualification)).hexdigest()


__all__ = [
    "AUTHORITY",
    "EXECUTION_AUTHORIZED",
    "NOT_APPLICABLE",
    "NOT_QUALIFIED",
    "PRIMARY_METRIC_ID",
    "PRODUCTION",
    "QUALIFIED",
    "RESULT_CLAIMED",
    "SEED_ORDER",
    "STATUS",
    "SYSTEM_NAMES",
    "SYSTEM_ORDER",
    "AuthorityZeroBaseQualification",
    "BaseQualificationError",
    "FrozenBaseQualificationRow",
    "base_qualification_sha256",
    "canonical_base_qualification_bytes",
    "collect_base_qualification_rows",
    "evaluate_base_qualification",
]
