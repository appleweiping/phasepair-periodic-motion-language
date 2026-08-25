"""Frozen, data-free PhaseSet experiment matrix and 33-run DAG.

The public package can describe and validate the preregistered execution graph,
but it cannot authorize data access, server access, GPU use, or a scientific
claim.  All returned objects are immutable and all serialized forms are strict
canonical JSON.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
from pathlib import Path


AUTHORITY = 0
EXECUTION_AUTHORIZED = False
PRODUCTION = False
RESULT_CLAIMED = False
SCHEMA = "phaseset-experiment-matrix-v1"
PLAN_SCHEMA = "phaseset-run-dag-v1"
STATUS = "DATA_FREE_CONTRACT_ONLY_NO_EXECUTION_NO_RESULT"

SEEDS = (1729, 2718, 31415)
BASE_SYSTEMS = (
    ("B0", "ActorMean"),
    ("B1", "SetPMA"),
    ("B2", "SocialTemporal"),
)
RESIDUAL_SYSTEMS = (
    ("01", "generic six tokens"),
    ("02", "marginal Morlet power"),
    ("03", "mean/difference DCT"),
    ("04", "PhasePair pair-bag"),
    ("05", "coverage/missing-only"),
    ("06", "incidence-shuffled"),
    ("07", "phase-stripped full"),
    ("08", "PhaseSet full"),
)
FINAL_SYSTEMS = (
    ("00", "qualified group base"),
    *RESIDUAL_SYSTEMS,
)
HYPOTHESIS_COMPARISONS = (
    ("H1", "08", "00"),
    ("H2", "08", "01"),
    ("H3", "08", "02"),
    ("H4", "08", "03"),
    ("H5", "08", "04"),
    ("H6", "08", "05"),
    ("H7", "08", "06"),
    ("H8", "08", "07"),
)


class ExperimentContractError(ValueError):
    """Raised when the frozen matrix or DAG is changed or malformed."""


@dataclass(frozen=True, slots=True)
class ExperimentRun:
    """One immutable row in the registered PhaseSet DAG."""

    run_id: str
    role: str
    system_id: str
    system_name: str
    seed: int
    depends_on_run_ids: tuple[str, ...]
    required_artifact_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExperimentPlan:
    """The exact 9-base-plus-24-residual authority-zero plan."""

    runs: tuple[ExperimentRun, ...]
    matrix_sha256: str
    schema: str = PLAN_SCHEMA
    authority: int = AUTHORITY
    execution_authorized: bool = EXECUTION_AUTHORIZED
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED
    status: str = STATUS

    @property
    def base_runs(self) -> tuple[ExperimentRun, ...]:
        return tuple(row for row in self.runs if row.role == "BASE_QUALIFICATION")

    @property
    def residual_runs(self) -> tuple[ExperimentRun, ...]:
        return tuple(row for row in self.runs if row.role == "RESIDUAL_TRAIN")


@dataclass(frozen=True, slots=True)
class BaseScore:
    """One caller-frozen validation score bound to a base terminal digest."""

    run_id: str
    bidirectional_r1_numerator: int
    bidirectional_r1_denominator: int
    parameter_count: int
    frozen_runtime_latency_ns: int
    terminal_sha256: str


@dataclass(frozen=True, slots=True)
class BaseQualification:
    """Deterministic result of the frozen nine-run qualification rule."""

    winner_system_id: str
    winner_system_name: str
    winner_parameter_count: int
    winner_frozen_runtime_latency_ns: int
    system_mean_rows: tuple[tuple[str, int, int], ...]
    system_resource_rows: tuple[tuple[str, int, int], ...]
    completion_sha256s: tuple[str, ...]
    winner_terminal_sha256s: tuple[str, ...]
    score_rows_sha256: str
    schema: str = "phaseset-base-qualification-v1"
    status: str = "WINNER_SELECTED_FROM_COMPLETE_NINE_ROWS"
    authority: int = AUTHORITY
    production: bool = PRODUCTION
    result_claimed: bool = RESULT_CLAIMED


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _reject_float(value: str) -> object:
    raise ExperimentContractError(f"floating-point JSON is forbidden: {value}")


def _reject_constant(value: str) -> object:
    raise ExperimentContractError(f"non-finite JSON is forbidden: {value}")


def _object_without_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ExperimentContractError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _matrix_payload() -> dict[str, object]:
    return {
        "attempt_directory_reuse_allowed": False,
        "authority": AUTHORITY,
        "base_run_count": 9,
        "base_systems": [
            {"system_id": system_id, "system_name": name} for system_id, name in BASE_SYSTEMS
        ],
        "execution_authorized": EXECUTION_AUTHORIZED,
        "final_system_count": 9,
        "final_systems": [
            {"system_id": system_id, "system_name": name} for system_id, name in FINAL_SYSTEMS
        ],
        "hypotheses": [
            {
                "baseline_system_id": baseline,
                "hypothesis_id": hypothesis_id,
                "treatment_system_id": treatment,
            }
            for hypothesis_id, treatment, baseline in HYPOTHESIS_COMPARISONS
        ],
        "holm_family": [f"H{index}" for index in range(2, 9)],
        "parameter_matching": {
            "control_system_ids": [f"{index:02d}" for index in range(1, 8)],
            "full_system_id": "08",
            "relative_tolerance_decimal": "0.01",
        },
        "periodic_cache_source": "winning_base_checkpoint_by_seed",
        "production": PRODUCTION,
        "qualification": {
            "complete_row_count": 9,
            "metric": "validation_bidirectional_r1_mean",
            "must_select_one_winner": True,
            "tie_break_order": [
                "higher_three_seed_mean",
                "fewer_parameters",
                "lower_frozen_runtime_latency_ns",
                "smaller_base_system_id",
            ],
        },
        "qualified_base_final_system_id": "00",
        "qualified_base_retrained": False,
        "residual_run_count": 24,
        "residual_systems": [
            {"system_id": system_id, "system_name": name} for system_id, name in RESIDUAL_SYSTEMS
        ],
        "result_claimed": RESULT_CLAIMED,
        "run_count": 33,
        "schema": SCHEMA,
        "sealed_test_max_consumptions": 1,
        "seeds": list(SEEDS),
        "status": STATUS,
    }


def canonical_experiment_matrix_bytes() -> bytes:
    """Return the exact semantic matrix as canonical ASCII JSON plus LF."""

    return _canonical_json_bytes(_matrix_payload())


def experiment_matrix_sha256() -> str:
    return hashlib.sha256(canonical_experiment_matrix_bytes()).hexdigest()


def _validate_matrix_payload(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ExperimentContractError("experiment matrix must be one JSON object")
    expected = _matrix_payload()
    if value != expected:
        raise ExperimentContractError("experiment matrix differs from the frozen v1 census")
    return value


def validate_experiment_matrix_bytes(raw: object) -> bytes:
    """Validate exact canonical bytes and return a detached byte copy."""

    if type(raw) is not bytes:
        raise TypeError("raw must be exact built-in bytes")
    if raw.startswith(b"\xef\xbb\xbf") or b"\r" in raw or not raw.endswith(b"\n"):
        raise ExperimentContractError("matrix must be BOM-free LF-terminated UTF-8 JSON")
    try:
        parsed = json.loads(
            raw.decode("ascii", "strict"),
            object_pairs_hook=_object_without_duplicates,
            parse_float=_reject_float,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ExperimentContractError("matrix is not strict ASCII JSON") from exc
    _validate_matrix_payload(parsed)
    expected = canonical_experiment_matrix_bytes()
    if raw != expected:
        raise ExperimentContractError("matrix bytes are not canonical sorted JSON plus LF")
    return bytes(raw)


def load_experiment_matrix(path: str | Path) -> bytes:
    """Load and validate a frozen matrix without granting execution authority."""

    return validate_experiment_matrix_bytes(Path(path).read_bytes())


def base_run_ids() -> tuple[str, ...]:
    return tuple(
        f"phaseset-run-v1/BASE_QUALIFICATION/{seed}/{system_id}"
        for system_id, _ in BASE_SYSTEMS
        for seed in SEEDS
    )


def residual_run_ids() -> tuple[str, ...]:
    return tuple(
        f"phaseset-run-v1/RESIDUAL_TRAIN/{seed}/{system_id}"
        for system_id, _ in RESIDUAL_SYSTEMS
        for seed in SEEDS
    )


def parse_run_id(run_id: object) -> tuple[str, int, str]:
    if type(run_id) is not str:
        raise TypeError("run_id must be exact built-in str")
    parts = run_id.split("/")
    if len(parts) != 4 or parts[0] != "phaseset-run-v1":
        raise ExperimentContractError("run_id must use the exact phaseset-run-v1 form")
    role, raw_seed, system_id = parts[1:]
    if raw_seed not in {str(seed) for seed in SEEDS}:
        raise ExperimentContractError("run_id seed is outside the frozen census")
    allowed = BASE_SYSTEMS if role == "BASE_QUALIFICATION" else RESIDUAL_SYSTEMS
    if role not in {"BASE_QUALIFICATION", "RESIDUAL_TRAIN"}:
        raise ExperimentContractError("run_id role is outside the frozen census")
    if system_id not in {item[0] for item in allowed}:
        raise ExperimentContractError("run_id system is outside its role census")
    seed = int(raw_seed)
    expected = f"phaseset-run-v1/{role}/{seed}/{system_id}"
    if run_id != expected:
        raise ExperimentContractError("run_id is not in canonical form")
    return role, seed, system_id


def build_experiment_plan() -> ExperimentPlan:
    """Construct the immutable 33-run DAG; this performs no execution."""

    all_base_ids = base_run_ids()
    base_names = dict(BASE_SYSTEMS)
    residual_names = dict(RESIDUAL_SYSTEMS)
    base_rows = tuple(
        ExperimentRun(
            run_id=run_id,
            role="BASE_QUALIFICATION",
            system_id=parse_run_id(run_id)[2],
            system_name=base_names[parse_run_id(run_id)[2]],
            seed=parse_run_id(run_id)[1],
            depends_on_run_ids=(),
            required_artifact_ids=(
                "prepared-data-manifest-v1",
                "split-audit-v1",
                "runtime-preflight-v1",
            ),
        )
        for run_id in all_base_ids
    )
    residual_rows = tuple(
        ExperimentRun(
            run_id=run_id,
            role="RESIDUAL_TRAIN",
            system_id=parse_run_id(run_id)[2],
            system_name=residual_names[parse_run_id(run_id)[2]],
            seed=parse_run_id(run_id)[1],
            depends_on_run_ids=all_base_ids,
            required_artifact_ids=(
                "base-qualification-v1",
                f"qualified-base-checkpoint-v1/{parse_run_id(run_id)[1]}",
                f"periodic-cache-v1/{parse_run_id(run_id)[1]}",
                "prepared-data-manifest-v1",
                "split-audit-v1",
                "runtime-preflight-v1",
            ),
        )
        for run_id in residual_run_ids()
    )
    return ExperimentPlan(
        runs=base_rows + residual_rows,
        matrix_sha256=experiment_matrix_sha256(),
    )


def _run_payload(row: ExperimentRun) -> dict[str, object]:
    return {
        "depends_on_run_ids": list(row.depends_on_run_ids),
        "required_artifact_ids": list(row.required_artifact_ids),
        "role": row.role,
        "run_id": row.run_id,
        "seed": row.seed,
        "system_id": row.system_id,
        "system_name": row.system_name,
    }


def canonical_experiment_plan_bytes(plan: object | None = None) -> bytes:
    checked = build_experiment_plan() if plan is None else plan
    if type(checked) is not ExperimentPlan:
        raise TypeError("plan must be exact ExperimentPlan")
    canonical = build_experiment_plan()
    if checked != canonical:
        raise ExperimentContractError("plan differs from the canonical 33-run DAG")
    return _canonical_json_bytes(
        {
            "authority": checked.authority,
            "base_run_count": len(checked.base_runs),
            "execution_authorized": checked.execution_authorized,
            "matrix_sha256": checked.matrix_sha256,
            "production": checked.production,
            "residual_run_count": len(checked.residual_runs),
            "result_claimed": checked.result_claimed,
            "run_count": len(checked.runs),
            "runs": [_run_payload(row) for row in checked.runs],
            "schema": checked.schema,
            "status": checked.status,
        }
    )


def validate_experiment_plan_bytes(raw: object) -> ExperimentPlan:
    if type(raw) is not bytes:
        raise TypeError("raw must be exact built-in bytes")
    expected = canonical_experiment_plan_bytes()
    if raw != expected:
        raise ExperimentContractError("plan bytes differ from the canonical 33-run DAG")
    return build_experiment_plan()


def experiment_plan_sha256(plan: object | None = None) -> str:
    return hashlib.sha256(canonical_experiment_plan_bytes(plan)).hexdigest()


def _lower_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be exact built-in str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ExperimentContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def qualify_base(scores: object) -> BaseQualification:
    """Select one base winner after validating all nine registered rows."""

    if type(scores) is not tuple:
        raise TypeError("scores must be an exact tuple")
    by_run: dict[str, tuple[Fraction, int, int, str]] = {}
    for ordinal, value in enumerate(scores):
        if type(value) is not BaseScore:
            raise TypeError(f"scores[{ordinal}] must be exact BaseScore")
        role, _, _ = parse_run_id(value.run_id)
        if role != "BASE_QUALIFICATION" or value.run_id in by_run:
            raise ExperimentContractError("base score identities must be unique base run_ids")
        if (
            type(value.bidirectional_r1_numerator) is not int
            or type(value.bidirectional_r1_denominator) is not int
            or value.bidirectional_r1_denominator <= 0
        ):
            raise ExperimentContractError("base scores must use exact integer fractions")
        if type(value.parameter_count) is not int or value.parameter_count <= 0:
            raise ExperimentContractError("base parameter_count must be a positive exact int")
        if type(value.frozen_runtime_latency_ns) is not int or value.frozen_runtime_latency_ns <= 0:
            raise ExperimentContractError(
                "base frozen_runtime_latency_ns must be a positive exact int"
            )
        score = Fraction(
            value.bidirectional_r1_numerator,
            value.bidirectional_r1_denominator,
        )
        if not Fraction(0, 1) <= score <= Fraction(1, 1):
            raise ExperimentContractError("base score must be in [0,1]")
        by_run[value.run_id] = (
            score,
            value.parameter_count,
            value.frozen_runtime_latency_ns,
            _lower_sha256(value.terminal_sha256, "terminal_sha256"),
        )
    expected_ids = base_run_ids()
    if set(by_run) != set(expected_ids):
        raise ExperimentContractError("base qualification requires exactly all nine base rows")
    if len({by_run[run_id][3] for run_id in expected_ids}) != len(expected_ids):
        raise ExperimentContractError("base terminal receipts must be unique across nine rows")

    by_system_seed = {
        (parse_run_id(run_id)[2], parse_run_id(run_id)[1]): by_run[run_id][0]
        for run_id in expected_ids
    }
    metadata: dict[str, tuple[int, int]] = {}
    for system_id, _ in BASE_SYSTEMS:
        rows = tuple(
            by_run[f"phaseset-run-v1/BASE_QUALIFICATION/{seed}/{system_id}"] for seed in SEEDS
        )
        observed = {(row[1], row[2]) for row in rows}
        if len(observed) != 1:
            raise ExperimentContractError(
                "base parameter count and frozen latency must agree across seeds"
            )
        metadata[system_id] = next(iter(observed))
    means = {
        system_id: sum(
            (by_system_seed[(system_id, seed)] for seed in SEEDS),
            start=Fraction(0, 1),
        )
        / len(SEEDS)
        for system_id, _ in BASE_SYSTEMS
    }
    winner = min(
        (system_id for system_id, _ in BASE_SYSTEMS),
        key=lambda system_id: (
            -means[system_id],
            metadata[system_id][0],
            metadata[system_id][1],
            system_id,
        ),
    )
    score_rows = [
        {
            "bidirectional_r1_denominator": by_run[run_id][0].denominator,
            "bidirectional_r1_numerator": by_run[run_id][0].numerator,
            "frozen_runtime_latency_ns": by_run[run_id][2],
            "parameter_count": by_run[run_id][1],
            "run_id": run_id,
            "terminal_sha256": by_run[run_id][3],
        }
        for run_id in expected_ids
    ]
    return BaseQualification(
        winner_system_id=winner,
        winner_system_name=dict(BASE_SYSTEMS)[winner],
        winner_parameter_count=metadata[winner][0],
        winner_frozen_runtime_latency_ns=metadata[winner][1],
        system_mean_rows=tuple(
            (system_id, means[system_id].numerator, means[system_id].denominator)
            for system_id, _ in BASE_SYSTEMS
        ),
        system_resource_rows=tuple(
            (system_id, metadata[system_id][0], metadata[system_id][1])
            for system_id, _ in BASE_SYSTEMS
        ),
        completion_sha256s=tuple(by_run[run_id][3] for run_id in expected_ids),
        winner_terminal_sha256s=tuple(
            by_run[f"phaseset-run-v1/BASE_QUALIFICATION/{seed}/{winner}"][3] for seed in SEEDS
        ),
        score_rows_sha256=hashlib.sha256(_canonical_json_bytes(score_rows)).hexdigest(),
    )


def canonical_base_qualification_bytes(value: object) -> bytes:
    if type(value) is not BaseQualification:
        raise TypeError("value must be exact BaseQualification")
    if (
        value.schema != "phaseset-base-qualification-v1"
        or value.status != "WINNER_SELECTED_FROM_COMPLETE_NINE_ROWS"
        or value.authority != 0
        or value.production is not False
        or value.result_claimed is not False
    ):
        raise ExperimentContractError("qualification must remain authority zero/no-result")
    names = dict(BASE_SYSTEMS)
    if (
        type(value.winner_system_id) is not str
        or type(value.winner_system_name) is not str
        or value.winner_system_id not in names
        or value.winner_system_name != names[value.winner_system_id]
    ):
        raise ExperimentContractError("qualification winner identity is invalid")
    if type(value.winner_parameter_count) is not int or value.winner_parameter_count <= 0:
        raise ExperimentContractError("qualification winner parameter count is invalid")
    if (
        type(value.winner_frozen_runtime_latency_ns) is not int
        or value.winner_frozen_runtime_latency_ns <= 0
    ):
        raise ExperimentContractError("qualification winner latency is invalid")
    if type(value.completion_sha256s) is not tuple or len(value.completion_sha256s) != 9:
        raise ExperimentContractError("qualification must bind nine terminals")
    if type(value.winner_terminal_sha256s) is not tuple or len(value.winner_terminal_sha256s) != 3:
        raise ExperimentContractError("qualification must bind three winner terminals")
    for digest in value.completion_sha256s:
        _lower_sha256(digest, "completion_sha256")
    if len(set(value.completion_sha256s)) != 9:
        raise ExperimentContractError("qualification terminals must be unique")
    for digest in value.winner_terminal_sha256s:
        _lower_sha256(digest, "winner_terminal_sha256")
    expected_winner_digests = tuple(
        value.completion_sha256s[
            base_run_ids().index(
                f"phaseset-run-v1/BASE_QUALIFICATION/{seed}/{value.winner_system_id}"
            )
        ]
        for seed in SEEDS
    )
    if value.winner_terminal_sha256s != expected_winner_digests:
        raise ExperimentContractError(
            "qualification winner terminals must match the ordered nine-row census"
        )
    if (
        type(value.system_mean_rows) is not tuple
        or len(value.system_mean_rows) != len(BASE_SYSTEMS)
        or any(type(row) is not tuple or len(row) != 3 for row in value.system_mean_rows)
    ):
        raise ExperimentContractError("qualification system means must be exact triples")
    if tuple(row[0] for row in value.system_mean_rows) != tuple(names):
        raise ExperimentContractError("qualification system means must be ordered B0,B1,B2")
    means: dict[str, Fraction] = {}
    for system_id, numerator, denominator in value.system_mean_rows:
        if type(numerator) is not int or type(denominator) is not int or denominator <= 0:
            raise ExperimentContractError("qualification mean rows must be exact fractions")
        mean = Fraction(numerator, denominator)
        if not Fraction(0, 1) <= mean <= Fraction(1, 1):
            raise ExperimentContractError("qualification mean rows must be in [0,1]")
        if mean.numerator != numerator or mean.denominator != denominator:
            raise ExperimentContractError("qualification mean fractions must be reduced")
        means[system_id] = mean
    if (
        type(value.system_resource_rows) is not tuple
        or len(value.system_resource_rows) != len(BASE_SYSTEMS)
        or any(type(row) is not tuple or len(row) != 3 for row in value.system_resource_rows)
    ):
        raise ExperimentContractError("qualification resources must be exact triples")
    if tuple(row[0] for row in value.system_resource_rows) != tuple(names):
        raise ExperimentContractError("qualification resources must be ordered B0,B1,B2")
    resources: dict[str, tuple[int, int]] = {}
    for system_id, parameter_count, latency_ns in value.system_resource_rows:
        if (
            type(parameter_count) is not int
            or parameter_count <= 0
            or type(latency_ns) is not int
            or latency_ns <= 0
        ):
            raise ExperimentContractError("qualification resources must be positive exact ints")
        resources[system_id] = (parameter_count, latency_ns)
    expected_winner = min(
        names,
        key=lambda system_id: (
            -means[system_id],
            resources[system_id][0],
            resources[system_id][1],
            system_id,
        ),
    )
    if value.winner_system_id != expected_winner:
        raise ExperimentContractError("qualification winner violates the frozen tie-break")
    if resources[expected_winner] != (
        value.winner_parameter_count,
        value.winner_frozen_runtime_latency_ns,
    ):
        raise ExperimentContractError("qualification winner resources are inconsistent")
    _lower_sha256(value.score_rows_sha256, "score_rows_sha256")
    return _canonical_json_bytes(
        {
            "authority": value.authority,
            "completion_sha256s": list(value.completion_sha256s),
            "production": value.production,
            "result_claimed": value.result_claimed,
            "schema": value.schema,
            "score_rows_sha256": value.score_rows_sha256,
            "status": value.status,
            "system_mean_rows": [
                {
                    "mean_denominator": denominator,
                    "mean_numerator": numerator,
                    "system_id": system_id,
                }
                for system_id, numerator, denominator in value.system_mean_rows
            ],
            "system_resource_rows": [
                {
                    "frozen_runtime_latency_ns": latency_ns,
                    "parameter_count": parameter_count,
                    "system_id": system_id,
                }
                for system_id, parameter_count, latency_ns in value.system_resource_rows
            ],
            "winner_frozen_runtime_latency_ns": value.winner_frozen_runtime_latency_ns,
            "winner_parameter_count": value.winner_parameter_count,
            "winner_system_id": value.winner_system_id,
            "winner_system_name": value.winner_system_name,
            "winner_terminal_sha256s": list(value.winner_terminal_sha256s),
        }
    )


def validate_control_parameter_counts(counts: object) -> tuple[tuple[str, int], ...]:
    """Require controls 01--07 to remain within inclusive +/-1% of full 08."""

    if type(counts) is not dict or any(type(key) is not str for key in counts):
        raise TypeError("counts must be an exact str-keyed dict")
    expected = {f"{index:02d}" for index in range(1, 9)}
    if set(counts) != expected:
        raise ExperimentContractError("parameter census must contain exactly systems 01-08")
    if any(type(value) is not int or value <= 0 for value in counts.values()):
        raise ExperimentContractError("parameter counts must be positive exact ints")
    full = counts["08"]
    for system_id in (f"{index:02d}" for index in range(1, 8)):
        if Fraction(abs(counts[system_id] - full), full) > Fraction(1, 100):
            raise ExperimentContractError(
                f"system {system_id} exceeds the +/-1% full-control parameter bound"
            )
    return tuple((system_id, counts[system_id]) for system_id in sorted(expected))


__all__ = [
    "AUTHORITY",
    "BASE_SYSTEMS",
    "BaseQualification",
    "BaseScore",
    "EXECUTION_AUTHORIZED",
    "ExperimentContractError",
    "ExperimentPlan",
    "ExperimentRun",
    "FINAL_SYSTEMS",
    "HYPOTHESIS_COMPARISONS",
    "PLAN_SCHEMA",
    "PRODUCTION",
    "RESIDUAL_SYSTEMS",
    "RESULT_CLAIMED",
    "SCHEMA",
    "SEEDS",
    "STATUS",
    "base_run_ids",
    "build_experiment_plan",
    "canonical_experiment_matrix_bytes",
    "canonical_experiment_plan_bytes",
    "canonical_base_qualification_bytes",
    "experiment_matrix_sha256",
    "experiment_plan_sha256",
    "load_experiment_matrix",
    "parse_run_id",
    "qualify_base",
    "residual_run_ids",
    "validate_control_parameter_counts",
    "validate_experiment_matrix_bytes",
    "validate_experiment_plan_bytes",
]
