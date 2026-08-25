"""Content-bound resource, claim, and publication rendering contracts.

The module never reads a dataset, checkpoint, endpoint, or local output path.
It consumes already frozen aggregate objects and emits deterministic bytes for
the private host to write.  Real-result authority remains external.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math

import numpy as np

from phaseset_core.experiments import FINAL_SYSTEMS, HYPOTHESIS_COMPARISONS, SEEDS
from phaseset_core.statistics import (
    HYPOTHESIS_ORDER,
    ProvenancedHypothesisResult,
    REPORT_SCHEMA,
    StatisticalReport,
    canonical_statistical_report_bytes,
)


RESOURCE_SCHEMA = "phaseset-resource-report-v1"
CLAIM_SCHEMA = "phaseset-claim-report-v1"
PUBLICATION_SCHEMA = "phaseset-publication-bundle-v1"
PRIMARY_WORKLOAD = "SEALED_TEST_FULL_GALLERY_INFERENCE"
SCALING_WORKLOAD = "FROZEN_SYNTHETIC_K_SCALING"
FORMAL_MEASUREMENT_TYPE = "REAL_TARGET_RUNTIME"
SYNTHETIC_MEASUREMENT_TYPE = "SYNTHETIC_DATA_FREE"
SCALING_GROUP_SIZES = (3, 4, 8, 16, 27, 32, 128, 256)
PROFILE_SEED = SEEDS[0]


class PublicationContractError(ValueError):
    """A resource, claim, or rendered publication binding is malformed."""


@dataclass(frozen=True, slots=True)
class ResourceMeasurement:
    """One immutable profiler result, represented only by public-safe values."""

    workload: str
    system_id: str
    seed: int
    group_size: int | None
    parameter_count: int
    forward_flops: int
    peak_allocated_bytes: int
    window_count: int
    elapsed_ns: int
    checkpoint_sha256: str
    terminal_sha256: str
    environment_sha256: str
    device_sha256: str
    profiler_code_sha256: str
    measurement_artifact_sha256: str
    precision: str
    measurement_type: str


@dataclass(frozen=True, slots=True)
class ResourceReport:
    """Exact 27-row final census plus the registered full-model K curve."""

    measurements: tuple[ResourceMeasurement, ...]
    environment_sha256: str
    device_sha256: str
    profiler_code_sha256: str
    evaluation_aggregate_sha256: str
    precision: str
    measurement_type: str
    schema: str = RESOURCE_SCHEMA
    authority: int = 0
    production: bool = False
    result_claimed: bool = False
    status: str = "CONTENT_BOUND_RESOURCE_EVIDENCE_NO_AUTHORITY"


@dataclass(frozen=True, slots=True)
class ClaimDecision:
    hypothesis_id: str
    treatment_system_id: str
    baseline_system_id: str
    effect: float
    confidence_lower: float
    raw_p_value: float
    adjusted_p_value: float | None
    multiplicity_rejected: bool
    stable_superiority: bool


@dataclass(frozen=True, slots=True)
class ClaimReport:
    statistics_sha256: str
    decisions: tuple[ClaimDecision, ...]
    base_improvement_supported: bool
    structure_supported: bool
    topology_supported: bool
    phase_supported: bool
    maximum_claim: str
    permitted_claims: tuple[str, ...]
    schema: str = CLAIM_SCHEMA
    authority: int = 0
    production: bool = False
    result_claimed: bool = False
    status: str = "DETERMINISTIC_DECISION_TABLE_NO_PUBLICATION_AUTHORITY"


@dataclass(frozen=True, slots=True, repr=False)
class PublicationArtifact:
    relative_name: str
    media_type: str
    content: bytes = field(repr=False)


@dataclass(frozen=True, slots=True, repr=False)
class PublicationBundle:
    evaluation_aggregate_sha256: str
    statistics_sha256: str
    resource_report_sha256: str
    claim_report_sha256: str
    artifacts: tuple[PublicationArtifact, ...] = field(repr=False)
    manifest: bytes = field(repr=False)
    schema: str = PUBLICATION_SCHEMA
    authority: int = 0
    production: bool = False
    result_claimed: bool = False
    status: str = "CONTENT_BOUND_RENDER_NO_PUBLICATION_AUTHORITY"


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


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise PublicationContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _positive_int(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise PublicationContractError(f"{label} must be a positive exact int")
    return value


def _measurement_payload(value: ResourceMeasurement) -> dict[str, object]:
    if type(value) is not ResourceMeasurement:
        raise TypeError("measurement must be exact ResourceMeasurement")
    system_ids = {system_id for system_id, _ in FINAL_SYSTEMS}
    if value.system_id not in system_ids or value.seed not in SEEDS:
        raise PublicationContractError("resource system/seed identity is invalid")
    if value.workload == PRIMARY_WORKLOAD:
        if value.group_size is not None:
            raise PublicationContractError("primary workload group_size must be null")
    elif value.workload == SCALING_WORKLOAD:
        if (
            value.system_id != "08"
            or value.seed != PROFILE_SEED
            or value.group_size not in SCALING_GROUP_SIZES
        ):
            raise PublicationContractError("K-scaling measurement identity is invalid")
    else:
        raise PublicationContractError("resource workload is outside the frozen census")
    if value.precision not in {"FP32", "BF16"}:
        raise PublicationContractError("resource precision must be FP32 or BF16")
    if value.measurement_type not in {
        FORMAL_MEASUREMENT_TYPE,
        SYNTHETIC_MEASUREMENT_TYPE,
    }:
        raise PublicationContractError("resource measurement_type is invalid")
    for name in (
        "parameter_count",
        "forward_flops",
        "peak_allocated_bytes",
        "window_count",
        "elapsed_ns",
    ):
        _positive_int(getattr(value, name), name)
    for name in (
        "checkpoint_sha256",
        "terminal_sha256",
        "environment_sha256",
        "device_sha256",
        "profiler_code_sha256",
        "measurement_artifact_sha256",
    ):
        _lower_sha256(getattr(value, name), name)
    throughput = Fraction(value.window_count * 1_000_000_000, value.elapsed_ns)
    return {
        "checkpoint_sha256": value.checkpoint_sha256,
        "device_sha256": value.device_sha256,
        "elapsed_ns": value.elapsed_ns,
        "environment_sha256": value.environment_sha256,
        "forward_flops": value.forward_flops,
        "group_size": value.group_size,
        "measurement_artifact_sha256": value.measurement_artifact_sha256,
        "measurement_type": value.measurement_type,
        "parameter_count": value.parameter_count,
        "peak_allocated_bytes": value.peak_allocated_bytes,
        "precision": value.precision,
        "profiler_code_sha256": value.profiler_code_sha256,
        "seed": value.seed,
        "system_id": value.system_id,
        "terminal_sha256": value.terminal_sha256,
        "throughput_windows_per_second_denominator": throughput.denominator,
        "throughput_windows_per_second_numerator": throughput.numerator,
        "window_count": value.window_count,
        "workload": value.workload,
    }


def _validate_resource_census(
    measurements: object,
    *,
    evaluation_rows: tuple[dict[str, object], ...] | None,
) -> tuple[ResourceMeasurement, ...]:
    if type(measurements) is not tuple or any(
        type(row) is not ResourceMeasurement for row in measurements
    ):
        raise TypeError("measurements must be a tuple of exact ResourceMeasurement rows")
    expected_primary = tuple(
        (PRIMARY_WORKLOAD, system_id, seed, None)
        for system_id, _ in FINAL_SYSTEMS
        for seed in SEEDS
    )
    expected_scaling = tuple(
        (SCALING_WORKLOAD, "08", PROFILE_SEED, group_size)
        for group_size in SCALING_GROUP_SIZES
    )
    identities = tuple(
        (row.workload, row.system_id, row.seed, row.group_size) for row in measurements
    )
    if identities != expected_primary + expected_scaling:
        raise PublicationContractError(
            "resource rows must be ordered 27-system/seed then registered K scaling"
        )
    tuple(_measurement_payload(row) for row in measurements)
    if len({row.measurement_artifact_sha256 for row in measurements}) != len(
        measurements
    ):
        raise PublicationContractError("resource measurement artifacts must be unique")
    for name in (
        "environment_sha256",
        "device_sha256",
        "profiler_code_sha256",
        "precision",
        "measurement_type",
    ):
        if len({getattr(row, name) for row in measurements}) != 1:
            raise PublicationContractError(f"resource rows must share one {name}")
    for system_id, _ in FINAL_SYSTEMS:
        rows = tuple(
            row
            for row in measurements
            if row.workload == PRIMARY_WORKLOAD and row.system_id == system_id
        )
        if len({row.parameter_count for row in rows}) != 1:
            raise PublicationContractError(
                f"system {system_id} parameter count changed across seeds"
            )
        if len({row.forward_flops for row in rows}) != 1:
            raise PublicationContractError(
                f"system {system_id} forward FLOPs changed across seeds"
            )
    full_primary = next(
        row
        for row in measurements
        if row.workload == PRIMARY_WORKLOAD
        and row.system_id == "08"
        and row.seed == PROFILE_SEED
    )
    for row in measurements[len(expected_primary) :]:
        if (
            row.checkpoint_sha256 != full_primary.checkpoint_sha256
            or row.terminal_sha256 != full_primary.terminal_sha256
            or row.parameter_count != full_primary.parameter_count
        ):
            raise PublicationContractError(
                "K-scaling rows must reuse the registered full-model checkpoint"
            )
    if evaluation_rows is not None:
        if len(evaluation_rows) != len(expected_primary):
            raise PublicationContractError(
                "resource binding requires the exact 27 evaluation rows"
            )
        for measurement, score_row in zip(
            measurements[: len(expected_primary)],
            evaluation_rows,
            strict=True,
        ):
            if (
                score_row["system_id"] != measurement.system_id
                or score_row["seed"] != measurement.seed
                or score_row["checkpoint_sha256"]
                != measurement.checkpoint_sha256
                or score_row["terminal_sha256"] != measurement.terminal_sha256
                or score_row["environment_sha256"]
                != measurement.environment_sha256
            ):
                raise PublicationContractError(
                    "resource row differs from its evaluated checkpoint/terminal/environment"
                )
    return measurements


def build_resource_report(
    measurements: object,
    *,
    evaluation_aggregate: object,
) -> ResourceReport:
    """Validate the exact resource census and bind it to one evaluation aggregate."""

    if type(evaluation_aggregate) is not bytes or not evaluation_aggregate:
        raise TypeError("evaluation_aggregate must be nonempty exact bytes")
    parsed = _parse_evaluation_aggregate(evaluation_aggregate)
    score_rows = tuple(parsed["score_rows"])
    checked = _validate_resource_census(
        measurements,
        evaluation_rows=score_rows,
    )
    first = checked[0]
    return ResourceReport(
        measurements=checked,
        environment_sha256=first.environment_sha256,
        device_sha256=first.device_sha256,
        profiler_code_sha256=first.profiler_code_sha256,
        evaluation_aggregate_sha256=hashlib.sha256(evaluation_aggregate).hexdigest(),
        precision=first.precision,
        measurement_type=first.measurement_type,
    )


def canonical_resource_report_bytes(value: object) -> bytes:
    if type(value) is not ResourceReport:
        raise TypeError("value must be exact ResourceReport")
    checked = _validate_resource_census(
        value.measurements,
        evaluation_rows=None,
    )
    first = checked[0]
    for name in (
        "environment_sha256",
        "device_sha256",
        "profiler_code_sha256",
    ):
        if getattr(value, name) != getattr(first, name):
            raise PublicationContractError(f"resource report {name} is inconsistent")
    _lower_sha256(value.evaluation_aggregate_sha256, "evaluation_aggregate_sha256")
    if (
        value.precision != first.precision
        or value.measurement_type != first.measurement_type
        or value.schema != RESOURCE_SCHEMA
        or value.authority != 0
        or value.production is not False
        or value.result_claimed is not False
        or value.status != "CONTENT_BOUND_RESOURCE_EVIDENCE_NO_AUTHORITY"
    ):
        raise PublicationContractError("resource report identity is invalid")
    return _canonical_json_bytes(
        {
            "authority": value.authority,
            "device_sha256": value.device_sha256,
            "environment_sha256": value.environment_sha256,
            "evaluation_aggregate_sha256": value.evaluation_aggregate_sha256,
            "measurement_type": value.measurement_type,
            "measurements": [_measurement_payload(row) for row in value.measurements],
            "precision": value.precision,
            "production": value.production,
            "profiler_code_sha256": value.profiler_code_sha256,
            "result_claimed": value.result_claimed,
            "schema": value.schema,
            "status": value.status,
        }
    )


def build_claim_report(statistics: object) -> ClaimReport:
    """Apply the frozen fail-closed claim ladder to one formal report."""

    if type(statistics) is not StatisticalReport:
        raise TypeError("statistics must be exact StatisticalReport")
    statistics_raw = canonical_statistical_report_bytes(statistics)
    if (
        statistics.schema != REPORT_SCHEMA
        or statistics.provenance_complete is not True
        or any(
            type(row) is not ProvenancedHypothesisResult
            for row in statistics.hypotheses
        )
    ):
        raise PublicationContractError("claim decisions require a formal report")
    holm = {row.hypothesis_id: row for row in statistics.holm_adjustments}
    comparisons = {
        hypothesis_id: (treatment, baseline)
        for hypothesis_id, treatment, baseline in HYPOTHESIS_COMPARISONS
    }
    decisions: list[ClaimDecision] = []
    for row in statistics.hypotheses:
        assert type(row) is ProvenancedHypothesisResult
        adjusted = None if row.hypothesis_id == "H1" else holm[row.hypothesis_id]
        multiplicity_rejected = (
            row.raw_p_value <= 0.05
            if adjusted is None
            else adjusted.reject_at_0_05
        )
        stable = (
            row.effect > 0.0
            and row.confidence_lower > 0.0
            and multiplicity_rejected
        )
        treatment, baseline = comparisons[row.hypothesis_id]
        decisions.append(
            ClaimDecision(
                hypothesis_id=row.hypothesis_id,
                treatment_system_id=treatment,
                baseline_system_id=baseline,
                effect=row.effect,
                confidence_lower=row.confidence_lower,
                raw_p_value=row.raw_p_value,
                adjusted_p_value=(
                    None if adjusted is None else adjusted.adjusted_p_value
                ),
                multiplicity_rejected=multiplicity_rejected,
                stable_superiority=stable,
            )
        )
    if tuple(row.hypothesis_id for row in decisions) != HYPOTHESIS_ORDER:
        raise PublicationContractError("claim report hypothesis order is invalid")
    passed = {row.hypothesis_id: row.stable_superiority for row in decisions}
    base = passed["H1"]
    structure = base and passed["H2"]
    topology = base and passed["H5"] and passed["H7"]
    phase = base and passed["H8"]
    permitted: list[str] = []
    if base:
        permitted.append("BASE_IMPROVEMENT_ON_FROZEN_ENDPOINT")
    if structure:
        permitted.append("STRUCTURE_BEYOND_GENERIC_TOKEN_CAPACITY")
    if topology:
        permitted.append("INCIDENCE_TOPOLOGY_SUPPORTED")
    if phase:
        permitted.append("SIGNED_PHASE_INFORMATION_SUPPORTED")
    if topology and phase and structure:
        maximum = "TOPOLOGY_AND_PHASE_STRUCTURAL_CONTRIBUTION"
    elif topology:
        maximum = "INCIDENCE_TOPOLOGY_CONTRIBUTION"
    elif phase:
        maximum = "SIGNED_PHASE_CONTRIBUTION"
    elif structure:
        maximum = "STRUCTURE_BEYOND_GENERIC_CAPACITY"
    elif base:
        maximum = "BASE_IMPROVEMENT_ONLY_CAPACITY_COMPATIBLE"
    else:
        maximum = "NULL_OR_NEGATIVE_NO_IMPROVEMENT_CLAIM"
    return ClaimReport(
        statistics_sha256=hashlib.sha256(statistics_raw).hexdigest(),
        decisions=tuple(decisions),
        base_improvement_supported=base,
        structure_supported=structure,
        topology_supported=topology,
        phase_supported=phase,
        maximum_claim=maximum,
        permitted_claims=tuple(permitted),
    )


def canonical_claim_report_bytes(value: object) -> bytes:
    if type(value) is not ClaimReport:
        raise TypeError("value must be exact ClaimReport")
    _lower_sha256(value.statistics_sha256, "statistics_sha256")
    if (
        type(value.decisions) is not tuple
        or tuple(row.hypothesis_id for row in value.decisions) != HYPOTHESIS_ORDER
        or any(type(row) is not ClaimDecision for row in value.decisions)
    ):
        raise PublicationContractError("claim decisions are not the H1-H8 census")
    for row in value.decisions:
        for name in ("effect", "confidence_lower", "raw_p_value"):
            metric = getattr(row, name)
            if type(metric) is not float or not math.isfinite(metric):
                raise PublicationContractError(f"claim {name} must be finite float")
        if row.adjusted_p_value is not None and (
            type(row.adjusted_p_value) is not float
            or not math.isfinite(row.adjusted_p_value)
        ):
            raise PublicationContractError("claim adjusted p-value is invalid")
    if (
        value.schema != CLAIM_SCHEMA
        or value.authority != 0
        or value.production is not False
        or value.result_claimed is not False
        or value.status
        != "DETERMINISTIC_DECISION_TABLE_NO_PUBLICATION_AUTHORITY"
    ):
        raise PublicationContractError("claim report identity is invalid")
    return _canonical_json_bytes(
        {
            "authority": value.authority,
            "base_improvement_supported": value.base_improvement_supported,
            "decisions": [
                {
                    "adjusted_p_value_hex": (
                        None
                        if row.adjusted_p_value is None
                        else row.adjusted_p_value.hex()
                    ),
                    "baseline_system_id": row.baseline_system_id,
                    "confidence_lower_hex": row.confidence_lower.hex(),
                    "effect_hex": row.effect.hex(),
                    "hypothesis_id": row.hypothesis_id,
                    "multiplicity_rejected": row.multiplicity_rejected,
                    "raw_p_value_hex": row.raw_p_value.hex(),
                    "stable_superiority": row.stable_superiority,
                    "treatment_system_id": row.treatment_system_id,
                }
                for row in value.decisions
            ],
            "maximum_claim": value.maximum_claim,
            "permitted_claims": list(value.permitted_claims),
            "phase_supported": value.phase_supported,
            "production": value.production,
            "result_claimed": value.result_claimed,
            "schema": value.schema,
            "statistics_sha256": value.statistics_sha256,
            "status": value.status,
            "structure_supported": value.structure_supported,
            "topology_supported": value.topology_supported,
        }
    )


def _hex_probability(value: object, label: str) -> float:
    if type(value) is not str:
        raise PublicationContractError(f"{label} must be a canonical float hex string")
    try:
        parsed = float.fromhex(value)
    except ValueError as error:
        raise PublicationContractError(f"{label} is not a float hex string") from error
    if not math.isfinite(parsed) or not 0.0 <= parsed <= 1.0 or parsed.hex() != value:
        raise PublicationContractError(f"{label} is not a canonical probability")
    return parsed


def _parse_evaluation_aggregate(raw: bytes) -> dict[str, object]:
    try:
        parsed = json.loads(raw.decode("ascii"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise PublicationContractError("evaluation aggregate is not canonical JSON") from error
    if _canonical_json_bytes(parsed) != raw:
        raise PublicationContractError("evaluation aggregate bytes are not canonical")
    if (
        type(parsed) is not dict
        or parsed.get("schema") != "phaseset-evaluation-aggregate-v3"
        or parsed.get("split") != "test"
        or parsed.get("evaluation_type")
        != "REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL"
    ):
        raise PublicationContractError("publication requires the formal sealed-test aggregate")
    for name in (
        "caption_manifest_sha256",
        "census_provenance_sha256",
        "census_sha256",
        "evaluator_sha256",
        "hard_gallery_collection_sha256",
        "hard_gallery_freeze_binding_sha256",
        "score_execution_census_sha256",
        "sealed_test_consumption_sha256",
        "test_manifest_sha256",
    ):
        _lower_sha256(parsed.get(name), f"evaluation aggregate {name}")
    expected_identities = tuple(
        (system_id, seed) for system_id, _ in FINAL_SYSTEMS for seed in SEEDS
    )
    score_rows = parsed.get("score_rows")
    if type(score_rows) is not list or len(score_rows) != len(expected_identities):
        raise PublicationContractError("evaluation aggregate must contain exactly 27 score rows")
    required_score_keys = {
        "capacity_receipt_sha256",
        "checkpoint_sha256",
        "environment_sha256",
        "evaluation_report_sha256",
        "hard_text_to_motion_r1_hex",
        "primary_hex",
        "qualification_sha256",
        "run_manifest_sha256",
        "score_provenance_sha256",
        "score_sha256",
        "scoring_code_sha256",
        "seed",
        "system_id",
        "terminal_sha256",
        "text_tower_sha256",
    }
    digest_fields = tuple(
        name
        for name in required_score_keys
        if name.endswith("_sha256")
    )
    execution_rows: list[dict[str, object]] = []
    for ordinal, (row, identity) in enumerate(
        zip(score_rows, expected_identities, strict=True)
    ):
        if type(row) is not dict or set(row) != required_score_keys:
            raise PublicationContractError(
                f"evaluation score row {ordinal} has an invalid closed schema"
            )
        if (row["system_id"], row["seed"]) != identity:
            raise PublicationContractError(
                "evaluation score rows must be ordered systems 00-08 and fixed seeds"
            )
        for name in digest_fields:
            _lower_sha256(row[name], f"score_rows[{ordinal}].{name}")
        _hex_probability(row["primary_hex"], f"score_rows[{ordinal}].primary")
        _hex_probability(
            row["hard_text_to_motion_r1_hex"],
            f"score_rows[{ordinal}].hard_text_to_motion_r1",
        )
        execution_rows.append(
            {
                name: row[name]
                for name in (
                    "capacity_receipt_sha256",
                    "checkpoint_sha256",
                    "environment_sha256",
                    "qualification_sha256",
                    "run_manifest_sha256",
                    "scoring_code_sha256",
                    "seed",
                    "system_id",
                    "terminal_sha256",
                    "text_tower_sha256",
                )
            }
        )
    for name in (
        "checkpoint_sha256",
        "evaluation_report_sha256",
        "run_manifest_sha256",
        "score_provenance_sha256",
        "terminal_sha256",
    ):
        if len({row[name] for row in score_rows}) != len(score_rows):
            raise PublicationContractError(
                f"evaluation aggregate {name} census must contain 27 unique values"
            )
    for name in (
        "environment_sha256",
        "scoring_code_sha256",
        "text_tower_sha256",
    ):
        if len({row[name] for row in score_rows}) != 1:
            raise PublicationContractError(
                f"evaluation aggregate rows must share one {name}"
            )
    execution_census = hashlib.sha256(
        _canonical_json_bytes(
            {
                "rows": execution_rows,
                "schema": "phaseset-score-execution-census-v1",
            }
        )
    ).hexdigest()
    if execution_census != parsed["score_execution_census_sha256"]:
        raise PublicationContractError(
            "evaluation score-execution census digest does not match its 27 rows"
        )
    systems = parsed.get("systems")
    if (
        type(systems) is not list
        or [row.get("system_id") for row in systems if type(row) is dict]
        != [system_id for system_id, _ in FINAL_SYSTEMS]
    ):
        raise PublicationContractError("evaluation aggregate system census is invalid")
    required_system_keys = {
        "aggregation",
        "hard_text_to_motion_r1_mean_hex",
        "hard_text_to_motion_r1_population_std_hex",
        "primary_mean_hex",
        "primary_population_std_hex",
        "seed_evaluation_report_sha256s",
        "system_id",
    }
    for system_index, system in enumerate(systems):
        if type(system) is not dict or set(system) != required_system_keys:
            raise PublicationContractError("evaluation system summary schema is invalid")
        if (
            system["aggregation"]
            != "MEAN_OF_THREE_INDEPENDENT_SEED_METRICS_NO_LOGIT_ENSEMBLE"
        ):
            raise PublicationContractError("evaluation summary aggregation is invalid")
        start = system_index * len(SEEDS)
        rows = score_rows[start : start + len(SEEDS)]
        expected_reports = [row["evaluation_report_sha256"] for row in rows]
        if system["seed_evaluation_report_sha256s"] != expected_reports:
            raise PublicationContractError(
                "evaluation summary does not bind its three seed reports"
            )
        primary_values = np.asarray(
            [_hex_probability(row["primary_hex"], "seed primary") for row in rows],
            dtype=np.float64,
        )
        hard_values = np.asarray(
            [
                _hex_probability(
                    row["hard_text_to_motion_r1_hex"],
                    "seed hard R1",
                )
                for row in rows
            ],
            dtype=np.float64,
        )
        primary_mean = float(
            np.add.reduce(primary_values, dtype=np.float64) / len(SEEDS)
        )
        hard_mean = float(np.add.reduce(hard_values, dtype=np.float64) / len(SEEDS))
        expected_hex = {
            "primary_mean_hex": primary_mean.hex(),
            "primary_population_std_hex": float(
                np.sqrt(
                    np.add.reduce(
                        (primary_values - primary_mean) ** 2,
                        dtype=np.float64,
                    )
                    / len(SEEDS)
                )
            ).hex(),
            "hard_text_to_motion_r1_mean_hex": hard_mean.hex(),
            "hard_text_to_motion_r1_population_std_hex": float(
                np.sqrt(
                    np.add.reduce(
                        (hard_values - hard_mean) ** 2,
                        dtype=np.float64,
                    )
                    / len(SEEDS)
                )
            ).hex(),
        }
        for name, expected in expected_hex.items():
            if system[name] != expected:
                raise PublicationContractError(
                    f"evaluation system summary {name} was not rebuilt from seeds"
                )
    return parsed


def _table_one_tex(parsed: dict[str, object]) -> bytes:
    names = dict(FINAL_SYSTEMS)
    lines = [
        "% generated from phaseset-publication-bundle-v1",
        "\\begin{tabular}{clcc}",
        "\\toprule",
        "ID & System & Full primary (mean $\\pm$ pop. sd) & Hard T2M R@1 \\\\",
        "\\midrule",
    ]
    for row in parsed["systems"]:
        system_id = row["system_id"]
        primary = 100.0 * float.fromhex(row["primary_mean_hex"])
        primary_std = 100.0 * float.fromhex(row["primary_population_std_hex"])
        hard = 100.0 * float.fromhex(row["hard_text_to_motion_r1_mean_hex"])
        hard_std = 100.0 * float.fromhex(
            row["hard_text_to_motion_r1_population_std_hex"]
        )
        lines.append(
            f"{system_id} & {names[system_id]} & "
            f"{primary:.2f} $\\pm$ {primary_std:.2f} & "
            f"{hard:.2f} $\\pm$ {hard_std:.2f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return ("\n".join(lines) + "\n").encode("ascii")


def _table_two_tex(resource: ResourceReport) -> bytes:
    lines = [
        "% generated from phaseset-publication-bundle-v1",
        "\\begin{tabular}{crrr}",
        "\\toprule",
        "$K$ & FLOPs & Peak MiB & Windows/s \\\\",
        "\\midrule",
    ]
    for row in resource.measurements[27:]:
        throughput = Fraction(row.window_count * 1_000_000_000, row.elapsed_ns)
        lines.append(
            f"{row.group_size} & {row.forward_flops} & "
            f"{row.peak_allocated_bytes / 1048576.0:.2f} & "
            f"{float(throughput):.3f} \\\\"
        )
    lines.extend(("\\bottomrule", "\\end{tabular}"))
    return ("\n".join(lines) + "\n").encode("ascii")


def _summary_csv(parsed: dict[str, object]) -> bytes:
    lines = [
        "system_id,seed,primary_hex,hard_text_to_motion_r1_hex,score_sha256,"
        "evaluation_report_sha256"
    ]
    for row in parsed["score_rows"]:
        lines.append(
            ",".join(
                (
                    row["system_id"],
                    str(row["seed"]),
                    row["primary_hex"],
                    row["hard_text_to_motion_r1_hex"],
                    row["score_sha256"],
                    row["evaluation_report_sha256"],
                )
            )
        )
    return ("\n".join(lines) + "\n").encode("ascii")


def render_publication_bundle(
    *,
    evaluation_aggregate: object,
    statistics: object,
    resources: object,
    require_real_resources: bool = True,
) -> PublicationBundle:
    """Render all numeric publication surfaces from the same frozen aggregate."""

    if type(evaluation_aggregate) is not bytes or not evaluation_aggregate:
        raise TypeError("evaluation_aggregate must be nonempty exact bytes")
    if type(statistics) is not StatisticalReport:
        raise TypeError("statistics must be exact StatisticalReport")
    if type(resources) is not ResourceReport:
        raise TypeError("resources must be exact ResourceReport")
    if type(require_real_resources) is not bool:
        raise TypeError("require_real_resources must be exact bool")
    parsed = _parse_evaluation_aggregate(evaluation_aggregate)
    evaluation_sha256 = hashlib.sha256(evaluation_aggregate).hexdigest()
    statistics_raw = canonical_statistical_report_bytes(statistics)
    if any(
        type(row) is not ProvenancedHypothesisResult
        or row.aggregate_sha256 != evaluation_sha256
        for row in statistics.hypotheses
    ):
        raise PublicationContractError(
            "statistical report does not bind this evaluation aggregate"
        )
    if resources.evaluation_aggregate_sha256 != evaluation_sha256:
        raise PublicationContractError(
            "resource report does not bind this evaluation aggregate"
        )
    if require_real_resources and resources.measurement_type != FORMAL_MEASUREMENT_TYPE:
        raise PublicationContractError("formal rendering requires real target-runtime resources")
    resource_raw = canonical_resource_report_bytes(resources)
    claims = build_claim_report(statistics)
    claim_raw = canonical_claim_report_bytes(claims)
    artifacts = (
        PublicationArtifact(
            "results/evaluation-aggregate.json",
            "application/json",
            evaluation_aggregate,
        ),
        PublicationArtifact(
            "results/statistical-report.json",
            "application/json",
            statistics_raw,
        ),
        PublicationArtifact(
            "results/resource-report.json",
            "application/json",
            resource_raw,
        ),
        PublicationArtifact(
            "results/claim-report.json",
            "application/json",
            claim_raw,
        ),
        PublicationArtifact(
            "results/summary.csv",
            "text/csv",
            _summary_csv(parsed),
        ),
        PublicationArtifact(
            "paper/generated/table1.tex",
            "application/x-tex",
            _table_one_tex(parsed),
        ),
        PublicationArtifact(
            "paper/generated/table2.tex",
            "application/x-tex",
            _table_two_tex(resources),
        ),
    )
    artifact_rows = [
        {
            "media_type": artifact.media_type,
            "relative_name": artifact.relative_name,
            "sha256": hashlib.sha256(artifact.content).hexdigest(),
            "size": len(artifact.content),
        }
        for artifact in artifacts
    ]
    manifest = _canonical_json_bytes(
        {
            "artifacts": artifact_rows,
            "authority": 0,
            "claim_report_sha256": hashlib.sha256(claim_raw).hexdigest(),
            "evaluation_aggregate_sha256": evaluation_sha256,
            "production": False,
            "resource_report_sha256": hashlib.sha256(resource_raw).hexdigest(),
            "result_claimed": False,
            "schema": PUBLICATION_SCHEMA,
            "statistics_sha256": hashlib.sha256(statistics_raw).hexdigest(),
            "status": "CONTENT_BOUND_RENDER_NO_PUBLICATION_AUTHORITY",
        }
    )
    return PublicationBundle(
        evaluation_aggregate_sha256=evaluation_sha256,
        statistics_sha256=hashlib.sha256(statistics_raw).hexdigest(),
        resource_report_sha256=hashlib.sha256(resource_raw).hexdigest(),
        claim_report_sha256=hashlib.sha256(claim_raw).hexdigest(),
        artifacts=artifacts,
        manifest=manifest,
    )


def canonical_publication_manifest_bytes(value: object) -> bytes:
    if type(value) is not PublicationBundle:
        raise TypeError("value must be exact PublicationBundle")
    for name in (
        "evaluation_aggregate_sha256",
        "statistics_sha256",
        "resource_report_sha256",
        "claim_report_sha256",
    ):
        _lower_sha256(getattr(value, name), name)
    if (
        type(value.artifacts) is not tuple
        or any(type(row) is not PublicationArtifact for row in value.artifacts)
        or len({row.relative_name for row in value.artifacts}) != len(value.artifacts)
        or type(value.manifest) is not bytes
        or not value.manifest
        or value.schema != PUBLICATION_SCHEMA
        or value.authority != 0
        or value.production is not False
        or value.result_claimed is not False
        or value.status != "CONTENT_BOUND_RENDER_NO_PUBLICATION_AUTHORITY"
    ):
        raise PublicationContractError("publication bundle identity is invalid")
    parsed = json.loads(value.manifest.decode("ascii"))
    if _canonical_json_bytes(parsed) != value.manifest:
        raise PublicationContractError("publication manifest is not canonical")
    expected_rows = [
        {
            "media_type": row.media_type,
            "relative_name": row.relative_name,
            "sha256": hashlib.sha256(row.content).hexdigest(),
            "size": len(row.content),
        }
        for row in value.artifacts
    ]
    if (
        parsed.get("artifacts") != expected_rows
        or parsed.get("evaluation_aggregate_sha256")
        != value.evaluation_aggregate_sha256
        or parsed.get("statistics_sha256") != value.statistics_sha256
        or parsed.get("resource_report_sha256") != value.resource_report_sha256
        or parsed.get("claim_report_sha256") != value.claim_report_sha256
    ):
        raise PublicationContractError("publication manifest differs from artifact contents")
    return value.manifest


__all__ = [
    "CLAIM_SCHEMA",
    "ClaimDecision",
    "ClaimReport",
    "FORMAL_MEASUREMENT_TYPE",
    "PRIMARY_WORKLOAD",
    "PROFILE_SEED",
    "PUBLICATION_SCHEMA",
    "PublicationArtifact",
    "PublicationBundle",
    "PublicationContractError",
    "RESOURCE_SCHEMA",
    "ResourceMeasurement",
    "ResourceReport",
    "SCALING_GROUP_SIZES",
    "SCALING_WORKLOAD",
    "SYNTHETIC_MEASUREMENT_TYPE",
    "build_claim_report",
    "build_resource_report",
    "canonical_claim_report_bytes",
    "canonical_publication_manifest_bytes",
    "canonical_resource_report_bytes",
    "render_publication_bundle",
]
