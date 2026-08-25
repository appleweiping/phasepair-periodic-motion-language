"""Variable-caption metrics and preregistered PhaseSet inference.

The bootstrap resampling unit is one capture, not one caption.  All captions
belonging to a capture remain together in every resample, so captures with more
captions cannot silently receive greater weight.  The public entry point always
uses exactly 100,000 paired draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
import hashlib
import json
import math
import re

import numpy as np

from phaseset_core.experiments import (
    HYPOTHESIS_COMPARISONS as _REGISTERED_HYPOTHESIS_COMPARISONS,
)


BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SCHEMA = "phaseset-paired-capture-bootstrap-v1"
REPORT_SCHEMA = "phaseset-statistical-report-v1"
HYPOTHESIS_COMPARISONS = tuple(_REGISTERED_HYPOTHESIS_COMPARISONS)
HYPOTHESIS_ORDER = tuple(row[0] for row in HYPOTHESIS_COMPARISONS)
HOLM_FAMILY = tuple(f"H{index}" for index in range(2, 9))
STATUS = "CALLER_FROZEN_DATA_NO_EXECUTION_AUTHORITY"
_CAPTURE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class StatisticalContractError(ValueError):
    """Raised when statistical identities or values fail closed."""


@dataclass(frozen=True, slots=True)
class PairedCapture:
    """Paired binary caption outcomes for one immutable capture identity."""

    capture_id: str
    treatment_caption_hits: tuple[int, ...]
    control_caption_hits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VariableCaptionSummary:
    capture_count: int
    total_caption_count: int
    minimum_caption_count: int
    maximum_caption_count: int
    mean_caption_count_numerator: int
    mean_caption_count_denominator: int
    capture_rows_sha256: str


@dataclass(frozen=True, slots=True)
class BootstrapResult:
    draws: int
    seed: int
    capture_count: int
    observed_effect: float
    confidence_lower: float
    confidence_upper: float
    two_sided_p_value: float
    capture_rows_sha256: str
    index_stream_sha256: str
    schema: str = BOOTSTRAP_SCHEMA
    status: str = STATUS


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    hypothesis_id: str
    effect: float
    raw_p_value: float
    bootstrap_sha256: str


@dataclass(frozen=True, slots=True)
class HolmAdjustment:
    hypothesis_id: str
    raw_p_value: float
    sorted_rank: int
    adjusted_p_value: float
    reject_at_0_05: bool


@dataclass(frozen=True, slots=True)
class StatisticalReport:
    hypotheses: tuple[HypothesisResult, ...]
    holm_adjustments: tuple[HolmAdjustment, ...]
    schema: str = REPORT_SCHEMA
    authority: int = 0
    production: bool = False
    result_claimed: bool = False
    status: str = STATUS


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
    if type(value) is not str:
        raise TypeError(f"{label} must be exact built-in str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise StatisticalContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _probability(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be an exact int or float")
    checked = float(value)
    if not math.isfinite(checked) or not 0.0 <= checked <= 1.0:
        raise StatisticalContractError(f"{label} must be finite in [0,1]")
    return checked


def _effect(value: object, label: str) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be an exact int or float")
    checked = float(value)
    if not math.isfinite(checked) or not -1.0 <= checked <= 1.0:
        raise StatisticalContractError(f"{label} must be finite in [-1,1]")
    return checked


def _validate_hits(value: object, label: str) -> tuple[int, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a nonempty exact tuple")
    if any(type(item) is not int or item not in {0, 1} for item in value):
        raise StatisticalContractError(f"{label} values must be exact integer zero or one")
    return value


def _normalize_captures(captures: object) -> tuple[PairedCapture, ...]:
    if type(captures) is not tuple or not captures:
        raise TypeError("captures must be a nonempty exact tuple")
    by_id: dict[str, PairedCapture] = {}
    for ordinal, value in enumerate(captures):
        if type(value) is not PairedCapture:
            raise TypeError(f"captures[{ordinal}] must be exact PairedCapture")
        if type(value.capture_id) is not str or _CAPTURE_ID.fullmatch(value.capture_id) is None:
            raise StatisticalContractError(f"captures[{ordinal}].capture_id is not canonical")
        treatment = _validate_hits(
            value.treatment_caption_hits,
            f"captures[{ordinal}].treatment_caption_hits",
        )
        control = _validate_hits(
            value.control_caption_hits,
            f"captures[{ordinal}].control_caption_hits",
        )
        if len(treatment) != len(control):
            raise StatisticalContractError(
                f"captures[{ordinal}] does not pair the same caption census"
            )
        if value.capture_id in by_id:
            raise StatisticalContractError(f"duplicate capture_id: {value.capture_id}")
        by_id[value.capture_id] = PairedCapture(value.capture_id, treatment, control)
    return tuple(by_id[key] for key in sorted(by_id))


def _capture_rows_bytes(captures: tuple[PairedCapture, ...]) -> bytes:
    return _canonical_json_bytes(
        [
            {
                "capture_id": row.capture_id,
                "control_caption_hits": list(row.control_caption_hits),
                "treatment_caption_hits": list(row.treatment_caption_hits),
            }
            for row in captures
        ]
    )


def summarize_variable_captions(captures: object) -> VariableCaptionSummary:
    """Summarize caption cardinality without exposing per-capture rows."""

    checked = _normalize_captures(captures)
    counts = tuple(len(row.treatment_caption_hits) for row in checked)
    return VariableCaptionSummary(
        capture_count=len(checked),
        total_caption_count=sum(counts),
        minimum_caption_count=min(counts),
        maximum_caption_count=max(counts),
        mean_caption_count_numerator=sum(counts),
        mean_caption_count_denominator=len(counts),
        capture_rows_sha256=hashlib.sha256(_capture_rows_bytes(checked)).hexdigest(),
    )


def _capture_effects(captures: tuple[PairedCapture, ...]) -> np.ndarray:
    fractions = tuple(
        Fraction(
            sum(row.treatment_caption_hits) - sum(row.control_caption_hits),
            len(row.treatment_caption_hits),
        )
        for row in captures
    )
    return np.asarray([float(value) for value in fractions], dtype=np.float64)


def paired_capture_effect(captures: object) -> float:
    """Return the capture-macro treatment-minus-control effect."""

    checked = _normalize_captures(captures)
    exact = sum(
        (
            Fraction(
                sum(row.treatment_caption_hits) - sum(row.control_caption_hits),
                len(row.treatment_caption_hits),
            )
            for row in checked
        ),
        start=Fraction(0, 1),
    ) / len(checked)
    return float(exact)


def paired_capture_bootstrap(captures: object, *, seed: object) -> BootstrapResult:
    """Run exactly 100,000 deterministic paired capture-level draws."""

    checked = _normalize_captures(captures)
    if len(checked) < 2:
        raise StatisticalContractError("paired bootstrap requires at least two captures")
    if type(seed) is not int or not 0 <= seed <= (1 << 63) - 1:
        raise TypeError("seed must be an exact nonnegative 63-bit int")

    effects = _capture_effects(checked)
    rng = np.random.Generator(np.random.PCG64(seed))
    samples = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    index_hash = hashlib.sha256()
    capture_count = len(checked)
    maximum_elements = 1_000_000
    chunk_size = max(1, min(4096, maximum_elements // capture_count))
    offset = 0
    while offset < BOOTSTRAP_DRAWS:
        current = min(chunk_size, BOOTSTRAP_DRAWS - offset)
        indices = rng.integers(
            0,
            capture_count,
            size=(current, capture_count),
            dtype=np.uint64,
        )
        canonical_indices = np.asarray(indices, dtype="<u8", order="C")
        index_hash.update(canonical_indices.tobytes(order="C"))
        samples[offset : offset + current] = effects[indices].mean(axis=1)
        offset += current

    samples.sort()
    lower = float(samples[2_499])
    upper = float(samples[97_499])
    nonpositive = int(np.count_nonzero(samples <= 0.0))
    nonnegative = int(np.count_nonzero(samples >= 0.0))
    p_value = min(
        1.0,
        2.0 * min((nonpositive + 1) / 100_001, (nonnegative + 1) / 100_001),
    )
    capture_digest = hashlib.sha256(_capture_rows_bytes(checked)).hexdigest()
    return BootstrapResult(
        draws=BOOTSTRAP_DRAWS,
        seed=seed,
        capture_count=capture_count,
        observed_effect=float(effects.mean()),
        confidence_lower=lower,
        confidence_upper=upper,
        two_sided_p_value=p_value,
        capture_rows_sha256=capture_digest,
        index_stream_sha256=index_hash.hexdigest(),
    )


def canonical_bootstrap_result_bytes(result: object) -> bytes:
    if type(result) is not BootstrapResult:
        raise TypeError("result must be exact BootstrapResult")
    if (
        result.draws != BOOTSTRAP_DRAWS
        or type(result.seed) is not int
        or type(result.capture_count) is not int
        or result.capture_count < 2
        or result.schema != BOOTSTRAP_SCHEMA
        or result.status != STATUS
    ):
        raise StatisticalContractError("bootstrap result identity is invalid")
    for label, value in (
        ("observed_effect", result.observed_effect),
        ("confidence_lower", result.confidence_lower),
        ("confidence_upper", result.confidence_upper),
    ):
        _effect(value, label)
    _probability(result.two_sided_p_value, "two_sided_p_value")
    _lower_sha256(result.capture_rows_sha256, "capture_rows_sha256")
    _lower_sha256(result.index_stream_sha256, "index_stream_sha256")
    return _canonical_json_bytes(
        {
            "capture_count": result.capture_count,
            "capture_rows_sha256": result.capture_rows_sha256,
            "confidence_lower_hex": result.confidence_lower.hex(),
            "confidence_upper_hex": result.confidence_upper.hex(),
            "draws": result.draws,
            "index_stream_sha256": result.index_stream_sha256,
            "observed_effect_hex": result.observed_effect.hex(),
            "schema": result.schema,
            "seed": result.seed,
            "status": result.status,
            "two_sided_p_value_hex": result.two_sided_p_value.hex(),
        }
    )


def holm_h2_h8(p_values: object) -> tuple[HolmAdjustment, ...]:
    """Apply step-down Holm correction to exactly H2--H8."""

    if type(p_values) is not dict or any(type(key) is not str for key in p_values):
        raise TypeError("p_values must be an exact str-keyed dict")
    if set(p_values) != set(HOLM_FAMILY):
        raise StatisticalContractError("Holm family must be exactly H2-H8")
    normalized = {
        hypothesis_id: _probability(p_values[hypothesis_id], hypothesis_id)
        for hypothesis_id in HOLM_FAMILY
    }
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    family_size = len(ordered)
    running_max = 0.0
    by_id: dict[str, HolmAdjustment] = {}
    for rank, (hypothesis_id, raw_p) in enumerate(ordered, start=1):
        adjusted = max(running_max, min(1.0, (family_size - rank + 1) * raw_p))
        running_max = adjusted
        by_id[hypothesis_id] = HolmAdjustment(
            hypothesis_id=hypothesis_id,
            raw_p_value=raw_p,
            sorted_rank=rank,
            adjusted_p_value=adjusted,
            reject_at_0_05=adjusted <= 0.05,
        )
    return tuple(by_id[hypothesis_id] for hypothesis_id in HOLM_FAMILY)


def build_statistical_report(hypotheses: object) -> StatisticalReport:
    if type(hypotheses) is not tuple:
        raise TypeError("hypotheses must be an exact tuple")
    by_id: dict[str, HypothesisResult] = {}
    for ordinal, value in enumerate(hypotheses):
        if type(value) is not HypothesisResult:
            raise TypeError(f"hypotheses[{ordinal}] must be exact HypothesisResult")
        if value.hypothesis_id not in HYPOTHESIS_ORDER or value.hypothesis_id in by_id:
            raise StatisticalContractError("hypothesis identities must be unique H1-H8")
        normalized = HypothesisResult(
            hypothesis_id=value.hypothesis_id,
            effect=_effect(value.effect, f"{value.hypothesis_id}.effect"),
            raw_p_value=_probability(value.raw_p_value, f"{value.hypothesis_id}.raw_p"),
            bootstrap_sha256=_lower_sha256(
                value.bootstrap_sha256, f"{value.hypothesis_id}.bootstrap_sha256"
            ),
        )
        by_id[normalized.hypothesis_id] = normalized
    if set(by_id) != set(HYPOTHESIS_ORDER):
        raise StatisticalContractError("hypothesis census must be exactly H1-H8")
    ordered = tuple(by_id[hypothesis_id] for hypothesis_id in HYPOTHESIS_ORDER)
    holm = holm_h2_h8(
        {row.hypothesis_id: row.raw_p_value for row in ordered if row.hypothesis_id in HOLM_FAMILY}
    )
    return StatisticalReport(hypotheses=ordered, holm_adjustments=holm)


def canonical_statistical_report_bytes(report: object) -> bytes:
    if type(report) is not StatisticalReport:
        raise TypeError("report must be exact StatisticalReport")
    rebuilt = build_statistical_report(report.hypotheses)
    if report != rebuilt:
        raise StatisticalContractError("statistical report differs from canonical H1-H8 form")
    comparison_by_id = {
        hypothesis_id: (treatment_system_id, baseline_system_id)
        for hypothesis_id, treatment_system_id, baseline_system_id in HYPOTHESIS_COMPARISONS
    }
    return _canonical_json_bytes(
        {
            "authority": report.authority,
            "holm_adjustments": [
                {
                    "adjusted_p_value_hex": row.adjusted_p_value.hex(),
                    "hypothesis_id": row.hypothesis_id,
                    "raw_p_value_hex": row.raw_p_value.hex(),
                    "reject_at_0_05": row.reject_at_0_05,
                    "sorted_rank": row.sorted_rank,
                }
                for row in report.holm_adjustments
            ],
            "holm_family": list(HOLM_FAMILY),
            "hypotheses": [
                {
                    "baseline_system_id": comparison_by_id[row.hypothesis_id][1],
                    "bootstrap_sha256": row.bootstrap_sha256,
                    "effect_hex": row.effect.hex(),
                    "hypothesis_id": row.hypothesis_id,
                    "raw_p_value_hex": row.raw_p_value.hex(),
                    "treatment_system_id": comparison_by_id[row.hypothesis_id][0],
                }
                for row in report.hypotheses
            ],
            "production": report.production,
            "result_claimed": report.result_claimed,
            "schema": report.schema,
            "status": report.status,
        }
    )


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SCHEMA",
    "BootstrapResult",
    "HOLM_FAMILY",
    "HYPOTHESIS_COMPARISONS",
    "HYPOTHESIS_ORDER",
    "HolmAdjustment",
    "HypothesisResult",
    "PairedCapture",
    "REPORT_SCHEMA",
    "STATUS",
    "StatisticalContractError",
    "StatisticalReport",
    "VariableCaptionSummary",
    "build_statistical_report",
    "canonical_bootstrap_result_bytes",
    "canonical_statistical_report_bytes",
    "holm_h2_h8",
    "paired_capture_bootstrap",
    "paired_capture_effect",
    "summarize_variable_captions",
]
