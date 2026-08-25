"""Bidirectional capture-macro metrics and preregistered PhaseSet inference.

The primary endpoint is the mean of text-to-motion R@1 and any-positive
motion-to-text R@1.  Both directions are averaged inside a capture before
captures are macro-averaged.  The bootstrap resampling unit is one capture, so
captures with more captions or windows cannot silently receive greater weight.
The public entry point always uses exactly 100,000 paired draws.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import json
import math
import re

import numpy as np

from phaseset_core.evaluation import (
    BidirectionalMetrics,
    CaptureR1Contribution,
    DirectionMetrics,
    EvaluationReport,
)
from phaseset_core.experiments import (
    HYPOTHESIS_COMPARISONS as _REGISTERED_HYPOTHESIS_COMPARISONS,
    SEEDS as _REGISTERED_TRAINING_SEEDS,
)


BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SCHEMA = "phaseset-paired-capture-bootstrap-v2"
SEED_BLOCKED_BOOTSTRAP_SCHEMA = "phaseset-seed-blocked-capture-bootstrap-v1"
REPORT_SCHEMA = "phaseset-statistical-report-v4"
SYNTHETIC_REPORT_SCHEMA = "phaseset-statistical-report-synthetic-unbound-v1"
PRIMARY_ENDPOINT_ID = "capture_macro_half_t2m_r1_plus_m2t_any_positive_r1"
DIRECTION_ID = "treatment_minus_baseline"
HYPOTHESIS_COMPARISONS = tuple(_REGISTERED_HYPOTHESIS_COMPARISONS)
HYPOTHESIS_ORDER = tuple(row[0] for row in HYPOTHESIS_COMPARISONS)
TRAINING_SEEDS = tuple(_REGISTERED_TRAINING_SEEDS)
HOLM_FAMILY = tuple(f"H{index}" for index in range(2, 9))
STATUS = "CALLER_FROZEN_DATA_NO_EXECUTION_AUTHORITY"
SYNTHETIC_STATUS = "SYNTHETIC_UNBOUND_PROVENANCE_NO_SCIENTIFIC_RESULT"
_CAPTURE_ID = re.compile(r"[0-9a-f]{64}\Z")


class StatisticalContractError(ValueError):
    """Raised when statistical identities or values fail closed."""


@dataclass(frozen=True, slots=True)
class PairedCapture:
    """Paired two-direction R@1 outcomes for one immutable capture cluster."""

    capture_id: str
    caption_ids: tuple[str, ...]
    motion_ids: tuple[str, ...]
    treatment_text_to_motion_hits: tuple[int, ...]
    control_text_to_motion_hits: tuple[int, ...]
    treatment_motion_to_text_hits: tuple[int, ...]
    control_motion_to_text_hits: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class VariableCaptionSummary:
    capture_count: int
    total_caption_count: int
    minimum_caption_count: int
    maximum_caption_count: int
    mean_caption_count_numerator: int
    mean_caption_count_denominator: int
    total_motion_query_count: int
    minimum_motion_query_count: int
    maximum_motion_query_count: int
    mean_motion_query_count_numerator: int
    mean_motion_query_count_denominator: int
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
    primary_endpoint_id: str = PRIMARY_ENDPOINT_ID
    schema: str = BOOTSTRAP_SCHEMA
    status: str = STATUS


@dataclass(frozen=True, slots=True)
class SeedBlockedBootstrapResult:
    """Capture bootstrap after fixed-order averaging of paired training seeds."""

    draws: int
    resampling_seed: int
    training_seeds: tuple[int, ...]
    capture_count: int
    observed_effect: float
    confidence_lower: float
    confidence_upper: float
    two_sided_p_value: float
    seed_block_rows_sha256: str
    capture_census_sha256: str
    index_stream_sha256: str
    primary_endpoint_id: str = PRIMARY_ENDPOINT_ID
    schema: str = SEED_BLOCKED_BOOTSTRAP_SCHEMA
    status: str = STATUS


@dataclass(frozen=True, slots=True)
class HypothesisResult:
    """Legacy synthetic-only row retained for the tiny execution fixture.

    These caller-supplied values can never produce ``REPORT_SCHEMA``.  Formal
    H1--H8 reports are built from :class:`HypothesisComparison` objects and use
    :class:`ProvenancedHypothesisResult` rows instead.
    """

    hypothesis_id: str
    effect: float
    raw_p_value: float
    bootstrap_sha256: str


@dataclass(frozen=True, slots=True, repr=False)
class HypothesisComparison:
    """Raw, private comparison evidence consumed by the closed report builder.

    Artifact bytes are hashed inside this module; callers cannot inject their
    own aggregate/evaluator digest.  Evaluation contributions are paired and
    bootstrapped again here.  Canonical public reports retain only counts and
    digests, never these reports, artifact bytes, or query commitments.
    """

    hypothesis_id: str
    treatment_system_id: str
    baseline_system_id: str
    training_seeds: tuple[int, ...]
    treatment_evaluations: tuple[EvaluationReport, ...]
    baseline_evaluations: tuple[EvaluationReport, ...]
    resampling_seed: int
    aggregate_artifact: bytes
    evaluator_artifact: bytes


@dataclass(frozen=True, slots=True)
class ProvenancedHypothesisResult:
    hypothesis_id: str
    treatment_system_id: str
    baseline_system_id: str
    primary_endpoint_id: str
    direction: str
    effect: float
    confidence_lower: float
    confidence_upper: float
    raw_p_value: float
    resampling_seed: int
    training_seeds: tuple[int, ...]
    seed_count: int
    draws: int
    capture_count: int
    caption_count: int
    motion_count: int
    component_count: int
    evaluation_type: str
    census_provenance_sha256: str
    treatment_score_provenance_sha256s: tuple[str, ...]
    baseline_score_provenance_sha256s: tuple[str, ...]
    aggregate_sha256: str
    evaluator_sha256: str
    seed_block_rows_sha256: str
    capture_census_sha256: str
    index_stream_sha256: str
    bootstrap_sha256: str
    treatment_evaluation_sha256s: tuple[str, ...]
    baseline_evaluation_sha256s: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HolmAdjustment:
    hypothesis_id: str
    raw_p_value: float
    sorted_rank: int
    adjusted_p_value: float
    reject_at_0_05: bool


@dataclass(frozen=True, slots=True)
class StatisticalReport:
    hypotheses: tuple[HypothesisResult | ProvenancedHypothesisResult, ...]
    holm_adjustments: tuple[HolmAdjustment, ...]
    provenance_complete: bool = False
    schema: str = SYNTHETIC_REPORT_SCHEMA
    authority: int = 0
    production: bool = False
    result_claimed: bool = False
    status: str = SYNTHETIC_STATUS
    _comparisons: tuple[HypothesisComparison, ...] | None = field(
        default=None,
        repr=False,
        compare=False,
    )


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


def _validate_ids(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a nonempty exact tuple")
    if any(type(item) is not str or _CAPTURE_ID.fullmatch(item) is None for item in value):
        raise StatisticalContractError(f"{label} values must be lowercase SHA-256 identities")
    if value != tuple(sorted(value)) or len(value) != len(set(value)):
        raise StatisticalContractError(f"{label} must be unique and canonically sorted")
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
        caption_ids = _validate_ids(
            value.caption_ids,
            f"captures[{ordinal}].caption_ids",
        )
        motion_ids = _validate_ids(
            value.motion_ids,
            f"captures[{ordinal}].motion_ids",
        )
        treatment_text = _validate_hits(
            value.treatment_text_to_motion_hits,
            f"captures[{ordinal}].treatment_text_to_motion_hits",
        )
        control_text = _validate_hits(
            value.control_text_to_motion_hits,
            f"captures[{ordinal}].control_text_to_motion_hits",
        )
        treatment_motion = _validate_hits(
            value.treatment_motion_to_text_hits,
            f"captures[{ordinal}].treatment_motion_to_text_hits",
        )
        control_motion = _validate_hits(
            value.control_motion_to_text_hits,
            f"captures[{ordinal}].control_motion_to_text_hits",
        )
        if (
            len(treatment_text) != len(caption_ids)
            or len(control_text) != len(caption_ids)
        ):
            raise StatisticalContractError(
                f"captures[{ordinal}] does not pair the same text-query census"
            )
        if (
            len(treatment_motion) != len(motion_ids)
            or len(control_motion) != len(motion_ids)
        ):
            raise StatisticalContractError(
                f"captures[{ordinal}] does not pair the same motion-query census"
            )
        if value.capture_id in by_id:
            raise StatisticalContractError(f"duplicate capture_id: {value.capture_id}")
        by_id[value.capture_id] = PairedCapture(
            capture_id=value.capture_id,
            caption_ids=caption_ids,
            motion_ids=motion_ids,
            treatment_text_to_motion_hits=treatment_text,
            control_text_to_motion_hits=control_text,
            treatment_motion_to_text_hits=treatment_motion,
            control_motion_to_text_hits=control_motion,
        )
    return tuple(by_id[key] for key in sorted(by_id))


def _contribution_map(
    value: object,
    label: str,
) -> dict[bytes, CaptureR1Contribution]:
    if type(value) is not tuple or not value:
        raise TypeError(f"{label} must be a nonempty exact tuple")
    output: dict[bytes, CaptureR1Contribution] = {}
    for ordinal, row in enumerate(value):
        if type(row) is not CaptureR1Contribution:
            raise TypeError(f"{label}[{ordinal}] must be exact CaptureR1Contribution")
        if type(row.capture_commitment) is not bytes or len(row.capture_commitment) != 32:
            raise StatisticalContractError(f"{label}[{ordinal}] capture commitment is invalid")
        for identities, identity_label in (
            (row.caption_commitments, "caption"),
            (row.motion_commitments, "motion"),
        ):
            if (
                type(identities) is not tuple
                or not identities
                or any(type(item) is not bytes or len(item) != 32 for item in identities)
                or identities != tuple(sorted(identities))
                or len(identities) != len(set(identities))
            ):
                raise StatisticalContractError(
                    f"{label}[{ordinal}] {identity_label} commitments are not canonical"
                )
        text_hits = _validate_hits(
            row.text_to_motion_hits,
            f"{label}[{ordinal}].text_to_motion_hits",
        )
        motion_hits = _validate_hits(
            row.motion_to_text_hits,
            f"{label}[{ordinal}].motion_to_text_hits",
        )
        if len(text_hits) != len(row.caption_commitments):
            raise StatisticalContractError(
                f"{label}[{ordinal}] text hits do not match caption commitments"
            )
        if len(motion_hits) != len(row.motion_commitments):
            raise StatisticalContractError(
                f"{label}[{ordinal}] motion hits do not match motion commitments"
            )
        if row.capture_commitment in output:
            raise StatisticalContractError(f"{label} has duplicate capture commitment")
        output[row.capture_commitment] = row
    return output


def paired_captures_from_contributions(
    treatment: object,
    control: object,
) -> tuple[PairedCapture, ...]:
    """Pair two scored galleries only when their capture/query census is identical."""

    treatment_by_capture = _contribution_map(treatment, "treatment")
    control_by_capture = _contribution_map(control, "control")
    if set(treatment_by_capture) != set(control_by_capture):
        raise StatisticalContractError("treatment/control capture census differs")
    rows: list[PairedCapture] = []
    for commitment in sorted(treatment_by_capture):
        treatment_row = treatment_by_capture[commitment]
        control_row = control_by_capture[commitment]
        if (
            treatment_row.caption_commitments != control_row.caption_commitments
            or treatment_row.motion_commitments != control_row.motion_commitments
        ):
            raise StatisticalContractError(
                "treatment/control query identities differ within a capture"
            )
        rows.append(
            PairedCapture(
                capture_id=commitment.hex(),
                caption_ids=tuple(item.hex() for item in treatment_row.caption_commitments),
                motion_ids=tuple(item.hex() for item in treatment_row.motion_commitments),
                treatment_text_to_motion_hits=treatment_row.text_to_motion_hits,
                control_text_to_motion_hits=control_row.text_to_motion_hits,
                treatment_motion_to_text_hits=treatment_row.motion_to_text_hits,
                control_motion_to_text_hits=control_row.motion_to_text_hits,
            )
        )
    return _normalize_captures(tuple(rows))


def _capture_rows_bytes(captures: tuple[PairedCapture, ...]) -> bytes:
    return _canonical_json_bytes(
        [
            {
                "capture_id": row.capture_id,
                "caption_ids": list(row.caption_ids),
                "control_motion_to_text_hits": list(row.control_motion_to_text_hits),
                "control_text_to_motion_hits": list(row.control_text_to_motion_hits),
                "motion_ids": list(row.motion_ids),
                "treatment_motion_to_text_hits": list(row.treatment_motion_to_text_hits),
                "treatment_text_to_motion_hits": list(row.treatment_text_to_motion_hits),
            }
            for row in captures
        ]
    )


def summarize_variable_captions(captures: object) -> VariableCaptionSummary:
    """Summarize caption cardinality without exposing per-capture rows."""

    checked = _normalize_captures(captures)
    caption_counts = tuple(len(row.treatment_text_to_motion_hits) for row in checked)
    motion_counts = tuple(len(row.treatment_motion_to_text_hits) for row in checked)
    return VariableCaptionSummary(
        capture_count=len(checked),
        total_caption_count=sum(caption_counts),
        minimum_caption_count=min(caption_counts),
        maximum_caption_count=max(caption_counts),
        mean_caption_count_numerator=sum(caption_counts),
        mean_caption_count_denominator=len(caption_counts),
        total_motion_query_count=sum(motion_counts),
        minimum_motion_query_count=min(motion_counts),
        maximum_motion_query_count=max(motion_counts),
        mean_motion_query_count_numerator=sum(motion_counts),
        mean_motion_query_count_denominator=len(motion_counts),
        capture_rows_sha256=hashlib.sha256(_capture_rows_bytes(checked)).hexdigest(),
    )


def _capture_effect_fraction(row: PairedCapture) -> Fraction:
    text_effect = Fraction(
        sum(row.treatment_text_to_motion_hits) - sum(row.control_text_to_motion_hits),
        len(row.caption_ids),
    )
    motion_effect = Fraction(
        sum(row.treatment_motion_to_text_hits) - sum(row.control_motion_to_text_hits),
        len(row.motion_ids),
    )
    return (text_effect + motion_effect) / 2


def _capture_effects(captures: tuple[PairedCapture, ...]) -> np.ndarray:
    fractions = tuple(_capture_effect_fraction(row) for row in captures)
    return np.asarray([float(value) for value in fractions], dtype=np.float64)


def paired_capture_effect(captures: object) -> float:
    """Return the capture-macro treatment-minus-control effect."""

    checked = _normalize_captures(captures)
    exact = sum(
        (_capture_effect_fraction(row) for row in checked),
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


def _normalize_seed_blocks(
    value: object,
) -> tuple[tuple[int, tuple[PairedCapture, ...]], ...]:
    if type(value) is not tuple or len(value) != len(TRAINING_SEEDS):
        raise TypeError("seed blocks must be one exact tuple for each registered seed")
    normalized: list[tuple[int, tuple[PairedCapture, ...]]] = []
    reference: tuple[PairedCapture, ...] | None = None
    for ordinal, item in enumerate(value):
        if type(item) is not tuple or len(item) != 2:
            raise TypeError(f"seed blocks[{ordinal}] must be an exact (seed, captures) tuple")
        training_seed, captures = item
        if type(training_seed) is not int:
            raise TypeError(f"seed blocks[{ordinal}].seed must be an exact int")
        checked = _normalize_captures(captures)
        if reference is None:
            reference = checked
        else:
            for left, right in zip(reference, checked, strict=True):
                if (
                    left.capture_id != right.capture_id
                    or left.caption_ids != right.caption_ids
                    or left.motion_ids != right.motion_ids
                ):
                    raise StatisticalContractError(
                        "training-seed blocks do not share one capture/query census"
                    )
        normalized.append((training_seed, checked))
    if tuple(seed for seed, _ in normalized) != TRAINING_SEEDS:
        raise StatisticalContractError(
            "training-seed blocks must use the registered fixed order"
        )
    return tuple(normalized)


def _seed_block_rows_bytes(
    blocks: tuple[tuple[int, tuple[PairedCapture, ...]], ...],
) -> bytes:
    return _canonical_json_bytes(
        [
            {
                "captures": [
                    {
                        "caption_ids": list(row.caption_ids),
                        "capture_id": row.capture_id,
                        "control_motion_to_text_hits": list(
                            row.control_motion_to_text_hits
                        ),
                        "control_text_to_motion_hits": list(
                            row.control_text_to_motion_hits
                        ),
                        "motion_ids": list(row.motion_ids),
                        "treatment_motion_to_text_hits": list(
                            row.treatment_motion_to_text_hits
                        ),
                        "treatment_text_to_motion_hits": list(
                            row.treatment_text_to_motion_hits
                        ),
                    }
                    for row in captures
                ],
                "training_seed": training_seed,
            }
            for training_seed, captures in blocks
        ]
    )


def paired_seed_blocked_bootstrap(
    seed_blocks: object,
    *,
    resampling_seed: object,
) -> SeedBlockedBootstrapResult:
    """Bootstrap captures after averaging the three paired seed effects.

    Every seed is evaluated independently.  Raw logits are never combined.
    The registered seeds are fixed repeated blocks: for each capture their
    paired treatment-minus-control effects are averaged in registered order,
    and only captures are resampled.
    """

    checked = _normalize_seed_blocks(seed_blocks)
    reference = checked[0][1]
    if len(reference) < 2:
        raise StatisticalContractError(
            "seed-blocked paired bootstrap requires at least two captures"
        )
    if type(resampling_seed) is not int or not 0 <= resampling_seed <= (1 << 63) - 1:
        raise TypeError("resampling_seed must be an exact nonnegative 63-bit int")
    effects = np.asarray(
        [
            float(
                sum(
                    (_capture_effect_fraction(captures[index]) for _, captures in checked),
                    start=Fraction(0, 1),
                )
                / len(checked)
            )
            for index in range(len(reference))
        ],
        dtype=np.float64,
    )
    rng = np.random.Generator(np.random.PCG64(resampling_seed))
    samples = np.empty(BOOTSTRAP_DRAWS, dtype=np.float64)
    index_hash = hashlib.sha256()
    capture_count = len(reference)
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
    seed_block_rows_sha256 = hashlib.sha256(_seed_block_rows_bytes(checked)).hexdigest()
    return SeedBlockedBootstrapResult(
        draws=BOOTSTRAP_DRAWS,
        resampling_seed=resampling_seed,
        training_seeds=TRAINING_SEEDS,
        capture_count=capture_count,
        observed_effect=float(effects.mean()),
        confidence_lower=lower,
        confidence_upper=upper,
        two_sided_p_value=p_value,
        seed_block_rows_sha256=seed_block_rows_sha256,
        capture_census_sha256=_capture_census_sha256(reference),
        index_stream_sha256=index_hash.hexdigest(),
    )


def canonical_seed_blocked_bootstrap_result_bytes(result: object) -> bytes:
    if type(result) is not SeedBlockedBootstrapResult:
        raise TypeError("result must be exact SeedBlockedBootstrapResult")
    if (
        result.draws != BOOTSTRAP_DRAWS
        or type(result.resampling_seed) is not int
        or result.training_seeds != TRAINING_SEEDS
        or type(result.capture_count) is not int
        or result.capture_count < 2
        or result.primary_endpoint_id != PRIMARY_ENDPOINT_ID
        or result.schema != SEED_BLOCKED_BOOTSTRAP_SCHEMA
        or result.status != STATUS
    ):
        raise StatisticalContractError("seed-blocked bootstrap identity is invalid")
    for label, item in (
        ("observed_effect", result.observed_effect),
        ("confidence_lower", result.confidence_lower),
        ("confidence_upper", result.confidence_upper),
    ):
        _effect(item, label)
    _probability(result.two_sided_p_value, "two_sided_p_value")
    _lower_sha256(result.seed_block_rows_sha256, "seed_block_rows_sha256")
    _lower_sha256(result.capture_census_sha256, "capture_census_sha256")
    _lower_sha256(result.index_stream_sha256, "index_stream_sha256")
    return _canonical_json_bytes(
        {
            "capture_census_sha256": result.capture_census_sha256,
            "capture_count": result.capture_count,
            "confidence_lower_hex": result.confidence_lower.hex(),
            "confidence_upper_hex": result.confidence_upper.hex(),
            "draws": result.draws,
            "index_stream_sha256": result.index_stream_sha256,
            "observed_effect_hex": result.observed_effect.hex(),
            "primary_endpoint_id": result.primary_endpoint_id,
            "resampling_seed": result.resampling_seed,
            "schema": result.schema,
            "seed_block_rows_sha256": result.seed_block_rows_sha256,
            "status": result.status,
            "training_seeds": list(result.training_seeds),
            "two_sided_p_value_hex": result.two_sided_p_value.hex(),
        }
    )


def canonical_bootstrap_result_bytes(result: object) -> bytes:
    if type(result) is not BootstrapResult:
        raise TypeError("result must be exact BootstrapResult")
    if (
        result.draws != BOOTSTRAP_DRAWS
        or type(result.seed) is not int
        or type(result.capture_count) is not int
        or result.capture_count < 2
        or result.primary_endpoint_id != PRIMARY_ENDPOINT_ID
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
            "primary_endpoint_id": result.primary_endpoint_id,
            "schema": result.schema,
            "seed": result.seed,
            "status": result.status,
            "two_sided_p_value_hex": result.two_sided_p_value.hex(),
        }
    )


def _artifact_sha256(value: object, label: str) -> str:
    if type(value) is not bytes or not value:
        raise TypeError(f"{label} must be nonempty exact bytes")
    return hashlib.sha256(value).hexdigest()


def _direction_metrics_payload(value: object, label: str) -> dict[str, object]:
    if type(value) is not DirectionMetrics:
        raise TypeError(f"{label} must be exact DirectionMetrics")
    if (
        type(value.query_count) is not int
        or value.query_count < 1
        or type(value.aggregation_unit_count) is not int
        or value.aggregation_unit_count < 1
    ):
        raise StatisticalContractError(f"{label} counts are invalid")
    metrics: dict[str, float] = {}
    for name in (
        "recall_at_1",
        "recall_at_3",
        "recall_at_5",
        "recall_at_10",
    ):
        metrics[name] = _probability(getattr(value, name), f"{label}.{name}")
    median_rank = getattr(value, "median_rank")
    if type(median_rank) not in {int, float} or not math.isfinite(float(median_rank)):
        raise StatisticalContractError(f"{label}.median_rank must be finite")
    return {
        "aggregation_unit_count": value.aggregation_unit_count,
        "median_rank_hex": float(median_rank).hex(),
        "query_count": value.query_count,
        **{f"{name}_hex": metric.hex() for name, metric in metrics.items()},
    }


def _bidirectional_metrics_payload(value: object, label: str) -> dict[str, object]:
    if type(value) is not BidirectionalMetrics:
        raise TypeError(f"{label} must be exact BidirectionalMetrics")
    primary = _probability(value.primary, f"{label}.primary")
    return {
        "motion_to_text": _direction_metrics_payload(
            value.motion_to_text,
            f"{label}.motion_to_text",
        ),
        "primary_hex": primary.hex(),
        "text_to_motion": _direction_metrics_payload(
            value.text_to_motion,
            f"{label}.text_to_motion",
        ),
    }


def _evaluation_report_bytes(value: object, label: str) -> bytes:
    """Canonicalize one actual evaluator output only for an internal digest."""

    if type(value) is not EvaluationReport:
        raise TypeError(f"{label} must be exact EvaluationReport")
    _lower_sha256(value.dataset_sha256, f"{label}.dataset_sha256")
    if value.evaluation_type != "REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL":
        raise StatisticalContractError(
            f"{label} is not the preregistered real human-holistic endpoint"
        )
    _lower_sha256(
        value.census_provenance_sha256,
        f"{label}.census_provenance_sha256",
    )
    _lower_sha256(
        value.score_provenance_sha256,
        f"{label}.score_provenance_sha256",
    )
    if type(value.by_group_size) is not tuple or not value.by_group_size:
        raise StatisticalContractError(f"{label}.by_group_size must be nonempty")
    by_group_size: list[dict[str, object]] = []
    previous_group_size = 1
    for ordinal, row in enumerate(value.by_group_size):
        if type(row) is not tuple or len(row) != 2:
            raise StatisticalContractError(f"{label}.by_group_size row is malformed")
        group_size, metrics = row
        if type(group_size) is not int or group_size <= previous_group_size:
            raise StatisticalContractError(
                f"{label}.by_group_size must be strictly ordered K>=2"
            )
        previous_group_size = group_size
        by_group_size.append(
            {
                "group_size": group_size,
                "metrics": _bidirectional_metrics_payload(
                    metrics,
                    f"{label}.by_group_size[{ordinal}]",
                ),
            }
        )
    if (
        type(value.leave_one_component_out) is not tuple
        or len(value.leave_one_component_out) < 2
    ):
        raise StatisticalContractError(
            f"{label}.leave_one_component_out requires at least two components"
        )
    component_rows: list[dict[str, object]] = []
    labels: list[str] = []
    for ordinal, row in enumerate(value.leave_one_component_out):
        if type(row) is not tuple or len(row) != 2:
            raise StatisticalContractError(f"{label}.component row is malformed")
        component, metrics = row
        if type(component) is not str or not component:
            raise StatisticalContractError(f"{label}.component label is invalid")
        labels.append(component)
        component_rows.append(
            {
                "component": component,
                "metrics": _bidirectional_metrics_payload(
                    metrics,
                    f"{label}.component[{ordinal}]",
                ),
            }
        )
    if labels != sorted(set(labels)):
        raise StatisticalContractError(f"{label}.component labels are not canonical")
    contributions = _contribution_map(value.capture_contributions, f"{label}.contributions")
    contribution_rows = [
        {
            "caption_commitments": [item.hex() for item in row.caption_commitments],
            "capture_commitment": capture.hex(),
            "motion_commitments": [item.hex() for item in row.motion_commitments],
            "motion_to_text_hits": list(row.motion_to_text_hits),
            "text_to_motion_hits": list(row.text_to_motion_hits),
        }
        for capture, row in sorted(contributions.items())
    ]
    return _canonical_json_bytes(
        {
            "by_group_size": by_group_size,
            "capture_contributions": contribution_rows,
            "census_provenance_sha256": value.census_provenance_sha256,
            "dataset_sha256": value.dataset_sha256,
            "evaluation_type": value.evaluation_type,
            "full_gallery": _bidirectional_metrics_payload(
                value.full_gallery,
                f"{label}.full_gallery",
            ),
            "leave_one_component_out": component_rows,
            "macro_group_size": _bidirectional_metrics_payload(
                value.macro_group_size,
                f"{label}.macro_group_size",
            ),
            "schema": "phaseset-evaluation-report-digest-v2",
            "score_provenance_sha256": value.score_provenance_sha256,
        }
    )


def _capture_census_sha256(captures: tuple[PairedCapture, ...]) -> str:
    return hashlib.sha256(
        _canonical_json_bytes(
            [
                {
                    "caption_ids": list(row.caption_ids),
                    "capture_id": row.capture_id,
                    "motion_ids": list(row.motion_ids),
                }
                for row in captures
            ]
        )
    ).hexdigest()


def _comparison_identity(value: HypothesisComparison) -> tuple[str, str]:
    for name in ("hypothesis_id", "treatment_system_id", "baseline_system_id"):
        if type(getattr(value, name)) is not str:
            raise TypeError(f"comparison.{name} must be exact built-in str")
    registered = {
        hypothesis_id: (treatment_system_id, baseline_system_id)
        for hypothesis_id, treatment_system_id, baseline_system_id in HYPOTHESIS_COMPARISONS
    }
    if value.hypothesis_id not in registered:
        raise StatisticalContractError("comparison hypothesis_id must be H1-H8")
    expected = registered[value.hypothesis_id]
    actual = (value.treatment_system_id, value.baseline_system_id)
    if actual != expected:
        raise StatisticalContractError(
            f"{value.hypothesis_id} treatment/baseline identity differs from the registry"
        )
    return expected


def _build_provenanced_hypothesis(
    value: object,
) -> ProvenancedHypothesisResult:
    if type(value) is not HypothesisComparison:
        raise TypeError("comparison must be exact HypothesisComparison")
    _comparison_identity(value)
    if value.training_seeds != TRAINING_SEEDS:
        raise StatisticalContractError(
            f"{value.hypothesis_id} training seeds differ from the registry"
        )
    evaluation_pairs = (
        ("treatment", value.treatment_evaluations),
        ("baseline", value.baseline_evaluations),
    )
    checked: dict[str, tuple[EvaluationReport, ...]] = {}
    evaluation_raws: dict[str, tuple[bytes, ...]] = {}
    for side, reports in evaluation_pairs:
        if type(reports) is not tuple or len(reports) != len(TRAINING_SEEDS):
            raise TypeError(
                f"{value.hypothesis_id}.{side}_evaluations must contain three reports"
            )
        if any(type(report) is not EvaluationReport for report in reports):
            raise TypeError(
                f"{value.hypothesis_id}.{side}_evaluations must be exact reports"
            )
        checked[side] = reports
        evaluation_raws[side] = tuple(
            _evaluation_report_bytes(
                report,
                f"{value.hypothesis_id}.{side}_evaluations[{ordinal}]",
            )
            for ordinal, report in enumerate(reports)
        )
        if len({report.score_provenance_sha256 for report in reports}) != len(
            TRAINING_SEEDS
        ):
            raise StatisticalContractError(
                f"{value.hypothesis_id} {side} seed score provenance is not unique"
            )

    treatment_reports = checked["treatment"]
    baseline_reports = checked["baseline"]
    corpus_digests = {
        report.census_provenance_sha256
        for report in treatment_reports + baseline_reports
    }
    if len(corpus_digests) != 1:
        raise StatisticalContractError(
            f"{value.hypothesis_id} treatment/baseline seed corpus provenance differs"
        )
    component_censuses = {
        tuple(row[0] for row in report.leave_one_component_out)
        for report in treatment_reports + baseline_reports
    }
    if len(component_censuses) != 1:
        raise StatisticalContractError(
            f"{value.hypothesis_id} treatment/baseline seed component census differs"
        )
    treatment_components = next(iter(component_censuses))
    seed_blocks = tuple(
        (
            training_seed,
            paired_captures_from_contributions(
                treatment.capture_contributions,
                baseline.capture_contributions,
            ),
        )
        for training_seed, treatment, baseline in zip(
            TRAINING_SEEDS,
            treatment_reports,
            baseline_reports,
            strict=True,
        )
    )
    bootstrap = paired_seed_blocked_bootstrap(
        seed_blocks,
        resampling_seed=value.resampling_seed,
    )
    summary = summarize_variable_captions(seed_blocks[0][1])
    if summary.capture_count != bootstrap.capture_count:
        raise AssertionError("seed-block bootstrap and capture summary diverged")
    bootstrap_raw = canonical_seed_blocked_bootstrap_result_bytes(bootstrap)
    treatment_raws = evaluation_raws["treatment"]
    baseline_raws = evaluation_raws["baseline"]
    return ProvenancedHypothesisResult(
        hypothesis_id=value.hypothesis_id,
        treatment_system_id=value.treatment_system_id,
        baseline_system_id=value.baseline_system_id,
        primary_endpoint_id=bootstrap.primary_endpoint_id,
        direction=DIRECTION_ID,
        effect=bootstrap.observed_effect,
        confidence_lower=bootstrap.confidence_lower,
        confidence_upper=bootstrap.confidence_upper,
        raw_p_value=bootstrap.two_sided_p_value,
        resampling_seed=bootstrap.resampling_seed,
        training_seeds=bootstrap.training_seeds,
        seed_count=len(bootstrap.training_seeds),
        draws=bootstrap.draws,
        capture_count=bootstrap.capture_count,
        caption_count=summary.total_caption_count,
        motion_count=summary.total_motion_query_count,
        component_count=len(treatment_components),
        evaluation_type=treatment_reports[0].evaluation_type,
        census_provenance_sha256=treatment_reports[0].census_provenance_sha256,
        treatment_score_provenance_sha256s=tuple(
            report.score_provenance_sha256 for report in treatment_reports
        ),
        baseline_score_provenance_sha256s=tuple(
            report.score_provenance_sha256 for report in baseline_reports
        ),
        aggregate_sha256=_artifact_sha256(
            value.aggregate_artifact,
            f"{value.hypothesis_id}.aggregate_artifact",
        ),
        evaluator_sha256=_artifact_sha256(
            value.evaluator_artifact,
            f"{value.hypothesis_id}.evaluator_artifact",
        ),
        seed_block_rows_sha256=bootstrap.seed_block_rows_sha256,
        capture_census_sha256=bootstrap.capture_census_sha256,
        index_stream_sha256=bootstrap.index_stream_sha256,
        bootstrap_sha256=hashlib.sha256(bootstrap_raw).hexdigest(),
        treatment_evaluation_sha256s=tuple(
            hashlib.sha256(raw).hexdigest() for raw in treatment_raws
        ),
        baseline_evaluation_sha256s=tuple(
            hashlib.sha256(raw).hexdigest() for raw in baseline_raws
        ),
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
    row_types = {type(row) for row in hypotheses}
    if row_types == {HypothesisResult}:
        # Compatibility for the tiny synthetic executor fixture.  This path is
        # deliberately incapable of producing REPORT_SCHEMA or a result claim.
        synthetic_by_id: dict[str, HypothesisResult] = {}
        for value in hypotheses:
            if (
                value.hypothesis_id not in HYPOTHESIS_ORDER
                or value.hypothesis_id in synthetic_by_id
            ):
                raise StatisticalContractError(
                    "hypothesis identities must be unique H1-H8"
                )
            normalized = HypothesisResult(
                hypothesis_id=value.hypothesis_id,
                effect=_effect(value.effect, f"{value.hypothesis_id}.effect"),
                raw_p_value=_probability(
                    value.raw_p_value,
                    f"{value.hypothesis_id}.raw_p",
                ),
                bootstrap_sha256=_lower_sha256(
                    value.bootstrap_sha256,
                    f"{value.hypothesis_id}.bootstrap_sha256",
                ),
            )
            synthetic_by_id[normalized.hypothesis_id] = normalized
        if set(synthetic_by_id) != set(HYPOTHESIS_ORDER):
            raise StatisticalContractError("hypothesis census must be exactly H1-H8")
        synthetic_rows = tuple(
            synthetic_by_id[hypothesis_id] for hypothesis_id in HYPOTHESIS_ORDER
        )
        return StatisticalReport(
            hypotheses=synthetic_rows,
            holm_adjustments=holm_h2_h8(
                {
                    row.hypothesis_id: row.raw_p_value
                    for row in synthetic_rows
                    if row.hypothesis_id in HOLM_FAMILY
                }
            ),
            provenance_complete=False,
            schema=SYNTHETIC_REPORT_SCHEMA,
            status=SYNTHETIC_STATUS,
        )
    if row_types != {HypothesisComparison}:
        raise TypeError(
            "hypotheses must contain only exact HypothesisComparison rows "
            "(or only legacy synthetic HypothesisResult rows)"
        )

    comparison_by_id: dict[str, HypothesisComparison] = {}
    for value in hypotheses:
        _comparison_identity(value)
        if value.hypothesis_id in comparison_by_id:
            raise StatisticalContractError("comparison identities must be unique H1-H8")
        comparison_by_id[value.hypothesis_id] = value
    if set(comparison_by_id) != set(HYPOTHESIS_ORDER):
        raise StatisticalContractError("comparison census must be exactly H1-H8")
    ordered_comparisons = tuple(
        comparison_by_id[hypothesis_id] for hypothesis_id in HYPOTHESIS_ORDER
    )
    formal_rows = tuple(
        _build_provenanced_hypothesis(value) for value in ordered_comparisons
    )

    # H1--H8 share one frozen draw stream and evidence census.  Each registered
    # training seed is ranked independently; only its paired capture hits enter
    # the fixed-seed block mean.  Raw scores are never averaged across seeds.
    shared_fields = (
        "primary_endpoint_id",
        "direction",
        "resampling_seed",
        "training_seeds",
        "seed_count",
        "draws",
        "capture_count",
        "caption_count",
        "motion_count",
        "component_count",
        "evaluation_type",
        "census_provenance_sha256",
        "aggregate_sha256",
        "evaluator_sha256",
        "capture_census_sha256",
        "index_stream_sha256",
        "treatment_evaluation_sha256s",
        "treatment_score_provenance_sha256s",
    )
    shared_family = formal_rows
    for name in shared_fields:
        if len({getattr(row, name) for row in shared_family}) != 1:
            raise StatisticalContractError(
                f"H1-H8 must share frozen resampling provenance: {name} differs"
            )
    family = tuple(row for row in formal_rows if row.hypothesis_id in HOLM_FAMILY)
    holm = holm_h2_h8({row.hypothesis_id: row.raw_p_value for row in family})
    return StatisticalReport(
        hypotheses=formal_rows,
        holm_adjustments=holm,
        provenance_complete=True,
        schema=REPORT_SCHEMA,
        authority=0,
        production=False,
        result_claimed=False,
        status=STATUS,
        _comparisons=ordered_comparisons,
    )


def canonical_statistical_report_bytes(report: object) -> bytes:
    if type(report) is not StatisticalReport:
        raise TypeError("report must be exact StatisticalReport")
    if report.provenance_complete:
        if type(report._comparisons) is not tuple:
            raise StatisticalContractError(
                "formal statistical report is missing private comparison evidence"
            )
        rebuilt = build_statistical_report(report._comparisons)
    else:
        if report._comparisons is not None:
            raise StatisticalContractError(
                "synthetic statistical report cannot retain formal comparison evidence"
            )
        rebuilt = build_statistical_report(report.hypotheses)
    if report != rebuilt:
        raise StatisticalContractError(
            "statistical report differs from its canonical provenance reconstruction"
        )
    comparison_by_id = {
        hypothesis_id: (treatment_system_id, baseline_system_id)
        for hypothesis_id, treatment_system_id, baseline_system_id in HYPOTHESIS_COMPARISONS
    }
    if report.provenance_complete:
        hypothesis_payload = [
            {
                "aggregate_sha256": row.aggregate_sha256,
                "baseline_evaluation_sha256s": list(
                    row.baseline_evaluation_sha256s
                ),
                "baseline_score_provenance_sha256s": list(
                    row.baseline_score_provenance_sha256s
                ),
                "baseline_system_id": row.baseline_system_id,
                "bootstrap_sha256": row.bootstrap_sha256,
                "caption_count": row.caption_count,
                "capture_census_sha256": row.capture_census_sha256,
                "capture_count": row.capture_count,
                "seed_block_rows_sha256": row.seed_block_rows_sha256,
                "component_count": row.component_count,
                "census_provenance_sha256": row.census_provenance_sha256,
                "confidence_lower_hex": row.confidence_lower.hex(),
                "confidence_upper_hex": row.confidence_upper.hex(),
                "direction": row.direction,
                "draws": row.draws,
                "effect_hex": row.effect.hex(),
                "evaluator_sha256": row.evaluator_sha256,
                "evaluation_type": row.evaluation_type,
                "hypothesis_id": row.hypothesis_id,
                "index_stream_sha256": row.index_stream_sha256,
                "motion_count": row.motion_count,
                "primary_endpoint_id": row.primary_endpoint_id,
                "raw_p_value_hex": row.raw_p_value.hex(),
                "resampling_seed": row.resampling_seed,
                "seed_count": row.seed_count,
                "training_seeds": list(row.training_seeds),
                "treatment_evaluation_sha256s": list(
                    row.treatment_evaluation_sha256s
                ),
                "treatment_score_provenance_sha256s": list(
                    row.treatment_score_provenance_sha256s
                ),
                "treatment_system_id": row.treatment_system_id,
            }
            for row in report.hypotheses
            if type(row) is ProvenancedHypothesisResult
        ]
        if len(hypothesis_payload) != len(HYPOTHESIS_ORDER):
            raise StatisticalContractError("formal report contains a non-provenanced row")
    else:
        hypothesis_payload = [
            {
                "baseline_system_id": comparison_by_id[row.hypothesis_id][1],
                "bootstrap_sha256": row.bootstrap_sha256,
                "effect_hex": row.effect.hex(),
                "hypothesis_id": row.hypothesis_id,
                "raw_p_value_hex": row.raw_p_value.hex(),
                "treatment_system_id": comparison_by_id[row.hypothesis_id][0],
            }
            for row in report.hypotheses
            if type(row) is HypothesisResult
        ]
        if len(hypothesis_payload) != len(HYPOTHESIS_ORDER):
            raise StatisticalContractError("synthetic report contains a formal row")
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
            "hypotheses": hypothesis_payload,
            "production": report.production,
            "provenance_complete": report.provenance_complete,
            "result_claimed": report.result_claimed,
            "schema": report.schema,
            "shared_resampling_family": list(HYPOTHESIS_ORDER),
            "status": report.status,
        }
    )


__all__ = [
    "BOOTSTRAP_DRAWS",
    "BOOTSTRAP_SCHEMA",
    "BootstrapResult",
    "DIRECTION_ID",
    "HOLM_FAMILY",
    "HYPOTHESIS_COMPARISONS",
    "HYPOTHESIS_ORDER",
    "HolmAdjustment",
    "HypothesisComparison",
    "HypothesisResult",
    "PairedCapture",
    "PRIMARY_ENDPOINT_ID",
    "ProvenancedHypothesisResult",
    "REPORT_SCHEMA",
    "SEED_BLOCKED_BOOTSTRAP_SCHEMA",
    "SeedBlockedBootstrapResult",
    "STATUS",
    "SYNTHETIC_REPORT_SCHEMA",
    "SYNTHETIC_STATUS",
    "StatisticalContractError",
    "StatisticalReport",
    "VariableCaptionSummary",
    "TRAINING_SEEDS",
    "build_statistical_report",
    "canonical_bootstrap_result_bytes",
    "canonical_seed_blocked_bootstrap_result_bytes",
    "canonical_statistical_report_bytes",
    "holm_h2_h8",
    "paired_capture_bootstrap",
    "paired_captures_from_contributions",
    "paired_capture_effect",
    "paired_seed_blocked_bootstrap",
    "summarize_variable_captions",
]
