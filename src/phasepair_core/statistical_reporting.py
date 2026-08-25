"""Authority-zero collection and export of frozen PhasePair statistics.

This module fixes the current contract's 21 validation-row order and the
H1--H6 comparison identities.  It accepts already-frozen statistics from a
caller, validates and binds their identities, and applies Holm only to the
five secondary p-values.  It does not evaluate a gallery, run a bootstrap,
read project data, authorize execution, or claim a result.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import io
import json
import math
from types import MappingProxyType


AUTHORITY = 0
PRODUCTION = False
EXECUTION_AUTHORIZED = False
RESULT_CLAIMED = False
STATUS = "DATA_FREE_FROZEN_STATISTICS_AUTHORITY0_NO_RESULT"

SEED_ORDER = (1729, 2718, 31415)
SYSTEM_ORDER = ("00", "01", "02", "03", "04", "05", "06")
SYSTEM_NAMES = MappingProxyType(
    {
        "00": "mime",
        "01": "generic",
        "02": "wamo_marginal_wavelet",
        "03": "interedit_mean_difference_dct",
        "04": "no_relation",
        "05": "phase_stripped",
        "06": "phasepair_full",
    }
)
HYPOTHESIS_ORDER = ("H1", "H2", "H3", "H4", "H5", "H6")
HOLM_HYPOTHESIS_ORDER = ("H2", "H3", "H4", "H5", "H6")
HYPOTHESIS_COMPARISONS = MappingProxyType(
    {
        "H1": ("06", "00"),
        "H2": ("06", "01"),
        "H3": ("06", "02"),
        "H4": ("06", "03"),
        "H5": ("06", "04"),
        "H6": ("06", "05"),
    }
)

_EXPECTED_VALIDATION_IDENTITIES = tuple(
    (seed, system_id) for seed in SEED_ORDER for system_id in SYSTEM_ORDER
)


class StatisticalReportingError(ValueError):
    """Raised when frozen statistics do not satisfy the reporting contract."""


@dataclass(frozen=True)
class FrozenValidationResult:
    """Caller-provided full-gallery statistics for one frozen score row."""

    seed: int
    system_id: str
    run_id: str
    t2m_cluster_macro_r_at_1: float
    m2t_any_caption_r_at_1: float
    bidirectional_mean_r_at_1: float
    score_matrix_sha256: str
    metric_sha256: str


@dataclass(frozen=True)
class FrozenHypothesisResult:
    """Caller-provided effect and raw p-value for one fixed comparison."""

    hypothesis_id: str
    treatment_system_id: str
    baseline_system_id: str
    effect: float
    raw_p_value: float
    statistic_sha256: str


@dataclass(frozen=True)
class HolmAdjustment:
    """One secondary p-value after canonical Holm step-down adjustment."""

    hypothesis_id: str
    raw_p_value: float
    sorted_rank: int
    adjusted_p_value: float


@dataclass(frozen=True)
class ReportedHypothesisResult:
    """A frozen hypothesis row annotated with its secondary-family result."""

    hypothesis_id: str
    treatment_system_id: str
    baseline_system_id: str
    effect: float
    raw_p_value: float
    statistic_sha256: str
    holm_rank: int | None
    holm_adjusted_p_value: float | None


@dataclass(frozen=True)
class AuthorityZeroStatisticalReport:
    """Validated, immutable Authority0 view over caller-frozen statistics."""

    validation_rows: tuple[FrozenValidationResult, ...]
    hypothesis_rows: tuple[ReportedHypothesisResult, ...]


def _exact_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    return value


def _lower_sha256(value: object, label: str) -> str:
    checked = _exact_text(value, label)
    if len(checked) != 64 or any(
        character not in "0123456789abcdef" for character in checked
    ):
        raise StatisticalReportingError(f"{label} must be lowercase SHA-256 hex")
    return checked


def _finite_number(
    value: object,
    label: str,
    *,
    minimum: float,
    maximum: float,
) -> float:
    if type(value) not in {int, float}:
        raise TypeError(f"{label} must be an exact built-in int or float, not bool")
    normalized = float(value)
    if not math.isfinite(normalized):
        raise StatisticalReportingError(f"{label} must be finite")
    if normalized < minimum or normalized > maximum:
        raise StatisticalReportingError(
            f"{label} must be inside the closed interval [{minimum},{maximum}]"
        )
    # Collapse signed zero so each accepted numeric value has one export form.
    return 0.0 if normalized == 0.0 else normalized


def _probability(value: object, label: str) -> float:
    return _finite_number(value, label, minimum=0.0, maximum=1.0)


def _expected_run_id(seed: int, system_id: str) -> str:
    role = "BASE_TRAIN" if system_id == "00" else "RESIDUAL_HEAD_TRAIN"
    return f"phasepair-run-v2/{role}/{seed}/{system_id}"


def _normalize_validation_row(
    value: object,
    ordinal: int,
) -> FrozenValidationResult:
    label = f"validation_rows[{ordinal}]"
    if type(value) is not FrozenValidationResult:
        raise TypeError(f"{label} must be exact FrozenValidationResult")
    if type(value.seed) is not int:
        raise TypeError(f"{label}.seed must be an exact built-in int")
    if value.seed not in SEED_ORDER:
        raise StatisticalReportingError(
            f"{label}.seed is outside the frozen seed census"
        )
    system_id = _exact_text(value.system_id, f"{label}.system_id")
    if system_id not in SYSTEM_ORDER:
        raise StatisticalReportingError(
            f"{label}.system_id is outside the frozen system census"
        )
    run_id = _exact_text(value.run_id, f"{label}.run_id")
    expected_run_id = _expected_run_id(value.seed, system_id)
    if run_id != expected_run_id:
        raise StatisticalReportingError(
            f"{label}.run_id does not bind its exact seed/system identity"
        )

    t2m = _probability(
        value.t2m_cluster_macro_r_at_1,
        f"{label}.t2m_cluster_macro_r_at_1",
    )
    m2t = _probability(
        value.m2t_any_caption_r_at_1,
        f"{label}.m2t_any_caption_r_at_1",
    )
    bidirectional = _probability(
        value.bidirectional_mean_r_at_1,
        f"{label}.bidirectional_mean_r_at_1",
    )
    if bidirectional != 0.5 * (t2m + m2t):
        raise StatisticalReportingError(
            f"{label}.bidirectional_mean_r_at_1 is not the exact full-gallery mean"
        )

    return FrozenValidationResult(
        seed=value.seed,
        system_id=system_id,
        run_id=run_id,
        t2m_cluster_macro_r_at_1=t2m,
        m2t_any_caption_r_at_1=m2t,
        bidirectional_mean_r_at_1=bidirectional,
        score_matrix_sha256=_lower_sha256(
            value.score_matrix_sha256,
            f"{label}.score_matrix_sha256",
        ),
        metric_sha256=_lower_sha256(value.metric_sha256, f"{label}.metric_sha256"),
    )


def collect_frozen_validation_results(
    rows: object,
) -> tuple[FrozenValidationResult, ...]:
    """Validate exactly 21 identities and return seed-outer/system-inner rows.

    Input order is deliberately irrelevant.  Duplicate, missing, aliased, and
    out-of-census identities fail closed before a canonical tuple is returned.
    """

    if type(rows) is not tuple:
        raise TypeError("validation_rows must be an exact built-in tuple")
    by_identity: dict[tuple[int, str], FrozenValidationResult] = {}
    for ordinal, value in enumerate(rows):
        row = _normalize_validation_row(value, ordinal)
        identity = (row.seed, row.system_id)
        if identity in by_identity:
            raise StatisticalReportingError(
                f"duplicate validation seed/system identity: {identity!r}"
            )
        by_identity[identity] = row

    missing = [
        identity
        for identity in _EXPECTED_VALIDATION_IDENTITIES
        if identity not in by_identity
    ]
    if missing:
        raise StatisticalReportingError(
            f"missing validation seed/system identities: {missing!r}"
        )
    if len(by_identity) != len(_EXPECTED_VALIDATION_IDENTITIES):
        raise StatisticalReportingError(
            "validation result census must contain exactly 21 rows"
        )
    return tuple(by_identity[identity] for identity in _EXPECTED_VALIDATION_IDENTITIES)


def _normalize_hypothesis_row(
    value: object,
    ordinal: int,
) -> FrozenHypothesisResult:
    label = f"hypothesis_rows[{ordinal}]"
    if type(value) is not FrozenHypothesisResult:
        raise TypeError(f"{label} must be exact FrozenHypothesisResult")
    hypothesis_id = _exact_text(value.hypothesis_id, f"{label}.hypothesis_id")
    if hypothesis_id not in HYPOTHESIS_ORDER:
        raise StatisticalReportingError(
            f"{label}.hypothesis_id is outside the frozen H1-H6 census"
        )
    treatment = _exact_text(
        value.treatment_system_id,
        f"{label}.treatment_system_id",
    )
    baseline = _exact_text(value.baseline_system_id, f"{label}.baseline_system_id")
    if (treatment, baseline) != HYPOTHESIS_COMPARISONS[hypothesis_id]:
        raise StatisticalReportingError(
            f"{label} does not use the fixed {hypothesis_id} system comparison"
        )
    return FrozenHypothesisResult(
        hypothesis_id=hypothesis_id,
        treatment_system_id=treatment,
        baseline_system_id=baseline,
        effect=_finite_number(
            value.effect, f"{label}.effect", minimum=-1.0, maximum=1.0
        ),
        raw_p_value=_probability(value.raw_p_value, f"{label}.raw_p_value"),
        statistic_sha256=_lower_sha256(
            value.statistic_sha256,
            f"{label}.statistic_sha256",
        ),
    )


def collect_frozen_hypothesis_results(
    rows: object,
) -> tuple[FrozenHypothesisResult, ...]:
    """Validate and restore the exact H1--H6 order without deriving statistics."""

    if type(rows) is not tuple:
        raise TypeError("hypothesis_rows must be an exact built-in tuple")
    by_id: dict[str, FrozenHypothesisResult] = {}
    for ordinal, value in enumerate(rows):
        row = _normalize_hypothesis_row(value, ordinal)
        if row.hypothesis_id in by_id:
            raise StatisticalReportingError(
                f"duplicate hypothesis identity: {row.hypothesis_id}"
            )
        by_id[row.hypothesis_id] = row
    missing = [
        hypothesis_id
        for hypothesis_id in HYPOTHESIS_ORDER
        if hypothesis_id not in by_id
    ]
    if missing:
        raise StatisticalReportingError(f"missing hypothesis identities: {missing!r}")
    if len(by_id) != len(HYPOTHESIS_ORDER):
        raise StatisticalReportingError("hypothesis census must contain exactly H1-H6")
    return tuple(by_id[hypothesis_id] for hypothesis_id in HYPOTHESIS_ORDER)


def holm_step_down_secondary(
    p_values: object,
) -> tuple[HolmAdjustment, ...]:
    """Adjust exactly H2--H6 and return them in original hypothesis order.

    Sorting is by ``(raw p, hypothesis_id)``.  Rank multipliers are ``5..1``;
    the running maximum makes adjusted values monotone in sorted order, and
    every value is capped at one before the original H2--H6 order is restored.
    """

    if type(p_values) is not dict:
        raise TypeError("p_values must be an exact built-in dict")
    if any(type(key) is not str for key in p_values):
        raise TypeError("p_values keys must be exact built-in str values")
    if set(p_values) != set(HOLM_HYPOTHESIS_ORDER):
        missing = sorted(set(HOLM_HYPOTHESIS_ORDER) - set(p_values))
        extra = sorted(set(p_values) - set(HOLM_HYPOTHESIS_ORDER))
        raise StatisticalReportingError(
            f"Holm family must be exactly H2-H6; missing={missing!r}, extra={extra!r}"
        )

    normalized = {
        hypothesis_id: _probability(
            p_values[hypothesis_id], f"p_values[{hypothesis_id!r}]"
        )
        for hypothesis_id in HOLM_HYPOTHESIS_ORDER
    }
    ordered = sorted(normalized.items(), key=lambda item: (item[1], item[0]))
    by_id: dict[str, HolmAdjustment] = {}
    running_max = 0.0
    family_size = len(HOLM_HYPOTHESIS_ORDER)
    for rank, (hypothesis_id, raw_p_value) in enumerate(ordered, start=1):
        candidate = min(1.0, (family_size - rank + 1) * raw_p_value)
        adjusted = max(running_max, candidate)
        running_max = adjusted
        by_id[hypothesis_id] = HolmAdjustment(
            hypothesis_id=hypothesis_id,
            raw_p_value=raw_p_value,
            sorted_rank=rank,
            adjusted_p_value=adjusted,
        )
    return tuple(by_id[hypothesis_id] for hypothesis_id in HOLM_HYPOTHESIS_ORDER)


def build_authority0_statistical_report(
    validation_rows: object,
    hypothesis_rows: object,
) -> AuthorityZeroStatisticalReport:
    """Bind caller-frozen rows and add only the pre-registered Holm annotation."""

    validation = collect_frozen_validation_results(validation_rows)
    frozen_hypotheses = collect_frozen_hypothesis_results(hypothesis_rows)
    holm = holm_step_down_secondary(
        {
            row.hypothesis_id: row.raw_p_value
            for row in frozen_hypotheses
            if row.hypothesis_id in HOLM_HYPOTHESIS_ORDER
        }
    )
    holm_by_id = {row.hypothesis_id: row for row in holm}
    reported = tuple(
        ReportedHypothesisResult(
            hypothesis_id=row.hypothesis_id,
            treatment_system_id=row.treatment_system_id,
            baseline_system_id=row.baseline_system_id,
            effect=row.effect,
            raw_p_value=row.raw_p_value,
            statistic_sha256=row.statistic_sha256,
            holm_rank=(
                holm_by_id[row.hypothesis_id].sorted_rank
                if row.hypothesis_id in holm_by_id
                else None
            ),
            holm_adjusted_p_value=(
                holm_by_id[row.hypothesis_id].adjusted_p_value
                if row.hypothesis_id in holm_by_id
                else None
            ),
        )
        for row in frozen_hypotheses
    )
    return AuthorityZeroStatisticalReport(
        validation_rows=validation, hypothesis_rows=reported
    )


def _validated_report(value: object) -> AuthorityZeroStatisticalReport:
    if type(value) is not AuthorityZeroStatisticalReport:
        raise TypeError("report must be exact AuthorityZeroStatisticalReport")
    if type(value.hypothesis_rows) is not tuple:
        raise TypeError("report.hypothesis_rows must be an exact built-in tuple")
    frozen_hypotheses: list[FrozenHypothesisResult] = []
    for ordinal, row in enumerate(value.hypothesis_rows):
        if type(row) is not ReportedHypothesisResult:
            raise TypeError(
                f"report.hypothesis_rows[{ordinal}] must be exact ReportedHypothesisResult"
            )
        frozen_hypotheses.append(
            FrozenHypothesisResult(
                hypothesis_id=row.hypothesis_id,
                treatment_system_id=row.treatment_system_id,
                baseline_system_id=row.baseline_system_id,
                effect=row.effect,
                raw_p_value=row.raw_p_value,
                statistic_sha256=row.statistic_sha256,
            )
        )
    rebuilt = build_authority0_statistical_report(
        value.validation_rows,
        tuple(frozen_hypotheses),
    )
    if rebuilt != value:
        raise StatisticalReportingError(
            "report identity or Holm annotation changed after validation"
        )
    return rebuilt


def _float_hex(value: float) -> str:
    return value.hex()


def _validation_payload(row: FrozenValidationResult) -> dict[str, object]:
    return {
        "bidirectional_mean_r_at_1_hex": _float_hex(row.bidirectional_mean_r_at_1),
        "metric_sha256": row.metric_sha256,
        "m2t_any_caption_r_at_1_hex": _float_hex(row.m2t_any_caption_r_at_1),
        "run_id": row.run_id,
        "score_matrix_sha256": row.score_matrix_sha256,
        "seed": row.seed,
        "system_id": row.system_id,
        "system_name": SYSTEM_NAMES[row.system_id],
        "t2m_cluster_macro_r_at_1_hex": _float_hex(row.t2m_cluster_macro_r_at_1),
    }


def _hypothesis_payload(row: ReportedHypothesisResult) -> dict[str, object]:
    in_holm_family = row.hypothesis_id in HOLM_HYPOTHESIS_ORDER
    return {
        "baseline_system_id": row.baseline_system_id,
        "comparison": f"{row.treatment_system_id}-minus-{row.baseline_system_id}",
        "effect_hex": _float_hex(row.effect),
        "holm_adjusted_p_value_hex": (
            _float_hex(row.holm_adjusted_p_value)
            if row.holm_adjusted_p_value is not None
            else "NOT_APPLICABLE"
        ),
        "holm_family_member": in_holm_family,
        "holm_rank": row.holm_rank if row.holm_rank is not None else "NOT_APPLICABLE",
        "hypothesis_id": row.hypothesis_id,
        "raw_p_value_hex": _float_hex(row.raw_p_value),
        "statistic_sha256": row.statistic_sha256,
        "treatment_system_id": row.treatment_system_id,
    }


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def canonical_statistical_report_json_bytes(
    report: object,
) -> bytes:
    """Export one sorted-key UTF-8 JSON object with exactly one terminal LF."""

    checked = _validated_report(report)
    validation_rows = [_validation_payload(row) for row in checked.validation_rows]
    hypothesis_rows = [_hypothesis_payload(row) for row in checked.hypothesis_rows]
    validation_rows_sha256 = hashlib.sha256(
        _canonical_json_bytes(validation_rows)
    ).hexdigest()
    hypothesis_rows_sha256 = hashlib.sha256(
        _canonical_json_bytes(hypothesis_rows)
    ).hexdigest()
    return _canonical_json_bytes(
        {
            "authority": AUTHORITY,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "holm_family": list(HOLM_HYPOTHESIS_ORDER),
            "hypothesis_order": list(HYPOTHESIS_ORDER),
            "hypothesis_row_count": len(hypothesis_rows),
            "hypothesis_rows": hypothesis_rows,
            "hypothesis_rows_sha256": hypothesis_rows_sha256,
            "production": PRODUCTION,
            "result_claimed": RESULT_CLAIMED,
            "row_order": "seed_outer_system_inner",
            "schema": "phasepair-frozen-statistical-report-v1",
            "seed_order": list(SEED_ORDER),
            "status": STATUS,
            "system_order": list(SYSTEM_ORDER),
            "validation_row_count": len(validation_rows),
            "validation_rows": validation_rows,
            "validation_rows_sha256": validation_rows_sha256,
        }
    )


def _csv_bytes(header: list[str], rows: list[list[object]]) -> bytes:
    output = io.StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(header)
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def validation_table_csv_bytes(report: object) -> bytes:
    """Export the canonical 21-row frozen full-gallery table as UTF-8 CSV."""

    checked = _validated_report(report)
    header = [
        "authority",
        "production",
        "execution_authorized",
        "result_claimed",
        "status",
        "seed",
        "system_id",
        "system_name",
        "run_id",
        "t2m_cluster_macro_r_at_1_hex",
        "m2t_any_caption_r_at_1_hex",
        "bidirectional_mean_r_at_1_hex",
        "score_matrix_sha256",
        "metric_sha256",
    ]
    rows = [
        [
            AUTHORITY,
            str(PRODUCTION).lower(),
            str(EXECUTION_AUTHORIZED).lower(),
            str(RESULT_CLAIMED).lower(),
            STATUS,
            row.seed,
            row.system_id,
            SYSTEM_NAMES[row.system_id],
            row.run_id,
            _float_hex(row.t2m_cluster_macro_r_at_1),
            _float_hex(row.m2t_any_caption_r_at_1),
            _float_hex(row.bidirectional_mean_r_at_1),
            row.score_matrix_sha256,
            row.metric_sha256,
        ]
        for row in checked.validation_rows
    ]
    return _csv_bytes(header, rows)


def hypothesis_table_csv_bytes(report: object) -> bytes:
    """Export H1--H6, with Holm fields applicable only to H2--H6."""

    checked = _validated_report(report)
    header = [
        "authority",
        "production",
        "execution_authorized",
        "result_claimed",
        "status",
        "hypothesis_id",
        "comparison",
        "treatment_system_id",
        "baseline_system_id",
        "effect_hex",
        "raw_p_value_hex",
        "holm_family_member",
        "holm_rank",
        "holm_adjusted_p_value_hex",
        "statistic_sha256",
    ]
    rows = [
        [
            AUTHORITY,
            str(PRODUCTION).lower(),
            str(EXECUTION_AUTHORIZED).lower(),
            str(RESULT_CLAIMED).lower(),
            STATUS,
            row.hypothesis_id,
            f"{row.treatment_system_id}-minus-{row.baseline_system_id}",
            row.treatment_system_id,
            row.baseline_system_id,
            _float_hex(row.effect),
            _float_hex(row.raw_p_value),
            str(row.hypothesis_id in HOLM_HYPOTHESIS_ORDER).lower(),
            row.holm_rank if row.holm_rank is not None else "NOT_APPLICABLE",
            (
                _float_hex(row.holm_adjusted_p_value)
                if row.holm_adjusted_p_value is not None
                else "NOT_APPLICABLE"
            ),
            row.statistic_sha256,
        ]
        for row in checked.hypothesis_rows
    ]
    return _csv_bytes(header, rows)


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_authority_comment() -> str:
    return (
        "% authority=0 production=false execution_authorized=false "
        f"result_claimed=false status={STATUS}"
    )


def validation_table_latex_bytes(report: object) -> bytes:
    """Export the canonical 21-row table as an Authority0 LaTeX fragment."""

    checked = _validated_report(report)
    lines = [
        _latex_authority_comment(),
        r"\begin{tabular}{rllrrr}",
        r"\toprule",
        r"\multicolumn{6}{c}{\textbf{AUTHORITY 0 -- NO RESULT -- NOT AUTHORIZED}} \\",
        r"\midrule",
        r"Seed & System & Name & T2M R@1 (hex) & M2T R@1 (hex) & Mean R@1 (hex) \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(
            (
                str(row.seed),
                row.system_id,
                _latex_escape(SYSTEM_NAMES[row.system_id]),
                _float_hex(row.t2m_cluster_macro_r_at_1),
                _float_hex(row.m2t_any_caption_r_at_1),
                _float_hex(row.bidirectional_mean_r_at_1),
            )
        )
        + r" \\"
        for row in checked.validation_rows
    )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return ("\n".join(lines) + "\n").encode("utf-8")


def hypothesis_table_latex_bytes(report: object) -> bytes:
    """Export H1--H6 as an Authority0 LaTeX fragment."""

    checked = _validated_report(report)
    lines = [
        _latex_authority_comment(),
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"\multicolumn{6}{c}{\textbf{AUTHORITY 0 -- NO RESULT -- NOT AUTHORIZED}} \\",
        r"\midrule",
        r"Hypothesis & Comparison & Effect (hex) & Raw p (hex) & Holm rank & Holm p (hex) \\",
        r"\midrule",
    ]
    for row in checked.hypothesis_rows:
        rank = str(row.holm_rank) if row.holm_rank is not None else "N/A"
        adjusted = (
            _float_hex(row.holm_adjusted_p_value)
            if row.holm_adjusted_p_value is not None
            else "N/A"
        )
        lines.append(
            " & ".join(
                (
                    row.hypothesis_id,
                    f"{row.treatment_system_id}-{row.baseline_system_id}",
                    _float_hex(row.effect),
                    _float_hex(row.raw_p_value),
                    rank,
                    adjusted,
                )
            )
            + r" \\"
        )
    lines.extend((r"\bottomrule", r"\end{tabular}"))
    return ("\n".join(lines) + "\n").encode("utf-8")


__all__ = [
    "AUTHORITY",
    "EXECUTION_AUTHORIZED",
    "HYPOTHESIS_COMPARISONS",
    "HYPOTHESIS_ORDER",
    "HOLM_HYPOTHESIS_ORDER",
    "PRODUCTION",
    "RESULT_CLAIMED",
    "SEED_ORDER",
    "STATUS",
    "SYSTEM_NAMES",
    "SYSTEM_ORDER",
    "AuthorityZeroStatisticalReport",
    "FrozenHypothesisResult",
    "FrozenValidationResult",
    "HolmAdjustment",
    "ReportedHypothesisResult",
    "StatisticalReportingError",
    "build_authority0_statistical_report",
    "canonical_statistical_report_json_bytes",
    "collect_frozen_hypothesis_results",
    "collect_frozen_validation_results",
    "holm_step_down_secondary",
    "hypothesis_table_csv_bytes",
    "hypothesis_table_latex_bytes",
    "validation_table_csv_bytes",
    "validation_table_latex_bytes",
]
