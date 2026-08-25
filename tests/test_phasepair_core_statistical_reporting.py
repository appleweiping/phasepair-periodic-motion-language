"""Synthetic-only tests for Authority0 PhasePair statistical reporting."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import pytest

from phasepair_core import statistical_reporting as reporting


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _validation_rows() -> tuple[reporting.FrozenValidationResult, ...]:
    rows: list[reporting.FrozenValidationResult] = []
    for seed_ordinal, seed in enumerate(reporting.SEED_ORDER):
        for system_ordinal, system_id in enumerate(reporting.SYSTEM_ORDER):
            ordinal = seed_ordinal * len(reporting.SYSTEM_ORDER) + system_ordinal
            t2m = (ordinal + 1) / 32.0
            m2t = (ordinal + 2) / 32.0
            role = "BASE_TRAIN" if system_id == "00" else "RESIDUAL_HEAD_TRAIN"
            rows.append(
                reporting.FrozenValidationResult(
                    seed=seed,
                    system_id=system_id,
                    run_id=f"phasepair-run-v2/{role}/{seed}/{system_id}",
                    t2m_cluster_macro_r_at_1=t2m,
                    m2t_any_caption_r_at_1=m2t,
                    bidirectional_mean_r_at_1=0.5 * (t2m + m2t),
                    score_matrix_sha256=_digest(f"score/{seed}/{system_id}"),
                    metric_sha256=_digest(f"metric/{seed}/{system_id}"),
                )
            )
    return tuple(rows)


_RAW_P_VALUES = {
    "H1": 0.005,
    "H2": 0.01,
    "H3": 0.04,
    "H4": 0.03,
    "H5": 0.002,
    "H6": 0.05,
}


def _hypothesis_rows() -> tuple[reporting.FrozenHypothesisResult, ...]:
    return tuple(
        reporting.FrozenHypothesisResult(
            hypothesis_id=hypothesis_id,
            treatment_system_id=reporting.HYPOTHESIS_COMPARISONS[hypothesis_id][0],
            baseline_system_id=reporting.HYPOTHESIS_COMPARISONS[hypothesis_id][1],
            effect=(7 - ordinal) / 100.0,
            raw_p_value=_RAW_P_VALUES[hypothesis_id],
            statistic_sha256=_digest(f"statistic/{hypothesis_id}"),
        )
        for ordinal, hypothesis_id in enumerate(reporting.HYPOTHESIS_ORDER, start=1)
    )


def _report() -> reporting.AuthorityZeroStatisticalReport:
    # Both collectors must canonicalize, rather than trust, caller ordering.
    return reporting.build_authority0_statistical_report(
        tuple(reversed(_validation_rows())),
        tuple(reversed(_hypothesis_rows())),
    )


def test_fixed_census_order_names_and_hypothesis_comparisons() -> None:
    assert reporting.SEED_ORDER == (1729, 2718, 31415)
    assert reporting.SYSTEM_ORDER == ("00", "01", "02", "03", "04", "05", "06")
    assert tuple(reporting.SYSTEM_NAMES) == reporting.SYSTEM_ORDER
    assert reporting.HYPOTHESIS_ORDER == ("H1", "H2", "H3", "H4", "H5", "H6")
    assert reporting.HOLM_HYPOTHESIS_ORDER == ("H2", "H3", "H4", "H5", "H6")
    assert dict(reporting.HYPOTHESIS_COMPARISONS) == {
        "H1": ("06", "00"),
        "H2": ("06", "01"),
        "H3": ("06", "02"),
        "H4": ("06", "03"),
        "H5": ("06", "04"),
        "H6": ("06", "05"),
    }


def test_validation_collector_restores_seed_outer_system_inner_order() -> None:
    collected = reporting.collect_frozen_validation_results(
        tuple(reversed(_validation_rows()))
    )
    assert len(collected) == 21
    assert tuple((row.seed, row.system_id) for row in collected) == tuple(
        (seed, system_id)
        for seed in reporting.SEED_ORDER
        for system_id in reporting.SYSTEM_ORDER
    )
    assert collected[0].run_id == "phasepair-run-v2/BASE_TRAIN/1729/00"
    assert collected[1].run_id == "phasepair-run-v2/RESIDUAL_HEAD_TRAIN/1729/01"
    assert collected[-1].run_id == "phasepair-run-v2/RESIDUAL_HEAD_TRAIN/31415/06"


def test_validation_collector_rejects_duplicate_missing_or_rebound_identities() -> None:
    rows = _validation_rows()
    with pytest.raises(
        reporting.StatisticalReportingError, match="duplicate validation"
    ):
        reporting.collect_frozen_validation_results(rows[:-1] + (rows[0],))
    with pytest.raises(reporting.StatisticalReportingError, match="missing validation"):
        reporting.collect_frozen_validation_results(rows[:-1])
    with pytest.raises(TypeError, match="exact built-in tuple"):
        reporting.collect_frozen_validation_results(list(rows))

    rebound = replace(rows[0], run_id="phasepair-run-v2/RESIDUAL_HEAD_TRAIN/1729/01")
    with pytest.raises(
        reporting.StatisticalReportingError, match="exact seed/system identity"
    ):
        reporting.collect_frozen_validation_results((rebound,) + rows[1:])

    wrong_system = replace(rows[0], system_id="07")
    with pytest.raises(reporting.StatisticalReportingError, match="system census"):
        reporting.collect_frozen_validation_results((wrong_system,) + rows[1:])


@pytest.mark.parametrize(
    "mutation, message",
    [
        (
            lambda row: replace(row, seed=True),
            "exact built-in int",
        ),
        (
            lambda row: replace(row, t2m_cluster_macro_r_at_1=float("nan")),
            "finite",
        ),
        (
            lambda row: replace(row, m2t_any_caption_r_at_1=True),
            "not bool",
        ),
        (
            lambda row: replace(row, bidirectional_mean_r_at_1=0.0),
            "exact full-gallery mean",
        ),
        (
            lambda row: replace(row, score_matrix_sha256="A" * 64),
            "lowercase SHA-256",
        ),
    ],
)
def test_validation_collector_rejects_noncanonical_frozen_values(
    mutation, message: str
) -> None:
    rows = _validation_rows()
    with pytest.raises((TypeError, reporting.StatisticalReportingError), match=message):
        reporting.collect_frozen_validation_results((mutation(rows[0]),) + rows[1:])


def test_hypothesis_collector_restores_order_and_rejects_mapping_mutants() -> None:
    rows = _hypothesis_rows()
    collected = reporting.collect_frozen_hypothesis_results(tuple(reversed(rows)))
    assert tuple(row.hypothesis_id for row in collected) == reporting.HYPOTHESIS_ORDER
    assert tuple(
        (row.treatment_system_id, row.baseline_system_id) for row in collected
    ) == tuple(
        reporting.HYPOTHESIS_COMPARISONS[hypothesis_id]
        for hypothesis_id in reporting.HYPOTHESIS_ORDER
    )

    wrong_comparison = replace(rows[3], baseline_system_id="02")
    with pytest.raises(reporting.StatisticalReportingError, match="fixed H4"):
        reporting.collect_frozen_hypothesis_results(
            rows[:3] + (wrong_comparison,) + rows[4:]
        )
    with pytest.raises(
        reporting.StatisticalReportingError, match="duplicate hypothesis"
    ):
        reporting.collect_frozen_hypothesis_results(rows[:-1] + (rows[0],))
    with pytest.raises(reporting.StatisticalReportingError, match="missing hypothesis"):
        reporting.collect_frozen_hypothesis_results(rows[:-1])


def test_holm_known_vector_is_monotone_and_restored_to_h2_h6_order() -> None:
    adjusted = reporting.holm_step_down_secondary(
        {
            hypothesis_id: _RAW_P_VALUES[hypothesis_id]
            for hypothesis_id in reporting.HOLM_HYPOTHESIS_ORDER
        }
    )
    assert (
        tuple(row.hypothesis_id for row in adjusted) == reporting.HOLM_HYPOTHESIS_ORDER
    )
    by_id = {row.hypothesis_id: row for row in adjusted}
    assert {
        hypothesis_id: by_id[hypothesis_id].sorted_rank for hypothesis_id in by_id
    } == {
        "H2": 2,
        "H3": 4,
        "H4": 3,
        "H5": 1,
        "H6": 5,
    }
    assert by_id["H2"].adjusted_p_value == pytest.approx(0.04)
    assert by_id["H3"].adjusted_p_value == pytest.approx(0.09)
    assert by_id["H4"].adjusted_p_value == pytest.approx(0.09)
    assert by_id["H5"].adjusted_p_value == pytest.approx(0.01)
    assert by_id["H6"].adjusted_p_value == pytest.approx(0.09)
    sorted_adjusted = [
        row.adjusted_p_value
        for row in sorted(adjusted, key=lambda row: row.sorted_rank)
    ]
    assert sorted_adjusted == sorted(sorted_adjusted)


def test_holm_ties_use_hypothesis_id_and_extremes_are_capped() -> None:
    adjusted = reporting.holm_step_down_secondary(
        {"H2": 0.01, "H3": 0.01, "H4": 0, "H5": 1, "H6": 1}
    )
    by_id = {row.hypothesis_id: row for row in adjusted}
    assert by_id["H4"].sorted_rank == 1
    assert by_id["H2"].sorted_rank == 2
    assert by_id["H3"].sorted_rank == 3
    assert by_id["H5"].sorted_rank == 4
    assert by_id["H6"].sorted_rank == 5
    assert by_id["H4"].adjusted_p_value == 0.0
    assert by_id["H2"].adjusted_p_value == pytest.approx(0.04)
    assert by_id["H3"].adjusted_p_value == pytest.approx(0.04)
    assert by_id["H5"].adjusted_p_value == 1.0
    assert by_id["H6"].adjusted_p_value == 1.0


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), -0.1, 1.1])
def test_holm_rejects_bool_nonfinite_and_out_of_range_values(invalid: object) -> None:
    p_values: dict[str, object] = {
        hypothesis_id: 0.01 for hypothesis_id in reporting.HOLM_HYPOTHESIS_ORDER
    }
    p_values["H3"] = invalid
    with pytest.raises((TypeError, reporting.StatisticalReportingError)):
        reporting.holm_step_down_secondary(p_values)

    if invalid is True:
        with pytest.raises(TypeError, match="exact built-in dict"):
            reporting.holm_step_down_secondary(tuple(p_values.items()))


def test_holm_rejects_any_family_other_than_exactly_h2_h6() -> None:
    complete = {
        hypothesis_id: 0.01 for hypothesis_id in reporting.HOLM_HYPOTHESIS_ORDER
    }
    missing = dict(complete)
    del missing["H6"]
    with pytest.raises(reporting.StatisticalReportingError, match="exactly H2-H6"):
        reporting.holm_step_down_secondary(missing)
    extra = dict(complete)
    extra["H1"] = 0.01
    with pytest.raises(reporting.StatisticalReportingError, match="extra=.*H1"):
        reporting.holm_step_down_secondary(extra)


def test_report_never_adds_h1_to_holm_family_or_derives_input_statistics() -> None:
    report = _report()
    assert report.validation_rows == _validation_rows()
    assert (
        tuple(row.hypothesis_id for row in report.hypothesis_rows)
        == reporting.HYPOTHESIS_ORDER
    )
    h1 = report.hypothesis_rows[0]
    assert h1.effect == _hypothesis_rows()[0].effect
    assert h1.raw_p_value == _hypothesis_rows()[0].raw_p_value
    assert h1.holm_rank is None
    assert h1.holm_adjusted_p_value is None
    assert all(row.holm_rank is not None for row in report.hypothesis_rows[1:])


def test_json_csv_and_latex_exports_are_canonical_authority0_bytes() -> None:
    report = _report()
    raw_json = reporting.canonical_statistical_report_json_bytes(report)
    assert raw_json.endswith(b"\n") and b"\r" not in raw_json
    payload = json.loads(raw_json)
    assert payload["schema"] == "phasepair-frozen-statistical-report-v1"
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["execution_authorized"] is False
    assert payload["result_claimed"] is False
    assert payload["status"] == reporting.STATUS
    assert payload["row_order"] == "seed_outer_system_inner"
    assert payload["validation_row_count"] == 21
    assert payload["hypothesis_row_count"] == 6
    assert [(row["seed"], row["system_id"]) for row in payload["validation_rows"]] == [
        (seed, system_id)
        for seed in reporting.SEED_ORDER
        for system_id in reporting.SYSTEM_ORDER
    ]
    assert payload["hypothesis_rows"][0]["holm_rank"] == "NOT_APPLICABLE"
    assert (
        payload["hypothesis_rows"][0]["holm_adjusted_p_value_hex"] == "NOT_APPLICABLE"
    )
    assert all(
        type(row["raw_p_value_hex"]) is str for row in payload["hypothesis_rows"]
    )
    canonical = (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode("utf-8")
        + b"\n"
    )
    assert raw_json == canonical

    validation_csv = reporting.validation_table_csv_bytes(report)
    hypothesis_csv = reporting.hypothesis_table_csv_bytes(report)
    assert validation_csv.endswith(b"\n") and b"\r" not in validation_csv
    assert hypothesis_csv.endswith(b"\n") and b"\r" not in hypothesis_csv
    assert len(validation_csv.decode("utf-8").splitlines()) == 22
    assert len(hypothesis_csv.decode("utf-8").splitlines()) == 7
    assert validation_csv.splitlines()[1].startswith(
        b"0,false,false,false,DATA_FREE_FROZEN_STATISTICS_AUTHORITY0_NO_RESULT,1729,00,"
    )
    assert b"H1,06-minus-00,06,00" in hypothesis_csv
    assert b"H1,06-minus-00,06,00,0x" in hypothesis_csv
    assert b",false,NOT_APPLICABLE,NOT_APPLICABLE," in hypothesis_csv

    validation_latex = reporting.validation_table_latex_bytes(report)
    hypothesis_latex = reporting.hypothesis_table_latex_bytes(report)
    for raw in (validation_latex, hypothesis_latex):
        assert raw.endswith(b"\n") and b"\r" not in raw
        assert raw.startswith(
            b"% authority=0 production=false execution_authorized=false"
        )
        assert reporting.STATUS.encode("ascii") in raw
        visible = raw.split(b"\\begin{tabular}", maxsplit=1)[1]
        assert (
            b"\\multicolumn{6}{c}{\\textbf{AUTHORITY 0 -- NO RESULT -- NOT AUTHORIZED}}"
            in visible
        )
        assert not visible.lstrip().startswith(b"%")
    assert b"wamo\\_marginal\\_wavelet" in validation_latex
    assert b"H1 & 06-00" in hypothesis_latex
    assert b"H2 & 06-01" in hypothesis_latex
    assert {
        "json": (len(raw_json), hashlib.sha256(raw_json).hexdigest()),
        "validation_csv": (
            len(validation_csv),
            hashlib.sha256(validation_csv).hexdigest(),
        ),
        "hypothesis_csv": (
            len(hypothesis_csv),
            hashlib.sha256(hypothesis_csv).hexdigest(),
        ),
        "validation_latex": (
            len(validation_latex),
            hashlib.sha256(validation_latex).hexdigest(),
        ),
        "hypothesis_latex": (
            len(hypothesis_latex),
            hashlib.sha256(hypothesis_latex).hexdigest(),
        ),
    } == {
        "json": (
            12301,
            "befad72c553923155a5fe2aa699bcf6dbf751bd04f6d5672444ae22e20dd5b3a",
        ),
        "validation_csv": (
            7141,
            "f6245aa11807dcc4a1c7b4e07fccdc3f8d9ef40701423d71cff535ef858a24f3",
        ),
        "hypothesis_csv": (
            1585,
            "79d6420f064967e08127cad4440b9b933f97af9ec2fe25ca20be0e2ea339f856",
        ),
        "validation_latex": (
            2476,
            "8b8b3f84c9eaa29c1009fde6d4de07a0e58acec28c1243072af8e1ca0d15c3ed",
        ),
        "hypothesis_latex": (
            877,
            "e886850855a520e4112b9b26a2e631d5e880d4d904207a0f276cd3e9d07b98fe",
        ),
    }


def test_exports_revalidate_identity_and_holm_annotations_after_rebinding() -> None:
    report = _report()
    rebound_validation = replace(
        report.validation_rows[0],
        run_id="phasepair-run-v2/RESIDUAL_HEAD_TRAIN/1729/01",
    )
    rebound_report = replace(
        report,
        validation_rows=(rebound_validation,) + report.validation_rows[1:],
    )
    with pytest.raises(
        reporting.StatisticalReportingError, match="exact seed/system identity"
    ):
        reporting.canonical_statistical_report_json_bytes(rebound_report)

    forged_holm = replace(report.hypothesis_rows[1], holm_rank=5)
    forged_report = replace(
        report,
        hypothesis_rows=(report.hypothesis_rows[0], forged_holm)
        + report.hypothesis_rows[2:],
    )
    with pytest.raises(
        reporting.StatisticalReportingError, match="Holm annotation changed"
    ):
        reporting.hypothesis_table_csv_bytes(forged_report)


def test_authority_zero_no_result_constants_are_unambiguous() -> None:
    assert reporting.AUTHORITY == 0
    assert reporting.PRODUCTION is False
    assert reporting.EXECUTION_AUTHORIZED is False
    assert reporting.RESULT_CLAIMED is False
    assert reporting.STATUS.endswith("AUTHORITY0_NO_RESULT")
