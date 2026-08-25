from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from phaseset_core import statistics


def _captures() -> tuple[statistics.PairedCapture, ...]:
    return (
        statistics.PairedCapture("capture-b", (0, 0, 0), (0, 1, 1)),
        statistics.PairedCapture("capture-a", (1,), (0,)),
    )


def test_variable_caption_metric_is_capture_macro_not_caption_micro() -> None:
    captures = _captures()
    summary = statistics.summarize_variable_captions(captures)
    assert summary.capture_count == 2
    assert summary.total_caption_count == 4
    assert (summary.minimum_caption_count, summary.maximum_caption_count) == (1, 3)
    assert statistics.paired_capture_effect(captures) == pytest.approx(1 / 6)
    caption_micro_effect = (1 - 2) / 4
    assert statistics.paired_capture_effect(captures) != caption_micro_effect


def test_paired_capture_rows_are_immutable_and_strictly_paired() -> None:
    row = _captures()[0]
    with pytest.raises(FrozenInstanceError):
        row.capture_id = "changed"  # type: ignore[misc]
    with pytest.raises(statistics.StatisticalContractError, match="same caption census"):
        statistics.paired_capture_effect((statistics.PairedCapture("capture-a", (1,), (0, 1)),))
    with pytest.raises(statistics.StatisticalContractError, match="duplicate"):
        statistics.paired_capture_effect(
            (
                statistics.PairedCapture("capture-a", (1,), (0,)),
                statistics.PairedCapture("capture-a", (0,), (1,)),
            )
        )


def test_bootstrap_is_exactly_100k_paired_deterministic_and_order_independent() -> None:
    first = statistics.paired_capture_bootstrap(_captures(), seed=20260825)
    second = statistics.paired_capture_bootstrap(tuple(reversed(_captures())), seed=20260825)
    assert first == second
    assert first.draws == 100_000
    assert first.capture_count == 2
    assert len(first.index_stream_sha256) == 64
    raw = statistics.canonical_bootstrap_result_bytes(first)
    assert b'"capture_id"' not in raw
    assert b'capture-a"' not in raw and b'capture-b"' not in raw
    assert b'"draws":100000' in raw


def test_bootstrap_rejects_unpaired_or_too_small_inputs() -> None:
    with pytest.raises(statistics.StatisticalContractError, match="at least two"):
        statistics.paired_capture_bootstrap(
            (statistics.PairedCapture("capture-a", (1,), (0,)),), seed=1
        )
    with pytest.raises(statistics.StatisticalContractError):
        statistics.paired_capture_bootstrap(
            (
                statistics.PairedCapture("capture-a", (1,), (0,)),
                statistics.PairedCapture("capture-b", (2,), (0,)),
            ),
            seed=1,
        )


def test_holm_family_is_exactly_h2_through_h8_and_monotone() -> None:
    p_values = {
        "H2": 0.01,
        "H3": 0.04,
        "H4": 0.03,
        "H5": 0.2,
        "H6": 0.001,
        "H7": 0.9,
        "H8": 0.05,
    }
    rows = statistics.holm_h2_h8(p_values)
    assert tuple(row.hypothesis_id for row in rows) == statistics.HOLM_FAMILY
    sorted_rows = sorted(rows, key=lambda row: row.sorted_rank)
    assert [row.adjusted_p_value for row in sorted_rows] == sorted(
        row.adjusted_p_value for row in sorted_rows
    )
    with pytest.raises(statistics.StatisticalContractError, match="exactly H2-H8"):
        statistics.holm_h2_h8({**p_values, "H1": 0.5})


def test_statistical_report_requires_h1_h8_and_exports_only_frozen_values() -> None:
    hypotheses = tuple(
        statistics.HypothesisResult(
            hypothesis_id=f"H{index}",
            effect=index / 100,
            raw_p_value=min(1.0, index / 20),
            bootstrap_sha256=f"{index:x}" * 64,
        )
        for index in range(1, 9)
    )
    report = statistics.build_statistical_report(hypotheses)
    assert len(report.hypotheses) == 8
    assert len(report.holm_adjustments) == 7
    raw = statistics.canonical_statistical_report_bytes(report)
    assert b'"holm_family":["H2","H3","H4","H5","H6","H7","H8"]' in raw
    payload = json.loads(raw)
    assert (
        tuple(
            (
                row["hypothesis_id"],
                row["treatment_system_id"],
                row["baseline_system_id"],
            )
            for row in payload["hypotheses"]
        )
        == statistics.HYPOTHESIS_COMPARISONS
    )
    with pytest.raises(statistics.StatisticalContractError, match="exactly H1-H8"):
        statistics.build_statistical_report(hypotheses[:-1])
