from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import hashlib
import json

import numpy as np
import pytest

from phaseset_core import evaluation, statistics


def _identity(domain: str, index: int) -> str:
    return hashlib.sha256(f"{domain}-{index}".encode("ascii")).hexdigest()


def _row(
    index: int,
    treatment_text: tuple[int, ...],
    control_text: tuple[int, ...],
    treatment_motion: tuple[int, ...],
    control_motion: tuple[int, ...],
) -> statistics.PairedCapture:
    caption_ids = tuple(
        sorted(_identity(f"caption-{index}", ordinal) for ordinal in range(len(treatment_text)))
    )
    motion_ids = tuple(
        sorted(_identity(f"motion-{index}", ordinal) for ordinal in range(len(treatment_motion)))
    )
    return statistics.PairedCapture(
        capture_id=_identity("capture", index),
        caption_ids=caption_ids,
        motion_ids=motion_ids,
        treatment_text_to_motion_hits=treatment_text,
        control_text_to_motion_hits=control_text,
        treatment_motion_to_text_hits=treatment_motion,
        control_motion_to_text_hits=control_motion,
    )


def _captures() -> tuple[statistics.PairedCapture, ...]:
    return (
        _row(1, (0, 0, 0), (0, 1, 1), (0,), (0,)),
        _row(0, (1,), (0,), (1,), (0,)),
    )


def _evaluation_pair() -> tuple[evaluation.EvaluationReport, evaluation.EvaluationReport]:
    motion_commitments = tuple(bytes([index + 1]) * 32 for index in range(4))
    caption_commitments = tuple(bytes([index + 11]) * 32 for index in range(4))
    common = {
        "motion_commitments": motion_commitments,
        "caption_commitments": caption_commitments,
        "positive_motion_indices": tuple((index,) for index in range(4)),
        "group_sizes": np.asarray([3, 3, 4, 4], dtype=np.int64),
        "component_labels": ("C0", "C0", "C1", "C1"),
    }
    treatment = evaluation.evaluate_retrieval(
        evaluation.RetrievalDataset(scores=np.eye(4, dtype=np.float64), **common)
    )
    baseline = evaluation.evaluate_retrieval(
        evaluation.RetrievalDataset(
            scores=np.roll(np.eye(4, dtype=np.float64), 1, axis=0),
            **common,
        )
    )
    census_sha256 = _identity("formal-census", 0)
    return (
        replace(
            treatment,
            evaluation_type="REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL",
            census_provenance_sha256=census_sha256,
            score_provenance_sha256=_identity("formal-score", 8),
        ),
        replace(
            baseline,
            evaluation_type="REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL",
            census_provenance_sha256=census_sha256,
            score_provenance_sha256=_identity("formal-score", 0),
        ),
    )


def _formal_comparisons() -> tuple[statistics.HypothesisComparison, ...]:
    treatment, baseline = _evaluation_pair()
    treatment_evaluations = tuple(
        replace(
            treatment,
            score_provenance_sha256=_identity(f"formal-score-treatment-{seed}", 8),
        )
        for seed in statistics.TRAINING_SEEDS
    )
    baseline_evaluations = tuple(
        replace(
            baseline,
            score_provenance_sha256=_identity(f"formal-score-baseline-{seed}", 0),
        )
        for seed in statistics.TRAINING_SEEDS
    )
    return tuple(
        statistics.HypothesisComparison(
            hypothesis_id=hypothesis_id,
            treatment_system_id=treatment_system_id,
            baseline_system_id=baseline_system_id,
            training_seeds=statistics.TRAINING_SEEDS,
            treatment_evaluations=treatment_evaluations,
            baseline_evaluations=baseline_evaluations,
            resampling_seed=20260825,
            aggregate_artifact=b"frozen-aggregate-v1",
            evaluator_artifact=b"frozen-evaluator-v1",
        )
        for hypothesis_id, treatment_system_id, baseline_system_id in (
            statistics.HYPOTHESIS_COMPARISONS
        )
    )


@pytest.fixture(scope="module")
def formal_report() -> statistics.StatisticalReport:
    return statistics.build_statistical_report(_formal_comparisons())


def test_variable_caption_metric_is_capture_macro_not_caption_micro() -> None:
    captures = _captures()
    summary = statistics.summarize_variable_captions(captures)
    assert summary.capture_count == 2
    assert summary.total_caption_count == 4
    assert (summary.minimum_caption_count, summary.maximum_caption_count) == (1, 3)
    assert summary.total_motion_query_count == 2
    assert (summary.minimum_motion_query_count, summary.maximum_motion_query_count) == (1, 1)
    assert statistics.paired_capture_effect(captures) == pytest.approx(1 / 3)
    text_micro_effect = (1 - 2) / 4
    motion_micro_effect = (1 - 0) / 2
    bidirectional_query_micro_effect = 0.5 * (text_micro_effect + motion_micro_effect)
    assert statistics.paired_capture_effect(captures) != bidirectional_query_micro_effect


def test_paired_capture_rows_are_immutable_and_strictly_paired() -> None:
    row = _captures()[0]
    with pytest.raises(FrozenInstanceError):
        row.capture_id = "changed"  # type: ignore[misc]
    with pytest.raises(statistics.StatisticalContractError, match="text-query census"):
        statistics.paired_capture_effect(
            (
                statistics.PairedCapture(
                    capture_id=row.capture_id,
                    caption_ids=row.caption_ids,
                    motion_ids=row.motion_ids,
                    treatment_text_to_motion_hits=row.treatment_text_to_motion_hits,
                    control_text_to_motion_hits=(0, 1),
                    treatment_motion_to_text_hits=row.treatment_motion_to_text_hits,
                    control_motion_to_text_hits=row.control_motion_to_text_hits,
                ),
            )
        )
    with pytest.raises(statistics.StatisticalContractError, match="duplicate"):
        statistics.paired_capture_effect((row, row))


def test_evaluator_contributions_bind_both_directions_into_bootstrap_rows() -> None:
    motion_commitments = tuple(bytes([index + 1]) * 32 for index in range(4))
    caption_commitments = tuple(bytes([index + 11]) * 32 for index in range(4))
    positives = tuple((index,) for index in range(4))
    common = {
        "motion_commitments": motion_commitments,
        "caption_commitments": caption_commitments,
        "positive_motion_indices": positives,
        "group_sizes": np.asarray([3, 3, 4, 4], dtype=np.int64),
        "component_labels": ("C0", "C0", "C1", "C1"),
    }
    treatment_dataset = evaluation.RetrievalDataset(
        scores=np.eye(4, dtype=np.float64),
        **common,
    )
    control_dataset = evaluation.RetrievalDataset(
        scores=np.roll(np.eye(4, dtype=np.float64), 1, axis=0),
        **common,
    )
    treatment_report = evaluation.evaluate_retrieval(treatment_dataset)
    control_report = evaluation.evaluate_retrieval(control_dataset)
    rows = statistics.paired_captures_from_contributions(
        treatment_report.capture_contributions,
        control_report.capture_contributions,
    )
    assert statistics.paired_capture_effect(rows) == pytest.approx(
        treatment_report.full_gallery.primary - control_report.full_gallery.primary
    )
    tampered = list(control_report.capture_contributions)
    tampered[0] = evaluation.CaptureR1Contribution(
        capture_commitment=tampered[0].capture_commitment,
        caption_commitments=(b"z" * 32,),
        motion_commitments=tampered[0].motion_commitments,
        text_to_motion_hits=tampered[0].text_to_motion_hits,
        motion_to_text_hits=tampered[0].motion_to_text_hits,
    )
    with pytest.raises(statistics.StatisticalContractError, match="query identities differ"):
        statistics.paired_captures_from_contributions(
            treatment_report.capture_contributions,
            tuple(tampered),
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
    assert all(row.capture_id.encode("ascii") not in raw for row in _captures())
    assert b'"draws":100000' in raw
    assert statistics.PRIMARY_ENDPOINT_ID.encode("ascii") in raw


def test_bootstrap_rejects_unpaired_or_too_small_inputs() -> None:
    with pytest.raises(statistics.StatisticalContractError, match="at least two"):
        statistics.paired_capture_bootstrap((_captures()[0],), seed=1)
    with pytest.raises(statistics.StatisticalContractError):
        statistics.paired_capture_bootstrap(
            (_captures()[0], _row(2, (2,), (0,), (0,), (0,))),
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


def test_legacy_hand_filled_rows_are_explicitly_synthetic_only() -> None:
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
    assert report.schema == statistics.SYNTHETIC_REPORT_SCHEMA
    assert report.status == statistics.SYNTHETIC_STATUS
    assert not report.provenance_complete
    assert report.schema != statistics.REPORT_SCHEMA
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


def test_formal_report_derives_all_h1_h8_provenance_from_actual_results(
    formal_report: statistics.StatisticalReport,
) -> None:
    assert formal_report.schema == statistics.REPORT_SCHEMA
    assert formal_report.provenance_complete
    assert tuple(row.hypothesis_id for row in formal_report.hypotheses) == (
        statistics.HYPOTHESIS_ORDER
    )
    comparisons = _formal_comparisons()
    first_comparison = comparisons[0]
    expected_bootstrap = statistics.paired_seed_blocked_bootstrap(
        tuple(
            (
                seed,
                statistics.paired_captures_from_contributions(
                    treatment.capture_contributions,
                    baseline.capture_contributions,
                ),
            )
            for seed, treatment, baseline in zip(
                first_comparison.training_seeds,
                first_comparison.treatment_evaluations,
                first_comparison.baseline_evaluations,
                strict=True,
            )
        ),
        resampling_seed=first_comparison.resampling_seed,
    )
    first = formal_report.hypotheses[0]
    assert isinstance(first, statistics.ProvenancedHypothesisResult)
    assert first.treatment_system_id == "08"
    assert first.baseline_system_id == "00"
    assert first.primary_endpoint_id == statistics.PRIMARY_ENDPOINT_ID
    assert first.direction == statistics.DIRECTION_ID
    assert first.effect == expected_bootstrap.observed_effect
    assert first.confidence_lower == expected_bootstrap.confidence_lower
    assert first.confidence_upper == expected_bootstrap.confidence_upper
    assert first.raw_p_value == expected_bootstrap.two_sided_p_value
    assert first.resampling_seed == 20260825
    assert first.training_seeds == statistics.TRAINING_SEEDS
    assert first.seed_count == 3
    assert first.draws == statistics.BOOTSTRAP_DRAWS
    assert (first.capture_count, first.caption_count, first.motion_count) == (4, 4, 4)
    assert first.component_count == 2
    for name in (
        "aggregate_sha256",
        "evaluator_sha256",
        "seed_block_rows_sha256",
        "capture_census_sha256",
        "index_stream_sha256",
        "bootstrap_sha256",
    ):
        assert len(getattr(first, name)) == 64
    assert all(len(value) == 64 for value in first.treatment_evaluation_sha256s)
    assert all(len(value) == 64 for value in first.baseline_evaluation_sha256s)
    family = formal_report.hypotheses[1:]
    assert len({row.index_stream_sha256 for row in family}) == 1
    assert len({row.capture_census_sha256 for row in family}) == 1
    assert len({row.resampling_seed for row in family}) == 1
    expected_holm = statistics.holm_h2_h8(
        {row.hypothesis_id: row.raw_p_value for row in family}
    )
    assert formal_report.holm_adjustments == expected_holm


def test_formal_public_bytes_are_digest_only_and_reconstruct_provenance(
    formal_report: statistics.StatisticalReport,
) -> None:
    raw = statistics.canonical_statistical_report_bytes(formal_report)
    payload = json.loads(raw)
    assert payload["schema"] == statistics.REPORT_SCHEMA
    assert payload["provenance_complete"] is True
    assert payload["shared_resampling_family"] == list(statistics.HYPOTHESIS_ORDER)
    assert payload["hypotheses"][0]["caption_count"] == 4
    assert "confidence_lower_hex" in payload["hypotheses"][0]
    assert "seed_block_rows_sha256" in payload["hypotheses"][0]
    assert payload["hypotheses"][0]["training_seeds"] == list(
        statistics.TRAINING_SEEDS
    )
    for comparison in formal_report._comparisons or ():
        assert comparison.aggregate_artifact not in raw
        assert comparison.evaluator_artifact not in raw
        for report in comparison.treatment_evaluations:
            for contribution in report.capture_contributions:
                assert contribution.capture_commitment.hex().encode("ascii") not in raw
                assert contribution.capture_commitment not in raw


def test_formal_report_rejects_identity_stream_and_post_build_tampering(
    formal_report: statistics.StatisticalReport,
) -> None:
    comparisons = list(_formal_comparisons())
    comparisons[1] = replace(comparisons[1], baseline_system_id="00")
    with pytest.raises(statistics.StatisticalContractError, match="identity differs"):
        statistics.build_statistical_report(tuple(comparisons))

    comparisons = list(_formal_comparisons())
    comparisons[-1] = replace(comparisons[-1], resampling_seed=7)
    with pytest.raises(
        statistics.StatisticalContractError,
        match="frozen resampling provenance: resampling_seed differs",
    ):
        statistics.build_statistical_report(tuple(comparisons))

    comparisons = list(_formal_comparisons())
    comparisons[0] = replace(comparisons[0], resampling_seed=7)
    with pytest.raises(
        statistics.StatisticalContractError,
        match="H1-H8 must share frozen resampling provenance: resampling_seed differs",
    ):
        statistics.build_statistical_report(tuple(comparisons))

    first = formal_report.hypotheses[0]
    assert isinstance(first, statistics.ProvenancedHypothesisResult)
    forged_rows = (replace(first, effect=first.effect + 0.125),) + formal_report.hypotheses[1:]
    forged = replace(formal_report, hypotheses=forged_rows)
    with pytest.raises(
        statistics.StatisticalContractError,
        match="canonical provenance reconstruction",
    ):
        statistics.canonical_statistical_report_bytes(forged)
