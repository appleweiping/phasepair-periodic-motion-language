from __future__ import annotations

from dataclasses import replace
import hashlib
import json

import numpy as np
import pytest

from phaseset_core import evaluation, experiments, publication, statistics


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _canonical_json(value: object) -> bytes:
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


def _evaluation_aggregate() -> bytes:
    score_rows = []
    systems = []
    for system_id, _ in experiments.FINAL_SYSTEMS:
        seed_reports = []
        for seed in experiments.SEEDS:
            evaluation_report_sha256 = _digest(f"evaluation-{system_id}-{seed}")
            seed_reports.append(evaluation_report_sha256)
            score_rows.append(
                {
                    "capacity_receipt_sha256": _digest(
                        f"capacity-{system_id}-{seed}"
                    ),
                    "checkpoint_sha256": _digest(f"checkpoint-{system_id}-{seed}"),
                    "environment_sha256": _digest("environment"),
                    "evaluation_report_sha256": evaluation_report_sha256,
                    "hard_text_to_motion_r1_hex": float(0.5).hex(),
                    "primary_hex": float(0.75).hex(),
                    "qualification_sha256": _digest(f"qualification-{seed}"),
                    "run_manifest_sha256": _digest(f"manifest-{system_id}-{seed}"),
                    "score_provenance_sha256": _digest(
                        f"score-provenance-{system_id}-{seed}"
                    ),
                    "score_sha256": _digest(f"score-{system_id}-{seed}"),
                    "scoring_code_sha256": _digest("scoring-code"),
                    "seed": seed,
                    "system_id": system_id,
                    "terminal_sha256": _digest(f"terminal-{system_id}-{seed}"),
                    "text_tower_sha256": _digest("text-tower"),
                }
            )
        systems.append(
            {
                "aggregation": (
                    "MEAN_OF_THREE_INDEPENDENT_SEED_METRICS_NO_LOGIT_ENSEMBLE"
                ),
                "hard_text_to_motion_r1_mean_hex": float(0.5).hex(),
                "hard_text_to_motion_r1_population_std_hex": float(0.0).hex(),
                "primary_mean_hex": float(0.75).hex(),
                "primary_population_std_hex": float(0.0).hex(),
                "seed_evaluation_report_sha256s": seed_reports,
                "system_id": system_id,
            }
        )
    execution_keys = (
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
    execution_census = hashlib.sha256(
        _canonical_json(
            {
                "rows": [
                    {name: row[name] for name in execution_keys} for row in score_rows
                ],
                "schema": "phaseset-score-execution-census-v1",
            }
        )
    ).hexdigest()
    return _canonical_json(
        {
            "caption_manifest_sha256": _digest("caption-manifest"),
            "census_provenance_sha256": _digest("census-provenance"),
            "census_sha256": _digest("census"),
            "evaluation_type": "REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL",
            "evaluator_sha256": _digest("evaluator"),
            "hard_gallery_collection_sha256": _digest("hard-gallery"),
            "hard_gallery_freeze_binding_sha256": _digest("hard-freeze"),
            "schema": "phaseset-evaluation-aggregate-v3",
            "score_execution_census_sha256": execution_census,
            "score_rows": score_rows,
            "sealed_test_consumption_sha256": _digest("sealed-consumption"),
            "split": "test",
            "systems": systems,
            "test_manifest_sha256": _digest("test-manifest"),
        }
    )


def _reports() -> dict[str, tuple[evaluation.EvaluationReport, ...]]:
    motion_commitments = tuple(bytes([index + 1]) * 32 for index in range(4))
    caption_commitments = tuple(bytes([index + 11]) * 32 for index in range(4))
    common = {
        "motion_commitments": motion_commitments,
        "caption_commitments": caption_commitments,
        "positive_motion_indices": tuple((index,) for index in range(4)),
        "group_sizes": np.asarray([3, 3, 4, 4], dtype=np.int64),
        "component_labels": ("C0", "C0", "C1", "C1"),
    }
    reports = {}
    for system_id, _ in experiments.FINAL_SYSTEMS:
        rows = []
        for seed in experiments.SEEDS:
            scores = (
                np.eye(4, dtype=np.float64)
                if system_id == "08"
                else np.roll(np.eye(4, dtype=np.float64), 1, axis=0)
            )
            report = evaluation.evaluate_retrieval(
                evaluation.RetrievalDataset(scores=scores, **common)
            )
            rows.append(
                replace(
                    report,
                    evaluation_type="REAL_GT_HOLISTIC_CAPTURE_RETRIEVAL",
                    census_provenance_sha256=_digest("publication-census"),
                    score_provenance_sha256=_digest(
                        f"publication-score-{system_id}-{seed}"
                    ),
                )
            )
        reports[system_id] = tuple(rows)
    return reports


def _statistical_report(aggregate: bytes) -> statistics.StatisticalReport:
    reports = _reports()
    comparisons = tuple(
        statistics.HypothesisComparison(
            hypothesis_id=hypothesis_id,
            treatment_system_id=treatment,
            baseline_system_id=baseline,
            training_seeds=experiments.SEEDS,
            treatment_evaluations=reports[treatment],
            baseline_evaluations=reports[baseline],
            resampling_seed=20260825,
            aggregate_artifact=aggregate,
            evaluator_artifact=b"publication-test-evaluator",
        )
        for hypothesis_id, treatment, baseline in experiments.HYPOTHESIS_COMPARISONS
    )
    return statistics.build_statistical_report(comparisons)


def _resource_rows() -> tuple[publication.ResourceMeasurement, ...]:
    rows = []
    for system_index, (system_id, _) in enumerate(experiments.FINAL_SYSTEMS):
        for seed in experiments.SEEDS:
            rows.append(
                publication.ResourceMeasurement(
                    workload=publication.PRIMARY_WORKLOAD,
                    system_id=system_id,
                    seed=seed,
                    group_size=None,
                    parameter_count=1_000_000 + system_index,
                    forward_flops=10_000_000 + system_index,
                    peak_allocated_bytes=2_000_000 + seed,
                    window_count=64,
                    elapsed_ns=1_000_000 + seed,
                    checkpoint_sha256=_digest(f"checkpoint-{system_id}-{seed}"),
                    terminal_sha256=_digest(f"terminal-{system_id}-{seed}"),
                    environment_sha256=_digest("environment"),
                    device_sha256=_digest("device"),
                    profiler_code_sha256=_digest("profiler"),
                    measurement_artifact_sha256=_digest(
                        f"measurement-primary-{system_id}-{seed}"
                    ),
                    precision="FP32",
                    measurement_type=publication.SYNTHETIC_MEASUREMENT_TYPE,
                )
            )
    full = next(
        row
        for row in rows
        if row.system_id == "08" and row.seed == publication.PROFILE_SEED
    )
    for group_size in publication.SCALING_GROUP_SIZES:
        rows.append(
            publication.ResourceMeasurement(
                workload=publication.SCALING_WORKLOAD,
                system_id="08",
                seed=publication.PROFILE_SEED,
                group_size=group_size,
                parameter_count=full.parameter_count,
                forward_flops=10_000_000 * group_size,
                peak_allocated_bytes=2_000_000 * group_size,
                window_count=16,
                elapsed_ns=1_000_000 * group_size,
                checkpoint_sha256=full.checkpoint_sha256,
                terminal_sha256=full.terminal_sha256,
                environment_sha256=full.environment_sha256,
                device_sha256=full.device_sha256,
                profiler_code_sha256=full.profiler_code_sha256,
                measurement_artifact_sha256=_digest(
                    f"measurement-scaling-{group_size}"
                ),
                precision=full.precision,
                measurement_type=full.measurement_type,
            )
        )
    return tuple(rows)


def test_resource_claim_and_publication_surfaces_share_one_aggregate() -> None:
    aggregate = _evaluation_aggregate()
    report = _statistical_report(aggregate)
    resources = publication.build_resource_report(
        _resource_rows(),
        evaluation_aggregate=aggregate,
    )
    resource_raw = publication.canonical_resource_report_bytes(resources)
    assert b'"group_size":256' in resource_raw

    claims = publication.build_claim_report(report)
    assert claims.maximum_claim == "TOPOLOGY_AND_PHASE_STRUCTURAL_CONTRIBUTION"
    assert claims.topology_supported is True
    assert claims.phase_supported is True
    assert publication.canonical_claim_report_bytes(claims).endswith(b"\n")

    bundle = publication.render_publication_bundle(
        evaluation_aggregate=aggregate,
        statistics=report,
        resources=resources,
        require_real_resources=False,
    )
    assert publication.canonical_publication_manifest_bytes(bundle) == bundle.manifest
    names = tuple(row.relative_name for row in bundle.artifacts)
    assert names == (
        "results/evaluation-aggregate.json",
        "results/statistical-report.json",
        "results/resource-report.json",
        "results/claim-report.json",
        "results/summary.csv",
        "paper/generated/table1.tex",
        "paper/generated/table2.tex",
    )
    table = next(
        row.content
        for row in bundle.artifacts
        if row.relative_name == "paper/generated/table1.tex"
    )
    assert b"PhaseSet full" in table


def test_publication_fails_closed_on_census_and_resource_provenance() -> None:
    aggregate = _evaluation_aggregate()
    rows = _resource_rows()
    with pytest.raises(publication.PublicationContractError, match="ordered 27"):
        publication.build_resource_report(
            tuple(reversed(rows)),
            evaluation_aggregate=aggregate,
        )

    changed_rows = list(rows)
    changed_rows[0] = replace(
        changed_rows[0],
        checkpoint_sha256=_digest("unrelated-resource-checkpoint"),
    )
    with pytest.raises(
        publication.PublicationContractError,
        match="evaluated checkpoint/terminal/environment",
    ):
        publication.build_resource_report(
            tuple(changed_rows),
            evaluation_aggregate=aggregate,
        )

    resources = publication.build_resource_report(rows, evaluation_aggregate=aggregate)
    report = _statistical_report(aggregate)
    with pytest.raises(publication.PublicationContractError, match="real target-runtime"):
        publication.render_publication_bundle(
            evaluation_aggregate=aggregate,
            statistics=report,
            resources=resources,
        )

    changed = aggregate.replace(b'"split":"test"', b'"split":"validation"')
    with pytest.raises(publication.PublicationContractError, match="sealed-test"):
        publication.render_publication_bundle(
            evaluation_aggregate=changed,
            statistics=report,
            resources=resources,
            require_real_resources=False,
        )

    parsed = json.loads(aggregate)
    parsed["systems"][0]["primary_mean_hex"] = float(0.99).hex()
    with pytest.raises(
        publication.PublicationContractError,
        match="was not rebuilt from seeds",
    ):
        publication.render_publication_bundle(
            evaluation_aggregate=_canonical_json(parsed),
            statistics=report,
            resources=resources,
            require_real_resources=False,
        )
