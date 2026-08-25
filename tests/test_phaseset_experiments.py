from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
from pathlib import Path

import pytest

from phaseset_core import experiments


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_system_census_matches_phaseset_contract() -> None:
    assert experiments.BASE_SYSTEMS == (
        ("B0", "ActorMean"),
        ("B1", "SetPMA"),
        ("B2", "SocialTemporal"),
    )
    assert experiments.FINAL_SYSTEMS == (
        ("00", "qualified group base"),
        ("01", "generic six tokens"),
        ("02", "marginal Morlet power"),
        ("03", "mean/difference DCT"),
        ("04", "PhasePair pair-bag"),
        ("05", "coverage/missing-only"),
        ("06", "incidence-shuffled"),
        ("07", "phase-stripped full"),
        ("08", "PhaseSet full"),
    )


def test_frozen_matrix_file_is_exact_canonical_contract() -> None:
    path = ROOT / "configs" / "phaseset" / "experiment_matrix.json"
    raw = experiments.load_experiment_matrix(path)
    assert raw == experiments.canonical_experiment_matrix_bytes()
    assert len(experiments.experiment_matrix_sha256()) == 64
    forbidden = (
        b"00_" + b"PERIODIC_BASE",
        b"EARLY_" + b"FUSION",
        b"LATE_" + b"FUSION",
        b"group-" + b"context",
        b"no-" + b"relation",
        b"dyadic-" + b"only",
        b"fixed-" + b"caption",
        b"NOT_" + b"QUALIFIED",
    )
    assert all(token not in raw for token in forbidden)


def test_registered_dag_is_exactly_nine_plus_twenty_four() -> None:
    plan = experiments.build_experiment_plan()
    assert len(plan.runs) == 33
    assert len(plan.base_runs) == 9
    assert len(plan.residual_runs) == 24
    assert tuple(row.run_id for row in plan.base_runs) == experiments.base_run_ids()
    assert tuple(row.run_id for row in plan.residual_runs) == experiments.residual_run_ids()
    assert {row.seed for row in plan.runs} == {1729, 2718, 31415}
    assert {row.system_id for row in plan.base_runs} == {"B0", "B1", "B2"}
    assert {row.system_id for row in plan.residual_runs} == {
        f"{index:02d}" for index in range(1, 9)
    }


def test_every_residual_binds_all_base_runs_winner_checkpoint_and_seed_cache() -> None:
    plan = experiments.build_experiment_plan()
    base_ids = experiments.base_run_ids()
    for row in plan.residual_runs:
        assert row.depends_on_run_ids == base_ids
        assert row.required_artifact_ids == (
            "base-qualification-v1",
            f"qualified-base-checkpoint-v1/{row.seed}",
            f"periodic-cache-v1/{row.seed}",
            "prepared-data-manifest-v1",
            "split-audit-v1",
            "runtime-preflight-v1",
        )


def test_plan_is_immutable_and_exact_byte_bound() -> None:
    plan = experiments.build_experiment_plan()
    with pytest.raises(FrozenInstanceError):
        plan.status = "READY"  # type: ignore[misc]
    raw = experiments.canonical_experiment_plan_bytes(plan)
    assert experiments.validate_experiment_plan_bytes(raw) == plan
    with pytest.raises(experiments.ExperimentContractError):
        experiments.validate_experiment_plan_bytes(
            raw.replace(b'"run_count":33', b'"run_count":34')
        )


@pytest.mark.parametrize(
    "run_id",
    (
        "../escape",
        "phaseset-run-v1/BASE_QUALIFICATION/1729/B9",
        "phaseset-run-v1/BASE_QUALIFICATION/1729/08",
        "phaseset-run-v1/RESIDUAL_TRAIN/1/08",
        "phasepair-run-v2/BASE_TRAIN/1729/B0",
    ),
)
def test_run_id_parser_rejects_out_of_census_identities(run_id: str) -> None:
    with pytest.raises(experiments.ExperimentContractError):
        experiments.parse_run_id(run_id)


def test_hypothesis_and_holm_census_is_exact() -> None:
    assert experiments.HYPOTHESIS_COMPARISONS == (
        ("H1", "08", "00"),
        ("H2", "08", "01"),
        ("H3", "08", "02"),
        ("H4", "08", "03"),
        ("H5", "08", "04"),
        ("H6", "08", "05"),
        ("H7", "08", "06"),
        ("H8", "08", "07"),
    )
    payload = experiments.canonical_experiment_matrix_bytes()
    assert b'"holm_family":["H2","H3","H4","H5","H6","H7","H8"]' in payload
    assert b'"qualified_base_retrained":false' in payload
    assert b'"server_endpoint"' not in payload


def _base_scores(
    *,
    score_by_system: dict[str, tuple[int, int, int]] | None = None,
    parameter_count: dict[str, int] | None = None,
    latency_ns: dict[str, int] | None = None,
) -> tuple[experiments.BaseScore, ...]:
    scores = score_by_system or {
        "B0": (60, 62, 58),
        "B1": (70, 71, 69),
        "B2": (65, 66, 64),
    }
    parameters = parameter_count or {"B0": 10_000, "B1": 11_000, "B2": 12_000}
    latencies = latency_ns or {"B0": 1_000, "B1": 1_100, "B2": 1_200}
    rows: list[experiments.BaseScore] = []
    for index, run_id in enumerate(experiments.base_run_ids(), start=1):
        _, seed, system_id = experiments.parse_run_id(run_id)
        rows.append(
            experiments.BaseScore(
                run_id=run_id,
                bidirectional_r1_numerator=scores[system_id][experiments.SEEDS.index(seed)],
                bidirectional_r1_denominator=100,
                parameter_count=parameters[system_id],
                frozen_runtime_latency_ns=latencies[system_id],
                terminal_sha256=f"{index:064x}",
            )
        )
    return tuple(rows)


def test_complete_nine_rows_always_select_one_bound_winner() -> None:
    result = experiments.qualify_base(_base_scores())
    assert result.winner_system_id == "B1"
    assert result.winner_system_name == "SetPMA"
    assert result.status == "WINNER_SELECTED_FROM_COMPLETE_NINE_ROWS"
    assert len(result.completion_sha256s) == 9
    assert len(result.winner_terminal_sha256s) == 3
    assert not hasattr(result, "verdict")
    raw = experiments.canonical_base_qualification_bytes(result)
    assert b'"winner_system_id":"B1"' in raw
    assert b'"system_resource_rows"' in raw
    assert b'"authority":0' in raw


def test_base_tie_breaks_fewer_parameters_then_latency_then_smaller_id() -> None:
    tied_scores = {system_id: (70, 70, 70) for system_id, _ in experiments.BASE_SYSTEMS}
    fewer = experiments.qualify_base(
        _base_scores(
            score_by_system=tied_scores,
            parameter_count={"B0": 10_000, "B1": 9_000, "B2": 11_000},
        )
    )
    assert fewer.winner_system_id == "B1"

    lower_latency = experiments.qualify_base(
        _base_scores(
            score_by_system=tied_scores,
            parameter_count={"B0": 10_000, "B1": 10_000, "B2": 10_000},
            latency_ns={"B0": 1_000, "B1": 1_100, "B2": 900},
        )
    )
    assert lower_latency.winner_system_id == "B2"

    smaller_id = experiments.qualify_base(
        _base_scores(
            score_by_system=tied_scores,
            parameter_count={"B0": 10_000, "B1": 10_000, "B2": 10_000},
            latency_ns={"B0": 1_000, "B1": 1_000, "B2": 1_000},
        )
    )
    assert smaller_id.winner_system_id == "B0"


def test_base_qualification_requires_all_rows_and_stable_system_metadata() -> None:
    with pytest.raises(experiments.ExperimentContractError, match="all nine"):
        experiments.qualify_base(_base_scores()[:-1])

    rows = list(_base_scores())
    rows[1] = replace(rows[1], parameter_count=rows[1].parameter_count + 1)
    with pytest.raises(experiments.ExperimentContractError, match="agree across seeds"):
        experiments.qualify_base(tuple(rows))


def test_qualification_serializer_rejects_rebound_winner_terminals() -> None:
    result = experiments.qualify_base(_base_scores())
    forged = replace(
        result,
        winner_terminal_sha256s=tuple(reversed(result.winner_terminal_sha256s)),
    )
    with pytest.raises(experiments.ExperimentContractError, match="winner terminals"):
        experiments.canonical_base_qualification_bytes(forged)

    wrong_winner = replace(
        result,
        system_mean_rows=(
            ("B0", 1, 1),
            result.system_mean_rows[1],
            result.system_mean_rows[2],
        ),
    )
    with pytest.raises(experiments.ExperimentContractError, match="tie-break"):
        experiments.canonical_base_qualification_bytes(wrong_winner)


def test_residual_controls_are_within_inclusive_one_percent_of_full() -> None:
    counts = {f"{index:02d}": 9_900 for index in range(1, 8)}
    counts["01"] = 10_100
    counts["08"] = 10_000
    assert dict(experiments.validate_control_parameter_counts(counts)) == counts

    outside = dict(counts)
    outside["07"] = 10_101
    with pytest.raises(experiments.ExperimentContractError, match=r"\+/-1%"):
        experiments.validate_control_parameter_counts(outside)
