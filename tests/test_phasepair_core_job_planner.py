from __future__ import annotations

import hashlib
import json

import pytest

from phasepair_core import job_planner


def test_exact_9_plus_18_dag_and_canonical_roundtrip() -> None:
    plan = job_planner.canonical_job_plan()
    raw = job_planner.job_plan_bytes(plan)
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert job_planner.job_plan_sha256(plan) == hashlib.sha256(raw).digest()
    payload = json.loads(raw)
    assert payload["base_run_count"] == 9
    assert payload["residual_run_count"] == 18
    assert payload["run_count"] == 27
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["result_claimed"] is False
    assert payload["sealed_test_access"] is False
    rows = payload["runs"]
    assert [row["run_id"] for row in rows[:9]] == [
        f"phasepair-run-v2/BASE_TRAIN/{seed}/{system}"
        for system in ("00", "07", "08")
        for seed in (1729, 2718, 31415)
    ]
    assert [row["run_id"] for row in rows[9:]] == [
        f"phasepair-run-v2/RESIDUAL_HEAD_TRAIN/{seed}/{system}"
        for system in ("01", "02", "03", "04", "05", "06")
        for seed in (1729, 2718, 31415)
    ]
    assert all(row["depends_on_run_ids"] == [] for row in rows[:9])
    assert all(len(row["depends_on_run_ids"]) == 9 for row in rows[9:])
    for row in rows[9:]:
        assert row["required_mime_cache_run_id"] == (
            f"phasepair-run-v2/BASE_TRAIN/{row['seed']}/00"
        )
    rebuilt = job_planner.validate_job_plan_bytes(raw)
    assert job_planner.job_plan_bytes(rebuilt) == raw


def test_every_residual_depends_on_all_base_runs_and_same_seed_mime_cache() -> None:
    payload = json.loads(job_planner.job_plan_bytes(job_planner.canonical_job_plan()))
    base_ids = tuple(row["run_id"] for row in payload["runs"][:9])
    assert len(base_ids) == 9 and len(set(base_ids)) == 9
    for row in payload["runs"][9:]:
        assert tuple(row["depends_on_run_ids"]) == base_ids
        assert row["required_mime_cache_run_id"] in base_ids[:3]


def test_legacy_merge_reorder_extra_and_authority_mutants_are_rejected() -> None:
    raw = job_planner.job_plan_bytes(job_planner.canonical_job_plan())
    mutants: list[dict[str, object]] = []
    for key, value in (
        ("authority", 1),
        ("production", True),
        ("result_claimed", True),
        ("sealed_test_access", True),
        ("run_count", 26),
        ("schema", "phasepair-job-plan-v1"),
    ):
        mutant = json.loads(raw)
        mutant[key] = value
        mutants.append(mutant)
    missing = json.loads(raw)
    missing["runs"].pop()
    mutants.append(missing)
    reordered = json.loads(raw)
    reordered["runs"][0], reordered["runs"][1] = (
        reordered["runs"][1],
        reordered["runs"][0],
    )
    mutants.append(reordered)
    merged = json.loads(raw)
    merged["runs"][9]["system_id"] = "02+03"
    mutants.append(merged)
    for mutant in mutants:
        mutant_raw = (
            json.dumps(mutant, sort_keys=True, separators=(",", ":")).encode("ascii")
            + b"\n"
        )
        with pytest.raises(job_planner.JobPlanError):
            job_planner.validate_job_plan_bytes(mutant_raw)


def test_attempt_layout_is_exact_and_has_no_side_effect() -> None:
    expected = (
        "run_input.canonical.json",
        "source_manifest.json",
        "environment_manifest.json",
        "data_manifest.json",
        "model_manifest.json",
        "optimizer_manifest.json",
        "command.txt",
        "stdout.log",
        "stderr.log",
        "heartbeat.jsonl",
        "metrics.jsonl",
        "checkpoints/",
        "validation/",
        "terminal_status.json",
        "execution_receipt.json",
        "failure_receipt.json",
    )
    assert job_planner.attempt_layout(
        "phasepair-run-v2/BASE_TRAIN/1729/00",
        "attempt-0001",
    ) == expected
    assert job_planner.attempt_layout(
        "phasepair-run-v2/RESIDUAL_HEAD_TRAIN/31415/06",
        "attempt-0002",
    ) == expected


@pytest.mark.parametrize(
    ("run_id", "attempt_id"),
    [
        ("phasepair-run-v1/BASE_TRAIN/1729/00", "a"),
        ("phasepair-run-v2/BASE_TRAIN/1729/06", "a"),
        ("phasepair-run-v2/RESIDUAL_HEAD_TRAIN/1729/00", "a"),
        ("phasepair-run-v2/BASE_TRAIN/1/00", "a"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "a/b"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", ""),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "."),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", ".."),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "attempt."),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "CON"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "nul.txt"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "COM9.log"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "LPT1"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "a?b"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "a|b"),
        ("phasepair-run-v2/BASE_TRAIN/1729/00", "a" * 256),
    ],
)
def test_invalid_identity_or_attempt_fails_closed(run_id: object, attempt_id: object) -> None:
    with pytest.raises((TypeError, UnicodeError, job_planner.JobPlanError)):
        job_planner.attempt_layout(run_id, attempt_id)


def test_direct_plan_forgery_is_rejected() -> None:
    with pytest.raises(job_planner.JobPlanError):
        job_planner.PhasePairJobPlan()
    forged = object.__new__(job_planner.PhasePairJobPlan)
    with pytest.raises(job_planner.JobPlanError):
        job_planner.job_plan_bytes(forged)


def test_plan_registry_uses_identity_not_mutable_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = job_planner.canonical_job_plan()
    expected = job_planner.job_plan_bytes(plan)
    monkeypatch.setattr(
        job_planner.PhasePairJobPlan,
        "__eq__",
        lambda _self, _other: True,
        raising=False,
    )
    monkeypatch.setattr(
        job_planner.PhasePairJobPlan,
        "__hash__",
        lambda _self: 1,
        raising=False,
    )
    forged = object.__new__(job_planner.PhasePairJobPlan)
    with pytest.raises(job_planner.JobPlanError):
        job_planner.job_plan_bytes(forged)
    assert job_planner.job_plan_bytes(plan) == expected


def test_public_constant_rebind_does_not_change_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    baseline = job_planner.job_plan_bytes(job_planner.canonical_job_plan())
    monkeypatch.setattr(job_planner, "STATUS", "FORGED")
    monkeypatch.setattr(job_planner, "SCHEMA", "forged")
    assert job_planner.job_plan_bytes(job_planner.canonical_job_plan()) == baseline


def test_private_dependency_and_class_rebind_does_not_change_api(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_plan = job_planner.canonical_job_plan()
    baseline = job_planner.job_plan_bytes(baseline_plan)
    baseline_sha = job_planner.job_plan_sha256(baseline_plan)
    issued_type = type(baseline_plan)
    issued_error = job_planner.JobPlanError

    class Bomb:
        def __init__(self, *_: object, **__: object) -> None:
            raise AssertionError("rebound dependency was read")

        def __call__(self, *_: object, **__: object) -> object:
            raise AssertionError("rebound dependency was called")

    bomb = Bomb.__new__(Bomb)
    for name in (
        "_base_ids",
        "_residual_ids",
        "_parse_run_id",
        "_run_row",
        "_canonical_bytes",
        "_payload",
        "_validate_payload",
    ):
        monkeypatch.setattr(job_planner, name, bomb)
    monkeypatch.setattr(job_planner, "PhasePairJobPlan", Bomb)
    monkeypatch.setattr(job_planner, "JobPlanError", RuntimeError)
    monkeypatch.setattr(job_planner, "WeakKeyDictionary", Bomb)
    monkeypatch.setattr(job_planner.threading, "RLock", bomb)
    monkeypatch.setattr(job_planner.json, "dumps", bomb)
    monkeypatch.setattr(job_planner.json, "loads", bomb)
    monkeypatch.setattr(job_planner.hashlib, "sha256", bomb)

    replacement = job_planner.canonical_job_plan()
    assert type(replacement) is issued_type
    assert job_planner.job_plan_bytes(replacement) == baseline
    assert job_planner.job_plan_sha256(replacement) == baseline_sha
    parsed = job_planner.validate_job_plan_bytes(baseline)
    assert type(parsed) is issued_type
    assert job_planner.job_plan_bytes(parsed) == baseline
    assert job_planner.attempt_layout(
        "phasepair-run-v2/BASE_TRAIN/1729/00", "attempt-0001"
    )[0] == "run_input.canonical.json"
    forged = object.__new__(issued_type)
    with pytest.raises(issued_error):
        job_planner.job_plan_bytes(forged)
