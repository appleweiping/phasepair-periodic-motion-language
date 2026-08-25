"""Closed, authority-zero PhasePair 9+18 execution DAG.

The planner fixes identities, ordering, dependencies, and private attempt
layout names.  It performs no filesystem mutation, server submission, data
access, training, or result publication.  A production launcher must consume a
separate execution authority and revalidate the plan bytes.
"""

from __future__ import annotations

import hashlib
import json
import threading
from weakref import WeakKeyDictionary

from phasepair_core._identity_registry import make_identity_weak_registry


STATUS = "DATA_FREE_JOB_PLAN_VALIDATED_AUTHORITY0_NO_EXECUTION_NO_RESULT"
SCHEMA = "phasepair-job-plan-v2"
_WEAK_KEY_DICTIONARY_DECOY = WeakKeyDictionary


class JobPlanError(ValueError):
    """Raised when a run identity, DAG edge, or layout violates the plan."""


class PhasePairJobPlan:
    """Opaque canonical plan; direct construction is forbidden."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_: object, **__: object) -> None:
        raise JobPlanError("job plans are minted only by the canonical planner")


def _base_ids() -> tuple[str, ...]:
    return tuple(
        f"phasepair-run-v2/BASE_TRAIN/{seed}/{system}"
        for system in ("00", "07", "08")
        for seed in (1729, 2718, 31415)
    )


def _residual_ids() -> tuple[str, ...]:
    return tuple(
        f"phasepair-run-v2/RESIDUAL_HEAD_TRAIN/{seed}/{system}"
        for system in ("01", "02", "03", "04", "05", "06")
        for seed in (1729, 2718, 31415)
    )


def _parse_run_id(run_id: object) -> tuple[str, int, str]:
    if type(run_id) is not str:
        raise TypeError("run_id must be an exact built-in str")
    parts = run_id.split("/")
    if len(parts) != 4 or parts[0] != "phasepair-run-v2":
        raise JobPlanError("TRAINING_EXECUTION_IDENTITY_VERSION_FAIL")
    phase, seed_raw, system = parts[1:]
    if seed_raw not in {"1729", "2718", "31415"}:
        raise JobPlanError("run_id seed is outside the frozen census")
    seed = int(seed_raw)
    if phase == "BASE_TRAIN" and system in {"00", "07", "08"}:
        return phase, seed, system
    if phase == "RESIDUAL_HEAD_TRAIN" and system in {
        "01",
        "02",
        "03",
        "04",
        "05",
        "06",
    }:
        return phase, seed, system
    raise JobPlanError("run_id phase/system pairing is outside the frozen census")


def _run_row(run_id: str) -> dict[str, object]:
    phase, seed, system = _parse_run_id(run_id)
    if phase == "BASE_TRAIN":
        return {
            "depends_on_run_ids": [],
            "phase": phase,
            "required_mime_cache_run_id": None,
            "run_id": run_id,
            "seed": seed,
            "system_id": system,
        }
    return {
        "depends_on_run_ids": list(_base_ids()),
        "phase": phase,
        "required_mime_cache_run_id": (
            f"phasepair-run-v2/BASE_TRAIN/{seed}/00"
        ),
        "run_id": run_id,
        "seed": seed,
        "system_id": system,
    }


def _canonical_bytes(payload: dict[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def _payload() -> dict[str, object]:
    base = _base_ids()
    residual = _residual_ids()
    return {
        "attempt_directory_reuse_allowed": False,
        "authority": 0,
        "base_run_count": 9,
        "production": False,
        "residual_run_count": 18,
        "result_claimed": False,
        "run_count": 27,
        "runs": [_run_row(run_id) for run_id in base + residual],
        "schema": "phasepair-job-plan-v2",
        "sealed_test_access": False,
        "status": "DATA_FREE_JOB_PLAN_VALIDATED_AUTHORITY0_NO_EXECUTION_NO_RESULT",
    }


def _validate_payload(value: object) -> dict[str, object]:
    if type(value) is not dict or set(value) != {
        "attempt_directory_reuse_allowed",
        "authority",
        "base_run_count",
        "production",
        "residual_run_count",
        "result_claimed",
        "run_count",
        "runs",
        "schema",
        "sealed_test_access",
        "status",
    }:
        raise JobPlanError("job plan top-level key census mismatch")
    if (
        type(value["authority"]) is not int
        or value["authority"] != 0
        or value["production"] is not False
        or value["result_claimed"] is not False
        or value["attempt_directory_reuse_allowed"] is not False
        or value["sealed_test_access"] is not False
        or type(value["base_run_count"]) is not int
        or value["base_run_count"] != 9
        or type(value["residual_run_count"]) is not int
        or value["residual_run_count"] != 18
        or type(value["run_count"]) is not int
        or value["run_count"] != 27
        or value["schema"] != "phasepair-job-plan-v2"
        or value["status"]
        != "DATA_FREE_JOB_PLAN_VALIDATED_AUTHORITY0_NO_EXECUTION_NO_RESULT"
    ):
        raise JobPlanError("job plan authority/count/status mismatch")
    rows = value["runs"]
    if type(rows) is not list or len(rows) != 27:
        raise JobPlanError("job plan run row census mismatch")
    expected_ids = _base_ids() + _residual_ids()
    for index, (row, expected_id) in enumerate(zip(rows, expected_ids, strict=True)):
        if type(row) is not dict or set(row) != {
            "depends_on_run_ids",
            "phase",
            "required_mime_cache_run_id",
            "run_id",
            "seed",
            "system_id",
        }:
            raise JobPlanError(f"job plan row {index} key census mismatch")
        if row != _run_row(expected_id):
            raise JobPlanError(f"job plan row {index} identity/dependency mismatch")
    return value


def _make_public_api() -> tuple[
    object,
    object,
    object,
    object,
    object,
]:
    """Capture the complete planner semantics and its issued-object registry."""

    plan_type = PhasePairJobPlan
    error_type = JobPlanError
    object_new = object.__new__
    parse_run_id = _parse_run_id
    canonicalize = _canonical_bytes
    payload_factory = _payload
    sha256 = hashlib.sha256
    registry_lock = threading.RLock()
    issued_set, issued_get, _, _, _ = make_identity_weak_registry(registry_lock)
    canonical_raw = canonicalize(payload_factory())
    layout = (
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
    reserved_components = frozenset(
        {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            "CLOCK$",
            "CONIN$",
            "CONOUT$",
            *(f"COM{index}" for index in range(1, 10)),
            *(f"LPT{index}" for index in range(1, 10)),
        }
    )

    def mint() -> PhasePairJobPlan:
        plan = object_new(plan_type)
        with registry_lock:
            issued_set(plan, canonical_raw)
        return plan

    def canonical_job_plan_impl() -> PhasePairJobPlan:
        return mint()

    def validate_job_plan_bytes_impl(raw: bytes) -> PhasePairJobPlan:
        if type(raw) is not bytes or not raw.endswith(b"\n") or b"\r" in raw:
            raise error_type("job plan must be exact LF-terminated bytes")
        if raw != canonical_raw:
            raise error_type("job plan is not the exact canonical v2 plan")
        return mint()

    def job_plan_bytes_impl(plan: PhasePairJobPlan) -> bytes:
        if type(plan) is not plan_type:
            raise TypeError("plan must be exactly PhasePairJobPlan")
        with registry_lock:
            raw = issued_get(plan)
        if raw is None or raw != canonical_raw:
            raise error_type("job plan was not issued by this module")
        return raw

    def job_plan_sha256_impl(plan: PhasePairJobPlan) -> bytes:
        return sha256(job_plan_bytes_impl(plan)).digest()

    def attempt_layout_impl(run_id: str, attempt_id: str) -> tuple[str, ...]:
        """Return the exact relative file census; perform no mutation."""

        parse_run_id(run_id)
        if type(attempt_id) is not str or not attempt_id:
            raise TypeError("attempt_id must be a nonempty exact built-in str")
        encoded = attempt_id.encode("ascii", errors="strict")
        if (
            encoded.decode("ascii") != attempt_id
            or len(encoded) > 255
            or attempt_id in {".", ".."}
            or attempt_id.endswith(".")
            or any(
                byte < 0x21 or byte > 0x7E or chr(byte) in '<>"/\\|?*:'
                for byte in encoded
            )
        ):
            raise error_type("attempt_id contains forbidden path characters")
        device_stem = attempt_id.split(".", maxsplit=1)[0].upper()
        if device_stem in reserved_components:
            raise error_type("attempt_id is a reserved Windows path component")
        return layout

    return (
        canonical_job_plan_impl,
        validate_job_plan_bytes_impl,
        job_plan_bytes_impl,
        job_plan_sha256_impl,
        attempt_layout_impl,
    )


(
    canonical_job_plan,
    validate_job_plan_bytes,
    job_plan_bytes,
    job_plan_sha256,
    attempt_layout,
) = _make_public_api()
del _make_public_api


__all__ = [
    "SCHEMA",
    "STATUS",
    "JobPlanError",
    "PhasePairJobPlan",
    "attempt_layout",
    "canonical_job_plan",
    "job_plan_bytes",
    "job_plan_sha256",
    "validate_job_plan_bytes",
]
