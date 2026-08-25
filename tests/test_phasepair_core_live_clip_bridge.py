from __future__ import annotations

import copy
import gc
import inspect
import json
import os
from pathlib import Path
import sys
from types import MappingProxyType
import weakref

import pytest

from phasepair_core import live_clip_bridge as bridge


def _path_from_environment(name: str) -> Path | None:
    value = os.environ.get(name)
    return Path(value) if value else None


PRIVATE_WHEELHOUSE = _path_from_environment("PHASEPAIR_CLIP_PRIVATE_WHEELHOUSE")
PRIVATE_SNAPSHOT = _path_from_environment("PHASEPAIR_CLIP_PRIVATE_SNAPSHOT")
RIGHTS_RECEIPT = _path_from_environment("PHASEPAIR_CLIP_RIGHTS_RECEIPT")
EXACT_PYTHON = (
    PRIVATE_WHEELHOUSE / "preflight-venv" / "Scripts" / "python.exe"
    if PRIVATE_WHEELHOUSE is not None
    else None
)
REAL_RUNTIME_CONFIGURED = (
    PRIVATE_WHEELHOUSE is not None
    and PRIVATE_SNAPSHOT is not None
    and RIGHTS_RECEIPT is not None
    and EXACT_PYTHON is not None
    and EXACT_PYTHON.exists()
    and Path(sys.executable).resolve(strict=False)
    == EXACT_PYTHON.resolve(strict=False)
)


def test_public_surface_is_path_only_and_output_construction_is_closed() -> None:
    resolve = inspect.signature(bridge.resolve_local_clip_text_candidate)
    assert tuple(resolve.parameters) == (
        "snapshot_root",
        "wheelhouse_root",
        "rights_receipt",
    )
    forbidden = {"model", "state_dict", "rows", "parameter_rows", "loader"}
    assert forbidden.isdisjoint(resolve.parameters)

    prepare = inspect.signature(bridge.prepare_local_optimizer_candidate)
    assert tuple(prepare.parameters) == (
        "lease",
        "architecture",
        "motion_model",
        "project_weight",
        "project_bias",
        "logit_scale",
        "gate_references",
    )
    assert "resolved_text_rows" not in prepare.parameters

    with pytest.raises(TypeError):
        bridge.VerifiedLocalClipTextLease()
    with pytest.raises(TypeError):
        bridge.OptimizerCandidateAssessment()
    forged_assessment = object.__new__(bridge.OptimizerCandidateAssessment)
    with pytest.raises(bridge.LiveClipBridgeHold):
        bridge.canonical_optimizer_candidate_bytes(forged_assessment)


def test_forge_copy_subclass_and_nonpath_fail_before_adamw() -> None:
    forged = object.__new__(bridge.VerifiedLocalClipTextLease)

    class LeaseSubclass(bridge.VerifiedLocalClipTextLease):
        pass

    for value in (forged, copy.copy(forged), object.__new__(LeaseSubclass), object()):
        with pytest.raises(bridge.LiveClipBridgeHold):
            bridge.canonical_local_clip_candidate_receipt(value)
        with pytest.raises(bridge.LiveClipBridgeHold):
            bridge.local_clip_lease_state(value)

    with pytest.raises(TypeError):
        bridge.resolve_local_clip_text_candidate(
            "snapshot", Path("wheelhouse"), Path("rights")  # type: ignore[arg-type]
        )
    assert bridge._adamw_constructor_count_for_tests() == 0


def test_missing_artifacts_hold_without_importing_transformers_or_adamw(
    tmp_path: Path,
) -> None:
    before = set(sys.modules)
    with pytest.raises(bridge.LiveClipBridgeHold):
        bridge.resolve_local_clip_text_candidate(
            tmp_path / "missing-snapshot",
            tmp_path / "missing-wheelhouse",
            tmp_path / "missing-rights",
        )
    newly_loaded = set(sys.modules) - before
    assert "transformers" not in newly_loaded
    assert bridge._adamw_constructor_count_for_tests() == 0
    assert bridge._live_lease_count_for_tests() == 0


def test_helper_constant_and_dependency_rebinds_fail_closed(tmp_path: Path) -> None:
    def attempt() -> bridge.LiveClipBridgeHold:
        with pytest.raises(bridge.LiveClipBridgeHold) as caught:
            bridge.resolve_local_clip_text_candidate(
                tmp_path / "missing-snapshot",
                tmp_path / "missing-wheelhouse",
                tmp_path / "missing-rights",
            )
        return caught.value

    original_helper = bridge._safe_regular_file
    try:
        bridge._safe_regular_file = lambda *_args, **_kwargs: b""
        assert attempt().code == "HOLD_LIVE_CLIP_MODULE_REBOUND"
    finally:
        bridge._safe_regular_file = original_helper

    original_hidden_size = bridge._EXPECTED_CONFIG["hidden_size"]
    try:
        bridge._EXPECTED_CONFIG["hidden_size"] = 999
        assert attempt().code == "HOLD_LIVE_CLIP_MODULE_REBOUND"
    finally:
        bridge._EXPECTED_CONFIG["hidden_size"] = original_hidden_size

    original_partition_validator = bridge.contracts.validate_optimizer_partition
    try:
        bridge.contracts.validate_optimizer_partition = lambda *_args, **_kwargs: None
        assert attempt().code == "HOLD_LIVE_CLIP_DEPENDENCY_REBOUND"
    finally:
        bridge.contracts.validate_optimizer_partition = original_partition_validator

    assert bridge._adamw_constructor_count_for_tests() == 0
    assert bridge._live_lease_count_for_tests() == 0


def test_module_has_no_optimizer_constructor_or_public_artifact_injection() -> None:
    source = inspect.getsource(bridge)
    assert "torch.optim.AdamW(" not in source
    assert "optimizer_constructor(" not in source
    assert bridge._adamw_constructor_count_for_tests() == 0
    assert bridge.__all__ == [
        "OptimizerCandidateAssessment",
        "VerifiedLocalClipTextLease",
        "canonical_local_clip_candidate_receipt",
        "canonical_optimizer_candidate_bytes",
        "local_clip_lease_state",
        "prepare_local_optimizer_candidate",
        "resolve_local_clip_text_candidate",
    ]


def test_tensor_boundary_helpers_ignore_instance_method_shadows() -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(
        torch.arange(6, dtype=torch.float32).reshape(2, 3)
    )
    expected_bytes = bridge._tensor_bytes(parameter)
    expected_key = bridge._storage_key(parameter)
    expected_digest = bridge._binding_digest((("probe.weight", parameter),))
    storage = torch.Tensor.untyped_storage(parameter)

    def bomb(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shadowed Tensor or storage instance method was called")

    for name in (
        "detach",
        "clone",
        "to",
        "contiguous",
        "numpy",
        "is_contiguous",
        "untyped_storage",
        "storage_offset",
        "numel",
        "element_size",
        "all",
        "item",
        "isfinite",
    ):
        setattr(parameter, name, bomb)
    storage.data_ptr = bomb
    storage.nbytes = bomb

    assert bridge._tensor_bytes(parameter) == expected_bytes
    assert bridge._storage_key(parameter) == expected_key
    assert bridge._binding_digest((("probe.weight", parameter),)) == expected_digest
    assert bridge._exact_parameter(
        parameter,
        shape=(2, 3),
        torch=torch,
        label="PROBE",
    ) is parameter


def test_binding_and_exact_parameter_cannot_be_fooled_by_storage_decoys() -> None:
    torch = pytest.importorskip("torch")
    owner = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    alias = torch.nn.Parameter(torch.Tensor.detach(owner))
    decoy = torch.nn.Parameter(torch.zeros(4, dtype=torch.float32))
    alias.untyped_storage = lambda: torch.Tensor.untyped_storage(decoy)
    alias.storage_offset = lambda: 0
    alias.numel = lambda: 4
    alias.element_size = lambda: 4
    alias.is_contiguous = lambda: True
    with pytest.raises(
        bridge.LiveClipBridgeHold,
        match="HOLD_FULL_PARAMETER_ALIAS|HOLD_FULL_PARAMETER_STORAGE_OVERLAP",
    ):
        bridge._binding_digest((('owner', owner), ('alias', alias)))

    base = torch.zeros((3, 4), dtype=torch.float32)
    noncontiguous = torch.nn.Parameter(base[:, ::2])
    noncontiguous.untyped_storage = lambda: torch.Tensor.untyped_storage(decoy)
    noncontiguous.storage_offset = lambda: 0
    noncontiguous.numel = lambda: 6
    noncontiguous.element_size = lambda: 4
    noncontiguous.is_contiguous = lambda: True
    with pytest.raises(
        bridge.LiveClipBridgeHold,
        match="HOLD_PROBE_PARAMETER_INVARIANT",
    ):
        bridge._exact_parameter(
            noncontiguous,
            shape=(3, 2),
            torch=torch,
            label="PROBE",
        )


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), float("-inf")])
def test_parameter_boundaries_reject_real_nonfinite_values_despite_decoys(
    bad_value: float,
) -> None:
    torch = pytest.importorskip("torch")
    parameter = torch.nn.Parameter(
        torch.tensor([bad_value], dtype=torch.float32)
    )
    parameter.detach = lambda: torch.zeros((1,), dtype=torch.float32)
    parameter.all = lambda *_args, **_kwargs: torch.tensor(True)
    parameter.item = lambda: 0.0
    parameter.isfinite = lambda: torch.ones((1,), dtype=torch.bool)

    with pytest.raises(
        bridge.LiveClipBridgeHold,
        match="HOLD_PROBE_PARAMETER_INVARIANT",
    ):
        bridge._exact_parameter(
            parameter,
            shape=(1,),
            torch=torch,
            label="PROBE",
        )
    with pytest.raises(
        bridge.LiveClipBridgeHold,
        match="HOLD_PARAMETER_ROW_PROVENANCE",
    ):
        bridge._parameter_row_facts(
            parameter,
            shape=(1,),
            numel=1,
        )
    with pytest.raises(
        bridge.LiveClipBridgeHold,
        match="HOLD_PARAMETER_ROW_PROVENANCE",
    ):
        bridge._binding_digest((("probe", parameter),))


def test_contract_constants_are_immutable_observations() -> None:
    observed = MappingProxyType(
        {
            "authority": bridge.AUTHORITY,
            "production": bridge.PRODUCTION,
            "status": bridge.STATUS,
            "training_authorized": bridge.TRAINING_AUTHORIZED,
        }
    )
    assert dict(observed) == {
        "authority": 0,
        "production": False,
        "status": "VERIFIED_LOCAL_CPU_CANDIDATE_AUTHORITY0",
        "training_authorized": False,
    }


@pytest.mark.skipif(
    not REAL_RUNTIME_CONFIGURED,
    reason=(
        "requires explicit private CLIP path variables and the exact Py3.12 "
        "CPU wheelhouse interpreter"
    ),
)
def test_real_offline_candidate_two_group_join_is_one_shot_and_authority_zero() -> None:
    import torch

    from phasepair_core import initialization, readiness, torch_models

    assert PRIVATE_SNAPSHOT is not None
    assert PRIVATE_WHEELHOUSE is not None
    assert RIGHTS_RECEIPT is not None

    baseline = bridge._live_lease_count_for_tests()
    rng_before = torch.get_rng_state().clone()
    thread_count_before = torch.get_num_threads()
    lease = bridge.resolve_local_clip_text_candidate(
        PRIVATE_SNAPSHOT,
        PRIVATE_WHEELHOUSE,
        RIGHTS_RECEIPT,
    )
    assert torch.equal(rng_before, torch.get_rng_state())
    assert torch.get_num_threads() == thread_count_before
    assert bridge.local_clip_lease_state(lease) == bridge.ISSUED
    assert bridge._live_lease_count_for_tests() == baseline + 1

    candidate = json.loads(bridge.canonical_local_clip_candidate_receipt(lease))
    assert candidate["authority"] == 0
    assert candidate["production"] is False
    assert candidate["training_authorized"] is False
    assert candidate["text_trainable_tensor_count"] == 196
    assert candidate["text_trainable_numel"] == 63_165_952
    assert candidate["text_decay_tensor_count"] == 74
    assert candidate["text_no_decay_tensor_count"] == 122
    assert candidate["vision_parameter_count"] == 0
    assert candidate["adamw_constructor_count"] == 0

    motion = torch_models.build_motion_encoder("early", device="cpu")
    initialization.initialize_motion_model(motion, seed=1729)
    project_weight = torch.nn.Parameter(torch.zeros((512, 512), dtype=torch.float32))
    project_bias = torch.nn.Parameter(torch.zeros((512,), dtype=torch.float32))
    logit_scale = torch.nn.Parameter(torch.zeros((1,), dtype=torch.float32))
    gates = readiness.TrainingGateReferences(b"\x00" * 32)
    assessment = bridge.prepare_local_optimizer_candidate(
        lease,
        architecture="early",
        motion_model=motion,
        project_weight=project_weight,
        project_bias=project_bias,
        logit_scale=logit_scale,
        gate_references=gates,
    )
    assert assessment.status == "HOLD_NO_TRAINING_AUTHORITY_ISSUER"
    assert assessment.authority == 0
    assert assessment.full_tensor_count == 268
    assert assessment.full_numel == 76_442_625
    assert assessment.decay_tensor_count == 101
    assert assessment.decay_numel == 76_333_056
    assert assessment.no_decay_tensor_count == 167
    assert assessment.no_decay_numel == 109_569
    assert assessment.adamw_constructor_count == 0
    optimizer_receipt = json.loads(
        bridge.canonical_optimizer_candidate_bytes(assessment)
    )
    assert optimizer_receipt["status"] == "HOLD_NO_TRAINING_AUTHORITY_ISSUER"
    assert optimizer_receipt["adamw_constructor_count"] == 0
    assert bridge.local_clip_lease_state(lease) == bridge.BURNED
    with pytest.raises(bridge.LiveClipBridgeHold):
        bridge.canonical_local_clip_candidate_receipt(lease)

    reference = weakref.ref(lease)
    del lease
    gc.collect()
    assert reference() is None
    assert bridge._live_lease_count_for_tests() == baseline
    assert bridge._adamw_constructor_count_for_tests() == 0
