from __future__ import annotations

import gc
import hashlib
import json
import struct
import threading
import weakref
from pathlib import Path

import pytest
import torch
from torch import nn

from phasepair_core import _identity_registry as identity_registry
from phasepair_core import checkpoint
from phasepair_core import trainer


def _model() -> nn.Linear:
    model = nn.Linear(3, 2, bias=True, dtype=torch.float32, device="cpu")
    with torch.no_grad():
        model.weight.copy_(
            torch.tensor(
                [[0.25, -0.5, 0.75], [-0.125, 0.375, 0.625]],
                dtype=torch.float32,
            )
        )
        model.bias.copy_(torch.tensor([0.125, -0.25], dtype=torch.float32))
    return model


def _admission(model: nn.Linear) -> trainer.SyntheticOptimizerAdmission:
    return trainer.issue_cpu_synthetic_admission(
        model,
        decay_names=("weight",),
        no_decay_names=("bias",),
        run_id="phasepair-synthetic-run-001",
        attempt_id="attempt-001",
        system_id="00-synthetic",
        seed=1729,
    )


def _loss(model: nn.Linear) -> torch.Tensor:
    inputs = torch.tensor(
        [[1.0, -0.5, 0.25], [-0.25, 0.75, 1.0]], dtype=torch.float32
    )
    targets = torch.tensor([[0.5, -0.25], [0.125, 0.75]], dtype=torch.float32)
    return torch.mean((model(inputs) - targets) ** 2)


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _checkpoint_bindings(
    sections: tuple[tuple[str, bytes], ...],
    *,
    global_step: int,
) -> checkpoint.CheckpointBindings:
    by_name = dict(sections)
    return checkpoint.CheckpointBindings(
        run_id="phasepair-synthetic-run-001",
        attempt_id="attempt-001",
        system_id="00-synthetic",
        seed=1729,
        epoch_index=0,
        global_step=global_step,
        source_manifest_sha256=_digest("source"),
        data_manifest_sha256=_digest("data"),
        split_manifest_sha256=_digest("split"),
        environment_manifest_sha256=_digest("environment"),
        code_manifest_sha256=_digest("code"),
        model_manifest_sha256=_digest("model-manifest"),
        text_runtime_manifest_sha256=_digest("text-runtime"),
        optimizer_manifest_sha256=_digest("optimizer-manifest"),
        sampler_state_sha256=hashlib.sha256(by_name["sampler_state"]).digest(),
        dropout_state_sha256=hashlib.sha256(by_name["dropout_state"]).digest(),
        validation_state_sha256=hashlib.sha256(
            by_name["validation_state"]
        ).digest(),
        parent_checkpoint_sha256=None,
    )


def test_real_adamw_step_is_canonical_authority_zero_and_single_use() -> None:
    model = _model()
    admission = _admission(model)
    constructors_before = trainer.cpu_synthetic_optimizer_constructor_count()
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":ISSUED")
    session = trainer.open_cpu_synthetic_training_session(admission)
    assert trainer.cpu_synthetic_optimizer_constructor_count() == constructors_before + 1
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":CONSUMED")
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    receipt = trainer.run_cpu_synthetic_optimizer_step(
        session,
        _loss(model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    raw = trainer.canonical_step_receipt_bytes(receipt)
    payload = json.loads(raw)
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["result_claimed"] is False
    assert payload["status"] == trainer.STEP_STATUS
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert trainer.step_receipt_sha256(receipt) == hashlib.sha256(raw).digest()
    assert trainer.cpu_synthetic_session_state(session) == "ACTIVE"
    assert any(
        not torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    with pytest.raises(trainer.TrainingAdmissionBurnedError):
        trainer.open_cpu_synthetic_training_session(admission)


def test_two_steps_require_exact_monotonic_position() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    trainer.run_cpu_synthetic_optimizer_step(
        session,
        _loss(model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    trainer.run_cpu_synthetic_optimizer_step(
        session,
        _loss(model),
        epoch_index=0,
        global_step=1,
        batch_position=1,
        dropout_position=1,
    )
    assert trainer.cpu_synthetic_session_state(session) == "ACTIVE"
    with pytest.raises(trainer.TrainerContractError):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            _loss(model),
            epoch_index=0,
            global_step=3,
            batch_position=2,
            dropout_position=2,
        )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"


def test_missing_gradient_rolls_back_parameter_and_optimizer_state_then_retires() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    loss = torch.sum(model.weight**2)
    with pytest.raises(trainer.TrainingTransactionError) as error:
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            loss,
            epoch_index=0,
            global_step=0,
            batch_position=0,
            dropout_position=0,
        )
    assert isinstance(error.value.__cause__, trainer.TrainerContractError)
    assert tuple(parameter.grad for parameter in model.parameters()) == (None, None)
    assert all(
        torch.equal(old, new)
        for old, new in zip(before, model.parameters(), strict=True)
    )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"
    with pytest.raises(trainer.TrainingAdmissionBurnedError):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            _loss(model),
            epoch_index=0,
            global_step=0,
            batch_position=0,
            dropout_position=0,
        )


def test_second_step_missing_gradient_restores_clean_boundary_and_retires() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    trainer.run_cpu_synthetic_optimizer_step(
        session,
        _loss(model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    assert all(parameter.grad is None for parameter in model.parameters())
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    with pytest.raises(trainer.TrainingTransactionError):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            torch.sum(model.weight**2),
            epoch_index=0,
            global_step=1,
            batch_position=1,
            dropout_position=1,
        )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, model.parameters(), strict=True)
    )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"


def test_failed_backward_restores_exact_cpu_rng_state() -> None:
    class RandomBackwardMissingBias(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx: object, weight: torch.Tensor, bias: torch.Tensor
        ) -> torch.Tensor:
            del ctx, bias
            return torch.sum(weight**2)

        @staticmethod
        def backward(ctx: object, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
            del ctx
            torch.rand(1)
            return gradient * torch.ones((2, 3), dtype=torch.float32), None

    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    before_rng = torch.get_rng_state().clone()
    before_parameters = tuple(
        parameter.detach().clone() for parameter in model.parameters()
    )
    loss = RandomBackwardMissingBias.apply(model.weight, model.bias)
    with pytest.raises(trainer.TrainingTransactionError):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            loss,
            epoch_index=0,
            global_step=0,
            batch_position=0,
            dropout_position=0,
        )
    assert torch.equal(torch.get_rng_state(), before_rng)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            before_parameters, model.parameters(), strict=True
        )
    )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"


def test_external_stale_gradient_is_rejected_before_backward() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    model.weight.grad = torch.ones_like(model.weight)
    before = tuple(parameter.detach().clone() for parameter in model.parameters())
    with pytest.raises(trainer.TrainerContractError, match="stale gradient"):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            _loss(model),
            epoch_index=0,
            global_step=0,
            batch_position=0,
            dropout_position=0,
        )
    assert model.weight.grad is not None
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(before, model.parameters(), strict=True)
    )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"


def test_admission_rejects_bad_partition_types_alias_and_post_issue_drift() -> None:
    model = _model()
    with pytest.raises(trainer.TrainerContractError):
        trainer.issue_cpu_synthetic_admission(
            model,
            decay_names=("weight",),
            no_decay_names=("weight",),
            run_id="run",
            attempt_id="attempt",
            system_id="00",
            seed=1729,
        )
    with pytest.raises(TypeError):
        trainer.issue_cpu_synthetic_admission(
            model,
            decay_names=("weight",),
            no_decay_names=("bias",),
            run_id="run",
            attempt_id="attempt",
            system_id="00",
            seed=True,
        )
    admission = _admission(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(trainer.TrainerContractError):
        trainer.open_cpu_synthetic_training_session(admission)
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":BURNED")


def test_model_parameter_and_storage_ownership_is_exclusive_and_recoverable() -> None:
    model = _model()
    first_admission = _admission(model)
    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(model)

    session = trainer.open_cpu_synthetic_training_session(first_admission)
    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(model)

    with pytest.raises(trainer.TrainerContractError, match="global_step"):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            _loss(model),
            epoch_index=0,
            global_step=1,
            batch_position=0,
            dropout_position=0,
        )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"
    replacement = _admission(model)
    assert trainer.cpu_synthetic_admission_status(replacement).endswith(":ISSUED")


def test_parameter_loss_gradient_and_storage_method_fields_do_not_rebind_step() -> None:
    model = _model()

    def parameter_bomb(*_: object, **__: object) -> object:
        raise AssertionError("parameter/storage instance method was dispatched")

    def gradient_bomb(*_: object, **__: object) -> object:
        raise AssertionError("gradient instance method was dispatched")

    def loss_bomb(label: str):
        def fail(*_: object, **__: object) -> object:
            raise AssertionError(f"loss instance method was dispatched: {label}")

        return fail

    for parameter in model.parameters():
        parameter.detach = parameter_bomb
        parameter.clone = parameter_bomb
        parameter.to = parameter_bomb
        parameter.contiguous = parameter_bomb
        parameter.numpy = parameter_bomb
        parameter.is_contiguous = parameter_bomb
        parameter.untyped_storage = parameter_bomb
        parameter.storage_offset = parameter_bomb
        parameter.numel = parameter_bomb
        parameter.element_size = parameter_bomb
        storage = torch.Tensor.untyped_storage(parameter)
        storage.data_ptr = parameter_bomb
        storage.nbytes = parameter_bomb

    session = trainer.open_cpu_synthetic_training_session(_admission(model))

    def shadow_accumulated_gradient(parameter: torch.Tensor) -> None:
        gradient = parameter.grad
        assert type(gradient) is torch.Tensor
        gradient.detach = gradient_bomb
        gradient.clone = gradient_bomb
        gradient.to = gradient_bomb
        gradient.contiguous = gradient_bomb
        gradient.numpy = gradient_bomb
        gradient.is_contiguous = gradient_bomb
        gradient.storage_offset = gradient_bomb
        gradient.numel = gradient_bomb
        gradient.element_size = gradient_bomb

    handles = [
        parameter.register_post_accumulate_grad_hook(shadow_accumulated_gradient)
        for parameter in model.parameters()
    ]
    loss = _loss(model)
    loss.detach = loss_bomb("detach")
    loss.clone = loss_bomb("clone")
    loss.to = loss_bomb("to")
    loss.contiguous = loss_bomb("contiguous")
    loss.numpy = loss_bomb("numpy")
    loss.is_contiguous = loss_bomb("is_contiguous")
    loss.storage_offset = loss_bomb("storage_offset")
    loss.numel = loss_bomb("numel")
    loss.element_size = loss_bomb("element_size")
    loss.backward = loss_bomb("backward")
    loss.item = loss_bomb("item")

    receipt = trainer.run_cpu_synthetic_optimizer_step(
        session,
        loss,
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    for handle in handles:
        handle.remove()
    assert json.loads(trainer.canonical_step_receipt_bytes(receipt))["global_step"] == 0
    assert trainer.cpu_synthetic_session_state(session) == "ACTIVE"
    assert all(parameter.grad is None for parameter in model.parameters())


def test_parameter_untyped_storage_instance_field_cannot_hide_live_alias() -> None:
    owner_model = _model()
    owner_admission = _admission(owner_model)
    alias_model = _model()
    alias_model.weight = nn.Parameter(torch.Tensor.detach(owner_model.weight))
    decoy = nn.Parameter(torch.empty_like(alias_model.weight))
    alias_model.weight.untyped_storage = decoy.untyped_storage

    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(alias_model)
    assert trainer.cpu_synthetic_admission_status(owner_admission).endswith(":ISSUED")


def test_loss_detach_instance_field_cannot_hide_nonfinite_scalar() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    loss = _loss(model) * torch.tensor(float("nan"), dtype=torch.float32)
    loss.detach = lambda: torch.zeros((), dtype=torch.float32)

    with pytest.raises(trainer.TrainerContractError, match="finite differentiable"):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            loss,
            epoch_index=0,
            global_step=0,
            batch_position=0,
            dropout_position=0,
        )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"
    assert all(parameter.grad is None for parameter in model.parameters())


def test_identity_registry_tuple_rebind_cannot_hide_live_ownership(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    admission = _admission(model)
    monkeypatch.setattr(identity_registry, "tuple", lambda *_: (), raising=False)

    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(model)

    session = trainer.open_cpu_synthetic_training_session(admission)
    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(model)
    assert trainer.cpu_synthetic_session_state(session) == "ACTIVE"

    owner_model = _model()
    owner_admission = _admission(owner_model)
    alias_model = _model()
    alias_model.weight = nn.Parameter(owner_model.weight.detach())
    assert (
        alias_model.weight.untyped_storage().data_ptr()
        == owner_model.weight.untyped_storage().data_ptr()
    )
    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(alias_model)
    assert trainer.cpu_synthetic_admission_status(owner_admission).endswith(":ISSUED")

    gc_model = _model()
    gc_admission = _admission(gc_model)
    gc_session = trainer.open_cpu_synthetic_training_session(gc_admission)
    gc_session_ref = weakref.ref(gc_session)
    del gc_session
    gc.collect()
    assert gc_session_ref() is None
    after_gc = _admission(gc_model)
    assert trainer.cpu_synthetic_admission_status(after_gc).endswith(":ISSUED")

    admission_gc_model = _model()
    admission_gc = _admission(admission_gc_model)
    admission_gc_ref = weakref.ref(admission_gc)
    del admission_gc
    gc.collect()
    assert admission_gc_ref() is None
    after_admission_gc = _admission(admission_gc_model)
    assert trainer.cpu_synthetic_admission_status(after_admission_gc).endswith(
        ":ISSUED"
    )


def test_burned_admission_releases_ownership_while_token_is_live() -> None:
    model = _model()
    original = tuple(parameter.detach().clone() for parameter in model.parameters())
    admission = _admission(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    with pytest.raises(trainer.TrainerContractError):
        trainer.open_cpu_synthetic_training_session(admission)
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":BURNED")
    with torch.no_grad():
        for parameter, expected in zip(model.parameters(), original, strict=True):
            parameter.copy_(expected)
    replacement = _admission(model)
    assert trainer.cpu_synthetic_admission_status(replacement).endswith(":ISSUED")


def test_storage_impl_rebind_and_current_storage_conflict_fail_closed() -> None:
    same_pointer_model = _model()
    same_pointer_session = trainer.open_cpu_synthetic_training_session(
        _admission(same_pointer_model)
    )
    parameter = same_pointer_model.weight
    old_storage_id = int(parameter.untyped_storage()._cdata)
    old_pointer = parameter.untyped_storage().data_ptr()
    alias = torch.frombuffer(
        memoryview(parameter.detach().numpy()), dtype=torch.float32
    ).reshape_as(parameter)
    assert alias.untyped_storage().data_ptr() == old_pointer
    assert int(alias.untyped_storage()._cdata) != old_storage_id
    with torch.no_grad():
        parameter.data = alias
    with pytest.raises(trainer.TrainerContractError, match="storage binding drift"):
        trainer.run_cpu_synthetic_optimizer_step(
            same_pointer_session,
            _loss(same_pointer_model),
            epoch_index=0,
            global_step=0,
            batch_position=0,
            dropout_position=0,
        )
    assert trainer.cpu_synthetic_session_state(same_pointer_session) == "RETIRED"
    replacement = _admission(same_pointer_model)
    assert trainer.cpu_synthetic_admission_status(replacement).endswith(":ISSUED")

    rebound_model = _model()
    rebound_session = trainer.open_cpu_synthetic_training_session(
        _admission(rebound_model)
    )
    current_storage_owner = _model()
    with torch.no_grad():
        rebound_model.weight.data = current_storage_owner.weight.data
    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(current_storage_owner)
    with pytest.raises(trainer.TrainerContractError, match="storage binding drift"):
        trainer.export_cpu_synthetic_checkpoint_sections(
            rebound_session,
            sampler_state=b"sampler\n",
            dropout_state=b"dropout\n",
            validation_state=b"validation\n",
        )
    assert trainer.cpu_synthetic_session_state(rebound_session) == "RETIRED"

    replaced_model = _model()
    replaced_session = trainer.open_cpu_synthetic_training_session(
        _admission(replaced_model)
    )
    replacement_owner = _model()
    replaced_model.weight = replacement_owner.weight
    with pytest.raises(trainer.TrainerContractError, match="already bound"):
        _admission(replacement_owner)
    assert trainer.cpu_synthetic_session_state(replaced_session) == "ACTIVE"


def test_concurrent_duplicate_admission_has_exactly_one_owner() -> None:
    model = _model()
    barrier = threading.Barrier(3)
    outcomes: list[object] = []
    outcome_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            outcome: object = _admission(model)
        except BaseException as exc:  # both result classes are asserted below
            outcome = exc
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    admissions = [
        item
        for item in outcomes
        if type(item) is trainer.SyntheticOptimizerAdmission
    ]
    errors = [item for item in outcomes if type(item) is trainer.TrainerContractError]
    assert len(admissions) == 1
    assert len(errors) == 1
    assert "already bound" in str(errors[0])


def test_concurrent_open_same_admission_constructs_once() -> None:
    model = _model()
    admission = _admission(model)
    barrier = threading.Barrier(3)
    outcomes: list[object] = []
    outcome_lock = threading.Lock()
    constructors_before = trainer.cpu_synthetic_optimizer_constructor_count()

    def worker() -> None:
        barrier.wait()
        try:
            outcome: object = trainer.open_cpu_synthetic_training_session(admission)
        except BaseException as exc:  # both result classes are asserted below
            outcome = exc
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    sessions = [
        item for item in outcomes if type(item) is trainer.SyntheticTrainingSession
    ]
    errors = [
        item for item in outcomes if type(item) is trainer.TrainingAdmissionBurnedError
    ]
    assert len(sessions) == 1
    assert len(errors) == 1
    assert trainer.cpu_synthetic_optimizer_constructor_count() == constructors_before + 1


def test_concurrent_open_and_duplicate_issue_have_no_ownership_gap() -> None:
    model = _model()
    admission = _admission(model)
    barrier = threading.Barrier(3)
    outcomes: list[object] = []
    outcome_lock = threading.Lock()

    def opener() -> None:
        barrier.wait()
        try:
            outcome: object = trainer.open_cpu_synthetic_training_session(admission)
        except BaseException as exc:  # both result classes are asserted below
            outcome = exc
        with outcome_lock:
            outcomes.append(outcome)

    def issuer() -> None:
        barrier.wait()
        try:
            outcome: object = _admission(model)
        except BaseException as exc:  # both result classes are asserted below
            outcome = exc
        with outcome_lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=opener), threading.Thread(target=issuer)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()

    assert sum(
        type(item) is trainer.SyntheticTrainingSession for item in outcomes
    ) == 1
    ownership_errors = [
        item
        for item in outcomes
        if type(item) is trainer.TrainerContractError
        and "already bound" in str(item)
    ]
    assert len(ownership_errors) == 1


def test_direct_forgery_and_equal_value_objects_are_rejected() -> None:
    with pytest.raises(trainer.TrainerContractError):
        trainer.SyntheticOptimizerAdmission()
    with pytest.raises(trainer.TrainerContractError):
        trainer.SyntheticTrainingSession()
    with pytest.raises(trainer.TrainerContractError):
        trainer.StepReceipt()
    forged = object.__new__(trainer.SyntheticOptimizerAdmission)
    with pytest.raises(trainer.TrainerContractError):
        trainer.cpu_synthetic_admission_status(forged)
    forged_receipt = object.__new__(trainer.StepReceipt)
    with pytest.raises(trainer.TrainerContractError):
        trainer.canonical_step_receipt_bytes(forged_receipt)


def test_opaque_token_registries_use_identity_not_mutable_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _model()
    admission = _admission(model)
    monkeypatch.setattr(
        trainer.SyntheticOptimizerAdmission,
        "__eq__",
        lambda _self, _other: True,
        raising=False,
    )
    monkeypatch.setattr(
        trainer.SyntheticOptimizerAdmission,
        "__hash__",
        lambda _self: 1,
        raising=False,
    )
    forged_admission = object.__new__(trainer.SyntheticOptimizerAdmission)
    with pytest.raises(trainer.TrainerContractError):
        trainer.open_cpu_synthetic_training_session(forged_admission)
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":ISSUED")

    session = trainer.open_cpu_synthetic_training_session(admission)
    monkeypatch.setattr(
        trainer.SyntheticTrainingSession,
        "__eq__",
        lambda _self, _other: True,
        raising=False,
    )
    monkeypatch.setattr(
        trainer.SyntheticTrainingSession,
        "__hash__",
        lambda _self: 1,
        raising=False,
    )
    forged_session = object.__new__(trainer.SyntheticTrainingSession)
    with pytest.raises(trainer.TrainerContractError):
        trainer.cpu_synthetic_session_state(forged_session)

    receipt = trainer.run_cpu_synthetic_optimizer_step(
        session,
        _loss(model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    expected = trainer.canonical_step_receipt_bytes(receipt)
    monkeypatch.setattr(
        trainer.StepReceipt,
        "__eq__",
        lambda _self, _other: True,
        raising=False,
    )
    monkeypatch.setattr(
        trainer.StepReceipt,
        "__hash__",
        lambda _self: 1,
        raising=False,
    )
    forged_receipt = object.__new__(trainer.StepReceipt)
    with pytest.raises(trainer.TrainerContractError):
        trainer.canonical_step_receipt_bytes(forged_receipt)
    assert trainer.canonical_step_receipt_bytes(receipt) == expected


def test_concurrent_session_use_has_at_most_one_commit() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    barrier = threading.Barrier(3)
    outcomes: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        loss = _loss(model)
        barrier.wait()
        try:
            trainer.run_cpu_synthetic_optimizer_step(
                session,
                loss,
                epoch_index=0,
                global_step=0,
                batch_position=0,
                dropout_position=0,
            )
        except BaseException as exc:  # test records the exact terminal class below
            outcome = type(exc).__name__
        else:
            outcome = "COMMITTED"
        with lock:
            outcomes.append(outcome)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert outcomes.count("COMMITTED") == 1
    assert outcomes.count("TrainingAdmissionBurnedError") == 1


def test_checkpoint_resume_matches_uninterrupted_two_step_trajectory_bitwise(
    tmp_path: Path,
) -> None:
    uninterrupted_model = _model()
    uninterrupted = trainer.open_cpu_synthetic_training_session(
        _admission(uninterrupted_model)
    )
    trainer.run_cpu_synthetic_optimizer_step(
        uninterrupted,
        _loss(uninterrupted_model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    sections = trainer.export_cpu_synthetic_checkpoint_sections(
        uninterrupted,
        sampler_state=b"sampler/epoch0/position1\n",
        dropout_state=b"dropout/global-step1\n",
        validation_state=b"validation/none\n",
    )
    bindings = _checkpoint_bindings(sections, global_step=1)
    envelope = checkpoint.build_cpu_synthetic_checkpoint(bindings, sections)
    checkpoint_raw = checkpoint.canonical_checkpoint_bytes(envelope)
    target = (tmp_path / "step000001.ppckpt").absolute()
    assert checkpoint.write_checkpoint_atomic(envelope, target) == hashlib.sha256(
        checkpoint_raw
    ).digest()
    validated = checkpoint.read_checkpoint(target, expected_bindings=bindings)
    trainer.run_cpu_synthetic_optimizer_step(
        uninterrupted,
        _loss(uninterrupted_model),
        epoch_index=0,
        global_step=1,
        batch_position=1,
        dropout_position=1,
    )

    resumed_model = _model()
    resumed = trainer.resume_cpu_synthetic_training_session(
        _admission(resumed_model),
        validated,
    )
    trainer.run_cpu_synthetic_optimizer_step(
        resumed,
        _loss(resumed_model),
        epoch_index=0,
        global_step=1,
        batch_position=1,
        dropout_position=1,
    )
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            uninterrupted_model.parameters(),
            resumed_model.parameters(),
            strict=True,
        )
    )
    uninterrupted_sections = trainer.export_cpu_synthetic_checkpoint_sections(
        uninterrupted,
        sampler_state=b"sampler/epoch0/position2\n",
        dropout_state=b"dropout/global-step2\n",
        validation_state=b"validation/none\n",
    )
    resumed_sections = trainer.export_cpu_synthetic_checkpoint_sections(
        resumed,
        sampler_state=b"sampler/epoch0/position2\n",
        dropout_state=b"dropout/global-step2\n",
        validation_state=b"validation/none\n",
    )
    assert uninterrupted_sections == resumed_sections


def test_invalid_resume_fails_before_adamw_constructor_and_burns() -> None:
    source_model = _model()
    source = trainer.open_cpu_synthetic_training_session(_admission(source_model))
    trainer.run_cpu_synthetic_optimizer_step(
        source,
        _loss(source_model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    sections = list(
        trainer.export_cpu_synthetic_checkpoint_sections(
            source,
            sampler_state=b"sampler\n",
            dropout_state=b"dropout\n",
            validation_state=b"validation\n",
        )
    )
    optimizer_payload = json.loads(dict(sections)["optimizer_state"])
    optimizer_payload["groups"][0]["lr_hex"] = float(2.0e-4).hex()
    sections[2] = (
        "optimizer_state",
        json.dumps(
            optimizer_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n",
    )
    bindings = _checkpoint_bindings(tuple(sections), global_step=1)
    envelope = checkpoint.build_cpu_synthetic_checkpoint(bindings, tuple(sections))
    target_model = _model()
    admission = _admission(target_model)
    constructors_before = trainer.cpu_synthetic_optimizer_constructor_count()
    with pytest.raises(trainer.TrainerContractError):
        trainer.resume_cpu_synthetic_training_session(admission, envelope)
    assert trainer.cpu_synthetic_optimizer_constructor_count() == constructors_before
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":BURNED")
    assert all(parameter.grad is None for parameter in target_model.parameters())


def test_invalid_equal_length_rng_state_fails_before_adamw_constructor() -> None:
    source_model = _model()
    source = trainer.open_cpu_synthetic_training_session(_admission(source_model))
    trainer.run_cpu_synthetic_optimizer_step(
        source,
        _loss(source_model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    sections = list(
        trainer.export_cpu_synthetic_checkpoint_sections(
            source,
            sampler_state=b"sampler\n",
            dropout_state=b"dropout\n",
            validation_state=b"validation\n",
        )
    )
    rng_index = next(index for index, row in enumerate(sections) if row[0] == "rng_state")
    sections[rng_index] = ("rng_state", bytes(len(sections[rng_index][1])))
    bindings = _checkpoint_bindings(tuple(sections), global_step=1)
    envelope = checkpoint.build_cpu_synthetic_checkpoint(bindings, tuple(sections))
    target_model = _model()
    admission = _admission(target_model)
    constructors_before = trainer.cpu_synthetic_optimizer_constructor_count()
    with pytest.raises(trainer.TrainerContractError, match="RNG checkpoint state is invalid"):
        trainer.resume_cpu_synthetic_training_session(admission, envelope)
    assert trainer.cpu_synthetic_optimizer_constructor_count() == constructors_before
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":BURNED")
    assert all(parameter.grad is None for parameter in target_model.parameters())


@pytest.mark.parametrize("bad_second_moment", [-1.0, -0.0])
def test_negative_second_moment_fails_before_adamw_constructor(
    bad_second_moment: float,
) -> None:
    source_model = _model()
    source = trainer.open_cpu_synthetic_training_session(_admission(source_model))
    trainer.run_cpu_synthetic_optimizer_step(
        source,
        _loss(source_model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    sections = list(
        trainer.export_cpu_synthetic_checkpoint_sections(
            source,
            sampler_state=b"sampler\n",
            dropout_state=b"dropout\n",
            validation_state=b"validation\n",
        )
    )
    optimizer_index = next(
        index for index, row in enumerate(sections) if row[0] == "optimizer_state"
    )
    optimizer_payload = json.loads(sections[optimizer_index][1])
    second_moment = optimizer_payload["parameter_states"][0]["state"][
        "exp_avg_sq"
    ]
    raw = bytearray.fromhex(second_moment["raw_hex"])
    raw[:4] = struct.pack("<f", bad_second_moment)
    second_moment["raw_hex"] = raw.hex()
    sections[optimizer_index] = (
        "optimizer_state",
        json.dumps(
            optimizer_payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n",
    )
    bindings = _checkpoint_bindings(tuple(sections), global_step=1)
    envelope = checkpoint.build_cpu_synthetic_checkpoint(bindings, tuple(sections))
    target_model = _model()
    admission = _admission(target_model)
    target_before = tuple(
        parameter.detach().clone() for parameter in target_model.parameters()
    )
    rng_before = torch.get_rng_state().clone()
    constructors_before = trainer.cpu_synthetic_optimizer_constructor_count()
    with pytest.raises(trainer.TrainerContractError, match="nonnegative"):
        trainer.resume_cpu_synthetic_training_session(admission, envelope)
    assert trainer.cpu_synthetic_optimizer_constructor_count() == constructors_before
    assert trainer.cpu_synthetic_admission_status(admission).endswith(":BURNED")
    assert all(parameter.grad is None for parameter in target_model.parameters())
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(
            target_before, target_model.parameters(), strict=True
        )
    )
    replacement = _admission(target_model)
    assert trainer.cpu_synthetic_admission_status(replacement).endswith(":ISSUED")


def test_public_constants_and_direct_helpers_do_not_rebind_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline_model = _model()
    rebound_model = _model()
    baseline = trainer.open_cpu_synthetic_training_session(_admission(baseline_model))
    admission_type = trainer.SyntheticOptimizerAdmission
    session_type = trainer.SyntheticTrainingSession
    receipt_type = trainer.StepReceipt

    def bomb(*_: object, **__: object) -> object:
        raise AssertionError("rebound dependency was read")

    monkeypatch.setattr(trainer, "LEARNING_RATE", 0.5)
    monkeypatch.setattr(trainer, "BETAS", (0.1, 0.2))
    monkeypatch.setattr(trainer, "EPSILON", 0.5)
    monkeypatch.setattr(trainer, "DECAY_WEIGHT_DECAY", 0.5)
    monkeypatch.setattr(trainer, "NO_DECAY_WEIGHT_DECAY", 0.5)
    monkeypatch.setattr(trainer, "GRAD_CLIP_NORM", 99.0)
    monkeypatch.setattr(trainer, "STEP_STATUS", "FORGED")
    monkeypatch.setattr(trainer, "_named_parameters", bomb)
    monkeypatch.setattr(trainer, "_model_digest", bomb)
    monkeypatch.setattr(trainer, "_optimizer_digest", bomb)
    monkeypatch.setattr(trainer, "_receipt_bytes", bomb)
    monkeypatch.setattr(trainer, "_tensor_bytes", bomb)
    monkeypatch.setattr(trainer, "_tensor_record_bytes", bomb)
    monkeypatch.setattr(trainer, "_write_field", bomb)
    monkeypatch.setattr(trainer, "_exact_ascii", bomb)
    monkeypatch.setattr(trainer, "_exact_uint", bomb)
    monkeypatch.setattr(trainer, "_exact_name_tuple", bomb)
    monkeypatch.setattr(trainer, "_canonical_json_bytes", bomb)
    monkeypatch.setattr(trainer, "_tensor_json", bomb)
    monkeypatch.setattr(trainer, "_group_payloads", bomb)
    monkeypatch.setattr(trainer, "_parameter_binding", bomb)
    monkeypatch.setattr(trainer, "_admission_records_overlap", bomb)
    monkeypatch.setattr(trainer, "_require_unowned_admission", bomb)
    monkeypatch.setattr(trainer, "_require_nonnegative_float32_bytes", bomb)
    monkeypatch.setattr(trainer, "_retire_session_record", bomb)
    monkeypatch.setattr(trainer, "_ADMISSIONS", object())
    monkeypatch.setattr(trainer, "_ADMISSION_STATES", object())
    monkeypatch.setattr(trainer, "_SESSIONS", object())
    monkeypatch.setattr(trainer, "_RECEIPTS", object())
    monkeypatch.setattr(trainer, "_REGISTRY_LOCK", object())
    monkeypatch.setattr(trainer.torch.optim, "AdamW", bomb)
    monkeypatch.setattr(trainer.torch.nn.utils, "clip_grad_norm_", bomb)
    monkeypatch.setattr(trainer.hashlib, "sha256", bomb)
    monkeypatch.setattr(trainer.json, "dumps", bomb)
    monkeypatch.setattr(trainer.json, "loads", bomb)
    monkeypatch.setattr(trainer.struct, "pack", bomb)
    monkeypatch.setattr(trainer, "SyntheticOptimizerAdmission", object)
    monkeypatch.setattr(trainer, "SyntheticTrainingSession", object)
    monkeypatch.setattr(trainer, "StepReceipt", object)
    monkeypatch.setattr(trainer, "_AdmissionRecord", object)
    monkeypatch.setattr(trainer, "_SessionRecord", object)
    monkeypatch.setattr(trainer, "TrainerContractError", RuntimeError)
    monkeypatch.setattr(trainer, "TrainingAdmissionBurnedError", RuntimeError)
    monkeypatch.setattr(trainer, "TrainingTransactionError", RuntimeError)
    monkeypatch.setattr(trainer, "Tensor", object)
    monkeypatch.setattr(trainer, "nn", object())
    monkeypatch.setattr(trainer, "_checkpoint_module", object())
    monkeypatch.setattr(trainer, "copy", object())
    monkeypatch.setattr(trainer, "math", object())
    monkeypatch.setattr(trainer, "threading", object())
    monkeypatch.setattr(trainer, "hashlib", object())
    monkeypatch.setattr(trainer, "json", object())
    monkeypatch.setattr(trainer, "struct", object())
    monkeypatch.setattr(trainer, "torch", object())

    rebound_admission = _admission(rebound_model)
    assert type(rebound_admission) is admission_type
    rebound = trainer.open_cpu_synthetic_training_session(rebound_admission)
    assert type(rebound) is session_type
    baseline_receipt = trainer.run_cpu_synthetic_optimizer_step(
        baseline,
        _loss(baseline_model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    rebound_receipt = trainer.run_cpu_synthetic_optimizer_step(
        rebound,
        _loss(rebound_model),
        epoch_index=0,
        global_step=0,
        batch_position=0,
        dropout_position=0,
    )
    assert type(rebound_receipt) is receipt_type
    assert all(
        torch.equal(left, right)
        for left, right in zip(
            baseline_model.parameters(),
            rebound_model.parameters(),
            strict=True,
        )
    )
    assert trainer.canonical_step_receipt_bytes(
        baseline_receipt
    ) == trainer.canonical_step_receipt_bytes(rebound_receipt)


@pytest.mark.parametrize("operation", ["step", "export"])
def test_boundary_validation_exception_retires_and_releases(
    operation: str,
) -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    finite_loss = _loss(model)
    original = tuple(parameter.detach().clone() for parameter in model.parameters())
    with torch.no_grad():
        model.weight[0, 0] = float("nan")
    model.weight.detach = lambda: original[0]

    with pytest.raises(trainer.TrainerContractError, match="nonfinite"):
        if operation == "step":
            trainer.run_cpu_synthetic_optimizer_step(
                session,
                finite_loss,
                epoch_index=0,
                global_step=0,
                batch_position=0,
                dropout_position=0,
            )
        else:
            trainer.export_cpu_synthetic_checkpoint_sections(
                session,
                sampler_state=b"sampler\n",
                dropout_state=b"dropout\n",
                validation_state=b"validation\n",
            )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"

    with torch.no_grad():
        for parameter, expected in zip(model.parameters(), original, strict=True):
            parameter.copy_(expected)
    replacement = _admission(model)
    assert trainer.cpu_synthetic_admission_status(replacement).endswith(":ISSUED")


def test_external_parameter_drift_is_rejected_before_backward_and_retires() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    with torch.no_grad():
        model.weight.add_(0.25)
    drifted = tuple(parameter.detach().clone() for parameter in model.parameters())
    with pytest.raises(trainer.TrainerContractError):
        trainer.run_cpu_synthetic_optimizer_step(
            session,
            _loss(model),
            epoch_index=0,
            global_step=0,
            batch_position=0,
            dropout_position=0,
        )
    assert all(parameter.grad is None for parameter in model.parameters())
    assert all(
        torch.equal(expected, actual)
        for expected, actual in zip(drifted, model.parameters(), strict=True)
    )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"


def test_external_parameter_drift_blocks_checkpoint_export() -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    with torch.no_grad():
        model.bias.mul_(2.0)
    with pytest.raises(trainer.TrainerContractError):
        trainer.export_cpu_synthetic_checkpoint_sections(
            session,
            sampler_state=b"sampler\n",
            dropout_state=b"dropout\n",
            validation_state=b"validation\n",
        )
    assert trainer.cpu_synthetic_session_state(session) == "RETIRED"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("epoch_index", True),
        ("global_step", 1.0),
        ("batch_position", -1),
        ("dropout_position", 2**32),
    ],
)
def test_step_position_exact_types_fail_before_backward(field: str, value: object) -> None:
    model = _model()
    session = trainer.open_cpu_synthetic_training_session(_admission(model))
    kwargs: dict[str, object] = {
        "epoch_index": 0,
        "global_step": 0,
        "batch_position": 0,
        "dropout_position": 0,
    }
    kwargs[field] = value
    with pytest.raises((TypeError, trainer.TrainerContractError)):
        trainer.run_cpu_synthetic_optimizer_step(session, _loss(model), **kwargs)
    assert all(parameter.grad is None for parameter in model.parameters())
    assert trainer.cpu_synthetic_session_state(session) == "ACTIVE"
