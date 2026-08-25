from __future__ import annotations

import builtins
import hashlib
import ast
import inspect
import json
import math
import struct
import threading

import pytest
import torch

from phasepair_core import residual_head as residual


def _tokens(system: str, batch_size: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    tokens = torch.zeros((2, batch_size, 6, 13), dtype=torch.float32)
    tokens[..., 7:] = torch.eye(6, dtype=torch.float32).reshape(1, 1, 6, 6)
    mask = torch.ones((2, batch_size, 6), dtype=torch.bool)
    for row in range(batch_size):
        for slot in range(6):
            a = float(1 + row + slot)
            b = float(11 + row + slot)
            if system == "01_GENERIC":
                fields_ab = [0.0] * 7
                fields_ba = [0.0] * 7
            elif system == "02_WAMO_MARGINAL_WAVELET":
                combined = math.log(math.exp(a) + math.exp(b) - 1.0e-12)
                fields_ab = [a, b, combined, abs(a - b), 0.5 * (a + b), 0.0, 0.0]
                fields_ba = [b, a, combined, abs(a - b), 0.5 * (a + b), 0.0, 0.0]
            elif system == "03_INTEREDIT_MEAN_DIFFERENCE_DCT":
                group = slot // 2
                e_s = float(1 + row + group)
                e_d = float(11 + row + group)
                e_total = math.log(
                    math.exp(e_s) + math.exp(e_d) - 1.0e-12
                )
                selected = e_s if slot % 2 == 0 else e_d
                fields_ab = [
                    selected,
                    e_s,
                    e_d,
                    e_total,
                    selected - e_total,
                    0.0,
                    0.0,
                ]
                fields_ba = list(fields_ab)
            elif system == "04_NO_RELATION":
                fields_ab = [a, b, 0.0, 0.0, 0.0, 0.0, 0.0]
                fields_ba = [b, a, 0.0, 0.0, 0.0, 0.0, 0.0]
            elif system == "05_PHASE_STRIPPED":
                fields_ab = [a, b, 0.25, 0.0, 0.0, 0.0, 0.0]
                fields_ba = [b, a, 0.25, 0.0, 0.0, 0.0, 0.0]
            else:
                phase = -math.pi / 4.0
                cosine = math.cos(phase)
                sine = math.sin(phase)
                normalized_delay = -phase / math.pi
                fields_ab = [
                    a,
                    b,
                    0.25,
                    cosine,
                    sine,
                    normalized_delay,
                    1.0,
                ]
                fields_ba = [
                    b,
                    a,
                    0.25,
                    cosine,
                    -sine,
                    -normalized_delay,
                    1.0,
                ]
            tokens[0, row, slot, :7] = torch.tensor(fields_ab)
            tokens[1, row, slot, :7] = torch.tensor(fields_ba)
    if system == "03_INTEREDIT_MEAN_DIFFERENCE_DCT":
        mask[:, -1, -2:] = False
        tokens[:, -1, -2:, :7] = 0.0
    else:
        mask[:, -1, -1] = False
        tokens[:, -1, -1, :7] = 0.0
    return tokens.contiguous(), mask.contiguous()


def _normalized(rows: int) -> torch.Tensor:
    values = torch.arange(1, rows * 512 + 1, dtype=torch.float32).reshape(rows, 512)
    return (values / torch.linalg.vector_norm(values, dim=1, keepdim=True)).contiguous()


def _initialized(seed: int = 1729) -> residual.PhasePairResidualHead:
    model = residual.PhasePairResidualHead(device="cpu")
    receipt = residual.initialize_residual_head(model, seed)
    assert residual.verify_residual_head_initializer_receipt(receipt, model, seed) == hashlib.sha256(receipt).digest()
    return model


def test_01_closed_system_and_parameter_ledger() -> None:
    assert residual.AUTHORITY == 0
    assert residual.STATUS == "DATA_FREE_RESIDUAL_HEAD_NONPRODUCTION_AUTHORITY0"
    assert residual.SYSTEMS == (
        "01_GENERIC",
        "02_WAMO_MARGINAL_WAVELET",
        "03_INTEREDIT_MEAN_DIFFERENCE_DCT",
        "04_NO_RELATION",
        "05_PHASE_STRIPPED",
        "06_PHASEPAIR_FULL",
    )
    inventory = json.loads(residual.canonical_residual_inventory_bytes())
    assert inventory["parameter_count"] == 4
    assert inventory["parameter_numel"] == 135_168
    assert [row["name"] for row in inventory["rows"]] == [
        "residual.fc1.bias",
        "residual.fc1.weight",
        "residual.fc2.bias",
        "residual.fc2.weight",
    ]
    assert sum(
        len(residual.canonical_residual_parameter_bytes(row["name"], 1729)) // 4
        for row in inventory["rows"]
    ) == 135_168
    fc1 = residual.canonical_residual_parameter_bytes("residual.fc1.weight", 1729)
    assert hashlib.sha256(fc1).hexdigest() == "9a808f3e89adc71b7bcd48d81dd9dd684d0fbda203aa832e7337a1f1edc9da4d"
    name = b"residual.fc1.weight"
    prefix = b"phasepair-init-v1" + (1729).to_bytes(8, "big") + len(name).to_bytes(2, "big") + name
    bound = math.sqrt(6.0 / (13.0 + 256.0))
    independent = bytearray()
    for index in range(16):
        integer = int.from_bytes(
            hashlib.sha256(prefix + index.to_bytes(8, "big")).digest()[:8],
            "big",
        )
        unit = ((integer << 1) + 1) / float(1 << 65)
        independent.extend(struct.pack("<f", (2.0 * unit - 1.0) * bound))
    assert fc1[:64] == bytes(independent)


def test_02_initializer_is_identical_and_receipt_is_live_bound() -> None:
    first = _initialized()
    second = _initialized()
    first_rows = dict(first.named_parameters())
    second_rows = dict(second.named_parameters())
    assert tuple(first_rows) == ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias")
    assert sum(parameter.numel() for parameter in first_rows.values()) == 135_168
    for name in first_rows:
        assert torch.equal(first_rows[name], second_rows[name])
    with pytest.raises(residual.ResidualHeadContractError, match="already initialized"):
        residual.initialize_residual_head(first, 1729)
    forged = residual.canonical_residual_inventory_bytes()
    with pytest.raises(residual.ResidualHeadContractError):
        residual.verify_residual_head_initializer_receipt(forged, first, 1729)
    with torch.no_grad():
        first.fc1_bias[0] = 1.0
    receipt = residual.initialize_residual_head(
        residual.PhasePairResidualHead(device="cpu"), 1729
    )
    with pytest.raises(
        residual.ResidualHeadContractError,
        match="canonically initialized|does not bind",
    ):
        residual.verify_residual_head_initializer_receipt(receipt, first, 1729)


def test_03_dropout_exact_small_and_b128_goldens() -> None:
    small = json.loads(residual.canonical_head_dropout_receipt(1729, 0, 2))
    assert small["h0_sha256"] == "887afe8b4d19d147ddddff77117c77000239f37e22007a57c03289a79dedad7a"
    assert small["first16_keep"] == "1111111011111111"
    assert small["packed_sha256"] == "d778ee43b0b410b9c584a0a3bafb0e3c1077b95800ae2ada6357044fc8fe3a15"
    assert small["expanded_sha256"] == "92224beac462a3e5d7f7b1d65ae311fc4827f6a1954af1871dc457846ec13f58"
    large = json.loads(residual.canonical_head_dropout_receipt(1729, 0, 128))
    assert large["h0_sha256"] == "26a2a64ab0e8793c095e1ca15f807986097d8a119e5950774187d8b2189b3022"
    assert large["drop_count"] == 39_531
    mask = residual.canonical_head_dropout_mask(1729, 0, 2, torch.device("cpu"))
    assert mask.shape == (2, 2, 6, 256)
    assert mask.dtype == torch.float32 and mask.is_contiguous()
    assert set(mask.view(torch.int32).unique().tolist()) == {0, 0x3F8E38E4}


@pytest.mark.parametrize("system", residual.SYSTEMS)
def test_04_six_token_laws_and_dense_forward(system: str) -> None:
    model = _initialized()
    model.eval()
    tokens, mask = _tokens(system)
    text = _normalized(6)
    scores = residual.residual_ordered_scores(
        model, tokens, mask, text, system, None, None
    )
    assert scores.shape == (2, 2, 6)
    assert scores.dtype == torch.float32 and torch.isfinite(scores).all()
    assert torch.all(scores >= -1.0) and torch.all(scores <= 1.0)


def test_05_training_uses_one_exact_dense_dropout_call() -> None:
    model = _initialized()
    model.train()
    tokens, mask = _tokens("06_PHASEPAIR_FULL")
    first = model(
        tokens,
        system="06_PHASEPAIR_FULL",
        active_mask=mask,
        seed=1729,
        global_optimizer_step=0,
    )
    second = model(
        tokens,
        system="06_PHASEPAIR_FULL",
        active_mask=mask,
        seed=1729,
        global_optimizer_step=0,
    )
    changed = model(
        tokens,
        system="06_PHASEPAIR_FULL",
        active_mask=mask,
        seed=1729,
        global_optimizer_step=1,
    )
    assert torch.equal(first, second)
    assert not torch.equal(first, changed)
    with pytest.raises(residual.ResidualHeadHold):
        model(tokens, system="06_PHASEPAIR_FULL", active_mask=mask)
    model.eval()
    with pytest.raises(residual.ResidualHeadContractError, match="must not consume"):
        model(
            tokens,
            system="06_PHASEPAIR_FULL",
            active_mask=mask,
            seed=1729,
            global_optimizer_step=0,
        )


def test_06_all_masked_score_is_exact_zero_without_nan() -> None:
    model = _initialized()
    model.eval()
    tokens, mask = _tokens("01_GENERIC")
    mask.zero_()
    text = _normalized(6)
    scores = residual.residual_ordered_scores(
        model, tokens, mask, text, "01_GENERIC", None, None
    )
    assert torch.equal(scores, torch.zeros_like(scores))
    assert set(scores.view(torch.int32).unique().tolist()) == {0}


def test_07_full_objective_keeps_base_frozen_and_trains_only_head() -> None:
    model = _initialized()
    model.eval()
    tokens, mask = _tokens("06_PHASEPAIR_FULL")
    base = _normalized(4).reshape(2, 2, 512)
    text = _normalized(6)
    positive = torch.zeros((2, 6), dtype=torch.bool)
    positive[0, :3] = True
    positive[1, 3:] = True
    scale = torch.tensor([math.log(1.0 / 0.07)], dtype=torch.float32)
    output = residual.phasepair_residual_objective(
        model,
        tokens,
        mask,
        base,
        text,
        scale,
        positive,
        "06_PHASEPAIR_FULL",
        None,
        None,
    )
    assert output.status == "DATA_FREE_OBJECTIVE_COMPUTED_NONPRODUCTION_NO_RESULT"
    assert output.logits.shape == (2, 6)
    output.loss.backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert base.grad is None and text.grad is None and scale.grad is None


def test_08_token_and_type_mutants_fail_closed() -> None:
    model = _initialized()
    model.eval()
    tokens, mask = _tokens("06_PHASEPAIR_FULL")
    text = _normalized(6)
    bad = tokens.clone()
    bad[0, 0, 0, 7] = 0.0
    with pytest.raises(residual.ResidualHeadContractError, match="one-hot"):
        residual.residual_ordered_scores(
            model, bad, mask, text, "06_PHASEPAIR_FULL", None, None
        )
    bad = tokens.clone()
    bad[0, 0, 0, 8] = -0.0
    with pytest.raises(residual.ResidualHeadContractError, match="one-hot"):
        residual.residual_ordered_scores(
            model, bad, mask, text, "06_PHASEPAIR_FULL", None, None
        )
    bad = tokens.clone()
    bad[1, 0, 0, 4] = bad[0, 0, 0, 4]
    with pytest.raises(residual.ResidualHeadContractError, match="AB/BA"):
        residual.residual_ordered_scores(
            model, bad, mask, text, "06_PHASEPAIR_FULL", None, None
        )
    bad = tokens.clone()
    bad[:, -1, -1, 0] = -0.0
    with pytest.raises(residual.ResidualHeadContractError, match=r"exact \+0"):
        residual.residual_ordered_scores(
            model, bad, mask, text, "06_PHASEPAIR_FULL", None, None
        )
    subclass = tokens.as_subclass(type("TensorSubclass", (torch.Tensor,), {}))
    with pytest.raises(TypeError, match="exact torch.Tensor"):
        residual.residual_ordered_scores(
            model, subclass, mask, text, "06_PHASEPAIR_FULL", None, None
        )
    class StringSubclass(str):
        pass
    with pytest.raises(residual.ResidualHeadContractError, match="exact residual"):
        residual.residual_ordered_scores(
            model,
            tokens,
            mask,
            text,
            StringSubclass("06_PHASEPAIR_FULL"),
            None,
            None,
        )


def test_09_uninitialized_subclass_and_global_rebind_do_not_open_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uninitialized = residual.PhasePairResidualHead(device="cpu")
    uninitialized.eval()
    tokens, mask = _tokens("01_GENERIC")
    with pytest.raises(residual.ResidualHeadHold):
        uninitialized(tokens, system="01_GENERIC", active_mask=mask)

    class Derived(residual.PhasePairResidualHead):
        pass

    with pytest.raises(residual.ResidualHeadContractError, match="exact ResidualHead"):
        residual.initialize_residual_head(Derived(device="cpu"), 1729)

    baseline = residual.canonical_head_dropout_receipt(1729, 0, 2)
    monkeypatch.setattr(residual, "SYSTEMS", ())
    monkeypatch.setattr(residual, "AUTHORITY", 9)
    monkeypatch.setattr(residual, "torch", object())
    monkeypatch.setattr(residual, "_U64_MASK", 0)
    monkeypatch.setattr(residual, "_DROP_THRESHOLD", 0)
    monkeypatch.setattr(residual.hashlib, "sha256", lambda *_: b"forged")
    monkeypatch.setattr(residual.json, "dumps", lambda *_args, **_kwargs: "{}")
    assert residual.canonical_head_dropout_receipt(1729, 0, 2) == baseline


def test_10_strict_integer_and_shape_boundaries() -> None:
    for bad_seed in (True, -1, 1 << 64, 1.0):
        with pytest.raises(residual.ResidualHeadContractError):
            residual.canonical_head_dropout_receipt(bad_seed, 0, 2)
    for bad_step in (True, -1, 1 << 32, 1.0):
        with pytest.raises(residual.ResidualHeadContractError):
            residual.canonical_head_dropout_receipt(1729, bad_step, 2)
    for bad_batch in (True, 0, 129, 2.0):
        with pytest.raises(residual.ResidualHeadContractError):
            residual.canonical_head_dropout_receipt(1729, 0, bad_batch)


def test_11_initializer_is_one_atomic_locked_transaction() -> None:
    source = inspect.getsource(residual._make_public_api)
    tree = ast.parse(source)
    initializer = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "initialize_impl"
    )
    lock_scope = next(
        node
        for node in initializer.body
        if isinstance(node, ast.With)
        and any(
            isinstance(item.context_expr, ast.Name)
            and item.context_expr.id == "registry_lock"
            for item in node.items
        )
    )
    transaction = next(node for node in lock_scope.body if isinstance(node, ast.Try))
    assert transaction.finalbody

    def assignment_line(name: str, value: object) -> int:
        return min(
            node.lineno
            for node in ast.walk(transaction)
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == name for target in node.targets)
            and isinstance(node.value, ast.Constant)
            and node.value.value is value
        )

    receipt_line = min(
        node.lineno
        for node in ast.walk(transaction)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "receipt" for target in node.targets)
    )
    registry_line = min(
        node.lineno
        for node in ast.walk(transaction)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "weak_set"
    )
    committed_line = assignment_line("committed", True)
    assert receipt_line < registry_line < committed_line
    assert assignment_line("registry_inserted", True) < committed_line
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "weak_pop"
        for node in ast.walk(transaction.finalbody[0])
    )


def test_12_same_model_initializer_race_has_one_commit() -> None:
    model = residual.PhasePairResidualHead(device="cpu")
    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, object]] = []
    outcomes_lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        try:
            value: object = residual.initialize_residual_head(model, 1729)
            kind = "receipt"
        except BaseException as exc:  # pragma: no branch - both paths are asserted
            value = exc
            kind = "error"
        with outcomes_lock:
            outcomes.append((kind, value))

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=30.0)
        assert not thread.is_alive()

    receipts = [value for kind, value in outcomes if kind == "receipt"]
    errors = [value for kind, value in outcomes if kind == "error"]
    assert len(receipts) == 1 and type(receipts[0]) is bytes
    assert len(errors) == 1
    assert type(errors[0]) is residual.ResidualHeadContractError
    assert "already initialized" in str(errors[0])
    assert residual.verify_residual_head_initializer_receipt(
        receipts[0], model, 1729
    ) == hashlib.sha256(receipts[0]).digest()


def test_13_exact_object_schema_blocks_bypass_and_extra_state() -> None:
    tokens, mask = _tokens("01_GENERIC")
    text = _normalized(2)

    overridden = _initialized()
    overridden.eval()
    object.__setattr__(
        overridden,
        "forward",
        lambda *_args, **_kwargs: torch.zeros((2, 2, 6, 512)),
    )
    with pytest.raises(residual.ResidualHeadContractError, match="object schema"):
        residual.residual_ordered_scores(
            overridden, tokens, mask, text, "01_GENERIC", None, None
        )
    with pytest.raises(residual.ResidualHeadContractError, match="object schema"):
        overridden(tokens, system="01_GENERIC", active_mask=mask)

    plain_extra = residual.PhasePairResidualHead(device="cpu")
    plain_extra.extra = 1
    with pytest.raises(residual.ResidualHeadContractError, match="object schema"):
        residual.initialize_residual_head(plain_extra, 1729)

    module_extra = residual.PhasePairResidualHead(device="cpu")
    module_extra.extra = torch.nn.Linear(1, 1)
    with pytest.raises(residual.ResidualHeadContractError, match="registry must be empty"):
        residual.initialize_residual_head(module_extra, 1729)

    hooked = residual.PhasePairResidualHead(device="cpu")
    hooked.register_forward_hook(lambda *_args: None)
    with pytest.raises(residual.ResidualHeadContractError, match="registry must be empty"):
        residual.initialize_residual_head(hooked, 1729)


def test_14_live_parameter_identity_and_cross_head_ownership_are_closed() -> None:
    first = _initialized()
    second = residual.PhasePairResidualHead(device="cpu")
    for name in ("fc1_weight", "fc1_bias", "fc2_weight", "fc2_bias"):
        setattr(second, name, getattr(first, name))
    with pytest.raises(residual.ResidualHeadContractError, match="already owned"):
        residual.initialize_residual_head(second, 1729)

    storage_alias = residual.PhasePairResidualHead(device="cpu")
    storage_alias.fc1_weight = torch.nn.Parameter(first.fc1_weight.detach())
    with pytest.raises(residual.ResidualHeadContractError, match="storage.*owned"):
        residual.initialize_residual_head(storage_alias, 1729)

    first.eval()
    first.fc1_weight = torch.nn.Parameter(torch.zeros_like(first.fc1_weight))
    tokens, mask = _tokens("01_GENERIC")
    with pytest.raises(residual.ResidualHeadContractError, match="ownership"):
        first(tokens, system="01_GENERIC", active_mask=mask)

    storage_bound = _initialized()
    storage_bound.eval()
    parameter = storage_bound.fc1_weight
    old_storage_id = int(parameter.untyped_storage()._cdata)
    old_pointer = parameter.untyped_storage().data_ptr()
    alias = torch.frombuffer(
        memoryview(parameter.detach().numpy()), dtype=torch.float32
    ).reshape_as(parameter)
    assert alias.untyped_storage().data_ptr() == old_pointer
    assert int(alias.untyped_storage()._cdata) != old_storage_id
    with torch.no_grad():
        parameter.data = alias
    with pytest.raises(residual.ResidualHeadContractError, match="ownership"):
        storage_bound(tokens, system="01_GENERIC", active_mask=mask)

    current_owner = _initialized()
    with torch.no_grad():
        current_owner.fc1_weight.data = torch.empty_like(
            current_owner.fc1_weight
        )
    current_alias = residual.PhasePairResidualHead(device="cpu")
    current_alias.fc1_weight = torch.nn.Parameter(
        current_owner.fc1_weight.detach()
    )
    with pytest.raises(residual.ResidualHeadContractError, match="storage.*owned"):
        residual.initialize_residual_head(current_alias, 1729)


def test_15_wamo_and_interedit_derived_token_laws_fail_closed() -> None:
    model = _initialized()
    model.eval()
    text = _normalized(2)

    wamo, wamo_mask = _tokens("02_WAMO_MARGINAL_WAVELET")
    wamo[:, 0, 0, 0:2] = 0.0
    wamo[:, 0, 0, 2] = 0.0
    wamo[:, 0, 0, 3] = -0.0
    wamo[:, 0, 0, 4] = 0.0
    with pytest.raises(residual.ResidualHeadContractError, match=r"exact \+0"):
        residual.residual_ordered_scores(
            model,
            wamo,
            wamo_mask,
            text,
            "02_WAMO_MARGINAL_WAVELET",
            None,
            None,
        )

    bad_wamo, bad_wamo_mask = _tokens("02_WAMO_MARGINAL_WAVELET")
    bad_wamo[:, 0, 0, 3] += 0.5
    with pytest.raises(residual.ResidualHeadContractError, match="difference law"):
        residual.residual_ordered_scores(
            model,
            bad_wamo,
            bad_wamo_mask,
            text,
            "02_WAMO_MARGINAL_WAVELET",
            None,
            None,
        )

    bad_wamo_u, bad_wamo_u_mask = _tokens("02_WAMO_MARGINAL_WAVELET")
    bad_wamo_u[:, 0, 0, 2] = 7.0
    with pytest.raises(residual.ResidualHeadContractError, match="combined marginal"):
        residual.residual_ordered_scores(
            model,
            bad_wamo_u,
            bad_wamo_u_mask,
            text,
            "02_WAMO_MARGINAL_WAVELET",
            None,
            None,
        )

    negative_zero_mean, negative_zero_mean_mask = _tokens(
        "02_WAMO_MARGINAL_WAVELET"
    )
    combined = math.log(math.exp(1.0) + math.exp(-1.0) - 1.0e-12)
    negative_zero_mean[0, 0, 0, :5] = torch.tensor(
        [1.0, -1.0, combined, 2.0, -0.0], dtype=torch.float32
    )
    negative_zero_mean[1, 0, 0, :5] = torch.tensor(
        [-1.0, 1.0, combined, 2.0, -0.0], dtype=torch.float32
    )
    with pytest.raises(residual.ResidualHeadContractError, match=r"exact \+0"):
        residual.residual_ordered_scores(
            model,
            negative_zero_mean,
            negative_zero_mean_mask,
            text,
            "02_WAMO_MARGINAL_WAVELET",
            None,
            None,
        )

    below_floor, below_floor_mask = _tokens("02_WAMO_MARGINAL_WAVELET")
    minimum = torch.tensor(math.log(1.0e-12), dtype=torch.float32)
    below = torch.nextafter(minimum, torch.tensor(float("-inf"))).item()
    below_combined = math.log(2.0 * math.exp(below) - 1.0e-12)
    below_floor[:, 0, 0, :5] = torch.tensor(
        [below, below, below_combined, 0.0, below], dtype=torch.float32
    )
    with pytest.raises(residual.ResidualHeadContractError, match="below log"):
        residual.residual_ordered_scores(
            model,
            below_floor,
            below_floor_mask,
            text,
            "02_WAMO_MARGINAL_WAVELET",
            None,
            None,
        )

    at_floor, at_floor_mask = _tokens("02_WAMO_MARGINAL_WAVELET")
    floor = minimum.item()
    floor_combined = math.log(2.0 * math.exp(floor) - 1.0e-12)
    at_floor[:, 0, 0, :5] = torch.tensor(
        [floor, floor, floor_combined, 0.0, floor], dtype=torch.float32
    )
    floor_scores = residual.residual_ordered_scores(
        model,
        at_floor,
        at_floor_mask,
        text,
        "02_WAMO_MARGINAL_WAVELET",
        None,
        None,
    )
    assert torch.isfinite(floor_scores).all()

    extreme_wamo, extreme_wamo_mask = _tokens("02_WAMO_MARGINAL_WAVELET")
    maximum = torch.finfo(torch.float32).max
    extreme_wamo[:, 0, 0, 0] = maximum
    extreme_wamo[:, 0, 0, 1] = maximum
    extreme_wamo[:, 0, 0, 2] = -maximum
    extreme_wamo[:, 0, 0, 3] = 0.0
    extreme_wamo[:, 0, 0, 4] = maximum
    with pytest.raises(residual.ResidualHeadContractError, match="combined marginal"):
        residual.residual_ordered_scores(
            model,
            extreme_wamo,
            extreme_wamo_mask,
            text,
            "02_WAMO_MARGINAL_WAVELET",
            None,
            None,
        )

    interedit, interedit_mask = _tokens("03_INTEREDIT_MEAN_DIFFERENCE_DCT")
    interedit_mask[:, 0, 0] = False
    interedit[:, 0, 0, :7] = 0.0
    with pytest.raises(residual.ResidualHeadContractError, match="must be paired"):
        residual.residual_ordered_scores(
            model,
            interedit,
            interedit_mask,
            text,
            "03_INTEREDIT_MEAN_DIFFERENCE_DCT",
            None,
            None,
        )

    bad_interedit, bad_interedit_mask = _tokens(
        "03_INTEREDIT_MEAN_DIFFERENCE_DCT"
    )
    bad_interedit[:, 0, 1, 1] += 0.5
    with pytest.raises(residual.ResidualHeadContractError, match="token field law"):
        residual.residual_ordered_scores(
            model,
            bad_interedit,
            bad_interedit_mask,
            text,
            "03_INTEREDIT_MEAN_DIFFERENCE_DCT",
            None,
            None,
        )

    bad_interedit_total, bad_interedit_total_mask = _tokens(
        "03_INTEREDIT_MEAN_DIFFERENCE_DCT"
    )
    for slot in (0, 1):
        bad_interedit_total[:, 0, slot, 3] = 7.0
        bad_interedit_total[:, 0, slot, 4] = (
            bad_interedit_total[:, 0, slot, 0] - 7.0
        )
    with pytest.raises(residual.ResidualHeadContractError, match="total-energy"):
        residual.residual_ordered_scores(
            model,
            bad_interedit_total,
            bad_interedit_total_mask,
            text,
            "03_INTEREDIT_MEAN_DIFFERENCE_DCT",
            None,
            None,
        )

    extreme_interedit, extreme_interedit_mask = _tokens(
        "03_INTEREDIT_MEAN_DIFFERENCE_DCT"
    )
    for slot in (0, 1):
        extreme_interedit[:, 0, slot, 0:3] = maximum
        extreme_interedit[:, 0, slot, 3] = -maximum
        extreme_interedit[:, 0, slot, 4] = maximum
    with pytest.raises(residual.ResidualHeadContractError, match="total-energy"):
        residual.residual_ordered_scores(
            model,
            extreme_interedit,
            extreme_interedit_mask,
            text,
            "03_INTEREDIT_MEAN_DIFFERENCE_DCT",
            None,
            None,
        )


def test_16_parameter_name_requires_exact_str() -> None:
    class NameSubclass(str):
        pass

    with pytest.raises(residual.ResidualHeadContractError, match="exact str"):
        residual.canonical_residual_parameter_bytes(
            NameSubclass("residual.fc1.weight"), 1729
        )


def test_17_canonical_head_class_rebinding_is_blocked() -> None:
    original_call = residual.PhasePairResidualHead.__call__
    original_forward = residual.PhasePairResidualHead.forward
    with pytest.raises(residual.ResidualHeadContractError, match="class is sealed"):
        setattr(
            residual.PhasePairResidualHead,
            "__call__",
            lambda *_args, **_kwargs: torch.tensor([123.0]),
        )
    with pytest.raises(residual.ResidualHeadContractError, match="class is sealed"):
        setattr(
            residual.PhasePairResidualHead,
            "forward",
            lambda *_args, **_kwargs: torch.tensor([123.0]),
        )
    assert residual.PhasePairResidualHead.__call__ is original_call
    assert residual.PhasePairResidualHead.forward is original_forward
    uninitialized = residual.PhasePairResidualHead(device="cpu")
    with pytest.raises(residual.ResidualHeadContractError, match="shadowed"):
        uninitialized.forward = lambda *_args, **_kwargs: torch.tensor([123.0])
    uninitialized.eval()
    tokens, mask = _tokens("01_GENERIC")
    with pytest.raises(residual.ResidualHeadHold):
        uninitialized(tokens, system="01_GENERIC", active_mask=mask)


def test_18_full_delay_signed_valid_and_antiphase_laws() -> None:
    model = _initialized()
    model.eval()
    text = _normalized(2)

    bad_delay, bad_delay_mask = _tokens("06_PHASEPAIR_FULL")
    bad_delay[0, 0, 0, 5] = 2.0
    bad_delay[1, 0, 0, 5] = -2.0
    with pytest.raises(residual.ResidualHeadContractError, match="delay"):
        residual.residual_ordered_scores(
            model,
            bad_delay,
            bad_delay_mask,
            text,
            "06_PHASEPAIR_FULL",
            None,
            None,
        )

    unsigned, unsigned_mask = _tokens("06_PHASEPAIR_FULL")
    unsigned[:, 0, 0, 6] = 0.0
    with pytest.raises(residual.ResidualHeadContractError, match=r"exact \+0"):
        residual.residual_ordered_scores(
            model,
            unsigned,
            unsigned_mask,
            text,
            "06_PHASEPAIR_FULL",
            None,
            None,
        )

    off_circle, off_circle_mask = _tokens("06_PHASEPAIR_FULL")
    off_circle[:, 0, 0, 3] = 0.0
    off_circle[0, 0, 0, 4:6] = 0.0
    off_circle[1, 0, 0, 4:6] = -0.0
    with pytest.raises(residual.ResidualHeadContractError, match="unit circle"):
        residual.residual_ordered_scores(
            model,
            off_circle,
            off_circle_mask,
            text,
            "06_PHASEPAIR_FULL",
            None,
            None,
        )

    wrong_delay, wrong_delay_mask = _tokens("06_PHASEPAIR_FULL")
    wrong_delay[0, 0, 0, 5] = 0.125
    wrong_delay[1, 0, 0, 5] = -0.125
    with pytest.raises(residual.ResidualHeadContractError, match="inconsistent"):
        residual.residual_ordered_scores(
            model,
            wrong_delay,
            wrong_delay_mask,
            text,
            "06_PHASEPAIR_FULL",
            None,
            None,
        )

    wrong_unsigned, wrong_unsigned_mask = _tokens("06_PHASEPAIR_FULL")
    wrong_unsigned[:, 0, 0, 3] = 1.0
    wrong_unsigned[:, 0, 0, 4:6] = 0.0
    wrong_unsigned[:, 0, 0, 6] = 0.0
    with pytest.raises(residual.ResidualHeadContractError, match="anti-phase"):
        residual.residual_ordered_scores(
            model,
            wrong_unsigned,
            wrong_unsigned_mask,
            text,
            "06_PHASEPAIR_FULL",
            None,
            None,
        )

    antiphase, antiphase_mask = _tokens("06_PHASEPAIR_FULL")
    antiphase[:, 0, 0, 3] = -1.0
    antiphase[:, 0, 0, 4:6] = 0.0
    antiphase[:, 0, 0, 6] = 0.0
    scores = residual.residual_ordered_scores(
        model,
        antiphase,
        antiphase_mask,
        text,
        "06_PHASEPAIR_FULL",
        None,
        None,
    )
    assert scores.shape == (2, 2, 2)
    assert torch.isfinite(scores).all()


def test_19_parameter_hooks_and_builtin_sum_rebinding_are_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _initialized()
    model.eval()
    regular_hook = model.fc1_weight.register_hook(
        lambda gradient: torch.zeros_like(gradient)
    )
    tokens, mask = _tokens("01_GENERIC")
    with pytest.raises(residual.ResidualHeadContractError, match="hooks"):
        model(tokens, system="01_GENERIC", active_mask=mask)
    regular_hook.remove()
    model.fc1_weight._backward_hooks = None

    hidden_post_model = model
    hidden_post_model.eval()
    hidden_parameter = hidden_post_model.fc1_weight
    canonical_registry = hidden_parameter._post_accumulate_grad_hooks
    assert canonical_registry is not None and len(canonical_registry) == 0
    post_hook_calls: list[bool] = []

    def hostile_post_hook(parameter: torch.Tensor) -> None:
        post_hook_calls.append(True)
        if parameter.grad is not None:
            parameter.grad.zero_()

    hidden_parameter.register_post_accumulate_grad_hook(hostile_post_hook)
    with pytest.raises(residual.ResidualHeadContractError, match="hooks"):
        hidden_post_model(tokens, system="01_GENERIC", active_mask=mask)
    hidden_parameter._post_accumulate_grad_hooks = None
    with pytest.raises(residual.ResidualHeadContractError, match="hook registries"):
        hidden_post_model(tokens, system="01_GENERIC", active_mask=mask)

    canonical_registry.clear()
    hidden_parameter._post_accumulate_grad_hooks = canonical_registry
    output = hidden_post_model(tokens, system="01_GENERIC", active_mask=mask)
    output.sum().backward()
    assert post_hook_calls == []

    baseline = residual.canonical_head_dropout_receipt(1729, 0, 2)
    monkeypatch.setattr(builtins, "sum", lambda _values, _start=0: 6144)
    assert residual.canonical_head_dropout_receipt(1729, 0, 2) == baseline


def test_20_all_active_non_generic_tokens_enforce_exact_log_energy_floor() -> None:
    model = _initialized()
    model.eval()
    text = _normalized(2)
    minimum = torch.tensor(math.log(1.0e-12), dtype=torch.float32)
    below = torch.nextafter(minimum, torch.tensor(float("-inf"))).item()

    for system in (
        "04_NO_RELATION",
        "05_PHASE_STRIPPED",
        "06_PHASEPAIR_FULL",
    ):
        tokens, mask = _tokens(system)
        tokens[:, 0, 0, :2] = below
        with pytest.raises(residual.ResidualHeadContractError, match="below log"):
            residual.residual_ordered_scores(
                model,
                tokens,
                mask,
                text,
                system,
                None,
                None,
            )


def test_21_module_builtin_rebinding_cannot_open_invalid_input_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _initialized()
    model.eval()
    text = _normalized(2)
    valid_tokens, mask = _tokens("06_PHASEPAIR_FULL")
    baseline_scores = residual.residual_ordered_scores(
        model,
        valid_tokens,
        mask,
        text,
        "06_PHASEPAIR_FULL",
        None,
        None,
    )
    baseline_parameter = residual.canonical_residual_parameter_bytes(
        "residual.fc1.weight", 1729
    )
    error_type = residual.ResidualHeadContractError

    def bomb(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("rebound module global was read")

    for name in (
        "AssertionError",
        "BaseException",
        "TypeError",
        "UnicodeDecodeError",
        "any",
        "bool",
        "bytearray",
        "bytes",
        "dict",
        "float",
        "int",
        "len",
        "list",
        "min",
        "next",
        "range",
        "set",
        "sorted",
        "str",
        "tuple",
        "type",
        "zip",
    ):
        monkeypatch.setattr(residual, name, bomb, raising=False)
    monkeypatch.setattr(residual, "ResidualHeadContractError", RuntimeError)
    monkeypatch.setattr(residual, "ResidualHeadHold", RuntimeError)

    assert residual.canonical_residual_parameter_bytes(
        "residual.fc1.weight", 1729
    ) == baseline_parameter
    assert torch.equal(
        residual.residual_ordered_scores(
            model,
            valid_tokens,
            mask,
            text,
            "06_PHASEPAIR_FULL",
            None,
            None,
        ),
        baseline_scores,
    )

    invalid_tokens = valid_tokens.clone()
    invalid_tokens[:, 0, 0, :2] = float("nan")
    with pytest.raises(error_type, match="finite"):
        residual.residual_ordered_scores(
            model,
            invalid_tokens,
            mask,
            text,
            "06_PHASEPAIR_FULL",
            None,
            None,
        )


def test_22_exact_tensor_detach_shadow_cannot_replace_snapshot_source() -> None:
    model = _initialized()
    model.eval()
    valid_tokens, mask = _tokens("06_PHASEPAIR_FULL")
    invalid_tokens = valid_tokens.clone()
    invalid_tokens[:, 0, 0, :2] = float("nan")
    invalid_tokens.detach = lambda: valid_tokens

    assert type(invalid_tokens) is torch.Tensor
    with pytest.raises(residual.ResidualHeadContractError, match="finite"):
        residual.residual_ordered_scores(
            model,
            invalid_tokens,
            mask,
            _normalized(2),
            "06_PHASEPAIR_FULL",
            None,
            None,
        )

    strided_tokens = torch.empty((*valid_tokens.shape, 2), dtype=torch.float32)[
        ..., 0
    ]
    strided_tokens.copy_(valid_tokens)
    assert not torch.Tensor.is_contiguous(strided_tokens)
    strided_tokens.is_contiguous = lambda: True
    with pytest.raises(residual.ResidualHeadContractError, match=r"float32 C"):
        residual.residual_ordered_scores(
            model,
            strided_tokens,
            mask,
            _normalized(2),
            "06_PHASEPAIR_FULL",
            None,
            None,
        )


def test_23_parameter_storage_method_shadows_cannot_bypass_live_owner() -> None:
    owner = residual.PhasePairResidualHead(device="cpu")
    owner_receipt = residual.initialize_residual_head(owner, 1729)
    owner_before = torch.Tensor.clone(torch.Tensor.detach(owner.fc1_weight))

    instance_shadow = residual.PhasePairResidualHead(device="cpu")
    instance_shadow.fc1_weight = torch.nn.Parameter(
        torch.Tensor.detach(owner.fc1_weight)
    )
    decoy = torch.empty_like(instance_shadow.fc1_weight)
    instance_shadow.fc1_weight.untyped_storage = decoy.untyped_storage
    instance_shadow.fc1_weight.storage_offset = lambda: 0
    instance_shadow.fc1_weight.numel = lambda: 0
    instance_shadow.fc1_weight.is_contiguous = lambda: True
    with pytest.raises(residual.ResidualHeadContractError, match="storage.*owned"):
        residual.initialize_residual_head(instance_shadow, 1729)

    storage_shadow = residual.PhasePairResidualHead(device="cpu")
    storage_shadow.fc1_weight = torch.nn.Parameter(
        torch.Tensor.detach(owner.fc1_weight)
    )
    shared_storage = torch.Tensor.untyped_storage(storage_shadow.fc1_weight)
    decoy_storage = torch.Tensor.untyped_storage(
        torch.empty_like(storage_shadow.fc1_weight)
    )
    shared_storage.data_ptr = decoy_storage.data_ptr
    with pytest.raises(residual.ResidualHeadContractError, match="storage.*owned"):
        residual.initialize_residual_head(storage_shadow, 1729)

    assert torch.equal(owner.fc1_weight, owner_before)
    assert residual.verify_residual_head_initializer_receipt(
        owner_receipt,
        owner,
        1729,
    ) == hashlib.sha256(owner_receipt).digest()
