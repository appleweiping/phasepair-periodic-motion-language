from __future__ import annotations

import gc
import inspect
import struct
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phasepair_core import batching, contracts, dropout, runtime_dropout  # noqa: E402
from phasepair_core import torch_models as models  # noqa: E402


EXPECTED = {
    "mime": (244, 35_111_936),
    "early": (69, 13_014_016),
    "late": (142, 26_017_280),
}


def _batch(*, swapped: bool = False) -> batching.PreparedMotionBatch:
    actor_a = np.zeros((2, 3, 262), dtype=np.float32)
    actor_b = np.zeros((2, 3, 262), dtype=np.float32)
    relation_ab = np.zeros((2, 3, 799), dtype=np.float32)
    relation_ba = np.zeros((2, 3, 799), dtype=np.float32)
    actor_a[0, :2] = np.float32(0.25)
    actor_b[0, :2] = np.float32(-0.5)
    relation_ab[0, :2] = np.float32(0.125)
    relation_ba[0, :2] = np.float32(-0.25)
    actor_a[1, :3] = np.float32(0.75)
    actor_b[1, :3] = np.float32(-1.0)
    relation_ab[1, :3] = np.float32(0.375)
    relation_ba[1, :3] = np.float32(-0.5)
    if swapped:
        actor_a, actor_b = actor_b, actor_a
        relation_ab, relation_ba = relation_ba, relation_ab
    return batching.PreparedMotionBatch(
        actor_a,
        actor_b,
        relation_ab,
        relation_ba,
        np.array([[1, 1, 0], [1, 1, 1]], dtype=np.uint8),
        (2, 3),
        (b"A" * 32, b"B" * 32),
    )


def _fill_synthetic_state(model: models._MotionEncoderBase) -> None:
    rows = {row.name: row for row in contracts.canonical_motion_rows(model.architecture)}
    with torch.no_grad():
        for name, parameter in model.canonical_named_parameters():
            semantic_class = rows[name].semantic_class
            if semantic_class == contracts.LAYERNORM_GAMMA:
                parameter.fill_(1.0)
            elif semantic_class in {
                contracts.EMBEDDING_WEIGHT,
                contracts.QUERY_OR_OTHER_WEIGHT,
            }:
                values = torch.linspace(
                    -0.01,
                    0.01,
                    parameter.numel(),
                    dtype=torch.float32,
                    device=parameter.device,
                )
                parameter.copy_(values.reshape(parameter.shape))
            elif semantic_class == contracts.MATRIX_WEIGHT:
                parameter.zero_()
                checksum = sum(name.encode("ascii")) % 13
                parameter.diagonal().fill_((checksum + 1) * 1e-3)
            else:
                parameter.zero_()


class _BombBundle:
    def __getattribute__(self, name: str):
        raise AssertionError(f"eval read dropout bundle field {name}")


class _AllZeroBundle:
    architecture = "early"
    batch_size = 2
    time_steps = 3

    def __init__(self) -> None:
        self.take_calls = 0
        self.complete_calls = 0

    def take(self, site_ordinal: int, *, like: torch.Tensor) -> torch.Tensor:
        self.take_calls += 1
        return torch.zeros_like(like)

    def assert_complete(self) -> None:
        self.complete_calls += 1


class _BundleWrapper:
    def __init__(
        self, inner: runtime_dropout.OneForwardCanonicalDropoutBundle
    ) -> None:
        self.inner = inner

    @property
    def architecture(self) -> str:
        return self.inner.architecture

    @property
    def batch_size(self) -> int:
        return self.inner.batch_size

    @property
    def time_steps(self) -> int:
        return self.inner.time_steps

    def take(self, site_ordinal: int, *, like: torch.Tensor) -> torch.Tensor:
        return self.inner.take(site_ordinal, like=like)

    def assert_complete(self) -> None:
        return self.inner.assert_complete()


def _canonical_bundle(
    architecture: str,
    *,
    batch_size: int = 2,
    time_steps: int = 3,
) -> runtime_dropout.OneForwardCanonicalDropoutBundle:
    return runtime_dropout.OneForwardCanonicalDropoutBundle(
        architecture=architecture,
        seed=1729,
        epoch_index=0,
        global_optimizer_step=0,
        batch_size=batch_size,
        time_steps=time_steps,
        policy=dropout.RuntimePolicy(),
    )


def _mutant_take(mode: str):
    def take(
        self: runtime_dropout.OneForwardCanonicalDropoutBundle,
        site_ordinal: int,
        *,
        like: torch.Tensor,
    ) -> torch.Tensor:
        del self, site_ordinal
        shape = tuple(like.shape)
        if mode == "bits":
            return torch.ones(shape, dtype=torch.float32, device=like.device)
        if mode == "layout":
            wide = torch.zeros(
                (*shape[:-1], shape[-1] * 2),
                dtype=torch.float32,
                device=like.device,
            )
            return wide[..., ::2]
        if mode == "grad":
            return torch.zeros(
                shape,
                dtype=torch.float32,
                device=like.device,
                requires_grad=True,
            )
        if mode == "device":
            return torch.empty(shape, dtype=torch.float32, device="meta")
        if mode == "shape":
            return torch.empty((0,), dtype=torch.float32, device=like.device)
        raise AssertionError("unknown mask mutant")

    return take


def test_meta_parameter_inventories_match_frozen_motion_ledgers() -> None:
    for architecture, (tensor_count, numel) in EXPECTED.items():
        model = models.build_motion_encoder(architecture, device="meta")
        audit = model.audit_meta_inventory()
        expected_rows = contracts.canonical_motion_rows(architecture)
        assert audit.architecture == architecture
        assert audit.tensor_count == tensor_count
        assert audit.numel == numel
        assert audit.canonical_names == tuple(row.name for row in expected_rows)
        assert audit.meta_device is True
        assert len({id(parameter) for parameter in model.parameters()}) == tensor_count
        assert tuple(model.position_encoding.shape) == (300, 512)
        assert model.position_encoding.device.type == "meta"
        assert not model.position_encoding.requires_grad


def test_motion_embedding_and_parameter_inventory_use_type_primitives() -> None:
    def bomb(*_args: object, **_kwargs: object) -> object:
        raise AssertionError("shadowed Tensor or storage instance method was called")

    ordered = torch.zeros((2, 1, 512), dtype=torch.float32)
    ordered.detach = bomb
    ordered.is_contiguous = bomb
    assert models.MotionEmbeddings(ordered).ordered is ordered

    parameter = torch.nn.Parameter(torch.zeros((2, 3), dtype=torch.float32))
    storage = torch.Tensor.untyped_storage(parameter)
    for name in (
        "is_contiguous",
        "untyped_storage",
        "storage_offset",
        "numel",
        "element_size",
    ):
        setattr(parameter, name, bomb)
    storage.data_ptr = bomb
    storage.nbytes = bomb
    assert models._check_inventory_parameter(
        parameter,
        name="probe.weight",
        shape=(2, 3),
        require_meta=False,
    ) == 6


def test_parameter_inventory_rejects_real_shared_storage_despite_decoys() -> None:
    model = models.EarlyFusionMotionEncoder(device="cpu")
    by_name = dict(model.canonical_named_parameters())
    owner = by_name["tmr.input.linear.bias"]
    alias = by_name["tmr.input.ln.weight"]
    assert tuple(owner.shape) == tuple(alias.shape) == (512,)
    with torch.no_grad():
        alias.data = owner.data
    decoy = torch.nn.Parameter(torch.empty_like(alias))
    alias.untyped_storage = decoy.untyped_storage
    alias.storage_offset = lambda: 0
    alias.numel = lambda: 512
    alias.element_size = lambda: 4
    actual_storage = torch.Tensor.untyped_storage(alias)
    actual_storage.data_ptr = decoy.untyped_storage().data_ptr
    actual_storage.nbytes = decoy.untyped_storage().nbytes

    with pytest.raises(
        models.TorchModelContractError,
        match="storage identity repeated|storage ranges overlap",
    ):
        model._validate_parameter_inventory(require_meta=False)
    del model, by_name, owner, alias, decoy, actual_storage
    gc.collect()


def test_inventory_interval_check_rejects_distinct_overlapping_storages() -> None:
    backing = bytearray(24)
    owner = torch.nn.Parameter(
        torch.frombuffer(memoryview(backing)[0:16], dtype=torch.float32)
    )
    overlap = torch.nn.Parameter(
        torch.frombuffer(memoryview(backing)[8:24], dtype=torch.float32)
    )
    identities: set[tuple[str, int | None, int, int]] = set()
    intervals: list[tuple[str, int | None, int, int]] = []
    assert models._check_inventory_parameter(
        owner,
        name="owner",
        shape=(4,),
        require_meta=False,
        storage_identities=identities,
        storage_intervals=intervals,
    ) == 4

    decoy = torch.nn.Parameter(torch.zeros((4,), dtype=torch.float32))
    overlap.untyped_storage = decoy.untyped_storage
    overlap.storage_offset = lambda: 0
    overlap.numel = lambda: 4
    overlap.element_size = lambda: 4
    overlap.is_contiguous = lambda: True
    overlap_storage = torch.Tensor.untyped_storage(overlap)
    overlap_storage.data_ptr = decoy.untyped_storage().data_ptr
    overlap_storage.nbytes = decoy.untyped_storage().nbytes
    with pytest.raises(
        models.TorchModelContractError,
        match="storage ranges overlap",
    ):
        models._check_inventory_parameter(
            overlap,
            name="overlap",
            shape=(4,),
            require_meta=False,
            storage_identities=identities,
            storage_intervals=intervals,
        )


def test_dropout_mask_snapshot_closure_captures_type_primitives() -> None:
    captured = inspect.getclosurevars(models._DropoutRuntime.apply).nonlocals
    assert captured["tensor_type"] is torch.Tensor
    assert captured["tensor_detach"] is torch.Tensor.detach
    assert captured["tensor_clone"] is torch.Tensor.clone
    assert captured["tensor_is_contiguous"] is torch.Tensor.is_contiguous


def test_custom_topology_never_calls_framework_reset_parameters() -> None:
    with (
        mock.patch.object(
            torch.nn.Linear,
            "reset_parameters",
            side_effect=AssertionError("framework Linear initializer called"),
        ),
        mock.patch.object(
            torch.nn.LayerNorm,
            "reset_parameters",
            side_effect=AssertionError("framework LayerNorm initializer called"),
        ),
        mock.patch.object(
            torch.nn.MultiheadAttention,
            "_reset_parameters",
            side_effect=AssertionError("framework attention initializer called"),
        ),
    ):
        for architecture in EXPECTED:
            model = models.build_motion_encoder(architecture, device="meta")
            assert not any(
                isinstance(
                    module,
                    (torch.nn.Linear, torch.nn.LayerNorm, torch.nn.MultiheadAttention),
                )
                for module in model.modules()
            )
    source = inspect.getsource(models)
    assert "torch_functional.dropout" not in source
    assert "nn.Dropout" not in source


@pytest.mark.parametrize("architecture", tuple(EXPECTED))
def test_train_mode_without_external_bundle_is_an_exact_hold(
    architecture: str,
) -> None:
    model = models.build_motion_encoder(architecture, device="meta").train()
    with pytest.raises(models.TorchModelContractHold) as caught:
        model(_batch())
    assert caught.value.code == models.DROPOUT_HOLD_CODE
    assert str(caught.value) == models.DROPOUT_HOLD_CODE


def test_train_rejects_allzero_fake_wrapper_and_rebound_public_type() -> None:
    meta_model = models.build_motion_encoder("early", device="meta").train()
    allzero = _AllZeroBundle()
    inner = _canonical_bundle("early")
    wrapper = _BundleWrapper(inner)
    for supplied in (allzero, wrapper):
        with pytest.raises(
            models.TorchModelContractError,
            match="exact reviewed one-forward bundle",
        ):
            meta_model(_batch(), dropout_bundle=supplied)  # type: ignore[arg-type]
    assert allzero.take_calls == 0
    assert allzero.complete_calls == 0
    assert inner._lifecycle == 0
    with mock.patch.object(models, "OneForwardCanonicalDropoutBundle", _AllZeroBundle):
        with pytest.raises(
            models.TorchModelContractError,
            match="exact reviewed one-forward bundle",
        ):
            meta_model(_batch(), dropout_bundle=_AllZeroBundle())  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("architecture", "site_count"),
    (("mime", 49), ("early", 16), ("late", 32)),
)
def test_train_bundle_consumes_exact_canonical_traversal(
    architecture: str,
    site_count: int,
) -> None:
    model = models.build_motion_encoder(architecture, device="cpu").train()
    _fill_synthetic_state(model)
    bundle = _canonical_bundle(architecture)
    with torch.no_grad():
        output = model(_batch(), dropout_bundle=bundle).ordered
    assert output.shape == (2, 2, 512)
    assert bundle._lifecycle == 2 * site_count + 1
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.assert_complete()
    del model
    gc.collect()


@pytest.mark.parametrize("mode", ("bits", "layout", "grad", "device", "shape"))
def test_training_ignores_rebound_bundle_take_mutants(mode: str) -> None:
    model = models.EarlyFusionMotionEncoder(device="cpu").train()
    _fill_synthetic_state(model)
    bundle = _canonical_bundle("early")
    with mock.patch.object(
        runtime_dropout.OneForwardCanonicalDropoutBundle,
        "take",
        _mutant_take(mode),
    ), torch.no_grad():
        output = model(_batch(), dropout_bundle=bundle).ordered
    assert output.shape == (2, 2, 512)
    assert bool(torch.isfinite(output).all().item())
    assert bundle._lifecycle == 33
    del model
    gc.collect()


def test_dropout_runtime_uses_captured_original_take_for_multiplication() -> None:
    source = _canonical_bundle("early", batch_size=1, time_steps=1)
    expected_mask = source.take(
        0,
        like=torch.empty((2, 1, 4, 1, 1), dtype=torch.float32),
    )
    bundle = _canonical_bundle("early", batch_size=1, time_steps=1)
    runtime = models._DropoutRuntime("early", 1, 1, bundle)
    value = torch.ones((2, 1, 4, 1, 1), dtype=torch.float32)
    with mock.patch.object(
        runtime_dropout.OneForwardCanonicalDropoutBundle,
        "take",
        _mutant_take("bits"),
    ):
        output = runtime.apply(0, value)
    assert torch.equal(output, expected_mask)
    assert bundle._lifecycle == 2


def test_public_equation_globals_cannot_change_attention_or_keep_law() -> None:
    model = models.EarlyFusionMotionEncoder(device="cpu").eval()
    _fill_synthetic_state(model)
    with torch.no_grad():
        baseline = model(_batch()).ordered
    with (
        mock.patch.multiple(
            models,
            VECTOR_WIDTH=1,
            HEAD_COUNT=8,
            HEAD_WIDTH=64,
            FFN_WIDTH=1,
            LAYER_NORM_EPS=1.0,
        ),
        mock.patch.object(dropout, "KEEP_FLOAT32_BITS", 0x3F800000),
    ):
        meta = models.build_motion_encoder("early", device="meta").audit_meta_inventory()
        assert (meta.tensor_count, meta.numel) == (69, 13_014_016)
        with torch.no_grad():
            rebound = model(_batch()).ordered
        assert torch.equal(rebound, baseline)
        train_runtime = models._DropoutRuntime(
            "early", 2, 3, _canonical_bundle("early")
        )
        source = _canonical_bundle("early")
        expected_mask = source.take(
            0, like=torch.ones((2, 2, 4, 3, 3), dtype=torch.float32)
        )
        with mock.patch.object(
            runtime_dropout.OneForwardCanonicalDropoutBundle,
            "take",
            _mutant_take("bits"),
        ):
            applied = train_runtime.apply(
                0, torch.ones((2, 2, 4, 3, 3), dtype=torch.float32)
            )
        assert torch.equal(applied, expected_mask)
    del model
    gc.collect()


def test_runtime_bundle_and_helper_rebinds_cannot_change_training_output() -> None:
    model = models.EarlyFusionMotionEncoder(device="cpu").train()
    _fill_synthetic_state(model)
    baseline_bundle = _canonical_bundle("early")
    with torch.no_grad():
        baseline = model(_batch(), dropout_bundle=baseline_bundle).ordered

    class _FakeRuntime:
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise AssertionError("dynamically rebound runtime was constructed")

    def noop_complete(
        self: runtime_dropout.OneForwardCanonicalDropoutBundle,
    ) -> None:
        del self
        return None

    forged = {
        "_architecture": "late",
        "_seed": 2718,
        "_epoch_index": 1,
        "_global_optimizer_step": 44,
        "_batch_size": 128,
        "_time_steps": 300,
        "_lifecycle": 999,
        "_transition_lock": object(),
    }

    def forged_getattribute(
        self: runtime_dropout.OneForwardCanonicalDropoutBundle,
        name: str,
    ) -> object:
        if name in forged:
            return forged[name]
        return object.__getattribute__(self, name)

    rebound_bundle = _canonical_bundle("early")
    with mock.patch.multiple(
        models,
        _DropoutRuntime=_FakeRuntime,
        _make_dropout_runtime_class=lambda *args, **kwargs: _FakeRuntime,
        _make_motion_runtime_builder=lambda *args, **kwargs: _FakeRuntime,
        create=True,
    ), mock.patch.multiple(
        runtime_dropout.OneForwardCanonicalDropoutBundle,
        take=_mutant_take("bits"),
        assert_complete=noop_complete,
        architecture=property(lambda self: "late"),
        batch_size=property(lambda self: 128),
        time_steps=property(lambda self: 300),
    ), mock.patch.object(
        runtime_dropout.OneForwardCanonicalDropoutBundle,
        "__getattribute__",
        forged_getattribute,
    ), torch.no_grad():
        rebound = model(_batch(), dropout_bundle=rebound_bundle).ordered
    assert torch.equal(rebound, baseline)
    assert baseline_bundle._lifecycle == 33
    assert rebound_bundle._lifecycle == 33
    del model
    gc.collect()


@pytest.mark.parametrize("architecture", tuple(EXPECTED))
def test_eval_p2_forward_is_finite_l2_and_swaps_ordered_slices(
    architecture: str,
) -> None:
    model = models.build_motion_encoder(architecture, device="cpu").eval()
    _fill_synthetic_state(model)
    with torch.no_grad():
        original = model(_batch(), dropout_bundle=_BombBundle()).ordered
        swapped = model(_batch(swapped=True), dropout_bundle=_BombBundle()).ordered
    assert original.shape == (2, 2, 512)
    assert original.dtype == torch.float32
    assert original.is_contiguous()
    assert bool(torch.isfinite(original).all())
    assert torch.equal(original[0], swapped[1])
    assert torch.equal(original[1], swapped[0])
    assert not torch.equal(original[0], original[1])
    norms = torch.linalg.vector_norm(original, dim=-1)
    assert bool(torch.logical_or(norms == 0, torch.isclose(norms, torch.ones_like(norms))).all())
    del model
    gc.collect()


def test_dropout_keep_float_has_the_pinned_raw_bits() -> None:
    keep = struct.unpack("<f", bytes.fromhex("e4388e3f"))[0]
    word = torch.tensor([keep], dtype=torch.float32).view(torch.int32)
    assert int(word.item()) == 0x3F8E38E4
