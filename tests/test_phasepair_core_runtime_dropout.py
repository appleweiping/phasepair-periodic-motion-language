from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
import gc
import hashlib
import inspect
import threading
from types import MappingProxyType
from unittest import mock

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from phasepair_core import batching, dropout, runtime_dropout  # noqa: E402
from phasepair_core import torch_models  # noqa: E402


def _bundle(
    architecture: str = "early",
    *,
    batch_size: int = 1,
    time_steps: int = 1,
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


def _shape(architecture: str, ordinal: int, batch: int, time: int) -> tuple[int, ...]:
    site = dropout.site_inventory(architecture)[ordinal]
    return dropout.site_shape(site, batch_size=batch, time_steps=time)


def _like(
    architecture: str,
    ordinal: int,
    *,
    batch: int = 1,
    time: int = 1,
    requires_grad: bool = False,
) -> torch.Tensor:
    return torch.empty(
        _shape(architecture, ordinal, batch, time),
        dtype=torch.float32,
        device="cpu",
        requires_grad=requires_grad,
    )


def _words_bytes(value: torch.Tensor) -> bytes:
    words = value.detach().view(torch.int32).reshape(-1).tolist()
    return b"".join(word.to_bytes(4, "little", signed=False) for word in words)


def test_constructor_requires_exact_literals_and_exact_policy() -> None:
    for fields in (
        {"world_size": 2},
        {"gradient_checkpointing": True},
        {"hidden_recomputation": True},
        {"fused_dropout": True},
        {"framework_rng_calls": 1},
        {"framework_dropout_calls": 1},
        {"resolved_text_nonzero_dropout_sites": 1},
        {"eval_schedule_accesses": 1},
    ):
        with pytest.raises(runtime_dropout.RuntimeDropoutContractError):
            runtime_dropout.OneForwardCanonicalDropoutBundle(
                architecture="early",
                seed=1729,
                epoch_index=0,
                global_optimizer_step=0,
                batch_size=1,
                time_steps=1,
                policy=dropout.RuntimePolicy(**fields),
            )
    with pytest.raises(TypeError):
        runtime_dropout.OneForwardCanonicalDropoutBundle(
            architecture="early",
            seed=True,
            epoch_index=0,
            global_optimizer_step=0,
            batch_size=1,
            time_steps=1,
            policy=dropout.RuntimePolicy(),
        )
    with pytest.raises(TypeError):
        runtime_dropout.OneForwardCanonicalDropoutBundle(
            architecture="early",
            seed=1729,
            epoch_index=0,
            global_optimizer_step=0,
            batch_size=1,
            time_steps=1,
            policy=object(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("architecture", "first32", "drop_count", "expanded_sha"),
    (
        (
            "mime",
            "11111101110111110111111111101111",
            18,
            "69a38dc3bbfd1fa19a0a9a34077f7ad22f92d0d5b671d3a4e0daca862feb1646",
        ),
        (
            "early",
            "11111111011111111111111101100100",
            17,
            "3a99d4693584652b3cb16d901bf624e069b232af518189464ede175679673f34",
        ),
        (
            "late",
            "11111101011111111111111111101111",
            20,
            "bdf5690c01e1b0ebec14e4fae722b0c5f257a6d41c7434b6c41d8274ced8de7f",
        ),
    ),
)
def test_first_site_matches_exact_small_golden(
    architecture: str,
    first32: str,
    drop_count: int,
    expanded_sha: str,
) -> None:
    bundle = _bundle(architecture, batch_size=2, time_steps=3)
    mask = bundle.take(0, like=_like(architecture, 0, batch=2, time=3))
    words = mask.view(torch.int32).reshape(-1)
    actual_first = "".join(
        "1" if int(word.item()) == 0x3F8E38E4 else "0" for word in words[:32]
    )
    assert actual_first == first32
    assert int((words == 0).sum().item()) == drop_count
    assert hashlib.sha256(_words_bytes(mask)).hexdigest() == expanded_sha
    assert mask.dtype is torch.float32
    assert mask.device == torch.device("cpu")
    assert mask.is_contiguous()
    assert not mask.requires_grad


def test_take_returns_private_snapshot_and_accepts_grad_bearing_like() -> None:
    first = _bundle("early")
    second = _bundle("early")
    mask_a = first.take(0, like=_like("early", 0, requires_grad=True))
    mask_b = second.take(0, like=_like("early", 0, requires_grad=False))
    assert torch.equal(mask_a, mask_b)
    assert mask_a.data_ptr() != mask_b.data_ptr()
    before = mask_b.clone()
    mask_a.zero_()
    assert torch.equal(mask_b, before)
    assert not mask_a.requires_grad


def test_instance_configuration_and_state_are_ordinary_immutable() -> None:
    bundle = _bundle("early")
    attempts = (
        ("architecture", "late"),
        ("_architecture", "late"),
        ("_seed", 2718),
        ("_epoch_index", 1),
        ("_global_optimizer_step", 1),
        ("_batch_size", 2),
        ("_time_steps", 2),
        ("_policy", ()),
        ("_sites", ()),
        ("_site_count", 0),
        ("_ordinals", ()),
        ("_lifecycle", 999),
        ("_device", torch.device("meta")),
        ("_transition_lock", object()),
        ("new_attribute", object()),
    )
    for name, value in attempts:
        with pytest.raises(TypeError, match="ordinary-immutable"):
            setattr(bundle, name, value)
    with pytest.raises(TypeError, match="ordinary-immutable"):
        del bundle._seed
    assert bundle.architecture == "early"
    assert bundle.batch_size == 1
    assert bundle.time_steps == 1
    assert bundle._lifecycle == 0


def test_captured_lock_survives_factory_rebind_and_cannot_be_replaced() -> None:
    original_lock_type = type(threading.Lock())
    with mock.patch.object(runtime_dropout.threading, "Lock", lambda: object()):
        bundle = _bundle("early")
    assert type(bundle._transition_lock) is original_lock_type
    with pytest.raises(TypeError, match="ordinary-immutable"):
        bundle._transition_lock = threading.Lock()
    mask = bundle.take(0, like=_like("early", 0))
    assert tuple(mask.shape) == (2, 1, 4, 1, 1)
    assert bundle._lifecycle == 2


def test_two_thread_same_ordinal_has_one_success_and_terminal_loser() -> None:
    bundle = _bundle("early")
    barrier = threading.Barrier(3)

    def worker() -> tuple[str, object]:
        barrier.wait(timeout=10)
        try:
            return "success", bundle.take(0, like=_like("early", 0))
        except Exception as exc:  # noqa: BLE001 - exact outcome asserted below
            return "error", exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(worker), executor.submit(worker))
        barrier.wait(timeout=10)
        outcomes = tuple(future.result(timeout=20) for future in futures)

    successes = tuple(value for label, value in outcomes if label == "success")
    errors = tuple(value for label, value in outcomes if label == "error")
    assert len(successes) == 1
    assert len(errors) == 1
    assert type(successes[0]) is torch.Tensor
    assert type(errors[0]) is runtime_dropout.RuntimeDropoutContractError
    assert "canonical order" in str(errors[0])
    assert bundle._lifecycle == 3
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.take(1, like=_like("early", 1))
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.assert_complete()


def test_failed_lifecycle_cannot_be_restored_or_replayed_ordinary_way() -> None:
    bundle = _bundle("early")
    with pytest.raises(runtime_dropout.RuntimeDropoutContractError):
        bundle.take(1, like=_like("early", 0))
    burned_lifecycle = bundle._lifecycle
    assert burned_lifecycle == 1
    with pytest.raises(TypeError, match="ordinary-immutable"):
        bundle._lifecycle = 0
    with pytest.raises(TypeError, match="ordinary-immutable"):
        bundle._seed = 2718
    assert bundle._lifecycle == burned_lifecycle
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.take(0, like=_like("early", 0))
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.assert_complete()


def test_seed_assignment_cannot_change_first_mask() -> None:
    baseline = _bundle("early").take(0, like=_like("early", 0))
    bundle = _bundle("early")
    with pytest.raises(TypeError, match="ordinary-immutable"):
        bundle._seed = 2718
    actual = bundle.take(0, like=_like("early", 0))
    assert torch.equal(actual, baseline)
    assert bundle._lifecycle == 2


def test_exact_class_getattribute_rebind_cannot_forge_record_or_mask_seed() -> None:
    baseline_bundle = _bundle("early")
    baseline = baseline_bundle.take(0, like=_like("early", 0))
    bundle = _bundle("early")
    real_generate_mask = dropout.generate_mask
    calls: list[dict[str, object]] = []

    def recording_generate_mask(
        *args: object, **kwargs: object
    ) -> dropout.DropoutMask:
        calls.append(dict(kwargs))
        return real_generate_mask(*args, **kwargs)  # type: ignore[arg-type]

    forged = {
        "_architecture": "late",
        "_seed": 2718,
        "_epoch_index": 1,
        "_global_optimizer_step": 44,
        "_batch_size": 128,
        "_time_steps": 300,
        "_policy": ("forged",),
        "_sites": (),
        "_site_count": 32,
        "_ordinals": tuple(reversed(range(16))),
        "_lifecycle": 999,
        "_device": torch.device("meta"),
        "_transition_lock": object(),
    }

    def forged_getattribute(
        self: runtime_dropout.OneForwardCanonicalDropoutBundle,
        name: str,
    ) -> object:
        if name in forged:
            return forged[name]
        return object.__getattribute__(self, name)

    first_mask: torch.Tensor | None = None
    with mock.patch.object(
        runtime_dropout.OneForwardCanonicalDropoutBundle,
        "__getattribute__",
        forged_getattribute,
    ), mock.patch.object(dropout, "generate_mask", recording_generate_mask):
        assert bundle.architecture == "early"
        assert bundle.batch_size == 1
        assert bundle.time_steps == 1
        for ordinal in range(16):
            mask = bundle.take(ordinal, like=_like("early", ordinal))
            if ordinal == 0:
                first_mask = mask
        assert bundle.assert_complete() is None

    assert first_mask is not None
    assert torch.equal(first_mask, baseline)
    assert len(calls) == 16
    assert {call["seed"] for call in calls} == {1729}
    assert {call["epoch_index"] for call in calls} == {0}
    assert {call["global_optimizer_step"] for call in calls} == {0}
    assert calls[0]["site_name_ascii"] == "blocks.00.self.attn"
    assert calls[0]["shape"] == (2, 1, 4, 1, 1)
    assert bundle._lifecycle == 33


def test_inventory_closure_and_per_instance_snapshots_are_immutable() -> None:
    closure = inspect.getclosurevars(
        runtime_dropout.OneForwardCanonicalDropoutBundle.__init__
    )
    inventories = closure.nonlocals["inventories"]
    assert isinstance(inventories, MappingProxyType)
    assert tuple(inventories) == ("mime", "early", "late")
    with pytest.raises(TypeError):
        inventories["early"] = ()
    assert all(type(rows) is tuple for rows in inventories.values())
    assert all(
        type(row) is tuple
        for rows in inventories.values()
        for row in rows
    )

    first = _bundle("early")
    second = _bundle("early")
    assert first._sites is not second._sites
    assert first._sites == second._sites
    assert first._site_count == second._site_count == 16
    assert first._ordinals == second._ordinals == tuple(range(16))
    with pytest.raises(TypeError, match="ordinary-immutable"):
        first._sites = ()
    with pytest.raises(TypeError, match="ordinary-immutable"):
        first._ordinals = ()


def test_materialization_allzero_and_noop_rebinds_cannot_change_bytes() -> None:
    baseline = _bundle("early").take(0, like=_like("early", 0))
    original_frombuffer = torch.frombuffer

    def allzero(buffer: object, *, dtype: object) -> torch.Tensor:
        del dtype
        return torch.zeros(len(buffer) // 4, dtype=torch.float32)  # type: ignore[arg-type]

    def noop(buffer: object, *, dtype: object) -> object:
        del dtype
        return buffer

    for mutant in (allzero, noop):
        with mock.patch.object(runtime_dropout.torch, "frombuffer", mutant):
            actual = _bundle("early").take(0, like=_like("early", 0))
        assert torch.equal(actual, baseline)
    assert torch.frombuffer is original_frombuffer
    source = inspect.getsource(
        runtime_dropout.OneForwardCanonicalDropoutBundle.take
    )
    assert "actual_raw_bytes != expected[6]" in source


def test_wrong_order_fail_burns_and_cannot_replay() -> None:
    bundle = _bundle("early")
    generator = mock.Mock(wraps=dropout.generate_mask)
    with mock.patch.object(dropout, "generate_mask", generator):
        with pytest.raises(
            runtime_dropout.RuntimeDropoutContractError,
            match="canonical order",
        ):
            bundle.take(1, like=_like("early", 0))
        generator.assert_not_called()
        with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
            bundle.take(0, like=_like("early", 0))
        generator.assert_not_called()


@pytest.mark.parametrize("mode", ("type", "dtype", "layout", "shape", "device"))
def test_like_mutants_fail_burn_before_generation(mode: str) -> None:
    bundle = _bundle("early")
    if mode == "type":
        like: object = np.zeros(_shape("early", 0, 1, 1), dtype=np.float32)
    elif mode == "dtype":
        like = torch.empty(_shape("early", 0, 1, 1), dtype=torch.float64)
    elif mode == "layout":
        like = torch.empty((2, 1, 8, 1, 1), dtype=torch.float32)[:, :, ::2]
        assert tuple(like.shape) == _shape("early", 0, 1, 1)
        assert not like.is_contiguous()
    elif mode == "shape":
        like = torch.empty((2, 1, 4, 2, 2), dtype=torch.float32)
    elif mode == "device":
        like = torch.empty(_shape("early", 0, 1, 1), device="meta")
    else:
        raise AssertionError("unknown mutant")
    generator = mock.Mock(wraps=dropout.generate_mask)
    with mock.patch.object(dropout, "generate_mask", generator):
        with pytest.raises((TypeError, runtime_dropout.RuntimeDropoutContractError)):
            bundle.take(0, like=like)  # type: ignore[arg-type]
        generator.assert_not_called()
        with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
            bundle.take(0, like=_like("early", 0))


def test_forged_h0_and_keep_bytes_fail_closed_and_burn() -> None:
    real_generator = dropout.generate_mask

    def wrong_h0(*args: object, **kwargs: object) -> dropout.DropoutMask:
        generated = real_generator(*args, **kwargs)  # type: ignore[arg-type]
        return replace(generated, h0=b"\x00" * 32)

    def wrong_keep(*args: object, **kwargs: object) -> dropout.DropoutMask:
        generated = real_generator(*args, **kwargs)  # type: ignore[arg-type]
        mutated = generated.expanded_float32_le.replace(
            bytes.fromhex("e4388e3f"), bytes.fromhex("0000803f"), 1
        )
        return replace(generated, expanded_float32_le=mutated)

    for mutant in (wrong_h0, wrong_keep):
        bundle = _bundle("early")
        with mock.patch.object(dropout, "generate_mask", mutant):
            with pytest.raises(
                runtime_dropout.RuntimeDropoutContractError,
                match="literal H0/SplitMix64/raw-bit law",
            ):
                bundle.take(0, like=_like("early", 0))
        with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
            bundle.take(0, like=_like("early", 0))


def test_factory_and_module_helper_rebinding_cannot_change_output() -> None:
    assert not hasattr(runtime_dropout, "_make_bundle_class")
    baseline = _bundle("early").take(0, like=_like("early", 0))

    def bypass(*args: object, **kwargs: object) -> None:
        return None

    with mock.patch.multiple(
        runtime_dropout,
        _literal_shape=bypass,
        _expected_mask=bypass,
        _validate_generated=lambda *args, **kwargs: b"",
        _validate_policy=lambda *args, **kwargs: (1, False, False, False, 0, 0, 0, 0),
        _RuntimePolicyType=object,
        _DropoutMaskType=object,
        create=True,
    ):
        rebound = _bundle("early").take(0, like=_like("early", 0))
    assert torch.equal(rebound, baseline)


def test_generator_and_policy_types_are_captured_exactly() -> None:
    policy = dropout.RuntimePolicy()
    baseline = _bundle("early").take(0, like=_like("early", 0))
    with mock.patch.multiple(
        dropout,
        RuntimePolicy=object,
        generate_site_mask=mock.Mock(side_effect=AssertionError("rebound generator")),
    ):
        bundle = runtime_dropout.OneForwardCanonicalDropoutBundle(
            architecture="early",
            seed=1729,
            epoch_index=0,
            global_optimizer_step=0,
            batch_size=1,
            time_steps=1,
            policy=policy,
        )
        rebound = bundle.take(0, like=_like("early", 0))
    assert torch.equal(rebound, baseline)


def test_shared_hashlib_sha256_mutant_is_detected_and_burns() -> None:
    original_sha256 = hashlib.sha256

    class _ZeroDigest:
        def digest(self) -> bytes:
            return b"\x00" * 32

    def zero_sha256(value: object = b"") -> _ZeroDigest:
        original_sha256(bytes(value))
        return _ZeroDigest()

    bundle = _bundle("early")
    with mock.patch.object(dropout.hashlib, "sha256", zero_sha256):
        with pytest.raises(
            runtime_dropout.RuntimeDropoutContractError,
            match="literal H0/SplitMix64/raw-bit law",
        ):
            bundle.take(0, like=_like("early", 0))
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.take(0, like=_like("early", 0))


def test_generated_shape_bool_dimension_is_rejected_before_tuple_equality() -> None:
    real_generator = dropout.generate_mask

    def bool_dimension(*args: object, **kwargs: object) -> dropout.DropoutMask:
        generated = real_generator(*args, **kwargs)  # type: ignore[arg-type]
        forged_shape = (generated.shape[0], True, *generated.shape[2:])
        assert forged_shape == generated.shape
        return replace(generated, shape=forged_shape)

    bundle = _bundle("early")
    with mock.patch.object(dropout, "generate_mask", bool_dimension):
        with pytest.raises(TypeError, match=r"shape\[1\].*exact int"):
            bundle.take(0, like=_like("early", 0))
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.take(0, like=_like("early", 0))


def test_forged_exact_policy_cannot_use_module_validator_rebind() -> None:
    policy_type = dropout.RuntimePolicy
    forged = object.__new__(policy_type)
    canonical = dropout.RuntimePolicy()
    for field_name in canonical.__slots__:
        object.__setattr__(forged, field_name, getattr(canonical, field_name))
    object.__setattr__(forged, "gradient_checkpointing", 0)
    with mock.patch.multiple(
        runtime_dropout,
        _validate_policy=lambda *args, **kwargs: (1, False, False, False, 0, 0, 0, 0),
        _RuntimePolicyType=object,
        create=True,
    ), mock.patch.object(dropout, "RuntimePolicy", object):
        with pytest.raises(TypeError, match="exact built-in bool"):
            runtime_dropout.OneForwardCanonicalDropoutBundle(
                architecture="early",
                seed=1729,
                epoch_index=0,
                global_optimizer_step=0,
                batch_size=1,
                time_steps=1,
                policy=forged,
            )


def test_simultaneous_helper_hash_policy_and_type_rebind_fails_and_burns() -> None:
    policy = dropout.RuntimePolicy()

    class _ZeroDigest:
        def digest(self) -> bytes:
            return b"\x00" * 32

    def zero_sha256(value: object = b"") -> _ZeroDigest:
        bytes(value)
        return _ZeroDigest()

    bundle = runtime_dropout.OneForwardCanonicalDropoutBundle(
        architecture="early",
        seed=1729,
        epoch_index=0,
        global_optimizer_step=0,
        batch_size=1,
        time_steps=1,
        policy=policy,
    )
    with mock.patch.multiple(
        runtime_dropout,
        _literal_shape=lambda *args, **kwargs: (2, 1, 4, 1, 1),
        _expected_mask=lambda *args, **kwargs: (),
        _validate_generated=lambda *args, **kwargs: bytes(32),
        _validate_policy=lambda *args, **kwargs: (1, False, False, False, 0, 0, 0, 0),
        _RuntimePolicyType=object,
        _DropoutMaskType=object,
        create=True,
    ), mock.patch.multiple(
        dropout,
        RuntimePolicy=object,
        generate_site_mask=lambda *args, **kwargs: object(),
    ), mock.patch.object(dropout.hashlib, "sha256", zero_sha256):
        with pytest.raises(runtime_dropout.RuntimeDropoutContractError):
            bundle.take(0, like=_like("early", 0))
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.take(0, like=_like("early", 0))


def test_rebinding_informational_globals_cannot_change_output_law() -> None:
    baseline = _bundle("early").take(0, like=_like("early", 0))
    with mock.patch.multiple(
        runtime_dropout,
        STATUS="PASS",
        LAW_ID="wrong",
        H0_DOMAIN=b"wrong",
        KEEP_FLOAT32_BITS=0x3F800000,
        ARCHITECTURES=(),
        SITE_COUNTS={},
    ):
        rebound = _bundle("early").take(0, like=_like("early", 0))
    assert torch.equal(rebound, baseline)


def test_rebinding_dropout_equation_globals_is_independently_detected() -> None:
    for label, value in (
        ("H0_DOMAIN", b"wrong-domain"),
        ("PHI", 0),
        ("DROP_THRESHOLD", 0xFFFFFFFFFFFFFFFF),
        ("KEEP_FLOAT32_LE", bytes.fromhex("0000803f")),
    ):
        bundle = _bundle("early", batch_size=2, time_steps=3)
        with mock.patch.object(dropout, label, value):
            with pytest.raises(runtime_dropout.RuntimeDropoutContractError):
                bundle.take(0, like=_like("early", 0, batch=2, time=3))
        with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
            bundle.take(0, like=_like("early", 0, batch=2, time=3))


def test_assert_complete_requires_all_sites_and_is_exactly_once() -> None:
    early = _bundle("early")
    with pytest.raises(runtime_dropout.RuntimeDropoutContractError):
        early.assert_complete()
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        early.take(0, like=_like("early", 0))

    complete = _bundle("early")
    for ordinal in range(16):
        complete.take(ordinal, like=_like("early", ordinal))
    assert complete.assert_complete() is None
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        complete.assert_complete()
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        complete.take(15, like=_like("early", 15))


def _tiny_batch() -> batching.PreparedMotionBatch:
    return batching.PreparedMotionBatch(
        np.zeros((1, 1, 262), dtype=np.float32),
        np.zeros((1, 1, 262), dtype=np.float32),
        np.zeros((1, 1, 799), dtype=np.float32),
        np.zeros((1, 1, 799), dtype=np.float32),
        np.ones((1, 1), dtype=np.uint8),
        (1,),
        (b"A" * 32,),
    )


@pytest.mark.parametrize("architecture", ("mime", "early", "late"))
def test_bundle_records_exact_full_traversal_through_model(architecture: str) -> None:
    model = torch_models.build_motion_encoder(architecture, device="cpu").train()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        bundle = _bundle(architecture)
        output = model(_tiny_batch(), dropout_bundle=bundle).ordered
    assert output.shape == (2, 1, 512)
    assert bool(torch.isfinite(output).all().item())
    with pytest.raises(runtime_dropout.RuntimeDropoutBurnedError):
        bundle.assert_complete()
    del model
    gc.collect()
