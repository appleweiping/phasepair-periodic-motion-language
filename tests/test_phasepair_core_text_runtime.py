from __future__ import annotations

import copy
import gc
import inspect
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading
from types import SimpleNamespace
from typing import Callable
import weakref

import pytest

import phasepair_core.text_runtime as text_runtime


def _api(
    constructor: Callable[[tuple[str, ...]], object] | None = None,
    *,
    snapshot_stable: bool = True,
    runtime_complete: bool = True,
    active_text_dropout_sites: int = 0,
) -> object:
    if constructor is None:
        def default_constructor(payload: tuple[str, ...]) -> object:
            return ("synthetic-optimizer", payload)

        constructor = default_constructor
    return text_runtime._seal_text_runtime_api_for_tests(
        text_runtime._SyntheticContractSnapshot(
            text_runtime.CONTRACT_FAMILY,
            "tiny-primitives-only",
        ),
        text_runtime._SyntheticFilesystemOps(snapshot_stable),
        text_runtime._SyntheticLoaderOps(
            runtime_complete,
            active_text_dropout_sites,
        ),
        text_runtime._SyntheticOptimizerOps(constructor),
    )


def _lease_and_training_authority(api: object) -> tuple[object, object]:
    resolution = api.issue_resolution_authority()
    lease = api.issue_runtime_lease(resolution, payload=("row-a", "row-b"))
    return lease, api.issue_training_authority()


def test_production_api_signatures_are_closed() -> None:
    issue = inspect.signature(text_runtime.issue_phasepair_runtime_lease)
    canonical = inspect.signature(text_runtime.canonical_runtime_receipt_bytes)
    consume = inspect.signature(text_runtime.consume_phasepair_runtime_lease)

    assert tuple(issue.parameters) == (
        "snapshot_root",
        "wheelhouse_root",
        "architecture",
        "seed",
        "resolution_authority",
    )
    assert issue.parameters["snapshot_root"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert issue.parameters["wheelhouse_root"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    assert issue.parameters["architecture"].kind is inspect.Parameter.KEYWORD_ONLY
    assert issue.parameters["seed"].kind is inspect.Parameter.KEYWORD_ONLY
    assert issue.parameters["resolution_authority"].kind is inspect.Parameter.KEYWORD_ONLY
    assert tuple(canonical.parameters) == ("lease",)
    assert tuple(consume.parameters) == ("lease", "training_authority")
    assert consume.parameters["training_authority"].kind is inspect.Parameter.KEYWORD_ONLY

    forbidden = {
        "assessment",
        "rows",
        "model",
        "hash",
        "loader",
        "optimizer",
    }
    all_names = set(issue.parameters) | set(canonical.parameters) | set(
        consume.parameters
    )
    assert forbidden.isdisjoint(all_names)


@pytest.mark.parametrize(
    "opaque_type",
    [
        text_runtime.ResolutionAuthorityLease,
        text_runtime.TrainingAuthorityLease,
        text_runtime.ResolvedPhasePairRuntimeLease,
        text_runtime.PhasePairTrainingRuntime,
    ],
)
def test_production_opaque_types_have_no_public_constructor(
    opaque_type: type[object],
) -> None:
    with pytest.raises(TypeError):
        opaque_type()


def test_production_issue_is_fixed_hold_before_argument_observation() -> None:
    class Bomb:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(name)

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        text_runtime.issue_phasepair_runtime_lease(
            Bomb(),
            Bomb(),
            architecture=Bomb(),
            seed=Bomb(),
            resolution_authority=Bomb(),
        )

    assert caught.value.code == "HOLD_RESOLUTION_AUTHORITY_ISSUER_ABSENT"
    assert text_runtime._production_adamw_constructor_call_count_for_tests() == 0
    assert text_runtime._production_live_lease_count_for_tests() == 0


@pytest.mark.parametrize(
    "forbidden_name",
    [
        "assessment",
        "rows",
        "resolved_text_rows",
        "model",
        "tokenizer",
        "hashes",
        "loader",
        "optimizer_factory",
        "receipt_bytes",
    ],
)
def test_production_issue_rejects_every_caller_injection_keyword(
    forbidden_name: str,
) -> None:
    arguments = {
        "architecture": "mime",
        "seed": 0,
        "resolution_authority": object(),
        forbidden_name: object(),
    }
    with pytest.raises(TypeError):
        text_runtime.issue_phasepair_runtime_lease(
            Path("unused-snapshot"),
            Path("unused-wheelhouse"),
            **arguments,
        )
    assert text_runtime._production_adamw_constructor_call_count_for_tests() == 0


def test_production_registry_rejects_forge_copy_subclass_and_receipt_bytes() -> None:
    forged = object.__new__(text_runtime.ResolvedPhasePairRuntimeLease)

    class LeaseSubclass(text_runtime.ResolvedPhasePairRuntimeLease):
        pass

    subclass = object.__new__(LeaseSubclass)
    for candidate in (forged, copy.copy(forged), subclass, b"{}\n"):
        with pytest.raises((TypeError, text_runtime.TextRuntimeHold)):
            text_runtime.canonical_runtime_receipt_bytes(candidate)
        with pytest.raises((TypeError, text_runtime.TextRuntimeHold)):
            text_runtime.consume_phasepair_runtime_lease(
                candidate,
                training_authority=object(),
            )

    assert text_runtime._production_adamw_constructor_call_count_for_tests() == 0


def test_synthetic_receipt_is_canonical_authority_zero_nonproduction() -> None:
    api = _api()
    lease, training = _lease_and_training_authority(api)

    receipt = api.canonical_receipt_bytes(lease)
    decoded = json.loads(receipt)
    assert receipt.endswith(b"\n")
    assert receipt == (
        json.dumps(
            decoded,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )
    assert decoded["authority"] == 0
    assert decoded["production"] is False
    assert decoded["training_authorized"] is False
    assert decoded["lifecycle"] == "ISSUED"
    assert decoded["status"] == "SYNTHETIC_AUTHORITY0_NONPRODUCTION"

    runtime = api.consume_runtime_lease(lease, training_authority=training)
    assert type(runtime) is api.training_runtime_type
    assert api.runtime_lease_state(lease) == text_runtime.CONSUMED
    assert api.adamw_constructor_call_count() == 1
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.canonical_receipt_bytes(lease)
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(
            lease,
            training_authority=api.issue_training_authority(),
        )
    assert api.adamw_constructor_call_count() == 1


@pytest.mark.parametrize(
    ("snapshot_stable", "runtime_complete", "active_dropout"),
    [
        (False, True, 0),
        (True, False, 0),
        (True, True, 1),
    ],
)
def test_every_synthetic_preflight_failure_burns_before_constructor(
    snapshot_stable: bool,
    runtime_complete: bool,
    active_dropout: int,
) -> None:
    api = _api(
        snapshot_stable=snapshot_stable,
        runtime_complete=runtime_complete,
        active_text_dropout_sites=active_dropout,
    )
    lease, training = _lease_and_training_authority(api)

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        api.consume_runtime_lease(lease, training_authority=training)
    assert caught.value.code == "HOLD_SYNTHETIC_PREFLIGHT"
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(
            lease,
            training_authority=api.issue_training_authority(),
        )
    assert api.adamw_constructor_call_count() == 0


def test_synthetic_capabilities_and_leases_are_one_shot_and_factory_bound() -> None:
    api = _api()
    foreign = _api()
    resolution = api.issue_resolution_authority()
    lease = api.issue_runtime_lease(resolution)

    with pytest.raises(text_runtime.TextRuntimeHold):
        api.issue_runtime_lease(resolution)
    with pytest.raises(text_runtime.TextRuntimeHold):
        foreign.canonical_receipt_bytes(lease)
    with pytest.raises(text_runtime.TextRuntimeHold):
        foreign.consume_runtime_lease(
            lease,
            training_authority=foreign.issue_training_authority(),
        )

    forged = object.__new__(api.runtime_lease_type)

    class LeaseSubclass(api.runtime_lease_type):
        pass

    for candidate in (forged, copy.copy(lease), object.__new__(LeaseSubclass)):
        with pytest.raises(text_runtime.TextRuntimeHold):
            api.canonical_receipt_bytes(candidate)
    assert api.adamw_constructor_call_count() == 0

    foreign_training = foreign.issue_training_authority()
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(lease, training_authority=foreign_training)
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0


def test_training_authority_replay_burns_second_lease_before_constructor() -> None:
    api = _api()
    first_lease, training = _lease_and_training_authority(api)
    api.consume_runtime_lease(first_lease, training_authority=training)
    second_lease = api.issue_runtime_lease(api.issue_resolution_authority())

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        api.consume_runtime_lease(second_lease, training_authority=training)
    assert caught.value.code == "HOLD_TRAINING_AUTHORITY_NOT_ISSUED"
    assert api.runtime_lease_state(second_lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 1


def test_mutation_is_detected_and_burned_before_constructor() -> None:
    api = _api()
    lease, training = _lease_and_training_authority(api)
    object.__setattr__(lease, "_payload", ("replacement",))

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        api.consume_runtime_lease(lease, training_authority=training)
    assert caught.value.code == "HOLD_RUNTIME_LEASE_MUTATED"
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0


def test_registry_lock_is_used_before_hostile_runtime_lock_field() -> None:
    api = _api()
    lease, training = _lease_and_training_authority(api)

    class RecursiveHostileLock:
        def __init__(self) -> None:
            self.enter_calls = 0

        def __enter__(self) -> object:
            self.enter_calls += 1
            return api.canonical_receipt_bytes(lease)

        def __exit__(self, *_args: object) -> None:
            raise AssertionError("hostile lease lock was executed")

    hostile = RecursiveHostileLock()
    object.__setattr__(lease, "_lock", hostile)

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        api.consume_runtime_lease(lease, training_authority=training)
    assert caught.value.code == "HOLD_RUNTIME_LEASE_MUTATED"
    assert hostile.enter_calls == 0
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.canonical_receipt_bytes(lease)
    assert api.runtime_lease_state(lease) == text_runtime.BURNED


def test_missing_runtime_lock_burns_on_canonical_and_cannot_retry() -> None:
    api = _api()
    lease, _training = _lease_and_training_authority(api)
    object.__delattr__(lease, "_lock")

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        api.canonical_receipt_bytes(lease)
    assert caught.value.code == "HOLD_RUNTIME_LEASE_MUTATED"
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0
    with pytest.raises(text_runtime.TextRuntimeHold) as retried:
        api.canonical_receipt_bytes(lease)
    assert retried.value.code == "HOLD_RUNTIME_LEASE_NOT_ISSUED_STATE"
    assert api.runtime_lease_state(lease) == text_runtime.BURNED


def test_canonical_baseexception_from_hostile_descriptor_still_burns() -> None:
    class SyntheticAbort(BaseException):
        pass

    class ExplodingDescriptor:
        def __get__(self, _instance: object, _owner: type[object]) -> object:
            raise SyntheticAbort("hostile lease descriptor")

    api = _api()
    lease, _training = _lease_and_training_authority(api)
    setattr(api.runtime_lease_type, "_nonce", ExplodingDescriptor())

    with pytest.raises(SyntheticAbort):
        api.canonical_receipt_bytes(lease)
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.canonical_receipt_bytes(lease)
    assert api.runtime_lease_state(lease) == text_runtime.BURNED


def test_hostile_runtime_authority_reference_is_not_executed() -> None:
    api = _api()
    lease, training = _lease_and_training_authority(api)

    class HostileAuthority:
        def __getattribute__(self, name: str) -> object:
            raise AssertionError(f"hostile authority was observed: {name}")

    hostile = HostileAuthority()
    object.__setattr__(
        lease,
        "_resolution_authority_reference",
        hostile,
    )

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        api.consume_runtime_lease(lease, training_authority=training)
    assert caught.value.code == "HOLD_RUNTIME_LEASE_MUTATED"
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0


def test_authority_registry_rejects_forge_copy_and_subclass() -> None:
    api = _api()
    resolution = api.issue_resolution_authority()

    class ResolutionSubclass(api.resolution_authority_type):
        pass

    invalid_resolution = (
        object.__new__(api.resolution_authority_type),
        copy.copy(resolution),
        object.__new__(ResolutionSubclass),
    )
    for candidate in invalid_resolution:
        with pytest.raises(text_runtime.TextRuntimeHold):
            api.issue_runtime_lease(candidate)

    valid_lease = api.issue_runtime_lease(resolution)
    assert api.runtime_lease_state(valid_lease) == text_runtime.ISSUED

    for mode in ("forge", "copy", "subclass"):
        isolated = _api()
        lease = isolated.issue_runtime_lease(
            isolated.issue_resolution_authority()
        )
        authority = isolated.issue_training_authority()

        class TrainingSubclass(isolated.training_authority_type):
            pass

        if mode == "forge":
            candidate = object.__new__(isolated.training_authority_type)
        elif mode == "copy":
            candidate = copy.copy(authority)
        else:
            candidate = object.__new__(TrainingSubclass)
        with pytest.raises(text_runtime.TextRuntimeHold):
            isolated.consume_runtime_lease(
                lease,
                training_authority=candidate,
            )
        assert isolated.runtime_lease_state(lease) == text_runtime.BURNED
        assert isolated.adamw_constructor_call_count() == 0


def test_authority_uses_registry_lock_before_hostile_lock_field() -> None:
    api = _api()

    class HostileLock:
        def __init__(self) -> None:
            self.enter_calls = 0

        def __enter__(self) -> object:
            self.enter_calls += 1
            raise AssertionError("hostile authority lock was executed")

        def __exit__(self, *_args: object) -> None:
            raise AssertionError("hostile authority lock was executed")

    resolution = api.issue_resolution_authority()
    hostile = HostileLock()
    object.__setattr__(resolution, "_lock", hostile)

    with pytest.raises(text_runtime.TextRuntimeHold):
        api.issue_runtime_lease(resolution)
    assert hostile.enter_calls == 0
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.issue_runtime_lease(resolution)
    assert hostile.enter_calls == 0
    assert api.adamw_constructor_call_count() == 0


def test_hostile_training_authority_lock_burns_runtime_before_constructor() -> None:
    api = _api()
    lease, training = _lease_and_training_authority(api)

    class HostileLock:
        def __init__(self) -> None:
            self.enter_calls = 0

        def __enter__(self) -> object:
            self.enter_calls += 1
            raise AssertionError("hostile training lock was executed")

        def __exit__(self, *_args: object) -> None:
            raise AssertionError("hostile training lock was executed")

    hostile = HostileLock()
    object.__setattr__(training, "_lock", hostile)

    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(lease, training_authority=training)
    assert hostile.enter_calls == 0
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0

    second_lease = api.issue_runtime_lease(api.issue_resolution_authority())
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(
            second_lease,
            training_authority=training,
        )
    assert hostile.enter_calls == 0
    assert api.runtime_lease_state(second_lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 0


@pytest.mark.parametrize("mode", ["raises", "returns-none"])
def test_constructor_failure_burns_and_cannot_be_retried(mode: str) -> None:
    def constructor(_payload: tuple[str, ...]) -> object:
        if mode == "raises":
            raise RuntimeError("synthetic constructor failure")
        return None

    api = _api(constructor)
    lease, training = _lease_and_training_authority(api)

    with pytest.raises((RuntimeError, text_runtime.TextRuntimeHold)):
        api.consume_runtime_lease(lease, training_authority=training)
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 1
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(
            lease,
            training_authority=api.issue_training_authority(),
        )
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 1


def test_constructor_baseexception_burns_and_cannot_be_retried() -> None:
    class SyntheticAbort(BaseException):
        pass

    def constructor(_payload: tuple[str, ...]) -> object:
        raise SyntheticAbort("abort synthetic constructor")

    api = _api(constructor)
    lease, training = _lease_and_training_authority(api)

    with pytest.raises(SyntheticAbort):
        api.consume_runtime_lease(lease, training_authority=training)
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 1
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(
            lease,
            training_authority=api.issue_training_authority(),
        )
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 1


def test_constructor_time_mutation_is_detected_after_single_call() -> None:
    holder: dict[str, object] = {}

    class HostileLock:
        def __init__(self) -> None:
            self.enter_calls = 0

        def __enter__(self) -> object:
            self.enter_calls += 1
            raise AssertionError("hostile lock was executed")

        def __exit__(self, *_args: object) -> None:
            raise AssertionError("hostile lock was executed")

    hostile = HostileLock()

    def constructor(_payload: tuple[str, ...]) -> object:
        object.__setattr__(holder["lease"], "_lock", hostile)
        return object()

    api = _api(constructor)
    lease, training = _lease_and_training_authority(api)
    holder["lease"] = lease

    with pytest.raises(text_runtime.TextRuntimeHold) as caught:
        api.consume_runtime_lease(lease, training_authority=training)
    assert caught.value.code == "HOLD_RUNTIME_LEASE_MUTATED"
    assert hostile.enter_calls == 0
    assert api.runtime_lease_state(lease) == text_runtime.BURNED
    assert api.adamw_constructor_call_count() == 1
    with pytest.raises(text_runtime.TextRuntimeHold):
        api.consume_runtime_lease(
            lease,
            training_authority=api.issue_training_authority(),
        )
    assert api.adamw_constructor_call_count() == 1


def test_two_thread_race_allows_exactly_one_constructor() -> None:
    calls: list[tuple[str, ...]] = []
    calls_lock = threading.Lock()

    def constructor(payload: tuple[str, ...]) -> object:
        with calls_lock:
            calls.append(payload)
        return object()

    api = _api(constructor)
    lease = api.issue_runtime_lease(api.issue_resolution_authority())
    authorities = [api.issue_training_authority() for _ in range(2)]
    barrier = threading.Barrier(2)

    def consume(authority: object) -> object:
        barrier.wait()
        try:
            return api.consume_runtime_lease(
                lease,
                training_authority=authority,
            )
        except text_runtime.TextRuntimeHold as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(consume, authorities))

    assert sum(type(item) is api.training_runtime_type for item in results) == 1
    assert sum(isinstance(item, text_runtime.TextRuntimeHold) for item in results) == 1
    assert len(calls) == 1
    assert api.adamw_constructor_call_count() == 1
    assert api.runtime_lease_state(lease) == text_runtime.CONSUMED


def test_constructor_observes_consuming_state() -> None:
    holder: dict[str, object] = {}

    def constructor(_payload: tuple[str, ...]) -> object:
        api = holder["api"]
        lease = holder["lease"]
        holder["observed"] = api.runtime_lease_state(lease)
        return object()

    api = _api(constructor)
    lease, training = _lease_and_training_authority(api)
    holder.update(api=api, lease=lease)

    api.consume_runtime_lease(lease, training_authority=training)

    assert holder["observed"] == text_runtime.CONSUMING
    assert api.runtime_lease_state(lease) == text_runtime.CONSUMED


def test_weak_registry_releases_garbage_collected_lease() -> None:
    api = _api()
    lease = api.issue_runtime_lease(api.issue_resolution_authority())
    reference = weakref.ref(lease)
    assert api.live_runtime_lease_count() == 1

    del lease
    gc.collect()

    assert reference() is None
    assert api.live_runtime_lease_count() == 0


def test_sealed_factory_survives_module_global_rebinding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    api = _api()
    original_hold = text_runtime.TextRuntimeHold
    replacements = {
        "hashlib": SimpleNamespace(),
        "json": SimpleNamespace(),
        "threading": SimpleNamespace(),
        "weakref": SimpleNamespace(),
        "TextRuntimeHold": RuntimeError,
        "ISSUED": 97,
        "CONSUMING": 98,
        "CONSUMED": 99,
        "BURNED": 100,
    }
    for name, value in replacements.items():
        monkeypatch.setattr(text_runtime, name, value)

    lease, training = _lease_and_training_authority(api)
    assert json.loads(api.canonical_receipt_bytes(lease))["authority"] == 0
    runtime = api.consume_runtime_lease(lease, training_authority=training)
    assert type(runtime) is api.training_runtime_type
    assert api.runtime_lease_state(lease) == 2
    assert api.adamw_constructor_call_count() == 1
    with pytest.raises(original_hold):
        api.consume_runtime_lease(lease, training_authority=training)


def test_synthetic_types_can_never_enter_production_consumers() -> None:
    api = _api()
    lease, training = _lease_and_training_authority(api)

    with pytest.raises(TypeError):
        text_runtime.canonical_runtime_receipt_bytes(lease)
    with pytest.raises(TypeError):
        text_runtime.consume_phasepair_runtime_lease(
            lease,
            training_authority=training,
        )
    assert text_runtime._production_adamw_constructor_call_count_for_tests() == 0


def test_module_has_no_torch_transformers_or_runtime_loader_import() -> None:
    source = inspect.getsource(text_runtime)
    imports = {
        line.strip()
        for line in source.splitlines()
        if line.lstrip().startswith(("import ", "from "))
    }
    assert not any("torch" in line for line in imports)
    assert not any("transformers" in line for line in imports)
    assert not any("clip_resolution" in line for line in imports)
