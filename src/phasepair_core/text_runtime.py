"""Fail-closed live text-runtime bridge shell for PhasePair.

The production API in this module deliberately has no authority issuer.  It
does not inspect a model snapshot, import a loader, construct a model, or call
AdamW.  A private isolated factory exists only to exercise opaque-capability,
one-shot, concurrency, failure-burn, receipt, and garbage-collection semantics
with tiny primitive values.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
from types import MappingProxyType
from typing import Callable
import weakref


STATUS = "HOLD_RESOLUTION_AUTHORITY_ISSUER_ABSENT"
AUTHORITY = 0
PRODUCTION = False
TRAINING_AUTHORIZED = False
CONTRACT_FAMILY = "PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840"

HOLD_RESOLUTION_AUTHORITY_ISSUER_ABSENT = (
    "HOLD_RESOLUTION_AUTHORITY_ISSUER_ABSENT"
)
HOLD_RUNTIME_LEASE_NOT_ISSUED = "HOLD_RUNTIME_LEASE_NOT_ISSUED"
HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY = (
    "HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY"
)
HOLD_RUNTIME_LEASE_NOT_ISSUED_STATE = "HOLD_RUNTIME_LEASE_NOT_ISSUED_STATE"
HOLD_RUNTIME_LEASE_MUTATED = "HOLD_RUNTIME_LEASE_MUTATED"
HOLD_TRAINING_AUTHORITY_NOT_ISSUED = "HOLD_TRAINING_AUTHORITY_NOT_ISSUED"
HOLD_SYNTHETIC_PREFLIGHT = "HOLD_SYNTHETIC_PREFLIGHT"
HOLD_SYNTHETIC_OPTIMIZER_POSTCONDITION = (
    "HOLD_SYNTHETIC_OPTIMIZER_POSTCONDITION"
)

ISSUED = 0
CONSUMING = 1
CONSUMED = 2
BURNED = 3


class TextRuntimeError(ValueError):
    """A closed runtime-bridge invariant was violated."""


class TextRuntimeHold(TextRuntimeError):
    """A required authority or live invariant is absent."""

    def __init__(self, code: str) -> None:
        if type(code) is not str or not code:
            raise TypeError("hold code must be an exact nonempty built-in str")
        self.code = code
        super().__init__(code)


class ResolutionAuthorityLease:
    """Opaque production capability; this module has no issuer for it."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ResolutionAuthorityLease construction is internal")


class TrainingAuthorityLease:
    """Opaque production capability; this module has no issuer for it."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("TrainingAuthorityLease construction is internal")


class ResolvedPhasePairRuntimeLease:
    """Opaque live production lease; no production lease can currently exist."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ResolvedPhasePairRuntimeLease construction is internal")


class PhasePairTrainingRuntime:
    """Opaque production runtime; no production runtime can currently exist."""

    __slots__ = ("__weakref__",)

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("PhasePairTrainingRuntime construction is internal")


def _seal_absent_production_api(
    *,
    runtime_lease_type: type[ResolvedPhasePairRuntimeLease] = (
        ResolvedPhasePairRuntimeLease
    ),
    authority_type: type[TrainingAuthorityLease] = TrainingAuthorityLease,
    hold_type: type[TextRuntimeHold] = TextRuntimeHold,
    weak_registry_type: type[weakref.WeakKeyDictionary[object, object]] = (
        weakref.WeakKeyDictionary
    ),
) -> tuple[Callable[..., object], ...]:
    """Bind an empty production issuer registry and an inert constructor count."""

    issued: weakref.WeakKeyDictionary[object, object] = weak_registry_type()
    adamw_constructor_call_count = 0

    def issue(
        snapshot_root: Path,
        wheelhouse_root: Path,
        *,
        architecture: str,
        seed: int,
        resolution_authority: ResolutionAuthorityLease,
    ) -> ResolvedPhasePairRuntimeLease:
        del (
            snapshot_root,
            wheelhouse_root,
            architecture,
            seed,
            resolution_authority,
        )
        raise hold_type("HOLD_RESOLUTION_AUTHORITY_ISSUER_ABSENT")

    def canonical(lease: ResolvedPhasePairRuntimeLease) -> bytes:
        if type(lease) is not runtime_lease_type:
            raise TypeError(
                "lease must be exactly ResolvedPhasePairRuntimeLease"
            )
        if issued.get(lease) is None:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED")
        raise AssertionError("empty production issuer registry was populated")

    def consume(
        lease: ResolvedPhasePairRuntimeLease,
        *,
        training_authority: TrainingAuthorityLease,
    ) -> PhasePairTrainingRuntime:
        if type(lease) is not runtime_lease_type:
            raise TypeError(
                "lease must be exactly ResolvedPhasePairRuntimeLease"
            )
        if type(training_authority) is not authority_type:
            raise TypeError(
                "training_authority must be exactly TrainingAuthorityLease"
            )
        if issued.get(lease) is None:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED")
        raise AssertionError("empty production issuer registry was populated")

    def constructor_count() -> int:
        return adamw_constructor_call_count

    def lease_count() -> int:
        return len(issued)

    return issue, canonical, consume, constructor_count, lease_count


(
    issue_phasepair_runtime_lease,
    canonical_runtime_receipt_bytes,
    consume_phasepair_runtime_lease,
    _production_adamw_constructor_call_count_for_tests,
    _production_live_lease_count_for_tests,
) = _seal_absent_production_api()
del _seal_absent_production_api


class _SyntheticContractSnapshot:
    """Exact primitive-only contract values for the isolated test factory."""

    __slots__ = ("contract_family", "payload_label")

    def __init__(self, contract_family: str, payload_label: str) -> None:
        if type(contract_family) is not str or not contract_family:
            raise TypeError("contract_family must be an exact nonempty str")
        if type(payload_label) is not str or not payload_label:
            raise TypeError("payload_label must be an exact nonempty str")
        self.contract_family = contract_family
        self.payload_label = payload_label


class _SyntheticFilesystemOps:
    """Primitive preflight verdict; no filesystem callable is accepted."""

    __slots__ = ("snapshot_stable",)

    def __init__(self, snapshot_stable: bool = True) -> None:
        if type(snapshot_stable) is not bool:
            raise TypeError("snapshot_stable must be an exact bool")
        self.snapshot_stable = snapshot_stable


class _SyntheticLoaderOps:
    """Primitive loader/dropout verdict; no model or loader is accepted."""

    __slots__ = ("runtime_complete", "active_text_dropout_sites")

    def __init__(
        self,
        runtime_complete: bool = True,
        active_text_dropout_sites: int = 0,
    ) -> None:
        if type(runtime_complete) is not bool:
            raise TypeError("runtime_complete must be an exact bool")
        if (
            type(active_text_dropout_sites) is not int
            or active_text_dropout_sites < 0
        ):
            raise TypeError(
                "active_text_dropout_sites must be an exact nonnegative int"
            )
        self.runtime_complete = runtime_complete
        self.active_text_dropout_sites = active_text_dropout_sites


class _SyntheticOptimizerOps:
    """Private constructor spy target for lifecycle tests only."""

    __slots__ = ("constructor",)

    def __init__(self, constructor: Callable[[tuple[str, ...]], object]) -> None:
        if not callable(constructor):
            raise TypeError("constructor must be callable")
        self.constructor = constructor


class _SyntheticRuntimeApi:
    """Closed holder for one isolated synthetic issuer family."""

    __slots__ = (
        "resolution_authority_type",
        "training_authority_type",
        "runtime_lease_type",
        "training_runtime_type",
        "issue_resolution_authority",
        "issue_training_authority",
        "issue_runtime_lease",
        "canonical_receipt_bytes",
        "consume_runtime_lease",
        "runtime_lease_state",
        "adamw_constructor_call_count",
        "live_runtime_lease_count",
    )

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("_SyntheticRuntimeApi construction is internal")


def _seal_text_runtime_api_for_tests(
    contract_snapshot: _SyntheticContractSnapshot,
    filesystem_ops: _SyntheticFilesystemOps,
    loader_ops: _SyntheticLoaderOps,
    optimizer_ops: _SyntheticOptimizerOps,
) -> _SyntheticRuntimeApi:
    """Create an isolated authority-zero lifecycle harness.

    This seam accepts only tiny primitive verdicts.  It does not import a model
    runtime, inspect files, construct a production optimizer, or issue a type
    accepted by the production API.
    """

    if type(contract_snapshot) is not _SyntheticContractSnapshot:
        raise TypeError(
            "contract_snapshot must be exactly _SyntheticContractSnapshot"
        )
    if type(filesystem_ops) is not _SyntheticFilesystemOps:
        raise TypeError("filesystem_ops must be exactly _SyntheticFilesystemOps")
    if type(loader_ops) is not _SyntheticLoaderOps:
        raise TypeError("loader_ops must be exactly _SyntheticLoaderOps")
    if type(optimizer_ops) is not _SyntheticOptimizerOps:
        raise TypeError("optimizer_ops must be exactly _SyntheticOptimizerOps")

    contract_family = contract_snapshot.contract_family
    payload_label = contract_snapshot.payload_label
    snapshot_stable = filesystem_ops.snapshot_stable
    runtime_complete = loader_ops.runtime_complete
    active_dropout = loader_ops.active_text_dropout_sites
    optimizer_constructor = optimizer_ops.constructor

    sha256_factory = hashlib.sha256
    json_dumps = json.dumps
    rlock_factory = threading.RLock
    weak_registry_type = weakref.WeakKeyDictionary
    hold_type = TextRuntimeHold
    issued_state = 0
    consuming_state = 1
    consumed_state = 2
    burned_state = 3

    class SyntheticResolutionAuthorityLease:
        __slots__ = ("_lock", "_nonce", "__weakref__")

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise TypeError("synthetic resolution authority construction is internal")

    class SyntheticTrainingAuthorityLease:
        __slots__ = ("_lock", "_nonce", "__weakref__")

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise TypeError("synthetic training authority construction is internal")

    class SyntheticRuntimeLease:
        __slots__ = (
            "_lock",
            "_payload",
            "_payload_sha256",
            "_resolution_authority_reference",
            "_nonce",
            "__weakref__",
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise TypeError("synthetic runtime lease construction is internal")

    class SyntheticTrainingRuntime:
        __slots__ = (
            "_optimizer",
            "_payload",
            "_receipt_sha256",
            "__weakref__",
        )

        def __init__(self, *_args: object, **_kwargs: object) -> None:
            raise TypeError("synthetic training runtime construction is internal")

    capability_registry: weakref.WeakKeyDictionary[
        object, tuple[object, ...]
    ] = weak_registry_type()
    runtime_registry: weakref.WeakKeyDictionary[
        object, tuple[object, ...]
    ] = weak_registry_type()
    next_nonce = 0
    constructor_call_count = 0

    def canonical_json(value: object) -> bytes:
        return (
            json_dumps(
                value,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    def new_nonce() -> int:
        nonlocal next_nonce
        value = next_nonce
        next_nonce += 1
        return value

    def issue_capability(capability_type: type[object]) -> object:
        capability = object.__new__(capability_type)
        lock = rlock_factory()
        nonce = new_nonce()
        object.__setattr__(capability, "_lock", lock)
        object.__setattr__(capability, "_nonce", nonce)
        capability_registry[capability] = (
            id(capability),
            issued_state,
            nonce,
            lock,
        )
        return capability

    def issue_resolution_authority() -> object:
        return issue_capability(SyntheticResolutionAuthorityLease)

    def issue_training_authority() -> object:
        return issue_capability(SyntheticTrainingAuthorityLease)

    def consume_capability(
        capability: object,
        capability_type: type[object],
        error_code: str,
    ) -> None:
        if type(capability) is not capability_type:
            raise hold_type(error_code)
        record = capability_registry.get(capability)
        if record is None:
            raise hold_type(error_code)
        trusted_lock = record[3]
        try:
            with trusted_lock:
                record = capability_registry.get(capability)
                if record is None or record[1] != issued_state:
                    raise hold_type(error_code)
                try:
                    observed_lock = object.__getattribute__(
                        capability,
                        "_lock",
                    )
                    nonce = object.__getattribute__(capability, "_nonce")
                    if (
                        record[0] != id(capability)
                        or record[2] != nonce
                        or record[3] is not trusted_lock
                        or observed_lock is not trusted_lock
                    ):
                        raise hold_type(error_code)
                except BaseException:
                    capability_registry[capability] = (
                        record[0],
                        burned_state,
                        record[2],
                        record[3],
                    )
                    raise
                capability_registry[capability] = (
                    record[0],
                    consumed_state,
                    record[2],
                    record[3],
                )
        except BaseException:
            current = capability_registry.get(capability)
            if current is not None and current[1] == issued_state:
                capability_registry[capability] = (
                    current[0],
                    burned_state,
                    current[2],
                    current[3],
                )
            raise

    def payload_sha256(payload: tuple[str, ...]) -> bytes:
        return sha256_factory(
            canonical_json(
                {
                    "payload": list(payload),
                    "schema": "phasepair-synthetic-runtime-payload-v1",
                }
            )
        ).digest()

    def issue_runtime_lease(
        resolution_authority: object,
        *,
        payload: tuple[str, ...] = ("tiny",),
    ) -> object:
        if type(payload) is not tuple or not payload:
            raise TypeError("payload must be an exact nonempty built-in tuple")
        if any(type(item) is not str or not item for item in payload):
            raise TypeError("payload items must be exact nonempty built-in strings")
        consume_capability(
            resolution_authority,
            SyntheticResolutionAuthorityLease,
            "HOLD_RESOLUTION_AUTHORITY_NOT_ISSUED",
        )
        lease = object.__new__(SyntheticRuntimeLease)
        lock = rlock_factory()
        nonce = new_nonce()
        digest = payload_sha256(payload)
        object.__setattr__(lease, "_lock", lock)
        object.__setattr__(lease, "_payload", payload)
        object.__setattr__(lease, "_payload_sha256", digest)
        object.__setattr__(
            lease,
            "_resolution_authority_reference",
            resolution_authority,
        )
        object.__setattr__(lease, "_nonce", nonce)
        runtime_registry[lease] = (
            id(lease),
            issued_state,
            payload,
            digest,
            lock,
            resolution_authority,
            nonce,
        )
        return lease

    def burn_locked(lease: object, record: tuple[object, ...]) -> None:
        runtime_registry[lease] = (
            record[0],
            burned_state,
            (),
            b"",
            record[4],
            None,
            record[6],
        )
        for field, empty in (
            ("_payload", ()),
            ("_payload_sha256", b""),
            ("_resolution_authority_reference", None),
        ):
            try:
                object.__setattr__(lease, field, empty)
            except (AttributeError, TypeError):
                pass

    def validate_runtime_locked(
        lease: object,
        record: tuple[object, ...],
        trusted_lock: object,
    ) -> tuple[str, ...]:
        try:
            observed_lock = object.__getattribute__(lease, "_lock")
            payload = object.__getattribute__(lease, "_payload")
            digest = object.__getattribute__(lease, "_payload_sha256")
            authority = object.__getattribute__(
                lease,
                "_resolution_authority_reference",
            )
            nonce = object.__getattribute__(lease, "_nonce")
        except AttributeError as exc:
            raise hold_type("HOLD_RUNTIME_LEASE_MUTATED") from exc
        if (
            record[0] != id(lease)
            or record[2] is not payload
            or type(payload) is not tuple
            or record[3] != digest
            or type(digest) is not bytes
            or len(digest) != 32
            or record[4] is not trusted_lock
            or observed_lock is not trusted_lock
            or record[5] is not authority
            or record[6] != nonce
            or payload_sha256(payload) != digest
        ):
            raise hold_type("HOLD_RUNTIME_LEASE_MUTATED")
        return payload

    def receipt_payload(
        lease: object,
        record: tuple[object, ...],
        trusted_lock: object,
    ) -> dict[str, object]:
        payload = validate_runtime_locked(lease, record, trusted_lock)
        return {
            "authority": 0,
            "contract_family": contract_family,
            "hold_reasons": ["SYNTHETIC_NONPRODUCTION_ONLY"],
            "lifecycle": "ISSUED",
            "payload": list(payload),
            "payload_label": payload_label,
            "production": False,
            "schema": "phasepair-synthetic-live-text-runtime-receipt-v1",
            "status": "SYNTHETIC_AUTHORITY0_NONPRODUCTION",
            "training_authorized": False,
        }

    def canonical_receipt_bytes(lease: object) -> bytes:
        if type(lease) is not SyntheticRuntimeLease:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY")
        record = runtime_registry.get(lease)
        if record is None:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY")
        trusted_lock = record[4]
        try:
            with trusted_lock:
                record = runtime_registry.get(lease)
                if record is None or record[1] != issued_state:
                    raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_STATE")
                try:
                    receipt = canonical_json(
                        receipt_payload(lease, record, trusted_lock)
                    )
                    validate_runtime_locked(lease, record, trusted_lock)
                    return receipt
                except BaseException:
                    burn_locked(lease, record)
                    raise
        except BaseException:
            current = runtime_registry.get(lease)
            if current is not None and current[1] == issued_state:
                burn_locked(lease, current)
            raise

    def consume_runtime_lease(
        lease: object,
        *,
        training_authority: object,
    ) -> object:
        nonlocal constructor_call_count
        if type(lease) is not SyntheticRuntimeLease:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY")
        record = runtime_registry.get(lease)
        if record is None:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY")
        trusted_lock = record[4]
        claimed_issued_lease = False
        try:
            with trusted_lock:
                record = runtime_registry.get(lease)
                if record is None or record[1] != issued_state:
                    raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_STATE")
                claimed_issued_lease = True
                consuming = (
                    record[0],
                    consuming_state,
                    record[2],
                    record[3],
                    record[4],
                    record[5],
                    record[6],
                )
                try:
                    runtime_registry[lease] = consuming
                    payload = validate_runtime_locked(
                        lease,
                        consuming,
                        trusted_lock,
                    )
                    consume_capability(
                        training_authority,
                        SyntheticTrainingAuthorityLease,
                        "HOLD_TRAINING_AUTHORITY_NOT_ISSUED",
                    )
                    if (
                        not snapshot_stable
                        or not runtime_complete
                        or active_dropout != 0
                    ):
                        raise hold_type("HOLD_SYNTHETIC_PREFLIGHT")
                    receipt = canonical_json(
                        {
                            "authority": 0,
                            "contract_family": contract_family,
                            "payload": list(payload),
                            "production": False,
                            "schema": (
                                "phasepair-synthetic-live-text-runtime-receipt-v1"
                            ),
                            "status": "SYNTHETIC_AUTHORITY0_NONPRODUCTION",
                            "training_authorized": False,
                        }
                    )
                    validate_runtime_locked(lease, consuming, trusted_lock)
                    constructor_call_count += 1
                    optimizer = optimizer_constructor(payload)
                    if optimizer is None:
                        raise hold_type(
                            "HOLD_SYNTHETIC_OPTIMIZER_POSTCONDITION"
                        )
                    validate_runtime_locked(lease, consuming, trusted_lock)
                    runtime = object.__new__(SyntheticTrainingRuntime)
                    object.__setattr__(runtime, "_optimizer", optimizer)
                    object.__setattr__(runtime, "_payload", payload)
                    object.__setattr__(
                        runtime,
                        "_receipt_sha256",
                        sha256_factory(receipt).digest(),
                    )
                    runtime_registry[lease] = (
                        consuming[0],
                        consumed_state,
                        (),
                        b"",
                        consuming[4],
                        None,
                        consuming[6],
                    )
                    object.__setattr__(lease, "_payload", ())
                    object.__setattr__(lease, "_payload_sha256", b"")
                    object.__setattr__(
                        lease,
                        "_resolution_authority_reference",
                        None,
                    )
                    return runtime
                except BaseException:
                    current = runtime_registry.get(lease)
                    if current is not None:
                        burn_locked(lease, current)
                    raise
        except BaseException:
            current = runtime_registry.get(lease)
            if current is not None and (
                claimed_issued_lease or current[1] == issued_state
            ):
                burn_locked(lease, current)
            raise

    def runtime_lease_state(lease: object) -> int:
        if type(lease) is not SyntheticRuntimeLease:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY")
        record = runtime_registry.get(lease)
        if record is None:
            raise hold_type("HOLD_RUNTIME_LEASE_NOT_ISSUED_BY_FACTORY")
        state = record[1]
        if type(state) is not int or state not in {
            issued_state,
            consuming_state,
            consumed_state,
            burned_state,
        }:
            raise AssertionError("synthetic lifecycle registry is malformed")
        return state

    def adamw_constructor_call_count() -> int:
        return constructor_call_count

    def live_runtime_lease_count() -> int:
        return len(runtime_registry)

    api = object.__new__(_SyntheticRuntimeApi)
    values = MappingProxyType(
        {
            "resolution_authority_type": SyntheticResolutionAuthorityLease,
            "training_authority_type": SyntheticTrainingAuthorityLease,
            "runtime_lease_type": SyntheticRuntimeLease,
            "training_runtime_type": SyntheticTrainingRuntime,
            "issue_resolution_authority": issue_resolution_authority,
            "issue_training_authority": issue_training_authority,
            "issue_runtime_lease": issue_runtime_lease,
            "canonical_receipt_bytes": canonical_receipt_bytes,
            "consume_runtime_lease": consume_runtime_lease,
            "runtime_lease_state": runtime_lease_state,
            "adamw_constructor_call_count": adamw_constructor_call_count,
            "live_runtime_lease_count": live_runtime_lease_count,
        }
    )
    for field in _SyntheticRuntimeApi.__slots__:
        object.__setattr__(api, field, values[field])
    return api


__all__ = [
    "PhasePairTrainingRuntime",
    "ResolutionAuthorityLease",
    "ResolvedPhasePairRuntimeLease",
    "TrainingAuthorityLease",
    "canonical_runtime_receipt_bytes",
    "consume_phasepair_runtime_lease",
    "issue_phasepair_runtime_lease",
]
