"""Fail-closed PhasePair runtime and training-readiness observations.

The functions here deliberately do not turn opaque hashes into authority.  A
complete set of references is reported as *verification required*, never as a
training grant.
"""

from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from typing import Callable

import numpy as np


SIGNAL_PYTHON = "3.14.5"
SIGNAL_NUMPY = "2.4.6"
CONTRACT_FAMILY = "PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840"
STATUS_HOLD = "HOLD"
STATUS_REFERENCES_PRESENT = "REFERENCES_PRESENT_VERIFICATION_REQUIRED"
AUTHORITY = 0

GATE_FIELDS = (
    "resolved_clip_receipt_sha256",
    "dataset_rights_receipt_sha256",
    "owner_attestation_sha256",
    "lineage_manifest_sha256",
    "caption_manifest_sha256",
    "energy_floor_receipt_sha256",
    "server_authority_receipt_sha256",
    "gpu_runtime_receipt_sha256",
)


class ReadinessError(ValueError):
    """Raised for malformed local observations or gate references."""


def _require_exact_str(value: object, label: str) -> str:
    if type(value) is not str or not value:
        raise TypeError(f"{label} must be a nonempty exact built-in str")
    return value


def _require_exact_bool(value: object, label: str) -> bool:
    if type(value) is not bool:
        raise TypeError(f"{label} must be an exact built-in bool")
    return value


def _require_raw32_or_none(value: object, label: str) -> bytes | None:
    if value is None:
        return None
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be None or exact built-in bytes[32]")
    return value


@dataclass(frozen=True, slots=True)
class RuntimeObservation:
    python_version: str
    python_implementation: str
    numpy_version: str
    torch_version: str
    torch_cuda_version: str
    cuda_available: bool
    platform_string: str

    def __post_init__(self) -> None:
        for label in (
            "python_version",
            "python_implementation",
            "numpy_version",
            "torch_version",
            "torch_cuda_version",
            "platform_string",
        ):
            _require_exact_str(getattr(self, label), label)
        _require_exact_bool(self.cuda_available, "cuda_available")


def observe_local_runtime(
    *,
    torch_importer: Callable[[], object] | None = None,
) -> RuntimeObservation:
    """Observe the local process only; this is not a deployment receipt."""

    if torch_importer is None:
        import torch

        torch_module = torch
    else:
        if not callable(torch_importer):
            raise TypeError("torch_importer must be callable")
        torch_module = torch_importer()
    torch_version = _require_exact_str(getattr(torch_module, "__version__"), "torch version")
    version_object = getattr(torch_module, "version")
    raw_cuda = getattr(version_object, "cuda")
    cuda_version = "NONE" if raw_cuda is None else _require_exact_str(raw_cuda, "CUDA version")
    cuda_object = getattr(torch_module, "cuda")
    cuda_available = getattr(cuda_object, "is_available")()
    _require_exact_bool(cuda_available, "torch.cuda.is_available result")
    return RuntimeObservation(
        platform.python_version(),
        platform.python_implementation(),
        np.__version__,
        torch_version,
        cuda_version,
        cuda_available,
        platform.platform(),
    )


def validate_signal_oracle_runtime(observation: RuntimeObservation) -> str:
    if type(observation) is not RuntimeObservation:
        raise TypeError("observation must be exactly RuntimeObservation")
    checked = RuntimeObservation(
        observation.python_version,
        observation.python_implementation,
        observation.numpy_version,
        observation.torch_version,
        observation.torch_cuda_version,
        observation.cuda_available,
        observation.platform_string,
    )
    if checked.python_implementation != "CPython":
        raise ReadinessError("signal oracle requires CPython")
    if checked.python_version != "3.14.5" or checked.numpy_version != "2.4.6":
        raise ReadinessError("signal oracle runtime version mismatch")
    return "SIGNAL_ORACLE_RUNTIME_MATCH_NONPRODUCTION"


@dataclass(frozen=True, slots=True)
class TrainingGateReferences:
    scientific_contract_sha256: bytes
    resolved_clip_receipt_sha256: bytes | None = None
    dataset_rights_receipt_sha256: bytes | None = None
    owner_attestation_sha256: bytes | None = None
    lineage_manifest_sha256: bytes | None = None
    caption_manifest_sha256: bytes | None = None
    energy_floor_receipt_sha256: bytes | None = None
    server_authority_receipt_sha256: bytes | None = None
    gpu_runtime_receipt_sha256: bytes | None = None

    def __post_init__(self) -> None:
        if type(self.scientific_contract_sha256) is not bytes or len(self.scientific_contract_sha256) != 32:
            raise TypeError("scientific_contract_sha256 must be exact built-in bytes[32]")
        for field in (
            "resolved_clip_receipt_sha256",
            "dataset_rights_receipt_sha256",
            "owner_attestation_sha256",
            "lineage_manifest_sha256",
            "caption_manifest_sha256",
            "energy_floor_receipt_sha256",
            "server_authority_receipt_sha256",
            "gpu_runtime_receipt_sha256",
        ):
            _require_raw32_or_none(getattr(self, field), field)


@dataclass(frozen=True, slots=True)
class TrainingReadinessAssessment:
    status: str
    authority: int
    production: bool
    training_authorized: bool
    missing_gates: tuple[str, ...]
    contract_family: str

    def __post_init__(self) -> None:
        if type(self.status) is not str or self.status not in {
            "HOLD",
            "REFERENCES_PRESENT_VERIFICATION_REQUIRED",
        }:
            raise ReadinessError("unknown readiness status")
        if type(self.authority) is not int or self.authority != 0:
            raise ReadinessError("authority must remain literal zero")
        if type(self.production) is not bool or self.production:
            raise ReadinessError("production must remain literal false")
        if type(self.training_authorized) is not bool or self.training_authorized:
            raise ReadinessError("training_authorized must remain literal false")
        if type(self.missing_gates) is not tuple:
            raise TypeError("missing_gates must be an exact built-in tuple")
        gate_fields = (
            "resolved_clip_receipt_sha256",
            "dataset_rights_receipt_sha256",
            "owner_attestation_sha256",
            "lineage_manifest_sha256",
            "caption_manifest_sha256",
            "energy_floor_receipt_sha256",
            "server_authority_receipt_sha256",
            "gpu_runtime_receipt_sha256",
        )
        for index, gate in enumerate(self.missing_gates):
            if type(gate) is not str or gate not in gate_fields:
                raise ReadinessError(f"unknown missing gate at index {index}")
        expected_order = tuple(field for field in gate_fields if field in self.missing_gates)
        if self.missing_gates != expected_order or len(set(self.missing_gates)) != len(
            self.missing_gates
        ):
            raise ReadinessError("missing gates must be unique and in canonical order")
        if self.status == "HOLD" and not self.missing_gates:
            raise ReadinessError("HOLD requires at least one missing gate")
        if self.status == "REFERENCES_PRESENT_VERIFICATION_REQUIRED" and self.missing_gates:
            raise ReadinessError("references-present status forbids missing gates")
        if (
            type(self.contract_family) is not str
            or self.contract_family != "PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840"
        ):
            raise ReadinessError("contract family mismatch")


def assess_training_readiness(references: TrainingGateReferences) -> TrainingReadinessAssessment:
    """Check reference presence while refusing to elevate unverified digests."""

    if type(references) is not TrainingGateReferences:
        raise TypeError("references must be exactly TrainingGateReferences")
    gate_fields = (
        "resolved_clip_receipt_sha256",
        "dataset_rights_receipt_sha256",
        "owner_attestation_sha256",
        "lineage_manifest_sha256",
        "caption_manifest_sha256",
        "energy_floor_receipt_sha256",
        "server_authority_receipt_sha256",
        "gpu_runtime_receipt_sha256",
    )
    checked = TrainingGateReferences(
        references.scientific_contract_sha256,
        *(getattr(references, field) for field in gate_fields),
    )
    missing = tuple(field for field in gate_fields if getattr(checked, field) is None)
    status = "HOLD" if missing else "REFERENCES_PRESENT_VERIFICATION_REQUIRED"
    return TrainingReadinessAssessment(
        status,
        0,
        False,
        False,
        missing,
        "PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840",
    )


def canonical_assessment_bytes(assessment: TrainingReadinessAssessment) -> bytes:
    if type(assessment) is not TrainingReadinessAssessment:
        raise TypeError("assessment must be exactly TrainingReadinessAssessment")
    checked = TrainingReadinessAssessment(
        assessment.status,
        assessment.authority,
        assessment.production,
        assessment.training_authorized,
        assessment.missing_gates,
        assessment.contract_family,
    )
    payload = {
        "authority": checked.authority,
        "contract_family": checked.contract_family,
        "missing_gates": list(checked.missing_gates),
        "production": checked.production,
        "status": checked.status,
        "training_authorized": checked.training_authorized,
    }
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def hold_assessment() -> TrainingReadinessAssessment:
    return assess_training_readiness(TrainingGateReferences(b"\x00" * 32))


def main() -> int:
    sys.stdout.buffer.write(canonical_assessment_bytes(hold_assessment()))
    return 42


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AUTHORITY",
    "CONTRACT_FAMILY",
    "GATE_FIELDS",
    "ReadinessError",
    "RuntimeObservation",
    "SIGNAL_NUMPY",
    "SIGNAL_PYTHON",
    "STATUS_HOLD",
    "STATUS_REFERENCES_PRESENT",
    "TrainingGateReferences",
    "TrainingReadinessAssessment",
    "assess_training_readiness",
    "canonical_assessment_bytes",
    "hold_assessment",
    "main",
    "observe_local_runtime",
    "validate_signal_oracle_runtime",
]
