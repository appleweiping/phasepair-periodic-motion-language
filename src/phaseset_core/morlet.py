"""Independent 20-Hz Morlet oracle for PhaseSet skeleton activity.

PhasePair v0.1.0 remains a byte-frozen 30-Hz oracle.  PhaseSet consumes motion
resampled to 20 Hz, so reusing those tap bytes would shift every physical center
frequency by 2/3.  This module retains the registered six center frequencies
and the same zero-DC, unit-energy, three-cycle formula, while deriving a new
20-Hz bank with its own schema and immutable digest.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Final

import numpy as np


STATUS: Final = "DATA_FREE_PHASESET_20HZ_MORLET_ORACLE_AUTHORITY0"
PHASESET_MORLET_SCHEMA: Final = "phaseset-morlet-bank-v1/fs20/three-cycle"
SAMPLE_RATE_HZ: Final = 20.0
MORLET_FREQUENCIES_HZ: Final[tuple[float, ...]] = (
    0.75,
    1.125,
    1.6875,
    2.53125,
    3.796875,
    5.6953125,
)
MORLET_LENGTHS: Final[tuple[int, ...]] = (80, 54, 36, 24, 16, 11)
MORLET_SIGMAS: Final[tuple[float, ...]] = (
    13.333333333333334,
    8.88888888888889,
    5.925925925925926,
    3.950617283950617,
    2.633744855967078,
    1.755829903978052,
)
MORLET_ORACLE_SHA256: Final = (
    "633d754bba12dab0555722f4b8963409a110c8681796cc907d163096e2e42655"
)
MORLET_ZERO_DC_BOUND: Final = 2.220446049250313e-16
MORLET_UNIT_ENERGY_BOUND: Final = 2.220446049250313e-16
MORLET_EXPECTED_ZERO_DC_ERRORS: Final[tuple[float, ...]] = (
    1.1102230246251565e-16,
    1.0702444338622552e-16,
    1.747819990421485e-17,
    3.741738622089172e-17,
    1.6653345369377348e-16,
    1.1816510524180758e-16,
)
MORLET_EXPECTED_UNIT_ENERGY_ERRORS: Final[tuple[float, ...]] = (
    0.0,
    2.220446049250313e-16,
    1.1102230246251565e-16,
    0.0,
    2.220446049250313e-16,
    1.1102230246251565e-16,
)


class PhaseSetMorletError(ValueError):
    """A 20-Hz bank input or numerical invariant is invalid."""


class PhaseSetMorletOracleError(RuntimeError):
    """The rebuilt bank no longer matches the frozen PhaseSet oracle."""


@dataclass(frozen=True, slots=True)
class PhaseSetMorletBand:
    index: int
    frequency_hz: float
    length: int
    sigma: float
    kernel: np.ndarray


@dataclass(frozen=True, slots=True)
class _BuildReceipt:
    bands: tuple[PhaseSetMorletBand, ...]
    zero_dc_errors: tuple[float, ...]
    unit_energy_errors: tuple[float, ...]


def _build_formula_bank() -> _BuildReceipt:
    bands: list[PhaseSetMorletBand] = []
    zero_dc_errors: list[float] = []
    unit_energy_errors: list[float] = []
    for index, (frequency, length, sigma) in enumerate(
        zip(MORLET_FREQUENCIES_HZ, MORLET_LENGTHS, MORLET_SIGMAS, strict=True),
        start=1,
    ):
        if math.ceil(3.0 * SAMPLE_RATE_HZ / frequency) != length:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_SUPPORT_DRIFT")
        if SAMPLE_RATE_HZ / (2.0 * frequency) != sigma:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_SIGMA_DRIFT")
        offsets = np.ascontiguousarray(
            np.arange(length, dtype=np.float64) - (length - 1) / 2.0,
            dtype=np.float64,
        )
        gaussian = np.ascontiguousarray(
            np.exp(-(offsets * offsets) / (2.0 * sigma * sigma)),
            dtype=np.float64,
        )
        carrier = np.ascontiguousarray(
            gaussian
            * np.exp(
                -1j
                * 2.0
                * math.pi
                * frequency
                * offsets
                / SAMPLE_RATE_HZ
            ),
            dtype=np.complex128,
        )
        beta = np.sum(carrier, dtype=np.complex128) / np.sum(
            gaussian, dtype=np.float64
        )
        raw = np.ascontiguousarray(carrier - beta * gaussian, dtype=np.complex128)
        energy = np.sum(
            np.square(np.abs(raw), dtype=np.float64),
            dtype=np.float64,
        )
        if not math.isfinite(float(energy)) or energy <= 0.0:
            raise PhaseSetMorletOracleError("PHASESET_MORLET_RAW_ENERGY_INVALID")
        kernel = np.ascontiguousarray(raw / np.sqrt(energy), dtype=np.complex128)
        zero_dc_errors.append(
            float(np.abs(np.sum(kernel, dtype=np.complex128)))
        )
        unit_energy_errors.append(
            float(
                np.abs(
                    np.sum(
                        np.square(np.abs(kernel), dtype=np.float64),
                        dtype=np.float64,
                    )
                    - np.float64(1.0)
                )
            )
        )
        kernel.setflags(write=False)
        bands.append(PhaseSetMorletBand(index, frequency, length, sigma, kernel))
    return _BuildReceipt(
        tuple(bands),
        tuple(zero_dc_errors),
        tuple(unit_energy_errors),
    )


def morlet_bank_sha256(
    bands: tuple[PhaseSetMorletBand, ...] | None = None,
) -> str:
    """Hash uint16be lengths followed by float64be real/imaginary taps."""

    selected = morlet_kernel_bank() if bands is None else bands
    if type(selected) is not tuple or len(selected) != 6:
        raise PhaseSetMorletError("bands must be the exact six-band tuple")
    digest = hashlib.sha256()
    for expected_index, band in enumerate(selected, start=1):
        if type(band) is not PhaseSetMorletBand or band.index != expected_index:
            raise PhaseSetMorletError("band type or order is invalid")
        digest.update(struct.pack(">H", band.length))
        for value in band.kernel:
            digest.update(struct.pack(">dd", float(value.real), float(value.imag)))
    return digest.hexdigest()


def morlet_formula_bank_diagnostic() -> tuple[PhaseSetMorletBand, ...]:
    """Replay the exact 20-Hz formula without granting empirical authority."""

    return _build_formula_bank().bands


def morlet_kernel_bank() -> tuple[PhaseSetMorletBand, ...]:
    """Rebuild and enforce the complete independent PhaseSet 20-Hz oracle."""

    receipt = _build_formula_bank()
    if morlet_bank_sha256(receipt.bands) != MORLET_ORACLE_SHA256:
        raise PhaseSetMorletOracleError("PHASESET_MORLET_DIGEST_MISMATCH")
    if receipt.zero_dc_errors != MORLET_EXPECTED_ZERO_DC_ERRORS:
        raise PhaseSetMorletOracleError("PHASESET_MORLET_ZERO_DC_ORACLE_FAIL")
    if receipt.unit_energy_errors != MORLET_EXPECTED_UNIT_ENERGY_ERRORS:
        raise PhaseSetMorletOracleError("PHASESET_MORLET_UNIT_ENERGY_ORACLE_FAIL")
    if any(error > MORLET_ZERO_DC_BOUND for error in receipt.zero_dc_errors):
        raise PhaseSetMorletOracleError("PHASESET_MORLET_ZERO_DC_BOUND_FAIL")
    if any(error > MORLET_UNIT_ENERGY_BOUND for error in receipt.unit_energy_errors):
        raise PhaseSetMorletOracleError("PHASESET_MORLET_UNIT_ENERGY_BOUND_FAIL")
    return receipt.bands


def morlet_length_mask(valid_length: object) -> np.ndarray:
    if type(valid_length) is not int:
        raise TypeError("valid_length must be an exact built-in int")
    if not 1 <= valid_length <= 299:
        raise PhaseSetMorletError("valid_length must satisfy 1<=N<=299")
    return np.asarray(
        [valid_length >= support for support in MORLET_LENGTHS],
        dtype=np.uint8,
    )


__all__ = [
    "MORLET_EXPECTED_UNIT_ENERGY_ERRORS",
    "MORLET_EXPECTED_ZERO_DC_ERRORS",
    "MORLET_FREQUENCIES_HZ",
    "MORLET_LENGTHS",
    "MORLET_ORACLE_SHA256",
    "MORLET_SIGMAS",
    "MORLET_UNIT_ENERGY_BOUND",
    "MORLET_ZERO_DC_BOUND",
    "PHASESET_MORLET_SCHEMA",
    "PhaseSetMorletBand",
    "PhaseSetMorletError",
    "PhaseSetMorletOracleError",
    "SAMPLE_RATE_HZ",
    "STATUS",
    "morlet_bank_sha256",
    "morlet_formula_bank_diagnostic",
    "morlet_kernel_bank",
    "morlet_length_mask",
]
