"""Public, data-free PhaseSet research core.

PhaseSet extends the immutable PhasePair v0.1.0 pair primitive to dynamically
padded groups. The package deliberately ships no datasets, model weights,
private identifiers, server locators, checkpoints, or empirical claims.
"""

from __future__ import annotations


__version__ = "0.2.0a1"
STATUS = "DATA_FREE_MULTIPERSON_AUTHORITY0"
CONTRACT_FAMILY = "PHASESET_SCIENTIFIC_CONTRACT_V1"
LEGACY_CONTRACT_FAMILY = "PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840"

__all__ = [
    "CONTRACT_FAMILY",
    "LEGACY_CONTRACT_FAMILY",
    "STATUS",
    "__version__",
]
