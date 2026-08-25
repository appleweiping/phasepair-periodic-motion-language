"""Contract-bound, data-free PhasePair core primitives.

The package is intentionally non-production: it contains deterministic local
oracles and validators only.  It does not load InterHuman, CLIP weights, private
lineage material, checkpoints, or experiment results.
"""

from __future__ import annotations


STATUS = "DATA_FREE_NONPRODUCTION_AUTHORITY0"
CONTRACT_FAMILY = "PHASEPAIR_SCIENTIFIC_CONTRACT_20260824_165840"

__all__ = ["CONTRACT_FAMILY", "STATUS"]
