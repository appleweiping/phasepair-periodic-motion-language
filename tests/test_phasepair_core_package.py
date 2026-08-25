from __future__ import annotations

import phasepair_core


def test_package_is_explicitly_nonproduction() -> None:
    assert phasepair_core.STATUS == "DATA_FREE_NONPRODUCTION_AUTHORITY0"
    assert phasepair_core.CONTRACT_FAMILY.endswith("20260824_165840")
    assert tuple(phasepair_core.__all__) == ("CONTRACT_FAMILY", "STATUS")
