from __future__ import annotations

import hashlib

import pytest

from phasepair_core.lineage import (
    STATUS,
    LineageError,
    actor_commitment,
    build_actor_pair_manifest,
    canonical_json_bytes,
    crop_digest,
    crop_start,
    domain_hash,
    lineage_key_status,
    pair_commitment,
    source_lineage_bytes,
    validate_commitment_uniqueness,
)


def _row(ordinal: int, source_id: str = "QQ==") -> dict[str, object]:
    return {
        "schema": "phasepair-asset-source-authority-row-v1",
        "partition": "train",
        "partition_code": 0,
        "split_raw_sha256": "1" * 64,
        "official_source_line_ordinal": ordinal,
        "id_b64": source_id,
        "T": 305,
        "actor0_file_sha256": "2" * 64,
        "actor1_file_sha256": "3" * 64,
    }


def test_canonical_json_and_domain_hash_are_exact() -> None:
    value = {"z": "运动", "a": [1, True, None]}
    assert canonical_json_bytes(value) == '{"a":[1,true,null],"z":"运动"}\n'.encode()
    expected = hashlib.sha256(b"tag\0" + canonical_json_bytes(value)).digest()
    assert domain_hash("tag", value) == expected
    with pytest.raises(LineageError):
        canonical_json_bytes({"bad": float("nan")})


def test_source_actor_pair_and_manifest_contracts() -> None:
    allowlist = bytes(range(32))
    key = bytes(reversed(range(32)))
    source = source_lineage_bytes(_row(0), allowlist)
    assert len(source) == 97
    actor0 = actor_commitment(key, source, 0)
    actor1 = actor_commitment(key, source, 1)
    pair = pair_commitment(actor0, actor1)
    assert pair_commitment(actor1, actor0) == pair
    status, status_digest = lineage_key_status(key)
    assert status["key_status"] == "KEY_PRESENT_PRIVATE"
    assert len(status_digest) == 32
    manifest = build_actor_pair_manifest([_row(0), _row(1, "Qg==")], allowlist, key)
    assert manifest.row_count == 2
    assert len(manifest.private_bytes) == 41 + 32 + 32 + 4 + 2 * 193
    assert hashlib.sha256(manifest.private_bytes).hexdigest() == manifest.sha256
    assert STATUS.endswith("AUTHORITY0")


def test_actor_uniqueness_is_unconditional() -> None:
    actor = b"a" * 32
    pair = b"p" * 32
    with pytest.raises(LineageError, match="DUPLICATE_FATAL"):
        validate_commitment_uniqueness([actor, actor], [pair])
    with pytest.raises(LineageError, match="duplicate pair"):
        validate_commitment_uniqueness([b"a" * 32, b"b" * 32], [pair, pair])
    with pytest.raises(LineageError, match="distinct"):
        pair_commitment(actor, actor)


def test_g0_pair_and_crop_goldens() -> None:
    # G0a's crop-order fixture intentionally overrides the general signal
    # goldens with externally supplied all-zero/all-one commitments.
    actor_a = bytes(32)
    actor_b = b"\xff" * 32
    pair = pair_commitment(actor_a, actor_b)
    assert pair.hex() == "1cea6bd89332169d245430c8c5001a5be90d34e7926812bdfec0a5a07bf01d80"
    assert crop_digest(pair, 0).hex() == (
        "03449cbd8f20b23df926eae3f2e787df4f6931c2e54219283a2bc5a04b3596de"
    )
    assert crop_start(pair, 0, 305, "train") == 5
    assert crop_start(pair, 0, 305, "val") == 2
    assert crop_start(pair, 0, 305, "test") == 2
    assert crop_start(pair, 0, 300, "train") == 0
    assert crop_start(pair, 0, 0, "train") == 0


def test_fail_closed_types_and_closed_census() -> None:
    allowlist = b"a" * 32
    bad = _row(0)
    bad["extra"] = 1
    with pytest.raises(LineageError, match="key census"):
        source_lineage_bytes(bad, allowlist)
    bad = _row(0)
    bad["official_source_line_ordinal"] = True
    with pytest.raises(LineageError, match="exact integer"):
        source_lineage_bytes(bad, allowlist)
    bad = _row(0)
    bad["schema"] = type("SchemaSubclass", (str,), {})(
        "phasepair-asset-source-authority-row-v1"
    )
    with pytest.raises(LineageError, match="schema"):
        source_lineage_bytes(bad, allowlist)
    bad = _row(0)
    bad["id_b64"] = "QQ"
    with pytest.raises(LineageError, match="base64"):
        source_lineage_bytes(bad, allowlist)
    with pytest.raises(LineageError, match="bytes32"):
        actor_commitment(b"short", b"x" * 97, 0)
    with pytest.raises(LineageError, match="bytes32"):
        crop_start(b"short", 0, 300, "train")
    with pytest.raises(LineageError, match="exact integer"):
        crop_start(b"p" * 32, True, 300, "train")
    huge_t = _row(0)
    huge_t["T"] = 2**128
    assert len(source_lineage_bytes(huge_t, allowlist)) == 97
