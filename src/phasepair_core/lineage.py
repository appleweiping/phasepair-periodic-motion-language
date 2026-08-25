"""Data-free PhasePair lineage primitives.

This module implements only the byte contracts inherited by the fresh-CLEAN
``20260824_165840`` scientific contract.  It does not read a dataset, discover
host paths, or grant experiment authority.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import struct
from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence


STATUS = "DATA_FREE_NONPRODUCTION_AUTHORITY0"

_SHA256_HEX = re.compile(r"[0-9a-f]{64}\Z")
_PARTITIONS = {"train": 0, "val": 1, "test": 2}
_ASSET_KEYS = frozenset(
    {
        "schema",
        "partition",
        "partition_code",
        "split_raw_sha256",
        "official_source_line_ordinal",
        "id_b64",
        "T",
        "actor0_file_sha256",
        "actor1_file_sha256",
    }
)


class LineageError(ValueError):
    """Raised when a lineage byte contract is not satisfied."""


def _exact_int(
    value: Any, *, name: str, minimum: int, maximum: int | None
) -> int:
    if type(value) is not int or value < minimum or (
        maximum is not None and value > maximum
    ):
        upper = "unbounded" if maximum is None else str(maximum)
        raise LineageError(f"{name} must be an exact integer in [{minimum}, {upper}]")
    return value


def _raw32(value: Any, *, name: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise LineageError(f"{name} must be exact bytes32")
    return value


def _sha256_hex(value: Any, *, name: str) -> str:
    if type(value) is not str or _SHA256_HEX.fullmatch(value) is None:
        raise LineageError(f"{name} must be lowercase SHA-256 hex")
    return value


def canonical_json_bytes(value: Any) -> bytes:
    """Return the contract's canonical UTF-8 JSON representation plus LF."""

    try:
        text = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise LineageError("value is not canonical-JSON serializable") from exc
    return text.encode("utf-8") + b"\n"


def domain_hash(tag: str, value: Any) -> bytes:
    """Compute ``SHA256(ASCII(tag) || NUL || J(value))``."""

    if type(tag) is not str:
        raise LineageError("domain tag must be an exact string")
    try:
        tag_bytes = tag.encode("ascii")
    except UnicodeEncodeError as exc:
        raise LineageError("domain tag must be ASCII") from exc
    if not tag_bytes or b"\x00" in tag_bytes:
        raise LineageError("domain tag must be nonempty and NUL-free")
    return hashlib.sha256(tag_bytes + b"\x00" + canonical_json_bytes(value)).digest()


def validate_asset_source_authority(value: Mapping[str, Any]) -> dict[str, Any]:
    """Validate and copy a closed ``asset-source-authority-row-v1`` object."""

    if type(value) is not dict or set(value) != _ASSET_KEYS:
        raise LineageError("asset source authority must have the exact key census")
    if (
        type(value["schema"]) is not str
        or value["schema"] != "phasepair-asset-source-authority-row-v1"
    ):
        raise LineageError("wrong asset source authority schema")
    partition = value["partition"]
    if type(partition) is not str or partition not in _PARTITIONS:
        raise LineageError("partition must be train, val, or test")
    partition_code = _exact_int(
        value["partition_code"], name="partition_code", minimum=0, maximum=2
    )
    if partition_code != _PARTITIONS[partition]:
        raise LineageError("partition and partition_code disagree")
    for key in ("split_raw_sha256", "actor0_file_sha256", "actor1_file_sha256"):
        _sha256_hex(value[key], name=key)
    _exact_int(
        value["official_source_line_ordinal"],
        name="official_source_line_ordinal",
        minimum=0,
        maximum=0xFFFFFFFF,
    )
    _exact_int(value["T"], name="T", minimum=0, maximum=None)
    encoded_id = value["id_b64"]
    if type(encoded_id) is not str:
        raise LineageError("id_b64 must be an exact string")
    try:
        decoded_id = base64.b64decode(encoded_id, validate=True)
    except (ValueError, base64.binascii.Error) as exc:
        raise LineageError("id_b64 must be strict padded RFC4648 base64") from exc
    if not decoded_id or base64.b64encode(decoded_id).decode("ascii") != encoded_id:
        raise LineageError("id_b64 is empty or non-canonical")
    return dict(value)


def source_lineage_bytes(
    asset_source_authority: Mapping[str, Any], final_allowlist_raw32: bytes
) -> bytes:
    """Build the exact 97-byte source-lineage record."""

    row = validate_asset_source_authority(asset_source_authority)
    allowlist = _raw32(final_allowlist_raw32, name="final_allowlist_raw32")
    authority_raw32 = domain_hash("phasepair-asset-source-authority-row-v1", row)
    result = (
        b"phasepair-source-lineage-v1\x00"
        + struct.pack(">BI", row["partition_code"], row["official_source_line_ordinal"])
        + allowlist
        + authority_raw32
    )
    if len(result) != 97:
        raise AssertionError("internal source-lineage length mismatch")
    return result


def lineage_key_status(private_lineage_key: bytes | None) -> tuple[dict[str, Any], bytes]:
    """Return the public key-status object and its domain hash, never the key."""

    if private_lineage_key is None:
        value = {
            "schema": "phasepair-lineage-key-status-v1",
            "key_commitment_sha256": "NOT_APPLICABLE",
            "key_status": "KEY_MISSING",
        }
    else:
        key = _raw32(private_lineage_key, name="private_lineage_key")
        commitment = hashlib.sha256(
            b"phasepair-lineage-key-commitment-v1\x00" + key
        ).hexdigest()
        value = {
            "schema": "phasepair-lineage-key-status-v1",
            "key_commitment_sha256": commitment,
            "key_status": "KEY_PRESENT_PRIVATE",
        }
    return value, domain_hash("phasepair-lineage-key-status-v1", value)


def actor_commitment(
    private_lineage_key: bytes, source_lineage: bytes, original_actor_ordinal: int
) -> bytes:
    """Derive one actor HMAC without reading motion or captions."""

    key = _raw32(private_lineage_key, name="private_lineage_key")
    if type(source_lineage) is not bytes or len(source_lineage) != 97:
        raise LineageError("source_lineage must be exact bytes97")
    ordinal = _exact_int(
        original_actor_ordinal, name="original_actor_ordinal", minimum=0, maximum=1
    )
    message = (
        b"phasepair-actor-lineage-v1\x00"
        + struct.pack(">I", 97)
        + source_lineage
        + bytes((ordinal,))
    )
    return hmac.new(key, message, hashlib.sha256).digest()


def pair_commitment(actor0_raw32: bytes, actor1_raw32: bytes) -> bytes:
    """Derive the actor-order invariant pair commitment."""

    actor0 = _raw32(actor0_raw32, name="actor0_raw32")
    actor1 = _raw32(actor1_raw32, name="actor1_raw32")
    if actor0 == actor1:
        raise LineageError("actor commitments must be distinct")
    low, high = sorted((actor0, actor1))
    return hashlib.sha256(b"phasepair-pair-v1\x00" + low + high).digest()


def validate_commitment_uniqueness(
    actor_commitments: Iterable[bytes], pair_commitments: Iterable[bytes]
) -> None:
    """Enforce unconditional global uniqueness for actor and pair commitments."""

    actors = [_raw32(value, name="actor commitment") for value in actor_commitments]
    pairs = [_raw32(value, name="pair commitment") for value in pair_commitments]
    if len(actors) != len(set(actors)):
        raise LineageError("PHASEPAIR_ACTOR_COMMITMENT_DUPLICATE_FATAL")
    if len(pairs) != len(set(pairs)):
        raise LineageError("LINEAGE_COMMITMENT_FAIL: duplicate pair commitment")


@dataclass(frozen=True)
class ActorPairRecord:
    """One private manifest record; actor order remains the original source order."""

    source_lineage: bytes
    actor0_raw32: bytes
    actor1_raw32: bytes
    pair_raw32: bytes

    def to_bytes(self) -> bytes:
        if type(self.source_lineage) is not bytes or len(self.source_lineage) != 97:
            raise LineageError("record source_lineage must be bytes97")
        actor0 = _raw32(self.actor0_raw32, name="actor0_raw32")
        actor1 = _raw32(self.actor1_raw32, name="actor1_raw32")
        pair = _raw32(self.pair_raw32, name="pair_raw32")
        if pair_commitment(actor0, actor1) != pair:
            raise LineageError("record pair commitment does not recompute")
        return self.source_lineage + actor0 + actor1 + pair


@dataclass(frozen=True)
class ActorPairManifest:
    """Private bytes plus the only publishable value: their SHA-256 digest."""

    private_bytes: bytes
    sha256: str
    row_count: int


def build_actor_pair_manifest(
    asset_source_rows: Sequence[Mapping[str, Any]],
    final_allowlist_raw32: bytes,
    private_lineage_key: bytes,
) -> ActorPairManifest:
    """Build a data-free private actor-pair manifest from canonical authority rows."""

    if type(asset_source_rows) not in (list, tuple):
        raise LineageError("asset_source_rows must be an exact list or tuple")
    allowlist = _raw32(final_allowlist_raw32, name="final_allowlist_raw32")
    key = _raw32(private_lineage_key, name="private_lineage_key")
    _, key_status_raw32 = lineage_key_status(key)
    records: list[ActorPairRecord] = []
    for row in asset_source_rows:
        source = source_lineage_bytes(row, allowlist)
        actor0 = actor_commitment(key, source, 0)
        actor1 = actor_commitment(key, source, 1)
        records.append(ActorPairRecord(source, actor0, actor1, pair_commitment(actor0, actor1)))
    validate_commitment_uniqueness(
        (value for record in records for value in (record.actor0_raw32, record.actor1_raw32)),
        (record.pair_raw32 for record in records),
    )
    ordered = sorted(records, key=lambda record: record.pair_raw32)
    payload = (
        b"phasepair-actor-pair-lineage-manifest-v1\x00"
        + allowlist
        + key_status_raw32
        + struct.pack(">I", len(ordered))
        + b"".join(record.to_bytes() for record in ordered)
    )
    return ActorPairManifest(payload, hashlib.sha256(payload).hexdigest(), len(ordered))


def crop_digest(pair_raw32: bytes, training_seed: int) -> bytes:
    """Return the train-crop digest used to select the shared pair crop."""

    pair = _raw32(pair_raw32, name="pair_raw32")
    seed = _exact_int(training_seed, name="training_seed", minimum=0, maximum=2**64 - 1)
    return hashlib.sha256(
        b"phasepair-crop-v1\x00" + pair + b"\x00" + struct.pack(">Q", seed)
    ).digest()


def crop_start(pair_raw32: bytes, training_seed: int, raw_length: int, split: str) -> int:
    """Compute the frozen train crop or deterministic validation/test center crop."""

    pair = _raw32(pair_raw32, name="pair_raw32")
    seed = _exact_int(training_seed, name="training_seed", minimum=0, maximum=2**64 - 1)
    length = _exact_int(raw_length, name="raw_length", minimum=0, maximum=None)
    if type(split) is not str or split not in _PARTITIONS:
        raise LineageError("split must be train, val, or test")
    if length <= 300:
        return 0
    if split == "train":
        value = int.from_bytes(crop_digest(pair, seed)[:8], "big")
        return value % (length - 300 + 1)
    return (length - 300) // 2
