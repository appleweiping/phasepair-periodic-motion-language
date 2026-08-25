"""Public Embody3D metadata ingestion for the PhaseSet migration.

Only the two dataset index JSON documents are read.  The loader deliberately
does not follow the motion, audio, or text paths contained in those indexes and
does not cache source bytes.  Repeated per-actor rows are collapsed to one
capture keyed by ``(subset, capture_name)``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping
from urllib.parse import urlparse
from urllib.request import Request, urlopen


SOURCE_FPS = 30
OFFICIAL_METADATA_URLS = MappingProxyType(
    {
        "acting": (
            "https://raw.githubusercontent.com/facebookresearch/embody-3d/"
            "main/datasets/acting/dataset.json"
        ),
        "daylife": (
            "https://raw.githubusercontent.com/facebookresearch/embody-3d/"
            "main/datasets/daylife/dataset.json"
        ),
    }
)
DEFAULT_MAX_METADATA_BYTES = 64 * 1024 * 1024


class PhaseSetDataError(ValueError):
    """The public metadata source or its capture schema is invalid."""


@dataclass(frozen=True, slots=True, init=False)
class CaptureRecord:
    """One de-duplicated public capture index row.

    Participant identifiers and capture names remain runtime-only inputs.  No
    serializer in this module writes them to a public manifest.
    """

    subset: str
    capture_name: str = field(repr=False)
    length_frames: int
    participants: tuple[str, ...] = field(repr=False)
    text_participants: tuple[str, ...] = field(repr=False)

    def __init__(
        self,
        subset: str,
        capture_name: str,
        length_frames: int,
        participants: tuple[str, ...],
        text_participants: tuple[str, ...],
    ) -> None:
        _nonempty_text(subset, "subset")
        _nonempty_text(capture_name, "capture_name")
        if type(length_frames) is not int or length_frames <= 0:
            raise PhaseSetDataError("length_frames must be a positive exact int")
        checked_participants = _canonical_names(participants, "participants")
        checked_text = _canonical_names(text_participants, "text_participants")
        if not set(checked_text).issubset(checked_participants):
            raise PhaseSetDataError("text participants must be capture participants")
        object.__setattr__(self, "subset", subset)
        object.__setattr__(self, "capture_name", capture_name)
        object.__setattr__(self, "length_frames", length_frames)
        object.__setattr__(self, "participants", checked_participants)
        object.__setattr__(self, "text_participants", checked_text)

    @property
    def key(self) -> tuple[str, str]:
        return self.subset, self.capture_name

    @property
    def participant_count(self) -> int:
        return len(self.participants)

    @property
    def has_complete_actor_text(self) -> bool:
        return self.text_participants == self.participants


@dataclass(frozen=True, slots=True)
class PublicMetadata:
    """Immutable, capture-de-duplicated view of public index metadata."""

    captures: tuple[CaptureRecord, ...]
    document_sha256: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        if type(self.captures) is not tuple:
            raise TypeError("captures must be an exact tuple")
        keys = tuple(capture.key for capture in self.captures)
        if len(set(keys)) != len(keys):
            raise PhaseSetDataError("captures must be unique by (subset, capture_name)")
        if keys != tuple(sorted(keys, key=_capture_key_sort)):
            raise PhaseSetDataError("captures must be in canonical UTF-8 byte order")
        if type(self.document_sha256) is not tuple:
            raise TypeError("document_sha256 must be an exact tuple")
        for subset, digest in self.document_sha256:
            _nonempty_text(subset, "document subset")
            _lower_sha256(digest, "document digest")
        digest_subsets = tuple(subset for subset, _ in self.document_sha256)
        if len(set(digest_subsets)) != len(digest_subsets):
            raise PhaseSetDataError("document digest subsets must be unique")
        if digest_subsets != tuple(sorted(digest_subsets, key=lambda item: item.encode("utf-8"))):
            raise PhaseSetDataError("document digests must use canonical subset order")


def _nonempty_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    if not value or "\x00" in value:
        raise PhaseSetDataError(f"{label} must be nonempty and contain no NUL")
    return value


def _lower_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise PhaseSetDataError(f"{label} must be lowercase SHA-256 hex")
    return value


def _canonical_names(value: object, label: str) -> tuple[str, ...]:
    if type(value) is not tuple:
        raise TypeError(f"{label} must be an exact tuple")
    checked = tuple(_nonempty_text(item, f"{label} item") for item in value)
    if len(set(checked)) != len(checked):
        raise PhaseSetDataError(f"{label} must not contain duplicates")
    canonical = tuple(sorted(checked, key=lambda item: item.encode("utf-8")))
    if checked != canonical:
        raise PhaseSetDataError(f"{label} must be in canonical UTF-8 byte order")
    return checked


def _capture_key_sort(key: tuple[str, str]) -> tuple[bytes, bytes]:
    return key[0].encode("utf-8"), key[1].encode("utf-8")


def _unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    output: dict[str, object] = {}
    for key, value in pairs:
        if key in output:
            raise PhaseSetDataError("metadata JSON contains a duplicate object key")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> object:
    raise PhaseSetDataError(f"metadata JSON contains forbidden constant {value}")


def decode_public_metadata_json(
    payload: bytes,
    *,
    max_bytes: int = DEFAULT_MAX_METADATA_BYTES,
) -> Mapping[str, object]:
    """Decode one index document while rejecting duplicate JSON keys."""

    if type(payload) is not bytes:
        raise TypeError("payload must be exact built-in bytes")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise TypeError("max_bytes must be a positive exact int")
    if not payload or len(payload) > max_bytes:
        raise PhaseSetDataError("metadata payload is empty or exceeds the byte limit")
    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_json_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseSetDataError("metadata must be strict UTF-8 JSON") from exc
    if not isinstance(decoded, Mapping):
        raise PhaseSetDataError("metadata root must be a JSON object")
    return decoded


def _parse_multiperson(value: object, label: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if type(value) is not list:
        raise PhaseSetDataError(f"{label}.multiperson must be a JSON array or null")
    checked = tuple(_nonempty_text(item, f"{label}.multiperson item") for item in value)
    if len(set(checked)) != len(checked):
        raise PhaseSetDataError(f"{label}.multiperson contains a duplicate")
    return tuple(sorted(checked, key=lambda item: item.encode("utf-8")))


def _parse_document(subset: str, document: Mapping[str, object]) -> tuple[CaptureRecord, ...]:
    _nonempty_text(subset, "subset")
    aggregates: dict[tuple[str, str], dict[str, object]] = {}
    for outer_index, (participant, capture_rows) in enumerate(document.items()):
        actor = _nonempty_text(participant, f"participant[{outer_index}]")
        if not isinstance(capture_rows, Mapping):
            raise PhaseSetDataError("each participant value must be a capture object")
        for row_index, (capture_name, raw_row) in enumerate(capture_rows.items()):
            name = _nonempty_text(capture_name, f"capture[{outer_index},{row_index}]")
            if not isinstance(raw_row, Mapping):
                raise PhaseSetDataError("each capture row must be a JSON object")
            length = raw_row.get("length")
            if type(length) is not int or length <= 0:
                raise PhaseSetDataError("capture length must be a positive exact int")
            others = _parse_multiperson(raw_row.get("multiperson"), "capture row")
            if actor in others:
                raise PhaseSetDataError("multiperson must exclude the outer participant")
            declared = tuple(sorted((actor, *others), key=lambda item: item.encode("utf-8")))
            raw_text = raw_row.get("text")
            has_language_text = False
            if type(raw_text) is str:
                _nonempty_text(raw_text, "capture text path")
                has_language_text = True
            elif raw_text is not None:
                if type(raw_text) is not int or raw_text < 0:
                    raise PhaseSetDataError(
                        "capture text must be null, a nonempty string, or a nonnegative class int"
                    )

            key = subset, name
            aggregate = aggregates.get(key)
            if aggregate is None:
                aggregate = {
                    "declared": declared,
                    "length": length,
                    "rows": set(),
                    "text": set(),
                }
                aggregates[key] = aggregate
            if aggregate["declared"] != declared or aggregate["length"] != length:
                raise PhaseSetDataError("per-actor rows disagree on capture membership or length")
            rows = aggregate["rows"]
            text_rows = aggregate["text"]
            if not isinstance(rows, set) or not isinstance(text_rows, set):
                raise AssertionError("internal capture aggregate is malformed")
            if actor in rows:
                raise PhaseSetDataError("duplicate per-actor capture row")
            rows.add(actor)
            if has_language_text:
                text_rows.add(actor)

    captures: list[CaptureRecord] = []
    for key in sorted(aggregates, key=_capture_key_sort):
        aggregate = aggregates[key]
        declared = aggregate["declared"]
        rows = aggregate["rows"]
        text_rows = aggregate["text"]
        if (
            type(declared) is not tuple
            or not isinstance(rows, set)
            or not isinstance(text_rows, set)
        ):
            raise AssertionError("internal capture aggregate is malformed")
        if rows != set(declared):
            raise PhaseSetDataError("capture is missing one or more declared per-actor rows")
        captures.append(
            CaptureRecord(
                subset=key[0],
                capture_name=key[1],
                length_frames=int(aggregate["length"]),
                participants=declared,
                text_participants=tuple(sorted(text_rows, key=lambda item: item.encode("utf-8"))),
            )
        )
    return tuple(captures)


def ingest_public_metadata(
    documents: Mapping[str, Mapping[str, object]],
    *,
    document_sha256: Mapping[str, str] | None = None,
) -> PublicMetadata:
    """Ingest decoded public indexes and collapse their repeated actor rows."""

    if not isinstance(documents, Mapping) or not documents:
        raise TypeError("documents must be a nonempty mapping")
    captures: list[CaptureRecord] = []
    subset_names = tuple(_nonempty_text(subset, "subset") for subset in documents)
    for subset in sorted(subset_names, key=lambda item: item.encode("utf-8")):
        document = documents[subset]
        if not isinstance(document, Mapping):
            raise PhaseSetDataError("each metadata document must be a mapping")
        captures.extend(_parse_document(subset, document))
    captures.sort(key=lambda capture: _capture_key_sort(capture.key))
    keys = tuple(capture.key for capture in captures)
    if len(keys) != len(set(keys)):
        raise PhaseSetDataError("capture identity collides across input documents")

    digests: tuple[tuple[str, str], ...] = ()
    if document_sha256 is not None:
        if set(document_sha256) != set(documents):
            raise PhaseSetDataError("document digests must cover exactly the input subsets")
        digests = tuple(
            (subset, _lower_sha256(document_sha256[subset], "document digest"))
            for subset in sorted(document_sha256, key=lambda item: item.encode("utf-8"))
        )
    return PublicMetadata(tuple(captures), digests)


def eligible_group_text_captures(
    metadata: PublicMetadata,
    *,
    minimum_participants: int = 3,
    require_complete_actor_text: bool = True,
) -> tuple[CaptureRecord, ...]:
    """Select group captures without reading any referenced text or motion file."""

    if type(metadata) is not PublicMetadata:
        raise TypeError("metadata must be exactly PublicMetadata")
    if type(minimum_participants) is not int or minimum_participants < 2:
        raise PhaseSetDataError("minimum_participants must be an exact int >= 2")
    if type(require_complete_actor_text) is not bool:
        raise TypeError("require_complete_actor_text must be an exact bool")
    return tuple(
        capture
        for capture in metadata.captures
        if capture.participant_count >= minimum_participants
        and (capture.has_complete_actor_text or not require_complete_actor_text)
    )


def _fetch_metadata_bytes(url: str, *, timeout_seconds: float, max_bytes: int) -> bytes:
    if url not in OFFICIAL_METADATA_URLS.values():
        raise PhaseSetDataError("only the pinned official metadata URLs are allowed")
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "raw.githubusercontent.com":
        raise PhaseSetDataError("official metadata source must use pinned HTTPS hosting")
    request = Request(url, headers={"User-Agent": "PhaseSet-public-metadata-audit/1"})
    with urlopen(request, timeout=timeout_seconds) as response:  # noqa: S310 - exact allowlist
        final_url = response.geturl()
        final = urlparse(final_url)
        if (
            final_url != url
            or final.scheme != "https"
            or final.hostname != "raw.githubusercontent.com"
        ):
            raise PhaseSetDataError("metadata response left the exact official URL")
        raw = response.read(max_bytes + 1)
    if not raw or len(raw) > max_bytes:
        raise PhaseSetDataError("metadata response is empty or exceeds the byte limit")
    return raw


def load_official_public_metadata(
    *,
    timeout_seconds: float = 30.0,
    max_bytes: int = DEFAULT_MAX_METADATA_BYTES,
) -> PublicMetadata:
    """Fetch the two official public indexes in memory, without asset downloads."""

    if isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float)):
        raise TypeError("timeout_seconds must be numeric")
    if not 0.0 < float(timeout_seconds) <= 120.0:
        raise PhaseSetDataError("timeout_seconds must be in (0, 120]")
    if type(max_bytes) is not int or max_bytes <= 0:
        raise TypeError("max_bytes must be a positive exact int")
    documents: dict[str, Mapping[str, object]] = {}
    digests: dict[str, str] = {}
    for subset, url in OFFICIAL_METADATA_URLS.items():
        raw = _fetch_metadata_bytes(
            url,
            timeout_seconds=float(timeout_seconds),
            max_bytes=max_bytes,
        )
        documents[subset] = decode_public_metadata_json(raw, max_bytes=max_bytes)
        digests[subset] = hashlib.sha256(raw).hexdigest()
    return ingest_public_metadata(documents, document_sha256=digests)


def load_public_metadata_files(paths: Mapping[str, str | Path]) -> PublicMetadata:
    """Read caller-supplied local copies of the public indexes."""

    if not isinstance(paths, Mapping) or not paths:
        raise TypeError("paths must be a nonempty mapping")
    documents: dict[str, Mapping[str, object]] = {}
    digests: dict[str, str] = {}
    for subset, raw_path in paths.items():
        _nonempty_text(subset, "subset")
        path = Path(raw_path)
        raw = path.read_bytes()
        documents[subset] = decode_public_metadata_json(raw)
        digests[subset] = hashlib.sha256(raw).hexdigest()
    return ingest_public_metadata(documents, document_sha256=digests)


__all__ = [
    "CaptureRecord",
    "DEFAULT_MAX_METADATA_BYTES",
    "OFFICIAL_METADATA_URLS",
    "PhaseSetDataError",
    "PublicMetadata",
    "SOURCE_FPS",
    "decode_public_metadata_json",
    "eligible_group_text_captures",
    "ingest_public_metadata",
    "load_official_public_metadata",
    "load_public_metadata_files",
]
