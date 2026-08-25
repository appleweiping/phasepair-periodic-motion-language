"""Fail closed on unsafe tracked paths, local locators, and common secrets."""

from __future__ import annotations

import base64
import binascii
import io
import re
import subprocess
import sys
import urllib.parse
import zipfile
import zlib
from pathlib import Path, PurePosixPath
import xml.etree.ElementTree as ET

sys.dont_write_bytecode = True

from release_tree import ReleaseTreeError, resolve_release_tree  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PATH_PARTS = frozenset(
    {
        ".aris",
        ".agents",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        "checkpoints",
        "credentials",
        "data",
        "datasets",
        "downloads",
        "mlruns",
        "participant-records",
        "participants",
        "private",
        "private-data",
        "private_data",
        "recommend paper",
        "receipts-private",
        "refine-logs",
        "restricted",
        "results",
        "secrets",
        "server-inventory",
        "wandb",
        "wheelhouse",
    }
)
FORBIDDEN_MARKERS = (
    b"recommend" + b" " + b"paper" + b"/",
    b"refine" + b"-" + b"logs" + b"/",
)
LOCAL_LOCATOR_PATTERNS = (
    re.compile(rb"(?<![A-Za-z0-9])[A-Za-z]:[\\/](?![\\/])"),
    re.compile(rb"(?:^|[\s\"'(=])/(?:home|mnt|private|root|tmp|Users|var/tmp)/", re.I),
    re.compile(rb"(?:file|sftp|ssh)://", re.I),
    re.compile(rb"(?:^|[\s\"'(=])\\\\[A-Za-z0-9][A-Za-z0-9._-]{0,252}\\[^\\\s]+"),
    re.compile(
        rb"(?:\$\{?HOME\}?|"
        + b"%"
        + b"USERPROFILE"
        + b"%"
        + rb"|"
        + b"~"
        + b"/"
        + rb")",
        re.I,
    ),
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bglpat-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bhf_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\b"),
    re.compile(rb"\bAuthorization\s*:\s*(?:Basic|Bearer)\s+[A-Za-z0-9._~+/=-]{8,}", re.I),
    re.compile(
        rb"\b(?:api[_-]?key|access[_-]?token|auth[_-]?token|password|passwd|"
        rb"client[_-]?secret|private[_-]?key)\b\s*[:=]\s*[\"'][^\"'\r\n]{8,}[\"']",
        re.I,
    ),
    re.compile(rb"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]{2,5}\b"),
    re.compile(
        rb"\b(?:" + b"local" + b"host" + rb"|[A-Za-z0-9.-]+\.(?:internal|local|lan))"
        rb"(?::[0-9]{2,5})?\b",
        re.I,
    ),
    re.compile(rb"\b(?:10(?:\.[0-9]{1,3}){3}|127(?:\.[0-9]{1,3}){3}|"
               rb"169\.254(?:\.[0-9]{1,3}){2}|192\.168(?:\.[0-9]{1,3}){2}|"
               rb"172\.(?:1[6-9]|2[0-9]|3[01])(?:\.[0-9]{1,3}){2})\b"),
    re.compile(rb"(?:[?&](?:X-Amz-Signature|Signature|sig|token|access_token)=)[^&\s\"']+", re.I),
)
SENSITIVE_RECORD_PATTERNS = (
    re.compile(rb'"contains_(?:participant|subject|capture)_ids"\s*:\s*true', re.I),
    re.compile(rb'"(?:participant|subject)_(?:ids|names|emails)"\s*:\s*\[[^\]]*[^\s\]]', re.I),
    re.compile(rb'"(?:rights|license|release_form|download)_grant_sha256"\s*:\s*"[0-9a-f]{64}"', re.I),
    re.compile(rb'"(?:download_url|signed_url|dataset_url)"\s*:\s*"(?:https?|s3)://', re.I),
)
MAX_EMBEDDED_BYTES = 64 * 1024 * 1024
PNG_ALLOWED_CHUNKS = frozenset(
    {
        b"IHDR",
        b"PLTE",
        b"IDAT",
        b"IEND",
        b"cHRM",
        b"gAMA",
        b"iCCP",
        b"sBIT",
        b"sRGB",
        b"cICP",
        b"mDCV",
        b"cLLI",
        b"tRNS",
        b"bKGD",
        b"hIST",
        b"pHYs",
        b"tIME",
        b"tEXt",
        b"zTXt",
        b"iTXt",
    }
)
PDF_FORBIDDEN_FEATURES = re.compile(
    rb"/(?:EmbeddedFile|EmbeddedFiles|Filespec|AA|JavaScript|JS|Launch|"
    rb"RichMedia|SubmitForm|ImportData|GoToR|XFA|AcroForm|Collection|AFRelationship)\b"
)
PDF_STREAM = re.compile(
    rb"(?ms)^[0-9]+\s+[0-9]+\s+obj\s*"
    rb"(?P<dictionary><<(?:(?!\bendobj\b).)*?>>)\s*stream(?:\r\n|\n|\r)"
    rb"(?P<payload>.*?)"
    rb"(?:\r\n|\n|\r)endstream"
)
PDF_FILTERS = re.compile(rb"/Filter\s*(?:\[(?P<array>.*?)\]|/(?P<single>[A-Za-z0-9]+))", re.S)
PDF_IMAGE_FILTERS = frozenset({b"DCTDecode", b"JPXDecode", b"CCITTFaxDecode", b"JBIG2Decode"})
PDF_NAME_DELIMITERS = frozenset(b"\0\t\n\f\r ()<>[]{}/%")


class PublicReleaseAuditError(RuntimeError):
    """A tracked release item violates the public-only boundary."""


def _validated_paths(
    resolved: tuple[PurePosixPath, ...],
) -> tuple[PurePosixPath, ...]:
    paths: list[PurePosixPath] = []
    for path in resolved:
        decoded = path.as_posix()
        if any(part.casefold() in FORBIDDEN_PATH_PARTS for part in path.parts):
            raise PublicReleaseAuditError(f"forbidden tracked path: {decoded!r}")
        paths.append(path)
    return tuple(sorted(paths, key=lambda value: value.as_posix().encode("utf-8")))


def _scan_bytes(raw: bytes, label: str) -> None:
    if label != ".gitignore":
        for marker in FORBIDDEN_MARKERS:
            if marker.lower() in raw.lower():
                raise PublicReleaseAuditError(f"private/local marker in {label}")
        for pattern in LOCAL_LOCATOR_PATTERNS:
            if pattern.search(raw):
                raise PublicReleaseAuditError(f"private/local locator in {label}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(raw):
            raise PublicReleaseAuditError(f"credential or endpoint pattern in {label}")
    for pattern in SENSITIVE_RECORD_PATTERNS:
        if pattern.search(raw):
            raise PublicReleaseAuditError(f"participant/rights material in {label}")


def _scan_pptx(raw: bytes, label: str) -> None:
    required = {"[Content_Types].xml", "_rels/.rels", "ppt/presentation.xml"}
    observed: set[str] = set()
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                decoded_path = member_path.as_posix()
                if (
                    member_path.is_absolute()
                    or ".." in member_path.parts
                    or "\\" in member.filename
                    or decoded_path != member.filename.rstrip("/")
                ):
                    raise PublicReleaseAuditError(
                        f"unsafe archive member in {label}: {member.filename!r}"
                    )
                if member.flag_bits & 0x1:
                    raise PublicReleaseAuditError(f"encrypted PPTX member: {label}")
                if not member.is_dir() and decoded_path in observed:
                    raise PublicReleaseAuditError(f"duplicate PPTX member: {decoded_path}")
                observed.add(decoded_path)
                if member.file_size > MAX_EMBEDDED_BYTES:
                    raise PublicReleaseAuditError(f"oversized PPTX member: {label}")
                total += member.file_size
                if total > 256 * 1024 * 1024:
                    raise PublicReleaseAuditError(f"oversized PPTX payload: {label}")
                if member_path.parts[:2] == ("ppt", "embeddings"):
                    raise PublicReleaseAuditError(f"embedded PPTX attachment: {label}")
                if member_path.parts[:2] == ("ppt", "fonts"):
                    raise PublicReleaseAuditError(f"embedded PPTX font: {label}")
                if not member.is_dir():
                    payload = archive.read(member)
                    if len(payload) != member.file_size:
                        raise PublicReleaseAuditError(f"truncated PPTX member: {label}")
                    member_label = f"{label}!/{member.filename}"
                    suffix = member_path.suffix.casefold()
                    is_relationship = member_path.name.casefold().endswith(".rels")
                    is_media = member_path.parts[:2] == ("ppt", "media")
                    is_thumbnail = (
                        len(member_path.parts) == 2
                        and member_path.parts[0] == "docProps"
                        and member_path.stem == "thumbnail"
                    )
                    if is_media or is_thumbnail:
                        if suffix == ".png":
                            _scan_png(payload, member_label)
                        elif suffix in {".jpg", ".jpeg"}:
                            _scan_jpeg(payload, member_label)
                        elif suffix == ".svg":
                            _scan_svg(payload, member_label)
                        else:
                            raise PublicReleaseAuditError(
                                f"unsupported PPTX media payload: {member.filename!r}"
                            )
                    elif (suffix == ".xml" or is_relationship) and (
                        decoded_path == "[Content_Types].xml"
                        or member_path.parts[0] in {"_rels", "docProps", "ppt", "customXml"}
                    ):
                        _scan_xml(
                            payload,
                            member_label,
                            reject_external_relationships=is_relationship,
                        )
                    elif re.fullmatch(
                        r"ppt/printerSettings/printerSettings[1-9][0-9]*\.bin", decoded_path
                    ) and len(payload) <= 64 * 1024:
                        printable = b"\n".join(re.findall(rb"[\x20-\x7e]{8,}", payload))
                        printable += b"\n" + b"\n".join(
                            match.replace(b"\0", b"")
                            for match in re.findall(rb"(?:[\x20-\x7e]\x00){4,}", payload)
                        )
                        printable += b"\n" + b"\n".join(
                            match.replace(b"\0", b"")
                            for match in re.findall(rb"(?:\x00[\x20-\x7e]){4,}", payload)
                        )
                        if printable.strip():
                            _scan_bytes(printable, f"{member_label}!/printable-metadata")
                    else:
                        raise PublicReleaseAuditError(
                            f"unsupported PPTX member type/path: {member.filename!r}"
                        )
    except zipfile.BadZipFile as exc:
        raise PublicReleaseAuditError(f"invalid PPTX archive: {label}") from exc
    missing = required - observed
    if missing:
        raise PublicReleaseAuditError(f"PPTX lacks required members {sorted(missing)!r}: {label}")


def _scan_png(raw: bytes, label: str) -> None:
    """Validate PNG framing/CRC and inspect embedded textual diagram payloads."""

    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PublicReleaseAuditError(f"invalid PNG signature: {label}")
    offset = 8
    saw_iend = False
    saw_ihdr = False
    saw_idat = False
    chunk_count = 0
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise PublicReleaseAuditError(f"truncated PNG chunk: {label}")
        length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        chunk_count += 1
        if re.fullmatch(rb"[A-Za-z]{4}", chunk_type) is None:
            raise PublicReleaseAuditError(f"invalid PNG chunk type: {label}")
        if chunk_type not in PNG_ALLOWED_CHUNKS:
            raise PublicReleaseAuditError(
                f"unsupported PNG ancillary/critical chunk {chunk_type!r}: {label}"
            )
        end = offset + 12 + length
        if length > 64 * 1024 * 1024 or end > len(raw):
            raise PublicReleaseAuditError(f"invalid PNG chunk length: {label}")
        payload = raw[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(raw[offset + 8 + length : end], "big")
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise PublicReleaseAuditError(f"PNG CRC mismatch: {label}")
        if chunk_type == b"IHDR":
            if saw_ihdr or chunk_count != 1 or length != 13:
                raise PublicReleaseAuditError(f"invalid PNG IHDR topology: {label}")
            saw_ihdr = True
        elif not saw_ihdr:
            raise PublicReleaseAuditError(f"PNG chunk precedes IHDR: {label}")
        if chunk_type == b"IDAT":
            saw_idat = True
        elif chunk_type == b"tEXt":
            _scan_bytes(payload, f"{label}!/tEXt")
        elif chunk_type == b"zTXt":
            try:
                _, compressed = payload.split(b"\x00\x00", 1)
                decoded = zlib.decompress(compressed)
            except (ValueError, zlib.error) as exc:
                raise PublicReleaseAuditError(f"invalid compressed PNG text: {label}") from exc
            if len(decoded) > 32 * 1024 * 1024:
                raise PublicReleaseAuditError(f"oversized PNG text payload: {label}")
            _scan_bytes(decoded, f"{label}!/zTXt")
        elif chunk_type == b"iTXt":
            # keyword NUL, compression flag/method, language NUL, translated
            # keyword NUL, then UTF-8 text (optionally zlib-compressed).
            try:
                _, tail = payload.split(b"\x00", 1)
                compression_flag, compression_method = tail[0], tail[1]
                tail = tail[2:]
                _, tail = tail.split(b"\x00", 1)
                _, text_payload = tail.split(b"\x00", 1)
                if compression_flag == 1 and compression_method == 0:
                    text_payload = zlib.decompress(text_payload)
                elif compression_flag != 0:
                    raise ValueError("unsupported iTXt compression")
            except (IndexError, ValueError, zlib.error) as exc:
                raise PublicReleaseAuditError(f"invalid PNG international text: {label}") from exc
            if len(text_payload) > 32 * 1024 * 1024:
                raise PublicReleaseAuditError(f"oversized PNG text payload: {label}")
            _scan_bytes(text_payload, f"{label}!/iTXt")
        elif chunk_type == b"iCCP":
            try:
                profile_name, tail = payload.split(b"\0", 1)
                if not profile_name or tail[:1] != b"\0":
                    raise ValueError("invalid iCCP framing")
                decoded_profile = _bounded_flate(tail[1:], f"{label}!/iCCP")
            except ValueError as exc:
                raise PublicReleaseAuditError(f"invalid PNG iCCP profile: {label}") from exc
            _scan_bytes(profile_name, f"{label}!/iCCP-name")
            printable = b"\n".join(re.findall(rb"[\x20-\x7e]{8,}", decoded_profile))
            if printable:
                _scan_bytes(printable, f"{label}!/iCCP-profile")
        if chunk_type == b"IEND":
            saw_iend = True
            if end != len(raw):
                raise PublicReleaseAuditError(f"trailing bytes after PNG IEND: {label}")
            break
        offset = end
    if not saw_iend or not saw_idat:
        raise PublicReleaseAuditError(f"PNG lacks IDAT or IEND: {label}")


def _scan_jpeg(raw: bytes, label: str) -> None:
    """Validate JPEG framing and scan comment/application metadata, not entropy bytes."""

    if not raw.startswith(b"\xff\xd8") or not raw.endswith(b"\xff\xd9"):
        raise PublicReleaseAuditError(f"invalid JPEG framing: {label}")
    cursor = 2
    while cursor < len(raw) - 2:
        if raw[cursor] != 0xFF:
            raise PublicReleaseAuditError(f"invalid JPEG marker framing: {label}")
        while cursor < len(raw) and raw[cursor] == 0xFF:
            cursor += 1
        if cursor >= len(raw):
            raise PublicReleaseAuditError(f"truncated JPEG marker: {label}")
        marker = raw[cursor]
        cursor += 1
        if marker == 0xD9:
            return
        if marker in {0x01, *range(0xD0, 0xD8)}:
            continue
        if cursor + 2 > len(raw):
            raise PublicReleaseAuditError(f"truncated JPEG segment: {label}")
        size = int.from_bytes(raw[cursor : cursor + 2], "big")
        if size < 2 or cursor + size > len(raw):
            raise PublicReleaseAuditError(f"invalid JPEG segment size: {label}")
        payload = raw[cursor + 2 : cursor + size]
        cursor += size
        if marker == 0xDA:
            if b"\xff\xd9" not in raw[cursor:]:
                raise PublicReleaseAuditError(f"JPEG scan lacks EOI: {label}")
            return
        if marker == 0xFE or 0xE0 <= marker <= 0xEF:
            printable = b"\n".join(re.findall(rb"[\x20-\x7e]{8,}", payload))
            if printable:
                _scan_bytes(printable, f"{label}!/metadata-{marker:02x}")


def _xml_local_name(value: str) -> str:
    return value.rsplit("}", 1)[-1].rsplit(":", 1)[-1]


def _scan_data_uri(value: str, label: str, depth: int) -> None:
    if depth > 4 or not value.casefold().startswith("data:"):
        raise PublicReleaseAuditError(f"invalid or over-nested data URI: {label}")
    header, separator, encoded = value[5:].partition(",")
    if not separator:
        raise PublicReleaseAuditError(f"malformed data URI: {label}")
    fields = header.split(";")
    media_type = (fields[0] or "text/plain").casefold()
    try:
        if any(field.casefold() == "base64" for field in fields[1:]):
            compact = re.sub(r"\s+", "", encoded)
            payload = base64.b64decode(compact.encode("ascii", "strict"), validate=True)
        else:
            payload = urllib.parse.unquote_to_bytes(encoded)
    except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
        raise PublicReleaseAuditError(f"malformed data URI payload: {label}") from exc
    if len(payload) > MAX_EMBEDDED_BYTES:
        raise PublicReleaseAuditError(f"oversized data URI payload: {label}")
    if media_type == "image/png":
        _scan_png(payload, f"{label}!/data-png")
    elif media_type in {"image/jpeg", "image/jpg"}:
        _scan_jpeg(payload, f"{label}!/data-jpeg")
    elif media_type in {"image/svg+xml", "application/xml", "text/xml"}:
        _scan_xml(payload, f"{label}!/data-xml", depth=depth + 1)
    elif media_type.startswith("text/"):
        _scan_bytes(payload, f"{label}!/data-text")
    else:
        raise PublicReleaseAuditError(f"unsupported embedded data URI type {media_type!r}: {label}")


def _scan_xml(
    raw: bytes,
    label: str,
    *,
    depth: int = 0,
    reject_external_relationships: bool = False,
) -> ET.Element:
    if depth > 4 or len(raw) > MAX_EMBEDDED_BYTES:
        raise PublicReleaseAuditError(f"oversized or over-nested XML payload: {label}")
    svg_doctype = re.compile(
        rb'<!DOCTYPE\s+svg\s+PUBLIC\s+"-//W3C//DTD SVG 1\.1//EN"\s+'
        rb'"http://www\.w3\.org/Graphics/SVG/1\.1/DTD/svg11\.dtd"\s*>',
        re.I,
    )
    sanitized = svg_doctype.sub(b"", raw)
    lowered = sanitized.lower()
    if b"<!doctype" in lowered or b"<!entity" in lowered:
        raise PublicReleaseAuditError(f"DTD/entity is forbidden in public XML: {label}")
    _scan_bytes(raw, label)
    try:
        root = ET.fromstring(sanitized)
    except ET.ParseError as exc:
        raise PublicReleaseAuditError(f"invalid XML payload: {label}") from exc
    for element_index, element in enumerate(root.iter()):
        values: list[tuple[str, str]] = []
        if element.text:
            values.append(("text", element.text))
        if element.tail:
            values.append(("tail", element.tail))
        values.extend((name, value) for name, value in element.attrib.items())
        attributes = {_xml_local_name(key).casefold(): value for key, value in element.attrib.items()}
        if reject_external_relationships and attributes.get("targetmode", "").casefold() == "external":
            raise PublicReleaseAuditError(f"external OOXML relationship is forbidden: {label}")
        for value_index, (name, value) in enumerate(values):
            value_label = f"{label}!/xml-{element_index}-{value_index}-{_xml_local_name(name)}"
            encoded_value = value.encode("utf-8", "strict")
            _scan_bytes(encoded_value, value_label)
            stripped = value.strip()
            if stripped.casefold().startswith("data:"):
                _scan_data_uri(stripped, value_label, depth)
                continue
            for css_uri in re.findall(r"url\(\s*['\"]?([^)'\"]+)", value, re.I):
                if css_uri.casefold().startswith("data:"):
                    _scan_data_uri(css_uri, value_label, depth)
                else:
                    parsed_css = urllib.parse.urlsplit(css_uri)
                    if parsed_css.scheme or parsed_css.netloc:
                        raise PublicReleaseAuditError(f"external XML style URI is forbidden: {label}")
            if _xml_local_name(name).casefold() in {
                "href",
                "src",
                "target",
                "url",
                "resource",
            }:
                parsed = urllib.parse.urlsplit(stripped)
                if parsed.scheme.casefold() == "https":
                    if _xml_local_name(element.tag).casefold() in {"image", "use"}:
                        raise PublicReleaseAuditError(
                            f"external XML image/resource is forbidden: {label}"
                        )
                elif parsed.scheme or parsed.netloc:
                    raise PublicReleaseAuditError(f"unsupported XML URI scheme: {label}")
    return root


def _scan_svg(raw: bytes, label: str, *, depth: int = 0) -> None:
    root = _scan_xml(raw, label, depth=depth)
    if _xml_local_name(root.tag).casefold() != "svg":
        raise PublicReleaseAuditError(f"SVG asset lacks an svg root: {label}")
    for element in root.iter():
        if _xml_local_name(element.tag).casefold() in {"script", "iframe", "object", "embed"}:
            raise PublicReleaseAuditError(f"active SVG element is forbidden: {label}")


def _bounded_raw_deflate(raw: bytes, label: str) -> bytes:
    decoder = zlib.decompressobj(-15)
    try:
        decoded = decoder.decompress(raw, MAX_EMBEDDED_BYTES + 1)
        if len(decoded) > MAX_EMBEDDED_BYTES:
            raise PublicReleaseAuditError(f"oversized draw.io payload: {label}")
        decoded += decoder.flush(MAX_EMBEDDED_BYTES + 1 - len(decoded))
    except (ValueError, zlib.error) as exc:
        raise PublicReleaseAuditError(f"invalid compressed draw.io payload: {label}") from exc
    if len(decoded) > MAX_EMBEDDED_BYTES or not decoder.eof or decoder.unused_data:
        raise PublicReleaseAuditError(f"invalid or oversized draw.io payload: {label}")
    return decoded


def _scan_drawio(raw: bytes, label: str) -> None:
    root = _scan_xml(raw, label)
    if _xml_local_name(root.tag).casefold() != "mxfile":
        raise PublicReleaseAuditError(f"draw.io file lacks an mxfile root: {label}")
    compressed_default = root.attrib.get("compressed", "true").casefold() != "false"
    for index, diagram in enumerate(
        element for element in root.iter() if _xml_local_name(element.tag).casefold() == "diagram"
    ):
        if list(diagram) or not (diagram.text or "").strip():
            continue
        encoded = (diagram.text or "").strip()
        diagram_label = f"{label}!/diagram-{index}"
        if not compressed_default or encoded.startswith("<"):
            decoded = encoded.encode("utf-8")
        else:
            try:
                compressed = base64.b64decode(
                    re.sub(r"\s+", "", encoded).encode("ascii", "strict"), validate=True
                )
            except (UnicodeEncodeError, ValueError, binascii.Error) as exc:
                raise PublicReleaseAuditError(f"invalid draw.io base64 payload: {label}") from exc
            decoded = urllib.parse.unquote_to_bytes(
                _bounded_raw_deflate(compressed, diagram_label).decode("utf-8", "strict")
            )
        _scan_bytes(decoded, diagram_label)
        nested = _scan_xml(decoded, diagram_label, depth=1)
        if _xml_local_name(nested.tag).casefold() != "mxgraphmodel":
            raise PublicReleaseAuditError(f"draw.io diagram payload has wrong root: {label}")


def _bounded_flate(raw: bytes, label: str) -> bytes:
    decoder = zlib.decompressobj()
    try:
        decoded = decoder.decompress(raw, MAX_EMBEDDED_BYTES + 1)
        if len(decoded) > MAX_EMBEDDED_BYTES:
            raise PublicReleaseAuditError(f"oversized Flate PDF stream: {label}")
        decoded += decoder.flush(MAX_EMBEDDED_BYTES + 1 - len(decoded))
    except (ValueError, zlib.error) as exc:
        raise PublicReleaseAuditError(f"invalid Flate PDF stream: {label}") from exc
    if len(decoded) > MAX_EMBEDDED_BYTES or not decoder.eof or decoder.unused_data:
        raise PublicReleaseAuditError(f"invalid or oversized Flate PDF stream: {label}")
    return decoded


def _ascii_hex_decode(raw: bytes, label: str) -> bytes:
    compact = re.sub(rb"\s+", b"", raw)
    if compact.endswith(b">"):
        compact = compact[:-1]
    if len(compact) % 2:
        compact += b"0"
    try:
        decoded = binascii.unhexlify(compact)
    except (binascii.Error, ValueError) as exc:
        raise PublicReleaseAuditError(f"invalid ASCIIHex PDF stream: {label}") from exc
    if len(decoded) > MAX_EMBEDDED_BYTES:
        raise PublicReleaseAuditError(f"oversized ASCIIHex PDF stream: {label}")
    return decoded


def _ascii85_decode(raw: bytes, label: str) -> bytes:
    compact = raw.strip()
    if compact.endswith(b"~>"):
        compact = compact[:-2]
    try:
        decoded = base64.a85decode(compact, adobe=False, ignorechars=b" \t\r\n\v")
    except (ValueError, binascii.Error) as exc:
        raise PublicReleaseAuditError(f"invalid ASCII85 PDF stream: {label}") from exc
    if len(decoded) > MAX_EMBEDDED_BYTES:
        raise PublicReleaseAuditError(f"oversized ASCII85 PDF stream: {label}")
    return decoded


def _run_length_decode(raw: bytes, label: str) -> bytes:
    decoded = bytearray()
    cursor = 0
    terminated = False
    while cursor < len(raw):
        control = raw[cursor]
        cursor += 1
        if control == 128:
            terminated = True
            break
        if control <= 127:
            count = control + 1
            if cursor + count > len(raw):
                raise PublicReleaseAuditError(f"invalid RunLength PDF stream: {label}")
            decoded.extend(raw[cursor : cursor + count])
            cursor += count
        else:
            if cursor >= len(raw):
                raise PublicReleaseAuditError(f"invalid RunLength PDF stream: {label}")
            decoded.extend(raw[cursor : cursor + 1] * (257 - control))
            cursor += 1
        if len(decoded) > MAX_EMBEDDED_BYTES:
            raise PublicReleaseAuditError(f"oversized RunLength PDF stream: {label}")
    if not terminated or raw[cursor:].strip():
        raise PublicReleaseAuditError(f"invalid RunLength PDF stream: {label}")
    return bytes(decoded)


def _pdf_filters(dictionary: bytes, label: str) -> tuple[bytes, ...]:
    match = PDF_FILTERS.search(dictionary)
    if match is None:
        return ()
    if match.group("single") is not None:
        return (match.group("single"),)
    names = tuple(re.findall(rb"/([A-Za-z0-9]+)", match.group("array") or b""))
    if not names:
        raise PublicReleaseAuditError(f"malformed PDF filter array: {label}")
    return names


def _normalize_pdf_names(raw: bytes, label: str, *, strict_syntax: bool = True) -> bytes:
    """Decode every syntactic PDF Name ``#xx`` escape; reject malformed names."""

    normalized = bytearray()
    cursor = 0
    while cursor < len(raw):
        byte = raw[cursor]
        if byte == 0x25:  # PDF comment; names inside it are not syntax.
            end = cursor
            while end < len(raw) and raw[end] not in b"\r\n":
                end += 1
            normalized.extend(raw[cursor:end])
            cursor = end
            continue
        if byte == 0x28:  # Literal string, including escaped/nested parentheses.
            depth = 1
            end = cursor + 1
            escaped = False
            while end < len(raw) and depth:
                value = raw[end]
                if escaped:
                    escaped = False
                elif value == 0x5C:
                    escaped = True
                elif value == 0x28:
                    depth += 1
                elif value == 0x29:
                    depth -= 1
                end += 1
            if depth:
                if strict_syntax:
                    raise PublicReleaseAuditError(f"unterminated PDF literal string: {label}")
                normalized.extend(raw[cursor:])
                break
            normalized.extend(raw[cursor:end])
            cursor = end
            continue
        if raw.startswith(b"<<", cursor):
            normalized.extend(b"<<")
            cursor += 2
            continue
        if byte == 0x3C:  # Hex string.
            end = raw.find(b">", cursor + 1)
            if end < 0:
                if strict_syntax:
                    raise PublicReleaseAuditError(f"unterminated PDF hex string: {label}")
                normalized.extend(raw[cursor:])
                break
            normalized.extend(raw[cursor : end + 1])
            cursor = end + 1
            continue
        if byte != 0x2F:
            normalized.append(byte)
            cursor += 1
            continue
        end = cursor + 1
        while end < len(raw) and raw[end] not in PDF_NAME_DELIMITERS:
            end += 1
        if end == cursor + 1:
            if strict_syntax:
                raise PublicReleaseAuditError(f"empty PDF name token: {label}")
            normalized.append(byte)
            cursor += 1
            continue
        name = raw[cursor + 1 : end]
        decoded = bytearray()
        name_cursor = 0
        while name_cursor < len(name):
            if name[name_cursor] != 0x23:
                decoded.append(name[name_cursor])
                name_cursor += 1
                continue
            if name_cursor + 2 >= len(name) or re.fullmatch(
                rb"[0-9A-Fa-f]{2}", name[name_cursor + 1 : name_cursor + 3]
            ) is None:
                raise PublicReleaseAuditError(f"malformed PDF name escape: {label}")
            value = int(name[name_cursor + 1 : name_cursor + 3], 16)
            if value == 0:
                raise PublicReleaseAuditError(f"NUL PDF name escape: {label}")
            decoded.append(value)
            name_cursor += 3
        normalized.append(0x2F)
        normalized.extend(decoded)
        cursor = end
    return bytes(normalized)


def _decode_pdf_stream(
    payload: bytes, dictionary: bytes, label: str
) -> tuple[bytes, bool]:
    decoded = payload
    opaque_binary = re.search(
        rb"/(?:Length1|Length2|Length3)\b", dictionary
    ) is not None
    normalized_dictionary = _normalize_pdf_names(dictionary, f"{label}-dictionary")
    for stream_filter in _pdf_filters(normalized_dictionary, label):
        if stream_filter in {b"FlateDecode", b"Fl"}:
            decoded = _bounded_flate(decoded, label)
        elif stream_filter in {b"ASCIIHexDecode", b"AHx"}:
            decoded = _ascii_hex_decode(decoded, label)
        elif stream_filter in {b"ASCII85Decode", b"A85"}:
            decoded = _ascii85_decode(decoded, label)
        elif stream_filter in {b"RunLengthDecode", b"RL"}:
            decoded = _run_length_decode(decoded, label)
        elif stream_filter in PDF_IMAGE_FILTERS and re.search(
            rb"/Subtype\s*/Image\b", normalized_dictionary
        ):
            opaque_binary = True
            break
        else:
            raise PublicReleaseAuditError(
                f"unsupported non-image PDF filter {stream_filter!r}: {label}"
            )
    return decoded, opaque_binary


def _scan_pdf_strings(raw: bytes, label: str) -> None:
    for index, match in enumerate(re.finditer(rb"(?<!<)<([0-9A-Fa-f\s]+)>", raw)):
        compact = re.sub(rb"\s+", b"", match.group(1))
        if not compact:
            continue
        if len(compact) % 2:
            compact += b"0"
        try:
            decoded = binascii.unhexlify(compact)
        except binascii.Error as exc:
            raise PublicReleaseAuditError(f"malformed PDF hex string: {label}") from exc
        _scan_bytes(decoded, f"{label}!/hex-string-{index}")

    for index, match in enumerate(re.finditer(rb"(?s)\((?:\\.|[^\\()])*\)", raw)):
        value = match.group(0)[1:-1]
        value = re.sub(
            rb"\\([0-7]{1,3})",
            lambda item: bytes((int(item.group(1), 8) & 0xFF,)),
            value,
        )
        value = re.sub(rb"\\(?:\r\n|\r|\n)", b"", value)
        value = re.sub(
            rb"\\([nrtbf()\\])",
            lambda item: {
                b"n": b"\n",
                b"r": b"\r",
                b"t": b"\t",
                b"b": b"\b",
                b"f": b"\f",
                b"(": b"(",
                b")": b")",
                b"\\": b"\\",
            }[item.group(1)],
            value,
        )
        _scan_bytes(value, f"{label}!/literal-string-{index}")


def _validate_pdf_open_actions(raw: bytes, label: str) -> None:
    for match in re.finditer(rb"/OpenAction\s+(?P<value>\[[^\]]*\]|[0-9]+\s+[0-9]+\s+R)", raw):
        value = match.group("value")
        if value.startswith(b"["):
            if re.fullmatch(
                rb"\[\s*(?:[0-9]+\s+[0-9]+\s+R|[0-9]+)\s+"
                rb"/(?:Fit|FitB|FitH|FitBH|FitV|FitBV|XYZ)(?:\s+[-+0-9.]+)*\s*\]",
                value,
            ) is None:
                raise PublicReleaseAuditError(f"unsafe PDF OpenAction destination in {label}")
            continue
        object_number, generation, _ = value.split()
        object_match = re.search(
            rb"(?m)^"
            + re.escape(object_number)
            + rb"\s+"
            + re.escape(generation)
            + rb"\s+obj\s*(?P<body>.*?)\s*endobj",
            raw,
            re.S,
        )
        if object_match is None:
            raise PublicReleaseAuditError(f"unresolved PDF OpenAction in {label}")
        body = object_match.group("body")
        if (
            re.search(rb"/S\s*/GoTo\b", body) is None
            or re.search(rb"/D(?:est)?\s+(?:\[[^\]]+\]|[0-9]+\s+[0-9]+\s+R|\([^)]*\))", body)
            is None
            or PDF_FORBIDDEN_FEATURES.search(body)
        ):
            raise PublicReleaseAuditError(f"unsafe PDF OpenAction in {label}")


def _scan_pdf(raw: bytes, label: str) -> None:
    """Parse public PDF containers and inspect metadata and decoded streams."""

    if (
        not re.match(rb"%PDF-1\.[0-9]\r?(?:\n|$)", raw[:16])
        or b"%%EOF" not in raw[-2048:]
    ):
        raise PublicReleaseAuditError(f"invalid PDF framing: {label}")
    stream_matches = tuple(PDF_STREAM.finditer(raw))
    structural = bytearray(raw)
    for match in stream_matches:
        start, end = match.span("payload")
        structural[start:end] = b"\0" * (end - start)
    structural_bytes = bytes(structural)
    normalized_structural = _normalize_pdf_names(structural_bytes, label)
    _scan_bytes(normalized_structural, label)
    if PDF_FORBIDDEN_FEATURES.search(normalized_structural):
        raise PublicReleaseAuditError(f"PDF attachment or active content in {label}")
    _validate_pdf_open_actions(normalized_structural, label)
    _scan_pdf_strings(structural_bytes, label)
    for index, match in enumerate(stream_matches):
        dictionary = match.group("dictionary")
        payload = match.group("payload")
        stream_label = f"{label}!/stream-{index}"
        normalized_dictionary = _normalize_pdf_names(
            dictionary, f"{stream_label}-dictionary"
        )
        _scan_bytes(normalized_dictionary, f"{stream_label}-dictionary")
        decoded, opaque_binary = _decode_pdf_stream(payload, dictionary, stream_label)
        if opaque_binary:
            printable = b"\n".join(re.findall(rb"[\x20-\x7e]{8,}", decoded))
            if printable:
                _scan_bytes(printable, f"{stream_label}-printable-binary")
        else:
            normalized_decoded = _normalize_pdf_names(
                decoded, stream_label, strict_syntax=False
            )
            _scan_bytes(normalized_decoded, stream_label)
            if PDF_FORBIDDEN_FEATURES.search(normalized_decoded):
                raise PublicReleaseAuditError(
                    f"PDF attachment or active content in decoded stream: {label}"
                )
            _scan_pdf_strings(decoded, stream_label)


def audit() -> tuple[int, int]:
    tree = resolve_release_tree(ROOT)
    paths = _validated_paths(tree.paths)
    total_bytes = 0
    for relative in paths:
        raw = tree.read_bytes(relative)
        total_bytes += len(raw)
        label = relative.as_posix()
        suffix = relative.suffix.casefold()
        if suffix == ".pptx":
            _scan_pptx(raw, label)
        elif suffix == ".png":
            _scan_png(raw, label)
        elif suffix == ".pdf":
            _scan_pdf(raw, label)
        elif suffix == ".svg":
            _scan_svg(raw, label)
        elif suffix == ".drawio":
            _scan_drawio(raw, label)
        else:
            if b"\0" in raw:
                raise PublicReleaseAuditError(f"unexpected binary tracked file: {label}")
            _scan_bytes(raw, label)
    return len(paths), total_bytes


def main() -> int:
    file_count, total_bytes = audit()
    print(f"PUBLIC_RELEASE_AUDIT_PASS files={file_count} bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        PublicReleaseAuditError,
        ReleaseTreeError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as exc:
        print(f"PUBLIC_RELEASE_AUDIT_FAIL: {exc}")
        raise SystemExit(1) from exc
