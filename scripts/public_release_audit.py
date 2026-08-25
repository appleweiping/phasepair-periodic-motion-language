"""Fail closed on unsafe tracked paths, local locators, and common secrets."""

from __future__ import annotations

import io
import re
import subprocess
import sys
import zipfile
import zlib
from pathlib import Path, PurePosixPath

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
        "data",
        "datasets",
        "mlruns",
        "recommend paper",
        "refine-logs",
        "results",
        "wandb",
        "wheelhouse",
    }
)
FORBIDDEN_MARKERS = (
    b"D:" + b"/" + b"Project" + b"/",
    b"D:" + b"\\" + b"Project" + b"\\",
    b"/" + b"home" + b"/",
    b"/" + b"mnt" + b"/",
    b"/" + b"root" + b"/",
    b"/" + b"tmp" + b"/",
    b"recommend" + b" " + b"paper" + b"/",
    b"refine" + b"-" + b"logs" + b"/",
)
SECRET_PATTERNS = (
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bghp_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\b(?:[0-9]{1,3}\.){3}[0-9]{1,3}:[0-9]{2,5}\b"),
)


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
            if marker in raw:
                raise PublicReleaseAuditError(f"private/local marker in {label}")
    for pattern in SECRET_PATTERNS:
        if pattern.search(raw):
            raise PublicReleaseAuditError(f"credential or endpoint pattern in {label}")


def _scan_pptx(raw: bytes, label: str) -> None:
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise PublicReleaseAuditError(
                        f"unsafe archive member in {label}: {member.filename!r}"
                    )
                if not member.is_dir():
                    _scan_bytes(
                        archive.read(member),
                        f"{label}!/{member.filename}",
                    )
    except zipfile.BadZipFile as exc:
        raise PublicReleaseAuditError(f"invalid PPTX archive: {label}") from exc


def _scan_png(raw: bytes, label: str) -> None:
    """Validate PNG framing/CRC and inspect embedded textual diagram payloads."""

    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise PublicReleaseAuditError(f"invalid PNG signature: {label}")
    offset = 8
    saw_iend = False
    while offset < len(raw):
        if offset + 12 > len(raw):
            raise PublicReleaseAuditError(f"truncated PNG chunk: {label}")
        length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        end = offset + 12 + length
        if length > 64 * 1024 * 1024 or end > len(raw):
            raise PublicReleaseAuditError(f"invalid PNG chunk length: {label}")
        payload = raw[offset + 8 : offset + 8 + length]
        expected_crc = int.from_bytes(raw[offset + 8 + length : end], "big")
        actual_crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
        if actual_crc != expected_crc:
            raise PublicReleaseAuditError(f"PNG CRC mismatch: {label}")
        if chunk_type == b"tEXt":
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
        if chunk_type == b"IEND":
            saw_iend = True
            if end != len(raw):
                raise PublicReleaseAuditError(f"trailing bytes after PNG IEND: {label}")
            break
        offset = end
    if not saw_iend:
        raise PublicReleaseAuditError(f"PNG lacks IEND: {label}")


def _scan_pdf(raw: bytes, label: str) -> None:
    """Perform structural boundary checks and scan raw public PDF bytes."""

    if not raw.startswith(b"%PDF-") or b"%%EOF" not in raw[-2048:]:
        raise PublicReleaseAuditError(f"invalid PDF framing: {label}")
    _scan_bytes(raw, label)


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
