"""Fail-closed scanner for PhaseSet wheels, codeloads, and release assets."""

from __future__ import annotations

import argparse
import hashlib
import io
import os
import re
import stat
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True

from public_release_audit import (  # noqa: E402
    PublicReleaseAuditError,
    _scan_bytes,
    _scan_pdf,
    _scan_png,
    _scan_pptx,
    _validated_paths,
)
from release_tree import MANIFEST_HEADER, MANIFEST_RELATIVE, SHA256_PATTERN  # noqa: E402


MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_ARCHIVE_BYTES = 256 * 1024 * 1024
SUPPORTED_SUFFIXES = (
    ".whl",
    ".zip",
    ".tar.gz",
    ".tgz",
    ".pdf",
    ".pptx",
    ".png",
    ".svg",
    ".json",
    ".txt",
    ".md",
    ".sha256",
)


class ReleaseArtifactAuditError(RuntimeError):
    """A release artifact is malformed, unsafe, or not manifest-bound."""


def _safe_member_path(decoded: str) -> PurePosixPath:
    if not decoded or any(character in decoded for character in "\0\t\n\r\\:"):
        raise ReleaseArtifactAuditError(f"invalid archive member: {decoded!r}")
    path = PurePosixPath(decoded.rstrip("/"))
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != decoded.rstrip("/")
    ):
        raise ReleaseArtifactAuditError(f"unsafe archive member: {decoded!r}")
    return path


def _strip_common_root(
    members: dict[PurePosixPath, bytes],
) -> dict[PurePosixPath, bytes]:
    if not members:
        raise ReleaseArtifactAuditError("archive contains no regular files")
    roots = {path.parts[0] for path in members}
    if len(roots) != 1 or any(len(path.parts) < 2 for path in members):
        return members
    return {
        PurePosixPath(*path.parts[1:]): raw
        for path, raw in members.items()
    }


def _read_zip(raw: bytes, label: str) -> dict[PurePosixPath, bytes]:
    members: dict[PurePosixPath, bytes] = {}
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            total = 0
            for info in archive.infolist():
                path = _safe_member_path(info.filename)
                mode = info.external_attr >> 16
                if info.is_dir():
                    continue
                file_type = stat.S_IFMT(mode)
                if file_type and file_type != stat.S_IFREG:
                    raise ReleaseArtifactAuditError(
                        f"non-regular ZIP member in {label}: {info.filename!r}"
                    )
                if info.file_size > MAX_MEMBER_BYTES:
                    raise ReleaseArtifactAuditError(
                        f"oversized ZIP member in {label}: {info.filename!r}"
                    )
                total += info.file_size
                if total > MAX_ARCHIVE_BYTES:
                    raise ReleaseArtifactAuditError(f"oversized ZIP payload: {label}")
                if path in members:
                    raise ReleaseArtifactAuditError(
                        f"duplicate ZIP member in {label}: {info.filename!r}"
                    )
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise ReleaseArtifactAuditError(
                        f"ZIP size mismatch in {label}: {info.filename!r}"
                    )
                members[path] = payload
    except zipfile.BadZipFile as exc:
        raise ReleaseArtifactAuditError(f"invalid ZIP artifact: {label}") from exc
    return members


def _read_tar(raw: bytes, label: str) -> dict[PurePosixPath, bytes]:
    members: dict[PurePosixPath, bytes] = {}
    try:
        with tarfile.open(fileobj=io.BytesIO(raw), mode="r:*") as archive:
            total = 0
            for info in archive.getmembers():
                path = _safe_member_path(info.name)
                if info.isdir():
                    continue
                if not info.isfile():
                    raise ReleaseArtifactAuditError(
                        f"non-regular TAR member in {label}: {info.name!r}"
                    )
                if info.size > MAX_MEMBER_BYTES:
                    raise ReleaseArtifactAuditError(
                        f"oversized TAR member in {label}: {info.name!r}"
                    )
                total += info.size
                if total > MAX_ARCHIVE_BYTES:
                    raise ReleaseArtifactAuditError(f"oversized TAR payload: {label}")
                if path in members:
                    raise ReleaseArtifactAuditError(
                        f"duplicate TAR member in {label}: {info.name!r}"
                    )
                extracted = archive.extractfile(info)
                if extracted is None:
                    raise ReleaseArtifactAuditError(
                        f"unreadable TAR member in {label}: {info.name!r}"
                    )
                payload = extracted.read(MAX_MEMBER_BYTES + 1)
                if len(payload) != info.size:
                    raise ReleaseArtifactAuditError(
                        f"TAR size mismatch in {label}: {info.name!r}"
                    )
                members[path] = payload
    except tarfile.TarError as exc:
        raise ReleaseArtifactAuditError(f"invalid TAR artifact: {label}") from exc
    return members


def _scan_payload(raw: bytes, path: PurePosixPath, label: str) -> None:
    suffix = path.suffix.casefold()
    if suffix == ".pptx":
        _scan_pptx(raw, label)
    elif suffix == ".png":
        _scan_png(raw, label)
    elif suffix == ".pdf":
        _scan_pdf(raw, label)
    else:
        if b"\0" in raw:
            raise ReleaseArtifactAuditError(f"unexpected binary release member: {label}")
        scan_label = ".gitignore" if path == PurePosixPath(".gitignore") else label
        _scan_bytes(raw, scan_label)


def _parse_manifest(raw: bytes) -> dict[PurePosixPath, tuple[str, int]]:
    if not raw.startswith(MANIFEST_HEADER) or not raw.endswith(b"\n"):
        raise ReleaseArtifactAuditError("embedded release manifest is not canonical")
    expected: dict[PurePosixPath, tuple[str, int]] = {}
    body = raw[len(MANIFEST_HEADER) : -1]
    if not body:
        raise ReleaseArtifactAuditError("embedded release manifest is empty")
    previous: bytes | None = None
    for raw_row in body.split(b"\n"):
        try:
            digest, raw_size, decoded = raw_row.decode("utf-8").split("\t")
        except (UnicodeDecodeError, ValueError) as exc:
            raise ReleaseArtifactAuditError("malformed embedded manifest row") from exc
        path = _safe_member_path(decoded)
        if path == MANIFEST_RELATIVE or SHA256_PATTERN.fullmatch(digest) is None:
            raise ReleaseArtifactAuditError("invalid embedded manifest entry")
        if previous is not None and decoded.encode("utf-8") <= previous:
            raise ReleaseArtifactAuditError("embedded manifest order is not canonical")
        previous = decoded.encode("utf-8")
        if re.fullmatch(r"0|[1-9][0-9]*", raw_size) is None:
            raise ReleaseArtifactAuditError("invalid embedded manifest size")
        if path in expected:
            raise ReleaseArtifactAuditError("duplicate embedded manifest path")
        expected[path] = (digest, int(raw_size))
    return expected


def _verify_embedded_manifest(members: dict[PurePosixPath, bytes], label: str) -> None:
    manifest = members.get(MANIFEST_RELATIVE)
    if manifest is None:
        return
    expected = _parse_manifest(manifest)
    expected_paths = set(expected) | {MANIFEST_RELATIVE}
    if set(members) != expected_paths:
        missing = sorted(path.as_posix() for path in expected_paths - set(members))
        extra = sorted(path.as_posix() for path in set(members) - expected_paths)
        raise ReleaseArtifactAuditError(
            f"archive file set differs from embedded manifest in {label}: "
            f"missing={missing!r}, extra={extra!r}"
        )
    for path, (digest, size) in expected.items():
        raw = members[path]
        if len(raw) != size or hashlib.sha256(raw).hexdigest() != digest:
            raise ReleaseArtifactAuditError(
                f"embedded manifest mismatch in {label}: {path.as_posix()}"
            )


def audit_artifact(path: Path, *, require_manifest: bool = False) -> tuple[int, int]:
    raw = path.read_bytes()
    label = path.name
    lower = label.casefold()
    if lower.endswith((".whl", ".zip")):
        members = _read_zip(raw, label)
    elif lower.endswith((".tar.gz", ".tgz")):
        members = _read_tar(raw, label)
    else:
        _scan_payload(raw, PurePosixPath(label), label)
        return 1, len(raw)

    members = _strip_common_root(members)
    validated = _validated_paths(tuple(members))
    if require_manifest and MANIFEST_RELATIVE not in members:
        raise ReleaseArtifactAuditError(f"release manifest missing from {label}")
    _verify_embedded_manifest(members, label)
    total = 0
    for member in validated:
        payload = members[member]
        total += len(payload)
        _scan_payload(payload, member, f"{label}!/{member.as_posix()}")
    return len(validated), total


def _artifact_paths(inputs: list[str]) -> tuple[Path, ...]:
    paths: list[Path] = []
    for raw in inputs:
        path = Path(raw)
        if path.is_dir():
            paths.extend(
                candidate
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.name.casefold().endswith(SUPPORTED_SUFFIXES)
            )
        elif path.is_file():
            paths.append(path)
        else:
            raise ReleaseArtifactAuditError(f"artifact path does not exist: {raw!r}")
    ordered = tuple(sorted(set(paths), key=lambda value: os.fsencode(str(value))))
    if not ordered:
        raise ReleaseArtifactAuditError("no supported release artifacts found")
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    parser.add_argument(
        "--require-manifest",
        action="store_true",
        help="require every archive to contain a manifest-bound release tree",
    )
    args = parser.parse_args(argv)
    artifacts = _artifact_paths(args.paths)
    total_members = 0
    total_bytes = 0
    for path in artifacts:
        members, byte_count = audit_artifact(path, require_manifest=args.require_manifest)
        total_members += members
        total_bytes += byte_count
        print(f"ARTIFACT {hashlib.sha256(path.read_bytes()).hexdigest()} {path}")
    print(
        "RELEASE_ARTIFACT_AUDIT_PASS "
        f"artifacts={len(artifacts)} members={total_members} bytes={total_bytes}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        PublicReleaseAuditError,
        ReleaseArtifactAuditError,
        UnicodeDecodeError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"RELEASE_ARTIFACT_AUDIT_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
