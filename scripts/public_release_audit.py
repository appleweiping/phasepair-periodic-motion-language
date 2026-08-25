"""Fail closed on unsafe tracked paths, local locators, and common secrets."""

from __future__ import annotations

import re
import subprocess
import zipfile
from pathlib import Path, PurePosixPath


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


def _tracked_paths() -> tuple[PurePosixPath, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths: list[PurePosixPath] = []
    for raw in completed.stdout.split(b"\0"):
        if not raw:
            continue
        decoded = raw.decode("utf-8", errors="strict")
        path = PurePosixPath(decoded)
        if path.is_absolute() or ".." in path.parts:
            raise PublicReleaseAuditError(f"unsafe tracked path: {decoded!r}")
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


def _scan_pptx(path: Path, label: str) -> None:
    try:
        with zipfile.ZipFile(path) as archive:
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


def audit() -> tuple[int, int]:
    paths = _tracked_paths()
    total_bytes = 0
    for relative in paths:
        absolute = ROOT.joinpath(*relative.parts)
        if absolute.is_symlink() or not absolute.is_file():
            raise PublicReleaseAuditError(
                f"tracked item must be one regular file: {relative.as_posix()}"
            )
        raw = absolute.read_bytes()
        total_bytes += len(raw)
        label = relative.as_posix()
        if relative.suffix.casefold() == ".pptx":
            _scan_pptx(absolute, label)
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
        subprocess.CalledProcessError,
        UnicodeDecodeError,
    ) as exc:
        print(f"PUBLIC_RELEASE_AUDIT_FAIL: {exc}")
        raise SystemExit(1) from exc
