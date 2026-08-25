"""Scan every blob and message reachable from every Git ref for public-release leaks."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path, PurePosixPath
import subprocess
import sys

sys.dont_write_bytecode = True

from public_release_audit import (  # noqa: E402
    PublicReleaseAuditError,
    _scan_bytes,
    _scan_pdf,
    _scan_png,
    _scan_pptx,
    _validated_paths,
)


ROOT = Path(__file__).resolve().parents[1]


class AllRefsAuditError(RuntimeError):
    """A reachable Git object cannot be safely audited."""


def _git(*arguments: str, input_bytes: bytes | None = None) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if completed.returncode != 0:
        raise AllRefsAuditError("Git object traversal failed")
    return completed.stdout


def _reachable_objects() -> tuple[tuple[str, str | None], ...]:
    raw = _git("rev-list", "--objects", "--all")
    rows: list[tuple[str, str | None]] = []
    for line in raw.decode("utf-8", "strict").splitlines():
        object_id, separator, path = line.partition(" ")
        if len(object_id) != 40 or any(char not in "0123456789abcdef" for char in object_id):
            raise AllRefsAuditError("rev-list emitted a noncanonical object id")
        rows.append((object_id, path if separator else None))
    if not rows:
        raise AllRefsAuditError("repository has no reachable objects")
    return tuple(rows)


def _batch_read(object_ids: tuple[str, ...]) -> dict[str, tuple[str, bytes]]:
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.stdin is None or process.stdout is None:
        raise AllRefsAuditError("could not open Git batch object stream")
    result: dict[str, tuple[str, bytes]] = {}
    try:
        for object_id in object_ids:
            process.stdin.write(object_id.encode("ascii") + b"\n")
            process.stdin.flush()
            header = process.stdout.readline().decode("ascii", "strict").strip().split()
            if len(header) != 3 or header[0] != object_id:
                raise AllRefsAuditError("Git batch object header is malformed")
            object_type = header[1]
            size = int(header[2])
            if size < 0 or size > 256 * 1024 * 1024:
                raise AllRefsAuditError("reachable Git object exceeds audit size bound")
            payload = process.stdout.read(size)
            delimiter = process.stdout.read(1)
            if len(payload) != size or delimiter != b"\n":
                raise AllRefsAuditError("Git batch object payload is truncated")
            result[object_id] = (object_type, payload)
    finally:
        process.stdin.close()
        process.wait(timeout=30)
    if process.returncode != 0:
        raise AllRefsAuditError("Git batch object stream failed")
    return result


def _scan_blob(raw: bytes, path: PurePosixPath) -> None:
    label = path.as_posix()
    suffix = path.suffix.casefold()
    if suffix == ".pptx":
        _scan_pptx(raw, label)
    elif suffix == ".png":
        _scan_png(raw, label)
    elif suffix == ".pdf":
        _scan_pdf(raw, label)
    else:
        if b"\0" in raw:
            raise PublicReleaseAuditError(f"unexpected binary reachable blob: {label}")
        _scan_bytes(raw, label)


def audit_all_refs() -> tuple[int, int, int]:
    refs = tuple(
        line
        for line in _git("for-each-ref", "--format=%(refname)").decode("utf-8", "strict").splitlines()
        if line
    )
    if not refs:
        raise AllRefsAuditError("repository has no refs")
    rows = _reachable_objects()
    paths_by_object: dict[str, list[PurePosixPath]] = defaultdict(list)
    for object_id, raw_path in rows:
        if raw_path is not None:
            path = PurePosixPath(raw_path)
            _validated_paths((path,))
            paths_by_object[object_id].append(path)
    objects = _batch_read(tuple(dict.fromkeys(object_id for object_id, _ in rows)))
    blob_count = 0
    total_blob_bytes = 0
    for object_id, (object_type, payload) in objects.items():
        if object_type == "blob":
            paths = paths_by_object.get(object_id)
            if not paths:
                raise AllRefsAuditError("reachable blob has no path")
            for path in paths:
                _scan_blob(payload, path)
            blob_count += 1
            total_blob_bytes += len(payload)
        elif object_type in {"commit", "tag"}:
            _scan_bytes(payload, f"git-{object_type}:{object_id}")
        elif object_type != "tree":
            raise AllRefsAuditError("unexpected reachable Git object type")
    return len(refs), blob_count, total_blob_bytes


def main() -> int:
    refs, blobs, total_bytes = audit_all_refs()
    print(f"ALL_REFS_AUDIT_PASS refs={refs} blobs={blobs} bytes={total_bytes}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        AllRefsAuditError,
        OSError,
        PublicReleaseAuditError,
        subprocess.SubprocessError,
        UnicodeDecodeError,
        ValueError,
    ) as exc:
        print(f"ALL_REFS_AUDIT_FAIL: {exc}")
        raise SystemExit(1) from exc

