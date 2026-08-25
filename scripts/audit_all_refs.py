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
    _scan_drawio,
    _scan_pdf,
    _scan_png,
    _scan_pptx,
    _scan_svg,
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


def _reachable_objects() -> tuple[str, ...]:
    raw = _git("rev-list", "--objects", "--all")
    rows: list[str] = []
    for line in raw.decode("utf-8", "strict").splitlines():
        object_id = line.partition(" ")[0]
        if len(object_id) not in {40, 64} or any(
            char not in "0123456789abcdef" for char in object_id
        ):
            raise AllRefsAuditError("rev-list emitted a noncanonical object id")
        rows.append(object_id)
    if not rows:
        raise AllRefsAuditError("repository has no reachable objects")
    return tuple(dict.fromkeys(rows))


def _ref_targets() -> tuple[str, ...]:
    raw = _git("for-each-ref", "--format=%(objectname)")
    targets = tuple(line for line in raw.decode("ascii", "strict").splitlines() if line)
    if not targets or any(
        len(object_id) not in {40, 64}
        or any(character not in "0123456789abcdef" for character in object_id)
        for object_id in targets
    ):
        raise AllRefsAuditError("Git refs contain a malformed object ID")
    return targets


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
    elif suffix == ".svg":
        _scan_svg(raw, label)
    elif suffix == ".drawio":
        _scan_drawio(raw, label)
    else:
        if b"\0" in raw:
            raise PublicReleaseAuditError(f"unexpected binary reachable blob: {label}")
        _scan_bytes(raw, label)


def _tree_bindings(treeish: str) -> tuple[tuple[str, PurePosixPath], ...]:
    raw = _git("ls-tree", "-r", "-z", "--full-tree", treeish)
    bindings: list[tuple[str, PurePosixPath]] = []
    for row in raw.split(b"\0"):
        if not row:
            continue
        metadata, separator, raw_path = row.partition(b"\t")
        fields = metadata.split(b" ")
        if not separator or len(fields) != 3:
            raise AllRefsAuditError("Git tree traversal emitted a malformed row")
        mode, object_type, raw_object_id = fields
        if mode != b"100644" and mode != b"100755":
            raise AllRefsAuditError("reachable tree contains a symlink, gitlink, or special mode")
        if object_type != b"blob":
            raise AllRefsAuditError("recursive Git tree traversal emitted a non-blob leaf")
        try:
            object_id = raw_object_id.decode("ascii", "strict")
            decoded_path = raw_path.decode("utf-8", "strict")
        except UnicodeDecodeError as exc:
            raise AllRefsAuditError("Git tree path or object ID is not canonical text") from exc
        if len(object_id) not in {40, 64} or any(
            character not in "0123456789abcdef" for character in object_id
        ):
            raise AllRefsAuditError("Git tree traversal emitted a malformed object ID")
        path = PurePosixPath(decoded_path)
        if (
            not decoded_path
            or any(character in decoded_path for character in "\0\t\n\r\\:")
            or path.is_absolute()
            or path == PurePosixPath(".")
            or ".." in path.parts
            or path.as_posix() != decoded_path
        ):
            raise AllRefsAuditError(f"reachable tree has an unsafe path: {decoded_path!r}")
        _validated_paths((path,))
        bindings.append((object_id, path))
    return tuple(bindings)


def _tag_target(payload: bytes) -> str:
    first_line = payload.partition(b"\n")[0]
    prefix = b"object "
    if not first_line.startswith(prefix):
        raise AllRefsAuditError("annotated tag object lacks a target")
    try:
        target = first_line[len(prefix) :].decode("ascii", "strict")
    except UnicodeDecodeError as exc:
        raise AllRefsAuditError("annotated tag target is malformed") from exc
    if len(target) not in {40, 64} or any(
        character not in "0123456789abcdef" for character in target
    ):
        raise AllRefsAuditError("annotated tag target is malformed")
    return target


def audit_all_refs() -> tuple[int, int, int]:
    refs = tuple(
        line
        for line in _git("for-each-ref", "--format=%(refname)").decode("utf-8", "strict").splitlines()
        if line
    )
    if not refs:
        raise AllRefsAuditError("repository has no refs")
    object_ids = _reachable_objects()
    objects = _batch_read(object_ids)
    paths_by_object: dict[str, set[PurePosixPath]] = defaultdict(set)
    treeish_ids = {
        object_id for object_id, (object_type, _) in objects.items() if object_type == "commit"
    }
    targets = _ref_targets()
    for target in targets:
        current = target
        visited: set[str] = set()
        while current in objects and objects[current][0] == "tag":
            if current in visited:
                raise AllRefsAuditError("annotated tags contain a cycle")
            visited.add(current)
            current = _tag_target(objects[current][1])
        if current in objects and objects[current][0] == "tree":
            treeish_ids.add(current)
        elif current in objects and objects[current][0] == "blob":
            raise AllRefsAuditError("a public Git ref points directly to a pathless blob")
    binding_count = 0
    for treeish in sorted(treeish_ids):
        for object_id, path in _tree_bindings(treeish):
            if object_id not in objects or objects[object_id][0] != "blob":
                raise AllRefsAuditError("tree binding points outside the reachable blob census")
            paths_by_object[object_id].add(path)
            binding_count += 1
            if binding_count > 10_000_000:
                raise AllRefsAuditError("reachable tree path-binding census exceeds audit bound")
    blob_count = 0
    total_blob_bytes = 0
    for object_id, (object_type, payload) in objects.items():
        if object_type == "blob":
            paths = paths_by_object.get(object_id)
            if not paths:
                raise AllRefsAuditError("reachable blob has no path")
            for path in sorted(paths, key=lambda value: value.as_posix().encode("utf-8")):
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
