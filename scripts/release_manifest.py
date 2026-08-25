"""Build or verify the deterministic PhasePair Git-index release manifest."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
MANIFEST_RELATIVE = PurePosixPath("RELEASE_FILES.sha256")


class ManifestError(RuntimeError):
    """Raised when the tracked tree cannot be represented or verified exactly."""


def _tracked_paths() -> tuple[PurePosixPath, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "-z"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    raw_paths = completed.stdout.split(b"\0")
    if raw_paths and raw_paths[-1] == b"":
        raw_paths.pop()
    paths: list[PurePosixPath] = []
    for raw in raw_paths:
        try:
            decoded = raw.decode("utf-8", errors="strict")
        except UnicodeDecodeError as exc:
            raise ManifestError("tracked paths must be valid UTF-8") from exc
        if "\t" in decoded or "\n" in decoded or "\r" in decoded:
            raise ManifestError("tracked paths may not contain tabs or newlines")
        path = PurePosixPath(decoded)
        if path.is_absolute() or ".." in path.parts:
            raise ManifestError(f"unsafe tracked path: {decoded!r}")
        if path != MANIFEST_RELATIVE:
            paths.append(path)
    ordered = tuple(sorted(paths, key=lambda item: item.as_posix().encode("utf-8")))
    if len(ordered) != len(set(ordered)):
        raise ManifestError("duplicate tracked path")
    return ordered


def _index_blob(path: PurePosixPath) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", f":{path.as_posix()}"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _index_mode(path: PurePosixPath) -> str:
    completed = subprocess.run(
        ["git", "ls-files", "--stage", "-z", "--", path.as_posix()],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    entries = [entry for entry in completed.stdout.split(b"\0") if entry]
    if len(entries) != 1 or b"\t" not in entries[0]:
        raise ManifestError(f"expected one staged entry: {path}")
    header, listed_path = entries[0].split(b"\t", 1)
    if listed_path.decode("utf-8", errors="strict") != path.as_posix():
        raise ManifestError(f"staged path identity mismatch: {path}")
    fields = header.split(b" ")
    if len(fields) != 3:
        raise ManifestError(f"malformed staged entry: {path}")
    mode = fields[0].decode("ascii", errors="strict")
    if mode not in {"100644", "100755"}:
        raise ManifestError(f"tracked entry must be a regular file: {path}")
    return mode


def _row(path: PurePosixPath) -> str:
    absolute = ROOT.joinpath(*path.parts)
    if absolute.is_symlink() or not absolute.is_file():
        raise ManifestError(f"tracked release entry must be a regular file: {path}")
    _index_mode(path)
    raw = _index_blob(path)
    return f"{hashlib.sha256(raw).hexdigest()}\t{len(raw)}\t{path.as_posix()}"


def render_manifest() -> bytes:
    header = "# sha256\\tbytes\\tgit_index_path; self excluded\n"
    rows = "\n".join(_row(path) for path in _tracked_paths())
    return (header + rows + ("\n" if rows else "")).encode("utf-8")


def build_manifest() -> None:
    destination = ROOT / MANIFEST_RELATIVE.as_posix()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(render_manifest())
    temporary.replace(destination)


def verify_manifest() -> None:
    destination = ROOT / MANIFEST_RELATIVE.as_posix()
    if not destination.is_file() or destination.is_symlink():
        raise ManifestError("release manifest is missing or not a regular file")
    expected = render_manifest()
    actual = _index_blob(MANIFEST_RELATIVE)
    if actual != expected:
        raise ManifestError("release manifest does not exactly match the tracked tree")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "verify"))
    args = parser.parse_args(argv)
    if args.command == "build":
        build_manifest()
        print("RELEASE_MANIFEST_BUILT")
    else:
        verify_manifest()
        print("RELEASE_MANIFEST_VERIFIED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ManifestError, OSError, subprocess.CalledProcessError) as exc:
        print(f"RELEASE_MANIFEST_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
