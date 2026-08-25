"""Build or verify the deterministic PhasePair Git-index release manifest."""

from __future__ import annotations

import argparse
import hashlib
import subprocess
import sys
from pathlib import Path, PurePosixPath

sys.dont_write_bytecode = True

from release_tree import (  # noqa: E402
    MANIFEST_HEADER,
    MANIFEST_RELATIVE,
    ReleaseTreeError,
    clean_git_environment,
    git_tracked_paths,
    is_exact_git_checkout,
    load_archive_manifest,
)


ROOT = Path(__file__).resolve().parents[1]


class ManifestError(RuntimeError):
    """Raised when the tracked tree cannot be represented or verified exactly."""


def _tracked_paths() -> tuple[PurePosixPath, ...]:
    return tuple(
        path for path in git_tracked_paths(ROOT) if path != MANIFEST_RELATIVE
    )


def _index_blob(path: PurePosixPath) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", f":{path.as_posix()}"],
        cwd=ROOT,
        check=True,
        env=clean_git_environment(),
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _row(path: PurePosixPath) -> str:
    raw = _index_blob(path)
    return f"{hashlib.sha256(raw).hexdigest()}\t{len(raw)}\t{path.as_posix()}"


def render_manifest() -> bytes:
    header = MANIFEST_HEADER.decode("ascii")
    rows = "\n".join(_row(path) for path in _tracked_paths())
    return (header + rows + ("\n" if rows else "")).encode("utf-8")


def build_manifest() -> None:
    if not is_exact_git_checkout(ROOT):
        raise ManifestError("manifest build requires the exact Git checkout")
    destination = ROOT / MANIFEST_RELATIVE.as_posix()
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_bytes(render_manifest())
    temporary.replace(destination)


def verify_manifest() -> None:
    if not is_exact_git_checkout(ROOT):
        load_archive_manifest(ROOT)
        return
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
    except (
        ManifestError,
        OSError,
        ReleaseTreeError,
        subprocess.CalledProcessError,
    ) as exc:
        print(f"RELEASE_MANIFEST_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
