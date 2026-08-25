"""Verify that repository-local Markdown links resolve inside the public tree."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

sys.dont_write_bytecode = True

from release_tree import ReleaseTree, ReleaseTreeError, resolve_release_tree  # noqa: E402


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


class LinkAuditError(RuntimeError):
    """A local Markdown link is malformed, unsafe, or missing."""


def _resolve_target(document: PurePosixPath, raw_path: str) -> PurePosixPath | None:
    candidate = PurePosixPath(raw_path)
    if candidate.is_absolute():
        return None
    parts = list(document.parent.parts)
    for part in candidate.parts:
        if part in {"", "."}:
            continue
        if part == "..":
            if not parts:
                return None
            parts.pop()
        else:
            parts.append(part)
    return PurePosixPath(*parts)


def _tree_target_exists(tree: ReleaseTree, target: PurePosixPath) -> bool:
    if target in tree.paths:
        return True
    target_parts = target.parts
    return any(path.parts[: len(target_parts)] == target_parts for path in tree.paths)


def _link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    elif " " in target:
        target = target.split(" ", 1)[0]
    return unquote(target)


def audit() -> int:
    tree = resolve_release_tree(ROOT)
    checked = 0
    failures: list[str] = []
    for document in (path for path in tree.paths if path.suffix.casefold() == ".md"):
        text = tree.read_bytes(document).decode("utf-8", errors="strict")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in LINK_PATTERN.finditer(line):
                target = _link_target(match.group(1))
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                resolved = _resolve_target(document, parsed.path)
                if resolved is None:
                    failures.append(
                        f"{document.as_posix()}:{line_number}: unsafe {target!r}"
                    )
                    continue
                checked += 1
                if not _tree_target_exists(tree, resolved):
                    failures.append(
                        f"{document.as_posix()}:{line_number}: missing {target!r}"
                    )
    if failures:
        raise LinkAuditError("\n".join(failures))
    return checked


def main() -> int:
    checked = audit()
    print(f"LOCAL_MARKDOWN_LINK_AUDIT_PASS links={checked}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        LinkAuditError,
        OSError,
        ReleaseTreeError,
        subprocess.CalledProcessError,
        UnicodeError,
    ) as exc:
        print(f"LOCAL_MARKDOWN_LINK_AUDIT_FAIL:\n{exc}")
        raise SystemExit(1) from exc
