"""Verify that repository-local Markdown links resolve inside the public tree."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[1]
LINK_PATTERN = re.compile(r"(?<!!)\[[^\]]*\]\(([^)]+)\)")


class LinkAuditError(RuntimeError):
    """A local Markdown link is malformed, unsafe, or missing."""


def _tracked_markdown() -> tuple[Path, ...]:
    completed = subprocess.run(
        ["git", "ls-files", "--cached", "-z", "--", "*.md"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.PIPE,
    )
    paths = []
    for raw in completed.stdout.split(b"\0"):
        if raw:
            paths.append(ROOT.joinpath(*PurePosixPath(raw.decode("utf-8")).parts))
    return tuple(paths)


def _link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1]
    elif " " in target:
        target = target.split(" ", 1)[0]
    return unquote(target)


def audit() -> int:
    checked = 0
    failures: list[str] = []
    for document in _tracked_markdown():
        text = document.read_text(encoding="utf-8", errors="strict")
        for line_number, line in enumerate(text.splitlines(), start=1):
            for match in LINK_PATTERN.finditer(line):
                target = _link_target(match.group(1))
                parsed = urlsplit(target)
                if parsed.scheme or parsed.netloc or not parsed.path:
                    continue
                relative = PurePosixPath(parsed.path)
                if relative.is_absolute() or ".." in relative.parts and not (
                    document.parent / Path(*relative.parts)
                ).resolve().is_relative_to(ROOT):
                    failures.append(
                        f"{document.relative_to(ROOT)}:{line_number}: unsafe {target!r}"
                    )
                    continue
                resolved = (document.parent / Path(*relative.parts)).resolve()
                checked += 1
                if not resolved.is_relative_to(ROOT) or not resolved.exists():
                    failures.append(
                        f"{document.relative_to(ROOT)}:{line_number}: missing {target!r}"
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
    except (LinkAuditError, OSError, subprocess.CalledProcessError, UnicodeError) as exc:
        print(f"LOCAL_MARKDOWN_LINK_AUDIT_FAIL:\n{exc}")
        raise SystemExit(1) from exc
