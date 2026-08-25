from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _run(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "scripts/audit_all_refs.py", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )


def _git(root: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr


def test_all_refs_audit_rejects_a_removed_historical_endpoint(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    for name in ("audit_all_refs.py", "public_release_audit.py", "release_tree.py"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    _git(checkout, "init", "--quiet")
    _git(checkout, "config", "user.name", "PhaseSet Test")
    _git(checkout, "config", "user.email", "test@example.invalid")

    marker = checkout / "marker.txt"
    marker.write_text("public-safe\n", encoding="utf-8")
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "safe")
    clean = _run(checkout)
    assert clean.returncode == 0, clean.stdout + clean.stderr

    marker.write_text("198.51.100.2" + ":" + "33123\n", encoding="utf-8")
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "unsafe historical blob")
    marker.write_text("public-safe again\n", encoding="utf-8")
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "remove unsafe marker")

    rejected = _run(checkout)
    assert rejected.returncode == 1
    assert "credential or endpoint pattern" in rejected.stdout + rejected.stderr


def test_all_refs_audit_checks_every_historical_path_binding_for_shared_blob(
    tmp_path: Path,
) -> None:
    checkout = tmp_path / "checkout"
    scripts = checkout / "scripts"
    scripts.mkdir(parents=True)
    for name in ("audit_all_refs.py", "public_release_audit.py", "release_tree.py"):
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    _git(checkout, "init", "--quiet")
    _git(checkout, "config", "user.name", "PhaseSet Test")
    _git(checkout, "config", "user.email", "test@example.invalid")

    safe = checkout / "safe.txt"
    safe.write_text("identical public bytes\n", encoding="utf-8")
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "safe blob path")

    forbidden = checkout / "private" / "record.txt"
    forbidden.parent.mkdir()
    forbidden.write_bytes(safe.read_bytes())
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "same blob at forbidden path")
    forbidden.unlink()
    forbidden.parent.rmdir()
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "remove forbidden binding")

    rejected = _run(checkout)
    assert rejected.returncode == 1
    assert "forbidden tracked path" in rejected.stdout + rejected.stderr
