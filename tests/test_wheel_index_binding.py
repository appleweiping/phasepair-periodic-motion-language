from __future__ import annotations

import base64
import hashlib
import shutil
import subprocess
import sys
from pathlib import Path
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ("release_tree.py", "verify_wheel_against_index.py")
PHASEPAIR = b"LEGACY = 'staged'\n"
PHASESET = b"__version__ = '0.2.0'\n"
MODULE = b"VALUE = 7\n"


def _git(root: Path, *arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )


def _run(root: Path, wheel: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "scripts/verify_wheel_against_index.py", str(wheel)],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )


def _repository(tmp_path: Path) -> Path:
    root = tmp_path / "checkout"
    scripts = root / "scripts"
    phasepair = root / "src" / "phasepair_core"
    phaseset = root / "src" / "phaseset_core"
    scripts.mkdir(parents=True)
    phasepair.mkdir(parents=True)
    phaseset.mkdir(parents=True)
    for name in SCRIPTS:
        shutil.copy2(ROOT / "scripts" / name, scripts / name)
    (root / "pyproject.toml").write_text(
        """[project]
name = "phaseset-core"
version = "0.2.0"
requires-python = ">=3.12"
dependencies = ["numpy>=1.26,<3"]

[project.optional-dependencies]
test = ["pytest==9.0.3"]

[project.scripts]
phaseset = "phaseset_core.cli:main"
""",
        encoding="utf-8",
    )
    (root / "RELEASE_FILES.sha256").write_text(
        "# sha256\\tbytes\\tgit_index_path; self excluded\n",
        encoding="utf-8",
    )
    (phasepair / "__init__.py").write_bytes(PHASEPAIR)
    (phaseset / "__init__.py").write_bytes(PHASESET)
    (phaseset / "module.py").write_bytes(MODULE)
    assert _git(root, "init", "--quiet").returncode == 0
    assert _git(root, "add", "--all").returncode == 0
    return root


def _wheel(
    root: Path,
    *,
    sources: dict[str, bytes] | None = None,
    filename_version: str = "0.2.0",
    metadata_version: str = "0.2.0",
    entrypoint: str = "phaseset_core.cli:main",
    requires_dist: tuple[str, ...] = (
        "numpy<3,>=1.26",
        'pytest==9.0.3; extra == "test"',
    ),
    extra_members: dict[str, bytes] | None = None,
    executable_member: str | None = None,
    record_mode: str = "valid",
) -> Path:
    wheel = root / "dist" / f"phaseset_core-{filename_version}-py3-none-any.whl"
    wheel.parent.mkdir(exist_ok=True)
    payloads = {
        "phasepair_core/__init__.py": PHASEPAIR,
        "phaseset_core/__init__.py": PHASESET,
        "phaseset_core/module.py": MODULE,
    }
    if sources is not None:
        payloads = sources
    dist_info = f"phaseset_core-{metadata_version}.dist-info"
    metadata = (
        "Metadata-Version: 2.4\n"
        "Name: phaseset-core\n"
        f"Version: {metadata_version}\n"
        "Requires-Python: >=3.12\n"
        "Provides-Extra: test\n"
        + "".join(f"Requires-Dist: {requirement}\n" for requirement in requires_dist)
        + "\n"
    ).encode()
    members = {
        **payloads,
        f"{dist_info}/METADATA": metadata,
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{dist_info}/entry_points.txt": (
            f"[console_scripts]\nphaseset = {entrypoint}\n".encode()
        ),
    }
    if extra_members is not None:
        members.update(extra_members)
    record_path = f"{dist_info}/RECORD"
    record_rows = []
    for name, raw in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        if not (record_mode == "missing-row" and name == "phaseset_core/module.py"):
            record_rows.append(f"{name},sha256={digest},{len(raw)}")
    if record_mode == "wrong-hash":
        record_rows[0] = record_rows[0].replace("sha256=", "sha256=AAAA", 1)
    if record_mode == "self-hash":
        record_rows.append(f"{record_path},sha256=AAAA,1")
    else:
        record_rows.append(f"{record_path},,")
    members[record_path] = ("\n".join(record_rows) + "\n").encode()
    with zipfile.ZipFile(wheel, "w") as archive:
        for name, raw in members.items():
            if name == executable_member:
                info = zipfile.ZipInfo(name)
                info.external_attr = 0o100755 << 16
                archive.writestr(info, raw)
            else:
                archive.writestr(name, raw)
    return wheel


def test_wheel_is_bound_to_index_bytes_not_unstaged_worktree(tmp_path: Path) -> None:
    root = _repository(tmp_path)
    staged_wheel = _wheel(root)
    clean = _run(root, staged_wheel)
    assert clean.returncode == 0, clean.stdout + clean.stderr
    assert "sources=3" in clean.stdout

    (root / "src" / "phaseset_core" / "module.py").write_bytes(b"VALUE = 999\n")
    still_staged = _run(root, staged_wheel)
    assert still_staged.returncode == 0, still_staged.stdout + still_staged.stderr

    worktree_wheel = _wheel(
        root,
        sources={
            "phasepair_core/__init__.py": PHASEPAIR,
            "phaseset_core/__init__.py": PHASESET,
            "phaseset_core/module.py": b"VALUE = 999\n",
        },
    )
    rejected = _run(root, worktree_wheel)
    assert rejected.returncode == 1
    assert "source bytes differ from staged index" in rejected.stderr


@pytest.mark.parametrize(
    ("case", "expected"),
    (
        ("missing", "source set differs from staged index"),
        ("extra", "source set differs from staged index"),
        ("metadata-version", "dist-info directory does not bind staged name/version"),
        ("filename-version", "wheel filename does not bind staged name/version"),
        ("entrypoint", "entry points differ from staged project"),
    ),
)
def test_wheel_rejects_source_and_identity_drift(
    tmp_path: Path, case: str, expected: str
) -> None:
    root = _repository(tmp_path)
    sources = {
        "phasepair_core/__init__.py": PHASEPAIR,
        "phaseset_core/__init__.py": PHASESET,
        "phaseset_core/module.py": MODULE,
    }
    arguments: dict[str, object] = {}
    if case == "missing":
        sources.pop("phaseset_core/module.py")
    elif case == "extra":
        sources["other_package/unstaged.py"] = b"UNSTAGED = True\n"
    elif case == "metadata-version":
        arguments["metadata_version"] = "9.9.9"
    elif case == "filename-version":
        arguments["filename_version"] = "9.9.9"
    elif case == "entrypoint":
        arguments["entrypoint"] = "phaseset_core.cli:wrong"
    wheel = _wheel(root, sources=sources, **arguments)  # type: ignore[arg-type]
    rejected = _run(root, wheel)
    assert rejected.returncode == 1
    assert expected in rejected.stderr


@pytest.mark.parametrize(
    ("arguments", "expected"),
    (
        (
            {"extra_members": {"bootstrap.pth": b"import os\n"}},
            "member set is not allowlisted",
        ),
        (
            {"executable_member": "phaseset_core/module.py"},
            "executable wheel member is forbidden",
        ),
        (
            {"requires_dist": ("numpy<3,>=1.26", 'evilpkg==1; extra == "test"')},
            "Requires-Dist differs from staged project",
        ),
        ({"record_mode": "missing-row"}, "RECORD coverage differs"),
        ({"record_mode": "wrong-hash"}, "RECORD hash/size mismatch"),
        ({"record_mode": "self-hash"}, "RECORD self-row must have empty"),
    ),
)
def test_wheel_rejects_executable_unexpected_dependency_and_record_attacks(
    tmp_path: Path, arguments: dict[str, object], expected: str
) -> None:
    root = _repository(tmp_path)
    rejected = _run(root, _wheel(root, **arguments))  # type: ignore[arg-type]
    assert rejected.returncode == 1
    assert expected in rejected.stderr
