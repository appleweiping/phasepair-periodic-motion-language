from __future__ import annotations

import hashlib
import os
import stat
import subprocess
import sys
import tarfile
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HEADER = "# sha256\\tbytes\\tgit_index_path; self excluded\n"


def _run(*arguments: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-B", "scripts/audit_release_artifact.py", *arguments],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _manifest(files: dict[str, bytes]) -> bytes:
    rows = []
    for name, raw in sorted(files.items(), key=lambda item: item[0].encode("utf-8")):
        rows.append(f"{hashlib.sha256(raw).hexdigest()}\t{len(raw)}\t{name}")
    return (HEADER + "\n".join(rows) + "\n").encode()


def test_wheel_and_manifest_bound_codeload_are_scanned(tmp_path: Path) -> None:
    wheel = tmp_path / "phaseset_core-0.2.0a1-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("phaseset_core/__init__.py", "VERSION = 'safe'\n")
        archive.writestr(
            "phaseset_core-0.2.0a1.dist-info/METADATA",
            "Name: phaseset-core\n",
        )
    clean_wheel = _run(str(wheel))
    assert clean_wheel.returncode == 0, clean_wheel.stdout + clean_wheel.stderr

    files = {
        ".gitignore": b"private/\nresults/private/\n",
        "README.md": b"PhaseSet public archive\n",
    }
    codeload = tmp_path / "codeload.zip"
    with zipfile.ZipFile(codeload, "w") as archive:
        for name, raw in files.items():
            archive.writestr(f"phaseset-abc/{name}", raw)
        archive.writestr(
            "phaseset-abc/RELEASE_FILES.sha256",
            _manifest(files),
        )
    clean_codeload = _run("--require-manifest", str(codeload))
    assert clean_codeload.returncode == 0, (
        clean_codeload.stdout + clean_codeload.stderr
    )

    with zipfile.ZipFile(codeload, "w") as archive:
        archive.writestr("phaseset-abc/.gitignore", files[".gitignore"])
        archive.writestr("phaseset-abc/README.md", b"tampered\n")
        archive.writestr(
            "phaseset-abc/RELEASE_FILES.sha256",
            _manifest(files),
        )
    rejected = _run("--require-manifest", str(codeload))
    assert rejected.returncode == 1
    assert "embedded manifest mismatch" in rejected.stderr


def test_archive_links_traversal_and_nested_endpoint_are_rejected(tmp_path: Path) -> None:
    traversal = tmp_path / "traversal.whl"
    with zipfile.ZipFile(traversal, "w") as archive:
        archive.writestr("../escape.txt", "unsafe\n")
    assert _run(str(traversal)).returncode == 1

    linked = tmp_path / "linked.tar.gz"
    payload = tmp_path / "payload.txt"
    payload.write_text("safe\n", encoding="utf-8")
    with tarfile.open(linked, "w:gz") as archive:
        archive.add(payload, arcname="release/payload.txt")
        info = tarfile.TarInfo("release/link.txt")
        info.type = tarfile.SYMTYPE
        info.linkname = "payload.txt"
        info.mode = stat.S_IFLNK | 0o777
        archive.addfile(info)
    assert _run(str(linked)).returncode == 1

    endpoint = tmp_path / "endpoint.whl"
    with zipfile.ZipFile(endpoint, "w") as archive:
        archive.writestr(
            "phaseset_core/private.txt",
            "198.51.100.9" + ":" + "33123\n",
        )
    rejected = _run(str(endpoint))
    assert rejected.returncode == 1
    assert "credential or endpoint pattern" in rejected.stderr


def test_zip_symlink_is_rejected(tmp_path: Path) -> None:
    linked = tmp_path / "linked.whl"
    with zipfile.ZipFile(linked, "w") as archive:
        info = zipfile.ZipInfo("phaseset_core/link.py")
        info.create_system = 3
        info.external_attr = (stat.S_IFLNK | 0o777) << 16
        archive.writestr(info, os.fsencode("target.py"))
    rejected = _run(str(linked))
    assert rejected.returncode == 1
    assert "non-regular ZIP member" in rejected.stderr
