from __future__ import annotations

import base64
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from typing import Any
import zipfile
import zlib


ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ID = 1_345_199_252
FULL_NAME = "appleweiping/phaseset-multiperson-motion-language"
TAG = "v0.2.0"
SELF_NAME = "PHASESET_v0.2.0_ASSET_INVENTORY.json"


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_canonical_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    return completed.stdout.strip()


def _write_bound_wheel(path: Path) -> None:
    sources = {
        "phasepair_core/__init__.py": b"LEGACY = True\n",
        "phaseset_core/__init__.py": b"VERSION = '0.2.0'\n",
    }
    dist_info = "phaseset_core-0.2.0.dist-info"
    members = {
        **sources,
        f"{dist_info}/METADATA": (
            b"Metadata-Version: 2.4\nName: phaseset-core\nVersion: 0.2.0\n\n"
        ),
        f"{dist_info}/WHEEL": (
            b"Wheel-Version: 1.0\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
    }
    rows = []
    for name, raw in sorted(members.items()):
        digest = base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode()
        rows.append(f"{name},sha256={digest},{len(raw)}")
    record = f"{dist_info}/RECORD"
    rows.append(f"{record},,")
    members[record] = ("\n".join(rows) + "\n").encode()
    with zipfile.ZipFile(path, "w") as archive:
        for name, raw in members.items():
            archive.writestr(name, raw)


def _tag_checkout(tmp_path: Path, paper: bytes) -> tuple[Path, dict[str, str]]:
    checkout = tmp_path / "tag-checkout"
    (checkout / "scripts").mkdir(parents=True)
    (checkout / "src" / "phasepair_core").mkdir(parents=True)
    (checkout / "src" / "phaseset_core").mkdir(parents=True)
    (checkout / "paper").mkdir()
    for name in ("release_tree.py", "verify_wheel_against_index.py"):
        (checkout / "scripts" / name).write_bytes((ROOT / "scripts" / name).read_bytes())
    (checkout / "pyproject.toml").write_text(
        '[project]\nname = "phaseset-core"\nversion = "0.2.0"\n',
        encoding="utf-8",
    )
    (checkout / "RELEASE_FILES.sha256").write_text(
        "# sha256\\tbytes\\tgit_index_path; self excluded\n",
        encoding="utf-8",
    )
    (checkout / "src" / "phasepair_core" / "__init__.py").write_bytes(b"LEGACY = True\n")
    (checkout / "src" / "phaseset_core" / "__init__.py").write_bytes(
        b"VERSION = '0.2.0'\n"
    )
    (checkout / "paper" / "main.pdf").write_bytes(paper)
    _git(checkout, "init", "--quiet")
    _git(checkout, "config", "user.name", "PhaseSet Test")
    _git(checkout, "config", "user.email", "test@example.invalid")
    _git(checkout, "add", "--all")
    _git(checkout, "commit", "--quiet", "-m", "release fixture")
    _git(checkout, "tag", "-a", TAG, "-m", "release fixture tag")
    return checkout, {
        "tag": _git(checkout, "rev-parse", f"refs/tags/{TAG}"),
        "commit": _git(checkout, "rev-parse", "HEAD"),
        "tree": _git(checkout, "rev-parse", "HEAD^{tree}"),
    }


def _api_asset(asset_id: int, path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    return {
        "id": asset_id,
        "name": path.name,
        "size": len(raw),
        "digest": f"sha256:{hashlib.sha256(raw).hexdigest()}",
        "state": "uploaded",
    }


def _fixture(tmp_path: Path) -> tuple[dict[str, Path], Path]:
    assets = tmp_path / "assets"
    evidence = tmp_path / "evidence"
    assets.mkdir()
    evidence.mkdir()
    wheel = assets / "phaseset_core-0.2.0-py3-none-any.whl"
    paper = assets / "PhaseSet-v0.2.0-paper.pdf"
    _write_bound_wheel(wheel)
    paper.write_bytes(b"%PDF-1.7\n%%EOF\n")
    checkout, identity = _tag_checkout(tmp_path, paper.read_bytes())
    paths = {
        "repository": evidence / "repository.json",
        "release": evidence / "release.json",
        "tag_ref": evidence / "tag-ref.json",
        "tag_object": evidence / "tag-object.json",
        "commit": evidence / "commit.json",
        "expected": evidence / "expected-assets.json",
        "expected_sha": evidence / "expected-assets.trusted.sha256",
        "tag_checkout": checkout,
    }
    _write_json(
        paths["repository"],
        {
            "id": REPOSITORY_ID,
            "full_name": FULL_NAME,
            "private": False,
            "visibility": "public",
        },
    )
    _write_json(
        paths["tag_ref"],
        {"ref": f"refs/tags/{TAG}", "object": {"type": "tag", "sha": identity["tag"]}},
    )
    _write_json(
        paths["tag_object"],
        {
            "tag": TAG,
            "sha": identity["tag"],
            "object": {"type": "commit", "sha": identity["commit"]},
        },
    )
    _write_json(paths["commit"], {"sha": identity["commit"], "tree": {"sha": identity["tree"]}})
    rows = []
    for asset, kind, source in (
        (paper, "TAG_TREE_FILE", "paper/main.pdf"),
        (wheel, "WHEEL_FROM_TAG_INDEX", None),
    ):
        raw = asset.read_bytes()
        rows.append(
            {
                "kind": kind,
                "name": asset.name,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
                "source_path": source,
            }
        )
    _write_canonical_json(
        paths["expected"],
        {
            "artifacts": sorted(rows, key=lambda row: row["name"]),
            "release": {
                "tag_name": TAG,
                "tag_object_sha": identity["tag"],
                "tag_peeled_commit_sha": identity["commit"],
                "tree_sha": identity["tree"],
            },
            "repository": {"full_name": FULL_NAME, "id": REPOSITORY_ID},
            "schema": "phaseset-expected-release-assets-v1",
        },
    )
    paths["expected_sha"].write_text(
        hashlib.sha256(paths["expected"].read_bytes()).hexdigest(),
        encoding="ascii",
    )
    _release(paths["release"], assets)
    return paths, assets


def _release(path: Path, assets: Path) -> None:
    ids = {
        "PhaseSet-v0.2.0-paper.pdf": 101,
        "phaseset_core-0.2.0-py3-none-any.whl": 102,
        SELF_NAME: 103,
    }
    rows = [
        _api_asset(ids.get(asset.name, 1_000 + index), asset)
        for index, asset in enumerate(sorted(assets.iterdir(), key=lambda item: item.name))
    ]
    _write_json(
        path,
        {
            "id": 376_999_999,
            "tag_name": TAG,
            "draft": False,
            "prerelease": False,
            "assets": rows,
        },
    )


def _command(command: str, paths: dict[str, Path], assets: Path, receipt: Path) -> list[str]:
    arguments = [
        sys.executable,
        "-B",
        "scripts/release_asset_inventory.py",
        command,
        "--expected-repository-id",
        str(REPOSITORY_ID),
        "--expected-repository",
        FULL_NAME,
        "--expected-tag",
        TAG,
        "--repository-json",
        str(paths["repository"]),
        "--release-json",
        str(paths["release"]),
        "--tag-ref-json",
        str(paths["tag_ref"]),
        "--tag-object-json",
        str(paths["tag_object"]),
        "--commit-json",
        str(paths["commit"]),
        "--asset-dir",
        str(assets),
        "--expected-assets-json",
        str(paths["expected"]),
        "--expected-assets-sha256",
        paths["expected_sha"].read_text(encoding="ascii").strip(),
        "--tag-checkout",
        str(paths["tag_checkout"]),
    ]
    if command == "build":
        arguments.extend(("--self-asset-name", SELF_NAME, "--output", str(receipt)))
    else:
        arguments.extend(("--receipt", str(receipt)))
    return arguments


def _run(command: str, paths: dict[str, Path], assets: Path, receipt: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        _command(command, paths, assets, receipt),
        cwd=ROOT,
        check=False,
        text=True,
        capture_output=True,
    )


def _built(tmp_path: Path) -> tuple[dict[str, Path], Path, Path]:
    paths, assets = _fixture(tmp_path)
    receipt = assets / SELF_NAME
    built = _run("build", paths, assets, receipt)
    assert built.returncode == 0, built.stdout + built.stderr
    _release(paths["release"], assets)
    return paths, assets, receipt


def test_inventory_cross_binds_github_identity_and_downloaded_assets(tmp_path: Path) -> None:
    paths, assets, receipt = _built(tmp_path)
    raw = receipt.read_bytes()
    assert raw.endswith(b"\n") and raw.count(b"\n") == 1
    decoded = json.loads(raw)
    tag_ref = json.loads(paths["tag_ref"].read_text(encoding="utf-8"))["object"]["sha"]
    commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
    assert decoded["repository"] == {"full_name": FULL_NAME, "id": REPOSITORY_ID}
    assert decoded["release"]["tag_object_sha"] == tag_ref
    assert decoded["release"]["tag_peeled_commit_sha"] == commit["sha"]
    assert decoded["release"]["tree_sha"] == commit["tree"]["sha"]
    assert decoded["release"]["expected_assets_manifest_sha256"] == hashlib.sha256(
        paths["expected"].read_bytes()
    ).hexdigest()
    assert [row["name"] for row in decoded["artifacts"]] == sorted(
        path.name for path in assets.iterdir() if path.name != SELF_NAME
    )

    verified = _run("verify", paths, assets, receipt)
    assert verified.returncode == 0, verified.stdout + verified.stderr
    assert "assets=3" in verified.stdout


def test_inventory_rejects_github_identity_and_tree_drift(tmp_path: Path) -> None:
    paths, assets, receipt = _built(tmp_path)
    repository = json.loads(paths["repository"].read_text(encoding="utf-8"))
    repository["id"] += 1
    _write_json(paths["repository"], repository)
    rejected_repository = _run("verify", paths, assets, receipt)
    assert rejected_repository.returncode == 1
    assert "differs from trusted expectation" in rejected_repository.stderr

    repository["id"] = REPOSITORY_ID
    _write_json(paths["repository"], repository)
    commit = json.loads(paths["commit"].read_text(encoding="utf-8"))
    commit["tree"]["sha"] = "d" * 40
    _write_json(paths["commit"], commit)
    rejected_tree = _run("verify", paths, assets, receipt)
    assert rejected_tree.returncode == 1
    assert "manifest identity differs from GitHub tag evidence" in rejected_tree.stderr


def test_receipt_cannot_self_authenticate_tampered_payload(tmp_path: Path) -> None:
    paths, assets, receipt = _built(tmp_path)
    payload = assets / "PhaseSet-v0.2.0-paper.pdf"
    payload.write_bytes(b"%PDF-1.7\n% unrelated but scan-safe payload\n%%EOF\n")

    decoded = json.loads(receipt.read_text(encoding="utf-8"))
    for row in decoded["artifacts"]:
        if row["name"] == payload.name:
            row["size"] = payload.stat().st_size
            row["sha256"] = hashlib.sha256(payload.read_bytes()).hexdigest()
    receipt.write_text(
        json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    release = json.loads(paths["release"].read_text(encoding="utf-8"))
    for asset in release["assets"]:
        if asset["name"] == SELF_NAME:
            current = _api_asset(asset["id"], receipt)
            asset.update(current)
    _write_json(paths["release"], release)

    rejected = _run("verify", paths, assets, receipt)
    assert rejected.returncode == 1
    assert "frozen prepublication intent" in rejected.stderr


def test_inventory_requires_pre_and_post_self_upload_responses(tmp_path: Path) -> None:
    paths, assets, receipt = _built(tmp_path)
    rebuild = tmp_path / SELF_NAME
    rejected_build = _run("build", paths, assets, rebuild)
    assert rejected_build.returncode == 1
    assert "pre-self-upload" in rejected_build.stderr

    release = json.loads(paths["release"].read_text(encoding="utf-8"))
    release["assets"] = [row for row in release["assets"] if row["name"] != SELF_NAME]
    _write_json(paths["release"], release)
    receipt_only = tmp_path / "receipt-copy.json"
    receipt_only.write_bytes(receipt.read_bytes())
    receipt.unlink()
    rejected_verify = _run("verify", paths, assets, receipt_only)
    assert rejected_verify.returncode == 1
    assert "lacks the receipt asset itself" in rejected_verify.stderr


def test_inventory_scans_decoded_asset_payloads_and_rejects_unknown_types(
    tmp_path: Path,
) -> None:
    paths, assets = _fixture(tmp_path)
    secret = b"C:" + b"\\" + b"private" + b"\\capture.bin"
    compressed = zlib.compress(secret)
    (assets / "PhaseSet-v0.2.0-paper.pdf").write_bytes(
        b"%PDF-1.7\n1 0 obj\n<< /Length "
        + str(len(compressed)).encode()
        + b" /Filter /FlateDecode >>\nstream\n"
        + compressed
        + b"\nendstream\nendobj\n%%EOF\n"
    )
    _release(paths["release"], assets)
    receipt = assets / SELF_NAME
    rejected_secret = _run("build", paths, assets, receipt)
    assert rejected_secret.returncode == 1
    assert "private/local locator" in rejected_secret.stderr

    (assets / "PhaseSet-v0.2.0-paper.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
    (assets / "opaque.bin").write_bytes(b"not audited\n")
    _release(paths["release"], assets)
    rejected_type = _run("build", paths, assets, receipt)
    assert rejected_type.returncode == 1
    assert "unsupported release artifact type" in rejected_type.stderr


def test_frozen_prepublication_manifest_rejects_scan_safe_unrelated_wheel(
    tmp_path: Path,
) -> None:
    paths, assets = _fixture(tmp_path)
    wheel = assets / "phaseset_core-0.2.0-py3-none-any.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("unrelated_package/__init__.py", b"SAFE = True\n")
    _release(paths["release"], assets)
    rejected = _run("build", paths, assets, assets / SELF_NAME)
    assert rejected.returncode == 1
    assert "frozen prepublication intent" in rejected.stderr
