"""Build and verify a canonical, GitHub-evidenced release asset inventory.

The canonical receipt is not a trust root.  Building and verification both
require separately captured GitHub REST responses, and post-release
verification additionally requires the complete downloaded asset directory.
The receipt excludes its own digest to avoid self-reference; its bytes are
instead authenticated by the refreshed GitHub asset response and the local
download during ``verify``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from pathlib import PurePosixPath
import re
import subprocess
import sys
from typing import Any

sys.dont_write_bytecode = True

from audit_release_artifact import (  # noqa: E402
    ReleaseArtifactAuditError,
    audit_artifact,
)
from public_release_audit import PublicReleaseAuditError  # noqa: E402
from release_tree import clean_git_environment  # noqa: E402


SCHEMA = "phaseset-release-asset-inventory-v2"
EXPECTED_SCHEMA = "phaseset-expected-release-assets-v1"
SHA256 = re.compile(r"[0-9a-f]{64}")
GIT_OBJECT = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})")
FULL_NAME = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
TAG_NAME = re.compile(r"v[0-9A-Za-z][0-9A-Za-z._+-]*")


class ReleaseAssetInventoryError(RuntimeError):
    """The receipt, GitHub evidence, or downloaded assets do not cross-bind."""


@dataclass(frozen=True)
class GitHubAsset:
    asset_id: int
    name: str
    size: int
    sha256: str


@dataclass(frozen=True)
class GitHubEvidence:
    repository_id: int
    repository_full_name: str
    release_id: int
    tag_name: str
    tag_object_sha: str
    peeled_commit_sha: str
    tree_sha: str
    prerelease: bool
    assets: dict[str, GitHubAsset]


@dataclass(frozen=True)
class ExpectedAsset:
    name: str
    size: int
    sha256: str
    kind: str
    source_path: PurePosixPath | None


@dataclass(frozen=True)
class ExpectedAssets:
    sha256: str
    assets: dict[str, ExpectedAsset]


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    decoded: dict[str, Any] = {}
    for key, value in pairs:
        if key in decoded:
            raise ReleaseAssetInventoryError(f"duplicate JSON key: {key!r}")
        decoded[key] = value
    return decoded


def _read_json(path: Path, label: str) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise ReleaseAssetInventoryError(f"{label} is not a regular file")
    raw = path.read_bytes()
    try:
        decoded = json.loads(
            raw.decode("utf-8", "strict"),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=lambda value: (_ for _ in ()).throw(
                ReleaseAssetInventoryError(f"non-finite JSON value: {value}")
            ),
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseAssetInventoryError(f"{label} is not strict UTF-8 JSON") from exc
    if not isinstance(decoded, dict):
        raise ReleaseAssetInventoryError(f"{label} must be a JSON object")
    return decoded, raw


def _exact_keys(value: dict[str, Any], expected: set[str], label: str) -> None:
    if set(value) != expected:
        raise ReleaseAssetInventoryError(
            f"{label} keys differ: actual={sorted(value)!r} expected={sorted(expected)!r}"
        )


def _integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ReleaseAssetInventoryError(f"{label} must be a positive integer")
    return value


def _boolean(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise ReleaseAssetInventoryError(f"{label} must be boolean")
    return value


def _string(value: Any, label: str) -> str:
    if not isinstance(value, str) or not value or any(
        ord(character) < 32
        or 127 <= ord(character) <= 159
        or 0xD800 <= ord(character) <= 0xDFFF
        for character in value
    ):
        raise ReleaseAssetInventoryError(f"{label} must be a non-empty printable string")
    return value


def _asset_name(value: Any, label: str) -> str:
    decoded = _string(value, label)
    if (
        decoded in {".", ".."}
        or "/" in decoded
        or "\\" in decoded
        or ":" in decoded
        or Path(decoded).name != decoded
    ):
        raise ReleaseAssetInventoryError(f"{label} is not a safe basename")
    return decoded


def _sha256(value: Any, label: str) -> str:
    decoded = _string(value, label)
    if SHA256.fullmatch(decoded) is None:
        raise ReleaseAssetInventoryError(f"{label} must be lowercase SHA-256")
    return decoded


def _git_object(value: Any, label: str) -> str:
    decoded = _string(value, label)
    if GIT_OBJECT.fullmatch(decoded) is None:
        raise ReleaseAssetInventoryError(f"{label} must be a canonical Git object ID")
    return decoded


def _evidence(
    repository_json: Path,
    release_json: Path,
    tag_ref_json: Path,
    tag_object_json: Path,
    commit_json: Path,
    *,
    expected_repository_id: int,
    expected_repository: str,
    expected_tag: str,
) -> GitHubEvidence:
    repository, _ = _read_json(repository_json, "GitHub repository response")
    release, _ = _read_json(release_json, "GitHub release response")
    tag_ref, _ = _read_json(tag_ref_json, "GitHub tag-ref response")
    tag_object, _ = _read_json(tag_object_json, "GitHub tag-object response")
    commit, _ = _read_json(commit_json, "GitHub commit response")

    repository_id = _integer(repository.get("id"), "repository.id")
    full_name = _string(repository.get("full_name"), "repository.full_name")
    if FULL_NAME.fullmatch(full_name) is None:
        raise ReleaseAssetInventoryError("repository.full_name is malformed")
    if repository.get("private") is not False or repository.get("visibility") != "public":
        raise ReleaseAssetInventoryError("GitHub repository response is not public")
    if repository_id != expected_repository_id or full_name != expected_repository:
        raise ReleaseAssetInventoryError("GitHub repository response differs from trusted expectation")

    release_id = _integer(release.get("id"), "release.id")
    tag_name = _string(release.get("tag_name"), "release.tag_name")
    if TAG_NAME.fullmatch(tag_name) is None:
        raise ReleaseAssetInventoryError("release.tag_name is not a stable PhaseSet tag name")
    if tag_name != expected_tag:
        raise ReleaseAssetInventoryError("GitHub release tag differs from trusted expectation")
    if release.get("draft") is not False:
        raise ReleaseAssetInventoryError("release must be published, not a draft")
    prerelease = _boolean(release.get("prerelease"), "release.prerelease")

    expected_ref = f"refs/tags/{tag_name}"
    if tag_ref.get("ref") != expected_ref:
        raise ReleaseAssetInventoryError("tag-ref response does not match release.tag_name")
    ref_object = tag_ref.get("object")
    if not isinstance(ref_object, dict) or ref_object.get("type") != "tag":
        raise ReleaseAssetInventoryError("release tag must be an annotated Git tag")
    tag_sha = _git_object(ref_object.get("sha"), "tag-ref object SHA")

    if tag_object.get("tag") != tag_name:
        raise ReleaseAssetInventoryError("tag-object response name differs from release tag")
    if _git_object(tag_object.get("sha"), "tag-object SHA") != tag_sha:
        raise ReleaseAssetInventoryError("tag-object response differs from tag ref")
    peeled = tag_object.get("object")
    if not isinstance(peeled, dict) or peeled.get("type") != "commit":
        raise ReleaseAssetInventoryError("annotated tag does not peel to a commit")
    peeled_sha = _git_object(peeled.get("sha"), "peeled commit SHA")

    if _git_object(commit.get("sha"), "commit SHA") != peeled_sha:
        raise ReleaseAssetInventoryError("commit response differs from peeled tag commit")
    tree = commit.get("tree")
    if not isinstance(tree, dict):
        raise ReleaseAssetInventoryError("commit response lacks a tree object")
    tree_sha = _git_object(tree.get("sha"), "commit tree SHA")

    raw_assets = release.get("assets")
    if not isinstance(raw_assets, list):
        raise ReleaseAssetInventoryError("release.assets must be a list")
    assets: dict[str, GitHubAsset] = {}
    seen_ids: set[int] = set()
    for index, raw_asset in enumerate(raw_assets):
        if not isinstance(raw_asset, dict):
            raise ReleaseAssetInventoryError(f"release.assets[{index}] must be an object")
        asset_id = _integer(raw_asset.get("id"), f"release.assets[{index}].id")
        name = _asset_name(raw_asset.get("name"), f"release.assets[{index}].name")
        size = raw_asset.get("size")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseAssetInventoryError(f"release.assets[{index}].size is invalid")
        digest = _string(raw_asset.get("digest"), f"release.assets[{index}].digest")
        if not digest.startswith("sha256:"):
            raise ReleaseAssetInventoryError(f"release.assets[{index}] lacks a SHA-256 digest")
        digest_value = _sha256(digest.removeprefix("sha256:"), f"release.assets[{index}].digest")
        if raw_asset.get("state") != "uploaded":
            raise ReleaseAssetInventoryError(f"release.assets[{index}] is not uploaded")
        if name in assets or asset_id in seen_ids:
            raise ReleaseAssetInventoryError("GitHub release response has duplicate assets")
        assets[name] = GitHubAsset(asset_id, name, size, digest_value)
        seen_ids.add(asset_id)

    return GitHubEvidence(
        repository_id,
        full_name,
        release_id,
        tag_name,
        tag_sha,
        peeled_sha,
        tree_sha,
        prerelease,
        assets,
    )


def _asset_directory(path: Path, expected_names: set[str]) -> dict[str, tuple[int, str, bytes]]:
    if not path.is_dir() or path.is_symlink():
        raise ReleaseAssetInventoryError("asset directory is not a regular directory")
    entries = tuple(path.iterdir())
    actual_names: set[str] = set()
    records: dict[str, tuple[int, str, bytes]] = {}
    for entry in entries:
        name = _asset_name(entry.name, "downloaded asset name")
        if not entry.is_file() or entry.is_symlink():
            raise ReleaseAssetInventoryError(f"downloaded asset is not a regular file: {name!r}")
        if name in actual_names:
            raise ReleaseAssetInventoryError(f"duplicate downloaded asset: {name!r}")
        raw = entry.read_bytes()
        records[name] = (len(raw), hashlib.sha256(raw).hexdigest(), raw)
        actual_names.add(name)
    if actual_names != expected_names:
        raise ReleaseAssetInventoryError(
            "downloaded asset set differs from GitHub response: "
            f"missing={sorted(expected_names-actual_names)!r} "
            f"extra={sorted(actual_names-expected_names)!r}"
        )
    return records


def _audit_downloaded_assets(asset_dir: Path, names: set[str]) -> None:
    for name in sorted(names, key=lambda value: value.encode("utf-8")):
        try:
            audit_artifact(asset_dir / name)
        except (OSError, PublicReleaseAuditError, ReleaseArtifactAuditError) as exc:
            raise ReleaseAssetInventoryError(
                f"downloaded release asset failed public-safety audit: {name!r}: {exc}"
            ) from exc


def _source_path(value: Any, label: str) -> PurePosixPath:
    decoded = _string(value, label)
    path = PurePosixPath(decoded)
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != decoded
        or any(character in decoded for character in "\0\t\n\r\\:")
    ):
        raise ReleaseAssetInventoryError(f"{label} is not a canonical tag-tree path")
    return path


def _expected_assets(
    path: Path,
    trusted_sha256: str,
    evidence: GitHubEvidence,
) -> ExpectedAssets:
    trusted_digest = _sha256(trusted_sha256, "trusted expected-assets manifest SHA-256")
    decoded, raw = _read_json(path, "expected release assets manifest")
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != trusted_digest:
        raise ReleaseAssetInventoryError(
            "expected release assets manifest differs from independently trusted SHA-256"
        )
    if raw != _canonical_bytes(decoded):
        raise ReleaseAssetInventoryError("expected release assets manifest is not canonical JSON")
    _exact_keys(decoded, {"artifacts", "release", "repository", "schema"}, "expected manifest")
    if decoded["schema"] != EXPECTED_SCHEMA:
        raise ReleaseAssetInventoryError("expected release assets manifest schema is unsupported")
    repository = decoded["repository"]
    release = decoded["release"]
    artifacts = decoded["artifacts"]
    if not isinstance(repository, dict) or not isinstance(release, dict) or not isinstance(
        artifacts, list
    ):
        raise ReleaseAssetInventoryError("expected release assets manifest sections are malformed")
    expected_repository = {
        "full_name": evidence.repository_full_name,
        "id": evidence.repository_id,
    }
    expected_release = {
        "tag_name": evidence.tag_name,
        "tag_object_sha": evidence.tag_object_sha,
        "tag_peeled_commit_sha": evidence.peeled_commit_sha,
        "tree_sha": evidence.tree_sha,
    }
    if repository != expected_repository or release != expected_release:
        raise ReleaseAssetInventoryError(
            "expected release assets manifest identity differs from GitHub tag evidence"
        )
    parsed: dict[str, ExpectedAsset] = {}
    for index, row in enumerate(artifacts):
        if not isinstance(row, dict):
            raise ReleaseAssetInventoryError(f"expected artifacts[{index}] is not an object")
        _exact_keys(row, {"kind", "name", "sha256", "size", "source_path"}, "expected row")
        name = _asset_name(row["name"], f"expected artifacts[{index}].name")
        digest = _sha256(row["sha256"], f"expected artifacts[{index}].sha256")
        size = row["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseAssetInventoryError(f"expected artifacts[{index}].size is invalid")
        kind = _string(row["kind"], f"expected artifacts[{index}].kind")
        if kind == "WHEEL_FROM_TAG_INDEX":
            if not name.casefold().endswith(".whl") or row["source_path"] is not None:
                raise ReleaseAssetInventoryError("wheel expected row is malformed")
            source = None
        elif kind == "TAG_TREE_FILE":
            source = _source_path(row["source_path"], f"expected artifacts[{index}].source_path")
        elif kind == "PREPUBLICATION_BUILD":
            if row["source_path"] is not None:
                raise ReleaseAssetInventoryError("prepublication-build source_path must be null")
            source = None
        else:
            raise ReleaseAssetInventoryError(f"unsupported expected asset kind: {kind!r}")
        if name in parsed:
            raise ReleaseAssetInventoryError("expected release assets contain duplicate names")
        parsed[name] = ExpectedAsset(name, size, digest, kind, source)
    if list(parsed) != sorted(parsed, key=lambda value: value.encode("utf-8")):
        raise ReleaseAssetInventoryError("expected release assets are not in canonical name order")
    wheels = [row for row in parsed.values() if row.kind == "WHEEL_FROM_TAG_INDEX"]
    paper_rows = [
        row
        for row in parsed.values()
        if row.source_path == PurePosixPath("paper/main.pdf")
    ]
    if len(wheels) != 1 or len(paper_rows) != 1 or not paper_rows[0].name.casefold().endswith(
        ".pdf"
    ):
        raise ReleaseAssetInventoryError(
            "expected release assets require exactly one tag-index wheel and paper/main.pdf"
        )
    return ExpectedAssets(actual_digest, parsed)


def _git(checkout: Path, *arguments: str) -> bytes:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=checkout,
        env=clean_git_environment(),
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise ReleaseAssetInventoryError("tag checkout Git verification failed")
    return completed.stdout


def _verify_intended_assets(
    *,
    checkout: Path,
    evidence: GitHubEvidence,
    expected: ExpectedAssets,
    asset_dir: Path,
    local: dict[str, tuple[int, str, bytes]],
) -> None:
    if not checkout.is_dir() or checkout.is_symlink():
        raise ReleaseAssetInventoryError("tag checkout is not a regular directory")
    head = _git(checkout, "rev-parse", "HEAD").decode("ascii", "strict").strip()
    tree = _git(checkout, "rev-parse", "HEAD^{tree}").decode("ascii", "strict").strip()
    tag_object = _git(checkout, "rev-parse", f"refs/tags/{evidence.tag_name}").decode(
        "ascii", "strict"
    ).strip()
    tag_commit = _git(checkout, "rev-parse", f"refs/tags/{evidence.tag_name}^{{}}").decode(
        "ascii", "strict"
    ).strip()
    if (
        head != evidence.peeled_commit_sha
        or tree != evidence.tree_sha
        or tag_object != evidence.tag_object_sha
        or tag_commit != evidence.peeled_commit_sha
    ):
        raise ReleaseAssetInventoryError("tag checkout identity differs from GitHub evidence")
    if subprocess.run(
        ["git", "diff", "--quiet", "--exit-code"],
        cwd=checkout,
        env=clean_git_environment(),
        check=False,
    ).returncode != 0 or subprocess.run(
        ["git", "diff", "--cached", "--quiet", "--exit-code"],
        cwd=checkout,
        env=clean_git_environment(),
        check=False,
    ).returncode != 0:
        raise ReleaseAssetInventoryError("tag checkout has tracked worktree or index drift")
    if set(local) != set(expected.assets):
        raise ReleaseAssetInventoryError("downloaded payload set differs from frozen expected assets")
    for name, intended in expected.assets.items():
        size, digest, payload = local[name]
        if (size, digest) != (intended.size, intended.sha256):
            raise ReleaseAssetInventoryError(
                f"downloaded asset differs from frozen prepublication intent: {name!r}"
            )
        if intended.kind == "TAG_TREE_FILE":
            if intended.source_path is None:
                raise ReleaseAssetInventoryError("tag-tree expected asset lacks source path")
            source = _git(checkout, "cat-file", "blob", f"HEAD:{intended.source_path.as_posix()}")
            if source != payload:
                raise ReleaseAssetInventoryError(
                    f"release asset differs from tag-tree source: {name!r}"
                )
        elif intended.kind == "WHEEL_FROM_TAG_INDEX":
            verifier = checkout / "scripts" / "verify_wheel_against_index.py"
            completed = subprocess.run(
                [sys.executable, "-B", str(verifier), str(asset_dir / name)],
                cwd=checkout,
                env=clean_git_environment(),
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if completed.returncode != 0 or "WHEEL_INDEX_BINDING_PASS" not in completed.stdout:
                raise ReleaseAssetInventoryError(
                    f"downloaded wheel is not bound to the exact tag index: {name!r}"
                )


def _artifact_row(asset: GitHubAsset) -> dict[str, Any]:
    return {
        "github_asset_id": asset.asset_id,
        "name": asset.name,
        "sha256": asset.sha256,
        "size": asset.size,
    }


def _receipt(
    evidence: GitHubEvidence, self_asset_name: str, expected_assets_sha256: str
) -> dict[str, Any]:
    return {
        "artifacts": [
            _artifact_row(asset)
            for asset in sorted(evidence.assets.values(), key=lambda item: item.name.encode("utf-8"))
        ],
        "release": {
            "id": evidence.release_id,
            "prerelease": evidence.prerelease,
            "expected_assets_manifest_sha256": expected_assets_sha256,
            "self_asset_name": self_asset_name,
            "tag_name": evidence.tag_name,
            "tag_object_sha": evidence.tag_object_sha,
            "tag_peeled_commit_sha": evidence.peeled_commit_sha,
            "tree_sha": evidence.tree_sha,
        },
        "repository": {
            "full_name": evidence.repository_full_name,
            "id": evidence.repository_id,
        },
        "schema": SCHEMA,
    }


def _canonical_bytes(value: dict[str, Any]) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _validate_receipt(value: dict[str, Any], raw: bytes) -> dict[str, Any]:
    _exact_keys(value, {"artifacts", "release", "repository", "schema"}, "receipt")
    if value["schema"] != SCHEMA:
        raise ReleaseAssetInventoryError("receipt schema is unsupported")
    repository = value["repository"]
    release = value["release"]
    artifacts = value["artifacts"]
    if not isinstance(repository, dict) or not isinstance(release, dict) or not isinstance(artifacts, list):
        raise ReleaseAssetInventoryError("receipt sections have invalid types")
    _exact_keys(repository, {"full_name", "id"}, "receipt.repository")
    _exact_keys(
        release,
        {
            "id",
            "prerelease",
            "expected_assets_manifest_sha256",
            "self_asset_name",
            "tag_name",
            "tag_object_sha",
            "tag_peeled_commit_sha",
            "tree_sha",
        },
        "receipt.release",
    )
    _integer(repository["id"], "receipt.repository.id")
    full_name = _string(repository["full_name"], "receipt.repository.full_name")
    if FULL_NAME.fullmatch(full_name) is None:
        raise ReleaseAssetInventoryError("receipt.repository.full_name is malformed")
    _integer(release["id"], "receipt.release.id")
    _boolean(release["prerelease"], "receipt.release.prerelease")
    _sha256(
        release["expected_assets_manifest_sha256"],
        "receipt.release.expected_assets_manifest_sha256",
    )
    _asset_name(release["self_asset_name"], "receipt.release.self_asset_name")
    tag_name = _string(release["tag_name"], "receipt.release.tag_name")
    if TAG_NAME.fullmatch(tag_name) is None:
        raise ReleaseAssetInventoryError("receipt.release.tag_name is malformed")
    for key in ("tag_object_sha", "tag_peeled_commit_sha", "tree_sha"):
        _git_object(release[key], f"receipt.release.{key}")

    names: list[str] = []
    ids: set[int] = set()
    for index, row in enumerate(artifacts):
        if not isinstance(row, dict):
            raise ReleaseAssetInventoryError(f"receipt.artifacts[{index}] must be an object")
        _exact_keys(row, {"github_asset_id", "name", "sha256", "size"}, "artifact row")
        asset_id = _integer(row["github_asset_id"], f"receipt.artifacts[{index}].github_asset_id")
        name = _asset_name(row["name"], f"receipt.artifacts[{index}].name")
        size = row["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ReleaseAssetInventoryError(f"receipt.artifacts[{index}].size is invalid")
        _sha256(row["sha256"], f"receipt.artifacts[{index}].sha256")
        if asset_id in ids:
            raise ReleaseAssetInventoryError("receipt contains duplicate GitHub asset IDs")
        ids.add(asset_id)
        names.append(name)
    if names != sorted(names, key=lambda item: item.encode("utf-8")) or len(names) != len(set(names)):
        raise ReleaseAssetInventoryError("receipt artifacts are not unique canonical name order")
    if release["self_asset_name"] in set(names):
        raise ReleaseAssetInventoryError("receipt must exclude its own artifact row")
    if raw != _canonical_bytes(value):
        raise ReleaseAssetInventoryError("receipt bytes are not canonical JSON")
    return value


def _compare_evidence(
    receipt: dict[str, Any], evidence: GitHubEvidence, expected_assets_sha256: str
) -> str:
    repository = receipt["repository"]
    release = receipt["release"]
    expected_repository = {
        "full_name": evidence.repository_full_name,
        "id": evidence.repository_id,
    }
    expected_release = {
        "id": evidence.release_id,
        "prerelease": evidence.prerelease,
        "expected_assets_manifest_sha256": expected_assets_sha256,
        "self_asset_name": release["self_asset_name"],
        "tag_name": evidence.tag_name,
        "tag_object_sha": evidence.tag_object_sha,
        "tag_peeled_commit_sha": evidence.peeled_commit_sha,
        "tree_sha": evidence.tree_sha,
    }
    if repository != expected_repository or release != expected_release:
        raise ReleaseAssetInventoryError("receipt repository/release identity differs from GitHub evidence")
    return release["self_asset_name"]


def build_receipt(
    *,
    evidence: GitHubEvidence,
    asset_dir: Path,
    self_asset_name: str,
    output: Path,
    expected: ExpectedAssets,
    tag_checkout: Path,
) -> tuple[int, str]:
    self_name = _asset_name(self_asset_name, "self asset name")
    if self_name in evidence.assets:
        raise ReleaseAssetInventoryError(
            "build requires the pre-self-upload GitHub release response"
        )
    local = _asset_directory(asset_dir, set(evidence.assets))
    _audit_downloaded_assets(asset_dir, set(local))
    _verify_intended_assets(
        checkout=tag_checkout,
        evidence=evidence,
        expected=expected,
        asset_dir=asset_dir,
        local=local,
    )
    for name, asset in evidence.assets.items():
        size, digest, _ = local[name]
        if size != asset.size or digest != asset.sha256:
            raise ReleaseAssetInventoryError(
                f"downloaded asset differs from GitHub digest before receipt build: {name!r}"
            )
    if output.name != self_name:
        raise ReleaseAssetInventoryError("receipt output basename must equal self asset name")
    if output.exists() or output.is_symlink():
        raise ReleaseAssetInventoryError("receipt output already exists")
    value = _receipt(evidence, self_name, expected.sha256)
    raw = _canonical_bytes(value)
    _validate_receipt(value, raw)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as stream:
        stream.write(raw)
    return len(evidence.assets), hashlib.sha256(raw).hexdigest()


def verify_receipt(
    *,
    evidence: GitHubEvidence,
    asset_dir: Path,
    receipt_path: Path,
    expected: ExpectedAssets,
    tag_checkout: Path,
) -> tuple[int, str]:
    decoded, raw = _read_json(receipt_path, "release asset inventory receipt")
    receipt = _validate_receipt(decoded, raw)
    self_name = _compare_evidence(receipt, evidence, expected.sha256)
    if self_name not in evidence.assets:
        raise ReleaseAssetInventoryError(
            "post-release GitHub response lacks the receipt asset itself"
        )
    local = _asset_directory(asset_dir, set(evidence.assets))
    _audit_downloaded_assets(asset_dir, set(local))
    if local[self_name][2] != raw:
        raise ReleaseAssetInventoryError("receipt path bytes differ from downloaded receipt asset")

    receipt_rows = {row["name"]: row for row in receipt["artifacts"]}
    expected_payloads = set(evidence.assets) - {self_name}
    if set(receipt_rows) != expected_payloads:
        raise ReleaseAssetInventoryError(
            "receipt payload set differs from post-release GitHub assets"
        )
    payload_local = {name: row for name, row in local.items() if name != self_name}
    _verify_intended_assets(
        checkout=tag_checkout,
        evidence=evidence,
        expected=expected,
        asset_dir=asset_dir,
        local=payload_local,
    )
    for name, asset in evidence.assets.items():
        size, digest, _ = local[name]
        if size != asset.size or digest != asset.sha256:
            raise ReleaseAssetInventoryError(
                f"downloaded asset differs from post-release GitHub response: {name!r}"
            )
        if name == self_name:
            continue
        if receipt_rows[name] != _artifact_row(asset):
            raise ReleaseAssetInventoryError(
                f"receipt artifact row differs from GitHub response: {name!r}"
            )
    return len(evidence.assets), hashlib.sha256(raw).hexdigest()


def _add_evidence_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--expected-repository-id", type=int, required=True)
    parser.add_argument("--expected-repository", required=True)
    parser.add_argument("--expected-tag", required=True)
    parser.add_argument("--repository-json", type=Path, required=True)
    parser.add_argument("--release-json", type=Path, required=True)
    parser.add_argument("--tag-ref-json", type=Path, required=True)
    parser.add_argument("--tag-object-json", type=Path, required=True)
    parser.add_argument("--commit-json", type=Path, required=True)
    parser.add_argument("--asset-dir", type=Path, required=True)
    parser.add_argument("--expected-assets-json", type=Path, required=True)
    parser.add_argument("--expected-assets-sha256", required=True)
    parser.add_argument("--tag-checkout", type=Path, required=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    build = subparsers.add_parser("build")
    _add_evidence_arguments(build)
    build.add_argument("--self-asset-name", required=True)
    build.add_argument("--output", type=Path, required=True)
    verify = subparsers.add_parser("verify")
    _add_evidence_arguments(verify)
    verify.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args(argv)

    expected_repository_id = _integer(
        args.expected_repository_id, "expected repository ID"
    )
    expected_repository = _string(args.expected_repository, "expected repository")
    expected_tag = _string(args.expected_tag, "expected tag")
    if FULL_NAME.fullmatch(expected_repository) is None:
        raise ReleaseAssetInventoryError("expected repository is malformed")
    if TAG_NAME.fullmatch(expected_tag) is None:
        raise ReleaseAssetInventoryError("expected tag is malformed")

    evidence = _evidence(
        args.repository_json,
        args.release_json,
        args.tag_ref_json,
        args.tag_object_json,
        args.commit_json,
        expected_repository_id=expected_repository_id,
        expected_repository=expected_repository,
        expected_tag=expected_tag,
    )
    expected = _expected_assets(
        args.expected_assets_json,
        args.expected_assets_sha256,
        evidence,
    )
    if args.command == "build":
        count, digest = build_receipt(
            evidence=evidence,
            asset_dir=args.asset_dir,
            self_asset_name=args.self_asset_name,
            output=args.output,
            expected=expected,
            tag_checkout=args.tag_checkout,
        )
        print(f"RELEASE_ASSET_INVENTORY_BUILT artifacts={count} receipt_sha256={digest}")
    else:
        count, digest = verify_receipt(
            evidence=evidence,
            asset_dir=args.asset_dir,
            receipt_path=args.receipt,
            expected=expected,
            tag_checkout=args.tag_checkout,
        )
        print(f"RELEASE_ASSET_INVENTORY_VERIFIED assets={count} receipt_sha256={digest}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ReleaseAssetInventoryError) as exc:
        print(f"RELEASE_ASSET_INVENTORY_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
