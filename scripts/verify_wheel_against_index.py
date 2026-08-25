"""Bind a built PhaseSet wheel to the staged Git-index source bytes.

This verifier deliberately reads ``src/`` and ``pyproject.toml`` through the
Git index rather than through the worktree.  It is complementary to
``audit_release_artifact.py``: the latter scans archive safety, while this file
proves that the wheel's Python modules and identity metadata are exactly those
selected for the release commit.
"""

from __future__ import annotations

import argparse
import base64
import configparser
import csv
from collections import Counter
from dataclasses import dataclass
from email import policy
from email.parser import BytesParser
import hashlib
import io
import os
from pathlib import Path, PurePosixPath
import re
import stat
import subprocess
import sys
import tomllib
import zipfile

sys.dont_write_bytecode = True

from release_tree import (  # noqa: E402
    ReleaseTreeError,
    clean_git_environment,
    git_tracked_paths,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = (PurePosixPath("phasepair_core"), PurePosixPath("phaseset_core"))
MAX_MEMBER_BYTES = 64 * 1024 * 1024
MAX_WHEEL_BYTES = 256 * 1024 * 1024


class WheelIndexVerificationError(RuntimeError):
    """A wheel is not an exact package of the staged PhaseSet source."""


@dataclass(frozen=True)
class StagedProject:
    name: str
    version: str
    requires_python: str | None
    dependencies: tuple[str, ...]
    optional_dependencies: dict[str, tuple[str, ...]]
    entry_points: dict[str, dict[str, str]]


def _index_blob(path: PurePosixPath) -> bytes:
    completed = subprocess.run(
        ["git", "cat-file", "blob", f":{path.as_posix()}"],
        cwd=ROOT,
        check=True,
        env=clean_git_environment(),
        stdout=subprocess.PIPE,
    )
    return completed.stdout


def _safe_member_path(decoded: str) -> PurePosixPath:
    if not decoded or any(character in decoded for character in "\0\t\n\r\\:"):
        raise WheelIndexVerificationError(f"invalid wheel member: {decoded!r}")
    path = PurePosixPath(decoded.rstrip("/"))
    if (
        path.is_absolute()
        or path == PurePosixPath(".")
        or ".." in path.parts
        or path.as_posix() != decoded.rstrip("/")
    ):
        raise WheelIndexVerificationError(f"unsafe wheel member: {decoded!r}")
    return path


def _read_wheel(path: Path) -> dict[PurePosixPath, bytes]:
    raw = path.read_bytes()
    if len(raw) > MAX_WHEEL_BYTES:
        raise WheelIndexVerificationError("wheel exceeds the verification size bound")
    members: dict[PurePosixPath, bytes] = {}
    total = 0
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as archive:
            for info in archive.infolist():
                member = _safe_member_path(info.filename)
                mode = info.external_attr >> 16
                if info.is_dir():
                    continue
                if info.flag_bits & 0x1:
                    raise WheelIndexVerificationError("encrypted wheel members are forbidden")
                file_type = stat.S_IFMT(mode)
                if file_type and file_type != stat.S_IFREG:
                    raise WheelIndexVerificationError(
                        f"non-regular wheel member: {member.as_posix()}"
                    )
                if mode & 0o111:
                    raise WheelIndexVerificationError(
                        f"executable wheel member is forbidden: {member.as_posix()}"
                    )
                if info.file_size > MAX_MEMBER_BYTES:
                    raise WheelIndexVerificationError(
                        f"oversized wheel member: {member.as_posix()}"
                    )
                total += info.file_size
                if total > MAX_WHEEL_BYTES:
                    raise WheelIndexVerificationError("wheel payload exceeds the size bound")
                if member in members:
                    raise WheelIndexVerificationError(
                        f"duplicate wheel member: {member.as_posix()}"
                    )
                payload = archive.read(info)
                if len(payload) != info.file_size:
                    raise WheelIndexVerificationError(
                        f"wheel member size mismatch: {member.as_posix()}"
                    )
                members[member] = payload
    except zipfile.BadZipFile as exc:
        raise WheelIndexVerificationError("invalid wheel ZIP framing") from exc
    if not members:
        raise WheelIndexVerificationError("wheel contains no regular files")
    return members


def _requirement_list(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() and "\r" not in item and "\n" not in item
        for item in value
    ):
        raise WheelIndexVerificationError(f"{label} must be a list of requirement strings")
    return tuple(item.strip() for item in value)


def _staged_project() -> StagedProject:
    tracked = set(git_tracked_paths(ROOT))
    pyproject_path = PurePosixPath("pyproject.toml")
    if pyproject_path not in tracked:
        raise WheelIndexVerificationError("staged index lacks pyproject.toml")
    try:
        decoded = tomllib.loads(_index_blob(pyproject_path).decode("utf-8", "strict"))
        project = decoded["project"]
        name = project["name"]
        version = project["version"]
    except (KeyError, TypeError, UnicodeDecodeError, tomllib.TOMLDecodeError) as exc:
        raise WheelIndexVerificationError("staged project metadata is malformed") from exc
    if not isinstance(name, str) or not isinstance(version, str) or not name or not version:
        raise WheelIndexVerificationError("staged project name/version must be non-empty strings")
    requires_python = project.get("requires-python")
    if requires_python is not None and (
        not isinstance(requires_python, str)
        or not requires_python.strip()
        or "\r" in requires_python
        or "\n" in requires_python
    ):
        raise WheelIndexVerificationError("staged project.requires-python is malformed")
    dependencies = _requirement_list(project.get("dependencies", []), "staged dependencies")
    raw_optional = project.get("optional-dependencies", {})
    if not isinstance(raw_optional, dict):
        raise WheelIndexVerificationError("staged project.optional-dependencies is malformed")
    optional_dependencies: dict[str, tuple[str, ...]] = {}
    for extra, raw_requirements in raw_optional.items():
        if not isinstance(extra, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", extra):
            raise WheelIndexVerificationError("staged optional dependency extra is malformed")
        normalized = re.sub(r"[-_.]+", "-", extra).casefold()
        if normalized in optional_dependencies:
            raise WheelIndexVerificationError("staged optional dependency extras collide")
        optional_dependencies[normalized] = _requirement_list(
            raw_requirements, f"staged optional dependency {extra!r}"
        )

    entry_points: dict[str, dict[str, str]] = {}
    for project_key, group in (("scripts", "console_scripts"), ("gui-scripts", "gui_scripts")):
        raw_group = project.get(project_key, {})
        if not isinstance(raw_group, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in raw_group.items()
        ):
            raise WheelIndexVerificationError(f"staged project.{project_key} is malformed")
        if raw_group:
            entry_points[group] = dict(raw_group)
    raw_groups = project.get("entry-points", {})
    if not isinstance(raw_groups, dict):
        raise WheelIndexVerificationError("staged project.entry-points is malformed")
    for group, raw_group in raw_groups.items():
        if (
            not isinstance(group, str)
            or not isinstance(raw_group, dict)
            or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in raw_group.items()
            )
        ):
            raise WheelIndexVerificationError("staged project.entry-points is malformed")
        if group in entry_points:
            raise WheelIndexVerificationError(f"duplicate staged entry-point group: {group!r}")
        entry_points[group] = dict(raw_group)
    return StagedProject(
        name=name,
        version=version,
        requires_python=requires_python.strip() if requires_python is not None else None,
        dependencies=dependencies,
        optional_dependencies=optional_dependencies,
        entry_points=entry_points,
    )


def _expected_sources() -> dict[PurePosixPath, bytes]:
    expected: dict[PurePosixPath, bytes] = {}
    prefixes = tuple(PurePosixPath("src") / root for root in PACKAGE_ROOTS)
    for staged_path in git_tracked_paths(ROOT):
        if staged_path.suffix != ".py" or not any(
            staged_path == prefix or prefix in staged_path.parents for prefix in prefixes
        ):
            continue
        wheel_path = PurePosixPath(*staged_path.parts[1:])
        expected[wheel_path] = _index_blob(staged_path)
    for root in PACKAGE_ROOTS:
        if root / "__init__.py" not in expected:
            raise WheelIndexVerificationError(
                f"staged index lacks required package source: {root.as_posix()}/__init__.py"
            )
    return expected


def _canonical_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _wheel_distribution(value: str) -> str:
    return re.sub(r"[-_.]+", "_", value).casefold()


def _single_header(message: object, name: str) -> str:
    values = message.get_all(name, [])  # type: ignore[attr-defined]
    if len(values) != 1 or not isinstance(values[0], str):
        raise WheelIndexVerificationError(f"wheel METADATA must contain one {name} header")
    return values[0]


def _optional_single_header(message: object, name: str) -> str | None:
    values = message.get_all(name, [])  # type: ignore[attr-defined]
    if len(values) > 1 or any(not isinstance(value, str) for value in values):
        raise WheelIndexVerificationError(f"wheel METADATA has ambiguous {name} headers")
    return values[0] if values else None


def _canonical_requirement(value: str) -> tuple[str, tuple[str, ...], tuple[str, ...], str]:
    requirement, separator, marker = value.partition(";")
    match = re.fullmatch(
        r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)(?:\[([A-Za-z0-9._,-]+)\])?\s*(.*?)\s*",
        requirement,
    )
    if match is None or " @ " in requirement:
        raise WheelIndexVerificationError(f"unsupported or malformed requirement: {value!r}")
    name = _canonical_distribution(match.group(1))
    extras = tuple(
        sorted(
            {
                re.sub(r"[-_.]+", "-", extra).casefold()
                for extra in (match.group(2) or "").split(",")
                if extra
            }
        )
    )
    specifier = match.group(3).strip()
    if specifier.startswith("(") and specifier.endswith(")"):
        specifier = specifier[1:-1].strip()
    specifiers = tuple(
        sorted(re.sub(r"\s+", "", item) for item in specifier.split(",") if item.strip())
    )
    normalized_marker = ""
    if separator:
        normalized_marker = re.sub(r"\s+", " ", marker.strip()).replace("'", '"').casefold()
        normalized_marker = re.sub(r"\(\s*(.*?)\s*\)", r"(\1)", normalized_marker)
    return name, extras, specifiers, normalized_marker


def _expected_requirements(project: StagedProject) -> tuple[str, ...]:
    expected = list(project.dependencies)
    for extra, requirements in sorted(project.optional_dependencies.items()):
        for requirement in requirements:
            base, separator, marker = requirement.partition(";")
            if separator:
                expected.append(f'{base.strip()}; ({marker.strip()}) and extra == "{extra}"')
            else:
                expected.append(f'{base.strip()}; extra == "{extra}"')
    return tuple(expected)


def _metadata_requirements(message: object, project: StagedProject) -> None:
    actual_raw = message.get_all("Requires-Dist", [])  # type: ignore[attr-defined]
    if any(not isinstance(item, str) for item in actual_raw):
        raise WheelIndexVerificationError("wheel METADATA Requires-Dist is malformed")
    expected_raw = _expected_requirements(project)
    actual = [_canonical_requirement(item) for item in actual_raw]
    expected = [_canonical_requirement(item) for item in expected_raw]
    if Counter(actual) != Counter(expected):
        raise WheelIndexVerificationError(
            f"wheel Requires-Dist differs from staged project: actual={actual!r} expected={expected!r}"
        )
    provides_extra = message.get_all("Provides-Extra", [])  # type: ignore[attr-defined]
    if any(not isinstance(item, str) for item in provides_extra):
        raise WheelIndexVerificationError("wheel METADATA Provides-Extra is malformed")
    normalized_extras = tuple(
        sorted(re.sub(r"[-_.]+", "-", item).casefold() for item in provides_extra)
    )
    if normalized_extras != tuple(sorted(project.optional_dependencies)):
        raise WheelIndexVerificationError("wheel Provides-Extra differs from staged project")
    actual_python = _optional_single_header(message, "Requires-Python")
    if _python_specifier_terms(actual_python) != _python_specifier_terms(
        project.requires_python
    ):
        raise WheelIndexVerificationError("wheel Requires-Python differs from staged project")


def _python_specifier_terms(value: str | None) -> tuple[str, ...] | None:
    """Normalize only order and whitespace in a PEP 440 specifier set.

    Build backends are allowed to reorder comma-separated terms (setuptools
    emits ``<3.15,>=3.12`` for staged ``>=3.12,<3.15``).  Keeping every compact
    term byte-identical after sorting accepts that representation change while
    still rejecting a changed bound, operator, version, duplicate, or term set.
    """

    if value is None:
        return None
    compact = re.sub(r"\s+", "", value)
    terms = compact.split(",")
    if not compact or any(not term for term in terms):
        raise WheelIndexVerificationError("Requires-Python specifier set is malformed")
    return tuple(sorted(terms))


def _metadata_identity(
    members: dict[PurePosixPath, bytes], wheel: Path, project: StagedProject
) -> tuple[PurePosixPath, object]:
    filename_parts = wheel.name[:-4].split("-") if wheel.name.endswith(".whl") else []
    expected_distribution = _wheel_distribution(project.name)
    if (
        len(filename_parts) < 5
        or filename_parts[0].casefold() != expected_distribution
        or filename_parts[1] != project.version
    ):
        raise WheelIndexVerificationError("wheel filename does not bind staged name/version")

    metadata_paths = tuple(
        path for path in members if len(path.parts) == 2 and path.name == "METADATA"
    )
    if len(metadata_paths) != 1 or not metadata_paths[0].parent.name.endswith(".dist-info"):
        raise WheelIndexVerificationError("wheel must contain exactly one dist-info/METADATA")
    dist_info = metadata_paths[0].parent
    dist_info_roots = {
        path.parts[0] for path in members if path.parts[0].casefold().endswith(".dist-info")
    }
    if dist_info_roots != {dist_info.name}:
        raise WheelIndexVerificationError("wheel contains an extra or ambiguous dist-info tree")
    expected_dist_info = f"{expected_distribution}-{project.version}.dist-info"
    if dist_info.name.casefold() != expected_dist_info.casefold():
        raise WheelIndexVerificationError("dist-info directory does not bind staged name/version")

    message = BytesParser(policy=policy.default).parsebytes(members[metadata_paths[0]])
    metadata_name = _single_header(message, "Name")
    metadata_version = _single_header(message, "Version")
    if _canonical_distribution(metadata_name) != _canonical_distribution(project.name):
        raise WheelIndexVerificationError("wheel METADATA Name differs from staged project")
    if metadata_version != project.version:
        raise WheelIndexVerificationError("wheel METADATA Version differs from staged project")
    wheel_metadata = dist_info / "WHEEL"
    if wheel_metadata not in members:
        raise WheelIndexVerificationError("wheel lacks dist-info/WHEEL metadata")
    wheel_message = BytesParser(policy=policy.default).parsebytes(members[wheel_metadata])
    if _single_header(wheel_message, "Wheel-Version") != "1.0":
        raise WheelIndexVerificationError("wheel metadata version is not 1.0")
    _metadata_requirements(message, project)
    return dist_info, message


def _entry_points(
    members: dict[PurePosixPath, bytes],
    dist_info: PurePosixPath,
    expected: dict[str, dict[str, str]],
) -> None:
    path = dist_info / "entry_points.txt"
    if path not in members:
        if expected:
            raise WheelIndexVerificationError("wheel lacks staged entry-point metadata")
        return
    parser = configparser.ConfigParser(interpolation=None, strict=True)
    parser.optionxform = str
    try:
        parser.read_string(members[path].decode("utf-8", "strict"))
    except (configparser.Error, UnicodeDecodeError) as exc:
        raise WheelIndexVerificationError("wheel entry_points.txt is malformed") from exc
    actual = {section: dict(parser.items(section)) for section in parser.sections()}
    if actual != expected:
        raise WheelIndexVerificationError(
            f"wheel entry points differ from staged project: actual={actual!r} expected={expected!r}"
        )


def _known_generated_members(
    dist_info: PurePosixPath,
    expected_sources: dict[PurePosixPath, bytes],
    entry_points: dict[str, dict[str, str]],
) -> dict[PurePosixPath, bytes | None]:
    generated: dict[PurePosixPath, bytes | None] = {
        dist_info / "METADATA": None,
        dist_info / "WHEEL": None,
        dist_info / "RECORD": None,
    }
    if entry_points:
        generated[dist_info / "entry_points.txt"] = None
    top_level = dist_info / "top_level.txt"
    roots = tuple(sorted({path.parts[0] for path in expected_sources}))
    generated[top_level] = ("\n".join(roots) + "\n").encode("utf-8")

    tracked = set(git_tracked_paths(ROOT))
    for staged_path in sorted(tracked, key=lambda path: path.as_posix().encode("utf-8")):
        if len(staged_path.parts) != 1 or not re.fullmatch(
            r"(?i)(?:licen[cs]e|copying|notice)(?:[._-].*)?", staged_path.name
        ):
            continue
        generated[dist_info / "licenses" / staged_path.name] = _index_blob(staged_path)
    return generated


def _validate_member_allowlist(
    members: dict[PurePosixPath, bytes],
    expected_sources: dict[PurePosixPath, bytes],
    dist_info: PurePosixPath,
    entry_points: dict[str, dict[str, str]],
) -> None:
    generated = _known_generated_members(dist_info, expected_sources, entry_points)
    required = {
        dist_info / "METADATA",
        dist_info / "WHEEL",
        dist_info / "RECORD",
    }
    if entry_points:
        required.add(dist_info / "entry_points.txt")
    missing = sorted(path.as_posix() for path in required - set(members))
    allowed = set(expected_sources) | set(generated)
    extra = sorted(path.as_posix() for path in set(members) - allowed)
    if missing or extra:
        raise WheelIndexVerificationError(
            f"wheel member set is not allowlisted: missing={missing!r} extra={extra!r}"
        )
    for member, expected in generated.items():
        if member in members and expected is not None and members[member] != expected:
            raise WheelIndexVerificationError(
                f"wheel generated member differs from staged derivation: {member.as_posix()}"
            )


def _record_digest(raw: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(raw).digest()).rstrip(b"=").decode("ascii")


def _validate_record(members: dict[PurePosixPath, bytes], dist_info: PurePosixPath) -> None:
    record_path = dist_info / "RECORD"
    try:
        decoded = members[record_path].decode("utf-8", "strict")
        rows = tuple(csv.reader(io.StringIO(decoded, newline=""), strict=True))
    except (KeyError, UnicodeDecodeError, csv.Error) as exc:
        raise WheelIndexVerificationError("wheel RECORD is missing or malformed") from exc
    recorded: dict[PurePosixPath, tuple[str, str]] = {}
    for row in rows:
        if len(row) != 3:
            raise WheelIndexVerificationError("wheel RECORD rows must contain exactly three fields")
        member = _safe_member_path(row[0])
        if member in recorded:
            raise WheelIndexVerificationError(f"duplicate wheel RECORD row: {member.as_posix()}")
        recorded[member] = (row[1], row[2])
    if set(recorded) != set(members):
        missing = sorted(path.as_posix() for path in set(members) - set(recorded))
        extra = sorted(path.as_posix() for path in set(recorded) - set(members))
        raise WheelIndexVerificationError(
            f"wheel RECORD coverage differs from members: missing={missing!r} extra={extra!r}"
        )
    for member, raw in members.items():
        digest, size = recorded[member]
        if member == record_path:
            if digest or size:
                raise WheelIndexVerificationError("wheel RECORD self-row must have empty hash and size")
            continue
        if digest != f"sha256={_record_digest(raw)}" or size != str(len(raw)):
            raise WheelIndexVerificationError(
                f"wheel RECORD hash/size mismatch: {member.as_posix()}"
            )


def _source_set_digest(sources: dict[PurePosixPath, bytes]) -> str:
    digest = hashlib.sha256(b"phaseset-staged-wheel-source-set-v1\0")
    for path, raw in sorted(sources.items(), key=lambda item: item[0].as_posix().encode("utf-8")):
        encoded = path.as_posix().encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def verify_wheel(path: Path) -> tuple[int, str, str]:
    if path.suffix.casefold() != ".whl" or not path.is_file() or path.is_symlink():
        raise WheelIndexVerificationError(f"not a regular wheel: {path}")
    members = _read_wheel(path)
    project = _staged_project()
    expected = _expected_sources()
    actual = {member: raw for member, raw in members.items() if member.suffix == ".py"}
    missing = sorted((path.as_posix() for path in set(expected) - set(actual)))
    extra = sorted((path.as_posix() for path in set(actual) - set(expected)))
    if missing or extra:
        raise WheelIndexVerificationError(
            f"wheel Python source set differs from staged index: missing={missing!r} extra={extra!r}"
        )
    mismatched = sorted(
        path.as_posix() for path in expected if expected[path] != actual[path]
    )
    if mismatched:
        raise WheelIndexVerificationError(
            f"wheel Python source bytes differ from staged index: {mismatched!r}"
        )
    dist_info, _ = _metadata_identity(members, path, project)
    _entry_points(members, dist_info, project.entry_points)
    _validate_member_allowlist(members, expected, dist_info, project.entry_points)
    _validate_record(members, dist_info)
    return len(expected), project.version, _source_set_digest(expected)


def _wheel_paths(arguments: list[str]) -> tuple[Path, ...]:
    resolved: list[Path] = []
    for raw in arguments:
        path = Path(raw)
        if path.is_dir():
            resolved.extend(candidate for candidate in path.rglob("*.whl") if candidate.is_file())
        elif path.is_file():
            resolved.append(path)
        else:
            raise WheelIndexVerificationError(f"wheel path does not exist: {raw!r}")
    ordered = tuple(sorted(set(resolved), key=lambda value: os.fsencode(str(value))))
    if not ordered:
        raise WheelIndexVerificationError("no wheels found")
    return ordered


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args(argv)
    for path in _wheel_paths(args.paths):
        source_count, version, source_digest = verify_wheel(path)
        wheel_digest = hashlib.sha256(path.read_bytes()).hexdigest()
        print(
            "WHEEL_INDEX_BINDING_PASS "
            f"wheel={path} wheel_sha256={wheel_digest} version={version} "
            f"sources={source_count} staged_source_set_sha256={source_digest}"
        )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (
        OSError,
        ReleaseTreeError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
        WheelIndexVerificationError,
        zipfile.BadZipFile,
    ) as exc:
        print(f"WHEEL_INDEX_BINDING_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
