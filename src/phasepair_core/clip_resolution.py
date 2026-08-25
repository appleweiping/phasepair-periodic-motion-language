"""Read-only, fail-closed preflight for the pinned PhasePair CLIP snapshot.

This module deliberately does not import Hugging Face, Transformers, Torch, or
any network client.  It can inventory an already-present local directory and
describe separately supplied evidence, but it can never grant training or
production authority.  Even a complete preflight remains pending fresh review.

A successful inventory means two complete, independently rehashed scans had
equal root, census, per-file content, and metadata observations.  That is a
best-effort stable-snapshot check, not an operating-system atomic security seal.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable


MODEL_ID = "openai/clip-vit-base-patch32"
REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
TOTAL_BYTES = 608_863_580
AUTHORITY = 0
PRODUCTION = False
TRAINING_AUTHORIZED = False

# These public values are informational.  The public assessor recreates the
# literal contract internally, so rebinding a module global cannot relax it.
PINNED_FILES = (
    ("pytorch_model.bin", 605_247_071, "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f"),
    ("config.json", 4_186, "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770"),
    ("merges.txt", 524_657, "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85"),
    (
        "preprocessor_config.json",
        316,
        "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
    ),
    (
        "special_tokens_map.json",
        389,
        "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
    ),
    (
        "tokenizer_config.json",
        592,
        "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
    ),
    (
        "tokenizer.json",
        2_224_041,
        "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
    ),
    ("vocab.json", 862_328, "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363"),
)

# Deliberately unused by the assessor; retained only so mutation tests can
# demonstrate that private caches are not semantic authority either.
_INTERNAL_PIN_CACHE = PINNED_FILES
_INTERNAL_TOTAL_BYTES = TOTAL_BYTES
_INTERNAL_MODEL_ID = MODEL_ID
_INTERNAL_REVISION = REVISION


class ClipResolutionError(ValueError):
    """Raised when an input or purported assessment violates the closed schema."""


def _require_exact_bool(
    value: object,
    label: str,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> bool:
    if type(value) is not bool:
        raise _error_type(f"{label} must be an exact built-in bool")
    return value


def _require_exact_int(
    value: object,
    label: str,
    *,
    minimum: int = 0,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> int:
    if type(value) is not int or value < minimum:
        raise _error_type(
            f"{label} must be an exact built-in int >= {minimum}"
        )
    return value


def _require_exact_str(
    value: object,
    label: str,
    *,
    nonempty: bool = True,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> str:
    if type(value) is not str or (nonempty and not value):
        suffix = "nonempty " if nonempty else ""
        raise _error_type(f"{label} must be an exact {suffix}built-in str")
    return value


def _require_sha256(
    value: object,
    label: str,
    _exact_str: Callable[..., str] = _require_exact_str,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> str:
    text = _exact_str(value, label)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise _error_type(f"{label} must be lowercase SHA-256 hex")
    return text


def _require_safe_name(
    value: object,
    label: str,
    _exact_str: Callable[..., str] = _require_exact_str,
    _basename: Callable[[str], str] = os.path.basename,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> str:
    name = _exact_str(value, label)
    if (
        name in {".", ".."}
        or "\x00" in name
        or "/" in name
        or "\\" in name
        or _basename(name) != name
    ):
        raise _error_type(f"{label} must be one safe relative basename")
    return name


@dataclass(frozen=True, slots=True)
class ClipFilePin:
    """One exact expected snapshot-file identity."""

    path: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        _require_safe_name(self.path, "pin.path")
        _require_exact_int(self.bytes, "pin.bytes", minimum=1)
        _require_sha256(self.sha256, "pin.sha256")


@dataclass(frozen=True, slots=True)
class WheelIdentity:
    """Content identity for one locally retained runtime wheel."""

    distribution: str
    filename: str
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        distribution = _require_exact_str(self.distribution, "wheel.distribution")
        if distribution not in {"transformers", "huggingface-hub", "tokenizers"}:
            raise ClipResolutionError("wheel.distribution is outside the exact runtime set")
        filename = _require_safe_name(self.filename, "wheel.filename")
        if not filename.endswith(".whl"):
            raise ClipResolutionError("wheel.filename must end in .whl")
        _require_exact_int(self.bytes, "wheel.bytes", minimum=1)
        _require_sha256(self.sha256, "wheel.sha256")


@dataclass(frozen=True, slots=True)
class RuntimeWheelEvidence:
    """Exact three-wheel runtime evidence, copied on construction."""

    wheels: tuple[WheelIdentity, ...]

    def __post_init__(self) -> None:
        if type(self.wheels) is not tuple:
            raise ClipResolutionError("runtime.wheels must be an exact built-in tuple")
        copied: list[WheelIdentity] = []
        for index, wheel in enumerate(self.wheels):
            if type(wheel) is not WheelIdentity:
                raise ClipResolutionError(f"runtime.wheels[{index}] has the wrong exact type")
            copied.append(
                WheelIdentity(wheel.distribution, wheel.filename, wheel.bytes, wheel.sha256)
            )
        if tuple(wheel.distribution for wheel in copied) != (
            "transformers",
            "huggingface-hub",
            "tokenizers",
        ):
            raise ClipResolutionError("runtime wheels must be in the exact canonical order")
        if len({wheel.filename.casefold() for wheel in copied}) != 3:
            raise ClipResolutionError("runtime wheel filenames must be unique")
        object.__setattr__(self, "wheels", tuple(copied))


@dataclass(frozen=True, slots=True)
class TextLoaderEvidence:
    """Closed observations for the text-only loader and tokenizer path."""

    loader_class: str
    tokenizer_class: str
    local_files_only: bool
    trust_remote_code: bool
    text_model_called_once: bool
    exposes_final_ln_pooled_eos: bool
    exposes_pretrained_text_projection: bool
    vision_tower_instantiated: bool
    text_projection_requires_grad: bool
    active_text_dropout_sites: int
    tokenizer_max_length: int

    def __post_init__(self) -> None:
        _require_exact_str(self.loader_class, "loader_class")
        _require_exact_str(self.tokenizer_class, "tokenizer_class")
        if self.loader_class != "CLIPTextModelWithProjection":
            raise ClipResolutionError("loader_class must be CLIPTextModelWithProjection")
        if self.tokenizer_class != "CLIPTokenizerFast":
            raise ClipResolutionError("tokenizer_class must be CLIPTokenizerFast")
        for label in (
            "local_files_only",
            "trust_remote_code",
            "text_model_called_once",
            "exposes_final_ln_pooled_eos",
            "exposes_pretrained_text_projection",
            "vision_tower_instantiated",
            "text_projection_requires_grad",
        ):
            _require_exact_bool(getattr(self, label), label)
        if not self.local_files_only or self.trust_remote_code:
            raise ClipResolutionError("loader flags must be offline and remote-code disabled")
        if not self.text_model_called_once:
            raise ClipResolutionError("loader must call the text model exactly once")
        if not self.exposes_final_ln_pooled_eos or not self.exposes_pretrained_text_projection:
            raise ClipResolutionError("loader must expose pooled EOS and pretrained projection")
        if self.vision_tower_instantiated:
            raise ClipResolutionError("vision tower must never be instantiated")
        if self.text_projection_requires_grad:
            raise ClipResolutionError("pretrained text_projection must be frozen")
        if type(self.active_text_dropout_sites) is not int or self.active_text_dropout_sites != 0:
            raise ClipResolutionError("active_text_dropout_sites must be literal zero")
        if type(self.tokenizer_max_length) is not int or self.tokenizer_max_length != 77:
            raise ClipResolutionError("tokenizer_max_length must be literal 77")


@dataclass(frozen=True, slots=True)
class OpenTraceEvidence:
    """Closed relative-path census observed during an offline reload trace."""

    allowlist_paths: tuple[str, ...]
    opened_paths: tuple[str, ...]
    outside_paths: tuple[str, ...]

    def __post_init__(self) -> None:
        expected = (
            "pytorch_model.bin",
            "config.json",
            "merges.txt",
            "preprocessor_config.json",
            "special_tokens_map.json",
            "tokenizer_config.json",
            "tokenizer.json",
            "vocab.json",
        )
        for label in ("allowlist_paths", "opened_paths", "outside_paths"):
            value = getattr(self, label)
            if type(value) is not tuple:
                raise ClipResolutionError(f"trace.{label} must be an exact built-in tuple")
            for index, path in enumerate(value):
                _require_safe_name(path, f"trace.{label}[{index}]")
            if len(set(value)) != len(value):
                raise ClipResolutionError(f"trace.{label} must not contain duplicates")
        if self.allowlist_paths != expected:
            raise ClipResolutionError("trace allowlist must equal the exact eight-file order")
        if not self.opened_paths or any(path not in expected for path in self.opened_paths):
            raise ClipResolutionError("trace opened paths must be a nonempty allowlist subset")
        if self.outside_paths:
            raise ClipResolutionError("trace outside_paths must be empty")


@dataclass(frozen=True, slots=True)
class RightsEvidence:
    """Hash-bound rights-review verdict; it is not itself an authority grant."""

    verdict: str
    review_sha256: str
    private_research_use: bool
    redistribution_allowed: bool
    commercial_use_allowed: bool

    def __post_init__(self) -> None:
        _require_exact_str(self.verdict, "rights.verdict")
        if self.verdict != "PRIVATE_RESEARCH_ONLY_NO_REDISTRIBUTION":
            raise ClipResolutionError("rights verdict is not the frozen private-research verdict")
        _require_sha256(self.review_sha256, "rights.review_sha256")
        for label in (
            "private_research_use",
            "redistribution_allowed",
            "commercial_use_allowed",
        ):
            _require_exact_bool(getattr(self, label), f"rights.{label}")
        if (
            not self.private_research_use
            or self.redistribution_allowed
            or self.commercial_use_allowed
        ):
            raise ClipResolutionError("rights flags contradict the frozen verdict")


@dataclass(frozen=True, slots=True)
class SyntheticCaptionGoldenEvidence:
    """Content hashes for the frozen ASCII tokenizer/projection golden."""

    caption_count: int
    captions_ascii_sha256: str
    token_ids_sha256: str
    projected_float64_l2_sha256: str
    tokenizer_max_length: int
    projected_width: int

    def __post_init__(self) -> None:
        if type(self.caption_count) is not int or self.caption_count != 3:
            raise ClipResolutionError("golden.caption_count must be literal 3")
        for label in (
            "captions_ascii_sha256",
            "token_ids_sha256",
            "projected_float64_l2_sha256",
        ):
            _require_sha256(getattr(self, label), f"golden.{label}")
        if type(self.tokenizer_max_length) is not int or self.tokenizer_max_length != 77:
            raise ClipResolutionError("golden.tokenizer_max_length must be literal 77")
        if type(self.projected_width) is not int or self.projected_width != 512:
            raise ClipResolutionError("golden.projected_width must be literal 512")


@dataclass(frozen=True, slots=True)
class ClipResolutionEvidence:
    """Typed evidence bundle.  Opaque dictionaries are intentionally rejected."""

    runtime: RuntimeWheelEvidence | None = None
    loader: TextLoaderEvidence | None = None
    open_trace: OpenTraceEvidence | None = None
    rights: RightsEvidence | None = None
    golden: SyntheticCaptionGoldenEvidence | None = None

    def __post_init__(self) -> None:
        expected_types = (
            ("runtime", RuntimeWheelEvidence),
            ("loader", TextLoaderEvidence),
            ("open_trace", OpenTraceEvidence),
            ("rights", RightsEvidence),
            ("golden", SyntheticCaptionGoldenEvidence),
        )
        for label, expected_type in expected_types:
            value = getattr(self, label)
            if value is not None and type(value) is not expected_type:
                raise ClipResolutionError(f"evidence.{label} has the wrong exact type")


@dataclass(frozen=True, slots=True, init=False)
class ClipFileSnapshot:
    """Immutable output row; callers cannot construct output rows directly."""

    path: str
    bytes: int
    sha256: str

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ClipFileSnapshot construction is internal")


@dataclass(frozen=True, slots=True, init=False)
class ClipResolutionAssessment:
    """Immutable authority-zero preflight result with closed construction."""

    status: str
    authority: int
    production: bool
    training_authorized: bool
    files: tuple[ClipFileSnapshot, ...]
    total_bytes: int
    missing_files: tuple[str, ...]
    extra_files: tuple[str, ...]
    evidence: ClipResolutionEvidence
    _lease: object

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        raise TypeError("ClipResolutionAssessment construction is internal")


class _AssessmentLease:
    """Weak-key-able lifetime token owned strongly only by its assessment."""

    __slots__ = (
        "file_references",
        "evidence_references",
        "missing_reference",
        "extra_reference",
        "__weakref__",
    )

    def __init__(self) -> None:
        self.file_references: tuple[object, ...] = ()
        self.evidence_references: tuple[object, ...] = ()
        self.missing_reference: object = None
        self.extra_reference: object = None


@dataclass(frozen=True, slots=True)
class _FileIdentity:
    device: int
    inode: int
    mode: int
    links: int
    size: int
    mtime_ns: int
    ctime_ns: int
    attributes: int


@dataclass(frozen=True, slots=True)
class _FilesystemOps:
    lstat: Callable[[object], Any]
    scandir: Callable[[object], Any]
    open_fd: Callable[[object, int], int]
    fstat: Callable[[int], Any]
    read_fd: Callable[[int, int], bytes]
    close_fd: Callable[[int], None]

    def __post_init__(self) -> None:
        for label in ("lstat", "scandir", "open_fd", "fstat", "read_fd", "close_fd"):
            if not callable(getattr(self, label)):
                raise ClipResolutionError(f"filesystem operation {label} must be callable")


@dataclass(frozen=True, slots=True)
class _ScanOutcome:
    status: str | None
    files: tuple[ClipFileSnapshot, ...]
    total_bytes: int
    missing_files: tuple[str, ...]
    extra_files: tuple[str, ...]
    root_before: _FileIdentity | None = None
    root_after: _FileIdentity | None = None
    file_identities: tuple[_FileIdentity, ...] = ()
    census: tuple[str, ...] = ()
    double_pass_verified: bool = False


def _make_file_snapshot(
    path: str,
    byte_count: int,
    sha256: str,
    _snapshot_type: type[ClipFileSnapshot] = ClipFileSnapshot,
    _safe_name: Callable[[object, str], str] = _require_safe_name,
    _exact_int: Callable[..., int] = _require_exact_int,
    _sha256_hex: Callable[[object, str], str] = _require_sha256,
) -> ClipFileSnapshot:
    _safe_name(path, "snapshot.path")
    _exact_int(byte_count, "snapshot.bytes", minimum=0)
    _sha256_hex(sha256, "snapshot.sha256")
    value = object.__new__(_snapshot_type)
    object.__setattr__(value, "path", path)
    object.__setattr__(value, "bytes", byte_count)
    object.__setattr__(value, "sha256", sha256)
    return value


def _copy_runtime(
    value: RuntimeWheelEvidence,
    _runtime_type: type[RuntimeWheelEvidence] = RuntimeWheelEvidence,
    _wheel_type: type[WheelIdentity] = WheelIdentity,
) -> RuntimeWheelEvidence:
    if type(value) is not _runtime_type:
        raise ClipResolutionError("evidence.runtime has the wrong exact type")
    wheels = tuple(
        _wheel_type(wheel.distribution, wheel.filename, wheel.bytes, wheel.sha256)
        for wheel in value.wheels
    )
    return _runtime_type(wheels)


def _copy_evidence(
    value: ClipResolutionEvidence | None,
    _bundle_type: type[ClipResolutionEvidence] = ClipResolutionEvidence,
    _loader_type: type[TextLoaderEvidence] = TextLoaderEvidence,
    _trace_type: type[OpenTraceEvidence] = OpenTraceEvidence,
    _rights_type: type[RightsEvidence] = RightsEvidence,
    _golden_type: type[SyntheticCaptionGoldenEvidence] = SyntheticCaptionGoldenEvidence,
) -> ClipResolutionEvidence:
    if value is None:
        return _bundle_type()
    if type(value) is not _bundle_type:
        raise ClipResolutionError("evidence must be exactly ClipResolutionEvidence or None")
    runtime = None if value.runtime is None else _copy_runtime(value.runtime)
    loader = None
    if value.loader is not None:
        source = value.loader
        if type(source) is not _loader_type:
            raise ClipResolutionError("evidence.loader has the wrong exact type")
        loader = _loader_type(
            source.loader_class,
            source.tokenizer_class,
            source.local_files_only,
            source.trust_remote_code,
            source.text_model_called_once,
            source.exposes_final_ln_pooled_eos,
            source.exposes_pretrained_text_projection,
            source.vision_tower_instantiated,
            source.text_projection_requires_grad,
            source.active_text_dropout_sites,
            source.tokenizer_max_length,
        )
    open_trace = None
    if value.open_trace is not None:
        source_trace = value.open_trace
        if type(source_trace) is not _trace_type:
            raise ClipResolutionError("evidence.open_trace has the wrong exact type")
        open_trace = _trace_type(
            tuple(source_trace.allowlist_paths),
            tuple(source_trace.opened_paths),
            tuple(source_trace.outside_paths),
        )
    rights = None
    if value.rights is not None:
        source_rights = value.rights
        if type(source_rights) is not _rights_type:
            raise ClipResolutionError("evidence.rights has the wrong exact type")
        rights = _rights_type(
            source_rights.verdict,
            source_rights.review_sha256,
            source_rights.private_research_use,
            source_rights.redistribution_allowed,
            source_rights.commercial_use_allowed,
        )
    golden = None
    if value.golden is not None:
        source_golden = value.golden
        if type(source_golden) is not _golden_type:
            raise ClipResolutionError("evidence.golden has the wrong exact type")
        golden = _golden_type(
            source_golden.caption_count,
            source_golden.captions_ascii_sha256,
            source_golden.token_ids_sha256,
            source_golden.projected_float64_l2_sha256,
            source_golden.tokenizer_max_length,
            source_golden.projected_width,
        )
    return _bundle_type(runtime, loader, open_trace, rights, golden)


def _known_statuses() -> frozenset[str]:
    return frozenset(
        {
            "HOLD_ROOT_MISSING",
            "HOLD_ROOT_UNREADABLE",
            "HOLD_ROOT_NOT_DIRECTORY",
            "HOLD_ROOT_REPARSE",
            "HOLD_ROOT_UNRESOLVED",
            "HOLD_ROOT_NETWORK_PATH",
            "HOLD_SNAPSHOT_MISSING_FILES",
            "HOLD_SNAPSHOT_EXTRA_FILES",
            "HOLD_SNAPSHOT_CENSUS_MISMATCH",
            "HOLD_SNAPSHOT_ALIAS",
            "HOLD_FILE_MISSING",
            "HOLD_FILE_REPARSE",
            "HOLD_FILE_NONREGULAR",
            "HOLD_FILE_LINK_COUNT",
            "HOLD_FILE_IDENTITY_UNAVAILABLE",
            "HOLD_FILE_PATH_ESCAPE",
            "HOLD_FILE_SIZE_MISMATCH",
            "HOLD_FILE_SHA256_MISMATCH",
            "HOLD_FILE_TOCTOU",
            "HOLD_FILE_UNREADABLE",
            "HOLD_SNAPSHOT_TOTAL_BYTES_MISMATCH",
            "HOLD_SNAPSHOT_DOUBLE_PASS_MISMATCH",
            "HOLD_RUNTIME_WHEELS_ABSENT",
            "HOLD_TEXT_LOADER_EVIDENCE_ABSENT",
            "HOLD_OPEN_TRACE_EVIDENCE_ABSENT",
            "HOLD_RIGHTS_EVIDENCE_ABSENT",
            "HOLD_SYNTHETIC_GOLDEN_ABSENT",
            "HOLD_FRESH_REVIEW_REQUIRED",
        }
    )


def _validate_assessment(
    value: object,
    _assessment_type: type[ClipResolutionAssessment] = ClipResolutionAssessment,
    _snapshot_type: type[ClipFileSnapshot] = ClipFileSnapshot,
) -> ClipResolutionAssessment:
    if type(value) is not _assessment_type:
        raise ClipResolutionError("assessment has the wrong exact type")
    assessment = value
    if type(assessment.status) is not str or assessment.status not in _known_statuses():
        raise ClipResolutionError("assessment status is outside the exact HOLD vocabulary")
    if type(assessment.authority) is not int or assessment.authority != 0:
        raise ClipResolutionError("assessment authority must remain literal zero")
    if type(assessment.production) is not bool or assessment.production:
        raise ClipResolutionError("assessment production must remain literal false")
    if type(assessment.training_authorized) is not bool or assessment.training_authorized:
        raise ClipResolutionError("assessment training_authorized must remain literal false")
    if type(assessment.files) is not tuple:
        raise ClipResolutionError("assessment files must be an exact built-in tuple")
    seen_paths: set[str] = set()
    for index, item in enumerate(assessment.files):
        if type(item) is not _snapshot_type:
            raise ClipResolutionError(f"assessment.files[{index}] has the wrong exact type")
        _require_safe_name(item.path, f"assessment.files[{index}].path")
        _require_exact_int(item.bytes, f"assessment.files[{index}].bytes", minimum=0)
        _require_sha256(item.sha256, f"assessment.files[{index}].sha256")
        if item.path in seen_paths:
            raise ClipResolutionError("assessment files contain duplicate paths")
        seen_paths.add(item.path)
    _require_exact_int(assessment.total_bytes, "assessment.total_bytes", minimum=0)
    if assessment.total_bytes != sum(item.bytes for item in assessment.files):
        raise ClipResolutionError("assessment total_bytes disagrees with file snapshots")
    for label in ("missing_files", "extra_files"):
        names = getattr(assessment, label)
        if type(names) is not tuple:
            raise ClipResolutionError(f"assessment {label} must be an exact built-in tuple")
        for index, name in enumerate(names):
            _require_safe_name(name, f"assessment.{label}[{index}]")
        if tuple(sorted(names)) != names or len(set(names)) != len(names):
            raise ClipResolutionError(f"assessment {label} must be sorted and unique")
    _copy_evidence(assessment.evidence)
    return assessment


def _make_assessment(
    status: str,
    files: tuple[ClipFileSnapshot, ...],
    total_bytes: int,
    missing_files: tuple[str, ...],
    extra_files: tuple[str, ...],
    evidence: ClipResolutionEvidence,
    _assessment_type: type[ClipResolutionAssessment] = ClipResolutionAssessment,
) -> ClipResolutionAssessment:
    value = object.__new__(_assessment_type)
    object.__setattr__(value, "status", status)
    object.__setattr__(value, "authority", 0)
    object.__setattr__(value, "production", False)
    object.__setattr__(value, "training_authorized", False)
    object.__setattr__(value, "files", tuple(files))
    object.__setattr__(value, "total_bytes", total_bytes)
    object.__setattr__(value, "missing_files", tuple(sorted(missing_files)))
    object.__setattr__(value, "extra_files", tuple(sorted(extra_files)))
    object.__setattr__(value, "evidence", _copy_evidence(evidence))
    object.__setattr__(value, "_lease", None)
    return _validate_assessment(value)


def _stat_identity(
    value: object,
    _identity_type: type[_FileIdentity] = _FileIdentity,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> _FileIdentity:
    names = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_nlink",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    observed: list[int] = []
    for name in names:
        item = getattr(value, name, None)
        if type(item) is not int:
            raise _error_type(f"filesystem identity lacks exact {name}")
        observed.append(item)
    attributes = getattr(value, "st_file_attributes", 0)
    if type(attributes) is not int:
        raise _error_type("filesystem identity has invalid file attributes")
    return _identity_type(*observed, attributes)


def _is_reparse(
    identity: _FileIdentity,
    _is_link: Callable[[int], bool] = stat.S_ISLNK,
) -> bool:
    return _is_link(identity.mode) or bool(identity.attributes & 0x400)


def _same_path_and_handle_identity(left: _FileIdentity, right: _FileIdentity) -> bool:
    """Compare path and handle stats without Windows' incompatible ctime view."""

    return (
        left.device,
        left.inode,
        left.mode,
        left.links,
        left.size,
        left.mtime_ns,
        left.attributes,
    ) == (
        right.device,
        right.inode,
        right.mode,
        right.links,
        right.size,
        right.mtime_ns,
        right.attributes,
    )


def _real_ops() -> _FilesystemOps:
    return _FilesystemOps(os.lstat, os.scandir, os.open, os.fstat, os.read, os.close)


def _validate_pins(
    pins: tuple[ClipFilePin, ...],
    expected_total_bytes: int,
    _pin_type: type[ClipFilePin] = ClipFilePin,
    _exact_int: Callable[..., int] = _require_exact_int,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> tuple[ClipFilePin, ...]:
    if type(pins) is not tuple:
        raise _error_type("pins must be an exact built-in tuple")
    copied: list[ClipFilePin] = []
    for index, pin in enumerate(pins):
        if type(pin) is not _pin_type:
            raise _error_type(f"pins[{index}] has the wrong exact type")
        copied.append(_pin_type(pin.path, pin.bytes, pin.sha256))
    exact_names = (
        "pytorch_model.bin",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    )
    if tuple(pin.path for pin in copied) != exact_names:
        raise _error_type("pins must use the exact eight-file canonical order")
    if len({pin.path.casefold() for pin in copied}) != 8:
        raise _error_type("pins contain path aliases")
    _exact_int(expected_total_bytes, "expected_total_bytes", minimum=1)
    if sum(pin.bytes for pin in copied) != expected_total_bytes:
        raise _error_type("pin bytes do not sum to expected_total_bytes")
    return tuple(copied)


def _component_status(
    identity: _FileIdentity,
    *,
    final: bool,
    _reparse: Callable[[_FileIdentity], bool] = _is_reparse,
    _is_directory: Callable[[int], bool] = stat.S_ISDIR,
) -> str | None:
    if _reparse(identity):
        return "HOLD_ROOT_REPARSE"
    if not _is_directory(identity.mode):
        return "HOLD_ROOT_NOT_DIRECTORY" if final else "HOLD_ROOT_UNRESOLVED"
    return None


def _check_root_components(
    root: Path,
    ops: _FilesystemOps,
    _identity: Callable[[object], _FileIdentity] = _stat_identity,
    _status: Callable[..., str | None] = _component_status,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> tuple[str | None, _FileIdentity | None]:
    if not root.is_absolute():
        raise _error_type("internal root path must be absolute")
    current = root.__class__(root.anchor)
    parts = root.parts[1:]
    if not parts:
        parts = ()
    final_identity: _FileIdentity | None = None
    paths = [current]
    paths.extend(current.joinpath(*parts[: index + 1]) for index in range(len(parts)))
    for index, component in enumerate(paths):
        try:
            identity = _identity(ops.lstat(component))
        except FileNotFoundError:
            return "HOLD_ROOT_MISSING", None
        except (OSError, _error_type):
            return "HOLD_ROOT_UNREADABLE", None
        status = _status(identity, final=index == len(paths) - 1)
        if status is not None:
            return status, None
        final_identity = identity
    return None, final_identity


def _resolve_root(
    root: Path,
    ops: _FilesystemOps,
    _absolute_path: Callable[[object], str] = os.path.abspath,
    _filesystem_path: Callable[[object], str] = os.fspath,
    _check_components: Callable[
        [Path, _FilesystemOps], tuple[str | None, _FileIdentity | None]
    ] = _check_root_components,
) -> tuple[str | None, Path | None, _FileIdentity | None]:
    try:
        lexical = root.__class__(_absolute_path(_filesystem_path(root)))
    except (OSError, TypeError, ValueError):
        return "HOLD_ROOT_UNRESOLVED", None, None
    text = _filesystem_path(lexical)
    if lexical.anchor.startswith("\\\\") or text.startswith("\\\\") or text.startswith("//"):
        return "HOLD_ROOT_NETWORK_PATH", None, None
    status, lexical_identity = _check_components(lexical, ops)
    if status is not None:
        return status, None, None
    try:
        resolved = lexical.resolve(strict=True)
    except (OSError, RuntimeError):
        return "HOLD_ROOT_UNRESOLVED", None, None
    if not resolved.is_absolute():
        return "HOLD_ROOT_UNRESOLVED", None, None
    status, resolved_identity = _check_components(resolved, ops)
    if status is not None:
        return status, None, None
    if lexical_identity != resolved_identity:
        return "HOLD_ROOT_REPARSE", None, None
    return None, resolved, resolved_identity


def _hash_one_file(
    root: Path,
    pin: ClipFilePin,
    ops: _FilesystemOps,
    _identity: Callable[[object], _FileIdentity] = _stat_identity,
    _reparse: Callable[[_FileIdentity], bool] = _is_reparse,
    _is_regular: Callable[[int], bool] = stat.S_ISREG,
    _same_identity: Callable[[_FileIdentity, _FileIdentity], bool] = (
        _same_path_and_handle_identity
    ),
    _sha256_factory: Callable[[], Any] = hashlib.sha256,
    _snapshot: Callable[[str, int, str], ClipFileSnapshot] = _make_file_snapshot,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
    _open_flags: int = (
        os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    ),
) -> tuple[str | None, ClipFileSnapshot | None, _FileIdentity | None]:
    candidate = root / pin.path
    try:
        before = _identity(ops.lstat(candidate))
    except FileNotFoundError:
        return "HOLD_FILE_MISSING", None, None
    except (OSError, _error_type):
        return "HOLD_FILE_UNREADABLE", None, None
    if _reparse(before):
        return "HOLD_FILE_REPARSE", None, None
    if not _is_regular(before.mode):
        return "HOLD_FILE_NONREGULAR", None, None
    if before.links != 1:
        return "HOLD_FILE_LINK_COUNT", None, None
    if before.device <= 0 or before.inode <= 0:
        return "HOLD_FILE_IDENTITY_UNAVAILABLE", None, None
    if before.size != pin.bytes:
        return "HOLD_FILE_SIZE_MISMATCH", None, None
    try:
        resolved_candidate = candidate.resolve(strict=True)
    except (OSError, RuntimeError):
        return "HOLD_FILE_UNREADABLE", None, None
    if resolved_candidate.parent != root or resolved_candidate.name != pin.path:
        return "HOLD_FILE_PATH_ESCAPE", None, None

    fd: int | None = None
    status: str | None = None
    digest = _sha256_factory()
    byte_count = 0
    opened: _FileIdentity | None = None
    after_fd: _FileIdentity | None = None
    after_path: _FileIdentity | None = None
    try:
        fd = ops.open_fd(candidate, _open_flags)
        opened = _identity(ops.fstat(fd))
        if not _same_identity(before, opened):
            status = "HOLD_FILE_TOCTOU"
        elif _reparse(opened) or not _is_regular(opened.mode):
            status = "HOLD_FILE_NONREGULAR"
        elif opened.links != 1:
            status = "HOLD_FILE_LINK_COUNT"
        else:
            while True:
                chunk = ops.read_fd(fd, 1_048_576)
                if type(chunk) is not bytes:
                    status = "HOLD_FILE_UNREADABLE"
                    break
                if not chunk:
                    break
                byte_count += len(chunk)
                if byte_count > pin.bytes:
                    status = "HOLD_FILE_SIZE_MISMATCH"
                    break
                digest.update(chunk)
            after_fd = _identity(ops.fstat(fd))
            after_path = _identity(ops.lstat(candidate))
            if status is None:
                stable_path = before == after_path
                stable_handle = opened == after_fd
                same_opened_object = _same_identity(before, opened)
                same_closed_object = _same_identity(after_path, after_fd)
                if not (
                    stable_path
                    and stable_handle
                    and same_opened_object
                    and same_closed_object
                ):
                    status = "HOLD_FILE_TOCTOU"
    except FileNotFoundError:
        status = "HOLD_FILE_MISSING"
    except (OSError, _error_type):
        status = "HOLD_FILE_UNREADABLE"
    finally:
        if fd is not None:
            try:
                ops.close_fd(fd)
            except OSError:
                status = "HOLD_FILE_UNREADABLE"
    if status is not None:
        return status, None, None
    if byte_count != pin.bytes:
        return "HOLD_FILE_SIZE_MISMATCH", None, None
    observed_sha256 = digest.hexdigest()
    snapshot = _snapshot(pin.path, byte_count, observed_sha256)
    if observed_sha256 != pin.sha256:
        return "HOLD_FILE_SHA256_MISMATCH", snapshot, before
    return None, snapshot, before


def _read_directory_census(
    root: Path,
    pins: tuple[ClipFilePin, ...],
    ops: _FilesystemOps,
) -> tuple[str | None, tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    try:
        with ops.scandir(root) as iterator:
            names = tuple(entry.name for entry in iterator)
    except OSError:
        return "HOLD_ROOT_UNREADABLE", (), (), ()
    if any(type(name) is not str for name in names):
        return "HOLD_SNAPSHOT_ALIAS", (), (), ()
    census = tuple(sorted(names))
    expected = {pin.path for pin in pins}
    actual = set(census)
    missing = tuple(sorted(expected - actual))
    extra = tuple(sorted(actual - expected))
    if len({name.casefold() for name in census}) != len(census):
        return "HOLD_SNAPSHOT_ALIAS", census, missing, extra
    if missing and extra:
        return "HOLD_SNAPSHOT_CENSUS_MISMATCH", census, missing, extra
    if missing:
        return "HOLD_SNAPSHOT_MISSING_FILES", census, missing, ()
    if extra:
        return "HOLD_SNAPSHOT_EXTRA_FILES", census, (), extra
    return None, census, (), ()


def _scan_snapshot_once(
    root: Path,
    pins: tuple[ClipFilePin, ...],
    expected_total_bytes: int,
    ops: _FilesystemOps,
    _resolve: Callable[
        [Path, _FilesystemOps], tuple[str | None, Path | None, _FileIdentity | None]
    ] = _resolve_root,
    _hash_file: Callable[
        [Path, ClipFilePin, _FilesystemOps],
        tuple[str | None, ClipFileSnapshot | None, _FileIdentity | None],
    ] = _hash_one_file,
    _census: Callable[
        [Path, tuple[ClipFilePin, ...], _FilesystemOps],
        tuple[str | None, tuple[str, ...], tuple[str, ...], tuple[str, ...]],
    ] = _read_directory_census,
    _identity: Callable[[object], _FileIdentity] = _stat_identity,
    _reparse: Callable[[_FileIdentity], bool] = _is_reparse,
    _is_regular: Callable[[int], bool] = stat.S_ISREG,
    _outcome_type: type[_ScanOutcome] = _ScanOutcome,
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> _ScanOutcome:
    status, resolved_root, root_before = _resolve(root, ops)
    if status is not None or resolved_root is None or root_before is None:
        return _outcome_type(status, (), 0, (), ())
    census_status, census, missing, extra = _census(resolved_root, pins, ops)
    if census_status is not None:
        return _outcome_type(
            census_status,
            (),
            0,
            missing,
            extra,
            root_before=root_before,
            census=census,
        )

    snapshots: list[ClipFileSnapshot] = []
    identities: list[_FileIdentity] = []
    identity_keys: set[tuple[int, int]] = set()
    total_bytes = 0
    for pin in pins:
        item_status, snapshot, identity = _hash_file(resolved_root, pin, ops)
        if snapshot is not None:
            snapshots.append(snapshot)
            total_bytes += snapshot.bytes
        if identity is not None:
            key = (identity.device, identity.inode)
            if key in identity_keys and item_status is None:
                return _outcome_type(
                    "HOLD_SNAPSHOT_ALIAS",
                    tuple(snapshots),
                    total_bytes,
                    (),
                    (),
                    root_before=root_before,
                    file_identities=tuple(identities),
                    census=census,
                )
            identity_keys.add(key)
            identities.append(identity)
        if item_status is not None:
            return _outcome_type(
                item_status,
                tuple(snapshots),
                total_bytes,
                (),
                (),
                root_before=root_before,
                file_identities=tuple(identities),
                census=census,
            )
        if identity is None:
            return _outcome_type(
                "HOLD_SNAPSHOT_ALIAS",
                tuple(snapshots),
                total_bytes,
                (),
                (),
                root_before=root_before,
                file_identities=tuple(identities),
                census=census,
            )

    closing_status, closing_census, closing_missing, closing_extra = _census(
        resolved_root, pins, ops
    )
    if (
        closing_status is not None
        or closing_census != census
        or closing_missing
        or closing_extra
    ):
        return _outcome_type(
            "HOLD_FILE_TOCTOU",
            tuple(snapshots),
            total_bytes,
            (),
            (),
            root_before=root_before,
            file_identities=tuple(identities),
            census=census,
        )
    for pin, original_identity in zip(pins, identities, strict=True):
        try:
            closing_identity = _identity(ops.lstat(resolved_root / pin.path))
        except (FileNotFoundError, OSError, _error_type):
            return _outcome_type(
                "HOLD_FILE_TOCTOU",
                tuple(snapshots),
                total_bytes,
                (),
                (),
                root_before=root_before,
                file_identities=tuple(identities),
                census=census,
            )
        if (
            closing_identity != original_identity
            or _reparse(closing_identity)
            or not _is_regular(closing_identity.mode)
            or closing_identity.links != 1
        ):
            return _outcome_type(
                "HOLD_FILE_TOCTOU",
                tuple(snapshots),
                total_bytes,
                (),
                (),
                root_before=root_before,
                file_identities=tuple(identities),
                census=census,
            )
    status, checked_root, root_after = _resolve(resolved_root, ops)
    if status is not None or checked_root != resolved_root or root_after != root_before:
        return _outcome_type(
            "HOLD_FILE_TOCTOU",
            tuple(snapshots),
            total_bytes,
            (),
            (),
            root_before=root_before,
            root_after=root_after,
            file_identities=tuple(identities),
            census=census,
        )
    if total_bytes != expected_total_bytes:
        return _outcome_type(
            "HOLD_SNAPSHOT_TOTAL_BYTES_MISMATCH",
            tuple(snapshots),
            total_bytes,
            (),
            (),
            root_before=root_before,
            root_after=root_after,
            file_identities=tuple(identities),
            census=census,
        )
    return _outcome_type(
        None,
        tuple(snapshots),
        total_bytes,
        (),
        (),
        root_before=root_before,
        root_after=root_after,
        file_identities=tuple(identities),
        census=census,
    )


def _scan_snapshot(
    root: Path,
    pins: tuple[ClipFilePin, ...],
    expected_total_bytes: int,
    ops: _FilesystemOps,
    _single_pass: Callable[
        [Path, tuple[ClipFilePin, ...], int, _FilesystemOps], _ScanOutcome
    ] = _scan_snapshot_once,
    _outcome_type: type[_ScanOutcome] = _ScanOutcome,
) -> _ScanOutcome:
    """Require two equal full scans; this is a stability check, not an OS-atomic seal."""

    first = _single_pass(root, pins, expected_total_bytes, ops)
    if first.status is not None:
        return first
    second = _single_pass(root, pins, expected_total_bytes, ops)
    if second.status is not None:
        return second
    if first != second:
        return _outcome_type(
            "HOLD_SNAPSHOT_DOUBLE_PASS_MISMATCH",
            first.files,
            first.total_bytes,
            first.missing_files,
            first.extra_files,
            root_before=first.root_before,
            root_after=first.root_after,
            file_identities=first.file_identities,
            census=first.census,
        )
    return _outcome_type(
        None,
        first.files,
        first.total_bytes,
        first.missing_files,
        first.extra_files,
        root_before=first.root_before,
        root_after=first.root_after,
        file_identities=first.file_identities,
        census=first.census,
        double_pass_verified=True,
    )


def _evidence_hold_status(evidence: ClipResolutionEvidence) -> str:
    if evidence.runtime is None:
        return "HOLD_RUNTIME_WHEELS_ABSENT"
    if evidence.loader is None:
        return "HOLD_TEXT_LOADER_EVIDENCE_ABSENT"
    if evidence.open_trace is None:
        return "HOLD_OPEN_TRACE_EVIDENCE_ABSENT"
    if evidence.rights is None:
        return "HOLD_RIGHTS_EVIDENCE_ABSENT"
    if evidence.golden is None:
        return "HOLD_SYNTHETIC_GOLDEN_ABSENT"
    return "HOLD_FRESH_REVIEW_REQUIRED"


def _legacy_assess_local_clip_snapshot_for_tests(
    root: Path,
    *,
    pins: tuple[ClipFilePin, ...],
    expected_total_bytes: int,
    evidence: ClipResolutionEvidence | None = None,
    operations: _FilesystemOps | None = None,
) -> ClipResolutionAssessment:
    """Private seam for tiny files and deterministic filesystem-race tests."""

    if type(root) is not type(Path()):
        raise TypeError("root must be an exact platform pathlib.Path")
    checked_pins = _validate_pins(pins, expected_total_bytes)
    checked_evidence = _copy_evidence(evidence)
    if operations is None:
        checked_operations = _real_ops()
    elif type(operations) is _FilesystemOps:
        checked_operations = operations
    else:
        raise TypeError("operations must be exactly _FilesystemOps or None")
    outcome = _scan_snapshot(
        root,
        checked_pins,
        expected_total_bytes,
        checked_operations,
    )
    status = outcome.status or _evidence_hold_status(checked_evidence)
    return _make_assessment(
        status,
        outcome.files,
        outcome.total_bytes,
        outcome.missing_files,
        outcome.extra_files,
        checked_evidence,
    )


def _legacy_public_assess_local_clip_snapshot(root: Path) -> ClipResolutionAssessment:
    """Assess one existing local snapshot without accepting injected identities.

    The root must be the exact platform ``pathlib.Path`` type.  This public call
    has no hash, metadata, evidence, download, loader, or network injection
    parameters.  Consequently, a valid tree stops at missing-evidence HOLD.
    """

    if type(root) is not type(Path()):
        raise TypeError("root must be an exact platform pathlib.Path")
    pins = (
        ClipFilePin(
            "pytorch_model.bin",
            605_247_071,
            "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
        ),
        ClipFilePin(
            "config.json",
            4_186,
            "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
        ),
        ClipFilePin(
            "merges.txt",
            524_657,
            "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
        ),
        ClipFilePin(
            "preprocessor_config.json",
            316,
            "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
        ),
        ClipFilePin(
            "special_tokens_map.json",
            389,
            "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
        ),
        ClipFilePin(
            "tokenizer_config.json",
            592,
            "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
        ),
        ClipFilePin(
            "tokenizer.json",
            2_224_041,
            "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
        ),
        ClipFilePin(
            "vocab.json",
            862_328,
            "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
        ),
    )
    return _legacy_assess_local_clip_snapshot_for_tests(
        root,
        pins=pins,
        expected_total_bytes=608_863_580,
    )


def _wheel_payload(value: WheelIdentity) -> dict[str, object]:
    return {
        "bytes": value.bytes,
        "distribution": value.distribution,
        "filename": value.filename,
        "sha256": value.sha256,
    }


def _evidence_payload(value: ClipResolutionEvidence) -> dict[str, object]:
    runtime: object = None
    if value.runtime is not None:
        runtime = {"wheels": [_wheel_payload(wheel) for wheel in value.runtime.wheels]}
    loader: object = None
    if value.loader is not None:
        loader = {
            "active_text_dropout_sites": value.loader.active_text_dropout_sites,
            "exposes_final_ln_pooled_eos": value.loader.exposes_final_ln_pooled_eos,
            "exposes_pretrained_text_projection": (
                value.loader.exposes_pretrained_text_projection
            ),
            "loader_class": value.loader.loader_class,
            "local_files_only": value.loader.local_files_only,
            "text_model_called_once": value.loader.text_model_called_once,
            "text_projection_requires_grad": value.loader.text_projection_requires_grad,
            "tokenizer_class": value.loader.tokenizer_class,
            "tokenizer_max_length": value.loader.tokenizer_max_length,
            "trust_remote_code": value.loader.trust_remote_code,
            "vision_tower_instantiated": value.loader.vision_tower_instantiated,
        }
    open_trace: object = None
    if value.open_trace is not None:
        open_trace = {
            "allowlist_paths": list(value.open_trace.allowlist_paths),
            "opened_paths": list(value.open_trace.opened_paths),
            "outside_paths": list(value.open_trace.outside_paths),
        }
    rights: object = None
    if value.rights is not None:
        rights = {
            "commercial_use_allowed": value.rights.commercial_use_allowed,
            "private_research_use": value.rights.private_research_use,
            "redistribution_allowed": value.rights.redistribution_allowed,
            "review_sha256": value.rights.review_sha256,
            "verdict": value.rights.verdict,
        }
    golden: object = None
    if value.golden is not None:
        golden = {
            "caption_count": value.golden.caption_count,
            "captions_ascii_sha256": value.golden.captions_ascii_sha256,
            "projected_float64_l2_sha256": value.golden.projected_float64_l2_sha256,
            "projected_width": value.golden.projected_width,
            "token_ids_sha256": value.golden.token_ids_sha256,
            "tokenizer_max_length": value.golden.tokenizer_max_length,
        }
    return {
        "golden": golden,
        "loader": loader,
        "open_trace": open_trace,
        "rights": rights,
        "runtime": runtime,
    }


def _legacy_canonical_assessment_bytes(assessment: ClipResolutionAssessment) -> bytes:
    """Serialize a revalidated assessment as exact-key canonical JSON plus LF."""

    checked = _validate_assessment(assessment)
    payload = {
        "authority": checked.authority,
        "evidence": _evidence_payload(_copy_evidence(checked.evidence)),
        "extra_files": list(checked.extra_files),
        "files": [
            {"bytes": item.bytes, "path": item.path, "sha256": item.sha256}
            for item in checked.files
        ],
        "missing_files": list(checked.missing_files),
        "model_id": "openai/clip-vit-base-patch32",
        "production": checked.production,
        "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        "schema": "phasepair-clip-resolution-preflight-v1",
        "status": checked.status,
        "total_bytes": checked.total_bytes,
        "training_authorized": checked.training_authorized,
    }
    if set(payload) != {
        "authority",
        "evidence",
        "extra_files",
        "files",
        "missing_files",
        "model_id",
        "production",
        "revision",
        "schema",
        "status",
        "total_bytes",
        "training_authorized",
    }:
        raise AssertionError("internal canonical assessment key census mismatch")
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _seal_clip_resolution_api(
    _path_type: type[Path] = type(Path()),
    _pin_type: type[ClipFilePin] = ClipFilePin,
    _wheel_type: type[WheelIdentity] = WheelIdentity,
    _runtime_type: type[RuntimeWheelEvidence] = RuntimeWheelEvidence,
    _loader_type: type[TextLoaderEvidence] = TextLoaderEvidence,
    _trace_type: type[OpenTraceEvidence] = OpenTraceEvidence,
    _rights_type: type[RightsEvidence] = RightsEvidence,
    _golden_type: type[SyntheticCaptionGoldenEvidence] = SyntheticCaptionGoldenEvidence,
    _bundle_type: type[ClipResolutionEvidence] = ClipResolutionEvidence,
    _snapshot_type: type[ClipFileSnapshot] = ClipFileSnapshot,
    _assessment_type: type[ClipResolutionAssessment] = ClipResolutionAssessment,
    _lease_type: type[_AssessmentLease] = _AssessmentLease,
    _identity_type: type[_FileIdentity] = _FileIdentity,
    _ops_type: type[_FilesystemOps] = _FilesystemOps,
    _outcome_type: type[_ScanOutcome] = _ScanOutcome,
    _scan_function: Callable[
        [Path, tuple[ClipFilePin, ...], int, _FilesystemOps], _ScanOutcome
    ] = _scan_snapshot,
    _sha256_factory: Callable[..., Any] = hashlib.sha256,
    _json_dumps: Callable[..., str] = json.dumps,
    _filesystem_path: Callable[[object], str] = os.fspath,
    _absolute_path: Callable[[object], str] = os.path.abspath,
    _weak_registry_type: type[weakref.WeakKeyDictionary[Any, Any]] = (
        weakref.WeakKeyDictionary
    ),
    _real_operations: _FilesystemOps = _real_ops(),
    _error_type: type[ClipResolutionError] = ClipResolutionError,
) -> tuple[
    Callable[[Path], ClipResolutionAssessment],
    Callable[..., ClipResolutionAssessment],
    Callable[[ClipResolutionAssessment], bytes],
    Callable[[], int],
]:
    """Create the one-time, closed-over authority-zero API implementation."""

    exact_names = (
        "pytorch_model.bin",
        "config.json",
        "merges.txt",
        "preprocessor_config.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "vocab.json",
    )
    scan_statuses = frozenset(
        {
            "HOLD_ROOT_MISSING",
            "HOLD_ROOT_UNREADABLE",
            "HOLD_ROOT_NOT_DIRECTORY",
            "HOLD_ROOT_REPARSE",
            "HOLD_ROOT_UNRESOLVED",
            "HOLD_ROOT_NETWORK_PATH",
            "HOLD_SNAPSHOT_MISSING_FILES",
            "HOLD_SNAPSHOT_EXTRA_FILES",
            "HOLD_SNAPSHOT_CENSUS_MISMATCH",
            "HOLD_SNAPSHOT_ALIAS",
            "HOLD_FILE_MISSING",
            "HOLD_FILE_REPARSE",
            "HOLD_FILE_NONREGULAR",
            "HOLD_FILE_LINK_COUNT",
            "HOLD_FILE_IDENTITY_UNAVAILABLE",
            "HOLD_FILE_PATH_ESCAPE",
            "HOLD_FILE_SIZE_MISMATCH",
            "HOLD_FILE_SHA256_MISMATCH",
            "HOLD_FILE_TOCTOU",
            "HOLD_FILE_UNREADABLE",
            "HOLD_SNAPSHOT_TOTAL_BYTES_MISMATCH",
            "HOLD_SNAPSHOT_DOUBLE_PASS_MISMATCH",
        }
    )
    root_statuses = frozenset(
        {
            "HOLD_ROOT_MISSING",
            "HOLD_ROOT_UNREADABLE",
            "HOLD_ROOT_NOT_DIRECTORY",
            "HOLD_ROOT_REPARSE",
            "HOLD_ROOT_UNRESOLVED",
            "HOLD_ROOT_NETWORK_PATH",
        }
    )
    canonical_keys = frozenset(
        {
            "authority",
            "evidence",
            "extra_files",
            "files",
            "missing_files",
            "model_id",
            "production",
            "revision",
            "schema",
            "status",
            "total_bytes",
            "training_authorized",
        }
    )
    production_specs = (
        (
            "pytorch_model.bin",
            605_247_071,
            "a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f",
        ),
        (
            "config.json",
            4_186,
            "b575ef3c36f2a057fa19e221650105052d61cc9c1a972ec15019c6261ec98770",
        ),
        (
            "merges.txt",
            524_657,
            "f526393189112391ce6f9795d4695f704121ce452c3aad1f5335cc41337eba85",
        ),
        (
            "preprocessor_config.json",
            316,
            "910e70b3956ac9879ebc90b22fb3bc8a75b6a0677814500101a4c072bd7857bd",
        ),
        (
            "special_tokens_map.json",
            389,
            "f8c0d6c39aee3f8431078ef6646567b0aba7f2246e9c54b8b99d55c22b707cbf",
        ),
        (
            "tokenizer_config.json",
            592,
            "34b7336e4bee12e0a9730eaf5189f582ef3c3eea5027f65730e5717256755aad",
        ),
        (
            "tokenizer.json",
            2_224_041,
            "b556ac8c99757ffb677208af34bc8c6721572114111a6e0aaf5fa69ff0b8d842",
        ),
        (
            "vocab.json",
            862_328,
            "5047b556ce86ccaf6aa22b3ffccfc52d391ea4accdab9c2f2407da5b742d4363",
        ),
    )

    # Assessments own leases strongly; this map owns only weak lease keys.  Its
    # values are primitive provenance, a digest, and an integer owner id, never
    # an assessment or lease reference.
    issued: weakref.WeakKeyDictionary[_AssessmentLease, tuple[object, ...]] = (
        _weak_registry_type()
    )

    def exact_str(value: object, label: str) -> str:
        if type(value) is not str or not value:
            raise _error_type(f"{label} must be an exact nonempty built-in str")
        return value

    def exact_bool(value: object, label: str) -> bool:
        if type(value) is not bool:
            raise _error_type(f"{label} must be an exact built-in bool")
        return value

    def exact_int(value: object, label: str, minimum: int = 0) -> int:
        if type(value) is not int or value < minimum:
            raise _error_type(f"{label} must be an exact built-in int >= {minimum}")
        return value

    def sha256_hex(value: object, label: str) -> str:
        text = exact_str(value, label)
        if len(text) != 64 or any(
            character not in "0123456789abcdef" for character in text
        ):
            raise _error_type(f"{label} must be lowercase SHA-256 hex")
        return text

    def safe_name(value: object, label: str) -> str:
        name = exact_str(value, label)
        if name in {".", ".."} or "\x00" in name or "/" in name or "\\" in name:
            raise _error_type(f"{label} must be one safe relative basename")
        return name

    def raw_instance(instance_type: type[Any], values: tuple[tuple[str, object], ...]) -> Any:
        instance = object.__new__(instance_type)
        for field, value in values:
            object.__setattr__(instance, field, value)
        return instance

    def pin_from_spec(spec: tuple[str, int, str]) -> ClipFilePin:
        return raw_instance(
            _pin_type,
            (("path", spec[0]), ("bytes", spec[1]), ("sha256", spec[2])),
        )

    def freeze_pins(
        pins: tuple[ClipFilePin, ...], expected_total_bytes: int
    ) -> tuple[tuple[tuple[str, int, str], ...], tuple[ClipFilePin, ...]]:
        if type(pins) is not tuple:
            raise _error_type("pins must be an exact built-in tuple")
        specs: list[tuple[str, int, str]] = []
        for index, pin in enumerate(pins):
            if type(pin) is not _pin_type:
                raise _error_type(f"pins[{index}] has the wrong exact type")
            specs.append(
                (
                    safe_name(pin.path, f"pins[{index}].path"),
                    exact_int(pin.bytes, f"pins[{index}].bytes", 1),
                    sha256_hex(pin.sha256, f"pins[{index}].sha256"),
                )
            )
        frozen = tuple(specs)
        if tuple(spec[0] for spec in frozen) != exact_names:
            raise _error_type("pins must use the exact eight-file canonical order")
        if len({spec[0].casefold() for spec in frozen}) != 8:
            raise _error_type("pins contain path aliases")
        exact_int(expected_total_bytes, "expected_total_bytes", 1)
        if sum(spec[1] for spec in frozen) != expected_total_bytes:
            raise _error_type("pin bytes do not sum to expected_total_bytes")
        return frozen, tuple(pin_from_spec(spec) for spec in frozen)

    def freeze_evidence(value: ClipResolutionEvidence | None) -> tuple[object, ...]:
        if value is None:
            return (None, None, None, None, None)
        if type(value) is not _bundle_type:
            raise _error_type("evidence must be exactly ClipResolutionEvidence or None")

        runtime: object = None
        if value.runtime is not None:
            if type(value.runtime) is not _runtime_type:
                raise _error_type("evidence.runtime has the wrong exact type")
            if type(value.runtime.wheels) is not tuple:
                raise _error_type("runtime.wheels must be an exact built-in tuple")
            wheel_specs: list[tuple[str, str, int, str]] = []
            for index, wheel in enumerate(value.runtime.wheels):
                if type(wheel) is not _wheel_type:
                    raise _error_type(f"runtime.wheels[{index}] has the wrong exact type")
                distribution = exact_str(
                    wheel.distribution, f"runtime.wheels[{index}].distribution"
                )
                filename = safe_name(wheel.filename, f"runtime.wheels[{index}].filename")
                if not filename.endswith(".whl"):
                    raise _error_type("runtime wheel filename must end in .whl")
                wheel_specs.append(
                    (
                        distribution,
                        filename,
                        exact_int(wheel.bytes, f"runtime.wheels[{index}].bytes", 1),
                        sha256_hex(wheel.sha256, f"runtime.wheels[{index}].sha256"),
                    )
                )
            runtime = tuple(wheel_specs)
            if tuple(spec[0] for spec in runtime) != (
                "transformers",
                "huggingface-hub",
                "tokenizers",
            ):
                raise _error_type("runtime wheels must be in the exact canonical order")
            if len({spec[1].casefold() for spec in runtime}) != 3:
                raise _error_type("runtime wheel filenames must be unique")

        loader: object = None
        if value.loader is not None:
            if type(value.loader) is not _loader_type:
                raise _error_type("evidence.loader has the wrong exact type")
            loader_class = exact_str(value.loader.loader_class, "loader.loader_class")
            tokenizer_class = exact_str(
                value.loader.tokenizer_class, "loader.tokenizer_class"
            )
            if loader_class != "CLIPTextModelWithProjection":
                raise _error_type("loader_class must be CLIPTextModelWithProjection")
            if tokenizer_class != "CLIPTokenizerFast":
                raise _error_type("tokenizer_class must be CLIPTokenizerFast")
            loader = (
                loader_class,
                tokenizer_class,
                exact_bool(value.loader.local_files_only, "loader.local_files_only"),
                exact_bool(value.loader.trust_remote_code, "loader.trust_remote_code"),
                exact_bool(
                    value.loader.text_model_called_once, "loader.text_model_called_once"
                ),
                exact_bool(
                    value.loader.exposes_final_ln_pooled_eos,
                    "loader.exposes_final_ln_pooled_eos",
                ),
                exact_bool(
                    value.loader.exposes_pretrained_text_projection,
                    "loader.exposes_pretrained_text_projection",
                ),
                exact_bool(
                    value.loader.vision_tower_instantiated,
                    "loader.vision_tower_instantiated",
                ),
                exact_bool(
                    value.loader.text_projection_requires_grad,
                    "loader.text_projection_requires_grad",
                ),
                exact_int(
                    value.loader.active_text_dropout_sites,
                    "loader.active_text_dropout_sites",
                ),
                exact_int(
                    value.loader.tokenizer_max_length, "loader.tokenizer_max_length", 1
                ),
            )
            if (
                not loader[2]
                or loader[3]
                or not loader[4]
                or not loader[5]
                or not loader[6]
                or loader[7]
                or loader[8]
                or loader[9] != 0
                or loader[10] != 77
            ):
                raise _error_type("loader evidence contradicts the frozen text-only path")

        open_trace: object = None
        if value.open_trace is not None:
            if type(value.open_trace) is not _trace_type:
                raise _error_type("evidence.open_trace has the wrong exact type")
            trace_parts: list[tuple[str, ...]] = []
            for label, paths in (
                ("allowlist_paths", value.open_trace.allowlist_paths),
                ("opened_paths", value.open_trace.opened_paths),
                ("outside_paths", value.open_trace.outside_paths),
            ):
                if type(paths) is not tuple:
                    raise _error_type(f"trace.{label} must be an exact built-in tuple")
                checked = tuple(
                    safe_name(path, f"trace.{label}[{index}]")
                    for index, path in enumerate(paths)
                )
                if len(set(checked)) != len(checked):
                    raise _error_type(f"trace.{label} must not contain duplicates")
                trace_parts.append(checked)
            open_trace = tuple(trace_parts)
            if open_trace[0] != exact_names:
                raise _error_type("trace allowlist must equal the exact eight-file order")
            if not open_trace[1] or any(path not in exact_names for path in open_trace[1]):
                raise _error_type("trace opened paths must be a nonempty allowlist subset")
            if open_trace[2]:
                raise _error_type("trace outside_paths must be empty")

        rights: object = None
        if value.rights is not None:
            if type(value.rights) is not _rights_type:
                raise _error_type("evidence.rights has the wrong exact type")
            verdict = exact_str(value.rights.verdict, "rights.verdict")
            rights = (
                verdict,
                sha256_hex(value.rights.review_sha256, "rights.review_sha256"),
                exact_bool(value.rights.private_research_use, "rights.private_research_use"),
                exact_bool(
                    value.rights.redistribution_allowed,
                    "rights.redistribution_allowed",
                ),
                exact_bool(
                    value.rights.commercial_use_allowed,
                    "rights.commercial_use_allowed",
                ),
            )
            if (
                rights[0] != "PRIVATE_RESEARCH_ONLY_NO_REDISTRIBUTION"
                or not rights[2]
                or rights[3]
                or rights[4]
            ):
                raise _error_type("rights evidence contradicts the frozen verdict")

        golden: object = None
        if value.golden is not None:
            if type(value.golden) is not _golden_type:
                raise _error_type("evidence.golden has the wrong exact type")
            golden = (
                exact_int(value.golden.caption_count, "golden.caption_count", 1),
                sha256_hex(
                    value.golden.captions_ascii_sha256,
                    "golden.captions_ascii_sha256",
                ),
                sha256_hex(value.golden.token_ids_sha256, "golden.token_ids_sha256"),
                sha256_hex(
                    value.golden.projected_float64_l2_sha256,
                    "golden.projected_float64_l2_sha256",
                ),
                exact_int(
                    value.golden.tokenizer_max_length,
                    "golden.tokenizer_max_length",
                    1,
                ),
                exact_int(value.golden.projected_width, "golden.projected_width", 1),
            )
            if golden[0] != 3 or golden[4] != 77 or golden[5] != 512:
                raise _error_type("golden evidence contradicts the frozen dimensions")
        return (runtime, loader, open_trace, rights, golden)

    def build_evidence(frozen: tuple[object, ...]) -> ClipResolutionEvidence:
        runtime_spec, loader_spec, trace_spec, rights_spec, golden_spec = frozen
        runtime = None
        if runtime_spec is not None:
            wheels = tuple(
                raw_instance(
                    _wheel_type,
                    (
                        ("distribution", spec[0]),
                        ("filename", spec[1]),
                        ("bytes", spec[2]),
                        ("sha256", spec[3]),
                    ),
                )
                for spec in runtime_spec
            )
            runtime = raw_instance(_runtime_type, (("wheels", wheels),))
        loader = None
        if loader_spec is not None:
            loader = raw_instance(
                _loader_type,
                tuple(
                    zip(
                        (
                            "loader_class",
                            "tokenizer_class",
                            "local_files_only",
                            "trust_remote_code",
                            "text_model_called_once",
                            "exposes_final_ln_pooled_eos",
                            "exposes_pretrained_text_projection",
                            "vision_tower_instantiated",
                            "text_projection_requires_grad",
                            "active_text_dropout_sites",
                            "tokenizer_max_length",
                        ),
                        loader_spec,
                        strict=True,
                    )
                ),
            )
        open_trace = None
        if trace_spec is not None:
            open_trace = raw_instance(
                _trace_type,
                (
                    ("allowlist_paths", trace_spec[0]),
                    ("opened_paths", trace_spec[1]),
                    ("outside_paths", trace_spec[2]),
                ),
            )
        rights = None
        if rights_spec is not None:
            rights = raw_instance(
                _rights_type,
                (
                    ("verdict", rights_spec[0]),
                    ("review_sha256", rights_spec[1]),
                    ("private_research_use", rights_spec[2]),
                    ("redistribution_allowed", rights_spec[3]),
                    ("commercial_use_allowed", rights_spec[4]),
                ),
            )
        golden = None
        if golden_spec is not None:
            golden = raw_instance(
                _golden_type,
                (
                    ("caption_count", golden_spec[0]),
                    ("captions_ascii_sha256", golden_spec[1]),
                    ("token_ids_sha256", golden_spec[2]),
                    ("projected_float64_l2_sha256", golden_spec[3]),
                    ("tokenizer_max_length", golden_spec[4]),
                    ("projected_width", golden_spec[5]),
                ),
            )
        return raw_instance(
            _bundle_type,
            (
                ("runtime", runtime),
                ("loader", loader),
                ("open_trace", open_trace),
                ("rights", rights),
                ("golden", golden),
            ),
        )

    def evidence_references(value: ClipResolutionEvidence) -> tuple[object, ...]:
        references: list[object] = [value]
        for component in (
            value.runtime,
            value.loader,
            value.open_trace,
            value.rights,
            value.golden,
        ):
            if component is not None:
                references.append(component)
        if value.runtime is not None:
            references.append(value.runtime.wheels)
            references.extend(value.runtime.wheels)
        if value.open_trace is not None:
            references.extend(
                (
                    value.open_trace.allowlist_paths,
                    value.open_trace.opened_paths,
                    value.open_trace.outside_paths,
                )
            )
        return tuple(references)

    def evidence_status(frozen: tuple[object, ...]) -> str:
        if frozen[0] is None:
            return "HOLD_RUNTIME_WHEELS_ABSENT"
        if frozen[1] is None:
            return "HOLD_TEXT_LOADER_EVIDENCE_ABSENT"
        if frozen[2] is None:
            return "HOLD_OPEN_TRACE_EVIDENCE_ABSENT"
        if frozen[3] is None:
            return "HOLD_RIGHTS_EVIDENCE_ABSENT"
        if frozen[4] is None:
            return "HOLD_SYNTHETIC_GOLDEN_ABSENT"
        return "HOLD_FRESH_REVIEW_REQUIRED"

    def freeze_scan(
        outcome: _ScanOutcome,
        pin_specs: tuple[tuple[str, int, str], ...],
        expected_total_bytes: int,
    ) -> tuple[object, ...]:
        if type(outcome) is not _outcome_type:
            raise _error_type("scan outcome has the wrong exact type")
        if outcome.status is not None and (
            type(outcome.status) is not str or outcome.status not in scan_statuses
        ):
            raise _error_type("scan outcome has an unknown status")
        if type(outcome.files) is not tuple:
            raise _error_type("scan files must be an exact built-in tuple")
        files: list[tuple[str, int, str]] = []
        for index, item in enumerate(outcome.files):
            if type(item) is not _snapshot_type:
                raise _error_type(f"scan.files[{index}] has the wrong exact type")
            files.append(
                (
                    safe_name(item.path, f"scan.files[{index}].path"),
                    exact_int(item.bytes, f"scan.files[{index}].bytes"),
                    sha256_hex(item.sha256, f"scan.files[{index}].sha256"),
                )
            )
        frozen_files = tuple(files)
        total_bytes = exact_int(outcome.total_bytes, "scan.total_bytes")
        if total_bytes != sum(item[1] for item in frozen_files):
            raise _error_type("scan total_bytes disagrees with frozen files")
        if len(frozen_files) > len(pin_specs):
            raise _error_type("scan contains more files than the exact pin census")
        for index, item in enumerate(frozen_files):
            pin = pin_specs[index]
            if item[0] != pin[0] or item[1] != pin[1]:
                raise _error_type("scan files are not a canonical pin prefix")
            hash_differs = item[2] != pin[2]
            allowed_mismatch = (
                outcome.status == "HOLD_FILE_SHA256_MISMATCH"
                and index == len(frozen_files) - 1
            )
            if hash_differs != allowed_mismatch:
                raise _error_type("scan file hashes contradict the scan status")
        if outcome.status == "HOLD_FILE_SHA256_MISMATCH" and not frozen_files:
            raise _error_type("SHA mismatch status requires one observed file")

        frozen_name_groups: list[tuple[str, ...]] = []
        for label, values in (
            ("missing_files", outcome.missing_files),
            ("extra_files", outcome.extra_files),
        ):
            if type(values) is not tuple:
                raise _error_type(f"scan {label} must be an exact built-in tuple")
            checked = tuple(
                safe_name(name, f"scan.{label}[{index}]")
                for index, name in enumerate(values)
            )
            if tuple(sorted(checked)) != checked or len(set(checked)) != len(checked):
                raise _error_type(f"scan {label} must be sorted and unique")
            frozen_name_groups.append(checked)
        missing, extra = frozen_name_groups

        def freeze_identity(value: object, label: str) -> object:
            if value is None:
                return None
            if type(value) is not _identity_type:
                raise _error_type(f"{label} has the wrong exact type")
            fields = (
                value.device,
                value.inode,
                value.mode,
                value.links,
                value.size,
                value.mtime_ns,
                value.ctime_ns,
                value.attributes,
            )
            for index, field in enumerate(fields):
                if type(field) is not int:
                    raise _error_type(f"{label}[{index}] must be an exact built-in int")
            return fields

        root_before = freeze_identity(outcome.root_before, "scan.root_before")
        root_after = freeze_identity(outcome.root_after, "scan.root_after")
        if type(outcome.file_identities) is not tuple:
            raise _error_type("scan.file_identities must be an exact built-in tuple")
        file_identities = tuple(
            freeze_identity(identity, f"scan.file_identities[{index}]")
            for index, identity in enumerate(outcome.file_identities)
        )
        if type(outcome.census) is not tuple:
            raise _error_type("scan.census must be an exact built-in tuple")
        census = tuple(
            safe_name(name, f"scan.census[{index}]")
            for index, name in enumerate(outcome.census)
        )
        if tuple(sorted(census)) != census or len(set(census)) != len(census):
            raise _error_type("scan census must be sorted and unique")
        double_pass_verified = exact_bool(
            outcome.double_pass_verified, "scan.double_pass_verified"
        )

        if outcome.status is None:
            if (
                frozen_files != pin_specs
                or total_bytes != expected_total_bytes
                or missing
                or extra
                or root_before is None
                or root_before != root_after
                or len(file_identities) != len(pin_specs)
                or census != tuple(sorted(exact_names))
                or not double_pass_verified
            ):
                raise _error_type(
                    "successful scan is not a complete stable double-pass snapshot"
                )
        elif outcome.status in root_statuses:
            if frozen_files or total_bytes or missing or extra:
                raise _error_type("root HOLD cannot carry file observations")
        elif outcome.status == "HOLD_SNAPSHOT_MISSING_FILES":
            if frozen_files or total_bytes or not missing or extra:
                raise _error_type("missing-files HOLD has inconsistent census fields")
        elif outcome.status == "HOLD_SNAPSHOT_EXTRA_FILES":
            if frozen_files or total_bytes or missing or not extra:
                raise _error_type("extra-files HOLD has inconsistent census fields")
        elif outcome.status == "HOLD_SNAPSHOT_CENSUS_MISMATCH":
            if frozen_files or total_bytes or not missing or not extra:
                raise _error_type("census HOLD requires both missing and extra files")
        elif outcome.status != "HOLD_SNAPSHOT_ALIAS" and (missing or extra):
            raise _error_type("non-census HOLD cannot carry missing or extra files")
        if outcome.status is not None and double_pass_verified:
            raise _error_type("HOLD scan cannot claim double-pass verification")
        return (
            outcome.status,
            frozen_files,
            total_bytes,
            missing,
            extra,
            root_before,
            root_after,
            file_identities,
            census,
            double_pass_verified,
        )

    def rebuild_scan(
        frozen: tuple[object, ...],
        pin_specs: tuple[tuple[str, int, str], ...],
        expected_total_bytes: int,
    ) -> tuple[object, ...]:
        (
            status,
            files,
            total_bytes,
            missing,
            extra,
            root_before,
            root_after,
            file_identities,
            census,
            double_pass_verified,
        ) = frozen

        def build_identity(spec: object) -> _FileIdentity | None:
            if spec is None:
                return None
            return _identity_type(*spec)

        snapshots = tuple(
            raw_instance(
                _snapshot_type,
                (("path", item[0]), ("bytes", item[1]), ("sha256", item[2])),
            )
            for item in files
        )
        outcome = _outcome_type(
            status,
            snapshots,
            total_bytes,
            missing,
            extra,
            root_before=build_identity(root_before),
            root_after=build_identity(root_after),
            file_identities=tuple(build_identity(spec) for spec in file_identities),
            census=census,
            double_pass_verified=double_pass_verified,
        )
        return freeze_scan(outcome, pin_specs, expected_total_bytes)

    def output_from_provenance(
        provenance: tuple[object, ...], lease: object
    ) -> ClipResolutionAssessment:
        _, _, _, frozen_scan, frozen_evidence = provenance
        scan_status, files, total_bytes, missing, extra = frozen_scan[:5]
        status = scan_status if scan_status is not None else evidence_status(frozen_evidence)
        snapshots = tuple(
            raw_instance(
                _snapshot_type,
                (("path", item[0]), ("bytes", item[1]), ("sha256", item[2])),
            )
            for item in files
        )
        evidence = build_evidence(frozen_evidence)
        return raw_instance(
            _assessment_type,
            (
                ("status", status),
                ("authority", 0),
                ("production", False),
                ("training_authorized", False),
                ("files", snapshots),
                ("total_bytes", total_bytes),
                ("missing_files", missing),
                ("extra_files", extra),
                ("evidence", evidence),
                ("_lease", lease),
            ),
        )

    def provenance_digest(provenance: tuple[object, ...]) -> bytes:
        return _sha256_factory(repr(provenance).encode("utf-8")).digest()

    def revalidate_provenance(provenance: tuple[object, ...]) -> tuple[object, ...]:
        if type(provenance) is not tuple or len(provenance) != 5:
            raise _error_type("issued provenance has the wrong closed shape")
        root_text, pin_specs, expected_total_bytes, frozen_scan, frozen_evidence = provenance
        exact_str(root_text, "provenance.root")
        if type(pin_specs) is not tuple:
            raise _error_type("provenance pins have the wrong closed shape")
        pin_objects = tuple(pin_from_spec(spec) for spec in pin_specs)
        checked_specs, _ = freeze_pins(pin_objects, expected_total_bytes)
        if checked_specs != pin_specs:
            raise _error_type("provenance pin reconstruction changed bytes")
        rebuilt_evidence = build_evidence(frozen_evidence)
        if freeze_evidence(rebuilt_evidence) != frozen_evidence:
            raise _error_type("provenance evidence reconstruction changed bytes")
        if rebuild_scan(frozen_scan, pin_specs, expected_total_bytes) != frozen_scan:
            raise _error_type("provenance scan reconstruction changed bytes")
        return provenance

    def issue(provenance: tuple[object, ...]) -> ClipResolutionAssessment:
        revalidate_provenance(provenance)
        lease = _lease_type()
        assessment = output_from_provenance(provenance, lease)
        seal = provenance_digest(provenance)
        lease.file_references = (assessment.files, *assessment.files)
        lease.evidence_references = evidence_references(assessment.evidence)
        lease.missing_reference = assessment.missing_files
        lease.extra_reference = assessment.extra_files
        issued[lease] = (
            id(assessment),
            provenance,
            seal,
            tuple(id(value) for value in lease.file_references),
            tuple(id(value) for value in lease.evidence_references),
            id(lease.missing_reference),
            id(lease.extra_reference),
        )
        return assessment

    def assess_impl(
        root: Path,
        pins: tuple[ClipFilePin, ...],
        expected_total_bytes: int,
        evidence: ClipResolutionEvidence | None,
        operations: _FilesystemOps,
    ) -> ClipResolutionAssessment:
        if type(root) is not _path_type:
            raise TypeError("root must be an exact platform pathlib.Path")
        pin_specs, scan_pins = freeze_pins(pins, expected_total_bytes)
        frozen_evidence = freeze_evidence(evidence)
        if type(operations) is not _ops_type:
            raise TypeError("operations must be exactly _FilesystemOps")
        for label in ("lstat", "scandir", "open_fd", "fstat", "read_fd", "close_fd"):
            if not callable(getattr(operations, label)):
                raise _error_type(f"filesystem operation {label} must be callable")
        outcome = _scan_function(root, scan_pins, expected_total_bytes, operations)
        frozen_scan = freeze_scan(outcome, pin_specs, expected_total_bytes)
        root_text = exact_str(
            _absolute_path(_filesystem_path(root)),
            "root path",
        )
        return issue(
            (root_text, pin_specs, expected_total_bytes, frozen_scan, frozen_evidence)
        )

    production_pins = tuple(pin_from_spec(spec) for spec in production_specs)

    def public_assess(root: Path) -> ClipResolutionAssessment:
        """Run the sealed, production-pin, authority-zero local preflight."""

        return assess_impl(root, production_pins, 608_863_580, None, _real_operations)

    def private_test_assess(
        root: Path,
        *,
        pins: tuple[ClipFilePin, ...],
        expected_total_bytes: int,
        evidence: ClipResolutionEvidence | None = None,
        operations: _FilesystemOps | None = None,
    ) -> ClipResolutionAssessment:
        """Private tiny-file and metadata seam backed by the same sealed semantics."""

        checked_operations = _real_operations if operations is None else operations
        if type(checked_operations) is not _ops_type:
            raise TypeError("operations must be exactly _FilesystemOps or None")
        return assess_impl(
            root,
            pins,
            expected_total_bytes,
            evidence,
            checked_operations,
        )

    def frozen_evidence_payload(frozen: tuple[object, ...]) -> dict[str, object]:
        runtime_spec, loader_spec, trace_spec, rights_spec, golden_spec = frozen
        runtime_payload: object = None
        if runtime_spec is not None:
            runtime_payload = {
                "wheels": [
                    {
                        "bytes": spec[2],
                        "distribution": spec[0],
                        "filename": spec[1],
                        "sha256": spec[3],
                    }
                    for spec in runtime_spec
                ]
            }
        loader_payload: object = None
        if loader_spec is not None:
            loader_payload = {
                "active_text_dropout_sites": loader_spec[9],
                "exposes_final_ln_pooled_eos": loader_spec[5],
                "exposes_pretrained_text_projection": loader_spec[6],
                "loader_class": loader_spec[0],
                "local_files_only": loader_spec[2],
                "text_model_called_once": loader_spec[4],
                "text_projection_requires_grad": loader_spec[8],
                "tokenizer_class": loader_spec[1],
                "tokenizer_max_length": loader_spec[10],
                "trust_remote_code": loader_spec[3],
                "vision_tower_instantiated": loader_spec[7],
            }
        trace_payload: object = None
        if trace_spec is not None:
            trace_payload = {
                "allowlist_paths": list(trace_spec[0]),
                "opened_paths": list(trace_spec[1]),
                "outside_paths": list(trace_spec[2]),
            }
        rights_payload: object = None
        if rights_spec is not None:
            rights_payload = {
                "commercial_use_allowed": rights_spec[4],
                "private_research_use": rights_spec[2],
                "redistribution_allowed": rights_spec[3],
                "review_sha256": rights_spec[1],
                "verdict": rights_spec[0],
            }
        golden_payload: object = None
        if golden_spec is not None:
            golden_payload = {
                "caption_count": golden_spec[0],
                "captions_ascii_sha256": golden_spec[1],
                "projected_float64_l2_sha256": golden_spec[3],
                "projected_width": golden_spec[5],
                "token_ids_sha256": golden_spec[2],
                "tokenizer_max_length": golden_spec[4],
            }
        return {
            "golden": golden_payload,
            "loader": loader_payload,
            "open_trace": trace_payload,
            "rights": rights_payload,
            "runtime": runtime_payload,
        }

    def canonical(assessment: ClipResolutionAssessment) -> bytes:
        """Rebuild an issued assessment from sealed provenance before encoding."""

        if type(assessment) is not _assessment_type:
            raise _error_type("assessment has the wrong exact type")
        try:
            lease = assessment._lease
        except AttributeError as exc:
            raise _error_type("assessment has no sealed provenance lease") from exc
        if type(lease) is not _lease_type:
            raise _error_type("assessment provenance lease has the wrong exact type")
        record = issued.get(lease)
        if record is None or record[0] != id(assessment):
            raise _error_type("assessment was not issued by the sealed assessor")
        (
            _,
            provenance,
            seal,
            issued_file_ids,
            issued_evidence_ids,
            issued_missing_id,
            issued_extra_id,
        ) = record
        if provenance_digest(provenance) != seal:
            raise _error_type("assessment provenance seal mismatch")
        revalidate_provenance(provenance)
        root_text, pin_specs, expected_total_bytes, frozen_scan, provenance_evidence = (
            provenance
        )
        rescan_pins = tuple(pin_from_spec(spec) for spec in pin_specs)
        rescan_outcome = _scan_function(
            _path_type(root_text),
            rescan_pins,
            expected_total_bytes,
            _real_operations,
        )
        rescanned = freeze_scan(rescan_outcome, pin_specs, expected_total_bytes)
        if rescanned != frozen_scan:
            raise _error_type("current double-pass snapshot differs from issued provenance")
        expected = output_from_provenance(provenance, None)

        file_refs = lease.file_references
        evidence_refs = lease.evidence_references
        missing_ref = lease.missing_reference
        extra_ref = lease.extra_reference
        if (
            tuple(id(value) for value in file_refs) != issued_file_ids
            or tuple(id(value) for value in evidence_refs) != issued_evidence_ids
            or id(missing_ref) != issued_missing_id
            or id(extra_ref) != issued_extra_id
        ):
            raise _error_type("assessment lease identity ledger changed")

        if assessment.files is not file_refs[0] or any(
            current is not original
            for current, original in zip(assessment.files, file_refs[1:], strict=True)
        ):
            raise _error_type("assessment file snapshot identity changed")
        if assessment.missing_files is not missing_ref or assessment.extra_files is not extra_ref:
            raise _error_type("assessment census tuple identity changed")
        try:
            frozen_evidence = freeze_evidence(assessment.evidence)
            current_evidence_refs = evidence_references(assessment.evidence)
        except _error_type:
            raise
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise _error_type("assessment evidence snapshot is malformed") from exc
        if len(current_evidence_refs) != len(evidence_refs) or any(
            current is not original
            for current, original in zip(
                current_evidence_refs, evidence_refs, strict=True
            )
        ):
            raise _error_type("assessment evidence snapshot identity changed")

        try:
            current_files = tuple(
                (
                    safe_name(item.path, f"assessment.files[{index}].path"),
                    exact_int(item.bytes, f"assessment.files[{index}].bytes"),
                    sha256_hex(item.sha256, f"assessment.files[{index}].sha256"),
                )
                for index, item in enumerate(assessment.files)
                if type(item) is _snapshot_type
            )
        except _error_type:
            raise
        except (AttributeError, IndexError, TypeError, ValueError) as exc:
            raise _error_type("assessment file snapshot is malformed") from exc
        if len(current_files) != len(assessment.files):
            raise _error_type("assessment files contain a forged output row")
        expected_evidence = freeze_evidence(expected.evidence)
        if (
            type(assessment.status) is not str
            or assessment.status != expected.status
            or type(assessment.authority) is not int
            or assessment.authority != 0
            or type(assessment.production) is not bool
            or assessment.production
            or type(assessment.training_authorized) is not bool
            or assessment.training_authorized
            or current_files
            != tuple((item.path, item.bytes, item.sha256) for item in expected.files)
            or type(assessment.total_bytes) is not int
            or assessment.total_bytes != expected.total_bytes
            or type(assessment.missing_files) is not tuple
            or assessment.missing_files != expected.missing_files
            or type(assessment.extra_files) is not tuple
            or assessment.extra_files != expected.extra_files
            or frozen_evidence != expected_evidence
        ):
            raise _error_type("assessment fields disagree with sealed provenance")

        _, files, total_bytes, missing, extra = frozen_scan[:5]
        payload = {
            "authority": 0,
            "evidence": frozen_evidence_payload(provenance_evidence),
            "extra_files": list(extra),
            "files": [
                {"bytes": item[1], "path": item[0], "sha256": item[2]}
                for item in files
            ],
            "missing_files": list(missing),
            "model_id": "openai/clip-vit-base-patch32",
            "production": False,
            "revision": "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
            "schema": "phasepair-clip-resolution-preflight-v1",
            "status": expected.status,
            "total_bytes": total_bytes,
            "training_authorized": False,
        }
        if set(payload) != canonical_keys or not payload["status"].startswith("HOLD_"):
            raise AssertionError("internal sealed canonical assessment mismatch")
        return (
            _json_dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )

    def issued_lease_count_for_tests() -> int:
        """Return only the live weak-key count for bounded lifetime regression tests."""

        return len(issued)

    return public_assess, private_test_assess, canonical, issued_lease_count_for_tests


(
    assess_local_clip_snapshot,
    _assess_local_clip_snapshot_for_tests,
    canonical_assessment_bytes,
    _issued_lease_count_for_tests,
) = _seal_clip_resolution_api()
del _seal_clip_resolution_api


__all__ = [
    "AUTHORITY",
    "ClipFilePin",
    "ClipFileSnapshot",
    "ClipResolutionAssessment",
    "ClipResolutionError",
    "ClipResolutionEvidence",
    "MODEL_ID",
    "OpenTraceEvidence",
    "PINNED_FILES",
    "PRODUCTION",
    "REVISION",
    "RightsEvidence",
    "RuntimeWheelEvidence",
    "SyntheticCaptionGoldenEvidence",
    "TOTAL_BYTES",
    "TRAINING_AUTHORIZED",
    "TextLoaderEvidence",
    "WheelIdentity",
    "assess_local_clip_snapshot",
    "canonical_assessment_bytes",
]
