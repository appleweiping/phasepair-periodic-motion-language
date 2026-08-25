"""Participant-disjoint component commitments and split auditing.

The public manifest contains only domain-separated SHA-256 component
commitments and aggregate counts.  Participant identifiers and capture names
are consumed in memory from the official public index, but are never included
in an audit report or public manifest serialization.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Iterable, Mapping

from phaseset_core.data import (
    CaptureRecord,
    PublicMetadata,
    SOURCE_FPS,
    eligible_group_text_captures,
)


MANIFEST_SCHEMA = "phaseset-embody3d-participant-split-v1"
COMPONENT_COMMITMENT_ALGORITHM = "SHA256_CANONICAL_JSON_COMPONENT_V1"
COMPONENT_COMMITMENT_DOMAIN = b"phaseset-embody3d-participant-component-v1\x00"
SPLIT_NAMES = ("train", "validation", "test")


class SplitAuditError(ValueError):
    """A component commitment, split statistic, or leakage invariant failed."""


@dataclass(frozen=True, slots=True)
class AggregateStatistics:
    participant_count: int
    capture_count: int
    total_frames: int
    subset_counts: tuple[tuple[str, int], ...]
    k_counts: tuple[tuple[int, int], ...]

    @property
    def hours_at_30fps(self) -> float:
        return self.total_frames / (SOURCE_FPS * 3600.0)

    def subset_count(self, subset: str) -> int:
        return dict(self.subset_counts).get(subset, 0)

    def k_count(self, participant_count: int) -> int:
        return dict(self.k_counts).get(participant_count, 0)


@dataclass(frozen=True, slots=True)
class CooccurrenceComponent:
    commitment_sha256: str
    statistics: AggregateStatistics
    participants: tuple[str, ...] = field(repr=False)
    captures: tuple[CaptureRecord, ...] = field(repr=False)


@dataclass(frozen=True, slots=True)
class ComponentExpectation:
    label: str
    commitment_sha256: str
    statistics: AggregateStatistics


@dataclass(frozen=True, slots=True)
class ZeroShotK3Declaration:
    held_out_k: int
    train_capture_count: int
    validation_capture_count: int
    test_capture_count: int
    claim: str
    scope_limit: str


@dataclass(frozen=True, slots=True)
class SplitManifest:
    dataset: str
    minimum_participants: int
    require_complete_actor_text: bool
    components: tuple[ComponentExpectation, ...]
    split_components: tuple[tuple[str, tuple[str, ...]], ...]
    expected_statistics: tuple[tuple[str, AggregateStatistics], ...]
    zero_shot_k3: ZeroShotK3Declaration

    def labels_for(self, split: str) -> tuple[str, ...]:
        return dict(self.split_components)[split]

    def expected_for(self, split: str) -> AggregateStatistics:
        return dict(self.expected_statistics)[split]


@dataclass(frozen=True, slots=True)
class SplitAuditReport:
    dataset: str
    eligible_capture_count: int
    component_count: int
    split_statistics: tuple[tuple[str, AggregateStatistics], ...]
    participant_overlap_count: int
    cross_split_capture_count: int
    k3_zero_shot_verified: bool
    claim: str
    scope_limit: str

    def statistics_for(self, split: str) -> AggregateStatistics:
        return dict(self.split_statistics)[split]


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _lower_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise SplitAuditError(f"{label} must be lowercase SHA-256 hex")
    return value


def _nonempty_text(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    if not value or "\x00" in value:
        raise SplitAuditError(f"{label} must be nonempty and contain no NUL")
    return value


def _canonical_capture_key(capture: CaptureRecord) -> tuple[bytes, bytes]:
    return capture.subset.encode("utf-8"), capture.capture_name.encode("utf-8")


def aggregate_statistics(captures: Iterable[CaptureRecord]) -> AggregateStatistics:
    """Compute deterministic aggregate counts for captures."""

    checked = tuple(captures)
    if any(type(capture) is not CaptureRecord for capture in checked):
        raise TypeError("captures must contain only exact CaptureRecord values")
    keys = tuple(capture.key for capture in checked)
    if len(keys) != len(set(keys)):
        raise SplitAuditError("captures contain a duplicate canonical identity")
    participants = {participant for capture in checked for participant in capture.participants}
    subsets = Counter(capture.subset for capture in checked)
    k_counts = Counter(capture.participant_count for capture in checked)
    return AggregateStatistics(
        participant_count=len(participants),
        capture_count=len(checked),
        total_frames=sum(capture.length_frames for capture in checked),
        subset_counts=tuple(sorted(subsets.items(), key=lambda item: item[0].encode("utf-8"))),
        k_counts=tuple(sorted(k_counts.items())),
    )


def component_commitment(captures: Iterable[CaptureRecord]) -> str:
    """Commit to one exact component without publishing its preimage."""

    checked = tuple(sorted(tuple(captures), key=_canonical_capture_key))
    if not checked:
        raise SplitAuditError("a component must contain at least one capture")
    keys = tuple(capture.key for capture in checked)
    if len(set(keys)) != len(keys):
        raise SplitAuditError("a component cannot contain duplicate captures")
    members = tuple(
        sorted(
            {participant for capture in checked for participant in capture.participants},
            key=lambda item: item.encode("utf-8"),
        )
    )
    payload = {
        "captures": [
            {
                "capture_name": capture.capture_name,
                "length_frames": capture.length_frames,
                "participants": list(capture.participants),
                "subset": capture.subset,
                "text_participants": list(capture.text_participants),
            }
            for capture in checked
        ],
        "participants": list(members),
        "schema": "phaseset-embody3d-participant-component-preimage-v1",
    }
    return hashlib.sha256(COMPONENT_COMMITMENT_DOMAIN + _canonical_json_bytes(payload)).hexdigest()


def build_cooccurrence_components(
    captures: Iterable[CaptureRecord],
) -> tuple[CooccurrenceComponent, ...]:
    """Build connected components of the participant co-occurrence hypergraph."""

    checked = tuple(captures)
    if not checked:
        return ()
    if any(type(capture) is not CaptureRecord for capture in checked):
        raise TypeError("captures must contain only exact CaptureRecord values")
    keys = tuple(capture.key for capture in checked)
    if len(keys) != len(set(keys)):
        raise SplitAuditError("captures contain a duplicate canonical identity")

    parent: dict[str, str] = {}

    def find(item: str) -> str:
        parent.setdefault(item, item)
        while parent[item] != item:
            parent[item] = parent[parent[item]]
            item = parent[item]
        return item

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        if left_root.encode("utf-8") <= right_root.encode("utf-8"):
            parent[right_root] = left_root
        else:
            parent[left_root] = right_root

    for capture in checked:
        if capture.participant_count < 2:
            raise SplitAuditError("co-occurrence components require at least two participants")
        anchor = capture.participants[0]
        find(anchor)
        for participant in capture.participants[1:]:
            union(anchor, participant)

    member_groups: dict[str, set[str]] = {}
    for participant in parent:
        member_groups.setdefault(find(participant), set()).add(participant)
    capture_groups: dict[str, list[CaptureRecord]] = {root: [] for root in member_groups}
    for capture in checked:
        roots = {find(participant) for participant in capture.participants}
        if len(roots) != 1:
            raise AssertionError("union-find failed to connect one capture")
        capture_groups[next(iter(roots))].append(capture)

    output: list[CooccurrenceComponent] = []
    for root, members_set in member_groups.items():
        members = tuple(sorted(members_set, key=lambda item: item.encode("utf-8")))
        component_captures = tuple(sorted(capture_groups[root], key=_canonical_capture_key))
        statistics = aggregate_statistics(component_captures)
        if statistics.participant_count != len(members):
            raise AssertionError("component participant count is inconsistent")
        output.append(
            CooccurrenceComponent(
                commitment_sha256=component_commitment(component_captures),
                statistics=statistics,
                participants=members,
                captures=component_captures,
            )
        )
    return tuple(sorted(output, key=lambda component: component.commitment_sha256))


def _exact_int(value: object, label: str, *, minimum: int = 0) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < minimum:
        raise SplitAuditError(f"{label} must be >= {minimum}")
    return value


def _closed_mapping(value: object, label: str, keys: set[str]) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    observed = set(value)
    if observed != keys:
        raise SplitAuditError(f"{label} keys differ from the closed schema")
    return value


def _parse_string_counts(value: object, label: str) -> tuple[tuple[str, int], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    output: list[tuple[str, int]] = []
    for key, count in value.items():
        output.append((_nonempty_text(key, f"{label} key"), _exact_int(count, f"{label} count")))
    if any(count == 0 for _, count in output):
        raise SplitAuditError(f"{label} must omit zero-valued rows")
    return tuple(sorted(output, key=lambda item: item[0].encode("utf-8")))


def _parse_k_counts(value: object, label: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    output: list[tuple[int, int]] = []
    for raw_k, count in value.items():
        if type(raw_k) is not str or not raw_k.isascii() or not raw_k.isdecimal():
            raise SplitAuditError(f"{label} keys must be decimal strings")
        k = int(raw_k)
        output.append((k, _exact_int(count, f"{label} count", minimum=1)))
    if len({k for k, _ in output}) != len(output):
        raise SplitAuditError(f"{label} contains a duplicate K")
    return tuple(sorted(output))


def _parse_statistics(value: object, label: str) -> AggregateStatistics:
    row = _closed_mapping(
        value,
        label,
        {"participant_count", "capture_count", "total_frames", "subset_counts", "k_counts"},
    )
    statistics = AggregateStatistics(
        participant_count=_exact_int(row["participant_count"], f"{label}.participant_count"),
        capture_count=_exact_int(row["capture_count"], f"{label}.capture_count"),
        total_frames=_exact_int(row["total_frames"], f"{label}.total_frames"),
        subset_counts=_parse_string_counts(row["subset_counts"], f"{label}.subset_counts"),
        k_counts=_parse_k_counts(row["k_counts"], f"{label}.k_counts"),
    )
    if sum(count for _, count in statistics.subset_counts) != statistics.capture_count:
        raise SplitAuditError(f"{label} subset counts do not sum to capture_count")
    if sum(count for _, count in statistics.k_counts) != statistics.capture_count:
        raise SplitAuditError(f"{label} K counts do not sum to capture_count")
    return statistics


def split_manifest_from_mapping(value: object) -> SplitManifest:
    """Validate the closed public split-manifest schema."""

    root = _closed_mapping(
        value,
        "manifest",
        {
            "schema",
            "dataset",
            "source_fps",
            "eligibility",
            "component_commitment_algorithm",
            "components",
            "splits",
            "expected_split_statistics",
            "zero_shot_k3",
            "public_safety",
        },
    )
    if root["schema"] != MANIFEST_SCHEMA:
        raise SplitAuditError("manifest schema is not supported")
    if root["source_fps"] != SOURCE_FPS:
        raise SplitAuditError("manifest source_fps must be 30")
    if root["component_commitment_algorithm"] != COMPONENT_COMMITMENT_ALGORITHM:
        raise SplitAuditError("component commitment algorithm is not supported")
    safety = _closed_mapping(
        root["public_safety"],
        "public_safety",
        {"contains_participant_ids", "contains_capture_names", "contains_asset_paths"},
    )
    if any(safety[key] is not False for key in safety):
        raise SplitAuditError("public manifest must declare that private identifiers are absent")
    eligibility = _closed_mapping(
        root["eligibility"],
        "eligibility",
        {"minimum_participants", "require_complete_actor_text"},
    )
    minimum_participants = _exact_int(
        eligibility["minimum_participants"], "minimum_participants", minimum=3
    )
    require_text = eligibility["require_complete_actor_text"]
    if type(require_text) is not bool or require_text is not True:
        raise SplitAuditError("require_complete_actor_text must be true")

    raw_components = root["components"]
    if type(raw_components) is not list or not raw_components:
        raise TypeError("components must be a nonempty JSON array")
    components: list[ComponentExpectation] = []
    for index, raw_component in enumerate(raw_components):
        row = _closed_mapping(
            raw_component,
            f"components[{index}]",
            {"label", "commitment_sha256", "statistics"},
        )
        components.append(
            ComponentExpectation(
                label=_nonempty_text(row["label"], f"components[{index}].label"),
                commitment_sha256=_lower_sha256(
                    row["commitment_sha256"], f"components[{index}].commitment_sha256"
                ),
                statistics=_parse_statistics(row["statistics"], f"components[{index}].statistics"),
            )
        )
    labels = tuple(component.label for component in components)
    commitments = tuple(component.commitment_sha256 for component in components)
    if len(set(labels)) != len(labels) or len(set(commitments)) != len(commitments):
        raise SplitAuditError("component labels and commitments must both be unique")

    raw_splits = _closed_mapping(root["splits"], "splits", set(SPLIT_NAMES))
    split_components: list[tuple[str, tuple[str, ...]]] = []
    assigned_labels: list[str] = []
    for split in SPLIT_NAMES:
        raw_labels = raw_splits[split]
        if type(raw_labels) is not list or not raw_labels:
            raise TypeError(f"splits.{split} must be a nonempty JSON array")
        checked_labels = tuple(
            _nonempty_text(label, f"splits.{split} label") for label in raw_labels
        )
        if len(set(checked_labels)) != len(checked_labels):
            raise SplitAuditError(f"splits.{split} contains a duplicate component")
        split_components.append((split, checked_labels))
        assigned_labels.extend(checked_labels)
    if len(assigned_labels) != len(set(assigned_labels)) or set(assigned_labels) != set(labels):
        raise SplitAuditError("every component label must be assigned to exactly one split")

    raw_expected = _closed_mapping(
        root["expected_split_statistics"], "expected_split_statistics", set(SPLIT_NAMES)
    )
    expected = tuple(
        (split, _parse_statistics(raw_expected[split], f"expected_split_statistics.{split}"))
        for split in SPLIT_NAMES
    )

    raw_zero = _closed_mapping(
        root["zero_shot_k3"],
        "zero_shot_k3",
        {
            "held_out_k",
            "train_capture_count",
            "validation_capture_count",
            "test_capture_count",
            "claim",
            "scope_limit",
        },
    )
    zero = ZeroShotK3Declaration(
        held_out_k=_exact_int(raw_zero["held_out_k"], "zero_shot_k3.held_out_k", minimum=3),
        train_capture_count=_exact_int(
            raw_zero["train_capture_count"], "zero_shot_k3.train_capture_count"
        ),
        validation_capture_count=_exact_int(
            raw_zero["validation_capture_count"], "zero_shot_k3.validation_capture_count"
        ),
        test_capture_count=_exact_int(
            raw_zero["test_capture_count"], "zero_shot_k3.test_capture_count"
        ),
        claim=_nonempty_text(raw_zero["claim"], "zero_shot_k3.claim"),
        scope_limit=_nonempty_text(raw_zero["scope_limit"], "zero_shot_k3.scope_limit"),
    )
    if zero.held_out_k != 3:
        raise SplitAuditError("this manifest version freezes held_out_k=3")
    return SplitManifest(
        dataset=_nonempty_text(root["dataset"], "dataset"),
        minimum_participants=minimum_participants,
        require_complete_actor_text=require_text,
        components=tuple(components),
        split_components=tuple(split_components),
        expected_statistics=expected,
        zero_shot_k3=zero,
    )


def load_split_manifest(path: str | Path) -> SplitManifest:
    """Load a public aggregate-only split manifest, rejecting duplicate keys."""

    raw = Path(path).read_bytes()

    def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
        output: dict[str, object] = {}
        for key, item in pairs:
            if key in output:
                raise SplitAuditError("split manifest contains a duplicate JSON key")
            output[key] = item
        return output

    def reject_constant(value: str) -> object:
        raise SplitAuditError(f"split manifest contains forbidden constant {value}")

    try:
        decoded = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SplitAuditError("split manifest must be strict UTF-8 JSON") from exc
    return split_manifest_from_mapping(decoded)


def validate_participant_disjoint(
    split_participants: Mapping[str, Iterable[str]],
) -> None:
    """Fail if any runtime participant occurs in more than one split."""

    if set(split_participants) != set(SPLIT_NAMES):
        raise SplitAuditError("participant mapping must contain train, validation, and test")
    seen: set[str] = set()
    for split in SPLIT_NAMES:
        participants = set(split_participants[split])
        if seen.intersection(participants):
            raise SplitAuditError("participant leakage exists across splits")
        seen.update(participants)


def _assert_statistics(
    observed: AggregateStatistics,
    expected: AggregateStatistics,
    label: str,
) -> None:
    if observed != expected:
        raise SplitAuditError(f"aggregate statistics differ for {label}")


def audit_split(metadata: PublicMetadata, manifest: SplitManifest) -> SplitAuditReport:
    """Re-derive the committed split and prove participant/capture disjointness."""

    if type(metadata) is not PublicMetadata:
        raise TypeError("metadata must be exactly PublicMetadata")
    if type(manifest) is not SplitManifest:
        raise TypeError("manifest must be exactly SplitManifest")
    eligible = eligible_group_text_captures(
        metadata,
        minimum_participants=manifest.minimum_participants,
        require_complete_actor_text=manifest.require_complete_actor_text,
    )
    components = build_cooccurrence_components(eligible)
    observed_by_commitment = {component.commitment_sha256: component for component in components}
    expected_by_commitment = {
        component.commitment_sha256: component for component in manifest.components
    }
    if set(observed_by_commitment) != set(expected_by_commitment):
        raise SplitAuditError("observed component commitment set differs from the manifest")

    label_to_component: dict[str, CooccurrenceComponent] = {}
    for expected in manifest.components:
        observed = observed_by_commitment[expected.commitment_sha256]
        _assert_statistics(observed.statistics, expected.statistics, expected.label)
        label_to_component[expected.label] = observed

    split_captures: dict[str, list[CaptureRecord]] = {split: [] for split in SPLIT_NAMES}
    split_participants: dict[str, set[str]] = {split: set() for split in SPLIT_NAMES}
    capture_splits: dict[tuple[str, str], set[str]] = {}
    for split, labels in manifest.split_components:
        for label in labels:
            component = label_to_component[label]
            split_captures[split].extend(component.captures)
            split_participants[split].update(component.participants)
            for capture in component.captures:
                capture_splits.setdefault(capture.key, set()).add(split)

    validate_participant_disjoint(split_participants)
    cross_split_capture_count = sum(len(splits) > 1 for splits in capture_splits.values())
    if cross_split_capture_count:
        raise SplitAuditError("a capture occurs in more than one split")
    if set(capture_splits) != {capture.key for capture in eligible}:
        raise SplitAuditError("split assignment does not cover every eligible capture exactly once")

    observed_statistics: list[tuple[str, AggregateStatistics]] = []
    for split in SPLIT_NAMES:
        statistics = aggregate_statistics(split_captures[split])
        _assert_statistics(statistics, manifest.expected_for(split), split)
        observed_statistics.append((split, statistics))

    zero = manifest.zero_shot_k3
    k3_counts = {
        split: dict(statistics.k_counts).get(zero.held_out_k, 0)
        for split, statistics in observed_statistics
    }
    expected_k3 = {
        "train": zero.train_capture_count,
        "validation": zero.validation_capture_count,
        "test": zero.test_capture_count,
    }
    if k3_counts != expected_k3 or k3_counts["train"] or k3_counts["validation"]:
        raise SplitAuditError("K=3 zero-shot declaration does not match the audited split")
    if k3_counts["test"] <= 0:
        raise SplitAuditError("K=3 zero-shot test set must be nonempty")

    return SplitAuditReport(
        dataset=manifest.dataset,
        eligible_capture_count=len(eligible),
        component_count=len(components),
        split_statistics=tuple(observed_statistics),
        participant_overlap_count=0,
        cross_split_capture_count=0,
        k3_zero_shot_verified=True,
        claim=zero.claim,
        scope_limit=zero.scope_limit,
    )


__all__ = [
    "AggregateStatistics",
    "COMPONENT_COMMITMENT_ALGORITHM",
    "COMPONENT_COMMITMENT_DOMAIN",
    "CooccurrenceComponent",
    "ComponentExpectation",
    "MANIFEST_SCHEMA",
    "SPLIT_NAMES",
    "SplitAuditError",
    "SplitAuditReport",
    "SplitManifest",
    "ZeroShotK3Declaration",
    "aggregate_statistics",
    "audit_split",
    "build_cooccurrence_components",
    "component_commitment",
    "load_split_manifest",
    "split_manifest_from_mapping",
    "validate_participant_disjoint",
]
