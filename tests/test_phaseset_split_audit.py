from __future__ import annotations

import json
from pathlib import Path

import pytest

from phaseset_core.data import CaptureRecord, PublicMetadata
from phaseset_core import split_audit


ROOT = Path(__file__).resolve().parents[1]
FIXED_MANIFEST = ROOT / "configs" / "phaseset" / "embody3d_split_v1.json"


def _capture(
    name: str,
    participants: tuple[str, ...],
    *,
    subset: str,
    frames: int,
) -> CaptureRecord:
    ordered = tuple(sorted(participants, key=lambda item: item.encode("utf-8")))
    return CaptureRecord(subset, name, frames, ordered, ordered)


def _statistics_mapping(statistics: split_audit.AggregateStatistics) -> dict[str, object]:
    return {
        "participant_count": statistics.participant_count,
        "capture_count": statistics.capture_count,
        "total_frames": statistics.total_frames,
        "subset_counts": dict(statistics.subset_counts),
        "k_counts": {str(k): count for k, count in statistics.k_counts},
    }


def _synthetic_fixture() -> tuple[PublicMetadata, split_audit.SplitManifest]:
    captures = (
        _capture(
            "capture-train",
            ("train-a", "train-b", "train-c", "train-d"),
            subset="acting",
            frames=120,
        ),
        _capture(
            "capture-validation",
            ("validation-a", "validation-b", "validation-c", "validation-d"),
            subset="daylife",
            frames=150,
        ),
        _capture(
            "capture-test",
            ("test-a", "test-b", "test-c"),
            subset="acting",
            frames=180,
        ),
    )
    captures = tuple(sorted(captures, key=lambda item: item.key))
    metadata = PublicMetadata(captures)
    components = split_audit.build_cooccurrence_components(captures)
    by_capture = {component.captures[0].capture_name: component for component in components}
    labels = {
        "T": by_capture["capture-train"],
        "V": by_capture["capture-validation"],
        "E": by_capture["capture-test"],
    }
    expected = {
        "train": labels["T"].statistics,
        "validation": labels["V"].statistics,
        "test": labels["E"].statistics,
    }
    mapping = {
        "schema": split_audit.MANIFEST_SCHEMA,
        "dataset": "synthetic-public-index",
        "source_fps": 30,
        "eligibility": {
            "minimum_participants": 3,
            "require_complete_actor_text": True,
        },
        "component_commitment_algorithm": split_audit.COMPONENT_COMMITMENT_ALGORITHM,
        "components": [
            {
                "label": label,
                "commitment_sha256": component.commitment_sha256,
                "statistics": _statistics_mapping(component.statistics),
            }
            for label, component in labels.items()
        ],
        "splits": {"train": ["T"], "validation": ["V"], "test": ["E"]},
        "expected_split_statistics": {
            split: _statistics_mapping(statistics) for split, statistics in expected.items()
        },
        "zero_shot_k3": {
            "held_out_k": 3,
            "train_capture_count": 0,
            "validation_capture_count": 0,
            "test_capture_count": 1,
            "claim": "Synthetic K=3 is held out.",
            "scope_limit": "Synthetic fixture only.",
        },
        "public_safety": {
            "contains_participant_ids": False,
            "contains_capture_names": False,
            "contains_asset_paths": False,
        },
    }
    return metadata, split_audit.split_manifest_from_mapping(mapping)


def test_fixed_manifest_has_16_commitments_and_400_96_76_counts() -> None:
    manifest = split_audit.load_split_manifest(FIXED_MANIFEST)
    assert len(manifest.components) == 16
    assert tuple(
        manifest.expected_for(split).capture_count for split in split_audit.SPLIT_NAMES
    ) == (
        400,
        96,
        76,
    )
    assert manifest.expected_for("train").participant_count == 48
    assert manifest.expected_for("validation").participant_count == 10
    assert manifest.expected_for("test").participant_count == 11
    assert manifest.expected_for("test").k_counts == ((3, 27), (4, 49))


def test_public_manifest_contains_no_identifier_or_asset_preimage() -> None:
    raw = FIXED_MANIFEST.read_text(encoding="utf-8")
    decoded = json.loads(raw)
    assert '"participants"' not in raw
    assert '"capture_name"' not in raw
    assert "/social_ai/" not in raw
    assert all(
        set(component) == {"label", "commitment_sha256", "statistics"}
        for component in decoded["components"]
    )


def test_component_construction_is_order_independent_and_repr_hides_preimage() -> None:
    metadata, _ = _synthetic_fixture()
    forward = split_audit.build_cooccurrence_components(metadata.captures)
    reverse = split_audit.build_cooccurrence_components(reversed(metadata.captures))
    assert tuple(component.commitment_sha256 for component in forward) == tuple(
        component.commitment_sha256 for component in reverse
    )
    assert "train-a" not in repr(forward)
    assert "capture-train" not in repr(forward)


def test_synthetic_audit_proves_disjointness_and_zero_shot_k3() -> None:
    metadata, manifest = _synthetic_fixture()
    report = split_audit.audit_split(metadata, manifest)
    assert report.eligible_capture_count == 3
    assert report.component_count == 3
    assert report.participant_overlap_count == 0
    assert report.cross_split_capture_count == 0
    assert report.k3_zero_shot_verified is True
    assert report.statistics_for("test").k_count(3) == 1


def test_participant_overlap_and_manifest_tampering_fail_closed() -> None:
    with pytest.raises(split_audit.SplitAuditError, match="leakage"):
        split_audit.validate_participant_disjoint(
            {
                "train": {"shared-actor"},
                "validation": {"validation-actor"},
                "test": {"shared-actor"},
            }
        )
    metadata, manifest = _synthetic_fixture()
    tampered_component = split_audit.ComponentExpectation(
        label=manifest.components[0].label,
        commitment_sha256="0" * 64,
        statistics=manifest.components[0].statistics,
    )
    tampered = split_audit.SplitManifest(
        dataset=manifest.dataset,
        minimum_participants=manifest.minimum_participants,
        require_complete_actor_text=manifest.require_complete_actor_text,
        components=(tampered_component, *manifest.components[1:]),
        split_components=manifest.split_components,
        expected_statistics=manifest.expected_statistics,
        zero_shot_k3=manifest.zero_shot_k3,
    )
    with pytest.raises(split_audit.SplitAuditError, match="commitment set"):
        split_audit.audit_split(metadata, tampered)
