from __future__ import annotations

import json

import pytest

from phaseset_core import data


def _actor_row(
    actor: str,
    participants: tuple[str, ...],
    *,
    length: int = 90,
    text: bool = True,
) -> dict[str, object]:
    return {
        "length": length,
        "id": 1,
        "text": f"index-only/{actor}.json" if text else None,
        "multiperson": [participant for participant in participants if participant != actor],
        "audio": "ignored-by-index-ingestion.wav",
    }


def _document(
    participants: tuple[str, ...] = ("actor-alpha", "actor-beta", "actor-gamma"),
    *,
    capture_name: str = "capture-one",
) -> dict[str, object]:
    return {actor: {capture_name: _actor_row(actor, participants)} for actor in participants}


def test_ingestion_collapses_per_actor_rows_to_one_capture() -> None:
    metadata = data.ingest_public_metadata({"acting": _document()})
    assert len(metadata.captures) == 1
    capture = metadata.captures[0]
    assert capture.key == ("acting", "capture-one")
    assert capture.participant_count == 3
    assert capture.length_frames == 90
    assert capture.has_complete_actor_text is True
    assert not hasattr(capture, "text_paths")
    assert "actor-alpha" not in repr(capture)
    assert "capture-one" not in repr(capture)
    assert data.eligible_group_text_captures(metadata) == (capture,)


def test_capture_identity_includes_subset_and_is_canonically_sorted() -> None:
    metadata = data.ingest_public_metadata(
        {"daylife": _document(capture_name="same"), "acting": _document(capture_name="same")}
    )
    assert tuple(capture.key for capture in metadata.captures) == (
        ("acting", "same"),
        ("daylife", "same"),
    )


def test_filter_requires_complete_actor_text_and_k_at_least_three() -> None:
    participants = ("actor-alpha", "actor-beta", "actor-gamma")
    document = _document(participants)
    document["actor-beta"]["capture-one"] = _actor_row(  # type: ignore[index]
        "actor-beta", participants, text=False
    )
    metadata = data.ingest_public_metadata({"acting": document})
    assert data.eligible_group_text_captures(metadata) == ()
    assert (
        data.eligible_group_text_captures(metadata, require_complete_actor_text=False)
        == metadata.captures
    )


def test_integer_class_labels_are_valid_index_rows_but_not_language_text() -> None:
    document = _document()
    document["actor-beta"]["capture-one"]["text"] = 4  # type: ignore[index]
    metadata = data.ingest_public_metadata({"acting": document})
    assert metadata.captures[0].has_complete_actor_text is False
    assert data.eligible_group_text_captures(metadata) == ()


def test_conflicting_or_incomplete_actor_rows_fail_closed() -> None:
    participants = ("actor-alpha", "actor-beta", "actor-gamma")
    conflicting = _document(participants)
    conflicting["actor-beta"]["capture-one"] = _actor_row(  # type: ignore[index]
        "actor-beta", participants, length=91
    )
    with pytest.raises(data.PhaseSetDataError, match="disagree"):
        data.ingest_public_metadata({"acting": conflicting})

    incomplete = _document(participants)
    del incomplete["actor-gamma"]
    with pytest.raises(data.PhaseSetDataError, match="missing"):
        data.ingest_public_metadata({"acting": incomplete})


def test_json_decoder_rejects_duplicate_keys_and_non_utf8() -> None:
    duplicate = b'{"actor":{},"actor":{}}'
    with pytest.raises(data.PhaseSetDataError, match="duplicate"):
        data.decode_public_metadata_json(duplicate)
    with pytest.raises(data.PhaseSetDataError, match="UTF-8"):
        data.decode_public_metadata_json(b"\xff")
    with pytest.raises(data.PhaseSetDataError, match="forbidden constant"):
        data.decode_public_metadata_json(b'{"actor":NaN}')


def test_decoded_json_round_trip_keeps_only_index_level_booleans() -> None:
    raw = json.dumps(_document(), sort_keys=True).encode("utf-8")
    decoded = data.decode_public_metadata_json(raw)
    metadata = data.ingest_public_metadata({"acting": decoded})
    capture = metadata.captures[0]
    assert capture.text_participants == capture.participants
    assert "index-only" not in repr(capture)


def test_official_source_allowlist_is_exact_and_public() -> None:
    assert tuple(data.OFFICIAL_METADATA_URLS) == ("acting", "daylife")
    assert all(
        url.startswith("https://raw.githubusercontent.com/facebookresearch/embody-3d/")
        for url in data.OFFICIAL_METADATA_URLS.values()
    )
