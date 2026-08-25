"""Synthetic-only checks for strict PhasePair caption processing."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from phasepair_core import caption_processing as captions


def _raw32(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _batch_fixture() -> dict[str, object]:
    pair_commitments = (_raw32(100), _raw32(200))
    caption_commitments = (
        (_raw32(10), _raw32(11), _raw32(12)),
        (_raw32(20), _raw32(21), _raw32(22)),
    )
    payloads = (
        (b"Walk left", b"Walk\tright", "Cafe\u0301".encode()),
        (b"Turn", b"Clap twice", b"Sit down"),
    )
    input_ids = np.full((6, 77), 1, dtype=np.int64)
    attention_mask = np.zeros((6, 77), dtype=np.int64)
    for row in range(6):
        input_ids[row, :3] = [0, 10 + row, 2]
        attention_mask[row, :3] = 1
    return {
        "pair_commitments": pair_commitments,
        "caption_commitments": caption_commitments,
        "caption_ordinals": ((0, 1, 2), (0, 1, 2)),
        "caption_payloads": payloads,
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "tokenizer_manifest_sha256": _digest("tokenizer"),
        "caption_text_manifest_sha256": _digest("caption-text"),
        "caption_lineage_manifest_sha256": _digest("caption-lineage"),
        "normalizer_manifest_sha256": _digest("normalizer"),
        "bos_token_id": 0,
        "eos_token_id": 2,
        "pad_token_id": 1,
        "vocab_size": 100,
    }


def test_three_caption_parser_accepts_lf_crlf_and_optional_terminal_lf() -> None:
    assert captions.parse_three_caption_object(b"one\ntwo\nthree") == (
        b"one",
        b"two",
        b"three",
    )
    assert captions.parse_three_caption_object(b"one\r\ntwo\r\nthree\r\n") == (
        b"one",
        b"two",
        b"three",
    )
    assert captions.parse_three_caption_object("甲\n乙\n丙\n".encode()) == tuple(
        value.encode() for value in ("甲", "乙", "丙")
    )


@pytest.mark.parametrize(
    "raw,message",
    [
        (b"", "empty object"),
        (b"one\ntwo", "exactly 3"),
        (b"one\ntwo\nthree\nfour", "exactly 3"),
        (b"one\ntwo\nthree\n\n", "nonempty"),
        (b"one\n\nthree", "nonempty"),
        (b"one\rtwo\nsecond\nthird", "lone CR"),
        (b"one\r\r\nsecond\nthird", "lone CR"),
        (b"\xef\xbb\xbfone\ntwo\nthree", "BOM"),
        (b"one\n\xff\nthree", "strict UTF-8"),
        (b"one\x00\ntwo\nthree", "NUL"),
    ],
)
def test_three_caption_parser_fails_closed(raw: bytes, message: str) -> None:
    with pytest.raises(captions.CaptionProcessingError, match=message):
        captions.parse_three_caption_object(raw)
    with pytest.raises(TypeError, match="exact built-in bytes"):
        captions.parse_three_caption_object(bytearray(b"one\ntwo\nthree"))


def test_normalizer_is_nfc_and_uses_only_the_six_frozen_whitespace_codepoints() -> None:
    raw = "\t  Cafe\u0301\v\fTEST  \t".encode()
    assert captions.normalize_caption_payload(raw) == "Café TEST".encode()
    assert captions.normalize_caption_payload("\u00a0A\u00a0".encode()) == (
        "\u00a0A\u00a0".encode()
    )
    assert captions.normalize_caption_payload(b"Mixed CASE") == b"Mixed CASE"
    assert captions.normalize_caption_payload(b" \t\v\f ") == b""
    with pytest.raises(captions.CaptionProcessingError, match="line terminator"):
        captions.normalize_caption_payload(b"a\nb")


def test_normalizer_manifest_is_exact_canonical_and_runtime_bound() -> None:
    runtime = _digest("python-runtime")
    raw = captions.caption_normalizer_manifest_bytes(
        python_runtime_manifest_sha256=runtime
    )
    payload = json.loads(raw)
    assert payload == {
        "algorithm": "NFC_THEN_TRIM_W_THEN_COLLAPSE_W_V1",
        "python_runtime_manifest_sha256": runtime,
        "schema": "phasepair-caption-normalizer-manifest-v1",
        "unicodedata_unidata_version": payload["unicodedata_unidata_version"],
        "whitespace_codepoints": [9, 10, 11, 12, 13, 32],
    }
    assert raw == (
        json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        + b"\n"
    )
    expected = hashlib.sha256(
        b"phasepair-caption-normalizer-manifest-v1\x00" + raw
    ).hexdigest()
    assert (
        captions.caption_normalizer_manifest_sha256(
            python_runtime_manifest_sha256=runtime
        )
        == expected
    )


def test_prepared_batch_binds_three_rows_per_pair_and_hides_raw_text_from_receipt() -> None:
    batch = captions.PreparedCaptionBatch(**_batch_fixture())
    raw = captions.prepared_caption_batch_bytes(batch)
    payload = json.loads(raw)

    assert payload["schema"] == "phasepair-prepared-caption-batch-v1"
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["execution_authorized"] is False
    assert payload["result_claimed"] is False
    assert payload["status"] == captions.STATUS
    assert payload["batch_size"] == 2
    assert payload["caption_count"] == 6
    assert payload["tokenizer_max_length"] == 77
    assert payload["tokenizer_truncation"] is True
    assert payload["tokenizer_padding"] == "max_length"
    assert payload["row_order"] == (
        "pair_batch_order_outer_sampler_caption_order_inner"
    )
    assert "active_lengths" not in payload
    assert payload["tokenizer_mapping_verified"] is False
    assert payload["tokenizer_mapping_authority"] == (
        "CALLER_FROZEN_ARRAYS_AWAITING_EXECUTION_ADAPTER"
    )
    assert b"Walk left" not in raw and "Café".encode() not in raw
    assert batch.normalized_payloads[0] == (
        b"Walk left",
        b"Walk right",
        "Café".encode(),
    )
    assert not batch.input_ids.flags.writeable
    assert not batch.attention_mask.flags.writeable
    assert captions.prepared_caption_batch_sha256(batch) == hashlib.sha256(
        raw
    ).hexdigest()


def test_batch_snapshots_arrays_and_is_deterministic() -> None:
    fixture = _batch_fixture()
    first = captions.PreparedCaptionBatch(**fixture)
    first_raw = captions.prepared_caption_batch_bytes(first)
    fixture["input_ids"][:] = 99
    fixture["attention_mask"][:] = 0
    assert captions.prepared_caption_batch_bytes(first) == first_raw
    second = captions.PreparedCaptionBatch(**_batch_fixture())
    assert captions.prepared_caption_batch_bytes(second) == first_raw
    with pytest.raises(AttributeError, match="immutable"):
        first.input_ids = np.zeros((6, 77), dtype=np.int64)


class _ArraySubclass(np.ndarray):
    pass


@pytest.mark.parametrize(
    "mutation,message",
    [
        (lambda value: value.update(input_ids=value["input_ids"].astype(np.int32)), "dtype int64"),
        (lambda value: value.update(input_ids=value["input_ids"][:, ::-1]), "C-contiguous"),
        (lambda value: value.update(input_ids=value["input_ids"][:-1].copy()), "shape"),
        (lambda value: value["input_ids"].__setitem__((0, 1), 100), "out-of-vocabulary"),
        (lambda value: value["attention_mask"].__setitem__((0, 1), 2), "binary"),
        (lambda value: value["attention_mask"].__setitem__((0, 4), 1), "active prefix"),
        (lambda value: value["input_ids"].__setitem__((0, 0), 9), "begin with BOS"),
        (lambda value: value["input_ids"].__setitem__((0, 2), 9), "end with EOS"),
        (lambda value: value["input_ids"].__setitem__((0, 4), 9), "inactive token"),
        (lambda value: value.update(vocab_size=True), "exact built-in int"),
        (lambda value: value.update(tokenizer_manifest_sha256="A" * 64), "lowercase SHA-256"),
        (
            lambda value: value.update(pair_commitments=(value["pair_commitments"][0],) * 2),
            "pair commitments must be distinct",
        ),
        (
            lambda value: value.update(
                caption_commitments=(value["caption_commitments"][0],) * 2
            ),
            "globally distinct",
        ),
        (
            lambda value: value.update(caption_ordinals=((0, 0, 2), (0, 1, 2))),
            "permutation",
        ),
    ],
)
def test_batch_rejects_numeric_identity_and_manifest_mutants(
    mutation, message: str
) -> None:
    fixture = _batch_fixture()
    mutation(fixture)
    with pytest.raises((TypeError, captions.CaptionProcessingError), match=message):
        captions.PreparedCaptionBatch(**fixture)


def test_batch_rejects_array_subclasses_and_noncanonical_payload_rows() -> None:
    fixture = _batch_fixture()
    fixture["input_ids"] = fixture["input_ids"].view(_ArraySubclass)
    with pytest.raises(TypeError, match="not a subclass"):
        captions.PreparedCaptionBatch(**fixture)

    fixture = _batch_fixture()
    fixture["caption_payloads"] = (
        (b"bad\nline", b"two", b"three"),
        fixture["caption_payloads"][1],
    )
    with pytest.raises(captions.CaptionProcessingError, match="line terminator"):
        captions.PreparedCaptionBatch(**fixture)

    fixture = _batch_fixture()
    fixture["caption_payloads"] = list(fixture["caption_payloads"])
    with pytest.raises(TypeError, match="exact tuple"):
        captions.PreparedCaptionBatch(**fixture)


def test_public_receipt_api_and_authority_constants_are_unambiguous() -> None:
    with pytest.raises(TypeError, match="exactly PreparedCaptionBatch"):
        captions.prepared_caption_batch_bytes(object())
    assert captions.AUTHORITY == 0
    assert captions.PRODUCTION is False
    assert captions.EXECUTION_AUTHORIZED is False
    assert captions.RESULT_CLAIMED is False
    assert captions.STATUS.endswith("AUTHORITY0_NO_RESULT")
    assert captions.CAPTIONS_PER_PAIR == 3
    assert captions.TOKENIZER_MAX_LENGTH == 77
    assert captions.NORMALIZER_WHITESPACE_CODEPOINTS == (9, 10, 11, 12, 13, 32)


def test_terminal_bare_cr_is_not_misread_as_crlf() -> None:
    with pytest.raises(captions.CaptionProcessingError, match="lone CR"):
        captions.parse_three_caption_object(b"one\r\ntwo\r\nthree\r")


def test_issued_batch_rejects_field_rebinding_and_unissued_exact_objects() -> None:
    batch = captions.PreparedCaptionBatch(**_batch_fixture())
    object.__setattr__(batch, "pair_commitments", tuple(reversed(batch.pair_commitments)))
    with pytest.raises(captions.CaptionProcessingError, match="fields were rebound"):
        captions.prepared_caption_batch_bytes(batch)

    forged = object.__new__(captions.PreparedCaptionBatch)
    object.__setattr__(forged, "canonical_receipt", b"FORGED\n")
    with pytest.raises(captions.CaptionProcessingError, match="not issued"):
        captions.prepared_caption_batch_bytes(forged)
