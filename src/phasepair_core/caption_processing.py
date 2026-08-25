"""Strict private-caption parsing, normalization, and token-batch binding.

The parser and normalizer implement the frozen PhasePair byte rules.  The
batch builder snapshots caller-produced tokenizer arrays shaped ``[3B,77]``
and binds them to pair/caption commitments and original caption payloads.  It
does not read files, invoke a tokenizer, expose caption text in its receipt,
authorize training, or claim an experiment result.
"""

from __future__ import annotations

import hashlib
import json
import struct
import threading
import unicodedata

import numpy as np

from phasepair_core._identity_registry import make_identity_weak_registry


AUTHORITY = 0
PRODUCTION = False
EXECUTION_AUTHORIZED = False
RESULT_CLAIMED = False
STATUS = "DATA_FREE_CAPTION_PROCESSING_AUTHORITY0_NO_RESULT"
CAPTIONS_PER_PAIR = 3
TOKENIZER_MAX_LENGTH = 77
NORMALIZER_ALGORITHM = "NFC_THEN_TRIM_W_THEN_COLLAPSE_W_V1"
NORMALIZER_WHITESPACE_CODEPOINTS = (9, 10, 11, 12, 13, 32)
_NORMALIZER_WHITESPACE = frozenset(chr(value) for value in NORMALIZER_WHITESPACE_CODEPOINTS)
_UTF8_BOM = b"\xef\xbb\xbf"


class CaptionProcessingError(ValueError):
    """A caption byte, identity, or tokenizer-batch contract was violated."""


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise TypeError(f"{label} must be exact built-in bytes[32]")
    return value


def _lower_sha256(value: object, label: str) -> str:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact built-in str")
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise CaptionProcessingError(f"{label} must be lowercase SHA-256 hex")
    return value


def _exact_int(
    value: object,
    label: str,
    *,
    minimum: int,
    maximum: int,
) -> int:
    if type(value) is not int:
        raise TypeError(f"{label} must be an exact built-in int")
    if value < minimum or value > maximum:
        raise CaptionProcessingError(
            f"{label} must be inside the closed interval [{minimum},{maximum}]"
        )
    return value


def _caption_payload(value: object, label: str) -> bytes:
    if type(value) is not bytes:
        raise TypeError(f"{label} must be exact built-in bytes")
    if not value:
        raise CaptionProcessingError(f"{label} must be nonempty")
    if value.startswith(_UTF8_BOM):
        raise CaptionProcessingError(f"{label} must not contain a UTF-8 BOM")
    if b"\r" in value or b"\n" in value:
        raise CaptionProcessingError(f"{label} must not contain a line terminator")
    if b"\x00" in value:
        raise CaptionProcessingError(f"{label} must not contain NUL")
    try:
        value.decode("utf-8", errors="strict")
    except UnicodeDecodeError as exc:
        raise CaptionProcessingError(f"{label} must be strict UTF-8") from exc
    return value


def parse_three_caption_object(value: object) -> tuple[bytes, bytes, bytes]:
    """Parse exactly three nonempty strict-UTF-8 records from frozen bytes.

    LF separates records.  A single CR immediately before an LF is removed;
    every other CR is a hard failure.  One terminal LF terminates the third
    record and does not create a fourth record.
    """

    if type(value) is not bytes:
        raise TypeError("caption object must be exact built-in bytes")
    if not value:
        raise CaptionProcessingError("CAPTION_PARSE_FAIL: empty object")
    if value.startswith(_UTF8_BOM):
        raise CaptionProcessingError("CAPTION_PARSE_FAIL: UTF-8 BOM")
    terminal_lf = value.endswith(b"\n")
    parts = value.split(b"\n")
    if terminal_lf:
        parts.pop()
    records: list[bytes] = []
    for ordinal, part in enumerate(parts):
        immediately_precedes_lf = ordinal < len(parts) - 1 or terminal_lf
        if immediately_precedes_lf and part.endswith(b"\r"):
            part = part[:-1]
        if b"\r" in part:
            raise CaptionProcessingError("CAPTION_PARSE_FAIL: lone CR")
        try:
            records.append(_caption_payload(part, f"caption[{ordinal}]"))
        except (TypeError, CaptionProcessingError) as exc:
            raise CaptionProcessingError(f"CAPTION_PARSE_FAIL: {exc}") from exc
    if len(records) != CAPTIONS_PER_PAIR:
        raise CaptionProcessingError("CAPTION_PARSE_FAIL: object must contain exactly 3 records")
    return records[0], records[1], records[2]


def normalize_caption_payload(value: object) -> bytes:
    """Apply NFC, trim only W, then collapse each internal W run to U+0020."""

    payload = _caption_payload(value, "caption payload")
    text = unicodedata.normalize("NFC", payload.decode("utf-8", errors="strict"))
    start = 0
    while start < len(text) and text[start] in _NORMALIZER_WHITESPACE:
        start += 1
    end = len(text)
    while end > start and text[end - 1] in _NORMALIZER_WHITESPACE:
        end -= 1
    output: list[str] = []
    in_whitespace = False
    for character in text[start:end]:
        if character in _NORMALIZER_WHITESPACE:
            if not in_whitespace:
                output.append(" ")
            in_whitespace = True
        else:
            output.append(character)
            in_whitespace = False
    normalized = "".join(output).encode("utf-8", errors="strict")
    if normalized.startswith(_UTF8_BOM) or b"\r" in normalized or b"\n" in normalized:
        raise CaptionProcessingError("normalized caption contains a forbidden marker")
    if b"\x00" in normalized:
        raise CaptionProcessingError("normalized caption contains NUL")
    return normalized


def caption_normalizer_manifest_bytes(
    *,
    python_runtime_manifest_sha256: str,
) -> bytes:
    """Return the exact public normalizer manifest for this Python runtime."""

    runtime = _lower_sha256(
        python_runtime_manifest_sha256, "python_runtime_manifest_sha256"
    )
    payload = {
        "algorithm": NORMALIZER_ALGORITHM,
        "python_runtime_manifest_sha256": runtime,
        "schema": "phasepair-caption-normalizer-manifest-v1",
        "unicodedata_unidata_version": unicodedata.unidata_version,
        "whitespace_codepoints": list(NORMALIZER_WHITESPACE_CODEPOINTS),
    }
    return _canonical_json_bytes(payload)


def caption_normalizer_manifest_sha256(
    *,
    python_runtime_manifest_sha256: str,
) -> str:
    """Return ``DH(schema, manifest)`` as lowercase hex."""

    raw = caption_normalizer_manifest_bytes(
        python_runtime_manifest_sha256=python_runtime_manifest_sha256
    )
    return hashlib.sha256(b"phasepair-caption-normalizer-manifest-v1\x00" + raw).hexdigest()


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


def _immutable_int64_matrix(value: object, label: str) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise TypeError(f"{label} must be an exact numpy.ndarray, not a subclass")
    if value.dtype != np.dtype(np.int64):
        raise CaptionProcessingError(f"{label} must have exact dtype int64")
    if value.ndim != 2:
        raise CaptionProcessingError(f"{label} must be two-dimensional")
    if not value.flags.c_contiguous:
        raise CaptionProcessingError(f"{label} must be C-contiguous")
    copied = np.array(value, dtype=np.int64, copy=True, order="C", subok=False)
    return np.frombuffer(copied.tobytes(order="C"), dtype=np.int64).reshape(copied.shape)


def _tuple_rows(value: object, label: str, *, row_count: int) -> tuple[tuple[object, ...], ...]:
    if type(value) is not tuple or len(value) != row_count:
        raise TypeError(f"{label} must be an exact tuple with {row_count} rows")
    rows: list[tuple[object, ...]] = []
    for row_index, row in enumerate(value):
        if type(row) is not tuple or len(row) != CAPTIONS_PER_PAIR:
            raise TypeError(
                f"{label}[{row_index}] must be an exact tuple with 3 items"
            )
        rows.append(row)
    return tuple(rows)


def _array_sha256(tag: bytes, value: np.ndarray) -> str:
    canonical = value.astype("<i8", copy=False).tobytes(order="C")
    return hashlib.sha256(
        tag + b"\x00" + struct.pack(">II", *value.shape) + canonical
    ).hexdigest()


class PreparedCaptionBatch:
    """Immutable, internally validated caption/token batch."""

    __slots__ = (
        "__weakref__",
        "attention_mask",
        "canonical_receipt",
        "caption_commitments",
        "caption_ordinals",
        "caption_payloads",
        "input_ids",
        "normalized_payloads",
        "pair_commitments",
    )

    def __init__(
        self,
        *,
        pair_commitments: object,
        caption_commitments: object,
        caption_ordinals: object,
        caption_payloads: object,
        input_ids: object,
        attention_mask: object,
        tokenizer_manifest_sha256: str,
        caption_text_manifest_sha256: str,
        caption_lineage_manifest_sha256: str,
        normalizer_manifest_sha256: str,
        bos_token_id: int,
        eos_token_id: int,
        pad_token_id: int,
        vocab_size: int,
    ) -> None:
        if type(pair_commitments) is not tuple or not pair_commitments:
            raise TypeError("pair_commitments must be an exact nonempty tuple")
        pairs = tuple(
            _raw32(value, f"pair_commitments[{index}]")
            for index, value in enumerate(pair_commitments)
        )
        if len(set(pairs)) != len(pairs):
            raise CaptionProcessingError("pair commitments must be distinct")
        batch_size = len(pairs)
        commitment_rows = _tuple_rows(
            caption_commitments, "caption_commitments", row_count=batch_size
        )
        payload_rows = _tuple_rows(
            caption_payloads, "caption_payloads", row_count=batch_size
        )
        ordinal_rows = _tuple_rows(
            caption_ordinals, "caption_ordinals", row_count=batch_size
        )
        checked_commitments = tuple(
            tuple(
                _raw32(value, f"caption_commitments[{row}][{ordinal}]")
                for ordinal, value in enumerate(values)
            )
            for row, values in enumerate(commitment_rows)
        )
        flattened_commitments = tuple(
            value for row in checked_commitments for value in row
        )
        if len(set(flattened_commitments)) != CAPTIONS_PER_PAIR * batch_size:
            raise CaptionProcessingError("caption commitments must be globally distinct")
        checked_ordinals = tuple(
            tuple(
                _exact_int(
                    value,
                    f"caption_ordinals[{row}][{column}]",
                    minimum=0,
                    maximum=2,
                )
                for column, value in enumerate(values)
            )
            for row, values in enumerate(ordinal_rows)
        )
        if any(tuple(sorted(row)) != (0, 1, 2) for row in checked_ordinals):
            raise CaptionProcessingError(
                "each caption_ordinals row must be a permutation of (0,1,2)"
            )
        checked_payloads = tuple(
            tuple(
                _caption_payload(value, f"caption_payloads[{row}][{ordinal}]")
                for ordinal, value in enumerate(values)
            )
            for row, values in enumerate(payload_rows)
        )
        normalized = tuple(
            tuple(normalize_caption_payload(value) for value in row)
            for row in checked_payloads
        )

        ids = _immutable_int64_matrix(input_ids, "input_ids")
        mask = _immutable_int64_matrix(attention_mask, "attention_mask")
        expected_shape = (CAPTIONS_PER_PAIR * batch_size, TOKENIZER_MAX_LENGTH)
        if ids.shape != expected_shape or mask.shape != expected_shape:
            raise CaptionProcessingError(
                f"input_ids and attention_mask must both have shape {expected_shape}"
            )
        vocab = _exact_int(vocab_size, "vocab_size", minimum=3, maximum=2**31 - 1)
        bos = _exact_int(bos_token_id, "bos_token_id", minimum=0, maximum=vocab - 1)
        eos = _exact_int(eos_token_id, "eos_token_id", minimum=0, maximum=vocab - 1)
        pad = _exact_int(pad_token_id, "pad_token_id", minimum=0, maximum=vocab - 1)
        if bool(np.any(ids < 0)) or bool(np.any(ids >= vocab)):
            raise CaptionProcessingError("input_ids contain an out-of-vocabulary token")
        if not bool(np.logical_or(mask == 0, mask == 1).all()):
            raise CaptionProcessingError("attention_mask must be binary int64")
        for row in range(expected_shape[0]):
            zeros = np.flatnonzero(mask[row] == 0)
            active_length = int(zeros[0]) if zeros.size else TOKENIZER_MAX_LENGTH
            if active_length < 2:
                raise CaptionProcessingError("each token row must contain BOS and EOS")
            expected_mask = np.zeros(TOKENIZER_MAX_LENGTH, dtype=np.int64)
            expected_mask[:active_length] = 1
            if not bool(np.array_equal(mask[row], expected_mask)):
                raise CaptionProcessingError("attention_mask must be an exact active prefix")
            if int(ids[row, 0]) != bos or int(ids[row, active_length - 1]) != eos:
                raise CaptionProcessingError("token row must begin with BOS and end with EOS")
            if active_length < TOKENIZER_MAX_LENGTH and not bool(
                np.all(ids[row, active_length:] == pad)
            ):
                raise CaptionProcessingError("inactive token positions must equal PAD")

        tokenizer_manifest = _lower_sha256(
            tokenizer_manifest_sha256, "tokenizer_manifest_sha256"
        )
        caption_text_manifest = _lower_sha256(
            caption_text_manifest_sha256, "caption_text_manifest_sha256"
        )
        caption_lineage_manifest = _lower_sha256(
            caption_lineage_manifest_sha256, "caption_lineage_manifest_sha256"
        )
        normalizer_manifest = _lower_sha256(
            normalizer_manifest_sha256, "normalizer_manifest_sha256"
        )

        identity_bytes = b"".join(
            pairs[row]
            + checked_commitments[row][ordinal]
            + bytes((checked_ordinals[row][ordinal],))
            + struct.pack(">I", len(checked_payloads[row][ordinal]))
            + checked_payloads[row][ordinal]
            for row in range(batch_size)
            for ordinal in range(CAPTIONS_PER_PAIR)
        )
        normalized_bytes = b"".join(
            pairs[row]
            + checked_commitments[row][ordinal]
            + bytes((checked_ordinals[row][ordinal],))
            + struct.pack(">I", len(normalized[row][ordinal]))
            + normalized[row][ordinal]
            for row in range(batch_size)
            for ordinal in range(CAPTIONS_PER_PAIR)
        )
        payload = {
            "attention_mask_sha256": _array_sha256(
                b"phasepair-caption-attention-mask-i64-v1", mask
            ),
            "authority": AUTHORITY,
            "batch_size": batch_size,
            "bos_token_id": bos,
            "caption_count": CAPTIONS_PER_PAIR * batch_size,
            "caption_identity_sha256": hashlib.sha256(
                b"phasepair-caption-batch-identity-v1\x00" + identity_bytes
            ).hexdigest(),
            "caption_lineage_manifest_sha256": caption_lineage_manifest,
            "caption_ordinal_order_sha256": hashlib.sha256(
                b"phasepair-caption-original-ordinal-order-v1\x00"
                + bytes(value for row in checked_ordinals for value in row)
            ).hexdigest(),
            "caption_normalized_sha256": hashlib.sha256(
                b"phasepair-caption-batch-normalized-v1\x00" + normalized_bytes
            ).hexdigest(),
            "caption_text_manifest_sha256": caption_text_manifest,
            "eos_token_id": eos,
            "execution_authorized": EXECUTION_AUTHORIZED,
            "input_ids_sha256": _array_sha256(
                b"phasepair-caption-input-ids-i64-v1", ids
            ),
            "normalizer_manifest_sha256": normalizer_manifest,
            "pad_token_id": pad,
            "production": PRODUCTION,
            "result_claimed": RESULT_CLAIMED,
            "row_order": "pair_batch_order_outer_sampler_caption_order_inner",
            "schema": "phasepair-prepared-caption-batch-v1",
            "status": STATUS,
            "tokenizer_manifest_sha256": tokenizer_manifest,
            "tokenizer_mapping_authority": (
                "CALLER_FROZEN_ARRAYS_AWAITING_EXECUTION_ADAPTER"
            ),
            "tokenizer_mapping_verified": False,
            "tokenizer_max_length": TOKENIZER_MAX_LENGTH,
            "tokenizer_padding": "max_length",
            "tokenizer_truncation": True,
            "vocab_size": vocab,
        }
        object.__setattr__(self, "pair_commitments", pairs)
        object.__setattr__(self, "caption_commitments", checked_commitments)
        object.__setattr__(self, "caption_ordinals", checked_ordinals)
        object.__setattr__(self, "caption_payloads", checked_payloads)
        object.__setattr__(self, "normalized_payloads", normalized)
        object.__setattr__(self, "input_ids", ids)
        object.__setattr__(self, "attention_mask", mask)
        object.__setattr__(self, "canonical_receipt", _canonical_json_bytes(payload))
        record = (
            pairs,
            checked_commitments,
            checked_ordinals,
            checked_payloads,
            normalized,
            ids,
            mask,
            object.__getattribute__(self, "canonical_receipt"),
        )
        with _caption_batch_lock:
            _caption_batch_set(self, record)

    def __setattr__(self, _name: str, _value: object) -> None:
        raise AttributeError("PreparedCaptionBatch is immutable")


def prepared_caption_batch_bytes(value: object) -> bytes:
    """Return the canonical public-safe Authority-0 batch receipt."""

    return _validated_caption_batch_record(value)[7]


def prepared_caption_batch_sha256(value: object) -> str:
    """Return the lowercase SHA-256 identity of one prepared batch receipt."""

    return hashlib.sha256(prepared_caption_batch_bytes(value)).hexdigest()


_caption_batch_lock = threading.RLock()
_caption_batch_set, _caption_batch_get, _, _, _ = make_identity_weak_registry(
    _caption_batch_lock
)


def _validated_caption_batch_record(value: object) -> tuple[object, ...]:
    if type(value) is not PreparedCaptionBatch:
        raise TypeError("batch must be exactly PreparedCaptionBatch")
    with _caption_batch_lock:
        record = _caption_batch_get(value)
        if type(record) is not tuple or len(record) != 8:
            raise CaptionProcessingError("caption batch was not issued by this module")
        field_names = (
            "pair_commitments",
            "caption_commitments",
            "caption_ordinals",
            "caption_payloads",
            "normalized_payloads",
            "input_ids",
            "attention_mask",
            "canonical_receipt",
        )
        try:
            observed = tuple(
                object.__getattribute__(value, name) for name in field_names
            )
        except AttributeError as exc:
            raise CaptionProcessingError("caption batch fields were rebound") from exc
        if any(left is not right for left, right in zip(observed, record, strict=True)):
            raise CaptionProcessingError("caption batch fields were rebound")
        return record


def validate_prepared_caption_batch(value: object) -> PreparedCaptionBatch:
    """Prove module issuance and unchanged immutable component identities."""

    _validated_caption_batch_record(value)
    return value


def prepared_caption_batch_components(value: object) -> tuple[object, ...]:
    """Return the immutable issued components for package-internal joins."""

    return _validated_caption_batch_record(value)


__all__ = [
    "AUTHORITY",
    "CAPTIONS_PER_PAIR",
    "CaptionProcessingError",
    "EXECUTION_AUTHORIZED",
    "NORMALIZER_ALGORITHM",
    "NORMALIZER_WHITESPACE_CODEPOINTS",
    "PRODUCTION",
    "PreparedCaptionBatch",
    "RESULT_CLAIMED",
    "STATUS",
    "TOKENIZER_MAX_LENGTH",
    "caption_normalizer_manifest_bytes",
    "caption_normalizer_manifest_sha256",
    "normalize_caption_payload",
    "parse_three_caption_object",
    "prepared_caption_batch_bytes",
    "prepared_caption_batch_components",
    "prepared_caption_batch_sha256",
    "validate_prepared_caption_batch",
]
