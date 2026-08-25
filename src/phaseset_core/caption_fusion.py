"""Data-free contract for auxiliary machine-fused group captions.

The public package does not call a model or serialize caption text.  A trusted
host may inject a backend that accepts only the frozen prompt plus human text
from one ten-second window.  Actor commitments are consumed locally to make
the request order canonical; they are never exposed to the backend or receipt.

The resulting text is explicitly machine-fused, auxiliary, authority zero,
and provisionally reviewed by the same model family.  It is not a human group
annotation and cannot close a scientific execution or semantic-review gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import re
import unicodedata
from typing import Final, Protocol


MODEL_ID: Final = "gpt-5.6-sol"
PROMPT_VERSION: Final = "phaseset-group-caption-prompt-v1"
REQUEST_SCHEMA: Final = "phaseset-group-caption-request-v1"
OUTPUT_SCHEMA: Final = "phaseset-group-caption-output-v1"
RECEIPT_SCHEMA: Final = "phaseset-group-caption-receipt-v1"
PROVENANCE_SCHEMA: Final = "phaseset-group-caption-provenance-v1"
RETRY_POLICY_VERSION: Final = "phaseset-group-caption-retry-v1"
PERMUTATION_AUDIT_VERSION: Final = "phaseset-group-caption-permutation-audit-v1"
ANNOTATION_KIND: Final = "MACHINE_FUSED_GROUP_CAPTION"
TASK_SCOPE: Final = "AUXILIARY_10S_GROUP_CAPTION_RETRIEVAL_ONLY"
SAME_FAMILY_REVIEW_STATUS: Final = (
    "PROVISIONAL_SAME_MODEL_FAMILY_REVIEW_NOT_GATE_CLOSING"
)
HUMAN_ACTOR_SOURCE: Final = "OFFICIAL_HUMAN_10S_ACTOR_DESCRIPTION"
HUMAN_HOLISTIC_SOURCE: Final = "OFFICIAL_HUMAN_HOLISTIC_CONTEXT"
MAX_ATTEMPTS: Final = 3
PERMUTATION_AUDIT_RUNS: Final = 3
MAX_INPUT_TEXT_BYTES: Final = 16 * 1024
MAX_OUTPUT_TEXT_BYTES: Final = 8 * 1024
TEMPERATURE_DECIMAL: Final = "0"
TOP_P_DECIMAL: Final = "1"
GENERATION_SEED: Final = 1729
MAX_OUTPUT_TOKENS: Final = 512
RESPONSE_FORMAT: Final = "STRICT_JSON_SCHEMA_GROUP_DESCRIPTION_AND_SEMANTIC_PARAPHRASE"

FROZEN_PROMPT: Final = """\
You produce one auxiliary machine-fused group caption for exactly one 10-second
multi-person window. The only evidence you may use is the supplied set of at
least three official human per-actor descriptions and, when present, official
human holistic context from that same window. Treat people as anonymous and do
not depend on their input order. Preserve supported actions, interactions, and
temporal relations without inventing identities or facts. You must not request,
infer, or use skeleton coordinates, motion-model features, spectra, frequency
bands, phase, lag, coherence, model scores, dataset-split statistics, or any
cross-window information. Return exactly one group_description and one
meaning-preserving semantic_paraphrase. Both fields are machine-fused auxiliary
text and must never be represented as human group annotation.
Never identify a person by input position or with first/second/third,
person one/two, former/latter, above/below, or similar order-dependent deixis.
"""
PROMPT_SHA256: Final = hashlib.sha256(FROZEN_PROMPT.encode("utf-8")).hexdigest()
RETRY_POLICY_SHA256: Final = hashlib.sha256(
    (
        f"{RETRY_POLICY_VERSION}|max_attempts={MAX_ATTEMPTS}|"
        "retry=CaptionFusionTransientError|delay=none"
    ).encode("ascii")
).hexdigest()

_SPLITS: Final = ("train", "val", "test")


class CaptionFusionContractError(ValueError):
    """Caption input, provenance, backend output, or receipt is invalid."""


class CaptionFusionTransientError(RuntimeError):
    """The sole backend failure class eligible for the frozen retry policy."""


def _raw32(value: object, label: str) -> bytes:
    if type(value) is not bytes or len(value) != 32:
        raise CaptionFusionContractError(f"{label} must be exact bytes[32]")
    return value


def _lower_sha256(value: object, label: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise CaptionFusionContractError(f"{label} must be lowercase SHA-256 hex")
    return value


def _normalized_text(value: object, label: str, *, byte_limit: int) -> str:
    if type(value) is not str:
        raise CaptionFusionContractError(f"{label} must be exact built-in str")
    if "\x00" in value:
        raise CaptionFusionContractError(f"{label} must not contain NUL")
    normalized = " ".join(unicodedata.normalize("NFKC", value).split())
    if not normalized:
        raise CaptionFusionContractError(f"{label} must be nonempty")
    try:
        encoded = normalized.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise CaptionFusionContractError(f"{label} must be valid Unicode text") from exc
    if len(encoded) > byte_limit:
        raise CaptionFusionContractError(f"{label} exceeds its UTF-8 byte limit")
    return normalized


def _optional_normalized_text(value: object, label: str, *, byte_limit: int) -> str | None:
    if value is None:
        return None
    return _normalized_text(value, label, byte_limit=byte_limit)


def _semantic_key(value: str) -> str:
    return _normalized_text(value, "semantic text", byte_limit=MAX_OUTPUT_TEXT_BYTES).casefold()


_ORDER_DEICTIC = re.compile(
    r"\b(?:first|second|third|fourth|fifth|last)\s+(?:described\s+)?"
    r"(?:person|actor|participant)\b|"
    r"\b(?:person|actor|participant)\s+(?:one|two|three|four|five|[1-9])\b|"
    r"\b(?:former|latter|above-described|below-described)\b",
    flags=re.IGNORECASE,
)


def _reject_order_deixis(value: str) -> None:
    if _ORDER_DEICTIC.search(value):
        raise CaptionFusionContractError(
            "generated caption contains an order-dependent person reference"
        )


def _update_digest(digest: object, label: bytes, value: bytes) -> None:
    if not isinstance(digest, type(hashlib.sha256())):
        raise TypeError("digest must be a hashlib SHA-256 object")
    digest.update(len(label).to_bytes(4, "big"))
    digest.update(label)
    digest.update(len(value).to_bytes(8, "big"))
    digest.update(value)


@dataclass(frozen=True, slots=True, repr=False)
class HumanActorCaption:
    """One asserted official human actor description from one window."""

    text: str = field(repr=False)
    actor_commitment: bytes = field(repr=False)
    window_commitment: bytes = field(repr=False)
    source_record_sha256: str = field(repr=False)
    source_kind: str = HUMAN_ACTOR_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _normalized_text(self.text, "actor caption", byte_limit=MAX_INPUT_TEXT_BYTES),
        )
        _raw32(self.actor_commitment, "actor_commitment")
        _raw32(self.window_commitment, "window_commitment")
        _lower_sha256(self.source_record_sha256, "source_record_sha256")
        if self.source_kind != HUMAN_ACTOR_SOURCE:
            raise CaptionFusionContractError(
                f"actor caption source must be {HUMAN_ACTOR_SOURCE}"
            )

    def __repr__(self) -> str:
        return "HumanActorCaption(<redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class HumanHolisticContext:
    """Optional official human holistic context for the identical window."""

    text: str = field(repr=False)
    window_commitment: bytes = field(repr=False)
    source_record_sha256: str = field(repr=False)
    source_kind: str = HUMAN_HOLISTIC_SOURCE

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "text",
            _normalized_text(self.text, "holistic context", byte_limit=MAX_INPUT_TEXT_BYTES),
        )
        _raw32(self.window_commitment, "window_commitment")
        _lower_sha256(self.source_record_sha256, "source_record_sha256")
        if self.source_kind != HUMAN_HOLISTIC_SOURCE:
            raise CaptionFusionContractError(
                f"holistic context source must be {HUMAN_HOLISTIC_SOURCE}"
            )

    def __repr__(self) -> str:
        return "HumanHolisticContext(<redacted>)"


@dataclass(frozen=True, slots=True)
class CaptionFusionProvenance:
    """Digest-only human-caption and split provenance assertions.

    A test input requires two additional sealed receipts.  Train/validation
    inputs must not carry them, preventing a test receipt from being silently
    repurposed or an unsealed test input from entering fusion.
    """

    split: str
    dataset_manifest_sha256: str
    caption_manifest_sha256: str
    human_text_rights_assertion_sha256: str
    sealed_test_manifest_sha256: str | None = None
    sealed_test_caption_sha256: str | None = None
    schema: str = PROVENANCE_SCHEMA

    def __post_init__(self) -> None:
        if type(self.split) is not str or self.split not in _SPLITS:
            raise CaptionFusionContractError("split must be exactly train, val, or test")
        for name in (
            "dataset_manifest_sha256",
            "caption_manifest_sha256",
            "human_text_rights_assertion_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if self.schema != PROVENANCE_SCHEMA:
            raise CaptionFusionContractError(f"provenance schema must be {PROVENANCE_SCHEMA}")
        sealed = (self.sealed_test_manifest_sha256, self.sealed_test_caption_sha256)
        if self.split == "test":
            if any(value is None for value in sealed):
                raise CaptionFusionContractError(
                    "test fusion requires sealed test manifest and caption provenance"
                )
            _lower_sha256(sealed[0], "sealed_test_manifest_sha256")
            _lower_sha256(sealed[1], "sealed_test_caption_sha256")
        elif any(value is not None for value in sealed):
            raise CaptionFusionContractError(
                "sealed test provenance is forbidden outside the test split"
            )

    @property
    def test_provenance_asserted(self) -> bool:
        return self.split == "test"

    def digest(self) -> str:
        digest = hashlib.sha256(b"phaseset-caption-fusion-provenance-digest-v1")
        for label, value in (
            (b"schema", self.schema),
            (b"split", self.split),
            (b"dataset", self.dataset_manifest_sha256),
            (b"caption", self.caption_manifest_sha256),
            (b"rights", self.human_text_rights_assertion_sha256),
            (b"sealed-test-manifest", self.sealed_test_manifest_sha256 or "ABSENT"),
            (b"sealed-test-caption", self.sealed_test_caption_sha256 or "ABSENT"),
        ):
            _update_digest(digest, label, value.encode("ascii"))
        return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class CaptionFusionTestAdmission:
    """Private host evidence that one exact test-fusion grant was consumed."""

    provenance_sha256: str
    sealed_test_manifest_sha256: str
    sealed_test_caption_sha256: str
    external_grant_sha256: str
    consumption_sha256: str
    externally_verified: bool = True
    consumed: bool = True
    schema: str = "phaseset-caption-fusion-test-admission-v1"

    def __post_init__(self) -> None:
        for name in (
            "provenance_sha256",
            "sealed_test_manifest_sha256",
            "sealed_test_caption_sha256",
            "external_grant_sha256",
            "consumption_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if (
            self.externally_verified is not True
            or self.consumed is not True
            or self.schema != "phaseset-caption-fusion-test-admission-v1"
        ):
            raise CaptionFusionContractError("test admission is not verified and consumed")

    @property
    def sha256(self) -> str:
        value = {
            "consumed": self.consumed,
            "consumption_sha256": self.consumption_sha256,
            "external_grant_sha256": self.external_grant_sha256,
            "externally_verified": self.externally_verified,
            "provenance_sha256": self.provenance_sha256,
            "schema": self.schema,
            "sealed_test_caption_sha256": self.sealed_test_caption_sha256,
            "sealed_test_manifest_sha256": self.sealed_test_manifest_sha256,
        }
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        return hashlib.sha256(raw).hexdigest()


class CaptionFusionTestGate(Protocol):
    """Trusted-host gate that verifies and consumes an external test grant."""

    def verify_and_consume(
        self,
        provenance: CaptionFusionProvenance,
    ) -> CaptionFusionTestAdmission:
        """Return admission for this exact provenance after one consumption."""


@dataclass(frozen=True, slots=True, repr=False)
class CaptionFusionInput:
    """Raw in-memory input; this type deliberately has no serializer."""

    actor_captions: tuple[HumanActorCaption, ...] = field(repr=False)
    provenance: CaptionFusionProvenance = field(repr=False)
    holistic_context: HumanHolisticContext | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if type(self.actor_captions) is not tuple or len(self.actor_captions) < 3:
            raise CaptionFusionContractError(
                "fusion requires an exact tuple of at least three actor captions"
            )
        if any(type(item) is not HumanActorCaption for item in self.actor_captions):
            raise CaptionFusionContractError(
                "actor_captions may contain only exact HumanActorCaption values"
            )
        if type(self.provenance) is not CaptionFusionProvenance:
            raise CaptionFusionContractError("provenance must be exact CaptionFusionProvenance")
        if self.holistic_context is not None and type(self.holistic_context) is not HumanHolisticContext:
            raise CaptionFusionContractError(
                "holistic_context must be exact HumanHolisticContext or None"
            )
        actor_keys = tuple(item.actor_commitment for item in self.actor_captions)
        if len(set(actor_keys)) != len(actor_keys):
            raise CaptionFusionContractError("actor commitments must be distinct in one window")
        window_keys = {item.window_commitment for item in self.actor_captions}
        if len(window_keys) != 1:
            raise CaptionFusionContractError("all actor captions must come from the same window")
        window_key = next(iter(window_keys))
        if self.holistic_context is not None and self.holistic_context.window_commitment != window_key:
            raise CaptionFusionContractError(
                "holistic context must come from the same window as every actor caption"
            )

    @property
    def window_commitment(self) -> bytes:
        return self.actor_captions[0].window_commitment

    @property
    def actor_count(self) -> int:
        return len(self.actor_captions)

    def __repr__(self) -> str:
        return f"CaptionFusionInput(actor_count={self.actor_count}, <redacted>)"


@dataclass(frozen=True, slots=True)
class CaptionFusionBackendManifest:
    """Frozen model revision, runtime, decoding, and response-format contract."""

    model_revision_sha256: str
    backend_implementation_sha256: str
    inference_runtime_sha256: str
    model_id: str = MODEL_ID
    temperature_decimal: str = TEMPERATURE_DECIMAL
    top_p_decimal: str = TOP_P_DECIMAL
    generation_seed: int = GENERATION_SEED
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    response_format: str = RESPONSE_FORMAT
    schema: str = "phaseset-caption-fusion-backend-manifest-v1"

    def __post_init__(self) -> None:
        for name in (
            "model_revision_sha256",
            "backend_implementation_sha256",
            "inference_runtime_sha256",
        ):
            _lower_sha256(getattr(self, name), name)
        if (
            self.model_id != MODEL_ID
            or self.temperature_decimal != TEMPERATURE_DECIMAL
            or self.top_p_decimal != TOP_P_DECIMAL
            or self.generation_seed != GENERATION_SEED
            or self.max_output_tokens != MAX_OUTPUT_TOKENS
            or self.response_format != RESPONSE_FORMAT
            or self.schema != "phaseset-caption-fusion-backend-manifest-v1"
        ):
            raise CaptionFusionContractError("backend manifest changed a frozen runtime field")

    @property
    def sha256(self) -> str:
        value = {
            item: getattr(self, item)
            for item in (
                "backend_implementation_sha256",
                "generation_seed",
                "inference_runtime_sha256",
                "max_output_tokens",
                "model_id",
                "model_revision_sha256",
                "response_format",
                "schema",
                "temperature_decimal",
                "top_p_decimal",
            )
        }
        raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("ascii") + b"\n"
        return hashlib.sha256(raw).hexdigest()


@dataclass(frozen=True, slots=True, repr=False)
class CaptionFusionBackendRequest:
    """Closed text-only request passed to a host-injected backend."""

    actor_descriptions: tuple[str, ...] = field(repr=False)
    holistic_context: str | None = field(repr=False)
    model_id: str = MODEL_ID
    prompt: str = field(default=FROZEN_PROMPT, repr=False)
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = PROMPT_SHA256
    backend_manifest_sha256: str = ""
    model_revision_sha256: str = ""
    temperature_decimal: str = TEMPERATURE_DECIMAL
    top_p_decimal: str = TOP_P_DECIMAL
    generation_seed: int = GENERATION_SEED
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    response_format: str = RESPONSE_FORMAT
    schema: str = REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if type(self.actor_descriptions) is not tuple or len(self.actor_descriptions) < 3:
            raise CaptionFusionContractError("backend request requires at least three descriptions")
        checked = tuple(
            _normalized_text(value, "actor description", byte_limit=MAX_INPUT_TEXT_BYTES)
            for value in self.actor_descriptions
        )
        if checked != self.actor_descriptions:
            raise CaptionFusionContractError("backend actor descriptions must be normalized")
        checked_context = _optional_normalized_text(
            self.holistic_context,
            "holistic context",
            byte_limit=MAX_INPUT_TEXT_BYTES,
        )
        if checked_context != self.holistic_context:
            raise CaptionFusionContractError("backend holistic context must be normalized")
        if (
            self.model_id != MODEL_ID
            or self.prompt != FROZEN_PROMPT
            or self.prompt_version != PROMPT_VERSION
            or self.prompt_sha256 != PROMPT_SHA256
            or self.temperature_decimal != TEMPERATURE_DECIMAL
            or self.top_p_decimal != TOP_P_DECIMAL
            or self.generation_seed != GENERATION_SEED
            or self.max_output_tokens != MAX_OUTPUT_TOKENS
            or self.response_format != RESPONSE_FORMAT
            or self.schema != REQUEST_SCHEMA
        ):
            raise CaptionFusionContractError("backend request changed a frozen contract field")
        _lower_sha256(self.backend_manifest_sha256, "backend_manifest_sha256")
        _lower_sha256(self.model_revision_sha256, "model_revision_sha256")

    def __repr__(self) -> str:
        return f"CaptionFusionBackendRequest(actor_count={len(self.actor_descriptions)}, <redacted>)"


@dataclass(frozen=True, slots=True, repr=False)
class CaptionFusionBackendOutput:
    """Exactly two generated text fields plus frozen model/schema assertions."""

    group_description: str = field(repr=False)
    semantic_paraphrase: str = field(repr=False)
    model_id: str = MODEL_ID
    schema: str = OUTPUT_SCHEMA

    def __post_init__(self) -> None:
        group = _normalized_text(
            self.group_description,
            "group_description",
            byte_limit=MAX_OUTPUT_TEXT_BYTES,
        )
        paraphrase = _normalized_text(
            self.semantic_paraphrase,
            "semantic_paraphrase",
            byte_limit=MAX_OUTPUT_TEXT_BYTES,
        )
        if _semantic_key(group) == _semantic_key(paraphrase):
            raise CaptionFusionContractError(
                "semantic_paraphrase must be a distinct surface realization"
            )
        _reject_order_deixis(group)
        _reject_order_deixis(paraphrase)
        if self.model_id != MODEL_ID or self.schema != OUTPUT_SCHEMA:
            raise CaptionFusionContractError("backend output changed model id or output schema")
        object.__setattr__(self, "group_description", group)
        object.__setattr__(self, "semantic_paraphrase", paraphrase)

    def __repr__(self) -> str:
        return "CaptionFusionBackendOutput(<redacted>)"


class CaptionFusionBackend(Protocol):
    """Host-injected GPT backend; this public module performs no network I/O."""

    manifest: CaptionFusionBackendManifest

    def generate(self, request: CaptionFusionBackendRequest) -> CaptionFusionBackendOutput:
        """Return exactly one group description and one semantic paraphrase."""


@dataclass(frozen=True, slots=True)
class CaptionFusionReceipt:
    """The only serializable artifact: digest-only, authority-zero metadata."""

    input_sha256: str
    output_sha256: str
    provenance_sha256: str
    admission_sha256: str
    backend_manifest_sha256: str
    actor_count: int
    attempts_per_audit_run: tuple[int, ...]
    attempt_count: int
    retry_count: int
    schema: str = RECEIPT_SCHEMA
    annotation_kind: str = ANNOTATION_KIND
    task_scope: str = TASK_SCOPE
    model_id: str = MODEL_ID
    prompt_version: str = PROMPT_VERSION
    prompt_sha256: str = PROMPT_SHA256
    temperature_decimal: str = TEMPERATURE_DECIMAL
    top_p_decimal: str = TOP_P_DECIMAL
    generation_seed: int = GENERATION_SEED
    max_output_tokens: int = MAX_OUTPUT_TOKENS
    response_format: str = RESPONSE_FORMAT
    request_schema: str = REQUEST_SCHEMA
    output_schema: str = OUTPUT_SCHEMA
    retry_policy_version: str = RETRY_POLICY_VERSION
    retry_policy_sha256: str = RETRY_POLICY_SHA256
    retry_max_attempts: int = MAX_ATTEMPTS
    permutation_audit_version: str = PERMUTATION_AUDIT_VERSION
    permutation_audit_runs: int = PERMUTATION_AUDIT_RUNS
    same_family_review_status: str = SAME_FAMILY_REVIEW_STATUS
    machine_fused: bool = True
    human_annotation: bool = False
    public_verification_performed: bool = False
    authority: int = 0
    scientific_result_claimed: bool = False
    result_claimed: bool = False


@dataclass(frozen=True, slots=True, repr=False)
class MachineFusedCaption:
    """Private runtime text plus its public-safe digest receipt."""

    group_description: str = field(repr=False)
    semantic_paraphrase: str = field(repr=False)
    receipt: CaptionFusionReceipt

    def __post_init__(self) -> None:
        if type(self.receipt) is not CaptionFusionReceipt:
            raise CaptionFusionContractError("receipt must be exact CaptionFusionReceipt")
        canonical_caption_fusion_receipt_bytes(self.receipt)
        group = _normalized_text(
            self.group_description,
            "group_description",
            byte_limit=MAX_OUTPUT_TEXT_BYTES,
        )
        paraphrase = _normalized_text(
            self.semantic_paraphrase,
            "semantic_paraphrase",
            byte_limit=MAX_OUTPUT_TEXT_BYTES,
        )
        if group != self.group_description or paraphrase != self.semantic_paraphrase:
            raise CaptionFusionContractError("machine-fused caption text must be normalized")
        if _semantic_key(group) == _semantic_key(paraphrase):
            raise CaptionFusionContractError("machine-fused paraphrase must be distinct")
        if _output_digest(group, paraphrase) != self.receipt.output_sha256:
            raise CaptionFusionContractError(
                "machine-fused private text differs from its digest receipt"
            )

    def __repr__(self) -> str:
        return f"MachineFusedCaption(receipt={self.receipt!r}, text=<redacted>)"


def _provenance_digest(value: CaptionFusionProvenance) -> str:
    if type(value) is not CaptionFusionProvenance:
        raise CaptionFusionContractError("provenance must be exact CaptionFusionProvenance")
    return value.digest()


def _input_digest(value: CaptionFusionInput) -> str:
    digest = hashlib.sha256(b"phaseset-caption-fusion-input-digest-v1")
    _update_digest(digest, b"provenance", _provenance_digest(value.provenance).encode("ascii"))
    _update_digest(digest, b"window-commitment", value.window_commitment)
    canonical = sorted(value.actor_captions, key=lambda item: item.actor_commitment)
    for item in canonical:
        _update_digest(digest, b"actor-commitment", item.actor_commitment)
        _update_digest(digest, b"actor-text", item.text.encode("utf-8"))
        _update_digest(digest, b"actor-source", item.source_record_sha256.encode("ascii"))
    if value.holistic_context is None:
        _update_digest(digest, b"holistic", b"ABSENT")
    else:
        _update_digest(digest, b"holistic-text", value.holistic_context.text.encode("utf-8"))
        _update_digest(
            digest,
            b"holistic-source",
            value.holistic_context.source_record_sha256.encode("ascii"),
        )
    return digest.hexdigest()


def _output_digest(group_description: str, semantic_paraphrase: str) -> str:
    digest = hashlib.sha256(b"phaseset-caption-fusion-output-digest-v1")
    _update_digest(digest, b"group", _semantic_key(group_description).encode("utf-8"))
    _update_digest(digest, b"paraphrase", _semantic_key(semantic_paraphrase).encode("utf-8"))
    return digest.hexdigest()


def deterministic_actor_permutations(
    value: CaptionFusionInput,
) -> tuple[tuple[HumanActorCaption, ...], ...]:
    """Return three distinct, reproducible pseudo-random actor orders."""

    if type(value) is not CaptionFusionInput:
        raise CaptionFusionContractError("value must be exact CaptionFusionInput")
    canonical = tuple(sorted(value.actor_captions, key=lambda item: item.actor_commitment))
    orders: list[tuple[HumanActorCaption, ...]] = [canonical]
    seen = {tuple(item.actor_commitment for item in canonical)}
    counter = 0
    while len(orders) < PERMUTATION_AUDIT_RUNS:
        decorated = []
        for item in canonical:
            key = hashlib.sha256(
                b"phaseset-caption-fusion-audit-order-v1"
                + value.window_commitment
                + counter.to_bytes(8, "big")
                + item.actor_commitment
            ).digest()
            decorated.append((key, item.actor_commitment, item))
        candidate = tuple(item for _, _, item in sorted(decorated))
        identity = tuple(item.actor_commitment for item in candidate)
        if identity not in seen:
            orders.append(candidate)
            seen.add(identity)
        counter += 1
        if counter > 4096:
            raise CaptionFusionContractError("could not construct distinct audit permutations")
    return tuple(orders)


def _backend_request(
    value: CaptionFusionInput,
    manifest: CaptionFusionBackendManifest,
) -> CaptionFusionBackendRequest:
    return CaptionFusionBackendRequest(
        tuple(item.text for item in value.actor_captions),
        value.holistic_context.text if value.holistic_context is not None else None,
        backend_manifest_sha256=manifest.sha256,
        model_revision_sha256=manifest.model_revision_sha256,
    )


def _invoke_backend(
    backend: CaptionFusionBackend,
    request: CaptionFusionBackendRequest,
) -> tuple[CaptionFusionBackendOutput, int]:
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            output = backend.generate(request)
        except CaptionFusionTransientError:
            if attempt == MAX_ATTEMPTS:
                raise CaptionFusionContractError(
                    "caption backend exhausted the frozen retry policy"
                ) from None
            continue
        except Exception:
            raise CaptionFusionContractError(
                "caption backend failed with a nonretryable redacted error"
            ) from None
        if type(output) is not CaptionFusionBackendOutput:
            raise CaptionFusionContractError(
                "backend must return exact CaptionFusionBackendOutput"
            )
        return output, attempt
    raise AssertionError("unreachable retry state")


def _admission_sha256(
    value: CaptionFusionInput,
    test_gate: CaptionFusionTestGate | None,
) -> str:
    provenance_sha256 = _provenance_digest(value.provenance)
    if value.provenance.split != "test":
        if test_gate is not None:
            raise CaptionFusionContractError("test gate is forbidden outside the test split")
        return hashlib.sha256(
            b"phaseset-caption-fusion-local-admission-v1\x00"
            + provenance_sha256.encode("ascii")
        ).hexdigest()
    if test_gate is None or not callable(getattr(test_gate, "verify_and_consume", None)):
        raise CaptionFusionContractError(
            "test fusion requires a trusted host verify-and-consume gate"
        )
    try:
        admission = test_gate.verify_and_consume(value.provenance)
    except Exception:
        raise CaptionFusionContractError("test fusion admission failed") from None
    if type(admission) is not CaptionFusionTestAdmission:
        raise CaptionFusionContractError("test gate returned the wrong admission type")
    if (
        admission.provenance_sha256 != provenance_sha256
        or admission.sealed_test_manifest_sha256
        != value.provenance.sealed_test_manifest_sha256
        or admission.sealed_test_caption_sha256
        != value.provenance.sealed_test_caption_sha256
    ):
        raise CaptionFusionContractError("test admission does not bind exact provenance")
    return admission.sha256


def fuse_group_caption(
    value: CaptionFusionInput,
    backend: CaptionFusionBackend,
    *,
    test_gate: CaptionFusionTestGate | None = None,
) -> MachineFusedCaption:
    """Run the frozen permutation audit through one host-injected backend.

    Three distinct pseudo-random physical orders are sent to the backend.
    Normalized output must be identical across all runs. This strict stability
    check is deterministic evidence only; semantic review remains provisional
    because generation and review share one model family.
    """

    if type(value) is not CaptionFusionInput:
        raise CaptionFusionContractError("value must be exact CaptionFusionInput")
    manifest = getattr(backend, "manifest", None)
    if type(manifest) is not CaptionFusionBackendManifest:
        raise CaptionFusionContractError("backend requires an exact frozen manifest")
    if not callable(getattr(backend, "generate", None)):
        raise CaptionFusionContractError("backend must implement generate(request)")
    admission_sha256 = _admission_sha256(value, test_gate)

    expected_semantics: tuple[str, str] | None = None
    selected: CaptionFusionBackendOutput | None = None
    attempts: list[int] = []
    for permutation in deterministic_actor_permutations(value):
        permuted = CaptionFusionInput(permutation, value.provenance, value.holistic_context)
        request = _backend_request(permuted, manifest)
        output, attempt_count = _invoke_backend(backend, request)
        semantics = (
            _semantic_key(output.group_description),
            _semantic_key(output.semantic_paraphrase),
        )
        if expected_semantics is None:
            expected_semantics = semantics
            selected = output
        elif semantics != expected_semantics:
            raise CaptionFusionContractError(
                "normalized output changed across actor-order audit runs"
            )
        attempts.append(attempt_count)

    if selected is None:
        raise AssertionError("permutation audit produced no output")
    total_attempts = sum(attempts)
    receipt = CaptionFusionReceipt(
        input_sha256=_input_digest(value),
        output_sha256=_output_digest(
            selected.group_description,
            selected.semantic_paraphrase,
        ),
        provenance_sha256=_provenance_digest(value.provenance),
        admission_sha256=admission_sha256,
        backend_manifest_sha256=manifest.sha256,
        actor_count=value.actor_count,
        attempts_per_audit_run=tuple(attempts),
        attempt_count=total_attempts,
        retry_count=total_attempts - PERMUTATION_AUDIT_RUNS,
    )
    canonical_caption_fusion_receipt_bytes(receipt)
    return MachineFusedCaption(
        selected.group_description,
        selected.semantic_paraphrase,
        receipt,
    )


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("ascii")
        + b"\n"
    )


def canonical_caption_fusion_receipt_bytes(record: object) -> bytes:
    """Validate and serialize only the public-safe digest receipt."""

    if type(record) is not CaptionFusionReceipt:
        raise CaptionFusionContractError("record must be exact CaptionFusionReceipt")
    for name in (
        "input_sha256",
        "output_sha256",
        "provenance_sha256",
        "admission_sha256",
        "backend_manifest_sha256",
        "prompt_sha256",
        "retry_policy_sha256",
    ):
        _lower_sha256(getattr(record, name), name)
    if type(record.actor_count) is not int or record.actor_count < 3:
        raise CaptionFusionContractError("receipt actor_count must be an exact int >= 3")
    if (
        type(record.attempts_per_audit_run) is not tuple
        or len(record.attempts_per_audit_run) != PERMUTATION_AUDIT_RUNS
        or any(type(item) is not int or not 1 <= item <= MAX_ATTEMPTS for item in record.attempts_per_audit_run)
    ):
        raise CaptionFusionContractError("receipt attempts_per_audit_run is invalid")
    if (
        type(record.attempt_count) is not int
        or record.attempt_count != sum(record.attempts_per_audit_run)
        or type(record.retry_count) is not int
        or record.retry_count != record.attempt_count - PERMUTATION_AUDIT_RUNS
    ):
        raise CaptionFusionContractError("receipt attempt/retry census is inconsistent")
    frozen = {
        "schema": RECEIPT_SCHEMA,
        "annotation_kind": ANNOTATION_KIND,
        "task_scope": TASK_SCOPE,
        "model_id": MODEL_ID,
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": PROMPT_SHA256,
        "temperature_decimal": TEMPERATURE_DECIMAL,
        "top_p_decimal": TOP_P_DECIMAL,
        "generation_seed": GENERATION_SEED,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "response_format": RESPONSE_FORMAT,
        "request_schema": REQUEST_SCHEMA,
        "output_schema": OUTPUT_SCHEMA,
        "retry_policy_version": RETRY_POLICY_VERSION,
        "retry_policy_sha256": RETRY_POLICY_SHA256,
        "retry_max_attempts": MAX_ATTEMPTS,
        "permutation_audit_version": PERMUTATION_AUDIT_VERSION,
        "permutation_audit_runs": PERMUTATION_AUDIT_RUNS,
        "same_family_review_status": SAME_FAMILY_REVIEW_STATUS,
        "machine_fused": True,
        "human_annotation": False,
        "public_verification_performed": False,
        "authority": 0,
        "scientific_result_claimed": False,
        "result_claimed": False,
    }
    for name, expected in frozen.items():
        if getattr(record, name) != expected:
            raise CaptionFusionContractError(f"receipt frozen field {name} changed")

    return _canonical_json_bytes(
        {
            **frozen,
            "actor_count": record.actor_count,
            "attempt_count": record.attempt_count,
            "attempts_per_audit_run": list(record.attempts_per_audit_run),
            "admission_sha256": record.admission_sha256,
            "backend_manifest_sha256": record.backend_manifest_sha256,
            "input_sha256": record.input_sha256,
            "output_sha256": record.output_sha256,
            "provenance_sha256": record.provenance_sha256,
            "retry_count": record.retry_count,
        }
    )


__all__ = [
    "ANNOTATION_KIND",
    "CaptionFusionBackend",
    "CaptionFusionBackendManifest",
    "CaptionFusionBackendOutput",
    "CaptionFusionBackendRequest",
    "CaptionFusionContractError",
    "CaptionFusionInput",
    "CaptionFusionProvenance",
    "CaptionFusionReceipt",
    "CaptionFusionTestAdmission",
    "CaptionFusionTestGate",
    "CaptionFusionTransientError",
    "FROZEN_PROMPT",
    "HumanActorCaption",
    "HumanHolisticContext",
    "MAX_ATTEMPTS",
    "MODEL_ID",
    "MachineFusedCaption",
    "PERMUTATION_AUDIT_RUNS",
    "PROMPT_SHA256",
    "SAME_FAMILY_REVIEW_STATUS",
    "TASK_SCOPE",
    "canonical_caption_fusion_receipt_bytes",
    "deterministic_actor_permutations",
    "fuse_group_caption",
]
