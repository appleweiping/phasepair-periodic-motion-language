from __future__ import annotations

from dataclasses import fields, replace
import hashlib
import json

import pytest

from phaseset_core import caption_fusion


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _raw(label: str) -> bytes:
    return hashlib.sha256(label.encode("utf-8")).digest()


def _provenance(split: str = "train") -> caption_fusion.CaptionFusionProvenance:
    sealed_manifest = _sha("sealed-test-manifest") if split == "test" else None
    sealed_caption = _sha("sealed-test-caption") if split == "test" else None
    return caption_fusion.CaptionFusionProvenance(
        split=split,
        dataset_manifest_sha256=_sha("private-dataset-manifest"),
        caption_manifest_sha256=_sha("private-caption-manifest"),
        human_text_rights_assertion_sha256=_sha("private-rights-assertion"),
        sealed_test_manifest_sha256=sealed_manifest,
        sealed_test_caption_sha256=sealed_caption,
    )


def _input(
    *,
    split: str = "train",
    order: tuple[int, ...] = (2, 0, 1),
    holistic: bool = True,
) -> caption_fusion.CaptionFusionInput:
    window = _raw("same-private-window")
    texts = (
        "Actor one moves near an opaque marker.",
        "Actor two turns toward the group.",
        "Actor three raises an arm.",
    )
    actors = tuple(
        caption_fusion.HumanActorCaption(
            texts[index],
            bytes([index + 1]) * 32,
            window,
            _sha(f"official-human-actor-record-{index}"),
        )
        for index in order
    )
    context = (
        caption_fusion.HumanHolisticContext(
            "The group performs a coordinated exchange around a shared prop.",
            window,
            _sha("official-human-holistic-record"),
        )
        if holistic
        else None
    )
    return caption_fusion.CaptionFusionInput(actors, _provenance(split), context)


class _StableBackend:
    manifest = caption_fusion.CaptionFusionBackendManifest(
        model_revision_sha256=_sha("frozen-gpt-5.6-sol-revision"),
        backend_implementation_sha256=_sha("synthetic-host-injected-backend"),
        inference_runtime_sha256=_sha("synthetic-inference-runtime"),
    )

    def __init__(self) -> None:
        self.requests: list[caption_fusion.CaptionFusionBackendRequest] = []

    def generate(
        self,
        request: caption_fusion.CaptionFusionBackendRequest,
    ) -> caption_fusion.CaptionFusionBackendOutput:
        self.requests.append(request)
        return caption_fusion.CaptionFusionBackendOutput(
            "Three people coordinate a turn and an arm gesture.",
            "A trio performs a coordinated turn while one person lifts an arm.",
        )


class _TestGate:
    def __init__(self) -> None:
        self.consumed = False

    def verify_and_consume(
        self,
        provenance: caption_fusion.CaptionFusionProvenance,
    ) -> caption_fusion.CaptionFusionTestAdmission:
        if self.consumed:
            raise RuntimeError("already consumed")
        self.consumed = True
        assert provenance.sealed_test_manifest_sha256 is not None
        assert provenance.sealed_test_caption_sha256 is not None
        return caption_fusion.CaptionFusionTestAdmission(
            provenance_sha256=provenance.digest(),
            sealed_test_manifest_sha256=provenance.sealed_test_manifest_sha256,
            sealed_test_caption_sha256=provenance.sealed_test_caption_sha256,
            external_grant_sha256=_sha("external-test-fusion-grant"),
            consumption_sha256=_sha("single-test-fusion-consumption"),
        )


def test_frozen_model_prompt_and_text_only_backend_surface() -> None:
    assert caption_fusion.MODEL_ID == "gpt-5.6-sol"
    assert hashlib.sha256(caption_fusion.FROZEN_PROMPT.encode("utf-8")).hexdigest() == (
        caption_fusion.PROMPT_SHA256
    )
    prompt = " ".join(caption_fusion.FROZEN_PROMPT.casefold().split())
    for forbidden in (
        "skeleton",
        "spectra",
        "frequency bands",
        "phase",
        "coherence",
        "dataset-split statistics",
    ):
        assert forbidden in prompt
    assert {item.name for item in fields(caption_fusion.CaptionFusionBackendRequest)} == {
        "actor_descriptions",
        "holistic_context",
        "model_id",
        "prompt",
        "prompt_version",
        "prompt_sha256",
        "backend_manifest_sha256",
        "model_revision_sha256",
        "temperature_decimal",
        "top_p_decimal",
        "generation_seed",
        "max_output_tokens",
        "response_format",
        "schema",
    }
    assert not {
        "skeletons",
        "spectrum",
        "phase",
        "coherence",
        "split",
        "statistics",
        "actor_commitments",
        "path",
    } & {item.name for item in fields(caption_fusion.CaptionFusionBackendRequest)}


def test_canonicalizes_actor_text_and_audits_three_distinct_orders() -> None:
    value = _input()
    orders = caption_fusion.deterministic_actor_permutations(value)
    identities = tuple(
        tuple(item.actor_commitment for item in permutation) for permutation in orders
    )
    assert len(orders) == caption_fusion.PERMUTATION_AUDIT_RUNS == 3
    assert len(set(identities)) == 3

    backend = _StableBackend()
    result = caption_fusion.fuse_group_caption(value, backend)
    assert len(backend.requests) == 3
    expected = tuple(
        tuple(item.text for item in permutation)
        for permutation in orders
    )
    observed = tuple(request.actor_descriptions for request in backend.requests)
    assert observed == expected
    assert len(set(observed)) == 3
    assert all(request.model_id == "gpt-5.6-sol" for request in backend.requests)
    assert all(
        request.backend_manifest_sha256 == backend.manifest.sha256
        and request.model_revision_sha256 == backend.manifest.model_revision_sha256
        and request.temperature_decimal == "0"
        and request.top_p_decimal == "1"
        and request.generation_seed == 1729
        and request.max_output_tokens == 512
        for request in backend.requests
    )
    assert result.receipt.attempts_per_audit_run == (1, 1, 1)
    assert result.receipt.attempt_count == 3
    assert result.receipt.retry_count == 0

    reversed_backend = _StableBackend()
    reversed_value = _input(order=(1, 0, 2))
    reversed_result = caption_fusion.fuse_group_caption(reversed_value, reversed_backend)
    assert tuple(request.actor_descriptions for request in reversed_backend.requests) == expected
    assert reversed_result.receipt.input_sha256 == result.receipt.input_sha256
    assert reversed_result.receipt.output_sha256 == result.receipt.output_sha256


def test_input_digest_binds_private_window_lineage_without_serializing_it() -> None:
    first = _input()
    new_window = _raw("different-private-window")
    actors = tuple(
        replace(item, window_commitment=new_window) for item in first.actor_captions
    )
    holistic = replace(first.holistic_context, window_commitment=new_window)
    second = caption_fusion.CaptionFusionInput(actors, first.provenance, holistic)
    first_result = caption_fusion.fuse_group_caption(first, _StableBackend())
    second_result = caption_fusion.fuse_group_caption(second, _StableBackend())
    assert first_result.receipt.input_sha256 != second_result.receipt.input_sha256
    raw = caption_fusion.canonical_caption_fusion_receipt_bytes(second_result.receipt)
    assert new_window not in raw


def test_only_receipt_serializes_and_it_is_redacted_machine_fused_authority_zero() -> None:
    value = _input(split="test")
    result = caption_fusion.fuse_group_caption(
        value,
        _StableBackend(),
        test_gate=_TestGate(),
    )
    raw = caption_fusion.canonical_caption_fusion_receipt_bytes(result.receipt)
    decoded = json.loads(raw)

    assert decoded["annotation_kind"] == "MACHINE_FUSED_GROUP_CAPTION"
    assert decoded["task_scope"] == "AUXILIARY_10S_GROUP_CAPTION_RETRIEVAL_ONLY"
    assert decoded["model_id"] == "gpt-5.6-sol"
    assert decoded["machine_fused"] is True
    assert decoded["human_annotation"] is False
    assert decoded["authority"] == 0
    assert decoded["scientific_result_claimed"] is False
    assert decoded["result_claimed"] is False
    assert decoded["public_verification_performed"] is False
    assert decoded["same_family_review_status"] == (
        "PROVISIONAL_SAME_MODEL_FAMILY_REVIEW_NOT_GATE_CLOSING"
    )
    assert decoded["temperature_decimal"] == "0"
    assert decoded["top_p_decimal"] == "1"
    assert decoded["generation_seed"] == 1729
    assert "input_split" not in decoded
    assert "test_provenance_asserted" not in decoded

    secret_fragments = (
        value.actor_captions[0].text.encode(),
        value.holistic_context.text.encode(),
        result.group_description.encode(),
        result.semantic_paraphrase.encode(),
        bytes([1]) * 32,
        bytes([2]) * 32,
        bytes([3]) * 32,
        value.window_commitment,
    )
    assert all(fragment not in raw for fragment in secret_fragments)
    assert "private" not in repr(value).casefold()
    assert "coordinate" not in repr(result).casefold()


@pytest.mark.parametrize(
    ("split", "sealed_manifest", "sealed_caption", "message"),
    (
        ("test", None, None, "test fusion requires"),
        ("test", _sha("sealed"), None, "test fusion requires"),
        ("train", _sha("sealed"), _sha("caption"), "forbidden outside"),
        ("val", None, _sha("caption"), "forbidden outside"),
    ),
)
def test_test_caption_provenance_fails_closed(
    split: str,
    sealed_manifest: str | None,
    sealed_caption: str | None,
    message: str,
) -> None:
    with pytest.raises(caption_fusion.CaptionFusionContractError, match=message):
        caption_fusion.CaptionFusionProvenance(
            split,
            _sha("dataset"),
            _sha("captions"),
            _sha("rights"),
            sealed_manifest,
            sealed_caption,
        )


def test_test_fusion_requires_matching_single_consumption_host_gate() -> None:
    value = _input(split="test")
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="verify-and-consume"):
        caption_fusion.fuse_group_caption(value, _StableBackend())
    gate = _TestGate()
    caption_fusion.fuse_group_caption(value, _StableBackend(), test_gate=gate)
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="admission failed"):
        caption_fusion.fuse_group_caption(value, _StableBackend(), test_gate=gate)

    forged = replace(
        value.provenance,
        sealed_test_manifest_sha256="0" * 64,
        sealed_test_caption_sha256="1" * 64,
    )

    class _MismatchedGate(_TestGate):
        def verify_and_consume(
            self,
            provenance: caption_fusion.CaptionFusionProvenance,
        ) -> caption_fusion.CaptionFusionTestAdmission:
            admission = super().verify_and_consume(provenance)
            return replace(admission, sealed_test_caption_sha256="2" * 64)

    forged_input = caption_fusion.CaptionFusionInput(
        value.actor_captions,
        forged,
        value.holistic_context,
    )
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="exact provenance"):
        caption_fusion.fuse_group_caption(
            forged_input,
            _StableBackend(),
            test_gate=_MismatchedGate(),
        )


def test_requires_three_distinct_human_captions_from_exactly_one_window() -> None:
    value = _input()
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="at least three"):
        caption_fusion.CaptionFusionInput(value.actor_captions[:2], value.provenance)

    first, second, third = value.actor_captions
    mismatch = caption_fusion.HumanActorCaption(
        second.text,
        second.actor_commitment,
        _raw("different-window"),
        second.source_record_sha256,
    )
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="same window"):
        caption_fusion.CaptionFusionInput((first, mismatch, third), value.provenance)

    duplicate = caption_fusion.HumanActorCaption(
        second.text,
        first.actor_commitment,
        first.window_commitment,
        second.source_record_sha256,
    )
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="distinct"):
        caption_fusion.CaptionFusionInput((first, duplicate, third), value.provenance)

    wrong_holistic = caption_fusion.HumanHolisticContext(
        "A holistic sentence.",
        _raw("different-window"),
        _sha("holistic"),
    )
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="same window"):
        caption_fusion.CaptionFusionInput(
            value.actor_captions,
            value.provenance,
            wrong_holistic,
        )

    with pytest.raises(caption_fusion.CaptionFusionContractError, match="source must"):
        caption_fusion.HumanActorCaption(
            "Machine-authored text.",
            _raw("actor"),
            _raw("window"),
            _sha("source"),
            source_kind="MACHINE_GENERATED",
        )


def test_backend_model_output_type_and_permutation_stability_fail_closed() -> None:
    value = _input(holistic=False)

    class _WrongModel(_StableBackend):
        manifest = None

    wrong_model = _WrongModel()
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="frozen manifest"):
        caption_fusion.fuse_group_caption(value, wrong_model)
    assert not wrong_model.requests

    class _WrongType(_StableBackend):
        def generate(self, request: caption_fusion.CaptionFusionBackendRequest) -> object:
            self.requests.append(request)
            return {"group_description": "not a closed output"}

    with pytest.raises(caption_fusion.CaptionFusionContractError, match="exact"):
        caption_fusion.fuse_group_caption(value, _WrongType())

    class _Unstable(_StableBackend):
        def generate(
            self,
            request: caption_fusion.CaptionFusionBackendRequest,
        ) -> caption_fusion.CaptionFusionBackendOutput:
            self.requests.append(request)
            count = len(self.requests)
            return caption_fusion.CaptionFusionBackendOutput(
                f"The group performs version {count}.",
                f"Version {count} describes the group activity.",
            )

    with pytest.raises(caption_fusion.CaptionFusionContractError, match="actor-order"):
        caption_fusion.fuse_group_caption(value, _Unstable())

    class _OrderDeictic(_StableBackend):
        def generate(
            self,
            request: caption_fusion.CaptionFusionBackendRequest,
        ) -> caption_fusion.CaptionFusionBackendOutput:
            return caption_fusion.CaptionFusionBackendOutput(
                "The first described person turns toward the second person.",
                "Person one faces person two during the group action.",
            )

    with pytest.raises(caption_fusion.CaptionFusionContractError, match="redacted"):
        caption_fusion.fuse_group_caption(value, _OrderDeictic())
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="order-dependent"):
        caption_fusion.CaptionFusionBackendOutput(
            "The first described person turns toward the second person.",
            "Person one faces person two during the group action.",
        )


def test_frozen_transient_retry_policy_records_attempts_and_exhaustion() -> None:
    class _RetryOnce(_StableBackend):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def generate(
            self,
            request: caption_fusion.CaptionFusionBackendRequest,
        ) -> caption_fusion.CaptionFusionBackendOutput:
            self.calls += 1
            if self.calls == 1:
                raise caption_fusion.CaptionFusionTransientError("synthetic transient")
            return super().generate(request)

    result = caption_fusion.fuse_group_caption(_input(), _RetryOnce())
    assert result.receipt.attempts_per_audit_run == (2, 1, 1)
    assert result.receipt.attempt_count == 4
    assert result.receipt.retry_count == 1
    assert result.receipt.retry_max_attempts == 3

    class _AlwaysRetry(_StableBackend):
        def generate(
            self,
            request: caption_fusion.CaptionFusionBackendRequest,
        ) -> caption_fusion.CaptionFusionBackendOutput:
            raise caption_fusion.CaptionFusionTransientError("still unavailable")

    with pytest.raises(caption_fusion.CaptionFusionContractError, match="exhausted"):
        caption_fusion.fuse_group_caption(_input(), _AlwaysRetry())

    private_text = _input().actor_captions[0].text

    class _LeakingFailure(_StableBackend):
        def generate(
            self,
            request: caption_fusion.CaptionFusionBackendRequest,
        ) -> caption_fusion.CaptionFusionBackendOutput:
            raise RuntimeError(f"LEAK {request.actor_descriptions[0]}")

    with pytest.raises(caption_fusion.CaptionFusionContractError) as captured:
        caption_fusion.fuse_group_caption(_input(), _LeakingFailure())
    assert captured.value.__cause__ is None
    assert private_text not in str(captured.value)


def test_receipt_serializer_rejects_forged_claims_and_inconsistent_census() -> None:
    receipt = caption_fusion.fuse_group_caption(_input(), _StableBackend()).receipt
    for forged in (
        replace(receipt, human_annotation=True),
        replace(receipt, authority=1),
        replace(receipt, scientific_result_claimed=True),
        replace(receipt, same_family_review_status="APPROVED"),
        replace(receipt, attempts_per_audit_run=(1, 1, 2), attempt_count=3),
        replace(receipt, admission_sha256="0" * 63),
    ):
        with pytest.raises(caption_fusion.CaptionFusionContractError):
            caption_fusion.canonical_caption_fusion_receipt_bytes(forged)

    result = caption_fusion.fuse_group_caption(_input(), _StableBackend())
    with pytest.raises(caption_fusion.CaptionFusionContractError, match="private text"):
        caption_fusion.MachineFusedCaption(
            "A different generated group description.",
            result.semantic_paraphrase,
            result.receipt,
        )
