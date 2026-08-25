"""Synthetic-only tests for the full PhasePair base sampler manifest."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from phasepair_core import (
    batching,
    caption_processing,
    sampler_manifest,
    training_batch,
)


def _raw32(value: int) -> bytes:
    return value.to_bytes(32, "big")


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _manifest_inputs(count: int = 4) -> dict[str, object]:
    pairs = tuple(_raw32(index + 1) for index in range(count))
    caption_commitments = tuple(
        tuple(_raw32(1000 + 3 * row + ordinal) for ordinal in range(3))
        for row in range(count)
    )
    anchors = np.eye(count, dtype=np.float64)
    return {
        "pair_commitments": pairs,
        "caption_commitments": caption_commitments,
        "caption_mean_anchors": anchors,
        "seed": 1729,
    }


def _manifest(count: int = 4) -> sampler_manifest.BaseSamplerManifest:
    return sampler_manifest.build_base_sampler_manifest(**_manifest_inputs(count))


def _prepared_for(
    planned: sampler_manifest.PlannedTrainingBatch,
    *,
    caption_commitments: tuple[tuple[bytes, bytes, bytes], ...] | None = None,
    caption_ordinals: tuple[tuple[int, int, int], ...] | None = None,
    pairs: tuple[bytes, ...] | None = None,
    sampler_manifest_sha256: str | None = None,
    sampler_seed: int | None = None,
    epoch_index: int | None = None,
    batch_index: int | None = None,
    global_step: int | None = None,
) -> training_batch.PreparedTrainingBatch:
    planned_pairs = planned.pair_commitments if pairs is None else pairs
    batch_size = len(planned_pairs)
    motion = batching.PreparedMotionBatch(
        actor_a=np.zeros((batch_size, 1, 262), dtype=np.float32),
        actor_b=np.zeros((batch_size, 1, 262), dtype=np.float32),
        relation_ab=np.zeros((batch_size, 1, 799), dtype=np.float32),
        relation_ba=np.zeros((batch_size, 1, 799), dtype=np.float32),
        valid_mask=np.ones((batch_size, 1), dtype=np.uint8),
        valid_lengths=(1,) * batch_size,
        pair_commitments=planned_pairs,
    )
    ordered_captions = (
        planned.ordered_caption_commitments
        if caption_commitments is None
        else caption_commitments
    )
    ordered_ordinals = (
        planned.caption_ordinal_orders
        if caption_ordinals is None
        else caption_ordinals
    )
    payloads = tuple(
        tuple(f"caption-{row}-{ordinal}".encode() for ordinal in range(3))
        for row in range(batch_size)
    )
    ids = np.full((3 * batch_size, 77), 1, dtype=np.int64)
    mask = np.zeros((3 * batch_size, 77), dtype=np.int64)
    for row in range(3 * batch_size):
        ids[row, :3] = [0, 10 + row, 2]
        mask[row, :3] = 1
    caption_batch = caption_processing.PreparedCaptionBatch(
        pair_commitments=planned_pairs,
        caption_commitments=ordered_captions,
        caption_ordinals=ordered_ordinals,
        caption_payloads=payloads,
        input_ids=ids,
        attention_mask=mask,
        tokenizer_manifest_sha256=_digest("tokenizer"),
        caption_text_manifest_sha256=_digest("caption-text"),
        caption_lineage_manifest_sha256=_digest("caption-lineage"),
        normalizer_manifest_sha256=_digest("normalizer"),
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        vocab_size=100,
    )
    return training_batch.PreparedTrainingBatch(
        motion_batch=motion,
        caption_batch=caption_batch,
        sampler_manifest_sha256=(
            planned.manifest_sha256
            if sampler_manifest_sha256 is None
            else sampler_manifest_sha256
        ),
        sampler_seed=planned.seed if sampler_seed is None else sampler_seed,
        epoch_index=planned.epoch_index if epoch_index is None else epoch_index,
        batch_index=planned.batch_index if batch_index is None else batch_index,
        global_step=planned.global_step if global_step is None else global_step,
    )


def test_manifest_freezes_all_30_epochs_and_every_caption_permutation() -> None:
    manifest = _manifest()
    private = json.loads(sampler_manifest.base_sampler_manifest_private_bytes(manifest))
    public_raw = sampler_manifest.base_sampler_manifest_receipt_bytes(manifest)
    public = json.loads(public_raw)

    assert private["schema"] == "phasepair-base-sampler-manifest-v1"
    assert private["epoch_count"] == 30
    assert [row["epoch_index"] for row in private["epochs"]] == list(range(30))
    assert all(len(row["batches"]) == 1 for row in private["epochs"])
    assert private["total_optimizer_steps"] == 30
    for epoch in private["epochs"]:
        assert sorted(epoch["batches"][0]["pair_row_indices"]) == [0, 1, 2, 3]
        assert all(
            sorted(order) == [0, 1, 2]
            for order in epoch["batches"][0]["caption_ordinal_orders"]
        )
    assert public["schema"] == "phasepair-base-sampler-manifest-receipt-v1"
    assert public["authority"] == 0
    assert public["production"] is False
    assert public["execution_authorized"] is False
    assert public["result_claimed"] is False
    assert public["status"] == sampler_manifest.STATUS
    assert public["epoch_count"] == 30
    assert public["pair_count"] == 4
    assert public["manifest_sha256"] == sampler_manifest.base_sampler_manifest_sha256(
        manifest
    )
    assert public_raw.endswith(b"\n") and b"\r" not in public_raw


def test_planned_batch_exposes_exact_pair_and_permuted_caption_commitments() -> None:
    inputs = _manifest_inputs()
    manifest = sampler_manifest.build_base_sampler_manifest(**inputs)
    planned = sampler_manifest.planned_training_batch(
        manifest, epoch_index=7, batch_index=0
    )
    assert planned.seed == 1729
    assert planned.manifest_sha256 == sampler_manifest.base_sampler_manifest_sha256(
        manifest
    )
    assert planned.epoch_index == 7
    assert planned.batch_index == 0
    assert planned.global_step == 7
    assert planned.pair_commitments == tuple(
        inputs["pair_commitments"][index] for index in planned.pair_row_indices
    )
    assert planned.ordered_caption_commitments == tuple(
        tuple(inputs["caption_commitments"][index][ordinal] for ordinal in order)
        for index, order in zip(
            planned.pair_row_indices,
            planned.caption_ordinal_orders,
            strict=True,
        )
    )


def test_prepared_training_batch_joins_one_exact_sampler_cursor() -> None:
    manifest = _manifest()
    planned = sampler_manifest.planned_training_batch(
        manifest, epoch_index=7, batch_index=0
    )
    prepared = _prepared_for(planned)
    joined = sampler_manifest.join_prepared_batch_to_sampler(
        manifest,
        prepared,
        epoch_index=7,
        batch_index=0,
    )
    raw = sampler_manifest.sampler_batch_join_bytes(joined)
    payload = json.loads(raw)
    assert payload["schema"] == "phasepair-sampler-prepared-batch-join-v1"
    assert payload["seed"] == 1729
    assert payload["epoch_index"] == 7
    assert payload["batch_index"] == 0
    assert payload["manifest_sha256"] == sampler_manifest.base_sampler_manifest_sha256(
        manifest
    )
    assert payload["prepared_training_batch_sha256"] == (
        training_batch.prepared_training_batch_sha256(prepared)
    )
    prepared_payload = json.loads(
        training_batch.prepared_training_batch_bytes(prepared)
    )
    assert prepared_payload["sampler_manifest_sha256"] == planned.manifest_sha256
    assert prepared_payload["sampler_seed"] == planned.seed
    assert prepared_payload["epoch_index"] == planned.epoch_index
    assert prepared_payload["batch_index"] == planned.batch_index
    assert prepared_payload["global_step"] == planned.global_step
    assert payload["authority"] == 0
    assert payload["result_claimed"] is False
    assert sampler_manifest.sampler_batch_join_sha256(joined) == hashlib.sha256(
        raw
    ).hexdigest()


def test_join_rejects_pair_or_caption_order_drift() -> None:
    manifest = _manifest()
    planned = sampler_manifest.planned_training_batch(
        manifest, epoch_index=7, batch_index=0
    )
    reversed_pairs = tuple(reversed(planned.pair_commitments))
    prepared = _prepared_for(planned, pairs=reversed_pairs)
    with pytest.raises(sampler_manifest.SamplerManifestError, match="motion pair order"):
        sampler_manifest.join_prepared_batch_to_sampler(
            manifest, prepared, epoch_index=7, batch_index=0
        )

    wrong_caption_order = (
        tuple(reversed(planned.ordered_caption_commitments[0])),
    ) + planned.ordered_caption_commitments[1:]
    prepared = _prepared_for(planned, caption_commitments=wrong_caption_order)
    with pytest.raises(
        sampler_manifest.SamplerManifestError, match="caption commitment order"
    ):
        sampler_manifest.join_prepared_batch_to_sampler(
            manifest, prepared, epoch_index=7, batch_index=0
        )

    wrong_caption_ordinals = (
        tuple(reversed(planned.caption_ordinal_orders[0])),
    ) + planned.caption_ordinal_orders[1:]
    prepared = _prepared_for(planned, caption_ordinals=wrong_caption_ordinals)
    with pytest.raises(
        sampler_manifest.SamplerManifestError, match="caption ordinals"
    ):
        sampler_manifest.join_prepared_batch_to_sampler(
            manifest, prepared, epoch_index=7, batch_index=0
        )


@pytest.mark.parametrize(
    ("override", "message"),
    [
        (
            {"sampler_manifest_sha256": _digest("other-sampler-manifest")},
            "manifest identity differs",
        ),
        ({"sampler_seed": 2718}, "seed differs"),
        ({"epoch_index": 8}, "epoch differs"),
        ({"batch_index": 1}, "batch index differs"),
        ({"global_step": 8}, "global step differs"),
    ],
)
def test_join_rejects_each_prepared_batch_provenance_drift(
    override: dict[str, object], message: str
) -> None:
    manifest = _manifest()
    planned = sampler_manifest.planned_training_batch(
        manifest, epoch_index=7, batch_index=0
    )
    prepared = _prepared_for(planned, **override)
    with pytest.raises(sampler_manifest.SamplerManifestError, match=message):
        sampler_manifest.join_prepared_batch_to_sampler(
            manifest, prepared, epoch_index=7, batch_index=0
        )


def test_repeated_plan_content_cannot_relabel_a_batch_to_another_epoch() -> None:
    manifest = _manifest(count=1)
    original = sampler_manifest.planned_training_batch(
        manifest, epoch_index=0, batch_index=0
    )
    repeated = next(
        planned
        for epoch in range(1, sampler_manifest.EPOCH_COUNT)
        if (
            planned := sampler_manifest.planned_training_batch(
                manifest, epoch_index=epoch, batch_index=0
            )
        ).ordered_caption_commitments
        == original.ordered_caption_commitments
    )
    prepared = _prepared_for(original)

    with pytest.raises(sampler_manifest.SamplerManifestError, match="epoch differs"):
        sampler_manifest.join_prepared_batch_to_sampler(
            manifest,
            prepared,
            epoch_index=repeated.epoch_index,
            batch_index=0,
        )


def test_repeated_plan_content_cannot_relabel_a_batch_to_another_seed() -> None:
    first_inputs = _manifest_inputs(count=1)
    first = sampler_manifest.build_base_sampler_manifest(**first_inputs)
    second_inputs = _manifest_inputs(count=1)
    second_inputs["seed"] = 2718
    second = sampler_manifest.build_base_sampler_manifest(**second_inputs)
    collision = next(
        (first_plan, second_plan)
        for epoch in range(sampler_manifest.EPOCH_COUNT)
        if (
            first_plan := sampler_manifest.planned_training_batch(
                first, epoch_index=epoch, batch_index=0
            )
        ).ordered_caption_commitments
        == (
            second_plan := sampler_manifest.planned_training_batch(
                second, epoch_index=epoch, batch_index=0
            )
        ).ordered_caption_commitments
    )
    first_plan, second_plan = collision
    prepared = _prepared_for(first_plan)

    with pytest.raises(
        sampler_manifest.SamplerManifestError, match="manifest identity differs"
    ):
        sampler_manifest.join_prepared_batch_to_sampler(
            second,
            prepared,
            epoch_index=second_plan.epoch_index,
            batch_index=0,
        )

    wrong_seed = _prepared_for(second_plan, sampler_seed=first_plan.seed)
    with pytest.raises(sampler_manifest.SamplerManifestError, match="seed differs"):
        sampler_manifest.join_prepared_batch_to_sampler(
            second,
            wrong_seed,
            epoch_index=second_plan.epoch_index,
            batch_index=0,
        )


def test_join_rejects_object_new_forged_caption_and_training_batches() -> None:
    manifest = _manifest(count=1)
    planned = sampler_manifest.planned_training_batch(
        manifest, epoch_index=0, batch_index=0
    )
    legitimate = _prepared_for(planned)

    forged_caption = object.__new__(caption_processing.PreparedCaptionBatch)
    object.__setattr__(
        forged_caption, "pair_commitments", planned.pair_commitments
    )
    object.__setattr__(
        forged_caption,
        "caption_commitments",
        planned.ordered_caption_commitments,
    )
    object.__setattr__(
        forged_caption, "caption_ordinals", planned.caption_ordinal_orders
    )
    object.__setattr__(
        forged_caption,
        "input_ids",
        np.full((3, 77), np.nan, dtype=np.float32),
    )
    object.__setattr__(
        forged_caption, "canonical_receipt", b"FORGED_CAPTION_BATCH\n"
    )
    with pytest.raises(
        caption_processing.CaptionProcessingError, match="not issued"
    ):
        training_batch.PreparedTrainingBatch(
            motion_batch=legitimate.motion_batch,
            caption_batch=forged_caption,
            sampler_manifest_sha256=planned.manifest_sha256,
            sampler_seed=planned.seed,
            epoch_index=planned.epoch_index,
            batch_index=planned.batch_index,
            global_step=planned.global_step,
        )

    forged_training = object.__new__(training_batch.PreparedTrainingBatch)
    object.__setattr__(
        forged_training, "motion_batch", legitimate.motion_batch
    )
    object.__setattr__(
        forged_training, "caption_batch", legitimate.caption_batch
    )
    object.__setattr__(
        forged_training,
        "positive_mask",
        np.zeros_like(legitimate.positive_mask),
    )
    object.__setattr__(
        forged_training, "canonical_receipt", b"FORGED_PREPARED_BATCH\n"
    )
    object.__setattr__(
        forged_training, "sampler_manifest_sha256", planned.manifest_sha256
    )
    object.__setattr__(forged_training, "sampler_seed", planned.seed)
    object.__setattr__(forged_training, "epoch_index", planned.epoch_index)
    object.__setattr__(forged_training, "batch_index", planned.batch_index)
    object.__setattr__(forged_training, "global_step", planned.global_step)
    with pytest.raises(training_batch.TrainingBatchError, match="not issued"):
        sampler_manifest.join_prepared_batch_to_sampler(
            manifest,
            forged_training,
            epoch_index=planned.epoch_index,
            batch_index=planned.batch_index,
        )


def test_manifest_is_deterministic_snapshots_anchors_and_changes_with_seed() -> None:
    inputs = _manifest_inputs()
    first = sampler_manifest.build_base_sampler_manifest(**inputs)
    first_private = sampler_manifest.base_sampler_manifest_private_bytes(first)
    inputs["caption_mean_anchors"][:] = 0
    assert sampler_manifest.base_sampler_manifest_private_bytes(first) == first_private
    second = _manifest()
    assert sampler_manifest.base_sampler_manifest_private_bytes(second) == first_private
    changed_inputs = _manifest_inputs()
    changed_inputs["seed"] = 2718
    changed = sampler_manifest.build_base_sampler_manifest(**changed_inputs)
    assert sampler_manifest.base_sampler_manifest_private_bytes(changed) != first_private


@pytest.mark.parametrize(
    "mutation,message",
    [
        (
            lambda value: value.update(pair_commitments=tuple(reversed(value["pair_commitments"]))),
            "strictly raw32 ascending",
        ),
        (
            lambda value: value.update(
                caption_commitments=(value["caption_commitments"][0],) * 4
            ),
            "globally distinct",
        ),
        (
            lambda value: value.update(
                caption_mean_anchors=value["caption_mean_anchors"].astype(np.float32)
            ),
            "float64",
        ),
        (
            lambda value: value.update(
                caption_mean_anchors=value["caption_mean_anchors"][:, ::-1]
            ),
            "C-contiguous",
        ),
        (lambda value: value.update(seed=True), "exact built-in int"),
        (lambda value: value.update(seed=1), "outside the frozen"),
    ],
)
def test_manifest_rejects_identity_numeric_layout_and_seed_mutants(
    mutation, message: str
) -> None:
    inputs = _manifest_inputs()
    mutation(inputs)
    with pytest.raises((TypeError, sampler_manifest.SamplerManifestError), match=message):
        sampler_manifest.build_base_sampler_manifest(**inputs)


def test_cursor_and_opaque_types_fail_closed() -> None:
    manifest = _manifest()
    with pytest.raises(TypeError, match="exact built-in int"):
        sampler_manifest.planned_training_batch(
            manifest, epoch_index=True, batch_index=0
        )
    with pytest.raises(sampler_manifest.SamplerManifestError, match="closed interval"):
        sampler_manifest.planned_training_batch(
            manifest, epoch_index=30, batch_index=0
        )
    with pytest.raises(sampler_manifest.SamplerManifestError, match="closed interval"):
        sampler_manifest.planned_training_batch(
            manifest, epoch_index=0, batch_index=1
        )
    with pytest.raises(sampler_manifest.SamplerManifestError, match="minted only"):
        sampler_manifest.BaseSamplerManifest()
    with pytest.raises(sampler_manifest.SamplerManifestError, match="minted only"):
        sampler_manifest.SamplerBatchJoin()
    with pytest.raises(TypeError, match="exactly BaseSamplerManifest"):
        sampler_manifest.base_sampler_manifest_receipt_bytes(object())
    with pytest.raises(TypeError, match="exactly SamplerBatchJoin"):
        sampler_manifest.sampler_batch_join_bytes(object())


def test_authority_zero_constants_are_unambiguous() -> None:
    assert sampler_manifest.AUTHORITY == 0
    assert sampler_manifest.PRODUCTION is False
    assert sampler_manifest.EXECUTION_AUTHORIZED is False
    assert sampler_manifest.RESULT_CLAIMED is False
    assert sampler_manifest.STATUS.endswith("AUTHORITY0_NO_RESULT")
    assert sampler_manifest.SEEDS == (1729, 2718, 31415)
    assert sampler_manifest.EPOCH_COUNT == 30
