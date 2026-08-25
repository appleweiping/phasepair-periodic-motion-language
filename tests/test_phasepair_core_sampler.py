from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from phasepair_core import sampler


def _commitments(count: int) -> tuple[bytes, ...]:
    return tuple(index.to_bytes(32, "big") for index in range(count))


def _anchors(count: int) -> np.ndarray:
    anchors = np.zeros((count, 2), dtype=np.float64)
    angles = np.linspace(0.0, 1.0, count, dtype=np.float64)
    anchors[:, 0] = np.cos(angles)
    anchors[:, 1] = np.sin(angles)
    anchors /= np.linalg.norm(anchors, ord=2, axis=1, keepdims=True)
    return anchors


def test_anchor_norm_is_portable_exact_l2_not_vector_norm_alias(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    anchors = np.array(
        [[3.0 / 5.0, 4.0 / 5.0], [5.0 / 13.0, 12.0 / 13.0]],
        dtype=np.float64,
    )

    def forbidden(*_args: object, **_kwargs: object) -> np.ndarray:
        raise AssertionError("sampler must not require NumPy 2.x vector_norm")

    # NumPy 1.26 has no vector_norm attribute; raising on the NumPy 2.x alias
    # proves the implementation takes the portable but still exact L2 route.
    monkeypatch.setattr(np.linalg, "vector_norm", forbidden, raising=False)
    plan = sampler.build_epoch_batch_plan(
        _commitments(2),
        anchors,
        seed=1729,
        epoch_index=0,
    )
    assert sorted(index for batch in plan.batches for index in batch) == [0, 1]

    nonunit = anchors.copy()
    nonunit[0, 0] = 0.5
    with pytest.raises(sampler.SamplerContractError, match="L2 normalized"):
        sampler.build_epoch_batch_plan(
            _commitments(2),
            nonunit,
            seed=1729,
            epoch_index=0,
        )


def test_eta_and_toy_window_goldens() -> None:
    assert sampler.curriculum_eta(0) == 0.0
    assert sampler.curriculum_eta(13) == 0.25
    assert sampler.curriculum_eta(29) == 0.25
    assert sampler.hardness_window(3, 1000) == (992, 873, 127)
    assert sampler.hardness_window(12, 1000) == (749, 686, 127)
    mean_e3 = 1.0 - np.mean(np.arange(873, 1000, dtype=np.float64)) / 999.0
    mean_e12 = 1.0 - np.mean(np.arange(686, 813, dtype=np.float64)) / 999.0
    assert mean_e3 == pytest.approx(0.06306306306306303)
    assert mean_e12 == pytest.approx(0.2502502502502503)
    assert mean_e12 > mean_e3


@pytest.mark.parametrize("count", (1, 127, 128, 129, 257))
@pytest.mark.parametrize("epoch", (0, 3, 12, 13, 29))
def test_each_epoch_is_exact_cover_with_canonical_batch_sizes(count: int, epoch: int) -> None:
    plan = sampler.build_epoch_batch_plan(
        _commitments(count),
        _anchors(count),
        seed=1729,
        epoch_index=epoch,
    )
    flattened = tuple(index for batch in plan.batches for index in batch)
    assert sorted(flattened) == list(range(count))
    assert len(flattened) == len(set(flattened)) == count
    expected = [128] * (count // 128)
    if count % 128:
        expected.append(count % 128)
    assert [len(batch) for batch in plan.batches] == expected
    assert plan.status == "DATA_FREE_SAMPLER_PLAN_NONPRODUCTION_NO_RECEIPT"


def test_anchor_permutation_is_deterministic_and_seed_epoch_bound() -> None:
    commitments = _commitments(20)
    first = sampler.anchor_permutation(commitments, seed=1729, epoch_index=0)
    assert first == sampler.anchor_permutation(commitments, seed=1729, epoch_index=0)
    assert first != sampler.anchor_permutation(commitments, seed=2718, epoch_index=0)
    assert first != sampler.anchor_permutation(commitments, seed=1729, epoch_index=1)


def test_hardness_ranking_ties_use_raw_commitment_order() -> None:
    commitments = _commitments(130)
    anchors = np.zeros((130, 2), dtype=np.float64)
    anchors[:, 0] = 1.0
    plan = sampler.build_epoch_batch_plan(
        commitments,
        anchors,
        seed=1729,
        epoch_index=13,
    )
    anchor = plan.batches[0][0]
    candidates = tuple(index for index in range(130) if index != anchor)
    _, start, width = sampler.hardness_window(13, len(candidates))
    assert plan.batches[0][1:] == candidates[start : start + width]


def test_caption_permutation_is_exact_three_distinct_and_bound() -> None:
    captions = (b"C" * 32, b"A" * 32, b"B" * 32)
    first = sampler.caption_permutation(
        b"P" * 32,
        captions,
        seed=1729,
        epoch_index=4,
    )
    assert sorted(first) == [0, 1, 2]
    assert first == sampler.caption_permutation(
        b"P" * 32,
        captions,
        seed=1729,
        epoch_index=4,
    )
    assert first != sampler.caption_permutation(
        b"P" * 32,
        captions,
        seed=2718,
        epoch_index=4,
    )
    with pytest.raises(sampler.SamplerContractError):
        sampler.caption_permutation(
            b"P" * 32,
            (b"A" * 32, b"A" * 32, b"B" * 32),
            seed=1729,
            epoch_index=4,
        )


def test_input_contract_rejects_unsorted_types_layout_and_nonunit_rows() -> None:
    anchors = _anchors(2)
    cases = (
        ((b"B" * 32, b"A" * 32), anchors),
        (_commitments(2), anchors.astype(np.float32)),
        (_commitments(2), np.asfortranarray(anchors)),
    )
    for commitments, candidate_anchors in cases:
        with pytest.raises((TypeError, sampler.SamplerContractError)):
            sampler.build_epoch_batch_plan(
                commitments,
                candidate_anchors,
                seed=1729,
                epoch_index=0,
            )
    nonunit = anchors.copy()
    nonunit[0] *= 0.5
    with pytest.raises(sampler.SamplerContractError):
        sampler.build_epoch_batch_plan(
            _commitments(2),
            nonunit,
            seed=1729,
            epoch_index=0,
        )
    with pytest.raises(TypeError):
        sampler.build_epoch_batch_plan(
            _commitments(2),
            anchors,
            seed=True,
            epoch_index=0,
        )


def test_plan_uses_private_anchor_snapshot() -> None:
    anchors = _anchors(20)
    plan = sampler.build_epoch_batch_plan(
        _commitments(20),
        anchors,
        seed=1729,
        epoch_index=4,
    )
    anchors[:] = np.nan
    assert sorted(index for batch in plan.batches for index in batch) == list(range(20))


def test_public_globals_cannot_change_epoch_or_hash_domains() -> None:
    commitments = _commitments(8)
    captions = (b"A" * 32, b"B" * 32, b"C" * 32)
    baseline_anchor = sampler.anchor_permutation(commitments, seed=1729, epoch_index=0)
    baseline_caption = sampler.caption_permutation(
        commitments[0], captions, seed=1729, epoch_index=0
    )
    with mock.patch.multiple(
        sampler,
        EPOCH_COUNT=31,
        ANCHOR_ORDER_DOMAIN=b"forged-anchor-domain",
        CAPTION_ORDER_DOMAIN=b"forged-caption-domain",
    ):
        assert sampler.anchor_permutation(
            commitments, seed=1729, epoch_index=0
        ) == baseline_anchor
        assert sampler.caption_permutation(
            commitments[0], captions, seed=1729, epoch_index=0
        ) == baseline_caption
        with pytest.raises(sampler.SamplerContractError):
            sampler.curriculum_eta(30)


def test_epoch_plan_constructor_derives_batches_and_rejects_precomputed_forge() -> None:
    commitments = _commitments(2)
    anchors = _anchors(2)
    expected = sampler.build_epoch_batch_plan(
        commitments, anchors, seed=1729, epoch_index=0
    )
    direct = sampler.EpochBatchPlan(
        commitments, anchors, seed=1729, epoch_index=0
    )
    assert direct == expected
    with pytest.raises(TypeError):
        sampler.EpochBatchPlan(1729, 0, ((0, 1),))
