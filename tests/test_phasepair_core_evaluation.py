"""Synthetic-only checks for PhasePair evaluation and cluster bootstrap.

These fixtures do not contain project data and do not constitute a result.
"""

from __future__ import annotations

from dataclasses import replace
import hashlib

import numpy as np
import pytest

from phasepair_core.bootstrap import (
    BootstrapError,
    FIXED_SEED_ORDER,
    PASS,
    PRODUCTION_DRAW_COUNT,
    SamplingGateEvidence,
    bootstrap_index_matrix,
    null_centered_bootstrap_t,
    paired_bootstrap_inference,
    paired_cluster_effect,
    paired_percentile_interval,
)
from phasepair_core.evaluation import EvaluationError, evaluate_full_gallery


OUTPUT_SCOPE = "DATA_FREE_NONPRODUCTION"
NO_RESULT = "NO_RESULT"


def _raw32(value: int) -> bytes:
    return value.to_bytes(32, "big")


class _LyingArray(np.ndarray):
    """Adversarial ndarray view whose scalar reads do not match its bytes."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if np.isscalar(value):
            return 1
        return value


class _DynamicNdimArray(np.ndarray):
    """Array subclass that fails if shape metadata is read before rejection."""

    def __getattribute__(self, name):
        if name == "ndim":
            raise RuntimeError("DYNAMIC_NDIM_READ")
        return super().__getattribute__(name)


def _gallery_fixture() -> dict[str, object]:
    motion_sources = (_raw32(100), _raw32(200))
    caption_sources = (motion_sources[0],) * 3 + (motion_sources[1],) * 3
    motion_ties = (_raw32(1), _raw32(2))
    caption_ties = tuple(_raw32(10 + ordinal) for ordinal in range(6))
    scores = np.array(
        [
            [0.9, 0.8, 0.7, 0.5, 0.4, 0.3],
            [0.1, 0.8, 0.9, 0.5, 0.6, 0.7],
        ],
        dtype=np.float32,
    )
    positive_mask = np.equal(
        np.asarray(motion_sources, dtype="|S32")[:, None],
        np.asarray(caption_sources, dtype="|S32")[None, :],
    )
    return {
        "scores": scores,
        "positive_mask": positive_mask,
        "motion_source_cluster_ids": motion_sources,
        "caption_source_cluster_ids": caption_sources,
        "motion_pair_commitments": motion_ties,
        "caption_commitments": caption_ties,
    }


def test_full_gallery_multi_positive_and_contract_ties_are_exact() -> None:
    result = evaluate_full_gallery(**_gallery_fixture())

    # Caption 1 ties and chooses motion 0 by pair commitment. Caption 3 ties
    # but its positive motion 1 therefore ranks second. M2T motion 1's first
    # positive is rank 3 because two negative tied captions precede it.
    np.testing.assert_array_equal(result.t2m.ranks, [1, 1, 2, 2, 1, 1])
    np.testing.assert_array_equal(result.m2t.ranks, [1, 3])
    assert result.t2m.r_at_1 == pytest.approx(2.0 / 3.0)
    assert result.t2m.r_at_3 == result.t2m.r_at_5 == result.t2m.r_at_10 == 1.0
    assert result.t2m.median_rank == 1.0
    assert result.m2t.r_at_1 == 0.5
    assert result.m2t.r_at_3 == result.m2t.r_at_5 == result.m2t.r_at_10 == 1.0
    assert result.m2t.median_rank == 2.0
    assert result.t2m_cluster_macro_r_at_1 == pytest.approx(2.0 / 3.0)
    assert result.m2t_any_caption_r_at_1 == 0.5
    assert result.bidirectional_mean_r_at_1 == pytest.approx(7.0 / 12.0)
    np.testing.assert_array_equal(result.t2m_top1_by_source, [[True, True, False], [False, True, True]])
    np.testing.assert_array_equal(result.m2t_any_caption_top1_by_source, [True, False])
    assert not result.t2m.ranks.flags.writeable
    assert result.t2m.recall_at(1) == result.t2m.r_at_1
    for aliased_cutoff in (True, 1.0, np.int64(1)):
        with pytest.raises(EvaluationError, match="exact built-in int"):
            result.t2m.recall_at(aliased_cutoff)  # type: ignore[arg-type]


def test_full_gallery_is_invariant_to_caption_permutation_but_rejects_row_reorder() -> None:
    fixture = _gallery_fixture()
    original = evaluate_full_gallery(**fixture)
    column_order = np.array([5, 2, 0, 4, 1, 3])

    permuted = evaluate_full_gallery(
        np.ascontiguousarray(fixture["scores"][:, column_order]),
        positive_mask=np.ascontiguousarray(fixture["positive_mask"][:, column_order]),
        motion_source_cluster_ids=fixture["motion_source_cluster_ids"],
        caption_source_cluster_ids=tuple(fixture["caption_source_cluster_ids"][i] for i in column_order),
        motion_pair_commitments=fixture["motion_pair_commitments"],
        caption_commitments=tuple(fixture["caption_commitments"][i] for i in column_order),
    )

    assert permuted.t2m.r_at_1 == original.t2m.r_at_1
    assert permuted.t2m.median_rank == original.t2m.median_rank
    assert permuted.m2t.r_at_1 == original.m2t.r_at_1
    assert permuted.m2t.median_rank == original.m2t.median_rank
    assert permuted.bidirectional_mean_r_at_1 == original.bidirectional_mean_r_at_1

    row_order = np.array([1, 0])
    with pytest.raises(EvaluationError, match="raw32-ascending"):
        evaluate_full_gallery(
            np.ascontiguousarray(fixture["scores"][row_order]),
            positive_mask=np.ascontiguousarray(fixture["positive_mask"][row_order]),
            motion_source_cluster_ids=tuple(fixture["motion_source_cluster_ids"][i] for i in row_order),
            caption_source_cluster_ids=fixture["caption_source_cluster_ids"],
            motion_pair_commitments=tuple(fixture["motion_pair_commitments"][i] for i in row_order),
            caption_commitments=fixture["caption_commitments"],
        )


def test_positive_mask_rejects_duplicate_text_cross_cluster_leakage() -> None:
    fixture = _gallery_fixture()
    leaked = fixture["positive_mask"].copy()
    leaked[0, 3] = True
    fixture["positive_mask"] = leaked
    with pytest.raises(EvaluationError, match="source-cluster identity"):
        evaluate_full_gallery(**fixture)


def test_evaluation_rejects_lying_ndarray_subclasses_before_any_read() -> None:
    fixture = _gallery_fixture()
    lying_scores = np.zeros_like(fixture["scores"]).view(_LyingArray)
    fixture["scores"] = lying_scores
    with pytest.raises(EvaluationError, match="not a subclass"):
        evaluate_full_gallery(**fixture)

    fixture = _gallery_fixture()
    lying_mask = np.zeros_like(fixture["positive_mask"]).view(_LyingArray)
    fixture["positive_mask"] = lying_mask
    with pytest.raises(EvaluationError, match="not a subclass"):
        evaluate_full_gallery(**fixture)


@pytest.mark.parametrize(
    "mutation, message",
    [
        (lambda f: f.update(scores=f["scores"].astype(np.float64)), "float32"),
        (lambda f: f.update(scores=f["scores"][:, ::-1]), "C-contiguous"),
        (lambda f: f["scores"].__setitem__((0, 0), np.nan), "finite"),
        (lambda f: f.update(scores=f["scores"][:, :-1].copy()), "shape must be exactly"),
        (lambda f: f.update(positive_mask=f["positive_mask"].astype(np.uint8)), "dtype bool"),
        (lambda f: f.update(positive_mask=f["positive_mask"][:, ::-1]), "C-contiguous"),
        (lambda f: f.update(positive_mask=f["positive_mask"][:, :-1].copy()), "shape must exactly"),
        (
            lambda f: f.update(caption_commitments=(f["caption_commitments"][0],) * 6),
            "unique",
        ),
        (
            lambda f: f.update(caption_source_cluster_ids=f["caption_source_cluster_ids"][:-1] + (_raw32(999),)),
            "resolve",
        ),
    ],
)
def test_full_gallery_rejects_invalid_shape_identity_or_storage(mutation, message: str) -> None:
    fixture = _gallery_fixture()
    mutation(fixture)
    with pytest.raises(EvaluationError, match=message):
        evaluate_full_gallery(**fixture)


def test_nested_cluster_effect_does_not_treat_three_seeds_as_samples() -> None:
    baseline_t = np.zeros((3, 2, 3), dtype=np.bool_)
    baseline_m = np.zeros((3, 2), dtype=np.bool_)
    treatment_t = baseline_t.copy()
    treatment_m = baseline_m.copy()
    treatment_t[:, 0, :] = True
    treatment_m[:, 0] = True

    effect = paired_cluster_effect(
        treatment_t,
        baseline_t,
        treatment_m,
        baseline_m,
        pair_commitments=(_raw32(1), _raw32(2)),
        seed_order=FIXED_SEED_ORDER,
    )
    np.testing.assert_array_equal(effect.cluster_differences, [18, 0])
    np.testing.assert_array_equal(effect.per_seed_sums, [6, 6, 6])
    assert effect.t2m_sum == 9
    assert effect.m2t_sum == 3
    assert effect.total_sum == 18
    assert effect.pair_commitments == (_raw32(1), _raw32(2))
    assert effect.seed_order == FIXED_SEED_ORDER
    assert effect.delta_t2m == effect.delta_m2t == effect.delta == 0.5
    np.testing.assert_array_equal(effect.delta_by_seed, [0.5, 0.5, 0.5])
    assert not effect.cluster_differences.flags.writeable


def test_bootstrap_counter_golden_and_serialized_prefix() -> None:
    indices = bootstrap_index_matrix(3, bytes(32), draw_count=2)
    np.testing.assert_array_equal(indices, [[2, 1, 1], [0, 0, 1]])
    encoded = indices.astype(np.uint8).tobytes(order="C")
    assert hashlib.sha256(encoded).hexdigest() == "4f39a628bef711358232cbbbd0db29369bffc32401e7c57c5b5ae19274e05199"
    assert not indices.flags.writeable


def test_bootstrap_t_and_interval_use_complete_cluster_rows() -> None:
    differences = np.array([-1, 2, 0], dtype=np.int64)
    indices = np.tile(np.array([[0, 1, 2]], dtype=np.uint32), (PRODUCTION_DRAW_COUNT, 1))

    bootstrap_t = null_centered_bootstrap_t(differences, indices)
    assert bootstrap_t.observed_sum == 1
    assert bootstrap_t.observed_sum_squares == 5
    assert bootstrap_t.observed_variance_numerator == 14
    assert bootstrap_t.exceed_count == 0
    assert bootstrap_t.zero_variance_draw_count == 0
    assert (bootstrap_t.p_numerator, bootstrap_t.p_denominator) == (1, 100001)

    interval = paired_percentile_interval(differences, indices)
    assert (interval.lower_numerator, interval.upper_numerator, interval.denominator) == (40, 40, 2160)
    assert interval.lower == interval.upper == pytest.approx(1.0 / 54.0)

def test_bootstrap_rejects_atom_and_index_leakage() -> None:
    t = np.zeros((3, 2, 3), dtype=np.bool_)
    m = np.zeros((3, 2), dtype=np.bool_)
    with pytest.raises(BootstrapError, match="dtype bool"):
        paired_cluster_effect(
            t.astype(np.int8),
            t,
            m,
            m,
            pair_commitments=(_raw32(1), _raw32(2)),
            seed_order=FIXED_SEED_ORDER,
        )
    with pytest.raises(BootstrapError, match="shape"):
        paired_cluster_effect(
            t[:, :, :2],
            t[:, :, :2],
            m,
            m,
            pair_commitments=(_raw32(1), _raw32(2)),
            seed_order=FIXED_SEED_ORDER,
        )
    with pytest.raises(BootstrapError, match="C-contiguous"):
        paired_cluster_effect(
            t[:, :, ::-1],
            t,
            m,
            m,
            pair_commitments=(_raw32(1), _raw32(2)),
            seed_order=FIXED_SEED_ORDER,
        )
    with pytest.raises(BootstrapError, match="not a subclass"):
        paired_cluster_effect(
            t.view(_LyingArray),
            t,
            m,
            m,
            pair_commitments=(_raw32(1), _raw32(2)),
            seed_order=FIXED_SEED_ORDER,
        )
    with pytest.raises(BootstrapError, match="not a subclass"):
        paired_cluster_effect(
            t.view(_DynamicNdimArray),
            t,
            m,
            m,
            pair_commitments=(_raw32(1), _raw32(2)),
            seed_order=FIXED_SEED_ORDER,
        )
    with pytest.raises(BootstrapError, match="raw32-ascending"):
        paired_cluster_effect(
            t,
            t,
            m,
            m,
            pair_commitments=(_raw32(2), _raw32(1)),
            seed_order=FIXED_SEED_ORDER,
        )
    with pytest.raises(BootstrapError, match="seed_order"):
        paired_cluster_effect(
            t,
            t,
            m,
            m,
            pair_commitments=(_raw32(1), _raw32(2)),
            seed_order=(1729, 31415, 2718),
        )
    with pytest.raises(BootstrapError, match="seed_order"):
        paired_cluster_effect(
            t,
            t,
            m,
            m,
            pair_commitments=(_raw32(1), _raw32(2)),
            seed_order=(1729.0, 2718.0, 31415.0),  # type: ignore[arg-type]
        )
    with pytest.raises(BootstrapError, match=r"bytes\[32\]"):
        bootstrap_index_matrix(3, b"short", draw_count=2)

    differences = np.array([-1, 2, 0], dtype=np.int64)
    bad_indices = np.zeros((PRODUCTION_DRAW_COUNT, 3), dtype=np.uint32)
    bad_indices[0, 0] = 3
    with pytest.raises(BootstrapError, match="out-of-range"):
        null_centered_bootstrap_t(differences, bad_indices)

    with pytest.raises(BootstrapError, match="dtype uint32"):
        null_centered_bootstrap_t(
            differences,
            np.zeros((PRODUCTION_DRAW_COUNT, 3), dtype=np.uint64),
        )
    noncontiguous = np.zeros((PRODUCTION_DRAW_COUNT, 6), dtype=np.uint32)[:, ::2]
    assert not noncontiguous.flags.c_contiguous
    with pytest.raises(BootstrapError, match="C-contiguous"):
        null_centered_bootstrap_t(differences, noncontiguous)
    lying_indices = np.zeros((PRODUCTION_DRAW_COUNT, 3), dtype=np.uint32).view(_LyingArray)
    with pytest.raises(BootstrapError, match="not a subclass"):
        null_centered_bootstrap_t(differences, lying_indices)
    wrapped_unsigned = np.array([2**64 - 1, 2, 0], dtype=np.uint64)
    with pytest.raises(BootstrapError, match=r"\[-18, 18\]"):
        null_centered_bootstrap_t(wrapped_unsigned, np.zeros_like(bad_indices))
    with pytest.raises(BootstrapError, match=r"\[-18, 18\]"):
        paired_percentile_interval(wrapped_unsigned, np.zeros_like(bad_indices))


def _passing_gates() -> SamplingGateEvidence:
    return SamplingGateEvidence(PASS, PASS, PASS)


def _balanced_canonical_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[bytes, ...],
    bytes,
]:
    cluster_count = 50
    baseline_t = np.zeros((3, cluster_count, 3), dtype=np.bool_)
    baseline_m = np.zeros((3, cluster_count), dtype=np.bool_)
    treatment_t = baseline_t.copy()
    treatment_t[0, :25, 0] = True
    treatment_m = baseline_m.copy()
    pairs = tuple(_raw32(i + 1) for i in range(cluster_count))
    evaluator_commitment = _raw32(7001)
    return (
        treatment_t,
        baseline_t,
        treatment_m,
        baseline_m,
        pairs,
        evaluator_commitment,
    )


def _canonical_call(
    treatment_t: np.ndarray,
    baseline_t: np.ndarray,
    treatment_m: np.ndarray,
    baseline_m: np.ndarray,
    pairs: tuple[bytes, ...],
    evaluator_commitment: bytes,
    *,
    gates: SamplingGateEvidence | None = None,
):
    return paired_bootstrap_inference(
        treatment_t,
        baseline_t,
        treatment_m,
        baseline_m,
        pair_commitments=pairs,
        seed_order=FIXED_SEED_ORDER,
        sampling_gates=_passing_gates() if gates is None else gates,
        test_evaluator_commitment_raw32=evaluator_commitment,
    )


def test_canonical_inference_recomputes_raw_atoms_and_all_sampling_gates() -> None:
    inputs = _balanced_canonical_inputs()
    treatment_t, baseline_t, treatment_m, baseline_m, pairs, commitment = inputs
    result = _canonical_call(*inputs)
    assert result.effect.delta == pytest.approx(1.0 / 36.0)
    assert result.bootstrap_t.p_value == pytest.approx(1.0 / 100001.0)
    assert result.interval.lower < result.effect.delta <= result.interval.upper
    assert result.centered_q > 0
    assert result.centered_q * result.centered_q >= 30 * result.centered_w4
    assert 10 * result.max_centered_r2 <= result.centered_q

    # The former accepted-invalid aggregate substitutions cannot enter the
    # canonical API because it has no ClusterEffect parameter.
    direction_forged = replace(
        result.effect,
        t2m_sum=result.effect.t2m_sum - 3,
        m2t_sum=result.effect.m2t_sum + 1,
        delta_t2m=(result.effect.t2m_sum - 3) / (9.0 * 50),
        delta_m2t=(result.effect.m2t_sum + 1) / (3.0 * 50),
    )
    shifted_seed_sums = np.array(result.effect.per_seed_sums, copy=True)
    shifted_seed_sums[0] += 1
    shifted_seed_sums[1] -= 1
    shifted_seed_sums.setflags(write=False)
    shifted_seed_deltas = shifted_seed_sums.astype(np.float64) / (6.0 * 50)
    shifted_seed_deltas.setflags(write=False)
    seed_forged = replace(
        result.effect,
        per_seed_sums=shifted_seed_sums,
        delta_by_seed=shifted_seed_deltas,
    )
    for forged in (direction_forged, seed_forged):
        with pytest.raises(TypeError, match="unexpected keyword argument 'effect'"):
            paired_bootstrap_inference(
                treatment_t,
                baseline_t,
                treatment_m,
                baseline_m,
                effect=forged,
                pair_commitments=pairs,
                seed_order=FIXED_SEED_ORDER,
                sampling_gates=_passing_gates(),
                test_evaluator_commitment_raw32=commitment,
            )

    failed_gate = SamplingGateEvidence(PASS, "FAIL", PASS)
    with pytest.raises(BootstrapError, match="all three"):
        _canonical_call(
            treatment_t,
            baseline_t,
            treatment_m,
            baseline_m,
            pairs,
            commitment,
            gates=failed_gate,
        )

    with pytest.raises(TypeError, match="unexpected keyword argument 'index_matrix'"):
        paired_bootstrap_inference(
            treatment_t,
            baseline_t,
            treatment_m,
            baseline_m,
            index_matrix=np.zeros((1, 1), dtype=np.uint32),
            pair_commitments=pairs,
            seed_order=FIXED_SEED_ORDER,
            sampling_gates=_passing_gates(),
            test_evaluator_commitment_raw32=commitment,
        )
    with pytest.raises(BootstrapError, match=r"bytes\[32\]"):
        _canonical_call(
            treatment_t,
            baseline_t,
            treatment_m,
            baseline_m,
            pairs,
            b"wrong",
        )

    # Returned arrays are bytes-backed, and later mutation of mutable raw
    # inputs cannot rewrite the internally recomputed effect.
    for array in (
        result.effect.cluster_differences,
        result.effect.per_seed_sums,
        result.effect.delta_by_seed,
    ):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    frozen_differences = result.effect.cluster_differences.copy()
    treatment_t[:] = False
    treatment_m[:] = True
    np.testing.assert_array_equal(result.effect.cluster_differences, frozen_differences)


def test_canonical_inference_holds_small_or_unrepresentative_rosters() -> None:
    baseline_t = np.zeros((3, 3, 3), dtype=np.bool_)
    baseline_m = np.zeros((3, 3), dtype=np.bool_)
    treatment_t = baseline_t.copy()
    treatment_t[0, 0, 0] = True
    small_m = baseline_m.copy()
    small_pairs = (_raw32(1), _raw32(2), _raw32(3))
    with pytest.raises(BootstrapError, match="at least 50"):
        _canonical_call(
            treatment_t,
            baseline_t,
            small_m,
            baseline_m,
            small_pairs,
            _raw32(7002),
        )

    cluster_count = 50
    large_t = np.zeros((3, cluster_count, 3), dtype=np.bool_)
    large_m = np.zeros((3, cluster_count), dtype=np.bool_)
    one_cluster = large_t.copy()
    one_cluster[0, 0, 0] = True
    large_pairs = tuple(_raw32(i + 1) for i in range(cluster_count))
    with pytest.raises(BootstrapError, match="maximum-influence"):
        _canonical_call(
            one_cluster,
            large_t,
            large_m,
            large_m,
            large_pairs,
            _raw32(7003),
        )

    ten_clusters = large_t.copy()
    ten_clusters[0, :10, 0] = True
    with pytest.raises(BootstrapError, match="effective-cluster-count"):
        _canonical_call(
            ten_clusters,
            large_t,
            large_m,
            large_m,
            large_pairs,
            _raw32(7004),
        )

    constant_positive = large_t.copy()
    constant_positive[0, :, 0] = True
    with pytest.raises(BootstrapError, match="positive cluster variance"):
        _canonical_call(
            constant_positive,
            large_t,
            large_m,
            large_m,
            large_pairs,
            _raw32(7005),
        )

    negative_baseline = large_t.copy()
    negative_baseline[0, :25, 0] = True
    with pytest.raises(BootstrapError, match="strictly positive"):
        _canonical_call(
            large_t,
            negative_baseline,
            large_m,
            large_m,
            large_pairs,
            _raw32(7006),
        )


def test_module_is_synthetic_no_result_only() -> None:
    assert OUTPUT_SCOPE == "DATA_FREE_NONPRODUCTION"
    assert NO_RESULT == "NO_RESULT"
