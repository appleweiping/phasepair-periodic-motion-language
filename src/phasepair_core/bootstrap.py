"""Exact paired source-cluster bootstrap primitives for PhasePair.

Seeds, captions, and retrieval directions are nested atoms inside each
source-cluster row.  Inference resamples only complete cluster rows.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np


BOOTSTRAP_DOMAIN = b"phasepair-bootstrap-v2"
PRODUCTION_DRAW_COUNT = 100_000
PERCENTILE_DRAW_COUNT = 10_000
SEED_COUNT = 3
CAPTIONS_PER_SOURCE = 3
FIXED_SEED_ORDER = (1729, 2718, 31415)
PASS = "PASS"


class BootstrapError(ValueError):
    """Raised when paired-cluster bootstrap evidence is not contract-exact."""


def _readonly_copy(array: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(array)
    # A bytes-backed ndarray cannot have WRITEABLE re-enabled by a caller.
    result = np.frombuffer(contiguous.tobytes(order="C"), dtype=contiguous.dtype).reshape(
        contiguous.shape
    )
    return result


@dataclass(frozen=True)
class ClusterEffect:
    """Exact nested-atom reduction before any cluster resampling."""

    cluster_differences: np.ndarray
    pair_commitments: tuple[bytes, ...]
    seed_order: tuple[int, int, int]
    t2m_sum: int
    m2t_sum: int
    total_sum: int
    per_seed_sums: np.ndarray
    delta_t2m: float
    delta_m2t: float
    delta: float
    delta_by_seed: np.ndarray


@dataclass(frozen=True)
class BootstrapTResult:
    exceed_count: int
    zero_variance_draw_count: int
    p_numerator: int
    p_denominator: int
    p_value: float
    observed_sum: int
    observed_sum_squares: int
    observed_variance_numerator: int


@dataclass(frozen=True)
class PercentileInterval:
    lower_numerator: int
    upper_numerator: int
    denominator: int
    lower: float
    upper: float


@dataclass(frozen=True)
class PairedBootstrapInference:
    effect: ClusterEffect
    sampling_gates: SamplingGateEvidence
    bootstrap_t: BootstrapTResult
    interval: PercentileInterval
    centered_q: int
    centered_w4: int
    max_centered_r2: int


@dataclass(frozen=True)
class SamplingGateEvidence:
    """Closed evidence for the three non-numeric sampling preconditions."""

    complete_source_cluster_roster: str
    paired_query_source_alignment: str
    nested_seed_caption_direction_atoms: str


def _pair_commitments(values: object, cluster_count: int) -> tuple[bytes, ...]:
    if isinstance(values, (bytes, bytearray, memoryview, str)):
        raise BootstrapError("pair_commitments must be a sequence of raw32 values")
    try:
        normalized = tuple(values)  # type: ignore[arg-type]
    except TypeError as exc:
        raise BootstrapError("pair_commitments must be a finite sequence") from exc
    if len(normalized) != cluster_count:
        raise BootstrapError("pair_commitments must contain exactly C rows")
    for ordinal, value in enumerate(normalized):
        if type(value) is not bytes or len(value) != 32:
            raise BootstrapError(f"pair_commitments[{ordinal}] must be exactly bytes[32]")
    if len(set(normalized)) != cluster_count:
        raise BootstrapError("pair_commitments must be unique")
    if normalized != tuple(sorted(normalized)):
        raise BootstrapError("pair_commitments must be in exact raw32-ascending order")
    return normalized


def _seed_order(value: object) -> tuple[int, int, int]:
    if (
        type(value) is not tuple
        or len(value) != len(FIXED_SEED_ORDER)
        or any(type(seed) is not int for seed in value)
        or value != FIXED_SEED_ORDER
    ):
        raise BootstrapError(f"seed_order must be exactly {FIXED_SEED_ORDER}")
    return FIXED_SEED_ORDER


def _binary_atoms(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise BootstrapError(f"{name} must be an exact numpy.ndarray, not a subclass")
    if value.dtype != np.dtype(np.bool_):
        raise BootstrapError(f"{name} must have exact dtype bool")
    if value.shape != shape:
        raise BootstrapError(f"{name} must have exact shape {shape}")
    if not value.flags.c_contiguous:
        raise BootstrapError(f"{name} must be C-contiguous")
    return np.array(value, dtype=np.bool_, copy=True, order="C", subok=False)


def paired_cluster_effect(
    treatment_t2m_top1: np.ndarray,
    baseline_t2m_top1: np.ndarray,
    treatment_m2t_top1: np.ndarray,
    baseline_m2t_top1: np.ndarray,
    *,
    pair_commitments: tuple[bytes, ...],
    seed_order: tuple[int, int, int],
) -> ClusterEffect:
    """Diagnostic reduction of nested atoms into cluster rows.

    T2M inputs have shape ``[3, C, 3]`` and M2T inputs ``[3, C]``.
    The returned ``cluster_differences`` are the exact integer ``D_c`` rows
    consumed by the paired bootstrap; seeds are never treated as sample rows.
    A returned aggregate is not accepted by the canonical inference entry
    point, which always repeats this reduction from all four raw tensors.
    """

    atom_inputs = (
        ("treatment_t2m_top1", treatment_t2m_top1),
        ("baseline_t2m_top1", baseline_t2m_top1),
        ("treatment_m2t_top1", treatment_m2t_top1),
        ("baseline_m2t_top1", baseline_m2t_top1),
    )
    for name, value in atom_inputs:
        if type(value) is not np.ndarray:
            raise BootstrapError(f"{name} must be an exact numpy.ndarray, not a subclass")
    if treatment_t2m_top1.ndim != 3:
        raise BootstrapError("treatment_t2m_top1 must have shape [3, C, 3]")
    if treatment_t2m_top1.shape[0] != SEED_COUNT or treatment_t2m_top1.shape[2] != CAPTIONS_PER_SOURCE:
        raise BootstrapError("treatment_t2m_top1 must have exact shape [3, C, 3]")
    cluster_count = treatment_t2m_top1.shape[1]
    if cluster_count == 0:
        raise BootstrapError("at least one source cluster is required")
    normalized_pairs = _pair_commitments(pair_commitments, cluster_count)
    normalized_seed_order = _seed_order(seed_order)

    t_shape = (SEED_COUNT, cluster_count, CAPTIONS_PER_SOURCE)
    m_shape = (SEED_COUNT, cluster_count)
    treatment_t = _binary_atoms("treatment_t2m_top1", treatment_t2m_top1, t_shape)
    baseline_t = _binary_atoms("baseline_t2m_top1", baseline_t2m_top1, t_shape)
    treatment_m = _binary_atoms("treatment_m2t_top1", treatment_m2t_top1, m_shape)
    baseline_m = _binary_atoms("baseline_m2t_top1", baseline_m2t_top1, m_shape)

    t_diff = treatment_t.astype(np.int64) - baseline_t.astype(np.int64)
    m_diff = treatment_m.astype(np.int64) - baseline_m.astype(np.int64)
    t2m_sum = int(t_diff.sum(dtype=np.int64))
    m2t_sum = int(m_diff.sum(dtype=np.int64))
    cluster_differences = t_diff.sum(axis=(0, 2), dtype=np.int64) + 3 * m_diff.sum(
        axis=0, dtype=np.int64
    )
    per_seed_sums = t_diff.sum(axis=(1, 2), dtype=np.int64) + 3 * m_diff.sum(
        axis=1, dtype=np.int64
    )
    total_sum = int(cluster_differences.sum(dtype=np.int64))
    if total_sum != t2m_sum + 3 * m2t_sum:
        raise AssertionError("internal nested-atom reduction mismatch")

    return ClusterEffect(
        cluster_differences=_readonly_copy(cluster_differences),
        pair_commitments=normalized_pairs,
        seed_order=normalized_seed_order,
        t2m_sum=t2m_sum,
        m2t_sum=m2t_sum,
        total_sum=total_sum,
        per_seed_sums=_readonly_copy(per_seed_sums),
        delta_t2m=t2m_sum / (9.0 * cluster_count),
        delta_m2t=m2t_sum / (3.0 * cluster_count),
        delta=total_sum / (18.0 * cluster_count),
        delta_by_seed=_readonly_copy(per_seed_sums.astype(np.float64) / (6.0 * cluster_count)),
    )


def bootstrap_index_matrix(
    cluster_count: int,
    test_evaluator_commitment_raw32: bytes,
    *,
    draw_count: int = PRODUCTION_DRAW_COUNT,
) -> np.ndarray:
    """Generate the exact SHA-256 counter bootstrap index matrix.

    This is the contract's PRNG replacement.  A caller may instead supply an
    already committed matrix to the inference functions below.
    """

    if type(cluster_count) is not int or not (1 <= cluster_count < 2**32):
        raise BootstrapError("cluster_count must be an integer in [1, 2^32)")
    if type(draw_count) is not int or not (1 <= draw_count < 2**32):
        raise BootstrapError("draw_count must be an integer in [1, 2^32)")
    if type(test_evaluator_commitment_raw32) is not bytes or len(test_evaluator_commitment_raw32) != 32:
        raise BootstrapError("test_evaluator_commitment_raw32 must be exactly bytes[32]")

    result = np.empty((draw_count, cluster_count), dtype=np.uint32)
    prefix = BOOTSTRAP_DOMAIN + b"\x00" + test_evaluator_commitment_raw32
    for draw_ordinal in range(draw_count):
        draw_bytes = draw_ordinal.to_bytes(4, "big")
        for position_ordinal in range(cluster_count):
            payload = prefix + draw_bytes + position_ordinal.to_bytes(4, "big")
            counter = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            result[draw_ordinal, position_ordinal] = counter % cluster_count
    return _readonly_copy(result)


def _cluster_differences(value: np.ndarray) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise BootstrapError("cluster_differences must be an exact numpy.ndarray, not a subclass")
    if value.ndim != 1 or value.size == 0:
        raise BootstrapError("cluster_differences must be a nonempty vector")
    if value.dtype.kind not in "iu" or value.dtype == np.dtype(np.bool_):
        raise BootstrapError("cluster_differences must have an integer dtype")
    if value.dtype.kind == "u":
        outside_range = bool(np.any(value > 18))
    else:
        outside_range = bool(np.any(value < -18) or np.any(value > 18))
    if outside_range:
        raise BootstrapError("each cluster difference must be in [-18, 18]")
    normalized = value.astype(np.int64, copy=True)
    return normalized


def _index_matrix(value: np.ndarray, cluster_count: int, draw_count: int) -> np.ndarray:
    if type(value) is not np.ndarray:
        raise BootstrapError("index_matrix must be an exact numpy.ndarray, not a subclass")
    if value.dtype != np.dtype(np.uint32):
        raise BootstrapError("index_matrix must have exact dtype uint32")
    if not value.flags.c_contiguous:
        raise BootstrapError("index_matrix must be C-contiguous")
    if value.shape != (draw_count, cluster_count):
        raise BootstrapError(
            f"index_matrix must have exact shape ({draw_count}, {cluster_count})"
        )
    if np.any(value < 0) or np.any(value >= cluster_count):
        raise BootstrapError("index_matrix contains an out-of-range cluster ordinal")
    return _readonly_copy(
        np.array(value, dtype=np.uint32, copy=True, order="C", subok=False)
    )


def _draw_sums(differences: np.ndarray, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    draw_count = indices.shape[0]
    sums = np.empty(draw_count, dtype=np.int64)
    sums_of_squares = np.empty(draw_count, dtype=np.int64)
    for start in range(0, draw_count, 4096):
        stop = min(start + 4096, draw_count)
        sampled = differences[indices[start:stop]]
        sums[start:stop] = sampled.sum(axis=1, dtype=np.int64)
        sums_of_squares[start:stop] = np.square(sampled, dtype=np.int64).sum(
            axis=1, dtype=np.int64
        )
    return sums, sums_of_squares


def null_centered_bootstrap_t(
    cluster_differences: np.ndarray,
    index_matrix: np.ndarray,
) -> BootstrapTResult:
    """Diagnostic primitive for the exact one-sided bootstrap-t arithmetic.

    This function intentionally does not establish production sampling gates
    or bind an evaluator commitment.  Use :func:`paired_bootstrap_inference`
    for canonical inference.
    """

    differences = _cluster_differences(cluster_differences)
    cluster_count = differences.size
    indices = _index_matrix(index_matrix, cluster_count, PRODUCTION_DRAW_COUNT)
    return _null_centered_bootstrap_t_validated(differences, indices)


def _null_centered_bootstrap_t_validated(
    differences: np.ndarray,
    indices: np.ndarray,
) -> BootstrapTResult:
    draw_sums, draw_square_sums = _draw_sums(differences, indices)

    cluster_count = differences.size
    observed_sum = int(differences.sum(dtype=np.int64))
    observed_sum_squares = int(np.square(differences, dtype=np.int64).sum(dtype=np.int64))
    observed_variance = cluster_count * observed_sum_squares - observed_sum * observed_sum
    exceed_count = 0
    zero_variance_draw_count = 0
    for draw_sum_np, draw_square_sum_np in zip(draw_sums, draw_square_sums):
        draw_sum = int(draw_sum_np)
        draw_square_sum = int(draw_square_sum_np)
        draw_variance = cluster_count * draw_square_sum - draw_sum * draw_sum
        centered_numerator = draw_sum - observed_sum
        if draw_variance == 0:
            zero_variance_draw_count += 1
            if centered_numerator > 0:
                exceed_count += 1
        elif centered_numerator > 0 and (
            centered_numerator * centered_numerator * observed_variance
            >= observed_sum * observed_sum * draw_variance
        ):
            exceed_count += 1

    p_numerator = 1 + exceed_count
    p_denominator = 1 + PRODUCTION_DRAW_COUNT
    return BootstrapTResult(
        exceed_count=exceed_count,
        zero_variance_draw_count=zero_variance_draw_count,
        p_numerator=p_numerator,
        p_denominator=p_denominator,
        p_value=p_numerator / p_denominator,
        observed_sum=observed_sum,
        observed_sum_squares=observed_sum_squares,
        observed_variance_numerator=observed_variance,
    )


def paired_percentile_interval(
    cluster_differences: np.ndarray,
    index_matrix: np.ndarray,
) -> PercentileInterval:
    """Diagnostic primitive for the contract's interpolated 95% interval.

    This function intentionally permits the frozen C=3 arithmetic toy.  It is
    not a canonical inference entry point.
    """

    differences = _cluster_differences(cluster_differences)
    cluster_count = differences.size
    indices = _index_matrix(index_matrix, cluster_count, PRODUCTION_DRAW_COUNT)
    return _paired_percentile_interval_validated(differences, indices)


def _paired_percentile_interval_validated(
    differences: np.ndarray,
    indices: np.ndarray,
) -> PercentileInterval:
    cluster_count = differences.size
    first_indices = indices[:PERCENTILE_DRAW_COUNT]
    draw_sums, _ = _draw_sums(differences, first_indices)
    ordered = np.sort(draw_sums, kind="stable")
    lower_numerator = int(ordered[249]) + 39 * int(ordered[250])
    upper_numerator = 39 * int(ordered[9749]) + int(ordered[9750])
    denominator = 720 * cluster_count
    return PercentileInterval(
        lower_numerator=lower_numerator,
        upper_numerator=upper_numerator,
        denominator=denominator,
        lower=lower_numerator / denominator,
        upper=upper_numerator / denominator,
    )


def paired_bootstrap_inference(
    treatment_t2m_top1: np.ndarray,
    baseline_t2m_top1: np.ndarray,
    treatment_m2t_top1: np.ndarray,
    baseline_m2t_top1: np.ndarray,
    *,
    pair_commitments: tuple[bytes, ...],
    seed_order: tuple[int, int, int],
    sampling_gates: SamplingGateEvidence,
    test_evaluator_commitment_raw32: bytes,
) -> PairedBootstrapInference:
    """Produce canonical inference only after every sampling gate passes.

    The low-level bootstrap helpers remain available for contract diagnostics
    (including the frozen C=3 toy).  This canonical entry point fails closed
    before computing p-values or intervals unless all production assumptions
    are established.
    """

    # Never accept a caller-supplied aggregate: directional and per-seed
    # decompositions are not recoverable from D_c alone.
    effect = paired_cluster_effect(
        treatment_t2m_top1,
        baseline_t2m_top1,
        treatment_m2t_top1,
        baseline_m2t_top1,
        pair_commitments=pair_commitments,
        seed_order=seed_order,
    )
    cluster_count, observed_sum, observed_variance, centered_q, centered_w4, max_r2 = (
        _revalidate_cluster_effect(effect)
    )
    _validate_sampling_gates(sampling_gates)
    if cluster_count < 50:
        raise BootstrapError("canonical inference requires at least 50 source clusters")
    if observed_sum <= 0:
        raise BootstrapError("canonical inference requires a strictly positive paired effect")
    if observed_variance <= 0:
        raise BootstrapError("canonical inference requires positive cluster variance")
    if 10 * max_r2 > centered_q:
        raise BootstrapError("canonical inference failed the maximum-influence gate")
    if centered_q * centered_q < 30 * centered_w4:
        raise BootstrapError("canonical inference failed the effective-cluster-count gate")

    index_matrix = bootstrap_index_matrix(
        cluster_count,
        test_evaluator_commitment_raw32,
    )
    # Both calculations consume this same internally generated immutable
    # snapshot; no caller-controlled matrix is accepted by this entry point.
    bootstrap_t = _null_centered_bootstrap_t_validated(
        effect.cluster_differences,
        index_matrix,
    )
    interval = _paired_percentile_interval_validated(
        effect.cluster_differences,
        index_matrix,
    )
    return PairedBootstrapInference(
        effect=effect,
        sampling_gates=sampling_gates,
        bootstrap_t=bootstrap_t,
        interval=interval,
        centered_q=centered_q,
        centered_w4=centered_w4,
        max_centered_r2=max_r2,
    )


def _validate_sampling_gates(value: object) -> None:
    if type(value) is not SamplingGateEvidence:
        raise BootstrapError("sampling_gates must be exact SamplingGateEvidence")
    gate_values = (
        value.complete_source_cluster_roster,
        value.paired_query_source_alignment,
        value.nested_seed_caption_direction_atoms,
    )
    if any(type(gate) is not str or gate != PASS for gate in gate_values):
        raise BootstrapError("all three closed sampling gates must be exact PASS")


def _revalidate_cluster_effect(effect: object) -> tuple[int, int, int, int, int, int]:
    if type(effect) is not ClusterEffect:
        raise BootstrapError("effect must be exact ClusterEffect")
    differences = effect.cluster_differences
    if (
        not isinstance(differences, np.ndarray)
        or differences.dtype != np.dtype(np.int64)
        or differences.ndim != 1
        or differences.size == 0
        or not differences.flags.c_contiguous
        or differences.flags.writeable
    ):
        raise BootstrapError("effect cluster_differences representation is not canonical")
    if np.any(differences < -18) or np.any(differences > 18):
        raise BootstrapError("effect cluster_differences are outside [-18, 18]")
    cluster_count = differences.size
    if effect.pair_commitments != _pair_commitments(effect.pair_commitments, cluster_count):
        raise BootstrapError("effect pair commitments are not canonical")
    _seed_order(effect.seed_order)

    per_seed = effect.per_seed_sums
    if (
        not isinstance(per_seed, np.ndarray)
        or per_seed.dtype != np.dtype(np.int64)
        or per_seed.shape != (SEED_COUNT,)
        or not per_seed.flags.c_contiguous
        or per_seed.flags.writeable
    ):
        raise BootstrapError("effect per_seed_sums representation is not canonical")
    delta_by_seed = effect.delta_by_seed
    if (
        not isinstance(delta_by_seed, np.ndarray)
        or delta_by_seed.dtype != np.dtype(np.float64)
        or delta_by_seed.shape != (SEED_COUNT,)
        or not delta_by_seed.flags.c_contiguous
        or delta_by_seed.flags.writeable
    ):
        raise BootstrapError("effect delta_by_seed representation is not canonical")
    integer_fields = (effect.t2m_sum, effect.m2t_sum, effect.total_sum)
    if any(type(field) is not int for field in integer_fields):
        raise BootstrapError("effect aggregate sums must be exact integers")
    if not (-9 * cluster_count <= effect.t2m_sum <= 9 * cluster_count):
        raise BootstrapError("effect t2m_sum is outside its exact atom range")
    if not (-3 * cluster_count <= effect.m2t_sum <= 3 * cluster_count):
        raise BootstrapError("effect m2t_sum is outside its exact atom range")
    observed_sum = int(differences.sum(dtype=np.int64))
    if effect.total_sum != observed_sum or effect.total_sum != effect.t2m_sum + 3 * effect.m2t_sum:
        raise BootstrapError("effect integer aggregates are inconsistent")
    if int(per_seed.sum(dtype=np.int64)) != effect.total_sum:
        raise BootstrapError("effect per-seed aggregates are inconsistent")
    if np.any(per_seed < -6 * cluster_count) or np.any(per_seed > 6 * cluster_count):
        raise BootstrapError("effect per-seed aggregates are outside their atom range")

    expected_delta_t = effect.t2m_sum / (9.0 * cluster_count)
    expected_delta_m = effect.m2t_sum / (3.0 * cluster_count)
    expected_delta = effect.total_sum / (18.0 * cluster_count)
    expected_delta_by_seed = per_seed.astype(np.float64) / (6.0 * cluster_count)
    scalar_deltas = (effect.delta_t2m, effect.delta_m2t, effect.delta)
    if any(type(value) is not float or not np.isfinite(value) for value in scalar_deltas):
        raise BootstrapError("effect scalar deltas must be finite exact floats")
    if (
        np.float64(effect.delta_t2m).tobytes() != np.float64(expected_delta_t).tobytes()
        or np.float64(effect.delta_m2t).tobytes() != np.float64(expected_delta_m).tobytes()
        or np.float64(effect.delta).tobytes() != np.float64(expected_delta).tobytes()
        or delta_by_seed.tobytes(order="C") != expected_delta_by_seed.tobytes(order="C")
    ):
        raise BootstrapError("effect normalized aggregates are inconsistent")

    observed_sum_squares = int(np.square(differences, dtype=np.int64).sum(dtype=np.int64))
    observed_variance = cluster_count * observed_sum_squares - observed_sum * observed_sum
    centered_residuals = cluster_count * differences - observed_sum
    residual_squares = np.square(centered_residuals, dtype=np.int64)
    centered_q = int(residual_squares.sum(dtype=np.int64))
    centered_w4 = sum(int(value) * int(value) for value in residual_squares)
    max_r2 = int(residual_squares.max())
    if centered_q != cluster_count * observed_variance:
        raise AssertionError("internal centered-moment identity mismatch")
    return cluster_count, observed_sum, observed_variance, centered_q, centered_w4, max_r2


__all__ = [
    "BOOTSTRAP_DOMAIN",
    "BootstrapError",
    "BootstrapTResult",
    "CAPTIONS_PER_SOURCE",
    "ClusterEffect",
    "FIXED_SEED_ORDER",
    "PASS",
    "PERCENTILE_DRAW_COUNT",
    "PRODUCTION_DRAW_COUNT",
    "PairedBootstrapInference",
    "PercentileInterval",
    "SamplingGateEvidence",
    "SEED_COUNT",
    "bootstrap_index_matrix",
    "null_centered_bootstrap_t",
    "paired_bootstrap_inference",
    "paired_cluster_effect",
    "paired_percentile_interval",
]
