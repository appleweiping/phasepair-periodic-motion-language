"""Synthetic-only tests for semantic nine-run base qualification."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
import hashlib
import json

import pytest

from phasepair_core import base_qualification as qualification
from phasepair_core import execution_schema


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


_BOUNDARY_SCORES = {
    "00": (600, 620, 600),
    "07": (605, 600, 590),
    "08": (604, 610, 590),
}


def _rows(
    scores: dict[str, tuple[int, int, int]] | None = None,
) -> tuple[qualification.FrozenBaseQualificationRow, ...]:
    selected_scores = _BOUNDARY_SCORES if scores is None else scores
    rows: list[qualification.FrozenBaseQualificationRow] = []
    for system_id in qualification.SYSTEM_ORDER:
        for seed_index, seed in enumerate(qualification.SEED_ORDER):
            rows.append(
                qualification.FrozenBaseQualificationRow(
                    system_id=system_id,
                    seed=seed,
                    run_id=f"phasepair-run-v2/BASE_TRAIN/{seed}/{system_id}",
                    terminal_success=True,
                    terminal_execution_receipt_sha256=_digest(
                        f"terminal/{system_id}/{seed}"
                    ),
                    selected_checkpoint_sha256=_digest(
                        f"checkpoint/{system_id}/{seed}"
                    ),
                    full_score_matrix_sha256=_digest(f"score/{system_id}/{seed}"),
                    evaluator_sha256=_digest("evaluator/frozen"),
                    optimizer_opt_receipt_sha256=_digest(
                        f"optimizer-opt/{system_id}/{seed}"
                    ),
                    dropout_bd_receipt_sha256=_digest(f"dropout-bd/{system_id}/{seed}"),
                    overfit_receipt_sha256=_digest(f"overfit/{system_id}/{seed}"),
                    numerical_receipt_sha256=_digest(f"numerical/{system_id}/{seed}"),
                    metric_id=qualification.PRIMARY_METRIC_ID,
                    primary_numerator=selected_scores[system_id][seed_index],
                    primary_denominator=1000,
                )
            )
    return tuple(rows)


def test_census_matches_execution_schema_and_plan_order() -> None:
    assert qualification.SEED_ORDER == execution_schema.SEEDS == (1729, 2718, 31415)
    assert (
        qualification.SYSTEM_ORDER
        == execution_schema.BASE_SYSTEMS
        == (
            "00",
            "07",
            "08",
        )
    )
    assert dict(qualification.SYSTEM_NAMES) == {
        "00": "mime",
        "07": "early",
        "08": "late",
    }
    assert qualification.PRIMARY_METRIC_ID == (
        "validation_full_gallery_cluster_macro_bidirectional_mean_R1"
    )


def test_collector_restores_system_outer_seed_inner_order() -> None:
    collected = qualification.collect_base_qualification_rows(tuple(reversed(_rows())))
    assert len(collected) == 9
    assert tuple((row.system_id, row.seed) for row in collected) == tuple(
        (system_id, seed)
        for system_id in qualification.SYSTEM_ORDER
        for seed in qualification.SEED_ORDER
    )
    assert collected[0].run_id == "phasepair-run-v2/BASE_TRAIN/1729/00"
    assert collected[-1].run_id == "phasepair-run-v2/BASE_TRAIN/31415/08"
    with pytest.raises(TypeError, match="exact built-in tuple"):
        qualification.collect_base_qualification_rows(list(_rows()))


def test_collector_rejects_duplicate_missing_and_rebound_identity() -> None:
    rows = _rows()
    with pytest.raises(qualification.BaseQualificationError, match="duplicate"):
        qualification.collect_base_qualification_rows(rows[:-1] + (rows[0],))
    with pytest.raises(qualification.BaseQualificationError, match="missing"):
        qualification.collect_base_qualification_rows(rows[:-1])

    wrong_run = replace(rows[0], run_id="phasepair-run-v2/BASE_TRAIN/1729/07")
    with pytest.raises(qualification.BaseQualificationError, match="exact BASE_TRAIN"):
        qualification.collect_base_qualification_rows((wrong_run,) + rows[1:])

    wrong_system = replace(rows[0], system_id="06")
    with pytest.raises(qualification.BaseQualificationError, match="base census"):
        qualification.collect_base_qualification_rows((wrong_system,) + rows[1:])


def test_terminal_must_be_exact_success_before_scientific_gate() -> None:
    rows = _rows()
    failed = replace(rows[0], terminal_success=False)
    with pytest.raises(
        qualification.BaseQualificationError, match="not a successfully"
    ):
        qualification.evaluate_base_qualification((failed,) + rows[1:])
    non_bool = replace(rows[0], terminal_success=1)
    with pytest.raises(TypeError, match="exact built-in bool"):
        qualification.evaluate_base_qualification((non_bool,) + rows[1:])


@pytest.mark.parametrize(
    "field",
    [
        "terminal_execution_receipt_sha256",
        "selected_checkpoint_sha256",
        "full_score_matrix_sha256",
        "evaluator_sha256",
        "optimizer_opt_receipt_sha256",
        "dropout_bd_receipt_sha256",
        "overfit_receipt_sha256",
        "numerical_receipt_sha256",
    ],
)
def test_every_bound_sha_must_be_exact_lowercase_sha256(field: str) -> None:
    rows = _rows()
    mutated = replace(rows[0], **{field: "A" * 64})
    with pytest.raises(qualification.BaseQualificationError, match="lowercase SHA-256"):
        qualification.collect_base_qualification_rows((mutated,) + rows[1:])


def test_shared_global_receipt_bundle_is_bound_without_invented_non_alias_rule() -> (
    None
):
    shared = _digest("shared-global-prerequisite-bundle")
    rows = tuple(
        replace(
            row,
            overfit_receipt_sha256=shared,
            numerical_receipt_sha256=shared,
        )
        for row in _rows()
    )
    result = qualification.evaluate_base_qualification(tuple(reversed(rows)))
    assert result.qualified is True
    assert all(row.overfit_receipt_sha256 == shared for row in result.rows)
    assert all(row.numerical_receipt_sha256 == shared for row in result.rows)


def test_wrong_metric_score_hash_evaluator_or_denominator_fails_closed() -> None:
    rows = _rows()
    wrong_metric = replace(rows[0], metric_id="hard32_r_at_1")
    with pytest.raises(qualification.BaseQualificationError, match="primary metric"):
        qualification.collect_base_qualification_rows((wrong_metric,) + rows[1:])

    wrong_score_hash = replace(rows[0], full_score_matrix_sha256="0" * 63)
    with pytest.raises(qualification.BaseQualificationError, match="lowercase SHA-256"):
        qualification.collect_base_qualification_rows((wrong_score_hash,) + rows[1:])

    wrong_evaluator = replace(rows[0], evaluator_sha256=_digest("other evaluator"))
    with pytest.raises(
        qualification.BaseQualificationError, match="one frozen evaluator"
    ):
        qualification.collect_base_qualification_rows((wrong_evaluator,) + rows[1:])

    wrong_denominator = replace(rows[0], primary_denominator=999)
    with pytest.raises(qualification.BaseQualificationError, match="one denominator"):
        qualification.collect_base_qualification_rows((wrong_denominator,) + rows[1:])


@pytest.mark.parametrize("invalid", [True, float("nan"), float("inf"), 1.0])
def test_primary_metric_rejects_bool_nonfinite_and_all_floats(invalid: object) -> None:
    rows = _rows()
    mutated = replace(rows[0], primary_numerator=invalid)
    with pytest.raises(TypeError, match="exact built-in int"):
        qualification.collect_base_qualification_rows((mutated,) + rows[1:])


def test_primary_metric_rejects_invalid_exact_fraction_bounds() -> None:
    rows = _rows()
    too_large = replace(rows[0], primary_numerator=1001)
    with pytest.raises(qualification.BaseQualificationError, match=r"inside \[0,1\]"):
        qualification.collect_base_qualification_rows((too_large,) + rows[1:])
    zero_denominator = replace(rows[0], primary_denominator=0)
    with pytest.raises(qualification.BaseQualificationError, match="closed interval"):
        qualification.collect_base_qualification_rows((zero_denominator,) + rows[1:])


def test_exact_boundary_minus_point005_qualifies() -> None:
    result = qualification.evaluate_base_qualification(tuple(reversed(_rows())))
    assert result.mime_mean == Fraction(91, 150)
    assert result.early_mean == Fraction(359, 600)
    assert result.late_mean == Fraction(451, 750)
    assert result.same_seed_stronger_comparator_win_count == 2
    assert result.minimum_same_seed_delta == Fraction(-1, 200)
    assert result.mean_strictly_above_early is True
    assert result.mean_strictly_above_late is True
    assert result.at_least_two_of_three_stronger_wins is True
    assert result.minimum_delta_at_least_minus_0_005 is True
    assert result.qualified is True


def test_below_minus_point005_fails_only_threshold_gate() -> None:
    scores = dict(_BOUNDARY_SCORES)
    scores["00"] = (599, 620, 600)
    result = qualification.evaluate_base_qualification(_rows(scores))
    assert result.minimum_same_seed_delta == Fraction(-3, 500)
    assert result.mean_strictly_above_early is True
    assert result.mean_strictly_above_late is True
    assert result.same_seed_stronger_comparator_win_count == 2
    assert result.minimum_delta_at_least_minus_0_005 is False
    assert result.qualified is False


def test_mean_tie_is_not_strictly_above_and_does_not_qualify() -> None:
    scores = {
        "00": (600, 600, 600),
        "07": (600, 600, 600),
        "08": (590, 590, 590),
    }
    result = qualification.evaluate_base_qualification(_rows(scores))
    assert result.mime_mean == result.early_mean
    assert result.mean_strictly_above_early is False
    assert result.qualified is False


def test_same_seed_tie_does_not_count_as_stronger_comparator_win() -> None:
    scores = {
        "00": (600, 610, 620),
        "07": (600, 600, 620),
        "08": (590, 610, 600),
    }
    result = qualification.evaluate_base_qualification(_rows(scores))
    assert result.same_seed_stronger_comparator_win_count == 0
    assert result.at_least_two_of_three_stronger_wins is False
    assert result.qualified is False


def test_exactly_one_stronger_comparator_win_fails_only_two_of_three_gate() -> None:
    scores = {
        "00": (610, 610, 610),
        "07": (600, 610, 615),
        "08": (600, 610, 615),
    }
    result = qualification.evaluate_base_qualification(_rows(scores))
    assert result.mean_strictly_above_early is True
    assert result.mean_strictly_above_late is True
    assert result.same_seed_stronger_comparator_win_count == 1
    assert result.minimum_same_seed_delta == Fraction(-1, 200)
    assert result.minimum_delta_at_least_minus_0_005 is True
    assert result.at_least_two_of_three_stronger_wins is False
    assert result.qualified is False


def test_nonqualification_is_completed_scientific_veto_not_terminal_failure() -> None:
    scores = {
        "00": (570, 570, 570),
        "07": (580, 580, 580),
        "08": (590, 590, 590),
    }
    result = qualification.evaluate_base_qualification(_rows(scores))
    assert result.qualified is False
    raw = qualification.canonical_base_qualification_bytes(result)
    payload = json.loads(raw)
    assert payload["completed_run_count"] == 9
    assert payload["all_terminals_successful"] is True
    assert payload["qualified"] is False
    assert payload["qualification_outcome"] == qualification.NOT_QUALIFIED
    assert payload["qualification_failure_kind"] == "SCIENTIFIC"
    assert payload["execution_authorized"] is False
    assert payload["production"] is False
    assert payload["result_claimed"] is False


def test_canonical_bytes_are_order_independent_authority0_and_hash_bound() -> None:
    canonical = qualification.evaluate_base_qualification(_rows())
    reversed_input = qualification.evaluate_base_qualification(tuple(reversed(_rows())))
    raw = qualification.canonical_base_qualification_bytes(canonical)
    assert raw == qualification.canonical_base_qualification_bytes(reversed_input)
    assert raw.endswith(b"\n") and b"\r" not in raw

    payload = json.loads(raw)
    assert payload["schema"] == "phasepair-base-qualification-semantic-v1"
    assert payload["authority"] == 0
    assert payload["execution_authorized"] is False
    assert payload["production"] is False
    assert payload["result_claimed"] is False
    assert payload["qualified"] is True
    assert payload["qualification_outcome"] == qualification.QUALIFIED
    assert payload["qualification_failure_kind"] == qualification.NOT_APPLICABLE
    assert payload["row_order"] == "system_outer_seed_inner"
    assert [(row["system_id"], row["seed"]) for row in payload["rows"]] == [
        (system_id, seed)
        for system_id in qualification.SYSTEM_ORDER
        for seed in qualification.SEED_ORDER
    ]
    assert payload["rows_sha256"] == canonical.rows_sha256
    assert (
        qualification.base_qualification_sha256(canonical)
        == hashlib.sha256(raw).hexdigest()
    )
    assert (len(raw), hashlib.sha256(raw).hexdigest(), canonical.rows_sha256) == (
        10154,
        "fa4efd2cc88c1373ae3cbb65673cf9927bfbeb141838b36c93f6d0a5b55bb242",
        "cde343bc71dc326dce83e42b1a6ddfd918165c7941c3b349dc51fa78450b7dee",
    )
    assert raw == (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
        + b"\n"
    )


def test_export_rejects_row_score_hash_and_derived_gate_rebinding() -> None:
    result = qualification.evaluate_base_qualification(_rows())
    rebound_row = replace(
        result.rows[0],
        full_score_matrix_sha256=_digest("valid-but-rebound-score-matrix"),
    )
    rebound_rows = replace(
        result,
        rows=(rebound_row,) + result.rows[1:],
    )
    with pytest.raises(
        qualification.BaseQualificationError, match="commitment or derived gate"
    ):
        qualification.canonical_base_qualification_bytes(rebound_rows)

    forged_verdict = replace(result, qualified=False)
    with pytest.raises(
        qualification.BaseQualificationError, match="commitment or derived gate"
    ):
        qualification.base_qualification_sha256(forged_verdict)

    forged_commitment = replace(result, rows_sha256=_digest("forged rows"))
    with pytest.raises(
        qualification.BaseQualificationError, match="commitment or derived gate"
    ):
        qualification.canonical_base_qualification_bytes(forged_commitment)


def test_module_remains_authority0_and_data_free() -> None:
    assert qualification.AUTHORITY == 0
    assert qualification.PRODUCTION is False
    assert qualification.EXECUTION_AUTHORIZED is False
    assert qualification.RESULT_CLAIMED is False
    assert qualification.STATUS == "DATA_FREE_BASE_QUALIFICATION_AUTHORITY0_NO_RESULT"


def test_public_global_and_stdlib_rebinding_cannot_change_closed_semantics() -> None:
    rows = _rows()
    result = qualification.evaluate_base_qualification(rows)
    raw = qualification.canonical_base_qualification_bytes(result)
    digest = qualification.base_qualification_sha256(result)
    names = (
        "AUTHORITY",
        "PRODUCTION",
        "EXECUTION_AUTHORIZED",
        "RESULT_CLAIMED",
        "STATUS",
        "SEED_ORDER",
        "SYSTEM_ORDER",
        "SYSTEM_NAMES",
        "PRIMARY_METRIC_ID",
        "QUALIFIED",
        "NOT_QUALIFIED",
        "NOT_APPLICABLE",
        "collect_base_qualification_rows",
        "_canonical_json_bytes",
        "_row_payload",
    )
    originals = {name: getattr(qualification, name) for name in names}
    original_dumps = qualification.json.dumps
    original_sha256 = qualification.hashlib.sha256
    try:
        qualification.AUTHORITY = 99
        qualification.PRODUCTION = True
        qualification.EXECUTION_AUTHORIZED = True
        qualification.RESULT_CLAIMED = True
        qualification.STATUS = "FORGED_RESULT"
        qualification.SEED_ORDER = (1729,)
        qualification.SYSTEM_ORDER = ("06",)
        qualification.SYSTEM_NAMES = {"00": "forged"}
        qualification.PRIMARY_METRIC_ID = "hard32"
        qualification.QUALIFIED = qualification.NOT_QUALIFIED
        qualification.NOT_QUALIFIED = "QUALIFIED"
        qualification.NOT_APPLICABLE = "APPLICABLE"
        qualification.collect_base_qualification_rows = lambda _rows: ()
        qualification._canonical_json_bytes = lambda _value: b"FORGED\n"
        qualification._row_payload = lambda _row: {"forged": True}
        qualification.json.dumps = lambda *_args, **_kwargs: "FORGED"
        qualification.hashlib.sha256 = lambda _value=b"": (_ for _ in ()).throw(
            AssertionError("rebound sha256 called")
        )

        rebuilt = qualification.evaluate_base_qualification(rows)
        assert rebuilt == result
        assert qualification.canonical_base_qualification_bytes(rebuilt) == raw
        assert qualification.base_qualification_sha256(rebuilt) == digest
    finally:
        for name, value in originals.items():
            setattr(qualification, name, value)
        qualification.json.dumps = original_dumps
        qualification.hashlib.sha256 = original_sha256
