from __future__ import annotations

import hashlib
import json
from pathlib import Path
from weakref import WeakKeyDictionary

import pytest

from phasepair_core import execution_schema as execution


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _completion_payload(system: str, seed: int) -> dict[str, object]:
    run_id = f"phasepair-run-v2/BASE_TRAIN/{seed}/{system}"
    return {
        "architecture": execution.BASE_ARCHITECTURES[system],
        "evaluator_sha256": _digest("one-evaluator"),
        "example_count": 30_000,
        "log_sha256": _digest(f"log/{system}/{seed}"),
        "model_final_state_sha256": _digest(f"model/{system}/{seed}"),
        "optimizer_final_state_sha256": _digest(f"optimizer/{system}/{seed}"),
        "optimizer_step_count": 300,
        "run_id": run_id,
        "run_input_sha256": _digest(f"input/{system}/{seed}"),
        "schema": "phasepair-base-train-completion-v2",
        "seed": seed,
        "selected_checkpoint_sha256": _digest(f"checkpoint/{system}/{seed}"),
        "selected_completed_epoch": 13,
        "selected_epoch_index": 12,
        "status": "EXECUTION_COMPLETE_NONTERMINAL_AWAITING_QUALIFICATION",
        "system_id": system,
        "validation_metric_sha256": _digest(f"metric/{system}/{seed}"),
    }


def _completions() -> tuple[execution.BaseTrainCompletion, ...]:
    return tuple(
        execution.validate_base_train_completion(
            execution.canonical_json_bytes(_completion_payload(system, seed))
        )
        for system in execution.BASE_SYSTEMS
        for seed in execution.SEEDS
    )


_GOOD_NUMERATORS = {
    "00": (600, 610, 590),
    "07": (580, 570, 580),
    "08": (590, 600, 580),
}


def _scores(
    completions: tuple[execution.BaseTrainCompletion, ...],
    numerators: dict[str, tuple[int, int, int]] = _GOOD_NUMERATORS,
) -> tuple[execution.BaseValidationScore, ...]:
    rows: list[execution.BaseValidationScore] = []
    for completion in completions:
        system = completion.system_id
        seed_index = execution.SEEDS.index(completion.seed)
        rows.append(
            execution.BaseValidationScore(
                system,
                completion.seed,
                numerators[system][seed_index],
                1000,
                execution.base_train_completion_sha256(completion),
            )
        )
    return tuple(rows)


def _terminal_payload(
    completion: execution.BaseTrainCompletion,
    qualification: execution.BaseQualification,
    *,
    outcome: str = "SUCCESS",
    cache_override: str | None = None,
) -> dict[str, object]:
    completion_payload = json.loads(
        execution.base_train_completion_bytes(completion).decode("utf-8")
    )
    qualified = qualification.verdict == execution.QUALIFIED
    cache = (
        _digest(f"cache/{completion.seed}")
        if qualified and completion.system_id == "00" and outcome == "SUCCESS"
        else execution.NOT_APPLICABLE
    )
    if cache_override is not None:
        cache = cache_override
    selected = (
        completion_payload["selected_checkpoint_sha256"]
        if outcome == "SUCCESS"
        else execution.NOT_APPLICABLE
    )
    return {
        "authority": 0,
        "completion_sha256": execution.base_train_completion_sha256(completion),
        "production": False,
        "qualification_sha256": execution.base_qualification_sha256(qualification),
        "result_claimed": False,
        "run_id": completion.run_id,
        "run_input_sha256": completion_payload["run_input_sha256"],
        "schema": "phasepair-base-terminal-receipt-v2",
        "seed": completion.seed,
        "selected_checkpoint_sha256": selected,
        "status": "DATA_FREE_TERMINAL_SCHEMA_VALIDATED_NO_RESULT",
        "system_id": completion.system_id,
        "terminal_outcome": outcome,
        "training_frozen_base_cache_sha256": cache,
    }


def _terminals(
    completions: tuple[execution.BaseTrainCompletion, ...],
    qualification: execution.BaseQualification,
) -> tuple[execution.BaseTerminalReceipt, ...]:
    return tuple(
        execution.validate_base_terminal_receipt(
            execution.canonical_json_bytes(
                _terminal_payload(completion, qualification)
            ),
            completion,
            qualification,
        )
        for completion in completions
    )


def _cache_payload(
    terminal: execution.BaseTerminalReceipt,
) -> dict[str, object]:
    terminal_payload = json.loads(
        execution.base_terminal_receipt_bytes(terminal).decode("utf-8")
    )
    return {
        "authority": 0,
        "base_terminal_sha256": execution.base_terminal_receipt_sha256(terminal),
        "cache_sha256": terminal_payload["training_frozen_base_cache_sha256"],
        "production": False,
        "result_claimed": False,
        "run_id": terminal.run_id,
        "schema": "phasepair-mime-cache-receipt-v2",
        "seed": terminal_payload["seed"],
        "status": "DATA_FREE_CACHE_SCHEMA_VALIDATED_NO_RESULT",
        "system_id": "00",
    }


def _caches(
    terminals: tuple[execution.BaseTerminalReceipt, ...],
) -> tuple[execution.MimeCacheReceipt, ...]:
    return tuple(
        execution.validate_mime_cache_receipt(
            execution.canonical_json_bytes(_cache_payload(terminal)), terminal
        )
        for terminal in terminals[:3]
    )


def test_canonical_v2_census_is_exact_authority_zero_and_lf() -> None:
    assert execution.base_run_ids() == tuple(
        f"phasepair-run-v2/BASE_TRAIN/{seed}/{system}"
        for system in ("00", "07", "08")
        for seed in (1729, 2718, 31415)
    )
    assert execution.residual_run_ids() == tuple(
        f"phasepair-run-v2/RESIDUAL_HEAD_TRAIN/{seed}/{system}"
        for system in ("01", "02", "03", "04", "05", "06")
        for seed in (1729, 2718, 31415)
    )
    raw = execution.canonical_execution_census_bytes()
    assert raw.endswith(b"\n") and b"\r" not in raw
    assert execution.validate_execution_census(raw) == raw
    payload = json.loads(raw)
    assert payload["base_run_count"] == 9
    assert payload["residual_run_count"] == 18
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["execution_authorized"] is False
    assert payload["result_claimed"] is False


def test_legacy_v1_old_tmr06_and_fifteen_residual_census_are_rejected() -> None:
    for run_id in (
        "phasepair-run-v1/BASE_TRAIN/1729/00",
        "phasepair-run-v2/BASE_TRAIN/1729/06",
        "phasepair-run-v2/RESIDUAL_HEAD_TRAIN/1729/07",
        "phasepair-run-v2/RESIDUAL_HEAD_TRAIN/1729/08",
    ):
        with pytest.raises(
            execution.ExecutionSchemaError,
            match="TRAINING_EXECUTION_IDENTITY_VERSION_FAIL",
        ):
            execution.parse_run_id(run_id)

    census = json.loads(execution.canonical_execution_census_bytes())
    census["residual_run_ids"] = census["residual_run_ids"][:15]
    census["residual_run_count"] = 15
    with pytest.raises(
        execution.ExecutionSchemaError,
        match="TRAINING_EXECUTION_IDENTITY_VERSION_FAIL",
    ):
        execution.validate_execution_census(execution.canonical_json_bytes(census))


def test_completion_closed_schema_and_canonical_json_mutants() -> None:
    payload = _completion_payload("00", 1729)
    raw = execution.canonical_json_bytes(payload)
    completion = execution.validate_base_train_completion(raw)
    assert execution.base_train_completion_bytes(completion) == raw

    extra = dict(payload)
    extra["claim"] = "RESULT"
    with pytest.raises(execution.ExecutionSchemaError, match="keys are not closed"):
        execution.validate_base_train_completion(execution.canonical_json_bytes(extra))

    missing = dict(payload)
    del missing["log_sha256"]
    with pytest.raises(execution.ExecutionSchemaError, match="keys are not closed"):
        execution.validate_base_train_completion(execution.canonical_json_bytes(missing))

    bool_step = dict(payload)
    bool_step["optimizer_step_count"] = True
    with pytest.raises(TypeError, match="exact built-in int"):
        execution.validate_base_train_completion(
            execution.canonical_json_bytes(bool_step)
        )

    wrong_epoch = dict(payload)
    wrong_epoch["selected_completed_epoch"] = 14
    with pytest.raises(execution.ExecutionSchemaError, match="epoch mismatch"):
        execution.validate_base_train_completion(
            execution.canonical_json_bytes(wrong_epoch)
        )

    noncanonical = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
    with pytest.raises(execution.ExecutionSchemaError, match="not canonical"):
        execution.validate_base_train_completion(noncanonical)

    duplicate = raw[:-2] + b',"seed":1729}\n'
    with pytest.raises(execution.ExecutionSchemaError, match="duplicate JSON key"):
        execution.validate_base_train_completion(duplicate)

    bom = b"\xef\xbb\xbf" + raw
    with pytest.raises(execution.ExecutionSchemaError, match="BOM"):
        execution.validate_base_train_completion(bom)


def test_closed_artifacts_reject_direct_construction_and_post_issue_tamper() -> None:
    with pytest.raises(execution.ExecutionSchemaError, match="closed validator"):
        execution.BaseTrainCompletion()
    completion = _completions()[0]
    original = execution.base_train_completion_bytes(completion)
    object.__setattr__(completion, "_canonical", original + b" ")
    with pytest.raises(execution.ExecutionSchemaError, match="changed after issuance"):
        execution.base_train_completion_bytes(completion)


def test_artifact_rebinding_cannot_admit_equal_never_issued_objects(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completions = _completions()
    scores = _scores(completions)
    qualification = execution.qualify_nine_base_completions(completions, scores)
    terminals = _terminals(completions, qualification)
    caches = _caches(terminals)
    lease = execution._issue_qualified_mime_base_lease(
        qualification, caches, terminals
    )

    issued = (
        (
            "BaseTrainCompletion",
            completions[0],
            execution.base_train_completion_bytes,
            execution._COMPLETIONS,
        ),
        (
            "BaseQualification",
            qualification,
            execution.base_qualification_bytes,
            execution._QUALIFICATIONS,
        ),
        (
            "BaseTerminalReceipt",
            terminals[0],
            execution.base_terminal_receipt_bytes,
            execution._TERMINALS,
        ),
        (
            "MimeCacheReceipt",
            caches[0],
            execution.mime_cache_receipt_bytes,
            execution._CACHES,
        ),
        (
            "QualifiedMimeBaseLease",
            lease,
            execution.qualified_mime_base_lease_bytes,
            execution._LEASES,
        ),
    )
    artifact_types = {name: type(value) for name, value, _serialize, _store in issued}
    original_bytes = {
        name: serialize(value) for name, value, serialize, _store in issued
    }

    class NeverIssuedEqualArtifact:
        __slots__ = ("_canonical", "_target_hash", "__weakref__")

        def __init__(self, canonical: bytes, target_hash: int) -> None:
            self._canonical = canonical
            self._target_hash = target_hash

        def __hash__(self) -> int:
            return self._target_hash

        def __eq__(self, _other: object) -> bool:
            return True

    impostors: list[tuple[object, object]] = []
    for name, value, serialize, store in issued:
        impostor = NeverIssuedEqualArtifact(original_bytes[name], hash(value))
        assert store.get(impostor) == original_bytes[name]
        impostors.append((serialize, impostor))
        monkeypatch.setattr(execution, name, NeverIssuedEqualArtifact)

    original_stores = {
        "_COMPLETIONS": execution._COMPLETIONS,
        "_QUALIFICATIONS": execution._QUALIFICATIONS,
        "_TERMINALS": execution._TERMINALS,
        "_CACHES": execution._CACHES,
        "_LEASES": execution._LEASES,
    }
    for name in original_stores:
        monkeypatch.setattr(execution, name, WeakKeyDictionary())
    monkeypatch.setattr(
        execution,
        "_mint",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rebound mint called")
        ),
    )
    monkeypatch.setattr(
        execution,
        "_issued_payload",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rebound issued-payload called")
        ),
    )
    for name in (
        "_completion_payload",
        "_qualification_payload",
        "_terminal_payload",
        "_cache_payload",
        "_lease_payload",
    ):
        monkeypatch.setattr(
            execution,
            name,
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                AssertionError("rebound artifact payload called")
            ),
        )

    for serialize, impostor in impostors:
        with pytest.raises(TypeError, match="wrong exact type"):
            serialize(impostor)

    completion = execution.validate_base_train_completion(
        original_bytes["BaseTrainCompletion"]
    )
    rebuilt_qualification = execution.qualify_nine_base_completions(
        completions, scores
    )
    terminal = execution.validate_base_terminal_receipt(
        original_bytes["BaseTerminalReceipt"], completion, rebuilt_qualification
    )
    cache = execution.validate_mime_cache_receipt(
        original_bytes["MimeCacheReceipt"], terminal
    )
    rebuilt_lease = execution._issue_qualified_mime_base_lease(
        rebuilt_qualification,
        (cache, *caches[1:]),
        (terminal, *terminals[1:]),
    )
    rebuilt = {
        "BaseTrainCompletion": completion,
        "BaseQualification": rebuilt_qualification,
        "BaseTerminalReceipt": terminal,
        "MimeCacheReceipt": cache,
        "QualifiedMimeBaseLease": rebuilt_lease,
    }
    for name, value in rebuilt.items():
        assert type(value) is artifact_types[name]
        assert original_stores[
            {
                "BaseTrainCompletion": "_COMPLETIONS",
                "BaseQualification": "_QUALIFICATIONS",
                "BaseTerminalReceipt": "_TERMINALS",
                "MimeCacheReceipt": "_CACHES",
                "QualifiedMimeBaseLease": "_LEASES",
            }[name]
        ].get(value) is not None

    assert completion.run_id == completions[0].run_id
    assert rebuilt_qualification.verdict == "QUALIFIED"
    assert terminal.run_id == terminals[0].run_id
    assert cache.seed == caches[0].seed
    assert rebuilt_lease.authority == 0
    assert rebuilt_lease.execution_authorized is False


def test_nine_completion_gate_qualifies_only_in_exact_order() -> None:
    completions = _completions()
    qualification = execution.qualify_nine_base_completions(
        completions, _scores(completions)
    )
    assert qualification.verdict == execution.QUALIFIED
    raw = execution.base_qualification_bytes(qualification)
    payload = json.loads(raw)
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["result_claimed"] is False
    assert payload["status"].endswith("NO_RESULT")
    assert payload["same_seed_win_count"] == 3

    with pytest.raises(execution.ExecutionSchemaError, match="canonical"):
        execution.qualify_nine_base_completions(
            tuple(reversed(completions)), tuple(reversed(_scores(completions)))
        )
    with pytest.raises(execution.ExecutionSchemaError, match="exactly nine"):
        execution.qualify_nine_base_completions(
            completions[:8], _scores(completions)[:8]
        )
    duplicated = completions[:-1] + (completions[-2],)
    with pytest.raises(execution.ExecutionSchemaError, match="canonical"):
        execution.qualify_nine_base_completions(
            duplicated, _scores(completions)
        )


def test_gate_not_qualified_for_mean_win_count_or_minimum_delta_failure() -> None:
    completions = _completions()

    mean_fail = {
        "00": (570, 570, 570),
        "07": (580, 580, 580),
        "08": (590, 590, 590),
    }
    assert (
        execution.qualify_nine_base_completions(
            completions, _scores(completions, mean_fail)
        ).verdict
        == execution.NOT_QUALIFIED
    )

    win_count_fail = {
        "00": (620, 590, 590),
        "07": (600, 600, 600),
        "08": (610, 600, 600),
    }
    assert (
        execution.qualify_nine_base_completions(
            completions, _scores(completions, win_count_fail)
        ).verdict
        == execution.NOT_QUALIFIED
    )

    min_delta_fail = {
        "00": (700, 700, 590),
        "07": (600, 600, 600),
        "08": (600, 600, 600),
    }
    assert (
        execution.qualify_nine_base_completions(
            completions, _scores(completions, min_delta_fail)
        ).verdict
        == execution.NOT_QUALIFIED
    )


def test_score_rows_require_exact_types_digest_order_and_common_denominator() -> None:
    completions = _completions()
    scores = list(_scores(completions))
    score = scores[0]
    with pytest.raises(TypeError, match="exact built-in int"):
        execution.BaseValidationScore(
            score.system_id,
            True,
            score.numerator,
            score.denominator,
            score.completion_sha256,
        )

    scores[0] = execution.BaseValidationScore(
        score.system_id,
        score.seed,
        score.numerator,
        score.denominator,
        _digest("wrong-completion"),
    )
    with pytest.raises(execution.ExecutionSchemaError, match="digest mismatch"):
        execution.qualify_nine_base_completions(completions, tuple(scores))

    scores = list(_scores(completions))
    original = scores[-1]
    scores[-1] = execution.BaseValidationScore(
        original.system_id,
        original.seed,
        original.numerator,
        2000,
        original.completion_sha256,
    )
    with pytest.raises(execution.ExecutionSchemaError, match="one denominator"):
        execution.qualify_nine_base_completions(completions, tuple(scores))


def test_terminal_cache_applicability_for_qualified_and_unqualified_gates() -> None:
    completions = _completions()
    qualified = execution.qualify_nine_base_completions(
        completions, _scores(completions)
    )

    mime_payload = _terminal_payload(completions[0], qualified)
    execution.validate_base_terminal_receipt(
        execution.canonical_json_bytes(mime_payload), completions[0], qualified
    )
    early_payload = _terminal_payload(completions[3], qualified)
    execution.validate_base_terminal_receipt(
        execution.canonical_json_bytes(early_payload), completions[3], qualified
    )

    bad_mime = _terminal_payload(
        completions[0], qualified, cache_override=execution.NOT_APPLICABLE
    )
    with pytest.raises(execution.ExecutionSchemaError, match="lowercase SHA-256"):
        execution.validate_base_terminal_receipt(
            execution.canonical_json_bytes(bad_mime), completions[0], qualified
        )
    bad_early = _terminal_payload(
        completions[3], qualified, cache_override=_digest("illegal-early-cache")
    )
    with pytest.raises(execution.ExecutionSchemaError, match="not applicable"):
        execution.validate_base_terminal_receipt(
            execution.canonical_json_bytes(bad_early), completions[3], qualified
        )

    not_qualified = execution.qualify_nine_base_completions(
        completions,
        _scores(
            completions,
            {
                "00": (570, 570, 570),
                "07": (580, 580, 580),
                "08": (590, 590, 590),
            },
        ),
    )
    unqualified_mime = _terminal_payload(completions[0], not_qualified)
    execution.validate_base_terminal_receipt(
        execution.canonical_json_bytes(unqualified_mime),
        completions[0],
        not_qualified,
    )
    bad_unqualified = _terminal_payload(
        completions[0],
        not_qualified,
        cache_override=_digest("illegal-unqualified-cache"),
    )
    with pytest.raises(execution.ExecutionSchemaError, match="not applicable"):
        execution.validate_base_terminal_receipt(
            execution.canonical_json_bytes(bad_unqualified),
            completions[0],
            not_qualified,
        )


def test_terminal_and_lease_reject_same_run_alternate_completion_crossbind() -> None:
    completions = _completions()
    qualification = execution.qualify_nine_base_completions(
        completions, _scores(completions)
    )

    alternate_payload = _completion_payload("00", 1729)
    alternate_payload["selected_checkpoint_sha256"] = _digest(
        "alternate-checkpoint/00/1729"
    )
    alternate_payload["validation_metric_sha256"] = _digest(
        "alternate-metric/00/1729"
    )
    alternate_payload["model_final_state_sha256"] = _digest(
        "alternate-model/00/1729"
    )
    alternate = execution.validate_base_train_completion(
        execution.canonical_json_bytes(alternate_payload)
    )
    assert alternate.run_id == completions[0].run_id
    assert (
        execution.base_train_completion_sha256(alternate)
        != execution.base_train_completion_sha256(completions[0])
    )

    crossbound_raw = execution.canonical_json_bytes(
        _terminal_payload(alternate, qualification)
    )
    with pytest.raises(
        execution.ExecutionSchemaError,
        match="qualification-bound original",
    ):
        execution.validate_base_terminal_receipt(
            crossbound_raw, alternate, qualification
        )

    terminals = _terminals(completions, qualification)
    caches = _caches(terminals)
    internally_crossbound = execution._mint(
        execution.BaseTerminalReceipt,
        crossbound_raw,
        execution._TERMINALS,
    )
    with pytest.raises(
        execution.ExecutionSchemaError,
        match="qualification-bound original",
    ):
        execution._issue_qualified_mime_base_lease(
            qualification,
            caches,
            (internally_crossbound, *terminals[1:]),
        )


def test_private_lease_issuer_requires_qualified_three_caches_nine_successes() -> None:
    completions = _completions()
    qualification = execution.qualify_nine_base_completions(
        completions, _scores(completions)
    )
    terminals = _terminals(completions, qualification)
    caches = _caches(terminals)

    with pytest.raises(execution.ExecutionSchemaError, match="closed validator"):
        execution.QualifiedMimeBaseLease()

    lease = execution._issue_qualified_mime_base_lease(
        qualification, caches, terminals
    )
    assert lease.authority == 0
    assert lease.execution_authorized is False
    lease_payload = json.loads(execution.qualified_mime_base_lease_bytes(lease))
    assert lease_payload["production"] is False
    assert lease_payload["result_claimed"] is False
    assert lease_payload["status"].endswith("NO_RESULT")
    assert len(lease_payload["cache_sha256s"]) == 3
    assert len(lease_payload["terminal_sha256s"]) == 9

    with pytest.raises(execution.ExecutionSchemaError, match="exactly three"):
        execution._issue_qualified_mime_base_lease(
            qualification, caches[:2], terminals
        )
    with pytest.raises(execution.ExecutionSchemaError, match="seed-ascending"):
        execution._issue_qualified_mime_base_lease(
            qualification, tuple(reversed(caches)), terminals
        )
    with pytest.raises(execution.ExecutionSchemaError, match="exactly nine"):
        execution._issue_qualified_mime_base_lease(
            qualification, caches, terminals[:8]
        )

    not_qualified = execution.qualify_nine_base_completions(
        completions,
        _scores(
            completions,
            {
                "00": (570, 570, 570),
                "07": (580, 580, 580),
                "08": (590, 590, 590),
            },
        ),
    )
    with pytest.raises(execution.ExecutionSchemaError, match="verdict QUALIFIED"):
        execution._issue_qualified_mime_base_lease(
            not_qualified, caches, terminals
        )


def test_cache_receipt_rejects_wrong_terminal_content_and_non_mime_owner() -> None:
    completions = _completions()
    qualification = execution.qualify_nine_base_completions(
        completions, _scores(completions)
    )
    terminals = _terminals(completions, qualification)

    payload = _cache_payload(terminals[0])
    payload["cache_sha256"] = _digest("wrong-cache")
    with pytest.raises(execution.ExecutionSchemaError, match="content digest"):
        execution.validate_mime_cache_receipt(
            execution.canonical_json_bytes(payload), terminals[0]
        )

    early_payload = dict(_cache_payload(terminals[0]))
    early_payload["run_id"] = terminals[3].run_id
    early_payload["seed"] = completions[3].seed
    early_payload["base_terminal_sha256"] = execution.base_terminal_receipt_sha256(
        terminals[3]
    )
    with pytest.raises(execution.ExecutionSchemaError, match="only MIME"):
        execution.validate_mime_cache_receipt(
            execution.canonical_json_bytes(early_payload), terminals[3]
        )


def test_module_is_stdlib_data_free_and_contains_no_training_or_checkpoint_io() -> None:
    source = (
        Path(__file__).parents[1] / "src" / "phasepair_core" / "execution_schema.py"
    ).read_text(encoding="utf-8")
    assert "import torch" not in source
    assert "torch.optim" not in source
    assert "optimizer.step(" not in source
    assert "loss.backward(" not in source
    assert "open(" not in source
    assert "os.replace" not in source
    assert execution.AUTHORITY == 0
    assert execution.PRODUCTION is False
    assert execution.EXECUTION_AUTHORIZED is False
    assert execution.RESULT_CLAIMED is False


def test_public_global_and_stdlib_rebinding_cannot_change_closed_semantics() -> None:
    completions = _completions()
    scores = _scores(completions)
    qualification = execution.qualify_nine_base_completions(completions, scores)
    terminals = _terminals(completions, qualification)
    caches = _caches(terminals)
    completion_raw = execution.base_train_completion_bytes(completions[0])
    terminal_raw = execution.base_terminal_receipt_bytes(terminals[0])
    cache_raw = execution.mime_cache_receipt_bytes(caches[0])
    census_raw = execution.canonical_execution_census_bytes()
    qualification_raw = execution.base_qualification_bytes(qualification)

    names = (
        "SEEDS",
        "BASE_SYSTEMS",
        "RESIDUAL_SYSTEMS",
        "BASE_ARCHITECTURES",
        "QUALIFIED",
        "NOT_QUALIFIED",
        "NOT_APPLICABLE",
        "STATUS",
        "base_run_ids",
        "residual_run_ids",
        "parse_run_id",
        "canonical_json_bytes",
        "BaseValidationScore",
    )
    originals = {name: getattr(execution, name) for name in names}
    original_sha256 = execution.hashlib.sha256
    original_dumps = execution.json.dumps
    original_loads = execution.json.loads
    try:
        execution.SEEDS = (1729,)
        execution.BASE_SYSTEMS = ("06",)
        execution.RESIDUAL_SYSTEMS = ("01", "02", "03", "04", "05")
        execution.BASE_ARCHITECTURES = {"00": "wrong"}
        execution.QUALIFIED = "NOT_QUALIFIED"
        execution.NOT_QUALIFIED = "QUALIFIED"
        execution.NOT_APPLICABLE = _digest("fake-not-applicable")
        execution.STATUS = "RESULT"
        execution.base_run_ids = lambda: ("phasepair-run-v1/BAD",)
        execution.residual_run_ids = lambda: ()
        execution.parse_run_id = lambda _value: ("BAD", 0, "06")
        execution.canonical_json_bytes = lambda _value: b"FORGED\n"
        execution.BaseValidationScore = object
        execution.hashlib.sha256 = lambda _value=b"": (_ for _ in ()).throw(
            AssertionError("rebound sha256 called")
        )
        execution.json.dumps = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rebound dumps called")
        )
        execution.json.loads = lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("rebound loads called")
        )

        assert execution.canonical_execution_census_bytes() == census_raw
        assert execution.validate_execution_census(census_raw) == census_raw
        rebuilt_completion = execution.validate_base_train_completion(completion_raw)
        rebuilt_qualification = execution.qualify_nine_base_completions(
            completions, scores
        )
        assert (
            execution.base_qualification_bytes(rebuilt_qualification)
            == qualification_raw
        )
        rebuilt_terminal = execution.validate_base_terminal_receipt(
            terminal_raw, rebuilt_completion, rebuilt_qualification
        )
        rebuilt_cache = execution.validate_mime_cache_receipt(
            cache_raw, rebuilt_terminal
        )
        lease = execution._issue_qualified_mime_base_lease(
            rebuilt_qualification,
            (rebuilt_cache, *caches[1:]),
            (rebuilt_terminal, *terminals[1:]),
        )
        assert lease.authority == 0
        assert lease.execution_authorized is False
    finally:
        for name, value in originals.items():
            setattr(execution, name, value)
        execution.hashlib.sha256 = original_sha256
        execution.json.dumps = original_dumps
        execution.json.loads = original_loads
