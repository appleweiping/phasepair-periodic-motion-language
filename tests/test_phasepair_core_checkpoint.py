from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import pytest

from phasepair_core import checkpoint


def _digest(label: str) -> bytes:
    return hashlib.sha256(label.encode("ascii")).digest()


def _bindings(**overrides: object) -> checkpoint.CheckpointBindings:
    values: dict[str, object] = {
        "run_id": "phasepair-synthetic-run-001",
        "attempt_id": "attempt-001",
        "system_id": "00-synthetic",
        "seed": 1729,
        "epoch_index": 0,
        "global_step": 2,
        "source_manifest_sha256": _digest("source"),
        "data_manifest_sha256": _digest("data"),
        "split_manifest_sha256": _digest("split"),
        "environment_manifest_sha256": _digest("environment"),
        "code_manifest_sha256": _digest("code"),
        "model_manifest_sha256": _digest("model"),
        "text_runtime_manifest_sha256": _digest("text"),
        "optimizer_manifest_sha256": _digest("optimizer"),
        "sampler_state_sha256": _digest("payload/sampler_state"),
        "dropout_state_sha256": _digest("payload/dropout_state"),
        "validation_state_sha256": _digest("payload/validation_state"),
        "parent_checkpoint_sha256": None,
    }
    values.update(overrides)
    return checkpoint.CheckpointBindings(**values)


def _sections() -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (name, f"payload/{name}".encode("ascii"))
        for name in checkpoint.REQUIRED_SECTIONS
    )


def test_canonical_roundtrip_has_exact_closed_authority_zero_manifest() -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    raw = checkpoint.canonical_checkpoint_bytes(value)
    assert raw.startswith(checkpoint.MAGIC)
    assert checkpoint.checkpoint_sha256(value) == hashlib.sha256(raw).digest()
    validated = checkpoint.validate_cpu_synthetic_checkpoint_bytes(
        raw,
        expected_bindings=bindings,
    )
    assert checkpoint.canonical_checkpoint_bytes(validated) == raw
    assert checkpoint.checkpoint_sections(validated) == _sections()

    prefix = len(checkpoint.MAGIC)
    manifest_length = int.from_bytes(raw[prefix : prefix + 8], "big")
    manifest = raw[prefix + 8 : prefix + 8 + manifest_length]
    payload = json.loads(manifest)
    assert payload["authority"] == 0
    assert payload["production"] is False
    assert payload["result_claimed"] is False
    assert payload["status"] == checkpoint.STATUS
    assert [row["name"] for row in payload["sections"]] == list(
        checkpoint.REQUIRED_SECTIONS
    )


def test_tamper_truncate_extra_and_crossbinding_fail_closed() -> None:
    bindings = _bindings()
    raw = checkpoint.canonical_checkpoint_bytes(
        checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    )
    mutants = (
        raw[:-1],
        raw + b"x",
        raw[: len(raw) // 2] + bytes([raw[len(raw) // 2] ^ 1]) + raw[len(raw) // 2 + 1 :],
    )
    for mutant in mutants:
        with pytest.raises(checkpoint.CheckpointContractError):
            checkpoint.validate_cpu_synthetic_checkpoint_bytes(
                mutant,
                expected_bindings=bindings,
            )
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.validate_cpu_synthetic_checkpoint_bytes(
            raw,
            expected_bindings=_bindings(global_step=3),
        )


def test_atomic_write_read_and_no_overwrite(tmp_path: Path) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    target = (tmp_path / "checkpoint.ppckpt").absolute()
    digest = checkpoint.write_checkpoint_atomic(value, target)
    assert digest == checkpoint.checkpoint_sha256(value)
    assert target.read_bytes() == checkpoint.canonical_checkpoint_bytes(value)
    loaded = checkpoint.read_checkpoint(target, expected_bindings=bindings)
    assert checkpoint.checkpoint_sha256(loaded) == digest
    assert not tuple(tmp_path.glob(".*.tmp"))
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.write_checkpoint_atomic(value, target)


def test_hardlink_and_symlink_are_rejected_where_supported(tmp_path: Path) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    target = (tmp_path / "checkpoint.ppckpt").absolute()
    checkpoint.write_checkpoint_atomic(value, target)

    hardlink = (tmp_path / "hardlink.ppckpt").absolute()
    os.link(target, hardlink)
    with pytest.raises(checkpoint.CheckpointPathError):
        checkpoint.read_checkpoint(target, expected_bindings=bindings)
    hardlink.unlink()

    symlink = (tmp_path / "symlink.ppckpt").absolute()
    try:
        symlink.symlink_to(target)
    except OSError:
        pytest.skip("current account cannot create symlinks")
    with pytest.raises(checkpoint.CheckpointPathError):
        checkpoint.read_checkpoint(symlink, expected_bindings=bindings)


def test_section_census_order_and_types_are_exact() -> None:
    bindings = _bindings()
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections()[:-1])
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.build_cpu_synthetic_checkpoint(bindings, tuple(reversed(_sections())))
    bad = list(_sections())
    bad[0] = (bad[0][0], bytearray(bad[0][1]))
    with pytest.raises(TypeError):
        checkpoint.build_cpu_synthetic_checkpoint(bindings, tuple(bad))
    with pytest.raises(TypeError):
        checkpoint.build_cpu_synthetic_checkpoint(bindings, list(_sections()))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", True),
        ("epoch_index", 30),
        ("global_step", 1.0),
        ("run_id", "contains space"),
        ("code_manifest_sha256", b"x" * 31),
        ("parent_checkpoint_sha256", bytearray(b"x" * 32)),
    ],
)
def test_binding_types_and_ranges_are_closed(field: str, value: object) -> None:
    with pytest.raises((TypeError, checkpoint.CheckpointContractError)):
        _bindings(**{field: value})


def test_direct_validated_checkpoint_forgery_is_rejected() -> None:
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.ValidatedCheckpoint()
    forged = object.__new__(checkpoint.ValidatedCheckpoint)
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.canonical_checkpoint_bytes(forged)


def test_checkpoint_identity_and_bindings_do_not_use_mutable_equality(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    raw = checkpoint.canonical_checkpoint_bytes(value)

    monkeypatch.setattr(
        checkpoint.ValidatedCheckpoint,
        "__eq__",
        lambda _self, _other: True,
        raising=False,
    )
    monkeypatch.setattr(
        checkpoint.ValidatedCheckpoint,
        "__hash__",
        lambda _self: 1,
        raising=False,
    )
    forged = object.__new__(checkpoint.ValidatedCheckpoint)
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.canonical_checkpoint_bytes(forged)
    assert checkpoint.canonical_checkpoint_bytes(value) == raw

    def equality_must_not_run(_self: object, _other: object) -> bool:
        raise AssertionError("CheckpointBindings equality is not authoritative")

    monkeypatch.setattr(checkpoint.CheckpointBindings, "__eq__", equality_must_not_run)
    monkeypatch.setattr(checkpoint.CheckpointBindings, "__ne__", equality_must_not_run)
    validated = checkpoint.validate_cpu_synthetic_checkpoint_bytes(
        raw,
        expected_bindings=bindings,
    )
    assert checkpoint.canonical_checkpoint_bytes(validated) == raw
    with pytest.raises(checkpoint.CheckpointContractError, match="binding mismatch"):
        checkpoint.validate_cpu_synthetic_checkpoint_bytes(
            raw,
            expected_bindings=_bindings(run_id="phasepair-synthetic-run-002"),
        )


def test_relative_path_and_parent_alias_are_rejected(tmp_path: Path) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    with pytest.raises(checkpoint.CheckpointPathError):
        checkpoint.write_checkpoint_atomic(value, Path("relative.ppckpt"))
    alias_parent = tmp_path / "alias"
    try:
        alias_parent.symlink_to(tmp_path, target_is_directory=True)
    except OSError:
        return
    with pytest.raises(checkpoint.CheckpointPathError):
        checkpoint.write_checkpoint_atomic(value, (alias_parent / "x.ppckpt").absolute())


def test_concurrent_atomic_writers_never_overwrite(tmp_path: Path) -> None:
    bindings = _bindings()
    first = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    alternate_sections = list(_sections())
    alternate_sections[1] = ("model_state", b"alternate-model-state")
    second = checkpoint.build_cpu_synthetic_checkpoint(
        bindings,
        tuple(alternate_sections),
    )
    target = (tmp_path / "race.ppckpt").absolute()
    barrier = threading.Barrier(3)
    outcomes: list[tuple[str, bytes | None]] = []
    output_lock = threading.Lock()

    def writer(value: checkpoint.ValidatedCheckpoint) -> None:
        barrier.wait()
        try:
            digest = checkpoint.write_checkpoint_atomic(value, target)
        except checkpoint.CheckpointPathError:
            result = ("REJECTED", None)
        else:
            result = ("COMMITTED", digest)
        with output_lock:
            outcomes.append(result)

    threads = [
        threading.Thread(target=writer, args=(first,)),
        threading.Thread(target=writer, args=(second,)),
    ]
    for thread in threads:
        thread.start()
    barrier.wait()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive()
    assert [item[0] for item in outcomes].count("COMMITTED") == 1
    assert [item[0] for item in outcomes].count("REJECTED") == 1
    actual = hashlib.sha256(target.read_bytes()).digest()
    committed = next(item[1] for item in outcomes if item[0] == "COMMITTED")
    assert actual == committed


def test_public_constant_rebind_does_not_change_checkpoint_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    sections = _sections()
    baseline = checkpoint.canonical_checkpoint_bytes(
        checkpoint.build_cpu_synthetic_checkpoint(bindings, sections)
    )
    monkeypatch.setattr(checkpoint, "MAGIC", b"FORGED\x00")
    monkeypatch.setattr(checkpoint, "SCHEMA", "forged-schema")
    monkeypatch.setattr(checkpoint, "STATUS", "FORGED_PASS")
    monkeypatch.setattr(checkpoint, "REQUIRED_SECTIONS", ("forged",))
    monkeypatch.setattr(checkpoint, "_MAX_FILE_BYTES", 1)
    rebuilt = checkpoint.canonical_checkpoint_bytes(
        checkpoint.build_cpu_synthetic_checkpoint(bindings, sections)
    )
    assert rebuilt == baseline
    validated = checkpoint.validate_cpu_synthetic_checkpoint_bytes(
        baseline,
        expected_bindings=bindings,
    )
    assert checkpoint.canonical_checkpoint_bytes(validated) == baseline


def test_private_dependency_and_class_rebind_does_not_change_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    sections = _sections()
    baseline_value = checkpoint.build_cpu_synthetic_checkpoint(bindings, sections)
    baseline = checkpoint.canonical_checkpoint_bytes(baseline_value)
    baseline_sha = checkpoint.checkpoint_sha256(baseline_value)
    bindings_type = type(bindings)
    value_type = type(baseline_value)
    contract_error = checkpoint.CheckpointContractError

    class Bomb:
        def __init__(self, *_: object, **__: object) -> None:
            raise AssertionError("rebound dependency was read")

        def __call__(self, *_: object, **__: object) -> object:
            raise AssertionError("rebound dependency was called")

    bomb = Bomb.__new__(Bomb)
    for name in (
        "_exact_uint",
        "_ascii",
        "_raw32",
        "_optional_raw32",
        "_validate_bindings",
        "_bindings_object",
        "_bindings_from_object",
        "_sections",
        "_manifest_bytes",
        "_mint_validated",
        "_crossbind_state_sections",
        "_parse",
        "_metadata_is_reparse",
        "_safe_parent",
        "_stable_read",
        "_CheckpointRecord",
        "_ISSUED",
    ):
        monkeypatch.setattr(checkpoint, name, bomb)
    monkeypatch.setattr(checkpoint, "CheckpointBindings", Bomb)
    monkeypatch.setattr(checkpoint, "ValidatedCheckpoint", Bomb)
    monkeypatch.setattr(checkpoint, "CheckpointContractError", RuntimeError)
    monkeypatch.setattr(checkpoint, "CheckpointPathError", RuntimeError)
    monkeypatch.setattr(checkpoint, "WeakKeyDictionary", Bomb)
    monkeypatch.setattr(checkpoint, "Path", Bomb)
    monkeypatch.setattr(checkpoint, "hashlib", bomb)
    monkeypatch.setattr(checkpoint, "json", bomb)
    monkeypatch.setattr(checkpoint, "os", bomb)
    monkeypatch.setattr(checkpoint, "stat", bomb)
    monkeypatch.setattr(checkpoint, "tempfile", bomb)
    monkeypatch.setattr(checkpoint, "threading", bomb)

    rebuilt = checkpoint.build_cpu_synthetic_checkpoint(bindings, sections)
    assert type(rebuilt) is value_type
    assert checkpoint.canonical_checkpoint_bytes(rebuilt) == baseline
    assert checkpoint.checkpoint_sha256(rebuilt) == baseline_sha
    parsed = checkpoint.validate_cpu_synthetic_checkpoint_bytes(
        baseline,
        expected_bindings=bindings,
    )
    assert type(parsed) is value_type
    assert checkpoint.checkpoint_sections(parsed) == sections
    assert type(checkpoint.checkpoint_bindings(parsed)) is bindings_type
    forged = object.__new__(value_type)
    with pytest.raises(contract_error):
        checkpoint.canonical_checkpoint_bytes(forged)


def test_issued_record_isolated_from_caller_binding_mutation() -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    raw = checkpoint.canonical_checkpoint_bytes(value)
    object.__setattr__(bindings, "global_step", 3)
    object.__setattr__(bindings, "seed", True)
    assert checkpoint.canonical_checkpoint_bytes(value) == raw
    issued = checkpoint.checkpoint_bindings(value)
    assert issued.global_step == 2
    assert issued.seed == 1729
    with pytest.raises(TypeError):
        checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())


def test_double_pass_stable_read_rejects_between_pass_change(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    target = (tmp_path / "changing.ppckpt").absolute()
    checkpoint.write_checkpoint_atomic(value, target)
    original_lstat = type(target).lstat
    original_open = type(target).open
    target_lstats = 0

    def changing_lstat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstats
        if self == target:
            target_lstats += 1
            if target_lstats == 3:
                raw = target.read_bytes()
                with original_open(target, "wb") as handle:
                    handle.write(raw[:-1] + bytes([raw[-1] ^ 1]))
        return original_lstat(self, *args, **kwargs)

    monkeypatch.setattr(type(target), "lstat", changing_lstat)
    with pytest.raises(checkpoint.CheckpointPathError):
        checkpoint.read_checkpoint(target, expected_bindings=bindings)


def test_postcommit_verification_failure_removes_new_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    target = (tmp_path / "postcommit-failure.ppckpt").absolute()
    original_lstat = type(target).lstat
    target_lstats = 0

    def failing_lstat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstats
        if self == target and target.exists():
            target_lstats += 1
            if target_lstats == 2:
                raise OSError("controlled post-commit read failure")
        return original_lstat(self, *args, **kwargs)

    monkeypatch.setattr(type(target), "lstat", failing_lstat)
    with pytest.raises(checkpoint.CheckpointPathError):
        checkpoint.write_checkpoint_atomic(value, target)
    assert not target.exists()


def test_first_postlink_identity_failure_removes_only_owned_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    target = (tmp_path / "first-postlink-failure.ppckpt").absolute()
    original_lstat = type(target).lstat
    injected = False

    def failing_first_lstat(
        self: Path, *args: object, **kwargs: object
    ) -> os.stat_result:
        nonlocal injected
        metadata = original_lstat(self, *args, **kwargs)
        # Before commit the target is absent.  Immediately after os.link the
        # temp inode has exactly two names, making this seam invariant across
        # the differing pathlib implementations in CPython 3.12 and 3.14.
        if self == target and metadata.st_nlink == 2 and not injected:
            injected = True
            raise OSError("controlled first post-link identity failure")
        return metadata

    monkeypatch.setattr(type(target), "lstat", failing_first_lstat)
    with pytest.raises(
        checkpoint.CheckpointPathError,
        match="committed target identity is unavailable",
    ):
        checkpoint.write_checkpoint_atomic(value, target)
    assert injected
    assert not target.exists()
    assert not tuple(tmp_path.glob("*.committed-retirement"))
    assert not tuple(tmp_path.glob(".*.tmp"))


def test_failed_commit_cleanup_never_deletes_replaced_target(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bindings = _bindings()
    value = checkpoint.build_cpu_synthetic_checkpoint(bindings, _sections())
    target = (tmp_path / "replaced-after-commit.ppckpt").absolute()
    original_lstat = type(target).lstat
    original_open = type(target).open
    target_lstats = 0
    outsider = b"not-the-checkpoint-writer's-inode"

    def replacing_lstat(self: Path, *args: object, **kwargs: object) -> os.stat_result:
        nonlocal target_lstats
        if self == target and target.exists():
            target_lstats += 1
            if target_lstats == 2:
                target.unlink()
                with original_open(target, "wb") as handle:
                    handle.write(outsider)
        return original_lstat(self, *args, **kwargs)

    monkeypatch.setattr(type(target), "lstat", replacing_lstat)
    with pytest.raises(checkpoint.CheckpointContractError):
        checkpoint.write_checkpoint_atomic(value, target)
    assert target.read_bytes() == outsider
    assert not tuple(tmp_path.glob(".*.tmp"))
