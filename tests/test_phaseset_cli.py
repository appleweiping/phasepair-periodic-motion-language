from __future__ import annotations

import hashlib
from io import StringIO
import json
from pathlib import Path

import pytest

from phaseset_core import cli, execution, experiments


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "phaseset" / "training.json"


def _invoke(
    arguments: list[str],
    *,
    runtime_adapter: execution.RuntimeAdapter | None = None,
) -> tuple[int, str, str]:
    stdout = StringIO()
    stderr = StringIO()
    code = cli.main(
        arguments,
        stdout=stdout,
        stderr=stderr,
        runtime_adapter=runtime_adapter,
    )
    return code, stdout.getvalue(), stderr.getvalue()


def test_parser_exposes_complete_protocol_command_census() -> None:
    help_text = cli.build_parser().format_help()
    expected = (
        "preflight",
        "prepare-data",
        "audit-split",
        "run-base",
        "qualify-base",
        "build-periodic-cache",
        "run-residual",
        "evaluate",
        "bootstrap",
        "render-paper",
        "resume",
    )
    assert len(expected) == 11
    assert cli.COMMANDS == expected
    for command in expected:
        assert command in help_text


def test_preflight_emits_all_holds_without_endpoint_or_path() -> None:
    code, stdout, stderr = _invoke(["preflight", "--config", str(CONFIG)])
    payload = json.loads(stdout)
    assert code == cli.EXIT_HOLD
    assert stderr == ""
    assert payload["ready"] is False
    assert payload["external_receipt_verified"] is False
    assert "HOLD_SERVER_ENDPOINT_ABSENT" in payload["hold_codes"]
    assert str(CONFIG) not in stdout
    assert "http" + "://" not in stdout and "https" + "://" not in stdout


def test_every_execution_command_stops_before_external_side_effect() -> None:
    commands = (
        ["prepare-data"],
        ["audit-split"],
        ["run-base", "--run-id", experiments.base_run_ids()[0]],
        ["qualify-base"],
        ["build-periodic-cache", "--seed", "1729"],
        ["run-residual", "--run-id", experiments.residual_run_ids()[0]],
        ["evaluate", "--split", "validation"],
        ["evaluate", "--split", "test"],
        ["bootstrap"],
        ["render-paper"],
        ["resume"],
    )
    for command in commands:
        code, stdout, stderr = _invoke([*command, "--config", str(CONFIG)])
        assert code == cli.EXIT_HOLD
        assert stderr == ""
        payload = json.loads(stdout)
        assert payload["status"] == "HELD"
        assert payload["execution_authorized"] is False


def test_invalid_run_id_is_rejected_without_echoing_input() -> None:
    secret_like_input = "https" + "://" + "private.invalid/token-value"
    code, stdout, stderr = _invoke(
        ["run-base", "--run-id", secret_like_input, "--config", str(CONFIG)]
    )
    assert code == cli.EXIT_USAGE_OR_CONTRACT
    assert stdout == ""
    payload = json.loads(stderr)
    assert payload["error_code"] == "EXECUTION_CONTRACT_REJECTED"
    assert secret_like_input not in stderr


def test_missing_config_fails_closed_without_echoing_path(tmp_path: Path) -> None:
    missing = tmp_path / "private-server-config.json"
    code, stdout, stderr = _invoke(["preflight", "--config", str(missing)])
    assert code == cli.EXIT_HOLD
    assert stderr == ""
    assert json.loads(stdout)["hold_codes"] == ["HOLD_TRAINING_CONFIG_ABSENT"]
    assert str(missing) not in stdout


def test_installed_default_config_falls_back_to_same_authority_zero_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    missing_packaged_default = tmp_path / "installed-layout" / "training.json"
    monkeypatch.setattr(cli, "_DEFAULT_CONFIG", missing_packaged_default)
    code, stdout, stderr = _invoke(["preflight"])
    assert code == cli.EXIT_HOLD
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["ready"] is False
    assert "HOLD_SERVER_ENDPOINT_ABSENT" in payload["hold_codes"]
    assert "HOLD_TRAINING_CONFIG_ABSENT" not in payload["hold_codes"]
    assert str(missing_packaged_default) not in stdout


def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


class _SyntheticCLIAdapter:
    commands = execution.COMMANDS
    adapter_sha256 = _digest("synthetic-cli-adapter")
    handler_manifest_sha256 = execution.runtime_handler_manifest_sha256(commands)

    def __init__(self, *, fail_command: str | None = None) -> None:
        self.fail_command = fail_command

    def admit(
        self,
        request: execution.RuntimeAdmissionRequest,
    ) -> execution.RuntimeAdmission:
        return execution.RuntimeAdmission(
            runtime_mode="SYNTHETIC_DATA_FREE",
            admitted_at_utc="2026-08-25T23:00:00Z",
            adapter_sha256=self.adapter_sha256,
            handler_manifest_sha256=request.handler_manifest_sha256,
            plan_sha256=request.plan_sha256,
            matrix_sha256=request.matrix_sha256,
            training_config_sha256=request.training_config_sha256,
            data_manifest_sha256=_digest("synthetic-cli-data"),
            prepared_data_manifest_sha256=_digest("synthetic-cli-prepared"),
            split_audit_sha256=_digest("synthetic-cli-split"),
            rights_assertion_sha256=_digest("synthetic-cli-rights-not-a-grant"),
            runtime_assertion_sha256=_digest("synthetic-cli-runtime"),
            execution_assertion_sha256=_digest("synthetic-cli-execution"),
            authority=0,
            execution_authorized=True,
            production=False,
            external_authentication_asserted=False,
            public_verification_performed=False,
            result_claimed=False,
            status="SYNTHETIC_DATA_FREE_NO_SCIENTIFIC_RESULT",
        )

    def handle(self, intent: execution.CommandIntent) -> execution.CommandResult:
        if intent.admission_sha256 is None:
            raise AssertionError("adapter command was not admitted")
        return execution.CommandResult(
            command=intent.command,
            outcome=("FAILED" if intent.command == self.fail_command else "COMPLETED"),
            completed_at_utc="2026-08-25T23:01:00Z",
            run_id=intent.run_id,
            seed=intent.seed,
            split=intent.split,
            artifact_sha256s=(_digest(f"synthetic-cli-{intent.command}"),),
            admission_sha256=intent.admission_sha256,
            runtime_mode=intent.runtime_mode,
            authority=intent.authority,
            production=intent.production,
        )


def test_injected_synthetic_cli_executes_without_weakening_public_default() -> None:
    adapter = _SyntheticCLIAdapter()
    code, stdout, stderr = _invoke(
        ["preflight", "--config", str(CONFIG)],
        runtime_adapter=adapter,
    )
    assert code == cli.EXIT_OK
    assert stderr == ""
    assert json.loads(stdout)["notice"] == "SYNTHETIC / NO SCIENTIFIC RESULT"

    code, stdout, stderr = _invoke(
        ["prepare-data", "--config", str(CONFIG)],
        runtime_adapter=adapter,
    )
    assert code == cli.EXIT_OK
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["outcome"] == "COMPLETED"
    assert payload["notice"] == "SYNTHETIC / NO SCIENTIFIC RESULT"
    assert payload["result_claimed"] is False

    default_code, default_stdout, default_stderr = _invoke(
        ["prepare-data", "--config", str(CONFIG)]
    )
    assert default_code == cli.EXIT_HOLD
    assert default_stderr == ""
    assert json.loads(default_stdout)["status"] == "HELD"


def test_injected_cli_returns_distinct_failure_exit_without_leaking_details() -> None:
    adapter = _SyntheticCLIAdapter(fail_command="run-residual")
    code, stdout, stderr = _invoke(
        [
            "run-residual",
            "--run-id",
            experiments.residual_run_ids()[0],
            "--config",
            str(CONFIG),
        ],
        runtime_adapter=adapter,
    )
    assert code == cli.EXIT_EXECUTION_FAILED
    assert stderr == ""
    payload = json.loads(stdout)
    assert payload["outcome"] == "FAILED"
    assert payload["result_claimed"] is False
    assert "http" + "://" not in stdout


def test_adapter_exception_fails_closed_without_echoing_private_detail() -> None:
    adapter = _SyntheticCLIAdapter()

    def explode(_: execution.CommandIntent) -> execution.CommandResult:
        raise RuntimeError("https" + "://private.invalid/credential-value")

    adapter.handle = explode  # type: ignore[method-assign]
    code, stdout, stderr = _invoke(
        ["prepare-data", "--config", str(CONFIG)],
        runtime_adapter=adapter,
    )
    assert code == cli.EXIT_EXECUTION_FAILED
    assert stdout == ""
    payload = json.loads(stderr)
    assert payload["error_code"] == "RUNTIME_ADAPTER_FAILED_CLOSED"
    assert "private.invalid" not in stderr
    assert "credential-value" not in stderr
