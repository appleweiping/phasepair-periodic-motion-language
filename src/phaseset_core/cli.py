"""Fail-closed command surface for the PhaseSet execution protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Sequence, TextIO

from phaseset_core.execution import (
    AUTHORITY,
    COMMANDS,
    DataFreeRunner,
    ExecutionContractError,
    ExecutionHold,
    RuntimeAdapter,
    RuntimeAdmission,
    canonical_command_result_bytes,
    canonical_preflight_bytes,
    canonical_public_training_config_bytes,
    canonical_runtime_admission_bytes,
    load_training_config,
)


EXIT_OK = 0
EXIT_USAGE_OR_CONTRACT = 2
EXIT_HOLD = 3
EXIT_EXECUTION_FAILED = 4
_DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "phaseset" / "training.json"


def _canonical_bytes(value: object) -> bytes:
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


def _add_config(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="public fail-closed training configuration",
    )


def _load_cli_config(path: Path) -> bytes:
    try:
        return load_training_config(path)
    except FileNotFoundError:
        if path == _DEFAULT_CONFIG:
            return canonical_public_training_config_bytes()
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phaseset",
        description="PhaseSet authority-zero execution protocol",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser("preflight", help="enumerate external prerequisite holds")
    _add_config(preflight)

    prepare_data = subparsers.add_parser(
        "prepare-data", help="request registered private data preparation"
    )
    _add_config(prepare_data)
    prepare_data.add_argument("--source-manifest", type=Path)
    prepare_data.add_argument("--prepared-root", type=Path)

    audit_split = subparsers.add_parser("audit-split", help="audit a private split manifest")
    _add_config(audit_split)
    audit_split.add_argument("--split-manifest", type=Path)

    run_base = subparsers.add_parser("run-base", help="request one registered base run")
    _add_config(run_base)
    run_base.add_argument("--run-id", required=True)

    qualify = subparsers.add_parser("qualify-base", help="qualify all nine base terminals")
    _add_config(qualify)
    qualify.add_argument("--terminal-root", type=Path)

    cache = subparsers.add_parser(
        "build-periodic-cache", help="request a qualified seed-specific periodic cache"
    )
    _add_config(cache)
    cache.add_argument("--seed", required=True, type=int)

    residual = subparsers.add_parser("run-residual", help="request one registered residual run")
    _add_config(residual)
    residual.add_argument("--run-id", required=True)

    evaluate = subparsers.add_parser(
        "evaluate", help="request validation or sealed-test evaluation"
    )
    _add_config(evaluate)
    evaluate.add_argument("--split", required=True, choices=("validation", "test"))
    evaluate.add_argument("--terminal-root", type=Path)

    bootstrap = subparsers.add_parser(
        "bootstrap", help="request the registered 100k paired-capture bootstrap"
    )
    _add_config(bootstrap)
    bootstrap.add_argument("--evaluation-manifest", type=Path)

    render = subparsers.add_parser(
        "render-paper", help="render only from a frozen statistical aggregate"
    )
    _add_config(render)
    render.add_argument("--statistical-report", type=Path)

    resume = subparsers.add_parser("resume", help="request a new attempt from a checkpoint")
    _add_config(resume)
    resume.add_argument("--attempt-dir", type=Path)

    return parser


def _emit(raw: bytes, stream: TextIO) -> None:
    stream.write(raw.decode("ascii"))
    stream.flush()


def _hold_bytes(command: str, hold_codes: tuple[str, ...]) -> bytes:
    return _canonical_bytes(
        {
            "authority": AUTHORITY,
            "command": command,
            "execution_authorized": False,
            "external_receipt_verified": False,
            "hold_codes": list(hold_codes),
            "production": False,
            "result_claimed": False,
            "status": "HELD",
        }
    )


def _contract_error_bytes(command: str, code: str) -> bytes:
    return _canonical_bytes(
        {
            "authority": AUTHORITY,
            "command": command,
            "error_code": code,
            "execution_authorized": False,
            "production": False,
            "result_claimed": False,
            "status": "REJECTED",
        }
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    stdout: TextIO | None = None,
    stderr: TextIO | None = None,
    runtime_adapter: RuntimeAdapter | None = None,
) -> int:
    """Parse one command, emit canonical JSON, and never perform external I/O."""

    output = sys.stdout if stdout is None else stdout
    errors = sys.stderr if stderr is None else stderr
    parser = build_parser()
    try:
        namespace = parser.parse_args(argv)
    except SystemExit as exc:
        return int(exc.code)

    command = namespace.command
    if command not in COMMANDS:
        _emit(_contract_error_bytes(str(command), "COMMAND_OUTSIDE_CENSUS"), errors)
        return EXIT_USAGE_OR_CONTRACT
    try:
        config_raw = _load_cli_config(namespace.config)
        runner = DataFreeRunner(config_raw, runtime_adapter=runtime_adapter)
        if command == "preflight":
            report = runner.preflight()
            if type(report) is RuntimeAdmission:
                _emit(canonical_runtime_admission_bytes(report), output)
                return EXIT_OK
            _emit(canonical_preflight_bytes(report), output)
            return EXIT_OK if report.ready else EXIT_HOLD

        result = runner.execute_command(
            command,
            run_id=getattr(namespace, "run_id", None),
            seed=getattr(namespace, "seed", None),
            split=getattr(namespace, "split", None),
        )
        _emit(canonical_command_result_bytes(result), output)
        if result.outcome == "COMPLETED":
            return EXIT_OK
        if result.outcome == "HELD":
            return EXIT_HOLD
        return EXIT_EXECUTION_FAILED
    except FileNotFoundError:
        _emit(_hold_bytes(command, ("HOLD_TRAINING_CONFIG_ABSENT",)), output)
        return EXIT_HOLD
    except ExecutionHold as exc:
        _emit(_hold_bytes(command, exc.hold_codes), output)
        return EXIT_HOLD
    except (ExecutionContractError, TypeError, ValueError):
        _emit(_contract_error_bytes(command, "EXECUTION_CONTRACT_REJECTED"), errors)
        return EXIT_USAGE_OR_CONTRACT
    except Exception:
        _emit(_contract_error_bytes(command, "RUNTIME_ADAPTER_FAILED_CLOSED"), errors)
        return EXIT_EXECUTION_FAILED

    _emit(_contract_error_bytes(command, "PUBLIC_RUNNER_CANNOT_EXECUTE"), errors)
    return EXIT_USAGE_OR_CONTRACT


if __name__ == "__main__":  # pragma: no cover - exercised through main in tests
    raise SystemExit(main())


__all__ = [
    "EXIT_HOLD",
    "EXIT_OK",
    "EXIT_USAGE_OR_CONTRACT",
    "build_parser",
    "main",
]
