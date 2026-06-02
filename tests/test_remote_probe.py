from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vla_zoo.cli.main import app
from vla_zoo.runtime.remote_probe import (
    STATUS_OK,
    STATUS_PREDICT_FAILED,
    STATUS_UNREACHABLE,
    format_remote_probe_markdown,
    probe_remote_model,
)

runner = CliRunner()

HEALTHY = {"ready": True, "model": "openvla", "runtime": "server", "status": "ok"}
ACTION = {"action_space": "eef_delta", "data": [0.0] * 7, "shape": [7], "metadata": {}}


def _ok_health(remote_url: str, timeout: float) -> tuple[str | None, dict[str, object] | None]:
    return None, dict(HEALTHY)


def test_probe_records_action_when_healthy() -> None:
    result = probe_remote_model(
        model_name="openvla",
        remote_url="http://gpu-box:8000",
        health_fn=_ok_health,
        predict_fn=lambda *_: dict(ACTION),
    )

    assert result.status == STATUS_OK
    assert result.is_ok
    assert result.health == HEALTHY
    assert result.action == ACTION
    assert result.error is None


def test_probe_reports_unreachable_without_calling_predict() -> None:
    calls: list[str] = []

    def _predict(*_: object) -> dict[str, object]:
        calls.append("called")
        return {}

    result = probe_remote_model(
        model_name="openvla",
        remote_url="http://gpu-box:8000",
        health_fn=lambda *_: ("connection refused", None),
        predict_fn=_predict,
    )

    assert result.status == STATUS_UNREACHABLE
    assert result.action is None
    assert result.error == "connection refused"
    assert calls == []  # predict must not run when health fails


def test_probe_reports_predict_failure_after_health_ok() -> None:
    def _boom(*_: object) -> dict[str, object]:
        raise RuntimeError("remote predict blew up")

    result = probe_remote_model(
        model_name="openvla",
        remote_url="http://gpu-box:8000",
        health_fn=_ok_health,
        predict_fn=_boom,
    )

    assert result.status == STATUS_PREDICT_FAILED
    assert result.health == HEALTHY
    assert result.action is None
    assert "remote predict blew up" in (result.error or "")


def test_default_health_handles_unreachable_server() -> None:
    # Port 9 (discard) refuses fast; the default urllib path must degrade to a result.
    result = probe_remote_model(
        model_name="openvla",
        remote_url="http://127.0.0.1:9",
        timeout=0.25,
    )

    assert result.status == STATUS_UNREACHABLE
    assert result.action is None


def test_markdown_renders_status_and_no_quality_claim() -> None:
    result = probe_remote_model(
        model_name="openvla",
        remote_url="http://gpu-box:8000",
        health_fn=_ok_health,
        predict_fn=lambda *_: dict(ACTION),
    )
    markdown = format_remote_probe_markdown(result)

    assert "# Remote VLA Probe: openvla" in markdown
    assert "not a robot task-success benchmark" in markdown
    assert "Recorded Action" in markdown


def test_cli_remote_probe_help() -> None:
    result = runner.invoke(app, ["remote-probe", "--help"])

    assert result.exit_code == 0
    assert "--remote-url" in result.output
    assert "--strict" in result.output


def test_cli_remote_probe_strict_exits_on_unreachable(tmp_path: Path) -> None:
    out = tmp_path / "probe.json"
    result = runner.invoke(
        app,
        [
            "remote-probe",
            "--model",
            "openvla",
            "--remote-url",
            "http://127.0.0.1:9",
            "--timeout",
            "0.25",
            "--out",
            str(out),
            "--strict",
        ],
    )

    assert result.exit_code == 1
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["status"] == STATUS_UNREACHABLE
