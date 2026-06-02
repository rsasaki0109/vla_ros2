from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vla_zoo.cli.main import app
from vla_zoo.runtime.smolvla_plan import (
    build_smolvla_remote_plan,
    format_smolvla_remote_plan_markdown,
)

runner = CliRunner()


def test_build_plan_uses_isolated_env_and_smolvla_extra() -> None:
    plan = build_smolvla_remote_plan(public_host="gpu-box", port=8000)

    assert plan.model_name == "smolvla"
    assert plan.public_url == "http://gpu-box:8000"
    assert plan.env_create_command == ("python3", "-m", "venv", ".venv-smolvla")
    # install targets the venv pip and the smolvla extra, not the base env
    assert plan.install_command[0] == ".venv-smolvla/bin/pip"
    assert plan.install_command[-1] == ".[cli,server,smolvla]"


def test_server_command_carries_checkpoint_and_device() -> None:
    plan = build_smolvla_remote_plan(pretrained="lerobot/smolvla_base", device="cuda:0", port=8000)
    server = plan.server_command

    assert "serve" in server
    assert server[server.index("--model") + 1] == "smolvla"
    assert server[server.index("--pretrained") + 1] == "lerobot/smolvla_base"
    assert server[server.index("--device") + 1] == "cuda:0"
    assert server[server.index("--port") + 1] == "8000"


def test_robot_side_commands_reference_remote_url() -> None:
    plan = build_smolvla_remote_plan(public_host="gpu-box", port=8000)

    assert plan.health_command == ("curl", "-fsS", "http://gpu-box:8000/health")
    assert "--remote-url" in plan.predict_probe_command
    assert "http://gpu-box:8000" in plan.predict_probe_command
    assert "smolvla=http://gpu-box:8000" in plan.compare_command
    assert "remote-smoke-report" in plan.ros_plan_command


def test_to_dict_is_json_serializable_with_shell_variants() -> None:
    payload = build_smolvla_remote_plan().to_dict()

    assert payload["install_command_shell"].startswith(".venv-smolvla/bin/pip install")
    assert isinstance(payload["server_command"], list)
    assert json.dumps(payload)  # serializable


def test_markdown_documents_isolation_and_makes_no_quality_claim() -> None:
    markdown = format_smolvla_remote_plan_markdown(build_smolvla_remote_plan())

    assert "# SmolVLA Remote Serving Plan" in markdown
    assert "Isolated Environment" in markdown
    assert ".[cli,server,smolvla]" in markdown
    lowered = markdown.lower()
    assert "no claim about" in lowered or "no /v1/predict" in lowered
    assert "task-success quality" in lowered


def test_cli_smolvla_remote_plan_help() -> None:
    result = runner.invoke(app, ["smolvla-remote-plan", "--help"])

    assert result.exit_code == 0
    assert "--public-host" in result.output
    assert "--venv-dir" in result.output


def test_cli_smolvla_remote_plan_writes_artifacts(tmp_path: Path) -> None:
    json_out = tmp_path / "plan.json"
    md_out = tmp_path / "plan.md"

    result = runner.invoke(
        app,
        [
            "smolvla-remote-plan",
            "--public-host",
            "gpu-box",
            "--port",
            "8000",
            "--out",
            str(json_out),
            "--markdown-out",
            str(md_out),
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["model_name"] == "smolvla"
    assert "SmolVLA Remote Serving Plan" in md_out.read_text(encoding="utf-8")
