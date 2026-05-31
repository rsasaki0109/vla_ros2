from __future__ import annotations

from typer.testing import CliRunner

from vla_zoo.cli.main import app


def test_cli_list() -> None:
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "dummy" in result.output
    assert "openvla" in result.output


def test_cli_demo_pybullet_help() -> None:
    result = CliRunner().invoke(app, ["demo", "pybullet", "--help"])
    assert result.exit_code == 0
    assert "--model" in result.output
    assert "--runtime" in result.output


def test_cli_compare_adapters() -> None:
    result = CliRunner().invoke(app, ["compare", "adapters"])
    assert result.exit_code == 0
    assert "dummy" in result.output
    assert "openvla" in result.output


def test_cli_compare_pybullet_help() -> None:
    result = CliRunner().invoke(app, ["compare", "pybullet", "--help"])
    assert result.exit_code == 0
    assert "--models" in result.output
    assert "--allow-local-heavy" in result.output
