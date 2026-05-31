from __future__ import annotations

from typer.testing import CliRunner

from vla_zoo.cli.main import app


def test_cli_list() -> None:
    result = CliRunner().invoke(app, ["list"])
    assert result.exit_code == 0
    assert "dummy" in result.output
    assert "openvla" in result.output
