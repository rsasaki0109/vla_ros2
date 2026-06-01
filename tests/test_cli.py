from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vla_zoo.cli.main import _load_json_manifest, _manifest_int, _manifest_targets, app


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
    assert "--manifest" in result.output
    assert "--remote-map" in result.output
    assert "--allow-local-heavy" in result.output
    assert "--markdown-out" in result.output


def test_cli_compare_manifest_loads_targets(tmp_path: Path) -> None:
    manifest = tmp_path / "comparison.json"
    manifest.write_text(
        json.dumps(
            {
                "render_stride": 48,
                "models": [
                    {
                        "name": "dummy",
                        "runtime": "remote",
                        "remote_url": "http://127.0.0.1:8010",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    payload = _load_json_manifest(manifest)
    targets = _manifest_targets(payload)

    assert _manifest_int(payload, "render_stride", 12) == 48
    assert targets[0].model_name == "dummy"
    assert targets[0].runtime == "remote"
    assert targets[0].remote_url == "http://127.0.0.1:8010"
