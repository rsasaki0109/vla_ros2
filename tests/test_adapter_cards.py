from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vla_zoo.cli.main import app
from vla_zoo.compare.cards import (
    adapter_card_payload,
    format_adapter_card_markdown,
    write_adapter_cards,
)
from vla_zoo.core.registry import get_adapter_info


def test_adapter_card_payload_exposes_runtime_contract() -> None:
    payload = adapter_card_payload(get_adapter_info("openvla"), status="available")

    assert payload["name"] == "openvla"
    assert payload["status"] == "available"
    assert payload["runtime_contract"]["action_space"] == "eef_delta"
    assert "single RGB image" in payload["runtime_contract"]["input_requirements"]
    assert "OpenVLA" in payload["metadata"]["upstream_project"]


def test_adapter_card_markdown_includes_caveats() -> None:
    markdown = format_adapter_card_markdown(get_adapter_info("smolvla"), status="available")

    assert "# smolvla Adapter Card" in markdown
    assert "Runtime Contract" in markdown
    assert "lerobot/smolvla_base" in markdown
    assert "not a robot task-success claim" in markdown


def test_write_adapter_cards_creates_index_and_cards(tmp_path: Path) -> None:
    paths = write_adapter_cards(
        tmp_path,
        models=("dummy", "openvla"),
        status_provider=lambda name: f"status:{name}",
    )

    assert tmp_path / "README.md" in paths
    assert tmp_path / "dummy.md" in paths
    assert tmp_path / "openvla.md" in paths
    index = (tmp_path / "README.md").read_text(encoding="utf-8")
    assert "vla_zoo Adapter Cards" in index
    assert "[card](openvla.md)" in index
    assert "status:openvla" in (tmp_path / "openvla.md").read_text(encoding="utf-8")


def test_cli_info_includes_capability_payload() -> None:
    result = CliRunner().invoke(app, ["info", "pi0"])

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["name"] == "pi0"
    assert payload["runtime_contract"]["remote_runtime"] == "recommended"
    assert "not completed" in payload["verification"]


def test_cli_compare_cards_writes_markdown(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "cards",
            "--models",
            "dummy,openvla",
            "--out-dir",
            str(tmp_path),
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert str(tmp_path / "README.md") in payload["files"]
    assert (tmp_path / "dummy.md").exists()
    assert (tmp_path / "openvla.md").exists()
