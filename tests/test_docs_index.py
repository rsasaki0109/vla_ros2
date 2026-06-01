from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vla_zoo.cli.main import app
from vla_zoo.docs.artifact_index import (
    CATEGORIES,
    DEFAULT_ARTIFACTS,
    ArtifactEntry,
    artifact_index_payload,
    build_artifact_index,
    format_artifact_index_html,
    format_artifact_index_table,
)

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_default_catalog_uses_known_categories() -> None:
    assert DEFAULT_ARTIFACTS
    for entry in DEFAULT_ARTIFACTS:
        assert entry.category in CATEGORIES
        assert entry.kind in ("generated", "checked", "manual")


def test_default_artifacts_exist_in_repo() -> None:
    index = build_artifact_index(root=REPO_ROOT)
    assert index.ok_overall, [entry.path for entry in index.missing]


def test_build_flags_missing(tmp_path: Path) -> None:
    entries = (
        ArtifactEntry(
            title="present",
            path="present.md",
            category="model evidence",
            status="generated",
            kind="generated",
        ),
        ArtifactEntry(
            title="absent",
            path="absent.md",
            category="ROS2",
            status="planned",
            kind="checked",
        ),
    )
    (tmp_path / "present.md").write_text("x", encoding="utf-8")

    index = build_artifact_index(entries, root=tmp_path)

    assert index.ok_overall is False
    assert len(index.missing) == 1
    assert index.missing[0].path == "absent.md"


def test_payload_and_renderers(tmp_path: Path) -> None:
    (tmp_path / "present.md").write_text("x", encoding="utf-8")
    entries = (
        ArtifactEntry(
            title="present report",
            path="present.md",
            category="simulation",
            status="verified",
            kind="checked",
            source_command="vla-zoo demo gif-report",
            caveat="runtime artifact",
        ),
    )
    index = build_artifact_index(entries, root=tmp_path)

    payload = artifact_index_payload(index)
    assert payload["count"] == 1
    assert payload["missing"] == 0
    assert json.dumps(payload)  # serializable

    html = format_artifact_index_html(index, title="My Index")
    assert "My Index" in html
    assert 'href="present.md"' in html
    assert "runtime artifact" in html

    table = format_artifact_index_table(index)
    assert "simulation" in table
    assert "count=1" in table


def test_html_marks_missing(tmp_path: Path) -> None:
    entries = (
        ArtifactEntry(
            title="gone",
            path="gone.html",
            category="ROS2",
            status="planned",
            kind="generated",
        ),
    )
    index = build_artifact_index(entries, root=tmp_path)
    html = format_artifact_index_html(index)
    assert "(missing)" in html
    assert 'href="gone.html"' not in html


def test_cli_report_index_help() -> None:
    result = CliRunner().invoke(app, ["report", "index", "--help"])
    assert result.exit_code == 0
    assert "--out" in result.output
    assert "--html-out" in result.output
    assert "--strict" in result.output


def test_cli_report_index_writes_artifacts(tmp_path: Path) -> None:
    json_out = tmp_path / "artifact_index.json"
    html_out = tmp_path / "artifact_index.html"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "index",
            "--root",
            str(REPO_ROOT),
            "--out",
            str(json_out),
            "--html-out",
            str(html_out),
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(json_out.read_text(encoding="utf-8"))
    assert payload["count"] == len(DEFAULT_ARTIFACTS)
    assert "<table>" in html_out.read_text(encoding="utf-8")
