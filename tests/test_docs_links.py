from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vla_zoo.cli.main import app
from vla_zoo.docs.links import (
    check_paths,
    extract_links,
    format_link_report_table,
    link_report_payload,
)


def test_extract_links_handles_markdown_and_html() -> None:
    text = (
        "[doc](report.md) and ![gif](assets/demo.gif) plus "
        '<a href="page.html">x</a> <img src="pic.png"> '
        "[ext](https://example.com) [titled](dir/file.json \"Title\") "
        "[anchor](#section)"
    )
    links = extract_links(text)
    assert "report.md" in links
    assert "assets/demo.gif" in links
    assert "page.html" in links
    assert "pic.png" in links
    assert "https://example.com" in links
    assert 'dir/file.json "Title"' in links
    assert "#section" in links


def test_check_paths_flags_missing_and_skips_external(tmp_path: Path) -> None:
    (tmp_path / "exists.md").write_text("ok", encoding="utf-8")
    page = tmp_path / "index.md"
    page.write_text(
        "[good](exists.md) [bad](missing.html) "
        "[ext](https://example.com) [anchor](#top)",
        encoding="utf-8",
    )

    report = check_paths([page], root=tmp_path)

    assert report.checked == 2  # good + bad, external/anchor excluded
    assert report.ok == 1
    assert report.skipped_external == 1
    broken = report.broken
    assert len(broken) == 1
    assert broken[0].link == "missing.html"
    assert report.ok_overall is False


def test_check_paths_resolves_root_rooted_links(tmp_path: Path) -> None:
    (tmp_path / "docs").mkdir()
    (tmp_path / "top.md").write_text("root target", encoding="utf-8")
    page = tmp_path / "docs" / "index.html"
    page.write_text('<a href="/top.md">home</a>', encoding="utf-8")

    report = check_paths([page], root=tmp_path)

    assert report.ok_overall is True
    assert report.ok == 1


def test_check_paths_reports_missing_source(tmp_path: Path) -> None:
    report = check_paths([tmp_path / "nope.md"], root=tmp_path)

    assert report.ok_overall is False
    assert report.broken[0].status == "source_missing"


def test_relative_links_resolve_against_source_dir(tmp_path: Path) -> None:
    (tmp_path / "docs" / "assets").mkdir(parents=True)
    (tmp_path / "docs" / "assets" / "gallery.html").write_text("g", encoding="utf-8")
    page = tmp_path / "docs" / "index.html"
    page.write_text('<a href="assets/gallery.html">g</a>', encoding="utf-8")

    report = check_paths([page], root=tmp_path)

    assert report.ok_overall is True


def test_report_payload_and_table(tmp_path: Path) -> None:
    page = tmp_path / "index.md"
    page.write_text("[bad](missing.md)", encoding="utf-8")
    report = check_paths([page], root=tmp_path)

    payload = link_report_payload(report)
    assert payload["broken"] == 1
    assert payload["ok_overall"] is False
    assert json.dumps(payload)  # serializable

    table = format_link_report_table(report)
    assert "missing" in table
    assert "broken=1" in table


def test_cli_report_link_check_strict(tmp_path: Path) -> None:
    (tmp_path / "exists.md").write_text("ok", encoding="utf-8")
    good = tmp_path / "good.md"
    good.write_text("[ok](exists.md)", encoding="utf-8")
    bad = tmp_path / "bad.md"
    bad.write_text("[broken](missing.md)", encoding="utf-8")

    runner = CliRunner()
    ok_result = runner.invoke(
        app,
        ["report", "link-check", "--paths", str(good), "--root", str(tmp_path), "--strict"],
    )
    assert ok_result.exit_code == 0

    bad_result = runner.invoke(
        app,
        ["report", "link-check", "--paths", str(bad), "--root", str(tmp_path), "--strict"],
    )
    assert bad_result.exit_code == 1


def test_cli_report_link_check_json_out(tmp_path: Path) -> None:
    (tmp_path / "exists.md").write_text("ok", encoding="utf-8")
    page = tmp_path / "page.md"
    page.write_text("[ok](exists.md)", encoding="utf-8")
    out = tmp_path / "report.json"

    result = CliRunner().invoke(
        app,
        [
            "report",
            "link-check",
            "--paths",
            str(page),
            "--root",
            str(tmp_path),
            "--out",
            str(out),
            "--json",
        ],
    )
    assert result.exit_code == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["ok"] == 1
    assert payload["broken"] == 0
