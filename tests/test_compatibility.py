from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from vla_zoo.cli.main import app
from vla_zoo.compare.compatibility import (
    compatibility_matrix,
    format_compatibility_markdown,
    get_robot_profile,
)


def test_openvla_fits_single_camera_eef_profile() -> None:
    robot = get_robot_profile("single-camera-eef")
    results = compatibility_matrix(robot=robot, models=("openvla",))

    assert len(results) == 1
    result = results[0]
    assert result.model == "openvla"
    assert result.status == "compatible"
    assert result.compatible
    assert result.score == 100


def test_smolvla_is_blocked_for_single_camera_no_state_profile() -> None:
    robot = get_robot_profile("single-camera-eef")
    result = compatibility_matrix(robot=robot, models=("smolvla",))[0]
    issue_codes = {issue.code for issue in result.issues}

    assert result.status == "blocked"
    assert "camera_count" in issue_codes
    assert "state_required" in issue_codes
    assert "action_space" in issue_codes


def test_pi0_fits_multi_camera_state_profile_with_review() -> None:
    robot = get_robot_profile("multi-camera-state-arm")
    result = compatibility_matrix(robot=robot, models=("pi0",))[0]

    assert result.status in {"compatible", "needs_review"}
    assert result.score >= 90
    assert result.compatible
    assert result.recommendations


def test_groot_domain_blocks_manipulation_profile() -> None:
    robot = get_robot_profile("multi-camera-state-arm")
    result = compatibility_matrix(robot=robot, models=("groot",))[0]

    assert result.status == "blocked"
    assert any(issue.code == "domain" for issue in result.issues)


def test_compatibility_markdown_is_report_ready() -> None:
    robot = get_robot_profile("single-camera-eef")
    results = compatibility_matrix(robot=robot, models=("openvla", "smolvla"))
    markdown = format_compatibility_markdown(results, robot=robot)

    assert "VLA Robot Compatibility" in markdown
    assert "single-camera-eef" in markdown
    assert "`openvla`" in markdown
    assert "`smolvla`" in markdown
    assert "deployment-shape check" in markdown


def test_cli_compare_compatibility_json() -> None:
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "compatibility",
            "--robot-profile",
            "single-camera-eef",
            "--models",
            "openvla,smolvla",
            "--json",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.output)
    assert payload["robot_profile"]["name"] == "single-camera-eef"
    by_model = {item["model"]: item for item in payload["results"]}
    assert by_model["openvla"]["status"] == "compatible"
    assert by_model["smolvla"]["status"] == "blocked"


def test_cli_compare_compatibility_writes_markdown(tmp_path: Path) -> None:
    out = tmp_path / "compatibility.md"
    result = CliRunner().invoke(
        app,
        [
            "compare",
            "compatibility",
            "--models",
            "openvla",
            "--markdown-out",
            str(out),
        ],
    )

    assert result.exit_code == 0
    assert "Markdown written" in result.output
    assert "`openvla`" in out.read_text(encoding="utf-8")


def test_cli_compare_compatibility_lists_profiles() -> None:
    result = CliRunner().invoke(app, ["compare", "compatibility", "--list-profiles"])

    assert result.exit_code == 0
    assert "single-camera-eef" in result.output
    assert "multi-camera-state-arm" in result.output
