from __future__ import annotations

import json
from pathlib import Path

from vla_zoo.runtime.ros_smoke import (
    check_ros_remote_smoke,
    format_ros_remote_smoke_check_markdown,
)


def _write_jsonl(path: Path, *rows: dict[str, object]) -> None:
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def test_ros_remote_smoke_check_accepts_remote_runtime_logs(tmp_path: Path) -> None:
    action_log = tmp_path / "actions.jsonl"
    status_log = tmp_path / "status.jsonl"
    diagnostics_log = tmp_path / "diagnostics.jsonl"
    _write_jsonl(
        action_log,
        {
            "model_name": "dummy",
            "adapter_name": "RemoteVLAClient",
            "action_space": "eef_delta",
            "data": [0.0] * 7,
        },
    )
    _write_jsonl(
        status_log,
        {
            "model_name": "dummy",
            "adapter_name": "RemoteVLAClient",
            "ready": True,
            "dry_run": True,
            "last_latency_ms": 9.5,
            "status_text": "ready",
            "metadata": {"runtime": "remote", "remote_url": "http://127.0.0.1:8766"},
        },
    )
    _write_jsonl(
        diagnostics_log,
        {
            "status": [
                {
                    "level": 0,
                    "values": [
                        {"key": "runtime", "value": "remote"},
                        {"key": "remote_url", "value": "http://127.0.0.1:8766"},
                    ],
                }
            ]
        },
    )

    check = check_ros_remote_smoke(
        action_log=action_log,
        status_log=status_log,
        diagnostics_log=diagnostics_log,
        expected_model="dummy",
        expected_remote_url="http://127.0.0.1:8766",
    )
    markdown = format_ros_remote_smoke_check_markdown(check)

    assert check.ok
    assert check.remote_action_count == 1
    assert check.remote_status_count == 1
    assert check.remote_diagnostics_count == 1
    assert "`RemoteVLAClient`" in markdown


def test_ros_remote_smoke_check_reports_missing_remote_url(tmp_path: Path) -> None:
    status_log = tmp_path / "status.jsonl"
    _write_jsonl(
        status_log,
        {
            "model_name": "dummy",
            "adapter_name": "RemoteVLAClient",
            "ready": True,
            "dry_run": True,
            "metadata": {"runtime": "remote", "remote_url": "http://wrong:8000"},
        },
    )

    check = check_ros_remote_smoke(
        action_log=tmp_path / "missing_actions.jsonl",
        status_log=status_log,
        diagnostics_log=tmp_path / "missing_diagnostics.jsonl",
        expected_model="dummy",
        expected_remote_url="http://127.0.0.1:8766",
    )

    assert not check.ok
    assert any("remote_url" in issue for issue in check.issues)
