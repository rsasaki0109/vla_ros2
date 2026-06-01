from __future__ import annotations

from pathlib import Path

from vla_zoo.runtime.ros_plan import (
    build_ros_remote_smoke_plan,
    format_ros_remote_smoke_plan_markdown,
)


def test_ros_remote_smoke_plan_uses_remote_launch_and_reports() -> None:
    plan = build_ros_remote_smoke_plan(
        model_name="pi0",
        remote_url="http://gpu-box:8002",
        output_dir="results/ros2_pi0",
    )

    assert "remote_smoke_record.launch.py" in plan.launch_command
    assert "model_name:=pi0" in plan.launch_command
    assert "remote_url:=http://gpu-box:8002" in plan.launch_command
    assert "vla-zoo" in plan.server_command
    assert "serve" in plan.server_command
    assert "pi0" in plan.server_command
    assert "--device" in plan.server_command
    assert "results/ros2_pi0/vla_status.jsonl" in plan.dashboard_command
    assert "results/ros2_pi0/vla_actions.jsonl" in plan.action_trace_command


def test_ros_remote_smoke_plan_markdown_is_dry_run_oriented() -> None:
    plan = build_ros_remote_smoke_plan()
    markdown = format_ros_remote_smoke_plan_markdown(plan)

    assert "--dtype bfloat16" in markdown
    assert "dry-run" in markdown
    assert "GPU Server" in markdown
    assert "ROS2 Runtime Recording" in markdown
    assert "vla-zoo report bundle" in markdown


def test_remote_smoke_record_launch_exists() -> None:
    path = Path("ros2/vla_zoo/launch/remote_smoke_record.launch.py")

    text = path.read_text(encoding="utf-8")
    assert "remote_url" in text
    assert "vla_smoke_input_node" in text
    assert "vla_runtime_recorder" in text
