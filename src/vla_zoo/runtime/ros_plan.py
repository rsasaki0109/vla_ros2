from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import PurePath
from shlex import quote
from urllib.parse import urlparse

from vla_zoo.runtime.server_plan import build_server_plan


@dataclass(frozen=True)
class RosRemoteSmokePlan:
    model_name: str
    remote_url: str
    output_dir: str
    duration_sec: float
    server_command: tuple[str, ...]
    launch_command: tuple[str, ...]
    dashboard_command: tuple[str, ...]
    action_trace_command: tuple[str, ...]
    action_analysis_command: tuple[str, ...]
    bundle_command: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "server_command",
            "launch_command",
            "dashboard_command",
            "action_trace_command",
            "action_analysis_command",
            "bundle_command",
        ):
            payload[f"{key}_shell"] = shell_join(payload[key])
        return payload


def shell_join(command: tuple[str, ...] | list[str]) -> str:
    return " ".join(quote(str(part)) for part in command)


def _remote_host_and_port(remote_url: str) -> tuple[str, int]:
    parsed = urlparse(remote_url)
    host = parsed.hostname or "gpu-box"
    if parsed.port is not None:
        port = parsed.port
    elif parsed.scheme == "https":
        port = 443
    else:
        port = 80
    return host, port


def build_ros_remote_smoke_plan(
    *,
    model_name: str = "openvla",
    remote_url: str = "http://gpu-box:8001",
    output_dir: str = "results/ros2_remote_smoke",
    duration_sec: float = 30.0,
    dtype: str | None = "bfloat16",
    instruction: str = "pick up the red block",
    task_id: str = "ros2_remote_smoke_pick_red_block",
    publish_actions_in_dry_run: bool = True,
) -> RosRemoteSmokePlan:
    """Build commands for a ROS2 remote runtime smoke recording."""

    host, port = _remote_host_and_port(remote_url)
    server_plan = build_server_plan([model_name], public_host=host, base_port=port, dtype=dtype)
    output = PurePath(output_dir)
    launch_command = (
        "timeout",
        f"{duration_sec:g}s",
        "ros2",
        "launch",
        "vla_zoo",
        "remote_smoke_record.launch.py",
        f"model_name:={model_name}",
        f"remote_url:={remote_url}",
        f"output_dir:={output_dir}",
        f"instruction:={instruction}",
        f"task_id:={task_id}",
        "dry_run:=true",
        f"publish_actions_in_dry_run:={str(publish_actions_in_dry_run).lower()}",
    )
    status_log = str(output / "vla_status.jsonl")
    diagnostics_log = str(output / "vla_diagnostics.jsonl")
    action_log = str(output / "vla_actions.jsonl")
    return RosRemoteSmokePlan(
        model_name=model_name,
        remote_url=remote_url,
        output_dir=output_dir,
        duration_sec=duration_sec,
        server_command=server_plan.entries[0].command,
        launch_command=launch_command,
        dashboard_command=(
            "vla-zoo",
            "compare",
            "dashboard",
            "--status-log",
            status_log,
            "--diagnostics-log",
            diagnostics_log,
            "--out",
            str(output / "dashboard.html"),
            "--title",
            f"vla_zoo ROS2 Remote Smoke: {model_name}",
        ),
        action_trace_command=(
            "vla-zoo",
            "ros",
            "action-trace",
            "--action-log",
            action_log,
            "--out",
            str(output / "action_trace.html"),
            "--title",
            f"vla_zoo ROS2 Remote Smoke: {model_name} Actions",
        ),
        action_analysis_command=(
            "vla-zoo",
            "ros",
            "action-analyze",
            "--action-log",
            action_log,
            "--out",
            str(output / "action_analysis.json"),
            "--markdown-out",
            str(output / "action_analysis.md"),
            "--title",
            f"vla_zoo ROS2 Remote Smoke: {model_name} Action Analysis",
        ),
        bundle_command=(
            "vla-zoo",
            "report",
            "bundle",
            "--status-log",
            status_log,
            "--diagnostics-log",
            diagnostics_log,
            "--out",
            str(output / "report_bundle.zip"),
            "--title",
            f"vla_zoo ROS2 Remote Smoke: {model_name}",
        ),
    )


def format_ros_remote_smoke_plan_markdown(plan: RosRemoteSmokePlan) -> str:
    return "\n".join(
        [
            "# ROS2 Remote Smoke Plan",
            "",
            "This plan runs heavyweight VLA inference on a GPU server while the ROS2",
            "runtime node stays on the robot-side process. It is dry-run by default and",
            "records status, diagnostics, and typed action messages for reports.",
            "",
            "## 1. GPU Server",
            "",
            "```bash",
            shell_join(plan.server_command),
            "```",
            "",
            "## 2. ROS2 Runtime Recording",
            "",
            "Run for the requested duration with `timeout`; remove the prefix to stop manually.",
            "",
            "```bash",
            shell_join(plan.launch_command),
            "```",
            "",
            "## 3. Reports",
            "",
            "```bash",
            shell_join(plan.dashboard_command),
            shell_join(plan.action_trace_command),
            shell_join(plan.action_analysis_command),
            shell_join(plan.bundle_command),
            "```",
            "",
            "## Settings",
            "",
            f"- model: `{plan.model_name}`",
            f"- remote_url: `{plan.remote_url}`",
            f"- output_dir: `{plan.output_dir}`",
            f"- suggested_duration_sec: `{plan.duration_sec:g}`",
            "",
        ]
    )
