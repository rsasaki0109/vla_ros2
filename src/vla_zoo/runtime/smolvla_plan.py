from __future__ import annotations

from dataclasses import asdict, dataclass

from vla_zoo.runtime.ros_plan import build_ros_remote_smoke_plan, shell_join
from vla_zoo.runtime.server_plan import build_server_plan

DEFAULT_PRETRAINED = "lerobot/smolvla_base"
DEFAULT_VENV_DIR = ".venv-smolvla"
SMOLVLA_EXTRAS = "cli,server,smolvla"


@dataclass(frozen=True)
class SmolVLARemotePlan:
    """Reproducible bring-up plan for a remote SmolVLA server.

    This is a command plan plus an environment-isolation guide, not a recorded run
    and not a claim that SmolVLA reaches any task-success quality.
    """

    model_name: str
    pretrained: str
    host: str
    public_host: str
    port: int
    public_url: str
    device: str
    dtype: str | None
    venv_dir: str
    extras: str
    instruction: str
    env_create_command: tuple[str, ...]
    install_command: tuple[str, ...]
    server_command: tuple[str, ...]
    health_command: tuple[str, ...]
    predict_probe_command: tuple[str, ...]
    compare_command: tuple[str, ...]
    ros_plan_command: tuple[str, ...]
    caveat: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        for key in (
            "env_create_command",
            "install_command",
            "server_command",
            "health_command",
            "predict_probe_command",
            "compare_command",
            "ros_plan_command",
        ):
            payload[key] = list(getattr(self, key))
            payload[f"{key}_shell"] = shell_join(getattr(self, key))
        return payload


def build_smolvla_remote_plan(
    *,
    pretrained: str = DEFAULT_PRETRAINED,
    host: str = "0.0.0.0",
    public_host: str = "gpu-box",
    port: int = 8000,
    device: str = "cuda:0",
    dtype: str | None = None,
    venv_dir: str = DEFAULT_VENV_DIR,
    instruction: str = "pick up the red block",
) -> SmolVLARemotePlan:
    """Build an isolated-environment remote SmolVLA serving plan."""

    public_url = f"http://{public_host}:{port}"
    server_plan = build_server_plan(
        ["smolvla"],
        host=host,
        public_host=public_host,
        base_port=port,
        device=device,
        dtype=dtype,
        pretrained={"smolvla": pretrained},
    )
    server_command = server_plan.entries[0].command
    pip = f"{venv_dir}/bin/pip"
    python = f"{venv_dir}/bin/python"
    ros_plan = build_ros_remote_smoke_plan(model_name="smolvla", remote_url=public_url)
    return SmolVLARemotePlan(
        model_name="smolvla",
        pretrained=pretrained,
        host=host,
        public_host=public_host,
        port=port,
        public_url=public_url,
        device=device,
        dtype=dtype,
        venv_dir=venv_dir,
        extras=SMOLVLA_EXTRAS,
        instruction=instruction,
        env_create_command=("python3", "-m", "venv", venv_dir),
        install_command=(pip, "install", "-e", f".[{SMOLVLA_EXTRAS}]"),
        server_command=server_command,
        health_command=("curl", "-fsS", f"{public_url}/health"),
        predict_probe_command=(
            python,
            "examples/python/smolvla_remote_client.py",
            "--remote-url",
            public_url,
            "--instruction",
            instruction,
        ),
        compare_command=(
            "vla-zoo",
            "compare",
            "pybullet",
            "--models",
            "smolvla",
            "--runtime",
            "remote",
            "--remote-map",
            f"smolvla={public_url}",
        ),
        ros_plan_command=ros_plan.smoke_report_command,
        caveat=(
            "SmolVLA needs the pinned lerobot[smolvla] stack, which conflicts with the "
            "openvla extra; keep it in a dedicated virtual environment. This plan records "
            "no /v1/predict response and makes no policy-quality or task-success claim."
        ),
    )


def format_smolvla_remote_plan_markdown(plan: SmolVLARemotePlan) -> str:
    return "\n".join(
        [
            "# SmolVLA Remote Serving Plan",
            "",
            "Run LeRobot SmolVLA as a remote inference server in an isolated environment,",
            "then consume it from the robot-side runtime through `runtime=remote`. This file",
            "is a reproducible bring-up plan, not a recorded run, and makes no claim about",
            "SmolVLA task-success quality.",
            "",
            "## 1. Isolated Environment",
            "",
            "`lerobot[smolvla]` pins specific `transformers`/`torch` versions that clash with",
            "the `openvla` extra, so install it in a dedicated virtual environment:",
            "",
            "```bash",
            shell_join(plan.env_create_command),
            shell_join(plan.install_command),
            "```",
            "",
            "## 2. SmolVLA Server (GPU box)",
            "",
            "```bash",
            shell_join(plan.server_command),
            "```",
            "",
            "Confirm readiness before sending requests:",
            "",
            "```bash",
            shell_join(plan.health_command),
            "```",
            "",
            "## 3. Robot-Side Consumption",
            "",
            "```bash",
            shell_join(plan.predict_probe_command),
            shell_join(plan.compare_command),
            "```",
            "",
            "Generate a matching ROS2 remote smoke recording plan:",
            "",
            "```bash",
            shell_join(plan.ros_plan_command),
            "```",
            "",
            "## Settings",
            "",
            f"- model: `{plan.model_name}`",
            f"- pretrained: `{plan.pretrained}`",
            f"- public_url: `{plan.public_url}`",
            f"- device: `{plan.device}`",
            f"- dtype: `{plan.dtype if plan.dtype is not None else '-'}`",
            f"- venv_dir: `{plan.venv_dir}`",
            f"- extras: `{plan.extras}`",
            "",
            "## Caveat",
            "",
            plan.caveat,
            "",
        ]
    )
