from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Literal

from vla_zoo.compare.profiles import MethodProfile, method_profiles
from vla_zoo.core.registry import AdapterInfo, get_adapter_info
from vla_zoo.core.types import ActionSpace

IssueLevel = Literal["warning", "error"]


@dataclass(frozen=True)
class RobotProfile:
    """Robot-side capabilities used to check adapter fit before loading weights."""

    name: str
    description: str
    camera_count: int
    action_spaces: tuple[ActionSpace, ...]
    has_instruction: bool = True
    has_state: bool = False
    supports_action_chunks: bool = False
    control_hz: float | None = None
    domains: tuple[str, ...] = ("manipulation",)
    notes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class CompatibilityIssue:
    level: IssueLevel
    code: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterCompatibility:
    model: str
    robot_profile: str
    status: Literal["compatible", "needs_review", "blocked"]
    adapter_status: str
    score: int
    action_space: str
    action_shape: str
    local_runtime: str
    remote_runtime: str
    issues: tuple[CompatibilityIssue, ...]
    recommendations: tuple[str, ...]

    @property
    def compatible(self) -> bool:
        return self.status != "blocked"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["compatible"] = self.compatible
        return payload


ROBOT_PROFILE_PRESETS: dict[str, RobotProfile] = {
    "dry-run-arm": RobotProfile(
        name="dry-run-arm",
        description="One optional camera, no proprioception, neutral EEF dry-run target.",
        camera_count=1,
        action_spaces=("eef_delta",),
        has_state=False,
        supports_action_chunks=False,
        control_hz=5.0,
        notes=("Useful for CI, docs, and ROS2 dry-run launch validation.",),
    ),
    "single-camera-eef": RobotProfile(
        name="single-camera-eef",
        description="Single RGB camera manipulator exposing an end-effector action bridge.",
        camera_count=1,
        action_spaces=("eef_delta", "eef_pose", "gripper"),
        has_state=False,
        supports_action_chunks=False,
        control_hz=5.0,
        notes=("Closest lightweight fit for OpenVLA-style single-image policies.",),
    ),
    "multi-camera-state-arm": RobotProfile(
        name="multi-camera-state-arm",
        description="Manipulator with multiple cameras, robot state, and a custom action bridge.",
        camera_count=3,
        action_spaces=("eef_delta", "eef_pose", "joint_position", "gripper", "custom"),
        has_state=True,
        supports_action_chunks=True,
        control_hz=10.0,
        notes=("Useful target for LeRobot/SmolVLA and remote action-chunk policies.",),
    ),
    "mobile-base": RobotProfile(
        name="mobile-base",
        description="Mobile robot base accepting twist commands.",
        camera_count=1,
        action_spaces=("base_twist", "custom"),
        has_state=False,
        supports_action_chunks=False,
        control_hz=10.0,
        domains=("mobile",),
        notes=("Manipulation VLAs need a separate bridge before this profile is viable.",),
    ),
    "humanoid-generalist": RobotProfile(
        name="humanoid-generalist",
        description="Humanoid/generalist target with multimodal observations and custom actions.",
        camera_count=4,
        action_spaces=("custom", "joint_position", "joint_velocity", "eef_delta", "gripper"),
        has_state=True,
        supports_action_chunks=True,
        control_hz=20.0,
        domains=("humanoid/generalist", "manipulation"),
        notes=("Still requires a hardware-specific bridge and safety controller.",),
    ),
}


def get_robot_profile(name: str) -> RobotProfile:
    """Return a built-in robot profile by name."""

    normalized = name.strip().lower().replace("_", "-")
    try:
        return ROBOT_PROFILE_PRESETS[normalized]
    except KeyError as exc:
        choices = ", ".join(sorted(ROBOT_PROFILE_PRESETS))
        msg = f"Unknown robot profile {name!r}. Available profiles: {choices}"
        raise ValueError(msg) from exc


def _joined_requirements(profile: MethodProfile) -> str:
    return " ".join((*profile.input_requirements, profile.proprioception)).lower()


def _required_camera_count(profile: MethodProfile) -> int:
    requirements = _joined_requirements(profile)
    if "image optional" in requirements:
        return 0
    if "multi-camera" in requirements or "multimodal observations" in requirements:
        return 2
    if "single rgb image" in requirements or "image" in requirements or "camera" in requirements:
        return 1
    return 0


def _state_requirement(profile: MethodProfile) -> Literal["none", "expected", "required"]:
    requirements = _joined_requirements(profile)
    if "not required" in requirements or "state optional" in requirements:
        return "none"
    if "required" in requirements:
        return "required"
    if "expected" in requirements or "robot state" in requirements:
        return "expected"
    return "none"


def _requires_instruction(profile: MethodProfile) -> bool:
    requirements = _joined_requirements(profile)
    return (
        "instruction" in requirements
        or "language" in requirements
        or "task context" in requirements
    )


def _requires_action_chunks(profile: MethodProfile) -> bool:
    chunks = profile.action_chunks.lower()
    return "expected" in chunks or "required" in chunks


def _score(issues: Sequence[CompatibilityIssue]) -> int:
    penalty = 0
    for issue in issues:
        penalty += 35 if issue.level == "error" else 10
    return max(0, 100 - penalty)


def _status(
    issues: Sequence[CompatibilityIssue],
) -> Literal["compatible", "needs_review", "blocked"]:
    if any(issue.level == "error" for issue in issues):
        return "blocked"
    if issues:
        return "needs_review"
    return "compatible"


def _recommendations(
    profile: MethodProfile,
    issues: Sequence[CompatibilityIssue],
) -> tuple[str, ...]:
    recommendations: list[str] = []
    codes = {issue.code for issue in issues}
    if "camera_count" in codes:
        recommendations.append("Add the required camera streams or remap the adapter image inputs.")
    if "state_required" in codes or "state_expected" in codes:
        recommendations.append(
            "Provide proprioception in VLAObservation.state or ROS joint state inputs."
        )
    if "action_space" in codes:
        recommendations.append("Add a robot-specific action bridge for the adapter action space.")
    if "action_chunks" in codes:
        recommendations.append(
            "Enable chunk scheduling or consume VLAActionChunk in the downstream bridge."
        )
    if "domain" in codes:
        recommendations.append(
            "Use a robot profile that matches the adapter domain before benchmarking."
        )
    if profile.remote_runtime.lower().startswith("recommended"):
        recommendations.append(
            "Run the adapter behind a remote GPU server for robot-side ROS2 deployment."
        )
    if not recommendations:
        recommendations.append(
            "Run a dry-run ROS2 smoke test before connecting any hardware bridge."
        )
    return tuple(recommendations)


def check_adapter_compatibility(
    profile: MethodProfile,
    robot: RobotProfile,
) -> AdapterCompatibility:
    """Check one adapter method profile against one robot-side capability profile."""

    issues: list[CompatibilityIssue] = []
    required_cameras = _required_camera_count(profile)
    if robot.camera_count < required_cameras:
        issues.append(
            CompatibilityIssue(
                level="error",
                code="camera_count",
                message=(
                    f"adapter expects at least {required_cameras} camera stream(s); "
                    f"robot profile declares {robot.camera_count}"
                ),
            )
        )

    if _requires_instruction(profile) and not robot.has_instruction:
        issues.append(
            CompatibilityIssue(
                level="error",
                code="instruction",
                message=(
                    "adapter expects language/task input but robot profile disables instruction"
                ),
            )
        )

    state_requirement = _state_requirement(profile)
    if state_requirement == "required" and not robot.has_state:
        issues.append(
            CompatibilityIssue(
                level="error",
                code="state_required",
                message="adapter requires robot state/proprioception",
            )
        )
    elif state_requirement == "expected" and not robot.has_state:
        issues.append(
            CompatibilityIssue(
                level="warning",
                code="state_expected",
                message=(
                    "adapter expects robot state; output quality or schema may be invalid "
                    "without it"
                ),
            )
        )

    action_space = profile.action_space
    if action_space not in robot.action_spaces:
        issues.append(
            CompatibilityIssue(
                level="error",
                code="action_space",
                message=(
                    f"adapter outputs {action_space!r}; robot profile supports "
                    f"{', '.join(robot.action_spaces)}"
                ),
            )
        )

    if _requires_action_chunks(profile) and not robot.supports_action_chunks:
        issues.append(
            CompatibilityIssue(
                level="error",
                code="action_chunks",
                message="adapter expects action chunks but robot profile consumes single actions",
            )
        )

    if profile.domain and profile.domain not in robot.domains:
        issues.append(
            CompatibilityIssue(
                level="error",
                code="domain",
                message=(
                    f"adapter domain {profile.domain!r} does not match robot domains "
                    f"{', '.join(robot.domains)}"
                ),
            )
        )

    status = _status(issues)
    return AdapterCompatibility(
        model=profile.name,
        robot_profile=robot.name,
        status=status,
        adapter_status=profile.status,
        score=_score(issues),
        action_space=profile.action_space,
        action_shape=profile.action_shape,
        local_runtime=profile.local_runtime,
        remote_runtime=profile.remote_runtime,
        issues=tuple(issues),
        recommendations=_recommendations(profile, issues),
    )


def compatibility_matrix(
    *,
    robot: RobotProfile,
    models: Sequence[str] | None = None,
    status_provider: Callable[[str], str] | None = None,
) -> list[AdapterCompatibility]:
    """Check a robot profile against a set of registered adapter names."""

    adapters: list[AdapterInfo] | None = None
    if models is not None:
        adapters = [get_adapter_info(model) for model in models]
    profiles = method_profiles(adapters=adapters, status_provider=status_provider)
    return [check_adapter_compatibility(profile, robot) for profile in profiles]


def format_robot_profiles_markdown() -> str:
    lines = [
        "## Robot Profiles",
        "",
        "| Profile | Cameras | State | Chunks | Actions | Domains |",
        "|---|---:|---|---|---|---|",
    ]
    for profile in ROBOT_PROFILE_PRESETS.values():
        lines.append(
            f"| `{profile.name}` | {profile.camera_count} | {profile.has_state} | "
            f"{profile.supports_action_chunks} | {', '.join(profile.action_spaces)} | "
            f"{', '.join(profile.domains)} |"
        )
    return "\n".join(lines) + "\n"


def format_compatibility_markdown(
    results: Sequence[AdapterCompatibility],
    *,
    robot: RobotProfile,
    title: str = "VLA Robot Compatibility",
) -> str:
    lines = [
        f"## {title}",
        "",
        f"Robot profile: `{robot.name}`",
        "",
        f"- cameras: {robot.camera_count}",
        f"- state: {robot.has_state}",
        f"- action chunks: {robot.supports_action_chunks}",
        f"- action spaces: {', '.join(robot.action_spaces)}",
        f"- domains: {', '.join(robot.domains)}",
        "",
        "| Model | Fit | Adapter status | Score | Action | Issues | Next step |",
        "|---|---|---|---:|---|---|---|",
    ]
    for result in results:
        issues = "<br>".join(f"{issue.level}: {issue.message}" for issue in result.issues) or "-"
        next_step = result.recommendations[0] if result.recommendations else "-"
        action = f"{result.action_space} {result.action_shape}"
        lines.append(
            f"| `{result.model}` | {result.status} | {result.adapter_status} | "
            f"{result.score} | {action} | {issues} | {next_step} |"
        )
    lines.extend(
        [
            "",
            "This is a deployment-shape check. It does not validate model quality, "
            "calibration, safety, or real robot task success.",
        ]
    )
    return "\n".join(lines) + "\n"
