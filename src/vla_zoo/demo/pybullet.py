from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from html import escape
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from vla_zoo import load_model
from vla_zoo.core.model import BaseVLA
from vla_zoo.core.types import VLAAction, VLAActionChunk, VLAObservation

DEFAULT_OUT = Path("docs/assets/simulation_pick_place.gif")
WIDTH = 960
HEIGHT = 540
FPS = 24
STEPS_PER_FRAME = 10
HEAVY_LOCAL_MODELS = frozenset({"openvla"})
CUBE_INITIAL_POSITION = (0.58, -0.16, 0.035)
CUBE_GOAL_POSITION = (0.58, 0.22, 0.035)
TASK_GOAL_TOLERANCE_M = 0.15
PHASE_ORDER = (
    "observe",
    "approach",
    "descend",
    "close gripper",
    "lift",
    "transport",
    "place",
    "open gripper",
    "retreat",
)


def font(size: int, *, mono: bool = False) -> Any:
    family = "DejaVuSansMono.ttf" if mono else "DejaVuSans.ttf"
    path = Path("/usr/share/fonts/truetype/dejavu") / family
    if path.exists():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


TITLE = font(26)
SMALL = font(16)
MONO = font(15, mono=True)


@dataclass(frozen=True)
class Waypoint:
    name: str
    position: tuple[float, float, float]
    gripper: float
    steps: int
    attach: bool = False
    detach: bool = False


@dataclass(frozen=True)
class RenderSample:
    image: Image.Image
    phase: str
    position: tuple[float, float, float]
    cube_position: tuple[float, float, float]
    cube_goal_position: tuple[float, float, float]
    scripted_action: tuple[float, float, float, float]
    adapter_action: tuple[float, float, float, float] | None
    adapter_error: str | None
    adapter_latency_ms: float | None
    adapter_query_count: int
    adapter_query_fresh: bool
    attached: bool
    sim_time: float
    model_name: str
    runtime: str
    frame_index: int


@dataclass(frozen=True)
class PyBulletDemoConfig:
    model_name: str = "dummy"
    runtime: str = "local"
    remote_url: str = "http://localhost:8000"
    instruction: str = "pick up the red block"
    out: Path = DEFAULT_OUT
    model_call_every: int = 8
    render_stride: int = 3
    adapter_kwargs: dict[str, Any] | None = None


@dataclass(frozen=True)
class PyBulletComparisonResult:
    model_name: str
    runtime: str
    ok: bool
    remote_url: str | None = None
    frames: int = 0
    adapter_queries: int = 0
    adapter_errors: int = 0
    mean_latency_ms: float | None = None
    max_latency_ms: float | None = None
    mean_abs_action: float | None = None
    task_success: bool = False
    cube_lifted: bool = False
    cube_moved_distance: float | None = None
    final_cube_distance_to_goal: float | None = None
    grasp_attached_frames: int = 0
    phase_completion: float = 0.0
    last_error: str | None = None


@dataclass(frozen=True)
class PyBulletComparisonTarget:
    model_name: str
    runtime: str = "local"
    remote_url: str = "http://localhost:8000"
    adapter_kwargs: dict[str, Any] | None = None


def import_pybullet() -> tuple[Any, Any]:
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:
        msg = (
            "PyBullet is required to generate the real simulation GIF. "
            'Install it with: pip install "vla_zoo[sim]"'
        )
        raise RuntimeError(msg) from exc
    return p, pybullet_data


def lerp(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
    t: float,
) -> tuple[float, float, float]:
    t = max(0.0, min(1.0, t))
    t = t * t * (3.0 - 2.0 * t)
    return (
        float(a[0] + (b[0] - a[0]) * t),
        float(a[1] + (b[1] - a[1]) * t),
        float(a[2] + (b[2] - a[2]) * t),
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def distance3(
    a: tuple[float, float, float],
    b: tuple[float, float, float],
) -> float:
    return float(math.sqrt(sum((a[index] - b[index]) ** 2 for index in range(3))))


def make_model(config: PyBulletDemoConfig) -> BaseVLA:
    kwargs = dict(config.adapter_kwargs or {})
    if config.runtime == "remote":
        kwargs.setdefault("remote_url", config.remote_url)
    return load_model(config.model_name, runtime=config.runtime, **kwargs)


def prediction_to_demo_action(
    prediction: VLAAction | VLAActionChunk,
) -> tuple[float, float, float, float]:
    action = prediction.actions[0] if isinstance(prediction, VLAActionChunk) else prediction
    flat = np.asarray(action.to_numpy(), dtype=np.float32).reshape(-1)
    if flat.size >= 7:
        raw = (float(flat[0]), float(flat[1]), float(flat[2]), float(flat[6]))
    else:
        padded = np.zeros((4,), dtype=np.float32)
        padded[: min(4, flat.size)] = flat[: min(4, flat.size)]
        raw = (float(padded[0]), float(padded[1]), float(padded[2]), float(padded[3]))
    return (
        clamp(raw[0], -1.0, 1.0),
        clamp(raw[1], -1.0, 1.0),
        clamp(raw[2], -1.0, 1.0),
        clamp(raw[3], -1.0, 1.0),
    )


def predict_adapter_action(
    model: BaseVLA,
    image: Image.Image,
    config: PyBulletDemoConfig,
    *,
    phase: str,
    target: tuple[float, float, float],
    gripper: float,
    attached: bool,
    sim_time: float,
) -> tuple[tuple[float, float, float, float] | None, str | None, float | None]:
    observation = VLAObservation(
        instruction=config.instruction,
        images={"primary": image},
        state={
            "end_effector_target": target,
            "gripper_open": gripper,
            "attached": attached,
        },
        timestamp=sim_time,
        metadata={
            "demo": "pybullet",
            "phase": phase,
            "runtime": config.runtime,
        },
    )
    try:
        start = perf_counter()
        prediction = model.predict(observation=observation)
        latency_ms = (perf_counter() - start) * 1000.0
        return prediction_to_demo_action(prediction), None, latency_ms
    except Exception as exc:
        return None, str(exc), None


def setup_world(p: Any, pybullet_data: Any) -> tuple[int, int, int]:
    p.connect(p.DIRECT)
    p.resetSimulation()
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.setTimeStep(1.0 / 240.0)

    p.loadURDF("plane.urdf")
    table = p.loadURDF("table/table.urdf", [0.62, 0.0, -0.63], [0, 0, 0, 1])
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], [0, 0, 0, 1], useFixedBase=True)

    cube_visual = p.createVisualShape(
        p.GEOM_BOX,
        halfExtents=[0.035, 0.035, 0.035],
        rgbaColor=[0.9, 0.06, 0.06, 1.0],
    )
    cube_collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.035, 0.035, 0.035])
    cube = p.createMultiBody(
        baseMass=0.08,
        baseCollisionShapeIndex=cube_collision,
        baseVisualShapeIndex=cube_visual,
        basePosition=list(CUBE_INITIAL_POSITION),
        baseOrientation=[0, 0, 0, 1],
    )
    p.changeDynamics(cube, -1, lateralFriction=1.2, spinningFriction=0.02, rollingFriction=0.02)
    p.changeDynamics(table, -1, lateralFriction=1.0)
    return robot, cube, table


def set_initial_pose(p: Any, robot: int) -> None:
    rest = [0.0, -0.55, 0.0, -2.35, 0.0, 1.9, 0.8]
    for joint, value in enumerate(rest):
        p.resetJointState(robot, joint, value)
    p.resetJointState(robot, 9, 0.04)
    p.resetJointState(robot, 10, 0.04)


def control_robot(
    p: Any,
    robot: int,
    target: tuple[float, float, float],
    gripper: float,
    orientation: tuple[float, float, float, float],
) -> None:
    joint_positions = p.calculateInverseKinematics(
        robot,
        11,
        target,
        targetOrientation=orientation,
        maxNumIterations=80,
        residualThreshold=1e-4,
    )
    for joint in range(7):
        p.setJointMotorControl2(
            robot,
            joint,
            p.POSITION_CONTROL,
            targetPosition=joint_positions[joint],
            force=220,
            maxVelocity=1.5,
        )
    finger = 0.04 * clamp(gripper, 0.0, 1.0)
    for joint in (9, 10):
        p.setJointMotorControl2(
            robot,
            joint,
            p.POSITION_CONTROL,
            targetPosition=finger,
            force=80,
            maxVelocity=0.3,
        )


def render_camera(p: Any) -> Image.Image:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.48, 0.02, 0.28],
        distance=1.85,
        yaw=48,
        pitch=-30,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=48, aspect=WIDTH / HEIGHT, nearVal=0.02, farVal=5.0)
    _, _, rgba, _, _ = p.getCameraImage(
        WIDTH,
        HEIGHT,
        view,
        proj,
        renderer=p.ER_TINY_RENDERER,
    )
    array = np.asarray(rgba, dtype=np.uint8).reshape((HEIGHT, WIDTH, 4))
    return Image.fromarray(array[:, :, :3], "RGB")


def overlay(sample: RenderSample) -> Image.Image:
    image = sample.image.copy()
    draw = ImageDraw.Draw(image, "RGBA")
    draw.rounded_rectangle((24, 20, 544, 142), radius=18, fill=(8, 12, 20, 215))
    draw.text((42, 34), "vla_zoo PyBullet simulation", fill=(245, 247, 250), font=TITLE)
    draw.text(
        (44, 72),
        f"adapter={sample.model_name} runtime={sample.runtime}",
        fill=(188, 199, 216),
        font=SMALL,
    )
    draw.text(
        (44, 96),
        f"phase={sample.phase}  t={sample.sim_time:0.2f}s  attached={sample.attached}",
        fill=(125, 211, 252),
        font=SMALL,
    )
    draw.text(
        (44, 120),
        "Franka Panda URDF + cube + gravity + constraint grasp",
        fill=(188, 199, 216),
        font=SMALL,
    )

    draw.rounded_rectangle((600, 20, 936, 170), radius=18, fill=(8, 12, 20, 215))
    draw.text((620, 36), "adapter VLAAction", fill=(245, 247, 250), font=SMALL)
    action = sample.adapter_action or sample.scripted_action
    if sample.adapter_error:
        error = sample.adapter_error[:38]
        draw.text((620, 54), f"adapter error: {error}", fill=(248, 113, 113), font=MONO)
    elif sample.adapter_action is None:
        draw.text((620, 54), "scripted control shown", fill=(188, 199, 216), font=MONO)
    else:
        draw.text((620, 54), "model prediction shown", fill=(125, 211, 252), font=MONO)
    query_color = (74, 222, 128) if sample.adapter_query_fresh else (188, 199, 216)
    draw.text(
        (620, 148),
        f"adapter queries: {sample.adapter_query_count}",
        fill=query_color,
        font=MONO,
    )
    if sample.adapter_latency_ms is not None:
        draw.text(
            (800, 148),
            f"{sample.adapter_latency_ms:0.1f}ms",
            fill=(188, 199, 216),
            font=MONO,
        )
    labels = ("dx", "dy", "dz", "grip")
    for index, (label, value) in enumerate(zip(labels, action, strict=True)):
        y = 70 + index * 19
        draw.text((620, y - 5), label, fill=(188, 199, 216), font=MONO)
        draw.rounded_rectangle((670, y, 910, y + 9), radius=4, fill=(45, 55, 72))
        center = 790
        width = int(clamp(value, -1.0, 1.0) * 105)
        color = (34, 211, 238, 255) if label != "grip" else (74, 222, 128, 255)
        if width >= 0:
            draw.rounded_rectangle((center, y, center + width, y + 9), radius=4, fill=color)
        else:
            draw.rounded_rectangle((center + width, y, center, y + 9), radius=4, fill=color)

    draw.rounded_rectangle((24, 458, 936, 526), radius=18, fill=(8, 12, 20, 210))
    draw.text((44, 474), "observation", fill=(125, 211, 252), font=SMALL)
    draw.text((190, 474), "adapter.predict()", fill=(245, 247, 250), font=SMALL)
    draw.text((386, 474), "VLAAction", fill=(74, 222, 128), font=SMALL)
    draw.line((142, 485, 180, 485), fill=(125, 211, 252), width=3)
    draw.polygon([(180, 485), (170, 480), (170, 490)], fill=(125, 211, 252))
    draw.line((332, 485, 376, 485), fill=(74, 222, 128), width=3)
    draw.polygon([(376, 485), (366, 480), (366, 490)], fill=(74, 222, 128))

    x0, y0, width = 520, 482, 380
    draw.rounded_rectangle((x0, y0, x0 + width, y0 + 10), radius=5, fill=(45, 55, 72))
    current_index = PHASE_ORDER.index(sample.phase) if sample.phase in PHASE_ORDER else 0
    progress = (current_index + 1) / len(PHASE_ORDER)
    draw.rounded_rectangle(
        (x0, y0, x0 + int(width * progress), y0 + 10),
        radius=5,
        fill=(34, 211, 238),
    )
    for index, _ in enumerate(PHASE_ORDER):
        x = x0 + int(width * index / max(1, len(PHASE_ORDER) - 1))
        draw.line((x, y0 - 4, x, y0 + 14), fill=(148, 160, 178), width=1)
    draw.text((520, 504), f"phase timeline: {sample.phase}", fill=(188, 199, 216), font=MONO)
    return image


def run_simulation(config: PyBulletDemoConfig) -> list[RenderSample]:
    model = make_model(config)
    p, pybullet_data = import_pybullet()
    robot, cube, _ = setup_world(p, pybullet_data)
    set_initial_pose(p, robot)

    orientation = p.getQuaternionFromEuler([math.pi, 0.0, -math.pi / 4.0])
    waypoints = [
        Waypoint("observe", (0.42, -0.12, 0.36), 1.0, 45),
        Waypoint("approach", (0.58, -0.16, 0.25), 1.0, 55),
        Waypoint("descend", (0.58, -0.16, 0.105), 1.0, 45),
        Waypoint("close gripper", (0.58, -0.16, 0.105), 0.0, 38, attach=True),
        Waypoint("lift", (0.58, -0.16, 0.35), 0.0, 55),
        Waypoint("transport", (0.58, 0.22, 0.35), 0.0, 70),
        Waypoint("place", (0.58, 0.22, 0.105), 0.0, 50),
        Waypoint("open gripper", (0.58, 0.22, 0.105), 1.0, 38, detach=True),
        Waypoint("retreat", (0.38, 0.05, 0.38), 1.0, 55),
    ]

    current = (0.38, -0.08, 0.36)
    current_gripper = 1.0
    grasp_constraint: int | None = None
    samples: list[RenderSample] = []
    sim_step = 0
    last_target = current
    last_adapter_action: tuple[float, float, float, float] | None = None
    last_adapter_error: str | None = None
    last_adapter_latency_ms: float | None = None
    rendered_frames = 0
    adapter_query_count = 0

    for waypoint in waypoints:
        start = current
        start_gripper = current_gripper
        for step in range(waypoint.steps):
            progress = step / max(1, waypoint.steps - 1)
            target = lerp(start, waypoint.position, progress)
            gripper = start_gripper + (waypoint.gripper - start_gripper) * progress
            control_robot(p, robot, target, gripper, orientation)

            for _ in range(STEPS_PER_FRAME):
                p.stepSimulation()
                sim_step += 1

            if waypoint.attach and step == waypoint.steps // 2 and grasp_constraint is None:
                grasp_constraint = p.createConstraint(
                    robot,
                    11,
                    cube,
                    -1,
                    p.JOINT_FIXED,
                    [0, 0, 0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                )
            if waypoint.detach and step == waypoint.steps // 2 and grasp_constraint is not None:
                p.removeConstraint(grasp_constraint)
                grasp_constraint = None

            if step % config.render_stride == 0:
                raw = render_camera(p)
                scripted_action = (
                    (target[0] - last_target[0]) * 8.0,
                    (target[1] - last_target[1]) * 8.0,
                    (target[2] - last_target[2]) * 8.0,
                    1.0 - gripper,
                )
                if config.model_call_every > 0 and rendered_frames % config.model_call_every == 0:
                    adapter_query_count += 1
                    (
                        last_adapter_action,
                        last_adapter_error,
                        last_adapter_latency_ms,
                    ) = predict_adapter_action(
                        model,
                        raw,
                        config,
                        phase=waypoint.name,
                        target=target,
                        gripper=gripper,
                        attached=grasp_constraint is not None,
                        sim_time=sim_step / 240.0,
                    )
                    adapter_query_fresh = True
                else:
                    adapter_query_fresh = False
                cube_position_raw, _ = p.getBasePositionAndOrientation(cube)
                cube_position = (
                    float(cube_position_raw[0]),
                    float(cube_position_raw[1]),
                    float(cube_position_raw[2]),
                )
                samples.append(
                    RenderSample(
                        image=raw,
                        phase=waypoint.name,
                        position=target,
                        cube_position=cube_position,
                        cube_goal_position=CUBE_GOAL_POSITION,
                        scripted_action=scripted_action,
                        adapter_action=last_adapter_action,
                        adapter_error=last_adapter_error,
                        adapter_latency_ms=last_adapter_latency_ms,
                        adapter_query_count=adapter_query_count,
                        adapter_query_fresh=adapter_query_fresh,
                        attached=grasp_constraint is not None,
                        sim_time=sim_step / 240.0,
                        model_name=config.model_name,
                        runtime=config.runtime,
                        frame_index=rendered_frames,
                    )
                )
                last_target = target
                rendered_frames += 1
        current = waypoint.position
        current_gripper = waypoint.gripper

    if samples:
        raw = render_camera(p)
        scripted_action = (
            (current[0] - last_target[0]) * 8.0,
            (current[1] - last_target[1]) * 8.0,
            (current[2] - last_target[2]) * 8.0,
            1.0 - current_gripper,
        )
        cube_position_raw, _ = p.getBasePositionAndOrientation(cube)
        cube_position = (
            float(cube_position_raw[0]),
            float(cube_position_raw[1]),
            float(cube_position_raw[2]),
        )
        samples.append(
            RenderSample(
                image=raw,
                phase=waypoints[-1].name,
                position=current,
                cube_position=cube_position,
                cube_goal_position=CUBE_GOAL_POSITION,
                scripted_action=scripted_action,
                adapter_action=last_adapter_action,
                adapter_error=last_adapter_error,
                adapter_latency_ms=last_adapter_latency_ms,
                adapter_query_count=adapter_query_count,
                adapter_query_fresh=False,
                attached=grasp_constraint is not None,
                sim_time=sim_step / 240.0,
                model_name=config.model_name,
                runtime=config.runtime,
                frame_index=rendered_frames,
            )
        )

    p.disconnect()
    return samples


def summarize_pybullet_samples(
    model_name: str,
    runtime: str,
    samples: list[RenderSample],
    *,
    remote_url: str | None = None,
    last_error: str | None = None,
) -> PyBulletComparisonResult:
    fresh_samples = [sample for sample in samples if sample.adapter_query_fresh]
    error_samples = [sample for sample in fresh_samples if sample.adapter_error is not None]
    latencies = [
        sample.adapter_latency_ms
        for sample in fresh_samples
        if sample.adapter_latency_ms is not None
    ]
    actions = [
        np.asarray(sample.adapter_action, dtype=np.float32)
        for sample in fresh_samples
        if sample.adapter_action is not None
    ]
    query_count = max((sample.adapter_query_count for sample in samples), default=0)
    action_values = np.concatenate(actions) if actions else np.asarray([], dtype=np.float32)
    mean_abs_action = (
        float(np.mean(np.abs(action_values))) if action_values.size else None
    )
    cube_positions = [sample.cube_position for sample in samples]
    initial_cube_position = cube_positions[0] if cube_positions else None
    final_cube_position = cube_positions[-1] if cube_positions else None
    cube_goal_position = samples[-1].cube_goal_position if samples else CUBE_GOAL_POSITION
    max_cube_z = max((position[2] for position in cube_positions), default=0.0)
    cube_lifted = max_cube_z > 0.12
    cube_moved_distance = (
        distance3(initial_cube_position, final_cube_position)
        if initial_cube_position is not None and final_cube_position is not None
        else None
    )
    final_cube_distance_to_goal = (
        distance3(final_cube_position, cube_goal_position)
        if final_cube_position is not None
        else None
    )
    grasp_attached_frames = sum(1 for sample in samples if sample.attached)
    phase_indices = [
        PHASE_ORDER.index(sample.phase) for sample in samples if sample.phase in PHASE_ORDER
    ]
    phase_completion = (
        (max(phase_indices) + 1) / len(PHASE_ORDER) if phase_indices else 0.0
    )
    task_success = (
        bool(samples)
        and cube_lifted
        and grasp_attached_frames > 0
        and final_cube_distance_to_goal is not None
        and final_cube_distance_to_goal <= TASK_GOAL_TOLERANCE_M
        and phase_completion >= 1.0
    )
    observed_error = last_error or next(
        (sample.adapter_error for sample in reversed(fresh_samples) if sample.adapter_error),
        None,
    )
    return PyBulletComparisonResult(
        model_name=model_name,
        runtime=runtime,
        ok=bool(samples) and query_count > 0 and not error_samples and observed_error is None,
        remote_url=remote_url,
        frames=len(samples),
        adapter_queries=query_count,
        adapter_errors=len(error_samples),
        mean_latency_ms=float(np.mean(latencies)) if latencies else None,
        max_latency_ms=float(np.max(latencies)) if latencies else None,
        mean_abs_action=mean_abs_action,
        task_success=task_success,
        cube_lifted=cube_lifted,
        cube_moved_distance=cube_moved_distance,
        final_cube_distance_to_goal=final_cube_distance_to_goal,
        grasp_attached_frames=grasp_attached_frames,
        phase_completion=phase_completion,
        last_error=observed_error,
    )


def compare_pybullet_models(
    model_names: list[str],
    *,
    runtime: str = "local",
    remote_url: str = "http://localhost:8000",
    remote_urls: dict[str, str] | None = None,
    instruction: str = "pick up the red block",
    model_call_every: int = 8,
    render_stride: int = 12,
    allow_local_heavy: bool = False,
) -> list[PyBulletComparisonResult]:
    remote_url_overrides = remote_urls or {}
    targets = [
        PyBulletComparisonTarget(
            model_name=model_name,
            runtime=runtime,
            remote_url=remote_url_overrides.get(model_name.strip().lower(), remote_url),
        )
        for model_name in model_names
        if model_name.strip()
    ]
    return compare_pybullet_targets(
        targets,
        instruction=instruction,
        model_call_every=model_call_every,
        render_stride=render_stride,
        allow_local_heavy=allow_local_heavy,
    )


def compare_pybullet_targets(
    targets: list[PyBulletComparisonTarget],
    *,
    instruction: str = "pick up the red block",
    model_call_every: int = 8,
    render_stride: int = 12,
    allow_local_heavy: bool = False,
) -> list[PyBulletComparisonResult]:
    results: list[PyBulletComparisonResult] = []
    for target in targets:
        model_name = target.model_name
        runtime = target.runtime
        canonical = model_name.strip().lower()
        if not canonical:
            continue
        if runtime == "local" and canonical in HEAVY_LOCAL_MODELS and not allow_local_heavy:
            results.append(
                PyBulletComparisonResult(
                    model_name=model_name,
                    runtime=runtime,
                    ok=False,
                    remote_url=None,
                    last_error=(
                        "local heavy adapter skipped to avoid model download; "
                        "use --allow-local-heavy or --runtime remote"
                    ),
                )
            )
            continue

        config = PyBulletDemoConfig(
            model_name=model_name,
            runtime=runtime,
            remote_url=target.remote_url,
            instruction=instruction,
            model_call_every=model_call_every,
            render_stride=render_stride,
            adapter_kwargs=target.adapter_kwargs,
        )
        try:
            samples = run_simulation(config)
        except Exception as exc:
            results.append(
                PyBulletComparisonResult(
                    model_name=model_name,
                    runtime=runtime,
                    ok=False,
                    remote_url=target.remote_url if runtime == "remote" else None,
                    last_error=str(exc),
                )
            )
            continue
        results.append(
            summarize_pybullet_samples(
                model_name,
                runtime,
                samples,
                remote_url=target.remote_url if runtime == "remote" else None,
            )
        )
    return results


def format_pybullet_comparison_markdown(
    results: list[PyBulletComparisonResult],
    *,
    title: str = "PyBullet VLA Runtime Comparison",
) -> str:
    lines = [
        f"## {title}",
        "",
        "| Model | Runtime | Endpoint | OK | Frames | Queries | Errors | "
        "Task | Lifted | Goal dist m | Cube moved m | Phase | Mean latency ms | "
        "Mean abs action | Note |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        endpoint = result.remote_url or "-"
        mean_latency = (
            f"{result.mean_latency_ms:0.2f}" if result.mean_latency_ms is not None else "-"
        )
        mean_action = (
            f"{result.mean_abs_action:0.3f}" if result.mean_abs_action is not None else "-"
        )
        goal_distance = (
            f"{result.final_cube_distance_to_goal:0.3f}"
            if result.final_cube_distance_to_goal is not None
            else "-"
        )
        moved_distance = (
            f"{result.cube_moved_distance:0.3f}"
            if result.cube_moved_distance is not None
            else "-"
        )
        phase = f"{result.phase_completion:0.2f}"
        note = (result.last_error or "-").replace("|", "\\|")
        lines.append(
            f"| `{result.model_name}` | `{result.runtime}` | {endpoint} | "
            f"{str(result.ok).lower()} | {result.frames} | {result.adapter_queries} | "
            f"{result.adapter_errors} | {str(result.task_success).lower()} | "
            f"{str(result.cube_lifted).lower()} | {goal_distance} | {moved_distance} | "
            f"{phase} | {mean_latency} | {mean_action} | {note} |"
        )
    lines.extend(
        [
            "",
            "This is a runtime smoke comparison on the same deterministic PyBullet scene. "
            "It measures adapter availability, query behavior, errors, latency, action "
            "magnitude, and scripted-scene task telemetry; it is not a model-quality benchmark.",
        ]
    )
    return "\n".join(lines) + "\n"


def _html_metric(value: float | None, precision: int = 2) -> str:
    if value is None:
        return "-"
    return f"{value:.{precision}f}"


def _html_bar(value: float | None, maximum: float, color: str) -> str:
    width = (
        0.0
        if value is None or maximum <= 0
        else clamp((value / maximum) * 100.0, 0.0, 100.0)
    )
    return (
        '<div class="meter">'
        f'<span style="width:{width:.1f}%;background:{escape(color)}"></span>'
        "</div>"
    )


def format_pybullet_comparison_html(
    results: list[PyBulletComparisonResult],
    *,
    title: str = "PyBullet VLA Runtime Comparison",
) -> str:
    ok_count = sum(1 for result in results if result.ok)
    task_success_count = sum(1 for result in results if result.task_success)
    total_queries = sum(result.adapter_queries for result in results)
    total_errors = sum(result.adapter_errors for result in results)
    latencies = [
        result.mean_latency_ms for result in results if result.mean_latency_ms is not None
    ]
    actions = [
        result.mean_abs_action for result in results if result.mean_abs_action is not None
    ]
    max_latency = max(latencies, default=0.0)
    max_action = max(actions, default=0.0)
    payload = json.dumps([asdict(result) for result in results], indent=2).replace(
        "</",
        "<\\/",
    )

    rows = []
    for result in results:
        status = "ok" if result.ok else "error"
        task_status = "ok" if result.task_success else "error"
        task_label = "success" if result.task_success else "miss"
        note = result.last_error or "-"
        latency = _html_metric(result.mean_latency_ms)
        action = _html_metric(result.mean_abs_action, precision=3)
        goal_distance = _html_metric(result.final_cube_distance_to_goal, precision=3)
        moved_distance = _html_metric(result.cube_moved_distance, precision=3)
        rows.append(
            "<tr>"
            f"<td><code>{escape(result.model_name)}</code></td>"
            f"<td><code>{escape(result.runtime)}</code></td>"
            f"<td>{escape(result.remote_url or '-')}</td>"
            f'<td><span class="badge {status}">{status}</span></td>'
            f'<td><span class="badge {task_status}">{task_label}</span></td>'
            f"<td>{str(result.cube_lifted).lower()}</td>"
            f"<td>{goal_distance}</td>"
            f"<td>{moved_distance}</td>"
            f"<td>{result.phase_completion:0.2f}</td>"
            f"<td>{result.frames}</td>"
            f"<td>{result.adapter_queries}</td>"
            f"<td>{result.adapter_errors}</td>"
            f"<td>{latency}{_html_bar(result.mean_latency_ms, max_latency, '#22c55e')}</td>"
            f"<td>{action}{_html_bar(result.mean_abs_action, max_action, '#38bdf8')}</td>"
            f"<td>{escape(note)}</td>"
            "</tr>"
        )

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{escape(title)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f8fafc;
      --panel: #ffffff;
      --muted: #5b687a;
      --text: #172033;
      --line: #d7dee8;
      --ok: #16a34a;
      --error: #e11d48;
      --accent: #0284c7;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont,
        "Segoe UI", sans-serif;
    }}
    main {{
      max-width: 1180px;
      margin: 0 auto;
      padding: 36px 20px 48px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(28px, 4vw, 44px);
      letter-spacing: 0;
    }}
    p {{
      color: var(--muted);
      line-height: 1.55;
    }}
    .cards {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin: 24px 0;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 16px;
    }}
    .label {{
      color: var(--muted);
      font-size: 13px;
    }}
    .value {{
      margin-top: 8px;
      font-size: 26px;
      font-weight: 700;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      overflow: hidden;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 11px 12px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }}
    th {{
      color: #475569;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0;
    }}
    code {{
      color: #075985;
    }}
    .badge {{
      display: inline-block;
      border-radius: 999px;
      padding: 3px 9px;
      font-size: 12px;
      font-weight: 700;
    }}
    .badge.ok {{
      color: #052e16;
      background: var(--ok);
    }}
    .badge.error {{
      color: #4c0519;
      background: var(--error);
    }}
    .meter {{
      height: 6px;
      margin-top: 6px;
      background: #e5e7eb;
      border-radius: 999px;
      overflow: hidden;
    }}
    .meter span {{
      display: block;
      height: 100%;
    }}
    details {{
      margin-top: 20px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: var(--panel);
      padding: 12px 14px;
    }}
    pre {{
      overflow: auto;
      color: #334155;
    }}
  </style>
</head>
<body>
  <main>
    <h1>{escape(title)}</h1>
    <p>
      Static VLA runtime report generated from the deterministic PyBullet smoke scene.
      It is useful for comparing adapter availability, remote server wiring, latency, and
      scripted-scene task telemetry. It is not a model-quality benchmark.
    </p>
    <section class="cards">
      <div class="card"><div class="label">models</div><div class="value">{len(results)}</div></div>
      <div class="card"><div class="label">ok</div><div class="value">{ok_count}</div></div>
      <div class="card">
        <div class="label">task success</div><div class="value">{task_success_count}</div>
      </div>
      <div class="card">
        <div class="label">adapter queries</div><div class="value">{total_queries}</div>
      </div>
      <div class="card">
        <div class="label">adapter errors</div><div class="value">{total_errors}</div>
      </div>
    </section>
    <table>
      <thead>
        <tr>
          <th>Model</th><th>Runtime</th><th>Endpoint</th><th>Status</th>
          <th>Task</th><th>Lifted</th><th>Goal dist m</th><th>Moved m</th><th>Phase</th>
          <th>Frames</th><th>Queries</th><th>Errors</th>
          <th>Mean latency ms</th><th>Mean abs action</th><th>Note</th>
        </tr>
      </thead>
      <tbody>
        {"".join(rows)}
      </tbody>
    </table>
    <details>
      <summary>Raw JSON</summary>
      <pre>{escape(payload)}</pre>
    </details>
  </main>
</body>
</html>
"""


def render_pybullet_demo(config: PyBulletDemoConfig) -> dict[str, object]:
    config.out.parent.mkdir(parents=True, exist_ok=True)
    samples = run_simulation(config)
    frames = [overlay(sample) for sample in samples]
    frames[0].save(
        config.out,
        save_all=True,
        append_images=frames[1:] + [frames[-1]] * 12,
        duration=int(1000 / FPS),
        loop=0,
        optimize=True,
    )
    return {
        "out": str(config.out),
        "frames": len(frames),
        "model": config.model_name,
        "runtime": config.runtime,
    }


def main() -> None:
    result = render_pybullet_demo(PyBulletDemoConfig())
    print(f"{result['out']} ({result['frames']} frames)")


if __name__ == "__main__":
    main()
