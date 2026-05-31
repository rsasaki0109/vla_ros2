from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
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
    scripted_action: tuple[float, float, float, float]
    adapter_action: tuple[float, float, float, float] | None
    adapter_error: str | None
    attached: bool
    sim_time: float
    model_name: str
    runtime: str


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
) -> tuple[tuple[float, float, float, float] | None, str | None]:
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
        return prediction_to_demo_action(model.predict(observation=observation)), None
    except Exception as exc:
        return None, str(exc)


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
        basePosition=[0.58, -0.16, 0.035],
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

    draw.rounded_rectangle((600, 20, 936, 152), radius=18, fill=(8, 12, 20, 215))
    draw.text((620, 36), "adapter VLAAction", fill=(245, 247, 250), font=SMALL)
    action = sample.adapter_action or sample.scripted_action
    if sample.adapter_error:
        error = sample.adapter_error[:38]
        draw.text((620, 54), f"adapter error: {error}", fill=(248, 113, 113), font=MONO)
    elif sample.adapter_action is None:
        draw.text((620, 54), "scripted control shown", fill=(188, 199, 216), font=MONO)
    else:
        draw.text((620, 54), "model prediction shown", fill=(125, 211, 252), font=MONO)
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
    return image


def run_simulation(config: PyBulletDemoConfig) -> list[RenderSample]:
    p, pybullet_data = import_pybullet()
    model = make_model(config)
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
    rendered_frames = 0

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
                    last_adapter_action, last_adapter_error = predict_adapter_action(
                        model,
                        raw,
                        config,
                        phase=waypoint.name,
                        target=target,
                        gripper=gripper,
                        attached=grasp_constraint is not None,
                        sim_time=sim_step / 240.0,
                    )
                samples.append(
                    RenderSample(
                        image=raw,
                        phase=waypoint.name,
                        position=target,
                        scripted_action=scripted_action,
                        adapter_action=last_adapter_action,
                        adapter_error=last_adapter_error,
                        attached=grasp_constraint is not None,
                        sim_time=sim_step / 240.0,
                        model_name=config.model_name,
                        runtime=config.runtime,
                    )
                )
                last_target = target
                rendered_frames += 1
        current = waypoint.position
        current_gripper = waypoint.gripper

    p.disconnect()
    return samples


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
