#!/usr/bin/env python3
"""Render the README simulator GIF: a Franka Panda pick-and-place in PyBullet,
driven through the real vla_ros2 runtime.

Every control tick builds a VLAObservation and calls the vla_ros2 local runtime
(`load_model("scripted")`); the returned VLAAction's end-effector delta and
gripper channel command the arm. The scene is a real PyBullet physics sim — not
a recorded action plot.

Run (uses the project venv which has pybullet + vla_ros2 installed):
    .venv/bin/python scripts/record_sim_demo.py
Writes docs/assets/sim_demo.gif.
"""

from __future__ import annotations

import os

import numpy as np
import pybullet as p
import pybullet_data
from PIL import Image

from vla_ros2 import load_model

OUT = "docs/assets/sim_demo.gif"
W, H = 512, 384
RENDER_EVERY = 2  # subsample frames to keep the GIF small
EEF_LINK = 11  # panda_hand
FINGERS = (9, 10)
ARM_JOINTS = list(range(7))

# (phase label fed to the runtime, number of control ticks)
PHASES = [
    ("approach", 26),
    ("descend", 18),
    ("close", 12),
    ("lift", 18),
    ("transport", 26),
    ("place", 16),
    ("open", 12),
    ("retreat", 22),
]


def reset_arm(robot: int, target: np.ndarray) -> None:
    angles = p.calculateInverseKinematics(robot, EEF_LINK, target.tolist())
    for j in ARM_JOINTS:
        p.resetJointState(robot, j, angles[j])


def drive_to(robot: int, target: np.ndarray, opening: float) -> None:
    angles = p.calculateInverseKinematics(robot, EEF_LINK, target.tolist())
    for j in ARM_JOINTS:
        p.setJointMotorControl2(robot, j, p.POSITION_CONTROL, angles[j], force=240)
    for f in FINGERS:
        p.setJointMotorControl2(robot, f, p.POSITION_CONTROL, opening, force=60)


def render() -> np.ndarray:
    view = p.computeViewMatrixFromYawPitchRoll(
        cameraTargetPosition=[0.55, 0.0, 0.18],
        distance=1.25,
        yaw=48,
        pitch=-32,
        roll=0,
        upAxisIndex=2,
    )
    proj = p.computeProjectionMatrixFOV(fov=55, aspect=W / H, nearVal=0.1, farVal=3.1)
    img = p.getCameraImage(W, H, view, proj, renderer=p.ER_TINY_RENDERER)[2]
    return np.reshape(img, (H, W, 4))[:, :, :3].astype(np.uint8)


def main() -> None:
    runtime = load_model("scripted")

    p.connect(p.DIRECT)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.8)
    p.loadURDF("plane.urdf")
    # The scripted policy's integrated end-effector path lands the grasp near
    # (0.70, -0.34) and releases near (0.70, -0.06); place the props there so the
    # runtime's own action stream performs a coherent pick-and-place.
    p.loadURDF("tray/traybox.urdf", [0.70, -0.04, 0.0], globalScaling=0.6)
    cube_start = [0.70, -0.34, 0.025]
    cube = p.loadURDF("cube_small.urdf", cube_start, globalScaling=1.1)
    p.changeVisualShape(cube, -1, rgbaColor=[0.85, 0.1, 0.12, 1.0])
    robot = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

    eef = np.array([0.45, -0.28, 0.45])
    reset_arm(robot, eef)
    for f in FINGERS:
        p.resetJointState(robot, f, 0.04)
    for _ in range(60):
        p.stepSimulation()

    frames: list[Image.Image] = []
    grasp_constraint: int | None = None
    step_scale = 0.012
    tick = 0

    for phase, ticks in PHASES:
        for _ in range(ticks):
            tick += 1
            action = runtime.predict(
                instruction="pick up the red block and place it in the tray",
                phase=phase,
            )
            dxyz = np.asarray(action.data[:3], dtype=float) * step_scale
            grip = float(action.data[6])
            eef = eef + dxyz
            eef[2] = max(eef[2], 0.16)
            opening = 0.04 if grip >= 0 else 0.0
            drive_to(robot, eef, opening)
            for _ in range(6):
                p.stepSimulation()

            cube_pos = np.asarray(p.getBasePositionAndOrientation(cube)[0])
            eef_now = np.asarray(p.getLinkState(robot, EEF_LINK)[0])
            if grip < 0 and grasp_constraint is None and np.linalg.norm(eef_now - cube_pos) < 0.2:
                grasp_constraint = p.createConstraint(
                    robot, EEF_LINK, cube, -1, p.JOINT_FIXED,
                    [0, 0, 0], [0, 0, 0.02], list(cube_pos - eef_now),
                )
            if grip >= 0 and grasp_constraint is not None:
                p.removeConstraint(grasp_constraint)
                grasp_constraint = None

            if tick % RENDER_EVERY == 0:
                frames.append(Image.fromarray(render()))

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    palette_frames = [
        f.quantize(colors=128, method=Image.MEDIANCUT, dither=Image.NONE) for f in frames
    ]
    palette_frames[0].save(
        OUT,
        save_all=True,
        append_images=palette_frames[1:],
        duration=90,
        loop=0,
        optimize=True,
    )
    p.disconnect()
    print(f"wrote {OUT} ({len(frames)} frames)")


if __name__ == "__main__":
    main()
