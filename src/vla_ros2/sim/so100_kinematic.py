"""Minimal SO-100-style kinematic stand-in for SmolVLA closed-loop demos.

This is not a physics simulator. It integrates 6D joint commands and renders
synthetic top/wrist views sized for ``lerobot/smolvla_base`` (256x256 RGB).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

JOINT_LIMITS_LOW = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
JOINT_LIMITS_HIGH = np.array([180.0, 180.0, 180.0, 180.0, 180.0, 1.0], dtype=np.float32)


@dataclass
class SO100Scene:
    joint_state: np.ndarray
    red_cube_xy: np.ndarray
    blue_cube_xy: np.ndarray

    def copy(self) -> SO100Scene:
        return SO100Scene(
            joint_state=self.joint_state.copy(),
            red_cube_xy=self.red_cube_xy.copy(),
            blue_cube_xy=self.blue_cube_xy.copy(),
        )


def clip_joint_state(state: np.ndarray) -> np.ndarray:
    clipped = np.asarray(state, dtype=np.float32).reshape(6)
    return np.clip(clipped, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)


def apply_joint_action(
    state: np.ndarray,
    action: np.ndarray,
    *,
    blend: float = 0.35,
) -> np.ndarray:
    """Blend toward the commanded joint target (SO-100 datasets use absolute targets)."""

    current = clip_joint_state(state)
    target = clip_joint_state(action)
    updated = (1.0 - blend) * current + blend * target
    return clip_joint_state(updated)


def forward_kinematics_2d(joint_deg: np.ndarray) -> tuple[float, float]:
    """Planar end-effector position from the first three joint angles (degrees)."""

    angles = np.deg2rad(joint_deg[:3])
    link = 0.18
    x = 0.25
    y = 0.12
    for theta in angles:
        x += link * float(np.cos(theta))
        y += link * float(np.sin(theta))
    return x, y


def render_top_view(scene: SO100Scene, *, size: int = 256) -> np.ndarray:
    img = Image.new("RGB", (size, size), (235, 238, 242))
    draw = ImageDraw.Draw(img)
    table_y = int(size * 0.72)
    draw.rectangle((0, table_y, size, size), fill=(150, 158, 168))

    def cube(xy: np.ndarray, color: tuple[int, int, int]) -> None:
        px = int(np.clip(xy[0], 0.0, 1.0) * (size - 40)) + 20
        py = int(np.clip(xy[1], 0.0, 1.0) * (table_y - 50)) + 20
        draw.rectangle((px, py, px + 22, py + 22), fill=color)

    cube(scene.blue_cube_xy, (40, 90, 200))
    cube(scene.red_cube_xy, (210, 40, 40))

    base_x, base_y = int(size * 0.18), table_y
    draw.rectangle((base_x - 14, base_y - 10, base_x + 14, base_y), fill=(90, 95, 102))

    x, y = forward_kinematics_2d(scene.joint_state)
    ex = int(np.clip(x, 0.0, 1.0) * (size - 30)) + 15
    ey = int(np.clip(y, 0.0, 1.0) * (table_y - 30)) + 15
    draw.line((base_x, base_y, ex, ey), fill=(30, 30, 30), width=5)
    draw.ellipse((ex - 8, ey - 8, ex + 8, ey + 8), fill=(240, 120, 30))

    return np.asarray(img, dtype=np.uint8)


def render_wrist_view(scene: SO100Scene, *, size: int = 256) -> np.ndarray:
    grip = float(scene.joint_state[5])
    open_amt = int(180 + grip * 60)
    img = Image.new("RGB", (size, size), (open_amt, open_amt, open_amt + 10))
    draw = ImageDraw.Draw(img)
    gap = int(12 + (1.0 - grip) * 28)
    cx, cy = size // 2, size // 2
    draw.rectangle((cx - 50, cy - gap, cx - 10, cy + gap), fill=(70, 70, 75))
    draw.rectangle((cx + 10, cy - gap, cx + 50, cy + gap), fill=(70, 70, 75))
    return np.asarray(img, dtype=np.uint8)


def scene_from_dataset_state(
    state: np.ndarray,
    *,
    red_cube_xy: tuple[float, float] = (0.62, 0.55),
    blue_cube_xy: tuple[float, float] = (0.62, 0.35),
) -> SO100Scene:
    return SO100Scene(
        joint_state=clip_joint_state(state),
        red_cube_xy=np.array(red_cube_xy, dtype=np.float32),
        blue_cube_xy=np.array(blue_cube_xy, dtype=np.float32),
    )


def observation_images(scene: SO100Scene, *, size: int = 256) -> dict[str, np.ndarray]:
    top = render_top_view(scene, size=size)
    wrist = render_wrist_view(scene, size=size)
    return {"camera1": top, "camera2": wrist, "camera3": top}
