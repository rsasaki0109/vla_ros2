#!/usr/bin/env python3
"""Closed-loop SmolVLA demo on a LeRobot-aligned SO-100 kinematic stand-in.

Every control tick builds a ``VLAObservation`` from synthetic 256x256 camera views
and 6D proprioception, calls ``load_model("smolvla")``, and integrates the returned
6D action into a minimal joint-space scene. The scene is initialized from episode 0 of
``lerobot/svla_so100_stacking`` (task + starting joints).

Honesty note: this is **not** the official SO-100 physics sim and **not** a task-success
benchmark. ``lerobot/smolvla_base`` is a base checkpoint; fine-tuning on your robot or
sim is expected for reliable stacking. The loop, inference, and action stream are real.

Run (GPU + SmolVLA extras required):

    .venv-smolvla/bin/python scripts/record_smolvla_so100_demo.py

Writes ``docs/assets/smolvla_so100_demo.gif``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vla_ros2 import load_model
from vla_ros2.core.types import VLAObservation
from vla_ros2.sim.so100_kinematic import (
    SO100Scene,
    apply_joint_action,
    observation_images,
    scene_from_dataset_state,
)

OUT = Path("docs/assets/smolvla_so100_demo.gif")
DATASET = "lerobot/svla_so100_stacking"
DEFAULT_PRETRAINED = "lerobot/smolvla_base"
RENDER_EVERY = 2
MAX_STEPS_DEFAULT = 60


def _load_lerobot_episode_start(episode: int) -> tuple[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(DATASET, episodes=[episode])
    frame = dataset[0]
    instruction = str(frame["task"])
    state = np.asarray(frame["observation.state"], dtype=np.float32).reshape(6)
    return instruction, state


def _annotate(frame: Image.Image, *, step: int, instruction: str) -> Image.Image:
    banner = Image.new("RGB", (frame.width, 28), (24, 24, 28))
    ImageDraw.Draw(banner).text((8, 6), f"smolvla | step {step:03d}", fill=(230, 230, 230))
    caption = Image.new("RGB", (frame.width, 22), (18, 18, 22))
    ImageDraw.Draw(caption).text((8, 4), instruction[:72], fill=(200, 200, 205))
    stacked = Image.new("RGB", (frame.width, frame.height + 50), (0, 0, 0))
    stacked.paste(banner, (0, 0))
    stacked.paste(frame, (0, 28))
    stacked.paste(caption, (0, 28 + frame.height))
    return stacked


def run_demo(
    *,
    out: Path,
    steps: int,
    episode: int,
    device: str,
    pretrained: str,
    local_files_only: bool,
    action_blend: float,
) -> None:
    instruction, start_state = _load_lerobot_episode_start(episode)
    scene = scene_from_dataset_state(start_state)
    runtime = load_model(
        "smolvla",
        device=device,
        pretrained=pretrained,
        local_files_only=local_files_only,
    )

    frames: list[Image.Image] = []
    for step in range(steps):
        images = observation_images(scene)
        observation = VLAObservation(
            instruction=instruction,
            images=images,
            state={"state": scene.joint_state.copy()},
            metadata={"source": "so100_kinematic_demo", "episode": episode, "step": step},
        )
        action = runtime.predict(observation=observation)
        scene = SO100Scene(
            joint_state=apply_joint_action(
                scene.joint_state,
                action.data,
                blend=action_blend,
            ),
            red_cube_xy=scene.red_cube_xy,
            blue_cube_xy=scene.blue_cube_xy,
        )

        if step % RENDER_EVERY == 0:
            top = images["camera1"]
            wrist = images["camera2"]
            panel = Image.fromarray(np.hstack([top, wrist]))
            frames.append(_annotate(panel, step=step, instruction=instruction))

    out.parent.mkdir(parents=True, exist_ok=True)
    duration_ms = int(1000 * RENDER_EVERY / 5)
    frames[0].save(
        out,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
    )
    print(f"wrote {out} ({len(frames)} frames, {steps} control steps)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--steps", type=int, default=MAX_STEPS_DEFAULT)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--action-blend", type=float, default=0.35)
    args = parser.parse_args()

    run_demo(
        out=args.out,
        steps=args.steps,
        episode=args.episode,
        device=args.device,
        pretrained=args.pretrained,
        local_files_only=args.local_files_only,
        action_blend=args.action_blend,
    )


if __name__ == "__main__":
    main()
