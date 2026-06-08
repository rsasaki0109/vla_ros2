#!/usr/bin/env python3
"""Record a SmolVLA × Gazebo actuation demo GIF.

Launches ``gz_smolvla.launch.py`` with actuation enabled, subscribes to live
``/joint_states``, renders the same SO-100 kinematic views used by
``vla_smolvla_input_node``, and writes ``docs/assets/gz_smolvla_demo.gif``.

Honesty note: camera views are synthetic (not Gazebo RGB). Joint motion comes
from the real Gazebo + ros2_control graph when ``--launch`` is used.

Run (GPU + ROS2 Jazzy + colcon build):

    ./scripts/record_gz_smolvla_demo.sh

Offline kinematic fallback (no Gazebo; same renderer, no real actuation):

    .venv-smolvla/bin/python scripts/record_gz_smolvla_demo.py --offline
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from vla_ros2 import load_model
from vla_ros2.core.types import VLAObservation
from vla_ros2.sim.gazebo_smolvla import gazebo_joints_to_smolvla_state
from vla_ros2.sim.so100_kinematic import (
    SO100Scene,
    apply_joint_action,
    observation_images,
    scene_from_dataset_state,
)

OUT = Path("docs/assets/gz_smolvla_demo.gif")
DATASET = "lerobot/svla_so100_stacking"
DEFAULT_PRETRAINED = "lerobot/smolvla_base"
RENDER_EVERY = 2


def _load_lerobot_episode_start(episode: int) -> tuple[str, np.ndarray]:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    dataset = LeRobotDataset(DATASET, episodes=[episode])
    frame = dataset[0]
    instruction = str(frame["task"])
    state = np.asarray(frame["observation.state"], dtype=np.float32).reshape(6)
    return instruction, state


def _annotate(
    frame: Image.Image,
    *,
    step: int,
    instruction: str,
    mode: str,
    action_dim: int | None,
) -> Image.Image:
    banner = Image.new("RGB", (frame.width, 28), (24, 24, 28))
    suffix = f" | action dim={action_dim}" if action_dim is not None else ""
    ImageDraw.Draw(banner).text(
        (8, 6),
        f"gz_smolvla ({mode}) | step {step:03d}{suffix}",
        fill=(230, 230, 230),
    )
    caption = Image.new("RGB", (frame.width, 22), (18, 18, 22))
    ImageDraw.Draw(caption).text((8, 4), instruction[:72], fill=(200, 200, 205))
    stacked = Image.new("RGB", (frame.width, frame.height + 50), (0, 0, 0))
    stacked.paste(banner, (0, 0))
    stacked.paste(frame, (0, 28))
    stacked.paste(caption, (0, 28 + frame.height))
    return stacked


def _panel_from_joint_names(names: list[str], positions: list[float]) -> Image.Image:
    state = np.asarray(
        gazebo_joints_to_smolvla_state(names, positions),
        dtype=np.float32,
    )
    scene = scene_from_dataset_state(state)
    images = observation_images(scene)
    return Image.fromarray(np.hstack([images["camera1"], images["camera2"]]))


def run_offline(
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
    action_dim: int | None = None
    for step in range(steps):
        images = observation_images(scene)
        observation = VLAObservation(
            instruction=instruction,
            images=images,
            state={"state": scene.joint_state.copy()},
            metadata={"source": "gz_smolvla_offline", "episode": episode, "step": step},
        )
        action = runtime.predict(observation=observation)
        action_dim = int(action.data.size)
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
            panel = Image.fromarray(np.hstack([images["camera1"], images["camera2"]]))
            frames.append(
                _annotate(
                    panel,
                    step=step,
                    instruction=instruction,
                    mode="offline",
                    action_dim=action_dim,
                )
            )

    _save_gif(out, frames, steps)


def _save_gif(out: Path, frames: list[Image.Image], steps: int) -> None:
    if not frames:
        raise RuntimeError("no frames captured")
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


def run_ros(
    *,
    out: Path,
    duration_sec: float,
    episode: int,
    pretrained: str,
    enable_actuation: bool,
    ros_domain_id: int,
) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from vla_ros2_msgs.msg import VLAAction
        from vla_ros2_ros.qos import action_qos
    except ImportError as exc:
        msg = "ROS2 Python packages required; run ./scripts/record_gz_smolvla_demo.sh"
        raise RuntimeError(msg) from exc

    instruction, _ = _load_lerobot_episode_start(episode)
    repo_root = Path(__file__).resolve().parents[1]
    install_setup = repo_root / "install" / "setup.bash"
    if not install_setup.is_file():
        msg = "colcon install/setup.bash missing; build ros2 packages first"
        raise RuntimeError(msg)

    env = os.environ.copy()
    env["ROS_DOMAIN_ID"] = str(ros_domain_id)
    env["PYTHONPATH"] = os.pathsep.join(
        [str(repo_root / "src"), env.get("PYTHONPATH", "")]
    ).strip(os.pathsep)

    launch_cmd = [
        "bash",
        "-lc",
        (
            f"set +u; source /opt/ros/jazzy/setup.bash; "
            f"source {install_setup}; set -u; "
            f"ros2 launch vla_ros2_gz gz_smolvla.launch.py "
            f"dry_run:=false publish_actions_in_dry_run:=true "
            f"enable_actuation:={'true' if enable_actuation else 'false'} "
            f"control_hz:=2.0 pretrained:={pretrained}"
        ),
    ]
    launch = subprocess.Popen(
        launch_cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )

    class Recorder(Node):
        def __init__(self) -> None:
            super().__init__("gz_smolvla_recorder")
            self.latest_joint: JointState | None = None
            self.latest_action_dim: int | None = None
            self.frames: list[Image.Image] = []
            self.step = 0
            self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
            self.create_subscription(VLAAction, "/vla/action", self._action_cb, action_qos(10))

        def _joint_cb(self, msg: JointState) -> None:
            self.latest_joint = msg

        def _action_cb(self, msg: VLAAction) -> None:
            self.latest_action_dim = len(msg.data)

        def maybe_capture(self) -> None:
            if self.latest_joint is None:
                return
            if self.step % RENDER_EVERY != 0:
                self.step += 1
                return
            panel = _panel_from_joint_names(
                list(self.latest_joint.name),
                list(self.latest_joint.position),
            )
            self.frames.append(
                _annotate(
                    panel,
                    step=self.step,
                    instruction=instruction,
                    mode="gazebo",
                    action_dim=self.latest_action_dim,
                )
            )
            self.step += 1

    def _cleanup() -> None:
        try:
            os.killpg(launch.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        launch.wait(timeout=30)

    rclpy.init()
    node = Recorder()
    end = time.time() + duration_sec
    try:
        while time.time() < end and launch.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            node.maybe_capture()
        if launch.poll() is not None and launch.stderr:
            err = launch.stderr.read().decode("utf-8", errors="replace")
            if err.strip():
                print(err, file=sys.stderr)
            raise RuntimeError("gz_smolvla launch exited early")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _cleanup()

    _save_gif(out, node.frames, node.step)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--offline", action="store_true", help="kinematic loop without Gazebo")
    parser.add_argument("--duration-sec", type=float, default=45.0)
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--action-blend", type=float, default=0.35)
    parser.add_argument("--enable-actuation", action="store_true", default=True)
    parser.add_argument("--ros-domain-id", type=int, default=92)
    args = parser.parse_args()

    if args.offline:
        run_offline(
            out=args.out,
            steps=args.steps,
            episode=args.episode,
            device=args.device,
            pretrained=args.pretrained,
            local_files_only=args.local_files_only,
            action_blend=args.action_blend,
        )
        return

    run_ros(
        out=args.out,
        duration_sec=args.duration_sec,
        episode=args.episode,
        pretrained=args.pretrained,
        enable_actuation=args.enable_actuation,
        ros_domain_id=args.ros_domain_id,
    )


if __name__ == "__main__":
    main()
