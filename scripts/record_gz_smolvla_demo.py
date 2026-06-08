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
import contextlib
import json
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
    metrics_out: Path | None = None,
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
    start_joints = scene.joint_state.copy()
    max_joint_delta = 0.0
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
        max_joint_delta = max(
            max_joint_delta,
            float(np.max(np.abs(scene.joint_state - start_joints))),
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
    if metrics_out is not None:
        payload = {
            "pretrained": pretrained,
            "episode": episode,
            "instruction": instruction,
            "mode": "offline",
            "steps": steps,
            "frames_captured": len(frames),
            "actions_received": steps,
            "max_joint_delta_rad": max_joint_delta,
            "final_joint_positions": scene.joint_state.reshape(-1).tolist(),
            "action_blend": action_blend,
        }
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {metrics_out}")


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
    metrics_out: Path | None,
    warmup_sec: float,
) -> None:
    try:
        import rclpy
        from rclpy.node import Node
        from sensor_msgs.msg import JointState
        from vla_ros2_msgs.msg import VLAAction, VLAStatus
        from vla_ros2_ros.qos import action_qos, status_qos
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
    venv_bin = repo_root / ".venv-smolvla" / "bin"
    if venv_bin.is_dir():
        env["PATH"] = f"{venv_bin}{os.pathsep}{env.get('PATH', '')}"
    path_prefix = f"export PATH={venv_bin}:$PATH; " if venv_bin.is_dir() else ""

    launch_cmd = [
        "bash",
        "-lc",
        (
            f"set +u; source /opt/ros/jazzy/setup.bash; "
            f"source {install_setup}; set -u; "
            f"{path_prefix}"
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
            self.action_count = 0
            self.runtime_ready = False
            self.recording = False
            self.start_positions: list[float] | None = None
            self.max_joint_delta = 0.0
            self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)
            self.create_subscription(VLAAction, "/vla/action", self._action_cb, action_qos(10))
            self.create_subscription(VLAStatus, "/vla/status", self._status_cb, status_qos(10))

        def _status_cb(self, msg: VLAStatus) -> None:
            if msg.ready:
                self.runtime_ready = True

        def _begin_recording(self) -> None:
            if self.recording:
                return
            self.recording = True
            if self.latest_joint is not None and self.latest_joint.position:
                self.start_positions = list(self.latest_joint.position)
                self.max_joint_delta = 0.0
            self.get_logger().info("runtime ready; recording started")

        def _joint_cb(self, msg: JointState) -> None:
            self.latest_joint = msg
            if not self.recording:
                return
            positions = list(msg.position)
            if self.start_positions is None and positions:
                self.start_positions = positions
            if self.start_positions is not None and positions:
                delta = max(
                    abs(current - start)
                    for current, start in zip(positions, self.start_positions, strict=False)
                )
                self.max_joint_delta = max(self.max_joint_delta, delta)

        def _action_cb(self, msg: VLAAction) -> None:
            self.latest_action_dim = len(msg.data)
            self.action_count += 1
            if self.runtime_ready:
                self._begin_recording()

        def maybe_capture(self) -> None:
            if not self.recording or self.latest_joint is None:
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
        with contextlib.suppress(ProcessLookupError):
            os.killpg(launch.pid, signal.SIGTERM)
        launch.wait(timeout=30)

    rclpy.init()
    node = Recorder()
    warmup_end = time.time() + warmup_sec
    capture_end: float | None = None
    try:
        while launch.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.2)
            if not node.recording:
                if time.time() > warmup_end:
                    err = ""
                    if launch.stderr is not None:
                        err = launch.stderr.read().decode("utf-8", errors="replace")
                    if err.strip():
                        print(err, file=sys.stderr)
                    msg = (
                        f"runtime did not publish ready + action within {warmup_sec:.0f}s "
                        f"(actions={node.action_count}, ready={node.runtime_ready})"
                    )
                    raise RuntimeError(msg)
                continue
            if capture_end is None:
                capture_end = time.time() + duration_sec
            node.maybe_capture()
            if capture_end is not None and time.time() >= capture_end:
                break
        if launch.poll() is not None and not node.recording:
            err = ""
            if launch.stderr is not None:
                err = launch.stderr.read().decode("utf-8", errors="replace")
            if err.strip():
                print(err, file=sys.stderr)
            raise RuntimeError("gz_smolvla launch exited before inference started")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        _cleanup()

    _save_gif(out, node.frames, node.step)
    if metrics_out is not None:
        final_positions = (
            list(node.latest_joint.position) if node.latest_joint is not None else []
        )
        payload = {
            "pretrained": pretrained,
            "episode": episode,
            "instruction": instruction,
            "mode": "gazebo",
            "warmup_sec": warmup_sec,
            "duration_sec": duration_sec,
            "frames_captured": len(node.frames),
            "control_steps": node.step,
            "actions_received": node.action_count,
            "max_joint_delta_rad": node.max_joint_delta,
            "final_joint_positions": final_positions,
            "enable_actuation": enable_actuation,
        }
        metrics_out.parent.mkdir(parents=True, exist_ok=True)
        metrics_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {metrics_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--offline", action="store_true", help="kinematic loop without Gazebo")
    parser.add_argument("--duration-sec", type=float, default=45.0)
    parser.add_argument(
        "--warmup-sec",
        type=float,
        default=180.0,
        help="Wait for runtime ready + first /vla/action before capturing (Gazebo mode).",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--pretrained", default=DEFAULT_PRETRAINED)
    parser.add_argument("--local-files-only", action="store_true", default=True)
    parser.add_argument("--action-blend", type=float, default=0.35)
    parser.add_argument("--enable-actuation", action="store_true", default=True)
    parser.add_argument("--ros-domain-id", type=int, default=92)
    parser.add_argument("--metrics-out", type=Path, default=None)
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
            metrics_out=args.metrics_out,
        )
        return

    run_ros(
        out=args.out,
        duration_sec=args.duration_sec,
        episode=args.episode,
        pretrained=args.pretrained,
        enable_actuation=args.enable_actuation,
        ros_domain_id=args.ros_domain_id,
        metrics_out=args.metrics_out,
        warmup_sec=args.warmup_sec,
    )


if __name__ == "__main__":
    main()
