"""Record a single-process ROS2 remote runtime trace against a real vla-zoo server.

This hosts the real ``VLARuntimeNode`` (in ``runtime=remote`` mode), the synthetic input
node, and the log recorder under one executor, then writes the action/status/diagnostics
JSONL logs that ``vla-zoo ros remote-smoke-check`` consumes.

Why single-process: the standard flow is the 3-process ``smoke_record.launch.py``. That
needs cross-process DDS discovery, which relies on multicast. On hosts whose loopback
interface has no ``MULTICAST`` flag (``ip link show lo``), cross-process discovery fails.
Same-process discovery works, so this harness records the identical real path
(node -> RemoteVLAClient -> HTTP server, real ROS2 message types) without that dependency.

It is runtime-path evidence, not a task-success claim: the input is a synthetic frame.

Run (with the ROS2 overlay sourced and a server already serving the model):

    source /opt/ros/<distro>/setup.bash && source install/setup.bash
    python3 scripts/record_ros2_remote_trace.py \
        --model smolvla --remote-url http://127.0.0.1:8013 \
        --duration 22 --output-dir results/ros2_remote_smolvla
"""

from __future__ import annotations

import argparse
import tempfile
import time
from pathlib import Path


def _write_params(*, model, remote_url, instruction, task_id, output_dir, hz) -> str:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    text = f"""vla_runtime_node:
  ros__parameters:
    model_name: {model}
    runtime: remote
    remote_url: {remote_url}
    instruction_msg_type: vla_instruction
    require_image: true
    dry_run: true
    publish_actions_in_dry_run: true
    control_hz: {hz}
vla_smoke_input_node:
  ros__parameters:
    image_topic: /camera/image_raw
    instruction_topic: /vla/instruction
    joint_state_topic: /joint_states
    publish_hz: {hz}
    instruction: "{instruction}"
    task_id: {task_id}
vla_runtime_log_recorder:
  ros__parameters:
    action_topic: /vla/action
    status_topic: /vla/status
    diagnostics_topic: /diagnostics
    action_log_path: {out / "vla_actions.jsonl"}
    status_log_path: {out / "vla_status.jsonl"}
    diagnostics_log_path: {out / "vla_diagnostics.jsonl"}
    record_actions: true
    record_status: true
    record_diagnostics: true
    flush_every: 1
"""
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, encoding="utf-8"
    ) as fd:
        fd.write(text)
        return fd.name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="smolvla")
    parser.add_argument("--remote-url", default="http://127.0.0.1:8013")
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--task-id", default="ros2_remote_trace")
    parser.add_argument("--duration", type=float, default=22.0)
    parser.add_argument("--hz", type=float, default=5.0)
    parser.add_argument("--output-dir", default="results/ros2_remote_smolvla")
    args = parser.parse_args()

    params_file = _write_params(
        model=args.model,
        remote_url=args.remote_url,
        instruction=args.instruction,
        task_id=args.task_id,
        output_dir=args.output_dir,
        hz=args.hz,
    )

    import rclpy
    from rclpy.executors import MultiThreadedExecutor
    from vla_zoo_ros.log_recorder import RuntimeLogRecorder
    from vla_zoo_ros.node import VLARuntimeNode
    from vla_zoo_ros.smoke_input import VLASmokeInputNode

    rclpy.init(args=["--ros-args", "--params-file", params_file])
    runtime = VLARuntimeNode()
    recorder = RuntimeLogRecorder()
    inputs = VLASmokeInputNode()

    executor = MultiThreadedExecutor()
    for node in (runtime, recorder, inputs):
        executor.add_node(node)

    start = time.perf_counter()
    try:
        while time.perf_counter() - start < args.duration:
            executor.spin_once(timeout_sec=0.1)
    finally:
        for node in (inputs, runtime, recorder):
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    print(
        f"recorded ~{time.perf_counter() - start:.1f}s of {args.model} remote trace "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
