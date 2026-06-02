# Jetson + Remote GPU Deployment

The realistic deployment split for heavy VLA models is: a **lightweight robot-side
runtime** (e.g. on a Jetson or the robot computer) that talks ROS2 and runs the safety
guards, plus a **remote GPU box** that serves the heavyweight VLA model over HTTP. This
keeps heavy CUDA/transformer dependencies off the robot computer and out of the ROS2
environment.

This is an operations guide. It is runtime-centric and makes **no task-success claim**:
it wires up topology, serving, guards, and bridges; it does not assert robot-skill
quality.

## Topology

```mermaid
flowchart LR
  Cam[Camera / state] --> Node[ROS2 runtime node<br/>RemoteVLAClient<br/>Jetson / robot PC]
  Node -->|HTTP /v1/predict| Server[VLA server<br/>vla-zoo serve<br/>remote GPU box]
  Server -->|action| Node
  Node --> Guard[Clip + watchdog guards]
  Guard --> Bridge[MoveIt Servo / ros2_control bridge example]
  Bridge --> Controller[Deterministic controller 100-1000 Hz]
  Estop[E-stop] --> Bridge
```

The robot side stays lightweight: it only needs ROS2, the `vla_zoo` core, and `httpx`.
The GPU box carries the model dependencies in an isolated environment.

## 1. Remote GPU box: serve the model

Install the model's heavy extra in a dedicated environment (kept off the robot) and serve:

```bash
# on the GPU box, in an isolated env (see the per-model remote docs)
vla-zoo serve --model openvla --host 0.0.0.0 --port 8000 --device cuda:0
```

Generate a reproducible server plan and verify health before wiring the robot:

```bash
vla-zoo serve-plan --models openvla --markdown-out server_plan.md
vla-zoo remote-probe --model openvla --remote-url http://gpu-box:8000 --strict
```

`remote-probe` checks `/health` before `/v1/predict`, so a down or mismatched server fails
fast. See the per-model remote paths: [OpenVLA](openvla_remote.md),
[SmolVLA](smolvla_remote.md), [pi0](pi0_remote.md). GR00T stays
[blocked](groot_remote.md) until its stack exists.

## 2. Robot side (Jetson): ROS2 remote runtime

Run the runtime node in `remote` mode pointing at the GPU box. It defaults to dry-run safe
(no action publication unless explicitly enabled):

```bash
ros2 launch vla_zoo remote.launch.py \
  model_name:=openvla \
  remote_url:=http://gpu-box:8000 \
  dry_run:=true \
  require_image:=true \
  stale_image_timeout_sec:=1.0 \
  stale_instruction_timeout_sec:=5.0 \
  clip_actions:=true
```

See [ROS2 integration](ros2_integration.md) for topics, QoS, and parameters.

## 3. Safety guards (always on)

The node runs the pure, unit-tested guards from `vla_zoo.runtime.guard`:

- **Action clipping** clamps to the adapter's declared `low`/`high` (or configured
  `action_low`/`action_high`) and reports a clip rate.
- **Staleness watchdog** stops inference on stale image/instruction inputs and publishes a
  clear status (`waiting for image`, `stale image: <age>s`, `stale instruction: <age>s`).

See [Safety](safety.md). These shape and flag the action stream only; they never actuate.

## 4. Hardware bridge (example layer)

A bridge example consumes `/vla/action`, re-runs the guards, and forwards to the robot.
Both are **dry-run safe** (log-only without `--engage`):

```bash
# Cartesian / teleop path
python examples/ros2/moveit_servo_bridge.py            # dry-run
# controller-driven path
python examples/ros2/ros2_control_bridge.py            # dry-run
```

See the [MoveIt Servo](ros2_integration.md#moveit-servo-bridge-example-dry-run-safe) and
[ros2_control](ros2_integration.md#ros2_control-bridge-example-dry-run-safe) bridge
examples.

## What this guide does not provide

- Task-success or policy-quality numbers (out of scope; runtime-centric only).
- A turnkey hardware bridge — the bridges are examples; a real deployment must add an
  e-stop, workspace/joint-limit validation, and a high-rate deterministic controller.
- Jetson-specific CUDA/driver setup for local on-device inference; the recommended path
  keeps heavy inference on the remote GPU box.
