# vla_ros2 Plan and Handoff

Updated: 2026-06-08 JST

## Position

`vla_ros2` is a ROS2-native on-robot runtime for Vision-Language-Action models.
It was refocused from the former `vla_zoo` (a broad runtime + benchmark +
adapter hub). The single goal now: run a VLA policy on a robot through ROS2.

```text
camera + instruction + robot state + timestamp
  -> stable VLA adapter boundary (local inference)
  -> typed VLAAction / VLAActionChunk
  -> ROS2 topic + /diagnostics
```

### What was removed in the pivot
- `benchmark/`, `compare/`, `demo/`, `docs/` Python packages and all generated
  report artifacts (GIF/HTML/leaderboard/roofline, `examples/`, `results/`,
  `log/`, `build/`, `install/`).
- The remote-GPU HTTP path: `runtime/remote.py`, `server.py`, `schemas.py`,
  `health.py`, and the corresponding registry/node/config/launch plumbing.
  Inference is **on-robot local only**.
- The large multi-command CLI, replaced by a minimal `vla-ros2 list / predict`.

## Current shape

- Python API: `load_model()`, `list_models()` (`src/vla_ros2/__init__.py`).
- Core contracts: `VLAObservation`, `VLAAction`, `VLAActionChunk`, `ActionSpec`
  (`src/vla_ros2/core/types.py`); adapter base + registry in `core/`.
- Adapters: `dummy`, `random`, `scripted`, `openvla`, `smolvla`, `pi0`, `groot`.
- Runtime: `runtime/local.py`, `runtime/guard.py` (clip + watchdog),
  `runtime/diagnostics.py`.
- ROS2: `ros2/vla_ros2` (node, launch, config) + `ros2/vla_ros2_msgs` (messages).

## Conventions

- New code matches the package name `vla_ros2` and module `vla_ros2_ros`.
- Keep `dry_run` defaulting to `true` in every launch file (safety test guards it).
- `load_model` only supports `runtime="local"`.

## Verify

```bash
pip install -e ".[dev]"
ruff check src tests ros2
mypy src/vla_ros2
pytest
# ROS2: colcon build --packages-select vla_ros2_msgs vla_ros2
#       ros2 launch vla_ros2 dummy.launch.py
```

## Possible next steps
- Real-robot bring-up notes / a hardware integration guide under `ros2/`.
- A small launch test that actually spins the node with `launch_testing`.
- Trim adapter `description` metadata that still mentions remote-first roles.
