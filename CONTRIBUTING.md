# Contributing

Thanks for helping make `vla_ros2` a practical ROS2-native runtime layer for VLA policies.

## Development Setup

```bash
pip install -e ".[dev]"
ruff check src tests ros2
mypy src/vla_ros2
pytest
```

ROS2 packages (see [ros2/WORKSPACE.md](ros2/WORKSPACE.md)):

```bash
./scripts/bootstrap_ros2_workspace.sh
source install/setup.bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
ros2 launch vla_ros2 dummy.launch.py
```

## Contribution Areas

- Adapters: OpenVLA variants, SmolVLA, openpi/pi0, GR00T-style remote runtimes.
- Runtime: HTTP server/client, scheduling, schemas, health checks.
- ROS2: launch files, QoS, diagnostics, lifecycle support, safe bridge examples.
- Benchmarks: smoke tasks, LIBERO, SimplerEnv, rosbag replay, Genesis, Isaac.
- Docs: deployment guides, model cards, safety notes, reproducible examples.

## Adapter Requirements

Every adapter should document:

- input requirements
- output `ActionSpec`
- expected control rate
- whether it emits action chunks
- whether it needs proprioception
- local vs remote support
- optional dependencies and install extras
- upstream license caveats

Do not vendor external model repositories or weights. Keep heavy dependencies behind extras.

## Safety Rules

- Do not add direct robot actuation to the core runtime.
- Default examples should be dry-run or message-publishing only.
- Hardware bridges must include stale-action timeout, clipping, watchdog guidance, and explicit opt-in.
- Tests must not require a GPU or download model weights.

## Pull Requests

Before opening a PR, run:

```bash
ruff check src tests ros2
mypy src/vla_ros2
pytest
```

If ROS2 files changed, also run `./scripts/bootstrap_ros2_workspace.sh` (or the
manual steps in `ros2/WORKSPACE.md`) and `colcon test --packages-select vla_ros2
--python-testing pytest` when ROS 2 is available.
