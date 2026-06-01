# Contributing

Thanks for helping make `vla_zoo` a practical ROS2-native runtime layer for VLA policies.

## Development Setup

```bash
pip install -e ".[dev,cli,server,sim]"
ruff check .
mypy src/vla_zoo
pytest
```

ROS2 packages are built separately:

```bash
pip install -e .
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
ros2 launch vla_zoo dummy.launch.py
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
ruff check .
mypy src/vla_zoo
pytest
```

If ROS2 files changed, also run a ROS2 smoke build when available. Include the commands you ran in the PR description.
