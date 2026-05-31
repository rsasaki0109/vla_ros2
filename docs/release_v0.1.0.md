# v0.1.0 Release Notes Draft

`vla_zoo` v0.1.0 is the runtime seed for a ROS2-native VLA infrastructure layer.

## Highlights

- Python API: `from vla_zoo import load_model, list_models`
- No-GPU dummy adapter for CI, docs, demos, and dry-run operation
- OpenVLA adapter scaffold with lazy optional dependencies
- Local and remote inference runtime paths
- FastAPI inference server and HTTP client
- ROS2 node, messages, launch files, and dry-run configs
- PyBullet pick-and-place smoke demo with generated GIF
- Adapter comparison CLI with per-model remote endpoint manifests
- Self-contained HTML runtime comparison reports
- Smoke benchmark abstraction and placeholder backends
- Safety-first docs: no direct actuation by default

## Try It

```bash
pip install -e ".[dev,cli,server,sim]"
vla-zoo predict --model dummy --instruction "hello"
vla-zoo demo pybullet --model dummy --out docs/assets/simulation_pick_place.gif
vla-zoo compare adapters
```

Remote smoke path:

```bash
vla-zoo serve --model dummy --host 127.0.0.1 --port 8010
vla-zoo compare pybullet --manifest examples/compare/pybullet_dummy_remote.json
```

ROS2 dry-run path:

```bash
pip install -e .
colcon build --base-paths ros2 --symlink-install
source install/setup.bash
ros2 launch vla_zoo dummy.launch.py
```

## Known Limits

- `vla_zoo` does not train VLA models.
- Real adapters may require large GPUs and upstream model licenses.
- Real hardware deployment requires robot-specific bridge packages and safety checks.
- OpenVLA, openpi, SmolVLA, and GR00T are external projects; weights are not redistributed.

## Validation Checklist

- [ ] `pip install -e ".[dev,cli,server,sim]"`
- [ ] `ruff check .`
- [ ] `mypy src/vla_zoo`
- [ ] `pytest`
- [ ] `vla-zoo predict --model dummy --instruction "hello"`
- [ ] `vla-zoo compare adapters`
- [ ] `vla-zoo demo pybullet --model dummy --out docs/assets/simulation_pick_place.gif`
- [ ] `vla-zoo compare pybullet --models dummy --html-out /tmp/vla_zoo_report.html`
- [ ] optional ROS2 smoke build
