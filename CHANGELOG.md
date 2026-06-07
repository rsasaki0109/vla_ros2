# Changelog

All notable changes to `vla_ros2` are documented here.

## [0.1.0] - Unreleased

Initial runtime seed for a ROS2-native VLA adapter hub.

### Added

- Stable Python API: `load_model()` and `list_models()`.
- Core typed runtime boundary: `VLAObservation`, `VLAAction`, `VLAActionChunk`, `ActionSpec`.
- Built-in adapter registry with Python entry-point support.
- Always-available `dummy` adapter.
- Lazy OpenVLA adapter scaffold.
- Remote inference server/client with FastAPI schemas.
- Typer CLI: list, info, predict, serve, bench, demo, compare.
- PyBullet pick-and-place smoke demo and GIF generation.
- PyBullet runtime comparison commands, remote endpoint maps, and JSON manifests.
- Static HTML report and interactive dashboard generation for comparison results.
- GitHub Pages demo site under `docs/`.
- Dashboard preview and social preview image generation for README and Pages.
- ROS2 packages, messages, node, launch files, and dry-run configs.
- Benchmark scaffolds and smoke benchmark runner.
- Architecture, adapter, ROS2, benchmark, comparison, and safety docs.
- GitHub issue templates, PR template, labels, and contribution guidance.

### Safety

- No direct robot actuation in the core runtime.
- Dry-run ROS2 launch path.
- Heavy model dependencies remain optional.
- Tests run without GPU or model downloads.
