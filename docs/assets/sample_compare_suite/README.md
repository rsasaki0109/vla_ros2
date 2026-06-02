# vla_zoo Comparison Suite

Generated at: `2026-06-02T01:22:13.818532+00:00`

This artifact directory is intended for README snippets, issue reports, and release notes. It compares VLA runtime integration shape first, then optional deterministic PyBullet smoke-scene telemetry.

## Artifacts

- `method_profiles.json`: structured adapter integration profiles
- `method_profiles.md`: README-ready adapter integration table
- `pybullet_results.json`: deterministic smoke-scene runtime and task telemetry
- `pybullet_results.md`: README-ready PyBullet comparison table
- `pybullet_report.html`: self-contained PyBullet comparison report
- `runtime_dashboard.html`: interactive static dashboard for PyBullet results

## Reproduce

```bash
vla-zoo compare suite --out-dir docs/assets/sample_compare_suite --models dummy,scripted,random --runtime local --instruction 'pick up the red block' --model-call-every 24 --render-stride 80
```

## VLA Method Profiles

| Method | Family | Role | Inputs | Action | Chunks | Local | Remote | Status |
|---|---|---|---|---|---|---|---|---|
| `dummy` | dry-run baseline | CI/runtime smoke sanity check | image optional<br>instruction optional<br>state optional | eef_delta (7,): zero 7-DoF end-effector delta | optional via chunk_size | supported | supported | available |
| `groot` | humanoid/generalist foundation model | experimental humanoid/generalist adapter target | multimodal observations<br>instruction/task context<br>robot state expected | custom adapter-specific: humanoid/generalist action interface | adapter-specific | experimental placeholder | recommended | experimental |
| `openvla` | VLA foundation model | single-image VLA reference adapter | single RGB image<br>natural language instruction<br>optional unnormalization key | eef_delta (7,): OpenVLA-style 7-DoF action | no | supported with optional ML dependencies | recommended for robot-side ROS2 | missing optional deps: pip install "vla_zoo[openvla]" |
| `pi0` | pi-family VLA | remote-first action-chunk VLA target | images per policy config<br>natural language instruction<br>robot state expected | custom checkpoint-specific; lerobot/pi0 is (6,), lerobot/pi0_base is (32,): policy-specific continuous manipulation action | expected | disabled by default; enable_local=True in a dedicated GPU env | recommended | experimental |
| `random` | stochastic baseline | action plumbing and visualization stress check | instruction optional<br>image optional<br>state optional | eef_delta (7,): seeded random 7-DoF end-effector delta | no | supported | supported | available |
| `scripted` | rule-based baseline | upper-bound sanity check for the scripted smoke scene | phase metadata<br>instruction optional<br>image optional | eef_delta (7,): phase-aware 7-DoF end-effector delta | no | supported | supported | available |
| `smolvla` | LeRobot policy | multi-camera/state/action-chunk compact VLA target | multi-camera images<br>natural language instruction<br>robot state | custom checkpoint-specific; lerobot/smolvla_base is (6,): policy-specific continuous action | internal queue; chunk output optional | supported with optional LeRobot dependencies | recommended | missing optional deps: pip install "vla_zoo[smolvla]" |

These profiles describe runtime integration shape, not model quality. External model weights, datasets, and licenses are not redistributed by vla_zoo.

## PyBullet VLA Runtime Comparison

| Model | Runtime | Endpoint | OK | Frames | Queries | Errors | Task | Lifted | Goal dist m | Cube moved m | Phase | Mean latency ms | Mean abs action | Note |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `dummy` | `local` | - | true | 10 | 1 | 0 | true | true | 0.136 | 0.471 | 1.00 | 0.04 | 0.000 | - |
| `scripted` | `local` | - | true | 10 | 1 | 0 | true | true | 0.136 | 0.471 | 1.00 | 0.17 | 0.375 | - |
| `random` | `local` | - | true | 10 | 1 | 0 | true | true | 0.136 | 0.471 | 1.00 | 0.09 | 0.117 | - |

This is a runtime smoke comparison on the same deterministic PyBullet scene. It measures adapter availability, query behavior, errors, latency, action magnitude, and scripted-scene task telemetry; it is not a model-quality benchmark.

## Scope

- Method profiles do not load model weights.
- PyBullet reports are deterministic runtime smoke checks, not model-quality claims.
- External model projects and checkpoints are not redistributed by vla_zoo.
- Real robot deployment still requires robot-specific action bridges and safety checks.
