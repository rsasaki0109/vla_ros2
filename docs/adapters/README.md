# vla_zoo Adapter Cards

These cards document adapter runtime contracts. They do not claim model quality, zero-shot success, or hardware readiness.

| Adapter | Family | Action | Runtime | Verification | Card |
|---|---|---|---|---|---|
| `dummy` | dry-run baseline | eef_delta (7,) | local: supported; remote: supported | Exercised by unit tests, CLI predict, ROS2 dry-run launch, and PyBullet smoke reports. | [card](dummy.md) |
| `scripted` | rule-based baseline | eef_delta (7,) | local: supported; remote: supported | Exercised in deterministic PyBullet comparison reports as a scripted smoke baseline. | [card](scripted.md) |
| `random` | stochastic baseline | eef_delta (7,) | local: supported; remote: supported | Exercised in deterministic PyBullet comparison reports as a seeded baseline. | [card](random.md) |
| `openvla` | VLA foundation model | eef_delta (7,) | local: supported with optional ML dependencies; remote: recommended for robot-side ROS2 | Adapter scaffold and prompt path are documented. In the latest local run, the CUDA prompt probe did not complete because free GPU memory was insufficient. | [card](openvla.md) |
| `pi0` | pi-family VLA | custom checkpoint-specific; lerobot/pi0 is (6,), lerobot/pi0_base is (32,) | local: disabled by default; enable_local=True in a dedicated GPU env; remote: recommended | Remote-first adapter path is implemented. Local real-model action probe has not completed in this repository; use a dedicated GPU serving environment. | [card](pi0.md) |
| `smolvla` | LeRobot policy | custom checkpoint-specific; lerobot/smolvla_base is (6,) | local: supported with optional LeRobot dependencies; remote: recommended | Local CUDA inference-path probe completed with lerobot/smolvla_base, including a PyBullet-rendered multi-camera/state observation path. This is not a robot task-success claim. | [card](smolvla.md) |
| `groot` | humanoid/generalist foundation model | custom adapter-specific | local: experimental placeholder; remote: recommended | Experimental placeholder only; real GR00T inference is not verified in this repository yet. | [card](groot.md) |

Generate these files from the registry:

```bash
vla-zoo compare cards --out-dir docs/adapters
```
