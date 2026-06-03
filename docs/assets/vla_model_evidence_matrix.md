# VLA Model Evidence Matrix

This matrix records what has actually been exercised through `vla_zoo` and what is still only planned. It is intentionally runtime-centric.

It is not a model-quality leaderboard. `verified` means the repository contains a checked-in runtime artifact or deterministic test for that cell.

| Model | Family | Adapter status | Contract | GPU inference | Remote server | ROS2 remote | PyBullet tasks | Policy quality |
|---|---|---|---|---|---|---|---|---|
| `dummy` | dry-run baseline | available | verified: Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats. | not_applicable: Dummy is a neutral runtime baseline and does not require GPU inference. | verified: Temporary HTTP server returned typed actions through /v1/predict. | verified: Recorded ROS2 node to RemoteVLAClient to HTTP dummy server path. | verified: Recorded on all three deterministic PyBullet smoke tasks. | not_applicable: Dummy intentionally emits neutral actions; it is not a VLA policy. |
| `scripted` | rule-based baseline | available | verified: Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats. | not_applicable: scripted is a lightweight baseline and does not require GPU inference. | partial: The runtime can serve this adapter, but the recorded remote smoke sample is dummy-only. | planned: ROS2 remote recording path exists; no dedicated recorded sample is checked in yet. | verified: Recorded on all three deterministic PyBullet smoke tasks. | not_applicable: scripted is a baseline for runtime comparison, not a VLA model. |
| `random` | stochastic baseline | available | verified: Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats. | not_applicable: random is a lightweight baseline and does not require GPU inference. | partial: The runtime can serve this adapter, but the recorded remote smoke sample is dummy-only. | planned: ROS2 remote recording path exists; no dedicated recorded sample is checked in yet. | verified: Recorded on all three deterministic PyBullet smoke tasks. | not_applicable: random is a baseline for runtime comparison, not a VLA model. |
| `openvla` | VLA foundation model | available | verified: Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats. | verified: 4-bit (nf4) loading fits a 16 GB consumer GPU: ~4.6 GB peak VRAM, ~1.1-2.7 s per inference. Measured via scripts/measure_openvla_runtime.py. | verified: A real `vla-zoo serve --model openvla --load-in-4bit` server passed a health-first probe and returned a typed 7-DoF action over HTTP /v1/predict (recorded end-to-end on a 16 GB GPU). | verified: The real VLARuntimeNode ran in remote mode against a live OpenVLA-7b (4-bit) server: recorded 7 RemoteVLAClient actions + 143 status/diagnostics with 0 inference errors (vla-zoo ros remote-smoke-check passed). | partial: Recorded a real-scene action probe: OpenVLA-7b (4-bit) driven on real PyBullet-rendered frames (21 queries, 7-DoF, latency p50 ~2.0 s), exercising the real image preprocessing path that synthetic-frame probes skip. Runtime path on real renders, not a task-success benchmark. | not_verified: No task-success or robot-skill claim is made for OpenVLA in this repository. The real-scene action probe upgrades the input from synthetic noise to a real render but makes no task-success or policy-quality claim. |
| `pi0` | pi-family VLA | experimental | verified: Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats. | planned: Needs a dedicated openpi or LeRobot serving environment and a recorded action probe. | planned: Remote-first deployment path with a reproducible pi0 server plan and LeRobot/openpi version-compatibility docs; a recorded pi0 /v1/predict run from a version-matched server is still needed. | planned: ROS2 remote launch can target a pi0 server; checked-in action logs are still needed. | planned: Runner has the observation plumbing; real pi0 remote task traces are not checked in. | not_verified: No task-success or robot-skill claim is made for pi0/openpi in this repository. |
| `smolvla` | LeRobot policy | missing optional deps: pip install "vla_zoo[smolvla]" | verified: Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats. | verified: CUDA inference measured at ~0.97 GB peak VRAM and ~60-133 ms steady latency (real-time capable). Captured via scripts/measure_lerobot_runtime.py. | verified: A real `vla-zoo serve --model smolvla` server passed a health-first probe and returned a typed 6-DoF action over HTTP /v1/predict (recorded end-to-end). | verified: The real VLARuntimeNode ran in remote mode against a live SmolVLA server: recorded 14 RemoteVLAClient actions + 106 status/diagnostics with 0 inference errors (vla-zoo ros remote-smoke-check passed). | partial: Recorded a real-scene action probe: SmolVLA driven on real PyBullet-rendered frames (21 queries, 6-DoF, latency p50 ~382 ms with a fresh encode per query), exercising the real image preprocessing path that synthetic-frame probes skip. Runtime path on real renders, not a task-success benchmark. | not_verified: SmolVLA base still needs robot/task-specific fine-tuning and calibration. The real-scene action probe upgrades the input from synthetic noise to a real render but makes no task-success or policy-quality claim. |
| `groot` | humanoid/generalist foundation model | experimental | partial: Runtime contract is declared, but GR00T is blocked until the NVIDIA Isaac GR00T stack is wired in; no inference is implemented. | blocked: Requires the dedicated NVIDIA Isaac GR00T stack and a recorded action probe; blocked until a real serving adapter exists. | blocked: Expected to run through a remote serving environment once a real GR00T serving adapter lands; blocked until then. | planned: ROS2 remote launch can target a GR00T server after real serving support exists. | not_verified: No real GR00T action traces are checked in. | not_verified: No task-success or robot-skill claim is made for GR00T in this repository. |

## Evidence Links

### dummy

- Upstream: vla_zoo
- Next step: Keep this as the CI and ROS2 remote wiring baseline.
- Caveat: Exercised by unit tests, CLI predict, ROS2 dry-run launch, and PyBullet smoke reports.

| Cell | Status | Evidence |
|---|---|---|
| Contract | verified | Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats.<br>[adapter card](../adapters/dummy.md) |
| Local runtime | verified | CPU local predict path is covered by tests, CLI predict, and smoke reports.<br>[action playground check](../reports/model_comparison.md) |
| GPU inference | not_applicable | Dummy is a neutral runtime baseline and does not require GPU inference.<br>- |
| Remote server | verified | Temporary HTTP server returned typed actions through /v1/predict.<br>[remote runtime smoke](../reports/remote_runtime_smoke.md) |
| ROS2 remote | verified | Recorded ROS2 node to RemoteVLAClient to HTTP dummy server path.<br>[ROS2 remote evidence](sample_ros2_remote_dummy/remote_smoke_check.md) |
| PyBullet tasks | verified | Recorded on all three deterministic PyBullet smoke tasks.<br>[baseline task report](sample_task_verification/baseline_tasks.md) |
| Policy quality | not_applicable | Dummy intentionally emits neutral actions; it is not a VLA policy.<br>- |

### scripted

- Upstream: vla_zoo
- Next step: Keep using this baseline for simulation/report regression checks.
- Caveat: Exercised in deterministic PyBullet comparison reports as a scripted smoke baseline.

| Cell | Status | Evidence |
|---|---|---|
| Contract | verified | Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats.<br>[adapter card](../adapters/scripted.md) |
| Local runtime | verified | CPU local predict path is exercised as a lightweight baseline.<br>[action playground check](../reports/model_comparison.md) |
| GPU inference | not_applicable | scripted is a lightweight baseline and does not require GPU inference.<br>- |
| Remote server | partial | The runtime can serve this adapter, but the recorded remote smoke sample is dummy-only.<br>[server plan](sample_compare_suite/gpu_server_plan.md) |
| ROS2 remote | planned | ROS2 remote recording path exists; no dedicated recorded sample is checked in yet.<br>[ROS2 remote plan](ros2_remote_smoke_plan.md) |
| PyBullet tasks | verified | Recorded on all three deterministic PyBullet smoke tasks.<br>[baseline task report](sample_task_verification/baseline_tasks.md) |
| Policy quality | not_applicable | scripted is a baseline for runtime comparison, not a VLA model.<br>- |

### random

- Upstream: vla_zoo
- Next step: Keep using this baseline for simulation/report regression checks.
- Caveat: Exercised in deterministic PyBullet comparison reports as a seeded baseline.

| Cell | Status | Evidence |
|---|---|---|
| Contract | verified | Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats.<br>[adapter card](../adapters/random.md) |
| Local runtime | verified | CPU local predict path is exercised as a lightweight baseline.<br>[action playground check](../reports/model_comparison.md) |
| GPU inference | not_applicable | random is a lightweight baseline and does not require GPU inference.<br>- |
| Remote server | partial | The runtime can serve this adapter, but the recorded remote smoke sample is dummy-only.<br>[server plan](sample_compare_suite/gpu_server_plan.md) |
| ROS2 remote | planned | ROS2 remote recording path exists; no dedicated recorded sample is checked in yet.<br>[ROS2 remote plan](ros2_remote_smoke_plan.md) |
| PyBullet tasks | verified | Recorded on all three deterministic PyBullet smoke tasks.<br>[baseline task report](sample_task_verification/baseline_tasks.md) |
| Policy quality | not_applicable | random is a baseline for runtime comparison, not a VLA model.<br>- |

### openvla

- Upstream: OpenVLA
- Next step: Local 4-bit GPU, remote /v1/predict, a ROS2 remote trace, and a real-scene PyBullet action probe are all verified; the remaining gap is task-success / policy quality, which stays unclaimed without a graded benchmark.
- Caveat: Adapter scaffold and prompt path are documented. In the latest local run, the CUDA prompt probe did not complete because free GPU memory was insufficient.

| Cell | Status | Evidence |
|---|---|---|
| Contract | verified | Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats.<br>[adapter card](../adapters/openvla.md) |
| Local runtime | verified | OpenVLA-7b loaded and predicted a 7-DoF action through the public adapter on a local RTX 4070 Ti SUPER (4-bit), with measured load time, VRAM, and latency.<br>[local runtime evidence](../openvla_local_runtime.md) |
| GPU inference | verified | 4-bit (nf4) loading fits a 16 GB consumer GPU: ~4.6 GB peak VRAM, ~1.1-2.7 s per inference. Measured via scripts/measure_openvla_runtime.py.<br>[local runtime evidence](../openvla_local_runtime.md) |
| Remote server | verified | A real `vla-zoo serve --model openvla --load-in-4bit` server passed a health-first probe and returned a typed 7-DoF action over HTTP /v1/predict (recorded end-to-end on a 16 GB GPU).<br>[OpenVLA remote probe](sample_task_verification/openvla_remote_probe.md), [OpenVLA remote path](../openvla_remote.md) |
| ROS2 remote | verified | The real VLARuntimeNode ran in remote mode against a live OpenVLA-7b (4-bit) server: recorded 7 RemoteVLAClient actions + 143 status/diagnostics with 0 inference errors (vla-zoo ros remote-smoke-check passed).<br>[ROS2 remote smoke check](sample_ros2_remote_openvla/remote_smoke_check.md), [ROS2 remote plan](ros2_remote_smoke_plan.md) |
| PyBullet tasks | partial | Recorded a real-scene action probe: OpenVLA-7b (4-bit) driven on real PyBullet-rendered frames (21 queries, 7-DoF, latency p50 ~2.0 s), exercising the real image preprocessing path that synthetic-frame probes skip. Runtime path on real renders, not a task-success benchmark.<br>[real-scene action probe](sample_pybullet_openvla/runtime_action_probe.md), [external adapter status](sample_task_verification/external_adapter_status.md) |
| Policy quality | not_verified | No task-success or robot-skill claim is made for OpenVLA in this repository. The real-scene action probe upgrades the input from synthetic noise to a real render but makes no task-success or policy-quality claim.<br>- |

### pi0

- Upstream: openpi / LeRobot
- Next step: Stand up a dedicated pi0/openpi server and record /v1/predict plus ROS2 remote logs.
- Caveat: Remote-first adapter path is implemented. Local real-model action probe has not completed in this repository; use a dedicated GPU serving environment.

| Cell | Status | Evidence |
|---|---|---|
| Contract | verified | Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats.<br>[adapter card](../adapters/pi0.md) |
| Local runtime | blocked | Local load fails on a concrete config-schema mismatch: the cached lerobot/pi0 checkpoint carries PI0Config fields (resize_imgs_with_padding, adapt_to_pi_aloha, num_steps, ...) that LeRobot 0.5.1 rejects. Needs a version-matched checkpoint.<br>[pi0 compatibility note](sample_task_verification/pi0_compatibility_probe.md) |
| GPU inference | planned | Needs a dedicated openpi or LeRobot serving environment and a recorded action probe.<br>[adapter card](../adapters/pi0.md) |
| Remote server | planned | Remote-first deployment path with a reproducible pi0 server plan and LeRobot/openpi version-compatibility docs; a recorded pi0 /v1/predict run from a version-matched server is still needed.<br>[pi0 remote path](../pi0_remote.md), [pi0 server plan](pi0_server_plan.md) |
| ROS2 remote | planned | ROS2 remote launch can target a pi0 server; checked-in action logs are still needed.<br>[ROS2 remote plan](ros2_remote_smoke_plan.md) |
| PyBullet tasks | planned | Runner has the observation plumbing; real pi0 remote task traces are not checked in.<br>[external adapter status](sample_task_verification/external_adapter_status.md) |
| Policy quality | not_verified | No task-success or robot-skill claim is made for pi0/openpi in this repository.<br>- |

### smolvla

- Upstream: LeRobot SmolVLA
- Next step: Local GPU, remote /v1/predict, a ROS2 remote trace, and a real-scene PyBullet action probe are all verified; the remaining gap is task-success / policy quality, which stays unclaimed without a fine-tuned checkpoint and graded benchmark.
- Caveat: Local CUDA inference-path probe completed with lerobot/smolvla_base, including a PyBullet-rendered multi-camera/state observation path. This is not a robot task-success claim.

| Cell | Status | Evidence |
|---|---|---|
| Contract | verified | Built-in adapter metadata declares inputs, action shape, runtime mode, and caveats.<br>[adapter card](../adapters/smolvla.md) |
| Local runtime | verified | lerobot/smolvla_base loaded and predicted a 6-DoF action through load_model('smolvla') on a local RTX 4070 Ti SUPER, with measured load/VRAM/latency.<br>[local runtime evidence](../smolvla_local_runtime.md) |
| GPU inference | verified | CUDA inference measured at ~0.97 GB peak VRAM and ~60-133 ms steady latency (real-time capable). Captured via scripts/measure_lerobot_runtime.py.<br>[local runtime evidence](../smolvla_local_runtime.md) |
| Remote server | verified | A real `vla-zoo serve --model smolvla` server passed a health-first probe and returned a typed 6-DoF action over HTTP /v1/predict (recorded end-to-end).<br>[SmolVLA remote probe](sample_task_verification/smolvla_remote_probe.md), [SmolVLA remote plan](smolvla_remote_smoke_plan.md) |
| ROS2 remote | verified | The real VLARuntimeNode ran in remote mode against a live SmolVLA server: recorded 14 RemoteVLAClient actions + 106 status/diagnostics with 0 inference errors (vla-zoo ros remote-smoke-check passed).<br>[ROS2 remote smoke check](sample_ros2_remote_smolvla/remote_smoke_check.md), [ROS2 remote plan](ros2_remote_smoke_plan.md) |
| PyBullet tasks | partial | Recorded a real-scene action probe: SmolVLA driven on real PyBullet-rendered frames (21 queries, 6-DoF, latency p50 ~382 ms with a fresh encode per query), exercising the real image preprocessing path that synthetic-frame probes skip. Runtime path on real renders, not a task-success benchmark.<br>[real-scene action probe](sample_pybullet_smolvla/runtime_action_probe.md), [SmolVLA PyBullet report](sample_task_verification/smolvla_pybullet_report.html) |
| Policy quality | not_verified | SmolVLA base still needs robot/task-specific fine-tuning and calibration. The real-scene action probe upgrades the input from synthetic noise to a real render but makes no task-success or policy-quality claim.<br>- |

### groot

- Upstream: Isaac GR00T
- Next step: Replace the placeholder with a real GR00T serving adapter before claiming inference.
- Caveat: Experimental placeholder only; blocked until the NVIDIA Isaac GR00T stack is wired in. No GR00T inference is implemented or verified in this repository.

| Cell | Status | Evidence |
|---|---|---|
| Contract | partial | Runtime contract is declared, but GR00T is blocked until the NVIDIA Isaac GR00T stack is wired in; no inference is implemented.<br>[adapter card](../adapters/groot.md), [blocked status](../groot_remote.md) |
| Local runtime | blocked | Blocked until the NVIDIA Isaac GR00T stack is wired in; no GR00T inference ships and the adapter raises instead of fabricating actions.<br>[blocked status](../groot_remote.md), [external adapter status](sample_task_verification/external_adapter_status.md) |
| GPU inference | blocked | Requires the dedicated NVIDIA Isaac GR00T stack and a recorded action probe; blocked until a real serving adapter exists.<br>[blocked status](../groot_remote.md) |
| Remote server | blocked | Expected to run through a remote serving environment once a real GR00T serving adapter lands; blocked until then.<br>[blocked status](../groot_remote.md), [GPU server plan](sample_compare_suite/gpu_server_plan.md) |
| ROS2 remote | planned | ROS2 remote launch can target a GR00T server after real serving support exists.<br>[ROS2 remote plan](ros2_remote_smoke_plan.md) |
| PyBullet tasks | not_verified | No real GR00T action traces are checked in.<br>- |
| Policy quality | not_verified | No task-success or robot-skill claim is made for GR00T in this repository.<br>- |

## Reading Rules

- `verified`: checked-in tests, logs, or reports exist for the runtime path.
- `partial`: the adapter path exists, but the evidence is incomplete for that cell.
- `planned`: commands or scaffolding exist, but a checked-in run is still missing.
- `blocked`: the current repo run could not complete due to dependency, memory, or stack limits.
- `not_verified`: no claim is made.
- `not_applicable`: the cell does not apply to that baseline or adapter.

For real robots, this matrix must be paired with action clipping, stale-action watchdogs, emergency stop integration, and a hardware-specific bridge.
