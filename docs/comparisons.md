# Comparisons

| Project | Primary value | vla_zoo relationship |
|---|---|---|
| LeRobot | datasets, policies, training, hardware workflows | vla_zoo should run LeRobot/SmolVLA policies through ROS2 |
| OpenVLA | open VLA model and fine-tuning code | vla_zoo wraps it behind a stable runtime and ROS2 topics |
| openpi | pi-family model code and checkpoints | vla_zoo provides deployment and runtime boundary |
| Isaac GR00T | humanoid/generalist foundation model stack | vla_zoo provides experimental adapter and ROS2-facing interface |
| SimplerEnv | real-to-sim policy evaluation benchmark | vla_zoo can run adapters against it with unified metrics |
| LIBERO | manipulation benchmark suites | vla_zoo can expose LIBERO tasks through the same model API |
| Genesis | scalable simulation platform | vla_zoo treats Genesis as a benchmark/runtime backend |

`vla_zoo` is intentionally infrastructure-focused. It is the ROS2-native runtime boundary around model projects and benchmark projects, not a replacement for them.

## Runtime Comparison Workflow

Use the adapter table first. It does not load model weights:

```bash
vla-zoo compare adapters
```

Then run the same deterministic PyBullet smoke scene for each runtime path:

```bash
vla-zoo compare pybullet --models dummy,openvla,pi0,smolvla,groot
```

The local comparison skips heavy OpenVLA loading by default to avoid accidental downloads. For real model-to-model checks, run each VLA behind a remote server and compare from the robot-side environment:

```bash
vla-zoo compare pybullet \
  --models openvla,pi0,smolvla,groot \
  --runtime remote \
  --remote-url http://gpu-box:8000
```

The comparison output is intentionally runtime-centric: frames, adapter query count, adapter errors, latency, and action magnitude. It is not a claim of task success or model quality.
