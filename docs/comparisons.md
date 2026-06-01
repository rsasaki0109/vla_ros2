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
