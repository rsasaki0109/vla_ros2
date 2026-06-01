# OpenVLA Prompt Probe

This is a direct OpenVLA adapter predict probe on one PyBullet RGB frame.
It is not a robot skill benchmark.

- status: `error`
- model: `openvla/openvla-7b`
- device: `cuda:0`
- dtype: `bfloat16`
- image source: `docs/assets/simulation_scripted.gif#frame77`

## Result

OpenVLA weights and optional dependencies were available, but local CUDA
execution did not complete because the GPU did not have enough free memory
during this run.

This is not counted as completed multi-task OpenVLA verification. Rerun with a
free GPU or use a remote OpenVLA server.
