# OpenVLA Prompt Probe

This is a direct OpenVLA adapter predict probe on one PyBullet RGB frame.
It is not a robot skill benchmark.

- status: `resolved` (originally `error` on bf16; fixed via 4-bit)
- model: `openvla/openvla-7b`
- device: `cuda:0`
- dtype: `bfloat16` (original attempt) / `nf4-4bit` (resolution)
- image source: `docs/assets/simulation_scripted.gif#frame77`

## Result

OpenVLA weights and optional dependencies were available, but this bf16 run did
not complete because the GPU did not have enough free memory: bf16 weights are
~15 GB and do not fit alongside activations on a 16 GB card.

## Resolution

This was resolved by loading in 4-bit (nf4). OpenVLA-7b now loads and predicts a
7-DoF action on the same class of GPU (GPU) at ~4.6 GB peak VRAM
and ~1.1-2.7 s latency. See the measured profile in
[OpenVLA local runtime evidence](../../openvla_local_runtime.md).

This is still a runtime-path result, not a multi-task OpenVLA skill benchmark.
