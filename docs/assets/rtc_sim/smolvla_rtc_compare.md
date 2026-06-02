# Real-Time Chunking scheduler simulation

- Source: `pybullet-rtc-record:pick_red_block (model smolvla, real per-cycle delays, p50 16 ticks @ 30 Hz, late cycles 0)`
- Chunks: 81 (horizon 50, execute 12, inference delay 16 ticks)
- Control rate: 30 Hz, action dim 6

| Strategy | Mean boundary jump | Max boundary jump | Mean step jump |
|---|---:|---:|---:|
| naive async | 1.1634 | 2.6125 | 0.2192 |
| RTC freeze | 0.1467 | 0.6490 | 0.1431 |

**Boundary-jump reduction: 87.4%** (lower boundary jump means smoother chunk transitions).

> Deterministic simulation of latency-aware action-chunk scheduling (Real-Time Chunking style). Models the freeze-prefix + soft-mask blend over pre-computed chunks; it does not run a diffusion/flow policy or the gradient-guided sampler. Continuity is a runtime scheduling property, not a policy-quality or task-success claim.
