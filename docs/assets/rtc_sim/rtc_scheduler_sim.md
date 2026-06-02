# Real-Time Chunking scheduler simulation

- Source: `synthetic(seed=7, mode_strength=0.6)`
- Chunks: 14 (horizon 16, execute 8, inference delay 4 ticks)
- Control rate: 50 Hz, action dim 3

| Strategy | Mean boundary jump | Max boundary jump | Mean step jump |
|---|---:|---:|---:|
| naive async | 1.3300 | 2.1479 | 0.4308 |
| RTC freeze | 0.3150 | 0.4296 | 0.3583 |

**Boundary-jump reduction: 76.3%** (lower boundary jump means smoother chunk transitions).

> Deterministic simulation of latency-aware action-chunk scheduling (Real-Time Chunking style). Models the freeze-prefix + soft-mask blend over pre-computed chunks; it does not run a diffusion/flow policy or the gradient-guided sampler. Continuity is a runtime scheduling property, not a policy-quality or task-success claim.
