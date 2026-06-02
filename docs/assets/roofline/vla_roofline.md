# VLA roofline floor vs recorded latency

- Hardware: **GPU** (672 GB/s, 88 TFLOPS fp16 nominal)
- Floor = single-forward, batch-1 memory-bound lower bound `weight_bytes / bandwidth`.

| Model | Weights | Floor (ms) | Bound by | Measured p50 (ms) | Headroom | Real-time band |
|---|---:|---:|:--:|---:|---:|:--|
| smolvla | 0.90 GB | 1.34 | memory | 381.9 | 285× | usable (sub-second) |
| openvla | 3.50 GB | 5.21 | memory | 1996.8 | 383× | slow (>1 s) |
| pi0 | 6.60 GB | 9.82 | memory | — | — | unknown |

> First-order roofline floor (VLA-Perf style): the single-forward memory-bound hardware lower bound weight_bytes/bandwidth at batch 1. NOT an achievable latency -- recorded p50 also pays for multi-step decode + framework overhead, so measured/floor is optimization headroom. Hardware peaks are nominal vendor specs. No policy-quality claim.
