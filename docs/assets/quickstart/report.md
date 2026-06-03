# vla_zoo quickstart

- Schema: `vla-zoo-quickstart/v1`
- Episodes per adapter: 5
- Status: ✅ runtime boundary works

| Model | Action space | Dim | Latency p50 (ms) | Latency mean (ms) | Rate (Hz) | Sample action |
|---|---|---|---|---|---|---|
| dummy | eef_delta | 7 | 0.01 | 0.01 | 74285.54 | [0.000, 0.000, 0.000, 0.000, 0.000, 0.000, …] |
| scripted | eef_delta | 7 | 0.05 | 0.07 | 15179.16 | [0.200, -0.100, 0.200, 0.000, 0.000, 0.000, …] |
| random | eef_delta | 7 | 0.01 | 0.03 | 38551.76 | [-0.095, -0.007, 0.195, 0.217, -0.071, 0.036, …] |

## Next steps

- [VLA runtime leaderboard](https://rsasaki0109.github.io/vla_zoo/assets/leaderboard/vla_runtime_leaderboard.html)
- [VLA model evidence matrix](https://rsasaki0109.github.io/vla_zoo/assets/vla_model_evidence_matrix.html)
- [PyBullet GIF gallery](https://rsasaki0109.github.io/vla_zoo/assets/gif_suite/)
- [Full docs & demo site](https://rsasaki0109.github.io/vla_zoo/)

Runtime-boundary smoke check on pure-Python baselines (dummy/scripted/random) — no GPU, weights, or PyBullet. It proves load_model() -> predict() -> typed action works locally and measures latency. Baselines are infrastructure baselines, NOT VLA policies; this is not a model-quality or task-success claim. See the linked evidence for real-adapter runtime paths.

