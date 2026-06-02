# GR00T Path (Blocked Until the NVIDIA Isaac GR00T Stack)

GR00T is an **experimental, blocked** adapter in vla_zoo. The adapter declares a
runtime contract and registry metadata, but it ships **no inference** and makes
**no task-success claim**. It stays blocked until the external NVIDIA Isaac GR00T
stack is wired in behind a real serving adapter.

This is an honest status page, not integration guidance for a working model. The
[evidence matrix](assets/vla_model_evidence_matrix.html) keeps every GR00T runtime
cell at `blocked` or `partial` for exactly this reason.

## Why blocked

- The Isaac GR00T runtime is an external NVIDIA stack with its own dependencies,
  weights, and license; vla_zoo does not redistribute or vendor it. It is **not a
  pip-installable package** — `gr00t`, `isaac-gr00t`, and `nvidia-gr00t` all 404 on
  PyPI; the real runtime is the [NVIDIA Isaac-GR00T](https://github.com/NVIDIA/Isaac-GR00T)
  GitHub stack. This is recorded as a reproducible
  [block probe](assets/sample_task_verification/groot_block_probe.md).
- `vla_zoo.adapters.groot:GR00TAdapter` intentionally **does not fabricate
  actions**. `predict_observation` raises:
  - `MissingDependencyError` when the upstream `gr00t` package is absent, and
  - `NotImplementedError` even when it imports, because a real serving adapter is
    still required.
- The single source of truth for the message is `GROOT_BLOCKED_NOTE` in
  `src/vla_zoo/adapters/groot.py`; the registry verification text and the evidence
  matrix reuse the same wording.

## Expected observation/action contract (to be confirmed)

These are the expectations a real serving adapter must satisfy. Concrete shapes,
camera counts, and control rates are stack/embodiment specific and are **not**
pinned by this placeholder.

| Field | Expectation |
|---|---|
| Observation: vision | One or more RGB camera frames (humanoid head/hands typical). |
| Observation: instruction | Natural-language task/instruction string. |
| Observation: state | Humanoid proprioceptive state vector; layout is embodiment specific. |
| Action: interface | Humanoid/generalist action interface. |
| Action: shape | Checkpoint specific; `GROOT_ACTION_SPEC` is a placeholder to override. |
| Action: chunking | GR00T-class models typically emit chunks, so robot-side code should expect `VLAActionChunk`. |
| Control rate | Stack dependent. |

## What would unblock it

1. A version-matched NVIDIA Isaac GR00T stack installed in a **dedicated** serving
   environment (kept out of the base `vla_zoo` install and out of tests).
2. A real serving adapter that maps the contract above onto `/v1/predict`.
3. A recorded action probe from that server, after which the `remote_server` and
   `gpu_inference` cells can move off `blocked`.

Until all three exist, GR00T remains experimental and blocked by design.
