# vla_zoo Multi-Task Verification Samples

These artifacts record runtime verification, not robot skill claims.

## What Was Run

```bash
vla-zoo compare tasks \
  --models dummy,scripted,random \
  --tasks all \
  --model-call-every 24 \
  --render-stride 80 \
  --out results/vla_task_verification/baseline_tasks.json \
  --markdown-out results/vla_task_verification/baseline_tasks.md \
  --html-out results/vla_task_verification/baseline_tasks.html

vla-zoo compare tasks \
  --models openvla,pi0,smolvla,groot \
  --tasks all \
  --model-call-every 100 \
  --render-stride 120 \
  --out results/vla_task_verification/external_adapter_status.json \
  --markdown-out results/vla_task_verification/external_adapter_status.md \
  --html-out results/vla_task_verification/external_adapter_status.html
```

## Artifacts

- `baseline_tasks.json`: multi-task runtime results for `dummy`, `scripted`, and `random`
- `baseline_tasks.md`: Markdown report for the baseline task run
- `baseline_tasks.html`: self-contained HTML report for the baseline task run
- `external_adapter_status.json`: status run for OpenVLA/openpi/SmolVLA/GR00T adapters
- `external_adapter_status.md`: Markdown status report for external adapters
- `external_adapter_status.html`: self-contained HTML status report for external adapters
- `openvla_prompt_probe.md`: sanitized OpenVLA local CUDA prompt probe result

## Interpretation

The baseline adapters completed three PyBullet runtime tasks:

- `pick_red_block`
- `move_red_block_left`
- `move_red_block_right`

OpenVLA, openpi, SmolVLA, and GR00T are not represented here as completed
real-policy multi-task results. OpenVLA had local weights and dependencies
available, but the local CUDA run did not complete due to insufficient free GPU
memory during this run. openpi, SmolVLA, and GR00T remain placeholder/remote
adapter targets in this repository.
