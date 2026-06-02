# pi0 Compatibility Probe

Date: 2026-06-03 (supersedes the 2026-06-02 note)

This probe time-boxed an attempt to load a public LeRobot pi0 checkpoint as a
local `vla_zoo` action probe — the same runtime-path evidence recorded for
SmolVLA and OpenVLA-7b — and records the precise boundary that remains.

## Environment

| Component | Version |
|---|---|
| GPU | NVIDIA GeForce GPU |
| Local package env | `.venv-smolvla` |
| LeRobot | `0.5.1` |
| draccus | `0.10.0` |
| torch | `2.10.0+cu128` |
| transformers | `5.3.0` |
| huggingface_hub | `1.17.0` |

## Checkpoint / version matrix

| Checkpoint | Config decode (LeRobot 0.5.1) | Weights | Local inference |
|---|---|---|---|
| `lerobot/pi0` | **Fails** — `draccus.DecodingError` | n/a | Blocked (schema) |
| `lerobot/pi0_base` | **OK** — `PI0Config`, 32D action, `n_action_steps=50`, `chunk_size=50` | 14.0 GB `model.safetensors`, bf16 fits ~8.9 GB / 16 GB | Blocked (gated tokenizer) |

### 1. `lerobot/pi0` — permanent config-schema mismatch

`PreTrainedConfig.from_pretrained("lerobot/pi0")` raises `draccus.utils.DecodingError`.
The cached checkpoint config carries fields that the installed `PI0Config`
(the newer OpenPI-port schema) does not accept:

```
The fields `resize_imgs_with_padding`, `adapt_to_pi_aloha`,
`use_delta_joint_actions_aloha`, `proj_width`, `num_steps`, `use_cache`,
`attention_implementation`, `train_state_proj` are not valid for PI0Config
```

This is the old config schema and is permanently rejected by LeRobot 0.5.1
unless a matching (older) LeRobot is pinned.

### 2. `lerobot/pi0_base` — version-matched, then license-gated

`lerobot/pi0_base` is the **version-matched** checkpoint: its config decodes
cleanly under LeRobot 0.5.1 (`PI0Config`, `max_action_dim=32`,
`n_action_steps=50`, `chunk_size=50`). `vla_zoo` now defaults local pi0 loading
to it.

The float32 config (`dtype: "float32"`) does not fit a 16 GB GPU (the ~3.3 B
PaliGemma + action-expert weights OOM during load). The adapter therefore grew a
`dtype` override; building the model with `dtype="bfloat16"` fits the card
(~8.9 GB constructed footprint, with headroom for activations).

The weights themselves (14.0 GB `model.safetensors`) are downloadable. The
remaining block is the **processor/tokenizer**: pi0's pre/post-processors require
the `google/paligemma-3b-pt-224` tokenizer, which is a **gated repository**:

```
GatedRepoError: 401 Client Error.
Cannot access gated repo for url
https://huggingface.co/google/paligemma-3b-pt-224/resolve/main/tokenizer_config.json.
Access to model google/paligemma-3b-pt-224 is restricted.
```

So local pi0 inference is blocked on a license acceptance, not a version,
config-schema, or GPU-memory issue.

## Reproduce

```bash
# config decode + bf16 construct (no gated assets needed)
HF_HUB_OFFLINE=1 PYTHONPATH=src .venv-smolvla/bin/python -c "
from lerobot.policies.pi0.modeling_pi0 import PI0Policy  # registers the choice
from lerobot.configs.policies import PreTrainedConfig
print(type(PreTrainedConfig.from_pretrained('lerobot/pi0_base')).__name__)  # PI0Config
"

# the full local probe (blocked at the gated tokenizer until the license is accepted)
PYTHONPATH=src .venv-smolvla/bin/python -m vla_zoo.cli.main demo action-probe \
  --model pi0 --runtime local --allow-local-heavy \
  --pretrained lerobot/pi0_base --device cuda --adapter-kwarg dtype=bfloat16 \
  --out docs/assets/sample_pybullet_pi0/pi0_action_probe.jsonl \
  --summary-md docs/assets/sample_pybullet_pi0/runtime_action_probe.md
```

## Runtime position

`vla_zoo` keeps `load_model("pi0")` lightweight and remote-first by default; local
loading is explicit (`load_model("pi0", enable_local=True,
pretrained="lerobot/pi0_base", dtype="bfloat16")`). To unblock the local
real-scene action probe — recorded the same way as the SmolVLA and OpenVLA
probes — accept the `google/paligemma-3b-pt-224` license on Hugging Face and run
with an authorized token (drop `HF_HUB_OFFLINE`). This note records the
compatibility boundary so no task-success or policy-quality claim is implied for
pi0; the matrix `local_runtime` cell stays `blocked` with this precise reason.
