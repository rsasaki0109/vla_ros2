#!/usr/bin/env bash
# Fine-tune lerobot/smolvla_base on lerobot/svla_so100_stacking (SO-100 stacking).
#
# Requires: .venv-smolvla (or any env with lerobot[smolvla] + CUDA).
#
# Demo (200 steps, small batch — wiring check, not task-quality training):
#   ./scripts/finetune_smolvla_so100.sh
#
# Longer run:
#   STEPS=20000 BATCH_SIZE=16 ./scripts/finetune_smolvla_so100.sh
#
# Use the checkpoint with vla_ros2 / Gazebo:
#   PRETRAINED="$(./scripts/finetune_smolvla_so100.sh --print-checkpoint)"
#   .venv-smolvla/bin/python scripts/record_smolvla_so100_demo.py --pretrained "$PRETRAINED"
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

DATASET="${DATASET:-lerobot/svla_so100_stacking}"
BASE_POLICY="${BASE_POLICY:-lerobot/smolvla_base}"
OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/smolvla_so100_stacking}"
JOB_NAME="${JOB_NAME:-smolvla_so100_stacking}"
STEPS="${STEPS:-200}"
BATCH_SIZE="${BATCH_SIZE:-8}"
SAVE_STEPS="${SAVE_STEPS:-100}"
NUM_WORKERS="${NUM_WORKERS:-4}"
DEVICE="${DEVICE:-cuda}"
# svla_so100_stacking uses top/wrist keys; smolvla_base expects camera1/2/3.
RENAME_MAP="${RENAME_MAP:-{\"observation.images.top\": \"observation.images.camera1\", \"observation.images.wrist\": \"observation.images.camera2\"}}"
EMPTY_CAMERAS="${EMPTY_CAMERAS:-1}"

PYTHON="${PYTHON:-}"
if [[ -z "${PYTHON}" ]]; then
  if [[ -x "${REPO_ROOT}/.venv-smolvla/bin/python" ]]; then
    PYTHON="${REPO_ROOT}/.venv-smolvla/bin/python"
  else
    PYTHON="python3"
  fi
fi

LEROBOT_TRAIN="${LEROBOT_TRAIN:-}"
if [[ -z "${LEROBOT_TRAIN}" ]]; then
  VENV_TRAIN="$(dirname "${PYTHON}")/lerobot-train"
  if [[ -x "${VENV_TRAIN}" ]]; then
    LEROBOT_TRAIN="${VENV_TRAIN}"
  else
    LEROBOT_TRAIN="lerobot-train"
  fi
fi

checkpoint_path() {
  local root="$1"
  if [[ -L "${root}/checkpoints/last" ]]; then
    readlink -f "${root}/checkpoints/last/pretrained_model"
    return 0
  fi
  local latest
  latest="$(find "${root}/checkpoints" -mindepth 2 -maxdepth 2 -type d -name pretrained_model 2>/dev/null | sort | tail -1)"
  if [[ -n "${latest}" ]]; then
    echo "${latest}"
    return 0
  fi
  return 1
}

if [[ "${1:-}" == "--print-checkpoint" ]]; then
  checkpoint_path "${OUTPUT_DIR}" || {
    echo "no checkpoint under ${OUTPUT_DIR}" >&2
    exit 1
  }
  exit 0
fi

echo "=== SmolVLA fine-tune: ${BASE_POLICY} on ${DATASET} ==="
echo "output_dir=${OUTPUT_DIR} steps=${STEPS} batch_size=${BATCH_SIZE} device=${DEVICE}"

if ! "${PYTHON}" - <<'PY'
import sys
try:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
except ImportError:
    sys.exit(1)
if not torch.cuda.is_available():
    sys.exit(2)
sys.exit(0)
PY
then
  echo "SmolVLA extras + CUDA required. Try: pip install -e '.[smolvla]'" >&2
  exit 1
fi

"${LEROBOT_TRAIN}" \
  --policy.path="${BASE_POLICY}" \
  --dataset.repo_id="${DATASET}" \
  --output_dir="${OUTPUT_DIR}" \
  --job_name="${JOB_NAME}" \
  --steps="${STEPS}" \
  --batch_size="${BATCH_SIZE}" \
  --save_checkpoint=true \
  --save_freq="${SAVE_STEPS}" \
  --num_workers="${NUM_WORKERS}" \
  --policy.device="${DEVICE}" \
  --policy.push_to_hub=false \
  --policy.empty_cameras="${EMPTY_CAMERAS}" \
  --rename_map="${RENAME_MAP}"

CKPT="$(checkpoint_path "${OUTPUT_DIR}" || true)"
if [[ -n "${CKPT}" ]]; then
  echo "checkpoint: ${CKPT}"
  echo "next: --pretrained ${CKPT}"
else
  echo "training finished; no checkpoint directory found under ${OUTPUT_DIR}" >&2
fi
