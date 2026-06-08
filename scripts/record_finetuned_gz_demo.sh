#!/usr/bin/env bash
# Record docs/assets/gz_smolvla_demo.gif using the latest fine-tuned checkpoint.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/smolvla_so100_stacking_20k}"
CKPT="$(OUTPUT_DIR="${OUTPUT_DIR}" ./scripts/finetune_smolvla_so100.sh --print-checkpoint)"

exec ./scripts/record_gz_smolvla_demo.sh --pretrained "${CKPT}" "$@"
