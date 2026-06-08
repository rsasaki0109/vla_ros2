#!/usr/bin/env bash
# SmolVLA × Gazebo closed loop + browser playground (no real robot).
#
# Terminal 1 (this script): starts Gazebo + runtime + bridge; keeps running.
# Terminal 2: python scripts/vla_playground.py --ros
#
# Requires GPU + SmolVLA extras + ROS2 Jazzy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/vla_gz_env.sh"

vla_gz_prepare_env "${REPO_ROOT}"

PRETRAINED="${PRETRAINED:-}"
if [[ -z "${PRETRAINED}" ]]; then
  PRETRAINED="$(OUTPUT_DIR=checkpoints/smolvla_so100_stacking_20k ./scripts/finetune_smolvla_so100.sh --print-checkpoint 2>/dev/null || true)"
  PRETRAINED="${PRETRAINED:-lerobot/smolvla_base}"
fi

echo "=== Playground × Gazebo ==="
echo "pretrained=${PRETRAINED}"
echo "In another terminal: .venv-smolvla/bin/python scripts/vla_playground.py --ros"
echo ""

exec ros2 launch vla_ros2_gz gz_smolvla.launch.py \
  dry_run:=false \
  publish_actions_in_dry_run:=true \
  enable_actuation:=true \
  publish_instruction:=false \
  control_hz:=2.0 \
  pretrained:="${PRETRAINED}"
