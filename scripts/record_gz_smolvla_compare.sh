#!/usr/bin/env bash
# Record SmolVLA base vs fine-tuned comparison GIFs (offline + Gazebo).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/vla_gz_env.sh"

vla_gz_prepare_env "${REPO_ROOT}"

OUTPUT_DIR="${OUTPUT_DIR:-checkpoints/smolvla_so100_stacking_20k}"
FINETUNED="$(OUTPUT_DIR="${OUTPUT_DIR}" ./scripts/finetune_smolvla_so100.sh --print-checkpoint 2>/dev/null || true)"
FINETUNED="${FINETUNED:-lerobot/smolvla_base}"
BASE="${BASE:-lerobot/smolvla_base}"
DURATION="${DURATION:-60}"
WARMUP="${WARMUP:-180}"
STEPS="${STEPS:-60}"
EPISODE="${EPISODE:-0}"
PYTHON="${PYTHON:-${REPO_ROOT}/.venv-smolvla/bin/python}"

echo "=== SmolVLA compare (offline + Gazebo) ==="
echo "base=${BASE}"
echo "finetuned=${FINETUNED}"
echo ""

echo "[offline 1/2] base..."
"${PYTHON}" scripts/record_gz_smolvla_demo.py --offline \
  --steps "${STEPS}" --episode "${EPISODE}" --pretrained "${BASE}" \
  --out docs/assets/gz_smolvla_offline_base.gif \
  --metrics-out docs/assets/gz_smolvla_offline_base_metrics.json

echo "[offline 2/2] fine-tuned..."
"${PYTHON}" scripts/record_gz_smolvla_demo.py --offline \
  --steps "${STEPS}" --episode "${EPISODE}" --pretrained "${FINETUNED}" \
  --out docs/assets/gz_smolvla_offline_finetuned.gif \
  --metrics-out docs/assets/gz_smolvla_offline_finetuned_metrics.json

vla_gz_kill_stacks
sleep 2

echo "[gazebo 1/2] base..."
./scripts/record_gz_smolvla_demo.sh \
  --pretrained "${BASE}" \
  --out docs/assets/gz_smolvla_demo_base.gif \
  --duration-sec "${DURATION}" --warmup-sec "${WARMUP}" --episode "${EPISODE}" \
  --metrics-out docs/assets/gz_smolvla_demo_base_metrics.json

vla_gz_kill_stacks
sleep 3

echo "[gazebo 2/2] fine-tuned..."
./scripts/record_gz_smolvla_demo.sh \
  --pretrained "${FINETUNED}" \
  --out docs/assets/gz_smolvla_demo_finetuned.gif \
  --duration-sec "${DURATION}" --warmup-sec "${WARMUP}" --episode "${EPISODE}" \
  --metrics-out docs/assets/gz_smolvla_demo_finetuned_metrics.json

"${PYTHON}" - <<'PY'
import json
from pathlib import Path

root = Path("docs/assets")
offline = {
    "base": json.loads((root / "gz_smolvla_offline_base_metrics.json").read_text()),
    "finetuned": json.loads((root / "gz_smolvla_offline_finetuned_metrics.json").read_text()),
}
gazebo = {
    "base": json.loads((root / "gz_smolvla_demo_base_metrics.json").read_text()),
    "finetuned": json.loads((root / "gz_smolvla_demo_finetuned_metrics.json").read_text()),
}
summary = {
    "instruction": offline["base"]["instruction"],
    "episode": offline["base"]["episode"],
    "note": (
        "Offline loop applies policy actions in the SO-100 kinematic stand-in. "
        "Gazebo metrics are from live joint_states; not a formal success benchmark."
    ),
    "offline": offline,
    "gazebo": gazebo,
}
out = root / "gz_smolvla_compare_metrics.json"
out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
print(f"wrote {out}")
PY

echo "Done. See docs/assets/gz_smolvla_compare_metrics.json"
