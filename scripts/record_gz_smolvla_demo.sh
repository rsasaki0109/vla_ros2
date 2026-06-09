#!/usr/bin/env bash
# Record docs/assets/gz_smolvla_demo.gif from the live Gazebo closed loop.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/vla_gz_env.sh"

vla_gz_prepare_env "${REPO_ROOT}"
vla_gz_kill_stacks
sleep 2

PYTHON="${PYTHON:-}"
if venv_py="$(vla_gz_venv_python "${REPO_ROOT}")"; then
  PYTHON="${venv_py}"
else
  PYTHON="python3"
fi

"${PYTHON}" scripts/record_gz_smolvla_demo.py "$@"
