#!/usr/bin/env bash
# Shared helpers for SmolVLA × Gazebo scripts (venv-aware colcon build).
set -euo pipefail

vla_gz_kill_stacks() {
  pkill -9 -f 'gz sim.launch.py|vla_runtime_node|vla_smolvla_input_node|vla_smolvla_joint_bridge|gz_smolvla_recorder' 2>/dev/null || true
}

vla_gz_repo_root() {
  local here
  here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
  printf '%s' "${here}"
}

vla_gz_venv_python() {
  local root="$1"
  if [[ -x "${root}/.venv-smolvla/bin/python" ]]; then
    printf '%s' "${root}/.venv-smolvla/bin/python"
    return 0
  fi
  return 1
}

vla_gz_colcon_build() {
  local root="$1"
  shift
  local -a cmake_args=()
  local venv_py=""
  if venv_py="$(vla_gz_venv_python "${root}")"; then
    cmake_args=(--cmake-args "-DPython3_EXECUTABLE=${venv_py}")
  fi
  colcon build --base-paths "${root}/ros2" \
    --packages-select vla_ros2_msgs vla_ros2 vla_ros2_gz \
    --allow-overriding vla_ros2 vla_ros2_gz \
    "${cmake_args[@]}" "$@"
}

vla_gz_patch_runtime_shebang() {
  local root="$1"
  local venv_py=""
  venv_py="$(vla_gz_venv_python "${root}")" || return 0
  local runtime="${root}/install/vla_ros2/lib/vla_ros2/vla_runtime_node"
  if [[ -f "${runtime}" ]]; then
    sed -i "1c#!${venv_py}" "${runtime}"
  fi
}

vla_gz_prepare_env() {
  local root="$1"
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
  fi
  vla_gz_colcon_build "${root}" >/tmp/vla_gz_colcon_build.log 2>&1
  vla_gz_patch_runtime_shebang "${root}"
  set +u
  # shellcheck disable=SC1091
  source "${root}/install/setup.bash"
  set -u
  export PYTHONPATH="${root}/src:${PYTHONPATH:-}"
  export ROS_DOMAIN_ID="${GZ_SMOLVLA_ROS_DOMAIN_ID:-92}"
  rm -f "${ROS_HOME:-$HOME/.ros}/locks/ros2-control-controller-spawner.lock"
  local venv_py=""
  if venv_py="$(vla_gz_venv_python "${root}")"; then
    export PATH="$(dirname "${venv_py}"):${PATH}"
  fi
}
