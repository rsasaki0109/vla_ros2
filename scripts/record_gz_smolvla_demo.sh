#!/usr/bin/env bash
# Record docs/assets/gz_smolvla_demo.gif from the live Gazebo closed loop.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi

PYTHON="${PYTHON:-}"
if [[ -x "${REPO_ROOT}/.venv-smolvla/bin/python" ]]; then
  PYTHON="${REPO_ROOT}/.venv-smolvla/bin/python"
else
  PYTHON="python3"
fi

colcon build --base-paths ros2 --packages-select vla_ros2_msgs vla_ros2 vla_ros2_gz \
  --allow-overriding vla_ros2 vla_ros2_gz >/tmp/record_gz_smolvla_build.log 2>&1
set +u
# shellcheck disable=SC1091
source install/setup.bash
set -u

export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${GZ_SMOLVLA_ROS_DOMAIN_ID:-92}"
rm -f "${ROS_HOME:-$HOME/.ros}/locks/ros2-control-controller-spawner.lock"
pkill -9 -f 'gz sim.launch.py|vla_smolvla|vla_runtime_node' 2>/dev/null || true
sleep 2

"${PYTHON}" scripts/record_gz_smolvla_demo.py "$@"
