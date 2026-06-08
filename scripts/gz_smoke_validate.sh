#!/usr/bin/env bash
# Gazebo closed-loop validation from ros2/SIM.md (Phase 1 + optional Phase 2).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi
if [[ -f install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source install/setup.bash
  set -u
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
export ROS_DOMAIN_ID="${GZ_ROS_DOMAIN_ID:-77}"

PHASE="${1:-all}"
LOCK_FILE="${ROS_HOME:-$HOME/.ros}/locks/ros2-control-controller-spawner.lock"
LAUNCH_PID=""
PROBE_TIMEOUT="${GZ_PROBE_TIMEOUT_SEC:-60}"
STARTUP_TIMEOUT="${GZ_STARTUP_TIMEOUT_SEC:-90}"

pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

cleanup() {
  if [[ -n "${LAUNCH_PID}" ]]; then
    kill "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  pkill -f 'gz sim.launch.py|vla_action_bridge_node|vla_runtime_node|vla_smoke_input_node' 2>/dev/null || true
}
trap cleanup EXIT

prepare() {
  colcon build --base-paths ros2 --packages-select vla_ros2_msgs vla_ros2 vla_ros2_gz >/tmp/vla_gz_build.log 2>&1
  set +u
  # shellcheck disable=SC1091
  source install/setup.bash
  set -u
  rm -f "${LOCK_FILE}"
  pkill -9 -f 'gz sim.launch.py|ruby.*gz sim|vla_action_bridge_node|vla_runtime_node|vla_smoke_input_node|controller_manager/spawner' 2>/dev/null || true
  sleep 3
  export ROS_DOMAIN_ID="${GZ_ROS_DOMAIN_ID:-$((77 + RANDOM % 50))}"
  echo "Using ROS_DOMAIN_ID=${ROS_DOMAIN_ID}"
}

wait_for_topic() {
  local topic="$1"
  local timeout_sec="$2"
  local start
  start="$(date +%s)"
  while (( $(date +%s) - start < timeout_sec )); do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

wait_for_controllers() {
  local timeout_sec="$1"
  local start
  start="$(date +%s)"
  while (( $(date +%s) - start < timeout_sec )); do
    if ros2 control list_controllers 2>/dev/null | grep -q 'joint_trajectory_controller.*active'; then
      ros2 control list_controllers 2>/dev/null
      return 0
    fi
    sleep 1
  done
  return 1
}

launch_gz_smoke() {
  local model_name="$1"
  local enable_actuation="$2"
  local dry_run="$3"
  ros2 launch vla_ros2_gz gz_smoke.launch.py \
    gz_args:="-s" \
    model_name:="${model_name}" \
    enable_actuation:="${enable_actuation}" \
    dry_run:="${dry_run}" \
    publish_actions_in_dry_run:=true &
  LAUNCH_PID=$!
}

phase_1() {
  echo "=== Gazebo Phase 1: runtime graph (no actuation) ==="
  prepare
  launch_gz_smoke dummy false true
  wait_for_topic /vla/action "${STARTUP_TIMEOUT}" || fail "/vla/action did not appear"
  python3 "${REPO_ROOT}/scripts/gz_smoke_probe.py" --phase 1 --timeout-sec "${PROBE_TIMEOUT}"
  pass "Gazebo Phase 1 runtime graph"
  cleanup
  trap - EXIT
  LAUNCH_PID=""
}

phase_2() {
  echo "=== Gazebo Phase 2: bridge actuation (random adapter) ==="
  prepare
  launch_gz_smoke random true false
  wait_for_topic /vla/action "${STARTUP_TIMEOUT}" || fail "/vla/action did not appear"
  wait_for_controllers "${STARTUP_TIMEOUT}" || fail "joint_trajectory_controller did not become active"
  python3 "${REPO_ROOT}/scripts/gz_smoke_probe.py" --phase 2 --timeout-sec 45
  pass "Gazebo Phase 2 actuation"
  cleanup
  trap - EXIT
  LAUNCH_PID=""
}

case "$PHASE" in
  1) phase_1 ;;
  2) phase_2 ;;
  all)
    phase_1
    phase_2
    ;;
  *)
    echo "usage: $0 [1|2|all]" >&2
    exit 2
    ;;
esac
