#!/usr/bin/env bash
# Phase A/B/C gate checks from ros2/BRINGUP.md (C = parse-only bridge, no actuation).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f install/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source install/setup.bash
  set -u
fi
export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"

PHASE="${1:-all}"
PARAMS_FILE="${BRINGUP_PARAMS_FILE:-${REPO_ROOT}/ros2/vla_ros2/config/bringup.dashcam.example.yaml}"
INSTRUCTION="${BRINGUP_INSTRUCTION:-pick up the cup}"
READY_TIMEOUT_SEC="${BRINGUP_READY_TIMEOUT_SEC:-45}"
INSTR_PID=""
SMOKE_PID=""
BRIDGE_PID=""

pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

phase_a() {
  echo "=== Phase A: launch smoke test ==="
  colcon test --packages-select vla_ros2 --python-testing pytest \
    --event-handlers console_direct+ >/tmp/vla_bringup_phase_a.log 2>&1
  colcon test-result --verbose
  pass "Phase A smoke launch test"
}

wait_for_topic() {
  local topic="$1"
  local timeout_sec="${2:-30}"
  local start
  start="$(date +%s)"
  while true; do
    if ros2 topic list 2>/dev/null | grep -qx "$topic"; then
      return 0
    fi
    if (( $(date +%s) - start >= timeout_sec )); then
      return 1
    fi
    sleep 0.5
  done
}

wait_for_ready_status() {
  local timeout_sec="$1"
  local start ready status_text status_json
  start="$(date +%s)"
  while (( $(date +%s) - start < timeout_sec )); do
    status_json="$(ros2 topic echo /vla/status --once 2>/dev/null || true)"
    if [[ -z "$status_json" ]]; then
      sleep 0.5
      continue
    fi
    ready="$(printf '%s\n' "$status_json" | awk '/^ready:/{print $2; exit}')"
    status_text="$(printf '%s\n' "$status_json" | sed -n 's/^status_text: //p' | head -1 | sed "s/'//g")"
    if [[ "$ready" == "true" ]]; then
      printf '%s\n' "$status_json"
      pass "Phase B dry-run ready: ${status_text}"
      return 0
    fi
    sleep 0.5
  done
  fail "timed out waiting for ready=true on /vla/status"
}

phase_b() {
  echo "=== Phase B: dry-run on live I/O ==="
  [[ -f "$PARAMS_FILE" ]] || fail "params file not found: $PARAMS_FILE"

  ros2 launch vla_ros2 dummy.launch.py \
    params_file:="${PARAMS_FILE}" \
    instruction_msg_type:=string \
    dry_run:=true \
    publish_actions_in_dry_run:=false &
  local launch_pid=$!

  cleanup() {
    if [[ -n "${INSTR_PID}" ]]; then
      kill "${INSTR_PID}" 2>/dev/null || true
    fi
    kill "$launch_pid" 2>/dev/null || true
    wait "$launch_pid" 2>/dev/null || true
  }
  trap cleanup EXIT

  wait_for_topic /vla/status 45 || fail "/vla/status did not appear"

  python3 "${REPO_ROOT}/scripts/publish_instruction.py" \
    --text "${INSTRUCTION}" --repeat-hz 2.0 &
  INSTR_PID=$!

  wait_for_ready_status "$READY_TIMEOUT_SEC"
}

phase_c() {
  echo "=== Phase C: controller bridge parse (smoke graph, no actuation) ==="
  if [[ -f /opt/ros/jazzy/setup.bash ]]; then
    set +u
    # shellcheck disable=SC1091
    source /opt/ros/jazzy/setup.bash
    set -u
  fi
  colcon build --base-paths ros2 --packages-select vla_ros2_msgs vla_ros2 \
    --allow-overriding vla_ros2 >/tmp/vla_bringup_phase_c_build.log 2>&1
  set +u
  # shellcheck disable=SC1091
  source install/setup.bash
  set -u
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  export ROS_DOMAIN_ID="${BRINGUP_ROS_DOMAIN_ID:-88}"

  ros2 launch vla_ros2 smoke.launch.py \
    dry_run:=true publish_actions_in_dry_run:=true &
  SMOKE_PID=$!
  ros2 launch vla_ros2 controller_bridge.launch.py \
    enable_actuation:=false publish_cmd_vel:=false &
  BRIDGE_PID=$!

  cleanup() {
    if [[ -n "${SMOKE_PID}" ]]; then
      kill "${SMOKE_PID}" 2>/dev/null || true
    fi
    if [[ -n "${BRIDGE_PID}" ]]; then
      kill "${BRIDGE_PID}" 2>/dev/null || true
    fi
    wait "${SMOKE_PID}" "${BRIDGE_PID}" 2>/dev/null || true
  }
  trap cleanup EXIT

  wait_for_topic /vla/bridge/parsed 45 || fail "/vla/bridge/parsed did not appear"
  python3 - <<'PY' || fail "bridge did not publish parsed action"
import sys
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class Probe(Node):
    def __init__(self):
        super().__init__("phase_c_probe")
        self.payload = ""
        self.create_subscription(String, "/vla/bridge/parsed", self.cb, 10)

    def cb(self, msg: String) -> None:
        self.payload = msg.data

rclpy.init()
node = Probe()
end = time.time() + 20.0
while time.time() < end and not node.payload:
    rclpy.spin_once(node, timeout_sec=0.2)
ok = "action_space" in node.payload
node.destroy_node()
if rclpy.ok():
    rclpy.shutdown()
sys.exit(0 if ok else 1)
PY
  pass "Phase C bridge parsed VLAAction"
}

case "$PHASE" in
  a|A) phase_a ;;
  b|B) phase_b ;;
  c|C) phase_c ;;
  all)
    phase_a
    phase_b
    ;;
  *)
    echo "usage: $0 [a|b|c|all]" >&2
    exit 2
    ;;
esac
