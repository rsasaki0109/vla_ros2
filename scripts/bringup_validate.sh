#!/usr/bin/env bash
# Phase A/B gate checks from ros2/BRINGUP.md (no actuation).
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

case "$PHASE" in
  a|A) phase_a ;;
  b|B) phase_b ;;
  all)
    phase_a
    phase_b
    ;;
  *)
    echo "usage: $0 [a|b|all]" >&2
    exit 2
    ;;
esac
