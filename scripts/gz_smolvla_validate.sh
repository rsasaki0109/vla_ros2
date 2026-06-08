#!/usr/bin/env bash
# SmolVLA × Gazebo closed-loop gate (GPU required for inference).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [[ -f /opt/ros/jazzy/setup.bash ]]; then
  set +u
  # shellcheck disable=SC1091
  source /opt/ros/jazzy/setup.bash
  set -u
fi

LOCK_FILE="${ROS_HOME:-$HOME/.ros}/locks/ros2-control-controller-spawner.lock"
LAUNCH_PID=""
PHASE="${1:-infer}"

pass() { printf 'PASS: %s\n' "$*"; }
fail() { printf 'FAIL: %s\n' "$*" >&2; exit 1; }

cleanup() {
  if [[ -n "${LAUNCH_PID}" ]]; then
    kill "${LAUNCH_PID}" 2>/dev/null || true
    wait "${LAUNCH_PID}" 2>/dev/null || true
  fi
  pkill -f 'gz sim.launch.py|vla_smolvla|vla_runtime_node' 2>/dev/null || true
}
trap cleanup EXIT

prepare() {
  if ! python3 - <<'PY'
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
    echo "Installing SmolVLA extras into the active Python environment..."
    pip install -e ".[smolvla]" >/tmp/gz_smolvla_pip.log 2>&1 || fail "pip install -e '.[smolvla]' failed"
    python3 - <<'PY' || fail "SmolVLA extras + CUDA required after pip install"
import sys
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # noqa: F401
if not torch.cuda.is_available():
    sys.exit(2)
sys.exit(0)
PY
  fi
  colcon build --base-paths ros2 --packages-select vla_ros2_msgs vla_ros2 vla_ros2_gz \
    --allow-overriding vla_ros2 vla_ros2_gz >/tmp/gz_smolvla_build.log 2>&1
  set +u
  # shellcheck disable=SC1091
  source install/setup.bash
  set -u
  export PYTHONPATH="${REPO_ROOT}/src:${PYTHONPATH:-}"
  export ROS_DOMAIN_ID="${GZ_SMOLVLA_ROS_DOMAIN_ID:-92}"
  rm -f "${LOCK_FILE}"
  pkill -9 -f 'gz sim.launch.py|vla_smolvla|vla_runtime_node' 2>/dev/null || true
  sleep 2
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

phase_infer() {
  echo "=== SmolVLA × Gazebo: inference graph (no actuation) ==="
  prepare
  ros2 launch vla_ros2_gz gz_smolvla.launch.py \
    dry_run:=false \
    publish_actions_in_dry_run:=true \
    enable_actuation:=false \
    control_hz:=2.0 &
  LAUNCH_PID=$!

  wait_for_topic /vla/action 120 || fail "/vla/action did not appear"
  python3 - <<'PY' || fail "SmolVLA runtime did not publish ready status + custom action"
import sys
import time

import rclpy
from rclpy.node import Node
from vla_ros2_msgs.msg import VLAAction, VLAStatus
from vla_ros2_ros.qos import action_qos, status_qos

class Probe(Node):
    def __init__(self):
        super().__init__("gz_smolvla_probe")
        self.actions = []
        self.ready = False
        self.create_subscription(VLAAction, "/vla/action", self._action_cb, action_qos(10))
        self.create_subscription(VLAStatus, "/vla/status", self._status_cb, status_qos(10))

    def _action_cb(self, msg: VLAAction) -> None:
        self.actions.append(msg)

    def _status_cb(self, msg: VLAStatus) -> None:
        if msg.ready:
            self.ready = True

rclpy.init()
node = Probe()
end = time.time() + 300.0
while time.time() < end and not (node.actions and node.ready):
    rclpy.spin_once(node, timeout_sec=0.2)
ok = bool(node.actions and node.ready and node.actions[0].model_name == "smolvla")
if ok:
    action = node.actions[0]
    ok = len(action.data) >= 6 and action.action_space in {"custom", ""}
    print(
        f"model={action.model_name} space={action.action_space} "
        f"dim={len(action.data)} adapter={action.adapter_name}"
    )
node.destroy_node()
if rclpy.ok():
    rclpy.shutdown()
sys.exit(0 if ok else 1)
PY
  pass "SmolVLA × Gazebo inference graph"
}

case "$PHASE" in
  infer|i) phase_infer ;;
  *)
    echo "usage: $0 [infer]" >&2
    exit 2
    ;;
esac
