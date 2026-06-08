#!/usr/bin/env bash
#
# Record the README hero GIF from the REAL vla_ros2 ROS2 runtime.
#
# It launches the runtime node + a synthetic-input node (no robot, no GPU),
# shows the live ROS2 node/topic graph and a real typed VLAAction message, then
# renders the terminal capture to docs/assets/runtime_demo.gif.
#
# Requirements: a sourced ROS2 (tested on Jazzy), `colcon`, `asciinema`, and
# `agg` (https://github.com/asciinema/agg). Run from the repo root.
#
# Usage:
#   ./scripts/record_runtime_demo.sh
#
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

: "${ROS_SETUP:=/opt/ros/jazzy/setup.bash}"
CAST="$(mktemp --suffix=.cast)"
OUT="docs/assets/runtime_demo.gif"

# Build the messages + ament_python package once.
# shellcheck disable=SC1090
source "$ROS_SETUP"
colcon build --packages-select vla_ros2_msgs vla_ros2

DEMO="$(mktemp --suffix=.sh)"
cat > "$DEMO" <<EOF
cd "$REPO_ROOT"
source "$ROS_SETUP" >/dev/null 2>&1
source install/setup.bash >/dev/null 2>&1
export PYTHONPATH="$REPO_ROOT/src:\$PYTHONPATH"
export RCUTILS_COLORIZED_OUTPUT=0

C='\033[1;36m'; G='\033[1;32m'; Y='\033[1;33m'; D='\033[0;90m'; N='\033[0m'
hr(){ printf "\${D}────────────────────────────────────────────────────────\${N}\n"; }
type_cmd(){ printf "\${G}\\\$ \${N}\${Y}%s\${N}\n" "\$1"; sleep 0.6; }

clear
printf "\${C}  vla_ros2 \${N}— ROS2-native on-robot VLA runtime\n"
printf "\${D}  camera + instruction + robot state  ->  VLA adapter  ->  typed VLAAction (ROS2 topic)\${N}\n"
hr; sleep 1

type_cmd "ros2 launch vla_ros2 smoke.launch.py   # dummy adapter, dry-run, synthetic input"
ros2 launch vla_ros2 smoke.launch.py >/tmp/vla_launch.log 2>&1 &
LP=\$!
sleep 6
printf "\${G}runtime up.\${N}\n\n"; sleep 0.8

type_cmd "ros2 node list | grep vla"
ros2 node list 2>/dev/null | grep vla | sort -u
echo; sleep 1

type_cmd "ros2 topic list | grep -E 'vla|diagnostics'"
ros2 topic list 2>/dev/null | grep -E 'vla|diagnostics'
echo; sleep 1.2

type_cmd "ros2 topic echo /vla/action --once   # a real typed action message"
timeout 5 ros2 topic echo /vla/action --once 2>/dev/null
sleep 1.2

type_cmd "ros2 topic hz /vla/action   # publishing at control_hz"
timeout 4 ros2 topic hz /vla/action 2>/dev/null | head -3
echo; sleep 0.8

hr
printf "\${G}  real ROS2 runtime — no robot, no GPU, no toy simulation\${N}\n"
sleep 1.5

kill \$LP >/dev/null 2>&1 || true
pkill -f vla_runtime_node >/dev/null 2>&1 || true
pkill -f vla_smoke_input >/dev/null 2>&1 || true
pkill -f vla_runtime_recorder >/dev/null 2>&1 || true
sleep 1
EOF

mkdir -p docs/assets
asciinema rec --overwrite --cols 92 --rows 34 -c "bash $DEMO" "$CAST" || true
agg --theme monokai --font-size 16 --speed 1.3 "$CAST" "$OUT"
rm -f "$DEMO" "$CAST"
echo "wrote $OUT"
