#!/usr/bin/env bash
# Bootstrap a colcon workspace for vla_ros2 ROS2 packages from a source checkout.
#
# Usage:
#   ./scripts/bootstrap_ros2_workspace.sh              # core profile
#   VLA_ROS2_PROFILE=gz ./scripts/bootstrap_ros2_workspace.sh
#
# Do not use `set -u` — sourcing ROS setup files references unset variables.

set -eo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROS_DISTRO="${ROS_DISTRO:-jazzy}"
ROS_SETUP="/opt/ros/${ROS_DISTRO}/setup.bash"
PROFILE="${VLA_ROS2_PROFILE:-core}"
VENV_DIR="${VLA_ROS2_VENV:-${REPO_ROOT}/.venv}"
SKIP_ROSDEP="${VLA_ROS2_SKIP_ROSDEP:-0}"
SKIP_PIP="${VLA_ROS2_SKIP_PIP:-0}"
SKIP_BUILD="${VLA_ROS2_SKIP_BUILD:-0}"

log() {
  printf '[bootstrap] %s\n' "$*"
}

die() {
  printf '[bootstrap] ERROR: %s\n' "$*" >&2
  exit 1
}

case "${PROFILE}" in
  core)
    COLCON_EXTRA=(--packages-up-to vla_ros2)
    ;;
  gz)
    COLCON_EXTRA=(--packages-up-to vla_ros2_gz)
    ;;
  all)
    COLCON_EXTRA=()
    ;;
  *)
    die "unknown VLA_ROS2_PROFILE=${PROFILE} (use core, gz, or all)"
    ;;
esac

[[ -f "${ROS_SETUP}" ]] || die "missing ${ROS_SETUP}; install ROS 2 ${ROS_DISTRO} first"
# shellcheck disable=SC1090
source "${ROS_SETUP}"

if [[ "${SKIP_ROSDEP}" != "1" ]]; then
  if command -v rosdep >/dev/null 2>&1; then
    log "rosdep update (best effort)"
    rosdep update >/dev/null 2>&1 || true
    log "rosdep install for ros2/ (skip ament_python)"
    rosdep install --from-paths "${REPO_ROOT}/ros2" --ignore-src -y \
      --skip-keys "ament_python" || die "rosdep install failed"
  else
    log "rosdep not found; install ros-jazzy-ros-base or python3-rosdep"
  fi
fi

if [[ "${SKIP_PIP}" != "1" ]]; then
  if [[ ! -d "${VENV_DIR}" ]]; then
    log "creating venv at ${VENV_DIR}"
    python3 -m venv "${VENV_DIR}"
  fi
  # shellcheck disable=SC1091
  source "${VENV_DIR}/bin/activate"
  log "pip install -e .[dev]"
  pip install -U pip >/dev/null
  pip install -e "${REPO_ROOT}/.[dev]"
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  cd "${REPO_ROOT}"
  log "colcon build profile=${PROFILE}"
  colcon build --base-paths "${REPO_ROOT}/ros2" "${COLCON_EXTRA[@]}"
fi

cat <<EOF

Bootstrap complete (profile=${PROFILE}).

  source ${REPO_ROOT}/install/setup.bash
  export PYTHONPATH="${REPO_ROOT}/src:\${PYTHONPATH}"

Smoke test:
  ros2 launch vla_ros2 smoke.launch.py

Docs:
  ${REPO_ROOT}/ros2/WORKSPACE.md
  ${REPO_ROOT}/ros2/BRINGUP.md
  ${REPO_ROOT}/ros2/SIM.md

EOF
