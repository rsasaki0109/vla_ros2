"""Pytest path setup for ROS2 python modules living under ros2/."""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_PKG = REPO_ROOT / "src"
if SRC_PKG.is_dir():
    sys.path.insert(0, str(SRC_PKG))
ROS_VLA_PKG = REPO_ROOT / "ros2" / "vla_ros2"
if ROS_VLA_PKG.is_dir():
    sys.path.insert(0, str(ROS_VLA_PKG))
