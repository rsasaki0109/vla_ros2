"""Demonstration utilities for vla_zoo."""

from vla_zoo.demo.action_playground import (
    ActionPlaygroundRecord,
    build_action_playground_records,
    format_action_playground_html,
)
from vla_zoo.demo.gif_suite import (
    PyBulletGifResult,
    PyBulletGifSpec,
    build_pybullet_gif_specs,
    format_gif_gallery_markdown,
    render_pybullet_gif_suite,
)
from vla_zoo.demo.pybullet import PyBulletDemoConfig, render_pybullet_demo

__all__ = [
    "ActionPlaygroundRecord",
    "PyBulletDemoConfig",
    "PyBulletGifResult",
    "PyBulletGifSpec",
    "build_action_playground_records",
    "build_pybullet_gif_specs",
    "format_action_playground_html",
    "format_gif_gallery_markdown",
    "render_pybullet_demo",
    "render_pybullet_gif_suite",
]
