# Security Policy

## Supported Versions

`vla_ros2` is pre-1.0. Security fixes target the latest `main` branch until release branches exist.

## Reporting Vulnerabilities

Please report security issues through GitHub private vulnerability reporting when available, or contact the repository owner privately.

Include:

- affected version or commit
- reproduction steps
- impact
- whether the issue affects local runtime, remote server, ROS2 node, or generated artifacts

## Robotics Safety Boundary

`vla_ros2` publishes action messages by default. It does not directly command hardware in the core package.

Security or safety-sensitive reports include:

- remote inference server request handling
- unsafe default launch behavior
- action messages that bypass dry-run assumptions
- dependency confusion or unintended heavy dependency execution
- examples that could encourage direct actuation without watchdogs

Hardware-specific bridges should live outside the core runtime or require explicit opt-in with safety documentation.
