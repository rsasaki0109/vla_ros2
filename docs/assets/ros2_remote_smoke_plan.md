# ROS2 Remote Smoke Plan

This plan runs heavyweight VLA inference on a GPU server while the ROS2
runtime node stays on the robot-side process. It is dry-run by default and
records status, diagnostics, and typed action messages for reports.

## 1. GPU Server

```bash
vla-zoo serve --model openvla --host 0.0.0.0 --port 8001 --pretrained openvla/openvla-7b --device cuda:0 --dtype bfloat16 --unnorm-key bridge_orig
```

## 2. ROS2 Runtime Recording

Run for the requested duration with `timeout`; remove the prefix to stop manually.

```bash
timeout 30s ros2 launch vla_zoo remote_smoke_record.launch.py model_name:=openvla remote_url:=http://gpu-box:8001 output_dir:=results/ros2_remote_openvla 'instruction:=pick up the red block' task_id:=ros2_remote_smoke_pick_red_block dry_run:=true publish_actions_in_dry_run:=true
```

## 3. Reports

```bash
vla-zoo compare dashboard --status-log results/ros2_remote_openvla/vla_status.jsonl --diagnostics-log results/ros2_remote_openvla/vla_diagnostics.jsonl --out results/ros2_remote_openvla/dashboard.html --title 'vla_zoo ROS2 Remote Smoke: openvla'
vla-zoo ros action-trace --action-log results/ros2_remote_openvla/vla_actions.jsonl --out results/ros2_remote_openvla/action_trace.html --title 'vla_zoo ROS2 Remote Smoke: openvla Actions'
vla-zoo ros action-analyze --action-log results/ros2_remote_openvla/vla_actions.jsonl --out results/ros2_remote_openvla/action_analysis.json --markdown-out results/ros2_remote_openvla/action_analysis.md --title 'vla_zoo ROS2 Remote Smoke: openvla Action Analysis'
vla-zoo report bundle --status-log results/ros2_remote_openvla/vla_status.jsonl --diagnostics-log results/ros2_remote_openvla/vla_diagnostics.jsonl --out results/ros2_remote_openvla/report_bundle.zip --title 'vla_zoo ROS2 Remote Smoke: openvla'
```

## Settings

- model: `openvla`
- remote_url: `http://gpu-box:8001`
- output_dir: `results/ros2_remote_openvla`
- suggested_duration_sec: `30`
