from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeNodeParams:
    model_name: str
    runtime: str
    dry_run: bool
    instruction_msg_type: str
    image_topic: str
    instruction_topic: str
    joint_state_topic: str
    action_topic: str
    action_chunk_topic: str
    status_topic: str
    diagnostics_topic: str
    publish_action_chunk: bool
    publish_diagnostics: bool
    publish_actions_in_dry_run: bool
    control_hz: float
    max_queue_size: int
    require_image: bool
    stale_image_timeout_sec: float
    stale_instruction_timeout_sec: float
    clip_actions: bool
    action_low: tuple[float, ...]
    action_high: tuple[float, ...]
    device: str
    pretrained: str
    unnorm_key: str
