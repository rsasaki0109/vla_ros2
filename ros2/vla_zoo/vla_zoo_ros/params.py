from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class RuntimeNodeParams:
    model_name: str
    runtime: str
    dry_run: bool
    image_topic: str
    instruction_topic: str
    joint_state_topic: str
    action_topic: str
    action_chunk_topic: str
    status_topic: str
    publish_action_chunk: bool
    control_hz: float
    max_queue_size: int
    device: str
    pretrained: str
    unnorm_key: str
    remote_url: str
