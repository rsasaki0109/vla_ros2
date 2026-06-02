from __future__ import annotations

from typing import Any

from vla_zoo.adapters.smolvla import SmolVLAAdapter
from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.model import VLAAdapter
from vla_zoo.core.types import ActionSpec, VLAAction, VLAActionChunk, VLAObservation

DEFAULT_PI0_PRETRAINED = "lerobot/pi0_base"

PI0_ACTION_SPEC = ActionSpec(
    action_space="custom",
    shape=(32,),
    description=(
        "LeRobot/OpenPI pi0 continuous action. Shape and semantics are checkpoint-specific; "
        "the version-matched lerobot/pi0_base exposes a 32D action, while the older "
        "lerobot/pi0 config schema is rejected by LeRobot 0.5.1. The runtime action spec is "
        "derived from the loaded checkpoint config."
    ),
)


def _import_pi0_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi0.modeling_pi0 import PI0Policy
    except ImportError as exc:
        msg = (
            "pi0 adapter requires LeRobot pi0 dependencies. "
            'Install them with: pip install "vla_zoo[openpi]"'
        )
        raise MissingDependencyError(msg) from exc
    return torch, PI0Policy, make_pre_post_processors


class Pi0Adapter(SmolVLAAdapter):
    """Remote-first pi0 adapter with optional local LeRobot loading."""

    name = "pi0"
    model_id = DEFAULT_PI0_PRETRAINED
    action_spec = PI0_ACTION_SPEC
    default_pretrained = DEFAULT_PI0_PRETRAINED
    adapter_label = "LeRobot/OpenPI pi0"

    def __init__(
        self,
        *,
        enable_local: bool = False,
        pretrained: str | None = None,
        strict: bool = False,
        **kwargs: Any,
    ) -> None:
        self._local_enabled = bool(enable_local or pretrained)
        if not self._local_enabled:
            VLAAdapter.__init__(
                self,
                name=self.name,
                model_id=self.model_id,
                action_spec=self.action_spec,
                metadata={
                    "remote_first": True,
                    "enable_local": False,
                    **kwargs,
                },
            )
            return

        super().__init__(
            pretrained=pretrained or self.default_pretrained,
            strict=strict,
            action_spec=kwargs.pop("action_spec", None),
            **kwargs,
        )
        self._local_enabled = True

    @classmethod
    def from_config(cls, **kwargs: Any) -> Pi0Adapter:
        return cls(**kwargs)

    def _import_policy_dependencies(self) -> tuple[Any, Any, Any]:
        return _import_pi0_dependencies()

    def predict_observation(self, observation: VLAObservation) -> VLAAction | VLAActionChunk:
        if not self._local_enabled:
            msg = (
                "Local pi0/openpi inference is disabled by default because pi0 checkpoints "
                "are heavy and checkpoint/config compatibility varies by LeRobot version. "
                "Use runtime='remote' for robot-side deployment, or call "
                "load_model('pi0', enable_local=True, pretrained='lerobot/pi0_base') "
                "in a dedicated GPU environment."
            )
            raise NotImplementedError(msg)
        return super().predict_observation(observation)
