from __future__ import annotations

from typing import Any

import numpy as np

from vla_zoo.core.errors import MissingDependencyError
from vla_zoo.core.model import VLAAdapter
from vla_zoo.core.types import ActionSpec, VLAAction, VLAObservation

DEFAULT_OPENVLA_SPEC = ActionSpec(
    action_space="eef_delta",
    shape=(7,),
    description="OpenVLA-style 7-DoF action; adapter-specific unnormalization may apply.",
)


def _import_openvla_dependencies() -> tuple[Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForVision2Seq, AutoProcessor
    except ImportError as exc:
        msg = (
            "OpenVLA adapter requires optional ML dependencies. "
            'Install them with: pip install "vla_zoo[openvla]"'
        )
        raise MissingDependencyError(msg) from exc
    return torch, Image, (AutoProcessor, AutoModelForVision2Seq)


class OpenVLAAdapter(VLAAdapter):
    """Lazy Hugging Face adapter for OpenVLA models."""

    name = "openvla"
    model_id = "openvla/openvla-7b"
    action_spec = DEFAULT_OPENVLA_SPEC

    def __init__(
        self,
        *,
        pretrained: str = "openvla/openvla-7b",
        device: str = "cuda:0",
        dtype: str = "bfloat16",
        attn_implementation: str | None = None,
        unnorm_key: str | None = None,
        trust_remote_code: bool = True,
        do_sample: bool = False,
        prompt_template: str = "In: What action should the robot take to {instruction}?\n Out:",
        action_spec: ActionSpec | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            name=self.name,
            model_id=pretrained,
            action_spec=action_spec or self.action_spec,
        )
        torch, image_module, hf_classes = _import_openvla_dependencies()
        auto_processor, auto_model = hf_classes
        self._torch = torch
        self._image_module = image_module
        self.pretrained = pretrained
        self.device = device
        self.dtype = dtype
        self.unnorm_key = unnorm_key
        self.do_sample = do_sample
        self.prompt_template = prompt_template

        torch_dtype = getattr(torch, dtype, None) if dtype else None
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation
        model_kwargs.update(kwargs)

        self.processor = auto_processor.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
        )
        self.model = auto_model.from_pretrained(pretrained, **model_kwargs)
        if hasattr(self.model, "to"):
            self.model = self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @classmethod
    def from_config(cls, **kwargs: Any) -> OpenVLAAdapter:
        return cls(**kwargs)

    def _primary_image(self, observation: VLAObservation) -> Any:
        if "primary" not in observation.images:
            msg = "OpenVLA requires observation.images['primary']"
            raise ValueError(msg)
        image = observation.images["primary"]
        if isinstance(image, self._image_module.Image):
            return image
        return self._image_module.fromarray(np.asarray(image)).convert("RGB")

    def predict_observation(self, observation: VLAObservation) -> VLAAction:
        image = self._primary_image(observation)
        prompt = self.prompt_template.format(instruction=observation.instruction)
        inputs = self.processor(prompt, image, return_tensors="pt")
        if hasattr(inputs, "to"):
            inputs = inputs.to(self.device)

        if hasattr(self.model, "predict_action"):
            try:
                action = self.model.predict_action(
                    **inputs,
                    unnorm_key=self.unnorm_key,
                    do_sample=self.do_sample,
                )
            except TypeError:
                action = self.model.predict_action(
                    image,
                    prompt,
                    unnorm_key=self.unnorm_key,
                    do_sample=self.do_sample,
                )
        else:
            with self._torch.inference_mode():
                generated = self.model.generate(**inputs, do_sample=self.do_sample)
            action = self.processor.decode(generated[0], skip_special_tokens=True)

        data = np.asarray(action, dtype=np.float32)
        if data.shape != self.action_spec.shape:
            data = data.reshape(self.action_spec.shape)
        return VLAAction(
            data=data,
            spec=self.action_spec,
            metadata={
                "model": self.pretrained,
                "adapter": type(self).__name__,
                "unnorm_key": self.unnorm_key,
            },
        )
