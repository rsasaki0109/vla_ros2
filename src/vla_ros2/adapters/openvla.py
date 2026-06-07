from __future__ import annotations

from typing import Any

import numpy as np

from vla_ros2.core.errors import MissingDependencyError
from vla_ros2.core.model import VLAAdapter
from vla_ros2.core.types import ActionSpec, VLAAction, VLAObservation

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
            'Install them with: pip install "vla_ros2[openvla]"'
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
        attn_implementation: str | None = "eager",
        unnorm_key: str | None = None,
        trust_remote_code: bool = True,
        do_sample: bool = False,
        load_in_4bit: bool = False,
        load_in_8bit: bool = False,
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
        self.torch_dtype = torch_dtype
        model_kwargs: dict[str, Any] = {"trust_remote_code": trust_remote_code}
        if torch_dtype is not None:
            model_kwargs["torch_dtype"] = torch_dtype
        if attn_implementation is not None:
            model_kwargs["attn_implementation"] = attn_implementation

        # bitsandbytes quantization lets the 7B model fit a 16 GB consumer GPU (bf16 weights
        # are ~15 GB and do not fit alongside activations). Quantized models are placed via
        # ``device_map`` at load time and must not be moved with ``.to(device)`` afterward.
        quantized = load_in_4bit or load_in_8bit
        if quantized:
            model_kwargs["quantization_config"] = self._build_quantization_config(
                load_in_4bit=load_in_4bit,
                load_in_8bit=load_in_8bit,
                compute_dtype=torch_dtype,
            )
            model_kwargs["device_map"] = {"": device}
        model_kwargs.update(kwargs)

        self.processor = auto_processor.from_pretrained(
            pretrained,
            trust_remote_code=trust_remote_code,
        )
        self.model = auto_model.from_pretrained(pretrained, **model_kwargs)
        if not quantized and hasattr(self.model, "to"):
            self.model = self.model.to(device)
        if hasattr(self.model, "eval"):
            self.model.eval()

    @staticmethod
    def _build_quantization_config(
        *,
        load_in_4bit: bool,
        load_in_8bit: bool,
        compute_dtype: Any,
    ) -> Any:
        try:
            from transformers import BitsAndBytesConfig
        except ImportError as exc:  # pragma: no cover - mirrors the main dependency gate
            msg = (
                "4-bit/8-bit loading requires bitsandbytes. "
                'Install it with: pip install "vla_ros2[openvla]" bitsandbytes'
            )
            raise MissingDependencyError(msg) from exc
        if load_in_4bit:
            return BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
        return BitsAndBytesConfig(load_in_8bit=True)

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
        if self.torch_dtype is not None and hasattr(inputs, "items"):
            for key, value in inputs.items():
                if hasattr(value, "is_floating_point") and value.is_floating_point():
                    inputs[key] = value.to(dtype=self.torch_dtype)

        if hasattr(self.model, "predict_action"):
            predict_inputs = dict(inputs)
            # OpenVLA's remote-code predict_action appends an action-start token
            # to input_ids but does not extend attention_mask, so passing the
            # tokenizer mask can create a one-token generation mismatch.
            predict_inputs.pop("attention_mask", None)
            try:
                action = self.model.predict_action(
                    **predict_inputs,
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
