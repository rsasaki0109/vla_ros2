"""Hugging Face SmolVLM2 adapter (the VLM family SmolVLA is built on)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from vla_ros2.core.errors import MissingDependencyError
from vla_ros2.vlm.base import BaseVLM, ImageArray

DEFAULT_PRETRAINED = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"

_YES_WORDS = ("Yes", "yes", "YES")
_NO_WORDS = ("No", "no", "NO")


def _import_dependencies() -> tuple[Any, Any, Any, Any]:
    try:
        import torch
        from PIL import Image
        from transformers import AutoModelForImageTextToText, AutoProcessor
    except ImportError as exc:
        msg = (
            "SmolVLM adapter requires transformers and torch. "
            'Install them with: pip install "vla_ros2[vlm]"'
        )
        raise MissingDependencyError(msg) from exc
    return torch, Image, AutoModelForImageTextToText, AutoProcessor


class SmolVLMAdapter(BaseVLM):
    """Local SmolVLM2 inference through ``transformers``."""

    name = "smolvlm"

    def __init__(
        self,
        *,
        pretrained: str | None = None,
        device: str = "auto",
        dtype: str | None = None,
        local_files_only: bool = False,
        revision: str | None = None,
        image_splitting: bool | None = None,
        **_: object,
    ) -> None:
        torch, image_cls, model_cls, processor_cls = _import_dependencies()
        self._torch = torch
        self._image_cls = image_cls
        self.model_id = pretrained or DEFAULT_PRETRAINED
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        if dtype is None:
            dtype = "bfloat16" if device.startswith("cuda") else "float32"
        self.dtype = getattr(torch, dtype)
        self.processor = processor_cls.from_pretrained(
            self.model_id, local_files_only=local_files_only, revision=revision
        )
        self.model = model_cls.from_pretrained(
            self.model_id,
            dtype=self.dtype,
            local_files_only=local_files_only,
            revision=revision,
        ).to(device)
        self.model.eval()
        if image_splitting is not None:
            # Tiling the frame is ~15x slower on CPU but separates success/failure
            # far better than one downscaled view; keep the checkpoint default
            # unless latency matters more than accuracy.
            self.processor.image_processor.do_image_splitting = image_splitting
        tokenizer = self.processor.tokenizer
        self._yes_ids = self._first_token_ids(tokenizer, _YES_WORDS)
        self._no_ids = self._first_token_ids(tokenizer, _NO_WORDS)

    @staticmethod
    def _first_token_ids(tokenizer: Any, words: Sequence[str]) -> list[int]:
        ids: set[int] = set()
        for word in words:
            for variant in (word, f" {word}"):
                tokens = tokenizer.encode(variant, add_special_tokens=False)
                if tokens:
                    ids.add(int(tokens[0]))
        return sorted(ids)

    def _inputs(self, prompt: str, images: Sequence[ImageArray]) -> Any:
        content: list[dict[str, Any]] = [
            {"type": "image", "image": self._image_cls.fromarray(image).convert("RGB")}
            for image in images
        ]
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        inputs = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)
        return inputs

    def generate(
        self,
        prompt: str,
        images: Sequence[ImageArray] = (),
        *,
        max_new_tokens: int = 16,
    ) -> str:
        inputs = self._inputs(prompt, images)
        with self._torch.inference_mode():
            output = self.model.generate(**inputs, do_sample=False, max_new_tokens=max_new_tokens)
        new_tokens = output[:, inputs["input_ids"].shape[1] :]
        decoded = self.processor.batch_decode(new_tokens, skip_special_tokens=True)
        return str(decoded[0]).strip()

    def yes_probability(self, prompt: str, images: Sequence[ImageArray] = ()) -> float | None:
        if not self._yes_ids or not self._no_ids:
            return None
        inputs = self._inputs(prompt, images)
        with self._torch.inference_mode():
            logits = self.model(**inputs).logits[0, -1].float()
        probs = self._torch.softmax(logits, dim=-1)
        p_yes = float(probs[self._yes_ids].sum())
        p_no = float(probs[self._no_ids].sum())
        total = p_yes + p_no
        if total <= 0.0:
            return None
        return p_yes / total
