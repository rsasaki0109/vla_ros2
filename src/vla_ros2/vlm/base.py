"""Base class for vision-language models (VLMs) used beside the VLA policy.

A VLM never commands the robot. It answers questions about camera frames at a
low rate (task success, scene checks) so the action loop can stay on the VLA.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

ImageArray: TypeAlias = NDArray[np.uint8]


class BaseVLM(ABC):
    """Local VLM that answers a text prompt about zero or more RGB frames."""

    name: str = "base"
    model_id: str = "none"

    @abstractmethod
    def generate(
        self,
        prompt: str,
        images: Sequence[ImageArray] = (),
        *,
        max_new_tokens: int = 16,
    ) -> str:
        """Return the model's free-form answer to ``prompt``."""

    def yes_probability(self, prompt: str, images: Sequence[ImageArray] = ()) -> float | None:
        """Return P(answer is "yes") for a yes/no prompt, or ``None`` if unsupported.

        Adapters that expose logits should override this; it is a calibrated
        signal, unlike parsing the generated text.
        """

        return None

    def info(self) -> dict[str, Any]:
        return {"name": self.name, "model_id": self.model_id}
