"""Lightweight image serialization helpers for JSON runtimes."""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image


def encode_image_base64(image: Any, *, format: str = "JPEG") -> dict[str, str]:
    """Encode a PIL-compatible image as a base64 payload."""

    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    return {
        "encoding": f"{format.lower()}_base64",
        "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
    }


def decode_image_base64(payload: dict[str, str]) -> Image.Image:
    """Decode a base64 image payload into a PIL image."""

    encoding = payload.get("encoding", "")
    if not encoding.endswith("_base64"):
        msg = f"Unsupported image encoding {encoding!r}"
        raise ValueError(msg)
    raw = base64.b64decode(payload["data"])
    return Image.open(io.BytesIO(raw)).convert("RGB")
