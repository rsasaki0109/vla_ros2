from __future__ import annotations

import argparse

from PIL import Image

from vla_zoo import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one OpenVLA prediction.")
    parser.add_argument("--image", default="examples/assets/example.png")
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--pretrained", default="openvla/openvla-7b")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--unnorm-key", default="bridge_orig")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image = Image.open(args.image).convert("RGB")
    model = load_model(
        "openvla",
        pretrained=args.pretrained,
        device=args.device,
        dtype=args.dtype,
        unnorm_key=args.unnorm_key,
    )
    print(model.predict(image=image, instruction=args.instruction))


if __name__ == "__main__":
    main()
