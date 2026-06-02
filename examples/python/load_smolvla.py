from __future__ import annotations

import argparse

import numpy as np

from vla_zoo import load_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local LeRobot SmolVLA probe.")
    parser.add_argument("--pretrained", default="lerobot/smolvla_base")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    model = load_model(
        "smolvla",
        pretrained=args.pretrained,
        device=args.device,
        local_files_only=args.local_files_only,
    )
    image = np.zeros((256, 256, 3), dtype=np.uint8)
    state = np.zeros(6, dtype=np.float32)
    action = model.predict(image=image, instruction=args.instruction, state=state)
    print(action)
    print(action.to_numpy())


if __name__ == "__main__":
    main()
