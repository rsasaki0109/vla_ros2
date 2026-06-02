from __future__ import annotations

import argparse
import os

import numpy as np

from vla_zoo import load_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Call a remote pi0/openpi server through the vla_zoo remote runtime.",
    )
    parser.add_argument(
        "--remote-url",
        default=os.environ.get("VLA_ZOO_REMOTE_URL", "http://gpu-box:8000"),
        help="Base URL of the pi0 server, e.g. http://gpu-box:8000.",
    )
    parser.add_argument("--instruction", default="pick up the red block")
    args = parser.parse_args()

    # pi0 is remote-first: the LeRobot/openpi stack lives in the server environment,
    # so the robot-side client only needs the lightweight remote runtime.
    model = load_model("pi0", runtime="remote", remote_url=args.remote_url)
    image = np.zeros((224, 224, 3), dtype=np.uint8)
    state = np.zeros(7, dtype=np.float32)
    action = model.predict(image=image, instruction=args.instruction, state=state)
    print(action)


if __name__ == "__main__":
    main()
