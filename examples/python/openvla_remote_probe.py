from __future__ import annotations

import argparse
import json
import os

from vla_zoo.runtime.remote_probe import probe_remote_model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Health-first probe against a remote OpenVLA server.",
    )
    parser.add_argument(
        "--remote-url",
        default=os.environ.get("VLA_ZOO_REMOTE_URL", "http://gpu-box:8000"),
        help="Base URL of the OpenVLA server, e.g. http://gpu-box:8000.",
    )
    parser.add_argument("--model", default="openvla")
    parser.add_argument("--instruction", default="pick up the red block")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    result = probe_remote_model(
        model_name=args.model,
        remote_url=args.remote_url,
        instruction=args.instruction,
        timeout=args.timeout,
    )
    print(json.dumps(result.to_dict(), indent=2))
    if not result.is_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
