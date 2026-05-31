from __future__ import annotations

import os

from vla_zoo import load_model


def main() -> None:
    remote_url = os.environ.get("VLA_ZOO_REMOTE_URL", "http://localhost:8000")
    model = load_model("dummy", runtime="remote", remote_url=remote_url)
    print(model.predict(image=None, instruction="test"))


if __name__ == "__main__":
    main()
