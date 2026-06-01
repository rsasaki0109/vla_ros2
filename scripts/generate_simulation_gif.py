from __future__ import annotations

from pathlib import Path

from vla_zoo.demo.pybullet import PyBulletDemoConfig, render_pybullet_demo


def main() -> None:
    result = render_pybullet_demo(
        PyBulletDemoConfig(out=Path("docs/assets/simulation_pick_place.gif"))
    )
    print(f"{result['out']} ({result['frames']} frames)")


if __name__ == "__main__":
    main()
