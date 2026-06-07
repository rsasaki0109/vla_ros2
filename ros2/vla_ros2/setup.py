from glob import glob

from setuptools import find_packages, setup

package_name = "vla_ros2"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vla_ros2 contributors",
    maintainer_email="maintainers@example.com",
    description="ROS2 runtime node and launch files for vla_ros2.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vla_action_replay_node = vla_ros2_ros.action_replay:main",
            "vla_runtime_node = vla_ros2_ros.node:main",
            "vla_runtime_recorder = vla_ros2_ros.log_recorder:main",
            "vla_smoke_input_node = vla_ros2_ros.smoke_input:main",
        ],
    },
)
