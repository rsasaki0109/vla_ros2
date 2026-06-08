from glob import glob

from setuptools import find_packages, setup

package_name = "vla_ros2_gz"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["tests"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/urdf", glob("urdf/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="vla_ros2 contributors",
    maintainer_email="maintainers@example.com",
    description="Gazebo Sim integration and VLAAction bridge for vla_ros2.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "vla_action_bridge_node = vla_ros2_gz_ros.action_bridge:main",
            "vla_smolvla_joint_bridge_node = vla_ros2_gz_ros.smolvla_joint_bridge:main",
        ],
    },
)
