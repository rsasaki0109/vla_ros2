from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import Command, FindExecutable, LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    use_sim_time = LaunchConfiguration("use_sim_time")
    gz_args = LaunchConfiguration("gz_args")

    robot_description_content = Command(
        [
            PathJoinSubstitution([FindExecutable(name="xacro")]),
            " ",
            PathJoinSubstitution(
                [FindPackageShare("vla_ros2_gz"), "urdf", "vla_arm.urdf.xacro"]
            ),
        ]
    )
    robot_description = {"robot_description": robot_description_content}
    robot_controllers = PathJoinSubstitution(
        [FindPackageShare("vla_ros2_gz"), "config", "vla_arm_controllers.yaml"]
    )

    robot_state_publisher = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        output="screen",
        parameters=[robot_description, {"use_sim_time": use_sim_time}],
    )

    gz_spawn_entity = Node(
        package="ros_gz_sim",
        executable="create",
        output="screen",
        arguments=["-topic", "robot_description", "-name", "vla_arm", "-allow_renaming", "true"],
    )

    joint_state_broadcaster_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_state_broadcaster"],
    )
    joint_trajectory_controller_spawner = Node(
        package="controller_manager",
        executable="spawner",
        arguments=["joint_trajectory_controller", "--param-file", robot_controllers],
    )

    bridge = Node(
        package="ros_gz_bridge",
        executable="parameter_bridge",
        arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        output="screen",
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument("use_sim_time", default_value="true"),
            DeclareLaunchArgument("gz_args", default_value=""),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        PathJoinSubstitution(
                            [FindPackageShare("ros_gz_sim"), "launch", "gz_sim.launch.py"]
                        )
                    ]
                ),
                launch_arguments={"gz_args": [gz_args, " -r -v 1 empty.sdf"]}.items(),
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=gz_spawn_entity,
                    on_exit=[joint_state_broadcaster_spawner],
                )
            ),
            RegisterEventHandler(
                event_handler=OnProcessExit(
                    target_action=joint_state_broadcaster_spawner,
                    on_exit=[joint_trajectory_controller_spawner],
                )
            ),
            bridge,
            robot_state_publisher,
            gz_spawn_entity,
        ]
    )
