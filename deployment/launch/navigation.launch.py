"""Launch file for NavVLA navigation node."""

from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("navvla"))

    nav_config_default = str(package_share / "config" / "nav.yaml")
    preprocess_config_default = str(package_share / "config" / "preprocess.yaml")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "nav_config",
                default_value=nav_config_default,
                description="Path to navigation config",
            ),
            DeclareLaunchArgument(
                "preprocess_config",
                default_value=preprocess_config_default,
                description="Path to preprocessing config",
            ),
            Node(
                package="navvla",
                executable="navigation_node",
                name="navigation",
                output="screen",
                emulate_tty=True,
                arguments=[
                    "--nav-config",
                    LaunchConfiguration("nav_config"),
                    "--preprocess-config",
                    LaunchConfiguration("preprocess_config"),
                ],
            ),
        ]
    )
