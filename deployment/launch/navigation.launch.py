"""Launch file for NavVLA navigation node."""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("navvla"))

    nav_config_default = str(package_share / "config" / "nav.yaml")
    preprocess_config_default = str(package_share / "config" / "preprocess.yaml")
    pfoe_config = str(package_share / "config" / "pfoe.yaml")

    with open(pfoe_config, "r") as f:
        rel_data_dir = yaml.safe_load(f)["pfoe"]["ros__parameters"]["episode_data_dir"]
    episode_data_dir_default = os.path.join(str(package_share), rel_data_dir)

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
            DeclareLaunchArgument(
                "episode_data_dir",
                default_value=episode_data_dir_default,
                description="Path to episode data directory",
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
            # voice guidance
            Node(
                package="navvla",
                executable="voice_node",
                name="voice_guidance",
                output="screen",
                emulate_tty=True,
            ),


            # pfoe: CLIP encoder (Python)
            Node(
                package="pfoe",
                executable="clip_encoder_node.py",
                name="clip_encoder",
                output="screen",
                parameters=[pfoe_config],
            ),
            # pfoe: particle filter + /prompt publisher (C++)
            Node(
                package="pfoe",
                executable="pfoe_node",
                name="pfoe",
                output="screen",
                parameters=[
                    pfoe_config,
                    {"episode_data_dir": LaunchConfiguration("episode_data_dir")},
                ],
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    str(package_share / "launch" / "pfoe_debug.launch.py")
                ),
                launch_arguments={
                    "episode_data_dir": LaunchConfiguration("episode_data_dir")
                }.items(),
            ),
        ]
    )
