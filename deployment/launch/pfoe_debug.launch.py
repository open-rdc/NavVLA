"""Launch file for PFoE topomap debug GUI (RViz node clicker)."""

import os
from pathlib import Path

import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = Path(get_package_share_directory("navvla"))
    pfoe_config = str(package_share / "config" / "pfoe.yaml")
    rviz_config = str(package_share / "rviz" / "pfoe_debug.rviz")

    with open(pfoe_config, "r") as f:
        params = yaml.safe_load(f)["pfoe"]["ros__parameters"]
    episode_data_dir_default = os.path.join(str(package_share), params["episode_data_dir"])
    traj_name_default = params["traj_name"]

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "episode_data_dir",
                default_value=episode_data_dir_default,
                description="Path to episode data directory",
            ),
            DeclareLaunchArgument(
                "traj_name",
                default_value=traj_name_default,
                description="Episode name under episode_data_dir",
            ),
            Node(
                package="pfoe",
                executable="topomap_viz_node.py",
                name="topomap_viz_node",
                output="screen",
                parameters=[
                    {
                        "episode_data_dir": LaunchConfiguration("episode_data_dir"),
                        "traj_name": LaunchConfiguration("traj_name"),
                        "frame_id": "map",
                    }
                ],
            ),
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="screen",
                arguments=["-d", rviz_config],
            ),
        ]
    )
