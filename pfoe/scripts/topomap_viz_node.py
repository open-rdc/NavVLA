#!/usr/bin/env python3
import math
import pickle
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from std_msgs.msg import ColorRGBA, Int32, Int32MultiArray
from geometry_msgs.msg import Point, Pose, PoseArray, PointStamped, Quaternion
from visualization_msgs.msg import Marker, MarkerArray


def yaw_to_quaternion(yaw: float) -> Quaternion:
    return Quaternion(z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


class TopomapVizNode(Node):
    def __init__(self):
        super().__init__("topomap_viz_node")
        data_dir = self.declare_parameter("episode_data_dir", "data/tsudanuma").value
        traj_name = self.declare_parameter("traj_name", "episode01").value
        self.frame_id = self.declare_parameter("frame_id", "map").value
        self.marker_scale = self.declare_parameter("marker_scale", 0.3).value
        self.label_stride = self.declare_parameter("label_stride", 10).value

        episode_dir = Path(data_dir) / traj_name
        with (episode_dir / "traj_data.pkl").open("rb") as f:
            traj = pickle.load(f)
        positions = np.asarray(traj["position"], dtype=np.float32)
        yaws = np.asarray(traj["yaw"], dtype=np.float32)
        prompt_path = episode_dir / "traj_prompt.txt"
        self.prompts = prompt_path.read_text(encoding="utf-8").splitlines() if prompt_path.exists() else []

        n_nodes = len(self.prompts) if self.prompts else len(positions)
        if len(positions) == n_nodes:
            self.positions = positions
            self.yaws = yaws
        else:
            self.get_logger().warn(
                f"position count {len(positions)} != node count {n_nodes}; "
                "resampling positions to align nodes with pfoe time index"
            )
            idx = np.arange(n_nodes) * len(positions) // n_nodes
            self.positions = positions[idx]
            self.yaws = yaws[idx]

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.marker_pub = self.create_publisher(MarkerArray, "/topomap/markers", latched)
        self.estimate_pub = self.create_publisher(Marker, "/pfoe/estimate_marker", 10)
        self.particles_pub = self.create_publisher(PoseArray, "/pfoe/particles_viz", 10)
        self.set_time_pub = self.create_publisher(Int32, "/pfoe/set_time_idx", 10)

        self.create_subscription(PointStamped, "/clicked_point", self.clicked_callback, 10)
        self.create_subscription(Int32, "/pfoe/time_idx", self.time_idx_callback, 10)
        self.create_subscription(Int32MultiArray, "/pfoe/particles", self.particles_callback, 10)

        self.publish_topomap()
        self.get_logger().info(
            f"topomap_viz_node ready ({len(self.positions)} nodes from {episode_dir})"
        )

    def position_of(self, time_idx: int) -> np.ndarray:
        i = max(0, min(len(self.positions) - 1, time_idx - 1))
        return self.positions[i]

    def make_point(self, xy: np.ndarray) -> Point:
        return Point(x=float(xy[0]), y=float(xy[1]), z=0.0)

    def publish_topomap(self):
        markers = MarkerArray()

        line = Marker()
        line.header.frame_id = self.frame_id
        line.ns = "trajectory"
        line.id = 0
        line.type = Marker.LINE_STRIP
        line.action = Marker.ADD
        line.scale.x = self.marker_scale * 0.2
        line.color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8)
        line.points = [self.make_point(p) for p in self.positions]
        markers.markers.append(line)

        nodes = Marker()
        nodes.header.frame_id = self.frame_id
        nodes.ns = "nodes"
        nodes.id = 1
        nodes.type = Marker.SPHERE_LIST
        nodes.action = Marker.ADD
        nodes.scale.x = nodes.scale.y = nodes.scale.z = self.marker_scale
        nodes.color = ColorRGBA(r=0.1, g=0.4, b=1.0, a=1.0)
        nodes.points = [self.make_point(p) for p in self.positions]
        markers.markers.append(nodes)

        for i in range(0, len(self.positions), max(1, int(self.label_stride))):
            text = Marker()
            text.header.frame_id = self.frame_id
            text.ns = "labels"
            text.id = 100 + i
            text.type = Marker.TEXT_VIEW_FACING
            text.action = Marker.ADD
            text.pose.position = self.make_point(self.positions[i])
            text.pose.position.z = self.marker_scale * 1.5
            text.scale.z = self.marker_scale
            text.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            text.text = str(i)
            markers.markers.append(text)

        self.marker_pub.publish(markers)

    def clicked_callback(self, msg: PointStamped):
        click = np.array([msg.point.x, msg.point.y], dtype=np.float32)
        i = int(np.argmin(np.sum((self.positions - click) ** 2, axis=1)))
        time_idx = i + 1
        self.set_time_pub.publish(Int32(data=time_idx))
        prompt = self.prompts[i] if i < len(self.prompts) else ""
        self.get_logger().info(f"clicked node {i} (time_idx {time_idx}): {prompt}")

    def time_idx_callback(self, msg: Int32):
        xy = self.position_of(msg.data)
        marker = Marker()
        marker.header.frame_id = self.frame_id
        marker.ns = "estimate"
        marker.id = 0
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position = self.make_point(xy)
        marker.scale.x = marker.scale.y = marker.scale.z = self.marker_scale * 2.5
        marker.color = ColorRGBA(r=0.1, g=1.0, b=0.1, a=0.9)
        self.estimate_pub.publish(marker)

    def particles_callback(self, msg: Int32MultiArray):
        poses = PoseArray()
        poses.header.frame_id = self.frame_id
        for time_idx in msg.data:
            i = max(0, min(len(self.positions) - 1, time_idx - 1))
            pose = Pose()
            pose.position = self.make_point(self.positions[i])
            pose.orientation = yaw_to_quaternion(float(self.yaws[i]))
            poses.poses.append(pose)
        self.particles_pub.publish(poses)


def main():
    rclpy.init()
    rclpy.spin(TopomapVizNode())


if __name__ == "__main__":
    main()
