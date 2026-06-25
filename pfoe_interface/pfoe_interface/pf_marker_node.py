#!/usr/bin/env python3
import math
import pickle
from pathlib import Path

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSDurabilityPolicy, QoSProfile

from std_msgs.msg import ColorRGBA, Int32, Int32MultiArray
from geometry_msgs.msg import Point, Pose, PoseArray, Quaternion
from visualization_msgs.msg import (
    InteractiveMarker,
    InteractiveMarkerControl,
    Marker,
    MarkerArray,
)
from interactive_markers import InteractiveMarkerServer, MenuHandler

NODE_COLOR = ColorRGBA(r=0.0, g=1.0, b=0.0, a=0.6)
ESTIMATE_COLOR = ColorRGBA(r=1.0, g=0.0, b=0.0, a=0.6)


def yaw_to_quaternion(yaw):
    return Quaternion(z=math.sin(yaw * 0.5), w=math.cos(yaw * 0.5))


class PfMarkerNode(Node):
    def __init__(self, **kwargs):
        super().__init__("pf_marker_node", **kwargs)
        data_dir = self.declare_parameter("episode_data_dir", "data/tsudanuma").value
        traj_name = self.declare_parameter("traj_name", "episode01").value
        self.frame_id = self.declare_parameter("frame_id", "map").value
        self.marker_scale = self.declare_parameter("marker_scale", 0.3).value
        self.label_stride = self.declare_parameter("label_stride", 10).value

        self.load_topomap(Path(data_dir) / traj_name)

        latched = QoSProfile(depth=1, durability=QoSDurabilityPolicy.TRANSIENT_LOCAL)
        self.topomap_pub = self.create_publisher(MarkerArray, "/topomap/markers", latched)
        self.particles_pub = self.create_publisher(PoseArray, "/pfoe/particles_viz", 10)
        self.set_time_pub = self.create_publisher(Int32, "/pfoe/set_time_idx", 10)

        self.create_subscription(Int32, "/pfoe/time_idx", self.time_idx_callback, 10)
        self.create_subscription(Int32MultiArray, "/pfoe/particles", self.particles_callback, 10)

        self.server = InteractiveMarkerServer(self, "pfoe_interface")
        self.menu_handler = MenuHandler()
        self.init_menu()

        self.estimate_idx = -1
        self.publish_topomap()
        self.build_markers()

    def load_topomap(self, episode_dir):
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

    def make_point(self, xy):
        return Point(x=float(xy[0]), y=float(xy[1]), z=0.0)

    def init_menu(self):
        self.menu_handler.insert("particle_reset", callback=self.particle_reset_callback)

    def make_node_marker(self, i, color):
        int_marker = InteractiveMarker()
        int_marker.header.frame_id = self.frame_id
        int_marker.pose.position = self.make_point(self.positions[i])
        int_marker.pose.orientation = yaw_to_quaternion(float(self.yaws[i]))
        int_marker.scale = self.marker_scale * 2.0
        int_marker.name = str(i)
        int_marker.description = ""

        cylinder = Marker()
        cylinder.type = Marker.CYLINDER
        cylinder.scale.x = cylinder.scale.y = self.marker_scale * 2.0
        cylinder.scale.z = 0.02
        cylinder.pose.position.z = 0.01
        cylinder.color = color

        control = InteractiveMarkerControl()
        control.interaction_mode = InteractiveMarkerControl.BUTTON
        control.always_visible = True
        control.markers.append(cylinder)
        int_marker.controls.append(control)
        return int_marker

    def insert_node_marker(self, i, color):
        int_marker = self.make_node_marker(i, color)
        self.server.insert(int_marker, feedback_callback=self.process_feedback)
        self.menu_handler.apply(self.server, int_marker.name)

    def build_markers(self):
        for i in range(len(self.positions)):
            color = ESTIMATE_COLOR if i == self.estimate_idx else NODE_COLOR
            self.insert_node_marker(i, color)
        self.server.applyChanges()

    def process_feedback(self, feedback):
        pass

    def particle_reset_callback(self, feedback):
        i = int(feedback.marker_name)
        time_idx = i + 1
        self.set_time_pub.publish(Int32(data=time_idx))
        prompt = self.prompts[i] if i < len(self.prompts) else ""
        self.get_logger().info(f"particle_reset on node {i} (time_idx {time_idx}): {prompt}")

    def time_idx_callback(self, msg):
        new_idx = max(0, min(len(self.positions) - 1, msg.data - 1))
        if new_idx == self.estimate_idx:
            return
        old_idx = self.estimate_idx
        self.estimate_idx = new_idx
        if old_idx >= 0:
            self.insert_node_marker(old_idx, NODE_COLOR)
        self.insert_node_marker(new_idx, ESTIMATE_COLOR)
        self.server.applyChanges()

    def particles_callback(self, msg):
        poses = PoseArray()
        poses.header.frame_id = self.frame_id
        for time_idx in msg.data:
            i = max(0, min(len(self.positions) - 1, time_idx - 1))
            pose = Pose()
            pose.position = self.make_point(self.positions[i])
            pose.orientation = yaw_to_quaternion(float(self.yaws[i]))
            poses.poses.append(pose)
        self.particles_pub.publish(poses)

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

        self.topomap_pub.publish(markers)


def main():
    rclpy.init()
    rclpy.spin(PfMarkerNode())


if __name__ == "__main__":
    main()
