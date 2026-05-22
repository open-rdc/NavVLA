#!/usr/bin/env python3

import math
import pickle
from pathlib import Path
import time

import cv2
import numpy as np
from nav_msgs.msg import Odometry
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_system_default
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge


SAMPLE_INTERVAL = 0.2


class DataCreator(Node):
    CROP_SIZE = 224

    def __init__(self):
        super().__init__('create_data')
        self.bridge = CvBridge()
        self.collect_flag = False
        self.latest_odom = None
        self.latest_image = None

        repo_root = Path(__file__).parent.parent.parent
        data_base_dir = repo_root / 'data'
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        self.dataset_dir = data_base_dir / f'{timestamp}_dataset'
        self.dataset_dir.mkdir(parents=True, exist_ok=True)

        self.current_episode_index = 0
        self.current_sample_index = 0
        self.total_collected_samples = 0
        self.current_episode_dir = None
        self.current_positions = []
        self.current_yaws = []

        self.create_subscription(Empty, '/flag', self.flag_callback, qos_profile_system_default)
        self.create_subscription(Odometry, '/Odometry', self.odom_callback, qos_profile_system_default)
        self.create_subscription(Image, '/image_raw', self.image_callback, qos_profile_system_default)
        self.create_timer(SAMPLE_INTERVAL, self.timer_callback)
        
    def flag_callback(self, _msg):
        self.collect_flag = not self.collect_flag
        if self.collect_flag:
            self._start_new_episode()
        else:
            self._finalize_current_episode()
            self.get_logger().info('🔴Data collect stopped')

    def _start_new_episode(self):
        self.current_episode_index += 1
        self.current_sample_index = 0
        self.current_positions = []
        self.current_yaws = []

        self.current_episode_dir = self.dataset_dir / f'episode{self.current_episode_index:02d}'
        self.current_episode_dir.mkdir(parents=True, exist_ok=True)
        self.get_logger().info(f'⚪Create data started: {self.current_episode_dir.name}')

    def _center_crop(self, image):
        height, width = image.shape[:2]
        if width <= height:
            raise ValueError(f'Expected aspect ratio > 1 (width > height), got {width}x{height}')

        side = height
        top = 0
        left = (width - side) // 2
        square_image = image[top:top + side, left:left + side]

        interpolation = cv2.INTER_AREA if side >= self.CROP_SIZE else cv2.INTER_LINEAR
        return cv2.resize(square_image, (self.CROP_SIZE, self.CROP_SIZE), interpolation=interpolation)

    def _yaw_from_quaternion(self, x, y, z, w):
        siny_cosp = 2.0 * (w * z + x * y)
        cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
        return math.atan2(siny_cosp, cosy_cosp)

    def _finalize_current_episode(self):
        if self.current_episode_dir is None:
            return

        if self.current_sample_index == 0:
            self.current_episode_dir.rmdir()
            self.current_episode_dir = None
            self.get_logger().info('🔴No samples in current episode, removed empty directory')
            return

        positions = np.asarray(self.current_positions, dtype=np.float32)
        yaws = np.asarray(self.current_yaws, dtype=np.float32)
        yaws = np.unwrap(yaws).astype(np.float32)

        traj_data = {
            'position': positions,
            'yaw': yaws,
        }
        traj_data_path = self.current_episode_dir / 'traj_data.pkl'
        with traj_data_path.open('wb') as f:
            pickle.dump(traj_data, f)

        self.get_logger().info(
            f'🔵Saved traj_data.pkl for episode{self.current_episode_index:02d} '
            f'({self.current_sample_index} frames)'
        )
        self.current_episode_dir = None

    def odom_callback(self, msg):
        self.latest_odom = msg

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.latest_image = None
            self.get_logger().warning(f'Failed to convert image from /image_raw: {e}')

    def timer_callback(self):
        if not self.collect_flag:
            return
        if self.latest_odom is None or self.current_episode_dir is None or self.latest_image is None:
            self.get_logger().info(
                f'not ready latest_odom: {self.latest_odom}, '
                f'current_episode_dir: {self.current_episode_dir}, latest_image: {self.latest_image is not None}'
            )
            return

        image = self.latest_image
        cropped_image = self._center_crop(image)


        image_path = self.current_episode_dir / f'{self.current_sample_index}.jpg'
        cv2.imwrite(str(image_path), cropped_image, [int(cv2.IMWRITE_JPEG_QUALITY), 95])

        pose = self.latest_odom.pose.pose
        position = pose.position
        orientation = pose.orientation

        self.current_positions.append([position.x, position.y])
        self.current_yaws.append(
            self._yaw_from_quaternion(
                orientation.x,
                orientation.y,
                orientation.z,
                orientation.w,
            )
        )

        self.get_logger().info(f'🟢Collected episode{self.current_episode_index:02d} #{self.current_sample_index}')
        self.current_sample_index += 1
        self.total_collected_samples += 1

    def save_data(self) -> None:
        if self.current_episode_index == 0:
            self.get_logger().info('🔴No data to save')
            return
        if self.collect_flag:
            self._finalize_current_episode()
        self.get_logger().info(
            f'🔵Saved {self.total_collected_samples} samples in {self.current_episode_index} episodes to {self.dataset_dir}'
        )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DataCreator()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrupted by user')
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()