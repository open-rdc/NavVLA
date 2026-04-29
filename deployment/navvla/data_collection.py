#!/usr/bin/env python3

import math
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from ament_index_python.packages import get_package_prefix
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Empty

SAMPLE_INTERVAL = 0.1  # 10 Hz
HORIZON = 8


class DataCollectionNode(Node):
    """
    単眼カメラ(USBカメラ)とMid-360(FAST-LIO)のオドメトリを記録するノード。
    /flag トピック(ジョイスティックAボタン等)で開始/停止を切り替える。
    収集データは NavVLA の EdgeNavigationDataset が読める形式で保存される。
    """

    def __init__(self) -> None:
        super().__init__('data_collection_node')
        self.bridge = CvBridge()

        try:
            # install/navvla -> install -> <ws_root>
            ws_root = Path(get_package_prefix('navvla')).parent.parent
            default_save_dir = str(ws_root / 'src' / 'NavVLA' / 'training' / 'dataset')
        except Exception:
            default_save_dir = str(Path.home() / 'navvla_dataset')

        self.declare_parameter('save_dir', default_save_dir)
        self.save_dir = Path(
            self.get_parameter('save_dir').get_parameter_value().string_value
        )

        self.raw_data_buffer = []

        self.latest_image: Optional[np.ndarray] = None
        self.latest_pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.is_recording = False
        self._odom_received = False
        self._img_received = False

        self.create_subscription(
            Image,
            '/image_raw',
            self._image_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Odometry,
            '/Odometry',
            self._odom_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            Empty,
            '/flag',
            self._flag_callback,
            10,
        )

        self.create_timer(SAMPLE_INTERVAL, self._timer_callback)

        self.get_logger().info('==========================================')
        self.get_logger().info('Data Collection Node Started')
        self.get_logger().info('Waiting for /image_raw and /Odometry...')
        self.get_logger().info('Publish to /flag to START/STOP recording')
        self.get_logger().info('==========================================')

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            self.latest_image = cv2.resize(cv_img, (224, 224), interpolation=cv2.INTER_LINEAR)
            if not self._img_received:
                self._img_received = True
                self.get_logger().info('Camera image received.')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')

    def _odom_callback(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        self.latest_pose = np.array([pos.x, pos.y, yaw], dtype=np.float32)
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info('Odometry received.')

    def _flag_callback(self, _msg: Empty) -> None:
        self.is_recording = not self.is_recording
        status = 'STARTED' if self.is_recording else 'PAUSED'
        self.get_logger().info(
            f'Recording {status}. Buffer size: {len(self.raw_data_buffer)}'
        )

    def _timer_callback(self) -> None:
        if not self.is_recording:
            return
        if self.latest_image is None or not self._odom_received:
            return

        self.raw_data_buffer.append((
            self.latest_image.copy(),
            self.latest_pose.copy(),
            time.time(),
        ))

        if len(self.raw_data_buffer) % 10 == 0:
            self.get_logger().info(f'Collected {len(self.raw_data_buffer)} frames.')

    def save_data(self) -> None:
        num_samples = len(self.raw_data_buffer)
        if num_samples < HORIZON + 1:
            self.get_logger().warn(
                f'Not enough data ({num_samples} frames). Need at least {HORIZON + 1}.'
            )
            return

        self.get_logger().info(f'Saving {num_samples} frames...')

        data_dir = self.save_dir
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        dataset_name = f'navvla_{timestamp_str}'
        traj_dir = data_dir / dataset_name / 'traj_0'
        traj_dir.mkdir(parents=True, exist_ok=True)

        positions = []
        yaws = []

        for i, (img, pose, _ts) in enumerate(self.raw_data_buffer):
            # EdgeNavigationDataset は traj_dir/{i}.jpg を期待する
            cv2.imwrite(str(traj_dir / f'{i}.jpg'), img)
            positions.append([pose[0], pose[1]])
            yaws.append(pose[2])

        traj_data = {
            'position': np.array(positions, dtype=np.float32),
            'yaw': np.array(yaws, dtype=np.float32),
        }
        with open(traj_dir / 'traj_data.pkl', 'wb') as f:
            pickle.dump(traj_data, f)

        with open(data_dir / dataset_name / 'traj_names.txt', 'w') as f:
            f.write('traj_0\n')

        self.get_logger().info(f'Saved dataset to: {traj_dir.parent}')


def main(args=None):
    rclpy.init(args=args)
    node = DataCollectionNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Finishing data collection...')
    finally:
        node.save_data()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
