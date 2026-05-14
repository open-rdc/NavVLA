#!/usr/bin/env python3

import math
import pickle
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
import shutil

SAMPLE_INTERVAL = 0.1  # 10 Hz

# data_collection/data_collection.py -> parents[1] = NavVLA/
_DEFAULT_SAVE_DIR = str(Path(__file__).resolve().parents[1] / 'training' / 'dataset')


class DataCollectionNode(Node):
    """
    単眼カメラ(USBカメラ)とMid-360(FAST-LIO)のオドメトリを記録するノード。
    /flag トピック(ジョイスティックAボタン等)で開始/停止を切り替える。
    収集データは NavVLA の EdgeNavigationDataset が読める形式で保存される。
    """

    def __init__(self) -> None:
        super().__init__('data_collection_node')
        self.bridge = CvBridge()

        self.declare_parameter('save_dir', _DEFAULT_SAVE_DIR)
        self.save_dir = Path(
            self.get_parameter('save_dir').get_parameter_value().string_value
        )

        # Training dataset parameters — must match training/config/network.yaml and dataset.yaml.
        # Used to compute the minimum trajectory length that yields at least one trainable sample.
        # Formula (from training/data/dataset.py):
        #   min_frames = (context_size + len_traj_pred) * waypoint_spacing + end_slack + 1
        self.declare_parameter('context_size', 5)
        self.declare_parameter('len_traj_pred', 8)
        self.declare_parameter('waypoint_spacing', 1)
        self.declare_parameter('end_slack', 3)

        context_size    = self.get_parameter('context_size').get_parameter_value().integer_value
        len_traj_pred   = self.get_parameter('len_traj_pred').get_parameter_value().integer_value
        waypoint_spacing = self.get_parameter('waypoint_spacing').get_parameter_value().integer_value
        end_slack       = self.get_parameter('end_slack').get_parameter_value().integer_value
        self.min_frames = (context_size + len_traj_pred) * waypoint_spacing + end_slack + 1

        self.raw_data_buffer = []
        self.frame_count = 0
        self.temp_dir = None
        self.chunk_size = 1000
        self.dataset_dir: Optional[Path] = None
        self.traj_count = 0
        self.traj_names = []

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
        self.get_logger().info(f'Save dir: {self.save_dir}')
        self.get_logger().info(f'Min frames required: {self.min_frames}')
        self.get_logger().info('==========================================')

    def _image_callback(self, msg: Image) -> None:
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

            h, w = cv_img.shape[:2]
            crop_size = min(h, w)
            x_start = (w - crop_size) // 2
            y_start = (h - crop_size) // 2
            cropped_img = cv_img[y_start:y_start + crop_size, x_start:x_start + crop_size]
            self.latest_image = cv2.resize(cropped_img, (224, 224), interpolation=cv2.INTER_LINEAR)
            if not self._img_received:
                self._img_received = True
                self.get_logger().info('Camera image received.')
        except Exception as e:
            self.get_logger().error(f'Failed to convert image: {e}')


    def _odom_callback(self, msg: Odometry) -> None:
        pos = msg.pose.pose.position
        q = msg.pose.pose.orientation
        self.latest_pose = np.array([pos.x, pos.y, q.w, q.x, q.y, q.z], dtype=np.float32)
        if not self._odom_received:
            self._odom_received = True
            self.get_logger().info('Odometry received.')


    def _flag_callback(self, _msg: Empty) -> None:
        self.is_recording = not self.is_recording
        status = 'STARTED' if self.is_recording else 'PAUSED'

        if self.is_recording:
            timestamp_str = time.strftime('%Y%m%d_%H%M%S')
            self.temp_dir = Path('/tmp') / f'navvla_recording_{timestamp_str}'
            self.temp_dir.mkdir(parents=True, exist_ok=True)
            dataset_name = f'navvla_{timestamp_str}'
            self.dataset_dir = self.save_dir / dataset_name
            self.dataset_dir.mkdir(parents=True, exist_ok=True)
            self.raw_data_buffer = []
            self.frame_count = 0
            self.traj_count = 0
            self.traj_names = []
            self.get_logger().info(f'🟢 recording {status}. Temp dir: {self.temp_dir}')
        else:
            self._flush_buffer(final=True)
            self.get_logger().info(f' ⏸️ Recording {status}. Buffer size: {len(self.raw_data_buffer)}')


    def _timer_callback(self) -> None:
        if not self.is_recording:
            return
        if self.latest_image is None or not self._odom_received:
            return

        x, y, qw, qx, qy, qz = self.latest_pose
        yaw = math.atan2(
            2.0 * (qw * qz + qx * qy),
            1.0 - 2.0 * (qy * qy + qz * qz),
        )
                
        pose_with_yaw = np.array([x, y, yaw], dtype=np.float32)
        chunk_frame_id = len(self.raw_data_buffer)

        self.raw_data_buffer.append({
            'frame_id' : chunk_frame_id,
            'pose': pose_with_yaw.copy(),  # 12 bytes
        })

        if self.temp_dir is not None:
            ok = cv2.imwrite(
                str(self.temp_dir / f'{chunk_frame_id:06d}.jpg'),
                self.latest_image
            )
            if not ok:
                self.raw_data_buffer.pop()  # pose と画像の対応を保つ
                self.get_logger().error(
                    f'❌Failed to write image {chunk_frame_id:06d}.jpg — frame dropped.'
                )
                return

        self.frame_count += 1

        if len(self.raw_data_buffer) >= self.chunk_size:
            self._flush_buffer(final=False)
        
        if self.frame_count % 10 == 0:
            self.get_logger().info(
                f'📊 Collected {self.frame_count} frames. '
                f'Buffer: {len(self.raw_data_buffer)} / {self.chunk_size}'
            )

    def _reset_temp_dir(self) -> None:
        if self.temp_dir is not None and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
            except Exception as e:
                self.get_logger().error(f'❌Failed to clean temp dir: {e}')
        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        self.temp_dir = Path('/tmp') / f'navvla_recording_{timestamp_str}_{self.traj_count}'
        self.temp_dir.mkdir(parents=True, exist_ok=True)

    def _flush_buffer(self, final: bool) -> None:
        num_samples = len(self.raw_data_buffer)
        if num_samples == 0:
            return

        if final and num_samples < self.min_frames:
            self.get_logger().warn(
                f'⚠️Not enough data ({num_samples} frames). Need at least {self.min_frames}.'
            )
            self.raw_data_buffer = []
            if self.temp_dir is not None and self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    self.get_logger().error(f'❌Failed to clean temp dir: {e}')
            self.temp_dir = None
            return

        if not final and num_samples < self.chunk_size:
            return

        if self.dataset_dir is None:
            timestamp_str = time.strftime('%Y%m%d_%H%M%S')
            self.dataset_dir = self.save_dir / f'navvla_{timestamp_str}'
            self.dataset_dir.mkdir(parents=True, exist_ok=True)

        traj_name = f'traj_{self.traj_count}'
        traj_dir = self.dataset_dir / traj_name

        try:
            traj_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.get_logger().error(f'❌Failed to create directory: {e}')
            return

        positions = []
        yaws = []

        for out_index, metadata in enumerate(self.raw_data_buffer):
            pose = metadata['pose']
            positions.append([pose[0], pose[1]])
            yaws.append(pose[2])
            if self.temp_dir is not None:
                src_path = self.temp_dir / f"{metadata['frame_id']:06d}.jpg"
                dst_path = traj_dir / f'{out_index}.jpg'
                if src_path.exists():
                    shutil.move(str(src_path), str(dst_path))
                else:
                    self.get_logger().error(f'❌Missing temp image: {src_path}')
                    return

        traj_data = {
            'position': np.array(positions, dtype=np.float32),
            'yaw': np.array(yaws, dtype=np.float32),
        }

        try:
            with open(traj_dir / 'traj_data.pkl', 'wb') as f:
                pickle.dump(traj_data, f)

            self.traj_names.append(traj_name)
            with open(self.dataset_dir / 'traj_names.txt', 'w') as f:
                f.write(''.join(f'{name}\n' for name in self.traj_names))

            self.get_logger().info(f'💾Saved {num_samples} frames to: {traj_dir}')
        except Exception as e:
            self.get_logger().error(f'❌Failed to save data: {e}')
            return

        self.traj_count += 1
        self.raw_data_buffer = []
        if final:
            if self.temp_dir is not None and self.temp_dir.exists():
                try:
                    shutil.rmtree(self.temp_dir)
                except Exception as e:
                    self.get_logger().error(f'❌Failed to clean temp dir: {e}')
            self.temp_dir = None
        else:
            self._reset_temp_dir()

    def save_data(self) -> None:
        if self.frame_count == 0 and not self.traj_names:
            self.get_logger().warn('⚠️No data collected. Nothing to save.')
            return

        self._flush_buffer(final=True)

        if self.dataset_dir is not None:
            self.get_logger().info(f'Saved dataset to: {self.dataset_dir}')

        if self.temp_dir is not None and self.temp_dir.exists():
            try:
                shutil.rmtree(self.temp_dir)
                self.get_logger().info(f'🧹 Cleaned up temp dir: {self.temp_dir}')
            except Exception as e:
                self.get_logger().error(f'❌Failed to clean temp dir: {e}')

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
