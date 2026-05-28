#!/usr/bin/env python3
import sys
import numpy as np
import torch
import clip
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
from PIL import Image as PILImage

sys.path.insert(0, "/home/kyo/orne_ws/src/NavVLA")
from deployment.navvla.preprocess import image_to_cv2


class ClipEncoderNode(Node):
    def __init__(self):
        super().__init__("clip_encoder_node")
        model_name = self.declare_parameter("clip_model", "ViT-B/32").value
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()
        self.sub = self.create_subscription(Image, "/image_raw", self.cb, 10)
        self.pub = self.create_publisher(Float32MultiArray, "/image_feature", 10)
        self.get_logger().info(
            f"clip_encoder_node ready (model={model_name}, device={self.device})"
        )

    def cb(self, msg: Image):
        cv_img = image_to_cv2(msg, (224, 224))
        pil_img = PILImage.fromarray(cv_img[:, :, ::-1])
        tensor = self.preprocess(pil_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor).squeeze().cpu().numpy()
        out = Float32MultiArray()
        out.data = feat.astype(np.float32).tolist()
        self.pub.publish(out)


def main():
    rclpy.init()
    rclpy.spin(ClipEncoderNode())


if __name__ == "__main__":
    main()
