from __future__ import annotations

import cv2
import numpy as np
from sensor_msgs.msg import Image


def render_debug_image(image: np.ndarray, prompt: str) -> np.ndarray:
    out = image.copy()
    cv2.putText(out, prompt, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1, cv2.LINE_AA)
    return out


def to_image_msg(image: np.ndarray, stamp) -> Image:
    msg = Image()
    msg.header.stamp = stamp
    msg.height = int(image.shape[0])
    msg.width = int(image.shape[1])
    msg.encoding = "bgr8"
    msg.step = int(image.shape[1]) * 3
    msg.data = image.tobytes()
    return msg
