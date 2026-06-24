#!/usr/bin/env python3

from __future__ import annotations
import queue
import threading

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import pyttsx3


class Voice_Guidance(Node):
    def __init__(self) -> None:
        super().__init__("voice_guidance")

        self._last_spoken = None
        self._queue = queue.Queue()

        self._worker = threading.Thread(target=self._tts_loop, daemon=True)
        self._worker.start()

        self.create_subscription(String, "/prompt", self.prompt_callback, 10)
        self.get_logger().info("voice_guidance ready")

    def prompt_callback(self, msg: String) -> None:
        text = msg.data.strip()
        if  not text or text == "No language instruction":
            return
        
        if text == self._last_spoken:
            return
        
        self._last_spoken = text
        self._queue.put(text)

    def _tts_loop(self) -> None:
        engine = pyttsx3.init()
        engine.setProperty("rate", 80)
        engine.setProperty("volume", 1.0)

        while True:
            text = self._queue.get()
            engine.say(text)
            engine.runAndWait()

def main(args=None) -> None:
    rclpy.init(args=args)
    node = Voice_Guidance()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ =="__main__":
    main()
