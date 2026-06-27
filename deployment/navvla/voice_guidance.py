#!/usr/bin/env python3

from __future__ import annotations
import queue
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import threading

import json
import os
import shutil
import subprocess
from ament_index_python.packages import get_package_share_directory

class Voice_Guidance(Node):
    def __init__(self) -> None:
        super().__init__("voice_guidance")

        self._last_spoken = None
        self._queue = queue.Queue()
        self._voice_dir = os.path.join(get_package_share_directory("navvla"), "voice")
        try:
            with open (os.path.join(self._voice_dir, "prompts.json")) as f:
                self._prompt_map =json.load(f)
        except (OSError, json.JSONDecodeError) as e:
            self.get_logger().error(f"failed to load prompts.json: {e}")
            self._prompt_map = {}
        self._worker = threading.Thread(target=self._play_loop, daemon=True)
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

    def _play_loop(self) -> None:
        while True:
            text = self._queue.get()
            wav = self._prompt_map.get(text)
            if wav is None:
                self.get_logger().warn(f"no wav for: {text!r}")
                continue
            self._play(os.path.join(self._voice_dir, wav))

    def _play(self, path: str) -> None:
        player = shutil.which("paplay") or shutil.which("aplay")
        if player is None:
            self.get_logger().error("no audio player (paplay/aplay)")
            return
        subprocess.run([player, path], check=False)
        
        
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
