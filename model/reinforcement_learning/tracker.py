from ultralytics import YOLO
import threading
import dxcam
import cv2
import numpy as np
from .config import RL_Config
import time
from collections import deque
from dataclasses import dataclass
import sys
from pympler import asizeof

@dataclass(slots=True)
class TrackResult:
    epoch: int
    x1: int
    y1: int
    x2: int
    y2: int
    cls: int
    conf: float

class Tracker:
    def __init__(self, config: RL_Config) -> None:
        self.config = config
        self.recorder = dxcam.create(device_idx=config.device_idx, output_color="BGR")
        self.model = YOLO(config.weight_path)
        self.thread = threading.Thread(target=self.run)
        self.thread.daemon = True
        self.thread.start()
        self.frame_queue = deque(maxlen=100)
        
    def run(self):
        self.recorder.start(target_fps=120)
        while True:
            frame = self.recorder.get_latest_frame()
            result = self.model.track(frame, persist=True, **self.config.yolo_config, verbose=False)
            if result[0].boxes is not None and len(result[0].boxes) > 0:
                boxes = result[0].boxes
                xyxy = boxes.xyxy[0].cpu().numpy()
                cls = boxes.cls[0].item()
                conf = boxes.conf[0].item()
                track_result = TrackResult(
                    epoch=int(time.time() * 1000),
                    x1=int(xyxy[0]),
                    y1=int(xyxy[1]),
                    x2=int(xyxy[2]),
                    y2=int(xyxy[3]),
                    cls=int(cls),
                    conf=conf
                )
                self.frame_queue.append(track_result)
                print(asizeof.asizeof(self.frame_queue))