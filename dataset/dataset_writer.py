import numpy as np
from typing import List, Dict, Any
from emulator.objects.base import HitObject
from emulator.beatmap import Beatmap
from emulator.difficulty import Difficulty
from PIL import Image
import base64
from io import BytesIO
from typing import Tuple
import json
from emulator.config import DanserConfig
import os
from utils.utils import osu_pixels_to_normal_coords
import cv2

OBJECT_CLASS_MAP = {
    "hitcircle": "circle",
    "slider": "slider",
    "spinner": "spinner",
    "ball": "circle",
    "approachcircle": "approaching_circle",
}

class DatasetWriter:
    """
    Records frames and hit object bounding boxes to separate index and data JSON files.
    """
    def __init__(self, beatmap: Beatmap, difficulty: Difficulty, dataset_path: str, config: DanserConfig):
        self.beatmap = beatmap
        self.difficulty = difficulty
        self.config = config
        self.dataset_path = dataset_path

        self.index_content: Dict[str, Any] = {
            'resolution': (config.width, config.height),
            'beatmap_name': beatmap.title,
            'difficulty_name': difficulty.difficulty_name,
            'fps': config.fps,
            'total_frames': 0,
            'objects_count': {k: 0 for k in OBJECT_CLASS_MAP.values()},
            'frames': [],  
        }
        self.index_content['objects_count']['empty'] = 0

        self.data_content: Dict[str, Any] = {
            'images': [],
            'objects': {k: [] for k in OBJECT_CLASS_MAP.values()},
        }

        self.data_content['objects']['empty'] = []

    def normalize_box(self, box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, h = self.config.width, self.config.height
        x1, y1, x2, y2 = box
        x1, y1 = osu_pixels_to_normal_coords(x1, y1, w, h)
        x2, y2 = osu_pixels_to_normal_coords(x2, y2, w, h)
        return (x1/w, y1/h, x2/w, y2/h)

    def encode_image(self, frame: np.ndarray) -> str:
        """Convert NumPy frame to compressed JPEG base64 string."""
        if len(frame.shape) == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def write_frame(self, frame: np.ndarray, visible_objects: List[HitObject], current_time: float):
        """
        Record a frame and all visible hit object bounding boxes.
        """
        self.data_content['images'].append(self.encode_image(frame))
        frame_boxes: Dict[str, List[Tuple[float,float,float,float]]] = {k: [] for k in self.data_content['objects']}

        frame_objects: Dict[str, int] = {k: 0 for k in OBJECT_CLASS_MAP.values()}
        frame_objects['empty'] = 0

        for obj in visible_objects:
            cls = obj.__class__.__name__.lower()
            if cls not in OBJECT_CLASS_MAP:
                continue
            key = OBJECT_CLASS_MAP[cls]
            if key == 'approaching_circle':
                box = obj.get_bounding_box(self.difficulty.difficulty.get_radius(), current_time)
                if box == (0, 0, 0, 0):
                    continue
            else:
                box = obj.get_bounding_box(self.difficulty.difficulty.get_radius())
            norm = self.normalize_box(box)
            frame_boxes[key].append(norm)
            frame_objects[key] += 1
            self.index_content['objects_count'][key] += 1
            
            
            if key == 'slider':
                ball_box = obj.ball.get_bounding_box(self.difficulty.difficulty.get_radius())
                norm_ball = self.normalize_box(ball_box)
                frame_boxes['circle'].append(norm_ball)
                frame_objects['circle'] += 1
                self.index_content['objects_count']["circle"] += 1
                
        if not any(frame_boxes[k] for k in frame_boxes if k != 'empty'):
            frame_boxes['empty'] = [()]  
            frame_objects['empty'] = 1
            self.index_content['objects_count']['empty'] += 1
        else:
            frame_boxes['empty'] = []

        self.index_content['frames'].append({
            'frame_index': len(self.data_content['images']) - 1,
            'objects': frame_objects
        })
        for k, boxes in frame_boxes.items():
            self.data_content['objects'][k].append(boxes)
        self.index_content['total_frames'] = len(self.data_content['images'])

    def save(self):
        """Write index and data JSON content to disk."""
        os.makedirs(self.dataset_path, exist_ok=True)
        base_name = f"{self.beatmap.title}_{self.difficulty.difficulty_name}"
        
        index_path = os.path.join(self.dataset_path, f"{base_name}_index.json")
        with open(index_path, 'w') as f:
            json.dump(self.index_content, f)
            
        data_path = os.path.join(self.dataset_path, f"{base_name}_data.json")
        with open(data_path, 'w') as f:
            json.dump(self.data_content, f)
