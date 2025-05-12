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

OBJECT_CLASS_MAP = {
    "hitcircle": "circle",
    "slider": "slider",
    "spinner": "spinner",
    "ball": "ball",
}

class DatasetWriter:
    """
    Records frames and hit object bounding boxes to a JSON dataset.
    """
    def __init__(self, beatmap: Beatmap, difficulty: Difficulty, dataset_path: str, config: DanserConfig):
        self.beatmap = beatmap
        self.difficulty = difficulty
        self.config = config
        self.dataset_path = dataset_path

        # Initialize content structure
        self.content: Dict[str, Any] = {
            'resolution': (config.width, config.height),
            'beatmap_name': beatmap.title,
            'difficulty_name': difficulty.difficulty_name,
            'fps': config.fps,
            'total_frames': 0,
            'objects_count': {k: 0 for k in OBJECT_CLASS_MAP.values()},
            'objects_count': {**{k: 0 for k in OBJECT_CLASS_MAP.values()}, 'empty': 0},
            'images': [],
            'objects': {k: [] for k in OBJECT_CLASS_MAP.values()},
        }
        # Ensure 'empty' key
        self.content['objects']['empty'] = []

    def normalize_box(self, box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, h = self.config.width, self.config.height
        x1, y1, x2, y2 = box
        x1, y1 = osu_pixels_to_normal_coords(x1, y1, w, h)
        x2, y2 = osu_pixels_to_normal_coords(x2, y2, w, h)
        return (x1/w, y1/h, x2/w, y2/h)

    def encode_image(self, frame: np.ndarray) -> str:
        """Convert NumPy frame to compressed JPEG base64 string."""
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def write_frame(self, frame: np.ndarray, visible_objects: List[HitObject]):
        """
        Record a frame and all visible hit object bounding boxes.
        """
        self.content['images'].append(self.encode_image(frame))
        frame_boxes: Dict[str, List[Tuple[float,float,float,float]]] = {k: [] for k in self.content['objects']}

        for obj in visible_objects:
            cls = obj.__class__.__name__.lower()
            if cls not in OBJECT_CLASS_MAP:
                continue
            key = OBJECT_CLASS_MAP[cls]
            box = obj.get_bounding_box(self.difficulty.difficulty.get_radius())

            norm = self.normalize_box(box)

            frame_boxes[key].append(norm)
            self.content['objects_count'][key] += 1
            if key == 'slider' and hasattr(obj, 'ball'):
                ball_box = obj.ball.get_bounding_box(self.difficulty.difficulty.get_radius())
                norm_ball = self.normalize_box(ball_box)
                frame_boxes['ball'].append(norm_ball)
                self.content['objects_count']['ball'] += 1

        if not any(frame_boxes[k] for k in frame_boxes if k != 'empty'):
            frame_boxes['empty'] = [()]  # placeholder
            self.content['objects_count']['empty'] += 1
        else:
            frame_boxes['empty'] = []

        # Append per-frame lists
        for k, boxes in frame_boxes.items():
            self.content['objects'][k].append(boxes)

        # Update total_frames
        self.content['total_frames'] = len(self.content['images'])

    def save(self):
        """Write JSON content to disk."""
        os.makedirs(self.dataset_path, exist_ok=True)
        fn = f"{self.beatmap.title}_{self.difficulty.difficulty_name}_dataset.json"
        path = os.path.join(self.dataset_path, fn)
        with open(path, 'w') as f:
            json.dump(self.content, f)
