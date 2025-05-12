import numpy as np
from typing import List, Dict, Any
from emulator.objects.base import HitObject
from emulator.beatmap import Beatmap
from emulator.difficulty import Difficulty
from PIL import Image
import base64
from io import BytesIO

OBJECT_CLASS_MAP = {
    "hitcircle": "circle",
    "slider": "slider",
    "spinner": "spinner",
    "ball": "ball",
}

class DatasetWriter:
    def __init__(self, beatmap: Beatmap, difficulty: Difficulty, dataset_path: str):
        self.dataset_path = dataset_path
        self.beatmap = beatmap
        self.difficulty = difficulty
        
        self.content: Dict[str, Any] = {
            "beatmap_name": beatmap.title,
            "difficulty_name": difficulty.difficulty_name,
            "objects_count": {
                "circle": 0,
                "slider": 0,
                "spinner": 0,
                "ball": 0,
                "empty": 0,
            },
            "images": {},
            "objects": {
                "circle": {},
                "slider": {},
                "spinner": {},
                "ball": {},
                "empty": [],
            }
        }

    def encode_image_to_base64_png(self, frame: np.ndarray) -> str:
        """Compress a NumPy image array as JPEG and return base64 string."""
        pil_image = Image.fromarray(frame)
        buffer = BytesIO()
        pil_image.save(buffer, format="JPEG", quality=85, optimize=True)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def write_frame(self, frame: np.ndarray, current_time: int, visible_objects: List[HitObject]):
        """Write a frame to the dataset with all visible objects and their bounding boxes"""
        self.content["images"][current_time] = self.encode_image_to_base64_png(frame)
        for obj in visible_objects:
            obj_type = OBJECT_CLASS_MAP[obj.__class__.__name__.lower()]
            bounding_box = obj.get_bounding_box(self.difficulty.difficulty.get_radius())
            self.content["objects_count"][obj_type] += 1
            
