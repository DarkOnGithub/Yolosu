import os
import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple
import logging
import multiprocessing as mp
from multiprocessing import Lock, Queue

import cv2
import numpy as np
from PIL import Image

from emulator.beatmap import Beatmap
from emulator.difficulty import Difficulty
from emulator.config import DanserConfig
from emulator.objects.base import HitObject
from utils.utils import osu_pixels_to_normal_coords


OBJECT_CLASS_MAP = {
    "hitcircle": "circle",
    "slider": "slider",
    "spinner": "spinner",
    "sliderball": "ball",
    "approachcircle": "approaching_circle",
    "repeatpoint": "repeat_point",
}


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


class DatasetWriter:
    """
    Records frames and hit object bounding boxes into JSON files directly in memory.
    Process-safe implementation using locks and queues.
    """

    def __init__(
        self,
        beatmap: Beatmap,
        difficulty: Difficulty,
        dataset_path: str,
        config: DanserConfig,
    ):
        self.beatmap = beatmap
        self.difficulty = difficulty
        self.dataset_path = dataset_path
        self.config = config

        # Prepare index and data structures
        self.index: Dict[str, Any] = {
            'resolution': (config.width, config.height),
            'beatmap_name': beatmap.title,
            'difficulty_name': difficulty.difficulty_name,
            'fps': config.fps,
            'total_frames': 0,
            'objects_count': {cls: 0 for cls in OBJECT_CLASS_MAP.values()},
            'frames': []
        }
        self.index['objects_count']['empty'] = 0

        self.data: Dict[str, Any] = {
            'images': [],  # base64 JPEG frames
            'objects': {cls: [] for cls in OBJECT_CLASS_MAP.values()}
        }
        self.data['objects']['empty'] = []

        self.frame_idx = 0
        
        # Add locks for thread safety
        self.index_lock = Lock()
        self.data_lock = Lock()
        
        # Create a queue for collecting results
        self.result_queue = Queue()

    def _normalize_box(self, box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, h = self.config.width, self.config.height
        x1, y1, x2, y2 = box
        nx1, ny1 = osu_pixels_to_normal_coords(x1, y1, w, h)
        nx2, ny2 = osu_pixels_to_normal_coords(x2, y2, w, h)
        return nx1 / w, ny1 / h, nx2 / w, ny2 / h

    def _encode_frame(self, frame: np.ndarray) -> str:
        # Ensure grayscale
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim != 2:
            raise ValueError(f"Invalid frame dimensions: {frame.ndim}")
            
        # Convert to PIL Image
        img = Image.fromarray(frame)
        
        # Save to buffer
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85, optimize=True)
        
        # Encode to base64
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def write_frame(self, frame: np.ndarray, visible_objects: List[HitObject], current_time: float) -> None:
        """Write a frame and its object data to the dataset."""
        if frame.ndim == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame.copy()

        frame_boxes: Dict[str, List] = {cls: [] for cls in self.data['objects']}
        frame_counts: Dict[str, int] = {cls: 0 for cls in self.index['objects_count']}

        valid = False
        radius = self.difficulty.difficulty.get_radius()

        for obj in visible_objects:
            key = OBJECT_CLASS_MAP.get(obj.__class__.__name__.lower())
            if not key:
                continue

            try:
                box = (
                    obj.get_bounding_box(radius, current_time)
                    if key == 'approaching_circle'
                    else obj.get_bounding_box(radius)
                )
                if box == (0, 0, 0, 0):
                    continue

                norm_box = self._normalize_box(box)
                frame_boxes[key].append(norm_box)
                frame_counts[key] += 1
                valid = True

                # Handle slider ball
                if key == 'slider' and getattr(obj, 'ball', None):
                    ball_box = obj.ball.get_bounding_box(radius)
                    if ball_box != (0, 0, 0, 0):
                        frame_boxes['ball'].append(self._normalize_box(ball_box))
                        frame_counts['ball'] += 1
            except Exception as e:
                logging.warning(f"Error processing object {obj.__class__.__name__}: {str(e)}")
                continue

        if not valid:
            frame_boxes['empty'].append(())
            frame_counts['empty'] = 1

        # Encode and store image
        try:
            encoded = self._encode_frame(gray_frame)
            
            # Put results in queue instead of directly modifying shared data
            self.result_queue.put({
                'frame_idx': self.frame_idx,
                'encoded_image': encoded,
                'frame_boxes': frame_boxes,
                'frame_counts': frame_counts
            })
            
            # Update frame index
            with self.index_lock:
                self.frame_idx += 1
                
        except Exception as e:
            logging.error(f"Error encoding frame: {str(e)}")
            return

    def process_queue(self):
        """Process all items in the queue and update shared data structures."""
        while not self.result_queue.empty():
            result = self.result_queue.get()
            
            with self.data_lock:
                self.data['images'].append(result['encoded_image'])
                for cls, boxes in result['frame_boxes'].items():
                    self.data['objects'][cls].append(boxes)
            
            with self.index_lock:
                self.index['frames'].append({
                    'frame_index': result['frame_idx'],
                    'objects': result['frame_counts']
                })
                self.index['total_frames'] = result['frame_idx'] + 1
                
                # Update object counts
                for cls, count in result['frame_counts'].items():
                    self.index['objects_count'][cls] += count

    def save(self) -> None:
        """Save the dataset to files with process safety."""
        # Process any remaining items in the queue
        self.process_queue()
        
        os.makedirs(self.dataset_path, exist_ok=True)

        base = f"{self.beatmap.title}_{self.difficulty.difficulty_name}"
        idx_file = os.path.join(self.dataset_path, f"{base}_index.json")
        data_file = os.path.join(self.dataset_path, f"{base}_data.json")
        
        # Use file locking when writing
        with open(idx_file, 'w') as f:
            with self.index_lock:
                json.dump(self.index, f, cls=NumpyEncoder)

        with open(data_file, 'w') as f:
            with self.data_lock:
                json.dump(self.data, f, cls=NumpyEncoder)
