import os
import json
import base64
from io import BytesIO
from typing import Any, Dict, List, Tuple
import logging
import multiprocessing as mp
from multiprocessing import Queue
import numpy as np
from PIL import Image
import cv2
import threading
import time
from collections import deque
from queue import Empty

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


def write_frame(frame: np.ndarray, visible_objects: List[HitObject], current_time: float, result_queue: Queue, 
                config: DanserConfig, difficulty: Difficulty, frame_counter, index_lock) -> None:
    """Write a frame and its object data to the dataset queue."""
    if frame.ndim == 3:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray_frame = frame.copy()
        
    all_classes = list(OBJECT_CLASS_MAP.values()) + ['empty']
    frame_boxes: Dict[str, List] = {cls: [] for cls in all_classes}
    frame_counts: Dict[str, int] = {cls: 0 for cls in all_classes}

    valid = False
    radius = difficulty.difficulty.get_radius()

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

            w, h = config.width, config.height
            x1, y1, x2, y2 = box
            nx1, ny1 = osu_pixels_to_normal_coords(x1, y1, w, h)
            nx2, ny2 = osu_pixels_to_normal_coords(x2, y2, w, h)
            norm_box = (nx1 / w, ny1 / h, nx2 / w, ny2 / h)
            
            frame_boxes[key].append(norm_box)
            frame_counts[key] += 1
            valid = True

            if key == 'slider' and getattr(obj, 'ball', None):
                ball_box = obj.ball.get_bounding_box(radius)
                if ball_box != (0, 0, 0, 0):
                    nx1, ny1 = osu_pixels_to_normal_coords(ball_box[0], ball_box[1], w, h)
                    nx2, ny2 = osu_pixels_to_normal_coords(ball_box[2], ball_box[3], w, h)
                    frame_boxes['ball'].append((nx1 / w, ny1 / h, nx2 / w, ny2 / h))
                    frame_counts['ball'] += 1
            
        except Exception as e:
            logging.warning(f"Error processing object {obj.__class__.__name__}: {str(e)}")
            continue

    if not valid:
        frame_boxes['empty'].append(())
        frame_counts['empty'] = 1

    try:
        img = Image.fromarray(gray_frame)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85, optimize=True)
        encoded = base64.b64encode(buf.getvalue()).decode('utf-8')
        
        with index_lock:
            frame_idx = frame_counter.value
            frame_counter.value += 1
        
        result_queue.put({
            'frame_idx': frame_idx,
            'encoded_image': encoded,
            'frame_boxes': frame_boxes,
            'frame_counts': frame_counts
        })
            
    except Exception as e:
        logging.error(f"Error encoding frame: {str(e)}")
        return


class DatasetWriter:
    """
    Records frames and hit object bounding boxes into JSON files.
    Process-safe implementation using shared memory and queues.
    """

    def __init__(
        self,
        beatmap: Beatmap,
        difficulty: Difficulty,
        dataset_path: str,
        config: DanserConfig,
    ):
        self._stop_event = threading.Event()
        
        self.beatmap = beatmap
        self.difficulty = difficulty
        self.dataset_path = dataset_path
        self.config = config

        self.result_queue = Queue()
        
        all_classes = list(OBJECT_CLASS_MAP.values()) + ['empty']
        
        self.index = {
            'resolution': (config.width, config.height),
            'beatmap_name': beatmap.title,
            'difficulty_name': difficulty.difficulty_name,
            'fps': config.fps,
            'total_frames': 0,
            'objects_count': {cls: 0 for cls in all_classes},
            'frames': deque(maxlen=1000)
        }

        self.data = {
            'images': [], 
            'objects': {cls: [] for cls in all_classes}
        }
        
        self.frame_counter = mp.Value('i', 0)
        self.index_lock = mp.Lock()
        
        self.worker_thread = threading.Thread(target=self._process_queue_worker, daemon=True)
        self.worker_thread.start()

    def _process_queue_worker(self):
        """Background worker to process the result queue."""
        total_size = 0
        while not self._stop_event.is_set():
            try:
                result = self.result_queue.get(timeout=0.1)
                if result is None:  
                    break
                total_size += len(str(result))
                self.data['images'].append(result['encoded_image'])
                for cls, boxes in result['frame_boxes'].items():
                    self.data['objects'][cls].append(boxes)
                
                self.index['frames'].append({
                    'frame_index': result['frame_idx'],
                    'objects': result['frame_counts']
                })
                self.index['total_frames'] = result['frame_idx'] + 1
                
                for cls, count in result['frame_counts'].items():
                    self.index['objects_count'][cls] += count
                    
            except Empty:
                continue
            except Exception as e:
                logging.error(f"Error processing queue: {str(e)}")
                continue

    def _normalize_box(self, box: Tuple[float, float, float, float]) -> Tuple[float, float, float, float]:
        w, h = self.config.width, self.config.height
        x1, y1, x2, y2 = box
        nx1, ny1 = osu_pixels_to_normal_coords(x1, y1, w, h)
        nx2, ny2 = osu_pixels_to_normal_coords(x2, y2, w, h)
        return nx1 / w, ny1 / h, nx2 / w, ny2 / h

    def _encode_frame(self, frame: np.ndarray) -> str:
        if frame.ndim == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        elif frame.ndim != 2:
            raise ValueError(f"Invalid frame dimensions: {frame.ndim}")
            
        img = Image.fromarray(frame)
        buf = BytesIO()
        img.save(buf, format='JPEG', quality=85, optimize=True)
        return base64.b64encode(buf.getvalue()).decode('utf-8')

    def write_frame(self, frame: np.ndarray, visible_objects: List[HitObject], current_time: float) -> None:
        """Write a frame and its object data to the dataset."""
        write_frame(
            frame=frame,
            visible_objects=visible_objects,
            current_time=current_time,
            result_queue=self.result_queue,
            config=self.config,
            difficulty=self.difficulty,
            frame_counter=self.frame_counter,
            index_lock=self.index_lock
        )

    def save(self) -> None:
        """Save the dataset to files with process safety."""
        self._stop_event.set()
        
        while not self.result_queue.empty():
            time.sleep(0.1)
            
        self.result_queue.put(None)
        self.worker_thread.join()
        
        os.makedirs(self.dataset_path, exist_ok=True)
        
        base = f"{self.beatmap.title}_{self.difficulty.difficulty_name}"
        idx_file = os.path.join(self.dataset_path, f"{base}_index.json")
        data_file = os.path.join(self.dataset_path, f"{base}_data.json")
        
        index_data = {
            'resolution': self.index['resolution'],
            'beatmap_name': self.index['beatmap_name'],
            'difficulty_name': self.index['difficulty_name'],
            'fps': self.index['fps'],
            'total_frames': self.index['total_frames'],
            'objects_count': dict(self.index['objects_count']),
            'frames': list(self.index['frames'])
        }
            
        data_data = {
            'images': list(self.data['images']),
            'objects': {cls: list(boxes) for cls, boxes in self.data['objects'].items()}
        }
        
        with open(idx_file, 'w') as f:
            json.dump(index_data, f, cls=NumpyEncoder)
            f.flush()
            
        with open(data_file, 'w') as f:
            json.dump(data_data, f, cls=NumpyEncoder)
            f.flush()

    def __del__(self):
        """Cleanup resources."""
        self._stop_event.set()
        if hasattr(self, 'worker_thread') and self.worker_thread.is_alive():
            self.result_queue.put(None)
            self.worker_thread.join()
