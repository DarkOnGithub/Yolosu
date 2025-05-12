import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
import os
from typing import Dict, Any, List, Tuple, Optional
import time
class DatasetLoader:
    """
    Loads and plays back a dataset exported by DatasetWriter.
    """
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.content: Dict[str, Any] = {}
        self.load_dataset()

    def load_dataset(self):
        """Load dataset from JSON."""
        with open(self.dataset_path, 'r') as f:
            self.content = json.load(f)

    def decode_base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to NumPy image."""
        data = base64.b64decode(base64_string)
        return np.array(Image.open(BytesIO(data)))

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Return the frame at index, or None if out of range."""
        if not (0 <= frame_index < len(self.content['images'])):
            return None
        return self.decode_base64_to_image(self.content['images'][frame_index])


    def get_objects_at_frame(self, frame_index: int) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """
        Return a mapping from object types to lists of bounding boxes for the given frame.
        No extra nesting—each value is the exact list of boxes per frame.
        """
        objs: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for obj_type, frames in self.content['objects'].items():
            if 0 <= frame_index < len(frames):
                boxes = frames[frame_index]
                # Ensure valid list
                if isinstance(boxes, list):
                    objs[obj_type] = boxes
        return objs

    def play_video(self, fps: int = None, window_name: str = "Dataset Playback"):
        """Play back the video with drawn bounding boxes."""
        fps = fps or self.content.get('fps', 60)
        delay = 1.0 / fps
        res_w, res_h = self.content['resolution']

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, res_w, res_h)

        try:
            for idx in range(len(self.content['images'])):
                frame = self.get_frame(idx)
                if frame is None:
                    continue
                bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                for obj_type, boxes in self.get_objects_at_frame(idx).items():
                    if obj_type == 'empty':
                        continue
                    for box in boxes:
                        x1, y1, x2, y2 = box
                        x1, x2 = int(x1*res_w), int(x2*res_w)
                        y1, y2 = int(y1*res_h), int(y2*res_h)
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0,255,0), 2)
                        cv2.putText(bgr, obj_type, (x1, max(0,y1-5)),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 1)
                cv2.imshow(window_name, bgr)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                time.sleep(delay)
        finally:
            cv2.destroyAllWindows()

    def get_dataset_info(self) -> Dict[str, Any]:
        """Return metadata about the dataset."""
        return {
            'resolution': self.content['resolution'],
            'beatmap_name': self.content['beatmap_name'],
            'difficulty_name': self.content['difficulty_name'],
            'fps': self.content['fps'],
            'total_frames': len(self.content['images']),
            'objects_count': self.content['objects_count'],
        }
