import numpy as np
import json
import base64
from io import BytesIO
from PIL import Image
import cv2
from typing import Dict, Any, List, Tuple, Optional
import time

class DatasetLoader:
    """
    Loads and plays back a dataset from separate index and data files.
    Data file is loaded lazily only when needed.
    """
    def __init__(self, index_path: str):
        self.index_path = index_path
        self.data_path = index_path.replace('_index.json', '_data.json')
        self.index_content: Dict[str, Any] = {}
        self.data_content: Optional[Dict[str, Any]] = None
        self.load_index()

    def load_index(self):
        """Load only the index file."""
        with open(self.index_path, 'r') as f:
            self.index_content = json.load(f)

    def load_data(self):
        """Lazily load the data file when needed."""
        if self.data_content is None:
            with open(self.data_path, 'r') as f:
                self.data_content = json.loads(f.read())

    def decode_base64_to_image(self, base64_string: str) -> np.ndarray:
        """Convert base64 string to NumPy image."""
        data = base64.b64decode(base64_string)
        img = Image.open(BytesIO(data))
        if img.mode != 'L':
            img = img.convert('L')
        return np.array(img)

    def get_frame(self, frame_index: int) -> Optional[np.ndarray]:
        """Return the frame at index, or None if out of range."""
        self.load_data()  
        if not (0 <= frame_index < len(self.data_content['images'])):
            return None
        return self.decode_base64_to_image(self.data_content['images'][frame_index])

    def get_objects_at_frame(self, frame_index: int) -> Dict[str, List[Tuple[float, float, float, float]]]:
        """
        Return a mapping from object types to lists of bounding boxes for the given frame.
        No extra nesting—each value is the exact list of boxes per frame.
        """
        self.load_data()  
        objs: Dict[str, List[Tuple[float, float, float, float]]] = {}
        for obj_type, frames in self.data_content['objects'].items():
            if 0 <= frame_index < len(frames):
                boxes = frames[frame_index]
                if isinstance(boxes, list):
                    objs[obj_type] = boxes
        return objs

    def get_frame_info(self, frame_index: int) -> Optional[Dict[str, Any]]:
        """Get metadata about a specific frame from the index."""
        if 0 <= frame_index < len(self.index_content['frames']):
            return self.index_content['frames'][frame_index]
        return None

    def find_frames_with_objects(self, min_counts: Dict[str, int] = None) -> List[int]:
        """
        Find frames that contain at least the specified number of each object type.
        
        Args:
            min_counts: Dictionary mapping object types to minimum counts
                       (e.g., {'circle': 2, 'slider': 1})
        
        Returns:
            List of frame indices that match the criteria
        """
        if min_counts is None:
            return list(range(len(self.index_content['frames'])))
            
        matching_frames = []
        for frame_info in self.index_content['frames']:
            frame_objects = frame_info['objects']
            matches = True
            for obj_type, min_count in min_counts.items():
                if frame_objects.get(obj_type, 0) < min_count:
                    matches = False
                    break
            if matches:
                matching_frames.append(frame_info['frame_index'])
        return matching_frames

    def play_video(self, fps: int = None, window_name: str = "Dataset Playback"):
        """Play back the video with drawn bounding boxes."""
        self.load_data()  
        fps = fps or self.index_content.get('fps', 60)
        delay = 1.0 / fps
        res_w, res_h = self.index_content['resolution']

        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, res_w, res_h)

        try:
            for idx in range(len(self.data_content['images'])):
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
                        cv2.rectangle(bgr, (x1, y1), (x2, y2), (0,255,0), 1)
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
            'resolution': self.index_content['resolution'],
            'beatmap_name': self.index_content['beatmap_name'],
            'difficulty_name': self.index_content['difficulty_name'],
            'fps': self.index_content['fps'],
            'total_frames': len(self.index_content['frames']),
            'objects_count': self.index_content['objects_count'],
        }

    def unload_data(self):
        """Unload the data file to free memory."""
        self.data_content = None
