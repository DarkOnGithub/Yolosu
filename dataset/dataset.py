import os
import json
import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
from io import BytesIO
import base64
from .dataset_loader import DatasetLoader
from .dataset_writer import DatasetWriter
from emulator.parser import beatmap_parser
from emulator.player import Player
from emulator.config import DanserConfig
from emulator.beatmap import Beatmap
import cv2
from tqdm import tqdm
import logging
import traceback
import time

class Dataset:
    """
    A class that manages multiple smaller datasets and provides balanced object selection.
    Each dataset is split into two files:
    - index file: contains metadata and frame information
    - data file: contains images and bounding boxes
    """
    def __init__(self, dataset_folder: str, config: DanserConfig, target_counts: Dict[str, int] = None):
        """
        Initialize the dataset manager.
        
        Args:
            dataset_folder: Path to folder containing dataset files
            config: The danser configuration to use
            target_counts: Dictionary mapping object types to target counts (e.g., {'circle': 100, 'spinner': 50})
        """
        self.dataset_folder = dataset_folder
        self.config = config
        self.target_counts = target_counts or {
            'circle': 100,
            'slider': 100,
            'spinner': 50,
            'approaching_circle': 100,
        }
        
        self.dataset_loaders: Dict[str, DatasetLoader] = {}
        self.object_counts: Dict[str, int] = {k: 0 for k in self.target_counts.keys()}
        self.combined_index: List[Tuple[DatasetLoader, int]] = []
        self._load_datasets()
        self._build_combined_index()
        
    def _load_datasets(self):
        """Find and load all dataset index files."""
        index_files = [f for f in os.listdir(self.dataset_folder) if f.endswith('_index.json')]
        for file in tqdm(index_files, desc="Loading datasets"):
            file_path = os.path.join(self.dataset_folder, file)
            try:
                self.dataset_loaders[file_path] = DatasetLoader(file_path)
            except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as e:
                logging.warning(f"Warning: Could not load dataset from {file_path}: {str(e)}")
                continue

    def _is_empty_frame(self, objects: Dict[str, List[Tuple[float, float, float, float]]]) -> bool:
        """Check if a frame has no objects of interest."""
        return not any(obj_type in self.target_counts and boxes for obj_type, boxes in objects.items())

    def _build_combined_index(self):
        """Build a combined index of all frames from all datasets, avoiding duplicates."""
        self.combined_index = []
        empty_frames = []
        added_frames = set()
        
        total_frames = sum(loader.index_content['total_frames'] for loader in self.dataset_loaders.values())
        
        with tqdm(total=total_frames, desc="Building combined index") as pbar:
            for loader in self.dataset_loaders.values():
                for frame_idx in range(loader.index_content['total_frames']):
                    objects = loader.get_objects_at_frame(frame_idx)
                    frame_key = f"{loader.index_path}_{frame_idx}"
                    
                    if frame_key in added_frames:
                        continue
                        
                    if self._is_empty_frame(objects):
                        empty_frames.append((loader, frame_idx))
                    else:
                        self.combined_index.append((loader, frame_idx))
                        added_frames.add(frame_key)
                    pbar.update(1)
        
        total_frames = len(self.combined_index) + len(empty_frames)
        num_empty_frames = int(total_frames * 0.15)
        
        if empty_frames:
            random.shuffle(empty_frames)
            for loader, frame_idx in empty_frames[:num_empty_frames]:
                frame_key = f"{loader.index_path}_{frame_idx}"
                if frame_key not in added_frames:
                    self.combined_index.append((loader, frame_idx))
                    added_frames.add(frame_key)
        
        logging.info("Shuffling dataset...")
        random.shuffle(self.combined_index)
        logging.info(f"Total unique frames: {len(added_frames)}")

    def get_balanced_batch(self) -> List[Tuple[np.ndarray, Dict[str, List[Tuple[float, float, float, float]]]]]:
        """
        Get frames until target object counts are reached using the combined index.
        """
        batch = []
        max_attempts = len(self.combined_index)
        attempts = 0
        used_frames = set()
        
        while attempts < max_attempts:
            attempts += 1
            
            all_targets_reached = True
            for obj_type, target in self.target_counts.items():
                if self.object_counts[obj_type] < target:
                    all_targets_reached = False
                    break
            
            if all_targets_reached:
                break
            
            if attempts >= len(self.combined_index):
                break
                
            loader, frame_idx = self.combined_index[attempts - 1]
            
            frame_key = f"{loader.index_path}_{frame_idx}"
            if frame_key in used_frames:
                continue
                
            objects = loader.get_objects_at_frame(frame_idx)
            should_use = False

            for obj_type in objects:
                boxes = objects[obj_type]
                if obj_type in self.target_counts and len(boxes) > 0:
                    remaining = self.target_counts[obj_type] - self.object_counts[obj_type]
                    if remaining > 0:
                        should_use = True
                        self.object_counts[obj_type] += len(boxes)

            if should_use:
                frame = loader.get_frame(frame_idx)
                if frame is not None:
                    batch.append((frame, objects))
                    used_frames.add(frame_key)
        
        return batch
    
    def reset_counts(self):
        """Reset the object counts."""
        self.object_counts = {k: 0 for k in self.target_counts.keys()}
    
    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about all loaded datasets."""
        return {
            'total_datasets': len(self.dataset_loaders),
            'target_counts': self.target_counts,
            'current_counts': self.object_counts,
            'datasets': [loader.get_dataset_info() for loader in self.dataset_loaders.values()]
        }

    @classmethod
    def create_from_beatmaps(cls, beatmaps_folder: str, output_folder: str, config: DanserConfig,
                           num_beatmaps: int = 10, difficulties_per_beatmap: int = 1,
                           visualize: bool = False,
                           object_counts: Dict[str, int] = None) -> 'Dataset':
        """
        Create a dataset by randomly selecting beatmaps and difficulties.
        """
        os.makedirs(output_folder, exist_ok=True)
        
        beatmap_files = [f for f in os.listdir(beatmaps_folder) if f.endswith('.osu') or os.path.isdir(os.path.join(beatmaps_folder, f))]
        if not beatmap_files:
            raise ValueError(f"No .osu files found in {beatmaps_folder}")
        
        selected_beatmaps = random.sample(beatmap_files, min(num_beatmaps, len(beatmap_files)))
        logging.info(f"Selected beatmaps: {selected_beatmaps}")
        for beatmap_file in selected_beatmaps:
            try:
                beatmap_path = os.path.join(beatmaps_folder, beatmap_file)
                beatmap = beatmap_parser.extract_beatmap(beatmap_path, is_full_path=True)
                available_difficulties = beatmap.difficulties   
                if not available_difficulties:
                    logging.warning(f"Warning: No difficulties found in {beatmap_file}")
                    continue
                
                num_diffs = min(difficulties_per_beatmap, len(available_difficulties))
                selected_difficulties = random.sample(available_difficulties, num_diffs)
                logging.info(f"Selected difficulties: {selected_difficulties}")
                beatmap.parse_difficulties([diff.difficulty_name for diff in selected_difficulties])
                for difficulty in selected_difficulties:
                    try:
                        logging.info(f"Creating dataset for {beatmap_file} - {difficulty.difficulty_name}")
                        
                        player = Player(beatmap=beatmap, difficulty=difficulty, config=config)
                        player.play(visualize=False)
                        
                        
                    except Exception as e:
                        logging.error(f"Error processing difficulty {difficulty.difficulty_name} in {beatmap_file}: {str(e)}")
                        logging.error(traceback.format_exc())
                        continue
                        
            except Exception as e:
                logging.error(f"Error processing beatmap {beatmap_file}: {str(e)}")
                logging.error(traceback.format_exc())
                continue

        return cls(output_folder, config, object_counts)

    def export_yolo(self, output_folder: str, split_ratio: float = 0.8):
        """
        Export the dataset in YOLO format, respecting target_counts for each object type.
        Uses get_balanced_batch to ensure balanced object distribution.
        
        Args:
            output_folder: Where to save the YOLO dataset
            split_ratio: Ratio of training to validation data (default: 0.8)
        """
        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'images', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'images', 'val'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'labels', 'train'), exist_ok=True)
        os.makedirs(os.path.join(output_folder, 'labels', 'val'), exist_ok=True)
        
        
        class_names = ['circle', 'slider', 'spinner', 'approaching_circle',"ball", "repeat_point"]
        with open(os.path.join(output_folder, 'classes.txt'), 'w') as f:
            f.write('\n'.join(class_names))
        
        balanced_frames = self.get_balanced_batch()
        yolo_frames = []
        
        
        for frame, objects in balanced_frames:
            if frame is None:
                continue
                
            if len(frame.shape) == 3:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
            frame_id = len(yolo_frames)
            
            yolo_boxes = []
            for obj_type, boxes in objects.items():
                if obj_type not in class_names:
                    continue
                class_idx = class_names.index(obj_type)
                for box in boxes:
                    x1, y1, x2, y2 = box
                    x_center = (x1 + x2) / 2
                    y_center = (y1 + y2) / 2
                    width = x2 - x1
                    height = y2 - y1
                    if width > 0 and height > 0:
                        yolo_boxes.append(f"{class_idx} {x_center} {y_center} {width} {height}")
            if frame.size > 0:
                yolo_frames.append((frame, yolo_boxes, frame_id))
        
        
        random.shuffle(yolo_frames)
        split_idx = int(len(yolo_frames) * split_ratio)
        train_frames = yolo_frames[:split_idx]
        val_frames = yolo_frames[split_idx:]
        
        
        for frame, yolo_boxes, frame_id in train_frames:
            base_name = f"train_{frame_id}"
            img = Image.fromarray(frame)
            img.save(os.path.join(output_folder, 'images', 'train', f"{base_name}.jpg"))
            
            with open(os.path.join(output_folder, 'labels', 'train', f"{base_name}.txt"), 'w') as f:
                if yolo_boxes:
                    f.write('\n'.join(yolo_boxes))
        
        
        for frame, yolo_boxes, frame_id in val_frames:
            base_name = f"val_{frame_id}"
            img = Image.fromarray(frame)
            img.save(os.path.join(output_folder, 'images', 'val', f"{base_name}.jpg"))
            
            with open(os.path.join(output_folder, 'labels', 'val', f"{base_name}.txt"), 'w') as f:
                if yolo_boxes:
                    f.write('\n'.join(yolo_boxes))
        
        
        yaml_content = f"""path: {os.path.abspath(output_folder)}
train: images/train
val: images/val
nc: {len(class_names)}
names: {class_names}
"""
        with open(os.path.join(output_folder, 'dataset.yaml'), 'w') as f:
            f.write(yaml_content)
        
        logging.info(f"Exported dataset with object counts:")
        for obj_type, count in self.object_counts.items():
            logging.info(f"{obj_type}: {count}/{self.target_counts[obj_type]}")
        
    @staticmethod
    def create_visualization_video(yolo_dataset_path: str, 
                                 output_path: str, 
                                 fps: int = 30,
                                 class_colors: Dict[str, Tuple[int, int, int]] = None):
        """
        Create a video visualization of a YOLO dataset with colored bounding boxes.
        """
        
        classes_file = os.path.join(yolo_dataset_path, 'classes.txt')
        if not os.path.exists(classes_file):
            logging.error(f"Classes file not found at {classes_file}")
            return
            
        with open(classes_file, 'r') as f:
            class_names = [line.strip() for line in f.readlines()]

        
        if class_colors is None:
            default_colors = [
                (0, 255, 0),    
                (255, 0, 0),    
                (0, 0, 255),    
                (255, 255, 0),  
                (255, 0, 255)   
            ]
            class_colors = {name: default_colors[i % len(default_colors)] 
                          for i, name in enumerate(class_names)}

        
        image_files = []
        for root, _, files in os.walk(os.path.join(yolo_dataset_path, 'images')):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):
                    image_files.append(os.path.join(root, file))
        
        if not image_files:
            logging.error("No images found in the dataset")
            return

        
        image_files.sort()

        
        first_image = cv2.imread(image_files[0])
        if first_image is None:
            logging.error("Could not read first image")
            return

        height, width = first_image.shape[:2]
        
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        logging.info(f"Creating visualization video with {len(image_files)} images...")
        
        for img_path in tqdm(image_files, desc="Creating visualization"):
            
            frame = cv2.imread(img_path)
            if frame is None:
                continue

            
            label_path = img_path.replace('images', 'labels').replace('.jpg', '.txt').replace('.jpeg', '.txt').replace('.png', '.txt')
            
            if os.path.exists(label_path):
                with open(label_path, 'r') as f:
                    for line in f:
                        class_id, x_center, y_center, width, height = map(float, line.strip().split())
                        x1 = int((x_center - width/2) * frame.shape[1])
                        y1 = int((y_center - height/2) * frame.shape[0])
                        x2 = int((x_center + width/2) * frame.shape[1])
                        y2 = int((y_center + height/2) * frame.shape[0])
                        
                        class_name = class_names[int(class_id)]
                        color = class_colors[class_name]
                        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
                        cv2.putText(frame, class_name, (x1, y1 - 10), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            frame_num = os.path.basename(img_path).split('.')[0]
            cv2.putText(frame, f"Frame: {frame_num}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            video_writer.write(frame)

        video_writer.release()
        logging.info(f"Visualization video saved to {output_path}")
        