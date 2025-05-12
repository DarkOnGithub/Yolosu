import os
import shutil
import random
from typing import Dict, List, Optional, Set, Tuple
import logging
from pathlib import Path
import yaml
from tqdm import tqdm
from collections import defaultdict
import json
import ijson  


class DatasetBalancer:
    def __init__(self, 
                 source_datasets: List[str],
                 output_dir: str,
                 target_counts: Dict[str, int],
                 train_ratio: float = 0.8,
                 val_ratio: float = 0.1,
                 test_ratio: float = 0.1,
                 json_chunk_size: int = 1000):
        """
        Initialize the dataset balancer.
        """
        self.source_datasets = source_datasets
        self.output_dir = output_dir
        self.target_counts = target_counts
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.json_chunk_size = json_chunk_size
        
        total_ratio = train_ratio + val_ratio + test_ratio
        if not abs(total_ratio - 1.0) < 1e-6:
            raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
            
        self._create_directory_structure()
        
    def _create_directory_structure(self):
        """Create the YOLO dataset directory structure."""
        
        for split in ['train', 'val', 'test']:
            os.makedirs(os.path.join(self.output_dir, split, 'images'), exist_ok=True)
            os.makedirs(os.path.join(self.output_dir, split, 'labels'), exist_ok=True)
            
        
        yaml_content = {
            'path': self.output_dir,
            'train': 'train/images',
            'val': 'val/images',
            'test': 'test/images',
            'names': {
                0: 'circle',
                1: 'slider',
                2: 'spinner'
            }
        }
        
        with open(os.path.join(self.output_dir, 'dataset.yaml'), 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)
            
    def _get_frame_objects(self, label_path: str) -> Set[int]:
        """Get all object types present in a frame."""
        object_types = set()
        with open(label_path, 'r') as f:
            for line in f:
                obj_type = int(line.split()[0])
                object_types.add(obj_type)
        return object_types

    def _process_json_chunk(self, json_path: str, start_idx: int) -> Tuple[List[dict], int]:
        """
        Process a chunk of JSON data using streaming parser.
        """
        chunk_data = []
        current_idx = 0
        
        with open(json_path, 'rb') as f:
            parser = ijson.parse(f)
            current_obj = {}
            current_key = None
            
            for prefix, event, value in parser:
                if prefix == '' and event == 'start_map':
                    current_obj = {}
                elif prefix == '' and event == 'end_map':
                    if current_idx >= start_idx:
                        chunk_data.append(current_obj)
                        if len(chunk_data) >= self.json_chunk_size:
                            return chunk_data, current_idx + 1
                    current_idx += 1
                elif '.' not in prefix:  
                    current_key = prefix
                    current_obj[current_key] = value
                    
        return chunk_data, current_idx

    def _process_dataset_chunk(self, dataset_path: str, start_idx: int) -> Tuple[Dict[str, List[tuple]], int]:
        """
        Process a chunk of the dataset and return frame information.
        """
        frame_info = defaultdict(list)
        current_idx = start_idx
        
        json_files = [f for f in os.listdir(dataset_path) if f.endswith('.json')]
        for json_file in json_files:
            json_path = os.path.join(dataset_path, json_file)
            json_chunk, next_json_idx = self._process_json_chunk(json_path, current_idx)
            
            for frame_data in json_chunk:
                img_file = frame_data.get('image_file')
                if not img_file:
                    continue
                    
                label_file = os.path.splitext(img_file)[0] + '.txt'
                label_path = os.path.join(dataset_path, 'labels', label_file)
                
                if not os.path.exists(label_path):
                    continue
                    
                
                frame_objects = self._get_frame_objects(label_path)
                
                
                for obj_type in frame_objects:
                    if obj_type == 0:  
                        frame_info['circle'].append((os.path.join(dataset_path, 'images', img_file), label_path))
                    elif obj_type == 1:  
                        frame_info['slider'].append((os.path.join(dataset_path, 'images', img_file), label_path))
                    elif obj_type == 2:  
                        frame_info['spinner'].append((os.path.join(dataset_path, 'images', img_file), label_path))
            current_idx = next_json_idx
                
        return frame_info, current_idx

    def create_balanced_dataset(self):
        """Create a balanced dataset by processing datasets in chunks and keeping frames together."""
        selected_frames = set()  
        object_counts = {obj_type: 0 for obj_type in self.target_counts.keys()}
        
        
        for dataset_path in self.source_datasets:
            current_idx = 0
            while True:
                
                frame_info, next_idx = self._process_dataset_chunk(dataset_path, current_idx)
                if not frame_info:  
                    break
                    
                
                for obj_type, target_count in self.target_counts.items():
                    if object_counts[obj_type] >= target_count:
                        continue
                        
                    available_frames = [(img_path, label_path) for img_path, label_path in frame_info[obj_type]
                                      if img_path not in selected_frames]
                    
                    if not available_frames:
                        continue
                        
                    
                    frames_to_add = min(len(available_frames), 
                                      target_count - object_counts[obj_type])
                    selected_frames.update(img_path for img_path, _ in available_frames[:frames_to_add])
                    
                    
                    for img_path, label_path in available_frames[:frames_to_add]:
                        frame_objects = self._get_frame_objects(label_path)
                        for obj in frame_objects:
                            if obj == 0:  
                                object_counts['circle'] += 1
                            elif obj == 1:  
                                object_counts['slider'] += 1
                            elif obj == 2:  
                                object_counts['spinner'] += 1
                current_idx = next_idx
                
                
                if all(count >= target for count, target in zip(object_counts.values(), self.target_counts.values())):
                    break
        
        
        selected_frames = list(selected_frames)
        random.shuffle(selected_frames)
        
        n_frames = len(selected_frames)
        n_train = int(n_frames * self.train_ratio)
        n_val = int(n_frames * self.val_ratio)
        
        train_frames = selected_frames[:n_train]
        val_frames = selected_frames[n_train:n_train + n_val]
        test_frames = selected_frames[n_train + n_val:]
        
        
        for split, frames in [('train', train_frames), ('val', val_frames), ('test', test_frames)]:
            for img_path in tqdm(frames, desc=f"Copying files to {split}"):
                label_path = os.path.splitext(img_path)[0].replace('images', 'labels') + '.txt'
                
                
                img_filename = os.path.basename(img_path)
                shutil.copy2(img_path, os.path.join(self.output_dir, split, 'images', img_filename))
                
                
                label_filename = os.path.basename(label_path)
                shutil.copy2(label_path, os.path.join(self.output_dir, split, 'labels', label_filename))
        
        
        logging.info("Final object counts:")
        for obj_type, count in object_counts.items():
            logging.info(f"{obj_type}: {count} (target: {self.target_counts[obj_type]})")
        logging.info("Balanced dataset creation completed!") 