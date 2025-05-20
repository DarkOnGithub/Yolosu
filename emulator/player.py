import cv2
import numpy as np
from typing import List, Optional, Tuple
import os
import subprocess
import logging
import json
from tqdm import tqdm
from .objects.base import HitObject
from .objects.slider import Slider
from .objects.spinner import Spinner
from .objects.hitcircle import HitCircle
from emulator.difficulty import Difficulty
from utils.utils import osu_pixels_to_normal_coords
from maths.curves.curve import CurveType
from .config import DanserConfig
from .beatmap import Beatmap
from dataset.dataset_writer import DatasetWriter, write_frame
from .objects.approaching_circle import ApproachCircle
from .objects.repeat_point import RepeatPoint
import multiprocessing as mp
from multiprocessing import Pool, cpu_count

_g_video_path: str
_g_fps: float
_g_start_time: int
_g_hit_objects: List  
_g_approach_time: float
_g_radius: float
_g_resolution: Tuple[int, int]
_g_slider_multiplier: float
_g_timing_points: List 
_g_dataset_writer = None
_g_shared_data = None
_g_shared_index = None
_g_result_queue = None
_g_config = None
_g_difficulty = None
_g_frame_counter = None
_g_index_lock = None

def init_worker(video_path: str, fps: float, start_time: int,
                hit_objects: List, approach_time: float,
                radius: float, resolution: Tuple[int, int],
                slider_multiplier: float, timing_points: List,
                dataset_dir: str, config, beatmap, difficulty, result_queue, frame_counter, index_lock):
    """Initialize globals for each worker process"""
    global _g_video_path, _g_fps, _g_start_time, _g_hit_objects
    global _g_approach_time, _g_radius, _g_resolution
    global _g_slider_multiplier, _g_timing_points, _g_dataset_writer
    global _g_shared_data, _g_shared_index
    global _g_result_queue, _g_config, _g_difficulty, _g_frame_counter, _g_index_lock

    _g_video_path = video_path
    _g_fps = fps
    _g_start_time = start_time
    _g_hit_objects = hit_objects
    _g_approach_time = approach_time
    _g_radius = radius
    _g_resolution = resolution
    _g_slider_multiplier = slider_multiplier
    _g_timing_points = timing_points


    frame_counter.value = 0
    index_lock = mp.Lock()

    _g_result_queue = result_queue
    _g_config = config
    _g_difficulty = difficulty
    _g_frame_counter = frame_counter
    _g_index_lock = index_lock

def is_visible(obj, current_time: int, frame_end_time: int) -> bool:
    """Helper function to determine if an object should be visible"""
    visibility_start = obj.time - _g_approach_time
    hit_end_time = obj.time
    
    if isinstance(obj, Slider):
        hit_end_time = obj.time + obj.calculate_duration(
            _g_slider_multiplier,
            _g_timing_points
        )
        
        for repeat_point in obj.repeat_points:
            repeat_visibility_start = repeat_point.time - _g_approach_time
            repeat_min_visibility_time = repeat_visibility_start + (_g_approach_time * 0.3)
            
            if (repeat_min_visibility_time <= frame_end_time and
                repeat_point.time >= current_time):
                return True
                
    elif isinstance(obj, Spinner):
        hit_end_time = obj.end_time
        
    if isinstance(obj, (HitCircle, Slider)):
        min_approach_time = obj.approaching_circle.appear + (_g_approach_time * 0.3)
        if (min_approach_time <= frame_end_time and 
            obj.approaching_circle.time >= current_time):
            return True
    
    min_visibility_time = visibility_start + (_g_approach_time * 0.3)
    
    return (min_visibility_time <= frame_end_time and
            hit_end_time >= current_time)

def update_slider_positions(visible_objects: List[HitObject], current_time: int):
    """Update slider positions for visible objects"""
    for obj in visible_objects:
        if isinstance(obj, Slider) and obj.ball and current_time >= obj.time:
            duration = obj.calculate_duration(
                _g_slider_multiplier,
                _g_timing_points
            )
            obj.update_ball_position(current_time, duration)

def process_frame_range(frame_range: Tuple[int, int]):
    """Worker: process frames in [start, end) and write via global writer"""
    start_frame, end_frame = frame_range
    cap = cv2.VideoCapture(_g_video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    ms_per_frame = 1000.0 / _g_fps

    for frame_num in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
            
        current_time = _g_start_time + int(frame_num * ms_per_frame)
        frame_end_time = current_time + int(ms_per_frame)

        visible_objects = []
        for obj in _g_hit_objects:
            visibility_start = obj.time - _g_approach_time
            hit_end_time = obj.time
            
            if isinstance(obj, Slider):
                hit_end_time = obj.time + obj.calculate_duration(
                    _g_slider_multiplier,
                    _g_timing_points
                )
                for repeat_point in obj.repeat_points:
                    repeat_visibility_start = repeat_point.time - _g_approach_time
                    repeat_min_visibility_time = repeat_visibility_start + (_g_approach_time * 0.3)
                    
                    if (repeat_min_visibility_time <= frame_end_time and
                        repeat_point.time >= current_time):
                        visible_objects.append(repeat_point)
                        
            elif isinstance(obj, Spinner):
                hit_end_time = obj.end_time
                
            if isinstance(obj, (HitCircle, Slider)):
                min_approach_time = obj.approaching_circle.appear + (_g_approach_time * 0.3)
                if (min_approach_time <= frame_end_time and 
                    obj.approaching_circle.time >= current_time):
                    approaching_circle = obj.approaching_circle
                    visible_objects.append(approaching_circle)
            
            min_visibility_time = visibility_start + (_g_approach_time * 0.3)
            
            if (min_visibility_time <= frame_end_time and
                hit_end_time >= current_time):
                visible_objects.append(obj)
                if isinstance(obj, Slider) and obj.ball and current_time >= obj.time:
                    visible_objects.append(obj.ball)

            if (obj.time - _g_approach_time) > frame_end_time:
                break

        update_slider_positions(visible_objects, current_time)
        write_frame(
            frame=frame,
            visible_objects=visible_objects,
            current_time=current_time,
            result_queue=_g_result_queue,
            config=_g_config,
            difficulty=_g_difficulty,
            frame_counter=_g_frame_counter,
            index_lock=_g_index_lock
        )
    cap.release()

class Player:
    def __init__(self, beatmap: Beatmap, difficulty: Difficulty, config: Optional[DanserConfig] = None):
        self.difficulty = difficulty
        self.beatmap = beatmap
        
        self.hit_objects = sorted(difficulty.hit_objects.objects, key=lambda x: x.time)
        
        if config is None:
            config = DanserConfig(
                title=os.path.basename(beatmap.path),
                difficulty=difficulty.name,
                record=True,
                quickstart=True
            )
        self.config = config
        
        approach_rate = difficulty.difficulty.approach_rate
        circle_size = difficulty.difficulty.circle_size
        self.approach_rate = approach_rate
        self.circle_size = circle_size
        
        self.radius = self.difficulty.difficulty.get_radius()
        self.approach_time = self.difficulty.difficulty.get_approach_time()
        
        self.video_path = self._generate_video()
        
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Failed to open video file: {self.video_path}")
            
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.ms_per_frame = 1000 / self.fps
        self.resolution_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.resolution_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        first_hit_time = self.hit_objects[0].time if self.hit_objects else 0
        start_time = first_hit_time - min(1800, self.approach_time)
        if start_time <= 0.01:
            start_time = -min(1800, self.approach_time)
        self.start_time = start_time - 1000

        self.frame_counter = mp.Value('i', 0)
        self.index_lock = mp.Lock()
        
        self.dataset_writer = DatasetWriter(
            self.beatmap, 
            self.difficulty, 
            self.config.dataset_dir, 
            self.config
        )
        self.result_queue = self.dataset_writer.result_queue
        
        self.num_processes = cpu_count()
        self.slider_multiplier = difficulty.difficulty.slider_multiplier
        self.timing_points = tuple((tp.time, tp.beat_length, tp.uninherited) 
                                 for tp in difficulty.timing_points.points)
        
    def _generate_video(self) -> str:
        """Generate video using danser"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        beatmap_name = self.beatmap.title
        output_name = f"{beatmap_name}_{self.difficulty.difficulty_name}.mp4"
        output_path = os.path.join(self.config.output_dir, output_name)
        
        if os.path.exists(output_path):
            cap = cv2.VideoCapture(output_path)
            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                cap.release()
                if width == self.config.width and height == self.config.height:
                    logging.info(f"Video already exists with same resolution: {output_path}")
                    return output_path
        
        self.config.output_name = output_name

        base_dir = os.path.dirname(os.path.dirname(__file__))

        settings_dir = os.path.join(base_dir, "settings")
        danser_settings_path = os.path.join(settings_dir, "danser_settings.json")
        default_settings_path = os.path.join(base_dir, "danser", "settings", "default.json")

        os.makedirs(settings_dir, exist_ok=True)
        
        with open(default_settings_path, 'r') as f:
            settings = json.load(f)
            
        settings["General"]["VerboseImportLogs"] = False
        settings["General"]["OsuSongsDir"] = "..\\beatmaps"
        settings["General"]["OsuSkinsDir"] = "..\\skins"
        
        settings["Audio"]["GeneralVolume"] = 0
        settings["Audio"]["MusicVolume"] = 0
        settings["Audio"]["SampleVolume"] = 0
        settings["Audio"]["IgnoreBeatmapSamples"] = True
        settings["Audio"]["IgnoreBeatmapSampleVolume"] = True
        
        settings["Recording"]["ShowFFmpegLogs"] = False
        settings['Recording']['FrameWidth'] = self.config.width
        settings['Recording']['FrameHeight'] = self.config.height
        settings['Recording']['FPS'] = self.config.fps
        settings['Recording']['custom']["CustomOptions"] = "-an -quiet -nostats"  
        
        settings['Playfield']['SeizureWarning']['Enabled'] = False
        settings['Playfield']['LeadInTime'] = 0
        settings['Playfield']['LeadInHold'] = 0
        settings['Playfield']['FadeOutTime'] = 0
        
        
        with open(danser_settings_path, 'w') as f:
            json.dump(settings, f, indent='\t')

        relative_path = os.path.join(os.path.basename(settings_dir), os.path.basename(danser_settings_path).replace(".json", ""))

        danser_exe = os.path.join(base_dir, "danser", "danser-cli.exe")
        command = f'{danser_exe} -settings="..\\..\\{relative_path}" {self.config.to_danser_args(self.beatmap, self.difficulty)}'

        try:
            logging.info(f"Generating video with command: {command}")
            process = subprocess.Popen(
                command,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )

            with tqdm(total=100, desc="Generating video") as pbar:
                last_progress = 0
                for line in process.stdout:
                    if "Progress:" in line:
                        try:
                            progress = int(line.split("Progress:")[1].split("%")[0].strip())
                            if progress > last_progress:
                                pbar.update(progress - last_progress)
                                last_progress = progress
                        except (ValueError, IndexError):
                            continue

            process.wait()
            if process.returncode == 0:
                logging.info(f"Video generated successfully: {output_path}")
                return output_path
            else:
                raise RuntimeError(f"Failed to generate video using danser: {process.returncode}")
        except subprocess.CalledProcessError as e:
            logging.error(f"Failed to generate video: {e}")
            raise RuntimeError(f"Failed to generate video using danser: {e}")

    def get_current_objects(self, current_time: int) -> List[HitObject]:
        """Get all objects that should be visible at the current time"""
        visible_objects = []
        frame_end_time = current_time + self.ms_per_frame
        
        for obj in self.hit_objects:
            visibility_start = obj.time - self.approach_time
            hit_end_time = obj.time
            
            if isinstance(obj, Slider):
                hit_end_time = obj.time + obj.calculate_duration(
                    self.difficulty.difficulty.slider_multiplier,
                    self.timing_points
                )
                for repeat_point in obj.repeat_points:
                    repeat_visibility_start = repeat_point.time - self.approach_time
                    repeat_min_visibility_time = repeat_visibility_start + (self.approach_time * 0.3)
                    
                    if (repeat_min_visibility_time <= frame_end_time and
                        repeat_point.time >= current_time):
                        visible_objects.append(repeat_point)

            elif isinstance(obj, Spinner):
                hit_end_time = obj.end_time
                
            if isinstance(obj, (HitCircle, Slider)):
                min_approach_time = obj.approaching_circle.appear + (self.approach_time * 0.3)
                if (min_approach_time <= frame_end_time and 
                    obj.approaching_circle.time >= current_time):
                    approaching_circle = obj.approaching_circle
                    visible_objects.append(approaching_circle)
            
            min_visibility_time = visibility_start + (self.approach_time * 0.3)
            
            if (min_visibility_time <= frame_end_time and
                hit_end_time >= current_time):
                visible_objects.append(obj)
                
            if (obj.time - self.approach_time) > frame_end_time:
                break
             
        return visible_objects
    
    def draw_bounding_box(self, frame: np.ndarray, obj: HitObject, alpha: float = 0.9, current_time: float = 0):
        """Draw a bounding box for a hit object on the frame"""
        if isinstance(obj, ApproachCircle):
            x1, y1, x2, y2 = obj.get_bounding_box(self.radius, current_time)
        else:
            x1, y1, x2, y2 = obj.get_bounding_box(self.radius)
        
        x1, y1 = osu_pixels_to_normal_coords(x1, y1, self.resolution_width, self.resolution_height)
        x2, y2 = osu_pixels_to_normal_coords(x2, y2, self.resolution_width, self.resolution_height)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        overlay = frame.copy()
        label = obj.__class__.__name__
        
        color = (0, 255, 0)
        if isinstance(obj, Slider):
            if obj.curve_type == CurveType.CIRCULAR_ARC:
                label += " Perfect"
                color = (0, 0, 255)
            elif obj.curve_type == CurveType.BEZIER:
                label += " Bezier"
                color = (255, 0, 0)
            elif obj.curve_type == CurveType.LINE:
                label += " Linear"
                color = (0, 255, 0)
            elif obj.curve_type == CurveType.CATMULL:
                label += " Catmull"
                color = (255, 255, 0)
  
            control_points = [(obj.x, obj.y)] + obj.control_points
            for i, (cx, cy) in enumerate(control_points):
                cx, cy = osu_pixels_to_normal_coords(cx, cy, self.resolution_width, self.resolution_height)
                cx, cy = int(cx), int(cy)
                cv2.circle(overlay, (cx, cy), 10, (0, 128, 0), -1)  
                cv2.putText(overlay, f"CP{i}", (cx+5, cy+5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
            
            for i, repeat_point in enumerate(obj.repeat_points):
                rx, ry = osu_pixels_to_normal_coords(repeat_point.x, repeat_point.y, 
                                                   self.resolution_width, self.resolution_height)
                rx, ry = int(rx), int(ry)
                repeat_color = (0, 255, 255) if repeat_point.is_reverse else (255, 255, 0)
                cv2.rectangle(overlay, (rx, ry), (rx, ry), repeat_color, 2)
                label = f"R{i+1}"
                if repeat_point.is_reverse:
                    label += "R"
                cv2.putText(overlay, label, (rx+5, ry+5),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, repeat_color, 1)
            
            path_points = obj.calculate_path_points(1000)
            for i in range(len(path_points)-1):
                p1 = path_points[i]
                p2 = path_points[i+1]
                p1x, p1y = osu_pixels_to_normal_coords(p1[0], p1[1], self.resolution_width, self.resolution_height)
                p2x, p2y = osu_pixels_to_normal_coords(p2[0], p2[1], self.resolution_width, self.resolution_height)
                cv2.line(overlay, (int(p1x), int(p1y)), (int(p2x), int(p2y)), color, 3)
            
            if obj.ball:
                ball_x1, ball_y1, ball_x2, ball_y2 = obj.ball.get_bounding_box(self.radius)
                ball_x1, ball_y1 = osu_pixels_to_normal_coords(ball_x1, ball_y1, self.resolution_width, self.resolution_height)
                ball_x2, ball_y2 = osu_pixels_to_normal_coords(ball_x2, ball_y2, self.resolution_width, self.resolution_height)
                ball_x1, ball_y1, ball_x2, ball_y2 = map(int, [ball_x1, ball_y1, ball_x2, ball_y2])

                cv2.line(overlay, (ball_x1, ball_y1), (ball_x2, ball_y2), (255, 0, 255), 4)
                cv2.line(overlay, (ball_x1, ball_y2), (ball_x2, ball_y1), (255, 0, 255), 4)
                cv2.rectangle(overlay, (ball_x1, ball_y1), (ball_x2, ball_y2), (255, 0, 255), 4)
                cv2.putText(overlay, "Ball", (ball_x1, ball_y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        elif isinstance(obj, RepeatPoint):
            label += f" {'R' if obj.is_reverse else 'F'}{obj.edge_index}"
            color = (0, 255, 255) if obj.is_reverse else (255, 255, 0)

        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, y1-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.addWeighted(overlay, alpha, frame, 1-alpha, 0, frame)
        
    
    def draw_frame_info(self, frame, playback_speed: float, current_frame: int, current_time: int):
        """Draw playback information on the frame."""
        info_text = f"Speed: {playback_speed:.1f}x | Frame: {current_frame} | Time: {current_time}"
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    def visualize_frame(self, frame, current_time: int):
        """Visualize objects on the current frame."""
        visible_objects = self.get_current_objects(current_time)
        for obj in visible_objects:
            self.draw_bounding_box(frame, obj, current_time=current_time)
        cv2.imshow('Osu! Gameplay', frame)

    def handle_playback_controls(self, key: int, paused: bool, playback_speed: float, current_frame: int) -> Tuple[bool, float, int]:
        """Handle keyboard controls for playback."""
        if key == ord('q'):  
            return False, playback_speed, current_frame
        elif key == ord(' '):  
            return not paused, playback_speed, current_frame
        elif key == ord('+'):  
            return paused, min(playback_speed + 0.5, 4.0), current_frame
        elif key == ord('-'):  
            return paused, max(playback_speed - 0.5, 0.25), current_frame
        elif key == ord('f'):  
            if paused:
                return paused, playback_speed, current_frame + 1
        elif key == ord('b'):  
            if paused and current_frame > 0:
                return paused, playback_speed, current_frame - 1
        return paused, playback_speed, current_frame

    def _play_processing(self):
        """Handle processing mode with multiprocessing"""
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        chunk_size = max(1, self.frame_count // (self.num_processes * 4))
        frame_ranges = [(i, min(i + chunk_size, self.frame_count)) 
                       for i in range(0, self.frame_count, chunk_size)]
        
        initargs = (
            self.video_path,
            self.fps,
            self.start_time,
            tuple(self.hit_objects),          
            self.approach_time,
            self.radius,
            (self.resolution_width, self.resolution_height),
            self.slider_multiplier,
            tuple(self.timing_points),          
            self.config.dataset_dir,
            self.config,
            self.beatmap,
            self.difficulty,
            self.result_queue,
            self.frame_counter,
            self.index_lock
        )
        with Pool(processes=self.num_processes, initializer=init_worker, initargs=initargs) as pool:
            for _ in tqdm(pool.imap_unordered(process_frame_range, frame_ranges), 
                         total=len(frame_ranges), desc="Processing frames"):
                pass
        
        self.cap.release()
        
        while not self.result_queue.empty():
            result = self.result_queue.get()
            if result is None:
                break
            self.dataset_writer._process_queue_worker()

    def play(self, visualize: bool = False):
        """Play the video with bounding boxes and player controls.
        When visualize=False, it runs in dataset creation mode without any window rendering."""
        if visualize:
            self._play_visualization()
        else:
            self._play_processing()
            
        self.dataset_writer.save()

    def _play_visualization(self):
        """Handle visualization mode with interactive controls"""
        current_frame = 0
        paused = False
        playback_speed = 1.0

        cv2.namedWindow('Osu! Gameplay', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Osu! Gameplay', 1920, 1080)
        
        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                
                current_time = self.start_time + int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000 / self.fps)
                visible_objects = self.get_current_objects(current_time)
                self.update_sliders(current_time - 16, visible_objects)
                
                self.draw_frame_info(frame, playback_speed, current_frame, current_time)
                self.visualize_frame(frame, current_time)
                cv2.imshow('Osu! Gameplay', frame)
                
                current_frame += 1
            
            wait_time = int(1000 / (self.fps * playback_speed))
            key = cv2.waitKey(wait_time) & 0xFF
            
            paused, playback_speed, current_frame = self.handle_playback_controls(key, paused, playback_speed, current_frame)
            
            if key == ord('q'):
                break
            
            if paused and (key == ord('f') or key == ord('b')):
                current_time = self.start_time + int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000 / self.fps)
                visible_objects = self.get_current_objects(current_time)
                self.update_sliders(current_time, visible_objects)
                
                self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                ret, frame = self.cap.read()
                if ret:
                    self.visualize_frame(frame, current_time)
                    self.draw_frame_info(frame, playback_speed, current_frame, current_time)
                    cv2.imshow('Osu! Gameplay', frame)
        
        self.cap.release()
        cv2.destroyAllWindows()

    def update_sliders(self, current_time: int, visible_objects: List[HitObject]):
        """Update slider positions for visible objects"""
        for obj in visible_objects:
            if isinstance(obj, Slider) and obj.ball and current_time >= obj.time:
                duration = obj.calculate_duration(
                    self.difficulty.difficulty.slider_multiplier,
                    self.timing_points
                )
                obj.update_ball_position(current_time, duration)

    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap'):
            self.cap.release() 