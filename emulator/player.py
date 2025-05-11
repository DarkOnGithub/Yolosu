import cv2
import numpy as np
from typing import List, Optional
import os
import subprocess
import logging
import json
from .objects.base import HitObject
from .objects.slider import Slider
from .objects.spinner import Spinner
from emulator.difficulty import Difficulty
from utils.utils import osu_pixels_to_normal_coords
from .objects.slider import CurveType
from .config import DanserConfig
from .beatmap import Beatmap

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
        start_time = first_hit_time - self.approach_time
        if start_time <= 0.01:
            start_time = -self.approach_time
        self.start_time = start_time - 1000
                
    def _generate_video(self) -> str:
        """Generate video using danser"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        beatmap_name = self.beatmap.title
        output_name = f"{beatmap_name}_{self.difficulty.metadata.version}.mp4"
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
        settings['Recording']['custom']["CustomOptions"] = "-an"  
        
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
            out = subprocess.run(command, shell=True, check=True)
            if out.returncode == 0:
                logging.info(f"Video generated successfully: {output_path}")
                return output_path
            else:
                raise RuntimeError(f"Failed to generate video using danser: {out.returncode}")
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
                    self.difficulty.timing_points.points
                )
            elif isinstance(obj, Spinner):
                hit_end_time = obj.end_time
            
            if (visibility_start <= frame_end_time and
                hit_end_time >= current_time):
                visible_objects.append(obj)
                
            if (obj.time - self.approach_time) > frame_end_time:
                break
             
        return visible_objects
    
    def draw_bounding_box(self, frame: np.ndarray, obj: HitObject, alpha: float = 0.5):
        """Draw a bounding box for a hit object on the frame"""
        x1, y1, x2, y2 = obj.get_bounding_box(self.radius)
        x1, y1 = osu_pixels_to_normal_coords(x1, y1, self.resolution_width, self.resolution_height)
        x2, y2 = osu_pixels_to_normal_coords(x2, y2, self.resolution_width, self.resolution_height)
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        
        overlay = frame.copy()
        label = obj.__class__.__name__
        
        color = (0, 255, 0)
        if isinstance(obj, Slider):
            if obj.curve_type == CurveType.PERFECT:
                label += " Perfect"
                color = (0, 0, 255)
            elif obj.curve_type == CurveType.BEZIER:
                label += " Bezier"
                color = (255, 0, 0)
            elif obj.curve_type == CurveType.LINEAR:
                label += " Linear"
                color = (0, 255, 0)
            elif obj.curve_type == CurveType.CATMULL:
                label += " Catmull"
                color = (255, 255, 0)
            
            control_points = [(obj.x, obj.y)] + obj.control_points
            for i, (cx, cy) in enumerate(control_points):
                cx, cy = osu_pixels_to_normal_coords(cx, cy, self.resolution_width, self.resolution_height)
                cx, cy = int(cx), int(cy)
                cv2.circle(overlay, (cx, cy), 5, (255, 0, 255), -1)  # Purple for control points
                cv2.putText(overlay, f"CP{i}", (cx+5, cy+5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
            
            # Draw curve points
            path_points = obj.calculate_path_points(100)  # Use fewer points for visualization
            for i, (px, py) in enumerate(path_points):
                px, py = osu_pixels_to_normal_coords(px, py, self.resolution_width, self.resolution_height)
                px, py = int(px), int(py)
                cv2.circle(overlay, (px, py), 2, (0, 255, 255), -1)  # Yellow for curve points
                
                # Draw lines between curve points
                if i > 0:
                    prev_px, prev_py = osu_pixels_to_normal_coords(path_points[i-1][0], path_points[i-1][1], 
                                                                 self.resolution_width, self.resolution_height)
                    prev_px, prev_py = int(prev_px), int(prev_py)
                    cv2.line(overlay, (prev_px, prev_py), (px, py), (0, 255, 255), 1)
            
            current_time = self.start_time + int(self.cap.get(cv2.CAP_PROP_POS_FRAMES) * 1000 / self.fps)
            if current_time >= obj.time and obj.ball:
                duration = obj.calculate_duration(
                    self.difficulty.difficulty.slider_multiplier,
                    self.difficulty.timing_points.points
                )
                obj.update_ball_position(current_time, duration)
                
                ball_x1, ball_y1, ball_x2, ball_y2 = obj.ball.get_bounding_box(self.radius)
                ball_x1, ball_y1 = osu_pixels_to_normal_coords(ball_x1, ball_y1, self.resolution_width, self.resolution_height)
                ball_x2, ball_y2 = osu_pixels_to_normal_coords(ball_x2, ball_y2, self.resolution_width, self.resolution_height)
                ball_x1, ball_y1, ball_x2, ball_y2 = map(int, [ball_x1, ball_y1, ball_x2, ball_y2])
                
                cv2.rectangle(overlay, (ball_x1, ball_y1), (ball_x2, ball_y2), (255, 255, 255), 2)
                cv2.putText(overlay, "Ball", (ball_x1, ball_y1-10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, 2)
        cv2.putText(overlay, label, (x1, y1-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def play(self):
        """Play the video with bounding boxes and player controls"""
        current_frame = 0
        paused = False
        playback_speed = 1.0

        while True:
            if not paused:
                ret, frame = self.cap.read()
                if not ret:
                    break
                current_time = self.start_time + int(current_frame * 1000 / self.fps)
                visible_objects = self.get_current_objects(current_time)
                for obj in visible_objects:
                    self.draw_bounding_box(frame, obj)

                info_text = f"Speed: {playback_speed:.1f}x | Frame: {current_frame}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                
                cv2.imshow('Osu! Gameplay', frame)
                current_frame += 1
            
            key = cv2.waitKey(int(1000 / (self.fps * playback_speed))) & 0xFF
            
            if key == ord('q'):  
                break
            elif key == ord(' '):  
                paused = not paused
            elif key == ord('+'):  
                playback_speed = min(playback_speed + 0.5, 4.0)
            elif key == ord('-'):  
                playback_speed = max(playback_speed - 0.5, 0.25)
            elif key == ord('f'):  
                if paused:
                    ret, frame = self.cap.read()
                    if ret:
                        current_time = self.start_time + int(current_frame * 1000 / self.fps)
                        visible_objects = self.get_current_objects(current_time)
                        for obj in visible_objects:
                            self.draw_bounding_box(frame, obj)
                        cv2.imshow('Osu! Gameplay', frame)
                        current_frame += 1
            elif key == ord('b'):  
                if paused and current_frame > 0:
                    current_frame -= 1
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
                    ret, frame = self.cap.read()
                    if ret:
                        current_time = self.start_time + int(current_frame * 1000 / self.fps)
                        visible_objects = self.get_current_objects(current_time)
                        for obj in visible_objects:
                            self.draw_bounding_box(frame, obj)
                        cv2.imshow('Osu! Gameplay', frame)
                        
        self.cap.release()
        cv2.destroyAllWindows()
        
    def __del__(self):
        """Cleanup when object is destroyed"""
        if hasattr(self, 'cap'):
            self.cap.release() 