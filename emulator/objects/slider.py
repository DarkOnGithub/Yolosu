from typing import List, Tuple, Optional
from .base import HitObject, HitObjectType
from maths.curves.curve import CurveType, MultiCurve, CurveDef
from .approaching_circle import ApproachCircle
from .repeat_point import RepeatPoint
from functools import lru_cache

class SliderBall:
    """Represents the slider ball that follows the slider path"""
    def __init__(self, x: float, y: float, time: int):
        self.x = x
        self.y = y
        self.time = time
    
    def update_position(self, x: float, y: float, time: int):
        """Update the slider ball's position and time"""
        self.x = x
        self.y = y
        self.time = time
        
    def get_bounding_box(self, radius: float) -> Tuple[float, float, float, float]:
        """Get the bounding box of the slider ball (x1, y1, x2, y2)"""
        ball_radius = radius
        return (
            self.x - ball_radius,
            self.y - ball_radius,
            self.x + ball_radius,
            self.y + ball_radius
        )

class Slider(HitObject):
    """Represents a slider in osu!"""
    MAX_PATH_LENGTH = 100_000_000
    MAX_REPEATS = 10_000  

    def __init__(self, x: int, y: int, time: int, type: HitObjectType, 
                 hit_sound: int, approach_time: float, curve_type: str = "L",
                 control_points: List[Tuple[int, int]] = None,
                 slides: int = 1, length: float = 0.0,
                 edge_sounds: List[int] = None,
                 edge_additions: List[Tuple[int, int]] = None,
                 extras: Optional[Tuple[int, int, int, int]] = None):
        super().__init__(x, y, time, type, hit_sound, extras)

        self.approaching_circle = ApproachCircle(x, y, time, approach_time)
        self.curve_type = curve_type
        self.control_points = control_points or []
        self.slides = min(slides, self.MAX_REPEATS)
        self.length = min(length, self.MAX_PATH_LENGTH)
        self.edge_sounds = edge_sounds or []
        self.edge_additions = edge_additions or []
        self._path_points = None
        self._validate()
        self.ball = SliderBall(x=self.x, y=self.y, time=self.time)
        self.repeat_points = []
        self._duration = None
        self._multi_curve = None
        
        if self.curve_type == "P":            
            if len(self.control_points) < 2:
                self.curve_type = CurveType.LINE
            elif len(self.control_points) > 2:
                self.curve_type = CurveType.BEZIER
            else:
                self.curve_type = CurveType.CIRCULAR_ARC
                
        elif self.curve_type == "L":
            self.curve_type = CurveType.LINE
        elif self.curve_type == "B":
            self.curve_type = CurveType.BEZIER
        elif self.curve_type == "C":
            self.curve_type = CurveType.CATMULL

    def _validate(self):
        """Validate slider specific data"""
        if self.type != HitObjectType.SLIDER:
            raise ValueError(f"Invalid type for Slider: {self.type}")
        if not self.control_points:
            raise ValueError("Slider must have at least one control point")
        if self.slides < 1:
            raise ValueError(f"Slider must have at least 1 slide, got {self.slides}")
        if self.length <= 0:
            raise ValueError(f"Slider length must be positive, got {self.length}")
    
    def get_bounding_box(self, radius: int) -> Tuple[float, float, float, float]:
        """Get the bounding box of the slider including all repeat points"""
        path_points = self.calculate_path_points()
        x_coords = [p[0] for p in path_points]
        y_coords = [p[1] for p in path_points]
        
        for repeat_point in self.repeat_points:
            x_coords.append(repeat_point.x)
            y_coords.append(repeat_point.y)
        
        return (
            min(x_coords) - radius,  
            min(y_coords) - radius,  
            max(x_coords) + radius,  
            max(y_coords) + radius   
        )
        
    @lru_cache(maxsize=1)
    def calculate_path_points(self, num_points: int = 500) -> List[Tuple[float, float]]:
        """Calculate the points along the slider path with caching"""
        if not self.control_points:
            return [(self.x, self.y)]

        if self._multi_curve is None:
            curve_def = CurveDef(
                curve_type=self.curve_type,
                points=[(self.x, self.y)] + self.control_points
            )
            self._multi_curve = MultiCurve([curve_def], self.length)
            
        points = []
        for i in range(num_points):
            t = i / (num_points - 1)            
            point = self._multi_curve.point_at(t)
            points.append((float(point[0]), float(point[1])))
        return points

    def update_ball_position(self, current_time: int, duration: float):
        """Update the slider ball position based on current time"""
        if current_time < self.time:
            x, y = self.x, self.y
        elif current_time > self.time + duration:
            path_points = self.calculate_path_points()
            x, y = path_points[-1]
        else:
            path_points = self.calculate_path_points()
            total_slides = self.slides
            slide_duration = duration / total_slides
            current_slide = int((current_time - self.time) / slide_duration)
            slide_progress = ((current_time - self.time) % slide_duration) / slide_duration
            
            is_reverse = current_slide % 2 == 1
            if is_reverse:
                slide_progress = 1 - slide_progress
                
            point_index = int(slide_progress * (len(path_points) - 1))
            x, y = path_points[point_index]
            
        self.ball.update_position(float(x), float(y), current_time)
        
    @classmethod
    def from_line(cls, line: str) -> 'Slider':
        """Create a slider from a line in the .osu file"""
        parts = line.strip().split(',')
        if len(parts) < 8:
            raise ValueError(f"Invalid slider line: {line}")
            
        x = int(parts[0])
        y = int(parts[1])
        time = int(parts[2])
        
        hit_sound = int(parts[4])
        
        curve_info = parts[5].split('|')
        curve_type = curve_info[0]  
        
        control_points = []
        for i in range(1, len(curve_info)):
            if curve_info[i]:
                coords = curve_info[i].split(':')
                if len(coords) == 2:
                    x_coord, y_coord = map(int, coords)
                    control_points.append((x_coord, y_coord))
        
        slides = int(parts[6])
        length = float(parts[7])
        
        edge_sounds = []
        edge_additions = []
        
        if len(parts) > 8 and parts[8]:
            edge_sounds = list(map(int, parts[8].split('|')))
            
        if len(parts) > 9 and parts[9]:
            for addition in parts[9].split('|'):
                if addition:
                    sample_set, addition_set = map(int, addition.split(':'))
                    edge_additions.append((sample_set, addition_set))
        
        extras = None
        if len(parts) > 10:
            extras_parts = parts[10].strip().split(':')
            if len(extras_parts) >= 4:
                extras = (int(extras_parts[0]), int(extras_parts[1]), 
                          int(extras_parts[2]), int(extras_parts[3]))
                    
        return cls(
            x=x,
            y=y,
            time=time,
            type=HitObjectType.SLIDER,
            hit_sound=hit_sound,
            curve_type=curve_type,
            control_points=control_points,
            slides=slides,
            length=length,
            edge_sounds=edge_sounds,
            edge_additions=edge_additions,
            extras=extras
        )
        
    def __repr__(self) -> str:
        return f"Slider(x={self.x}, y={self.y}, time={self.time}, curve_type={self.curve_type}, slides={self.slides}, length={self.length})"

    @lru_cache(maxsize=1)
    def calculate_duration(self, slider_multiplier: float, timing_points) -> int:
        """Calculate the total duration of the slider in milliseconds with caching"""
        if self._duration is not None:
            return self._duration

        slider_velocity = 1.0
        beat_length = 1000.0
        for timing_point in timing_points:
            _time, _beat_length, _uninherited = timing_point
            if _time <= self.time:
                if _uninherited:
                    beat_length = _beat_length
                else:
                    slider_velocity = -100 / _beat_length

        slide_duration = (self.length / (slider_multiplier * 100 * slider_velocity)) * beat_length
        total_duration = int(slide_duration * self.slides)
        self.repeat_points = []
        for i in range(1, self.slides):
            repeat_time = self.time + int(slide_duration * i)
            is_reverse = i % 2 == 0
            
            path_points = self.calculate_path_points()
            if is_reverse:
                pos = path_points[0]
            else:
                pos = path_points[-1]
                
            repeat_point = RepeatPoint(
                x=int(pos[0]),
                y=int(pos[1]),
                time=repeat_time,
                type=HitObjectType.SLIDER,
                hit_sound=self.hit_sound,
                approach_time=self.approaching_circle.appear,
                is_reverse=is_reverse,
                edge_index=i,
                extras=self.extras
            )
            self.repeat_points.append(repeat_point)
            
        self._duration = total_duration
        return total_duration
