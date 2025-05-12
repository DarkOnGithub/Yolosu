from .base import HitObject, HitObjectType
from typing import Optional, Tuple

class Spinner(HitObject):
    """Represents a spinner in osu!"""
    def __init__(self, x: int, y: int, time: int, type: HitObjectType, 
                 hit_sound: int, end_time: int = 0,
                 extras: Optional[Tuple[int, int, int, int]] = None):
        super().__init__(x, y, time, type, hit_sound, extras)
        self.end_time = end_time
        
        self._validate()
    
    def _validate(self):
        """Validate spinner specific data"""
        if self.type != HitObjectType.SPINNER:
            raise ValueError(f"Invalid type for Spinner: {self.type}")
        if self.end_time <= self.time:
            raise ValueError(f"Spinner end time must be after start time, got {self.end_time}")
    
    def get_bounding_box(self, radius: int) -> Tuple[float, float, float, float]:
        """Get the bounding box of the spinner (x1, y1, x2, y2)"""
        radius = 256 * 0.9
        center_x, center_y = 256, 192
        return (
            center_x - radius,  
            center_y - radius,  
            center_x + radius,  
            center_y + radius   
        )
            
    @classmethod
    def from_line(cls, line: str) -> 'Spinner':
        """Create a spinner from a line in the .osu file"""
        parts = line.strip().split(',')
        if len(parts) < 6:
            raise ValueError(f"Invalid spinner line: {line}")
            
        x = int(parts[0])
        y = int(parts[1])
        time = int(parts[2])
        
        hit_sound = int(parts[4])
        end_time = int(parts[5])
        
        return cls(
            x=x,
            y=y,
            time=time,
            type=HitObjectType.SPINNER,
            hit_sound=hit_sound,
            end_time=end_time
        ) 
    
    def __repr__(self) -> str:
        return f"Spinner(x={self.x}, y={self.y}, time={self.time}, end_time={self.end_time})"
