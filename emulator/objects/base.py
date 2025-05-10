from typing import Tuple, Optional
from enum import Enum

class HitObjectType(Enum):
    HIT_CIRCLE = 1
    SLIDER = 2
    SPINNER = 8

class HitObject:
    """Base class for all hit objects in osu!"""
    def __init__(self, x: int, y: int, time: int, type: HitObjectType, 
                 hit_sound: int, extras: Optional[Tuple[int, int, int, int]] = None):
        self.x = x
        self.y = y
        self.time = time
        self.type = type
        self.hit_sound = hit_sound
        self.extras = extras
        
    
    def _validate(self):
        """Validate the hit object data"""
        if not 0 <= self.x <= 512:
            raise ValueError(f"X coordinate must be between 0 and 512, got {self.x}")
        if not 0 <= self.y <= 384:
            raise ValueError(f"Y coordinate must be between 0 and 384, got {self.y}")
        if self.time < 0:
            raise ValueError(f"Time must be non-negative, got {self.time}")
    
    def get_bounding_box(self, radius: int) -> Tuple[float, float, float, float]:
        """Get the bounding box of the object (x1, y1, x2, y2)"""
        raise NotImplementedError("Subclasses must implement get_bounding_box")
            
    @classmethod
    def from_line(cls, line: str) -> 'HitObject':
        """Create a hit object from a line in the .osu file"""
        parts = line.strip().split(',')
        if len(parts) < 6:
            raise ValueError(f"Invalid hit object line: {line}")
            
        x = int(parts[0])
        y = int(parts[1])
        time = int(parts[2])
        type_value = int(parts[3])
        hit_sound = int(parts[4])
        
        extras = None
        if len(parts) > 6:
            extras = tuple(map(int, parts[6:10]))
            
        return cls(x=x, y=y, time=time, type=HitObjectType(type_value), 
                  hit_sound=hit_sound, extras=extras) 
        
    