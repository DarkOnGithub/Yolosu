from typing import Optional, Tuple
from .base import HitObject, HitObjectType
from .approaching_circle import ApproachCircle

class HitCircle(HitObject):
    """Represents a hit circle in osu!"""
    def __init__(self, x: int, y: int, time: int, type: HitObjectType, approach_time: float,
                 hit_sound: int, extras: Optional[Tuple[int, int, int, int]] = None):
        super().__init__(x, y, time, type, hit_sound, extras)
        self.approaching_circle = ApproachCircle(x, y, time, approach_time)
        if self.type != HitObjectType.HIT_CIRCLE:
            raise ValueError(f"Invalid type for HitCircle: {self.type}")
    
    def get_bounding_box(self, radius: int) -> Tuple[float, float, float, float]:
        """Get the bounding box of the hit circle (x1, y1, x2, y2)"""
        
        return (
            self.x - radius,  
            self.y - radius,  
            self.x + radius,  
            self.y + radius   
        )
            
    @classmethod
    def from_line(cls, line: str) -> 'HitCircle':
        """Create a hit circle from a line in the .osu file"""
        obj = super().from_line(line)
        return cls(
            x=obj.x,
            y=obj.y,
            time=obj.time,
            type=obj.type,
            hit_sound=obj.hit_sound,
            extras=obj.extras
        ) 
    
    def __repr__(self) -> str:
        return f"HitCircle(x={self.x}, y={self.y}, time={self.time})"
