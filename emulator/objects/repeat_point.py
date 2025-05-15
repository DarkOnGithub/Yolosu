from typing import Tuple, Optional
from .base import HitObject, HitObjectType
from .approaching_circle import ApproachCircle
import math

class RepeatPoint(HitObject):
    """Represents a repeat point in a slider"""
    
    def __init__(self, x: int, y: int, time: int, type: HitObjectType, 
                 hit_sound: int, approach_time: float, is_reverse: bool = False,
                 edge_index: int = 0, extras: Optional[Tuple[int, int, int, int]] = None):
        super().__init__(x, y, time, type, hit_sound, extras)
        
        self.approaching_circle = ApproachCircle(x, y, time, approach_time)
        self.is_reverse = is_reverse
        self.edge_index = edge_index
        
    def get_bounding_box(self, radius: float) -> Tuple[float, float, float, float]:
        """Get the bounding box of the repeat point (x1, y1, x2, y2)"""
        return (
            self.x - radius,
            self.y - radius,
            self.x + radius,
            self.y + radius
        )
        
    def __repr__(self) -> str:
        return f"RepeatPoint(x={self.x}, y={self.y}, time={self.time}, is_reverse={self.is_reverse}, edge_index={self.edge_index})" 