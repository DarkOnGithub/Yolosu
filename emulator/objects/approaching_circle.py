from typing import Tuple

class ApproachCircle:
    def __init__(self, x: float, y: float, time: float, approach_time_ms: float):
        self.x: float = x
        self.y: float = y
        self.time: float = time
        self.appear: float = time - approach_time_ms

    def get_bounding_box(self, base_radius: float, time: float) -> Tuple[float, float, float, float]:
        """Returns bounding box as (x_min, y_min, width, height)"""
        if time < (self.appear) or time >= (self.time):
            return (0, 0, 0, 0)  

        progress: float = (time - self.appear) / (self.time - self.appear)
        progress = max(0.0, min(1.0, progress))

        scale: float = 4.0 - 3.0 * progress
        approach_radius = (base_radius * scale)
        center_x = self.x
        center_y = self.y
        
        x_min = center_x - approach_radius
        y_min = center_y - approach_radius
        width = approach_radius * 2
        height = approach_radius * 2
        return (x_min, y_min, x_min + width, y_min + height)