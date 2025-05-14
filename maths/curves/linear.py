import math
from typing import Tuple, Optional

class Linear:
    def __init__(self, point1: Tuple[float, float], point2: Tuple[float, float]):
        self.point1 = point1
        self.point2 = point2
        self.custom_length: Optional[float] = None

    def get_length(self) -> float:
        if self.custom_length is not None:
            return self.custom_length
        return math.sqrt((self.point2[0] - self.point1[0])**2 + (self.point2[1] - self.point1[1])**2)

    def point_at(self, t: float) -> Tuple[float, float]:
        return (
            self.point1[0] + (self.point2[0] - self.point1[0]) * t,
            self.point1[1] + (self.point2[1] - self.point1[1]) * t
        )

    def get_start_angle(self) -> float:
        return math.atan2(self.point2[1] - self.point1[1], self.point2[0] - self.point1[0])

    def get_end_angle(self) -> float:
        return self.get_start_angle()