import math
from typing import Optional
import numpy as np

class Linear:
    def __init__(self, point1: np.ndarray, point2: np.ndarray):
        self.point1 = point1
        self.point2 = point2
        self.custom_length: Optional[float] = None

    def get_length(self) -> float:
        if self.custom_length is not None:
            return self.custom_length
        return np.sqrt(np.sum((self.point2 - self.point1)**2))

    def point_at(self, t: float) -> np.ndarray:
        return self.point1 + (self.point2 - self.point1) * t

    def get_start_angle(self) -> float:
        return math.atan2(self.point2[1] - self.point1[1], self.point2[0] - self.point1[0])

    def get_end_angle(self) -> float:
        return self.get_start_angle()