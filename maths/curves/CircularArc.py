
from typing import Tuple
import math

class CircularArc:
    def __init__(self, pt1: Tuple[float, float], pt2: Tuple[float, float], pt3: Tuple[float, float]):
        self.pt1 = pt1
        self.pt2 = pt2
        self.pt3 = pt3
        self.dir = 1
        self.unstable = False
        
        if self._is_straight_line(pt1, pt2, pt3):
            self.unstable = True
            
        d = 2 * (pt1[0] * (pt2[1] - pt3[1]) + pt2[0] * (pt3[1] - pt1[1]) + pt3[0] * (pt1[1] - pt2[1]))
        a_sq = self._len_sq(pt1)
        b_sq = self._len_sq(pt2)
        c_sq = self._len_sq(pt3)
        
        self.centre = (
            (a_sq * (pt2[1] - pt3[1]) + b_sq * (pt3[1] - pt1[1]) + c_sq * (pt1[1] - pt2[1])) / d,
            (a_sq * (pt3[0] - pt2[0]) + b_sq * (pt1[0] - pt3[0]) + c_sq * (pt2[0] - pt1[0])) / d
        )
        
        self.r = self._distance(pt1, self.centre)
        self.start_angle = self._angle(pt1, self.centre)
        
        end_angle = self._angle(pt3, self.centre)
        while end_angle < self.start_angle:
            end_angle += 2 * math.pi
            
        self.total_angle = end_angle - self.start_angle
        
        a_to_c = (pt3[1] - pt1[1], -(pt3[0] - pt1[0]))
        if self._dot(a_to_c, (pt2[0] - pt1[0], pt2[1] - pt1[1])) < 0:
            self.dir = -self.dir
            self.total_angle = 2 * math.pi - self.total_angle
            
    def _is_straight_line(self, a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
        return abs((b[1] - a[1]) * (c[0] - a[0]) - (c[1] - a[1]) * (b[0] - a[0])) < 1e-6
        
    def _len_sq(self, pt: Tuple[float, float]) -> float:
        return pt[0] * pt[0] + pt[1] * pt[1]
        
    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)
        
    def _angle(self, pt: Tuple[float, float], centre: Tuple[float, float]) -> float:
        return math.atan2(pt[1] - centre[1], pt[0] - centre[0])
        
    def _dot(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return a[0] * b[0] + a[1] * b[1]
        
    def point_at(self, t: float) -> Tuple[float, float]:
        theta = self.start_angle + self.dir * t * self.total_angle
        return (
            math.cos(theta) * self.r + self.centre[0],
            math.sin(theta) * self.r + self.centre[1]
        )
