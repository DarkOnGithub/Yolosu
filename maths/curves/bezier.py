import math
from typing import List, Tuple
import numpy as np

class Bezier:
    def __init__(self, points: List[Tuple[float, float]]):
        self.points = points
        self.control_length = 0.0
        self.approx_length = 0.0
        
        # Calculate control length
        for i in range(1, len(self.points)):
            self.control_length += math.sqrt(
                (self.points[i][0] - self.points[i-1][0])**2 + 
                (self.points[i][1] - self.points[i-1][1])**2
            )
        
        self.approx_length = self.control_length
        self.calculate_length()

    def calculate_length(self):
        """Calculates the approximate length of the curve to 2 decimal points of accuracy in most cases"""
        length = 0.0
        sections = math.ceil(self.control_length)
        
        previous = self.points[0]
        for i in range(1, int(sections) + 1):
            current = self.point_at(i / sections)
            length += math.sqrt(
                (current[0] - previous[0])**2 + 
                (current[1] - previous[1])**2
            )
            previous = current
            
        self.approx_length = length

    def point_at(self, t: float) -> Tuple[float, float]:
        """Calculate point on Bezier curve at parameter t using Bernstein polynomials"""
        n = len(self.points) - 1
        x = 0.0
        y = 0.0
        
        for i in range(n + 1):
            b = self._bernstein(i, n, t)
            x += self.points[i][0] * b
            y += self.points[i][1] * b
            
        return (x, y)

    def get_length(self) -> float:
        return self.approx_length

    def get_start_angle(self) -> float:
        if len(self.points) < 2:
            return 0.0
        next_point = self.point_at(1.0 / self.control_length)
        return math.atan2(
            next_point[1] - self.points[0][1],
            next_point[0] - self.points[0][0]
        )

    def get_end_angle(self) -> float:
        if len(self.points) < 2:
            return 0.0
        prev_point = self.point_at(1.0 - 1.0/self.control_length)
        return math.atan2(
            self.points[-1][1] - prev_point[1],
            self.points[-1][0] - prev_point[0]
        )

    @staticmethod
    def _binomial_coefficient(n: int, k: int) -> int:
        """Calculate binomial coefficient C(n,k) using multiplicative formula"""
        if k < 0 or k > n:
            return 0
            
        if k == 0 or k == n:
            return 1
            
        k = min(k, n - k)
        c = 1
        
        for i in range(1, k + 1):
            c = c * (n + 1 - i) // i
            
        return c

    def _bernstein(self, i: int, n: int, t: float) -> float:
        """Calculate Bernstein polynomial B(i,n,t)"""
        return self._binomial_coefficient(n, i) * (t ** i) * ((1.0 - t) ** (n - i)) 