import math
import numpy as np

class Bezier:
    def __init__(self, points: np.ndarray):
        self.points = points
        self.control_length = 0.0
        self.approx_length = 0.0
        
        diffs = np.diff(self.points, axis=0)
        self.control_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))
        
        self.approx_length = self.control_length
        self.calculate_length()

    def calculate_length(self):
        """Calculates the approximate length of the curve to 2 decimal points of accuracy in most cases"""
        sections = math.ceil(self.control_length)
        
        t = np.linspace(0, 1, sections + 1)
        points = np.array([self.point_at(ti) for ti in t])
        
        diffs = np.diff(points, axis=0)
        self.approx_length = np.sum(np.sqrt(np.sum(diffs**2, axis=1)))

    def point_at(self, t: float) -> np.ndarray:
        """Calculate point on Bezier curve at parameter t using Bernstein polynomials"""
        n = len(self.points) - 1
        point = np.zeros(2)
        
        for i in range(n + 1):
            b = self._bernstein(i, n, t)
            point += self.points[i] * b
            
        return point

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