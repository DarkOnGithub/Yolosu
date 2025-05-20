from .bezier_approximator import BezierApproximator
from .circular_arc import CircularArc
from typing import List, Tuple
from enum import IntEnum
import bisect
from .linear import Linear
import numpy as np

def process_bezier(points: np.ndarray) -> np.ndarray:
    out_points = []
    last_index = 0
    for i in range(len(points)):
        multi = i < len(points) - 2 and np.array_equal(points[i], points[i + 1])
        if multi or i == len(points) - 1:
            sub_points = points[last_index:i + 1]
            if len(sub_points) == 2:
                inter = sub_points
            else:
                approximator = BezierApproximator(sub_points)
                inter = np.array(approximator.create_bezier())

            if len(out_points) == 0 or not np.array_equal(out_points[-1], inter[0]):
                out_points.extend(inter)
            else:
                out_points.extend(inter[1:])

            if multi:
                i += 1
            last_index = i 

    return np.array(out_points)

def process_linear(points: List[Tuple[float, float]]) -> np.ndarray:
    if not isinstance(points, np.ndarray):
        points = np.array(points)
    mask = np.ones(len(points), dtype=bool)
    for i in range(len(points) - 1):
        if np.array_equal(points[i], points[i + 1]):
            mask[i] = False
    return points[mask]

def approximate_circular_arc(pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray, detail: float = 0.5) -> np.ndarray:
    arc = CircularArc(pt1, pt2, pt3)
    
    if arc.unstable:
        return np.array([pt1, pt2, pt3])
        
    segments = int(abs(arc.total_angle * arc.r) * detail)
    points = np.zeros((segments + 1, 2))
    
    points[0] = pt1
    points[segments] = pt3
    
    for i in range(1, segments):
        points[i] = arc.point_at(i / segments)
        
    return points

def process_perfect(points: np.ndarray) -> np.ndarray:
    if len(points) > 3:
        return process_bezier(points)
    elif len(points) < 3 or CircularArc(points[0], points[1], points[2])._is_straight_line(points[0], points[1], points[2]):
        return process_linear(points)
    else:
        return approximate_circular_arc(points[0], points[1], points[2])

class CurveType(IntEnum):
    LINE = 0
    BEZIER = 1
    CIRCULAR_ARC = 2
    CATMULL = 3

class CurveDef:
    def __init__(self, curve_type: CurveType, points: List[Tuple[float, float]]):
        self.curve_type = curve_type
        self.points = np.array(points)
    
def process_catmull(points: np.ndarray) -> np.ndarray:
    out_points = []
    
    for i in range(len(points) - 1):
        p1 = points[i - 1] if i - 1 >= 0 else points[i]
        p2 = points[i]
        p3 = points[i + 1] if i + 1 < len(points) else p2 + (p2 - p1)
        p4 = points[i + 2] if i + 2 < len(points) else p3 + (p3 - p2)
        
        t = np.linspace(0, 1, 100)
        t3 = t**3
        t2 = t**2
        
        x = (-0.5 * p1[0] + 1.5 * p2[0] - 1.5 * p3[0] + 0.5 * p4[0]) * t3 + \
            (p1[0] - 2.5 * p2[0] + 2 * p3[0] - 0.5 * p4[0]) * t2 + \
            (-0.5 * p1[0] + 0.5 * p3[0]) * t + \
            p2[0]
            
        y = (-0.5 * p1[1] + 1.5 * p2[1] - 1.5 * p3[1] + 0.5 * p4[1]) * t3 + \
            (p1[1] - 2.5 * p2[1] + 2 * p3[1] - 0.5 * p4[1]) * t2 + \
            (-0.5 * p1[1] + 0.5 * p3[1]) * t + \
            p2[1]
            
        out_points.extend(np.column_stack((x, y)))
            
    return np.array(out_points)

class MultiCurve:
    MIN_PART_WIDTH = 0.0001

    def __init__(self, curve_defs: List[CurveDef], length: float):
        self.lines: List[Linear] = []
        self.points: np.ndarray = np.array([])
        self.sections: np.ndarray = np.array([])
        self.length: float = 0.0
        self.cum_length: np.ndarray = np.array([])
        self.first_point: np.ndarray = curve_defs[0].points[0] if curve_defs else np.array([0.0, 0.0])

        for curve_def in curve_defs:
            c_points1 = self._process_curve(curve_def, False)
            self.c_points1 = c_points1
            c_points2 = c_points1 if curve_def.curve_type != CurveType.CIRCULAR_ARC else self._process_curve(curve_def, True)
        
            n_lines = [None] * max(0, len(self.lines) + len(c_points1) - 1)
            n_lines[:len(self.lines)] = self.lines
            for i in range(len(c_points1) - 1):
                n_lines[len(self.lines) + i] = Linear(c_points1[i], c_points1[i + 1])
            self.lines = n_lines

            skip = 1 if len(self.points) > 0 and np.array_equal(self.points[-1], c_points2[0]) else 0
            if len(self.points) == 0:
                self.points = c_points2
            else:
                self.points = np.vstack((self.points, c_points2[skip:]))

        self.length = length
        self.cum_length = np.zeros(max(1, len(self.points)))
        if len(self.points) > 1:
            diffs = np.diff(self.points, axis=0)
            self.cum_length[1:] = np.cumsum(np.sqrt(np.sum(diffs**2, axis=1)))
        
        self.sections = np.zeros(len(self.lines) + 1)
        prev = 0.0
        for i in range(len(self.lines)):
            prev += self.lines[i].get_length()
            self.sections[i + 1] = prev
        
    def _process_curve(self, curve_def: CurveDef, lazer: bool) -> np.ndarray:
        if curve_def.curve_type == CurveType.CIRCULAR_ARC:
            return process_perfect(curve_def.points)
        elif curve_def.curve_type == CurveType.LINE:
            return process_linear(curve_def.points)
        elif curve_def.curve_type == CurveType.BEZIER:
            return process_bezier(curve_def.points)
        elif curve_def.curve_type == CurveType.CATMULL:
            return process_catmull(curve_def.points)
        return np.array([])

    def point_at(self, t: float) -> np.ndarray:
        if not self.lines or self.length == 0:
            return self.first_point

        desired_width = self.length * max(0.0, min(1.0, t))
        index = bisect.bisect_left(self.sections[1:], desired_width)
        index = min(index, len(self.lines) - 1)

        if self.sections[index + 1] - self.sections[index] == 0:
            return self.lines[index].point1

        return self.lines[index].point_at(
            (desired_width - self.sections[index]) / 
            (self.sections[index + 1] - self.sections[index])
        )

    def get_length(self) -> float:
        return self.length

    def get_length_lazer(self) -> float:
        return self.cum_length[-1] if len(self.cum_length) > 0 else 0.0

    def get_start_angle(self) -> float:
        return self.lines[0].get_start_angle() if self.lines else 0.0

    def get_end_angle(self) -> float:
        return self.lines[-1].get_end_angle() if self.lines else 0.0

    def get_lines(self) -> List[Linear]:
        return self.lines

    def _get_line_at(self, t: float) -> Linear:
        if not self.lines:
            return Linear(np.array([0.0, 0.0]), np.array([0.0, 0.0]))

        desired_width = self.length * max(0.0, min(1.0, t))
        index = bisect.bisect_left(self.sections[1:], desired_width)
        index = min(index, len(self.lines) - 1)
        return self.lines[index]

    def get_start_angle_at(self, t: float) -> float:
        return self._get_line_at(t).get_start_angle() if self.lines else 0.0

    def get_end_angle_at(self, t: float) -> float:
        return self._get_line_at(t).get_end_angle() if self.lines else 0.0

