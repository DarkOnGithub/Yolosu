from .bezier_approximator import BezierApproximator
from .circular_arc import CircularArc
from typing import List, Tuple, Optional
import math
from enum import IntEnum
import bisect
from .linear import Linear
import cv2

def process_bezier(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out_points = []
    last_index = 0
    for i in range(len(points)):
        multi = i < len(points) - 2 and points[i] == points[i + 1]
        if multi or i == len(points) - 1:
            sub_points = points[last_index:i + 1]
            if len(sub_points) == 2:
                inter = sub_points
            else:
                approximator = BezierApproximator(sub_points)
                inter = approximator.create_bezier()

            if len(out_points) == 0 or out_points[-1] != inter[0]:
                out_points.extend(inter)
            else:
                out_points.extend(inter[1:])

            if multi:
                i += 1
            last_index = i + 1

    return out_points

def process_linear(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out_points = []
    for i in range(len(points)):
        if i < len(points) - 1 and points[i] == points[i + 1]:  
            continue
        out_points.append(points[i])
    return out_points

def approximate_circular_arc(pt1: Tuple[float, float], pt2: Tuple[float, float], pt3: Tuple[float, float], detail: float = 0.5) -> List[Tuple[float, float]]:
    arc = CircularArc(pt1, pt2, pt3)
    
    if arc.unstable:
        return [pt1, pt2, pt3]
        
    segments = int(abs(arc.total_angle * arc.r) * detail)
    points = [None] * (segments + 1)
    
    points[0] = pt1
    points[segments] = pt3
    
    for i in range(1, segments):
        points[i] = arc.point_at(i / segments)
        
    return points

def process_perfect(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
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
        self.points = points
    
def process_catmull(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    out_points = []
    
    for i in range(len(points) - 1):
        p1 = points[i - 1] if i - 1 >= 0 else points[i]
        p2 = points[i]
        p3 = points[i + 1] if i + 1 < len(points) else (
            p2[0] + (p2[0] - p1[0]),
            p2[1] + (p2[1] - p1[1])
        )
        p4 = points[i + 2] if i + 2 < len(points) else (
            p3[0] + (p3[0] - p2[0]),
            p3[1] + (p3[1] - p2[1])
        )
        
        # Approximate Catmull-Rom curve with 100 points (increased from 50)
        for t in range(100):
            t = t / 99.0
            point = (
                (-0.5 * p1[0] + 1.5 * p2[0] - 1.5 * p3[0] + 0.5 * p4[0]) * t**3 +
                (p1[0] - 2.5 * p2[0] + 2 * p3[0] - 0.5 * p4[0]) * t**2 +
                (-0.5 * p1[0] + 0.5 * p3[0]) * t +
                p2[0],
                (-0.5 * p1[1] + 1.5 * p2[1] - 1.5 * p3[1] + 0.5 * p4[1]) * t**3 +
                (p1[1] - 2.5 * p2[1] + 2 * p3[1] - 0.5 * p4[1]) * t**2 +
                (-0.5 * p1[1] + 0.5 * p3[1]) * t +
                p2[1]
            )
            out_points.append(point)
            
    return out_points

class MultiCurve:
    MIN_PART_WIDTH = 0.0001

    def __init__(self, curve_defs: List[CurveDef]):
        self.lines: List[Linear] = []
        self.points: List[Tuple[float, float]] = []
        self.sections: List[float] = []
        self.length: float = 0.0
        self.cum_length: List[float] = []
        self.first_point: Tuple[float, float] = curve_defs[0].points[0] if curve_defs else (0.0, 0.0)

        for curve_def in curve_defs:
            c_points1 = self._process_curve(curve_def, False)
            c_points2 = c_points1 if curve_def.curve_type != CurveType.CIRCULAR_ARC else self._process_curve(curve_def, True)
        
            n_lines = [None] * max(0, len(self.lines) + len(c_points1) - 1)
            n_lines[:len(self.lines)] = self.lines
            for i in range(len(c_points1) - 1):
                n_lines[len(self.lines) + i] = Linear(c_points1[i], c_points1[i + 1])
            self.lines = n_lines

            skip = 1 if self.points and self.points[-1] == c_points2[0] else 0
            n_points = [None] * (len(self.points) + len(c_points2) - skip)
            n_points[:len(self.points)] = self.points
            n_points[len(self.points):] = c_points2[skip:]
            self.points = n_points

        self.length = sum(line.get_length() for line in self.lines)

        self.cum_length = [0.0] * max(1, len(self.points))
        for i in range(len(self.points) - 1):
            self.cum_length[i + 1] = self.cum_length[i] + math.sqrt(
                (self.points[i + 1][0] - self.points[i][0])**2 + 
                (self.points[i + 1][1] - self.points[i][1])**2
            )
        
        self.sections = [0.0] * (len(self.lines) + 1)
        prev = 0.0
        for i in range(len(self.lines)):
            prev += self.lines[i].get_length()
            self.sections[i + 1] = prev
        
    def _process_curve(self, curve_def: CurveDef, lazer: bool) -> List[Tuple[float, float]]:
        if curve_def.curve_type == CurveType.CIRCULAR_ARC:
            return process_perfect(curve_def.points)
        elif curve_def.curve_type == CurveType.LINE:
            return process_linear(curve_def.points)
        elif curve_def.curve_type == CurveType.BEZIER:
            return process_bezier(curve_def.points)
        elif curve_def.curve_type == CurveType.CATMULL:
            return process_catmull(curve_def.points)
        return []

    def point_at(self, t: float) -> Tuple[float, float]:
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
        return self.cum_length[-1] if self.cum_length else 0.0

    def get_start_angle(self) -> float:
        return self.lines[0].get_start_angle() if self.lines else 0.0

    def get_end_angle(self) -> float:
        return self.lines[-1].get_end_angle() if self.lines else 0.0

    def get_lines(self) -> List[Linear]:
        return self.lines

    def _get_line_at(self, t: float) -> Linear:
        if not self.lines:
            return Linear((0.0, 0.0), (0.0, 0.0))

        desired_width = self.length * max(0.0, min(1.0, t))
        index = bisect.bisect_left(self.sections[1:], desired_width)
        index = min(index, len(self.lines) - 1)
        return self.lines[index]

    def get_start_angle_at(self, t: float) -> float:
        return self._get_line_at(t).get_start_angle() if self.lines else 0.0

    def get_end_angle_at(self, t: float) -> float:
        return self._get_line_at(t).get_end_angle() if self.lines else 0.0

