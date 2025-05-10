from typing import List, Tuple
import math
import numpy as np

class CurveType:
    """Enum for curve types"""
    LINEAR = "L"
    PERFECT = "P"  
    BEZIER = "B"
    CATMULL = "C"

def calculate_linear_points(start: Tuple[float, float], end: Tuple[float, float], num_points: int) -> List[Tuple[float, float]]:
    """Calculate points for a linear curve from start to end"""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        x = start[0] + t * (end[0] - start[0])
        y = start[1] + t * (end[1] - start[1])
        points.append((x, y))
    return points

def calculate_perfect_circle_points(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], 
                                    num_points: int) -> List[Tuple[float, float]]:
    """Calculate points for a perfect circle curve passing through points a, b, and c"""
    if is_collinear(a, b, c) or are_points_identical([a, b, c]):
        return calculate_linear_points(a, c, num_points)
    
    center, radius = find_circle_center_radius(a, b, c)
    if center is None:
        return calculate_linear_points(a, c, num_points)
    
    angles = []
    for point in [a, b, c]:
        angle = math.atan2(point[1] - center[1], point[0] - center[0])
        angles.append(angle)
    
    start_angle = angles[0]
    mid_angle = angles[1]
    end_angle = angles[2]
    
    while start_angle < 0:
        start_angle += 2 * math.pi
    while mid_angle < 0:
        mid_angle += 2 * math.pi
    while end_angle < 0:
        end_angle += 2 * math.pi
    
    angle_diff1 = (mid_angle - start_angle) % (2 * math.pi)
    angle_diff2 = (end_angle - mid_angle) % (2 * math.pi)
    
    if angle_diff1 > math.pi:
        angle_diff1 = 2 * math.pi - angle_diff1
    if angle_diff2 > math.pi:
        angle_diff2 = 2 * math.pi - angle_diff2
    
    total_angle = angle_diff1 + angle_diff2
    
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        angle = start_angle + t * total_angle
        x = center[0] + radius * math.cos(angle)
        y = center[1] + radius * math.sin(angle)
        points.append((x, y))
    
    return points

def calculate_bezier_points(control_points: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
    """Calculate points for a Bezier curve with given control points"""
    if len(control_points) < 2:
        return control_points
    
    segments = []
    current_segment = [control_points[0]]
    
    for i in range(1, len(control_points)):
        if control_points[i] != control_points[i-1]:
            current_segment.append(control_points[i])
        else:
            if len(current_segment) > 1:
                segments.append(current_segment)
            current_segment = [control_points[i]]
    
    if len(current_segment) > 1:
        segments.append(current_segment)
    
    all_points = []
    for segment in segments:
        segment_points = bezier_curve(segment, num_points // len(segments))
        all_points.extend(segment_points)
    
    if len(all_points) > num_points:
        step = len(all_points) / num_points
        return [all_points[min(int(i * step), len(all_points) - 1)] for i in range(num_points)]
    
    return all_points

def calculate_catmull_points(control_points: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
    """Calculate points for a Catmull-Rom curve with given control points"""
    if len(control_points) < 4:
        return calculate_bezier_points(control_points, num_points)
    
    padded_points = [control_points[0]] + control_points + [control_points[-1]]
    
    points = []
    num_segments = len(control_points) - 1
    points_per_segment = num_points // num_segments
    
    for i in range(num_segments):
        p0 = padded_points[i]
        p1 = padded_points[i + 1]
        p2 = padded_points[i + 2]
        p3 = padded_points[i + 3]
        
        segment_points = catmull_rom_segment(p0, p1, p2, p3, points_per_segment)
        points.extend(segment_points)
    
    if len(points) > num_points:
        step = len(points) / num_points
        return [points[min(int(i * step), len(points) - 1)] for i in range(num_points)]
    
    return points

def is_collinear(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    """Check if three points are collinear"""
    return abs((b[1] - a[1]) * (c[0] - b[0]) - (b[0] - a[0]) * (c[1] - b[1])) < 1e-6

def are_points_identical(points: List[Tuple[float, float]]) -> bool:
    """Check if all points are identical"""
    if not points:
        return True
    return all(p == points[0] for p in points)

def find_circle_center_radius(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    """Find the center and radius of a circle passing through three points"""
    try:
        ab_mid = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
        bc_mid = ((b[0] + c[0]) / 2, (b[1] + c[1]) / 2)
        
        if b[0] - a[0] == 0:  
            ab_slope_perpendicular = 0
        elif b[1] - a[1] == 0:  
            ab_slope_perpendicular = float('inf')
        else:
            ab_slope = (b[1] - a[1]) / (b[0] - a[0])
            ab_slope_perpendicular = -1 / ab_slope
        
        if c[0] - b[0] == 0:  
            bc_slope_perpendicular = 0
        elif c[1] - b[1] == 0:  
            bc_slope_perpendicular = float('inf')
        else:
            bc_slope = (c[1] - b[1]) / (c[0] - b[0])
            bc_slope_perpendicular = -1 / bc_slope
        
        if ab_slope_perpendicular == float('inf'):
            center_x = ab_mid[0]
            if bc_slope_perpendicular == 0:
                center_y = bc_mid[1]
            else:
                center_y = bc_slope_perpendicular * (center_x - bc_mid[0]) + bc_mid[1]
        elif bc_slope_perpendicular == float('inf'):
            center_x = bc_mid[0]
            if ab_slope_perpendicular == 0:
                center_y = ab_mid[1]
            else:
                center_y = ab_slope_perpendicular * (center_x - ab_mid[0]) + ab_mid[1]
        elif ab_slope_perpendicular == 0:
            center_y = ab_mid[1]
            center_x = bc_mid[0] + (center_y - bc_mid[1]) / bc_slope_perpendicular if bc_slope_perpendicular != 0 else bc_mid[0]
        elif bc_slope_perpendicular == 0:
            center_y = bc_mid[1]
            center_x = ab_mid[0] + (center_y - ab_mid[1]) / ab_slope_perpendicular if ab_slope_perpendicular != 0 else ab_mid[0]
        else:
            center_x = (bc_mid[1] - ab_mid[1] + ab_slope_perpendicular * ab_mid[0] - bc_slope_perpendicular * bc_mid[0]) / (ab_slope_perpendicular - bc_slope_perpendicular)
            center_y = ab_slope_perpendicular * (center_x - ab_mid[0]) + ab_mid[1]
        
        radius = math.sqrt((center_x - a[0]) ** 2 + (center_y - a[1]) ** 2)
        
        return (center_x, center_y), radius
    except:
        return None, 0

def bezier_curve(control_points: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
    """Calculate a Bezier curve with the De Casteljau algorithm"""
    if len(control_points) < 2:
        return control_points
    
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        point = de_casteljau(control_points, t)
        points.append(point)
    
    return points

def de_casteljau(control_points: List[Tuple[float, float]], t: float) -> Tuple[float, float]:
    """De Casteljau algorithm for Bezier curve evaluation at parameter t"""
    points = control_points.copy()
    n = len(points)
    
    for r in range(1, n):
        for i in range(n - r):
            x = (1 - t) * points[i][0] + t * points[i+1][0]
            y = (1 - t) * points[i][1] + t * points[i+1][1]
            points[i] = (x, y)
    
    return points[0]

def catmull_rom_segment(p0: Tuple[float, float], p1: Tuple[float, float], 
                        p2: Tuple[float, float], p3: Tuple[float, float], 
                        num_points: int) -> List[Tuple[float, float]]:
    """Calculate a segment of Catmull-Rom curve"""
    points = []
    for i in range(num_points):
        t = i / (num_points - 1) if num_points > 1 else 0
        t2 = t * t
        t3 = t2 * t
        
        
        b0 = -0.5 * t3 + t2 - 0.5 * t
        b1 = 1.5 * t3 - 2.5 * t2 + 1.0
        b2 = -1.5 * t3 + 2.0 * t2 + 0.5 * t
        b3 = 0.5 * t3 - 0.5 * t2
        
        x = b0 * p0[0] + b1 * p1[0] + b2 * p2[0] + b3 * p3[0]
        y = b0 * p0[1] + b1 * p1[1] + b2 * p2[1] + b3 * p3[1]
        
        points.append((x, y))
    
    return points 