from typing import List, Tuple
import numpy as np
from scipy.interpolate import CubicSpline 
from math import comb
import threading

class CurveType:
    LINEAR = "L"
    PERFECT = "P"  
    BEZIER = "B"
    CATMULL = "C"

PRECISION_BEZIER_QUANTIZATION = 0.0001
BEZIER_QUANTIZATIONSQ = PRECISION_BEZIER_QUANTIZATION * PRECISION_BEZIER_QUANTIZATION


def calculate_linear_points(start: Tuple[float, float], end: Tuple[float, float], num_points: int) -> List[Tuple[float, float]]:
    t = np.linspace(0, 1, num_points)
    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])
    return list(zip(x, y))

def calculate_perfect_circle_points(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], 
                                    num_points: int) -> List[Tuple[float, float]]:
    if is_collinear(a, b, c) or are_points_identical([a, b, c]):
        return calculate_linear_points(a, c, num_points)
    
    center, radius = find_circle_center_radius(a, b, c)
    if center is None or radius <= 0:
        return calculate_linear_points(a, c, num_points)
    
    points = np.array([a, b, c])
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    angles = np.mod(angles, 2 * np.pi)
    
    angle_diffs = np.diff(angles)
    angle_diffs = np.mod(angle_diffs, 2 * np.pi)
    angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
    
    cross_product = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    direction = -1 if cross_product < 0 else 1
    
    total_angle = np.sum(angle_diffs) * direction
    
    t = np.linspace(0, 1, num_points)
    angles = angles[0] + t * total_angle
    
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    return list(zip(x, y))

def calculate_bezier_points(
    control_points: List[Tuple[float, float]]
) -> List[Tuple[float, float]]:
    pts = control_points
    n_total = len(pts)
    if n_total == 0:
        return []

    max_n = n_total
    buf1 = np.zeros((max_n,   2), dtype=np.float64)  
    buf2 = np.zeros((2*max_n-1, 2), dtype=np.float64)

    def is_flat_enough(cp: np.ndarray) -> bool:
        diffs = cp[:-2] - 2.0*cp[1:-1] + cp[2:]
        sqlens = np.einsum('ij,ij->i', diffs, diffs)
        return np.all(sqlens <= BEZIER_QUANTIZATIONSQ)

    def subdivide(cp: np.ndarray, left: np.ndarray, right: np.ndarray):
        n = cp.shape[0]
        mid = buf1[:n]          # view of buf1
        mid[:] = cp             # copy control points in

        for i in range(n):
            left[i]            = mid[0]
            right[n - i - 1]   = mid[n - i - 1]
            # collapse one level
            mid[:n-i-1] = 0.5*(mid[:n-i-1] + mid[1:n-i])

    def approximate_bezier(cp_list: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        cp = np.array(cp_list, dtype=np.float64)
        n  = cp.shape[0]

        nonlocal buf1, buf2
        if buf1.shape[0] < n:
            buf1 = np.zeros((n, 2), dtype=np.float64)
        if buf2.shape[0] < 2*n-1:
            buf2 = np.zeros((2*n-1, 2), dtype=np.float64)

        stack: List[np.ndarray] = [cp]
        result: List[Tuple[float,float]] = []

        while stack:
            seg = stack.pop()
            m   = seg.shape[0]
            if m < 3 or is_flat_enough(seg):
                l = buf2[:2*m-1].reshape((2*m-1,2))
                r = buf1[:m].reshape((m,2))
                subdivide(seg, l, r)

                l[m:2*m-1] = r[1:m]

                result.append((float(seg[0,0]), float(seg[0,1])))
                for i in range(1, m-1):
                    idx = 2 * i
                    p   = (l[idx-1] + 2.0*l[idx] + l[idx+1]) * 0.25
                    result.append((float(p[0]), float(p[1])))
            else:
                left  = np.empty((seg.shape[0],2), dtype=np.float64)
                right = np.empty((seg.shape[0],2), dtype=np.float64)
                subdivide(seg, left, right)
                stack.append(right)
                stack.append(left)

        return result

    out: List[Tuple[float,float]] = []
    last_idx = 0
    i = 0
    while i < n_total:
        multi = (i < n_total - 2) and (pts[i] == pts[i+1])
        if multi or i == n_total - 1:
            seg = pts[last_idx:i+1]
            if len(seg) == 1:
                inter = [seg[0]]
            elif len(seg) == 2:
                inter = [seg[0], seg[1]]
            else:
                inter = approximate_bezier(seg)

            if not out or out[-1] != inter[0]:
                out.extend(inter)
            else:
                out.extend(inter[1:])

            i = i + 2 if multi else i + 1
            last_idx = i
        else:
            i += 1

    if out and out[-1] != pts[-1]:
        out.append(pts[-1])
    elif not out:
        out.append(pts[-1])

    # Ensure points are evenly distributed
    if len(out) > 2:
        total_length = 0
        for i in range(len(out)-1):
            dx = out[i+1][0] - out[i][0]
            dy = out[i+1][1] - out[i][1]
            total_length += (dx*dx + dy*dy)**0.5
        
        if total_length > 0:
            target_points = min(len(out), 100)  # Limit maximum points
            new_out = [out[0]]
            current_length = 0
            target_segment_length = total_length / (target_points - 1)
            
            for i in range(1, len(out)):
                dx = out[i][0] - out[i-1][0]
                dy = out[i][1] - out[i-1][1]
                segment_length = (dx*dx + dy*dy)**0.5
                current_length += segment_length
                
                if current_length >= target_segment_length:
                    new_out.append(out[i])
                    current_length = 0
            
            if new_out[-1] != out[-1]:
                new_out.append(out[-1])
            out = new_out

    return out

def calculate_catmull_points(control_points: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
    if len(control_points) < 4:
        return calculate_bezier_points(control_points, num_points)
    points = np.array(control_points)
    chord_lengths = np.zeros(len(points))
    for i in range(1, len(points)):
        chord_lengths[i] = chord_lengths[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    if chord_lengths[-1] > 0:
        chord_lengths = chord_lengths / chord_lengths[-1]
    
    t = np.linspace(0, 1, num_points)
    result = []
    
    for i in range(len(points) - 3):
        p0, p1, p2, p3 = points[i:i+4]
        t0, t1, t2, t3 = chord_lengths[i:i+4]
        
        def basis(t, t0, t1, t2, t3):
            t2_t1 = t2 - t1
            t3_t2 = t3 - t2
            t1_t0 = t1 - t0
            
            if t2_t1 == 0 or t3_t2 == 0 or t1_t0 == 0:
                return 0
                
            a = (t - t1) / t2_t1
            b = (t - t2) / t3_t2
            c = (t - t0) / t1_t0
            
            return np.array([
                -0.5 * c * (1 - c),
                0.5 * (1 + 2*c - 3*c*c),
                0.5 * (1 + 2*a - 3*a*a),
                -0.5 * a * (1 - a)
            ])
        
        segment_points = []
        for ti in t:
            if t1 <= ti <= t2:
                b = basis(ti, t0, t1, t2, t3)
                point = b[0] * p0 + b[1] * p1 + b[2] * p2 + b[3] * p3
                segment_points.append(tuple(point))
        
        result.extend(segment_points)
    
    if len(result) > num_points:
        indices = np.linspace(0, len(result)-1, num_points, dtype=int)
        result = [result[i] for i in indices]
    elif len(result) < num_points:
        result.extend([result[-1]] * (num_points - len(result)))
    
    return result

def is_collinear(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    points = np.array([a, b, c])
    return np.abs(np.linalg.det(np.column_stack([points, np.ones(3)]))) < 1e-10

def are_points_identical(points: List[Tuple[float, float]]) -> bool:
    if not points:
        return True
    points_array = np.array(points)
    return np.all(np.all(points_array == points_array[0], axis=1))

def find_circle_center_radius(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    try:
        points = np.array([a, b, c])
        
        mid1 = (points[0] + points[1]) / 2
        mid2 = (points[1] + points[2]) / 2
        v1 = points[1] - points[0]
        v2 = points[2] - points[1]
        
        if np.all(v1 == 0) or np.all(v2 == 0):
            return None, 0
            
        perp1 = np.array([-v1[1], v1[0]])
        perp2 = np.array([-v2[1], v2[0]])
        
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        A = np.column_stack([perp1, -perp2])
        b = mid2 - mid1
        
        try:
            center = mid1 + perp1 * np.linalg.solve(A, b)[0]
        except np.linalg.LinAlgError:
            return None, 0
            
        radius = np.linalg.norm(center - points[0])
        
        return tuple(center), radius
    except:
        return None, 0 

