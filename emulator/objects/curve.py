from typing import List, Tuple
import numpy as np
from scipy.interpolate import CubicSpline 
from math import comb

class CurveType:
    LINEAR = "L"
    PERFECT = "P"  
    BEZIER = "B"
    CATMULL = "C"

PRECISION_BEZIER_QUANTIZATION = 0.001
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
    control_points: List[Tuple[float, float]],
    num_points: int
) -> List[Tuple[float, float]]:
    """
    cps = np.array(control_points, dtype=float)
    n = cps.shape[0] - 1
    if n < 0 or num_points <= 0:
        return []
    if n == 0:
        return [tuple(cps[0])] * num_points
    if n == 1:
        t = np.linspace(0, 1, num_points)
        pts = np.outer(1 - t, cps[0]) + np.outer(t, cps[1])
        return [tuple(pt) for pt in pts]

    t = np.linspace(0, 1, num_points)[:, None]
    binom = np.array([comb(n, ki) for ki in range(n + 1)], dtype=float)
    k = np.arange(n + 1)
    t_pow = t ** k
    one_minus_t_pow = (1 - t) ** (n - k)
    B = binom * t_pow * one_minus_t_pow
    points = B @ cps  
    return [tuple(pt) for pt in points]
    """
    if not control_points:
        return []

    control_points_np = _to_np_arrays(control_points)
    n_cps = len(control_points_np)

    if n_cps == 0:
        return []
    if n_cps == 1:
        return [control_points[0]] * max(1, num_points) 

    output_path_points_adaptive: List[np.ndarray] = []
    to_flatten_stack: List[List[np.ndarray]] = []
    free_buffers_stack: List[List[np.ndarray]] = []

    initial_curve_segment = [p.copy() for p in control_points_np]
    to_flatten_stack.append(initial_curve_segment)

    flatness_threshold = BEZIER_QUANTIZATIONSQ

    while to_flatten_stack:
        current_segment_cps = to_flatten_stack.pop()
        if _is_flat_enough(current_segment_cps, flatness_threshold):
            _approximate_flat_segment(current_segment_cps, output_path_points_adaptive)
            free_buffers_stack.append(current_segment_cps)
        else:
            l_child_cps_buffer = [np.zeros(2, dtype=float) for _ in range(n_cps)]
            if free_buffers_stack:
                r_child_cps_buffer = free_buffers_stack.pop()
            else:
                r_child_cps_buffer = [np.zeros(2, dtype=float) for _ in range(n_cps)]
            
            _subdivide(current_segment_cps, l_child_cps_buffer, r_child_cps_buffer)
            for i in range(n_cps):
                current_segment_cps[i] = l_child_cps_buffer[i]
            
            to_flatten_stack.append(r_child_cps_buffer)
            to_flatten_stack.append(current_segment_cps)

    if n_cps > 0:
        if not output_path_points_adaptive or not np.allclose(output_path_points_adaptive[-1], control_points_np[-1]):
             output_path_points_adaptive.append(control_points_np[n_cps - 1].copy())


    if num_points == 0:
        return []
    if not output_path_points_adaptive:
        if n_cps == 1: 
            return [control_points[0]] * max(1, num_points)
        if n_cps > 1: 
            return calculate_linear_points(control_points[0], control_points[-1], num_points)
        return []

    unique_adaptive_points: List[np.ndarray] = []
    if output_path_points_adaptive:
        unique_adaptive_points.append(output_path_points_adaptive[0])
        for i in range(1, len(output_path_points_adaptive)):
            if not np.allclose(output_path_points_adaptive[i], output_path_points_adaptive[i-1], atol=1e-9): # Stricter tolerance
                unique_adaptive_points.append(output_path_points_adaptive[i])
    
    if not unique_adaptive_points: 
        return [] 
    if num_points == 1: 
        return [_to_tuples([unique_adaptive_points[0]])[0]]
    if len(unique_adaptive_points) == 1:
        return [_to_tuples([unique_adaptive_points[0]])[0]] * num_points

    path_data_np = np.array(unique_adaptive_points)
    segment_lengths = np.sqrt(np.sum(np.diff(path_data_np, axis=0)**2, axis=1))
    cumulative_distances = np.zeros(len(path_data_np))
    cumulative_distances[1:] = np.cumsum(segment_lengths)

    if cumulative_distances[-1] < 1e-9: 
        return [_to_tuples([path_data_np[0]])[0]] * num_points

    u_parameter_orig = cumulative_distances / cumulative_distances[-1]
    u_parameter_new = np.linspace(0, 1, num_points)
    
    valid_indices = np.where(np.concatenate(([True], np.diff(u_parameter_orig) > 1e-9)))[0]
    if len(valid_indices) < 2: 
        if len(u_parameter_orig) >=2 and np.allclose(u_parameter_orig[0], u_parameter_orig[-1]): 
            return [_to_tuples([path_data_np[0]])[0]] * num_points
        return [_to_tuples([path_data_np[0]])[0]] * num_points


    u_parameter_orig_mono = u_parameter_orig[valid_indices]
    path_data_np_mono = path_data_np[valid_indices]

    if len(path_data_np_mono) >= 4 and num_points > 2 : 
        try:
            cs_x = CubicSpline(u_parameter_orig_mono, path_data_np_mono[:, 0])
            cs_y = CubicSpline(u_parameter_orig_mono, path_data_np_mono[:, 1])
            x_coords_new = cs_x(u_parameter_new)
            y_coords_new = cs_y(u_parameter_new)
        except ValueError: 
            x_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 0])
            y_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 1])

    else: 
        x_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 0])
        y_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 1])
            
    return list(zip(x_coords_new, y_coords_new))


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

def _to_np_arrays(points_tuples: List[Tuple[float, float]]) -> List[np.ndarray]:
    return [np.array(p, dtype=float) for p in points_tuples]

def _to_tuples(points_np: List[np.ndarray]) -> List[Tuple[float, float]]:
    return [tuple(p) for p in points_np]

def _is_flat_enough(control_points_np: List[np.ndarray], flatness_sq_threshold: float) -> bool:
    if len(control_points_np) < 3:
        return True
    for i in range(1, len(control_points_np) - 1):
        d_vector = control_points_np[i-1] - 2 * control_points_np[i] + control_points_np[i+1]
        if np.dot(d_vector, d_vector) > flatness_sq_threshold:
            return False
    return True

def _subdivide(
    control_points_np: List[np.ndarray],
    l_child_cps_output: List[np.ndarray],
    r_child_cps_output: List[np.ndarray]
):
    N = len(control_points_np)
    midpoints_buffer = [p.copy() for p in control_points_np]
    for i in range(N):
        l_child_cps_output[i] = midpoints_buffer[0].copy()
        r_child_cps_output[N - 1 - i] = midpoints_buffer[N - 1 - i].copy()
        for j in range(N - 1 - i):
            midpoints_buffer[j] = (midpoints_buffer[j] + midpoints_buffer[j+1]) * 0.5

def _approximate_flat_segment(
    segment_control_points_np: List[np.ndarray],
    output_path_points: List[np.ndarray]
):
    N = len(segment_control_points_np)
    if N == 0:
        return
    output_path_points.append(segment_control_points_np[0].copy())
    if N < 2:
        return
    if N == 2: 
        return

    l_for_subdivision = [np.zeros(2, dtype=float) for _ in range(N)]
    r_for_subdivision = [np.zeros(2, dtype=float) for _ in range(N)]
    _subdivide(segment_control_points_np, l_for_subdivision, r_for_subdivision)

    temp_combined_sequence = [np.zeros(2, dtype=float) for _ in range(2 * N - 1)]
    for k_idx in range(N):
        temp_combined_sequence[k_idx] = l_for_subdivision[k_idx]
    for k_idx in range(N - 1):
        temp_combined_sequence[N + k_idx] = r_for_subdivision[k_idx + 1]

    for i in range(1, N - 1):
        idx_in_combined = 2 * i
        approximated_point = (temp_combined_sequence[idx_in_combined - 1] +
                              2 * temp_combined_sequence[idx_in_combined] +
                              temp_combined_sequence[idx_in_combined + 1]) * 0.25
        output_path_points.append(approximated_point)

def calculate_bezier_points(
    control_points_tuples: List[Tuple[float, float]],
    num_points: int
) -> List[Tuple[float, float]]:
    if not control_points_tuples:
        return []

    control_points_np = _to_np_arrays(control_points_tuples)
    n_cps = len(control_points_np)

    if n_cps == 0:
        return []
    if n_cps == 1:
        return [control_points_tuples[0]] * max(1, num_points) 

    output_path_points_adaptive: List[np.ndarray] = []
    to_flatten_stack: List[List[np.ndarray]] = []
    free_buffers_stack: List[List[np.ndarray]] = []

    initial_curve_segment = [p.copy() for p in control_points_np]
    to_flatten_stack.append(initial_curve_segment)

    flatness_threshold = BEZIER_QUANTIZATIONSQ

    while to_flatten_stack:
        current_segment_cps = to_flatten_stack.pop()
        if _is_flat_enough(current_segment_cps, flatness_threshold):
            _approximate_flat_segment(current_segment_cps, output_path_points_adaptive)
            free_buffers_stack.append(current_segment_cps)
        else:
            l_child_cps_buffer = [np.zeros(2, dtype=float) for _ in range(n_cps)]
            if free_buffers_stack:
                r_child_cps_buffer = free_buffers_stack.pop()
            else:
                r_child_cps_buffer = [np.zeros(2, dtype=float) for _ in range(n_cps)]
            
            _subdivide(current_segment_cps, l_child_cps_buffer, r_child_cps_buffer)
            for i in range(n_cps):
                current_segment_cps[i] = l_child_cps_buffer[i]
            
            to_flatten_stack.append(r_child_cps_buffer)
            to_flatten_stack.append(current_segment_cps)

    if n_cps > 0:
        if not output_path_points_adaptive or not np.allclose(output_path_points_adaptive[-1], control_points_np[-1]):
             output_path_points_adaptive.append(control_points_np[n_cps - 1].copy())


    if num_points == 0:
        return []
    if not output_path_points_adaptive:
        if n_cps == 1: 
            return [control_points_tuples[0]] * max(1, num_points)
        if n_cps > 1: 
            return calculate_linear_points(control_points_tuples[0], control_points_tuples[-1], num_points)
        return []

    unique_adaptive_points: List[np.ndarray] = []
    if output_path_points_adaptive:
        unique_adaptive_points.append(output_path_points_adaptive[0])
        for i in range(1, len(output_path_points_adaptive)):
            if not np.allclose(output_path_points_adaptive[i], output_path_points_adaptive[i-1], atol=1e-9): # Stricter tolerance
                unique_adaptive_points.append(output_path_points_adaptive[i])
    
    if not unique_adaptive_points: 
        return [] 
    if num_points == 1: 
        return [_to_tuples([unique_adaptive_points[0]])[0]]
    if len(unique_adaptive_points) == 1:
        return [_to_tuples([unique_adaptive_points[0]])[0]] * num_points

    path_data_np = np.array(unique_adaptive_points)
    segment_lengths = np.sqrt(np.sum(np.diff(path_data_np, axis=0)**2, axis=1))
    cumulative_distances = np.zeros(len(path_data_np))
    cumulative_distances[1:] = np.cumsum(segment_lengths)

    if cumulative_distances[-1] < 1e-9: 
        return [_to_tuples([path_data_np[0]])[0]] * num_points

    u_parameter_orig = cumulative_distances / cumulative_distances[-1]
    u_parameter_new = np.linspace(0, 1, num_points)
    
    valid_indices = np.where(np.concatenate(([True], np.diff(u_parameter_orig) > 1e-9)))[0]
    if len(valid_indices) < 2: 
        if len(u_parameter_orig) >=2 and np.allclose(u_parameter_orig[0], u_parameter_orig[-1]): 
            return [_to_tuples([path_data_np[0]])[0]] * num_points
        return [_to_tuples([path_data_np[0]])[0]] * num_points


    u_parameter_orig_mono = u_parameter_orig[valid_indices]
    path_data_np_mono = path_data_np[valid_indices]

    if len(path_data_np_mono) >= 4 and num_points > 2 : 
        try:
            cs_x = CubicSpline(u_parameter_orig_mono, path_data_np_mono[:, 0])
            cs_y = CubicSpline(u_parameter_orig_mono, path_data_np_mono[:, 1])
            x_coords_new = cs_x(u_parameter_new)
            y_coords_new = cs_y(u_parameter_new)
        except ValueError: 
            x_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 0])
            y_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 1])

    else: 
        x_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 0])
        y_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 1])
            
    return list(zip(x_coords_new, y_coords_new))


def calculate_piecewise_linear_interpolated_points(
    control_points: List[Tuple[float, float]],
    num_points: int
) -> List[Tuple[float, float]]:
    if not control_points:
        return []
    if num_points == 0:
        return []
    
    if len(control_points) == 1:
        return [control_points[0]] * max(1, num_points)

    path_data_np = np.array(control_points, dtype=float)

    if len(path_data_np) == 1: 
        return [tuple(path_data_np[0])] * num_points

    segment_lengths = np.sqrt(np.sum(np.diff(path_data_np, axis=0)**2, axis=1))
    cumulative_distances = np.zeros(len(path_data_np))
    cumulative_distances[1:] = np.cumsum(segment_lengths)

    if cumulative_distances[-1] < 1e-9: 
        return [tuple(path_data_np[0])] * num_points

    u_parameter_orig = cumulative_distances / cumulative_distances[-1]
    u_parameter_new = np.linspace(0, 1, num_points)
    
    valid_indices = np.where(np.concatenate(([True], np.diff(u_parameter_orig) > 1e-9)))[0]
    if len(valid_indices) < 1: 
        return []
    if len(valid_indices) == 1: 
        return [tuple(path_data_np[valid_indices[0]])] * num_points

    u_parameter_orig_mono = u_parameter_orig[valid_indices]
    path_data_np_mono = path_data_np[valid_indices]
    
    if len(path_data_np_mono) == 1: 
        return [tuple(path_data_np_mono[0])] * num_points

    x_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 0])
    y_coords_new = np.interp(u_parameter_new, u_parameter_orig_mono, path_data_np_mono[:, 1])
            
    return list(zip(x_coords_new, y_coords_new))
