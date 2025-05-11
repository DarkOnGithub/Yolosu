from typing import List, Tuple
import math
import numpy as np
from scipy.interpolate import splprep, splev
import bezier

class CurveType:
    """Enum for curve types"""
    LINEAR = "L"
    PERFECT = "P"  
    BEZIER = "B"
    CATMULL = "C"

def calculate_linear_points(start: Tuple[float, float], end: Tuple[float, float], num_points: int) -> List[Tuple[float, float]]:
    """Calculate points for a linear curve from start to end using numpy for better accuracy"""
    t = np.linspace(0, 1, num_points)
    x = start[0] + t * (end[0] - start[0])
    y = start[1] + t * (end[1] - start[1])
    return list(zip(x, y))

def calculate_perfect_circle_points(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], 
                                    num_points: int) -> List[Tuple[float, float]]:
    """Calculate points for a perfect circle curve passing through points a, b, and c"""
    if is_collinear(a, b, c) or are_points_identical([a, b, c]):
        return calculate_linear_points(a, c, num_points)
    
    center, radius = find_circle_center_radius(a, b, c)
    if center is None or radius <= 0:
        return calculate_linear_points(a, c, num_points)
    
    # Calculate angles using numpy for better accuracy
    points = np.array([a, b, c])
    angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
    # Normalize angles to [0, 2π]
    angles = np.mod(angles, 2 * np.pi)
    
    # Calculate angle differences
    angle_diffs = np.diff(angles)
    angle_diffs = np.mod(angle_diffs, 2 * np.pi)
    angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
    
    # Determine direction (clockwise or counterclockwise)
    cross_product = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    direction = -1 if cross_product < 0 else 1
    
    # Calculate total angle to traverse
    total_angle = np.sum(angle_diffs) * direction
    
    # Generate points
    t = np.linspace(0, 1, num_points)
    angles = angles[0] + t * total_angle
    
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    
    return list(zip(x, y))


def calculate_bezier_points(
    knots_to_interpolate: List[Tuple[float, float]], 
    num_total_output_points: int
) -> List[Tuple[float, float]]:
    """
    Calculates points for a smooth curve that passes THROUGH each of the provided `knots_to_interpolate`.
    The curve is a sequence of cubic Bezier segments, forming a C1 continuous spline.

    Args:
        knots_to_interpolate: A list of (x, y) tuples that the curve must pass through.
        num_total_output_points: The total number of points to generate for the entire curve.

    Returns:
        A list of (x, y) tuples representing points along the interpolating spline.
    """
    if not knots_to_interpolate:
        return []
    if num_total_output_points <= 0:
        return []

    # Convert to NumPy array and remove consecutive duplicates
    points_np_orig = np.array(knots_to_interpolate, dtype=float)
    if len(points_np_orig) == 0: 
        return []
    
    P_list = [points_np_orig[0]]
    for i in range(1, len(points_np_orig)):
        if not np.all(points_np_orig[i] == P_list[-1]):
            P_list.append(points_np_orig[i])
    P = np.array(P_list)

    num_unique_knots = len(P)

    if num_unique_knots == 0:
        return []
    if num_unique_knots == 1:
        return [tuple(P[0])] * num_total_output_points
    
    if num_total_output_points == 1:
        return [tuple(P[0])]

    # For num_unique_knots >= 2:
    # Calculate tangents at each knot P[i]
    tangents = np.zeros_like(P)
    if num_unique_knots == 2: # Special case for a single segment (line)
        tangents[0] = P[1] - P[0]
        tangents[1] = P[1] - P[0] # Tangent at P1, same direction for consistent CP calculation
    else: # num_unique_knots > 2
        tangents[0] = P[1] - P[0]  # Forward difference for the start
        tangents[num_unique_knots - 1] = P[num_unique_knots - 1] - P[num_unique_knots - 2]  # Backward difference for the end
        for i in range(1, num_unique_knots - 1):
            tangents[i] = (P[i+1] - P[i-1]) / 2.0 # Central difference for interior knots

    all_segments_dense_points_list = []
    SAMPLES_PER_SEGMENT_DENSE_FIXED = 50 # Fixed samples for intermediate dense representation

    num_segments = num_unique_knots - 1
    for i in range(num_segments):
        P_start = P[i]
        P_end = P[i+1]
        
        m_start = tangents[i]
        m_end = tangents[i+1]

        cp1 = P_start + m_start / 3.0
        cp2 = P_end - m_end / 3.0
        
        segment_control_points = np.array([P_start, cp1, cp2, P_end])
        
        nodes = np.asfortranarray(segment_control_points.T)
        evaluated_segment_pts = np.array([]) # Initialize
        try:
            curve = bezier.Curve(nodes, degree=3)
            
            # Determine samples for this specific segment
            if num_segments == 1: # If only one segment in total
                current_segment_samples = num_total_output_points
            else: # Multiple segments, use dense sampling for intermediate stage
                current_segment_samples = SAMPLES_PER_SEGMENT_DENSE_FIXED
            
            if current_segment_samples < 2 : current_segment_samples = 2

            s_vals = np.linspace(0.0, 1.0, current_segment_samples)
            evaluated_segment_pts = curve.evaluate_multi(s_vals).T

        except Exception as e:
            print(f"Warning: Bezier curve construction/evaluation failed for segment {i}. Nodes: {segment_control_points}. Error: {e}. Using linear for this segment.")
            num_samples_for_failed_segment = SAMPLES_PER_SEGMENT_DENSE_FIXED
            if num_segments == 1: num_samples_for_failed_segment = num_total_output_points
            if num_samples_for_failed_segment < 2: num_samples_for_failed_segment = 2
            
            linear_pts_tuples = calculate_linear_points(tuple(P_start), tuple(P_end), num_samples_for_failed_segment)
            if linear_pts_tuples:
                 evaluated_segment_pts = np.array(linear_pts_tuples)
            elif P_start is not None : # if linear_points returned empty (e.g. num_points=0)
                 evaluated_segment_pts = np.array([P_start]) # fallback to at least the start point
            # if evaluated_segment_pts is still empty, it will be skipped by later vstack logic
        
        if len(evaluated_segment_pts) > 0:
            if i == 0:
                all_segments_dense_points_list.append(evaluated_segment_pts)
            else:
                # Skip the first point of subsequent segments (it's the same as the last point of the previous one)
                if len(evaluated_segment_pts) > 1:
                    all_segments_dense_points_list.append(evaluated_segment_pts[1:])
                elif len(evaluated_segment_pts) == 1: 
                    # Add if different from very last point added
                    if all_segments_dense_points_list and len(all_segments_dense_points_list[-1]) > 0:
                        if not np.all(evaluated_segment_pts[0] == all_segments_dense_points_list[-1][-1]):
                            all_segments_dense_points_list.append(evaluated_segment_pts)
                    else:
                        all_segments_dense_points_list.append(evaluated_segment_pts)

    if not all_segments_dense_points_list or not any(len(arr) > 0 for arr in all_segments_dense_points_list):
        return [tuple(P[0])] * num_total_output_points 

    combined_dense_points = np.vstack([arr for arr in all_segments_dense_points_list if len(arr) > 0])

    if combined_dense_points.shape[0] == 0:
        return [tuple(P[0])] * num_total_output_points
    
    num_generated_dense_points = len(combined_dense_points)
    if num_generated_dense_points == 1:
        final_points_array = np.tile(combined_dense_points[0], (num_total_output_points, 1))
    else:
        if num_total_output_points == 1:
            indices = np.array([0], dtype=int)
        else:
            indices = np.linspace(0, num_generated_dense_points - 1, num_total_output_points)
            indices = np.round(indices).astype(int)
            indices = np.clip(indices, 0, num_generated_dense_points - 1)
        final_points_array = combined_dense_points[indices]
        
    return [tuple(pt) for pt in final_points_array]

def calculate_catmull_points(control_points: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
    """Calculate points for a Catmull-Rom curve with improved parameterization"""
    if len(control_points) < 4:
        return calculate_bezier_points(control_points, num_points)
    
    # Convert to numpy arrays for better performance
    points = np.array(control_points)
    
    # Calculate chord lengths for better parameterization
    chord_lengths = np.zeros(len(points))
    for i in range(1, len(points)):
        chord_lengths[i] = chord_lengths[i-1] + np.linalg.norm(points[i] - points[i-1])
    
    # Normalize chord lengths
    if chord_lengths[-1] > 0:
        chord_lengths = chord_lengths / chord_lengths[-1]
    
    # Generate points using improved Catmull-Rom interpolation
    t = np.linspace(0, 1, num_points)
    result = []
    
    for i in range(len(points) - 3):
        p0, p1, p2, p3 = points[i:i+4]
        t0, t1, t2, t3 = chord_lengths[i:i+4]
        
        # Calculate basis functions
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
        
        # Calculate points for this segment
        segment_points = []
        for ti in t:
            if t1 <= ti <= t2:
                b = basis(ti, t0, t1, t2, t3)
                point = b[0] * p0 + b[1] * p1 + b[2] * p2 + b[3] * p3
                segment_points.append(tuple(point))
        
        result.extend(segment_points)
    
    # Ensure we have exactly num_points
    if len(result) > num_points:
        indices = np.linspace(0, len(result)-1, num_points, dtype=int)
        result = [result[i] for i in indices]
    elif len(result) < num_points:
        # Pad with last point if needed
        result.extend([result[-1]] * (num_points - len(result)))
    
    return result

def is_collinear(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
    """Check if three points are collinear using numpy for better accuracy"""
    points = np.array([a, b, c])
    return np.abs(np.linalg.det(np.column_stack([points, np.ones(3)]))) < 1e-10

def are_points_identical(points: List[Tuple[float, float]]) -> bool:
    """Check if all points are identical using numpy for better accuracy"""
    if not points:
        return True
    points_array = np.array(points)
    return np.all(np.all(points_array == points_array[0], axis=1))

def find_circle_center_radius(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
    """Find the center and radius of a circle passing through three points using numpy for better accuracy"""
    try:
        points = np.array([a, b, c])
        
        # Calculate midpoints
        mid1 = (points[0] + points[1]) / 2
        mid2 = (points[1] + points[2]) / 2
        
        # Calculate perpendicular vectors
        v1 = points[1] - points[0]
        v2 = points[2] - points[1]
        
        # Handle special cases
        if np.all(v1 == 0) or np.all(v2 == 0):
            return None, 0
            
        # Calculate perpendicular bisectors
        perp1 = np.array([-v1[1], v1[0]])
        perp2 = np.array([-v2[1], v2[0]])
        
        # Normalize perpendicular vectors
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = perp2 / np.linalg.norm(perp2)
        
        # Find intersection of perpendicular bisectors
        A = np.column_stack([perp1, -perp2])
        b = mid2 - mid1
        
        try:
            center = mid1 + perp1 * np.linalg.solve(A, b)[0]
        except np.linalg.LinAlgError:
            return None, 0
            
        # Calculate radius
        radius = np.linalg.norm(center - points[0])
        
        return tuple(center), radius
    except:
        return None, 0 