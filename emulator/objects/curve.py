# from typing import List, Tuple
# import numpy as np



# def calculate_linear_points(start: Tuple[float, float], end: Tuple[float, float], num_points: int) -> List[Tuple[float, float]]:
#     t = np.linspace(0, 1, num_points)
#     x = start[0] + t * (end[0] - start[0])
#     y = start[1] + t * (end[1] - start[1])
#     return list(zip(x, y))

# def calculate_perfect_circle_points(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float], 
#                                     num_points: int) -> List[Tuple[float, float]]:
#     if is_collinear(a, b, c) or are_points_identical([a, b, c]):
#         return calculate_linear_points(a, c, num_points)
    
#     center, radius = find_circle_center_radius(a, b, c)
#     if center is None or radius <= 0:
#         return calculate_linear_points(a, c, num_points)
    
#     points = np.array([a, b, c])
#     angles = np.arctan2(points[:, 1] - center[1], points[:, 0] - center[0])
    
#     angles = np.mod(angles, 2 * np.pi)
    
#     angle_diffs = np.diff(angles)
#     angle_diffs = np.mod(angle_diffs, 2 * np.pi)
#     angle_diffs = np.where(angle_diffs > np.pi, 2 * np.pi - angle_diffs, angle_diffs)
    
#     cross_product = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
#     direction = -1 if cross_product < 0 else 1
    
#     total_angle = np.sum(angle_diffs) * direction
    
#     t = np.linspace(0, 1, num_points)
#     angles = angles[0] + t * total_angle
    
#     x = center[0] + radius * np.cos(angles)
#     y = center[1] + radius * np.sin(angles)
    
#     return list(zip(x, y))

# BEZIER_QUANTIZATION = 0.5
# BEZIER_QUANTIZATIONSQ = BEZIER_QUANTIZATION * BEZIER_QUANTIZATION

# def is_flat_enough(control_points: List[Tuple[float, float]]) -> bool:
#     for i in range(1, len(control_points) - 1):
#         p0 = np.array(control_points[i-1])
#         p1 = np.array(control_points[i])
#         p2 = np.array(control_points[i+1])
        
#         derivative = p0 - 2 * p1 + p2
#         if np.sum(derivative * derivative) > BEZIER_QUANTIZATIONSQ:
#             return False
#     return True

# def subdivide_bezier(control_points: List[Tuple[float, float]]) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]]]:
#     n = len(control_points)
#     left = [None] * n
#     right = [None] * n
    
#     midpoints = [np.array(p) for p in control_points]
    
#     for i in range(n):
#         left[i] = tuple(midpoints[0])
#         right[n-i-1] = tuple(midpoints[n-i-1])
        
#         for j in range(n-i-1):
#             midpoints[j] = (midpoints[j] + midpoints[j+1]) * 0.5
    
#     return left, right

# def approximate_bezier_segment(control_points: List[Tuple[float, float]], output: List[Tuple[float, float]]):
#     n = len(control_points)
#     if n == 0:
#         return
    
#     left, right = subdivide_bezier(control_points)
    
#     output.append(control_points[0])
    
#     for i in range(1, n-1):
#         p = (np.array(left[i-1]) + 2 * np.array(left[i]) + np.array(left[i+1])) * 0.25
#         output.append(tuple(p))

# def _calculate_bezier_points(control_points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
#     if not control_points:
#         return []
    
#     output = []
#     to_flatten = [control_points.copy()]
#     free_buffers = []
    
#     while to_flatten:
#         parent = to_flatten.pop()
        
#         if is_flat_enough(parent):
#             approximate_bezier_segment(parent, output)
#             free_buffers.append(parent)
#             continue
        
#         left_child, right_child = subdivide_bezier(parent)
        
#         for i in range(len(parent)):
#             parent[i] = left_child[i]
        
#         to_flatten.append(right_child)
#         to_flatten.append(parent)
    
#     output.append(control_points[-1])
#     return output

# def calculate_bezier_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
#     if not points:
#         return []
    
#     if len(points) <= 2:
#         return points
    
#     out_points = []
#     last_index = 0
    
#     for i in range(len(points)):
#         multi = i < len(points)-2 and points[i] == points[i+1]
        
#         if multi or i == len(points)-1:
#             sub_points = points[last_index:i+1]
            
#             if len(sub_points) == 2:
#                 inter = [sub_points[0], sub_points[1]]
#             else:
#                 inter = _calculate_bezier_points(sub_points)
            
#             if not out_points or out_points[-1] != inter[0]:
#                 out_points.extend(inter)
#             else:
#                 out_points.extend(inter[1:])
            
#             if multi:
#                 i += 1
            
#             last_index = i
    
#     return out_points

# def calculate_catmull_points(control_points: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
#     if len(control_points) < 4:
#         return calculate_bezier_points(control_points, num_points)
#     points = np.array(control_points)
#     chord_lengths = np.zeros(len(points))
#     for i in range(1, len(points)):
#         chord_lengths[i] = chord_lengths[i-1] + np.linalg.norm(points[i] - points[i-1])
    
#     if chord_lengths[-1] > 0:
#         chord_lengths = chord_lengths / chord_lengths[-1]
    
#     t = np.linspace(0, 1, num_points)
#     result = []
    
#     for i in range(len(points) - 3):
#         p0, p1, p2, p3 = points[i:i+4]
#         t0, t1, t2, t3 = chord_lengths[i:i+4]
        
#         def basis(t, t0, t1, t2, t3):
#             t2_t1 = t2 - t1
#             t3_t2 = t3 - t2
#             t1_t0 = t1 - t0
            
#             if t2_t1 == 0 or t3_t2 == 0 or t1_t0 == 0:
#                 return 0
                
#             a = (t - t1) / t2_t1
            
#             c = (t - t0) / t1_t0
            
#             return np.array([
#                 -0.5 * c * (1 - c),
#                 0.5 * (1 + 2*c - 3*c*c),
#                 0.5 * (1 + 2*a - 3*a*a),
#                 -0.5 * a * (1 - a)
#             ])
        
#         segment_points = []
#         for ti in t:
#             if t1 <= ti <= t2:
#                 b = basis(ti, t0, t1, t2, t3)
#                 point = b[0] * p0 + b[1] * p1 + b[2] * p2 + b[3] * p3
#                 segment_points.append(tuple(point))
        
#         result.extend(segment_points)
    
#     if len(result) > num_points:
#         indices = np.linspace(0, len(result)-1, num_points, dtype=int)
#         result = [result[i] for i in indices]
#     elif len(result) < num_points:
#         result.extend([result[-1]] * (num_points - len(result)))
    
#     return result

# def is_collinear(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> bool:
#     points = np.array([a, b, c])
#     return np.abs(np.linalg.det(np.column_stack([points, np.ones(3)]))) < 1e-10

# def are_points_identical(points: List[Tuple[float, float]]) -> bool:
#     if not points:
#         return True
#     points_array = np.array(points)
#     return np.all(np.all(points_array == points_array[0], axis=1))

# def find_circle_center_radius(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> Tuple[Tuple[float, float], float]:
#     try:
#         points = np.array([a, b, c])
        
#         mid1 = (points[0] + points[1]) / 2
#         mid2 = (points[1] + points[2]) / 2
#         v1 = points[1] - points[0]
#         v2 = points[2] - points[1]
        
#         if np.all(v1 == 0) or np.all(v2 == 0):
#             return None, 0
            
#         perp1 = np.array([-v1[1], v1[0]])
#         perp2 = np.array([-v2[1], v2[0]])
        
#         perp1 = perp1 / np.linalg.norm(perp1)
#         perp2 = perp2 / np.linalg.norm(perp2)
        
#         A = np.column_stack([perp1, -perp2])
#         b = mid2 - mid1
        
#         try:
#             center = mid1 + perp1 * np.linalg.solve(A, b)[0]
#         except np.linalg.LinAlgError:
#             return None, 0
            
#         radius = np.linalg.norm(center - points[0])
        
#         return tuple(center), radius
#     except Exception:
#         return None, 0 

