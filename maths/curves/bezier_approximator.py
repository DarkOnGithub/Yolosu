import numpy as np

BEZIER_QUANTIZATION = 0.5
BEZIER_QUANTIZATIONSQ = BEZIER_QUANTIZATION ** 2

class BezierApproximator:
    def __init__(self, control_points: list[tuple[float, float]]):
        self.control_points = np.array(control_points, dtype=np.float32)
        if self.control_points.shape[1] != 2:
            raise ValueError("Control points must be 2D")
        self.count = len(control_points)
        self.sub_division_buffer_1 = np.empty((self.count, 2), dtype=np.float32)
        self.sub_division_buffer_2 = np.empty((self.count * 2 - 1, 2), dtype=np.float32)

    def is_flat_enough(self, control_points: np.ndarray) -> bool:
        for i in range(1, len(control_points) - 1):
            p1 = control_points[i - 1]
            p2 = control_points[i]
            p3 = control_points[i + 1]
            if np.sum((p1 - 2 * p2 + p3) ** 2) > BEZIER_QUANTIZATIONSQ:
                return False
        return True

    def subdivide(self, control_points: np.ndarray, left: np.ndarray, right: np.ndarray):
        mid_points = self.sub_division_buffer_1
        mid_points[:] = control_points

        for i in range(self.count):
            left[i] = mid_points[0]
            right[self.count - i - 1] = mid_points[self.count - i - 1]
            for j in range(self.count - i - 1):
                mid_points[j] = (mid_points[j] + mid_points[j + 1]) * 0.5

    def approximate(self, control_points: np.ndarray, output: list[tuple[float, float]]):
        l = self.sub_division_buffer_2
        r = self.sub_division_buffer_1

        self.subdivide(control_points, l, r)

        for i in range(self.count - 1):
            l[self.count + i] = r[i + 1]

        output.append(tuple(control_points[0]))

        for i in range(1, self.count - 1):
            index = 2 * i
            p = (l[index - 1] + 2 * l[index] + l[index + 1]) * 0.25
            output.append((p[0], p[1]))

    def create_bezier(self) -> list[tuple[float, float]]:
        output = []
        if self.count == 0:
            return output

        to_flatten = []
        free_buffers = []

        nCP = np.array(self.control_points, dtype=np.float32)
        to_flatten.append(nCP)

        while to_flatten:
            parent = to_flatten.pop()
            if self.is_flat_enough(parent):
                self.approximate(parent, output)
                free_buffers.append(parent)
                continue

            right = free_buffers.pop() if free_buffers else np.empty((self.count, 2), dtype=np.float32)
            left = np.empty_like(parent) 

            self.subdivide(parent, left, right)

            parent[:] = left
            to_flatten.append(right)
            to_flatten.append(parent)

        output.append(tuple(self.control_points[-1]))
        return output