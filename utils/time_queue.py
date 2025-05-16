from sortedcontainers import SortedList
import time
from typing import Any, Optional

class TimeQueue:
    def __init__(self, max_size: int = 1000):
        self._data = {}
        self._timestamps = SortedList()
        self.max_size = max_size

    def add(self, element: Any, timestamp: float = None) -> None:
        """
        Add an element with a timestamp (O(log n)).
        """
        timestamp = timestamp if timestamp is not None else time.time()
        self._data[timestamp] = element
        self._timestamps.add(timestamp)
        self._cleanup()

    def get_closest(self, target_time: float) -> Optional[tuple[float, Any]]:
        """
        Get the element with the timestamp closest to target_time (O(log n)).
        """
        self._cleanup()
        if not self._timestamps:
            return None

        idx = self._timestamps.bisect_left(target_time)
        if idx == 0:
            closest = self._timestamps[0]
        elif idx == len(self._timestamps):
            closest = self._timestamps[-1]
        else:
            left = self._timestamps[idx - 1]
            right = self._timestamps[idx]
            closest = left if abs(left - target_time) <= abs(right - target_time) else right

        return (closest, self._data[closest])

    def pop_oldest(self) -> Optional[tuple[float, Any]]:
        """
        Remove and return the oldest element (O(log n)).
        """
        self._cleanup()
        if not self._timestamps:
            return None

        timestamp = self._timestamps.pop(0)
        element = self._data.pop(timestamp, None)
        return (timestamp, element) if element is not None else None

    def _cleanup(self) -> None:
        """Remove oldest elements if exceeding max_size (O(k log n))."""
        while len(self._timestamps) > self.max_size:
            timestamp = self._timestamps.pop(0)
            self._data.pop(timestamp, None)

    def __len__(self) -> int:
        """Return the number of elements (O(1) after cleanup)."""
        self._cleanup()
        return len(self._data)