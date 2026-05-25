from __future__ import annotations

import time
from collections import deque


class FPSMeter:
    def __init__(self, window: int = 30):
        self.window = deque(maxlen=window)
        self.last_time: float | None = None

    def tick(self) -> float:
        now = time.perf_counter()
        if self.last_time is None:
            self.last_time = now
            return 0.0
        delta = now - self.last_time
        self.last_time = now
        if delta > 0:
            self.window.append(1.0 / delta)
        if not self.window:
            return 0.0
        return sum(self.window) / len(self.window)

