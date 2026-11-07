"""Strictly increasing millisecond timestamps for MediaPipe VIDEO running mode."""
from __future__ import annotations

import time


class MonotonicMs:
    """Returns int milliseconds from a monotonic clock, forced to be strictly increasing."""

    def __init__(self):
        self._last = -1

    def now(self) -> int:
        ts = int(time.monotonic() * 1000)
        if ts <= self._last:
            ts = self._last + 1
        self._last = ts
        return ts
