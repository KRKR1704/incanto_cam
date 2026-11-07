"""FPS counter and per-stage timing."""
from __future__ import annotations

import time
from collections import defaultdict, deque
from contextlib import contextmanager


class FpsCounter:
    """Rolling FPS over the last `window` ticks."""

    def __init__(self, window: int = 60):
        self._times: deque[float] = deque(maxlen=window)

    def tick(self) -> None:
        self._times.append(time.perf_counter())

    @property
    def fps(self) -> float:
        if len(self._times) < 2:
            return 0.0
        span = self._times[-1] - self._times[0]
        return (len(self._times) - 1) / span if span > 0 else 0.0


class StageTimer:
    """Accumulates per-stage durations; use `with timer.stage("name"):`."""

    def __init__(self):
        self._total: dict[str, float] = defaultdict(float)
        self._count: dict[str, int] = defaultdict(int)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self._total[name] += time.perf_counter() - start
            self._count[name] += 1

    def add(self, name: str, seconds: float) -> None:
        self._total[name] += seconds
        self._count[name] += 1

    def averages_ms(self) -> dict[str, float]:
        return {k: 1000 * self._total[k] / self._count[k] for k in self._total if self._count[k]}

    def reset(self) -> None:
        self._total.clear()
        self._count.clear()
