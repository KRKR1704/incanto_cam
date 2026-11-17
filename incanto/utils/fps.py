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


class UniqueFps:
    """How many frames per second actually show something new.

    Webcams in a dim room lengthen their exposure and fall to ~15 fps, but the driver still delivers 30 frames a second:
    every other one is a repeat of the one before (measured: 50% of a 5-minute clip). `update()` takes each processed
    frame (any BGR image) and says whether it differs from the previous one: both are shrunk to 96x54 grey and a frame is
    a repeat when the mean absolute difference is below `threshold` levels (a repeated frame is bit-identical, so 0;
    sensor noise alone, averaged over that shrink, stays well above 0.05).

    `interval_fps()` is the unique rate since its last call (for the 5-second log line). `warning` turns on once the rate
    over the last `warn_after_s` seconds has stayed under `warn_below`, and off again above `warn_below + 2`.
    """

    SIZE = (96, 54)

    def __init__(self, warn_below: float = 20.0, warn_after_s: float = 10.0, threshold: float = 0.03):
        self.warn_below, self.warn_after_s, self.threshold = warn_below, warn_after_s, threshold
        self.warning = False
        self._prev = None
        self._recent: deque[tuple[float, bool]] = deque()   # (time, unique) for the warning window
        self._interval_unique, self._interval_start = 0, None

    def update(self, frame, now: float) -> bool:
        """Judge one processed frame (BGR) against the previous one; returns True if it is new."""
        import cv2
        gray = cv2.cvtColor(cv2.resize(frame, self.SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        unique = self._prev is None or float(cv2.absdiff(gray, self._prev).mean()) >= self.threshold
        self._prev = gray
        self.feed(unique, now)
        return unique

    def feed(self, unique: bool, now: float) -> None:
        """Record a verdict made elsewhere (the clip checker, tests): updates the rates and the warning."""
        if self._interval_start is None:
            self._interval_start = now
        self._interval_unique += unique
        self._recent.append((now, unique))
        while self._recent and now - self._recent[0][0] > self.warn_after_s:
            self._recent.popleft()
        span = now - self._recent[0][0]
        if span >= self.warn_after_s * 0.95:               # a full window of data
            rate = sum(u for _, u in self._recent) / span
            if rate < self.warn_below:
                self.warning = True
            elif rate >= self.warn_below + 2.0:
                self.warning = False

    def interval_fps(self, now: float) -> float:
        """Unique frames per second since the previous call (0 if nothing was processed)."""
        start, count = self._interval_start, self._interval_unique
        self._interval_start, self._interval_unique = now, 0
        return count / (now - start) if start is not None and now > start else 0.0

    def window_fps(self) -> float:
        if len(self._recent) < 2:
            return 0.0
        span = self._recent[-1][0] - self._recent[0][0]
        return sum(u for _, u in self._recent) / span if span > 0 else 0.0
