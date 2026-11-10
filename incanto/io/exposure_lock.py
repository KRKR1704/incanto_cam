"""Exposure lock while a plate is in use.

The plate is captured with auto exposure. Afterwards auto exposure keeps reacting to you (standing
up, leaning in) and the whole picture swings by tens of levels. So after a successful capture this
sweeps manual exposure values, picks the one whose picture is closest in brightness to the plate,
and holds it. Only values that keep 30 fps are tried (exposure <= -5, i.e. at most 1/32 s).

It is a small state machine advanced once per main-loop frame; it never blocks. It does not touch
the camera itself: it calls `set_exposure(value)` (a number = manual, None = auto), so the camera's
own safety net (io/camera.py: every camera switched to manual is restored to auto on any exit)
covers the lock exactly as it covers a manual exposure from the config.
"""
from __future__ import annotations

import logging
import math
from typing import Callable

from incanto.io.camera import EXPOSURE_FPS_LIMIT

log = logging.getLogger(__name__)

# log2 seconds, brightest first. Each step halves the light, so the sweep can stop at the first value
# that is darker than the plate.
CANDIDATES = tuple(EXPOSURE_FPS_LIMIT - i for i in range(5))  # -5 .. -9
MEASURE_FRAMES = 5


class ExposureLock:
    IDLE, SWEEPING, LOCKED, AUTO = "idle", "sweeping", "locked", "auto"

    def __init__(self, set_exposure: Callable[[float | None], None], tolerance: float, settle_s: float,
                 candidates: tuple[float, ...] = CANDIDATES, measure_frames: int = MEASURE_FRAMES):
        if any(c > EXPOSURE_FPS_LIMIT for c in candidates):
            raise ValueError(f"exposure lock candidates must be <= {EXPOSURE_FPS_LIMIT:g} to keep 30 fps")
        self._set = set_exposure
        self.tolerance, self.settle_s = tolerance, settle_s
        self.candidates, self.measure_frames = tuple(candidates), measure_frames
        self.state = self.IDLE
        self.value: float | None = None       # the locked exposure while LOCKED
        self.ratio: float | None = None       # live / plate brightness at the locked (or best) value
        self.reason = ""                      # why the camera stayed on auto
        self._resume: float | None = None     # lock to go back to if a recapture is aborted
        self._resume_ratio: float | None = None
        self._measured: list[tuple[float, float]] = []
        self._samples: list[float] = []
        self._index = 0
        self._phase_start = 0.0

    @property
    def sweeping(self) -> bool:
        return self.state == self.SWEEPING

    def describe(self) -> str:
        if self.state == self.LOCKED:
            return f"locked at {self.value:g} for the plate (brightness {self.ratio:.2f}x the plate's)"
        if self.state == self.SWEEPING:
            return "choosing an exposure for the plate"
        if self.state == self.AUTO:
            return f"auto ({self.reason})"
        return "auto"

    # -- driven by the main loop -------------------------------------------------------------------
    def start(self, now: float) -> None:
        """A plate was just captured in auto exposure: begin the sweep."""
        self._measured, self._index, self._resume = [], 0, None
        self.value = self.ratio = None
        self.state = self.SWEEPING
        self._try(self.candidates[0], now)
        log.info("Exposure lock: sweeping manual exposures %s to match the plate.",
                 ", ".join(f"{c:g}" for c in self.candidates))

    def _try(self, value: float, now: float) -> None:
        self._samples, self._phase_start = [], now
        self._set(value)

    def update(self, ratio: float | None, now: float) -> bool:
        """Feed one frame's background brightness ratio (live / plate; None = not measurable).
        Returns True on the frame the sweep finishes."""
        if self.state != self.SWEEPING or now - self._phase_start < self.settle_s:
            return False
        if ratio is not None:
            self._samples.append(ratio)
        timed_out = now - self._phase_start > self.settle_s + 3.0
        if len(self._samples) < self.measure_frames and not timed_out:
            return False
        if self._samples:
            mean = sum(self._samples) / len(self._samples)
            self._measured.append((self.candidates[self._index], mean))
        # Each further candidate is darker still: stop once this one is at or below the plate's brightness.
        darker = bool(self._samples) and mean <= 1.0
        if not darker and not timed_out and self._index + 1 < len(self.candidates):
            self._index += 1
            self._try(self.candidates[self._index], now)
            return False
        self._finish()
        return True

    def _finish(self) -> None:
        table = ", ".join(f"{v:g}: {r:.2f}x" for v, r in self._measured) or "nothing measurable"
        usable = [(abs(math.log(max(r, 1e-3))), v, r) for v, r in self._measured]
        if usable:
            _, best, ratio = min(usable)
            self.ratio = ratio
            if abs(ratio - 1.0) <= self.tolerance:
                if best != self.candidates[self._index]:
                    self._set(best)
                self.state, self.value = self.LOCKED, best
                log.info("Exposure lock: locked at %g; live brightness is %.2fx the plate's (tolerance +/-%.0f%%). "
                         "Measured %s.", best, ratio, self.tolerance * 100, table)
                return
            self.reason = (f"closest manual exposure {best:g} gives {ratio:.2f}x the plate's brightness, "
                           f"outside +/-{self.tolerance:.0%}")
        else:
            self.reason = "the background could not be measured during the sweep"
        self._set(None)
        self.state, self.value = self.AUTO, None
        log.warning("Exposure lock: staying on AUTO exposure: %s. Measured %s (only exposures <= %g keep 30 fps).",
                    self.reason, table, EXPOSURE_FPS_LIMIT)

    def release(self) -> None:
        """Back to auto exposure before a recapture. Remembers the lock in case the capture fails."""
        if self.state in (self.SWEEPING, self.LOCKED):
            self._resume = self.value if self.state == self.LOCKED else None
            self._resume_ratio = self.ratio
            self._set(None)
            log.info("Exposure lock: released, camera back on auto exposure for the capture.")
        self.state, self.value = self.IDLE, None

    def restore(self) -> None:
        """The recapture was aborted or cancelled: the old plate stays, so its lock comes back."""
        if self._resume is not None:
            self.state, self.value, self.ratio = self.LOCKED, self._resume, self._resume_ratio
            self._resume = None
            self._set(self.value)
            log.info("Exposure lock: capture did not finish; re-locked at %g for the plate still in use.", self.value)
