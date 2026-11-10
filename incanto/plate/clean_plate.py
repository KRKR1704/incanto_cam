"""Captures and stores the empty-room clean plate.

The plate is a photo of your room, so it is LOCAL-ONLY: it lives in `plates/` (gitignored, and the
pre-commit hook refuses images) and is never committed.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

PLATE_IMAGE = "plate.png"
PLATE_META = "plate.json"
PERSON_THRESHOLD = 0.5  # mask value above which a pixel counts as "person" for coverage checks


class PlateAccumulator:
    """Running sum of BGR frames; `result()` is their mean as a float32 plate at full resolution."""

    def __init__(self) -> None:
        self._sum: np.ndarray | None = None
        self.count = 0

    def add(self, frame: np.ndarray) -> None:
        if self._sum is None:
            self._sum = np.zeros(frame.shape, np.float32)
        elif frame.shape != self._sum.shape:
            raise ValueError(f"frame shape {frame.shape} != plate shape {self._sum.shape}")
        cv2.accumulate(frame, self._sum)
        self.count += 1

    def result(self) -> np.ndarray:
        if self._sum is None or self.count == 0:
            raise ValueError("no frames accumulated")
        return self._sum / np.float32(self.count)


def person_coverage(mask: np.ndarray) -> float:
    """Fraction of the frame the person mask covers (0..1)."""
    return float((mask > PERSON_THRESHOLD).mean())


@dataclass
class PlateMeta:
    width: int
    height: int
    frames: int
    exposure_mode: str            # "auto" | "manual"
    exposure_value: float | None  # log2 seconds when manual
    mean_brightness: float        # 0-255, mean over all pixels and channels
    timestamp: str                # UTC, ISO 8601
    peak_person_coverage: float   # worst mask coverage seen during the capture


@dataclass
class Plate:
    data: np.ndarray              # float32 BGR, full output resolution
    meta: PlateMeta

    def __post_init__(self) -> None:
        self.u8 = np.clip(self.data + 0.5, 0, 255).astype(np.uint8)  # what the compositor blends


def make_plate(data: np.ndarray, frames: int, exposure: float | None, peak_coverage: float) -> Plate:
    h, w = data.shape[:2]
    meta = PlateMeta(width=w, height=h, frames=frames,
                     exposure_mode="auto" if exposure is None else "manual", exposure_value=exposure,
                     mean_brightness=float(data.mean()),
                     timestamp=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                     peak_person_coverage=peak_coverage)
    return Plate(data, meta)


def save_plate(plate: Plate, directory: Path) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    image = directory / PLATE_IMAGE
    if not cv2.imwrite(str(image), plate.u8):  # lossless PNG
        raise OSError(f"could not write {image}")
    (directory / PLATE_META).write_text(json.dumps(asdict(plate.meta), indent=2), encoding="utf-8")
    return image


def load_plate(directory: Path, width: int, height: int) -> Plate | None:
    """Load the saved plate, or None if absent or invalid (a resolution change invalidates it)."""
    image, meta_path = directory / PLATE_IMAGE, directory / PLATE_META
    if not image.exists() or not meta_path.exists():
        return None
    try:
        meta = PlateMeta(**json.loads(meta_path.read_text(encoding="utf-8")))
    except (ValueError, TypeError, KeyError) as exc:
        log.warning("Ignoring saved plate: unreadable sidecar (%s)", exc)
        return None
    if (meta.width, meta.height) != (width, height):
        log.warning("Saved plate is %dx%d but the camera is %dx%d: plate invalidated, press B to capture a new one.",
                    meta.width, meta.height, width, height)
        return None
    data = cv2.imread(str(image), cv2.IMREAD_COLOR)
    if data is None or data.shape[:2] != (height, width):
        log.warning("Ignoring saved plate: image missing or wrong size")
        return None
    return Plate(data.astype(np.float32), meta)


class PlateCapture:
    """Non-blocking capture sequence, advanced once per main-loop frame so the live feed never stalls.

    countdown (preview shows the count) -> settle (auto exposure adapts to the empty room) ->
    capturing (N frames averaged) -> done, or aborted if the segmenter still sees a person.
    The vcam output is untouched: this only produces preview text via `banner()` and audio-cue
    names via `pop_events()` ("tick" for each countdown second, "capturing", "error" on abort).
    """

    IDLE, COUNTDOWN, SETTLE, CAPTURING, DONE, ABORTED = (
        "idle", "countdown", "settle", "capturing", "done", "aborted")

    def __init__(self, frames: int, countdown_s: float, settle_s: float, max_coverage: float):
        self.frames, self.countdown_s, self.settle_s, self.max_coverage = frames, countdown_s, settle_s, max_coverage
        self.state = self.IDLE
        self.abort_reason = ""
        self.peak_coverage = 0.0
        self._acc = PlateAccumulator()
        self._phase_start = 0.0
        self._events: list[str] = []
        self._ticks_left = 0

    def pop_events(self) -> list[str]:
        """Audio cues due since the last call (the caller plays them)."""
        events, self._events = self._events, []
        return events

    @property
    def active(self) -> bool:
        return self.state in (self.COUNTDOWN, self.SETTLE, self.CAPTURING)

    def start(self, now: float) -> None:
        self.state, self.abort_reason, self.peak_coverage = self.COUNTDOWN, "", 0.0
        self._acc = PlateAccumulator()
        self._phase_start = now
        self._ticks_left = int(np.ceil(self.countdown_s))  # one tick per whole second still to go
        self._events = ["tick"] if self._ticks_left else []
        log.info("Plate capture: leave the frame. Countdown %.0fs, settle %.1fs, then %d frames.",
                 self.countdown_s, self.settle_s, self.frames)

    def cancel(self) -> None:
        if self.active:
            self.state = self.IDLE

    def update(self, frame: np.ndarray, mask: np.ndarray | None, now: float) -> None:
        if self.state == self.COUNTDOWN:
            left = int(np.ceil(self.countdown_s - (now - self._phase_start)))
            if 0 < left < self._ticks_left:  # the displayed number just dropped: one beep per second
                self._ticks_left = left
                self._events.append("tick")
        if self.state == self.COUNTDOWN and now - self._phase_start >= self.countdown_s:
            self.state, self._phase_start = self.SETTLE, now
        if self.state == self.SETTLE and now - self._phase_start >= self.settle_s:
            self.state = self.CAPTURING
            self._events.append("capturing")
            log.info("Plate capture: capturing %d frames now.", self.frames)
        if self.state == self.CAPTURING:
            if mask is not None:
                coverage = person_coverage(mask)
                self.peak_coverage = max(self.peak_coverage, coverage)
                if coverage > self.max_coverage:
                    self.state = self.ABORTED
                    self.abort_reason = (f"Person still in frame ({coverage:.1%} of pixels, "
                                         f"{self.max_coverage:.1%} allowed). Plate NOT saved. Press B to retry.")
                    self._phase_start = now
                    self._events.append("error")
                    log.warning("Plate capture aborted: %s", self.abort_reason)
                    return
            self._acc.add(frame)
            if self._acc.count >= self.frames:
                self.state = self.DONE

    def result(self) -> np.ndarray:
        return self._acc.result()

    def banner(self, now: float) -> tuple[str, bool] | None:
        """(text, is_warning) to draw on the preview, or None."""
        if self.state == self.COUNTDOWN:
            left = max(0.0, self.countdown_s - (now - self._phase_start))
            return f"Leave the frame: {int(np.ceil(left))}", False
        if self.state == self.SETTLE:
            return "Hold still: settling exposure...", False
        if self.state == self.CAPTURING:
            return f"Capturing plate {self._acc.count}/{self.frames}", False
        if self.state == self.ABORTED and now - self._phase_start < 6.0:
            return self.abort_reason, True
        return None
