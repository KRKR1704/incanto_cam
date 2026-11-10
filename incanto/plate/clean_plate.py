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
ABORT_IMAGE_PREFIX = "capture-abort"
# Static-blob check (measured on this camera: a person sitting still shows recent motion in 6-40% of
# their blob, static background in about 0.3%).
MOTION_WIDTH = 320      # the check runs on frames this wide
MOTION_LEVELS = 8.0     # a pixel "moved" if its brightness changed by more than this between two frames
MOTION_WINDOW_S = 1.0   # ...and stays "recently moved" for this long
MIN_BLOB = 0.003        # segmenter blobs smaller than this fraction of the frame are speckle


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


class PersonMotion:
    """Tells a person from furniture the segmenter mistakes for one (a gaming chair with a headrest).

    Fed every frame of the countdown and the capture. A segmenter blob in which hardly anything moved
    during the last second (and whose outline did not change) is furniture and does not count; a blob
    with motion inside it is a person. A blob is also a person until a full second has been observed.
    """

    def __init__(self, max_static_motion: float = 0.02):
        self.max_static_motion = max_static_motion  # blobs with a smaller share of moving pixels are furniture
        self._prev: np.ndarray | None = None
        self._prev_blob: np.ndarray | None = None
        self._last_motion: np.ndarray | None = None
        self._first = 0.0
        self.person: np.ndarray | None = None   # bool, MOTION_WIDTH wide: blobs counted as a person
        self.static: np.ndarray | None = None   # bool: blobs ignored as furniture
        self.person_coverage = 0.0              # fractions of the frame
        self.static_coverage = 0.0

    def update(self, frame: np.ndarray, mask: np.ndarray, now: float) -> float:
        """Returns the fraction of the frame covered by blobs that count as a person."""
        h, w = frame.shape[:2]
        size = (min(MOTION_WIDTH, w), max(1, round(h * min(MOTION_WIDTH, w) / w)))
        gray = cv2.cvtColor(cv2.resize(frame, size, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        blob = cv2.resize(np.ascontiguousarray(mask, dtype=np.float32), size, interpolation=cv2.INTER_AREA) > PERSON_THRESHOLD
        if self._prev is None or self._prev.shape != gray.shape:
            self._last_motion = np.full(gray.shape, -1e9)
            self._first = now
        else:
            # auto exposure moves the whole picture a little from frame to frame: compare at equal mean
            prev = self._prev * (float(gray.mean()) / max(float(self._prev.mean()), 1.0))
            moved = (np.abs(gray - prev) > MOTION_LEVELS) | (blob != self._prev_blob)
            self._last_motion[moved] = now
        self._prev, self._prev_blob = gray, blob
        recent = (now - self._last_motion) < MOTION_WINDOW_S
        observed = now - self._first >= MOTION_WINDOW_S
        person, static = np.zeros(blob.shape, bool), np.zeros(blob.shape, bool)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(blob.astype(np.uint8), connectivity=8)
        for i in range(1, n):
            if stats[i, cv2.CC_STAT_AREA] < MIN_BLOB * blob.size:
                continue
            x, y, bw, bh = (int(v) for v in stats[i, :4])
            part = labels[y:y + bh, x:x + bw] == i
            still = observed and float(recent[y:y + bh, x:x + bw][part].mean()) < self.max_static_motion
            (static if still else person)[y:y + bh, x:x + bw] |= part
        self.person, self.static = person, static
        self.person_coverage, self.static_coverage = float(person.mean()), float(static.mean())
        return self.person_coverage


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
    capturing (N frames averaged) -> done, or aborted if the segmenter still sees a person. Segmenter
    blobs that do not move (furniture, see PersonMotion) are not a person. `start(force=True)` skips
    the person check altogether.
    The vcam output is untouched: this only produces preview text via `banner()` and audio-cue
    names via `pop_events()` ("tick" for each countdown second, "capturing", "error" on abort).
    """

    IDLE, COUNTDOWN, SETTLE, CAPTURING, DONE, ABORTED = (
        "idle", "countdown", "settle", "capturing", "done", "aborted")

    def __init__(self, frames: int, countdown_s: float, settle_s: float, max_coverage: float,
                 max_static_motion: float = 0.02):
        self.frames, self.countdown_s, self.settle_s, self.max_coverage = frames, countdown_s, settle_s, max_coverage
        self.max_static_motion = max_static_motion
        self.force = False
        self.static_coverage = 0.0   # largest furniture coverage ignored during the capture
        self.raw_coverage = 0.0      # largest plain segmenter coverage during the capture (the old check)
        self.abort_frame: np.ndarray | None = None  # what was seen when the capture aborted, for the debug image
        self._motion = PersonMotion(max_static_motion)
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

    def start(self, now: float, force: bool = False) -> None:
        self.state, self.abort_reason, self.peak_coverage = self.COUNTDOWN, "", 0.0
        self.force, self.static_coverage, self.raw_coverage, self.abort_frame = force, 0.0, 0.0, None
        self._motion = PersonMotion(self.max_static_motion)
        if force:
            log.warning("Plate capture FORCED (Shift+B): the person check is off. Anything in the frame, "
                        "you included, ends up in the plate.")
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
        if mask is not None and not self.force and self.active:
            self._motion.update(frame, mask, now)  # watched from the first countdown frame on
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
                self.raw_coverage = max(self.raw_coverage, person_coverage(mask))
            if mask is not None and not self.force:
                coverage = self._motion.person_coverage  # moving blobs only; still furniture is not a person
                self.peak_coverage = max(self.peak_coverage, coverage)
                self.static_coverage = max(self.static_coverage, self._motion.static_coverage)
                if coverage > self.max_coverage:
                    self.state = self.ABORTED
                    self.abort_reason = (f"Person still in frame ({coverage:.1%} of pixels moving, "
                                         f"{self.max_coverage:.1%} allowed). Plate NOT saved. Press B to retry, "
                                         "Shift+B to capture without this check.")
                    self._phase_start = now
                    self._events.append("error")
                    self.abort_frame = frame.copy()
                    log.warning("Plate capture aborted: %s (segmenter coverage %.1f%%, of which %.1f%% still and ignored)",
                                self.abort_reason, self.raw_coverage * 100, self._motion.static_coverage * 100)
                    return
            self._acc.add(frame)
            if self._acc.count >= self.frames:
                self.state = self.DONE

    def result(self) -> np.ndarray:
        return self._acc.result()

    def abort_image(self) -> np.ndarray | None:
        """The frame the capture aborted on, with what was detected drawn in: red = counted as a person
        (it moved), yellow = segmenter blob ignored as furniture (it did not). None unless aborted."""
        if self.abort_frame is None or self._motion.person is None:
            return None
        img = self.abort_frame.copy()
        size = (img.shape[1], img.shape[0])
        for region, colour in ((self._motion.static, (0, 255, 255)), (self._motion.person, (0, 0, 255))):
            m = cv2.resize(region.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            tint = np.empty_like(img)
            tint[:] = colour
            cv2.copyTo(cv2.addWeighted(img, 0.5, tint, 0.5, 0), m, img)
            contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, contours, -1, colour, 2)
        text = (f"capture aborted: red = person (moving) {self._motion.person_coverage:.1%} | "
                f"yellow = still, ignored {self._motion.static_coverage:.1%}")
        cv2.putText(img, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(img, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        return img

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


def save_abort_image(capture: PlateCapture, directory: Path) -> Path | None:
    """Write the aborted capture's debug image (LOCAL-ONLY: it shows the room and maybe you; images
    under docs/evidence/ are gitignored and the pre-commit hook refuses them). Returns the path."""
    img = capture.abort_image()
    if img is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ABORT_IMAGE_PREFIX}-{datetime.now().strftime('%Y%m%d-%H%M%S')}.png"
    if not cv2.imwrite(str(path), img):
        raise OSError(f"could not write {path}")
    return path
