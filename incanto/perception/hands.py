"""MediaPipe hand landmark tracking."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from incanto.utils.clock import MonotonicMs


@dataclass(frozen=True)
class Hand:
    """One tracked hand. No MediaPipe types leak out of this module."""
    handedness: str          # "Left" or "Right", as reported by MediaPipe (see note below)
    score: float             # handedness confidence 0..1
    landmarks: np.ndarray    # (21, 3) float32; x, y normalized to the frame (0..1), z relative depth


# Note: MediaPipe documents that it expects a mirrored ("selfie view") image. Checked on our
# unmirrored camera frames (docs/evidence/phase-2/): the label matches the person's real hand
# (their right hand appears on the image-left and is labelled "Right").


class HandTracker:
    def __init__(self, model_path, *, num_hands: int = 2, min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._clock = MonotonicMs()
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ))

    def detect(self, small_bgr: np.ndarray) -> list[Hand]:
        """Detect hands in the downscaled BGR perception frame."""
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._clock.now())
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            hands.append(Hand(handed[0].category_name, float(handed[0].score), pts))
        return hands

    def close(self) -> None:
        self._landmarker.close()


@dataclass(frozen=True)
class HandsResult:
    """Most recent hand-tracking result, stamped with the capture time of the frame it came from."""
    hands: list[Hand]
    frame_time: float        # time.perf_counter() when that frame was captured
    infer_ms: float          # how long the HandLandmarker call took
    seq: int                 # increments with every published result (0 = nothing yet)


class HandsWorker:
    """Runs HandTracker on its own thread, always on the LATEST submitted frame.

    `submit()` replaces any pending frame (stale frames are dropped, nothing queues up) and
    never blocks. `latest()` returns the newest published result immediately; the main loop
    never waits on hand tracking. Result age = now - result.frame_time.
    """

    def __init__(self, tracker: HandTracker):
        import threading

        self._tracker = tracker
        self._cond = threading.Condition()
        self._pending: tuple[np.ndarray, float] | None = None
        self._result = HandsResult([], 0.0, 0.0, 0)
        self._stop = False
        self._error: Exception | None = None
        self._published = 0
        self._published_since: float | None = None
        self._infer_total = 0.0
        self._thread = threading.Thread(target=self._run, name="hands-worker", daemon=True)
        self._thread.start()

    def submit(self, small_bgr: np.ndarray, frame_time: float) -> None:
        with self._cond:
            self._pending = (small_bgr, frame_time)  # overwrite: drop the stale frame
            self._cond.notify()

    def latest(self) -> HandsResult:
        if self._error:
            raise RuntimeError(f"hands worker stopped: {self._error}") from self._error
        return self._result

    def _run(self) -> None:
        import time
        try:
            while True:
                with self._cond:
                    self._cond.wait_for(lambda: self._pending is not None or self._stop)
                    if self._stop:
                        return
                    frame, frame_time = self._pending
                    self._pending = None
                t0 = time.perf_counter()
                hands = self._tracker.detect(frame)
                done = time.perf_counter()
                with self._cond:
                    self._result = HandsResult(hands, frame_time, 1000 * (done - t0), self._result.seq + 1)
                    self._published += 1
                    self._infer_total += done - t0
                    if self._published_since is None:
                        self._published_since = t0
        except Exception as exc:  # noqa: BLE001 - surfaced to the main thread via latest()
            self._error = exc

    def pop_stats(self) -> tuple[float, float]:
        """(results per second, mean inference ms) since the previous call."""
        import time
        with self._cond:
            now = time.perf_counter()
            hz = ms = 0.0
            if self._published and self._published_since is not None and now > self._published_since:
                hz = self._published / (now - self._published_since)
                ms = 1000 * self._infer_total / self._published
            self._published, self._infer_total, self._published_since = 0, 0.0, now
            return hz, ms

    def close(self) -> None:
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self._thread.join(timeout=3.0)
        self._tracker.close()
