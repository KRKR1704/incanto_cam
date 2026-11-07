"""Webcam capture via OpenCV."""
from __future__ import annotations

import logging

import cv2
import numpy as np

log = logging.getLogger(__name__)


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class Camera:
    def __init__(self, width: int, height: int, fps: int, index: int = 0,
                 backends: tuple[int, ...] = (cv2.CAP_MSMF, cv2.CAP_DSHOW)):
        self.index = index
        # Media Foundation first: on the dev laptop DirectShow was capped at ~5 fps at 720p
        # (it ignores the MJPG request), while MSMF delivers the full 30 fps.
        self._cap = None
        for backend in backends:
            cap = cv2.VideoCapture(index, backend)
            if cap.isOpened():
                self._cap = cap
                break
            cap.release()
        if self._cap is None:
            raise CameraError(
                f"No webcam found at index {index}. Check that a camera is connected, "
                "not in use by another app (close Meet/Zoom/OBS), and that Windows camera "
                "privacy settings allow desktop apps."
            )
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self.width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        log.info("Camera %d opened: %dx%d @ %.1f fps (requested %dx%d @ %d)",
                 index, self.width, self.height, self.fps, width, height, fps)
        if (self.width, self.height) != (width, height):
            log.warning("Camera did not accept %dx%d; using %dx%d", width, height,
                        self.width, self.height)

    def read(self, timer=None) -> np.ndarray:
        """Return the next BGR frame, raising CameraError if the camera stops delivering.

        With a StageTimer, records "grab" (waiting for the camera) and "retrieve"
        (decode/convert) separately.
        """
        if timer is None:
            ok = self._cap.grab()
            ok, frame = self._cap.retrieve() if ok else (False, None)
        else:
            with timer.stage("grab"):
                ok = self._cap.grab()
            with timer.stage("retrieve"):
                ok, frame = self._cap.retrieve() if ok else (False, None)
        if not ok or frame is None:
            raise CameraError("Failed to read a frame from the webcam (disconnected or in use?).")
        return frame

    def show_driver_settings(self) -> bool:
        """Ask the driver to show its own property dialog (DirectShow only)."""
        return bool(self._cap.set(cv2.CAP_PROP_SETTINGS, 1))

    def exposure_state(self) -> dict[str, float]:
        """Current exposure-related properties as reported by the backend."""
        return {
            "exposure": self._cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_exposure": self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
        }

    def lock_exposure(self, value: float) -> bool:
        """Disable auto-exposure and set a fixed exposure. Returns True if it read back as set."""
        before = self.exposure_state()
        # 0.25 is the OpenCV convention for "manual"; some backends use 0 or 1.
        auto_ret = self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
        exp_ret = self._cap.set(cv2.CAP_PROP_EXPOSURE, value)
        after = self.exposure_state()
        accepted = bool(exp_ret) and abs(after["exposure"] - value) < 1e-6
        log.info("lock-exposure: requested exposure=%s | before=%s | set(auto=0.25)->%s set(exposure)->%s | after=%s | ACCEPTED=%s",
                 value, before, auto_ret, exp_ret, after, accepted)
        return accepted

    def release(self) -> None:
        self._cap.release()
