"""Webcam capture via OpenCV."""
from __future__ import annotations

import atexit
import logging
import threading

import cv2
import numpy as np

log = logging.getLogger(__name__)

# Safety net: the driver REMEMBERS manual exposure across processes, so a crash while we hold manual
# exposure could leave the webcam dark in every app. Every Camera that switched to manual is tracked
# here and put back to auto by an atexit hook (and by main's try/finally and signal handlers).
_MANUAL_CAMERAS: set["Camera"] = set()
_MANUAL_LOCK = threading.Lock()


def restore_auto_exposure_all(reason: str = "exit") -> int:
    """Put every camera we left on manual exposure back to auto. Safe to call repeatedly; returns count."""
    with _MANUAL_LOCK:
        cams = list(_MANUAL_CAMERAS)
    restored = 0
    for cam in cams:
        try:
            if cam.set_auto_exposure():
                restored += 1
            log.info("Safety net (%s): restored auto exposure on camera %d.", reason, cam.index)
        except Exception as exc:  # noqa: BLE001 - never raise from an exit path
            log.error("Safety net (%s): could NOT restore auto exposure on camera %d: %s. "
                      "Run: python -m incanto --reset-camera", reason, cam.index, exc)
    return restored


atexit.register(restore_auto_exposure_all, "atexit")


def install_exit_guard() -> None:
    """Restore auto exposure on SIGTERM/SIGBREAK and on console close (Windows); Ctrl+C is covered by
    the KeyboardInterrupt path. Call once from the entry point (main thread only)."""
    import signal

    def _on_signal(signum, _frame):
        restore_auto_exposure_all(f"signal {signum}")
        raise SystemExit(1)

    for name in ("SIGTERM", "SIGBREAK"):
        sig = getattr(signal, name, None)
        if sig is not None:
            try:
                signal.signal(sig, _on_signal)
            except (ValueError, OSError):
                pass
    try:  # closing the console window kills the process without running atexit
        import ctypes
        from ctypes import wintypes

        handler_t = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.DWORD)

        def _console_handler(ctrl_type):
            if ctrl_type in (2, 5, 6):  # CTRL_CLOSE / LOGOFF / SHUTDOWN
                restore_auto_exposure_all(f"console event {ctrl_type}")
            return False  # let the default handling continue

        global _CONSOLE_HANDLER
        _CONSOLE_HANDLER = handler_t(_console_handler)  # keep a reference alive
        ctypes.windll.kernel32.SetConsoleCtrlHandler(_CONSOLE_HANDLER, True)
    except (AttributeError, OSError):
        pass

EXPOSURE_MIN, EXPOSURE_MAX = -13.0, 0.0
# Exposure is log2(seconds) on this backend: -5 = 31 ms, about one 30 fps frame period. Longer than
# that and the camera must lower its frame rate (measured: auto exposure in a dim room -> 15 fps).
EXPOSURE_FPS_LIMIT = -5.0
_AUTO = "auto"  # ThreadedCamera: pending request to return to auto exposure


class CameraError(RuntimeError):
    """Raised when the webcam cannot be opened or read."""


class Camera:
    def __init__(self, width: int, height: int, fps: int, index: int = 0,
                 backends: tuple[int, ...] = (cv2.CAP_MSMF, cv2.CAP_DSHOW)):
        self.index = index
        self._manual_exposure = False  # True once we have switched the driver to manual exposure
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
        """Properties as the backend reports them. NOTE: MSMF's exposure/auto-exposure readback is
        stale (always the same value no matter what was set), so never use it to verify a change."""
        return {
            "exposure": self._cap.get(cv2.CAP_PROP_EXPOSURE),
            "auto_exposure": self._cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            "fps": self._cap.get(cv2.CAP_PROP_FPS),
        }

    def mean_brightness(self, frames: int = 8) -> float:
        """Mean pixel value (0-255) of a frame after reading `frames` frames (lets exposure settle)."""
        frame = None
        for _ in range(frames):
            frame = self.read()
        return float(frame.mean())

    def set_exposure(self, value: float) -> bool:
        """Switch to manual exposure at `value` (log2 seconds). True if the driver accepted both writes."""
        value = float(min(EXPOSURE_MAX, max(EXPOSURE_MIN, value)))
        self._manual_exposure = True
        with _MANUAL_LOCK:
            _MANUAL_CAMERAS.add(self)
        auto_ok = self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # 0.25 = manual (verified on this camera)
        exp_ok = self._cap.set(cv2.CAP_PROP_EXPOSURE, value)
        if value > EXPOSURE_FPS_LIMIT:
            log.warning("exposure %.0f is longer than one 30 fps frame period; the camera may drop below 30 fps",
                        value)
        return bool(auto_ok and exp_ok)

    def set_auto_exposure(self) -> bool:
        """Hand exposure back to the camera. The driver REMEMBERS manual exposure across processes, so
        this must be done explicitly (0.75 = auto; verified: it restores normal brightness)."""
        ok = bool(self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75))
        self._manual_exposure = False
        with _MANUAL_LOCK:
            _MANUAL_CAMERAS.discard(self)
        return ok

    def apply_auto_exposure(self) -> bool:
        """set_auto_exposure + a before/after brightness check, for use at camera open."""
        import time
        before = self.mean_brightness()
        ok = self.set_auto_exposure()
        time.sleep(0.6)
        after = self.mean_brightness(10)
        log.info("exposure: requested auto | set() accepted=%s | mean brightness %.0f -> %.0f", ok, before, after)
        return ok

    def apply_exposure(self, value: float) -> bool:
        """set_exposure + a before/after brightness check, for use at camera open (single-threaded)."""
        before = self.mean_brightness()
        accepted = self.set_exposure(value)
        import time
        time.sleep(0.6)  # let the sensor settle
        after = self.mean_brightness(10)
        log.info("exposure: requested manual %.0f | set() accepted=%s | mean brightness %.0f -> %.0f "
                 "(MSMF readback is unreliable, so brightness is the check)", value, accepted, before, after)
        return accepted

    def release(self) -> None:
        if self._manual_exposure:
            # Don't leave the driver stuck on a manual exposure for the next app (e.g. Meet using the webcam).
            self.set_auto_exposure()
            log.info("Restored auto exposure on the camera.")
        with _MANUAL_LOCK:
            _MANUAL_CAMERAS.discard(self)
        self._cap.release()


class ThreadedCamera:
    """Wraps a Camera: a background thread keeps reading, `read()` returns the newest frame.

    Camera waits and decode time then overlap with perception/render work on the main thread.
    `read()` blocks until a frame newer than the last one returned is available, so no frame is
    processed twice and none is returned stale.
    """

    def __init__(self, cam: Camera):
        import threading
        from incanto.utils.fps import StageTimer

        self._cam = cam
        self.width, self.height = cam.width, cam.height
        self.timer = StageTimer()  # grab/retrieve timings, measured on the capture thread
        self._cond = threading.Condition()
        self._frame: np.ndarray | None = None
        self._seq = 0
        self._last_seq = 0
        self._pending_exposure: float | str | None = None  # a number = manual, _AUTO = back to auto
        self._frame_time = 0.0
        self.frame_time = 0.0  # perf_counter() when the frame last returned by read() arrived
        self._error: Exception | None = None
        self._stop = False
        self._delivered = 0
        self._delivered_since = None
        self._thread = threading.Thread(target=self._run, name="camera-capture", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        import time
        try:
            while not self._stop:
                pending, self._pending_exposure = self._pending_exposure, None
                if pending == _AUTO:  # apply between reads: VideoCapture is not thread-safe
                    log.info("exposure set to auto (accepted=%s)", self._cam.set_auto_exposure())
                elif pending is not None:
                    ok = self._cam.set_exposure(pending)
                    log.info("exposure set to %.0f (accepted=%s)", pending, ok)
                frame = self._cam.read(self.timer)
                with self._cond:
                    self._frame = frame
                    self._frame_time = time.perf_counter()
                    self._seq += 1
                    self._delivered += 1
                    if self._delivered_since is None:
                        self._delivered_since = time.perf_counter()
                    self._cond.notify_all()
        except Exception as exc:  # noqa: BLE001 - reported to the main thread via read()
            with self._cond:
                self._error = exc
                self._cond.notify_all()

    def read(self, timer=None) -> np.ndarray:
        if timer is not None:
            with timer.stage("wait_frame"):
                return self._read()
        return self._read()

    def _read(self) -> np.ndarray:
        with self._cond:
            ok = self._cond.wait_for(lambda: self._seq > self._last_seq or self._error, timeout=5.0)
            if self._error:
                raise CameraError(f"Capture thread stopped: {self._error}")
            if not ok:
                raise CameraError("Timed out waiting for a frame from the webcam.")
            self._last_seq = self._seq
            self.frame_time = self._frame_time
            return self._frame

    def pop_delivery_fps(self) -> float:
        """Frames the camera delivered per second since the previous call (independent of the loop)."""
        import time
        with self._cond:
            now = time.perf_counter()
            fps = 0.0
            if self._delivered_since is not None and now > self._delivered_since:
                fps = self._delivered / (now - self._delivered_since)
            self._delivered, self._delivered_since = 0, now
            return fps

    def exposure_state(self) -> dict[str, float]:
        return self._cam.exposure_state()

    def request_exposure(self, value: float) -> None:
        """Ask the capture thread to switch to manual exposure `value` before its next read."""
        self._pending_exposure = value

    def request_auto_exposure(self) -> None:
        """Ask the capture thread to hand exposure back to the camera before its next read."""
        self._pending_exposure = _AUTO

    def release(self) -> None:
        self._stop = True
        try:
            self._thread.join(timeout=3.0)
        finally:
            self._cam.release()  # restores auto exposure if we left it manual
