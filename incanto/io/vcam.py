"""Virtual camera output via pyvirtualcam."""
from __future__ import annotations

import logging

import cv2
import numpy as np
import pyvirtualcam

log = logging.getLogger(__name__)


class VCamError(RuntimeError):
    """Raised when the virtual camera cannot be started."""


class VirtualCamera:
    def __init__(self, width: int, height: int, fps: int, backend: str = "obs"):
        self.width, self.height = width, height
        try:
            self._cam = pyvirtualcam.Camera(
                width=width, height=height, fps=fps,
                fmt=pyvirtualcam.PixelFormat.RGB, backend=backend,
            )
        except Exception as exc:
            raise VCamError(
                f"Could not start the '{backend}' virtual camera: {exc}\n"
                "  - Is OBS Studio installed? Its virtual camera driver is required "
                "(open OBS once, click 'Start Virtual Camera', then stop it and close OBS, "
                "to register the driver).\n"
                "  - Is another app (including OBS itself) already using the virtual camera? "
                "Stop OBS's virtual camera and close it."
            ) from exc
        self.device = self._cam.device
        log.info("Virtual camera started: %s (%dx%d @ %d fps, backend=%s)",
                 self.device, width, height, fps, backend)

    def send(self, frame_bgr: np.ndarray) -> None:
        if frame_bgr.shape[1] != self.width or frame_bgr.shape[0] != self.height:
            raise ValueError(
                f"frame is {frame_bgr.shape[1]}x{frame_bgr.shape[0]}, "
                f"virtual camera expects {self.width}x{self.height}"
            )
        self._cam.send(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

    def close(self) -> None:
        self._cam.close()
