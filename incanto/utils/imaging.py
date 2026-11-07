"""Small image helpers shared by perception and rendering."""
from __future__ import annotations

import cv2
import numpy as np


def downscale(frame: np.ndarray, target_width: int) -> np.ndarray:
    """Resize to `target_width`, keeping the aspect ratio. Never upscales."""
    h, w = frame.shape[:2]
    if w <= target_width:
        return frame
    target_height = max(1, round(h * target_width / w))
    return cv2.resize(frame, (target_width, target_height), interpolation=cv2.INTER_AREA)
