"""Blends frame, mask, plate, and effect layers."""
from __future__ import annotations

import cv2
import numpy as np


def composite(frame: np.ndarray, mask: np.ndarray, plate: np.ndarray) -> np.ndarray:
    """Replace the person with the plate, blending with the soft float mask (no hard edges).

    out = frame * (1 - mask) + plate * mask, where `mask` is float32 in 0..1 at frame resolution.
    Pass a uint8 plate (`Plate.u8`) for the fast path (cv2.blendLinear); a float plate is blended in float.
    """
    if frame.shape != plate.shape:
        raise ValueError(f"plate {plate.shape} does not match frame {frame.shape}")
    mask = np.ascontiguousarray(mask, dtype=np.float32)
    if plate.dtype == np.uint8 and frame.dtype == np.uint8:
        return cv2.blendLinear(frame, plate, 1.0 - mask, mask)
    m = mask[:, :, None]
    out = frame.astype(np.float32) * (1.0 - m) + plate.astype(np.float32) * m
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


class Compositor:
    """Builds the frame sent to the virtual camera.

    `dt` (seconds since the previous frame) is part of the signature so effects can be time-based,
    not frame-based. With `vanish=True` and a plate, the person is replaced by the clean plate.
    """

    def render(self, frame: np.ndarray, mask: np.ndarray | None, plate: np.ndarray | None,
               dt: float, vanish: bool = False) -> np.ndarray:
        if vanish and mask is not None and plate is not None:
            return composite(frame, mask, plate)
        return frame
