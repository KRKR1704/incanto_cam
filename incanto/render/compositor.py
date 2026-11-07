"""Blends frame, mask, plate, and effect layers."""
from __future__ import annotations

import numpy as np


class Compositor:
    """Builds the frame sent to the virtual camera.

    For now it is a pass-through: no effects exist yet. `dt` (seconds since the previous
    frame) is already part of the signature so effects can be time-based, not frame-based.
    """

    def render(self, frame: np.ndarray, mask: np.ndarray | None, plate: np.ndarray | None,
               dt: float) -> np.ndarray:
        return frame
