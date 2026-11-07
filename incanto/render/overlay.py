"""Debug HUD + our own landmark drawing, for the preview window only."""
from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from incanto.perception.hands import Hand

_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
_GREEN = (0, 255, 0)
_BLACK = (0, 0, 0)


def _text(img, text, org, scale=0.7, color=_GREEN, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_fps(img: np.ndarray, fps: float) -> None:
    _text(img, f"{fps:5.1f} FPS", (16, 36), scale=1.0)


def tint_mask(img: np.ndarray, mask: np.ndarray, strength: float = 0.45) -> None:
    """Add a green tint where the person mask is set (in place; one cheap channel op)."""
    green = np.ascontiguousarray(img[:, :, 1])
    cv2.add(green, (mask * (255 * strength)).astype(np.uint8), dst=green)
    img[:, :, 1] = green


def draw_hands(img: np.ndarray, hands: list[Hand]) -> None:
    h, w = img.shape[:2]
    for hand in hands:
        pts = (hand.landmarks[:, :2] * (w, h)).astype(np.int32)
        for a, b in _CONNECTIONS:
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                     (255, 200, 0), 2, cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, (int(p[0]), int(p[1])), 4, (0, 0, 255), -1, cv2.LINE_AA)
        x, y = int(pts[0][0]), int(pts[0][1])  # wrist
        _text(img, f"{hand.handedness} {hand.score:.2f}", (x - 40, y + 30), scale=0.6,
              color=(255, 255, 0))


def draw_debug(img: np.ndarray, mask: np.ndarray | None, hands: list[Hand],
               stage_ms: dict[str, float] | None = None) -> None:
    """Draw the full debug HUD onto `img`, which MUST be a copy (preview only)."""
    if mask is not None:
        tint_mask(img, mask)
    draw_hands(img, hands)
    if stage_ms:
        line = "  ".join(f"{k} {v:.0f}ms" for k, v in stage_ms.items())
        _text(img, line, (16, img.shape[0] - 16), scale=0.6)
