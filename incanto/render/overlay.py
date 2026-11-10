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


def draw_banner(img: np.ndarray, text: str, warning: bool = False) -> None:
    """Large centered message for the preview only (plate countdown, capture warnings)."""
    h, w = img.shape[:2]
    color = (60, 60, 255) if warning else (0, 255, 255)
    scale = 0.8 if warning else 1.6
    # wrap long warnings onto several lines
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0][0] > w - 80 and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    y = h // 2 - (len(lines) - 1) * int(26 * scale + 10)
    for ln in lines:
        size = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)[0]
        _text(img, ln, ((w - size[0]) // 2, y), scale=scale, color=color, thickness=3 if not warning else 2)
        y += int(52 * scale)


def draw_status(img: np.ndarray, text: str, row: int = 0) -> None:
    """Small status line under the FPS counter (preview only)."""
    _text(img, text, (16, 70 + row * 30), scale=0.7, color=(0, 200, 255))


_SOURCE_COLORS = np.array([(0, 0, 0), (0, 255, 0), (255, 128, 0), (0, 255, 255), (0, 0, 255)], np.uint8)  # BGR


def draw_vanish_sources(img: np.ndarray, result, strength: float = 0.55) -> None:
    """Colour-code which source covered each pixel of the vanish mask (preview only).

    green = segmenter, yellow = segmenter dilation margin, blue = hand hull, red = plate difference
    (only pixels no earlier source covered, so red shows what the plate difference alone adds).
    """
    thr = 0.5
    # one label per pixel, assigned lowest priority first so that seg > hull > grow > plate-diff
    label = np.zeros(result.seg.shape, np.uint8)
    label[result.diff > thr] = 4
    label[result.grown > thr] = 3
    label[result.hull > thr] = 2
    label[result.seg > thr] = 1
    h, w = img.shape[:2]
    x, y, bw, bh = cv2.boundingRect(label)  # blend only inside the box around everything covered
    if bw > 0:
        crop = label[y:y + bh, x:x + bw]
        layer = _SOURCE_COLORS[crop]
        fx, fy = w / label.shape[1], h / label.shape[0]
        X0, Y0, X1, Y1 = int(x * fx), int(y * fy), min(w, int((x + bw) * fx)), min(h, int((y + bh) * fy))
        size = (X1 - X0, Y1 - Y0)
        layer = cv2.resize(layer, size, interpolation=cv2.INTER_NEAREST)
        covered = cv2.resize((crop > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
        region = img[Y0:Y1, X0:X1]
        cv2.copyTo(cv2.addWeighted(region, 1 - strength, layer, strength, 0), covered, region)
    if result.zone.any():  # outline of the "near me" zone: plate difference only counts inside it
        zone = (result.zone > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w / zone.shape[1], h / zone.shape[0]
        for c in contours:
            cv2.polylines(img, [(c * (sx, sy)).astype(np.int32)], True, (255, 0, 255), 2, cv2.LINE_AA)
    _text(img, "green seg | yellow grow | blue hands | red plate-diff | magenta = near-me zone",
          (16, img.shape[0] - 46), scale=0.55)


def draw_leak(img: np.ndarray, out: np.ndarray, plate_roi: np.ndarray,
              roi: tuple[int, int, int, int], gain: int = 4) -> None:
    """Leak view (preview only): |vcam output - matched plate| amplified and shown in red.

    Where you vanished the output IS the matched plate, so anything red there is you leaking through.
    Only the vanish region (`roi`, where `plate_roi` is the matched plate) is examined; the rest of the
    frame is the live room and is left alone."""
    x0, y0, x1, y1 = roi
    ch = cv2.split(cv2.absdiff(out[y0:y1, x0:x1], plate_roi))
    d = cv2.convertScaleAbs(cv2.max(cv2.max(ch[0], ch[1]), ch[2]), alpha=gain)
    part = img[y0:y1, x0:x1]
    part[:, :, 2] = cv2.add(part[:, :, 2], d)
    part[:, :, 0] = cv2.subtract(part[:, :, 0], d)
    part[:, :, 1] = cv2.subtract(part[:, :, 1], d)
    _text(img, f"LEAK VIEW: red = |output - matched plate| x{gain} inside the vanish region; should be none where you were",
          (16, img.shape[0] - 46), scale=0.5, color=(0, 200, 255))
