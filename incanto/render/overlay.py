"""Debug HUD + our own landmark drawing, for the preview window only.

The preview may be smaller than the camera frame (`preview.scale`): every function here takes the
preview image and scales its text and positions by `image width / 1280`.
"""
from __future__ import annotations

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from incanto.perception.hands import Hand

_CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]
_GREEN = (0, 255, 0)
_BLACK = (0, 0, 0)
REFERENCE_WIDTH = 1280   # text sizes and positions below are for a frame this wide
MIN_FONT_SCALE = 0.45    # smaller than this is unreadable


def _k(img: np.ndarray) -> float:
    return img.shape[1] / REFERENCE_WIDTH


def _font(img: np.ndarray, scale: float, thickness: int) -> tuple[float, int]:
    k = _k(img)
    return max(MIN_FONT_SCALE, scale * k), max(1, round(thickness * k))


def _put(img, text, org, scale, color, thickness):
    """Text with a black outline. The outline is the same text shifted by a pixel or two, not a thicker
    stroke: OpenCV spaces letters wider at a larger thickness, so a thick outline drifts away from
    the text (badly at small sizes)."""
    o = max(1, round(2 * _k(img)))
    x, y = org
    for dx, dy in ((-o, 0), (o, 0), (0, -o), (0, o)):
        cv2.putText(img, text, (x + dx, y + dy), cv2.FONT_HERSHEY_SIMPLEX, scale, _BLACK, thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _text(img, text, org, scale=0.7, color=_GREEN, thickness=2):
    """`org` is in the image's own pixels; `scale`/`thickness` are for a 1280-wide frame."""
    scale, thickness = _font(img, scale, thickness)
    _put(img, text, org, scale, color, thickness)


def _at(img: np.ndarray, x: float, y: float) -> tuple[int, int]:
    """A position given for a 1280-wide frame; negative y counts from the bottom edge."""
    k = _k(img)
    return round(x * k), round(y * k) if y >= 0 else img.shape[0] + round(y * k)


def draw_fps(img: np.ndarray, fps: float) -> None:
    _text(img, f"{fps:5.1f} FPS", _at(img, 16, 36), scale=1.0)


def tint_mask(img: np.ndarray, mask: np.ndarray, strength: float = 0.45) -> None:
    """Add a green tint where the person mask is set (in place; one cheap channel op)."""
    if mask.shape[:2] != img.shape[:2]:
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    green = np.ascontiguousarray(img[:, :, 1])
    cv2.add(green, (mask * (255 * strength)).astype(np.uint8), dst=green)
    img[:, :, 1] = green


def draw_hands(img: np.ndarray, hands: list[Hand]) -> None:
    h, w = img.shape[:2]
    k = _k(img)
    for hand in hands:
        pts = (hand.landmarks[:, :2] * (w, h)).astype(np.int32)
        for a, b in _CONNECTIONS:
            cv2.line(img, tuple(int(v) for v in pts[a]), tuple(int(v) for v in pts[b]),
                     (255, 200, 0), max(1, round(2 * k)), cv2.LINE_AA)
        for p in pts:
            cv2.circle(img, (int(p[0]), int(p[1])), max(2, round(4 * k)), (0, 0, 255), -1, cv2.LINE_AA)
        x, y = int(pts[0][0]), int(pts[0][1])  # wrist
        _text(img, f"{hand.handedness} {hand.score:.2f}", (x - round(40 * k), y + round(30 * k)), scale=0.6,
              color=(255, 255, 0))


def draw_debug(img: np.ndarray, mask: np.ndarray | None, hands: list[Hand],
               stage_ms: dict[str, float] | None = None) -> None:
    """Draw the full debug HUD onto `img`, which MUST be a copy (preview only)."""
    if mask is not None:
        tint_mask(img, mask)
    draw_hands(img, hands)
    if stage_ms:
        line = "  ".join(f"{k} {v:.0f}ms" for k, v in stage_ms.items())
        _text(img, line, _at(img, 16, -16), scale=0.6)


def draw_banner(img: np.ndarray, text: str, warning: bool = False) -> None:
    """Large centered message for the preview only (plate countdown, capture warnings)."""
    h, w = img.shape[:2]
    k = _k(img)
    color = (60, 60, 255) if warning else (0, 255, 255)
    scale, thickness = _font(img, 0.8 if warning else 1.6, 2 if warning else 3)
    # wrap long warnings onto several lines
    words, lines, line = text.split(), [], ""
    for word in words:
        trial = f"{line} {word}".strip()
        if cv2.getTextSize(trial, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > w - round(80 * k) and line:
            lines.append(line)
            line = word
        else:
            line = trial
    lines.append(line)
    step = int(52 * scale + 6)
    y = h // 2 - (len(lines) - 1) * step // 2
    for ln in lines:
        size = cv2.getTextSize(ln, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0]
        _put(img, ln, ((w - size[0]) // 2, y), scale, color, thickness)
        y += step


def draw_status(img: np.ndarray, text: str, row: int = 0) -> None:
    """Small status line under the FPS counter (preview only)."""
    _text(img, text, _at(img, 16, 70 + row * 30), scale=0.7, color=(0, 200, 255))


_SOURCE_COLORS = np.array([(0, 0, 0), (0, 255, 0), (255, 128, 0), (0, 255, 255), (0, 0, 255)], np.uint8)  # BGR


def draw_vanish_sources(img: np.ndarray, result, strength: float = 0.55) -> None:
    """Colour-code which source covered each pixel of the vanish mask (preview only).

    green = segmenter, yellow = segmenter dilation margin, blue = hand hull, red = plate difference
    (only pixels no earlier source covered, so red shows what the plate difference alone adds).
    """
    thr = 0.5
    h, w = img.shape[:2]
    # a preview smaller than the masks does not need every mask pixel: read every 2nd (a quarter of the work)
    st = 2 if w < result.seg.shape[1] * 1.5 else 1
    seg, grown, hull, diff = (a[::st, ::st] for a in (result.seg, result.grown, result.hull, result.diff))
    # one label per pixel, assigned lowest priority first so that seg > hull > grow > plate-diff
    label = np.zeros(seg.shape, np.uint8)
    label[diff > thr] = 4
    label[grown > thr] = 3
    label[hull > thr] = 2
    label[seg > thr] = 1
    x, y, bw, bh = cv2.boundingRect(label)  # blend only inside the box around everything covered
    if bw > 0:
        crop = label[y:y + bh, x:x + bw]
        layer = _SOURCE_COLORS[crop]
        fx, fy = w / label.shape[1], h / label.shape[0]
        X0, Y0, X1, Y1 = int(x * fx), int(y * fy), min(w, int((x + bw) * fx)), min(h, int((y + bh) * fy))
        size = (X1 - X0, Y1 - Y0)
        if size[0] > 0 and size[1] > 0:
            layer = cv2.resize(layer, size, interpolation=cv2.INTER_NEAREST)
            covered = cv2.resize((crop > 0).astype(np.uint8), size, interpolation=cv2.INTER_NEAREST)
            region = img[Y0:Y1, X0:X1]
            cv2.copyTo(cv2.addWeighted(region, 1 - strength, layer, strength, 0), covered, region)
    if result.zone.any():  # outline of the "near me" zone: plate difference only counts inside it
        zone = (result.zone[::st, ::st] > 0.5).astype(np.uint8)
        contours, _ = cv2.findContours(zone, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        sx, sy = w / zone.shape[1], h / zone.shape[0]
        for c in contours:
            cv2.polylines(img, [(c * (sx, sy)).astype(np.int32)], True, (255, 0, 255), max(1, round(2 * _k(img))),
                          cv2.LINE_AA)
    _text(img, "green seg | yellow grow | blue hands | red plate-diff | magenta = near-me zone",
          _at(img, 16, -46), scale=0.55)


def draw_leak(img: np.ndarray, out: np.ndarray, plate_roi: np.ndarray,
              roi: tuple[int, int, int, int], gain: int = 4) -> None:
    """Leak view (preview only): |vcam output - matched plate| amplified and shown in red.

    Where you vanished the output IS the matched plate, so anything red there is you leaking through.
    Only the vanish region (`roi` in `out`'s pixels, where `plate_roi` is the matched plate) is
    examined; the rest of the frame is the live room and is left alone. `img` may be a scaled-down
    preview of `out`."""
    x0, y0, x1, y1 = roi
    ch = cv2.split(cv2.absdiff(out[y0:y1, x0:x1], plate_roi))
    d = cv2.convertScaleAbs(cv2.max(cv2.max(ch[0], ch[1]), ch[2]), alpha=gain)
    fx, fy = img.shape[1] / out.shape[1], img.shape[0] / out.shape[0]
    X0, Y0, X1, Y1 = round(x0 * fx), round(y0 * fy), min(img.shape[1], round(x1 * fx)), min(img.shape[0], round(y1 * fy))
    if X1 > X0 and Y1 > Y0:
        if (X1 - X0, Y1 - Y0) != (x1 - x0, y1 - y0):
            # shrink with a max filter first, so a thin leak does not disappear in the smaller preview
            step = max(1, round(1 / max(fx, 1e-6)))
            if step > 1:
                d = cv2.dilate(d, np.ones((step, step), np.uint8))
            d = cv2.resize(d, (X1 - X0, Y1 - Y0), interpolation=cv2.INTER_NEAREST)
        part = img[Y0:Y1, X0:X1]
        part[:, :, 2] = cv2.add(part[:, :, 2], d)
        part[:, :, 0] = cv2.subtract(part[:, :, 0], d)
        part[:, :, 1] = cv2.subtract(part[:, :, 1], d)
    _text(img, f"LEAK VIEW: red = |output - matched plate| x{gain}; should be none where you were",
          _at(img, 16, -46), scale=0.5, color=(0, 200, 255))
