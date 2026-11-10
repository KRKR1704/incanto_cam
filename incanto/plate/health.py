"""Plate health: does the stored plate still match the room the camera sees right now?

A stale plate (yesterday's light, moved furniture) cannot be hidden by brightness matching and shows
up as a "masked person" silhouette, so vanish must not run on it. The check compares the visible
background (everything away from the person) with the same pixels of the plate:

* raw diff       mean live level - mean plate level
* global gain    live / plate per colour channel
* max cell gain  the largest live / plate ratio of any grid cell and channel

It is a pure function of two images and a "covered" mask, cheap enough to run a few times a second.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

STRIDE = 4               # measured on every 4th pixel in each direction
COVERED = 0.02           # mask value above which a pixel belongs to the person
MIN_BACKGROUND = 200     # sampled background pixels needed for any verdict
MIN_CELL_BG = 0.15       # a grid cell needs this fraction of background pixels to be judged
MIN_CELL_LEVEL = 30.0    # ...and a plate channel at least this bright (a ratio of two dark values is noise)


@dataclass(frozen=True)
class HealthLimits:
    max_raw_diff: float = 20.0
    gain_min: float = 0.8
    gain_max: float = 1.25
    max_cell_gain: float = 1.5


@dataclass(frozen=True)
class PlateHealth:
    raw_diff: float                 # mean live - mean plate over the background, levels
    gain: tuple[float, float, float]  # global live / plate, BGR
    max_cell_gain: float            # 1.0 when no cell could be judged
    background: float               # fraction of the frame that counted as background
    reasons: tuple[str, ...]        # empty when healthy

    @property
    def ok(self) -> bool:
        return not self.reasons

    def describe(self) -> str:
        text = (f"raw diff {self.raw_diff:+.1f} | gain BGR {self.gain[0]:.2f}/{self.gain[1]:.2f}/{self.gain[2]:.2f} "
                f"| max cell gain {self.max_cell_gain:.2f} | background {self.background:.0%}")
        return text + (" | FAILED: " + "; ".join(self.reasons) if self.reasons else " | healthy")


def measure_plate_health(live: np.ndarray, plate: np.ndarray, covered: np.ndarray | None,
                         limits: HealthLimits, cells: tuple[int, int] = (8, 6),
                         margin_px: int = 0) -> PlateHealth | None:
    """Compare `live` with `plate` (same-size BGR uint8) over the background.

    `covered` (float 0..1, same size, or None) marks the person; it is grown by `margin_px` before
    being excluded. Returns None when too little background is visible to judge.
    """
    if live.shape != plate.shape:
        raise ValueError(f"live {live.shape} does not match plate {plate.shape}")
    sub = (slice(None, None, STRIDE), slice(None, None, STRIDE))
    lv, pv = live[sub].astype(np.float32), plate[sub].astype(np.float32)
    if covered is None:
        bg = np.ones(lv.shape[:2], bool)
    else:
        cov = (covered[sub] > COVERED).astype(np.uint8)
        r = round(margin_px / STRIDE)
        if r > 0:
            cov = cv2.dilate(cov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        bg = cov == 0
    if int(bg.sum()) < MIN_BACKGROUND:
        return None
    live_mean, plate_mean = lv[bg].mean(axis=0), pv[bg].mean(axis=0)
    raw_diff = float(live_mean.mean() - plate_mean.mean())
    gain = live_mean / np.maximum(plate_mean, 1.0)

    gx, gy = cells
    bgf = bg.astype(np.float32)
    k = bgf[..., None]
    live_c = cv2.resize(lv * k, (gx, gy), interpolation=cv2.INTER_AREA).reshape(gy, gx, 3)
    plate_c = cv2.resize(pv * k, (gx, gy), interpolation=cv2.INTER_AREA).reshape(gy, gx, 3)
    frac = cv2.resize(bgf, (gx, gy), interpolation=cv2.INTER_AREA).reshape(gy, gx)
    valid = (frac >= MIN_CELL_BG)[..., None] & (plate_c > MIN_CELL_LEVEL * frac[..., None])
    max_cell = float((live_c[valid] / plate_c[valid]).max()) if valid.any() else 1.0

    reasons = []
    if abs(raw_diff) > limits.max_raw_diff:
        reasons.append(f"raw diff {raw_diff:+.1f} beyond +/-{limits.max_raw_diff:g}")
    if gain.min() < limits.gain_min or gain.max() > limits.gain_max:
        reasons.append(f"gain {gain.min():.2f}..{gain.max():.2f} outside {limits.gain_min:g}-{limits.gain_max:g}")
    if max_cell > limits.max_cell_gain:
        reasons.append(f"cell gain {max_cell:.2f} above {limits.max_cell_gain:g}")
    return PlateHealth(raw_diff, tuple(float(g) for g in gain), max_cell, float(bgf.mean()), tuple(reasons))
