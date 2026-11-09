"""The mask used to replace the person with the clean plate.

Principle: over-covering is free (the plate IS the background), under-covering reveals the person.
Three sources are unioned (each can be switched off with `vanish.mask_mode`):

1. the segmenter mask, grown by `dilate_px`;
2. a filled convex hull of each hand's landmarks, grown by `hand_hull_dilate_px` (the segmentation
   model misses thin fingers, and landmarks are ~50 ms old);
3. pixels that differ from the brightness-matched plate (morphologically cleaned).

The plate is first matched to the live brightness/colour (per-channel gain, optionally offset) using
background pixels well outside the mask, because auto exposure re-adjusts when you step back in.

All mask work happens at perception resolution; the finished mask is upscaled once.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass

import cv2
import numpy as np

from incanto.config import VanishConfig
from incanto.perception.hands import Hand

REFERENCE_WIDTH = 1280  # config pixel sizes are defined at this frame width
COVERED = 0.02          # mask value above which a pixel counts as covered when choosing background
MATCH_STRIDE = 4        # background statistics are gathered on every 4th pixel in each direction
GAIN_LIMITS = (0.5, 2.0)
OFFSET_LIMITS = (-40.0, 40.0)
OUTLIER_LEVELS = 40.0   # background pixels further than this from the fit are ignored (leftover person)


@dataclass
class VanishResult:
    mask: np.ndarray        # float32 0..1 at full output resolution (use for compositing)
    plate: np.ndarray       # uint8 matched plate at full output resolution
    seg: np.ndarray         # perception-res float32 pieces, for the debug view
    grown: np.ndarray       # segmenter mask after dilation
    hull: np.ndarray
    diff: np.ndarray


class VanishStats:
    """Window of background-brightness measurements, logged every few seconds."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.live = self.raw_plate = self.matched_plate = self.fraction = 0.0
        self.gain = np.zeros(3)
        self.offset = np.zeros(3)
        self.n = 0
        self.diff_frames = self.diff_skipped = 0

    def add(self, live: float, raw_plate: float, matched_plate: float, fraction: float,
            gain: np.ndarray, offset: np.ndarray) -> None:
        self.live += live
        self.raw_plate += raw_plate
        self.matched_plate += matched_plate
        self.fraction += fraction
        self.gain += gain
        self.offset += offset
        self.n += 1


def _ellipse(radius: int) -> np.ndarray | None:
    if radius <= 0:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


class VanishMask:
    def __init__(self, cfg: VanishConfig, out_size: tuple[int, int], small_size: tuple[int, int]):
        self.cfg = cfg
        self.out_size = out_size
        self.small_size = small_size  # (w, h) of the perception frame
        s = small_size[0] / REFERENCE_WIDTH
        self._k_grow = _ellipse(round(cfg.dilate_px * s))
        self._k_hull = _ellipse(round(cfg.hand_hull_dilate_px * s))
        self._k_open = _ellipse(round(cfg.plate_diff_open_px * s))
        self._k_close = _ellipse(round(cfg.plate_diff_close_px * s))
        self._k_margin = _ellipse(max(1, round(cfg.match_margin_px * s / MATCH_STRIDE)))
        feather = round(cfg.feather_px * s)
        self._feather = (feather * 2 + 1) if feather > 0 else 0
        self._plate_u8: np.ndarray | None = None
        self._plate_small: np.ndarray | None = None
        self.stats = VanishStats()
        self._reset_match()

    # -- plate ---------------------------------------------------------------------------------
    def set_plate(self, plate_u8: np.ndarray) -> None:
        self._plate_u8 = plate_u8
        self._plate_small = cv2.resize(plate_u8, self.small_size, interpolation=cv2.INTER_AREA)
        self._reset_match()

    def _reset_match(self) -> None:
        self.gain = np.ones(3, np.float32)
        self.offset = np.zeros(3, np.float32)
        self._have_match = False

    # -- brightness matching -------------------------------------------------------------------
    def _lut(self) -> np.ndarray:
        x = np.arange(256, dtype=np.float32)[:, None]
        return np.clip(x * self.gain + self.offset + 0.5, 0, 255).astype(np.uint8).reshape(1, 256, 3)

    def _estimate_match(self, small: np.ndarray, covered: np.ndarray) -> None:
        """Update the running per-channel gain (and offset) from background far outside the mask."""
        sub = (slice(None, None, MATCH_STRIDE), slice(None, None, MATCH_STRIDE))
        cov = (covered[sub] > COVERED).astype(np.uint8)
        if self._k_margin is not None:
            cov = cv2.dilate(cov, self._k_margin)
        bg = cov == 0
        live = small[sub][bg].astype(np.float32)
        plate = self._plate_small[sub][bg].astype(np.float32)
        fraction = float(bg.mean())
        if len(live) < 200:  # hardly any background visible: keep the last estimate
            return
        mode = self.cfg.plate_match
        gain, offset = np.ones(3, np.float32), np.zeros(3, np.float32)
        keep = np.ones(len(live), bool)
        for _ in range(2):  # second pass drops outliers (leftover person, moved objects)
            lm, pm = live[keep].mean(0), plate[keep].mean(0)
            if mode == "gain_offset":
                var = ((plate[keep] - pm) ** 2).mean(0)
                cov_lp = ((plate[keep] - pm) * (live[keep] - lm)).mean(0)
                flat = var < 25.0  # nearly flat channel: slope is meaningless, fall back to gain only
                gain = np.where(flat, lm / np.maximum(pm, 1.0), cov_lp / np.maximum(var, 1e-3))
                offset = np.where(flat, 0.0, lm - gain * pm)
            else:
                gain = lm / np.maximum(pm, 1.0)
            gain = np.clip(gain, *GAIN_LIMITS).astype(np.float32)
            offset = np.clip(offset, *OFFSET_LIMITS).astype(np.float32)
            keep = np.abs(live - (plate * gain + offset)).max(1) < OUTLIER_LEVELS
            if keep.sum() < 100:
                keep[:] = True
                break
        if mode == "off":  # measure only: the plate is used unmodified
            gain, offset = np.ones(3, np.float32), np.zeros(3, np.float32)
        a = self.cfg.plate_match_smoothing if self._have_match else 1.0  # first estimate: take it fully
        self.gain = (1 - a) * self.gain + a * gain
        self.offset = (1 - a) * self.offset + a * offset
        self._have_match = True
        matched = np.clip(plate * self.gain + self.offset, 0, 255)
        self.stats.add(float(live.mean()), float(plate.mean()), float(matched.mean()), fraction,
                       self.gain, self.offset)

    # -- masks ---------------------------------------------------------------------------------
    def _hull_mask(self, hands: list[Hand]) -> np.ndarray:
        w, h = self.small_size
        canvas = np.zeros((h, w), np.uint8)
        for hand in hands:
            pts = (hand.landmarks[:, :2] * (w, h)).astype(np.int32)
            cv2.fillConvexPoly(canvas, cv2.convexHull(pts), 255)
        if hands and self._k_hull is not None:
            canvas = cv2.dilate(canvas, self._k_hull)
        if hands and self._feather:
            canvas = cv2.GaussianBlur(canvas, (self._feather, self._feather), 0)
        return canvas.astype(np.float32) * (1.0 / 255.0)

    def _diff_mask(self, small: np.ndarray, plate_matched_small: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(cv2.absdiff(small, plate_matched_small), cv2.COLOR_BGR2GRAY)
        gray = cv2.blur(gray, (3, 3))
        binary = (gray > self.cfg.plate_diff_threshold).astype(np.uint8) * 255
        if self._k_open is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._k_open)
        if self._k_close is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._k_close)
        if float((binary > 0).mean()) > self.cfg.plate_diff_max_fraction:
            self.stats.diff_skipped += 1  # most of the frame differs: a lighting change, not a person
            return None
        self.stats.diff_frames += 1
        if self._feather:
            binary = cv2.GaussianBlur(binary, (self._feather, self._feather), 0)
        return binary.astype(np.float32) * (1.0 / 255.0)

    def compute(self, small: np.ndarray, seg_small: np.ndarray, hands: list[Hand],
                timer=None) -> VanishResult:
        """Build the vanish mask and matched plate for the current frame.

        `small` is the perception-resolution BGR frame, `seg_small` the segmenter's mask at the same
        size, `hands` the newest hand landmarks (any age).
        """
        if self._plate_u8 is None:
            raise RuntimeError("set_plate() must be called first")
        cfg = self.cfg

        def stage(name):
            return timer.stage(name) if timer else nullcontext()

        with stage("vm_seg"):
            grown = cv2.dilate(seg_small, self._k_grow) if self._k_grow is not None else seg_small
        zeros = np.zeros_like(seg_small)
        with stage("vm_hands"):
            hull = self._hull_mask(hands) if cfg.mask_mode in ("seg_hands", "full") else zeros
        covered = np.maximum(grown, hull)

        with stage("vm_match"):
            self._estimate_match(small, covered)  # always measured; applied unless plate_match is off
            lut = self._lut() if cfg.plate_match != "off" else None
            plate_small = cv2.LUT(self._plate_small, lut) if lut is not None else self._plate_small
        with stage("vm_diff"):
            diff = self._diff_mask(small, plate_small) if cfg.mask_mode == "full" else None
            if diff is None:
                diff = zeros
        mask_small = np.maximum(covered, diff) if cfg.mask_mode == "full" else covered
        with stage("vm_up"):
            mask = cv2.resize(mask_small, self.out_size, interpolation=cv2.INTER_LINEAR)
            plate_full = cv2.LUT(self._plate_u8, lut) if lut is not None else self._plate_u8
        return VanishResult(mask, plate_full, seg_small, grown, hull, diff)
