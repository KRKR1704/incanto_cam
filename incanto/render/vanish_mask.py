"""The mask used to replace the person with the clean plate.

Principle: over-covering is free (the plate IS the background), under-covering reveals the person.
Three sources are unioned (each can be switched off with `vanish.mask_mode`):

1. the segmenter mask, grown by `dilate_px`;
2. a filled convex hull of each hand's landmarks, grown by `hand_hull_dilate_px` (the segmentation
   model misses thin fingers, and landmarks are ~50 ms old);
3. pixels that differ from the brightness-matched plate (morphologically cleaned), allowed only inside
   a "near me" zone: (1) and (2) dilated by `near_me_px`. Elsewhere moved objects stay live.

The plate is first matched to the live brightness/colour using background pixels well outside the
mask, because auto exposure re-adjusts when you step back in and the room is lit unevenly:
`gain_map` (default) estimates a per-channel gain on a coarse grid and interpolates it, `gain` and
`gain_offset` are single global corrections, `off` leaves the plate alone.

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
MIN_CELL_BG = 0.15      # a grid cell needs this fraction of background pixels, else it borrows neighbours
MIN_CELL_PLATE = 10.0   # ...and a plate brighter than this (gain from a black patch is meaningless)
MATCH_EVERY = 2         # brightness is estimated every 2nd frame: it drifts slowly and this halves the cost
APPLY_EVERY = 4         # the full-resolution matched plate is rebuilt every 4th frame (the gain drifts slowly)
STATS_EVERY = 2         # background statistics for the log are taken on every 2nd estimate
ROI_PAD = 4             # perception pixels added around the mask's bounding box


@dataclass
class VanishResult:
    mask: np.ndarray        # float32 0..1, covering only `roi` at full output resolution
    plate: np.ndarray       # uint8 matched plate for `roi`
    roi: tuple[int, int, int, int] | None  # (x0, y0, x1, y1) in output pixels where mask > 0; None = nothing to replace
    seg: np.ndarray         # perception-res float32 pieces, for the debug view
    grown: np.ndarray       # segmenter mask after dilation
    hull: np.ndarray
    diff: np.ndarray        # plate-difference mask after the near-me limit
    zone: np.ndarray        # float32 0..1 "near me" zone in which the plate difference is allowed


class VanishStats:
    """Window of background measurements, logged every few seconds."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.live = self.raw_plate = self.matched_plate = self.fraction = 0.0
        self.gain = np.zeros(3)
        self.offset = np.zeros(3)
        self.gain_min, self.gain_max = 9.0, 0.0
        self.resid_mean_abs = self.resid_signed = 0.0
        self.resid_max_abs = 0.0
        self.cells = 0
        self.n = 0
        self.resid_n = 0
        self.diff_frames = self.diff_skipped = 0


def _ellipse(radius: int) -> np.ndarray | None:
    if radius <= 0:
        return None
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))


def fill_missing_cells(grid: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Give every invalid cell the average of its valid 8-neighbours, repeating until all are filled."""
    grid, valid = grid.copy(), valid.copy()
    kernel = np.ones((3, 3), np.float32)
    for _ in range(max(grid.shape[:2])):
        if valid.all() or not valid.any():
            break
        vf = valid.astype(np.float32)
        num = cv2.filter2D(grid * vf[..., None], -1, kernel, borderType=cv2.BORDER_CONSTANT)
        den = cv2.filter2D(vf, -1, kernel, borderType=cv2.BORDER_CONSTANT)
        new = (~valid) & (den > 0)
        grid[new] = num[new] / den[new][:, None]
        valid |= new
    return grid, valid


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
        self._k_zone = _ellipse(max(1, round(cfg.near_me_px * s / MATCH_STRIDE)))
        feather = round(cfg.feather_px * s)
        self._feather = (feather * 2 + 1) if feather > 0 else 0
        self._cells = (cfg.plate_match_cells_x, cfg.plate_match_cells_y)
        self._plate_u8: np.ndarray | None = None
        self._plate_small: np.ndarray | None = None
        self._frame_no = 0
        self._estimates = 0
        self._matched_full: np.ndarray | None = None
        self.stats = VanishStats()
        self._reset_match()

    # -- plate ---------------------------------------------------------------------------------
    def set_plate(self, plate_u8: np.ndarray) -> None:
        self._plate_u8 = plate_u8
        self._plate_small = cv2.resize(plate_u8, self.small_size, interpolation=cv2.INTER_AREA)
        self._reset_match()

    def _reset_match(self) -> None:
        self.gain = np.ones(3, np.float32)       # global modes
        self.offset = np.zeros(3, np.float32)
        gx, gy = self._cells
        self.grid = np.ones((gy, gx, 3), np.float32)  # gain_map mode
        self._have_match = False
        self._matched_full = None

    def _apply(self, plate: np.ndarray, roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """The plate with the current brightness/colour correction. `roi` = (x0, y0, x1, y1) in the
        plate's own pixels restricts the work (and the result) to that rectangle."""
        mode = self.cfg.plate_match
        h, w = plate.shape[:2]
        x0, y0, x1, y1 = roi if roi else (0, 0, w, h)
        part = np.ascontiguousarray(plate[y0:y1, x0:x1])
        if mode == "off":
            return part
        if mode == "gain_map":
            # bilinear gain for exactly these pixels: sample the full-frame gain field over the ROI
            gx, gy = self._cells
            sx, sy = gx / w, gy / h
            m = np.float32([[sx, 0, 0.5 * sx - 0.5 + x0 * sx], [0, sy, 0.5 * sy - 0.5 + y0 * sy]])
            gain = cv2.warpAffine(self.grid, m, (x1 - x0, y1 - y0), flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                                  borderMode=cv2.BORDER_REPLICATE)
            return cv2.multiply(part, gain, dtype=cv2.CV_8U)
        x = np.arange(256, dtype=np.float32)[:, None]
        lut = np.clip(x * self.gain + self.offset + 0.5, 0, 255).astype(np.uint8).reshape(1, 256, 3)
        return cv2.LUT(part, lut)

    # -- brightness matching -------------------------------------------------------------------
    def _update_global(self, live: np.ndarray, plate: np.ndarray) -> None:
        """Single per-channel gain (and optionally offset) from flattened background pixels."""
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
        a = self.cfg.plate_match_smoothing if self._have_match else 1.0  # first estimate: take it fully
        self.gain = (1 - a) * self.gain + a * gain
        self.offset = (1 - a) * self.offset + a * offset
        self._have_match = True

    def _cell_sums(self, live: np.ndarray, plate: np.ndarray, keep: np.ndarray):
        """Per grid cell: area-mean of live and plate over `keep` pixels, and the kept fraction."""
        gx, gy = self._cells
        k = keep[..., None]
        return (cv2.resize(live * k, (gx, gy), interpolation=cv2.INTER_AREA),
                cv2.resize(plate * k, (gx, gy), interpolation=cv2.INTER_AREA),
                cv2.resize(keep, (gx, gy), interpolation=cv2.INTER_AREA))

    def _cell_gain(self, lc, pc, kc):
        valid = (kc >= MIN_CELL_BG) & ((pc / np.maximum(kc, 1e-6)[..., None]).mean(2) > MIN_CELL_PLATE)
        gain = np.clip(lc / np.maximum(pc, 1e-3), *GAIN_LIMITS).astype(np.float32)
        return gain, valid

    def _robust_background(self, live: np.ndarray, plate: np.ndarray, bgf: np.ndarray) -> np.ndarray:
        """Background pixels that look like the plate under plausible lighting, so that a leftover
        piece of the person (or a moved object) cannot pull the gain estimate. The reference is the
        per-channel MEDIAN live/plate ratio, allowed to vary by -35%/+50% from place to place."""
        bg = bgf > 0
        lv, pv = live[bg], plate[bg]
        sel = pv.min(axis=1) > 20.0
        if sel.sum() < 100:
            return bgf
        med = np.clip(np.median(lv[sel][::2] / pv[sel][::2], axis=0), *GAIN_LIMITS).astype(np.float32)
        lo, hi = 0.65 * med * plate - 10.0, 1.5 * med * plate + 10.0
        return bgf * ((live >= lo) & (live <= hi)).all(axis=2)

    def _update_map(self, live: np.ndarray, plate: np.ndarray, bgf: np.ndarray) -> None:
        """Per-cell, per-channel gain from background pixels; sparse cells borrow from neighbours."""
        gain, valid = self._cell_gain(*self._cell_sums(live, plate, bgf))
        if not valid.any():
            return  # almost no background visible: keep the last estimate
        filled, _ = fill_missing_cells(gain, valid)
        h, w = bgf.shape
        pred = plate * cv2.resize(filled, (w, h), interpolation=cv2.INTER_LINEAR)
        keep = bgf * (np.abs(live - pred).max(axis=2) < OUTLIER_LEVELS)  # drop leftover person / moved objects
        gain2, valid2 = self._cell_gain(*self._cell_sums(live, plate, keep.astype(np.float32)))
        if valid2.any():
            gain, valid = gain2, valid2
        gain, _ = fill_missing_cells(gain, valid)
        a = self.cfg.plate_match_smoothing if self._have_match else 1.0
        self.grid = (1 - a) * self.grid + a * gain
        self._have_match = True

    def _estimate_match(self, small: np.ndarray, covered: np.ndarray) -> None:
        sub = (slice(None, None, MATCH_STRIDE), slice(None, None, MATCH_STRIDE))
        cov = (covered[sub] > COVERED).astype(np.uint8)
        if self._k_margin is not None:
            cov = cv2.dilate(cov, self._k_margin)
        bg = cov == 0
        live = small[sub].astype(np.float32)
        plate = self._plate_small[sub].astype(np.float32)
        bgf = bg.astype(np.float32)
        fraction = float(bgf.mean())
        if bg.sum() < 200:  # hardly any background visible: keep the last estimate
            return
        # Fit and report on plausible background only, so leftover person or moved objects neither
        # pull the correction nor inflate the residual figures (in every mode, for a fair A/B).
        bgf = self._robust_background(live, plate, bgf).astype(np.float32)
        bg = bgf > 0
        if bg.sum() < 200:
            return
        mode = self.cfg.plate_match
        if mode == "gain_map":
            self._update_map(live, plate, bgf)
        elif mode in ("gain", "gain_offset"):
            self._update_global(live[bg], plate[bg])
        self._estimates += 1
        if self._estimates % STATS_EVERY == 1:
            self._record_stats(live, plate, bg, bgf, fraction)

    def _record_stats(self, live, plate, bg, bgf, fraction) -> None:
        """Residual live - matched plate per grid cell (over background pixels), for the log."""
        st = self.stats
        h, w = bg.shape
        if self.cfg.plate_match == "gain_map":
            matched = plate * cv2.resize(self.grid, (w, h), interpolation=cv2.INTER_LINEAR)
            gains = self.grid
        elif self.cfg.plate_match == "off":
            matched, gains = plate, np.ones(3, np.float32)
        else:
            matched, gains = plate * self.gain + self.offset, self.gain
        matched = np.clip(matched, 0, 255)
        gx, gy = self._cells
        resid = cv2.resize((live - matched).mean(axis=2) * bgf, (gx, gy), interpolation=cv2.INTER_AREA)
        cell_bg = cv2.resize(bgf, (gx, gy), interpolation=cv2.INTER_AREA)
        ok = cell_bg >= MIN_CELL_BG
        st.live += float(live[bg].mean())
        st.raw_plate += float(plate[bg].mean())
        st.matched_plate += float(matched[bg].mean())
        st.fraction += fraction
        st.gain += np.asarray(gains, np.float64).reshape(-1, 3).mean(0)
        st.offset += self.offset if self.cfg.plate_match == "gain_offset" else 0.0
        st.gain_min = min(st.gain_min, float(np.min(gains)))
        st.gain_max = max(st.gain_max, float(np.max(gains)))
        st.n += 1
        if ok.any():
            r = resid[ok] / cell_bg[ok]
            st.resid_mean_abs += float(np.abs(r).mean())
            st.resid_signed += float(r.mean())
            st.resid_max_abs = max(st.resid_max_abs, float(np.abs(r).max()))
            st.cells += int(ok.sum())
            st.resid_n += 1

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

    def _near_me_zone(self, covered: np.ndarray) -> np.ndarray:
        """Soft 0..1 zone (perception res): the covered area dilated by `near_me_px`."""
        sub = (covered[::MATCH_STRIDE, ::MATCH_STRIDE] > COVERED).astype(np.uint8)
        zone = cv2.dilate(sub, self._k_zone) if self._k_zone is not None else sub
        return cv2.resize(zone.astype(np.float32), self.small_size, interpolation=cv2.INTER_LINEAR)

    def _diff_mask(self, small: np.ndarray, plate_matched_small: np.ndarray,
                   zone: np.ndarray) -> np.ndarray | None:
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
        return binary.astype(np.float32) * (1.0 / 255.0) * zone  # only near me: moved objects elsewhere stay live

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
            if self._frame_no % MATCH_EVERY == 0:
                self._estimate_match(small, covered)  # always measured; applied unless plate_match is off
            self._frame_no += 1
            plate_small = self._apply(self._plate_small)
        with stage("vm_diff"):
            zone = self._near_me_zone(covered) if cfg.mask_mode == "full" else zeros
            diff = self._diff_mask(small, plate_small, zone) if cfg.mask_mode == "full" else None
            if diff is None:
                diff = zeros
        mask_small = np.maximum(covered, diff) if cfg.mask_mode == "full" else covered
        # Everything at full resolution happens only inside the bounding box of the mask.
        with stage("vm_up"):
            ys, xs = np.nonzero(mask_small[::2, ::2] > 0.003)
            if len(xs) == 0:
                return VanishResult(zeros[:0, :0], self._plate_u8[:0, :0], None, seg_small, grown, hull, diff, zone)
            sh, sw = mask_small.shape
            x0, x1 = max(0, int(xs.min()) * 2 - ROI_PAD), min(sw, int(xs.max()) * 2 + 2 + ROI_PAD)
            y0, y1 = max(0, int(ys.min()) * 2 - ROI_PAD), min(sh, int(ys.max()) * 2 + 2 + ROI_PAD)
            ow, oh = self.out_size
            fx, fy = ow / sw, oh / sh
            roi = (int(x0 * fx), int(y0 * fy), min(ow, int(np.ceil(x1 * fx))), min(oh, int(np.ceil(y1 * fy))))
            # upscale just that window of the small mask, aligned to the full-resolution pixel grid
            m = np.float32([[1 / fx, 0, 0.5 / fx - 0.5 + roi[0] / fx], [0, 1 / fy, 0.5 / fy - 0.5 + roi[1] / fy]])
            mask = cv2.warpAffine(mask_small, m, (roi[2] - roi[0], roi[3] - roi[1]),
                                  flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP, borderMode=cv2.BORDER_REPLICATE)
        with stage("vm_plate"):
            if self._matched_full is None or self._frame_no % APPLY_EVERY == 0:
                self._matched_full = self._apply(self._plate_u8)
            plate_roi = np.ascontiguousarray(self._matched_full[roi[1]:roi[3], roi[0]:roi[2]])
        return VanishResult(mask, plate_roi, roi, seg_small, grown, hull, diff, zone)
