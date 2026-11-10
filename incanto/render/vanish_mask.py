"""The mask used to replace the person with the clean plate.

Principle: over-covering is free (the plate IS the background), under-covering reveals the person.
Three sources are unioned (each can be switched off with `vanish.mask_mode`) into a SOLID core
(alpha exactly 1.0 wherever any source covers) with the feather only in a ring outside it:

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
ROI_PAD = 2             # half-resolution pixels added around the mask's bounding box


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
        self.diff_frames = 0
        self.guard_frames = self.guard_frames_any = self.guard_cells = 0  # per-cell lighting guard
        self.seg_empty = 0  # frames where the segmenter found (almost) nobody
        self.large_blob_frames = 0  # frames where a body-sized difference outside the zone was accepted


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
        # The core and its feather ring are built at HALF the perception resolution (cheaper; the
        # result is upsampled anyway), so their sizes are halved here.
        self._k_core_close = _ellipse(round(cfg.core_close_px * s / 2))
        self._feather_r = round(cfg.outer_feather_px * s / 2)  # ring width beyond the solid core, in half-res px
        self._prev_diff: np.ndarray | None = None
        self._guard_cut: np.ndarray | None = None
        self._prev_zone: np.ndarray | None = None
        self._cells = (cfg.plate_match_cells_x, cfg.plate_match_cells_y)
        self._plate_u8: np.ndarray | None = None
        self._plate_small: np.ndarray | None = None
        self._frame_no = 0
        self._estimates = 0
        self._matched_full: np.ndarray | None = None
        self._matched_small: np.ndarray | None = None
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
        self._matched_small = None

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
        return canvas.astype(np.float32) * (1.0 / 255.0)

    def _near_me_zone(self, covered: np.ndarray) -> np.ndarray:
        """Soft 0..1 zone (perception res): the covered area dilated by `near_me_px`, released slowly
        so a one-frame segmenter dropout does not shrink it."""
        sub = (covered[::MATCH_STRIDE, ::MATCH_STRIDE] > COVERED).astype(np.uint8)
        zone = cv2.dilate(sub, self._k_zone) if self._k_zone is not None else sub
        zone = cv2.resize(zone.astype(np.float32), self.small_size, interpolation=cv2.INTER_LINEAR)
        if self._prev_zone is not None:
            zone = np.maximum(zone, self._prev_zone * self.cfg.plate_diff_release_decay)
            zone[zone < 0.02] = 0.0
        self._prev_zone = zone
        return zone

    def _difference(self, small: np.ndarray, plate_small: np.ndarray) -> np.ndarray:
        """uint8 |live - matched plate| by the configured metric (levels, 0-255)."""
        metric = self.cfg.plate_diff_metric
        if metric == "gray":
            d = cv2.cvtColor(cv2.absdiff(small, plate_small), cv2.COLOR_BGR2GRAY)
        elif metric == "max_channel":
            b, g, r = cv2.split(cv2.absdiff(small, plate_small))  # colour-aware: similar brightness, other colour
            d = cv2.max(cv2.max(b, g), r)
        else:  # lab: lightness half-weighted, chroma full weight
            a = cv2.cvtColor(small, cv2.COLOR_BGR2LAB).astype(np.float32)
            b_ = cv2.cvtColor(plate_small, cv2.COLOR_BGR2LAB).astype(np.float32)
            dd = a - b_
            d = np.clip(np.sqrt(0.5 * dd[..., 0] ** 2 + dd[..., 1] ** 2 + dd[..., 2] ** 2), 0, 255).astype(np.uint8)
        return cv2.blur(d, (3, 3))

    def _cell_guard(self, small: np.ndarray, plate_small: np.ndarray, binary: np.ndarray,
                    covered: np.ndarray) -> np.ndarray:
        """Per grid cell: drop the difference where it is just a lighting wash the gain map missed.

        A cell is a wash if most of its uncovered pixels differ AND one neutral, plausible gain
        (live ~= gain * plate, equal across channels, 0.6-1.6) explains it to within a few levels:
        a person never keeps the plate's texture under a single neutral gain. Counted in stats."""
        cfg, st = self.cfg, self.stats
        st.guard_frames += 1
        if st.guard_frames % 2 == 0 and self._guard_cut is not None:  # lighting changes slowly: every 2nd frame
            return binary * (1 - self._guard_cut) if self._guard_cut.any() else binary
        self._guard_cut = None
        sub = (slice(None, None, MATCH_STRIDE), slice(None, None, MATCH_STRIDE))
        flag = binary[sub] > 0
        unc = covered[sub] <= COVERED
        cand = flag & unc
        if not cand.any():
            self._guard_cut = np.zeros(self.small_size[::-1], np.uint8)
            return binary
        gx, gy = self._cells

        def area(a):
            return cv2.resize(a, (gx, gy), interpolation=cv2.INTER_AREA)

        live, plate = small[sub].astype(np.float32), plate_small[sub].astype(np.float32)
        uf, cf = unc.astype(np.float32), cand.astype(np.float32)
        unc_frac, cand_frac = area(uf), area(cf)
        gain = np.clip(area(live * uf[..., None]) / np.maximum(area(plate * uf[..., None]), 1e-3), *GAIN_LIMITS)
        gain_px = cv2.resize(gain.astype(np.float32), (live.shape[1], live.shape[0]), interpolation=cv2.INTER_NEAREST)
        resid_px = np.abs(live - plate * gain_px).max(axis=2)
        resid = area(resid_px * cf) / np.maximum(cand_frac, 1e-6)
        wash = cand_frac / np.maximum(unc_frac, 1e-6)
        neutral = gain.max(axis=2) / np.maximum(gain.min(axis=2), 1e-3) < 1.2
        plausible = (gain.mean(axis=2) > 0.6) & (gain.mean(axis=2) < 1.6)
        suppress = ((unc_frac >= 0.2) & (wash > cfg.plate_diff_wash_fraction)
                    & (resid < cfg.plate_diff_wash_residual) & neutral & plausible)
        n = int(suppress.sum())
        cut = cv2.resize(suppress.astype(np.uint8), self.small_size, interpolation=cv2.INTER_NEAREST)
        self._guard_cut = cut
        if n == 0:
            return binary
        st.guard_cells += n
        st.guard_frames_any += 1
        return binary * (1 - cut)

    def _large_blobs(self, binary: np.ndarray, zone: np.ndarray) -> np.ndarray | None:
        """Connected differences at least `plate_diff_large_blob_fraction` of the frame (body-sized)."""
        frac = self.cfg.plate_diff_large_blob_fraction
        if frac <= 0:
            return None
        sub = binary[::4, ::4]  # body-sized blobs survive a 4x decimation
        n, labels, stats, _ = cv2.connectedComponentsWithStats(sub, connectivity=8)
        if n <= 1:
            return None
        big = [i for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= frac * sub.size]
        if not big:
            return None
        keep = np.isin(labels, big).astype(np.uint8)
        large = cv2.resize(keep, self.small_size, interpolation=cv2.INTER_NEAREST).astype(np.float32)
        if float(((large > 0) & (zone < 0.5))[::4, ::4].mean()) > 0.005:  # beyond the zone by > 0.5% of the frame
            self.stats.large_blob_frames += 1
        return large

    def _diff_mask(self, small: np.ndarray, plate_small: np.ndarray, zone: np.ndarray,
                   covered: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        binary = (self._difference(small, plate_small) > cfg.plate_diff_threshold).astype(np.uint8) * 255
        if self._k_open is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, self._k_open)
        if self._k_close is not None:
            binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, self._k_close)
        if cfg.plate_diff_cell_guard:
            binary = self._cell_guard(small, plate_small, binary, covered)
        allowed = zone  # near me; moved objects elsewhere stay live...
        large = self._large_blobs(binary, zone)
        if large is not None:  # ...unless the difference is body-sized (a segmenter dropout must not expose me)
            allowed = np.maximum(zone, large)
        cur = binary.astype(np.float32) * (1.0 / 255.0) * allowed
        if self._prev_diff is not None:  # fast attack, slow release
            cur = np.maximum(cur, self._prev_diff * cfg.plate_diff_release_decay)
            cur[cur < 0.02] = 0.0
        self._prev_diff = cur
        self.stats.diff_frames += 1
        return cur

    def _solid_core_alpha(self, union: np.ndarray) -> np.ndarray:
        """alpha == 1.0 wherever any source covers; the feather lives only in a ring OUTSIDE that core.

        Built at half resolution. The ring starts one half-pixel beyond the core so that the bilinear
        upsampling cannot pull alpha below 1.0 at the core's edge."""
        h2, w2 = union.shape[0] // 2, union.shape[1] // 2
        u = np.maximum(np.maximum(union[0:2 * h2:2, 0:2 * w2:2], union[1:2 * h2:2, 0:2 * w2:2]),
                       np.maximum(union[0:2 * h2:2, 1:2 * w2:2], union[1:2 * h2:2, 1:2 * w2:2]))  # 2x2 max-pool
        core = (u > self.cfg.core_threshold).astype(np.uint8)
        if self._k_core_close is not None:
            core = cv2.morphologyEx(core, cv2.MORPH_CLOSE, self._k_core_close)
        if not core.any() or self._feather_r <= 0:
            return core.astype(np.float32)
        dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 3)  # 0 inside the core, grows outward
        a = np.clip(1.0 - np.maximum(dist - 1.0, 0.0) / float(self._feather_r), 0.0, 1.0)
        return a * a * (3.0 - 2.0 * a)

    def reset_temporal(self) -> None:
        """Forget the previous frames' difference mask and zone (call when vanish is switched on)."""
        self._prev_diff = self._prev_zone = self._guard_cut = None

    def matched_plate_full(self) -> np.ndarray:
        """The full-resolution brightness-matched plate (what the vanished pixels are drawn from)."""
        if self._matched_full is None:
            self._matched_full = self._apply(self._plate_u8)
        return self._matched_full

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

        if float((seg_small[::4, ::4] > 0.5).mean()) < 0.02:
            self.stats.seg_empty += 1  # dropout: the other sources must carry the mask
        with stage("vm_seg"):
            if self._k_grow is not None:  # dilate in uint8: 4x faster than float32 with an elliptical kernel
                grown = cv2.dilate((seg_small * 255.0).astype(np.uint8), self._k_grow).astype(np.float32) * (1.0 / 255.0)
            else:
                grown = seg_small
        zeros = np.zeros_like(seg_small)
        with stage("vm_hands"):
            hull = self._hull_mask(hands) if cfg.mask_mode in ("seg_hands", "full") else zeros
        covered = np.maximum(grown, hull)

        with stage("vm_match"):
            if self._frame_no % MATCH_EVERY == 0:
                self._estimate_match(small, covered)  # always measured; applied unless plate_match is off
            if self._matched_small is None or self._frame_no % MATCH_EVERY == 0:
                self._matched_small = self._apply(self._plate_small)  # changes only when the estimate does
            self._frame_no += 1
            plate_small = self._matched_small
        with stage("vm_diff"):
            if cfg.mask_mode == "full":
                zone = self._near_me_zone(covered)
                diff = self._diff_mask(small, plate_small, zone, covered)
            else:
                zone = diff = zeros
        with stage("vm_core"):
            alpha = self._solid_core_alpha(np.maximum(covered, diff))
        # Everything at full resolution happens only inside the bounding box of the mask.
        with stage("vm_up"):
            ys, xs = np.nonzero(alpha > 0.003)
            if len(xs) == 0:
                return VanishResult(zeros[:0, :0], self._plate_u8[:0, :0], None, seg_small, grown, hull, diff, zone)
            sh, sw = alpha.shape
            x0, x1 = max(0, int(xs.min()) - ROI_PAD), min(sw, int(xs.max()) + 1 + ROI_PAD)
            y0, y1 = max(0, int(ys.min()) - ROI_PAD), min(sh, int(ys.max()) + 1 + ROI_PAD)
            ow, oh = self.out_size
            fx, fy = ow / sw, oh / sh
            roi = (round(x0 * fx), round(y0 * fy), min(ow, round(x1 * fx)), min(oh, round(y1 * fy)))
            mask = cv2.resize(np.ascontiguousarray(alpha[y0:y1, x0:x1]), (roi[2] - roi[0], roi[3] - roi[1]),
                              interpolation=cv2.INTER_LINEAR)
        with stage("vm_plate"):
            if self._matched_full is None or self._frame_no % APPLY_EVERY == 0:
                self._matched_full = self._apply(self._plate_u8)
            plate_roi = np.ascontiguousarray(self._matched_full[roi[1]:roi[3], roi[0]:roi[2]])
        return VanishResult(mask, plate_roi, roi, seg_small, grown, hull, diff, zone)
