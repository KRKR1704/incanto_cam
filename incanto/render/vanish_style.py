"""Vanish styles: what is shown while you are vanished.

The per-frame mask (render/vanish_mask.py) is kept as it is: any mask can leave an outline of you
when you move fast, so there are three ways to use it:

* ``freeze``  the whole frame is the plate. No mask at all, so nothing of you can show however you
              move. The plate is gain-matched to the live frame once, when vanish turns on, and then
              held fixed; synthetic grain (render/grain.py) keeps the picture from looking frozen.
              Cheapest: the mask is not even computed.
* ``region``  a large soft region shows the plate and the rest stays live. The region is the union
              of every mask source seen during this vanish session (max-hold, so it only grows),
              grown by `region_margin_px` and feathered by `region_feather_px`. Outside it the gain
              map and the plate refresh keep running; the refresh never touches the region.
* ``live``    the per-frame mask composite, unchanged.

Turning vanish on or off crossfades (`transition_s`, time-based with the frame's dt) between the
ordinary frame and the vanished output; ``live`` keeps its hard cut. Phase 6 replaces this with the
dissolve. Before reappearing in ``freeze`` the plate is gain-matched again to the live frame, and the
frozen picture is moved onto that match over the fade, so the way back is smooth too.
"""
from __future__ import annotations

import logging
from contextlib import nullcontext

import cv2
import numpy as np

from incanto.config import VANISH_STYLES, VanishConfig
from incanto.perception.hands import Hand
from incanto.plate.clean_plate import Plate
from incanto.render.compositor import composite_roi
from incanto.render.grain import GrainProfile, GrainSynth
from incanto.render.vanish_mask import MATCH_STRIDE, REFERENCE_WIDTH, VanishMask, VanishResult

log = logging.getLogger(__name__)

ALPHA_EPS = 0.003  # region alpha below this counts as zero


class CrossFade:
    """0 = ordinary frame, 1 = vanished output. Moves linearly toward its target at 1/duration per second,
    using the frame's own dt, so the fade takes `duration_s` at any frame rate."""

    def __init__(self, duration_s: float):
        self.duration_s = duration_s
        self.alpha = 0.0

    def step(self, target_on: bool, dt: float) -> float:
        if self.duration_s <= 0:
            self.alpha = 1.0 if target_on else 0.0
        else:
            delta = max(dt, 0.0) / self.duration_s
            self.alpha = min(1.0, self.alpha + delta) if target_on else max(0.0, self.alpha - delta)
        return self.alpha


class RegionHold:
    """The region style's mask: max-hold of the mask sources, dilated and feathered.

    Works at 1/MATCH_STRIDE of the perception size (small and cheap: a 120 px margin is ~15 px there)
    and upsamples only the finished alpha, once per change."""

    def __init__(self, small_size: tuple[int, int], out_size: tuple[int, int], margin_px: int, feather_px: int,
                 threshold: float):
        w, h = small_size
        self.q = (max(1, w // MATCH_STRIDE), max(1, h // MATCH_STRIDE))
        self.out_size = out_size
        self.threshold = threshold
        k = w / REFERENCE_WIDTH / MATCH_STRIDE  # config px (at 1280 wide) -> quarter-resolution px
        radius = round(margin_px * k)
        self._kernel = (cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
                        if radius > 0 else None)
        self._feather = feather_px * k
        self.reset()

    def reset(self) -> None:
        self.hold = np.zeros((self.q[1], self.q[0]), np.uint8)
        self._alpha_q: np.ndarray | None = None
        self._full: tuple | None = None
        self._dirty = False

    @property
    def area(self) -> int:
        """Quarter-resolution cells held so far (never shrinks until reset())."""
        return int(self.hold.sum())

    def update(self, sources: np.ndarray | None) -> bool:
        """Union `sources` (perception-res, float 0..1) into the hold. True if the hold grew."""
        if sources is None:
            return False
        cells = cv2.resize((sources > self.threshold).astype(np.float32), self.q, interpolation=cv2.INTER_AREA) > 0
        grew = bool((cells & (self.hold == 0)).any())
        if grew:
            self.hold |= cells.astype(np.uint8)
            self._dirty = True
        return grew

    def alpha_q(self) -> np.ndarray:
        """float32 0..1 at quarter resolution: 1 inside the grown hold, a smooth fall-off outside it."""
        if self._alpha_q is None or self._dirty:
            core = cv2.dilate(self.hold, self._kernel) if self._kernel is not None else self.hold.copy()
            if not core.any():
                alpha = np.zeros(core.shape, np.float32)
            elif self._feather <= 0:
                alpha = core.astype(np.float32)
            else:
                dist = cv2.distanceTransform(1 - core, cv2.DIST_L2, 3)
                a = np.clip(1.0 - dist / self._feather, 0.0, 1.0)
                alpha = (a * a * (3.0 - 2.0 * a)).astype(np.float32)
            self._alpha_q, self._full, self._dirty = alpha, None, False
        return self._alpha_q

    def exclude_q(self) -> np.ndarray:
        """uint8 1 wherever the region has any alpha: the plate refresh must stay out of it."""
        return (self.alpha_q() > ALPHA_EPS).astype(np.uint8)

    def full(self) -> tuple[tuple[int, int, int, int], np.ndarray] | None:
        """(roi, alpha) at output resolution, alpha float32 covering just roi; None while empty. Cached."""
        a = self.alpha_q()
        if self._full is None:
            ys, xs = np.nonzero(a > ALPHA_EPS)
            if len(xs) == 0:
                self._full = ()
            else:
                qw, qh = self.q
                ow, oh = self.out_size
                x0, x1 = max(0, int(xs.min()) - 1), min(qw, int(xs.max()) + 2)
                y0, y1 = max(0, int(ys.min()) - 1), min(qh, int(ys.max()) + 2)
                fx, fy = ow / qw, oh / qh
                roi = (round(x0 * fx), round(y0 * fy), min(ow, round(x1 * fx)), min(oh, round(y1 * fy)))
                mask = cv2.resize(np.ascontiguousarray(a[y0:y1, x0:x1]), (roi[2] - roi[0], roi[3] - roi[1]),
                                  interpolation=cv2.INTER_LINEAR)
                self._full = (roi, mask)
        return self._full or None


class VanishStyler:
    """Owns the vanish style, the V crossfade, and everything frozen. One call per frame: `render()`.

    `want_on` is whether vanish is switched on (V, granted by the plate health check). The styler
    notices the edges itself, so the caller needs no begin/end calls."""

    def __init__(self, cfg: VanishConfig, vmask: VanishMask, out_size: tuple[int, int], seed: int | None = None):
        self.cfg = cfg
        self.vmask = vmask
        self.out_size = out_size
        self.style = cfg.style
        self.fade = CrossFade(cfg.transition_s)
        self.region = RegionHold(vmask.small_size, out_size, cfg.region_margin_px, cfg.region_feather_px,
                                 cfg.core_threshold)
        self._synth = GrainSynth(out_size, seed) if cfg.grain_scale > 0 else None
        self._profile: GrainProfile | None = None
        self._plate_u8: np.ndarray | None = None
        self._amp: np.ndarray | None = None       # int16 grain amplitudes for the current frozen/region plate
        self._base: np.ndarray | None = None      # freeze: the gain-matched plate, held fixed
        self._fresh: np.ndarray | None = None     # freeze: the plate re-matched to the live frame for the way back
        self._engaged = False                     # vanish is on (as far as the styler has seen)
        self._need_begin = False                  # the style changed mid-session: set it up on the next frame
        self._warned_no_noise = False

    # -- setup ---------------------------------------------------------------------------------
    def set_plate(self, plate: Plate) -> None:
        """A new plate (capture or load): also its noise profile. Call while vanish is off."""
        self._plate_u8 = plate.u8
        self._profile = GrainProfile(plate.noise, plate.u8) if plate.noise is not None else None
        self._warned_no_noise = False
        self._release()

    def set_style(self, style: str) -> None:
        if style not in VANISH_STYLES:
            raise ValueError(f"unknown vanish style {style!r}")
        if style == self.style:
            return
        self.style = style
        self._base = self._fresh = self._amp = None
        self._need_begin = True
        self.region.reset()  # a region belongs to one session of one style
        if style == "live":
            self.fade.alpha = 1.0 if self._engaged else 0.0  # the masked composite has no fade
        elif not self._engaged and self.fade.alpha > 0:
            self.fade.alpha = 0.0  # a fade-out in progress belongs to the old style

    def cycle_style(self) -> str:
        self.set_style(VANISH_STYLES[(VANISH_STYLES.index(self.style) + 1) % len(VANISH_STYLES)])
        return self.style

    def _release(self) -> None:
        self._base = self._fresh = self._amp = None
        self._engaged = False
        self._need_begin = False
        self.fade.alpha = 0.0
        self.region.reset()

    @property
    def active(self) -> bool:
        """True while anything vanished is on screen (on, or fading out)."""
        return self._engaged or self.fade.alpha > 0

    @property
    def grain_available(self) -> bool:
        return self._synth is not None and self._profile is not None

    # -- per-frame -----------------------------------------------------------------------------
    def render(self, frame: np.ndarray, small: np.ndarray, seg_small: np.ndarray, hands: list[Hand],
               want_on: bool, dt: float, timer=None, refresh: bool = False) -> tuple[np.ndarray, VanishResult | None]:
        """The frame to send. Returns (output, vanish result for the debug views or None).
        `refresh`: the plate is healthy, so the background refresh may run (live and region styles)."""
        def stage(name):
            return timer.stage(name) if timer else nullcontext()

        if want_on and not self._engaged:
            self._engaged, self._need_begin = True, True
        elif not want_on and self._engaged:
            self._engaged = False
            if self.style == "freeze" and self._base is not None:
                with stage("vs_rematch"):  # a fresh gain match for the way back
                    self._fresh = self.vmask.snapshot_matched(small, seg_small, hands)
        if self._need_begin and self._engaged:
            self._need_begin = False
            with stage("vs_begin"):
                self._begin(small, seg_small, hands)

        if self.style == "live":
            alpha = self.fade.alpha = 1.0 if self._engaged else 0.0
        else:
            alpha = self.fade.step(self._engaged, dt)
        if alpha <= 0.0 and not self._engaged:
            self._base = self._fresh = self._amp = None  # fade finished: let go of the frozen picture
            if self.style == "region":
                self.region.reset()
            return frame, None

        if self.style == "freeze":
            return self._render_freeze(frame, alpha, timer), None
        if self.style == "region":
            return self._render_region(frame, small, seg_small, hands, alpha, refresh, timer)
        return self._render_live(frame, small, seg_small, hands, refresh, timer)

    def _begin(self, small, seg_small, hands) -> None:
        """Set the current style up (vanish just turned on, or the style changed while it is on)."""
        vm = self.vmask
        if self.style == "freeze":
            if self._base is not None and self._fresh is not None and self.fade.alpha > 0:
                self._base = self._frozen_layer(self.fade.alpha)  # turned on again mid fade-out: keep what is shown
            else:
                self._base = vm.snapshot_matched(small, seg_small, hands)  # gain map against the live frame, once
            self._fresh = None
            self._amp = self._amplitudes(self._base)
        elif self.style == "region":
            if self.fade.alpha <= 0:
                self.region.reset()
            vm.reset_temporal()
            self._amp = self._amplitudes(vm.matched_plate_full())

    def _amplitudes(self, matched: np.ndarray) -> np.ndarray | None:
        scale = self.cfg.grain_scale
        if self._synth is None or scale <= 0:
            return None
        if self._profile is None:
            if not self._warned_no_noise:
                self._warned_no_noise = True
                log.warning("This plate has no noise profile (it was saved before grain existed): no synthetic "
                            "grain. Press B to capture a new plate.")
            return None
        return self._profile.amplitude(matched, scale)

    def _frozen_layer(self, alpha: float) -> np.ndarray:
        """The frozen plate; while fading out it moves from the held match onto the fresh one."""
        if self._fresh is None or alpha >= 1.0:
            return self._base
        return cv2.addWeighted(self._base, alpha, self._fresh, 1.0 - alpha, 0.0)

    def _render_freeze(self, frame: np.ndarray, alpha: float, timer) -> np.ndarray:
        if self._base is None:  # style switched to freeze while vanish is off and fading: nothing to show
            return frame
        with timer.stage("vs_freeze") if timer else nullcontext():
            layer = self._frozen_layer(alpha)
            frozen = self._synth.apply(layer, self._amp) if self._synth and self._amp is not None else layer
            if alpha >= 1.0:
                return frozen
            return cv2.addWeighted(frame, 1.0 - alpha, frozen, alpha, 0.0)

    def _render_live(self, frame, small, seg_small, hands, refresh, timer):
        vm = self.vmask
        res = vm.compute(small, seg_small, hands, timer)
        if refresh:
            with timer.stage("vm_refresh") if timer else nullcontext():
                vm.refresh_plate(frame)
        with timer.stage("composite") if timer else nullcontext():
            out = composite_roi(frame, res.mask, res.plate, res.roi) if res.roi else frame
        return out, res

    def _render_region(self, frame, small, seg_small, hands, alpha, refresh, timer):
        vm = self.vmask
        res = vm.compute(small, seg_small, hands, timer)
        if self._engaged:  # the hold only grows while vanished (and not while fading out)
            self.region.update(res.sources)
        if refresh:
            with timer.stage("vm_refresh") if timer else nullcontext():
                vm.refresh_plate(frame, exclude=self.region.exclude_q())
        with timer.stage("vs_region") if timer else nullcontext():
            full = self.region.full()
            if full is None:
                return frame, res
            roi, mask = full
            plate = vm.matched_plate_roi(roi)
            if self._synth and self._amp is not None:
                plate = self._synth.apply(plate, self._amp, roi)
            if alpha < 1.0:
                mask = mask * np.float32(alpha)
            return composite_roi(frame, mask, plate, roi), res
