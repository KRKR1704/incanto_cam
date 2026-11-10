"""Plate health: does the stored plate still match the room the camera sees right now?

The plate is judged AFTER brightness matching, the way the vanish effect uses it. Auto exposure
swings the raw picture by tens of levels when you stand up or lean in; that is not a plate problem
and the gain map removes it, so raw brightness and global gain are not failure criteria.

* residual   what the gain map leaves behind: per grid cell, the mean of (live - matched plate) over
             the background, then the mean of |that| over the cells. It is the same figure the vanish
             log prints as "per-cell residual mean|r|", fitted here on a single frame. A cell that is
             off by several levels is a patch that would show as a lighter or darker shape.
             Measured on this camera: about 1 for a fresh plate, up to 2.7 with a 1.5x exposure change
             or a moved chair and opened door, 4.2-5.1 for a plate taken the evening before.
* structure  normalised correlation of the image gradients (edges) of plate and live. Brightness
             cancels out, so exposure changes do not move it; a moved camera, rearranged furniture or
             a different room do. Measured: 0.95 unchanged, 0.75 at a quarter of the exposure, 0.54
             with a moved chair and an opened door, 0.20 with the camera shifted by 2% of the width.
* gain       only a loose sanity cap on the global live / plate gain.

Per-cell figures only use cells that are mostly background (`min_cell_background`); cells mostly
covered by the person are ignored. Everything is a pure function of two images and a "covered" mask.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SCALE = 4                # measured on the images area-averaged down by this factor
COVERED = 0.02           # mask value above which a pixel belongs to the person
MIN_CELLS = 3            # fewer judged cells than this = no verdict
EDGE_FLOOR = 6.0         # mean squared gradient a cell needs (plate or live) to count as textured
# The fit below mirrors VanishMask._update_map (same constants), so the residual means the same thing.
FIT_GAIN_LIMITS = (0.5, 2.0)
FIT_OUTLIER_LEVELS = 40.0
FIT_MIN_CELL_BG = 0.15
FIT_MIN_CELL_PLATE = 10.0


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


@dataclass(frozen=True)
class HealthLimits:
    max_residual: float = 3.5          # mean over cells of |cell-mean (live - matched plate)|, levels
    min_structure: float = 0.35        # gradient correlation of plate vs live, 0..1
    gain_min: float = 0.5              # sanity cap on the global gain only
    gain_max: float = 2.0
    min_cell_background: float = 0.5   # a cell needs this share of background pixels to be judged
    hysteresis: float = 0.2            # after a failure, pass only with this much margin inside the limits


@dataclass(frozen=True)
class PlateHealth:
    residual: float          # mean over judged cells of |cell-mean (live - matched plate)|
    max_cell_residual: float
    structure: float | None  # gradient correlation over textured cells; None = no texture to compare
    gain: float              # global live / plate (median over the background)
    cells: int               # cells judged (mostly background)
    background: float        # fraction of the frame that counted as background
    reasons: tuple[str, ...]  # empty when healthy

    @property
    def ok(self) -> bool:
        return not self.reasons

    def describe(self) -> str:
        structure = "n/a" if self.structure is None else f"{self.structure:.2f}"
        text = (f"matched residual {self.residual:.2f} (worst cell {self.max_cell_residual:.1f}) | structure "
                f"{structure} | gain {self.gain:.2f} | {self.cells} cells | background {self.background:.0%}")
        return text + (" | FAILED: " + "; ".join(self.reasons) if self.reasons else " | healthy")


def _area(a: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    out = cv2.resize(a, size, interpolation=cv2.INTER_AREA)
    return out.reshape(size[1], size[0], -1) if a.ndim == 3 else out.reshape(size[1], size[0])


def _cell_gain(live: np.ndarray, plate: np.ndarray, keep: np.ndarray, cells: tuple[int, int]):
    k = keep[..., None]
    lc, pc, kc = _area(live * k, cells), _area(plate * k, cells), _area(keep, cells)
    valid = (kc >= FIT_MIN_CELL_BG) & ((pc / np.maximum(kc, 1e-6)[..., None]).mean(axis=2) > FIT_MIN_CELL_PLATE)
    return np.clip(lc / np.maximum(pc, 1e-3), *FIT_GAIN_LIMITS).astype(np.float32), valid


def fit_gain_map(live: np.ndarray, plate: np.ndarray, bgf: np.ndarray, cells: tuple[int, int]):
    """Single-frame version of the vanish mask's gain map: returns (grid, plausible background, global
    gain) or None when no cell has enough background. `live`/`plate` float32 BGR, `bgf` float32 0/1."""
    bg = bgf > 0
    lv, pv = live[bg], plate[bg]
    sel = pv.min(axis=1) > 20.0
    robust, median = bgf, 1.0
    if int(sel.sum()) >= 100:  # keep pixels that look like the plate under plausible lighting
        ratio = np.median(lv[sel] / pv[sel], axis=0)
        median = float(ratio.mean())
        med = np.clip(ratio, *FIT_GAIN_LIMITS).astype(np.float32)
        lo, hi = 0.65 * med * plate - 10.0, 1.5 * med * plate + 10.0
        robust = bgf * ((live >= lo) & (live <= hi)).all(axis=2)
    gain, valid = _cell_gain(live, plate, robust, cells)
    if not valid.any():
        return None
    size = (live.shape[1], live.shape[0])
    filled, _ = fill_missing_cells(gain, valid)
    pred = plate * cv2.resize(filled, size, interpolation=cv2.INTER_LINEAR)
    keep = robust * (np.abs(live - pred).max(axis=2) < FIT_OUTLIER_LEVELS)
    gain2, valid2 = _cell_gain(live, plate, keep.astype(np.float32), cells)
    if valid2.any():
        gain, valid = gain2, valid2
    grid, _ = fill_missing_cells(gain, valid)
    return grid, robust.astype(np.float32), median


def measure_plate_health(live: np.ndarray, plate: np.ndarray, covered: np.ndarray | None,
                         limits: HealthLimits, cells: tuple[int, int] = (8, 6), margin_px: int = 0,
                         recovering: bool = False) -> PlateHealth | None:
    """Compare `live` with `plate` (same-size BGR uint8) over the background, after gain matching.

    `covered` (float 0..1, same size, or None) marks the person; it is grown by `margin_px` and
    excluded. `recovering=True` applies the hysteresis margin (use it after a failed check).
    Returns None when too few cells are mostly background to judge.
    """
    if live.shape != plate.shape:
        raise ValueError(f"live {live.shape} does not match plate {plate.shape}")
    h, w = live.shape[:2]
    q = (max(1, w // SCALE), max(1, h // SCALE))
    lv = _area(live, q).astype(np.float32)
    pv = _area(plate, q).astype(np.float32)
    if covered is None:
        bg = np.ones(q[::-1], bool)
    else:
        cov = (_area(np.ascontiguousarray(covered, dtype=np.float32), q) > COVERED).astype(np.uint8)
        r = round(margin_px / SCALE)
        if r > 0:
            cov = cv2.dilate(cov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        bg = cov == 0
    bgf = bg.astype(np.float32)
    gx, gy = cells
    need = min(MIN_CELLS, gx * gy)
    frac = _area(bgf, cells)
    mostly_background = frac >= limits.min_cell_background
    if int(mostly_background.sum()) < need:
        return None
    fit = fit_gain_map(lv, pv, bgf, cells)
    if fit is None:
        return None
    grid, robust, global_gain = fit

    # Residual the gain map leaves, per cell (signed mean over the cell's background), as in the vanish log.
    matched = np.clip(pv * cv2.resize(grid, q, interpolation=cv2.INTER_LINEAR), 0, 255)
    rfrac = _area(robust, cells)
    judged = mostly_background & (rfrac >= FIT_MIN_CELL_BG)
    n = int(judged.sum())
    if n < need:
        return None
    cell_resid = np.abs(_area((lv - matched).mean(axis=2) * robust, cells)[judged] / rfrac[judged])
    residual, worst = float(cell_resid.mean()), float(cell_resid.max())

    # Structure: do the edges sit where they sat? Gradient correlation is blind to brightness.
    gl = cv2.cvtColor(lv, cv2.COLOR_BGR2GRAY) * (1.0 / max(global_gain, 1e-3))  # "textured" at matched brightness
    gp = cv2.cvtColor(pv, cv2.COLOR_BGR2GRAY)
    lx, ly = cv2.Sobel(gl, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gl, cv2.CV_32F, 0, 1, ksize=3)
    px, py = cv2.Sobel(gp, cv2.CV_32F, 1, 0, ksize=3), cv2.Sobel(gp, cv2.CV_32F, 0, 1, ksize=3)
    inner = cv2.erode(bg.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(np.float32)  # Sobel reads neighbours
    e_l = _area((lx * lx + ly * ly) * inner, cells)
    e_p = _area((px * px + py * py) * inner, cells)
    dot = _area((lx * px + ly * py) * inner, cells)
    textured = mostly_background & (np.maximum(e_l, e_p) > EDGE_FLOOR * frac)
    structure = None
    if textured.any():
        structure = float(dot[textured].sum() / max(float(np.sqrt(e_l[textured].sum() * e_p[textured].sum())), 1e-6))

    margin = limits.hysteresis if recovering else 0.0
    max_residual = limits.max_residual * (1.0 - margin)
    min_structure = limits.min_structure + margin * (1.0 - limits.min_structure)
    reasons = []
    if residual > max_residual:
        reasons.append(f"matched residual {residual:.2f} above {max_residual:.2f}")
    if structure is not None and structure < min_structure:
        reasons.append(f"structure {structure:.2f} below {min_structure:.2f}")
    if not limits.gain_min <= global_gain <= limits.gain_max:
        reasons.append(f"gain {global_gain:.2f} outside {limits.gain_min:g}-{limits.gain_max:g}")
    return PlateHealth(residual, worst, structure, global_gain, n, float(bgf.mean()), tuple(reasons))


def background_gain(live: np.ndarray, plate: np.ndarray, covered: np.ndarray | None,
                    margin_px: int = 0) -> float | None:
    """Mean brightness of the live background relative to the same pixels of the plate (1.0 = equal).
    Used to pick the exposure lock; None when hardly any background is visible."""
    step = (slice(None, None, SCALE), slice(None, None, SCALE))
    lv, pv = live[step], plate[step]
    if covered is None:
        bg = np.ones(lv.shape[:2], bool)
    else:
        cov = (covered[step] > COVERED).astype(np.uint8)
        r = round(margin_px / SCALE)
        if r > 0:
            cov = cv2.dilate(cov, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
        bg = cov == 0
    if int(bg.sum()) < 200:
        return None
    return float(lv[bg].mean() / max(float(pv[bg].mean()), 1.0))


class HealthMonitor:
    """Turns a stream of health checks into decisions, with hysteresis and a sustained-failure timer.

    `recovering` is True from the first failed check until a check passes the stricter limits, and
    should be fed back into the next `measure_plate_health(..., recovering=...)`. `tripped(now)` is
    True once checks have failed for `fail_s` without a pass in between. Checks that could not judge
    (None) change nothing.
    """

    def __init__(self, fail_s: float):
        self.fail_s = fail_s
        self.reset()

    def reset(self) -> None:
        self.health: PlateHealth | None = None  # latest verdict
        self.failing_since: float | None = None

    @property
    def recovering(self) -> bool:
        return self.failing_since is not None

    @property
    def ok(self) -> bool:
        return self.health is not None and self.failing_since is None

    def update(self, health: PlateHealth | None, now: float) -> bool:
        """Record a check. Returns True when the ok/failing state changed (worth a log line)."""
        if health is None:
            return False
        was_failing, first = self.failing_since is not None, self.health is None
        self.health = health
        if health.ok:
            self.failing_since = None
        elif self.failing_since is None:
            self.failing_since = now
        return first or was_failing != (self.failing_since is not None)

    def tripped(self, now: float) -> bool:
        return self.failing_since is not None and now - self.failing_since >= self.fail_s


class StalenessNudge:
    """Soft warning, separate from the hard limit: the residual has stayed above `residual` for
    `hold_s` while vanished, so the light has probably changed and a new plate would look better.
    Vanish stays on. `update()` returns True once per episode (time to beep); `showing` stays True
    until the residual has dropped back under 80% of the limit."""

    def __init__(self, residual: float, hold_s: float):
        self.residual, self.hold_s = residual, hold_s
        self.reset()

    def reset(self) -> None:
        self.above_since: float | None = None
        self.showing = False

    def update(self, health: PlateHealth | None, now: float, vanished: bool) -> bool:
        if not vanished:
            self.reset()
            return False
        if health is None:
            return False
        if health.residual > self.residual:
            if self.above_since is None:
                self.above_since = now
        elif health.residual < 0.8 * self.residual:
            self.reset()
        if self.above_since is not None and not self.showing and now - self.above_since >= self.hold_s:
            self.showing = True
            return True
        return False
