"""Synthetic camera grain for plate pixels that are shown instead of the live camera.

A plate is the mean of 30 frames, so it is much smoother than a live frame; a frozen or plate-only
picture next to (or instead of) live camera pixels looks dead. During plate capture the per-pixel
temporal std is measured (`PlateAccumulator.noise`) and stored with the plate. Here that profile
becomes fresh random noise on every frame:

* `GrainProfile` turns the stored std map into per-pixel amplitudes for a given (gain-matched) plate.
  The camera's noise depends on brightness, so the std is rescaled by the measured luma curve: where
  matching made the plate brighter or darker, the grain follows.
* `GrainSynth` adds the noise. Drawing 2.7 million normals per frame costs ~20 ms, so a few banks of
  unit-variance noise are drawn once and every frame uses a different bank at a random offset
  (independent between consecutive frames, no visible repeat). The arithmetic is int16 and saturating.
"""
from __future__ import annotations

import cv2
import numpy as np

NOISE_UNIT = 32       # bank value of one standard deviation of unit noise (int16)
AMP_UNIT = 64         # amplitude value of one level of std (int16): 1/64 level resolution
AMP_MAX = 1023        # amplitude cap = 16 levels of std
BANKS = 4
PAD = 64              # banks are this much larger than the frame; each frame takes a random crop
RATIO_LIMITS = (0.25, 4.0)  # how far the luma curve may rescale the stored std
AMP_SCALE = 4         # amplitudes are worked out at 1/4 size and enlarged: the stored std is smooth anyway


class GrainProfile:
    """Per-pixel noise std measured at plate capture, plus the camera's noise-vs-brightness curve."""

    def __init__(self, std: np.ndarray, plate_u8: np.ndarray):
        if std.shape != plate_u8.shape:
            raise ValueError(f"noise profile {std.shape} does not match the plate {plate_u8.shape}")
        self.std = std.astype(np.float32)
        plate_luma = cv2.cvtColor(plate_u8, cv2.COLOR_BGR2GRAY)
        h, w = plate_u8.shape[:2]
        self._full = (w, h)
        self._q = (max(1, w // AMP_SCALE), max(1, h // AMP_SCALE))
        self._std_q = cv2.resize(self.std, self._q, interpolation=cv2.INTER_AREA)
        self._plate_luma_q = cv2.cvtColor(cv2.resize(plate_u8, self._q, interpolation=cv2.INTER_AREA),
                                          cv2.COLOR_BGR2GRAY)
        mean_std = self.std.mean(axis=2)
        luma = plate_luma.ravel()
        total = np.bincount(luma, weights=mean_std.ravel(), minlength=256)
        count = np.bincount(luma, minlength=256).astype(np.float64)
        known = count > 0
        curve = np.zeros(256)
        curve[known] = total[known] / count[known]
        if known.any():  # luma values the plate never had borrow from their neighbours
            idx = np.arange(256)
            curve = np.interp(idx, idx[known], curve[known])
            curve = np.convolve(np.pad(curve, 6, mode="edge"), np.ones(13) / 13.0, mode="valid")
        self.curve = np.maximum(curve, 0.05).astype(np.float32)

    def amplitude(self, matched_u8: np.ndarray, scale: float = 1.0) -> np.ndarray:
        """int16 HxWx3 amplitudes (std * AMP_UNIT) for the pixels of `matched_u8`, the gain-matched plate.
        Worked out at 1/AMP_SCALE size (about 5 ms instead of 35 ms at 1280x720) and enlarged."""
        luma = cv2.cvtColor(cv2.resize(matched_u8, self._q, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        ratio = np.clip(self.curve[luma] / self.curve[self._plate_luma_q], *RATIO_LIMITS)
        amp = np.clip(self._std_q * ratio[..., None] * (scale * AMP_UNIT), 0, AMP_MAX).astype(np.int16)
        return cv2.resize(amp, self._full, interpolation=cv2.INTER_LINEAR)

    @staticmethod
    def uniform(std_levels: float, shape: tuple[int, int]) -> np.ndarray:
        """Amplitude map with one std everywhere (tests, and a fallback if ever needed)."""
        return np.full((shape[0], shape[1], 3), min(std_levels * AMP_UNIT, AMP_MAX), np.int16)


class GrainSynth:
    """Adds fresh grain to uint8 BGR images of one fixed size."""

    def __init__(self, size: tuple[int, int], seed: int | None = None):
        self.size = size  # (width, height)
        w, h = size
        rng = np.random.default_rng(seed)
        self._rng = rng
        self._banks = [np.rint(rng.standard_normal((h + PAD, w + PAD, 3), dtype=np.float32) * NOISE_UNIT).astype(np.int16)
                       for _ in range(BANKS)]
        self._last = -1

    def _noise(self, roi: tuple[int, int, int, int] | None) -> np.ndarray:
        """A different bank every call (so consecutive frames are independent), cropped at a random offset."""
        w, h = self.size
        if self._last < 0:
            k = int(self._rng.integers(BANKS))
        else:
            k = int(self._rng.integers(BANKS - 1))
            k += k >= self._last  # any bank but the previous one
        self._last = k
        ox, oy = (int(v) for v in self._rng.integers(0, PAD + 1, 2))
        x0, y0, x1, y1 = roi if roi else (0, 0, w, h)
        return self._banks[k][oy + y0:oy + y1, ox + x0:ox + x1]

    def apply(self, base: np.ndarray, amplitude: np.ndarray | None,
              roi: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """base + fresh grain, saturating. With `roi` = (x0, y0, x1, y1) `base` is just that rectangle of the
        frame and `amplitude` is the full-frame map (it is cropped here). None amplitude = no grain."""
        if amplitude is None:
            return base
        if roi:
            x0, y0, x1, y1 = roi
            amplitude = amplitude[y0:y1, x0:x1]
        noise = self._noise(roi)
        grain = cv2.multiply(np.ascontiguousarray(noise), np.ascontiguousarray(amplitude), dtype=cv2.CV_16S,
                             scale=1.0 / (NOISE_UNIT * AMP_UNIT))
        return cv2.add(base, grain, dtype=cv2.CV_8U)
