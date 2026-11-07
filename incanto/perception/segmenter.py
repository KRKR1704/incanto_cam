"""Person mask generation with edge and temporal smoothing."""
from __future__ import annotations

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from incanto.utils.clock import MonotonicMs


class Segmenter:
    """MediaPipe ImageSegmenter (selfie model) -> float32 person mask in 0..1.

    `segment()` takes the downscaled BGR perception frame and returns a mask at the full
    output resolution `out_size = (width, height)`. Smoothing is done at perception
    resolution (cheap) and the finished mask is upscaled last.
    """

    def __init__(self, model_path, out_size: tuple[int, int], *, blur_px: int = 5,
                 feather_low: float = 0.35, feather_high: float = 0.65,
                 temporal_alpha: float = 0.6, every_n: int = 1):
        self.out_size = out_size
        self.blur_px = (blur_px | 1) if blur_px > 0 else 0  # Gaussian kernels must be odd
        self.feather_low, self.feather_high = feather_low, feather_high
        self.alpha = temporal_alpha
        self.every_n = max(1, every_n)
        self._clock = MonotonicMs()
        self._frame_no = 0
        self._prev_small: np.ndarray | None = None
        self._last_full = np.zeros((out_size[1], out_size[0]), np.float32)
        self._segmenter = vision.ImageSegmenter.create_from_options(vision.ImageSegmenterOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            output_confidence_masks=True,
            output_category_mask=False,
        ))

    def _smooth_edges(self, mask: np.ndarray) -> np.ndarray:
        if self.blur_px:
            mask = cv2.GaussianBlur(mask, (self.blur_px, self.blur_px), 0)
        # Feathered threshold: smoothstep between low and high.
        t = np.clip((mask - self.feather_low) / (self.feather_high - self.feather_low), 0.0, 1.0)
        return t * t * (3.0 - 2.0 * t)

    def segment(self, small_bgr: np.ndarray) -> np.ndarray:
        run = self._frame_no % self.every_n == 0
        self._frame_no += 1
        if not run:
            return self._last_full  # reuse the last mask on skipped frames
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        result = self._segmenter.segment_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._clock.now())
        raw = np.array(result.confidence_masks[0].numpy_view(), dtype=np.float32).reshape(rgb.shape[:2])
        mask = self._smooth_edges(raw)
        if self._prev_small is not None and self._prev_small.shape == mask.shape:
            mask = self.alpha * mask + (1.0 - self.alpha) * self._prev_small
        self._prev_small = mask
        self._last_full = cv2.resize(mask, self.out_size, interpolation=cv2.INTER_LINEAR).astype(
            np.float32, copy=False)
        return self._last_full

    def close(self) -> None:
        self._segmenter.close()
