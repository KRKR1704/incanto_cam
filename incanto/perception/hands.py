"""MediaPipe hand landmark tracking."""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

from incanto.utils.clock import MonotonicMs


@dataclass(frozen=True)
class Hand:
    """One tracked hand. No MediaPipe types leak out of this module."""
    handedness: str          # "Left" or "Right", as reported by MediaPipe (see note below)
    score: float             # handedness confidence 0..1
    landmarks: np.ndarray    # (21, 3) float32; x, y normalized to the frame (0..1), z relative depth


# Note: MediaPipe assumes a mirrored ("selfie view") image. Our frames are not mirrored,
# so the label is from the camera's point of view and may be the opposite of the user's
# actual hand. Gesture code should rely on geometry rather than this label where possible.


class HandTracker:
    def __init__(self, model_path, *, num_hands: int = 2, min_detection_confidence: float = 0.5,
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5):
        self._clock = MonotonicMs()
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        ))

    def detect(self, small_bgr: np.ndarray) -> list[Hand]:
        """Detect hands in the downscaled BGR perception frame."""
        rgb = cv2.cvtColor(small_bgr, cv2.COLOR_BGR2RGB)
        result = self._landmarker.detect_for_video(
            mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), self._clock.now())
        hands = []
        for lms, handed in zip(result.hand_landmarks, result.handedness):
            pts = np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32)
            hands.append(Hand(handed[0].category_name, float(handed[0].score), pts))
        return hands

    def close(self) -> None:
        self._landmarker.close()
