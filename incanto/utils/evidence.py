"""Scripted evidence capture: prompts on the preview, PNG screenshots, and a real vcam grab."""
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

# (file name, on-screen instruction)
STEPS = [
    ("1-just-me.png", "Step 1/3: just you, hands DOWN"),
    ("2-one-hand.png", "Step 2/3: raise ONE hand, palm to camera"),
    ("3-both-hands.png", "Step 3/3: raise BOTH hands, palms to camera"),
]
PREPARE_S = 6.0   # time to get into position before the shot
HOLD_S = 2.0      # shown after the shot ("captured")


def _beep(freq: int) -> None:
    try:
        import winsound
        winsound.Beep(freq, 120)
    except Exception:  # noqa: BLE001 - audio feedback is optional
        pass


def grab_vcam_frames(out_dir: Path, name: str = "vcam-output") -> list[Path]:
    """Open every DirectShow device we can and save one frame from each.

    The physical webcam is already held by Incanto, so the devices that open are the virtual
    camera(s). The caller inspects the files to confirm which one is the virtual camera.
    """
    saved = []
    for index in range(6):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue
        frame = None
        for _ in range(15):
            ok, f = cap.read()
            if ok and f is not None:
                frame = f
        cap.release()
        if frame is not None:
            path = out_dir / f"{name}-dshow{index}.png"
            cv2.imwrite(str(path), frame)
            log.info("Saved DirectShow device %d frame (%dx%d) -> %s", index, frame.shape[1],
                     frame.shape[0], path.name)
            saved.append(path)
    return saved


class EvidenceSession:
    def __init__(self, out_dir: str | Path):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self._step = 0
        self._step_start: float | None = None
        self._captured = False
        self.done = False

    def update(self, preview: np.ndarray, vcam_frame_check=None) -> None:
        """Draw the current prompt onto `preview` (a copy) and save it when the time comes."""
        if self.done:
            return
        now = time.perf_counter()
        if self._step_start is None:
            self._step_start = now
        name, prompt = STEPS[self._step]
        elapsed = now - self._step_start
        remaining = PREPARE_S - elapsed
        shot = None
        if not self._captured and remaining <= 0:
            shot = preview.copy()  # includes HUD, no prompt text
            self._captured = True
        if not self._captured:
            text = f"{prompt}  ({remaining:.0f})"
        else:
            text = f"{prompt}  - captured"
        cv2.putText(preview, text, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(preview, text, (16, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
        if shot is not None:
            cv2.imwrite(str(self.out_dir / name), shot)
            log.info("Saved %s", name)
            _beep(1200)
            if self._step == 0 and vcam_frame_check is not None:
                vcam_frame_check()
        elif self._captured and elapsed >= PREPARE_S + HOLD_S:
            self._step += 1
            self._step_start = None
            self._captured = False
            if self._step >= len(STEPS):
                self.done = True
                _beep(600)
