"""Benchmark HandLandmarker cost for num_hands=1 vs 2 with 0, 1 and 2 hands visible.

Records a short clip for each pose from the webcam (with on-screen prompts and a beep), then
replays the SAME frames through each num_hands setting so the comparison is fair.

Run from the project root:  python scripts/bench_hands.py [--out docs/evidence/phase-2/bench-hands.txt]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from incanto.config import load_config  # noqa: E402
from incanto.io.camera import Camera  # noqa: E402
from incanto.perception.hands import HandTracker  # noqa: E402
from incanto.utils.imaging import downscale  # noqa: E402

POSES = [
    (0, "NO hands in view (hands down / out of frame)"),
    (1, "ONE hand raised, palm to camera"),
    (2, "BOTH hands raised, palms to camera"),
]
PREP_S, CLIP_S = 5.0, 5.0


def beep(freq: int) -> None:
    try:
        import winsound
        winsound.Beep(freq, 150)
    except Exception:  # noqa: BLE001
        pass


def record(cfg) -> dict[int, list]:
    cam = Camera(cfg.video.width, cfg.video.height, cfg.video.fps)
    clips: dict[int, list] = {}
    try:
        for n_hands, prompt in POSES:
            frames = []
            start = time.perf_counter()
            beeped = False
            while True:
                elapsed = time.perf_counter() - start
                frame = cam.read()
                shown = frame.copy()
                label = f"{prompt}  "
                if elapsed < PREP_S:
                    label += f"(get ready: {PREP_S - elapsed:.0f})"
                else:
                    if not beeped:
                        beep(1200)
                        beeped = True
                    label += "RECORDING"
                    frames.append(downscale(frame, cfg.video.perception_width))
                cv2.putText(shown, label, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(shown, label, (16, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2, cv2.LINE_AA)
                cv2.imshow("bench_hands", shown)
                cv2.waitKey(1)
                if elapsed >= PREP_S + CLIP_S:
                    break
            clips[n_hands] = frames
            print(f"recorded {len(frames)} frames for pose '{n_hands} hand(s)'", flush=True)
    finally:
        cam.release()
        cv2.destroyAllWindows()
    beep(600)
    return clips


def bench(cfg, clips) -> list[str]:
    model = cfg.models.resolve("hand_landmarker")
    p = cfg.perception
    lines = [f"{'num_hands':>9} {'visible':>8} {'detected(avg)':>14} {'mean ms':>8} {'p50':>7} {'p90':>7}  frames"]
    for num_hands in (1, 2):
        for visible, frames in clips.items():
            tracker = HandTracker(model, num_hands=num_hands,
                                  min_detection_confidence=p.hand_min_detection_confidence,
                                  min_presence_confidence=p.hand_min_presence_confidence,
                                  min_tracking_confidence=p.hand_min_tracking_confidence)
            try:
                for f in frames[:10]:  # warm-up, not timed
                    tracker.detect(f)
                times, found = [], []
                for f in frames[10:]:
                    t0 = time.perf_counter()
                    hands = tracker.detect(f)
                    times.append(1000 * (time.perf_counter() - t0))
                    found.append(len(hands))
            finally:
                tracker.close()
            q = statistics.quantiles(times, n=10)
            lines.append(f"{num_hands:>9} {visible:>8} {statistics.mean(found):>14.2f} "
                         f"{statistics.mean(times):>8.1f} {statistics.median(times):>7.1f} {q[8]:>7.1f}  {len(times)}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="docs/evidence/phase-2/bench-hands.txt")
    args = parser.parse_args()
    cfg = load_config()
    clips = record(cfg)
    lines = bench(cfg, clips)
    text = "\n".join(lines)
    print(text)
    Path(args.out).write_text(
        "HandLandmarker inference time per frame at perception_width="
        f"{cfg.video.perception_width}; identical recorded frames per row group.\n\n{text}\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
