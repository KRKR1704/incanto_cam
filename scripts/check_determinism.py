"""Replay a clip twice and compare the perception output frame by frame.

    .venv\\Scripts\\python scripts/check_determinism.py recordings/<clip> [--frames N] [--report docs/evidence/phase-4/determinism.txt]

Two independent sets of hand tracker and segmenter run in lockstep over the same recorded frames, with the replay's
deterministic timestamps (recorded times, not the wall clock). Reports the maximum difference in hand landmarks and in
the person mask, and whether the hands (count, handedness) agree on every frame.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from incanto.config import load_config  # noqa: E402
from incanto.io.replay import Clip, ClipError  # noqa: E402
from incanto.perception.hands import HandTracker  # noqa: E402
from incanto.perception.segmenter import Segmenter  # noqa: E402
from incanto.utils.imaging import downscale  # noqa: E402


def build(cfg, size):
    p = cfg.perception
    tracker = HandTracker(cfg.models.resolve("hand_landmarker"), num_hands=p.num_hands,
                          min_detection_confidence=p.hand_min_detection_confidence,
                          min_presence_confidence=p.hand_min_presence_confidence,
                          min_tracking_confidence=p.hand_min_tracking_confidence)
    seg = Segmenter(cfg.models.resolve("selfie_segmenter"), size, blur_px=p.mask_blur_px, feather_low=p.mask_feather_low,
                    feather_high=p.mask_feather_high, temporal_alpha=p.mask_temporal_alpha, every_n=p.segment_every_n,
                    temporal_mode=p.mask_temporal, release_decay=p.mask_release_decay)
    return tracker, seg


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--frames", type=int, help="only the first N frames")
    ap.add_argument("--report", help="also write the report to this .txt file")
    args = ap.parse_args()
    cfg = load_config()
    try:
        clip = Clip.load(args.clip)
    except ClipError as exc:
        print(exc, file=sys.stderr)
        return 2
    size = (clip.width, clip.height)
    (tr_a, sg_a), (tr_b, sg_b) = build(cfg, size), build(cfg, size)
    n = min(clip.n_frames, args.frames or clip.n_frames)
    max_lm = max_mask = 0.0
    mask_diff_px = mask_total_px = 0
    hand_mismatch = frames_with_hands = lm_nonzero = mask_nonzero = 0
    last_ms, t_start = -1, time.perf_counter()
    prev_gray, repeats = None, 0   # frames the camera repeated (identical picture to the one before)
    for i, t, frame in clip.frames():
        if i >= n:
            break
        ms = max(int(round((t - clip.first_time) * 1000)), last_ms + 1)  # what ReplaySource.timestamp_ms() gives
        last_ms = ms
        small = downscale(frame, cfg.video.perception_width)
        gray = cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        if prev_gray is not None and float(cv2.absdiff(gray, prev_gray).mean()) < 0.02:
            repeats += 1
        prev_gray = gray
        ha, hb = tr_a.detect(small, ms), tr_b.detect(small, ms)
        sg_a.segment(small, ms)
        sg_b.segment(small, ms)
        frames_with_hands += bool(ha or hb)
        if len(ha) != len(hb) or [h.handedness for h in ha] != [h.handedness for h in hb]:
            hand_mismatch += 1
        else:
            for x, y in zip(ha, hb):
                d = float(np.abs(x.landmarks - y.landmarks).max())
                max_lm = max(max_lm, d)
                lm_nonzero += d > 0
        d = np.abs(sg_a.small - sg_b.small)
        max_mask = max(max_mask, float(d.max()))
        mask_nonzero += bool(d.max() > 0)
        mask_diff_px += int((d > 1e-3).sum())
        mask_total_px += d.size
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{n} frames", file=sys.stderr)
    for obj in (tr_a, tr_b, sg_a, sg_b):
        obj.close()
    lines = [f"Determinism check: {clip.path}",
             f"{n} frames replayed twice with independent hand trackers and segmenters, in {time.perf_counter() - t_start:.0f} s "
             f"(timestamps from the recorded frame times; hands synchronous)",
             f"camera: {repeats} of {n} frames ({100.0 * repeats / max(n, 1):.0f}%) show the same picture as the frame before "
             f"(the driver repeating a frame: the picture updates at about {30.0 * (1 - repeats / max(n, 1)):.0f} fps)",
             f"hands: present on {frames_with_hands} frames | frames where count/handedness differ between runs: {hand_mismatch} | "
             f"max landmark difference {max_lm:.3g} (normalised units; 1.0 = frame width) | frames with any landmark difference: {lm_nonzero}",
             f"mask: max per-pixel difference {max_mask:.3g} (0..1) | frames with any difference: {mask_nonzero} | "
             f"pixels differing by more than 0.001: {mask_diff_px} of {mask_total_px}"]
    identical = hand_mismatch == 0 and max_lm == 0.0 and max_mask == 0.0
    near = hand_mismatch == 0 and max_lm < 1e-4 and max_mask < 1e-3
    lines.append("RESULT: " + ("IDENTICAL hands and masks on every frame." if identical else
                               "near-identical (differences below 1e-4 landmark / 1e-3 mask)." if near else
                               "NOT deterministic: differences above the tolerance, see above."))
    text = "\n".join(lines)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0 if (identical or near) else 1


if __name__ == "__main__":
    raise SystemExit(main())
