"""Measure the audio-video offset of a recorded clip from claps.

    .venv\\Scripts\\python scripts/check_av_sync.py recordings/<clip> [--report docs/evidence/phase-4/av-sync.txt]

Finds sound spikes in the clip's audio, then, around each, the video frame where your wrists meet (hand tracking, run
synchronously on just those frames). offset = audio time - video time of the same clap; target: within +-1 frame (33 ms).
If there is a consistent offset it suggests `audio.offset_ms` (added to every audio timestamp, live and replay).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from incanto.config import load_config  # noqa: E402
from incanto.io.replay import Clip, ClipError  # noqa: E402
from incanto.perception.hands import HandTracker  # noqa: E402
from incanto.utils.avsync import contact_time, detect_claps_audio, summarize  # noqa: E402
from incanto.utils.imaging import downscale  # noqa: E402

WINDOW_S = 0.7   # frames within +-this of an audio spike are examined


def wrist_distances(clip: Clip, tracker: HandTracker, windows: list[tuple[float, float]], width: int):
    """Per window: (frame times, distance between the two wrists as a fraction of the frame width, NaN if a hand is missing)."""
    out = [([], []) for _ in windows]
    last_ms = -1
    for i, t, frame in clip.frames():
        for k, (a, b) in enumerate(windows):
            if a <= t <= b:
                ms = max(int(round((t - clip.first_time) * 1000)), last_ms + 1)
                last_ms = ms
                hands = tracker.detect(downscale(frame, width), ms)
                d = np.nan
                if len(hands) >= 2:
                    w0, w1 = hands[0].landmarks[0, :2], hands[1].landmarks[0, :2]
                    d = float(np.linalg.norm((w0 - w1) * (1.0, clip.height / clip.width)))
                out[k][0].append(t)
                out[k][1].append(d)
                break
    return [(np.array(t), np.array(d)) for t, d in out]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--report", help="also write the report to this .txt file")
    ap.add_argument("--audio-from", type=float, default=0.0, help="ignore audio before this many seconds into the clip")
    ap.add_argument("--audio-to", type=float, default=None, help="ignore audio after this many seconds into the clip")
    args = ap.parse_args()
    cfg = load_config()
    try:
        clip = Clip.load(args.clip)
    except ClipError as exc:
        print(exc, file=sys.stderr)
        return 2
    if not clip.has_audio:
        print("This clip has no audio: nothing to sync.", file=sys.stderr)
        return 2
    lines = [f"A/V sync check: {clip.path}",
             f"clip: {clip.n_frames} frames over {clip.duration:.1f}s, audio {clip.samplerate} Hz "
             f"from {clip.meta['audio'].get('device')}, audio.offset_ms in effect for live: {cfg.audio.offset_ms:g} "
             "(this check uses the raw recorded times)"]
    audio, t0 = clip.audio(), clip.audio_start_time
    lo = clip.first_time + args.audio_from
    hi = clip.first_time + (args.audio_to if args.audio_to is not None else clip.duration)
    claps = [t for t in detect_claps_audio(audio, clip.samplerate, t0) if lo <= t <= hi]
    lines.append(f"sound spikes found: {len(claps)}")
    if not claps:
        lines.append("No claps found: clap sharply (flat hands, loud) a few times in front of the camera and record again.")
        return _finish(lines, args.report, 1)
    tracker = HandTracker(cfg.models.resolve("hand_landmarker"), num_hands=2,
                          min_detection_confidence=cfg.perception.hand_min_detection_confidence,
                          min_presence_confidence=cfg.perception.hand_min_presence_confidence,
                          min_tracking_confidence=cfg.perception.hand_min_tracking_confidence)
    try:
        series = wrist_distances(clip, tracker, [(c - WINDOW_S, c + WINDOW_S) for c in claps], cfg.video.perception_width)
    finally:
        tracker.close()
    offsets = []
    lines.append(f"{'#':>2} {'audio t (s)':>11} {'video t (s)':>11} {'offset ms':>10}  how")
    for n, (c, (times, dist)) in enumerate(zip(claps, series), 1):
        res = contact_time(times, dist) if len(times) else None
        if res is None or np.isnan(dist).all():
            lines.append(f"{n:>2} {c - clip.first_time:>11.3f} {'-':>11} {'-':>10}  no two-hand approach seen around this spike")
            continue
        tv, how = res
        # a clap needs the wrists to actually come together; ignore spikes where they never got close
        if np.nanmin(dist) > 0.25:
            lines.append(f"{n:>2} {c - clip.first_time:>11.3f} {tv - clip.first_time:>11.3f} {'-':>10}  wrists never came close "
                         f"(min {np.nanmin(dist):.2f} of the frame width): not a clap, skipped")
            continue
        off = (c - tv) * 1000.0
        offsets.append(off)
        lines.append(f"{n:>2} {c - clip.first_time:>11.3f} {tv - clip.first_time:>11.3f} {off:>10.1f}  {how}")
    s = summarize(offsets)
    lines.append("")
    if s["n"] == 0:
        lines.append("RESULT: no usable claps (audio spike without the wrists meeting).")
        return _finish(lines, args.report, 1)
    lines.append(f"RESULT: {s['n']} claps | median offset {s['median_ms']:+.1f} ms | spread (MAD) {s['mad_ms']:.1f} ms | "
                 f"range {s['min_ms']:+.1f} .. {s['max_ms']:+.1f} ms")
    lines.append("offset = audio time - video time of the same clap (positive: audio stamped later than the picture)")
    if s["within_one_frame"]:
        lines.append("PASS: median within +-1 frame (33 ms); no correction needed.")
    else:
        lines.append(f"OFFSET beyond 1 frame. Set audio.offset_ms: {s['suggested_audio_offset_ms']:.0f} in config.yaml "
                     "(added to every audio timestamp), then re-run with the new value to confirm.")
    if s["mad_ms"] > 33:
        lines.append("WARNING: the offsets are scattered (MAD > 1 frame): not a stable offset, or the claps were not clean.")
    return _finish(lines, args.report, 0)


def _finish(lines: list[str], report: str | None, code: int) -> int:
    text = "\n".join(lines)
    print(text)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
