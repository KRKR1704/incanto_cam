"""Measure the audio-video offset of a recorded clip from claps.

    .venv\\Scripts\\python scripts/check_av_sync.py recordings/<clip> [--report docs/evidence/phase-4/av-sync.txt]

For each clap: the sound spike in the audio, and the moment of contact in the video. offset = audio time - video time of
the same clap, both on the camera's clock (positive: audio stamped later than the picture); target: within +-1 frame.

Which spikes are claps: with tests/fixtures/labels.yaml the first sound spike after each labelled clap cue (people react
to a beep 0.5-1 s late, so the search runs from cue - tolerance to cue + 2 s); without labels, every spike.

The video moment is found from MOTION, not from hand tracking: the tracker loses both hands for 150-350 ms around a clap
(measured on the guided clip: lost from about -80 ms to +100..250 ms, in every clap), so "the frame where the wrists are
closest" cannot be seen. Hands move fast right up to the contact and almost stop after it, so the last frame still showing
the fast approach and the first frame showing it stopped bracket the contact (incanto/utils/avsync.motion_bracket). Frames
the camera only repeated are skipped; many webcams, this one included, update the picture at about 15 fps in a dim room, so
a bracket is about 66 ms wide and one clap pins the offset only to that. The median over many claps, and the intersection of
all brackets, are the real answer. `--wrists` also runs the hand tracker and reports how often it lost the hands.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from incanto.config import load_config  # noqa: E402
from incanto.guided import LABELS_PATH, load_labels  # noqa: E402
from incanto.io.replay import Clip, ClipError  # noqa: E402
from incanto.utils.avsync import detect_claps_audio, motion_bracket, summarize_brackets  # noqa: E402

LEAD_S, LAG_S = 0.9, 0.5          # frames from spike - LEAD to spike + LAG are examined
CHANGED_LEVELS = 12               # a pixel "changed" if its brightness moved by more than this between two frames


def motion_energy(clip: Clip, t0: float, t1: float):
    """(frame times, percent of pixels that changed since the previous frame) for frames captured in [t0, t1]."""
    i0, i1 = clip.index_at(t0), clip.index_at(t1) + 1
    times, energy, prev = [], [], None
    for i, t, frame in clip.frames(max(0, i0 - 1), i1):
        g = cv2.GaussianBlur(cv2.cvtColor(cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY),
                             (5, 5), 0).astype(np.float32)
        times.append(t)
        energy.append(np.nan if prev is None else float((np.abs(g - prev) > CHANGED_LEVELS).mean() * 100))
        prev = g
    return np.array(times), np.array(energy)


def hands_lost_report(clip: Clip, spikes: list[float], cfg) -> list[str]:
    """For each spike: the span (ms from the spike) in which no two hands were tracked, to show why landmarks cannot time a clap."""
    from incanto.perception.hands import HandTracker
    from incanto.utils.imaging import downscale
    tracker = HandTracker(cfg.models.resolve("hand_landmarker"), num_hands=2)
    out, last = [], -1
    try:
        for n, s in enumerate(spikes, 1):
            seen = []
            i0, i1 = clip.index_at(s - LEAD_S), clip.index_at(s + LAG_S) + 1
            for i, t, frame in clip.frames(i0, i1):
                ms = max(int(round((t - clip.first_time) * 1000)), last + 1)
                last = ms
                seen.append((t - s, len(tracker.detect(downscale(frame, 640), ms))))
            lost = [dt for dt, k in seen if k < 2 and -0.4 < dt < 0.4]
            out.append(f"  clap {n}: two hands tracked until {max([dt for dt, k in seen if k >= 2 and dt < 0] or [float('nan')]) * 1000:+.0f} ms, "
                       f"lost for {len(lost)} of {sum(1 for dt, _ in seen if -0.4 < dt < 0.4)} frames within +-400 ms")
    finally:
        tracker.close()
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--report", help="also write the report to this .txt file")
    ap.add_argument("--labels", default=str(LABELS_PATH), help="labels.yaml used to pick the clap spikes")
    ap.add_argument("--wrists", action="store_true", help="also report how often hand tracking lost the hands")
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
    lines = [f"A/V sync check: {clip.path.name}",
             f"clip: {clip.n_frames} frames over {clip.duration:.1f}s; audio {clip.samplerate} Hz from "
             f"{clip.meta['audio'].get('device')}, starting {1000 * (clip.audio_start_time - clip.first_time):.0f} ms after the first frame; "
             f"audio.offset_ms in config: {cfg.audio.offset_ms:g} (this check uses the raw recorded times)"]
    a, t0 = clip.audio(), clip.audio_start_time
    spikes_all = detect_claps_audio(a, clip.samplerate, t0, min_rise_db=8.0, floor_dbfs=-55.0)
    cues = [g for g in load_labels(Path(args.labels))["clips"].get(clip.path.name, {}).get("gestures", []) if g["gesture"] == "clap"]
    if cues:
        spikes, picked = [], []
        for g in cues:
            c = clip.first_time + g["t"]
            cand = [s for s in spikes_all if c - g["tolerance_s"] <= s <= c + 2.0]
            if cand:
                spikes.append(cand[0])
                picked.append(g["t"])
        lines.append(f"labelled clap cues: {len(cues)}; a sound spike was found for {len(spikes)} of them "
                     f"(searched from cue - tolerance to cue + 2 s)")
    else:
        spikes, picked = list(spikes_all), [None] * len(spikes_all)
        lines.append(f"no labelled claps for this clip: using all {len(spikes)} sound spikes")
    if not spikes:
        lines.append("No claps found: clap sharply (flat hands, loud) a few times in front of the camera and record again.")
        return finish(lines, args.report, 1)
    lines.append("")
    lines.append(f"{'#':>2} {'cue (s)':>8} {'sound (s)':>9} {'last fast':>10} {'first stop':>10}  {'offset (audio - video), ms':>27}  note")
    brackets = []
    for n, (s, cue) in enumerate(zip(spikes, picked), 1):
        times, energy = motion_energy(clip, s - LEAD_S, s + LAG_S)
        br = motion_bracket(times, energy, s)
        rel = s - clip.first_time
        if br is None:
            lines.append(f"{n:>2} {'' if cue is None else format(cue, '.2f'):>8} {rel:>9.3f} {'-':>10} {'-':>10}  {'-':>27}  no clear approach-then-stop in the video")
            continue
        tl, tf = br
        lo, hi = 1000 * (s - tf), 1000 * (s - tl)
        brackets.append((lo, hi))
        lines.append(f"{n:>2} {'' if cue is None else format(cue, '.2f'):>8} {rel:>9.3f} {tl - clip.first_time:>10.3f} {tf - clip.first_time:>10.3f}  "
                     f"{lo:>+9.0f} .. {hi:>+4.0f}  (mid {(lo + hi) / 2:+.0f})")
    summary = summarize_brackets(brackets)
    lines.append("")
    if summary["n"] == 0:
        lines.append("RESULT: no clap could be timed in the video.")
        return finish(lines, args.report, 1)
    lines.append(f"offset = audio time - video time of the same clap; each clap gives an interval because the picture updates every "
                 f"~66 ms (the contact lies between the last fast frame and the first stopped one)")
    lines.append(f"RESULT: {summary['n']} claps | median of midpoints {summary['median_ms']:+.1f} ms | mean {summary['mean_ms']:+.1f} ms | "
                 f"spread: std {summary['std_ms']:.1f} ms, MAD {summary['mad_ms']:.1f} ms | the median lies inside {summary['contain_median']} of "
                 f"{summary['n']} intervals")
    inter = summary["intersection_ms"]
    lines.append("intersection of all intervals: " + (f"{inter[0]:+.0f} .. {inter[1]:+.0f} ms (one constant offset explains every clap)"
                                                      if inter else "EMPTY (no single offset explains every clap)"))
    if summary["within_one_frame"] and (inter is None or (inter[0] > -34 and inter[1] < 34) or abs(summary["median_ms"]) <= 33):
        lines.append("PASS: median offset within +-1 frame (33 ms): no audio.offset_ms correction needed."
                     + ("" if inter and abs(inter[0]) <= 33 else "  (the interval's edge reaches slightly beyond 33 ms, i.e. within the camera's own 66 ms picture step)"))
    else:
        lines.append(f"OFFSET beyond 1 frame. Set audio.offset_ms: {summary['suggested_audio_offset_ms']:.0f} in config.yaml "
                     "(added to every audio timestamp), then re-run with it to confirm.")
    if args.wrists:
        lines.append("")
        lines.append("hand tracking around each clap (why wrist landmarks cannot time a clap):")
        lines.extend(hands_lost_report(clip, spikes, cfg))
    return finish(lines, args.report, 0)


def finish(lines: list[str], report: str | None, code: int) -> int:
    text = "\n".join(lines)
    print(text)
    if report:
        Path(report).parent.mkdir(parents=True, exist_ok=True)
        Path(report).write_text(text + "\n", encoding="utf-8")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
