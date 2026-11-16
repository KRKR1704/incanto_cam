"""Unique frames per second of a recorded clip, by segment of the guided plan.

    .venv\Scripts\python scripts/check_unique_fps.py recordings/<clip> [--plan plans/fixture-plan.yaml] [--report out.txt]

A frame is "unique" when it differs from the previous one (the driver repeats frames when the camera drops its real frame
rate). Uses the same test as the live monitor (incanto.utils.fps.UniqueFps) and also reports the mean brightness. Segment
boundaries come from the plan and the labelled cue times (the clip starts a fraction of a second before the plan).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from incanto.guided import LABELS_PATH, load_labels, load_plan  # noqa: E402
from incanto.io.replay import Clip, ClipError  # noqa: E402
from incanto.utils.fps import UniqueFps  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--plan", default=str(ROOT / "plans" / "fixture-plan.yaml"))
    ap.add_argument("--report")
    ap.add_argument("--labels", default=str(LABELS_PATH))
    args = ap.parse_args()
    try:
        clip = Clip.load(args.clip)
    except ClipError as exc:
        print(exc, file=sys.stderr)
        return 2
    plan = load_plan(args.plan)
    entry = load_labels(Path(args.labels))["clips"].get(clip.path.name, {})
    first = (entry.get("gestures") or [{}])[0]
    cue0 = first.get("cue_time", first.get("t"))
    offset = (cue0 - plan.cues[0].t) if cue0 is not None else 0.0   # clip time = plan time + offset
    rests = sorted(plan.rests, key=lambda r: r.start)
    segs = [("lead-in + snap cues", 0.0, rests[0].start), ("rest", rests[0].start, rests[0].end),
            ("clap cues", rests[0].end, rests[1].start), ("rest", rests[1].start, rests[1].end),
            ("thumbs_up cues", rests[1].end, rests[2].start), ("rest", rests[2].start, rests[2].end),
            ("talk with hands", plan.talks[0].start, plan.talks[0].end)]
    flags, luma, times, diffs, prev = [], [], [], [], None
    mon = UniqueFps()
    for i, t, frame in clip.frames():
        flags.append(mon.update(frame, t))              # the live monitor's own test
        g = cv2.cvtColor(cv2.resize(frame, UniqueFps.SIZE, interpolation=cv2.INTER_AREA), cv2.COLOR_BGR2GRAY)
        luma.append(float(g.mean()))
        diffs.append(0.0 if prev is None else float(cv2.absdiff(g, prev).mean()))
        prev = g
        times.append(t - clip.first_time)
    flags, luma, times, diffs = np.array(flags), np.array(luma), np.array(times), np.array(diffs[1:])
    pct = lambda m: 100 * float(np.mean(m))  # noqa: E731
    lines = [f"Unique frames per second: {clip.path.name} ({clip.n_frames} frames, {clip.duration:.0f} s, delivered "
             f"{(clip.n_frames - 1) / clip.duration:.2f} fps)",
             "unique = differs from the previous frame (mean abs difference of 96x54 grey >= 0.03 levels); same test as the live monitor",
             f"frame-difference histogram (levels): exactly 0: {pct(diffs == 0):.1f}% | 0-0.03: {pct((diffs > 0) & (diffs < 0.03)):.1f}% | "
             f"0.03-0.1: {pct((diffs >= 0.03) & (diffs < 0.1)):.1f}% | 0.1-0.3: {pct((diffs >= 0.1) & (diffs < 0.3)):.1f}% | >=0.3: {pct(diffs >= 0.3):.1f}%"
             "  (nothing between 0 and 0.3 means the threshold is not borderline)", "",
             f"{'segment':<22} {'clip time (s)':>15} {'frames':>7} {'unique fps':>10} {'mean luma':>9}"]
    for name, a, b in segs:
        m = (times >= a + offset) & (times < b + offset)
        if m.sum() > 1:
            lines.append(f"{name:<22} {a + offset:>7.1f}-{b + offset:<7.1f} {int(m.sum()):>7} "
                         f"{flags[m].sum() / (times[m][-1] - times[m][0]):>10.1f} {luma[m].mean():>9.1f}")
    lines.append(f"{'whole clip':<22} {0:>7.1f}-{times[-1]:<7.1f} {len(times):>7} {flags.sum() / times[-1]:>10.1f} {luma.mean():>9.1f}")
    sim, state, on_s, change = UniqueFps(), False, 0.0, []
    for k, (t, f) in enumerate(zip(times, flags)):
        sim.feed(bool(f), t)
        if k and sim.warning:
            on_s += t - times[k - 1]
        if sim.warning != state:
            state = sim.warning
            change.append(f"{'ON' if state else 'off'} at {t:.0f} s")
    lines += ["", f"live monitor on this clip (warns when the last 10 s stay under 20 unique fps): the warning would show for "
                  f"{on_s:.0f} of {times[-1]:.0f} s ({100 * on_s / times[-1]:.0f}%); changes: {', '.join(change[:12]) or 'none'}"]
    text = "\n".join(lines)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
