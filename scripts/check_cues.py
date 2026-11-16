"""Sanity-check the labelled cues of a guided clip against what is actually in the recording.

    .venv\\Scripts\\python scripts/check_cues.py recordings/<clip> [--report docs/evidence/phase-4/cue-check.txt]

For every cue in tests/fixtures/labels.yaml: is a hand visible near the cue time, did a sound spike happen (snaps and
claps), and for thumbs-ups was a thumbs-up pose seen? It reports WHEN, relative to the cue beep, so that cues that look
flubbed or off-time can be reviewed. It never changes labels.yaml; whether to drop or adjust a cue is a human decision.
A cue is "late" when the gesture happened but after the label window (cue + tolerance); people react to a beep 0.5-1 s late.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from incanto.config import load_config  # noqa: E402
from incanto.guided import LABELS_PATH, load_labels  # noqa: E402
from incanto.io.replay import Clip, ClipError  # noqa: E402
from incanto.perception.hands import HandTracker  # noqa: E402
from incanto.utils.avsync import detect_claps_audio  # noqa: E402
from incanto.utils.imaging import downscale  # noqa: E402

PRE_S, POST_S = 0.8, 2.2   # hands are examined from cue - PRE to cue + POST
TIPS, PIPS = (8, 12, 16, 20), (6, 10, 14, 18)


def thumbs_up_like(hand) -> bool:
    """Thumb pointing up above a curled fist (image y grows downward). A rough pose test for reporting, not the Phase 5 detector."""
    lm = hand.landmarks
    thumb_up = lm[4, 1] < lm[3, 1] < lm[2, 1] and lm[4, 1] < lm[TIPS, 1].min() - 0.02
    curled = all(lm[t, 1] > lm[p, 1] for t, p in zip(TIPS, PIPS))
    return bool(thumb_up and curled)


def episodes(times: list[float], gap: float = 0.25) -> list[tuple[float, float]]:
    """Group frame times into runs (start, end); a gap longer than `gap` seconds starts a new run."""
    out = []
    for t in times:
        if out and t - out[-1][1] <= gap:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def scan(clip: Clip, tracker, t0: float, t1: float, last_ms: list[int]):
    """[(time, number of hands, thumbs-up-like hand present)] for frames in [t0, t1]."""
    out = []
    for i, t, frame in clip.frames(clip.index_at(t0), clip.index_at(t1) + 1):
        ms = max(int(round((t - clip.first_time) * 1000)), last_ms[0] + 1)
        last_ms[0] = ms
        hands = tracker.detect(downscale(frame, 640), ms)
        out.append((t, len(hands), any(thumbs_up_like(h) for h in hands)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--report")
    ap.add_argument("--labels", default=str(LABELS_PATH))
    args = ap.parse_args()
    cfg = load_config()
    try:
        clip = Clip.load(args.clip)
    except ClipError as exc:
        print(exc, file=sys.stderr)
        return 2
    entry = load_labels(Path(args.labels))["clips"].get(clip.path.name)
    if entry is None:
        print(f"{clip.path.name} is not in {args.labels}", file=sys.stderr)
        return 2
    spikes = [s - clip.first_time for s in detect_claps_audio(clip.audio(), clip.samplerate, clip.audio_start_time,
                                                              min_rise_db=8.0, floor_dbfs=-55.0)]
    tracker = HandTracker(cfg.models.resolve("hand_landmarker"), num_hands=2)
    last = [-1]
    lines = [f"Cue sanity check: {clip.path.name} ({len(entry['gestures'])} cues). labels.yaml is NOT modified.",
             "delay = seconds after the cue beep. Label window: cue +- tolerance. 'spikes' = sharp sounds within cue-tolerance .. cue+2 s.", "",
             f"{'#':>2} {'gesture':<10} {'cue':>7}  {'first sound':>11} {'sounds':>6}  {'hand frames':>11}  {'pose / hands':>12}  verdict",
             "   (hand frames = frames with a hand tracked / with two hands, in the 3 s scanned around the cue; pose = when the thumbs-up pose starts, relative to the cue)"]
    flagged, counts = [], {"OK": 0, "LATE": 0, "CHECK": 0}
    try:
        for n, g in enumerate(entry["gestures"], 1):
            cue, tol = g["t"], g["tolerance_s"]
            t_abs = clip.first_time + cue
            frames = scan(clip, tracker, t_abs - PRE_S, t_abs + POST_S, last)
            with_hand = [t - t_abs for t, k, _ in frames if k >= 1]
            two = [t - t_abs for t, k, _ in frames if k >= 2]
            pose = [t - t_abs for t, _, up in frames if up]
            near = [s - cue for s in spikes if -tol <= s - cue <= 2.0]
            first = near[0] if near else None
            kind = g["gesture"]
            notes = []
            if kind in ("snap", "clap"):
                if first is None:
                    verdict = "CHECK"; notes.append("no sound spike near the cue")
                elif first > tol:
                    verdict = "LATE"; notes.append(f"sound {first:+.2f} s after the cue, beyond the +{tol:g} s window")
                else:
                    verdict = "OK"
                if kind == "clap" and not two:
                    verdict = "CHECK"; notes.append("two hands never tracked together near the cue")
                if kind == "snap" and not with_hand:
                    verdict = "CHECK"; notes.append("no hand tracked near the cue")
                if len(near) > 2 and kind == "snap":
                    notes.append(f"{len(near)} sounds in the window: repeated snaps or speech")
                if kind == "clap" and len(near) > 1:
                    notes.append(f"{len(near)} sounds in the window")
            else:
                eps = episodes(pose)           # runs of thumbs-up-like frames, as (start, end) relative to the cue
                hit = [e for e in eps if e[1] >= -tol and e[0] <= tol]   # one that overlaps the label window
                if not eps:
                    verdict = "CHECK"; notes.append("no thumbs-up pose seen (rough landmark test)" + ("" if with_hand else "; no hand tracked at all"))
                elif hit:
                    verdict = "OK"
                    if hit[0][0] <= -PRE_S + 0.05:
                        notes.append("pose already up when the scan began (anticipated, or the rough test also fires on a resting hand)")
                else:
                    verdict = "LATE"; notes.append(f"pose starts {eps[0][0]:+.2f} s after the cue, beyond the +{tol:g} s window")
            counts[verdict] += 1
            col = {"snap": "sound", "clap": "sound"}.get(kind)
            mid = f"{first:+.2f} s" if first is not None else "none"
            hs = f"{len(with_hand)}f/{len(two)}f2"
            if kind == "thumbs_up":
                eps = episodes(pose)
                ps = ("pose " + (f"{eps[0][0]:+.2f}" if eps[0][0] > -PRE_S + 0.05 else "<-0.8") + " s") if eps else "no pose"
            else:
                ps = "2 hands yes" if two else "2 hands no"
            lines.append(f"{n:>2} {kind:<10} {cue:>7.2f}  {mid if col else '-':>11} {len(near) if col else '-':>6}  {hs:>11}  {ps:>12}  {verdict}"
                         + (f"  <- {'; '.join(notes)}" if notes else ""))
            if verdict != "OK":
                flagged.append((n, kind, cue, verdict, "; ".join(notes)))
        # the talk segments: anything gesture-like that could be mistaken for a cue
        for seg in entry.get("no_gesture_segments", []):
            s0, s1 = seg["start"], seg["end"]
            sp = [s for s in spikes if s0 <= s <= s1]
            frames = scan(clip, tracker, clip.first_time + s0, clip.first_time + s1, last)
            ups = [t - clip.first_time for t, _, up in frames if up]
            runs = sum(1 for a, b in zip(ups, ups[1:]) if b - a > 1.0) + (1 if ups else 0)
            hands_frac = np.mean([k >= 1 for _, k, _ in frames]) if frames else 0.0
            lines += ["", f"talk segment {s0:.0f}-{s1:.0f} s (nothing should fire): hands visible in {100 * hands_frac:.0f}% of frames; "
                          f"{len(sp)} sharp sound spikes (speech and desk noise included); thumbs-up-like pose in {len(ups)} frames "
                          f"({runs} separate episodes" + (f", first at {ups[0]:.1f} s" if ups else "") + ")"]
    finally:
        tracker.close()
    lines += ["", f"SUMMARY: {counts['OK']} OK, {counts['LATE']} late (gesture happened, after the label window), "
                  f"{counts['CHECK']} to check (nothing gesture-like found)"]
    if flagged:
        lines.append("cues to review: " + ", ".join(f"#{n} {k} @{c:.2f}s ({v})" for n, k, c, v, _ in flagged))
    text = "\n".join(lines)
    print(text)
    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        Path(args.report).write_text(text + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
