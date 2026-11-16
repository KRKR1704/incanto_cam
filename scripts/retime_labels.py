"""Retime the labels of a guided clip to the gestures that actually happened.

    .venv\\Scripts\\python scripts/retime_labels.py recordings/<clip> [--table out.txt] [--write]

People react to the cue beep 0.5-1.3 s late, so the cue time is a poor ground truth. For every labelled cue this finds the
gesture itself in the window cue_time .. cue_time + 1.5 s and records it as `event_time`; `cue_time` keeps the original.
Phase 5 evaluates against event_time.

* snap / clap: the sound onset of the gesture. Sounds within 150 ms of any beep the guide played (the two soft ticks at
  cue-2 s and cue-1 s, and the cue beep) are ignored. A SNAP also needs a raised hand near the face/shoulder when it happens
  (hand tracking; the hand's highest point within 1.4 face-heights below the top of the head and within 1.5 head-widths of
  the head, the head taken from the person mask). A hand lost at the snap itself is judged from the nearest tracked frame.
* thumbs_up: the onset of the thumbs-up pose (a rough landmark test, held at least 0.4 s) found in cue - 2 s .. cue + 1.5 s.

Anything ambiguous is flagged (several candidates, hand lost, pose that began before the cue, ...). A cue with nothing found is
listed as unresolved and NOT written to the gestures list. The printed table is the review step: without --write nothing is
changed. Default tolerance after retiming: +-0.3 s.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from incanto.config import load_config  # noqa: E402
from incanto.guided import LABELS_HEADER, LABELS_PATH, load_labels  # noqa: E402
from incanto.io.replay import Clip, ClipError  # noqa: E402
from incanto.perception.hands import HandTracker  # noqa: E402
from incanto.perception.segmenter import Segmenter  # noqa: E402
from incanto.utils.avsync import detect_claps_audio, highpass  # noqa: E402
from incanto.utils.imaging import downscale  # noqa: E402

WINDOW_S = 1.5
BEEP_MARGIN_S = 0.15
TICK_LEN_S, GO_LEN_S = 0.09, 0.22          # incanto/io/sounds.py: "tick" 90 ms, "go" 220 ms
NEW_TOLERANCE_S = 0.3
MIN_POSE_S = 0.4
HAND_LOST_S = 0.08                          # nearest tracked frame further than this from the sound = "hand lost at the snap"
THUMB_TIPS, THUMB_PIPS = (8, 12, 16, 20), (6, 10, 14, 18)
DROPPED = {21: "no thumbs-up pose in cue .. cue + 1.5 s; a pose only appears 2.5 s after the cue (121.9 s) and runs into cue 22: looks flubbed"}   # cue number -> reason


def beep_windows(cue: float) -> list[tuple[float, float]]:
    """(start, end) of the periods around each guide beep in which sounds are ignored."""
    return [(cue - 2 - BEEP_MARGIN_S, cue - 2 + TICK_LEN_S + BEEP_MARGIN_S),
            (cue - 1 - BEEP_MARGIN_S, cue - 1 + TICK_LEN_S + BEEP_MARGIN_S),
            (cue - BEEP_MARGIN_S, cue + GO_LEN_S + BEEP_MARGIN_S)]


def thumbs_up_like(hand) -> bool:
    lm = hand.landmarks
    return bool(lm[4, 1] < lm[3, 1] < lm[2, 1] and lm[4, 1] < lm[THUMB_TIPS, 1].min() - 0.02
                and all(lm[t, 1] > lm[p, 1] for t, p in zip(THUMB_TIPS, THUMB_PIPS)))


def head_box(mask: np.ndarray):
    """(head top y, head centre x, head width, face height) in normalised units, from the person mask; None if nobody."""
    m = mask > 0.5
    rows = np.flatnonzero(m.any(axis=1))
    if len(rows) == 0:
        return None
    h, w = m.shape
    top = rows[0]
    band = slice(top + max(1, int(0.03 * h)), top + max(2, int(0.14 * h)))
    widths = m[band].sum(axis=1)
    cols = np.flatnonzero(m[band].any(axis=0))
    if len(cols) == 0 or np.median(widths) == 0:
        return None
    head_w_px = float(np.median(widths))
    return top / h, float(np.mean(cols) / w), head_w_px / w, 1.12 * head_w_px / h


def near_face(hand, box) -> bool:
    top, cx, hw, fh = box
    return bool(hand.landmarks[:, 1].min() <= top + 1.4 * fh and abs(float(hand.landmarks[:, 0].mean()) - cx) <= 1.5 * hw)


def runs(times: list[float], gap: float = 0.25) -> list[tuple[float, float]]:
    out: list[tuple[float, float]] = []
    for t in times:
        if out and t - out[-1][1] <= gap:
            out[-1] = (out[-1][0], t)
        else:
            out.append((t, t))
    return out


def level_dbfs(hp: np.ndarray, rate: int, t_audio: float) -> float:
    i, n = int(t_audio * rate), int(0.005 * rate)
    seg = hp[max(0, i - n):i + n]
    return 20 * np.log10(max(float(np.abs(seg).max()), 1e-6)) if len(seg) else -120.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("clip")
    ap.add_argument("--table", help="also write the table to this .txt file")
    ap.add_argument("--write", action="store_true", help="write the retimed entry to tests/fixtures/labels.yaml")
    ap.add_argument("--labels", default=str(LABELS_PATH))
    args = ap.parse_args()
    cfg = load_config()
    try:
        clip = Clip.load(args.clip)
    except ClipError as exc:
        print(exc, file=sys.stderr)
        return 2
    labels = load_labels(Path(args.labels))
    entry = labels["clips"].get(clip.path.name)
    if entry is None:
        print(f"{clip.path.name} is not in {args.labels}", file=sys.stderr)
        return 2
    rate, a0 = clip.samplerate, clip.audio_start_time
    audio_off = a0 - clip.first_time                     # audio sample 0 in clip time
    onsets = [s - clip.first_time for s in detect_claps_audio(clip.audio(), rate, a0, min_rise_db=8.0, floor_dbfs=-55.0)]
    hp = highpass(clip.audio().astype(np.float64), rate)
    tracker = HandTracker(cfg.models.resolve("hand_landmarker"), num_hands=2)
    segmenter = Segmenter(cfg.models.resolve("selfie_segmenter"), (clip.width, clip.height))
    last = [-1]

    def frames_in(t0: float, t1: float):
        """[(time rel. to clip start, hands, head box)] for every frame in [t0, t1] (the segmenter warms up on 8 frames before)."""
        out = []
        i0 = clip.index_at(clip.first_time + t0)
        for i, t, frame in clip.frames(max(0, i0 - 8), clip.index_at(clip.first_time + t1) + 1):
            ms = max(int(round((t - clip.first_time) * 1000)), last[0] + 1)
            last[0] = ms
            small = downscale(frame, 640)
            hands = tracker.detect(small, ms)
            segmenter.segment(small, ms)
            if i >= i0:
                out.append((t - clip.first_time, hands, head_box(segmenter.small)))
        return out

    rows, gestures, unresolved, dropped = [], [], [], []
    try:
        for n, g in enumerate(entry["gestures"], 1):
            kind = g["gesture"]
            cue = g["cue_time"] if "cue_time" in g else g["t"]
            if n in DROPPED:
                dropped.append({"cue_number": n, "gesture": kind, "cue_time": cue, "reason": DROPPED[n]})
                rows.append((n, kind, cue, None, None, "dropped", DROPPED[n]))
                continue
            flags, alts, method, event = [], [], "", None
            if kind in ("snap", "clap"):
                excl = beep_windows(cue)
                allc = [s for s in onsets if cue <= s <= cue + WINDOW_S]
                cands = [s for s in allc if not any(a <= s <= b for a, b in excl)]
                if len(allc) != len(cands):
                    flags.append(f"{len(allc) - len(cands)} sound(s) inside a beep window ignored")
                if kind == "snap":
                    fr = frames_in(cue, cue + WINDOW_S + 0.3)
                    valid = []
                    for s in cands:
                        near = sorted((abs(t - s), hands, box) for t, hands, box in fr if hands and box)[:1]
                        if not near:
                            flags.append(f"no hand tracked within the window at the sound +{s - cue:.2f}")
                            continue
                        dt, hands, box = near[0]
                        ok = any(near_face(h, box) for h in hands)
                        if ok:
                            valid.append((s, dt))
                    cands_used = [s for s, _ in valid]
                    method = "sound onset + raised hand near face"
                    if valid:
                        event = valid[0][0]
                        if valid[0][1] > HAND_LOST_S:
                            flags.append(f"hand lost at the snap (judged from a frame {valid[0][1] * 1000:.0f} ms away)")
                    rejected = [s for s in cands if s not in cands_used]
                    if rejected and valid:
                        flags.append(f"{len(rejected)} sound(s) without a raised hand ignored")
                else:
                    cands_used = cands
                    method = "sound onset"
                    event = cands[0] if cands else None
                alts = [round(float(s), 3) for s in cands_used[1:]]
                if len(cands_used) > 1:
                    flags.append(f"{len(cands_used)} candidate onsets: first used, others kept as alt_event_times")
                if event is not None and event - cue < 0.25:
                    flags.append("onset within 0.25 s of the cue (anticipated?)")
                if event is not None:
                    flags.append(f"level {level_dbfs(hp, rate, event - audio_off):.0f} dBFS") if level_dbfs(hp, rate, event - audio_off) < -45 else None
                if event is None:
                    flags.append("no sound onset in the window" if not cands else "no sound with a raised hand")
            else:  # thumbs_up
                fr = frames_in(cue - 4.0, cue + WINDOW_S)
                pose = [t for t, hands, _ in fr if any(thumbs_up_like(h) for h in hands)]
                eps = [e for e in runs(pose) if e[1] - e[0] >= MIN_POSE_S]
                method = "pose onset (rough landmark test)"
                onsets_in = [e for e in eps if cue <= e[0] <= cue + WINDOW_S]
                held = [e for e in eps if e[0] < cue <= e[1] + 0.25]          # a pose that began earlier and covers the cue
                if onsets_in:
                    event = onsets_in[0][0]
                    flags.append(f"held {onsets_in[0][1] - onsets_in[0][0]:.1f} s")
                    if len(onsets_in) > 1:
                        flags.append(f"{len(onsets_in)} pose onsets in the window: first used")
                        alts = [round(float(e[0]), 3) for e in onsets_in[1:]]
                elif held:
                    first_up = held[0][0] if held[0][0] > cue - 3.9 else None
                    flags.append("pose already up at the cue: no onset inside the window (held "
                                 + (f"since {held[0][0]:.1f} s, {cue - held[0][0]:.1f} s before the cue" if first_up else "for more than 3.9 s before the cue")
                                 + f" until {held[0][1]:.1f} s)")
                else:
                    later = [e for e in eps if e[0] > cue + WINDOW_S]
                    flags.append("no thumbs-up pose in the window" + (f"; one starts at +{later[0][0] - cue:.2f} s, outside it" if later else ""))
            if event is None:
                unresolved.append({"cue_number": n, "gesture": kind, "cue_time": cue, "reason": "; ".join(flags)})
                rows.append((n, kind, cue, None, None, method, "; ".join(flags)))
            else:
                ev = round(float(event), 3)
                item = {"gesture": kind, "cue_time": float(cue), "event_time": ev, "tolerance_s": NEW_TOLERANCE_S, "method": method}
                if alts:
                    item["alt_event_times"] = alts
                gestures.append(item)
                rows.append((n, kind, cue, ev, ev - cue, method, "; ".join(flags)))
    finally:
        tracker.close()

    ambiguous = [r for r in rows if r[6] and not r[6].startswith("held") and r[5] != "dropped"]
    lines = [f"Retiming proposal for {clip.path.name} (time = seconds from the clip's first video frame; audio times are the raw audio "
             f"clock, audio.offset_ms {cfg.audio.offset_ms:g}; the A/V check puts audio ~25 ms earlier than the picture)",
             f"tolerance after retiming: +-{NEW_TOLERANCE_S:g} s | beep windows ignored: cue-2, cue-1 and cue beeps +-150 ms | labels.yaml "
             + ("WRITTEN" if args.write else "NOT modified (dry run)"), "",
             f"{'#':>2} {'gesture':<10} {'cue_time':>8} {'event_time':>10} {'delta':>7}  {'method':<38} notes"]
    for n, kind, cue, ev, d, method, notes in rows:
        lines.append(f"{n:>2} {kind:<10} {cue:>8.3f} " + (f"{ev:>10.3f} {d:>+7.2f}" if ev is not None else f"{'-':>10} {'-':>7}") + f"  {method:<38} {notes}")
    deltas = [r[4] for r in rows if r[4] is not None]
    if deltas:
        lines += ["", f"delta event - cue: median {np.median(deltas):+.2f} s, min {min(deltas):+.2f}, max {max(deltas):+.2f}  ({len(deltas)} retimed, "
                      f"{len(dropped)} dropped, {len(unresolved)} unresolved)"]
    if dropped:
        lines.append("dropped: " + "; ".join(f"#{d['cue_number']} {d['gesture']} @{d['cue_time']:.2f}: {d['reason']}" for d in dropped))
    if unresolved:
        lines.append("UNRESOLVED (left out of gestures): " + "; ".join(f"#{u['cue_number']} {u['gesture']} @{u['cue_time']:.2f}: {u['reason']}" for u in unresolved))
    lines.append(f"ambiguous or notable entries: {len(ambiguous)} of {len(rows)} (see notes)")
    text = "\n".join(lines)
    print(text)
    if args.table:
        Path(args.table).parent.mkdir(parents=True, exist_ok=True)
        Path(args.table).write_text(text + "\n", encoding="utf-8")
    if args.write:
        new = {k: v for k, v in entry.items() if k not in ("gestures",)}
        new["gestures"] = gestures
        new["tolerance_s"] = NEW_TOLERANCE_S
        new["time_base"] = "seconds from the clip's first video frame; event_time of sounds is on the raw audio clock"
        new["retimed"] = "scripts/retime_labels.py: event_time = the gesture itself (cue_time is when the guide beeped)"
        if dropped:
            new["dropped"] = dropped
        if unresolved:
            new["unresolved"] = unresolved
        labels["clips"][clip.path.name] = new
        Path(args.labels).write_text(LABELS_HEADER + yaml.safe_dump(labels, sort_keys=True, default_flow_style=None), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
