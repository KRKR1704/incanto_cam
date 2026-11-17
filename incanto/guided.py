"""Guided recording: prompts you through a plan of gestures, records the cue times, writes the labels file.

A plan (plans/*.yaml) lists segments: a gesture repeated `count` times with a random `gap_s` between cues, a rest,
or a talk segment with no cues (a negative example: nothing should fire while you talk with your hands).
Each cue gets a countdown (two soft ticks, then a higher beep and the big "SNAP NOW" at the cue time), so you can
do the gesture ON the beep. The cue time is when that beep is triggered; labels carry a tolerance window around it
(default +-0.6 s), covering beep latency and reaction time.

`GuidedSession` is pure (the caller passes the clock and plays the sounds), so it is unit-tested without a camera.
`tests/fixtures/labels.yaml` is the only fixture file that is committed; the clips are local-only.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import yaml

GESTURES = ("snap", "clap", "thumbs_up")
COUNTDOWN_S = 2.0         # ticks at cue-2 and cue-1, the beep at the cue itself
NOW_SHOWN_S = 1.2         # how long "SNAP NOW" stays up
MIN_GAP_S = 3.5           # cues are never closer than this (countdown + the gesture itself)
LABELS_PATH = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "labels.yaml"


class PlanError(ValueError):
    """The plan file is malformed."""


@dataclass(frozen=True)
class Cue:
    gesture: str
    t: float               # seconds from the start of the session
    index: int             # 1-based within its segment
    of: int
    hint: str = ""


@dataclass(frozen=True)
class Talk:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class Rest:
    start: float
    end: float
    text: str


@dataclass
class GuidedPlan:
    name: str
    lead_in_s: float
    tolerance_s: float
    cues: list[Cue] = field(default_factory=list)
    talks: list[Talk] = field(default_factory=list)
    rests: list[Rest] = field(default_factory=list)
    duration_s: float = 0.0

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for c in self.cues:
            out[c.gesture] = out.get(c.gesture, 0) + 1
        return out


def parse_plan(data: dict, name: str = "plan") -> GuidedPlan:
    """Build the schedule (all times relative to the session start) from a plan mapping. Deterministic for a seed."""
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list) or not data["segments"]:
        raise PlanError("a plan needs a non-empty 'segments' list")
    rng = random.Random(data.get("seed", 0))
    lead_in = float(data.get("lead_in_s", 6.0))
    tol = float(data.get("tolerance_s", 0.6))
    if lead_in < COUNTDOWN_S + 1 or tol <= 0:
        raise PlanError("lead_in_s must be at least 3 and tolerance_s positive")
    plan = GuidedPlan(str(data.get("name", name)), lead_in, tol)
    t = lead_in                    # time of the next cue (or the start of the next rest/talk)
    prev_cue = False
    for i, seg in enumerate(data["segments"]):
        if not isinstance(seg, dict):
            raise PlanError(f"segment {i + 1} is not a mapping")
        if "gesture" in seg:
            g = seg["gesture"]
            if g not in GESTURES:
                raise PlanError(f"segment {i + 1}: unknown gesture {g!r} (known: {', '.join(GESTURES)})")
            count = int(seg.get("count", 1))
            gap = seg.get("gap_s", [4.0, 6.0])
            lo, hi = (gap, gap) if isinstance(gap, (int, float)) else (float(gap[0]), float(gap[1]))
            if count < 1 or lo < MIN_GAP_S or hi < lo:
                raise PlanError(f"segment {i + 1}: count must be >= 1 and gap_s at least {MIN_GAP_S}")
            if prev_cue:
                t += rng.uniform(lo, hi)
            for n in range(count):
                plan.cues.append(Cue(g, round(t, 3), n + 1, count, str(seg.get("hint", ""))))
                prev_cue = True
                if n < count - 1:
                    t += rng.uniform(lo, hi)
            t += NOW_SHOWN_S + 2.0   # finish the last gesture before whatever comes next
            prev_cue = False
        elif "rest_s" in seg:
            d = float(seg["rest_s"])
            plan.rests.append(Rest(t, t + d, str(seg.get("text", "Rest"))))
            t += d
        elif "talk_s" in seg:
            d = float(seg["talk_s"])
            plan.talks.append(Talk(t, t + d, str(seg.get("text", "Talk normally and use your hands"))))
            t += d
        else:
            raise PlanError(f"segment {i + 1} needs one of: gesture, rest_s, talk_s")
    plan.duration_s = t + 2.0
    return plan


def load_plan(path: str | Path) -> GuidedPlan:
    path = Path(path)
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise PlanError(f"plan file not found: {path}") from None
    except yaml.YAMLError as exc:
        raise PlanError(f"could not parse {path}: {exc}") from exc
    return parse_plan(data, path.stem)


@dataclass(frozen=True)
class Prompt:
    text: str
    sub: str = ""
    go: bool = False


class GuidedSession:
    """Runs a plan against a clock. Call `update(now)` once per frame: it returns the sound cues due ("tick", "go",
    "ok") and keeps `prompt` (what to draw), `fired` (cue gesture, time) and `finished` up to date."""

    def __init__(self, plan: GuidedPlan, now: float):
        self.plan, self.t0 = plan, now
        self.fired: list[tuple[str, float]] = []      # (gesture, absolute time of the beep)
        self.finished = False
        self.prompt: Prompt | None = None
        self._next = 0                                 # next cue to fire
        self._ticked: set[tuple[int, int]] = set()
        self._done_sent = False

    def update(self, now: float) -> list[str]:
        t, plan, events = now - self.t0, self.plan, []
        cues = plan.cues
        # cue beeps: ticks 2 s and 1 s before, the go beep at the cue
        for i in range(self._next, len(cues)):
            c = cues[i]
            for k in (2, 1):
                if t >= c.t - k and (i, k) not in self._ticked:
                    self._ticked.add((i, k))
                    if t < c.t - k + 0.5:              # not for a countdown we were already past
                        events.append("tick")
            if i == self._next and t >= c.t:
                self.fired.append((c.gesture, now))
                events.append("go")
                self._next += 1
            if c.t - COUNTDOWN_S > t:
                break
        self.prompt = self._prompt_at(t)
        if t >= plan.duration_s and not self._done_sent:
            self._done_sent, self.finished = True, True
            events.append("ok")
        return events

    def _prompt_at(self, t: float) -> Prompt | None:
        plan = self.plan
        for talk in plan.talks:
            if talk.start <= t < talk.end:
                left = int(talk.end - t)
                return Prompt(talk.text, f"{left // 60}:{left % 60:02d} left")
        for rest in plan.rests:
            if rest.start <= t < rest.end:
                return Prompt(rest.text, f"next segment in {int(rest.end - t) + 1} s")
        cues = plan.cues
        # the cue whose "NOW" is showing, else the next one coming
        for c in cues:
            if c.t <= t < c.t + NOW_SHOWN_S:
                return Prompt(f"{c.gesture.replace('_', ' ').upper()} NOW", f"{c.index}/{c.of}", go=True)
        upcoming = next((c for c in cues if c.t > t), None)
        if upcoming is not None:
            name = upcoming.gesture.replace("_", " ").upper()
            until = upcoming.t - t
            if until <= COUNTDOWN_S:
                return Prompt(f"{name} in {int(until) + 1}", upcoming.hint)
            return Prompt(f"Next: {name} ({upcoming.index}/{upcoming.of})", upcoming.hint or "get ready")
        return Prompt("Done" if t >= plan.duration_s - 2.0 else "Relax")

    def label_entries(self, clip_start_time: float) -> tuple[list[dict], list[dict]]:
        """(gesture labels, talk segments) with times in seconds from the clip's first video frame."""
        # event_time starts as the cue time; scripts/retime_labels.py replaces it with when the gesture really happened
        gestures = [{"gesture": g, "cue_time": round(ta - clip_start_time, 3), "event_time": round(ta - clip_start_time, 3),
                     "tolerance_s": self.plan.tolerance_s, "method": "cue beep (not retimed)"}
                    for g, ta in self.fired]
        talks = [{"start": round(self.t0 + tk.start - clip_start_time, 3), "end": round(self.t0 + tk.end - clip_start_time, 3)}
                 for tk in self.plan.talks]
        return gestures, talks


# -- labels.yaml --------------------------------------------------------------------------------
LABELS_HEADER = (
    "# Gesture labels for the recorded clips (Phase 4). This is the ONLY file under tests/fixtures/ that is committed;\n"
    "# the clips themselves are local-only (recordings/ or tests/fixtures/, both gitignored). Written by\n"
    "# `python -m incanto --record-guided <plan>`.\n"
    "# Times are seconds from the clip's first video frame. cue_time = when the guide beeped; event_time = when the gesture\n"
    "# really happened (people react 0.5-1.3 s late): Phase 5 evaluation uses event_time, and a detection counts if it lands\n"
    "# within event_time +- tolerance_s. After a recording, run scripts/retime_labels.py to set event_time from the recording\n"
    "# (until then event_time = cue_time and method says so). alt_event_times = further genuine onsets in the same window.\n"
    "# dropped / unresolved list cues left out of gestures, with the reason. no_gesture_segments are talk-with-your-hands\n"
    "# stretches in which nothing may fire.\n")


def load_labels(path: Path = LABELS_PATH) -> dict:
    if not Path(path).exists():
        return {"clips": {}}
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    data.setdefault("clips", {})
    return data


def write_labels(path: Path, clip: str, gestures: list[dict], talks: list[dict], extra: dict | None = None) -> None:
    """Add (or replace) one clip's entry in labels.yaml, keeping every other clip."""
    labels = load_labels(path)
    entry = {**(extra or {}), "gestures": gestures, "no_gesture_segments": talks}
    labels["clips"][clip] = entry
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(LABELS_HEADER + yaml.safe_dump(labels, sort_keys=True, default_flow_style=None),
                          encoding="utf-8")
