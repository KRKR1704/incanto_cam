"""Guided recording: plan schedule, prompts and beeps, labels.yaml writer."""
import pytest
import yaml

from incanto.guided import (COUNTDOWN_S, MIN_GAP_S, GuidedSession, PlanError, load_labels, load_plan, parse_plan,
                            write_labels)

PLAN = {"name": "t", "seed": 1, "lead_in_s": 6, "tolerance_s": 0.6, "segments": [
    {"gesture": "snap", "count": 3, "gap_s": [4.0, 5.0], "hint": "snap by your face"},
    {"rest_s": 5},
    {"gesture": "clap", "count": 2, "gap_s": 4.0},
    {"talk_s": 20, "text": "Talk"}]}


def test_plan_schedule_counts_gaps_and_segments():
    plan = parse_plan(PLAN)
    assert plan.counts() == {"snap": 3, "clap": 2}
    snaps = [c.t for c in plan.cues if c.gesture == "snap"]
    assert snaps[0] == 6.0 and all(4.0 <= b - a <= 5.0 for a, b in zip(snaps, snaps[1:]))
    assert [(c.index, c.of) for c in plan.cues if c.gesture == "clap"] == [(1, 2), (2, 2)]
    assert plan.rests[0].start > snaps[-1] and plan.rests[0].end - plan.rests[0].start == pytest.approx(5)
    assert plan.cues[-1].t < plan.rests[0].end + 10 and plan.talks[0].start >= plan.rests[0].end
    assert plan.talks[0].end - plan.talks[0].start == pytest.approx(20) and plan.duration_s > plan.talks[0].end


def test_plan_is_deterministic_for_a_seed_and_changes_with_it():
    a, b = parse_plan(PLAN), parse_plan(PLAN)
    assert [c.t for c in a.cues] == [c.t for c in b.cues]
    assert [c.t for c in parse_plan({**PLAN, "seed": 2}).cues] != [c.t for c in a.cues]


@pytest.mark.parametrize("bad", [
    {}, {"segments": []}, {"segments": [{"gesture": "wave", "count": 1}]}, {"segments": [{"gesture": "snap", "gap_s": 1.0}]},
    {"segments": [{"count": 3}]}, {"lead_in_s": 1, "segments": [{"gesture": "snap"}]}, {"segments": ["snap"]},
    {"segments": [{"gesture": "snap", "count": 0}]}])
def test_bad_plans_are_rejected(bad):
    with pytest.raises(PlanError):
        parse_plan(bad)


def test_shipped_fixture_plan_is_10_10_10_plus_two_minutes_of_talking():
    plan = load_plan("plans/fixture-plan.yaml")
    assert plan.counts() == {"snap": 10, "clap": 10, "thumbs_up": 10}
    assert plan.tolerance_s == 0.6
    assert len(plan.talks) == 1 and plan.talks[0].end - plan.talks[0].start == pytest.approx(120)
    gaps = [b.t - a.t for a, b in zip(plan.cues, plan.cues[1:]) if a.gesture == b.gesture]
    assert min(gaps) >= MIN_GAP_S
    assert 240 < plan.duration_s < 360                      # roughly five minutes


def run(plan, step=0.05):
    s = GuidedSession(plan, now=1000.0)
    events, prompts = [], []
    t = 1000.0
    while not s.finished and t < 1000.0 + plan.duration_s + 5:
        for e in s.update(t):
            events.append((round(t - 1000.0, 2), e))
        prompts.append((round(t - 1000.0, 2), s.prompt))
        t += step
    return s, events, prompts


def test_session_beeps_ticks_then_go_at_each_cue_and_records_times():
    plan = parse_plan(PLAN)
    s, events, _ = run(plan)
    gos = [t for t, e in events if e == "go"]
    assert len(gos) == 5 == len(s.fired)
    for g, cue in zip(gos, plan.cues):
        assert g == pytest.approx(cue.t, abs=0.06)
        ticks = [t for t, e in events if e == "tick" and cue.t - 2.1 <= t <= cue.t]
        assert len(ticks) == 2 and ticks[0] == pytest.approx(cue.t - 2, abs=0.06) and ticks[1] == pytest.approx(cue.t - 1, abs=0.06)
    assert [g for g, _ in s.fired] == [c.gesture for c in plan.cues]
    assert [e for _, e in events].count("ok") == 1 and s.finished
    assert all(abs((ta - 1000.0) - c.t) < 0.06 for (_, ta), c in zip(s.fired, plan.cues))


def test_session_prompts_say_what_to_do():
    plan = parse_plan(PLAN)
    _, _, prompts = run(plan)
    def at(t):  # the prompt shown at the frame nearest to t
        return min(prompts, key=lambda p: abs(p[0] - t))[1]
    first = plan.cues[0].t
    assert "Next: SNAP (1/3)" in at(first - 4.0).text
    assert at(first - 1.5).text == "SNAP in 2" and not at(first - 1.5).go
    assert at(first + 0.5).text == "SNAP NOW" and at(first + 0.5).go
    assert at(plan.talks[0].start + 3).text == "Talk" and "left" in at(plan.talks[0].start + 3).sub
    assert at(plan.rests[0].start + 1).text == "Rest"


def test_labels_are_relative_to_the_first_video_frame_with_the_tolerance():
    plan = parse_plan(PLAN)
    s, _, _ = run(plan)
    gestures, talks = s.label_entries(clip_start_time=999.5)             # the clip began 0.5 s before the plan
    assert gestures[0] == {"gesture": "snap", "t": pytest.approx(plan.cues[0].t + 0.5, abs=0.06), "tolerance_s": 0.6}
    assert len(gestures) == 5
    assert talks == [{"start": pytest.approx(plan.talks[0].start + 0.5, abs=1e-3),
                      "end": pytest.approx(plan.talks[0].end + 0.5, abs=1e-3)}]


def test_labels_writer_creates_merges_replaces_and_keeps_the_header(tmp_path):
    path = tmp_path / "fixtures" / "labels.yaml"
    assert load_labels(path) == {"clips": {}}
    g1 = [{"gesture": "snap", "t": 8.1, "tolerance_s": 0.6}]
    write_labels(path, "guided-a", g1, [{"start": 100.0, "end": 220.0}], {"plan": "p"})
    write_labels(path, "guided-b", [{"gesture": "clap", "t": 9.0, "tolerance_s": 0.6}], [])
    text = path.read_text()
    assert text.startswith("# Gesture labels") and "ONLY file under tests/fixtures/" in text
    data = yaml.safe_load(text)
    assert set(data["clips"]) == {"guided-a", "guided-b"}
    assert data["clips"]["guided-a"]["gestures"] == g1 and data["clips"]["guided-a"]["plan"] == "p"
    assert data["clips"]["guided-a"]["no_gesture_segments"] == [{"start": 100.0, "end": 220.0}]
    write_labels(path, "guided-a", [], [])                                  # re-recording replaces only that clip
    data = yaml.safe_load(path.read_text())
    assert data["clips"]["guided-a"]["gestures"] == [] and data["clips"]["guided-b"]["gestures"][0]["gesture"] == "clap"


def test_missing_plan_file_and_bad_yaml_are_plan_errors(tmp_path):
    with pytest.raises(PlanError, match="not found"):
        load_plan(tmp_path / "none.yaml")
    (tmp_path / "bad.yaml").write_text("segments: [unclosed")
    with pytest.raises(PlanError):
        load_plan(tmp_path / "bad.yaml")
