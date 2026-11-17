"""Replay tests against the recorded fixture clips. The clips are local-only (gitignored), so on a machine or a CI
run without them these tests SKIP; only labels.yaml is committed."""
from pathlib import Path

import numpy as np
import pytest

from incanto.guided import LABELS_PATH, load_labels
from incanto.io.replay import Clip, ReplaySource

ROOT = Path(__file__).resolve().parent.parent
SEARCH = (ROOT / "recordings", ROOT / "tests" / "fixtures")


def find_clip(name: str) -> Path | None:
    for base in SEARCH:
        if (base / name / "meta.json").exists():
            return base / name
    return None


def labelled_clips():
    return sorted(load_labels(LABELS_PATH)["clips"].items())


def test_labels_file_is_well_formed_if_present():
    if not LABELS_PATH.exists():
        pytest.skip("tests/fixtures/labels.yaml not recorded yet (python -m incanto --record-guided plans/fixture-plan.yaml)")
    for name, entry in labelled_clips():
        for g in entry["gestures"]:
            ev = g.get("event_time", g.get("t"))                 # "t" is the pre-retiming format
            assert g["gesture"] in ("snap", "clap", "thumbs_up") and ev >= 0 and g["tolerance_s"] > 0
            if "cue_time" in g:
                assert g["event_time"] >= 0 and "method" in g
        for d in entry.get("dropped", []) + entry.get("unresolved", []):
            assert d["gesture"] in ("snap", "clap", "thumbs_up") and d["reason"]
        for seg in entry["no_gesture_segments"]:
            assert seg["end"] > seg["start"] >= 0


def test_fixture_clips_replay_in_order_with_labels_inside_the_clip():
    clips = labelled_clips()
    if not clips:
        pytest.skip("no labelled clips (labels.yaml has none)")
    checked = 0
    for name, entry in clips:
        path = find_clip(name)
        if path is None:
            continue                                         # clip is local-only and not on this machine
        clip = Clip.load(path)
        assert (np.diff(clip.frame_times) > 0).all()
        assert all(0 <= g.get("event_time", g.get("t")) <= clip.duration for g in entry["gestures"])
        src = ReplaySource(clip, realtime=False)
        first = src.read()
        assert first.shape == (clip.height, clip.width, 3)
        src.release()
        checked += 1
    if not checked:
        pytest.skip("the labelled clips are not on this machine (recordings/ and tests/fixtures/ clips are local-only)")


def test_replay_of_a_missing_clip_is_a_clean_error_not_a_crash():
    from incanto.main import main
    assert main(["--replay", str(ROOT / "recordings" / "definitely-not-here"), "--no-preview"]) == 1


def test_cli_rejects_replay_only_options_without_replay():
    from incanto.main import main
    for argv in (["--render-out", "x.avi"], ["--fast"], ["--replay", "a", "--record-guided", "b"]):
        with pytest.raises(SystemExit):
            main(argv)
