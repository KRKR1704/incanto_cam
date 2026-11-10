"""Plate health check, background refresh (never inside the near-me zone), capture audio cues."""
import dataclasses

import cv2
import numpy as np
import pytest

from incanto.config import PlateConfig, VanishConfig
from incanto.io.sounds import CUES, SoundCues
from incanto.plate.clean_plate import PlateCapture
from incanto.plate.health import HealthLimits, measure_plate_health
from incanto.render.vanish_mask import REFRESH_EVERY, VanishMask

W, H = 640, 360
LIMITS = HealthLimits()


def make_plate():
    y, x = np.mgrid[0:H, 0:W]
    base = 110 + 40 * np.sin(x / 70.0) * np.cos(y / 50.0) + (x / W) * 30
    return np.stack([base, base * 0.9 + 10, base * 1.1 - 10], axis=2).clip(0, 255).astype(np.uint8)


def scaled(img, gain):
    return np.clip(img.astype(np.float32) * gain, 0, 255).astype(np.uint8)


def person_seg(x0=260, x1=380, y0=120, y1=300):
    seg = np.zeros((H, W), np.float32)
    seg[y0:y1, x0:x1] = 1.0
    return seg


# -- health check ------------------------------------------------------------------------------

def test_defaults_match_the_agreed_limits():
    p = PlateConfig()
    assert (p.health_max_raw_diff, p.health_gain_min, p.health_gain_max, p.health_max_cell_gain) == (20.0, 0.8, 1.25, 1.5)
    assert (LIMITS.max_raw_diff, LIMITS.gain_min, LIMITS.gain_max, LIMITS.max_cell_gain) == (20.0, 0.8, 1.25, 1.5)
    assert p.countdown_s == 5.0 and p.sounds is True
    assert VanishConfig().plate_refresh_alpha == pytest.approx(0.02)


def test_matching_plate_is_healthy():
    plate = make_plate()
    h = measure_plate_health(scaled(plate, 1.05), plate, None, LIMITS)
    assert h.ok and h.reasons == ()
    assert h.raw_diff == pytest.approx(0.05 * plate.mean(), abs=1.5)
    assert all(g == pytest.approx(1.05, abs=0.02) for g in h.gain)
    assert h.max_cell_gain == pytest.approx(1.05, abs=0.03)
    assert h.background == 1.0


def test_stale_plate_like_the_live_test_fails():
    """Yesterday evening's plate against today's room: raw diff about +50, gain far above 1.25."""
    live = make_plate()
    plate = scaled(live, 0.55)
    h = measure_plate_health(live, plate, None, LIMITS)
    assert not h.ok
    assert h.raw_diff > 20
    assert any("raw diff" in r for r in h.reasons) and any("gain" in r for r in h.reasons)
    assert "FAILED" in h.describe()


def test_raw_diff_limit_alone():
    plate = make_plate()
    darker = np.clip(plate.astype(np.int16) - 23, 0, 255).astype(np.uint8)  # offset, gain stays near 0.8+
    h = measure_plate_health(darker, plate, None, HealthLimits(gain_min=0.5))
    assert not h.ok and len(h.reasons) == 1 and "raw diff -" in h.reasons[0]
    assert measure_plate_health(darker, plate, None, HealthLimits(max_raw_diff=30, gain_min=0.5)).ok


@pytest.mark.parametrize("gain, ok", [(0.75, False), (0.85, True), (1.2, True), (1.3, False)])
def test_global_gain_limits(gain, ok):
    plate = make_plate()
    h = measure_plate_health(scaled(plate, gain), plate, None, HealthLimits(max_raw_diff=255))
    assert h.ok is ok
    if not ok:
        assert "gain" in h.reasons[0]


def test_one_colour_channel_out_of_range_fails():
    plate = make_plate()
    live = plate.copy()
    live[..., 0] = scaled(plate[..., 0], 0.7)  # colour cast: blue drops, brightness barely moves
    h = measure_plate_health(live, plate, None, LIMITS)
    assert not h.ok and h.gain[0] < 0.8 < h.gain[1]


def test_single_bad_cell_fails_even_when_global_numbers_pass():
    plate = make_plate()
    live = plate.copy()
    live[0:60, 0:80] = scaled(plate[0:60, 0:80], 1.9)  # one grid cell (8x6 grid: 80x60 px) lit very differently
    h = measure_plate_health(live, plate, None, LIMITS)
    assert abs(h.raw_diff) < 20 and all(0.8 < g < 1.25 for g in h.gain)
    assert h.max_cell_gain > 1.5
    assert not h.ok and "cell gain" in h.reasons[0]


def test_person_is_excluded_from_the_measurement():
    plate = make_plate()
    live = plate.copy()
    live[120:300, 260:380] = (250, 250, 250)  # the person: nothing like the plate
    assert not measure_plate_health(live, plate, None, LIMITS).ok
    h = measure_plate_health(live, plate, person_seg(), LIMITS, margin_px=15)
    assert h.ok and h.background < 1.0


def test_dark_cells_are_not_judged():
    plate = np.full((H, W, 3), 8, np.uint8)
    live = np.full((H, W, 3), 14, np.uint8)  # ratio 1.75, but on near-black pixels it means nothing
    h = measure_plate_health(live, plate, None, HealthLimits(gain_max=3.0))
    assert h.max_cell_gain == 1.0 and h.ok


def test_no_verdict_without_background():
    plate = make_plate()
    assert measure_plate_health(plate, plate, np.ones((H, W), np.float32), LIMITS) is None


def test_vanish_mask_check_health_works_before_vanish_is_on():
    plate = make_plate()
    v = VanishMask(VanishConfig(), (W, H), (W, H))
    assert v.check_health(plate, person_seg(), [], LIMITS) is None  # no plate yet
    v.set_plate(plate)
    live = scaled(plate, 1.1)
    live[120:300, 260:380] = (30, 220, 60)
    assert v.check_health(live, person_seg(), [], LIMITS).ok
    stale = VanishMask(VanishConfig(), (W, H), (W, H))
    stale.set_plate(scaled(plate, 0.55))
    assert not stale.check_health(live, person_seg(), [], LIMITS).ok


# -- background refresh ------------------------------------------------------------------------

def refresh_run(live, seg, frames=40, mode="full", **cfg_kw):
    """Run compute() + refresh_plate() for `frames` frames; returns (mask object, original plate, last result)."""
    plate = make_plate()
    v = VanishMask(dataclasses.replace(VanishConfig(), mask_mode=mode, **cfg_kw), (W, H), (W, H))
    v.set_plate(plate)
    r = None
    for _ in range(frames):
        r = v.compute(live, seg, [])
        v.refresh_plate(live)
    return v, plate, r


def drifted_scene(gain=1.08):
    """The room got a little brighter (well under the plate-difference threshold) and I am in it."""
    live = scaled(make_plate(), gain)
    live[120:300, 260:380] = (30, 220, 60)
    return live


def zone_px(v, seg):
    """The near-me zone for `seg`, recomputed independently of the refresh code path."""
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * round(v.cfg.dilate_px * 0.5) + 1,) * 2)
    grown = cv2.dilate((seg > 0).astype(np.uint8), k)
    r = round(v.cfg.near_me_px * 0.5)  # config px are at 1280 wide; these tests run at 640
    return cv2.dilate(grown, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))) > 0


@pytest.mark.parametrize("mode", ["full", "seg_hands", "seg_only"])
def test_refresh_never_touches_the_near_me_zone(mode):
    live, seg = drifted_scene(), person_seg()
    v, plate, r = refresh_run(live, seg, mode=mode)
    assert v.stats.refresh_n == 40 // REFRESH_EVERY
    changed = (v.plate != plate).any(axis=2)
    assert changed.any()
    zone = zone_px(v, seg)
    assert not changed[zone].any()                       # inside the zone the plate is exactly as captured
    assert not changed[120:300, 260:380].any()           # ...and of course under me
    if mode == "full":
        assert not changed[r.zone > 0].any()             # the zone the mask itself used
    # the guard: nothing within plate_refresh_guard_px beyond the zone either
    g = round(v.cfg.plate_refresh_guard_px * 0.5)
    guard = cv2.dilate(zone.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * g + 1, 2 * g + 1))) > 0
    assert not changed[guard].any()
    # the plate handed to the caller is not modified in place
    assert (plate == make_plate()).all()


def test_refresh_moves_the_far_background_toward_live_slowly():
    live, seg = drifted_scene(), person_seg()
    v, plate, _ = refresh_run(live, seg, frames=REFRESH_EVERY)  # exactly one pass = 4 frames' worth
    far = (slice(20, 80), slice(20, 120))
    step = (v.plate[far].astype(float) - plate[far]).mean() / (live[far].astype(float) - plate[far]).mean()
    assert step == pytest.approx(1 - 0.98 ** REFRESH_EVERY, abs=0.03)   # ~0.02 per frame
    v, plate, _ = refresh_run(live, seg, frames=400)
    assert np.abs(v.plate[far].astype(int) - live[far]).max() <= 1      # converged after a few hundred frames
    assert v.stats.refresh_coverage / v.stats.refresh_n > 0.3           # coverage is logged


def test_refresh_follows_the_zone_when_i_move():
    live_a, seg_a = drifted_scene(), person_seg()
    plate = make_plate()
    v = VanishMask(VanishConfig(), (W, H), (W, H))
    v.set_plate(plate)
    seg_b = person_seg(x0=60, x1=160)
    live_b = scaled(plate, 1.08)
    live_b[120:300, 60:160] = (30, 220, 60)
    before = None
    for i in range(80):
        live, seg = (live_a, seg_a) if i < 40 else (live_b, seg_b)
        if i == 40:
            before = v.plate.copy()
        v.compute(live, seg, [])
        v.refresh_plate(live)
    new_zone = zone_px(v, seg_b)
    assert (v.plate[new_zone] == before[new_zone]).all()  # frozen from the moment I stand there


def test_refresh_skips_things_that_differ_from_the_plate():
    """Someone the segmenter does not see (far from the zone) must not be burned into the plate."""
    live, seg = drifted_scene(), person_seg()
    live[40:140, 520:600] = (20, 20, 230)  # second person / pet / moved chair, not segmented
    v, plate, _ = refresh_run(live, seg)
    assert (v.plate[40:140, 520:600] == plate[40:140, 520:600]).all()
    assert (v.plate[20:80, 20:120] != plate[20:80, 20:120]).any()  # ordinary background still refreshes


def test_refresh_pauses_when_the_segmenter_sees_nobody():
    live = drifted_scene()
    v, plate, _ = refresh_run(live, np.zeros((H, W), np.float32))
    assert (v.plate == plate).all()
    assert v.stats.refresh_n == 0 and v.stats.refresh_paused == 40 // REFRESH_EVERY


def test_refresh_off_with_alpha_zero_and_reset_by_a_new_plate():
    live, seg = drifted_scene(), person_seg()
    v, plate, _ = refresh_run(live, seg, plate_refresh_alpha=0.0)
    assert v.plate is plate and v.stats.refresh_n == 0
    v, plate, _ = refresh_run(live, seg)
    fresh = make_plate()
    v.set_plate(fresh)
    assert v.plate is fresh
    assert v.refresh_plate(live) is False  # nothing to refresh from until compute() has run again


# -- audio cues --------------------------------------------------------------------------------

def capture_events(mask_value, countdown_s=5.0):
    cap = PlateCapture(frames=5, countdown_s=countdown_s, settle_s=1.0, max_coverage=0.02)
    cap.start(0.0)
    events, t = cap.pop_events(), 0.0
    mask = np.full((36, 64), mask_value, np.float32)
    frame = np.full((36, 64, 3), 80, np.uint8)
    while t < 12 and cap.state not in (cap.DONE, cap.ABORTED):
        t += 1 / 30
        cap.update(frame, mask, t)
        events += cap.pop_events()
    return cap, events


def test_one_tick_per_countdown_second_then_capturing():
    cap, events = capture_events(0.0)
    assert cap.state == cap.DONE
    assert events == ["tick"] * 5 + ["capturing"]
    assert cap.pop_events() == []
    _, events = capture_events(0.0, countdown_s=3.0)
    assert events == ["tick"] * 3 + ["capturing"]


def test_abort_emits_the_error_cue():
    cap, events = capture_events(1.0)
    assert cap.state == cap.ABORTED
    assert events == ["tick"] * 5 + ["capturing", "error"]


def test_cues_are_distinct_and_muting_is_silent():
    assert {"tick", "capturing", "ok", "error"} <= set(CUES)
    assert len({CUES[c] for c in ("tick", "capturing", "ok", "error")}) == 4
    played = []
    muted = SoundCues(False, beep=lambda f, ms: played.append((f, ms)))
    muted.play("error")
    assert not muted.enabled and played == []
    with pytest.raises(KeyError):
        muted.play("no-such-cue")


def test_cues_play_off_the_main_thread():
    import threading
    done, played = threading.Event(), []

    def beep(freq, ms):
        played.append((freq, ms, threading.current_thread().name))
        if len(played) == len(CUES["ok"]):
            done.set()

    SoundCues(True, beep=beep).play("ok")
    assert done.wait(2.0)
    assert [(f, ms) for f, ms, _ in played] == list(CUES["ok"])
    assert all(name == "sound-cues" for _, _, name in played)
