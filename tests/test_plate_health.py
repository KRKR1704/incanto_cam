"""Plate health (judged after gain matching), background refresh (never inside the near-me zone), capture audio cues."""
import dataclasses

import cv2
import numpy as np
import pytest

from incanto.config import PlateConfig, VanishConfig
from incanto.io.sounds import CUES, SoundCues
from incanto.plate.clean_plate import PlateCapture
from incanto.plate.health import (HealthLimits, HealthMonitor, PlateHealth, StalenessNudge,
                                  measure_plate_health)
from incanto.perception.hands import Hand
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


# -- health check: judged after gain matching -----------------------------------------------------

def room():
    """A plate with real edges (furniture, a door frame), so structure can be compared."""
    img = make_plate().astype(np.float32)
    rng = np.random.default_rng(7)
    for _ in range(40):
        x, y = int(rng.integers(0, W - 60)), int(rng.integers(0, H - 50))
        w, h = int(rng.integers(20, 90)), int(rng.integers(15, 70))
        img[y:y + h, x:x + w] = img[y:y + h, x:x + w] * rng.uniform(0.55, 1.35) + rng.uniform(-10, 10, 3)
    return np.clip(img, 20, 235).astype(np.uint8)


def test_defaults_come_from_config():
    p = PlateConfig()
    assert (p.health_max_residual, p.health_min_structure, p.health_gain_min, p.health_gain_max,
            p.health_min_cell_background, p.health_hysteresis) == (
        LIMITS.max_residual, LIMITS.min_structure, LIMITS.gain_min, LIMITS.gain_max,
        LIMITS.min_cell_background, LIMITS.hysteresis)
    assert (p.health_gain_min, p.health_gain_max, p.health_min_cell_background, p.health_fail_s) == (0.5, 2.0, 0.5, 3.0)
    assert not hasattr(p, "health_max_raw_diff") and not hasattr(p, "health_max_cell_gain")  # no longer criteria
    assert p.countdown_s == 5.0 and p.sounds is True and p.lock_exposure is True
    assert VanishConfig().plate_refresh_alpha == pytest.approx(0.02)


def test_matching_plate_is_healthy():
    plate = room()
    h = measure_plate_health(plate, plate, None, LIMITS)
    assert h.ok and h.reasons == ()
    assert h.residual < 0.3 and h.structure > 0.99 and h.gain == pytest.approx(1.0, abs=0.01)
    assert h.cells == 48 and h.background == 1.0
    assert "healthy" in h.describe()


@pytest.mark.parametrize("gain", [0.64, 0.7, 0.86, 1.22, 1.5])
def test_exposure_swings_do_not_fail_a_good_plate(gain):
    """The live test: auto exposure moved the raw picture by -51..+30 levels (gain 0.64..1.22) while
    the plate was fine. Raw difference and global gain are no longer failure criteria."""
    plate = room()
    live = scaled(plate, gain)
    assert abs(float(live.mean()) - float(plate.mean())) > 15  # the old +/-20 raw-diff limit would trip on most
    h = measure_plate_health(live, plate, person_seg(), LIMITS, margin_px=15)
    assert h.ok, h.describe()
    assert h.residual < 1.5 and h.structure > 0.9
    assert h.gain == pytest.approx(gain, abs=0.05)


def test_gain_is_only_a_loose_sanity_cap():
    plate = np.clip(room().astype(np.float32) * 0.5 + 20, 0, 255).astype(np.uint8)  # dim plate: no clipping at 2.3x
    for gain, ok in ((0.45, False), (0.55, True), (1.9, True), (2.3, False)):
        h = measure_plate_health(scaled(plate, gain), plate, None, dataclasses.replace(LIMITS, max_residual=255))
        assert h.ok is ok, (gain, h.describe())
        if not ok:
            assert len(h.reasons) == 1 and "gain" in h.reasons[0]


def test_lighting_the_gain_map_cannot_follow_fails_on_residual():
    """Hard-edged light and shadow patches (another lamp, sun through blinds): matched cell by cell,
    the plate is still several levels off in many cells."""
    plate = room()
    rng = np.random.default_rng(3)
    field = np.ones((H, W), np.float32)
    for _ in range(14):
        x, y = int(rng.integers(0, W - 120)), int(rng.integers(0, H - 90))
        field[y:y + int(rng.integers(50, 110)), x:x + int(rng.integers(60, 140))] *= rng.choice([0.72, 1.38])
    live = np.clip(plate * field[..., None], 0, 255).astype(np.uint8)
    h = measure_plate_health(live, plate, None, LIMITS)
    assert not h.ok and h.residual > LIMITS.max_residual
    assert any("matched residual" in r for r in h.reasons)
    assert 0.5 < h.gain < 2.0 and not any("gain" in r for r in h.reasons)


@pytest.mark.parametrize("shift", [48, 120])
def test_moved_camera_fails_on_structure_whatever_the_brightness(shift):
    plate = room()
    for gain in (1.0, 0.7, 1.3):
        live = scaled(np.roll(plate, shift, axis=1), gain)
        h = measure_plate_health(live, plate, None, LIMITS)
        assert not h.ok and h.structure < LIMITS.min_structure, h.describe()
        assert any("structure" in r for r in h.reasons)


def test_different_room_fails_on_structure():
    h = measure_plate_health(room()[::-1, ::-1].copy(), room(), None, LIMITS)
    assert not h.ok and h.structure < 0.2


def test_structure_ignores_exposure_and_small_local_changes():
    plate = room()
    for gain in (0.6, 1.5):
        assert measure_plate_health(scaled(plate, gain), plate, None, LIMITS).structure > 0.95
    live = plate.copy()
    live[200:330, 420:520] = (40, 45, 50)  # a chair that was moved: one corner of the room changed
    h = measure_plate_health(live, plate, None, LIMITS)
    assert h.ok and LIMITS.min_structure < h.structure < 0.99


def test_structure_is_skipped_on_a_blank_wall():
    wall = np.full((H, W, 3), 150, np.uint8)
    h = measure_plate_health(scaled(wall, 0.8), wall, None, LIMITS)
    assert h.structure is None and h.ok and "n/a" in h.describe()


def test_person_is_excluded_from_the_measurement():
    plate = room()
    live = plate.copy()
    live[120:300, 260:380] = (250, 250, 250)  # the person: nothing like the plate
    h = measure_plate_health(live, plate, person_seg(), LIMITS, margin_px=15)
    assert h.ok and h.background < 1.0 and h.residual < 1.0 and h.structure > 0.95


# cells are 80x60 px (8x6 grid on 640x360). This mask leaves, per cell: columns 3-4 x rows 2-5 nothing,
# columns 2 and 5 x rows 2-5 a 30 px strip (38% background), columns 3-4 x row 1 a 20 px strip (33%).
MOSTLY_ME = (slice(80, 360), slice(190, 450))
LOW_BG_STRIPS = [(slice(120, 360), slice(160, 190)), (slice(120, 360), slice(450, 480))]


def close_up_seg():
    seg = np.zeros((H, W), np.float32)
    seg[MOSTLY_ME] = 1.0
    return seg


def test_only_cells_that_are_mostly_background_are_counted():
    plate = room()
    h = measure_plate_health(plate, plate, close_up_seg(), LIMITS)
    assert h.cells == 30  # 48 - 16 (columns 2-5, rows 2-5) - 2 (columns 3-4, row 1)
    loose = measure_plate_health(plate, plate, close_up_seg(), dataclasses.replace(LIMITS, min_cell_background=0.3))
    assert loose.cells == 40


def test_leftovers_in_cells_mostly_covered_by_me_do_not_count():
    """The segmenter misses a strip of me (arm, hair, shadow) beside the mask. Those cells are mostly
    covered, so by default they are not judged; judged anyway, the same frame fails."""
    plate = room()
    live = plate.copy()
    rng = np.random.default_rng(5)
    for rows, cols in LOW_BG_STRIPS:
        part = plate[rows, cols].astype(np.float32) * 1.42  # plausible as lighting, wrong as background
        part[::6] *= 0.72                                    # ...and with edges the plate does not have
        live[rows, cols] = np.clip(part + rng.normal(0, 2, part.shape), 0, 255).astype(np.uint8)
    strict = measure_plate_health(live, plate, close_up_seg(), LIMITS)
    loose = measure_plate_health(live, plate, close_up_seg(), dataclasses.replace(LIMITS, min_cell_background=0.3))
    assert strict.cells == 30 and loose.cells == 40
    assert strict.ok, strict.describe()
    assert strict.structure > 0.9 and strict.structure > loose.structure + 0.05, (strict.describe(), loose.describe())
    assert loose.max_cell_residual > 2 * strict.max_cell_residual and loose.residual > strict.residual
    assert not loose.ok


def test_no_verdict_when_i_fill_the_frame():
    plate = room()
    seg = np.zeros((H, W), np.float32)
    seg[:, 30:] = 1.0  # leaning into the camera: a 30 px sliver of background, no cell is half background
    assert measure_plate_health(plate, plate, seg, LIMITS) is None
    assert measure_plate_health(plate, plate, np.ones((H, W), np.float32), LIMITS) is None


def test_hysteresis_needs_margin_after_a_failure():
    plate = room()
    live = scaled(np.roll(plate, 3, axis=1), 1.1)  # a nudged camera: structure below 1 but not terrible
    s = measure_plate_health(live, plate, None, LIMITS).structure
    limits = dataclasses.replace(LIMITS, min_structure=s - 0.01, hysteresis=0.2)
    assert measure_plate_health(live, plate, None, limits).ok
    again = measure_plate_health(live, plate, None, limits, recovering=True)
    assert not again.ok and "structure" in again.reasons[0]
    residual_limits = dataclasses.replace(LIMITS, max_residual=1.0, min_structure=-1.0)
    r = measure_plate_health(live, plate, None, residual_limits).residual
    residual_limits = dataclasses.replace(residual_limits, max_residual=r * 1.1)
    assert measure_plate_health(live, plate, None, residual_limits).ok
    assert not measure_plate_health(live, plate, None, residual_limits, recovering=True).ok


def test_monitor_trips_only_after_a_sustained_failure():
    plate = room()
    good = measure_plate_health(plate, plate, None, LIMITS)
    bad = measure_plate_health(np.roll(plate, 80, axis=1), plate, None, LIMITS)
    m = HealthMonitor(fail_s=3.0)
    assert not m.ok and not m.recovering and not m.tripped(0.0)  # nothing judged yet
    assert m.update(good, 0.0) is True and m.ok
    assert m.update(good, 0.5) is False
    assert m.update(bad, 1.0) is True and not m.ok and m.recovering
    assert not m.tripped(1.0) and not m.tripped(3.9)   # failing, but not for 3 s yet
    assert m.update(None, 2.0) is False and m.recovering  # a frame that could not be judged changes nothing
    assert m.update(bad, 3.5) is False
    assert m.tripped(4.0)                                # 3 s after the first failed check
    assert m.update(good, 4.5) is True and m.ok and not m.tripped(9.0)
    # a pass in between restarts the clock
    m.update(bad, 10.0); m.update(good, 12.0); m.update(bad, 12.5)
    assert not m.tripped(13.5) and m.tripped(15.5)
    m.reset()
    assert m.health is None and not m.recovering


def test_standing_up_and_sitting_down_never_trips():
    """Stand up (exposure drops, more of the room visible), sit down (it rises), lean in (few cells)."""
    plate = room()
    m = HealthMonitor(fail_s=3.0)
    v = VanishMask(VanishConfig(), (W, H), (W, H))
    v.set_plate(plate)
    t = 0.0
    for gain, seg in ((1.06, person_seg()), (0.68, person_seg(y0=0, y1=200)), (0.86, person_seg()),
                      (1.22, person_seg()), (0.7, np.zeros((H, W), np.float32)), (1.06, close_up_seg())) * 2:
        live = scaled(plate, gain)
        live[seg > 0] = (30, 220, 60)
        for _ in range(8):  # 4 s in each pose
            t += 0.5
            m.update(v.check_health(live, seg, [], LIMITS, recovering=m.recovering), t)
            assert m.ok and not m.tripped(t), (gain, m.health.describe())


def test_vanish_mask_check_health_works_before_vanish_is_on():
    plate = room()
    v = VanishMask(VanishConfig(), (W, H), (W, H))
    assert v.check_health(plate, person_seg(), [], LIMITS) is None  # no plate yet
    assert v.background_gain(plate, person_seg(), []) is None
    v.set_plate(plate)
    live = scaled(plate, 0.8)
    live[120:300, 260:380] = (30, 220, 60)
    assert v.check_health(live, person_seg(), [], LIMITS).ok
    assert v.background_gain(live, person_seg(), []) == pytest.approx(0.8, abs=0.02)
    moved = VanishMask(VanishConfig(), (W, H), (W, H))
    moved.set_plate(np.roll(plate, 60, axis=1))
    assert not moved.check_health(live, person_seg(), [], LIMITS).ok


# -- background refresh: up to the mask + a small margin ------------------------------------------

def refresh_run(live, seg, frames=40, mode="full", hands=(), **cfg_kw):
    """Run compute() + refresh_plate() for `frames` frames; returns (mask object, original plate, last result)."""
    plate = make_plate()
    v = VanishMask(dataclasses.replace(VanishConfig(), mask_mode=mode, **cfg_kw), (W, H), (W, H))
    v.set_plate(plate)
    r = None
    for _ in range(frames):
        r = v.compute(live, seg, list(hands))
        v.refresh_plate(live)
    return v, plate, r


def drifted_scene(gain=1.08):
    """The room got a little brighter (well under the plate-difference threshold) and I am in it."""
    live = scaled(make_plate(), gain)
    live[120:300, 260:380] = (30, 220, 60)
    return live


def grow(binary, px):
    return cv2.dilate(binary.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))) > 0


def full_mask(r):
    out = np.zeros((H, W), np.float32)
    if r.roi:
        x0, y0, x1, y1 = r.roi
        out[y0:y1, x0:x1] = r.mask
    return out


MARGIN = round(VanishConfig().plate_refresh_margin_px * 0.5)  # config px are at 1280 wide; these tests run at 640


@pytest.mark.parametrize("mode", ["full", "seg_hands", "seg_only"])
def test_refresh_never_touches_the_mask_or_its_margin(mode):
    live, seg = drifted_scene(), person_seg()
    v, plate, r = refresh_run(live, seg, mode=mode)
    assert v.stats.refresh_n == 40 // REFRESH_EVERY
    changed = (v.plate != plate).any(axis=2)
    assert changed.any()
    mask = full_mask(r) > 0                               # the finished mask, feather included
    assert mask[120:300, 260:380].all()
    assert not changed[grow(mask, MARGIN)].any()          # under the mask and in the margin: exactly as captured
    for source in (r.seg, r.grown, r.hull, r.diff):       # every single source, each with the margin
        assert not changed[grow(source > 0, MARGIN)].any()
    assert (plate == make_plate()).all()                  # the plate handed to the caller is not modified in place


def test_refresh_reaches_close_to_me():
    """The point of the change: the area right around me follows the light. The old rule kept
    everything within near_me_px + 40 px (55 px at this size) of the mask frozen."""
    live, seg = drifted_scene(), person_seg()
    v, plate, r = refresh_run(live, seg)
    changed = (v.plate != plate).any(axis=2)
    mask = full_mask(r) > 0
    near = grow(mask, 34) & ~grow(mask, 24)   # a ring 24-34 px outside the mask (48-68 px at 1280 wide)
    assert near.sum() > 3000 and changed[near].mean() > 0.95
    old_frozen = grow(seg > 0, 8 + 35 + 20)   # grown segmenter + near_me_px + the old 40 px guard
    assert changed[old_frozen & ~grow(mask, 24)].mean() > 0.95
    assert v.stats.refresh_coverage / v.stats.refresh_n > 0.6  # most of the frame now (it was ~0.35 in the live run)


def test_refresh_moves_the_background_toward_live_slowly():
    live, seg = drifted_scene(), person_seg()
    v, plate, _ = refresh_run(live, seg, frames=REFRESH_EVERY)  # exactly one pass = 4 frames' worth
    far = (slice(20, 80), slice(20, 120))
    step = (v.plate[far].astype(float) - plate[far]).mean() / (live[far].astype(float) - plate[far]).mean()
    assert step == pytest.approx(1 - 0.98 ** REFRESH_EVERY, abs=0.03)   # ~0.02 per frame
    v, plate, _ = refresh_run(live, seg, frames=400)
    assert np.abs(v.plate[far].astype(int) - live[far]).max() <= 1      # converged after a few hundred frames


def test_refresh_needs_every_source_to_agree_it_is_background():
    live, seg = drifted_scene(), person_seg()
    # a hand held out to the side: only the hand-hull source covers it (and it looks like the wall)
    hand = Hand("Right", 0.9, np.stack([0.8 + 0.04 * np.cos(np.linspace(0, 6.28, 21)),
                                        0.3 + 0.04 * np.sin(np.linspace(0, 6.28, 21)), np.zeros(21)], axis=1).astype(np.float32))
    v, plate, r = refresh_run(live, seg, mode="seg_hands", hands=[hand])
    changed = (v.plate != plate).any(axis=2)
    assert (r.hull > 0).sum() > 500 and not changed[grow(r.hull > 0, MARGIN)].any()
    # part of me the segmenter misses, found only by the plate difference (full mode)
    live2 = live.copy()
    live2[150:260, 380:420] = (30, 220, 60)
    v, plate, r = refresh_run(live2, seg, mode="full")
    changed = (v.plate != plate).any(axis=2)
    assert (r.diff[150:260, 384:416] > 0).all()
    assert not changed[150:260, 380:420].any() and not changed[grow(r.diff > 0, MARGIN)].any()


def test_refresh_only_where_the_pixel_matches_the_matched_plate():
    """All sources say background, but the pixel is not the wall (someone or something the segmenter
    cannot see): it must not be burned into the plate, however close to or far from me."""
    live, seg = drifted_scene(), person_seg()
    live[150:260, 420:450] = (20, 20, 230)   # 40 px beside me: outside the mask and its margin
    live[40:140, 540:600] = (20, 20, 230)    # far away
    v, plate, r = refresh_run(live, seg, mode="seg_only")  # no plate-difference source: only the match rule protects
    assert not (full_mask(r)[150:260, 420:450] > 0).any()
    for region in ((slice(150, 260), slice(420, 450)), (slice(40, 140), slice(540, 600))):
        assert (v.plate[region] == plate[region]).all()
    assert (v.plate[20:80, 20:120] != plate[20:80, 20:120]).any()  # ordinary background still refreshes
    # a brightness drift that matches after gain matching IS refreshed, even though raw live != raw plate
    assert abs(float(live[20:80, 20:120].mean()) - float(plate[20:80, 20:120].mean())) > 5


def test_refresh_follows_the_mask_when_i_move():
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
        r = v.compute(live, seg, [])
        v.refresh_plate(live)
    under_me = grow(seg_b > 0, 8)
    assert (v.plate[under_me] == before[under_me]).all()  # frozen from the moment I stand there
    assert (v.plate[120:300, 300:340] != before[120:300, 300:340]).any()  # where I was is refreshed again


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


# -- staleness nudge --------------------------------------------------------------------------

def health_with(residual):
    return PlateHealth(residual, residual * 3, 0.9, 1.0, 30, 0.6, ())


def test_nudge_after_ten_seconds_above_the_soft_limit_and_only_once():
    assert (PlateConfig().health_nudge_residual, PlateConfig().health_nudge_s) == (2.0, 10.0)
    n = StalenessNudge(residual=2.0, hold_s=10.0)
    fired = [n.update(health_with(2.6), t * 0.5, True) for t in range(40)]  # 20 s above the limit
    assert fired.count(True) == 1 and fired.index(True) == 20               # at 10.0 s, a single beep
    assert n.showing
    assert n.update(health_with(1.9), 21.0, True) is False and n.showing    # 1.9 is not clearly below: keep showing
    assert n.update(health_with(1.5), 21.5, True) is False and not n.showing  # under 80% of the limit: cleared
    assert not any(n.update(health_with(2.6), 22.0 + t * 0.5, True) for t in range(19))
    assert n.update(health_with(2.6), 32.0, True) is True                   # a new episode nudges again


def test_nudge_needs_vanish_on_and_a_sustained_residual():
    n = StalenessNudge(2.0, 10.0)
    assert not any(n.update(health_with(3.0), t, False) for t in range(30))  # not vanished: never
    for t in range(30):  # dips under the limit every 8 s: never 10 s in a row
        assert n.update(health_with(1.0 if t % 8 == 0 else 2.5), float(t), True) is False
    n.reset()
    n.update(health_with(2.5), 100.0, True)
    assert n.update(None, 105.0, True) is False and n.above_since == 100.0   # an unjudged frame changes nothing
    assert n.update(health_with(2.5), 111.0, True) is True
    n.update(health_with(2.5), 112.0, False)                                 # vanish switched off: reset
    assert not n.showing and n.above_since is None
    assert "nudge" in CUES and len(CUES["nudge"]) == 1                       # a single soft beep


# -- audio cues --------------------------------------------------------------------------------

def capture_events(person, countdown_s=5.0):
    cap = PlateCapture(frames=5, countdown_s=countdown_s, settle_s=1.0, max_coverage=0.02)
    cap.start(0.0)
    events, t, i = cap.pop_events(), 0.0, 0
    while t < 12 and cap.state not in (cap.DONE, cap.ABORTED):
        t += 1 / 30
        i += 1
        frame = np.full((90, 160, 3), 80, np.uint8)
        mask = np.zeros((90, 160), np.float32)
        if person:  # someone pacing through the frame
            x = 20 + (i * 3) % 100
            frame[20:80, x:x + 30] = 200
            mask[20:80, x:x + 30] = 1.0
        cap.update(frame, mask, t)
        events += cap.pop_events()
    return cap, events


def test_one_tick_per_countdown_second_then_capturing():
    cap, events = capture_events(False)
    assert cap.state == cap.DONE
    assert events == ["tick"] * 5 + ["capturing"]
    assert cap.pop_events() == []
    _, events = capture_events(False, countdown_s=3.0)
    assert events == ["tick"] * 3 + ["capturing"]


def test_abort_emits_the_error_cue():
    cap, events = capture_events(True)
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
