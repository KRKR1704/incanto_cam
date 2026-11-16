"""Vanish styles: freeze output = gain-matched plate + grain, grain statistics, region growth, crossfade."""
import dataclasses

import numpy as np
import pytest

from incanto.config import ConfigError, VanishConfig, load_config
from incanto.plate.clean_plate import PlateAccumulator, load_plate, make_plate, save_plate
from incanto.render.grain import AMP_UNIT, GrainProfile, GrainSynth
from incanto.render.vanish_mask import VanishMask
from incanto.render.vanish_style import CrossFade, RegionHold, VanishStyler

W, H = 640, 360


def make_plate_img():
    y, x = np.mgrid[0:H, 0:W]
    base = 90 + 50 * np.sin(x / 70.0) * np.cos(y / 50.0) + (x / W) * 30
    return np.stack([base, base * 0.9 + 10, base * 1.1 - 10], axis=2).clip(0, 255).astype(np.uint8)


def plate_with_noise(std):
    img = make_plate_img()
    return make_plate(img.astype(np.float32), 30, None, 0.0, np.full(img.shape, std, np.float32))


def seg_box(x0, x1, y0=100, y1=300):
    seg = np.zeros((H, W), np.float32)
    seg[y0:y1, x0:x1] = 1.0
    return seg


def live_frame(plate_u8, gain, person=(200, 400)):
    live = np.clip(plate_u8.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    live[100:300, person[0]:person[1]] = (30, 200, 60)
    return live


def styler(style="freeze", **kw):
    cfg = dataclasses.replace(VanishConfig(), style=style, **kw)
    vm = VanishMask(cfg, (W, H), (W, H))
    return VanishStyler(cfg, vm, (W, H), seed=1), vm


def start(st, vm, plate):
    vm.set_plate(plate.u8)
    st.set_plate(plate)


# -- crossfade ---------------------------------------------------------------------------------
def test_crossfade_alpha_follows_dt():
    f = CrossFade(0.4)
    assert f.step(True, 0.1) == pytest.approx(0.25)
    assert f.step(True, 0.1) == pytest.approx(0.5)
    assert f.step(True, 0.3) == 1.0                       # clamped
    assert f.step(False, 0.2) == pytest.approx(0.5)
    assert f.step(False, 0.4) == 0.0


def test_crossfade_is_frame_rate_independent():
    a, b = CrossFade(0.4), CrossFade(0.4)
    for _ in range(6):
        a.step(True, 1 / 30)       # 0.2 s at 30 fps
    for _ in range(12):
        b.step(True, 1 / 60)       # 0.2 s at 60 fps
    assert a.alpha == pytest.approx(b.alpha) == pytest.approx(0.5)


def test_crossfade_zero_duration_is_a_cut_and_negative_dt_ignored():
    f = CrossFade(0.0)
    assert f.step(True, 0.0) == 1.0 and f.step(False, 0.0) == 0.0
    g = CrossFade(0.4)
    g.step(True, 0.2)
    assert g.step(True, -1.0) == pytest.approx(0.5)


# -- grain -------------------------------------------------------------------------------------
def test_grain_statistics_match_the_profile():
    base = np.full((H, W, 3), 128, np.uint8)
    std = np.empty((H, W, 3), np.float32)
    std[:, :W // 2] = 1.5
    std[:, W // 2:] = 4.0
    prof = GrainProfile(std, base)
    amp = prof.amplitude(base)               # unchanged brightness: the stored std as is
    synth = GrainSynth((W, H), seed=3)
    d = synth.apply(base, amp).astype(np.float32) - 128.0
    assert abs(d.mean()) < 0.05
    assert d[:, :W // 2].std() == pytest.approx(1.5, rel=0.06)
    assert d[:, W // 2:].std() == pytest.approx(4.0, rel=0.06)


def test_grain_is_fresh_every_frame():
    base = np.full((H, W, 3), 128, np.uint8)
    amp = GrainProfile.uniform(3.0, (H, W))
    synth = GrainSynth((W, H), seed=4)
    frames = [synth.apply(base, amp).astype(np.float32) - 128 for _ in range(6)]
    for a, b in zip(frames, frames[1:]):
        assert not np.array_equal(a, b)
        assert abs(np.corrcoef(a.ravel(), b.ravel())[0, 1]) < 0.05


def test_grain_follows_brightness_curve():
    """Noise that grows with brightness: a plate matched brighter gets stronger grain."""
    y, x = np.mgrid[0:H, 0:W]
    ramp = np.repeat((40 + 150 * x / W).astype(np.uint8)[..., None], 3, axis=2)
    std = (0.5 + ramp.astype(np.float32) / 40.0)         # std 1.5 .. 5.25, rising with luma
    prof = GrainProfile(std, ramp)
    same = prof.amplitude(ramp)
    brighter = prof.amplitude(np.clip(ramp.astype(np.float32) * 1.3, 0, 255).astype(np.uint8))
    assert same.mean() == pytest.approx(std.mean() * AMP_UNIT, rel=0.03)
    assert brighter.mean() > same.mean() * 1.15
    off = prof.amplitude(ramp, scale=0.0)
    assert not off.any()


def test_amplitude_ratio_is_bounded():
    base = np.full((H, W, 3), 100, np.uint8)
    prof = GrainProfile(np.full((H, W, 3), 2.0, np.float32), base)
    amp = prof.amplitude(np.full((H, W, 3), 250, np.uint8))
    assert amp.max() <= 4.0 * 2.0 * AMP_UNIT + 1


def test_accumulator_measures_temporal_noise():
    rng = np.random.default_rng(0)
    truth = make_plate_img().astype(np.float32)
    acc = PlateAccumulator()
    for _ in range(30):
        acc.add(np.clip(truth + rng.normal(0, 2.5, truth.shape), 0, 255).astype(np.float32))
    std = acc.noise()
    assert std.shape == truth.shape
    mid = std[20:-20, 20:-20]
    assert mid.mean() == pytest.approx(2.5, rel=0.08)
    assert PlateAccumulator().noise() is None


def test_noise_profile_round_trips_with_the_plate(tmp_path):
    plate = plate_with_noise(2.0)
    save_plate(plate, tmp_path)
    back = load_plate(tmp_path, W, H)
    assert back.meta.has_noise
    assert back.noise.mean() == pytest.approx(2.0, abs=0.05)
    save_plate(make_plate(plate.data, 30, None, 0.0), tmp_path)   # a plate without a profile removes the old one
    assert load_plate(tmp_path, W, H).noise is None


# -- freeze ------------------------------------------------------------------------------------
def test_freeze_output_is_gain_matched_plate_plus_grain():
    plate = plate_with_noise(2.0)
    st, vm = styler("freeze", transition_s=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.25)
    seg = seg_box(200, 400)
    out, res = st.render(live, live, seg, [], True, 1 / 30)
    assert res is None                                    # no mask is computed at all
    base = st._base
    bg = (slice(0, 60), slice(0, W))                      # background strip well away from the person
    assert base[bg].astype(np.float32).mean() == pytest.approx(plate.u8[bg].astype(np.float32).mean() * 1.25, rel=0.02)
    d = out.astype(np.float32) - base.astype(np.float32)
    assert abs(d.mean()) < 0.1
    assert d.std() == pytest.approx(2.0, rel=0.08)        # the profile's std
    # the person's area shows the plate, not the person
    p = (slice(150, 250), slice(250, 350))
    assert np.abs(out[p].astype(np.float32) - base[p]).mean() < 4
    assert np.abs(out[p].astype(np.float32) - live[p]).mean() > 30


def test_freeze_without_grain_equals_the_gain_matched_plate_exactly():
    plate = plate_with_noise(2.0)
    st, vm = styler("freeze", transition_s=0.0, grain_scale=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.25)
    out, _ = st.render(live, live, seg_box(200, 400), [], True, 1 / 30)
    assert np.array_equal(out, st._base)
    out2, _ = st.render(live, live, seg_box(200, 400), [], True, 1 / 30)
    assert np.array_equal(out2, out)


def test_freeze_is_held_fixed_whatever_the_live_picture_does():
    plate = plate_with_noise(2.0)
    st, vm = styler("freeze", transition_s=0.0, grain_scale=0.0)
    start(st, vm, plate)
    a = live_frame(plate.u8, 1.25, person=(200, 400))
    out_a, _ = st.render(a, a, seg_box(200, 400), [], True, 1 / 30)
    b = live_frame(plate.u8, 0.7, person=(10, 200))       # stood up and walked, and the lights changed
    out_b, _ = st.render(b, b, seg_box(10, 200), [], True, 1 / 30)
    assert np.array_equal(out_a, out_b)


def test_freeze_missing_noise_profile_gives_no_grain():
    plate = make_plate(make_plate_img().astype(np.float32), 30, None, 0.0)   # old plate: no profile
    st, vm = styler("freeze", transition_s=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.0)
    out, _ = st.render(live, live, seg_box(200, 400), [], True, 1 / 30)
    assert np.array_equal(out, st._base) and not st.grain_available


def test_freeze_fades_in_and_out_by_dt_and_rematches_before_reappearing():
    plate = plate_with_noise(2.0)
    st, vm = styler("freeze", transition_s=0.4, grain_scale=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.0)
    seg = seg_box(200, 400)
    out, _ = st.render(live, live, seg, [], True, 0.1)     # 25 % of the way
    assert st.fade.alpha == pytest.approx(0.25)
    expect = np.clip(0.75 * live.astype(np.float32) + 0.25 * st._base.astype(np.float32) + 0.5, 0, 255)
    assert np.abs(out.astype(np.float32) - expect).max() <= 1.5
    st.render(live, live, seg, [], True, 0.5)
    assert st.fade.alpha == 1.0
    # lighting drifted while frozen; before reappearing the plate is matched to the new light
    dim = live_frame(plate.u8, 0.8)
    held = st._base.copy()
    out, _ = st.render(dim, dim, seg, [], False, 0.1)      # fade-out starts
    assert st.fade.alpha == pytest.approx(0.75)
    assert st._fresh is not None
    bg = (slice(0, 60), slice(0, W))
    assert st._fresh[bg].astype(np.float32).mean() == pytest.approx(plate.u8[bg].astype(np.float32).mean() * 0.8, rel=0.03)
    assert np.array_equal(st._base, held)                  # the held picture itself is untouched
    st.render(dim, dim, seg, [], False, 0.5)
    out, res = st.render(dim, dim, seg, [], False, 0.1)
    assert st.fade.alpha == 0.0 and np.array_equal(out, dim) and not st.active


def test_turning_on_again_mid_fade_keeps_what_is_shown():
    plate = plate_with_noise(2.0)
    st, vm = styler("freeze", transition_s=0.4, grain_scale=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.0)
    seg = seg_box(200, 400)
    st.render(live, live, seg, [], True, 1.0)
    dim = live_frame(plate.u8, 0.8)
    st.render(dim, dim, seg, [], False, 0.2)               # halfway out
    shown = st._frozen_layer(0.5).copy()
    st.render(dim, dim, seg, [], True, 0.0)                # V again
    assert np.array_equal(st._base, shown) and st._fresh is None


# -- live and region ---------------------------------------------------------------------------
def test_live_style_is_the_unchanged_masked_composite():
    plate = plate_with_noise(2.0)
    st, vm = styler("live", transition_s=0.4)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.0)
    out, res = st.render(live, live, seg_box(200, 400), [], True, 1 / 30)   # no fade: it is on at once
    assert res is not None and st.fade.alpha == 1.0
    assert np.abs(out[150:250, 250:350].astype(np.float32) - plate.u8[150:250, 250:350]).mean() < 4
    out, res = st.render(live, live, seg_box(200, 400), [], False, 1 / 30)
    assert res is None and np.array_equal(out, live)


def test_region_only_grows_while_vanished():
    hold = RegionHold((W, H), (W, H), 120, 60, 0.1)
    areas, snapshots = [], []
    for x in (50, 300, 150, 520, 100):                     # walks around; the segmenter mask leaves each spot
        src = np.zeros((H, W), np.float32)
        src[120:260, x:x + 60] = 1.0
        hold.update(src)
        areas.append(hold.area)
        snapshots.append(hold.hold.copy())
    assert areas == sorted(areas) and areas[-1] > areas[0]
    for earlier, later in zip(snapshots, snapshots[1:]):
        assert (later >= earlier).all()                    # nothing ever drops out
    assert not hold.update(np.zeros((H, W), np.float32))   # an empty mask changes nothing
    assert hold.area == areas[-1]
    hold.reset()
    assert hold.area == 0


def test_region_is_margin_dilated_and_feathered():
    hold = RegionHold((W, H), (W, H), 120, 60, 0.1)
    src = np.zeros((H, W), np.float32)
    src[160:200, 300:340] = 1.0
    hold.update(src)
    a = hold.alpha_q()
    k = W / 1280 / 4                                        # config px -> quarter-resolution cells
    row, edge = 180 // 4, 340 / 4                           # the source's right edge, in cells
    margin, feather = 120 * k, 60 * k
    assert a[row, int(edge + margin * 0.8)] == 1.0          # within the margin
    assert a[row, int(edge + margin + feather + 3)] == 0.0  # beyond margin + feather
    assert 0.0 < a[row, int(edge + margin + feather / 2)] < 1.0  # in the feather
    roi, mask = hold.full()
    assert mask.shape == (roi[3] - roi[1], roi[2] - roi[0]) and mask.max() <= 1.0 and mask.min() >= 0.0


def test_region_style_shows_plate_inside_and_live_outside():
    plate = plate_with_noise(2.0)
    st, vm = styler("region", transition_s=0.0, grain_scale=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.0)
    out, res = st.render(live, live, seg_box(200, 400), [], True, 1 / 30)
    assert res is not None
    assert np.abs(out[150:250, 250:350].astype(np.float32) - plate.u8[150:250, 250:350]).mean() < 4
    # far from the person the picture is the live frame, untouched
    far = (slice(0, 40), slice(560, W))
    assert np.array_equal(out[far], live[far])
    area = st.region.area
    # the person walks off; the region stays and does not shrink
    out, _ = st.render(live_frame(plate.u8, 1.0, person=(10, 60)), live_frame(plate.u8, 1.0, person=(10, 60)),
                       seg_box(10, 60), [], True, 1 / 30)
    assert st.region.area >= area


def test_region_hold_stops_growing_when_vanish_is_off():
    plate = plate_with_noise(2.0)
    st, vm = styler("region", transition_s=0.4, grain_scale=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.0)
    st.render(live, live, seg_box(200, 400), [], True, 0.2)
    area = st.region.area
    other = live_frame(plate.u8, 1.0, person=(20, 120))
    st.render(other, other, seg_box(20, 120), [], False, 0.1)      # fading out: no more growth
    assert st.fade.alpha == pytest.approx(0.25) and st.region.area == area
    st.render(other, other, seg_box(20, 120), [], False, 0.2)      # fade over: the session's region is dropped
    assert st.region.area == 0


def test_style_cycle_and_switch_while_vanished():
    plate = plate_with_noise(2.0)
    st, vm = styler("freeze", transition_s=0.0)
    start(st, vm, plate)
    live = live_frame(plate.u8, 1.0)
    seg = seg_box(200, 400)
    st.render(live, live, seg, [], True, 1 / 30)
    seen = [st.style]
    for _ in range(3):
        seen.append(st.cycle_style())
        out, _ = st.render(live, live, seg, [], True, 1 / 30)      # every style renders while vanished
        assert out.shape == live.shape
    assert seen == ["freeze", "region", "live", "freeze"]


def test_config_validation(tmp_path):
    for bad in ("style: blur", "region_margin_px: -1", "transition_s: -0.1", "grain_scale: 9"):
        f = tmp_path / "c.yaml"
        f.write_text(f"vanish:\n  {bad}\n")
        with pytest.raises(ConfigError):
            load_config(f)
    f = tmp_path / "ok.yaml"
    f.write_text("vanish:\n  style: region\n  grain_scale: 0\n")
    cfg = load_config(f)
    assert cfg.vanish.style == "region" and cfg.vanish.grain_scale == 0
    assert VanishConfig().style == "freeze" and VanishConfig().transition_s == 0.4
    assert VanishConfig().region_margin_px == 120 and VanishConfig().region_feather_px == 60
