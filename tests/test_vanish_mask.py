"""Vanish mask: dilation, hand hulls, plate difference, brightness matching, mask modes."""
import dataclasses

import numpy as np
import pytest

from incanto.config import VanishConfig
from incanto.perception.hands import Hand
from incanto.render.vanish_mask import VanishMask, fill_missing_cells

W, H = 640, 360


def make_plate():
    """Smooth, textured background so gain/offset can be estimated."""
    y, x = np.mgrid[0:H, 0:W]
    base = 80 + 60 * np.sin(x / 70.0) * np.cos(y / 50.0) + (x / W) * 40
    return np.stack([base, base * 0.9 + 10, base * 1.1 - 10], axis=2).clip(0, 255).astype(np.uint8)


def scene(gain=1.0):
    plate = make_plate()
    live = np.clip(plate.astype(np.float32) * gain, 0, 255).astype(np.uint8)
    live[100:300, 200:400] = (30, 200, 60)  # the "person": very different from the plate
    return plate, live


def seg_covering(x0, x1):
    seg = np.zeros((H, W), np.float32)
    seg[100:300, x0:x1] = 1.0  # segmenter finds only part of the person
    return seg


def full_mask(r):
    """Expand a VanishResult's ROI mask into a full-frame array (zeros outside the ROI)."""
    out = np.zeros((H, W), np.float32)
    if r.roi:
        x0, y0, x1, y1 = r.roi
        out[y0:y1, x0:x1] = r.mask
    return out


def full_plate(r, fallback):
    out = fallback.copy()
    if r.roi:
        x0, y0, x1, y1 = r.roi
        out[y0:y1, x0:x1] = r.plate
    return out


def vm(mode="full", **kw):
    cfg = dataclasses.replace(VanishConfig(), mask_mode=mode, **kw)
    return VanishMask(cfg, (W, H), (W, H))


def hand_at(cx, cy, r=0.04):
    angles = np.linspace(0, 2 * np.pi, 21, endpoint=False)
    pts = np.stack([cx + r * np.cos(angles), cy + r * np.sin(angles), np.zeros(21)], axis=1).astype(np.float32)
    return Hand("Right", 0.9, pts)


def test_dilation_grows_segmenter_mask():
    plate, live = scene()
    v = vm("seg_only", dilate_px=40)
    v.set_plate(plate)
    r = v.compute(live, seg_covering(200, 400), [])
    assert full_mask(r).shape == (H, W) and r.mask.dtype == np.float32
    assert r.roi is not None and r.mask.shape == (r.roi[3] - r.roi[1], r.roi[2] - r.roi[0])  # only the ROI is built
    assert full_mask(r)[200, 190] > 0.9 and full_mask(r)[200, 410] > 0.9  # grown beyond the segmenter edge
    assert full_mask(r)[200, 100] == 0.0


def test_hand_hull_covers_missed_hand():
    plate, live = scene()
    v = vm("seg_hands", dilate_px=0, hand_hull_dilate_px=10)
    v.set_plate(plate)
    hands = [hand_at(0.8, 0.5)]
    r = v.compute(live, np.zeros((H, W), np.float32), hands)
    assert full_mask(r)[int(0.5 * H), int(0.8 * W)] > 0.9
    assert full_mask(r)[10, 10] == 0.0
    v2 = vm("seg_only", dilate_px=0)
    v2.set_plate(plate)
    assert v2.compute(live, np.zeros((H, W), np.float32), hands).roi is None  # nothing to replace


def test_plate_difference_fills_what_segmenter_missed():
    plate, live = scene()
    seg = seg_covering(200, 260)  # only a strip of the person
    seg_hands = vm("seg_hands", dilate_px=0)
    seg_hands.set_plate(plate)
    full = vm("full", dilate_px=0, near_me_px=200)  # wide zone: the unsegmented rest of the person is "near me"
    full.set_plate(plate)
    missed = full_mask(seg_hands.compute(live, seg, []))[200, 350]
    covered = full_mask(full.compute(live, seg, []))[200, 350]
    assert missed < 0.1 and covered > 0.9
    assert full_mask(full.compute(live, seg, []))[20, 20] < 0.05  # empty background stays untouched


def test_brightness_matching_estimates_gain_and_fixes_difference():
    plate, live = scene(gain=1.2)
    v = vm("full", plate_match="gain", dilate_px=0)
    v.set_plate(plate)
    r = v.compute(live, seg_covering(200, 400), [])
    assert v.gain == pytest.approx([1.2, 1.2, 1.2], abs=0.03)
    # inside the ROI the matched plate is the plate brightened by the true 1.2 (the raw one is ~17% off)
    x0, y0, x1, y1 = r.roi
    expected = np.clip(plate[y0:y1, x0:x1].astype(np.float32) * 1.2, 0, 255)
    assert abs(float(r.plate.mean()) - float(expected.mean())) < 3
    assert abs(float(plate[y0:y1, x0:x1].mean()) - float(expected.mean())) > 10
    assert full_mask(r)[20, 20] < 0.05  # matching must prevent the plate-difference mask flagging the whole frame


def test_cell_guard_ignores_a_lighting_wash_but_not_a_person():
    plate, live = scene(gain=1.45)                      # whole room 45% brighter and the gain map is off
    kw = dict(plate_match="off", dilate_px=0, plate_diff_threshold=10, near_me_px=700)
    guarded = vm("full", **kw)
    guarded.set_plate(plate)
    r = guarded.compute(live, seg_covering(200, 400), [])
    assert guarded.stats.guard_cells > 0 and guarded.stats.guard_frames_any == 1
    assert full_mask(r)[20, 20] < 0.05                  # the wash did not freeze the frame
    unguarded = vm("full", plate_diff_cell_guard=False, **kw)
    unguarded.set_plate(plate)
    assert full_mask(unguarded.compute(live, seg_covering(200, 400), []))[20, 20] > 0.9  # what the guard prevents
    # a person (other colours, not the plate's texture under one gain) is never treated as a wash
    plate, live = scene()
    g2 = vm("full", plate_match="off", dilate_px=0, near_me_px=700)
    g2.set_plate(plate)
    r = g2.compute(live, np.zeros((H, W), np.float32), [])
    assert full_mask(r)[200, 300] > 0.99


def test_large_blob_covers_a_segmenter_dropout_but_small_objects_elsewhere_stay_live():
    plate, live = scene()
    empty = np.zeros((H, W), np.float32)              # segmenter found nobody: no zone at all
    on = vm("full", dilate_px=0)
    on.set_plate(plate)
    assert full_mask(on.compute(live, empty, []))[200, 300] > 0.99      # body-sized difference accepted anywhere
    assert on.stats.large_blob_frames == 1 and on.stats.seg_empty == 1
    off = vm("full", dilate_px=0, plate_diff_large_blob_fraction=0.0)
    off.set_plate(plate)
    assert full_mask(off.compute(live, empty, []))[200, 300] == 0.0     # documented limitation when disabled
    live = plate.copy()
    live[20:60, 500:560] = (0, 0, 255)                # small moved object, ~2% of the frame, nobody near it
    small = vm("full", dilate_px=0)
    small.set_plate(plate)
    assert full_mask(small.compute(live, empty, []))[40, 530] == 0.0


def test_colour_metric_catches_similar_brightness_clothes():
    plate = np.full((H, W, 3), 100, np.uint8)
    live = plate.copy()
    live[100:300, 200:400] = (60, 100, 140)           # grey level 107 vs 100, but clearly another colour
    seg = seg_covering(200, 230)
    results = {}
    for metric in ("gray", "max_channel", "lab"):
        v = vm("full", dilate_px=0, near_me_px=200, plate_match="off", plate_diff_metric=metric,
               plate_diff_cell_guard=False)
        v.set_plate(plate)
        results[metric] = float(full_mask(v.compute(live, seg, []))[200, 330])
    assert results["gray"] == 0.0
    assert results["max_channel"] > 0.99 and results["lab"] > 0.99


def test_plate_difference_attacks_fast_and_releases_slowly():
    plate, live = scene()
    v = vm("full", dilate_px=0, near_me_px=200, plate_match="off", plate_diff_large_blob_fraction=0.0)
    v.set_plate(plate)
    empty_live = plate.copy()
    seg = seg_covering(200, 400)
    first = v.compute(live, seg, [])
    results = [v.compute(empty_live, np.zeros((H, W), np.float32), []) for _ in range(14)]
    diff = [float(r.diff[200, 300]) for r in results]
    assert first.diff[200, 300] > 0.99                              # attack: instant
    assert 0.7 < diff[0] < 0.9 and diff[0] > diff[3] > diff[8]      # release: decays by the configured factor
    assert full_mask(results[0])[200, 300] == 1.0                   # the solid core holds while it is still above the threshold
    assert full_mask(results[-1])[200, 300] == 0.0                  # and it does not linger forever
    v.reset_temporal()
    assert v._prev_diff is None


def test_solid_core_with_feather_only_outside():
    plate, live = scene()
    v = vm("seg_only", dilate_px=0, core_close_px=0, outer_feather_px=12)  # small == output here: ring is 6 px
    v.set_plate(plate)
    mask = full_mask(v.compute(live, seg_covering(200, 400), []))
    core = mask[110:290, 210:390]
    assert core.min() == 1.0                                       # exactly 1.0 across the covered area
    assert mask[200, 100] == 0.0                                   # beyond the ring
    ring = mask[200, 190:200]
    assert (np.diff(ring) >= -1e-6).all() and 0.0 < ring[-3] < 1.0  # smooth ramp that rises toward the core
    assert mask[200, 199] > 0.8 and mask[200, 399] == 1.0          # the ring hugs the core edge, outside it
    assert mask[200, 405:412].max() < 1.0                          # on the right side the ring is outside the core too


def test_gain_offset_recovers_offset():
    plate, live = scene()
    live = np.clip(plate.astype(np.float32) * 0.9 + 15, 0, 255).astype(np.uint8)
    live[100:300, 200:400] = (30, 200, 60)
    v = vm("full", plate_match="gain_offset", dilate_px=0)
    v.set_plate(plate)
    v.compute(live, seg_covering(200, 400), [])
    assert v.gain == pytest.approx([0.9] * 3, abs=0.05)
    assert v.offset == pytest.approx([15.0] * 3, abs=4)


def test_result_pieces_for_debug_view():
    plate, live = scene()
    v = vm("full")
    v.set_plate(plate)
    r = v.compute(live, seg_covering(200, 400), [hand_at(0.8, 0.5)])
    for piece in (r.seg, r.grown, r.hull, r.diff):
        assert piece.shape == (H, W)
    assert r.plate.dtype == np.uint8 and r.plate.shape[:2] == r.mask.shape


def test_plate_difference_only_near_me_moved_objects_elsewhere_stay_live():
    plate, live = scene()
    live[20:80, 500:600] = (0, 0, 255)  # an object that moved, far from the person
    v = vm("full", dilate_px=0, near_me_px=70)
    v.set_plate(plate)
    r = v.compute(live, seg_covering(200, 400), [])
    assert full_mask(r)[50, 550] < 0.05          # stays live: not replaced by the old plate
    assert r.zone[200, 300] > 0.9 and r.zone[50, 550] < 0.05
    wide = vm("full", dilate_px=0, near_me_px=700)
    wide.set_plate(plate)
    assert full_mask(wide.compute(live, seg_covering(200, 400), []))[50, 550] > 0.9  # zone covering it -> vanished


def test_near_me_zone_follows_hand_hulls():
    plate, live = scene()
    live[250:300, 520:560] = (0, 0, 255)  # something that changed right next to the raised hand
    v = vm("full", dilate_px=0, near_me_px=70)
    v.set_plate(plate)
    r = v.compute(live, np.zeros((H, W), np.float32), [hand_at(0.8, 0.75)])
    assert r.zone[int(0.75 * H), int(0.8 * W)] > 0.9


def uneven_scene(left=0.8, right=1.3):
    plate = make_plate()
    ramp = np.linspace(left, right, W, dtype=np.float32)[None, :, None]  # lighting varies across the frame
    live = np.clip(plate.astype(np.float32) * ramp, 0, 255).astype(np.uint8)
    live[100:300, 200:400] = (30, 200, 60)
    return plate, live


def settled(mode, plate, live, frames=6):
    v = vm("full", plate_match=mode, dilate_px=0, plate_match_smoothing=1.0)
    v.set_plate(plate)
    for _ in range(frames):
        v.compute(live, seg_covering(200, 400), [])
    return v


def test_gain_map_beats_global_gain_on_uneven_lighting():
    plate, live = uneven_scene()
    global_v, map_v = settled("gain", plate, live), settled("gain_map", plate, live)
    for v in (global_v, map_v):
        v.stats.reset()
        for _ in range(4):  # statistics are sampled on every 2nd estimate, estimates on every 2nd frame
            v.compute(live, seg_covering(200, 400), [])
    g, m = global_v.stats, map_v.stats
    assert m.resid_mean_abs / m.resid_n < 0.5 * (g.resid_mean_abs / g.resid_n)
    assert m.resid_max_abs < g.resid_max_abs
    assert m.gain_min < 0.9 and m.gain_max > 1.2  # the grid really varies


def test_gain_map_applied_to_full_resolution_plate():
    plate, live = uneven_scene()
    v = settled("gain_map", plate, live)
    r = v.compute(live, seg_covering(200, 400), [])
    x0, y0, x1, y1 = r.roi
    ramp = np.linspace(0.8, 1.3, W, dtype=np.float32)[None, x0:x1, None]  # the true lighting across the ROI
    expected = np.clip(plate[y0:y1, x0:x1].astype(np.float32) * ramp, 0, 255)
    assert np.abs(r.plate.astype(np.float32) - expected).mean() < 4
    # and the left edge is darker than the right edge, as the lighting is
    assert r.plate[:, :20].mean() / plate[y0:y1, x0:x0 + 20].mean() < r.plate[:, -20:].mean() / plate[y0:y1, x1 - 20:x1].mean()


def test_fill_missing_cells_borrows_from_neighbours():
    grid = np.ones((6, 8, 3), np.float32) * 1.0
    grid[0, 0] = 1.4
    valid = np.ones((6, 8), bool)
    valid[2:4, 3:5] = False
    grid[2:4, 3:5] = 9.0  # garbage in the invalid cells
    filled, ok = fill_missing_cells(grid, valid)
    assert ok.all() and filled[2:4, 3:5].max() < 2.0
    _, none_ok = fill_missing_cells(grid, np.zeros((6, 8), bool))
    assert not none_ok.any()  # nothing valid: nothing to borrow


def test_leak_view_is_red_where_output_differs_from_matched_plate():
    from incanto.render import overlay
    plate = np.full((72, 128, 3), 100, np.uint8)
    out = plate.copy()
    out[20:40, 30:60] = (100, 100, 180)                # a leaking patch
    preview = out.copy()
    overlay.draw_leak(preview, out, plate, (0, 0, 128, 72))
    leak, clean = preview[30, 45], preview[60, 100]
    assert leak[2] > 200 and leak[0] < 100            # strongly red
    assert clean[2] == 100 and clean[0] == 100        # untouched where output == plate
