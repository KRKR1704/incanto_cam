"""Vanish mask: dilation, hand hulls, plate difference, brightness matching, mask modes."""
import dataclasses

import numpy as np
import pytest

from incanto.config import VanishConfig
from incanto.perception.hands import Hand
from incanto.render.vanish_mask import VanishMask

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
    assert r.mask.shape == (H, W) and r.mask.dtype == np.float32
    assert r.mask[200, 190] > 0.9 and r.mask[200, 410] > 0.9  # grown beyond the segmenter edge
    assert r.mask[200, 100] == 0.0


def test_hand_hull_covers_missed_hand():
    plate, live = scene()
    v = vm("seg_hands", dilate_px=0, hand_hull_dilate_px=10)
    v.set_plate(plate)
    hands = [hand_at(0.8, 0.5)]
    r = v.compute(live, np.zeros((H, W), np.float32), hands)
    assert r.mask[int(0.5 * H), int(0.8 * W)] > 0.9
    assert r.mask[10, 10] == 0.0
    v2 = vm("seg_only", dilate_px=0)
    v2.set_plate(plate)
    assert v2.compute(live, np.zeros((H, W), np.float32), hands).mask.max() == 0.0


def test_plate_difference_fills_what_segmenter_missed():
    plate, live = scene()
    seg = seg_covering(200, 260)  # only a strip of the person
    seg_hands = vm("seg_hands", dilate_px=0)
    seg_hands.set_plate(plate)
    full = vm("full", dilate_px=0)
    full.set_plate(plate)
    missed = seg_hands.compute(live, seg, []).mask[200, 350]
    covered = full.compute(live, seg, []).mask[200, 350]
    assert missed < 0.1 and covered > 0.9
    assert full.compute(live, seg, []).mask[20, 20] < 0.05  # empty background stays untouched


def test_brightness_matching_estimates_gain_and_fixes_difference():
    plate, live = scene(gain=1.2)
    v = vm("full", plate_match="gain", dilate_px=0)
    v.set_plate(plate)
    r = v.compute(live, seg_covering(200, 400), [])
    assert v.gain == pytest.approx([1.2, 1.2, 1.2], abs=0.03)
    # the matched plate is close to live in the background, the raw plate was ~20% darker
    bg = (slice(0, 80), slice(0, W))
    assert abs(float(r.plate[bg].mean()) - float(live[bg].mean())) < 3
    assert abs(float(plate[bg].mean()) - float(live[bg].mean())) > 10
    assert r.mask[20, 20] < 0.05  # matching must prevent the plate-difference mask flagging the whole frame


def test_without_matching_a_lighting_change_is_ignored_not_vanished():
    plate, live = scene(gain=1.5)
    v = vm("full", plate_match="off", dilate_px=0, plate_diff_threshold=10)
    v.set_plate(plate)
    r = v.compute(live, seg_covering(200, 400), [])
    assert v.stats.diff_skipped == 1  # most of the frame differs -> difference mask skipped, not a frozen frame
    assert r.mask[20, 20] < 0.05


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
    assert r.plate.shape == (H, W, 3) and r.plate.dtype == np.uint8
