"""Clean plate: averaging, save/load/invalidation, capture sequence, compositor, lighting measurement."""
import json

import numpy as np
import pytest

from incanto.plate.clean_plate import (PlateAccumulator, PlateCapture, load_plate,
                                       make_plate, save_plate)
from incanto.render.compositor import Compositor, composite


def solid(value, w=64, h=36):
    return np.full((h, w, 3), value, np.uint8)


def test_accumulator_averages_frames():
    acc = PlateAccumulator()
    for v in (10, 20, 30):
        acc.add(solid(v))
    out = acc.result()
    assert out.dtype == np.float32 and out.shape == (36, 64, 3)
    assert np.allclose(out, 20.0)


def test_save_load_roundtrip_and_sidecar(tmp_path):
    plate = make_plate(np.full((36, 64, 3), 100.4, np.float32), 30, None, 0.001)
    save_plate(plate, tmp_path)
    meta = json.loads((tmp_path / "plate.json").read_text())
    assert (meta["width"], meta["height"], meta["exposure_mode"]) == (64, 36, "auto")
    assert {"mean_brightness", "timestamp", "exposure_value"} <= set(meta)
    loaded = load_plate(tmp_path, 64, 36)
    assert loaded is not None and loaded.u8.mean() == pytest.approx(100, abs=1)


def test_load_invalidated_on_resolution_change_or_missing(tmp_path):
    assert load_plate(tmp_path, 64, 36) is None
    save_plate(make_plate(np.full((36, 64, 3), 50, np.float32), 5, -6.0, 0.0), tmp_path)
    assert load_plate(tmp_path, 1280, 720) is None
    assert json.loads((tmp_path / "plate.json").read_text())["exposure_mode"] == "manual"


def run_capture(mask_value, frames=5):
    cap = PlateCapture(frames=frames, countdown_s=3.0, settle_s=1.0, max_coverage=0.02)
    cap.start(0.0)
    mask = np.full((36, 64), mask_value, np.float32)
    t = 0.0
    states = []
    while t < 10 and cap.state not in (cap.DONE, cap.ABORTED):
        t += 0.1
        cap.update(solid(80), mask, t)
        states.append(cap.state)
    return cap, states, t


def test_capture_sequence_countdown_settle_then_capture():
    cap, states, t = run_capture(0.0)
    assert cap.state == cap.DONE
    assert states[0] == cap.COUNTDOWN and cap.SETTLE in states and cap.CAPTURING in states
    assert t >= 4.0  # 3 s countdown + 1 s settle before any frame is taken
    assert np.allclose(cap.result(), 80.0)


def test_capture_aborts_when_person_visible():
    cap, _, _ = run_capture(1.0)
    assert cap.state == cap.ABORTED
    assert "NOT saved" in cap.abort_reason
    with pytest.raises(ValueError):
        cap.result()  # nothing was accumulated: no bad plate


def test_composite_soft_mask():
    frame, plate = solid(200), solid(100)
    mask = np.zeros((36, 64), np.float32)
    mask[:, 32:] = 1.0
    mask[:, 16] = 0.5
    out = composite(frame, mask, plate)
    assert (out[:, 0] == 200).all() and (out[:, 40] == 100).all()
    assert abs(int(out[0, 16, 0]) - 150) <= 1  # soft edge is a blend, not a hard cut
    # uint8 fast path and float path agree
    out_f = composite(frame, mask, plate.astype(np.float32))
    assert np.abs(out.astype(int) - out_f.astype(int)).max() <= 1


def test_compositor_vanish_only_with_plate_and_flag():
    frame, plate, mask = solid(200), solid(100), np.ones((36, 64), np.float32)
    comp = Compositor()
    assert comp.render(frame, mask, plate, 0.03) is frame
    assert comp.render(frame, mask, None, 0.03, vanish=True) is frame
    assert (comp.render(frame, mask, plate, 0.03, vanish=True) == 100).all()
