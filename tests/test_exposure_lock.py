"""Exposure lock: sweep, choice, tolerance, recapture, and the restore-auto safety net on that path."""
import time

import cv2
import numpy as np
import pytest

from incanto.io import camera as camera_mod
from incanto.io.camera import EXPOSURE_FPS_LIMIT, Camera, ThreadedCamera, restore_auto_exposure_all
from incanto.io.exposure_lock import CANDIDATES, ExposureLock


class FakeRoom:
    """A camera whose picture halves in brightness per exposure step; `at_minus_5` is the brightness
    (relative to the plate) at exposure -5. Auto exposure gives 1.0."""

    def __init__(self, at_minus_5: float):
        self.at_minus_5 = at_minus_5
        self.calls: list[float | None] = []
        self.exposure: float | None = None

    def set(self, value):
        self.calls.append(value)
        self.exposure = value

    def ratio(self) -> float:
        return 1.0 if self.exposure is None else self.at_minus_5 * 2.0 ** (self.exposure + 5)


def run_sweep(room, tolerance=0.25, settle_s=0.7, **kw):
    lock = ExposureLock(room.set, tolerance, settle_s, **kw)
    lock.start(0.0)
    t, finished = 0.0, False
    while not finished and t < 30:
        t += 1 / 30
        finished = lock.update(room.ratio(), t)
    assert finished
    return lock, t


def test_candidates_all_keep_30_fps():
    assert CANDIDATES[0] == EXPOSURE_FPS_LIMIT == -5.0
    assert all(c <= -5 for c in CANDIDATES) and list(CANDIDATES) == sorted(CANDIDATES, reverse=True)
    with pytest.raises(ValueError):
        ExposureLock(lambda v: None, 0.25, 0.7, candidates=(-4.0, -5.0))


def test_locks_the_exposure_closest_to_the_plate():
    room = FakeRoom(at_minus_5=1.6)  # -5 is 1.6x too bright, -6 is 0.8x: -6 is closer
    lock, _ = run_sweep(room)
    assert lock.state == lock.LOCKED and lock.value == -6.0
    assert lock.ratio == pytest.approx(0.8)
    assert room.calls == [-5.0, -6.0] and room.exposure == -6.0  # stopped at the first value darker than the plate
    assert "locked at -6" in lock.describe()


def test_goes_back_to_the_brighter_value_when_it_was_closer():
    room = FakeRoom(at_minus_5=1.15)  # -5: 1.15x, -6: 0.575x
    lock, _ = run_sweep(room)
    assert lock.state == lock.LOCKED and lock.value == -5.0
    assert room.calls == [-5.0, -6.0, -5.0] and room.exposure == -5.0


def test_stays_on_auto_when_nothing_is_within_tolerance():
    room = FakeRoom(at_minus_5=0.62)  # a dim room: even the longest 30 fps exposure is much darker than the plate
    lock, _ = run_sweep(room)
    assert lock.state == lock.AUTO and lock.value is None
    assert room.calls == [-5.0, None] and room.exposure is None  # one probe, then auto again
    assert "0.62x" in lock.reason and "auto" in lock.describe()
    room = FakeRoom(at_minus_5=0.62)
    lock, _ = run_sweep(room, tolerance=0.4)  # a looser tolerance accepts it
    assert lock.state == lock.LOCKED and lock.value == -5.0


def test_waits_for_the_sensor_to_settle_and_averages_frames():
    room = FakeRoom(at_minus_5=0.9)
    lock = ExposureLock(room.set, 0.25, settle_s=0.7, measure_frames=5)
    lock.start(10.0)
    assert lock.sweeping
    assert not lock.update(5.0, 10.3) and not lock.update(5.0, 10.69)  # garbage during the settle time is ignored
    for i in range(4):
        assert not lock.update(0.9, 10.71 + i * 0.03)
    assert lock.update(0.9, 10.9)
    assert lock.state == lock.LOCKED and lock.ratio == pytest.approx(0.9)


def test_unmeasurable_background_ends_on_auto():
    room = FakeRoom(at_minus_5=1.0)
    lock = ExposureLock(room.set, 0.25, settle_s=0.1)
    lock.start(0.0)
    t = 0.0
    while lock.sweeping and t < 30:
        t += 0.1
        lock.update(None, t)
    assert lock.state == lock.AUTO and room.exposure is None and "could not be measured" in lock.reason


def test_recapture_returns_to_auto_then_locks_again_or_restores():
    room = FakeRoom(at_minus_5=1.6)
    lock, t = run_sweep(room)
    lock.release()                      # B pressed: capture needs auto exposure
    assert lock.state == lock.IDLE and room.exposure is None
    lock.restore()                      # capture aborted: the old plate and its lock come back
    assert lock.state == lock.LOCKED and lock.value == -6.0 and room.exposure == -6.0
    lock.release()
    lock.start(t)                       # capture succeeded: a fresh sweep for the new plate
    assert lock.sweeping and room.exposure == -5.0
    lock.restore()                      # nothing to restore once a new sweep has begun
    assert lock.sweeping
    never = ExposureLock(room.set, 0.25, 0.7)
    never.release(); never.restore()    # no lock yet: both are no-ops
    assert never.state == never.IDLE


# -- the safety net must cover the lock path -----------------------------------------------------

class FakeCap:
    def __init__(self):
        self.props, self.released = [], False

    def set(self, prop, value):
        self.props.append((prop, value))
        return True

    def get(self, prop):
        return 0.0

    def grab(self):
        time.sleep(0.002)
        return True

    def retrieve(self):
        return True, np.zeros((4, 4, 3), np.uint8)

    def release(self):
        self.released = True


def fake_camera() -> Camera:
    cam = Camera.__new__(Camera)  # skip opening a real device
    cam.index, cam._manual_exposure, cam._cap = 99, False, FakeCap()
    cam.width, cam.height, cam.fps = 4, 4, 30.0
    return cam


def is_auto(cap: FakeCap) -> bool:
    modes = [v for p, v in cap.props if p == cv2.CAP_PROP_AUTO_EXPOSURE]
    return bool(modes) and modes[-1] == 0.75


def wait_for(condition, timeout=2.0):
    end = time.time() + timeout
    while time.time() < end and not condition():
        time.sleep(0.01)
    return condition()


def test_lock_through_the_capture_thread_is_tracked_and_restored_by_the_safety_net():
    raw = fake_camera()
    cam = ThreadedCamera(raw)
    try:
        lock = ExposureLock(lambda v: cam.request_auto_exposure() if v is None else cam.request_exposure(v), 0.25, 0.0)
        lock.start(0.0)
        assert wait_for(lambda: (cv2.CAP_PROP_EXPOSURE, -5.0) in raw._cap.props)
        assert raw in camera_mod._MANUAL_CAMERAS and not is_auto(raw._cap)
        # what atexit, SIGTERM/SIGBREAK and the console-close handler all call:
        assert restore_auto_exposure_all("test") == 1
        assert is_auto(raw._cap) and raw not in camera_mod._MANUAL_CAMERAS
    finally:
        cam.release()


def test_release_restores_auto_after_a_lock_and_request_auto_works():
    raw = fake_camera()
    cam = ThreadedCamera(raw)
    cam.request_exposure(-6.0)
    assert wait_for(lambda: raw._manual_exposure)
    cam.request_auto_exposure()         # B pressed while locked
    assert wait_for(lambda: is_auto(raw._cap)) and raw not in camera_mod._MANUAL_CAMERAS
    cam.request_exposure(-6.0)          # locked again for the new plate
    assert wait_for(lambda: raw._manual_exposure) and raw in camera_mod._MANUAL_CAMERAS
    cam.release()                       # main's finally: every exit path that unwinds
    assert is_auto(raw._cap) and raw._cap.released and raw not in camera_mod._MANUAL_CAMERAS
