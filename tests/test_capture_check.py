"""Plate capture person check: still segmenter blobs are furniture, moving ones are people; forced
capture; the abort debug image; the half-size preview drawing."""
import cv2
import numpy as np
import pytest

from incanto.config import PlateConfig, PreviewConfig
from incanto.plate.clean_plate import (ABORT_IMAGE_PREFIX, MOTION_WINDOW_S, PersonMotion, PlateCapture,
                                       save_abort_image)
from incanto.render import overlay

W, H = 320, 180
FPS = 30
CHAIR = (slice(50, 170), slice(120, 180))  # 60 x 120 px = 12.5% of the frame, like the real false positive


def room(rng, gain=1.0):
    """A textured room with a chair, plus sensor noise (so 'static' is not 'bit-identical')."""
    y, x = np.mgrid[0:H, 0:W]
    base = 110 + 40 * np.sin(x / 35.0) * np.cos(y / 25.0)
    img = np.stack([base, base * 0.9 + 10, base * 1.1 - 10], axis=2)
    img[CHAIR] = (40, 40, 45)
    img[60:90, 135:165] = (210, 210, 215)  # the headrest pillow
    return np.clip(img * gain + rng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)


def chair_mask(rng=None):
    m = np.zeros((H, W), np.float32)
    m[CHAIR] = 1.0
    if rng is not None:  # the segmenter's edge is never perfectly steady
        m[CHAIR[0], 118:120] = rng.random((120, 2)) > 0.5
    return m


def add_person(frame, mask, x, y0=30, y1=170, w=40):
    frame[y0:y1, x:x + w] = (90, 120, 200)
    frame[y0:y0 + 30, x + 8:x + w - 8] = (140, 170, 220)
    mask[y0:y1, x:x + w] = 1.0


def run(scene, seconds=8.0, countdown_s=5.0, settle_s=1.0, frames=30, force=False, **kw):
    """Drive a PlateCapture with scene(i, t, rng) -> (frame, mask) at 30 fps."""
    cap = PlateCapture(frames=frames, countdown_s=countdown_s, settle_s=settle_s, max_coverage=0.02, **kw)
    cap.start(0.0, force=force)
    rng = np.random.default_rng(1)
    i = 0
    while i < seconds * FPS and cap.state not in (cap.DONE, cap.ABORTED):
        i += 1
        t = i / FPS
        frame, mask = scene(i, t, rng)
        cap.update(frame, mask, t)
    return cap


def test_defaults():
    assert PlateConfig().static_blob_max_motion == 0.02 and PlateConfig().max_person_coverage == 0.02
    assert PlateConfig().debug_dir == "docs/evidence/phase-3"
    assert PreviewConfig().scale == 0.5


def test_still_chair_the_segmenter_calls_a_person_does_not_abort():
    """The live failure: 12.8%, 12.0%, 12.6% 'person' with nobody in the frame."""
    cap = run(lambda i, t, rng: (room(rng), chair_mask(rng)))
    assert cap.state == cap.DONE
    assert cap.raw_coverage == pytest.approx(0.125, abs=0.01)      # what the old check saw: 6x the 2% limit
    assert cap.static_coverage == pytest.approx(0.125, abs=0.01)   # all of it recognised as furniture
    assert cap.peak_coverage == 0.0                                # nothing counted as a person
    assert cap.abort_image() is None


def test_exposure_settling_does_not_make_the_chair_a_person():
    # auto exposure brightening the empty room by 25% over the countdown and settle time
    cap = run(lambda i, t, rng: (room(rng, gain=1.0 + 0.25 * min(t / 6.0, 1.0)), chair_mask(rng)))
    assert cap.state == cap.DONE and cap.peak_coverage == 0.0


def test_chair_still_swivelling_early_in_the_countdown_is_fine_once_it_stops():
    def scene(i, t, rng):
        frame, mask = room(rng), chair_mask()
        if t < 3.0:  # I just got up: the chair rocks for a few seconds
            shift = int(6 * np.sin(t * 8))
            frame, mask = np.roll(frame, shift, axis=1), np.roll(mask, shift, axis=1)
        return frame, mask
    assert run(scene).state == PlateCapture.DONE


def test_person_walking_through_aborts():
    def scene(i, t, rng):
        frame, mask = room(rng), np.zeros((H, W), np.float32)
        add_person(frame, mask, x=10 + (i * 2) % 250)
        return frame, mask
    cap = run(scene)
    assert cap.state == cap.ABORTED and "Person still in frame" in cap.abort_reason
    assert "Shift+B" in cap.abort_reason
    assert cap.peak_coverage > 0.08 and cap.pop_events()[-1] == "error"


def test_person_standing_in_place_still_aborts():
    """Nobody holds as still as a chair: breathing and small shifts move a few pixels of the blob edge
    and the face every frame."""
    def scene(i, t, rng):
        frame, mask = room(rng), np.zeros((H, W), np.float32)
        add_person(frame, mask, x=230 + int(round(2.5 * np.sin(t * 2.0))), y0=30 + int(round(1.5 * np.sin(t * 3.1))))
        return frame, mask
    cap = run(scene)
    assert cap.state == cap.ABORTED


def test_chair_is_ignored_but_a_person_beside_it_is_counted():
    def scene(i, t, rng):
        frame, mask = room(rng), chair_mask(rng)
        add_person(frame, mask, x=230 + int(round(3 * np.sin(t * 2.0))))
        return frame, mask
    cap = run(scene)
    assert cap.state == cap.ABORTED
    person_area = 40 * 140 / (W * H)
    assert cap.peak_coverage == pytest.approx(person_area, abs=0.02)  # the person only, not person + chair (22%)


def test_person_who_leaves_before_the_capture_is_fine():
    def scene(i, t, rng):
        frame, mask = room(rng), chair_mask(rng)
        if t < 3.5:  # walking out during the countdown
            add_person(frame, mask, x=min(280, 200 + int(t * 30)))
        return frame, mask
    cap = run(scene)
    assert cap.state == cap.DONE and cap.static_coverage > 0.1


def test_person_sitting_in_the_chair_counts_as_one_moving_blob():
    def scene(i, t, rng):
        frame, mask = room(rng), chair_mask()
        add_person(frame, mask, x=130 + int(round(2 * np.sin(t * 2.5))), y0=60, w=40)  # overlaps the chair
        return frame, mask
    cap = run(scene)
    assert cap.state == cap.ABORTED and cap.peak_coverage > 0.12


def test_blob_counts_as_a_person_until_a_second_has_been_watched():
    rng = np.random.default_rng(2)
    m = PersonMotion()
    cover = [m.update(room(rng), chair_mask(), i / FPS) for i in range(1, 46)]
    early = [c for i, c in enumerate(cover, 1) if i / FPS < MOTION_WINDOW_S]
    assert min(early) > 0.1                      # not enough history yet: stay on the safe side
    assert cover[-1] == 0.0 and m.static_coverage > 0.1
    assert m.person.shape == m.static.shape == (H, W) and m.static[CHAIR].all()
    # no countdown and no settle time means nothing has been watched when capturing starts: abort
    assert run(lambda i, t, rng: (room(rng), chair_mask()), countdown_s=0.0, settle_s=0.0).state == PlateCapture.ABORTED


def test_static_check_can_be_switched_off():
    cap = run(lambda i, t, rng: (room(rng), chair_mask(rng)), max_static_motion=0.0)
    assert cap.state == cap.ABORTED  # every blob is a person again, as before


def test_shift_b_forces_the_capture_and_logs_a_warning(caplog):
    def scene(i, t, rng):
        frame, mask = room(rng), np.zeros((H, W), np.float32)
        add_person(frame, mask, x=10 + (i * 2) % 250)
        return frame, mask
    with caplog.at_level("WARNING"):
        cap = run(scene, force=True)
    assert cap.state == cap.DONE and cap.force
    assert any("FORCED" in r.message and "person check is off" in r.message for r in caplog.records)
    assert cap.result().shape == (H, W, 3)
    assert cap.raw_coverage > 0.08  # still reported
    cap.start(0.0)                 # the override does not stick
    assert cap.force is False


def test_abort_saves_a_local_only_debug_image(tmp_path):
    def scene(i, t, rng):
        frame, mask = room(rng), chair_mask(rng)
        add_person(frame, mask, x=230 + int(round(3 * np.sin(t * 2.0))))
        return frame, mask
    cap = run(scene)
    assert cap.state == cap.ABORTED
    img = cap.abort_image()
    assert img.shape == (H, W, 3) and img.dtype == np.uint8
    b, g, r = (img[100:160, 240:260, c].mean() for c in range(3))
    assert r > g + 40 and r > b                    # the person is marked red
    b, g, r = (img[100:160, 130:170, c].mean() for c in range(3))
    assert g > b + 40 and r > b + 40               # the chair is marked yellow (ignored)
    path = save_abort_image(cap, tmp_path / "evidence")
    assert path.exists() and path.name.startswith(ABORT_IMAGE_PREFIX) and path.suffix == ".png"
    assert cv2.imread(str(path)).shape == (H, W, 3)
    done = run(lambda i, t, rng: (room(rng), chair_mask(rng)))
    assert save_abort_image(done, tmp_path / "evidence") is None  # nothing to save unless aborted


def test_debug_images_are_gitignored():
    from incanto.config import PROJECT_ROOT
    rules = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "docs/evidence/**/*.png" in rules and "*.png" in rules


# -- half-size preview ---------------------------------------------------------------------------

class FakeResult:
    def __init__(self):
        self.seg = np.zeros((360, 640), np.float32)
        self.seg[100:300, 250:400] = 1.0
        self.grown = cv2.dilate(self.seg, np.ones((15, 15), np.uint8))
        self.hull = np.zeros_like(self.seg)
        self.diff = np.zeros_like(self.seg)
        self.zone = cv2.dilate(self.seg, np.ones((71, 71), np.uint8))


@pytest.mark.parametrize("size", [(1280, 720), (640, 360)])
def test_overlays_draw_on_a_preview_of_any_size(size):
    w, h = size
    out = np.full((720, 1280, 3), 90, np.uint8)
    preview = cv2.resize(out, size, interpolation=cv2.INTER_AREA) if size != (1280, 720) else out.copy()
    overlay.draw_vanish_sources(preview, FakeResult())
    overlay.draw_debug(preview, np.zeros((720, 1280), np.float32), [], {"hands": 15.0, "segment": 8.0})
    overlay.draw_fps(preview, 29.9)
    overlay.draw_status(preview, "VANISH ON | plate: ready")
    overlay.draw_status(preview, "Lighting changed - press B to refresh plate", row=1)
    overlay.draw_banner(preview, "Plate doesn't match the room - press B", warning=True)
    assert preview.shape == (h, w, 3)
    k = w / 1280
    seg_area = preview[round(120 * 2 * k):round(280 * 2 * k), round(270 * 2 * k):round(380 * 2 * k)]
    assert seg_area[..., 1].mean() > seg_area[..., 2].mean() + 30  # segmenter tinted green at the scaled position
    assert (preview[: round(50 * k), : round(250 * k)] != 90).any()  # the FPS text is inside the frame


def test_leak_view_maps_the_roi_onto_a_smaller_preview():
    out = np.full((720, 1280, 3), 90, np.uint8)
    roi = (400, 200, 800, 600)
    plate_roi = np.full((400, 400, 3), 90, np.uint8)
    out[300:304, 500:700] = 150  # a thin leak: 4 px tall at full size
    for size in ((1280, 720), (640, 360)):
        preview = cv2.resize(out, size, interpolation=cv2.INTER_AREA)
        before = preview.copy()
        overlay.draw_leak(preview, out, plate_roi, roi)
        k = size[0] / 1280
        leak = preview[round(300 * k):round(304 * k), round(520 * k):round(680 * k)]
        assert leak[..., 2].mean() > 200 and leak[..., 1].mean() < 60       # shown red, even at half size
        clean = preview[round(450 * k):round(550 * k), round(450 * k):round(750 * k)]
        assert (clean == before[round(450 * k):round(550 * k), round(450 * k):round(750 * k)]).all()
