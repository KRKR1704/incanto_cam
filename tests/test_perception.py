"""Shape/type checks for the hand tracker and segmenter on a blank image (needs models/)."""
import numpy as np
import pytest

from incanto.config import load_config
from incanto.perception.hands import HandTracker
from incanto.perception.segmenter import Segmenter
from incanto.utils.imaging import downscale

CFG = load_config()
HANDS_MODEL = CFG.models.resolve("hand_landmarker")
SEG_MODEL = CFG.models.resolve("selfie_segmenter")

needs_models = pytest.mark.skipif(
    not (HANDS_MODEL.exists() and SEG_MODEL.exists()),
    reason="run scripts/download_models.py first",
)


def blank(width=1280, height=720):
    return np.full((height, width, 3), 127, np.uint8)


def test_downscale_keeps_aspect_and_never_upscales():
    small = downscale(blank(), 640)
    assert small.shape == (360, 640, 3)
    assert downscale(small, 1280).shape == small.shape


@needs_models
def test_hand_tracker_on_blank_image_returns_empty_list():
    tracker = HandTracker(HANDS_MODEL, num_hands=2)
    try:
        for _ in range(3):  # several frames: timestamps must keep increasing without error
            assert tracker.detect(downscale(blank(), 640)) == []
    finally:
        tracker.close()


@needs_models
def test_segmenter_returns_full_res_float_mask_and_is_near_empty_on_blank_image():
    seg = Segmenter(SEG_MODEL, (1280, 720))
    try:
        for _ in range(3):
            mask = seg.segment(downscale(blank(), 640))
        assert mask.shape == (720, 1280)
        assert mask.dtype == np.float32
        assert 0.0 <= mask.min() and mask.max() <= 1.0
        assert mask.mean() < 0.1
    finally:
        seg.close()


@needs_models
def test_segmenter_every_n_reuses_last_mask():
    seg = Segmenter(SEG_MODEL, (1280, 720), every_n=3)
    try:
        first = seg.segment(downscale(blank(), 640))
        second = seg.segment(downscale(blank(), 640))
        assert second is first
    finally:
        seg.close()


@needs_models
def test_hands_worker_publishes_latest_result_without_blocking():
    import time

    from incanto.perception.hands import HandsWorker

    worker = HandsWorker(HandTracker(HANDS_MODEL, num_hands=2))
    try:
        assert worker.latest().seq == 0  # nothing yet, and latest() returned immediately
        stamp = time.perf_counter()
        for _ in range(5):  # burst of submits: stale frames are overwritten, never queued
            worker.submit(downscale(blank(), 640), stamp)
        deadline = time.perf_counter() + 5
        while worker.latest().seq == 0 and time.perf_counter() < deadline:
            time.sleep(0.01)
        result = worker.latest()
        assert result.seq >= 1
        assert result.hands == []
        assert result.frame_time == stamp
        assert result.seq <= 5
    finally:
        worker.close()


@needs_models
def test_segmenter_attack_release_grows_instantly_and_shrinks_slowly():
    class Fake:
        def __init__(self):
            self.value = 0.0

        def segment_for_video(self, image, ts):
            class Mask:
                def __init__(self, v):
                    self.v = v

                def numpy_view(self):
                    return np.full((36, 64), self.v, np.float32)

            class Result:
                pass

            r = Result()
            r.confidence_masks = [Mask(self.value)]
            return r

        def close(self):
            pass

    seg = Segmenter(SEG_MODEL, (64, 36), blur_px=0, temporal_mode="attack_release", release_decay=0.5)
    fake = Fake()
    seg._segmenter.close()
    seg._segmenter = fake
    frame = np.zeros((36, 64, 3), np.uint8)
    fake.value = 1.0
    assert seg.segment(frame).mean() == pytest.approx(1.0)   # attack: instant
    fake.value = 0.0
    assert seg.segment(frame).mean() == pytest.approx(0.5)   # release: previous * decay
    assert seg.segment(frame).mean() == pytest.approx(0.25)
    seg.temporal_mode = "symmetric"
    fake.value = 1.0
    assert seg.segment(frame).mean() < 0.9                    # symmetric blend does not jump
