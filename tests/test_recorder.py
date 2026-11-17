"""Recorder (own thread, timestamp sidecar), clip loading, replay source and replayed audio."""
import json
import time
import types

import numpy as np
import pytest

from incanto.config import ConfigError, load_config
from incanto.io.audio import AudioRing
from incanto.io.recorder import (ClipRecorder, RecordingError, read_frame_times, write_frame_times)
from incanto.io.replay import Clip, ClipError, EndOfClip, ReplayAudio, ReplaySource

SIZE = (64, 48)


def frame(k):
    img = np.zeros((SIZE[1], SIZE[0], 3), np.uint8)
    img[:, :, 1] = (k * 9) % 256
    img[k % SIZE[1], :, 2] = 255
    return img


def fake_mic(rate=8000):
    return types.SimpleNamespace(ring=AudioRing(rate, 5.0), samplerate=rate, device=types.SimpleNamespace(name="Test mic"))


def record(tmp_path, n=12, codec="lossless", mic=None, **kw):
    rec = ClipRecorder(tmp_path, SIZE, 30, codec, mic, name="clip", **kw)
    rec.start()
    for k in range(n):
        rec.submit(frame(k), 1000.0 + k * 0.0333 + (0.004 if k == 5 else 0.0))   # one late frame
    return rec


def test_sidecar_round_trips_and_validates(tmp_path):
    times = [10.0, 10.033, 10.0665, 10.1]
    write_frame_times(tmp_path / "f.csv", times)
    assert read_frame_times(tmp_path / "f.csv").tolist() == pytest.approx(times)
    (tmp_path / "bad.csv").write_text("a,b\n0,1\n")
    with pytest.raises(ValueError):
        read_frame_times(tmp_path / "bad.csv")
    (tmp_path / "gap.csv").write_text("frame,time_s\n0,1.0\n2,1.1\n")
    with pytest.raises(ValueError):
        read_frame_times(tmp_path / "gap.csv")


@pytest.mark.parametrize("codec", ["lossless", "mjpg"])
def test_recording_writes_video_sidecar_meta_and_audio(tmp_path, codec):
    mic = fake_mic()
    rec = ClipRecorder(tmp_path, SIZE, 30, codec, mic, name="clip")
    path = rec.start()
    for k in range(12):
        mic.ring.write(1000.0 + k * 0.0333, np.full(266, 0.1 * np.sin(k), np.float32))
        rec.submit(frame(k), 1000.0 + k * 0.0333)
        time.sleep(0.01)
    summary = rec.stop()
    assert summary.frames == 12 and summary.dropped == 0 and path == tmp_path / "clip"
    meta = json.loads((path / "meta.json").read_text())
    assert meta["frames"] == 12 and meta["codec"] == codec and meta["clock"] == "perf_counter"
    assert meta["audio"]["start_time"] == pytest.approx(1000.0) and meta["audio"]["samplerate"] == 8000
    assert meta["audio"]["samples"] == 12 * 266
    clip = Clip.load(path)
    assert clip.n_frames == 12 and clip.frame_times[0] == pytest.approx(1000.0)
    assert clip.audio().size == 12 * 266 and clip.has_audio
    frames = list(clip.frames())
    assert [i for i, _, _ in frames] == list(range(12))
    if codec == "lossless":
        assert all(np.array_equal(f, frame(i)) for i, _, f in frames)        # bit-exact
    assert not (path / "audio.wav").stat().st_size == 0


def test_frame_times_come_from_the_sidecar_not_the_nominal_fps(tmp_path):
    rec = record(tmp_path, n=10)
    summary = rec.stop()
    clip = Clip.load(summary.path)
    gaps = np.diff(clip.frame_times)
    assert gaps.max() > gaps.min() * 1.1 or gaps.std() > 0       # real, uneven capture times are kept
    assert clip.frame_times[5] == pytest.approx(1000.0 + 5 * 0.0333 + 0.004)
    assert clip.nominal_fps == 30


def test_no_microphone_means_no_audio_file(tmp_path):
    summary = record(tmp_path, n=3, mic=None).stop()
    clip = Clip.load(summary.path)
    assert not clip.has_audio and clip.audio().size == 0 and not (summary.path / "audio.wav").exists()
    assert json.loads((summary.path / "meta.json").read_text())["audio"] is None


def test_submit_never_blocks_and_drops_are_counted_when_the_encoder_is_slow(tmp_path):
    rec = ClipRecorder(tmp_path, SIZE, 30, "mjpg", None, queue_frames=2, name="slow")
    rec.start()
    real = rec._writer

    class SlowWriter:  # cv2.VideoWriter's methods are read-only, so wrap the whole object
        def write(self, f):
            time.sleep(0.15)
            real.write(f)

        def release(self):
            real.release()

    rec._writer = SlowWriter()
    t0 = time.perf_counter()
    for k in range(20):
        rec.submit(frame(k), 50.0 + k * 0.033)
    assert time.perf_counter() - t0 < 0.5                        # 20 submits did not wait for 20 slow encodes
    summary = rec.stop()
    assert summary.dropped > 0 and summary.frames + summary.dropped == 20
    assert Clip.load(summary.path).n_frames == summary.frames    # the sidecar lists only frames that were written


def test_wrong_size_frame_is_rejected_and_two_recordings_in_one_second_do_not_collide(tmp_path):
    rec = ClipRecorder(tmp_path, SIZE, 30, "mjpg", None, name="same")
    rec.start()
    with pytest.raises(ValueError):
        rec.submit(np.zeros((10, 10, 3), np.uint8), 0.0)
    rec.stop()
    second = ClipRecorder(tmp_path, SIZE, 30, "mjpg", None, name="same")
    assert second.start().name == "same-2"
    second.stop()


def test_refuses_to_start_without_free_disk(tmp_path):
    with pytest.raises(RecordingError, match="free"):
        ClipRecorder(tmp_path, SIZE, 30, "mjpg", None, min_free_gb=1e9).start()


def test_recording_config_validation(tmp_path):
    f = tmp_path / "c.yaml"
    f.write_text("recording:\n  codec: h264\n")
    with pytest.raises(ConfigError):
        load_config(f)
    assert load_config().recording.codec == "lossless"


# -- replay --------------------------------------------------------------------------------------
def make_clip(tmp_path, n=8, with_audio=True):
    mic = fake_mic() if with_audio else None
    rec = ClipRecorder(tmp_path, SIZE, 30, "lossless", mic, name="c")
    rec.start()
    for k in range(n):
        if mic:
            mic.ring.write(500.0 + k * 0.05, np.full(400, k / 10.0, np.float32))   # 0.05 s = 400 samples
        rec.submit(frame(k), 500.0 + k * 0.05)
        time.sleep(0.01)
    return Clip.load(rec.stop().path)


def test_replay_fast_returns_every_frame_with_recorded_times_then_ends(tmp_path):
    clip = make_clip(tmp_path)
    src = ReplaySource(clip, realtime=False)
    times = []
    for k in range(8):
        f = src.read()
        assert np.array_equal(f, frame(k))
        times.append(src.frame_time)
    assert times == pytest.approx([500.0 + k * 0.05 for k in range(8)])
    with pytest.raises(EndOfClip):
        src.read()


def test_replay_realtime_is_paced_by_the_recorded_times(tmp_path):
    clip = make_clip(tmp_path)                              # 0.35 s long
    src = ReplaySource(clip, realtime=True)
    t0 = time.perf_counter()
    for _ in range(8):
        src.read()
    assert 0.3 < time.perf_counter() - t0 < 0.8
    fast = ReplaySource(clip, realtime=False)
    t0 = time.perf_counter()
    for _ in range(8):
        fast.read()
    assert time.perf_counter() - t0 < 0.2


def test_replay_timestamps_are_deterministic_and_strictly_increasing(tmp_path):
    clip = make_clip(tmp_path)
    runs = []
    for _ in range(2):
        src = ReplaySource(clip, realtime=False)
        stamps = []
        for _ in range(8):
            src.read()
            stamps.append(src.timestamp_ms())
        runs.append(stamps)
    assert runs[0] == runs[1] and runs[0] == sorted(set(runs[0])) and runs[0][0] == 0
    # two frames closer than a millisecond still get increasing stamps
    src = ReplaySource(clip, realtime=False)
    src.frame_time = clip.first_time
    a = src.timestamp_ms()
    b = src.timestamp_ms()
    assert b == a + 1


def test_replay_audio_is_fed_on_the_recorded_clock_with_the_offset(tmp_path):
    clip = make_clip(tmp_path)
    feed = ReplayAudio(clip, offset_ms=0.0)
    assert feed.available
    feed.feed_until(500.0 + 0.10)                           # two blocks' worth, up to the 0.10 s mark
    assert feed.ring.total == int(0.10 * 8000)
    out, _, t0, _ = feed.ring.read_from(0)
    assert t0 == pytest.approx(500.0, abs=1e-3) and out[0] == pytest.approx(0.0) and out[-1] == pytest.approx(0.1, abs=1e-3)
    feed.feed_until(10_000.0)                               # far past the end: everything, once
    assert feed.ring.total == clip.audio().size
    delayed = ReplayAudio(clip, offset_ms=100.0)            # audio.offset_ms +100: audio arrives 0.1 s later
    delayed.feed_until(500.0 + 0.10)
    assert delayed.ring.total == 0
    delayed.feed_until(500.0 + 0.20)
    assert delayed.ring.total == int(0.10 * 8000)
    _, _, t_read, _ = delayed.ring.read_from(0)
    assert t_read == pytest.approx(500.0 + 0.1)             # and is stamped 0.1 s later too


def test_clip_without_audio_replays_silently(tmp_path):
    clip = make_clip(tmp_path, with_audio=False)
    feed = ReplayAudio(clip)
    assert not feed.available
    feed.feed_until(1e9)
    assert feed.ring.total == 0


def test_missing_or_broken_clip_raises_a_clear_error(tmp_path):
    with pytest.raises(ClipError, match="not found"):
        Clip.load(tmp_path / "nope")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ClipError, match="meta.json"):
        Clip.load(tmp_path / "empty")


# -- unique fps ----------------------------------------------------------------------------------
from incanto.utils.fps import UniqueFps  # noqa: E402


def textured(k):
    rng = np.random.default_rng(k)
    return rng.integers(0, 255, (54, 96, 3), dtype=np.uint8)


def run_unique(repeat_every, seconds, fps=30.0, start=0.0, mon=None):
    mon = mon or UniqueFps()
    t, k, unique = start, 0, 0
    for n in range(int(seconds * fps)):
        if repeat_every == 0 or n % repeat_every != 1:
            k += 1                                   # a new picture
        unique += mon.update(textured(k), t)
        t += 1 / fps
    return mon, unique


def test_unique_fps_counts_repeated_frames():
    mon, unique = run_unique(repeat_every=2, seconds=6)       # every second frame is a repeat: 15 unique fps
    assert unique == pytest.approx(90, abs=1)
    assert mon.interval_fps(6.0) == pytest.approx(15.0, abs=0.3)
    mon2, unique2 = run_unique(repeat_every=0, seconds=6)      # a real 30 fps camera
    assert unique2 == 180 and mon2.interval_fps(6.0) == pytest.approx(30.0, abs=0.5)


def test_unique_fps_warns_after_ten_seconds_under_twenty_and_clears_with_hysteresis():
    mon, _ = run_unique(2, 9.0)
    assert not mon.warning                                     # not a full 10 s window yet
    mon, _ = run_unique(2, 4.0, start=9.0, mon=mon)
    assert mon.warning and mon.window_fps() == pytest.approx(15.0, abs=0.5)
    mon, _ = run_unique(0, 3.0, start=13.0, mon=mon)           # light is back: 30 unique fps, the window recovers gradually
    assert mon.warning                                         # 7 s at 15 + 3 s at 30 is still under 22
    mon, _ = run_unique(0, 10.0, start=16.0, mon=mon)
    assert not mon.warning


def test_unique_fps_does_not_warn_at_a_healthy_rate_and_noise_alone_is_new():
    mon, _ = run_unique(0, 15.0)
    assert not mon.warning
    still = np.full((54, 96, 3), 90, np.uint8)
    m = UniqueFps()
    assert m.update(still, 0.0) and not m.update(still.copy(), 0.03)       # an identical frame is a repeat
    noisy = np.clip(still.astype(int) + np.random.default_rng(1).normal(0, 2, still.shape), 0, 255).astype(np.uint8)
    assert m.update(noisy, 0.06)                                           # sensor noise alone still counts as new
