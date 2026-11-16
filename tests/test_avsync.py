"""Clap detection in audio and wrist-contact time in video, on synthetic signals."""
import numpy as np
import pytest

from incanto.utils.avsync import contact_time, detect_claps_audio, summarize

RATE = 44100


def audio_with_claps(times, seconds=8.0, noise=0.003, seed=0, voice=True):
    rng = np.random.default_rng(seed)
    n = int(seconds * RATE)
    x = rng.normal(0, noise, n)
    if voice:                                               # low rumble / speech-like hum must not count
        x += 0.05 * np.sin(2 * np.pi * 140 * np.arange(n) / RATE)
    for t in times:
        i = int(t * RATE)
        k = np.arange(300)
        x[i:i + 300] += 0.6 * np.exp(-k / 40.0) * rng.choice([-1, 1], 300)   # a sharp click
    return x.astype(np.float32)


def test_claps_are_found_at_their_times_and_hum_is_ignored():
    truth = [1.0, 2.5, 4.2, 6.0]
    found = detect_claps_audio(audio_with_claps(truth), RATE, t0=100.0)
    assert len(found) == 4
    for f, t in zip(found, truth):
        assert f - 100.0 == pytest.approx(t, abs=0.004)     # within a few ms of the click


def test_quiet_noise_and_speech_alone_give_no_claps():
    assert detect_claps_audio(audio_with_claps([]), RATE, 0.0) == []
    assert detect_claps_audio(np.zeros(RATE * 3, np.float32), RATE, 0.0) == []
    assert detect_claps_audio(np.zeros(100, np.float32), RATE, 0.0) == []


def test_double_click_within_the_refractory_period_counts_once():
    found = detect_claps_audio(audio_with_claps([2.0, 2.2]), RATE, 0.0)
    assert len(found) == 1


def approach(t_contact, fps=30.0, v=1.2, start=0.5, lost_after=None, jitter=0.0):
    """Wrist distance falling at v (frame widths per second) to ~0.02 at t_contact; NaN after `lost_after` seconds."""
    times = np.arange(t_contact - start, t_contact + 0.4, 1 / fps) + jitter
    d = np.maximum(0.02, v * np.abs(t_contact - times)) + np.where(times > t_contact, 0.0, 0.0)
    d = np.where(times > t_contact, 0.02 + 0.5 * (times - t_contact), d)   # hands part again after the clap
    if lost_after is not None:
        d[times > t_contact - lost_after] = np.nan
    return times, d


def test_contact_time_recovers_the_moment_the_wrists_meet_between_frames():
    for true in (10.0, 10.0111, 10.0222):
        times, d = approach(true)
        t, how = contact_time(times, d)
        assert t == pytest.approx(true, abs=0.012)           # within about a third of a frame


def test_contact_time_extrapolates_when_the_tracker_loses_the_hands_at_contact():
    times, d = approach(10.0, lost_after=0.045)              # the last ~1.5 frames before contact are lost
    t, how = contact_time(times, d)
    assert "extrapolated" in how or "lost" in how
    assert t == pytest.approx(10.0, abs=0.05)


def test_contact_time_needs_an_approach():
    assert contact_time(np.arange(5.0), np.full(5, np.nan)) is None
    assert contact_time(np.arange(2.0), np.array([0.3, 0.2])) is None


def test_summary_reports_median_spread_and_the_suggested_offset():
    s = summarize([60.0, 70.0, 65.0, 62.0, 200.0])           # one bad clap does not move the median
    assert s["median_ms"] == 65.0 and s["suggested_audio_offset_ms"] == -65.0 and not s["within_one_frame"]
    assert s["mad_ms"] == 5.0
    ok = summarize([5.0, -10.0, 12.0])
    assert ok["within_one_frame"] and summarize([])["n"] == 0
