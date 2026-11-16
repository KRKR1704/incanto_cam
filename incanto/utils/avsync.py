"""Audio-video sync measurement from claps: pure numpy, tested on synthetic signals.

A clap is a sound spike (audio) AND the moment the wrists meet (video). Comparing the two times, both on the camera's
clock, gives the audio-video offset. Convention: offset = audio time - video time of the same event. Positive: the
audio is stamped later than the picture. `audio.offset_ms` is ADDED to audio timestamps, so the fix is its negative.
"""
from __future__ import annotations

import numpy as np

FRAME_S = 1.0 / 30.0


def highpass(x: np.ndarray, rate: float, cutoff: float = 1000.0) -> np.ndarray:
    """High-pass by subtracting a moving average about one `cutoff` period wide: removes voice rumble and DC so that
    only the sharp transient of a clap is left (no scipy needed)."""
    k = max(3, int(round(rate / cutoff)))
    return x - np.convolve(x, np.ones(k) / k, mode="same")


def detect_claps_audio(samples: np.ndarray, rate: float, t0: float, min_rise_db: float = 14.0,
                       floor_dbfs: float = -38.0, refractory_s: float = 0.5) -> list[float]:
    """Times (clock of `t0`, the time of sample 0) of sharp sound spikes. A spike is a 2 ms high-passed energy peak that
    stands `min_rise_db` above the median of the preceding 0.4 s and above `floor_dbfs`, and is the loudest within
    +-0.15 s. The time reported is the peak sample of the rectified high-passed signal."""
    if len(samples) < rate:
        return []
    hp = highpass(np.asarray(samples, np.float64), rate)
    block = max(1, int(rate * 0.002))
    n = len(hp) // block
    mag = np.abs(hp[:n * block]).reshape(n, block)
    env = 20 * np.log10(np.maximum(mag.max(axis=1), 1e-6))      # dBFS of each 2 ms block's peak
    ctx = int(0.4 / 0.002)
    hold = int(0.15 / 0.002)
    found: list[float] = []
    last = -1e9
    for i in np.argsort(-env):                                   # loudest first, so refractory keeps the best
        if env[i] < floor_dbfs:
            break
        lo = max(0, i - ctx)
        if i - lo < 20:
            continue
        if env[i] - float(np.median(env[lo:i - 5])) < min_rise_db:
            continue
        w = slice(max(0, i - hold), min(n, i + hold + 1))
        if env[i] < env[w].max():
            continue
        t = t0 + (i * block + int(np.argmax(mag[i]))) / rate
        if all(abs(t - f) >= refractory_s for f in found):
            found.append(t)
    return sorted(found)


def contact_time(times: np.ndarray, dist: np.ndarray, contact_dist: float = 0.04,
                 max_gap_s: float = 0.2) -> tuple[float, str] | None:
    """When the wrists meet, from per-frame wrist distance (NaN where a hand was not found).

    The closest tracked frame is refined to sub-frame precision with a parabola. Trackers often lose a hand at the
    moment of contact (the hands overlap): if the closest frame is followed by lost frames, the closing speed over
    its last frames is extrapolated to `contact_dist`. Returns (time, how) or None when no approach was seen."""
    times, dist = np.asarray(times, float), np.asarray(dist, float)
    ok = ~np.isnan(dist)
    if ok.sum() < 3:
        return None
    i = int(np.nanargmin(dist))
    after = np.arange(i + 1, len(dist))
    lost_after = len(after) > 0 and np.isnan(dist[after[0]]) and times[after[0]] - times[i] <= max_gap_s
    if lost_after and i >= 2 and ok[i - 1] and ok[i - 2]:
        v = (dist[i - 2] - dist[i]) / (times[i] - times[i - 2])        # closing speed, per second
        if v > 0 and dist[i] > contact_dist:
            t = times[i] + min((dist[i] - contact_dist) / v, max_gap_s)
            return float(t), "extrapolated (hands lost at contact)"
        return float(times[i]), "last tracked frame (hands lost at contact)"
    if 0 < i < len(dist) - 1 and ok[i - 1] and ok[i + 1]:
        y0, y1, y2 = dist[i - 1], dist[i], dist[i + 1]
        denom = y0 - 2 * y1 + y2
        shift = 0.5 * (y0 - y2) / denom if denom > 1e-12 else 0.0
        shift = float(np.clip(shift, -1, 1))
        step = times[i + 1] - times[i] if shift >= 0 else times[i] - times[i - 1]
        return float(times[i] + shift * step), "closest frame (parabola)"
    return float(times[i]), "closest frame"


def summarize(offsets_ms: list[float]) -> dict:
    """Median, spread (median absolute deviation), and whether the median is within one frame."""
    a = np.asarray(offsets_ms, float)
    if len(a) == 0:
        return {"n": 0}
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    return {"n": len(a), "median_ms": med, "mad_ms": mad, "min_ms": float(a.min()), "max_ms": float(a.max()),
            "within_one_frame": abs(med) <= 1000 * FRAME_S, "suggested_audio_offset_ms": -med}
