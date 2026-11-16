"""Replay a recorded clip as if it were the live camera and microphone.

`Clip` loads a recordings/<clip>/ folder. `ReplaySource` looks like a camera to the main loop (width, height,
read(), frame_time) but returns the recorded frames, paced by their recorded times (`realtime`) or as fast as the
pipeline can take them (`fast`). `ReplayAudio` feeds the recorded audio into an AudioRing on the same clock, so
audio consumers see the same thing they would live. Everything is driven by the recorded times, never by the
container's nominal fps.
"""
from __future__ import annotations

import json
import time
import wave
from pathlib import Path

import cv2
import numpy as np

from incanto.io.audio import AudioRing
from incanto.io.recorder import FRAMES_FILE, META_FILE, VIDEO_FILE, read_frame_times


class ClipError(RuntimeError):
    """The clip folder is missing or unusable."""


class EndOfClip(Exception):
    """Raised by ReplaySource.read() after the last frame."""


class Clip:
    def __init__(self, path: Path, meta: dict, frame_times: np.ndarray):
        self.path, self.meta, self.frame_times = path, meta, frame_times
        self.width, self.height = int(meta["width"]), int(meta["height"])
        self.nominal_fps = float(meta.get("nominal_fps", 30.0))
        self._audio: np.ndarray | None = None

    @classmethod
    def load(cls, path: str | Path) -> "Clip":
        path = Path(path)
        if not path.is_dir():
            raise ClipError(f"clip folder not found: {path}")
        for name in (META_FILE, FRAMES_FILE, VIDEO_FILE):
            if not (path / name).exists():
                raise ClipError(f"{path} has no {name} (is it a folder written by the recorder?)")
        try:
            meta = json.loads((path / META_FILE).read_text(encoding="utf-8"))
            times = read_frame_times(path / FRAMES_FILE)
        except (ValueError, KeyError) as exc:
            raise ClipError(f"{path}: unreadable clip metadata ({exc})") from exc
        if len(times) == 0:
            raise ClipError(f"{path} contains no frames")
        return cls(path, meta, times)

    @property
    def n_frames(self) -> int:
        return len(self.frame_times)

    @property
    def first_time(self) -> float:
        return float(self.frame_times[0])

    @property
    def duration(self) -> float:
        return float(self.frame_times[-1] - self.frame_times[0])

    @property
    def has_audio(self) -> bool:
        return bool(self.meta.get("audio")) and (self.path / self.meta["audio"]["file"]).exists()

    @property
    def samplerate(self) -> int:
        return int(self.meta["audio"]["samplerate"])

    @property
    def audio_start_time(self) -> float:
        """Camera-clock time of the first audio sample (raw: audio.offset_ms is not applied)."""
        return float(self.meta["audio"]["start_time"])

    def audio(self) -> np.ndarray:
        """The whole recording as mono float32 (empty if there is none)."""
        if self._audio is None:
            if not self.has_audio:
                self._audio = np.zeros(0, np.float32)
            else:
                with wave.open(str(self.path / self.meta["audio"]["file"]), "rb") as w:
                    raw = w.readframes(w.getnframes())
                self._audio = np.frombuffer(raw, "<i2").astype(np.float32) / 32767.0
        return self._audio

    def frames(self, first: int = 0, last: int | None = None):
        """Yield (index, capture time, BGR frame) in order, for frames first..last-1 (default: all). Seeking to
        `first` is exact for the lossless codec and for MJPG (every frame is a key frame)."""
        last = self.n_frames if last is None else min(last, self.n_frames)
        cap = cv2.VideoCapture(str(self.path / VIDEO_FILE))
        try:
            if first > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, first)
            for i in range(first, last):
                ok, frame = cap.read()
                if not ok:
                    raise ClipError(f"{self.path}: video ends at frame {i} but the sidecar lists {self.n_frames}")
                yield i, float(self.frame_times[i]), frame
        finally:
            cap.release()

    def index_at(self, t: float) -> int:
        """Index of the first frame captured at or after time t (clamped)."""
        return int(min(max(np.searchsorted(self.frame_times, t), 0), self.n_frames - 1))


class ReplaySource:
    """A camera stand-in for the main loop. `read()` returns the next recorded frame (raising EndOfClip after the
    last one); `frame_time` is that frame's recorded capture time. In `realtime` mode read() sleeps until the frame is
    due, measured from the first read; `fast` never sleeps."""

    def __init__(self, clip: Clip, realtime: bool = True):
        self.clip, self.realtime = clip, realtime
        self.width, self.height = clip.width, clip.height
        self.frame_time = clip.first_time
        self.index = -1
        self._frames = clip.frames()
        self._wall0: float | None = None

    def read(self, timer=None) -> np.ndarray:
        try:
            i, t, frame = next(self._frames)
        except StopIteration:
            raise EndOfClip() from None
        if self.realtime:
            if self._wall0 is None:
                self._wall0 = time.perf_counter() - (t - self.clip.first_time)
            delay = self._wall0 + (t - self.clip.first_time) - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
        self.index, self.frame_time = i, t
        return frame

    def timestamp_ms(self) -> int:
        """Deterministic, strictly increasing millisecond stamp for MediaPipe's VIDEO mode, from the recorded times."""
        ms = int(round((self.frame_time - self.clip.first_time) * 1000.0))
        ms = max(ms, getattr(self, "_last_ms", -1) + 1)
        self._last_ms = ms
        return ms

    def release(self) -> None:
        self._frames.close()


class ReplayAudio:
    """Feeds a clip's audio into an AudioRing as replay time advances, on the recorded clock, so the ring holds what
    it would have held live. `ring.offset_s` (audio.offset_ms) shifts the audio against the video."""

    CHUNK = 0.02  # seconds written per ring block, like the live 20 ms blocks

    def __init__(self, clip: Clip, ring_seconds: float = 10.0, offset_ms: float = 0.0):
        self.clip = clip
        self.samples = clip.audio()
        self.ring = AudioRing(clip.samplerate if clip.has_audio else 44100, ring_seconds, offset_ms / 1000.0)
        self._pos = 0
        self.available = clip.has_audio and len(self.samples) > 0

    def feed_until(self, t: float) -> None:
        """Make everything the microphone would have delivered by video time `t` available in the ring."""
        if not self.available:
            return
        rate = self.ring.samplerate
        t0 = self.clip.audio_start_time
        due = int((t - self.ring.offset_s - t0) * rate + 1e-6)          # samples whose corrected time is <= t
        due = min(max(due, 0), len(self.samples))
        step = max(1, int(self.CHUNK * rate))
        while self._pos < due:
            n = min(step, due - self._pos)
            self.ring.write(t0 + self._pos / rate, self.samples[self._pos:self._pos + n])
            self._pos += n
