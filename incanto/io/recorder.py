"""Records raw camera frames and microphone audio to a clip folder, on its own thread.

recordings/<timestamp>/            (local-only: gitignored, and the pre-commit hook refuses video/audio files)
    video.avi    raw camera frames, BEFORE any effect
    audio.wav    mono 16-bit PCM from the microphone ring (absent when there is no microphone)
    frames.csv   frame index -> capture time in seconds, one row per frame in video.avi
    meta.json    sizes, codec, nominal fps, and the audio start time on the same clock

Replay syncs on the recorded times, never on the container's nominal fps (the camera delivers 29.9-30.2).
Both clocks are time.perf_counter(): frame times are ThreadedCamera.frame_time, audio times come from AudioRing.

Codec. `lossless` is HFYU (huffyuv): every frame is stored bit-exact, so replay perception sees exactly the pixels
live perception saw, which is what the Phase 4 gate asks for. It costs about 930 KB per 720p frame (1.7 GB per
minute) and 8 ms of encode time per frame on the writer thread. `mjpg` is about 25x smaller (66 KB/frame) but a
second JPEG generation (mean error 1.7 levels, up to 30). MJPG quality cannot be raised in this OpenCV build (the
VIDEOWRITER_PROP_QUALITY parameter is rejected and the ffmpeg quality options are ignored), and FFV1 lossless costs
65 ms per frame, far too slow. H.264 is not available (no OpenH264 library).
"""
from __future__ import annotations

import csv
import json
import logging
import queue
import shutil
import threading
import time
import wave
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

VIDEO_FILE, AUDIO_FILE, FRAMES_FILE, META_FILE = "video.avi", "audio.wav", "frames.csv", "meta.json"
FOURCC = {"lossless": "HFYU", "mjpg": "MJPG"}
META_VERSION = 1


class RecordingError(RuntimeError):
    """A recording could not be started."""


def write_frame_times(path: Path, times: list[float]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["frame", "time_s"])
        for i, t in enumerate(times):
            w.writerow([i, f"{t:.6f}"])


def read_frame_times(path: Path) -> np.ndarray:
    """Frame index -> capture time (seconds, camera clock) from a sidecar written by the recorder."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    if not rows or rows[0][:2] != ["frame", "time_s"]:
        raise ValueError(f"{path} is not a frame-time sidecar")
    times = []
    for expected, row in enumerate(rows[1:]):
        if int(row[0]) != expected:
            raise ValueError(f"{path}: frame indices are not consecutive at row {expected}")
        times.append(float(row[1]))
    return np.asarray(times, np.float64)


@dataclass
class RecordingSummary:
    path: Path
    frames: int
    dropped: int
    duration_s: float
    audio_samples: int
    first_frame_time: float | None


class ClipRecorder:
    """One recording. `submit()` (main loop) never blocks: the frame is copied into a bounded queue and a writer
    thread encodes it. If the encoder falls more than `queue_frames` behind, frames are dropped and counted
    (and reported when recording stops) instead of slowing the main loop."""

    def __init__(self, root: Path, size: tuple[int, int], fps: float, codec: str = "lossless", mic=None,
                 queue_frames: int = 45, min_free_gb: float = 2.0, name: str | None = None):
        if codec not in FOURCC:
            raise ValueError(f"unknown codec {codec!r}")
        self.root, self.size, self.fps, self.codec, self.mic = Path(root), size, fps, codec, mic
        self.queue_frames, self.min_free_gb = queue_frames, min_free_gb
        self.name = name
        self.path: Path | None = None
        self.dropped = 0
        self.frames = 0               # frames written so far (readable from the main loop for the indicator)
        self._times: list[float] = []
        self._q: queue.Queue = queue.Queue(maxsize=queue_frames)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._error: Exception | None = None
        self._started = 0.0
        self._first_time: float | None = None
        self._audio_start: float | None = None
        self._audio_samples = 0
        self._audio_lost = 0

    @property
    def active(self) -> bool:
        return self._thread is not None and self._thread.is_alive() and not self._stop.is_set()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self._started if self._started else 0.0

    def start(self) -> Path:
        self.root.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(self.root).free / 1e9
        if free < self.min_free_gb:
            raise RecordingError(f"only {free:.1f} GB free on the recordings drive (recording.min_free_gb is "
                                 f"{self.min_free_gb:g}); free some space first")
        name = self.name or datetime.now().strftime("%Y%m%d-%H%M%S")
        path, n = self.root / name, 1
        while path.exists():           # two recordings in one second
            n += 1
            path = self.root / f"{name}-{n}"
        path.mkdir()
        self.path = path
        fourcc = cv2.VideoWriter_fourcc(*FOURCC[self.codec])
        self._writer = cv2.VideoWriter(str(path / VIDEO_FILE), fourcc, float(self.fps), self.size)
        if not self._writer.isOpened():
            raise RecordingError(f"the {self.codec} video writer could not be opened (codec {FOURCC[self.codec]})")
        self._wav = None
        self._cursor = 0
        if self.mic is not None:
            self._cursor = self.mic.ring.total  # audio starts now
            self._wav = wave.open(str(path / AUDIO_FILE), "wb")
            self._wav.setnchannels(1)
            self._wav.setsampwidth(2)
            self._wav.setframerate(int(round(self.mic.samplerate)))
        self._started = time.perf_counter()
        self._thread = threading.Thread(target=self._run, name="recorder", daemon=True)
        self._thread.start()
        log.info("REC started: %s (%s, %dx%d, audio %s)", path, self.codec, *self.size,
                 "yes" if self._wav else "none")
        return path

    def submit(self, frame: np.ndarray, t: float) -> None:
        """Queue a raw camera frame captured at time `t` (camera clock). Never blocks."""
        if not self.active:
            return
        if frame.shape[1] != self.size[0] or frame.shape[0] != self.size[1]:
            raise ValueError(f"frame {frame.shape[1]}x{frame.shape[0]} != recording size {self.size[0]}x{self.size[1]}")
        try:
            self._q.put_nowait((frame.copy(), t))  # copy: the camera may reuse its buffer
        except queue.Full:
            self.dropped += 1

    def _drain_audio(self) -> None:
        if self._wav is None:
            return
        samples, self._cursor, t0, lost = self.mic.ring.read_from(self._cursor, raw=True)
        if lost:
            self._audio_lost += lost
            log.warning("Recorder fell behind the microphone: %d samples lost", lost)
        if len(samples):
            if self._audio_start is None:
                self._audio_start = t0
            self._wav.writeframes((np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2").tobytes())
            self._audio_samples += len(samples)

    def _run(self) -> None:
        try:
            while True:
                try:
                    frame, t = self._q.get(timeout=0.02)
                    self._writer.write(frame)
                    if self._first_time is None:
                        self._first_time = t
                    self._times.append(t)
                    self.frames = len(self._times)
                except queue.Empty:
                    pass
                self._drain_audio()
                if self._stop.is_set() and self._q.empty():
                    break
            self._drain_audio()
        except Exception as exc:  # noqa: BLE001 - reported by stop()
            self._error = exc
            log.exception("Recorder thread failed")

    def stop(self) -> RecordingSummary:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        self._writer.release()
        if self._wav is not None:
            self._wav.close()
        path = self.path
        write_frame_times(path / FRAMES_FILE, self._times)
        duration = (self._times[-1] - self._times[0]) if len(self._times) > 1 else 0.0
        meta = {
            "version": META_VERSION, "clock": "perf_counter", "created": datetime.now().isoformat(timespec="seconds"),
            "video": VIDEO_FILE, "width": self.size[0], "height": self.size[1], "nominal_fps": self.fps,
            "codec": self.codec, "fourcc": FOURCC[self.codec], "frames": len(self._times), "dropped_frames": self.dropped,
            "first_frame_time": self._times[0] if self._times else None,
            "last_frame_time": self._times[-1] if self._times else None, "duration_s": duration,
            "audio": ({"file": AUDIO_FILE, "samplerate": int(round(self.mic.samplerate)), "start_time": self._audio_start,
                       "samples": self._audio_samples, "lost_samples": self._audio_lost, "device": self.mic.device.name}
                      if self._wav is not None and self._audio_start is not None else None),
        }
        (path / META_FILE).write_text(json.dumps(meta, indent=2), encoding="utf-8")
        eff = (len(self._times) - 1) / duration if duration > 0 else 0.0
        log.info("REC stopped: %s | %d frames in %.1fs (%.2f fps effective), %d dropped | audio %s%s", path,
                 len(self._times), duration, eff, self.dropped,
                 f"{self._audio_samples / self.mic.samplerate:.1f}s" if self._wav is not None and self._audio_start else "none",
                 f" | {self._audio_lost} audio samples lost" if self._audio_lost else "")
        if self.dropped:
            log.warning("%d frames were dropped because the encoder could not keep up (recording.codec: mjpg is "
                        "much lighter than lossless)", self.dropped)
        if self._error:
            raise RecordingError(f"the recorder thread failed: {self._error}")
        return RecordingSummary(path, len(self._times), self.dropped, duration, self._audio_samples,
                                self._times[0] if self._times else None)
