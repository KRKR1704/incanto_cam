"""Microphone stream feeding a thread-safe ring buffer, timestamped on the camera's clock.

Everything is on `time.perf_counter()`, the clock the camera frames are stamped with (see
ThreadedCamera.frame_time), so an audio sample and a video frame can be compared directly.

* `AudioRing`   mono float32 ring buffer; every block carries the time of its first sample. Consumers
                (the recorder, later the onset detector) keep their own cursor and read sequentially.
* `MicStream`   the sounddevice input stream (PortAudio's own callback thread), mono, ~20 ms blocks.
* device choice `pick_input_device()` never selects a virtual device (Voicemod, VB-Audio, Steam, ...) on its
                own: if the default input is one it warns and takes the first real microphone.
"""
from __future__ import annotations

import logging
import re
import threading
import time
from collections import deque
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)

# Names of virtual / software inputs. Not a bare "virtual": "HyperX Virtual Surround Sound" is a real headset.
_VIRTUAL = re.compile(
    r"voicemod|vb-?audio|\bcable\b|voicemeeter|steam streaming|virtual (audio )?cable|virtual mic|"
    r"\bobs\b|nvidia broadcast|krisp|stereo mix|what u hear|loopback|wave link|\bsonar\b|"
    r"droidcam|epoccam|iriun|xsplit|snap camera|dummy|mixline", re.I)
# Windows' "route to whatever the default is" entries: they are only as real as the default device
_ALIAS = re.compile(r"sound mapper|primary sound capture", re.I)


def looks_virtual(name: str) -> bool:
    return bool(_VIRTUAL.search(name))


@dataclass(frozen=True)
class InputDevice:
    index: int
    name: str
    hostapi: str
    channels: int
    samplerate: float
    virtual: bool
    alias: bool = False   # "Microsoft Sound Mapper" / "Primary Sound Capture Driver"


def list_input_devices() -> tuple[list[InputDevice], int | None]:
    """(input devices, index of the default input or None). Empty list if PortAudio is unavailable."""
    try:
        import sounddevice as sd
        devices, apis = sd.query_devices(), sd.query_hostapis()
        default = sd.default.device[0]
    except Exception as exc:  # noqa: BLE001 - no PortAudio, no driver: run without audio
        log.warning("Could not query audio devices: %s", exc)
        return [], None
    out = []
    for i, d in enumerate(devices):
        if d["max_input_channels"] > 0:
            name = d["name"]
            out.append(InputDevice(i, name, apis[d["hostapi"]]["name"], int(d["max_input_channels"]),
                                   float(d["default_samplerate"]), looks_virtual(name), bool(_ALIAS.search(name))))
    return out, (default if isinstance(default, int) and default >= 0 else None)


@dataclass(frozen=True)
class Selection:
    device: InputDevice | None
    reason: str
    warnings: tuple[str, ...] = ()


def pick_input_device(devices: list[InputDevice], default_index: int | None,
                      configured: str | int | None = None) -> Selection:
    """Choose the microphone. A configured device (index, or a name substring) is honoured, with a warning if it
    is virtual; automatic choice takes the default input unless it is virtual, else the first real microphone
    (preferring the default's host API). Pure function: testable with a made-up device list."""
    by_index = {d.index: d for d in devices}
    if not devices:
        return Selection(None, "no input device found")
    if configured is not None:
        if isinstance(configured, int):
            dev = by_index.get(configured)
        else:
            want = configured.lower()
            default_api = by_index[default_index].hostapi if default_index in by_index else None
            matches = [d for d in devices if want in d.name.lower()]
            matches.sort(key=lambda d: (d.hostapi != default_api, d.index))  # the default host API first
            dev = matches[0] if matches else None
        if dev is None:
            return Selection(None, f"audio.device {configured!r} matches no input device (see --list-audio-devices)")
        warn = (f"audio.device selects a virtual input ({dev.name}); using it because you asked for it",) if dev.virtual else ()
        return Selection(dev, f"configured audio.device {configured!r}", warn)
    default = by_index.get(default_index) if default_index is not None else None
    if default is not None and not default.virtual and not default.alias:
        return Selection(default, "the default input")
    default_api = default.hostapi if default else None
    real = [d for d in devices if not d.virtual and not d.alias]
    real.sort(key=lambda d: (d.hostapi != default_api, d.index))
    if default is not None:
        why = f"the default input '{default.name}' is virtual" if default.virtual else \
            f"the default input '{default.name}' routes to a virtual device"
    else:
        why = "there is no default input"
    if not real:
        return Selection(None, f"{why}, and no real microphone was found", (f"{why}; no real microphone found",))
    chosen = real[0]
    return Selection(chosen, f"first real microphone ({why})",
                     (f"{why}; not using it. Picked '{chosen.name}' ({chosen.hostapi}) instead. Set audio.device to choose another.",))


class AudioRing:
    """Thread-safe mono float32 ring buffer with a timestamp for every block.

    The writer (`write`) stores the time of each block's first sample. Readers keep a cursor (the number of
    samples they have consumed) and call `read_from(cursor)`; if they fall further behind than the capacity the
    oldest samples are gone and the number lost is reported. `offset_s` (audio.offset_ms) is added to the times
    that readers see; `raw=True` gives the uncorrected capture times (the recorder stores those).
    """

    def __init__(self, samplerate: float, seconds: float = 10.0, offset_s: float = 0.0):
        self.samplerate = float(samplerate)
        self.capacity = max(1, int(round(seconds * samplerate)))
        self.offset_s = offset_s
        self._buf = np.zeros(self.capacity, np.float32)
        self._total = 0                       # samples ever written
        self._blocks: deque[tuple[int, float]] = deque()  # (first sample index, time of it), oldest first
        self._lock = threading.Lock()

    @property
    def total(self) -> int:
        return self._total

    def write(self, t_first: float, samples: np.ndarray) -> None:
        samples = np.asarray(samples, np.float32).reshape(-1)
        n = len(samples)
        if n == 0:
            return
        if n > self.capacity:  # a block bigger than the whole ring: keep its tail
            t_first += (n - self.capacity) / self.samplerate
            samples, n = samples[-self.capacity:], self.capacity
        with self._lock:
            start = self._total
            pos = start % self.capacity
            first = min(n, self.capacity - pos)
            self._buf[pos:pos + first] = samples[:first]
            self._buf[:n - first] = samples[first:]
            self._total = start + n
            self._blocks.append((start, t_first))
            oldest = self._total - self.capacity
            while len(self._blocks) > 1 and self._blocks[1][0] <= oldest:
                self._blocks.popleft()

    def _time_of(self, index: int, raw: bool) -> float:
        """Time of sample `index` (must hold the lock)."""
        if not self._blocks:
            return 0.0
        start, t = self._blocks[0]
        for s, ts in self._blocks:
            if s > index:
                break
            start, t = s, ts
        return t + (index - start) / self.samplerate + (0.0 if raw else self.offset_s)

    def time_of(self, index: int, raw: bool = False) -> float:
        with self._lock:
            return self._time_of(index, raw)

    def read_from(self, cursor: int, raw: bool = False) -> tuple[np.ndarray, int, float, int]:
        """Everything written since `cursor`: (samples, new cursor, time of the first returned sample, samples lost
        because the reader fell more than `capacity` behind)."""
        with self._lock:
            end = self._total
            lost = max(0, end - self.capacity - cursor)
            start = cursor + lost
            n = end - start
            if n <= 0:
                return np.zeros(0, np.float32), end, self._time_of(end, raw), lost
            pos = start % self.capacity
            first = min(n, self.capacity - pos)
            out = np.concatenate((self._buf[pos:pos + first], self._buf[:n - first]))
            return out, end, self._time_of(start, raw), lost

    def latest(self, seconds: float) -> tuple[np.ndarray, float]:
        """The newest `seconds` of audio and the time of its first sample."""
        n = min(int(seconds * self.samplerate), self.capacity)
        with self._lock:
            start = max(0, self._total - n)
        samples, _, t0, _ = self.read_from(start)
        return samples, t0

    def window(self, t0: float, t1: float) -> np.ndarray:
        """Samples whose (offset-corrected) time lies in [t0, t1); empty if that is no longer in the ring."""
        with self._lock:
            if not self._blocks:
                return np.zeros(0, np.float32)
            oldest = max(0, self._total - self.capacity)
            i0, i1 = self._index_at(t0), self._index_at(t1)
            i0, i1 = max(i0, oldest), min(i1, self._total)
        if i1 <= i0:
            return np.zeros(0, np.float32)
        samples, _, _, _ = self.read_from(i0)
        return samples[:i1 - i0]

    def _index_at(self, t: float) -> int:
        """Sample index at time t (hold the lock); extrapolates linearly from the nearest block."""
        start, ts = self._blocks[0]
        for s, tb in self._blocks:
            if tb + self.offset_s > t:
                break
            start, ts = s, tb
        return start + int(round((t - ts - self.offset_s) * self.samplerate))

    def level_dbfs(self, seconds: float = 0.1) -> float:
        samples, _ = self.latest(seconds)
        if len(samples) == 0:
            return -120.0
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2)))
        return 20 * np.log10(max(rms, 1e-6))


class MicStream:
    """sounddevice input stream: mono, `block_ms` blocks, written into an AudioRing with capture times.

    The time of a block's first sample comes from PortAudio's ADC time (when the driver reports one), converted
    to perf_counter time at the moment of the callback. Callback jitter only ever makes a block look later, so the
    sample-clock anchor (time of sample 0) is tracked as a running minimum that creeps up by 100 ppm per second
    (the audio and system clocks drift a little): timestamps then advance exactly at the sample rate.
    """

    DRIFT = 1e-4

    def __init__(self, device: InputDevice, block_ms: int = 20, ring_seconds: float = 10.0, offset_ms: float = 0.0):
        self.device = device
        self.samplerate = device.samplerate
        self.blocksize = max(16, int(round(self.samplerate * block_ms / 1000.0)))
        self.ring = AudioRing(self.samplerate, ring_seconds, offset_ms / 1000.0)
        self.status_count = 0           # PortAudio over/underflow reports
        self._stream = None
        self._anchor: float | None = None
        self._written = 0
        self._channels = 1

    def _callback(self, indata, frames, time_info, status) -> None:
        arrive = time.perf_counter()
        if status:
            self.status_count += 1
        delta = time_info.currentTime - time_info.inputBufferAdcTime
        t_first = arrive - delta if 0.0 <= delta < 1.0 else arrive - frames / self.samplerate
        mono = indata[:, 0] if self._channels == 1 else indata.mean(axis=1)
        cand = t_first - self._written / self.samplerate
        if self._anchor is None:
            self._anchor = cand
        else:
            self._anchor = min(self._anchor + self.DRIFT * frames / self.samplerate, cand)
        self.ring.write(self._anchor + self._written / self.samplerate, mono)
        self._written += frames

    def start(self) -> bool:
        import sounddevice as sd
        for channels in (1, min(2, self.device.channels)):
            try:
                self._channels = channels
                self._stream = sd.InputStream(device=self.device.index, channels=channels, samplerate=self.samplerate,
                                              blocksize=self.blocksize, dtype="float32", callback=self._callback,
                                              latency="low")
                self._stream.start()
                log.info("Microphone: '%s' (%s), %d ch -> mono, %.0f Hz, %d-sample blocks (%.1f ms)", self.device.name,
                         self.device.hostapi, channels, self.samplerate, self.blocksize,
                         1000 * self.blocksize / self.samplerate)
                return True
            except Exception as exc:  # noqa: BLE001
                log.warning("Could not open '%s' with %d channel(s): %s", self.device.name, channels, exc)
                self._stream = None
        return False

    def stop(self) -> None:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:  # noqa: BLE001
                log.exception("Closing the microphone stream failed")
            self._stream = None


def start_microphone(cfg) -> MicStream | None:
    """Open the microphone per `cfg` (an AudioConfig). None, with a clear log line, if there is none to use."""
    if not cfg.enabled:
        log.info("Audio disabled (audio.enabled: false): running without a microphone.")
        return None
    devices, default = list_input_devices()
    sel = pick_input_device(devices, default, cfg.device)
    for w in sel.warnings:
        log.warning("%s", w)
    if sel.device is None:
        log.warning("Running WITHOUT audio: %s.", sel.reason)
        return None
    log.info("Audio input: '%s' [%s] (%s)", sel.device.name, sel.device.hostapi, sel.reason)
    mic = MicStream(sel.device, cfg.block_ms, cfg.ring_seconds, cfg.offset_ms)
    if not mic.start():
        log.warning("Running WITHOUT audio: the microphone could not be opened.")
        return None
    return mic


def print_devices() -> int:
    """--list-audio-devices"""
    devices, default = list_input_devices()
    if not devices:
        print("No audio input devices found.")
        return 1
    print(f"{'idx':>3}  {'host API':<22} {'ch':>2} {'rate':>6}  name")
    for d in devices:
        flags = []
        if d.index == default:
            flags.append("DEFAULT")
        if d.virtual:
            flags.append("virtual")
        if d.alias:
            flags.append("routes to the default")
        print(f"{d.index:>3}  {d.hostapi:<22} {d.channels:>2} {d.samplerate:>6.0f}  {d.name}"
              + (f"   [{', '.join(flags)}]" if flags else ""))
    sel = pick_input_device(devices, default)
    print()
    for w in sel.warnings:
        print("WARNING:", w)
    print("With audio.device: auto, Incanto would use:",
          f"[{sel.device.index}] {sel.device.name} ({sel.reason})" if sel.device else f"nothing ({sel.reason})")
    return 0
