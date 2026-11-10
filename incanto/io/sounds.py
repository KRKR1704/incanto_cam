"""Short audio cues (plate countdown, capturing, captured OK, error) so you can leave the frame
without watching the preview. Windows only (winsound); silent elsewhere or when muted in config.

Beeps are played on a small worker thread: winsound.Beep blocks for the length of the tone and the
main loop must never wait for it.
"""
from __future__ import annotations

import logging
import queue
import threading

try:
    import winsound
except ImportError:  # not Windows
    winsound = None

log = logging.getLogger(__name__)

# (frequency Hz, duration ms) sequences; each cue is recognisable by ear
CUES: dict[str, tuple[tuple[int, int], ...]] = {
    "tick": ((880, 90),),                             # one per countdown second
    "capturing": ((1320, 450),),                      # long high tone: stay out, frames are being taken
    "ok": ((1047, 110), (1319, 110), (1568, 220)),    # rising three notes: plate captured
    "error": ((330, 220), (220, 450)),                # falling low pair: aborted / plate does not match
}


class SoundCues:
    def __init__(self, enabled: bool = True, beep=None):
        self._beep = beep or (winsound.Beep if winsound else None)
        self.enabled = bool(enabled and self._beep)
        if enabled and not self._beep:
            log.info("Audio cues unavailable (winsound is Windows-only).")
        self._queue: queue.Queue = queue.Queue(maxsize=8)
        if self.enabled:
            threading.Thread(target=self._run, name="sound-cues", daemon=True).start()

    def play(self, cue: str) -> None:
        """Queue a cue; never blocks. Unknown cues raise KeyError (a typo should fail in tests)."""
        notes = CUES[cue]
        if not self.enabled:
            return
        try:
            self._queue.put_nowait(notes)
        except queue.Full:
            pass  # a backlog of beeps is worse than a dropped one

    def _run(self) -> None:
        while True:
            for freq, ms in self._queue.get():
                try:
                    self._beep(freq, ms)
                except RuntimeError:  # no audio device
                    pass
