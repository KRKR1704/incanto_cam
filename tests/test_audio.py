"""Audio ring buffer (timestamps on the camera clock) and microphone choice (never a virtual device by default)."""
import numpy as np
import pytest

from incanto.config import ConfigError, load_config
from incanto.io.audio import AudioRing, InputDevice, looks_virtual, pick_input_device

RATE = 1000.0


def ring(seconds=1.0, offset=0.0):
    return AudioRing(RATE, seconds, offset)


def test_ring_returns_what_was_written_in_order_with_timestamps():
    r = ring()
    r.write(10.0, np.arange(0, 20, dtype=np.float32))
    r.write(10.020, np.arange(20, 50, dtype=np.float32))
    out, cursor, t0, lost = r.read_from(0)
    assert out.tolist() == list(range(50)) and cursor == 50 and lost == 0
    assert t0 == pytest.approx(10.0)
    assert r.time_of(25) == pytest.approx(10.025)          # inside the second block
    out, cursor, t0, _ = r.read_from(cursor)
    assert len(out) == 0 and cursor == 50                  # nothing new


def test_ring_cursor_reads_are_sequential_across_writes():
    r = ring()
    cur, got = 0, []
    for k in range(5):
        r.write(5.0 + k * 0.02, np.full(20, k, np.float32))
        out, cur, t0, lost = r.read_from(cur)
        assert lost == 0 and t0 == pytest.approx(5.0 + k * 0.02)
        got.append(out)
    assert np.concatenate(got).tolist() == sum(([k] * 20 for k in range(5)), [])


def test_ring_wraps_and_reports_samples_lost_by_a_slow_reader():
    r = ring(seconds=0.1)                                  # capacity 100
    for k in range(10):                                    # 200 samples in total
        r.write(k * 0.02, np.arange(k * 20, k * 20 + 20, dtype=np.float32))
    out, cursor, t0, lost = r.read_from(0)
    assert lost == 100 and len(out) == 100 and out[0] == 100 and out[-1] == 199
    assert t0 == pytest.approx(0.100)                      # the time of the first sample that is still there
    assert r.total == 200


def test_ring_offset_applies_to_reads_but_raw_is_untouched():
    r = ring(offset=0.040)
    r.write(2.0, np.zeros(100, np.float32))
    _, _, t_corrected, _ = r.read_from(0)
    _, _, t_raw, _ = r.read_from(0, raw=True)
    assert t_corrected == pytest.approx(2.040) and t_raw == pytest.approx(2.0)


def test_ring_window_picks_samples_by_time_and_latest_and_level():
    r = ring(seconds=2.0)
    for k in range(10):
        r.write(100.0 + k * 0.1, np.full(100, k, np.float32))   # 1 s: block k holds value k
    w = r.window(100.3, 100.5)
    assert len(w) == 200 and w[0] == 3 and w[-1] == 4
    tail, t0 = r.latest(0.25)
    assert len(tail) == 250 and t0 == pytest.approx(100.75)
    assert r.window(500.0, 501.0).size == 0
    quiet = ring()
    quiet.write(0.0, np.full(500, 0.01, np.float32))
    assert quiet.level_dbfs(0.2) == pytest.approx(-40.0, abs=0.1)
    assert ring().level_dbfs() == -120.0


def test_ring_block_larger_than_capacity_keeps_the_tail():
    r = ring(seconds=0.1)
    r.write(0.0, np.arange(300, dtype=np.float32))
    out, _, t0, _ = r.read_from(0)
    assert len(out) == 100 and out[0] == 200 and t0 == pytest.approx(0.2)


# -- device choice -------------------------------------------------------------------------------
def dev(i, name, api="MME", virtual=None, alias=False):
    return InputDevice(i, name, api, 2, 44100.0, looks_virtual(name) if virtual is None else virtual, alias)


MACHINE = [dev(0, "Microsoft Sound Mapper - Input", alias=True), dev(1, "Microphone (Voicemod)"),
           dev(2, "Headset Microphone (HyperX Virt"), dev(3, "Microphone (Steam Streaming Mic"),
           dev(4, "Microphone Array (Intel Smart"), dev(13, "Microphone (Voicemod)", "Windows DirectSound"),
           dev(14, "Headset Microphone (HyperX Virtual Surround Sound)", "Windows DirectSound")]


@pytest.mark.parametrize("name,virtual", [
    ("Microphone (Voicemod)", True), ("CABLE Output (VB-Audio Virtual Cable)", True), ("VB-Audio Voicemeeter VAIO", True),
    ("Microphone (Steam Streaming Microphone)", True), ("Stereo Mix (Realtek HD Audio Stereo input)", True),
    ("OBS Virtual Camera audio", True), ("NVIDIA Broadcast", True),
    ("Headset Microphone (HyperX Virtual Surround Sound)", False),   # "Virtual" alone does not make it virtual
    ("Microphone Array (Intel Smart Sound Technology)", False), ("Microphone (Realtek HD Audio Mic input)", False)])
def test_virtual_detection(name, virtual):
    assert looks_virtual(name) is virtual


def test_virtual_default_is_replaced_by_the_first_real_microphone_with_a_warning():
    sel = pick_input_device(MACHINE, default_index=1)
    assert sel.device.index == 2                           # not Voicemod, not the sound mapper, not Steam
    assert sel.warnings and "Voicemod" in sel.warnings[0] and "virtual" in sel.warnings[0]
    assert "HyperX" in sel.warnings[0]


def test_sound_mapper_default_that_routes_to_a_virtual_device_is_not_trusted():
    sel = pick_input_device(MACHINE, default_index=0)
    assert sel.device.index == 2 and sel.warnings


def test_real_default_is_used_as_is():
    sel = pick_input_device(MACHINE, default_index=4)
    assert sel.device.index == 4 and not sel.warnings


def test_first_real_microphone_prefers_the_default_host_api():
    devices = [dev(5, "Microphone Array", "Windows WASAPI"), dev(1, "Microphone (Voicemod)"), dev(2, "USB Mic")]
    assert pick_input_device(devices, default_index=1).device.index == 2


def test_configured_device_by_name_or_index_and_virtual_choice_is_warned_not_refused():
    assert pick_input_device(MACHINE, 1, "intel").device.index == 4
    assert pick_input_device(MACHINE, 1, 4).device.index == 4
    sel = pick_input_device(MACHINE, 4, "voicemod")
    assert sel.device.index == 1 and sel.warnings          # honoured, with a warning
    assert pick_input_device(MACHINE, 1, "nonexistent").device is None
    assert pick_input_device(MACHINE, 1, 99).device is None


def test_no_microphone_at_all_runs_without_audio():
    assert pick_input_device([], None).device is None
    only_virtual = [dev(1, "Microphone (Voicemod)"), dev(2, "CABLE Output")]
    sel = pick_input_device(only_virtual, 1)
    assert sel.device is None and sel.warnings


def test_audio_config_parsing(tmp_path):
    def cfg(text):
        f = tmp_path / "c.yaml"
        f.write_text(text)
        return load_config(f).audio
    assert cfg("audio:\n  device: auto\n").device is None
    assert cfg("audio:\n  device: HyperX\n").device == "HyperX"
    assert cfg("audio:\n  device: 4\n").device == 4
    assert cfg("audio:\n  offset_ms: -80\n").offset_ms == -80
    assert cfg("{}").enabled and cfg("{}").block_ms == 20
    for bad in ("device: true", "block_ms: 1", "ring_seconds: 0"):
        with pytest.raises(ConfigError):
            cfg(f"audio:\n  {bad}\n")
