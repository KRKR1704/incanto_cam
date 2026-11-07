"""Windows-only performance helpers: opt out of power throttling, raise priority, report power state."""
from __future__ import annotations

import ctypes
import logging
import sys
import uuid
from ctypes import wintypes

import psutil

log = logging.getLogger(__name__)

IS_WINDOWS = sys.platform == "win32"

_PROCESS_POWER_THROTTLING = 4  # PROCESS_INFORMATION_CLASS::ProcessPowerThrottling
_PROCESS_POWER_THROTTLING_CURRENT_VERSION = 1
_PROCESS_POWER_THROTTLING_EXECUTION_SPEED = 0x1

_OVERLAYS = {
    "00000000-0000-0000-0000-000000000000": "Balanced (no overlay)",
    "961cc777-2547-4f9d-8174-7d86181b8a7a": "Best power efficiency",
    "3af9b8d9-7c97-431d-ad78-34a8bfc9f9f7": "Better battery",
    "ded574b5-45a0-4f42-8737-46345c09c238": "Best performance",
}


class _ProcessPowerThrottlingState(ctypes.Structure):
    _fields_ = [("Version", wintypes.ULONG), ("ControlMask", wintypes.ULONG), ("StateMask", wintypes.ULONG)]


def disable_power_throttling() -> bool:
    """Opt this process out of EcoQoS execution-speed throttling. Returns True on success."""
    if not IS_WINDOWS:
        return False
    try:
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        kernel32.SetProcessInformation.argtypes = [
            wintypes.HANDLE, ctypes.c_int, ctypes.c_void_p, wintypes.DWORD]
        kernel32.SetProcessInformation.restype = wintypes.BOOL
        state = _ProcessPowerThrottlingState(
            _PROCESS_POWER_THROTTLING_CURRENT_VERSION,
            _PROCESS_POWER_THROTTLING_EXECUTION_SPEED,
            0,  # StateMask=0 with the bit set in ControlMask => throttling forced OFF
        )
        ok = kernel32.SetProcessInformation(
            kernel32.GetCurrentProcess(), _PROCESS_POWER_THROTTLING,
            ctypes.byref(state), ctypes.sizeof(state))
        if not ok:
            log.warning("SetProcessInformation(ProcessPowerThrottling) failed: WinError %d",
                        ctypes.get_last_error())
        return bool(ok)
    except Exception as exc:  # noqa: BLE001 - diagnostics must never crash the app
        log.warning("disable_power_throttling error: %s", exc)
        return False


def set_above_normal_priority() -> bool:
    if not IS_WINDOWS:
        return False
    try:
        psutil.Process().nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
        return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not set ABOVE_NORMAL priority: %s", exc)
        return False


def apply_perf_tweaks() -> None:
    log.info("perf tweaks: power throttling opt-out=%s, ABOVE_NORMAL priority=%s",
             "OK" if disable_power_throttling() else "FAILED",
             "OK" if set_above_normal_priority() else "FAILED")


def own_dialog_open() -> bool:
    """True if this process currently has a visible top-level dialog window (class #32770)."""
    if not IS_WINDOWS:
        return False
    user32 = ctypes.WinDLL("user32")
    pid = ctypes.c_ulong(psutil.Process().pid).value
    found = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(hwnd, _lparam):
        owner = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(owner))
        if owner.value == pid and user32.IsWindowVisible(hwnd):
            cls = ctypes.create_unicode_buffer(64)
            user32.GetClassNameW(hwnd, cls, 64)
            if cls.value == "#32770":
                found.append(hwnd)
        return True

    user32.EnumWindows(visit, 0)
    return bool(found)


def power_mode() -> str:
    """Active Windows 11 power mode (the overlay slider), or 'unknown'."""
    if not IS_WINDOWS:
        return "n/a"
    try:
        powrprof = ctypes.WinDLL("powrprof")
        guid = (ctypes.c_ubyte * 16)()
        if powrprof.PowerGetEffectiveOverlayScheme(ctypes.byref(guid)) != 0:
            return "unknown"
        key = str(uuid.UUID(bytes_le=bytes(guid)))
        return _OVERLAYS.get(key, key)
    except Exception:  # noqa: BLE001
        return "unknown"


def power_source() -> str:
    """'AC 99%' / 'battery 80%' / 'no battery'."""
    batt = psutil.sensors_battery()
    if batt is None:
        return "no battery"
    return f"{'AC' if batt.power_plugged else 'battery'} {batt.percent:.0f}%"
