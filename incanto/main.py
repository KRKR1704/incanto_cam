"""Entry point: main loop, hotkeys, and wiring of all components."""
from __future__ import annotations

import argparse
import logging
import platform
import time

import cv2
import psutil

from incanto.config import ConfigError, load_config
from incanto.io.camera import Camera, CameraError
from incanto.io.vcam import VCamError, VirtualCamera
from incanto.utils.fps import FpsCounter, StageTimer
from incanto.utils.logging import setup_logging
from incanto.utils import winperf

log = logging.getLogger("incanto")
PREVIEW_WINDOW = "Incanto preview (Q to quit)"
FPS_LOG_INTERVAL_S = 5.0


def draw_fps(frame, fps: float) -> None:
    """Draw the FPS counter onto `frame` in place. Call only on the preview copy."""
    text = f"{fps:5.1f} FPS"
    cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(frame, text, (16, 36), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)


def show_camera_settings(config_path: str | None) -> int:
    """Open the DirectShow driver property dialog and wait until it is closed."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        log.error("Config error: %s", exc)
        return 2
    cam = None
    try:
        cam = Camera(cfg.video.width, cfg.video.height, cfg.video.fps, backends=(cv2.CAP_DSHOW,))
        if not cam.show_driver_settings():
            log.error("The driver did not open its settings dialog.")
            return 1
        log.info("Driver settings dialog requested. Change settings, then close the dialog.")
        deadline = time.perf_counter() + 15
        while not winperf.own_dialog_open():  # keep the graph alive while the dialog appears
            if time.perf_counter() > deadline:
                log.error("No settings dialog appeared within 15s.")
                return 1
            cam.read()
        while winperf.own_dialog_open():
            cam.read()
        log.info("Dialog closed. Exiting.")
        return 0
    except CameraError as exc:
        log.error("%s", exc)
        return 1
    finally:
        if cam:
            cam.release()


def run(config_path: str | None, max_seconds: float | None, show_preview: bool,
        lock_exposure: float | None = None, perf_tweaks: bool = True) -> int:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        log.error("Config error: %s", exc)
        return 2

    log.info("System: cpu=%s | power mode=%s | power=%s | perf tweaks=%s",
             platform.processor(), winperf.power_mode(), winperf.power_source(),
             "on" if perf_tweaks else "off")
    if perf_tweaks:
        winperf.apply_perf_tweaks()

    cam = vcam = None
    try:
        cam = Camera(cfg.video.width, cfg.video.height, cfg.video.fps)
        # Output size follows the real camera resolution so frames are always sent unchanged.
        vcam = VirtualCamera(cam.width, cam.height, cfg.video.fps, cfg.vcam.backend)
        if lock_exposure is not None:
            cam.lock_exposure(lock_exposure)
        if show_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)

        fps_counter, timer = FpsCounter(), StageTimer()
        rx_first = rx_last = None  # read-completion times, for the camera delivery rate
        rx_count = 0
        proc = psutil.Process()
        proc.cpu_percent(None)  # prime the counter
        start = last_log = time.perf_counter()
        frames = frames_since_log = 0
        log.info("Running. Press Q in the preview window to quit.")

        while True:
            frame = cam.read(timer)
            rx_last = time.perf_counter()
            if rx_first is None:
                rx_first = rx_last
            rx_count += 1
            with timer.stage("vcam"):
                vcam.send(frame)  # clean frame: no overlay ever reaches the virtual camera
            fps_counter.tick()
            frames += 1
            frames_since_log += 1

            if show_preview:
                preview = frame.copy()
                draw_fps(preview, fps_counter.fps)
                cv2.imshow(PREVIEW_WINDOW, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    log.info("Quit requested.")
                    break
                if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("Preview window closed.")
                    break

            now = time.perf_counter()
            if now - last_log >= FPS_LOG_INTERVAL_S:
                stages = " ".join(f"{k}={v:.1f}ms" for k, v in timer.averages_ms().items())
                rx_fps = (rx_count - 1) / (rx_last - rx_first) if rx_count > 1 and rx_last > rx_first else 0.0
                log.info("FPS %.1f (avg over %.1fs) | camera delivers %.1f fps | %s | cpu=%.0f%% (of one core)",
                         frames_since_log / (now - last_log), now - last_log, rx_fps, stages,
                         proc.cpu_percent(None))
                rx_first, rx_count = None, 0
                log.info("camera props: %s | power=%s", cam.exposure_state(), winperf.power_source())
                timer.reset()
                last_log, frames_since_log = now, 0
            if max_seconds is not None and now - start >= max_seconds:
                log.info("Reached --seconds limit.")
                break

        elapsed = time.perf_counter() - start
        log.info("Summary: %d frames in %.1fs = %.1f FPS average", frames, elapsed, frames / elapsed)
        return 0
    except (CameraError, VCamError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return 0
    finally:
        if vcam:
            vcam.close()
        if cam:
            cam.release()
        cv2.destroyAllWindows()
        log.info("Shut down cleanly.")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="incanto", description="Gesture-triggered video effects camera")
    parser.add_argument("--config", help="path to config.yaml (default: project config.yaml)")
    parser.add_argument("--seconds", type=float, help="exit automatically after this many seconds")
    parser.add_argument("--no-preview", action="store_true", help="disable the local preview window")
    parser.add_argument("--lock-exposure", type=float, metavar="VALUE",
                        help="disable auto-exposure and set this fixed exposure (backend units, e.g. -6)")
    parser.add_argument("--no-perf-tweaks", action="store_true",
                        help="do not opt out of power throttling or raise process priority")
    parser.add_argument("--camera-settings", action="store_true",
                        help="show the camera driver's own property dialog (DirectShow), wait for it to close, then exit")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    if args.camera_settings:
        return show_camera_settings(args.config)
    return run(args.config, args.seconds, not args.no_preview, args.lock_exposure, not args.no_perf_tweaks)


if __name__ == "__main__":
    raise SystemExit(main())
