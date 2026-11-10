"""Entry point: main loop, hotkeys, and wiring of all components."""
from __future__ import annotations

import argparse
import dataclasses
import logging
import platform
import threading
import time
from pathlib import Path

import cv2
import psutil

from incanto.config import ConfigError, load_config
from incanto.io.camera import (EXPOSURE_MAX, EXPOSURE_MIN, Camera, CameraError, ThreadedCamera,
                               install_exit_guard)
from incanto.io.sounds import SoundCues
from incanto.io.vcam import VCamError, VirtualCamera
from incanto.perception.hands import HandsWorker, HandTracker
from incanto.plate.clean_plate import (PlateCapture, load_plate, make_plate, save_plate)
from incanto.plate.health import HealthLimits
from incanto.perception.segmenter import Segmenter
from incanto.render import overlay
from incanto.render.compositor import Compositor
from incanto.render.vanish_mask import VanishMask
from incanto.utils import winperf
from incanto.utils.evidence import EvidenceSession, grab_vcam_frames
from incanto.utils.fps import FpsCounter, StageTimer
from incanto.utils.imaging import downscale
from incanto.utils.logging import setup_logging

log = logging.getLogger("incanto")
PREVIEW_WINDOW = "Incanto preview (Q quit, D debug view, B capture plate, V vanish)"
DEBUG_VIEWS = ("off", "sources", "leak")
FPS_LOG_INTERVAL_S = 5.0
PLATE_MISMATCH = "Plate doesn't match the room - press B"


class SetupError(RuntimeError):
    """A required file or resource is missing."""


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


def reset_camera(config_path: str | None) -> int:
    """Write auto exposure to the camera (the driver remembers manual exposure across processes)."""
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        log.error("Config error: %s", exc)
        return 2
    cam = None
    try:
        cam = Camera(cfg.video.width, cfg.video.height, cfg.video.fps)
        before = cam.mean_brightness()
        accepted = cam.set_auto_exposure()
        time.sleep(1.0)
        after = cam.mean_brightness(15)
        log.info("--reset-camera: auto exposure written (accepted=%s) | mean brightness before %.0f -> after %.0f",
                 accepted, before, after)
        return 0 if accepted else 1
    except CameraError as exc:
        log.error("%s", exc)
        return 1
    finally:
        if cam:
            cam.release()


def run(config_path: str | None, max_seconds: float | None, show_preview: bool,
        lock_exposure: float | None = None, perf_tweaks: bool = True,
        segment_every: int | None = None, evidence_dir: str | None = None,
        capture_mode: str | None = None, shots_dir: str | None = None,
        mask_mode: str | None = None, vanish_on_start: bool = False,
        debug_view: str = "sources") -> int:
    try:
        cfg = load_config(config_path)
    except ConfigError as exc:
        log.error("Config error: %s", exc)
        return 2
    if segment_every is not None:
        cfg = dataclasses.replace(
            cfg, perception=dataclasses.replace(cfg.perception, segment_every_n=segment_every))

    if mask_mode is not None:
        cfg = dataclasses.replace(cfg, vanish=dataclasses.replace(cfg.vanish, mask_mode=mask_mode))

    if capture_mode is not None:
        cfg = dataclasses.replace(cfg, video=dataclasses.replace(cfg.video, capture_mode=capture_mode))

    log.info("System: cpu=%s | power mode=%s | power=%s | perf tweaks=%s",
             platform.processor(), winperf.power_mode(), winperf.power_source(),
             "on" if perf_tweaks else "off")
    if perf_tweaks:
        winperf.apply_perf_tweaks()

    cam = raw_cam = vcam = hands = segmenter = None
    install_exit_guard()
    try:
        hands_model = cfg.models.resolve("hand_landmarker")
        seg_model = cfg.models.resolve("selfie_segmenter")
        for path in (hands_model, seg_model):
            if not path.exists():
                raise SetupError(f"Model file missing: {path}\n"
                                 "  Run: python scripts/download_models.py")

        raw_cam = Camera(cfg.video.width, cfg.video.height, cfg.video.fps)
        # --lock-exposure overrides camera.exposure from config.yaml. Applied before the capture thread starts.
        exposure = lock_exposure if lock_exposure is not None else cfg.camera.exposure
        if exposure is not None:
            raw_cam.apply_exposure(exposure)
        else:
            raw_cam.apply_auto_exposure()  # explicit: the driver keeps a previous manual setting
        threaded = cfg.video.capture_mode == "threaded"
        cam = ThreadedCamera(raw_cam) if threaded else raw_cam
        # Output size follows the real camera resolution so frames are always sent unchanged.
        out_size = (cam.width, cam.height)
        vcam = VirtualCamera(cam.width, cam.height, cfg.video.fps, cfg.vcam.backend)
        log.info("Capture mode: %s", cfg.video.capture_mode)

        p = cfg.perception
        tracker = HandTracker(hands_model, num_hands=p.num_hands,
                              min_detection_confidence=p.hand_min_detection_confidence,
                              min_presence_confidence=p.hand_min_presence_confidence,
                              min_tracking_confidence=p.hand_min_tracking_confidence)
        hands = HandsWorker(tracker) if p.hands_async else tracker
        segmenter = Segmenter(seg_model, out_size, blur_px=p.mask_blur_px,
                              feather_low=p.mask_feather_low, feather_high=p.mask_feather_high,
                              temporal_alpha=p.mask_temporal_alpha, every_n=p.segment_every_n,
                              temporal_mode=p.mask_temporal, release_decay=p.mask_release_decay)
        sw = min(cfg.video.perception_width, cam.width)  # perception frame size, as downscale() makes it
        vmask = VanishMask(cfg.vanish, out_size, (sw, max(1, round(cam.height * sw / cam.width))))
        compositor = Compositor()
        plate_dir = cfg.plate.resolve_dir()
        plate = load_plate(plate_dir, cam.width, cam.height)
        if plate:
            m = plate.meta
            log.info("Loaded saved plate FROM A PREVIOUS SESSION: %dx%d, %d frames, exposure %s, mean brightness "
                     "%.1f, taken %s. It must pass the health check against the room before vanish will use it; "
                     "press B to capture a fresh one.",
                     m.width, m.height, m.frames,
                     "auto" if m.exposure_value is None else f"manual {m.exposure_value:.0f}",
                     m.mean_brightness, m.timestamp)
        else:
            log.info("No usable plate in %s: press B (with the preview focused) to capture one.", plate_dir)
        capture = PlateCapture(cfg.plate.frames, cfg.plate.countdown_s, cfg.plate.settle_s,
                               cfg.plate.max_person_coverage)
        cues = SoundCues(cfg.plate.sounds)
        limits = HealthLimits(cfg.plate.health_max_raw_diff, cfg.plate.health_gain_min,
                              cfg.plate.health_gain_max, cfg.plate.health_max_cell_gain)
        vanish_on = False
        # V (or --vanish-on-start, for headless timing runs) only asks; the health check decides.
        want_vanish = bool(vanish_on_start and plate)
        previous_session = plate is not None  # a loaded plate is flagged until it passes a health check
        health = None             # latest PlateHealth verdict for the current plate (None = not judged yet)
        health_at = -1e9
        unhealthy_since = None
        shots = Path(shots_dir) if shots_dir else None  # local-only evidence images (gitignored)
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
        shot_no = 0
        vresult = None           # latest VanishResult, for the debug view
        if plate:
            vmask.set_plate(plate.u8)
        status_msg, status_until = "", 0.0
        log.info("Perception ready: perception_width=%d, hands=%d (%s), segment every %d frame(s), preview every %d frame(s)",
                 cfg.video.perception_width, p.num_hands, "async thread" if p.hands_async else "inline",
                 p.segment_every_n, cfg.preview.every_n)

        if show_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
        evidence = EvidenceSession(evidence_dir) if evidence_dir and show_preview else None

        fps_counter, timer = FpsCounter(), StageTimer()
        rx_first = rx_last = None  # read-completion times, for the camera delivery rate
        rx_count = 0
        proc = psutil.Process()
        proc.cpu_percent(None)  # prime the counter
        start = last_log = last_frame = time.perf_counter()
        frames = frames_since_log = 0
        hud_mode = DEBUG_VIEWS.index(debug_view)  # D cycles off -> sources -> leak
        last_stage_ms: dict[str, float] = {}
        log.info("Running. Q quits, D toggles the debug HUD, B captures a clean plate, V toggles vanish (applied to the vcam output too; "
                 "B/V/D/S/Q are read from the preview window). mask_mode=%s", cfg.vanish.mask_mode)

        while True:
            frame = cam.read(timer)
            now = time.perf_counter()
            dt = now - last_frame  # seconds since the previous frame; effects must use this
            last_frame = now
            rx_last = now
            if rx_first is None:
                rx_first = rx_last
            rx_count += 1

            with timer.stage("downscale"):
                small = downscale(frame, cfg.video.perception_width)
            if p.hands_async:
                # Never waits: hand the newest frame to the worker, use whatever result is newest.
                hands.submit(small, getattr(cam, "frame_time", now))
                hand_result = hands.latest()
                hand_list = hand_result.hands
                if hand_result.seq:
                    timer.add("hands_age", time.perf_counter() - hand_result.frame_time)
            else:
                with timer.stage("hands"):
                    hand_list = hands.detect(small)
            with timer.stage("segment"):
                mask = segmenter.segment(small)

            if capture.active:
                capture.update(frame, mask, now)  # raw frame, never the composited one
                if capture.state == capture.DONE:
                    plate = make_plate(capture.result(), cfg.plate.frames, exposure, capture.peak_coverage)
                    try:
                        saved = save_plate(plate, plate_dir)
                        log.info("Plate captured and saved (local-only): %s | %d frames | mean brightness %.1f | "
                                 "exposure %s | peak person coverage %.2f%%", saved, plate.meta.frames,
                                 plate.meta.mean_brightness, plate.meta.exposure_mode,
                                 capture.peak_coverage * 100)
                    except OSError as exc:
                        log.error("Plate captured but could not be saved: %s", exc)
                    if shots:
                        cv2.imwrite(str(shots / "plate.png"), plate.u8)
                    vmask.set_plate(plate.u8)
                    capture.state = capture.IDLE
                    status_msg, status_until = "Plate captured. Press V to vanish.", now + 4.0
                    cues.play("ok")
                    previous_session, health, health_at, unhealthy_since = False, None, -1e9, None
            for cue in capture.pop_events():
                cues.play(cue)

            # Plate health: checked every health_check_s whenever a plate exists, and at once when
            # vanish is asked for. A plate that does not match the room never reaches the output.
            if plate and not capture.active and segmenter.small is not None and (
                    want_vanish or now - health_at >= cfg.plate.health_check_s):
                health_at = now
                with timer.stage("health"):
                    checked = vmask.check_health(small, segmenter.small, hand_list, limits)
                if checked is not None:
                    changed = health is None or checked.ok != health.ok
                    health = checked
                    if health.ok:
                        unhealthy_since = None
                        if previous_session:
                            previous_session = False
                            log.info("Plate from a previous session passed the health check: %s", health.describe())
                        elif changed:
                            log.info("Plate health: %s", health.describe())
                    else:
                        if changed:
                            log.warning("Plate health%s: %s",
                                        " (plate from a previous session)" if previous_session else "",
                                        health.describe())
                        if unhealthy_since is None:
                            unhealthy_since = now
                        if vanish_on and now - unhealthy_since >= cfg.plate.health_fail_s:
                            vanish_on = False
                            log.warning("Vanish switched OFF: the plate no longer matches the room (%s). "
                                        "Press B to capture a new plate.", health.describe())
                            status_msg, status_until = PLATE_MISMATCH, now + 6.0
                            cues.play("error")
                if want_vanish:
                    want_vanish = False
                    if checked is None:
                        log.warning("Vanish refused: too little background visible to check the plate against the room.")
                        status_msg, status_until = "Can't check the plate: not enough background visible", now + 4.0
                        cues.play("error")
                    elif not checked.ok:
                        log.warning("Vanish refused: the plate%s doesn't match the room (%s). Press B to capture a new plate.",
                                    " from a previous session" if previous_session else "", checked.describe())
                        status_msg, status_until = PLATE_MISMATCH, now + 6.0
                        cues.play("error")
                    else:
                        vanish_on = True
                        vmask.stats.reset()
                        vmask.reset_temporal()
                        log.info("Vanish ON (plate health: %s)", checked.describe())
            vresult = None
            if vanish_on and plate and not capture.active:
                vresult = vmask.compute(small, segmenter.small, hand_list, timer)
                if health is not None and health.ok:  # background refresh, outside the near-me zone only
                    with timer.stage("vm_refresh"):
                        vmask.refresh_plate(frame)
                with timer.stage("composite"):
                    out = (compositor.render(frame, vresult.mask, vresult.plate, dt, vanish=True, roi=vresult.roi)
                           if vresult.roi else frame)
            else:
                out = frame
            with timer.stage("vcam"):
                vcam.send(out)  # clean frame: no debug drawing ever reaches the virtual camera
            fps_counter.tick()
            frames += 1
            frames_since_log += 1

            if show_preview and frames % cfg.preview.every_n == 0:
                t_preview = time.perf_counter()
                preview = out.copy()  # all debug drawing happens on this copy only
                if DEBUG_VIEWS[hud_mode] == "sources":
                    if vresult is not None:
                        overlay.draw_vanish_sources(preview, vresult)
                    overlay.draw_debug(preview, None if vresult is not None else mask, hand_list, last_stage_ms)
                elif DEBUG_VIEWS[hud_mode] == "leak":
                    if vresult is not None and vresult.roi:
                        overlay.draw_leak(preview, out, vresult.plate, vresult.roi)
                    else:
                        overlay.draw_status(preview, "Leak view needs vanish ON (press V)", row=1)
                overlay.draw_fps(preview, fps_counter.fps)
                if not plate:
                    plate_line = "plate: none (press B)"
                elif health is not None and not health.ok:
                    plate_line = ("plate from a previous session: " if previous_session else "") + PLATE_MISMATCH
                elif previous_session:
                    plate_line = "plate: from a previous session (not checked yet)"
                else:
                    plate_line = "plate: ready"
                state_line = ("VANISH ON" if vanish_on else "vanish off") + " | " + plate_line
                overlay.draw_status(preview, state_line)
                banner = capture.banner(now)
                if banner:
                    overlay.draw_banner(preview, *banner)
                elif status_msg and now < status_until:
                    overlay.draw_banner(preview, status_msg, warning=True)
                if evidence:
                    evidence.update(preview, lambda: threading.Thread(
                        target=grab_vcam_frames, args=(evidence.out_dir,), daemon=True).start())
                cv2.imshow(PREVIEW_WINDOW, preview)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q")):
                    log.info("Quit requested.")
                    break
                if key in (ord("["), ord("]")):
                    if exposure is None:
                        log.info("Exposure is auto; set camera.exposure or --lock-exposure to use [ and ]")
                    else:
                        exposure = min(EXPOSURE_MAX, max(EXPOSURE_MIN, exposure + (1 if key == ord("]") else -1)))
                        log.info("Exposure nudged to %.0f", exposure)
                        if threaded:
                            cam.request_exposure(exposure)
                        else:
                            cam.set_exposure(exposure)
                if key in (ord("b"), ord("B")):
                    if capture.active:
                        capture.cancel()
                        log.info("Plate capture cancelled.")
                    else:
                        vanish_on = want_vanish = False  # capture needs the raw feed; vcam keeps showing it live
                        capture.start(now)
                if key in (ord("s"), ord("S")):
                    if shots:
                        shot_no += 1
                        stem = f"vanished-{cfg.vanish.mask_mode}-{shot_no}" if vanish_on else f"live-with-me-{shot_no}"
                        cv2.imwrite(str(shots / f"{stem}.png"), out)  # the exact frame sent to the vcam: no HUD
                        cv2.imwrite(str(shots / f"{stem}-preview.png"), preview)  # with the debug view as shown
                        log.info("Saved screenshots (local-only): %s.png (vcam output) and %s-preview.png",
                                 shots / stem, stem)
                    else:
                        log.info("Start with --shots-dir DIR to use S for screenshots.")
                if key in (ord("v"), ord("V")):
                    if capture.active:
                        log.info("Ignoring V during plate capture.")
                    elif plate is None:
                        log.warning("No clean plate yet: press B first.")
                        status_msg, status_until = "No plate yet: press B first", now + 3.0
                        cues.play("error")
                    elif vanish_on:
                        vanish_on = False
                        log.info("Vanish OFF")
                    else:
                        want_vanish = True  # granted on the next frame if the plate passes the health check
                if key in (ord("d"), ord("D")):
                    hud_mode = (hud_mode + 1) % len(DEBUG_VIEWS)
                    log.info("Debug view: %s", DEBUG_VIEWS[hud_mode])
                if cv2.getWindowProperty(PREVIEW_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    log.info("Preview window closed.")
                    break
                if evidence and evidence.done:
                    log.info("Evidence capture finished.")
                    break
                timer.add("preview", time.perf_counter() - t_preview)

            now = time.perf_counter()
            if now - last_log >= FPS_LOG_INTERVAL_S:
                averages = {**(cam.timer.averages_ms() if threaded else {}), **timer.averages_ms()}
                hands_hz = hands_ms = None
                if p.hands_async:
                    hands_hz, hands_ms = hands.pop_stats()
                    averages["hands"] = hands_ms  # inference time, measured on the worker thread
                last_stage_ms = {k: averages[k] for k in ("hands", "segment") if k in averages}
                stages = " ".join(f"{k}={v:.1f}ms" for k, v in averages.items())
                if threaded:
                    rx_fps = cam.pop_delivery_fps()
                    cam.timer.reset()
                else:
                    rx_fps = (rx_count - 1) / (rx_last - rx_first) if rx_count > 1 and rx_last > rx_first else 0.0
                log.info("FPS %.1f (avg over %.1fs) | camera delivers %.1f fps | %s | cpu=%.0f%% (of one core)",
                         frames_since_log / (now - last_log), now - last_log, rx_fps, stages,
                         proc.cpu_percent(None))
                rx_first, rx_count = None, 0
                if p.hands_async:
                    log.info("hands worker: %.1f Hz, inference %.1f ms, result age %.1f ms (at main-loop use)",
                             hands_hz, hands_ms, averages.get("hands_age", 0.0))
                log.info("camera: exposure=%s (our setting; driver readback is unreliable) | power=%s",
                         "auto" if exposure is None else f"{exposure:.0f} manual", winperf.power_source())
                if vanish_on:
                    st = vmask.stats
                    if st.n:
                        n, rn = st.n, max(st.resid_n, 1)
                        g = st.gain / n
                        log.info("vanish lighting [%s]: background live %.1f | plate raw %.1f (diff %+.1f) -> "
                                 "matched %.1f (diff %+.1f) | per-cell residual mean|r| %.2f signed %+.2f max|r| %.1f "
                                 "(%d cells/frame) | gain BGR %.2f/%.2f/%.2f (cells %.2f..%.2f) offset %+.1f/%+.1f/%+.1f "
                                 "| background %.0f%%",
                                 cfg.vanish.plate_match, st.live / n, st.raw_plate / n,
                                 (st.live - st.raw_plate) / n, st.matched_plate / n,
                                 (st.live - st.matched_plate) / n, st.resid_mean_abs / rn, st.resid_signed / rn,
                                 st.resid_max_abs, st.cells // rn, *g, st.gain_min, st.gain_max,
                                 *(st.offset / n), st.fraction / n * 100)
                    else:
                        log.info("vanish lighting: not enough visible background to compare")
                    st = vmask.stats
                    gf = max(st.guard_frames, 1)
                    log.info("vanish mask: plate-diff frames %d | cell guard fired in %d of %d frames (%.1f cells when "
                             "it fired) | body-sized difference accepted outside the zone in %d frames | segmenter "
                             "empty in %d frames", st.diff_frames, st.guard_frames_any, st.guard_frames,
                             st.guard_cells / max(st.guard_frames_any, 1), st.large_blob_frames, st.seg_empty)
                    log.info("plate refresh: %d passes, each blended %.0f%% of the frame (alpha %.3f per frame, outside "
                             "the near-me zone + %d px) | paused %d (segmenter saw nobody) | plate health: %s",
                             st.refresh_n, st.refresh_coverage / max(st.refresh_n, 1) * 100,
                             cfg.vanish.plate_refresh_alpha, cfg.vanish.plate_refresh_guard_px, st.refresh_paused,
                             health.describe() if health else "not judged yet")
                    vmask.stats.reset()
                timer.reset()
                last_log, frames_since_log = now, 0
            if max_seconds is not None and now - start >= max_seconds:
                log.info("Reached --seconds limit.")
                break

        elapsed = time.perf_counter() - start
        log.info("Summary: %d frames in %.1fs = %.1f FPS average", frames, elapsed, frames / elapsed)
        return 0
    except (CameraError, VCamError, SetupError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return 0
    finally:
        for closable in (hands, segmenter, vcam):
            if closable:
                closable.close() if hasattr(closable, "close") else None
        # Release the camera on every path, even if setup failed before the wrapper existed; this
        # (plus the atexit hook in io/camera.py) restores auto exposure if we left it manual.
        for c in (cam, raw_cam if cam is None else None):
            if c:
                try:
                    c.release()
                except Exception:  # noqa: BLE001
                    log.exception("Camera release failed")
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
    parser.add_argument("--reset-camera", action="store_true",
                        help="write auto exposure to the camera, log brightness before/after, then exit "
                             "(use if the webcam looks dark after a crash)")
    parser.add_argument("--segment-every", type=int, metavar="N",
                        help="override perception.segment_every_n from config.yaml")
    parser.add_argument("--capture-mode", choices=("inline", "threaded"),
                        help="override video.capture_mode from config.yaml")
    parser.add_argument("--evidence", metavar="DIR",
                        help="guided capture of 3 preview screenshots + virtual camera frames into DIR")
    parser.add_argument("--mask-mode", choices=("seg_only", "seg_hands", "full"),
                        help="override vanish.mask_mode from config.yaml (A/B the vanish mask sources)")
    parser.add_argument("--vanish-on-start", action="store_true",
                        help="start with vanish ON when a saved plate exists and passes the health check "
                             "(for headless timing runs)")
    parser.add_argument("--debug-view", choices=DEBUG_VIEWS, default="sources",
                        help="initial preview debug view (D cycles off -> sources -> leak)")
    parser.add_argument("--shots-dir", metavar="DIR",
                        help="local-only evidence: S saves the clean vcam frame here (vanished.png when V is on, "
                             "else live-with-me.png); plate.png is saved on each plate capture")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    if args.reset_camera:
        return reset_camera(args.config)
    if args.camera_settings:
        return show_camera_settings(args.config)
    return run(args.config, args.seconds, not args.no_preview, args.lock_exposure,
               not args.no_perf_tweaks, args.segment_every, args.evidence, args.capture_mode,
               args.shots_dir, args.mask_mode,
               args.vanish_on_start, args.debug_view)


if __name__ == "__main__":
    raise SystemExit(main())
