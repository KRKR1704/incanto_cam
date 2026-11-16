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

from incanto.config import VANISH_STYLES, ConfigError, load_config
from incanto.guided import GuidedPlan, GuidedSession, PlanError, load_plan, write_labels, LABELS_PATH
from incanto.io.audio import print_devices, start_microphone
from incanto.io.recorder import ClipRecorder, RecordingError
from incanto.io.replay import Clip, ClipError, EndOfClip, ReplayAudio, ReplaySource
from incanto.io.camera import (EXPOSURE_MAX, EXPOSURE_MIN, Camera, CameraError, ThreadedCamera,
                               install_exit_guard)
from incanto.io.exposure_lock import ExposureLock
from incanto.io.sounds import SoundCues
from incanto.io.vcam import VCamError, VirtualCamera
from incanto.perception.hands import HandsWorker, HandTracker
from incanto.plate.clean_plate import (PlateCapture, load_plate, make_plate, save_abort_image, save_plate)
from incanto.plate.health import HealthLimits, HealthMonitor, StalenessNudge
from incanto.perception.segmenter import Segmenter
from incanto.render import overlay
from incanto.render.vanish_mask import VanishMask
from incanto.render.vanish_style import VanishStyler
from incanto.utils import winperf
from incanto.utils.evidence import EvidenceSession, grab_vcam_frames
from incanto.utils.fps import FpsCounter, StageTimer
from incanto.utils.imaging import downscale
from incanto.utils.logging import setup_logging

log = logging.getLogger("incanto")
PREVIEW_WINDOW = "Incanto preview (Q quit, D debug view, B capture plate, Shift+B force it, V vanish, G vanish style, R record)"
DEBUG_VIEWS = ("off", "sources", "leak")
FPS_LOG_INTERVAL_S = 5.0
PLATE_MISMATCH = "Plate doesn't match the room - press B"
LIGHTING_CHANGED = "Lighting changed - press B to refresh plate"


def _shift_held() -> bool:
    """True while a Shift key is down (Windows). cv2.waitKey only reports 'B', which Caps Lock also gives."""
    try:
        import ctypes
        return bool(ctypes.windll.user32.GetKeyState(0x10) & 0x8000)
    except (AttributeError, OSError):
        return False


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
        debug_view: str = "sources", vanish_style: str | None = None,
        replay: str | None = None, realtime: bool = True, render_out: str | None = None,
        record_guided: str | None = None, record_codec: str | None = None) -> int:
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

    if vanish_style is not None:
        cfg = dataclasses.replace(cfg, vanish=dataclasses.replace(cfg.vanish, style=vanish_style))

    if capture_mode is not None:
        cfg = dataclasses.replace(cfg, video=dataclasses.replace(cfg.video, capture_mode=capture_mode))

    log.info("System: cpu=%s | power mode=%s | power=%s | perf tweaks=%s",
             platform.processor(), winperf.power_mode(), winperf.power_source(),
             "on" if perf_tweaks else "off")
    if perf_tweaks:
        winperf.apply_perf_tweaks()

    if record_codec is not None:
        cfg = dataclasses.replace(cfg, recording=dataclasses.replace(cfg.recording, codec=record_codec))
    replaying = replay is not None
    plan = None
    if record_guided:
        try:
            plan = load_plan(record_guided)
        except PlanError as exc:
            log.error("%s", exc)
            return 2
        if replaying or not show_preview:
            log.error("--record-guided needs the live camera and the preview window (the prompts are drawn there)")
            return 2
    cam = raw_cam = vcam = hands = segmenter = recorder = mic = render_writer = None
    install_exit_guard()
    try:
        hands_model = cfg.models.resolve("hand_landmarker")
        seg_model = cfg.models.resolve("selfie_segmenter")
        for path in (hands_model, seg_model):
            if not path.exists():
                raise SetupError(f"Model file missing: {path}\n"
                                 "  Run: python scripts/download_models.py")

        if replaying:
            clip = Clip.load(replay)
            cam = ReplaySource(clip, realtime)
            exposure, threaded = None, False
            out_size = (cam.width, cam.height)
            log.info("REPLAY of %s: %d frames over %.1fs (%dx%d), audio %s | %s | hands run synchronously for "
                     "determinism; no webcam, microphone or virtual camera is opened", clip.path, clip.n_frames,
                     clip.duration, cam.width, cam.height,
                     f"{clip.audio().size / clip.samplerate:.1f}s" if clip.has_audio else "none",
                     "paced by the recorded times (--realtime)" if realtime else "as fast as possible (--fast)")
        else:
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
        async_hands = p.hands_async and not replaying   # replay: synchronous, so runs are deterministic
        tracker = HandTracker(hands_model, num_hands=p.num_hands,
                              min_detection_confidence=p.hand_min_detection_confidence,
                              min_presence_confidence=p.hand_min_presence_confidence,
                              min_tracking_confidence=p.hand_min_tracking_confidence)
        hands = HandsWorker(tracker) if async_hands else tracker
        segmenter = Segmenter(seg_model, out_size, blur_px=p.mask_blur_px,
                              feather_low=p.mask_feather_low, feather_high=p.mask_feather_high,
                              temporal_alpha=p.mask_temporal_alpha, every_n=p.segment_every_n,
                              temporal_mode=p.mask_temporal, release_decay=p.mask_release_decay)
        sw = min(cfg.video.perception_width, cam.width)  # perception frame size, as downscale() makes it
        vmask = VanishMask(cfg.vanish, out_size, (sw, max(1, round(cam.height * sw / cam.width))))
        styler = VanishStyler(cfg.vanish, vmask, out_size)  # what is shown while vanished (freeze | region | live)
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
        log.info("Vanish style: %s (G cycles %s); transition %.2fs; grain x%.2f", styler.style,
                 " > ".join(VANISH_STYLES), cfg.vanish.transition_s, cfg.vanish.grain_scale)
        capture = PlateCapture(cfg.plate.frames, cfg.plate.countdown_s, cfg.plate.settle_s,
                               cfg.plate.max_person_coverage, cfg.plate.static_blob_max_motion)
        cues = SoundCues(cfg.plate.sounds)
        pc = cfg.plate
        limits = HealthLimits(pc.health_max_residual, pc.health_min_structure, pc.health_gain_min,
                              pc.health_gain_max, pc.health_min_cell_background, pc.health_hysteresis)
        monitor = HealthMonitor(pc.health_fail_s)  # latest verdict, hysteresis, sustained-failure timer
        nudge = StalenessNudge(pc.health_nudge_residual, pc.health_nudge_s)  # soft "lighting changed" hint
        health_at = -1e9
        vanish_on = False
        # V (or --vanish-on-start, for headless timing runs) only asks; the health check decides.
        want_vanish = bool(vanish_on_start and plate)
        previous_session = plate is not None  # a loaded plate is flagged until it passes a health check

        def set_camera_exposure(value: float | None) -> None:  # None = auto; used by the exposure lock
            if threaded:
                cam.request_auto_exposure() if value is None else cam.request_exposure(value)
            else:
                cam.set_auto_exposure() if value is None else cam.set_exposure(value)

        # Exposure lock only when the camera runs on auto: a manual exposure from the config is already fixed.
        lock = (ExposureLock(set_camera_exposure, pc.lock_exposure_tolerance, pc.lock_exposure_settle_s)
                if pc.lock_exposure and exposure is None and not replaying else None)
        shots = Path(shots_dir) if shots_dir else None  # local-only evidence images (gitignored)
        if shots:
            shots.mkdir(parents=True, exist_ok=True)
        shot_no = 0
        vresult = None           # latest VanishResult, for the debug view
        if plate:
            vmask.set_plate(plate.u8)
            styler.set_plate(plate)
        status_msg, status_until = "", 0.0
        log.info("Perception ready: perception_width=%d, hands=%d (%s), segment every %d frame(s), preview every %d frame(s)",
                 cfg.video.perception_width, p.num_hands, "async thread" if async_hands else "inline",
                 p.segment_every_n, cfg.preview.every_n)

        # microphone (live) or the recorded audio fed back on the recorded clock (replay)
        audio_feed = ReplayAudio(clip, cfg.audio.ring_seconds, cfg.audio.offset_ms) if replaying else None
        if replaying:
            if not clip.has_audio:
                log.info("This clip has no audio.")
            mic_ring = audio_feed.ring if audio_feed.available else None
        else:
            mic = start_microphone(cfg.audio)
            mic_ring = mic.ring if mic else None

        def new_recorder(name: str | None = None) -> ClipRecorder:
            return ClipRecorder(cfg.recording.resolve_dir(), out_size, cfg.video.fps, cfg.recording.codec, mic,
                                cfg.recording.queue_frames, cfg.recording.min_free_gb, name)

        guided = None
        guided_done = False
        if render_out:
            ext = Path(render_out).suffix.lower()
            Path(render_out).parent.mkdir(parents=True, exist_ok=True)
            render_writer = cv2.VideoWriter(render_out, cv2.VideoWriter_fourcc(*("mp4v" if ext == ".mp4" else "MJPG")),
                                            float(clip.nominal_fps), out_size)
            if not render_writer.isOpened():
                raise SetupError(f"cannot write {render_out} (use a .mp4 or .avi path)")
            log.info("Rendering the composited output to %s (local-only: never commit it; no audio is written)", render_out)

        poll_key = getattr(cv2, "pollKey", None) or (lambda: cv2.waitKey(1))  # pollKey: OpenCV >= 4.5.4
        preview_size = None  # None = full size
        if cfg.preview.scale < 1:
            preview_size = (max(2, round(cam.width * cfg.preview.scale)), max(2, round(cam.height * cfg.preview.scale)))
        if show_preview:
            cv2.namedWindow(PREVIEW_WINDOW, cv2.WINDOW_AUTOSIZE)
        evidence = EvidenceSession(evidence_dir) if evidence_dir and show_preview else None

        fps_counter, timer = FpsCounter(), StageTimer()
        rx_first = rx_last = None  # read-completion times, for the camera delivery rate
        rx_count = 0
        proc = psutil.Process()
        proc.cpu_percent(None)  # prime the counter
        start = last_log = last_frame = time.perf_counter()
        if replaying:
            last_frame = clip.first_time - 1.0 / clip.nominal_fps  # the first dt is one nominal frame
        frames = frames_since_log = 0
        hud_mode = DEBUG_VIEWS.index(debug_view)  # D cycles off -> sources -> leak
        last_stage_ms: dict[str, float] = {}
        log.info("Running. Q quits, D toggles the debug HUD, B captures a clean plate (Shift+B: without the person "
                 "check), V toggles vanish (applied to the vcam output too; "
                 "B/V/D/S/Q are read from the preview window). mask_mode=%s", cfg.vanish.mask_mode)

        while True:
            try:
                frame = cam.read(timer)
            except EndOfClip:
                log.info("Replay finished: all %d frames.", clip.n_frames)
                break
            now = cam.frame_time if replaying else time.perf_counter()   # replay runs on the recorded clock
            frame_t = cam.frame_time if threaded or replaying else now    # when this frame was captured
            dt = now - last_frame  # seconds since the previous frame; effects must use this
            last_frame = now
            rx_last = time.perf_counter()
            if rx_first is None:
                rx_first = rx_last
            rx_count += 1

            with timer.stage("downscale"):
                small = downscale(frame, cfg.video.perception_width)
            if async_hands:
                # Never waits: hand the newest frame to the worker, use whatever result is newest.
                hands.submit(small, getattr(cam, "frame_time", now))
                hand_result = hands.latest()
                hand_list = hand_result.hands
                if hand_result.seq:
                    timer.add("hands_age", time.perf_counter() - hand_result.frame_time)
            else:
                with timer.stage("hands"):
                    hand_list = hands.detect(small, cam.timestamp_ms() if replaying else None)
            with timer.stage("segment"):
                mask = segmenter.segment(small, cam.timestamp_ms() if replaying else None)
            if audio_feed is not None:
                audio_feed.feed_until(now)
            if plan is not None and not guided_done:
                if guided is None:  # first frame: start recording and the plan together
                    try:
                        recorder = new_recorder("guided-" + time.strftime("%Y%m%d-%H%M%S"))
                        recorder.start()
                    except RecordingError as exc:
                        log.error("%s", exc)
                        return 1
                    guided = GuidedSession(plan, now)
                    log.info("Guided recording '%s' started: %s | %s | about %.0f s. Follow the prompts in the preview "
                             "window; the beep is the cue. Q aborts.", plan.name, plan.counts(),
                             f"{len(plan.talks)} talk segment(s)", plan.duration_s)
                for event in guided.update(now):
                    cues.play(event)
                if guided.finished:
                    guided_done = True

            if capture.active:
                capture.update(frame, mask, now)  # raw frame, never the composited one
                if capture.state == capture.DONE:
                    plate = make_plate(capture.result(), cfg.plate.frames, exposure, capture.peak_coverage,
                                       capture.noise())
                    try:
                        saved = save_plate(plate, plate_dir)
                        log.info("Plate captured and saved (local-only): %s | %d frames | mean brightness %.1f | "
                                 "exposure %s | %s", saved, plate.meta.frames,
                                 plate.meta.mean_brightness, plate.meta.exposure_mode,
                                 "person check SKIPPED (forced)" if capture.force else
                                 "moving person coverage %.2f%% | segmenter coverage %.1f%%, of which %.1f%% did "
                                 "not move and was treated as furniture" % (
                                     capture.peak_coverage * 100, capture.raw_coverage * 100,
                                     capture.static_coverage * 100))
                    except OSError as exc:
                        log.error("Plate captured but could not be saved: %s", exc)
                    if shots:
                        cv2.imwrite(str(shots / "plate.png"), plate.u8)
                    vmask.set_plate(plate.u8)
                    styler.set_plate(plate)
                    log.info("Plate noise profile: %s", "mean temporal std %.2f levels (BGR %.2f/%.2f/%.2f)" % (
                        plate.noise.mean(), *plate.noise.mean(axis=(0, 1))) if plate.noise is not None else "none")
                    capture.state = capture.IDLE
                    status_msg, status_until = "Plate captured. Press V to vanish.", now + 4.0
                    cues.play("ok")
                    previous_session, health_at = False, -1e9
                    monitor.reset()
                    nudge.reset()
                    if lock:
                        lock.start(now)
                elif capture.state == capture.ABORTED:
                    try:
                        shot = save_abort_image(capture, pc.resolve_debug_dir())
                        if shot:
                            log.warning("Capture abort debug image (local-only, never committed): %s | red = counted "
                                        "as a person (moving), yellow = still blob ignored as furniture", shot)
                    except (OSError, cv2.error) as exc:
                        log.error("Could not save the capture abort debug image: %s", exc)
                    if lock:
                        lock.restore()  # the old plate (if any) stays in use, and so does its exposure
            for cue in capture.pop_events():
                cues.play(cue)
            if lock and lock.sweeping:
                # the room is normally still empty; background-only brightness also works if you walk back in
                gain = vmask.background_gain(small, segmenter.small, hand_list) if segmenter.small is not None else None
                if lock.update(gain, now):
                    status_msg = (f"Exposure locked at {lock.value:g}. Press V to vanish." if lock.value is not None
                                  else "Exposure stays on auto. Press V to vanish.")
                    status_until, health_at = now + 4.0, -1e9

            # Plate health, judged after gain matching: checked every health_check_s whenever a plate
            # exists, and at once when vanish is asked for. Not during the exposure sweep (the picture is
            # deliberately too dark or too bright then); a V pressed meanwhile waits for it.
            if plate and not capture.active and segmenter.small is not None and not (lock and lock.sweeping) and (
                    want_vanish or now - health_at >= pc.health_check_s):
                health_at = now
                with timer.stage("health"):
                    checked = vmask.check_health(small, segmenter.small, hand_list, limits,
                                                 recovering=monitor.recovering)
                was_ok = monitor.ok
                changed = monitor.update(checked, now)
                if checked is not None:
                    if monitor.ok:
                        if previous_session:
                            previous_session = False
                            log.info("Plate from a previous session passed the health check: %s", checked.describe())
                        elif changed:
                            log.info("Plate health: %s", checked.describe())
                    else:
                        if changed:
                            log.warning("Plate health%s: %s (%s)",
                                        " (plate from a previous session)" if previous_session else "",
                                        checked.describe(),
                                        "freeze style: vanish stays on, the plate is re-matched before you reappear"
                                        if styler.style == "freeze" and vanish_on else
                                        "vanish switches off if this lasts %.0fs" % pc.health_fail_s)
                        # freeze shows a fixed picture: lighting drift is no reason to drop it mid-session
                        if vanish_on and styler.style != "freeze" and monitor.tripped(now):
                            vanish_on = False
                            log.warning("Vanish switched OFF: the plate has not matched the room for %.0fs (%s). "
                                        "Press B to capture a new plate.", pc.health_fail_s, checked.describe())
                            status_msg, status_until = PLATE_MISMATCH, now + 6.0
                            cues.play("error")
                if nudge.update(checked, now, vanish_on):
                    log.warning("Lighting changed: matched residual has been above %.1f for %.0fs (%s). Vanish stays "
                                "on; press B to refresh the plate.", pc.health_nudge_residual, pc.health_nudge_s,
                                checked.describe())
                    status_msg, status_until = LIGHTING_CHANGED, now + 6.0
                    cues.play("nudge")
                if want_vanish:
                    want_vanish = False
                    # No verdict (you fill the frame): trust the last one if this plate has passed before.
                    grant = checked.ok if checked is not None else (was_ok and not previous_session)
                    if grant:
                        vanish_on = True
                        vmask.stats.reset()
                        vmask.reset_temporal()
                        log.info("Vanish ON (plate health: %s)", checked.describe() if checked else
                                 "not enough background to re-check; the last check was healthy")
                    elif checked is None:
                        log.warning("Vanish refused: too little background visible to check the plate against the room.")
                        status_msg, status_until = "Can't check the plate: not enough background visible", now + 4.0
                        cues.play("error")
                    else:
                        log.warning("Vanish refused: the plate%s doesn't match the room (%s). Press B to capture a new plate.",
                                    " from a previous session" if previous_session else "", checked.describe())
                        status_msg, status_until = PLATE_MISMATCH, now + 6.0
                        cues.play("error")
            vresult = None
            if plate and segmenter.small is not None:
                # the styler follows vanish_on itself (crossfades on V, freezes or masks), see render/vanish_style.py
                out, vresult = styler.render(frame, small, segmenter.small, hand_list,
                                             vanish_on and not capture.active, dt, timer, refresh=monitor.ok)
            else:
                out = frame
            with timer.stage("vcam"):
                if vcam:
                    vcam.send(out)  # clean frame: no debug drawing ever reaches the virtual camera
            if render_writer is not None:
                render_writer.write(out)
            if recorder is not None and recorder.active:
                recorder.submit(frame, frame_t)  # the RAW camera frame, before any effect
            fps_counter.tick()
            frames += 1
            frames_since_log += 1

            if show_preview and frames % cfg.preview.every_n == 0:
                t_preview = time.perf_counter()
                # all debug drawing happens on this copy only; at preview.scale < 1 it is a smaller copy
                preview = out.copy() if preview_size is None else cv2.resize(out, preview_size, interpolation=cv2.INTER_AREA)
                if DEBUG_VIEWS[hud_mode] == "sources":
                    if vresult is not None:
                        overlay.draw_vanish_sources(preview, vresult)
                    overlay.draw_debug(preview, None if vresult is not None else mask, hand_list, last_stage_ms)
                elif DEBUG_VIEWS[hud_mode] == "leak":
                    if vresult is not None and vresult.roi:
                        overlay.draw_leak(preview, out, vresult.plate, vresult.roi)
                    elif styler.style == "freeze" and styler.active:
                        overlay.draw_status(preview, "Leak view: freeze has no mask, so nothing can leak (G: region/live)", row=1)
                    else:
                        overlay.draw_status(preview, "Leak view needs vanish ON (press V)", row=1)
                overlay.draw_fps(preview, fps_counter.fps)
                if recorder is not None and recorder.active:
                    overlay.draw_rec(preview, recorder.elapsed, recorder.dropped)
                if mic_ring is not None:
                    overlay.draw_audio_level(preview, mic_ring.level_dbfs())
                if guided is not None and guided.prompt is not None:
                    overlay.draw_cue(preview, guided.prompt.text, guided.prompt.sub, guided.prompt.go)
                if not plate:
                    plate_line = "plate: none (press B)"
                elif monitor.health is not None and not monitor.health.ok and (not vanish_on or monitor.tripped(now)):
                    plate_line = ("plate from a previous session: " if previous_session else "") + PLATE_MISMATCH
                elif previous_session:
                    plate_line = "plate: from a previous session (not checked yet)"
                else:
                    plate_line = "plate: ready"
                state_line = (("VANISH ON" if vanish_on else "vanish off") + f" [{styler.style}]" + " | " + plate_line)
                if vanish_on and nudge.showing:
                    overlay.draw_status(preview, LIGHTING_CHANGED, row=1)
                overlay.draw_status(preview, state_line)
                banner = capture.banner(now)
                if banner:
                    overlay.draw_banner(preview, *banner)
                elif status_msg and now < status_until:
                    overlay.draw_banner(preview, status_msg, warning=True)
                if evidence:
                    evidence.update(preview, lambda: threading.Thread(
                        target=grab_vcam_frames, args=(evidence.out_dir,), daemon=True).start())
                t_show = time.perf_counter()
                timer.add("pv_draw", t_show - t_preview)
                cv2.imshow(PREVIEW_WINDOW, preview)
                t_key = time.perf_counter()
                timer.add("pv_show", t_key - t_show)
                # pollKey paints the window and reads keys without sleeping; waitKey(1) sleeps until the next
                # timer tick (measured 1.4-8.7 ms per shown frame here)
                key = poll_key() & 0xFF
                timer.add("pv_key", time.perf_counter() - t_key)
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
                if key in (ord("r"), ord("R")):
                    if replaying:
                        log.info("R records the live camera; this is a replay.")
                    elif plan is not None:
                        log.info("A guided recording is running; it stops by itself (Q aborts).")
                    elif recorder is not None and recorder.active:
                        summary = recorder.stop()
                        status_msg, status_until = f"Recording saved: {summary.frames} frames", now + 4.0
                    else:
                        try:
                            recorder = new_recorder()
                            recorder.start()
                        except RecordingError as exc:
                            log.error("%s", exc)
                            status_msg, status_until = str(exc), now + 6.0
                            cues.play("error")
                            recorder = None
                if key in (ord("b"), ord("B")) and (replaying or plan is not None):
                    log.info("B (plate capture) needs the live camera and is off during replay and guided recording.")
                elif key in (ord("b"), ord("B")):
                    if capture.active:
                        capture.cancel()
                        log.info("Plate capture cancelled.")
                        if lock:
                            lock.restore()
                    else:
                        vanish_on = want_vanish = False  # capture needs the raw feed; vcam keeps showing it live
                        if lock:
                            lock.release()  # back to auto; the countdown and settle time let it adapt
                        capture.start(now, force=_shift_held())  # Shift+B: capture without the person check
                if key in (ord("s"), ord("S")):
                    if shots:
                        shot_no += 1
                        stem = f"vanished-{styler.style}-{cfg.vanish.mask_mode}-{shot_no}" if vanish_on else f"live-with-me-{shot_no}"
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
                        if styler.style == "freeze" and segmenter.small is not None:
                            # reappear safety: was the room changed while the frozen picture hid it?
                            with timer.stage("health"):
                                end = vmask.check_health(small, segmenter.small, hand_list, limits)
                            if end is not None and end.structure is not None and end.structure < limits.min_structure:
                                log.warning("Reappearing, but the room no longer matches the plate (structure %.2f below "
                                            "%.2f: something was moved while you were invisible?). Fading back over %.1fs "
                                            "instead of %.1fs. Press B for a new plate. (%s)", end.structure,
                                            limits.min_structure, cfg.vanish.reappear_fade_s, cfg.vanish.transition_s,
                                            end.describe())
                                styler.slow_exit(cfg.vanish.reappear_fade_s)
                            elif end is not None:
                                log.info("Reappearing: room still matches the plate (structure %s)",
                                         "n/a" if end.structure is None else f"{end.structure:.2f}")
                    else:
                        want_vanish = True  # granted on the next frame if the plate passes the health check
                if key in (ord("g"), ord("G")):
                    old = styler.style
                    log.info("Vanish style: %s -> %s%s", old, styler.cycle_style(),
                             " (vanish is on: switched at once, no crossfade between styles)" if vanish_on else "")
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

            if guided_done and recorder is not None and recorder.active:
                time.sleep(0.8)  # let the closing beep finish
                summary = recorder.stop()
                gestures, talks = guided.label_entries(summary.first_frame_time)
                write_labels(LABELS_PATH, summary.path.name, gestures, talks,
                             {"plan": plan.name, "frames": summary.frames, "dropped_frames": summary.dropped})
                log.info("Guided recording finished: %s | %d cues labelled, %d talk segment(s) | labels written to %s "
                         "(commit that file; the clip stays local-only)", summary.path, len(gestures), len(talks),
                         LABELS_PATH)
                if summary.dropped:
                    log.warning("%d frames were dropped while recording: the clip has gaps (see the sidecar). "
                                "Consider --record-codec mjpg and recording again.", summary.dropped)
                break

            now = time.perf_counter()
            if now - last_log >= FPS_LOG_INTERVAL_S:
                averages = {**(cam.timer.averages_ms() if threaded else {}), **timer.averages_ms()}
                hands_hz = hands_ms = None
                if async_hands:
                    hands_hz, hands_ms = hands.pop_stats()
                    averages["hands"] = hands_ms  # inference time, measured on the worker thread
                last_stage_ms = {k: averages[k] for k in ("hands", "segment") if k in averages}
                stages = " ".join(f"{k}={v:.1f}ms" for k, v in averages.items())
                if threaded:
                    rx_fps = cam.pop_delivery_fps()
                    cam.timer.reset()
                else:
                    rx_fps = (rx_count - 1) / (rx_last - rx_first) if rx_count > 1 and rx_last > rx_first else 0.0
                log.info("FPS %.1f (avg over %.1fs) | camera delivers %.1f fps | %s%s | cpu=%.0f%% (of one core)",
                         frames_since_log / (now - last_log), now - last_log, rx_fps, stages,
                         f" | vanish={styler.style}" if vanish_on else "",
                         proc.cpu_percent(None))
                rx_first, rx_count = None, 0
                if async_hands:
                    log.info("hands worker: %.1f Hz, inference %.1f ms, result age %.1f ms (at main-loop use)",
                             hands_hz, hands_ms, averages.get("hands_age", 0.0))
                log.info("camera: exposure=%s (our setting; driver readback is unreliable) | power=%s",
                         (lock.describe() if lock else "auto") if exposure is None else f"{exposure:.0f} manual",
                         winperf.power_source())
                if vanish_on and styler.style == "freeze":
                    log.info("vanish [freeze]: whole frame = gain-matched plate held fixed, grain %s | plate health: %s",
                             ("x%.2f of the measured noise" % cfg.vanish.grain_scale) if styler.grain_available
                             else "off (%s)" % ("grain_scale 0" if cfg.vanish.grain_scale <= 0 else "plate has no noise profile"),
                             monitor.health.describe() if monitor.health else "not judged yet")
                elif vanish_on:
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
                             "the mask + %d px) | paused %d (segmenter saw nobody) | plate health: %s",
                             st.refresh_n, st.refresh_coverage / max(st.refresh_n, 1) * 100,
                             cfg.vanish.plate_refresh_alpha, cfg.vanish.plate_refresh_margin_px, st.refresh_paused,
                             monitor.health.describe() if monitor.health else "not judged yet")
                    vmask.stats.reset()
                timer.reset()
                last_log, frames_since_log = now, 0
            if max_seconds is not None and now - start >= max_seconds:
                log.info("Reached --seconds limit.")
                break

        elapsed = time.perf_counter() - start
        log.info("Summary: %d frames in %.1fs = %.1f FPS average", frames, elapsed, frames / elapsed)
        return 0
    except (CameraError, VCamError, SetupError, ClipError, RecordingError) as exc:
        log.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        log.info("Interrupted.")
        return 0
    finally:
        if recorder is not None and recorder.active:
            try:
                recorder.stop()  # keep what was recorded even when the app exits mid-recording
                if plan is not None and not guided_done:
                    log.warning("Guided recording aborted before the end: no labels were written for this clip.")
            except Exception:  # noqa: BLE001
                log.exception("Stopping the recorder failed")
        if render_writer is not None:
            render_writer.release()
        if mic is not None:
            mic.stop()
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
    parser.add_argument("--vanish-style", choices=VANISH_STYLES,
                        help="override vanish.style from config.yaml (G cycles the styles in the preview)")
    parser.add_argument("--debug-view", choices=DEBUG_VIEWS, default="sources",
                        help="initial preview debug view (D cycles off -> sources -> leak)")
    parser.add_argument("--shots-dir", metavar="DIR",
                        help="local-only evidence: S saves the clean vcam frame here (vanished.png when V is on, "
                             "else live-with-me.png); plate.png is saved on each plate capture")
    parser.add_argument("--list-audio-devices", action="store_true",
                        help="list microphone inputs (virtual ones are marked), say which one would be used, then exit")
    parser.add_argument("--record-codec", choices=("lossless", "mjpg"),
                        help="override recording.codec (lossless = HFYU, bit-exact, 1.7 GB/min; mjpg = ~25x smaller)")
    parser.add_argument("--replay", metavar="CLIP",
                        help="replay a recorded clip folder (recordings/<clip>) through the full pipeline instead of the "
                             "webcam and microphone")
    pace = parser.add_mutually_exclusive_group()
    pace.add_argument("--realtime", action="store_true", help="replay paced by the recorded times (default)")
    pace.add_argument("--fast", action="store_true", help="replay as fast as the pipeline can go")
    parser.add_argument("--render-out", metavar="FILE",
                        help="with --replay: save the composited output (no audio) to a .mp4/.avi for local review; "
                             "never commit it")
    parser.add_argument("--record-guided", metavar="PLAN",
                        help="guided recording: prompts and beeps through a plan (e.g. plans/fixture-plan.yaml), records "
                             "the clip and writes tests/fixtures/labels.yaml")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    setup_logging(args.log_level)
    if args.list_audio_devices:
        return print_devices()
    if args.render_out and not args.replay:
        parser.error("--render-out needs --replay")
    if (args.fast or args.realtime) and not args.replay:
        parser.error("--realtime/--fast need --replay")
    if args.replay and args.record_guided:
        parser.error("--replay and --record-guided cannot be combined")
    if args.reset_camera:
        return reset_camera(args.config)
    if args.camera_settings:
        return show_camera_settings(args.config)
    return run(args.config, args.seconds, not args.no_preview, args.lock_exposure,
               not args.no_perf_tweaks, args.segment_every, args.evidence, args.capture_mode,
               args.shots_dir, args.mask_mode,
               args.vanish_on_start, args.debug_view, args.vanish_style,
               args.replay, not args.fast, args.render_out, args.record_guided, args.record_codec)


if __name__ == "__main__":
    raise SystemExit(main())
