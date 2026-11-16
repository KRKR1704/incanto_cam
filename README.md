# Incanto

Cast effects on your video calls with a gesture: a local, gesture-triggered video effects camera that outputs to a virtual camera (Google Meet, Zoom, Teams).

## Setup (Windows)

1. Install Python 3.10 or newer.
2. Install OBS Studio with default options. It is only needed for its virtual camera driver; you never need to open it.
3. Create a virtual environment from the project root:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   ```
4. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

5. **Always run with the venv's Python**: `.venv\Scripts\python -m incanto` (and `.venv\Scripts\python -m pytest`). The system `python` lacks the dependencies and fails with `No module named 'cv2'`.

See `docs/incanto-project-plan.md` and `docs/incanto-build-tasks.md` for the plan.

**Status: Phase 4 complete** (record and replay, with a labelled fixture clip). Phases 0-3 are complete (camera pipeline, perception, clean plate and vanish test).

## Vanish test (Phase 3)

Hotkeys work in the preview window: **B** captures a clean plate of the empty room (5 s countdown with a beep per second, leave the frame; a long tone means frames are being taken, three rising notes mean it was saved, two falling low notes mean it was aborted and nothing changed), **V** toggles vanish (applied to the virtual camera output too), **D** cycles the debug view (off, source colours, leak view where red shows you leaking through), **R** starts/stops a recording (see Record and replay below), **G** cycles the vanish style (freeze, region, live; each change is logged), **S** screenshot (with `--shots-dir`), **Q** quit.

`vanish.mask_mode` in `config.yaml` (or `--mask-mode`) picks what hides you: `seg_only` (segmenter), `seg_hands` (plus filled hand hulls) or `full` (plus pixels that differ from the brightness-matched plate). The plate is a photo of your room and is local-only (`plates/`, gitignored). Recapture it whenever furniture moves: in `full` mode anything that differs from the plate is replaced by the plate's version.

**Plate health.** Vanish only runs on a plate that matches the room. When you press V, and about twice a second after that, the plate is compared with the visible background *after* brightness matching, so auto-exposure swings (standing up, leaning in) do not count against it. Two things are judged, only in grid cells that are at least half background: the residual the gain map leaves per cell (`plate.health_max_residual`), and whether the edges of plate and live picture still line up (`plate.health_min_structure`, blind to brightness; catches a moved camera, rearranged furniture, another room). Global gain is only a loose sanity cap. If the check fails for `plate.health_fail_s` (3 s) vanish is switched off; on V it is refused straight away. Either way the preview shows "Plate doesn't match the room - press B", a WARNING is logged and the error sound plays. A plate loaded from a previous session is flagged as such until it passes this check. While vanish is on and the plate is healthy, the plate is slowly refreshed from the live picture outside the near-me zone (`vanish.plate_refresh_alpha`), so it follows daylight during a call; around you it stays as captured. `plate.sounds: false` mutes the beeps.

**Exposure lock.** With `camera.exposure: auto` and `plate.lock_exposure: true`, a successful capture is followed by a short sweep of manual exposures (only -5 and shorter, so the camera keeps 30 fps). The one closest to the plate's brightness is locked while the plate is in use; B returns to auto first, captures, then locks again. If nothing gets within `plate.lock_exposure_tolerance` of the plate's brightness (a dim room needs a longer exposure than 30 fps allows) the camera stays on auto and the log says why. The lock goes through the same safety net as a manual exposure: auto is restored on every exit path.

**Capture person check.** While you leave the frame, the capture watches every segmenter blob: one in which almost nothing moves during the last second (`plate.static_blob_max_motion`) is furniture, not you, and does not count towards `plate.max_person_coverage`. If the capture still aborts, a local-only debug image (`capture-abort-*.png`, in `plate.debug_dir`, never committed) shows what was counted: red = person, yellow = still blob ignored. **Shift+B** captures without the person check (a warning is logged).

**Lighting nudge.** If the matched residual stays above `plate.health_nudge_residual` (2.0) for `plate.health_nudge_s` (10 s) while vanished, you get one soft beep and "Lighting changed - press B to refresh plate"; vanish stays on. The plate refresh now reaches up to the mask plus `vanish.plate_refresh_margin_px` (15 px), but only on pixels every mask source calls background and that still match the gain-matched plate. `preview.scale` (default 0.5) draws the preview window at half size; the virtual camera output is never scaled.

**Vanish styles (3.9).** Any per-frame mask can leave an outline of you when you move fast, so `vanish.style` (default `freeze`; `--vanish-style` overrides it, **G** cycles it live) picks what is shown while vanished. The mask code is kept for all of them (Phase 6 needs it for the dust animation). The preview's status line shows the active style, e.g. `VANISH ON [freeze]`.
- `freeze`: the whole frame is the plate, with no mask at all, so no outline can appear however you move. When vanish turns on, the plate is gain-matched to the live frame once and then held fixed (no jump in brightness). Synthetic camera grain keeps it from looking frozen: during plate capture the per-pixel temporal std of the 30 frames is measured and stored with the plate (`plate-noise.png`, local-only like the plate); while frozen, fresh random grain matching that profile (scaled by brightness through the camera's measured noise-vs-luma curve) is added every frame. `vanish.grain_scale` scales it (0 = off). A plate saved before this has no profile: no grain until you press B. The mask is not computed, so this is the cheapest style (measured live: 29-30 FPS vanished). Health checks still refuse V on a stale plate, but in freeze lighting drift never switches vanish off mid-session (the soft "Lighting changed" nudge still shows). While vanish is off, a fresh gain match and the grain amplitudes are prepared a few milliseconds per frame, once a second, so pressing V does not stall a frame (`vs_begin` went from 41-58 ms to under 1 ms). **Reappearing:** just before you reappear the plate is gain-matched again to the live frame and the frozen picture moves onto that match during the fade. When V turns vanish off, the structure check runs once more; if it fails (the room no longer matches the plate) a WARNING is logged and the fade back takes `vanish.reappear_fade_s` (1.0 s) instead of `transition_s` (0.4 s).

**Tip: objects touched while invisible shift on reappear.** In freeze the room you see is the plate, so if you move or touch something while invisible (push the chair, pick up a cup), the call keeps seeing it where it was, and on reappearing it shifts to its new place over the fade. Move things back before you reappear, or accept the small jump; press B for a new plate if you rearranged the room for good.
- `region`: a large soft region shows the plate and the rest stays live (with the gain map and the plate refresh, which never touches the region). The region is the union of every mask source seen during the vanish session (max-hold, so it only grows), grown by `vanish.region_margin_px` (120) and feathered by `vanish.region_feather_px` (60). It resets when vanish ends.
- `live`: the per-frame mask composite, unchanged.

V on/off crossfades between the ordinary frame and the vanished output over `vanish.transition_s` (0.4 s, time-based using each frame's dt; 0 = hard cut). `live` keeps its hard cut. Phase 6 replaces this with the dissolve. Switching style with G while vanished takes effect at once, without a crossfade between styles.

`scripts/bench_vanish_styles.py` times each style on synthetic 1280x720 frames (run it with `PYTHONPATH=.`).

## Record and replay (Phase 4)

Everything here is local-only: clips, audio and renders live in `recordings/` (gitignored, and the pre-commit hook refuses video, audio and array files). Only `tests/fixtures/labels.yaml`, `.txt` reports and code are committed.

**Microphone.** `audio:` in `config.yaml`. The mic runs on its own stream (sounddevice, mono, 20 ms blocks) into a ring buffer stamped with `time.perf_counter()`, the same clock as the camera frames. `python -m incanto --list-audio-devices` lists inputs, marks virtual ones (Voicemod, VB-Audio, Steam Streaming, Stereo Mix, ...) and says which one would be used. With `audio.device: auto` a virtual default input is never used: a warning is logged and the first real microphone is picked (the log names it). Set `audio.device` to a name fragment (`HyperX`) or an index to choose yourself. With no usable microphone Incanto runs without audio and says so.

**Record.** **R** in the preview starts and stops a recording (a red REC indicator is drawn on the preview only). A recording is a folder `recordings/<timestamp>/`: `video.avi` (the raw camera frames, before any effect), `audio.wav` (mono 16-bit), `frames.csv` (frame index to capture time) and `meta.json` (sizes, codec, nominal fps, and the audio start time on the same clock). Replay syncs on those recorded times, never on the container's nominal fps. Encoding runs on a recorder thread; if it ever falls behind, frames are dropped and counted rather than slowing the loop (the log says so).
- `recording.codec: lossless` (default) is HFYU: bit-exact frames, so replay sees exactly the pixels live saw. It costs about 1.7 GB per minute. `mjpg` is about 25 times smaller but is a second JPEG generation (mean error 1.7 levels); this OpenCV build cannot raise MJPG quality, and FFV1 lossless is too slow (65 ms a frame). `--record-codec` overrides it for one run.
- Frames are the ones the main loop reads; if the loop is late the camera's newest frame wins, so a clip can have a few frames fewer than the camera delivered (about 28.5-29.9 effective fps against 30). Every frame in the clip carries its true time in `frames.csv`.

**Replay.** `python -m incanto --replay recordings/<clip>` feeds the video and audio through the full pipeline instead of the webcam and microphone, on the recorded clock. `--realtime` (default) paces frames by their recorded times; `--fast` runs as fast as the pipeline allows. In replay, hand tracking runs synchronously per frame (not on the async worker) and MediaPipe gets timestamps derived from the recorded times, so results are deterministic run to run. No virtual camera is opened. `--render-out out.mp4` (or `.avi`) saves the composited output of a replay for local review (no audio; local-only, never commit it). `--no-preview` runs it headless.

**Guided recording.** `python -m incanto --record-guided plans/fixture-plan.yaml` prompts you through a plan: for each cue two soft ticks, then a higher beep and a big on-preview prompt ("SNAP NOW"); do the gesture on the beep. The plan has 10 snaps, 10 claps, 10 thumbs-ups with random gaps and rests, then 2 minutes of "talk normally with your hands" with no cues. The cue times are written to `tests/fixtures/labels.yaml` (gesture, time from the clip's first frame, tolerance window of +-0.6 s, plus the talk stretches in which nothing may fire). Commit that file; the clip stays local.

**Checks.** `scripts/check_av_sync.py recordings/<clip>` finds each clap in the audio and in the video and reports the audio-video offset (target: within one frame, 33 ms); if there is a consistent offset it suggests `audio.offset_ms`, which is added to every audio timestamp, live and replay. The video side is timed from motion (the last frame that still shows the fast approach and the first that shows the hands stopped), not from wrist landmarks, because hand tracking loses both hands for 150-350 ms around a clap. `scripts/check_cues.py recordings/<clip>` checks every labelled cue against the recording (hand visible, sound spike, thumbs-up pose, when relative to the beep) and never edits `labels.yaml`. `scripts/check_determinism.py recordings/<clip>` replays the clip with two independent sets of hand tracker and segmenter and reports the largest difference in landmarks and masks. The scripts accept `--report docs/evidence/phase-4/<name>.txt` for a committable text report.

**Two things measured on the first guided clip.** (1) The webcam repeated every other frame (50% of frames identical): in this lighting the picture updates at about 15 fps even though 30 frames a second arrive, so timing from video is good to about 66 ms per event; more light, or a manual exposure that keeps 30 fps, fixes it. (2) People react to a beep 0.5-1 s late, so gestures land after the cue time: judge a label window as cue to cue + 1.5 s rather than cue +- 0.6 s, or retime the labels to the measured gesture times.

## Camera dark after a crash? `--reset-camera`

The camera driver remembers a manual exposure across processes. Incanto restores auto exposure on every normal exit path (error, Ctrl+C, closing the window, SIGTERM, console close, interpreter exit). A hard kill (Task Manager, power loss) cannot be intercepted, so if your webcam looks dark in other apps, run:

```
.venv\Scripts\python -m incanto --reset-camera
```

It writes auto exposure to the camera, logs the mean brightness before and after, and exits.

## Privacy

This repository is public, and Incanto is built so that nothing of you ever ends up in it:

- Everything runs locally. Camera frames, microphone audio and processed video go only to your own virtual camera.
- Recordings, screenshots, camera frames and clean plates (photos of your room) stay on your machine, in `recordings/`, `plates/` and `docs/evidence/`. They are gitignored and never committed or uploaded.
- Only text logs and reports are committed. `tests/fixtures/` holds only `labels.yaml`; the clips it refers to stay local, and replay tests skip cleanly when the clips are missing.
- A pre-commit hook refuses any image, video, audio or array file outside `assets/` (which is for stock files only). Install it once per clone: `python scripts/install_hooks.py`.
