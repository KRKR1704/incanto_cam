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

**Status: Phase 0** (setup and scaffolding only; no functionality yet).

## Vanish test (Phase 3)

Hotkeys work in the preview window: **B** captures a clean plate of the empty room (5 s countdown with a beep per second, leave the frame; a long tone means frames are being taken, three rising notes mean it was saved, two falling low notes mean it was aborted and nothing changed), **V** toggles vanish (applied to the virtual camera output too), **D** cycles the debug view (off, source colours, leak view where red shows you leaking through), **S** screenshot (with `--shots-dir`), **Q** quit.

`vanish.mask_mode` in `config.yaml` (or `--mask-mode`) picks what hides you: `seg_only` (segmenter), `seg_hands` (plus filled hand hulls) or `full` (plus pixels that differ from the brightness-matched plate). The plate is a photo of your room and is local-only (`plates/`, gitignored). Recapture it whenever furniture moves: in `full` mode anything that differs from the plate is replaced by the plate's version.

**Plate health.** Vanish only runs on a plate that matches the room. When you press V, and about twice a second after that, the plate is compared with the visible background *after* brightness matching, so auto-exposure swings (standing up, leaning in) do not count against it. Two things are judged, only in grid cells that are at least half background: the residual the gain map leaves per cell (`plate.health_max_residual`), and whether the edges of plate and live picture still line up (`plate.health_min_structure`, blind to brightness; catches a moved camera, rearranged furniture, another room). Global gain is only a loose sanity cap. If the check fails for `plate.health_fail_s` (3 s) vanish is switched off; on V it is refused straight away. Either way the preview shows "Plate doesn't match the room - press B", a WARNING is logged and the error sound plays. A plate loaded from a previous session is flagged as such until it passes this check. While vanish is on and the plate is healthy, the plate is slowly refreshed from the live picture outside the near-me zone (`vanish.plate_refresh_alpha`), so it follows daylight during a call; around you it stays as captured. `plate.sounds: false` mutes the beeps.

**Exposure lock.** With `camera.exposure: auto` and `plate.lock_exposure: true`, a successful capture is followed by a short sweep of manual exposures (only -5 and shorter, so the camera keeps 30 fps). The one closest to the plate's brightness is locked while the plate is in use; B returns to auto first, captures, then locks again. If nothing gets within `plate.lock_exposure_tolerance` of the plate's brightness (a dim room needs a longer exposure than 30 fps allows) the camera stays on auto and the log says why. The lock goes through the same safety net as a manual exposure: auto is restored on every exit path.

**Capture person check.** While you leave the frame, the capture watches every segmenter blob: one in which almost nothing moves during the last second (`plate.static_blob_max_motion`) is furniture, not you, and does not count towards `plate.max_person_coverage`. If the capture still aborts, a local-only debug image (`capture-abort-*.png`, in `plate.debug_dir`, never committed) shows what was counted: red = person, yellow = still blob ignored. **Shift+B** captures without the person check (a warning is logged).

**Lighting nudge.** If the matched residual stays above `plate.health_nudge_residual` (2.0) for `plate.health_nudge_s` (10 s) while vanished, you get one soft beep and "Lighting changed - press B to refresh plate"; vanish stays on. The plate refresh now reaches up to the mask plus `vanish.plate_refresh_margin_px` (15 px), but only on pixels every mask source calls background and that still match the gain-matched plate. `preview.scale` (default 0.5) draws the preview window at half size; the virtual camera output is never scaled.

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
