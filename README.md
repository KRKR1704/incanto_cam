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

Hotkeys work in the preview window: **B** captures a clean plate of the empty room (3 s countdown, leave the frame), **V** toggles vanish (applied to the virtual camera output too), **D** debug view, **S** screenshot (with `--shots-dir`), **Q** quit.

`vanish.mask_mode` in `config.yaml` (or `--mask-mode`) picks what hides you: `seg_only` (segmenter), `seg_hands` (plus filled hand hulls) or `full` (plus pixels that differ from the brightness-matched plate). The plate is a photo of your room and is local-only (`plates/`, gitignored). Recapture it whenever furniture moves: in `full` mode anything that differs from the plate is replaced by the plate's version.

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
