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

See `docs/incanto-project-plan.md` and `docs/incanto-build-tasks.md` for the plan.

**Status: Phase 0** (setup and scaffolding only; no functionality yet).

## Privacy

This repository is public, and Incanto is built so that nothing of you ever ends up in it:

- Everything runs locally. Camera frames, microphone audio and processed video go only to your own virtual camera.
- Recordings, screenshots, camera frames and clean plates (photos of your room) stay on your machine, in `recordings/`, `plates/` and `docs/evidence/`. They are gitignored and never committed or uploaded.
- Only text logs and reports are committed. `tests/fixtures/` holds only `labels.yaml`; the clips it refers to stay local, and replay tests skip cleanly when the clips are missing.
- A pre-commit hook refuses any image, video, audio or array file outside `assets/` (which is for stock files only). Install it once per clone: `python scripts/install_hooks.py`.
