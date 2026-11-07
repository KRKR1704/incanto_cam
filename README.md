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
