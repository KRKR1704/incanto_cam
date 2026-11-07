"""Download the official MediaPipe model files used by Incanto into models/."""
from __future__ import annotations

import sys
import urllib.request
from pathlib import Path

BASE = "https://storage.googleapis.com/mediapipe-models"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# name on disk -> official URL
MODELS = {
    "hand_landmarker.task": f"{BASE}/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    # Landscape (144x256) variant: our frames are 16:9, so it wastes no pixels on padding.
    "selfie_segmenter_landscape.tflite":
        f"{BASE}/image_segmenter/selfie_segmenter_landscape/float16/latest/selfie_segmenter_landscape.tflite",
}


def download(name: str, url: str, force: bool = False) -> bool:
    dest = MODELS_DIR / name
    if dest.exists() and dest.stat().st_size > 0 and not force:
        print(f"[skip] {name}: already present ({dest.stat().st_size:,} bytes)")
        return True
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(url, timeout=60) as resp, open(tmp, "wb") as out:
            while chunk := resp.read(1 << 16):
                out.write(chunk)
    except Exception as exc:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        print(f"[FAIL] {name}: {exc}\n       URL: {url}")
        return False
    size = tmp.stat().st_size
    if size == 0:
        tmp.unlink()
        print(f"[FAIL] {name}: downloaded file is empty\n       URL: {url}")
        return False
    tmp.replace(dest)
    print(f"[ok]   {name}: {size:,} bytes")
    return True


def main() -> int:
    force = "--force" in sys.argv
    MODELS_DIR.mkdir(exist_ok=True)
    results = [download(name, url, force) for name, url in MODELS.items()]
    return 0 if all(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
