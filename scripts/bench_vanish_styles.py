"""Per-frame cost of each vanish style on synthetic 1280x720 frames (no camera needed).

Run with the venv: .venv/Scripts/python -m scripts.bench_vanish_styles (from the project root)
Measures VanishStyler.render() only (mask work included for region/live), vanished and fully faded in."""
import dataclasses
import time

import cv2
import numpy as np

from incanto.config import VanishConfig
from incanto.plate.clean_plate import make_plate
from incanto.render.vanish_mask import VanishMask
from incanto.render.vanish_style import VanishStyler

W, H, SW, SH = 1280, 720, 640, 360


def main(frames: int = 300) -> None:
    rng = np.random.default_rng(0)
    y, x = np.mgrid[0:H, 0:W]
    base = 90 + 50 * np.sin(x / 90.0) * np.cos(y / 60.0)
    img = np.stack([base, base * 0.9 + 10, base * 1.1 - 10], axis=2).clip(0, 255).astype(np.float32)
    plate = make_plate(img, 30, None, 0.0, np.full(img.shape, 2.0, np.float32))
    for style in ("freeze", "region", "live"):
        for grain in (1.0, 0.0):
            if style == "live" and grain == 0.0:
                continue
            cfg = dataclasses.replace(VanishConfig(), style=style, grain_scale=grain, transition_s=0.0)
            vm = VanishMask(cfg, (W, H), (SW, SH))
            st = VanishStyler(cfg, vm, (W, H), seed=0)
            vm.set_plate(plate.u8)
            st.set_plate(plate)
            times = []
            for i in range(frames):
                live = np.clip(plate.u8.astype(np.float32) + rng.normal(0, 2, plate.u8.shape), 0, 255).astype(np.uint8)
                cx = 300 + int(250 * np.sin(i / 20.0))
                live[200:600, cx:cx + 260] = (30, 200, 60)
                small = cv2.resize(live, (SW, SH), interpolation=cv2.INTER_AREA)
                seg = np.zeros((SH, SW), np.float32)
                seg[100:300, cx // 2:cx // 2 + 130] = 1.0
                t = time.perf_counter()
                st.render(live, small, seg, [], True, 1 / 30, refresh=True)
                times.append((time.perf_counter() - t) * 1000)
            t = np.array(times[30:])
            print(f"{style:7s} grain x{grain:.0f}: mean {t.mean():5.1f} ms  p95 {np.percentile(t, 95):5.1f} ms  max {t.max():5.1f} ms")


if __name__ == "__main__":
    main()
