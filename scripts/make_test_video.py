"""
Make a short fake "murky water" video so everyone can test the pipeline before real clips arrive.

    python scripts/make_test_video.py

Writes data/clips/test_clip.mp4 (15 s, 640x360, 25 fps): greenish murky background, drifting
particles, swaying grass, and two dark fish-like shapes. Not realistic - just for testing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tidesight import config  # noqa: E402

W, H, FPS, SECONDS = 640, 360, 25, 15


def main() -> None:
    config.CLIPS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = config.CLIPS_DIR / "test_clip.mp4"
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    if not writer.isOpened():
        raise SystemExit("Could not open a video writer (is opencv-python installed?)")

    rng = np.random.default_rng(7)
    particles = rng.uniform([0, 0], [W, H], size=(150, 2))
    total = FPS * SECONDS
    for i in range(total):
        t = i / total
        frame = np.full((H, W, 3), (70, 95, 60), np.uint8)                   # murky green (BGR)
        frame = cv2.add(frame, rng.integers(0, 25, (H, W, 3), dtype=np.uint8))  # noise / turbidity

        for k in range(12):                                                     # swaying grass
            x0 = 30 + k * 52
            sway = int(12 * np.sin(2 * np.pi * (i / 40 + k / 5)))
            cv2.line(frame, (x0, H), (x0 + sway, H - 110), (40, 80, 35), 4)

        particles[:, 0] = (particles[:, 0] + rng.normal(0.6, 0.4, len(particles))) % W
        particles[:, 1] = (particles[:, 1] + rng.normal(0.2, 0.4, len(particles))) % H
        for x, y in particles.astype(int):                                      # drifting sediment
            cv2.circle(frame, (x, y), 1, (150, 165, 140), -1)

        if 0.15 < t < 0.85:                                                     # fish 1, left to right
            cx = int((t - 0.15) / 0.7 * W)
            cv2.ellipse(frame, (cx, 125), (38, 14), 0, 0, 360, (30, 40, 35), -1)
            cv2.fillPoly(frame, [np.array([(cx - 36, 125), (cx - 58, 110), (cx - 58, 140)])], (30, 40, 35))
        if 0.40 < t < 0.95:                                                     # fish 2, right to left
            cx = int(W - (t - 0.40) / 0.55 * W)
            cv2.ellipse(frame, (cx, 235), (26, 10), 0, 0, 360, (45, 55, 50), -1)
            cv2.fillPoly(frame, [np.array([(cx + 24, 235), (cx + 40, 224), (cx + 40, 246)])], (45, 55, 50))

        writer.write(cv2.GaussianBlur(frame, (5, 5), 0))
    writer.release()
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
