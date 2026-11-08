from __future__ import annotations

import argparse
import math
import shutil

import cv2

from ml.common import DATA_DIR, LABELS_DIR, SPLITS, ensure_settings, frame_name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    a = ap.parse_args()
    ensure_settings()
    from tidesight import detect
    from tidesight.contracts import video_info

    info = video_info(a.video)
    jobs = [(burst, t) for burst, spec in SPLITS["bursts"].items() for t in spec["t_s"]]
    jobs += [("test_handcounted", int(t)) for t in SPLITS["hand_counts"]]
    jobs.sort(key=lambda j: j[1])
    reader = detect.FrameReader(a.video, info["frame_count"], info["fps"])
    n_labels = 0
    try:
        for burst, t in jobs:
            frame = math.ceil(t * info["fps"] - 1e-9)
            img, _ = detect.prepare_frame(reader.read(frame))
            out = DATA_DIR / burst
            out.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out / f"{frame_name(t)}.png"), img)
            label = LABELS_DIR / burst / f"{frame_name(t)}.txt"
            if label.exists():
                shutil.copy(label, out / label.name)
                n_labels += 1
            print(f"{burst:20s} t={t:5d}s frame={frame}")
    finally:
        reader.close()
    print(f"Wrote {len(jobs)} frames ({n_labels} with labels) to {DATA_DIR}")


if __name__ == "__main__":
    main()
