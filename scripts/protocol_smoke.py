"""Generate synthetic protocol evidence with the released local detector."""
import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tidesight import config
from tidesight.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    args = parser.parse_args()
    if args.video.exists():
        raise ValueError("Choose a new synthetic video path")
    args.video.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(args.video), cv2.VideoWriter_fourcc(*"mp4v"), 1, (160, 120))
    if not writer.isOpened():
        raise ValueError("Could not create synthetic fixture")
    try:
        for index in range(1020):
            image = np.zeros((120, 160, 3), dtype=np.uint8)
            image[:, :, 0] = 80
            writer.write(image)
    finally:
        writer.release()
    config.DETECTOR_WEIGHTS = args.weights
    result = run_pipeline(args.video, False, protocol_start_s=60)
    print(f"Synthetic fixture only: {result['counts']}")


if __name__ == "__main__":
    main()
