"""
Activity triage: find the parts of the video where something is moving.

Owner: Video + classical CV
Branch: feat/video

Contract:
    run_triage(video_path) -> list[Segment]
    - Segment = {"start_s": float, "end_s": float, "activity_score": float 0-1}
    - sorted by start_s, no overlaps, end_s <= video duration
    - pad each busy stretch by config.TRIAGE_PAD_S; merge gaps < config.TRIAGE_MERGE_GAP_S
    - RECALL FIRST: when unsure, keep the segment. Missing a fish is worse than
      showing the reviewer some swaying eelgrass.
    - must not write files (pipeline.py does that).

PLACEHOLDER below: returns the whole video as one busy segment, so the pipeline runs.
Replace it with the real version, e.g.:
    1. read frames at ~5 fps (optionally enhance() them first)
    2. cv2.createBackgroundSubtractorMOG2 -> foreground mask
    3. morphological open/close; drop blobs smaller than a minimum area
    4. keep only motion that persists for a few sampled frames (drops drifting sediment)
    5. turn busy frames into segments, pad, merge, score by fraction of busy frames
"""
from __future__ import annotations

from pathlib import Path

from tidesight.contracts import Segment, video_info


def run_triage(video_path: str | Path) -> list[Segment]:
    info = video_info(video_path)
    # TODO(Video + classical CV): replace this placeholder with real motion triage.
    return [Segment(start_s=0.0, end_s=info["duration_s"], activity_score=1.0)]
