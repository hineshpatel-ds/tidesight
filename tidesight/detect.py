"""
Fish detection + tracking.

Owner: ML lead
Branch: feat/detect

Contract:
    run_detection(video_path, segments) -> list[Record]
    - look only at frames_to_process(segments, fps, frame_count, detect_step(fps))
    - frame numbers are ORIGINAL frame numbers (never 0, 1, 2 ... of the sampled frames)
    - read RAW frames (do NOT call enhance() here)
    - create every record with contracts.new_record(...)  (species fields get defaults)
    - track_id >= 1, unique within the clip; a fish keeps its id across frames
    - segments=None means "the whole video"
    - must not write files (pipeline.py does that).

PLACEHOLDER below: two fake fish swimming across the frame, so the pipeline runs.

Reference sketch for the real version (untested - verify on your machine):

    from ultralytics import YOLO
    model = YOLO(str(config.DETECTOR_WEIGHTS))
    cap = cv2.VideoCapture(str(video_path))
    wanted = set(frames)
    frame_idx = 0
    while True:
        ok = cap.grab()                         # grab() is cheap; only decode wanted frames
        if not ok:
            break
        if frame_idx in wanted:
            ok, img = cap.retrieve()
            res = model.track(img, persist=True, tracker=config.TRACKER,
                              conf=config.DETECT_CONF, imgsz=config.DETECT_IMGSZ, verbose=False)[0]
            if res.boxes.id is not None:
                for box, conf, tid in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist(),
                                          res.boxes.id.int().tolist()):
                    records.append(new_record(clip_id, frame_idx, fps, box, tid, conf))
        frame_idx += 1
    cap.release()

Things to check: speed on a laptop CPU (frames/second), whether tracks break across
segment gaps, and detection quality on turbid clips (precision / recall).
"""
from __future__ import annotations

from pathlib import Path

from tidesight.contracts import (Record, Segment, clip_id_from_path, detect_step,
                                 frames_to_process, new_record, video_info)


def run_detection(video_path: str | Path, segments: list[Segment] | None = None) -> list[Record]:
    info = video_info(video_path)
    clip_id = clip_id_from_path(video_path)
    fps, width, height = info["fps"], info["width"], info["height"]
    if segments is None:
        segments = [Segment(start_s=0.0, end_s=info["duration_s"], activity_score=1.0)]
    frames = frames_to_process(segments, fps, info["frame_count"], detect_step(fps))

    # TODO(ML lead): replace this placeholder with the real detector + tracker.
    records: list[Record] = []
    last = max(info["frame_count"] - 1, 1)
    fake_fish = ((1, 0.35, 0.12, True), (2, 0.65, 0.08, False))  # track_id, y position, size, left-to-right
    for frame in frames:
        progress = frame / last
        for track_id, y_frac, size, left_to_right in fake_fish:
            cx = int((progress if left_to_right else 1 - progress) * width)
            cy = int(y_frac * height)
            half_w, half_h = int(size * width / 2), int(size * width / 4)
            x1, y1 = max(0, cx - half_w), max(0, cy - half_h)
            x2, y2 = min(width - 1, cx + half_w), min(height - 1, cy + half_h)
            if x2 > x1 and y2 > y1:
                records.append(new_record(clip_id, frame, fps, (x1, y1, x2, y2), track_id, 0.9))
    return records
