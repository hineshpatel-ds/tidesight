
from __future__ import annotations

import logging
import time
from pathlib import Path

import cv2
import numpy as np

from tidesight import config
from tidesight.contracts import (Record, Segment, clip_id_from_path, detect_step,
                                 frames_to_process, new_record, video_info)

log = logging.getLogger(__name__)


FISH_WEIGHTS = Path(config.MODELS_DIR) / "tidesight-fish-v1.pt"
MODEL_INPUT_WIDTH = 1920      
FISH_IMGSZ = 1280             
FISH_CONF = 0.18             
SEEK_GAP_S = 2.0             
MAX_LINK_DIST = 1.5          
MAX_MISSED = 2       


class CentroidTracker:
    """Greedy nearest-centre linking for small, fast fish at low frame rates (no extra dependencies)."""

    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: dict[int, tuple[float, float, float, int]] = {}   # id -> (cx, cy, size, missed)

    def update(self, boxes: list[list[float]]) -> list[int]:
        ids = [0] * len(boxes)
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1])) for b in boxes]
        track_ids = list(self.tracks)
        if track_ids and boxes:
            cost = np.array([[np.hypot(cx - tx, cy - ty) / max(1.0, max(sz, ts))
                              for (tx, ty, ts, _) in (self.tracks[t] for t in track_ids)]
                             for (cx, cy, sz) in cents])
            used_r, used_c = set(), set()
            for flat in np.argsort(cost, axis=None):          # cheapest pairs first
                r, c = divmod(int(flat), cost.shape[1])
                if cost[r, c] > MAX_LINK_DIST:
                    break
                if r not in used_r and c not in used_c:
                    ids[r] = track_ids[c]
                    used_r.add(r); used_c.add(c)
        matched = set(ids)
        for t in track_ids:                       # age unmatched tracks
            if t not in matched:
                tx, ty, ts, missed = self.tracks[t]
                if missed + 1 > MAX_MISSED:
                    del self.tracks[t]
                else:
                    self.tracks[t] = (tx, ty, ts, missed + 1)
        for i, (cx, cy, sz) in enumerate(cents):  # update matched, start new tracks
            if ids[i] == 0:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = (cx, cy, sz, 0)
        return ids


def prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    """Training-identical preprocessing. Returns (model image, scale) where original = model / scale."""
    h, w = frame.shape[:2]
    scale = MODEL_INPUT_WIDTH / w
    img = cv2.resize(frame, (MODEL_INPUT_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA) \
        if scale != 1.0 else frame
    out = np.empty_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], (1, 99))
        out[..., c] = np.clip((img[..., c].astype(np.float32) - lo) * 255.0 / max(1.0, hi - lo), 0, 255)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR), scale


def load_detector():
    """Load the fish model. Fails loudly with the fix if the weights are missing."""
    from ultralytics import YOLO

    if not FISH_WEIGHTS.exists():
        raise FileNotFoundError(
            f"Fish model not found at {FISH_WEIGHTS}. Copy tidesight-fish-v1.pt from the team drive into models/.")
    return YOLO(str(FISH_WEIGHTS))


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    """Round + clip an xyxy box (original pixels) to the image. None if it collapses."""
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _groups(frames: list[int], max_gap: int) -> list[list[int]]:
    """Split sorted frame numbers into runs; a new run starts when the gap is > max_gap."""
    runs: list[list[int]] = []
    for f in frames:
        if runs and f - runs[-1][-1] <= max_gap:
            runs[-1].append(f)
        else:
            runs.append([f])
    return runs


def run_detection(video_path: str | Path, segments: list[Segment] | None = None) -> list[Record]:
    info = video_info(video_path)
    clip_id = clip_id_from_path(video_path)
    fps, width, height = info["fps"], info["width"], info["height"]
    if segments is None:
        segments = [Segment(start_s=0.0, end_s=info["duration_s"], activity_score=1.0)]
    frames = frames_to_process(segments, fps, info["frame_count"], detect_step(fps))
    if not frames:
        log.info("%s: no frames to process (triage found no activity)", clip_id)
        return []

    model = load_detector()
    tracker = CentroidTracker()        # fresh tracker per clip
    records: list[Record] = []
    n_done, infer_s = 0, 0.0
    started = time.perf_counter()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        for run in _groups(frames, max_gap=int(SEEK_GAP_S * fps)):
            wanted = set(run)
            cap.set(cv2.CAP_PROP_POS_FRAMES, run[0])      # jump to the start of this busy stretch
            frame_idx = run[0]
            while frame_idx <= run[-1]:
                if not cap.grab():
                    log.warning("%s: video ended at frame %d", clip_id, frame_idx)
                    break
                if frame_idx in wanted:
                    ok, img = cap.retrieve()
                    if ok and img is not None:
                        model_img, scale = prepare_frame(img)
                        t0 = time.perf_counter()
                        res = model.predict(model_img, conf=FISH_CONF, imgsz=FISH_IMGSZ, verbose=False)[0]
                        infer_s += time.perf_counter() - t0
                        n_done += 1
                        kept = []
                        for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                            clipped = _clip_box([v / scale for v in box], width, height)
                            if clipped is not None:
                                kept.append((clipped, conf))
                        for (clipped, conf), tid in zip(kept, tracker.update([list(b) for b, _ in kept])):
                            records.append(new_record(clip_id, frame_idx, fps, clipped, tid, conf))
                        if n_done % 50 == 0:
                            log.info("%s: %d/%d frames", clip_id, n_done, len(frames))
                frame_idx += 1
    finally:
        cap.release()

    total_s = time.perf_counter() - started
    if n_done:
        log.info("%s: %d frames in %.1f s (%.2f s/frame model, %.2f frames/s overall); %d boxes, %d tracks",
                 clip_id, n_done, total_s, infer_s / n_done, n_done / total_s,
                 len(records), len({r['track_id'] for r in records}))
    return records
