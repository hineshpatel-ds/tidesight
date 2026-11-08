from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TypedDict

import cv2
import numpy as np

from tidesight import config
from tidesight.contracts import (Record, Segment, clip_id_from_path, detect_step,
                                 frames_to_process, new_record, video_info)

log = logging.getLogger(__name__)

PREPROCESS_ID = "stretch1-99_clahe3-8x8_v1"
TRAINED_WITH_PREPROCESS = {"tidesight-fish-v1.pt"}
SEEK_MIN_GAP_S = 2.0

_REQUIRED_SETTINGS = ("DETECTOR_WEIGHTS", "DETECT_CONF", "DETECT_IMGSZ", "DETECT_INPUT_WIDTH",
                      "DETECT_PREPROCESS", "TRACK_LINK_DIST", "TRACK_MAX_MISSED", "TRACK_MAX_GAP_S")


class DetectionError(RuntimeError):
    pass


def check_settings() -> None:
    missing = [name for name in _REQUIRED_SETTINGS if not hasattr(config, name)]
    if missing:
        raise DetectionError(f"config.py is missing detector settings {missing}.")
    if config.DETECT_PREPROCESS != PREPROCESS_ID:
        raise DetectionError(f"config.DETECT_PREPROCESS={config.DETECT_PREPROCESS!r} but detect.py "
                             f"implements {PREPROCESS_ID!r}; the model would see the wrong images.")
    weights = Path(config.DETECTOR_WEIGHTS).name
    if weights not in TRAINED_WITH_PREPROCESS:
        raise DetectionError(f"config.DETECTOR_WEIGHTS={weights!r} was not trained with preprocessing "
                             f"{PREPROCESS_ID!r} (known: {sorted(TRAINED_WITH_PREPROCESS)}).")


def prepare_frame(frame: np.ndarray) -> tuple[np.ndarray, float]:
    h, w = frame.shape[:2]
    scale = config.DETECT_INPUT_WIDTH / w
    img = (cv2.resize(frame, (config.DETECT_INPUT_WIDTH, round(h * scale)), interpolation=cv2.INTER_AREA)
           if scale != 1.0 else frame)
    out = np.empty_like(img)
    for c in range(3):
        lo, hi = np.percentile(img[..., c], (1, 99))
        out[..., c] = np.clip((img[..., c].astype(np.float32) - lo) * 255.0 / max(1.0, hi - lo), 0, 255)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    l_ch = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l_ch)
    return cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2BGR), scale


def load_detector():
    from ultralytics import YOLO

    weights = Path(config.DETECTOR_WEIGHTS)
    if not weights.exists():
        raise DetectionError(f"Fish model not found at {weights}.")
    return YOLO(str(weights))


def _open_capture(video_path: str | Path):
    return cv2.VideoCapture(str(video_path))


class FrameReader:

    def __init__(self, video_path: str | Path, frame_count: int, fps: float) -> None:
        self.video_path = video_path
        self.frame_count = frame_count
        self.seek_gap = max(1, int(SEEK_MIN_GAP_S * fps))
        self.sequential_only = False
        self.cap = None
        self.next = 0
        self._reopen()

    def _reopen(self) -> None:
        if self.cap is not None:
            self.cap.release()
        self.cap = _open_capture(self.video_path)
        if not self.cap.isOpened():
            raise DetectionError(f"Cannot open video: {self.video_path}")
        self.next = 0

    def _reported_pos(self) -> int | None:
        pos = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
        return None if pos is None or pos < 0 else int(round(pos))

    def _try_seek(self, target: int) -> bool:
        if not self.cap.set(cv2.CAP_PROP_POS_FRAMES, target) or self._reported_pos() != target:
            return False
        self.next = target
        return True

    def _fall_back(self, target: int, why: str) -> None:
        log.warning("Seek to frame %d not verified (%s); decoding sequentially from frame 0.", target, why)
        self.sequential_only = True
        self._reopen()

    def read(self, target: int) -> np.ndarray:
        if not 0 <= target < self.frame_count:
            raise DetectionError(f"Requested frame {target} is outside the video (0..{self.frame_count - 1}).")
        seeked = False
        if target < self.next or (not self.sequential_only and target - self.next > self.seek_gap):
            if self.sequential_only:
                self._reopen()
            elif self._try_seek(target):
                seeked = True
            else:
                self._fall_back(target, "capture refused the seek or reported another position")
        while self.next < target:
            if not self.cap.grab():
                raise DetectionError(f"Could not decode frame {self.next} on the way to requested frame {target}.")
            self.next += 1
        if not self.cap.grab():
            raise DetectionError(f"Could not decode requested frame {target}.")
        ok, img = self.cap.retrieve()
        if not ok or img is None:
            raise DetectionError(f"Could not read pixels of requested frame {target}.")
        self.next = target + 1
        if seeked and self._reported_pos() != target + 1:
            self._fall_back(target, f"position after read was {self._reported_pos()}")
            return self.read(target)
        return img

    def close(self) -> None:
        if self.cap is not None:
            self.cap.release()


class CentroidTracker:

    def __init__(self) -> None:
        self.next_id = 1
        self.tracks: dict[int, tuple[float, float, float, int]] = {}
        self.last_time_s: float | None = None

    def update(self, boxes: list[list[float]], time_s: float) -> list[int]:
        if self.last_time_s is not None and time_s - self.last_time_s > config.TRACK_MAX_GAP_S:
            self.tracks.clear()
        self.last_time_s = time_s
        ids = [0] * len(boxes)
        cents = [((b[0] + b[2]) / 2, (b[1] + b[3]) / 2, max(b[2] - b[0], b[3] - b[1])) for b in boxes]
        track_ids = list(self.tracks)
        if track_ids and boxes:
            cost = np.array([[np.hypot(cx - tx, cy - ty) / max(1.0, sz, ts)
                              for (tx, ty, ts, _) in (self.tracks[t] for t in track_ids)]
                             for (cx, cy, sz) in cents])
            used_r, used_c = set(), set()
            for flat in np.argsort(cost, axis=None, kind="stable"):
                r, c = divmod(int(flat), cost.shape[1])
                if cost[r, c] > config.TRACK_LINK_DIST:
                    break
                if r not in used_r and c not in used_c:
                    ids[r] = track_ids[c]
                    used_r.add(r)
                    used_c.add(c)
        matched = set(ids)
        for t in track_ids:
            if t not in matched:
                tx, ty, ts, missed = self.tracks[t]
                if missed + 1 > config.TRACK_MAX_MISSED:
                    del self.tracks[t]
                else:
                    self.tracks[t] = (tx, ty, ts, missed + 1)
        for i, (cx, cy, sz) in enumerate(cents):
            if ids[i] == 0:
                ids[i] = self.next_id
                self.next_id += 1
            self.tracks[ids[i]] = (cx, cy, sz, 0)
        return ids


def _clip_box(box, width: int, height: int) -> tuple[int, int, int, int] | None:
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    x1, x2 = max(0, min(x1, width - 1)), max(0, min(x2, width - 1))
    y1, y2 = max(0, min(y1, height - 1)), max(0, min(y2, height - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _detect_frames(video_path: str | Path, frames: list[int], info: dict) -> dict[int, list[Record]]:
    check_settings()
    clip_id = clip_id_from_path(video_path)
    fps, width, height = info["fps"], info["width"], info["height"]
    model = load_detector()
    tracker = CentroidTracker()
    reader = FrameReader(video_path, info["frame_count"], fps)
    out: dict[int, list[Record]] = {}
    started, infer_s = time.perf_counter(), 0.0
    try:
        for frame_idx in frames:
            raw = reader.read(frame_idx)
            model_img, scale = prepare_frame(raw)
            t0 = time.perf_counter()
            res = model.predict(model_img, conf=config.DETECT_CONF, imgsz=config.DETECT_IMGSZ, verbose=False)[0]
            infer_s += time.perf_counter() - t0
            kept = []
            for box, conf in zip(res.boxes.xyxy.tolist(), res.boxes.conf.tolist()):
                clipped = _clip_box([v / scale for v in box], width, height)
                if clipped is not None:
                    kept.append((clipped, conf))
            ids = tracker.update([list(b) for b, _ in kept], frame_idx / fps)
            out[frame_idx] = [new_record(clip_id, frame_idx, fps, b, tid, conf)
                              for (b, conf), tid in zip(kept, ids)]
    finally:
        reader.close()
    if len(out) != len(frames):
        raise DetectionError(f"Processed {len(out)} of {len(frames)} requested frames.")
    total = time.perf_counter() - started
    log.info("%s: %d frames in %.1f s (%.2f s/frame model); %d boxes", clip_id, len(frames), total,
             infer_s / max(1, len(frames)), sum(len(v) for v in out.values()))
    return out


def run_detection(video_path: str | Path, segments: list[Segment] | None = None) -> list[Record]:
    info = video_info(video_path)
    if segments is None:
        segments = [Segment(start_s=0.0, end_s=info["duration_s"], activity_score=1.0)]
    frames = frames_to_process(segments, info["fps"], info["frame_count"], detect_step(info["fps"]))
    if not frames:
        log.info("%s: no frames to process (triage found no activity)", clip_id_from_path(video_path))
        return []
    by_frame = _detect_frames(video_path, frames, info)
    return [rec for f in frames for rec in by_frame[f]]


class ProtocolSample(TypedDict):
    index: int
    time_s: float
    frame: int
    status: str
    records: list[Record]


class ProtocolRun(TypedDict):
    clip_id: str
    fps: float
    frame_count: int
    start_s: float
    interval_s: float
    n_samples: int
    model: str
    preprocess: str
    conf: float
    samples: list[ProtocolSample]


def run_protocol_samples(video_path: str | Path, start_s: float) -> ProtocolRun:
    from tidesight.count import protocol_sample_frames

    info = video_info(video_path)
    points = protocol_sample_frames(start_s, info["fps"], info["frame_count"])
    by_frame = _detect_frames(video_path, [p["frame"] for p in points], info)
    return ProtocolRun(
        clip_id=clip_id_from_path(video_path), fps=info["fps"], frame_count=info["frame_count"],
        start_s=float(start_s), interval_s=float(config.PROTOCOL_INTERVAL_S),
        n_samples=len(points), model=Path(config.DETECTOR_WEIGHTS).name,
        preprocess=PREPROCESS_ID, conf=float(config.DETECT_CONF),
        samples=[ProtocolSample(index=p["index"], time_s=p["time_s"], frame=p["frame"], status="done",
                                records=by_frame[p["frame"]]) for p in points])
