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

How it works:
    1. read ~5 raw frames per second, shrink to _WORK_WIDTH px wide, greyscale + blur
    2. MOG2 background subtraction -> foreground mask. The model is first warmed up on
       the opening _WARMUP_S seconds, then the video is read again from the start, so
       the opening frames are judged by a trained model instead of being all "motion".
    3. morphological open (drops sediment specks) + close (joins a fish body into one blob)
    4. ignore "chronic" pixels that are foreground most of the time (swaying grass, kelp)
    5. a sampled frame is busy when at least one blob of _MIN_BLOB_FRAC of the frame is left
    6. drop single-sample blips: a busy sample needs a busy neighbour
    7. runs of busy samples -> segments: pad, clamp, merge, score = share of busy samples

Known limits (check on real clips): a fish that never moves is absorbed into the
background; a camera bump or big light change makes the next stretch busy (recall first).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence, TypedDict

import cv2
import numpy as np

from tidesight import config
from tidesight.contracts import Segment, time_from_frame, video_info

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- tuning (triage only)
# Shared values (padding, merge gap) come from config.py; these knobs only matter inside
# triage. Move any of them to config.py (via the App owner) if another module needs it.
_SAMPLE_FPS = 5.0           # frames per second triage looks at
_WORK_WIDTH = 640           # frames are shrunk to this width (smaller = faster, but tiny fish vanish)
_BLUR_KSIZE = 5             # Gaussian blur before background subtraction (hides sensor noise)
_WARMUP_S = 2.0             # seconds used to warm up the background model before judging frames
_MOG2_HISTORY = 150         # sampled frames of memory (~30 s at 5 fps)
_MOG2_VAR_THRESHOLD = 16.0  # lower = more sensitive (murky water has low contrast)
_OPEN_KSIZE = 5             # removes anything thinner than this: sediment specks, thin grass blades
_CLOSE_KSIZE = 5            # joins nearby pieces of one fish (bigger also glues specks together)
_MIN_BLOB_FRAC = 0.0005     # smallest blob that counts, as a share of the frame: 115 px at 640x360,
                            # about a 15x8 px fish. Raise it if sediment keeps clips busy; lower it
                            # if small fish are missed (check "fish missed by triage" on real clips).
_CHRONIC_ALPHA = 0.03       # how fast the "always moving" map learns (per sample)
_CHRONIC_LEVEL = 0.5        # pixels moving more than this share of the time are ignored
_GLOBAL_CHANGE_FRAC = 0.3   # above this share of the frame moving = camera bump / light change


class ActivitySample(TypedDict):
    """Triage's view of one sampled frame (for tests and a future activity timeline)."""
    frame: int          # original frame number
    time_s: float       # round(frame / fps, 3)
    motion: float       # share of the frame covered by blobs that count (0-1)
    busy: bool          # after the single-sample blip filter


def run_triage(video_path: str | Path) -> list[Segment]:
    info = video_info(video_path)
    samples = measure_activity(video_path)
    segments = segments_from_activity([s["time_s"] for s in samples],
                                      [s["busy"] for s in samples], info["duration_s"])
    busy_s = sum(s["end_s"] - s["start_s"] for s in segments)
    log.info("Triage: %d segment(s), %.1f of %.1f s busy (%.1f%% skipped)", len(segments),
             busy_s, info["duration_s"], 100 * (1 - busy_s / info["duration_s"]))
    return segments


# ======================================================================= step 1-6: video -> busy samples
def measure_activity(video_path: str | Path) -> list[ActivitySample]:
    """Read the video at ~_SAMPLE_FPS and decide, per sampled frame, whether something moves."""
    info = video_info(video_path)
    fps = info["fps"]
    step = max(1, round(fps / _SAMPLE_FPS))
    width, height = _work_size(info["width"], info["height"])
    frame_area = width * height
    min_blob = max(1, round(_MIN_BLOB_FRAC * frame_area))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_OPEN_KSIZE, _OPEN_KSIZE))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_CLOSE_KSIZE, _CLOSE_KSIZE))

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=_MOG2_HISTORY, varThreshold=_MOG2_VAR_THRESHOLD, detectShadows=False)
    warmup_frames = round(_WARMUP_S * fps)
    for _, grey in _sampled_frames(video_path, step, (width, height), stop_before=warmup_frames):
        subtractor.apply(grey)

    chronic = np.zeros((height, width), np.float32)   # running share of time each pixel moves
    frames: list[int] = []
    motion: list[float] = []
    for frame_idx, grey in _sampled_frames(video_path, step, (width, height)):
        fg = subtractor.apply(grey)
        moving = fg > 0
        moving_frac = float(moving.mean())

        mask = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
        mask[chronic > _CHRONIC_LEVEL] = 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:n, cv2.CC_STAT_AREA]
        kept = int(areas[areas >= min_blob].sum())

        if moving_frac < _GLOBAL_CHANGE_FRAC:   # don't let a camera bump mark the whole frame chronic
            cv2.accumulateWeighted(moving.astype(np.float32), chronic, _CHRONIC_ALPHA)
        frames.append(frame_idx)
        motion.append(kept / frame_area)

    raw_busy = [m > 0 for m in motion]
    busy = drop_blips(raw_busy)
    return [ActivitySample(frame=f, time_s=time_from_frame(f, fps), motion=round(m, 5), busy=b)
            for f, m, b in zip(frames, motion, busy)]


def _work_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, _WORK_WIDTH / width)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _sampled_frames(video_path: str | Path, step: int, size: tuple[int, int],
                    stop_before: int | None = None):
    """Yield (original frame number, small blurred greyscale frame) for every `step`th frame."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        frame_idx = 0
        while stop_before is None or frame_idx < stop_before:
            if frame_idx % step:
                if not cap.grab():
                    break
            else:
                ok, img = cap.read()
                if not ok:
                    break
                if (img.shape[1], img.shape[0]) != size:
                    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
                grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                yield frame_idx, cv2.GaussianBlur(grey, (_BLUR_KSIZE, _BLUR_KSIZE), 0)
            frame_idx += 1
    finally:
        cap.release()


def drop_blips(raw_busy: Sequence[bool]) -> list[bool]:
    """A busy sample stays busy only if a neighbouring sample is busy too (drops one-frame flickers)."""
    n = len(raw_busy)
    return [bool(raw_busy[i]) and ((i > 0 and raw_busy[i - 1]) or (i + 1 < n and raw_busy[i + 1]))
            for i in range(n)]


# ======================================================================= step 7: busy samples -> segments
def segments_from_activity(times: Sequence[float], busy: Sequence[bool], duration_s: float,
                           pad_s: float = config.TRIAGE_PAD_S,
                           merge_gap_s: float = config.TRIAGE_MERGE_GAP_S) -> list[Segment]:
    """Turn per-sample busy flags into padded, merged, scored segments that follow the contract."""
    runs: list[tuple[float, float]] = []
    run_start = run_end = None
    for t, b in zip(times, busy):
        if b:
            run_start = t if run_start is None else run_start
            run_end = t
        elif run_start is not None:
            runs.append((run_start, run_end))
            run_start = None
    if run_start is not None:
        runs.append((run_start, run_end))

    merged: list[list[float]] = []
    for start, end in runs:
        start, end = max(0.0, start - pad_s), min(duration_s, end + pad_s)
        if merged and start - merged[-1][1] < merge_gap_s:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])

    segments: list[Segment] = []
    for start, end in merged:
        start, end = round(start, 3), round(end, 3)
        if end <= start:
            continue
        inside = [b for t, b in zip(times, busy) if start <= t <= end]
        score = sum(inside) / len(inside) if inside else 1.0
        segments.append(Segment(start_s=start, end_s=end, activity_score=round(score, 3)))
    return segments
