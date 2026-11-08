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
    - SPOT CHECKS: no motion is not proof of no fish. A short window
      (_SPOT_CHECK_LEN_S) every _SPOT_CHECK_EVERY_S is always kept, starting at 0 s, so
      the detector also sees fish that never move. Spot checks are not padded and have
      activity_score 0 unless something moved inside them.
    - must not write files (pipeline.py does that).

How it works:
    1. read ~5 raw frames per second, shrink to _WORK_WIDTH px wide, greyscale + blur
    2. MOG2 background subtraction -> foreground mask. Before judging any frame, the
       model learns the empty scene from _BG_SAMPLES frames spread over the WHOLE clip
       (their per-pixel median, then the frames themselves for normal flicker). A fish
       that passes by is in only one or two of them, so it does not become background,
       even when it swims through in the first second.
    3. morphological open (drops sediment specks) + close (joins a fish body into one blob)
    4. ignore "chronic" pixels that are foreground most of the time (swaying grass, kelp)
    5. a sampled frame is busy when at least one blob of _MIN_BLOB_FRAC of the frame is left
    6. drop single-sample blips: a busy sample needs a busy neighbour
    7. runs of busy samples -> segments: pad, clamp; add spot checks; merge;
       score = share of busy samples

Known limits (check on real clips): a fish that never moves is seen only in spot
checks (the detector still counts it there); a fish that is in more than half of the
whole clip at the same spot becomes part of the background, and is caught by motion
only while it moves (or by spot checks); a camera bump or big light change makes the
next stretch busy (recall first).

Measured on real footage (T1-STAT2 411, 35.8 min 4K, Samkit's 25 labelled fish frames):
62.4% skipped, but 10 of 25 fish frames (566 of 1446 fish) fell in skipped footage, and
the MaxN of one event dropped from 165 to 45. Cause: distant schools in turbid water
differ from their surroundings by ~4 grey levels at _WORK_WIDTH, while plain water
already varies by 2-3; the same at 1280 px. A lower MOG2 noise floor, the LAB b channel,
frame differencing and per-region change energy did not separate them either. So on
murky footage motion triage is NOT safe: use it only where fish are clearly visible.
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
_BG_SAMPLES = 15            # frames spread over the whole clip used to learn the empty scene
_WARMUP_S = 2.0             # fallback only: warm up on the opening seconds if the clip can't seek
_SPOT_CHECK_EVERY_S = 10.0  # always keep a short window this often, even with no motion
_SPOT_CHECK_LEN_S = 0.4     # length of each spot check (2-3 detector frames at 5 fps)
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
                                      [s["busy"] for s in samples], info["duration_s"],
                                      spot_checks=spot_check_times(info["duration_s"]))
    kept_s = sum(s["end_s"] - s["start_s"] for s in segments)
    log.info("Triage: %d segment(s) (%d with motion), %.1f of %.1f s kept (%.1f%% skipped)",
             len(segments), sum(s["activity_score"] > 0 for s in segments), kept_s,
             info["duration_s"], 100 * (1 - kept_s / info["duration_s"]))
    return segments


def spot_check_times(duration_s: float, every_s: float = _SPOT_CHECK_EVERY_S) -> list[float]:
    """Start times of the always-kept spot checks: 0, every_s, 2*every_s, ... inside the video."""
    if duration_s <= 0 or every_s <= 0:
        return []
    return [round(i * every_s, 3) for i in range(int(np.ceil(duration_s / every_s - 1e-9)))]


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
    spread = _spread_frames(video_path, info["frame_count"], (width, height))
    if len(spread) >= 3:
        background = np.median(np.stack(spread), axis=0).astype(np.uint8)
        subtractor.apply(background, learningRate=1.0)   # start from the empty scene
        for grey in spread:                              # then learn its normal flicker
            subtractor.apply(grey)
    else:                                                # can't seek: old warm-up on the opening
        log.warning("Triage: could not sample frames across %s; warming up on the first %.0f s",
                    video_path, _WARMUP_S)
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


def _prepare(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Shrink to the work size, greyscale, blur."""
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    grey = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(grey, (_BLUR_KSIZE, _BLUR_KSIZE), 0)


def _spread_frames(video_path: str | Path, frame_count: int, size: tuple[int, int]) -> list[np.ndarray]:
    """_BG_SAMPLES prepared frames spread evenly over the whole clip (seeks, so it stays cheap).
    Exact frame positions don't matter here; frames that fail to decode are skipped."""
    cap = cv2.VideoCapture(str(video_path))
    frames: list[np.ndarray] = []
    try:
        for idx in np.linspace(0, max(0, frame_count - 1), _BG_SAMPLES).round().astype(int):
            if cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx)):
                ok, img = cap.read()
                if ok:
                    frames.append(_prepare(img, size))
    finally:
        cap.release()
    return frames


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
                yield frame_idx, _prepare(img, size)
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
                           merge_gap_s: float = config.TRIAGE_MERGE_GAP_S,
                           spot_checks: Sequence[float] = (),
                           spot_len_s: float = _SPOT_CHECK_LEN_S) -> list[Segment]:
    """Turn per-sample busy flags into padded, merged, scored segments that follow the contract.
    Each time in `spot_checks` adds an unpadded window of spot_len_s (kept inside the video)."""
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

    windows = [(max(0.0, start - pad_s), min(duration_s, end + pad_s)) for start, end in runs]
    for t in spot_checks:
        start = max(0.0, min(t, duration_s - spot_len_s))
        windows.append((start, min(duration_s, start + spot_len_s)))

    merged: list[list[float]] = []
    for start, end in sorted(windows):
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
