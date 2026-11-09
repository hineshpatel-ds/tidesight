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
    1. ~5 samples per second. Each sample is the AVERAGE of all frames in its 0.2 s
       (_AVERAGE_FRAMES), shrunk to _WORK_WIDTH px, greyscale, light blur. Averaging
       cuts sensor and compression noise, which is about as strong as a faint fish.
    2. the empty scene = per-pixel median of "anchor" samples (one every
       _ANCHOR_EVERY_S, the last _BG_WINDOW_S of them). Until enough anchors exist,
       _BG_SAMPLES frames spread over the WHOLE clip fill in, so a fish in the first
       second is not learned as background. A school that passes by is in a minority of
       the anchors, so the median ignores it (MOG2 learned such schools within ~30 s).
    3. smooth changes (turbidity drifting, light) are removed with a high-pass filter
       (_LIGHT_SIGMA) from both the sample and the empty scene.
    4. every pixel has its OWN noise level: the spread (MAD) of its high-passed anchors.
       Open water is calm, so a faint fish there counts; the measuring pole's edges and
       sunlit, textured seabed flicker a lot, so they need a much bigger change.
       A pixel moves when it differs from the empty scene by > _NOISE_K x its noise.
    5. specks: before that threshold, a grey-level opening removes changes narrower than
       the speck size, however bright (sediment, plankton). The speck size is at least
       _SPECK_KSIZE, and at least one video-compression block (_CODEC_BLOCK_PX of the
       ORIGINAL frame) wide, so a clip that is not shrunk does not see its own block
       flicker as fish. Then a close joins a fish body into one blob.
    6. ignore "chronic" pixels that move most of the time (swaying grass, kelp)
    7. a sample is busy when at least one blob of _MIN_BLOB_FRAC of the frame (and more
       than two compression blocks) is left; a busy sample needs a busy neighbour
    8. runs of busy samples -> segments: pad, clamp; add spot checks; merge;
       score = share of busy samples

Known limits: a fish that never moves is seen only in spot checks (the detector still
counts it there); a school that stays in one place for more than half of
_BG_WINDOW_S becomes background (it was flagged while arriving); a camera bump or big
light change makes the next stretch busy (recall first).

Measured on real footage (Coastal Action T1-STAT2 411, 35.8 min 4K HEVC, Samkit's 25
labelled fish frames / 1446 fish, CPU only):
    - previous MOG2 version: 62.4% skipped, but 10 of 25 fish frames (566 fish) were in
      skipped footage and one event's MaxN fell from 165 to 45. Distant schools in turbid
      water are only ~4 grey levels from their surroundings, under MOG2's noise floor, and
      a school that stays ~30 s was learned as background.
    - this version: 13.5% skipped, 0 of 25 fish frames and 0 of 1446 fish missed, MaxN of
      both events unchanged (165, 140). Its mask touches 19% of labelled fish boxes vs 1%
      of same-size plain-water boxes, so it responds to fish, not to everything. It skips
      little because something really moves most of the time in this clip.
    - 25.6 min for the 35.8 min clip (MOG2: 14.5 min); averaging every frame costs most.
    _NOISE_K is the trade-off knob (higher = skips more, risks missing faint fish).
"""
from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Iterable, Iterator, Sequence, TypedDict

import cv2
import numpy as np

from tidesight import config
from tidesight.contracts import Segment, time_from_frame, video_info

log = logging.getLogger(__name__)

# ---------------------------------------------------------------- tuning (triage only)
# Shared values (padding, merge gap) come from config.py; these knobs only matter inside
# triage. Move any of them to config.py (via the App owner) if another module needs it.
_SAMPLE_FPS = 5.0           # samples per second triage looks at
_AVERAGE_FRAMES = True      # average all frames of each 0.2 s (less noise; decoding ~40% slower)
_WORK_WIDTH = 640           # frames are shrunk to this width (smaller = faster, but tiny fish vanish)
_BLUR_KSIZE = 3             # light Gaussian blur after shrinking (hides pixel noise, keeps small fish)
_BG_SAMPLES = 15            # frames spread over the whole clip that stand in until enough anchors exist
_WARMUP_S = 2.0             # fallback only: use the opening seconds if the clip can't seek
_SPOT_CHECK_EVERY_S = 10.0  # always keep a short window this often, even with no motion
_SPOT_CHECK_LEN_S = 0.4     # length of each spot check (2-3 detector frames at 5 fps)
_ANCHOR_EVERY_S = 2.0       # one sample this often goes into the empty-scene window
_BG_WINDOW_S = 120.0        # the empty scene is the median of the anchors of the last 2 minutes
_MIN_ANCHORS = 15           # below this many anchors, the spread frames are added to the window
_REFRESH_EVERY_S = 4.0      # recompute the empty scene and noise levels this often (median is slow-ish)
_LIGHT_SIGMA = 15.0         # px; changes smoother than this are light / turbidity, not fish
_NOISE_K = 4.5              # a pixel moves when it is > this many times its own noise from the empty
                            # scene. On T1-STAT2 411 (cached samples): 4.5 kept all 25 labelled fish
                            # frames and skipped 13.5%; 5 skipped 17.2% but lost one frame (2 fish).
_NOISE_FLOOR = 0.3          # grey levels: no pixel is assumed calmer than this...
_NOISE_FLOOR_FRAC = 0.5     # ...or than this share of the frame's typical noise
_SPECK_KSIZE = 5            # px; changes narrower than this are specks (sediment, plankton), however
                            # bright. Labelled fish are ~9 px tall at 640.
_CODEC_BLOCK_PX = 8         # video compression flickers in 8x8 blocks of the ORIGINAL frame. A clip that
                            # is not shrunk (e.g. 640 px wide) keeps them 8 px wide, so its speck size
                            # grows to one block + 1 px; 4K shrunk 6x makes them ~1 px (no change).
_CLOSE_KSIZE = 5            # joins nearby pieces of one fish (bigger also glues specks together)
_MIN_BLOB_FRAC = 0.00017    # smallest blob that counts, as a share of the frame: 40 px at 640x360.
                            # Labelled fish in T1-STAT2 411 are ~20x9 px there, and are often only
                            # partly above the noise.
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


# ======================================================================= step 1-7: video -> busy samples
def measure_activity(video_path: str | Path) -> list[ActivitySample]:
    """Read the video at ~_SAMPLE_FPS and decide, per sample, whether something moves."""
    info = video_info(video_path)
    fps = info["fps"]
    step = max(1, round(fps / _SAMPLE_FPS))
    width, height = _work_size(info["width"], info["height"])
    spread = _spread_frames(video_path, info["frame_count"], (width, height))
    if len(spread) < 3:                                   # can't seek: use the opening seconds
        log.warning("Triage: could not sample frames across %s; using the first %.0f s as the empty scene",
                    video_path, _WARMUP_S)
        spread = [grey for _, grey in _sampled_frames(video_path, step, (width, height),
                                                      stop_before=round(_WARMUP_S * fps))]
    shrink = width / info["width"]
    return activity_from_samples(_sampled_frames(video_path, step, (width, height)), spread, fps / step, fps,
                                 speck_px=speck_size(shrink), min_blob_px=2 * codec_block_px(shrink) ** 2)


def codec_block_px(shrink: float) -> float:
    """Width of one video-compression block after the frame is shrunk by `shrink` (<= 1)."""
    return _CODEC_BLOCK_PX * min(1.0, shrink)


def speck_size(shrink: float) -> int:
    """Width (odd px) below which a change counts as a speck, for frames shrunk by `shrink` (<= 1)."""
    block = int(np.ceil(codec_block_px(shrink) - 1e-9)) + 1
    return max(_SPECK_KSIZE, block + (block % 2 == 0))


def activity_from_samples(samples: Iterable[tuple[int, np.ndarray]], fill: list[np.ndarray],
                          samples_per_s: float, fps: float, speck_px: int = _SPECK_KSIZE,
                          min_blob_px: float = 0.0) -> list[ActivitySample]:
    """Steps 2-7 for prepared samples (frame number, small blurred float32 grey frame).
    `fill` = frames spread over the whole clip (stand-ins for the empty scene at the start).
    A blob counts when it covers _MIN_BLOB_FRAC of the frame and more than min_blob_px pixels
    (two compression blocks: a single flickering block is not a fish)."""
    scene = EmptyScene(fill, window=max(_MIN_ANCHORS, round(_BG_WINDOW_S / _ANCHOR_EVERY_S)),
                       speck_px=speck_px)
    anchor_every = max(1, round(_ANCHOR_EVERY_S * samples_per_s))
    refresh_every = max(anchor_every, round(_REFRESH_EVERY_S * samples_per_s))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (_CLOSE_KSIZE, _CLOSE_KSIZE))
    chronic: np.ndarray | None = None       # running share of time each pixel moves
    frames: list[int] = []
    motion: list[float] = []
    for i, (frame_idx, grey) in enumerate(samples):
        if chronic is None:
            chronic = np.zeros(grey.shape, np.float32)
        if i % anchor_every == 0:
            scene.add_anchor(grey)
        if i % refresh_every == 0:
            scene.refresh()
        moving = scene.moving(grey)
        moving_frac = float(moving.mean())
        frame_area = moving.size
        min_blob = max(1, round(_MIN_BLOB_FRAC * frame_area), int(min_blob_px) + 1)

        mask = cv2.morphologyEx(moving.astype(np.uint8), cv2.MORPH_CLOSE, k_close)
        mask[chronic > _CHRONIC_LEVEL] = 0
        n, _, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        areas = stats[1:n, cv2.CC_STAT_AREA]
        kept = int(areas[areas >= min_blob].sum())

        if moving_frac < _GLOBAL_CHANGE_FRAC:   # don't let a camera bump mark the whole frame chronic
            cv2.accumulateWeighted(moving.astype(np.float32), chronic, _CHRONIC_ALPHA)
        frames.append(frame_idx)
        motion.append(kept / frame_area)

    busy = drop_blips([m > 0 for m in motion])
    return [ActivitySample(frame=f, time_s=time_from_frame(f, fps), motion=round(m, 5), busy=b)
            for f, m, b in zip(frames, motion, busy)]


class EmptyScene:
    """What the scene looks like without fish, and how much each pixel normally flickers.

    Kept from anchor samples: the empty scene is their per-pixel median; a pixel's noise
    is the spread (MAD) of its anchors after smooth light changes are removed."""

    def __init__(self, fill: list[np.ndarray], window: int, speck_px: int = _SPECK_KSIZE):
        self._speck = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (speck_px, speck_px))
        self._fill = [(g, _highpass(g)) for g in fill]    # stand-ins until enough anchors exist
        self._anchors: deque[tuple[np.ndarray, np.ndarray]] = deque(maxlen=window)
        self._scene_hp: np.ndarray | None = None
        self._noise: np.ndarray | None = None

    def add_anchor(self, grey: np.ndarray) -> None:
        self._anchors.append((grey, _highpass(grey)))

    def refresh(self) -> None:
        used = list(self._anchors)
        if len(used) < _MIN_ANCHORS:
            used += self._fill
        scene = np.median(np.stack([g for g, _ in used]), axis=0)
        self._scene_hp = _highpass(scene)
        # noise per pixel, measured on a half-size grid (it changes slowly across the frame)
        residuals = np.stack([hp[::2, ::2] for _, hp in used]) - self._scene_hp[::2, ::2]
        mad = 1.4826 * np.median(np.abs(residuals - np.median(residuals, axis=0)), axis=0)
        floor = max(_NOISE_FLOOR, _NOISE_FLOOR_FRAC * float(np.median(mad)))
        h, w = scene.shape
        self._noise = cv2.resize(np.maximum(mad, floor), (w, h), interpolation=cv2.INTER_LINEAR)

    def moving(self, grey: np.ndarray) -> np.ndarray:
        """Pixels that differ from the empty scene by more than _NOISE_K x their own noise, over
        an area at least speck_px wide (narrower changes are specks, however bright)."""
        if self._scene_hp is None:
            self.refresh()
        z = np.abs(_highpass(grey) - self._scene_hp) / self._noise
        z = cv2.morphologyEx(z, cv2.MORPH_OPEN, self._speck)   # grey-level opening: drops narrow peaks
        return z > _NOISE_K


def _highpass(img: np.ndarray) -> np.ndarray:
    """Remove changes smoother than _LIGHT_SIGMA px (drifting turbidity, light)."""
    return img - cv2.GaussianBlur(img, (0, 0), _LIGHT_SIGMA)


def _work_size(width: int, height: int) -> tuple[int, int]:
    scale = min(1.0, _WORK_WIDTH / width)
    return max(1, round(width * scale)), max(1, round(height * scale))


def _small_grey(img: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (img.shape[1], img.shape[0]) != size:
        img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def _blur(grey: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(grey.astype(np.float32), (_BLUR_KSIZE, _BLUR_KSIZE), 0)


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
                    frames.append(_blur(_small_grey(img, size)))
    finally:
        cap.release()
    return frames


def _sampled_frames(video_path: str | Path, step: int, size: tuple[int, int],
                    stop_before: int | None = None,
                    average: bool = _AVERAGE_FRAMES) -> Iterator[tuple[int, np.ndarray]]:
    """Yield (frame number, small blurred float32 greyscale sample) for every `step`th frame.
    With `average`, the sample is the mean of that frame and the next step-1 frames."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        frame_idx = 0
        while stop_before is None or frame_idx < stop_before:
            ok, img = cap.read()
            if not ok:
                break
            total = _small_grey(img, size).astype(np.float32)
            count = 1
            for _ in range(step - 1):
                if average:
                    ok, img = cap.read()
                    if not ok:
                        break
                    total += _small_grey(img, size)
                    count += 1
                elif not cap.grab():
                    break
            yield frame_idx, _blur(total / count)
            frame_idx += step
    finally:
        cap.release()


def drop_blips(raw_busy: Sequence[bool]) -> list[bool]:
    """A busy sample stays busy only if a neighbouring sample is busy too (drops one-frame flickers)."""
    n = len(raw_busy)
    return [bool(raw_busy[i]) and ((i > 0 and raw_busy[i - 1]) or (i + 1 < n and raw_busy[i + 1]))
            for i in range(n)]


# ======================================================================= step 8: busy samples -> segments
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
