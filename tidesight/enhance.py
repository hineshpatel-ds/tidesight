"""
Image enhancement for murky water.

Owner: Video + classical CV
Branch: feat/video

Contract:
    enhance(frame) -> frame
    - input:  BGR uint8 image (H, W, 3), as returned by cv2
    - output: BGR uint8 image with EXACTLY the same shape
    - used for DISPLAY (review screen, demo) and optionally by triage.
    - the detector does NOT use it (it reads raw frames).

How it works:
    1. colour correction (one pass over the frame; the numbers come from a small copy):
       a. water swallows red light first, so red is topped up from the green channel,
          which survives best ("channel compensation", Ancuti et al. 2018). Dark-red
          pixels get the most. Blue can be topped up too (_BLUE_COMP).
       b. murky water adds a coloured haze over everything. The darkest 1% of each colour
          channel is treated as haze and becomes black.
       c. grey-world white balance: scale each channel so the frame averages to grey.
       d. contrast stretch: one shared scale for all channels, so the brightest 0.5% just
          reaches white. Shared, so it cannot undo the white balance.
    2. light denoise: edge-preserving (bilateral) filter on brightness, small blur on
       colour, so the stretch and CLAHE do not turn sensor noise into coloured grain.
    3. CLAHE contrast boost on brightness only (LAB "L" channel).

    About 50 ms for a 1920x1080 frame, 170 ms for 4K, on a laptop CPU (no GPU, no model,
    no internet). The review screen only enhances one frame per fish, so this is plenty.

Known limits (check on real clips):
    - one correction for the whole frame: when the far water is greener than the near
      seabed, one end stays a bit green and the other turns a bit pink.
    - grey-world assumes the scene averages to grey, so a frame that really is mostly one
      colour (a close-up of eelgrass, an orange fish filling the frame) is pushed towards
      grey. Gains are capped by _MAX_GAIN so it never goes wild.
    - numbers are worked out per frame, so in a video the colours can shift slightly when
      a big fish swims through. colour_params() can be measured once and reused if that
      matters for the demo video.
    - triage reads raw frames: CLAHE adapts per frame, and that flicker would look like
      motion to background subtraction.
"""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np

# ---------------------------------------------------------------- tuning (enhance only)
_RED_COMP = 1.0         # red top-up strength (Ancuti et al. use 1.0); 0 = off
_BLUE_COMP = 0.0        # blue top-up strength; try 1.0 for very green, plankton-rich water
_HAZE_PCT = 1.0         # per channel, the darkest this % counts as haze and becomes black
_STRETCH_PCT = 0.5      # the darkest/brightest this % may clip to black/white in the stretch
_MAX_GAIN = 3.0         # cap for each white-balance gain and for the contrast stretch
_MIN_CONTRAST = 0.04    # a channel spread below this (~10 grey levels: blank frame, lens cap)
                        # means the frame is only white balanced, not de-hazed or stretched
_STATS_WIDTH = 320      # the numbers are measured on a copy this many px wide
_DENOISE_D = 9          # bilateral filter size on brightness in px (0 = off)
_DENOISE_SIGMA = 20.0   # brightness steps smaller than this are smoothed; bigger edges kept
_CHROMA_BLUR = 5        # blur size on colour only, hides coloured grain (0 = off)
_CLAHE_CLIP = 1.5       # contrast boost strength (higher = punchier, but flat murky water
                        # turns blotchy: 2.0 made video-compression blocks clearly visible)
_CLAHE_TILES = (8, 8)   # grid of regions that each get their own contrast boost

_CLAHE = cv2.createCLAHE(clipLimit=_CLAHE_CLIP, tileGridSize=_CLAHE_TILES)


class ColourParams(NamedTuple):
    """Per channel (B, G, R): out = gain * (c + top_up * (255 - c) * green / 255) + offset."""
    top_up: tuple[float, float, float]
    gain: tuple[float, float, float]
    offset: tuple[float, float, float]


def enhance(frame: np.ndarray) -> np.ndarray:
    _check_frame(frame)
    lab = cv2.cvtColor(correct_colour(frame), cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    if _DENOISE_D > 0:
        l_channel = cv2.bilateralFilter(l_channel, _DENOISE_D, _DENOISE_SIGMA, _DENOISE_D)
    l_channel = _CLAHE.apply(l_channel)
    if _CHROMA_BLUR > 0:
        a_channel = cv2.GaussianBlur(a_channel, (_CHROMA_BLUR, _CHROMA_BLUR), 0)
        b_channel = cv2.GaussianBlur(b_channel, (_CHROMA_BLUR, _CHROMA_BLUR), 0)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)


# ======================================================================= step 1: colour
def correct_colour(frame: np.ndarray, params: ColourParams | None = None) -> np.ndarray:
    """Step 1 only (top up, de-haze, white balance, stretch). BGR uint8 in and out."""
    _check_frame(frame)
    p = params or colour_params(frame)
    channels = cv2.split(frame)
    green = channels[1]
    return cv2.merge([_apply(c, green, k, gain, offset)
                      for c, k, gain, offset in zip(channels, p.top_up, p.gain, p.offset)])


def colour_params(frame: np.ndarray) -> ColourParams:
    """Work out the colour correction for this frame, from a small copy (cheap)."""
    _check_frame(frame)
    chans = cv2.split(_small_copy(frame).astype(np.float32) / 255.0)
    green = chans[1]
    mean_g = float(green.mean())
    top_up = (_BLUE_COMP * max(0.0, mean_g - float(chans[0].mean())), 0.0,   # only ever adds:
              _RED_COMP * max(0.0, mean_g - float(chans[2].mean())))          # never dims a channel
    chans = [c + k * (1 - c) * green for c, k in zip(chans, top_up)]

    spread = max(float(np.percentile(c, 100 - _STRETCH_PCT) - np.percentile(c, _STRETCH_PCT))
                 for c in chans)
    enough_contrast = spread >= _MIN_CONTRAST
    haze = np.array([float(np.percentile(c, _HAZE_PCT)) if enough_contrast else 0.0 for c in chans])
    chans = [c - h for c, h in zip(chans, haze)]

    means = np.array([float(c.mean()) for c in chans])
    wb = np.clip(means.mean() / np.maximum(means, 1e-6), 1 / _MAX_GAIN, _MAX_GAIN)

    if enough_contrast:
        lo, hi = np.percentile(np.stack([c * w for c, w in zip(chans, wb)]),
                               [_STRETCH_PCT, 100 - _STRETCH_PCT])
        stretch = min(_MAX_GAIN, 1.0 / max(float(hi - lo), 1e-6))
    else:
        lo, stretch = 0.0, 1.0
    gain = wb * stretch
    offset = -(haze * wb + lo) * stretch * 255.0
    return ColourParams(top_up=tuple(map(float, top_up)), gain=tuple(map(float, gain)),
                        offset=tuple(map(float, offset)))


def _apply(channel: np.ndarray, green: np.ndarray, k: float, gain: float, offset: float) -> np.ndarray:
    """gain * (c + k * (255 - c) * green / 255) + offset, rounded and clipped to 0-255."""
    if k <= 0:
        lut = np.clip(np.arange(256) * gain + offset + 0.5, 0, 255).astype(np.uint8)
        return cv2.LUT(channel, lut)
    boost = cv2.multiply(cv2.bitwise_not(channel), green, scale=k / 255.0)
    return cv2.addWeighted(channel, gain, boost, gain, offset)


def _small_copy(frame: np.ndarray) -> np.ndarray:
    h, w = frame.shape[:2]
    if w <= _STATS_WIDTH:
        return frame
    return cv2.resize(frame, (_STATS_WIDTH, max(1, round(h * _STATS_WIDTH / w))),
                      interpolation=cv2.INTER_AREA)


def _check_frame(frame: np.ndarray) -> None:
    if (not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3
            or frame.dtype != np.uint8 or frame.size == 0):
        raise ValueError("enhance() expects a BGR uint8 image of shape (H, W, 3)")
