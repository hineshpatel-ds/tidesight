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

The baseline below (CLAHE contrast boost) already works. Improve it if time allows:
colour correction (red-channel compensation / white balance), dehazing, denoise.
"""
from __future__ import annotations

import cv2
import numpy as np

_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def enhance(frame: np.ndarray) -> np.ndarray:
    if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("enhance() expects a BGR image of shape (H, W, 3)")
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    l_channel = _CLAHE.apply(l_channel)
    return cv2.cvtColor(cv2.merge((l_channel, a_channel, b_channel)), cv2.COLOR_LAB2BGR)
