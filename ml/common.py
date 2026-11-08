from __future__ import annotations

import json
import logging
from pathlib import Path

from tidesight import config

ML_DIR = Path(__file__).resolve().parent
DATA_DIR = ML_DIR / "data"
LABELS_DIR = ML_DIR / "labels"
RUNS_DIR = ML_DIR / "runs"
SPLITS = json.loads((ML_DIR / "splits.json").read_text())

_PROPOSED = dict(DETECTOR_WEIGHTS=config.MODELS_DIR / "tidesight-fish-v1.pt", DETECT_CONF=0.18,
                 DETECT_IMGSZ=1280, DETECT_INPUT_WIDTH=1920, DETECT_PREPROCESS="stretch1-99_clahe3-8x8_v1",
                 TRACK_LINK_DIST=1.5, TRACK_MAX_MISSED=2, TRACK_MAX_GAP_S=1.0,
                 PROTOCOL_N_SAMPLES=15, PROTOCOL_INTERVAL_S=60.0, PROTOCOL_EDGE_S=60.0)


def ensure_settings() -> None:
    if not hasattr(config, "DETECT_PREPROCESS"):
        logging.warning("config.py has no detector settings; using defaults for this script.")
        for k, v in _PROPOSED.items():
            setattr(config, k, v)


def frame_name(t_s: int) -> str:
    return f"{SPLITS['video']['clip_id']}_{t_s:04d}s"


def pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "0"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
