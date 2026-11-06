"""
The team contract, in code.

Owner: App + integration. Do NOT edit without telling the team first.

Every module reads and writes data in exactly these shapes. If your output passes
validate_records() / validate_segments(), the rest of the pipeline will accept it.
"""
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path
from typing import Iterable, TypedDict

from tidesight import config


class ContractError(ValueError):
    """Raised when data does not follow the contract."""


# ======================================================================= shapes
class Segment(TypedDict):
    """A stretch of video with activity. Output of run_triage()."""
    start_s: float          # seconds from the start of the clip
    end_s: float            # seconds, must be > start_s
    activity_score: float   # 0.0 - 1.0, higher = more activity


class SpeciesScore(TypedDict):
    species: str            # species_id from data/species_list.csv
    score: float            # 0.0 - 1.0


class Record(TypedDict):
    """One fish in one frame = one line of outputs/<clip_id>/detections.jsonl."""
    clip_id: str
    frame: int                  # 0-based frame index at the ORIGINAL video fps
    time_s: float               # round(frame / fps, 3)
    bbox: list[int]             # [x1, y1, x2, y2] pixels, ORIGINAL resolution, top-left origin
    track_id: int               # >= 1, unique within a clip
    det_conf: float             # detector confidence 0.0 - 1.0
    species: str                # species_id | "unknown" | "fish" (before the species step)
    species_conf: float         # 0.0 - 1.0 (0.0 before the species step)
    top3: list[SpeciesScore]    # best first; [] before the species step
    unknown: bool               # True exactly when species == "unknown"
    reviewed: bool              # True once a person confirmed / corrected it


class MaxNRow(TypedDict):
    """One row of outputs/<clip_id>/maxn.csv."""
    clip_id: str
    species_id: str
    common_name: str
    maxn: int          # most individuals of this species in any single frame
    frame: int         # first frame where that maximum happens (the evidence)
    time_s: float
    n_tracks: int      # number of distinct tracks (secondary; overcounts fish that leave and return)


RECORD_FIELDS = ["clip_id", "frame", "time_s", "bbox", "track_id", "det_conf",
                 "species", "species_conf", "top3", "unknown", "reviewed"]
SEGMENT_FIELDS = ["start_s", "end_s", "activity_score"]
MAXN_FIELDS = ["clip_id", "species_id", "common_name", "maxn", "frame", "time_s", "n_tracks"]


# ======================================================================= small helpers
def clip_id_from_path(video_path: str | Path) -> str:
    """'Cape LaHave 2024-07-12.MP4' -> 'cape_lahave_2024_07_12'."""
    clip_id = re.sub(r"[^a-z0-9]+", "_", Path(video_path).stem.lower()).strip("_")
    if not clip_id:
        raise ContractError(f"Cannot make a clip_id from {video_path!r}")
    return clip_id


def time_from_frame(frame: int, fps: float) -> float:
    return round(frame / fps, 3)


def detect_step(fps: float) -> int:
    """Look at every Nth frame so the detector sees about config.DETECT_FPS frames per second."""
    return max(1, round(fps / config.DETECT_FPS))


def frames_to_process(segments: Iterable[Segment], fps: float, frame_count: int, step: int) -> list[int]:
    """Original frame numbers inside the segments, on a global grid of every `step` frames."""
    frames: set[int] = set()
    for seg in segments:
        first = math.ceil(seg["start_s"] * fps - 1e-9)
        first = ((first + step - 1) // step) * step
        last = min(math.floor(seg["end_s"] * fps + 1e-9), frame_count - 1)
        frames.update(range(first, last + 1, step))
    return sorted(frames)


def clip_output_dir(clip_id: str) -> Path:
    out = config.OUTPUTS_DIR / clip_id
    out.mkdir(parents=True, exist_ok=True)
    return out


def video_info(video_path: str | Path) -> dict:
    """fps, frame_count, width, height, duration_s of a video."""
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if fps <= 0 or frame_count <= 0:
        raise ContractError(f"Video has no usable fps/frame count: {video_path}")
    return {"fps": fps, "frame_count": frame_count, "width": width, "height": height,
            "duration_s": round(frame_count / fps, 3)}


def new_record(clip_id: str, frame: int, fps: float, bbox: Iterable[float],
               track_id: int, det_conf: float) -> Record:
    """The ONLY way the detector should create a record (fills species fields with defaults)."""
    x1, y1, x2, y2 = (int(round(v)) for v in bbox)
    return Record(clip_id=clip_id, frame=int(frame), time_s=time_from_frame(frame, fps),
                  bbox=[x1, y1, x2, y2], track_id=int(track_id), det_conf=round(float(det_conf), 3),
                  species=config.FISH_PLACEHOLDER, species_conf=0.0, top3=[],
                  unknown=False, reviewed=False)


def set_track_species(records: list[Record], track_id: int, species: str, species_conf: float,
                      top3: list[SpeciesScore] | None = None, reviewed: bool = False) -> int:
    """Give EVERY row of one track the same species fields. Returns how many rows changed.

    Used by the species step (reviewed=False) and by the review screen (reviewed=True).
    """
    changed = 0
    for rec in records:
        if rec["track_id"] == track_id:
            rec["species"] = species
            rec["species_conf"] = round(float(species_conf), 3)
            if top3 is not None:
                rec["top3"] = [SpeciesScore(species=s["species"], score=round(float(s["score"]), 3))
                               for s in top3]
            rec["unknown"] = species == config.UNKNOWN_LABEL
            rec["reviewed"] = reviewed
            changed += 1
    return changed


# ======================================================================= file I/O
def write_jsonl(records: Iterable[Record], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for rec in records:
            f.write(json.dumps({k: rec[k] for k in RECORD_FIELDS}) + "\n")


def read_jsonl(path: str | Path) -> list[Record]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_segments(segments: list[Segment], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump([{k: s[k] for k in SEGMENT_FIELDS} for s in segments], f, indent=2)


def read_segments(path: str | Path) -> list[Segment]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def write_maxn_csv(rows: list[MaxNRow], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MAXN_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def read_maxn_csv(path: str | Path) -> list[MaxNRow]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        row["maxn"], row["frame"], row["n_tracks"] = int(row["maxn"]), int(row["frame"]), int(row["n_tracks"])
        row["time_s"] = float(row["time_s"])
    return rows


def load_species_list(path: str | Path = config.SPECIES_LIST) -> dict[str, dict]:
    """species_id -> row of data/species_list.csv, in file order."""
    with open(path, encoding="utf-8", newline="") as f:
        return {row["species_id"]: row for row in csv.DictReader(f)}


# ======================================================================= validation
def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _is_num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def validate_segments(segments: list[Segment], duration_s: float | None = None) -> list[str]:
    """Return a list of problems ([] = OK)."""
    errors: list[str] = []
    if not isinstance(segments, list):
        return ["segments must be a list"]
    prev_end = -1.0
    for i, seg in enumerate(segments):
        where = f"segment {i}"
        if not isinstance(seg, dict) or set(seg) != set(SEGMENT_FIELDS):
            errors.append(f"{where}: keys must be exactly {SEGMENT_FIELDS}")
            continue
        start, end, score = seg["start_s"], seg["end_s"], seg["activity_score"]
        if not all(_is_num(v) for v in (start, end, score)):
            errors.append(f"{where}: values must be numbers")
            continue
        if not 0 <= start < end:
            errors.append(f"{where}: need 0 <= start_s < end_s (got {start}, {end})")
        if duration_s is not None and end > duration_s + 0.001:
            errors.append(f"{where}: end_s {end} is past the end of the video ({duration_s})")
        if not 0 <= score <= 1:
            errors.append(f"{where}: activity_score must be 0-1 (got {score})")
        if start < prev_end:
            errors.append(f"{where}: segments must be sorted and must not overlap")
        prev_end = end
    return errors


def validate_records(records: list[Record], species_ids: Iterable[str] | None = None,
                     fps: float | None = None) -> list[str]:
    """Return a list of problems ([] = OK). Pass species_ids and fps for the strictest check."""
    errors: list[str] = []
    allowed = None if species_ids is None else set(species_ids) | {config.UNKNOWN_LABEL, config.FISH_PLACEHOLDER}
    seen_keys: set[tuple[int, int]] = set()
    track_labels: dict[int, tuple] = {}
    clip_ids: set[str] = set()

    for i, rec in enumerate(records):
        where = f"record {i + 1}"
        if not isinstance(rec, dict):
            errors.append(f"{where}: must be a JSON object")
            continue
        missing, extra = set(RECORD_FIELDS) - set(rec), set(rec) - set(RECORD_FIELDS)
        if missing or extra:
            errors.append(f"{where}: missing {sorted(missing)} / extra {sorted(extra)}")
            continue
        clip_ids.add(rec["clip_id"])

        if not _is_int(rec["frame"]) or rec["frame"] < 0:
            errors.append(f"{where}: frame must be an int >= 0")
        if not _is_num(rec["time_s"]) or rec["time_s"] < 0:
            errors.append(f"{where}: time_s must be a number >= 0")
        elif fps and _is_int(rec["frame"]) and abs(rec["time_s"] - time_from_frame(rec["frame"], fps)) > 0.0015:
            errors.append(f"{where}: time_s {rec['time_s']} != round(frame / fps, 3) "
                          f"= {time_from_frame(rec['frame'], fps)} (are you using ORIGINAL frame numbers?)")

        bbox = rec["bbox"]
        if not (isinstance(bbox, list) and len(bbox) == 4 and all(_is_int(v) for v in bbox)):
            errors.append(f"{where}: bbox must be [x1, y1, x2, y2] ints")
        elif not (0 <= bbox[0] < bbox[2] and 0 <= bbox[1] < bbox[3]):
            errors.append(f"{where}: bbox needs 0 <= x1 < x2 and 0 <= y1 < y2 (got {bbox})")

        if not _is_int(rec["track_id"]) or rec["track_id"] < 1:
            errors.append(f"{where}: track_id must be an int >= 1")
        for key in ("det_conf", "species_conf"):
            if not _is_num(rec[key]) or not 0 <= rec[key] <= 1:
                errors.append(f"{where}: {key} must be a number 0-1")

        if not isinstance(rec["species"], str):
            errors.append(f"{where}: species must be a string")
        elif allowed is not None and rec["species"] not in allowed:
            errors.append(f"{where}: species {rec['species']!r} is not in data/species_list.csv")

        top3 = rec["top3"]
        if not isinstance(top3, list) or len(top3) > 3 or not all(
                isinstance(t, dict) and set(t) == {"species", "score"} and _is_num(t["score"]) for t in top3):
            errors.append(f"{where}: top3 must be a list of up to 3 {{species, score}} objects")
        elif [t["score"] for t in top3] != sorted((t["score"] for t in top3), reverse=True):
            errors.append(f"{where}: top3 must be sorted best first")

        if not isinstance(rec["unknown"], bool) or not isinstance(rec["reviewed"], bool):
            errors.append(f"{where}: unknown and reviewed must be true/false")
        elif rec["unknown"] != (rec["species"] == config.UNKNOWN_LABEL):
            errors.append(f"{where}: unknown must be true exactly when species == 'unknown'")

        key = (rec["frame"], rec["track_id"])
        if key in seen_keys:
            errors.append(f"{where}: track {rec['track_id']} appears twice in frame {rec['frame']}")
        seen_keys.add(key)

        label = (rec["species"], rec["species_conf"], rec["unknown"], rec["reviewed"])
        first = track_labels.setdefault(rec["track_id"], label)
        if label != first:
            errors.append(f"{where}: track {rec['track_id']} has different species fields on different rows "
                          f"(species is decided per TRACK - use set_track_species)")

    if len(clip_ids) > 1:
        errors.append(f"records mix several clip_ids: {sorted(clip_ids)}")
    return errors
