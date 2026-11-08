"""Validate saved analysis without modifying it or its source video.

    python scripts/validate_outputs.py outputs/test_clip --video data/clips/test_clip.mp4 --strict

Exit codes: 0 = complete or explicitly PARTIAL validation, 1 = failure, 2 = CLI usage error.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tidesight import config  # noqa: E402
from tidesight.contracts import (MAXN_FIELDS, RECORD_FIELDS, load_species_list,  # noqa: E402
                                 read_maxn_csv, validate_records, validate_segments, video_info)
from tidesight.count import compute_maxn  # noqa: E402


def _number(value) -> bool:
    try:
        return type(value) in (int, float) and math.isfinite(value)
    except OverflowError:
        return False


def _load(path: Path, reader, problems: list[str]):
    """Keep expected input errors local to their file; None means unusable evidence."""
    try:
        result = reader(path)
        if result is None:
            raise ValueError("expected analysis data, got null")
        return result
    except (OSError, ValueError, TypeError, KeyError, OverflowError, csv.Error) as exc:
        problems.append(f"{path.name}: {exc}")
        return None


def _read_detections(path: Path) -> list:
    records = []
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except ValueError as exc:
                raise ValueError(f"line {line_number}: {exc}") from exc
    return records


def _read_counts(path: Path) -> list:
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != MAXN_FIELDS:
            raise ValueError(f"header must be {','.join(MAXN_FIELDS)}")
        for line_number, row in enumerate(reader, 2):
            if set(row) != set(MAXN_FIELDS) or any(value is None for value in row.values()):
                raise ValueError(f"row {line_number}: wrong number of columns")
    return read_maxn_csv(path)


def _record_problems(records: list, species: dict, fps: float | None) -> list[str]:
    """Guard nested types before passing data to the shared contract validator."""
    problems = []
    for index, rec in enumerate(records, 1):
        where = f"record {index}"
        if not isinstance(rec, dict) or set(rec) != set(RECORD_FIELDS):
            problems.append(f"{where}: keys must be exactly {RECORD_FIELDS}")
            continue
        for key in ("clip_id", "species"):
            if not isinstance(rec[key], str):
                problems.append(f"{where}: {key} must be a string")
        for key in ("frame", "track_id"):
            if type(rec[key]) is not int or not _number(rec[key]):
                problems.append(f"{where}: {key} must be a finite representable integer")
        for key in ("time_s", "det_conf", "species_conf"):
            if not _number(rec[key]):
                problems.append(f"{where}: {key} must be a finite number")
        if rec["species"] == config.FISH_PLACEHOLDER:
            problems.append(f"{where}: unfinished 'fish' species in final analysis")
        elif (rec["reviewed"] is False and _number(rec["species_conf"])
              and (rec["species"] == config.UNKNOWN_LABEL)
              != (rec["species_conf"] < config.UNKNOWN_THRESHOLD)):
            problems.append(f"{where}: unreviewed species must follow unknown threshold "
                            f"{config.UNKNOWN_THRESHOLD}")
        top3 = rec["top3"]
        if not isinstance(top3, list) or len(top3) > 3:
            problems.append(f"{where}: top3 must be a list of up to 3 suggestions")
        else:
            for suggestion in top3:
                if (not isinstance(suggestion, dict) or set(suggestion) != {"species", "score"}
                        or not isinstance(suggestion["species"], str)
                        or suggestion["species"] not in species
                        or not _number(suggestion["score"])
                        or not 0 <= suggestion["score"] <= 1):
                    problems.append(f"{where}: top3 needs catalog species and finite scores in 0-1")
    if problems:
        return problems
    return validate_records(records, species, fps)


def _segment_problems(segments, duration: float | None) -> list[str]:
    if isinstance(segments, list):
        for index, seg in enumerate(segments):
            if isinstance(seg, dict) and any(not _number(v) for v in seg.values()):
                return [f"segment {index}: values must be finite numbers"]
    return validate_segments(segments, duration)


def _source(args, clip_id: str, problems: list[str]):
    video = args.video
    if video is None:
        # Compare stems literally: clip IDs must not become glob expressions.
        candidates = sorted(p for p in config.CLIPS_DIR.glob("*")
                            if p.stem == clip_id and p.suffix.lower() in config.VIDEO_EXTS
                            and p.is_file())
        if len(candidates) == 1:
            video = candidates[0]
        elif len(candidates) > 1:
            problems.append("source video is ambiguous; select one with --video PATH")
        elif args.strict:
            problems.append("source video is missing; supply --video PATH for strict validation")
    if video is None:
        return None
    print(f"Source video: {video}")
    try:
        if not video.is_file():
            raise ValueError(f"not a readable file: {video}")
        info = video_info(video)
        if not all(_number(info[k]) and info[k] > 0 for k in
                   ("fps", "frame_count", "width", "height", "duration_s")):
            raise ValueError("unusable fps/frame count/dimensions/duration")
        return info
    except (OSError, ValueError, OverflowError) as exc:
        problems.append(f"source video: {exc}")
        return None


def _video_checks(records: list, segments: list | None, info: dict) -> list[str]:
    problems = []
    for index, rec in enumerate(records, 1):
        where = f"record {index}"
        frame = rec["frame"]
        if not 0 <= frame < info["frame_count"]:
            problems.append(f"{where}: frame {frame} outside source frame bounds "
                            f"0..{info['frame_count'] - 1}")
        x1, y1, x2, y2 = rec["bbox"]
        if not (0 <= x1 < x2 <= info["width"] and 0 <= y1 < y2 <= info["height"]):
            problems.append(f"{where}: bbox outside source {info['width']}x{info['height']}")
        # Membership uses original frame time, not the rounded saved timestamp.
        time_s = frame / info["fps"]
        if segments is not None and not any(
                seg["start_s"] - 1e-9 <= time_s <= seg["end_s"] + 1e-9 for seg in segments):
            problems.append(f"{where}: frame {frame} is outside every retained segment")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--video", type=Path, help="actual source video; overrides filename discovery")
    parser.add_argument("--strict", action="store_true", help="fail if source-video evidence is missing")
    args = parser.parse_args()
    out_dir = args.output_dir
    clip_id = out_dir.name
    problems: list[str] = []
    info = _source(args, clip_id, problems)
    if info is None:
        print("SKIPPED: source-video checks: original frame bounds, fps/timestamps, "
              "box dimensions, segment duration and source-time membership")

    segments = _load(out_dir / config.SEGMENTS_FILE,
                     lambda p: json.loads(p.read_text(encoding="utf-8")), problems)
    if segments is not None:
        errors = _segment_problems(segments, info["duration_s"] if info else None)
        problems.extend(f"{config.SEGMENTS_FILE}: {e}" for e in errors)
        if errors:
            segments = None
    records = _load(out_dir / config.DETECTIONS_FILE, _read_detections, problems)
    counts = _load(out_dir / config.MAXN_FILE, _read_counts, problems)
    species = _load(config.SPECIES_LIST, load_species_list, problems)
    if species is not None and not species:
        problems.append(f"{config.SPECIES_LIST.name}: species catalog is empty")
    if records is not None and species:
        errors = _record_problems(records, species, info["fps"] if info else None)
        problems.extend(f"{config.DETECTIONS_FILE}: {e}" for e in errors)
        if not errors:
            if any(rec["clip_id"] != clip_id for rec in records):
                problems.append(f"{config.DETECTIONS_FILE}: clip_id must be {clip_id!r} (folder name)")
            if info:
                problems.extend(f"{config.DETECTIONS_FILE}: {e}" for e in
                                _video_checks(records, segments, info))
            if counts is not None and counts != compute_maxn(records, species):
                problems.append(f"{config.MAXN_FILE}: does not match MaxN recomputed from detections")
    if problems:
        print(f"FAIL: {len(problems)} problem(s) in {out_dir}")
        for problem in problems[:30]:
            print(f"  - {problem}")
        return 1
    if info is None:
        print(f"PARTIAL: {out_dir} passed available checks; source-video evidence missing")
    else:
        print(f"OK: {out_dir} passed complete analysis consistency validation (not model accuracy)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
