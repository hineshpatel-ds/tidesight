"""
Check one clip's outputs against the team contract. Run this before every push.

    python scripts/validate_outputs.py outputs/test_clip

Exit code 0 = OK, 1 = problems found.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tidesight import config  # noqa: E402
from tidesight.contracts import (load_species_list, read_jsonl, read_maxn_csv,  # noqa: E402
                                 read_segments, validate_records, validate_segments, video_info)
from tidesight.count import compute_maxn  # noqa: E402


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 1
    out_dir = Path(sys.argv[1])
    clip_id = out_dir.name
    problems: list[str] = []

    video = next((p for p in config.CLIPS_DIR.glob(f"{clip_id}.*")
                  if p.suffix.lower() in config.VIDEO_EXTS), None)
    info = video_info(video) if video else None
    if info is None:
        print(f"NOTE: no video named {clip_id}.* in data/clips - skipping fps/duration checks")

    seg_path = out_dir / config.SEGMENTS_FILE
    det_path = out_dir / config.DETECTIONS_FILE
    maxn_path = out_dir / config.MAXN_FILE
    for path in (seg_path, det_path, maxn_path):
        if not path.exists():
            problems.append(f"missing file: {path}")

    if seg_path.exists():
        problems += [f"segments: {e}" for e in
                     validate_segments(read_segments(seg_path), info["duration_s"] if info else None)]
    if det_path.exists():
        species_list = load_species_list()
        records = read_jsonl(det_path)
        problems += [f"detections: {e}" for e in
                     validate_records(records, species_list, info["fps"] if info else None)]
        if any(r["clip_id"] != clip_id for r in records):
            problems.append(f"detections: clip_id must be {clip_id!r} (the folder name)")
        if maxn_path.exists() and read_maxn_csv(maxn_path) != compute_maxn(records, species_list):
            problems.append("maxn.csv is out of date - run: python -c \"from tidesight.pipeline import "
                            f"recount; recount('{clip_id}')\"")

    if problems:
        print(f"FAIL: {len(problems)} problem(s) in {out_dir}")
        for p in problems[:30]:
            print(f"  - {p}")
        return 1
    print(f"OK: {out_dir} follows the contract")
    return 0


if __name__ == "__main__":
    sys.exit(main())
