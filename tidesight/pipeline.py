"""
The pipeline that glues everyone's work together.

Owner: App + integration
Branch: feat/app

This is the ONLY module that writes to outputs/. It checks every step against the
contract and stops with a clear error if a step returns bad data.

Run it from the project folder:
    python -m tidesight.pipeline data\\clips\\test_clip.mp4
"""
from __future__ import annotations

import argparse
import logging
import shutil
import tempfile
from pathlib import Path
from uuid import uuid4

from tidesight import config
from tidesight.contracts import (ContractError, MaxNRow, Record, clip_id_from_path,
                                 clip_output_dir, load_species_list, read_jsonl,
                                 validate_records, validate_segments, video_info,
                                 write_jsonl, write_maxn_csv, write_segments)
from tidesight.count import compute_maxn
from tidesight.detect import run_detection
from tidesight.species import identify_species
from tidesight.triage import run_triage

log = logging.getLogger(__name__)


def _check(step: str, errors: list[str]) -> None:
    if errors:
        shown = "\n  - ".join(errors[:10])
        more = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise ContractError(f"{step} broke the contract:\n  - {shown}{more}")


def _start_energy_tracker(out_dir: Path):
    tracker = None
    try:
        from codecarbon import OfflineEmissionsTracker
        tracker = OfflineEmissionsTracker(country_iso_code=config.ENERGY_COUNTRY,
                                          output_dir=str(out_dir), output_file=config.ENERGY_FILE,
                                          log_level="error", save_to_file=True)
        tracker.start()
        return tracker
    except Exception as exc:  # energy is a nice-to-have; never block the pipeline
        log.warning("Energy tracking disabled: %s", exc)
        # start() may have partially initialized the tracker before raising.
        _stop_energy_tracker(tracker)
        return None


def _stop_energy_tracker(tracker) -> tuple[float | None, float | None]:
    if tracker is None:
        return None, None
    try:
        emissions_kg = tracker.stop()
        data = getattr(tracker, "final_emissions_data", None)
        energy_kwh = getattr(data, "energy_consumed", None)
        return energy_kwh, emissions_kg
    except Exception as exc:
        log.warning("Energy tracking failed: %s", exc)
        return None, None


def _cleanup_run_directory(directory: Path) -> None:
    """Cleanup is best effort; it must not change an already decided run outcome."""
    try:
        if directory.exists():
            shutil.rmtree(directory)
    except Exception as exc:
        log.warning("Output cleanup failed; retained %s: %s", directory, exc)


def _publish_run(staging: Path, out_dir: Path) -> None:
    """Publish for a single writer, rolling back ordinary rename exceptions.

    Two renames leave a brief missing-directory window. This is not a crash-safe
    transaction and does not coordinate concurrent readers, analysis or review.
    """
    backup = out_dir.with_name(f".{out_dir.name}-backup-{uuid4().hex}")
    had_previous = out_dir.exists()
    if had_previous:
        out_dir.rename(backup)
    try:
        staging.rename(out_dir)
    except Exception:
        if had_previous:
            try:
                backup.rename(out_dir)
            except Exception as recovery_error:
                # Never clean up this backup: it holds the last published data.
                raise RuntimeError(
                    f"Output publication and recovery failed. Previous results retained at "
                    f"{backup}; restore that directory to {out_dir} before retrying."
                ) from recovery_error
        raise
    if had_previous:
        _cleanup_run_directory(backup)


def run_pipeline(video_path: str | Path, measure_energy: bool = True) -> dict:
    """Stage a complete analysis, publish it, then return its published paths.

    On ordinary exceptions the last complete results survive, provided filesystem
    rollback works. A failed rollback retains a backup and reports its location.
    Process crashes/power loss and concurrent writers are outside this guarantee.
    """
    video_path = Path(video_path)
    clip_id = clip_id_from_path(video_path)
    out_dir = config.OUTPUTS_DIR / clip_id
    info = video_info(video_path)
    species_list = load_species_list()

    config.OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{clip_id}-run-", dir=config.OUTPUTS_DIR))
    try:
        tracker = _start_energy_tracker(staging) if measure_energy else None
        try:
            segments = run_triage(video_path)
            _check("run_triage", validate_segments(segments, info["duration_s"]))
            write_segments(segments, staging / config.SEGMENTS_FILE)

            records = run_detection(video_path, segments)
            _check("run_detection", validate_records(records, species_list, info["fps"]))

            records = identify_species(video_path, records)
            _check("identify_species", validate_records(records, species_list, info["fps"]))
            write_jsonl(records, staging / config.DETECTIONS_FILE)

            maxn = compute_maxn(records, species_list)
            write_maxn_csv(maxn, staging / config.MAXN_FILE)
        finally:
            energy_kwh, emissions_kg = _stop_energy_tracker(tracker)

        if energy_kwh is None or emissions_kg is None:
            # A failed start/stop may have left a partially written CSV.
            (staging / config.ENERGY_FILE).unlink(missing_ok=True)
            energy_kwh, emissions_kg = None, None
        busy_s = sum(s["end_s"] - s["start_s"] for s in segments)
        result = {
            "clip_id": clip_id,
            "output_dir": str(out_dir),
            "duration_s": info["duration_s"],
            "busy_s": round(busy_s, 3),
            "skipped_pct": round(100 * (1 - busy_s / info["duration_s"]), 1),
            "n_records": len(records),
            "n_tracks": len({r["track_id"] for r in records}),
            "segments": segments,
            "maxn": maxn,
            "energy_kwh": energy_kwh,
            "emissions_kg": emissions_kg,
        }
        _publish_run(staging, out_dir)
        return result
    finally:
        _cleanup_run_directory(staging)


def save_review(clip_id: str, records: list[Record]) -> list[MaxNRow]:
    """Called by the review screen after a person corrects species. Re-validates and recounts."""
    species_list = load_species_list()
    _check("review", validate_records(records, species_list))
    out_dir = clip_output_dir(clip_id)
    write_jsonl(records, out_dir / config.DETECTIONS_FILE)
    maxn = compute_maxn(records, species_list)
    write_maxn_csv(maxn, out_dir / config.MAXN_FILE)
    return maxn


def recount(clip_id: str) -> list[MaxNRow]:
    """Recompute maxn.csv from the saved detections.jsonl."""
    out_dir = config.OUTPUTS_DIR / clip_id
    records = read_jsonl(out_dir / config.DETECTIONS_FILE)
    return save_review(clip_id, records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the TideSight pipeline on one video.")
    parser.add_argument("video_path")
    parser.add_argument("--no-energy", action="store_true", help="skip CodeCarbon energy tracking")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    result = run_pipeline(args.video_path, measure_energy=not args.no_energy)
    print(f"\nClip:      {result['clip_id']}  ({result['duration_s']} s)")
    print(f"Skipped:   {result['skipped_pct']}% of the video (triage)")
    print(f"Detected:  {result['n_records']} boxes, {result['n_tracks']} tracks")
    print("MaxN:")
    for row in result["maxn"]:
        print(f"  {row['common_name']:<35} MaxN {row['maxn']}  at {row['time_s']} s  ({row['n_tracks']} tracks)")
    if result["energy_kwh"] is not None:
        print(f"Energy:    {result['energy_kwh'] * 1000:.3f} Wh, {result['emissions_kg'] * 1000:.3f} g CO2e")
    print(f"Outputs:   {result['output_dir']}")


if __name__ == "__main__":
    main()
