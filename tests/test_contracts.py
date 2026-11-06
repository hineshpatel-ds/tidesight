"""Run with:  python -m pytest -q"""
from tidesight import config
from tidesight.contracts import (clip_id_from_path, frames_to_process, new_record,
                                 set_track_species, validate_records, validate_segments)
from tidesight.count import compute_maxn

FPS = 25.0
SPECIES = {"atlantic_cod": {"common_name": "Atlantic cod"}, "cunner": {"common_name": "Cunner"}}


def make(frame, track_id, bbox=(10, 10, 50, 40)):
    return new_record("clip", frame, FPS, bbox, track_id, 0.8)


def test_clip_id_from_path():
    assert clip_id_from_path("Cape LaHave 2024-07-12.MP4") == "cape_lahave_2024_07_12"


def test_frames_are_original_numbers_on_a_grid():
    segs = [{"start_s": 0.0, "end_s": 1.0, "activity_score": 1.0},
            {"start_s": 2.1, "end_s": 2.5, "activity_score": 0.5}]
    assert frames_to_process(segs, FPS, 1000, 5) == [0, 5, 10, 15, 20, 25, 55, 60]


def test_maxn_counts_distinct_tracks_in_one_frame():
    recs = [make(0, 1), make(0, 2), make(5, 1), make(5, 2), make(5, 3), make(10, 4)]
    for tid in (1, 2, 3):
        set_track_species(recs, tid, "atlantic_cod", 0.9)
    set_track_species(recs, 4, "cunner", 0.7)
    rows = {r["species_id"]: r for r in compute_maxn(recs, SPECIES)}
    assert rows["atlantic_cod"]["maxn"] == 3 and rows["atlantic_cod"]["frame"] == 5
    assert rows["atlantic_cod"]["n_tracks"] == 3
    assert rows["cunner"]["maxn"] == 1


def test_valid_records_pass():
    recs = [make(0, 1), make(5, 1)]
    set_track_species(recs, 1, config.UNKNOWN_LABEL, 0.3, [{"species": "cunner", "score": 0.3}])
    assert validate_records(recs, SPECIES, FPS) == []


def test_validator_catches_common_mistakes():
    recs = [make(0, 1), make(5, 1), make(5, 1)]          # duplicate track in one frame
    recs[1]["species"] = "cunner"                          # track 1 has two species
    recs[0]["bbox"] = [50, 10, 10, 40]                     # x1 > x2
    recs[2]["time_s"] = 5.0                                # sampled index used as time
    errors = " ".join(validate_records(recs, SPECIES, FPS))
    assert "appears twice" in errors
    assert "per TRACK" in errors
    assert "x1 < x2" in errors
    assert "ORIGINAL frame numbers" in errors


def test_segments_must_be_sorted_and_inside_video():
    segs = [{"start_s": 5.0, "end_s": 8.0, "activity_score": 1.0},
            {"start_s": 2.0, "end_s": 3.0, "activity_score": 1.0}]
    errors = " ".join(validate_segments(segs, duration_s=7.0))
    assert "past the end" in errors and "sorted" in errors
