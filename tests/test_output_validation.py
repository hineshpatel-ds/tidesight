"""Read-only integration acceptance checks, using synthetic source evidence."""
import json
import sys

import cv2
import numpy as np
import pytest

from scripts import validate_outputs as validator


HEADER = "clip_id,species_id,common_name,maxn,frame,time_s,n_tracks\n"
ROW = "clip,atlantic_cod,Atlantic cod,1,5,0.5,1\n"


@pytest.fixture
def analysis(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    monkeypatch.setattr(validator.config, "CLIPS_DIR", clips)
    video = clips / "different_name.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    assert writer.isOpened()
    for _ in range(20):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()
    out = tmp_path / "clip"
    out.mkdir()
    (out / "segments.json").write_text(
        '[{"start_s": 0.5, "end_s": 1.0, "activity_score": 0.8}]', encoding="utf-8")
    rec = dict(clip_id="clip", frame=5, time_s=0.5, bbox=[0, 0, 64, 48],
               track_id=1, det_conf=0.8, species="atlantic_cod", species_conf=0.9,
               top3=[dict(species="atlantic_cod", score=0.9)], unknown=False, reviewed=False)
    (out / "detections.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")
    (out / "maxn.csv").write_text(HEADER + ROW, encoding="utf-8")
    return out, video, rec


def run(monkeypatch, capsys, out, *args):
    monkeypatch.setattr(sys, "argv", ["validate_outputs.py", str(out), *map(str, args)])
    code = validator.main()
    output = capsys.readouterr().out
    assert "Traceback" not in output
    return code, output


def save_record(out, rec):
    (out / "detections.jsonl").write_text(json.dumps(rec) + "\n", encoding="utf-8")


def test_explicit_source_complete_and_read_only(analysis, monkeypatch, capsys):
    out, video, _ = analysis
    paths = [video, *out.iterdir()]
    before = {p: p.read_bytes() for p in paths}
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == 0 and "OK:" in output and str(video) in output
    assert {p: p.read_bytes() for p in paths} == before
    assert set(out.iterdir()) == set(paths[1:])


def test_missing_source_is_partial_or_strict_failure(analysis, monkeypatch, capsys):
    out, _, _ = analysis
    code, output = run(monkeypatch, capsys, out)
    assert code == 0 and "PARTIAL:" in output and "SKIPPED" in output and "OK:" not in output
    code, output = run(monkeypatch, capsys, out, "--strict")
    assert code == 1 and "source video" in output and "OK:" not in output


def test_automatic_source_and_ambiguity(analysis, monkeypatch, capsys):
    out, video, _ = analysis
    matching = video.with_name("clip.avi")
    matching.write_bytes(video.read_bytes())
    assert run(monkeypatch, capsys, out, "--strict")[0] == 0
    matching.with_suffix(".mp4").write_bytes(video.read_bytes())
    code, output = run(monkeypatch, capsys, out, "--strict")
    assert code == 1 and "ambiguous" in output and "--video" in output
    assert run(monkeypatch, capsys, out, "--video", video, "--strict")[0] == 0


@pytest.mark.parametrize("field,value,reason", [
    ("frame", 20, "frame"), ("frame", -1, "frame"),
    ("time_s", 0.7, "time_s"), ("time_s", float("nan"), "finite"),
    ("time_s", 10**400, "finite"), ("frame", 10**400, "frame"),
    ("bbox", [0, 0, 65, 48], "bbox"), ("bbox", [0, 0, 64, 49], "bbox"),
    ("bbox", [1, 0, 1, 2], "bbox"), ("bbox", [False, 0, 5, 5], "bbox"),
    ("species", "fish", "fish"), ("clip_id", "other", "clip_id"),
    ("clip_id", [], "clip_id"), ("track_id", {}, "track_id"),
    ("top3", [{"species": "invented", "score": 0.9}], "top3"),
    ("top3", [{"species": "unknown", "score": 0.9}], "top3"),
    ("top3", [{"species": "atlantic_cod", "score": 1.1}], "top3"),
    ("top3", [{"species": "atlantic_cod", "score": -0.1}], "top3"),
    ("top3", [{"species": [], "score": 0.9}], "top3"),
    ("top3", [{"species": "atlantic_cod", "score": True}], "top3"),
    ("top3", [{"species": "atlantic_cod", "score": float("inf")}], "top3"),
    ("top3", [{"species": "atlantic_cod", "score": 0.2},
              {"species": "cunner", "score": 0.9}], "top3"),
])
def test_invalid_record(analysis, monkeypatch, capsys, field, value, reason):
    out, video, rec = analysis
    rec[field] = value
    save_record(out, rec)
    before = {p: p.read_bytes() for p in [video, *out.iterdir()]}
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == 1 and reason in output and "OK:" not in output
    assert {p: p.read_bytes() for p in before} == before


@pytest.mark.parametrize("frame,time_s,valid", [(4, 0.4, False), (5, 0.5, True),
                                                 (10, 1.0, True), (11, 1.1, False)])
def test_retained_segment_membership(analysis, monkeypatch, capsys, frame, time_s, valid):
    out, video, rec = analysis
    rec.update(frame=frame, time_s=time_s)
    save_record(out, rec)
    (out / "maxn.csv").write_text(HEADER + f"clip,atlantic_cod,Atlantic cod,1,{frame},{time_s},1\n")
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == (0 if valid else 1)
    if not valid:
        assert "retained segment" in output


@pytest.mark.parametrize("segments", ["[]", '[{"start_s":0,"end_s":2,"activity_score":0}]'])
def test_empty_analysis_is_valid(analysis, monkeypatch, capsys, segments):
    out, video, _ = analysis
    (out / "segments.json").write_text(segments)
    (out / "detections.jsonl").write_text("")
    (out / "maxn.csv").write_text(HEADER)
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == 0 and "OK:" in output


@pytest.mark.parametrize("name,content", [
    ("segments.json", None), ("detections.jsonl", None), ("maxn.csv", None),
    ("segments.json", "{"), ("segments.json", "{}"), ("segments.json", "null"),
    ("segments.json", '[{"start_s":0,"end_s":Infinity,"activity_score":1}]'),
    ("segments.json", '[{"start_s":0,"end_s":3,"activity_score":1}]'),
    ("segments.json", json.dumps([dict(start_s=0, end_s=10**400, activity_score=1)])),
    ("detections.jsonl", "{}\n"), ("detections.jsonl", "null\n"),
    ("detections.jsonl", "{broken\n"), ("maxn.csv", ""),
    ("maxn.csv", "wrong,header\n"),
    ("maxn.csv", HEADER + "clip,atlantic_cod,Atlantic cod,no,5,0.5,1\n"),
    ("maxn.csv", HEADER),
])
def test_file_errors_are_actionable(analysis, monkeypatch, capsys, name, content):
    out, video, _ = analysis
    path = out / name
    if content is None:
        path.unlink()
    else:
        path.write_text(content)
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == 1 and name in output


def test_bad_explicit_video_never_falls_back(analysis, monkeypatch, capsys):
    out, video, _ = analysis
    video.with_name("clip.avi").write_bytes(video.read_bytes())
    for source in [video.with_name("missing.avi"), out / "maxn.csv"]:
        code, output = run(monkeypatch, capsys, out, "--video", source)
        assert code == 1 and "source video" in output


def test_reviewed_species_preserves_model_evidence(analysis, monkeypatch, capsys):
    out, video, rec = analysis
    rec.update(reviewed=True, species_conf=0.3)
    rec["top3"] = [{"species": "cunner", "score": 0.3}]
    save_record(out, rec)
    assert run(monkeypatch, capsys, out, "--video", video, "--strict")[0] == 0


@pytest.mark.parametrize("species,confidence,unknown,valid", [
    ("atlantic_cod", 0.49, False, False), ("unknown", 0.5, True, False),
    ("atlantic_cod", 0.5, False, True), ("unknown", 0.49, True, True),
])
def test_unreviewed_unknown_threshold(analysis, monkeypatch, capsys, species, confidence, unknown, valid):
    out, video, rec = analysis
    rec.update(species=species, species_conf=confidence, unknown=unknown)
    save_record(out, rec)
    row = ROW if species == "atlantic_cod" else "clip,unknown,Unknown (needs expert review),1,5,0.5,1\n"
    (out / "maxn.csv").write_text(HEADER + row)
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == (0 if valid else 1)
    if not valid:
        assert "threshold" in output


def test_membership_uses_original_time_not_rounded_timestamp(analysis, monkeypatch, capsys):
    out, video, rec = analysis
    # Contract tolerates this saved rounding error, but source frame 5 is at 0.5s.
    rec["time_s"] = 0.501
    save_record(out, rec)
    (out / "segments.json").write_text('[{"start_s":0.501,"end_s":1,"activity_score":1}]')
    (out / "maxn.csv").write_text(HEADER + ROW.replace("0.5", "0.501"))
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == 1 and "retained segment" in output


def test_missing_species_evidence(analysis, monkeypatch, capsys):
    out, video, _ = analysis
    monkeypatch.setattr(validator.config, "SPECIES_LIST", out / "missing_species.csv")
    code, output = run(monkeypatch, capsys, out, "--video", video, "--strict")
    assert code == 1 and "missing_species.csv" in output
