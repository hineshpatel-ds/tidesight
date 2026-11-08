from __future__ import annotations

import pytest

from tidesight import config, detect
from tidesight.count import (ProtocolError, compute_maxn, compute_meancount, nearest_frame,
                             protocol_sample_frames)
from test_detect import FakeCapture, FakeModel, frame_index
from test_detect import detector_config

FPS_CASES = [24.0, 25.0, 30000 / 1001, 30.0]


def _rec(frame, track, species="fish", fps=30.0):
    return {"clip_id": "c", "frame": frame, "time_s": round(frame / fps, 3), "bbox": [0, 0, 1, 1],
            "track_id": track, "det_conf": 0.9, "species": species, "species_conf": 0.0,
            "top3": [], "unknown": False, "reviewed": False}


def test_maxn_counts_simultaneous_fish_and_is_unchanged():
    rows = compute_maxn([_rec(0, 1), _rec(6, 1), _rec(6, 2), _rec(12, 3)], species_list={})
    assert (rows[0]["maxn"], rows[0]["frame"], rows[0]["n_tracks"]) == (2, 6, 3)


def test_nearest_frame_rounds_ties_up():
    assert nearest_frame(1.979167, 24.0) == 48
    assert nearest_frame(0.5 / 24, 24.0) == 1
    assert nearest_frame(60.0, 30000 / 1001) == 1798


@pytest.mark.parametrize("fps", FPS_CASES)
@pytest.mark.parametrize("start", [60.0, 61.979167, 75.5, 123.04])
def test_protocol_frames_are_one_nearest_frame_per_sample(fps, start):
    n_frames = int((start + 14 * 60 + 61) * fps) + 1
    points = protocol_sample_frames(start, fps, n_frames)
    assert [p["index"] for p in points] == list(range(15))
    for p in points:
        assert p["time_s"] == pytest.approx(start + 60 * p["index"])
        assert abs(p["frame"] / fps - p["time_s"]) <= 0.5 / fps + 1e-9
    assert len({p["frame"] for p in points}) == 15


@pytest.mark.parametrize("start", [0.0, 59.99, -1.0, float("nan")])
def test_protocol_rejects_start_inside_first_minute(start):
    with pytest.raises(ProtocolError):
        protocol_sample_frames(start, 30.0, 30 * 36 * 60)


def test_protocol_rejects_clip_too_short_for_15_samples():
    with pytest.raises(ProtocolError, match="beyond"):
        protocol_sample_frames(60.0, 30.0, int(30 * (60 + 14 * 60 + 59)))
    protocol_sample_frames(60.0, 30.0, int(30 * (60 + 14 * 60 + 60)))


def test_protocol_rejects_unusable_video():
    with pytest.raises(ProtocolError):
        protocol_sample_frames(60.0, 0.0, 1000)


@pytest.fixture
def fake_video(monkeypatch):
    state = {"fps": 30.0, "n": 0, "kwargs": {}, "seen": []}

    def install(fps, n, **kwargs):
        state.update(fps=fps, n=n, kwargs=kwargs)

    monkeypatch.setattr(detect, "video_info", lambda p: {
        "fps": state["fps"], "frame_count": state["n"], "width": 160, "height": 120,
        "duration_s": round(state["n"] / state["fps"], 3)})
    monkeypatch.setattr(detect, "_open_capture", lambda p: FakeCapture(state["n"], **state["kwargs"]))
    monkeypatch.setattr(detect, "load_detector", lambda: FakeModel(boxes=[[20, 20, 60, 40], [100, 60, 140, 80]]))
    real = detect.prepare_frame

    def spy(frame):
        state["seen"].append(frame_index(frame))
        return real(frame)

    monkeypatch.setattr(detect, "prepare_frame", spy)
    install.state = state
    return install


@pytest.mark.parametrize("fps", FPS_CASES)
def test_protocol_detection_and_counting_use_the_same_single_frames(fake_video, fps):
    n = int(17 * 60 * fps)
    fake_video(fps, n)
    start = 61.979167
    run = detect.run_protocol_samples("unit_clip.mp4", start)
    expected = [p["frame"] for p in protocol_sample_frames(start, fps, n)]
    assert fake_video.state["seen"] == expected
    assert [s["frame"] for s in run["samples"]] == expected
    for s in run["samples"]:
        assert s["status"] == "done" and {r["frame"] for r in s["records"]} == {s["frame"]}
    result = compute_meancount(run)
    assert [r["frame"] for r in result["samples"]] == expected
    assert result["meancount_fish"] == 2.0


def test_protocol_fails_completely_if_one_sample_frame_cannot_be_read(fake_video):
    fps = 30.0
    n = 17 * 60 * 30
    frames = [p["frame"] for p in protocol_sample_frames(60.0, fps, n)]
    fake_video(fps, n, fail_retrieve_at=frames[7])
    with pytest.raises(detect.DetectionError, match=str(frames[7])):
        detect.run_protocol_samples("unit_clip.mp4", 60.0)


def test_protocol_rejects_short_clip_before_running_the_model(fake_video):
    fake_video(30.0, 30 * 10 * 60)
    with pytest.raises(ProtocolError):
        detect.run_protocol_samples("unit_clip.mp4", 60.0)
    assert fake_video.state["seen"] == []


def test_protocol_sample_ids_do_not_link_fish_across_minutes(fake_video):
    fake_video(30.0, 17 * 60 * 30)
    run = detect.run_protocol_samples("unit_clip.mp4", 60.0)
    ids = [r["track_id"] for s in run["samples"] for r in s["records"]]
    assert len(ids) == len(set(ids)) == 30


def _run(counts, status="done", n=None):
    samples = [{"index": i, "time_s": 60.0 + 60 * i, "frame": 1800 * (i + 1), "status": status,
                "records": [_rec(1800 * (i + 1), 100 * i + k) for k in range(c)]}
               for i, c in enumerate(counts)]
    return {"clip_id": "c", "n_samples": len(counts) if n is None else n, "start_s": 60.0,
            "interval_s": 60.0, "samples": samples}


def test_meancount_counts_completed_empty_samples_as_zero():
    result = compute_meancount(_run([0] * 13 + [21, 81]))
    assert result["meancount_fish"] == round(102 / 15, 3)
    assert [r["fish"] for r in result["samples"]][:13] == [0] * 13


def test_meancount_rejects_incomplete_samples():
    run = _run([1] * 15)
    run["samples"][4]["status"] = "skipped"
    with pytest.raises(ProtocolError, match="not completed"):
        compute_meancount(run)


@pytest.mark.parametrize("mutate", [
    lambda r: r["samples"].pop(),
    lambda r: r.update(n_samples=16),
    lambda r: r["samples"].reverse(),
])
def test_meancount_rejects_missing_or_inconsistent_samples(mutate):
    run = _run([1] * 15)
    mutate(run)
    with pytest.raises(ProtocolError):
        compute_meancount(run)


def test_meancount_never_combines_fish_from_different_frames():
    run = _run([1] * 15)
    run["samples"][0]["records"].append(_rec(1806, 999))
    with pytest.raises(ProtocolError, match="other than"):
        compute_meancount(run)


def test_meancount_reports_species_separately_with_full_denominator():
    run = _run([0] * 15)
    run["samples"][3]["records"] = [_rec(run["samples"][3]["frame"], 1, "atlantic_cod"),
                                    _rec(run["samples"][3]["frame"], 2, "unknown")]
    result = compute_meancount(run)
    assert result["meancount_by_species"] == {"atlantic_cod": round(1 / 15, 3), "unknown": round(1 / 15, 3)}
    assert result["n_samples"] == 15
