from __future__ import annotations

import cv2
import numpy as np
import pytest

from tidesight import config, detect
from tidesight.contracts import Segment, detect_step, frames_to_process, validate_records, video_info
from tidesight.count import compute_maxn, compute_meancount, sample_times


# ---------------------------------------------------------------- fakes
class _Tensor(list):
    def tolist(self):
        return list(self)

    def int(self):
        return _Tensor(int(v) for v in self)


class _Boxes:
    def __init__(self, xyxy, conf, ids):
        self.xyxy, self.conf = _Tensor(xyxy), _Tensor(conf)
        self.id = None if ids is None else _Tensor(ids)


class _Result:
    def __init__(self, boxes):
        self.boxes = boxes


class FakeYOLO:
    """Returns one tracked fish per call, plus a box that falls outside the image."""

    instances = 0

    def __init__(self):
        FakeYOLO.instances += 1
        self.calls = 0

    def predict(self, img, **kwargs):
        self.calls += 1
        h, w = img.shape[:2]
        # one fish drifting right a little each frame + one box hanging outside the image
        x = 10.4 + 3 * self.calls
        return [_Result(_Boxes([[x, 20.6, x + 40, 60.9], [w - 5, h - 5, w + 30, h + 30]],
                               [0.91, 0.4], None))]

    def track(self, img, **kwargs):
        self.calls += 1
        h, w = img.shape[:2]
        if self.calls == 1:   # ByteTrack often has no confirmed track yet on the first frame
            return [_Result(_Boxes([], [], None))]
        return [_Result(_Boxes([[10.4, 20.6, 50.2, 60.9], [w - 5, h - 5, w + 30, h + 30]],
                               [0.91, 0.4], [1, 2]))]


@pytest.fixture
def clip(tmp_path):
    path = tmp_path / "unit_clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (160, 120))
    for i in range(90):   # 3 s at 30 fps
        writer.write(np.full((120, 160, 3), i % 255, np.uint8))
    writer.release()
    return path


@pytest.fixture(autouse=True)
def fake_detector(monkeypatch):
    FakeYOLO.instances = 0
    monkeypatch.setattr(detect, "load_detector", lambda: FakeYOLO())


# ---------------------------------------------------------------- detect
def test_records_follow_contract(clip):
    records = detect.run_detection(clip, None)
    info = video_info(clip)
    assert records, "fake detector should produce records"
    assert validate_records(records, fps=info["fps"]) == []


def test_only_wanted_original_frames(clip):
    info = video_info(clip)
    segs = [Segment(start_s=1.0, end_s=2.0, activity_score=1.0)]
    wanted = set(frames_to_process(segs, info["fps"], info["frame_count"], detect_step(info["fps"])))
    records = detect.run_detection(clip, segs)
    assert {r["frame"] for r in records} <= wanted
    assert min(r["frame"] for r in records) >= 30      # original numbering, not 0, 1, 2...


def test_boxes_clipped_inside_image(clip):
    info = video_info(clip)
    for r in detect.run_detection(clip, None):
        x1, y1, x2, y2 = r["bbox"]
        assert 0 <= x1 < x2 <= info["width"] - 1 and 0 <= y1 < y2 <= info["height"] - 1


def test_fresh_model_per_clip(clip):
    detect.run_detection(clip, None)
    detect.run_detection(clip, None)
    assert FakeYOLO.instances == 2      # tracker state cannot leak between videos


def test_same_fish_keeps_its_track_id(clip):
    records = detect.run_detection(clip, None)
    near_origin = {r["track_id"] for r in records if r["bbox"][0] < 40}
    assert len(near_origin) == 1          # the drifting fish is linked across frames, not renumbered


def test_centroid_tracker_links_and_splits():
    tr = detect.CentroidTracker()
    a = tr.update([[0, 0, 60, 20], [500, 500, 560, 520]])
    b = tr.update([[40, 0, 100, 20], [900, 900, 960, 920]])   # first moved 40 px (< 1.5 x 60), second jumped far
    assert b[0] == a[0] and b[1] not in a


def test_no_segments_no_work(clip):
    assert detect.run_detection(clip, []) == []


# ---------------------------------------------------------------- count
def _rec(frame, track, species="fish", fps=30.0):
    return {"clip_id": "c", "frame": frame, "time_s": round(frame / fps, 3), "bbox": [0, 0, 1, 1],
            "track_id": track, "det_conf": 0.9, "species": species, "species_conf": 0.0,
            "top3": [], "unknown": False, "reviewed": False}


def test_maxn_counts_simultaneous_fish():
    recs = [_rec(0, 1), _rec(6, 1), _rec(6, 2), _rec(12, 3)]
    row = compute_maxn(recs)[0]
    assert (row["maxn"], row["frame"], row["n_tracks"]) == (2, 6, 3)


def test_sample_times_match_protocol():
    times = sample_times(60.0)
    assert len(times) == 15 and times[0] == 60.0 and times[-1] == 900.0


def test_meancount_window_and_zeros():
    # sample at 10 s: track 1 exactly at 10 s, track 2 at 10.8 s (inside +/-1 s), track 3 at 12 s (outside)
    recs = [_rec(300, 1), _rec(324, 2), _rec(360, 3)]
    out = compute_meancount(recs, [10.0, 20.0], window_s=1.0)
    assert [s["total"] for s in out["samples"]] == [2, 0]
    assert out["meancount_total"] == 1.0
    assert out["meancount_by_species"] == {"fish": 1.0}


def test_meancount_same_track_counted_once():
    recs = [_rec(294, 1), _rec(300, 1), _rec(306, 1)]
    assert compute_meancount(recs, [10.0])["samples"][0]["total"] == 1
