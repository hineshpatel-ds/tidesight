from __future__ import annotations

import cv2
import numpy as np
import pytest

from tidesight import config, detect
from tidesight.contracts import Segment, validate_records

PROPOSED = dict(
    DETECTOR_WEIGHTS=config.MODELS_DIR / "tidesight-fish-v1.pt", DETECT_CONF=0.18, DETECT_IMGSZ=1280,
    DETECT_INPUT_WIDTH=320, DETECT_PREPROCESS=detect.PREPROCESS_ID,
    TRACK_LINK_DIST=1.5, TRACK_MAX_MISSED=2, TRACK_MAX_GAP_S=1.0,
    PROTOCOL_N_SAMPLES=15, PROTOCOL_INTERVAL_S=60.0, PROTOCOL_EDGE_S=60.0,
)
W, H = 160, 120


@pytest.fixture(autouse=True)
def detector_config(monkeypatch):
    for name, value in PROPOSED.items():
        monkeypatch.setattr(config, name, value, raising=False)


def make_frame(i: int) -> np.ndarray:
    img = np.zeros((H, W, 3), np.uint8)
    img[..., 0], img[..., 1], img[..., 2] = i % 256, i // 256, 128
    return img


def frame_index(img: np.ndarray) -> int:
    return int(img[0, 0, 0]) + 256 * int(img[0, 0, 1])


class FakeCapture:

    def __init__(self, n_frames, refuse_seek=False, seek_offset=0, fail_grab_at=None,
                 fail_retrieve_at=None, report_after_read=None):
        self.n, self.pos, self.current = n_frames, 0, None
        self.refuse_seek, self.seek_offset = refuse_seek, seek_offset
        self.fail_grab_at, self.fail_retrieve_at = fail_grab_at, fail_retrieve_at
        self.report_after_read = report_after_read

    def isOpened(self):
        return True

    def set(self, prop, value):
        if prop != cv2.CAP_PROP_POS_FRAMES or self.refuse_seek:
            return False
        self.pos = max(0, int(value) + self.seek_offset)
        return True

    def get(self, prop):
        if prop == cv2.CAP_PROP_POS_FRAMES:
            if self.report_after_read is not None and self.current is not None:
                return float(self.report_after_read)
            return float(self.pos)
        return 0.0

    def grab(self):
        if self.pos >= self.n or self.pos == self.fail_grab_at:
            return False
        self.current, self.pos = self.pos, self.pos + 1
        return True

    def retrieve(self):
        if self.current is None or self.current == self.fail_retrieve_at:
            return False, None
        return True, make_frame(self.current)

    def read(self):
        return (self.retrieve() if self.grab() else (False, None))

    def release(self):
        pass


class _T(list):
    def tolist(self):
        return list(self)


class _Boxes:
    def __init__(self, xyxy, conf):
        self.xyxy, self.conf = _T(xyxy), _T(conf)


class _Res:
    def __init__(self, xyxy, conf):
        self.boxes = _Boxes(xyxy, conf)


class FakeModel:

    def __init__(self, boxes=None):
        self.boxes = boxes

    def predict(self, img, **kwargs):
        h, w = img.shape[:2]
        boxes = self.boxes if self.boxes is not None else [[20, 20, 60, 40], [w - 4, h - 4, w + 20, h + 20]]
        return [_Res(boxes, [0.9] * len(boxes))]


@pytest.fixture
def video(monkeypatch):
    state = {"kwargs": {}, "n": 300, "fps": 30.0, "seen": []}

    def configure(n=300, fps=30.0, **kwargs):
        state.update(n=n, fps=fps, kwargs=kwargs)

    monkeypatch.setattr(detect, "video_info", lambda p: {
        "fps": state["fps"], "frame_count": state["n"], "width": W, "height": H,
        "duration_s": round(state["n"] / state["fps"], 3)})
    monkeypatch.setattr(detect, "_open_capture", lambda p: FakeCapture(state["n"], **state["kwargs"]))
    monkeypatch.setattr(detect, "load_detector", lambda: FakeModel())
    real_prepare = detect.prepare_frame

    def spy(frame):
        state["seen"].append(frame_index(frame))
        return real_prepare(frame)

    monkeypatch.setattr(detect, "prepare_frame", spy)
    configure.state = state
    return configure


def seg(a, b):
    return [Segment(start_s=a, end_s=b, activity_score=1.0)]


def test_records_follow_contract_and_original_resolution(video):
    records = detect.run_detection("unit_clip.mp4", seg(0, 2))
    assert records and validate_records(records, fps=30.0) == []
    for r in records:
        x1, y1, x2, y2 = r["bbox"]
        assert 0 <= x1 < x2 <= W - 1 and 0 <= y1 < y2 <= H - 1
    assert {r["bbox"][0] for r in records if r["bbox"][0] < 40} == {10}


def test_each_record_is_labelled_with_the_frame_actually_decoded(video):
    records = detect.run_detection("unit_clip.mp4", seg(0, 9.9))
    assert video.state["seen"] == sorted({r["frame"] for r in records})


@pytest.mark.parametrize("kwargs", [dict(fail_grab_at=12), dict(fail_retrieve_at=12)])
def test_failed_read_of_requested_frame_raises_with_frame_number(video, kwargs):
    video(**kwargs)
    with pytest.raises(detect.DetectionError, match="12"):
        detect.run_detection("unit_clip.mp4", seg(0, 4))


def test_failed_read_on_the_way_to_a_frame_raises(video):
    video(fail_grab_at=7)
    with pytest.raises(detect.DetectionError):
        detect.run_detection("unit_clip.mp4", seg(0, 1))


def test_video_shorter_than_reported_raises(video, monkeypatch):
    monkeypatch.setattr(detect, "_open_capture", lambda p: FakeCapture(100))
    with pytest.raises(detect.DetectionError):
        detect.run_detection("unit_clip.mp4", seg(0, 9.9))


def test_requested_frame_outside_video_raises(video):
    reader = detect.FrameReader("unit_clip.mp4", 300, 30.0)
    with pytest.raises(detect.DetectionError, match="outside"):
        reader.read(300)


def test_seek_refused_falls_back_to_sequential_and_labels_correctly(video):
    video(refuse_seek=True)
    records = detect.run_detection("unit_clip.mp4", seg(8, 9))
    assert video.state["seen"] == sorted({r["frame"] for r in records})
    assert min(video.state["seen"]) == 240


@pytest.mark.parametrize("offset", [-3, 2])
def test_inexact_seek_is_detected_and_never_mislabels(video, offset):
    video(seek_offset=offset)
    records = detect.run_detection("unit_clip.mp4", seg(8, 9))
    assert video.state["seen"] == sorted({r["frame"] for r in records})


def test_capture_reporting_wrong_position_after_read_is_not_trusted(video):
    video(report_after_read=5)
    records = detect.run_detection("unit_clip.mp4", seg(8, 9))
    assert video.state["seen"] == sorted({r["frame"] for r in records})


def test_backward_and_sparse_reads_return_exact_frames(monkeypatch):
    monkeypatch.setattr(detect, "_open_capture", lambda p: FakeCapture(3000))
    reader = detect.FrameReader("x", 3000, 30.0)
    for target in (2000, 5, 1799, 1800, 0, 2999):
        assert frame_index(reader.read(target)) == target


def test_tracker_links_consecutive_sampled_frames():
    tr = detect.CentroidTracker()
    a = tr.update([[0, 0, 60, 20]], 0.0)
    b = tr.update([[40, 0, 100, 20]], 0.2)
    assert a == b


def test_tracker_does_not_link_across_unseen_gap_and_keeps_ids_unique():
    tr = detect.CentroidTracker()
    a = tr.update([[0, 0, 60, 20]], 0.0)
    b = tr.update([[0, 0, 60, 20]], 10.0)
    c = tr.update([[0, 0, 60, 20]], 10.2)
    assert a != b and b == c and len({*a, *b}) == 2


def test_tracker_expires_after_missed_frames():
    tr = detect.CentroidTracker()
    a = tr.update([[0, 0, 60, 20]], 0.0)
    for k in range(1, 4):
        tr.update([], 0.2 * k)
    assert tr.update([[0, 0, 60, 20]], 0.8) != a


def test_sparse_segments_get_new_ids_for_identical_positions(video):
    records = detect.run_detection("unit_clip.mp4", seg(0, 0.1) + seg(9.0, 9.1))
    by_frame = {}
    for r in records:
        by_frame.setdefault(r["frame"], set()).add(r["track_id"])
    assert len(by_frame) == 2
    first, second = by_frame.values()
    assert not first & second


def test_each_call_starts_a_fresh_tracker(video):
    one = detect.run_detection("unit_clip.mp4", seg(0, 0.1))
    two = detect.run_detection("unit_clip.mp4", seg(0, 0.1))
    assert [r["track_id"] for r in one] == [r["track_id"] for r in two]


def test_settings_missing_from_config_raise_before_any_work(video, monkeypatch):
    monkeypatch.delattr(config, "DETECT_INPUT_WIDTH")
    with pytest.raises(detect.DetectionError, match="DETECT_INPUT_WIDTH"):
        detect.run_detection("unit_clip.mp4", seg(0, 1))
    assert video.state["seen"] == []


def test_settings_wrong_preprocessing_id_raises(video, monkeypatch):
    monkeypatch.setattr(config, "DETECT_PREPROCESS", "something_else")
    with pytest.raises(detect.DetectionError, match="preprocess|DETECT_PREPROCESS"):
        detect.run_detection("unit_clip.mp4", seg(0, 1))


def test_settings_reject_weights_not_trained_with_this_preprocessing(video, monkeypatch):
    monkeypatch.setattr(config, "DETECTOR_WEIGHTS", config.MODELS_DIR / "cfd-yolov12x-1.00.pt")
    with pytest.raises(detect.DetectionError, match="cfd-yolov12x"):
        detect.run_detection("unit_clip.mp4", seg(0, 1))
    assert video.state["seen"] == []


def test_settings_confidence_and_size_come_from_config(video, monkeypatch):
    seen = {}

    class Spy(FakeModel):
        def predict(self, img, **kw):
            seen.update(kw, width=img.shape[1])
            return super().predict(img, **kw)

    monkeypatch.setattr(detect, "load_detector", lambda: Spy())
    monkeypatch.setattr(config, "DETECT_CONF", 0.33)
    monkeypatch.setattr(config, "DETECT_IMGSZ", 640)
    detect.run_detection("unit_clip.mp4", seg(0, 0.1))
    assert (seen["conf"], seen["imgsz"], seen["width"]) == (0.33, 640, 320)


def test_missing_weights_fail_clearly(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "DETECTOR_WEIGHTS", tmp_path / "nope.pt")
    pytest.importorskip("ultralytics")
    with pytest.raises(detect.DetectionError, match="nope.pt"):
        detect.load_detector()


def test_no_segments_no_work(video):
    assert detect.run_detection("unit_clip.mp4", []) == []
    assert video.state["seen"] == []


def test_pipeline_keeps_previous_complete_results_when_a_frame_fails(tmp_path, monkeypatch):
    from tidesight import pipeline

    clips, outputs = tmp_path / "clips", tmp_path / "outputs"
    clips.mkdir()
    outputs.mkdir()
    monkeypatch.setattr(config, "CLIPS_DIR", clips)
    monkeypatch.setattr(config, "OUTPUTS_DIR", outputs)
    path = clips / "pipe_clip.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (W, H))
    for i in range(60):
        writer.write(make_frame(i))
    writer.release()
    monkeypatch.setattr(detect, "load_detector", lambda: FakeModel())

    pipeline.run_pipeline(path, measure_energy=False)
    published = {p.name: p.read_bytes() for p in (outputs / "pipe_clip").iterdir() if p.is_file()}
    assert config.DETECTIONS_FILE in published

    monkeypatch.setattr(detect, "_open_capture", lambda p: FakeCapture(60, fail_grab_at=12))
    with pytest.raises(detect.DetectionError):
        pipeline.run_pipeline(path, measure_energy=False)
    after = {p.name: p.read_bytes() for p in (outputs / "pipe_clip").iterdir() if p.is_file()}
    assert after == published
