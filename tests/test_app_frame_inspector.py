"""Saved-frame inspector behavior, geometry and actual Streamlit review refresh."""
import ast
from copy import deepcopy
from pathlib import Path

import cv2
import numpy as np
import pytest

from tidesight import config
from tidesight.contracts import new_record, set_track_species, write_jsonl, read_jsonl
from test_app_review import review_app, species_widgets, submit

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def inspector():
    # Load app-owned pure helpers without executing Streamlit's top-level UI.
    tree = ast.parse(APP.read_text(encoding="utf-8"))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name.startswith("inspector_")]
    namespace = {"cv2": cv2, "config": config}
    from tidesight.enhance import enhance
    namespace["enhance"] = enhance
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(APP), "exec"), namespace)
    return namespace


def records():
    rows = [new_record("alpha", f, 25, box, tid, score)
            for f, box, tid, score in [(15, [40, 30, 75, 55], 9, .9),
                                      (5, [5, 8, 30, 25], 2, .7),
                                      (15, [5, 8, 30, 25], 2, .7)]]
    set_track_species(rows, 2, "pollock", .62, reviewed=True)
    set_track_species(rows, 9, "unknown", .3)
    return rows


def test_sparse_saved_frame_selection_and_box_label_join(inspector, tmp_path):
    path = tmp_path / "detections.jsonl"
    write_jsonl(records(), path)
    rows = read_jsonl(path)
    original = deepcopy(rows)
    groups = inspector["inspector_frames"](rows)
    assert list(groups) == [5, 15]
    assert [r["track_id"] for r in groups[15]] == [2, 9]
    assert inspector["inspector_frame_label"](15, groups[15]) == "Frame 15 (0.600 s)"
    evidence = inspector["inspector_evidence"](groups[15], {"pollock": {"common_name": "Pollock"}})
    assert evidence == [
        {"Track": 2, "Saved species decision": "Pollock", "Review status": "Reviewed",
         "Detector model score": .7, "Original species model score": .62},
        {"Track": 9, "Saved species decision": "Unknown", "Review status": "Not reviewed",
         "Detector model score": .9, "Original species model score": .3}]
    assert rows == original
    assert inspector["inspector_frames"]([]) == {}


@pytest.fixture
def numbered_video(tmp_path):
    video = tmp_path / "numbered.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 25, (80, 60))
    assert writer.isOpened()
    for i in range(20):
        writer.write(np.full((60, 80, 3), i * 10, dtype=np.uint8))
    writer.release()
    return video


@pytest.mark.parametrize("frame", [0, 5, 15, 19])
def test_retrieves_original_frame_by_number_without_seek(inspector, numbered_video, frame):
    image = inspector["inspector_read_frame"](str(numbered_video), frame)
    assert image.shape == (60, 80, 3)
    assert abs(float(image.mean()) - frame * 10) < 3


@pytest.mark.parametrize("frame", [-1, 20, 100])
def test_out_of_range_frame_is_unavailable(inspector, numbered_video, frame):
    with pytest.raises(ValueError):
        inspector["inspector_read_frame"](str(numbered_video), frame)


def test_missing_video_is_unavailable(inspector, tmp_path):
    with pytest.raises(ValueError):
        inspector["inspector_read_frame"](str(tmp_path / "missing.mp4"), 5)


@pytest.mark.parametrize("bad_position", [0, 2, float("nan")])
def test_unverifiable_decoder_position_rejected_and_released(inspector, monkeypatch, bad_position):
    class Capture:
        released = False
        def isOpened(self): return True
        def get(self, field):
            return {cv2.CAP_PROP_FRAME_COUNT: 10, cv2.CAP_PROP_FRAME_WIDTH: 80,
                    cv2.CAP_PROP_FRAME_HEIGHT: 60, cv2.CAP_PROP_POS_FRAMES: bad_position}[field]
        def read(self): return True, np.zeros((60, 80, 3), dtype=np.uint8)
        def release(self): self.released = True
    cap = Capture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda _: cap)
    with pytest.raises(ValueError):
        inspector["inspector_read_frame"]("unsupported", 0)
    assert cap.released


@pytest.mark.parametrize("enhanced", [False, True])
def test_boxes_use_original_coordinates_after_enhancement(inspector, enhanced):
    raw = np.zeros((60, 80, 3), dtype=np.uint8)
    before = raw.copy()
    # Deterministic display transform; raw must remain intact.
    inspector["enhance"] = lambda img: img + 20
    rows = inspector["inspector_frames"](records())[15]
    image = inspector["inspector_overlay"](raw, rows, enhanced)
    assert image.shape == raw.shape
    np.testing.assert_array_equal(raw, before)
    # Vertical sides, away from labels; a transposed or scaled box misses these.
    np.testing.assert_array_equal(image[45, 40], [0, 255, 0])
    np.testing.assert_array_equal(image[18, 5], [0, 255, 0])
    np.testing.assert_array_equal(image[58, 78], [20, 20, 20] if enhanced else [0, 0, 0])


def test_enhancement_shape_change_rejected(inspector):
    inspector["enhance"] = lambda img: img[:20, :20]
    with pytest.raises(ValueError, match="dimensions"):
        inspector["inspector_overlay"](np.zeros((60, 80, 3), dtype=np.uint8), records(), True)


def test_outside_box_rejected_instead_of_clipped(inspector):
    rows = records()
    rows[0]["bbox"][2] = 81
    with pytest.raises(ValueError, match="box"):
        inspector["inspector_overlay"](np.zeros((60, 80, 3), dtype=np.uint8), rows, False)


def evidence_table(app):
    return next(d.value for d in app.dataframe if "Saved species decision" in d.value.columns)


def test_actual_app_selection_and_saved_correction_refresh(review_app):
    app, outputs, _ = review_app
    selector = next((s for s in app.selectbox if s.label == "Saved frame"), None)
    assert selector is not None
    assert selector.options == ["Frame 0 (0.000 s)", "Frame 5 (0.200 s)"]
    selector.select(5).run()
    species_widgets(app)[0].select("unknown")
    app.checkbox[0].check()
    submit(app)
    assert next(s for s in app.selectbox if s.label == "Saved frame").value == 5
    row = evidence_table(app).set_index("Track").loc[1]
    assert row["Saved species decision"] == "Unknown"
    assert row["Review status"] == "Reviewed"
    assert row["Original species model score"] == .62
    assert any("not calibrated" in c.value.lower() for c in app.caption)
    assert len(app.get("download_button")) == 1


def test_empty_results_explained_without_frame_choices(review_app):
    app, outputs, _ = review_app
    write_jsonl([], outputs / "alpha" / config.DETECTIONS_FILE)
    app.run()
    assert not app.exception
    assert not any(s.label == "Saved frame" for s in app.selectbox)
    assert any("no saved detections" in i.value.lower() for i in app.info)
    assert len(app.get("download_button")) == 1


def test_unavailable_frame_retains_evidence_and_review(review_app):
    app, outputs, _ = review_app
    rows = read_jsonl(outputs / "alpha" / config.DETECTIONS_FILE)
    rows = [r for r in rows if r["frame"] == 5]
    for r in rows:
        r["frame"] = 100
        r["time_s"] = 4.0
    write_jsonl(rows, outputs / "alpha" / config.DETECTIONS_FILE)
    app.run()
    assert not app.exception
    assert any("unavailable" in w.value.lower() for w in app.warning)
    assert len(evidence_table(app)) == 3
    assert len(app.checkbox) == 3
    assert len(app.get("download_button")) == 1


def test_right_edge_track_number_stays_visible(inspector, monkeypatch):
    raw = np.zeros((100, 160, 3), dtype=np.uint8)
    row = new_record("edge", 0, 25, [150, 50, 160, 90], 123, .8)
    real_put_text = cv2.putText
    placements = []
    def draw(image, text, origin, font, scale, color, thickness):
        size, baseline = cv2.getTextSize(text, font, scale, thickness)
        placements.append((origin, size, baseline))
        return real_put_text(image, text, origin, font, scale, color, thickness)
    monkeypatch.setattr(cv2, "putText", draw)
    inspector["inspector_overlay"](raw, [row], False)
    assert placements
    for (x, y), (width, height), baseline in placements:
        assert x >= 0 and x + width < 160
        assert y - height >= 0 and y + baseline < 100
