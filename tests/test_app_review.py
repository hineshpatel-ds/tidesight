"""Review behavior through the actual Streamlit form and pipeline output files."""
from copy import deepcopy
from pathlib import Path
import shutil

import pytest
from streamlit.testing.v1 import AppTest

from tidesight import config, pipeline
from tidesight.contracts import (ContractError, new_record, read_jsonl, read_maxn_csv,
                                 set_track_species, write_segments)

APP = Path(__file__).resolve().parents[1] / "app.py"


@pytest.fixture
def review_app(tmp_path, monkeypatch):
    clips = tmp_path / "clips"
    clips.mkdir()
    outputs = tmp_path / "outputs"
    # Small real video fixture: no weights, cloud services, or model inference.
    import cv2
    import numpy as np
    video = clips / "alpha.mp4"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 25, (80, 60))
    assert writer.isOpened()
    for _ in range(10):
        writer.write(np.zeros((60, 80, 3), dtype=np.uint8))
    writer.release()
    shutil.copyfile(video, clips / "beta.mp4")
    monkeypatch.setattr(config, "CLIPS_DIR", clips)
    monkeypatch.setattr(config, "OUTPUTS_DIR", outputs)
    initial = {}
    for clip_id in ("alpha", "beta"):
        records = [new_record(clip_id, frame, 25, [1, 1, 30, 30], tid, 0.8)
                   for frame in (0, 5) for tid in (1, 2, 3)]
        for tid, species, score in ((1, "atlantic_cod", 0.62), (2, "pollock", 0.71),
                                    (3, "cunner", 0.81)):
            set_track_species(records, tid, species, score,
                              [{"species": species, "score": score}], reviewed=tid == 3)
        pipeline.save_review(clip_id, records)
        write_segments([{"start_s": 0.0, "end_s": 0.4, "activity_score": 1.0}],
                       outputs / clip_id / config.SEGMENTS_FILE)
        initial[clip_id] = deepcopy(records)
    app = AppTest.from_file(str(APP)).run(timeout=10)
    assert not app.exception
    return app, outputs, initial


def species_widgets(app):
    return [s for s in app.selectbox if s.label == "Species"]


def submit(app):
    next(b for b in app.button if b.label == "Save review").click().run(timeout=10)
    assert not app.exception


def saved(outputs, clip="alpha"):
    return read_jsonl(outputs / clip / config.DETECTIONS_FILE)


def test_untouched_save_does_not_write_or_review_tracks(review_app):
    app, outputs, initial = review_app
    paths = [outputs / "alpha" / name for name in (config.DETECTIONS_FILE, config.MAXN_FILE)]
    before = [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths]
    submit(app)
    assert saved(outputs) == initial["alpha"]
    assert [(p.read_bytes(), p.stat().st_mtime_ns) for p in paths] == before
    assert not app.success
    assert any("select" in x.value.lower() or "check" in x.value.lower() for x in app.info)


def test_explicit_unchanged_confirmation_preserves_other_tracks_and_model_evidence(review_app):
    app, outputs, initial = review_app
    app.checkbox[0].check()
    # Editing another selector alone must not count as confirming that track.
    species_widgets(app)[1].select("unknown")
    submit(app)
    expected = deepcopy(initial["alpha"])
    for rec in expected:
        if rec["track_id"] == 1:
            rec["reviewed"] = True
    assert saved(outputs) == expected
    assert app.success
    assert all(not c.value for c in app.checkbox)
    assert species_widgets(app)[1].value == "pollock"


@pytest.mark.parametrize("decision, count", [("pollock", 2), ("unknown", 1)])
def test_correction_applies_all_frames_preserves_evidence_and_recounts(review_app, decision, count):
    app, outputs, initial = review_app
    species_widgets(app)[0].select(decision)
    app.checkbox[0].check()
    submit(app)
    expected = deepcopy(initial["alpha"])
    for rec in expected:
        if rec["track_id"] == 1:
            rec.update(species=decision, unknown=decision == "unknown", reviewed=True)
    assert saved(outputs) == expected
    rows = {r["species_id"]: r for r in read_maxn_csv(outputs / "alpha" / config.MAXN_FILE)}
    assert "atlantic_cod" not in rows
    assert rows[decision]["maxn"] == count
    assert rows[decision]["frame"] == 0
    assert rows[decision]["n_tracks"] == count
    displayed = app.dataframe[1].value.set_index("species_id")
    assert displayed.loc[decision, "maxn"] == count
    assert "atlantic_cod" not in displayed.index
    assert any("original model" in m.value.lower() and "0.62" in m.value for m in app.markdown)
    assert any("human" in c.value.lower() and "confidence" in c.value.lower() for c in app.caption)
    assert app.success
    assert all(not c.value for c in app.checkbox)
    submit(app)
    assert saved(outputs) == expected


def test_previously_reviewed_track_can_be_deliberately_corrected(review_app):
    app, outputs, initial = review_app
    species_widgets(app)[2].select("unknown")
    app.checkbox[2].check()
    submit(app)
    expected = deepcopy(initial["alpha"])
    for rec in expected:
        if rec["track_id"] == 3:
            rec.update(species="unknown", unknown=True)
    assert saved(outputs) == expected


def test_clip_switch_discards_pending_confirmation_and_species(review_app):
    app, outputs, initial = review_app
    app.checkbox[0].check()
    species_widgets(app)[0].select("unknown")
    app.sidebar.selectbox[0].select_index(1).run()
    assert all(not c.value for c in app.checkbox)
    assert species_widgets(app)[0].value == "atlantic_cod"
    app.sidebar.selectbox[0].select_index(0).run()
    assert all(not c.value for c in app.checkbox)
    assert species_widgets(app)[0].value == "atlantic_cod"
    submit(app)
    assert saved(outputs) == initial["alpha"]
    assert saved(outputs, "beta") == initial["beta"]


def test_fresh_analysis_discards_pending_confirmation(review_app, monkeypatch):
    app, outputs, _ = review_app
    original_run = pipeline.run_pipeline
    monkeypatch.setattr(pipeline, "run_pipeline", lambda path: original_run(path, measure_energy=False))
    species_widgets(app)[0].select("unknown")
    app.checkbox[0].check()
    app.sidebar.button[0].click().run(timeout=10)
    assert not app.exception
    assert all(not c.value for c in app.checkbox)
    assert species_widgets(app)[0].value == "pollock"
    assert all(not r["reviewed"] for r in saved(outputs))


def test_save_failure_has_no_success_and_keeps_saved_data(review_app, monkeypatch):
    app, outputs, initial = review_app
    def fail(*args):
        raise ContractError("test review validation failure")
    monkeypatch.setattr(pipeline, "save_review", fail)
    app.checkbox[0].check()
    submit(app)
    assert saved(outputs) == initial["alpha"]
    assert not app.success
    assert app.error
