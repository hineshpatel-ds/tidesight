import copy
import json

import pytest

from tidesight import config, pipeline
from tidesight.contracts import new_record, read_jsonl
from tidesight.contracts import validate_protocol, read_protocol_outputs


@pytest.fixture
def protocol(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path)
    info = dict(fps=1.0, frame_count=1020, width=100, height=80, duration_s=1020.0)
    monkeypatch.setattr(pipeline, "video_info", lambda path: info)
    samples = [dict(index=index, time_s=float(60 + index * 60), frame=60 + index * 60,
                    status="done", records=[]) for index in range(15)]
    samples[0]["records"] = [new_record("survey", 60, 1, [1, 2, 20, 30], 1, .9)]
    run = dict(clip_id="survey", fps=1.0, frame_count=1020, start_s=60.0,
               interval_s=60.0, n_samples=15, model="test", preprocess="test", conf=.18,
               samples=samples)
    monkeypatch.setattr(pipeline, "run_protocol_samples", lambda *args: copy.deepcopy(run), raising=False)
    return tmp_path, run


def test_persist_review_recount(protocol):
    root, run = protocol
    result = pipeline.run_pipeline("survey.mp4", False, protocol_start_s=60)
    assert result["counts"]["meancount_fish"] == .067
    records = read_jsonl(root / "survey" / config.DETECTIONS_FILE)
    assert records[0]["species"] == "unknown"
    records[0].update(reviewed=True)
    pipeline.save_review("survey", records)
    pipeline.recount("survey")
    saved = json.loads((root / "survey" / "protocol.json").read_text())
    assert saved["samples"][0]["records"][0]["reviewed"] is True
    assert len(saved["samples"]) == 15
    assert (root / "survey" / "sample_counts.csv").read_text().count("\n") == 16


@pytest.mark.parametrize("damage", ["missing", "failed", "bounds", "source"])
def test_bad_run_preserves_previous(protocol, damage):
    root, run = protocol
    pipeline.run_pipeline("survey.mp4", False, protocol_start_s=60)
    before = {path.name: path.read_bytes() for path in (root / "survey").iterdir()}
    if damage == "missing":
        run["samples"].pop()
    elif damage == "failed":
        run["samples"][1]["status"] = "failed"
    elif damage == "source":
        run["fps"] = 2
    else:
        run["samples"][1]["frame"] = 2000
    with pytest.raises(ValueError):
        pipeline.run_pipeline("survey.mp4", False, protocol_start_s=60)
    assert before == {path.name: path.read_bytes() for path in (root / "survey").iterdir()}


def test_persisted_manifest_rejects_tampering(protocol):
    root, _ = protocol
    pipeline.run_pipeline("survey.mp4", False, protocol_start_s=60)
    run = json.loads((root / "survey" / "protocol.json").read_text())
    records = read_jsonl(root / "survey" / config.DETECTIONS_FILE)
    for key, value in (("status", "failed"), ("frame", 2000), ("time_s", 61)):
        altered = copy.deepcopy(run)
        altered["samples"][0][key] = value
        with pytest.raises(ValueError):
            validate_protocol(altered, records)


def test_review_correction_changes_species_export(protocol):
    root, _ = protocol
    pipeline.run_pipeline("survey.mp4", False, protocol_start_s=60)
    records = read_jsonl(root / "survey" / config.DETECTIONS_FILE)
    records[0].update(species="pollock", unknown=False, reviewed=True)
    pipeline.save_review("survey", records)
    counts = json.loads((root / "survey" / "counts.json").read_text())
    assert counts["meancount_by_species"] == {"pollock": .067}
    assert 'pollock' in (root / "survey" / "sample_counts.csv").read_text()


def test_export_tamper_and_review_failure(protocol, monkeypatch):
    root, _ = protocol
    pipeline.run_pipeline("survey.mp4", False, protocol_start_s=60)
    directory = root / "survey"
    records = read_jsonl(directory / config.DETECTIONS_FILE)
    before = {path.name: path.read_bytes() for path in directory.iterdir()}

    def fail(*args):
        raise OSError("disk full")

    monkeypatch.setattr(pipeline, "write_maxn_csv", fail)
    with pytest.raises(OSError):
        pipeline.save_review("survey", records)
    assert before == {path.name: path.read_bytes() for path in directory.iterdir()}
    (directory / "sample_counts.csv").write_text("wrong export")
    with pytest.raises(ValueError, match="CSV"):
        read_protocol_outputs(directory, records)


def test_protocol_screen_includes_empty_sample(protocol, monkeypatch, tmp_path):
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    root, _ = protocol
    clips = tmp_path / "clips"
    clips.mkdir()
    (clips / "survey.mp4").touch()
    monkeypatch.setattr(config, "CLIPS_DIR", clips)
    pipeline.run_pipeline("survey.mp4", False, protocol_start_s=60)
    import tidesight.contracts as contracts
    monkeypatch.setattr(contracts, "video_info", pipeline.video_info)
    app = AppTest.from_file(str(Path(__file__).resolve().parents[1] / "app.py")).run(timeout=15)
    assert not app.exception
    selector = next(item for item in app.selectbox if item.label == "Saved frame")
    assert len(selector.options) == 15
    selector.select(120).run(timeout=15)
    assert not app.exception
    assert any(item.label == "Mean fish per sampled frame" for item in app.metric)
