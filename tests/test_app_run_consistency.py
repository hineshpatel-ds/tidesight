"""Deliberate failures at analysis, serialization, energy and publication boundaries."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import sys

import pytest

from tidesight import config, pipeline
from tidesight.contracts import (ContractError, new_record, read_jsonl, read_maxn_csv,
                                 read_segments, set_track_species)
from test_app_review import review_app  # shared actual-app/video fixture


def snapshot(directory):
    return {p.name: p.read_bytes() for p in directory.iterdir()} if directory.exists() else None


@pytest.fixture
def analysis(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "OUTPUTS_DIR", tmp_path / "outputs")
    monkeypatch.setattr(pipeline, "video_info", lambda _: {"fps": 25, "duration_s": 10})
    state = {"segments": [{"start_s": 0., "end_s": 2., "activity_score": 1.}],
             "records": [new_record("alpha", 0, 25, [1, 1, 20, 20], 1, .8)]}
    set_track_species(state["records"], 1, "pollock", .8, reviewed=True)
    monkeypatch.setattr(pipeline, "run_triage", lambda _: deepcopy(state["segments"]))
    monkeypatch.setattr(pipeline, "run_detection", lambda *_: deepcopy(state["records"]))
    monkeypatch.setattr(pipeline, "identify_species", lambda _, records: records)
    trackers = []

    class Tracker:
        def __init__(self, **kwargs):
            self.path = Path(kwargs["output_dir"]) / kwargs["output_file"]
            self.stops = 0
            self.final_emissions_data = SimpleNamespace(energy_consumed=.002)
            trackers.append(self)

        def start(self):
            if state.get("energy_failure") == "start":
                self.path.write_text("partial start")
                raise RuntimeError("energy start fault")

        def stop(self):
            self.stops += 1
            self.path.write_text("energy_consumed,emissions\n0.002,0.001\n")
            if state.get("energy_failure") == "stop":
                raise RuntimeError("energy stop fault")
            return .001

    monkeypatch.setitem(sys.modules, "codecarbon", SimpleNamespace(OfflineEmissionsTracker=Tracker))
    out = config.OUTPUTS_DIR / "alpha"

    def run(energy=True):
        return pipeline.run_pipeline("alpha.mp4", measure_energy=energy)

    def change():
        state["segments"][0]["end_s"] = 4.
        set_track_species(state["records"], 1, "atlantic_cod", .9)
        state["records"].append(new_record("alpha", 0, 25, [21, 1, 40, 20], 2, .7))
        set_track_species(state["records"], 2, "atlantic_cod", .9)

    return SimpleNamespace(run=run, change=change, state=state, out=out, trackers=trackers)


@pytest.mark.parametrize("prior", [False, True])
@pytest.mark.parametrize("boundary", ["run_triage", "run_detection", "identify_species",
                                     "compute_maxn", "write_segments", "write_jsonl", "write_maxn_csv",
                                     "validate_segments", "validate_records"])
def test_failed_analysis_preserves_last_complete_results(analysis, monkeypatch, prior, boundary):
    if prior:
        analysis.run()
    before = snapshot(analysis.out)
    analysis.change()

    def fail(*args):
        if boundary.startswith("write_"):
            Path(args[-1]).write_text("partial serialized output")
        raise RuntimeError(f"injected {boundary}")

    monkeypatch.setattr(pipeline, boundary, fail)
    with pytest.raises(RuntimeError, match="injected"):
        analysis.run()
    assert snapshot(analysis.out) == before
    assert analysis.trackers[-1].stops == 1
    assert list(config.OUTPUTS_DIR.iterdir()) == ([analysis.out] if prior else [])


@pytest.mark.parametrize("prior", [False, True])
def test_invalid_model_data_does_not_publish(analysis, prior):
    if prior:
        analysis.run()
    before = snapshot(analysis.out)
    analysis.state["segments"][0]["end_s"] = 100.
    with pytest.raises(ContractError):
        analysis.run()
    assert snapshot(analysis.out) == before


@pytest.mark.parametrize("prior,boundary", [(False, "publish"), (True, "backup"), (True, "publish")])
def test_rename_failure_restores_previous_directory(analysis, monkeypatch, prior, boundary):
    if prior:
        analysis.run()
    before = snapshot(analysis.out)
    analysis.change()
    rename = Path.rename
    failed = False

    def fail_once(src, dst):
        nonlocal failed
        target = Path(dst)
        if not failed and ((boundary == "backup" and src == analysis.out) or
                           (boundary == "publish" and target == analysis.out)):
            failed = True
            raise PermissionError("injected locked directory")
        return rename(src, dst)

    monkeypatch.setattr(Path, "rename", fail_once)
    with pytest.raises(PermissionError, match="locked directory"):
        analysis.run()
    assert failed
    assert snapshot(analysis.out) == before
    assert list(config.OUTPUTS_DIR.iterdir()) == ([analysis.out] if prior else [])


def test_failed_rollback_retains_backup_and_reports_its_location(analysis, monkeypatch):
    analysis.run()
    before = snapshot(analysis.out)
    analysis.change()
    rename = Path.rename
    backups = []

    def fail_publish_and_recovery(src, dst):
        if src == analysis.out:
            backups.append(Path(dst))
            return rename(src, dst)
        if Path(dst) == analysis.out:
            raise PermissionError("injected publication/recovery fault")
        return rename(src, dst)

    monkeypatch.setattr(Path, "rename", fail_publish_and_recovery)
    with pytest.raises(RuntimeError) as caught:
        analysis.run()
    assert len(backups) == 1
    assert str(backups[0]) in str(caught.value)
    assert "recover" in str(caught.value).lower()
    assert snapshot(backups[0]) == before
    assert not analysis.out.exists()


def test_post_commit_cleanup_failure_still_returns_published_success(analysis, monkeypatch, caplog):
    analysis.run()
    analysis.change()
    import shutil
    remove = shutil.rmtree

    def fail_backup_cleanup(path, *args, **kwargs):
        if "backup" in Path(path).name:
            raise PermissionError("backup cleanup fault")
        return remove(path, *args, **kwargs)

    monkeypatch.setattr(shutil, "rmtree", fail_backup_cleanup)
    result = analysis.run()
    assert Path(result["output_dir"]) == analysis.out
    assert read_jsonl(analysis.out / config.DETECTIONS_FILE) == analysis.state["records"]
    assert "cleanup" in caplog.text.lower()


@pytest.mark.parametrize("failure", ["start", "stop"])
def test_optional_energy_failure_never_publishes_partial_or_old_energy(analysis, failure, caplog):
    analysis.run()
    analysis.change()
    analysis.state["energy_failure"] = failure
    result = analysis.run()
    assert result["energy_kwh"] is None and result["emissions_kg"] is None
    assert not (analysis.out / config.ENERGY_FILE).exists()
    assert "energy" in caplog.text.lower()
    assert analysis.trackers[-1].stops == 1


def test_energy_cleanup_failure_aborts_publication(analysis, monkeypatch):
    analysis.run()
    before = snapshot(analysis.out)
    analysis.state["energy_failure"] = "stop"
    unlink = Path.unlink

    def fail_energy_cleanup(path, *args, **kwargs):
        if path.name == config.ENERGY_FILE:
            raise PermissionError("energy cleanup fault")
        return unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_energy_cleanup)
    with pytest.raises(PermissionError, match="energy cleanup"):
        analysis.run()
    assert snapshot(analysis.out) == before


def test_failed_run_cleanup_does_not_mask_original_error(analysis, monkeypatch, caplog):
    analysis.run()
    before = snapshot(analysis.out)
    import shutil

    def fail_cleanup(*_):
        raise PermissionError("injected staging cleanup fault")

    def fail_detection(*_):
        raise ValueError("original detection failure")

    monkeypatch.setattr(shutil, "rmtree", fail_cleanup)
    monkeypatch.setattr(pipeline, "run_detection", fail_detection)
    with pytest.raises(ValueError, match="original detection failure"):
        analysis.run()
    assert snapshot(analysis.out) == before
    assert "retained" in caplog.text
    assert analysis.trackers[-1].stops == 1


@pytest.mark.parametrize("empty", [False, True])
def test_successful_retry_publishes_consistent_files_and_public_path(analysis, monkeypatch, empty):
    first = analysis.run()
    assert first["energy_kwh"] == .002 and first["emissions_kg"] == .001
    assert Path(first["output_dir"]) == analysis.out
    assert (analysis.out / config.ENERGY_FILE).read_text() == "energy_consumed,emissions\n0.002,0.001\n"
    before = snapshot(analysis.out)
    analysis.change()
    with monkeypatch.context() as patch:
        def fail(*_):
            raise RuntimeError("injected failure")
        patch.setattr(pipeline, "run_detection", fail)
        with pytest.raises(RuntimeError):
            analysis.run()
    assert snapshot(analysis.out) == before
    if empty:
        analysis.state["records"] = []
    result = analysis.run(energy=False)
    assert Path(result["output_dir"]) == analysis.out
    assert analysis.out.is_dir()
    assert read_segments(analysis.out / config.SEGMENTS_FILE) == result["segments"]
    assert read_jsonl(analysis.out / config.DETECTIONS_FILE) == analysis.state["records"]
    assert read_maxn_csv(analysis.out / config.MAXN_FILE) == result["maxn"]
    assert result["n_records"] == len(analysis.state["records"])
    assert not (analysis.out / config.ENERGY_FILE).exists()
    assert list(config.OUTPUTS_DIR.iterdir()) == [analysis.out]


@pytest.mark.parametrize("prior", [False, True])
def test_app_failed_analysis_is_clear_persistent_and_retryable(review_app, monkeypatch, prior):
    app, outputs, _ = review_app
    out = outputs / "alpha"
    if not prior:
        import shutil
        shutil.rmtree(out)
        app.run()
    before = snapshot(out)
    displayed_before = [table.value.copy() for table in app.dataframe]
    app.session_state["review_saved"] = "alpha"
    real_run = pipeline.run_pipeline

    def fail(*_):
        raise RuntimeError("injected analysis failure")

    monkeypatch.setattr(pipeline, "run_pipeline", fail)
    app.sidebar.button[0].click().run(timeout=10)
    assert not app.exception
    assert any("Analysis failed" in e.value and "injected" in e.value for e in app.error)
    assert not app.success
    assert snapshot(out) == before
    if prior:
        assert any("last complete" in x.value.lower() for x in app.warning)
        assert len(app.dataframe) == len(displayed_before)
        for actual, expected in zip(app.dataframe, displayed_before):
            assert actual.value.equals(expected)
    else:
        assert not app.dataframe
        assert any("no complete" in x.value.lower() for x in app.info)
    app.run()
    assert app.error and not app.success
    app.sidebar.selectbox[0].select_index(1).run()
    assert not app.error
    app.sidebar.selectbox[0].select_index(0).run()
    assert app.error
    monkeypatch.setattr(pipeline, "run_pipeline", lambda path: real_run(path, measure_energy=False))
    app.sidebar.button[0].click().run(timeout=10)
    assert not app.exception and not app.error
    assert any("Done:" in x.value for x in app.success)
    assert all(not c.value for c in app.checkbox)
    app.run()
    assert not app.error


@pytest.mark.parametrize("missing", [config.SEGMENTS_FILE, config.MAXN_FILE, config.DETECTIONS_FILE])
def test_app_does_not_display_incomplete_results(review_app, missing):
    app, outputs, _ = review_app
    (outputs / "alpha" / missing).unlink()
    app.run()
    assert not app.exception
    assert not app.dataframe and not app.checkbox
    assert any("no complete" in x.value.lower() for x in app.info)
