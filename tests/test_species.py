"""Species tests. Run with:  python -m pytest -q tests/test_species.py   (no model download needed)"""
import cv2
import numpy as np
import pytest

import tidesight.species as species
from tidesight import config
from tidesight.contracts import load_species_list, new_record, validate_records, video_info

W, H, FPS, N = 320, 240, 10, 20
RED_BOX, BLUE_BOX = [20, 20, 80, 80], [200, 120, 260, 180]


@pytest.fixture
def clip(tmp_path):
    """A red square on the left and a blue square on the right, in every frame."""
    path = tmp_path / "squares.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for _ in range(N):
        frame = np.full((H, W, 3), 128, np.uint8)
        cv2.rectangle(frame, tuple(RED_BOX[:2]), tuple(RED_BOX[2:]), (0, 0, 255), -1)     # BGR red
        cv2.rectangle(frame, tuple(BLUE_BOX[:2]), tuple(BLUE_BOX[2:]), (255, 0, 0), -1)   # BGR blue
        writer.write(frame)
    writer.release()
    return path


def records_for(clip_path):
    fps = video_info(clip_path)["fps"]
    recs = []
    for frame in range(0, N, 2):
        recs.append(new_record("squares", frame, fps, RED_BOX, track_id=1, det_conf=0.9))
        recs.append(new_record("squares", frame, fps, BLUE_BOX, track_id=2, det_conf=0.8))
    return recs


class FakeModel:
    """Crops embed by colour (RGB): red -> first species, blue -> second, grey -> nothing in particular.
    Text is ignored; classes are one-hot so each species owns one axis."""
    logit_scale = 100.0

    def __init__(self, n_species):
        self.n = n_species

    def embed_texts(self, texts):
        return np.ones((len(texts), self.n), np.float32) / np.sqrt(self.n)

    def embed_images(self, images):
        out = np.zeros((len(images), self.n), np.float32)
        for i, im in enumerate(images):
            r, g, b = im.reshape(-1, 3).mean(0)
            out[i, 0] = r / 255
            out[i, 1] = b / 255
            out[i, 2:] = 0.3                     # a little of everything else
        return out / np.linalg.norm(out, axis=1, keepdims=True)


def use_fake_model(monkeypatch, n_species):
    monkeypatch.setattr(species, "_get_model", lambda: FakeModel(n_species))
    monkeypatch.setattr(species, "species_classes",
                        lambda model, sl: (list(sl), np.eye(len(sl), dtype=np.float32)))


def test_default_marks_every_track_unknown_for_the_reviewer(clip, monkeypatch):
    monkeypatch.setattr(species, "_get_model", lambda: pytest.fail("model must not load when it is off"))
    recs = species.identify_species(clip, records_for(clip))
    assert all(r["species"] == config.UNKNOWN_LABEL and r["unknown"] and r["species_conf"] == 0.0
               and r["top3"] == [] and not r["reviewed"] for r in recs)
    assert validate_records(recs, load_species_list(), video_info(clip)["fps"]) == []


def test_zero_shot_names_one_species_per_track_from_original_frame_crops(clip, monkeypatch):
    sl = load_species_list()
    use_fake_model(monkeypatch, len(sl))
    monkeypatch.setattr(species, "_ZERO_SHOT", True)
    recs = species.identify_species(clip, records_for(clip))
    first, second = list(sl)[:2]
    by_track = {t: {(r["species"], r["species_conf"], tuple(s["species"] for s in r["top3"]))
                    for r in recs if r["track_id"] == t} for t in (1, 2)}
    assert len(by_track[1]) == 1 and len(by_track[2]) == 1        # every row of a track agrees
    (s1, c1, top1), = by_track[1]
    (s2, c2, top2), = by_track[2]
    assert (s1, top1[0]) == (first, first) and c1 >= config.UNKNOWN_THRESHOLD
    assert (s2, top2[0]) == (second, second) and c2 >= config.UNKNOWN_THRESHOLD
    assert validate_records(recs, sl, video_info(clip)["fps"]) == []


def test_zero_shot_low_score_becomes_unknown_but_keeps_suggestions(clip, monkeypatch):
    sl = load_species_list()
    use_fake_model(monkeypatch, len(sl))
    monkeypatch.setattr(species, "_ZERO_SHOT", True)
    monkeypatch.setattr(config, "UNKNOWN_THRESHOLD", 1.01)          # nothing can reach it
    recs = species.identify_species(clip, records_for(clip))
    assert all(r["species"] == config.UNKNOWN_LABEL and r["unknown"] for r in recs)
    assert all(len(r["top3"]) == 3 and r["top3"][0]["score"] >= r["top3"][1]["score"] for r in recs)


def test_no_records_needs_no_model(clip, monkeypatch):
    monkeypatch.setattr(species, "_ZERO_SHOT", True)
    monkeypatch.setattr(species, "_get_model", lambda: pytest.fail("no fish, no model"))
    assert species.identify_species(clip, []) == []
