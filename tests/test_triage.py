"""Triage tests (Video + classical CV). Run with:  python -m pytest -q tests/test_triage.py"""
import cv2
import numpy as np
import pytest

from tidesight import config
from tidesight.contracts import validate_segments, video_info
from tidesight.triage import drop_blips, run_triage, segments_from_activity

STEP = 0.2  # 5 samples per second


def samples(duration_s, busy_ranges):
    """Sample times every 0.2 s and busy flags that are True inside any (start, end) range."""
    times = [round(i * STEP, 3) for i in range(int(duration_s / STEP) + 1)]
    return times, [any(a <= t <= b for a, b in busy_ranges) for t in times]


def check_contract(segs, duration_s):
    assert validate_segments(segs, duration_s) == []
    for prev, nxt in zip(segs, segs[1:]):                       # the validator does not check this
        assert nxt["start_s"] - prev["end_s"] >= config.TRIAGE_MERGE_GAP_S


# ------------------------------------------------------------------ busy flags -> segments
def test_nothing_busy_gives_no_segments():
    times, busy = samples(10, [])
    assert segments_from_activity(times, busy, 10.0) == []


def test_one_run_is_padded_by_one_second():
    times, busy = samples(20, [(5.0, 7.0)])
    segs = segments_from_activity(times, busy, 20.0)
    assert [(s["start_s"], s["end_s"]) for s in segs] == [(4.0, 8.0)]
    check_contract(segs, 20.0)


def test_padding_is_clamped_to_the_video():
    times, busy = samples(10, [(0.0, 0.4), (9.6, 10.0)])
    segs = segments_from_activity(times, busy, 10.0)
    assert segs[0]["start_s"] == 0.0 and segs[-1]["end_s"] == 10.0
    check_contract(segs, 10.0)


def test_gaps_under_two_seconds_are_merged():
    times, busy = samples(20, [(2.0, 3.0), (6.4, 7.0)])         # padded: 1-4 and 5.4-8 -> gap 1.4 s
    segs = segments_from_activity(times, busy, 20.0)
    assert [(s["start_s"], s["end_s"]) for s in segs] == [(1.0, 8.0)]
    check_contract(segs, 20.0)


def test_gaps_of_two_seconds_or_more_stay_apart():
    times, busy = samples(20, [(2.0, 3.0), (8.0, 9.0)])         # padded: 1-4 and 7-10 -> gap 3 s
    segs = segments_from_activity(times, busy, 20.0)
    assert [(s["start_s"], s["end_s"]) for s in segs] == [(1.0, 4.0), (7.0, 10.0)]
    check_contract(segs, 20.0)


def test_activity_score_is_share_of_busy_samples():
    times, busy = samples(20, [(5.0, 7.0)])                     # 11 busy of 21 samples in 4.0-8.0
    (seg,) = segments_from_activity(times, busy, 20.0)
    assert seg["activity_score"] == round(11 / 21, 3)


def test_single_sample_blips_are_dropped():
    assert drop_blips([False, True, False, True, True, False, True]) == \
        [False, False, False, True, True, False, False]


# ------------------------------------------------------------------ real video files
# Same look as scripts/make_test_video.py (640x360, 150 one-pixel specks), but short and
# with the fish at a known time, so the expected answer is exact.
W, H, FPS, SECONDS = 640, 360, 10, 10


def write_clip(path, fish_from_s=None, fish_to_s=None):
    """Murky noisy background with drifting specks; optionally a dark fish crossing left to right."""
    rng = np.random.default_rng(3)
    specks = rng.uniform([0, 0], [W, H], size=(150, 2))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (W, H))
    for i in range(FPS * SECONDS):
        t = i / FPS
        frame = np.full((H, W, 3), (70, 95, 60), np.uint8)
        frame = cv2.add(frame, rng.integers(0, 25, (H, W, 3), dtype=np.uint8))
        specks[:, 0] = (specks[:, 0] + rng.normal(0.6, 0.4, len(specks))) % W
        specks[:, 1] = (specks[:, 1] + rng.normal(0.2, 0.4, len(specks))) % H
        for x, y in specks.astype(int):
            cv2.circle(frame, (x, y), 1, (150, 165, 140), -1)
        if fish_from_s is not None and fish_from_s <= t <= fish_to_s:
            cx = int((t - fish_from_s) / (fish_to_s - fish_from_s) * W)
            cv2.ellipse(frame, (cx, 180), (38, 14), 0, 0, 360, (30, 40, 35), -1)
        writer.write(cv2.GaussianBlur(frame, (5, 5), 0))
    writer.release()
    return path


@pytest.fixture
def quiet_clip(tmp_path):
    return write_clip(tmp_path / "quiet.mp4")


@pytest.fixture
def fish_clip(tmp_path):
    return write_clip(tmp_path / "fish.mp4", fish_from_s=4.0, fish_to_s=6.0)


def test_only_drifting_sediment_is_skipped(quiet_clip):
    assert run_triage(quiet_clip) == []


def test_fish_is_kept_and_quiet_parts_are_skipped(fish_clip):
    segs = run_triage(fish_clip)
    check_contract(segs, video_info(fish_clip)["duration_s"])
    assert len(segs) == 1
    (seg,) = segs
    assert seg["start_s"] <= 4.0 and seg["end_s"] >= 6.0       # recall: the whole crossing is covered
    assert seg["start_s"] >= 2.0 and seg["end_s"] <= 8.0       # the quiet start and end are skipped
