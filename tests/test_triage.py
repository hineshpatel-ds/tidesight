"""Triage tests (Video + classical CV). Run with:  python -m pytest -q tests/test_triage.py"""
import cv2
import numpy as np
import pytest

from tidesight import config
from tidesight.contracts import frames_to_process, validate_segments, video_info
from tidesight.triage import (_SPOT_CHECK_EVERY_S, _SPOT_CHECK_LEN_S, codec_block_px, drop_blips, run_triage,
                              segments_from_activity, speck_size, spot_check_times)

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


# ------------------------------------------------------------------ spot checks
def test_spot_checks_start_at_zero_and_repeat():
    assert spot_check_times(30.0, 10.0) == [0.0, 10.0, 20.0]
    assert spot_check_times(25.0, 10.0) == [0.0, 10.0, 20.0]
    assert spot_check_times(5.0, 10.0) == [0.0]


def test_spot_checks_are_kept_without_motion():
    times, busy = samples(30, [])
    segs = segments_from_activity(times, busy, 30.0, spot_checks=[0.0, 10.0, 20.0], spot_len_s=0.4)
    assert [(s["start_s"], s["end_s"], s["activity_score"]) for s in segs] == \
        [(0.0, 0.4, 0.0), (10.0, 10.4, 0.0), (20.0, 20.4, 0.0)]
    check_contract(segs, 30.0)


def test_spot_check_near_motion_is_merged_and_end_is_clamped():
    times, busy = samples(20, [(11.0, 12.0)])                   # padded: 10-13, spot 10-10.4 inside
    segs = segments_from_activity(times, busy, 20.0, spot_checks=[0.0, 10.0, 19.9], spot_len_s=0.4)
    assert [(s["start_s"], s["end_s"]) for s in segs] == [(0.0, 0.4), (10.0, 13.0), (19.6, 20.0)]
    check_contract(segs, 20.0)


# ------------------------------------------------------------------ real video files
# Same look as scripts/make_test_video.py (640x360, 150 one-pixel specks), but short and
# with the fish at a known time, so the expected answer is exact.
W, H, FPS, SECONDS = 640, 360, 10, 10


def write_clip(path, fish_from_s=None, fish_to_s=None, still_fish=False):
    """Murky noisy background with drifting specks; optionally a dark fish crossing left to right
    between fish_from_s and fish_to_s, and/or a dark fish that sits still mid-frame the whole time."""
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
        if still_fish:
            cv2.ellipse(frame, (320, 180), (38, 14), 0, 0, 360, (30, 40, 35), -1)
        writer.write(cv2.GaussianBlur(frame, (5, 5), 0))
    writer.release()
    return path


@pytest.fixture
def quiet_clip(tmp_path):
    return write_clip(tmp_path / "quiet.mp4")


@pytest.fixture
def fish_clip(tmp_path):
    return write_clip(tmp_path / "fish.mp4", fish_from_s=4.0, fish_to_s=6.0)


def with_motion(segs):
    return [s for s in segs if s["activity_score"] > 0]


def is_spot_check(seg):
    return seg["activity_score"] == 0 and seg["end_s"] - seg["start_s"] <= _SPOT_CHECK_LEN_S + 1e-9


def test_only_drifting_sediment_is_skipped(quiet_clip):
    segs = run_triage(quiet_clip)
    check_contract(segs, SECONDS)
    assert with_motion(segs) == []
    assert segs and all(is_spot_check(s) for s in segs)          # only the always-kept spot checks


def test_fish_is_kept_and_quiet_parts_are_skipped(fish_clip):
    segs = run_triage(fish_clip)
    check_contract(segs, video_info(fish_clip)["duration_s"])
    assert len(with_motion(segs)) == 1
    (seg,) = with_motion(segs)
    assert seg["start_s"] <= 4.0 and seg["end_s"] >= 6.0       # recall: the whole crossing is covered
    assert seg["start_s"] >= 2.0 and seg["end_s"] <= 8.0       # the quiet start and end are skipped
    assert all(is_spot_check(s) for s in segs if s is not seg)


def test_clip_that_cannot_seek_falls_back_to_opening_warm_up(fish_clip, monkeypatch):
    import tidesight.triage as triage
    monkeypatch.setattr(triage, "_spread_frames", lambda *args: [])
    (seg,) = with_motion(run_triage(fish_clip))
    assert seg["start_s"] <= 4.0 and seg["end_s"] >= 6.0


# Regressions from the PR #3 review (Gavin): no motion is not proof of no fish.
def test_fish_in_the_first_half_second_is_kept(tmp_path):
    """The background is learned from the whole clip, so an early fish is not learned as background."""
    clip = write_clip(tmp_path / "early.mp4", fish_from_s=0.2, fish_to_s=0.6)
    segs = with_motion(run_triage(clip))
    assert segs and segs[0]["start_s"] <= 0.2 and segs[0]["end_s"] >= 0.6


def write_faint_clip(path, fish_from_s=4.0, fish_to_s=6.0, seconds=10, contrast=4):
    """1280x720 (shrunk 2x by triage, like real footage), flat murky water with per-frame sensor noise
    and a fish only `contrast` grey levels darker than the water: like the distant schools in the
    Coastal Action clip T1-STAT2 411 (~4 levels), which the old MOG2 triage could not see.
    Checked: the MOG2 version found nothing here at contrast 3-6; it needed 8."""
    w, h = 1280, 720
    rng = np.random.default_rng(5)
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for i in range(FPS * seconds):
        t = i / FPS
        frame = np.full((h, w, 3), (70, 95, 60), np.float32) + rng.normal(0, 3, (h, w, 1))
        if fish_from_s <= t <= fish_to_s:
            cx = int((t - fish_from_s) / (fish_to_s - fish_from_s) * w)
            body = np.zeros((h, w), np.uint8)
            cv2.ellipse(body, (cx, 300), (60, 22), 0, 0, 360, 1, -1)
            frame[body > 0] -= contrast
        writer.write(np.clip(frame, 0, 255).astype(np.uint8))
    writer.release()
    return path


def test_faint_fish_in_murky_water_is_kept(tmp_path):
    clip = write_faint_clip(tmp_path / "faint.mp4")
    segs = with_motion(run_triage(clip))
    assert segs and any(s["start_s"] <= 4.2 and s["end_s"] >= 5.8 for s in segs)
    assert all(s["start_s"] >= 2.0 for s in segs)                 # noise alone does not count as motion


def test_speck_size_follows_the_video_compression_grid():
    assert speck_size(1.0) == 9          # a 640 px clip is not shrunk: 8 px codec blocks stay 8 px
    assert speck_size(1 / 6) == 5        # 4K shrunk to 640: blocks are ~1 px, the fish-scale minimum applies
    assert codec_block_px(1 / 2) == 4.0


def test_fish_that_never_moves_is_still_shown_to_the_detector(tmp_path):
    """Motion can't see it, so spot checks must send frames from the start and every few seconds."""
    clip = write_clip(tmp_path / "still.mp4", still_fish=True)
    info = video_info(clip)
    segs = run_triage(clip)
    check_contract(segs, info["duration_s"])
    frames = frames_to_process(segs, info["fps"], info["frame_count"], step=2)
    assert frames and frames[0] == 0
    times = [0.0] + [f / info["fps"] for f in frames] + [info["duration_s"]]
    assert max(b - a for a, b in zip(times, times[1:])) <= _SPOT_CHECK_EVERY_S
