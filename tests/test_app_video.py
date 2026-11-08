"""Sample-generation behavior; real FFmpeg check runs when the tool is available."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import pytest

from scripts import make_test_video as sample


def prepare(monkeypatch, tmp_path, ffmpeg):
    monkeypatch.setattr(sample.config, "CLIPS_DIR", tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_test_video.py", "--ffmpeg", ffmpeg])
    output = tmp_path / "test_clip.mp4"
    output.write_bytes(b"previous sample")
    return output


def test_missing_encoder_preserves_existing_sample(monkeypatch, tmp_path):
    output = prepare(monkeypatch, tmp_path, str(tmp_path / "missing-ffmpeg.exe"))
    with pytest.raises(SystemExit, match="FFmpeg"):
        sample.main()
    assert output.read_bytes() == b"previous sample"
    assert list(tmp_path.iterdir()) == [output]


def test_failed_conversion_preserves_sample_and_cleans_staging(monkeypatch, tmp_path):
    output = prepare(monkeypatch, tmp_path, sys.executable)

    def failed_encoder(command, **kwargs):
        Path(command[-1]).write_bytes(b"partial encoding")
        raise subprocess.CalledProcessError(1, command, stderr="Unknown encoder 'libx264'")

    monkeypatch.setattr(subprocess, "run", failed_encoder)
    with pytest.raises(SystemExit, match="libx264"):
        sample.main()
    assert output.read_bytes() == b"previous sample"
    assert list(tmp_path.iterdir()) == [output]


def test_generated_sample_is_h264_with_original_frame_contract(monkeypatch, tmp_path):
    ffmpeg = os.environ.get("TIDESIGHT_TEST_FFMPEG") or shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("Install FFmpeg or set TIDESIGHT_TEST_FFMPEG for real encoding check")
    output = prepare(monkeypatch, tmp_path, ffmpeg)
    sample.main()
    cap = cv2.VideoCapture(str(output))
    try:
        assert cap.isOpened()
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 255) for i in range(4))
        # OpenCV backends may report the codec name or the MP4 sample-entry tag.
        assert codec in {"avc1", "h264"}, f"Browser sample must be H.264, got {codec}"
        assert cap.get(cv2.CAP_PROP_FPS) == 25
        frames = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            assert frame.shape == (360, 640, 3)
            frames += 1
        assert frames == 375  # 15 seconds, every original frame retained
    finally:
        cap.release()
    data = output.read_bytes()
    assert data.index(b"moov") < data.index(b"mdat")  # faststart metadata
    assert list(tmp_path.iterdir()) == [output]
