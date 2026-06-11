from __future__ import annotations

import sys

import pytest

cv2 = pytest.importorskip("cv2")
pytest.importorskip("matplotlib")

from penalty_kick_distancer.config import ProcessingConfig, TargetConfig
from penalty_kick_distancer.video_processor import process_video


class FakeBoxes:
    data = [[40, 40, 60, 60, 0.95, 0]]


class FakeResult:
    boxes = FakeBoxes()
    names = {0: "Ball"}


class FakeModel:
    def __call__(self, frame, verbose=False):
        return [FakeResult()]


def test_process_video_with_mocked_detection(tmp_path):
    try:
        import numpy as np
    except ImportError:
        pytest.skip("numpy is not installed")

    video_path = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), 5.0, (100, 100))
    if not writer.isOpened():
        pytest.skip("OpenCV MP4 writer is unavailable in this environment")
    for _ in range(3):
        writer.write(np.zeros((100, 100, 3), dtype=np.uint8))
    writer.release()

    config = ProcessingConfig(
        target=TargetConfig(x=50, y=50, width=20, height=20),
        frame_stride=1,
        max_frames=3,
        output_dir=tmp_path / "out",
    )
    output_video, best_frame, metrics, per_frame, metadata, plot_path = process_video(
        video_path,
        FakeModel(),
        config,
    )

    assert output_video.endswith(".mp4")
    assert best_frame is not None
    assert metrics["detections"] == 3
    assert metrics["final_grade"] == "SSS"
    assert len(per_frame) == 3
    assert metadata["metrics"]["total_processed_frames"] == 3
    assert plot_path is not None

