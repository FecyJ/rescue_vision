from __future__ import annotations

import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.record_cli import record_session
from rescue_vision.camera.recording import FrameRecorder


class FakeSource:
    image_size = (8, 6)

    def __init__(self, *, fail_at: int | None = None) -> None:
        self.fail_at = fail_at
        self.sequence = 0
        self.started = False
        self.stopped = False

    def start(self) -> None:
        self.started = True

    def read(self, timeout: float = 1.0) -> CameraFrame:
        if self.fail_at is not None and self.sequence == self.fail_at:
            raise RuntimeError("synthetic camera failure")
        sequence = self.sequence
        self.sequence += 1
        return CameraFrame(
            sequence=sequence,
            timestamp_ns=1_000_000_000 + sequence * 50_000_000,
            image_bgr=np.full((6, 8, 3), sequence, dtype=np.uint8),
        )

    def stop(self) -> None:
        self.stopped = True


def make_recorder(tmp_path) -> FrameRecorder:
    return FrameRecorder(
        tmp_path / "recording",
        image_size=(8, 6),
        config_snapshot={"camera": {"fps": 20}},
        versions={"code": "test"},
    )


def test_record_session_runs_complete_capture_and_releases_resources(tmp_path) -> None:
    source = FakeSource()
    recorder = make_recorder(tmp_path)

    delivered = record_session(source, recorder, frame_limit=3)

    assert delivered == 3
    assert source.started
    assert source.stopped
    assert recorder.written_frames == 3


def test_record_session_releases_resources_after_camera_failure(tmp_path) -> None:
    source = FakeSource(fail_at=1)
    recorder = make_recorder(tmp_path)

    with pytest.raises(RuntimeError, match="synthetic camera failure"):
        record_session(source, recorder, frame_limit=3)

    assert source.stopped
    assert recorder.written_frames == 1


@pytest.mark.parametrize(
    ("frame_limit", "duration_seconds", "message"),
    [
        (0, None, "frame_limit"),
        (None, 0.0, "duration_seconds"),
        (1, 1.0, "mutually exclusive"),
    ],
)
def test_record_session_rejects_invalid_limits(
    tmp_path,
    frame_limit,
    duration_seconds,
    message,
) -> None:
    with pytest.raises(ValueError, match=message):
        record_session(
            FakeSource(),
            make_recorder(tmp_path),
            frame_limit=frame_limit,
            duration_seconds=duration_seconds,
        )
