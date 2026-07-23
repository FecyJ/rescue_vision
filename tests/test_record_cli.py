from __future__ import annotations

import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.record_cli import (
    record_session,
    undistort_camera_frame,
)
from rescue_vision.camera.replay import RecordingSource
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.geometry.camera_model import (
    IMAGE_BORDER_FILL_VALUE,
    CameraCalibration,
    CameraModel,
    CameraModelType,
)


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


def test_record_session_applies_frame_transform_before_writing(tmp_path) -> None:
    source = FakeSource()
    recorder = make_recorder(tmp_path)

    def brighten(frame: CameraFrame) -> CameraFrame:
        return CameraFrame(
            sequence=frame.sequence,
            timestamp_ns=frame.timestamp_ns,
            image_bgr=np.full((6, 8, 3), 123, dtype=np.uint8),
            metadata=frame.metadata,
        )

    record_session(
        source,
        recorder,
        frame_limit=1,
        frame_transform=brighten,
    )

    with RecordingSource(tmp_path / "recording") as replay:
        assert np.all(replay.read().image_bgr == 123)


def test_record_session_rejects_transform_that_changes_frame_identity(
    tmp_path,
) -> None:
    source = FakeSource()
    recorder = make_recorder(tmp_path)

    def change_sequence(frame: CameraFrame) -> CameraFrame:
        return CameraFrame(
            sequence=frame.sequence + 1,
            timestamp_ns=frame.timestamp_ns,
            image_bgr=frame.image_bgr,
        )

    with pytest.raises(ValueError, match="preserve sequence"):
        record_session(
            source,
            recorder,
            frame_limit=1,
            frame_transform=change_sequence,
        )
    assert source.stopped


def test_undistort_camera_frame_records_coordinate_identity() -> None:
    calibration = CameraCalibration(
        model=CameraModelType.PINHOLE,
        image_size=(8, 6),
        K=np.eye(3),
        D=np.zeros(5),
        new_K=np.eye(3),
    )
    source_frame = CameraFrame(
        sequence=2,
        timestamp_ns=123,
        image_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
        metadata={"source": "fake"},
    )

    result = undistort_camera_frame(
        source_frame,
        camera_model=CameraModel(calibration),
    )

    assert result.sequence == source_frame.sequence
    assert result.timestamp_ns == source_frame.timestamp_ns
    assert result.metadata["image_coordinate_system"] == "undistorted_pixel"
    assert result.metadata["intrinsics_fingerprint_sha256"] == (
        calibration.fingerprint()
    )
    assert result.metadata["undistort_fill_value"] == IMAGE_BORDER_FILL_VALUE


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
