from __future__ import annotations

import json
import queue

import cv2
import numpy as np
import pytest

import rescue_vision.camera.replay as replay_module
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.replay import (
    ImageDirectorySource,
    RecordingSource,
    VideoFileSource,
)
from rescue_vision.camera.recording import FrameRecorder


def frame(sequence: int, value: int) -> CameraFrame:
    return CameraFrame(
        sequence=sequence,
        timestamp_ns=1_000_000_000 + sequence * 50_000_000,
        image_bgr=np.full((6, 8, 3), value, dtype=np.uint8),
        metadata={"exposure_time_us": 1000 + sequence},
    )


def test_recorder_roundtrip_is_deterministic(tmp_path) -> None:
    session = tmp_path / "recording_001"
    recorder = FrameRecorder(
        session,
        image_size=(8, 6),
        config_snapshot={"schema_version": 1, "camera": {"fps": 20}},
        versions={"code": "abc", "config": "v1"},
        queue_capacity=2,
    )
    recorder.start()
    assert recorder.record(frame(7, 11))
    assert recorder.record(frame(8, 22))
    recorder.stop()
    assert recorder.written_frames == 2
    assert (session / "annotations.json").exists()

    source = RecordingSource(session)
    source.start()
    first = source.read()
    second = source.read()
    with pytest.raises(EOFError):
        source.read()
    source.stop()
    assert (first.sequence, second.sequence) == (7, 8)
    assert (first.timestamp_ns, second.timestamp_ns) == (1_350_000_000, 1_400_000_000)
    assert np.array_equal(first.image_bgr, frame(7, 11).image_bgr)
    assert first.metadata["exposure_time_us"] == 1007
    assert first.metadata["image_coordinate_system"] == "raw_pixel"
    assert first.age_ns(first.timestamp_ns + 10_000_000) == 10_000_000
    assert first.is_stale(first.timestamp_ns + 151_000_000, 150.0)

    replay = RecordingSource(session)
    replay.start()
    repeated = replay.read()
    replay.stop()
    assert repeated.sequence == first.sequence
    assert repeated.timestamp_ns == first.timestamp_ns
    assert np.array_equal(repeated.image_bgr, first.image_bgr)

    session_document = json.loads((session / "session.json").read_text())
    assert session_document["schema_version"] == 3
    assert session_document["image_coordinate_system"] == "raw_pixel"
    assert session_document["intrinsics_fingerprint_sha256"] is None
    assert session_document["undistort_fill_value"] is None
    assert session_document["completed"] is True
    assert session_document["statistics"]["dropped_frames"] == 0

    first_path = session / json.loads(
        (session / "frames.jsonl").read_text().splitlines()[0]
    )["image_path"]
    first_path.write_bytes(b"corrupt")
    corrupt = RecordingSource(session)
    corrupt.start()
    with pytest.raises(ValueError, match="hash mismatch"):
        corrupt.read()
    corrupt.stop()


def test_recorder_queue_full_does_not_block(tmp_path, monkeypatch) -> None:
    recorder = FrameRecorder(
        tmp_path / "recording",
        image_size=(8, 6),
        config_snapshot={"schema_version": 1},
        versions={"code": "abc"},
        queue_capacity=1,
    )
    recorder.start()

    def full(_item) -> None:
        raise queue.Full

    monkeypatch.setattr(recorder._queue, "put_nowait", full)
    assert recorder.record(frame(0, 0)) is False
    assert recorder.dropped_frames == 1
    recorder.stop()
    session = json.loads(
        (tmp_path / "recording" / "session.json").read_text(encoding="utf-8")
    )
    assert session["completed"] is False


def test_undistorted_recording_replays_calibration_identity(tmp_path) -> None:
    fingerprint = "a" * 64
    recorder = FrameRecorder(
        tmp_path / "recording",
        image_size=(8, 6),
        config_snapshot={"schema_version": 3},
        versions={"code": "abc"},
        image_coordinate_system="undistorted_pixel",
        intrinsics_fingerprint_sha256=fingerprint,
        valid_pixel_ratio=0.95,
        undistort_fill_value=114,
    )
    recorder.start()
    recorder.record(frame(0, 10))
    recorder.stop()

    with RecordingSource(tmp_path / "recording") as source:
        replayed = source.read()

    assert replayed.metadata["image_coordinate_system"] == "undistorted_pixel"
    assert replayed.metadata["intrinsics_fingerprint_sha256"] == fingerprint
    assert replayed.metadata["valid_pixel_ratio"] == 0.95
    assert replayed.metadata["undistort_fill_value"] == 114


def test_image_directory_source_has_stable_order_and_time(tmp_path) -> None:
    cv2.imwrite(str(tmp_path / "b.png"), frame(0, 20).image_bgr)
    cv2.imwrite(str(tmp_path / "a.png"), frame(0, 10).image_bgr)
    source = ImageDirectorySource(tmp_path, fps=10.0, start_timestamp_ns=5)
    assert source.image_size == (8, 6)
    source.start()
    first = source.read()
    second = source.read()
    assert first.metadata["filename"] == "a.png"
    assert second.metadata["filename"] == "b.png"
    assert (first.timestamp_ns, second.timestamp_ns) == (5, 100_000_005)
    with pytest.raises(EOFError):
        source.read()
    source.stop()


class FakeCapture:
    def __init__(self) -> None:
        self.frames = [frame(0, 1).image_bgr, frame(1, 2).image_bgr]
        self.released = False

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FPS: 25.0,
            cv2.CAP_PROP_FRAME_WIDTH: 8.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 6.0,
        }[property_id]

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self) -> None:
        self.released = True


def test_video_source_uses_frame_index_for_timestamp(monkeypatch, tmp_path) -> None:
    capture = FakeCapture()
    monkeypatch.setattr(replay_module.cv2, "VideoCapture", lambda _path: capture)
    source = VideoFileSource(tmp_path / "video.mp4", start_timestamp_ns=100)
    source.start()
    assert source.image_size == (8, 6)
    assert source.read().timestamp_ns == 100
    assert source.read().timestamp_ns == 40_000_100
    with pytest.raises(EOFError):
        source.read()
    source.stop()
    assert capture.released
