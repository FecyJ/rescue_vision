from __future__ import annotations

import json

import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.record_cli import parse_tags
from rescue_vision.data.build_manifest import build_dataset_records
from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.motion import (
    MANUAL_MOTION_LOG_FILENAME,
    MANUAL_MOTION_STREAM_NAME,
    ManualMotionLogWriter,
)


CALIBRATION_ID = "camera-front-20260729"


def test_recording_builds_dataset_manifest(tmp_path) -> None:
    recording = tmp_path / "recording-a"
    tags = {name: f"value-{name}" for name in REQUIRED_TAGS}
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={},
        session_tags=tags,
        image_coordinate_system="undistorted_pixel",
        calibration_id=CALIBRATION_ID,
        valid_pixel_ratio=0.95,
        undistort_fill_value=114,
    )
    recorder.start()
    assert recorder.record(
        CameraFrame(
            sequence=4,
            timestamp_ns=123456,
            image_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
        )
    )
    recorder.stop()

    records = build_dataset_records(
        [recording],
        dataset_root=tmp_path,
    )
    assert len(records) == 1
    record = records[0]
    assert record["sample_id"] == "recording-a/frame_00000004"
    assert record["recording_id"] == "recording-a"
    assert record["timestamp_ns"] == 123456
    assert record["image_coordinate_system"] == "undistorted_pixel"
    assert record["calibration_id"] == CALIBRATION_ID
    assert record["valid_pixel_ratio"] == 0.95
    assert record["undistort_fill_value"] == 114
    assert record["tags"] == dict(sorted(tags.items()))
    assert not str(record["image_path"]).startswith("/")


def test_manifest_validates_manual_motion_stream_time_coverage(tmp_path) -> None:
    recording = tmp_path / "manual-recording"
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={},
        session_tags={name: "known" for name in REQUIRED_TAGS},
        image_coordinate_system="undistorted_pixel",
        calibration_id=CALIBRATION_ID,
        valid_pixel_ratio=0.95,
        undistort_fill_value=114,
        recording_kind="supervised_manual_motion",
        auxiliary_streams={
            MANUAL_MOTION_STREAM_NAME: MANUAL_MOTION_LOG_FILENAME
        },
    )
    recorder.start()
    motion_log = ManualMotionLogWriter(
        recording / MANUAL_MOTION_LOG_FILENAME
    )
    motion_log.start(timestamp_ns=10)
    assert recorder.record(
        CameraFrame(
            sequence=0,
            timestamp_ns=20,
            image_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
        )
    )
    motion_log.stop(timestamp_ns=30)
    recorder.stop()

    records = build_dataset_records(
        [recording],
        dataset_root=tmp_path,
    )

    assert len(records) == 1
    assert records[0]["timestamp_ns"] == 20


def test_manifest_rejects_missing_session_tags(tmp_path) -> None:
    recording = tmp_path / "recording-a"
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={},
        image_coordinate_system="undistorted_pixel",
        calibration_id=CALIBRATION_ID,
        valid_pixel_ratio=0.95,
        undistort_fill_value=114,
    )
    recorder.start()
    recorder.record(
        CameraFrame(
            sequence=0,
            timestamp_ns=1,
            image_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
        )
    )
    recorder.stop()
    with pytest.raises(ValueError, match="missing session tags"):
        build_dataset_records(
            [recording],
            dataset_root=tmp_path,
        )


def test_manifest_detects_missing_image(tmp_path) -> None:
    recording = tmp_path / "recording-a"
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={},
        session_tags={name: "unknown" for name in REQUIRED_TAGS},
        image_coordinate_system="undistorted_pixel",
        calibration_id=CALIBRATION_ID,
        valid_pixel_ratio=0.95,
        undistort_fill_value=114,
    )
    recorder.start()
    recorder.record(
        CameraFrame(
            sequence=0,
            timestamp_ns=1,
            image_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
        )
    )
    recorder.stop()
    frame_record = json.loads(
        (recording / "frames.jsonl").read_text(encoding="utf-8")
    )
    (recording / frame_record["image_path"]).unlink()
    with pytest.raises(ValueError, match="does not exist"):
        build_dataset_records(
            [recording],
            dataset_root=tmp_path,
        )


def test_manifest_rejects_raw_pixel_recording(tmp_path) -> None:
    recording = tmp_path / "recording-a"
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={},
        session_tags={name: "unknown" for name in REQUIRED_TAGS},
    )
    recorder.start()
    recorder.record(
        CameraFrame(
            sequence=0,
            timestamp_ns=1,
            image_bgr=np.zeros((6, 8, 3), dtype=np.uint8),
        )
    )
    recorder.stop()

    with pytest.raises(ValueError, match="require undistorted_pixel"):
        build_dataset_records(
            [recording],
            dataset_root=tmp_path,
        )


def test_record_tag_parser_is_explicit_and_strict() -> None:
    tags = parse_tags(["lighting=bright", "distance=near"])
    assert tags["lighting"] == "bright"
    assert tags["distance"] == "near"
    assert tags["occlusion"] == "unknown"
    with pytest.raises(ValueError, match="Unknown tag"):
        parse_tags(["not_a_tag=value"])
