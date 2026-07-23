from __future__ import annotations

import json

import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.recording import FrameRecorder
from rescue_vision.data.check_recording import (
    PICAMERA2_METADATA,
    check_requirements,
    inspect_recording,
)


def make_recording(tmp_path):
    directory = tmp_path / "capture-smoke"
    recorder = FrameRecorder(
        directory,
        image_size=(8, 6),
        config_snapshot={"camera": {"fps": 20}},
        versions={"code": "test"},
    )
    recorder.start()
    for sequence, value in enumerate((20, 80, 140)):
        assert recorder.record(
            CameraFrame(
                sequence=sequence,
                timestamp_ns=1_000_000_000 + sequence * 50_000_000,
                image_bgr=np.full((6, 8, 3), value, dtype=np.uint8),
                metadata={
                    name: ("picamera2" if name == "source" else sequence + 1)
                    for name in PICAMERA2_METADATA
                },
            )
        )
    recorder.stop()
    return directory


def test_inspection_replays_hashes_and_reports_capture_health(tmp_path) -> None:
    report = inspect_recording(make_recording(tmp_path))

    assert report["frame_count"] == 3
    assert report["image_coordinate_system"] == "raw_pixel"
    assert report["effective_fps"] == pytest.approx(20.0)
    assert report["drop_ratio"] == 0.0
    assert report["sequence_gap_count"] == 0
    assert report["metadata_coverage"]["sensor_timestamp_ns"]["ratio"] == 1.0
    assert report["luma"]["mean_median"] == pytest.approx(80.0)
    assert (
        check_requirements(
            report,
            minimum_frames=3,
            maximum_drop_ratio=0.0,
            minimum_fps_ratio=1.0,
            required_metadata=PICAMERA2_METADATA,
            require_undistorted=False,
        )
        == []
    )


def test_requirements_report_actionable_failures(tmp_path) -> None:
    report = inspect_recording(make_recording(tmp_path))
    report["configured_fps"] = 40.0
    report["drop_ratio"] = 0.25
    report["metadata_coverage"]["lens_position"]["present_frames"] = 2

    failures = check_requirements(
        report,
        minimum_frames=4,
        maximum_drop_ratio=0.05,
        minimum_fps_ratio=0.8,
        required_metadata=("lens_position",),
        require_undistorted=True,
    )

    assert any("frame_count" in failure for failure in failures)
    assert any("drop_ratio" in failure for failure in failures)
    assert any("effective_fps" in failure for failure in failures)
    assert any("lens_position" in failure for failure in failures)
    assert any("undistorted_pixel" in failure for failure in failures)
    assert any("undistort_fill_value" in failure for failure in failures)


def test_inspection_rejects_incomplete_or_corrupt_recording(tmp_path) -> None:
    directory = make_recording(tmp_path)
    session_path = directory / "session.json"
    session = json.loads(session_path.read_text(encoding="utf-8"))
    session["completed"] = False
    session_path.write_text(json.dumps(session), encoding="utf-8")
    with pytest.raises(ValueError, match="not marked completed"):
        inspect_recording(directory)

    session["completed"] = True
    session_path.write_text(json.dumps(session), encoding="utf-8")
    frame = json.loads(
        (directory / "frames.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    (directory / frame["image_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        inspect_recording(directory)
