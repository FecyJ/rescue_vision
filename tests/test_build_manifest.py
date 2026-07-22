from __future__ import annotations

import json

import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.camera.record_cli import parse_tags
from rescue_vision.data.build_manifest import build_dataset_records
from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.camera.recording import FrameRecorder


def test_recording_builds_verified_dataset_manifest(tmp_path) -> None:
    recording = tmp_path / "recording-a"
    tags = {name: f"value-{name}" for name in REQUIRED_TAGS}
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={"schema_version": 1},
        versions={"code": "abc-dirty"},
        session_tags=tags,
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
        dataset_version="dataset-v1",
    )
    assert len(records) == 1
    record = records[0]
    assert record["sample_id"] == "recording-a/frame_00000004"
    assert record["recording_id"] == "recording-a"
    assert record["dataset_version"] == "dataset-v1"
    assert record["timestamp_ns"] == 123456
    assert record["tags"] == dict(sorted(tags.items()))
    assert not str(record["image_path"]).startswith("/")


def test_manifest_rejects_missing_session_tags(tmp_path) -> None:
    recording = tmp_path / "recording-a"
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={"schema_version": 1},
        versions={"code": "abc"},
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
            dataset_version="dataset-v1",
        )


def test_manifest_detects_image_corruption(tmp_path) -> None:
    recording = tmp_path / "recording-a"
    recorder = FrameRecorder(
        recording,
        image_size=(8, 6),
        config_snapshot={"schema_version": 1},
        versions={"code": "abc"},
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
    frame_record = json.loads(
        (recording / "frames.jsonl").read_text(encoding="utf-8")
    )
    (recording / frame_record["image_path"]).write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="hash mismatch"):
        build_dataset_records(
            [recording],
            dataset_root=tmp_path,
            dataset_version="dataset-v1",
        )


def test_record_tag_parser_is_explicit_and_strict() -> None:
    tags = parse_tags(["lighting=bright", "distance=near"])
    assert tags["lighting"] == "bright"
    assert tags["distance"] == "near"
    assert tags["occlusion"] == "unknown"
    with pytest.raises(ValueError, match="Unknown tag"):
        parse_tags(["not_a_tag=value"])
