from __future__ import annotations

import json
import sys

import pytest

import rescue_vision.camera.record_cli as record_cli
import rescue_vision.data.build_manifest as manifest_cli
import rescue_vision.data.split_manifest as split_cli
import rescue_vision.evaluation.cli as evaluation_cli
from rescue_vision.data.split_manifest import REQUIRED_TAGS


def test_record_cli_rejects_invalid_frame_limit(
    monkeypatch,
    tmp_path,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescue-vision-record",
            "--config",
            str(tmp_path / "runtime.yaml"),
            "--output",
            str(tmp_path / "recording"),
            "--frames",
            "0",
        ],
    )
    with pytest.raises(SystemExit) as raised:
        record_cli.main()
    assert raised.value.code == 2


def test_manifest_cli_writes_jsonl(monkeypatch, tmp_path) -> None:
    record = {"schema_version": 2, "sample_id": "session/frame_00000000"}
    monkeypatch.setattr(
        manifest_cli,
        "build_dataset_records",
        lambda *_args, **_kwargs: [record],
    )
    output = tmp_path / "out" / "manifest.jsonl"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescue-vision-manifest",
            "--dataset-root",
            str(tmp_path),
            "--dataset-version",
            "dataset-v1",
            "--output",
            str(output),
            str(tmp_path / "recording"),
        ],
    )
    manifest_cli.main()
    assert json.loads(output.read_text(encoding="utf-8")) == record


def test_split_cli_writes_empty_split_warning_before_nonzero_exit(
    monkeypatch,
    tmp_path,
) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "dataset_version": "dataset-v1",
                "sample_id": "session/frame_00000000",
                "recording_id": "session",
                "image_coordinate_system": "undistorted_pixel",
                "intrinsics_fingerprint_sha256": "a" * 64,
                "valid_pixel_ratio": 0.95,
                "undistort_fill_value": 114,
                "tags": {name: "known" for name in REQUIRED_TAGS},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "split.jsonl"
    report = tmp_path / "split.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescue-vision-split",
            str(manifest),
            "--output",
            str(output),
            "--report",
            str(report),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        split_cli.main()
    assert raised.value.code == 1
    assert output.is_file()
    assert json.loads(report.read_text(encoding="utf-8"))["warnings"]


def test_evaluation_cli_writes_report(monkeypatch, tmp_path) -> None:
    records = tmp_path / "evaluation.jsonl"
    records.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "sample_id": "sample",
                "object_id": "danger",
                "ground_truth_class": "blue_danger",
                "predicted_class": None,
                "confidence": None,
                "ground_truth_ground_mm": None,
                "predicted_ground_mm": None,
                "capture_timestamp_ns": 1,
                "result_timestamp_ns": 2,
                "tags": {},
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "report.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rescue-vision-evaluate",
            str(records),
            "--output",
            str(output),
            "--model-version",
            "model-v1",
            "--dataset-version",
            "dataset-v1",
            "--code-version",
            "code-v1",
        ],
    )
    evaluation_cli.main()
    assert json.loads(output.read_text(encoding="utf-8"))["record_count"] == 1
