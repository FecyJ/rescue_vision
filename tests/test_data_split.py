from __future__ import annotations

import pytest

from rescue_vision.data.split_manifest import REQUIRED_TAGS, split_records


def record(sample_id: str, recording_id: str) -> dict:
    return {
        "schema_version": 2,
        "dataset_version": "dataset-v1",
        "sample_id": sample_id,
        "recording_id": recording_id,
        "image_path": f"{sample_id}.png",
        "image_coordinate_system": "undistorted_pixel",
        "intrinsics_fingerprint_sha256": "a" * 64,
        "valid_pixel_ratio": 0.95,
        "undistort_fill_value": 114,
        "tags": {tag: "known" for tag in REQUIRED_TAGS},
    }


def test_split_is_deterministic_and_groups_never_leak() -> None:
    records = [
        record(f"recording_{group}/frame_{frame}", f"recording_{group}")
        for group in range(20)
        for frame in range(3)
    ]
    first, first_report = split_records(records, seed="fixed")
    second, second_report = split_records(records, seed="fixed")
    assert first == second
    assert first_report == second_report

    group_splits: dict[str, set[str]] = {}
    for item in first:
        group_splits.setdefault(item["recording_id"], set()).add(item["split"])
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert set(first_report["tag_distribution"])


def test_split_rejects_missing_stratification_tag() -> None:
    invalid = record("sample", "recording")
    del invalid["tags"]["lighting"]
    with pytest.raises(ValueError, match="missing stratification"):
        split_records([invalid], seed="fixed")


def test_split_rejects_mixed_dataset_versions() -> None:
    first = record("sample-a", "recording-a")
    second = record("sample-b", "recording-b")
    second["dataset_version"] = "dataset-v2"
    with pytest.raises(ValueError, match="does not match"):
        split_records([first, second], seed="fixed")


def test_split_rejects_raw_or_unidentified_image_coordinates() -> None:
    invalid = record("sample", "recording")
    invalid["image_coordinate_system"] = "raw_pixel"
    with pytest.raises(ValueError, match="undistorted_pixel"):
        split_records([invalid], seed="fixed")


def test_split_rejects_nonstandard_undistort_fill() -> None:
    invalid = record("sample", "recording")
    invalid["undistort_fill_value"] = 0
    with pytest.raises(ValueError, match="must be 114"):
        split_records([invalid], seed="fixed")
