from __future__ import annotations

import json

import pytest

from manual_tests.dataset_perception import _read_manifest


def test_dataset_perception_reads_schema_v2_manifest(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    record = {
        "schema_version": 2,
        "sample_id": "session/frame_00000000",
        "image_path": "session/frames/00000000_1.jpg",
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    assert _read_manifest(path) == [record]


def test_dataset_perception_rejects_empty_or_invalid_manifest(tmp_path) -> None:
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no samples"):
        _read_manifest(path)
    path.write_text("{broken\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"manifest.jsonl:1"):
        _read_manifest(path)
