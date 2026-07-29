"""将一个或多个记录目录转换为严格的数据集清单。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rescue_vision.data.split_manifest import REQUIRED_TAGS
from rescue_vision.geometry.camera_model import IMAGE_BORDER_FILL_VALUE
from rescue_vision.motion.recording import (
    MANUAL_MOTION_LOG_FILENAME,
    MANUAL_MOTION_STREAM_NAME,
    inspect_manual_motion_log,
)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    return value


def _relative_to_root(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as error:
        raise ValueError(f"{path} is outside dataset root {root}.") from error


def build_dataset_records(
    recording_directories: list[Path],
    *,
    dataset_root: Path,
) -> list[dict[str, Any]]:
    if not recording_directories:
        raise ValueError("At least one recording directory is required.")

    output: list[dict[str, Any]] = []
    seen_recording_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for directory_value in recording_directories:
        directory = directory_value.expanduser().resolve()
        session = _load_json(directory / "session.json")
        motion_report: dict[str, object] | None = None
        recording_kind = session.get("recording_kind")
        auxiliary_streams = session.get("auxiliary_streams")
        if recording_kind not in {
            "camera",
            "supervised_manual_motion",
        } or not isinstance(auxiliary_streams, dict):
            raise ValueError(
                f"{directory}: invalid recording_kind/auxiliary_streams."
            )
        if recording_kind == "camera" and auxiliary_streams:
            raise ValueError(
                f"{directory}: camera recording cannot declare auxiliary streams."
            )
        if recording_kind == "supervised_manual_motion":
            if auxiliary_streams != {
                MANUAL_MOTION_STREAM_NAME: MANUAL_MOTION_LOG_FILENAME
            }:
                raise ValueError(f"{directory}: invalid manual_motion stream.")
            motion_report = inspect_manual_motion_log(
                directory / MANUAL_MOTION_LOG_FILENAME
            )
        if session.get("completed") is not True:
            raise ValueError(f"{directory}: recording is not marked completed.")
        if session.get("image_coordinate_system") != "undistorted_pixel":
            raise ValueError(
                f"{directory}: target datasets require undistorted_pixel "
                "recordings."
            )
        calibration_id = session.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError(f"{directory}: invalid calibration_id.")
        valid_pixel_ratio = session.get("valid_pixel_ratio")
        if (
            isinstance(valid_pixel_ratio, bool)
            or not isinstance(valid_pixel_ratio, (int, float))
            or not 0.0 < float(valid_pixel_ratio) <= 1.0
        ):
            raise ValueError(f"{directory}: invalid valid_pixel_ratio.")
        undistort_fill_value = session.get("undistort_fill_value")
        if undistort_fill_value != IMAGE_BORDER_FILL_VALUE:
            raise ValueError(
                f"{directory}: undistort_fill_value must be "
                f"{IMAGE_BORDER_FILL_VALUE}."
            )
        recording_id = session.get("recording_id")
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError(f"{directory}: invalid recording_id.")
        if recording_id in seen_recording_ids:
            raise ValueError(f"Duplicate recording_id {recording_id!r}.")
        seen_recording_ids.add(recording_id)

        tags = session.get("tags")
        if not isinstance(tags, dict):
            raise ValueError(f"{directory}: session tags must be a mapping.")
        missing_tags = sorted(REQUIRED_TAGS - set(tags))
        if missing_tags:
            raise ValueError(f"{directory}: missing session tags {missing_tags}.")
        if not all(isinstance(tags[name], str) and tags[name] for name in REQUIRED_TAGS):
            raise ValueError(f"{directory}: session tags must be non-empty strings.")

        annotation_path = directory / "annotations.json"
        annotations = _load_json(annotation_path)
        if annotations.get("recording_id") != recording_id:
            raise ValueError(
                f"{annotation_path}: recording_id does not match session."
            )

        frame_lines = (directory / "frames.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if not any(line.strip() for line in frame_lines):
            raise ValueError(f"{directory}: frames.jsonl is empty.")
        frame_count = 0
        previous_sequence = -1
        previous_timestamp_ns = -1
        first_timestamp_ns: int | None = None
        for line_number, line in enumerate(frame_lines, start=1):
            if not line.strip():
                continue
            frame = json.loads(line)
            if not isinstance(frame, dict):
                raise ValueError(
                    f"{directory}/frames.jsonl:{line_number}: invalid schema."
                )
            sequence = frame.get("sequence")
            timestamp_ns = frame.get("timestamp_ns")
            if (
                isinstance(sequence, bool)
                or not isinstance(sequence, int)
                or sequence < 0
                or isinstance(timestamp_ns, bool)
                or not isinstance(timestamp_ns, int)
                or timestamp_ns < 0
            ):
                raise ValueError(
                    f"{directory}/frames.jsonl:{line_number}: invalid sequence/time."
                )
            if sequence <= previous_sequence or timestamp_ns <= previous_timestamp_ns:
                raise ValueError(
                    f"{directory}/frames.jsonl:{line_number}: sequence and "
                    "timestamp_ns must be strictly increasing."
                )
            previous_sequence = sequence
            previous_timestamp_ns = timestamp_ns
            if first_timestamp_ns is None:
                first_timestamp_ns = timestamp_ns
            image_path = directory / str(frame.get("image_path"))
            if not image_path.is_file():
                raise ValueError(
                    f"{directory}/frames.jsonl:{line_number}: image does not exist."
                )

            sample_id = f"{recording_id}/frame_{sequence:08d}"
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r}.")
            seen_sample_ids.add(sample_id)
            frame_count += 1
            output.append(
                {
                    "sample_id": sample_id,
                    "recording_id": recording_id,
                    "frame_sequence": sequence,
                    "timestamp_ns": timestamp_ns,
                    "image_path": _relative_to_root(image_path, dataset_root),
                    "image_coordinate_system": "undistorted_pixel",
                    "calibration_id": calibration_id,
                    "valid_pixel_ratio": float(valid_pixel_ratio),
                    "undistort_fill_value": undistort_fill_value,
                    "annotation_manifest": _relative_to_root(
                        annotation_path,
                        dataset_root,
                    ),
                    "tags": {name: tags[name] for name in sorted(REQUIRED_TAGS)},
                }
            )
        if motion_report is not None:
            assert first_timestamp_ns is not None
            if (
                int(motion_report["first_timestamp_ns"])
                > first_timestamp_ns
                or int(motion_report["last_timestamp_ns"])
                < previous_timestamp_ns
            ):
                raise ValueError(
                    f"{directory}: manual motion log does not cover frame "
                    "timestamps."
                )
        statistics = session.get("statistics")
        if not isinstance(statistics, dict):
            raise ValueError(f"{directory}: session statistics must be a mapping.")
        if statistics.get("written_frames") != frame_count:
            raise ValueError(
                f"{directory}: written_frames does not match frames.jsonl "
                f"({statistics.get('written_frames')!r} != {frame_count})."
            )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a strict dataset JSONL from recording directories."
    )
    parser.add_argument("recordings", type=Path, nargs="+")
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    records = build_dataset_records(
        args.recordings,
        dataset_root=args.dataset_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
