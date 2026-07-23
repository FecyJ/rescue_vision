"""将一个或多个记录目录转换为严格的数据集清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from rescue_vision.data.split_manifest import REQUIRED_TAGS


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
    dataset_version: str,
    verify_images: bool = True,
) -> list[dict[str, Any]]:
    if not dataset_version:
        raise ValueError("dataset_version must be non-empty.")
    if not recording_directories:
        raise ValueError("At least one recording directory is required.")

    output: list[dict[str, Any]] = []
    seen_recording_ids: set[str] = set()
    seen_sample_ids: set[str] = set()
    for directory_value in recording_directories:
        directory = directory_value.expanduser().resolve()
        session = _load_json(directory / "session.json")
        if session.get("schema_version") != 2:
            raise ValueError(f"{directory}: session schema_version must be 2.")
        if session.get("completed") is not True:
            raise ValueError(f"{directory}: recording is not marked completed.")
        if session.get("image_coordinate_system") != "undistorted_pixel":
            raise ValueError(
                f"{directory}: target datasets require undistorted_pixel "
                "recordings."
            )
        intrinsics_fingerprint = session.get(
            "intrinsics_fingerprint_sha256"
        )
        if (
            not isinstance(intrinsics_fingerprint, str)
            or len(intrinsics_fingerprint) != 64
            or intrinsics_fingerprint != intrinsics_fingerprint.lower()
            or any(
                character not in "0123456789abcdef"
                for character in intrinsics_fingerprint.lower()
            )
        ):
            raise ValueError(
                f"{directory}: invalid intrinsics fingerprint."
            )
        valid_pixel_ratio = session.get("valid_pixel_ratio")
        if (
            isinstance(valid_pixel_ratio, bool)
            or not isinstance(valid_pixel_ratio, (int, float))
            or not 0.0 < float(valid_pixel_ratio) <= 1.0
        ):
            raise ValueError(f"{directory}: invalid valid_pixel_ratio.")
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
        if annotations.get("schema_version") != 1:
            raise ValueError(f"{annotation_path}: schema_version must be 1.")
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
        for line_number, line in enumerate(frame_lines, start=1):
            if not line.strip():
                continue
            frame = json.loads(line)
            if not isinstance(frame, dict) or frame.get("schema_version") != 1:
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
            image_path = directory / str(frame.get("image_path"))
            expected_hash = frame.get("image_sha256")
            if not isinstance(expected_hash, str) or len(expected_hash) != 64:
                raise ValueError(
                    f"{directory}/frames.jsonl:{line_number}: invalid image hash."
                )
            try:
                int(expected_hash, 16)
            except ValueError as error:
                raise ValueError(
                    f"{directory}/frames.jsonl:{line_number}: image hash "
                    "must be hexadecimal."
                ) from error
            if verify_images:
                actual_hash = hashlib.sha256(image_path.read_bytes()).hexdigest()
                if actual_hash != expected_hash:
                    raise ValueError(f"Image hash mismatch for {image_path}.")

            sample_id = f"{recording_id}/frame_{sequence:08d}"
            if sample_id in seen_sample_ids:
                raise ValueError(f"Duplicate sample_id {sample_id!r}.")
            seen_sample_ids.add(sample_id)
            frame_count += 1
            output.append(
                {
                    "schema_version": 1,
                    "dataset_version": dataset_version,
                    "sample_id": sample_id,
                    "recording_id": recording_id,
                    "frame_sequence": sequence,
                    "timestamp_ns": timestamp_ns,
                    "image_path": _relative_to_root(image_path, dataset_root),
                    "image_sha256": expected_hash,
                    "image_coordinate_system": "undistorted_pixel",
                    "intrinsics_fingerprint_sha256": intrinsics_fingerprint,
                    "valid_pixel_ratio": float(valid_pixel_ratio),
                    "annotation_manifest": _relative_to_root(
                        annotation_path,
                        dataset_root,
                    ),
                    "tags": {name: tags[name] for name in sorted(REQUIRED_TAGS)},
                }
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
    parser.add_argument("--dataset-version", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-image-verification",
        action="store_true",
        help="Skip SHA-256 verification (faster but unsafe for release manifests).",
    )
    args = parser.parse_args()

    records = build_dataset_records(
        args.recordings,
        dataset_root=args.dataset_root,
        dataset_version=args.dataset_version,
        verify_images=not args.skip_image_verification,
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
