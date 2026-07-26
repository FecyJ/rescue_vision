"""按记录组确定性划分数据集，防止相邻视频帧跨集合泄漏。"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from rescue_vision.geometry.camera_model import IMAGE_BORDER_FILL_VALUE

SCHEMA_VERSION = 2
REQUIRED_TAGS = {
    "lighting",
    "distance",
    "occlusion",
    "motion_blur",
    "background",
    "target_pose",
    "contact_state",
}


def _unit_hash(seed: str, group_id: str) -> float:
    digest = hashlib.sha256(f"{seed}\0{group_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / 2**64


def assign_split(
    group_id: str,
    *,
    seed: str,
    train_ratio: float,
    validation_ratio: float,
) -> str:
    value = _unit_hash(seed, group_id)
    if value < train_ratio:
        return "train"
    if value < train_ratio + validation_ratio:
        return "validation"
    return "test"


def split_records(
    records: list[dict[str, Any]],
    *,
    seed: str,
    train_ratio: float = 0.7,
    validation_ratio: float = 0.15,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if not (0 < train_ratio < 1):
        raise ValueError("train_ratio must be in (0, 1).")
    if not (0 <= validation_ratio < 1):
        raise ValueError("validation_ratio must be in [0, 1).")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be less than 1.")

    seen_samples: set[str] = set()
    group_splits: dict[str, str] = {}
    output: list[dict[str, Any]] = []
    split_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    tag_counts: dict[str, Counter[str]] = defaultdict(Counter)
    dataset_version: str | None = None

    for index, record in enumerate(records):
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError(
                f"Record {index} schema_version must be {SCHEMA_VERSION}."
            )
        sample_id = record.get("sample_id")
        recording_id = record.get("recording_id")
        record_dataset_version = record.get("dataset_version")
        image_coordinate_system = record.get("image_coordinate_system")
        intrinsics_fingerprint = record.get(
            "intrinsics_fingerprint_sha256"
        )
        valid_pixel_ratio = record.get("valid_pixel_ratio")
        undistort_fill_value = record.get("undistort_fill_value")
        tags = record.get("tags")
        if (
            not isinstance(record_dataset_version, str)
            or not record_dataset_version
        ):
            raise ValueError(f"Record {index} has invalid dataset_version.")
        if dataset_version is None:
            dataset_version = record_dataset_version
        elif record_dataset_version != dataset_version:
            raise ValueError(
                f"Record {index} dataset_version {record_dataset_version!r} "
                f"does not match {dataset_version!r}."
            )
        if not isinstance(sample_id, str) or not sample_id:
            raise ValueError(f"Record {index} has invalid sample_id.")
        if sample_id in seen_samples:
            raise ValueError(f"Duplicate sample_id {sample_id!r}.")
        seen_samples.add(sample_id)
        if not isinstance(recording_id, str) or not recording_id:
            raise ValueError(f"Record {index} has invalid recording_id.")
        if image_coordinate_system != "undistorted_pixel":
            raise ValueError(
                f"Record {index} image_coordinate_system must be "
                "'undistorted_pixel'."
            )
        if (
            not isinstance(intrinsics_fingerprint, str)
            or len(intrinsics_fingerprint) != 64
            or intrinsics_fingerprint != intrinsics_fingerprint.lower()
            or any(
                character not in "0123456789abcdef"
                for character in intrinsics_fingerprint
            )
        ):
            raise ValueError(
                f"Record {index} has invalid intrinsics fingerprint."
            )
        if (
            isinstance(valid_pixel_ratio, bool)
            or not isinstance(valid_pixel_ratio, (int, float))
            or not 0.0 < float(valid_pixel_ratio) <= 1.0
        ):
            raise ValueError(
                f"Record {index} has invalid valid_pixel_ratio."
            )
        if undistort_fill_value != IMAGE_BORDER_FILL_VALUE:
            raise ValueError(
                f"Record {index} undistort_fill_value must be "
                f"{IMAGE_BORDER_FILL_VALUE}."
            )
        if not isinstance(tags, dict):
            raise ValueError(f"Record {index} tags must be a mapping.")
        missing_tags = sorted(REQUIRED_TAGS - set(tags))
        if missing_tags:
            raise ValueError(
                f"Record {index} is missing stratification tags {missing_tags}."
            )
        if not all(isinstance(tags[key], str) and tags[key] for key in REQUIRED_TAGS):
            raise ValueError(f"Record {index} tags must be non-empty strings.")

        split = group_splits.setdefault(
            recording_id,
            assign_split(
                recording_id,
                seed=seed,
                train_ratio=train_ratio,
                validation_ratio=validation_ratio,
            ),
        )
        enriched = dict(record)
        enriched["split"] = split
        output.append(enriched)
        split_counts[split] += 1
        for tag_name in sorted(REQUIRED_TAGS):
            tag_counts[f"{split}:{tag_name}"][str(tags[tag_name])] += 1

    for split in group_splits.values():
        group_counts[split] += 1

    split_names = ("train", "validation", "test")
    sample_counts = {
        split: int(split_counts[split]) for split in split_names
    }
    explicit_group_counts = {
        split: int(group_counts[split]) for split in split_names
    }
    total_samples = len(output)
    realized_ratios = {
        split: (
            sample_counts[split] / total_samples
            if total_samples
            else 0.0
        )
        for split in split_names
    }
    warnings = [
        {
            "code": "empty_split",
            "split": split,
            "message": f"{split} contains no recording groups or samples.",
        }
        for split in split_names
        if explicit_group_counts[split] == 0
    ]
    report = {
        "schema_version": 1,
        "dataset_version": dataset_version,
        "seed": seed,
        "ratios": {
            "train": train_ratio,
            "validation": validation_ratio,
            "test": 1.0 - train_ratio - validation_ratio,
        },
        "realized_ratios": realized_ratios,
        "sample_counts": sample_counts,
        "group_counts": explicit_group_counts,
        "warnings": warnings,
        "tag_distribution": {
            name: dict(sorted(counts.items()))
            for name, counts in sorted(tag_counts.items())
        },
    }
    return output, report


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deterministically split a dataset by recording_id."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--seed", default="rescue-vision-v1")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--validation-ratio", type=float, default=0.15)
    args = parser.parse_args()

    records, report = split_records(
        _read_jsonl(args.manifest),
        seed=args.seed,
        train_ratio=args.train_ratio,
        validation_ratio=args.validation_ratio,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    if report["warnings"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
