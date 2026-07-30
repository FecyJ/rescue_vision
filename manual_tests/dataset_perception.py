"""按数据清单批量检查实际 Hailo Pose 输出，不绕过样本身份。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic_ns

import cv2

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.config import load_runtime_config
from rescue_vision.perception import TargetPoseDetector


def _read_manifest(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON.") from error
        if (
            not isinstance(record, dict)
            or record.get("schema_version") != 2
            or not isinstance(record.get("sample_id"), str)
            or not isinstance(record.get("image_path"), str)
        ):
            raise ValueError(f"{path}:{line_number}: invalid manifest record.")
        records.append(record)
    if not records:
        raise ValueError(f"Manifest contains no samples: {path}")
    return records


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the configured Hailo detector over a schema-v2 manifest."
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.yaml"))
    args = parser.parse_args()

    records = _read_manifest(args.manifest)
    config = load_runtime_config(args.config)
    geometry = config.build_geometry()
    if geometry is None:
        raise RuntimeError("Target dataset inference requires enabled intrinsics.")
    backend = config.hailo.build_backend()
    if backend is None:
        raise RuntimeError("hailo.enabled must be true for this manual check.")

    detector = TargetPoseDetector(
        backend,
        class_mapping=config.hailo.model_class_mapping(),
        detection_threshold=config.perception.detection_threshold,
        k0_threshold=config.perception.k0_threshold,
        color_classifier=config.perception.color_classifier,
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=geometry.ground_projector,
    )
    output_records: list[dict[str, object]] = []
    try:
        for record in records:
            sample_id = str(record["sample_id"])
            image_path = args.dataset_root / str(record["image_path"])
            image = cv2.imread(str(image_path))
            if image is None:
                raise ValueError(f"Cannot decode manifest image: {image_path}")
            actual_size = (int(image.shape[1]), int(image.shape[0]))
            if actual_size != config.camera.image_size:
                raise ValueError(
                    f"Image {image_path} size {actual_size} does not match "
                    f"runtime camera {config.camera.image_size}."
                )
            timestamp_ns = monotonic_ns()
            frame = CameraFrame(
                int(record.get("frame_sequence", 0)), timestamp_ns, image
            )
            observations = detector.detect(frame, image)
            for index, observation in enumerate(observations):
                output_records.append(
                    {
                        "schema_version": 2,
                        "sample_id": sample_id,
                        "observation_id": f"observation_{index:04d}",
                        "model_target_class": observation.model_target_class.value,
                        "target_class": observation.target_class.value,
                        "detection_confidence": observation.detection_confidence,
                        "class_probabilities": observation.class_probabilities.as_dict(),
                        "box_undistorted": [
                            observation.box.x_min,
                            observation.box.y_min,
                            observation.box.x_max,
                            observation.box.y_max,
                        ],
                        "color_segmentation": {
                            "candidate_class": (
                                observation.color_segmentation.candidate_class.value
                            ),
                            "status": observation.color_segmentation.status.value,
                            "roi_box_undistorted": [
                                observation.color_segmentation.roi_box.x_min,
                                observation.color_segmentation.roi_box.y_min,
                                observation.color_segmentation.roi_box.x_max,
                                observation.color_segmentation.roi_box.y_max,
                            ],
                            "color_fraction": (
                                observation.color_segmentation.color_fraction
                            ),
                            "dominance": observation.color_segmentation.dominance,
                        },
                        "k0_undistorted": (
                            [observation.k0.u, observation.k0.v]
                            if observation.k0 is not None
                            else None
                        ),
                        "k0_confidence": observation.k0_confidence,
                        "ground_mm": (
                            [
                                observation.ground_point.x,
                                observation.ground_point.y,
                            ]
                            if observation.ground_point is not None
                            else None
                        ),
                        "capture_timestamp_ns": observation.capture_timestamp_ns,
                        "result_timestamp_ns": observation.result_timestamp_ns,
                        "quality": sorted(
                            item.value for item in observation.quality
                        ),
                        "model_version": observation.model_version,
                        "model_sha256": observation.model_sha256,
                    }
                )
    finally:
        detector.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"
            for record in output_records
        ),
        encoding="utf-8",
    )
    print(
        f"Processed {len(records)} samples; wrote {len(output_records)} "
        f"observations to {args.output}."
    )


if __name__ == "__main__":
    main()
