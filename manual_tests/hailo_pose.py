"""使用实际 runtime.yaml 和部署包检查单张去畸变图像的 Hailo Pose 输出。"""

from __future__ import annotations

import argparse
import json
from time import monotonic_ns

import cv2

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.config import load_runtime_config
from rescue_vision.perception import TargetPoseDetector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument(
        "--undistorted-image",
        required=True,
        help="已经按当前 CameraModel 去畸变、且几何预处理与训练一致的 BGR 图像。",
    )
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    backend = config.hailo.build_backend()
    if backend is None:
        raise RuntimeError("hailo.enabled must be true for this manual check.")

    image = cv2.imread(args.undistorted_image)
    if image is None:
        backend.close()
        raise FileNotFoundError(f"Cannot read image: {args.undistorted_image}")
    expected_size = config.camera.image_size
    actual_size = (int(image.shape[1]), int(image.shape[0]))
    if actual_size != expected_size:
        backend.close()
        raise ValueError(
            f"Image size {actual_size} does not match runtime camera {expected_size}."
        )

    geometry = config.build_geometry()
    projector = geometry.ground_projector if geometry is not None else None
    detector = TargetPoseDetector(
        backend,
        detection_threshold=config.perception.detection_threshold,
        k0_threshold=config.perception.k0_threshold,
        color_classifier=config.perception.color_classifier,
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=projector,
        center_cross_refinement=config.perception.center_cross_refinement,
        safe_zone_color=config.perception.safe_zone_color,
    )
    capture_timestamp_ns = monotonic_ns()
    frame = CameraFrame(0, capture_timestamp_ns, image)
    try:
        result = detector.detect(frame, image)
        for observation in result.observations:
            print(
                json.dumps(
                    {
                        "frame_sequence": observation.frame_sequence,
                        "capture_timestamp_ns": observation.capture_timestamp_ns,
                        "result_timestamp_ns": observation.result_timestamp_ns,
                        "model_target_class": (
                            observation.model_target_class.value
                        ),
                        "target_class": observation.target_class.value,
                        "class_probabilities": (
                            observation.class_probabilities.as_dict()
                        ),
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
                        "quality": sorted(item.value for item in observation.quality),
                    },
                    ensure_ascii=False,
                )
            )
        print(
            json.dumps(
                {
                    "field_features": {
                        "center_cross": result.field_features.center_cross is not None,
                        "safe_zone_count": len(result.field_features.safe_zones),
                    }
                },
                ensure_ascii=False,
            )
        )
    finally:
        detector.close()


if __name__ == "__main__":
    main()
