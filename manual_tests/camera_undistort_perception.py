"""按 runtime.yaml 实时预览去畸变后的任务目标和场地特征 Pose 观测。"""

from __future__ import annotations

import argparse
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.camera.viewer import OpenCvFrameViewer
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    FieldFeatureDetectionResult,
    FieldPoseKeypoint,
    TargetPoseDetector,
    render_target_observations,
)


def _format_pixel(point: UndistortedPixel | None) -> str:
    if point is None:
        return "-"
    return f"(u={point.u:.1f},v={point.v:.1f})"


def _format_ground(point: GroundPoint | None) -> str:
    if point is None:
        return "-"
    return f"(x={point.x:.1f},y={point.y:.1f})mm"


def _format_keypoint(keypoint: FieldPoseKeypoint) -> str:
    return (
        f"pixel={_format_pixel(keypoint.undistorted)} "
        f"ground={_format_ground(keypoint.ground)}"
    )


def _format_field_feature_coordinates(
    field_features: FieldFeatureDetectionResult | None,
) -> str:
    """Format model field-feature coordinates without introducing new geometry."""

    if field_features is None:
        return "center_cross_k0=- safe_zone_keypoints=-"

    center_cross = field_features.center_cross
    center_text = (
        "-"
        if center_cross is None
        else _format_keypoint(center_cross.intersection)
    )
    safe_zone_text = "-"
    if field_features.safe_zones:
        safe_zone_text = "|".join(
            (
                f"{zone.physical_color.value}:"
                f"K0[{_format_keypoint(zone.ground_anchor)}] "
                f"K1[{_format_keypoint(zone.image_left_landmark)}] "
                f"K2[{_format_keypoint(zone.image_right_landmark)}]"
            )
            for zone in field_features.safe_zones
        )
    return (
        f"center_cross_k0[{center_text}] "
        f"safe_zone_keypoints[{safe_zone_text}]"
    )


def _draw_field_feature_coordinates(
    preview: np.ndarray,
    field_features: FieldFeatureDetectionResult | None,
) -> None:
    """Overlay field-feature ground coordinates on the live preview."""

    if field_features is None:
        return
    if (
        not isinstance(preview, np.ndarray)
        or preview.ndim != 3
        or preview.shape[2] != 3
        or preview.dtype != np.uint8
    ):
        raise TypeError("preview must be a uint8 OpenCV image array.")

    image = preview
    height = int(image.shape[0])
    cross = field_features.center_cross
    if cross is not None:
        point = cross.intersection.undistorted
        anchor = (
            (round(point.u), round(point.v))
            if point is not None
            else (round(cross.box.x_min), round(cross.box.y_min))
        )
        ground = _format_ground(cross.intersection.ground)
        cv2.putText(
            image,
            f"center K0 {ground}",
            (anchor[0], max(20, anchor[1] - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

    for zone_index, zone in enumerate(field_features.safe_zones):
        keypoints = (
            zone.ground_anchor,
            zone.image_left_landmark,
            zone.image_right_landmark,
        )
        for keypoint_index, keypoint in enumerate(keypoints):
            point = keypoint.undistorted
            if point is None:
                continue
            cv2.putText(
                image,
                f"{zone.physical_color.value} K{keypoint_index} "
                f"{_format_ground(keypoint.ground)}",
                (
                    round(point.u) + 8,
                    min(
                        height - 8,
                        round(point.v) + 20 + zone_index * 60,
                    ),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 0, 255),
                2,
                cv2.LINE_AA,
            )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument(
        "--frames",
        type=int,
        help="处理指定帧数后退出；默认持续运行到按 Q/Esc。",
    )
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")

    config = load_runtime_config(args.config)
    geometry = config.build_geometry()
    if geometry is None:
        raise RuntimeError(
            "geometry.intrinsics_enabled must be true for live perception."
        )

    backend = config.hailo.build_backend()
    if backend is None:
        raise RuntimeError(
            "hailo.enabled must be true for live perception."
        )

    source_class = (
        Picamera2Source
        if config.camera.backend == "picamera2"
        else RpicamSource
    )
    source = source_class(
        image_size=config.camera.image_size,
        fps=config.camera.fps,
        lens_position=config.camera.lens_position,
    )

    # 检测器从构造开始接管 backend；参数校验失败时也会先关闭设备。
    detector = TargetPoseDetector(
        backend=backend,
        detection_threshold=config.perception.detection_threshold,
        k0_threshold=config.perception.k0_threshold,
        color_classifier=config.perception.color_classifier,
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=geometry.ground_projector,
        center_cross_refinement=config.perception.center_cross_refinement,
        safe_zone_color=config.perception.safe_zone_color,
    )

    with (
        source,
        detector,
        OpenCvFrameViewer(
            "undistorted perception — Q/Esc to stop"
        ) as viewer,
    ):
        processed_frames = 0
        while args.frames is None or processed_frames < args.frames:
            frame = source.read(timeout=1.0)
            undistort_started_ns = monotonic_ns()
            undistorted_bgr = geometry.camera_model.undistort_image(
                frame.image_bgr
            )
            undistort_finished_ns = monotonic_ns()
            detection_result = detector.detect_realtime(
                frame,
                undistorted_bgr,
            )
            result_timestamp_ns = monotonic_ns()
            processed_frames += 1

            field_features = detection_result.field_features
            preview = render_target_observations(
                undistorted_bgr,
                detection_result.observations,
                dropped_stale_age_ms=detection_result.dropped_stale_age_ms,
                field_features=field_features,
            )
            _draw_field_feature_coordinates(preview, field_features)
            ground_points: list[str] = []
            for observation in detection_result.observations:
                ground_point = observation.ground_point
                if ground_point is None:
                    continue
                ground_text = (
                    f"x={ground_point.x:.0f} y={ground_point.y:.0f} mm"
                )
                cv2.putText(
                    preview,
                    ground_text,
                    (
                        round(observation.box.x_min),
                        min(
                            preview.shape[0] - 8,
                            round(observation.box.y_max) + 22,
                        ),
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                ground_points.append(
                    f"{observation.target_class.value}=({ground_point.x:.0f},"
                    f"{ground_point.y:.0f})mm"
                )

            if field_features is None:
                center_cross_count = 0
                safe_zone_count = 0
                safe_zone_colors = "-"
            else:
                center_cross_count = int(field_features.center_cross is not None)
                safe_zone_count = len(field_features.safe_zones)
                safe_zone_colors = ",".join(
                    zone.physical_color.value for zone in field_features.safe_zones
                ) or "-"
            field_coordinate_text = _format_field_feature_coordinates(field_features)
            print(
                f"frame={frame.sequence} "
                f"targets={len(detection_result.observations)} "
                f"ground={';'.join(ground_points) or '-'} "
                f"center_cross={center_cross_count} "
                f"safe_zones={safe_zone_count} "
                f"safe_zone_colors={safe_zone_colors}",
                flush=True,
            )
            print(
                "field_coordinates=ground_frame=robot_x_forward_y_left "
                f"{field_coordinate_text}",
                flush=True,
            )

            age_ms = (result_timestamp_ns - frame.timestamp_ns) / 1_000_000.0
            undistort_ms = (
                undistort_finished_ns - undistort_started_ns
            ) / 1_000_000.0
            status = (
                "STALE dropped: "
                f"{detection_result.dropped_stale_age_ms:.1f} ms"
                if detection_result.stale_dropped
                else f"age {age_ms:.1f} ms"
            )
            color = (
                (0, 0, 255)
                if detection_result.stale_dropped
                else (0, 255, 0)
            )
            cv2.putText(
                preview,
                f"{status} | undistort {undistort_ms:.1f} ms",
                (20, 32),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
                cv2.LINE_AA,
            )
            if not viewer.show(preview):
                break


if __name__ == "__main__":
    main()
