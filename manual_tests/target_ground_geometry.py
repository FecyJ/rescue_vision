"""从真实相机帧检查目标地面几何估计与可视化结果。"""

from __future__ import annotations

import argparse
import json
import math
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.picamera2_source import Picamera2Source
from rescue_vision.camera.rpicam_source import RpicamSource
from rescue_vision.camera.viewer import OpenCvFrameViewer
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.perception import (
    TargetClass,
    TargetGroundGeometry,
    TargetObservation,
    TargetPoseDetector,
)


_TARGET_COLORS: dict[TargetClass, tuple[int, int, int]] = {
    TargetClass.GREEN_SUPPLY: (0, 200, 0),
    TargetClass.BLACK_CORE: (80, 80, 80),
    TargetClass.ORANGE_INJURED: (0, 128, 255),
    TargetClass.BLUE_DANGER: (255, 200, 0),
}
_FOOTPRINT_COLOR = (255, 0, 255)
_CENTER_COLOR = (0, 255, 255)
_K0_COLOR = (0, 0, 255)


def _target_color(target_class: TargetClass) -> tuple[int, int, int]:
    # 四类任务标签是全部取值，不需要未知类兜底色。
    return _TARGET_COLORS[target_class]


def _draw_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    color: tuple[int, int, int],
    *,
    scale: float = 0.55,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        2,
        cv2.LINE_AA,
    )


def _draw_observation_mask(
    preview: np.ndarray,
    observation: TargetObservation,
) -> None:
    segmentation = observation.color_segmentation
    roi_box = segmentation.roi_box
    x_min = max(0, int(math.floor(roi_box.x_min)))
    y_min = max(0, int(math.floor(roi_box.y_min)))
    x_max = min(preview.shape[1], int(math.ceil(roi_box.x_max)))
    y_max = min(preview.shape[0], int(math.ceil(roi_box.y_max)))
    if x_max <= x_min or y_max <= y_min:
        return

    roi = preview[y_min:y_max, x_min:x_max]
    mask = segmentation.mask
    height = min(roi.shape[0], mask.shape[0])
    width = min(roi.shape[1], mask.shape[1])
    if height <= 0 or width <= 0:
        return
    selected = mask[:height, :width] != 0
    if not np.any(selected):
        return

    color = np.asarray(
        _target_color(segmentation.candidate_class),
        dtype=np.float32,
    )
    selected_roi = roi[:height, :width]
    selected_roi[selected] = (
        selected_roi[selected].astype(np.float32) * 0.55
        + color * 0.45
    ).astype(np.uint8)


def _project_geometry_to_image(
    projector: GroundProjector,
    estimate: TargetGroundGeometry,
) -> tuple[tuple[int, int], np.ndarray] | None:
    if estimate.center_ground is None or not estimate.footprint_ground:
        return None
    try:
        center_pixel = projector.ground_to_pixel(estimate.center_ground)
        footprint_pixels = projector.ground_to_pixels(
            estimate.footprint_ground
        )
    except ValueError:
        return None
    coordinates = [center_pixel.u, center_pixel.v]
    coordinates.extend(
        coordinate
        for point in footprint_pixels
        for coordinate in (point.u, point.v)
    )
    if not all(math.isfinite(value) for value in coordinates):
        return None

    center = (round(center_pixel.u), round(center_pixel.v))
    footprint = np.asarray(
        [(round(point.u), round(point.v)) for point in footprint_pixels],
        dtype=np.int32,
    ).reshape(-1, 1, 2)
    if len(footprint) < 3:
        return None
    return center, footprint


def _draw_target(
    preview: np.ndarray,
    projector: GroundProjector,
    observation: TargetObservation,
    estimate: TargetGroundGeometry | None,
    *,
    geometry_stale: bool,
) -> None:
    _draw_observation_mask(preview, observation)

    box = observation.box
    box_color = _target_color(observation.target_class)
    box_start = (round(box.x_min), round(box.y_min))
    box_end = (round(box.x_max), round(box.y_max))
    cv2.rectangle(preview, box_start, box_end, box_color, 2)
    if observation.k0 is not None:
        cv2.circle(
            preview,
            (round(observation.k0.u), round(observation.k0.v)),
            6,
            _K0_COLOR,
            -1,
        )

    label_x = max(0, box_start[0])
    label_y = max(20, box_start[1] - 8)
    _draw_text(
        preview,
        (
            f"{observation.target_class.value} "
            f"conf={observation.detection_confidence:.2f} "
            f"hsv={observation.color_segmentation.status.value}"
        ),
        (label_x, label_y),
        box_color,
    )

    if geometry_stale:
        _draw_text(
            preview,
            "ground geometry: STALE dropped",
            (label_x, label_y + 22),
            (0, 0, 255),
        )
        return
    if estimate is None:
        return

    if estimate.center_ground is None:
        quality = ",".join(
            item.value
            for item in sorted(estimate.quality, key=lambda item: item.value)
        )
        _draw_text(
            preview,
            f"geometry unavailable: {quality or 'unknown'}",
            (label_x, label_y + 22),
            (0, 0, 255),
        )
        return

    projected = _project_geometry_to_image(projector, estimate)
    if projected is None:
        _draw_text(
            preview,
            "geometry projection failed",
            (label_x, label_y + 22),
            (0, 0, 255),
        )
        return

    center_pixel, footprint = projected
    cv2.polylines(
        preview,
        [footprint],
        True,
        _FOOTPRINT_COLOR,
        3,
        cv2.LINE_AA,
    )
    cv2.circle(preview, center_pixel, 8, _CENTER_COLOR, -1)
    for point in footprint.reshape(-1, 2):
        cv2.circle(
            preview,
            (int(point[0]), int(point[1])),
            4,
            _FOOTPRINT_COLOR,
            -1,
        )

    center = estimate.center_ground
    assert center is not None
    _draw_text(
        preview,
        (
            f"center=({center.x:.0f},{center.y:.0f})mm "
            f"score={estimate.fit_score or 0.0:.2f} "
            f"unc={estimate.center_uncertainty_mm or 0.0:.1f}mm"
        ),
        (label_x, min(preview.shape[0] - 8, box_end[1] + 22)),
        _CENTER_COLOR,
    )
    _draw_text(
        preview,
        f"footprint={len(estimate.footprint_ground)} points",
        (label_x, min(preview.shape[0] - 8, box_end[1] + 44)),
        _FOOTPRINT_COLOR,
    )


def _print_results(
    frame_sequence: int,
    observations: tuple[TargetObservation, ...],
    estimates: tuple[TargetGroundGeometry, ...],
    *,
    detection_stale_age_ms: float | None,
    geometry_stale_age_ms: float | None,
) -> None:
    if detection_stale_age_ms is not None:
        payload = {
            "frame_sequence": frame_sequence,
            "status": "pose_stale_dropped",
            "age_ms": detection_stale_age_ms,
        }
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
        return
    if geometry_stale_age_ms is not None:
        payload = {
            "frame_sequence": frame_sequence,
            "status": "ground_geometry_stale_dropped",
            "age_ms": geometry_stale_age_ms,
        }
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
        return
    if not observations:
        payload = {"frame_sequence": frame_sequence, "status": "no_target"}
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)
        return

    for observation, estimate in zip(observations, estimates, strict=True):
        payload = {
            "frame_sequence": frame_sequence,
            "target_class": observation.target_class.value,
            "center_ground_mm": (
                [estimate.center_ground.x, estimate.center_ground.y]
                if estimate.center_ground is not None
                else None
            ),
            "footprint_ground_mm": [
                [point.x, point.y] for point in estimate.footprint_ground
            ],
            "yaw_rad": estimate.yaw_rad,
            "center_uncertainty_mm": estimate.center_uncertainty_mm,
            "fit_score": estimate.fit_score,
            "silhouette_iou": estimate.silhouette_iou,
            "contact_residual_px": estimate.contact_residual_px,
            "method": estimate.method.value,
            "quality": sorted(item.value for item in estimate.quality),
        }
        print(json.dumps(payload, ensure_ascii=False, allow_nan=False), flush=True)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "使用真实相机和 Hailo Pose 检查 TargetGroundGeometryEstimator；"
            "Q/Esc 退出。"
        )
    )
    parser.add_argument("--config", default="configs/runtime.yaml")
    parser.add_argument(
        "--frames",
        type=int,
        help="处理指定帧数后退出；默认持续运行到按 Q/Esc。",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        help="终端 JSONL 输出间隔，单位秒；默认 0.5。",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.frames is not None and args.frames <= 0:
        parser.error("--frames must be positive")
    if not math.isfinite(args.print_interval) or args.print_interval <= 0.0:
        parser.error("--print-interval must be positive and finite")

    config = load_runtime_config(args.config)
    geometry = config.build_geometry()
    if geometry is None:
        raise RuntimeError(
            "geometry.intrinsics_enabled must be true for live perception."
        )
    projector = geometry.ground_projector
    if projector is None or not projector.supports_robot_projection:
        raise RuntimeError(
            "A valid ground mapping with full camera extrinsics is required."
        )

    estimator = config.perception.build_target_ground_geometry_estimator(
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=projector,
    )
    if estimator is None:
        raise RuntimeError(
            "perception.target_ground_geometry.enabled must be true."
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

    backend = config.hailo.build_backend()
    if backend is None:
        raise RuntimeError("hailo.enabled must be true for live perception.")

    detector = TargetPoseDetector(
        backend=backend,
        detection_threshold=config.perception.detection_threshold,
        k0_threshold=config.perception.k0_threshold,
        color_classifier=config.perception.color_classifier,
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=projector,
        center_cross_refinement=config.perception.center_cross_refinement,
        safe_zone_color=config.perception.safe_zone_color,
    )

    with (
        detector,
        source,
        OpenCvFrameViewer(
            "target ground geometry — Q/Esc to stop"
        ) as viewer,
    ):
        processed_frames = 0
        last_print_timestamp_ns: int | None = None
        while args.frames is None or processed_frames < args.frames:
            frame = source.read(timeout=1.0)
            undistorted_bgr = geometry.camera_model.undistort_image(
                frame.image_bgr
            )
            detection_result = detector.detect_realtime(
                frame,
                undistorted_bgr,
            )

            geometry_result = None
            if not detection_result.stale_dropped:
                geometry_result = estimator.estimate_realtime(
                    detection_result.observations
                )
            processed_frames += 1

            geometry_stale = (
                geometry_result is not None and geometry_result.stale_dropped
            )
            if geometry_stale:
                estimates_for_draw: tuple[TargetGroundGeometry | None, ...] = (
                    (None,) * len(detection_result.observations)
                )
            elif geometry_result is None:
                estimates_for_draw = ()
            else:
                estimates_for_draw = geometry_result.estimates

            preview = undistorted_bgr.copy()
            for observation, estimate in zip(
                detection_result.observations,
                estimates_for_draw,
                strict=True,
            ):
                _draw_target(
                    preview,
                    projector,
                    observation,
                    estimate,
                    geometry_stale=geometry_stale,
                )

            now_ns = monotonic_ns()
            age_ms = detection_result.timing.capture_to_result_ms
            if detection_result.stale_dropped:
                status = (
                    "POSE STALE dropped: "
                    f"{detection_result.dropped_stale_age_ms:.1f} ms"
                )
                status_color = (0, 0, 255)
            elif geometry_stale:
                assert geometry_result is not None
                status = (
                    "GEOMETRY STALE dropped: "
                    f"{geometry_result.dropped_stale_age_ms:.1f} ms"
                )
                status_color = (0, 0, 255)
            else:
                status = (
                    f"age {age_ms:.1f} ms | "
                    f"targets {len(detection_result.observations)}"
                )
                status_color = (0, 255, 0)
            _draw_text(preview, status, (20, 32), status_color, scale=0.7)

            if (
                last_print_timestamp_ns is None
                or now_ns - last_print_timestamp_ns
                >= args.print_interval * 1_000_000_000
            ):
                _print_results(
                    frame.sequence,
                    detection_result.observations,
                    () if geometry_result is None else geometry_result.estimates,
                    detection_stale_age_ms=(
                        detection_result.dropped_stale_age_ms
                        if detection_result.stale_dropped
                        else None
                    ),
                    geometry_stale_age_ms=(
                        geometry_result.dropped_stale_age_ms
                        if geometry_stale
                        else None
                    ),
                )
                last_print_timestamp_ns = now_ns

            if not viewer.show(preview):
                break


if __name__ == "__main__":
    main()
