"""离线检查中心十字绝对位姿观测，不访问相机、串口或 Hailo。"""

from __future__ import annotations

import argparse
from collections.abc import Iterator
from contextlib import contextmanager
import json
import math
from pathlib import Path
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import CenterCrossPoseObservation, FieldPose2D
from rescue_vision.perception import FieldFeatureDetectionResult


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


@contextmanager
def _frames(path: Path) -> Iterator[Iterator[tuple[int, np.ndarray]]]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        yield iter(((0, image),))
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    def video_frames() -> Iterator[tuple[int, np.ndarray]]:
        sequence = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield sequence, image
            sequence += 1
        if sequence == 0:
            raise ValueError(f"Video contains no decodable frames: {path}")

    try:
        yield video_frames()
    finally:
        capture.release()


def _point(point) -> list[float]:
    return [point.x, point.y]


def _pixel(point) -> tuple[int, int]:
    return round(point.u), round(point.v)


def _pose_record(pose: FieldPose2D) -> dict[str, object]:
    return {
        "position_field_mm": _point(pose.position),
        "heading_rad": pose.heading_rad,
        "heading_deg": math.degrees(pose.heading_rad),
    }


def _localization_record(
    observation: CenterCrossPoseObservation,
) -> dict[str, object]:
    return {
        "frame_sequence": observation.frame_sequence,
        "capture_timestamp_ns": observation.capture_timestamp_ns,
        "result_timestamp_ns": observation.result_timestamp_ns,
        "selected_pose": (
            _pose_record(observation.selected_pose)
            if observation.selected_pose is not None
            else None
        ),
        "selection_source": (
            observation.selection_source.value
            if observation.selection_source is not None
            else None
        ),
        "confidence": observation.confidence,
        "quality": sorted(item.value for item in observation.quality),
        "candidates": [
            {
                **_pose_record(candidate.pose),
                "quarter_turn_index": candidate.quarter_turn_index,
                "position_uncertainty_mm": candidate.position_uncertainty_mm,
                "heading_uncertainty_deg": math.degrees(
                    candidate.heading_uncertainty_rad
                ),
            }
            for candidate in observation.candidates
        ],
        "terminals": [
            {
                "direction_robot": [
                    terminal.direction_forward,
                    terminal.direction_left,
                ],
                "kind": terminal.kind.value,
                "distance_mm": terminal.distance_mm,
                "confidence": terminal.confidence,
            }
            for terminal in observation.terminals
        ],
    }


def _feature_record(result: FieldFeatureDetectionResult) -> dict[str, object]:
    cross = result.center_cross
    return {
        "center_cross": (
            {
                "axis_count": len(cross.axes),
                "intersection_ground_mm": (
                    _point(cross.intersection_ground)
                    if cross.intersection_ground is not None
                    else None
                ),
                "confidence": cross.confidence,
                "quality": sorted(item.value for item in cross.quality),
            }
            if cross is not None
            else None
        ),
        "safe_zones": [
            {
                "physical_color": zone.physical_color.value,
                "confidence": zone.confidence,
                "quality": sorted(item.value for item in zone.quality),
            }
            for zone in result.safe_zones
        ],
        "boundary_feature_count": len(result.boundary_features),
    }


def _draw_polygon(
    image: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    if len(points) >= 3:
        cv2.polylines(
            image,
            [np.asarray(points, dtype=np.int32)],
            True,
            color,
            thickness,
            cv2.LINE_AA,
        )


def _status_lines(observation: CenterCrossPoseObservation) -> list[str]:
    if observation.selected_pose is not None:
        pose = observation.selected_pose
        source = observation.selection_source
        return [
            (
                f"SELECTED x={pose.position.x:.1f} y={pose.position.y:.1f} mm "
                f"heading={math.degrees(pose.heading_rad):.1f} deg"
            ),
            (
                f"source={source.value if source is not None else 'none'} "
                f"confidence={observation.confidence:.2f}"
            ),
        ]
    headings = ", ".join(
        f"{math.degrees(candidate.pose.heading_rad):.1f}"
        for candidate in observation.candidates
    )
    quality = ",".join(sorted(item.value for item in observation.quality))
    return [
        f"NO UNIQUE POSE candidates_deg=[{headings}]",
        f"quality={quality or 'none'}",
    ]


def _put_status(
    image: np.ndarray,
    observation: CenterCrossPoseObservation,
) -> None:
    lines = _status_lines(observation)
    terminal_text = ", ".join(
        (
            f"{item.kind.value}:{item.distance_mm:.0f}mm"
            if item.distance_mm is not None
            else item.kind.value
        )
        for item in observation.terminals
        if item.kind.value != "unknown"
    )
    if terminal_text:
        lines.append(f"terminals={terminal_text}")
    for index, line in enumerate(lines):
        origin = (12, 28 + index * 25)
        cv2.putText(
            image, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (0, 0, 0), 4, cv2.LINE_AA,
        )
        cv2.putText(
            image, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
            0.58, (255, 255, 255), 1, cv2.LINE_AA,
        )


def _undistorted_overlay(
    image: np.ndarray,
    result: FieldFeatureDetectionResult,
    observation: CenterCrossPoseObservation,
) -> np.ndarray:
    output = image.copy()
    for zone in result.safe_zones:
        color = (0, 0, 255) if zone.physical_color.value == "red" else (255, 0, 0)
        _draw_polygon(
            output,
            [_pixel(point) for point in zone.polygon_undistorted],
            color,
            3,
        )
    cross = result.center_cross
    if cross is not None:
        for axis in cross.axes:
            cv2.line(
                output,
                _pixel(axis.start_undistorted),
                _pixel(axis.end_undistorted),
                (0, 220, 0),
                3,
                cv2.LINE_AA,
            )
        if cross.intersection_undistorted is not None:
            cv2.circle(
                output,
                _pixel(cross.intersection_undistorted),
                7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
    for feature in result.boundary_features:
        points = [_pixel(point) for point in feature.points_undistorted]
        if len(points) == 1:
            cv2.circle(output, points[0], 6, (0, 128, 255), 2, cv2.LINE_AA)
        else:
            cv2.line(output, points[0], points[1], (0, 128, 255), 1, cv2.LINE_AA)
    _put_status(output, observation)
    return output


def _ground_pixels(
    projector: GroundProjector,
    points,
) -> list[tuple[int, int]]:
    return [
        (round(pixel.u), round(pixel.v))
        for pixel in projector.ground_to_bev_pixels(points)
    ]


def _bev_overlay(
    image: np.ndarray,
    projector: GroundProjector,
    result: FieldFeatureDetectionResult,
    observation: CenterCrossPoseObservation,
) -> np.ndarray:
    output = projector.make_bev_image(image)
    for zone in result.safe_zones:
        if zone.polygon_ground is None:
            continue
        color = (0, 0, 255) if zone.physical_color.value == "red" else (255, 0, 0)
        _draw_polygon(
            output,
            _ground_pixels(projector, zone.polygon_ground),
            color,
            3,
        )
    cross = result.center_cross
    if cross is not None:
        for axis in cross.axes:
            if axis.start_ground is None or axis.end_ground is None:
                continue
            pixels = _ground_pixels(
                projector,
                (axis.start_ground, axis.end_ground),
            )
            cv2.line(output, pixels[0], pixels[1], (0, 220, 0), 3, cv2.LINE_AA)
        if cross.intersection_ground is not None:
            center = cross.intersection_ground
            center_pixel = _ground_pixels(projector, (center,))[0]
            cv2.circle(output, center_pixel, 7, (0, 255, 255), 2, cv2.LINE_AA)
            for terminal in observation.terminals:
                if terminal.distance_mm is None:
                    continue
                endpoint = GroundPoint(
                    center.x + terminal.direction_forward * terminal.distance_mm,
                    center.y + terminal.direction_left * terminal.distance_mm,
                )
                endpoint_pixel = _ground_pixels(projector, (endpoint,))[0]
                cv2.line(
                    output, center_pixel, endpoint_pixel,
                    (255, 255, 0), 2, cv2.LINE_AA,
                )
    _put_status(output, observation)
    return output


def _prior_pose(values: list[float] | None) -> FieldPose2D | None:
    if values is None:
        return None
    x_mm, y_mm, heading_deg = values
    return FieldPose2D(FieldPoint(x_mm, y_mm), math.radians(heading_deg))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run center-cross localization on an image or video."
    )
    parser.add_argument("input", type=Path, help="图片或视频路径")
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--already-undistorted", action="store_true")
    parser.add_argument(
        "--prior-field-pose",
        type=float,
        nargs=3,
        metavar=("X_MM", "Y_MM", "HEADING_DEG"),
        help=(
            "每帧使用同一个已知先验；运动车辆视频不能用它冒充"
            "连续推算。"
        ),
    )
    args = parser.parse_args()
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive.")

    config = load_runtime_config(args.config)
    geometry = config.build_geometry()
    if geometry is None or geometry.ground_projector is None:
        raise RuntimeError("中心十字定位需要启用可用的地面映射。")
    projector = geometry.ground_projector
    if projector.bev_config is None:
        raise RuntimeError("中心十字手动检查需要地面映射包含 BEV 配置。")
    detector = config.perception.build_field_feature_detector(
        static_map=config.world.static_map,
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=projector,
    )
    localizer = config.build_center_cross_localizer(ground_projector=projector)
    if detector is None or localizer is None:
        raise RuntimeError(
            "需要同时启用 perception.field_features、localization 和地面映射。"
        )
    prior_pose = _prior_pose(args.prior_field_pose)
    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)

    processed = 0
    selected = 0
    try:
        with _frames(args.input) as frames, args.output_jsonl.open(
            "w", encoding="utf-8"
        ) as output:
            for sequence, input_image in frames:
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                actual_size = (input_image.shape[1], input_image.shape[0])
                if actual_size != config.camera.image_size:
                    raise ValueError(
                        f"Input frame {sequence} size {actual_size} does not "
                        f"match runtime camera {config.camera.image_size}."
                    )
                image = (
                    input_image
                    if args.already_undistorted
                    else geometry.camera_model.undistort_image(input_image)
                )
                capture_timestamp_ns = monotonic_ns()
                features = detector.detect(
                    CameraFrame(sequence, capture_timestamp_ns, image),
                    image,
                    valid_mask=geometry.camera_model.valid_mask,
                )
                observation = localizer.localize(features, prior_pose=prior_pose)
                output.write(
                    json.dumps(
                        {
                            "field_features": _feature_record(features),
                            "localization": _localization_record(observation),
                        },
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )
                undistorted = _undistorted_overlay(image, features, observation)
                bev = _bev_overlay(image, projector, features, observation)
                if args.overlay_dir is not None:
                    paths = (
                        args.overlay_dir / f"{sequence:06d}_undistorted.jpg",
                        args.overlay_dir / f"{sequence:06d}_bev.jpg",
                    )
                    for path, overlay in zip(paths, (undistorted, bev)):
                        if not cv2.imwrite(str(path), overlay):
                            raise RuntimeError(f"Failed to write overlay: {path}")
                if args.display:
                    cv2.imshow("center cross - undistorted", undistorted)
                    cv2.imshow("center cross - BEV", bev)
                    if cv2.waitKey(1) & 0xFF in {27, ord("q"), ord("Q")}:
                        processed += 1
                        selected += int(observation.selected_pose is not None)
                        break
                processed += 1
                selected += int(observation.selected_pose is not None)
    finally:
        if args.display:
            cv2.destroyAllWindows()
    print(
        f"Processed {processed} frames; unique pose selected for {selected}; "
        f"wrote {args.output_jsonl}."
    )


if __name__ == "__main__":
    main()
