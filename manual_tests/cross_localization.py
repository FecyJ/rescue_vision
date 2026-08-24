"""检测中心十字和安全区，显示 BEV 叠加并检查绝对定位，不访问串口或 Hailo。"""

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
def _frames(path: Path) -> Iterator[Iterator[CameraFrame]]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        yield iter((CameraFrame(0, monotonic_ns(), image),))
        return

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")

    def video_frames() -> Iterator[CameraFrame]:
        sequence = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield CameraFrame(sequence, monotonic_ns(), image)
            sequence += 1
        if sequence == 0:
            raise ValueError(f"Video contains no decodable frames: {path}")

    try:
        yield video_frames()
    finally:
        capture.release()


def _build_camera_source(config):
    """Create the configured source without opening camera hardware."""

    from rescue_vision.camera.picamera2_source import Picamera2Source
    from rescue_vision.camera.rpicam_source import RpicamSource

    source_class = (
        Picamera2Source
        if config.camera.backend == "picamera2"
        else RpicamSource
    )
    return source_class(
        image_size=config.camera.image_size,
        fps=config.camera.fps,
        lens_position=config.camera.lens_position,
    )


@contextmanager
def _camera_frames(
    config,
    *,
    source_factory=_build_camera_source,
) -> Iterator[Iterator[CameraFrame]]:
    """Open the configured latest-frame source and close it on every exit."""

    source = source_factory(config)

    def latest_frames() -> Iterator[CameraFrame]:
        while True:
            yield source.read(timeout=1.0)

    with source:
        yield latest_frames()


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


def _put_label(
    image: np.ndarray,
    label: str,
    center: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    origin = (center[0] + 5, center[1] - 5)
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (0, 0, 0),
        3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        label,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        color,
        1,
        cv2.LINE_AA,
    )


def _draw_region(
    image: np.ndarray,
    points: list[tuple[int, int]],
    color: tuple[int, int, int],
    label: str,
    thickness: int = 3,
) -> None:
    """Fill and outline one detected region, then put its label at the centroid."""

    if len(points) < 3:
        return
    polygon = np.asarray(points, dtype=np.int32)
    tinted = image.copy()
    cv2.fillPoly(tinted, [polygon], color)
    cv2.addWeighted(tinted, 0.18, image, 0.82, 0.0, dst=image)
    cv2.polylines(
        image,
        [polygon],
        True,
        color,
        thickness,
        cv2.LINE_AA,
    )
    center = tuple(np.rint(np.mean(polygon, axis=0)).astype(int))
    _put_label(image, label, (int(center[0]), int(center[1])), color)


def _ground_centroid(points: tuple[GroundPoint, ...]) -> GroundPoint:
    return GroundPoint(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def _format_safe_zone_evidence(result: FieldFeatureDetectionResult) -> str:
    evidence: list[str] = []
    for zone in result.safe_zones:
        if zone.polygon_ground is None:
            evidence.append(
                f"{zone.physical_color.value}:no_ground"
                f"(conf={zone.confidence:.2f})"
            )
            continue
        center = _ground_centroid(zone.polygon_ground)
        evidence.append(
            f"{zone.physical_color.value}"
            f"@({center.x:.0f},{center.y:.0f})mm"
            f"(conf={zone.confidence:.2f})"
        )
    return ", ".join(evidence) if evidence else "none"


def _print_localization_calculation(
    result: FieldFeatureDetectionResult,
    observation: CenterCrossPoseObservation,
) -> None:
    """Print the actual same-frame evidence and the selected pose calculation."""

    pose = observation.selected_pose
    if pose is None:
        return
    cross = result.center_cross
    cross_text = "none"
    if cross is not None and cross.intersection_ground is not None:
        cross_text = (
            f"({cross.intersection_ground.x:.1f},"
            f" {cross.intersection_ground.y:.1f})mm"
        )
    terminals = [
        (
            f"{terminal.kind.value}@{terminal.distance_mm:.0f}mm"
            if terminal.distance_mm is not None
            else terminal.kind.value
        )
        + (
            f" dir=({terminal.direction_forward:.2f},"
            f"{terminal.direction_left:.2f})"
            f"(conf={terminal.confidence:.2f})"
        )
        for terminal in observation.terminals
        if terminal.kind.value != "unknown"
    ]
    terminal_text = ", ".join(terminals) if terminals else "none"
    source = (
        observation.selection_source.value
        if observation.selection_source is not None
        else "none"
    )
    quality = ",".join(sorted(item.value for item in observation.quality))
    quality_text = quality if quality else "none"
    print(
        f"[定位成功] frame={observation.frame_sequence} "
        f"cross_ground={cross_text}; "
        f"safe_zone_evidence=[{_format_safe_zone_evidence(result)}]; "
        f"terminals=[{terminal_text}]\n"
        f"  field_pose=(x={pose.position.x:.1f}, y={pose.position.y:.1f})mm "
        f"heading={math.degrees(pose.heading_rad):.1f}deg "
        f"source={source} confidence={observation.confidence:.2f} "
        f"quality={quality_text}",
        flush=True,
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


def _put_lines(image: np.ndarray, lines: list[str]) -> None:
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
    _put_lines(image, lines)


def _undistorted_overlay(
    image: np.ndarray,
    result: FieldFeatureDetectionResult,
    observation: CenterCrossPoseObservation,
) -> np.ndarray:
    output = image.copy()
    for zone in result.safe_zones:
        color = (0, 0, 255) if zone.physical_color.value == "red" else (255, 0, 0)
        _draw_region(
            output,
            [_pixel(point) for point in zone.polygon_undistorted],
            color,
            f"SAFE {zone.physical_color.value}",
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
            center_pixel = _pixel(cross.intersection_undistorted)
            cv2.circle(
                output,
                center_pixel,
                7,
                (0, 255, 255),
                2,
                cv2.LINE_AA,
            )
            _put_label(output, "CROSS", center_pixel, (0, 255, 255))
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
        _draw_region(
            output,
            _ground_pixels(projector, zone.polygon_ground),
            color,
            f"SAFE {zone.physical_color.value}",
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
            _put_label(output, "CROSS", center_pixel, (0, 255, 255))
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
        description=(
            "Run center-cross localization on an image, video or configured "
            "live camera."
        )
    )
    parser.add_argument("input", type=Path, nargs="?", help="图片或视频路径")
    parser.add_argument(
        "--camera",
        action="store_true",
        help="按 runtime.yaml 打开最新帧相机源并自动显示实时界面。",
    )
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument(
        "--print-interval",
        type=float,
        default=0.5,
        metavar="SECONDS",
        help=(
            "唯一定位成功时打印一次计算结果的最小间隔；"
            "设为 0 打印每个成功帧（默认 0.5）。"
        ),
    )
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
    if args.camera == (args.input is not None):
        parser.error("provide exactly one of input or --camera.")
    if args.camera and args.already_undistorted:
        parser.error("--already-undistorted is only valid for file input.")
    if args.max_frames is not None and args.max_frames <= 0:
        parser.error("--max-frames must be positive.")
    if not math.isfinite(args.print_interval) or args.print_interval < 0.0:
        parser.error("--print-interval must be finite and non-negative.")
    print_interval_ns_float = args.print_interval * 1_000_000_000
    if not math.isfinite(print_interval_ns_float):
        parser.error("--print-interval is too large.")

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
    last_print_timestamp_ns: int | None = None
    print_interval_ns = round(print_interval_ns_float)
    show_display = args.display or args.camera
    frame_context = (
        _camera_frames(config)
        if args.camera
        else _frames(args.input)
    )
    try:
        if show_display:
            cv2.namedWindow("center cross - undistorted", cv2.WINDOW_NORMAL)
            cv2.namedWindow("center cross - BEV", cv2.WINDOW_NORMAL)
        with frame_context as frames, args.output_jsonl.open(
            "w", encoding="utf-8"
        ) as output:
            for raw_frame in frames:
                if args.max_frames is not None and processed >= args.max_frames:
                    break
                sequence = raw_frame.sequence
                input_image = raw_frame.image_bgr
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
                stale_age_ms: float | None = None
                if args.camera:
                    realtime = detector.detect_realtime(
                        raw_frame,
                        image,
                        valid_mask=geometry.camera_model.valid_mask,
                        include_boundary_features=False,
                    )
                    if realtime.stale_dropped:
                        stale_age_ms = realtime.dropped_stale_age_ms
                        features = None
                    else:
                        features = realtime.result
                else:
                    features = detector.detect(
                        raw_frame,
                        image,
                        valid_mask=geometry.camera_model.valid_mask,
                    )
                if features is None:
                    assert stale_age_ms is not None
                    observation = None
                    payload = {
                        "field_features": None,
                        "localization": None,
                        "stale_dropped_age_ms": stale_age_ms,
                    }
                    undistorted = image.copy()
                    bev = projector.make_bev_image(image)
                    _put_lines(
                        undistorted,
                        [f"STALE dropped age={stale_age_ms:.1f} ms"],
                    )
                    _put_lines(
                        bev,
                        [f"STALE dropped age={stale_age_ms:.1f} ms"],
                    )
                    selected_current = False
                else:
                    observation = localizer.localize(
                        features,
                        prior_pose=prior_pose,
                    )
                    payload = {
                        "field_features": _feature_record(features),
                        "localization": _localization_record(observation),
                        "stale_dropped_age_ms": None,
                    }
                    undistorted = _undistorted_overlay(
                        image,
                        features,
                        observation,
                    )
                    bev = _bev_overlay(
                        image,
                        projector,
                        features,
                        observation,
                    )
                    selected_current = observation.selected_pose is not None
                    if selected_current:
                        result_timestamp_ns = observation.result_timestamp_ns
                        if (
                            last_print_timestamp_ns is None
                            or print_interval_ns == 0
                            or result_timestamp_ns - last_print_timestamp_ns
                            >= print_interval_ns
                        ):
                            _print_localization_calculation(
                                features,
                                observation,
                            )
                            last_print_timestamp_ns = result_timestamp_ns
                output.write(
                    json.dumps(
                        payload,
                        ensure_ascii=False,
                        allow_nan=False,
                    ) + "\n"
                )
                if args.camera:
                    output.flush()
                if args.overlay_dir is not None:
                    paths = (
                        args.overlay_dir / f"{sequence:06d}_undistorted.jpg",
                        args.overlay_dir / f"{sequence:06d}_bev.jpg",
                    )
                    for path, overlay in zip(paths, (undistorted, bev)):
                        if not cv2.imwrite(str(path), overlay):
                            raise RuntimeError(f"Failed to write overlay: {path}")
                if show_display:
                    cv2.imshow("center cross - undistorted", undistorted)
                    cv2.imshow("center cross - BEV", bev)
                    if cv2.waitKey(1) & 0xFF in {27, ord("q"), ord("Q")}:
                        processed += 1
                        selected += int(selected_current)
                        break
                processed += 1
                selected += int(selected_current)
    finally:
        if show_display:
            cv2.destroyAllWindows()
    print(
        f"Processed {processed} frames; unique pose selected for {selected}; "
        f"wrote {args.output_jsonl}."
    )


if __name__ == "__main__":
    main()
