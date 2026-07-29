"""对图片或视频离线检查传统视觉场地特征，不访问相机和 Hailo。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import monotonic_ns
from collections.abc import Iterator

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.config import load_runtime_config
from rescue_vision.perception import FieldFeatureDetectionResult


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def _inputs(path: Path) -> Iterator[tuple[int, np.ndarray]]:
    if path.suffix.lower() in IMAGE_SUFFIXES:
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f"Cannot decode image: {path}")
        yield 0, image
        return
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Cannot open video: {path}")
    sequence = 0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            yield sequence, image
            sequence += 1
    finally:
        capture.release()
    if sequence == 0:
        raise ValueError(f"Video contains no decodable frames: {path}")


def _pixel(point) -> list[float]:
    return [point.u, point.v]


def _ground(point) -> list[float]:
    return [point.x, point.y]


def _line(line) -> dict[str, object]:
    return {
        "undistorted": [
            _pixel(line.start_undistorted),
            _pixel(line.end_undistorted),
        ],
        "ground_mm": (
            [_ground(line.start_ground), _ground(line.end_ground)]
            if line.start_ground is not None
            else None
        ),
    }


def _record(result: FieldFeatureDetectionResult) -> dict[str, object]:
    return {
        "frame_sequence": result.frame_sequence,
        "capture_timestamp_ns": result.capture_timestamp_ns,
        "result_timestamp_ns": result.result_timestamp_ns,
        "image_size": list(result.image_size),
        "safe_zones": [
            {
                "physical_color": item.physical_color.value,
                "polygon_undistorted": [
                    _pixel(point) for point in item.polygon_undistorted
                ],
                "polygon_ground_mm": (
                    [_ground(point) for point in item.polygon_ground]
                    if item.polygon_ground is not None
                    else None
                ),
                "entrance": _line(item.entrance) if item.entrance else None,
                "divider": _line(item.divider) if item.divider else None,
                "halves": [
                    {
                        "side": half.side.value,
                        "polygon_undistorted": [
                            _pixel(point)
                            for point in half.polygon_undistorted
                        ],
                        "polygon_ground_mm": [
                            _ground(point) for point in half.polygon_ground
                        ],
                    }
                    for half in item.halves
                ],
                "confidence": item.confidence,
                "quality": sorted(value.value for value in item.quality),
            }
            for item in result.safe_zones
        ],
        "start_zones": [
            {
                "polygon_undistorted": [
                    _pixel(point) for point in item.polygon_undistorted
                ],
                "polygon_ground_mm": (
                    [_ground(point) for point in item.polygon_ground]
                    if item.polygon_ground is not None
                    else None
                ),
                "confidence": item.confidence,
                "quality": sorted(value.value for value in item.quality),
            }
            for item in result.start_zones
        ],
        "center_cross": (
            {
                "axes": [_line(axis) for axis in result.center_cross.axes],
                "intersection_undistorted": (
                    _pixel(result.center_cross.intersection_undistorted)
                    if result.center_cross.intersection_undistorted is not None
                    else None
                ),
                "intersection_ground_mm": (
                    _ground(result.center_cross.intersection_ground)
                    if result.center_cross.intersection_ground is not None
                    else None
                ),
                "confidence": result.center_cross.confidence,
                "quality": sorted(
                    value.value for value in result.center_cross.quality
                ),
            }
            if result.center_cross is not None
            else None
        ),
        "boundary_features": [
            {
                "kind": item.kind.value,
                "points_undistorted": [
                    _pixel(point) for point in item.points_undistorted
                ],
                "points_ground_mm": (
                    [_ground(point) for point in item.points_ground]
                    if item.points_ground is not None
                    else None
                ),
                "confidence": item.confidence,
                "quality": sorted(value.value for value in item.quality),
            }
            for item in result.boundary_features
        ],
    }


def _draw_polygon(
    image: np.ndarray,
    points,
    color: tuple[int, int, int],
    thickness: int = 2,
) -> None:
    polygon = np.rint([(point.u, point.v) for point in points]).astype(np.int32)
    cv2.polylines(image, [polygon], True, color, thickness, cv2.LINE_AA)


def _overlay(
    image: np.ndarray,
    result: FieldFeatureDetectionResult,
) -> np.ndarray:
    output = image.copy()
    for safe in result.safe_zones:
        color = (0, 0, 255) if safe.physical_color.value == "red" else (255, 0, 0)
        _draw_polygon(output, safe.polygon_undistorted, color, 3)
        for half in safe.halves:
            half_color = (
                (0, 255, 255)
                if half.side.value == "approach_left"
                else (255, 255, 0)
            )
            _draw_polygon(output, half.polygon_undistorted, half_color)
        for line, line_color in (
            (safe.entrance, (255, 0, 255)),
            (safe.divider, (0, 0, 0)),
        ):
            if line is not None:
                cv2.line(
                    output,
                    tuple(
                        np.rint(
                            (
                                line.start_undistorted.u,
                                line.start_undistorted.v,
                            )
                        ).astype(int)
                    ),
                    tuple(
                        np.rint(
                            (
                                line.end_undistorted.u,
                                line.end_undistorted.v,
                            )
                        ).astype(int)
                    ),
                    line_color,
                    3,
                    cv2.LINE_AA,
                )
    for start in result.start_zones:
        _draw_polygon(output, start.polygon_undistorted, (255, 0, 255), 3)
    if result.center_cross is not None:
        for axis in result.center_cross.axes:
            cv2.line(
                output,
                (
                    round(axis.start_undistorted.u),
                    round(axis.start_undistorted.v),
                ),
                (
                    round(axis.end_undistorted.u),
                    round(axis.end_undistorted.v),
                ),
                (0, 200, 0),
                3,
                cv2.LINE_AA,
            )
    for feature in result.boundary_features:
        points = [
            (round(point.u), round(point.v))
            for point in feature.points_undistorted
        ]
        if len(points) == 1:
            cv2.circle(output, points[0], 6, (0, 128, 255), 2, cv2.LINE_AA)
        else:
            cv2.line(output, points[0], points[1], (0, 128, 255), 1, cv2.LINE_AA)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run configured traditional field-feature detection offline."
    )
    parser.add_argument("input", type=Path, help="图片或视频路径")
    parser.add_argument("--config", type=Path, default=Path("configs/runtime.yaml"))
    parser.add_argument("--output-jsonl", type=Path, required=True)
    parser.add_argument("--overlay-dir", type=Path)
    parser.add_argument(
        "--already-undistorted",
        action="store_true",
        help="输入已经处于当前 CameraModel 的去畸变坐标系。",
    )
    args = parser.parse_args()

    config = load_runtime_config(args.config)
    geometry = config.build_geometry()
    camera_model = geometry.camera_model if geometry is not None else None
    projector = geometry.ground_projector if geometry is not None else None
    if not args.already_undistorted and camera_model is None:
        raise RuntimeError(
            "Raw input requires enabled intrinsics; otherwise pass "
            "--already-undistorted explicitly."
        )
    detector = config.perception.build_field_feature_detector(
        max_observation_age_ms=config.processing.max_observation_age_ms,
        ground_projector=projector,
    )
    if detector is None:
        raise RuntimeError("perception.field_features.enabled must be true.")

    args.output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if args.overlay_dir is not None:
        args.overlay_dir.mkdir(parents=True, exist_ok=True)
    processed = 0
    with args.output_jsonl.open("w", encoding="utf-8") as output:
        for sequence, input_image in _inputs(args.input):
            actual_size = (int(input_image.shape[1]), int(input_image.shape[0]))
            if actual_size != config.camera.image_size:
                raise ValueError(
                    f"Input frame {sequence} size {actual_size} does not match "
                    f"runtime camera {config.camera.image_size}."
                )
            image = (
                input_image
                if args.already_undistorted
                else camera_model.undistort_image(input_image)
            )
            valid_mask = (
                camera_model.valid_mask
                if camera_model is not None
                else np.full(image.shape[:2], 255, dtype=np.uint8)
            )
            capture_timestamp_ns = monotonic_ns()
            result = detector.detect(
                CameraFrame(sequence, capture_timestamp_ns, image),
                image,
                valid_mask=valid_mask,
            )
            output.write(
                json.dumps(_record(result), ensure_ascii=False, allow_nan=False)
                + "\n"
            )
            if args.overlay_dir is not None:
                overlay_path = args.overlay_dir / f"{sequence:06d}.jpg"
                if not cv2.imwrite(str(overlay_path), _overlay(image, result)):
                    raise RuntimeError(f"Failed to write overlay: {overlay_path}")
            processed += 1
    print(
        f"Processed {processed} frames; wrote observations to "
        f"{args.output_jsonl}."
    )


if __name__ == "__main__":
    main()
