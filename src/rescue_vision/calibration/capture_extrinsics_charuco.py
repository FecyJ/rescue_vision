#!/usr/bin/env python3
"""Interactively capture fixed-camera ChArUco ground-calibration images."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import cv2

from rescue_vision.calibration.calibrate_extrinsics_ground import (
    ground_calibration_frame_metadata,
)
from rescue_vision.calibration.capture_extrinsics_ground import (
    capture_role,
    create_session_directory,
    make_preview,
    prompt_capture_count,
    prompt_edge_margins,
    prompt_global_coordinate,
    save_json,
)
from rescue_vision.calibration.capture_chessboard_images import (
    IMAGE_SIZE,
    camera_binding_metadata,
    lock_focus,
)
from rescue_vision.calibration.charuco_board import (
    CharucoDetection,
    charuco_detection_jitter_px,
    create_charuco_board,
    detect_charuco_board,
    dictionary_from_name,
)


CALIBRATION_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = CALIBRATION_DIR / "calibration_captures"
WINDOW_NAME = "ChArUco Ground Extrinsic Calibration Capture"


@dataclass(frozen=True, slots=True)
class CapturedCharucoImage:
    name: str
    image_path: Path
    board_origin_outer_corner_global_mm: tuple[float, float]
    role: Literal["fit", "holdout"]
    charuco_corner_count: int
    marker_count: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture fixed-camera multi-position ChArUco calibration images."
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Empty output directory; default is a timestamped calibration session.",
    )
    parser.add_argument("--squares-x", type=int, default=12)
    parser.add_argument("--squares-y", type=int, default=9)
    parser.add_argument("--square-size-mm", type=float, default=15.0)
    parser.add_argument(
        "--marker-size-mm",
        type=float,
        default=10.0,
        help="Printed ArUco marker side length inside each square.",
    )
    parser.add_argument(
        "--dictionary",
        default="DICT_4X4_100",
        help="OpenCV ArUco dictionary, e.g. DICT_4X4_100.",
    )
    parser.add_argument(
        "--minimum-charuco-corners",
        type=int,
        default=8,
        help="Minimum visible ChArUco corners accepted per image; default: 8.",
    )
    parser.add_argument(
        "--board-rotation-degrees",
        type=int,
        choices=(0, 90, 180, 270),
        default=0,
        help="Counter-clockwise rotation from OpenCV board x to field x.",
    )
    parser.add_argument(
        "--max-detection-scale",
        type=float,
        default=2.0,
        help="Maximum temporary enlargement for small/distant boards.",
    )
    parser.add_argument(
        "--minimum-sharpness",
        type=float,
        default=50.0,
        help="Minimum full-frame Laplacian variance accepted for capture.",
    )
    parser.add_argument(
        "--stability-frames",
        type=int,
        default=3,
        help="Consecutive detections required before accepting a capture.",
    )
    parser.add_argument(
        "--maximum-corner-jitter-px",
        type=float,
        default=2.0,
        help="Maximum common-corner displacement across stability frames.",
    )
    parser.add_argument("--lens-position", type=float, default=None)
    parser.add_argument("--preview-scale", type=float, default=0.5)
    return parser.parse_args(argv)


def build_charuco_calibration_document(
    session_dir: Path,
    *,
    squares_x: int,
    squares_y: int,
    square_size_mm: float,
    marker_size_mm: float,
    dictionary_name: str,
    minimum_charuco_corners: int,
    board_rotation_degrees: int,
    long_margin_mm: float,
    short_margin_mm: float,
    images: Sequence[CapturedCharucoImage],
) -> dict[str, Any]:
    if squares_x < 2 or squares_y < 2:
        raise ValueError("squares_x and squares_y must both be >=2.")
    if not math.isfinite(square_size_mm) or square_size_mm <= 0.0:
        raise ValueError("square_size_mm must be positive and finite.")
    if not math.isfinite(marker_size_mm) or not 0.0 < marker_size_mm < square_size_mm:
        raise ValueError("marker_size_mm must be positive and smaller than square_size_mm.")
    if minimum_charuco_corners < 8:
        raise ValueError("minimum_charuco_corners must be >=8.")
    if board_rotation_degrees not in {0, 90, 180, 270}:
        raise ValueError("board_rotation_degrees must be 0, 90, 180 or 270.")
    if not math.isfinite(long_margin_mm) or long_margin_mm < 0.0:
        raise ValueError("long_margin_mm must be finite and non-negative.")
    if not math.isfinite(short_margin_mm) or short_margin_mm < 0.0:
        raise ValueError("short_margin_mm must be finite and non-negative.")
    margin = {
        "left": float(long_margin_mm),
        "right": float(long_margin_mm),
        "bottom": float(short_margin_mm),
        "top": float(short_margin_mm),
    }
    image_documents = []
    for image in images:
        relative = image.image_path.resolve().relative_to(session_dir.resolve())
        image_documents.append(
            {
                "name": image.name,
                "image": relative.as_posix(),
                "board_origin_outer_corner_global_mm": list(
                    image.board_origin_outer_corner_global_mm
                ),
                "role": image.role,
                "charuco_corner_count": image.charuco_corner_count,
                "marker_count": image.marker_count,
            }
        )
    return {
        "coordinate_frame": ground_calibration_frame_metadata(),
        "board": {
            "board_type": "charuco",
            "chessboard_size_squares": [squares_x, squares_y],
            "square_size_mm": float(square_size_mm),
            "marker_size_mm": float(marker_size_mm),
            "dictionary": dictionary_name,
            "minimum_charuco_corners": minimum_charuco_corners,
            "reference": "opencv_board_origin_outer_corner",
            "printed_face": "camera",
            "board_rotation_degrees": board_rotation_degrees,
            "edge_margin_mm": margin,
        },
        "images": image_documents,
    }


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.preview_scale <= 0.0 or not math.isfinite(args.preview_scale):
        raise ValueError("--preview-scale must be positive and finite.")
    if args.minimum_charuco_corners < 8:
        raise ValueError("--minimum-charuco-corners must be >=8.")
    if not math.isfinite(args.max_detection_scale) or args.max_detection_scale < 1.0:
        raise ValueError("--max-detection-scale must be finite and >=1.")
    if not math.isfinite(args.minimum_sharpness) or args.minimum_sharpness <= 0.0:
        raise ValueError("--minimum-sharpness must be positive and finite.")
    if args.stability_frames < 2:
        raise ValueError("--stability-frames must be >=2.")
    if (
        not math.isfinite(args.maximum_corner_jitter_px)
        or args.maximum_corner_jitter_px <= 0.0
    ):
        raise ValueError("--maximum-corner-jitter-px must be positive and finite.")

    board = create_charuco_board(
        (args.squares_x, args.squares_y),
        args.square_size_mm,
        args.marker_size_mm,
        args.dictionary,
    )
    # Resolve the dictionary before opening the camera so a typo fails early.
    dictionary_from_name(args.dictionary)
    long_margin_mm, short_margin_mm = prompt_edge_margins()
    target = prompt_capture_count()
    session_dir = create_session_directory(args.session)
    images_dir = session_dir / "images"
    detected_dir = session_dir / "detected"
    records: list[CapturedCharucoImage] = []
    board_json_path = session_dir / "board_calibration.json"
    save_json(
        board_json_path,
        build_charuco_calibration_document(
            session_dir,
            squares_x=args.squares_x,
            squares_y=args.squares_y,
            square_size_mm=args.square_size_mm,
            marker_size_mm=args.marker_size_mm,
            dictionary_name=args.dictionary,
            minimum_charuco_corners=args.minimum_charuco_corners,
            board_rotation_degrees=args.board_rotation_degrees,
            long_margin_mm=long_margin_mm,
            short_margin_mm=short_margin_mm,
            images=records,
        ),
    )

    from picamera2 import Picamera2

    camera = Picamera2()
    camera_started = False
    window_created = False
    try:
        camera_config = camera.create_video_configuration(
            main={"size": IMAGE_SIZE, "format": "RGB888"},
            buffer_count=4,
        )
        camera.configure(camera_config)
        camera.start()
        camera_started = True
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        window_created = True
        print("Waiting for exposure and white balance to settle...")
        import time

        time.sleep(2.0)
        lens_position = lock_focus(camera, args.lens_position)
        binding_metadata = camera_binding_metadata(camera)
        save_json(
            session_dir / "session.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "image_size": list(IMAGE_SIZE),
                "pixel_format": "RGB888",
                "board_type": "charuco",
                "chessboard_size_squares": [args.squares_x, args.squares_y],
                "square_size_mm": args.square_size_mm,
                "marker_size_mm": args.marker_size_mm,
                "dictionary": args.dictionary,
                "minimum_charuco_corners": args.minimum_charuco_corners,
                "board_rotation_degrees": args.board_rotation_degrees,
                "printed_face": "camera",
                "max_detection_scale": args.max_detection_scale,
                "minimum_sharpness": args.minimum_sharpness,
                "stability_frames": args.stability_frames,
                "maximum_corner_jitter_px": args.maximum_corner_jitter_px,
                "long_margin_mm": long_margin_mm,
                "short_margin_mm": short_margin_mm,
                "target_valid_images": target,
                "lens_position": lens_position,
                **binding_metadata,
            },
        )
        print(f"Output: {session_dir.resolve()}")
        print("每个站位输入 OpenCV 板原点侧外框角坐标后，将 ChArUco 板放好并拍摄。")
        print("最后一张自动为 holdout；Q/Esc 退出并保留已完成 JSON。\n")

        for index in range(target):
            reference = prompt_global_coordinate(
                index + 1,
                target,
                reference_label="OpenCV 板原点侧外框角",
            )
            status = "棋盘放置完成后按 Enter/Space；需要检测到足够 ChArUco 角点。"
            while True:
                frame = camera.capture_array("main")
                preview = make_preview(
                    frame,
                    len(records),
                    target,
                    lens_position,
                    status,
                    args.preview_scale,
                    reference,
                    reference_label="OpenCV-origin outer corner",
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q"), 27):
                    return
                if key not in (32, 10, 13):
                    continue
                detection = detect_charuco_board(
                    frame,
                    board,
                    minimum_corners=args.minimum_charuco_corners,
                    max_detection_scale=args.max_detection_scale,
                )
                if detection is None:
                    status = (
                        f"Rejected: fewer than {args.minimum_charuco_corners} "
                        "ChArUco corners; move board closer or improve light."
                    )
                    print(status)
                    continue
                stability_detections: list[CharucoDetection] = [detection]
                for _ in range(args.stability_frames - 1):
                    stability_frame = camera.capture_array("main")
                    stability_detection = detect_charuco_board(
                        stability_frame,
                        board,
                        minimum_corners=args.minimum_charuco_corners,
                        max_detection_scale=args.max_detection_scale,
                    )
                    if stability_detection is None:
                        break
                    stability_detections.append(stability_detection)
                if len(stability_detections) != args.stability_frames:
                    status = "Rejected: ChArUco detection was not stable across frames."
                    print(status)
                    continue
                jitter_px = charuco_detection_jitter_px(
                    stability_detections,
                    minimum_common_corners=args.minimum_charuco_corners,
                )
                if jitter_px > args.maximum_corner_jitter_px:
                    status = (
                        f"Rejected: corner jitter {jitter_px:.2f}px exceeds "
                        f"{args.maximum_corner_jitter_px:.2f}px."
                    )
                    print(status)
                    continue
                sharpness = float(
                    cv2.Laplacian(
                        cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY),
                        cv2.CV_64F,
                    ).var()
                )
                if sharpness < args.minimum_sharpness:
                    status = (
                        f"Rejected: sharpness {sharpness:.1f} is below "
                        f"{args.minimum_sharpness:.1f}."
                    )
                    print(status)
                    continue

                image_name = f"charuco_{index + 1:03d}.png"
                image_path = images_dir / image_name
                detected_path = detected_dir / image_name
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Failed to write captured image {image_path}")
                detected_preview = frame.copy()
                if detection.marker_corners:
                    cv2.aruco.drawDetectedMarkers(
                        detected_preview,
                        list(detection.marker_corners),
                        detection.marker_ids.reshape(-1, 1),
                    )
                cv2.aruco.drawDetectedCornersCharuco(
                    detected_preview,
                    detection.charuco_corners.reshape(-1, 1, 2).astype("float32"),
                    detection.charuco_ids.reshape(-1, 1),
                )
                if not cv2.imwrite(str(detected_path), detected_preview):
                    raise OSError(f"Failed to write detection preview {detected_path}")

                records.append(
                    CapturedCharucoImage(
                        name=f"station_{index + 1:02d}",
                        image_path=image_path,
                        board_origin_outer_corner_global_mm=reference,
                        role=capture_role(index, target),
                        charuco_corner_count=len(detection.charuco_ids),
                        marker_count=len(detection.marker_ids),
                    )
                )
                save_json(
                    board_json_path,
                    build_charuco_calibration_document(
                        session_dir,
                        squares_x=args.squares_x,
                        squares_y=args.squares_y,
                        square_size_mm=args.square_size_mm,
                        marker_size_mm=args.marker_size_mm,
                        dictionary_name=args.dictionary,
                        minimum_charuco_corners=args.minimum_charuco_corners,
                        board_rotation_degrees=args.board_rotation_degrees,
                        long_margin_mm=long_margin_mm,
                        short_margin_mm=short_margin_mm,
                        images=records,
                    ),
                )
                print(
                    f"Saved {image_name}; role={records[-1].role}; "
                    f"charuco={len(detection.charuco_ids)}, "
                    f"markers={len(detection.marker_ids)}, "
                    f"scale={detection.detection_scale:.1f}, sharpness={sharpness:.1f}, "
                    f"jitter={jitter_px:.2f}px"
                )
                break

        print(f"\nFinished. Valid images saved: {len(records)}/{target}")
        print(f"Board JSON: {board_json_path}")
        print(f"Images:    {images_dir}")
        print(f"Overlays:  {detected_dir}")
    finally:
        if window_created:
            cv2.destroyWindow(WINDOW_NAME)
        if camera_started:
            camera.stop()
        camera.close()


if __name__ == "__main__":
    main()
