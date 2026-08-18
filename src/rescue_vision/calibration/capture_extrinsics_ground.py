#!/usr/bin/env python3
"""Interactively capture fixed-camera ground-calibration board images.

The operator enters the symmetric long/short edge margins once, enters the
number of stations, and then supplies the field-coordinate position of the
board outer lower-left corner before each capture.  The last accepted image is
marked as the holdout station for the downstream extrinsic solver.

The camera is never moved.  The board must remain flat on the ground, with its
long edge parallel to field ``x`` and its short edge parallel to field ``y``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
import time
from typing import Any, Callable, Literal, Sequence

import cv2

from rescue_vision.calibration.calibrate_extrinsics_ground import (
    ground_calibration_frame_metadata,
)
from rescue_vision.calibration.capture_chessboard_images import (
    IMAGE_SIZE,
    PATTERN_SIZE,
    camera_binding_metadata,
    detect_chessboard,
    lock_focus,
)


CALIBRATION_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = CALIBRATION_DIR / "calibration_captures"
WINDOW_NAME = "Ground Extrinsic Calibration Capture"
PHYSICAL_SQUARE_COUNT = (12, 9)


@dataclass(frozen=True, slots=True)
class CapturedBoardImage:
    """Metadata for one accepted board image."""

    name: str
    image_path: Path
    reference_outer_corner_global_mm: tuple[float, float]
    role: Literal["fit", "holdout"]
    sharpness: float


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Interactively capture fixed-camera multi-position ground "
            "calibration board images."
        )
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help=(
            "Empty output session directory. If omitted, a timestamped "
            "directory is created under calibration_captures/."
        ),
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=15.0,
        help="Measured chessboard square side length; default: 15.0 mm.",
    )
    parser.add_argument(
        "--lens-position",
        type=float,
        default=None,
        help=(
            "Fixed LensPosition. If omitted, autofocus runs once before "
            "capture and the resulting position is locked."
        ),
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Live preview scale relative to the full image.",
    )
    parser.add_argument(
        "--max-detection-scale",
        type=float,
        default=2.0,
        help="Maximum temporary enlargement for small/distant boards.",
    )
    parser.add_argument(
        "--detected-corner-order",
        choices=("reference_first", "reference_last"),
        default="reference_first",
        help=(
            "Whether OpenCV's first detected corner is the marked reference "
            "corner or its 180-degree opposite."
        ),
    )
    return parser.parse_args(argv)


def _prompt_float(
    prompt: str,
    *,
    minimum: float,
    input_fn: Callable[[str], str] = input,
) -> float:
    while True:
        try:
            value = float(input_fn(prompt).strip())
        except (EOFError, KeyboardInterrupt):
            raise
        except ValueError:
            print("请输入数字。")
            continue
        if not math.isfinite(value) or value < minimum:
            print(f"请输入不小于 {minimum:g} 的有限数字。")
            continue
        return value


def prompt_edge_margins(
    *,
    input_fn: Callable[[str], str] = input,
) -> tuple[float, float]:
    """Prompt symmetric long/short-direction margins in millimetres."""

    print(
        "请输入棋盘外框到首个内角点的边距。"
        "本脚本假设左右长边方向边距相同、上下短边方向边距相同。"
    )
    long_margin_mm = _prompt_float(
        "长边方向边距（左/右均同，mm）：",
        minimum=0.0,
        input_fn=input_fn,
    )
    short_margin_mm = _prompt_float(
        "短边方向边距（下/上均同，mm）：",
        minimum=0.0,
        input_fn=input_fn,
    )
    return long_margin_mm, short_margin_mm


def prompt_capture_count(
    *,
    input_fn: Callable[[str], str] = input,
) -> int:
    """Prompt the station count required by the multi-image solver."""

    while True:
        try:
            count = int(input_fn("采集张数（至少 4，最后一张作 holdout）：").strip())
        except (EOFError, KeyboardInterrupt):
            raise
        except ValueError:
            print("请输入整数。")
            continue
        if count < 4:
            print("至少需要 4 张：至少 3 张 fit 和 1 张 holdout。")
            continue
        return count


def parse_global_coordinate(text: str) -> tuple[float, float]:
    """Parse ``x y`` or ``x,y`` field coordinates in millimetres."""

    normalized = text.strip().replace(",", " ")
    values = normalized.split()
    if len(values) != 2:
        raise ValueError("坐标必须是两个数字，例如 300 800 或 300,800。")
    try:
        coordinate = (float(values[0]), float(values[1]))
    except ValueError as error:
        raise ValueError("坐标必须只包含数字。") from error
    if not all(math.isfinite(value) for value in coordinate):
        raise ValueError("坐标必须是有限数字。")
    return coordinate


def prompt_global_coordinate(
    index: int,
    total: int,
    *,
    input_fn: Callable[[str], str] = input,
    reference_label: str = "棋盘外框左下角",
) -> tuple[float, float]:
    """Prompt the outer lower-left board corner in the field frame."""

    while True:
        try:
            text = input_fn(
                f"第 {index}/{total} 站：输入{reference_label}全局坐标 "
                "x y（mm）："
            )
            return parse_global_coordinate(text)
        except (EOFError, KeyboardInterrupt):
            raise
        except ValueError as error:
            print(f"坐标无效：{error}")


def edge_margin_document(
    long_margin_mm: float,
    short_margin_mm: float,
) -> dict[str, float]:
    """Expand symmetric direction margins into the solver's four-side schema."""

    if not math.isfinite(long_margin_mm) or long_margin_mm < 0.0:
        raise ValueError(f"long_margin_mm must be non-negative, got {long_margin_mm!r}.")
    if not math.isfinite(short_margin_mm) or short_margin_mm < 0.0:
        raise ValueError(
            f"short_margin_mm must be non-negative, got {short_margin_mm!r}."
        )
    return {
        "left": float(long_margin_mm),
        "right": float(long_margin_mm),
        "bottom": float(short_margin_mm),
        "top": float(short_margin_mm),
    }


def capture_role(index: int, total: int) -> Literal["fit", "holdout"]:
    if not 0 <= index < total:
        raise ValueError(f"index must be in [0, {total}), got {index}.")
    return "holdout" if index == total - 1 else "fit"


def build_board_calibration_document(
    session_dir: Path,
    *,
    square_size_mm: float,
    long_margin_mm: float,
    short_margin_mm: float,
    detected_corner_order: Literal["reference_first", "reference_last"],
    images: Sequence[CapturedBoardImage],
) -> dict[str, Any]:
    """Build the JSON consumed by ``calibrate_extrinsics_ground``."""

    if not math.isfinite(square_size_mm) or square_size_mm <= 0.0:
        raise ValueError(f"square_size_mm must be positive, got {square_size_mm!r}.")
    if detected_corner_order not in {"reference_first", "reference_last"}:
        raise ValueError(f"Unsupported detected_corner_order: {detected_corner_order!r}")
    margin = edge_margin_document(long_margin_mm, short_margin_mm)
    image_documents = []
    for image in images:
        relative_image = image.image_path.resolve().relative_to(session_dir.resolve())
        image_documents.append(
            {
                "name": image.name,
                "image": relative_image.as_posix(),
                "reference_outer_corner_global_mm": list(
                    image.reference_outer_corner_global_mm
                ),
                "role": image.role,
                "sharpness_laplacian_variance": float(image.sharpness),
            }
        )

    return {
        "coordinate_frame": ground_calibration_frame_metadata(),
        "board": {
            "pattern_size_internal_corners": list(PATTERN_SIZE),
            "physical_square_count": list(PHYSICAL_SQUARE_COUNT),
            "square_size_mm": float(square_size_mm),
            "reference": "lower_left_outer_corner",
            "reference_corner_marked": True,
            "edge_margin_mm": margin,
            "detected_corner_order": detected_corner_order,
        },
        "images": image_documents,
    }


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def create_session_directory(session: Path | None) -> Path:
    if session is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir = CAPTURES_DIR / f"ground_mapping_{stamp}"
    else:
        session_dir = session.expanduser().resolve()
    if session_dir.exists() and any(session_dir.iterdir()):
        raise FileExistsError(
            f"Output session {session_dir} already exists and is not empty; "
            "choose a new --session directory."
        )
    (session_dir / "images").mkdir(parents=True, exist_ok=False)
    (session_dir / "detected").mkdir()
    return session_dir


def make_preview(
    frame: Any,
    saved_count: int,
    target: int,
    lens_position: float,
    status: str,
    preview_scale: float,
    reference: tuple[float, float] | None,
    *,
    reference_label: str = "Outer lower-left global",
) -> Any:
    preview = frame.copy()
    lines = [
        f"Saved: {saved_count}/{target}",
        f"LensPosition: {lens_position:.4f}",
        "Space/Enter: capture    Q/Esc: quit",
        status,
    ]
    if reference is not None:
        lines.append(
            f"{reference_label}: ({reference[0]:.1f}, {reference[1]:.1f}) mm"
        )
    for index, text in enumerate(lines):
        cv2.putText(
            preview,
            text,
            (30, 45 + index * 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0) if index < 3 else (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
    if preview_scale != 1.0:
        preview = cv2.resize(
            preview,
            None,
            fx=preview_scale,
            fy=preview_scale,
            interpolation=cv2.INTER_AREA,
        )
    return preview


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if not math.isfinite(args.square_size_mm) or args.square_size_mm <= 0.0:
        raise ValueError("--square-size-mm must be positive and finite.")
    if args.preview_scale <= 0.0 or not math.isfinite(args.preview_scale):
        raise ValueError("--preview-scale must be positive and finite.")
    if (
        args.max_detection_scale < 1.0
        or args.max_detection_scale > 4.0
        or not math.isfinite(args.max_detection_scale)
    ):
        raise ValueError("--max-detection-scale must be finite in [1.0, 4.0].")

    # These prompts intentionally happen before camera startup so an operator
    # can correct the station plan before opening a hardware resource.
    long_margin_mm, short_margin_mm = prompt_edge_margins()
    target = prompt_capture_count()
    session_dir = create_session_directory(args.session)
    images_dir = session_dir / "images"
    detected_dir = session_dir / "detected"
    records: list[CapturedBoardImage] = []
    save_json(
        session_dir / "board_calibration.json",
        build_board_calibration_document(
            session_dir,
            square_size_mm=args.square_size_mm,
            long_margin_mm=long_margin_mm,
            short_margin_mm=short_margin_mm,
            detected_corner_order=args.detected_corner_order,
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
        time.sleep(2.0)
        lens_position = lock_focus(camera, args.lens_position)
        binding_metadata = camera_binding_metadata(camera)
        save_json(
            session_dir / "session.json",
            {
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "image_size": list(IMAGE_SIZE),
                "pixel_format": "RGB888",
                "pattern_size_internal_corners": list(PATTERN_SIZE),
                "physical_square_count": list(PHYSICAL_SQUARE_COUNT),
                "square_size_mm": args.square_size_mm,
                "long_margin_mm": long_margin_mm,
                "short_margin_mm": short_margin_mm,
                "edge_margin_mm": edge_margin_document(
                    long_margin_mm,
                    short_margin_mm,
                ),
                "detected_corner_order": args.detected_corner_order,
                "max_detection_scale": args.max_detection_scale,
                "target_valid_images": target,
                "lens_position": lens_position,
                **binding_metadata,
                "preview_scale": args.preview_scale,
            },
        )

        print(f"Output: {session_dir.resolve()}")
        print("每个站位输入外框左下角全局坐标后，将棋盘放好并按 Enter/Space 拍摄。")
        print("最后一张自动标为 holdout；Q/Esc 可退出并保留已完成 JSON。\n")

        status = "等待输入当前站位坐标。"
        for index in range(target):
            reference = prompt_global_coordinate(index + 1, target)
            status = "棋盘放置完成后按 Enter/Space；检测不到完整棋盘会拒绝本次拍摄。"
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
                )
                cv2.imshow(WINDOW_NAME, preview)
                key = cv2.waitKeyEx(1)
                if key in (ord("q"), ord("Q"), 27):
                    return
                if key not in (32, 10, 13):
                    continue

                metadata = camera.capture_metadata()
                found, corners, sharpness = detect_chessboard(
                    frame,
                    max_detection_scale=args.max_detection_scale,
                )
                if not found or corners is None:
                    status = (
                        "Rejected: full 11x8 corners not detected; "
                        f"sharpness={sharpness:.1f}; adjust board and retry."
                    )
                    print(status)
                    continue

                image_name = f"board_{index + 1:03d}.png"
                image_path = images_dir / image_name
                detected_path = detected_dir / image_name
                if not cv2.imwrite(str(image_path), frame):
                    raise OSError(f"Failed to write captured image {image_path}")
                detected_preview = frame.copy()
                cv2.drawChessboardCorners(
                    detected_preview,
                    PATTERN_SIZE,
                    corners,
                    True,
                )
                if not cv2.imwrite(str(detected_path), detected_preview):
                    raise OSError(f"Failed to write detection preview {detected_path}")

                record = CapturedBoardImage(
                    name=f"station_{index + 1:02d}",
                    image_path=image_path,
                    reference_outer_corner_global_mm=reference,
                    role=capture_role(index, target),
                    sharpness=sharpness,
                )
                records.append(record)
                save_json(
                    session_dir / "board_calibration.json",
                    build_board_calibration_document(
                        session_dir,
                        square_size_mm=args.square_size_mm,
                        long_margin_mm=long_margin_mm,
                        short_margin_mm=short_margin_mm,
                        detected_corner_order=args.detected_corner_order,
                        images=records,
                    ),
                )
                print(
                    f"Saved {image_name}; role={record.role}; "
                    f"sharpness={sharpness:.1f}; "
                    f"LensPosition={metadata.get('LensPosition')}"
                )
                break

        print(f"\nFinished. Valid images saved: {len(records)}/{target}")
        print(f"Board JSON: {session_dir / 'board_calibration.json'}")
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
