#!/usr/bin/env python3
"""Capture fisheye calibration images with a live OpenCV preview.

Board:
- 12 x 9 physical squares
- 11 x 8 internal corners

Window controls:
- Space / Enter: capture and validate one image
- Q / Esc: quit

Examples:
    python3 capture_chessboard_images.py
    python3 capture_chessboard_images.py --target 50
    python3 capture_chessboard_images.py --lens-position 0.8
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CALIBRATION_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = CALIBRATION_DIR / "calibration_captures"

IMAGE_SIZE = (2304, 1296)  # (width, height)
PATTERN_SIZE = (11, 8)      # internal corners
WINDOW_NAME = "Chessboard Calibration Capture"

CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_EXHAUSTIVE
    | cv2.CALIB_CB_ACCURACY
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture chessboard images for OpenCV fisheye calibration."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=CAPTURES_DIR,
        help="Root output directory.",
    )
    parser.add_argument(
        "--target",
        type=int,
        default=50,
        help="Target number of valid images.",
    )
    parser.add_argument(
        "--lens-position",
        type=float,
        default=None,
        help=(
            "Fixed LensPosition. If omitted, autofocus runs once at startup "
            "and the resulting position is locked."
        ),
    )
    parser.add_argument(
        "--preview-scale",
        type=float,
        default=0.5,
        help="Preview window scale relative to 2304x1296, default: 0.5.",
    )
    parser.add_argument(
        "--max-detection-scale",
        type=float,
        default=2.0,
        help=(
            "Maximum temporary image enlargement used when the board is "
            "small in the frame; default: 2.0."
        ),
    )
    return parser.parse_args()


def create_session_directory(root: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = root / f"chessboard_2304x1296_{stamp}"
    (session_dir / "images").mkdir(parents=True)
    (session_dir / "detected").mkdir()
    return session_dir


def lock_focus(camera: Any, requested_position: float | None) -> float:
    """Choose one focus position, then keep the camera in manual focus."""
    from libcamera import controls
    if requested_position is None:
        print("\nPlace the chessboard near the main working distance.")
        input("Press Enter to run autofocus once...")

        success = camera.autofocus_cycle()
        lens_position = float(camera.capture_metadata()["LensPosition"])

        print(f"Autofocus success: {success}")
        print(f"Autofocus LensPosition: {lens_position:.4f}")
    else:
        lens_position = requested_position

    camera.set_controls(
        {
            "AfMode": controls.AfModeEnum.Manual,
            "LensPosition": lens_position,
        }
    )
    time.sleep(0.3)

    actual = float(camera.capture_metadata()["LensPosition"])
    print(f"Locked LensPosition: {actual:.4f}\n")
    return actual


def _integer_camera_sequence(
    value: Any,
    *,
    attributes: tuple[str, ...],
    name: str,
) -> list[int]:
    if all(hasattr(value, attribute) for attribute in attributes):
        values = [getattr(value, attribute) for attribute in attributes]
    else:
        try:
            values = list(value)
        except TypeError as error:
            raise RuntimeError(
                f"Camera {name} must contain {len(attributes)} integers, got {value!r}."
            ) from error
    if len(values) != len(attributes) or any(
        isinstance(item, bool) or not isinstance(item, (int, np.integer))
        for item in values
    ):
        raise RuntimeError(
            f"Camera {name} must contain {len(attributes)} integers, got {value!r}."
        )
    return [int(item) for item in values]


def camera_binding_metadata(camera: Any) -> dict[str, Any]:
    """Read stable camera identity, sensor size and active scaler crop."""

    properties = getattr(camera, "camera_properties", None)
    if not isinstance(properties, dict):
        raise RuntimeError("Picamera2 camera_properties must be available.")
    model = properties.get("Model")
    if not isinstance(model, str) or not model.strip():
        raise RuntimeError(
            f"Camera properties do not provide a non-empty Model: {properties!r}."
        )
    pixel_array_size = _integer_camera_sequence(
        properties.get("PixelArraySize"),
        attributes=("width", "height"),
        name="PixelArraySize",
    )
    metadata = camera.capture_metadata()
    if not isinstance(metadata, dict):
        raise RuntimeError("Picamera2 capture_metadata() must return a mapping.")
    scaler_crop = _integer_camera_sequence(
        metadata.get("ScalerCrop"),
        attributes=("x", "y", "width", "height"),
        name="ScalerCrop",
    )
    return {
        "camera_model": model.strip(),
        "sensor_pixel_array_size": pixel_array_size,
        "scaler_crop": scaler_crop,
    }


def _detection_scales(max_scale: float) -> tuple[float, ...]:
    if not np.isfinite(max_scale) or max_scale < 1.0 or max_scale > 4.0:
        raise ValueError(
            f"max_detection_scale must be in [1.0, 4.0], got {max_scale!r}."
        )
    candidates = [1.0, 1.5, 2.0, 3.0, max_scale]
    return tuple(
        sorted(
            {
                round(float(scale), 6)
                for scale in candidates
                if scale <= max_scale
            }
        )
    )


def _scaled_gray(gray: np.ndarray, scale: float) -> np.ndarray:
    if scale == 1.0:
        return gray
    return cv2.resize(
        gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )


def _normalize_detected_corners(
    corners: np.ndarray | None,
    scale: float,
) -> np.ndarray | None:
    if corners is None:
        return None
    values = np.asarray(corners, dtype=np.float64).reshape(-1, 2)
    if len(values) != PATTERN_SIZE[0] * PATTERN_SIZE[1]:
        return None
    if scale != 1.0:
        values /= scale
    if not np.all(np.isfinite(values)):
        return None
    return values.reshape(-1, 1, 2).astype(np.float32)


def _find_full_pattern(
    gray: np.ndarray,
    *,
    scale: float,
    flags: int,
) -> np.ndarray | None:
    found, corners = cv2.findChessboardCornersSB(
        _scaled_gray(gray, scale),
        PATTERN_SIZE,
        flags=flags,
    )
    if not found:
        return None
    return _normalize_detected_corners(corners, scale)


def _find_larger_pattern(
    gray: np.ndarray,
    *,
    scale: float,
) -> np.ndarray | None:
    """Use SB's larger-pattern mode to recover a full board at small scale.

    A smaller requested pattern lets OpenCV search an oversized board.  We
    accept the result only when all 11x8 physical internal corners are
    returned; accepting an arbitrary partial lattice would lose its absolute
    row/column offset and could produce a plausible but wrong extrinsic pose.
    """

    larger_flags = CHESSBOARD_FLAGS | cv2.CALIB_CB_LARGER
    detection_gray = _scaled_gray(gray, scale)
    for requested_pattern in ((7, 5), (5, 4), (3, 3)):
        if hasattr(cv2, "findChessboardCornersSBWithMeta"):
            found, corners, _meta = cv2.findChessboardCornersSBWithMeta(
                detection_gray,
                requested_pattern,
                flags=larger_flags,
            )
        else:
            found, corners = cv2.findChessboardCornersSB(
                detection_gray,
                requested_pattern,
                flags=larger_flags,
            )
        if not found:
            continue
        normalized = _normalize_detected_corners(corners, scale)
        if normalized is not None:
            return normalized
    return None


def _find_legacy_pattern(
    gray: np.ndarray,
    *,
    scale: float,
) -> np.ndarray | None:
    detection_gray = _scaled_gray(gray, scale)
    found, corners = cv2.findChessboardCorners(
        detection_gray,
        PATTERN_SIZE,
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    normalized = _normalize_detected_corners(corners, scale) if found else None
    if normalized is None:
        return None
    refined_gray = detection_gray
    refined = cv2.cornerSubPix(
        refined_gray,
        normalized * scale,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.01,
        ),
    )
    return _normalize_detected_corners(refined, scale)


def detect_chessboard(frame_bgr, *, max_detection_scale: float = 2.0):
    """Detect the complete 11x8 board, using safe small-board fallbacks.

    The primary detector is unchanged.  If a distant board is too small for
    the original pass, the same full pattern is retried on 1.5x/2x images,
    then OpenCV's ``CALIB_CB_LARGER`` and legacy adaptive-threshold detector
    are tried.  Every accepted result still contains all 88 corners.
    """

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    scales = _detection_scales(max_detection_scale)

    corners = _find_full_pattern(
        gray,
        scale=1.0,
        flags=CHESSBOARD_FLAGS,
    )
    if corners is not None:
        return True, corners, sharpness

    for scale in scales[1:]:
        corners = _find_full_pattern(
            gray,
            scale=scale,
            flags=CHESSBOARD_FLAGS,
        )
        if corners is not None:
            return True, corners, sharpness

    for scale in scales:
        corners = _find_larger_pattern(gray, scale=scale)
        if corners is not None:
            return True, corners, sharpness

    for scale in scales:
        corners = _find_legacy_pattern(gray, scale=scale)
        if corners is not None:
            return True, corners, sharpness

    return False, None, sharpness


def make_preview(
    frame,
    saved_count: int,
    target: int,
    lens_position: float,
    status: str,
    preview_scale: float,
):
    preview = frame.copy()

    lines = [
        f"Saved: {saved_count}/{target}",
        f"LensPosition: {lens_position:.4f}",
        "Space/Enter: capture    Q/Esc: quit",
        status,
    ]

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


def save_json(path: Path, data: dict) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    from picamera2 import Picamera2

    args = parse_args()
    _detection_scales(args.max_detection_scale)
    session_dir = create_session_directory(args.output)
    images_dir = session_dir / "images"
    detected_dir = session_dir / "detected"
    records_path = session_dir / "images.jsonl"

    camera = Picamera2()
    config = camera.create_video_configuration(
        main={
            "size": IMAGE_SIZE,
            # Picamera2 RGB888 arrays can be passed directly to OpenCV.
            "format": "RGB888",
        },
        buffer_count=4,
    )
    camera.configure(config)
    camera.start()

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        print("Waiting for exposure and white balance to settle...")
        time.sleep(2.0)

        lens_position = lock_focus(camera, args.lens_position)
        binding_metadata = camera_binding_metadata(camera)

        session_info = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_size": list(IMAGE_SIZE),
            "pixel_format": "RGB888",
            "pattern_size_internal_corners": list(PATTERN_SIZE),
            "physical_square_count": [12, 9],
            "lens_position": lens_position,
            **binding_metadata,
            "target_valid_images": args.target,
            "preview_scale": args.preview_scale,
        }
        save_json(session_dir / "session.json", session_info)

        print("Window controls:")
        print("  Space / Enter: capture")
        print("  Q / Esc: quit")
        print(f"Output: {session_dir.resolve()}\n")

        saved_count = 0
        attempt_count = 0
        status = "Move and tilt the board before each capture."

        while saved_count < args.target:
            frame = camera.capture_array("main")
            preview = make_preview(
                frame,
                saved_count,
                args.target,
                lens_position,
                status,
                args.preview_scale,
            )
            cv2.imshow(WINDOW_NAME, preview)

            key = cv2.waitKeyEx(1)

            if key in (ord("q"), ord("Q"), 27):
                break

            if key not in (32, 10, 13):
                continue

            attempt_count += 1
            metadata = camera.capture_metadata()
            found, corners, sharpness = detect_chessboard(
                frame,
                max_detection_scale=args.max_detection_scale,
            )

            if not found:
                status = (
                    "Rejected: full 11x8 corners not detected; "
                    f"sharpness={sharpness:.1f}"
                )
                print(status)
                continue

            filename = f"chessboard_{saved_count:03d}.png"
            image_path = images_dir / filename
            preview_path = detected_dir / filename

            cv2.imwrite(str(image_path), frame)

            detected_preview = frame.copy()
            cv2.drawChessboardCorners(
                detected_preview,
                PATTERN_SIZE,
                corners,
                True,
            )
            cv2.imwrite(str(preview_path), detected_preview)

            record = {
                "index": saved_count,
                "attempt": attempt_count,
                "filename": filename,
                "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                "sharpness_laplacian_variance": sharpness,
                "lens_position": metadata.get("LensPosition"),
                "exposure_time_us": metadata.get("ExposureTime"),
                "analogue_gain": metadata.get("AnalogueGain"),
                "colour_temperature": metadata.get("ColourTemperature"),
            }

            with records_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(record, ensure_ascii=False) + "\n")

            saved_count += 1
            status = f"Saved {filename}; sharpness={sharpness:.1f}"
            print(
                f"{status}, exposure={record['exposure_time_us']} us, "
                f"gain={record['analogue_gain']}"
            )

        print(f"\nFinished. Valid images saved: {saved_count}")
        print(f"Images:   {images_dir.resolve()}")
        print(f"Overlays: {detected_dir.resolve()}")
        print(f"Metadata: {records_path.resolve()}")

    finally:
        cv2.destroyAllWindows()
        camera.stop()
        camera.close()


if __name__ == "__main__":
    main()
