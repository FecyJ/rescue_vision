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


def detect_chessboard(frame_bgr):
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCornersSB(
        gray,
        PATTERN_SIZE,
        flags=CHESSBOARD_FLAGS,
    )
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    return found, corners, sharpness


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

        session_info = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "image_size": list(IMAGE_SIZE),
            "pixel_format": "RGB888",
            "pattern_size_internal_corners": list(PATTERN_SIZE),
            "physical_square_count": [12, 9],
            "lens_position": lens_position,
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
            found, corners, sharpness = detect_chessboard(frame)

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
