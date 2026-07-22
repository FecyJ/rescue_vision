#!/usr/bin/env python3
"""Calculate camera extrinsics and image-to-ground mapping.

Coordinate conventions
----------------------
Robot/ground frame:
    x: forward
    y: left
    z: up
    unit: millimetres

OpenCV camera frame:
    x: image right
    y: image down
    z: camera forward

The script uses raw image points and known robot-ground coordinates. It:
1. undistorts raw pixels into the new_K image coordinate system;
2. fits a direct undistorted-pixel -> ground-mm homography;
3. estimates the robot-frame -> camera-frame pose with planar IPPE;
4. generates a BEV preview whose top is robot-forward and left is robot-left.

Default input layout:

    calibration_captures/ground_mapping/
        ground_image.png
        ground_points.json
        correspondences.json      # generated interactively if missing

Default outputs:

    output/ground_mapping.json
    output/ground_mapping.npz
    output/ground_diagnostics/

Example:
    python -m rescue_vision.calibration.calibrate_extrinsics_ground
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from rescue_vision.geometry.camera_model import CameraCalibration, CameraModel
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.geometry.types import RawPixel


CALIBRATION_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = CALIBRATION_DIR / "calibration_captures"
OUTPUT_DIR = CALIBRATION_DIR / "output"

DEFAULT_SESSION_DIR = CAPTURES_DIR / "ground_mapping"
WINDOW_NAME = "Ground Correspondence Collector"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Calculate camera extrinsics and ground homography."
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help="Ground calibration data directory.",
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help=(
            "Intrinsic selected_calibration.json. Defaults to the latest "
            "timestamped intrinsic output."
        ),
    )
    parser.add_argument(
        "--image",
        type=Path,
        default=None,
        help="Raw ground calibration image. Default: <session>/ground_image.png",
    )
    parser.add_argument(
        "--ground-points",
        type=Path,
        default=None,
        help="Known ground point file. Default: <session>/ground_points.json",
    )
    parser.add_argument(
        "--correspondences",
        type=Path,
        default=None,
        help=(
            "Raw pixel/ground correspondence file. Default: "
            "<session>/correspondences.json"
        ),
    )
    parser.add_argument(
        "--recollect",
        action="store_true",
        help="Ignore existing correspondences and click all points again.",
    )
    parser.add_argument(
        "--display-scale",
        type=float,
        default=0.5,
        help="Interactive image display scale.",
    )
    parser.add_argument(
        "--ransac-threshold-mm",
        type=float,
        default=20.0,
        help="RANSAC inlier threshold for ground homography, in mm.",
    )
    parser.add_argument("--bev-x-min-mm", type=float, default=-300.0)
    parser.add_argument("--bev-x-max-mm", type=float, default=2500.0)
    parser.add_argument("--bev-y-min-mm", type=float, default=-1200.0)
    parser.add_argument("--bev-y-max-mm", type=float, default=1200.0)
    parser.add_argument("--bev-mm-per-pixel", type=float, default=5.0)
    parser.add_argument(
        "--output-name",
        default="ground_mapping",
        help="Base name of JSON and NPZ output files.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def find_latest_intrinsics() -> Path:
    candidates = sorted(
        path
        for path in OUTPUT_DIR.glob("intrinsics_*/selected_calibration.json")
        if path.is_file()
    )
    if not candidates:
        raise FileNotFoundError(
            f"No selected_calibration.json found under {OUTPUT_DIR}"
        )
    return candidates[-1]


def load_intrinsics(
    path: Path,
) -> tuple[CameraCalibration, dict[str, Any]]:
    data = load_json(path)
    calibration = CameraCalibration.from_dict(data)
    return calibration, data


def load_ground_points(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    raw_points = data["points"] if isinstance(data, dict) else data

    points: list[dict[str, Any]] = []
    for index, item in enumerate(raw_points):
        if "ground_mm" in item:
            x_mm, y_mm = item["ground_mm"]
        else:
            x_mm = item["x_mm"]
            y_mm = item["y_mm"]

        points.append(
            {
                "name": item.get("name", f"P{index + 1:02d}"),
                "ground_mm": [float(x_mm), float(y_mm)],
            }
        )

    return points


def collect_correspondences(
    image: np.ndarray,
    image_path: Path,
    ground_points: list[dict[str, Any]],
    output_path: Path,
    display_scale: float,
) -> list[dict[str, Any]]:
    if display_scale <= 0.0:
        raise ValueError("--display-scale must be positive")

    selected: list[dict[str, Any]] = []
    pending_click: list[float] | None = None

    def mouse_callback(event: int, x: int, y: int, _flags: int, _param: Any) -> None:
        nonlocal pending_click
        if event == cv2.EVENT_LBUTTONDOWN:
            pending_click = [x / display_scale, y / display_scale]

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    try:
        while True:
            preview = image.copy()

            for index, item in enumerate(selected):
                u, v = item["pixel_raw"]
                cv2.circle(
                    preview,
                    (round(u), round(v)),
                    8,
                    (0, 255, 0),
                    -1,
                )
                cv2.putText(
                    preview,
                    item["name"],
                    (round(u) + 10, round(v) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

            if pending_click is not None:
                u, v = pending_click
                cv2.drawMarker(
                    preview,
                    (round(u), round(v)),
                    (0, 220, 255),
                    cv2.MARKER_CROSS,
                    25,
                    3,
                )

            if len(selected) < len(ground_points):
                target = ground_points[len(selected)]
                x_mm, y_mm = target["ground_mm"]
                status = (
                    f"Click {target['name']}: ground=({x_mm:.1f}, {y_mm:.1f}) mm; "
                    "Enter accept, Backspace undo, Q quit"
                )
            else:
                status = "All points collected; Enter save, Backspace undo"

            cv2.putText(
                preview,
                f"Collected: {len(selected)}/{len(ground_points)}",
                (30, 45),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.9,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                preview,
                status,
                (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 220, 255),
                2,
                cv2.LINE_AA,
            )

            shown = cv2.resize(
                preview,
                None,
                fx=display_scale,
                fy=display_scale,
                interpolation=cv2.INTER_AREA,
            )
            cv2.imshow(WINDOW_NAME, shown)

            key = cv2.waitKeyEx(20)

            if key in (ord("q"), ord("Q"), 27):
                raise KeyboardInterrupt("Point collection cancelled")

            if key in (8, 127, 3014656):
                if pending_click is not None:
                    pending_click = None
                elif selected:
                    selected.pop()
                continue

            if key not in (10, 13, 32):
                continue

            if len(selected) == len(ground_points):
                break

            if pending_click is None:
                continue

            target = ground_points[len(selected)]
            selected.append(
                {
                    "name": target["name"],
                    "ground_mm": target["ground_mm"],
                    "pixel_raw": [
                        float(pending_click[0]),
                        float(pending_click[1]),
                    ],
                }
            )
            pending_click = None

    finally:
        cv2.destroyWindow(WINDOW_NAME)

    result = {
        "image": image_path.name,
        "coordinate_frame": {
            "x": "robot_forward",
            "y": "robot_left",
            "z": "up",
            "unit": "mm",
        },
        "points": selected,
    }
    save_json(output_path, result)
    print(f"Correspondences saved: {output_path}")
    return selected


def load_correspondences(path: Path) -> list[dict[str, Any]]:
    data = load_json(path)
    raw_points = data["points"] if isinstance(data, dict) else data

    result: list[dict[str, Any]] = []
    for index, item in enumerate(raw_points):
        result.append(
            {
                "name": item.get("name", f"P{index + 1:02d}"),
                "pixel_raw": [
                    float(item["pixel_raw"][0]),
                    float(item["pixel_raw"][1]),
                ],
                "ground_mm": [
                    float(item["ground_mm"][0]),
                    float(item["ground_mm"][1]),
                ],
            }
        )
    return result


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float64),
        homography,
    ).reshape(-1, 2)


def fit_ground_homography(
    undistorted_pixels: np.ndarray,
    ground_points_mm: np.ndarray,
    threshold_mm: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    initial_h, mask = cv2.findHomography(
        undistorted_pixels,
        ground_points_mm,
        method=cv2.RANSAC,
        ransacReprojThreshold=threshold_mm,
    )
    if initial_h is None or mask is None:
        raise RuntimeError("Homography fitting failed")

    inliers = mask.reshape(-1).astype(bool)
    if np.count_nonzero(inliers) < 4:
        raise RuntimeError("Fewer than four homography inliers")

    refined_h, _ = cv2.findHomography(
        undistorted_pixels[inliers],
        ground_points_mm[inliers],
        method=0,
    )
    if refined_h is None:
        raise RuntimeError("Homography refinement failed")

    predicted = transform_points(undistorted_pixels, refined_h)
    errors_mm = np.linalg.norm(predicted - ground_points_mm, axis=1)
    return refined_h, inliers, errors_mm


def estimate_planar_pose(
    ground_points_mm: np.ndarray,
    undistorted_pixels: np.ndarray,
    new_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    object_points = np.column_stack(
        (
            ground_points_mm,
            np.zeros(len(ground_points_mm), dtype=np.float64),
        )
    ).astype(np.float64)
    image_points = undistorted_pixels.reshape(-1, 1, 2).astype(np.float64)

    result = cv2.solvePnPGeneric(
        object_points,
        image_points,
        new_camera_matrix,
        np.zeros((4, 1), dtype=np.float64),
        flags=cv2.SOLVEPNP_IPPE,
    )

    success, rvecs, tvecs = result[:3]
    if not success or len(rvecs) == 0:
        raise RuntimeError("Planar pose estimation failed")

    candidates: list[tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    for rvec, tvec in zip(rvecs, tvecs, strict=True):
        rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)
        rotation, _ = cv2.Rodrigues(rvec)

        camera_position_robot = -rotation.T @ tvec
        camera_points = (rotation @ object_points.T + tvec).T

        projected, _ = cv2.projectPoints(
            object_points,
            rvec,
            tvec,
            new_camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
        )
        projected = projected.reshape(-1, 2)
        pixel_rmse = float(
            np.sqrt(np.mean(np.sum((projected - undistorted_pixels) ** 2, axis=1)))
        )

        physically_valid = (
            camera_position_robot[2, 0] > 0.0
            and np.all(camera_points[:, 2] > 0.0)
        )
        score = pixel_rmse if physically_valid else pixel_rmse + 1e6
        candidates.append(
            (
                score,
                rvec,
                tvec,
                rotation,
                camera_position_robot,
            )
        )

    candidates.sort(key=lambda item: item[0])
    score, rvec, tvec, rotation, camera_position_robot = candidates[0]
    pixel_rmse = score if score < 1e6 else score - 1e6

    return (
        rvec,
        tvec,
        rotation,
        camera_position_robot,
        float(pixel_rmse),
    )


def pose_ground_homography(
    rotation_robot_to_camera: np.ndarray,
    translation_robot_to_camera: np.ndarray,
    new_camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    ground_to_image = new_camera_matrix @ np.column_stack(
        (
            rotation_robot_to_camera[:, 0],
            rotation_robot_to_camera[:, 1],
            translation_robot_to_camera.reshape(3),
        )
    )
    ground_to_image /= ground_to_image[2, 2]

    image_to_ground = np.linalg.inv(ground_to_image)
    image_to_ground /= image_to_ground[2, 2]
    return ground_to_image, image_to_ground


def save_point_preview(
    image: np.ndarray,
    correspondences: list[dict[str, Any]],
    undistorted_pixels: np.ndarray,
    output_path: Path,
) -> None:
    preview = image.copy()
    for item, pixel in zip(correspondences, undistorted_pixels, strict=True):
        u, v = pixel
        cv2.circle(preview, (round(u), round(v)), 7, (0, 255, 0), -1)
        cv2.putText(
            preview,
            item["name"],
            (round(u) + 9, round(v) - 9),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imwrite(str(output_path), preview)


def main() -> None:
    args = parse_args()

    session_dir = args.session.expanduser().resolve()
    image_path = (
        args.image.expanduser().resolve()
        if args.image is not None
        else session_dir / "ground_image.png"
    )
    ground_points_path = (
        args.ground_points.expanduser().resolve()
        if args.ground_points is not None
        else session_dir / "ground_points.json"
    )
    correspondences_path = (
        args.correspondences.expanduser().resolve()
        if args.correspondences is not None
        else session_dir / "correspondences.json"
    )
    intrinsics_path = (
        args.intrinsics.expanduser().resolve()
        if args.intrinsics is not None
        else find_latest_intrinsics().resolve()
    )

    session_dir.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    diagnostics_dir = OUTPUT_DIR / "ground_diagnostics"
    diagnostics_dir.mkdir(parents=True, exist_ok=True)

    calibration, intrinsic_data = load_intrinsics(intrinsics_path)
    camera_model = CameraModel(calibration)
    image_size = calibration.image_size
    new_camera_matrix = calibration.new_K

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read ground image: {image_path}")

    height, width = image.shape[:2]
    if (width, height) != image_size:
        raise ValueError(
            f"Ground image size {(width, height)} does not match "
            f"intrinsics {image_size}"
        )

    if args.recollect or not correspondences_path.exists():
        ground_points = load_ground_points(ground_points_path)
        correspondences = collect_correspondences(
            image,
            image_path,
            ground_points,
            correspondences_path,
            args.display_scale,
        )
    else:
        correspondences = load_correspondences(correspondences_path)

    if len(correspondences) < 4:
        raise RuntimeError("At least four ground correspondences are required")

    raw_pixels = np.asarray(
        [item["pixel_raw"] for item in correspondences],
        dtype=np.float64,
    )
    ground_points_mm = np.asarray(
        [item["ground_mm"] for item in correspondences],
        dtype=np.float64,
    )

    undistorted_pixels = np.asarray(
        [
            [point.u, point.v]
            for point in camera_model.undistort_pixels(
                [RawPixel(float(u), float(v)) for u, v in raw_pixels]
            )
        ],
        dtype=np.float64,
    )
    undistorted_image = camera_model.undistort_image(image)

    image_to_ground, inliers, ground_errors_mm = fit_ground_homography(
        undistorted_pixels,
        ground_points_mm,
        args.ransac_threshold_mm,
    )
    ground_to_image = np.linalg.inv(image_to_ground)
    ground_to_image /= ground_to_image[2, 2]

    (
        rvec_robot_to_camera,
        tvec_robot_to_camera,
        rotation_robot_to_camera,
        camera_position_robot,
        pose_reprojection_rmse_px,
    ) = estimate_planar_pose(
        ground_points_mm[inliers],
        undistorted_pixels[inliers],
        new_camera_matrix,
    )

    (
        pose_ground_to_image,
        pose_image_to_ground,
    ) = pose_ground_homography(
        rotation_robot_to_camera,
        tvec_robot_to_camera,
        new_camera_matrix,
    )

    rotation_camera_to_robot = rotation_robot_to_camera.T
    translation_camera_to_robot = camera_position_robot.reshape(3)

    robot_to_camera = np.eye(4, dtype=np.float64)
    robot_to_camera[:3, :3] = rotation_robot_to_camera
    robot_to_camera[:3, 3] = tvec_robot_to_camera.reshape(3)

    camera_to_robot = np.eye(4, dtype=np.float64)
    camera_to_robot[:3, :3] = rotation_camera_to_robot
    camera_to_robot[:3, 3] = translation_camera_to_robot

    bev_config = BevConfig(
        x_min=args.bev_x_min_mm,
        x_max=args.bev_x_max_mm,
        y_min=args.bev_y_min_mm,
        y_max=args.bev_y_max_mm,
        mm_per_pixel=args.bev_mm_per_pixel,
    )
    bev_width = bev_config.width
    bev_height = bev_config.height

    ground_to_bev = GroundProjector.make_ground_to_bev_matrix(
        bev_config,
    )
    image_to_bev = ground_to_bev @ image_to_ground

    bev = cv2.warpPerspective(
        undistorted_image,
        image_to_bev,
        (bev_width, bev_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    cv2.imwrite(
        str(diagnostics_dir / "undistorted_ground_image.png"),
        undistorted_image,
    )
    cv2.imwrite(
        str(diagnostics_dir / "bev_preview.png"),
        bev,
    )
    save_point_preview(
        undistorted_image,
        correspondences,
        undistorted_pixels,
        diagnostics_dir / "undistorted_correspondences.png",
    )

    per_point = []
    for item, raw, undistorted, inlier, error in zip(
        correspondences,
        raw_pixels,
        undistorted_pixels,
        inliers,
        ground_errors_mm,
        strict=True,
    ):
        per_point.append(
            {
                "name": item["name"],
                "pixel_raw": raw.tolist(),
                "pixel_undistorted": undistorted.tolist(),
                "ground_mm": item["ground_mm"],
                "homography_inlier": bool(inlier),
                "ground_error_mm": float(error),
            }
        )

    inlier_errors = ground_errors_mm[inliers]

    result = {
        "schema_version": 2,
        "coordinate_frames": {
            "robot": {
                "x": "forward",
                "y": "left",
                "z": "up",
                "unit": "mm",
            },
            "camera_opencv": {
                "x": "right",
                "y": "down",
                "z": "forward",
                "unit": "mm",
            },
            "image": "undistorted pixels produced with new_camera_matrix",
        },
        "source": {
            "intrinsics": str(intrinsics_path),
            "ground_image": str(image_path),
            "correspondences": str(correspondences_path),
        },
        "intrinsics": {
            "model_type": calibration.model.value,
            "fingerprint_sha256": calibration.fingerprint(),
            "quality": intrinsic_data.get("quality"),
        },
        "image_size": list(image_size),
        "lens_position": intrinsic_data.get("lens_position"),
        "image_to_ground": image_to_ground.tolist(),
        "ground_to_image": ground_to_image.tolist(),
        "direct_homography_error": {
            "ransac_threshold_mm": float(args.ransac_threshold_mm),
            "inlier_count": int(np.count_nonzero(inliers)),
            "total_count": int(len(inliers)),
            "mean_inlier_error_mm": float(np.mean(inlier_errors)),
            "median_inlier_error_mm": float(np.median(inlier_errors)),
            "max_inlier_error_mm": float(np.max(inlier_errors)),
        },
        "extrinsics": {
            "mapping": "p_camera = R_robot_to_camera @ p_robot + t_robot_to_camera",
            "rvec_robot_to_camera": rvec_robot_to_camera.reshape(3).tolist(),
            "rotation_robot_to_camera": rotation_robot_to_camera.tolist(),
            "translation_robot_to_camera_mm": (
                tvec_robot_to_camera.reshape(3).tolist()
            ),
            "camera_position_robot_mm": (
                camera_position_robot.reshape(3).tolist()
            ),
            "robot_to_camera_4x4": robot_to_camera.tolist(),
            "camera_to_robot_4x4": camera_to_robot.tolist(),
            "pose_reprojection_rmse_px": pose_reprojection_rmse_px,
            "pose_ground_to_image": pose_ground_to_image.tolist(),
            "pose_image_to_ground": pose_image_to_ground.tolist(),
        },
        "bev": {
            "x_min_mm": float(args.bev_x_min_mm),
            "x_max_mm": float(args.bev_x_max_mm),
            "y_min_mm": float(args.bev_y_min_mm),
            "y_max_mm": float(args.bev_y_max_mm),
            "mm_per_pixel": float(args.bev_mm_per_pixel),
            "width_px": bev_width,
            "height_px": bev_height,
            "orientation": {
                "top": "robot_forward",
                "left": "robot_left",
            },
            "ground_to_bev": ground_to_bev.tolist(),
            "image_to_bev": image_to_bev.tolist(),
        },
        "points": per_point,
    }

    json_path = OUTPUT_DIR / f"{args.output_name}.json"
    npz_path = OUTPUT_DIR / f"{args.output_name}.npz"

    save_json(json_path, result)
    np.savez(
        npz_path,
        image_to_ground=image_to_ground,
        ground_to_image=ground_to_image,
        rotation_robot_to_camera=rotation_robot_to_camera,
        translation_robot_to_camera=tvec_robot_to_camera,
        camera_position_robot=camera_position_robot,
        robot_to_camera=robot_to_camera,
        camera_to_robot=camera_to_robot,
        ground_to_bev=ground_to_bev,
        image_to_bev=image_to_bev,
    )

    print("\nGround calibration result")
    print(
        f"Homography inliers: {np.count_nonzero(inliers)}/{len(inliers)}"
    )
    print(f"Mean inlier error:  {np.mean(inlier_errors):.2f} mm")
    print(f"Median inlier error:{np.median(inlier_errors):.2f} mm")
    print(f"Max inlier error:   {np.max(inlier_errors):.2f} mm")
    print(f"Pose reprojection:  {pose_reprojection_rmse_px:.3f} px")
    print(
        "Camera position in robot frame [x, y, z] mm:\n"
        f"{camera_position_robot.reshape(3)}"
    )
    print("\nimage_to_ground =")
    print(image_to_ground)
    print(f"\nJSON:        {json_path}")
    print(f"NPZ:         {npz_path}")
    print(f"Diagnostics: {diagnostics_dir}")


if __name__ == "__main__":
    main()
