#!/usr/bin/env python3
"""Calculate camera extrinsics and image-to-ground mapping.

Coordinate conventions
----------------------
Robot/ground frame:
    origin: midpoint between the two drive-wheel ground contact points
    x: forward
    y: left
    z: up
    unit: millimetres

OpenCV camera frame:
    x: image right
    y: image down
    z: camera forward

The script uses multiple raw board images and known field coordinates. It:
1. undistorts raw pixels into the new_K image coordinate system;
2. expands each board's 11x8 corners from a known field reference point;
3. fits a direct undistorted-pixel -> robot-ground-mm homography;
4. estimates the robot-frame -> camera-frame pose with planar IPPE and LM refinement;
5. generates a BEV preview whose top is robot-forward and left is robot-left.

Default input layout:

    calibration_captures/ground_mapping/
        board_calibration.json
        images/*.png

Default outputs:

    output/ground_mapping_<timestamp>/ground_mapping.json
    output/ground_mapping_<timestamp>/ground_mapping.npz
    output/ground_mapping_<timestamp>/diagnostics/

Example:
    python -m rescue_vision.calibration.calibrate_extrinsics_ground
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from rescue_vision.config import load_runtime_config
from rescue_vision.geometry.camera_model import (
    CameraCalibration,
    CameraModel,
    CameraModelType,
)
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.geometry.types import (
    RawPixel,
    robot_frame_metadata,
)
from rescue_vision.calibration.capture_chessboard_images import (
    PATTERN_SIZE,
    detect_chessboard,
)
from rescue_vision.calibration.charuco_board import (
    charuco_points_field_mm,
    create_charuco_board,
    detect_charuco_board,
)


CALIBRATION_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CALIBRATION_DIR.parents[2]
CAPTURES_DIR = CALIBRATION_DIR / "calibration_captures"
OUTPUT_DIR = CALIBRATION_DIR / "output"

DEFAULT_SESSION_DIR = CAPTURES_DIR / "ground_mapping"
DEFAULT_RUNTIME_CONFIG_PATH = PROJECT_ROOT / "configs" / "runtime.yaml"
# The field frame used only while collecting this calibration is anchored at
# the robot start pose.  Runtime geometry remains expressed in the canonical
# robot frame returned by ``robot_frame_metadata``.
GROUND_CALIBRATION_FRAME = {
    "origin": "robot_start_position",
    "x": "field_x",
    "y": "field_y",
    "z": "up",
    "unit": "mm",
    "robot_origin_global_mm": [0.0, 0.0],
    "robot_x_positive": "field_y_positive",
    "robot_y_positive": "field_x_negative",
}
PHYSICAL_SQUARE_COUNT = (12, 9)
DEFAULT_BOARD_CALIBRATION_NAME = "board_calibration.json"


@dataclass(frozen=True, slots=True)
class BoardImageSpec:
    """One raw image and its normalized board reference point."""

    image_path: Path
    reference_inner_corner_field_mm: tuple[float, float]
    role: Literal["fit", "holdout"]
    name: str
    reference_corner_field_mm: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class BoardCalibrationSpec:
    """Validated multi-image board calibration description."""

    source_path: Path
    square_size_mm: float
    edge_margin_mm: tuple[float, float, float, float]
    detected_corner_order: Literal["reference_first", "reference_last"]
    images: tuple[BoardImageSpec, ...]
    board_type: Literal["chessboard", "charuco"] = "chessboard"
    chessboard_size_squares: tuple[int, int] = PHYSICAL_SQUARE_COUNT
    marker_size_mm: float | None = None
    dictionary_name: str | None = None
    minimum_charuco_corners: int = 8
    board_rotation_degrees: int = 0
    reference: Literal[
        "lower_left_inner_corner",
        "lower_left_outer_corner",
        "opencv_board_origin_outer_corner",
    ] = "lower_left_inner_corner"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Calculate fixed-camera extrinsics from multiple automatically "
            "detected chessboard or ChArUco images."
        )
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=DEFAULT_SESSION_DIR,
        help="Ground calibration data directory.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_RUNTIME_CONFIG_PATH,
        help=(
            "Runtime YAML used to resolve geometry.intrinsics_path when "
            "--intrinsics is omitted."
        ),
    )
    parser.add_argument(
        "--intrinsics",
        type=Path,
        default=None,
        help=(
            "Override selected_calibration.json. By default the path comes "
            "from geometry.intrinsics_path in --config."
        ),
    )
    parser.add_argument(
        "--board-calibration",
        type=Path,
        default=None,
        help=(
            "JSON describing the board geometry and image positions. "
            "Default: <session>/board_calibration.json"
        ),
    )
    parser.add_argument(
        "--ransac-threshold-px",
        type=float,
        default=3.0,
        help="RANSAC threshold in undistorted pixels.",
    )
    parser.add_argument(
        "--minimum-station-inlier-ratio",
        type=float,
        default=0.6,
        help="Minimum point inlier ratio for a fit station.",
    )
    parser.add_argument(
        "--maximum-corners-per-station",
        type=int,
        default=24,
        help="Spatially balanced point cap used by each fit station.",
    )
    parser.add_argument(
        "--maximum-mean-inlier-error-mm",
        type=float,
        default=20.0,
        help="Maximum usable mean inlier error, in mm.",
    )
    parser.add_argument(
        "--maximum-inlier-error-mm",
        type=float,
        default=50.0,
        help="Maximum usable worst inlier error, in mm.",
    )
    parser.add_argument(
        "--maximum-pose-rmse-px",
        type=float,
        default=5.0,
        help="Maximum usable planar-pose reprojection RMSE, in pixels.",
    )
    parser.add_argument(
        "--maximum-holdout-mean-error-mm",
        type=float,
        default=30.0,
        help="Maximum usable mean ground error on held-out board positions.",
    )
    parser.add_argument(
        "--maximum-holdout-error-mm",
        type=float,
        default=75.0,
        help="Maximum usable worst ground error on held-out board positions.",
    )
    parser.add_argument(
        "--maximum-leave-one-out-mean-error-mm",
        type=float,
        default=30.0,
        help="Maximum worst-station mean error during leave-one-station-out.",
    )
    parser.add_argument(
        "--maximum-mapping-disagreement-mm",
        type=float,
        default=20.0,
        help="Maximum direct-vs-physical mapping disagreement.",
    )
    parser.add_argument(
        "--max-detection-scale",
        type=float,
        default=2.0,
        help="Maximum temporary enlargement for small/distant boards.",
    )
    parser.add_argument(
        "--minimum-charuco-sharpness",
        type=float,
        default=50.0,
        help="Minimum Laplacian variance accepted for a ChArUco solver image.",
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


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def ground_calibration_frame_metadata() -> dict[str, Any]:
    """Return the field-frame contract used by board calibration JSON."""

    return {
        "origin": GROUND_CALIBRATION_FRAME["origin"],
        "x": GROUND_CALIBRATION_FRAME["x"],
        "y": GROUND_CALIBRATION_FRAME["y"],
        "z": GROUND_CALIBRATION_FRAME["z"],
        "unit": GROUND_CALIBRATION_FRAME["unit"],
        "robot_origin_global_mm": list(
            GROUND_CALIBRATION_FRAME["robot_origin_global_mm"]
        ),
        "robot_x_positive": GROUND_CALIBRATION_FRAME["robot_x_positive"],
        "robot_y_positive": GROUND_CALIBRATION_FRAME["robot_y_positive"],
    }


def _finite_pair(value: object, *, name: str) -> tuple[float, float]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError(f"{name} must be a two-element list of numbers.")
    try:
        result = (float(value[0]), float(value[1]))
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers, got {value!r}.") from error
    if not all(np.isfinite(item) for item in result):
        raise ValueError(f"{name} must contain finite numbers, got {value!r}.")
    return result


def _finite_margin(value: object, *, name: str) -> tuple[float, float, float, float]:
    if not isinstance(value, dict):
        raise ValueError(
            f"{name} must contain left, right, bottom and top millimetres."
        )
    keys = ("left", "right", "bottom", "top")
    if any(key not in value for key in keys):
        raise ValueError(
            f"{name} must contain keys {keys}, got {list(value)!r}."
        )
    try:
        result = tuple(float(value[key]) for key in keys)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must contain numbers, got {value!r}.") from error
    if not all(np.isfinite(item) and item >= 0.0 for item in result):
        raise ValueError(
            f"{name} values must be finite and non-negative, got {value!r}."
        )
    return result  # type: ignore[return-value]


def load_board_calibration(path: Path) -> BoardCalibrationSpec:
    """Load and validate the multi-image chessboard calibration description.

    Image paths are resolved relative to the JSON file.  When the reference is
    the outer lower-left corner, the configured left/bottom margins are added
    to obtain the first internal corner used for object-point coordinates.
    """

    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"Board calibration {path} must be a JSON object.")

    frame = data.get("coordinate_frame")
    expected_frame = ground_calibration_frame_metadata()
    if frame != expected_frame:
        raise ValueError(
            f"Board calibration {path}.coordinate_frame must equal "
            f"{expected_frame!r}, got {frame!r}."
        )

    board = data.get("board")
    if not isinstance(board, dict):
        raise ValueError(f"Board calibration {path}.board must be an object.")
    board_type = board.get("board_type", "chessboard")
    if board_type not in {"chessboard", "charuco"}:
        raise ValueError(
            f"Board calibration board_type must be 'chessboard' or 'charuco', "
            f"got {board_type!r}."
        )

    if board_type == "chessboard":
        pattern_size = board.get("pattern_size_internal_corners")
        if pattern_size != list(PATTERN_SIZE):
            raise ValueError(
                f"Board calibration pattern_size_internal_corners must be "
                f"{list(PATTERN_SIZE)}, got {pattern_size!r}."
            )
        if board.get("physical_square_count") != list(PHYSICAL_SQUARE_COUNT):
            raise ValueError(
                f"Board calibration physical_square_count must be "
                f"{list(PHYSICAL_SQUARE_COUNT)}, got "
                f"{board.get('physical_square_count')!r}."
            )
        chessboard_size_squares = PHYSICAL_SQUARE_COUNT
    else:
        raw_size = board.get("chessboard_size_squares")
        if (
            not isinstance(raw_size, (list, tuple))
            or len(raw_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 2
                for value in raw_size
            )
        ):
            raise ValueError(
                "Charuco board chessboard_size_squares must contain two "
                f"integers >=2, got {raw_size!r}."
            )
        chessboard_size_squares = (int(raw_size[0]), int(raw_size[1]))

    try:
        square_size_mm = float(board.get("square_size_mm"))
    except (TypeError, ValueError) as error:
        raise ValueError(
            "Board calibration square_size_mm must be a number."
        ) from error
    if not np.isfinite(square_size_mm) or square_size_mm <= 0.0:
        raise ValueError(
            f"Board calibration square_size_mm must be positive and finite, "
            f"got {square_size_mm!r}."
        )

    marker_size_mm: float | None = None
    dictionary_name: str | None = None
    minimum_charuco_corners = 8
    board_rotation_degrees = 0
    if board_type == "charuco":
        try:
            marker_size_mm = float(board.get("marker_size_mm"))
        except (TypeError, ValueError) as error:
            raise ValueError("Charuco marker_size_mm must be a number.") from error
        if not np.isfinite(marker_size_mm) or marker_size_mm <= 0.0:
            raise ValueError(
                f"Charuco marker_size_mm must be positive, got {marker_size_mm!r}."
            )
        if marker_size_mm >= square_size_mm:
            raise ValueError(
                "Charuco marker_size_mm must be smaller than square_size_mm."
            )
        dictionary_name = board.get("dictionary")
        if not isinstance(dictionary_name, str) or not dictionary_name:
            raise ValueError("Charuco board.dictionary must be a non-empty string.")
        minimum_charuco_corners = board.get("minimum_charuco_corners", 8)
        if (
            isinstance(minimum_charuco_corners, bool)
            or not isinstance(minimum_charuco_corners, int)
            or minimum_charuco_corners < 8
        ):
            raise ValueError(
                "Charuco minimum_charuco_corners must be an integer >=8."
            )
        if board.get("printed_face") != "camera":
            raise ValueError(
                "Charuco board.printed_face must be 'camera'; face-down boards "
                "would mirror the OpenCV board coordinate frame."
            )
        board_rotation_degrees = board.get("board_rotation_degrees")
        if (
            isinstance(board_rotation_degrees, bool)
            or board_rotation_degrees not in {0, 90, 180, 270}
        ):
            raise ValueError(
                "Charuco board_rotation_degrees must be one of 0, 90, 180 or 270."
            )

    reference = board.get("reference")
    if reference not in {
        "lower_left_inner_corner",
        "lower_left_outer_corner",
        "opencv_board_origin_outer_corner",
    }:
        raise ValueError(
            "Board calibration board.reference must be "
            "'lower_left_inner_corner', 'lower_left_outer_corner' or "
            "'opencv_board_origin_outer_corner'."
        )
    if board_type == "charuco" and reference != "opencv_board_origin_outer_corner":
        raise ValueError(
            "Charuco calibration requires "
            "reference='opencv_board_origin_outer_corner' so the physically "
            "marked OpenCV board origin and rotation are explicit."
        )
    if board_type == "chessboard" and reference == "opencv_board_origin_outer_corner":
        raise ValueError(
            "Chessboard calibration cannot use the ChArUco OpenCV board origin reference."
        )
    if board_type == "chessboard" and board.get("reference_corner_marked") is not True:
        raise ValueError(
            "Board calibration board.reference_corner_marked must be true; "
            "a plain symmetric chessboard cannot resolve its 180-degree "
            "corner-order ambiguity."
        )
    edge_margin_mm = _finite_margin(
        board.get("edge_margin_mm"),
        name=f"Board calibration {path}.board.edge_margin_mm",
    )
    detected_corner_order = board.get("detected_corner_order", "reference_first")
    if detected_corner_order not in {"reference_first", "reference_last"}:
        raise ValueError(
            f"Board calibration detected_corner_order must be "
            "'reference_first' or 'reference_last'; the physical reference "
            f"corner must be marked to resolve 180-degree ambiguity, got "
            f"{detected_corner_order!r}."
        )

    raw_images = data.get("images")
    if not isinstance(raw_images, list):
        raise ValueError(f"Board calibration {path}.images must be a list.")
    if len(raw_images) < 4:
        raise ValueError(
            "Board calibration requires at least four images, including one "
            "holdout image."
        )

    images: list[BoardImageSpec] = []
    seen_paths: set[Path] = set()
    seen_names: set[str] = set()
    for index, item in enumerate(raw_images):
        if not isinstance(item, dict):
            raise ValueError(f"Board calibration images[{index}] must be an object.")
        image_name = item.get("image")
        if not isinstance(image_name, str) or not image_name:
            raise ValueError(
                f"Board calibration images[{index}].image must be a non-empty string."
            )
        image_path = Path(image_name).expanduser()
        if not image_path.is_absolute():
            image_path = (path.parent / image_path).resolve()
        else:
            image_path = image_path.resolve()
        if image_path in seen_paths:
            raise ValueError(f"Board calibration repeats image path {image_name!r}.")
        seen_paths.add(image_path)

        role = item.get("role")
        if role not in {"fit", "holdout"}:
            raise ValueError(
                f"Board calibration images[{index}].role must be 'fit' or "
                f"'holdout', got {role!r}."
            )
        if reference == "lower_left_inner_corner":
            reference_key = "reference_inner_corner_global_mm"
        elif reference == "lower_left_outer_corner":
            reference_key = "reference_outer_corner_global_mm"
        else:
            reference_key = "board_origin_outer_corner_global_mm"
        reference_field_mm = _finite_pair(
            item.get(reference_key),
            name=(
                f"Board calibration images[{index}]"
                f".{reference_key}"
            ),
        )
        if reference == "lower_left_outer_corner":
            reference_inner_corner_mm = (
                reference_field_mm[0] + edge_margin_mm[0],
                reference_field_mm[1] + edge_margin_mm[2],
            )
        else:
            reference_inner_corner_mm = reference_field_mm
        name = item.get("name", image_path.stem)
        if not isinstance(name, str) or not name.strip():
            raise ValueError(
                f"Board calibration images[{index}].name must be a non-empty string."
            )
        name = name.strip()
        if name in seen_names:
            raise ValueError(f"Board calibration repeats station name {name!r}.")
        seen_names.add(name)
        images.append(
            BoardImageSpec(
                image_path=image_path,
                reference_inner_corner_field_mm=reference_inner_corner_mm,
                role=role,
                name=name,
                reference_corner_field_mm=reference_field_mm,
            )
        )

    fit_count = sum(item.role == "fit" for item in images)
    holdout_count = sum(item.role == "holdout" for item in images)
    if fit_count < 3 or holdout_count < 1:
        raise ValueError(
            "Board calibration requires at least three fit images and one "
            f"holdout image, got fit={fit_count}, holdout={holdout_count}."
        )
    references = [item.reference_corner_field_mm for item in images]
    if len(set(references)) != len(references):
        raise ValueError("Board calibration station reference coordinates must be unique.")
    fit_references = np.asarray(
        [
            item.reference_corner_field_mm
            for item in images
            if item.role == "fit"
        ],
        dtype=np.float64,
    )
    fit_minimum = np.min(fit_references, axis=0)
    fit_maximum = np.max(fit_references, axis=0)
    if np.linalg.matrix_rank(
        fit_references - np.mean(fit_references, axis=0),
        tol=1e-9,
    ) < 2:
        raise ValueError(
            "Fit-station reference coordinates must span two field dimensions."
        )
    for item in images:
        if item.role != "holdout":
            continue
        reference_value = np.asarray(item.reference_corner_field_mm, dtype=np.float64)
        if np.any(reference_value < fit_minimum) or np.any(reference_value > fit_maximum):
            raise ValueError(
                f"Holdout station {item.name!r} lies outside the fit-station "
                "reference bounding box; holdout must validate interpolation."
            )

    return BoardCalibrationSpec(
        source_path=path.resolve(),
        square_size_mm=square_size_mm,
        edge_margin_mm=edge_margin_mm,
        detected_corner_order=detected_corner_order,
        images=tuple(images),
        reference=reference,
        board_type=board_type,
        chessboard_size_squares=chessboard_size_squares,
        marker_size_mm=marker_size_mm,
        dictionary_name=dictionary_name,
        minimum_charuco_corners=minimum_charuco_corners,
        board_rotation_degrees=board_rotation_degrees,
    )


def board_points_field_mm(
    reference_inner_corner_field_mm: tuple[float, float],
    square_size_mm: float,
) -> np.ndarray:
    """Return the 11x8 board inner corners in field x/y order."""

    if not np.isfinite(square_size_mm) or square_size_mm <= 0.0:
        raise ValueError(
            f"square_size_mm must be positive and finite, got {square_size_mm!r}."
        )
    columns, rows = PATTERN_SIZE
    reference = np.asarray(reference_inner_corner_field_mm, dtype=np.float64)
    if reference.shape != (2,) or not np.all(np.isfinite(reference)):
        raise ValueError(
            "reference_inner_corner_field_mm must contain two finite numbers."
        )
    points = [
        reference + np.array([column * square_size_mm, row * square_size_mm])
        for row in range(rows)
        for column in range(columns)
    ]
    return np.asarray(points, dtype=np.float64)


def field_points_to_robot_ground(field_points_mm: np.ndarray) -> np.ndarray:
    """Convert calibration field x/y points to the canonical robot x/y frame."""

    points = np.asarray(field_points_mm, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            f"field_points_mm must have shape (N, 2), got {points.shape}."
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("field_points_mm must contain only finite values.")
    # robot x = field y and robot y = -field x, as declared in the JSON frame.
    return np.column_stack((points[:, 1], -points[:, 0]))


@dataclass(frozen=True, slots=True)
class BoardObservation:
    """Detected corners and derived ground points for one board image."""

    spec: BoardImageSpec
    image: np.ndarray
    undistorted_image: np.ndarray
    raw_pixels: np.ndarray
    undistorted_pixels: np.ndarray
    field_points_mm: np.ndarray
    robot_points_mm: np.ndarray
    sharpness: float
    corner_ids: np.ndarray | None = None


@dataclass(frozen=True, slots=True)
class StationRobustFit:
    """Station-balanced direct homography used for filtering and diagnostics."""

    ground_to_image: np.ndarray
    image_to_ground: np.ndarray
    full_inliers: np.ndarray
    pose_ground_points_mm: np.ndarray
    pose_undistorted_pixels: np.ndarray
    valid_station_names: tuple[str, ...]
    station_metrics: dict[str, dict[str, float | int | bool]]
    leave_one_out_metrics: dict[str, dict[str, float]]
    worst_leave_one_out_mean_error_mm: float


def detect_board_observation(
    spec: BoardImageSpec,
    board: BoardCalibrationSpec,
    camera_model: CameraModel,
    *,
    max_detection_scale: float = 2.0,
    minimum_charuco_sharpness: float = 50.0,
) -> BoardObservation:
    """Detect and orient one board image without manual pixel clicks."""

    image = cv2.imread(str(spec.image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read chessboard image: {spec.image_path}")
    height, width = image.shape[:2]
    if (width, height) != camera_model.image_size:
        raise ValueError(
            f"Chessboard image {spec.image_path} has size {(width, height)}, "
            f"but intrinsics require {camera_model.image_size}."
        )

    if board.board_type == "charuco":
        charuco_board = create_charuco_board(
            board.chessboard_size_squares,
            board.square_size_mm,
            board.marker_size_mm,
            board.dictionary_name,
        )
        detection = detect_charuco_board(
            image,
            charuco_board,
            minimum_corners=board.minimum_charuco_corners,
            max_detection_scale=max_detection_scale,
            camera_matrix=(
                camera_model.calibration.K
                if camera_model.calibration.model is not CameraModelType.FISHEYE
                else None
            ),
            distortion=(
                camera_model.calibration.D
                if camera_model.calibration.model is not CameraModelType.FISHEYE
                else None
            ),
        )
        if detection is None:
            raise RuntimeError(
                f"ChArUco image {spec.image_path} has fewer than "
                f"{board.minimum_charuco_corners} usable corners."
            )
        raw_pixels = detection.charuco_corners
        corner_ids = detection.charuco_ids
        field_points_mm = charuco_points_field_mm(
            charuco_board,
            corner_ids,
            spec.reference_corner_field_mm
            or spec.reference_inner_corner_field_mm,
            board.edge_margin_mm,
            board.board_rotation_degrees,
        )
        sharpness = float(
            cv2.Laplacian(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()
        )
        if (
            not np.isfinite(minimum_charuco_sharpness)
            or minimum_charuco_sharpness <= 0.0
        ):
            raise ValueError("minimum_charuco_sharpness must be positive and finite.")
        if sharpness < minimum_charuco_sharpness:
            raise RuntimeError(
                f"ChArUco image {spec.image_path} sharpness {sharpness:.1f} is "
                f"below {minimum_charuco_sharpness:.1f}."
            )
    else:
        found, corners, sharpness = detect_chessboard(
            image,
            max_detection_scale=max_detection_scale,
        )
        if (
            not found
            or corners is None
            or len(corners) != PATTERN_SIZE[0] * PATTERN_SIZE[1]
        ):
            raise RuntimeError(
                f"Chessboard image {spec.image_path} does not contain the required "
                f"{PATTERN_SIZE[0]}x{PATTERN_SIZE[1]} internal corners."
            )

        corner_grid = np.asarray(corners, dtype=np.float64).reshape(
            PATTERN_SIZE[1], PATTERN_SIZE[0], 2
        )
        if board.detected_corner_order == "reference_last":
            corner_grid = np.flip(corner_grid, axis=(0, 1))
        raw_pixels = corner_grid.reshape(-1, 2)
        corner_ids = None
        field_points_mm = board_points_field_mm(
            spec.reference_inner_corner_field_mm,
            board.square_size_mm,
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
    robot_points_mm = field_points_to_robot_ground(field_points_mm)
    return BoardObservation(
        spec=spec,
        image=image,
        undistorted_image=camera_model.undistort_image(image),
        raw_pixels=raw_pixels,
        undistorted_pixels=undistorted_pixels,
        field_points_mm=field_points_mm,
        robot_points_mm=robot_points_mm,
        sharpness=sharpness,
        corner_ids=corner_ids,
    )


def load_intrinsics(
    path: Path,
) -> tuple[CameraCalibration, dict[str, Any]]:
    data = load_json(path)
    if not isinstance(data, dict):
        raise ValueError(
            f"Intrinsic JSON must be an object, got {type(data).__name__}."
        )
    calibration = CameraCalibration.from_json(path)
    return calibration, data


def load_configured_intrinsics(
    config_path: Path,
) -> tuple[Path, CameraModel, dict[str, Any]]:
    """Load the intrinsic JSON selected by the strict runtime configuration."""

    config = load_runtime_config(config_path)
    if not config.geometry.intrinsics_enabled:
        raise ValueError(
            f"Runtime config {config_path} has "
            "geometry.intrinsics_enabled=false; enable it or pass "
            "--intrinsics explicitly."
        )

    camera_model = config.build_camera_model()
    if camera_model is None or config.geometry.intrinsics_path is None:
        raise ValueError(
            f"Runtime config {config_path} does not provide usable "
            "geometry.intrinsics_path."
        )

    intrinsics_path = config.geometry.intrinsics_path
    intrinsic_data = load_json(intrinsics_path)
    if not isinstance(intrinsic_data, dict):
        raise ValueError(
            f"Intrinsic JSON {intrinsics_path} must be an object, got "
            f"{type(intrinsic_data).__name__}."
        )
    return intrinsics_path, camera_model, intrinsic_data


def validate_capture_session(
    board: BoardCalibrationSpec,
    calibration: CameraCalibration,
) -> dict[str, Any]:
    """Bind ground images to their capture conditions and selected intrinsics."""

    session_path = board.source_path.parent / "session.json"
    data = load_json(session_path)
    if not isinstance(data, dict):
        raise ValueError(f"Capture session {session_path} must be a JSON object.")
    if data.get("image_size") != list(calibration.image_size):
        raise ValueError(
            f"Capture session image_size {data.get('image_size')!r} does not "
            f"match intrinsics {calibration.image_size!r}."
        )
    capture_lens = data.get("lens_position")
    if (
        isinstance(capture_lens, bool)
        or not isinstance(capture_lens, (int, float))
        or not np.isfinite(capture_lens)
    ):
        raise ValueError("Capture session lens_position must be a finite number.")
    if calibration.lens_position is None:
        raise ValueError(
            "Selected intrinsics do not record lens_position; regenerate the "
            "current-format intrinsic calibration before ground solving."
        )
    if not np.isclose(
        float(capture_lens),
        calibration.lens_position,
        rtol=0.0,
        atol=1e-3,
    ):
        raise ValueError(
            f"Capture lens_position {capture_lens!r} does not match intrinsics "
            f"{calibration.lens_position!r}."
        )
    intrinsic_binding = {
        "camera_model": calibration.camera_model,
        "sensor_pixel_array_size": (
            None
            if calibration.sensor_pixel_array_size is None
            else list(calibration.sensor_pixel_array_size)
        ),
        "scaler_crop": (
            None if calibration.scaler_crop is None else list(calibration.scaler_crop)
        ),
    }
    if any(value is None for value in intrinsic_binding.values()):
        raise ValueError(
            "Selected intrinsics do not contain complete camera binding metadata; "
            "regenerate the current-format intrinsic calibration."
        )
    for key, expected in intrinsic_binding.items():
        if data.get(key) != expected:
            raise ValueError(
                f"Capture session {key} {data.get(key)!r} does not match "
                f"intrinsics {expected!r}."
            )
    session_board_type = data.get("board_type", "chessboard")
    if session_board_type != board.board_type:
        raise ValueError(
            f"Capture board_type {session_board_type!r} does not match "
            f"board calibration {board.board_type!r}."
        )
    if not (
        np.isclose(board.edge_margin_mm[0], board.edge_margin_mm[1])
        and np.isclose(board.edge_margin_mm[2], board.edge_margin_mm[3])
    ):
        raise ValueError(
            "Current capture workflow requires symmetric left/right and bottom/top margins."
        )
    numeric_checks = {
        "square_size_mm": board.square_size_mm,
        "long_margin_mm": board.edge_margin_mm[0],
        "short_margin_mm": board.edge_margin_mm[2],
    }
    if board.board_type == "charuco":
        numeric_checks["marker_size_mm"] = float(board.marker_size_mm)
        exact_checks = {
            "pixel_format": "RGB888",
            "chessboard_size_squares": list(board.chessboard_size_squares),
            "dictionary": board.dictionary_name,
            "minimum_charuco_corners": board.minimum_charuco_corners,
            "board_rotation_degrees": board.board_rotation_degrees,
            "printed_face": "camera",
        }
    else:
        exact_checks = {
            "pixel_format": "RGB888",
            "pattern_size_internal_corners": list(PATTERN_SIZE),
            "physical_square_count": list(PHYSICAL_SQUARE_COUNT),
            "detected_corner_order": board.detected_corner_order,
        }
    for key, expected in numeric_checks.items():
        actual = data.get(key)
        if (
            isinstance(actual, bool)
            or not isinstance(actual, (int, float))
            or not np.isclose(float(actual), expected, rtol=0.0, atol=1e-9)
        ):
            raise ValueError(
                f"Capture session {key} must equal {expected!r}, got {actual!r}."
            )
    for key, expected in exact_checks.items():
        if data.get(key) != expected:
            raise ValueError(
                f"Capture session {key} must equal {expected!r}, got "
                f"{data.get(key)!r}."
            )
    return data


def transform_points(points: np.ndarray, homography: np.ndarray) -> np.ndarray:
    return cv2.perspectiveTransform(
        points.reshape(-1, 1, 2).astype(np.float64),
        homography,
    ).reshape(-1, 2)


def _spatially_balanced_indices(
    points: np.ndarray,
    maximum_count: int,
) -> np.ndarray:
    """Select deterministic farthest-point samples so one station cannot dominate."""

    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 2 or not np.all(np.isfinite(values)):
        raise ValueError(f"points must have finite shape (N, 2), got {values.shape}.")
    if maximum_count < 4:
        raise ValueError(f"maximum_count must be >=4, got {maximum_count}.")
    if len(values) <= maximum_count:
        return np.arange(len(values), dtype=np.int64)
    selected = [int(np.argmin(values[:, 0] + values[:, 1]))]
    minimum_squared_distance = np.full(len(values), np.inf, dtype=np.float64)
    while len(selected) < maximum_count:
        latest = values[selected[-1]]
        squared_distance = np.sum((values - latest) ** 2, axis=1)
        minimum_squared_distance = np.minimum(
            minimum_squared_distance,
            squared_distance,
        )
        minimum_squared_distance[selected] = -1.0
        selected.append(int(np.argmax(minimum_squared_distance)))
    return np.asarray(sorted(selected), dtype=np.int64)


def fit_station_robust_homography(
    observations: list[BoardObservation],
    *,
    threshold_px: float,
    minimum_station_inlier_ratio: float,
    maximum_corners_per_station: int,
) -> StationRobustFit:
    """Fit ground-to-image geometry with equalized, station-level support."""

    if len(observations) < 3:
        raise ValueError(
            f"At least three fit stations are required, got {len(observations)}."
        )
    if not np.isfinite(threshold_px) or threshold_px <= 0.0:
        raise ValueError(f"threshold_px must be positive, got {threshold_px!r}.")
    if (
        not np.isfinite(minimum_station_inlier_ratio)
        or not 0.0 < minimum_station_inlier_ratio <= 1.0
    ):
        raise ValueError(
            "minimum_station_inlier_ratio must be in (0, 1], got "
            f"{minimum_station_inlier_ratio!r}."
        )
    sampled: list[tuple[BoardObservation, np.ndarray]] = []
    for observation in observations:
        indices = _spatially_balanced_indices(
            observation.robot_points_mm,
            maximum_corners_per_station,
        )
        sampled.append((observation, indices))
    sampled_ground = np.concatenate(
        [observation.robot_points_mm[indices] for observation, indices in sampled]
    )
    sampled_pixels = np.concatenate(
        [observation.undistorted_pixels[indices] for observation, indices in sampled]
    )
    initial_ground_to_image, _mask = cv2.findHomography(
        sampled_ground,
        sampled_pixels,
        method=cv2.RANSAC,
        ransacReprojThreshold=float(threshold_px),
    )
    if initial_ground_to_image is None:
        raise RuntimeError("Station-balanced homography fitting failed.")

    station_metrics: dict[str, dict[str, float | int | bool]] = {}
    valid_names: list[str] = []
    full_masks: list[np.ndarray] = []
    for observation in observations:
        predicted = transform_points(
            observation.robot_points_mm,
            initial_ground_to_image,
        )
        errors = np.linalg.norm(predicted - observation.undistorted_pixels, axis=1)
        mask = errors <= threshold_px
        ratio = float(np.mean(mask))
        valid = int(np.count_nonzero(mask)) >= 8 and ratio >= minimum_station_inlier_ratio
        station_metrics[observation.spec.name] = {
            "corner_count": len(mask),
            "inlier_count": int(np.count_nonzero(mask)),
            "inlier_ratio": ratio,
            "mean_pixel_error": float(np.mean(errors)),
            "max_pixel_error": float(np.max(errors)),
            "valid": valid,
        }
        full_masks.append(mask)
        if valid:
            valid_names.append(observation.spec.name)
    if len(valid_names) < 3:
        raise RuntimeError(
            "Fewer than three fit stations satisfy the station inlier gate: "
            f"valid={valid_names!r}."
        )

    pose_ground_parts: list[np.ndarray] = []
    pose_pixel_parts: list[np.ndarray] = []
    for (observation, indices), mask in zip(sampled, full_masks, strict=True):
        if observation.spec.name not in valid_names:
            continue
        selected = indices[mask[indices]]
        if len(selected) < 4:
            continue
        pose_ground_parts.append(observation.robot_points_mm[selected])
        pose_pixel_parts.append(observation.undistorted_pixels[selected])
    pose_ground = np.concatenate(pose_ground_parts)
    pose_pixels = np.concatenate(pose_pixel_parts)
    refined_ground_to_image, _ = cv2.findHomography(
        pose_ground,
        pose_pixels,
        method=0,
    )
    if refined_ground_to_image is None:
        raise RuntimeError("Station-balanced homography refinement failed.")
    direct_image_to_ground = np.linalg.inv(refined_ground_to_image)
    direct_image_to_ground /= direct_image_to_ground[2, 2]

    final_masks: list[np.ndarray] = []
    for observation in observations:
        errors = np.linalg.norm(
            transform_points(observation.robot_points_mm, refined_ground_to_image)
            - observation.undistorted_pixels,
            axis=1,
        )
        final_masks.append(
            (errors <= threshold_px)
            & (observation.spec.name in valid_names)
        )
        final_mask = final_masks[-1]
        station_metrics[observation.spec.name].update(
            {
                "inlier_count": int(np.count_nonzero(final_mask)),
                "inlier_ratio": float(np.mean(final_mask)),
                "mean_pixel_error": float(np.mean(errors)),
                "max_pixel_error": float(np.max(errors)),
            }
        )
        if observation.spec.name in valid_names and (
            int(np.count_nonzero(final_mask)) < 8
            or float(np.mean(final_mask)) < minimum_station_inlier_ratio
        ):
            raise RuntimeError(
                f"Fit station {observation.spec.name!r} fell below the station "
                "inlier gate after homography refinement."
            )

    final_pose_ground_parts: list[np.ndarray] = []
    final_pose_pixel_parts: list[np.ndarray] = []
    for (observation, indices), mask in zip(sampled, final_masks, strict=True):
        if observation.spec.name not in valid_names:
            continue
        selected = indices[mask[indices]]
        if len(selected) < 4:
            raise RuntimeError(
                f"Fit station {observation.spec.name!r} has fewer than four "
                "balanced inliers after homography refinement."
            )
        final_pose_ground_parts.append(observation.robot_points_mm[selected])
        final_pose_pixel_parts.append(observation.undistorted_pixels[selected])
    pose_ground = np.concatenate(final_pose_ground_parts)
    pose_pixels = np.concatenate(final_pose_pixel_parts)

    leave_one_out: dict[str, dict[str, float]] = {}
    for omitted_name in valid_names:
        loo_ground_parts: list[np.ndarray] = []
        loo_pixel_parts: list[np.ndarray] = []
        omitted_observation: BoardObservation | None = None
        for (observation, indices), mask in zip(sampled, final_masks, strict=True):
            if observation.spec.name == omitted_name:
                omitted_observation = observation
                continue
            if observation.spec.name not in valid_names:
                continue
            selected = indices[mask[indices]]
            loo_ground_parts.append(observation.robot_points_mm[selected])
            loo_pixel_parts.append(observation.undistorted_pixels[selected])
        assert omitted_observation is not None
        loo_ground_to_image, _ = cv2.findHomography(
            np.concatenate(loo_ground_parts),
            np.concatenate(loo_pixel_parts),
            method=0,
        )
        if loo_ground_to_image is None:
            raise RuntimeError(
                f"Leave-one-station-out fitting failed for {omitted_name!r}."
            )
        loo_image_to_ground = np.linalg.inv(loo_ground_to_image)
        errors_mm = np.linalg.norm(
            transform_points(
                omitted_observation.undistorted_pixels,
                loo_image_to_ground,
            )
            - omitted_observation.robot_points_mm,
            axis=1,
        )
        leave_one_out[omitted_name] = {
            "mean_error_mm": float(np.mean(errors_mm)),
            "max_error_mm": float(np.max(errors_mm)),
        }
    worst_loo_mean = max(
        item["mean_error_mm"] for item in leave_one_out.values()
    )
    return StationRobustFit(
        ground_to_image=refined_ground_to_image,
        image_to_ground=direct_image_to_ground,
        full_inliers=np.concatenate(final_masks),
        pose_ground_points_mm=pose_ground,
        pose_undistorted_pixels=pose_pixels,
        valid_station_names=tuple(valid_names),
        station_metrics=station_metrics,
        leave_one_out_metrics=leave_one_out,
        worst_leave_one_out_mean_error_mm=float(worst_loo_mean),
    )


def estimate_planar_pose(
    ground_points_mm: np.ndarray,
    undistorted_pixels: np.ndarray,
    new_camera_matrix: np.ndarray,
    *,
    prefer_iterative: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    if (
        ground_points_mm.ndim != 2
        or ground_points_mm.shape[1] != 2
        or undistorted_pixels.ndim != 2
        or undistorted_pixels.shape[1] != 2
        or len(ground_points_mm) != len(undistorted_pixels)
        or len(ground_points_mm) < 4
    ):
        raise ValueError(
            "Planar pose inputs must both have shape (N, 2) with N >= 4 and "
            f"matching lengths, got ground={ground_points_mm.shape}, "
            f"pixels={undistorted_pixels.shape}."
        )
    if new_camera_matrix.shape != (3, 3) or not np.all(
        np.isfinite(new_camera_matrix)
    ):
        raise ValueError(
            "new_camera_matrix must be a finite 3x3 matrix, got "
            f"{new_camera_matrix.shape}."
        )
    if not np.all(np.isfinite(ground_points_mm)) or not np.all(
        np.isfinite(undistorted_pixels)
    ):
        raise ValueError("Planar pose inputs must contain only finite values.")
    centered_ground = ground_points_mm - np.mean(ground_points_mm, axis=0)
    if np.linalg.matrix_rank(centered_ground, tol=1e-9) < 2:
        raise ValueError("Planar pose ground points must not be collinear.")
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

    if prefer_iterative:
        iterative_success, iterative_rvec, iterative_tvec = cv2.solvePnP(
            object_points,
            image_points,
            new_camera_matrix,
            np.zeros((4, 1), dtype=np.float64),
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if iterative_success:
            iterative_rvec = np.asarray(iterative_rvec, dtype=np.float64).reshape(3, 1)
            iterative_tvec = np.asarray(iterative_tvec, dtype=np.float64).reshape(3, 1)
            iterative_rotation, _ = cv2.Rodrigues(iterative_rvec)
            iterative_camera_position = -iterative_rotation.T @ iterative_tvec
            iterative_camera_points = (
                iterative_rotation @ object_points.T + iterative_tvec
            ).T
            if (
                iterative_camera_position[2, 0] > 0.0
                and np.all(iterative_camera_points[:, 2] > 0.0)
            ):
                iterative_projected, _ = cv2.projectPoints(
                    object_points,
                    iterative_rvec,
                    iterative_tvec,
                    new_camera_matrix,
                    np.zeros((4, 1), dtype=np.float64),
                )
                iterative_rmse = float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                (
                                    iterative_projected.reshape(-1, 2)
                                    - undistorted_pixels
                                )
                                ** 2,
                                axis=1,
                            )
                        )
                    )
                )
                candidates.append(
                    (
                        iterative_rmse,
                        iterative_rvec,
                        iterative_tvec,
                        iterative_rotation,
                        iterative_camera_position,
                    )
                )

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
        if physically_valid:
            # IPPE is useful for enumerating the two planar pose branches,
            # but its closed-form result can be numerically weak when many
            # points are far from the object origin.  Refine each physically
            # valid branch against all points before comparing reprojection
            # error.  This also makes multi-position boards behave like the
            # single-board synthetic geometry used by the tests.
            if hasattr(cv2, "solvePnPRefineLM"):
                rvec, tvec = cv2.solvePnPRefineLM(
                    object_points,
                    image_points,
                    new_camera_matrix,
                    np.zeros((4, 1), dtype=np.float64),
                    rvec,
                    tvec,
                )
                rotation, _ = cv2.Rodrigues(rvec)
                camera_position_robot = -rotation.T @ tvec
                camera_points = (rotation @ object_points.T + tvec).T
                physically_valid = (
                    camera_position_robot[2, 0] > 0.0
                    and np.all(camera_points[:, 2] > 0.0)
                )
            if not physically_valid:
                continue
            projected, _ = cv2.projectPoints(
                object_points,
                rvec,
                tvec,
                new_camera_matrix,
                np.zeros((4, 1), dtype=np.float64),
            )
            projected = projected.reshape(-1, 2)
            pixel_rmse = float(
                np.sqrt(
                    np.mean(
                        np.sum((projected - undistorted_pixels) ** 2, axis=1)
                    )
                )
            )
            candidates.append(
                (
                pixel_rmse,
                rvec,
                tvec,
                rotation,
                camera_position_robot,
                )
            )

    if not candidates:
        raise RuntimeError(
            "All planar pose candidates are physically invalid: the camera "
            "must be above the robot ground plane and all calibration points "
            "must be in front of it."
        )
    candidates.sort(key=lambda item: item[0])
    pixel_rmse, rvec, tvec, rotation, camera_position_robot = candidates[0]

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
    ground_scale = float(ground_to_image[2, 2])
    if not np.isfinite(ground_scale) or abs(ground_scale) < 1e-12:
        raise ValueError(
            f"Pose ground homography has invalid normalization scale "
            f"{ground_scale!r}."
        )
    ground_to_image /= ground_scale

    image_to_ground = np.linalg.inv(ground_to_image)
    image_scale = float(image_to_ground[2, 2])
    if not np.isfinite(image_scale) or abs(image_scale) < 1e-12:
        raise ValueError(
            f"Pose image homography has invalid normalization scale "
            f"{image_scale!r}."
        )
    image_to_ground /= image_scale
    return ground_to_image, image_to_ground


def save_board_preview(
    image: np.ndarray,
    undistorted_pixels: np.ndarray,
    output_path: Path,
    *,
    reference_index: int | None = 0,
) -> None:
    """Save an undistorted board overlay, optionally highlighting a reference."""

    preview = image.copy()
    for index, (u, v) in enumerate(undistorted_pixels):
        color = (0, 0, 255) if index == reference_index else (0, 255, 0)
        cv2.circle(preview, (round(float(u)), round(float(v))), 5, color, -1)
        if index == reference_index:
            cv2.putText(
                preview,
                "reference",
                (round(float(u)) + 10, round(float(v)) - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                color,
                2,
                cv2.LINE_AA,
            )
    cv2.imwrite(str(output_path), preview)


def board_reference_diagnostic_fields(
    board: BoardCalibrationSpec,
    spec: BoardImageSpec,
) -> dict[str, Any]:
    """Return the entered station reference in the board's coordinate model."""

    reference = spec.reference_corner_field_mm
    if reference is None:
        reference = spec.reference_inner_corner_field_mm
    if board.board_type == "charuco":
        return {"board_origin_outer_corner_global_mm": list(reference)}
    key = (
        "reference_inner_corner_global_mm"
        if board.reference == "lower_left_inner_corner"
        else "reference_outer_corner_global_mm"
    )
    return {
        key: list(reference),
        "reference_inner_corner_global_mm": list(
            spec.reference_inner_corner_field_mm
        ),
    }


def main() -> None:
    args = parse_args()

    session_dir = args.session.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    board_calibration_path = (
        args.board_calibration.expanduser().resolve()
        if args.board_calibration is not None
        else session_dir / DEFAULT_BOARD_CALIBRATION_NAME
    )

    if args.intrinsics is None:
        intrinsics_path, camera_model, intrinsic_data = (
            load_configured_intrinsics(config_path)
        )
    else:
        intrinsics_path = args.intrinsics.expanduser().resolve()
        calibration, intrinsic_data = load_intrinsics(intrinsics_path)
        camera_model = CameraModel(calibration)
    calibration = camera_model.calibration
    image_size = calibration.image_size
    new_camera_matrix = calibration.new_K

    board = load_board_calibration(board_calibration_path)
    capture_session = validate_capture_session(board, calibration)
    observations: list[BoardObservation] = []
    for spec in board.images:
        try:
            observation = detect_board_observation(
                spec,
                board,
                camera_model,
                max_detection_scale=args.max_detection_scale,
                minimum_charuco_sharpness=args.minimum_charuco_sharpness,
            )
        except (FileNotFoundError, RuntimeError, ValueError) as error:
            raise RuntimeError(
                f"Failed to process board image {spec.image_path}: {error}"
            ) from error
        observations.append(observation)

    fit_observations = [item for item in observations if item.spec.role == "fit"]
    holdout_observations = [
        item for item in observations if item.spec.role == "holdout"
    ]
    if not fit_observations or not holdout_observations:
        raise RuntimeError("Both fit and holdout board images are required")

    fit_undistorted_pixels = np.concatenate(
        [item.undistorted_pixels for item in fit_observations], axis=0
    )
    fit_ground_points_mm = np.concatenate(
        [item.robot_points_mm for item in fit_observations], axis=0
    )

    station_fit = fit_station_robust_homography(
        fit_observations,
        threshold_px=args.ransac_threshold_px,
        minimum_station_inlier_ratio=args.minimum_station_inlier_ratio,
        maximum_corners_per_station=args.maximum_corners_per_station,
    )
    inliers = station_fit.full_inliers

    (
        rvec_robot_to_camera,
        tvec_robot_to_camera,
        rotation_robot_to_camera,
        camera_position_robot,
        pose_reprojection_rmse_px,
    ) = estimate_planar_pose(
        station_fit.pose_ground_points_mm,
        station_fit.pose_undistorted_pixels,
        new_camera_matrix,
        prefer_iterative=board.board_type == "charuco",
    )

    (
        pose_ground_to_image,
        pose_image_to_ground,
    ) = pose_ground_homography(
        rotation_robot_to_camera,
        tvec_robot_to_camera,
        new_camera_matrix,
    )
    ground_to_image = pose_ground_to_image
    image_to_ground = pose_image_to_ground
    ground_errors_mm = np.linalg.norm(
        transform_points(fit_undistorted_pixels, image_to_ground)
        - fit_ground_points_mm,
        axis=1,
    )
    mapping_disagreement = np.linalg.norm(
        transform_points(fit_undistorted_pixels, station_fit.image_to_ground)
        - transform_points(fit_undistorted_pixels, image_to_ground),
        axis=1,
    )
    maximum_mapping_disagreement_mm = float(np.max(mapping_disagreement))

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

    bev_ground_corners = np.asarray(
        [
            [bev_config.x_min, bev_config.y_min],
            [bev_config.x_min, bev_config.y_max],
            [bev_config.x_max, bev_config.y_min],
            [bev_config.x_max, bev_config.y_max],
        ],
        dtype=np.float64,
    )
    bev_corner_pixels = transform_points(bev_ground_corners, ground_to_image)
    bev_mapping_disagreement = np.linalg.norm(
        transform_points(bev_corner_pixels, station_fit.image_to_ground)
        - bev_ground_corners,
        axis=1,
    )
    mapping_disagreement = np.concatenate(
        (mapping_disagreement, bev_mapping_disagreement)
    )
    maximum_mapping_disagreement_mm = float(np.max(mapping_disagreement))

    inlier_errors = ground_errors_mm[inliers]
    mean_inlier_error_mm = float(np.mean(inlier_errors))
    max_inlier_error_mm = float(np.max(inlier_errors))

    holdout_errors: list[float] = []
    holdout_image_metrics: dict[str, dict[str, float]] = {}
    for observation in holdout_observations:
        predicted_ground = transform_points(
            observation.undistorted_pixels,
            image_to_ground,
        )
        errors = np.linalg.norm(
            predicted_ground - observation.robot_points_mm,
            axis=1,
        )
        holdout_errors.extend(float(value) for value in errors)
        holdout_image_metrics[observation.spec.name] = {
            "mean_error_mm": float(np.mean(errors)),
            "median_error_mm": float(np.median(errors)),
            "max_error_mm": float(np.max(errors)),
        }
    holdout_errors_array = np.asarray(holdout_errors, dtype=np.float64)
    holdout_mean_error_mm = float(np.mean(holdout_errors_array))
    holdout_max_error_mm = float(np.max(holdout_errors_array))

    thresholds = {
        "maximum_mean_inlier_error_mm": float(
            args.maximum_mean_inlier_error_mm
        ),
        "maximum_inlier_error_mm": float(args.maximum_inlier_error_mm),
        "maximum_pose_rmse_px": float(args.maximum_pose_rmse_px),
        "maximum_holdout_mean_error_mm": float(
            args.maximum_holdout_mean_error_mm
        ),
        "maximum_holdout_error_mm": float(args.maximum_holdout_error_mm),
        "maximum_leave_one_out_mean_error_mm": float(
            args.maximum_leave_one_out_mean_error_mm
        ),
        "maximum_mapping_disagreement_mm": float(
            args.maximum_mapping_disagreement_mm
        ),
    }
    if any(
        not np.isfinite(value) or value <= 0.0
        for value in thresholds.values()
    ):
        raise ValueError(
            f"Ground mapping quality thresholds must be finite and positive, "
            f"got {thresholds!r}."
        )
    quality_failures = []
    if mean_inlier_error_mm > thresholds["maximum_mean_inlier_error_mm"]:
        quality_failures.append("mean_inlier_error_mm")
    if max_inlier_error_mm > thresholds["maximum_inlier_error_mm"]:
        quality_failures.append("max_inlier_error_mm")
    if pose_reprojection_rmse_px > thresholds["maximum_pose_rmse_px"]:
        quality_failures.append("pose_reprojection_rmse_px")
    if holdout_mean_error_mm > thresholds["maximum_holdout_mean_error_mm"]:
        quality_failures.append("holdout_mean_error_mm")
    if holdout_max_error_mm > thresholds["maximum_holdout_error_mm"]:
        quality_failures.append("holdout_max_error_mm")
    if (
        station_fit.worst_leave_one_out_mean_error_mm
        > thresholds["maximum_leave_one_out_mean_error_mm"]
    ):
        quality_failures.append("leave_one_out_mean_error_mm")
    if (
        maximum_mapping_disagreement_mm
        > thresholds["maximum_mapping_disagreement_mm"]
    ):
        quality_failures.append("mapping_disagreement_mm")

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    result_dir = OUTPUT_DIR / f"{args.output_name}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=False)
    diagnostics_dir = result_dir / "diagnostics"
    diagnostics_dir.mkdir()

    preview_observation = fit_observations[0]
    bev = cv2.warpPerspective(
        preview_observation.undistorted_image,
        image_to_bev,
        (bev_width, bev_height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
    )

    cv2.imwrite(
        str(diagnostics_dir / "bev_preview.png"),
        bev,
    )
    per_image = []
    per_point = []
    fit_offset = 0
    for observation in fit_observations:
        start = fit_offset
        end = start + len(observation.raw_pixels)
        image_inliers = inliers[start:end]
        image_errors = ground_errors_mm[start:end]
        fit_offset = end
        per_image.append(
            {
                "name": observation.spec.name,
                "image": str(observation.spec.image_path),
                "role": observation.spec.role,
                **board_reference_diagnostic_fields(board, observation.spec),
                "sharpness_laplacian_variance": observation.sharpness,
                "corner_count": len(observation.raw_pixels),
                "homography_inlier_count": int(np.count_nonzero(image_inliers)),
                "homography_mean_error_mm": float(np.mean(image_errors)),
                "homography_max_error_mm": float(np.max(image_errors)),
            }
        )
        corner_id_values = (
            observation.corner_ids
            if observation.corner_ids is not None
            else np.full(len(observation.raw_pixels), -1, dtype=np.int64)
        )
        for corner_index, (raw, undistorted, field, robot, image_inlier, error, corner_id) in enumerate(
            zip(
                observation.raw_pixels,
                observation.undistorted_pixels,
                observation.field_points_mm,
                observation.robot_points_mm,
                image_inliers,
                image_errors,
                corner_id_values,
                strict=True,
            )
        ):
            point_record = {
                    "image": observation.spec.name,
                    "corner_index": corner_index,
                    "pixel_raw": raw.tolist(),
                    "pixel_undistorted": undistorted.tolist(),
                    "field_global_mm": field.tolist(),
                    "ground_robot_mm": robot.tolist(),
                    "homography_inlier": bool(image_inlier),
                    "ground_error_mm": float(error),
                }
            if observation.corner_ids is not None:
                point_record["board_corner_id"] = int(corner_id)
            per_point.append(point_record)

    for observation in holdout_observations:
        errors = np.linalg.norm(
            transform_points(observation.undistorted_pixels, image_to_ground)
            - observation.robot_points_mm,
            axis=1,
        )
        per_image.append(
            {
                "name": observation.spec.name,
                "image": str(observation.spec.image_path),
                "role": observation.spec.role,
                **board_reference_diagnostic_fields(board, observation.spec),
                "sharpness_laplacian_variance": observation.sharpness,
                "corner_count": len(observation.raw_pixels),
                "ground_mean_error_mm": float(np.mean(errors)),
                "ground_median_error_mm": float(np.median(errors)),
                "ground_max_error_mm": float(np.max(errors)),
            }
        )
        corner_id_values = (
            observation.corner_ids
            if observation.corner_ids is not None
            else np.full(len(observation.raw_pixels), -1, dtype=np.int64)
        )
        for corner_index, (raw, undistorted, field, robot, error, corner_id) in enumerate(
            zip(
                observation.raw_pixels,
                observation.undistorted_pixels,
                observation.field_points_mm,
                observation.robot_points_mm,
                errors,
                corner_id_values,
                strict=True,
            )
        ):
            point_record = {
                    "image": observation.spec.name,
                    "corner_index": corner_index,
                    "pixel_raw": raw.tolist(),
                    "pixel_undistorted": undistorted.tolist(),
                    "field_global_mm": field.tolist(),
                    "ground_robot_mm": robot.tolist(),
                    "holdout_ground_error_mm": float(error),
                }
            if observation.corner_ids is not None:
                point_record["board_corner_id"] = int(corner_id)
            per_point.append(point_record)

    for index, observation in enumerate(observations, start=1):
        save_board_preview(
            observation.undistorted_image,
            observation.undistorted_pixels,
            diagnostics_dir / f"{index:02d}_{observation.spec.name}_corners.png",
            reference_index=None if board.board_type == "charuco" else 0,
        )

    board_diagnostics: dict[str, Any] = {
        "board_type": board.board_type,
        "square_size_mm": board.square_size_mm,
        "reference": board.reference,
        "edge_margin_mm": {
            "left": board.edge_margin_mm[0],
            "right": board.edge_margin_mm[1],
            "bottom": board.edge_margin_mm[2],
            "top": board.edge_margin_mm[3],
        },
    }
    if board.board_type == "charuco":
        board_diagnostics.update(
            {
                "chessboard_size_squares": list(board.chessboard_size_squares),
                "marker_size_mm": board.marker_size_mm,
                "dictionary": board.dictionary_name,
                "minimum_charuco_corners": board.minimum_charuco_corners,
                "printed_face": "camera",
                "board_rotation_degrees": board.board_rotation_degrees,
            }
        )
    else:
        board_diagnostics.update(
            {
                "pattern_size_internal_corners": list(PATTERN_SIZE),
                "physical_square_count": list(PHYSICAL_SQUARE_COUNT),
                "reference_corner_marked": True,
                "detected_corner_order": board.detected_corner_order,
            }
        )

    quality = {
        "usable": not quality_failures,
        "physically_valid": True,
        "thresholds": thresholds,
        "failures": quality_failures,
    }
    result = {
        "calibration_id": calibration.calibration_id,
        "model_type": calibration.model.value,
        "image_size": list(image_size),
        "coordinate_frame": robot_frame_metadata(),
        "image_to_ground": image_to_ground.tolist(),
        "quality": quality,
        "extrinsics": {
            "rotation_robot_to_camera": rotation_robot_to_camera.tolist(),
            "translation_robot_to_camera_mm": (
                tvec_robot_to_camera.reshape(3).tolist()
            ),
        },
        "bev": {
            "x_min_mm": float(args.bev_x_min_mm),
            "x_max_mm": float(args.bev_x_max_mm),
            "y_min_mm": float(args.bev_y_min_mm),
            "y_max_mm": float(args.bev_y_max_mm),
            "mm_per_pixel": float(args.bev_mm_per_pixel),
        },
    }
    diagnostics = {
        "calibration_id": calibration.calibration_id,
        "model_type": calibration.model.value,
        "image_size": list(image_size),
        "source": {
            "intrinsics": str(intrinsics_path),
            "board_calibration": str(board.source_path),
            "capture_session": str(board.source_path.parent / "session.json"),
        },
        "field_calibration_frame": ground_calibration_frame_metadata(),
        "board": board_diagnostics,
        "images": per_image,
        "coordinate_frames": {
            "robot": robot_frame_metadata(),
            "camera_opencv": {
                "x": "right",
                "y": "down",
                "z": "forward",
                "unit": "mm",
            },
            "image": "undistorted pixels produced with new_camera_matrix",
        },
        "intrinsics": {
            "quality": intrinsic_data.get("quality"),
            "lens_position": calibration.lens_position,
            "capture_lens_position": capture_session["lens_position"],
        },
        "homography": {
            "role": "station_balanced_outlier_filter_and_initialization",
            "ransac_threshold_px": float(args.ransac_threshold_px),
            "inlier_count": int(np.count_nonzero(inliers)),
            "total_count": int(len(fit_undistorted_pixels)),
            "mean_inlier_error_mm": mean_inlier_error_mm,
            "median_inlier_error_mm": float(np.median(inlier_errors)),
            "max_inlier_error_mm": max_inlier_error_mm,
            "direct_image_to_ground": station_fit.image_to_ground.tolist(),
            "valid_station_names": list(station_fit.valid_station_names),
            "per_station": station_fit.station_metrics,
            "leave_one_station_out": station_fit.leave_one_out_metrics,
            "worst_leave_one_out_mean_error_mm": (
                station_fit.worst_leave_one_out_mean_error_mm
            ),
            "physical_mapping_disagreement_mean_mm": float(
                np.mean(mapping_disagreement)
            ),
            "physical_mapping_disagreement_max_mm": (
                maximum_mapping_disagreement_mm
            ),
        },
        "holdout": {
            "image_count": len(holdout_observations),
            "corner_count": len(holdout_errors_array),
            "mean_error_mm": holdout_mean_error_mm,
            "median_error_mm": float(np.median(holdout_errors_array)),
            "max_error_mm": holdout_max_error_mm,
            "per_image": holdout_image_metrics,
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
            "physically_valid": True,
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
            "ground_to_bev": ground_to_bev.tolist(),
            "image_to_bev": image_to_bev.tolist(),
        },
        "points": per_point,
    }

    json_path = result_dir / f"{args.output_name}.json"
    npz_path = result_dir / f"{args.output_name}.npz"

    save_json(json_path, result)
    save_json(
        diagnostics_dir / "ground_mapping_diagnostics.json",
        diagnostics,
    )
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
    print(f"Holdout mean error: {holdout_mean_error_mm:.2f} mm")
    print(f"Holdout max error:  {holdout_max_error_mm:.2f} mm")
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
