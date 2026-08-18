"""ChArUco board construction, detection and ground-point expansion."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any
from collections.abc import Sequence

import cv2
import numpy as np


@dataclass(frozen=True, slots=True)
class CharucoDetection:
    """Visible ChArUco corners and their stable board IDs."""

    charuco_corners: np.ndarray
    charuco_ids: np.ndarray
    marker_corners: tuple[np.ndarray, ...]
    marker_ids: np.ndarray
    detection_scale: float


def dictionary_from_name(name: str) -> Any:
    """Resolve an OpenCV ArUco dictionary name without importing aruco eagerly."""

    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "This OpenCV build has no cv2.aruco module; install an OpenCV "
            "contrib build before using ChArUco calibration."
        )
    value = getattr(aruco, name, None)
    if value is None or not name.startswith("DICT_"):
        raise ValueError(f"Unknown ArUco dictionary {name!r}.")
    return aruco.getPredefinedDictionary(value)


def create_charuco_board(
    chessboard_size_squares: tuple[int, int],
    square_size_mm: float,
    marker_size_mm: float,
    dictionary_name: str,
) -> Any:
    """Create the physical board model used by both capture and solving."""

    columns, rows = chessboard_size_squares
    if columns < 2 or rows < 2:
        raise ValueError(
            f"chessboard_size_squares must be >=2 in both axes, got "
            f"{chessboard_size_squares!r}."
        )
    if not math.isfinite(square_size_mm) or square_size_mm <= 0.0:
        raise ValueError(f"square_size_mm must be positive, got {square_size_mm!r}.")
    if not math.isfinite(marker_size_mm) or marker_size_mm <= 0.0:
        raise ValueError(f"marker_size_mm must be positive, got {marker_size_mm!r}.")
    if marker_size_mm >= square_size_mm:
        raise ValueError(
            f"marker_size_mm must be smaller than square_size_mm, got "
            f"marker={marker_size_mm!r}, square={square_size_mm!r}."
        )
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "CharucoBoard"):
        raise RuntimeError(
            "This OpenCV build does not provide cv2.aruco.CharucoBoard."
        )
    return aruco.CharucoBoard(
        (int(columns), int(rows)),
        float(square_size_mm),
        float(marker_size_mm),
        dictionary_from_name(dictionary_name),
    )


def detect_charuco_board(
    image: np.ndarray,
    board: Any,
    *,
    minimum_corners: int = 8,
    minimum_grid_rows: int = 3,
    minimum_grid_columns: int = 3,
    max_detection_scale: float = 2.0,
    camera_matrix: np.ndarray | None = None,
    distortion: np.ndarray | None = None,
) -> CharucoDetection | None:
    """Detect a spatially useful partial ChArUco board at one or more scales."""

    if image.ndim not in {2, 3} or image.size == 0:
        raise ValueError(f"image must be a non-empty 2D/3D array, got {image.shape}.")
    if minimum_corners < 4:
        raise ValueError(f"minimum_corners must be >=4, got {minimum_corners}.")
    if minimum_grid_rows < 2 or minimum_grid_columns < 2:
        raise ValueError("minimum_grid_rows and minimum_grid_columns must be >=2.")
    if not math.isfinite(max_detection_scale) or max_detection_scale < 1.0:
        raise ValueError(
            f"max_detection_scale must be finite and >=1, got {max_detection_scale!r}."
        )
    aruco = getattr(cv2, "aruco", None)
    if aruco is None or not hasattr(aruco, "CharucoDetector"):
        raise RuntimeError(
            "This OpenCV build does not provide cv2.aruco.CharucoDetector."
        )

    if (camera_matrix is None) != (distortion is None):
        raise ValueError("camera_matrix and distortion must be provided together.")
    if camera_matrix is not None:
        camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        distortion = np.asarray(distortion, dtype=np.float64)
        if camera_matrix.shape != (3, 3) or not np.all(np.isfinite(camera_matrix)):
            raise ValueError("camera_matrix must be a finite 3x3 matrix.")
        if distortion.size == 0 or not np.all(np.isfinite(distortion)):
            raise ValueError("distortion must contain finite coefficients.")
    scales = [1.0]
    if max_detection_scale > 1.0:
        scales.append(float(max_detection_scale))
    best: CharucoDetection | None = None
    chessboard_columns, _chessboard_rows = board.getChessboardSize()
    corner_columns = int(chessboard_columns) - 1
    for scale in scales:
        detector_parameters = aruco.DetectorParameters()
        detector_parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        charuco_parameters = aruco.CharucoParameters()
        charuco_parameters.tryRefineMarkers = True
        if camera_matrix is not None:
            scaled_camera_matrix = camera_matrix.copy()
            scaled_camera_matrix[:2, :] *= scale
            charuco_parameters.cameraMatrix = scaled_camera_matrix
            assert distortion is not None
            charuco_parameters.distCoeffs = distortion
        detector = aruco.CharucoDetector(
            board,
            charuco_parameters,
            detector_parameters,
        )
        detected_image = image
        if scale != 1.0:
            detected_image = cv2.resize(
                image,
                None,
                fx=scale,
                fy=scale,
                interpolation=cv2.INTER_CUBIC,
            )
        charuco_corners, charuco_ids, marker_corners, marker_ids = (
            detector.detectBoard(detected_image)
        )
        if charuco_corners is None or charuco_ids is None:
            continue
        corners = (
            np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2) / scale
        )
        ids = np.asarray(charuco_ids, dtype=np.int64).reshape(-1)
        if len(corners) != len(ids) or len(corners) < minimum_corners:
            continue
        if not np.all(np.isfinite(corners)) or len(np.unique(ids)) != len(ids):
            continue
        rows = ids // corner_columns
        columns = ids % corner_columns
        if (
            len(np.unique(rows)) < minimum_grid_rows
            or len(np.unique(columns)) < minimum_grid_columns
        ):
            continue
        centered = corners - np.mean(corners, axis=0)
        if np.linalg.matrix_rank(centered, tol=1e-6) < 2:
            continue
        marker_values = () if marker_corners is None else tuple(
            np.asarray(item, dtype=np.float64) / scale for item in marker_corners
        )
        marker_id_values = (
            np.empty((0,), dtype=np.int64)
            if marker_ids is None
            else np.asarray(marker_ids, dtype=np.int64).reshape(-1)
        )
        candidate = CharucoDetection(
            charuco_corners=corners,
            charuco_ids=ids,
            marker_corners=marker_values,
            marker_ids=marker_id_values,
            detection_scale=scale,
        )
        if best is None or len(candidate.charuco_ids) > len(best.charuco_ids):
            best = candidate
    return best


def charuco_detection_jitter_px(
    detections: Sequence[CharucoDetection],
    *,
    minimum_common_corners: int,
) -> float:
    """Return worst common-corner displacement from the temporal median."""

    if len(detections) < 2:
        raise ValueError("At least two detections are required for stability checking.")
    common_ids = set(int(value) for value in detections[0].charuco_ids)
    for detection in detections[1:]:
        common_ids.intersection_update(int(value) for value in detection.charuco_ids)
    if len(common_ids) < minimum_common_corners:
        return math.inf
    ordered_ids = sorted(common_ids)
    frames = []
    for detection in detections:
        points_by_id = {
            int(corner_id): corner
            for corner_id, corner in zip(
                detection.charuco_ids,
                detection.charuco_corners,
                strict=True,
            )
        }
        frames.append(np.asarray([points_by_id[value] for value in ordered_ids]))
    values = np.asarray(frames, dtype=np.float64)
    median = np.median(values, axis=0)
    return float(np.max(np.linalg.norm(values - median, axis=2)))


def charuco_points_field_mm(
    board: Any,
    charuco_ids: np.ndarray,
    board_origin_outer_corner_global_mm: tuple[float, float],
    edge_margin_mm: tuple[float, float, float, float],
    board_rotation_degrees: int,
) -> np.ndarray:
    """Map IDs through the explicitly oriented OpenCV board coordinate frame."""

    ids = np.asarray(charuco_ids, dtype=np.int64).reshape(-1)
    board_points = np.asarray(board.getChessboardCorners(), dtype=np.float64)
    if board_points.ndim != 2 or board_points.shape[1] != 3:
        raise ValueError(f"Charuco board points must have shape (N, 3), got {board_points.shape}.")
    if len(ids) == 0 or np.any(ids < 0) or np.any(ids >= len(board_points)):
        raise ValueError(f"Charuco IDs are outside board range, got {ids.tolist()!r}.")
    outer = np.asarray(board_origin_outer_corner_global_mm, dtype=np.float64)
    if outer.shape != (2,) or not np.all(np.isfinite(outer)):
        raise ValueError(
            "board_origin_outer_corner_global_mm must contain two finite values."
        )
    if board_rotation_degrees not in {0, 90, 180, 270}:
        raise ValueError(
            "board_rotation_degrees must be one of 0, 90, 180 or 270, got "
            f"{board_rotation_degrees!r}."
        )
    left, _right, bottom, _top = edge_margin_mm
    local_points = board_points[ids, :2] + np.array(
        [left, bottom], dtype=np.float64
    )
    angle = math.radians(board_rotation_degrees)
    rotation = np.array(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=np.float64,
    )
    return outer + local_points @ rotation.T
