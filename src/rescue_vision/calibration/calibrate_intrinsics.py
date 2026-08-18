#!/usr/bin/env python3
"""比较三种 OpenCV 相机模型并输出最优内参。

候选模型：
1. pinhole：标准 5 参数针孔畸变模型；
2. pinhole_rational：Rational 扩展针孔模型；
3. fisheye：OpenCV Fisheye 4 参数模型。

模型选择使用 K 折交叉验证。每一折只用训练图求相机内参，再在未参与
求参的验证图上重新估计棋盘外参并计算原始畸变像素上的重投影误差。
最终按验证集全局 RMSE 选择模型，然后用全部有效图片重新计算最终参数。

目录均相对于本文件：

    calibration_captures/
        chessboard_2304x1296_YYYYMMDD_HHMMSS/
            images/*.png
            session.json

    output/
        intrinsics_YYYYMMDD_HHMMSS/
            comparison.json
            selected_calibration.json
            selected_calibration.npz
            models/*.json
            diagnostics/*
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from rescue_vision.geometry.camera_model import IMAGE_BORDER_FILL_VALUE


CALIBRATION_DIR = Path(__file__).resolve().parent
CAPTURES_DIR = CALIBRATION_DIR / "calibration_captures"
OUTPUT_ROOT = CALIBRATION_DIR / "output"

IMAGE_SIZE = (2304, 1296)  # (width, height)
PATTERN_SIZE = (11, 8)      # internal corners
PHYSICAL_SQUARE_COUNT = (12, 9)
MINIMUM_VIEWS = 15
MINIMUM_TRAIN_VIEWS = 10

MODEL_PINHOLE = "pinhole"
MODEL_PINHOLE_RATIONAL = "pinhole_rational"
MODEL_FISHEYE = "fisheye"
MODEL_ORDER = (
    MODEL_PINHOLE,
    MODEL_PINHOLE_RATIONAL,
    MODEL_FISHEYE,
)

CHESSBOARD_FLAGS = (
    cv2.CALIB_CB_NORMALIZE_IMAGE
    | cv2.CALIB_CB_EXHAUSTIVE
    | cv2.CALIB_CB_ACCURACY
)

CALIBRATION_CRITERIA = (
    cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
    300,
    1e-9,
)


@dataclass(slots=True)
class CalibrationFit:
    model_type: str
    rms_px: float
    K: np.ndarray
    D: np.ndarray
    rvecs: tuple[np.ndarray, ...]
    tvecs: tuple[np.ndarray, ...]
    solver_variant: str


@dataclass(slots=True)
class ValidationResult:
    image_index: int
    rmse_px: float | None
    error_message: str | None


@dataclass(slots=True)
class CandidateSummary:
    model_type: str
    fold_records: list[dict[str, Any]]
    validation_results: list[ValidationResult]

    @property
    def successful_folds(self) -> int:
        return sum(record["status"] == "success" for record in self.fold_records)

    @property
    def total_folds(self) -> int:
        return len(self.fold_records)

    @property
    def successful_validation_views(self) -> int:
        return sum(result.rmse_px is not None for result in self.validation_results)

    @property
    def total_validation_views(self) -> int:
        return len(self.validation_results)

    def successful_errors(self) -> np.ndarray:
        return np.asarray(
            [
                result.rmse_px
                for result in self.validation_results
                if result.rmse_px is not None
            ],
            dtype=np.float64,
        )

    def metrics(self) -> dict[str, Any]:
        errors = self.successful_errors()
        if errors.size == 0:
            global_rmse = math.inf
            mean_rmse = math.inf
            median_rmse = math.inf
            max_rmse = math.inf
        else:
            global_rmse = float(np.sqrt(np.mean(np.square(errors))))
            mean_rmse = float(np.mean(errors))
            median_rmse = float(np.median(errors))
            max_rmse = float(np.max(errors))

        success_rate = (
            self.successful_validation_views / self.total_validation_views
            if self.total_validation_views
            else 0.0
        )

        return {
            "successful_folds": self.successful_folds,
            "total_folds": self.total_folds,
            "successful_validation_views": self.successful_validation_views,
            "total_validation_views": self.total_validation_views,
            "validation_pose_success_rate": success_rate,
            "cv_global_rmse_px": global_rmse,
            "cv_mean_view_rmse_px": mean_rmse,
            "cv_median_view_rmse_px": median_rmse,
            "cv_max_view_rmse_px": max_rmse,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare pinhole, rational and fisheye camera models."
    )
    parser.add_argument(
        "--session",
        type=Path,
        default=None,
        help="Capture session. Defaults to the latest 2304x1296 session.",
    )
    parser.add_argument(
        "--square-size-mm",
        type=float,
        default=15.0,
        help="Measured chessboard square side length in millimetres.",
    )
    parser.add_argument(
        "--detection-scale",
        type=float,
        default=0.5,
        help=(
            "Chessboard detection scale in (0, 1]. Corners are refined on "
            "the full-resolution image; default: 0.5."
        ),
    )
    parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of cross-validation folds, default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
        help="Deterministic cross-validation shuffle seed.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.35,
        help="Pinhole getOptimalNewCameraMatrix alpha in [0, 1].",
    )
    parser.add_argument(
        "--balance",
        type=float,
        default=0.35,
        help="Fisheye undistortion balance in [0, 1].",
    )
    parser.add_argument(
        "--minimum-pose-success-rate",
        type=float,
        default=0.95,
        help="Minimum validation pose success rate for a model to be eligible.",
    )
    parser.add_argument(
        "--maximum-usable-cv-rmse",
        type=float,
        default=2.0,
        help="Selected result is marked unusable above this validation RMSE.",
    )
    parser.add_argument(
        "--output-prefix",
        default="intrinsics",
        help="Prefix of the timestamped output directory.",
    )
    parser.add_argument(
        "--calibration-id",
        default=None,
        help="Readable calibration identity; defaults to the output directory name.",
    )
    return parser.parse_args()


def find_latest_session() -> Path:
    sessions = sorted(
        path
        for path in CAPTURES_DIR.glob("chessboard_2304x1296_*")
        if path.is_dir()
    )
    if not sessions:
        raise FileNotFoundError(f"No capture session found under {CAPTURES_DIR}")
    return sessions[-1]


def create_run_directory(prefix: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = OUTPUT_ROOT / f"{prefix}_{stamp}"
    suffix = 1
    while run_dir.exists():
        run_dir = OUTPUT_ROOT / f"{prefix}_{stamp}_{suffix:02d}"
        suffix += 1

    (run_dir / "models").mkdir(parents=True)
    (run_dir / "diagnostics").mkdir()
    return run_dir


def collect_image_paths(session_dir: Path) -> list[Path]:
    images_dir = session_dir / "images"
    if not images_dir.is_dir():
        images_dir = session_dir

    paths: list[Path] = []
    for suffix in ("*.png", "*.jpg", "*.jpeg"):
        paths.extend(images_dir.glob(suffix))
    return sorted(paths)


def load_session_metadata(session_dir: Path) -> dict[str, Any]:
    path = session_dir / "session.json"
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def create_object_template(square_size_mm: float) -> np.ndarray:
    columns, rows = PATTERN_SIZE
    points = np.zeros((columns * rows, 3), dtype=np.float64)
    points[:, :2] = (
        np.mgrid[0:columns, 0:rows]
        .T.reshape(-1, 2)
        .astype(np.float64)
        * square_size_mm
    )
    return np.ascontiguousarray(points)


def detect_corners(
    image: np.ndarray,
    detection_scale: float,
) -> np.ndarray | None:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if detection_scale < 1.0:
        detection_image = cv2.resize(
            gray,
            None,
            fx=detection_scale,
            fy=detection_scale,
            interpolation=cv2.INTER_AREA,
        )
    else:
        detection_image = gray

    found, corners = cv2.findChessboardCornersSB(
        detection_image,
        PATTERN_SIZE,
        flags=CHESSBOARD_FLAGS,
    )

    # Very small or strongly distorted boards can fail after downscaling. In
    # that case retry once at the original resolution.
    if not found and detection_scale < 1.0:
        found, corners = cv2.findChessboardCornersSB(
            gray,
            PATTERN_SIZE,
            flags=CHESSBOARD_FLAGS,
        )
        detected_at_full_resolution = True
    else:
        detected_at_full_resolution = detection_scale == 1.0

    if not found:
        return None

    corners = np.ascontiguousarray(corners.reshape(-1, 1, 2), dtype=np.float32)
    if not detected_at_full_resolution:
        corners /= float(detection_scale)

    # Refine on the original 2304x1296 image so the calibration does not lose
    # precision merely because detection was accelerated on a smaller image.
    cv2.cornerSubPix(
        gray,
        corners,
        winSize=(5, 5),
        zeroZone=(-1, -1),
        criteria=(
            cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_MAX_ITER,
            50,
            1e-4,
        ),
    )

    return np.ascontiguousarray(corners.reshape(-1, 2), dtype=np.float64)


def load_observations(
    image_paths: list[Path],
    object_template: np.ndarray,
    detected_dir: Path,
    detection_scale: float,
) -> tuple[list[np.ndarray], list[np.ndarray], list[Path], list[dict[str, str]]]:
    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    valid_paths: list[Path] = []
    rejected: list[dict[str, str]] = []

    detected_dir.mkdir(parents=True, exist_ok=True)

    for image_path in image_paths:
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            rejected.append({"image": image_path.name, "reason": "read_failed"})
            continue

        height, width = image.shape[:2]
        if (width, height) != IMAGE_SIZE:
            rejected.append(
                {
                    "image": image_path.name,
                    "reason": f"wrong_size_{width}x{height}",
                }
            )
            continue

        corners = detect_corners(image, detection_scale)
        if corners is None:
            rejected.append(
                {"image": image_path.name, "reason": "corners_not_found"}
            )
            continue

        preview = image.copy()
        drawable = np.ascontiguousarray(
            corners.reshape(-1, 1, 2),
            dtype=np.float32,
        )
        cv2.drawChessboardCorners(preview, PATTERN_SIZE, drawable, True)
        cv2.imwrite(str(detected_dir / image_path.name), preview)

        object_points.append(object_template.copy())
        image_points.append(corners)
        valid_paths.append(image_path)

    return object_points, image_points, valid_paths, rejected


def pinhole_inputs(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    objects = [
        np.ascontiguousarray(points, dtype=np.float32)
        for points in object_points
    ]
    images = [
        np.ascontiguousarray(points.reshape(-1, 1, 2), dtype=np.float32)
        for points in image_points
    ]
    return objects, images


def fisheye_inputs(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    objects = [
        np.ascontiguousarray(points.reshape(1, -1, 3), dtype=np.float64)
        for points in object_points
    ]
    images = [
        np.ascontiguousarray(points.reshape(1, -1, 2), dtype=np.float64)
        for points in image_points
    ]
    return objects, images


def fit_pinhole(
    model_type: str,
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> CalibrationFit:
    objects, images = pinhole_inputs(object_points, image_points)

    flags = 0
    if model_type == MODEL_PINHOLE_RATIONAL:
        flags |= cv2.CALIB_RATIONAL_MODEL

    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(
        objects,
        images,
        IMAGE_SIZE,
        None,
        None,
        flags=flags,
        criteria=CALIBRATION_CRITERIA,
    )

    if not np.isfinite(rms) or not np.all(np.isfinite(K)) or not np.all(np.isfinite(D)):
        raise RuntimeError(f"{model_type} returned non-finite parameters.")

    return CalibrationFit(
        model_type=model_type,
        rms_px=float(rms),
        K=np.asarray(K, dtype=np.float64),
        D=np.asarray(D, dtype=np.float64).reshape(-1, 1),
        rvecs=tuple(np.asarray(value, dtype=np.float64) for value in rvecs),
        tvecs=tuple(np.asarray(value, dtype=np.float64) for value in tvecs),
        solver_variant="cv2.calibrateCamera",
    )


def initial_fisheye_K(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    pinhole_guess: np.ndarray | None,
) -> np.ndarray:
    width, height = IMAGE_SIZE

    if pinhole_guess is not None and np.all(np.isfinite(pinhole_guess)):
        K = np.asarray(pinhole_guess, dtype=np.float64).copy()
    else:
        objects, images = pinhole_inputs(object_points, image_points)
        K = cv2.initCameraMatrix2D(objects, images, IMAGE_SIZE, aspectRatio=1.0)
        K = np.asarray(K, dtype=np.float64)

    # Keep the initial principal point inside the image and avoid extreme focal
    # values that make OpenCV's fisheye extrinsic initializer singular.
    diagonal_radius = 0.5 * math.hypot(width, height)
    nominal_focal = diagonal_radius / math.radians(60.0)

    fx = float(K[0, 0]) if np.isfinite(K[0, 0]) else nominal_focal
    fy = float(K[1, 1]) if np.isfinite(K[1, 1]) else nominal_focal
    focal = math.sqrt(max(fx, 1.0) * max(fy, 1.0))
    focal = float(np.clip(focal, 0.25 * width, 2.5 * width))

    return np.array(
        [
            [focal, 0.0, width / 2.0],
            [0.0, focal, height / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def fit_fisheye(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    pinhole_guess: np.ndarray | None,
) -> CalibrationFit:
    objects, images = fisheye_inputs(object_points, image_points)
    K0 = initial_fisheye_K(object_points, image_points, pinhole_guess)
    D0 = np.zeros((4, 1), dtype=np.float64)

    attempts = (
        (
            "recompute_and_condition_check",
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            | cv2.fisheye.CALIB_FIX_SKEW
            | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
            | cv2.fisheye.CALIB_CHECK_COND,
        ),
        (
            "recompute_without_condition_check",
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            | cv2.fisheye.CALIB_FIX_SKEW
            | cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC,
        ),
        (
            "fixed_extrinsic_initialization",
            cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
            | cv2.fisheye.CALIB_FIX_SKEW,
        ),
    )

    successful: list[CalibrationFit] = []
    errors: list[str] = []

    for variant, flags in attempts:
        try:
            K = K0.copy()
            D = D0.copy()
            rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
                objects,
                images,
                IMAGE_SIZE,
                K,
                D,
                flags=flags,
                criteria=CALIBRATION_CRITERIA,
            )

            fit = CalibrationFit(
                model_type=MODEL_FISHEYE,
                rms_px=float(rms),
                K=np.asarray(K, dtype=np.float64),
                D=np.asarray(D, dtype=np.float64).reshape(4, 1),
                rvecs=tuple(np.asarray(value, dtype=np.float64) for value in rvecs),
                tvecs=tuple(np.asarray(value, dtype=np.float64) for value in tvecs),
                solver_variant=variant,
            )

            if (
                np.isfinite(fit.rms_px)
                and np.all(np.isfinite(fit.K))
                and np.all(np.isfinite(fit.D))
            ):
                successful.append(fit)
        except (cv2.error, RuntimeError) as error:
            errors.append(f"{variant}: {error}")

    if not successful:
        joined = "\n".join(errors)
        raise RuntimeError(f"All fisheye calibration attempts failed:\n{joined}")

    # Training RMS is used only to choose among solver variants of the same
    # fisheye model. Cross-validation still decides between camera models.
    return min(successful, key=lambda fit: fit.rms_px)


def fit_model(
    model_type: str,
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    pinhole_guess: np.ndarray | None = None,
) -> CalibrationFit:
    if model_type in {MODEL_PINHOLE, MODEL_PINHOLE_RATIONAL}:
        return fit_pinhole(model_type, object_points, image_points)
    if model_type == MODEL_FISHEYE:
        return fit_fisheye(object_points, image_points, pinhole_guess)
    raise ValueError(f"Unsupported model: {model_type}")


def project_points(
    fit: CalibrationFit,
    object_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
) -> np.ndarray:
    if fit.model_type == MODEL_FISHEYE:
        projected, _ = cv2.fisheye.projectPoints(
            np.ascontiguousarray(object_points.reshape(1, -1, 3), dtype=np.float64),
            rvec,
            tvec,
            fit.K,
            fit.D.reshape(4, 1),
        )
    else:
        projected, _ = cv2.projectPoints(
            np.ascontiguousarray(object_points, dtype=np.float64),
            rvec,
            tvec,
            fit.K,
            fit.D,
        )
    return projected.reshape(-1, 2)


def solve_validation_pose(
    fit: CalibrationFit,
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    obj = np.ascontiguousarray(object_points, dtype=np.float64)
    observed = np.ascontiguousarray(image_points.reshape(-1, 1, 2), dtype=np.float64)

    if fit.model_type == MODEL_FISHEYE:
        pose_points = cv2.fisheye.undistortPoints(
            observed,
            fit.K,
            fit.D.reshape(4, 1),
            R=np.eye(3, dtype=np.float64),
            P=fit.K,
        )
        pose_D = np.zeros((4, 1), dtype=np.float64)
    else:
        pose_points = observed
        pose_D = fit.D

    success, rvec, tvec = cv2.solvePnP(
        obj,
        pose_points,
        fit.K,
        pose_D,
        flags=cv2.SOLVEPNP_IPPE,
    )

    if not success:
        success, rvec, tvec = cv2.solvePnP(
            obj,
            pose_points,
            fit.K,
            pose_D,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )

    if not success:
        raise RuntimeError("solvePnP failed")

    if hasattr(cv2, "solvePnPRefineLM"):
        rvec, tvec = cv2.solvePnPRefineLM(
            obj,
            pose_points,
            fit.K,
            pose_D,
            rvec,
            tvec,
        )

    return rvec, tvec


def view_rmse(observed: np.ndarray, projected: np.ndarray) -> float:
    residuals = observed.reshape(-1, 2) - projected.reshape(-1, 2)
    squared_distance = np.sum(np.square(residuals), axis=1)
    return float(np.sqrt(np.mean(squared_distance)))


def evaluate_validation_view(
    fit: CalibrationFit,
    object_points: np.ndarray,
    image_points: np.ndarray,
) -> float:
    rvec, tvec = solve_validation_pose(fit, object_points, image_points)
    projected = project_points(fit, object_points, rvec, tvec)
    return view_rmse(image_points, projected)


def make_folds(view_count: int, fold_count: int, seed: int) -> list[np.ndarray]:
    if fold_count < 2:
        raise ValueError("At least two folds are required.")
    if view_count < MINIMUM_VIEWS:
        raise ValueError(f"At least {MINIMUM_VIEWS} valid views are required.")

    max_folds = max(2, view_count // 3)
    fold_count = min(fold_count, max_folds, view_count)

    indices = np.arange(view_count)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    return [fold for fold in np.array_split(indices, fold_count) if len(fold)]


def cross_validate(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    image_paths: list[Path],
    folds: list[np.ndarray],
) -> dict[str, CandidateSummary]:
    summaries = {
        model_type: CandidateSummary(model_type, [], [])
        for model_type in MODEL_ORDER
    }

    all_indices = np.arange(len(object_points))

    for fold_number, validation_indices in enumerate(folds, start=1):
        validation_set = set(int(index) for index in validation_indices)
        training_indices = [
            int(index) for index in all_indices if int(index) not in validation_set
        ]

        if len(training_indices) < MINIMUM_TRAIN_VIEWS:
            raise RuntimeError(
                f"Fold {fold_number} leaves only {len(training_indices)} training views; "
                f"need at least {MINIMUM_TRAIN_VIEWS}."
            )

        train_objects = [object_points[index] for index in training_indices]
        train_images = [image_points[index] for index in training_indices]

        fold_fits: dict[str, CalibrationFit] = {}

        for model_type in MODEL_ORDER:
            try:
                pinhole_guess = (
                    fold_fits[MODEL_PINHOLE].K
                    if model_type == MODEL_FISHEYE and MODEL_PINHOLE in fold_fits
                    else None
                )
                fit = fit_model(
                    model_type,
                    train_objects,
                    train_images,
                    pinhole_guess=pinhole_guess,
                )
                fold_fits[model_type] = fit
                summaries[model_type].fold_records.append(
                    {
                        "fold": fold_number,
                        "status": "success",
                        "training_view_count": len(training_indices),
                        "validation_view_count": len(validation_indices),
                        "training_rms_px": fit.rms_px,
                        "solver_variant": fit.solver_variant,
                    }
                )
            except (cv2.error, RuntimeError, ValueError) as error:
                summaries[model_type].fold_records.append(
                    {
                        "fold": fold_number,
                        "status": "failed",
                        "training_view_count": len(training_indices),
                        "validation_view_count": len(validation_indices),
                        "error": str(error),
                    }
                )
                for index in validation_indices:
                    summaries[model_type].validation_results.append(
                        ValidationResult(
                            image_index=int(index),
                            rmse_px=None,
                            error_message="calibration_failed",
                        )
                    )
                continue

            fit = fold_fits[model_type]
            for index in validation_indices:
                image_index = int(index)
                try:
                    rmse = evaluate_validation_view(
                        fit,
                        object_points[image_index],
                        image_points[image_index],
                    )
                    summaries[model_type].validation_results.append(
                        ValidationResult(image_index, rmse, None)
                    )
                except (cv2.error, RuntimeError, ValueError) as error:
                    summaries[model_type].validation_results.append(
                        ValidationResult(image_index, None, str(error))
                    )

        print(f"Completed cross-validation fold {fold_number}/{len(folds)}")

    # Keep output deterministic and aligned with filenames.
    for summary in summaries.values():
        summary.validation_results.sort(key=lambda item: item.image_index)

    return summaries


def calculate_fit_view_errors(
    fit: CalibrationFit,
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> list[float]:
    errors: list[float] = []
    for obj, observed, rvec, tvec in zip(
        object_points,
        image_points,
        fit.rvecs,
        fit.tvecs,
        strict=True,
    ):
        projected = project_points(fit, obj, rvec, tvec)
        errors.append(view_rmse(observed, projected))
    return errors


def create_new_camera_matrix(
    fit: CalibrationFit,
    alpha: float,
    balance: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    if fit.model_type == MODEL_FISHEYE:
        new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
            fit.K,
            fit.D.reshape(4, 1),
            IMAGE_SIZE,
            np.eye(3, dtype=np.float64),
            balance=balance,
            new_size=IMAGE_SIZE,
            fov_scale=1.0,
        )
        return np.asarray(new_K, dtype=np.float64), {
            "method": "fisheye_balance",
            "balance": balance,
            "fov_scale": 1.0,
        }

    new_K, roi = cv2.getOptimalNewCameraMatrix(
        fit.K,
        fit.D,
        IMAGE_SIZE,
        alpha,
        IMAGE_SIZE,
    )
    return np.asarray(new_K, dtype=np.float64), {
        "method": "pinhole_alpha",
        "alpha": alpha,
        "valid_roi": [int(value) for value in roi],
    }


def undistort_image(
    fit: CalibrationFit,
    new_K: np.ndarray,
    image: np.ndarray,
) -> np.ndarray:
    identity = np.eye(3, dtype=np.float64)

    if fit.model_type == MODEL_FISHEYE:
        map1, map2 = cv2.fisheye.initUndistortRectifyMap(
            fit.K,
            fit.D.reshape(4, 1),
            identity,
            new_K,
            IMAGE_SIZE,
            cv2.CV_16SC2,
        )
    else:
        map1, map2 = cv2.initUndistortRectifyMap(
            fit.K,
            fit.D,
            identity,
            new_K,
            IMAGE_SIZE,
            cv2.CV_16SC2,
        )

    return cv2.remap(
        image,
        map1,
        map2,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=(IMAGE_BORDER_FILL_VALUE,) * 3,
    )


def distortion_parameter_names(model_type: str, count: int) -> list[str]:
    if model_type == MODEL_FISHEYE:
        return ["k1", "k2", "k3", "k4"][:count]

    names = [
        "k1",
        "k2",
        "p1",
        "p2",
        "k3",
        "k4",
        "k5",
        "k6",
        "s1",
        "s2",
        "s3",
        "s4",
        "tau_x",
        "tau_y",
    ]
    return names[:count]


def finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def summary_to_json(
    summary: CandidateSummary,
    image_paths: list[Path],
) -> dict[str, Any]:
    metrics = summary.metrics()
    validation = []
    for result in summary.validation_results:
        validation.append(
            {
                "image": image_paths[result.image_index].name,
                "rmse_px": result.rmse_px,
                "error": result.error_message,
            }
        )

    serializable_metrics = {
        key: finite_or_none(value) if isinstance(value, float) else value
        for key, value in metrics.items()
    }

    return {
        "model_type": summary.model_type,
        "metrics": serializable_metrics,
        "folds": summary.fold_records,
        "validation_views": validation,
    }


def choose_model(
    summaries: dict[str, CandidateSummary],
    final_fits: dict[str, CalibrationFit],
    minimum_pose_success_rate: float,
) -> tuple[str, list[str]]:
    eligible: list[tuple[float, str]] = []
    reasons: list[str] = []

    for model_type in MODEL_ORDER:
        summary = summaries[model_type]
        metrics = summary.metrics()

        if model_type not in final_fits:
            reasons.append(f"{model_type}: final full-data calibration failed")
            continue
        if summary.successful_folds != summary.total_folds:
            reasons.append(f"{model_type}: one or more CV folds failed")
            continue
        if metrics["validation_pose_success_rate"] < minimum_pose_success_rate:
            reasons.append(
                f"{model_type}: validation pose success rate below "
                f"{minimum_pose_success_rate:.2f}"
            )
            continue
        score = metrics["cv_global_rmse_px"]
        if not math.isfinite(score):
            reasons.append(f"{model_type}: no finite validation score")
            continue

        eligible.append((float(score), model_type))

    if eligible:
        eligible.sort()
        return eligible[0][1], reasons

    # Always preserve the best available result for diagnosis, but it will be
    # marked unusable in selected_calibration.json.
    fallback: list[tuple[float, str]] = []
    for model_type, fit in final_fits.items():
        score = summaries[model_type].metrics()["cv_global_rmse_px"]
        if math.isfinite(score):
            fallback.append((float(score), model_type))

    if not fallback:
        raise RuntimeError("No camera model produced a usable calibration result.")

    fallback.sort()
    reasons.append("No model met all eligibility requirements; using best diagnostic result.")
    return fallback[0][1], reasons


def quality_from_score(
    score: float,
    selected_summary: CandidateSummary,
    maximum_usable_cv_rmse: float,
    minimum_pose_success_rate: float,
) -> dict[str, Any]:
    metrics = selected_summary.metrics()
    all_folds_succeeded = (
        selected_summary.successful_folds == selected_summary.total_folds
    )
    pose_success_ok = (
        metrics["validation_pose_success_rate"] >= minimum_pose_success_rate
    )
    usable = (
        all_folds_succeeded
        and pose_success_ok
        and math.isfinite(score)
        and score <= maximum_usable_cv_rmse
    )

    if not usable:
        grade = "failed"
    elif score <= 0.7:
        grade = "good"
    elif score <= 1.2:
        grade = "acceptable"
    else:
        grade = "warning"

    return {
        "usable": usable,
        "grade": grade,
        "selection_metric": "k_fold_validation_global_rmse_px",
        "selection_score_px": finite_or_none(score),
        "maximum_usable_cv_rmse_px": maximum_usable_cv_rmse,
        "minimum_validation_pose_success_rate": minimum_pose_success_rate,
    }


def make_model_document(
    calibration_id: str,
    fit: CalibrationFit,
    new_K: np.ndarray,
    undistortion_settings: dict[str, Any],
    per_view_errors: list[float],
    cv_summary: CandidateSummary,
    image_paths: list[Path],
    source_session: Path,
    session_metadata: dict[str, Any],
    square_size_mm: float,
) -> dict[str, Any]:
    per_view = {
        path.name: float(error)
        for path, error in zip(image_paths, per_view_errors, strict=True)
    }

    cv_metrics = cv_summary.metrics()
    cv_metrics = {
        key: finite_or_none(value) if isinstance(value, float) else value
        for key, value in cv_metrics.items()
    }

    return {
        "calibration_id": calibration_id,
        "calibration_type": "intrinsics",
        "model_type": fit.model_type,
        "opencv_version": cv2.__version__,
        "image_size": list(IMAGE_SIZE),
        "pattern_size_internal_corners": list(PATTERN_SIZE),
        "physical_square_count": list(PHYSICAL_SQUARE_COUNT),
        "square_size_mm": square_size_mm,
        "camera_matrix": fit.K.tolist(),
        "distortion": fit.D.reshape(-1).tolist(),
        "distortion_parameter_names": distortion_parameter_names(
            fit.model_type,
            fit.D.size,
        ),
        "new_camera_matrix": new_K.tolist(),
        "undistortion": undistortion_settings,
        "solver_variant": fit.solver_variant,
        "metrics": {
            "cross_validation": cv_metrics,
            "final_fit_rms_px": fit.rms_px,
            "final_fit_mean_view_rmse_px": float(np.mean(per_view_errors)),
            "final_fit_median_view_rmse_px": float(np.median(per_view_errors)),
            "final_fit_max_view_rmse_px": float(np.max(per_view_errors)),
            "final_fit_per_view_rmse_px": per_view,
        },
        "source_session": str(source_session.resolve()),
        "lens_position": session_metadata.get("lens_position"),
        "used_images": [path.name for path in image_paths],
    }


def make_runtime_calibration_document(
    calibration_id: str,
    fit: CalibrationFit,
    new_K: np.ndarray,
    quality: dict[str, Any],
    lens_position: float | None,
    camera_binding: dict[str, Any],
) -> dict[str, Any]:
    """Build the minimal JSON consumed by ``CameraCalibration.from_json``."""

    return {
        "calibration_id": calibration_id,
        "model_type": fit.model_type,
        "image_size": list(IMAGE_SIZE),
        "camera_matrix": fit.K.tolist(),
        "distortion": fit.D.reshape(-1).tolist(),
        "new_camera_matrix": new_K.tolist(),
        "lens_position": lens_position,
        "camera_model": camera_binding.get("camera_model"),
        "sensor_pixel_array_size": camera_binding.get("sensor_pixel_array_size"),
        "scaler_crop": camera_binding.get("scaler_crop"),
        "quality": quality,
    }


def save_json(path: Path, data: Any) -> None:
    path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_model_diagnostics(
    model_dir: Path,
    fit: CalibrationFit,
    new_K: np.ndarray,
    per_view_errors: list[float],
    image_paths: list[Path],
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)

    with (model_dir / "per_view_errors.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image", "rmse_px"])
        for path, error in zip(image_paths, per_view_errors, strict=True):
            writer.writerow([path.name, f"{error:.9f}"])

    preview_image = cv2.imread(str(image_paths[0]), cv2.IMREAD_COLOR)
    if preview_image is not None:
        undistorted = undistort_image(fit, new_K, preview_image)
        comparison = np.hstack((preview_image, undistorted))
        cv2.imwrite(str(model_dir / "raw_vs_undistorted.jpg"), comparison)

    worst_index = int(np.argmax(per_view_errors))
    worst_image = cv2.imread(str(image_paths[worst_index]), cv2.IMREAD_COLOR)
    if worst_image is None:
        return

    projected = project_points(
        fit,
        object_points[worst_index],
        fit.rvecs[worst_index],
        fit.tvecs[worst_index],
    )
    observed = image_points[worst_index]

    overlay = worst_image.copy()
    for observed_point, projected_point in zip(observed, projected, strict=True):
        observed_xy = tuple(int(round(value)) for value in observed_point)
        projected_xy = tuple(int(round(value)) for value in projected_point)
        cv2.circle(overlay, observed_xy, 3, (0, 255, 0), -1)
        cv2.circle(overlay, projected_xy, 3, (0, 0, 255), -1)
        cv2.line(overlay, observed_xy, projected_xy, (0, 255, 255), 1)

    cv2.putText(
        overlay,
        f"Worst view: {image_paths[worst_index].name}, "
        f"RMSE={per_view_errors[worst_index]:.3f}px",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.imwrite(str(model_dir / "worst_reprojection_overlay.jpg"), overlay)


def main() -> None:
    args = parse_args()

    if not 0.0 < args.detection_scale <= 1.0:
        raise ValueError("--detection-scale must be in (0, 1].")
    if not 0.0 <= args.alpha <= 1.0:
        raise ValueError("--alpha must be in [0, 1].")
    if not 0.0 <= args.balance <= 1.0:
        raise ValueError("--balance must be in [0, 1].")

    session_dir = args.session.resolve() if args.session else find_latest_session()
    image_paths = collect_image_paths(session_dir)
    if not image_paths:
        raise FileNotFoundError(f"No images found in {session_dir}")

    run_dir = create_run_directory(args.output_prefix)
    calibration_id = (
        args.calibration_id.strip()
        if isinstance(args.calibration_id, str) and args.calibration_id.strip()
        else run_dir.name
    )
    print(f"Input session: {session_dir}")
    print(f"Output run:   {run_dir}")

    object_template = create_object_template(args.square_size_mm)
    object_points, image_points, valid_paths, rejected_images = load_observations(
        image_paths,
        object_template,
        run_dir / "diagnostics" / "detected",
        args.detection_scale,
    )

    print(f"Valid images: {len(valid_paths)} / {len(image_paths)}")
    if len(valid_paths) < MINIMUM_VIEWS:
        raise RuntimeError(
            f"Only {len(valid_paths)} valid images; need at least {MINIMUM_VIEWS}."
        )

    folds = make_folds(len(valid_paths), args.folds, args.seed)
    summaries = cross_validate(
        object_points,
        image_points,
        valid_paths,
        folds,
    )

    final_fits: dict[str, CalibrationFit] = {}
    final_fit_errors: dict[str, str] = {}

    basic_guess: np.ndarray | None = None
    for model_type in MODEL_ORDER:
        try:
            fit = fit_model(
                model_type,
                object_points,
                image_points,
                pinhole_guess=basic_guess if model_type == MODEL_FISHEYE else None,
            )
            final_fits[model_type] = fit
            if model_type == MODEL_PINHOLE:
                basic_guess = fit.K
        except (cv2.error, RuntimeError, ValueError) as error:
            final_fit_errors[model_type] = str(error)

    selected_type, selection_notes = choose_model(
        summaries,
        final_fits,
        args.minimum_pose_success_rate,
    )

    session_metadata = load_session_metadata(session_dir)
    model_documents: dict[str, dict[str, Any]] = {}

    for model_type in MODEL_ORDER:
        if model_type not in final_fits:
            save_json(
                run_dir / "models" / f"{model_type}.json",
                {
                    "calibration_id": calibration_id,
                    "model_type": model_type,
                    "status": "failed",
                    "error": final_fit_errors.get(model_type, "unknown error"),
                    "cross_validation": summary_to_json(
                        summaries[model_type], valid_paths
                    ),
                },
            )
            continue

        fit = final_fits[model_type]
        new_K, undistortion_settings = create_new_camera_matrix(
            fit,
            args.alpha,
            args.balance,
        )
        per_view_errors = calculate_fit_view_errors(
            fit,
            object_points,
            image_points,
        )

        document = make_model_document(
            calibration_id,
            fit,
            new_K,
            undistortion_settings,
            per_view_errors,
            summaries[model_type],
            valid_paths,
            session_dir,
            session_metadata,
            args.square_size_mm,
        )
        model_documents[model_type] = document
        save_json(run_dir / "models" / f"{model_type}.json", document)

        save_model_diagnostics(
            run_dir / "diagnostics" / model_type,
            fit,
            new_K,
            per_view_errors,
            valid_paths,
            object_points,
            image_points,
        )

    selected_summary = summaries[selected_type]
    selected_score = selected_summary.metrics()["cv_global_rmse_px"]
    quality = quality_from_score(
        selected_score,
        selected_summary,
        args.maximum_usable_cv_rmse,
        args.minimum_pose_success_rate,
    )

    selected_fit = final_fits[selected_type]
    selected_new_K = np.asarray(
        model_documents[selected_type]["new_camera_matrix"],
        dtype=np.float64,
    )
    selected_document = make_runtime_calibration_document(
        calibration_id,
        selected_fit,
        selected_new_K,
        quality,
        session_metadata.get("lens_position"),
        session_metadata,
    )
    selected_json_path = run_dir / "selected_calibration.json"
    save_json(selected_json_path, selected_document)

    np.savez_compressed(
        run_dir / "selected_calibration.npz",
        model_type=np.asarray(selected_type),
        image_size=np.asarray(IMAGE_SIZE, dtype=np.int32),
        K=selected_fit.K,
        D=selected_fit.D,
        new_K=selected_new_K,
    )

    comparison = {
        "calibration_id": calibration_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "opencv_version": cv2.__version__,
        "source_session": str(session_dir.resolve()),
        "output_run": str(run_dir.resolve()),
        "image_size": list(IMAGE_SIZE),
        "valid_image_count": len(valid_paths),
        "rejected_images": rejected_images,
        "cross_validation": {
            "fold_count": len(folds),
            "seed": args.seed,
            "fold_indices": [fold.tolist() for fold in folds],
        },
        "candidates": {
            model_type: summary_to_json(summaries[model_type], valid_paths)
            for model_type in MODEL_ORDER
        },
        "final_fit_errors": final_fit_errors,
        "selected_model_type": selected_type,
        "quality": quality,
        "selection_notes": selection_notes,
    }
    save_json(run_dir / "comparison.json", comparison)

    # Copy the selected model's diagnostics into one predictable subdirectory.
    selected_diagnostics = run_dir / "diagnostics" / "selected"
    shutil.copytree(
        run_dir / "diagnostics" / selected_type,
        selected_diagnostics,
    )

    print("\nModel comparison:")
    for model_type in MODEL_ORDER:
        metrics = summaries[model_type].metrics()
        score = metrics["cv_global_rmse_px"]
        score_text = f"{score:.4f} px" if math.isfinite(score) else "failed"
        print(
            f"  {model_type:20s} CV RMSE={score_text}, "
            f"folds={summaries[model_type].successful_folds}/"
            f"{summaries[model_type].total_folds}, "
            f"pose_success={metrics['validation_pose_success_rate']:.1%}"
        )

    print(f"\nSelected model: {selected_type}")
    print(f"Quality: {quality['grade']}, usable={quality['usable']}")
    print(f"Result: {selected_json_path}")

    if not quality["usable"]:
        print(
            "WARNING: the best candidate did not pass the configured quality "
            "threshold. Keep the result for diagnosis, but do not use it for "
            "ground mapping yet."
        )


if __name__ == "__main__":
    main()
