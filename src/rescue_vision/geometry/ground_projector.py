"""去畸变像素、机器人地面坐标与 BEV 像素之间的映射。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from rescue_vision.geometry.camera_model import CameraCalibration
from rescue_vision.geometry.types import (
    BevPixel,
    GroundPoint,
    UndistortedPixel,
    array_to_points,
    pixels_to_array,
)


FloatArray = npt.NDArray[np.float64]


def _validated_homography(value: npt.ArrayLike, name: str) -> FloatArray:
    matrix = np.ascontiguousarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"{name} must have shape (3, 3), got {matrix.shape}.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values.")
    scale = float(np.max(np.abs(matrix)))
    condition = np.linalg.cond(matrix / scale) if scale > 0 else float("inf")
    if not np.isfinite(condition) or condition > 1e12:
        raise ValueError(
            f"{name} must be stably invertible, condition_number={condition}."
        )
    return matrix


@dataclass(frozen=True, slots=True)
class BevConfig:
    """BEV 覆盖的机器人地面范围，所有长度单位为 mm。"""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    mm_per_pixel: float

    def __post_init__(self) -> None:
        values = (
            self.x_min,
            self.x_max,
            self.y_min,
            self.y_max,
            self.mm_per_pixel,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f"BEV values must be finite, got {values}.")
        if self.x_max <= self.x_min:
            raise ValueError("BEV x_max must be greater than x_min.")
        if self.y_max <= self.y_min:
            raise ValueError("BEV y_max must be greater than y_min.")
        if self.mm_per_pixel <= 0:
            raise ValueError("BEV mm_per_pixel must be positive.")

        width_float = (self.y_max - self.y_min) / self.mm_per_pixel
        height_float = (self.x_max - self.x_min) / self.mm_per_pixel
        if not np.isclose(width_float, round(width_float), atol=1e-9):
            raise ValueError("BEV lateral range must be divisible by mm_per_pixel.")
        if not np.isclose(height_float, round(height_float), atol=1e-9):
            raise ValueError("BEV forward range must be divisible by mm_per_pixel.")

    @property
    def width(self) -> int:
        return round((self.y_max - self.y_min) / self.mm_per_pixel)

    @property
    def height(self) -> int:
        return round((self.x_max - self.x_min) / self.mm_per_pixel)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> BevConfig:
        return cls(
            x_min=float(data["x_min_mm"]),
            x_max=float(data["x_max_mm"]),
            y_min=float(data["y_min_mm"]),
            y_max=float(data["y_max_mm"]),
            mm_per_pixel=float(data["mm_per_pixel"]),
        )


class GroundProjector:
    """负责 ``UndistortedPixel``、``GroundPoint`` 和 BEV 的转换。"""

    def __init__(
        self,
        image_to_ground: npt.ArrayLike,
        bev_config: BevConfig | None = None,
    ) -> None:
        self.image_to_ground = _validated_homography(
            image_to_ground,
            "image_to_ground",
        )
        self.ground_to_image = np.linalg.inv(self.image_to_ground)
        self.bev_config = bev_config

        self.ground_to_bev: FloatArray | None = None
        self.bev_to_ground: FloatArray | None = None
        self.image_to_bev: FloatArray | None = None
        if bev_config is not None:
            self.ground_to_bev = self.make_ground_to_bev_matrix(bev_config)
            self.bev_to_ground = np.linalg.inv(self.ground_to_bev)
            self.image_to_bev = self.ground_to_bev @ self.image_to_ground

    @staticmethod
    def make_ground_to_bev_matrix(config: BevConfig) -> FloatArray:
        """地面到 BEV：前方在上（v 小），左方在左（u 小）。"""

        scale = 1.0 / config.mm_per_pixel
        return np.array(
            [
                [0.0, -scale, config.y_max * scale],
                [-scale, 0.0, config.x_max * scale],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        camera_calibration: CameraCalibration,
    ) -> GroundProjector:
        """加载地面映射，并拒绝与当前内参不匹配的产物。"""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        quality = data.get("quality")
        if not isinstance(quality, dict) or quality.get("usable") is not True:
            raise ValueError(
                "Ground mapping quality.usable must be true; inspect its "
                "quality failures and recalibrate."
            )
        if quality.get("physically_valid") is not True:
            raise ValueError(
                "Ground mapping pose must be marked physically_valid."
            )

        image_size = tuple(int(value) for value in data["image_size"])
        if image_size != camera_calibration.image_size:
            raise ValueError(
                f"Ground mapping image_size {image_size} does not match "
                f"intrinsics {camera_calibration.image_size}."
            )

        intrinsic = data.get("intrinsics")
        if not isinstance(intrinsic, dict):
            raise ValueError("Ground mapping is missing intrinsics metadata.")
        actual_model = str(intrinsic.get("model_type"))
        if actual_model != camera_calibration.model.value:
            raise ValueError(
                f"Ground mapping model {actual_model!r} does not match "
                f"intrinsics {camera_calibration.model.value!r}."
            )
        actual_calibration_id = intrinsic.get("calibration_id")
        if actual_calibration_id != camera_calibration.calibration_id:
            raise ValueError(
                "Ground mapping was produced with different intrinsic "
                "parameters (calibration_id mismatch)."
            )

        bev_data = data.get("bev")
        bev_config = (
            BevConfig.from_dict(bev_data)
            if isinstance(bev_data, dict)
            else None
        )
        return cls(data["image_to_ground"], bev_config)

    def pixels_to_ground(
        self,
        pixels: Sequence[UndistortedPixel],
    ) -> list[GroundPoint]:
        if not pixels:
            return []
        points = cv2.perspectiveTransform(
            pixels_to_array(pixels).reshape(-1, 1, 2),
            self.image_to_ground,
        ).reshape(-1, 2)
        return array_to_points(points, GroundPoint)

    def pixel_to_ground(self, pixel: UndistortedPixel) -> GroundPoint:
        return self.pixels_to_ground([pixel])[0]

    def ground_to_pixels(
        self,
        ground_points: Sequence[GroundPoint],
    ) -> list[UndistortedPixel]:
        if not ground_points:
            return []
        points = np.asarray(
            [[point.x, point.y] for point in ground_points],
            dtype=np.float64,
        )
        pixels = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self.ground_to_image,
        ).reshape(-1, 2)
        return array_to_points(pixels, UndistortedPixel)

    def ground_to_pixel(self, ground_point: GroundPoint) -> UndistortedPixel:
        return self.ground_to_pixels([ground_point])[0]

    def ground_to_bev_pixels(
        self,
        ground_points: Sequence[GroundPoint],
    ) -> list[BevPixel]:
        if not ground_points:
            return []
        if self.ground_to_bev is None:
            raise ValueError("BEV config is not set.")
        points = np.asarray(
            [[point.x, point.y] for point in ground_points],
            dtype=np.float64,
        )
        bev_pixels = cv2.perspectiveTransform(
            points.reshape(-1, 1, 2),
            self.ground_to_bev,
        ).reshape(-1, 2)
        return array_to_points(bev_pixels, BevPixel)

    def ground_to_bev_pixel(self, point: GroundPoint) -> BevPixel:
        return self.ground_to_bev_pixels([point])[0]

    def bev_pixels_to_ground(
        self,
        pixels: Sequence[BevPixel],
    ) -> list[GroundPoint]:
        if not pixels:
            return []
        if self.bev_to_ground is None:
            raise ValueError("BEV config is not set.")
        points = cv2.perspectiveTransform(
            pixels_to_array(pixels).reshape(-1, 1, 2),
            self.bev_to_ground,
        ).reshape(-1, 2)
        return array_to_points(points, GroundPoint)

    def bev_pixel_to_ground(self, pixel: BevPixel) -> GroundPoint:
        return self.bev_pixels_to_ground([pixel])[0]

    def make_bev_image(self, undistorted_image: np.ndarray) -> np.ndarray:
        if self.bev_config is None or self.image_to_bev is None:
            raise ValueError("BEV config is not set.")
        return cv2.warpPerspective(
            undistorted_image,
            self.image_to_bev,
            (self.bev_config.width, self.bev_config.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
