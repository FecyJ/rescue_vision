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
    RobotPoint3D,
    UndistortedPixel,
    array_to_points,
    pixels_to_array,
    robot_frame_metadata,
)


FloatArray = npt.NDArray[np.float64]

_GROUND_MAPPING_JSON_FIELDS = frozenset(
    {
        "calibration_id",
        "model_type",
        "image_size",
        "coordinate_frame",
        "image_to_ground",
        "quality",
        "extrinsics",
        "bev",
    }
)
_GROUND_EXTRINSICS_JSON_FIELDS = frozenset(
    {
        "rotation_robot_to_camera",
        "translation_robot_to_camera_mm",
    }
)
_GROUND_BEV_JSON_FIELDS = frozenset(
    {
        "x_min_mm",
        "x_max_mm",
        "y_min_mm",
        "y_max_mm",
        "mm_per_pixel",
    }
)


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


def _validated_camera_matrix(value: npt.ArrayLike) -> FloatArray:
    matrix = _validated_homography(value, "new_camera_matrix")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("new_camera_matrix focal lengths must be positive.")
    return matrix


def _validated_rotation(value: npt.ArrayLike) -> FloatArray:
    rotation = np.ascontiguousarray(value, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(
            "rotation_robot_to_camera must have shape (3, 3), "
            f"got {rotation.shape}."
        )
    if not np.all(np.isfinite(rotation)):
        raise ValueError(
            "rotation_robot_to_camera must contain only finite values."
        )
    if not np.allclose(
        rotation @ rotation.T,
        np.eye(3),
        atol=1e-6,
        rtol=0.0,
    ) or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(
            "rotation_robot_to_camera must be an orthonormal proper rotation."
        )
    return rotation


def _validated_translation(value: npt.ArrayLike) -> FloatArray:
    translation = np.ascontiguousarray(value, dtype=np.float64).reshape(-1)
    if translation.shape != (3,):
        raise ValueError(
            "translation_robot_to_camera_mm must contain three values, "
            f"got shape {translation.shape}."
        )
    if not np.all(np.isfinite(translation)):
        raise ValueError(
            "translation_robot_to_camera_mm must contain only finite values."
        )
    return translation


@dataclass(frozen=True, slots=True)
class BevConfig:
    """BEV 覆盖的机器人地面范围，原点为两轮接地点中点，单位为 mm。"""

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
        *,
        new_camera_matrix: npt.ArrayLike | None = None,
        rotation_robot_to_camera: npt.ArrayLike | None = None,
        translation_robot_to_camera_mm: npt.ArrayLike | None = None,
    ) -> None:
        self.image_to_ground = _validated_homography(
            image_to_ground,
            "image_to_ground",
        )
        self.ground_to_image = np.linalg.inv(self.image_to_ground)
        self.bev_config = bev_config

        projection_values = (
            new_camera_matrix,
            rotation_robot_to_camera,
            translation_robot_to_camera_mm,
        )
        if any(value is not None for value in projection_values) and not all(
            value is not None for value in projection_values
        ):
            raise ValueError(
                "new_camera_matrix, rotation_robot_to_camera and "
                "translation_robot_to_camera_mm must be provided together."
            )
        self.new_camera_matrix: FloatArray | None = None
        self.rotation_robot_to_camera: FloatArray | None = None
        self.translation_robot_to_camera_mm: FloatArray | None = None
        self.camera_position_robot_mm: FloatArray | None = None
        if new_camera_matrix is not None:
            assert rotation_robot_to_camera is not None
            assert translation_robot_to_camera_mm is not None
            self.new_camera_matrix = _validated_camera_matrix(
                new_camera_matrix
            )
            self.rotation_robot_to_camera = _validated_rotation(
                rotation_robot_to_camera
            )
            self.translation_robot_to_camera_mm = _validated_translation(
                translation_robot_to_camera_mm
            )
            self.camera_position_robot_mm = (
                -self.rotation_robot_to_camera.T
                @ self.translation_robot_to_camera_mm
            )

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
        """加载当前最小地面映射 JSON，并校验内参身份。"""

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(
                f"Ground mapping JSON must be an object, got "
                f"{type(data).__name__}."
            )
        unknown = sorted(set(data) - _GROUND_MAPPING_JSON_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown keys in ground mapping JSON: {unknown}."
            )

        coordinate_frame = data.get("coordinate_frame")
        expected_frame = robot_frame_metadata()
        if coordinate_frame != expected_frame:
            raise ValueError(
                "Ground mapping coordinate_frame must use the robot frame "
                f"origin={expected_frame['origin']!r}, got {coordinate_frame!r}."
            )

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

        calibration_id = data.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError(
                "Ground mapping is missing a non-empty calibration_id."
            )
        if calibration_id != camera_calibration.calibration_id:
            raise ValueError(
                "Ground mapping was produced with different intrinsic "
                "parameters (calibration_id mismatch)."
            )

        model_type = data.get("model_type")
        if model_type != camera_calibration.model.value:
            raise ValueError(
                f"Ground mapping model {model_type!r} does not match "
                f"intrinsics {camera_calibration.model.value!r}."
            )

        image_size_value = data.get("image_size")
        if (
            not isinstance(image_size_value, (list, tuple))
            or len(image_size_value) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in image_size_value
            )
        ):
            raise ValueError(
                "Ground mapping image_size must be integer [width, height]."
            )
        image_size = tuple(image_size_value)
        if image_size != camera_calibration.image_size:
            raise ValueError(
                f"Ground mapping image_size {image_size} does not match "
                f"intrinsics {camera_calibration.image_size}."
            )

        bev_data = data.get("bev")
        if isinstance(bev_data, dict):
            unknown_bev = sorted(set(bev_data) - _GROUND_BEV_JSON_FIELDS)
            if unknown_bev:
                raise ValueError(
                    f"Unknown keys in ground mapping bev: {unknown_bev}."
                )
        bev_config = (
            BevConfig.from_dict(bev_data)
            if isinstance(bev_data, dict)
            else None
        )
        extrinsics = data.get("extrinsics")
        if extrinsics is not None and not isinstance(extrinsics, dict):
            raise ValueError("Ground mapping extrinsics must be a mapping.")
        if isinstance(extrinsics, dict):
            unknown_extrinsics = sorted(
                set(extrinsics) - _GROUND_EXTRINSICS_JSON_FIELDS
            )
            if unknown_extrinsics:
                raise ValueError(
                    "Unknown keys in ground mapping extrinsics: "
                    f"{unknown_extrinsics}."
                )
            rotation = extrinsics.get("rotation_robot_to_camera")
            translation = extrinsics.get(
                "translation_robot_to_camera_mm"
            )
            if rotation is None or translation is None:
                raise ValueError(
                    "Ground mapping extrinsics must contain "
                    "rotation_robot_to_camera and "
                    "translation_robot_to_camera_mm."
                )
            rotation_value = _validated_rotation(rotation)
            translation_value = _validated_translation(translation)
            pose_ground_to_image = camera_calibration.new_K @ np.column_stack(
                (
                    rotation_value[:, 0],
                    rotation_value[:, 1],
                    translation_value,
                )
            )
            if abs(float(np.linalg.det(pose_ground_to_image))) < 1e-12:
                raise ValueError("Ground mapping physical pose is singular on z=0.")
            pose_image_to_ground = np.linalg.inv(pose_ground_to_image)
            pose_image_to_ground /= pose_image_to_ground[2, 2]
            stored_image_to_ground = _validated_homography(
                data["image_to_ground"],
                "image_to_ground",
            )
            stored_image_to_ground /= stored_image_to_ground[2, 2]
            if not np.allclose(
                stored_image_to_ground,
                pose_image_to_ground,
                rtol=1e-7,
                atol=1e-7,
            ):
                raise ValueError(
                    "Ground mapping image_to_ground must be derived from its "
                    "physical extrinsics."
                )
        else:
            rotation = None
            translation = None
        return cls(
            data["image_to_ground"],
            bev_config,
            new_camera_matrix=(
                camera_calibration.new_K
                if rotation is not None
                else None
            ),
            rotation_robot_to_camera=rotation,
            translation_robot_to_camera_mm=translation,
        )

    @property
    def supports_robot_projection(self) -> bool:
        """是否加载了可用于三维机器人系投影的完整物理外参。"""

        return self.new_camera_matrix is not None

    def project_robot_points(
        self,
        points: Sequence[RobotPoint3D],
    ) -> list[UndistortedPixel]:
        """把机器人系三维点投影到全尺寸去畸变像素。"""

        if not points:
            return []
        if (
            self.new_camera_matrix is None
            or self.rotation_robot_to_camera is None
            or self.translation_robot_to_camera_mm is None
        ):
            raise ValueError(
                "Ground mapping does not contain full camera extrinsics."
            )
        values = np.asarray(
            [[point.x, point.y, point.z] for point in points],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(values)):
            raise ValueError("RobotPoint3D values must be finite.")
        camera_points = (
            self.rotation_robot_to_camera @ values.T
        ).T + self.translation_robot_to_camera_mm
        depth = camera_points[:, 2]
        if np.any(depth <= 1e-9):
            raise ValueError(
                "RobotPoint3D values must project in front of the camera."
            )
        homogeneous = (
            self.new_camera_matrix @ camera_points.T
        ).T
        pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
        return array_to_points(pixels, UndistortedPixel)

    def project_robot_point(
        self,
        point: RobotPoint3D,
    ) -> UndistortedPixel:
        return self.project_robot_points([point])[0]

    def pixel_to_horizontal_plane(
        self,
        pixel: UndistortedPixel,
        *,
        z_mm: float,
    ) -> RobotPoint3D:
        """将去畸变像素射线与机器人系 ``z=z_mm`` 平面求交。"""

        if (
            self.new_camera_matrix is None
            or self.rotation_robot_to_camera is None
            or self.camera_position_robot_mm is None
        ):
            raise ValueError(
                "Ground mapping does not contain full camera extrinsics."
            )
        values = np.asarray((pixel.u, pixel.v, z_mm), dtype=np.float64)
        if not np.all(np.isfinite(values)):
            raise ValueError("pixel and z_mm must be finite.")
        ray_camera = np.linalg.solve(
            self.new_camera_matrix,
            np.asarray((pixel.u, pixel.v, 1.0), dtype=np.float64),
        )
        ray_robot = self.rotation_robot_to_camera.T @ ray_camera
        if abs(float(ray_robot[2])) <= 1e-12:
            raise ValueError(
                "Pixel ray is parallel to the requested horizontal plane."
            )
        scale = (
            float(z_mm) - float(self.camera_position_robot_mm[2])
        ) / float(ray_robot[2])
        if scale <= 0.0:
            raise ValueError(
                "Requested horizontal plane lies behind the pixel ray."
            )
        point = self.camera_position_robot_mm + scale * ray_robot
        return RobotPoint3D(
            float(point[0]),
            float(point[1]),
            float(point[2]),
        )

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
