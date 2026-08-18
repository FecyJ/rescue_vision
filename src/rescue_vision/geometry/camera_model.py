"""相机自身几何：去畸变图像与像素坐标。"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from rescue_vision.geometry.types import (
    RawPixel,
    UndistortedPixel,
    array_to_pixels,
    pixels_to_array,
)


FloatArray = npt.NDArray[np.float64]

_CALIBRATION_JSON_FIELDS = frozenset(
    {
        "calibration_id",
        "model_type",
        "image_size",
        "camera_matrix",
        "distortion",
        "new_camera_matrix",
        "lens_position",
        "camera_model",
        "sensor_pixel_array_size",
        "scaler_crop",
        "quality",
    }
)
IMAGE_BORDER_FILL_VALUE = 114


class CameraModelType(str, Enum):
    """项目支持的相机投影/畸变模型。"""

    PINHOLE = "pinhole"
    PINHOLE_RATIONAL = "pinhole_rational"
    FISHEYE = "fisheye"

    @classmethod
    def parse(cls, value: str | CameraModelType) -> CameraModelType:
        if isinstance(value, cls):
            return value

        aliases = {
            "opencv_pinhole": cls.PINHOLE,
            "opencv_pinhole_rational": cls.PINHOLE_RATIONAL,
            "opencv_fisheye": cls.FISHEYE,
        }

        normalized = str(value).strip().lower()
        if normalized in aliases:
            return aliases[normalized]
        return cls(normalized)


@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """固定分辨率下的一套相机内参标定结果。

    ``calibration_id`` 是关联内参与地面映射的可读身份，不是版本号或校验和。
    ``image_size`` 的顺序为 ``(width, height)``。

    ``K`` 是原始畸变图像的内参矩阵；``new_K`` 是去畸变输出图像使用的
    内参矩阵。Pinhole 与 Pinhole Rational 都使用 OpenCV 标准相机接口，
    Fisheye 使用 ``cv2.fisheye`` 接口。
    """

    calibration_id: str
    model: CameraModelType
    image_size: tuple[int, int]
    K: FloatArray
    D: FloatArray
    new_K: FloatArray
    lens_position: float | None = None
    camera_model: str | None = None
    sensor_pixel_array_size: tuple[int, int] | None = None
    scaler_crop: tuple[int, int, int, int] | None = None

    def __post_init__(self) -> None:
        calibration_id = str(self.calibration_id).strip()
        model = CameraModelType.parse(self.model)
        image_size = tuple(int(value) for value in self.image_size)
        K = np.ascontiguousarray(self.K, dtype=np.float64)
        D = np.ascontiguousarray(self.D, dtype=np.float64).reshape(-1, 1)
        new_K = np.ascontiguousarray(self.new_K, dtype=np.float64)
        lens_position = (
            None if self.lens_position is None else float(self.lens_position)
        )
        camera_model = (
            None if self.camera_model is None else str(self.camera_model).strip()
        )
        sensor_pixel_array_size = (
            None
            if self.sensor_pixel_array_size is None
            else tuple(int(value) for value in self.sensor_pixel_array_size)
        )
        scaler_crop = (
            None
            if self.scaler_crop is None
            else tuple(int(value) for value in self.scaler_crop)
        )

        if not calibration_id:
            raise ValueError("calibration_id must be a non-empty string.")
        if len(image_size) != 2 or any(value <= 0 for value in image_size):
            raise ValueError(
                f"image_size must be two positive integers, got {image_size}."
            )
        if K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {K.shape}.")
        if new_K.shape != (3, 3):
            raise ValueError(f"new_K must have shape (3, 3), got {new_K.shape}.")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(new_K)):
            raise ValueError("K and new_K must contain only finite values.")
        if not np.all(np.isfinite(D)):
            raise ValueError("D must contain only finite values.")
        if lens_position is not None and not np.isfinite(lens_position):
            raise ValueError("lens_position must be finite when provided.")
        binding_values = (camera_model, sensor_pixel_array_size, scaler_crop)
        if any(value is not None for value in binding_values) and not all(
            value is not None for value in binding_values
        ):
            raise ValueError(
                "camera_model, sensor_pixel_array_size and scaler_crop must be "
                "provided together."
            )
        if camera_model is not None:
            if not camera_model:
                raise ValueError("camera_model must be non-empty.")
            assert sensor_pixel_array_size is not None
            assert scaler_crop is not None
            if (
                len(sensor_pixel_array_size) != 2
                or any(value <= 0 for value in sensor_pixel_array_size)
            ):
                raise ValueError(
                    "sensor_pixel_array_size must contain two positive integers."
                )
            if (
                len(scaler_crop) != 4
                or scaler_crop[0] < 0
                or scaler_crop[1] < 0
                or scaler_crop[2] <= 0
                or scaler_crop[3] <= 0
                or scaler_crop[0] + scaler_crop[2] > sensor_pixel_array_size[0]
                or scaler_crop[1] + scaler_crop[3] > sensor_pixel_array_size[1]
            ):
                raise ValueError(
                    "scaler_crop must be [x, y, width, height] inside the sensor array."
                )
        if model is CameraModelType.FISHEYE and D.size != 4:
            raise ValueError("Fisheye D must contain exactly 4 parameters.")
        if model is not CameraModelType.FISHEYE and D.size not in {
            4,
            5,
            8,
            12,
            14,
        }:
            raise ValueError(
                "Pinhole D must contain 4, 5, 8, 12 or 14 parameters."
            )

        object.__setattr__(self, "calibration_id", calibration_id)
        object.__setattr__(self, "model", model)
        object.__setattr__(self, "image_size", image_size)
        object.__setattr__(self, "K", K)
        object.__setattr__(self, "D", D)
        object.__setattr__(self, "new_K", new_K)
        object.__setattr__(self, "lens_position", lens_position)
        object.__setattr__(self, "camera_model", camera_model)
        object.__setattr__(self, "sensor_pixel_array_size", sensor_pixel_array_size)
        object.__setattr__(self, "scaler_crop", scaler_crop)

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        allow_unusable: bool = False,
    ) -> CameraCalibration:
        """从 ``selected_calibration.json`` 加载。

        默认拒绝加载被标定脚本标记为不可用的结果；诊断时可以显式传入
        ``allow_unusable=True``。
        """

        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data, allow_unusable=allow_unusable)

    @classmethod
    def from_dict(
        cls,
        data: dict[str, object],
        *,
        allow_unusable: bool = False,
    ) -> CameraCalibration:
        """从运行时内参 JSON 的最小格式加载并执行统一校验。"""

        if not isinstance(data, dict):
            raise ValueError(
                f"Calibration JSON must be an object, got {type(data).__name__}."
            )
        unknown = sorted(set(data) - _CALIBRATION_JSON_FIELDS)
        if unknown:
            raise ValueError(
                f"Unknown keys in calibration JSON: {unknown}."
            )

        quality = data.get("quality")
        if not isinstance(quality, dict):
            raise ValueError(
                "Calibration JSON must contain a quality mapping."
            )
        usable = quality.get("usable")
        if not isinstance(usable, bool):
            raise ValueError(
                "Calibration JSON quality.usable must be a boolean."
            )
        if not allow_unusable and not usable:
            raise ValueError(
                "Calibration result is marked unusable. Inspect comparison.json "
                "and diagnostics before loading it."
            )

        calibration_id = data.get("calibration_id")
        if not isinstance(calibration_id, str) or not calibration_id.strip():
            raise ValueError("Calibration JSON is missing a non-empty calibration_id.")

        model_value = data.get("model_type")
        if model_value is None:
            raise KeyError("Calibration JSON is missing 'model_type'.")

        K_value = data.get("camera_matrix")
        D_value = data.get("distortion")
        new_K_value = data.get("new_camera_matrix")

        if K_value is None or D_value is None or new_K_value is None:
            raise KeyError(
                "Calibration JSON must contain camera_matrix, distortion "
                "and new_camera_matrix."
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
                "Calibration image_size must be integer [width, height]."
            )

        return cls(
            calibration_id=calibration_id,
            model=CameraModelType.parse(model_value),
            image_size=tuple(int(value) for value in image_size_value),
            K=np.asarray(K_value, dtype=np.float64),
            D=np.asarray(D_value, dtype=np.float64),
            new_K=np.asarray(new_K_value, dtype=np.float64),
            lens_position=data.get("lens_position"),
            camera_model=data.get("camera_model"),
            sensor_pixel_array_size=data.get("sensor_pixel_array_size"),
            scaler_crop=data.get("scaler_crop"),
        )

class CameraModel:
    """把原始畸变图像/像素转换到固定的去畸变像素坐标系。"""

    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration
        self.model = CameraModelType.parse(calibration.model)
        self.image_size = tuple(calibration.image_size)

        self.K = np.ascontiguousarray(calibration.K, dtype=np.float64)
        self.D = np.ascontiguousarray(calibration.D, dtype=np.float64).reshape(-1, 1)
        self.new_K = np.ascontiguousarray(calibration.new_K, dtype=np.float64)

        self.map1, self.map2 = self._create_undistort_maps()

        # 去畸变图中哪些像素确实来自原始图像，可用于排除填充边缘。
        source_mask = np.full(self.image_size[::-1], 255, dtype=np.uint8)
        self.valid_mask = cv2.remap(
            source_mask,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    @classmethod
    def from_json(
        cls,
        path: str | Path,
        *,
        allow_unusable: bool = False,
    ) -> CameraModel:
        return cls(
            CameraCalibration.from_json(
                path,
                allow_unusable=allow_unusable,
            )
        )

    def _create_undistort_maps(self) -> tuple[np.ndarray, np.ndarray]:
        identity = np.eye(3, dtype=np.float64)

        if self.model is CameraModelType.FISHEYE:
            return cv2.fisheye.initUndistortRectifyMap(
                K=self.K,
                D=self.D.reshape(4, 1),
                R=identity,
                P=self.new_K,
                size=self.image_size,
                m1type=cv2.CV_16SC2,
            )

        return cv2.initUndistortRectifyMap(
            cameraMatrix=self.K,
            distCoeffs=self.D,
            R=identity,
            newCameraMatrix=self.new_K,
            size=self.image_size,
            m1type=cv2.CV_16SC2,
        )

    def undistort_pixels(
        self,
        pixels: Sequence[RawPixel],
    ) -> list[UndistortedPixel]:
        """将原始畸变像素转换为去畸变图像中的像素。"""

        pixel_list = list(pixels)
        if not pixel_list:
            return []

        points = np.ascontiguousarray(
            pixels_to_array(pixel_list).reshape(-1, 1, 2),
            dtype=np.float64,
        )

        if self.model is CameraModelType.FISHEYE:
            undistorted = cv2.fisheye.undistortPoints(
                points,
                K=self.K,
                D=self.D.reshape(4, 1),
                R=np.eye(3, dtype=np.float64),
                P=self.new_K,
            )
        else:
            undistorted = cv2.undistortPoints(
                points,
                cameraMatrix=self.K,
                distCoeffs=self.D,
                R=np.eye(3, dtype=np.float64),
                P=self.new_K,
            )

        return list(array_to_pixels(undistorted, UndistortedPixel))

    def undistort_pixel(self, pixel: RawPixel) -> UndistortedPixel:
        return self.undistort_pixels([pixel])[0]

    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """将原始畸变图像转换为固定 ``new_K`` 坐标系下的图像。"""

        if image.ndim not in {2, 3}:
            raise ValueError(
                f"image must have 2 or 3 dimensions, got shape {image.shape}."
            )
        actual_size = (int(image.shape[1]), int(image.shape[0]))
        if actual_size != self.image_size:
            raise ValueError(
                f"Image size {actual_size} does not match calibration "
                f"{self.image_size}."
            )

        undistorted = cv2.remap(
            image,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(
                IMAGE_BORDER_FILL_VALUE
                if image.ndim == 2
                else (IMAGE_BORDER_FILL_VALUE,) * image.shape[2]
            ),
        )
        undistorted[self.valid_mask == 0] = IMAGE_BORDER_FILL_VALUE
        return undistorted
