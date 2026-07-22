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

    ``image_size`` 的顺序为 ``(width, height)``。

    ``K`` 是原始畸变图像的内参矩阵；``new_K`` 是去畸变输出图像使用的
    内参矩阵。Pinhole 与 Pinhole Rational 都使用 OpenCV 标准相机接口，
    Fisheye 使用 ``cv2.fisheye`` 接口。
    """

    model: CameraModelType
    image_size: tuple[int, int]
    K: FloatArray
    D: FloatArray
    new_K: FloatArray

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

        if (
            not allow_unusable
            and data.get("quality", {}).get("usable") is False
        ):
            raise ValueError(
                "Calibration result is marked unusable. Inspect comparison.json "
                "and diagnostics before loading it."
            )

        model_value = data.get("model_type", data.get("model"))
        if model_value is None:
            raise KeyError("Calibration JSON is missing 'model_type'.")

        K_value = data.get("camera_matrix", data.get("K"))
        D_value = data.get("distortion", data.get("D"))
        new_K_value = data.get("new_camera_matrix", data.get("new_K"))

        if K_value is None or D_value is None or new_K_value is None:
            raise KeyError(
                "Calibration JSON must contain camera_matrix, distortion "
                "and new_camera_matrix."
            )

        return cls(
            model=CameraModelType.parse(model_value),
            image_size=tuple(int(value) for value in data["image_size"]),
            K=np.asarray(K_value, dtype=np.float64),
            D=np.asarray(D_value, dtype=np.float64),
            new_K=np.asarray(new_K_value, dtype=np.float64),
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

        self._validate_calibration()
        self.map1, self.map2 = self._create_undistort_maps()

        # 去畸变图中哪些像素确实来自原始图像，可用于排除黑边。
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

    def _validate_calibration(self) -> None:
        if self.K.shape != (3, 3):
            raise ValueError(f"K must have shape (3, 3), got {self.K.shape}.")
        if self.new_K.shape != (3, 3):
            raise ValueError(
                f"new_K must have shape (3, 3), got {self.new_K.shape}."
            )
        if self.model is CameraModelType.FISHEYE and self.D.size != 4:
            raise ValueError("Fisheye D must contain exactly 4 parameters.")
        if self.model is not CameraModelType.FISHEYE and self.D.size not in {
            4,
            5,
            8,
            12,
            14,
        }:
            raise ValueError(
                "Pinhole D must contain 4, 5, 8, 12 or 14 parameters."
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

        return cv2.remap(
            image,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )