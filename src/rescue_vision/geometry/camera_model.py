"""
相机自身几何:去畸变、像素射线
去畸变用fisheye模型
"""

from dataclasses import dataclass
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from rescue_vision.geometry.types import RawPixel, UndistortedPixel, pixels_to_array, array_to_pixels

FloatArray= npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class CameraCalibration:
    """
    所有参数必须对应 calibration_size 分辨率。

    image_size 格式：(width, height)

    K: 原始畸变图像的内参矩阵。

    D: 畸变系数
        PINHOLE 模型可以为 4、5、8、12 或 14 个参数；
        FISHEYE 模型通常必须为 4 个参数。

    new_K: 去畸变后得到的固定内参矩阵。
    """
    image_size: tuple[int, int]
    K: FloatArray
    D: FloatArray
    new_K: FloatArray

class CameraModel:
    """
    输入：RawPixel，原始相机图像中的畸变像素坐标
    输出：UndistortedPixel，去畸变图像中的像素坐标
    """
    def __init__(self, calibration: CameraCalibration):
        self.calibration = calibration
        self.K = calibration.K
        self.D = calibration.D
        self.new_K = calibration.new_K
        self.image_size = calibration.image_size
        if self.D.shape[0] != 4:
            raise ValueError("D must have 4 parameters for fisheye model.")
        
        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K=calibration.K,
            D=calibration.D,
            R=np.eye(3, dtype=np.float64),
            P=calibration.new_K,
            size=calibration.image_size,
            m1type=cv2.CV_16SC2,
        )

        # 去畸变后哪些像素来自有效原图
        # 后续做颜色分割或 BEV 时可排除黑边
        source_mask = np.full(calibration.image_size[::-1], 255, dtype=np.uint8)
        self.valid_mask = cv2.remap(
            source_mask,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

    def undistort_pixels(self, pixels: Sequence[RawPixel]) -> Sequence[UndistortedPixel]:
        """
        将原始畸变像素坐标转换为去畸变像素坐标
        """
        points = pixels_to_array(list(pixels)).reshape(-1, 1, 2)  # shape=(N, 1, 2)
        undistorted_points = cv2.fisheye.undistortPoints(
            points,
            K=self.K,
            D=self.D,
            P=self.new_K,
        )
        return array_to_pixels(undistorted_points, UndistortedPixel)

    def undistort_pixel(self,pixel: RawPixel) -> UndistortedPixel:
        return self.undistort_pixels([pixel])[0]
    
    def undistort_image(self, image: np.ndarray) -> np.ndarray:
        """
        将原始畸变图像转换为去畸变图像
        """
        undistorted_image = cv2.remap(
            image,
            self.map1,
            self.map2,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return undistorted_image