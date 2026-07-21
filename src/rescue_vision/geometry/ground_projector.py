"""
去畸变像素 <-> 地面映射
"""
from dataclasses import dataclass
from collections.abc import Sequence

import cv2
import numpy as np
import numpy.typing as npt

from rescue_vision.geometry.types import RawPixel, UndistortedPixel, GroundPoint, pixels_to_array, array_to_pixels

FloatArray= npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class BevConfig:
    """
    BEV覆盖的机器人地面范围，单位: mm
    x：左右方向
    y：前后方向
    """
    x_min: float
    x_max: float
    y_min: float
    y_max: float
    mm_per_pixel: float

    @property
    def width(self) -> int:   # 机器人左方为 Y+
        return round((self.y_max - self.y_min) / self.mm_per_pixel)
    @property
    def height(self) -> int:   # 机器人前方为 X+
        return round((self.x_max - self.x_min) / self.mm_per_pixel)

class GroundProjector:
    """
    负责去畸变图像和机器人地面之间的映射。
    1. UndistortedPixel <-> GroundPoint
    2. 去畸变图像 -> BEV
    """
    @staticmethod
    def _make_ground_to_bev_matrix(config: BevConfig) -> FloatArray:
        """
        机器人地面坐标系 -> BEV像素坐标系
        """
        scale = 1.0 / config.mm_per_pixel
        return np.array([
            [0, -scale, scale * config.x_min],
            [-scale, 0, scale * config.y_min],
            [0, 0, 1],
        ], dtype=np.float64)


    def __init__(self, image_to_ground: FloatArray, bev_config: BevConfig | None = None):
        """
        image_to_ground: shape=(3, 3) 的单应矩阵
        """
        self.image_to_ground = image_to_ground
        self.ground_to_image = np.linalg.inv(image_to_ground)
        self.bev_config = bev_config
        
        if bev_config is not None:
            self.bev_to_ground = self._make_ground_to_bev_matrix(bev_config)
            self.ground_to_bev = np.linalg.inv(self.bev_to_ground)
            self.image_to_bev = self.ground_to_bev @ self.image_to_ground
    
    def pixels_to_ground(self, pixels: Sequence[UndistortedPixel]) -> list[GroundPoint]:
        """
        将去畸变图像中的像素坐标转换为机器人地面坐标系中的二维点
        """
        pixel_array = pixels_to_array(pixels)  # shape=(N, 2)
        ground_points = cv2.perspectiveTransform(
            pixel_array.reshape(-1, 1, 2), self.image_to_ground
        ).reshape(-1, 2)
        return array_to_pixels(ground_points, GroundPoint)
    
    def pixel_to_ground(self, pixel: UndistortedPixel) -> GroundPoint:
        return self.pixels_to_ground([pixel])[0]

    def ground_to_pixels(self, ground_points: Sequence[GroundPoint]) -> list[UndistortedPixel]:
        """
        将机器人地面坐标系中的二维点转换为去畸变图像中的像素坐标
        """
        points_array = np.array(
            [[[p.x, p.y]] for p in ground_points],dtype=np.float64
        )
        pixel_points = cv2.perspectiveTransform(
            points_array.reshape(-1, 1, 2), self.ground_to_image
        ).reshape(-1, 2)
        return array_to_pixels(pixel_points, UndistortedPixel)

    def ground_to_pixel(self, ground_point: GroundPoint) -> UndistortedPixel:
        return self.ground_to_pixels([ground_point])[0]
    
    def make_bev_image(self, undistorted_image: np.ndarray) -> np.ndarray:
        """
        将去畸变图像转换为 BEV 图像
        """
        if self.bev_config is None:
            raise ValueError("BEV config is not set.")
        
        bev_image = cv2.warpPerspective(
            undistorted_image,
            self.image_to_bev,
            (self.bev_config.width, self.bev_config.height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        return bev_image