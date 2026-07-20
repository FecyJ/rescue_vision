"""
各坐标系点的定义
"""

from dataclasses import dataclass
from typing import Iterable, TypeVar

import numpy as np
import numpy.typing as npt

FloatArray= npt.NDArray[np.float64]

@dataclass(frozen=True, slots=True)
class RawPixel:
    """
    原始相机图像中的畸变像素坐标
    坐标约定：
        u 向右增大
        v 向下增大
    """
    u: float
    v: float
    
    def as_array(self) -> FloatArray:
        return np.array([self.u, self.v], dtype=np.float64)

@dataclass(frozen=True, slots=True)
class UndistortedPixel:
    """
    去畸变图像中的像素坐标
    """
    u: float
    v: float
    
    def as_array(self) -> FloatArray:
        return np.array([self.u, self.v], dtype=np.float64)

@dataclass(frozen=True, slots=True)
class GroundPoint:
    """
    机器人地面坐标系中的二维点，单位 mm
    坐标约定：
        x 向机器人前方增大
        y 向机器人左方增大
    """
    x: float
    y: float


@dataclass(frozen=True, slots=True)
class FieldPoint:
    """
    场地全局坐标系中的二维点，单位 mm
    坐标约定：
        原点定义为场地中心
        x 右方增大
        y 上方增大
    """
    x: float
    y: float

@dataclass(frozen=True, slots=True)
class BevPixel:
    """
    鸟瞰图中的像素坐标
    坐标约定：
        图像上方对应机器人前方，即 v- -> x+
        图像左侧对应机器人左方，即 u- -> y+
    """
    u: float
    v: float
    
    def as_array(self) -> FloatArray:
        return np.array([self.u, self.v], dtype=np.float64)


def pixels_to_array(pixels: Iterable[RawPixel | UndistortedPixel | BevPixel]) -> FloatArray:
    """
    将像素点列表转换为 shape=(N, 2) 的二维数组
    """
    return np.array([[p.u, p.v] for p in pixels], dtype=np.float64)

def array_to_pixels(array: FloatArray, pixel_type: Iterable[RawPixel | UndistortedPixel | BevPixel]) -> list[RawPixel | UndistortedPixel | BevPixel]:
    """
    将 shape=(N, 2) 的二维数组转换为像素点列表
    """
    return [pixel_type(u, v) for u, v in array.reshape(-1, 2)]