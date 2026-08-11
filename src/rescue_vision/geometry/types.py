"""视觉系统使用的坐标点类型。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

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
class RobotPoint3D:
    """
    机器人坐标系中的三维点，单位 mm。

    坐标约定：
        x 向机器人前方增大
        y 向机器人左方增大
        z 向上增大
    """

    x: float
    y: float
    z: float


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


PixelPoint = RawPixel | UndistortedPixel | BevPixel
Point2D = PixelPoint | GroundPoint | FieldPoint
PointT = TypeVar("PointT", bound=Point2D)


def pixels_to_array(pixels: Iterable[PixelPoint]) -> FloatArray:
    """
    将像素点列表转换为 shape=(N, 2) 的二维数组
    """
    values = [[point.u, point.v] for point in pixels]
    return np.asarray(values, dtype=np.float64).reshape(-1, 2)


def array_to_points(
    array: FloatArray,
    point_type: Callable[[float, float], PointT],
) -> list[PointT]:
    """
    将 shape=(N, 2) 的二维数组转换为指定二维点列表。
    """
    return [point_type(float(a), float(b)) for a, b in array.reshape(-1, 2)]


# 兼容已有调用；名称将在上层迁移完成后移除。
array_to_pixels = array_to_points
