"""视觉系统使用的坐标点类型。"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Iterable
from typing import TypeVar

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

# These identifiers are written into calibration metadata.  Keep the origin
# definition here so calibration and runtime loading cannot silently diverge.
ROBOT_FRAME_ORIGIN = "midpoint_between_drive_wheel_contact_points"
FIELD_FRAME_ORIGIN = "center_cross_intersection"


def robot_frame_metadata() -> dict[str, str]:
    """Return the canonical robot-frame metadata for calibration artifacts."""

    return {
        "origin": ROBOT_FRAME_ORIGIN,
        "x": "robot_forward",
        "y": "robot_left",
        "z": "up",
        "unit": "mm",
    }


def field_frame_metadata() -> dict[str, str]:
    """Return the canonical field-frame metadata used by map consumers."""

    return {
        "origin": FIELD_FRAME_ORIGIN,
        "x": "right_along_horizontal_center_marking",
        "y": "red_safe_zone_along_vertical_center_marking",
        "unit": "mm",
    }

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
        原点为两驱动轮接地点连线的中点
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
        原点为两驱动轮接地点连线的中点
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
        原点为场地中心十字点划线的交点
        x 沿水平点划线向右增大
        y 沿竖直点划线指向红色安全区增大
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


@dataclass(frozen=True, slots=True)
class MapPixel:
    """
    场地图 PNG 中的显示像素坐标。

    原点在 PNG 左上角，``u`` 向右、``v`` 向下；它不是相机像素，也不是
    `BevPixel`。`MapSnapshotAttributes` 负责它与 `FieldPoint` 的映射。
    """

    u: float
    v: float

    def as_array(self) -> FloatArray:
        return np.array([self.u, self.v], dtype=np.float64)


PixelPoint = RawPixel | UndistortedPixel | BevPixel | MapPixel
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
