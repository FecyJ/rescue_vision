"""夹爪内侧的独立颜色证据；不检测物块、不计数、不修改模型类别。"""
from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np

from rescue_vision.geometry.types import UndistortedPixel
from rescue_vision.perception.color_segmentation import _class_mask
from rescue_vision.perception.types import (
    COLOR_TARGET_CLASSES,
    HsvColorClassifierConfig,
    TargetClass,
    UndistortedBoundingBox,
)


@dataclass(frozen=True, slots=True)
class GripperColorConfig:
    enabled: bool = True
    # 全尺寸去畸变图的归一化 (u, v)，留出夹臂和底部车体。
    polygon_normalized: tuple[tuple[float, float], ...] = (
        (0.47, 0.85), (0.53, 0.85), (0.57, 0.97), (0.43, 0.97),
    )
    min_component_fraction: float = 0.03
    black_min_thickness_fraction: float = 0.12
    shadow_min_value: int = 30
    orange_bbox_min_color_fraction: float = 0.15
    orange_distinct_max_bbox_iou: float = 0.20
    orange_distinct_min_k0_distance_px: float = 20.0

    def __post_init__(self) -> None:
        if (isinstance(self.shadow_min_value, bool)
                or not isinstance(self.shadow_min_value, int)
                or not 1 <= self.shadow_min_value <= 255):
            raise ValueError(f"gripper_color.shadow_min_value must be an integer in [1,255], got {self.shadow_min_value!r}")
        if not isinstance(self.enabled, bool):
            raise ValueError(f"gripper_color.enabled must be boolean, got {self.enabled!r}")
        points = self.polygon_normalized
        if len(points) < 3 or any(
            len(p) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float))
                              or not math.isfinite(v) or not 0 <= v <= 1 for v in p)
            for p in points
        ):
            raise ValueError(f"gripper_color.polygon_normalized requires normalized (u,v) points, got {points!r}")
        contour = np.asarray(points, np.float32)
        if not cv2.isContourConvex(contour) or cv2.contourArea(contour) <= 1e-6:
            raise ValueError(f"gripper_color polygon must be convex with positive area, got {points!r}")
        if (isinstance(self.min_component_fraction, bool)
                or not math.isfinite(self.min_component_fraction)
                or not 0 < self.min_component_fraction <= 1):
            raise ValueError(f"gripper_color.min_component_fraction must be in (0,1], got {self.min_component_fraction!r}")
        if (isinstance(self.black_min_thickness_fraction, bool)
                or not isinstance(self.black_min_thickness_fraction, (int, float))
                or not math.isfinite(self.black_min_thickness_fraction)
                or not 0 < self.black_min_thickness_fraction <= 1):
            raise ValueError(
                "gripper_color.black_min_thickness_fraction must be in (0,1], "
                f"got {self.black_min_thickness_fraction!r}"
            )
        if (
            isinstance(self.orange_bbox_min_color_fraction, bool)
            or not isinstance(self.orange_bbox_min_color_fraction, (int, float))
            or not math.isfinite(self.orange_bbox_min_color_fraction)
            or not 0.0 < self.orange_bbox_min_color_fraction <= 1.0
        ):
            raise ValueError(
                "gripper_color.orange_bbox_min_color_fraction must be finite in "
                f"(0,1], got {self.orange_bbox_min_color_fraction!r}"
            )
        if (
            isinstance(self.orange_distinct_max_bbox_iou, bool)
            or not isinstance(self.orange_distinct_max_bbox_iou, (int, float))
            or not math.isfinite(self.orange_distinct_max_bbox_iou)
            or not 0.0 <= self.orange_distinct_max_bbox_iou <= 1.0
        ):
            raise ValueError(
                "gripper_color.orange_distinct_max_bbox_iou must be finite in "
                f"[0,1], got {self.orange_distinct_max_bbox_iou!r}"
            )
        if (
            isinstance(self.orange_distinct_min_k0_distance_px, bool)
            or not isinstance(self.orange_distinct_min_k0_distance_px, (int, float))
            or not math.isfinite(self.orange_distinct_min_k0_distance_px)
            or self.orange_distinct_min_k0_distance_px <= 0.0
        ):
            raise ValueError(
                "gripper_color.orange_distinct_min_k0_distance_px must be finite "
                f"and positive, got {self.orange_distinct_min_k0_distance_px!r}"
            )


@dataclass(frozen=True, slots=True)
class GripperColorComponent:
    """一个达到门限的最大连通色块，轮廓使用全图去畸变像素坐标。"""

    target_class: TargetClass
    fraction: float
    contour: tuple[UndistortedPixel, ...]


@dataclass(frozen=True, slots=True)
class GripperColorObservation:
    """时间/帧号继承所属快照；fraction 是最大连通色块占内侧 ROI 的比例。"""

    polygon: tuple[UndistortedPixel, ...]
    component_fractions: tuple[tuple[TargetClass, float], ...]
    present_classes: frozenset[TargetClass]
    black_raw_fraction: float = 0.0
    black_chromatic_fraction: float = 0.0
    components: tuple[GripperColorComponent, ...] = ()


def bounding_box_overlaps_gripper_polygon(
    box: UndistortedBoundingBox,
    polygon: tuple[UndistortedPixel, ...],
    *,
    include_boundary: bool = False,
) -> bool:
    """Check convex ROI overlap, optionally including edge/corner contact."""

    if len(polygon) < 3:
        return False
    polygon_contour = np.asarray(
        [(point.u, point.v) for point in polygon],
        np.float32,
    )
    box_contour = np.asarray(
        [
            (box.x_min, box.y_min),
            (box.x_max, box.y_min),
            (box.x_max, box.y_max),
            (box.x_min, box.y_max),
        ],
        np.float32,
    )
    if include_boundary:
        # Separating-axis test in float64 preserves subpixel gaps and includes
        # zero-area contact. The ROI is convex by GripperColorConfig contract.
        roi = np.asarray([(point.u, point.v) for point in polygon], np.float64)
        rectangle = np.asarray([
            (box.x_min, box.y_min), (box.x_max, box.y_min),
            (box.x_max, box.y_max), (box.x_min, box.y_max),
        ], np.float64)
        for contour in (roi, rectangle):
            for edge in np.roll(contour, -1, axis=0) - contour:
                axis = np.asarray((-edge[1], edge[0]))
                roi_projection = roi @ axis
                box_projection = rectangle @ axis
                if (roi_projection.max() < box_projection.min()
                        or box_projection.max() < roi_projection.min()):
                    return False
        return True
    intersection_area, _ = cv2.intersectConvexConvex(polygon_contour, box_contour)
    return bool(intersection_area > 0.0)


def observe_gripper_colors(
    image_bgr: np.ndarray,
    config: GripperColorConfig,
    colors: HsvColorClassifierConfig,
) -> GripperColorObservation | None:
    if not config.enabled:
        return None
    if image_bgr.dtype != np.uint8 or image_bgr.ndim != 3 or image_bgr.shape[2] != 3 or min(image_bgr.shape[:2]) < 1:
        raise ValueError(f"gripper color image must be nonempty uint8 BGR, got {image_bgr.dtype}, {image_bgr.shape}")
    height, width = image_bgr.shape[:2]
    polygon = tuple(UndistortedPixel(u * (width - 1), v * (height - 1))
                    for u, v in config.polygon_normalized)
    contour = np.rint([(p.u, p.v) for p in polygon]).astype(np.int32)
    x, y, w, h = cv2.boundingRect(contour)
    interior = np.zeros((h, w), np.uint8)
    cv2.fillConvexPoly(interior, contour - (x, y), 255)
    hsv = cv2.cvtColor(image_bgr[y:y+h, x:x+w], cv2.COLOR_BGR2HSV)
    area = max(1, cv2.countNonZero(interior))
    # 目标几何 HSV 的 V 下限为亮面设计；夹爪遮阴后，仍可靠的 H/S
    # 不能因为 V<70 就被黑色范围覆盖。仅本门禁降低彩色 V 下限，
    # 沿用配置的 H/S 和上界，既保留暗绿，也保留暗蓝/暗橙的冲突证据。
    masks = {
        target_class: cv2.bitwise_and(_class_mask(
            hsv, target_class, colors,
            shadow_min_value=(None if target_class is TargetClass.BLACK_CORE else config.shadow_min_value),
        ), interior)
        for target_class in COLOR_TARGET_CLASSES
    }
    chromatic = np.zeros_like(interior)
    for target_class, mask in masks.items():
        if target_class is not TargetClass.BLACK_CORE:
            chromatic |= mask
    black = masks[TargetClass.BLACK_CORE]
    black_raw_fraction = cv2.countNonZero(black) / area
    black_chromatic_fraction = cv2.countNonZero(cv2.bitwise_and(black, chromatic)) / area
    masks[TargetClass.BLACK_CORE] = cv2.bitwise_and(black, cv2.bitwise_not(chromatic))
    fractions = []
    components = []
    for target_class in COLOR_TARGET_CLASSES:
        mask = masks[target_class]
        if target_class is TargetClass.BLACK_CORE:
            # 圆形开运算去除细线，不按整个连通域的长宽比删物块：
            # 黑色四面体与点画线相连时，仍保留厚实的实体核心。
            diameter = max(3, math.ceil(min(w, h) * config.black_min_thickness_fraction))
            diameter += 1 - diameter % 2
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (diameter, diameter))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                                    borderType=cv2.BORDER_CONSTANT, borderValue=0)
        _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        largest_label = (
            int(np.argmax(stats[1:, cv2.CC_STAT_AREA])) + 1
            if len(stats) > 1
            else 0
        )
        largest = (
            int(stats[largest_label, cv2.CC_STAT_AREA])
            if largest_label
            else 0
        )
        fraction = largest / area
        fractions.append((target_class, fraction))
        if fraction < config.min_component_fraction:
            continue
        component_mask = np.where(labels == largest_label, 255, 0).astype(np.uint8)
        contours, _ = cv2.findContours(
            component_mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE,
        )
        if not contours:
            continue
        component_contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
        components.append(GripperColorComponent(
            target_class=target_class,
            fraction=fraction,
            contour=tuple(
                UndistortedPixel(float(u + x), float(v + y))
                for u, v in component_contour
            ),
        ))
    return GripperColorObservation(
        polygon=polygon,
        component_fractions=tuple(fractions),
        present_classes=frozenset(component.target_class for component in components),
        black_raw_fraction=black_raw_fraction,
        black_chromatic_fraction=black_chromatic_fraction,
        components=tuple(components),
    )
