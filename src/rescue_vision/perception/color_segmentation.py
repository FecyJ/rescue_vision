"""检测框 ROI 内的 HSV 颜色分类与分割。"""

from __future__ import annotations

import math

import cv2
import numpy as np

from rescue_vision.perception.types import (
    COLOR_TARGET_CLASSES,
    ClassProbabilities,
    ColorSegmentationStatus,
    HsvColorClassifierConfig,
    RoiColorSegmentation,
    TargetClass,
    UndistortedBoundingBox,
)


def _clean_mask(
    mask: np.ndarray,
    config: HsvColorClassifierConfig,
) -> np.ndarray:
    kernel = np.ones(
        (config.morphology_kernel_size, config.morphology_kernel_size),
        dtype=np.uint8,
    )
    cleaned = mask
    if config.open_iterations:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_OPEN,
            kernel,
            iterations=config.open_iterations,
        )
    if config.close_iterations:
        cleaned = cv2.morphologyEx(
            cleaned,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=config.close_iterations,
        )

    minimum_area = max(
        1,
        math.ceil(config.min_component_area_fraction * cleaned.size),
    )
    if minimum_area <= 1 or not np.any(cleaned):
        return cleaned
    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        cleaned,
        connectivity=8,
    )
    keep = np.zeros(component_count, dtype=bool)
    if component_count > 1:
        keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= minimum_area
    return np.where(keep[labels], 255, 0).astype(np.uint8)


def _class_mask(
    roi_hsv: np.ndarray,
    target_class: TargetClass,
    config: HsvColorClassifierConfig,
) -> np.ndarray:
    mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
    for hsv_range in config.ranges_for(target_class):
        current = cv2.inRange(
            roi_hsv,
            np.asarray(hsv_range.lower, dtype=np.uint8),
            np.asarray(hsv_range.upper, dtype=np.uint8),
        )
        mask = cv2.bitwise_or(mask, current)
    return _clean_mask(mask, config)


def segment_roi_colors(
    image_hsv: np.ndarray,
    box: UndistortedBoundingBox,
    config: HsvColorClassifierConfig,
    *, target_class: TargetClass | None = None,
) -> tuple[RoiColorSegmentation, ClassProbabilities]:
    """在一个已校验检测框中产生顶部颜色掩码和最终分类概率。"""

    if (
        image_hsv.dtype != np.uint8
        or image_hsv.ndim != 3
        or image_hsv.shape[2] != 3
    ):
        raise ValueError(
            "image_hsv must be a uint8 array with shape (height, width, 3), "
            f"got dtype={image_hsv.dtype}, shape={image_hsv.shape}."
        )
    image_size = (int(image_hsv.shape[1]), int(image_hsv.shape[0]))
    box.validate_image_size(image_size)
    x_min = math.floor(box.x_min)
    y_min = math.floor(box.y_min)
    x_max = min(image_size[0], math.ceil(box.x_max))
    y_max = min(image_size[1], math.ceil(box.y_max))
    roi_box = UndistortedBoundingBox(
        float(x_min),
        float(y_min),
        float(x_max),
        float(y_max),
    )
    roi_hsv = image_hsv[y_min:y_max, x_min:x_max]
    masks = {
        target_class: _class_mask(roi_hsv, target_class, config)
        for target_class in COLOR_TARGET_CLASSES
    }
    counts = {
        target_class: int(cv2.countNonZero(mask))
        for target_class, mask in masks.items()
    }
    total_colored = sum(counts.values())
    if total_colored == 0:
        empty_mask = np.zeros(roi_hsv.shape[:2], dtype=np.uint8)
        return (
            RoiColorSegmentation(
                candidate_class=None,
                status=ColorSegmentationStatus.INSUFFICIENT,
                roi_box=roi_box,
                mask=empty_mask,
                color_fraction=0.0,
                dominance=0.0,
            ),
            ClassProbabilities(0.25, 0.25, 0.25, 0.25),
        )

    ranked = sorted(
        COLOR_TARGET_CLASSES,
        key=lambda target_class: counts[target_class],
        reverse=True,
    )
    candidate_class = ranked[0] if target_class is None else target_class
    candidate_count = counts[candidate_class]
    runner_up_count = counts[ranked[1]]
    roi_area = roi_hsv.shape[0] * roi_hsv.shape[1]
    color_fraction = candidate_count / roi_area
    dominance = candidate_count / total_colored
    dominance_margin = (candidate_count - runner_up_count) / total_colored

    if color_fraction < config.min_color_fraction:
        status = ColorSegmentationStatus.INSUFFICIENT
    elif (
        dominance < config.min_color_dominance
        or dominance_margin < config.min_dominance_margin
    ):
        status = ColorSegmentationStatus.AMBIGUOUS
    else:
        status = ColorSegmentationStatus.ACCEPTED

    segmentation = RoiColorSegmentation(
        candidate_class=candidate_class if candidate_count else None,
        status=status,
        roi_box=roi_box,
        mask=masks[candidate_class],
        color_fraction=color_fraction,
        dominance=dominance,
    )
    probabilities = ClassProbabilities(
        **{item.value: counts[item] / total_colored for item in COLOR_TARGET_CLASSES}
    )
    return segmentation, probabilities


def segment_bgr_roi_colors(
    image_bgr: np.ndarray,
    box: UndistortedBoundingBox,
    config: HsvColorClassifierConfig,
    *, target_class: TargetClass | None = None,
) -> tuple[RoiColorSegmentation, ClassProbabilities]:
    """只把一个检测 ROI 转换到 HSV 后执行颜色分类。

    检测器通常只收到少量目标；避免先把整张高分辨率图像转换成 HSV，
    同时保留 ``segment_roi_colors`` 的绝对 ROI 坐标和掩码契约。
    """

    if (
        image_bgr.dtype != np.uint8
        or image_bgr.ndim != 3
        or image_bgr.shape[2] != 3
    ):
        raise ValueError(
            "image_bgr must be a uint8 array with shape (height, width, 3), "
            f"got dtype={image_bgr.dtype}, shape={image_bgr.shape}."
        )
    image_size = (int(image_bgr.shape[1]), int(image_bgr.shape[0]))
    box.validate_image_size(image_size)
    x_min = math.floor(box.x_min)
    y_min = math.floor(box.y_min)
    x_max = min(image_size[0], math.ceil(box.x_max))
    y_max = min(image_size[1], math.ceil(box.y_max))
    roi_bgr = image_bgr[y_min:y_max, x_min:x_max]
    roi_hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    local_box = UndistortedBoundingBox(
        0.0,
        0.0,
        float(x_max - x_min),
        float(y_max - y_min),
    )
    segmentation, probabilities = segment_roi_colors(roi_hsv, local_box, config, target_class=target_class)
    return (
        RoiColorSegmentation(
            candidate_class=segmentation.candidate_class,
            status=segmentation.status,
            roi_box=UndistortedBoundingBox(
                float(x_min),
                float(y_min),
                float(x_max),
                float(y_max),
            ),
            mask=segmentation.mask,
            color_fraction=segmentation.color_fraction,
            dominance=segmentation.dominance,
        ),
        probabilities,
    )
