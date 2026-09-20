"""从目标颜色掩码估计夹爪需要的横向开口。"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
import math

import cv2
import numpy as np

from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception.detector import MODEL_GROUND_FORWARD_BIAS_MM
from rescue_vision.perception.types import TargetClass, TargetObservation


@dataclass(frozen=True, slots=True)
class GripperWidthEstimatorConfig:
    """夹爪宽度估计的门限和安全余量。"""

    center_y_half_range_mm: float = 5.0
    clearance_mm: float = 4.0
    min_mask_pixels: int = 1

    def __post_init__(self) -> None:
        if isinstance(self.center_y_half_range_mm, bool):
            raise ValueError(
                "center_y_half_range_mm must be a real number, "
                f"got {self.center_y_half_range_mm!r}."
            )
        center_range = float(self.center_y_half_range_mm)
        if not math.isfinite(center_range) or center_range <= 0.0:
            raise ValueError(
                "center_y_half_range_mm must be finite and positive, "
                f"got {self.center_y_half_range_mm!r}."
            )
        object.__setattr__(self, "center_y_half_range_mm", center_range)

        if isinstance(self.clearance_mm, bool):
            raise ValueError(
                f"clearance_mm must be a real number, got {self.clearance_mm!r}."
            )
        clearance = float(self.clearance_mm)
        if not math.isfinite(clearance) or clearance < 0.0:
            raise ValueError(
                "clearance_mm must be finite and non-negative, "
                f"got {self.clearance_mm!r}."
            )
        object.__setattr__(self, "clearance_mm", clearance)

        if (
            isinstance(self.min_mask_pixels, bool)
            or not isinstance(self.min_mask_pixels, int)
            or self.min_mask_pixels <= 0
        ):
            raise ValueError(
                "min_mask_pixels must be a positive integer, "
                f"got {self.min_mask_pixels!r}."
            )


@dataclass(frozen=True, slots=True)
class GripperWidthMeasurement:
    """单个居中目标的地面横向宽度测量结果。

    ``center_x_mm``/``center_y_mm`` 来自目标 K0；``front_x_mm`` 是颜色掩码
    地面投影的最大 ``x``，``center_to_front_mm`` 即用户定义的 ``d``。
    ``left_y_mm`` 是机器人左侧边界，即机器人坐标系中较大的 ``y``；
    ``right_y_mm`` 是机器人右侧边界，即较小的 ``y``。边界像素使用去畸变
    全图像素坐标，便于独立测试程序在预览中标记。
    """

    frame_sequence: int
    target_class: TargetClass
    center_x_mm: float
    center_y_mm: float
    left_y_mm: float
    right_y_mm: float
    front_x_mm: float
    center_to_front_mm: float
    width_mm: float
    opening_width_mm: float
    left_edge_pixel: UndistortedPixel
    right_edge_pixel: UndistortedPixel
    front_edge_pixel: UndistortedPixel

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError(
                "frame_sequence must be a non-negative integer, "
                f"got {self.frame_sequence!r}."
            )
        if not isinstance(self.target_class, TargetClass):
            raise ValueError("target_class must be a TargetClass value.")
        for name in (
            "center_x_mm",
            "center_y_mm",
            "left_y_mm",
            "right_y_mm",
            "front_x_mm",
            "center_to_front_mm",
            "width_mm",
            "opening_width_mm",
        ):
            if isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a real number.")
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")
            if name in {
                "center_to_front_mm",
                "width_mm",
                "opening_width_mm",
            } and value < 0.0:
                raise ValueError(f"{name} must be non-negative, got {value!r}.")
            object.__setattr__(self, name, value)
        if self.left_y_mm < self.right_y_mm:
            raise ValueError(
                "left_y_mm must be greater than or equal to right_y_mm, "
                f"got {self.left_y_mm} < {self.right_y_mm}."
            )
        if not math.isclose(
            self.width_mm,
            self.left_y_mm - self.right_y_mm,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "width_mm must equal left_y_mm - right_y_mm, "
                f"got {self.width_mm} versus "
                f"{self.left_y_mm - self.right_y_mm}."
            )
        if not math.isclose(
            self.center_to_front_mm,
            self.front_x_mm - self.center_x_mm,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError(
                "center_to_front_mm must equal front_x_mm - center_x_mm, "
                f"got {self.center_to_front_mm} versus "
                f"{self.front_x_mm - self.center_x_mm}."
            )
        if not isinstance(self.left_edge_pixel, UndistortedPixel):
            raise ValueError("left_edge_pixel must be an UndistortedPixel value.")
        if not isinstance(self.right_edge_pixel, UndistortedPixel):
            raise ValueError("right_edge_pixel must be an UndistortedPixel value.")
        if not isinstance(self.front_edge_pixel, UndistortedPixel):
            raise ValueError("front_edge_pixel must be an UndistortedPixel value.")
        for name, point in (
            ("left_edge_pixel", self.left_edge_pixel),
            ("right_edge_pixel", self.right_edge_pixel),
            ("front_edge_pixel", self.front_edge_pixel),
        ):
            if not math.isfinite(point.u) or not math.isfinite(point.v):
                raise ValueError(f"{name} must contain finite coordinates.")

    @property
    def forward_distance_mm(self) -> float:
        """按用户定义返回 ``center_x_mm + d_mm`` 的前进距离。"""

        return self.center_x_mm + self.center_to_front_mm


def average_gripper_width_measurements(
    measurements: Sequence[GripperWidthMeasurement],
    *,
    clearance_mm: float = 4.0,
) -> GripperWidthMeasurement:
    """平均一个采样窗口内的有效宽度测量。

    调用方可以先收集固定数量的感知帧，再只传入其中成功测得目标的帧；
    因而目标丢失帧不会影响平均值。帧序号使用最后一个有效测量的序号，
    类别使用窗口内票数最多者（票数相同时使用最后一个有效类别）。
    """

    if not isinstance(measurements, Sequence) or not measurements:
        raise ValueError("measurements must contain at least one measurement.")
    if isinstance(clearance_mm, bool):
        raise ValueError(f"clearance_mm must be a real number, got {clearance_mm!r}.")
    clearance = float(clearance_mm)
    if not math.isfinite(clearance) or clearance < 0.0:
        raise ValueError(
            "clearance_mm must be finite and non-negative, "
            f"got {clearance_mm!r}."
        )
    if not all(isinstance(item, GripperWidthMeasurement) for item in measurements):
        raise TypeError("measurements must contain GripperWidthMeasurement values.")

    def mean(values: Sequence[float]) -> float:
        return math.fsum(values) / len(values)

    target_counts = Counter(item.target_class for item in measurements)
    last_class_index = {
        target_class: max(
            index
            for index, item in enumerate(measurements)
            if item.target_class is target_class
        )
        for target_class in target_counts
    }
    target_class = max(
        target_counts,
        key=lambda value: (target_counts[value], last_class_index[value]),
    )
    width = mean([item.width_mm for item in measurements])
    return GripperWidthMeasurement(
        frame_sequence=measurements[-1].frame_sequence,
        target_class=target_class,
        center_x_mm=mean([item.center_x_mm for item in measurements]),
        center_y_mm=mean([item.center_y_mm for item in measurements]),
        left_y_mm=mean([item.left_y_mm for item in measurements]),
        right_y_mm=mean([item.right_y_mm for item in measurements]),
        front_x_mm=mean([item.front_x_mm for item in measurements]),
        center_to_front_mm=mean(
            [item.center_to_front_mm for item in measurements]
        ),
        width_mm=width,
        opening_width_mm=width + clearance,
        left_edge_pixel=UndistortedPixel(
            mean([item.left_edge_pixel.u for item in measurements]),
            mean([item.left_edge_pixel.v for item in measurements]),
        ),
        right_edge_pixel=UndistortedPixel(
            mean([item.right_edge_pixel.u for item in measurements]),
            mean([item.right_edge_pixel.v for item in measurements]),
        ),
        front_edge_pixel=UndistortedPixel(
            mean([item.front_edge_pixel.u for item in measurements]),
            mean([item.front_edge_pixel.v for item in measurements]),
        ),
    )


def _mask_pixels_in_full_image(
    observation: TargetObservation,
) -> tuple[UndistortedPixel, ...]:
    segmentation = observation.color_segmentation
    mask = segmentation.mask
    # A projective transform reaches an extrema on the foreground boundary.
    # Keep every contour pixel (including disconnected components), but avoid
    # allocating one coordinate object for every interior mask pixel.
    contours, _ = cv2.findContours(
        np.ascontiguousarray(mask),
        cv2.RETR_LIST,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        return ()
    roi = segmentation.roi_box
    # 使用像素中心而不是 ROI 左上角，减少离散化引入的边界偏差。
    return tuple(
        UndistortedPixel(
            float(roi.x_min) + float(point[0]) + 0.5,
            float(roi.y_min) + float(point[1]) + 0.5,
        )
        for contour in contours
        for point in contour.reshape(-1, 2)
    )


def _envelope_pixels(
    observation: TargetObservation,
    *,
    min_mask_pixels: int,
) -> tuple[UndistortedPixel, ...]:
    """Use HSV geometry when present, otherwise the model bbox bottom edge."""

    if cv2.countNonZero(observation.color_segmentation.mask) >= min_mask_pixels:
        pixels = _mask_pixels_in_full_image(observation)
        if pixels:
            return pixels
    box = observation.box
    return (
        UndistortedPixel(box.x_min, box.y_max),
        UndistortedPixel(box.x_max, box.y_max),
    )


def estimate_gripper_width(
    observation: TargetObservation,
    ground_projector: GroundProjector,
    config: GripperWidthEstimatorConfig | None = None,
) -> GripperWidthMeasurement | None:
    """估计一个横向居中目标的地面宽度。

    只有目标 K0 已成功投影、其中心满足严格开区间
    ``-center_y_half_range_mm < y < center_y_half_range_mm`` 时返回结果。
    HSV 掩码只提供优先几何；掩码不足或歧义时改用模型 bbox 底边。
    """

    if not isinstance(observation, TargetObservation):
        raise TypeError(
            "observation must be a TargetObservation, got "
            f"{type(observation).__name__}."
        )
    if not isinstance(ground_projector, GroundProjector):
        raise TypeError(
            "ground_projector must be a GroundProjector, got "
            f"{type(ground_projector).__name__}."
        )
    estimator_config = (
        GripperWidthEstimatorConfig() if config is None else config
    )
    if not isinstance(estimator_config, GripperWidthEstimatorConfig):
        raise TypeError(
            "config must be a GripperWidthEstimatorConfig or None, got "
            f"{type(estimator_config).__name__}."
        )

    center: GroundPoint | None = observation.ground_point
    if center is None:
        return None
    if not (
        -estimator_config.center_y_half_range_mm < center.y
        < estimator_config.center_y_half_range_mm
    ):
        return None
    pixels = _envelope_pixels(
        observation,
        min_mask_pixels=estimator_config.min_mask_pixels,
    )
    ground_points = ground_projector.pixels_to_ground(pixels)
    if len(ground_points) != len(pixels):
        raise RuntimeError(
            "GroundProjector returned a different number of points than requested."
        )
    ground_values = np.asarray(
        [[point.x, point.y] for point in ground_points],
        dtype=np.float64,
    )
    if not np.all(np.isfinite(ground_values)):
        raise ValueError("Projected mask ground coordinates must all be finite.")
    ground_y = ground_values[:, 1]
    ground_x = ground_values[:, 0] + MODEL_GROUND_FORWARD_BIAS_MM

    right_index = int(np.argmin(ground_y))
    left_index = int(np.argmax(ground_y))
    front_index = int(np.argmax(ground_x))
    right_y = float(ground_y[right_index])
    left_y = float(ground_y[left_index])
    center_x = float(center.x)
    front_x = float(ground_x[front_index])
    center_to_front = front_x - center_x
    if center_to_front < -1e-9:
        raise ValueError(
            "Projected object front is behind its ground center: "
            f"front_x={front_x:.3f}, center_x={center_x:.3f}."
        )
    width = left_y - right_y
    return GripperWidthMeasurement(
        frame_sequence=observation.frame_sequence,
        target_class=observation.target_class,
        center_x_mm=center_x,
        center_y_mm=center.y,
        left_y_mm=left_y,
        right_y_mm=right_y,
        front_x_mm=front_x,
        center_to_front_mm=center_to_front,
        width_mm=width,
        opening_width_mm=width + estimator_config.clearance_mm,
        left_edge_pixel=pixels[left_index],
        right_edge_pixel=pixels[right_index],
        front_edge_pixel=pixels[front_index],
    )


@dataclass(frozen=True, slots=True)
class TargetGroundEnvelope:
    """颜色前景在机器人地面的近似包络，不是实测底面轮廓。"""

    frame_sequence: int
    capture_timestamp_ns: int
    target_class: TargetClass
    center: GroundPoint
    corners: tuple[GroundPoint, GroundPoint, GroundPoint, GroundPoint]

    def __post_init__(self) -> None:
        for name in ("frame_sequence", "capture_timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer, got {value!r}.")
        if not isinstance(self.target_class, TargetClass):
            raise ValueError(f"Invalid target_class {self.target_class!r}.")
        if len(self.corners) != 4:
            raise ValueError(f"Expected four envelope corners, got {self.corners!r}.")
        for point in (self.center, *self.corners):
            if not isinstance(point, GroundPoint) or not all(math.isfinite(v) for v in (point.x, point.y)):
                raise ValueError(f"Expected finite GroundPoint, got {point!r}.")


def measure_target_envelope(
    observation: TargetObservation,
    ground_projector: GroundProjector,
    *,
    min_mask_pixels: int = 1,
) -> TargetGroundEnvelope | None:
    """与居中门限无关的单目标测量；无有效 K0/颜色掩码时返回缺失。"""
    if isinstance(min_mask_pixels, bool) or not isinstance(min_mask_pixels, int) or min_mask_pixels <= 0:
        raise ValueError(f"min_mask_pixels must be a positive integer, got {min_mask_pixels!r}.")
    if observation.ground_point is None:
        return None
    pixels = _envelope_pixels(observation, min_mask_pixels=min_mask_pixels)
    points = ground_projector.pixels_to_ground(pixels)
    xs = [point.x + MODEL_GROUND_FORWARD_BIAS_MM for point in points]
    ys = [point.y for point in points]
    # K0 必须包含在包络中；顶部投影误差不应产生反向的前进距离。
    xs.append(observation.ground_point.x)
    ys.append(observation.ground_point.y)
    if not all(math.isfinite(value) for value in (*xs, *ys)):
        raise ValueError("Target envelope projection contains nonfinite coordinates.")
    return TargetGroundEnvelope(
        observation.frame_sequence, observation.capture_timestamp_ns,
        observation.target_class, observation.ground_point,
        (GroundPoint(min(xs), min(ys)), GroundPoint(max(xs), min(ys)),
         GroundPoint(max(xs), max(ys)), GroundPoint(min(xs), max(ys))),
    )
