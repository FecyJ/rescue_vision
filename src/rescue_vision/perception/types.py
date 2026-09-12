"""任务目标模型输出与统一观测类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

import numpy as np
import numpy.typing as npt

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel


class TargetClass(str, Enum):
    """与 ``docs/Pose视觉模型约定.md`` 一致的任务目标标签。"""

    GREEN_SUPPLY = "green_supply"
    BLACK_CORE = "black_core"
    ORANGE_INJURED = "orange_injured"
    BLUE_DANGER = "blue_danger"


class PoseModelClass(str, Enum):
    """YOLO Pose v3 固定训练类别；顺序也是模型 class ID。"""

    GREEN_SUPPLY = "green_supply"
    BLACK_CORE = "black_core"
    ORANGE_INJURED = "orange_injured"
    BLUE_DANGER = "blue_danger"
    CENTER_CROSS = "center_cross"
    SAFE_ZONE = "safe_zone"


POSE_MODEL_CLASSES = tuple(PoseModelClass)


class ObservationQuality(str, Enum):
    """不会被静默丢弃的观测质量信息。"""

    COLOR_EVIDENCE_INSUFFICIENT = "color_evidence_insufficient"
    COLOR_EVIDENCE_AMBIGUOUS = "color_evidence_ambiguous"
    HIGH_CONFIDENCE_COLOR_OVERRIDE = "high_confidence_color_override"
    POSE_COLOR_CONFLICT = "pose_color_conflict"
    K0_UNAVAILABLE = "k0_unavailable"


class ColorSegmentationStatus(str, Enum):
    """ROI 颜色分类的判定状态。"""

    ACCEPTED = "accepted"
    INSUFFICIENT = "insufficient"
    AMBIGUOUS = "ambiguous"


def _probability(value: float, location: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{location} must be finite and in [0, 1], got {value!r}.")
    return converted


@dataclass(frozen=True, slots=True)
class ClassProbabilities:
    """四类任务目标的条件类别分布；检测置信度独立保存。"""

    green_supply: float
    black_core: float
    orange_injured: float
    blue_danger: float

    def __post_init__(self) -> None:
        values = self.as_dict()
        for name, value in values.items():
            _probability(value, f"class_probabilities.{name}")
        total = sum(values.values())
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                f"class probabilities must sum to 1.0, got {total!r}."
            )

    @classmethod
    def from_top_class(
        cls,
        target_class: TargetClass,
        confidence: float,
    ) -> ClassProbabilities:
        """把只有 top-1 分数的模型结果保守转换为完整分布。"""

        confidence = _probability(confidence, "confidence")
        # Top-1 backends do not provide calibrated alternative-class scores.
        # Keep detection confidence on TargetObservation, not as a fifth class.
        return cls(**{item.value: float(item is target_class) for item in TargetClass})

    def as_dict(self) -> dict[str, float]:
        return {
            TargetClass.GREEN_SUPPLY.value: self.green_supply,
            TargetClass.BLACK_CORE.value: self.black_core,
            TargetClass.ORANGE_INJURED.value: self.orange_injured,
            TargetClass.BLUE_DANGER.value: self.blue_danger,
        }

    def probability(self, target_class: TargetClass) -> float:
        return self.as_dict()[target_class.value]


@dataclass(frozen=True, slots=True)
class UndistortedBoundingBox:
    """去畸变图像中的水平矩形框，边界顺序为左、上、右、下。"""

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        values = (self.x_min, self.y_min, self.x_max, self.y_max)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"bounding box must be finite, got {values!r}.")
        if self.x_min < 0.0 or self.y_min < 0.0:
            raise ValueError(f"bounding box minima must be non-negative, got {values!r}.")
        if self.x_max <= self.x_min or self.y_max <= self.y_min:
            raise ValueError(f"bounding box must have positive area, got {values!r}.")

    def validate_image_size(self, image_size: tuple[int, int]) -> None:
        width, height = image_size
        if width <= 0 or height <= 0:
            raise ValueError(f"image_size must be positive, got {image_size!r}.")
        if self.x_max > width or self.y_max > height:
            raise ValueError(
                f"bounding box {self!r} is outside image_size {image_size!r}."
            )

    @property
    def area(self) -> float:
        return (self.x_max - self.x_min) * (self.y_max - self.y_min)

    def iou(self, other: UndistortedBoundingBox) -> float:
        width = max(0.0, min(self.x_max, other.x_max) - max(self.x_min, other.x_min))
        height = max(0.0, min(self.y_max, other.y_max) - max(self.y_min, other.y_min))
        intersection = width * height
        union = self.area + other.area - intersection
        return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True, slots=True)
class HsvRange:
    """OpenCV HSV 闭区间，H 为 0..179，S/V 为 0..255。"""

    lower: tuple[int, int, int]
    upper: tuple[int, int, int]

    def __post_init__(self) -> None:
        for name, value in (("lower", self.lower), ("upper", self.upper)):
            if (
                not isinstance(value, tuple)
                or len(value) != 3
                or any(
                    isinstance(component, bool)
                    or not isinstance(component, int)
                    for component in value
                )
            ):
                raise ValueError(
                    f"{name} must be an integer (H, S, V) tuple, got {value!r}."
                )
        limits = (179, 255, 255)
        for index, (lower, upper, limit) in enumerate(
            zip(self.lower, self.upper, limits, strict=True)
        ):
            if not 0 <= lower <= upper <= limit:
                raise ValueError(
                    f"HSV component {index} must satisfy 0 <= lower <= upper "
                    f"<= {limit}, got {lower}..{upper}."
                )

    def overlaps(self, other: HsvRange) -> bool:
        return all(
            max(first_lower, second_lower) <= min(first_upper, second_upper)
            for first_lower, first_upper, second_lower, second_upper in zip(
                self.lower,
                self.upper,
                other.lower,
                other.upper,
                strict=True,
            )
        )


COLOR_TARGET_CLASSES = (
    TargetClass.GREEN_SUPPLY,
    TargetClass.BLACK_CORE,
    TargetClass.ORANGE_INJURED,
    TargetClass.BLUE_DANGER,
)


@dataclass(frozen=True, slots=True)
class HsvColorClassifierConfig:
    """ROI HSV 分类、分割及基础去噪参数。"""

    green_supply: tuple[HsvRange, ...]
    black_core: tuple[HsvRange, ...]
    orange_injured: tuple[HsvRange, ...]
    blue_danger: tuple[HsvRange, ...]
    min_color_fraction: float
    min_color_dominance: float
    min_dominance_margin: float
    morphology_kernel_size: int
    open_iterations: int
    close_iterations: int
    min_component_area_fraction: float

    def __post_init__(self) -> None:
        for target_class in COLOR_TARGET_CLASSES:
            ranges = self.ranges_for(target_class)
            if (
                not isinstance(ranges, tuple)
                or not ranges
                or not all(isinstance(item, HsvRange) for item in ranges)
            ):
                raise ValueError(
                    f"{target_class.value} must contain at least one HsvRange."
                )
        for name, value in (
            ("min_color_fraction", self.min_color_fraction),
            ("min_color_dominance", self.min_color_dominance),
            ("min_dominance_margin", self.min_dominance_margin),
            ("min_component_area_fraction", self.min_component_area_fraction),
        ):
            _probability(value, name)
        if (
            isinstance(self.morphology_kernel_size, bool)
            or not isinstance(self.morphology_kernel_size, int)
            or self.morphology_kernel_size <= 0
            or self.morphology_kernel_size % 2 == 0
        ):
            raise ValueError(
                "morphology_kernel_size must be a positive odd integer."
            )
        for name, value in (
            ("open_iterations", self.open_iterations),
            ("close_iterations", self.close_iterations),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
        class_ranges = tuple(
            (target_class, hsv_range)
            for target_class in COLOR_TARGET_CLASSES
            for hsv_range in self.ranges_for(target_class)
        )
        for index, (first_class, first_range) in enumerate(class_ranges):
            for second_class, second_range in class_ranges[index + 1 :]:
                if (
                    first_class is not second_class
                    and first_range.overlaps(second_range)
                ):
                    raise ValueError(
                        f"HSV ranges for {first_class.value} and "
                        f"{second_class.value} must not overlap."
                    )

    def ranges_for(self, target_class: TargetClass) -> tuple[HsvRange, ...]:
        return getattr(self, target_class.value)


Uint8Array = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class RoiColorSegmentation:
    """去畸变检测框 ROI 内的顶部颜色候选及只读二值掩码。"""

    candidate_class: TargetClass | None
    status: ColorSegmentationStatus
    roi_box: UndistortedBoundingBox
    mask: Uint8Array
    color_fraction: float
    dominance: float

    def __post_init__(self) -> None:
        if self.candidate_class is not None and not isinstance(self.candidate_class, TargetClass):
            raise ValueError("candidate_class must be a TargetClass.")
        if not isinstance(self.status, ColorSegmentationStatus):
            raise ValueError("status must be a ColorSegmentationStatus.")
        if not isinstance(self.roi_box, UndistortedBoundingBox):
            raise ValueError("roi_box must be an UndistortedBoundingBox.")
        bounds = (
            self.roi_box.x_min,
            self.roi_box.y_min,
            self.roi_box.x_max,
            self.roi_box.y_max,
        )
        if any(not float(value).is_integer() for value in bounds):
            raise ValueError(f"roi_box bounds must be integer-valued, got {bounds!r}.")
        expected_shape = (
            int(self.roi_box.y_max - self.roi_box.y_min),
            int(self.roi_box.x_max - self.roi_box.x_min),
        )
        mask = np.asarray(self.mask)
        if mask.dtype != np.uint8 or mask.ndim != 2 or mask.shape != expected_shape:
            raise ValueError(
                "mask must be a uint8 ROI array with shape "
                f"{expected_shape}, got dtype={mask.dtype}, shape={mask.shape}."
            )
        if np.any((mask != 0) & (mask != 255)):
            raise ValueError("mask values must be 0 or 255.")
        owned_mask = np.ascontiguousarray(mask).copy()
        owned_mask.flags.writeable = False
        object.__setattr__(self, "mask", owned_mask)
        _probability(self.color_fraction, "color_fraction")
        _probability(self.dominance, "dominance")
        if self.candidate_class is None and (
            np.any(owned_mask)
            or self.color_fraction != 0.0
            or self.dominance != 0.0
        ):
            raise ValueError(
                "absent color candidate requires an empty mask and zero evidence."
            )
        if self.status is ColorSegmentationStatus.ACCEPTED and (
            self.candidate_class is None
            or not np.any(owned_mask)
            or self.color_fraction == 0.0
            or self.dominance == 0.0
        ):
            raise ValueError(
                "accepted color segmentation requires a known non-empty candidate."
            )
        if (
            self.candidate_class is not None
            and (
                not np.any(owned_mask)
                or self.color_fraction == 0.0
                or self.dominance == 0.0
            )
        ):
            raise ValueError(
                "known color candidate requires a non-empty mask and evidence."
            )


@dataclass(frozen=True, slots=True)
class PoseKeypoint:
    """单个 v3 Pose 槽位；低置信度点仍保留原始分数供领域层门控。"""

    point: UndistortedPixel | None
    confidence: float

    def __post_init__(self) -> None:
        _probability(self.confidence, "confidence")
        if self.point is not None and (
            not isinstance(self.point, UndistortedPixel)
            or not math.isfinite(self.point.u)
            or not math.isfinite(self.point.v)
        ):
            raise ValueError(f"point must be a finite UndistortedPixel, got {self.point!r}.")
        if self.point is None and self.confidence != 0.0:
            raise ValueError("an unavailable keypoint must have zero confidence.")


@dataclass(frozen=True, slots=True)
class ModelDetection:
    """推理后端输出；坐标已经反映射到去畸变输入图像。"""

    model_class_id: int
    confidence: float
    box: UndistortedBoundingBox
    keypoints: tuple[PoseKeypoint, PoseKeypoint, PoseKeypoint]

    def __post_init__(self) -> None:
        if (
            isinstance(self.model_class_id, bool)
            or not isinstance(self.model_class_id, int)
            or self.model_class_id < 0
        ):
            raise ValueError(
                f"model_class_id must be a non-negative integer, got "
                f"{self.model_class_id!r}."
            )
        _probability(self.confidence, "confidence")
        if self.model_class_id >= len(POSE_MODEL_CLASSES):
            raise ValueError(
                f"model_class_id must be in [0, {len(POSE_MODEL_CLASSES) - 1}], "
                f"got {self.model_class_id}."
            )
        if (
            not isinstance(self.keypoints, tuple)
            or len(self.keypoints) != 3
            or not all(isinstance(item, PoseKeypoint) for item in self.keypoints)
        ):
            raise ValueError("keypoints must contain exactly three PoseKeypoint values.")
        model_class = POSE_MODEL_CLASSES[self.model_class_id]
        if model_class is not PoseModelClass.SAFE_ZONE and any(
            item.point is not None for item in self.keypoints[1:]
        ):
            raise ValueError(
                f"{model_class.value} must not expose K1/K2 in YOLO Pose v3."
            )
        # if (
        #     model_class is PoseModelClass.SAFE_ZONE
        #     and self.keypoints[1].point is not None
        #     and self.keypoints[2].point is not None
        #     and self.keypoints[1].point.u >= self.keypoints[2].point.u
        # ):
        #     raise ValueError("safe_zone keypoints must satisfy u(K1) < u(K2).")

    @property
    def model_class(self) -> PoseModelClass:
        return POSE_MODEL_CLASSES[self.model_class_id]


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """一帧中的单个任务目标观测。"""

    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    image_size: tuple[int, int]
    model_target_class: TargetClass
    target_class: TargetClass
    class_probabilities: ClassProbabilities
    detection_confidence: float
    box: UndistortedBoundingBox
    color_segmentation: RoiColorSegmentation
    k0: UndistortedPixel | None
    k0_confidence: float
    ground_point: GroundPoint | None
    quality: frozenset[ObservationQuality]

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative.")
        if (
            isinstance(self.capture_timestamp_ns, bool)
            or not isinstance(self.capture_timestamp_ns, int)
            or self.capture_timestamp_ns < 0
        ):
            raise ValueError("capture_timestamp_ns must be non-negative.")
        if (
            isinstance(self.result_timestamp_ns, bool)
            or not isinstance(self.result_timestamp_ns, int)
            or self.result_timestamp_ns < self.capture_timestamp_ns
        ):
            raise ValueError(
                "result_timestamp_ns must not be earlier than capture_timestamp_ns."
            )
        if (
            not isinstance(self.image_size, tuple)
            or len(self.image_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.image_size
            )
        ):
            raise ValueError(
                f"image_size must be positive integer (width, height), got "
                f"{self.image_size!r}."
            )
        if not isinstance(self.target_class, TargetClass):
            raise ValueError("target_class must be a TargetClass value.")
        if not isinstance(self.model_target_class, TargetClass):
            raise ValueError("model_target_class must be a TargetClass value.")
        if not isinstance(self.class_probabilities, ClassProbabilities):
            raise ValueError(
                "class_probabilities must be a ClassProbabilities value."
            )
        _probability(self.detection_confidence, "detection_confidence")
        self.box.validate_image_size(self.image_size)
        if not isinstance(self.color_segmentation, RoiColorSegmentation):
            raise ValueError(
                "color_segmentation must be a RoiColorSegmentation value."
            )
        self.color_segmentation.roi_box.validate_image_size(self.image_size)
        expected_roi = (
            math.floor(self.box.x_min),
            math.floor(self.box.y_min),
            math.ceil(self.box.x_max),
            math.ceil(self.box.y_max),
        )
        actual_roi = (
            self.color_segmentation.roi_box.x_min,
            self.color_segmentation.roi_box.y_min,
            self.color_segmentation.roi_box.x_max,
            self.color_segmentation.roi_box.y_max,
        )
        if actual_roi != expected_roi:
            raise ValueError(
                f"color_segmentation roi_box {actual_roi!r} does not match "
                f"rasterized detection box {expected_roi!r}."
            )
        _probability(self.k0_confidence, "k0_confidence")
        if self.k0 is not None:
            width, height = self.image_size
            if not (
                math.isfinite(self.k0.u)
                and math.isfinite(self.k0.v)
                and 0.0 <= self.k0.u < width
                and 0.0 <= self.k0.v < height
            ):
                raise ValueError(
                    f"k0 {self.k0!r} is outside image_size {self.image_size!r}."
                )
        if self.ground_point is not None and self.k0 is None:
            raise ValueError("ground_point requires an available k0.")
        if self.ground_point is not None and not (
            math.isfinite(self.ground_point.x)
            and math.isfinite(self.ground_point.y)
        ):
            raise ValueError(f"ground_point must be finite, got {self.ground_point!r}.")
        if not all(isinstance(item, ObservationQuality) for item in self.quality):
            raise ValueError("quality must contain only ObservationQuality values.")
