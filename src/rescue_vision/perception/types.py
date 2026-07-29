"""任务目标模型输出与统一观测类型。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel


class TargetClass(str, Enum):
    """与 ``docs/Pose视觉模型约定.md`` 一致的任务目标标签。"""

    GREEN_SUPPLY = "green_supply"
    BLACK_CORE = "black_core"
    ORANGE_INJURED = "orange_injured"
    BLUE_DANGER = "blue_danger"
    UNKNOWN = "unknown"


class ObservationQuality(str, Enum):
    """不会被静默丢弃的观测质量信息。"""

    LOW_CLASS_CONFIDENCE = "low_class_confidence"
    K0_UNAVAILABLE = "k0_unavailable"


def _probability(value: float, location: str) -> float:
    converted = float(value)
    if not math.isfinite(converted) or not 0.0 <= converted <= 1.0:
        raise ValueError(f"{location} must be finite and in [0, 1], got {value!r}.")
    return converted


@dataclass(frozen=True, slots=True)
class ClassProbabilities:
    """四类任务目标和运行时 ``unknown`` 的概率分布。"""

    green_supply: float
    black_core: float
    orange_injured: float
    blue_danger: float
    unknown: float

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
        values = {item: 0.0 for item in TargetClass}
        if target_class is TargetClass.UNKNOWN:
            values[TargetClass.UNKNOWN] = 1.0
        else:
            values[target_class] = confidence
            values[TargetClass.UNKNOWN] = 1.0 - confidence
        return cls(
            green_supply=values[TargetClass.GREEN_SUPPLY],
            black_core=values[TargetClass.BLACK_CORE],
            orange_injured=values[TargetClass.ORANGE_INJURED],
            blue_danger=values[TargetClass.BLUE_DANGER],
            unknown=values[TargetClass.UNKNOWN],
        )

    def as_dict(self) -> dict[str, float]:
        return {
            TargetClass.GREEN_SUPPLY.value: self.green_supply,
            TargetClass.BLACK_CORE.value: self.black_core,
            TargetClass.ORANGE_INJURED.value: self.orange_injured,
            TargetClass.BLUE_DANGER.value: self.blue_danger,
            TargetClass.UNKNOWN.value: self.unknown,
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
class ModelDetection:
    """推理后端输出；坐标已经反映射到去畸变输入图像。"""

    model_class_id: int
    confidence: float
    box: UndistortedBoundingBox
    k0: UndistortedPixel | None
    k0_confidence: float

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
        _probability(self.k0_confidence, "k0_confidence")
        if self.k0 is not None and (
            not math.isfinite(self.k0.u) or not math.isfinite(self.k0.v)
        ):
            raise ValueError(f"k0 must be finite, got {self.k0!r}.")


@dataclass(frozen=True, slots=True)
class TargetObservation:
    """一帧中的单个任务目标观测。"""

    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    image_size: tuple[int, int]
    target_class: TargetClass
    class_probabilities: ClassProbabilities
    detection_confidence: float
    box: UndistortedBoundingBox
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
        if not isinstance(self.class_probabilities, ClassProbabilities):
            raise ValueError(
                "class_probabilities must be a ClassProbabilities value."
            )
        _probability(self.detection_confidence, "detection_confidence")
        self.box.validate_image_size(self.image_size)
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
