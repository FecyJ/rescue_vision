"""任务目标感知的数据契约与可替换推理后端。"""

from rescue_vision.perception.backend import FakeInferenceBackend, InferenceBackend
from rescue_vision.perception.detector import (
    RealtimeDetectionResult,
    StaleObservationError,
    TargetPoseDetector,
)
from rescue_vision.perception.types import (
    COLOR_TARGET_CLASSES,
    ClassProbabilities,
    ColorSegmentationStatus,
    HsvColorClassifierConfig,
    HsvRange,
    ModelDetection,
    ObservationQuality,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)

__all__ = [
    "COLOR_TARGET_CLASSES",
    "ClassProbabilities",
    "ColorSegmentationStatus",
    "FakeInferenceBackend",
    "HsvColorClassifierConfig",
    "HsvRange",
    "InferenceBackend",
    "ModelDetection",
    "ObservationQuality",
    "RealtimeDetectionResult",
    "RoiColorSegmentation",
    "StaleObservationError",
    "TargetClass",
    "TargetObservation",
    "TargetPoseDetector",
    "UndistortedBoundingBox",
]
