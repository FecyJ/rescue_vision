"""任务目标感知的数据契约与可替换推理后端。"""

from rescue_vision.perception.backend import FakeInferenceBackend, InferenceBackend
from rescue_vision.perception.detector import TargetPoseDetector
from rescue_vision.perception.types import (
    ClassProbabilities,
    ModelDetection,
    ObservationQuality,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)

__all__ = [
    "ClassProbabilities",
    "FakeInferenceBackend",
    "InferenceBackend",
    "ModelDetection",
    "ObservationQuality",
    "TargetClass",
    "TargetObservation",
    "TargetPoseDetector",
    "UndistortedBoundingBox",
]
