"""把模型检测转换为带时间、语义和地面坐标的统一观测。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import math
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception.backend import InferenceBackend
from rescue_vision.perception.color_segmentation import segment_roi_colors
from rescue_vision.perception.field_boundary import FieldBoundaryMask, FieldMaskState
from rescue_vision.perception.types import (
    ClassProbabilities,
    ColorSegmentationStatus,
    HsvColorClassifierConfig,
    ModelDetection,
    ObservationQuality,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
)


class StaleObservationError(ValueError):
    """推理完成时，输入帧已经超过允许的观测年龄。"""

    def __init__(self, age_ms: float, max_age_ms: float) -> None:
        self.age_ms = float(age_ms)
        self.max_age_ms = float(max_age_ms)
        super().__init__(
            f"Frame observation age {self.age_ms:.3f} ms exceeds "
            f"{self.max_age_ms:.3f} ms."
        )


@dataclass(frozen=True, slots=True)
class RealtimeDetectionResult:
    """实时检测结果；过期观测只记录丢弃原因，不向下游泄漏。"""

    observations: tuple[TargetObservation, ...]
    dropped_stale_age_ms: float | None = None

    @property
    def stale_dropped(self) -> bool:
        return self.dropped_stale_age_ms is not None


@dataclass(frozen=True, slots=True)
class _ProcessedDetection:
    detection: ModelDetection
    model_target_class: TargetClass
    target_class: TargetClass
    class_probabilities: ClassProbabilities
    color_segmentation: RoiColorSegmentation
    k0: UndistortedPixel | None
    ground_point: GroundPoint | None
    quality: frozenset[ObservationQuality]


class TargetPoseDetector:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        class_mapping: Mapping[int, TargetClass],
        detection_threshold: float,
        k0_threshold: float,
        color_classifier: HsvColorClassifierConfig,
        max_observation_age_ms: float,
        ground_projector: GroundProjector | None = None,
    ) -> None:
        self._backend = backend
        self._closed = False
        try:
            if not isinstance(class_mapping, Mapping) or not class_mapping:
                raise ValueError(
                    "class_mapping must be a non-empty mapping; runtime "
                    "callers should use config.hailo.model_class_mapping()."
                )
            if any(
                isinstance(key, bool)
                or not isinstance(key, int)
                or key < 0
                or not isinstance(value, TargetClass)
                for key, value in class_mapping.items()
            ):
                raise ValueError(
                    "class_mapping must map non-negative integer model IDs "
                    "to TargetClass values; runtime callers should use "
                    "config.hailo.model_class_mapping()."
                )
            self._class_mapping = dict(class_mapping)
            self._detection_threshold = self._threshold(
                detection_threshold, "detection_threshold"
            )
            self._k0_threshold = self._threshold(
                k0_threshold, "k0_threshold"
            )
            if not isinstance(color_classifier, HsvColorClassifierConfig):
                raise ValueError(
                    "color_classifier must be an HsvColorClassifierConfig."
                )
            self._color_classifier = color_classifier
            converted_max_age_ms = float(max_observation_age_ms)
            if (
                not math.isfinite(converted_max_age_ms)
                or converted_max_age_ms <= 0.0
            ):
                raise ValueError(
                    "max_observation_age_ms must be positive and finite."
                )
            self._max_observation_age_ms = converted_max_age_ms
            self._ground_projector = ground_projector
        except BaseException:
            try:
                self.close()
            except BaseException:
                pass
            raise

    @staticmethod
    def _threshold(value: float, location: str) -> float:
        converted = float(value)
        if not 0.0 <= converted <= 1.0:
            raise ValueError(f"{location} must be in [0, 1].")
        return converted

    def detect(
        self,
        frame: CameraFrame,
        undistorted_image_bgr: np.ndarray,
        *,
        field_mask: FieldBoundaryMask | None = None,
        result_timestamp_ns: int | None = None,
    ) -> list[TargetObservation]:
        if (
            undistorted_image_bgr.ndim != 3
            or undistorted_image_bgr.shape[2] != 3
            or undistorted_image_bgr.dtype != np.uint8
        ):
            raise ValueError(
                "undistorted_image_bgr must be uint8 with shape "
                f"(height, width, 3), got dtype={undistorted_image_bgr.dtype}, "
                f"shape={undistorted_image_bgr.shape}."
            )
        image_size = (
            int(undistorted_image_bgr.shape[1]),
            int(undistorted_image_bgr.shape[0]),
        )
        frame_size = (int(frame.image_bgr.shape[1]), int(frame.image_bgr.shape[0]))
        if image_size != frame_size:
            raise ValueError(
                f"Undistorted image_size {image_size} does not match frame {frame_size}."
            )
        if (
            field_mask is not None
            and field_mask.image_state.shape
            != (image_size[1], image_size[0])
        ):
            raise ValueError(
                "field_mask image shape must match the undistorted image, got "
                f"{field_mask.image_state.shape} and {image_size}."
            )

        inference_image_bgr = (
            field_mask.mask_for_inference(
                undistorted_image_bgr,
                timestamp_ns=frame.timestamp_ns,
            )
            if field_mask is not None
            else undistorted_image_bgr
        )
        detections = self._backend.infer(inference_image_bgr)
        candidate_detections: list[ModelDetection] = []
        for detection in detections:
            if detection.confidence < self._detection_threshold:
                continue
            detection.box.validate_image_size(image_size)
            if detection.model_class_id not in self._class_mapping:
                raise ValueError(
                    f"Model class ID {detection.model_class_id} has no configured mapping."
                )
            candidate_detections.append(detection)

        image_hsv = (
            cv2.cvtColor(undistorted_image_bgr, cv2.COLOR_BGR2HSV)
            if candidate_detections
            else None
        )
        processed: list[_ProcessedDetection] = []
        for detection in candidate_detections:
            assert image_hsv is not None
            quality: set[ObservationQuality] = set()
            model_target_class = self._class_mapping[detection.model_class_id]
            color_segmentation, probabilities = segment_roi_colors(
                image_hsv,
                detection.box,
                self._color_classifier,
            )
            if color_segmentation.status is ColorSegmentationStatus.INSUFFICIENT:
                target_class = TargetClass.UNKNOWN
                quality.add(ObservationQuality.COLOR_EVIDENCE_INSUFFICIENT)
            elif color_segmentation.status is ColorSegmentationStatus.AMBIGUOUS:
                target_class = TargetClass.UNKNOWN
                quality.add(ObservationQuality.COLOR_EVIDENCE_AMBIGUOUS)
            else:
                target_class = color_segmentation.candidate_class
                if (
                    model_target_class is not TargetClass.UNKNOWN
                    and model_target_class is not target_class
                ):
                    quality.add(ObservationQuality.POSE_COLOR_CONFLICT)

            k0 = detection.k0
            if k0 is None or detection.k0_confidence < self._k0_threshold:
                k0 = None
                ground_point = None
                quality.add(ObservationQuality.K0_UNAVAILABLE)
            else:
                width, height = image_size
                if not (0.0 <= k0.u < width and 0.0 <= k0.v < height):
                    raise ValueError(
                        f"Model k0 {k0!r} is outside image_size {image_size!r}."
                    )
                ground_point = (
                    self._ground_projector.pixel_to_ground(k0)
                    if self._ground_projector is not None
                    else None
                )

            if (
                field_mask is not None
                and field_mask.filter_ready
                and field_mask.usable_at(frame.timestamp_ns)
            ):
                field_state = (
                    field_mask.state_at_pixel(k0.u, k0.v)
                    if k0 is not None
                    else FieldMaskState.UNCERTAIN
                )
                if field_state is FieldMaskState.UNCERTAIN:
                    quality.add(ObservationQuality.FIELD_BOUNDARY_UNCERTAIN)
                elif field_state is FieldMaskState.OUTSIDE:
                    conservative_classes = {
                        target_class,
                        model_target_class,
                    }
                    if conservative_classes.isdisjoint(
                        {TargetClass.BLUE_DANGER, TargetClass.UNKNOWN}
                    ):
                        continue
                    quality.add(ObservationQuality.OUTSIDE_FIELD_SUSPECTED)

            processed.append(
                _ProcessedDetection(
                    detection=detection,
                    model_target_class=model_target_class,
                    target_class=target_class,
                    class_probabilities=probabilities,
                    color_segmentation=color_segmentation,
                    k0=k0,
                    ground_point=ground_point,
                    quality=frozenset(quality),
                )
            )

        completed_timestamp_ns = (
            monotonic_ns() if result_timestamp_ns is None else result_timestamp_ns
        )
        if completed_timestamp_ns < frame.timestamp_ns:
            raise ValueError(
                "result_timestamp_ns must not be earlier than frame timestamp."
            )
        age_ms = (completed_timestamp_ns - frame.timestamp_ns) / 1_000_000.0
        if age_ms > self._max_observation_age_ms:
            raise StaleObservationError(
                age_ms,
                self._max_observation_age_ms,
            )

        return [
            TargetObservation(
                frame_sequence=frame.sequence,
                capture_timestamp_ns=frame.timestamp_ns,
                result_timestamp_ns=completed_timestamp_ns,
                image_size=image_size,
                model_target_class=item.model_target_class,
                target_class=item.target_class,
                class_probabilities=item.class_probabilities,
                detection_confidence=item.detection.confidence,
                box=item.detection.box,
                color_segmentation=item.color_segmentation,
                k0=item.k0,
                k0_confidence=item.detection.k0_confidence,
                ground_point=item.ground_point,
                quality=item.quality,
            )
            for item in processed
        ]

    def detect_realtime(
        self,
        frame: CameraFrame,
        undistorted_image_bgr: np.ndarray,
        *,
        field_mask: FieldBoundaryMask | None = None,
        result_timestamp_ns: int | None = None,
    ) -> RealtimeDetectionResult:
        """检测最新帧，并安全丢弃偶发的过期观测。

        只把过期异常转换为结构化丢弃结果；输入、模型、类别映射和硬件
        异常仍会原样抛出，避免掩盖真实故障。
        """

        try:
            observations = self.detect(
                frame,
                undistorted_image_bgr,
                field_mask=field_mask,
                result_timestamp_ns=result_timestamp_ns,
            )
        except StaleObservationError as exc:
            return RealtimeDetectionResult(
                observations=(),
                dropped_stale_age_ms=exc.age_ms,
            )
        return RealtimeDetectionResult(observations=tuple(observations))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend.close()

    def __enter__(self) -> TargetPoseDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
