"""把模型检测转换为带时间、语义和地面坐标的统一观测。"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math
from collections.abc import Callable
from time import monotonic_ns

import cv2
import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception.backend import InferenceBackend
from rescue_vision.perception.color_segmentation import segment_bgr_roi_colors
from rescue_vision.perception.field_feature_types import (
    CenterCrossConfirmation,
    CenterCrossObservation,
    FieldFeatureDetectionResult,
    FieldFeatureQuality,
    FieldPoseKeypoint,
    LineSegmentObservation,
    SafeZoneColor,
    SafeZoneObservation,
)
from rescue_vision.perception.types import (
    COLOR_TARGET_CLASSES,
    ClassProbabilities,
    ColorSegmentationStatus,
    HsvColorClassifierConfig,
    ModelDetection,
    ObservationQuality,
    POSE_MODEL_CLASSES,
    PoseModelClass,
    RoiColorSegmentation,
    TargetClass,
    TargetObservation,
)
from rescue_vision.perception.timing import PerceptionTiming


# 当前相机/模型地面投影实测的统一前向偏差；目标和场地关键点共用同一修正。
MODEL_GROUND_FORWARD_BIAS_MM = 225.0


def _correct_model_ground_point(point: GroundPoint) -> GroundPoint:
    return GroundPoint(
        point.x + MODEL_GROUND_FORWARD_BIAS_MM,
        point.y,
    )


class StaleObservationError(ValueError):
    """推理完成时，输入帧已经超过允许的观测年龄。"""

    def __init__(
        self,
        age_ms: float,
        max_age_ms: float,
        *,
        timing: PerceptionTiming | None = None,
    ) -> None:
        self.age_ms = float(age_ms)
        self.max_age_ms = float(max_age_ms)
        self.timing = timing
        super().__init__(
            f"Frame observation age {self.age_ms:.3f} ms exceeds "
            f"{self.max_age_ms:.3f} ms."
        )


@dataclass(frozen=True, slots=True)
class RealtimeDetectionResult:
    """实时检测结果；过期观测只记录丢弃原因，不向下游泄漏。"""

    frame_sequence: int
    observations: tuple[TargetObservation, ...]
    field_features: FieldFeatureDetectionResult | None
    timing: PerceptionTiming
    dropped_stale_age_ms: float | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be a non-negative integer.")
        if not isinstance(self.timing, PerceptionTiming):
            raise TypeError("timing must be a PerceptionTiming value.")
        if self.dropped_stale_age_ms is not None and (
            not math.isfinite(float(self.dropped_stale_age_ms))
            or self.dropped_stale_age_ms < 0.0
        ):
            raise ValueError("dropped_stale_age_ms must be finite and non-negative.")

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


@dataclass(frozen=True, slots=True)
class PoseDetectionResult:
    """一次六类模型推理产生的同帧任务目标与场地地标。"""

    observations: tuple[TargetObservation, ...]
    field_features: FieldFeatureDetectionResult
    timing: PerceptionTiming

    def __iter__(self):
        return iter(self.observations)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int) -> TargetObservation:
        return self.observations[index]


@dataclass(frozen=True, slots=True)
class CenterCrossRefinementConfig:
    enabled: bool = True
    canny_low: int = 50
    canny_high: int = 150
    hough_threshold: int = 16
    min_line_length_px: float = 18.0
    max_line_gap_px: float = 12.0
    max_intersection_distance_px: float = 18.0
    min_axis_angle_deg: float = 65.0

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        if not 0 <= self.canny_low < self.canny_high <= 255:
            raise ValueError("Canny thresholds must satisfy 0 <= low < high <= 255.")
        if self.hough_threshold <= 0:
            raise ValueError("hough_threshold must be positive.")
        for name in (
            "min_line_length_px", "max_intersection_distance_px", "min_axis_angle_deg"
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if not math.isfinite(self.max_line_gap_px) or self.max_line_gap_px < 0.0:
            raise ValueError("max_line_gap_px must be non-negative and finite.")
        if self.min_axis_angle_deg > 90.0:
            raise ValueError("min_axis_angle_deg must be <= 90.")


@dataclass(frozen=True, slots=True)
class SafeZoneColorConfig:
    """安全区底色证据；阈值未标定时保持禁用并输出 unknown。"""

    enabled: bool = False
    red_hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = ()
    blue_hsv_ranges: tuple[tuple[tuple[int, int, int], tuple[int, int, int]], ...] = ()
    min_fraction: float = 0.08
    min_margin: float = 0.03

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        if not 0.0 <= float(self.min_fraction) <= 1.0:
            raise ValueError("min_fraction must be in [0, 1].")
        if not 0.0 <= float(self.min_margin) <= 1.0:
            raise ValueError("min_margin must be in [0, 1].")
        if self.enabled and (not self.red_hsv_ranges or not self.blue_hsv_ranges):
            raise ValueError("enabled safe-zone color classification requires both ranges.")


class TargetPoseDetector:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        detection_threshold: float,
        k0_threshold: float,
        color_classifier: HsvColorClassifierConfig,
        max_observation_age_ms: float,
        ground_projector: GroundProjector | None = None,
        center_cross_refinement: CenterCrossRefinementConfig | None = None,
        safe_zone_color: SafeZoneColorConfig | None = None,
        clock_ns: Callable[[], int] = monotonic_ns,
    ) -> None:
        self._backend = backend
        self._closed = False
        try:
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
            self._center_cross_refinement = (
                center_cross_refinement or CenterCrossRefinementConfig()
            )
            self._safe_zone_color = safe_zone_color or SafeZoneColorConfig()
            if not callable(clock_ns):
                raise TypeError("clock_ns must be callable.")
            self._clock_ns = clock_ns
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
        result_timestamp_ns: int | None = None,
    ) -> PoseDetectionResult:
        processing_started_ns = max(frame.timestamp_ns, self._clock_ns())
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

        inference_image_bgr = undistorted_image_bgr
        detections = self._backend.infer(inference_image_bgr)
        inference_completed_ns = max(frame.timestamp_ns, self._clock_ns())
        candidate_detections: list[ModelDetection] = []
        for detection in detections:
            if detection.confidence < self._detection_threshold:
                continue
            detection.box.validate_image_size(image_size)
            width, height = image_size
            for index, keypoint in enumerate(detection.keypoints):
                point = keypoint.point
                if point is not None and not (
                    0.0 <= point.u < width and 0.0 <= point.v < height
                ):
                    raise ValueError(
                        f"Model K{index} {point!r} is outside image_size {image_size!r}."
                    )
            candidate_detections.append(detection)

        processed: list[_ProcessedDetection] = []
        for detection in candidate_detections:
            if detection.model_class not in POSE_MODEL_CLASSES[:4]:
                continue
            quality: set[ObservationQuality] = set()
            model_target_class = COLOR_TARGET_CLASSES[detection.model_class_id]
            color_segmentation, probabilities = segment_bgr_roi_colors(
                undistorted_image_bgr,
                detection.box,
                self._color_classifier,
                target_class=model_target_class,
            )
            if color_segmentation.status is ColorSegmentationStatus.INSUFFICIENT:
                quality.add(ObservationQuality.COLOR_EVIDENCE_INSUFFICIENT)
            elif color_segmentation.status is ColorSegmentationStatus.AMBIGUOUS:
                quality.add(ObservationQuality.COLOR_EVIDENCE_AMBIGUOUS)
            # Model labels are authoritative. HSV supplies jaw geometry only.
            target_class = model_target_class
            probabilities = ClassProbabilities.from_top_class(target_class, detection.confidence)

            k0_keypoint = detection.keypoints[0]
            k0 = k0_keypoint.point
            if k0 is None or k0_keypoint.confidence < self._k0_threshold:
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
                if ground_point is not None:
                    ground_point = _correct_model_ground_point(ground_point)

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

        # Field-feature refinement and safe-zone colour classification are part
        # of the observation latency, so finish them before taking the result
        # timestamp.
        provisional_timestamp_ns = max(frame.timestamp_ns, self._clock_ns())
        field_features = self._build_field_features(
            frame,
            undistorted_image_bgr,
            candidate_detections,
            provisional_timestamp_ns,
        )
        completed_timestamp_ns = (
            self._clock_ns() if result_timestamp_ns is None else result_timestamp_ns
        )
        if completed_timestamp_ns < frame.timestamp_ns:
            raise ValueError(
                "result_timestamp_ns must not be earlier than frame timestamp."
            )
        age_ms = (completed_timestamp_ns - frame.timestamp_ns) / 1_000_000.0
        if age_ms > self._max_observation_age_ms:
            timing = self._timing(
                frame,
                processing_started_ns,
                inference_completed_ns,
                completed_timestamp_ns,
                result_timestamp_ns,
            )
            raise StaleObservationError(
                age_ms,
                self._max_observation_age_ms,
                timing=timing,
            )

        timing = self._timing(
            frame,
            processing_started_ns,
            inference_completed_ns,
            completed_timestamp_ns,
            result_timestamp_ns,
        )
        if field_features.result_timestamp_ns != completed_timestamp_ns:
            field_features = replace(
                field_features,
                result_timestamp_ns=completed_timestamp_ns,
            )

        observations = tuple(
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
                k0_confidence=item.detection.keypoints[0].confidence,
                ground_point=item.ground_point,
                quality=item.quality,
            )
            for item in processed
        )
        return PoseDetectionResult(observations, field_features, timing)

    def _timing(
        self,
        frame: CameraFrame,
        processing_started_ns: int,
        inference_completed_ns: int,
        completed_timestamp_ns: int,
        override_timestamp_ns: int | None,
    ) -> PerceptionTiming:
        # Existing offline fixtures may provide a synthetic completion time;
        # keep their contract valid while production uses the real clock.
        if override_timestamp_ns is not None:
            processing_started_ns = frame.timestamp_ns
            inference_completed_ns = frame.timestamp_ns
        return PerceptionTiming(
            capture_timestamp_ns=frame.timestamp_ns,
            submitted_timestamp_ns=None,
            processing_started_timestamp_ns=processing_started_ns,
            inference_completed_timestamp_ns=inference_completed_ns,
            result_timestamp_ns=completed_timestamp_ns,
        )

    def _field_keypoint(self, detection: ModelDetection, index: int) -> FieldPoseKeypoint:
        keypoint = detection.keypoints[index]
        if keypoint.point is None or keypoint.confidence < self._k0_threshold:
            return FieldPoseKeypoint(None, None, 0.0)
        ground = (
            self._ground_projector.pixel_to_ground(keypoint.point)
            if self._ground_projector is not None
            else None
        )
        if ground is not None:
            ground = _correct_model_ground_point(ground)
        return FieldPoseKeypoint(keypoint.point, ground, keypoint.confidence)

    @staticmethod
    def _point_line_distance(point: UndistortedPixel, line: tuple[int, int, int, int]) -> float:
        x1, y1, x2, y2 = line
        denominator = math.hypot(x2 - x1, y2 - y1)
        if denominator <= 1e-9:
            return float("inf")
        return abs((y2 - y1) * point.u - (x2 - x1) * point.v + x2 * y1 - y2 * x1) / denominator

    def _refine_center_axes(
        self,
        image_bgr: np.ndarray,
        detection: ModelDetection,
        intersection: FieldPoseKeypoint,
    ) -> tuple[tuple[LineSegmentObservation, ...], tuple[float, ...], float | None]:
        config = self._center_cross_refinement
        point = intersection.undistorted
        if not config.enabled or point is None:
            return (), (), None
        box = detection.box
        x0, y0 = math.floor(box.x_min), math.floor(box.y_min)
        x1, y1 = math.ceil(box.x_max), math.ceil(box.y_max)
        roi = image_bgr[y0:y1, x0:x1]
        if roi.size == 0:
            return (), (), None
        edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), config.canny_low, config.canny_high)
        raw = cv2.HoughLinesP(
            edges,
            1.0,
            np.pi / 180.0,
            config.hough_threshold,
            minLineLength=config.min_line_length_px,
            maxLineGap=config.max_line_gap_px,
        )
        if raw is None:
            return (), (), None
        candidates: list[tuple[float, tuple[int, int, int, int]]] = []
        for values in raw[:, 0, :]:
            line = (int(values[0] + x0), int(values[1] + y0), int(values[2] + x0), int(values[3] + y0))
            if self._point_line_distance(point, line) <= config.max_intersection_distance_px:
                angle = math.atan2(line[3] - line[1], line[2] - line[0]) % math.pi
                candidates.append((angle, line))
        if not candidates:
            return (), (), None
        first = max(candidates, key=lambda item: math.hypot(item[1][2] - item[1][0], item[1][3] - item[1][1]))
        second_candidates = [
            item for item in candidates
            if math.degrees(min(abs(item[0] - first[0]), math.pi - abs(item[0] - first[0]))) >= config.min_axis_angle_deg
        ]
        selected = [first]
        if second_candidates:
            selected.append(max(second_candidates, key=lambda item: math.hypot(item[1][2] - item[1][0], item[1][3] - item[1][1])))
        axes: list[LineSegmentObservation] = []
        residuals: list[float] = []
        for _angle, line in selected:
            start = UndistortedPixel(float(line[0]), float(line[1]))
            end = UndistortedPixel(float(line[2]), float(line[3]))
            start_ground = end_ground = None
            if self._ground_projector is not None:
                start_ground = _correct_model_ground_point(
                    self._ground_projector.pixel_to_ground(start)
                )
                end_ground = _correct_model_ground_point(
                    self._ground_projector.pixel_to_ground(end)
                )
            axes.append(LineSegmentObservation(start, end, start_ground, end_ground))
            residuals.append(self._point_line_distance(point, line))
        angle_deg = None
        if len(selected) == 2:
            raw_angle = abs(selected[0][0] - selected[1][0]) % math.pi
            angle_deg = math.degrees(min(raw_angle, math.pi - raw_angle))
        return tuple(axes), tuple(residuals), angle_deg

    def _safe_zone_identity(self, image_bgr: np.ndarray, box) -> SafeZoneColor:
        config = self._safe_zone_color
        if not config.enabled:
            return SafeZoneColor.UNKNOWN
        x0, y0 = math.floor(box.x_min), math.floor(box.y_min)
        x1, y1 = math.ceil(box.x_max), math.ceil(box.y_max)
        hsv = cv2.cvtColor(image_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        if hsv.size == 0:
            return SafeZoneColor.UNKNOWN
        fractions: dict[SafeZoneColor, float] = {}
        for color, ranges in ((SafeZoneColor.RED, config.red_hsv_ranges), (SafeZoneColor.BLUE, config.blue_hsv_ranges)):
            mask = np.zeros(hsv.shape[:2], np.uint8)
            for lower, upper in ranges:
                mask |= cv2.inRange(hsv, np.asarray(lower, np.uint8), np.asarray(upper, np.uint8))
            fractions[color] = float(np.count_nonzero(mask)) / mask.size
        ranked = sorted(fractions.items(), key=lambda item: item[1], reverse=True)
        winner, winner_fraction = ranked[0]
        if (
            winner_fraction < config.min_fraction
            or winner_fraction - ranked[1][1] < config.min_margin
        ):
            return SafeZoneColor.UNKNOWN
        return winner

    def _build_field_features(
        self,
        frame: CameraFrame,
        image_bgr: np.ndarray,
        detections: list[ModelDetection],
        result_timestamp_ns: int,
    ) -> FieldFeatureDetectionResult:
        cross: CenterCrossObservation | None = None
        zones: list[SafeZoneObservation] = []
        for detection in detections:
            if detection.model_class is PoseModelClass.CENTER_CROSS:
                intersection = self._field_keypoint(detection, 0)
                axes, residuals, angle = self._refine_center_axes(image_bgr, detection, intersection)
                quality = set()
                if intersection.undistorted is None:
                    quality.add(FieldFeatureQuality.KEYPOINT_UNAVAILABLE)
                if len(axes) != 2:
                    quality.add(FieldFeatureQuality.AXIS_REFINEMENT_UNAVAILABLE)
                candidate = CenterCrossObservation(
                    detection.box,
                    intersection,
                    axes,
                    detection.confidence,
                    frozenset(quality),
                    CenterCrossConfirmation.CANDIDATE,
                    residuals,
                    angle,
                )
                if cross is None or candidate.confidence > cross.confidence:
                    cross = candidate
            elif detection.model_class is PoseModelClass.SAFE_ZONE:
                identity = self._safe_zone_identity(image_bgr, detection.box)
                quality = set()
                keypoints = tuple(self._field_keypoint(detection, index) for index in range(3))
                if any(item.undistorted is None for item in keypoints):
                    quality.add(FieldFeatureQuality.KEYPOINT_UNAVAILABLE)
                if identity is SafeZoneColor.UNKNOWN:
                    quality.add(FieldFeatureQuality.IDENTITY_UNRESOLVED)
                zones.append(SafeZoneObservation(
                    detection.box,
                    keypoints[0],
                    keypoints[1],
                    keypoints[2],
                    identity,
                    detection.confidence,
                    frozenset(quality),
                ))
        return FieldFeatureDetectionResult(
            frame.sequence,
            frame.timestamp_ns,
            result_timestamp_ns,
            (image_bgr.shape[1], image_bgr.shape[0]),
            tuple(zones),
            cross,
        )

    def detect_realtime(
        self,
        frame: CameraFrame,
        undistorted_image_bgr: np.ndarray,
        *,
        result_timestamp_ns: int | None = None,
    ) -> RealtimeDetectionResult:
        """检测最新帧，并安全丢弃偶发的过期观测。

        只把过期异常转换为结构化丢弃结果；输入、模型、类别映射和硬件
        异常仍会原样抛出，避免掩盖真实故障。
        """

        try:
            detection_result = self.detect(
                frame,
                undistorted_image_bgr,
                result_timestamp_ns=result_timestamp_ns,
            )
        except StaleObservationError as exc:
            if exc.timing is None:
                raise RuntimeError("stale observation did not include timing") from exc
            return RealtimeDetectionResult(
                frame_sequence=frame.sequence,
                observations=(),
                field_features=None,
                timing=exc.timing,
                dropped_stale_age_ms=exc.age_ms,
            )
        return RealtimeDetectionResult(
            frame_sequence=frame.sequence,
            observations=detection_result.observations,
            field_features=detection_result.field_features,
            timing=detection_result.timing,
        )

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._backend.close()

    def __enter__(self) -> TargetPoseDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
