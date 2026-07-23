"""把模型检测转换为带时间、语义和地面坐标的统一观测。"""

from __future__ import annotations

from collections.abc import Mapping
from time import monotonic_ns

import numpy as np

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.perception.backend import InferenceBackend
from rescue_vision.perception.types import (
    ClassProbabilities,
    ObservationQuality,
    TargetClass,
    TargetObservation,
)


class TargetPoseDetector:
    def __init__(
        self,
        backend: InferenceBackend,
        *,
        class_mapping: Mapping[int, TargetClass],
        detection_threshold: float,
        semantic_threshold: float,
        k0_threshold: float,
        max_observation_age_ms: float,
        ground_projector: GroundProjector | None = None,
    ) -> None:
        if not class_mapping:
            raise ValueError("class_mapping must not be empty.")
        if any(
            isinstance(key, bool) or not isinstance(key, int) or key < 0
            for key in class_mapping
        ):
            raise ValueError("class_mapping keys must be non-negative integer IDs.")
        self._backend = backend
        self._class_mapping = dict(class_mapping)
        self._detection_threshold = self._threshold(
            detection_threshold, "detection_threshold"
        )
        self._semantic_threshold = self._threshold(
            semantic_threshold, "semantic_threshold"
        )
        self._k0_threshold = self._threshold(k0_threshold, "k0_threshold")
        if self._semantic_threshold < self._detection_threshold:
            raise ValueError(
                "semantic_threshold must be >= detection_threshold."
            )
        if max_observation_age_ms <= 0.0:
            raise ValueError("max_observation_age_ms must be positive.")
        self._max_observation_age_ms = float(max_observation_age_ms)
        self._ground_projector = ground_projector

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
    ) -> list[TargetObservation]:
        if (
            undistorted_image_bgr.ndim != 3
            or undistorted_image_bgr.shape[2] != 3
        ):
            raise ValueError(
                "undistorted_image_bgr must have shape (height, width, 3), got "
                f"{undistorted_image_bgr.shape}."
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

        detections = self._backend.infer(undistorted_image_bgr)
        result_timestamp_ns = (
            monotonic_ns() if result_timestamp_ns is None else result_timestamp_ns
        )
        if result_timestamp_ns < frame.timestamp_ns:
            raise ValueError(
                "result_timestamp_ns must not be earlier than frame timestamp."
            )
        age_ms = (result_timestamp_ns - frame.timestamp_ns) / 1_000_000.0
        if age_ms > self._max_observation_age_ms:
            raise ValueError(
                f"Frame observation age {age_ms:.3f} ms exceeds "
                f"{self._max_observation_age_ms:.3f} ms."
            )

        observations: list[TargetObservation] = []
        for detection in detections:
            if detection.confidence < self._detection_threshold:
                continue
            detection.box.validate_image_size(image_size)
            if detection.model_class_id not in self._class_mapping:
                raise ValueError(
                    f"Model class ID {detection.model_class_id} has no configured mapping."
                )

            quality: set[ObservationQuality] = set()
            mapped_class = self._class_mapping[detection.model_class_id]
            if detection.confidence < self._semantic_threshold:
                target_class = TargetClass.UNKNOWN
                probabilities = ClassProbabilities.from_top_class(
                    TargetClass.UNKNOWN, 1.0
                )
                quality.add(ObservationQuality.LOW_CLASS_CONFIDENCE)
            else:
                target_class = mapped_class
                probabilities = ClassProbabilities.from_top_class(
                    mapped_class, detection.confidence
                )

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

            observations.append(
                TargetObservation(
                    frame_sequence=frame.sequence,
                    capture_timestamp_ns=frame.timestamp_ns,
                    result_timestamp_ns=result_timestamp_ns,
                    image_size=image_size,
                    target_class=target_class,
                    class_probabilities=probabilities,
                    box=detection.box,
                    k0=k0,
                    k0_confidence=detection.k0_confidence,
                    ground_point=ground_point,
                    quality=frozenset(quality),
                    model_version=self._backend.model_version,
                    model_sha256=self._backend.model_sha256,
                )
            )
        return observations

    def close(self) -> None:
        self._backend.close()

    def __enter__(self) -> TargetPoseDetector:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()
