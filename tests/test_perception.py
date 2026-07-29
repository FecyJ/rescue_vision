from __future__ import annotations

import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.evaluation.report import evaluate_records
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    ClassProbabilities,
    FakeInferenceBackend,
    ModelDetection,
    ObservationQuality,
    StaleObservationError,
    TargetClass,
    TargetObservation,
    TargetPoseDetector,
    UndistortedBoundingBox,
)
from rescue_vision.perception.evaluation_adapter import (
    TargetAnnotation,
    observations_to_evaluation_records,
)


def detection(
    class_id: int = 0,
    *,
    confidence: float = 0.8,
    k0: UndistortedPixel | None = UndistortedPixel(5.0, 6.0),
    k0_confidence: float = 0.9,
    box: UndistortedBoundingBox | None = None,
) -> ModelDetection:
    return ModelDetection(
        model_class_id=class_id,
        confidence=confidence,
        box=box or UndistortedBoundingBox(2.0, 3.0, 8.0, 9.0),
        k0=k0,
        k0_confidence=k0_confidence,
    )


def frame() -> CameraFrame:
    return CameraFrame(7, 1_000_000_000, np.zeros((12, 16, 3), dtype=np.uint8))


def detector(
    batches,
    *,
    projector: GroundProjector | None = None,
) -> TargetPoseDetector:
    return TargetPoseDetector(
        FakeInferenceBackend(batches),
        class_mapping={
            0: TargetClass.GREEN_SUPPLY,
            1: TargetClass.BLUE_DANGER,
        },
        detection_threshold=0.25,
        semantic_threshold=0.5,
        k0_threshold=0.5,
        max_observation_age_ms=150.0,
        ground_projector=projector,
    )


def test_target_classes_follow_pose_convention() -> None:
    assert [item.value for item in TargetClass] == [
        "green_supply",
        "black_core",
        "orange_injured",
        "blue_danger",
        "unknown",
    ]
    with pytest.raises(ValueError):
        TargetClass("hazard")


def test_detector_closes_backend_when_constructor_validation_fails() -> None:
    backend = FakeInferenceBackend([])
    with pytest.raises(ValueError, match="model_class_mapping"):
        TargetPoseDetector(
            backend,
            class_mapping=(TargetClass.GREEN_SUPPLY,),  # type: ignore[arg-type]
            detection_threshold=0.25,
            semantic_threshold=0.5,
            k0_threshold=0.5,
            max_observation_age_ms=150.0,
        )

    assert backend.closed


def test_detector_builds_observation_and_projects_k0() -> None:
    projector = GroundProjector(np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 1]]))
    observations = detector([[detection()]], projector=projector).detect(
        frame(),
        np.zeros((12, 16, 3), dtype=np.uint8),
        result_timestamp_ns=1_020_000_000,
    )
    assert len(observations) == 1
    observation = observations[0]
    assert observation.target_class is TargetClass.GREEN_SUPPLY
    assert observation.class_probabilities.green_supply == pytest.approx(0.8)
    assert observation.class_probabilities.unknown == pytest.approx(0.2)
    assert observation.detection_confidence == pytest.approx(0.8)
    assert observation.ground_point == GroundPoint(10.0, 18.0)
    assert observation.quality == frozenset()


def test_low_class_and_k0_confidence_degrade_conservatively() -> None:
    observation = detector(
        [[detection(confidence=0.4, k0_confidence=0.2)]]
    ).detect(
        frame(),
        np.zeros((12, 16, 3), dtype=np.uint8),
        result_timestamp_ns=1_010_000_000,
    )[0]
    assert observation.target_class is TargetClass.UNKNOWN
    assert observation.class_probabilities.green_supply == pytest.approx(0.4)
    assert observation.class_probabilities.unknown == pytest.approx(0.6)
    assert observation.detection_confidence == pytest.approx(0.4)
    assert observation.k0 is None
    assert observation.ground_point is None
    assert observation.quality == frozenset(
        {
            ObservationQuality.LOW_CLASS_CONFIDENCE,
            ObservationQuality.K0_UNAVAILABLE,
        }
    )


def test_detector_filters_low_detection_and_supports_empty() -> None:
    result = detector([[detection(confidence=0.1)], []]).detect(
        frame(),
        np.zeros((12, 16, 3), dtype=np.uint8),
        result_timestamp_ns=1_010_000_000,
    )
    assert result == []


def test_detector_rejects_stale_or_unmapped_results() -> None:
    with pytest.raises(StaleObservationError, match="exceeds") as raised:
        detector([[detection()]]).detect(
            frame(),
            np.zeros((12, 16, 3), dtype=np.uint8),
            result_timestamp_ns=1_151_000_000,
        )
    assert raised.value.age_ms == pytest.approx(151.0)
    assert raised.value.max_age_ms == pytest.approx(150.0)
    with pytest.raises(ValueError, match="no configured mapping"):
        detector([[detection(class_id=4)]]).detect(
            frame(),
            np.zeros((12, 16, 3), dtype=np.uint8),
            result_timestamp_ns=1_010_000_000,
        )


def test_realtime_detector_drops_only_stale_observations() -> None:
    result = detector([[detection()]]).detect_realtime(
        frame(),
        np.zeros((12, 16, 3), dtype=np.uint8),
        result_timestamp_ns=1_151_000_000,
    )

    assert result.observations == ()
    assert result.stale_dropped
    assert result.dropped_stale_age_ms == pytest.approx(151.0)


def test_realtime_detector_returns_current_observations() -> None:
    result = detector([[detection()]]).detect_realtime(
        frame(),
        np.zeros((12, 16, 3), dtype=np.uint8),
        result_timestamp_ns=1_010_000_000,
    )

    assert len(result.observations) == 1
    assert not result.stale_dropped
    assert result.dropped_stale_age_ms is None


def test_realtime_detector_preserves_non_stale_errors() -> None:
    with pytest.raises(ValueError, match="no configured mapping"):
        detector([[detection(class_id=4)]]).detect_realtime(
            frame(),
            np.zeros((12, 16, 3), dtype=np.uint8),
            result_timestamp_ns=1_010_000_000,
        )


def test_invalid_observation_coordinates_fail() -> None:
    with pytest.raises(ValueError, match="outside"):
        TargetObservation(
            frame_sequence=0,
            capture_timestamp_ns=0,
            result_timestamp_ns=1,
            image_size=(10, 10),
            target_class=TargetClass.GREEN_SUPPLY,
            class_probabilities=ClassProbabilities.from_top_class(
                TargetClass.GREEN_SUPPLY, 0.8
            ),
            detection_confidence=0.8,
            box=UndistortedBoundingBox(0, 0, 5, 5),
            k0=UndistortedPixel(11, 1),
            k0_confidence=0.9,
            ground_point=None,
            quality=frozenset(),
        )


def test_observations_to_evaluation_records_full_chain() -> None:
    observations = detector(
        [
            [
                detection(
                    class_id=1,
                    box=UndistortedBoundingBox(0, 0, 4, 4),
                ),
                detection(
                    class_id=0,
                    box=UndistortedBoundingBox(10, 0, 14, 4),
                ),
            ]
        ]
    ).detect(
        frame(),
        np.zeros((12, 16, 3), dtype=np.uint8),
        result_timestamp_ns=1_020_000_000,
    )
    annotations = [
        TargetAnnotation(
            "truth_1",
            TargetClass.GREEN_SUPPLY,
            UndistortedBoundingBox(0, 0, 4, 4),
        ),
        TargetAnnotation(
            "truth_2",
            TargetClass.ORANGE_INJURED,
            UndistortedBoundingBox(5, 5, 9, 9),
        ),
    ]
    records = observations_to_evaluation_records(
        sample_id="sample",
        annotations=annotations,
        observations=observations,
        capture_timestamp_ns=1_000_000_000,
        result_timestamp_ns=1_020_000_000,
    )
    report = evaluate_records(
        records,
    )
    assert report["failure_count"] == 3
    assert report["per_class"]["blue_danger"]["false_positive"] == 1
    assert report["per_class"]["orange_injured"]["false_negative"] == 1
    matched = next(record for record in records if record["object_id"] == "truth_1")
    assert matched["confidence"] == pytest.approx(0.8)
    assert matched["quality"] == []
