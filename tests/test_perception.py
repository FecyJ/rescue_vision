from __future__ import annotations

from dataclasses import replace
import time

import cv2
import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.evaluation.report import evaluate_records
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    ClassProbabilities,
    ColorSegmentationStatus,
    FakeInferenceBackend,
    HsvColorClassifierConfig,
    HsvRange,
    ModelDetection,
    ObservationQuality,
    PerceptionFrameRenderer,
    RoiColorSegmentation,
    StaleObservationError,
    TargetClass,
    TargetObservation,
    TargetPoseDetector,
    UndistortedBoundingBox,
    render_target_observations,
)
from rescue_vision.perception.evaluation_adapter import (
    TargetAnnotation,
    observations_to_evaluation_records,
)


def color_config(**overrides: object) -> HsvColorClassifierConfig:
    config = HsvColorClassifierConfig(
        green_supply=(HsvRange((35, 70, 71), (84, 255, 255)),),
        black_core=(HsvRange((0, 0, 0), (179, 255, 70)),),
        orange_injured=(
            HsvRange((0, 90, 80), (20, 255, 255)),
            HsvRange((170, 90, 80), (179, 255, 255)),
        ),
        blue_danger=(HsvRange((85, 50, 71), (110, 255, 255)),),
        min_color_fraction=0.15,
        min_color_dominance=0.70,
        min_dominance_margin=0.20,
        morphology_kernel_size=3,
        open_iterations=1,
        close_iterations=1,
        min_component_area_fraction=0.002,
    )
    return replace(config, **overrides)


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


def image_with_regions(
    *regions: tuple[UndistortedBoundingBox, tuple[int, int, int]],
) -> np.ndarray:
    hsv = np.full((12, 16, 3), (0, 0, 114), dtype=np.uint8)
    for box, color in regions:
        hsv[
            int(box.y_min) : int(box.y_max),
            int(box.x_min) : int(box.x_max),
        ] = color
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def frame(image: np.ndarray) -> CameraFrame:
    return CameraFrame(7, 1_000_000_000, image)


def detector(
    batches,
    *,
    projector: GroundProjector | None = None,
    classifier: HsvColorClassifierConfig | None = None,
) -> TargetPoseDetector:
    return TargetPoseDetector(
        FakeInferenceBackend(batches),
        class_mapping={
            0: TargetClass.GREEN_SUPPLY,
            1: TargetClass.BLUE_DANGER,
        },
        detection_threshold=0.25,
        k0_threshold=0.5,
        color_classifier=classifier or color_config(),
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
            k0_threshold=0.5,
            color_classifier=color_config(),
            max_observation_age_ms=150.0,
        )

    assert backend.closed


def test_detector_uses_hsv_class_and_projects_k0() -> None:
    projector = GroundProjector(np.array([[2.0, 0, 0], [0, 3.0, 0], [0, 0, 1]]))
    box = UndistortedBoundingBox(2.0, 3.0, 8.0, 9.0)
    image = image_with_regions((box, (60, 255, 200)))
    observations = detector([[detection()]], projector=projector).detect(
        frame(image),
        image,
        result_timestamp_ns=1_020_000_000,
    )

    assert len(observations) == 1
    observation = observations[0]
    assert observation.model_target_class is TargetClass.GREEN_SUPPLY
    assert observation.target_class is TargetClass.GREEN_SUPPLY
    assert observation.class_probabilities.green_supply == pytest.approx(1.0)
    assert observation.class_probabilities.unknown == 0.0
    assert observation.detection_confidence == pytest.approx(0.8)
    assert observation.ground_point == GroundPoint(10.0, 18.0)
    assert observation.quality == frozenset()
    segmentation = observation.color_segmentation
    assert segmentation.status is ColorSegmentationStatus.ACCEPTED
    assert segmentation.roi_box == box
    assert segmentation.mask.shape == (6, 6)
    assert np.all(segmentation.mask == 255)
    assert not segmentation.mask.flags.writeable
    with pytest.raises(ValueError):
        segmentation.mask[0, 0] = 0


@pytest.mark.parametrize(
    ("hsv", "expected"),
    [
        ((60, 255, 200), TargetClass.GREEN_SUPPLY),
        ((0, 0, 30), TargetClass.BLACK_CORE),
        ((10, 255, 200), TargetClass.ORANGE_INJURED),
        ((175, 255, 200), TargetClass.ORANGE_INJURED),
        ((95, 150, 200), TargetClass.BLUE_DANGER),
    ],
)
def test_initial_hsv_ranges_classify_official_target_colors(
    hsv: tuple[int, int, int],
    expected: TargetClass,
) -> None:
    box = UndistortedBoundingBox(2.0, 3.0, 8.0, 9.0)
    image = image_with_regions((box, hsv))
    observation = detector([[detection()]]).detect(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )[0]

    assert observation.target_class is expected
    assert observation.color_segmentation.candidate_class is expected


def test_hsv_overrides_pose_class_and_records_conflict() -> None:
    box = UndistortedBoundingBox(2.0, 3.0, 8.0, 9.0)
    blue_image = image_with_regions((box, (95, 150, 200)))
    blue = detector([[detection(class_id=0)]]).detect(
        frame(blue_image),
        blue_image,
        result_timestamp_ns=1_010_000_000,
    )[0]
    assert blue.model_target_class is TargetClass.GREEN_SUPPLY
    assert blue.target_class is TargetClass.BLUE_DANGER
    assert blue.quality == frozenset({ObservationQuality.POSE_COLOR_CONFLICT})

    green_image = image_with_regions((box, (60, 255, 200)))
    green = detector([[detection(class_id=1)]]).detect(
        frame(green_image),
        green_image,
        result_timestamp_ns=1_010_000_000,
    )[0]
    assert green.model_target_class is TargetClass.BLUE_DANGER
    assert green.target_class is TargetClass.GREEN_SUPPLY
    assert ObservationQuality.POSE_COLOR_CONFLICT in green.quality


def test_insufficient_color_and_k0_degrade_conservatively() -> None:
    image = image_with_regions()
    observation = detector(
        [[detection(k0_confidence=0.2)]]
    ).detect(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )[0]

    assert observation.target_class is TargetClass.UNKNOWN
    assert observation.class_probabilities.unknown == 1.0
    assert observation.color_segmentation.candidate_class is TargetClass.UNKNOWN
    assert observation.color_segmentation.status is ColorSegmentationStatus.INSUFFICIENT
    assert observation.k0 is None
    assert observation.ground_point is None
    assert observation.quality == frozenset(
        {
            ObservationQuality.COLOR_EVIDENCE_INSUFFICIENT,
            ObservationQuality.K0_UNAVAILABLE,
        }
    )


def test_low_coverage_keeps_candidate_roi_mask_for_diagnostics() -> None:
    box = UndistortedBoundingBox(2.2, 3.2, 8.0, 9.0)
    hsv = np.full((12, 16, 3), (0, 0, 114), dtype=np.uint8)
    hsv[3:6, 2:5] = (60, 255, 200)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    observation = detector(
        [[detection(box=box)]],
        classifier=color_config(
            min_color_fraction=0.5,
            open_iterations=0,
            close_iterations=0,
        ),
    ).detect(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )[0]

    segmentation = observation.color_segmentation
    assert observation.target_class is TargetClass.UNKNOWN
    assert segmentation.candidate_class is TargetClass.GREEN_SUPPLY
    assert segmentation.status is ColorSegmentationStatus.INSUFFICIENT
    assert segmentation.roi_box == UndistortedBoundingBox(2.0, 3.0, 8.0, 9.0)
    assert segmentation.mask.shape == (6, 6)
    assert np.count_nonzero(segmentation.mask) == 9


def test_ambiguous_two_color_roi_degrades_to_unknown() -> None:
    box = UndistortedBoundingBox(0.0, 0.0, 16.0, 12.0)
    hsv = np.full((12, 16, 3), (95, 150, 200), dtype=np.uint8)
    hsv[:, :8] = (60, 255, 200)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    observation = detector(
        [[detection(box=box)]],
        classifier=color_config(open_iterations=0, close_iterations=0),
    ).detect(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )[0]

    assert observation.target_class is TargetClass.UNKNOWN
    assert observation.class_probabilities.unknown == 1.0
    assert observation.color_segmentation.status is ColorSegmentationStatus.AMBIGUOUS
    assert observation.color_segmentation.dominance == pytest.approx(0.5)
    assert ObservationQuality.COLOR_EVIDENCE_AMBIGUOUS in observation.quality


def test_morphology_removes_isolated_color_noise() -> None:
    box = UndistortedBoundingBox(2.0, 3.0, 8.0, 9.0)
    hsv = np.full((12, 16, 3), (0, 0, 114), dtype=np.uint8)
    hsv[5, 5] = (60, 255, 200)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    observation = detector([[detection(box=box)]]).detect(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )[0]

    assert observation.target_class is TargetClass.UNKNOWN
    assert not np.any(observation.color_segmentation.mask)


def test_detector_filters_low_detection_and_supports_empty() -> None:
    image = image_with_regions()
    result = detector([[detection(confidence=0.1)], []]).detect(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )
    assert result == []


def test_detector_rejects_stale_or_unmapped_results() -> None:
    image = image_with_regions()
    with pytest.raises(StaleObservationError, match="exceeds") as raised:
        detector([[detection()]]).detect(
            frame(image),
            image,
            result_timestamp_ns=1_151_000_000,
        )
    assert raised.value.age_ms == pytest.approx(151.0)
    assert raised.value.max_age_ms == pytest.approx(150.0)
    with pytest.raises(ValueError, match="no configured mapping"):
        detector([[detection(class_id=4)]]).detect(
            frame(image),
            image,
            result_timestamp_ns=1_010_000_000,
        )


def test_realtime_detector_drops_only_stale_observations() -> None:
    image = image_with_regions()
    result = detector([[detection()]]).detect_realtime(
        frame(image),
        image,
        result_timestamp_ns=1_151_000_000,
    )

    assert result.observations == ()
    assert result.stale_dropped
    assert result.dropped_stale_age_ms == pytest.approx(151.0)


def test_realtime_detector_returns_current_observations() -> None:
    image = image_with_regions()
    result = detector([[detection()]]).detect_realtime(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )

    assert len(result.observations) == 1
    assert not result.stale_dropped
    assert result.dropped_stale_age_ms is None


def test_realtime_detector_preserves_non_stale_errors() -> None:
    image = image_with_regions()
    with pytest.raises(ValueError, match="no configured mapping"):
        detector([[detection(class_id=4)]]).detect_realtime(
            frame(image),
            image,
            result_timestamp_ns=1_010_000_000,
        )


def test_perception_visualization_isolated_and_marks_stale_results() -> None:
    image = image_with_regions()
    observations = detector([[detection()]]).detect(
        frame(image),
        image,
        result_timestamp_ns=1_010_000_000,
    )
    original = image.copy()

    preview = render_target_observations(image, observations)
    stale_preview = render_target_observations(
        image,
        (),
        dropped_stale_age_ms=151.0,
    )

    assert np.array_equal(image, original)
    assert preview.shape == image.shape
    assert not np.array_equal(preview, image)
    assert not np.array_equal(stale_preview, image)


def test_perception_frame_renderer_keeps_latest_result_off_realtime_thread() -> None:
    image = image_with_regions()
    source_frame = CameraFrame(7, time.monotonic_ns() - 1_000_000, image)
    renderer = PerceptionFrameRenderer(
        lambda: detector([[detection()]]),
    )
    renderer.start()
    try:
        renderer.submit(source_frame)
        deadline = time.monotonic() + 1.0
        rendered = None
        while time.monotonic() < deadline:
            rendered = renderer.latest()
            if rendered is not None:
                break
            time.sleep(0.001)
        assert rendered is not None
        assert rendered.sequence == source_frame.sequence
        assert rendered.timestamp_ns == source_frame.timestamp_ns
        assert not np.array_equal(rendered.image_bgr, source_frame.image_bgr)
        snapshot = renderer.latest_snapshot()
        assert snapshot is not None
        assert snapshot.frame_sequence == source_frame.sequence
        assert len(snapshot.observations) == 1
    finally:
        renderer.stop()


def test_perception_frame_renderer_initializes_detector_during_start() -> None:
    calls: list[object] = []

    def build_detector() -> TargetPoseDetector:
        detector_instance = detector([[detection()]])
        calls.append(detector_instance)
        return detector_instance

    renderer = PerceptionFrameRenderer(build_detector)
    renderer.start()
    try:
        assert len(calls) == 1
    finally:
        renderer.stop()


def test_invalid_observation_coordinates_fail() -> None:
    box = UndistortedBoundingBox(0, 0, 5, 5)
    segmentation = RoiColorSegmentation(
        candidate_class=TargetClass.GREEN_SUPPLY,
        status=ColorSegmentationStatus.ACCEPTED,
        roi_box=box,
        mask=np.full((5, 5), 255, dtype=np.uint8),
        color_fraction=1.0,
        dominance=1.0,
    )
    with pytest.raises(ValueError, match="outside"):
        TargetObservation(
            frame_sequence=0,
            capture_timestamp_ns=0,
            result_timestamp_ns=1,
            image_size=(10, 10),
            model_target_class=TargetClass.GREEN_SUPPLY,
            target_class=TargetClass.GREEN_SUPPLY,
            class_probabilities=ClassProbabilities.from_top_class(
                TargetClass.GREEN_SUPPLY, 1.0
            ),
            detection_confidence=0.8,
            box=box,
            color_segmentation=segmentation,
            k0=UndistortedPixel(11, 1),
            k0_confidence=0.9,
            ground_point=None,
            quality=frozenset(),
        )


def test_observations_to_evaluation_records_full_chain() -> None:
    blue_box = UndistortedBoundingBox(0, 0, 4, 4)
    green_box = UndistortedBoundingBox(10, 0, 14, 4)
    image = image_with_regions(
        (blue_box, (95, 150, 200)),
        (green_box, (60, 255, 200)),
    )
    observations = detector(
        [
            [
                detection(class_id=1, box=blue_box),
                detection(class_id=0, box=green_box),
            ]
        ]
    ).detect(
        frame(image),
        image,
        result_timestamp_ns=1_020_000_000,
    )
    annotations = [
        TargetAnnotation(
            "truth_1",
            TargetClass.GREEN_SUPPLY,
            blue_box,
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
    assert matched["model_predicted_class"] == "blue_danger"
    assert matched["hsv_candidate_class"] == "blue_danger"
    assert matched["hsv_status"] == "accepted"
    assert matched["hsv_color_fraction"] == pytest.approx(1.0)
    assert matched["hsv_dominance"] == pytest.approx(1.0)
    assert matched["quality"] == []
