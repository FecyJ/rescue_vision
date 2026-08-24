from __future__ import annotations

import cv2
import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.perception import (
    BoundaryFeatureKind,
    BoundaryFeatureObservation,
    FakeInferenceBackend,
    FieldBoundaryConfig,
    FieldBoundaryEstimator,
    FieldBoundaryMask,
    FieldFeatureDetectionResult,
    FieldFeatureQuality,
    FieldMaskState,
    HsvColorClassifierConfig,
    HsvRange,
    ModelDetection,
    ObservationQuality,
    TargetClass,
    TargetPoseDetector,
    UndistortedBoundingBox,
)
from rescue_vision.world import (
    PhysicalRegionKind,
    PhysicalStaticRegion,
    StaticFieldMap,
    default_static_field_map,
)


def boundary_config(**overrides: object) -> FieldBoundaryConfig:
    values: dict[str, object] = {
        "enabled": True,
        "hard_mask_enabled": True,
        "min_candidate_confidence": 0.25,
        "min_confirmations": 2,
        "max_missed_frames": 1,
        "line_angle_tolerance_deg": 12.0,
        "line_distance_tolerance_mm": 20.0,
        "ransac_inlier_distance_mm": 5.0,
        "rectangle_tolerance_fraction": 0.20,
        "boundary_band_mm": 10.0,
        "segment_extension_mm": 5.0,
        "min_filter_confidence": 0.30,
        "max_mask_age_ms": 100.0,
        "mask_blur_radius_px": 0,
        "neutral_fill_bgr": (114, 114, 114),
    }
    values.update(overrides)
    return FieldBoundaryConfig(**values)  # type: ignore[arg-type]


def static_map() -> StaticFieldMap:
    field = PhysicalStaticRegion(
        "field",
        PhysicalRegionKind.FIELD,
        (
            FieldPoint(-1500.0, -1500.0),
            FieldPoint(1500.0, -1500.0),
            FieldPoint(1500.0, 1500.0),
            FieldPoint(-1500.0, 1500.0),
        ),
    )
    return StaticFieldMap(default_static_field_map().center_cross, (field,))


def projector() -> GroundProjector:
    return GroundProjector(
        np.eye(3),
        BevConfig(0.0, 100.0, -50.0, 50.0, 1.0),
    )


def boundary_observation(
    start: GroundPoint,
    end: GroundPoint,
    *,
    timestamp_ns: int,
    confidence: float = 0.40,
) -> BoundaryFeatureObservation:
    direction = np.asarray((end.x - start.x, end.y - start.y), dtype=np.float64)
    direction /= np.linalg.norm(direction)
    normal = np.asarray((-direction[1], direction[0]), dtype=np.float64)
    midpoint = np.asarray(((start.x + end.x) / 2, (start.y + end.y) / 2))
    if float(np.dot(normal, -midpoint)) < 0.0:
        normal = -normal
    offset = -float(np.dot(normal, midpoint))
    return BoundaryFeatureObservation(
        kind=BoundaryFeatureKind.FENCE_BASE_SEGMENT,
        points_undistorted=(
            UndistortedPixel(start.x, start.y + 50.0),
            UndistortedPixel(end.x, end.y + 50.0),
        ),
        points_ground=(start, end),
        confidence=confidence,
        quality=frozenset({FieldFeatureQuality.LOW_CONFIDENCE_BOUNDARY}),
        capture_timestamp_ns=timestamp_ns,
        interior_normal_ground=(float(normal[0]), float(normal[1])),
        line_offset_mm=offset,
    )


def feature_result(
    timestamp_ns: int,
    boundaries: tuple[BoundaryFeatureObservation, ...],
) -> FieldFeatureDetectionResult:
    return FieldFeatureDetectionResult(
        frame_sequence=timestamp_ns // 1_000_000,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns + 1,
        image_size=(100, 100),
        safe_zones=(),
        start_zones=(),
        center_cross=None,
        boundary_features=boundaries,
    )


def state_at_ground(
    mask: FieldBoundaryMask,
    ground: GroundPoint,
    ground_projector: GroundProjector,
) -> FieldMaskState:
    pixel = ground_projector.ground_to_bev_pixel(ground)
    return FieldMaskState(mask.bev_state[round(pixel.v), round(pixel.u)])


def test_boundary_requires_temporal_confirmation_and_keeps_open_extent() -> None:
    ground_projector = projector()
    estimator = FieldBoundaryEstimator(
        boundary_config(),
        static_map=static_map(),
        ground_projector=ground_projector,
    )
    valid = np.full((100, 100), 255, dtype=np.uint8)
    first_time = 1_000_000_000
    first_boundary = boundary_observation(
        GroundPoint(50.0, -30.0),
        GroundPoint(50.0, 30.0),
        timestamp_ns=first_time,
    )

    first = estimator.update(
        feature_result(first_time, (first_boundary,)),
        valid_mask=valid,
    )
    assert first.boundaries == ()
    assert not first.hard_mask_ready

    second_time = first_time + 20_000_000
    second_boundary = boundary_observation(
        GroundPoint(50.0, -30.0),
        GroundPoint(50.0, 30.0),
        timestamp_ns=second_time,
    )
    second = estimator.update(
        feature_result(second_time, (second_boundary,)),
        valid_mask=valid,
    )

    assert len(second.boundaries) == 1
    assert second.hard_mask_ready
    assert state_at_ground(
        second, GroundPoint(20.0, 0.0), ground_projector
    ) is FieldMaskState.INSIDE
    assert state_at_ground(
        second, GroundPoint(80.0, 0.0), ground_projector
    ) is FieldMaskState.OUTSIDE
    assert state_at_ground(
        second, GroundPoint(80.0, 45.0), ground_projector
    ) is FieldMaskState.UNCERTAIN


def test_no_boundary_degrades_to_uncertain_without_hard_mask() -> None:
    estimator = FieldBoundaryEstimator(
        boundary_config(min_confirmations=1),
        static_map=static_map(),
        ground_projector=projector(),
    )
    result = estimator.update(
        feature_result(1_000_000_000, ()),
        valid_mask=np.full((100, 100), 255, dtype=np.uint8),
    )

    assert not result.hard_mask_ready
    assert not result.filter_ready
    assert np.all(result.bev_state == FieldMaskState.UNCERTAIN)


def test_reusing_same_frame_cannot_satisfy_temporal_confirmation() -> None:
    estimator = FieldBoundaryEstimator(
        boundary_config(),
        static_map=static_map(),
        ground_projector=projector(),
    )
    timestamp_ns = 1_000_000_000
    observation = boundary_observation(
        GroundPoint(50.0, -30.0),
        GroundPoint(50.0, 30.0),
        timestamp_ns=timestamp_ns,
    )
    current = feature_result(timestamp_ns, (observation,))
    valid = np.full((100, 100), 255, dtype=np.uint8)
    estimator.update(current, valid_mask=valid)

    with pytest.raises(ValueError, match="strictly increasing"):
        estimator.update(current, valid_mask=valid)


def test_inference_mask_only_replaces_outside_and_preserves_source() -> None:
    image = np.full((8, 10, 3), (10, 20, 30), dtype=np.uint8)
    original = image.copy()
    state = np.full((8, 10), FieldMaskState.UNCERTAIN, dtype=np.uint8)
    state[:, :3] = FieldMaskState.OUTSIDE
    mask = FieldBoundaryMask(
        capture_timestamp_ns=100,
        expires_timestamp_ns=200,
        image_state=state,
        bev_state=state,
        boundaries=(),
        confidence=0.8,
        filter_ready=True,
        hard_mask_ready=True,
        neutral_fill_bgr=(114, 114, 114),
        blur_radius_px=0,
    )

    masked = mask.mask_for_inference(image, timestamp_ns=150)

    assert np.array_equal(image, original)
    assert np.all(masked[:, :3] == 114)
    assert np.array_equal(masked[:, 3:], image[:, 3:])
    assert mask.mask_for_inference(image, timestamp_ns=201) is image


def color_config() -> HsvColorClassifierConfig:
    return HsvColorClassifierConfig(
        green_supply=(HsvRange((35, 70, 71), (84, 255, 255)),),
        black_core=(HsvRange((0, 0, 0), (179, 255, 70)),),
        orange_injured=(HsvRange((0, 90, 80), (20, 255, 255)),),
        blue_danger=(HsvRange((85, 50, 71), (110, 255, 255)),),
        min_color_fraction=0.15,
        min_color_dominance=0.70,
        min_dominance_margin=0.20,
        morphology_kernel_size=3,
        open_iterations=0,
        close_iterations=0,
        min_component_area_fraction=0.002,
    )


def test_post_filter_drops_normal_target_but_keeps_danger_outside() -> None:
    hsv = np.full((12, 20, 3), (0, 0, 114), dtype=np.uint8)
    hsv[2:10, 1:9] = (60, 255, 200)
    hsv[2:10, 11:19] = (95, 150, 200)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detections = (
        ModelDetection(
            0,
            0.8,
            UndistortedBoundingBox(1.0, 2.0, 9.0, 10.0),
            UndistortedPixel(5.0, 8.0),
            0.9,
        ),
        ModelDetection(
            1,
            0.8,
            UndistortedBoundingBox(11.0, 2.0, 19.0, 10.0),
            UndistortedPixel(15.0, 8.0),
            0.9,
        ),
    )
    detector = TargetPoseDetector(
        FakeInferenceBackend((detections,)),
        class_mapping={0: TargetClass.GREEN_SUPPLY, 1: TargetClass.BLUE_DANGER},
        detection_threshold=0.25,
        k0_threshold=0.5,
        color_classifier=color_config(),
        max_observation_age_ms=150.0,
    )
    state = np.full(image.shape[:2], FieldMaskState.OUTSIDE, dtype=np.uint8)
    field_mask = FieldBoundaryMask(
        capture_timestamp_ns=1_000_000_000,
        expires_timestamp_ns=1_100_000_000,
        image_state=state,
        bev_state=state,
        boundaries=(),
        confidence=0.8,
        filter_ready=True,
        hard_mask_ready=False,
        neutral_fill_bgr=(114, 114, 114),
        blur_radius_px=0,
    )

    observations = detector.detect(
        CameraFrame(1, 1_000_000_000, image),
        image,
        field_mask=field_mask,
        result_timestamp_ns=1_010_000_000,
    )

    assert len(observations) == 1
    assert observations[0].target_class is TargetClass.BLUE_DANGER
    assert ObservationQuality.OUTSIDE_FIELD_SUSPECTED in observations[0].quality


def test_stale_mask_does_not_filter_or_add_boundary_quality() -> None:
    hsv = np.full((12, 20, 3), (60, 255, 200), dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detection = ModelDetection(
        0,
        0.8,
        UndistortedBoundingBox(1.0, 2.0, 9.0, 10.0),
        UndistortedPixel(5.0, 8.0),
        0.9,
    )
    detector = TargetPoseDetector(
        FakeInferenceBackend(((detection,),)),
        class_mapping={0: TargetClass.GREEN_SUPPLY},
        detection_threshold=0.25,
        k0_threshold=0.5,
        color_classifier=color_config(),
        max_observation_age_ms=150.0,
    )
    state = np.full(image.shape[:2], FieldMaskState.OUTSIDE, dtype=np.uint8)
    stale = FieldBoundaryMask(
        capture_timestamp_ns=1,
        expires_timestamp_ns=2,
        image_state=state,
        bev_state=state,
        boundaries=(),
        confidence=0.8,
        filter_ready=True,
        hard_mask_ready=True,
        neutral_fill_bgr=(114, 114, 114),
        blur_radius_px=0,
    )

    observations = detector.detect(
        CameraFrame(1, 1_000_000_000, image),
        image,
        field_mask=stale,
        result_timestamp_ns=1_010_000_000,
    )

    assert len(observations) == 1
    assert observations[0].quality == frozenset()


def test_model_danger_conflict_is_not_dropped_outside() -> None:
    hsv = np.full((12, 20, 3), (60, 255, 200), dtype=np.uint8)
    image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    detection = ModelDetection(
        1,
        0.8,
        UndistortedBoundingBox(1.0, 2.0, 9.0, 10.0),
        UndistortedPixel(5.0, 8.0),
        0.9,
    )
    detector = TargetPoseDetector(
        FakeInferenceBackend(((detection,),)),
        class_mapping={1: TargetClass.BLUE_DANGER},
        detection_threshold=0.25,
        k0_threshold=0.5,
        color_classifier=color_config(),
        max_observation_age_ms=150.0,
    )
    state = np.full(image.shape[:2], FieldMaskState.OUTSIDE, dtype=np.uint8)
    field_mask = FieldBoundaryMask(
        capture_timestamp_ns=1_000_000_000,
        expires_timestamp_ns=1_100_000_000,
        image_state=state,
        bev_state=state,
        boundaries=(),
        confidence=0.8,
        filter_ready=True,
        hard_mask_ready=False,
        neutral_fill_bgr=(114, 114, 114),
        blur_radius_px=0,
    )

    observations = detector.detect(
        CameraFrame(1, 1_000_000_000, image),
        image,
        field_mask=field_mask,
        result_timestamp_ns=1_010_000_000,
    )

    assert len(observations) == 1
    assert observations[0].model_target_class is TargetClass.BLUE_DANGER
    assert observations[0].target_class is TargetClass.GREEN_SUPPLY
    assert ObservationQuality.POSE_COLOR_CONFLICT in observations[0].quality
    assert ObservationQuality.OUTSIDE_FIELD_SUSPECTED in observations[0].quality
