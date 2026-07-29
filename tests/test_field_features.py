from __future__ import annotations

import cv2
import numpy as np
import pytest

from rescue_vision.camera.frame import CameraFrame
from rescue_vision.geometry.ground_projector import BevConfig, GroundProjector
from rescue_vision.perception import (
    BoundaryFeatureKind,
    FieldFeatureConfig,
    FieldFeatureDetector,
    FieldFeatureQuality,
    HsvRange,
    SafeZoneColor,
    SafeZoneSide,
    StaleObservationError,
)


def ranges(lower, upper) -> tuple[HsvRange, ...]:
    return (HsvRange(tuple(lower), tuple(upper)),)


def config(**overrides) -> FieldFeatureConfig:
    values = {
        "enabled": True,
        "safe_red": (
            HsvRange((0, 80, 80), (12, 255, 255)),
            HsvRange((170, 80, 80), (179, 255, 255)),
        ),
        "safe_blue": ranges((90, 60, 70), (130, 255, 255)),
        "start_magenta": ranges((145, 80, 80), (165, 255, 255)),
        "entrance_purple": ranges((130, 60, 50), (144, 255, 255)),
        "dark_marking": ranges((0, 0, 0), (179, 255, 80)),
        "morphology_kernel_size": 3,
        "open_iterations": 0,
        "close_iterations": 1,
        "min_region_area_fraction": 0.001,
        "min_rectangularity": 0.50,
        "safe_width_mm": 200.0,
        "safe_depth_mm": 100.0,
        "start_side_mm": 60.0,
        "dimension_tolerance_fraction": 0.25,
        "entrance_color_fraction": 0.08,
        "divider_dark_fraction": 0.08,
        "center_min_axis_span_fraction": 0.20,
        "center_max_gap_fraction": 0.06,
        "center_min_gap_count": 2,
        "center_perpendicular_tolerance_deg": 15.0,
        "boundary_canny_low_threshold": 50,
        "boundary_canny_high_threshold": 150,
        "boundary_min_line_length_fraction": 0.20,
        "boundary_corner_tolerance_deg": 20.0,
        "boundary_max_features": 6,
    }
    values.update(overrides)
    return FieldFeatureConfig(**values)


def projector() -> GroundProjector:
    bev = BevConfig(
        x_min=0.0,
        x_max=400.0,
        y_min=-200.0,
        y_max=200.0,
        mm_per_pixel=1.0,
    )
    ground_to_bev = GroundProjector.make_ground_to_bev_matrix(bev)
    return GroundProjector(np.linalg.inv(ground_to_bev), bev)


def bgr(hue: int, saturation: int = 255, value: int = 255) -> tuple[int, int, int]:
    hsv = np.asarray([[[hue, saturation, value]]], dtype=np.uint8)
    pixel = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(component) for component in pixel)


def frame(image: np.ndarray, *, timestamp_ns: int = 1_000_000) -> CameraFrame:
    return CameraFrame(7, timestamp_ns, image)


def valid_mask(image: np.ndarray) -> np.ndarray:
    return np.full(image.shape[:2], 255, dtype=np.uint8)


def safe_zone_scene(*, include_purple: bool = True) -> np.ndarray:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (100, 50), (300, 150), bgr(3), thickness=-1)
    cv2.line(image, (200, 50), (200, 150), (0, 0, 0), thickness=5)
    if include_purple:
        cv2.line(image, (100, 150), (300, 150), bgr(138), thickness=7)
    return image


def test_safe_zone_uses_approach_frame_and_keeps_physical_color() -> None:
    image = safe_zone_scene()
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
        ground_projector=projector(),
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    assert len(result.safe_zones) == 1
    observation = result.safe_zones[0]
    assert observation.physical_color is SafeZoneColor.RED
    assert observation.entrance is not None
    assert observation.divider is not None
    assert {half.side for half in observation.halves} == set(SafeZoneSide)
    left = next(
        half
        for half in observation.halves
        if half.side is SafeZoneSide.APPROACH_LEFT
    )
    right = next(
        half
        for half in observation.halves
        if half.side is SafeZoneSide.APPROACH_RIGHT
    )
    assert np.mean([point.u for point in left.polygon_undistorted]) < 200.0
    assert np.mean([point.u for point in right.polygon_undistorted]) > 200.0
    assert FieldFeatureQuality.SIDE_UNRESOLVED not in observation.quality


def test_safe_zone_does_not_invent_sides_without_entrance_evidence() -> None:
    image = safe_zone_scene(include_purple=False)
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
        ground_projector=projector(),
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    observation = result.safe_zones[0]
    assert observation.entrance is None
    assert observation.halves == ()
    assert FieldFeatureQuality.ENTRANCE_UNRESOLVED in observation.quality
    assert FieldFeatureQuality.SIDE_UNRESOLVED in observation.quality


def test_safe_zone_without_projector_keeps_pixel_observation_only() -> None:
    image = safe_zone_scene()
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    observation = result.safe_zones[0]
    assert observation.polygon_ground is None
    assert observation.halves == ()
    assert FieldFeatureQuality.NO_GROUND_PROJECTION in observation.quality
    assert FieldFeatureQuality.SIDE_UNRESOLVED in observation.quality


def test_start_zone_is_unlabelled_region_with_ground_corners() -> None:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.rectangle(image, (20, 250), (80, 310), bgr(150), thickness=-1)
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
        ground_projector=projector(),
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    assert len(result.start_zones) == 1
    observation = result.start_zones[0]
    assert len(observation.polygon_undistorted) == 4
    assert observation.polygon_ground is not None
    assert not hasattr(observation, "zone_number")
    assert not hasattr(observation, "zone_id")


def test_center_cross_recovers_two_dashed_perpendicular_axes() -> None:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    for start in range(60, 341, 45):
        cv2.line(image, (start, 200), (min(start + 28, 340), 200), (0, 0, 0), 3)
        cv2.line(image, (200, start), (200, min(start + 28, 340)), (0, 0, 0), 3)
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
        ground_projector=projector(),
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    assert result.center_cross is not None
    assert len(result.center_cross.axes) == 2
    assert result.center_cross.intersection_ground is not None
    assert FieldFeatureQuality.PARTIAL not in result.center_cross.quality


def test_single_long_marking_is_only_partial_center_evidence() -> None:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    for start in range(40, 341, 45):
        cv2.line(image, (start, 200), (min(start + 28, 360), 200), (0, 0, 0), 3)
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
        ground_projector=projector(),
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    assert result.center_cross is not None
    assert len(result.center_cross.axes) == 1
    assert result.center_cross.intersection_undistorted is None
    assert FieldFeatureQuality.PARTIAL in result.center_cross.quality


def test_solid_perpendicular_lines_are_not_center_cross() -> None:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.line(image, (40, 80), (360, 80), (0, 0, 0), 5)
    cv2.line(image, (40, 80), (40, 360), (0, 0, 0), 5)
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
        ground_projector=projector(),
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    assert result.center_cross is None


def test_boundary_candidates_are_explicitly_low_confidence_and_bounded() -> None:
    image = np.full((400, 400, 3), 255, dtype=np.uint8)
    cv2.line(image, (40, 80), (360, 80), (0, 0, 0), 5)
    cv2.line(image, (40, 80), (40, 360), (0, 0, 0), 5)
    detector = FieldFeatureDetector(
        config(boundary_max_features=3),
        max_observation_age_ms=100.0,
        ground_projector=projector(),
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=1_100_000,
    )

    assert 1 <= len(result.boundary_features) <= 3
    assert any(
        item.kind is BoundaryFeatureKind.FIELD_CORNER
        for item in result.boundary_features
    )
    assert all(
        FieldFeatureQuality.LOW_CONFIDENCE_BOUNDARY in item.quality
        for item in result.boundary_features
    )


def test_undistortion_fill_boundary_is_not_reported_as_a_field_feature() -> None:
    image = np.full((400, 400, 3), 114, dtype=np.uint8)
    image[40:360, 40:360] = 255
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    mask[40:360, 40:360] = 255
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=100.0,
    )

    result = detector.detect(
        frame(image),
        image,
        valid_mask=mask,
        result_timestamp_ns=1_100_000,
    )

    assert result.boundary_features == ()


def test_field_detector_validates_frame_age_and_image_contract() -> None:
    image = np.full((40, 40, 3), 255, dtype=np.uint8)
    detector = FieldFeatureDetector(
        config(),
        max_observation_age_ms=1.0,
    )

    with pytest.raises(StaleObservationError):
        detector.detect(
            frame(image, timestamp_ns=1_000_000),
            image,
            valid_mask=valid_mask(image),
            result_timestamp_ns=3_000_000,
        )
    realtime = detector.detect_realtime(
        frame(image, timestamp_ns=1_000_000),
        image,
        valid_mask=valid_mask(image),
        result_timestamp_ns=3_000_000,
    )
    assert realtime.stale_dropped
    assert realtime.result is None
    assert realtime.dropped_stale_age_ms == pytest.approx(2.0)
    with pytest.raises(ValueError, match="uint8"):
        detector.detect(
            frame(image),
            image.astype(np.float32),
            valid_mask=valid_mask(image),
            result_timestamp_ns=1_100_000,
        )
    with pytest.raises(ValueError, match="matching"):
        detector.detect(
            frame(image),
            image,
            valid_mask=np.full((39, 40), 255, dtype=np.uint8),
            result_timestamp_ns=1_100_000,
        )
    with pytest.raises(ValueError, match="either 0 or 255"):
        detector.detect(
            frame(image),
            image,
            valid_mask=np.full(image.shape[:2], 1, dtype=np.uint8),
            result_timestamp_ns=1_100_000,
        )
    with pytest.raises(ValueError, match="at least one valid pixel"):
        detector.detect(
            frame(image),
            image,
            valid_mask=np.zeros(image.shape[:2], dtype=np.uint8),
            result_timestamp_ns=1_100_000,
        )


def test_disabled_config_is_not_silently_instantiated() -> None:
    with pytest.raises(ValueError, match="enabled=true"):
        FieldFeatureDetector(
            config(enabled=False),
            max_observation_age_ms=100.0,
        )
