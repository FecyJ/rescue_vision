from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.localization import (
    CenterCrossLocalizationQuality,
    CenterCrossPoseCandidate,
    CenterCrossPoseObservation,
    CenterCrossSelectionSource,
    FieldPose2D,
    SafeZoneCornerLocalizer,
    SafeZoneCornerPoseObservation,
    StaticFieldLandmarkTracker,
    StaticLandmarkTrackingConfig,
    angular_distance,
    select_same_frame_pose_observation,
)
from rescue_vision.perception import (
    CenterCrossConfirmation,
    CenterCrossObservation,
    FieldFeatureDetectionResult,
    FieldFeatureSearchHint,
    FieldFeatureQuality,
    FieldPoseKeypoint,
    LineSegmentObservation,
    SafeZoneColor,
    SafeZoneCornerRole,
    SafeZoneObservation,
    UndistortedBoundingBox,
)
from rescue_vision.world import (
    PhysicalRegionKind,
    PhysicalStaticRegion,
    StaticFieldMap,
    StaticSafeZoneLandmarks,
    TeamColor,
    default_static_field_map,
)


def static_map() -> StaticFieldMap:
    def region(
        region_id: str,
        kind: PhysicalRegionKind,
        min_x: float,
        max_x: float,
        min_y: float,
        max_y: float,
    ) -> PhysicalStaticRegion:
        return PhysicalStaticRegion(
            region_id,
            kind,
            (
                FieldPoint(min_x, min_y),
                FieldPoint(max_x, min_y),
                FieldPoint(max_x, max_y),
                FieldPoint(min_x, max_y),
            ),
        )

    return StaticFieldMap(
        default_static_field_map().center_cross,
        (
            region("red-material", PhysicalRegionKind.RED_MATERIAL, -300, 0, 1200, 1500),
            region("red-injured", PhysicalRegionKind.RED_INJURED, 0, 300, 1200, 1500),
            region("blue-injured", PhysicalRegionKind.BLUE_INJURED, -300, 0, -1500, -1200),
            region("blue-material", PhysicalRegionKind.BLUE_MATERIAL, 0, 300, -1500, -1200),
        ),
        (
            StaticSafeZoneLandmarks(
                TeamColor.RED,
                FieldPoint(0.0, 1200.0),
                FieldPoint(-300.0, 1200.0),
                FieldPoint(300.0, 1200.0),
                True,
                True,
            ),
            StaticSafeZoneLandmarks(
                TeamColor.BLUE,
                FieldPoint(0.0, -1200.0),
                FieldPoint(-300.0, -1200.0),
                FieldPoint(300.0, -1200.0),
                True,
                True,
            ),
        ),
    )


def test_safe_zone_landmarks_require_measurement_and_common_baseline() -> None:
    with pytest.raises(ValueError, match="must be measured"):
        StaticSafeZoneLandmarks(
            TeamColor.RED,
            FieldPoint(0.0, 1200.0),
            FieldPoint(-300.0, 1200.0),
            FieldPoint(300.0, 1200.0),
            False,
            True,
        )
    with pytest.raises(ValueError, match="collinear"):
        StaticSafeZoneLandmarks(
            TeamColor.RED,
            FieldPoint(0.0, 1250.0),
            FieldPoint(-300.0, 1200.0),
            FieldPoint(300.0, 1200.0),
            True,
            True,
        )


def empty_result(timestamp_ns: int) -> FieldFeatureDetectionResult:
    return FieldFeatureDetectionResult(
        frame_sequence=timestamp_ns,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        image_size=(640, 480),
        safe_zones=(),
        center_cross=None,
    )


def cross_result(
    timestamp_ns: int,
    center: GroundPoint,
    *,
    rotation_rad: float = 0.0,
    confirmation: CenterCrossConfirmation = CenterCrossConfirmation.CANDIDATE,
    confidence: float = 0.8,
) -> FieldFeatureDetectionResult:
    forward = (math.cos(rotation_rad), math.sin(rotation_rad))
    left = (-math.sin(rotation_rad), math.cos(rotation_rad))
    axes = []
    for direction in (forward, left):
        start = GroundPoint(
            center.x - 300.0 * direction[0],
            center.y - 300.0 * direction[1],
        )
        end = GroundPoint(
            center.x + 300.0 * direction[0],
            center.y + 300.0 * direction[1],
        )
        axes.append(
            LineSegmentObservation(
                UndistortedPixel(start.x, start.y),
                UndistortedPixel(end.x, end.y),
                start,
                end,
            )
        )
    return FieldFeatureDetectionResult(
        frame_sequence=timestamp_ns,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
        image_size=(640, 480),
        safe_zones=(),
        center_cross=CenterCrossObservation(
            box=UndistortedBoundingBox(0.0, 0.0, 639.0, 479.0),
            intersection=FieldPoseKeypoint(
                UndistortedPixel(center.x, center.y),
                center,
                confidence,
            ),
            axes=tuple(axes),
            confidence=confidence,
            quality=frozenset(),
            confirmation=confirmation,
            axis_fit_residuals_px=(0.5, 0.5),
            axis_angle_deg=90.0,
        ),
    )


def field_to_ground(pose: FieldPose2D, point: FieldPoint) -> GroundPoint:
    dx = point.x - pose.position.x
    dy = point.y - pose.position.y
    cosine = math.cos(pose.heading_rad)
    sine = math.sin(pose.heading_rad)
    return GroundPoint(cosine * dx + sine * dy, -sine * dx + cosine * dy)


def test_tracker_builds_map_hint_with_center_axis_directions() -> None:
    tracker = StaticFieldLandmarkTracker(
        static_map(),
        StaticLandmarkTrackingConfig(confirmation_hits=2),
    )
    pose = FieldPose2D(FieldPoint(-1350.0, -1350.0), math.pi / 2.0)

    hint = tracker.search_hint(
        pose,
        position_uncertainty_mm=20.0,
        heading_uncertainty_rad=math.radians(2.0),
    )

    assert hint is not None
    assert hint.cross_center_ground.x == pytest.approx(1350.0)
    assert hint.cross_center_ground.y == pytest.approx(-1350.0)
    assert {item.physical_color for item in hint.safe_zone_regions} == {
        SafeZoneColor.RED,
        SafeZoneColor.BLUE,
    }
    assert hint.cross_axis_directions_ground[0] == pytest.approx((0.0, -1.0))
    assert hint.cross_axis_directions_ground[1] == pytest.approx((1.0, 0.0))


def test_tracker_disables_hint_for_uncertain_prior() -> None:
    tracker = StaticFieldLandmarkTracker(static_map())
    pose = FieldPose2D(FieldPoint(0.0, 0.0), 0.0)
    assert tracker.search_hint(
        pose,
        position_uncertainty_mm=301.0,
        heading_uncertainty_rad=0.0,
    ) is None


def _safe_zone_from_pose(
    pose: FieldPose2D,
    *,
    color: SafeZoneColor = SafeZoneColor.RED,
    swap_image_order: bool = False,
) -> SafeZoneObservation:
    anchor_field = FieldPoint(0.0, 1200.0 if color is SafeZoneColor.RED else -1200.0)
    first_field = FieldPoint(-300.0, anchor_field.y)
    second_field = FieldPoint(300.0, anchor_field.y)
    first_ground = field_to_ground(pose, first_field)
    second_ground = field_to_ground(pose, second_field)
    if swap_image_order:
        first_ground, second_ground = second_ground, first_ground
    return SafeZoneObservation(
        UndistortedBoundingBox(0.0, 0.0, 100.0, 100.0),
        FieldPoseKeypoint(UndistortedPixel(50.0, 70.0), field_to_ground(pose, anchor_field), 0.95),
        FieldPoseKeypoint(UndistortedPixel(20.0, 70.0), first_ground, 0.95),
        FieldPoseKeypoint(UndistortedPixel(80.0, 70.0), second_ground, 0.95),
        color,
        0.9,
        frozenset(),
    )


def test_safe_zone_corner_pair_recovers_unique_pose() -> None:
    pose = FieldPose2D(FieldPoint(120.0, -240.0), math.radians(23.0))
    zone = _safe_zone_from_pose(pose)
    result = FieldFeatureDetectionResult(
        1, 1_000_000, 1_100_000, (640, 480), (zone,), None
    )

    observation = SafeZoneCornerLocalizer(static_map()).localize(result, prior_pose=pose)

    assert observation is not None
    assert observation.pose.position.x == pytest.approx(pose.position.x, abs=1e-6)
    assert observation.pose.position.y == pytest.approx(pose.position.y, abs=1e-6)
    assert angular_distance(observation.pose.heading_rad, pose.heading_rad) < 1e-9
    assert observation.fit_residual_mm < 1e-6
    assert observation.source == "red_safe_zone_corners"


def test_safe_zone_k0_and_one_corner_can_recover_pose() -> None:
    pose = FieldPose2D(FieldPoint(120.0, -240.0), math.radians(23.0))
    zone = replace(
        _safe_zone_from_pose(pose),
        image_right_landmark=FieldPoseKeypoint(None, None, 0.0),
        quality=frozenset({FieldFeatureQuality.KEYPOINT_UNAVAILABLE}),
    )
    result = FieldFeatureDetectionResult(
        1, 1_000_000, 1_100_000, (640, 480), (zone,), None
    )

    observation = SafeZoneCornerLocalizer(static_map()).localize(
        result,
        prior_pose=pose,
    )

    assert observation is not None
    assert observation.pose.position.x == pytest.approx(pose.position.x, abs=1e-6)
    assert observation.pose.position.y == pytest.approx(pose.position.y, abs=1e-6)
    assert angular_distance(observation.pose.heading_rad, pose.heading_rad) < 1e-9
    assert observation.used_roles == (
        SafeZoneCornerRole.GROUND_ANCHOR,
        SafeZoneCornerRole.ENTRANCE_LEFT,
    )


def test_safe_zone_rejects_position_when_k0_corner_lengths_do_not_match() -> None:
    pose = FieldPose2D(FieldPoint(120.0, -240.0), math.radians(23.0))
    zone = _safe_zone_from_pose(pose)
    anchor = zone.ground_anchor.ground
    assert anchor is not None
    bad_zone = replace(
        zone,
        ground_anchor=FieldPoseKeypoint(
            zone.ground_anchor.undistorted,
            GroundPoint(anchor.x + 300.0, anchor.y),
            zone.ground_anchor.confidence,
        ),
    )
    result = FieldFeatureDetectionResult(
        1, 1_000_000, 1_100_000, (640, 480), (bad_zone,), None
    )

    assert SafeZoneCornerLocalizer(static_map()).localize(
        result,
        prior_pose=pose,
    ) is None


def test_safe_zone_k1_k2_without_k0_cannot_correct_position() -> None:
    pose = FieldPose2D(FieldPoint(120.0, -240.0), math.radians(23.0))
    zone = replace(
        _safe_zone_from_pose(pose),
        ground_anchor=FieldPoseKeypoint(None, None, 0.0),
        quality=frozenset({FieldFeatureQuality.KEYPOINT_UNAVAILABLE}),
    )
    result = FieldFeatureDetectionResult(
        1, 1_000_000, 1_100_000, (640, 480), (zone,), None
    )

    assert SafeZoneCornerLocalizer(static_map()).localize(
        result,
        prior_pose=pose,
    ) is None


def test_safe_zone_single_corner_cannot_create_pose() -> None:
    zone = SafeZoneObservation(
        UndistortedBoundingBox(0.0, 0.0, 100.0, 100.0),
        FieldPoseKeypoint(None, None, 0.0),
        FieldPoseKeypoint(UndistortedPixel(10.0, 20.0), GroundPoint(100.0, 200.0), 0.9),
        FieldPoseKeypoint(None, None, 0.0),
        SafeZoneColor.BLUE,
        0.8,
        frozenset({FieldFeatureQuality.KEYPOINT_UNAVAILABLE}),
    )
    result = FieldFeatureDetectionResult(
        2, 2_000_000, 2_100_000, (640, 480), (zone,), None
    )
    assert SafeZoneCornerLocalizer(static_map()).localize(
        result, prior_pose=FieldPose2D(FieldPoint(0.0, 0.0), 0.0)
    ) is None


def test_safe_zone_image_order_hypotheses_use_prior() -> None:
    pose = FieldPose2D(FieldPoint(-80.0, 140.0), -0.3)
    zone = _safe_zone_from_pose(pose, swap_image_order=True)
    result = FieldFeatureDetectionResult(
        3, 3_000_000, 3_100_000, (640, 480), (zone,), None
    )

    observation = SafeZoneCornerLocalizer(static_map()).localize(result, prior_pose=pose)

    assert observation is not None
    assert observation.pose.position.x == pytest.approx(pose.position.x, abs=1e-6)
    assert observation.pose.position.y == pytest.approx(pose.position.y, abs=1e-6)


def test_tracker_hint_carries_prior_uncertainties() -> None:
    tracker = StaticFieldLandmarkTracker(static_map())

    hint = tracker.search_hint(
        FieldPose2D(FieldPoint(0.0, 0.0), 0.0),
        position_uncertainty_mm=40.0,
        heading_uncertainty_rad=math.radians(3.0),
    )

    assert hint is not None
    assert hint.cross_position_uncertainty_mm == pytest.approx(40.0)
    assert hint.cross_heading_uncertainty_rad == pytest.approx(math.radians(3.0))


def test_tracker_hint_radius_expands_with_uncertainty() -> None:
    tracker = StaticFieldLandmarkTracker(static_map())
    pose = FieldPose2D(FieldPoint(-1000.0, -1000.0), 0.0)

    tight = tracker.search_hint(
        pose,
        position_uncertainty_mm=10.0,
        heading_uncertainty_rad=0.0,
    )
    loose = tracker.search_hint(
        pose,
        position_uncertainty_mm=90.0,
        heading_uncertainty_rad=math.radians(5.0),
    )

    assert tight is not None and loose is not None
    assert loose.cross_radius_mm > tight.cross_radius_mm


def test_search_hint_rejects_nonpositive_uncertainty() -> None:
    with pytest.raises(ValueError, match="cross_position_uncertainty_mm"):
        FieldFeatureSearchHint(
            GroundPoint(0.0, 0.0),
            100.0,
            ((1.0, 0.0), (0.0, 1.0)),
            0.0,
            0.1,
            (),
        )
    with pytest.raises(ValueError, match="cross_heading_uncertainty_rad"):
        FieldFeatureSearchHint(
            GroundPoint(0.0, 0.0),
            100.0,
            ((1.0, 0.0), (0.0, 1.0)),
            10.0,
            0.0,
            (),
        )


def test_cross_candidate_confirms_after_two_consistent_frames() -> None:
    tracker = StaticFieldLandmarkTracker(
        static_map(),
        StaticLandmarkTrackingConfig(
            confirmation_hits=2,
            confirmation_ground_tolerance_mm=150.0,
        ),
    )

    first = tracker.confirm_center_cross(
        cross_result(1_000_000_000, GroundPoint(500.0, 300.0))
    )
    second = tracker.confirm_center_cross(
        cross_result(1_100_000_000, GroundPoint(530.0, 280.0))
    )

    assert first.center_cross is not None
    assert first.center_cross.confirmation is CenterCrossConfirmation.CANDIDATE
    assert second.center_cross is not None
    assert (
        second.center_cross.confirmation
        is CenterCrossConfirmation.TEMPORAL_CONFIRMED
    )


def test_cross_confirmation_resets_after_timeout() -> None:
    tracker = StaticFieldLandmarkTracker(
        static_map(),
        StaticLandmarkTrackingConfig(
            confirmation_hits=2,
            max_confirmation_age_ms=150.0,
        ),
    )

    first = tracker.confirm_center_cross(cross_result(0, GroundPoint(0.0, 0.0)))
    gapped = tracker.confirm_center_cross(
        cross_result(400_000_000, GroundPoint(1.0, 0.0))
    )
    third = tracker.confirm_center_cross(
        cross_result(500_000_000, GroundPoint(2.0, 0.0))
    )

    assert first.center_cross is not None
    assert first.center_cross.confirmation is CenterCrossConfirmation.CANDIDATE
    assert gapped.center_cross is not None
    assert gapped.center_cross.confirmation is CenterCrossConfirmation.CANDIDATE
    assert third.center_cross is not None
    assert (
        third.center_cross.confirmation
        is CenterCrossConfirmation.TEMPORAL_CONFIRMED
    )


def test_cross_confirmation_rejects_backwards_frames() -> None:
    tracker = StaticFieldLandmarkTracker(static_map())

    later = tracker.confirm_center_cross(cross_result(1_000_000_000, GroundPoint(0.0, 0.0)))
    earlier = tracker.confirm_center_cross(
        cross_result(900_000_000, GroundPoint(0.0, 0.0))
    )
    follow = tracker.confirm_center_cross(cross_result(1_000_000_000, GroundPoint(0.0, 0.0)))

    assert later.center_cross is not None
    assert later.center_cross.confirmation is CenterCrossConfirmation.CANDIDATE
    assert earlier.center_cross is not None
    assert earlier.center_cross.confirmation is CenterCrossConfirmation.CANDIDATE
    assert follow.center_cross is not None
    assert (
        follow.center_cross.confirmation
        is CenterCrossConfirmation.TEMPORAL_CONFIRMED
    )


def test_cross_confirmation_resets_when_cross_disappears() -> None:
    tracker = StaticFieldLandmarkTracker(static_map())

    tracker.confirm_center_cross(cross_result(0, GroundPoint(0.0, 0.0)))
    tracker.confirm_center_cross(empty_result(50_000_000))
    single = tracker.confirm_center_cross(cross_result(100_000_000, GroundPoint(0.0, 0.0)))

    assert single.center_cross is not None
    assert single.center_cross.confirmation is CenterCrossConfirmation.CANDIDATE


def test_cross_confirmation_applies_motion_limits() -> None:
    config = StaticLandmarkTrackingConfig(confirmation_ground_tolerance_mm=50.0)
    tight = StaticFieldLandmarkTracker(
        static_map(),
        config,
        max_linear_velocity_m_s=0.0,
        max_angular_velocity_rad_s=0.0,
    )
    loose = StaticFieldLandmarkTracker(
        static_map(),
        config,
        max_linear_velocity_m_s=2.0,
        max_angular_velocity_rad_s=0.0,
    )
    first = cross_result(0, GroundPoint(1000.0, 1000.0))
    second = cross_result(100_000_000, GroundPoint(1150.0, 1000.0))

    tight_first = tight.confirm_center_cross(first)
    tight_second = tight.confirm_center_cross(second)
    loose_first = loose.confirm_center_cross(first)
    loose_second = loose.confirm_center_cross(second)

    assert tight_first.center_cross is not None
    assert tight_second.center_cross is not None
    assert (
        tight_second.center_cross.confirmation
        is CenterCrossConfirmation.CANDIDATE
    )
    assert loose_second.center_cross is not None
    assert (
        loose_second.center_cross.confirmation
        is CenterCrossConfirmation.TEMPORAL_CONFIRMED
    )
    del tight_first, loose_first


def test_prior_guided_cross_confirms_immediately_and_resets_pending() -> None:
    tracker = StaticFieldLandmarkTracker(static_map())

    guided = tracker.confirm_center_cross(
        cross_result(
            0,
            GroundPoint(0.0, 0.0),
            confirmation=CenterCrossConfirmation.PRIOR_GUIDED,
        )
    )
    single = tracker.confirm_center_cross(cross_result(50_000_000, GroundPoint(0.0, 0.0)))

    assert guided.center_cross is not None
    assert guided.center_cross.confirmation is CenterCrossConfirmation.PRIOR_GUIDED
    assert single.center_cross is not None
    assert single.center_cross.confirmation is CenterCrossConfirmation.CANDIDATE


def cross_pose_observation(
    pose: FieldPose2D,
    *,
    confidence: float,
    with_pose: bool = True,
) -> CenterCrossPoseObservation:
    candidates = tuple(
        CenterCrossPoseCandidate(
            FieldPose2D(pose.position, pose.heading_rad + index * (math.pi / 2.0)),
            index,
            40.0,
            math.radians(5.0),
        )
        for index in range(4)
    )
    return CenterCrossPoseObservation(
        1,
        1_000_000,
        1_100_000,
        candidates if with_pose else (),
        (),
        candidates[0].pose if with_pose else None,
        CenterCrossSelectionSource.PRIOR if with_pose else None,
        confidence,
        frozenset({CenterCrossLocalizationQuality.NO_DIRECTION_ANCHOR})
        if not with_pose
        else frozenset(),
    )


def corner_pose_observation(
    pose: FieldPose2D,
    *,
    confidence: float,
) -> SafeZoneCornerPoseObservation:
    return SafeZoneCornerPoseObservation(
        1,
        1_000_000,
        1_100_000,
        SafeZoneColor.RED,
        (SafeZoneCornerRole.ENTRANCE_LEFT, SafeZoneCornerRole.BACK_RIGHT),
        pose,
        30.0,
        math.radians(3.0),
        10.0,
        confidence,
    )


def test_same_frame_arbitration_prefers_smaller_prior_innovation() -> None:
    prior = FieldPose2D(FieldPoint(0.0, 0.0), 0.0)
    cross = cross_pose_observation(
        FieldPose2D(FieldPoint(30.0, 0.0), 0.0),
        confidence=0.5,
    )
    far_corner = corner_pose_observation(
        FieldPose2D(FieldPoint(200.0, 0.0), 0.0),
        confidence=0.99,
    )
    near_corner = corner_pose_observation(
        FieldPose2D(FieldPoint(10.0, 0.0), 0.0),
        confidence=0.5,
    )

    kept_cross, kept_corner = select_same_frame_pose_observation(
        cross,
        far_corner,
        prior_pose=prior,
    )
    assert kept_cross is cross
    assert kept_corner is None

    kept_cross, kept_corner = select_same_frame_pose_observation(
        cross,
        near_corner,
        prior_pose=prior,
    )
    assert kept_cross is None
    assert kept_corner is near_corner


def test_same_frame_arbitration_prefers_confidence_without_prior() -> None:
    cross = cross_pose_observation(
        FieldPose2D(FieldPoint(30.0, 0.0), 0.0),
        confidence=0.9,
    )
    weak_corner = corner_pose_observation(
        FieldPose2D(FieldPoint(-500.0, 0.0), 0.0),
        confidence=0.4,
    )
    strong_corner = corner_pose_observation(
        FieldPose2D(FieldPoint(-500.0, 0.0), 0.0),
        confidence=0.95,
    )

    kept_cross, kept_corner = select_same_frame_pose_observation(
        cross,
        weak_corner,
        prior_pose=None,
    )
    assert kept_cross is cross
    assert kept_corner is None

    kept_cross, kept_corner = select_same_frame_pose_observation(
        cross,
        strong_corner,
        prior_pose=None,
    )
    assert kept_cross is None
    assert kept_corner is strong_corner


def test_same_frame_arbitration_skips_poseless_cross() -> None:
    poseless = cross_pose_observation(
        FieldPose2D(FieldPoint(0.0, 0.0), 0.0),
        confidence=0.9,
        with_pose=False,
    )
    corner = corner_pose_observation(
        FieldPose2D(FieldPoint(10.0, 0.0), 0.0),
        confidence=0.3,
    )

    kept_cross, kept_corner = select_same_frame_pose_observation(
        poseless,
        corner,
        prior_pose=None,
    )

    assert kept_cross is None
    assert kept_corner is corner
