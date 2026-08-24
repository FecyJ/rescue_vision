from __future__ import annotations

import math

import pytest

from rescue_vision.geometry.types import FieldPoint, GroundPoint, UndistortedPixel
from rescue_vision.localization import (
    CenterCrossLocalizationQuality,
    CenterCrossLocalizer,
    CenterCrossLocalizerConfig,
    CenterCrossSelectionSource,
    CenterLineTerminalKind,
    FieldPose2D,
    angular_distance,
)
from rescue_vision.perception import (
    BoundaryFeatureKind,
    BoundaryFeatureObservation,
    CenterCrossObservation,
    FieldFeatureDetectionResult,
    FieldFeatureQuality,
    LineSegmentObservation,
    SafeZoneColor,
    SafeZoneObservation,
)
from rescue_vision.world import (
    CenterCrossRay,
    CenterCrossTerminal,
    StaticCenterCross,
    StaticFieldMap,
    default_static_field_map,
)


def config(**overrides: object) -> CenterCrossLocalizerConfig:
    values: dict[str, object] = {
        "enabled": True,
        "ray_min_forward_distance_mm": 500.0,
        "ray_max_forward_distance_mm": 1800.0,
        "ray_max_lateral_distance_mm": 120.0,
        "ray_angle_tolerance_deg": 10.0,
        "min_anchor_confidence": 0.25,
        "max_prior_heading_innovation_deg": 20.0,
        "position_uncertainty_floor_mm": 20.0,
        "heading_uncertainty_floor_deg": 3.0,
    }
    values.update(overrides)
    return CenterCrossLocalizerConfig(**values)  # type: ignore[arg-type]


def line(
    start: GroundPoint,
    end: GroundPoint,
    *,
    ground: bool = True,
) -> LineSegmentObservation:
    return LineSegmentObservation(
        UndistortedPixel(start.x + 2000.0, start.y + 2000.0),
        UndistortedPixel(end.x + 2000.0, end.y + 2000.0),
        start if ground else None,
        end if ground else None,
    )


def cross(
    *,
    center: GroundPoint = GroundPoint(100.0, -50.0),
    axis_angle_rad: float = 0.0,
    partial: bool = False,
    ground: bool = True,
) -> CenterCrossObservation:
    cosine = math.cos(axis_angle_rad)
    sine = math.sin(axis_angle_rad)

    def endpoint(distance: float, perpendicular: bool = False) -> GroundPoint:
        dx, dy = ((-sine, cosine) if perpendicular else (cosine, sine))
        return GroundPoint(center.x + distance * dx, center.y + distance * dy)

    axes = (line(endpoint(-400.0), endpoint(400.0), ground=ground),)
    if not partial:
        axes += (
            line(
                endpoint(-400.0, True),
                endpoint(400.0, True),
                ground=ground,
            ),
        )
    intersection_pixel = (
        None
        if partial
        else UndistortedPixel(center.x + 2000.0, center.y + 2000.0)
    )
    return CenterCrossObservation(
        axes=axes,
        intersection_undistorted=intersection_pixel,
        intersection_ground=(center if ground and not partial else None),
        confidence=0.8 if not partial else 0.3,
        quality=(
            frozenset()
            if not partial
            else frozenset({FieldFeatureQuality.PARTIAL})
        ),
    )


def safe_zone(
    color: SafeZoneColor,
    center: GroundPoint,
    *,
    confidence: float = 0.9,
    partial: bool = False,
) -> SafeZoneObservation:
    polygon_ground = (
        GroundPoint(center.x - 100.0, center.y - 50.0),
        GroundPoint(center.x + 100.0, center.y - 50.0),
        GroundPoint(center.x + 100.0, center.y + 50.0),
        GroundPoint(center.x - 100.0, center.y + 50.0),
    )
    polygon_pixels = tuple(
        UndistortedPixel(point.x + 2000.0, point.y + 2000.0)
        for point in polygon_ground
    )
    return SafeZoneObservation(
        physical_color=color,
        polygon_undistorted=polygon_pixels,
        polygon_ground=polygon_ground,
        entrance=None,
        divider=None,
        halves=(),
        confidence=confidence,
        quality=frozenset(
            {
                FieldFeatureQuality.ENTRANCE_UNRESOLVED,
                FieldFeatureQuality.DIVIDER_UNRESOLVED,
                FieldFeatureQuality.SIDE_UNRESOLVED,
            }
            | ({FieldFeatureQuality.PARTIAL} if partial else set())
        ),
    )


def boundary(
    start: GroundPoint,
    end: GroundPoint,
    *,
    confidence: float = 0.4,
) -> BoundaryFeatureObservation:
    return BoundaryFeatureObservation(
        kind=BoundaryFeatureKind.FENCE_BASE_SEGMENT,
        points_undistorted=(
            UndistortedPixel(start.x + 2000.0, start.y + 2000.0),
            UndistortedPixel(end.x + 2000.0, end.y + 2000.0),
        ),
        points_ground=(start, end),
        capture_timestamp_ns=1_000_000,
        confidence=confidence,
        quality=frozenset({FieldFeatureQuality.LOW_CONFIDENCE_BOUNDARY}),
    )


def result(
    *,
    center_cross: CenterCrossObservation | None = None,
    safe_zones: tuple[SafeZoneObservation, ...] = (),
    boundaries: tuple[BoundaryFeatureObservation, ...] = (),
) -> FieldFeatureDetectionResult:
    return FieldFeatureDetectionResult(
        frame_sequence=7,
        capture_timestamp_ns=1_000_000,
        result_timestamp_ns=1_100_000,
        image_size=(400, 400),
        safe_zones=safe_zones,
        start_zones=(),
        center_cross=cross() if center_cross is None else center_cross,
        boundary_features=boundaries,
    )


def localizer(
    *,
    static_map: StaticFieldMap | None = None,
    **overrides: object,
) -> CenterCrossLocalizer:
    return CenterCrossLocalizer(
        config(**overrides),
        static_map=(
            default_static_field_map() if static_map is None else static_map
        ),
        max_observation_age_ms=100.0,
    )


def test_center_cross_produces_four_quarter_turn_candidates() -> None:
    observation = localizer().localize(result())

    assert len(observation.candidates) == 4
    headings = [candidate.pose.heading_rad for candidate in observation.candidates]
    assert headings == pytest.approx([-math.pi / 2.0, 0.0, math.pi / 2.0, math.pi])
    assert observation.selected_pose is None
    assert CenterCrossLocalizationQuality.NO_DIRECTION_ANCHOR in observation.quality

    zero = min(
        observation.candidates,
        key=lambda item: angular_distance(item.pose.heading_rad, 0.0),
    )
    assert zero.pose.position == FieldPoint(-100.0, 50.0)


@pytest.mark.parametrize("true_heading", [0.0, 0.37, math.pi - 1e-6, -math.pi + 1e-6])
def test_prior_selects_rotated_candidate_and_preserves_transform(
    true_heading: float,
) -> None:
    center = GroundPoint(120.0, -40.0)
    features = result(
        center_cross=cross(center=center, axis_angle_rad=-true_heading)
    )
    observation = localizer().localize(
        features,
        prior_pose=FieldPose2D(FieldPoint(0.0, 0.0), true_heading + 0.01),
    )

    assert observation.selection_source is CenterCrossSelectionSource.PRIOR
    assert observation.selected_pose is not None
    assert angular_distance(observation.selected_pose.heading_rad, true_heading) < 1e-6
    heading = observation.selected_pose.heading_rad
    expected = FieldPoint(
        -(math.cos(heading) * center.x - math.sin(heading) * center.y),
        -(math.sin(heading) * center.x + math.cos(heading) * center.y),
    )
    assert observation.selected_pose.position.x == pytest.approx(expected.x)
    assert observation.selected_pose.position.y == pytest.approx(expected.y)


def test_red_and_blue_safe_zones_anchor_global_heading() -> None:
    features = result(
        safe_zones=(
            safe_zone(SafeZoneColor.RED, GroundPoint(100.0, 1150.0)),
            safe_zone(SafeZoneColor.BLUE, GroundPoint(100.0, -1250.0)),
        )
    )
    observation = localizer().localize(features)

    assert observation.selected_pose is not None
    assert observation.selected_pose.heading_rad == pytest.approx(0.0)
    assert (
        observation.selection_source
        is CenterCrossSelectionSource.RED_BLUE_SAFE_ZONES
    )
    assert {
        terminal.kind for terminal in observation.terminals
    } >= {
        CenterLineTerminalKind.RED_SAFE_ZONE,
        CenterLineTerminalKind.BLUE_SAFE_ZONE,
    }


def test_static_map_is_the_authority_for_safe_zone_direction() -> None:
    reversed_map = StaticFieldMap(
        StaticCenterCross(
            FieldPoint(0.0, 0.0),
            (
                CenterCrossTerminal(
                    CenterCrossRay.POSITIVE_X,
                    CenterLineTerminalKind.PLAIN_BOUNDARY,
                ),
                CenterCrossTerminal(
                    CenterCrossRay.NEGATIVE_X,
                    CenterLineTerminalKind.PLAIN_BOUNDARY,
                ),
                CenterCrossTerminal(
                    CenterCrossRay.POSITIVE_Y,
                    CenterLineTerminalKind.BLUE_SAFE_ZONE,
                ),
                CenterCrossTerminal(
                    CenterCrossRay.NEGATIVE_Y,
                    CenterLineTerminalKind.RED_SAFE_ZONE,
                ),
            ),
        ),
        (),
    )
    observation = localizer(static_map=reversed_map).localize(
        result(
            safe_zones=(
                safe_zone(SafeZoneColor.RED, GroundPoint(100.0, 1150.0)),
            )
        )
    )

    assert observation.selected_pose is not None
    assert angular_distance(observation.selected_pose.heading_rad, math.pi) < 1e-9


@pytest.mark.parametrize(
    ("color", "zone_center", "source"),
    [
        (
            SafeZoneColor.RED,
            GroundPoint(100.0, 1150.0),
            CenterCrossSelectionSource.RED_SAFE_ZONE,
        ),
        (
            SafeZoneColor.BLUE,
            GroundPoint(100.0, -1250.0),
            CenterCrossSelectionSource.BLUE_SAFE_ZONE,
        ),
    ],
)
def test_one_colored_safe_zone_is_enough_for_unique_heading(
    color: SafeZoneColor,
    zone_center: GroundPoint,
    source: CenterCrossSelectionSource,
) -> None:
    observation = localizer().localize(
        result(safe_zones=(safe_zone(color, zone_center),))
    )

    assert observation.selected_pose is not None
    assert observation.selected_pose.heading_rad == pytest.approx(0.0)
    assert observation.selection_source is source


def test_partial_blue_zone_still_requires_and_can_pass_ray_anchor_gate() -> None:
    observation = localizer().localize(
        result(
            safe_zones=(
                safe_zone(
                    SafeZoneColor.BLUE,
                    GroundPoint(100.0, -1250.0),
                    confidence=0.388,
                    partial=True,
                ),
            )
        )
    )

    assert observation.selected_pose is not None
    assert observation.selection_source is CenterCrossSelectionSource.BLUE_SAFE_ZONE


def test_plain_boundary_does_not_resolve_180_degree_ambiguity() -> None:
    observation = localizer().localize(
        result(
            boundaries=(
                boundary(GroundPoint(1300.0, -500.0), GroundPoint(1300.0, 500.0)),
            )
        )
    )

    assert observation.selected_pose is None
    assert any(
        item.kind is CenterLineTerminalKind.PLAIN_BOUNDARY
        for item in observation.terminals
    )
    assert (
        CenterCrossLocalizationQuality.BOUNDARY_ONLY_AMBIGUITY
        in observation.quality
    )


def test_off_axis_and_low_confidence_regions_do_not_anchor() -> None:
    observation = localizer().localize(
        result(
            safe_zones=(
                safe_zone(SafeZoneColor.RED, GroundPoint(900.0, 900.0)),
                safe_zone(
                    SafeZoneColor.BLUE,
                    GroundPoint(100.0, -1250.0),
                    confidence=0.1,
                ),
            )
        )
    )

    assert observation.selected_pose is None
    assert all(
        item.kind is not CenterLineTerminalKind.RED_SAFE_ZONE
        and item.kind is not CenterLineTerminalKind.BLUE_SAFE_ZONE
        for item in observation.terminals
    )


def test_conflicting_safe_zone_directions_are_rejected() -> None:
    observation = localizer().localize(
        result(
            safe_zones=(
                safe_zone(SafeZoneColor.RED, GroundPoint(100.0, 1150.0)),
                safe_zone(SafeZoneColor.BLUE, GroundPoint(1300.0, -50.0)),
            )
        )
    )

    assert observation.selected_pose is None
    assert (
        CenterCrossLocalizationQuality.CONFLICTING_DIRECTION_ANCHORS
        in observation.quality
    )


def test_prior_innovation_and_anchor_prior_conflict_are_explicit() -> None:
    rejected_prior = localizer(max_prior_heading_innovation_deg=5.0).localize(
        result(),
        prior_pose=FieldPose2D(FieldPoint(0.0, 0.0), math.radians(30.0)),
    )
    assert rejected_prior.selected_pose is None
    assert (
        CenterCrossLocalizationQuality.PRIOR_INNOVATION_REJECTED
        in rejected_prior.quality
    )

    anchor_conflict = localizer(max_prior_heading_innovation_deg=10.0).localize(
        result(
            safe_zones=(
                safe_zone(SafeZoneColor.RED, GroundPoint(100.0, 1150.0)),
            )
        ),
        prior_pose=FieldPose2D(FieldPoint(0.0, 0.0), math.pi),
    )
    assert anchor_conflict.selected_pose is None
    assert (
        CenterCrossLocalizationQuality.ANCHOR_PRIOR_CONFLICT
        in anchor_conflict.quality
    )


@pytest.mark.parametrize(
    ("features", "quality"),
    [
        (
            result(center_cross=cross(partial=True)),
            CenterCrossLocalizationQuality.PARTIAL_CENTER_CROSS,
        ),
        (
            result(center_cross=cross(ground=False)),
            CenterCrossLocalizationQuality.NO_GROUND_PROJECTION,
        ),
    ],
)
def test_incomplete_crosses_do_not_produce_pose_candidates(
    features: FieldFeatureDetectionResult,
    quality: CenterCrossLocalizationQuality,
) -> None:
    observation = localizer().localize(features)
    assert observation.candidates == ()
    assert quality in observation.quality


def test_missing_and_stale_center_observations_are_explicit() -> None:
    missing = FieldFeatureDetectionResult(
        frame_sequence=7,
        capture_timestamp_ns=1_000_000,
        result_timestamp_ns=1_100_000,
        image_size=(400, 400),
        safe_zones=(),
        start_zones=(),
        center_cross=None,
        boundary_features=(),
    )
    missing_observation = localizer().localize(missing)
    assert (
        CenterCrossLocalizationQuality.MISSING_CENTER_CROSS
        in missing_observation.quality
    )

    stale = localizer().localize(
        result(),
        current_timestamp_ns=102_000_000,
    )
    assert stale.candidates == ()
    assert CenterCrossLocalizationQuality.STALE_OBSERVATION in stale.quality


def test_config_and_pose_reject_non_finite_values() -> None:
    with pytest.raises(ValueError, match="finite"):
        FieldPose2D(FieldPoint(0.0, 0.0), float("nan"))
    with pytest.raises(ValueError, match="must exceed"):
        config(
            ray_min_forward_distance_mm=1000.0,
            ray_max_forward_distance_mm=1000.0,
        )
