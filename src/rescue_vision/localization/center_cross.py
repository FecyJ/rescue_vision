"""Absolute pose observations from the mapped center cross and static anchors."""

from __future__ import annotations

import math

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization.types import (
    CenterCrossLocalizationQuality,
    CenterCrossLocalizerConfig,
    CenterCrossPoseCandidate,
    CenterCrossPoseObservation,
    CenterCrossSelectionSource,
    CenterLineTerminalKind,
    CenterLineTerminalObservation,
    FieldPose2D,
    angular_distance,
    normalize_angle,
)
from rescue_vision.perception.field_feature_types import (
    BoundaryFeatureKind,
    FieldFeatureDetectionResult,
    SafeZoneColor,
)


def _unit(dx: float, dy: float) -> tuple[float, float]:
    length = math.hypot(dx, dy)
    if not math.isfinite(length) or length <= 1e-9:
        raise ValueError("center-cross axis endpoints must be finite and distinct.")
    return dx / length, dy / length


def _axis_directions(
    result: FieldFeatureDetectionResult,
) -> tuple[tuple[float, float], ...]:
    cross = result.center_cross
    assert cross is not None
    directions: list[tuple[float, float]] = []
    for axis in cross.axes:
        assert axis.start_ground is not None
        assert axis.end_ground is not None
        direction = _unit(
            axis.end_ground.x - axis.start_ground.x,
            axis.end_ground.y - axis.start_ground.y,
        )
        directions.extend((direction, (-direction[0], -direction[1])))
    return tuple(directions)


def _ray_metrics(
    origin: GroundPoint,
    direction: tuple[float, float],
    point: GroundPoint,
) -> tuple[float, float, float]:
    dx = point.x - origin.x
    dy = point.y - origin.y
    forward = dx * direction[0] + dy * direction[1]
    lateral = abs(dx * direction[1] - dy * direction[0])
    angle = math.atan2(lateral, max(forward, 1e-12))
    return forward, lateral, angle


def _centroid(points: tuple[GroundPoint, ...]) -> GroundPoint:
    return GroundPoint(
        sum(point.x for point in points) / len(points),
        sum(point.y for point in points) / len(points),
    )


def _segment_ray_distance(
    origin: GroundPoint,
    direction: tuple[float, float],
    start: GroundPoint,
    end: GroundPoint,
) -> float | None:
    sx = end.x - start.x
    sy = end.y - start.y
    denominator = direction[0] * sy - direction[1] * sx
    if math.isclose(denominator, 0.0, abs_tol=1e-9):
        return None
    qx = start.x - origin.x
    qy = start.y - origin.y
    ray_distance = (qx * sy - qy * sx) / denominator
    segment_fraction = (qx * direction[1] - qy * direction[0]) / denominator
    if ray_distance < 0.0 or not 0.0 <= segment_fraction <= 1.0:
        return None
    return ray_distance


class CenterCrossLocalizer:
    """Turn one field-feature frame into center-cross pose candidates."""

    def __init__(
        self,
        config: CenterCrossLocalizerConfig,
        *,
        max_observation_age_ms: float,
    ) -> None:
        if not isinstance(config, CenterCrossLocalizerConfig):
            raise ValueError("config must be a CenterCrossLocalizerConfig.")
        if not config.enabled:
            raise ValueError("CenterCrossLocalizer requires config.enabled=true.")
        age = float(max_observation_age_ms)
        if not math.isfinite(age) or age <= 0.0:
            raise ValueError("max_observation_age_ms must be positive and finite.")
        self._config = config
        self._max_observation_age_ms = age

    def _associated(
        self,
        origin: GroundPoint,
        direction: tuple[float, float],
        point: GroundPoint,
    ) -> tuple[bool, float]:
        forward, lateral, angle = _ray_metrics(origin, direction, point)
        config = self._config
        accepted = (
            config.ray_min_forward_distance_mm
            <= forward
            <= config.ray_max_forward_distance_mm
            and lateral <= config.ray_max_lateral_distance_mm
            and angle <= math.radians(config.ray_angle_tolerance_deg)
        )
        return accepted, forward

    def _terminal_for_direction(
        self,
        result: FieldFeatureDetectionResult,
        origin: GroundPoint,
        direction: tuple[float, float],
    ) -> CenterLineTerminalObservation:
        safe_matches: list[tuple[float, float, CenterLineTerminalKind]] = []
        for zone in result.safe_zones:
            if (
                zone.polygon_ground is None
                or zone.confidence < self._config.min_anchor_confidence
            ):
                continue
            accepted, distance = self._associated(
                origin,
                direction,
                _centroid(zone.polygon_ground),
            )
            if accepted:
                kind = (
                    CenterLineTerminalKind.RED_SAFE_ZONE
                    if zone.physical_color is SafeZoneColor.RED
                    else CenterLineTerminalKind.BLUE_SAFE_ZONE
                )
                safe_matches.append((zone.confidence, distance, kind))

        kinds = {item[2] for item in safe_matches}
        if len(kinds) == 1:
            confidence, distance, kind = max(safe_matches)
            return CenterLineTerminalObservation(
                direction[0], direction[1], kind, distance, confidence
            )
        if len(kinds) > 1:
            return CenterLineTerminalObservation(
                direction[0],
                direction[1],
                CenterLineTerminalKind.UNKNOWN,
                None,
                0.0,
            )

        boundary_matches: list[tuple[float, float]] = []
        for feature in result.boundary_features:
            if (
                feature.points_ground is None
                or feature.confidence < self._config.min_anchor_confidence
            ):
                continue
            if feature.kind is BoundaryFeatureKind.FIELD_CORNER:
                accepted, distance = self._associated(
                    origin,
                    direction,
                    feature.points_ground[0],
                )
                if accepted:
                    boundary_matches.append((feature.confidence, distance))
                continue
            distance = _segment_ray_distance(
                origin,
                direction,
                feature.points_ground[0],
                feature.points_ground[1],
            )
            if (
                distance is not None
                and self._config.ray_min_forward_distance_mm
                <= distance
                <= self._config.ray_max_forward_distance_mm
            ):
                boundary_matches.append((feature.confidence, distance))
        if boundary_matches:
            confidence, distance = max(boundary_matches)
            return CenterLineTerminalObservation(
                direction[0],
                direction[1],
                CenterLineTerminalKind.PLAIN_BOUNDARY,
                distance,
                confidence,
            )
        return CenterLineTerminalObservation(
            direction[0],
            direction[1],
            CenterLineTerminalKind.UNKNOWN,
            None,
            0.0,
        )

    def _candidates(
        self,
        result: FieldFeatureDetectionResult,
    ) -> tuple[CenterCrossPoseCandidate, ...]:
        cross = result.center_cross
        assert cross is not None
        assert cross.intersection_ground is not None
        first_axis = cross.axes[0]
        assert first_axis.start_ground is not None
        assert first_axis.end_ground is not None
        direction = _unit(
            first_axis.end_ground.x - first_axis.start_ground.x,
            first_axis.end_ground.y - first_axis.start_ground.y,
        )
        axis_angle = math.atan2(direction[1], direction[0])
        heading_floor = math.radians(self._config.heading_uncertainty_floor_deg)
        center = cross.intersection_ground
        candidates: list[CenterCrossPoseCandidate] = []
        for quarter_turn in range(4):
            heading = normalize_angle(-axis_angle + quarter_turn * math.pi / 2.0)
            cosine = math.cos(heading)
            sine = math.sin(heading)
            position = FieldPoint(
                -(cosine * center.x - sine * center.y),
                -(sine * center.x + cosine * center.y),
            )
            candidates.append(
                CenterCrossPoseCandidate(
                    FieldPose2D(position, heading),
                    quarter_turn,
                    self._config.position_uncertainty_floor_mm,
                    heading_floor,
                )
            )
        candidates.sort(key=lambda item: item.pose.heading_rad)
        return tuple(
            CenterCrossPoseCandidate(
                item.pose,
                index,
                item.position_uncertainty_mm,
                item.heading_uncertainty_rad,
            )
            for index, item in enumerate(candidates)
        )

    @staticmethod
    def _nearest_candidate(
        candidates: tuple[CenterCrossPoseCandidate, ...],
        heading_rad: float,
    ) -> CenterCrossPoseCandidate:
        return min(
            candidates,
            key=lambda item: angular_distance(item.pose.heading_rad, heading_rad),
        )

    def localize(
        self,
        result: FieldFeatureDetectionResult,
        *,
        prior_pose: FieldPose2D | None = None,
        current_timestamp_ns: int | None = None,
    ) -> CenterCrossPoseObservation:
        if not isinstance(result, FieldFeatureDetectionResult):
            raise ValueError("result must be a FieldFeatureDetectionResult.")
        if prior_pose is not None and not isinstance(prior_pose, FieldPose2D):
            raise ValueError("prior_pose must be a FieldPose2D or None.")
        now = (
            result.result_timestamp_ns
            if current_timestamp_ns is None
            else current_timestamp_ns
        )
        if (
            isinstance(now, bool)
            or not isinstance(now, int)
            or now < result.result_timestamp_ns
        ):
            raise ValueError(
                "current_timestamp_ns must be a non-negative integer not earlier "
                "than result_timestamp_ns."
            )
        quality: set[CenterCrossLocalizationQuality] = set()
        age_ms = (now - result.capture_timestamp_ns) / 1_000_000.0
        if age_ms > self._max_observation_age_ms:
            quality.add(CenterCrossLocalizationQuality.STALE_OBSERVATION)
            return self._empty(result, quality)
        cross = result.center_cross
        if cross is None:
            quality.add(CenterCrossLocalizationQuality.MISSING_CENTER_CROSS)
            return self._empty(result, quality)
        if len(cross.axes) != 2:
            quality.add(CenterCrossLocalizationQuality.PARTIAL_CENTER_CROSS)
            return self._empty(result, quality)
        if (
            cross.intersection_ground is None
            or any(
                axis.start_ground is None or axis.end_ground is None
                for axis in cross.axes
            )
        ):
            quality.add(CenterCrossLocalizationQuality.NO_GROUND_PROJECTION)
            return self._empty(result, quality)

        candidates = self._candidates(result)
        directions = _axis_directions(result)
        terminals = tuple(
            self._terminal_for_direction(
                result,
                cross.intersection_ground,
                direction,
            )
            for direction in directions
        )
        red = [
            item for item in terminals
            if item.kind is CenterLineTerminalKind.RED_SAFE_ZONE
        ]
        blue = [
            item for item in terminals
            if item.kind is CenterLineTerminalKind.BLUE_SAFE_ZONE
        ]
        anchored_headings = [
            normalize_angle(
                math.pi / 2.0
                - math.atan2(item.direction_left, item.direction_forward)
            )
            for item in red
        ] + [
            normalize_angle(
                -math.pi / 2.0
                - math.atan2(item.direction_left, item.direction_forward)
            )
            for item in blue
        ]
        selected: CenterCrossPoseCandidate | None = None
        source: CenterCrossSelectionSource | None = None
        max_innovation = math.radians(
            self._config.max_prior_heading_innovation_deg
        )
        if anchored_headings:
            reference = anchored_headings[0]
            if any(
                angular_distance(reference, heading)
                > math.radians(self._config.ray_angle_tolerance_deg)
                for heading in anchored_headings[1:]
            ):
                quality.add(
                    CenterCrossLocalizationQuality.CONFLICTING_DIRECTION_ANCHORS
                )
            else:
                selected = self._nearest_candidate(candidates, reference)
                if prior_pose is not None and angular_distance(
                    selected.pose.heading_rad,
                    prior_pose.heading_rad,
                ) > max_innovation:
                    quality.add(CenterCrossLocalizationQuality.ANCHOR_PRIOR_CONFLICT)
                    selected = None
                elif red and blue:
                    source = CenterCrossSelectionSource.RED_BLUE_SAFE_ZONES
                elif red:
                    source = CenterCrossSelectionSource.RED_SAFE_ZONE
                else:
                    source = CenterCrossSelectionSource.BLUE_SAFE_ZONE
        elif prior_pose is not None:
            nearest = self._nearest_candidate(candidates, prior_pose.heading_rad)
            if (
                angular_distance(nearest.pose.heading_rad, prior_pose.heading_rad)
                <= max_innovation
            ):
                selected = nearest
                source = CenterCrossSelectionSource.PRIOR
            else:
                quality.add(
                    CenterCrossLocalizationQuality.PRIOR_INNOVATION_REJECTED
                )
        else:
            quality.add(CenterCrossLocalizationQuality.NO_DIRECTION_ANCHOR)

        if not anchored_headings and any(
            item.kind is CenterLineTerminalKind.PLAIN_BOUNDARY for item in terminals
        ):
            quality.add(CenterCrossLocalizationQuality.BOUNDARY_ONLY_AMBIGUITY)
        return CenterCrossPoseObservation(
            frame_sequence=result.frame_sequence,
            capture_timestamp_ns=result.capture_timestamp_ns,
            result_timestamp_ns=result.result_timestamp_ns,
            candidates=candidates,
            terminals=terminals,
            selected_pose=None if selected is None else selected.pose,
            selection_source=source,
            confidence=0.0 if selected is None else cross.confidence,
            quality=frozenset(quality),
        )

    @staticmethod
    def _empty(
        result: FieldFeatureDetectionResult,
        quality: set[CenterCrossLocalizationQuality],
    ) -> CenterCrossPoseObservation:
        return CenterCrossPoseObservation(
            frame_sequence=result.frame_sequence,
            capture_timestamp_ns=result.capture_timestamp_ns,
            result_timestamp_ns=result.result_timestamp_ns,
            candidates=(),
            terminals=(),
            selected_pose=None,
            selection_source=None,
            confidence=0.0,
            quality=frozenset(quality),
        )
