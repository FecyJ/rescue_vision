"""Static-map search hints, short-lived landmark tracks and safe-zone poses."""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
import math

import numpy as np

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization.types import (
    CenterCrossPoseObservation,
    FieldPose2D,
    angular_distance,
    normalize_angle,
)
from rescue_vision.perception.field_feature_types import (
    FieldFeatureDetectionResult,
    FieldFeatureSearchHint,
    CenterCrossConfirmation,
    SafeZoneColor,
    SafeZoneCornerRole,
    SafeZoneSearchRegion,
)
from rescue_vision.world.static_map import StaticFieldMap, TeamColor


@dataclass(frozen=True, slots=True)
class StaticLandmarkTrackingConfig:
    max_track_age_ms: float = 250.0
    max_prior_position_uncertainty_mm: float = 300.0
    max_prior_heading_uncertainty_deg: float = 15.0
    cross_base_radius_mm: float = 180.0
    safe_zone_base_margin_mm: float = 120.0
    confirmation_hits: int = 2
    max_confirmation_age_ms: float = 150.0
    confirmation_ground_tolerance_mm: float = 120.0
    confirmation_axis_tolerance_deg: float = 20.0

    def __post_init__(self) -> None:
        for name in (
            "max_track_age_ms",
            "max_prior_position_uncertainty_mm",
            "max_prior_heading_uncertainty_deg",
            "cross_base_radius_mm",
            "safe_zone_base_margin_mm",
            "max_confirmation_age_ms",
            "confirmation_ground_tolerance_mm",
            "confirmation_axis_tolerance_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if (
            isinstance(self.confirmation_hits, bool)
            or not isinstance(self.confirmation_hits, int)
            or self.confirmation_hits < 2
        ):
            raise ValueError("confirmation_hits must be an integer >= 2.")


@dataclass(frozen=True, slots=True)
class StaticLandmarkTrack:
    landmark_id: str
    last_seen_timestamp_ns: int
    confidence: float
    matching_residual_mm: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.landmark_id, str) or not self.landmark_id:
            raise ValueError("landmark_id must be non-empty.")
        if self.last_seen_timestamp_ns < 0:
            raise ValueError("last_seen_timestamp_ns must be non-negative.")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1].")
        if self.matching_residual_mm is not None and (
            not math.isfinite(self.matching_residual_mm)
            or self.matching_residual_mm < 0.0
        ):
            raise ValueError("matching_residual_mm must be non-negative and finite.")


def _field_to_ground(pose: FieldPose2D, point: FieldPoint) -> GroundPoint:
    dx = point.x - pose.position.x
    dy = point.y - pose.position.y
    cosine = math.cos(pose.heading_rad)
    sine = math.sin(pose.heading_rad)
    return GroundPoint(
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
    )


class StaticFieldLandmarkTracker:
    """Keep only bounded evidence and build recoverable map-based search hints."""

    def __init__(
        self,
        static_map: StaticFieldMap,
        config: StaticLandmarkTrackingConfig | None = None,
        *,
        max_linear_velocity_m_s: float = 0.25,
        max_angular_velocity_rad_s: float = 1.0,
    ) -> None:
        if not isinstance(static_map, StaticFieldMap):
            raise TypeError("static_map must be a StaticFieldMap.")
        self._static_map = static_map
        self.config = config or StaticLandmarkTrackingConfig()
        self._max_linear_velocity_m_s = float(max_linear_velocity_m_s)
        self._max_angular_velocity_rad_s = float(max_angular_velocity_rad_s)
        if (
            not math.isfinite(self._max_linear_velocity_m_s)
            or self._max_linear_velocity_m_s < 0.0
            or not math.isfinite(self._max_angular_velocity_rad_s)
            or self._max_angular_velocity_rad_s < 0.0
        ):
            raise ValueError("motion limits must be non-negative and finite.")
        self._tracks: dict[str, StaticLandmarkTrack] = {}
        self._pending_cross: tuple[
            int,
            GroundPoint,
            tuple[float, float],
            int,
        ] | None = None

    def tracks(self, current_timestamp_ns: int) -> tuple[StaticLandmarkTrack, ...]:
        maximum_age_ns = round(self.config.max_track_age_ms * 1_000_000)
        return tuple(
            track
            for track in self._tracks.values()
            if 0 <= current_timestamp_ns - track.last_seen_timestamp_ns <= maximum_age_ns
        )

    def search_hint(
        self,
        pose: FieldPose2D | None,
        *,
        position_uncertainty_mm: float | None,
        heading_uncertainty_rad: float | None,
    ) -> FieldFeatureSearchHint | None:
        config = self.config
        if (
            pose is None
            or position_uncertainty_mm is None
            or heading_uncertainty_rad is None
            or position_uncertainty_mm > config.max_prior_position_uncertainty_mm
            or heading_uncertainty_rad
            > math.radians(config.max_prior_heading_uncertainty_deg)
        ):
            return None
        cross_ground = _field_to_ground(
            pose,
            self._static_map.center_cross.intersection_field,
        )
        three_sigma_position = 3.0 * position_uncertainty_mm
        cross_range = math.hypot(cross_ground.x, cross_ground.y)
        heading_margin = cross_range * math.sin(
            min(math.pi / 2.0, 3.0 * heading_uncertainty_rad)
        )
        cross_radius = (
            config.cross_base_radius_mm
            + three_sigma_position
            + heading_margin
        )
        regions: list[SafeZoneSearchRegion] = []
        for color, team_color in (
            (SafeZoneColor.RED, TeamColor.RED),
            (SafeZoneColor.BLUE, TeamColor.BLUE),
        ):
            polygon = self._static_map.safe_zone_polygon_field(team_color)
            if polygon is None:
                continue
            ground_polygon = tuple(_field_to_ground(pose, point) for point in polygon)
            maximum_range = max(math.hypot(point.x, point.y) for point in ground_polygon)
            margin = (
                config.safe_zone_base_margin_mm
                + three_sigma_position
                + maximum_range
                * math.sin(min(math.pi / 2.0, 3.0 * heading_uncertainty_rad))
            )
            regions.append(SafeZoneSearchRegion(color, ground_polygon, margin))
        cosine = math.cos(pose.heading_rad)
        sine = math.sin(pose.heading_rad)
        axis_directions_ground = (
            (cosine, -sine),
            (sine, cosine),
        )
        return FieldFeatureSearchHint(
            cross_ground,
            cross_radius,
            axis_directions_ground,
            # 观测不确定度不会真正为零，但先验可能声称完全确定；钳到正值
            # 保持 hint 契约，同时不放大 ROI。
            max(position_uncertainty_mm, 1e-3),
            max(heading_uncertainty_rad, 1e-6),
            tuple(regions),
        )

    def update(
        self,
        result: FieldFeatureDetectionResult,
        *,
        pose: FieldPose2D | None = None,
    ) -> None:
        if not isinstance(result, FieldFeatureDetectionResult):
            raise TypeError("result must be a FieldFeatureDetectionResult.")
        seen = False
        cross = result.center_cross
        if cross is not None and len(cross.axes) == 2:
            residual = None
            if pose is not None and cross.intersection_ground is not None:
                expected = _field_to_ground(
                    pose,
                    self._static_map.center_cross.intersection_field,
                )
                residual = math.hypot(
                    cross.intersection_ground.x - expected.x,
                    cross.intersection_ground.y - expected.y,
                )
            self._tracks["center_cross"] = StaticLandmarkTrack(
                "center_cross",
                result.capture_timestamp_ns,
                cross.confidence,
                residual,
            )
            seen = True
        for zone in result.safe_zones:
            if zone.physical_color is SafeZoneColor.UNKNOWN:
                continue
            landmark_id = f"{zone.physical_color.value}_safe_zone"
            residual = None
            team_color = (
                TeamColor.RED
                if zone.physical_color is SafeZoneColor.RED
                else TeamColor.BLUE
            )
            field_polygon = self._static_map.safe_zone_polygon_field(team_color)
            if (
                pose is not None
                and field_polygon is not None
                and zone.ground_anchor.ground is not None
            ):
                expected_field = FieldPoint(
                    sum(point.x for point in field_polygon) / len(field_polygon),
                    sum(point.y for point in field_polygon) / len(field_polygon),
                )
                expected = _field_to_ground(pose, expected_field)
                observed = zone.ground_anchor.ground
                residual = math.hypot(
                    observed.x - expected.x,
                    observed.y - expected.y,
                )
            self._tracks[landmark_id] = StaticLandmarkTrack(
                landmark_id,
                result.capture_timestamp_ns,
                zone.confidence,
                residual,
            )
            seen = True
        del seen

    @staticmethod
    def _cross_axis_angles(
        result: FieldFeatureDetectionResult,
    ) -> tuple[float, float] | None:
        cross = result.center_cross
        if (
            cross is None
            or len(cross.axes) != 2
            or any(
                axis.start_ground is None or axis.end_ground is None
                for axis in cross.axes
            )
        ):
            return None
        angles = tuple(
            math.atan2(
                axis.end_ground.y - axis.start_ground.y,
                axis.end_ground.x - axis.start_ground.x,
            )
            % math.pi
            for axis in cross.axes
        )
        return (angles[0], angles[1])

    @staticmethod
    def _axis_set_difference(
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        def difference(a: float, b: float) -> float:
            raw = abs(a - b) % math.pi
            return min(raw, math.pi - raw)

        return min(
            max(difference(first[0], second[0]), difference(first[1], second[1])),
            max(difference(first[0], second[1]), difference(first[1], second[0])),
        )

    def confirm_center_cross(
        self,
        result: FieldFeatureDetectionResult,
    ) -> FieldFeatureDetectionResult:
        """Confirm a no-prior candidate across bounded consecutive frames."""

        cross = result.center_cross
        if (
            cross is None
            or len(cross.axes) != 2
            or cross.intersection_ground is None
        ):
            self._pending_cross = None
            return result
        if cross.confirmation is CenterCrossConfirmation.PRIOR_GUIDED:
            self._pending_cross = None
            return result
        angles = self._cross_axis_angles(result)
        assert angles is not None
        pending = self._pending_cross
        hits = 1
        if pending is not None:
            previous_timestamp_ns, previous_point, previous_angles, previous_hits = pending
            dt_s = (result.capture_timestamp_ns - previous_timestamp_ns) / 1_000_000_000.0
            if (
                0.0 < dt_s <= self.config.max_confirmation_age_ms / 1000.0
            ):
                range_mm = max(
                    math.hypot(previous_point.x, previous_point.y),
                    math.hypot(
                        cross.intersection_ground.x,
                        cross.intersection_ground.y,
                    ),
                )
                allowed_position = (
                    self.config.confirmation_ground_tolerance_mm
                    + self._max_linear_velocity_m_s * 1000.0 * dt_s
                    + range_mm
                    * math.sin(
                        min(
                            math.pi / 2.0,
                            self._max_angular_velocity_rad_s * dt_s,
                        )
                    )
                )
                point_delta = math.hypot(
                    cross.intersection_ground.x - previous_point.x,
                    cross.intersection_ground.y - previous_point.y,
                )
                allowed_axis = math.radians(
                    self.config.confirmation_axis_tolerance_deg
                ) + self._max_angular_velocity_rad_s * dt_s
                if (
                    point_delta <= allowed_position
                    and self._axis_set_difference(angles, previous_angles)
                    <= allowed_axis
                ):
                    hits = previous_hits + 1
        self._pending_cross = (
            result.capture_timestamp_ns,
            cross.intersection_ground,
            angles,
            hits,
        )
        if hits < self.config.confirmation_hits:
            return result
        confirmed = replace(
            cross,
            confirmation=CenterCrossConfirmation.TEMPORAL_CONFIRMED,
        )
        return replace(result, center_cross=confirmed)


@dataclass(frozen=True, slots=True)
class SafeZoneCornerLocalizerConfig:
    max_observation_age_ms: float = 250.0
    min_baseline_mm: float = 150.0
    max_k0_corner_distance_error_mm: float = 80.0
    max_fit_residual_mm: float = 80.0
    position_uncertainty_floor_mm: float = 30.0
    heading_uncertainty_floor_deg: float = 3.0

    def __post_init__(self) -> None:
        for name in (
            "max_observation_age_ms",
            "min_baseline_mm",
            "max_k0_corner_distance_error_mm",
            "max_fit_residual_mm",
            "position_uncertainty_floor_mm",
            "heading_uncertainty_floor_deg",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")


@dataclass(frozen=True, slots=True)
class SafeZoneCornerPoseObservation:
    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    physical_color: SafeZoneColor
    used_roles: tuple[SafeZoneCornerRole, ...]
    pose: FieldPose2D
    position_uncertainty_mm: float
    heading_uncertainty_rad: float
    fit_residual_mm: float
    confidence: float

    def __post_init__(self) -> None:
        for name in (
            "frame_sequence",
            "capture_timestamp_ns",
            "result_timestamp_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if not isinstance(self.physical_color, SafeZoneColor):
            raise TypeError("physical_color must be a SafeZoneColor.")
        if not all(isinstance(role, SafeZoneCornerRole) for role in self.used_roles):
            raise TypeError("used_roles must contain SafeZoneCornerRole values.")
        if len(self.used_roles) < 2 or len(set(self.used_roles)) != len(self.used_roles):
            raise ValueError("used_roles must contain at least two unique roles.")
        if self.result_timestamp_ns < self.capture_timestamp_ns:
            raise ValueError("result timestamp must not precede capture timestamp.")
        if not isinstance(self.pose, FieldPose2D):
            raise TypeError("pose must be a FieldPose2D.")
        for name in (
            "position_uncertainty_mm",
            "heading_uncertainty_rad",
            "confidence",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite.")
        if not 0.0 < self.confidence <= 1.0:
            raise ValueError("confidence must be in (0, 1].")
        if not math.isfinite(self.fit_residual_mm) or self.fit_residual_mm < 0.0:
            raise ValueError("fit_residual_mm must be non-negative and finite.")

    @property
    def source(self) -> str:
        return f"{self.physical_color.value}_safe_zone_corners"


class SafeZoneCornerRejectionReason(str, Enum):
    """安全区角点定位被拒的原因；用于区分观测年龄门与三点几何门。"""

    OBSERVATION_STALE = "observation_stale"
    NO_PRIOR_POSE = "no_prior_pose"
    ENTRANCE_CORNERS_MISSING = "entrance_corners_missing"
    BASELINE_TOO_SHORT = "baseline_too_short"
    LANDMARKS_UNUSABLE = "landmarks_unusable"
    INSUFFICIENT_CORRESPONDENCES = "insufficient_correspondences"
    GROUND_ANCHOR_MISSING = "ground_anchor_missing"
    K0_CORNER_DISTANCE_MISMATCH = "k0_corner_distance_mismatch"
    HEADING_UNDEFINED = "heading_undefined"
    FIT_RESIDUAL_TOO_LARGE = "fit_residual_too_large"
    NO_POSE_CANDIDATE = "no_pose_candidate"
    AMBIGUOUS_POSE_CANDIDATE = "ambiguous_pose_candidate"


@dataclass(frozen=True, slots=True)
class SafeZoneCornerLocalization:
    """一次安全区角点定位的结果；观测与拒绝原因恰有其一。"""

    observation: SafeZoneCornerPoseObservation | None = None
    rejection: SafeZoneCornerRejectionReason | None = None

    def __post_init__(self) -> None:
        if (self.observation is None) == (self.rejection is None):
            raise ValueError("exactly one of observation and rejection must be set.")
        if self.observation is not None and not isinstance(
            self.observation, SafeZoneCornerPoseObservation
        ):
            raise TypeError(
                "observation must be a SafeZoneCornerPoseObservation."
            )
        if self.rejection is not None and not isinstance(
            self.rejection, SafeZoneCornerRejectionReason
        ):
            raise TypeError("rejection must be a SafeZoneCornerRejectionReason.")


def _rejected(reason: SafeZoneCornerRejectionReason) -> SafeZoneCornerLocalization:
    return SafeZoneCornerLocalization(rejection=reason)


def _point_distance(
    first: GroundPoint | FieldPoint,
    second: GroundPoint | FieldPoint,
) -> float:
    return math.hypot(first.x - second.x, first.y - second.y)


def _line_rotation(
    ground_first: GroundPoint,
    ground_second: GroundPoint,
    field_first: FieldPoint,
    field_second: FieldPoint,
) -> float:
    ground_angle = math.atan2(
        ground_second.y - ground_first.y,
        ground_second.x - ground_first.x,
    )
    field_angle = math.atan2(
        field_second.y - field_first.y,
        field_second.x - field_first.x,
    )
    return normalize_angle(field_angle - ground_angle)


def _mean_angles(angles: list[float]) -> float | None:
    if not angles:
        return None
    sine = sum(math.sin(angle) for angle in angles)
    cosine = sum(math.cos(angle) for angle in angles)
    if math.hypot(sine, cosine) <= 1e-9:
        return None
    return normalize_angle(math.atan2(sine, cosine))


class SafeZoneCornerLocalizer:
    """枚举 v3 安全区身份和图像左右角点对应，并用先验选择唯一位姿。"""

    def __init__(
        self,
        static_map: StaticFieldMap,
        config: SafeZoneCornerLocalizerConfig | None = None,
    ) -> None:
        if not isinstance(static_map, StaticFieldMap):
            raise TypeError("static_map must be a StaticFieldMap.")
        self._static_map = static_map
        self.config = config or SafeZoneCornerLocalizerConfig()

    def localize(
        self,
        result: FieldFeatureDetectionResult,
        *,
        prior_pose: FieldPose2D | None = None,
        current_timestamp_ns: int | None = None,
    ) -> SafeZoneCornerLocalization:
        """返回观测或拒绝原因；两者恰有其一，调用方必须区分。"""

        now = result.result_timestamp_ns if current_timestamp_ns is None else current_timestamp_ns
        if now < result.result_timestamp_ns:
            raise ValueError("current_timestamp_ns must not precede result timestamp.")
        if (
            now - result.capture_timestamp_ns
            > round(self.config.max_observation_age_ms * 1_000_000)
        ):
            return _rejected(SafeZoneCornerRejectionReason.OBSERVATION_STALE)
        if prior_pose is None:
            # K1/K2 是图像左右语义；对称近场线在无先验时至少保留 180° 歧义。
            return _rejected(SafeZoneCornerRejectionReason.NO_PRIOR_POSE)
        candidates: list[SafeZoneCornerPoseObservation] = []
        first_rejection: SafeZoneCornerRejectionReason | None = None
        for zone in result.safe_zones:
            left = zone.image_left_landmark
            right = zone.image_right_landmark
            if left.ground is None and right.ground is None:
                first_rejection = first_rejection or (
                    SafeZoneCornerRejectionReason.ENTRANCE_CORNERS_MISSING
                )
                continue
            if (
                left.ground is not None
                and right.ground is not None
                and _point_distance(left.ground, right.ground)
                < self.config.min_baseline_mm
            ):
                first_rejection = first_rejection or (
                    SafeZoneCornerRejectionReason.BASELINE_TOO_SHORT
                )
                continue
            colors = (
                (zone.physical_color,)
                if zone.physical_color is not SafeZoneColor.UNKNOWN
                else (SafeZoneColor.RED, SafeZoneColor.BLUE)
            )
            for color in colors:
                team_color = TeamColor.RED if color is SafeZoneColor.RED else TeamColor.BLUE
                landmarks = self._static_map.safe_zone_landmarks_for(team_color)
                if landmarks is None or not landmarks.measured or not landmarks.usable:
                    first_rejection = first_rejection or (
                        SafeZoneCornerRejectionReason.LANDMARKS_UNUSABLE
                    )
                    continue
                for swap in (False, True):
                    world_left, world_right = (
                        (landmarks.near_field_corner_b, landmarks.near_field_corner_a)
                        if swap
                        else (landmarks.near_field_corner_a, landmarks.near_field_corner_b)
                    )
                    correspondences: list[
                        tuple[GroundPoint, FieldPoint, SafeZoneCornerRole, float]
                    ] = []
                    if zone.ground_anchor.ground is not None:
                        correspondences.append(
                            (
                                zone.ground_anchor.ground,
                                landmarks.ground_anchor_field,
                                SafeZoneCornerRole.GROUND_ANCHOR,
                                zone.ground_anchor.confidence,
                            )
                        )
                    correspondences.extend(
                        (
                            ground_point,
                            field_point,
                            role,
                            confidence,
                        )
                        for ground_point, field_point, role, confidence in (
                            (
                                left.ground,
                                world_left,
                                (
                                    SafeZoneCornerRole.ENTRANCE_RIGHT
                                    if swap
                                    else SafeZoneCornerRole.ENTRANCE_LEFT
                                ),
                                left.confidence,
                            ),
                            (
                                right.ground,
                                world_right,
                                (
                                    SafeZoneCornerRole.ENTRANCE_LEFT
                                    if swap
                                    else SafeZoneCornerRole.ENTRANCE_RIGHT
                                ),
                                right.confidence,
                            ),
                        )
                        if ground_point is not None
                    )
                    if len(correspondences) < 2:
                        first_rejection = first_rejection or (
                            SafeZoneCornerRejectionReason.INSUFFICIENT_CORRESPONDENCES
                        )
                        continue

                    anchor_entry = next(
                        (
                            item
                            for item in correspondences
                            if item[2] is SafeZoneCornerRole.GROUND_ANCHOR
                        ),
                        None,
                    )
                    if anchor_entry is None:
                        # Position correction must include K0. K1/K2 alone
                        # may define a line, but cannot pass the requested
                        # K0-to-corner metric consistency check.
                        first_rejection = first_rejection or (
                            SafeZoneCornerRejectionReason.GROUND_ANCHOR_MISSING
                        )
                        continue
                    anchor_ground, anchor_field, _anchor_role, _anchor_confidence = (
                        anchor_entry
                    )
                    matching_corners = []
                    for ground_point, field_point, role, _confidence in correspondences:
                        if role is SafeZoneCornerRole.GROUND_ANCHOR:
                            continue
                        observed_distance = _point_distance(anchor_ground, ground_point)
                        expected_distance = _point_distance(anchor_field, field_point)
                        if (
                            abs(observed_distance - expected_distance)
                            <= self.config.max_k0_corner_distance_error_mm
                        ):
                            matching_corners.append(
                                (ground_point, field_point, role, _confidence)
                            )
                    if not matching_corners:
                        first_rejection = first_rejection or (
                            SafeZoneCornerRejectionReason.K0_CORNER_DISTANCE_MISMATCH
                        )
                        continue
                    selected_correspondences = (
                        correspondences
                        if len(matching_corners) == 2
                        else [anchor_entry, matching_corners[0]]
                    )

                    line_angles = [
                        _line_rotation(
                            first_ground,
                            second_ground,
                            first_field,
                            second_field,
                        )
                        for (
                            first_ground,
                            first_field,
                            _first_role,
                            _first_confidence,
                        ), (
                            second_ground,
                            second_field,
                            _second_role,
                            _second_confidence,
                        ) in combinations(selected_correspondences, 2)
                    ]
                    heading = _mean_angles(line_angles)
                    if heading is None:
                        first_rejection = first_rejection or (
                            SafeZoneCornerRejectionReason.HEADING_UNDEFINED
                        )
                        continue
                    cosine = math.cos(heading)
                    sine = math.sin(heading)
                    ground_mean = np.mean(
                        np.asarray(
                            [
                                (point.x, point.y)
                                for point, _field, _role, _confidence
                                in selected_correspondences
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                    field_mean = np.mean(
                        np.asarray(
                            [
                                (point.x, point.y)
                                for _ground, point, _role, _confidence
                                in selected_correspondences
                            ],
                            dtype=np.float64,
                        ),
                        axis=0,
                    )
                    translation = field_mean - np.asarray(
                        (
                            cosine * ground_mean[0] - sine * ground_mean[1],
                            sine * ground_mean[0] + cosine * ground_mean[1],
                        ),
                        dtype=np.float64,
                    )
                    predicted = np.asarray(
                        [
                            (
                                cosine * ground_point.x
                                - sine * ground_point.y
                                + translation[0],
                                sine * ground_point.x
                                + cosine * ground_point.y
                                + translation[1],
                            )
                            for ground_point, _field_point, _role, _confidence
                            in selected_correspondences
                        ],
                        dtype=np.float64,
                    )
                    expected = np.asarray(
                        [
                            (field_point.x, field_point.y)
                            for _ground_point, field_point, _role, _confidence
                            in selected_correspondences
                        ],
                        dtype=np.float64,
                    )
                    point_errors = np.linalg.norm(predicted - expected, axis=1)
                    residual = float(np.sqrt(np.mean(point_errors**2)))
                    if residual > self.config.max_fit_residual_mm:
                        first_rejection = first_rejection or (
                            SafeZoneCornerRejectionReason.FIT_RESIDUAL_TOO_LARGE
                        )
                        continue
                    used_roles = tuple(
                        role for _ground_point, _field_point, role, _confidence
                        in selected_correspondences
                    )
                    confidences = [
                        confidence
                        for _ground_point, _field_point, _role, confidence
                        in selected_correspondences
                    ]
                    confidence = min(
                        1.0,
                        zone.confidence * float(np.mean(confidences))
                        * max(0.1, 1.0 - residual / self.config.max_fit_residual_mm),
                    )
                    candidates.append(SafeZoneCornerPoseObservation(
                        result.frame_sequence,
                        result.capture_timestamp_ns,
                        result.result_timestamp_ns,
                        color,
                        used_roles,
                        FieldPose2D(
                            FieldPoint(float(translation[0]), float(translation[1])),
                            heading,
                        ),
                        max(self.config.position_uncertainty_floor_mm, residual),
                        math.radians(self.config.heading_uncertainty_floor_deg),
                        residual,
                        max(0.01, confidence),
                    ))
        ranked = sorted(
            [
                (
                    _innovation(
                        item.pose,
                        prior_pose,
                        position_uncertainty_mm=item.position_uncertainty_mm,
                        heading_uncertainty_rad=item.heading_uncertainty_rad,
                    ),
                    item,
                )
                for item in candidates
            ],
            key=lambda pair: pair[0],
        )
        if not ranked:
            # 循环内首个命中的原因比重采均值本身更能指示现场问题；没有任何
            # 候选时才回落到聚合原因。
            return _rejected(
                first_rejection
                or SafeZoneCornerRejectionReason.NO_POSE_CANDIDATE
            )
        if len(ranked) > 1 and math.isclose(
            ranked[0][0], ranked[1][0], rel_tol=0.0, abs_tol=1e-6
        ):
            return _rejected(
                SafeZoneCornerRejectionReason.AMBIGUOUS_POSE_CANDIDATE
            )
        return SafeZoneCornerLocalization(observation=ranked[0][1])


def _innovation(
    pose: FieldPose2D,
    prior: FieldPose2D,
    *,
    position_uncertainty_mm: float,
    heading_uncertainty_rad: float,
) -> float:
    position_delta = math.hypot(
        pose.position.x - prior.position.x,
        pose.position.y - prior.position.y,
    )
    heading_delta = angular_distance(pose.heading_rad, prior.heading_rad)
    return (
        position_delta / max(position_uncertainty_mm, 1e-6)
        + heading_delta / max(heading_uncertainty_rad, 1e-6)
    )


def select_same_frame_pose_observation(
    cross: CenterCrossPoseObservation | None,
    corner: SafeZoneCornerPoseObservation | None,
    *,
    prior_pose: FieldPose2D | None,
) -> tuple[CenterCrossPoseObservation | None, SafeZoneCornerPoseObservation | None]:
    """同帧同时产生十字与安全区角点绝对位姿时只保留一项提交。

    两类观测来自同一幅 BEV 的相关视觉证据，融合器重复使用会二次压缩方差。
    有先验时比较归一化位置/航向创新，无先验时比较置信度；并列时取置信度
    更高者，仍并列时保留十字。未选出绝对位姿的十字观测不参与竞争。
    """

    if corner is None:
        return cross, None
    if cross is None or cross.selected_pose is None:
        return None, corner
    assert cross.selected_pose is not None
    cross_uncertainty = min(
        (
            item
            for item in cross.candidates
            if math.isclose(
                angular_distance(
                    item.pose.heading_rad,
                    cross.selected_pose.heading_rad,
                ),
                0.0,
                abs_tol=1e-9,
            )
        ),
        key=lambda item: item.position_uncertainty_mm,
        default=None,
    )
    corner_innovation = (
        None
        if prior_pose is None
        else _innovation(
            corner.pose,
            prior_pose,
            position_uncertainty_mm=corner.position_uncertainty_mm,
            heading_uncertainty_rad=corner.heading_uncertainty_rad,
        )
    )
    cross_innovation = (
        None
        if prior_pose is None or cross_uncertainty is None
        else _innovation(
            cross.selected_pose,
            prior_pose,
            position_uncertainty_mm=cross_uncertainty.position_uncertainty_mm,
            heading_uncertainty_rad=cross_uncertainty.heading_uncertainty_rad,
        )
    )
    if corner_innovation is not None and cross_innovation is not None:
        if corner_innovation < cross_innovation:
            return None, corner
        return cross, None
    if corner.confidence > cross.confidence:
        return None, corner
    return cross, None
