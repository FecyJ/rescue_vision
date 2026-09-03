"""场地特征检测结果的逐帧观测契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception.types import UndistortedBoundingBox


class SafeZoneColor(str, Enum):
    """安全区的物理颜色；己方/对方语义由后续地图层解释。"""

    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"


class SafeZoneSide(str, Enum):
    """从场内面向安全区入口时的稳定左右语义。"""

    APPROACH_LEFT = "approach_left"
    APPROACH_RIGHT = "approach_right"


class SafeZoneCornerRole(str, Enum):
    """安全区定位使用的稳定几何点语义。"""

    GROUND_ANCHOR = "ground_anchor"
    ENTRANCE_LEFT = "entrance_left"
    ENTRANCE_RIGHT = "entrance_right"
    BACK_LEFT = "back_left"
    BACK_RIGHT = "back_right"


class BoundaryFeatureKind(str, Enum):
    """围栏样式不稳定时仍可输出的低精度边界特征。"""

    FENCE_BASE_SEGMENT = "fence_base_segment"
    FIELD_CORNER = "field_corner"


class FieldFeatureQuality(str, Enum):
    """场地特征观测的显式降级原因。"""

    NO_GROUND_PROJECTION = "no_ground_projection"
    PARTIAL = "partial"
    ENTRANCE_UNRESOLVED = "entrance_unresolved"
    DIVIDER_UNRESOLVED = "divider_unresolved"
    SIDE_UNRESOLVED = "side_unresolved"
    LOW_CONFIDENCE_BOUNDARY = "low_confidence_boundary"
    KEYPOINT_UNAVAILABLE = "keypoint_unavailable"
    AXIS_REFINEMENT_UNAVAILABLE = "axis_refinement_unavailable"
    IDENTITY_UNRESOLVED = "identity_unresolved"


class CenterCrossConfirmation(str, Enum):
    """中心十字候选是否已具备定位消费所需的确认来源。"""

    CANDIDATE = "candidate"
    PRIOR_GUIDED = "prior_guided"
    TEMPORAL_CONFIRMED = "temporal_confirmed"


def _finite(value: float, location: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return converted


def _probability(value: float, location: str) -> float:
    converted = _finite(value, location)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{location} must be in [0, 1], got {value!r}.")
    return converted


def _positive(value: float, location: str) -> float:
    converted = _finite(value, location)
    if converted <= 0.0:
        raise ValueError(f"{location} must be positive, got {value!r}.")
    return converted


def _validate_pixel_polygon(
    polygon: tuple[UndistortedPixel, ...],
    location: str,
) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{location} must contain at least three pixels.")
    for index, point in enumerate(polygon):
        if not isinstance(point, UndistortedPixel):
            raise ValueError(f"{location}[{index}] must be an UndistortedPixel.")
        _finite(point.u, f"{location}[{index}].u")
        _finite(point.v, f"{location}[{index}].v")


def _validate_pixels(
    points: tuple[UndistortedPixel, ...],
    expected: int,
    location: str,
) -> None:
    if len(points) != expected:
        raise ValueError(f"{location} must contain {expected} pixels.")
    for index, point in enumerate(points):
        if not isinstance(point, UndistortedPixel):
            raise ValueError(f"{location}[{index}] must be an UndistortedPixel.")
        _finite(point.u, f"{location}[{index}].u")
        _finite(point.v, f"{location}[{index}].v")


def _validate_ground_polygon(
    polygon: tuple[GroundPoint, ...] | None,
    pixel_count: int,
    location: str,
) -> None:
    if polygon is None:
        return
    if len(polygon) != pixel_count:
        raise ValueError(
            f"{location} must match the pixel polygon length {pixel_count}."
        )
    for index, point in enumerate(polygon):
        if not isinstance(point, GroundPoint):
            raise ValueError(f"{location}[{index}] must be a GroundPoint.")
        _finite(point.x, f"{location}[{index}].x")
        _finite(point.y, f"{location}[{index}].y")


@dataclass(frozen=True, slots=True)
class LineSegmentObservation:
    """同一条线段在去畸变像素和可选机器人地面坐标中的表示。"""

    start_undistorted: UndistortedPixel
    end_undistorted: UndistortedPixel
    start_ground: GroundPoint | None
    end_ground: GroundPoint | None

    def __post_init__(self) -> None:
        if not isinstance(self.start_undistorted, UndistortedPixel):
            raise ValueError("start_undistorted must be an UndistortedPixel.")
        if not isinstance(self.end_undistorted, UndistortedPixel):
            raise ValueError("end_undistorted must be an UndistortedPixel.")
        for name, point in (
            ("start_undistorted", self.start_undistorted),
            ("end_undistorted", self.end_undistorted),
        ):
            _finite(point.u, f"{name}.u")
            _finite(point.v, f"{name}.v")
        if self.start_undistorted == self.end_undistorted:
            raise ValueError("line segment endpoints must be distinct.")
        if (self.start_ground is None) != (self.end_ground is None):
            raise ValueError("ground line endpoints must both be set or both be None.")
        if self.start_ground is not None and self.start_ground == self.end_ground:
            raise ValueError("ground line endpoints must be distinct.")
        if self.start_ground is not None:
            assert self.end_ground is not None
            for name, point in (
                ("start_ground", self.start_ground),
                ("end_ground", self.end_ground),
            ):
                _finite(point.x, f"{name}.x")
                _finite(point.y, f"{name}.y")


@dataclass(frozen=True, slots=True)
class SafeZoneHalfObservation:
    side: SafeZoneSide
    polygon_undistorted: tuple[UndistortedPixel, ...]
    polygon_ground: tuple[GroundPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.side, SafeZoneSide):
            raise ValueError("side must be a SafeZoneSide.")
        _validate_pixel_polygon(self.polygon_undistorted, "polygon_undistorted")
        _validate_ground_polygon(
            self.polygon_ground,
            len(self.polygon_undistorted),
            "polygon_ground",
        )


@dataclass(frozen=True, slots=True)
class SafeZoneCornerObservation:
    """安全区紫色围框的语义角点；地面坐标缺失时不能用于定位。"""

    role: SafeZoneCornerRole
    undistorted: UndistortedPixel
    ground: GroundPoint | None
    confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.role, SafeZoneCornerRole):
            raise ValueError("role must be a SafeZoneCornerRole.")
        if not isinstance(self.undistorted, UndistortedPixel):
            raise ValueError("undistorted must be an UndistortedPixel.")
        _finite(self.undistorted.u, "undistorted.u")
        _finite(self.undistorted.v, "undistorted.v")
        if self.ground is not None:
            if not isinstance(self.ground, GroundPoint):
                raise ValueError("ground must be a GroundPoint or None.")
            _finite(self.ground.x, "ground.x")
            _finite(self.ground.y, "ground.y")
        _probability(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class SafeZoneSearchRegion:
    """由定位先验投影到机器人地面系的安全区搜索范围。"""

    physical_color: SafeZoneColor
    polygon_ground: tuple[GroundPoint, ...]
    margin_mm: float

    def __post_init__(self) -> None:
        if not isinstance(self.physical_color, SafeZoneColor):
            raise ValueError("physical_color must be a SafeZoneColor.")
        _validate_ground_polygon(
            self.polygon_ground,
            len(self.polygon_ground),
            "polygon_ground",
        )
        if len(self.polygon_ground) < 3:
            raise ValueError("polygon_ground must contain at least three points.")
        if _finite(self.margin_mm, "margin_mm") < 0.0:
            raise ValueError("margin_mm must be non-negative.")


@dataclass(frozen=True, slots=True)
class FieldFeatureSearchHint:
    """可丢弃的定位搜索提示；只约束本帧候选，不构成视觉观测。"""

    cross_center_ground: GroundPoint
    cross_radius_mm: float
    cross_axis_directions_ground: tuple[tuple[float, float], ...]
    cross_position_uncertainty_mm: float
    cross_heading_uncertainty_rad: float
    safe_zone_regions: tuple[SafeZoneSearchRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.cross_center_ground, GroundPoint):
            raise ValueError("cross_center_ground must be a GroundPoint.")
        _finite(self.cross_center_ground.x, "cross_center_ground.x")
        _finite(self.cross_center_ground.y, "cross_center_ground.y")
        if _positive(self.cross_radius_mm, "cross_radius_mm") <= 0.0:
            raise ValueError("cross_radius_mm must be positive.")
        _positive(
            self.cross_position_uncertainty_mm,
            "cross_position_uncertainty_mm",
        )
        heading_uncertainty = _positive(
            self.cross_heading_uncertainty_rad,
            "cross_heading_uncertainty_rad",
        )
        if heading_uncertainty >= math.pi:
            raise ValueError(
                "cross_heading_uncertainty_rad must be less than pi radians."
            )
        if len(self.cross_axis_directions_ground) != 2:
            raise ValueError(
                "cross_axis_directions_ground must contain two directions."
            )
        for index, direction in enumerate(self.cross_axis_directions_ground):
            if not isinstance(direction, tuple) or len(direction) != 2:
                raise ValueError(
                    f"cross_axis_directions_ground[{index}] must be a pair."
                )
            forward = _finite(
                direction[0],
                f"cross_axis_directions_ground[{index}][0]",
            )
            left = _finite(
                direction[1],
                f"cross_axis_directions_ground[{index}][1]",
            )
            if not math.isclose(math.hypot(forward, left), 1.0, abs_tol=1e-6):
                raise ValueError("cross axis directions must be unit vectors.")
        dot = sum(
            first * second
            for first, second in zip(
                self.cross_axis_directions_ground[0],
                self.cross_axis_directions_ground[1],
            )
        )
        if not math.isclose(dot, 0.0, abs_tol=1e-6):
            raise ValueError("cross axis directions must be perpendicular.")
        if not all(
            isinstance(item, SafeZoneSearchRegion)
            for item in self.safe_zone_regions
        ):
            raise ValueError(
                "safe_zone_regions must contain SafeZoneSearchRegion values."
            )
        colors = [item.physical_color for item in self.safe_zone_regions]
        if len(colors) != len(set(colors)):
            raise ValueError("safe-zone search colors must be unique.")


@dataclass(frozen=True, slots=True)
class SafeZoneObservation:
    physical_color: SafeZoneColor
    polygon_undistorted: tuple[UndistortedPixel, ...]
    polygon_ground: tuple[GroundPoint, ...] | None
    entrance: LineSegmentObservation | None
    divider: LineSegmentObservation | None
    halves: tuple[SafeZoneHalfObservation, ...]
    confidence: float
    quality: frozenset[FieldFeatureQuality]
    corners: tuple[SafeZoneCornerObservation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.physical_color, SafeZoneColor):
            raise ValueError("physical_color must be a SafeZoneColor.")
        _validate_pixel_polygon(self.polygon_undistorted, "polygon_undistorted")
        _validate_ground_polygon(
            self.polygon_ground,
            len(self.polygon_undistorted),
            "polygon_ground",
        )
        if len(self.halves) not in {0, 2}:
            raise ValueError("halves must be empty or contain left and right.")
        if not all(
            isinstance(item, SafeZoneHalfObservation) for item in self.halves
        ):
            raise ValueError("halves must contain SafeZoneHalfObservation values.")
        if self.halves and {
            item.side for item in self.halves
        } != set(SafeZoneSide):
            raise ValueError("halves must contain approach_left and approach_right.")
        if not all(isinstance(item, SafeZoneCornerObservation) for item in self.corners):
            raise ValueError("corners must contain SafeZoneCornerObservation values.")
        roles = [item.role for item in self.corners]
        if len(roles) != len(set(roles)):
            raise ValueError("safe-zone corner roles must be unique.")
        _probability(self.confidence, "confidence")
        if not all(isinstance(item, FieldFeatureQuality) for item in self.quality):
            raise ValueError("quality must contain FieldFeatureQuality values.")
        if self.halves and (
            self.entrance is None
            or self.divider is None
            or self.polygon_ground is None
            or FieldFeatureQuality.SIDE_UNRESOLVED in self.quality
        ):
            raise ValueError(
                "resolved halves require entrance, divider and ground coordinates."
            )


@dataclass(frozen=True, slots=True)
class StartZoneObservation:
    polygon_undistorted: tuple[UndistortedPixel, ...]
    polygon_ground: tuple[GroundPoint, ...] | None
    confidence: float
    quality: frozenset[FieldFeatureQuality]

    def __post_init__(self) -> None:
        _validate_pixel_polygon(self.polygon_undistorted, "polygon_undistorted")
        _validate_ground_polygon(
            self.polygon_ground,
            len(self.polygon_undistorted),
            "polygon_ground",
        )
        _probability(self.confidence, "confidence")
        if not all(isinstance(item, FieldFeatureQuality) for item in self.quality):
            raise ValueError("quality must contain FieldFeatureQuality values.")


@dataclass(frozen=True, slots=True)
class CenterCrossObservation:
    axes: tuple[LineSegmentObservation, ...]
    intersection_undistorted: UndistortedPixel | None
    intersection_ground: GroundPoint | None
    confidence: float
    quality: frozenset[FieldFeatureQuality]
    confirmation: CenterCrossConfirmation
    axis_fit_residuals_px: tuple[float, ...]
    axis_angle_deg: float | None
    intersection_extrapolated: bool

    def __post_init__(self) -> None:
        if len(self.axes) not in {1, 2}:
            raise ValueError("axes must contain one partial or two complete axes.")
        if not all(isinstance(axis, LineSegmentObservation) for axis in self.axes):
            raise ValueError("axes must contain LineSegmentObservation values.")
        if self.intersection_undistorted is not None:
            if not isinstance(self.intersection_undistorted, UndistortedPixel):
                raise ValueError(
                    "intersection_undistorted must be an UndistortedPixel."
                )
            _finite(self.intersection_undistorted.u, "intersection_undistorted.u")
            _finite(self.intersection_undistorted.v, "intersection_undistorted.v")
        if self.intersection_ground is not None:
            if not isinstance(self.intersection_ground, GroundPoint):
                raise ValueError("intersection_ground must be a GroundPoint.")
            _finite(self.intersection_ground.x, "intersection_ground.x")
            _finite(self.intersection_ground.y, "intersection_ground.y")
        if (self.intersection_undistorted is None) != (
            self.intersection_ground is None
        ) and all(axis.start_ground is not None for axis in self.axes):
            raise ValueError(
                "projected center intersection pixel and ground point must agree."
            )
        if len(self.axes) == 1 and self.intersection_undistorted is not None:
            raise ValueError("a partial center cross cannot have an intersection.")
        if len(self.axes) == 2 and self.intersection_undistorted is None:
            raise ValueError("a complete center cross requires an intersection.")
        if len(self.axes) == 1 and FieldFeatureQuality.PARTIAL not in self.quality:
            raise ValueError("a single center axis must be marked partial.")
        if not isinstance(self.confirmation, CenterCrossConfirmation):
            raise ValueError("confirmation must be a CenterCrossConfirmation.")
        if len(self.axis_fit_residuals_px) != len(self.axes):
            raise ValueError("axis_fit_residuals_px must match axes length.")
        for index, residual in enumerate(self.axis_fit_residuals_px):
            if _finite(residual, f"axis_fit_residuals_px[{index}]") < 0.0:
                raise ValueError("axis fit residuals must be non-negative.")
        if len(self.axes) == 1 and self.axis_angle_deg is not None:
            raise ValueError("a partial center cross cannot have an axis angle.")
        if len(self.axes) == 2:
            if self.axis_angle_deg is None:
                raise ValueError("a complete center cross requires axis_angle_deg.")
            angle = _finite(self.axis_angle_deg, "axis_angle_deg")
            if not 0.0 < angle <= 90.0:
                raise ValueError("axis_angle_deg must be in (0, 90].")
        if not isinstance(self.intersection_extrapolated, bool):
            raise ValueError("intersection_extrapolated must be a boolean.")
        if len(self.axes) == 1 and self.intersection_extrapolated:
            raise ValueError("a partial center cross cannot extrapolate an intersection.")
        _probability(self.confidence, "confidence")
        if not all(isinstance(item, FieldFeatureQuality) for item in self.quality):
            raise ValueError("quality must contain FieldFeatureQuality values.")


@dataclass(frozen=True, slots=True)
class BoundaryFeatureObservation:
    kind: BoundaryFeatureKind
    points_undistorted: tuple[UndistortedPixel, ...]
    points_ground: tuple[GroundPoint, ...] | None
    capture_timestamp_ns: int
    confidence: float
    quality: frozenset[FieldFeatureQuality]
    interior_normal_ground: tuple[float, float] | None = None
    line_offset_mm: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, BoundaryFeatureKind):
            raise ValueError("kind must be a BoundaryFeatureKind.")
        expected = 1 if self.kind is BoundaryFeatureKind.FIELD_CORNER else 2
        _validate_pixels(
            self.points_undistorted,
            expected,
            f"{self.kind.value}.points_undistorted",
        )
        if self.points_ground is not None and len(self.points_ground) != expected:
            raise ValueError(
                f"{self.kind.value} ground points must match pixel points."
            )
        if self.points_ground is not None:
            for index, point in enumerate(self.points_ground):
                if not isinstance(point, GroundPoint):
                    raise ValueError(
                        f"points_ground[{index}] must be a GroundPoint."
                    )
                _finite(point.x, f"points_ground[{index}].x")
                _finite(point.y, f"points_ground[{index}].y")
        _probability(self.confidence, "confidence")
        if not all(isinstance(item, FieldFeatureQuality) for item in self.quality):
            raise ValueError("quality must contain FieldFeatureQuality values.")
        if FieldFeatureQuality.LOW_CONFIDENCE_BOUNDARY not in self.quality:
            raise ValueError("boundary features must be marked low confidence.")
        if self.capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be non-negative.")
        if (self.interior_normal_ground is None) != (self.line_offset_mm is None):
            raise ValueError(
                "interior_normal_ground and line_offset_mm must be set together."
            )
        if self.interior_normal_ground is not None:
            if self.kind is not BoundaryFeatureKind.FENCE_BASE_SEGMENT:
                raise ValueError("only fence base segments can carry line parameters.")
            if self.points_ground is None:
                raise ValueError("line parameters require ground points.")
            nx, ny = self.interior_normal_ground
            offset = float(self.line_offset_mm)
            if not all(math.isfinite(value) for value in (nx, ny, offset)):
                raise ValueError("boundary line parameters must be finite.")
            if not math.isclose(math.hypot(nx, ny), 1.0, abs_tol=1e-6):
                raise ValueError("interior_normal_ground must be a unit vector.")
            for point in self.points_ground:
                if not math.isclose(
                    nx * point.x + ny * point.y + offset,
                    0.0,
                    abs_tol=1e-5,
                ):
                    raise ValueError("ground endpoints must lie on the boundary line.")


@dataclass(frozen=True, slots=True)
class FieldPoseKeypoint:
    """v3 场地关键点在去畸变像素与机器人地面系中的同帧观测。"""

    undistorted: UndistortedPixel | None
    ground: GroundPoint | None
    confidence: float

    def __post_init__(self) -> None:
        _probability(self.confidence, "confidence")
        if self.undistorted is None:
            if self.ground is not None or self.confidence != 0.0:
                raise ValueError("unavailable field keypoint requires no ground point and zero confidence.")
            return
        if not isinstance(self.undistorted, UndistortedPixel):
            raise ValueError("undistorted must be an UndistortedPixel or None.")
        _finite(self.undistorted.u, "undistorted.u")
        _finite(self.undistorted.v, "undistorted.v")
        if self.ground is not None:
            if not isinstance(self.ground, GroundPoint):
                raise ValueError("ground must be a GroundPoint or None.")
            _finite(self.ground.x, "ground.x")
            _finite(self.ground.y, "ground.y")


@dataclass(frozen=True, slots=True)
class SafeZonePoseObservation:
    """v3 安全区实例；K1/K2 只有当前图像左右语义。"""

    box: UndistortedBoundingBox
    ground_anchor: FieldPoseKeypoint
    image_left_landmark: FieldPoseKeypoint
    image_right_landmark: FieldPoseKeypoint
    physical_color: SafeZoneColor
    confidence: float
    quality: frozenset[FieldFeatureQuality]

    def __post_init__(self) -> None:
        if not isinstance(self.box, UndistortedBoundingBox):
            raise ValueError("box must be an UndistortedBoundingBox.")
        for name in ("ground_anchor", "image_left_landmark", "image_right_landmark"):
            if not isinstance(getattr(self, name), FieldPoseKeypoint):
                raise ValueError(f"{name} must be a FieldPoseKeypoint.")
        if not isinstance(self.physical_color, SafeZoneColor):
            raise ValueError("physical_color must be a SafeZoneColor.")
        _probability(self.confidence, "confidence")
        if not all(isinstance(item, FieldFeatureQuality) for item in self.quality):
            raise ValueError("quality must contain FieldFeatureQuality values.")
        left = self.image_left_landmark.undistorted
        right = self.image_right_landmark.undistorted
        if left is not None and right is not None and left.u > right.u:
            left_landmark = self.image_left_landmark
            right_landmark = self.image_right_landmark
            object.__setattr__(self, "image_left_landmark", right_landmark)
            object.__setattr__(self, "image_right_landmark", left_landmark)
        if (
            self.physical_color is SafeZoneColor.UNKNOWN
            and FieldFeatureQuality.IDENTITY_UNRESOLVED not in self.quality
        ):
            raise ValueError("unknown safe-zone identity must be marked unresolved.")


@dataclass(frozen=True, slots=True)
class CenterCrossPoseObservation:
    """v3 中心十字实例；轴线是模型定位后的可选局部精修。"""

    box: UndistortedBoundingBox
    intersection: FieldPoseKeypoint
    axes: tuple[LineSegmentObservation, ...]
    confidence: float
    quality: frozenset[FieldFeatureQuality]
    confirmation: CenterCrossConfirmation = CenterCrossConfirmation.CANDIDATE
    axis_fit_residuals_px: tuple[float, ...] = ()
    axis_angle_deg: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.box, UndistortedBoundingBox):
            raise ValueError("box must be an UndistortedBoundingBox.")
        if not isinstance(self.intersection, FieldPoseKeypoint):
            raise ValueError("intersection must be a FieldPoseKeypoint.")
        if len(self.axes) not in {0, 1, 2} or not all(
            isinstance(axis, LineSegmentObservation) for axis in self.axes
        ):
            raise ValueError("axes must contain zero, one, or two line segments.")
        if len(self.axis_fit_residuals_px) != len(self.axes):
            raise ValueError("axis_fit_residuals_px must match axes length.")
        if len(self.axes) == 2:
            angle = _finite(self.axis_angle_deg, "axis_angle_deg")
            if not 0.0 < angle <= 90.0:
                raise ValueError("axis_angle_deg must be in (0, 90].")
        elif self.axis_angle_deg is not None:
            raise ValueError("axis_angle_deg requires two refined axes.")
        _probability(self.confidence, "confidence")
        if not all(isinstance(item, FieldFeatureQuality) for item in self.quality):
            raise ValueError("quality must contain FieldFeatureQuality values.")

    @property
    def intersection_undistorted(self) -> UndistortedPixel | None:
        return self.intersection.undistorted

    @property
    def intersection_ground(self) -> GroundPoint | None:
        return self.intersection.ground


# 新版公共名称。旧 OpenCV 形状契约不再由 FieldFeatureDetectionResult 使用。
SafeZoneObservation = SafeZonePoseObservation
CenterCrossObservation = CenterCrossPoseObservation


@dataclass(frozen=True, slots=True)
class FieldFeatureDetectionResult:
    """一帧场地特征检测的完整输出。"""

    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    image_size: tuple[int, int]
    safe_zones: tuple[SafeZoneObservation, ...]
    center_cross: CenterCrossObservation | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.frame_sequence, bool)
            or not isinstance(self.frame_sequence, int)
            or self.frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be non-negative.")
        if self.capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be non-negative.")
        if self.result_timestamp_ns < self.capture_timestamp_ns:
            raise ValueError(
                "result_timestamp_ns must not be earlier than capture_timestamp_ns."
            )
        if (
            not isinstance(self.image_size, tuple)
            or len(self.image_size) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, int)
                or value <= 0
                for value in self.image_size
            )
        ):
            raise ValueError("image_size must be positive integer (width, height).")
        if not all(
            isinstance(item, SafeZoneObservation) for item in self.safe_zones
        ):
            raise ValueError("safe_zones must contain SafeZoneObservation values.")
        if self.center_cross is not None and not isinstance(
            self.center_cross,
            CenterCrossObservation,
        ):
            raise ValueError("center_cross must be a CenterCrossObservation or None.")


@dataclass(frozen=True, slots=True)
class RealtimeFieldFeatureResult:
    """实时场地检测结果；过期帧只保留丢弃原因。"""

    result: FieldFeatureDetectionResult | None
    dropped_stale_age_ms: float | None = None

    def __post_init__(self) -> None:
        if (self.result is None) == (self.dropped_stale_age_ms is None):
            raise ValueError(
                "exactly one of result and dropped_stale_age_ms must be set."
            )
        if self.dropped_stale_age_ms is not None:
            _positive(self.dropped_stale_age_ms, "dropped_stale_age_ms")

    @property
    def stale_dropped(self) -> bool:
        return self.dropped_stale_age_ms is not None
