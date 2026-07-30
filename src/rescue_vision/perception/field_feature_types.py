"""传统视觉场地特征的配置与逐帧观测契约。"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.perception.types import HsvRange


class SafeZoneColor(str, Enum):
    """安全区的物理颜色；己方/对方语义由后续地图层解释。"""

    RED = "red"
    BLUE = "blue"


class SafeZoneSide(str, Enum):
    """从场内面向安全区入口时的稳定左右语义。"""

    APPROACH_LEFT = "approach_left"
    APPROACH_RIGHT = "approach_right"


class BoundaryFeatureKind(str, Enum):
    """围栏样式不稳定时仍可输出的低精度边界特征。"""

    FENCE_BASE_SEGMENT = "fence_base_segment"
    FIELD_CORNER = "field_corner"


class FieldFeatureQuality(str, Enum):
    """传统视觉观测的显式降级原因。"""

    NO_GROUND_PROJECTION = "no_ground_projection"
    PARTIAL = "partial"
    ENTRANCE_UNRESOLVED = "entrance_unresolved"
    DIVIDER_UNRESOLVED = "divider_unresolved"
    SIDE_UNRESOLVED = "side_unresolved"
    LOW_CONFIDENCE_BOUNDARY = "low_confidence_boundary"


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
class FieldFeatureConfig:
    """颜色、形态学和几何筛选的单一运行时权威。"""

    enabled: bool
    safe_red: tuple[HsvRange, ...]
    safe_blue: tuple[HsvRange, ...]
    start_magenta: tuple[HsvRange, ...]
    entrance_purple: tuple[HsvRange, ...]
    dark_marking: tuple[HsvRange, ...]
    morphology_kernel_size: int
    open_iterations: int
    close_iterations: int
    min_region_area_fraction: float
    min_rectangularity: float
    safe_width_mm: float
    safe_depth_mm: float
    start_side_mm: float
    dimension_tolerance_fraction: float
    entrance_color_fraction: float
    divider_dark_fraction: float
    center_min_axis_span_fraction: float
    center_max_gap_fraction: float
    center_min_gap_count: int
    center_perpendicular_tolerance_deg: float
    boundary_canny_low_threshold: int
    boundary_canny_high_threshold: int
    boundary_min_line_length_fraction: float
    boundary_corner_tolerance_deg: float
    boundary_max_features: int

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        for name in (
            "safe_red",
            "safe_blue",
            "start_magenta",
            "entrance_purple",
            "dark_marking",
        ):
            ranges = getattr(self, name)
            if (
                not isinstance(ranges, tuple)
                or not ranges
                or not all(isinstance(item, HsvRange) for item in ranges)
            ):
                raise ValueError(f"{name} must contain at least one HsvRange.")
        if (
            isinstance(self.morphology_kernel_size, bool)
            or not isinstance(self.morphology_kernel_size, int)
            or self.morphology_kernel_size <= 0
            or self.morphology_kernel_size % 2 == 0
        ):
            raise ValueError(
                "morphology_kernel_size must be a positive odd integer."
            )
        for name in ("open_iterations", "close_iterations"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
        if (
            isinstance(self.center_min_gap_count, bool)
            or not isinstance(self.center_min_gap_count, int)
            or self.center_min_gap_count <= 0
        ):
            raise ValueError("center_min_gap_count must be a positive integer.")
        for name in (
            "boundary_canny_low_threshold",
            "boundary_canny_high_threshold",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or not 0 <= value <= 255
            ):
                raise ValueError(f"{name} must be an integer in [0, 255].")
        if self.boundary_canny_low_threshold >= self.boundary_canny_high_threshold:
            raise ValueError(
                "boundary_canny_low_threshold must be less than "
                "boundary_canny_high_threshold."
            )
        for name in (
            "min_region_area_fraction",
            "min_rectangularity",
            "dimension_tolerance_fraction",
            "entrance_color_fraction",
            "divider_dark_fraction",
            "center_min_axis_span_fraction",
            "center_max_gap_fraction",
            "boundary_min_line_length_fraction",
        ):
            _probability(getattr(self, name), name)
        if self.dimension_tolerance_fraction >= 1.0:
            raise ValueError("dimension_tolerance_fraction must be less than 1.")
        for name in ("safe_width_mm", "safe_depth_mm", "start_side_mm"):
            _positive(getattr(self, name), name)
        for name in (
            "center_perpendicular_tolerance_deg",
            "boundary_corner_tolerance_deg",
        ):
            value = _positive(getattr(self, name), name)
            if value >= 45.0:
                raise ValueError(f"{name} must be less than 45 degrees.")
        if (
            isinstance(self.boundary_max_features, bool)
            or not isinstance(self.boundary_max_features, int)
            or self.boundary_max_features <= 0
        ):
            raise ValueError("boundary_max_features must be a positive integer.")

    def ranges_for_safe_color(
        self,
        color: SafeZoneColor,
    ) -> tuple[HsvRange, ...]:
        if not isinstance(color, SafeZoneColor):
            raise ValueError("color must be a SafeZoneColor.")
        return self.safe_red if color is SafeZoneColor.RED else self.safe_blue


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
class SafeZoneObservation:
    physical_color: SafeZoneColor
    polygon_undistorted: tuple[UndistortedPixel, ...]
    polygon_ground: tuple[GroundPoint, ...] | None
    entrance: LineSegmentObservation | None
    divider: LineSegmentObservation | None
    halves: tuple[SafeZoneHalfObservation, ...]
    confidence: float
    quality: frozenset[FieldFeatureQuality]

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
        _probability(self.confidence, "confidence")
        if not all(isinstance(item, FieldFeatureQuality) for item in self.quality):
            raise ValueError("quality must contain FieldFeatureQuality values.")


@dataclass(frozen=True, slots=True)
class BoundaryFeatureObservation:
    kind: BoundaryFeatureKind
    points_undistorted: tuple[UndistortedPixel, ...]
    points_ground: tuple[GroundPoint, ...] | None
    confidence: float
    quality: frozenset[FieldFeatureQuality]

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


@dataclass(frozen=True, slots=True)
class FieldFeatureDetectionResult:
    """一帧传统视觉场地特征的完整输出。"""

    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    image_size: tuple[int, int]
    safe_zones: tuple[SafeZoneObservation, ...]
    start_zones: tuple[StartZoneObservation, ...]
    center_cross: CenterCrossObservation | None
    boundary_features: tuple[BoundaryFeatureObservation, ...]

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
        if not all(
            isinstance(item, StartZoneObservation) for item in self.start_zones
        ):
            raise ValueError("start_zones must contain StartZoneObservation values.")
        if self.center_cross is not None and not isinstance(
            self.center_cross,
            CenterCrossObservation,
        ):
            raise ValueError("center_cross must be a CenterCrossObservation or None.")
        if not all(
            isinstance(item, BoundaryFeatureObservation)
            for item in self.boundary_features
        ):
            raise ValueError(
                "boundary_features must contain BoundaryFeatureObservation values."
            )


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
