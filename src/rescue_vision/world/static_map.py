"""Physical static-map facts shared by localization and world semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from rescue_vision.geometry.types import FieldPoint


class TeamColor(str, Enum):
    RED = "red"
    BLUE = "blue"
    UNKNOWN = "unknown"


class PhysicalRegionKind(str, Enum):
    FIELD = "field"
    RED_MATERIAL = "red_material"
    RED_INJURED = "red_injured"
    BLUE_MATERIAL = "blue_material"
    BLUE_INJURED = "blue_injured"
    START_ZONE = "start_zone"


class CenterCrossRay(str, Enum):
    POSITIVE_X = "positive_x"
    NEGATIVE_X = "negative_x"
    POSITIVE_Y = "positive_y"
    NEGATIVE_Y = "negative_y"

    @property
    def angle_rad(self) -> float:
        return {
            CenterCrossRay.POSITIVE_X: 0.0,
            CenterCrossRay.NEGATIVE_X: math.pi,
            CenterCrossRay.POSITIVE_Y: math.pi / 2.0,
            CenterCrossRay.NEGATIVE_Y: -math.pi / 2.0,
        }[self]


class CenterLineTerminalKind(str, Enum):
    RED_SAFE_ZONE = "red_safe_zone"
    BLUE_SAFE_ZONE = "blue_safe_zone"
    PLAIN_BOUNDARY = "plain_boundary"
    UNKNOWN = "unknown"


def _finite(value: float, location: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return converted


def _validate_polygon(
    polygon: tuple[FieldPoint, ...],
    location: str,
) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{location} must contain at least three FieldPoint values.")
    for index, point in enumerate(polygon):
        if not isinstance(point, FieldPoint):
            raise ValueError(f"{location}[{index}] must be a FieldPoint.")
        _finite(point.x, f"{location}[{index}].x")
        _finite(point.y, f"{location}[{index}].y")
    area_twice = sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    )
    if math.isclose(area_twice, 0.0, abs_tol=1e-9):
        raise ValueError(f"{location} must have non-zero area.")


@dataclass(frozen=True, slots=True)
class PhysicalStaticRegion:
    region_id: str
    kind: PhysicalRegionKind
    polygon_field: tuple[FieldPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ValueError("region_id must be a non-empty string.")
        if not isinstance(self.kind, PhysicalRegionKind):
            raise ValueError("kind must be a PhysicalRegionKind.")
        _validate_polygon(self.polygon_field, "polygon_field")


@dataclass(frozen=True, slots=True)
class CenterCrossTerminal:
    ray: CenterCrossRay
    kind: CenterLineTerminalKind

    def __post_init__(self) -> None:
        if not isinstance(self.ray, CenterCrossRay):
            raise ValueError("ray must be a CenterCrossRay.")
        if not isinstance(self.kind, CenterLineTerminalKind):
            raise ValueError("kind must be a CenterLineTerminalKind.")


@dataclass(frozen=True, slots=True)
class StaticCenterCross:
    intersection_field: FieldPoint
    terminals: tuple[CenterCrossTerminal, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.intersection_field, FieldPoint):
            raise ValueError("intersection_field must be a FieldPoint.")
        _finite(self.intersection_field.x, "intersection_field.x")
        _finite(self.intersection_field.y, "intersection_field.y")
        if self.intersection_field != FieldPoint(0.0, 0.0):
            raise ValueError(
                "intersection_field must be [0, 0] because the field frame "
                "origin is the center-cross intersection."
            )
        if len(self.terminals) != len(CenterCrossRay):
            raise ValueError("terminals must contain all four center-cross rays.")
        if not all(isinstance(item, CenterCrossTerminal) for item in self.terminals):
            raise ValueError("terminals must contain CenterCrossTerminal values.")
        rays = {item.ray for item in self.terminals}
        if rays != set(CenterCrossRay):
            raise ValueError("terminals must contain each CenterCrossRay exactly once.")

    def rays_for_terminal(
        self,
        kind: CenterLineTerminalKind,
    ) -> tuple[CenterCrossRay, ...]:
        if not isinstance(kind, CenterLineTerminalKind):
            raise ValueError("kind must be a CenterLineTerminalKind.")
        return tuple(item.ray for item in self.terminals if item.kind is kind)


@dataclass(frozen=True, slots=True)
class StaticFieldMap:
    center_cross: StaticCenterCross
    regions: tuple[PhysicalStaticRegion, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.center_cross, StaticCenterCross):
            raise ValueError("center_cross must be a StaticCenterCross.")
        if not all(isinstance(item, PhysicalStaticRegion) for item in self.regions):
            raise ValueError("regions must contain PhysicalStaticRegion values.")
        ids = [item.region_id for item in self.regions]
        if len(ids) != len(set(ids)):
            raise ValueError("physical static region IDs must be unique.")
        field_count = sum(
            item.kind is PhysicalRegionKind.FIELD for item in self.regions
        )
        if field_count > 1:
            raise ValueError("static_map may contain at most one field region.")

    @staticmethod
    def _bounds(
        regions: tuple[PhysicalStaticRegion, ...],
    ) -> tuple[float, float, float, float] | None:
        if not regions:
            return None
        points = tuple(
            point for region in regions for point in region.polygon_field
        )
        return (
            min(point.x for point in points),
            max(point.x for point in points),
            min(point.y for point in points),
            max(point.y for point in points),
        )

    def safe_zone_dimensions_mm(
        self,
        color: TeamColor,
    ) -> tuple[float, float] | None:
        """Return physical safe-zone width/depth from its two configured halves."""

        if color is TeamColor.UNKNOWN:
            raise ValueError("safe-zone dimensions require red or blue color.")
        kinds = (
            {PhysicalRegionKind.RED_MATERIAL, PhysicalRegionKind.RED_INJURED}
            if color is TeamColor.RED
            else {
                PhysicalRegionKind.BLUE_MATERIAL,
                PhysicalRegionKind.BLUE_INJURED,
            }
        )
        selected = tuple(region for region in self.regions if region.kind in kinds)
        if len(selected) != 2 or {region.kind for region in selected} != kinds:
            return None
        bounds = self._bounds(selected)
        if bounds is None:
            return None
        min_x, max_x, min_y, max_y = bounds
        return max_x - min_x, max_y - min_y

    def start_zone_dimensions_mm(self) -> tuple[tuple[float, float], ...]:
        """Return bounding-box width/depth for every configured start zone."""

        dimensions: list[tuple[float, float]] = []
        for region in self.regions:
            if region.kind is not PhysicalRegionKind.START_ZONE:
                continue
            bounds = self._bounds((region,))
            assert bounds is not None
            min_x, max_x, min_y, max_y = bounds
            dimensions.append((max_x - min_x, max_y - min_y))
        return tuple(dimensions)


def default_static_field_map() -> StaticFieldMap:
    """Return coordinate-defining landmarks without claiming measured regions."""

    return StaticFieldMap(
        center_cross=StaticCenterCross(
            intersection_field=FieldPoint(0.0, 0.0),
            terminals=(
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
                    CenterLineTerminalKind.RED_SAFE_ZONE,
                ),
                CenterCrossTerminal(
                    CenterCrossRay.NEGATIVE_Y,
                    CenterLineTerminalKind.BLUE_SAFE_ZONE,
                ),
            ),
        ),
        regions=(),
    )
