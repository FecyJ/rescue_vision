"""把跟踪目标与场地区域整理为单调时间轴上的世界快照。"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
import math

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception.types import ClassProbabilities, TargetClass
from rescue_vision.tracking.tracker import TrackStatus, TrackedTarget


class RegionKind(str, Enum):
    FIELD = "field"
    OWN_MATERIAL = "own_material"
    OWN_INJURED = "own_injured"
    OPPONENT_SAFE = "opponent_safe"


class HazardState(str, Enum):
    CLEAR = "clear"
    SUSPECTED = "suspected"
    CONFIRMED = "confirmed"


class WorldUncertainty(str, Enum):
    STALE_VISION = "stale_vision"
    MISSING_ROBOT_FIELD_POSITION = "missing_robot_field_position"
    TARGET_WITHOUT_GROUND_POINT = "target_without_ground_point"
    UNCONFIRMED_TARGET = "unconfirmed_target"
    UNKNOWN_TARGET = "unknown_target"
    STALE_OPPONENT = "stale_opponent"


def _finite(value: float, name: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return converted


def _probability(value: float, name: str) -> float:
    converted = _finite(value, name)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value!r}.")
    return converted


def _validate_polygon(
    polygon: tuple[FieldPoint, ...],
    name: str,
) -> None:
    if len(polygon) < 3:
        raise ValueError(f"{name} must contain at least three FieldPoint values.")
    for index, point in enumerate(polygon):
        if not isinstance(point, FieldPoint):
            raise ValueError(f"{name}[{index}] must be a FieldPoint.")
        _finite(point.x, f"{name}[{index}].x")
        _finite(point.y, f"{name}[{index}].y")
    area_twice = sum(
        first.x * second.y - second.x * first.y
        for first, second in zip(polygon, polygon[1:] + polygon[:1])
    )
    if math.isclose(area_twice, 0.0, abs_tol=1e-9):
        raise ValueError(f"{name} must have non-zero area.")


def _point_on_segment(
    point: FieldPoint,
    start: FieldPoint,
    end: FieldPoint,
) -> bool:
    cross = (
        (point.y - start.y) * (end.x - start.x)
        - (point.x - start.x) * (end.y - start.y)
    )
    if not math.isclose(cross, 0.0, abs_tol=1e-7):
        return False
    return (
        min(start.x, end.x) - 1e-7
        <= point.x
        <= max(start.x, end.x) + 1e-7
        and min(start.y, end.y) - 1e-7
        <= point.y
        <= max(start.y, end.y) + 1e-7
    )


def _contains(
    polygon: tuple[FieldPoint, ...],
    point: FieldPoint,
) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        if _point_on_segment(point, previous, current):
            return True
        crosses = (current.y > point.y) != (previous.y > point.y)
        if crosses:
            crossing_x = (
                (previous.x - current.x)
                * (point.y - current.y)
                / (previous.y - current.y)
                + current.x
            )
            if point.x < crossing_x:
                inside = not inside
        previous = current
    return inside


@dataclass(frozen=True, slots=True)
class StaticRegion:
    region_id: str
    kind: RegionKind
    polygon_field: tuple[FieldPoint, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.region_id, str) or not self.region_id.strip():
            raise ValueError("region_id must be a non-empty string.")
        if not isinstance(self.kind, RegionKind):
            raise ValueError("kind must be a RegionKind.")
        _validate_polygon(self.polygon_field, "polygon_field")

    def contains(self, point: FieldPoint) -> bool:
        return _contains(self.polygon_field, point)


@dataclass(frozen=True, slots=True)
class OpponentOccupancy:
    opponent_id: str
    polygon_field: tuple[FieldPoint, ...]
    confidence: float
    timestamp_ns: int

    def __post_init__(self) -> None:
        if not isinstance(self.opponent_id, str) or not self.opponent_id.strip():
            raise ValueError("opponent_id must be a non-empty string.")
        _validate_polygon(self.polygon_field, "polygon_field")
        _probability(self.confidence, "confidence")
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")

    def contains(self, point: FieldPoint) -> bool:
        return _contains(self.polygon_field, point)


@dataclass(frozen=True, slots=True)
class WorldModelConfig:
    max_visual_age_ms: float
    opponent_max_age_ms: float
    danger_confirm_threshold: float
    danger_suspect_threshold: float
    unknown_suspect_threshold: float

    def __post_init__(self) -> None:
        for name in ("max_visual_age_ms", "opponent_max_age_ms"):
            value = _finite(getattr(self, name), name)
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        for name in (
            "danger_confirm_threshold",
            "danger_suspect_threshold",
            "unknown_suspect_threshold",
        ):
            _probability(getattr(self, name), name)
        if self.danger_suspect_threshold > self.danger_confirm_threshold:
            raise ValueError(
                "danger_suspect_threshold must not exceed "
                "danger_confirm_threshold."
            )


@dataclass(frozen=True, slots=True)
class WorldTarget:
    track_id: int
    track_status: TrackStatus
    ever_confirmed: bool
    target_class: TargetClass
    class_probabilities: ClassProbabilities
    confidence: float
    hazard_state: HazardState
    ground_point: GroundPoint | None
    field_point: FieldPoint | None
    last_seen_timestamp_ns: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.track_id, bool)
            or not isinstance(self.track_id, int)
            or self.track_id <= 0
        ):
            raise ValueError("track_id must be a positive integer.")
        if not isinstance(self.track_status, TrackStatus):
            raise ValueError("track_status must be a TrackStatus.")
        if not isinstance(self.ever_confirmed, bool):
            raise ValueError("ever_confirmed must be a boolean.")
        if not isinstance(self.target_class, TargetClass):
            raise ValueError("target_class must be a TargetClass.")
        if not isinstance(self.class_probabilities, ClassProbabilities):
            raise ValueError(
                "class_probabilities must be a ClassProbabilities."
            )
        _probability(self.confidence, "confidence")
        if not isinstance(self.hazard_state, HazardState):
            raise ValueError("hazard_state must be a HazardState.")
        for name, point, point_type in (
            ("ground_point", self.ground_point, GroundPoint),
            ("field_point", self.field_point, FieldPoint),
        ):
            if point is not None:
                if not isinstance(point, point_type):
                    raise ValueError(
                        f"{name} must be a {point_type.__name__} or None."
                    )
                _finite(point.x, f"{name}.x")
                _finite(point.y, f"{name}.y")
        if (
            isinstance(self.last_seen_timestamp_ns, bool)
            or not isinstance(self.last_seen_timestamp_ns, int)
            or self.last_seen_timestamp_ns < 0
        ):
            raise ValueError(
                "last_seen_timestamp_ns must be a non-negative integer."
            )


@dataclass(frozen=True, slots=True)
class WorldSnapshot:
    timestamp_ns: int
    visual_timestamp_ns: int
    regions: tuple[StaticRegion, ...]
    targets: tuple[WorldTarget, ...]
    opponent_occupancies: tuple[OpponentOccupancy, ...]
    robot_field_point: FieldPoint | None
    robot_region_kinds: frozenset[RegionKind]
    uncertainties: frozenset[WorldUncertainty]

    def __post_init__(self) -> None:
        for name, value in (
            ("timestamp_ns", self.timestamp_ns),
            ("visual_timestamp_ns", self.visual_timestamp_ns),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.visual_timestamp_ns > self.timestamp_ns:
            raise ValueError(
                "visual_timestamp_ns must not be later than timestamp_ns."
            )
        if len({target.track_id for target in self.targets}) != len(self.targets):
            raise ValueError("targets must contain unique track IDs.")
        if len(
            {
                occupancy.opponent_id
                for occupancy in self.opponent_occupancies
            }
        ) != len(self.opponent_occupancies):
            raise ValueError(
                "opponent_occupancies must contain unique opponent IDs."
            )
        if self.robot_field_point is not None and not isinstance(
            self.robot_field_point,
            FieldPoint,
        ):
            raise ValueError(
                "robot_field_point must be a FieldPoint or None."
            )
        if not all(
            isinstance(kind, RegionKind) for kind in self.robot_region_kinds
        ):
            raise ValueError(
                "robot_region_kinds must contain only RegionKind values."
            )
        if not all(
            isinstance(item, WorldUncertainty) for item in self.uncertainties
        ):
            raise ValueError(
                "uncertainties must contain only WorldUncertainty values."
            )

    @property
    def visual_age_ms(self) -> float:
        return (self.timestamp_ns - self.visual_timestamp_ns) / 1_000_000.0

    def target(self, track_id: int) -> WorldTarget | None:
        return next(
            (target for target in self.targets if target.track_id == track_id),
            None,
        )

    def robot_in_region(self, kind: RegionKind) -> bool:
        return kind in self.robot_region_kinds

    def target_region_kinds(
        self,
        track_id: int,
    ) -> frozenset[RegionKind] | None:
        """返回目标所在静态区域；缺少 FieldPoint 时返回 None。"""

        target = self.target(track_id)
        if target is None:
            raise KeyError(f"Unknown track_id {track_id}.")
        if target.field_point is None:
            return None
        return frozenset(
            region.kind
            for region in self.regions
            if region.contains(target.field_point)
        )

    def target_in_opponent_occupancy(
        self,
        track_id: int,
    ) -> bool | None:
        """判断目标是否落入对手占据；缺少 FieldPoint 时返回 None。"""

        target = self.target(track_id)
        if target is None:
            raise KeyError(f"Unknown track_id {track_id}.")
        if target.field_point is None:
            return None
        return any(
            occupancy.contains(target.field_point)
            for occupancy in self.opponent_occupancies
        )

    def hazards_within_ground_distance(
        self,
        distance_mm: float,
    ) -> tuple[WorldTarget, ...]:
        if not math.isfinite(distance_mm) or distance_mm < 0.0:
            raise ValueError("distance_mm must be finite and non-negative.")
        return tuple(
            target
            for target in self.targets
            if target.hazard_state is not HazardState.CLEAR
            and target.ground_point is not None
            and math.hypot(target.ground_point.x, target.ground_point.y)
            <= distance_mm
        )


class WorldModel:
    def __init__(
        self,
        config: WorldModelConfig,
        regions: tuple[StaticRegion, ...] | list[StaticRegion],
    ) -> None:
        self._config = config
        self._regions = tuple(regions)
        if not all(
            isinstance(region, StaticRegion) for region in self._regions
        ):
            raise ValueError("regions must contain only StaticRegion values.")
        region_ids = [region.region_id for region in self._regions]
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("Static region IDs must be unique.")
        self._opponents: dict[str, OpponentOccupancy] = {}
        self._last_timestamp_ns: int | None = None
        self._last_visual_timestamp_ns: int | None = None

    def update(
        self,
        *,
        timestamp_ns: int,
        visual_timestamp_ns: int,
        tracks: tuple[TrackedTarget, ...] | list[TrackedTarget],
        robot_field_point: FieldPoint | None = None,
        target_field_points: Mapping[int, FieldPoint] | None = None,
        opponent_occupancies: (
            tuple[OpponentOccupancy, ...] | list[OpponentOccupancy]
        ) = (),
    ) -> WorldSnapshot:
        self._validate_time(timestamp_ns, visual_timestamp_ns)
        tracks = tuple(tracks)
        if not all(isinstance(track, TrackedTarget) for track in tracks):
            raise ValueError("tracks must contain only TrackedTarget values.")
        if len({track.track_id for track in tracks}) != len(tracks):
            raise ValueError("tracks must contain unique track_id values.")
        field_points = dict(target_field_points or {})
        extra_ids = set(field_points) - {track.track_id for track in tracks}
        if extra_ids:
            raise ValueError(
                f"target_field_points contains unknown track IDs {sorted(extra_ids)}."
            )
        for track_id, point in field_points.items():
            if not isinstance(point, FieldPoint):
                raise ValueError(
                    f"target_field_points[{track_id}] must be a FieldPoint."
                )
        if robot_field_point is not None and not isinstance(
            robot_field_point,
            FieldPoint,
        ):
            raise ValueError("robot_field_point must be a FieldPoint or None.")

        uncertainties: set[WorldUncertainty] = set()
        visual_age_ms = (
            timestamp_ns - visual_timestamp_ns
        ) / 1_000_000.0
        if visual_age_ms > self._config.max_visual_age_ms:
            uncertainties.add(WorldUncertainty.STALE_VISION)
        if robot_field_point is None:
            uncertainties.add(WorldUncertainty.MISSING_ROBOT_FIELD_POSITION)

        targets: list[WorldTarget] = []
        for track in sorted(tracks, key=lambda item: item.track_id):
            hazard_state = self._hazard_state(track)
            if track.ground_point is None:
                uncertainties.add(WorldUncertainty.TARGET_WITHOUT_GROUND_POINT)
            if not track.ever_confirmed:
                uncertainties.add(WorldUncertainty.UNCONFIRMED_TARGET)
            if track.target_class is TargetClass.UNKNOWN:
                uncertainties.add(WorldUncertainty.UNKNOWN_TARGET)
            targets.append(
                WorldTarget(
                    track_id=track.track_id,
                    track_status=track.status,
                    ever_confirmed=track.ever_confirmed,
                    target_class=track.target_class,
                    class_probabilities=track.class_probabilities,
                    confidence=track.confidence,
                    hazard_state=hazard_state,
                    ground_point=track.ground_point,
                    field_point=field_points.get(track.track_id),
                    last_seen_timestamp_ns=track.last_seen_timestamp_ns,
                )
            )

        opponent_occupancies = tuple(opponent_occupancies)
        opponent_ids = [
            occupancy.opponent_id
            for occupancy in opponent_occupancies
            if isinstance(occupancy, OpponentOccupancy)
        ]
        if len(opponent_ids) != len(opponent_occupancies):
            raise ValueError(
                "opponent_occupancies must contain only OpponentOccupancy values."
            )
        if len(set(opponent_ids)) != len(opponent_ids):
            raise ValueError(
                "opponent_occupancies must contain unique opponent IDs."
            )
        for occupancy in opponent_occupancies:
            if occupancy.timestamp_ns > timestamp_ns:
                raise ValueError(
                    f"Opponent {occupancy.opponent_id!r} timestamp is in the future."
                )
            previous = self._opponents.get(occupancy.opponent_id)
            if (
                previous is not None
                and occupancy.timestamp_ns < previous.timestamp_ns
            ):
                raise ValueError(
                    f"Opponent {occupancy.opponent_id!r} timestamp moved "
                    f"backwards from {previous.timestamp_ns} to "
                    f"{occupancy.timestamp_ns}."
                )
        for occupancy in opponent_occupancies:
            self._opponents[occupancy.opponent_id] = occupancy
        fresh_opponents: dict[str, OpponentOccupancy] = {}
        for opponent_id, occupancy in self._opponents.items():
            age_ms = (timestamp_ns - occupancy.timestamp_ns) / 1_000_000.0
            if age_ms <= self._config.opponent_max_age_ms:
                fresh_opponents[opponent_id] = occupancy
            else:
                uncertainties.add(WorldUncertainty.STALE_OPPONENT)
        self._opponents = fresh_opponents

        robot_regions = frozenset(
            region.kind
            for region in self._regions
            if robot_field_point is not None and region.contains(robot_field_point)
        )
        self._last_timestamp_ns = timestamp_ns
        self._last_visual_timestamp_ns = visual_timestamp_ns
        return WorldSnapshot(
            timestamp_ns=timestamp_ns,
            visual_timestamp_ns=visual_timestamp_ns,
            regions=self._regions,
            targets=tuple(targets),
            opponent_occupancies=tuple(
                self._opponents[key] for key in sorted(self._opponents)
            ),
            robot_field_point=robot_field_point,
            robot_region_kinds=robot_regions,
            uncertainties=frozenset(uncertainties),
        )

    def reset(self) -> None:
        self._opponents.clear()
        self._last_timestamp_ns = None
        self._last_visual_timestamp_ns = None

    def _validate_time(
        self,
        timestamp_ns: int,
        visual_timestamp_ns: int,
    ) -> None:
        for name, value in (
            ("timestamp_ns", timestamp_ns),
            ("visual_timestamp_ns", visual_timestamp_ns),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer.")
        if visual_timestamp_ns > timestamp_ns:
            raise ValueError("visual_timestamp_ns must not be in the future.")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError(
                f"timestamp_ns moved backwards from {self._last_timestamp_ns} "
                f"to {timestamp_ns}."
            )
        if (
            self._last_visual_timestamp_ns is not None
            and visual_timestamp_ns < self._last_visual_timestamp_ns
        ):
            raise ValueError(
                "visual_timestamp_ns moved backwards from "
                f"{self._last_visual_timestamp_ns} to {visual_timestamp_ns}."
            )

    def _hazard_state(self, track: TrackedTarget) -> HazardState:
        danger = track.class_probabilities.blue_danger
        unknown = track.class_probabilities.unknown
        if (
            track.ever_confirmed
            and danger >= self._config.danger_confirm_threshold
        ):
            return HazardState.CONFIRMED
        if (
            danger >= self._config.danger_suspect_threshold
            or unknown >= self._config.unknown_suspect_threshold
            or track.status is not TrackStatus.CONFIRMED
            or track.target_class is TargetClass.UNKNOWN
        ):
            return HazardState.SUSPECTED
        return HazardState.CLEAR
