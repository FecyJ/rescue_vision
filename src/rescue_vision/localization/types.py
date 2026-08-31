"""Center-cross localization contracts in the canonical field frame."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from rescue_vision.geometry.types import FieldPoint
from rescue_vision.world.static_map import CenterLineTerminalKind


def normalize_angle(angle_rad: float) -> float:
    """Normalize a finite angle to ``[-pi, pi]`` with a stable ``+pi`` edge."""

    converted = float(angle_rad)
    if not math.isfinite(converted):
        raise ValueError(f"angle_rad must be finite, got {angle_rad!r}.")
    normalized = (converted + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(normalized, -math.pi, abs_tol=1e-12) and converted > 0.0:
        return math.pi
    return normalized


def angular_distance(first_rad: float, second_rad: float) -> float:
    """Return the unsigned shortest angular distance in radians."""

    return abs(normalize_angle(first_rad - second_rad))


def _finite(value: float, location: str) -> float:
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return converted


def _non_negative(value: float, location: str) -> float:
    converted = _finite(value, location)
    if converted < 0.0:
        raise ValueError(f"{location} must be non-negative, got {value!r}.")
    return converted


def _probability(value: float, location: str) -> float:
    converted = _finite(value, location)
    if not 0.0 <= converted <= 1.0:
        raise ValueError(f"{location} must be in [0, 1], got {value!r}.")
    return converted


class CenterCrossSelectionSource(str, Enum):
    RED_SAFE_ZONE = "red_safe_zone"
    BLUE_SAFE_ZONE = "blue_safe_zone"
    RED_BLUE_SAFE_ZONES = "red_blue_safe_zones"
    STATIC_MAP_TERMINAL = "static_map_terminal"
    PRIOR = "prior"


class CenterCrossLocalizationQuality(str, Enum):
    MISSING_CENTER_CROSS = "missing_center_cross"
    PARTIAL_CENTER_CROSS = "partial_center_cross"
    NO_GROUND_PROJECTION = "no_ground_projection"
    STALE_OBSERVATION = "stale_observation"
    NO_DIRECTION_ANCHOR = "no_direction_anchor"
    BOUNDARY_ONLY_AMBIGUITY = "boundary_only_ambiguity"
    CONFLICTING_DIRECTION_ANCHORS = "conflicting_direction_anchors"
    PRIOR_INNOVATION_REJECTED = "prior_innovation_rejected"
    ANCHOR_PRIOR_CONFLICT = "anchor_prior_conflict"
    UNCONFIRMED_CENTER_CROSS = "unconfirmed_center_cross"


@dataclass(frozen=True, slots=True)
class CenterCrossLocalizerConfig:
    enabled: bool
    ray_min_forward_distance_mm: float
    ray_max_forward_distance_mm: float
    ray_max_lateral_distance_mm: float
    ray_angle_tolerance_deg: float
    min_anchor_confidence: float
    max_prior_heading_innovation_deg: float
    position_uncertainty_floor_mm: float
    heading_uncertainty_floor_deg: float

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise ValueError("enabled must be a boolean.")
        minimum = _non_negative(
            self.ray_min_forward_distance_mm,
            "ray_min_forward_distance_mm",
        )
        maximum = _non_negative(
            self.ray_max_forward_distance_mm,
            "ray_max_forward_distance_mm",
        )
        if maximum <= minimum:
            raise ValueError(
                "ray_max_forward_distance_mm must exceed "
                "ray_min_forward_distance_mm."
            )
        _non_negative(
            self.ray_max_lateral_distance_mm,
            "ray_max_lateral_distance_mm",
        )
        for name in (
            "ray_angle_tolerance_deg",
            "max_prior_heading_innovation_deg",
            "heading_uncertainty_floor_deg",
        ):
            value = _finite(getattr(self, name), name)
            if not 0.0 < value < 180.0:
                raise ValueError(f"{name} must be in (0, 180), got {value!r}.")
        _probability(self.min_anchor_confidence, "min_anchor_confidence")
        floor = _finite(
            self.position_uncertainty_floor_mm,
            "position_uncertainty_floor_mm",
        )
        if floor <= 0.0:
            raise ValueError("position_uncertainty_floor_mm must be positive.")


@dataclass(frozen=True, slots=True)
class FieldPose2D:
    """Robot pose in FieldPoint coordinates.

    ``heading_rad`` is the counter-clockwise angle from field ``+x`` to the
    robot forward axis.
    """

    position: FieldPoint
    heading_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.position, FieldPoint):
            raise ValueError("position must be a FieldPoint.")
        _finite(self.position.x, "position.x")
        _finite(self.position.y, "position.y")
        object.__setattr__(self, "heading_rad", normalize_angle(self.heading_rad))


@dataclass(frozen=True, slots=True)
class FieldPositionObservation:
    """已知场地单点在新鲜航向先验下形成的位置观测，不含航向测量。"""

    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    position: FieldPoint
    position_uncertainty_mm: float
    confidence: float
    source: str

    def __post_init__(self) -> None:
        for name in ("frame_sequence", "capture_timestamp_ns", "result_timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.result_timestamp_ns < self.capture_timestamp_ns:
            raise ValueError("result timestamp must not precede capture timestamp.")
        if not isinstance(self.position, FieldPoint):
            raise ValueError("position must be a FieldPoint.")
        _finite(self.position.x, "position.x")
        _finite(self.position.y, "position.y")
        if _finite(self.position_uncertainty_mm, "position_uncertainty_mm") <= 0.0:
            raise ValueError("position_uncertainty_mm must be positive.")
        _probability(self.confidence, "confidence")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("source must be a non-empty string.")


@dataclass(frozen=True, slots=True)
class CenterLineTerminalObservation:
    direction_forward: float
    direction_left: float
    kind: CenterLineTerminalKind
    distance_mm: float | None
    confidence: float

    def __post_init__(self) -> None:
        forward = _finite(self.direction_forward, "direction_forward")
        left = _finite(self.direction_left, "direction_left")
        norm = math.hypot(forward, left)
        if not math.isclose(norm, 1.0, abs_tol=1e-6):
            raise ValueError(
                "terminal direction must be a unit vector, "
                f"got norm {norm!r}."
            )
        if not isinstance(self.kind, CenterLineTerminalKind):
            raise ValueError("kind must be a CenterLineTerminalKind.")
        if self.distance_mm is not None:
            _non_negative(self.distance_mm, "distance_mm")
        _probability(self.confidence, "confidence")


@dataclass(frozen=True, slots=True)
class CenterCrossPoseCandidate:
    pose: FieldPose2D
    quarter_turn_index: int
    position_uncertainty_mm: float
    heading_uncertainty_rad: float

    def __post_init__(self) -> None:
        if not isinstance(self.pose, FieldPose2D):
            raise ValueError("pose must be a FieldPose2D.")
        if (
            isinstance(self.quarter_turn_index, bool)
            or not isinstance(self.quarter_turn_index, int)
            or not 0 <= self.quarter_turn_index <= 3
        ):
            raise ValueError("quarter_turn_index must be an integer in [0, 3].")
        if _finite(self.position_uncertainty_mm, "position_uncertainty_mm") <= 0.0:
            raise ValueError("position_uncertainty_mm must be positive.")
        if _finite(self.heading_uncertainty_rad, "heading_uncertainty_rad") <= 0.0:
            raise ValueError("heading_uncertainty_rad must be positive.")


@dataclass(frozen=True, slots=True)
class CenterCrossPoseObservation:
    frame_sequence: int
    capture_timestamp_ns: int
    result_timestamp_ns: int
    candidates: tuple[CenterCrossPoseCandidate, ...]
    terminals: tuple[CenterLineTerminalObservation, ...]
    selected_pose: FieldPose2D | None
    selection_source: CenterCrossSelectionSource | None
    confidence: float
    quality: frozenset[CenterCrossLocalizationQuality]

    def __post_init__(self) -> None:
        for name in (
            "frame_sequence",
            "capture_timestamp_ns",
            "result_timestamp_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.result_timestamp_ns < self.capture_timestamp_ns:
            raise ValueError(
                "result_timestamp_ns must not precede capture_timestamp_ns."
            )
        if len(self.candidates) not in {0, 4}:
            raise ValueError("candidates must be empty or contain four poses.")
        if not all(
            isinstance(item, CenterCrossPoseCandidate) for item in self.candidates
        ):
            raise ValueError("candidates contains an invalid value.")
        if not all(
            isinstance(item, CenterLineTerminalObservation)
            for item in self.terminals
        ):
            raise ValueError("terminals contains an invalid value.")
        if self.selected_pose is not None and not isinstance(
            self.selected_pose,
            FieldPose2D,
        ):
            raise ValueError("selected_pose must be a FieldPose2D or None.")
        if (self.selected_pose is None) != (self.selection_source is None):
            raise ValueError(
                "selected_pose and selection_source must both be present or absent."
            )
        if self.selection_source is not None and not isinstance(
            self.selection_source,
            CenterCrossSelectionSource,
        ):
            raise ValueError("selection_source is invalid.")
        _probability(self.confidence, "confidence")
        if not all(
            isinstance(item, CenterCrossLocalizationQuality) for item in self.quality
        ):
            raise ValueError("quality contains an invalid value.")
