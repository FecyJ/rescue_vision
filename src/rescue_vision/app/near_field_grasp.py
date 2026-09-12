"""近场物资选组与近似动作包络；不访问设备、不解释首轮交付。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from itertools import combinations
import math
from typing import cast

from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.app.breakup_planner import physical_radii
from rescue_vision.perception.target_ground_geometry import TargetGroundGeometryConfig
from rescue_vision.geometry.types import GroundPoint, UndistortedPixel
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.motion.gripper_kinematics import GripperKinematics
from rescue_vision.perception.detector import MODEL_GROUND_FORWARD_BIAS_MM
from rescue_vision.perception.gripper_width import TargetGroundEnvelope, measure_target_envelope
from rescue_vision.perception.types import TargetClass, TargetObservation
from rescue_vision.tracking import MultiTargetTracker, TrackStatus

__all__ = [
    "CandidateGeometry",
    "DEFAULT_NEAR_FIELD_POLICY",
    "GraspScore",
    "GraspSelection",
    "GraspTarget",
    "GraspTargetTracker",
    "NearFieldHandoffPrior",
    "NearFieldGraspPolicy",
    "NearFieldGraspPlan",
    "NearFieldGraspSelector",
    "polygon_distance",
]

SUPPLIES = frozenset((TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE))
GRASPABLE_CLASSES = frozenset((*SUPPLIES, TargetClass.ORANGE_INJURED))


@dataclass(frozen=True, slots=True)
class NearFieldHandoffPrior:
    """远场已确认目标交给近场决策层的最小静态先验。"""

    target_class: TargetClass
    ground_point: GroundPoint
    source_track_id: int | None = None

    def __post_init__(self) -> None:
        if self.target_class not in GRASPABLE_CLASSES:
            raise ValueError(
                "target_class must be a green_supply, black_core, or "
                f"orange_injured value, got {self.target_class!r}."
            )
        if not isinstance(self.ground_point, GroundPoint) or not all(
            math.isfinite(value)
            for value in (self.ground_point.x, self.ground_point.y)
        ):
            raise ValueError(
                "ground_point must contain finite GroundPoint coordinates."
            )
        if self.source_track_id is not None and (
            isinstance(self.source_track_id, bool)
            or not isinstance(self.source_track_id, int)
            or self.source_track_id <= 0
        ):
            raise ValueError("source_track_id must be a positive integer or None.")

    def matches(
        self,
        observation: TargetObservation,
        *,
        max_distance_mm: float,
    ) -> bool:
        """判断当前观测是否仍像远场交接的同类目标。"""

        if not isinstance(observation, TargetObservation):
            raise TypeError("observation must be a TargetObservation.")
        if (
            isinstance(max_distance_mm, bool)
            or not isinstance(max_distance_mm, (int, float))
            or not math.isfinite(float(max_distance_mm))
            or float(max_distance_mm) < 0.0
        ):
            raise ValueError("max_distance_mm must be finite and non-negative.")
        point = observation.ground_point
        return (
            observation.target_class is self.target_class
            and observation.model_target_class is self.target_class
            and point is not None
            and math.hypot(
                point.x - self.ground_point.x,
                point.y - self.ground_point.y,
            )
            <= float(max_distance_mm)
        )


@dataclass(frozen=True, slots=True)
class NearFieldGraspPolicy:
    """一次近场动作允许选择的类别和成员容量。"""

    allowed_classes: frozenset[TargetClass]
    max_targets: int

    def __post_init__(self) -> None:
        if not isinstance(self.allowed_classes, frozenset) or not self.allowed_classes:
            raise ValueError("allowed_classes must be a non-empty frozenset.")
        if not self.allowed_classes <= GRASPABLE_CLASSES:
            raise ValueError(
                "allowed_classes must be a subset of green_supply/black_core/orange_injured."
            )
        if (
            isinstance(self.max_targets, bool)
            or not isinstance(self.max_targets, int)
            or not 1 <= self.max_targets <= 3
        ):
            raise ValueError("max_targets must be an integer in [1, 3].")


DEFAULT_NEAR_FIELD_POLICY = NearFieldGraspPolicy(GRASPABLE_CLASSES, 3)


def _rotate(point: GroundPoint, angle: float) -> GroundPoint:
    c, s = math.cos(angle), math.sin(angle)
    return GroundPoint(c * point.x - s * point.y, s * point.x + c * point.y)


def _rectangle(x0: float, x1: float, y0: float, y1: float) -> tuple[GroundPoint, ...]:
    return (GroundPoint(x0, y0), GroundPoint(x1, y0), GroundPoint(x1, y1), GroundPoint(x0, y1))


def _bounds(points: tuple[GroundPoint, ...]) -> tuple[float, float, float, float]:
    return min(p.x for p in points), max(p.x for p in points), min(p.y for p in points), max(p.y for p in points)


def _segment_distance(p: GroundPoint, a: GroundPoint, b: GroundPoint) -> float:
    dx, dy = b.x - a.x, b.y - a.y
    length = dx * dx + dy * dy
    t = 0.0 if length == 0 else max(0.0, min(1.0, ((p.x - a.x) * dx + (p.y - a.y) * dy) / length))
    return math.hypot(p.x - a.x - t * dx, p.y - a.y - t * dy)


def polygon_distance(a: tuple[GroundPoint, ...], b: tuple[GroundPoint, ...]) -> float:
    """凸包距离，包含接触和包含关系；毫米。"""
    # 分离轴判定，包括退化点/线段，避免 intersectConvexConvex 的边界误差。
    separated = False
    for polygon in (a, b):
        for start, end in zip(polygon, polygon[1:] + polygon[:1]):
            nx, ny = end.y - start.y, start.x - end.x
            if nx == ny == 0:
                continue
            pa = [p.x * nx + p.y * ny for p in a]
            pb = [p.x * nx + p.y * ny for p in b]
            if max(pa) < min(pb) or max(pb) < min(pa):
                separated = True
    if not separated:
        return 0.0
    return min(_segment_distance(p, start, end)
               for points, polygon in ((a, b), (b, a)) for p in points
               for start, end in zip(polygon, polygon[1:] + polygon[:1]))


@dataclass(frozen=True, slots=True)
class GraspTarget:
    track_id: int
    observation: TargetObservation
    envelope: TargetGroundEnvelope | None
    confirmed: bool
    selectable: bool
    observed: bool = True
    handoff_matched: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id <= 0:
            raise ValueError(f"Invalid track_id {self.track_id!r}.")
        if not isinstance(self.observation, TargetObservation):
            raise TypeError("observation must be a TargetObservation.")
        for name in ("confirmed", "selectable", "observed", "handoff_matched"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.envelope is not None and not isinstance(self.envelope, TargetGroundEnvelope):
            raise TypeError("envelope must be a TargetGroundEnvelope or None.")
        if self.envelope is not None and (self.envelope.frame_sequence != self.observation.frame_sequence or self.envelope.capture_timestamp_ns != self.observation.capture_timestamp_ns):
            raise ValueError("Envelope and observation must belong to the same frame.")
        if self.envelope is not None and self.envelope.target_class is not self.observation.target_class:
            raise ValueError("Envelope and observation target classes must match.")


class GraspTargetTracker:
    """复用 tracker 的关联结果；危险历史在轨迹存续期内不因漏检清空。"""

    def __init__(self, tracker: MultiTargetTracker, projector: GroundProjector, config: NearFieldGraspConfig, *, max_relative_speed_mm_s: float = 1000.0):
        if not isinstance(tracker, MultiTargetTracker):
            raise TypeError("tracker must be a MultiTargetTracker.")
        if not isinstance(projector, GroundProjector):
            raise TypeError("projector must be a GroundProjector.")
        if not isinstance(config, NearFieldGraspConfig):
            raise TypeError("config must be a NearFieldGraspConfig.")
        self.tracker, self.projector, self.config = tracker, projector, config
        # 近场停车窗口以 K0 空间连续性为主；一次类别抖动不应把同一物块
        # 重新编号。危险/质量门禁仍在本类按原始观测单独维护，故不会因
        # 这个关联策略而把明确危险目标变成可抓目标。
        self.tracker.enable_soft_class_association()
        self.tracker.enable_low_confidence_retention()
        if not math.isfinite(max_relative_speed_mm_s) or max_relative_speed_mm_s <= 0:
            raise ValueError(f"Invalid max_relative_speed_mm_s {max_relative_speed_mm_s!r}.")
        self.max_relative_speed_mm_s = max_relative_speed_mm_s
        self._memory: dict[int, GraspTarget] = {}
        self._forbidden: set[int] = set()
        self._last_timestamp_ns = -1
        self._handoff_prior: NearFieldHandoffPrior | None = None
        self._handoff_track_id: int | None = None

    def set_handoff_prior(self, prior: NearFieldHandoffPrior | None) -> None:
        """设置仅用于当前近场决策会话的远场目标先验。"""

        if prior is not None and not isinstance(prior, NearFieldHandoffPrior):
            raise TypeError("prior must be a NearFieldHandoffPrior or None.")
        if prior != self._handoff_prior:
            self._handoff_track_id = None
        self._handoff_prior = prior

    def reset(self) -> None:
        """清空本轮近场关联和历史几何。"""

        self.tracker.reset()
        self._memory.clear()
        self._forbidden.clear()
        self._last_timestamp_ns = -1
        self._handoff_prior = None
        self._handoff_track_id = None

    def update(self, timestamp_ns: int, observations: tuple[TargetObservation, ...] | list[TargetObservation]) -> tuple[GraspTarget, ...]:
        observations = tuple(observations)
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
            raise ValueError(f"timestamp_ns must be a non-negative integer, got {timestamp_ns!r}.")
        if not all(isinstance(observation, TargetObservation) for observation in observations):
            raise TypeError("observations must contain TargetObservation values.")
        if timestamp_ns <= self._last_timestamp_ns:
            raise ValueError(f"Expected increasing capture timestamp, got {timestamp_ns}.")
        self._last_timestamp_ns = timestamp_ns
        # 与底层 tracker 使用同一去重结果，避免被 tracker 压掉的重复检测又在
        # 本层作为“未关联障碍”重新加入。
        observations = self.tracker.deduplicate_observations(observations)
        tracks = self.tracker.update(timestamp_ns, observations)
        live_ids = {t.track_id for t in tracks}
        if self._handoff_track_id not in live_ids:
            self._handoff_track_id = None
        self._forbidden.intersection_update(live_ids)
        handoff_observation = None
        if self._handoff_prior is not None and self._handoff_track_id is None:
            matching_observations = tuple(
                observation
                for observation in observations
                if self._handoff_prior.matches(
                    observation,
                    max_distance_mm=self.tracker.config.max_association_ground_mm,
                )
            )
            handoff_observation = min(
                matching_observations,
                key=lambda observation: (
                    math.hypot(
                        cast(GroundPoint, observation.ground_point).x
                        - self._handoff_prior.ground_point.x,
                        cast(GroundPoint, observation.ground_point).y
                        - self._handoff_prior.ground_point.y,
                    ),
                    observation.frame_sequence,
                    observation.box.x_min,
                    observation.box.y_min,
                ),
                default=None,
            )
        result = []
        unused = list(observations)
        for track in tracks:
            if track.last_seen_timestamp_ns != timestamp_ns:
                previous = self._memory.get(track.track_id)
                if previous is not None:
                    envelope = previous.envelope
                    if envelope is not None:
                        age_s = (timestamp_ns - previous.observation.capture_timestamp_ns) / 1e9
                        x0, x1, y0, y1 = _bounds(envelope.corners)
                        pad = age_s * self.max_relative_speed_mm_s
                        envelope = replace(envelope, corners=_rectangle(x0-pad, x1+pad, y0-pad, y1+pad))
                    result.append(replace(previous, envelope=envelope, confirmed=False, observed=False))
                continue
            # tracker 保存的是本帧匹配观测的 box/K0，唯一消费一次，不再独立做关联。
            index = next((i for i, obs in enumerate(unused) if obs.box == track.box and obs.k0 == track.k0), None)
            if index is None:
                raise RuntimeError(f"Cannot resolve current observation for track {track.track_id}.")
            obs = unused.pop(index)
            # 一个远场目标只能交接给一个局部目标。关联门限内若有多个同类
            # 观测，选择离远场 K0 最近者，避免多个近场 ID 同时获得交接优先级。
            if obs is handoff_observation:
                self._handoff_track_id = track.track_id
            handoff_matched = track.track_id == self._handoff_track_id
            explicit_forbidden = (
                obs.target_class is TargetClass.BLUE_DANGER
                or obs.model_target_class is TargetClass.BLUE_DANGER
            )
            clean_graspable = (
                obs.target_class in GRASPABLE_CLASSES
                and obs.model_target_class in GRASPABLE_CLASSES
                and obs.ground_point is not None
                and track.confidence >= self.tracker.config.min_confidence
            )
            if explicit_forbidden:
                # 明确危险证据在轨迹存续期内保持，不能被后续类别抖动解除。
                self._forbidden.add(track.track_id)
            try:
                envelope = measure_target_envelope(obs, self.projector, min_mask_pixels=self.config.min_mask_pixels)
            except ValueError:
                envelope = None
            result.append(GraspTarget(
                track.track_id,
                obs,
                envelope,
                track.status is TrackStatus.CONFIRMED,
                track.track_id not in self._forbidden
                and clean_graspable,
                handoff_matched=handoff_matched,
            ))
        self._memory = {
            t.track_id: (t if t.observed else self._memory.get(t.track_id, t))
            for t in result
        }
        # tracker可能丢弃低置信新轨迹；这些检测仍作为不可选障碍，不能消失。
        for index, obs in enumerate(unused):
            try:
                envelope = measure_target_envelope(
                    obs,
                    self.projector,
                    min_mask_pixels=self.config.min_mask_pixels,
                )
            except ValueError:
                envelope = None
            result.append(
                GraspTarget(
                    1_000_000_000 + index,
                    obs,
                    envelope,
                    False,
                    False,
                )
            )
        return tuple(result)


@dataclass(frozen=True, slots=True)
class CandidateGeometry:
    """单个候选目标的横向开口与前进行程诊断，单位均为 mm。

    ``x0_mm``/``x1_mm``/``depth_mm`` 只用于观察颜色掩码投影的纵向范围，
    不再作为目标自身的容纳硬约束。前进行程一律由最远 K0 的 ``x`` 决定：
    绿/黑组减去 ``target_final_x_mm``，单个橙色目标减去
    ``orange_target_final_x_mm``。``corridor_end_x_mm`` 与实际扫掠走廊
    使用同一公式，包含夹爪末端前伸距离。
    """

    track_id: int
    target_class: TargetClass
    frame_sequence: int
    capture_timestamp_ns: int
    k0_x_mm: float | None
    x0_mm: float | None
    x1_mm: float | None
    depth_mm: float | None
    opening_width_mm: float | None
    target_final_x_mm: float | None
    corridor_start_x_mm: float | None
    corridor_end_x_mm: float | None
    corridor_half_width_mm: float | None
    forward_distance_mm: float | None
    alignment_angle_rad: float | None
    eligible: bool
    reason: str | None
    handoff_matched: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.track_id, bool) or not isinstance(self.track_id, int) or self.track_id <= 0:
            raise ValueError("track_id must be a positive integer.")
        if not isinstance(self.target_class, TargetClass):
            raise ValueError("target_class must be a TargetClass value.")
        for name in ("frame_sequence", "capture_timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        for name in (
            "k0_x_mm",
            "x0_mm",
            "x1_mm",
            "depth_mm",
            "opening_width_mm",
            "target_final_x_mm",
            "corridor_start_x_mm",
            "corridor_end_x_mm",
            "corridor_half_width_mm",
            "forward_distance_mm",
            "alignment_angle_rad",
        ):
            value = getattr(self, name)
            if value is not None:
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                    raise ValueError(f"{name} must be finite when present.")
                object.__setattr__(self, name, float(value))
        if not isinstance(self.eligible, bool):
            raise ValueError("eligible must be a boolean.")
        if not isinstance(self.handoff_matched, bool):
            raise ValueError("handoff_matched must be a boolean.")
        if self.reason is not None and (not isinstance(self.reason, str) or not self.reason):
            raise ValueError("reason must be a non-empty string or None.")

    def as_log_line(self) -> str:
        """返回稳定的单行诊断文本，便于现场 grep/导入。"""

        def number(value: float | None) -> str:
            return "none" if value is None else f"{value:.2f}"

        return (
            "grasp_candidate "
            f"track_id={self.track_id} class={self.target_class.value} "
            f"frame={self.frame_sequence} capture_timestamp_ns={self.capture_timestamp_ns} "
            f"eligible={str(self.eligible).lower()} "
            f"x0_mm={number(self.x0_mm)} x1_mm={number(self.x1_mm)} "
            f"depth_mm={number(self.depth_mm)} k0_x_mm={number(self.k0_x_mm)} "
            f"opening_mm={number(self.opening_width_mm)} "
            f"target_final_x_mm={number(self.target_final_x_mm)} "
            f"corridor_start_x_mm={number(self.corridor_start_x_mm)} "
            f"corridor_end_x_mm={number(self.corridor_end_x_mm)} "
            f"corridor_half_width_mm={number(self.corridor_half_width_mm)} "
            f"forward_distance_mm={number(self.forward_distance_mm)} "
            f"alignment_angle_rad={number(self.alignment_angle_rad)} "
            f"handoff_prior_match={str(self.handoff_matched).lower()} "
            f"reason={self.reason or 'eligible'}"
        )


@dataclass(frozen=True, slots=True)
class GraspScore:
    rule_points: float
    orange_priority: float
    count: float
    clearance: float
    distance: float
    alignment: float
    total: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.rule_points, bool)
            or not isinstance(self.rule_points, (int, float))
            or not math.isfinite(float(self.rule_points))
            or self.rule_points <= 0.0
        ):
            raise ValueError("score.rule_points must be finite and positive.")
        object.__setattr__(self, "rule_points", float(self.rule_points))
        for name in ("orange_priority", "count", "clearance", "distance", "alignment", "total"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"score.{name} must be a number, got {value!r}.")
            value = float(value)
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"score.{name} must be finite and in [0, 1], got {value!r}.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class NearFieldGraspPlan:
    frame_sequence: int
    capture_timestamp_ns: int
    members: tuple[GraspTarget, ...]
    alignment_point: GroundPoint
    alignment_angle_rad: float
    bounds: tuple[GroundPoint, ...]  # 计划对准后的机器人系包络
    opening_width_mm: float
    maximum_opening_mm: float
    opening_servo_angles_deg: tuple[float, float]
    forward_distance_mm: float
    clearance_mm: float
    score: GraspScore
    regions: tuple[tuple[GroundPoint, ...], ...]  # 当前机器人系动作包络

    def __post_init__(self) -> None:
        for name in ("frame_sequence", "capture_timestamp_ns"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
        if not isinstance(self.members, tuple) or not self.members:
            raise ValueError("members must be a non-empty tuple.")
        if not all(isinstance(item, GraspTarget) for item in self.members) or len({item.track_id for item in self.members}) != len(self.members):
            raise ValueError("members must contain unique GraspTarget values.")
        if any(item.observation.frame_sequence != self.frame_sequence or item.observation.capture_timestamp_ns != self.capture_timestamp_ns for item in self.members):
            raise ValueError("plan members must come from the plan frame and capture timestamp.")
        if not isinstance(self.alignment_point, GroundPoint):
            raise TypeError("alignment_point must be a GroundPoint.")
        if not all(math.isfinite(value) for value in (self.alignment_point.x, self.alignment_point.y)):
            raise ValueError("alignment_point must contain finite coordinates.")
        for name in ("alignment_angle_rad", "opening_width_mm", "maximum_opening_mm", "forward_distance_mm", "clearance_mm"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"{name} must be finite, got {value!r}.")
            value = float(value)
            if name == "clearance_mm":
                if math.isnan(value) or value < 0.0:
                    raise ValueError(f"{name} must be non-negative or +inf, got {value!r}.")
            elif not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")
            if name != "alignment_angle_rad" and value < 0:
                raise ValueError(f"{name} must be non-negative, got {value!r}.")
            object.__setattr__(self, name, value)
        if self.opening_width_mm > self.maximum_opening_mm + 1e-9:
            raise ValueError("opening_width_mm cannot exceed maximum_opening_mm.")
        if not isinstance(self.bounds, tuple) or len(self.bounds) < 4 or not all(isinstance(p, GroundPoint) and math.isfinite(p.x) and math.isfinite(p.y) for p in self.bounds):
            raise ValueError("bounds must contain at least four GroundPoint values.")
        if not isinstance(self.regions, tuple) or not self.regions or not all(isinstance(region, tuple) and region and all(isinstance(p, GroundPoint) and math.isfinite(p.x) and math.isfinite(p.y) for p in region) for region in self.regions):
            raise ValueError("regions must contain non-empty GroundPoint tuples.")
        if not isinstance(self.opening_servo_angles_deg, tuple) or len(self.opening_servo_angles_deg) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not 0.0 <= float(v) <= 180.0 for v in self.opening_servo_angles_deg):
            raise ValueError("opening_servo_angles_deg must contain two angles in [0, 180].")
        if not isinstance(self.score, GraspScore):
            raise TypeError("score must be a GraspScore.")

    @property
    def member_ids(self) -> tuple[int, ...]:
        return tuple(t.track_id for t in self.members)

    @property
    def width_mm(self) -> float:
        _, _, y0, y1 = _bounds(self.bounds)
        return y1 - y0


@dataclass(frozen=True, slots=True)
class GraspSelection:
    """Selected valid plan plus diagnostics for rejected candidates.

    With locked_ids, rejections refer only to the locked group. Without a lock,
    a valid plan may coexist with rejection diagnostics for other candidates.
    """

    plan: NearFieldGraspPlan | None
    rejections: tuple[str, ...]
    preview_plan: NearFieldGraspPlan | None = None

    def __post_init__(self) -> None:
        if self.plan is not None and not isinstance(self.plan, NearFieldGraspPlan):
            raise TypeError("plan must be a NearFieldGraspPlan or None.")
        if self.preview_plan is not None and not isinstance(
            self.preview_plan, NearFieldGraspPlan
        ):
            raise TypeError("preview_plan must be a NearFieldGraspPlan or None.")
        if not isinstance(self.rejections, tuple) or not all(isinstance(item, str) and item for item in self.rejections):
            raise ValueError("rejections must contain non-empty strings.")


class NearFieldGraspSelector:
    def __init__(self, config: NearFieldGraspConfig, projector: GroundProjector,
                 kinematics: GripperKinematics, *, open_servo_angles_deg: tuple[float, float],
                 closed_servo_angles_deg: tuple[float, float], target_geometry: TargetGroundGeometryConfig):
        if not isinstance(config, NearFieldGraspConfig):
            raise TypeError("config must be a NearFieldGraspConfig.")
        if not isinstance(projector, GroundProjector):
            raise TypeError("projector must be a GroundProjector.")
        if not isinstance(kinematics, GripperKinematics):
            raise TypeError("kinematics must be a GripperKinematics.")
        for name, angles in (("open_servo_angles_deg", open_servo_angles_deg), ("closed_servo_angles_deg", closed_servo_angles_deg)):
            if not isinstance(angles, tuple) or len(angles) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0.0 <= float(value) <= 180.0 for value in angles):
                raise ValueError(f"{name} must contain two angles in [0, 180].")
        if not isinstance(target_geometry, TargetGroundGeometryConfig):
            raise TypeError("target_geometry must be TargetGroundGeometryConfig")
        self.target_geometry = target_geometry
        self.config, self.projector, self.kinematics = config, projector, kinematics
        self.open_angles, self.closed_angles = open_servo_angles_deg, closed_servo_angles_deg
        # 反解同时验证舵机方向、端点和值域。
        self.servo_angles(0.0)
        self.max_left_angle = self.closed_angles[0] - self.open_angles[0]
        self.max_right_angle = self.open_angles[1] - self.closed_angles[1]
        if not 0.0 < self.max_left_angle <= 90.0:
            raise ValueError(
                "Configured left gripper travel must provide a relative angle "
                f"in (0, 90], got {self.max_left_angle!r}."
            )
        if not 0.0 < self.max_right_angle <= 90.0:
            raise ValueError(
                "Configured right gripper travel must provide a relative angle "
                f"in (0, 90], got {self.max_right_angle!r}."
            )
        self.max_angle = min(self.max_left_angle, self.max_right_angle)
        self.maximum_opening_mm = (
            kinematics.left_tip_position(self.max_left_angle).y
            - kinematics.right_tip_position(self.max_right_angle).y
        )

    @property
    def default_policy(self) -> NearFieldGraspPolicy:
        return NearFieldGraspPolicy(GRASPABLE_CLASSES, self.config.max_targets)

    def servo_angles(self, width: float) -> tuple[float, float]:
        return self.kinematics.servo_angles_for_opening(width,
            open_left_angle_deg=self.open_angles[0], open_right_angle_deg=self.open_angles[1],
            closed_left_angle_deg=self.closed_angles[0], closed_right_angle_deg=self.closed_angles[1])

    def servo_angles_for_edges(
        self,
        left_tip_y_mm: float,
        right_tip_y_mm: float,
    ) -> tuple[float, float]:
        """分别把左右夹爪末端的目标横向位置反解为舵机角度。"""

        return self.kinematics.servo_angles_for_edge_positions(
            left_tip_y_mm,
            right_tip_y_mm,
            open_left_angle_deg=self.open_angles[0],
            open_right_angle_deg=self.open_angles[1],
            closed_left_angle_deg=self.closed_angles[0],
            closed_right_angle_deg=self.closed_angles[1],
        )

    @staticmethod
    def _validate_alignment_tolerance(value: float) -> float:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or float(value) <= 0.0
        ):
            raise ValueError(
                "alignment_tolerance_mm must be finite and positive, "
                f"got {value!r}."
            )
        return float(value)

    @staticmethod
    def _eligible(
        target: GraspTarget,
        policy: NearFieldGraspPolicy | None = None,
    ) -> bool:
        """应用层再次执行类别/质量门禁，避免手工构造对象绕过 tracker。"""
        if policy is None:
            policy = DEFAULT_NEAR_FIELD_POLICY
        if not isinstance(policy, NearFieldGraspPolicy):
            raise TypeError("policy must be a NearFieldGraspPolicy or None.")
        observation = target.observation
        return (
            target.selectable
            and (target.confirmed or target.handoff_matched)
            and target.observed
            and target.envelope is not None
            and observation.target_class in policy.allowed_classes
            and observation.model_target_class is not TargetClass.BLUE_DANGER
            and not observation.quality
        )

    def _orange_isolation_rejection(
        self,
        target: GraspTarget,
        targets: tuple[GraspTarget, ...],
        *,
        align: bool = True,
    ) -> str | None:
        """检查橙色伤员周围的独立性禁入区。"""

        if target.observation.target_class is not TargetClass.ORANGE_INJURED:
            return None
        if target.envelope is None:
            return f"orange_isolation_unknown_ground_track:{target.track_id}"
        center = target.envelope.center
        radius = self.config.orange_isolation_radius_mm
        for other in targets:
            if other.track_id == target.track_id:
                continue
            # 只有当前帧可定位的其它目标才能证明其落入 50 mm 禁区。
            # 失观轨迹或缺 K0 检测不能被无条件推定在橙色目标旁边。
            if not other.observed or other.observation.ground_point is None:
                continue
            other_point = other.observation.ground_point
            distance_mm = math.hypot(
                center.x - other_point.x,
                center.y - other_point.y,
            )
            if distance_mm <= radius + 1e-9:
                if self._orange_side_rear_supply_clear(
                    target,
                    other,
                    align=align,
                ):
                    continue
                return (
                    "orange_not_isolated_track:"
                    f"{other.track_id}:distance_mm={distance_mm:.1f}"
                )
        return None

    def _orange_side_rear_supply_clear(
        self,
        orange: GraspTarget,
        other: GraspTarget,
        *,
        align: bool,
    ) -> bool:
        """圆形近邻区内仅放行有完整当前包络证明位于扫掠外的侧后方物资。

        危险/未知、缺包络、前方或贴邻目标仍执行原隔离门禁；不减小隔离半径。
        """
        if (other.observation.target_class not in SUPPLIES
                or not self._eligible(other, self.default_policy)
                or other.envelope is None or orange.envelope is None):
            return False
        try:
            geometries = [self._group_geometry(
                (orange,), align=align,
                range_limit_mm=self.config.max_range_mm + self.config.range_hysteresis_mm,
            )]
            if (
                align
                and abs(orange.envelope.center.y)
                <= self.config.center_tolerance_mm
                + self.config.alignment_hysteresis_mm
            ):
                geometries.append(self._group_geometry(
                    (orange,), align=False,
                    range_limit_mm=self.config.max_range_mm + self.config.range_hysteresis_mm,
                ))
        except ValueError:
            return False
        if polygon_distance(orange.envelope.corners, other.envelope.corners) <= self.config.clearance_mm:
            return False
        margin = self.config.corridor_lateral_margin_mm + self.config.clearance_mm
        for geometry in geometries:
            if geometry.reasons:
                return False
            points = tuple(_rotate(point, -geometry.angle_rad) for point in other.envelope.corners)
            x0, _, y0, y1 = _bounds(points)
            orange_center = _rotate(orange.envelope.center, -geometry.angle_rad)
            outside_sweep = (y0 > geometry.left_tip_y_mm + margin
                             or y1 < geometry.right_tip_y_mm - margin)
            if x0 <= orange_center.x or not outside_sweep:
                return False
        return True

    def _gripper_tip_x_mm(self, servo_angles_deg: tuple[float, float]) -> float:
        """返回给定舵机角度下夹爪末端的前向 ``x``，含走廊起点下限。"""

        left = self.kinematics.left_tip_position(
            self.closed_angles[0] - servo_angles_deg[0],
        )
        right = self.kinematics.right_tip_position(
            servo_angles_deg[1] - self.closed_angles[1],
        )
        return max(left.x, right.x, self.config.corridor_start_x_mm)

    def _sweep_end_x_mm(
        self,
        servo_angles_deg: tuple[float, float],
        distance: float,
    ) -> float:
        """返回夹爪开口前进 ``distance`` 后的扫掠前端 ``x``。

        执行走廊与 ``CandidateGeometry.corridor_end_x_mm`` 诊断共用本公式，
        夹爪末端前伸距离不会被漏算成只有 ``corridor_start_x_mm`` 加行程。
        """

        return self._gripper_tip_x_mm(servo_angles_deg) + distance

    def _regions(
        self,
        angle: float,
        distance: float,
        left_tip_y_mm: float,
        right_tip_y_mm: float,
    ) -> tuple[tuple[GroundPoint, ...], ...]:
        """返回夹爪开口随前进扫过的单个矩形走廊。

        目标物资之间可以在揽入过程中重新排列，因此目标自身的纵向包络不
        参与门禁。走廊只回答一个问题：前进这段距离时，非目标关键点是否
        会进入夹爪张开的横向范围。
        """

        servo = self.servo_angles_for_edges(left_tip_y_mm, right_tip_y_mm)
        # Sweep reaches the physical fingertips, not just corridor_start + travel.
        corridor = _rectangle(
            self.config.corridor_start_x_mm,
            self._sweep_end_x_mm(servo, distance),
            right_tip_y_mm - self.config.corridor_lateral_margin_mm,
            left_tip_y_mm + self.config.corridor_lateral_margin_mm,
        )
        return (tuple(_rotate(point, angle) for point in corridor),)

    def region_pixels(self, region: tuple[GroundPoint, ...]) -> tuple[UndistortedPixel, ...]:
        # 与观测侧的唯一部署前向偏差成对抵消，不复制投影矩阵。
        return tuple(self.projector.ground_to_pixels(tuple(GroundPoint(p.x - MODEL_GROUND_FORWARD_BIAS_MM, p.y) for p in region)))

    def _obstacle_distance(self, target: GraspTarget, regions: tuple[tuple[GroundPoint, ...], ...]) -> float:
        # 停车后的近场走廊只使用当前帧可靠 K0。历史失观轨迹的膨胀包络
        # 会把侧方静态目标扩张成横跨走廊的幽灵障碍；当前帧缺少地面点时
        # 同样不能用旧包络或旧像素框代替当前空间证据。
        if target.observed and target.observation.ground_point is not None:
            point = (target.observation.ground_point,)
            gap = min(polygon_distance(point, region) for region in regions)
            if target.observation.target_class is TargetClass.BLUE_DANGER:
                # 蓝块有实体：中心在路径外也可能被夹到，所以按实体半径收缩
                # 净空。取内切半径——一定被实体占据的圆盘——与解团接触判定
                # 同一条原则“内切半径才成立”；外接半径会把只是近旁、并不在
                # 夹取路径上的蓝块当成阻挡，制造不必要的换组和解团。蓝块实体
                # 始终参与扫掠检查，危险证据不因此删除。
                radius, _ = physical_radii(self.target_geometry.geometry_for(TargetClass.BLUE_DANGER))
                gap = max(0.0, gap - radius)
            return gap
        return math.inf

    @staticmethod
    def _target_ground_center(target: GraspTarget) -> GroundPoint | None:
        """返回当前目标的 K0 中心，优先使用颜色包络中心。"""

        if target.envelope is not None:
            return target.envelope.center
        return target.observation.ground_point

    def _side_neighbor_metrics(
        self,
        plan: NearFieldGraspPlan,
        target: GraspTarget,
    ) -> tuple[float, float] | None:
        """返回目标与计划成员在预测抓取轴下的最小中心差。

        该门禁只回答“是否侧向紧邻且纵向基本齐平”。目标明显位于
        计划成员前方或后方时，纵向差会超过门限并放行；实际前进段的
        走廊阻挡仍由 ``_obstacle_distance`` 单独检查。
        """

        if not target.observed:
            return None
        target_center = self._target_ground_center(target)
        if target_center is None:
            return None
        target_aligned = _rotate(target_center, -plan.alignment_angle_rad)
        closest: tuple[float, float] | None = None
        for member in plan.members:
            member_center = self._target_ground_center(member)
            if member_center is None:
                continue
            member_aligned = _rotate(
                member_center,
                -plan.alignment_angle_rad,
            )
            metrics = (
                abs(target_aligned.x - member_aligned.x),
                abs(target_aligned.y - member_aligned.y),
            )
            if closest is None or metrics < closest:
                closest = metrics
        return closest

    def _side_neighbor_rejection(
        self,
        plan: NearFieldGraspPlan,
        targets: tuple[GraspTarget, ...],
    ) -> str | None:
        """检查蓝色侧邻，以及单橙计划的侧邻门禁。

        橙色独立性只约束单橙计划；绿/黑计划不会因为侧边出现橙色而
        被淘汰。蓝色危险目标的侧邻安全门禁仍保留。
        """

        supply_members = tuple(
            member
            for member in plan.members
            if member.observation.target_class in SUPPLIES
        )
        single_orange = (
            len(plan.members) == 1
            and plan.members[0].observation.target_class
            is TargetClass.ORANGE_INJURED
        )
        if not supply_members and not single_orange:
            return None
        side_neighbors: list[tuple[GraspTarget, float, float]] = []
        for other in targets:
            if other.track_id in plan.member_ids or not other.observed:
                continue
            metrics = self._side_neighbor_metrics(plan, other)
            if metrics is None:
                continue
            dx_mm, dy_mm = metrics
            if (
                dx_mm > self.config.side_neighbor_longitudinal_margin_mm
                or dy_mm > self.config.side_neighbor_lateral_margin_mm
                or dy_mm <= 1e-6
            ):
                continue
            side_neighbors.append((other, dx_mm, dy_mm))
        has_adjacent_supply = any(
            other.observation.target_class in SUPPLIES
            for other, _, _ in side_neighbors
        )
        for other, dx_mm, dy_mm in side_neighbors:
            other_class = other.observation.target_class
            if other_class is TargetClass.BLUE_DANGER:
                # Actual jaw sweep below is authoritative; adjacency alone is not a veto.
                continue
            if other_class is not TargetClass.ORANGE_INJURED:
                continue
            if other_class is TargetClass.ORANGE_INJURED and not single_orange:
                continue
            if (
                len(plan.members) == 1
                and plan.members[0].observation.target_class
                is TargetClass.GREEN_SUPPLY
                and not has_adjacent_supply
            ):
                prefix = "side_adjacent_incompatible_single_green"
            else:
                prefix = "side_adjacent_incompatible"
            return (
                f"{prefix}:track={other.track_id}:"
                f"class={other_class.value}:"
                f"dx_mm={dx_mm:.1f}:dy_mm={dy_mm:.1f}"
            )
        return None

    @dataclass(frozen=True, slots=True)
    class _GroupGeometry:
        points: tuple[GroundPoint, ...]
        centers: tuple[GroundPoint, ...]
        center: GroundPoint
        rotated_points: tuple[GroundPoint, ...]
        angle_rad: float
        x0_mm: float
        x1_mm: float
        y0_mm: float
        y1_mm: float
        opening_width_mm: float
        opening_servo_angles_deg: tuple[float, float]
        left_tip_y_mm: float
        right_tip_y_mm: float
        x_front_mm: float
        target_final_x_mm: float
        corridor_start_x_mm: float
        corridor_end_x_mm: float
        corridor_half_width_mm: float
        forward_distance_mm: float
        reasons: tuple[str, ...]

    def _mechanically_reachable(
        self,
        points: tuple[GroundPoint, ...],
        angle_rad: float,
    ) -> bool:
        """Return whether both jaws can reach the rotated object envelope.

        ``angle_rad`` is the vehicle turn from the current heading.  The
        envelope is evaluated in the post-turn robot frame and each jaw is
        solved independently.  This deliberately does not use the envelope
        centre as a centring requirement: a target is reachable whenever the
        left and right physical edges can be covered by the real servo
        travels.
        """

        rotated = tuple(_rotate(point, -angle_rad) for point in points)
        _, _, y0, y1 = _bounds(rotated)
        half_clearance = self.config.clearance_mm / 2.0
        left_tip_y = y1 + half_clearance
        right_tip_y = y0 - half_clearance
        if not rotated or any(point.x <= 0.0 for point in rotated):
            return False
        if left_tip_y - right_tip_y > self.maximum_opening_mm + 1e-9:
            return False
        try:
            self.servo_angles_for_edges(left_tip_y, right_tip_y)
        except ValueError:
            return False
        return True

    def _minimum_reachable_angle(
        self,
        points: tuple[GroundPoint, ...],
        *,
        align: bool,
    ) -> float:
        """Find the smallest signed turn that makes both jaws reachable.

        The useful physical range is the forward half-plane.  A coarse scan
        finds the first feasible interval on each side and a binary search
        refines its boundary.  The selector is a bounded background operation
        and this avoids adding a second geometric approximation or a centre
        tolerance to the control path.
        """

        if not align or self._mechanically_reachable(points, 0.0):
            return 0.0

        centres = tuple(points)
        centre = GroundPoint(
            sum(point.x for point in centres) / len(centres),
            sum(point.y for point in centres) / len(centres),
        )
        preferred_sign = 1.0 if centre.y >= 0.0 else -1.0
        step = math.radians(1.0)
        maximum = math.pi / 2.0 - 1e-6
        best: float | None = None
        for sign in (preferred_sign, -preferred_sign):
            previous = 0.0
            distance = step
            while distance <= maximum + 1e-9:
                candidate = sign * min(distance, maximum)
                if self._mechanically_reachable(points, candidate):
                    low, high = previous, abs(candidate)
                    for _ in range(45):
                        middle = (low + high) / 2.0
                        if self._mechanically_reachable(points, sign * middle):
                            high = middle
                        else:
                            low = middle
                    # Land inside the feasible interval, not on its floating-point
                    # boundary. A small geometry margin avoids repeated micro-turns.
                    refined = sign * high
                    interior = sign * min(high + math.radians(2.0), maximum)
                    if self._mechanically_reachable(points, interior):
                        refined = interior
                    if best is None or abs(refined) < abs(best) - 1e-12:
                        best = refined
                    break
                previous = abs(candidate)
                distance += step
        return 0.0 if best is None else best

    def _group_geometry(
        self,
        members: tuple[GraspTarget, ...],
        *,
        align: bool = True,
        alignment_tolerance_mm: float | None = None,
        range_limit_mm: float | None = None,
    ) -> _GroupGeometry:
        """计算候选组几何；门禁理由也由此处统一生成。"""

        envelopes: list[TargetGroundEnvelope] = []
        for member in members:
            if member.envelope is None:
                raise ValueError("missing_ground_envelope")
            envelopes.append(member.envelope)
        points = tuple(point for envelope in envelopes for point in envelope.corners)
        centers = tuple(envelope.center for envelope in envelopes)
        center = GroundPoint(
            (min(point.x for point in centers) + max(point.x for point in centers)) / 2.0,
            (min(point.y for point in centers) + max(point.y for point in centers)) / 2.0,
        )
        reasons: list[str] = []
        range_limit = (
            self.config.max_range_mm
            if range_limit_mm is None
            else float(range_limit_mm)
        )
        if not math.isfinite(range_limit) or range_limit <= 0.0:
            raise ValueError("range_limit_mm must be finite and positive.")
        if any(point.x <= 0.0 or math.hypot(point.x, point.y) > range_limit for point in centers):
            reasons.append("outside_near_field")
        # ``alignment_tolerance_mm`` remains accepted for callers of the
        # historical API, but it is no longer a centring gate.  Physical
        # reachability is evaluated at the current heading first and only then
        # is the minimum necessary turn selected.
        if alignment_tolerance_mm is not None:
            self._validate_alignment_tolerance(alignment_tolerance_mm)
        angle = self._minimum_reachable_angle(points, align=align)
        clearance_half = self.config.clearance_mm / 2.0
        rotated_points = tuple(_rotate(point, -angle) for point in points)
        rotated_centers = tuple(_rotate(point, -angle) for point in centers)
        x0, x1, y0, y1 = _bounds(rotated_points)
        left_tip_y = y1 + clearance_half
        right_tip_y = y0 - clearance_half
        # 开口宽度仍包含总余量，但左右末端分别反解到各自边界，不再使用
        # 一个统一的对称相对角度。任何超出真实最大开口的包络都拒绝，
        # 不能通过收窄夹爪去夹包络的一部分。
        if left_tip_y - right_tip_y > self.maximum_opening_mm + 1e-9:
            reasons.append("maximum_opening_exceeded")
        opening = left_tip_y - right_tip_y
        if reasons:
            opening_servo_angles = self.open_angles
        else:
            try:
                opening_servo_angles = self.servo_angles_for_edges(
                    left_tip_y,
                    right_tip_y,
                )
            except ValueError as exc:
                # Keep the concrete mechanical reason in diagnostics and let
                # the selector reject this candidate rather than allowing an
                # exception to become a retry loop in the control state.
                reasons.append(str(exc).split(":", 1)[0])
                opening_servo_angles = self.open_angles
        # 绿/黑目标之间允许在揽入过程中滑动/转动；前进行程由最远目标 K0
        # 决定。单个橙色目标同样使用可靠底面中心 K0，颜色上表面投影的
        # 纵向拉长不作为前进深度：前进距离为最前 K0 减去橙色终点。
        x_front = max(point.x for point in rotated_centers)
        is_single_orange = (
            len(members) == 1
            and members[0].observation.target_class is TargetClass.ORANGE_INJURED
        )
        if is_single_orange:
            target_final_x = self.config.orange_target_final_x_mm
        else:
            target_final_x = self.config.target_final_x_mm
        forward_distance = max(0.0, x_front - target_final_x)
        # 与 `_regions()` 的实际扫掠走廊共用同一前端公式。
        corridor_end = self._sweep_end_x_mm(opening_servo_angles, forward_distance)
        corridor_half_width = max(
            abs(right_tip_y - self.config.corridor_lateral_margin_mm),
            abs(left_tip_y + self.config.corridor_lateral_margin_mm),
        )
        if forward_distance > self.config.max_forward_distance_mm:
            reasons.append("forward_distance_exceeded")
        return self._GroupGeometry(
            points=points,
            centers=centers,
            center=center,
            rotated_points=rotated_points,
            angle_rad=angle,
            x0_mm=x0,
            x1_mm=x1,
            y0_mm=y0,
            y1_mm=y1,
            opening_width_mm=opening,
            opening_servo_angles_deg=opening_servo_angles,
            left_tip_y_mm=left_tip_y,
            right_tip_y_mm=right_tip_y,
            x_front_mm=x_front,
            target_final_x_mm=target_final_x,
            corridor_start_x_mm=self.config.corridor_start_x_mm,
            corridor_end_x_mm=corridor_end,
            corridor_half_width_mm=corridor_half_width,
            forward_distance_mm=forward_distance,
            reasons=tuple(reasons),
        )

    def candidate_geometry(
        self,
        target: GraspTarget,
        *,
        policy: NearFieldGraspPolicy | None = None,
    ) -> CandidateGeometry:
        """返回一个目标的几何诊断；不可选目标也会被记录。"""

        if not isinstance(target, GraspTarget):
            raise TypeError("target must be a GraspTarget.")
        reasons: list[str] = []
        if not self._eligible(target, policy):
            reasons.append("ineligible_target")
        if target.envelope is None:
            reasons.append("missing_ground_envelope")
            return CandidateGeometry(
                track_id=target.track_id,
                target_class=target.observation.target_class,
                frame_sequence=target.observation.frame_sequence,
                capture_timestamp_ns=target.observation.capture_timestamp_ns,
                k0_x_mm=(target.observation.ground_point.x if target.observation.ground_point is not None else None),
                x0_mm=None,
                x1_mm=None,
                depth_mm=None,
                opening_width_mm=None,
                target_final_x_mm=self.config.target_final_x_mm,
                corridor_start_x_mm=self.config.corridor_start_x_mm,
                corridor_end_x_mm=None,
                corridor_half_width_mm=None,
                forward_distance_mm=None,
                alignment_angle_rad=None,
                eligible=False,
                reason=",".join(reasons),
                handoff_matched=target.handoff_matched,
            )
        try:
            geometry = self._group_geometry((target,))
        except ValueError as exc:
            reasons.append(str(exc))
            return CandidateGeometry(
                track_id=target.track_id,
                target_class=target.observation.target_class,
                frame_sequence=target.observation.frame_sequence,
                capture_timestamp_ns=target.observation.capture_timestamp_ns,
                k0_x_mm=target.envelope.center.x,
                x0_mm=None,
                x1_mm=None,
                depth_mm=None,
                opening_width_mm=None,
                target_final_x_mm=self.config.target_final_x_mm,
                corridor_start_x_mm=self.config.corridor_start_x_mm,
                corridor_end_x_mm=None,
                corridor_half_width_mm=None,
                forward_distance_mm=None,
                alignment_angle_rad=None,
                eligible=False,
                reason=",".join(reasons),
                handoff_matched=target.handoff_matched,
            )
        reasons.extend(geometry.reasons)
        return CandidateGeometry(
            track_id=target.track_id,
            target_class=target.observation.target_class,
            frame_sequence=target.observation.frame_sequence,
            capture_timestamp_ns=target.observation.capture_timestamp_ns,
            k0_x_mm=target.envelope.center.x,
            x0_mm=geometry.x0_mm,
            x1_mm=geometry.x1_mm,
            depth_mm=geometry.x1_mm - geometry.x0_mm,
            opening_width_mm=geometry.opening_width_mm,
            target_final_x_mm=geometry.target_final_x_mm,
            corridor_start_x_mm=geometry.corridor_start_x_mm,
            corridor_end_x_mm=geometry.corridor_end_x_mm,
            corridor_half_width_mm=geometry.corridor_half_width_mm,
            forward_distance_mm=geometry.forward_distance_mm,
            alignment_angle_rad=geometry.angle_rad,
            eligible=not reasons,
            reason=",".join(reasons) or None,
            handoff_matched=target.handoff_matched,
        )

    def candidate_geometries(
        self,
        targets: tuple[GraspTarget, ...] | list[GraspTarget],
        *,
        policy: NearFieldGraspPolicy | None = None,
    ) -> tuple[CandidateGeometry, ...]:
        """按 track ID 返回当前帧所有目标的诊断。"""

        targets = tuple(targets)
        diagnostics: list[CandidateGeometry] = []
        for target in sorted(targets, key=lambda item: item.track_id):
            diagnostic = self.candidate_geometry(target, policy=policy)
            isolation_rejection = self._orange_isolation_rejection(
                target,
                targets,
            )
            if isolation_rejection is not None:
                reason = ",".join(
                    item
                    for item in (diagnostic.reason, isolation_rejection)
                    if item
                )
                diagnostic = replace(
                    diagnostic,
                    eligible=False,
                    reason=reason,
                )
            diagnostics.append(diagnostic)
        return tuple(diagnostics)

    def _build(
        self,
        members: tuple[GraspTarget, ...],
        *,
        align: bool = True,
        alignment_tolerance_mm: float | None = None,
        range_limit_mm: float | None = None,
        policy: NearFieldGraspPolicy | None = None,
    ) -> NearFieldGraspPlan:
        if policy is None:
            policy = self.default_policy
        if not isinstance(policy, NearFieldGraspPolicy):
            raise TypeError("policy must be a NearFieldGraspPolicy or None.")
        cfg, k = self.config, self.kinematics
        if not 1 <= len(members) <= policy.max_targets or any(
            not self._eligible(t, policy) for t in members
        ):
            raise ValueError("ineligible_members_or_capacity")
        if any(
            member.observation.target_class is TargetClass.ORANGE_INJURED
            for member in members
        ) and len(members) != 1:
            raise ValueError("orange_injured_single_only")
        if len({member.observation.frame_sequence for member in members}) != 1 or len({member.observation.capture_timestamp_ns for member in members}) != 1:
            raise ValueError("members_must_share_capture_frame")
        geometry = self._group_geometry(
            members,
            align=align,
            alignment_tolerance_mm=alignment_tolerance_mm,
            range_limit_mm=range_limit_mm,
        )
        if geometry.reasons:
            raise ValueError(geometry.reasons[0])
        servo = geometry.opening_servo_angles_deg
        regions = self._regions(
            geometry.angle_rad,
            geometry.forward_distance_mm,
            geometry.left_tip_y_mm,
            geometry.right_tip_y_mm,
        )
        return NearFieldGraspPlan(
            max(t.observation.frame_sequence for t in members),
            max(t.observation.capture_timestamp_ns for t in members),
            tuple(sorted(members, key=lambda t: t.track_id)),
            geometry.center,
            geometry.angle_rad,
            _rectangle(geometry.x0_mm, geometry.x1_mm, geometry.y0_mm, geometry.y1_mm),
            geometry.opening_width_mm,
            self.maximum_opening_mm,
            servo,
            geometry.forward_distance_mm,
            0.0,
            GraspScore(1, 0, 0, 0, 0, 0, 0),
            regions,
        )

    @staticmethod
    def _groups_by_farthest_anchor(
        candidates: tuple[GraspTarget, ...],
        max_targets: int,
    ) -> tuple[tuple[GraspTarget, ...], ...]:
        """以不同物资作为最远 X 锚点枚举走廊长度对应的候选组。"""

        ordered = tuple(sorted(
            candidates,
            key=lambda target: (
                cast(TargetGroundEnvelope, target.envelope).center.x,
                target.track_id,
            ),
        ))
        groups: list[tuple[GraspTarget, ...]] = []
        seen: set[tuple[int, ...]] = set()
        for anchor in ordered:
            anchor_x = cast(TargetGroundEnvelope, anchor.envelope).center.x
            preceding = tuple(
                target
                for target in ordered
                if target.track_id != anchor.track_id
                and cast(TargetGroundEnvelope, target.envelope).center.x
                <= anchor_x + 1e-9
            )
            for size in range(1, max_targets + 1):
                for companions in combinations(preceding, size - 1):
                    group = tuple(sorted((*companions, anchor), key=lambda item: item.track_id))
                    key = tuple(target.track_id for target in group)
                    if key not in seen:
                        seen.add(key)
                        groups.append(group)
        return tuple(groups)

    def select(
        self,
        targets: tuple[GraspTarget, ...] | list[GraspTarget],
        *,
        locked_ids: tuple[int, ...] | None = None,
        policy: NearFieldGraspPolicy | None = None,
        align: bool = True,
        alignment_tolerance_mm: float | None = None,
    ) -> GraspSelection:
        targets = tuple(targets)
        if not all(isinstance(target, GraspTarget) for target in targets):
            raise TypeError("targets must contain GraspTarget values.")
        if policy is None:
            policy = self.default_policy
        if not isinstance(policy, NearFieldGraspPolicy):
            raise TypeError("policy must be a NearFieldGraspPolicy or None.")
        if not isinstance(align, bool):
            raise ValueError("align must be a boolean.")
        if alignment_tolerance_mm is not None:
            alignment_tolerance_mm = self._validate_alignment_tolerance(
                alignment_tolerance_mm
            )
        if locked_ids is not None:
            locked_ids = tuple(locked_ids)
            if not locked_ids or len(set(locked_ids)) != len(locked_ids) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in locked_ids):
                raise ValueError("locked_ids must contain unique positive integers or None.")
        rejections: list[str] = []
        candidates = sorted(
            (t for t in targets if self._eligible(t, policy)),
            key=lambda t: (
                not t.handoff_matched,
                math.hypot(
                    cast(TargetGroundEnvelope, t.envelope).center.x,
                    cast(TargetGroundEnvelope, t.envelope).center.y,
                ),
                t.track_id,
            ),
        )
        isolated_candidates: list[GraspTarget] = []
        for candidate in candidates:
            if locked_ids is not None and candidate.track_id not in locked_ids:
                continue
            rejection = self._orange_isolation_rejection(
                candidate,
                targets,
                align=align,
            )
            if rejection is not None:
                rejections.append(rejection)
                continue
            isolated_candidates.append(candidate)
        # 首轮比赛规则要求恰好一个绿色物资。远场交接目标只作为排序偏好
        # （见上方候选排序与 ``_plan_rank``），不再独占候选：每个可选绿色
        # 各自形成一个单目标计划，由排名决定实际抓取哪一个。交接目标在
        # 当前帧无法生成合法计划时由下一个绿色接管，否则近场会为一个
        # 不可交付的目标空转到 ``alignment_timeout`` 再重选，反复空转。
        first_single_green = (
            policy.allowed_classes == frozenset((TargetClass.GREEN_SUPPLY,))
            and policy.max_targets == 1
        )
        candidates = isolated_candidates[: self.config.max_candidates]
        if locked_ids is not None:
            locked_targets = tuple(
                target for target in targets if target.track_id in locked_ids
            )
            if any(
                not self._eligible(target, policy)
                for target in locked_targets
            ):
                return GraspSelection(None, ("locked_member_not_selectable",))
            groups = [locked_targets]
        elif first_single_green:
            groups = [(target,) for target in candidates]
        else:
            # 伤员从候选生成开始就是严格单目标方案；绿/黑只能彼此组合。
            # 不能先生成混合组、再依赖后续异常把橙色剔除。
            orange_groups = [
                (target,)
                for target in candidates
                if target.observation.target_class is TargetClass.ORANGE_INJURED
            ]
            supply_groups = self._groups_by_farthest_anchor(
                tuple(
                    target
                    for target in candidates
                    if target.observation.target_class in SUPPLIES
                ),
                policy.max_targets,
            )
            groups = [*orange_groups, *supply_groups]
        if locked_ids is not None and (
            not groups
            or tuple(sorted(t.track_id for t in groups[0])) != locked_ids
            or any(not t.observed for t in groups[0])
        ):
            return GraspSelection(None, ("locked_members_missing",))
        if locked_ids is not None:
            for target in targets:
                if target.track_id not in locked_ids:
                    continue
                rejection = self._orange_isolation_rejection(
                    target,
                    targets,
                    align=align,
                )
                if rejection is not None:
                    return GraspSelection(None, (rejection,))
        plans, preview_plans, seen = [], [], set()
        for group in groups:
            try:
                while True:
                    plan = self._build(
                        group,
                        align=align,
                        policy=policy,
                        alignment_tolerance_mm=alignment_tolerance_mm,
                        range_limit_mm=(
                            self.config.max_range_mm
                            + self.config.range_hysteresis_mm
                            if locked_ids is not None
                            or any(member.handoff_matched for member in group)
                            else None
                        ),
                    )
                    preview_plans.append(self._score_plan(plan, clearance=0.0))
                    side_rejection = self._side_neighbor_rejection(plan, targets)
                    if side_rejection is not None:
                        raise ValueError(side_rejection)
                    # 对准前也检查预测旋转后的走廊。已知危险目标不应
                    # 把车辆引入一个随后必然失败的旋转计划。
                    extra = []
                    clearance = math.inf
                    for target in targets:
                        if target.track_id in plan.member_ids:
                            continue
                        if (
                            target.observed
                            and (
                                target.observation.target_class
                                in {TargetClass.BLUE_DANGER}
                                or target.observation.model_target_class
                                in {TargetClass.BLUE_DANGER}
                            )
                            and target.observation.ground_point is None
                        ):
                            raise ValueError(
                                "unknown_target_geometry_missing:"
                                f"{target.track_id}:"
                                f"{target.observation.target_class.value}"
                            )
                        gap = self._obstacle_distance(target, plan.regions)
                        if gap <= 1e-6:
                            if (
                                self._eligible(target, policy)
                                and target.observation.target_class in SUPPLIES
                                and all(
                                    member.observation.target_class in SUPPLIES
                                    for member in plan.members
                                )
                            ):
                                extra.append(target)
                            else:
                                raise ValueError(f"blocked_target:{target.track_id}:{target.observation.target_class.value}")
                        clearance = min(clearance, gap)
                    if not extra:
                        break
                    if locked_ids is not None:
                        raise ValueError("locked_members_changed")
                    group = tuple((*group, *extra))
                    if len(group) > policy.max_targets:
                        raise ValueError("incidental_capacity_exceeded")
                if plan.member_ids in seen:
                    continue
                seen.add(plan.member_ids)
                plans.append(self._score_plan(plan, clearance=clearance))
            except ValueError as exc:
                rejections.append(str(exc))
        # 安全条件已经是硬门禁；先按比赛规则总分排序。同分时用单橙优先
        # 权重及数量、净空、距离、对准代价形成稳定次序。
        ranked_plans = tuple(sorted(plans, key=self._plan_rank))
        best = ranked_plans[0] if ranked_plans else None
        if (
            locked_ids is None
            and best is not None
            and len(best.members) < policy.max_targets
            and all(member.observation.target_class in SUPPLIES for member in best.members)
        ):
            # A handoff can confirm its seed before adjacent local tracks have
            # enough hits. Do not freeze a singleton during that startup race.
            for other in targets:
                if (
                    other.track_id in best.member_ids
                    or other.confirmed
                    or not self._eligible(replace(other, confirmed=True), policy)
                    or other.observation.target_class not in SUPPLIES
                ):
                    continue
                assert other.envelope is not None
                if any(
                    abs(other.envelope.center.x - member.envelope.center.x)
                    <= self.config.side_neighbor_longitudinal_margin_mm
                    and abs(other.envelope.center.y - member.envelope.center.y)
                    <= self.config.side_neighbor_lateral_margin_mm
                    for member in best.members
                    if member.envelope is not None
                ):
                    return GraspSelection(None, ("waiting_adjacent_supply_confirmation",), best)
        preview = best or (
            min(preview_plans, key=self._plan_rank) if preview_plans else None
        )
        return GraspSelection(best, tuple(sorted(set(rejections))), preview)

    def _score_plan(
        self,
        plan: NearFieldGraspPlan,
        *,
        clearance: float,
    ) -> NearFieldGraspPlan:
        is_single_orange = (
            len(plan.members) == 1
            and plan.members[0].observation.target_class
            is TargetClass.ORANGE_INJURED
        )
        features = (
            1.0 if is_single_orange else 0.0,
            len(plan.member_ids) / 3,
            min(1.0, clearance / self.config.clearance_scale_mm),
            max(
                0.0,
                1 - plan.forward_distance_mm / self.config.max_forward_distance_mm,
            ),
            max(
                0.0,
                1 - abs(plan.alignment_angle_rad) / (math.pi / 2),
            ),
        )
        total = math.fsum(
            weight * feature
            for weight, feature in zip(self.config.weights, features)
        ) / math.fsum(self.config.weights)
        return replace(
            plan,
            clearance_mm=clearance,
            score=GraspScore(
                math.fsum(
                    {
                        TargetClass.GREEN_SUPPLY: self.config.green_score_points,
                        TargetClass.BLACK_CORE: self.config.black_score_points,
                        TargetClass.ORANGE_INJURED: self.config.orange_score_points,
                    }[member.observation.target_class]
                    for member in plan.members
                ),
                *features,
                total,
            ),
        )

    @staticmethod
    def _plan_rank(plan: NearFieldGraspPlan) -> tuple[object, ...]:
        return (
            -plan.score.rule_points,
            -plan.score.orange_priority,
            not any(member.handoff_matched for member in plan.members),
            -plan.score.total,
            -plan.clearance_mm,
            plan.forward_distance_mm,
            plan.member_ids,
        )

    def recheck(
        self,
        plan: NearFieldGraspPlan,
        targets: tuple[GraspTarget, ...],
        *,
        progress_mm: float,
        policy: NearFieldGraspPolicy | None = None,
    ) -> str | None:
        """只检验锁定动作的剩余走廊；不临时换组或增加前进行程。"""
        if not isinstance(plan, NearFieldGraspPlan):
            raise TypeError("plan must be a NearFieldGraspPlan.")
        targets = tuple(targets)
        if not all(isinstance(target, GraspTarget) for target in targets):
            raise TypeError("targets must contain GraspTarget values.")
        if policy is None:
            policy = self.default_policy
        if not isinstance(policy, NearFieldGraspPolicy):
            raise TypeError("policy must be a NearFieldGraspPolicy or None.")
        if not math.isfinite(progress_mm) or progress_mm < 0:
            raise ValueError(f"progress_mm must be finite and nonnegative, got {progress_mm!r}.")
        plan_y0 = min(point.y for point in plan.bounds)
        plan_y1 = max(point.y for point in plan.bounds)
        clearance_half = self.config.clearance_mm / 2.0
        plan_left_tip_y = plan_y1 + clearance_half
        plan_right_tip_y = plan_y0 - clearance_half
        regions = self._regions(
            0.0,
            max(0.0, plan.forward_distance_mm - progress_mm),
            plan_left_tip_y,
            plan_right_tip_y,
        )
        by_id = {t.track_id: t for t in targets}
        for member in plan.members:
            if member.observation.target_class is not TargetClass.ORANGE_INJURED:
                continue
            current = by_id.get(member.track_id)
            if (
                current is None
                or not current.observed
                or current.observation.ground_point is None
            ):
                return (
                    "orange_isolation_unknown_ground_track:"
                    f"{member.track_id}"
                )
            isolation_rejection = self._orange_isolation_rejection(
                current,
                targets,
                align=False,
            )
            if isolation_rejection is not None:
                return isolation_rejection
        for member in plan.members:
            current = by_id.get(member.track_id)
            if current is not None and current.observed:
                if not self._eligible(current, policy):
                    return f"member_class_risk:{current.track_id}"
                if current.envelope is not None:
                    _, _, y0, y1 = _bounds(current.envelope.corners)
                    if (
                        y1 + clearance_half > plan_left_tip_y + 1e-6
                        or y0 - clearance_half < plan_right_tip_y - 1e-6
                    ):
                        return f"member_outside_opening:{current.track_id}"
        for target in targets:
            if target.track_id not in plan.member_ids and self._obstacle_distance(target, regions) <= 1e-6:
                return f"new_sweep_obstacle:{target.track_id}:{target.observation.target_class.value}"
        return None
