"""多目标近场收拢的纯逻辑准备器和运动状态机。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from threading import Condition, Thread
import time

from rescue_vision.app.breakup_planner import BreakupPlan, BreakupSceneContext, BreakupTarget, physical_radii, plan_breakup
from rescue_vision.app.near_field_grasp import (
    GraspSelection, GraspTarget, GraspTargetTracker, NearFieldGraspPlan,
    NearFieldGraspPolicy, NearFieldGraspSelector, NearFieldHandoffPrior,
)
from rescue_vision.motion.approach_speed import approach_speed_m_s
from rescue_vision.motion.gripper_kinematics import GripperKinematics
from rescue_vision.motion.protocol import OdometryImu
from rescue_vision.motion.stationary import StationaryMotionEvidence
from rescue_vision.localization import normalize_angle
from rescue_vision.geometry.types import GroundPoint
from rescue_vision.perception import PerceptionSnapshot, TargetObservation
from rescue_vision.perception.types import ObservationQuality, TargetClass

__all__ = [
    "GraspPreparation",
    "GraspPreparationSession",
    "GripperWidthPickupDecision",
    "GripperWidthPickupResult",
    "GripperWidthPickupSequence",
    "GripperWidthPickupState",
    "GraspPreparationWorker",
]


def _timestamp_ns(value: int, name: str = "timestamp_ns") -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer, got {value!r}.")
    return value


def _validate_candidate_exclusions(snapshot: PerceptionSnapshot, indices: frozenset[int]) -> None:
    if not isinstance(indices, frozenset) or any(
        isinstance(index, bool) or not isinstance(index, int)
        or not 0 <= index < len(snapshot.observations) for index in indices
    ):
        raise ValueError(f"excluded_observation_indices outside frame of {len(snapshot.observations)} observations: {indices!r}")


def _same_frame_observation(
    candidate: TargetObservation,
    excluded: TargetObservation,
) -> bool:
    """Match an exclusion through tracker de-duplication without losing identity."""

    if candidate is excluded:
        return True
    if (
        candidate.frame_sequence != excluded.frame_sequence
        or candidate.capture_timestamp_ns != excluded.capture_timestamp_ns
        or candidate.target_class is not excluded.target_class
        or candidate.model_target_class is not excluded.model_target_class
    ):
        return False
    if candidate.ground_point is not None and excluded.ground_point is not None:
        return (
            math.hypot(
                candidate.ground_point.x - excluded.ground_point.x,
                candidate.ground_point.y - excluded.ground_point.y,
            ) <= 15.0
            and candidate.box.iou(excluded.box) >= 0.7
        )
    return candidate.box == excluded.box and candidate.k0 == excluded.k0


class GripperWidthPickupState(str, Enum):
    SEARCH = "search"
    ALIGNING = "aligning"
    VERIFYING = "verifying"
    OPENING = "opening"
    FORWARD = "forward"
    CLOSING = "closing"
    COMPLETE = "complete"
    ABORTED = "aborted"


class GraspSceneAction(str, Enum):
    GRASP = "grasp"
    MOTION = "motion"
    RECOVERY = "recovery"
    OBSERVE = "observe"
    EXIT = "exit"


@dataclass(frozen=True, slots=True)
class GraspPreparation:
    capture_timestamp_ns: int
    selection: GraspSelection
    targets: tuple[GraspTarget, ...]
    ready: bool = False
    checked_member_ids: tuple[int, ...] | None = None
    session_id: int = 0
    confirmation_count: int = 0
    confirmation_required: int = 1
    # 后台准备完成时的应用单调时钟。它与 plan 的采集时刻有意分开，
    # 让提交门禁可以分别记录几何年龄和准备结果年龄。
    prepared_timestamp_ns: int | None = None
    result_timestamp_ns: int | None = None
    recovery_plan: BreakupPlan | None = None
    recovery_checked: bool = False

    @property
    def action(self) -> GraspSceneAction:
        if self.selection.plan is not None:
            return (GraspSceneAction.MOTION if self.selection.plan.alignment_angle_rad != 0
                    else GraspSceneAction.GRASP)
        if self.recovery_plan is not None:
            return GraspSceneAction.RECOVERY
        return GraspSceneAction.EXIT if self.recovery_checked else GraspSceneAction.OBSERVE

    def __post_init__(self) -> None:
        if isinstance(self.capture_timestamp_ns, bool) or not isinstance(self.capture_timestamp_ns, int) or self.capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be a non-negative integer.")
        if not isinstance(self.selection, GraspSelection):
            raise TypeError("selection must be a GraspSelection.")
        if not isinstance(self.targets, tuple) or not all(isinstance(target, GraspTarget) for target in self.targets):
            raise ValueError("targets must contain GraspTarget values.")
        if not isinstance(self.ready, bool):
            raise ValueError("ready must be a boolean.")
        if not isinstance(self.recovery_checked, bool):
            raise ValueError(f"Invalid recovery_checked={self.recovery_checked!r}")
        if self.recovery_plan is not None and not isinstance(self.recovery_plan, BreakupPlan):
            raise TypeError(f"Invalid recovery_plan={self.recovery_plan!r}")
        if self.selection.plan is not None and self.recovery_plan is not None:
            raise ValueError("A scene cannot freeze both grasp and recovery actions.")
        if self.checked_member_ids is not None:
            if not isinstance(self.checked_member_ids, tuple) or len(set(self.checked_member_ids)) != len(self.checked_member_ids) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in self.checked_member_ids):
                raise ValueError("checked_member_ids must contain unique positive integers.")
        if isinstance(self.session_id, bool) or not isinstance(self.session_id, int) or self.session_id < 0:
            raise ValueError("session_id must be a non-negative integer.")
        for name in ("confirmation_count", "confirmation_required"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer.")
        if self.confirmation_required == 0:
            raise ValueError("confirmation_required must be positive.")
        if self.confirmation_count > self.confirmation_required:
            raise ValueError(
                "confirmation_count cannot exceed confirmation_required."
            )
        for name in ("prepared_timestamp_ns", "result_timestamp_ns"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(
                    f"{name} must be a non-negative integer or None."
                )
        if (
            self.result_timestamp_ns is not None
            and self.result_timestamp_ns < self.capture_timestamp_ns
        ):
            raise ValueError(
                "result_timestamp_ns cannot precede capture_timestamp_ns."
            )

    @property
    def confirmation_progress(self) -> tuple[int, int]:
        return self.confirmation_count, self.confirmation_required

    def plan_age_ns(self, now_ns: int) -> int | None:
        now_ns = _timestamp_ns(now_ns, "now_ns")
        plan = self.selection.plan
        if plan is None or now_ns < plan.capture_timestamp_ns:
            return None
        return now_ns - plan.capture_timestamp_ns

    def preparation_age_ns(self, now_ns: int) -> int | None:
        now_ns = _timestamp_ns(now_ns, "now_ns")
        prepared = self.prepared_timestamp_ns
        if prepared is None:
            prepared = self.result_timestamp_ns
        if prepared is None:
            prepared = self.capture_timestamp_ns
        if now_ns < prepared:
            return None
        return now_ns - prepared


class GraspPreparationSession:
    """单线程消费新感知帧并维护唯一的近场确认窗口。"""

    def __init__(self, tracker: GraspTargetTracker, selector: NearFieldGraspSelector):
        if not isinstance(tracker, GraspTargetTracker):
            raise TypeError("tracker must be a GraspTargetTracker.")
        if not isinstance(selector, NearFieldGraspSelector):
            raise TypeError("selector must be a NearFieldGraspSelector.")
        self.tracker, self.selector = tracker, selector
        self._locked_ids: tuple[int, ...] | None = None
        self._locked_member_classes: tuple[TargetClass, ...] | None = None
        self._last_selection: GraspSelection | None = None
        self._confirmation_count = 0
        self._confirmation_last_frame_sequence: int | None = None
        self._confirmation_plan: NearFieldGraspPlan | None = None
        self._last_frame_sequence: int | None = None
        self._last_preparation: GraspPreparation | None = None
        self._locked_member_points: dict[int, GroundPoint] = {}
        self._recovery_plan: BreakupPlan | None = None
        self._recovery_count = 0
        self._frozen_scene_ids: set[int] = set()

    def reset(self) -> None:
        """清空确认窗口并重置近场 tracker。"""

        self.tracker.reset()
        self._locked_ids = None
        self._locked_member_classes = None
        self._last_selection = None
        self._confirmation_count = 0
        self._confirmation_last_frame_sequence = None
        self._confirmation_plan = None
        self._last_frame_sequence = None
        self._last_preparation = None
        self._locked_member_points.clear()
        self._recovery_plan = None
        self._recovery_count = 0
        self._frozen_scene_ids.clear()

    def _clear_confirmation(self) -> None:
        self._confirmation_count = 0
        self._confirmation_last_frame_sequence = None
        self._confirmation_plan = None

    def _freeze_distinct_scene_targets(
        self,
        targets: tuple[GraspTarget, ...],
    ) -> tuple[GraspTarget, ...]:
        """Promote a spatially distinct one-frame detection in this stopped scene."""

        existing = [
            target
            for target in targets
            if target.observed
            and (
                target.confirmed
                or target.handoff_matched
                or target.track_id in self._frozen_scene_ids
            )
        ]
        promoted: list[GraspTarget] = []
        for target in targets:
            peers = tuple(
                item for item in existing if item.track_id != target.track_id
            )
            if (
                target.observed
                and target.selectable
                and target.envelope is not None
                and not target.confirmed
                and target.track_id not in self._frozen_scene_ids
                and (
                    not peers
                    or max(
                        target.observation.box.iou(item.observation.box)
                        for item in peers
                    )
                    <= self.selector.config.stopped_scene_new_target_max_bbox_iou
                )
            ):
                self._frozen_scene_ids.add(target.track_id)
                existing.append(target)
            promoted.append(
                replace(target, confirmed=True)
                if target.track_id in self._frozen_scene_ids
                else target
            )
        return tuple(promoted)

    def _canonicalize_locked_targets(
        self,
        targets: tuple[GraspTarget, ...],
        locked_ids: tuple[int, ...] | None,
    ) -> tuple[GraspTarget, ...]:
        """Keep a locked physical member stable when its tracker ID changes."""

        if locked_ids is None or not self._locked_member_points:
            return targets
        used_actual: set[int] = set()
        replacements: dict[int, int] = {}
        max_distance = self.tracker.tracker.config.max_association_ground_mm * 1.5
        for canonical_id in locked_ids:
            if canonical_id not in self._locked_member_points:
                continue
            reference = self._locked_member_points[canonical_id]
            direct = next(
                (
                    target
                    for target in targets
                    if target.track_id == canonical_id
                    and target.observed
                    and target.observation.target_class is not TargetClass.BLUE_DANGER
                    and target.observation.model_target_class is not TargetClass.BLUE_DANGER
                    and target.observation.ground_point is not None
                    and math.hypot(
                        target.observation.ground_point.x - reference.x,
                        target.observation.ground_point.y - reference.y,
                    )
                    <= max_distance
                ),
                None,
            )
            if direct is not None:
                used_actual.add(direct.track_id)
                continue
            candidates = tuple(
                target
                for target in targets
                if target.observed
                and target.track_id not in used_actual
                and target.observation.target_class is not TargetClass.BLUE_DANGER
                and target.observation.model_target_class is not TargetClass.BLUE_DANGER
                and target.observation.ground_point is not None
                and math.hypot(
                    target.observation.ground_point.x - reference.x,
                    target.observation.ground_point.y - reference.y,
                )
                <= max_distance
            )
            candidate = min(
                candidates,
                key=lambda target: (
                    math.hypot(
                        target.observation.ground_point.x - reference.x,
                        target.observation.ground_point.y - reference.y,
                    ),
                    target.track_id,
                ),
                default=None,
            )
            if candidate is None:
                # Preserve an explicit danger identity as well.  It will not
                # produce a grasp plan, but mapping it to the locked physical
                # member lets the safety gate invalidate immediately instead
                # of mistaking an ID change for a one-frame disappearance.
                danger_candidates = tuple(
                    target
                    for target in targets
                    if target.observed
                    and target.track_id not in used_actual
                    and (
                        target.observation.target_class is TargetClass.BLUE_DANGER
                        or target.observation.model_target_class is TargetClass.BLUE_DANGER
                    )
                    and target.observation.ground_point is not None
                    and math.hypot(
                        target.observation.ground_point.x - reference.x,
                        target.observation.ground_point.y - reference.y,
                    )
                    <= max_distance
                )
                candidate = min(
                    danger_candidates,
                    key=lambda target: (
                        math.hypot(
                            target.observation.ground_point.x - reference.x,
                            target.observation.ground_point.y - reference.y,
                        ),
                        target.track_id,
                    ),
                    default=None,
                )
            if candidate is not None:
                used_actual.add(candidate.track_id)
                replacements[candidate.track_id] = canonical_id
        if not replacements:
            return targets
        # A freshly created tracker ID may collide with an old canonical ID
        # that now belongs to another object.  Give mapped members priority and
        # move only the unrelated diagnostic target to a temporary positive ID;
        # never create duplicate member IDs in a plan.
        ordered = sorted(
            targets,
            key=lambda target: target.track_id not in replacements,
        )
        next_id = max(
            (target.track_id for target in targets),
            default=0,
        ) + 1_000_000_000
        used_ids: set[int] = set()
        canonicalized: dict[int, int] = {}
        for target in ordered:
            desired = replacements.get(target.track_id, target.track_id)
            if desired in used_ids:
                desired = next_id
                next_id += 1
            used_ids.add(desired)
            canonicalized[target.track_id] = desired
        return tuple(
            replace(target, track_id=canonicalized[target.track_id])
            for target in targets
        )

    @staticmethod
    def _is_explicit_invalidation(
        selection: GraspSelection,
        targets: tuple[GraspTarget, ...],
        locked_ids: tuple[int, ...] | None,
    ) -> bool:
        """只把明确危险/不合法几何作为确认失效，不把短暂漏检当阻挡。"""

        if locked_ids is not None:
            locked = {
                target.track_id: target
                for target in targets
                if target.track_id in locked_ids and target.observed
            }
            for target in locked.values():
                observation = target.observation
                # 锁定目标的身份已由 tracker 确认；旋转/运动模糊把某一帧
                # 识别成 unknown 只是缺证据，不是危险证据，不能作废整个
                # 确认窗口。明确蓝色危险和颜色冲突仍然立即失效。
                if (
                    observation.target_class is TargetClass.BLUE_DANGER
                    or observation.model_target_class is TargetClass.BLUE_DANGER
                ):
                    return True
        invalid_prefixes = (
            "blocked_target:",
            "side_adjacent_incompatible",
            "orange_not_isolated_track:",
            "orange_isolation_unknown_ground_track:",
            "orange_injured_single_only",
            "locked_members_changed",
            "locked_member_class_changed",
            "maximum_opening_exceeded",
            "forward_distance_exceeded",
            "outside_near_field",
            "member_class_risk:",
            "member_outside_opening:",
        )
        # A returned plan has passed every geometry gate. Rejections alongside
        # it describe other candidates, not a failure of the selected group.
        return selection.plan is None and any(
            reason.startswith(invalid_prefixes)
            for reason in selection.rejections
        )

    def update(self, snapshot: PerceptionSnapshot, *, locked_ids: tuple[int, ...] | None,
               policy: NearFieldGraspPolicy | None = None,
               session_id: int = 0,
               handoff_prior: NearFieldHandoffPrior | None = None,
               excluded_observation_indices: frozenset[int] = frozenset(),
               require_handoff: bool = False,
               recovery_context: BreakupSceneContext | None = None) -> GraspPreparation:
        if not isinstance(snapshot, PerceptionSnapshot):
            raise TypeError("snapshot must be a PerceptionSnapshot.")
        _validate_candidate_exclusions(snapshot, excluded_observation_indices)
        if locked_ids is None:
            locked_ids = self._locked_ids
        if locked_ids is not None:
            locked_ids = tuple(locked_ids)
        if policy is None:
            policy = self.selector.default_policy
        if not isinstance(policy, NearFieldGraspPolicy):
            raise TypeError("policy must be a NearFieldGraspPolicy or None.")
        if not isinstance(require_handoff, bool):
            raise ValueError("require_handoff must be a boolean.")
        if isinstance(session_id, bool) or not isinstance(session_id, int) or session_id < 0:
            raise ValueError("session_id must be a non-negative integer.")
        if snapshot.frame_sequence == self._last_frame_sequence:
            if self._last_preparation is None:
                raise RuntimeError("duplicate frame arrived before preparation output.")
            return self._last_preparation
        if locked_ids is not None:
            locked_ids = tuple(locked_ids)
            if (
                not locked_ids
                or len(set(locked_ids)) != len(locked_ids)
                or any(
                    isinstance(item, bool)
                    or not isinstance(item, int)
                    or item <= 0
                    for item in locked_ids
                )
            ):
                raise ValueError(
                    "locked_ids must contain unique positive integers or None."
                )
        if locked_ids != self._locked_ids:
            previous_plan = (
                None
                if self._last_selection is None
                else self._last_selection.plan
            )
            self._clear_confirmation()
            self._locked_ids = locked_ids
            if locked_ids is None:
                self._locked_member_classes = None
                self._locked_member_points.clear()
            elif previous_plan is not None and previous_plan.member_ids == locked_ids:
                self._locked_member_classes = tuple(
                    member.observation.target_class
                    for member in previous_plan.members
                )
            else:
                self._locked_member_classes = None

        self.tracker.set_handoff_prior(
            handoff_prior
            if locked_ids is None
            else None
        )
        targets = self.tracker.update(snapshot.capture_timestamp_ns, snapshot.observations)
        targets = self._canonicalize_locked_targets(targets, locked_ids)
        targets = self._freeze_distinct_scene_targets(targets)
        excluded = tuple(
            snapshot.observations[index]
            for index in excluded_observation_indices
        )
        # Keep the complete scene as obstacle evidence; only candidacy changes.
        targets = tuple(
            replace(target, selectable=False)
            if any(
                _same_frame_observation(target.observation, item)
                for item in excluded
            )
            else target
            for target in targets
        )
        selection = self.selector.select(
            targets,
            locked_ids=locked_ids,
            policy=policy,
            require_handoff=require_handoff,
            # 先在当前朝向检查左右末端的物理可达性；只有边界确实越过
            # 一侧行程时才生成一次 ALIGNING 计划，不能因中心偏离就旋转。
            align=True,
        )
        self._last_selection = selection
        plan = selection.plan
        # The scene decision owns its core and counts this very frame. The
        # consumer never needs to return a lock before confirmation can begin.
        if locked_ids is None and plan is not None:
            locked_ids = plan.member_ids
            self._locked_ids = locked_ids
        if (
            locked_ids is not None
            and self._locked_member_classes is None
            and plan is not None
            and plan.member_ids == locked_ids
        ):
            self._locked_member_classes = tuple(
                member.observation.target_class for member in plan.members
            )
        if locked_ids is not None and plan is not None and plan.member_ids == locked_ids:
            self._locked_member_points = {
                member.track_id: member.observation.ground_point
                for member in plan.members
                if member.observation.ground_point is not None
            }
        if plan is not None and self._locked_member_classes is not None:
            current_classes = tuple(
                member.observation.target_class for member in plan.members
            )
            if current_classes != self._locked_member_classes:
                # 已锁定目标的任务类别不能随局部 tracker 的类别抖动静默改变；
                # 按当前尝试失效处理，等待新的有效观察。
                selection = GraspSelection(
                    None,
                    ("locked_member_class_changed",),
                    plan,
                )
                plan = None
        class_changed = False
        if locked_ids is not None and self._locked_member_classes is not None:
            observed_classes = {
                target.track_id: target.observation.target_class
                for target in targets
                if target.track_id in locked_ids and target.observed
            }
            class_changed = any(
                track_id in observed_classes
                and observed_classes[track_id] is not expected
                for track_id, expected in zip(
                    locked_ids,
                    self._locked_member_classes,
                    strict=True,
                )
            )
        explicit_invalidation = class_changed or self._is_explicit_invalidation(
            selection,
            targets,
            locked_ids,
        )
        if explicit_invalidation:
            self._clear_confirmation()
            if not any(
                reason.startswith(
                    (
                        "blocked_target:",
                        "side_adjacent_incompatible",
                        "orange_not_isolated_track:",
                        "orange_isolation_unknown_ground_track:",
                        "maximum_opening_exceeded",
                        "forward_distance_exceeded",
                        "outside_near_field",
                        "member_class_risk:",
                        "member_outside_opening:",
                        "locked_member_class_changed",
                    )
                )
                for reason in selection.rejections
            ):
                selection = GraspSelection(
                    None,
                    ("confirmation_invalidated_explicit_risk",),
                    selection.preview_plan,
                )
        elif locked_ids is not None and plan is not None:
            is_new_frame = (
                plan.frame_sequence
                != self._confirmation_last_frame_sequence
            )
            if is_new_frame:
                self._confirmation_last_frame_sequence = plan.frame_sequence
                self._confirmation_count = min(
                    self.selector.config.confirmation_frames,
                    self._confirmation_count + 1,
                )
                self._confirmation_plan = plan
            elif self._confirmation_plan is None:
                self._confirmation_plan = plan
            if self._confirmation_count >= self.selector.config.confirmation_frames:
                self._confirmation_plan = plan
        ready = (
            locked_ids is not None
            and self._confirmation_plan is not None
            and self._confirmation_count >= self.selector.config.confirmation_frames
        )
        if plan is None and ready:
            # 一次漏检不撤销已经取得的确认；提交时仍会检查确认计划的
            # 真实采集年龄，不能用当前帧时刻伪造旧几何的新鲜度。
            selection = GraspSelection(
                self._confirmation_plan,
                selection.rejections,
                selection.preview_plan,
            )
        elif plan is not None and ready:
            selection = GraspSelection(
                plan,
                selection.rejections,
                selection.preview_plan,
            )
        result = GraspPreparation(
            snapshot.capture_timestamp_ns,
            selection,
            targets,
            ready,
            checked_member_ids=locked_ids,
            session_id=session_id,
            confirmation_count=min(
                self._confirmation_count,
                self.selector.config.confirmation_frames,
            ),
            confirmation_required=self.selector.config.confirmation_frames,
            result_timestamp_ns=snapshot.result_timestamp_ns,
        )
        if plan is None and recovery_context is not None:
            result = self._prepare_recovery(result, recovery_context, policy, frozenset(
                target.track_id for target in targets
                if any(_same_frame_observation(target.observation, item) for item in excluded)))
        self._last_frame_sequence = snapshot.frame_sequence
        self._last_preparation = result
        return result

    def _prepare_recovery(self, prepared: GraspPreparation, context: BreakupSceneContext,
                          policy: NearFieldGraspPolicy, excluded: frozenset[int]) -> GraspPreparation:
        # Missing contact or danger geometry is evidence to reobserve, never
        # permission to push. Only concrete grasp obstruction invokes recovery.
        blocked = any(reason.startswith(("blocked_target:", "maximum_opening_exceeded",
                                         "left_tip_y_mm", "right_tip_y_mm", "locked_members_changed"))
                      for reason in prepared.selection.rejections)
        if not blocked:
            return prepared
        if any(item.observed and item.observation.target_class is TargetClass.BLUE_DANGER
               and item.observation.ground_point is None for item in prepared.targets):
            return prepared
        targets = []
        for item in prepared.targets:
            observation = item.observation
            if not item.observed or observation.ground_point is None:
                continue
            radius, safety_radius = physical_radii(self.selector.target_geometry.geometry_for(observation.target_class))
            targets.append(BreakupTarget(item.track_id, prepared.capture_timestamp_ns,
                                         observation.target_class, observation.ground_point,
                                         radius, safety_radius))
        previous = context.previous_plan
        rejected: list[str] = []
        candidates = plan_breakup(
            tuple(targets), config=context.config, origin=context.origin,
            heading_rad=context.heading_rad, static_map=context.static_map,
            field_bounds=context.field_bounds, front_mm=context.front_mm,
            allowed_classes=policy.allowed_classes, approach=False,
            attempt=context.attempt, previous_aim=None if previous is None else previous.aim_field,
            previous_penetration_mm=0.0 if previous is None else previous.penetration_mm,
            non_contact_ids=frozenset(excluded),
            rejection_reasons=prepared.selection.rejections, rejections=rejected,
        )
        if context.objective_field is not None:
            objective = context.objective_field
            candidates = tuple(candidate for candidate in candidates
                               if any(math.hypot(point.x-objective.x, point.y-objective.y)
                                      <= context.config.cluster_group_ground_mm
                                      for point in candidate.member_field_points))
        if self._recovery_plan is not None:
            anchor = self._recovery_plan.aim_field
            candidates = tuple(sorted(candidates, key=lambda item:
                math.hypot(item.aim_field.x-anchor.x, item.aim_field.y-anchor.y)))
        candidate = candidates[0] if candidates else None
        if candidate is None:
            self._recovery_count = 0
            self._recovery_plan = None
        else:
            same = (self._recovery_plan is not None and
                    math.hypot(candidate.aim_field.x-self._recovery_plan.aim_field.x,
                               candidate.aim_field.y-self._recovery_plan.aim_field.y)
                    <= context.config.cluster_group_ground_mm)
            self._recovery_count = min(context.config.breakup_confirmation_frames,
                                       self._recovery_count + 1 if same else 1)
            self._recovery_plan = candidate
        return replace(prepared, recovery_plan=candidate, recovery_checked=True,
                       ready=candidate is not None and self._recovery_count >= context.config.breakup_confirmation_frames,
                       confirmation_count=self._recovery_count,
                       confirmation_required=context.config.breakup_confirmation_frames)


class GraspPreparationWorker:
    """有界的最新帧近场准备线程；不在运动控制线程执行几何计算。"""

    def __init__(
        self,
        session: GraspPreparationSession,
        selector: NearFieldGraspSelector,
        *,
        diagnostics_callback=None,
    ):
        if not isinstance(session, GraspPreparationSession):
            raise TypeError("session must be a GraspPreparationSession.")
        if not isinstance(selector, NearFieldGraspSelector):
            raise TypeError("selector must be a NearFieldGraspSelector.")
        self.session = session
        self.selector = selector
        self._condition = Condition()
        self._pending: tuple[
            PerceptionSnapshot,
            tuple[int, ...] | None,
            NearFieldGraspPolicy,
            int,
            NearFieldHandoffPrior | None,
            frozenset[int],
            bool,
            BreakupSceneContext | None,
        ] | None = None
        self._latest: GraspPreparation | None = None
        self._computing_session_id: int | None = None
        self._stopping = False
        self._started = False
        self._last_submitted_timestamp_ns = -1
        self._worker_session_id: int | None = None
        self._requested_session_id = 0
        self._diagnostics_callback = diagnostics_callback
        self._pending_diagnostic: tuple[GraspPreparation, NearFieldGraspPolicy] | None = None
        self._diagnostic_thread = Thread(target=self._diagnostic_loop, name="grasp-diagnostics", daemon=True)
        self.error: str | None = None
        self._thread = Thread(target=self._loop, name="near-field-grasp", daemon=True)

    def __enter__(self) -> "GraspPreparationWorker":
        self._thread.start()
        self._diagnostic_thread.start()
        self._started = True
        return self

    def __exit__(self, *_args: object) -> None:
        if not self._started:
            return
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
        self._thread.join()
        self._diagnostic_thread.join()

    def begin(self, session_id: int) -> None:
        """切换到新运输轮次；旧结果立即失效。"""

        if isinstance(session_id, bool) or not isinstance(session_id, int) or session_id < 0:
            raise ValueError("session_id must be a non-negative integer.")
        with self._condition:
            self._latest = None
            self._pending = None
            self._worker_session_id = None
            self._last_submitted_timestamp_ns = -1
            self.error = None
            self._pending_diagnostic = None
            self._requested_session_id = session_id
            self._condition.notify_all()

    def submit(
        self,
        snapshot: PerceptionSnapshot,
        *,
        session_id: int,
        policy: NearFieldGraspPolicy,
        locked_ids: tuple[int, ...] | None,
        handoff_prior: NearFieldHandoffPrior | None = None,
        excluded_observation_indices: frozenset[int] = frozenset(),
        require_handoff: bool = False,
        recovery_context: BreakupSceneContext | None = None,
    ) -> None:
        if not isinstance(snapshot, PerceptionSnapshot):
            raise TypeError("snapshot must be a PerceptionSnapshot.")
        _validate_candidate_exclusions(snapshot, excluded_observation_indices)
        if isinstance(session_id, bool) or not isinstance(session_id, int) or session_id < 0:
            raise ValueError("session_id must be a non-negative integer.")
        if not isinstance(policy, NearFieldGraspPolicy):
            raise TypeError("policy must be a NearFieldGraspPolicy.")
        if handoff_prior is not None and not isinstance(
            handoff_prior, NearFieldHandoffPrior
        ):
            raise TypeError(
                "handoff_prior must be a NearFieldHandoffPrior or None."
            )
        if not isinstance(require_handoff, bool):
            raise ValueError("require_handoff must be a boolean.")
        with self._condition:
            if session_id != getattr(self, "_requested_session_id", session_id):
                return
            if snapshot.capture_timestamp_ns <= self._last_submitted_timestamp_ns:
                return
            self._last_submitted_timestamp_ns = snapshot.capture_timestamp_ns
            self._pending = (
                snapshot,
                locked_ids,
                policy,
                session_id,
                handoff_prior,
                excluded_observation_indices,
                require_handoff,
                recovery_context,
            )
            self._condition.notify_all()

    def pending(self, session_id: int) -> bool:
        with self._condition:
            return (session_id == self._requested_session_id and
                    (self._pending is not None or self._computing_session_id == session_id))

    def latest(self, session_id: int) -> GraspPreparation | None:
        with self._condition:
            result = self._latest
            return result if result is not None and result.session_id == session_id else None

    def _loop(self) -> None:
        try:
            while True:
                with self._condition:
                    while not self._stopping and self._pending is None:
                        self._condition.wait(0.03)
                    if self._stopping:
                        return
                    request, self._pending = self._pending, None
                    self._computing_session_id = self._requested_session_id
                assert request is not None
                (
                    snapshot,
                    locked_ids,
                    policy,
                    session_id,
                    handoff_prior,
                    excluded_observation_indices,
                    require_handoff,
                    recovery_context,
                ) = request
                if self._worker_session_id != session_id:
                    self.session.reset()
                    self._worker_session_id = session_id
                try:
                    prepared = self.session.update(
                        snapshot,
                        locked_ids=locked_ids,
                        policy=policy,
                        session_id=session_id,
                        handoff_prior=handoff_prior,
                        excluded_observation_indices=excluded_observation_indices,
                        require_handoff=require_handoff,
                        recovery_context=recovery_context,
                    )
                except Exception as exc:
                    with self._condition:
                        # 已完成的唯一确认窗口不能因旁路一次异常被抹掉；
                        # 动作层仍会检查其中计划的真实年龄。
                        self.error = f"planning:{type(exc).__name__}:{exc}"
                        self._computing_session_id = None
                    continue
                prepared = replace(
                    prepared,
                    prepared_timestamp_ns=time.monotonic_ns(),
                )
                with self._condition:
                    self._computing_session_id = None
                    if session_id == getattr(self, "_requested_session_id", session_id):
                        self._latest = prepared
                        self.error = None
                        if self._diagnostics_callback is not None:
                            self._pending_diagnostic = (prepared, policy)
                            self._condition.notify_all()
        finally:
            with self._condition:
                self._stopping = True
                self._condition.notify_all()

    def _diagnostic_loop(self) -> None:
        # 单个最新结果槽；先限频再算几何。慢日志不能占住下一帧准备器。
        next_report_ns = 0
        while True:
            with self._condition:
                while not self._stopping:
                    delay_ns = next_report_ns - time.monotonic_ns()
                    if self._pending_diagnostic is not None and delay_ns <= 0:
                        break
                    self._condition.wait(timeout=max(0.001, delay_ns / 1e9) if delay_ns > 0 else None)
                if self._stopping:
                    return
                pending = self._pending_diagnostic
                self._pending_diagnostic = None
            assert pending is not None
            prepared, policy = pending
            try:
                diagnostics = self.selector.candidate_geometries(prepared.targets, policy=policy)
                with self._condition:
                    current = prepared.session_id == self._requested_session_id
                if current:
                    self._diagnostics_callback(diagnostics)
            except Exception as exc:
                with self._condition:
                    self.error = f"diagnostics:{type(exc).__name__}:{exc}"
            next_report_ns = time.monotonic_ns() + 1_000_000_000


def _positive(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}.")
    return float(value)


def _nonnegative(value: float, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(
            f"{name} must be finite and non-negative, got {value!r}."
        )
    return float(value)


@dataclass(frozen=True, slots=True)
class GripperWidthPickupResult:
    member_ids: tuple[int, ...]
    member_classes: tuple[str, ...]
    completed_timestamp_ns: int
    final_servo_angles_deg: tuple[float, float]
    capture_confirmed: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.member_ids, tuple) or not self.member_ids or len(set(self.member_ids)) != len(self.member_ids) or any(isinstance(item, bool) or not isinstance(item, int) or item <= 0 for item in self.member_ids):
            raise ValueError("member_ids must contain unique positive integers.")
        if not isinstance(self.member_classes, tuple) or len(self.member_classes) != len(self.member_ids) or not all(isinstance(item, str) and item for item in self.member_classes):
            raise ValueError("member_classes must match member_ids with non-empty strings.")
        if isinstance(self.completed_timestamp_ns, bool) or not isinstance(self.completed_timestamp_ns, int) or self.completed_timestamp_ns < 0:
            raise ValueError("completed_timestamp_ns must be a non-negative integer.")
        if not isinstance(self.final_servo_angles_deg, tuple) or len(self.final_servo_angles_deg) != 2 or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or not 0.0 <= float(item) <= 180.0 for item in self.final_servo_angles_deg):
            raise ValueError("final_servo_angles_deg must contain two angles in [0, 180].")
        if not isinstance(self.capture_confirmed, bool):
            raise ValueError("capture_confirmed must be a boolean.")


@dataclass(frozen=True, slots=True)
class GripperWidthPickupDecision:
    timestamp_ns: int
    state: GripperWidthPickupState
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    gripper_angles_deg: tuple[float, float] | None
    soft_brake: bool
    reason: str
    min_wheel_velocity_m_s: float | None = None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if not isinstance(self.state, GripperWidthPickupState):
            raise TypeError("state must be a GripperWidthPickupState.")
        for name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite, got {value!r}.")
            object.__setattr__(self, name, float(value))
        if self.gripper_angles_deg is not None and (not isinstance(self.gripper_angles_deg, tuple) or len(self.gripper_angles_deg) != 2 or any(isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(float(item)) or not 0.0 <= float(item) <= 180.0 for item in self.gripper_angles_deg)):
            raise ValueError("gripper_angles_deg must contain two angles in [0, 180] or None.")
        if not isinstance(self.soft_brake, bool):
            raise ValueError("soft_brake must be a boolean.")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string.")
        if self.min_wheel_velocity_m_s is not None and (
            isinstance(self.min_wheel_velocity_m_s, bool)
            or not isinstance(self.min_wheel_velocity_m_s, (int, float))
            or not math.isfinite(float(self.min_wheel_velocity_m_s))
            or float(self.min_wheel_velocity_m_s) < 0.0
        ):
            raise ValueError(
                "min_wheel_velocity_m_s must be finite and non-negative or None."
            )


class GripperWidthPickupSequence:
    def __init__(self, *, gripper_full_travel_time_s: float, forward_speed_m_s: float,
                 closed_servo_angles_deg: tuple[float, float], max_observation_age_ms: float,
                 alignment_kp_rad_s: float = 1.0, alignment_max_angular_velocity_rad_s: float = 0.35,
                 alignment_min_wheel_velocity_m_s: float | None = None,
                 alignment_timeout_ms: float = 8_000.0,
                 alignment_continue_max_age_ms: float = 400.0,
                 grasp_commit_max_observation_age_ms: float = 150.0,
                 stationary_max_gyro_rad_s: float = 0.03,
                 fine_alignment_zone_rad: float = 0.08,
                 fine_alignment_min_wheel_velocity_m_s: float = 0.0,
                 cruise_speed_scale: float = 1.0,
                 terminal_speed_gain_s_inv: float = 1.0,
                 deceleration_m_s2: float = 0.5):
        self.travel_ns = round(_positive(gripper_full_travel_time_s, "gripper_full_travel_time_s") * 1e9)
        self.speed = _positive(forward_speed_m_s, "forward_speed_m_s")
        self.cruise_speed_scale = _positive(cruise_speed_scale, "cruise_speed_scale")
        self.terminal_speed_gain_s_inv = _positive(
            terminal_speed_gain_s_inv, "terminal_speed_gain_s_inv"
        )
        self.deceleration_m_s2 = _positive(deceleration_m_s2, "deceleration_m_s2")
        self.age_ns = round(_positive(max_observation_age_ms, "max_observation_age_ms") * 1e6)
        # 对准续转窗口宽于单帧观测年龄：感知慢于一帧时不中断旋转。
        self.alignment_continue_ns = round(
            _positive(alignment_continue_max_age_ms, "alignment_continue_max_age_ms") * 1e6
        )
        self.commit_age_ns = round(
            _positive(
                grasp_commit_max_observation_age_ms,
                "grasp_commit_max_observation_age_ms",
            )
            * 1e6
        )
        self.stationary_max_gyro_rad_s = _positive(stationary_max_gyro_rad_s, "stationary_max_gyro_rad_s")
        self.motion_evidence = StationaryMotionEvidence(
            max_gap_ns=self.commit_age_ns, max_gyro_rad_s=self.stationary_max_gyro_rad_s,
        )
        self.fine_alignment_zone_rad = _positive(
            fine_alignment_zone_rad,
            "fine_alignment_zone_rad",
        )
        self.fine_alignment_min_wheel_velocity_m_s = _nonnegative(
            fine_alignment_min_wheel_velocity_m_s,
            "fine_alignment_min_wheel_velocity_m_s",
        )
        self.kp = _positive(alignment_kp_rad_s, "alignment_kp_rad_s")
        self.max_angular = _positive(alignment_max_angular_velocity_rad_s, "alignment_max_angular_velocity_rad_s")
        self.alignment_timeout_ns = round(
            _positive(alignment_timeout_ms, "alignment_timeout_ms") * 1e6
        )
        self.alignment_min_wheel_velocity_m_s = (
            None
            if alignment_min_wheel_velocity_m_s is None
            else _positive(
                alignment_min_wheel_velocity_m_s,
                "alignment_min_wheel_velocity_m_s",
            )
        )
        if not isinstance(closed_servo_angles_deg, tuple) or len(closed_servo_angles_deg) != 2 or any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(float(v)) or not 0 <= float(v) <= 180 for v in closed_servo_angles_deg):
            raise ValueError(f"Invalid closed servo angles {closed_servo_angles_deg!r}.")
        self.closed_angles = closed_servo_angles_deg
        self.state = GripperWidthPickupState.SEARCH
        self.locked_ids: tuple[int, ...] | None = None
        self.active_plan: NearFieldGraspPlan | None = None
        self.result: GripperWidthPickupResult | None = None
        self._last_ns = -1
        self._phase_ns = 0
        self._forward_heading_rad: float | None = None
        self._distance_start: float | None = None
        self._last_progress_mm = 0.0
        self._progress_ns = 0
        self._abort_capture_ns = -1
        self._rejected_capture_ns = -1
        self._abort_reason = ""
        self._alignment_plan: NearFieldGraspPlan | None = None
        self._alignment_started_ns: int | None = None
        self._alignment_start_heading_rad: float | None = None
        self._alignment_turn_started_ns: int | None = None
        self._alignment_direction = 0.0
        self._alignment_budget_rad = 0.0
        self._alignment_progress_rad = 0.0
        self._alignment_attempts = 0
        self.alignment_motion_allowance_ns = 0
        self._alignment_completed_ns = -1
        self._alignment_finished = False

    def observe_motion(self, message: OdometryImu) -> None:
        """消费真实编码器/IMU样本；接收时间与相机同为主机单调时钟。

        计数变化、旋转、无效传感器、样本倒退或遥测间断均使静止证据失效。
        不把零速指令、重复消息或重新发布的视觉结果当作静止证据。
        """
        self.motion_evidence.observe(message)

    def motion_diagnostic(self, timestamp_ns: int) -> str:
        """记录静止证据来源，区分视觉延迟和底盘实际仍在移动。"""
        return self.motion_evidence.diagnostic(timestamp_ns)

    def _stationary_plan_valid(self, now_ns: int, prep: GraspPreparation) -> bool:
        plan = prep.selection.plan
        return (
            plan is not None
            and self.motion_evidence.capture_valid(
                plan.capture_timestamp_ns, now_ns, max_age_ns=self.alignment_timeout_ns
            )
            and (publication_age := prep.preparation_age_ns(now_ns)) is not None
            and 0 <= publication_age <= self.age_ns
            # 缺失后缓存的确认计划不能借新快照/新发布时间续命。
            and plan.capture_timestamp_ns == prep.capture_timestamp_ns
            and self._current_plan_evidence_valid(prep)
        )

    @staticmethod
    def _current_plan_evidence_valid(prep: GraspPreparation) -> bool:
        """Require clean current-frame evidence before opening the jaws."""

        plan = prep.selection.plan
        if plan is None or plan.capture_timestamp_ns != prep.capture_timestamp_ns:
            return False
        by_id = {target.track_id: target for target in prep.targets}
        return all(
            (current := by_id.get(member.track_id)) is not None
            and current.observed
            and current.selectable
            and current.envelope is not None
            for member in plan.members
        )

    def reset(self) -> None:
        """为下一趟运输清空动作状态。"""

        self.state = GripperWidthPickupState.SEARCH
        self.locked_ids = None
        self.active_plan = None
        self.result = None
        self._last_ns = -1
        self._phase_ns = 0
        self._distance_start = None
        self._last_progress_mm = 0.0
        self._progress_ns = 0
        self._abort_capture_ns = -1
        self._rejected_capture_ns = -1
        self._abort_reason = ""
        self._alignment_plan = None
        self._alignment_started_ns = None
        self._alignment_start_heading_rad = None
        self._alignment_turn_started_ns = None
        self._alignment_direction = 0.0
        self._alignment_budget_rad = 0.0
        self._alignment_progress_rad = 0.0
        self._alignment_attempts = 0
        self.alignment_motion_allowance_ns = 0
        self._alignment_completed_ns = -1
        self._alignment_finished = False

    def progress_mm(self, cumulative_distance_m: float | None) -> float:
        return 0.0 if cumulative_distance_m is None or self._distance_start is None else max(0.0, (cumulative_distance_m - self._distance_start) * 1000)

    def _abort(self, now_ns: int, reason: str, prep: GraspPreparation | None) -> GripperWidthPickupDecision:
        self.state = GripperWidthPickupState.ABORTED
        self._abort_capture_ns = now_ns  # 必须是动作退出之后采集的新证据
        self._abort_reason = reason
        self.active_plan = None
        self.locked_ids = None
        self._alignment_plan = None
        self._alignment_started_ns = None
        self._alignment_start_heading_rad = None
        self._alignment_turn_started_ns = None
        self._alignment_direction = 0.0
        self._alignment_budget_rad = 0.0
        self._alignment_progress_rad = 0.0
        self._alignment_finished = False
        return self._decision(now_ns, reason, brake=True)

    def _replan(self, now_ns: int, reason: str) -> GripperWidthPickupDecision:
        """开爪前证据变化只解锁重选，并清空旧候选的对准计时。"""

        self.state = GripperWidthPickupState.SEARCH
        self._rejected_capture_ns = now_ns
        self.active_plan = None
        self.locked_ids = None
        self._alignment_plan = None
        self._alignment_started_ns = None
        self._alignment_start_heading_rad = None
        self._alignment_turn_started_ns = None
        self._alignment_direction = 0.0
        self._alignment_budget_rad = 0.0
        self._alignment_progress_rad = 0.0
        self._alignment_finished = False
        return self._decision(now_ns, f"candidate_replan:{reason}", brake=True)

    def alignment_timeout_reached(self, timestamp_ns: int) -> bool:
        """报告当前近场尝试的对准/确认总预算是否已超时。"""

        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError(f"Invalid timestamp_ns {timestamp_ns!r}.")
        return (
            self._alignment_started_ns is not None
            and self.state
            in {
                GripperWidthPickupState.SEARCH,
                GripperWidthPickupState.ALIGNING,
                GripperWidthPickupState.VERIFYING,
            }
            and self.active_plan is None
            and timestamp_ns - self._alignment_started_ns
            >= self.alignment_timeout_ns + self.alignment_motion_allowance_ns
        )

    def hold_for_static_path_validation(
        self,
        timestamp_ns: int,
        member_ids: tuple[int, ...],
    ) -> None:
        """锁定待复核组，并让静态路径等待也受对准时限约束。"""

        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError(f"Invalid timestamp_ns {timestamp_ns!r}.")
        member_ids = tuple(member_ids)
        if (
            not member_ids
            or len(set(member_ids)) != len(member_ids)
            or any(
                isinstance(item, bool) or not isinstance(item, int) or item <= 0
                for item in member_ids
            )
        ):
            raise ValueError("member_ids must contain unique positive integers.")
        if self.active_plan is not None:
            raise RuntimeError("Cannot wait for a path while an active plan exists.")
        if self.state not in {
            GripperWidthPickupState.SEARCH,
            GripperWidthPickupState.ALIGNING,
            GripperWidthPickupState.VERIFYING,
        }:
            raise RuntimeError(
                "Static path validation is only valid before the active plan."
            )
        if self.locked_ids is not None and self.locked_ids != member_ids:
            raise RuntimeError("Static path validation plan changed while locked.")
        self.locked_ids = member_ids
        self.state = GripperWidthPickupState.VERIFYING
        if self._alignment_started_ns is None:
            self._alignment_started_ns = timestamp_ns

    def _decision(self, now_ns: int, reason: str, *, speed: float = 0.0, angular: float = 0.0,
                  angles: tuple[float, float] | None = None, brake: bool = False) -> GripperWidthPickupDecision:
        minimum = self.alignment_min_wheel_velocity_m_s
        return GripperWidthPickupDecision(
            now_ns,
            self.state,
            speed,
            angular,
            angles,
            brake,
            reason,
            min_wheel_velocity_m_s=(
                minimum
                if self.state is GripperWidthPickupState.ALIGNING
                else (0.0 if speed > 0.0 else None)
            ),
        )

    def _start_alignment_turn(
        self,
        now_ns: int,
        plan: NearFieldGraspPlan,
        heading_rad: float | None,
    ) -> None:
        """Start one bounded coarse/correction turn for the locked target."""

        self._alignment_attempts += 1
        # At most two turns; each gets only the time its requested angle needs.
        turn_speed = max(0.05, min(self.max_angular, self.kp * abs(plan.alignment_angle_rad)))
        if heading_rad is not None:
            self.alignment_motion_allowance_ns += round((abs(plan.alignment_angle_rad) / turn_speed + 0.2) * 1e9)
        self._alignment_plan = plan
        self._alignment_direction = math.copysign(
            1.0,
            plan.alignment_angle_rad,
        )
        self._alignment_budget_rad = min(
            abs(plan.alignment_angle_rad),
            math.pi / 2.0,
        )
        self._alignment_progress_rad = 0.0
        self._alignment_start_heading_rad = heading_rad
        self._alignment_turn_started_ns = now_ns
        if self._alignment_started_ns is None:
            self._alignment_started_ns = now_ns
        self.state = GripperWidthPickupState.ALIGNING

    def _alignment_turn_complete(
        self,
        now_ns: int,
        heading_rad: float | None,
    ) -> bool:
        """Use IMU progress when available, otherwise a bounded fallback timer."""

        start = self._alignment_start_heading_rad
        bounded_elapsed = (
            self._alignment_turn_started_ns is not None
            and now_ns - self._alignment_turn_started_ns
            >= min(self.alignment_continue_ns, 500_000_000)
        )
        if heading_rad is not None and start is not None:
            delta = normalize_angle(heading_rad - start)
            self._alignment_progress_rad = max(
                0.0,
                delta if self._alignment_direction > 0.0 else -delta,
            )
            return (
                self._alignment_progress_rad >= self._alignment_budget_rad - 1e-3
            )
        if self._alignment_started_ns is None:
            return False
        # Without a motion history there is deliberately no time compensation;
        # this is only a short, bounded turn after which a new frame must prove
        # the geometry again.
        return bounded_elapsed

    def step(
        self,
        timestamp_ns: int,
        preparation: GraspPreparation | None,
        *,
        cumulative_distance_m: float | None,
        path_clear: bool = True,
        heading_rad: float | None = None,
    ) -> GripperWidthPickupDecision:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0 or timestamp_ns < self._last_ns:
            raise ValueError(f"Invalid/nonmonotonic timestamp_ns {timestamp_ns!r}.")
        self._last_ns = timestamp_ns
        if cumulative_distance_m is not None and (isinstance(cumulative_distance_m, bool) or not math.isfinite(cumulative_distance_m)):
            raise ValueError(f"Invalid cumulative_distance_m {cumulative_distance_m!r}.")
        if heading_rad is not None and (
            isinstance(heading_rad, bool)
            or not isinstance(heading_rad, (int, float))
            or not math.isfinite(float(heading_rad))
        ):
            raise ValueError(f"Invalid heading_rad {heading_rad!r}.")
        if not isinstance(path_clear, bool):
            raise ValueError("path_clear must be a boolean.")
        if preparation is not None and not isinstance(preparation, GraspPreparation):
            raise TypeError("preparation must be a GraspPreparation or None.")
        now, prep = timestamp_ns, preparation
        if self.state is GripperWidthPickupState.COMPLETE:
            return self._decision(now, "complete_capture_unconfirmed", brake=True)
        current_prep = (
            prep
            if prep is not None and (0 <= now - prep.capture_timestamp_ns <= self.age_ns
                                     or self._stationary_plan_valid(now, prep))
            else None
        )
        if (
            self.active_plan is None
            and self.state
            in {
                GripperWidthPickupState.SEARCH,
                GripperWidthPickupState.ALIGNING,
                GripperWidthPickupState.VERIFYING,
            }
            and self._alignment_started_ns is None
        ):
            # 预算从首次进入近场尝试开始，覆盖无结果等待、必要对准和
            # 唯一确认窗口；重复控制周期不会重新起算。
            self._alignment_started_ns = now
        if self.alignment_timeout_reached(now):
            self.state = GripperWidthPickupState.SEARCH
            self.locked_ids = None
            self._alignment_plan = None
            self._alignment_started_ns = None
            self._alignment_start_heading_rad = None
            self._alignment_turn_started_ns = None
            self._alignment_direction = 0.0
            self._alignment_budget_rad = 0.0
            self._alignment_progress_rad = 0.0
            return self._decision(now, "alignment_timeout", brake=True)
        if self.state is GripperWidthPickupState.ABORTED:
            if (
                current_prep is None
                or current_prep.capture_timestamp_ns <= self._abort_capture_ns
                or current_prep.checked_member_ids is not None
                or current_prep.selection.plan is None
            ):
                return self._decision(now, self._abort_reason, brake=True)
            self.state = GripperWidthPickupState.SEARCH
            self.result = None
        if self.active_plan is None:
            plan = current_prep.selection.plan if current_prep is not None else None
            matching = current_prep is not None and current_prep.checked_member_ids == self.locked_ids
            if self.locked_ids is None and plan is not None and current_prep is not None:
                matching = current_prep.checked_member_ids in (None, plan.member_ids)
                if not matching:
                    plan = None
            if plan is not None and plan.capture_timestamp_ns <= self._rejected_capture_ns:
                plan = None
            if self.locked_ids is not None and not matching:
                plan = None
            if plan is None:
                hard_failure = (
                    current_prep is not None
                    and matching
                    and any(
                        reason.startswith(
                            (
                                "blocked_target:",
                                "side_adjacent_incompatible",
                                "orange_not_isolated_track:",
                                "orange_isolation_unknown_ground_track:",
                                "ineligible_members_or_capacity",
                                "orange_injured_single_only",
                                "locked_members_changed",
                                "locked_member_class_changed",
                                "maximum_opening_exceeded",
                                "forward_distance_exceeded",
                                "outside_near_field",
                                "member_class_risk:",
                                "member_outside_opening:",
                                "confirmation_invalidated_explicit_risk",
                                # 对准后仍够不到：不是数据问题，等下去也不会好，
                                # 解锁重选并把证据交给上层升级到解团。
                                "left_tip_y_mm",
                                "right_tip_y_mm",
                            )
                        )
                        for reason in current_prep.selection.rejections
                    )
                )
                if self.locked_ids is not None and hard_failure:
                    return self._replan(
                        now,
                        "candidate_invalid:" + ",".join(current_prep.selection.rejections),
                    )
                if self.state is GripperWidthPickupState.ALIGNING and self._alignment_plan is not None:
                    # A missing result is not permission to keep turning on an
                    # old visual angle.  The one bounded turn ends on measured
                    # IMU progress (or on the short fallback budget) and then
                    # waits for a new current-frame geometry check.
                    if not self._alignment_finished and not self._alignment_turn_complete(now, heading_rad):
                        return self._decision(
                            now,
                            "alignment_recent_checked_plan",
                            angular=self._angular(self._alignment_plan),
                        )
                    if not self._alignment_finished:
                        self._alignment_finished = True
                        self._alignment_completed_ns = now
                    return self._decision(
                        now,
                        "waiting_locked_target_observation",
                        brake=True,
                    )
                if self.locked_ids is not None:
                    return self._decision(
                        now,
                        "waiting_locked_target_observation",
                        brake=True,
                    )
                return self._decision(now, "waiting_eligible_group", brake=True)
            if (self.motion_evidence.latest is not None
                    and self.state is not GripperWidthPickupState.ALIGNING
                    and not self._stationary_plan_valid(now, current_prep)):
                return self._decision(now, "waiting_stationary_scene", brake=True)
            self.locked_ids = plan.member_ids
            if plan.alignment_angle_rad != 0:
                if (
                    self.state is GripperWidthPickupState.ALIGNING
                    and self._alignment_plan is not None
                ):
                    new_after_finished = False
                    same_visual_plan = (
                        plan.frame_sequence == self._alignment_plan.frame_sequence
                        and plan.capture_timestamp_ns
                        == self._alignment_plan.capture_timestamp_ns
                    )
                    if self._alignment_finished:
                        if same_visual_plan or plan.capture_timestamp_ns <= self._alignment_completed_ns:
                            return self._decision(
                                now,
                                "alignment_waiting_for_new_current_geometry",
                                brake=True,
                            )
                        # A genuinely newer frame may justify the one allowed
                        # correction; it must not inherit the completed turn's
                        # progress or restart the overall attempt deadline.
                        self._alignment_plan = None
                        self._alignment_finished = False
                        new_after_finished = True
                    if not new_after_finished and self._alignment_turn_complete(now, heading_rad):
                        self._alignment_finished = True
                        self._alignment_completed_ns = now
                        return self._decision(
                            now,
                            "alignment_bounded_turn_complete_wait_current_geometry",
                            brake=True,
                        )
                    if not new_after_finished:
                        return self._decision(
                            now, "align_group_envelope" if self._alignment_attempts == 1 else "align_group_envelope_correction",
                            angular=self._angular(self._alignment_plan),
                        )
                same_visual_plan = (
                    self.state is GripperWidthPickupState.VERIFYING
                    and self._alignment_plan is not None
                    and plan.frame_sequence == self._alignment_plan.frame_sequence
                    and plan.capture_timestamp_ns
                    == self._alignment_plan.capture_timestamp_ns
                )
                if same_visual_plan:
                    return self._decision(
                        now,
                        "alignment_waiting_for_new_current_geometry",
                        brake=True,
                    )
                direction = math.copysign(1.0, plan.alignment_angle_rad)
                if self._alignment_attempts >= 2:
                    return self._replan(
                        now,
                        "alignment_attempt_budget_exhausted",
                    )
                if (
                    self._alignment_attempts > 0
                    and self._alignment_direction != 0.0
                    and direction != self._alignment_direction
                ):
                    # A reverse after the first bounded turn is the single
                    # allowed correction.  A second reversal is an actual
                    # failure, not another reason to oscillate.
                    if self._alignment_attempts >= 2:
                        return self._replan(
                            now,
                            "alignment_direction_changed_after_bounded_turn",
                        )
                self._start_alignment_turn(now, plan, heading_rad)
                return self._decision(
                    now,
                    "align_group_envelope" if self._alignment_attempts == 1 else "align_group_envelope_correction",
                    angular=self._angular(plan),
                )
            self.state = GripperWidthPickupState.VERIFYING
            self._alignment_plan = plan
            # 零角度只表示进入允许范围；唯一确认窗口由 preparation.ready
            # 给出，动作层不再串接第二个帧计数或静止等待。
            if self._alignment_started_ns is None:
                self._alignment_started_ns = now
            assert current_prep is not None
            plan_age_ns = current_prep.plan_age_ns(now)
            preparation_age_ns = current_prep.preparation_age_ns(now)
            if (
                plan_age_ns is None
                or (plan_age_ns > self.commit_age_ns
                    and not self._stationary_plan_valid(now, current_prep))
            ):
                return self._decision(
                    now,
                    "confirmation_waiting_for_fresh_plan",
                    brake=True,
                )
            if self.motion_evidence.latest is not None and not self._stationary_plan_valid(now, current_prep):
                return self._decision(now, "confirmation_waiting_for_stationary_capture", brake=True)
            if (preparation_age_ns is None or preparation_age_ns < 0
                    or (preparation_age_ns > self.commit_age_ns
                        and not self._stationary_plan_valid(now, current_prep))):
                return self._decision(
                    now,
                    "confirmation_waiting_for_fresh_preparation",
                    brake=True,
                )
            if not self._current_plan_evidence_valid(current_prep):
                return self._decision(
                    now,
                    "confirmation_waiting_for_current_clean_geometry",
                    brake=True,
                )
            if not current_prep.ready:
                return self._decision(
                    now,
                    "confirmation_in_progress",
                    brake=True,
                )
            if cumulative_distance_m is None:
                return self._decision(now, "waiting_odometry", brake=True)
            if not path_clear:
                return self._abort(now, "near_field_path_blocked", current_prep)
            self.active_plan = plan
            self.result = None
            self._alignment_started_ns = None
            self._distance_start = cumulative_distance_m
            self._forward_heading_rad = heading_rad
            self._phase_ns = now
            self.state = GripperWidthPickupState.OPENING
            return self._decision(now, "open_group_width", angles=plan.opening_servo_angles_deg, brake=True)

        plan = self.active_plan
        # 合爪指令已在有证据时提交；此后只有计时和零底盘意图。
        # 被夹臂遮挡不应把已提交动作改称需要重新抓取。
        if self.state is GripperWidthPickupState.CLOSING:
            if now - self._phase_ns < self.travel_ns:
                return self._decision(now, "closing_gripper", brake=True)
            self.state = GripperWidthPickupState.COMPLETE
            self.result = GripperWidthPickupResult(plan.member_ids, tuple(t.observation.target_class.value for t in plan.members), now, self.closed_angles)
            return self._decision(now, "complete_capture_unconfirmed", brake=True)
        if self.state is GripperWidthPickupState.OPENING:
            elapsed_ns = now - self._phase_ns
            if elapsed_ns < self.travel_ns:
                if cumulative_distance_m is None:
                    return self._abort(now, "critical_odometry_unavailable", current_prep)
                if self._distance_start is not None and cumulative_distance_m < self._distance_start - 0.005:
                    return self._abort(now, "encoder_direction_mismatch", current_prep)
                speed = self._opening_advance_speed(plan, cumulative_distance_m)
                return self._decision(now, "opening_gripper_while_advancing" if speed > 0 else "opening_gripper", speed=speed, angular=self._forward_heading_correction(heading_rad, speed) if speed > 0 else 0.0, brake=speed == 0)

            # 走廊证据只在静止确认时检查；开爪完成后按冻结执行计划开环前进。
            if cumulative_distance_m is None:
                return self._abort(now, "critical_odometry_unavailable", current_prep)
            if self._distance_start is not None and cumulative_distance_m < self._distance_start - 0.005:
                return self._abort(now, "encoder_direction_mismatch", current_prep)
            self.state = GripperWidthPickupState.FORWARD
            self._last_progress_mm = 0.0
            self._progress_ns = now

        # 视觉快照在这里不再是运动门禁；定距和运动健康仍是硬门禁。
        if cumulative_distance_m is None:
            return self._abort(now, "critical_odometry_unavailable", current_prep)
        if self._distance_start is not None and cumulative_distance_m < self._distance_start - 0.005:
            return self._abort(now, "encoder_direction_mismatch", current_prep)
        if self.state is GripperWidthPickupState.FORWARD:
            progress = self.progress_mm(cumulative_distance_m)
            if progress >= self._last_progress_mm + 0.5:
                self._last_progress_mm = progress
                self._progress_ns = now
            if now - self._progress_ns > max(1_000_000_000, 2 * self.age_ns):
                return self._abort(now, "encoder_no_forward_progress", prep)
            if self.progress_mm(cumulative_distance_m) >= plan.forward_distance_mm:
                self.state = GripperWidthPickupState.CLOSING
                self._phase_ns = now
                return self._decision(now, "distance_reached_close_gripper", angles=self.closed_angles, brake=True)
            # 速度仍受剩余定距限制，但不受视觉观测年龄/退化状态降速。
            remaining_m = (plan.forward_distance_mm - self.progress_mm(cumulative_distance_m)) / 1000
            speed = approach_speed_m_s(remaining_m, self.speed, self.cruise_speed_scale,
                                        self.deceleration_m_s2, precision_approach=True,
                                        terminal_speed_gain_s_inv=self.terminal_speed_gain_s_inv)
            return self._decision(now, "forward_encoder_heading_hold", speed=speed,
                                  angular=self._forward_heading_correction(heading_rad, speed))
        return self._abort(now, "invalid_pickup_state", current_prep)

    def _forward_heading_correction(self, heading_rad: float | None, speed_m_s: float) -> float:
        # Keep the heading frozen at the checked opening pose, including servo
        # travel and the very first encoder step; never recenter on a new image.
        if heading_rad is None or self._forward_heading_rad is None:
            return 0.0
        error = normalize_angle(self._forward_heading_rad-heading_rad)
        # Deceleration scales steering too: no last-millimetre pivot while
        # the encoder endpoint is approached at a tiny forward speed.
        scale = min(1.0, speed_m_s/self.speed)
        return scale * max(-self.max_angular, min(self.max_angular, self.kp*error))

    def _opening_advance_speed(self, plan: NearFieldGraspPlan, distance_m: float) -> float:
        # Enable alongside cruise acceleration only. Do not consume frozen K0
        # distance twice: _distance_start remains the original commit odometry.
        if self.cruise_speed_scale <= 1.0:
            return 0.0
        mechanics = GripperKinematics()
        front = mechanics.pivot_x_mm + math.hypot(mechanics.tip_offset_x_mm, mechanics.tip_offset_y_mm)
        # A closed-to-open arm stays laterally inside its final opening for
        # relative angles <= 90 degrees. Reject other calibrated sweeps.
        left = self.closed_angles[0] - plan.opening_servo_angles_deg[0]
        right = plan.opening_servo_angles_deg[1] - self.closed_angles[1]
        if not (0 <= left <= 90 and 0 <= right <= 90):
            return 0.0
        # Existing checked corridor must also contain the entire moving arm.
        if len(plan.regions) != 1 or abs(plan.alignment_angle_rad) > 1e-6:
            return 0.0
        region = plan.regions[0]
        if min(p.y for p in region) > -mechanics.pivot_half_spacing_mm or max(p.y for p in region) < mechanics.pivot_half_spacing_mm:
            return 0.0
        available_mm = min(min(p.x for p in plan.bounds) - front,
                           max(p.x for p in region) - front,
                           plan.forward_distance_mm) - 20.0
        remaining = (available_mm - self.progress_mm(distance_m)) / 1000.0
        if remaining <= 0:
            return 0.0
        # Slow to a halt before contact even if servo travel takes its full time.
        return min(self.speed * self.cruise_speed_scale, remaining,
                   math.sqrt(2 * self.deceleration_m_s2 * max(0.0, remaining - 0.02)))

    def _angular(self, plan: NearFieldGraspPlan) -> float:
        return max(-self.max_angular, min(self.max_angular, self.kp * plan.alignment_angle_rad))
