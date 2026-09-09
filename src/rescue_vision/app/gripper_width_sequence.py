"""多目标近场收拢的纯逻辑准备器和运动状态机。"""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
import math
from threading import Condition, Thread
import time

from rescue_vision.app.near_field_grasp import (
    GraspSelection, GraspTarget, GraspTargetTracker, NearFieldGraspPlan,
    NearFieldGraspPolicy, NearFieldGraspSelector, NearFieldHandoffPrior,
)
from rescue_vision.motion.protocol import OdometryImu, SensorFlags
from rescue_vision.perception import PerceptionSnapshot
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


class GripperWidthPickupState(str, Enum):
    SEARCH = "search"
    ALIGNING = "aligning"
    VERIFYING = "verifying"
    OPENING = "opening"
    FORWARD = "forward"
    CLOSING = "closing"
    COMPLETE = "complete"
    ABORTED = "aborted"


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

    def __post_init__(self) -> None:
        if isinstance(self.capture_timestamp_ns, bool) or not isinstance(self.capture_timestamp_ns, int) or self.capture_timestamp_ns < 0:
            raise ValueError("capture_timestamp_ns must be a non-negative integer.")
        if not isinstance(self.selection, GraspSelection):
            raise TypeError("selection must be a GraspSelection.")
        if not isinstance(self.targets, tuple) or not all(isinstance(target, GraspTarget) for target in self.targets):
            raise ValueError("targets must contain GraspTarget values.")
        if not isinstance(self.ready, bool):
            raise ValueError("ready must be a boolean.")
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
        self._alignment_latched = False
        self._last_frame_sequence: int | None = None
        self._last_preparation: GraspPreparation | None = None

    def reset(self) -> None:
        """清空确认窗口并重置近场 tracker。"""

        self.tracker.reset()
        self._locked_ids = None
        self._locked_member_classes = None
        self._last_selection = None
        self._confirmation_count = 0
        self._confirmation_last_frame_sequence = None
        self._confirmation_plan = None
        self._alignment_latched = False
        self._last_frame_sequence = None
        self._last_preparation = None

    def _clear_confirmation(self) -> None:
        self._confirmation_count = 0
        self._confirmation_last_frame_sequence = None
        self._confirmation_plan = None
        self._alignment_latched = False

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
                if observation.target_class in {
                    TargetClass.BLUE_DANGER,
                    TargetClass.UNKNOWN,
                } or observation.model_target_class is TargetClass.BLUE_DANGER:
                    return True
                if ObservationQuality.POSE_COLOR_CONFLICT in observation.quality:
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
               handoff_prior: NearFieldHandoffPrior | None = None) -> GraspPreparation:
        if not isinstance(snapshot, PerceptionSnapshot):
            raise TypeError("snapshot must be a PerceptionSnapshot.")
        if locked_ids is not None:
            locked_ids = tuple(locked_ids)
        if policy is None:
            policy = self.selector.default_policy
        if not isinstance(policy, NearFieldGraspPolicy):
            raise TypeError("policy must be a NearFieldGraspPolicy or None.")
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
        alignment_tolerance_mm = self.selector.config.center_tolerance_mm + (
            self.selector.config.alignment_hysteresis_mm
            if self._alignment_latched
            else 0.0
        )
        selection = self.selector.select(
            targets,
            locked_ids=locked_ids,
            policy=policy,
            alignment_tolerance_mm=alignment_tolerance_mm,
            require_handoff_match=(
                handoff_prior is not None
                and locked_ids is None
                and policy.allowed_classes
                == frozenset((TargetClass.GREEN_SUPPLY,))
                and policy.max_targets == 1
            ),
        )
        self._last_selection = selection
        plan = selection.plan
        if (
            locked_ids is not None
            and self._locked_member_classes is None
            and plan is not None
            and plan.member_ids == locked_ids
        ):
            self._locked_member_classes = tuple(
                member.observation.target_class for member in plan.members
            )
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
            if plan.alignment_angle_rad != 0.0:
                self._clear_confirmation()
            else:
                self._alignment_latched = True
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
        self._last_frame_sequence = snapshot.frame_sequence
        self._last_preparation = result
        return result


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
        ] | None = None
        self._latest: GraspPreparation | None = None
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
    ) -> None:
        if not isinstance(snapshot, PerceptionSnapshot):
            raise TypeError("snapshot must be a PerceptionSnapshot.")
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
            )
            self._condition.notify_all()

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
                assert request is not None
                (
                    snapshot,
                    locked_ids,
                    policy,
                    session_id,
                    handoff_prior,
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
                    )
                except Exception as exc:
                    with self._condition:
                        # 已完成的唯一确认窗口不能因旁路一次异常被抹掉；
                        # 动作层仍会检查其中计划的真实年龄。
                        self.error = f"planning:{type(exc).__name__}:{exc}"
                    continue
                prepared = replace(
                    prepared,
                    prepared_timestamp_ns=time.monotonic_ns(),
                )
                with self._condition:
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
                 grasp_commit_max_observation_age_ms: float = 150.0,
                 stationary_max_gyro_rad_s: float = 0.03,
                 fine_alignment_zone_rad: float = 0.08,
                 fine_alignment_min_wheel_velocity_m_s: float = 0.0):
        self.travel_ns = round(_positive(gripper_full_travel_time_s, "gripper_full_travel_time_s") * 1e9)
        self.speed = _positive(forward_speed_m_s, "forward_speed_m_s")
        self.age_ns = round(_positive(max_observation_age_ms, "max_observation_age_ms") * 1e6)
        self.commit_age_ns = round(
            _positive(
                grasp_commit_max_observation_age_ms,
                "grasp_commit_max_observation_age_ms",
            )
            * 1e6
        )
        self.stationary_max_gyro_rad_s = _positive(stationary_max_gyro_rad_s, "stationary_max_gyro_rad_s")
        self._motion: OdometryImu | None = None
        self._stationary_since_ns: int | None = None
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
        self._distance_start: float | None = None
        self._last_progress_mm = 0.0
        self._progress_ns = 0
        self._abort_capture_ns = -1
        self._abort_reason = ""
        self._alignment_plan: NearFieldGraspPlan | None = None
        self._alignment_direction: int | None = None
        self._alignment_started_ns: int | None = None

    def observe_motion(self, message: OdometryImu) -> None:
        """消费真实编码器/IMU样本；接收时间与相机同为主机单调时钟。

        计数变化、旋转、无效传感器、样本倒退或遥测间断均使静止证据失效。
        不把零速指令、重复消息或重新发布的视觉结果当作静止证据。
        """
        if not isinstance(message, OdometryImu):
            raise TypeError("message must be OdometryImu.")
        previous = self._motion
        self._motion = message
        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID
        valid = (
            message.sensor_flags & required == required
            and not message.sensor_flags & (SensorFlags.SAMPLE_OVERRUN | SensorFlags.GYRO_SATURATED)
            and abs(message.gyro_z_rad_s) <= self.stationary_max_gyro_rad_s
        )
        continuous = (
            previous is not None
            and previous.sensor_flags & required == required
            and not previous.sensor_flags & (SensorFlags.SAMPLE_OVERRUN | SensorFlags.GYRO_SATURATED)
            and abs(previous.gyro_z_rad_s) <= self.stationary_max_gyro_rad_s
            and 0 <= message.received_timestamp_ns - previous.received_timestamp_ns <= self.commit_age_ns
            and 0 < (message.sample_timestamp_us - previous.sample_timestamp_us) * 1000 <= self.commit_age_ns
            and message.left_encoder_count == previous.left_encoder_count
            and message.right_encoder_count == previous.right_encoder_count
        )
        if not valid or not continuous:
            self._stationary_since_ns = None
        elif self._stationary_since_ns is None:
            # 使用第二个样本的接收时刻，避免把尚未证实的区间算成静止。
            self._stationary_since_ns = message.received_timestamp_ns

    def motion_diagnostic(self, timestamp_ns: int) -> str:
        """记录静止证据来源，区分视觉延迟和底盘实际仍在移动。"""
        since = "none" if self._stationary_since_ns is None else f"{self._stationary_since_ns / 1e6:.1f}"
        motion = self._motion
        age = "none" if motion is None else f"{(timestamp_ns - motion.received_timestamp_ns) / 1e6:.1f}"
        gyro = "none" if motion is None else f"{motion.gyro_z_rad_s:.4f}"
        return f"stationary_since_ms={since} motion_age_ms={age} gyro_z_rad_s={gyro}"

    def _stationary_plan_valid(self, now_ns: int, prep: GraspPreparation) -> bool:
        plan = prep.selection.plan
        motion = self._motion
        return (
            plan is not None
            and motion is not None
            and self._stationary_since_ns is not None
            and self._stationary_since_ns <= plan.capture_timestamp_ns <= motion.received_timestamp_ns
            and 0 <= now_ns - motion.received_timestamp_ns <= self.commit_age_ns
            and 0 <= now_ns - plan.capture_timestamp_ns <= self.age_ns
            # 缺失后缓存的确认计划不能借新快照/新发布时间续命。
            and plan.capture_timestamp_ns == prep.capture_timestamp_ns
            and all(member.observed for member in plan.members)
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
        self._abort_reason = ""
        self._alignment_plan = None
        self._alignment_direction = None
        self._alignment_started_ns = None

    def progress_mm(self, cumulative_distance_m: float | None) -> float:
        return 0.0 if cumulative_distance_m is None or self._distance_start is None else max(0.0, (cumulative_distance_m - self._distance_start) * 1000)

    def _abort(self, now_ns: int, reason: str, prep: GraspPreparation | None) -> GripperWidthPickupDecision:
        self.state = GripperWidthPickupState.ABORTED
        self._abort_capture_ns = now_ns  # 必须是动作退出之后采集的新证据
        self._abort_reason = reason
        self.active_plan = None
        self.locked_ids = None
        self._alignment_plan = None
        self._alignment_direction = None
        self._alignment_started_ns = None
        return self._decision(now_ns, reason, brake=True)

    def _replan(self, now_ns: int, reason: str) -> GripperWidthPickupDecision:
        """开爪前证据变化只解锁重选，并清空旧候选的对准计时。"""

        self.state = GripperWidthPickupState.SEARCH
        self.active_plan = None
        self.locked_ids = None
        self._alignment_plan = None
        self._alignment_direction = None
        self._alignment_started_ns = None
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
            >= self.alignment_timeout_ns
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
        if (
            self.state is GripperWidthPickupState.ALIGNING
            and self._alignment_plan is not None
            and abs(self._alignment_plan.alignment_angle_rad)
            <= self.fine_alignment_zone_rad
        ):
            minimum = self.fine_alignment_min_wheel_velocity_m_s
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
                else None
            ),
        )

    def step(
        self,
        timestamp_ns: int,
        preparation: GraspPreparation | None,
        *,
        cumulative_distance_m: float | None,
        path_clear: bool = True,
    ) -> GripperWidthPickupDecision:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0 or timestamp_ns < self._last_ns:
            raise ValueError(f"Invalid/nonmonotonic timestamp_ns {timestamp_ns!r}.")
        self._last_ns = timestamp_ns
        if cumulative_distance_m is not None and (isinstance(cumulative_distance_m, bool) or not math.isfinite(cumulative_distance_m)):
            raise ValueError(f"Invalid cumulative_distance_m {cumulative_distance_m!r}.")
        if not isinstance(path_clear, bool):
            raise ValueError("path_clear must be a boolean.")
        if preparation is not None and not isinstance(preparation, GraspPreparation):
            raise TypeError("preparation must be a GraspPreparation or None.")
        now, prep = timestamp_ns, preparation
        if self.state is GripperWidthPickupState.COMPLETE:
            return self._decision(now, "complete_capture_unconfirmed", brake=True)
        current_prep = (
            prep
            if prep is not None and 0 <= now - prep.capture_timestamp_ns <= self.age_ns
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
            self._alignment_direction = None
            self._alignment_started_ns = None
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
            if (
                self.locked_ids is None
                and current_prep is not None
                and current_prep.checked_member_ids is not None
            ):
                # 解锁后的后台迟到结果仍属于旧锁组，不能作为新的无锁方案接管。
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
                    age = now - self._alignment_plan.capture_timestamp_ns
                    if 0 <= age <= self.age_ns:
                        return self._decision(now, "alignment_recent_checked_plan", angular=self._angular(self._alignment_plan) * 0.5)
                if self.locked_ids is not None:
                    return self._decision(
                        now,
                        "waiting_locked_target_observation",
                        brake=True,
                    )
                return self._decision(now, "waiting_eligible_group", brake=True)
            same_locked_group = self.locked_ids == plan.member_ids
            self.locked_ids = plan.member_ids
            if plan.alignment_angle_rad != 0:
                direction = 1 if plan.alignment_angle_rad > 0 else -1
                if same_locked_group and self._alignment_direction is not None:
                    if self.state is GripperWidthPickupState.VERIFYING:
                        # VERIFYING 只是对准稳定性确认；同一组重新超出
                        # 对准允许范围时必须恢复旋转，不能锁死在等待态。
                        # 先停车再换向；此处不重置整次近场确认/对准超时。
                        self.state = GripperWidthPickupState.ALIGNING
                        self._alignment_direction = direction
                        self._alignment_plan = plan
                        return self._decision(
                            now,
                            "alignment_verify_lost_realign",
                            angular=self._angular(plan),
                        )
                    if direction != self._alignment_direction:
                        self.state = GripperWidthPickupState.VERIFYING
                        self._alignment_plan = plan
                        return self._decision(
                            now,
                            "alignment_crossed_zero_verify",
                            brake=True,
                        )
                self._alignment_direction = direction
                self.state = GripperWidthPickupState.ALIGNING
                self._alignment_plan = plan
                if self._alignment_started_ns is None:
                    self._alignment_started_ns = now
                return self._decision(now, "align_group_envelope", angular=self._angular(plan))
            self.state = GripperWidthPickupState.VERIFYING
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
            if self._motion is not None and not self._stationary_plan_valid(now, current_prep):
                return self._decision(now, "confirmation_waiting_for_stationary_capture", brake=True)
            if preparation_age_ns is None or preparation_age_ns > self.commit_age_ns:
                return self._decision(
                    now,
                    "confirmation_waiting_for_fresh_preparation",
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
            self._alignment_direction = None
            self._alignment_started_ns = None
            self._distance_start = cumulative_distance_m
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
                return self._decision(now, "opening_gripper", brake=True)

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
            return self._decision(now, "forward_open_loop", speed=min(self.speed, max(0.005, remaining_m)))
        return self._abort(now, "invalid_pickup_state", current_prep)

    def _angular(self, plan: NearFieldGraspPlan) -> float:
        return max(-self.max_angular, min(self.max_angular, self.kp * plan.alignment_angle_rad))
