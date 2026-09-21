"""正式比赛流程：基于当前正式动作编排，纯逻辑不打开硬件。"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rescue_vision.app.cluster_breakup import GripperPosture
from rescue_vision.app.gate_clearance import (
    GateClearanceSession,
    maybe_begin_clearance,
    reacquire_path_clear,
    reacquire_plan_matches_cargo,
    step_clearance,
)
from rescue_vision.app.breakup_planner import (
    BreakupPlan, BreakupSceneContext, BreakupTarget, physical_radii, plan_breakup,
    safe_zone_intersection, same_local_group, segment_clear, robot_clearance_mm,
)
from rescue_vision.perception.target_ground_geometry import TargetGroundGeometryConfig
from rescue_vision.perception.gripper_color import (
    GripperColorConfig,
    bounding_box_overlaps_gripper_polygon,
)

from rescue_vision.app.gripper_width_sequence import (
    GraspPreparation,
    GraspSceneAction,
    GripperWidthPickupDecision,
    GripperWidthPickupResult,
    GripperWidthPickupSequence,
    GripperWidthPickupState,
)
from rescue_vision.app.near_field_grasp import (
    GraspTarget,
    NearFieldGraspPlan,
    NearFieldHandoffPrior,
    NearFieldGraspPolicy,
    polygon_distance,
)
from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import (
    FieldPose2D,
    SafeZoneCornerLocalizer,
    SafeZoneCornerPoseObservation,
    normalize_angle,
)
from rescue_vision.perception import (
    ObservationQuality,
    FieldFeatureDetectionResult,
    PerceptionSnapshot,
    SafeZoneColor,
    SafeZoneObservation,
    TargetClass,
    TargetObservation,
    UndistortedBoundingBox,
)
from rescue_vision.tracking import MultiTargetTracker, TrackStatus, TrackedTarget
from rescue_vision.mission import SafetySignals
from rescue_vision.motion.approach_speed import approach_speed_m_s
from rescue_vision.motion.controller import (
    MotionAccelerationOverrides,
    WheelAccelerationOverrides,
)
from rescue_vision.motion.gripper_kinematics import GripperKinematics
from rescue_vision.motion.protocol import OdometryImu
from rescue_vision.motion.stationary import StationaryMotionEvidence
from rescue_vision.motion.relative_action import (
    RelativeActionController, RelativeActionProfile, RelativeActionFeedback,
    RelativeActionCommand, RelativeActionKind, RelativeActionPhase,
)
from rescue_vision.motion.point_action import PointActionController
from rescue_vision.world.static_map import PhysicalRegionKind, StaticFieldMap, TeamColor

if TYPE_CHECKING:
    from rescue_vision.config import AppConfig, MatchRuntimeConfig
class MatchState(str, Enum):
    """正式动作策略的状态。"""

    BOOT = "boot"
    PREFLIGHT = "preflight"
    STARTUP_TURN_RIGHT = "startup_turn_right"
    STARTUP_TURN_SETTLE = "startup_turn_settle"
    STARTUP_FORWARD = "startup_forward"
    STARTUP_FORWARD_SETTLE = "startup_forward_settle"
    SEARCH_CLUSTER = "search_cluster"
    ALIGN_CLUSTER_ONCE = "align_cluster_once"
    APPROACH_CLUSTER = "approach_cluster"
    BREAKUP_SETTLE = "breakup_settle"
    RELOCATE_FORWARD = "relocate_forward"
    BREAKUP_FORWARD = "breakup_forward"
    BREAKUP_BACKWARD = "breakup_backward"
    OPEN_GRIPPER_SETTLE = "open_gripper_settle"
    CLOSE_GRIPPER_SETTLE = "close_gripper_settle"
    CLOSE_GRIPPER_SPIN = "close_gripper_spin"
    CHECK_ISOLATED_GREEN = "check_isolated_green"
    TRANSPORT_ALIGN_GREEN = "transport_align_green"
    TRANSPORT_APPROACH_GREEN = "transport_approach_green"
    TRANSPORT_NEAR_FIELD_GRASP = "transport_near_field_grasp"
    TRANSPORT_GREEDY_SCAN = "transport_greedy_scan"
    TRANSPORT_PRE_CLOSE_RECHECK = "transport_pre_close_recheck"
    TRANSPORT_CLOSE_GRIPPER = "transport_close_gripper"
    TRANSPORT_ALIGN_RED_ZONE = "transport_align_red_zone"
    TRANSPORT_FORWARD = "transport_forward"
    TRANSPORT_RELEASE = "transport_release"
    GATE_CLEARANCE = "gate_clearance"
    MISGRASP_OPEN = "misgrasp_open"
    MISGRASP_BACKUP = "misgrasp_backup"
    MISGRASP_SETTLE = "misgrasp_settle"
    RETURN_BACKUP = "return_backup"
    FINISH_STOP = "finish_stop"
    TERMINAL_STOP = "terminal_stop"
    # 独立 CC 流程状态；共享枚举仅用于复用同一硬件循环和安全区运输链。
    CC_ALIGN_CLUSTER = "cc_align_cluster"
    CC_BREAKUP_FORWARD = "cc_breakup_forward"
    CC_BREAKUP_OPEN_GAP = "cc_breakup_open_gap"
    CC_BREAKUP_BACKWARD = "cc_breakup_backward"
    CC_BREAKUP_CLOSE_GAP = "cc_breakup_close_gap"
    CC_SCAN_GREEN = "cc_scan_green"
    CC_SCAN_ORANGE = "cc_scan_orange"
    CC_SCAN_SUPPLY = "cc_scan_supply"
    CC_SCAN_MIXED_CLUSTER = "cc_scan_mixed_cluster"
    CC_ALIGN_TARGET = "cc_align_target"
    CC_APPROACH_TARGET = "cc_approach_target"
    CC_GREEN_CLOSE_GAP = "cc_green_close_gap"
    CC_ORANGE_OPEN_GAP = "cc_orange_open_gap"
    CC_ORANGE_FORWARD = "cc_orange_forward"
    CC_ORANGE_CLOSE_GAP = "cc_orange_close_gap"
    # 无解团（no-breakup）变体的相对动作序列状态；由 match_nb 子类驱动。
    NB_OPENING_SEQUENCE = "nb_opening_sequence"
    NB_OPENING_GRIPPER_OPEN = "nb_opening_gripper_open"


class MatchStartArea(str, Enum):
    """正式流程支持的物理启动区域。"""

    AREA_2 = "2"
    AREA_3 = "3"

    @classmethod
    def parse(cls, value: object) -> MatchStartArea:
        """把 CLI/调用方输入解析为受限的启动区域枚举。"""

        if isinstance(value, cls):
            return value
        # ``python -m rescue_vision.app.match`` can load this module once as
        # ``rescue_vision.app.match`` through the package and once as
        # ``__main__``.  Accept an equivalent Enum from the other module copy.
        if isinstance(value, Enum):
            value = value.value
        if isinstance(value, bool):
            raise ValueError(f"start_area must be '2' or '3', got {value!r}.")
        try:
            return cls(str(value))
        except ValueError as exc:
            raise ValueError(
                f"start_area must be '2' or '3', got {value!r}."
            ) from exc


class GraspRoute(str, Enum):
    """近场停车后的一次性抓取路由。"""

    DECIDING = "deciding"
    DIRECT_NEAR = "direct_near"
    FAR_REAPPROACH = "far_reapproach"
    RESELECT = "reselect"
    BREAKUP = "breakup"


@dataclass(frozen=True, slots=True)
class GraspRouteDecision:
    """停车窗口内的确定性抓取路由结果；仅供 MatchSequence 内部使用。"""

    route: GraspRoute
    plan: NearFieldGraspPlan | None = None
    target: TrackedTarget | None = None
    candidate_count: int = 0
    rejection_reasons: tuple[str, ...] = ()
    elapsed_ms: float = 0.0


@dataclass(frozen=True, slots=True)
class MatchPreflight:
    """启动前的最小安全证据。"""

    telemetry_fresh: bool
    watchdog_armed: bool
    emergency_stop_clear: bool
    zero_speed_command_accepted: bool
    camera_observation_fresh: bool
    ground_mapping_enabled: bool

    def __post_init__(self) -> None:
        for name in (
            "telemetry_fresh",
            "watchdog_armed",
            "emergency_stop_clear",
            "zero_speed_command_accepted",
            "camera_observation_fresh",
            "ground_mapping_enabled",
        ):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")

    @property
    def ready(self) -> bool:
        return all(
            (
                self.telemetry_fresh,
                self.watchdog_armed,
                self.emergency_stop_clear,
                self.zero_speed_command_accepted,
                self.camera_observation_fresh,
                self.ground_mapping_enabled,
            )
        )


@dataclass(frozen=True, slots=True)
class MatchDecision:
    """单个控制周期的差速 twist、可选轮速意图与夹爪姿态。"""

    timestamp_ns: int
    state: MatchState
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    gripper_posture: GripperPosture
    reason: str
    selected_track_id: int | None = None
    gripper_angles_deg: tuple[float, float] | None = None
    soft_brake: bool = False
    min_wheel_velocity_m_s: float | None = None
    wheel_speeds_m_s: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if not isinstance(self.state, MatchState):
            raise ValueError("state must be a MatchState.")
        for name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not isinstance(self.gripper_posture, GripperPosture):
            raise ValueError("gripper_posture must be a GripperPosture.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string.")
        if self.selected_track_id is not None and (
            isinstance(self.selected_track_id, bool)
            or not isinstance(self.selected_track_id, int)
            or self.selected_track_id <= 0
        ):
            raise ValueError("selected_track_id must be a positive integer or None.")
        if self.gripper_angles_deg is not None and (
            not isinstance(self.gripper_angles_deg, tuple)
            or len(self.gripper_angles_deg) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 180.0
                for value in self.gripper_angles_deg
            )
        ):
            raise ValueError(
                "gripper_angles_deg must contain two angles in [0, 180] or None."
            )
        if not isinstance(self.soft_brake, bool):
            raise ValueError("soft_brake must be a boolean.")
        if self.min_wheel_velocity_m_s is not None and (
            isinstance(self.min_wheel_velocity_m_s, bool)
            or not isinstance(self.min_wheel_velocity_m_s, (int, float))
            or not math.isfinite(float(self.min_wheel_velocity_m_s))
            or float(self.min_wheel_velocity_m_s) < 0.0
        ):
            raise ValueError(
                "min_wheel_velocity_m_s must be finite and non-negative or None."
            )
        if self.wheel_speeds_m_s is not None and (
            not isinstance(self.wheel_speeds_m_s, tuple)
            or len(self.wheel_speeds_m_s) != 2
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                for value in self.wheel_speeds_m_s
            )
        ):
            raise ValueError("wheel_speeds_m_s must contain two finite speeds or None.")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _print_state_banner(state: MatchState, reason: str) -> None:
    """用醒目的单行横幅记录状态，便于终端和日志快速定位。"""

    print(
        f"\n==================== state={state.value} "
        f"reason={reason} ====================",
        flush=True,
    )


def _central_symmetric_field_point(point: FieldPoint) -> FieldPoint:
    """把场地坐标绕中心十字原点做 180° 中心对称。"""

    return FieldPoint(-point.x, -point.y)


def configure_match_start_area(
    config: AppConfig,
    start_area: MatchStartArea | str | int,
) -> AppConfig:
    """返回指定启动区域的正式流程配置副本。

    运行配置以区域 2 为基准。区域 3 使用中心十字为原点的 180° 对称：
    初始场地位姿、己方运输终点和带符号的场地刹车过冲坐标全部变换，
    并把己方颜色切换为蓝色。``world.static_map`` 不变，它是固定物理
    地图，已经同时包含红、蓝安全区及其地标。
    """

    from rescue_vision.config import AppConfig

    if not isinstance(config, AppConfig):
        raise TypeError("config must be an AppConfig.")
    area = MatchStartArea.parse(start_area)
    if area is MatchStartArea.AREA_2:
        return config

    initial = config.localization.fusion.initial_pose
    mirrored_initial = FieldPose2D(
        _central_symmetric_field_point(initial.position),
        normalize_angle(initial.heading_rad + math.pi),
    )
    match = replace(
        config.match,
        safe_zone_fallback_target_field=_central_symmetric_field_point(
            config.match.safe_zone_fallback_target_field
        ),
        safe_zone_injured_target_field=_central_symmetric_field_point(
            config.match.safe_zone_injured_target_field
        ),
    )
    localization = replace(
        config.localization,
        fusion=replace(config.localization.fusion, initial_pose=mirrored_initial),
    )
    world = replace(config.world, team_color=TeamColor.BLUE)
    return replace(config, match=match, localization=localization, world=world)


@dataclass(frozen=True, slots=True)
class _GroundClusterMeasurement:
    """当前目标团的中心和最前方有效地面锚点距离，单位 mm。"""

    center: GroundPoint
    nearest_forward_x_mm: float


@dataclass(frozen=True, slots=True)
class _PoseHistoryEntry:
    """场地航位在一个控制时刻的快照，用于对齐延迟观测。"""

    timestamp_ns: int
    cumulative_distance_m: float
    heading_rad: float
    position: FieldPoint


@dataclass(frozen=True, slots=True)
class _AttemptFailureRecord:
    """按物理目标/区域记忆一次失败，而不是按 tracker ID 记忆。"""

    target_class: TargetClass | None
    relative_point: GroundPoint | None
    field_point: FieldPoint | None
    robot_position: FieldPoint | None
    heading_rad: float | None
    track_id: int | None
    reason: str
    timestamp_ns: int
    region_field_points: tuple[FieldPoint, ...] = ()


@dataclass(slots=True)
class BreakupMarkedTarget:
    """Last measured position of a scoring block; never a predicted grasp point."""

    target_class: TargetClass
    field_point: FieldPoint
    capture_timestamp_ns: int
    capture_distance_m: float
    track_id: int | None = None


@dataclass(slots=True)
class GraspTask:
    """Physical grasp intent; sessions are scene revisions, not new attempts."""

    task_id: int
    started_ns: int
    deadline_ns: int
    entry_track_id: int | None
    entry_class: TargetClass | None
    entry_field: FieldPoint | None
    entry_ground: GroundPoint | None
    core: tuple[GraspTarget, ...] = ()
    recovery_count: int = 0
    scene_revision: int = 0
    scene_floor_ns: int = -1
    failure_recorded_revision: int = -1
    last_progress_ns: int = 0
    blocking: tuple[str, ...] = ()
    recovery_aim: FieldPoint | None = None
    marked_targets: tuple[BreakupMarkedTarget, ...] = ()
    recovery_reobserve_started_ns: int | None = None
    recovery_reobserve_deadline_ns: int | None = None


class MatchSequence:
    """可重放的正式动作流程；step() 只消费观测、里程和航向并输出意图。"""

    INITIAL_HEADING_RAD = -math.pi / 2.0

    # 独立的抓取—运输联调入口覆盖为只搜索，不执行正式解团路由。
    _first_green_blocked_routes_to_breakup = True
    _dynamic_breakup_enabled = True
    # 没有发出过任何运动指令的退出原因：允许原地重观测一次，不写失败记忆。
    # ``confirmation_timeout`` 不在此列：它既表示"完全没有方案"的路由超时，
    # 也表示提交窗口用尽，两者都应当直接退出，重观测只会拉长零速等待。
    _NEAR_FIELD_REOBSERVE_REASONS = frozenset({"scene_evidence_unavailable"})
    _NEAR_FIELD_REOBSERVE_LIMIT = 1

    def __init__(
        self,
        config: MatchRuntimeConfig,
        *,
        tracker: MultiTargetTracker,
        gripper_full_travel_time_s: float,
        team_color: TeamColor,
        initial_field_position: FieldPoint | None = None,
        safe_zone_corner_localizer: SafeZoneCornerLocalizer | None = None,
        static_map: StaticFieldMap | None = None,
        breakup_clearance_mm: float = 240.0,
        near_field_pickup: GripperWidthPickupSequence | None = None,
        near_field_grasp_config: NearFieldGraspConfig | None = None,
        transport_corridor_half_width_mm: float | None = None,
        breakup_target_geometry: TargetGroundGeometryConfig | None = None,
        gripper_color_config: GripperColorConfig | None = None,
    ) -> None:
        from rescue_vision.config import MatchRuntimeConfig

        if not isinstance(config, MatchRuntimeConfig):
            raise TypeError("config must be a MatchRuntimeConfig.")
        if not config.enabled:
            raise ValueError("MatchSequence requires enabled config.")
        if not isinstance(tracker, MultiTargetTracker):
            raise TypeError("tracker must be a MultiTargetTracker.")
        if near_field_pickup is not None and not isinstance(
            near_field_pickup, GripperWidthPickupSequence
        ):
            raise TypeError(
                "near_field_pickup must be a GripperWidthPickupSequence or None."
            )
        if near_field_grasp_config is not None and not isinstance(
            near_field_grasp_config, NearFieldGraspConfig
        ):
            raise TypeError(
                "near_field_grasp_config must be a NearFieldGraspConfig or None."
            )
        if transport_corridor_half_width_mm is not None:
            if (
                isinstance(transport_corridor_half_width_mm, bool)
                or not isinstance(transport_corridor_half_width_mm, (int, float))
                or not math.isfinite(float(transport_corridor_half_width_mm))
                or float(transport_corridor_half_width_mm) <= 0.0
            ):
                raise ValueError(
                    "transport_corridor_half_width_mm must be finite and positive."
                )
        if not isinstance(team_color, TeamColor):
            raise TypeError("team_color must be a TeamColor.")
        if gripper_color_config is not None and not isinstance(
            gripper_color_config,
            GripperColorConfig,
        ):
            raise TypeError(
                "gripper_color_config must be a GripperColorConfig or None."
            )
        for name, point in (
            ("initial_field_position", initial_field_position),
        ):
            if point is not None and not isinstance(point, FieldPoint):
                raise TypeError(f"{name} must be a FieldPoint or None.")
        if (
            isinstance(gripper_full_travel_time_s, bool)
            or not isinstance(gripper_full_travel_time_s, (int, float))
            or not math.isfinite(float(gripper_full_travel_time_s))
            or float(gripper_full_travel_time_s) <= 0.0
        ):
            raise ValueError(
                "gripper_full_travel_time_s must be finite and positive."
            )
        self._breakup_grasp_preparation: GraspPreparation | None = None
        self._breakup_target_geometry = breakup_target_geometry
        self._breakup_plan: BreakupPlan | None = None
        self._breakup_proposal: BreakupPlan | None = None
        # 近场 RECOVERY 已经算出接触计划的解团：对准转向只修正接触射线，
        # 不使计划失效。该字段与 ``_breakup_proposal`` 分开，因为后者既表示
        # "待对准的候选"也表示"待确认的观察目标"。
        self._breakup_frozen_plan: BreakupPlan | None = None
        self._breakup_attempts: list[BreakupPlan] = []
        self._breakup_rejected_grasp_ids: set[int] = set()
        self._breakup_phase_started_ns: int | None = None
        self._breakup_stopped_ns: int | None = None
        self._breakup_reference_frames: set[tuple[int, int]] = set()
        self._breakup_last_capture_ns = -1
        self._breakup_actual_forward_mm = 0.0
        self._breakup_actual_backward_mm = 0.0
        self._breakup_retreat_mm = 0.0
        evidence_config = near_field_grasp_config or NearFieldGraspConfig()
        # 夹爪实际开口之外的走廊余量：包络外侧的一半 ``clearance_mm`` 加
        # 近场走廊横向安全余量。远场走廊门禁与近场扫掠使用同一组数值。
        self._path_corridor_margin_mm = (
            evidence_config.clearance_mm / 2.0
            + evidence_config.corridor_lateral_margin_mm
        )
        self._stationary_motion = (near_field_pickup.motion_evidence if near_field_pickup is not None
                                   else StationaryMotionEvidence(
                                       max_gap_ns=round(evidence_config.grasp_commit_max_observation_age_ms*1e6),
                                       max_gyro_rad_s=evidence_config.stationary_max_gyro_rad_s))
        self._breakup_feedback_since_ns: int | None = None
        self._breakup_feedback_pose: tuple[float | None, float | None] | None = None
        self._breakup_anchor: FieldPoint | None = None
        self._breakup_reference_current = False
        self._breakup_observation_deadline_ns: int | None = None
        self._breakup_no_plan_since_ns: int | None = None
        self._breakup_failed_aims: list[FieldPoint] = []
        # Keep the robot pose alongside physical breakup failures.  A turn or
        # a new session does not change this memory; a verified translation can
        # make a later retry a genuinely different approach.
        self._breakup_failed_aim_positions: list[FieldPoint | None] = []
        self._breakup_attempt_positions: list[FieldPoint | None] = []
        self._rotation_budget_commit_diagnostic: str | None = None
        self._rotation_budget_checked_frame: tuple[int, int] | None = None
        self._breakup_search_frame: tuple[int, int] | None = None
        self._breakup_wait_detail = "not_observing"
        self._breakup_last_failure_diagnostic: str | None = None
        # 最近一次解团选组的逐候选拒绝原因，用于解释"选不出接触计划"。
        self._breakup_plan_rejections: tuple[str, ...] = ()
        # 当前确认集合首帧对应的机器人位姿；位姿变化才作废既有确认。
        self._breakup_confirm_pose: tuple[float | None, float | None] | None = None
        self._gate_clearance: GateClearanceSession | None = None
        self._gate_clearance_attempted = False
        self.config = config
        self._motion_profile = RelativeActionProfile()
        self._motion_wheel_track_m = 0.2
        self._motion_max_wheel_speed_m_s = 1.5
        self._motion_max_angular_rad_s = 1.0
        self._motion_wheel_weights = (1.0, 1.0)
        self._noncontact_key: str | None = None
        self._noncontact_controller: RelativeActionController | None = None
        self._noncontact_started_ns: int | None = None
        self._noncontact_budget_s = self._motion_profile.action_timeout_s
        self._noncontact_base_distance = 0.0
        self._noncontact_last_heading = 0.0
        self._noncontact_turn_progress = 0.0
        self._noncontact_direction = 1.0
        self._noncontact_last_command: RelativeActionCommand | None = None
        self._noncontact_stop_confirmed = False
        self._fallback_last_heading_rad: float | None = None
        self._tracker = tracker
        self._team_color = team_color
        # 正式运输路线沿己方安全区所在的场地 y 方向前进；区域 2/红方为
        # +y，区域 3/蓝方为 -y。UNKNOWN 仅保留旧纯逻辑调用的 +y 行为，
        # 正式配置会在启动区域装配时确定颜色。
        self._safe_zone_forward_y_sign = (
            -1.0 if team_color is TeamColor.BLUE else 1.0
        )
        self._gripper_full_travel_time_ns = round(
            float(gripper_full_travel_time_s) * 1_000_000_000
        )
        self._near_field_pickup = near_field_pickup
        self._near_field_grasp_config = near_field_grasp_config
        self._gripper_color_config = gripper_color_config or GripperColorConfig()
        self._transport_corridor_half_width_mm = (
            float(transport_corridor_half_width_mm)
            if transport_corridor_half_width_mm is not None
            else float(config.green_path_half_width_mm)
        )
        self._near_field_session_id = 0
        self.grasp_planning_pending = False
        self._perception_interval_ns: int | None = None
        self._near_field_no_plan_budget_ms: float | None = None
        # 同一片区域连续"没有发出过动作"的退出次数；真实动作或离开区域即清零。
        self._near_field_reobserve_retries = 0
        self._grasp_task: GraspTask | None = None
        self._next_grasp_task_id = 1
        self.state = MatchState.BOOT
        self._started = False
        self._last_timestamp_ns: int | None = None
        self._last_tracker_frame_sequence: int | None = None
        self._preview_target_history: dict[
            tuple[int, int], tuple[TrackedTarget, ...]
        ] = {}
        self._startup_turn_last_heading: float | None = None
        self._startup_turn_progress_rad = 0.0
        self._startup_forward_base_distance_m: float | None = None
        self._straight_pid_integral = 0.0
        self._straight_pid_last_error: float | None = None
        self._straight_pid_last_timestamp_ns: int | None = None
        self._cluster_approach_base_distance_m: float | None = None
        self._cluster_approach_travel_distance_m: float | None = None
        self._breakup_forward_base_distance_m: float | None = None
        self._breakup_segment_stop_started_ns: int | None = None
        self._breakup_segment_stop_deadline_ns: int | None = None
        self._breakup_backward_base_distance_m: float | None = None
        self._spin_last_heading: float | None = None
        self._spin_progress_rad = 0.0
        self._settle_until_ns = 0
        self._gripper_phase_started_ns: int | None = None
        self._selected_track_id: int | None = None
        self._cluster_selected_track_ids: tuple[int, ...] = ()
        self._selected_green_ground: GroundPoint | None = None
        self._green_align_lost_since_ns: int | None = None
        self._green_align_hold_ns = round(
            float(config.green_align_hold_ms) * 1_000_000
        )
        self._green_approach_base_distance_m: float | None = None
        self._green_approach_distance_m: float | None = None
        self._transport_forward_base_distance_m: float | None = None
        self._transport_forward_distance_m: float | None = None
        self._cargo_capture_floor_ns: int | None = None
        self._counted_pickup_session: int | None = None
        self._misgrasp_started_ns = 0
        self._misgrasp_base_distance_m: float | None = None
        self._misgrasp_heading_rad: float | None = None
        self._misgrasp_release_pose: FieldPose2D | None = None
        self._misgrasp_breakup_active = False
        self._greedy_active = False
        self._greedy_started_ns = 0
        self._greedy_last_heading: float | None = None
        self._greedy_progress_rad = 0.0
        self._transport_count = 0
        self._transport_target_classes: tuple[TargetClass, ...] = ()
        self._breakup_only = False
        self._near_field_route = GraspRoute.DECIDING
        self._near_field_confirmation_started_ns: int | None = None
        self._near_field_handoff_prior: NearFieldHandoffPrior | None = None
        self._near_field_last_failure_diagnostic: str | None = None
        self._near_field_far_reapproach_used = False
        self._near_field_route_rejections: tuple[str, ...] = ()
        self._near_field_route_elapsed_ms = 0.0
        self._near_field_route_candidate_count = 0
        self._return_backup_base_distance_m: float | None = None
        self._initial_field_position = initial_field_position
        self._fallback_field_position = initial_field_position
        self._fallback_last_distance_m: float | None = None
        self._pose_history: list[_PoseHistoryEntry] = []
        self._green_target_field_point: FieldPoint | None = None
        self._green_alignment_target_heading_rad: float | None = None
        self._green_turn_start_heading_rad: float | None = None
        self._green_turn_direction = 0.0
        self._green_turn_budget_rad = 0.0
        self._green_turn_progress_rad = 0.0
        self._green_turn_attempts = 0
        self._green_turn_active = False
        self._green_alignment_wait_since_ns: int | None = None
        self._near_field_failures: list[_AttemptFailureRecord] = []
        self._safe_zone_scan_last_heading: float | None = None
        self._safe_zone_scan_progress_rad = 0.0
        self._cluster_search_angular_velocity_rad_s = (
            config.cluster_search_angular_velocity_rad_s
        )
        self._cluster_search_last_heading: float | None = None
        self._cluster_search_progress_rad = 0.0
        self._cluster_align_hold_center: GroundPoint | None = None
        self._cluster_align_lost_since_ns: int | None = None
        self._cluster_align_hold_ns = round(
            float(config.cluster_align_hold_ms) * 1_000_000
        )
        self._consecutive_cluster_losses = 0
        self._relocate_forward_base_distance_m: float | None = None
        if safe_zone_corner_localizer is not None and not isinstance(
            safe_zone_corner_localizer,
            SafeZoneCornerLocalizer,
        ):
            raise TypeError(
                "safe_zone_corner_localizer must be a SafeZoneCornerLocalizer "
                "or None."
            )
        if static_map is not None and not isinstance(static_map, StaticFieldMap):
            raise TypeError("static_map must be a StaticFieldMap or None.")
        if isinstance(breakup_clearance_mm, bool):
            raise ValueError(
                "breakup_clearance_mm must be a finite nonnegative number."
            )
        try:
            clearance_mm = float(breakup_clearance_mm)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "breakup_clearance_mm must be a finite nonnegative number, "
                f"got {breakup_clearance_mm!r}."
            ) from exc
        if not math.isfinite(clearance_mm) or clearance_mm < 0.0:
            raise ValueError(
                "breakup_clearance_mm must be finite and nonnegative, "
                f"got {breakup_clearance_mm!r}."
            )
        self._breakup_static_map = static_map
        self._breakup_clearance_mm = clearance_mm
        self._safe_zone_corner_localizer = safe_zone_corner_localizer
        self._safe_zone_phase = "idle"
        self._transport_opened = False
        self._latest_heading_rad: float | None = None
        self._raw_heading_rad: float | None = None
        self._heading_offset_rad = 0.0
        self._latest_perception: PerceptionSnapshot | None = None
        self._green_reference_samples: list[GroundPoint] = []
        self._green_reference_last_seen_ns: int | None = None
        self._green_reference: GroundPoint | None = None
        self._green_reference_heading_rad: float | None = None
        self._green_reference_distance_m: float | None = None
        self._green_alignment_started_ns: int | None = None
        self._green_alignment_last_frame_sequence: int | None = None
        self._green_alignment_stable_count = 0
        self._opportunistic_single_green = False
        self._near_field_group_preview = False
        self._green_realign_pending = False
        self._green_realign_done = False
        self._green_preclose_consumed_track_ids: set[int] = set()
        self._green_preclose_carried_count = 0
        self._green_preclose_realign_active = False
        self._green_preclose_frame_floor: int | None = None
        self._green_preclose_recheck_started_ns: int | None = None
        self._green_preclose_recheck_hold_ns = round(
            float(config.green_preclose_recheck_hold_ms) * 1_000_000
        )
        self._safe_zone_confirmation_deadline_ns: int | None = None
        self._safe_zone_observation_deadline_ns: int | None = None
        self._safe_zone_key_samples: list[tuple[GroundPoint | None, GroundPoint | None, GroundPoint | None]] = []
        self._safe_zone_key_last_frame: int | None = None
        self._safe_zone_key_reobserve_until_ns: int | None = None
        self._safe_zone_key_reobserve_frame_floor: int | None = None
        self._safe_zone_keypoint_reverse_base_distance_m: float | None = None
        self._safe_zone_keypoint_reverse_attempted = False
        self._safe_zone_calibration_snapshot: PerceptionSnapshot | None = None
        self._safe_zone_calibration_zone: SafeZoneObservation | None = None
        self._safe_zone_keys: tuple[GroundPoint | None, GroundPoint | None, GroundPoint | None] | None = None
        self._safe_zone_calibration_pose: SafeZoneCornerPoseObservation | None = None
        self._safe_zone_calibration_heading_rad: float | None = None
        self._safe_zone_calibration_last_failure: str | None = None
        self._safe_zone_stop_since_ns: int | None = None
        self._safe_zone_bbox_turn_direction: float | None = None
        self._latest_speed_feedback: tuple[float | None, float | None] = (
            None,
            None,
        )
        self._path_recovery = False
        self._path_recovery_stopped_ns: int | None = None
        self._path_recovery_posture = GripperPosture.CLOSED
        self._return_phase = "idle"
        self._safe_zone_exit_base_distance_m: float | None = None
        self._d1_line_heading_rad: float | None = None
        self._d1_line_distance_m: float | None = None
        self._d1_line_start_position: FieldPoint | None = None
        self._d1_calibration_offset_mm = (
            self.config.safe_zone_calibration_start_offset_mm
        )
        self._d2_line_heading_rad: float | None = None
        self._d2_line_distance_m: float | None = None
        self._d2_line_start_position: FieldPoint | None = None
        self._latest_cumulative_distance_m: float | None = None
        self._action_settle_phase: str | None = None
        self._action_settle_until_ns: int | None = None
        self._search_frame_floor: int | None = None
        self._cluster_reference_samples: list[GroundPoint] = []
        self._cluster_distance_samples_mm: list[float] = []
        self._cluster_reference: GroundPoint | None = None
        self._cluster_reference_distance_mm: float | None = None
        self._cluster_capture_heading_rad: float | None = None
        self._cluster_reference_field_point: FieldPoint | None = None
        self._cluster_breakup_end_field_point: FieldPoint | None = None
        self._last_cluster_rejection_reason: str | None = None

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
        *,
        start_area: MatchStartArea | str | int = MatchStartArea.AREA_2,
    ) -> MatchSequence:
        """从唯一运行配置创建策略，不打开任何硬件资源。"""

        from rescue_vision.config import AppConfig

        if not isinstance(config, AppConfig):
            raise TypeError("config must be an AppConfig.")
        config = configure_match_start_area(config, start_area)
        runtime = config.match
        if not runtime.enabled:
            raise ValueError("match.enabled must be true.")
        if not config.geometry.ground_mapping_enabled:
            raise RuntimeError("Match flow requires ground mapping.")
        gripper = config.motion.gripper.build_calibration()
        if gripper is None:
            raise RuntimeError("Match flow requires gripper calibration.")
        near_field_pickup = GripperWidthPickupSequence(
            black_closed_servo_offset_deg=config.near_field_grasp.black_closed_servo_offset_deg,
            gripper_full_travel_time_s=gripper.full_travel_time_s,
            forward_speed_m_s=runtime.green_approach_speed_m_s,
            cruise_speed_scale=runtime.pickup_cruise_speed_scale,
            terminal_speed_gain_s_inv=runtime.pickup_terminal_speed_gain_s_inv,
            deceleration_m_s2=min(
                0.5, config.motion.max_linear_deceleration_m_s2
            ),
            closed_servo_angles_deg=(
                gripper.closed_left_angle_deg,
                gripper.closed_right_angle_deg,
            ),
            max_observation_age_ms=config.processing.max_observation_age_ms,
            alignment_kp_rad_s=runtime.green_alignment_kp_rad_s,
            alignment_max_angular_velocity_rad_s=(
                runtime.green_alignment_max_angular_velocity_rad_s
            ),
            alignment_min_wheel_velocity_m_s=(
                runtime.green_alignment_min_wheel_velocity_m_s
            ),
            alignment_timeout_ms=config.near_field_grasp.alignment_timeout_ms,
            alignment_continue_max_age_ms=(
                config.near_field_grasp.alignment_continue_max_age_ms
            ),
            grasp_commit_max_observation_age_ms=(
                config.near_field_grasp.grasp_commit_max_observation_age_ms
            ),
            stationary_max_gyro_rad_s=config.near_field_grasp.stationary_max_gyro_rad_s,
            fine_alignment_zone_rad=config.near_field_grasp.fine_alignment_zone_rad,
            fine_alignment_min_wheel_velocity_m_s=(
                config.near_field_grasp.fine_alignment_min_wheel_velocity_m_s
            ),
        )
        transport_angles = gripper.transport_angles_deg
        if transport_angles is None:
            raise RuntimeError("Match flow requires transport gripper calibration.")
        kinematics = GripperKinematics()
        transport_left_relative_angle_deg = (
            gripper.closed_left_angle_deg - transport_angles[0]
        )
        transport_right_relative_angle_deg = (
            transport_angles[1] - gripper.closed_right_angle_deg
        )
        transport_left_tip = kinematics.left_tip_position(
            transport_left_relative_angle_deg
        )
        transport_right_tip = kinematics.right_tip_position(
            transport_right_relative_angle_deg
        )
        transport_opening_width_mm = transport_left_tip.y - transport_right_tip.y
        if not math.isfinite(transport_opening_width_mm) or transport_opening_width_mm <= 0.0:
            raise RuntimeError(
                "Match flow transport gripper calibration produces an invalid opening."
            )
        initial = config.localization.fusion.initial_pose
        expected_position = (
            FieldPoint(-1350.0, -1350.0)
            if config.world.team_color is TeamColor.BLUE
            else FieldPoint(1350.0, 1350.0)
        )
        expected_heading = (
            -cls.INITIAL_HEADING_RAD
            if config.world.team_color is TeamColor.BLUE
            else cls.INITIAL_HEADING_RAD
        )
        if not (
            math.isclose(initial.position.x, expected_position.x, abs_tol=1e-6)
            and math.isclose(initial.position.y, expected_position.y, abs_tol=1e-6)
            and math.isclose(initial.heading_rad, expected_heading, abs_tol=1e-6)
        ):
            raise RuntimeError(
                "Match flow requires localization.fusion.initial_pose "
                "to match the selected start area: "
                f"expected position={expected_position}, heading_rad={expected_heading:g}; "
                f"got {initial!r}."
            )
        sequence = cls(
            runtime,
            tracker=config.tracking.build_tracker(),
            gripper_full_travel_time_s=gripper.full_travel_time_s,
            team_color=config.world.team_color,
            initial_field_position=initial.position,
            near_field_pickup=near_field_pickup,
            near_field_grasp_config=config.near_field_grasp,
            transport_corridor_half_width_mm=transport_opening_width_mm / 2.0,
            breakup_target_geometry=config.perception.target_ground_geometry,
            gripper_color_config=config.perception.gripper_color,
        )
        sequence._safe_zone_corner_localizer = config.build_safe_zone_corner_localizer()
        sequence._breakup_static_map = config.world.static_map
        sequence._breakup_clearance_mm = (
            config.match.robot_footprint_radius_mm
            + config.match.safety_margin_mm
        )
        profile = config.motion.action_profile
        sequence._motion_profile = replace(
            profile,
            linear_deceleration_m_s2=min(profile.linear_deceleration_m_s2,
                                         config.motion.max_linear_deceleration_m_s2),
            angular_deceleration_rad_s2=min(profile.angular_deceleration_rad_s2,
                                          config.motion.max_angular_deceleration_rad_s2),
        )
        sequence._motion_wheel_track_m = config.motion.wheel_track_m or 0.2
        sequence._motion_max_wheel_speed_m_s = config.motion.max_wheel_velocity_m_s
        sequence._motion_max_angular_rad_s = config.motion.max_angular_velocity_rad_s
        sequence._motion_wheel_weights = (config.motion.left_wheel_speed_weight,
                                         config.motion.right_wheel_speed_weight)
        return sequence

    @property
    def grasp_task(self) -> GraspTask | None:
        return self._grasp_task

    def _failure_region_tolerance_mm(self) -> float:
        """失败记忆"同一尝试区域"的场地坐标容差。"""

        return max(120.0, self.config.cluster_group_ground_mm)

    def _same_target_identity_tolerance_mm(self) -> float:
        """同一物理目标身份判定的场地坐标容差。

        必须窄于成团尺度：相邻物资不能被并入同一物理目标；机器人系
        位移经采集位姿补偿后只剩里程漂移，不需要更宽的容差。
        """

        return min(100.0, self.config.cluster_group_ground_mm)

    def _adopt_grasp_task(self, timestamp_ns: int, *, track_id: int | None,
                          target_class: TargetClass | None, point: GroundPoint | None,
                          field: FieldPoint | None) -> None:
        task = self._grasp_task
        same = task is not None and (
            (field is not None and task.entry_field is not None
             and target_class is task.entry_class
             and math.hypot(field.x-task.entry_field.x, field.y-task.entry_field.y)
                 <= self._same_target_identity_tolerance_mm())
            or (track_id is not None and track_id == task.entry_track_id
                and target_class is task.entry_class)
        )
        if same:
            return
        if task is not None:
            self._remember_near_field_failure(timestamp_ns, "physical_target_replaced")
        self._grasp_task = GraspTask(
            self._next_grasp_task_id, timestamp_ns,
            timestamp_ns + round(self.config.grasp_task_timeout_ms * 1e6),
            track_id, target_class, field, point, last_progress_ns=timestamp_ns,
        )
        self._next_grasp_task_id += 1

    def _member_continues_grasp_entry(
        self,
        member: GraspTarget,
        task: GraspTask,
        timestamp_ns: int,
    ) -> bool:
        """判断近场计划成员是否仍是任务入口的同一物理目标。

        机器人系坐标会随接近、制动整体平移，同一物理目标只能按采集
        位姿补偿后的场地坐标判断；交接命中、入口场地锚点和主 tracker
        的入口轨迹都算关联线索，解团推移换号不应被误认成换目标。
        """

        if member.handoff_matched:
            return True
        observation = member.observation
        point = observation.ground_point
        if point is None:
            return False
        pose = self._pose_at(observation.capture_timestamp_ns)
        if pose is None:
            return False
        member_field = self._field_point_from_pose(pose, point)
        tolerance = self._same_target_identity_tolerance_mm()
        if (
            task.entry_field is not None
            and (task.entry_class is None
                 or observation.target_class is task.entry_class)
            and math.hypot(
                member_field.x - task.entry_field.x,
                member_field.y - task.entry_field.y,
            ) <= tolerance
        ):
            return True
        entry_track = None if task.entry_track_id is None else next(
            (
                item
                for item in self._tracker.tracks
                if item.track_id == task.entry_track_id
            ),
            None,
        )
        if entry_track is not None and self._target_is_fresh(entry_track, timestamp_ns):
            track_field = self._field_point_for_target(entry_track)
            if (
                track_field is not None
                and observation.target_class is entry_track.target_class
                and math.hypot(
                    member_field.x - track_field.x,
                    member_field.y - track_field.y,
                ) <= tolerance
            ):
                return True
        return False

    def _refresh_grasp_task_entry(self, member: GraspTarget) -> None:
        """把任务入口锚定到当前可执行核心成员的几何。

        同一物理目标经过接近、制动或解团推移后，入口的机器人系/场地
        锚点跟随最新可靠几何；主 tracker 轨迹只作关联线索重新映射，
        失败记录使用各自记录时的区域点，不随锚点漂移。
        """

        task = self._grasp_task
        point = member.observation.ground_point
        if task is None or point is None:
            return
        pose = self._pose_at(member.observation.capture_timestamp_ns)
        if pose is not None:
            task.entry_field = self._field_point_from_pose(pose, point)
        task.entry_ground = point
        task.entry_class = member.observation.target_class
        matches = [(math.hypot(item.ground_point.x-point.x, item.ground_point.y-point.y), item)
                   for item in self._tracker.tracks
                   if item.target_class is task.entry_class and item.ground_point is not None]
        selected = min(matches, key=lambda pair: pair[0])[1] if matches else None
        task.entry_track_id = None if selected is None else selected.track_id
        self._selected_track_id = task.entry_track_id

    def _refresh_grasp_association(self, timestamp_ns: int) -> None:
        task = self._grasp_task
        if (task is None or task.entry_field is None or self.state not in {
                MatchState.TRANSPORT_ALIGN_GREEN, MatchState.TRANSPORT_APPROACH_GREEN}):
            return
        matches = []
        for target in self._tracker.tracks:
            if target.target_class is not task.entry_class or not self._target_is_fresh(target, timestamp_ns):
                continue
            point = self._field_point_for_target(target)
            if point is not None:
                distance = math.hypot(point.x-task.entry_field.x, point.y-task.entry_field.y)
                if distance <= min(100.0, self.config.cluster_group_ground_mm):
                    matches.append((distance, target))
        matches.sort(key=lambda item: item[0])
        if matches and (len(matches) == 1 or matches[1][0]-matches[0][0] > 20.0):
            self._selected_track_id = matches[0][1].track_id
            self._selected_green_ground = matches[0][1].ground_point

    def _mark_breakup_targets(self, preparation: GraspPreparation) -> bool:
        """Freeze the group's scoring objectives before alignment or contact."""
        plan = preparation.recovery_plan
        pose = self._pose_at(preparation.capture_timestamp_ns)
        if plan is None or pose is None:
            return False
        members = tuple(item for item in preparation.targets
                        if item.track_id in plan.member_ids and item.observed
                        and item.selectable and item.observation.ground_point is not None
                        and item.observation.target_class in {
                            TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE,
                            TargetClass.ORANGE_INJURED,
                        })
        if not members:
            return False
        if self._grasp_task is None:
            entry = next((item for item in members
                          if item.observation.target_class in self.near_field_policy.allowed_classes), None)
            if entry is None:
                return False
            point = entry.observation.ground_point
            self._adopt_grasp_task(preparation.capture_timestamp_ns, track_id=None,
                                  target_class=entry.observation.target_class, point=point,
                                  field=self._field_point_from_pose(pose, point))
        task = self._grasp_task
        assert task is not None
        # A further recovery cannot silently replace the original physical group.
        if not task.marked_targets:
            task.marked_targets = tuple(BreakupMarkedTarget(
                item.observation.target_class,
                self._field_point_from_pose(pose, item.observation.ground_point),
                preparation.capture_timestamp_ns, pose.cumulative_distance_m,
                next((track.track_id for track in self._tracker.tracks
                      if track.last_seen_timestamp_ns == preparation.capture_timestamp_ns
                      and track.target_class is item.observation.target_class
                      and track.ground_point == item.observation.ground_point), None),
            ) for item in members)
        return True

    def _update_breakup_marks(self, timestamp_ns: int) -> None:
        task, scene = self._grasp_task, self._latest_perception
        if task is None or not task.marked_targets or scene is None:
            return
        pose = self._pose_at(scene.capture_timestamp_ns)
        plan = (self._breakup_proposal if self.state is MatchState.BREAKUP_SETTLE
                else self._breakup_plan)
        if pose is None or plan is None or not self._fresh_perception(scene, timestamp_ns):
            return
        observations = [(index, item, self._field_point_from_pose(pose, item.ground_point))
                        for index, item in enumerate(scene.observations)
                        if item.ground_point is not None
                        and self._observation_is_selectable(item, 0.0, scene)]
        proposals = []
        c, s = math.cos(plan.heading_rad), math.sin(plan.heading_rad)
        for mark in task.marked_targets:
            if scene.capture_timestamp_ns <= mark.capture_timestamp_ns:
                continue
            # Allow measured contact travel along the push ray, but do not write
            # the predicted displacement back as an observed position.
            base = self._breakup_forward_base_distance_m
            peak = pose.cumulative_distance_m if base is None else max(
                pose.cumulative_distance_m, base + self._breakup_actual_forward_mm / 1000)
            push_mm = min(plan.forward_distance_mm,
                          max(0.0, peak-mark.capture_distance_m)*1000)
            if (self.state not in {MatchState.BREAKUP_FORWARD, MatchState.BREAKUP_BACKWARD,
                                   MatchState.CHECK_ISOLATED_GREEN}
                    or (self.state is MatchState.CHECK_ISOLATED_GREEN
                        and mark.capture_timestamp_ns > task.scene_floor_ns)):
                push_mm = 0.0
            tolerance = self._same_target_identity_tolerance_mm()
            matches = []
            for index, item, point in observations:
                if item.target_class is not mark.target_class:
                    continue
                dx, dy = point.x-mark.field_point.x, point.y-mark.field_point.y
                along, lateral = c*dx+s*dy, -s*dx+c*dy
                residual = math.hypot(along-_clamp(along, 0.0, push_mm), lateral)
                if residual <= tolerance:
                    matches.append((residual, index, point, item))
            matches.sort(key=lambda match: (match[0], match[1]))
            if matches and (len(matches) == 1 or matches[1][0]-matches[0][0] > 20.0):
                proposals.append((mark, matches[0]))
        for mark, (_, index, point, item) in proposals:
            if sum(match[1] == index for _, match in proposals) != 1:
                continue
            mark.field_point = point
            mark.capture_timestamp_ns = scene.capture_timestamp_ns
            mark.capture_distance_m = pose.cumulative_distance_m
            mark.track_id = next((track.track_id for track in self._tracker.tracks
                                  if track.frame_sequence == scene.frame_sequence
                                  and track.target_class is item.target_class
                                  and track.ground_point == item.ground_point), None)

    def _reacquire_breakup_target(self, timestamp_ns: int) -> MatchDecision:
        task = self._grasp_task
        assert task is not None
        pose = self._pose_at(timestamp_ns)
        scene = self._latest_perception
        if pose is not None and scene is not None:
            candidates = []
            for mark in task.marked_targets:
                if (mark.target_class not in self.near_field_policy.allowed_classes
                        or mark.capture_timestamp_ns <= task.scene_floor_ns
                        or mark.capture_timestamp_ns != scene.capture_timestamp_ns
                        or not self._fresh_perception(scene, timestamp_ns)):
                    continue
                dx, dy = mark.field_point.x-pose.position.x, mark.field_point.y-pose.position.y
                c, s = math.cos(pose.heading_rad), math.sin(pose.heading_rad)
                point = GroundPoint(c*dx+s*dy, -s*dx+c*dy)
                if point.x > 0:
                    candidates.append((mark, point))
            candidates.sort(key=lambda item: (
                item[0].target_class is not task.entry_class,
                math.hypot(item[1].x, item[1].y)))
            if candidates:
                mark, point = candidates[0]
                task.entry_class, task.entry_field = mark.target_class, mark.field_point
                task.entry_ground, task.entry_track_id = point, mark.track_id
                task.last_progress_ns = timestamp_ns
                decision = self._begin_near_field_grasp(timestamp_ns, handoff_prior=
                    NearFieldHandoffPrior(mark.target_class, point, mark.track_id))
                return replace(decision, reason="breakup_complete_replan_grasp_task")
        started = task.recovery_reobserve_started_ns
        assert started is not None
        if task.recovery_reobserve_deadline_ns is None:
            scene_delay_ms = 0.0 if scene is None else max(
                0.0, (scene.result_timestamp_ns-scene.capture_timestamp_ns)/1e6)
            interval_ms = (self._perception_interval_ns or 0) / 1e6
            window_ms = max(self.config.breakup_no_plan_reobserve_ms, 2*interval_ms+scene_delay_ms)
            task.recovery_reobserve_deadline_ns = min(task.deadline_ns, started+round(window_ms*1e6))
        if timestamp_ns >= task.recovery_reobserve_deadline_ns:
            return self._return_to_near_field_search(timestamp_ns, "breakup_marked_target_unobserved")
        return self._decision(timestamp_ns, 0.0, 0.0,
            f"breakup_reacquire_marked:seen={sum(m.capture_timestamp_ns > task.scene_floor_ns for m in task.marked_targets)}"
            f"/{len(task.marked_targets)},capture_age_ms={None if scene is None else (timestamp_ns-scene.capture_timestamp_ns)/1e6},"
            f"stationary_since_ns={self._breakup_stationary_since(timestamp_ns)},"
            f"prepare_age_ms=none,deadline_ns={task.recovery_reobserve_deadline_ns}",
            posture=GripperPosture.CLOSED, soft_brake=True)

    def grasp_task_diagnostic(self, timestamp_ns: int) -> str:
        task = self._grasp_task
        if task is None:
            return "grasp_task=none"
        return (f"grasp_task={task.task_id} task_stage={self.state.value} "
                f"entry_track={task.entry_track_id} entry_field={task.entry_field} "
                f"core={tuple(member.track_id for member in task.core)} "
                f"scene_revision={task.scene_revision} recoveries={task.recovery_count} "
                f"marked_targets={tuple((m.target_class.value, m.track_id, m.field_point, m.capture_timestamp_ns) for m in task.marked_targets)} "
                f"blocking={task.blocking} last_progress_ns={task.last_progress_ns} "
                f"task_deadline_ns={task.deadline_ns}")

    @property
    def selected_track_id(self) -> int | None:
        return self._selected_track_id

    @property
    def preview_selected_track_ids(self) -> tuple[int, ...]:
        """返回当前流程选中的目标 ID，供本地预览叠加使用。

        远场单目标、解团时锁定的目标团和近场已锁定的多目标计划都通过同一
        入口提供。这里不返回候选全集；候选仍由感知帧中的普通 bbox 表示。
        """

        selected: set[int] = set()
        if self._selected_track_id is not None:
            selected.add(self._selected_track_id)
        if self._near_field_pickup is not None:
            if self._near_field_pickup.active_plan is not None:
                selected.update(self._near_field_pickup.active_plan.member_ids)
            elif self._near_field_pickup.locked_ids is not None:
                selected.update(self._near_field_pickup.locked_ids)
        if not selected:
            selected.update(self._cluster_selected_track_ids)
        return tuple(sorted(selected))

    def preview_selected_targets(
        self,
        frame_sequence: int,
        capture_timestamp_ns: int,
    ) -> tuple[TrackedTarget, ...]:
        """返回属于指定图像帧的当前选中轨迹，避免预览错配旧 bbox。"""

        if (
            isinstance(frame_sequence, bool)
            or not isinstance(frame_sequence, int)
            or frame_sequence < 0
        ):
            raise ValueError("frame_sequence must be a non-negative integer.")
        if (
            isinstance(capture_timestamp_ns, bool)
            or not isinstance(capture_timestamp_ns, int)
            or capture_timestamp_ns < 0
        ):
            raise ValueError("capture_timestamp_ns must be a non-negative integer.")
        selected = set(self.preview_selected_track_ids)
        if self._grasp_task is not None:
            selected.update(mark.track_id for mark in self._grasp_task.marked_targets
                            if mark.track_id is not None
                            and mark.capture_timestamp_ns == capture_timestamp_ns)
        tracks = self._preview_target_history.get(
            (frame_sequence, capture_timestamp_ns),
            self._tracker.tracks,
        )
        return tuple(
            target
            for target in tracks
            if target.track_id in selected
            and target.frame_sequence == frame_sequence
            and target.last_seen_timestamp_ns == capture_timestamp_ns
        )

    def observe_grasp_motion(self, message: OdometryImu) -> None:
        """将真实轮编码器和IMU样本交给近场几何有效性检查。"""
        previous = self._stationary_motion.latest
        self._stationary_motion.observe(message)
        if (self._grasp_task is not None and previous is not None
                and message.sample_timestamp_us > previous.sample_timestamp_us
                and 0 <= message.received_timestamp_ns-previous.received_timestamp_ns <= self._stationary_motion.max_gap_ns
                and math.isfinite(message.gyro_z_rad_s)
                and self._stationary_motion.invalid_reason in {"rotation", "encoder_motion"}):
            self._grasp_task.last_progress_ns = message.received_timestamp_ns

    @property
    def near_field_enabled(self) -> bool:
        return self._near_field_pickup is not None

    @property
    def near_field_session_id(self) -> int:
        return self._near_field_session_id

    @property
    def near_field_locked_ids(self) -> tuple[int, ...] | None:
        return (
            None
            if self._near_field_pickup is None
            else self._near_field_pickup.locked_ids
        )

    @property
    def near_field_active_plan(self) -> NearFieldGraspPlan | None:
        return (
            None
            if self._near_field_pickup is None
            else self._near_field_pickup.active_plan
        )

    @property
    def near_field_result(self) -> GripperWidthPickupResult | None:
        return (
            None
            if self._near_field_pickup is None
            else self._near_field_pickup.result
        )

    @property
    def near_field_route(self) -> GraspRoute:
        return self._near_field_route

    @property
    def near_field_handoff_prior(self) -> NearFieldHandoffPrior | None:
        """返回当前近场决策阶段使用的远场目标先验。"""

        return self._near_field_handoff_prior

    @property
    def near_field_handoff_required(self) -> bool:
        """补夹尚未锁定执行计划时，要求近场计划保留交接目标。"""

        return (
            (self._greedy_active or bool(self._grasp_task and self._grasp_task.marked_targets)
             or bool(self._gate_clearance and self._gate_clearance.reacquiring))
            and self._near_field_handoff_prior is not None
            and self._near_field_pickup is not None
            and self._near_field_pickup.locked_ids is None
        )

    @property
    def near_field_last_failure_diagnostic(self) -> str | None:
        """返回最近一次近场路由失败的结构化诊断文本。"""

        return self._near_field_last_failure_diagnostic

    @property
    def near_field_route_diagnostic(self) -> str:
        elapsed = self._near_field_route_elapsed_ms
        return (
            f"route={self._near_field_route.value},"
            f"far_reapproach_used={self._near_field_far_reapproach_used},"
            f"candidates={self._near_field_route_candidate_count},"
            f"confirmation_ms={elapsed:.1f},"
            f"physical_failure_records={len(self._near_field_failures)},"
            f"rejections={self._near_field_route_rejections}"
        )

    def near_field_confirmation_diagnostic(
        self,
        timestamp_ns: int,
        preparation: GraspPreparation | None = None,
    ) -> str:
        """返回近场窗口、确认进度和两种年龄诊断。"""

        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        started = self._near_field_confirmation_started_ns
        elapsed = (
            None
            if started is None
            else max(0.0, (timestamp_ns - started) / 1_000_000.0)
        )
        prior = self._near_field_handoff_prior
        if prior is None:
            prior_text = "none"
        else:
            prior_text = (
                f"track={prior.source_track_id},class={prior.target_class.value},"
                f"xy=({prior.ground_point.x:.1f},{prior.ground_point.y:.1f})"
            )
        if preparation is None:
            plan_age_text = "none"
            preparation_age_text = "none"
            confirmation_text = "none"
            capture_age_text = "none"
        else:
            capture_age = timestamp_ns - preparation.capture_timestamp_ns
            capture_age_text = (
                "none" if capture_age < 0 else f"{capture_age / 1_000_000.0:.1f}"
            )
            plan_age = preparation.plan_age_ns(timestamp_ns)
            preparation_age = preparation.preparation_age_ns(timestamp_ns)
            plan_age_text = (
                "none" if plan_age is None else f"{plan_age / 1_000_000.0:.1f}"
            )
            preparation_age_text = (
                "none"
                if preparation_age is None
                else f"{preparation_age / 1_000_000.0:.1f}"
            )
            confirmation_text = (
                f"{preparation.confirmation_count}/"
                f"{preparation.confirmation_required}"
            )
        started_text = "none" if started is None else f"{started / 1_000_000.0:.1f}"
        elapsed_text = "none" if elapsed is None else f"{elapsed:.1f}"
        deadline_text = (
            "none"
            if started is None
            else f"{(started + round(self._near_field_handoff_timeout_ms() * 1_000_000.0)) / 1_000_000.0:.1f}"
        )
        return (
            f"confirmation_started_ms={started_text} "
            f"confirmation_elapsed_ms={elapsed_text} "
            f"attempt_deadline_ms={deadline_text} "
            f"no_plan_budget_ms={self._near_field_no_plan_wait_ms():.1f} "
            f"capture_age_ms={capture_age_text} "
            f"plan_age_ms={plan_age_text} "
            f"preparation_age_ms={preparation_age_text} "
            f"confirmation={confirmation_text} "
            f"handoff_prior={prior_text} "
            f"{self.grasp_task_diagnostic(timestamp_ns)} "
            + (self._near_field_pickup.motion_diagnostic(timestamp_ns)
               if self._near_field_pickup is not None else "")
        )

    def near_field_progress_mm(self, cumulative_distance_m: float | None) -> float:
        if self._near_field_pickup is None:
            return 0.0
        return self._near_field_pickup.progress_mm(cumulative_distance_m)

    def near_field_plan_path_clear(
        self,
        plan: NearFieldGraspPlan,
        heading_rad: float | None,
    ) -> bool | None:
        if not isinstance(plan, NearFieldGraspPlan):
            raise TypeError("plan must be a NearFieldGraspPlan.")
        return self._near_field_path_clear(plan, heading_rad)

    @property
    def breakup_preview_plan(self) -> BreakupPlan | None:
        if not self._dynamic_breakup_enabled or self.state not in self._BREAKUP_ACCELERATION_LIMIT_STATES:
            return None
        return self._breakup_plan or self._breakup_proposal

    @property
    def breakup_observing(self) -> bool:
        # 解团确认只消费 MatchSequence 自己的当前帧，不启动近场抓取准备器。
        # CLOSE_GRIPPER_SPIN 仅用于固定动作入口；正式解团闭爪后回到当前抓取任务。
        return self._dynamic_breakup_enabled and self.state is MatchState.CLOSE_GRIPPER_SPIN

    @property
    def breakup_last_failure_diagnostic(self) -> str | None:
        """Return the most recent bounded breakup failure for telemetry."""

        return self._breakup_last_failure_diagnostic

    def near_field_observation_window_open(self, timestamp_ns: int) -> bool:
        """报告近场是否已经完成刹车并可打开确认窗口。

        该门禁供硬件循环在提交后台准备任务前调用。动作状态机本身仍在
        下一次 ``step`` 中消费 settle，到期前的帧不能进入确认窗口。
        """

        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if self._stationary_motion.stationary_since(timestamp_ns) is None:
            return False
        if self.breakup_observing:
            return self._breakup_stopped_ns is not None
        if self.state is not MatchState.TRANSPORT_NEAR_FIELD_GRASP:
            return False
        if self._action_settle_phase != "near_field_grasp":
            is_open = True
        else:
            until_ns = self._action_settle_until_ns
            is_open = until_ns is not None and timestamp_ns >= until_ns
        if (
            is_open
            and self._near_field_route is GraspRoute.DECIDING
            and self._near_field_confirmation_started_ns is None
        ):
            # 决策预算从停车观察窗口真正打开时开始，不能被前置 settle 消耗。
            self._near_field_confirmation_started_ns = timestamp_ns
            scene = self._latest_perception
            if self._perception_interval_ns is not None and scene is not None:
                delivery_ms = (scene.result_timestamp_ns-scene.capture_timestamp_ns)/1e6
                frame_ms = self._perception_interval_ns/1e6
                # Freeze an evidence-based two-frame delivery window. It cannot
                # grow on each poll or consume the confirmed-plan deadline.
                submit_ms = self._near_field_handoff_timeout_ms()
                reserve_ms = self._stationary_motion.max_gap_ns/1e6
                self._near_field_no_plan_budget_ms = min(
                    max(self._near_field_no_plan_wait_ms(), 2*frame_ms+delivery_ms),
                    max(0.0, submit_ms-reserve_ms),
                )
        return is_open

    def grasp_recovery_context(self, snapshot: PerceptionSnapshot) -> BreakupSceneContext | None:
        """Capture-pose inputs only; the worker owns combination and boundary search."""
        task = self._grasp_task
        if (not self._dynamic_breakup_enabled or self._greedy_active
                or self._transport_target_classes or self._breakup_static_map is None
                or (task is not None and task.recovery_count >= self.config.breakup_max_attempts)):
            return None
        pose = self._pose_at(snapshot.capture_timestamp_ns)
        if pose is None:
            return None
        return BreakupSceneContext(
            self.config, pose.position, pose.heading_rad, self._breakup_static_map,
            self._physical_field_bounds(), GripperKinematics().left_tip_position(0).x,
            self._breakup_plan if task is not None and task.recovery_count else None,
            1 if task is None else task.recovery_count + 1,
            None if task is None else task.entry_field,
        )

    def grasp_scene_capture_valid(self, snapshot: PerceptionSnapshot, timestamp_ns: int) -> bool:
        """Stable capture and bounded delivery; generic stream expiry is separate."""
        return (
            snapshot.dropped_stale_age_ms is None
            and 0 <= timestamp_ns - snapshot.result_timestamp_ns <= self.config.green_max_age_ms * 1e6
            and self._stationary_motion.capture_valid(
                snapshot.capture_timestamp_ns, timestamp_ns,
                max_age_ns=round(self._near_field_handoff_timeout_ms() * 1e6),
            )
        )

    @property
    def near_field_policy(self) -> NearFieldGraspPolicy:
        gate = self._gate_clearance
        if gate is not None and gate.reacquiring:
            return NearFieldGraspPolicy(frozenset(gate.original_classes), len(gate.original_classes),
                                        obstacle_extent_required=True)
        if self._transport_count == 0:
            return NearFieldGraspPolicy(
                frozenset((TargetClass.GREEN_SUPPLY,)), 1,
                obstacle_extent_required=self._dynamic_breakup_enabled,
            )
        if self._greedy_active:
            return NearFieldGraspPolicy(
                frozenset((TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE)),
                max(1, self._supply_capacity() - len(self._transport_target_classes)),
                target_final_x_mm=(
                    None
                    if self._near_field_grasp_config is None
                    else self._near_field_grasp_config.greedy_target_final_x_mm
                ),
            )
        return NearFieldGraspPolicy(
            frozenset(
                (
                    TargetClass.GREEN_SUPPLY,
                    TargetClass.BLACK_CORE,
                    TargetClass.ORANGE_INJURED,
                )
            ),
            3
            if self._near_field_grasp_config is None
            else self._near_field_grasp_config.max_targets,
        )

    @property
    def transport_corridor_half_width_mm(self) -> float:
        """固定 ``TRANSPORT`` 姿态的物理开口半宽，单位 mm。"""

        return self._transport_corridor_half_width_mm

    @property
    def transport_corridor_effective_half_width_mm(self) -> float:
        """固定 ``TRANSPORT`` 走廊半宽（含近场横向安全余量），单位 mm。"""

        return self._transport_corridor_effective_half_width_mm()

    @property
    def estimated_field_position(self) -> FieldPoint | None:
        """当前受限航位估计；来源为初始位姿、编码器和陀螺仪。"""

        return self._fallback_field_position

    def preflight(
        self,
        timestamp_ns: int,
        checks: MatchPreflight,
    ) -> MatchDecision:
        self._validate_timestamp(timestamp_ns)
        if self.state is not MatchState.BOOT:
            raise RuntimeError("preflight() can only be called from BOOT.")
        if not isinstance(checks, MatchPreflight):
            raise TypeError("checks must be a MatchPreflight.")
        if not checks.ready:
            self.state = MatchState.TERMINAL_STOP
            return self._decision(timestamp_ns, 0.0, 0.0, "preflight_failed")
        self.state = MatchState.PREFLIGHT
        return self._decision(timestamp_ns, 0.0, 0.0, "preflight_ready")

    def start(self, timestamp_ns: int) -> MatchDecision:
        self._validate_timestamp(timestamp_ns)
        if self.state is not MatchState.PREFLIGHT:
            raise RuntimeError("start() requires a successful PREFLIGHT.")
        self._gate_clearance = None
        self._gate_clearance_attempted = False
        self._started = True
        self._startup_turn_last_heading = None
        self._startup_turn_progress_rad = 0.0
        self._reset_straight_pid()
        self._fallback_field_position = self._initial_field_position
        self._fallback_last_distance_m = None
        self._fallback_last_heading_rad = None
        self._noncontact_key = None
        self._noncontact_controller = None
        self._noncontact_started_ns = None
        self._latest_cumulative_distance_m = None
        self._pose_history.clear()
        self._grasp_task = None
        self._near_field_failures.clear()
        self._breakup_failed_aims.clear()
        self._breakup_failed_aim_positions.clear()
        self._breakup_attempts.clear()
        self._breakup_attempt_positions.clear()
        self._green_target_field_point = None
        self._green_alignment_target_heading_rad = None
        self._green_turn_start_heading_rad = None
        self._green_turn_direction = 0.0
        self._green_turn_budget_rad = 0.0
        self._green_turn_progress_rad = 0.0
        self._green_turn_attempts = 0
        self._green_turn_active = False
        self._green_alignment_wait_since_ns = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id = 0
            self._near_field_handoff_prior = None
            self._near_field_last_failure_diagnostic = None
            self._near_field_confirmation_started_ns = None
            self._near_field_far_reapproach_used = False
        self._transport_target_classes = ()
        self._cargo_capture_floor_ns = None
        self._counted_pickup_session = None
        self._greedy_active = False
        self._begin_safe_zone_scan()
        self._begin_cluster_search()
        self.state = MatchState.STARTUP_TURN_RIGHT
        return self._decision(timestamp_ns, 0.0, 0.0, "strategy_started")

    def _step_actions(
        self,
        timestamp_ns: int,
        *,
        perception: PerceptionSnapshot | None,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None = None,
        right_speed_feedback_m_s: float | None = None,
        safety: SafetySignals | None = None,
        near_field_preparation: GraspPreparation | None = None,
        near_field_path_clear: bool | None = None,
    ) -> MatchDecision:
        """消费最新快照并返回本周期控制意图。"""

        self._validate_timestamp(timestamp_ns)
        if not self._started:
            return self._decision(timestamp_ns, 0.0, 0.0, "waiting_start")
        if heading_rad is not None and not math.isfinite(float(heading_rad)):
            raise ValueError("heading_rad must be finite when present.")
        if perception is not None and not isinstance(perception, PerceptionSnapshot):
            raise TypeError("perception must be a PerceptionSnapshot or None.")
        if cumulative_distance_m is not None and not math.isfinite(
            float(cumulative_distance_m)
        ):
            raise ValueError("cumulative_distance_m must be finite when present.")
        self._update_fallback_field_position(heading_rad, cumulative_distance_m)
        self._record_pose_history(timestamp_ns, heading_rad, cumulative_distance_m)
        if safety is None:
            safety = SafetySignals.nominal(timestamp_ns)
        if not isinstance(safety, SafetySignals):
            raise TypeError("safety must be a SafetySignals or None.")
        if self.state in {
            MatchState.FINISH_STOP,
            MatchState.TERMINAL_STOP,
        }:
            return self._decision(timestamp_ns, 0.0, 0.0, "stop_is_latched")
        direct_reason = self._direct_safety_reason(safety)
        if direct_reason is not None:
            self.state = MatchState.TERMINAL_STOP
            return self._decision(timestamp_ns, 0.0, 0.0, direct_reason)
        self._update_tracker(timestamp_ns, perception)
        if self.state is MatchState.GATE_CLEARANCE:
            return step_clearance(self, timestamp_ns)
        if self.state in {MatchState.MISGRASP_OPEN, MatchState.MISGRASP_BACKUP, MatchState.MISGRASP_SETTLE}:
            return self._step_misgrasp_recovery(timestamp_ns, cumulative_distance_m)
        conflict = self._gripper_color_conflict(timestamp_ns)
        if conflict:
            return self._begin_misgrasp_recovery(timestamp_ns, conflict)
        gate_decision = maybe_begin_clearance(self, timestamp_ns)
        if gate_decision is not None:
            return gate_decision
        self._update_breakup_marks(timestamp_ns)
        self._refresh_grasp_association(timestamp_ns)
        self._breakup_grasp_preparation = near_field_preparation
        self._observe_breakup_feedback(timestamp_ns)
        task = self._grasp_task
        if (task is not None and timestamp_ns >= task.deadline_ns
                and self.state not in {
                    MatchState.BREAKUP_FORWARD, MatchState.BREAKUP_BACKWARD,
                    MatchState.OPEN_GRIPPER_SETTLE, MatchState.CLOSE_GRIPPER_SETTLE,
                }
                and self.near_field_active_plan is None):
            return self._return_to_near_field_search(timestamp_ns, "grasp_task_deadline")

        if self._dynamic_breakup_enabled:
            dynamic = self._step_dynamic_breakup(timestamp_ns, cumulative_distance_m)
            if dynamic is not None:
                return dynamic

        return self._dispatch_state(
            timestamp_ns,
            perception=perception,
            heading_rad=heading_rad,
            cumulative_distance_m=cumulative_distance_m,
            left_speed_feedback_m_s=left_speed_feedback_m_s,
            right_speed_feedback_m_s=right_speed_feedback_m_s,
            near_field_preparation=near_field_preparation,
            near_field_path_clear=near_field_path_clear,
        )

    def _dispatch_state(
        self,
        timestamp_ns: int,
        *,
        perception: PerceptionSnapshot | None,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None,
        right_speed_feedback_m_s: float | None,
        near_field_preparation: GraspPreparation | None,
        near_field_path_clear: bool | None,
    ) -> MatchDecision:
        """按当前状态分发控制意图；子类可先处理自身状态再委托父类。"""

        if self.state is MatchState.STARTUP_TURN_RIGHT:
            return self._step_startup_turn(timestamp_ns, heading_rad)
        if self.state is MatchState.STARTUP_TURN_SETTLE:
            return self._step_settle(
                timestamp_ns,
                MatchState.STARTUP_FORWARD,
                "startup_turn_settled",
            )
        if self.state is MatchState.STARTUP_FORWARD:
            return self._step_startup_forward(
                timestamp_ns,
                cumulative_distance_m,
                left_speed_feedback_m_s,
                right_speed_feedback_m_s,
            )
        if self.state is MatchState.STARTUP_FORWARD_SETTLE:
            return self._step_settle(
                timestamp_ns,
                MatchState.SEARCH_CLUSTER,
                "startup_forward_settled_search_cluster",
            )
        if self.state is MatchState.SEARCH_CLUSTER:
            return self._step_search_cluster(timestamp_ns, heading_rad)
        if self.state is MatchState.ALIGN_CLUSTER_ONCE:
            return self._step_align_cluster(timestamp_ns)
        if self.state is MatchState.APPROACH_CLUSTER:
            return self._step_approach_cluster(timestamp_ns, cumulative_distance_m)
        if self.state is MatchState.BREAKUP_SETTLE:
            return self._step_settle(
                timestamp_ns,
                MatchState.BREAKUP_FORWARD,
                "cluster_alignment_settled_start_breakup",
            )
        if self.state is MatchState.RELOCATE_FORWARD:
            return self._step_relocate_forward(timestamp_ns, cumulative_distance_m)
        if self.state is MatchState.BREAKUP_FORWARD:
            return self._step_breakup_forward(timestamp_ns, cumulative_distance_m)
        if self.state is MatchState.BREAKUP_BACKWARD:
            return self._step_breakup_backward(timestamp_ns, cumulative_distance_m)
        if self.state is MatchState.OPEN_GRIPPER_SETTLE:
            return self._step_gripper_settle(
                timestamp_ns,
                MatchState.BREAKUP_BACKWARD,
                GripperPosture.OPEN,
                "open_gripper_settled_start_backward_0_1m",
            )
        if self.state is MatchState.CLOSE_GRIPPER_SETTLE:
            return self._step_gripper_settle(
                timestamp_ns,
                MatchState.CLOSE_GRIPPER_SPIN,
                GripperPosture.CLOSED,
                "close_gripper_settled_spin_once",
            )
        if self.state is MatchState.CLOSE_GRIPPER_SPIN:
            target = self._find_isolated_green(timestamp_ns)
            if target is not None:
                return self._begin_green_transport(timestamp_ns, target)
            return self._step_spin(
                timestamp_ns,
                heading_rad,
                self.config.close_gripper_spin_angular_velocity_rad_s,
                GripperPosture.CLOSED,
                MatchState.CHECK_ISOLATED_GREEN,
                "close_gripper_spin_complete",
            )
        if self.state is MatchState.CHECK_ISOLATED_GREEN:
            return self._step_check_green(timestamp_ns)
        if self.state is MatchState.TRANSPORT_ALIGN_GREEN:
            return self._step_align_green(timestamp_ns)
        if self.state is MatchState.TRANSPORT_APPROACH_GREEN:
            return self._step_approach_green(timestamp_ns, cumulative_distance_m)
        if self.state is MatchState.TRANSPORT_GREEDY_SCAN:
            return self._step_greedy_scan(timestamp_ns, heading_rad)
        if self.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP:
            return self._step_near_field_grasp(
                timestamp_ns,
                cumulative_distance_m,
                near_field_preparation,
                near_field_path_clear,
            )
        if self.state is MatchState.TRANSPORT_PRE_CLOSE_RECHECK:
            return self._step_preclose_recheck(timestamp_ns)
        if self.state is MatchState.TRANSPORT_CLOSE_GRIPPER:
            return self._step_transport_close(timestamp_ns)
        if self.state is MatchState.TRANSPORT_ALIGN_RED_ZONE:
            return self._step_align_red_zone(
                timestamp_ns,
                perception,
                heading_rad,
                cumulative_distance_m,
            )
        if self.state is MatchState.TRANSPORT_FORWARD:
            return self._step_transport_forward(timestamp_ns, cumulative_distance_m)
        if self.state is MatchState.TRANSPORT_RELEASE:
            return self._step_transport_release(timestamp_ns)
        if self.state is MatchState.RETURN_BACKUP:
            return self._step_return_backup(timestamp_ns, cumulative_distance_m)
        return self._decision(timestamp_ns, 0.0, 0.0, f"unhandled_state:{self.state.value}")

    def _noncontact_feedback(self, timestamp_ns: int, progress: float) -> RelativeActionFeedback:
        latest = self._stationary_motion.latest
        age = (None if latest is None or latest.received_timestamp_ns > timestamp_ns else
               (timestamp_ns - latest.received_timestamp_ns) / 1e9)
        return RelativeActionFeedback(
            timestamp_ns, progress, self._latest_heading_rad,
            *self._latest_speed_feedback,
            None if latest is None else latest.gyro_z_rad_s,
            age, self._stationary_motion.stationary_since(timestamp_ns),
            None if latest is None else latest.received_timestamp_ns,
        )

    def _noncontact_motion(self, timestamp_ns: int, key: str, *, target: float,
                           speed: float, turn: bool = False, point: FieldPoint | None = None,
                           tolerance: float | None = None) -> RelativeActionCommand:
        """One action owns its goal, deadline and failure across control polls."""
        if self._noncontact_key != key:
            self._noncontact_key = key
            self._noncontact_started_ns = timestamp_ns
            self._noncontact_controller = None
            heading = self._latest_heading_rad
            position = self.estimated_field_position
            travel = abs(target) if point is None else (
                0.0 if position is None else math.hypot(point.x-position.x, point.y-position.y)/1000)
            rotation_s = 0.0
            if point is not None and position is not None and heading is not None:
                rotation_s = abs(normalize_angle(math.atan2(point.y-position.y, point.x-position.x)-heading)) / min(
                    self._motion_max_angular_rad_s, self.config.safe_zone_fallback_max_angular_velocity_rad_s)
            deceleration = (self._motion_profile.angular_deceleration_rad_s2 if turn
                            else self._motion_profile.linear_deceleration_m_s2)
            self._noncontact_budget_s = max(self._motion_profile.action_timeout_s,
                travel / speed + rotation_s + 2 * speed / deceleration
                + self._motion_profile.stationary_confirm_time_s + self._motion_profile.correction_timeout_s)
            self._noncontact_stop_confirmed = False
        assert self._noncontact_started_ns is not None
        heading, distance = self._latest_heading_rad, self._latest_cumulative_distance_m
        position = self.estimated_field_position
        missing = heading is None or (not turn and distance is None) or (point is not None and position is None)
        expired = timestamp_ns - self._noncontact_started_ns >= self._noncontact_budget_s * 1e9
        if missing or expired:
            command = RelativeActionCommand(0.0, 0.0,
                RelativeActionPhase.TIMEOUT if expired else RelativeActionPhase.WAITING_FEEDBACK,
                False, expired, "noncontact_deadline" if expired else "critical_motion_pose_unavailable",
                0.0, None, 0.0, 0.0, True)
            self._noncontact_last_command = command
            return command
        controller = self._noncontact_controller
        if controller is None:
            profile = replace(self._motion_profile, action_timeout_s=self._noncontact_budget_s)
            if point is None:
                controller = RelativeActionController(profile)
            else:
                controller = PointActionController(profile,
                    wheel_track_m=self._motion_wheel_track_m,
                    max_wheel_speed_m_s=self._motion_max_wheel_speed_m_s,
                    max_angular_velocity_rad_s=min(self._motion_max_angular_rad_s,
                        self.config.safe_zone_fallback_max_angular_velocity_rad_s),
                    left_wheel_weight=self._motion_wheel_weights[0],
                    right_wheel_weight=self._motion_wheel_weights[1])
            controller.set_tolerances(
                position_tolerance=tolerance or (self.config.noncontact_heading_tolerance_rad if turn
                                                else self.config.noncontact_distance_tolerance_m),
                heading_tolerance=(tolerance if turn and tolerance is not None
                                   else self.config.noncontact_heading_tolerance_rad))
            self._noncontact_base_distance = distance or 0.0
            self._noncontact_last_heading = heading
            self._noncontact_turn_progress = 0.0
            self._noncontact_direction = 1.0 if target >= 0 else -1.0
            if isinstance(controller, PointActionController):
                controller.begin_point(point, pose=FieldPose2D(position, heading),
                    timestamp_ns=self._noncontact_started_ns, cruise_speed_m_s=speed)
            else:
                controller.begin(RelativeActionKind.TURN if turn else RelativeActionKind.STRAIGHT,
                    target if target != 0 else 1e-9, start_heading_rad=heading,
                    timestamp_ns=self._noncontact_started_ns, cruise_speed=speed,
                    target_heading_rad=normalize_angle(heading + target) if turn else heading)
            self._noncontact_controller = controller
        if turn:
            self._noncontact_turn_progress += self._noncontact_direction * normalize_angle(
                heading - self._noncontact_last_heading)
            self._noncontact_last_heading = heading
            progress = self._noncontact_turn_progress
        else:
            progress = self._noncontact_direction * ((distance or 0.0) - self._noncontact_base_distance)
        feedback = self._noncontact_feedback(timestamp_ns, progress)
        if isinstance(controller, PointActionController):
            command = controller.update_point(feedback, FieldPose2D(position, heading))
        else:
            command = controller.update(feedback)
        if command.complete:
            self._noncontact_stop_confirmed = True
        self._noncontact_last_command = command
        return command

    def _noncontact_decision(self, timestamp_ns: int, command: RelativeActionCommand,
                             posture: GripperPosture = GripperPosture.CLOSED) -> MatchDecision:
        action_key = self._noncontact_key
        action_started_ns = self._noncontact_started_ns
        if command.timed_out:
            # Action timeouts are recoverable during a match.  Drop the stale
            # controller so the next control poll replans from current motion
            # feedback/pose instead of latching the whole mission stopped.
            # Explicit emergency/safety termination is handled separately in
            # step() and remains latched.
            self._finish_noncontact()
            command = replace(command, timed_out=False,
                              reason=f"recoverable_timeout:{action_key}:{command.reason}")
        snapshot = self._latest_perception
        age = None if snapshot is None else (timestamp_ns-snapshot.capture_timestamp_ns)/1e9
        evidence = self._stationary_motion.stationary_since(timestamp_ns)
        deadline = (action_started_ns or 0) + round(self._noncontact_budget_s*1e9)
        return self._decision(timestamp_ns, command.linear_velocity_m_s, command.angular_velocity_rad_s,
            f"noncontact={action_key},phase={command.phase.value},{command.reason},"
            f"capture_age_s={age},stationary_since_ns={evidence},"
            f"confirmation={'complete' if command.complete else command.phase.value},"
            f"preparation_age_s={(timestamp_ns-(action_started_ns or timestamp_ns))/1e9},deadline_ns={deadline}",
            posture=posture, min_wheel_velocity_m_s=0.0)

    def _finish_noncontact(self) -> None:
        self._noncontact_controller = None
        self._noncontact_key = None
        self._noncontact_started_ns = None

    def _step_startup_turn(self, timestamp_ns: int, heading_rad: float | None) -> MatchDecision:
        command = self._noncontact_motion(timestamp_ns, "startup_turn",
            target=math.copysign(self.config.startup_turn_angle_rad,
                                 self.config.startup_turn_angular_velocity_rad_s),
            speed=abs(self.config.startup_turn_angular_velocity_rad_s), turn=True)
        if not command.complete:
            return self._noncontact_decision(timestamp_ns, command)
        self._finish_noncontact()
        self.state = MatchState.STARTUP_TURN_SETTLE
        self._settle_until_ns = timestamp_ns
        return self._decision(timestamp_ns, 0.0, 0.0, "startup_turn_complete_wait")

    def _step_startup_forward(
        self, timestamp_ns: int, cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None, right_speed_feedback_m_s: float | None,
    ) -> MatchDecision:
        command = self._noncontact_motion(timestamp_ns, "startup_forward",
            target=self.config.startup_forward_distance_m,
            speed=self.config.startup_forward_speed_m_s)
        if not command.complete:
            return self._noncontact_decision(timestamp_ns, command)
        self._finish_noncontact()
        self.state = MatchState.STARTUP_FORWARD_SETTLE
        self._settle_until_ns = timestamp_ns
        return self._decision(timestamp_ns, 0.0, 0.0, "startup_forward_complete_wait")

    def _step_settle(
        self,
        timestamp_ns: int,
        next_state: MatchState,
        reason: str,
    ) -> MatchDecision:
        if timestamp_ns < self._settle_until_ns:
            return self._decision(timestamp_ns, 0.0, 0.0, "waiting_settle")
        self.state = next_state
        if next_state is MatchState.STARTUP_FORWARD:
            self._startup_forward_base_distance_m = None
            self._reset_straight_pid()
        if next_state is MatchState.SEARCH_CLUSTER:
            self._begin_cluster_search()
        angular = (
            self._cluster_search_angular_velocity_rad_s
            if next_state is MatchState.SEARCH_CLUSTER
            else 0.0
        )
        return self._decision(timestamp_ns, 0.0, angular, reason)

    def _cluster_alignment_decision(
        self,
        timestamp_ns: int,
        heading_error_rad: float,
    ) -> MatchDecision:
        if abs(heading_error_rad) <= self.config.cluster_align_tolerance_rad:
            self._cluster_align_hold_center = None
            self._cluster_align_lost_since_ns = None
            self._consecutive_cluster_losses = 0
            self.state = MatchState.APPROACH_CLUSTER
            self._cluster_approach_base_distance_m = None
            self._cluster_approach_travel_distance_m = None
            self._breakup_forward_base_distance_m = None
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "cluster_aligned_once_start_approach",
            )
        angular = _clamp(
            self.config.cluster_align_kp_rad_s * heading_error_rad,
            -self.config.cluster_align_max_angular_velocity_rad_s,
            self.config.cluster_align_max_angular_velocity_rad_s,
        )
        return self._decision(timestamp_ns, 0.0, angular, "align_cluster_once")

    def _step_relocate_forward(self, timestamp_ns: int,
                               cumulative_distance_m: float | None) -> MatchDecision:
        command = self._noncontact_motion(timestamp_ns, "relocate",
            target=self.config.cluster_relocate_distance_m, speed=self.config.cluster_relocate_speed_m_s)
        if not command.complete:
            return self._noncontact_decision(timestamp_ns, command)
        self._finish_noncontact()
        self._consecutive_cluster_losses = 0
        self._begin_cluster_search()
        self._breakup_only = False
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(timestamp_ns, 0.0, self._cluster_search_angular_velocity_rad_s,
                              "relocate_complete_resume_search")

    def _step_breakup_forward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        if cumulative_distance_m is None:
            return self._decision(timestamp_ns, 0.0, 0.0, "breakup_forward_waiting_for_odometry")
        if self._breakup_forward_base_distance_m is None:
            self._breakup_forward_base_distance_m = cumulative_distance_m
        travelled = cumulative_distance_m - self._breakup_forward_base_distance_m
        if travelled >= self.config.breakup_forward_distance_m - 1e-9:
            self.state = MatchState.OPEN_GRIPPER_SETTLE
            self._gripper_phase_started_ns = timestamp_ns
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "breakup_forward_limit_reached_open_gripper",
                posture=GripperPosture.OPEN,
            )
        return self._decision(
            timestamp_ns,
            self.config.breakup_forward_speed_m_s,
            0.0,
            "breakup_forward_to_plan_limit",
        )

    def _step_breakup_backward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        if cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "breakup_backward_waiting_for_odometry",
            )
        if self._breakup_backward_base_distance_m is None:
            self._breakup_backward_base_distance_m = cumulative_distance_m
        travelled = (
            self._breakup_backward_base_distance_m - cumulative_distance_m
        )
        if travelled >= self.config.breakup_backward_distance_m - 1e-9:
            self.state = MatchState.CLOSE_GRIPPER_SETTLE
            self._gripper_phase_started_ns = timestamp_ns
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "breakup_backward_limit_reached_close_gripper",
                posture=GripperPosture.CLOSED,
            )
        return self._decision(
            timestamp_ns,
            -self.config.breakup_backward_speed_m_s,
            0.0,
            "breakup_backward_to_plan_limit",
            posture=GripperPosture.OPEN,
        )

    def _step_gripper_settle(
        self,
        timestamp_ns: int,
        next_state: MatchState,
        posture: GripperPosture,
        reason: str,
    ) -> MatchDecision:
        if not self._gripper_action_completed(timestamp_ns):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "waiting_gripper_full_travel",
                posture=posture,
            )
        self.state = next_state
        self._spin_last_heading = None
        self._spin_progress_rad = 0.0
        if next_state is MatchState.BREAKUP_BACKWARD:
            self._breakup_backward_base_distance_m = None
        return self._decision(timestamp_ns, 0.0, 0.0, reason, posture=posture)

    def _gripper_action_completed(self, timestamp_ns: int) -> bool:
        """暂时跳过舵机全行程等待，动作指令下发后立即进入下一步。"""

        # 舵机等待时间暂时注释，保留原实现便于恢复：
        # started = self._gripper_phase_started_ns
        # return (
        #     started is not None
        #     and timestamp_ns - started >= self._gripper_full_travel_time_ns
        # )
        del timestamp_ns
        return True

    def _step_spin(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
        angular_velocity: float,
        posture: GripperPosture,
        next_state: MatchState,
        reason: str,
    ) -> MatchDecision:
        if heading_rad is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "spin_waiting_for_heading",
                posture=posture,
            )
        heading = heading_rad
        previous = self._spin_last_heading
        self._spin_last_heading = heading
        if previous is not None:
            self._spin_progress_rad += self._directional_delta(
                previous,
                heading,
                angular_velocity,
            )
        if self._spin_progress_rad >= self.config.spin_angle_rad:
            self.state = next_state
            if next_state is MatchState.CLOSE_GRIPPER_SETTLE:
                self._gripper_phase_started_ns = timestamp_ns
            return self._decision(timestamp_ns, 0.0, 0.0, reason, posture=posture)
        return self._decision(
            timestamp_ns,
            0.0,
            angular_velocity,
            f"{posture.value}_spin_one_turn",
            posture=posture,
        )

    def _step_check_green(self, timestamp_ns: int) -> MatchDecision:
        target = self._find_isolated_green(timestamp_ns)
        if target is None:
            self._selected_track_id = None
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(
                timestamp_ns,
                0.0,
                self._cluster_search_angular_velocity_rad_s,
                "no_isolated_green_repeat_breakup",
            )
        return self._begin_green_transport(timestamp_ns, target)







    def _update_tracker(
        self,
        timestamp_ns: int,
        perception: PerceptionSnapshot | None,
    ) -> None:
        if perception is None or not self._fresh_perception(perception, timestamp_ns):
            return
        if (
            self._last_tracker_frame_sequence is not None
            and perception.frame_sequence <= self._last_tracker_frame_sequence
        ):
            return
        tracks = self._tracker.update(
            perception.capture_timestamp_ns,
            perception.observations,
        )
        self._preview_target_history[
            (perception.frame_sequence, perception.capture_timestamp_ns)
        ] = tracks
        while len(self._preview_target_history) > 16:
            self._preview_target_history.pop(next(iter(self._preview_target_history)))
        self._last_tracker_frame_sequence = perception.frame_sequence

    def _reset_tracker_for_new_preview_epoch(self) -> None:
        """重置跟踪器时同步丢弃可能复用帧号的预览历史。"""

        self._tracker.reset()
        self._preview_target_history.clear()


    def _breakup_targets(self, timestamp_ns: int) -> tuple[BreakupTarget, ...]:
        geometry = self._breakup_target_geometry
        perception = self._latest_perception
        if geometry is None or perception is None or not self._fresh_perception(perception, timestamp_ns):
            return ()
        targets = []
        for target in self._tracker.tracks:
            if (target.ground_point is None or not target.ever_confirmed
                    or target.frame_sequence != perception.frame_sequence
                    or target.last_seen_timestamp_ns != perception.capture_timestamp_ns
                    or not self._target_is_fresh(target, timestamp_ns)):
                continue
            contact, safety = physical_radii(geometry.geometry_for(target.target_class))
            point = self._current_ground_point_for_track(target, timestamp_ns)
            if point is None:
                # Missing motion history cannot turn an old ray into current geometry,
                # nor silently remove a dangerous member from the full path check.
                return ()
            targets.append(BreakupTarget(target.track_id, target.last_seen_timestamp_ns,
                                         target.target_class, point, contact, safety))
        return tuple(targets)

    def _aim_in_failed_aims(self, aim: FieldPoint) -> bool:
        """只封锁真正失败过的瞄准点。

        旧实现拿失败计划的整组成员去和候选做多数匹配，一次确认超时就把
        整片区域判死；现场 20260912_0053 里一个失败点让 100 mm 内所有团
        连续 20 秒无法解团。
        """

        current_position = self.estimated_field_position
        positions = self._breakup_failed_aim_positions
        positions_aligned = len(positions) == len(self._breakup_failed_aims)
        for index, failed in enumerate(self._breakup_failed_aims):
            if math.hypot(aim.x - failed.x, aim.y - failed.y) > self.config.cluster_group_ground_mm:
                continue
            # An explicitly recorded relocation is the only way to make a
            # failed physical aim eligible again.  In particular, rotating in
            # place and starting a new session keep the failure active.
            failed_position = positions[index] if positions_aligned else None
            if (
                failed_position is not None
                and current_position is not None
                and math.hypot(
                    current_position.x - failed_position.x,
                    current_position.y - failed_position.y,
                ) >= 100.0
            ):
                continue
            return True
        return False

    @staticmethod
    def _boxes_overlap(a: UndistortedBoundingBox, b: UndistortedBoundingBox) -> bool:
        return (a.x_min <= b.x_max and b.x_min <= a.x_max
                and a.y_min <= b.y_max and b.y_min <= a.y_max)

    def _target_overlaps_safe_zone_bbox(
        self,
        box: UndistortedBoundingBox | None,
        perception: PerceptionSnapshot | None = None,
    ) -> bool:
        """图像上目标框与任一安全区框重叠即视为已交付物资。

        交付线（场地 y=1115）、静态地图多边形（y≥1200）和实测地标（y=1137）
        三个数值不一致，靠场地位姿反推会把刚推进安全区的物块重新当成候选，
        且位姿误差会让它更不可靠；图像重叠只看"物块是否压在安全区上"，
        不依赖位姿推算。没有安全区框（未启用颜色识别或无该帧特征）时不排除。
        """

        if box is None:
            return False
        perception = self._latest_perception if perception is None else perception
        features = None if perception is None else perception.field_features
        if features is None:
            return False
        return any(
            self._boxes_overlap(box, zone.box)
            for zone in features.safe_zones
        )

    def _track_box(self, track_id: int) -> UndistortedBoundingBox | None:
        return next(
            (target.box for target in self._tracker.tracks
             if target.track_id == track_id),
            None,
        )

    def _choose_breakup_plan(self, timestamp_ns: int, *, approach: bool = False,
                             required_ids: frozenset[int] | None = None,
                             required_aim_id: int | None = None) -> BreakupPlan | None:
        origin, heading = self.estimated_field_position, self._latest_heading_rad
        if origin is None or heading is None or self._breakup_static_map is None or self._breakup_target_geometry is None:
            self._last_cluster_rejection_reason = "breakup_missing_pose_map_or_geometry"
            return None
        targets = self._breakup_targets(timestamp_ns)
        # 已放入安全区的物资不再作为成组或瞄准点候选；它们仍留在 targets 中
        # 参加推移净空检查，不能被碰撞也不能被忽略。
        non_contact_ids = frozenset(
            target.track_id
            for target in targets
            if self._ground_in_safe_zone(target.center, timestamp_ns)
            or self._target_overlaps_safe_zone_bbox(self._track_box(target.track_id))
        )
        priority_ids: set[int] = set()
        for reason in self._near_field_route_rejections:
            if reason.startswith("blocked_target:"):
                parts = reason.split(":", 2)
                if len(parts) > 1 and parts[1].isdigit():
                    priority_ids.add(int(parts[1]))
            for token in reason.split(":"):
                if token.startswith("track=") and token[6:].isdigit():
                    priority_ids.add(int(token[6:]))
        plan_config = self.config
        if self._misgrasp_breakup_active:
            # The release footprint is fixed in the field before reversing. It
            # selects only this cargo, even after tracker IDs or ranks change.
            released_ids = frozenset(t.track_id for t in targets
                                    if self._target_in_misgrasp_release(t))
            non_contact_ids |= frozenset(t.track_id for t in targets
                                        if t.track_id not in released_ids)
            plan_config = replace(
                self.config,
                breakup_forward_distance_m=self.config.misgrasp_breakup_forward_distance_m,
                breakup_backward_distance_m=self.config.misgrasp_breakup_backward_distance_m,
            )
        args = dict(config=plan_config, origin=origin, heading_rad=heading,
                    static_map=self._breakup_static_map, field_bounds=self._physical_field_bounds(),
                    front_mm=GripperKinematics().left_tip_position(0).x,
                    allowed_classes=self.near_field_policy.allowed_classes,
                    approach=approach, required_ids=required_ids,
                    rejection_reasons=self._near_field_route_rejections,
                    priority_ids=frozenset(priority_ids), required_aim_id=required_aim_id,
                    non_contact_ids=non_contact_ids)
        rejections: list[str] = []
        candidates = plan_breakup(targets, rejections=rejections, **args)
        if self._misgrasp_breakup_active:
            candidates = tuple(p for p in candidates if self._misgrasp_breakup_heading_allowed(p))
        if self.state is MatchState.BREAKUP_SETTLE and self._breakup_proposal is not None:
            anchor = self._breakup_anchor or self._breakup_proposal.aim_field
            # Preserve the physical ray when peripheral members change its rank.
            candidates = tuple(sorted(candidates, key=lambda plan: math.hypot(
                plan.aim_field.x-anchor.x, plan.aim_field.y-anchor.y)))
        tried_groups: set[tuple[int, ...]] = set()
        for candidate in candidates:
            if self._aim_in_failed_aims(candidate.aim_field):
                rejections.append(
                    f"aim={candidate.aim_id} reason=group_in_failed_region "
                    f"members={candidate.member_ids}"
                )
                continue
            if candidate.member_ids in tried_groups:
                continue
            tried_groups.add(candidate.member_ids)
            history = [
                plan
                for index, plan in enumerate(self._breakup_attempts)
                if math.hypot(
                    plan.aim_field.x - candidate.aim_field.x,
                    plan.aim_field.y - candidate.aim_field.y,
                ) <= self.config.cluster_group_ground_mm
                and self._breakup_attempt_is_current(index)
            ]
            if len(history) >= self.config.breakup_max_attempts:
                rejections.append(
                    f"aim={candidate.aim_id} reason=group_attempts_exhausted "
                    f"members={candidate.member_ids} attempts={len(history)}"
                )
                continue
            if history:
                previous = history[-1]
                retries = plan_breakup(targets, **{**args, 'required_ids': frozenset(candidate.member_ids)},
                                       attempt=len(history)+1, previous_aim=previous.aim_field,
                                       previous_penetration_mm=previous.penetration_mm,
                                       rejections=rejections)
                if self._misgrasp_breakup_active:
                    retries = tuple(p for p in retries if self._misgrasp_breakup_heading_allowed(p))
                if retries:
                    self._breakup_plan_rejections = tuple(rejections[-8:])
                    return retries[0]
            else:
                self._breakup_plan_rejections = tuple(rejections[-8:])
                return candidate
        self._breakup_plan_rejections = tuple(rejections[-8:])
        self._last_cluster_rejection_reason = "breakup_no_contact_plan_or_retry_exhausted"
        return None

    def _breakup_attempt_is_current(self, index: int) -> bool:
        """Return whether an old completed push still has the same geometry."""

        if len(self._breakup_attempt_positions) != len(self._breakup_attempts):
            # Tests and legacy pure-logic callers may seed only plan history;
            # without a pose there is no evidence of a valid relocation.
            return True
        recorded = self._breakup_attempt_positions[index]
        current = self.estimated_field_position
        if recorded is None or current is None:
            return True
        return math.hypot(
            current.x - recorded.x,
            current.y - recorded.y,
        ) < 100.0

    def _try_dynamic_grasp(self, timestamp_ns: int) -> MatchDecision | None:
        if not self.config.opportunistic_single_green_enabled:
            return None
        preparation = self._breakup_grasp_preparation
        if self._breakup_preparation_valid(timestamp_ns):
            assert preparation is not None
            # preview_plan includes REJECTED geometry and is never execution evidence.
            plan = preparation.selection.plan
            if plan is not None and self._near_field_path_clear(plan, self._latest_heading_rad) is True:
                target = next((t for t in self._tracker.tracks if t.ground_point is not None and any(
                    m.observation.ground_point is not None
                    and m.observation.target_class is t.target_class
                    and math.hypot(t.ground_point.x-m.observation.ground_point.x,
                                   t.ground_point.y-m.observation.ground_point.y) < self.config.cluster_group_ground_mm
                    for m in plan.members)), None)
                if target is not None:
                    self._breakup_rejected_grasp_ids.discard(target.track_id)
                    self._breakup_only = False
                    return self._begin_green_transport(timestamp_ns, target, group_preview=True)
        target = self._find_approach_seed(timestamp_ns)
        if target is None or target.track_id in self._breakup_rejected_grasp_ids:
            return None
        self._breakup_only = False
        direct = self._transport_group_size(target, timestamp_ns) is not None
        return self._begin_green_transport(timestamp_ns, target, opportunistic=direct,
                                           group_preview=self._near_field_pickup is not None and not direct)

    def _breakup_geometric_block(self, timestamp_ns: int) -> bool:
        if self._breakup_only:
            # 即使已明确进入解团模式，也要等当前帧出现可靠的已确认 K0，
            # 不能用“需要解团”本身替代接触核心证据。
            return any(
                self._target_is_fresh(target, timestamp_ns)
                and self._graspable_target_is_usable(target)
                and target.target_class in self.near_field_policy.allowed_classes
                for target in self._tracker.tracks
            )
        for target in self._tracker.tracks:
            if (not self._target_is_fresh(target, timestamp_ns)
                    or not self._graspable_target_is_usable(target)
                    or target.target_class not in self.near_field_policy.allowed_classes
                    or target.ground_point is None):
                continue
            if self._candidate_path_blocked(target.ground_point, breakup=False):
                continue
            if (self._transport_side_neighbor_target(target, timestamp_ns) is not None
                    or not self._transport_orange_isolation_clear(target, timestamp_ns)
                    or not self._green_path_is_clear_for_point(
                        target, target.ground_point, timestamp_ns,
                        ignored_track_ids=self._preview_ignored_track_ids(target, timestamp_ns))):
                return True
        return False

    def _start_breakup_observation(self, timestamp_ns: int) -> None:
        self._near_field_session_id += 1
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
        self._near_field_handoff_prior = None
        self._breakup_grasp_preparation = None
        self._breakup_phase_started_ns = timestamp_ns
        self._breakup_stopped_ns = None
        self._breakup_reference_frames.clear()
        self._breakup_last_capture_ns = -1
        self._breakup_anchor = None
        self._breakup_proposal = None
        self._breakup_frozen_plan = None
        self._cluster_selected_track_ids = ()
        self._breakup_reference_current = False
        self._breakup_observation_deadline_ns = None
        self._breakup_wait_detail = "waiting_stationarity"
        self._breakup_last_failure_diagnostic = None
        self._breakup_plan_rejections = ()
        self._breakup_confirm_pose = None
        self._breakup_no_plan_since_ns: int | None = None

    def _start_breakup_attempt(
        self,
        timestamp_ns: int,
        plan: BreakupPlan | None = None,
    ) -> None:
        task = self._grasp_task
        if task is None and plan is not None and self.near_field_enabled:
            target = next((item for item in self._tracker.tracks if item.track_id == plan.aim_id), None)
            self._adopt_grasp_task(timestamp_ns, track_id=plan.aim_id,
                                  target_class=None if target is None else target.target_class,
                                  point=plan.aim, field=plan.aim_field)
            task = self._grasp_task
        if task is not None:
            task.blocking = self._near_field_route_rejections
            task.recovery_aim = None if plan is None else plan.aim_field
        self._start_breakup_observation(timestamp_ns)
        self._breakup_proposal = plan
        self._breakup_alignment_done = plan is None
        self._breakup_alignment_allowance_ns = (0 if plan is None or self._latest_heading_rad is None else
            round(self._breakup_alignment_time_s(plan) * 1e9))
        self._cluster_selected_track_ids = () if plan is None else plan.member_ids
        # 先按采集位姿补偿后的接触射线对准，再停稳确认并连续推进。
        self.state = MatchState.BREAKUP_SETTLE

    def _breakup_alignment_tolerance(self, plan: BreakupPlan) -> float:
        tolerance = self.config.cluster_align_tolerance_rad
        geometry = self._breakup_target_geometry
        target = next((target for target in self._tracker.tracks
                       if target.track_id == plan.aim_id), None)
        if geometry is not None and target is not None:
            radius, _ = physical_radii(geometry.geometry_for(target.target_class))
            # A fixed angular tolerance can miss a small distant contact disk.
            tolerance = min(tolerance, math.atan2(radius / 2, math.hypot(plan.aim.x, plan.aim.y)))
        return tolerance

    def _breakup_alignment_time_s(self, plan: BreakupPlan) -> float:
        error = abs(normalize_angle(plan.heading_rad - self._latest_heading_rad))
        maximum = self.config.cluster_align_max_angular_velocity_rad_s
        minimum = min(0.08, maximum)
        tolerance = self._breakup_alignment_tolerance(plan)
        # Integrate the actual saturated proportional angular command, including
        # its slow final segment; settle/observation retain their own budgets.
        return (max(0.0, error-maximum)/maximum
                + max(0.0, math.log(max(min(error, maximum), minimum)
                                    / max(tolerance, minimum)))
                + max(0.0, min(error, minimum)-tolerance)/minimum + 0.4)

    def _observe_breakup_feedback(self, timestamp_ns: int) -> None:
        """Pure-step callers can supply speed/odometry; hardware always uses encoder/IMU evidence."""
        if self._stationary_motion.latest is not None:
            return
        pose = (self._latest_cumulative_distance_m, self._latest_heading_rad)
        speeds = self._latest_speed_feedback
        if (any(value is None or not math.isfinite(value) for value in (*pose, *speeds))
                or max(abs(value) for value in speeds if value is not None)
                > self.config.safe_zone_calibration_stop_speed_threshold_m_s
                or (self._breakup_feedback_pose is not None and pose != self._breakup_feedback_pose)):
            self._breakup_feedback_since_ns = None
        elif self._breakup_feedback_since_ns is None:
            self._breakup_feedback_since_ns = timestamp_ns
        self._breakup_feedback_pose = pose

    def _breakup_stationary_since(self, timestamp_ns: int) -> int | None:
        since = (self._stationary_motion.stationary_since(timestamp_ns)
                 if self._stationary_motion.latest is not None else self._breakup_feedback_since_ns)
        if since is None or timestamp_ns-since < self.config.safe_zone_calibration_stop_confirm_time_s*1e9:
            return None
        return since

    def _breakup_capture_valid(self, capture_ns: int, timestamp_ns: int) -> bool:
        if self._stationary_motion.latest is not None:
            return self._stationary_motion.capture_valid(capture_ns, timestamp_ns,
                                                         max_age_ns=round(self.config.green_max_age_ms*1e6))
        since = self._breakup_stationary_since(timestamp_ns)
        return since is not None and since <= capture_ns <= timestamp_ns and timestamp_ns-capture_ns <= self.config.green_max_age_ms*1e6

    def _breakup_preparation_valid(self, timestamp_ns: int) -> bool:
        preparation = self._breakup_grasp_preparation
        if preparation is None or preparation.session_id != self._near_field_session_id:
            return False
        age = preparation.preparation_age_ns(timestamp_ns)
        return (age is not None and age <= self._stationary_motion.max_gap_ns
                and self._breakup_capture_valid(preparation.capture_timestamp_ns, timestamp_ns))

    def _breakup_new_stopped_frame(self, timestamp_ns: int) -> bool:
        # 只有机器人自身位姿确实变化才作废既有确认。静止区间会因为设备样本
        # 重复或质量位抖动重新落定，但那不是核心变化；冻结前仍会用当前静止
        # 区间重新校验候选帧的采集时间，所以这里不需要为安全多清一次。
        pose = (self._latest_cumulative_distance_m, self._latest_heading_rad)
        if self._breakup_confirm_pose is not None and (
            (pose[0] is not None and self._breakup_confirm_pose[0] is not None
             and abs(pose[0] - self._breakup_confirm_pose[0]) >= 0.01)
            or (pose[1] is not None and self._breakup_confirm_pose[1] is not None
                and abs(normalize_angle(pose[1] - self._breakup_confirm_pose[1])) >= 0.01)
        ):
            self._breakup_reference_frames.clear()
            self._breakup_reference_current = False
            self._breakup_anchor = None
            self._cluster_selected_track_ids = ()
            self._breakup_confirm_pose = None
        since = self._breakup_stationary_since(timestamp_ns)
        if since is None:
            # 遥测断档或静止证据尚未重新连续，只说明当前帧不能用于冻结。
            self._breakup_stopped_ns = None
            self._breakup_wait_detail = "waiting_stationarity_or_telemetry"
            return False
        # 静止区间重新落定只更新起点，不清确认。
        self._breakup_stopped_ns = since
        # Time budget starts when actual stationary evidence is available. Frame
        # delivery latency gets a bounded allowance, never an artificial new stop.
        # 参考确认需要 breakup_confirmation_frames 个不同采集，按每帧允许的
        # 观测年龄逐帧预算；只加单帧余量会让多帧确认在慢感知下必然超时。
        if self._breakup_observation_deadline_ns is None:
            self._breakup_no_plan_since_ns = timestamp_ns
            self._breakup_observation_deadline_ns = (
                timestamp_ns
                + self._cluster_align_hold_ns
                + self.config.breakup_confirmation_frames
                * round(self.config.green_max_age_ms * 1e6)
            )
        perception = self._latest_perception
        if perception is None:
            self._breakup_wait_detail = "waiting_perception_delivery"
            return False
        if not self._fresh_perception(perception, timestamp_ns):
            self._breakup_wait_detail = "observation_expired"
            return False
        if not self._breakup_capture_valid(perception.capture_timestamp_ns, timestamp_ns):
            self._breakup_wait_detail = "capture_outside_stationary_interval"
            return False
        key = (perception.frame_sequence, perception.capture_timestamp_ns)
        if perception.capture_timestamp_ns <= self._breakup_last_capture_ns or any(frame == key[0] for frame, _ in self._breakup_reference_frames):
            self._breakup_wait_detail = "waiting_distinct_capture"
            return False
        self._breakup_last_capture_ns = perception.capture_timestamp_ns
        if self._breakup_confirm_pose is None:
            # 记录首帧确认时的机器人位姿：此后的确认只在位姿未变时继续有效。
            self._breakup_confirm_pose = (
                self._latest_cumulative_distance_m, self._latest_heading_rad,
            )
        self._breakup_wait_detail = "new_stationary_capture"
        return True

    def _breakup_observation_expired(self, timestamp_ns: int) -> bool:
        if self._breakup_observation_deadline_ns is not None:
            return timestamp_ns >= self._breakup_observation_deadline_ns
        return (self._breakup_phase_started_ns is not None
                and timestamp_ns-self._breakup_phase_started_ns >= self._cluster_align_hold_ns + getattr(self, "_breakup_alignment_allowance_ns", 0))

    def _resume_dynamic_search(self, timestamp_ns: int, reason: str) -> MatchDecision:
        plan = self._breakup_proposal or self._breakup_plan
        perception = self._latest_perception
        observation_age = (
            "none"
            if perception is None
            else f"{max(0, timestamp_ns - perception.capture_timestamp_ns) / 1_000_000.0:.1f}"
        )
        if plan is not None and ("timeout" in reason or "rejection" in reason):
            self._breakup_failed_aims.append(plan.aim_field)
            self._breakup_failed_aim_positions.append(self.estimated_field_position)
            self._breakup_failed_aims = self._breakup_failed_aims[-32:]
            self._breakup_failed_aim_positions = self._breakup_failed_aim_positions[-32:]
            self._breakup_last_failure_diagnostic = (
                "breakup_failure "
                f"aim_id={plan.aim_id} members={plan.member_ids} "
                f"contact={plan.contact_ids} attempt={plan.attempt} "
                f"reason={reason} confirmation={len(self._breakup_reference_frames)}/"
                f"{self.config.breakup_confirmation_frames} "
                f"observation_age_ms={observation_age} "
                f"deadline_ns={self._breakup_observation_deadline_ns} "
                f"rejected_candidates=[{';'.join(self._breakup_plan_rejections)}]"
            )
        elif "timeout" in reason or "rejection" in reason:
            self._breakup_last_failure_diagnostic = (
                "breakup_failure aim_id=none members=none "
                f"reason={reason} confirmation={len(self._breakup_reference_frames)}/"
                f"{self.config.breakup_confirmation_frames} "
                f"observation_age_ms={observation_age} "
                f"deadline_ns={self._breakup_observation_deadline_ns} "
                f"rejected_candidates=[{';'.join(self._breakup_plan_rejections)}]"
            )
        if self._grasp_task is not None:
            self._remember_near_field_failure(timestamp_ns, reason)
            self._grasp_task = None
        self._begin_cluster_search()
        self._breakup_only = False
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(timestamp_ns, 0.0, 0.0, reason, posture=GripperPosture.CLOSED, soft_brake=True)

    def _step_dynamic_cluster_search(self, timestamp_ns: int, heading_rad: float | None) -> MatchDecision:
        if self._greedy_active:
            return self._finish_greedy_pickup(timestamp_ns, "no_direct_plan")
        if self._transport_target_classes:
            return self._start_safe_zone_transport(timestamp_ns, transport_opened=False,
                                                  posture=GripperPosture.CLOSED, reason="loaded_skip_breakup")
        grasp = self._try_dynamic_grasp(timestamp_ns)
        if grasp is not None:
            return grasp
        self._update_cluster_search_velocity()
        perception = self._latest_perception
        key = None if perception is None else (perception.frame_sequence, perception.capture_timestamp_ns)
        is_new = key is not None and key != self._breakup_search_frame
        if is_new:
            self._breakup_search_frame = key
        committed = self._advance_cluster_search_sweep(timestamp_ns, heading_rad)
        if committed is not None:
            return committed
        if is_new and self._breakup_geometric_block(timestamp_ns) and self.near_field_enabled:
            target = next((item for item in self._tracker.tracks
                           if self._target_is_fresh(item, timestamp_ns)
                           and self._graspable_target_is_usable(item)
                           and item.target_class in self.near_field_policy.allowed_classes
                           and item.ground_point is not None
                           and (current := self._current_ground_point_for_track(item, timestamp_ns)) is not None
                           and math.hypot(current.x, current.y) <= self._near_field_handoff_range_mm()
                           and not self._target_attempt_blocked(item, timestamp_ns)), None)
            if target is not None:
                point = self._current_ground_point_for_track(target, timestamp_ns)
                if point is not None:
                    self._adopt_grasp_task(timestamp_ns, track_id=target.track_id,
                                          target_class=target.target_class, point=point,
                                          field=self._field_point_for_target(target))
                    return self._begin_near_field_grasp(timestamp_ns, handoff_prior=
                        NearFieldHandoffPrior(target.target_class, point, target.track_id))
        if is_new and self._breakup_geometric_block(timestamp_ns) and not self.near_field_enabled:
            # 先确认当前区域确实有尚未失败的接触计划，再停车；计划本身
            # 只会在停稳后的当前帧重新生成和检查。
            plan = self._choose_breakup_plan(timestamp_ns, approach=False)
            if plan is not None:
                # 该计划只用于记录本次待确认区域；执行计划仍必须来自停稳后的当前帧。
                self._start_breakup_attempt(timestamp_ns, plan)
                return self._decision(timestamp_ns, 0.0, 0.0, "cluster_seen_stop_collect_reference", soft_brake=True)
        return self._decision(timestamp_ns, 0.0, self._cluster_search_angular_velocity_rad_s,
                              "search_cluster_right" if self._cluster_search_angular_velocity_rad_s < 0 else "search_cluster_left")

    def _collect_dynamic_breakup(self, timestamp_ns: int) -> MatchDecision | None:
        if self._breakup_observation_expired(timestamp_ns):
            return self._resume_dynamic_search(timestamp_ns, "breakup_reference_timeout_reselect")
        proposal = self._breakup_proposal
        if proposal is not None and not getattr(self, "_breakup_alignment_done", True):
            if self._latest_heading_rad is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "breakup_alignment_missing_heading", soft_brake=True)
            error = normalize_angle(proposal.heading_rad - self._latest_heading_rad)
            if abs(error) > self._breakup_alignment_tolerance(proposal):
                maximum = self.config.cluster_align_max_angular_velocity_rad_s
                return self._decision(timestamp_ns, 0.0, math.copysign(min(maximum, max(0.08, abs(error))), error),
                                      "breakup_align_contact_imu", posture=GripperPosture.CLOSED)
            self._breakup_alignment_done = True
        if self._breakup_frozen_plan is not None:
            # 近场已经确认过"抓不了、必须推开"并给出了接触计划；对准转向只是
            # 修正接触射线，不能让计划失效。回到近场重新观察会在
            # "近场判定必须解团 → 对准 → 弃团 → 近场再判定必须解团"之间死循环，
            # 而每次循环都把车转离已对准的朝向。直接冻结计划并推进。
            self._breakup_plan = self._breakup_frozen_plan
            self._breakup_no_plan_since_ns = None
            self._breakup_wait_detail = "frozen_contact_plan"
            return None
        if self._grasp_task is not None and self.near_field_enabled:
            # The turn changed camera geometry but did not end the grasp task.
            # 交接 prior 必须跟着走：否则这一转既丢了朝向也丢了目标，近场只能
            # 从零重扫。
            return self._begin_near_field_grasp(
                timestamp_ns,
                handoff_prior=(self._near_field_handoff_prior
                               or self._selected_handoff_prior(timestamp_ns)),
            )
        new_capture = self._breakup_new_stopped_frame(timestamp_ns)
        if new_capture:
            self._breakup_reference_current = False
            # Pin the physical impact core, not the entire connected component:
            # peripheral omissions and tracker ID changes must not erase evidence.
            candidate = self._choose_breakup_plan(
                timestamp_ns,
                approach=False,
            )
            if (candidate is not None and self._breakup_anchor is not None
                    and math.hypot(candidate.aim_field.x-self._breakup_anchor.x,
                                   candidate.aim_field.y-self._breakup_anchor.y)
                    > self.config.cluster_group_ground_mm):
                candidate = None
            if candidate is not None:
                self._breakup_no_plan_since_ns = None
                error = normalize_angle(candidate.heading_rad - self._latest_heading_rad)
                if abs(error) > self._breakup_alignment_tolerance(candidate):
                    # The stopped frame may refine the initial moving-frame ray.
                    # Align before driving; do not bend a supposedly straight push.
                    self._breakup_proposal = candidate
                    self._breakup_alignment_done = False
                    self._breakup_reference_frames.clear()
                    self._breakup_wait_detail = "contact_heading_requires_alignment"
                    return self._decision(timestamp_ns, 0.0, 0.0,
                        "breakup_reference:contact_heading_requires_alignment", soft_brake=True)
                if self._breakup_anchor is None:
                    self._breakup_anchor = candidate.aim_field
                self._breakup_proposal = candidate
                self._cluster_selected_track_ids = candidate.member_ids
                self._breakup_reference_current = True
                perception = self._latest_perception
                assert perception is not None
                if len(self._breakup_reference_frames) < self.config.breakup_confirmation_frames:
                    self._breakup_reference_frames.add((perception.frame_sequence, perception.capture_timestamp_ns))
            else:
                if self._breakup_no_plan_since_ns is None:
                    self._breakup_no_plan_since_ns = timestamp_ns
                self._breakup_wait_detail = "impact_core_unavailable_or_unsafe"
        # A boundary-arriving valid frame is processed before the no-plan cutoff.
        if (self._breakup_no_plan_since_ns is not None
                and timestamp_ns - self._breakup_no_plan_since_ns
                >= round(self.config.breakup_no_plan_reobserve_ms * 1e6)):
            return self._resume_dynamic_search(timestamp_ns, "breakup_reference_timeout_reselect")
        candidate = self._breakup_proposal
        if len(self._breakup_reference_frames) < self.config.breakup_confirmation_frames:
            return self._decision(timestamp_ns, 0.0, 0.0, f"breakup_reference:{self._breakup_wait_detail}", soft_brake=True)
        if not self._breakup_reference_current or candidate is None or not self._breakup_capture_valid(candidate.capture_timestamp_ns, timestamp_ns):
            return self._decision(timestamp_ns, 0.0, 0.0, "breakup_reference:waiting_current_safe_geometry", soft_brake=True)
        self._breakup_plan = candidate
        return None

    def _dynamic_segment_speed(self, remaining_mm: float, maximum_m_s: float,
                               *, stop_margin_mm: float | None = None) -> float:
        acceleration = self.config.breakup_deceleration_m_s2
        if self.config.breakup_max_linear_deceleration_m_s2 is not None:
            acceleration = min(
                acceleration,
                self.config.breakup_max_linear_deceleration_m_s2,
            )
        margin = self.config.breakup_braking_margin_mm if stop_margin_mm is None else stop_margin_mm
        distance_m = max(0.0, remaining_mm-margin)/1000.0
        return min(maximum_m_s, math.sqrt(2*acceleration*distance_m))

    def _step_dynamic_breakup(self, timestamp_ns: int, distance_m: float | None) -> MatchDecision | None:
        state = self.state
        if state not in {MatchState.BREAKUP_SETTLE,
                         MatchState.BREAKUP_FORWARD, MatchState.BREAKUP_BACKWARD,
                         MatchState.OPEN_GRIPPER_SETTLE, MatchState.CLOSE_GRIPPER_SETTLE,
                         MatchState.CLOSE_GRIPPER_SPIN, MatchState.CHECK_ISOLATED_GREEN}:
            return None
        if self._greedy_active:
            return self._finish_greedy_pickup(timestamp_ns, "no_direct_plan")
        if self._transport_target_classes:
            return self._start_safe_zone_transport(timestamp_ns, transport_opened=False,
                                                  posture=GripperPosture.CLOSED, reason="loaded_skip_breakup")
        if state is MatchState.CHECK_ISOLATED_GREEN and self._grasp_task is not None:
            return self._reacquire_breakup_target(timestamp_ns)
        if state is MatchState.BREAKUP_SETTLE:
            waiting = self._collect_dynamic_breakup(timestamp_ns)
            if waiting is not None:
                return waiting
            plan = self._breakup_plan
            assert plan is not None
            if distance_m is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "breakup_commit_missing_control_state", soft_brake=True)
            if self._misgrasp_breakup_active and timestamp_ns < self._settle_until_ns:
                return self._decision(timestamp_ns, 0.0, 0.0,
                    f"misgrasp_breakup_wait_closed:deadline_ns={self._settle_until_ns}",
                    posture=GripperPosture.CLOSED, soft_brake=True)
            self._breakup_forward_base_distance_m = distance_m
            self._breakup_segment_stop_started_ns = None
            self._breakup_segment_stop_deadline_ns = None
            self._breakup_retreat_mm = plan.backward_distance_mm
            self._breakup_actual_forward_mm = self._breakup_actual_backward_mm = 0.0
            self._safe_zone_stop_since_ns = None
            self.state = MatchState.BREAKUP_FORWARD
            return self._decision(timestamp_ns, 0.0, 0.0, "breakup_plan_frozen", soft_brake=True)
        if state in {MatchState.BREAKUP_FORWARD, MatchState.BREAKUP_BACKWARD}:
            plan = self._breakup_plan
            forward = state is MatchState.BREAKUP_FORWARD
            base = self._breakup_forward_base_distance_m if forward else self._breakup_backward_base_distance_m
            posture = GripperPosture.CLOSED
            deadline = self._breakup_segment_stop_deadline_ns
            if (deadline is not None and timestamp_ns >= deadline
                    and not self._safe_zone_vehicle_stopped(timestamp_ns)):
                # Continued motion or missing critical telemetry after braking
                # means control is not established.  Keep braking, but remain
                # recoverable when valid stationary evidence returns.
                reason = (f"breakup_stop_unconfirmed:segment={'forward' if forward else 'backward'},"
                          f"distance_m={distance_m},base_distance_m={base},"
                          f"stop_started_ns={self._breakup_segment_stop_started_ns},deadline_ns={deadline},"
                          f"{self._stationary_motion.diagnostic(timestamp_ns)}")
                self._breakup_last_failure_diagnostic = reason
                return self._decision(timestamp_ns, 0.0, 0.0, reason, posture=posture, soft_brake=True)
            if plan is None or base is None or distance_m is None or self._latest_heading_rad is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "breakup_safe_zone_guard_missing_pose_or_map", posture=posture, soft_brake=True)
            travel = max(0.0, (distance_m-base)*1000*(1 if forward else -1))
            if forward:
                self._breakup_actual_forward_mm = travel
            else:
                self._breakup_actual_backward_mm = travel
            remaining = (plan.forward_distance_mm if forward else self._breakup_retreat_mm)-travel
            # Forward distance already includes the physical penetration. Braking
            # clearance belongs outside that endpoint, not inside the material.
            stop_margin_mm = 1.0
            if remaining > stop_margin_mm and self._breakup_segment_stop_started_ns is None:
                self._safe_zone_stop_since_ns = None
                maximum = self.config.breakup_forward_speed_m_s if forward else self.config.breakup_backward_speed_m_s
                return self._decision(timestamp_ns, (1 if forward else -1)*self._dynamic_segment_speed(
                                          remaining, maximum, stop_margin_mm=stop_margin_mm),
                                      self._breakup_heading_correction(plan), "breakup_forward_dynamic" if forward else "breakup_backward_dynamic",
                                      posture=posture, min_wheel_velocity_m_s=0.0)
            if self._breakup_segment_stop_started_ns is None:
                # Reaching the endpoint is irreversible for this segment. A
                # rebound, rollback or external displacement cannot resume it.
                self._breakup_segment_stop_started_ns = timestamp_ns
                maximum = self.config.breakup_forward_speed_m_s if forward else self.config.breakup_backward_speed_m_s
                acceleration = self.config.breakup_deceleration_m_s2
                if self.config.breakup_max_linear_deceleration_m_s2 is not None:
                    acceleration = min(
                        acceleration,
                        self.config.breakup_max_linear_deceleration_m_s2,
                    )
                budget_s = (maximum / acceleration
                            + self.config.safe_zone_calibration_stop_confirm_time_s
                            + 2 * self._stationary_motion.max_gap_ns / 1e9)
                self._breakup_segment_stop_deadline_ns = timestamp_ns + round(budget_s * 1e9)
            deadline = self._breakup_segment_stop_deadline_ns
            assert deadline is not None
            if (self._stationary_motion.latest is not None
                    and self._stationary_motion.stationary_since(timestamp_ns) is None):
                return self._decision(timestamp_ns, 0.0, 0.0,
                    f"breakup_segment_stop:deadline_ns={deadline},remaining_mm={remaining:.1f},"
                    f"{self._stationary_motion.diagnostic(timestamp_ns)}",
                    posture=posture, soft_brake=True)
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(timestamp_ns, 0.0, 0.0,
                    f"breakup_waiting_segment_stop:stop_started_ns={self._breakup_segment_stop_started_ns},"
                    f"deadline_ns={deadline},speed_feedback={self._latest_speed_feedback}",
                    posture=posture, soft_brake=True)
            if forward:
                self._breakup_retreat_mm = plan.backward_distance_mm
                self._breakup_attempts.append(plan)
                self._breakup_attempts = self._breakup_attempts[-32:]
                self._breakup_attempt_positions.append(self.estimated_field_position)
                self._breakup_attempt_positions = self._breakup_attempt_positions[-32:]
                self._breakup_backward_base_distance_m = distance_m
                self._breakup_segment_stop_started_ns = None
                self._breakup_segment_stop_deadline_ns = None
                self._safe_zone_stop_since_ns = None
                self.state = MatchState.BREAKUP_BACKWARD
                return self._decision(timestamp_ns, 0.0, 0.0, "breakup_forward_complete_closed_retreat",
                                      posture=posture, soft_brake=True)
            self._breakup_rejected_grasp_ids.clear()
            self._breakup_only = False
            task = self._grasp_task
            if task is not None and self.near_field_enabled:
                task.recovery_count += 1
                task.scene_revision += 1
                task.core = ()
                task.last_progress_ns = timestamp_ns
                task.scene_floor_ns = timestamp_ns
                task.recovery_reobserve_started_ns = timestamp_ns
                task.recovery_reobserve_deadline_ns = None
                self.state = MatchState.CHECK_ISOLATED_GREEN
                return self._reacquire_breakup_target(timestamp_ns)
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(timestamp_ns, 0.0, self._cluster_search_angular_velocity_rad_s,
                                  "breakup_complete_resume_search", posture=posture, soft_brake=True)
        return None

    def _breakup_heading_correction(self, plan: BreakupPlan) -> float:
        heading = self._latest_heading_rad
        if heading is None:
            return 0.0
        return _clamp(normalize_angle(plan.heading_rad-heading)*self.config.cluster_align_kp_rad_s,
                      -self.config.cluster_align_max_angular_velocity_rad_s,
                      self.config.cluster_align_max_angular_velocity_rad_s)

    def _guard_dynamic_breakup(self, decision: MatchDecision, distance_m: float | None) -> MatchDecision:
        if decision.linear_velocity_m_s == 0:
            return decision
        plan, origin, heading = self._breakup_plan, self.estimated_field_position, self._latest_heading_rad
        if plan is None or distance_m is None or origin is None or heading is None or self._breakup_static_map is None:
            return replace(decision, linear_velocity_m_s=0.0, angular_velocity_rad_s=0.0,
                           soft_brake=True, reason="breakup_safe_zone_guard_missing_pose_or_map")
        if decision.state is MatchState.APPROACH_CLUSTER:
            base = self._cluster_approach_base_distance_m
            remaining = plan.approach_distance_mm + plan.forward_distance_mm - (distance_m-(distance_m if base is None else base))*1000
        elif decision.state is MatchState.BREAKUP_FORWARD:
            base = self._breakup_forward_base_distance_m
            remaining = plan.forward_distance_mm-(distance_m-(distance_m if base is None else base))*1000
        else:
            base = self._breakup_backward_base_distance_m
            remaining = -(self._breakup_retreat_mm-((distance_m if base is None else base)-distance_m)*1000)
        remaining += math.copysign(self.config.breakup_braking_margin_mm, remaining)
        end = FieldPoint(origin.x+remaining*math.cos(heading), origin.y+remaining*math.sin(heading))
        if not segment_clear(self._breakup_static_map, origin, end, self._physical_field_bounds(), self._robot_path_clearance_mm()):
            return self._start_path_reselection(decision.timestamp_ns, posture=decision.gripper_posture,
                                               reason="target_path_intersects_safe_zone_stop_and_reselect")
        return decision

    def _cluster_ground_measurement(
        self,
        timestamp_ns: int,
    ) -> _GroundClusterMeasurement | None:
        """返回目标团中心和最前方正向 K0 地面距离。"""

        if self._dynamic_breakup_enabled:
            # This legacy diagnostic helper is not part of the active simple
            # breakup state machine; keep its historical standoff measurement
            # for callers that still inspect the cluster reference.
            self._breakup_proposal = self._choose_breakup_plan(timestamp_ns, approach=True)
            if self._breakup_proposal is None:
                return None
            plan = self._breakup_proposal
            self._cluster_selected_track_ids = plan.member_ids
            return _GroundClusterMeasurement(plan.aim, math.hypot(plan.aim.x, plan.aim.y))
        self._cluster_selected_track_ids = ()
        # 已放入安全区的物资不再是搜索目标：它仍留在轨迹里继续参与障碍、
        # 危险和解团推移检查，但不能成为被选中去接近/解团的团成员。
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if target.ground_point is not None
            and target.ever_confirmed
            and self._target_is_fresh(target, timestamp_ns)
            and not self._ground_in_safe_zone(
                target.ground_point, target.last_seen_timestamp_ns
            )
            and not self._target_overlaps_safe_zone_bbox(target.box)
        )
        points = [target.ground_point for target in candidates]
        remaining_points = points
        preferred_points = self._preferred_cluster_ground_points(timestamp_ns)
        largest: list[GroundPoint] | None = None
        while remaining_points:
            largest = self._largest_ground_group(
                remaining_points,
                preferred_points=preferred_points,
            )
            if largest is None:
                return None
            selected_points = frozenset(largest)
            selected_classes = frozenset(
                target.target_class
                for target in candidates
                if target.ground_point in selected_points
            )
            if selected_classes != frozenset((TargetClass.BLUE_DANGER,)):
                break
            self._last_cluster_rejection_reason = "all_blue_cluster_ignored"
            remaining_points = [
                point for point in remaining_points
                if point not in selected_points
            ]
            if not remaining_points:
                largest = None
                break
        if largest is None:
            return None
        forward_x = [point.x for point in largest if point.x > 0.0]
        if not forward_x:
            return None
        selected_points = frozenset(largest)
        self._cluster_selected_track_ids = tuple(
            target.track_id
            for target in candidates
            if target.ground_point in selected_points
        )
        return _GroundClusterMeasurement(
            center=GroundPoint(
                sum(point.x for point in largest) / len(largest),
                sum(point.y for point in largest) / len(largest),
            ),
            nearest_forward_x_mm=min(forward_x),
        )

    def _connected_ground_group(
        self,
        points: list[GroundPoint],
        *,
        preferred_points: frozenset[GroundPoint] | None = None,
    ) -> list[GroundPoint] | None:
        if len(points) < self.config.cluster_min_detections:
            return None
        parent = list(range(len(points)))

        def find(index: int) -> int:
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            return index

        threshold = self.config.cluster_group_ground_mm
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                if (
                    math.hypot(
                        points[i].x - points[j].x,
                        points[i].y - points[j].y,
                    )
                    <= threshold
                ):
                    root_i = find(i)
                    root_j = find(j)
                    if root_i != root_j:
                        parent[root_j] = root_i
        groups: dict[int, list[GroundPoint]] = {}
        for index, point in enumerate(points):
            groups.setdefault(find(index), []).append(point)
        preferred = preferred_points or frozenset()
        largest = max(
            groups.values(),
            key=lambda group: (
                any(point in preferred for point in group),
                len(group),
            ),
        )
        if len(largest) < self.config.cluster_min_detections:
            return None
        return largest


    @staticmethod
    def _green_target_is_usable(target: TrackedTarget) -> bool:
        if (
            target.target_class is not TargetClass.GREEN_SUPPLY
            or target.status is not TrackStatus.CONFIRMED
            or not target.ever_confirmed
            or target.ground_point is None
        ):
            return False
        return True

    def _graspable_target_is_usable(self, target: TrackedTarget) -> bool:
        """确认的绿/黑物资或橙色伤员候选。"""
        if (self._ground_in_safe_zone(target.ground_point, target.last_seen_timestamp_ns)
                or self._target_overlaps_safe_zone_bbox(target.box)):
            return False

        if (
            target.target_class not in {
                TargetClass.GREEN_SUPPLY,
                TargetClass.BLACK_CORE,
                TargetClass.ORANGE_INJURED,
            }
            or target.status is not TrackStatus.CONFIRMED
            or not target.ever_confirmed
            or target.ground_point is None
        ):
            return False
        return True

    def _selected_green_is_usable(self, target: TrackedTarget) -> bool:
        """已选目标允许短时 coasting，但类别/质量变化仍不能继续夹取。

        地面点是否可用由调用方单独判断并 coast；这里只看类别与颜色质量。
        """

        allowed_classes = (
            {TargetClass.GREEN_SUPPLY}
            if self._transport_count == 0
            else {
                TargetClass.GREEN_SUPPLY,
                TargetClass.BLACK_CORE,
                TargetClass.ORANGE_INJURED,
            }
        )
        return (
            target.target_class in allowed_classes
            and target.ever_confirmed

        )

    def _selected_target(self) -> TrackedTarget | None:
        if self._selected_track_id is None:
            return None
        return next(
            (
                target
                for target in self._tracker.tracks
                if target.track_id == self._selected_track_id
            ),
            None,
        )

    def _ground_in_safe_zone(self, point: GroundPoint | None, capture_ns: int) -> bool:
        if point is None or self._breakup_static_map is None:
            return False
        now_ns = self._last_timestamp_ns or capture_ns
        current = self._ground_point_at_now(point, capture_ns, now_ns)
        field = self._field_point_from_ground(current if current is not None else point)
        return field is not None and safe_zone_intersection(self._breakup_static_map, field, field, 0.0) is True

    def _observation_is_selectable(
        self,
        item: TargetObservation,
        stowed_limit_mm: float | None,
        perception: PerceptionSnapshot | None = None,
    ) -> bool:
        """判断一条观测是否还能作为新的抓取候选。

        蓝色危险证据永远保留：已放入安全区的物资和夹在爪内的成员仍然必须
        能参与夹爪扫掠检查。补夹扫描期间，爪内成员的地面投影落在扫描本身
        已经要求新物资越过的同一前向界限之内，因此复用该界限排除，不引入
        第二套判定。
        """

        is_danger = (
            item.target_class is TargetClass.BLUE_DANGER
            or item.model_target_class is TargetClass.BLUE_DANGER
        )
        if not is_danger and self._ground_in_safe_zone(item.ground_point, item.capture_timestamp_ns):
            return False
        if not is_danger and self._target_overlaps_safe_zone_bbox(item.box, perception):
            return False
        if is_danger:
            return True
        gate = self._gate_clearance
        if gate is not None and gate.reacquiring:
            pose = self._pose_at(item.capture_timestamp_ns)
            if pose is None or item.ground_point is None or item.target_class not in gate.original_classes:
                return False
            field = self._field_point_from_pose(pose, item.ground_point)
            if math.hypot(field.x-gate.stash.x, field.y-gate.stash.y) > self.config.gate_clearance.reacquire_radius_mm:
                return False
        if stowed_limit_mm is None or item.ground_point is None:
            return True
        now_ns = self._last_timestamp_ns or item.capture_timestamp_ns
        current = self._ground_point_at_now(
            item.ground_point, item.capture_timestamp_ns, now_ns,
        )
        point = current if current is not None else item.ground_point
        return point.x > stowed_limit_mm

    def grasp_excluded_observation_indices(self, perception: PerceptionSnapshot) -> frozenset[int]:
        """Mark delivered/carried candidates without deleting obstacle evidence."""
        if not isinstance(perception, PerceptionSnapshot):
            raise TypeError("perception must be a PerceptionSnapshot.")
        stowed_limit = self._greedy_new_target_min_x_mm() if self._greedy_active else None
        return frozenset(index for index, item in enumerate(perception.observations)
                         if not self._observation_is_selectable(item, stowed_limit, perception))

    def _target_is_fresh(self, target: TrackedTarget, timestamp_ns: int) -> bool:
        return (
            timestamp_ns >= target.last_seen_timestamp_ns
            and not self._target_overlaps_safe_zone_bbox(target.box)
            and (timestamp_ns - target.last_seen_timestamp_ns) / 1_000_000.0
            <= self.config.green_max_age_ms
        )

    def _fresh_perception(
        self,
        perception: PerceptionSnapshot,
        timestamp_ns: int,
    ) -> bool:
        return (
            perception.dropped_stale_age_ms is None
            and perception.capture_timestamp_ns <= timestamp_ns
            and (timestamp_ns - perception.capture_timestamp_ns) / 1_000_000.0
            <= self.config.green_max_age_ms
        )

    def _straight_pid_output(
        self,
        timestamp_ns: int,
        left_speed_feedback_m_s: float | None,
        right_speed_feedback_m_s: float | None,
    ) -> float:
        """根据左右轮反馈生成启动直行的角速度修正。

        误差定义为 ``right - left``；右轮更快时输出负角速度，抵消车辆左偏。
        I/D 默认关闭，现场可在配置中逐步打开；缺少有效反馈时保守输出零。
        """

        if left_speed_feedback_m_s is None or right_speed_feedback_m_s is None:
            self._straight_pid_last_error = None
            self._straight_pid_last_timestamp_ns = None
            return 0.0
        error = right_speed_feedback_m_s - left_speed_feedback_m_s
        effective_error = (
            0.0
            if abs(error) <= self.config.startup_straight_pid_deadband_m_s
            else error
        )
        previous_error = self._straight_pid_last_error
        previous_timestamp_ns = self._straight_pid_last_timestamp_ns
        dt_s = (
            0.0
            if previous_timestamp_ns is None
            else max(0.0, (timestamp_ns - previous_timestamp_ns) / 1_000_000_000.0)
        )
        if dt_s > 0.0:
            self._straight_pid_integral = _clamp(
                self._straight_pid_integral + effective_error * dt_s,
                -self.config.startup_straight_pid_integral_limit,
                self.config.startup_straight_pid_integral_limit,
            )
        derivative = (
            0.0
            if previous_error is None or dt_s <= 0.0
            else (effective_error - previous_error) / dt_s
        )
        raw_output = -(
            self.config.startup_straight_pid_kp_rad_s_per_m_s * effective_error
            + self.config.startup_straight_pid_ki_rad_s_per_m
            * self._straight_pid_integral
            + self.config.startup_straight_pid_kd_rad_s_per_m_s2 * derivative
        )
        self._straight_pid_last_error = effective_error
        self._straight_pid_last_timestamp_ns = timestamp_ns
        return _clamp(
            raw_output,
            -self.config.startup_straight_pid_max_angular_velocity_rad_s,
            self.config.startup_straight_pid_max_angular_velocity_rad_s,
        )

    def _reset_straight_pid(self) -> None:
        self._straight_pid_integral = 0.0
        self._straight_pid_last_error = None
        self._straight_pid_last_timestamp_ns = None

    def _reset_green_alignment_gate(self) -> None:
        self._green_alignment_started_ns = None
        self._green_alignment_last_frame_sequence = None
        self._green_alignment_stable_count = 0

    def _begin_cluster_search(self) -> None:
        self._misgrasp_breakup_active = False
        # 离开这片区域：连续无动作退出的计数随之清零。
        self._near_field_reobserve_retries = 0
        self._breakup_plan = None
        self._breakup_proposal = None
        self._breakup_frozen_plan = None
        self._breakup_phase_started_ns = None
        self._breakup_stopped_ns = None
        self._breakup_reference_frames.clear()
        self._breakup_anchor = None
        self._breakup_reference_current = False
        self._breakup_observation_deadline_ns = None
        self._cluster_search_angular_velocity_rad_s = (
            self.config.cluster_search_angular_velocity_rad_s
        )
        # 旋转预算跨状态连续，不在这里清零；见 _reset_rotation_budget()。
        self._cluster_align_hold_center = None
        self._cluster_align_lost_since_ns = None
        self._return_phase = "idle"
        self._safe_zone_exit_base_distance_m = None
        self._d1_line_heading_rad = None
        self._d1_line_distance_m = None
        self._d1_line_start_position = None
        self._d1_calibration_offset_mm = (
            self.config.safe_zone_calibration_start_offset_mm
        )
        self._d2_line_heading_rad = None
        self._d2_line_distance_m = None
        self._d2_line_start_position = None
        self._action_settle_phase = None
        self._action_settle_until_ns = None
        self._search_frame_floor = None
        self._cluster_reference_samples = []
        self._cluster_distance_samples_mm = []
        self._cluster_reference = None
        self._cluster_reference_distance_mm = None
        self._cluster_capture_heading_rad = None
        self._cluster_reference_field_point = None
        self._cluster_breakup_end_field_point = None
        self._cluster_selected_track_ids = ()
        self._last_cluster_rejection_reason = None
        self._green_target_field_point = None
        self._green_alignment_target_heading_rad = None
        self._green_turn_start_heading_rad = None
        self._green_turn_direction = 0.0
        self._green_turn_budget_rad = 0.0
        self._green_turn_progress_rad = 0.0
        self._green_turn_attempts = 0
        self._green_turn_active = False
        self._green_alignment_wait_since_ns = None
        self._opportunistic_single_green = False
        self._near_field_group_preview = False
        self._green_realign_pending = False
        self._green_realign_done = False
        self._reset_green_alignment_gate()
        self._green_preclose_consumed_track_ids.clear()
        self._green_preclose_carried_count = 0
        self._green_preclose_realign_active = False
        self._green_preclose_frame_floor = None
        self._green_preclose_recheck_started_ns = None

    def _begin_safe_zone_scan(self) -> None:
        self._safe_zone_scan_last_heading = None
        self._safe_zone_scan_progress_rad = 0.0


    def _update_fallback_field_position(
        self, heading_rad: float | None, cumulative_distance_m: float | None,
    ) -> None:
        """Integrate an encoder arc; the same pose feeds capture-time history."""
        if heading_rad is None or cumulative_distance_m is None:
            return
        previous_distance = self._fallback_last_distance_m
        previous_heading = self._fallback_last_heading_rad
        if previous_distance is not None and self._fallback_field_position is not None:
            delta = cumulative_distance_m - previous_distance
            angle = 0.0 if previous_heading is None else normalize_angle(heading_rad - previous_heading)
            mid = heading_rad - angle / 2
            sinc = 1.0 if abs(angle) < 1e-9 else math.sin(angle / 2) / (angle / 2)
            self._fallback_field_position = FieldPoint(
                self._fallback_field_position.x + delta * 1000 * sinc * math.cos(mid),
                self._fallback_field_position.y + delta * 1000 * sinc * math.sin(mid),
            )
        self._fallback_last_distance_m = cumulative_distance_m
        self._fallback_last_heading_rad = heading_rad

    def _record_pose_history(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
    ) -> None:
        """Record measured pose pairs without inventing capture-time motion."""

        position = self._fallback_field_position
        if (
            heading_rad is None
            or cumulative_distance_m is None
            or position is None
            or not math.isfinite(float(heading_rad))
            or not math.isfinite(float(cumulative_distance_m))
        ):
            return
        entry = _PoseHistoryEntry(
            timestamp_ns,
            float(cumulative_distance_m),
            float(heading_rad),
            position,
        )
        if self._pose_history and self._pose_history[-1].timestamp_ns == timestamp_ns:
            self._pose_history[-1] = entry
        elif not self._pose_history or timestamp_ns > self._pose_history[-1].timestamp_ns:
            self._pose_history.append(entry)
        else:
            return
        del self._pose_history[:-256]

    def _pose_at(self, timestamp_ns: int) -> _PoseHistoryEntry | None:
        """Interpolate only inside recorded control history."""

        if not self._pose_history or timestamp_ns < self._pose_history[0].timestamp_ns:
            return None
        if timestamp_ns > self._pose_history[-1].timestamp_ns:
            return None
        for left, right in zip(self._pose_history, self._pose_history[1:]):
            if timestamp_ns == left.timestamp_ns:
                return left
            if left.timestamp_ns < timestamp_ns <= right.timestamp_ns:
                span = right.timestamp_ns - left.timestamp_ns
                ratio = 0.0 if span <= 0 else (timestamp_ns - left.timestamp_ns) / span
                heading_delta = normalize_angle(right.heading_rad - left.heading_rad)
                return _PoseHistoryEntry(
                    timestamp_ns,
                    left.cumulative_distance_m
                    + ratio * (right.cumulative_distance_m - left.cumulative_distance_m),
                    normalize_angle(left.heading_rad + ratio * heading_delta),
                    FieldPoint(
                        left.position.x + ratio * (right.position.x - left.position.x),
                        left.position.y + ratio * (right.position.y - left.position.y),
                    ),
                )
        if timestamp_ns == self._pose_history[-1].timestamp_ns:
            return self._pose_history[-1]
        return None

    @staticmethod
    def _field_point_from_pose(
        pose: _PoseHistoryEntry,
        ground_point: GroundPoint,
    ) -> FieldPoint:
        cosine = math.cos(pose.heading_rad)
        sine = math.sin(pose.heading_rad)
        return FieldPoint(
            pose.position.x + cosine * ground_point.x - sine * ground_point.y,
            pose.position.y + sine * ground_point.x + cosine * ground_point.y,
        )

    def _ground_point_at_now(
        self,
        ground_point: GroundPoint,
        capture_timestamp_ns: int,
        timestamp_ns: int,
    ) -> GroundPoint | None:
        """Transform a captured robot-frame point through measured motion."""

        capture_pose = self._pose_at(capture_timestamp_ns)
        current_pose = self._pose_at(timestamp_ns)
        if capture_pose is None or current_pose is None:
            return None
        field_point = self._field_point_from_pose(capture_pose, ground_point)
        delta_x = field_point.x - current_pose.position.x
        delta_y = field_point.y - current_pose.position.y
        cosine = math.cos(current_pose.heading_rad)
        sine = math.sin(current_pose.heading_rad)
        return GroundPoint(
            cosine * delta_x + sine * delta_y,
            -sine * delta_x + cosine * delta_y,
        )

    def _field_point_for_target(self, target: TrackedTarget) -> FieldPoint | None:
        point = target.ground_point
        if point is None:
            return None
        pose = self._pose_at(target.last_seen_timestamp_ns)
        if pose is None:
            return None
        return self._field_point_from_pose(pose, point)

    @staticmethod
    def _directional_delta(previous: float, current: float, angular_velocity: float) -> float:
        delta = normalize_angle(current - previous)
        if angular_velocity < 0.0:
            delta = -delta
        return max(0.0, delta)

    def _reset_rotation_budget(self) -> None:
        """开始一次动作尝试的旋转预算；空视野本身不能重置预算。"""

        self._cluster_search_last_heading = None
        self._cluster_search_progress_rad = 0.0
        self._rotation_budget_commit_diagnostic = None
        self._rotation_budget_checked_frame = None

    def _rotation_budget_text(self) -> str:
        text = f"progress_rad={self._cluster_search_progress_rad:.2f}"
        if self._rotation_budget_commit_diagnostic is None:
            return text
        return f"{text},{self._rotation_budget_commit_diagnostic}"

    def _advance_cluster_search_sweep(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
    ) -> MatchDecision | None:
        """累计"上一次执行动作之后转过的角度"；预算用尽时返回提交决策。

        预算按绝对转角累计，并且在搜索、对准、近场和重选之间连续：每个状态
        切换都清零会让"转满一圈"永远不成立（现场 20260912_0053 从 63.3 s
        转到 85.4 s，约两圈，计数器仍是 0），失败记忆因此只增不减。
        """

        if heading_rad is None:
            return None
        previous = self._cluster_search_last_heading
        self._cluster_search_last_heading = heading_rad
        if previous is None:
            return None
        self._cluster_search_progress_rad += abs(
            normalize_angle(heading_rad - previous)
        )
        if (
            self._cluster_search_progress_rad
            < self.config.cluster_search_sweep_angle_rad
        ):
            return None
        perception = self._latest_perception
        key = None if perception is None else (perception.frame_sequence, perception.capture_timestamp_ns)
        if key is None or key == self._rotation_budget_checked_frame:
            return None
        self._rotation_budget_checked_frame = key
        # 物理失败记忆不因原地转完一圈解除；一圈产出的是"必须落地"这个决定，
        # 而不是"忘记失败"。只有"这个目标需要解团"的路由标记随整圈重开。
        self._breakup_rejected_grasp_ids.clear()
        return self._commit_after_rotation_budget(timestamp_ns)

    def _commit_after_rotation_budget(self, timestamp_ns: int) -> MatchDecision | None:
        """一圈用完仍未执行任何动作时，必须推进到合法动作或换位。

        当前视野为空不能证明整片场地为空；先在失败记忆有效的前提下选择
        其它合法计划，没有合法计划（包括只看见蓝块）就执行有界换位。
        """

        if self.near_field_enabled:
            candidate = self._find_approach_seed(timestamp_ns)
            if candidate is not None:
                return self._begin_green_transport(timestamp_ns, candidate, group_preview=True)
            return self._start_rotation_budget_relocation(timestamp_ns)
        blocked_failed_aims = len(self._breakup_failed_aims)
        completed_attempts = len(self._breakup_attempts)
        plan = self._choose_breakup_plan(
            timestamp_ns,
            approach=False,
        )
        if plan is None:
            self._rotation_budget_commit_diagnostic = (
                "rotation_budget_no_legal_target "
                f"blocked_failed_aims={blocked_failed_aims} "
                f"completed_attempts={completed_attempts}"
            )
            return self._start_rotation_budget_relocation(timestamp_ns)
        self._reset_rotation_budget()
        self._rotation_budget_commit_diagnostic = (
            "rotation_budget_commit_alternative "
            f"aim_id={plan.aim_id} members={plan.member_ids} "
            f"blocked_failed_aims={blocked_failed_aims} "
            f"completed_attempts={completed_attempts}"
        )
        self._start_breakup_attempt(timestamp_ns, plan)
        self._breakup_only = True
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "rotation_budget_commit",
            posture=GripperPosture.CLOSED,
            soft_brake=True,
        )

    def _start_rotation_budget_relocation(self, timestamp_ns: int) -> MatchDecision:
        """Use one bounded, statically checked translation when no legal aim remains.

        This is a change of approach geometry, not permission to forget a
        failed aim.  The failure records stay intact and are re-evaluated only
        after the encoder-backed translation has actually changed the robot
        pose.
        """

        heading = self._latest_heading_rad
        distance_m = self.config.cluster_relocate_distance_m
        path_clear = (
            None
            if heading is None
            else self._relocation_path_clear(timestamp_ns, heading, distance_m)
        )
        if path_clear is not True:
            detail = "missing_path" if path_clear is None else "path_blocked"
            self._rotation_budget_commit_diagnostic = (
                f"rotation_budget_reselect_{detail}"
            )
            # 保留已耗尽预算，继续转到可通行方向；不要再花一整圈到同一个
            # 受阻朝向，也不要原地重复停车重观测。
            return self._decision(
                timestamp_ns, 0.0, self._cluster_search_angular_velocity_rad_s,
                f"rotation_budget_reselect_{detail}", posture=GripperPosture.CLOSED,
            )
        self._reset_rotation_budget()
        self._begin_cluster_search()
        self._breakup_only = True
        self._relocate_forward_base_distance_m = self._latest_cumulative_distance_m
        self.state = MatchState.RELOCATE_FORWARD
        self._rotation_budget_commit_diagnostic = (
            "rotation_budget_relocate "
            f"distance_mm={distance_m * 1000.0:.1f}"
        )
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "rotation_budget_relocate_start",
            posture=GripperPosture.CLOSED,
            soft_brake=True,
        )

    def _relocation_path_clear(
        self, timestamp_ns: int, heading_rad: float, distance_m: float,
    ) -> bool | None:
        """换位不接触目标；检查完整车体/夹爪扫掠及采集时刻对齐的障碍。"""
        clear = self._near_field_segment_clear(heading_rad, distance_m)
        if clear is not True:
            return clear
        perception = self._latest_perception
        geometry = self._breakup_target_geometry
        if perception is None or geometry is None or not self._fresh_perception(perception, timestamp_ns):
            return None
        margin = self.config.safety_margin_mm
        front = max(self.config.robot_footprint_radius_mm, self.config.breakup_gripper_offset_mm)
        end_mm = distance_m * 1000.0 + front + self.config.breakup_braking_margin_mm
        for observation in perception.observations:
            point = observation.ground_point
            if point is None:
                return None
            point = self._ground_point_at_now(point, perception.capture_timestamp_ns, timestamp_ns)
            if point is None:
                return None
            _, radius = physical_radii(geometry.geometry_for(observation.target_class))
            if (-self.config.robot_footprint_radius_mm - radius - margin <= point.x <= end_mm + radius + margin
                    and abs(point.y) <= self.config.robot_footprint_radius_mm + radius + margin):
                return False
        return True

    @staticmethod
    def _seconds_to_ns(seconds: float) -> int:
        return round(float(seconds) * 1_000_000_000)

    @staticmethod
    def _direct_safety_reason(safety: SafetySignals) -> str | None:
        if safety.external_stop_requested:
            return "external_stop"
        if safety.safety_accident:
            return "safety_accident"
        if safety.lost_control:
            return "lost_control"
        if safety.human_touched_after_start:
            return "human_touched_after_start"
        if safety.target_carried_on_robot:
            return "illegal_carry"
        if safety.active_attack:
            return "active_attack"
        return None

    def _validate_timestamp(self, timestamp_ns: int) -> None:
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if self._last_timestamp_ns is not None and timestamp_ns < self._last_timestamp_ns:
            raise ValueError(
                f"timestamp_ns moved backwards from {self._last_timestamp_ns} to {timestamp_ns}."
            )
        self._last_timestamp_ns = timestamp_ns

    def _decision(
        self,
        timestamp_ns: int,
        linear: float,
        angular: float,
        reason: str,
        *,
        posture: GripperPosture = GripperPosture.CLOSED,
        gripper_angles_deg: tuple[float, float] | None = None,
        soft_brake: bool = False,
        min_wheel_velocity_m_s: float | None = None,
        wheel_speeds_m_s: tuple[float, float] | None = None,
    ) -> MatchDecision:
        # 正式近场流程在远场只携带闭合夹爪接近目标；进入近场后，
        # ``gripper_angles_deg`` 才能覆盖这个闭合默认值。这样不会在
        # 远场或近场选组等待阶段提前下发固定 TRANSPORT 开口。
        if (
            self._near_field_pickup is not None
            and self.state
            in {
                MatchState.TRANSPORT_ALIGN_GREEN,
                MatchState.TRANSPORT_APPROACH_GREEN,
                MatchState.TRANSPORT_NEAR_FIELD_GRASP,
            }
            and gripper_angles_deg is None
        ):
            posture = GripperPosture.CLOSED
        if gripper_angles_deg is None and posture is GripperPosture.CLOSED and self._near_field_pickup is not None:
            classes = self._transport_target_classes
            pickup = self._near_field_pickup
            if (self.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP and pickup.active_plan is not None
                    and pickup.state is GripperWidthPickupState.CLOSING):
                classes = tuple(member.observation.target_class for member in pickup.active_plan.members)
            if TargetClass.BLACK_CORE in classes:
                gripper_angles_deg = pickup.closed_angles_for_classes(classes)
        return MatchDecision(
            timestamp_ns=timestamp_ns,
            state=self.state,
            linear_velocity_m_s=float(linear),
            angular_velocity_rad_s=float(angular),
            gripper_posture=posture,
            reason=reason,
            selected_track_id=self._selected_track_id,
            gripper_angles_deg=gripper_angles_deg,
            soft_brake=soft_brake,
            min_wheel_velocity_m_s=min_wheel_velocity_m_s,
            wheel_speeds_m_s=wheel_speeds_m_s,
        )

    CLUSTER_REFERENCE_AVERAGE_FRAMES = 5

    _D2_ACCELERATION_LIMIT_PHASES = frozenset(
        {
            "stopping_before_d2_opening",
            "opening_at_d2",
            "align_y_at_d2",
            "stopping_after_d2_heading",
            "closing_before_final_forward",
            "forward_final_open",
            "forward_final_closed",
            "stopping_before_exit_opening",
            "opening_after_transport",
        }
    )

    _BREAKUP_ACCELERATION_LIMIT_STATES = frozenset(
        {
            MatchState.ALIGN_CLUSTER_ONCE,
            MatchState.APPROACH_CLUSTER,
            MatchState.BREAKUP_SETTLE,
            MatchState.BREAKUP_FORWARD,
            MatchState.OPEN_GRIPPER_SETTLE,
            MatchState.BREAKUP_BACKWARD,
            MatchState.CLOSE_GRIPPER_SETTLE,
            MatchState.CLOSE_GRIPPER_SPIN,
            MatchState.CHECK_ISOLATED_GREEN,
            MatchState.RELOCATE_FORWARD,
        }
    )

    def step(
        self,
        timestamp_ns: int,
        *,
        perception: PerceptionSnapshot | None,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None = None,
        right_speed_feedback_m_s: float | None = None,
        safety: SafetySignals | None = None,
        near_field_preparation: GraspPreparation | None = None,
        near_field_path_clear: bool | None = None,
    ) -> MatchDecision:
        self._raw_heading_rad = heading_rad
        effective_heading = (
            None
            if heading_rad is None
            else normalize_angle(heading_rad + self._heading_offset_rad)
        )
        self._latest_heading_rad = effective_heading
        if perception is not None:
            previous = self._latest_perception
            if (perception.capture_timestamp_ns > timestamp_ns
                    or (previous is not None and
                        perception.capture_timestamp_ns < previous.capture_timestamp_ns)):
                perception = None
            else:
                if (previous is not None and perception.capture_timestamp_ns > previous.capture_timestamp_ns
                        and perception.frame_sequence > previous.frame_sequence):
                    self._perception_interval_ns = perception.capture_timestamp_ns-previous.capture_timestamp_ns
                self._latest_perception = perception
        self._latest_cumulative_distance_m = cumulative_distance_m
        self._latest_speed_feedback = (
            left_speed_feedback_m_s,
            right_speed_feedback_m_s,
        )
        decision = self._step_actions(
            timestamp_ns,
            perception=perception,
            heading_rad=effective_heading,
            cumulative_distance_m=cumulative_distance_m,
            left_speed_feedback_m_s=left_speed_feedback_m_s,
            right_speed_feedback_m_s=right_speed_feedback_m_s,
            safety=safety,
            near_field_preparation=near_field_preparation,
            near_field_path_clear=near_field_path_clear,
        )

        return self._guard_breakup_motion(decision, cumulative_distance_m)

    def _guard_breakup_motion(
        self,
        decision: MatchDecision,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """检查整个剩余直线路径；不依赖解团期间可能已锁定的视觉目标。"""

        if self._dynamic_breakup_enabled and decision.state in {
            MatchState.APPROACH_CLUSTER, MatchState.BREAKUP_FORWARD, MatchState.BREAKUP_BACKWARD,
        }:
            return self._guard_dynamic_breakup(decision, cumulative_distance_m)
        phases = {
            MatchState.APPROACH_CLUSTER,
            MatchState.BREAKUP_FORWARD,
            MatchState.BREAKUP_BACKWARD,
            MatchState.RELOCATE_FORWARD,
            MatchState.TRANSPORT_APPROACH_GREEN,
            MatchState.TRANSPORT_NEAR_FIELD_GRASP,
        }
        if decision.state not in phases or decision.linear_velocity_m_s == 0.0:
            return decision
        heading = self._latest_heading_rad
        if cumulative_distance_m is None or heading is None:
            return replace(decision, linear_velocity_m_s=0.0, angular_velocity_rad_s=0.0,
                           reason="breakup_safe_zone_guard_missing_pose_or_map")
        if decision.state is MatchState.APPROACH_CLUSTER:
            base = self._cluster_approach_base_distance_m
            travel = self._cluster_approach_travel_distance_m
            if base is None or travel is None:
                return replace(
                    decision,
                    linear_velocity_m_s=0.0,
                    angular_velocity_rad_s=0.0,
                    reason="breakup_safe_zone_guard_missing_pose_or_map",
                )
            remaining = max(0.0, travel - (cumulative_distance_m - base))
            # 接近前连同随后的固定前推一起检查，不能等贴近安全区才停车。
            remaining += self.config.breakup_forward_distance_m
        elif decision.state is MatchState.BREAKUP_FORWARD:
            base = self._breakup_forward_base_distance_m
            if base is None:
                return replace(
                    decision,
                    linear_velocity_m_s=0.0,
                    angular_velocity_rad_s=0.0,
                    reason="breakup_safe_zone_guard_missing_pose_or_map",
                )
            remaining = max(
                0.0,
                self.config.breakup_forward_distance_m
                - (cumulative_distance_m - base),
            )
        elif decision.state is MatchState.BREAKUP_BACKWARD:
            base = self._breakup_backward_base_distance_m
            if base is None:
                return replace(
                    decision,
                    linear_velocity_m_s=0.0,
                    angular_velocity_rad_s=0.0,
                    reason="breakup_safe_zone_guard_missing_pose_or_map",
                )
            remaining = -max(
                0.0,
                self.config.breakup_backward_distance_m
                - (base - cumulative_distance_m),
            )
        elif decision.state is MatchState.TRANSPORT_APPROACH_GREEN:
            base = self._green_approach_base_distance_m
            travel = self._green_approach_distance_m
            if base is None or travel is None:
                return replace(
                    decision,
                    linear_velocity_m_s=0.0,
                    angular_velocity_rad_s=0.0,
                    reason="breakup_safe_zone_guard_missing_pose_or_map",
                )
            remaining = max(0.0, travel - (cumulative_distance_m - base))
        elif decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP:
            pickup = self._near_field_pickup
            if pickup is None or pickup.state is not GripperWidthPickupState.FORWARD:
                return decision
            plan = pickup.active_plan
            if plan is None:
                return decision
            remaining = max(0.0, plan.forward_distance_mm / 1000.0 - pickup.progress_mm(cumulative_distance_m) / 1000.0)
            segment_clear = self._near_field_segment_clear(heading, remaining, plan=plan)
            if segment_clear is None:
                return replace(
                    decision,
                    linear_velocity_m_s=0.0,
                    angular_velocity_rad_s=0.0,
                    reason="near_field_safe_zone_guard_missing_pose_or_map",
                    soft_brake=True,
                )
            if not segment_clear:
                return self._start_path_reselection(
                    decision.timestamp_ns,
                    posture=GripperPosture.TRANSPORT,
                    reason="near_field_path_intersects_safe_zone_restart_search",
                )
            return decision
        else:
            base = self._relocate_forward_base_distance_m
            if base is None:
                return replace(
                    decision,
                    linear_velocity_m_s=0.0,
                    angular_velocity_rad_s=0.0,
                    reason="breakup_safe_zone_guard_missing_pose_or_map",
                )
            remaining = max(
                0.0,
                self.config.cluster_relocate_distance_m
                - (cumulative_distance_m - base),
            )
            path_clear = self._relocation_path_clear(decision.timestamp_ns, heading, remaining)
            if path_clear is None:
                return replace(
                    decision,
                    linear_velocity_m_s=0.0,
                    angular_velocity_rad_s=0.0,
                    reason="breakup_safe_zone_guard_missing_pose_or_map",
                    soft_brake=True,
                )
            if not path_clear:
                return self._start_path_reselection(
                    decision.timestamp_ns,
                    posture=decision.gripper_posture,
                    reason="target_path_intersects_safe_zone_stop_and_reselect",
                )
            return decision
        blocked = self._safe_zone_path_blocked(heading, remaining)
        if blocked is None:
            return replace(decision, linear_velocity_m_s=0.0, angular_velocity_rad_s=0.0,
                           reason="breakup_safe_zone_guard_missing_pose_or_map")
        if not blocked:
            return decision
        return self._start_path_reselection(
            decision.timestamp_ns,
            posture=decision.gripper_posture,
            reason="target_path_intersects_safe_zone_stop_and_reselect",
        )

    def _start_path_reselection(
        self,
        timestamp_ns: int,
        *,
        posture: GripperPosture,
        reason: str,
    ) -> MatchDecision:
        """放弃当前路线，停车确认后清空旧轨迹并重新选团/绿色目标。"""

        self._remember_near_field_failure(timestamp_ns, reason)
        self._grasp_task = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id += 1
        self._begin_cluster_search()
        self._selected_track_id = None
        self._selected_green_ground = None
        self._green_reference_samples = []
        self._green_reference_last_seen_ns = None
        self._green_reference = None
        self._green_reference_heading_rad = None
        self._green_reference_distance_m = None
        self._cluster_approach_base_distance_m = None
        self._cluster_approach_travel_distance_m = None
        self._breakup_forward_base_distance_m = None
        self._breakup_backward_base_distance_m = None
        self._green_approach_base_distance_m = None
        self._green_approach_distance_m = None
        self._green_align_lost_since_ns = None
        self._reset_green_alignment_gate()
        self._path_recovery = True
        self._path_recovery_stopped_ns = None
        self._path_recovery_posture = posture
        self._safe_zone_stop_since_ns = None
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            reason,
            posture=posture,
        )

    def _safe_zone_path_blocked(self, heading: float, distance_m: float) -> bool | None:
        """检查红蓝两区膨胀矩形与路径相交；地图/航位缺失返回 None。"""

        position = self.estimated_field_position
        if position is None or self._breakup_static_map is None:
            return None
        end = FieldPoint(position.x + distance_m * 1000.0 * math.cos(heading),
                         position.y + distance_m * 1000.0 * math.sin(heading))
        return safe_zone_intersection(self._breakup_static_map, position, end, self._breakup_clearance_mm)

    def _near_field_target_final_x_mm(self) -> float:
        """Return the endpoint used by the active near-field transport phase."""

        config = self._near_field_grasp_config
        if config is None:
            return 160.0
        return float(
            config.greedy_target_final_x_mm
            if self._greedy_active
            else config.target_final_x_mm
        )

    @property
    def near_field_target_final_x_mm(self) -> float:
        """Effective configured endpoint for near-field diagnostics, in mm."""

        return self._near_field_target_final_x_mm()

    def _candidate_path_blocked(self, point: GroundPoint, *, breakup: bool) -> bool:
        heading = self._latest_heading_rad
        if heading is None:
            return False  # 缺失信息由平移出口保持零速，不伪造可行路线。
        offset = (
            self.config.cluster_breakup_standoff_mm
            if breakup
            else (
                self.config.green_grab_offset_mm
                if self._near_field_pickup is None
                else self._near_field_target_final_x_mm()
            )
        )
        distance_m = max(0.0, math.hypot(point.x, point.y) - offset) / 1000.0
        if breakup:
            distance_m += self.config.breakup_forward_distance_m
        elif self._near_field_pickup is not None:
            return self._near_field_segment_clear(
                heading + math.atan2(point.y, point.x), distance_m,
            ) is False
        return self._safe_zone_path_blocked(
            heading + math.atan2(point.y, point.x), distance_m,
        ) is True

    def _field_point_from_ground(self, point: GroundPoint) -> FieldPoint | None:
        """把当前机器人地面系点转换为场地坐标，用于解团边界门禁。"""

        position = self.estimated_field_position
        heading = self._latest_heading_rad
        if position is None or heading is None:
            return None
        cosine = math.cos(heading)
        sine = math.sin(heading)
        return FieldPoint(
            position.x + cosine * point.x - sine * point.y,
            position.y + sine * point.x + cosine * point.y,
        )

    def _field_point_along_heading(
        self,
        distance_mm: float,
        heading_rad: float,
    ) -> FieldPoint | None:
        """返回从当前航位沿指定场地航向前进后的场地点。"""

        position = self.estimated_field_position
        if position is None:
            return None
        return FieldPoint(
            position.x + distance_mm * math.cos(heading_rad),
            position.y + distance_mm * math.sin(heading_rad),
        )

    def _breakup_center_within_field_boundary(
        self,
        field_point: FieldPoint | None,
    ) -> bool:
        """检查解团中点是否位于场界扣除夹爪偏移后的有效范围内。"""

        if field_point is None:
            return False
        min_x, max_x, min_y, max_y = self._breakup_allowed_field_bounds()
        return (
            min_x <= field_point.x <= max_x
            and min_y <= field_point.y <= max_y
        )

    def _breakup_allowed_field_bounds(self) -> tuple[float, float, float, float]:
        """Legacy point-only checks: field already inset by jaw reach."""
        min_x, max_x, min_y, max_y = self._physical_field_bounds()
        offset = self.config.breakup_gripper_offset_mm
        return min_x + offset, max_x - offset, min_y + offset, max_y - offset

    def _robot_path_clearance_mm(self) -> float:
        return max(self._breakup_clearance_mm,
                   robot_clearance_mm(self.config, GripperKinematics().left_tip_position(0).x))

    def _physical_field_bounds(self) -> tuple[float, float, float, float]:
        """返回未扣除任何实体尺寸的物理场界，路径检查按各实体包络扣一次。

        ``world.static_map`` 中的 field 区域是运行时首选权威；没有 field 区域
        的纯逻辑构造则使用配置的 ±1500 mm 回退值。
        """

        field_regions = tuple(
            region
            for region in (
                self._breakup_static_map.regions
                if self._breakup_static_map is not None
                else ()
            )
            if region.kind is PhysicalRegionKind.FIELD
        )
        if len(field_regions) == 1:
            points = field_regions[0].polygon_field
            min_x = min(point.x for point in points)
            max_x = max(point.x for point in points)
            min_y = min(point.y for point in points)
            max_y = max(point.y for point in points)
        else:
            extent = self.config.breakup_field_half_extent_mm
            min_x = min_y = -extent
            max_x = max_y = extent
        return min_x, max_x, min_y, max_y

    def _breakup_center_boundary_reason(
        self,
        field_point: FieldPoint | None,
    ) -> str:
        if field_point is None:
            return "breakup_center_boundary_missing_pose"
        min_x, max_x, min_y, max_y = self._breakup_allowed_field_bounds()
        return (
            "breakup_center_out_of_field_boundary:"
            f"center_field=({field_point.x:.1f},{field_point.y:.1f}),"
            f"allowed=({min_x:.1f}..{max_x:.1f},"
            f"{min_y:.1f}..{max_y:.1f})mm"
        )

    def _breakup_end_boundary_reason(
        self,
        field_point: FieldPoint | None,
    ) -> str:
        if field_point is None:
            return "breakup_end_boundary_missing_pose"
        min_x, max_x, min_y, max_y = self._breakup_allowed_field_bounds()
        return (
            "breakup_end_out_of_field_boundary:"
            f"end_field=({field_point.x:.1f},{field_point.y:.1f}),"
            f"allowed=({min_x:.1f}..{max_x:.1f},"
            f"{min_y:.1f}..{max_y:.1f})mm"
        )

    def _preferred_cluster_ground_points(
        self,
        timestamp_ns: int,
    ) -> frozenset[GroundPoint]:
        """仅首轮把规则要求的绿色物资所在团标为优先候选。"""

        if self._transport_count > 0:
            return frozenset()

        return frozenset(
            target.ground_point
            for target in self._tracker.tracks
            if target.ground_point is not None
            and target.target_class is TargetClass.GREEN_SUPPLY
            and target.ever_confirmed
            and self._target_is_fresh(target, timestamp_ns)
            # 已交付绿色同样不能作为首轮优先候选。
            and not self._ground_in_safe_zone(
                target.ground_point, target.last_seen_timestamp_ns
            )
            and not self._target_overlaps_safe_zone_bbox(target.box)
        )

    def _largest_ground_group(
        self,
        points: list[GroundPoint],
        *,
        preferred_points: frozenset[GroundPoint] | None = None,
    ) -> list[GroundPoint] | None:
        # 首轮按规则所需绿色、其余轮次按成员数处理完整连通团；拒绝危险路线后
        # 继续找下一团，不拆团制造假孤立目标。
        self._last_cluster_rejection_reason = None
        remaining = list(points)
        while remaining:
            group = self._connected_ground_group(
                remaining,
                preferred_points=preferred_points,
            )
            if group is None:
                return None
            center = GroundPoint(sum(p.x for p in group) / len(group),
                                 sum(p.y for p in group) / len(group))
            field_center = self._field_point_from_ground(center)
            if not self._breakup_center_within_field_boundary(field_center):
                self._last_cluster_rejection_reason = (
                    self._breakup_center_boundary_reason(field_center)
                )
            else:
                heading = self._latest_heading_rad
                forward_points = [point.x for point in group if point.x > 0.0]
                breakup_end = None
                if heading is not None and forward_points:
                    approach_mm = max(
                        0.0,
                        min(forward_points)
                        - self.config.cluster_breakup_standoff_mm,
                    )
                    breakup_heading = heading + math.atan2(center.y, center.x)
                    breakup_end = self._field_point_along_heading(
                        approach_mm
                        + self.config.breakup_forward_distance_m * 1000.0,
                        breakup_heading,
                    )
                if not self._breakup_center_within_field_boundary(breakup_end):
                    self._last_cluster_rejection_reason = (
                        self._breakup_end_boundary_reason(breakup_end)
                    )
                elif self._candidate_path_blocked(center, breakup=True):
                    self._last_cluster_rejection_reason = (
                        "breakup_candidate_path_blocked"
                    )
                else:
                    return group
            rejected = set(group)
            remaining = [point for point in remaining if point not in rejected]
        return None

    @property
    def estimated_field_heading_rad(self) -> float | None:
        """当前由陀螺仪并叠加最近一次视觉校正得到的场地航向。"""

        return self._latest_heading_rad

    @property
    def safe_zone_route_phase(self) -> str:
        """返回安全区路线的细分阶段，供车端诊断记录使用。"""

        return self._safe_zone_phase

    @property
    def safe_zone_motion_acceleration_limits(
        self,
    ) -> MotionAccelerationOverrides | None:
        """返回 D2→安全区末端当前应使用的车体加减速度覆盖。"""

        if (
            self._safe_zone_phase
            not in self._D2_ACCELERATION_LIMIT_PHASES
        ):
            return None
        return MotionAccelerationOverrides(
            linear_acceleration_m_s2=(
                self.config.safe_zone_d2_to_final_max_linear_acceleration_m_s2
            ),
            linear_deceleration_m_s2=(
                self.config.safe_zone_d2_to_final_max_linear_deceleration_m_s2
            ),
            angular_acceleration_rad_s2=(
                self.config.safe_zone_d2_to_final_max_angular_acceleration_rad_s2
            ),
            angular_deceleration_rad_s2=(
                self.config.safe_zone_d2_to_final_max_angular_deceleration_rad_s2
            ),
        )

    @property
    def breakup_motion_acceleration_limits(
        self,
    ) -> MotionAccelerationOverrides | None:
        """返回解团阶段当前应使用的车体加减速度覆盖。"""

        if self.state not in self._BREAKUP_ACCELERATION_LIMIT_STATES:
            return None
        return MotionAccelerationOverrides(
            linear_acceleration_m_s2=(
                self.config.breakup_max_linear_acceleration_m_s2
            ),
            linear_deceleration_m_s2=(
                self.config.breakup_max_linear_deceleration_m_s2
            ),
            angular_acceleration_rad_s2=(
                self.config.breakup_max_angular_acceleration_rad_s2
            ),
            angular_deceleration_rad_s2=(
                self.config.breakup_max_angular_deceleration_rad_s2
            ),
        )

    @property
    def motion_acceleration_limits(self) -> MotionAccelerationOverrides | None:
        """返回当前 match 动作应使用的车体加减速度覆盖。"""

        if self._noncontact_controller is not None:
            return MotionAccelerationOverrides(
                linear_deceleration_m_s2=self._motion_profile.linear_deceleration_m_s2,
                angular_deceleration_rad_s2=self._motion_profile.angular_deceleration_rad_s2,
            )
        breakup_limits = self.breakup_motion_acceleration_limits
        if breakup_limits is not None:
            return breakup_limits
        return self.safe_zone_motion_acceleration_limits

    @property
    def wheel_acceleration_limits(self) -> WheelAccelerationOverrides | None:
        """当前动作的独立左右轮加减速度覆盖。"""

        return None

    @property
    def safe_zone_calibration_pose(self) -> FieldPose2D | None:
        """最近一次 d1 安全区两点视觉校正后的场地位姿。"""

        observation = self._safe_zone_calibration_pose
        return None if observation is None else observation.pose

    def _begin_green_transport(
        self,
        timestamp_ns: int,
        target: TrackedTarget,
        *,
        opportunistic: bool = False,
        group_preview: bool = False,
    ) -> MatchDecision:
        """路径无阻挡时直接进入物资目标对准，不做反向回转确认。"""

        if self.near_field_enabled:
            self._adopt_grasp_task(
                timestamp_ns, track_id=target.track_id, target_class=target.target_class,
                point=self._current_ground_point_for_track(target, timestamp_ns),
                field=self._field_point_for_target(target),
            )
        self._selected_track_id = target.track_id
        self._cluster_selected_track_ids = ()
        self._breakup_only = False
        self._selected_green_ground = target.ground_point
        self._green_target_field_point = self._field_point_for_target(target)
        self._green_alignment_target_heading_rad = None
        self._green_turn_start_heading_rad = None
        self._green_turn_direction = 0.0
        self._green_turn_budget_rad = 0.0
        self._green_turn_progress_rad = 0.0
        self._green_turn_attempts = 0
        self._green_turn_active = False
        self._green_alignment_wait_since_ns = None
        self._green_align_lost_since_ns = None
        self._opportunistic_single_green = opportunistic
        self._near_field_group_preview = group_preview
        self._green_realign_pending = False
        self._green_realign_done = False
        self._green_preclose_consumed_track_ids.clear()
        self._green_preclose_carried_count = 1
        self._green_preclose_realign_active = False
        self._green_preclose_frame_floor = None
        self._green_preclose_recheck_started_ns = None
        self._green_align_lost_since_ns = None
        self._green_approach_base_distance_m = None
        self._green_approach_distance_m = None
        self._green_reference_samples = []
        self._green_reference_last_seen_ns = None
        self._green_reference = None
        self._green_reference_heading_rad = None
        self._green_reference_distance_m = None
        self._reset_green_alignment_gate()
        self.state = MatchState.TRANSPORT_ALIGN_GREEN
        self._begin_action_settle(timestamp_ns, "green_reference")
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            (
                f"near_field_group_preview:{target.track_id}"
                if group_preview
                else f"green_path_clear_opportunistic_single:{target.track_id}"
                if opportunistic
                else f"green_path_clear_align_direct:{target.track_id}"
            ),
            posture=GripperPosture.TRANSPORT,
        )

    def _begin_action_settle(
        self,
        timestamp_ns: int,
        phase: str,
    ) -> bool:
        """登记一次非阻塞零速保持；返回是否需要等待。"""

        duration_ns = self._seconds_to_ns(self.config.action_settle_time_s)
        self._action_settle_phase = None
        self._action_settle_until_ns = None
        if duration_ns <= 0:
            return False
        self._action_settle_phase = phase
        self._action_settle_until_ns = timestamp_ns + duration_ns
        return True

    def _consume_action_settle(
        self,
        timestamp_ns: int,
        phase: str,
        *,
        posture: GripperPosture,
        reason: str,
    ) -> MatchDecision | None:
        """在指定状态消费动作间隔；未到期时始终返回零速。"""

        if self._action_settle_phase != phase:
            return None
        until_ns = self._action_settle_until_ns
        if until_ns is None or timestamp_ns < until_ns:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                reason,
                posture=posture,
            )
        self._action_settle_phase = None
        self._action_settle_until_ns = None
        return None

    def _heading_hold_angular_velocity(
        self,
        desired_heading_rad: float | None,
        *,
        kp_rad_s: float,
        max_angular_velocity_rad_s: float,
        tolerance_rad: float,
    ) -> float | None:
        """按当前有效航向保持直线段方向；缺少航向时返回 None。"""

        current_heading = self._latest_heading_rad
        if desired_heading_rad is None or current_heading is None:
            return None
        error = normalize_angle(desired_heading_rad - current_heading)
        if abs(error) <= tolerance_rad:
            return 0.0
        return _clamp(
            kp_rad_s * error,
            -max_angular_velocity_rad_s,
            max_angular_velocity_rad_s,
        )

    def _refresh_forward_segment_after_settle(
        self,
        phase: str,
        cumulative_distance_m: float | None,
    ) -> None:
        """用停稳后的位姿/里程重新起算下一段，排除刹车残余位移。"""

        position = self._fallback_field_position
        if position is None or cumulative_distance_m is None:
            return
        if phase == "safe_before_d1_forward":
            target = self._safe_zone_d1_target()
            delta_x = target.x - position.x
            delta_y = target.y - position.y
            self._d1_line_start_position = position
            self._d1_line_heading_rad = math.atan2(delta_y, delta_x)
            self._d1_line_distance_m = math.hypot(
                delta_x,
                delta_y,
            ) / 1000.0
            self._transport_forward_base_distance_m = cumulative_distance_m
            self._transport_forward_distance_m = self._d1_line_distance_m
            return
        if phase == "safe_before_d2_forward":
            target = self._safe_zone_d2_target()
            delta_x = target.x - position.x
            delta_y = target.y - position.y
            self._d2_line_start_position = position
            self._d2_line_heading_rad = math.atan2(delta_y, delta_x)
            self._d2_line_distance_m = math.hypot(
                delta_x,
                delta_y,
            ) / 1000.0
            self._transport_forward_base_distance_m = cumulative_distance_m
            self._transport_forward_distance_m = self._d2_line_distance_m
            return
        if phase == "safe_before_final_forward":
            self._transport_forward_base_distance_m = cumulative_distance_m
            self._transport_forward_distance_m = max(
                0.0,
                self._safe_zone_forward_y_sign
                * (self._safe_zone_final_target_y_mm() - position.y)
                / 1000.0,
            )

    def _safe_zone_final_target_y_mm(self) -> float:
        """返回应用末段刹车提前量后的安全区停车 y 坐标。"""

        return (
            self._safe_zone_transport_endpoint().y
            - self._safe_zone_forward_y_sign
            * self._safe_zone_d2_to_final_braking_overrun_mm()
        )

    def _safe_zone_d2_to_final_speed_m_s(self) -> float:
        """返回当前运输类别对应的 d2→安全区末段速度。"""

        if self._transport_target_classes == (TargetClass.ORANGE_INJURED,):
            return self.config.safe_zone_orange_d2_to_final_speed_m_s
        return self.config.safe_zone_d2_to_final_speed_m_s

    def _safe_zone_d2_to_final_braking_overrun_mm(self) -> float:
        """返回当前运输类别对应的末段刹车过冲提前量。"""

        if self._transport_target_classes == (TargetClass.ORANGE_INJURED,):
            return self.config.safe_zone_orange_d2_to_final_braking_overrun_mm
        return self.config.safe_zone_d2_to_final_braking_overrun_mm

    @staticmethod
    def _target_center_in_safe_zone_bbox(
        target_box: UndistortedBoundingBox,
        safe_zone_boxes: tuple[UndistortedBoundingBox, ...],
    ) -> bool:
        """用目标 bbox 中心判断其是否落在任一安全区 bbox 内。"""

        center_u = 0.5 * (target_box.x_min + target_box.x_max)
        center_v = 0.5 * (target_box.y_min + target_box.y_max)
        return any(
            zone_box.x_min <= center_u <= zone_box.x_max
            and zone_box.y_min <= center_v <= zone_box.y_max
            for zone_box in safe_zone_boxes
        )

    def _has_searchable_collectible_information(self) -> bool:
        """只把安全区 bbox 外的有效非蓝目标视为搜索信息。"""

        perception = self._latest_perception
        if perception is None:
            return False
        field_features = perception.field_features
        safe_zone_boxes = (
            tuple(zone.box for zone in field_features.safe_zones)
            if field_features is not None
            else ()
        )
        return any(
            observation.target_class
            in {
                TargetClass.GREEN_SUPPLY,
                TargetClass.BLACK_CORE,
                TargetClass.ORANGE_INJURED,
            }
            and not self._target_center_in_safe_zone_bbox(
                observation.box,
                safe_zone_boxes,
            )
            for observation in perception.observations
        )

    def _update_cluster_search_velocity(self) -> None:
        """按当前帧信息量切换搜索转速，同时保留当前扫描方向。"""

        has_collectible_information = self._has_searchable_collectible_information()
        magnitude = abs(
            self.config.cluster_search_angular_velocity_rad_s
            if has_collectible_information
            else self.config.cluster_search_empty_angular_velocity_rad_s
        )
        self._cluster_search_angular_velocity_rad_s = math.copysign(
            magnitude,
            self._cluster_search_angular_velocity_rad_s,
        )

    def _step_search_cluster(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
    ) -> MatchDecision:
        if self._path_recovery:
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                self._path_recovery_stopped_ns = None
                return self._decision(timestamp_ns, 0.0, 0.0,
                                      "safe_zone_reselect_waiting_for_stop",
                                      posture=self._path_recovery_posture)
            if self._path_recovery_stopped_ns is None:
                self._path_recovery_stopped_ns = timestamp_ns
                self._reset_tracker_for_new_preview_epoch()
                # The fresh-frame gate below is based on capture time.  Reset
                # the sequence gate together with the tracker so a replay
                # source that reuses sequence numbers can still provide the
                # first post-stop observation.
                self._last_tracker_frame_sequence = None
            perception = self._latest_perception
            if (perception is None or not self._fresh_perception(perception, timestamp_ns)
                    or perception.capture_timestamp_ns
                    <= self._path_recovery_stopped_ns):
                # ``self._step_actions()`` updates the tracker before dispatch.  A
                # repeated pre-stop snapshot must not repopulate the tracks we
                # deliberately cleared at the stop boundary.
                self._reset_tracker_for_new_preview_epoch()
                self._last_tracker_frame_sequence = None
                return self._decision(timestamp_ns, 0.0, 0.0,
                                      "safe_zone_reselect_waiting_for_new_frame",
                                      posture=self._path_recovery_posture)
            self._path_recovery = False
            self._path_recovery_stopped_ns = None
            self._safe_zone_stop_since_ns = None
        if self._dynamic_breakup_enabled:
            return self._step_dynamic_cluster_search(timestamp_ns, heading_rad)
        self._update_cluster_search_velocity()
        # 一般阶段统一比较当前可接近的目标，不让正前方低分远目标
        # 抢占侧方近目标；种子与对准复核使用相同的路径条件。
        if (not self._breakup_only and self.config.opportunistic_single_green_enabled
                and self._near_field_pickup is not None and self._transport_count > 0):
            target = self._find_approach_seed(timestamp_ns)
            if target is not None:
                direct = self._transport_group_size(target, timestamp_ns) is not None
                return self._begin_green_transport(timestamp_ns, target,
                                                   opportunistic=direct, group_preview=not direct)
        if (
            not self._breakup_only
            and self._first_green_forward_corridor_blocked(timestamp_ns)
        ):
            # 首轮最近绿块已经被前向走廊中的其它目标挡住时，直接进入
            # 解团规划；不要先尝试更远的绿色候选。
            return self._enter_breakup_only_search(timestamp_ns)
        if (not self._breakup_only and self.config.opportunistic_single_green_enabled
                and (self._transport_count == 0 or self._near_field_pickup is None)):
            target = self._find_opportunistic_single_green(timestamp_ns)
            if target is not None:
                # 已确认目标进入对准；正式近场复用当前点，最终几何由近场确认。
                return self._begin_green_transport(
                    timestamp_ns,
                    target,
                    opportunistic=True,
                )
        measurement = self._cluster_ground_measurement(timestamp_ns)
        if measurement is None:
            committed = self._advance_cluster_search_sweep(timestamp_ns, heading_rad)
            if committed is not None:
                return committed
            rejection = self._last_cluster_rejection_reason
            self._last_cluster_rejection_reason = None
            reason = (
                "search_cluster_rejected_candidate:"
                f"{rejection}"
                if rejection is not None
                else (
                    "search_cluster_right"
                    if self._cluster_search_angular_velocity_rad_s < 0.0
                    else "search_cluster_left"
                )
            )
            return self._decision(
                timestamp_ns,
                0.0,
                self._cluster_search_angular_velocity_rad_s,
                reason,
            )
        self._cluster_reference_samples = [measurement.center]
        self._cluster_distance_samples_mm = [
            measurement.nearest_forward_x_mm
        ]
        self._cluster_capture_heading_rad = heading_rad
        self._cluster_align_hold_center = None
        self._cluster_align_lost_since_ns = None
        self._breakup_only = False
        self.state = MatchState.ALIGN_CLUSTER_ONCE
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "cluster_seen_stop_collect_reference",
        )

    def _step_align_cluster(self, timestamp_ns: int) -> MatchDecision:
        if self._cluster_reference is None:
            measurement = self._cluster_ground_measurement(timestamp_ns)
            if measurement is None:
                lost_since = self._cluster_align_lost_since_ns
                if lost_since is None:
                    self._cluster_align_lost_since_ns = timestamp_ns
                elif timestamp_ns - lost_since >= self._cluster_align_hold_ns:
                    self._begin_cluster_search()
                    self.state = MatchState.SEARCH_CLUSTER
                    return self._decision(
                        timestamp_ns,
                        0.0,
                        self._cluster_search_angular_velocity_rad_s,
                        "cluster_reference_collection_timeout_restart_search",
                    )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "cluster_collecting_reference_waiting_for_fresh_cluster",
                )
            self._cluster_align_lost_since_ns = None
            self._cluster_reference_samples.append(measurement.center)
            self._cluster_distance_samples_mm.append(
                measurement.nearest_forward_x_mm
            )
            if (
                len(self._cluster_reference_samples)
                < self.CLUSTER_REFERENCE_AVERAGE_FRAMES
            ):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "cluster_collecting_reference",
                )
            self._cluster_reference = GroundPoint(
                sum(
                    point.x for point in self._cluster_reference_samples
                )
                / len(self._cluster_reference_samples),
                sum(
                    point.y for point in self._cluster_reference_samples
                )
                / len(self._cluster_reference_samples),
            )
            self._cluster_reference_distance_mm = sum(
                self._cluster_distance_samples_mm
            ) / len(self._cluster_distance_samples_mm)
            field_center = self._field_point_from_ground(
                self._cluster_reference
            )
            if not self._breakup_center_within_field_boundary(field_center):
                reason = self._breakup_center_boundary_reason(field_center)
                return self._start_path_reselection(
                    timestamp_ns,
                    posture=GripperPosture.CLOSED,
                    reason=reason,
                )
            self._cluster_reference_field_point = field_center
            capture_heading = self._cluster_capture_heading_rad
            if capture_heading is not None:
                breakup_heading = capture_heading + math.atan2(
                    self._cluster_reference.y,
                    self._cluster_reference.x,
                )
                breakup_end = self._field_point_along_heading(
                    max(
                        0.0,
                        self._cluster_reference_distance_mm
                        - self.config.cluster_breakup_standoff_mm,
                    )
                    + self.config.breakup_forward_distance_m * 1000.0,
                    breakup_heading,
                )
                if not self._breakup_center_within_field_boundary(breakup_end):
                    return self._start_path_reselection(
                        timestamp_ns,
                        posture=GripperPosture.CLOSED,
                        reason=self._breakup_end_boundary_reason(breakup_end),
                    )
                self._cluster_breakup_end_field_point = breakup_end
        if (
            self._cluster_reference is None
            or self._cluster_capture_heading_rad is None
            or self._latest_heading_rad is None
        ):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "cluster_reference_waiting_for_heading",
            )
        reference_bearing = math.atan2(
            self._cluster_reference.y,
            self._cluster_reference.x,
        )
        target_heading = (
            self._cluster_capture_heading_rad + reference_bearing
        )
        heading_error = normalize_angle(
            target_heading - self._latest_heading_rad
        )
        return self._cluster_alignment_decision(timestamp_ns, heading_error)

    def _step_approach_cluster(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """使用锁定的均值参照点和距离，不再更新团观测。"""

        reference_distance = self._cluster_reference_distance_mm
        if reference_distance is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "approach_cluster_waiting_for_locked_reference",
            )
        if reference_distance <= self.config.cluster_breakup_standoff_mm:
            rejected = self._start_breakup_settle_if_allowed(
                timestamp_ns,
                cumulative_distance_m,
            )
            if rejected is not None:
                return rejected
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "cluster_standoff_reached_wait_0_3s",
            )
        if self._cluster_approach_travel_distance_m is None:
            self._cluster_approach_travel_distance_m = (
                reference_distance - self.config.cluster_breakup_standoff_mm
            ) / 1000.0
        if cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "approach_cluster_waiting_for_odometry",
            )
        if self._cluster_approach_base_distance_m is None:
            self._cluster_approach_base_distance_m = cumulative_distance_m
        travelled = cumulative_distance_m - self._cluster_approach_base_distance_m
        if travelled >= self._cluster_approach_travel_distance_m - 1e-9:
            rejected = self._start_breakup_settle_if_allowed(
                timestamp_ns,
                cumulative_distance_m,
            )
            if rejected is not None:
                return rejected
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "cluster_standoff_reached_wait_0_3s",
            )
        return self._decision(
            timestamp_ns,
            self.config.cluster_approach_speed_m_s,
            0.0,
            "approach_cluster_to_locked_standoff",
        )

    def _start_breakup_settle_if_allowed(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision | None:
        """解团前再次验证锁定中点边界，再进入非阻塞停稳阶段。"""

        field_center = self._cluster_reference_field_point
        if not self._breakup_center_within_field_boundary(field_center):
            reason = self._breakup_center_boundary_reason(field_center)
            return self._start_path_reselection(
                timestamp_ns,
                posture=GripperPosture.CLOSED,
                reason=reason,
            )
        reference = self._cluster_reference
        capture_heading = self._cluster_capture_heading_rad
        if reference is None or capture_heading is None:
            return self._start_path_reselection(
                timestamp_ns,
                posture=GripperPosture.CLOSED,
                reason="breakup_end_boundary_missing_target_heading",
            )
        target_heading = capture_heading + math.atan2(reference.y, reference.x)
        remaining_approach_m = 0.0
        if (
            cumulative_distance_m is not None
            and self._cluster_approach_base_distance_m is not None
            and self._cluster_approach_travel_distance_m is not None
        ):
            remaining_approach_m = max(
                0.0,
                self._cluster_approach_travel_distance_m
                - (
                    cumulative_distance_m
                    - self._cluster_approach_base_distance_m
                ),
            )
        else:
            remaining_approach_m = max(
                0.0,
                self._cluster_reference_distance_mm
                - self.config.cluster_breakup_standoff_mm,
            ) / 1000.0
        breakup_end = self._field_point_along_heading(
            (
                remaining_approach_m
                + self.config.breakup_forward_distance_m
            )
            * 1000.0,
            target_heading,
        )
        if not self._breakup_center_within_field_boundary(breakup_end):
            return self._start_path_reselection(
                timestamp_ns,
                posture=GripperPosture.CLOSED,
                reason=self._breakup_end_boundary_reason(breakup_end),
            )
        self._cluster_breakup_end_field_point = breakup_end
        self.state = MatchState.BREAKUP_SETTLE
        self._settle_until_ns = timestamp_ns + self._seconds_to_ns(
            self.config.breakup_settle_time_s
        )
        self._breakup_forward_base_distance_m = cumulative_distance_m
        return None

    def cluster_diagnostic(self, timestamp_ns: int) -> str:
        """返回当前新鲜团中心和最前方 x，供进度日志使用。"""

        if self._dynamic_breakup_enabled:
            plan = self._breakup_plan or self._breakup_proposal
            perception = self._latest_perception
            preparation = self._breakup_grasp_preparation
            obs_age = None if perception is None else perception.age_ns(timestamp_ns)
            prep_age = None if preparation is None else preparation.preparation_age_ns(timestamp_ns)
            diagnostic = (f"confirmation={len(self._breakup_reference_frames)}/{self.config.breakup_confirmation_frames},"
                          f"wait={self._breakup_wait_detail},"
                          f"misgrasp_locked_group={self._misgrasp_breakup_active},"
                          f"release_pose={self._misgrasp_release_pose},"
                          f"segment_stop_started_ns={self._breakup_segment_stop_started_ns},"
                          f"segment_stop_deadline_ns={self._breakup_segment_stop_deadline_ns},"
                          f"reference_current={self._breakup_reference_current},"
                          f"rejected_candidates=[{';'.join(self._breakup_plan_rejections)}],"
                          f"source={'encoder_imu' if self._stationary_motion.latest is not None else 'step_feedback'},"
                          f"frame={None if perception is None else perception.frame_sequence},"
                          f"capture_ns={None if perception is None else perception.capture_timestamp_ns},"
                          f"observation_age_ms={None if obs_age is None else round(obs_age/1e6, 1)},"
                          f"preparation_age_ms={None if prep_age is None else round(prep_age/1e6, 1)},"
                          f"preparation_capture_ns={None if preparation is None else preparation.capture_timestamp_ns},"
                          f"last_capture_ns={self._breakup_last_capture_ns},"
                          f"deadline_ns={self._breakup_observation_deadline_ns},"
                          f"alignment_deadline_ns={None if self._breakup_phase_started_ns is None else self._breakup_phase_started_ns + self._cluster_align_hold_ns + getattr(self, '_breakup_alignment_allowance_ns', 0)},"
                          f"no_plan_since_ns={self._breakup_no_plan_since_ns},"
                          f"no_plan_window_ms={self.config.breakup_no_plan_reobserve_ms},"
                          f"failed_regions={len(self._breakup_failed_aims)},"
                          f"rotation={self._rotation_budget_text()},"
                          f"{self._stationary_motion.diagnostic(timestamp_ns)},")
            if plan is None:
                return (diagnostic
                        + f"none,rejection={self._last_cluster_rejection_reason}"
                        + f",rejected_candidates=[{';'.join(self._breakup_plan_rejections)}]")
            return (diagnostic + f"aim={plan.aim_id}@({plan.aim.x:.1f},{plan.aim.y:.1f}),"
                    f"heading_error_rad={None if self._latest_heading_rad is None else round(normalize_angle(plan.heading_rad-self._latest_heading_rad), 4)},"
                    f"heading_tolerance_rad={self._breakup_alignment_tolerance(plan):.4f},"
                    f"members={plan.member_ids},contact={plan.contact_ids},attempt={plan.attempt},"
                    f"approach_mm={plan.approach_distance_mm:.1f},forward_mm={plan.forward_distance_mm:.1f},"
                    f"retreat_mm={self._breakup_retreat_mm:.1f},penetration_mm={plan.penetration_mm:.1f},"
                    f"actual_mm=({self._breakup_actual_forward_mm:.1f},{self._breakup_actual_backward_mm:.1f}),"
                    f"sweep_half_width_mm={plan.sweep_half_width_mm:.1f},"
                    f"age_ms={(timestamp_ns-plan.capture_timestamp_ns)/1e6:.1f},"
                    f"grasp_rejections={plan.rejection_reasons}")
        if (
            self._cluster_reference is not None
            and self._cluster_reference_distance_mm is not None
        ):
            return (
                "reference_locked="
                f"({self._cluster_reference.x:.1f},"
                f"{self._cluster_reference.y:.1f}),"
                f"nearest_forward_x_mm={self._cluster_reference_distance_mm:.1f}"
            )
        measurement = self._cluster_ground_measurement(timestamp_ns)
        if measurement is None:
            return "none"
        return (
            f"center_xy_mm=({measurement.center.x:.1f},{measurement.center.y:.1f}),"
            f"nearest_forward_x_mm={measurement.nearest_forward_x_mm:.1f}"
        )

    def _find_opportunistic_single_green(
        self,
        timestamp_ns: int,
    ) -> TrackedTarget | None:
        """寻找可直接转运的目标入口种子。

        首轮只选择最近的单绿。一般阶段绿/黑可按走廊成组；橙色伤员
        只有在固定走廊内不与任何其它目标混合时才作为单目标入口。
        """

        candidates = tuple(
            (target, group_size)
            for target in self._tracker.tracks
            if self._target_is_fresh(target, timestamp_ns)
            and not self._target_attempt_blocked(target, timestamp_ns)
            for group_size in (
                self._transport_group_size(target, timestamp_ns)
                if self._near_field_pickup is not None
                else (
                    1
                    if self._opportunistic_single_green_is_clear(
                        target,
                        timestamp_ns,
                    )
                    else None
                ),
            )
            if group_size is not None
        )
        first_single_green = self._transport_count == 0
        best = min(
            candidates,
            key=lambda item: (
                (
                    math.hypot(item[0].ground_point.x, item[0].ground_point.y)
                    if item[0].ground_point is not None
                    else math.inf,
                    abs(item[0].ground_point.y)
                    if item[0].ground_point is not None
                    else math.inf,
                    item[0].track_id,
                )
                if first_single_green
                else (
                    -item[1],
                    math.hypot(item[0].ground_point.x, item[0].ground_point.y)
                    if item[0].ground_point is not None
                    else math.inf,
                    abs(item[0].ground_point.y)
                    if item[0].ground_point is not None
                    else math.inf,
                    item[0].track_id,
                )
            ),
            default=None,
        )
        return None if best is None else best[0]

    def _first_green_forward_corridor_blocked(self, timestamp_ns: int) -> bool:
        """检查首轮最近绿块的固定前向走廊是否有禁止目标。"""

        if (
            not self._first_green_blocked_routes_to_breakup
            or self._near_field_pickup is None
            or self._transport_count != 0
        ):
            return False
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if self._target_is_fresh(target, timestamp_ns)
            and not self._target_attempt_blocked(target, timestamp_ns)
            and target.target_class is TargetClass.GREEN_SUPPLY
            and self._graspable_target_is_usable(target)
            and target.ground_point is not None
            and target.ground_point.x > 0.0
            and self._point_in_transport_corridor(target.ground_point)
        )
        nearest = min(
            candidates,
            key=lambda target: (
                math.hypot(target.ground_point.x, target.ground_point.y),
                abs(target.ground_point.y),
                target.track_id,
            ),
            default=None,
        )
        if nearest is None or nearest.ground_point is None:
            return False
        forbidden = {
            TargetClass.BLUE_DANGER,
            TargetClass.ORANGE_INJURED,
            TargetClass.BLACK_CORE,
        }
        for other in self._tracker.tracks:
            if (
                other.track_id == nearest.track_id
                or other.track_id in self._green_preclose_consumed_track_ids
                or not self._target_is_fresh(other, timestamp_ns)
            ):
                continue
            if other.ground_point is None:
                continue
            if not self._point_in_transport_corridor(
                other.ground_point,
                end_x_mm=nearest.ground_point.x,
            ):
                continue
            if other.target_class in forbidden:
                return True
        return False

    def _preview_ignored_track_ids(self, target: TrackedTarget, timestamp_ns: int,
                                   *, tracks: tuple[TrackedTarget, ...] | None = None) -> frozenset[int]:
        # 只有普通物资组可以共同收拢；伤员路径上的任何其它物块都不能忽略。
        if (target.target_class not in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE}
                or (self._dynamic_breakup_enabled and self._transport_count == 0)):
            return frozenset()
        return frozenset(other.track_id for other in (self._tracker.tracks if tracks is None else tracks)
                         if other.track_id != target.track_id
                         and self._target_is_fresh(other, timestamp_ns)
                         and self._graspable_target_is_usable(other)
                         and other.target_class in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE})

    def _find_approach_seed(
        self,
        timestamp_ns: int,
        *,
        minimum_last_seen_ns: int | None = None,
        prefer_nearest: bool = False,
    ) -> TrackedTarget | None:
        """寻找远距接近或近场交接的入口目标；不负责最终选组。

        ``minimum_last_seen_ns`` 供补夹扫描限制为扫描开始后的新观测；
        ``prefer_nearest`` 供补夹在全场候选中优先最近的绿/黑物块。
        """

        policy = self.near_field_policy
        stowed_limit = (
            self._greedy_new_target_min_x_mm()
            if self._greedy_active
            else None
        )
        current_points = {
            target.track_id: self._current_ground_point_for_track(target, timestamp_ns)
            for target in self._tracker.tracks
        }
        current_tracks = tuple(
            replace(target, ground_point=current_points[target.track_id])
            for target in self._tracker.tracks
        )
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if self._approach_seed_rejection(
                target,
                current_points[target.track_id],
                timestamp_ns,
                minimum_last_seen_ns=minimum_last_seen_ns,
                stowed_limit=stowed_limit,
                policy=policy,
                current_tracks=current_tracks,
            ) is None
        )
        if prefer_nearest:
            return min(
                candidates,
                key=lambda target: (
                    math.hypot(current_points[target.track_id].x, current_points[target.track_id].y),
                    abs(current_points[target.track_id].y),
                    target.track_id,
                ),
                default=None,
            )
        return min(
            candidates,
            key=lambda target: (
                math.hypot(current_points[target.track_id].x, current_points[target.track_id].y) > self._near_field_handoff_range_mm(),
                -({TargetClass.GREEN_SUPPLY: self._near_field_grasp_config.green_score_points,
                   TargetClass.BLACK_CORE: self._near_field_grasp_config.black_score_points,
                   TargetClass.ORANGE_INJURED: self._near_field_grasp_config.orange_score_points}[target.target_class]
                  if self._near_field_grasp_config is not None else 0),
                math.hypot(current_points[target.track_id].x, current_points[target.track_id].y),
                abs(current_points[target.track_id].y),
                target.track_id,
            ),
            default=None,
        )

    def _approach_seed_rejection(
        self,
        target: TrackedTarget,
        current: GroundPoint | None,
        timestamp_ns: int,
        *,
        minimum_last_seen_ns: int | None,
        stowed_limit: float | None,
        policy: NearFieldGraspPolicy,
        current_tracks: tuple[TrackedTarget, ...],
    ) -> str | None:
        """返回入口候选被淘汰的第一个门禁名；``None`` 表示可以作为种子。

        与 ``_find_approach_seed`` 共用同一套判据，现场的淘汰诊断因此不会
        与真实筛选逻辑脱节。
        """

        if not self._target_is_fresh(target, timestamp_ns):
            return "stale"
        if (minimum_last_seen_ns is not None
                and target.last_seen_timestamp_ns <= minimum_last_seen_ns):
            return "before_scan_start"
        if self._target_attempt_blocked(target, timestamp_ns):
            return "attempt_blocked"
        if (self._dynamic_breakup_enabled
                and target.track_id in self._breakup_rejected_grasp_ids):
            return "breakup_rejected"
        if target.target_class not in policy.allowed_classes:
            return f"class_not_in_policy:{target.target_class.value}"
        if not self._graspable_target_is_usable(target):
            return "not_usable"
        if target.ground_point is None:
            return "missing_ground_point"
        if current is None:
            return "missing_current_point"
        if current.x <= 0.0:
            return "behind_robot"
        if stowed_limit is not None and current.x <= stowed_limit:
            return "inside_stowed_limit"
        if self._candidate_path_blocked(current, breakup=False):
            return "path_blocked"
        if self._transport_side_neighbor_target(target, timestamp_ns) is not None:
            return "side_neighbor"
        if not self._transport_orange_isolation_clear(target, timestamp_ns):
            return "orange_not_isolated"
        if not self._green_path_is_clear_for_point(
            target, current, timestamp_ns,
            ignored_track_ids=self._preview_ignored_track_ids(target, timestamp_ns),
            tracks=current_tracks,
        ):
            return "corridor_blocked"
        return None

    def approach_seed_diagnostic(self, timestamp_ns: int) -> str:
        """列出每个新鲜目标被入口门禁淘汰的原因，供现场区分"没看见"与"被否决"。

        远场入口此前没有任何等价于近场 ``grasp_candidate`` 的淘汰日志，
        现场只能看到"车看见了却不靠近"。
        """

        policy = self.near_field_policy
        stowed_limit = (
            self._greedy_new_target_min_x_mm()
            if self._greedy_active
            else None
        )
        current_points = {
            target.track_id: self._current_ground_point_for_track(target, timestamp_ns)
            for target in self._tracker.tracks
        }
        current_tracks = tuple(
            replace(target, ground_point=current_points[target.track_id])
            for target in self._tracker.tracks
        )
        entries = []
        for target in self._tracker.tracks:
            if not self._target_is_fresh(target, timestamp_ns):
                continue
            rejection = self._approach_seed_rejection(
                target,
                current_points[target.track_id],
                timestamp_ns,
                minimum_last_seen_ns=None,
                stowed_limit=stowed_limit,
                policy=policy,
                current_tracks=current_tracks,
            )
            entries.append(
                f"track={target.track_id},{target.target_class.value},"
                f"{rejection or 'accepted'}"
            )
        return "none" if not entries else ";".join(entries)

    def _find_far_reapproach_target(
        self,
        timestamp_ns: int,
        *,
        tracks: tuple[TrackedTarget, ...] | None = None,
    ) -> TrackedTarget | None:
        """选择近场不可直接执行时仍值得远距接近的目标。"""

        policy = self.near_field_policy
        max_range = self._near_field_handoff_range_mm()
        source_tracks = self._tracker.tracks if tracks is None else tuple(tracks)
        candidates = []
        for target in source_tracks:
            point = target.ground_point
            if (
                not self._target_is_fresh(target, timestamp_ns)
                or self._target_attempt_blocked(target, timestamp_ns)
                or target.target_class not in policy.allowed_classes
                or not self._graspable_target_is_usable(target)
                or point is None
                or point.x <= 0.0
                or math.hypot(point.x, point.y) <= max_range
                or self._candidate_path_blocked(point, breakup=False)
                or self._transport_side_neighbor_target(
                    target,
                    timestamp_ns,
                    tracks=source_tracks,
                )
                is not None
                or not self._transport_orange_isolation_clear(
                    target,
                    timestamp_ns,
                    tracks=source_tracks,
                )
                or not self._green_path_is_clear_for_point(
                    target, point, timestamp_ns, tracks=source_tracks,
                    ignored_track_ids=self._preview_ignored_track_ids(target, timestamp_ns, tracks=source_tracks),
                )
            ):
                continue
            candidates.append(target)
        return min(
            candidates,
            key=lambda target: (
                math.hypot(target.ground_point.x, target.ground_point.y),
                abs(math.atan2(target.ground_point.y, target.ground_point.x)),
                target.track_id,
            ),
            default=None,
        )

    def _first_green_corridor_has_forbidden_target(
        self,
        rejection_reasons: tuple[str, ...],
    ) -> bool:
        """判断首轮最近单绿是否因前进走廊中的目标被拒绝。"""

        if (
            not self._first_green_blocked_routes_to_breakup
            or self._transport_count != 0
        ):
            return False
        forbidden_classes = {
            TargetClass.BLUE_DANGER.value,
            TargetClass.ORANGE_INJURED.value,
            TargetClass.BLACK_CORE.value,
        }
        return any(
            reason.startswith("blocked_target:")
            and reason.rsplit(":", 1)[-1] in forbidden_classes
            for reason in rejection_reasons
        )

    @staticmethod
    def _side_neighbor_has_no_safe_plan(
        rejection_reasons: tuple[str, ...],
    ) -> bool:
        """判断当前候选是否因蓝/橙侧邻而没有安全方案。"""

        return any(
            reason.startswith("side_adjacent_incompatible")
            for reason in rejection_reasons
        )

    def _near_field_handoff_target_missing(
        self,
        preparation: GraspPreparation | None,
    ) -> bool:
        """判断补夹交接目标是否在锁定执行计划前真正消失。"""

        if (
            not self._greedy_active
            or self._near_field_pickup is None
            or self._near_field_pickup.locked_ids is not None
            or self._near_field_handoff_prior is None
            or preparation is None
            or "handoff_target_missing" not in preparation.selection.rejections
        ):
            return False
        # A visible handoff target that merely lost its envelope or path
        # safety must not be replaced.  Only an unobserved handoff permits the
        # next supplementary target to take over.
        return not any(
            target.handoff_matched and target.observed
            for target in preparation.targets
        )

    def _near_field_route_decision(
        self,
        timestamp_ns: int,
        preparation: GraspPreparation | None,
    ) -> GraspRouteDecision | None:
        """在一次有界确认窗口内做近场、重接近或重选分流。"""

        if self._near_field_confirmation_started_ns is None:
            self._near_field_confirmation_started_ns = timestamp_ns
        elapsed_ms = max(
            0.0,
            (timestamp_ns - self._near_field_confirmation_started_ns) / 1_000_000.0,
        )
        selection = None if preparation is None else preparation.selection
        candidate_count = 0 if preparation is None else len(preparation.targets)
        self._near_field_route_candidate_count = candidate_count
        plan = None if selection is None else selection.plan
        rejection_reasons = () if selection is None else selection.rejections
        first_green_corridor_forbidden = (
            self._first_green_corridor_has_forbidden_target(rejection_reasons)
        )
        side_neighbor_no_safe_plan = (
            plan is None
            and self._side_neighbor_has_no_safe_plan(rejection_reasons)
        )
        geometric_block = (
            plan is None
            and (
                first_green_corridor_forbidden
                or side_neighbor_no_safe_plan
                or any(
                    (
                        reason.startswith("blocked_target:")
                        or (
                            self._dynamic_breakup_enabled
                            and (
                                reason
                                in {
                                    "maximum_opening_exceeded",
                                }
                                # 对准后仍够不到的目标（夹爪末端越不过中线、
                                # 开口物理不够）不是数据问题，等下去也不会好，
                                # 直接进入解团把目标分开再夹。
                                or reason.startswith(
                                    ("left_tip_y_mm", "right_tip_y_mm")
                                )
                            )
                        )
                    )
                    for reason in rejection_reasons
                )
            )
        )
        if geometric_block and self._dynamic_breakup_enabled and self._transport_count == 0:
            previous = self._selected_track_id
            if previous is None and self._near_field_handoff_prior is not None:
                previous = self._near_field_handoff_prior.source_track_id
            if previous is not None:
                self._breakup_rejected_grasp_ids.add(previous)
            # 交接已经清空主 tracker 的选中 ID。必须在丢弃 prior、开启下一
            # 会话前记录物理失败核心，否则同一目标会立刻再次成为“替代目标”。
            self._record_near_field_route_failure(
                GraspRouteDecision(
                    GraspRoute.BREAKUP,
                    candidate_count=candidate_count,
                    rejection_reasons=rejection_reasons,
                    elapsed_ms=elapsed_ms,
                ),
                preparation,
                kind="geometric_block_before_alternative",
                timestamp_ns=timestamp_ns,
            )
            alternative = self._find_approach_seed(timestamp_ns)
            if alternative is not None:
                return GraspRouteDecision(GraspRoute.FAR_REAPPROACH, target=alternative,
                                          candidate_count=candidate_count, rejection_reasons=rejection_reasons,
                                          elapsed_ms=elapsed_ms)
        if geometric_block:
            self._near_field_route = GraspRoute.BREAKUP
            self._near_field_route_elapsed_ms = elapsed_ms
            self._near_field_route_rejections = rejection_reasons
            route = GraspRouteDecision(
                GraspRoute.BREAKUP,
                candidate_count=candidate_count,
                rejection_reasons=rejection_reasons,
                elapsed_ms=elapsed_ms,
            )
            self._record_near_field_route_failure(
                route,
                preparation,
                kind=(
                    "first_green_corridor_forbidden_target"
                    if first_green_corridor_forbidden
                    else "side_neighbor_incompatible"
                    if side_neighbor_no_safe_plan
                    else "geometric_blocked_target"
                ),
                timestamp_ns=timestamp_ns,
            )
            return route
        if plan is not None:
            self._near_field_route = GraspRoute.DIRECT_NEAR
            self._near_field_route_elapsed_ms = elapsed_ms
            self._near_field_route_rejections = rejection_reasons
            # 近场选择结果会在同一目标 ID 上持续更新几何；handoff prior
            # 只用于首轮把远场目标交给局部 tracker。
            return GraspRouteDecision(
                GraspRoute.DIRECT_NEAR,
                plan=plan,
                candidate_count=candidate_count,
                rejection_reasons=rejection_reasons,
                elapsed_ms=elapsed_ms,
            )
        # 无方案等待用独立的短预算：车已停稳、场景静止，重复观测不会改变
        # 门禁结果，等满提交预算只是空转。提交窗口的预算见
        # ``_near_field_handoff_timeout_ms``，只用于停稳后当前帧的提交。
        if (elapsed_ms < self._near_field_no_plan_wait_ms()
                or (self.grasp_planning_pending and elapsed_ms < self._near_field_handoff_timeout_ms())):
            return None
        if self._near_field_far_reapproach_used:
            self._near_field_route = GraspRoute.RESELECT
            self._near_field_route_elapsed_ms = elapsed_ms
            self._near_field_route_rejections = rejection_reasons
            route = GraspRouteDecision(
                GraspRoute.RESELECT,
                candidate_count=candidate_count,
                rejection_reasons=rejection_reasons,
                elapsed_ms=elapsed_ms,
            )
            self._record_near_field_route_failure(
                route,
                preparation,
                kind="confirmation_timeout_after_reapproach",
                timestamp_ns=timestamp_ns,
            )
            return route
        far_target = (
            None if self._greedy_active
            else self._find_far_reapproach_target(timestamp_ns)
        )
        if far_target is not None and self._transport_count > 0:
            self._near_field_far_reapproach_used = True
            self._near_field_route = GraspRoute.FAR_REAPPROACH
            self._near_field_route_elapsed_ms = elapsed_ms
            self._near_field_route_rejections = rejection_reasons
            return GraspRouteDecision(
                GraspRoute.FAR_REAPPROACH,
                target=far_target,
                candidate_count=candidate_count,
                rejection_reasons=self._near_field_route_rejections,
                elapsed_ms=elapsed_ms,
            )
        self._near_field_route = GraspRoute.RESELECT
        self._near_field_route_elapsed_ms = elapsed_ms
        self._near_field_route_rejections = rejection_reasons
        route = GraspRouteDecision(
            GraspRoute.RESELECT,
            candidate_count=candidate_count,
            rejection_reasons=rejection_reasons,
            elapsed_ms=elapsed_ms,
        )
        self._record_near_field_route_failure(
            route,
            preparation,
            kind="confirmation_timeout_reselect",
            timestamp_ns=timestamp_ns,
        )
        return route

    def _near_field_handoff_timeout_ms(self) -> float:
        config = self._near_field_grasp_config
        return (
            NearFieldGraspConfig().alignment_timeout_ms
            if config is None
            else float(config.alignment_timeout_ms)
        ) + (0.0 if self._near_field_pickup is None else self._near_field_pickup.alignment_motion_allowance_ns / 1e6)

    def _near_field_no_plan_wait_ms(self) -> float:
        """完全选不出方案时的等待上限；不影响停稳后当前帧的提交预算。"""

        if self._near_field_no_plan_budget_ms is not None:
            return self._near_field_no_plan_budget_ms
        config = self._near_field_grasp_config
        # 近场未启用时本路径不可达；与 NearFieldGraspConfig 默认值保持一致。
        return (
            NearFieldGraspConfig().no_plan_wait_ms
            if config is None
            else float(config.no_plan_wait_ms)
        )

    def _near_field_has_executable_plan(self) -> bool:
        """会话是否已经持有可执行计划（已定路由或已锁定成员）。

        ``no_plan_wait_ms`` 是"完全没有方案"的短窗口。已经定下路由或锁定
        成员的会话并不缺方案，只是当前帧的准备结果还没跟上，属于提交阶段，
        应当由 ``_near_field_handoff_timeout_ms()`` 的提交窗口约束。
        """

        if self._near_field_route is not GraspRoute.DECIDING:
            return True
        pickup = self._near_field_pickup
        return pickup is not None and pickup.locked_ids is not None

    @staticmethod
    def _near_field_confirmation_progress(
        preparation: GraspPreparation | None,
    ) -> str:
        if preparation is None:
            return "none"
        return f"{preparation.confirmation_count}/{preparation.confirmation_required}"

    @staticmethod
    def _preparation_observation_age_ms(
        preparation: GraspPreparation | None,
        timestamp_ns: int,
    ) -> str:
        if preparation is None or timestamp_ns < preparation.capture_timestamp_ns:
            return "none"
        return f"{(timestamp_ns - preparation.capture_timestamp_ns) / 1_000_000.0:.1f}"

    @staticmethod
    def _preparation_result_age_ms(
        preparation: GraspPreparation | None,
        timestamp_ns: int,
    ) -> str:
        if preparation is None:
            return "none"
        prepared = preparation.prepared_timestamp_ns or preparation.result_timestamp_ns
        if prepared is None or timestamp_ns < prepared:
            return "none"
        return f"{(timestamp_ns - prepared) / 1_000_000.0:.1f}"

    def _failure_region_points(
        self,
        timestamp_ns: int,
        preparation: GraspPreparation | None,
    ) -> tuple[FieldPoint, ...]:
        """Return physical field points for the failed action core/region."""

        points: list[FieldPoint] = []
        for member in self._failure_core_members(preparation):
            ground = member.observation.ground_point
            if ground is None:
                continue
            current = self._ground_point_at_now(
                ground,
                member.observation.capture_timestamp_ns,
                timestamp_ns,
            )
            if current is None:
                current = ground
            field = self._field_point_from_ground(current)
            if field is not None:
                points.append(field)
        # 计划成员已经给出物理核心时不再按主 tracker ID 追加第二个点：
        # 两个编号空间不同，混入的 ID 点会无故扩大被封锁区域。
        target = None if points else self._selected_target()
        if target is not None:
            current = self._current_ground_point_for_track(target, timestamp_ns)
            if current is not None:
                field = self._field_point_from_ground(current)
                if field is not None:
                    points.append(field)
        if not points and self._near_field_handoff_prior is not None:
            field = self._field_point_from_ground(
                self._near_field_handoff_prior.ground_point
            )
            if field is not None:
                points.append(field)
        unique: list[FieldPoint] = []
        for point in points:
            if not any(
                math.hypot(point.x - existing.x, point.y - existing.y) <= 1e-6
                for existing in unique
            ):
                unique.append(point)
        return tuple(unique)

    def _failure_core_members(
        self,
        preparation: GraspPreparation | None,
    ) -> tuple[GraspTarget, ...]:
        """返回本次尝试实际指向的物理核心成员。

        主 tracker 与近场会话各自编号，失败记录必须绑定计划成员的当前
        几何；按 ID 回查主 tracker 可能取到另一个物理目标。
        """

        if self._grasp_task is not None and self._grasp_task.core:
            return self._grasp_task.core
        plan = None if self._near_field_pickup is None else self._near_field_pickup.active_plan
        if plan is None and preparation is not None:
            plan = preparation.selection.plan
        return () if plan is None else plan.members

    def _remember_near_field_failure(
        self,
        timestamp_ns: int,
        reason: str,
        preparation: GraspPreparation | None = None,
    ) -> None:
        """Record a failure so a new session cannot retry unchanged geometry."""

        task = self._grasp_task
        if task is not None:
            if task.failure_recorded_revision == task.scene_revision:
                return
            task.failure_recorded_revision = task.scene_revision
        core = self._failure_core_members(preparation)
        relative = None
        target_class = None
        track_id = None
        if core:
            member = core[0]
            target_class = member.observation.target_class
            relative = member.observation.ground_point
            current = (
                None
                if relative is None
                else self._ground_point_at_now(
                    relative,
                    member.observation.capture_timestamp_ns,
                    timestamp_ns,
                )
            )
            relative = current if current is not None else relative
            # 几何与 ID 取自同一物理成员：近场局部 ID 与主 tracker ID 不同源，
            # 混用会让诊断指向另一个目标。
            track_id = member.track_id
        else:
            target = self._selected_target()
            if target is not None:
                target_class = target.target_class
                track_id = target.track_id
                relative = self._current_ground_point_for_track(target, timestamp_ns)
        if target_class is None and self._near_field_handoff_prior is not None:
            target_class = self._near_field_handoff_prior.target_class
            relative = self._near_field_handoff_prior.ground_point
            track_id = self._near_field_handoff_prior.source_track_id
        region_points = self._failure_region_points(timestamp_ns, preparation)
        if task is not None:
            if task.entry_field is not None:
                region_points = tuple(dict.fromkeys((*region_points, task.entry_field)))
            target_class = task.entry_class
            track_id = task.entry_track_id
            relative = task.entry_ground
        position = self.estimated_field_position
        field_point = (task.entry_field if task is not None else
                       None if relative is None else self._field_point_from_ground(relative))
        record = _AttemptFailureRecord(
            target_class,
            relative,
            field_point,
            position,
            self._latest_heading_rad,
            track_id,
            reason,
            timestamp_ns,
            region_points,
        )
        if self._near_field_failures:
            previous = self._near_field_failures[-1]
            if (
                previous.reason == reason
                and previous.field_point == field_point
                and timestamp_ns - previous.timestamp_ns < 1_000_000_000
            ):
                return
        self._near_field_failures.append(record)
        del self._near_field_failures[:-32]
        self._near_field_last_failure_diagnostic = (
            (self._near_field_last_failure_diagnostic or "grasp_task_failure")
            + f" reason={reason} {self.grasp_task_diagnostic(timestamp_ns)} "
            + self._stationary_motion.diagnostic(timestamp_ns)
            + f" capture_age_ms={self._preparation_observation_age_ms(preparation, timestamp_ns)}"
            + f" preparation_age_ms={self._preparation_result_age_ms(preparation, timestamp_ns)}"
            + f" confirmation={self._near_field_confirmation_progress(preparation)}"
        )
        region_text = "none" if not region_points else ";".join(
            f"({point.x:.1f},{point.y:.1f})" for point in region_points
        )
        if self._near_field_last_failure_diagnostic is not None:
            self._near_field_last_failure_diagnostic += (
                f" track_id={track_id if track_id is not None else 'none'}"
                f" target_class={target_class.value if target_class is not None else 'none'}"
                f" region_field={region_text}"
                f" failure_records={len(self._near_field_failures)}"
            )

    def _target_attempt_blocked(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
    ) -> bool:
        """Reject only unchanged physical attempts, independent of tracker ID."""

        current = self._current_ground_point_for_track(target, timestamp_ns)
        if current is None:
            return False
        current_field = self._field_point_from_ground(current)
        for failure in self._near_field_failures:
            same_region = False
            if current_field is not None and failure.region_field_points:
                same_region = any(
                    math.hypot(
                        current_field.x - point.x,
                        current_field.y - point.y,
                    )
                    <= self._failure_region_tolerance_mm()
                    for point in failure.region_field_points
                )
            elif current_field is not None and failure.field_point is not None:
                same_region = math.hypot(
                    current_field.x - failure.field_point.x,
                    current_field.y - failure.field_point.y,
                ) <= self._failure_region_tolerance_mm()
            elif failure.relative_point is not None:
                same_region = math.hypot(
                    current.x - failure.relative_point.x,
                    current.y - failure.relative_point.y,
                ) <= self._failure_region_tolerance_mm()
            if not same_region:
                continue
            geometry_changed = (
                current_field is not None and failure.field_point is not None
                and math.hypot(current_field.x - failure.field_point.x,
                               current_field.y - failure.field_point.y) >= 100.0
            )
            current_position = self.estimated_field_position
            if (
                not geometry_changed
                and failure.robot_position is not None
                and current_position is not None
            ):
                geometry_changed = math.hypot(
                    current_position.x - failure.robot_position.x,
                    current_position.y - failure.robot_position.y,
                ) >= 100.0
            if not geometry_changed:
                return True
        return False

    def _record_near_field_route_failure(
        self,
        route: GraspRouteDecision,
        preparation: GraspPreparation | None,
        *,
        kind: str,
        timestamp_ns: int | None = None,
    ) -> None:
        """保留近场路由失败的细节，供状态切换后的终端日志输出。"""

        targets = () if preparation is None else preparation.targets
        matched_ids = tuple(
            target.track_id for target in targets if target.handoff_matched
        )
        prior = self._near_field_handoff_prior
        prior_track = "none" if prior is None else str(prior.source_track_id)
        prior_class = "none" if prior is None else prior.target_class.value
        rejection = "|".join(route.rejection_reasons) or "none"
        locked_ids = (
            "none"
            if preparation is None or preparation.checked_member_ids is None
            else ",".join(str(item) for item in preparation.checked_member_ids)
        )
        attempt_deadline_ns = (
            None
            if self._near_field_confirmation_started_ns is None
            else self._near_field_confirmation_started_ns
            + round(self._near_field_handoff_timeout_ms() * 1_000_000.0)
        )
        failure_timestamp_ns = (
            (self._last_timestamp_ns or 0)
            if timestamp_ns is None
            else timestamp_ns
        )
        self._near_field_last_failure_diagnostic = (
            "near_field_confirmation_failure "
            f"kind={kind} session={self._near_field_session_id} "
            f"elapsed_ms={route.elapsed_ms:.1f} "
            f"candidates={route.candidate_count} "
            f"live_targets={len(targets)} locked_ids={locked_ids} "
            f"handoff_prior_track={prior_track} "
            f"handoff_prior_class={prior_class} "
            f"handoff_prior_match_ids="
            f"{','.join(str(item) for item in matched_ids) or 'none'} "
            f"rejections={rejection} "
            f"confirmation_progress={self._near_field_confirmation_progress(preparation)} "
            f"observation_age_ms={self._preparation_observation_age_ms(preparation, failure_timestamp_ns)} "
            f"preparation_age_ms={self._preparation_result_age_ms(preparation, failure_timestamp_ns)} "
            f"attempt_deadline_ns={attempt_deadline_ns if attempt_deadline_ns is not None else 'none'}"
        )
        self._remember_near_field_failure(
            failure_timestamp_ns,
            kind,
            preparation,
        )

    def _breakup_plan_for_this_frame(self, timestamp_ns: int) -> BreakupPlan | None:
        """同一输入帧最多求一次接触计划，逐控制周期重复跑边界搜索不改变结果。"""

        perception = self._latest_perception
        key = (
            None
            if perception is None
            else (perception.frame_sequence, perception.capture_timestamp_ns)
        )
        if key is not None and key == self._breakup_search_frame:
            return None
        self._breakup_search_frame = key
        return self._choose_breakup_plan(timestamp_ns)

    def _enter_breakup_only_search(self, timestamp_ns: int) -> MatchDecision:
        """近场没有安全方案时只进入解团搜索，禁止同帧重入机会抓取。"""

        if self._greedy_active or self._transport_target_classes:
            return self._finish_greedy_pickup(timestamp_ns, "no_direct_plan")
        if self._grasp_task is not None and self._grasp_task.recovery_count >= self.config.breakup_max_attempts:
            return self._return_to_near_field_search(timestamp_ns, "recovery_budget_exhausted")
        if self.near_field_enabled and (self._grasp_task is not None or self.state is not MatchState.TRANSPORT_NEAR_FIELD_GRASP):
            return self._begin_near_field_grasp(timestamp_ns, handoff_prior=self._near_field_handoff_prior or self._selected_handoff_prior(timestamp_ns))
        plan = (
            self._breakup_plan_for_this_frame(timestamp_ns)
            if self._dynamic_breakup_enabled
            else None
        )
        if self._dynamic_breakup_enabled and plan is None:
            # 当前帧已经明确选不出合法接触计划：停车、停稳和重观测窗口都不会
            # 改变这个结论，按失败记忆带原因重选，不为不可能的动作停下。
            return self._return_to_near_field_search(
                timestamp_ns,
                "breakup_no_contact_plan",
            )
        if (
            self._near_field_last_failure_diagnostic is None
            or f"session={self._near_field_session_id}" not in self._near_field_last_failure_diagnostic
        ):
            self._near_field_last_failure_diagnostic = (
                "near_field_route_failure "
                f"kind=near_field_geometric_block_breakup session={self._near_field_session_id}"
            )
        self._remember_near_field_failure(
            timestamp_ns,
            "near_field_geometric_block_breakup",
            self._breakup_grasp_preparation,
        )
        if self._dynamic_breakup_enabled and self._selected_track_id is not None:
            self._breakup_rejected_grasp_ids.add(self._selected_track_id)
        self._near_field_route = GraspRoute.BREAKUP
        self._near_field_handoff_prior = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id += 1
        self._near_field_far_reapproach_used = False
        self._selected_track_id = None
        self._selected_green_ground = None
        self._begin_cluster_search()
        self._breakup_only = True
        self.state = MatchState.SEARCH_CLUSTER
        if self._dynamic_breakup_enabled:
            assert plan is not None
            self._start_breakup_attempt(timestamp_ns, plan)
            self._breakup_only = True
            return self._decision(timestamp_ns, 0.0, 0.0, "near_field_route:breakup", posture=GripperPosture.CLOSED, soft_brake=True)
        return self._decision(
            timestamp_ns,
            0.0,
            self._cluster_search_angular_velocity_rad_s,
            "near_field_route:breakup",
            posture=GripperPosture.CLOSED,
        )

    def _return_to_near_field_search(
        self,
        timestamp_ns: int,
        reason: str,
    ) -> MatchDecision:
        """数据/确认预算耗尽后回到带原因的搜索，不伪装成解团阻挡。"""

        if self._gate_clearance is not None and self._gate_clearance.reacquiring:
            # 回取失败后结束本次清障任务；不重开普通近场会话，否则会解除
            # 暂存点类别/数量约束并可能夹取其它目标。
            self._gate_clearance = None
            self._grasp_task = None
            self._near_field_handoff_prior = None
            if self._near_field_pickup is not None:
                self._near_field_pickup.reset()
                self._near_field_session_id += 1
            self._near_field_route = GraspRoute.RESELECT
            self._near_field_confirmation_started_ns = None
            self._selected_track_id = None
            self._selected_green_ground = None
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(
                timestamp_ns,
                0.0,
                self._cluster_search_angular_velocity_rad_s,
                f"gate_clearance_reacquire_failed:{reason}",
                posture=GripperPosture.OPEN,
            )
        if self._greedy_active:
            return self._finish_greedy_pickup(timestamp_ns, reason)
        if self._reobserve_without_motion(reason):
            # 还没有发出过任何运动指令，交接 prior 与任务都保留，只换会话号
            # 重新观察；不消耗物理失败记忆，也就不需要靠换位解锁同一片区域。
            return replace(
                self._begin_near_field_grasp(
                    timestamp_ns,
                    handoff_prior=(self._near_field_handoff_prior
                                   or self._selected_handoff_prior(timestamp_ns)),
                ),
                reason=f"near_field_reobserve_without_motion:{reason}",
            )
        self._remember_near_field_failure(
            timestamp_ns,
            reason,
            self._breakup_grasp_preparation,
        )
        if self._near_field_last_failure_diagnostic is not None:
            self._near_field_last_failure_diagnostic += " next_action=select_other_target_or_checked_relocation"
        self._grasp_task = None
        self._near_field_handoff_prior = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id += 1
        self._near_field_route = GraspRoute.RESELECT
        self._near_field_route_candidate_count = 0
        self._near_field_confirmation_started_ns = None
        self._near_field_far_reapproach_used = False
        self._selected_track_id = None
        self._selected_green_ground = None
        self._breakup_only = False
        alternative = self._find_approach_seed(timestamp_ns)
        if alternative is not None:
            return replace(self._begin_green_transport(timestamp_ns, alternative, group_preview=True),
                           reason=f"near_field_exit_approach_alternative:{reason}:{alternative.track_id}")
        self._begin_cluster_search()
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(
            timestamp_ns,
            0.0,
            self._cluster_search_angular_velocity_rad_s,
            f"near_field_route:reselect:{reason}",
            posture=GripperPosture.CLOSED,
        )

    def _reobserve_without_motion(self, reason: str) -> bool:
        """已定方案的会话丢帧时最多原地重观测一次，不写物理失败记忆。

        条件严格限定为"会话已经定下路由或锁定成员、但这次尝试没有发出过任何
        运动指令"。若按普通失败处理，会先把这片区域记成物理失败，再清掉交接
        prior、锁定目标和任务，必须转完一整圈并换位 300 mm 才能解锁，而重观测
        本身只需要一帧。完全没有方案时仍按 `CLAUDE.md` 的要求直接退出，不在此
        重试。同一片区域连续两次都拿不到确凿证据也照原路退出，不做无期限等待。
        """

        if reason not in self._NEAR_FIELD_REOBSERVE_REASONS:
            return False
        if not self._near_field_has_executable_plan():
            return False
        if self._near_field_reobserve_retries >= self._NEAR_FIELD_REOBSERVE_LIMIT:
            return False
        self._near_field_reobserve_retries += 1
        self._near_field_last_failure_diagnostic = (
            "near_field_reobserve_without_motion "
            f"reason={reason} attempt={self._near_field_reobserve_retries} "
            f"session={self._near_field_session_id} "
            "next_action=reobserve_same_region"
        )
        return True

    def _route_after_near_field_failure(
        self,
        timestamp_ns: int,
        *,
        reason: str = "near_field_action_failed",
        geometric_block: bool = False,
    ) -> MatchDecision:
        """按失败证据选择解团、一次重接近或带原因重选。"""

        if self._greedy_active:
            return self._finish_greedy_pickup(timestamp_ns, reason)
        if geometric_block:
            return self._enter_breakup_only_search(timestamp_ns)
        if self._near_field_far_reapproach_used:
            return self._return_to_near_field_search(
                timestamp_ns,
                f"{reason}_after_reapproach",
            )
        if self._transport_count > 0:
            far_target = self._find_far_reapproach_target(timestamp_ns)
            if far_target is not None:
                self._near_field_far_reapproach_used = True
                self._near_field_route = GraspRoute.FAR_REAPPROACH
                self._near_field_route_elapsed_ms = (
                    0.0
                    if self._near_field_confirmation_started_ns is None
                    else max(
                        0.0,
                        (
                            timestamp_ns
                            - self._near_field_confirmation_started_ns
                        )
                        / 1_000_000.0,
                    )
                )
                decision = self._begin_green_transport(
                    timestamp_ns,
                    far_target,
                    group_preview=True,
                )
                return replace(
                    decision,
                    reason=(
                        f"near_field_route:far_reapproach:"
                        f"{far_target.track_id}"
                    ),
                )
        return self._return_to_near_field_search(timestamp_ns, reason)

    def _opportunistic_single_green_is_clear(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        reference: GroundPoint | None = None,
    ) -> bool:
        """检查机会抓取的单物块门禁；正式近场路径使用固定 TRANSPORT。"""

        if not self.config.opportunistic_single_green_enabled:
            return False
        if self._near_field_pickup is not None:
            if not self._graspable_target_is_usable(target):
                return False
        elif not self._green_target_is_usable(target):
            return False
        if self._near_field_pickup is not None:
            return self._transport_single_green_is_clear(
                target,
                timestamp_ns,
                reference=reference,
            )
        point = target.ground_point if reference is None else reference
        if point is None or point.x <= 0.0:
            return False
        if not self._green_path_is_clear_for_point(target, point, timestamp_ns):
            return False
        for other in self._tracker.tracks:
            if (
                other.track_id == target.track_id
                or other.track_id
                in self._green_preclose_consumed_track_ids
                or not self._target_is_fresh(other, timestamp_ns)
            ):
                continue
            if other.ground_point is None:
                # _green_path_is_clear_for_point() already rejects this case;
                # keep the condition explicit for the singleton gate as well.
                return False
            if (
                math.hypot(
                    point.x - other.ground_point.x,
                    point.y - other.ground_point.y,
                )
                <= self.config.opportunistic_single_green_clearance_mm
            ):
                return False
        return True

    def _transport_corridor_effective_half_width_mm(self) -> float:
        """固定 ``TRANSPORT`` 姿态的走廊半宽，含配置的横向安全余量。"""

        margin = 0.0
        config = self._near_field_grasp_config
        if config is not None:
            margin = float(config.corridor_lateral_margin_mm)
        return self._transport_corridor_half_width_mm + margin

    def _transport_corridor_start_x_mm(self) -> float:
        config = self._near_field_grasp_config
        return 0.0 if config is None else float(config.corridor_start_x_mm)

    def _point_in_transport_corridor(
        self,
        point: GroundPoint,
        *,
        end_x_mm: float | None = None,
    ) -> bool:
        """判断 K0 是否落入当前机器人前向的固定 TRANSPORT 走廊。"""

        if not all(math.isfinite(value) for value in (point.x, point.y)):
            return False
        start_x_mm = self._transport_corridor_start_x_mm()
        if point.x < start_x_mm or abs(point.y) > self._transport_corridor_effective_half_width_mm():
            return False
        return end_x_mm is None or point.x <= end_x_mm

    def _transport_single_green_is_clear(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        reference: GroundPoint | None = None,
    ) -> bool:
        """兼容旧调用名；检查固定 TRANSPORT 走廊内的单绿/单黑入口组。"""

        return self._transport_group_size(
            target,
            timestamp_ns,
            reference=reference,
        ) is not None

    def _transport_orange_isolation_clear(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        tracks: tuple[TrackedTarget, ...] | None = None,
    ) -> bool:
        """检查正式入口的橙色目标 10 mm 独立性门禁。"""

        if target.target_class is not TargetClass.ORANGE_INJURED:
            return True
        point = target.ground_point
        if point is None:
            return False
        radius_mm = (
            10.0
            if self._near_field_grasp_config is None
            else self._near_field_grasp_config.orange_isolation_radius_mm
        )
        source_tracks = self._tracker.tracks if tracks is None else tuple(tracks)
        for other in source_tracks:
            if (
                other.track_id == target.track_id
                or other.track_id in self._green_preclose_consumed_track_ids
                or not self._target_is_fresh(other, timestamp_ns)
            ):
                continue
            # 只有当前可定位目标才能证明其落入 10 mm 禁区；失观或缺 K0
            # 目标不能被无条件推定在橙色目标旁边。
            if other.status is TrackStatus.COASTING or other.ground_point is None:
                continue
            if (
                math.hypot(
                    point.x - other.ground_point.x,
                    point.y - other.ground_point.y,
                )
                <= radius_mm + 1e-9
            ):
                relative = self._target_aligned_coordinates(point, other.ground_point)
                if (self._near_field_pickup is not None
                        and other.target_class in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE}
                        and self._graspable_target_is_usable(other)
                        and relative is not None
                        and relative[0] > math.hypot(point.x, point.y)
                        and abs(relative[1]) > self._transport_corridor_effective_half_width_mm()):
                    # 侧后方物资仅作为入口预览；最终由近场 K0 扫掠检查证明不混运。
                    continue
                return False
        return True

    def _transport_side_neighbor_target(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        tracks: tuple[TrackedTarget, ...] | None = None,
    ) -> TrackedTarget | None:
        """返回候选的蓝色侧邻，或单橙计划的蓝/橙侧邻。

        橙色邻近隔离只约束单橙目标；绿/黑候选不因橙色侧邻被拒绝。
        """

        if target.target_class in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE}:
            # The near-field planner checks actual swept geometry including blue extent.
            return None
        point = target.ground_point
        if point is None:
            return None
        config = self._near_field_grasp_config
        longitudinal_margin = (
            60.0
            if config is None
            else config.side_neighbor_longitudinal_margin_mm
        )
        lateral_margin = (
            90.0
            if config is None
            else config.side_neighbor_lateral_margin_mm
        )
        target_range_mm = math.hypot(point.x, point.y)
        source_tracks = self._tracker.tracks if tracks is None else tuple(tracks)
        for other in source_tracks:
            if (
                other.track_id == target.track_id
                or not self._target_is_fresh(other, timestamp_ns)
                or other.target_class
                not in {TargetClass.BLUE_DANGER, TargetClass.ORANGE_INJURED}
                or other.ground_point is None
                or (
                    other.target_class is TargetClass.ORANGE_INJURED
                    and target.target_class is not TargetClass.ORANGE_INJURED
                )
            ):
                continue
            relative = self._target_aligned_coordinates(
                point,
                other.ground_point,
            )
            if relative is None:
                continue
            forward_mm, lateral_mm = relative
            if (
                abs(forward_mm - target_range_mm) <= longitudinal_margin
                and 1e-6 < abs(lateral_mm) <= lateral_margin
            ):
                return other
        return None

    def _transport_group_size(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        reference: GroundPoint | None = None,
    ) -> int | None:
        """返回以目标 X 为终点时覆盖的合法目标数。

        橙色伤员必须独立转运；其中心 10 mm 半径内出现当前可定位的
        其它新鲜目标时拒绝。缺少地面点的无关目标不推定在禁区内。
        """

        point = target.ground_point if reference is None else reference
        if (
            not self._graspable_target_is_usable(target)
            or point is None
            or not self._point_in_transport_corridor(point)
            or self._candidate_path_blocked(point, breakup=False)
        ):
            return None
        stowed_limit = (
            self._greedy_new_target_min_x_mm()
            if self._greedy_active
            else None
        )
        if (
            stowed_limit is not None
            and target.target_class is not TargetClass.BLUE_DANGER
            and point.x <= stowed_limit
        ):
            return None
        policy = self.near_field_policy
        member_count = 1
        if target.target_class not in policy.allowed_classes:
            return None
        if not self._transport_orange_isolation_clear(target, timestamp_ns):
            return None
        if self._transport_side_neighbor_target(target, timestamp_ns) is not None:
            return None
        for other in self._tracker.tracks:
            if (
                other.track_id == target.track_id
                or other.track_id in self._green_preclose_consumed_track_ids
                or not self._target_is_fresh(other, timestamp_ns)
            ):
                continue
            other_point = other.ground_point
            if other_point is None:
                continue
            if (
                stowed_limit is not None
                and other.target_class is not TargetClass.BLUE_DANGER
                and other_point.x <= stowed_limit
            ):
                # Carried/delivered material is not a new member for capacity,
                # but it remains in the scene for the eventual sweep check.
                continue
            if not self._point_in_transport_corridor(
                other_point,
                end_x_mm=point.x,
            ):
                continue
            if target.target_class is TargetClass.ORANGE_INJURED:
                return None
            if other.target_class not in policy.allowed_classes:
                return None
            if other.target_class is TargetClass.ORANGE_INJURED:
                return None
            member_count += 1
            if member_count > policy.max_targets:
                return None
        return member_count

    def _find_isolated_green(
        self,
        timestamp_ns: int,
    ) -> TrackedTarget | None:
        """保留旧方法名；一般阶段也可返回固定走廊内的单个黑色物资。"""

        policy = self.near_field_policy
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if self._target_is_fresh(target, timestamp_ns)
            and target.target_class in policy.allowed_classes
            and self._graspable_target_is_usable(target)
            and target.ground_point is not None
            and self._green_path_is_clear_for_point(
                target,
                target.ground_point,
                timestamp_ns,
            )
        )
        return min(
            candidates,
            key=lambda item: (
                item.ground_point.x if item.ground_point is not None else math.inf,
                item.track_id,
            ),
            default=None,
        )


    def _green_path_is_clear(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
    ) -> bool:
        """判断目标接近走廊是否为空，坐标均为机器人地面系 mm。

        普通绿色接近要求目标在车前方；预闭爪重新对准可显式放宽这一条件。
        其它新鲜目标投影到“机器人到目标”的相对坐标后，只要满足
        ``0 <= aligned_forward < target_range`` 且横向中心距不超过
        ``_path_block_half_width_mm`` 即视为阻挡。
        没有地面坐标的其它目标按保守策略拒绝，因为无法证明走廊为空。
        """

        if not self._green_target_is_usable(target):
            return False
        point = target.ground_point
        assert point is not None
        return self._green_path_is_clear_for_point(target, point, timestamp_ns)

    def _green_path_is_clear_for_point(
        self,
        target: TrackedTarget,
        point: GroundPoint,
        timestamp_ns: int,
        *,
        ignored_track_ids: frozenset[int] = frozenset(),
        allow_behind: bool = False,
        lateral_half_width_mm: float | None = None,
        tracks: tuple[TrackedTarget, ...] | None = None,
    ) -> bool:
        """用指定绿色参考点检查目标对准后的接近走廊。

        走廊坐标系的 x 轴指向绿色目标，y 轴与其垂直；这样目标当前位于
        机器人左/右侧时，仍能正确判断其前方是否有其它物块。预闭爪候选重选
        阶段不主动转向搜索；候选进入通用对准动作后，可用 ``allow_behind=True``
        接受车后目标。

        横向阻挡门限按两块实体的内切半径之和加实际夹爪余量计算
        （见 ``_path_block_half_width_mm``）：固定半宽和外接半径会把只是
        近旁、并不在夹取路径上的异类目标当成阻挡，制造不必要的解团。
        调用方可用 ``lateral_half_width_mm`` 显式指定固定包络门限。
        """

        if not self._graspable_target_is_usable(target):
            return False
        if lateral_half_width_mm is not None and (
            isinstance(lateral_half_width_mm, bool)
            or not isinstance(lateral_half_width_mm, (int, float))
            or not math.isfinite(float(lateral_half_width_mm))
            or float(lateral_half_width_mm) < 0.0
        ):
            raise ValueError(
                "lateral_half_width_mm must be finite and non-negative."
            )
        if (
            (point.x <= 0.0 and not allow_behind)
            or self._candidate_path_blocked(point, breakup=False)
        ):
            return False
        target_range_mm = math.hypot(point.x, point.y)
        if target_range_mm <= 1e-6:
            return False
        for other in (self._tracker.tracks if tracks is None else tracks):
            if (
                other.track_id == target.track_id
                or other.track_id
                in self._green_preclose_consumed_track_ids
                or other.track_id in ignored_track_ids
                or not self._target_is_fresh(other, timestamp_ns)
            ):
                continue
            if other.ground_point is None:
                continue
            relative = self._target_aligned_coordinates(point, other.ground_point)
            if relative is None:
                return False
            other_forward_mm, other_lateral_mm = relative
            if not 0.0 <= other_forward_mm < target_range_mm:
                continue
            threshold_mm = (
                lateral_half_width_mm
                if lateral_half_width_mm is not None
                else self._path_block_half_width_mm(
                    target.target_class,
                    other.target_class,
                )
            )
            if abs(other_lateral_mm) <= threshold_mm:
                return False
        return True

    def _path_block_half_width_mm(
        self,
        reference_class: TargetClass,
        other_class: TargetClass,
    ) -> float:
        """返回其它目标算作“位于夹取路径上”的横向中心距门限，单位 mm。

        只按两块实体的内切半径之和叠加实际夹爪余量：目标包络外侧的一半
        ``clearance_mm`` 加近场走廊横向安全余量。外接半径和固定半宽会把
        只是近旁、并不在夹取路径上的异类目标当成阻挡，使搜索提前进入解团；
        内切半径是两块实体确实可能接触的距离，因此门限内才真正需要阻挡。
        缺少目标物理尺寸配置的纯逻辑调用方回退到
        ``green_path_half_width_mm``。
        """

        geometry = self._breakup_target_geometry
        if geometry is None:
            return float(self.config.green_path_half_width_mm)
        return (
            physical_radii(geometry.geometry_for(reference_class))[0]
            + physical_radii(geometry.geometry_for(other_class))[0]
            + self._path_corridor_margin_mm
        )

    @staticmethod
    def _target_aligned_coordinates(
        reference: GroundPoint,
        point: GroundPoint,
    ) -> tuple[float, float] | None:
        """把机器人地面系点投影到指向 ``reference`` 的相对坐标系。"""

        reference_range_mm = math.hypot(reference.x, reference.y)
        if reference_range_mm <= 1e-6:
            return None
        cos_theta = reference.x / reference_range_mm
        sin_theta = reference.y / reference_range_mm
        return (
            cos_theta * point.x + sin_theta * point.y,
            -sin_theta * point.x + cos_theta * point.y,
        )

    def _format_path_target_diagnostic(
        self,
        reference: GroundPoint | None,
        other: TrackedTarget,
        timestamp_ns: int,
        reference_class: TargetClass | None = None,
    ) -> tuple[str, bool]:
        """格式化相对走廊中的其它目标，并返回实际几何阻挡标记。

        门限与 ``_green_path_is_clear_for_point`` 使用同一个按实体尺寸计算的
        数值，日志里的 ``threshold_lateral_mm`` 就是当时实际采用的门限。
        """

        fresh = self._target_is_fresh(other, timestamp_ns)
        consumed = (
            other.track_id in self._green_preclose_consumed_track_ids
        )
        point = other.ground_point
        point_text = "unknown" if point is None else f"({point.x:.1f},{point.y:.1f})"
        relative = (
            None
            if reference is None or point is None
            else self._target_aligned_coordinates(reference, point)
        )
        if relative is None:
            forward_text = "unknown"
            lateral_text = "unknown"
        else:
            forward_text = f"{relative[0]:.1f}"
            lateral_text = f"{relative[1]:.1f}"

        geometrically_blocked = False
        unknown_ground = fresh and point is None and not consumed
        threshold_mm = (
            self._path_block_half_width_mm(reference_class, other.target_class)
            if reference_class is not None
            else self.config.green_path_half_width_mm
        )
        if (
            fresh
            and not consumed
            and relative is not None
            and reference is not None
        ):
            reference_range_mm = math.hypot(reference.x, reference.y)
            geometrically_blocked = (
                0.0 <= relative[0] < reference_range_mm
                and abs(relative[1]) <= threshold_mm
            )
        blocked_text = (
            "not_applicable_no_ground"
            if unknown_ground
            else str(geometrically_blocked).lower()
        )
        return (
            f"track={other.track_id},class={other.target_class.value},"
            f"xy={point_text},fresh={fresh},"
            f"aligned_forward={forward_text},aligned_lateral={lateral_text},"
            f"threshold_lateral_mm={threshold_mm:.1f},"
            f"consumed={consumed},"
            f"blocked={blocked_text}",
            geometrically_blocked,
        )

    def green_isolation_diagnostic(self, timestamp_ns: int) -> str:
        """返回绿色候选及所有其它目标的相对走廊诊断。

        保留原方法名以兼容日志调用方，但现在每个其它目标都会输出类别、原始
        GroundPoint、新鲜度、目标对齐坐标、实际门限和阻挡结果。
        """

        tracks = self._tracker.tracks
        green_tracks = tuple(
            target
            for target in tracks
            if target.target_class is TargetClass.GREEN_SUPPLY
        )
        if not green_tracks:
            other_entries = ",".join(
                (
                    f"track={other.track_id},class={other.target_class.value},"
                    f"xy={'unknown' if other.ground_point is None else f'({other.ground_point.x:.1f},{other.ground_point.y:.1f})'},"
                    f"fresh={self._target_is_fresh(other, timestamp_ns)},"
                    "aligned_forward=unknown,aligned_lateral=unknown,blocked=false"
                )
                for other in tracks
            )
            return "green=none;others=" + (other_entries or "none")

        entries: list[str] = []
        for target in green_tracks:
            point = target.ground_point
            point_text = "unknown" if point is None else f"({point.x:.1f},{point.y:.1f})"
            fresh = self._target_is_fresh(target, timestamp_ns)
            if not fresh:
                reason = "stale"
            elif not self._green_target_is_usable(target):
                reason = "not_usable"
            elif point is None:
                reason = "ground_missing"
            elif point.x <= 0.0:
                reason = "not_ahead"
            else:
                reason = "path_clear"

            other_entries: list[str] = []
            if point is not None:
                for other in tracks:
                    if other.track_id == target.track_id:
                        continue
                    detail, blocked = self._format_path_target_diagnostic(
                        point,
                        other,
                        timestamp_ns,
                        target.target_class,
                    )
                    other_entries.append(detail)
                    if reason == "path_clear" and blocked:
                        reason = f"path_track_{other.track_id}"
            entries.append(
                f"green_track={target.track_id},class={target.target_class.value},"
                f"xy={point_text},fresh={fresh},status={target.status.value},"
                f"path={reason},others=[{'|'.join(other_entries) or 'none'}]"
            )
        return ";".join(entries)

    def green_target_diagnostic(self, timestamp_ns: int) -> str:
        """返回绿/黑物资轨迹、选中标记及实时坐标和新鲜度。"""

        targets = tuple(
            target
            for target in self._tracker.tracks
            if target.target_class in {
                TargetClass.GREEN_SUPPLY,
                TargetClass.BLACK_CORE,
            }
        )
        if not targets:
            return "none"
        entries: list[str] = []
        for target in targets:
            point = target.ground_point
            point_text = (
                "unknown"
                if point is None
                else f"({point.x:.1f},{point.y:.1f})"
            )
            selected = "*" if target.track_id == self._selected_track_id else ""
            entries.append(
                f"{selected}track={target.track_id},xy_mm={point_text},"
                f"fresh={self._target_is_fresh(target, timestamp_ns)},"
                f"status={target.status.value}"
            )
        return ";".join(entries)

    def _find_preclose_green_target(
        self,
        timestamp_ns: int,
    ) -> TrackedTarget | None:
        """查找车体原点近距离范围内、尚未收入的绿色物块。

        复核发生在当前物块已经到达 ``green_grab_offset_mm`` 后。由于物块
        可能被夹爪接触、推动或遮挡，物块之间的相对位置不再适合作为稳定
        判据；这里恢复使用当前机器人地面坐标的原点半径。候选可以位于
        车前、侧方或车后；复核阶段不主动转向搜索，只把当前视野中通过门禁
        的观测交给已有绿色对准动作。目标路径仍须通过安全区和其它未收入
        目标的阻挡检查。
        """

        frame_floor = self._green_preclose_frame_floor
        ignored_track_ids = (
            frozenset()
            if self._selected_track_id is None
            else frozenset({self._selected_track_id})
        )
        candidates: list[TrackedTarget] = []
        for target in self._tracker.tracks:
            if (
                target.track_id == self._selected_track_id
                or target.track_id in self._green_preclose_consumed_track_ids
                or not self._target_is_fresh(target, timestamp_ns)
                or not self._green_target_is_usable(target)
                or target.ground_point is None
            ):
                continue
            if frame_floor is not None and target.frame_sequence <= frame_floor:
                # 到位前的旧地面坐标不能用于判断邻域；等待停车后的新帧，
                # 也给刚露出的目标留出 tracker 确认时间。
                continue
            point = target.ground_point
            target_range_mm = math.hypot(point.x, point.y)
            if target_range_mm > self.config.green_preclose_recheck_range_mm:
                continue
            if not self._green_path_is_clear_for_point(
                target,
                point,
                timestamp_ns,
                ignored_track_ids=ignored_track_ids,
                allow_behind=True,
            ):
                continue
            candidates.append(target)
        return min(
            candidates,
            key=lambda item: (
                math.hypot(item.ground_point.x, item.ground_point.y)
                if item.ground_point is not None
                else math.inf,
                abs(item.ground_point.y)
                if item.ground_point is not None
                else math.inf,
                item.track_id,
            ),
            default=None,
        )

    def _begin_preclose_green_realign(
        self,
        timestamp_ns: int,
    ) -> MatchDecision | None:
        """发现新的近距离绿块时，切回绿色对准并保留已收入物块计数。"""

        if (
            self._green_preclose_carried_count
            >= min(1 if self._transport_count == 0 else self._supply_capacity(),
                   self.config.green_preclose_max_carried_blocks)
        ):
            return None
        target = self._find_preclose_green_target(timestamp_ns)
        if target is None or target.ground_point is None:
            return None
        if self._selected_track_id is not None:
            self._green_preclose_consumed_track_ids.add(
                self._selected_track_id
            )
        self._green_preclose_carried_count += 1
        self._selected_track_id = target.track_id
        self._selected_green_ground = target.ground_point
        self._green_preclose_realign_active = True
        self._green_realign_pending = False
        self._green_realign_done = True
        self._green_align_lost_since_ns = None
        self._green_approach_base_distance_m = None
        self._green_approach_distance_m = None
        self._green_reference_samples = []
        self._green_reference_last_seen_ns = None
        self._green_reference = None
        self._green_reference_heading_rad = None
        self._green_reference_distance_m = None
        self._reset_green_alignment_gate()
        self._green_preclose_frame_floor = None
        self._green_preclose_recheck_started_ns = None
        self.state = MatchState.TRANSPORT_ALIGN_GREEN
        self._begin_action_settle(timestamp_ns, "green_preclose_realign")
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            f"green_preclose_green_found_realign:{target.track_id}",
            posture=GripperPosture.TRANSPORT,
        )

    def _begin_green_preclose_recheck(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """到达抓取偏移后停车，开始一段可确认新目标的复核窗口。"""

        # `_find_preclose_green_target()` intentionally uses the current robot
        # origin.  Only the frame floor is locked here, so the candidate's
        # ground point comes from a post-stop frame rather than a moving-frame
        # estimate.
        # 当前 step 已经把最新感知帧送入 tracker；复核必须从下一帧开始，
        # 避免把仍对应于行进过程的旧地面坐标当成停车后的邻居位置。
        self._green_preclose_frame_floor = self._last_tracker_frame_sequence
        self._green_preclose_recheck_started_ns = None
        self.state = MatchState.TRANSPORT_PRE_CLOSE_RECHECK
        self._begin_action_settle(timestamp_ns, "green_preclose_recheck")
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "green_grab_offset_reached_start_preclose_recheck",
            posture=GripperPosture.TRANSPORT,
        )

    def _finish_green_preclose_recheck(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """复核窗口结束后锁存闭爪动作。"""

        self._green_preclose_frame_floor = None
        self._green_preclose_recheck_started_ns = None
        self.state = MatchState.TRANSPORT_CLOSE_GRIPPER
        self._gripper_phase_started_ns = timestamp_ns
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "green_preclose_recheck_complete_close_gripper",
            posture=GripperPosture.CLOSED,
        )

    def _step_preclose_recheck(self, timestamp_ns: int) -> MatchDecision:
        """只在当前视野停车观察邻域，不做原地搜索转向。"""

        settling = self._consume_action_settle(
            timestamp_ns,
            "green_preclose_recheck",
            posture=GripperPosture.TRANSPORT,
            reason="green_preclose_waiting_for_vehicle_stop",
        )
        if settling is not None:
            return settling
        if self._green_preclose_recheck_started_ns is None:
            self._green_preclose_recheck_started_ns = timestamp_ns

        preclose_realign = self._begin_preclose_green_realign(timestamp_ns)
        if preclose_realign is not None:
            return preclose_realign

        started_ns = self._green_preclose_recheck_started_ns
        assert started_ns is not None
        if (
            timestamp_ns - started_ns
            < self._green_preclose_recheck_hold_ns
        ):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_preclose_rechecking_neighbor",
                posture=GripperPosture.TRANSPORT,
            )
        return self._finish_green_preclose_recheck(timestamp_ns)

    def _step_approach_green(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """按编码器距离接近绿色目标，并在闭爪前复核近距离绿色物块。"""

        if self._near_field_pickup is not None:
            return self._step_formal_green_approach(
                timestamp_ns,
                cumulative_distance_m,
            )

        settling = self._consume_action_settle(
            timestamp_ns,
            "green_before_approach",
            posture=GripperPosture.TRANSPORT,
            reason="green_waiting_after_alignment",
        )
        if settling is not None:
            return settling
        if self._action_settle_phase == "green_before_close":
            settling = self._consume_action_settle(
                timestamp_ns,
                "green_before_close",
                posture=GripperPosture.TRANSPORT,
                reason="green_waiting_before_close_gripper",
            )
            if settling is not None:
                return settling
            return self._finish_green_preclose_recheck(timestamp_ns)
        if cumulative_distance_m is None or self._green_approach_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_approach_waiting_for_odometry",
                posture=GripperPosture.TRANSPORT,
            )
        self._green_align_lost_since_ns = None
        if self._green_approach_base_distance_m is None:
            self._green_approach_base_distance_m = cumulative_distance_m
        travelled = cumulative_distance_m - self._green_approach_base_distance_m
        if travelled >= self._green_approach_distance_m - 1e-9:
            if self._green_realign_pending:
                self._green_realign_pending = False
                self._green_realign_done = True
                self._green_reference_samples = []
                self._green_reference_last_seen_ns = None
                self._green_reference = None
                self._green_reference_heading_rad = None
                self._green_reference_distance_m = None
                self._reset_green_alignment_gate()
                self._green_align_lost_since_ns = None
                self._green_approach_base_distance_m = None
                self._green_approach_distance_m = None
                self.state = MatchState.TRANSPORT_ALIGN_GREEN
                self._begin_action_settle(
                    timestamp_ns,
                    "green_realign",
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "green_near_standoff_reached_start_realign",
                    posture=GripperPosture.TRANSPORT,
                )
            if self._near_field_pickup is not None:
                # 近场流程的接近终点就是 handoff range。到达这里后必须把
                # 选择、张爪、定距前进和合爪交给近场状态机，不能落入旧的
                # green_grab_offset/pre-close 合爪路径，否则物块尚未揽入时
                # 就会被当作已抓取并进入运输。
                return self._begin_near_field_grasp(
                    timestamp_ns,
                    handoff_prior=self._selected_handoff_prior(timestamp_ns),
                )
            if (
                self._green_preclose_carried_count
                >= self.config.green_preclose_max_carried_blocks
            ):
                if self._begin_action_settle(timestamp_ns, "green_before_close"):
                    return self._decision(
                        timestamp_ns,
                        0.0,
                        0.0,
                        "green_grab_offset_reached_wait_before_close",
                        posture=GripperPosture.TRANSPORT,
                    )
                return self._finish_green_preclose_recheck(timestamp_ns)
            return self._begin_green_preclose_recheck(timestamp_ns)
        reference = self._green_reference
        heading_tolerance = (
            math.atan2(
                self.config.green_alignment_tolerance_mm,
                max(math.hypot(reference.x, reference.y), 1e-6),
            )
            if reference is not None
            else 0.0
        )
        angular = self._heading_hold_angular_velocity(
            self._green_reference_heading_rad,
            kp_rad_s=self.config.green_alignment_kp_rad_s,
            max_angular_velocity_rad_s=(
                self.config.green_alignment_max_angular_velocity_rad_s
            ),
            tolerance_rad=heading_tolerance,
        )
        if angular is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_approach_waiting_for_heading",
                posture=GripperPosture.TRANSPORT,
            )
        return self._decision(
            timestamp_ns,
            self.config.green_approach_speed_m_s,
            angular,
            (
                "approach_green_to_near_standoff_before_realign"
                if self._green_realign_pending
                else "approach_green_x_minus_150_no_realign"
            ),
            posture=GripperPosture.TRANSPORT,
        )

    def _current_ground_point_for_track(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
    ) -> GroundPoint | None:
        point = target.ground_point
        if point is None or not self._target_is_fresh(target, timestamp_ns):
            return None
        if target.last_seen_timestamp_ns == timestamp_ns:
            return point
        return self._ground_point_at_now(
            point,
            target.last_seen_timestamp_ns,
            timestamp_ns,
        )

    def _green_path_is_clear_at_now(
        self,
        target: TrackedTarget,
        point: GroundPoint,
        timestamp_ns: int,
    ) -> bool:
        """Check the approach corridor using capture-time pose alignment."""

        if point.x <= 0.0 or self._candidate_path_blocked(point, breakup=False):
            return False
        target_range_mm = math.hypot(point.x, point.y)
        if target_range_mm <= 1e-6:
            return False
        position = self.estimated_field_position
        if self._green_target_field_point is None:
            self._formal_green_target_heading(timestamp_ns, point)
        target_field = self._green_target_field_point
        if position is None or target_field is None:
            return False
        approach_distance_mm = max(
            0.0,
            target_range_mm - self._near_field_handoff_range_mm(),
        )
        bearing = math.atan2(
            target_field.y - position.y,
            target_field.x - position.x,
        )
        endpoint = FieldPoint(
            position.x + approach_distance_mm * math.cos(bearing),
            position.y + approach_distance_mm * math.sin(bearing),
        )
        low_x, high_x, low_y, high_y = self._physical_field_bounds()
        margin = self._robot_path_clearance_mm()
        if not (
            low_x + margin <= position.x <= high_x - margin
            and low_y + margin <= position.y <= high_y - margin
            and low_x + margin <= endpoint.x <= high_x - margin
            and low_y + margin <= endpoint.y <= high_y - margin
        ):
            return False
        ignored_ids = self._preview_ignored_track_ids(target, timestamp_ns)
        for other in self._tracker.tracks:
            if (
                other.track_id == target.track_id
                or other.track_id in ignored_ids
                or other.track_id in self._green_preclose_consumed_track_ids
                or not self._target_is_fresh(other, timestamp_ns)
            ):
                continue
            other_point = self._current_ground_point_for_track(other, timestamp_ns)
            if other_point is None:
                if other.target_class in {TargetClass.BLUE_DANGER}:
                    return False
                continue
            relative = self._target_aligned_coordinates(point, other_point)
            if relative is None:
                return False
            if 0.0 <= relative[0] < target_range_mm and abs(relative[1]) <= (
                self._path_block_half_width_mm(
                    target.target_class,
                    other.target_class,
                )
            ):
                return False
        return True

    def _formal_green_target_heading(
        self,
        timestamp_ns: int,
        point: GroundPoint,
    ) -> float | None:
        """Return the field heading to the target from measured pose."""

        position = self.estimated_field_position
        if position is None:
            return None
        if self._green_target_field_point is None:
            pose = self._pose_at(timestamp_ns)
            if pose is None:
                # A caller may provide a current-frame diagnostic reference
                # without a capture history.  It is safe to anchor that one
                # point at the current pose; later stale captures still return
                # ``None`` and cannot be time-compensated.
                if self._latest_heading_rad is None:
                    return None
                position = self.estimated_field_position
                if position is None:
                    return None
                self._green_target_field_point = FieldPoint(
                    position.x
                    + math.cos(self._latest_heading_rad) * point.x
                    - math.sin(self._latest_heading_rad) * point.y,
                    position.y
                    + math.sin(self._latest_heading_rad) * point.x
                    + math.cos(self._latest_heading_rad) * point.y,
                )
            else:
                self._green_target_field_point = self._field_point_from_pose(
                    pose,
                    point,
                )
        return math.atan2(
            self._green_target_field_point.y - position.y,
            self._green_target_field_point.x - position.x,
        )

    def _begin_formal_green_approach(
        self,
        timestamp_ns: int,
        point: GroundPoint,
        target_heading_rad: float,
    ) -> MatchDecision:
        self._green_alignment_target_heading_rad = target_heading_rad
        self._green_turn_active = False
        self._green_alignment_wait_since_ns = None
        self._green_approach_base_distance_m = self._latest_cumulative_distance_m
        self._green_approach_distance_m = max(
            0.0,
            (math.hypot(point.x, point.y) - self._near_field_handoff_range_mm())
            / 1000.0,
        )
        self._green_realign_pending = False
        self._green_realign_done = True
        self.state = MatchState.TRANSPORT_APPROACH_GREEN
        self._begin_action_settle(timestamp_ns, "green_before_approach")
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "green_aligned_start_imu_approach",
            posture=GripperPosture.TRANSPORT,
        )

    def _step_formal_green_align(self, timestamp_ns: int) -> MatchDecision:
        """Coarsely align with IMU feedback until the gripper can reach the target."""

        for phase, reason in (
            ("green_reference", "green_waiting_before_current_geometry"),
            ("green_realign", "green_waiting_after_bounded_turn"),
            ("green_preclose_realign", "green_waiting_before_bounded_turn"),
        ):
            settling = self._consume_action_settle(
                timestamp_ns,
                phase,
                posture=GripperPosture.TRANSPORT,
                reason=reason,
            )
            if settling is not None:
                return settling

        target = self._selected_target()
        point = None if target is None else self._selected_green_point(timestamp_ns)
        if point is None and target is None and self._green_reference is not None:
            # This fallback serves pure-logic callers that seed a reference
            # directly.  Production match always has a selected tracker target.
            point = self._green_reference
        if point is None:
            return self._hold_or_restart_green_target(timestamp_ns)
        if self._near_field_pickup is not None and math.hypot(point.x, point.y) <= self._near_field_handoff_range_mm():
            return self._begin_near_field_grasp(
                timestamp_ns,
                handoff_prior=self._selected_handoff_prior(timestamp_ns),
            )
        if target is not None and not self._green_path_is_clear_at_now(target, point, timestamp_ns):
            if self._dynamic_breakup_enabled and not self._candidate_path_blocked(point, breakup=False):
                return self._enter_breakup_only_search(timestamp_ns)
            self._selected_track_id = None
            self._selected_green_ground = None
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(
                timestamp_ns,
                0.0,
                self._cluster_search_angular_velocity_rad_s,
                "green_current_path_blocked_restart_search",
                posture=GripperPosture.CLOSED,
            )
        current_heading = self._latest_heading_rad
        if current_heading is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_alignment_waiting_for_imu_heading",
                posture=GripperPosture.TRANSPORT,
            )
        target_heading = self._formal_green_target_heading(timestamp_ns, point)
        if target_heading is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_alignment_waiting_for_pose_history",
                posture=GripperPosture.TRANSPORT,
            )
        self._green_alignment_target_heading_rad = target_heading
        self._green_reference = point
        self._green_reference_heading_rad = target_heading
        self._green_reference_distance_m = max(
            0.0,
            (math.hypot(point.x, point.y) - self._near_field_handoff_range_mm())
            / 1000.0,
        )
        heading_error = normalize_angle(target_heading - current_heading)
        range_mm = math.hypot(point.x, point.y)
        tolerance_rad = math.atan2(
            self.config.green_alignment_tolerance_mm,
            max(range_mm, 1e-6),
        )
        if abs(point.y) <= self.config.green_alignment_tolerance_mm or abs(heading_error) <= tolerance_rad:
            return self._begin_formal_green_approach(
                timestamp_ns,
                point,
                target_heading,
            )

        if self._green_turn_active:
            if self._green_turn_start_heading_rad is not None:
                self._green_turn_progress_rad = self._directional_delta(
                    self._green_turn_start_heading_rad,
                    current_heading,
                    self._green_turn_direction,
                )
            if math.copysign(1.0, heading_error) != math.copysign(1.0, self._green_turn_direction):
                self._green_turn_active = False
                if self._green_turn_attempts >= 2:
                    return self._return_to_near_field_search(
                        timestamp_ns,
                        "alignment_direction_changed_after_bounded_turn",
                    )
            elif self._green_turn_progress_rad >= self._green_turn_budget_rad - 1e-3:
                self._green_turn_active = False
                self._green_alignment_wait_since_ns = timestamp_ns
                if self._green_turn_attempts >= 2:
                    return self._return_to_near_field_search(
                        timestamp_ns,
                        "alignment_bounded_turn_exhausted",
                    )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "green_bounded_turn_complete_wait_current_geometry",
                    posture=GripperPosture.TRANSPORT,
                    soft_brake=True,
                )

        if not self._green_turn_active:
            if self._green_turn_attempts >= 2:
                return self._return_to_near_field_search(
                    timestamp_ns,
                    "alignment_attempt_budget_exhausted",
                )
            self._green_turn_attempts += 1
            self._green_turn_direction = math.copysign(1.0, heading_error)
            self._green_turn_start_heading_rad = current_heading
            self._green_turn_budget_rad = min(abs(heading_error), math.pi / 2.0)
            self._green_turn_progress_rad = 0.0
            self._green_turn_active = True
        angular = _clamp(
            self.config.green_alignment_kp_rad_s * heading_error,
            -self.config.green_alignment_max_angular_velocity_rad_s,
            self.config.green_alignment_max_angular_velocity_rad_s,
        )
        return self._decision(
            timestamp_ns,
            0.0,
            angular,
            "green_coarse_align_imu_closed_loop"
            if self._green_turn_attempts == 1
            else "green_correction_align_imu_closed_loop",
            posture=GripperPosture.TRANSPORT,
            min_wheel_velocity_m_s=self.config.green_alignment_min_wheel_velocity_m_s,
        )

    def _step_formal_green_approach(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        settling = self._consume_action_settle(
            timestamp_ns,
            "green_before_approach",
            posture=GripperPosture.TRANSPORT,
            reason="green_waiting_after_alignment",
        )
        if settling is not None:
            return settling
        if cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_approach_waiting_for_odometry",
                posture=GripperPosture.TRANSPORT,
            )
        target = self._selected_target()
        point = None if target is None else self._selected_green_point(timestamp_ns)
        if point is None and self._selected_track_id is not None:
            # 正常远场接近以及补夹都只在当前目标真正失鲜/消失后接管
            # 其它候选；补夹一旦选定目标仍在，就不能被旁边新出现的目标替换。
            target_missing = (
                self._greedy_selected_target_missing(timestamp_ns)
                if self._greedy_active
                else True
            )
            if not target_missing:
                return self._hold_or_restart_green_target(timestamp_ns)
            alternative = self._find_approach_seed(
                timestamp_ns,
                minimum_last_seen_ns=(
                    self._greedy_started_ns if self._greedy_active else None
                ),
                prefer_nearest=self._greedy_active,
            )
            if (alternative is not None
                    and self._current_ground_point_for_track(alternative, timestamp_ns) is not None):
                self._near_field_handoff_prior = None
                return self._begin_green_transport(timestamp_ns, alternative, group_preview=True)
            return self._hold_or_restart_green_target(timestamp_ns)
        if point is None and target is None and self._green_reference is not None:
            if self._greedy_active:
                return self._hold_or_restart_green_target(timestamp_ns)
            point = self._green_reference
            if self._green_approach_base_distance_m is not None and self._green_approach_distance_m is not None:
                if cumulative_distance_m is not None and cumulative_distance_m - self._green_approach_base_distance_m >= self._green_approach_distance_m - 1e-9:
                    return self._begin_near_field_grasp(timestamp_ns)
        if point is None:
            return self._hold_or_restart_green_target(timestamp_ns)
        if target is not None and not self._green_path_is_clear_at_now(target, point, timestamp_ns):
            if self._dynamic_breakup_enabled and not self._candidate_path_blocked(point, breakup=False):
                return self._enter_breakup_only_search(timestamp_ns)
            return self._start_path_reselection(
                timestamp_ns,
                posture=GripperPosture.TRANSPORT,
                reason="green_current_path_blocked_restart_search",
            )
        range_mm = math.hypot(point.x, point.y)
        remaining_m = max(
            0.0,
            (range_mm - self._near_field_handoff_range_mm()) / 1000.0,
        )
        if target is not None:
            self._green_approach_distance_m = remaining_m
            self._green_approach_base_distance_m = cumulative_distance_m
        if remaining_m <= 1e-6:
            return self._begin_near_field_grasp(
                timestamp_ns,
                handoff_prior=self._selected_handoff_prior(timestamp_ns),
            )
        target_heading = self._formal_green_target_heading(timestamp_ns, point)
        if target_heading is None or self._latest_heading_rad is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_approach_waiting_for_pose_history",
                posture=GripperPosture.TRANSPORT,
            )
        heading_error = normalize_angle(target_heading - self._latest_heading_rad)
        angular = _clamp(
            self.config.green_alignment_kp_rad_s * heading_error,
            -self.config.green_alignment_max_angular_velocity_rad_s,
            self.config.green_alignment_max_angular_velocity_rad_s,
        )
        return self._decision(
            timestamp_ns,
            approach_speed_m_s(
                remaining_m,
                self.config.green_approach_speed_m_s,
                self.config.pickup_cruise_speed_scale,
                self._near_field_pickup.deceleration_m_s2,
                precision_approach=True,
                terminal_speed_gain_s_inv=(
                    self.config.pickup_terminal_speed_gain_s_inv
                ),
            ),
            angular,
            "approach_green_with_capture_pose_alignment",
            posture=GripperPosture.TRANSPORT,
        )

    def _step_align_green(self, timestamp_ns: int) -> MatchDecision:
        """停车取 10 帧均值后做阻挡判断，再对准绿色目标。"""

        if self._near_field_pickup is not None:
            return self._step_formal_green_align(timestamp_ns)
        return self._align_green_legacy(timestamp_ns)

    def _align_green_legacy(self, timestamp_ns: int) -> MatchDecision:
        """无近场序列时的历史对准路径。

        变体的蓝色运输在真机验证过这条路径，因此保留为可显式调用的方法，
        而不是让它被 ``_step_formal_green_align`` 静默替换。
        """

        settling = self._consume_action_settle(
            timestamp_ns,
            "green_reference",
            posture=GripperPosture.TRANSPORT,
            reason="green_waiting_before_reference_collection",
        )
        if settling is not None:
            return settling
        settling = self._consume_action_settle(
            timestamp_ns,
            "green_realign",
            posture=GripperPosture.TRANSPORT,
            reason="green_waiting_after_near_standoff",
        )
        if settling is not None:
            return settling
        settling = self._consume_action_settle(
            timestamp_ns,
            "green_preclose_realign",
            posture=GripperPosture.TRANSPORT,
            reason="green_waiting_before_preclose_realign",
        )
        if settling is not None:
            return settling

        if self._green_reference is None:
            self._reset_green_alignment_gate()

        # 目标已经在近场时，直接把同一目标交给近场唯一确认窗口。
        # 不再先做远场十帧均值，再重复一次近场旋转/收集。
        if self._near_field_pickup is not None:
            target_point = self._selected_green_point(timestamp_ns)
            if (
                target_point is not None
                and math.hypot(target_point.x, target_point.y)
                <= self._near_field_handoff_range_mm()
            ):
                return self._begin_near_field_grasp(
                    timestamp_ns,
                    handoff_prior=self._selected_handoff_prior(timestamp_ns),
                )

        if (
            self._near_field_pickup is not None
            and self._green_reference is not None
            and math.hypot(
                self._green_reference.x,
                self._green_reference.y,
            ) <= self._near_field_handoff_range_mm()
        ):
            return self._begin_near_field_grasp(
                timestamp_ns,
                handoff_prior=self._selected_handoff_prior(timestamp_ns),
            )

        if self._green_reference is None:
            target = self._selected_target()
            target_point = self._selected_green_point(timestamp_ns)
            if target is None or target_point is None:
                return self._hold_or_restart_green_target(timestamp_ns)
            if (
                target.last_seen_timestamp_ns
                != self._green_reference_last_seen_ns
            ):
                self._green_reference_samples.append(target_point)
                self._green_reference_last_seen_ns = (
                    target.last_seen_timestamp_ns
                )
            # 主跟踪器已经确认身份；正式近场还会复核，不重复停车等十帧。
            required_samples = 1 if self._near_field_pickup is not None else 10
            if len(self._green_reference_samples) < required_samples:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "green_collecting_10_point_reference",
                    posture=GripperPosture.TRANSPORT,
                )
            self._green_reference = GroundPoint(
                sum(point.x for point in self._green_reference_samples)
                / len(self._green_reference_samples),
                sum(point.y for point in self._green_reference_samples)
                / len(self._green_reference_samples),
            )
            if self._latest_heading_rad is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "green_reference_waiting_for_heading",
                    posture=GripperPosture.TRANSPORT,
                )
            reference_range_mm = math.hypot(
                self._green_reference.x,
                self._green_reference.y,
            )
            if (
                self._near_field_pickup is not None
                and reference_range_mm <= self._near_field_handoff_range_mm()
            ):
                return self._begin_near_field_grasp(
                    timestamp_ns,
                    handoff_prior=self._selected_handoff_prior(timestamp_ns),
                )
            group_ignored_track_ids = (
                self._preview_ignored_track_ids(target, timestamp_ns)
                if self._near_field_group_preview else frozenset()
            )
            path_clear = self._green_path_is_clear_for_point(
                target,
                self._green_reference,
                timestamp_ns,
                ignored_track_ids=group_ignored_track_ids,
                allow_behind=self._green_preclose_realign_active,
                lateral_half_width_mm=(
                    self._transport_corridor_effective_half_width_mm()
                    if self._near_field_pickup is not None
                    and self._opportunistic_single_green
                    and not self._near_field_group_preview
                    else None
                ),
            )
            singleton_clear = (
                self._green_preclose_realign_active
                or self._near_field_group_preview
                or not self._opportunistic_single_green
                or self._opportunistic_single_green_is_clear(
                    target,
                    timestamp_ns,
                    reference=self._green_reference,
                )
            )
            if not path_clear or not singleton_clear:
                safe_zone_blocked = self._candidate_path_blocked(
                    self._green_reference,
                    breakup=False,
                )
                if safe_zone_blocked:
                    return self._start_path_reselection(
                        timestamp_ns,
                        posture=GripperPosture.TRANSPORT,
                        reason="target_path_intersects_safe_zone_stop_and_reselect",
                    )
                self._selected_track_id = None
                self._selected_green_ground = None
                self._begin_cluster_search()
                self.state = MatchState.SEARCH_CLUSTER
                return self._decision(
                    timestamp_ns,
                    0.0,
                    self._cluster_search_angular_velocity_rad_s,
                    (
                        "green_reference_path_blocked_restart_breakup_search"
                        if not path_clear
                        else "green_reference_not_singleton_restart_breakup_search"
                    ),
                )
            self._selected_green_ground = self._green_reference
            self._green_reference_heading_rad = normalize_angle(
                self._latest_heading_rad
                + math.atan2(
                    self._green_reference.y,
                    self._green_reference.x,
                )
            )
            near_realign_standoff_mm = (
                self._near_field_handoff_range_mm()
                if self._near_field_pickup is not None
                else self.config.opportunistic_single_green_realign_standoff_mm
            )
            # 二次对准适用于所有绿色目标；opportunistic 标志只控制
            # 搜索态的单物块净空门禁，不应决定目标距离校正是否生效。
            if (
                not self._green_realign_done
                and reference_range_mm > near_realign_standoff_mm
            ):
                self._green_realign_pending = True
                approach_standoff_mm = near_realign_standoff_mm
            else:
                self._green_realign_pending = False
                self._green_realign_done = True
                approach_standoff_mm = (
                    self.config.green_grab_offset_mm
                    if self._near_field_pickup is None
                    else self._near_field_handoff_range_mm()
                )
            self._green_reference_distance_m = max(
                0.0,
                (reference_range_mm - approach_standoff_mm) / 1000.0,
            )
        if (
            self._green_reference is None
            or self._green_reference_heading_rad is None
            or self._green_reference_distance_m is None
            or self._latest_heading_rad is None
        ):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_reference_waiting_for_heading",
                posture=GripperPosture.TRANSPORT,
            )
        if self._green_alignment_started_ns is None:
            self._green_alignment_started_ns = timestamp_ns
        elif (
            timestamp_ns - self._green_alignment_started_ns
            >= round(self.config.green_alignment_timeout_ms * 1_000_000)
        ):
            self._selected_track_id = None
            self._selected_green_ground = None
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(
                timestamp_ns,
                0.0,
                self._cluster_search_angular_velocity_rad_s,
                "green_alignment_timeout_restart_search",
                posture=GripperPosture.TRANSPORT,
            )
        self._green_align_lost_since_ns = None
        heading_error = normalize_angle(
            self._green_reference_heading_rad
            - self._latest_heading_rad
        )
        reference_range_mm = math.hypot(
            self._green_reference.x,
            self._green_reference.y,
        )
        heading_tolerance = math.atan2(
            self.config.green_alignment_tolerance_mm,
            max(reference_range_mm, 1e-6),
        )
        hysteresis_tolerance = math.atan2(
            self.config.green_alignment_tolerance_mm
            + self.config.green_alignment_hysteresis_mm,
            max(reference_range_mm, 1e-6),
        )
        error_abs = abs(heading_error)
        frame_sequence = self._last_tracker_frame_sequence
        new_frame = (
            frame_sequence is not None
            and frame_sequence != self._green_alignment_last_frame_sequence
        )
        within_tolerance = error_abs <= heading_tolerance
        within_hysteresis = error_abs <= hysteresis_tolerance
        if new_frame:
            if within_tolerance or (
                self._green_alignment_stable_count > 0
                and within_hysteresis
            ):
                self._green_alignment_stable_count += 1
            else:
                self._green_alignment_stable_count = 0
            self._green_alignment_last_frame_sequence = frame_sequence
        if (
            self._green_alignment_stable_count
            >= self.config.green_alignment_stable_frames
        ):
            self.state = MatchState.TRANSPORT_APPROACH_GREEN
            self._green_approach_base_distance_m = None
            self._green_approach_distance_m = max(
                0.0,
                self._green_reference_distance_m,
            )
            self._begin_action_settle(
                timestamp_ns,
                "green_before_approach",
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_aligned_y_zero_start_approach",
                posture=GripperPosture.TRANSPORT,
            )
        if within_tolerance or (
            self._green_alignment_stable_count > 0 and within_hysteresis
        ):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_alignment_stabilizing",
                posture=GripperPosture.TRANSPORT,
            )
        if not within_hysteresis:
            self._green_alignment_stable_count = 0
        return self._decision(
            timestamp_ns,
            0.0,
            _clamp(
                self.config.green_alignment_kp_rad_s * heading_error,
                -self.config.green_alignment_max_angular_velocity_rad_s,
                self.config.green_alignment_max_angular_velocity_rad_s,
            ),
            "align_green_relative_y_to_zero",
            posture=GripperPosture.TRANSPORT,
            min_wheel_velocity_m_s=(
                self.config.green_alignment_min_wheel_velocity_m_s
            ),
        )

    def _near_field_handoff_range_mm(self) -> float:
        config = self._near_field_grasp_config
        return (
            450.0
            if config is None
            else float(config.max_range_mm)
        )

    def _selected_handoff_prior(
        self,
        timestamp_ns: int,
    ) -> NearFieldHandoffPrior | None:
        """把当前远场已选目标压缩为近场决策使用的静态先验。"""

        target = self._selected_target()
        if (
            target is None
            or not self._target_is_fresh(target, timestamp_ns)
            or not self._graspable_target_is_usable(target)
            or target.ground_point is None
            or (
                self._transport_count == 0
                and target.target_class is not TargetClass.GREEN_SUPPLY
            )
        ):
            return None
        point = self._current_ground_point_for_track(target, timestamp_ns)
        if point is None:
            # 位姿历史不覆盖观测时刻时保留原始机器人系点，不静默丢掉交接。
            point = target.ground_point
        return NearFieldHandoffPrior(
            target.target_class,
            point,
            target.track_id,
        )

    def _begin_near_field_grasp(
        self,
        timestamp_ns: int,
        *,
        handoff_prior: NearFieldHandoffPrior | None = None,
    ) -> MatchDecision:
        """在绿色目标进入近场后开启一次独立的近场选组会话。"""

        assert self._near_field_pickup is not None
        if handoff_prior is not None and not isinstance(
            handoff_prior, NearFieldHandoffPrior
        ):
            raise TypeError(
                "handoff_prior must be a NearFieldHandoffPrior or None."
            )
        if (handoff_prior is not None
                and math.hypot(handoff_prior.ground_point.x, handoff_prior.ground_point.y)
                    > self._near_field_handoff_range_mm()):
            target = next((item for item in self._tracker.tracks
                           if item.track_id == handoff_prior.source_track_id
                           and item.target_class is handoff_prior.target_class), None)
            if target is not None:
                point = self._current_ground_point_for_track(target, timestamp_ns)
                if (point is not None and not self._candidate_path_blocked(point, breakup=False)
                        and self._green_path_is_clear_at_now(target, point, timestamp_ns)):
                    decision = self._begin_green_transport(timestamp_ns, target, group_preview=True)
                    self.state = MatchState.TRANSPORT_APPROACH_GREEN
                    return replace(decision, state=self.state, reason="far_target_continue_checked_approach")
            return self._return_to_near_field_search(timestamp_ns, "far_target_approach_blocked")
        if self._grasp_task is None and handoff_prior is not None:
            self._adopt_grasp_task(
                timestamp_ns, track_id=handoff_prior.source_track_id,
                target_class=handoff_prior.target_class, point=handoff_prior.ground_point,
                field=self._field_point_from_ground(handoff_prior.ground_point),
            )
        self._near_field_session_id += 1
        self._near_field_pickup.reset()
        self._near_field_handoff_prior = handoff_prior
        self._near_field_last_failure_diagnostic = None
        if not self._greedy_active:
            self._transport_target_classes = ()
            self._cargo_capture_floor_ns = None
        self._selected_track_id = None
        self._selected_green_ground = None
        self._green_reference_samples = []
        self._green_reference_last_seen_ns = None
        self._green_reference = None
        self._green_reference_heading_rad = None
        self._green_reference_distance_m = None
        self._reset_green_alignment_gate()
        self._green_approach_base_distance_m = None
        self._green_approach_distance_m = None
        self._green_preclose_consumed_track_ids.clear()
        self._green_preclose_carried_count = 0
        self._green_preclose_realign_active = False
        self._green_preclose_frame_floor = None
        self._green_preclose_recheck_started_ns = None
        self._near_field_group_preview = False
        self._breakup_only = False
        # 新会话重新推导结论；上一次解团的冻结计划不能跨会话沿用。
        self._breakup_frozen_plan = None
        self._near_field_route = GraspRoute.DECIDING
        # 窗口计时在停车后的 observation window 打开时才开始。
        self._near_field_confirmation_started_ns = None
        self._near_field_no_plan_budget_ms = None
        self._near_field_route_rejections = ()
        self._near_field_route_elapsed_ms = 0.0
        self._near_field_route_candidate_count = 0
        self.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
        if self._stationary_motion.stationary_since(timestamp_ns) is None:
            self._begin_action_settle(timestamp_ns, "near_field_grasp")
        else:
            self._action_settle_phase = None
            self._action_settle_until_ns = None
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "green_near_field_handoff_started",
            posture=GripperPosture.CLOSED,
            soft_brake=True,
        )

    def _near_field_path_clear(
        self,
        plan: NearFieldGraspPlan,
        heading_rad: float | None,
    ) -> bool | None:
        """检查冻结近场计划的完整前进行程是否仍在场内且避开双方安全区。"""

        position = self.estimated_field_position
        if position is None or heading_rad is None:
            return None
        plan_heading = normalize_angle(heading_rad + plan.alignment_angle_rad)
        distance_m = plan.forward_distance_mm / 1000.0
        return self._near_field_segment_clear(plan_heading, distance_m, plan=plan)

    def _preparation_with_current_alignment(
        self,
        preparation: GraspPreparation | None,
        timestamp_ns: int,
    ) -> tuple[GraspPreparation | None, bool]:
        """Express a delayed plan angle relative to the current IMU heading.

        The plan's opening geometry remains tied to its capture frame and is
        still age/stop validated before opening.  Only the remaining heading
        error is updated from the recorded capture pose, so a delayed result
        cannot command the old visual angle after the vehicle has moved.
        """

        if preparation is None or preparation.selection.plan is None:
            return preparation, False
        plan = preparation.selection.plan
        if abs(plan.alignment_angle_rad) <= 1e-9:
            return preparation, True
        capture_pose = self._pose_at(plan.capture_timestamp_ns)
        current_pose = self._pose_at(timestamp_ns)
        if capture_pose is None or current_pose is None:
            return preparation, False
        remaining = normalize_angle(
            capture_pose.heading_rad
            + plan.alignment_angle_rad
            - current_pose.heading_rad
        )
        updated_plan = replace(plan, alignment_angle_rad=remaining)
        updated_selection = replace(preparation.selection, plan=updated_plan)
        return replace(preparation, selection=updated_selection), True

    def _near_field_segment_clear(
        self,
        heading_rad: float,
        distance_m: float,
        *, plan: NearFieldGraspPlan | None = None,
    ) -> bool | None:
        if self._gate_clearance is not None and self._gate_clearance.reacquiring:
            return reacquire_path_clear(self, heading_rad, distance_m, plan)
        position = self.estimated_field_position
        if position is None or not math.isfinite(distance_m) or distance_m < 0.0:
            return None
        safe_zone_blocked = self._safe_zone_path_blocked(heading_rad, distance_m)
        if safe_zone_blocked is None or safe_zone_blocked:
            return None if safe_zone_blocked is None else False
        boundary = self._physical_field_bounds()
        margin = self._robot_path_clearance_mm()
        low_x, high_x, low_y, high_y = boundary
        low_x += margin
        high_x -= margin
        low_y += margin
        high_y -= margin
        end = FieldPoint(
            position.x + distance_m * 1000.0 * math.cos(heading_rad),
            position.y + distance_m * 1000.0 * math.sin(heading_rad),
        )
        return (
            low_x <= position.x <= high_x
            and low_y <= position.y <= high_y
            and low_x <= end.x <= high_x
            and low_y <= end.y <= high_y
        )

    @property
    def carried_target_count(self) -> int:
        """本趟已完成夹取的计划成员累计数；颜色门禁不估算物块数量。"""
        return len(self._transport_target_classes)

    def _cargo_is_legal(self) -> bool:
        cargo = self._transport_target_classes
        if self._transport_count == 0:
            return cargo == (TargetClass.GREEN_SUPPLY,)
        return cargo == (TargetClass.ORANGE_INJURED,) or (
            1 <= len(cargo) <= self._supply_capacity()
            and all(c in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE} for c in cargo)
        )

    def gripper_color_diagnostic(self, timestamp_ns: int) -> str:
        snapshot = self._latest_perception
        cargo = tuple(c.value for c in self._transport_target_classes)
        if snapshot is None or snapshot.gripper_color is None:
            return f"cargo={cargo},color_evidence=unavailable"
        evidence = snapshot.gripper_color
        orange_bbox_roi_overlap_count = self._orange_bbox_roi_overlap_count(snapshot)
        orange_candidates = self._orange_bbox_misgrasp_candidates(snapshot)
        orange_color_fractions = ",".join(
            f"{observation.color_segmentation.color_fraction:.4f}"
            for observation in self._orange_bbox_roi_observations(snapshot)
        )
        distinct_pair = self._distinct_orange_bbox_k0_pair(snapshot)
        distinct_pair_text = (
            "none"
            if distinct_pair is None
            else f"bbox_iou:{distinct_pair[0]:.4f},k0_distance_px:{distinct_pair[1]:.1f}"
        )
        fractions = ",".join(f"{c.value}:{f:.4f}" for c, f in evidence.component_fractions)
        return (
            f"cargo={cargo},frame={snapshot.frame_sequence},"
            f"capture_age_ms={(timestamp_ns-snapshot.capture_timestamp_ns)/1e6:.1f},"
            f"result_age_ms={(timestamp_ns-snapshot.result_timestamp_ns)/1e6:.1f},"
            f"cargo_capture_floor_ns={self._cargo_capture_floor_ns},"
            f"present={tuple(sorted(c.value for c in evidence.present_classes))},"
            f"black_raw_fraction={evidence.black_raw_fraction:.4f},"
            f"black_chromatic_fraction={evidence.black_chromatic_fraction:.4f},"
            f"orange_bbox_roi_overlap_count={orange_bbox_roi_overlap_count},"
            f"orange_bbox_color_qualified_count={len(orange_candidates)},"
            f"orange_bbox_color_fractions=({orange_color_fractions}),"
            f"orange_distinct_pair=({distinct_pair_text}),"
            f"components=({fractions})"
        )

    @staticmethod
    def _orange_bbox_roi_observations(
        snapshot: PerceptionSnapshot,
    ) -> tuple[TargetObservation, ...]:
        evidence = snapshot.gripper_color
        if evidence is None:
            return ()
        return tuple(
            observation
            for observation in snapshot.observations
            if observation.model_target_class is TargetClass.ORANGE_INJURED
            and bounding_box_overlaps_gripper_polygon(
                observation.box,
                evidence.polygon,
            )
        )

    @classmethod
    def _orange_bbox_roi_overlap_count(cls, snapshot: PerceptionSnapshot) -> int:
        return len(cls._orange_bbox_roi_observations(snapshot))

    def _orange_bbox_misgrasp_candidates(
        self,
        snapshot: PerceptionSnapshot,
    ) -> tuple[TargetObservation, ...]:
        minimum = self._gripper_color_config.orange_bbox_min_color_fraction
        return tuple(
            observation
            for observation in self._orange_bbox_roi_observations(snapshot)
            if observation.color_segmentation.candidate_class
            is TargetClass.ORANGE_INJURED
            and observation.color_segmentation.color_fraction >= minimum
        )

    def _distinct_orange_bbox_k0_pair(
        self,
        snapshot: PerceptionSnapshot,
    ) -> tuple[float, float] | None:
        """Return the first pair clearly separated by both bbox and model K0."""

        evidence = snapshot.gripper_color
        if evidence is None:
            return None
        candidates = self._orange_bbox_misgrasp_candidates(snapshot)
        config = self._gripper_color_config
        for index, first in enumerate(candidates):
            if first.k0 is None:
                continue
            for second in candidates[index + 1:]:
                if second.k0 is None:
                    continue
                bbox_iou = first.box.iou(second.box)
                k0_distance_px = math.hypot(
                    first.k0.u - second.k0.u,
                    first.k0.v - second.k0.v,
                )
                if (
                    bbox_iou <= config.orange_distinct_max_bbox_iou
                    and k0_distance_px
                    >= config.orange_distinct_min_k0_distance_px
                ):
                    return bbox_iou, k0_distance_px
        return None

    def _gripper_color_conflict(self, timestamp_ns: int) -> str | None:
        snapshot = self._latest_perception
        floor = self._cargo_capture_floor_ns
        safe_zone_final_push_or_exit = (
            self.state in {
                MatchState.TRANSPORT_FORWARD,
                MatchState.TRANSPORT_RELEASE,
            }
            and self._safe_zone_phase in {
                # Formal match closes before the final push; the transport
                # bench variant keeps the gripper open for the same phases.
                "closing_before_final_forward",
                "forward_final_closed",
                "forward_final_open",
                "stopping_before_exit_opening",
                "opening_after_transport",
            }
        )
        if (not self._transport_target_classes or floor is None
                or self.state is MatchState.RETURN_BACKUP
                or self._return_phase != "idle"
                or safe_zone_final_push_or_exit
                or snapshot is None or snapshot.gripper_color is None
                or snapshot.capture_timestamp_ns < floor
                or snapshot.result_timestamp_ns > timestamp_ns
                or not self._fresh_perception(snapshot, timestamp_ns)):
            return None
        allowed = (
            {TargetClass.GREEN_SUPPLY} if self._transport_count == 0 else
            {TargetClass.ORANGE_INJURED} if self._transport_target_classes == (TargetClass.ORANGE_INJURED,) else
            {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE}
        )
        conflicts = snapshot.gripper_color.present_classes - allowed
        if conflicts:
            return "conflicting_gripper_colors:" + ",".join(sorted(c.value for c in conflicts))
        if any(
            observation.model_target_class is TargetClass.BLUE_DANGER
            and bounding_box_overlaps_gripper_polygon(
                observation.box, snapshot.gripper_color.polygon,
                include_boundary=True,
            )
            for observation in snapshot.observations
        ):
            return "blue_detection_touches_gripper_roi"
        distinct_pair = self._distinct_orange_bbox_k0_pair(snapshot)
        if distinct_pair is not None:
            bbox_iou, k0_distance_px = distinct_pair
            return (
                "multiple_distinct_orange_detections_in_gripper_roi:"
                f"candidate_count={len(self._orange_bbox_misgrasp_candidates(snapshot))},"
                f"bbox_iou={bbox_iou:.4f},k0_distance_px={k0_distance_px:.1f}"
            )
        return None

    def _target_in_misgrasp_release(self, target: BreakupTarget) -> bool:
        """Associate fresh K0 with the fixed release footprint; never invent K0."""
        release = self._misgrasp_release_pose
        field = self._field_point_from_ground(target.center)
        if release is None or field is None:
            return False
        dx, dy = field.x - release.position.x, field.y - release.position.y
        c, s = math.cos(release.heading_rad), math.sin(release.heading_rad)
        x, y = c * dx + s * dy, -s * dx + c * dy
        jaws = GripperKinematics()
        margin = target.safety_radius_mm + self.config.safety_margin_mm
        return (jaws.pivot_x_mm - margin <= x <= jaws.left_tip_position(0).x + margin
                and abs(y) <= jaws.pivot_half_spacing_mm + margin)

    def _misgrasp_breakup_heading_allowed(self, plan: BreakupPlan) -> bool:
        release = self._misgrasp_release_pose
        # Compare against the fixed release heading, not the latest adjustment:
        # repeated small corrections cannot accumulate into a large turn.
        return (release is not None
                and abs(normalize_angle(plan.heading_rad - release.heading_rad))
                <= self.config.misgrasp_breakup_max_heading_change_rad)

    def _begin_misgrasp_recovery(self, timestamp_ns: int, reason: str) -> MatchDecision:
        self._misgrasp_release_pose = None
        self._misgrasp_breakup_active = False
        self.state = MatchState.MISGRASP_OPEN
        self._misgrasp_started_ns = timestamp_ns
        self._misgrasp_base_distance_m = None
        self._misgrasp_heading_rad = self._latest_heading_rad
        self._safe_zone_stop_since_ns = None
        self._grasp_task = None
        self._greedy_active = False
        self._near_field_handoff_prior = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id += 1
        return self._decision(timestamp_ns, 0.0, 0.0, "misgrasp_open:" + reason + ";" + self.gripper_color_diagnostic(timestamp_ns),
                              posture=GripperPosture.OPEN, soft_brake=True)

    def _step_misgrasp_recovery(self, timestamp_ns: int, distance_m: float | None) -> MatchDecision:
        heading = self._misgrasp_heading_rad
        if self.state is MatchState.MISGRASP_OPEN:
            # 开爪必须完成实际机械行程；不能复用当前被跳过的通用舵机等待。
            stationary_since = self._stationary_motion.stationary_since(timestamp_ns)
            if (timestamp_ns - self._misgrasp_started_ns < self._gripper_full_travel_time_ns
                    or stationary_since is None):
                return self._decision(timestamp_ns, 0.0, 0.0,
                    f"misgrasp_wait_open_stop:stationary_since_ns={stationary_since},"
                    f"open_deadline_ns={self._misgrasp_started_ns + self._gripper_full_travel_time_ns}",
                    posture=GripperPosture.OPEN, soft_brake=True)
            heading = self._latest_heading_rad
            if distance_m is None or heading is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "misgrasp_backup_requires_odometry_heading", posture=GripperPosture.OPEN)
            self._misgrasp_heading_rad = heading
            position = self.estimated_field_position
            self._misgrasp_release_pose = (None if position is None else FieldPose2D(position, heading))
            self._transport_target_classes = ()
            self._cargo_capture_floor_ns = None
            self._misgrasp_base_distance_m = distance_m
            self.state = MatchState.MISGRASP_BACKUP
        if self.state is MatchState.MISGRASP_BACKUP:
            evidence = self._stationary_motion
            latest = evidence.latest
            if (latest is None
                    or not 0 <= timestamp_ns - latest.received_timestamp_ns <= evidence.max_gap_ns
                    or evidence.invalid_reason in {
                        "sensor_invalid", "sample_overrun_or_gyro_saturated",
                        "host_sample_gap", "device_sample_gap", "duplicate_or_reversed_device_sample",
                    }):
                return self._decision(timestamp_ns, 0.0, 0.0,
                    "misgrasp_backup_control_state_invalid:" + evidence.diagnostic(timestamp_ns),
                    posture=GripperPosture.OPEN, soft_brake=True)
            base = self._misgrasp_base_distance_m
            if distance_m is None or base is None or heading is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "misgrasp_backup_requires_odometry_heading", posture=GripperPosture.OPEN)
            remaining_m = max(0.0, 0.250 - (base - distance_m))
            if remaining_m <= 1e-9:
                self.state = MatchState.MISGRASP_SETTLE
                return self._decision(timestamp_ns, 0.0, 0.0, "misgrasp_250mm_reached_stop", posture=GripperPosture.OPEN, soft_brake=True)
            if self._near_field_segment_clear(heading + math.pi, remaining_m) is not True:
                return self._decision(timestamp_ns, 0.0, 0.0, "misgrasp_backup_path_blocked_or_unlocalized", posture=GripperPosture.OPEN, soft_brake=True)
            angular = self._heading_hold_angular_velocity(
                heading, kp_rad_s=self.config.safe_zone_fallback_heading_kp_rad_s,
                max_angular_velocity_rad_s=self.config.safe_zone_fallback_max_angular_velocity_rad_s,
                tolerance_rad=self.config.safe_zone_fallback_heading_tolerance_rad,
            )
            if angular is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "misgrasp_backup_requires_heading", posture=GripperPosture.OPEN)
            deceleration = self._near_field_pickup.deceleration_m_s2 if self._near_field_pickup is not None else 0.5
            # 与接近曲线相同地预留 150 ms 响应路程，再按实际减速度收速；
            # 退出不是夹取精对准，不套用最后 50 mm 的一秒比例爬行。
            reaction_velocity = deceleration * 0.15
            braking_speed = math.sqrt(reaction_velocity**2 + 2 * deceleration * remaining_m) - reaction_velocity
            speed = min(self.config.return_backup_speed_m_s, max(0.005, braking_speed))
            return self._decision(timestamp_ns, -speed, angular,
                                  f"misgrasp_backup:remaining_mm={remaining_m * 1000:.1f}", posture=GripperPosture.OPEN)
        if self._stationary_motion.stationary_since(timestamp_ns) is None:
            return self._decision(timestamp_ns, 0.0, 0.0, "misgrasp_wait_reverse_stop:" + self._stationary_motion.diagnostic(timestamp_ns), posture=GripperPosture.OPEN, soft_brake=True)
        self._reset_tracker_for_new_preview_epoch()
        self._selected_track_id = None
        self._begin_cluster_search()
        self._reset_rotation_budget()
        if self._dynamic_breakup_enabled:
            self._misgrasp_breakup_active = True
            self._near_field_route_rejections = ()
            self._start_breakup_attempt(timestamp_ns)
            self._settle_until_ns = timestamp_ns + self._gripper_full_travel_time_ns
            return self._decision(timestamp_ns, 0.0, 0.0,
                "misgrasp_released_backed_250mm_breakup_locked_group",
                posture=GripperPosture.CLOSED, soft_brake=True)
        self.state = MatchState.SEARCH_CLUSTER
        # 不开放解团的独立入口继续原有重选流程。
        return self._decision(timestamp_ns, 0.0, self.config.cluster_search_angular_velocity_rad_s,
                              "misgrasp_released_backed_250mm_reselect", posture=GripperPosture.OPEN)

    _greedy_pickup_enabled = True

    def _supply_capacity(self) -> int:
        config = self._near_field_grasp_config
        return 3 if config is None else min(3, config.max_targets)

    def _can_greedy_pickup(self) -> bool:
        return (
            self._greedy_pickup_enabled
            and self._transport_count > 0
            and self._near_field_pickup is not None
            and 0 < len(self._transport_target_classes) < self._supply_capacity()
            and all(
                item in (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE)
                for item in self._transport_target_classes
            )
        )

    def _greedy_new_target_min_x_mm(self) -> float:
        config = self._near_field_grasp_config
        # 已收拢物资附近的观测不能当作新物资重复计数；保留其障碍证据。
        return (
            160.0 if config is None
            else config.greedy_target_final_x_mm + config.range_hysteresis_mm
        )

    def _begin_greedy_scan(self, timestamp_ns: int) -> MatchDecision:
        self._greedy_active = True
        self._greedy_started_ns = timestamp_ns
        self._greedy_last_heading = self._latest_heading_rad
        self._greedy_progress_rad = 0.0
        self.state = MatchState.TRANSPORT_GREEDY_SCAN
        self._selected_track_id = None
        return self._decision(
            timestamp_ns, 0.0, 0.0, "greedy_scan_started",
            posture=GripperPosture.CLOSED,
        )

    def _finish_greedy_pickup(self, timestamp_ns: int, reason: str) -> MatchDecision:
        if (
            self._greedy_active
            and (self._selected_track_id is not None or self._near_field_handoff_prior is not None)
            and reason not in {
                "scan_timeout", "scan_complete", "heading_unavailable",
                "supplementary_pickup_complete", "capacity_or_carried_member",
                "carried_or_delivered_member",
            }
        ):
            self._near_field_last_failure_diagnostic = (
                f"greedy_scan_resume reason={reason} "
                f"scan_started_ns={self._greedy_started_ns} "
                f"scan_age_ms={(timestamp_ns - self._greedy_started_ns) / 1e6:.1f} "
                f"scan_progress_rad={self._greedy_progress_rad:.3f}"
            )
            self._remember_near_field_failure(
                timestamp_ns, reason, self._breakup_grasp_preparation,
            )
            self._grasp_task = None
            self._near_field_handoff_prior = None
            self._breakup_grasp_preparation = None
            if self._near_field_pickup is not None:
                self._near_field_pickup.reset()
                self._near_field_session_id += 1
            self._selected_track_id = None
            self._selected_green_ground = None
            self._near_field_confirmation_started_ns = None
            self._near_field_route = GraspRoute.DECIDING
            self._greedy_last_heading = self._latest_heading_rad
            self.state = MatchState.TRANSPORT_GREEDY_SCAN
            # Preserve the original scan deadline and swept angle. A failed
            # physical target is excluded above, even after tracker renumbering.
            return self._step_greedy_scan(timestamp_ns, self._latest_heading_rad)
        self._greedy_active = False
        self._grasp_task = None
        self._near_field_handoff_prior = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id += 1
        return self._start_safe_zone_transport(
            timestamp_ns, transport_opened=False, posture=GripperPosture.CLOSED,
            reason=f"greedy_return:{reason}",
        )

    def _greedy_scan_angular_velocity_rad_s(self) -> float:
        """返回补夹扫描的当前角速度，单位 rad/s。

        当前帧没有可作为新物资的绿/黑信息（只有蓝色、已收拢或已交付物资）时
        使用全局空场搜索的快速角速度，避免在空场上慢速转满预算；出现可补夹
        信息后回到闭爪扫描角速度，不快速掠过刚出现的候选。方向沿用闭爪扫描
        方向，扫描过程中不换向。
        """

        perception = self._latest_perception
        stowed_limit_mm = self._greedy_new_target_min_x_mm()
        allowed_classes = self.near_field_policy.allowed_classes
        has_supply_information = perception is not None and any(
            observation.target_class in allowed_classes
            and self._observation_is_selectable(
                observation,
                stowed_limit_mm,
                perception,
            )
            for observation in perception.observations
        )
        magnitude = abs(
            self.config.close_gripper_spin_angular_velocity_rad_s
            if has_supply_information
            else self.config.cluster_search_empty_angular_velocity_rad_s
        )
        return math.copysign(
            magnitude,
            self.config.close_gripper_spin_angular_velocity_rad_s,
        )

    def _step_greedy_scan(
        self, timestamp_ns: int, heading_rad: float | None,
    ) -> MatchDecision:
        if self.carried_target_count >= self._supply_capacity():
            return self._finish_greedy_pickup(timestamp_ns, "capacity_or_carried_member")
        angular = self._greedy_scan_angular_velocity_rad_s()
        # 转速随当前帧信息量变化，时间上限必须按两者中较慢的一个计算，
        # 否则慢速段会被上限提前截断；完成仍以实际转角进度为准。
        slowest_angular_velocity_rad_s = min(
            abs(self.config.close_gripper_spin_angular_velocity_rad_s),
            abs(self.config.cluster_search_empty_angular_velocity_rad_s),
        )
        timeout_ns = round(
            (self.config.spin_angle_rad / slowest_angular_velocity_rad_s
             + self._near_field_handoff_timeout_ms() / 1000.0) * 1e9
        )
        if timestamp_ns - self._greedy_started_ns >= timeout_ns:
            return self._finish_greedy_pickup(timestamp_ns, "scan_timeout")
        if heading_rad is None:
            return self._finish_greedy_pickup(timestamp_ns, "heading_unavailable")
        previous = self._greedy_last_heading
        self._greedy_last_heading = heading_rad
        if previous is not None:
            self._greedy_progress_rad += self._directional_delta(
                previous, heading_rad, angular,
            )
        if self._greedy_progress_rad >= self.config.spin_angle_rad:
            return self._finish_greedy_pickup(timestamp_ns, "scan_complete")
        # 补夹与普通搜索共用入口候选门禁；这里只额外限制为本次扫描开始
        # 后的新观测，并按全场距离优先，不把近场交接半径当作发现范围。
        target = self._find_approach_seed(
            timestamp_ns,
            minimum_last_seen_ns=self._greedy_started_ns,
            prefer_nearest=True,
        )
        if target is not None:
            # 命中即进入全局夹取前置链：与搜索阶段找到目标走同一条
            # 对准—接近—近场唯一确认窗口，目标同时成为近场交接先验，
            # 由实际几何决定所需转角（需要反向时反向），不再重开一次
            # 自由选组会话。
            direct = self._transport_group_size(target, timestamp_ns) is not None
            return self._begin_green_transport(
                timestamp_ns,
                target,
                opportunistic=direct,
                group_preview=not direct,
            )
        return self._decision(
            timestamp_ns, 0.0, angular, "greedy_scan_supplies",
            posture=GripperPosture.CLOSED,
        )

    def _near_field_decision(
        self,
        timestamp_ns: int,
        decision: GripperWidthPickupDecision,
    ) -> MatchDecision:
        if self._near_field_pickup is None:
            return self._decision(
                timestamp_ns, 0.0, 0.0, "near_field_unavailable", posture=GripperPosture.CLOSED
            )
        if decision.state is GripperWidthPickupState.COMPLETE:
            result = self._near_field_pickup.result
            if result is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "near_field_complete_missing_result",
                    posture=GripperPosture.CLOSED,
                    soft_brake=True,
                )
            self._grasp_task = None
            self._near_field_handoff_prior = None
            self._near_field_far_reapproach_used = False
            picked_classes = tuple(TargetClass(item) for item in result.member_classes)
            if self._counted_pickup_session != self._near_field_session_id:
                self._transport_target_classes += picked_classes
                self._counted_pickup_session = self._near_field_session_id
                self._cargo_capture_floor_ns = timestamp_ns
            if not self._cargo_is_legal():
                return self._begin_misgrasp_recovery(timestamp_ns, "invalid_cargo_count_or_classes")
            if self._gate_clearance is not None and self._gate_clearance.reacquiring:
                self._gate_clearance = None
                self._greedy_active = False
                return self._start_safe_zone_transport(timestamp_ns, transport_opened=False,
                    posture=GripperPosture.CLOSED, reason="gate_clearance_reclaimed_start_normal_delivery")
            if self._greedy_active:
                # 补夹扫描已经命中并完成一次收拢：直接携已有物资返程，
                # 不再重复整圈扫描。剩余容量交由下一趟搜索决定。
                return self._finish_greedy_pickup(
                    timestamp_ns,
                    "supplementary_pickup_complete",
                )
            if self._can_greedy_pickup():
                return self._begin_greedy_scan(timestamp_ns)
            self._greedy_active = False
            return self._start_safe_zone_transport(
                timestamp_ns,
                transport_opened=False,
                posture=GripperPosture.CLOSED,
                reason="near_field_grasp_complete_start_safe_zone_d1_line",
            )
        if self._near_field_pickup.active_plan is not None:
            self._selected_track_id = self._near_field_pickup.active_plan.member_ids[0]
            # 已经提交了可执行计划：这次不再是"没有动作"的退出。
            self._near_field_reobserve_retries = 0
        elif self._near_field_pickup.locked_ids:
            self._selected_track_id = self._near_field_pickup.locked_ids[0]
        # 近场尚未形成可执行计划时继续闭爪；只有下面携带显式
        # ``gripper_angles_deg`` 的 OPENING/FORWARD 决策才会按实际
        # 目标包络映射张开角度。
        posture = GripperPosture.CLOSED
        if decision.state in {
            GripperWidthPickupState.CLOSING,
            GripperWidthPickupState.COMPLETE,
        }:
            posture = GripperPosture.CLOSED
        angles = decision.gripper_angles_deg
        if (
            angles is None
            and self._near_field_pickup.active_plan is not None
            and decision.state in {
                GripperWidthPickupState.OPENING,
                GripperWidthPickupState.FORWARD,
            }
        ):
            angles = self._near_field_pickup.active_plan.opening_servo_angles_deg
        if (
            angles is not None
            and decision.state
            in {
                GripperWidthPickupState.OPENING,
                GripperWidthPickupState.FORWARD,
            }
        ):
            # ``TRANSPORT`` here is only the semantic posture label; the
            # explicit angles above are the actual mapped servo command.
            posture = GripperPosture.TRANSPORT
        return self._decision(
            timestamp_ns,
            decision.linear_velocity_m_s,
            decision.angular_velocity_rad_s,
            f"near_field_{decision.state.value}:{decision.reason}",
            posture=posture,
            gripper_angles_deg=angles,
            soft_brake=decision.soft_brake,
            min_wheel_velocity_m_s=decision.min_wheel_velocity_m_s,
        )

    def _near_field_plan_has_delivered_member(
        self,
        plan: NearFieldGraspPlan | None,
    ) -> bool:
        """返回近场计划是否包含已经放入安全区的物资。

        蓝色危险证据不参与本判定：危险物进入安全区仍然要能阻挡夹爪扫掠。
        """

        if plan is None:
            return False
        stowed_limit = (
            self._greedy_new_target_min_x_mm()
            if self._greedy_active
            else None
        )
        return any(
            member.observation.target_class is not TargetClass.BLUE_DANGER
            and member.observation.model_target_class is not TargetClass.BLUE_DANGER
            and not self._observation_is_selectable(
                member.observation,
                stowed_limit,
                self._latest_perception,
            )
            for member in plan.members
        )

    def _latest_scene_preserves_grasp(self, plan: NearFieldGraspPlan, timestamp_ns: int) -> bool:
        """Cheap risk/core check against a newer scene while full planning runs."""
        scene = self._latest_perception
        if scene is None or scene.capture_timestamp_ns <= plan.capture_timestamp_ns:
            return True
        observations = scene.observations
        matched: set[int] = set()
        for member in plan.members:
            center = member.observation.ground_point
            if center is None:
                return False
            candidates = [(math.hypot(item.ground_point.x-center.x, item.ground_point.y-center.y), index)
                          for index, item in enumerate(observations)
                          if index not in matched and item.target_class is member.observation.target_class
                          and item.ground_point is not None]
            if not candidates or min(candidates)[0] > 30.0:
                return False
            matched.add(min(candidates)[1])
        for index, item in enumerate(observations):
            if index in matched:
                continue
            if item.ground_point is None:
                if item.target_class is TargetClass.BLUE_DANGER or item.model_target_class is TargetClass.BLUE_DANGER:
                    return False
                continue
            radius = 0.0
            if (
                item.target_class is TargetClass.BLUE_DANGER
                or item.model_target_class is TargetClass.BLUE_DANGER
            ):
                if self._breakup_target_geometry is None:
                    return False
                radius, _ = physical_radii(
                    self._breakup_target_geometry.geometry_for(TargetClass.BLUE_DANGER)
                )
            if any(polygon_distance((item.ground_point,), region) <= radius for region in plan.regions):
                return False
        return True

    def _step_near_field_grasp(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
        preparation: GraspPreparation | None,
        path_clear: bool | None,
    ) -> MatchDecision:
        assert self._near_field_pickup is not None
        settling = self._consume_action_settle(
            timestamp_ns,
            "near_field_grasp",
            posture=GripperPosture.TRANSPORT,
            reason="near_field_waiting_after_handoff",
        )
        if settling is not None:
            return settling
        if (self._near_field_pickup.active_plan is None
                and self._stationary_motion.latest is not None
                and self._near_field_pickup.state is not GripperWidthPickupState.ALIGNING
                and not self.near_field_observation_window_open(timestamp_ns)):
            return self._decision(timestamp_ns, 0.0, 0.0,
                                  "grasp_task_waiting_stationary_scene", soft_brake=True)
        if preparation is not None and preparation.session_id != self._near_field_session_id:
            preparation = None
        gate = self._gate_clearance
        if (
            gate is not None
            and gate.reacquiring
            and preparation is not None
            and preparation.selection.plan is not None
            and not reacquire_plan_matches_cargo(
                gate,
                preparation.selection.plan,
            )
        ):
            started = self._near_field_confirmation_started_ns
            if started is None:
                self._near_field_confirmation_started_ns = timestamp_ns
                started = timestamp_ns
            elapsed_ms = (timestamp_ns - started) / 1_000_000.0
            if elapsed_ms >= self.config.gate_clearance.observation_timeout_ms:
                return self._return_to_near_field_search(
                    timestamp_ns,
                    "original_cargo_incomplete",
                )
            planned = tuple(
                member.observation.target_class.value
                for member in preparation.selection.plan.members
            )
            required = tuple(item.value for item in gate.original_classes)
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "gate_clearance_reacquire_waiting_complete_cargo:"
                f"planned={planned},required={required},"
                f"confirmation=0/1,capture_age_ms="
                f"{self._preparation_observation_age_ms(preparation, timestamp_ns)},"
                f"preparation_age_ms="
                f"{self._preparation_result_age_ms(preparation, timestamp_ns)},"
                f"deadline_ns={started + round(self.config.gate_clearance.observation_timeout_ms * 1_000_000.0)}",
                posture=GripperPosture.OPEN,
                soft_brake=True,
            )
        task = self._grasp_task
        if task is not None and preparation is not None:
            if preparation.capture_timestamp_ns <= task.scene_floor_ns:
                preparation = None
        if (preparation is not None and preparation.selection.plan is not None
                and self._near_field_pickup.active_plan is None
                and not self._latest_scene_preserves_grasp(preparation.selection.plan, timestamp_ns)):
            preparation = None
        if preparation is not None and self._near_field_plan_has_delivered_member(
            preparation.selection.plan
        ):
            # 送入 worker 的观测过滤基于采集时刻位姿；消费层再按当前位姿
            # 复核一次。已交付物资不能重新成为候选，也不能冻结成执行计划。
            if self._greedy_active:
                return self._finish_greedy_pickup(
                    timestamp_ns,
                    "carried_or_delivered_member",
                )
            preparation = None
        if self._greedy_active and preparation is not None:
            plan = preparation.selection.plan
            if plan is not None and (
                len(plan.members) > self._supply_capacity() - len(self._transport_target_classes)
                or any(
                    member.observation.target_class not in self.near_field_policy.allowed_classes
                    or member.observation.ground_point is None
                    or not self._observation_is_selectable(
                        member.observation,
                        self._greedy_new_target_min_x_mm(),
                        self._latest_perception,
                    )
                    for member in plan.members
                )
            ):
                return self._finish_greedy_pickup(timestamp_ns, "capacity_or_carried_member")
        if (task is not None and preparation is not None
                and preparation.selection.plan is not None):
            members = preparation.selection.plan.members
            # 首个可执行核心按场地坐标判定是否仍是入口的同一物理目标：
            # 机器人系坐标被接近/制动位移整体污染，不能用来判断换目标。
            if (not task.core and preparation.ready and path_clear is True
                    and self._stationary_motion.capture_valid(preparation.capture_timestamp_ns, timestamp_ns,
                        max_age_ns=round(self._near_field_handoff_timeout_ms()*1e6))
                    and not any(self._member_continues_grasp_entry(member, task, timestamp_ns)
                                for member in members)):
                # A genuinely different executable core is an explicit objective
                # replacement. Record the old physical entry before adopting the
                # new core; retain this task's deadline and worker session.
                self._remember_near_field_failure(timestamp_ns, "entry_replaced_by_executable_core")
                task.failure_recorded_revision = -1
                task.last_progress_ns = timestamp_ns
                self._near_field_handoff_prior = None
            if not task.core:
                self._refresh_grasp_task_entry(members[0])
            task.core = members
            task.blocking = preparation.selection.rejections
        if preparation is not None and preparation.action is GraspSceneAction.RECOVERY:
            candidate = preparation.recovery_plan
            assert candidate is not None
            age_ns = preparation.preparation_age_ns(timestamp_ns)
            if (age_ns is None or age_ns > self.config.green_max_age_ms * 1e6
                    or not self._stationary_motion.capture_valid(
                        preparation.capture_timestamp_ns, timestamp_ns,
                        max_age_ns=round(self._near_field_handoff_timeout_ms() * 1e6))):
                started = self._near_field_confirmation_started_ns
                if started is not None and timestamp_ns-started >= self._near_field_no_plan_wait_ms()*1e6:
                    return self._return_to_near_field_search(timestamp_ns, "recovery_scene_invalid")
                return self._decision(timestamp_ns, 0.0, 0.0, "recovery_waiting_valid_scene", soft_brake=True)
            if self._greedy_active or self._transport_target_classes:
                return self._finish_greedy_pickup(timestamp_ns, "loaded_recovery_disabled")
            if self._latest_heading_rad is None or cumulative_distance_m is None:
                return self._decision(timestamp_ns, 0.0, 0.0, "recovery_missing_control_state", soft_brake=True)
            if not preparation.ready:
                return self._decision(timestamp_ns, 0.0, 0.0, "grasp_recovery_confirmation", soft_brake=True)
            if not self._mark_breakup_targets(preparation):
                return self._return_to_near_field_search(timestamp_ns, "breakup_missing_scoring_target")
            task = self._grasp_task
            if task is not None:
                task.blocking = preparation.selection.rejections
                task.recovery_aim = candidate.aim_field
            error = normalize_angle(candidate.heading_rad-self._latest_heading_rad)
            if abs(error) > self._breakup_alignment_tolerance(candidate):
                self._breakup_proposal = candidate
                # 近场已经判定当前目标抓不了，这份接触计划就是结论本身；
                # 对准转向只修正接触射线，不能让计划失效。
                self._breakup_frozen_plan = candidate
                self._breakup_alignment_done = False
                # Frozen turn is executed by the existing IMU action path;
                # after it ends, the same task requests a new stable scene.
                self._breakup_phase_started_ns = timestamp_ns
                self._breakup_observation_deadline_ns = None
                self._breakup_alignment_allowance_ns = round(self._breakup_alignment_time_s(candidate)*1e9)
                self.state = MatchState.BREAKUP_SETTLE
                return self._decision(timestamp_ns, 0.0, 0.0, "grasp_recovery_turn_frozen", soft_brake=True)
            self._breakup_plan = candidate
            self._breakup_forward_base_distance_m = cumulative_distance_m
            self._breakup_segment_stop_started_ns = None
            self._breakup_segment_stop_deadline_ns = None
            self._breakup_retreat_mm = candidate.backward_distance_mm
            self._breakup_actual_forward_mm = self._breakup_actual_backward_mm = 0.0
            self._safe_zone_stop_since_ns = None
            self.state = MatchState.BREAKUP_FORWARD
            return self._decision(timestamp_ns, 0.0, 0.0, "grasp_recovery_frozen", soft_brake=True)
        if preparation is not None and preparation.action is GraspSceneAction.EXIT:
            return self._return_to_near_field_search(timestamp_ns, "no_safe_grasp_or_recovery")
        if task is not None and self._near_field_pickup.active_plan is None:
            if preparation is None or preparation.action is GraspSceneAction.OBSERVE:
                started = self._near_field_confirmation_started_ns
                if started is None:
                    self._near_field_confirmation_started_ns = timestamp_ns
                    started = timestamp_ns
                # A frozen IMU turn keeps its progress when no new plan is ready.
                # 对准转向有自己的预算，也存在可执行计划；此时既不能判定
                # "没有方案"，也不能把转向时长算进那个短窗口。
                if self._near_field_pickup.state is not GripperWidthPickupState.ALIGNING:
                    waiting_ms = (timestamp_ns-started)/1e6
                    # "完全没有方案"才用短窗口；已经定下路由或锁定成员的会话
                    # 处于提交阶段，用独立的提交窗口，否则一次瞬时丢帧就会在
                    # 提交前把整个近场会话判死。
                    budget_ms = (
                        self._near_field_handoff_timeout_ms()
                        if self.grasp_planning_pending
                        or self._near_field_has_executable_plan()
                        else self._near_field_no_plan_wait_ms()
                    )
                    if waiting_ms >= budget_ms:
                        return self._return_to_near_field_search(timestamp_ns, "scene_evidence_unavailable")
                    return self._decision(timestamp_ns, 0.0, 0.0,
                                          "grasp_scene_computing" if self.grasp_planning_pending else "grasp_scene_evidence_pending",
                                          soft_brake=True)
            elif preparation.action in {GraspSceneAction.GRASP, GraspSceneAction.MOTION}:
                self._near_field_route = GraspRoute.DIRECT_NEAR
        if self._near_field_handoff_target_missing(preparation):
            next_target = self._find_approach_seed(
                timestamp_ns,
                minimum_last_seen_ns=self._greedy_started_ns,
                prefer_nearest=True,
            )
            if next_target is not None:
                self._near_field_handoff_prior = None
                decision = self._begin_green_transport(
                    timestamp_ns,
                    next_target,
                    group_preview=True,
                )
                return replace(
                    decision,
                    reason=(
                        "greedy_target_disappeared_next:"
                        f"{next_target.track_id}"
                    ),
                )
        step_preparation = preparation
        if (
            self._near_field_pickup.active_plan is None
            and self._near_field_pickup.locked_ids is None
            and self._near_field_route is GraspRoute.DECIDING
        ):
            route = self._near_field_route_decision(timestamp_ns, preparation)
            if route is None:
                step_preparation = None
            elif route.route is GraspRoute.DIRECT_NEAR:
                # 路由只决定一次候选入口；真正的确认由准备器的唯一窗口
                # 完成，计划几何仍随同一目标 ID 更新到最新有效帧。
                step_preparation = preparation
            elif route.route is GraspRoute.FAR_REAPPROACH:
                assert route.target is not None
                self._near_field_handoff_prior = None
                decision = self._begin_green_transport(
                    timestamp_ns,
                    route.target,
                    group_preview=True,
                )
                return replace(
                    decision,
                    reason=f"near_field_route:far_reapproach:{route.target.track_id}",
                )
            elif route.route is GraspRoute.RESELECT:
                return self._return_to_near_field_search(
                    timestamp_ns,
                    "confirmation_timeout",
                )
            elif route.route is GraspRoute.BREAKUP:
                return self._enter_breakup_only_search(timestamp_ns)
        if (
            self._near_field_route is GraspRoute.DIRECT_NEAR
            and self._near_field_pickup.active_plan is None
            and self._near_field_confirmation_started_ns is not None
            and timestamp_ns - self._near_field_confirmation_started_ns
            >= round(self._near_field_handoff_timeout_ms() * 1_000_000.0)
        ):
            return self._return_to_near_field_search(
                timestamp_ns,
                "confirmation_timeout",
            )
        step_preparation, _ = self._preparation_with_current_alignment(
            step_preparation,
            timestamp_ns,
        )
        plan = step_preparation.selection.plan if step_preparation is not None else None
        if (
            plan is not None
            and preparation is not None
            and self._near_field_route is not GraspRoute.DECIDING
            and plan.alignment_angle_rad == 0.0
            and path_clear is None
            and (
                self._near_field_pickup.locked_ids is None
                or self._near_field_pickup.locked_ids == plan.member_ids
            )
            and not self._near_field_pickup.alignment_timeout_reached(
                timestamp_ns
            )
        ):
            self._near_field_pickup.hold_for_static_path_validation(
                timestamp_ns,
                plan.member_ids,
            )
            return self._decision(
                timestamp_ns, 0.0, 0.0,
                "near_field_waiting_for_static_path_validation",
                posture=GripperPosture.TRANSPORT,
                soft_brake=True,
            )
        decision = self._near_field_pickup.step(
            timestamp_ns,
            step_preparation,
            cumulative_distance_m=cumulative_distance_m,
            path_clear=True if path_clear is None else path_clear,
            heading_rad=self._latest_heading_rad,
        )
        if (
            decision.state is GripperWidthPickupState.ABORTED
            and decision.reason != "near_field_path_blocked"
        ):
            self._near_field_last_failure_diagnostic = (
                "near_field_motion_failure "
                f"session={self._near_field_session_id} "
                f"reason={decision.reason} "
                f"progress_mm={self._near_field_pickup.progress_mm(cumulative_distance_m):.1f} "
                f"observation_age_ms={self._preparation_observation_age_ms(step_preparation, timestamp_ns)} "
                f"preparation_age_ms={self._preparation_result_age_ms(step_preparation, timestamp_ns)}"
            )
            if decision.reason in {
                "critical_odometry_unavailable",
                "encoder_direction_mismatch",
                "encoder_no_forward_progress",
            }:
                self._remember_near_field_failure(
                    timestamp_ns,
                    decision.reason,
                    step_preparation,
                )
                return self._return_to_near_field_search(
                    timestamp_ns,
                    f"near_field_motion_failure:{decision.reason}",
                )
            return self._return_to_near_field_search(
                timestamp_ns,
                f"pickup_aborted:{decision.reason}",
            )
        if decision.reason == "near_field_path_blocked":
            plan_heading = (
                None if plan is None or self._latest_heading_rad is None
                else self._latest_heading_rad + plan.alignment_angle_rad
            )
            safe_zone_blocked = (
                None if plan is None or plan_heading is None
                else self._safe_zone_path_blocked(plan_heading, plan.forward_distance_mm / 1000.0)
            )
            self._near_field_last_failure_diagnostic = (
                "near_field_route_failure "
                f"kind={decision.reason} session={self._near_field_session_id} "
                f"field_position={self.estimated_field_position} "
                f"heading_rad={self._latest_heading_rad} "
                f"plan_heading_rad={plan_heading} safe_zone_blocked={safe_zone_blocked} "
                f"forward_distance_mm={None if plan is None else plan.forward_distance_mm} "
                f"field_bounds_mm={self._breakup_allowed_field_bounds()} "
                f"clearance_mm={self._breakup_clearance_mm}"
            )
            # Moving objects cannot resolve a field boundary or safe-zone
            # intersection. Keep searching for a statically valid approach.
            return self._return_to_near_field_search(
                timestamp_ns,
                "static_path_blocked",
            )
        if decision.reason.startswith("candidate_replan:"):
            self._near_field_last_failure_diagnostic = (
                "near_field_route_failure "
                f"kind={decision.reason} session={self._near_field_session_id}"
            )
            geometric_block = any(
                token in decision.reason
                for token in (
                    "blocked_target:",
                    "side_adjacent_incompatible",
                    "orange_not_isolated_track:",
                    "maximum_opening_exceeded",
                    "left_tip_y_mm",
                    "right_tip_y_mm",
                )
            )
            return self._route_after_near_field_failure(
                timestamp_ns,
                reason="candidate_replan",
                geometric_block=geometric_block,
            )
        if decision.reason in {"candidate_replan_exhausted", "alignment_timeout"}:
            self._near_field_last_failure_diagnostic = (
                "near_field_route_failure "
                f"kind={decision.reason} session={self._near_field_session_id}"
            )
            return self._route_after_near_field_failure(
                timestamp_ns,
                reason=(
                    "confirmation_timeout"
                    if decision.reason == "alignment_timeout"
                    else "candidate_replan_exhausted"
                ),
            )
        return self._near_field_decision(timestamp_ns, decision)

    def _hold_or_restart_green_target(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """目标短时失鲜先停车等待，超时后重新解团，禁止旧坐标驱动旋转。"""

        lost_since = self._green_align_lost_since_ns
        if lost_since is None:
            self._green_align_lost_since_ns = timestamp_ns
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_align_waiting_for_current_ground_point",
                posture=GripperPosture.TRANSPORT,
            )
        if timestamp_ns - lost_since < self._green_align_hold_ns:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "green_align_waiting_for_current_ground_point",
                posture=GripperPosture.TRANSPORT,
            )
        if self._greedy_active:
            if self._greedy_selected_target_missing(timestamp_ns):
                alternative = self._find_approach_seed(
                    timestamp_ns,
                    minimum_last_seen_ns=self._greedy_started_ns,
                    prefer_nearest=True,
                )
                if alternative is not None:
                    self._near_field_handoff_prior = None
                    return self._begin_green_transport(
                        timestamp_ns,
                        alternative,
                        group_preview=True,
                    )
                return self._finish_greedy_pickup(
                    timestamp_ns,
                    "target_disappeared_no_next_target",
                )
            return self._finish_greedy_pickup(
                timestamp_ns,
                "target_geometry_unavailable",
            )
        self._green_align_lost_since_ns = None
        return self._return_to_near_field_search(timestamp_ns, "target_geometry_lost")

    def _greedy_selected_target_missing(self, timestamp_ns: int) -> bool:
        """判断补夹锁定入口是否真的不再可跟踪。"""

        target = self._selected_target()
        # A visible track with a temporarily missing K0/class-quality
        # measurement is not a disappearance.  Keep its identity fixed and
        # let the normal geometry/confirmation path hold or finish the load.
        return target is None or not self._target_is_fresh(target, timestamp_ns)

    def _selected_green_point(self, timestamp_ns: int) -> GroundPoint | None:
        """Return the selected target in the current robot frame.

        A target observation is expressed at its camera capture time.  When
        there is enough recorded encoder/IMU history, transform it through
        that capture pose and the current pose.  If the history does not cover
        the interval, return no compensated point; callers then hold or use a
        bounded action and wait for a fresh frame instead of fabricating a
        time compensation.
        """

        target = self._selected_target()
        if target is None:
            return None
        if (
            self._target_is_fresh(target, timestamp_ns)
            and self._selected_green_is_usable(target)
            and target.ground_point is not None
        ):
            current = self._ground_point_at_now(
                target.ground_point,
                target.last_seen_timestamp_ns,
                timestamp_ns,
            )
            if current is None:
                current_pose = self._pose_at(timestamp_ns)
                if self._green_target_field_point is not None and current_pose is not None:
                    delta_x = self._green_target_field_point.x - current_pose.position.x
                    delta_y = self._green_target_field_point.y - current_pose.position.y
                    cosine = math.cos(current_pose.heading_rad)
                    sine = math.sin(current_pose.heading_rad)
                    current = GroundPoint(
                        cosine * delta_x + sine * delta_y,
                        -sine * delta_x + cosine * delta_y,
                    )
            if current is None:
                # A capture at this exact control time is valid even when the
                # history has only one sample; no motion compensation is then
                # needed.
                if target.last_seen_timestamp_ns == timestamp_ns:
                    current = target.ground_point
                else:
                    return None
            self._selected_green_ground = current
            if self._green_target_field_point is None:
                pose = self._pose_at(target.last_seen_timestamp_ns)
                if pose is not None:
                    self._green_target_field_point = self._field_point_from_pose(
                        pose,
                        target.ground_point,
                    )
            return current
        return None

    def _step_transport_close(self, timestamp_ns: int) -> MatchDecision:
        """合爪完成后直接进入场地坐标运输，不扫描安全区。"""

        if not self._gripper_action_completed(timestamp_ns):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "closing_gripper_for_transport",
                posture=GripperPosture.CLOSED,
            )
        self._transport_target_classes = (TargetClass.GREEN_SUPPLY,) * max(1, self._green_preclose_carried_count)
        self._cargo_capture_floor_ns = timestamp_ns
        if not self._cargo_is_legal():
            return self._begin_misgrasp_recovery(timestamp_ns, "invalid_cargo_count_or_classes")
        return self._start_safe_zone_transport(
            timestamp_ns,
            transport_opened=False,
            posture=GripperPosture.CLOSED,
            reason="gripper_closed_start_safe_zone_d1_line",
        )

    def _start_safe_zone_transport(
        self,
        timestamp_ns: int,
        *,
        transport_opened: bool,
        posture: GripperPosture,
        reason: str,
    ) -> MatchDecision:
        """初始化 d1 安全区运输路线并返回首个路线意图。"""

        if self._transport_target_classes and self._cargo_capture_floor_ns is None:
            self._cargo_capture_floor_ns = timestamp_ns
        self.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
        self._transport_opened = transport_opened
        self._gripper_phase_started_ns = None
        self._safe_zone_phase = "align_d1_line"
        self._safe_zone_observation_deadline_ns = None
        self._safe_zone_confirmation_deadline_ns = None
        self._begin_action_settle(timestamp_ns, "safe_before_d1_line")
        self._transport_forward_base_distance_m = None
        self._transport_forward_distance_m = None
        self._safe_zone_key_samples = []
        self._safe_zone_key_last_frame = None
        self._safe_zone_key_reobserve_until_ns = None
        self._safe_zone_key_reobserve_frame_floor = None
        self._safe_zone_keypoint_reverse_base_distance_m = None
        self._safe_zone_keypoint_reverse_attempted = False
        self._safe_zone_calibration_snapshot = None
        self._safe_zone_calibration_zone = None
        self._safe_zone_keys = None
        self._safe_zone_calibration_pose = None
        self._safe_zone_calibration_heading_rad = None
        self._safe_zone_calibration_last_failure = None
        self._safe_zone_stop_since_ns = None
        self._safe_zone_bbox_turn_direction = None
        self._return_phase = "idle"
        self._safe_zone_exit_base_distance_m = None
        self._d1_line_heading_rad = None
        self._d1_line_distance_m = None
        self._d1_line_start_position = None
        self._d1_calibration_offset_mm = (
            self.config.safe_zone_calibration_start_offset_mm
        )
        self._d2_line_heading_rad = None
        self._d2_line_distance_m = None
        self._d2_line_start_position = None
        self._search_frame_floor = None
        position = self.estimated_field_position
        d1_target = self._safe_zone_d1_target()
        if position is not None and self._safe_zone_forward_y_sign * (
            position.y - d1_target.y
        ) >= -self.config.transport_align_tolerance_mm:
            minimum_target = self._safe_zone_d1_target_for_offset(
                self.config.safe_zone_calibration_min_offset_mm
            )
            if self._safe_zone_forward_y_sign * (
                position.y - minimum_target.y
            ) > self.config.transport_align_tolerance_mm:
                self._d1_calibration_offset_mm = (
                    self.config.safe_zone_calibration_min_offset_mm
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "gripper_closed_too_close_for_d1_calibration_reposition",
                    posture=posture,
                )
            self._safe_zone_phase = "stopping_before_calibration"
            self.state = MatchState.TRANSPORT_RELEASE
            return self._decision(
                timestamp_ns, 0.0, 0.0,
                "gripper_closed_already_beyond_d1_start_calibration",
                posture=posture,
            )
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            reason,
            posture=posture,
        )

    def _own_safe_zone_observation(self):
        perception = self._latest_perception
        if (
            perception is None
            or perception.dropped_stale_age_ms is not None
            or perception.field_features is None
        ):
            return None
        expected = SafeZoneColor(self._team_color.value)
        candidates = tuple(
            zone
            for zone in perception.field_features.safe_zones
            if zone.physical_color is expected
        )
        return max(candidates, key=lambda zone: zone.confidence, default=None)

    def _safe_zone_required_indices(self) -> tuple[int, int]:
        """按当前 bbox 图像位置选对侧；采样期间锁定同一对点。"""
        if self._safe_zone_key_samples:
            return (0, 2) if self._safe_zone_key_samples[0][2] is not None else (0, 1)
        zone = self._own_safe_zone_observation()
        perception = self._latest_perception
        if zone is None or perception is None or perception.field_features is None:
            return (0, 1)
        width, _ = perception.field_features.image_size
        return (0, 2) if zone.box.x_min + zone.box.x_max < width else (0, 1)

    def _safe_zone_required_keypoints(self, zone: SafeZoneObservation):
        points = (zone.ground_anchor, zone.image_left_landmark, zone.image_right_landmark)
        return tuple(points[index] for index in self._safe_zone_required_indices())

    def _safe_zone_has_required_ground_keypoints(self, zone: SafeZoneObservation | None) -> bool:
        return zone is not None and all(
            point.undistorted is not None and point.ground is not None
            for point in self._safe_zone_required_keypoints(zone)
        )

    @staticmethod
    def _safe_zone_keypoint_count(zone: SafeZoneObservation | None) -> int:
        """返回当前帧同时具有像素和地面坐标的安全区关键点数量。"""

        if zone is None:
            return 0
        return sum(
            keypoint.undistorted is not None and keypoint.ground is not None
            for keypoint in (
                zone.ground_anchor,
                zone.image_left_landmark,
                zone.image_right_landmark,
            )
        )

    def _safe_zone_bbox_center_text(self) -> str:
        """返回当前安全区 bbox 与图像水平中心，供现场诊断使用。"""

        perception = self._latest_perception
        zone = self._own_safe_zone_observation()
        if perception is None or perception.field_features is None or zone is None:
            return "none"
        image_width, _ = perception.field_features.image_size
        bbox_center_u = 0.5 * (zone.box.x_min + zone.box.x_max)
        box = zone.box
        return (f"bbox_u={bbox_center_u:.1f},image_center_u={image_width / 2.0:.1f},"
                f"bbox_xyxy=({box.x_min:.1f},{box.y_min:.1f},{box.x_max:.1f},{box.y_max:.1f}),"
                f"bbox_fully_visible={self._safe_zone_bbox_fully_visible(zone)}")

    def _safe_zone_bbox_fully_visible(self, zone: SafeZoneObservation | None) -> bool:
        """裁剪框的关键点不可信；完整框四边均须保留配置余量。"""
        perception = self._latest_perception
        if zone is None or perception is None or perception.field_features is None:
            return False
        width, height = perception.field_features.image_size
        margin = self.config.safe_zone_bbox_edge_margin_px
        box = zone.box
        return (margin <= box.x_min < box.x_max <= width - margin
                and margin <= box.y_min < box.y_max <= height - margin)

    def _safe_zone_required_points_visible(
        self,
        zone: SafeZoneObservation | None,
    ) -> bool:
        """完整框与所选两点均离开裁剪边缘，第三点允许缺失。"""

        perception = self._latest_perception
        if zone is None or perception is None or perception.field_features is None:
            return False
        if not self._safe_zone_bbox_fully_visible(zone):
            return False
        width, height = perception.field_features.image_size
        margin = self.config.safe_zone_bbox_edge_margin_px
        return all(
            point.undistorted is not None
            and margin <= point.undistorted.u <= width - margin
            and margin <= point.undistorted.v <= height - margin
            for point in self._safe_zone_required_keypoints(zone)
        )

    def _safe_zone_no_bbox_turn_command(self) -> float:
        """无 bbox 时沿当前朝己方安全区的搜索方向持续转动。

        这里只选择一次方向，不把安全区方向当作必须到达的目标航向；安全区
        的完整 K0/K1/K2 一旦出现，调用方会立即给出零速停车。
        """

        direction = self._safe_zone_bbox_turn_direction
        if direction is None:
            heading = self._latest_heading_rad
            if heading is None:
                direction = -1.0
            else:
                heading_error = normalize_angle(
                    self._safe_zone_forward_heading_rad() - heading
                )
                direction = 1.0 if heading_error > 1e-6 else -1.0
            self._safe_zone_bbox_turn_direction = direction
        # 夹取后看不到安全区时直接使用全局回退最大角速度，尽快重新捕获；
        # bbox 一旦出现，调用方再切回视觉限速。
        return direction * abs(self.config.safe_zone_fallback_max_angular_velocity_rad_s)

    def _safe_zone_forward_heading_rad(self) -> float:
        """返回从 d2 驶入己方安全区的场地航向。"""

        return self._safe_zone_forward_y_sign * math.pi / 2.0

    def _safe_zone_forward_heading_label(self) -> str:
        """返回用于诊断 reason 的 ``plus_90``/``minus_90`` 标签。"""

        return "plus_90" if self._safe_zone_forward_y_sign > 0.0 else "minus_90"

    def _safe_zone_heading_label(self) -> str:
        """返回用于兼容既有 reason 的 ``90``/``minus_90`` 标签。"""

        return "90" if self._safe_zone_forward_y_sign > 0.0 else "minus_90"

    def _step_safe_zone_bbox_key_search(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """可靠对侧两点直接取样，仅裁剪遮挡时调整视野。"""

        posture = (
            GripperPosture.CLOSED
        )
        perception = self._latest_perception
        zone = self._own_safe_zone_observation()
        if zone is None:
            return self._decision(
                timestamp_ns,
                0.0,
                self._safe_zone_no_bbox_turn_command(),
                "safe_zone_no_bbox_rotate_toward_"
                f"{self._safe_zone_forward_heading_label()}",
                posture=posture,
            )
        if (
            self._safe_zone_required_points_visible(zone)
            and self._safe_zone_has_required_ground_keypoints(zone)
        ):
            self._safe_zone_phase = "stopping_after_bbox_keypoints"
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_seen_stop_before_calibration",
                posture=posture,
            )
        # 仅缺视野才调整；框内缺点先等一轮新证据，不追两点中点。
        if perception is not None and perception.field_features is not None:
            width, height = perception.field_features.image_size
            margin = self.config.safe_zone_bbox_edge_margin_px
            left_clipped = zone.box.x_min < margin
            right_clipped = zone.box.x_max > width - margin
            vertical_clipped = zone.box.y_min < margin or zone.box.y_max > height - margin
            if ((left_clipped and right_clipped) or vertical_clipped):
                if not self._safe_zone_keypoint_reverse_attempted:
                    self._begin_safe_zone_keypoint_reverse()
                    return self._step_safe_zone_keypoint_reverse(timestamp_ns)
            elif left_clipped or right_clipped:
                self._safe_zone_observation_deadline_ns = None
                self._safe_zone_confirmation_deadline_ns = None
                self._safe_zone_bbox_turn_direction = 1.0 if left_clipped else -1.0
                self._safe_zone_phase = "scanning_safe_zone_keypoints"
                return self._step_safe_zone_keypoint_scan(timestamp_ns)
        self._begin_safe_zone_keypoint_reobserve(timestamp_ns)
        return self._decision(timestamp_ns, 0.0, 0.0,
                              "safe_zone_keypoints_missing_start_reobserve", posture=posture)

    def _transport_cruise_speed(self, target: FieldPoint, start: FieldPoint | None, baseline: float) -> float:
        position = self._fallback_field_position
        if position is None or start is None or self.config.pickup_cruise_speed_scale == 1.0:
            return baseline
        dx, dy = target.x - start.x, target.y - start.y
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return baseline
        # Along-line distance matches the coordinate crossing stop gate; a
        # lateral offset must not keep cruise speed active past the endpoint.
        remaining = ((target.x - position.x) * dx + (target.y - position.y) * dy) / length
        remaining = (remaining - self.config.transport_align_tolerance_mm) / 1000.0
        deceleration = self._near_field_pickup.deceleration_m_s2 if self._near_field_pickup is not None else 0.5
        return approach_speed_m_s(remaining, baseline, self.config.pickup_cruise_speed_scale, deceleration, precision_approach=False)

    def _safe_zone_d1_target(self) -> FieldPoint:
        """返回本趟选定的 d1 场地停车目标。"""

        return self._safe_zone_d1_target_for_offset(
            self._d1_calibration_offset_mm
        )

    def _safe_zone_d1_target_for_offset(self, offset_mm: float) -> FieldPoint:
        """返回指定安全区偏移经刹车补偿后的 d1 目标。"""

        endpoint = self._safe_zone_transport_endpoint()
        return FieldPoint(
                endpoint.x,
                endpoint.y
                - self._safe_zone_forward_y_sign
                * offset_mm,
        )

    def _safe_zone_d2_target(self) -> FieldPoint:
        """返回应用刹车过冲补偿后的 d2 场地停车目标。"""

        endpoint = self._safe_zone_transport_endpoint()
        return FieldPoint(
                endpoint.x,
                endpoint.y
                - self._safe_zone_forward_y_sign
                * self.config.safe_zone_open_offset_mm,
        )

    def _safe_zone_transport_endpoint(self) -> FieldPoint:
        """按本趟抓取类别返回物资区或伤员区推进终点。"""

        if self._transport_target_classes == (TargetClass.ORANGE_INJURED,):
            return self.config.safe_zone_injured_target_field
        return self.config.safe_zone_fallback_target_field

    def _safe_zone_line_coordinate_threshold_reached(
        self,
        target: FieldPoint,
        start: FieldPoint | None,
        tolerance_mm: float | None = None,
    ) -> bool:
        """判断直线目标是否命中容差，或沿计划方向已经越过目标。

        直线起点已经落在阈值内的坐标不参与判断，避免某一坐标在直线起点
        已经达标时提前结束。车辆偏航导致一次采样跨过二维容差窗口时，
        使用起点到目标向量的投影进度触发越过保护。``tolerance_mm`` 缺省
        时使用 ``transport_align_tolerance_mm``，供变体开场自定义到达容差。
        """

        position = self._fallback_field_position
        if position is None or start is None:
            return False
        tolerance = (
            self.config.transport_align_tolerance_mm
            if tolerance_mm is None
            else tolerance_mm
        )
        x_needed = abs(start.x - target.x) > tolerance
        y_needed = abs(start.y - target.y) > tolerance
        x_reached = not x_needed or abs(position.x - target.x) <= tolerance
        y_reached = not y_needed or abs(position.y - target.y) <= tolerance
        if x_reached and y_reached:
            return True

        delta_x = target.x - start.x
        delta_y = target.y - start.y
        target_distance_squared = delta_x * delta_x + delta_y * delta_y
        if target_distance_squared <= 1e-12:
            return False
        progress = (
            (position.x - start.x) * delta_x
            + (position.y - start.y) * delta_y
        )
        return progress >= target_distance_squared

    def _d1_line_coordinate_threshold_reached(self) -> bool:
        return self._safe_zone_line_coordinate_threshold_reached(
            self._safe_zone_d1_target(),
            self._d1_line_start_position,
        )

    def _d2_line_coordinate_threshold_reached(self) -> bool:
        return self._safe_zone_line_coordinate_threshold_reached(
            self._safe_zone_d2_target(),
            self._d2_line_start_position,
        )

    def _begin_safe_zone_keypoint_reverse(self) -> None:
        """在 bbox 居中但关键点不足时开始一次有界的小距离倒车。"""

        self._safe_zone_observation_deadline_ns = None
        self._safe_zone_confirmation_deadline_ns = None
        self._safe_zone_phase = "reversing_for_safe_zone_keypoints"
        self._safe_zone_keypoint_reverse_base_distance_m = (
            self._latest_cumulative_distance_m
        )
        self._safe_zone_keypoint_reverse_attempted = True
        self._safe_zone_stop_since_ns = None

    def _step_safe_zone_keypoint_reverse(self, timestamp_ns: int) -> MatchDecision:
        command = self._noncontact_motion(timestamp_ns, "keypoint_reverse",
            target=-self.config.safe_zone_keypoint_reverse_max_distance_m,
            speed=self.config.safe_zone_keypoint_reverse_speed_m_s)
        if not command.complete:
            return self._noncontact_decision(timestamp_ns, command)
        self._finish_noncontact()
        self._safe_zone_phase = "stopping_after_bbox_keypoints"
        self._safe_zone_keypoint_reverse_base_distance_m = None
        return self._decision(timestamp_ns, 0.0, 0.0, "safe_zone_keypoint_reverse_limit_reached")

    def _begin_safe_zone_keypoint_reobserve(self, timestamp_ns: int) -> None:
        """在关键点短缺后开启一次固定时限的静止新帧等待。"""

        self._safe_zone_phase = "reobserving_safe_zone_keypoints"
        if self._safe_zone_observation_deadline_ns is None:
            self._safe_zone_observation_deadline_ns = timestamp_ns + round(
                self.config.safe_zone_keypoint_reobserve_timeout_s * 1e9)
        self._safe_zone_key_reobserve_until_ns = timestamp_ns + round(
            self.config.safe_zone_keypoint_reobserve_timeout_s * 1_000_000_000
        )
        perception = self._latest_perception
        self._safe_zone_key_reobserve_frame_floor = (
            None if perception is None else perception.frame_sequence
        )
        self._safe_zone_key_samples = []
        self._safe_zone_key_last_frame = None
        self._safe_zone_calibration_snapshot = None
        self._safe_zone_calibration_zone = None

    def _safe_zone_keypoint_reobserve_scan_direction(self) -> float:
        """返回关键点缺失时沿用的扫描方向。"""

        direction = self._safe_zone_bbox_turn_direction
        if direction is not None:
            return direction
        heading = self._latest_heading_rad
        if heading is None:
            direction = -1.0
        else:
            heading_error = normalize_angle(
                self._safe_zone_forward_heading_rad() - heading
            )
            direction = 1.0 if heading_error > 1e-6 else -1.0
        self._safe_zone_bbox_turn_direction = direction
        return direction

    def _step_safe_zone_keypoint_reobserve(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """固定窗口重观测，失败退出纠偏，不重启追框。"""

        deadline = self._safe_zone_key_reobserve_until_ns
        if self._safe_zone_confirmation_deadline_ns is not None:
            deadline = self._safe_zone_confirmation_deadline_ns
        if (self._safe_zone_confirmation_deadline_ns is None
                and deadline is not None and self._safe_zone_observation_deadline_ns is not None):
            deadline = min(deadline, self._safe_zone_observation_deadline_ns)
        if deadline is None:
            self._safe_zone_phase = "searching_safe_zone_keypoints"
            return self._step_safe_zone_bbox_key_search(timestamp_ns)
        perception = self._latest_perception
        floor = self._safe_zone_key_reobserve_frame_floor
        is_new_frame = (
            perception is not None
            and (floor is None or perception.frame_sequence > floor)
        )
        if is_new_frame:
            zone = self._own_safe_zone_observation()
            if (zone is not None and not self._safe_zone_bbox_fully_visible(zone)
                    and self._safe_zone_capture_valid(perception.capture_timestamp_ns, timestamp_ns)):
                return self._step_safe_zone_bbox_key_search(timestamp_ns)
            if (
                self._safe_zone_required_points_visible(zone)
                and self._safe_zone_has_required_ground_keypoints(zone)
            ):
                if self._safe_zone_vehicle_stopped(timestamp_ns):
                    self._safe_zone_phase = "collecting_safe_zone_keys_closed"
                    return self._step_transport_release(timestamp_ns)
                self._safe_zone_phase = "stopping_after_bbox_keypoints"
                return self._decision(timestamp_ns, 0.0, 0.0,
                    "safe_zone_keypoints_reobserved_stop_before_calibration",
                    posture=GripperPosture.CLOSED)
        if timestamp_ns < deadline:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                self._safe_zone_calibration_wait_reason(timestamp_ns, "required_pair_missing"),
                posture=(
                    GripperPosture.CLOSED
                ),
            )

        if (self._stationary_motion.latest is not None
                and not self._safe_zone_vehicle_stopped(timestamp_ns)):
            return self._decision(timestamp_ns, 0.0, 0.0,
                self._safe_zone_calibration_wait_reason(timestamp_ns, "control_state_unavailable"),
                posture=GripperPosture.CLOSED)
        zone = self._own_safe_zone_observation()
        self._safe_zone_key_reobserve_until_ns = None
        self._safe_zone_key_reobserve_frame_floor = None
        self._safe_zone_key_samples = []
        self._safe_zone_key_last_frame = None
        self._safe_zone_calibration_snapshot = None
        self._safe_zone_calibration_zone = None
        # 一轮新证据没有改善：结束本次纠偏，用当前航位重新规划 d2 路线。
        # 不发布伪造的视觉校正；沿用当前编码器/IMU航位，失败不重启原地等待。
        self._safe_zone_calibration_last_failure = (
            self._safe_zone_calibration_last_failure or "required_pair_unavailable")
        self._safe_zone_phase = "align_d2_line"
        self.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
        return self._decision(timestamp_ns, 0.0, 0.0,
            "safe_zone_calibration_unavailable_continue_odometry:" + self._safe_zone_calibration_last_failure,
            posture=GripperPosture.CLOSED)

    def _step_safe_zone_keypoint_scan(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """以固定角速度扫描，完整 bbox 入镜后立即制动并重新静止观察。"""

        posture = GripperPosture.CLOSED
        zone = self._own_safe_zone_observation()
        if zone is not None and self._safe_zone_bbox_fully_visible(zone):
            self._safe_zone_phase = "stopping_after_bbox_keypoints"
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_bbox_fully_visible_stop_before_calibration",
                posture=posture,
            )
        direction = self._safe_zone_keypoint_reobserve_scan_direction()
        return self._decision(
            timestamp_ns,
            0.0,
            direction
            * self.config.safe_zone_bbox_turn_max_angular_velocity_rad_s,
            "safe_zone_keypoint_reobserve_scan",
            posture=posture,
        )

    def _collect_safe_zone_key_sample(self, timestamp_ns: int) -> bool:
        """从最新独立帧收集所选两点，不重复使用同一帧。"""

        perception = self._latest_perception
        zone = self._own_safe_zone_observation()
        previous = self._safe_zone_calibration_snapshot
        if previous is not None and not self._safe_zone_capture_valid(previous.capture_timestamp_ns, timestamp_ns, retained_sample=True):
            self._safe_zone_key_samples = []
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
        if perception is None or zone is None:
            self._safe_zone_calibration_last_failure = "safe_zone_not_detected"
            return False
        if not self._safe_zone_bbox_fully_visible(zone):
            self._safe_zone_calibration_last_failure = "bbox_or_required_points_clipped"
            self._safe_zone_key_samples = []
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
            return False
        if not self._safe_zone_required_points_visible(zone) or not self._safe_zone_has_required_ground_keypoints(zone):
            self._safe_zone_calibration_last_failure = "required_pair_missing_or_clipped"
            return False
        if not self._safe_zone_capture_valid(perception.capture_timestamp_ns, timestamp_ns):
            self._safe_zone_calibration_last_failure = "capture_outside_stationary_interval_or_stale"
            return False
        if (self._safe_zone_key_last_frame is not None
                and perception.frame_sequence <= self._safe_zone_key_last_frame):
            return False
        if not self._safe_zone_has_required_ground_keypoints(zone):
            return False
        if (self._safe_zone_calibration_snapshot is not None and
                perception.capture_timestamp_ns <= self._safe_zone_calibration_snapshot.capture_timestamp_ns):
            return False
        required = self._safe_zone_required_indices()
        points = (zone.ground_anchor, zone.image_left_landmark, zone.image_right_landmark)
        self._safe_zone_key_samples.append(tuple(
            point.ground if index in required else None for index, point in enumerate(points)
        ))
        self._safe_zone_calibration_last_failure = None
        self._safe_zone_key_last_frame = perception.frame_sequence
        self._safe_zone_calibration_snapshot = perception
        self._safe_zone_calibration_zone = zone
        return True

    def _lock_safe_zone_calibration_plan(self, timestamp_ns: int) -> bool:
        """用 2 个独立帧的所选两点均值拟合位姿；不补造缺失角点。"""

        if not self._safe_zone_required_points_visible(self._own_safe_zone_observation()):
            self._safe_zone_calibration_last_failure = "bbox_or_required_points_clipped"
            return False
        if len(self._safe_zone_key_samples) < 2:
            return False
        required = self._safe_zone_required_indices()
        means: list[GroundPoint | None] = [None, None, None]
        for index in required:
            points = [sample[index] for sample in self._safe_zone_key_samples]
            if any(point is None for point in points):
                self._safe_zone_calibration_last_failure = "required_pair_missing"
                return False
            means[index] = GroundPoint(
                sum(point.x for point in points) / len(points),
                sum(point.y for point in points) / len(points),
            )
        k0, k1, k2 = means
        other = means[required[1]]
        assert k0 is not None and other is not None
        if math.hypot(other.x - k0.x, other.y - k0.y) <= 1e-6:
            self._safe_zone_calibration_last_failure = "zero_key_baseline"
            return False

        localizer = self._safe_zone_corner_localizer
        snapshot = self._safe_zone_calibration_snapshot
        zone = self._safe_zone_calibration_zone
        position = self._fallback_field_position
        heading = self._latest_heading_rad
        raw_heading = self._raw_heading_rad
        if localizer is None:
            self._safe_zone_calibration_last_failure = "localizer_missing"
            return False
        if snapshot is None or snapshot.field_features is None or zone is None:
            self._safe_zone_calibration_last_failure = "feature_snapshot_missing"
            return False
        if position is None or heading is None or raw_heading is None:
            self._safe_zone_calibration_last_failure = "prior_pose_missing"
            return False
        if timestamp_ns < snapshot.field_features.result_timestamp_ns:
            self._safe_zone_calibration_last_failure = "feature_timestamp_in_future"
            return False

        # 两帧各自必须稳定，避免误差相反的点被均值掩盖。
        for index in required:
            mean = means[index]
            assert mean is not None
            for sample in self._safe_zone_key_samples:
                point = sample[index]
                assert point is not None
                if math.hypot(point.x - mean.x, point.y - mean.y) > localizer.config.max_fit_residual_mm:
                    self._safe_zone_calibration_last_failure = "unstable_required_pair"
                    return False

        averaged_zone = replace(
            zone,
            ground_anchor=replace(zone.ground_anchor, ground=k0),
            image_left_landmark=replace(zone.image_left_landmark, ground=k1,
                undistorted=zone.image_left_landmark.undistorted if k1 is not None else None,
                confidence=zone.image_left_landmark.confidence if k1 is not None else 0.0),
            image_right_landmark=replace(zone.image_right_landmark, ground=k2,
                undistorted=zone.image_right_landmark.undistorted if k2 is not None else None,
                confidence=zone.image_right_landmark.confidence if k2 is not None else 0.0),
        )
        averaged_features: FieldFeatureDetectionResult = replace(
            snapshot.field_features,
            safe_zones=(averaged_zone,),
        )
        # FieldPoseKeypoint.ground 已由 detector 通过 GroundProjector 生成；这里
        # 只把多帧均值交回同一套安全区静态地标/外参几何拟合，不复制投影矩阵。
        localization = localizer.localize(
            averaged_features,
            prior_pose=FieldPose2D(position, heading),
            current_timestamp_ns=timestamp_ns,
        )
        calibration = localization.observation
        if calibration is None:
            # 直接落定位器的拒绝原因：观测年龄门和两点几何门必须可区分。
            assert localization.rejection is not None
            self._safe_zone_calibration_last_failure = localization.rejection.value
            return False

        self._safe_zone_keys = (k0, k1, k2)
        self._safe_zone_calibration_pose = calibration
        self._safe_zone_calibration_heading_rad = (
            calibration.pose.heading_rad
        )
        self._fallback_field_position = calibration.pose.position
        self._heading_offset_rad = normalize_angle(
            calibration.pose.heading_rad - raw_heading
        )
        self._latest_heading_rad = calibration.pose.heading_rad
        self._fallback_last_heading_rad = calibration.pose.heading_rad
        self._safe_zone_calibration_last_failure = None
        return True

    def _step_align_red_zone(
        self,
        timestamp_ns: int,
        perception,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """执行 d1/d2 直线段和 d2 后的 +90° 对准。"""

        for phase, posture, reason in (
            (
                "safe_before_d1_line",
                GripperPosture.CLOSED,
                "safe_zone_waiting_before_d1_line_alignment",
            ),
            (
                "safe_before_d2_align_y",
                GripperPosture.OPEN,
                "safe_zone_waiting_before_d2_y_alignment",
            ),
        ):
            settling = self._consume_action_settle(
                timestamp_ns,
                phase,
                posture=posture,
                reason=reason,
            )
            if settling is not None:
                return settling
        del perception
        if self._safe_zone_phase in {"align_d1_line", "align_d2_line"}:
            return self._step_point_transport(timestamp_ns, self._safe_zone_phase == "align_d1_line")
        position = self._fallback_field_position
        if position is None or heading_rad is None or cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_route_waiting_for_field_odometry",
                posture=GripperPosture.CLOSED,
            )
        if self._safe_zone_phase == "align_y_at_d2":
            command = self._noncontact_motion(timestamp_ns, "d2_heading",
                target=normalize_angle(self._safe_zone_forward_heading_rad()-heading_rad),
                speed=self.config.safe_zone_fallback_max_angular_velocity_rad_s, turn=True,
                tolerance=self.config.safe_zone_fallback_heading_tolerance_rad)
            if not command.complete:
                return self._noncontact_decision(timestamp_ns, command, GripperPosture.OPEN)
            self._finish_noncontact()
            self._safe_zone_phase = "stopping_after_d2_heading"
            self._safe_zone_stop_since_ns = None
            return self._decision(timestamp_ns, 0.0, 0.0,
                "safe_zone_d2_heading_reached_stopped", posture=GripperPosture.OPEN)

        if self._safe_zone_phase == "stopping_after_d2_heading":
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_waiting_for_vehicle_stop_after_d2_heading",
                    posture=GripperPosture.OPEN,
                )
            self._transport_forward_base_distance_m = cumulative_distance_m
            self._transport_forward_distance_m = max(
                0.0,
                self._safe_zone_forward_y_sign
                * (self._safe_zone_final_target_y_mm() - position.y)
                / 1000.0,
            )
            self._gripper_phase_started_ns = timestamp_ns
            self._safe_zone_phase = "closing_before_final_forward"
            self.state = MatchState.TRANSPORT_RELEASE
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_d2_heading_"
                f"{self._safe_zone_heading_label()}_stopped_start_closing_gripper",
                posture=GripperPosture.CLOSED,
            )

        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "safe_zone_route_unknown_phase_hold",
            posture=(
                GripperPosture.OPEN
                if self._transport_opened
                else GripperPosture.CLOSED
            ),
        )

    def safe_zone_diagnostic(self, perception) -> str:
        """记录安全区路线和 d1 视觉校正结果。"""

        del perception
        position = self._fallback_field_position
        calibration_heading = self._safe_zone_calibration_heading_rad
        left_speed, right_speed = self._latest_speed_feedback
        stop_feedback_text = "(" + ",".join(
            "none" if speed is None else f"{speed:.3f}"
            for speed in (left_speed, right_speed)
        ) + ")"
        calibration_heading_text = (
            "none"
            if calibration_heading is None
            else f"{math.degrees(calibration_heading):.1f}"
        )
        calibration_pose = self._safe_zone_calibration_pose
        calibration_pose_text = (
            "none"
            if calibration_pose is None
            else (
                f"({calibration_pose.pose.position.x:.1f},"
                f"{calibration_pose.pose.position.y:.1f},"
                f"{math.degrees(calibration_pose.pose.heading_rad):.1f}),"
                f"residual_mm={calibration_pose.fit_residual_mm:.1f}"
            )
        )
        current_heading_text = (
            "none"
            if self._latest_heading_rad is None
            else f"{math.degrees(self._latest_heading_rad):.1f}"
        )
        d1_target = self._safe_zone_d1_target()
        d1_line_heading_text = (
            "none"
            if self._d1_line_heading_rad is None
            else f"{math.degrees(self._d1_line_heading_rad):.1f}"
        )
        d1_line_distance_text = (
            "none"
            if self._d1_line_distance_m is None
            else f"{self._d1_line_distance_m:.3f}"
        )
        d2_target = self._safe_zone_d2_target()
        final_speed = self._safe_zone_d2_to_final_speed_m_s()
        final_braking_overrun = self._safe_zone_d2_to_final_braking_overrun_mm()
        d2_line_heading_text = (
            "none"
            if self._d2_line_heading_rad is None
            else f"{math.degrees(self._d2_line_heading_rad):.1f}"
        )
        d2_line_distance_text = (
            "none"
            if self._d2_line_distance_m is None
            else f"{self._d2_line_distance_m:.3f}"
        )
        calibration_y_offset_text = (
            "-d1" if self._safe_zone_forward_y_sign > 0.0 else "+d1"
        )
        bbox_turn_direction_text = (
            "none"
            if self._safe_zone_bbox_turn_direction is None
            else (
                "left"
                if self._safe_zone_bbox_turn_direction > 0.0
                else "right"
            )
        )
        keypoint_count = self._safe_zone_keypoint_count(
            self._own_safe_zone_observation()
        )
        bbox_center_text = self._safe_zone_bbox_center_text()
        transport_endpoint = self._safe_zone_transport_endpoint()
        if position is None:
            return (
                f"route_phase={self._safe_zone_phase},"
                f"key_samples={len(self._safe_zone_key_samples)},"
                "field_position=unknown,"
                f"visual_calibration_pose={calibration_pose_text},"
                f"calibration_failure={self._safe_zone_calibration_last_failure},"
                f"calibration_heading_deg={calibration_heading_text},"
                f"stop_feedback_m_s={stop_feedback_text},"
                f"current_heading_deg={current_heading_text},"
                f"safe_zone_keypoints={keypoint_count}/3,required={self._safe_zone_required_indices()},"
                f"safe_zone_bbox_center={bbox_center_text},"
                f"safe_zone_bbox_turn_direction={bbox_turn_direction_text},"
                f"transport_endpoint=({transport_endpoint.x:.1f},{transport_endpoint.y:.1f}),"
                f"d1_target=({d1_target.x:.1f},{d1_target.y:.1f}),"
                f"d2_target=({d2_target.x:.1f},{d2_target.y:.1f}),"
                f"final_nominal_y={transport_endpoint.y:.0f},"
                f"final_target_y={self._safe_zone_final_target_y_mm():.0f},"
                f"final_speed_m_s={final_speed:.3f},"
                f"final_braking_overrun_mm={final_braking_overrun:.1f},"
                f"d1_line_heading_deg={d1_line_heading_text},"
                f"d1_line_distance_m={d1_line_distance_text},"
                f"d2_line_heading_deg={d2_line_heading_text},"
                f"d2_line_distance_m={d2_line_distance_text}"
            )
        return (
            f"route_phase={self._safe_zone_phase},"
            f"field_position=({position.x:.1f},{position.y:.1f}),"
            f"transport_endpoint=({transport_endpoint.x:.1f},{transport_endpoint.y:.1f}),"
            f"d1_target=({d1_target.x:.1f},{d1_target.y:.1f}),"
            f"d2_target=({d2_target.x:.1f},{d2_target.y:.1f}),"
            f"visual_calibration_pose={calibration_pose_text},"
            f"calibration_failure={self._safe_zone_calibration_last_failure},"
            f"calibration_heading_deg={calibration_heading_text},"
            f"stop_feedback_m_s={stop_feedback_text},"
            f"current_heading_deg={current_heading_text},"
            f"safe_zone_keypoints={keypoint_count}/3,required={self._safe_zone_required_indices()},"
            f"safe_zone_bbox_center={bbox_center_text},"
            f"safe_zone_bbox_turn_direction={bbox_turn_direction_text},"
            f"calibration_y={transport_endpoint.y:.1f}{calibration_y_offset_text},"
            f"d1_line_heading_deg={d1_line_heading_text},"
            f"d1_line_distance_m={d1_line_distance_text},"
            f"d2_line_heading_deg={d2_line_heading_text},"
            f"d2_line_distance_m={d2_line_distance_text},"
            f"final_nominal_y={transport_endpoint.y:.0f},"
            f"final_target_y={self._safe_zone_final_target_y_mm():.0f},"
            f"final_speed_m_s={final_speed:.3f},"
            f"final_braking_overrun_mm={final_braking_overrun:.1f},"
            f"d1_offset_mm={self._d1_calibration_offset_mm:.1f},"
            f"d1_offset_bounds_mm=("
            f"{self.config.safe_zone_calibration_min_offset_mm:.1f},"
            f"{self.config.safe_zone_calibration_start_offset_mm:.1f}),"
            f"d2={self.config.safe_zone_open_offset_mm:.1f},"
            f"key_samples={len(self._safe_zone_key_samples)}"
        )

    def _step_point_transport(self, timestamp_ns: int, d1: bool) -> MatchDecision:
        target = self._safe_zone_d1_target() if d1 else self._safe_zone_d2_target()
        command = self._noncontact_motion(timestamp_ns, "d1" if d1 else "d2",
            target=1.0, point=target,
            speed=self.config.safe_zone_grab_to_d1_speed_m_s if d1 else self.config.safe_zone_d1_to_d2_speed_m_s,
            tolerance=(self.config.transport_align_tolerance_mm if d1
                       else self.config.transport_d2_tolerance_mm)/1000)
        if not command.complete:
            return self._noncontact_decision(timestamp_ns, command)
        self._finish_noncontact()
        self._transport_forward_base_distance_m = None
        self._transport_forward_distance_m = None
        self._safe_zone_stop_since_ns = None
        self._action_settle_phase = None
        self.state = MatchState.TRANSPORT_RELEASE
        if d1:
            self._safe_zone_phase = "stopping_before_calibration"
            self._safe_zone_key_samples = []
            self._safe_zone_key_reobserve_until_ns = None
            self._safe_zone_key_reobserve_frame_floor = None
        else:
            self._safe_zone_phase = "stopping_before_d2_opening"
        return self._decision(timestamp_ns, 0.0, 0.0,
            "safe_zone_d1_point_reached_stopped" if d1 else "safe_zone_d2_point_reached_stopped")

    def _step_transport_forward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """执行点到点运输或保留原有的末端接触推进。"""
        if self._safe_zone_phase in {"forward_d1_line", "forward_d2_line"}:
            return self._step_point_transport(timestamp_ns, self._safe_zone_phase == "forward_d1_line")


        settle_phase = self._action_settle_phase
        for phase, posture, reason in (
            (
                "safe_before_d1_forward",
                GripperPosture.CLOSED,
                "safe_zone_waiting_after_d1_line_alignment",
            ),
            (
                "safe_before_d2_forward",
                GripperPosture.CLOSED,
                "safe_zone_waiting_after_d2_line_alignment",
            ),
            (
                "safe_before_final_forward",
                GripperPosture.CLOSED,
                "safe_zone_waiting_after_d2_heading_gripper_close",
            ),
        ):
            settling = self._consume_action_settle(
                timestamp_ns,
                phase,
                posture=posture,
                reason=reason,
            )
            if settling is not None:
                return settling
        if settle_phase in {
            "safe_before_d1_forward",
            "safe_before_d2_forward",
            "safe_before_final_forward",
        }:
            self._refresh_forward_segment_after_settle(
                settle_phase,
                cumulative_distance_m,
            )
        if (
            cumulative_distance_m is None
            or self._transport_forward_base_distance_m is None
            or self._transport_forward_distance_m is None
        ):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "transport_forward_waiting_for_odometry",
                posture=(
                    GripperPosture.CLOSED
                    if self._safe_zone_phase in {
                        "forward_d1_line",
                        "forward_d2_line",
                        "forward_final_closed",
                    }
                    else (
                        GripperPosture.OPEN
                        if self._transport_opened
                        else GripperPosture.CLOSED
                    )
                ),
            )
        travelled = (
            cumulative_distance_m - self._transport_forward_base_distance_m
        )

        if travelled >= self._transport_forward_distance_m - 1e-9:
            self._transport_count += 1
            self._gripper_phase_started_ns = None
            self._return_phase = "stopping_before_exit"
            self._safe_zone_exit_base_distance_m = None
            self._safe_zone_stop_since_ns = None
            self._safe_zone_phase = "stopping_before_exit_opening"
            self.state = MatchState.TRANSPORT_RELEASE
            if self._begin_action_settle(
                timestamp_ns,
                "safe_before_exit_opening",
            ):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_reached_transport_endpoint_wait_before_opening",
                    posture=GripperPosture.CLOSED,
                )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_reached_transport_endpoint_wait_before_opening",
                posture=GripperPosture.CLOSED,
            )
        angular = self._heading_hold_angular_velocity(
            self._safe_zone_forward_heading_rad(),
            kp_rad_s=self.config.safe_zone_fallback_heading_kp_rad_s,
            max_angular_velocity_rad_s=(
                self.config.safe_zone_fallback_max_angular_velocity_rad_s
            ),
            tolerance_rad=self.config.safe_zone_fallback_heading_tolerance_rad,
        )
        if angular is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_forward_waiting_for_heading",
                posture=GripperPosture.CLOSED,
            )
        return self._decision(
            timestamp_ns,
            self._safe_zone_d2_to_final_speed_m_s(),
            angular,
            "safe_zone_closed_gripper_forward_to_transport_endpoint",
            posture=GripperPosture.CLOSED,
        )

    def _step_return_backup(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """倒车退出停稳后直接转向搜索，中心纠偏由正常感知链消费。"""

        phase = self._return_phase
        if phase == "idle":
            self._return_phase = "stopping_before_exit"
            self._safe_zone_exit_base_distance_m = None
            self._safe_zone_stop_since_ns = None
            phase = self._return_phase

        if phase == "stopping_before_exit":
            if cumulative_distance_m is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_exit_waiting_for_odometry",
                    posture=GripperPosture.OPEN,
                )
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_waiting_for_vehicle_stop_before_exit",
                    posture=GripperPosture.OPEN,
                )
            self._safe_zone_exit_base_distance_m = cumulative_distance_m
            self._safe_zone_stop_since_ns = None
            self._return_phase = "exit_reverse"
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_vehicle_stopped_start_exit",
                posture=GripperPosture.OPEN,
            )

        if phase == "exit_reverse":
            command = self._noncontact_motion(timestamp_ns, "exit_reverse",
                target=-self.config.safe_zone_exit_distance_m, speed=self.config.return_backup_speed_m_s)
            if not command.complete:
                return self._noncontact_decision(timestamp_ns, command, GripperPosture.OPEN)
            self._finish_noncontact()
            self._return_phase = "stopping_after_exit"
            self._safe_zone_stop_since_ns = None
            return self._decision(timestamp_ns, 0.0, 0.0,
                "safe_zone_exit_distance_reached_wait_for_stop", posture=GripperPosture.OPEN)

        if phase == "stopping_after_exit":
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_waiting_for_vehicle_stop_after_exit",
                    posture=GripperPosture.OPEN,
                )
            if self._transport_count >= self.config.required_transports:
                self._return_phase = "idle"
                self._search_frame_floor = None
                self.state = MatchState.FINISH_STOP
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "all_transports_complete",
                    posture=GripperPosture.OPEN,
                )
            return self._finish_safe_zone_exit(timestamp_ns)

        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "safe_zone_exit_unknown_phase_hold",
            posture=GripperPosture.OPEN,
        )

    def _finish_safe_zone_exit(self, timestamp_ns: int) -> MatchDecision:
        """交付后清空本趟计数和旧候选，立即转向搜索。"""
        self._gate_clearance = None
        self._gate_clearance_attempted = False
        self._safe_zone_phase = "idle"
        self._safe_zone_stop_since_ns = None
        self._reset_tracker_for_new_preview_epoch()
        self._selected_track_id = None
        self._transport_target_classes = ()
        self._cargo_capture_floor_ns = None
        self._greedy_active = False
        self._return_backup_base_distance_m = None
        self._transport_forward_distance_m = None
        self._begin_cluster_search()
        self._reset_rotation_budget()
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(timestamp_ns, 0.0, self.config.cluster_search_angular_velocity_rad_s,
                              "safe_zone_exit_complete_start_search", posture=GripperPosture.OPEN)

    def _step_transport_release(self, timestamp_ns: int) -> MatchDecision:
        """处理 d2 张爪、转向、闭爪推进，以及推进停稳后的张爪释放。"""

        calibration_posture = (
            GripperPosture.CLOSED
        )

        if self._safe_zone_phase == "stopping_before_d2_opening":
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_d2_waiting_for_vehicle_stop_before_opening",
                    posture=GripperPosture.CLOSED,
                )
            settling = self._consume_action_settle(
                timestamp_ns,
                "safe_before_d2_opening",
                posture=GripperPosture.CLOSED,
                reason="safe_zone_d2_waiting_before_opening",
            )
            if settling is not None:
                return settling
            self._transport_opened = True
            self._gripper_phase_started_ns = timestamp_ns
            self._safe_zone_phase = "opening_at_d2"
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_d2_reached_start_opening",
                posture=GripperPosture.OPEN,
            )

        if self._safe_zone_phase == "opening_at_d2":
            if not self._gripper_action_completed(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "opening_gripper_at_endpoint_y_"
                    f"{'minus' if self._safe_zone_forward_y_sign > 0.0 else 'plus'}_d2",
                    posture=GripperPosture.OPEN,
                )
            self._gripper_phase_started_ns = None
            self._safe_zone_phase = "align_y_at_d2"
            self.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
            self._begin_action_settle(
                timestamp_ns,
                "safe_before_d2_align_y",
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "gripper_opened_at_d2_start_turn_to_"
                f"{self._safe_zone_heading_label()}",
                posture=GripperPosture.OPEN,
            )

        if self._safe_zone_phase == "closing_before_final_forward":
            if not self._gripper_action_completed(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "closing_gripper_before_safe_zone_forward",
                    posture=GripperPosture.CLOSED,
                )
            self._gripper_phase_started_ns = None
            self._safe_zone_phase = "forward_final_closed"
            self.state = MatchState.TRANSPORT_FORWARD
            self._begin_action_settle(
                timestamp_ns,
                "safe_before_final_forward",
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "gripper_closed_after_d2_heading_start_forward_settle",
                posture=GripperPosture.CLOSED,
            )

        if self._safe_zone_phase == "stopping_before_exit_opening":
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_waiting_for_vehicle_stop_before_opening",
                    posture=GripperPosture.CLOSED,
                )
            settling = self._consume_action_settle(
                timestamp_ns,
                "safe_before_exit_opening",
                posture=GripperPosture.CLOSED,
                reason="safe_zone_waiting_before_opening_after_forward",
            )
            if settling is not None:
                return settling
            self._transport_opened = True
            self._gripper_phase_started_ns = timestamp_ns
            self._safe_zone_phase = "opening_after_transport"
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_transport_stopped_start_opening",
                posture=GripperPosture.OPEN,
            )

        if self._safe_zone_phase == "stopping_before_calibration":
            self._safe_zone_observation_deadline_ns = None
            self._safe_zone_confirmation_deadline_ns = None
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_waiting_for_vehicle_stop_before_calibration",
                    posture=GripperPosture.CLOSED,
                )
            self._safe_zone_phase = "searching_safe_zone_keypoints"
            self._safe_zone_key_samples = []
            self._safe_zone_key_last_frame = None
            self._safe_zone_key_reobserve_until_ns = None
            self._safe_zone_key_reobserve_frame_floor = None
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
            self._safe_zone_calibration_last_failure = None
            self._safe_zone_bbox_turn_direction = None
            return self._step_safe_zone_bbox_key_search(timestamp_ns)

        if self._safe_zone_phase == "reobserving_safe_zone_keypoints":
            return self._step_safe_zone_keypoint_reobserve(timestamp_ns)

        if self._safe_zone_phase == "scanning_safe_zone_keypoints":
            return self._step_safe_zone_keypoint_scan(timestamp_ns)

        if self._safe_zone_phase == "reversing_for_safe_zone_keypoints":
            return self._step_safe_zone_keypoint_reverse(timestamp_ns)

        if self._safe_zone_phase == "searching_safe_zone_keypoints":
            return self._step_safe_zone_bbox_key_search(timestamp_ns)

        if self._safe_zone_phase == "stopping_after_bbox_keypoints":
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_waiting_for_vehicle_stop_after_keypoints",
                    posture=calibration_posture,
                )
            self._safe_zone_phase = "collecting_safe_zone_keys_closed"
            perception = self._latest_perception
            latency_ns = (max(0, perception.result_timestamp_ns - perception.capture_timestamp_ns)
                          if perception is not None else 0)
            if self._safe_zone_observation_deadline_ns is None:
                self._safe_zone_observation_deadline_ns = timestamp_ns + latency_ns + round(
                    self.config.safe_zone_keypoint_reobserve_timeout_s * 1e9)
            self._safe_zone_key_samples = []
            self._safe_zone_key_last_frame = None
            self._safe_zone_key_reobserve_until_ns = None
            self._safe_zone_key_reobserve_frame_floor = None
            self._safe_zone_keypoint_reverse_base_distance_m = None
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
            self._safe_zone_calibration_last_failure = None
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_stopped_start_calibration",
                posture=calibration_posture,
            )

        if self._safe_zone_phase == "collecting_safe_zone_keys_closed":
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                self._safe_zone_key_samples = []
                return self._decision(timestamp_ns, 0.0, 0.0,
                    "safe_zone_calibration_waiting_for_stationary:" + self._stationary_motion.diagnostic(timestamp_ns),
                    posture=calibration_posture)
            zone = self._own_safe_zone_observation()
            perception = self._latest_perception
            if (zone is not None and not self._safe_zone_bbox_fully_visible(zone)
                    and perception is not None
                    and self._safe_zone_capture_valid(perception.capture_timestamp_ns, timestamp_ns)):
                self._safe_zone_key_samples = []
                self._safe_zone_calibration_snapshot = None
                self._safe_zone_calibration_zone = None
                return self._step_safe_zone_bbox_key_search(timestamp_ns)
            collected = self._collect_safe_zone_key_sample(timestamp_ns)
            if collected and self._safe_zone_confirmation_deadline_ns is None:
                # 一次有效首帧触发独立确认窗口；重读、漏检、拟合重试均不能续期。
                self._safe_zone_confirmation_deadline_ns = timestamp_ns + round(
                    self.config.safe_zone_keypoint_reobserve_timeout_s * 1e9)
            if collected and len(self._safe_zone_key_samples) >= 2:
                if not self._lock_safe_zone_calibration_plan(timestamp_ns):
                    self._begin_safe_zone_keypoint_reobserve(timestamp_ns)
                    return self._decision(timestamp_ns, 0.0, 0.0,
                        "safe_zone_calibration_rejected:" + str(self._safe_zone_calibration_last_failure),
                        posture=calibration_posture)
                self._gripper_phase_started_ns = None
                self.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
                self._safe_zone_phase = "align_d2_line"
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_visual_calibrated_start_d2_line",
                    posture=calibration_posture,
                )
            deadline = (self._safe_zone_confirmation_deadline_ns
                        if self._safe_zone_confirmation_deadline_ns is not None
                        else self._safe_zone_observation_deadline_ns)
            if deadline is not None and timestamp_ns >= deadline:
                # 没有新有效帧才退出。直接走超时分支，不让同帧恢复入口递归重试。
                self._begin_safe_zone_keypoint_reobserve(timestamp_ns)
                return self._step_safe_zone_keypoint_reobserve(timestamp_ns)
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                self._safe_zone_calibration_wait_reason(timestamp_ns,
                    self._safe_zone_calibration_last_failure or "collecting"),
                posture=calibration_posture,
            )

        if self._safe_zone_phase == "opening_after_transport":
            if not self._gripper_action_completed(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "opening_gripper_after_safe_zone_push",
                    posture=GripperPosture.OPEN,
                )
            self._gripper_phase_started_ns = None
            self._safe_zone_phase = "idle"
            self.state = MatchState.RETURN_BACKUP
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "gripper_opened_after_safe_zone_push_start_exit",
                posture=GripperPosture.OPEN,
            )

        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "safe_zone_release_waiting",
            posture=(
                GripperPosture.OPEN
                if self._transport_opened
                else GripperPosture.CLOSED
            ),
        )

    def _safe_zone_calibration_wait_reason(self, timestamp_ns: int, cause: str) -> str:
        snapshot = self._latest_perception
        capture_age = None if snapshot is None else (timestamp_ns-snapshot.capture_timestamp_ns)/1e6
        ready_age = None if snapshot is None else (timestamp_ns-snapshot.result_timestamp_ns)/1e6
        return (f"safe_zone_wait:{cause},bbox_fully_visible={self._safe_zone_bbox_fully_visible(self._own_safe_zone_observation())},confirmed={len(self._safe_zone_key_samples)}/2,"
                f"capture_age_ms={capture_age},ready_age_ms={ready_age},"
                f"stationary_since_ns={self._safe_zone_stop_since_ns},"
                f"deadline_ns={self._safe_zone_confirmation_deadline_ns if self._safe_zone_confirmation_deadline_ns is not None else self._safe_zone_observation_deadline_ns},"
                f"first_frame_deadline_ns={self._safe_zone_observation_deadline_ns},"
                f"confirmation_deadline_ns={self._safe_zone_confirmation_deadline_ns},"
                f"{self._stationary_motion.diagnostic(timestamp_ns)}")

    def _safe_zone_capture_valid(self, capture_ns: int, timestamp_ns: int, *, retained_sample: bool = False) -> bool:
        localizer = self._safe_zone_corner_localizer
        max_age_ns = round((localizer.config.max_observation_age_ms
                            if localizer is not None else self.config.green_max_age_ms) * 1e6)
        if retained_sample:
            # 首帧入场时已过年龄门；仅在连续静止且有界确认期间保留，不能刷新采集时间。
            max_age_ns += round(self.config.safe_zone_keypoint_reobserve_timeout_s * 1e9)
        if self._stationary_motion.latest is not None:
            return self._stationary_motion.capture_valid(capture_ns, timestamp_ns, max_age_ns=max_age_ns)
        # Hardware-free step callers supply wheel feedback instead of device samples.
        since = self._safe_zone_stop_since_ns
        return (since is not None and since <= capture_ns <= timestamp_ns
                and timestamp_ns - capture_ns <= max_age_ns)

    def _safe_zone_vehicle_stopped(self, timestamp_ns: int) -> bool:
        """要求左右轮速反馈持续低于阈值，才允许进入下一步。"""

        if self._stationary_motion.latest is not None:
            since = self._stationary_motion.stationary_since(timestamp_ns)
            self._safe_zone_stop_since_ns = since
            confirm_s = (self._motion_profile.stationary_confirm_time_s
                         if self._noncontact_stop_confirmed else self.config.safe_zone_calibration_stop_confirm_time_s)
            return (since is not None and
                    self._stationary_motion.latest.received_timestamp_ns - since >= confirm_s * 1e9)
        left_speed, right_speed = self._latest_speed_feedback
        threshold = self.config.safe_zone_calibration_stop_speed_threshold_m_s
        if left_speed is None or right_speed is None:
            self._safe_zone_stop_since_ns = None
            return False
        if max(abs(left_speed), abs(right_speed)) > threshold:
            self._safe_zone_stop_since_ns = None
            return False
        if self._safe_zone_stop_since_ns is None:
            self._safe_zone_stop_since_ns = timestamp_ns
            return False
        confirm_time_ns = round(
            self.config.safe_zone_calibration_stop_confirm_time_s
            * 1_000_000_000
        )
        return timestamp_ns - self._safe_zone_stop_since_ns >= confirm_time_ns


def main() -> None:
    from rescue_vision.app.match_runtime import _run_hardware

    parser = argparse.ArgumentParser(
        description="Run the formal rescue match flow."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--start-area",
        choices=(MatchStartArea.AREA_2.value, MatchStartArea.AREA_3.value),
        default=MatchStartArea.AREA_2.value,
        help=(
            "选择正式流程启动区域：2 为地图右上角/红方，"
            "3 为地图左下角/蓝方（默认 2）。"
        ),
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help="Confirm a physical emergency stop and continuous supervision.",
    )
    parser.add_argument(
        "--local-preview",
        action="store_true",
        help="在本机窗口显示最新图像并叠加当前 state/reason；按 Q/Esc 退出。",
    )
    parser.add_argument("--jpeg-quality", type=int, default=80)
    parser.add_argument("--observer-image-interval-seconds", type=float, default=1.0)
    parser.add_argument("--log-dir", type=Path, default=Path("logs"))
    args = parser.parse_args()
    if not 1 <= args.jpeg_quality <= 100:
        parser.error("--jpeg-quality must be in [1, 100]")
    if (
        not math.isfinite(args.observer_image_interval_seconds)
        or args.observer_image_interval_seconds <= 0.0
    ):
        parser.error("--observer-image-interval-seconds must be positive")
    _run_hardware(
        args.config,
        supervised_stop_ready=args.supervised_physical_stop_ready,
        local_preview=args.local_preview,
        jpeg_quality=args.jpeg_quality,
        observer_image_interval_s=args.observer_image_interval_seconds,
        log_dir=args.log_dir,
        start_area=MatchStartArea.parse(args.start_area),
    )


if __name__ == "__main__":
    main()
