"""正式比赛流程：基于当前正式动作编排，纯逻辑不打开硬件。"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from rescue_vision.app.cluster_breakup import GripperPosture
from rescue_vision.app.gripper_width_sequence import (
    GraspPreparation,
    GripperWidthPickupDecision,
    GripperWidthPickupResult,
    GripperWidthPickupSequence,
    GripperWidthPickupState,
)
from rescue_vision.app.near_field_grasp import (
    NearFieldGraspPlan,
    NearFieldHandoffPrior,
    NearFieldGraspPolicy,
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
)
from rescue_vision.tracking import MultiTargetTracker, TrackStatus, TrackedTarget
from rescue_vision.mission import SafetySignals
from rescue_vision.motion.gripper_kinematics import GripperKinematics
from rescue_vision.motion.protocol import OdometryImu
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
    TRANSPORT_PRE_CLOSE_RECHECK = "transport_pre_close_recheck"
    TRANSPORT_CLOSE_GRIPPER = "transport_close_gripper"
    TRANSPORT_ALIGN_RED_ZONE = "transport_align_red_zone"
    TRANSPORT_FORWARD = "transport_forward"
    TRANSPORT_RELEASE = "transport_release"
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
    """单个控制周期的差速 twist 与夹爪姿态意图。"""

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
        safe_zone_d2_braking_overrun_x_mm=(
            -config.match.safe_zone_d2_braking_overrun_x_mm
        ),
        safe_zone_d2_braking_overrun_y_mm=(
            -config.match.safe_zone_d2_braking_overrun_y_mm
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

class MatchSequence:
    """可重放的正式动作流程；step() 只消费观测、里程和航向并输出意图。"""

    # 独立的抓取—运输联调入口覆盖为只搜索，不执行正式解团路由。
    _first_green_blocked_routes_to_breakup = True

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
        self.config = config
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
        self._transport_corridor_half_width_mm = (
            float(transport_corridor_half_width_mm)
            if transport_corridor_half_width_mm is not None
            else float(config.green_path_half_width_mm)
        )
        self._near_field_session_id = 0
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
        self._safe_zone_key_samples: list[tuple[GroundPoint, GroundPoint, GroundPoint]] = []
        self._safe_zone_key_last_frame: int | None = None
        self._safe_zone_calibration_snapshot: PerceptionSnapshot | None = None
        self._safe_zone_calibration_zone: SafeZoneObservation | None = None
        self._safe_zone_keys: tuple[GroundPoint, GroundPoint, GroundPoint] | None = None
        self._safe_zone_calibration_pose: SafeZoneCornerPoseObservation | None = None
        self._safe_zone_calibration_heading_rad: float | None = None
        self._safe_zone_calibration_last_failure: str | None = None
        self._safe_zone_stop_since_ns: int | None = None
        self._safe_zone_bbox_turn_direction: float | None = None
        self._safe_zone_reacquire_frame_floor: int | None = None
        self._safe_zone_calibration_after_exit = False
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
        self._d2_line_heading_rad: float | None = None
        self._d2_line_distance_m: float | None = None
        self._d2_line_start_position: FieldPoint | None = None
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
            gripper_full_travel_time_s=gripper.full_travel_time_s,
            forward_speed_m_s=runtime.green_approach_speed_m_s,
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
            math.pi / 2.0
            if config.world.team_color is TeamColor.BLUE
            else -math.pi / 2.0
        )
        if not (
            math.isclose(initial.position.x, expected_position.x, abs_tol=1e-6)
            and math.isclose(initial.position.y, expected_position.y, abs_tol=1e-6)
            and math.isclose(initial.heading_rad, expected_heading, abs_tol=1e-6)
        ):
            raise RuntimeError(
                "Match flow requires localization.fusion.initial_pose "
                "to match the selected start area: "
                "area 2=[1350 mm, 1350 mm, -90 deg] or "
                "area 3=[-1350 mm, -1350 mm, 90 deg]."
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
        )
        sequence._safe_zone_corner_localizer = config.build_safe_zone_corner_localizer()
        sequence._breakup_static_map = config.world.static_map
        sequence._breakup_clearance_mm = (
            config.match.robot_footprint_radius_mm
            + config.match.safety_margin_mm
        )
        return sequence

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
        selected = frozenset(self.preview_selected_track_ids)
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
        if self._near_field_pickup is not None:
            self._near_field_pickup.observe_motion(message)

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
        else:
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
        return (
            f"confirmation_started_ms={started_text} "
            f"confirmation_elapsed_ms={elapsed_text} "
            f"plan_age_ms={plan_age_text} "
            f"preparation_age_ms={preparation_age_text} "
            f"confirmation={confirmation_text} "
            f"handoff_prior={prior_text} "
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
        return is_open

    @property
    def near_field_policy(self) -> NearFieldGraspPolicy:
        if self._transport_count == 0:
            return NearFieldGraspPolicy(frozenset((TargetClass.GREEN_SUPPLY,)), 1)
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
        self._started = True
        self._startup_turn_last_heading = None
        self._startup_turn_progress_rad = 0.0
        self._reset_straight_pid()
        self._fallback_field_position = self._initial_field_position
        self._fallback_last_distance_m = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id = 0
            self._near_field_handoff_prior = None
            self._near_field_last_failure_diagnostic = None
            self._near_field_confirmation_started_ns = None
            self._near_field_far_reapproach_used = False
        self._transport_target_classes = ()
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

    def _step_startup_turn(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
    ) -> MatchDecision:
        if heading_rad is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "startup_turn_waiting_for_heading",
            )
        heading = heading_rad
        previous = self._startup_turn_last_heading
        self._startup_turn_last_heading = heading
        if previous is None:
            return self._decision(
                timestamp_ns,
                0.0,
                self.config.startup_turn_angular_velocity_rad_s,
                "startup_turn_right",
            )
        delta = self._directional_delta(
            previous,
            heading,
            self.config.startup_turn_angular_velocity_rad_s,
        )
        self._startup_turn_progress_rad += delta
        if self._startup_turn_progress_rad >= self.config.startup_turn_angle_rad:
            self.state = MatchState.STARTUP_TURN_SETTLE
            self._settle_until_ns = timestamp_ns + self._seconds_to_ns(
                self.config.startup_turn_settle_time_s
            )
            return self._decision(timestamp_ns, 0.0, 0.0, "startup_turn_complete_wait")
        return self._decision(
            timestamp_ns,
            0.0,
            self.config.startup_turn_angular_velocity_rad_s,
            "startup_turn_right",
        )

    def _step_startup_forward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None,
        right_speed_feedback_m_s: float | None,
    ) -> MatchDecision:
        if cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "startup_forward_waiting_for_odometry",
            )
        if self._startup_forward_base_distance_m is None:
            self._startup_forward_base_distance_m = cumulative_distance_m
        travelled = cumulative_distance_m - self._startup_forward_base_distance_m
        if travelled >= self.config.startup_forward_distance_m - 1e-9:
            self.state = MatchState.STARTUP_FORWARD_SETTLE
            self._settle_until_ns = timestamp_ns + self._seconds_to_ns(
                self.config.startup_forward_settle_time_s
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "startup_forward_complete_wait",
            )
        return self._decision(
            timestamp_ns,
            self.config.startup_forward_speed_m_s,
            self._straight_pid_output(
                timestamp_ns,
                left_speed_feedback_m_s,
                right_speed_feedback_m_s,
            ),
            "startup_forward_1_2m",
        )

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

    def _step_relocate_forward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        if cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "relocate_forward_waiting_for_odometry",
            )
        if self._relocate_forward_base_distance_m is None:
            self._relocate_forward_base_distance_m = cumulative_distance_m
        travelled = cumulative_distance_m - self._relocate_forward_base_distance_m
        if travelled >= self.config.cluster_relocate_distance_m - 1e-9:
            self._consecutive_cluster_losses = 0
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(
                timestamp_ns,
                0.0,
                self._cluster_search_angular_velocity_rad_s,
                "relocate_complete_resume_search",
            )
        return self._decision(
            timestamp_ns,
            self.config.cluster_relocate_speed_m_s,
            0.0,
            "relocate_forward",
        )

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
                "breakup_forward_0_7m_complete_open_gripper",
                posture=GripperPosture.OPEN,
            )
        return self._decision(
            timestamp_ns,
            self.config.breakup_forward_speed_m_s,
            0.0,
            "breakup_forward_0_7m_no_realign",
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
                "breakup_backward_0_1m_complete_close_gripper",
                posture=GripperPosture.CLOSED,
            )
        return self._decision(
            timestamp_ns,
            -self.config.breakup_backward_speed_m_s,
            0.0,
            "breakup_backward_0_1m",
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


    def _cluster_ground_measurement(
        self,
        timestamp_ns: int,
    ) -> _GroundClusterMeasurement | None:
        """返回目标团中心和最前方正向 K0 地面距离。"""

        self._cluster_selected_track_ids = ()
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if target.ground_point is not None
            and target.ever_confirmed
            and target.target_class is not TargetClass.UNKNOWN
            and self._target_is_fresh(target, timestamp_ns)
        )
        points = [target.ground_point for target in candidates]
        largest = self._largest_ground_group(
            points,
            preferred_points=self._preferred_cluster_ground_points(timestamp_ns),
        )
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
            or any(
                quality
                in {
                    ObservationQuality.COLOR_EVIDENCE_INSUFFICIENT,
                    ObservationQuality.COLOR_EVIDENCE_AMBIGUOUS,
                    ObservationQuality.POSE_COLOR_CONFLICT,
                }
                for quality in target.quality
            )
        ):
            return False
        return True

    @staticmethod
    def _graspable_target_is_usable(target: TrackedTarget) -> bool:
        """确认的绿/黑物资或橙色伤员候选。"""

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
        return not any(
            quality
            in {
                ObservationQuality.COLOR_EVIDENCE_INSUFFICIENT,
                ObservationQuality.COLOR_EVIDENCE_AMBIGUOUS,
                ObservationQuality.POSE_COLOR_CONFLICT,
            }
            for quality in target.quality
        )

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
            and not any(
                quality
                in {
                    ObservationQuality.COLOR_EVIDENCE_INSUFFICIENT,
                    ObservationQuality.COLOR_EVIDENCE_AMBIGUOUS,
                    ObservationQuality.POSE_COLOR_CONFLICT,
                }
                for quality in target.quality
            )
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

    def _target_is_fresh(self, target: TrackedTarget, timestamp_ns: int) -> bool:
        return (
            timestamp_ns >= target.last_seen_timestamp_ns
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
        self._cluster_search_angular_velocity_rad_s = (
            self.config.cluster_search_angular_velocity_rad_s
        )
        self._cluster_search_last_heading = None
        self._cluster_search_progress_rad = 0.0
        self._cluster_align_hold_center = None
        self._cluster_align_lost_since_ns = None
        self._return_phase = "idle"
        self._safe_zone_exit_base_distance_m = None
        self._d1_line_heading_rad = None
        self._d1_line_distance_m = None
        self._d1_line_start_position = None
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
        self,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
    ) -> None:
        """Integrate encoder displacement into the fallback FieldPoint estimate."""

        if heading_rad is None or cumulative_distance_m is None:
            return
        previous_distance = self._fallback_last_distance_m
        if previous_distance is None:
            self._fallback_last_distance_m = cumulative_distance_m
            return
        delta_distance_m = cumulative_distance_m - previous_distance
        if abs(delta_distance_m) > 0.0 and self._fallback_field_position is not None:
            self._fallback_field_position = FieldPoint(
                self._fallback_field_position.x
                + delta_distance_m * 1000.0 * math.cos(heading_rad),
                self._fallback_field_position.y
                + delta_distance_m * 1000.0 * math.sin(heading_rad),
            )
        self._fallback_last_distance_m = cumulative_distance_m

    def _advance_cluster_search_sweep(
        self,
        heading_rad: float | None,
    ) -> None:
        if heading_rad is None:
            return
        heading = heading_rad
        previous = self._cluster_search_last_heading
        self._cluster_search_last_heading = heading
        if previous is None:
            return
        self._cluster_search_progress_rad += self._directional_delta(
            previous,
            heading,
            self._cluster_search_angular_velocity_rad_s,
        )
        if (
            self._cluster_search_progress_rad
            >= self.config.cluster_search_sweep_angle_rad
        ):
            self._cluster_search_angular_velocity_rad_s *= -1.0
            self._cluster_search_progress_rad = 0.0
            self._cluster_search_last_heading = heading

    @staticmethod
    def _directional_delta(previous: float, current: float, angular_velocity: float) -> float:
        delta = normalize_angle(current - previous)
        if angular_velocity < 0.0:
            delta = -delta
        return max(0.0, delta)

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
        self._latest_perception = perception
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
            segment_clear = self._near_field_segment_clear(heading, remaining)
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
        polygons = tuple(self._breakup_static_map.safe_zone_polygon_field(color)
                         for color in (TeamColor.RED, TeamColor.BLUE))
        if any(polygon is None for polygon in polygons):
            return None
        margin = self._breakup_clearance_mm
        for polygon in polygons:
            assert polygon is not None
            low, high = 0.0, 1.0
            for origin, delta, lower, upper in (
                (position.x, distance_m * 1000.0 * math.cos(heading),
                 min(p.x for p in polygon) - margin, max(p.x for p in polygon) + margin),
                (position.y, distance_m * 1000.0 * math.sin(heading),
                 min(p.y for p in polygon) - margin, max(p.y for p in polygon) + margin),
            ):
                if abs(delta) < 1e-9:
                    if origin < lower or origin > upper:
                        break
                    continue
                entry, leave = sorted(((lower - origin) / delta, (upper - origin) / delta))
                low, high = max(low, entry), min(high, leave)
                if low > high:
                    break
            else:
                return True
        return False

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
                else (
                    self._near_field_grasp_config.target_final_x_mm
                    if self._near_field_grasp_config is not None
                    else 160.0
                )
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
        """返回场界扣除夹爪偏移后的 x/y 边界。

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
        offset = self.config.breakup_gripper_offset_mm
        return min_x + offset, max_x - offset, min_y + offset, max_y - offset

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
    def safe_zone_motion_acceleration_limit_m_s2(self) -> float | None:
        """返回 D2→安全区末端当前应使用的临时轮速加速度上限。"""

        limit = self.config.safe_zone_d2_to_final_max_wheel_acceleration_m_s2
        if limit is None:
            return None
        if (
            self._safe_zone_phase
            not in self._D2_ACCELERATION_LIMIT_PHASES
        ):
            return None
        return float(limit)

    @property
    def breakup_motion_acceleration_limit_m_s2(self) -> float | None:
        """返回解团阶段当前应使用的临时轮速加速度上限。"""

        limit = self.config.breakup_max_wheel_acceleration_m_s2
        if (
            limit is None
            or self.state not in self._BREAKUP_ACCELERATION_LIMIT_STATES
        ):
            return None
        return float(limit)

    @property
    def motion_acceleration_limit_m_s2(self) -> float | None:
        """返回当前 match 动作应使用的临时单轮最大加速度。"""

        breakup_limit = self.breakup_motion_acceleration_limit_m_s2
        if breakup_limit is not None:
            return breakup_limit
        return self.safe_zone_motion_acceleration_limit_m_s2

    @property
    def safe_zone_calibration_pose(self) -> FieldPose2D | None:
        """最近一次 d1 安全区三点视觉校正后的场地位姿。"""

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

        self._selected_track_id = target.track_id
        self._cluster_selected_track_ids = ()
        self._breakup_only = False
        self._selected_green_ground = target.ground_point
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

    def _update_cluster_search_velocity(self) -> None:
        """按当前帧信息量切换搜索转速，同时保留当前扫描方向。"""

        perception = self._latest_perception
        has_collectible_information = perception is not None and any(
            observation.target_class
            in {
                TargetClass.GREEN_SUPPLY,
                TargetClass.BLACK_CORE,
                TargetClass.ORANGE_INJURED,
            }
            for observation in perception.observations
        )
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
            and self._single_green_side_neighbor_requires_breakup(timestamp_ns)
        ):
            # 单绿旁边纵向齐平地贴着蓝/橙，且没有可共同收拢的其它
            # 绿/黑时，搜索态直接转入解团，不先接近到近场。
            return self._enter_breakup_only_search(timestamp_ns)
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
            self._advance_cluster_search_sweep(heading_rad)
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
            TargetClass.UNKNOWN,
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
        if target.target_class not in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE}:
            return frozenset()
        return frozenset(other.track_id for other in (self._tracker.tracks if tracks is None else tracks)
                         if other.track_id != target.track_id
                         and self._target_is_fresh(other, timestamp_ns)
                         and self._graspable_target_is_usable(other)
                         and other.target_class in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE})

    def _find_approach_seed(
        self,
        timestamp_ns: int,
    ) -> TrackedTarget | None:
        """寻找远距接近或近场交接的入口目标；不负责最终选组。"""

        policy = self.near_field_policy
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if self._target_is_fresh(target, timestamp_ns)
            and target.target_class in policy.allowed_classes
            and self._graspable_target_is_usable(target)
            and target.ground_point is not None
            and target.ground_point.x > 0.0
            and not self._candidate_path_blocked(
                target.ground_point,
                breakup=False,
            )
            and self._transport_side_neighbor_target(target, timestamp_ns) is None
            and self._transport_orange_isolation_clear(target, timestamp_ns)
            and self._green_path_is_clear_for_point(
                target, target.ground_point, timestamp_ns,
                ignored_track_ids=self._preview_ignored_track_ids(target, timestamp_ns),
            )
        )
        return min(
            candidates,
            key=lambda target: (
                math.hypot(target.ground_point.x, target.ground_point.y) > self._near_field_handoff_range_mm(),
                -({TargetClass.GREEN_SUPPLY: self._near_field_grasp_config.green_score_points,
                   TargetClass.BLACK_CORE: self._near_field_grasp_config.black_score_points,
                   TargetClass.ORANGE_INJURED: self._near_field_grasp_config.orange_score_points}[target.target_class]
                  if self._near_field_grasp_config is not None else 0),
                math.hypot(target.ground_point.x, target.ground_point.y),
                abs(target.ground_point.y),
                target.track_id,
            ),
            default=None,
        )

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
            TargetClass.UNKNOWN.value,
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
                    reason.startswith("blocked_target:")
                    for reason in rejection_reasons
                )
            )
        )
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
            )
            return route
        if plan is not None:
            self._near_field_route = GraspRoute.DIRECT_NEAR
            self._near_field_route_elapsed_ms = elapsed_ms
            self._near_field_route_rejections = rejection_reasons
            # 近场选择结果会在同一目标 ID 上持续更新几何；handoff prior
            # 只用于首轮把远场目标交给局部 tracker。
            self._near_field_handoff_prior = None
            return GraspRouteDecision(
                GraspRoute.DIRECT_NEAR,
                plan=plan,
                candidate_count=candidate_count,
                rejection_reasons=rejection_reasons,
                elapsed_ms=elapsed_ms,
            )
        timeout_ms = self._near_field_handoff_timeout_ms()
        if elapsed_ms < timeout_ms:
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
            )
            return route
        far_target = self._find_far_reapproach_target(
            timestamp_ns,
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
        )
        return route

    def _near_field_handoff_timeout_ms(self) -> float:
        config = self._near_field_grasp_config
        return 8_000.0 if config is None else float(config.alignment_timeout_ms)

    def _record_near_field_route_failure(
        self,
        route: GraspRouteDecision,
        preparation: GraspPreparation | None,
        *,
        kind: str,
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
            f"rejections={rejection}"
        )

    def _enter_breakup_only_search(self, timestamp_ns: int) -> MatchDecision:
        """近场没有安全方案时只进入解团搜索，禁止同帧重入机会抓取。"""

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
        self._begin_cluster_search()
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(
            timestamp_ns,
            0.0,
            self._cluster_search_angular_velocity_rad_s,
            f"near_field_route:reselect:{reason}",
            posture=GripperPosture.CLOSED,
        )

    def _route_after_near_field_failure(
        self,
        timestamp_ns: int,
        *,
        reason: str = "near_field_action_failed",
        geometric_block: bool = False,
    ) -> MatchDecision:
        """按失败证据选择解团、一次重接近或带原因重选。"""

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
        """检查正式入口的橙色目标 50 mm 独立性门禁。"""

        if target.target_class is not TargetClass.ORANGE_INJURED:
            return True
        point = target.ground_point
        if point is None:
            return False
        radius_mm = (
            50.0
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
            # 只有当前可定位目标才能证明其落入 50 mm 禁区；失观或缺 K0
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
                    # 侧后方物资仅作为入口预览；最终由近场完整包络证明不混运。
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

    def _transport_adjacent_supply_target(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        tracks: tuple[TrackedTarget, ...] | None = None,
    ) -> TrackedTarget | None:
        """返回预测抓取方向下横向邻近的其它绿/黑物资。"""

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
                not in {TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE}
                or other.ground_point is None
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

    def _single_green_side_neighbor_requires_breakup(
        self,
        timestamp_ns: int,
    ) -> bool:
        """判断单绿侧邻蓝且没有可增益绿黑时是否应先解团。"""

        for target in self._tracker.tracks:
            if (
                not self._target_is_fresh(target, timestamp_ns)
                or target.target_class is not TargetClass.GREEN_SUPPLY
                or not self._graspable_target_is_usable(target)
                or target.ground_point is None
                or not self._point_in_transport_corridor(target.ground_point)
            ):
                continue
            if self._transport_side_neighbor_target(target, timestamp_ns) is None:
                continue
            if self._transport_adjacent_supply_target(target, timestamp_ns) is None:
                return True
        return False

    def _transport_group_size(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        reference: GroundPoint | None = None,
    ) -> int | None:
        """返回以目标 X 为终点时覆盖的合法目标数。

        橙色伤员必须独立转运；其中心 50 mm 半径内出现当前可定位的
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
        其它新鲜目标投影到“机器人到目标”的相对坐标后，
        只要满足 ``0 <= aligned_forward < target_range`` 且
        ``abs(aligned_lateral) <= green_path_half_width_mm`` 即视为阻挡。
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
        """

        if not self._graspable_target_is_usable(target):
            return False
        if lateral_half_width_mm is None:
            lateral_half_width_mm = self.config.green_path_half_width_mm
        elif (
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
            if (
                0.0 <= other_forward_mm < target_range_mm
                and abs(other_lateral_mm) <= lateral_half_width_mm
            ):
                return False
        return True

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
    ) -> tuple[str, bool]:
        """格式化相对走廊中的其它目标，并返回实际几何阻挡标记。"""

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
        if (
            fresh
            and not consumed
            and relative is not None
            and reference is not None
        ):
            reference_range_mm = math.hypot(reference.x, reference.y)
            geometrically_blocked = (
                0.0 <= relative[0] < reference_range_mm
                and abs(relative[1]) <= self.config.green_path_half_width_mm
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
            f"consumed={consumed},"
            f"blocked={blocked_text}",
            geometrically_blocked,
        )

    def green_isolation_diagnostic(self, timestamp_ns: int) -> str:
        """返回绿色候选及所有其它目标的相对走廊诊断。

        保留原方法名以兼容日志调用方，但现在每个其它目标都会输出类别、原始
        GroundPoint、新鲜度、目标对齐坐标和阻挡结果。
        """

        tracks = self._tracker.tracks
        green_tracks = tuple(
            target
            for target in tracks
            if target.target_class is TargetClass.GREEN_SUPPLY
        )
        threshold_prefix = (
            f"threshold_lateral_mm={self.config.green_path_half_width_mm:.1f};"
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
            return threshold_prefix + "green=none;others=" + (other_entries or "none")

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
                    )
                    other_entries.append(detail)
                    if reason == "path_clear" and blocked:
                        reason = f"path_track_{other.track_id}"
            entries.append(
                f"green_track={target.track_id},class={target.target_class.value},"
                f"xy={point_text},fresh={fresh},status={target.status.value},"
                f"path={reason},others=[{'|'.join(other_entries) or 'none'}]"
            )
        return threshold_prefix + ";".join(entries)

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
            >= self.config.green_preclose_max_carried_blocks
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

    def _step_align_green(self, timestamp_ns: int) -> MatchDecision:
        """停车取 10 帧均值后做阻挡判断，再对准绿色目标。"""

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
        return NearFieldHandoffPrior(
            target.target_class,
            target.ground_point,
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
        self._near_field_session_id += 1
        self._near_field_pickup.reset()
        self._near_field_handoff_prior = handoff_prior
        self._near_field_last_failure_diagnostic = None
        self._transport_target_classes = ()
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
        self._near_field_route = GraspRoute.DECIDING
        # 窗口计时在停车后的 observation window 打开时才开始。
        self._near_field_confirmation_started_ns = None
        self._near_field_route_rejections = ()
        self._near_field_route_elapsed_ms = 0.0
        self._near_field_route_candidate_count = 0
        self.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
        self._begin_action_settle(timestamp_ns, "near_field_grasp")
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
        return self._near_field_segment_clear(plan_heading, distance_m)

    def _near_field_segment_clear(
        self,
        heading_rad: float,
        distance_m: float,
    ) -> bool | None:
        position = self.estimated_field_position
        if position is None or not math.isfinite(distance_m) or distance_m < 0.0:
            return None
        safe_zone_blocked = self._safe_zone_path_blocked(heading_rad, distance_m)
        if safe_zone_blocked is None or safe_zone_blocked:
            return None if safe_zone_blocked is None else False
        boundary = self._breakup_allowed_field_bounds()
        margin = self._breakup_clearance_mm
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
            self._near_field_handoff_prior = None
            self._near_field_far_reapproach_used = False
            self._transport_target_classes = tuple(
                TargetClass(item) for item in result.member_classes
            )
            return self._start_safe_zone_transport(
                timestamp_ns,
                transport_opened=False,
                posture=GripperPosture.CLOSED,
                reason="near_field_grasp_complete_start_safe_zone_d1_line",
            )
        if self._near_field_pickup.active_plan is not None:
            self._selected_track_id = self._near_field_pickup.active_plan.member_ids[0]
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
        if preparation is not None and preparation.session_id != self._near_field_session_id:
            preparation = None
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
        self._green_align_lost_since_ns = None
        self._selected_track_id = None
        self._selected_green_ground = None
        self._begin_cluster_search()
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(
            timestamp_ns,
            0.0,
            self._cluster_search_angular_velocity_rad_s,
            "green_target_timeout_restart_breakup_search",
        )

    def _selected_green_point(self, timestamp_ns: int) -> GroundPoint | None:
        """只返回已选目标当前新鲜的地面点，不用旧坐标继续驱动车辆。"""

        target = self._selected_target()
        if target is None:
            return None
        if (
            self._target_is_fresh(target, timestamp_ns)
            and self._selected_green_is_usable(target)
            and target.ground_point is not None
        ):
            self._selected_green_ground = target.ground_point
            return target.ground_point
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
        self._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
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

        self.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
        self._transport_opened = transport_opened
        self._gripper_phase_started_ns = None
        self._safe_zone_phase = "align_d1_line"
        self._begin_action_settle(timestamp_ns, "safe_before_d1_line")
        self._transport_forward_base_distance_m = None
        self._transport_forward_distance_m = None
        self._safe_zone_key_samples = []
        self._safe_zone_key_last_frame = None
        self._safe_zone_calibration_snapshot = None
        self._safe_zone_calibration_zone = None
        self._safe_zone_keys = None
        self._safe_zone_calibration_pose = None
        self._safe_zone_calibration_heading_rad = None
        self._safe_zone_calibration_last_failure = None
        self._safe_zone_stop_since_ns = None
        self._safe_zone_bbox_turn_direction = None
        self._safe_zone_reacquire_frame_floor = None
        self._safe_zone_calibration_after_exit = False
        self._return_phase = "idle"
        self._safe_zone_exit_base_distance_m = None
        self._d1_line_heading_rad = None
        self._d1_line_distance_m = None
        self._d1_line_start_position = None
        self._d2_line_heading_rad = None
        self._d2_line_distance_m = None
        self._d2_line_start_position = None
        self._search_frame_floor = None
        position = self.estimated_field_position
        d1_target = self._safe_zone_d1_target()
        if position is not None and self._safe_zone_forward_y_sign * (
            position.y - d1_target.y
        ) >= -self.config.transport_align_tolerance_mm:
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

    @staticmethod
    def _safe_zone_has_complete_ground_keypoints(
        zone: SafeZoneObservation | None,
    ) -> bool:
        """判断安全区是否同时具备可用于校准的 K0/K1/K2 地面点。"""

        return MatchSequence._safe_zone_keypoint_count(zone) == 3

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
        return f"bbox_u={bbox_center_u:.1f},image_center_u={image_width / 2.0:.1f}"

    def _safe_zone_bbox_turn_command(
        self,
        zone: SafeZoneObservation,
    ) -> float:
        """按 bbox 水平位置选择左/右原地旋转，正角速度为左转。"""

        perception = self._latest_perception
        if perception is None or perception.field_features is None:
            return 0.0
        image_width, _ = perception.field_features.image_size
        bbox_center_u = 0.5 * (zone.box.x_min + zone.box.x_max)
        image_center_u = image_width / 2.0
        if bbox_center_u < image_center_u:
            direction = 1.0
        elif bbox_center_u > image_center_u:
            direction = -1.0
        else:
            direction = self._safe_zone_bbox_turn_direction or 1.0
        self._safe_zone_bbox_turn_direction = direction
        return direction * abs(self.config.safe_zone_key_search_angular_velocity_rad_s)

    def _safe_zone_bbox_fully_visible(
        self,
        zone: SafeZoneObservation,
    ) -> bool:
        """要求安全区 bbox 离开图像裁剪边缘后才采集关键点。"""

        perception = self._latest_perception
        if perception is None or perception.field_features is None:
            return False
        width, height = perception.field_features.image_size
        margin = self.config.safe_zone_bbox_edge_margin_px
        return (
            zone.box.x_min >= margin
            and zone.box.y_min >= margin
            and zone.box.x_max <= width - margin
            and zone.box.y_max <= height - margin
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
        return direction * abs(self.config.safe_zone_key_search_angular_velocity_rad_s)

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
        """搜索己方安全区，直到出现完整 K0/K1/K2。"""

        posture = (
            GripperPosture.OPEN
            if self._safe_zone_calibration_after_exit
            else GripperPosture.CLOSED
        )
        perception = self._latest_perception
        zone = self._own_safe_zone_observation()
        reacquire_floor = self._safe_zone_reacquire_frame_floor
        if reacquire_floor is not None:
            if perception is None or perception.frame_sequence <= reacquire_floor:
                if zone is None:
                    angular_velocity = self._safe_zone_no_bbox_turn_command()
                else:
                    angular_velocity = self._safe_zone_bbox_turn_command(zone)
                return self._decision(
                    timestamp_ns,
                    0.0,
                    angular_velocity,
                    "safe_zone_visual_calibration_rejected_rotate_for_keypoints",
                    posture=posture,
                )
            # 拟合失败后的当前帧不能再次触发停止；至少等到一帧新的感知结果，
            # 再按关键点数量决定是继续转动还是进入停稳取样。
            self._safe_zone_reacquire_frame_floor = None
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
            self._safe_zone_bbox_fully_visible(zone)
            and self._safe_zone_has_complete_ground_keypoints(zone)
        ):
            self._safe_zone_phase = "stopping_after_bbox_keypoints"
            self._safe_zone_stop_since_ns = None
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_seen_stop_before_calibration",
                posture=posture,
            )
        return self._decision(
            timestamp_ns,
            0.0,
            self._safe_zone_bbox_turn_command(zone),
            "safe_zone_center_full_bbox_before_keypoints",
            posture=posture,
        )

    def _safe_zone_braking_compensated_target(
        self,
        nominal_target: FieldPoint,
    ) -> FieldPoint:
        """按配置的场地 x/y 刹车过冲提前量修正停车目标。"""

        return FieldPoint(
            nominal_target.x - self.config.safe_zone_d2_braking_overrun_x_mm,
            nominal_target.y - self.config.safe_zone_d2_braking_overrun_y_mm,
        )

    def _safe_zone_d1_target(self) -> FieldPoint:
        """返回应用刹车过冲补偿后的 d1 场地停车目标。"""

        endpoint = self._safe_zone_transport_endpoint()
        return self._safe_zone_braking_compensated_target(
            FieldPoint(
                endpoint.x,
                endpoint.y
                - self._safe_zone_forward_y_sign
                * self.config.safe_zone_calibration_start_offset_mm,
            )
        )

    def _safe_zone_d2_target(self) -> FieldPoint:
        """返回应用刹车过冲补偿后的 d2 场地停车目标。"""

        endpoint = self._safe_zone_transport_endpoint()
        return self._safe_zone_braking_compensated_target(
            FieldPoint(
                endpoint.x,
                endpoint.y
                - self._safe_zone_forward_y_sign
                * self.config.safe_zone_open_offset_mm,
            )
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
    ) -> bool:
        """判断直线目标是否命中容差，或沿计划方向已经越过目标。

        直线起点已经落在阈值内的坐标不参与判断，避免某一坐标在直线起点
        已经达标时提前结束。车辆偏航导致一次采样跨过二维容差窗口时，
        使用起点到目标向量的投影进度触发越过保护。
        """

        position = self._fallback_field_position
        if position is None or start is None:
            return False
        tolerance = self.config.transport_align_tolerance_mm
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

    def _collect_safe_zone_key_sample(self) -> bool:
        """从最新独立帧收集一次 K0/K1/K2，不重复使用同一帧。"""

        perception = self._latest_perception
        zone = self._own_safe_zone_observation()
        if perception is None or zone is None:
            return False
        if perception.frame_sequence == self._safe_zone_key_last_frame:
            return False
        k0 = zone.ground_anchor.ground
        k1 = zone.image_left_landmark.ground
        k2 = zone.image_right_landmark.ground
        if k0 is None or k1 is None or k2 is None:
            return False
        self._safe_zone_key_samples.append((k0, k1, k2))
        self._safe_zone_key_last_frame = perception.frame_sequence
        self._safe_zone_calibration_snapshot = perception
        self._safe_zone_calibration_zone = zone
        return True

    def _lock_safe_zone_calibration_plan(self, timestamp_ns: int) -> bool:
        """用 5 帧 K0/K1/K2 均值拟合并覆盖当前场地位姿。"""

        if len(self._safe_zone_key_samples) < 5:
            return False
        count = len(self._safe_zone_key_samples)
        k0 = GroundPoint(
            sum(sample[0].x for sample in self._safe_zone_key_samples) / count,
            sum(sample[0].y for sample in self._safe_zone_key_samples) / count,
        )
        k1 = GroundPoint(
            sum(sample[1].x for sample in self._safe_zone_key_samples) / count,
            sum(sample[1].y for sample in self._safe_zone_key_samples) / count,
        )
        k2 = GroundPoint(
            sum(sample[2].x for sample in self._safe_zone_key_samples) / count,
            sum(sample[2].y for sample in self._safe_zone_key_samples) / count,
        )
        if math.hypot(k2.x - k1.x, k2.y - k1.y) <= 1e-6:
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

        averaged_zone = replace(
            zone,
            ground_anchor=replace(zone.ground_anchor, ground=k0),
            image_left_landmark=replace(zone.image_left_landmark, ground=k1),
            image_right_landmark=replace(zone.image_right_landmark, ground=k2),
        )
        averaged_features: FieldFeatureDetectionResult = replace(
            snapshot.field_features,
            safe_zones=(averaged_zone,),
        )
        # FieldPoseKeypoint.ground 已由 detector 通过 GroundProjector 生成；这里
        # 只把多帧均值交回同一套安全区静态地标/外参几何拟合，不复制投影矩阵。
        calibration = localizer.localize(
            averaged_features,
            prior_pose=FieldPose2D(position, heading),
            current_timestamp_ns=timestamp_ns,
        )
        if calibration is None:
            self._safe_zone_calibration_last_failure = (
                "safe_zone_pose_fit_rejected"
            )
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
        position = self._fallback_field_position
        if position is None or heading_rad is None or cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_route_waiting_for_field_odometry",
                posture=GripperPosture.CLOSED,
            )
        if self._safe_zone_phase in {"align_d1_line", "align_d2_line"}:
            is_d1_line = self._safe_zone_phase == "align_d1_line"
            target = (
                self._safe_zone_d1_target()
                if is_d1_line
                else self._safe_zone_d2_target()
            )
            delta_x = target.x - position.x
            delta_y = target.y - position.y
            distance_mm = math.hypot(delta_x, delta_y)
            tolerance = self.config.transport_align_tolerance_mm
            if is_d1_line:
                self._d1_line_start_position = position
            else:
                self._d2_line_start_position = position
            if distance_mm <= 1e-6 or (
                abs(delta_x) <= tolerance and abs(delta_y) <= tolerance
            ):
                if is_d1_line:
                    self._d1_line_heading_rad = heading_rad
                    self._d1_line_distance_m = 0.0
                    forward_phase = "forward_d1_line"
                    settle_phase = "safe_before_d1_forward"
                    reason = "safe_zone_d1_coordinates_within_threshold_start_d1_pause"
                else:
                    self._d2_line_heading_rad = heading_rad
                    self._d2_line_distance_m = 0.0
                    forward_phase = "forward_d2_line"
                    settle_phase = "safe_before_d2_forward"
                    reason = "safe_zone_d2_coordinates_within_threshold_start_d2_pause"
                self._transport_forward_base_distance_m = cumulative_distance_m
                self._transport_forward_distance_m = 0.0
                self._safe_zone_stop_since_ns = None
                self._safe_zone_phase = forward_phase
                self.state = MatchState.TRANSPORT_FORWARD
                self._begin_action_settle(
                    timestamp_ns,
                    settle_phase,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    reason,
                    posture=GripperPosture.CLOSED,
                )

            target_heading = math.atan2(delta_y, delta_x)
            if is_d1_line:
                self._d1_line_heading_rad = target_heading
                turn_reason = "safe_zone_turn_to_d1_line"
                forward_phase = "forward_d1_line"
                settle_phase = "safe_before_d1_forward"
                start_reason = "safe_zone_d1_line_heading_reached_start_forward"
            else:
                self._d2_line_heading_rad = target_heading
                turn_reason = "safe_zone_turn_to_d2_line"
                forward_phase = "forward_d2_line"
                settle_phase = "safe_before_d2_forward"
                start_reason = "safe_zone_d2_line_heading_reached_start_forward"
            heading_error = normalize_angle(target_heading - heading_rad)
            if abs(heading_error) > self.config.safe_zone_fallback_heading_tolerance_rad:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    _clamp(
                        self.config.safe_zone_fallback_heading_kp_rad_s * heading_error,
                        -self.config.safe_zone_fallback_max_angular_velocity_rad_s,
                        self.config.safe_zone_fallback_max_angular_velocity_rad_s,
                    ),
                    turn_reason,
                    posture=GripperPosture.CLOSED,
                )

            if is_d1_line:
                self._d1_line_distance_m = distance_mm / 1000.0
                line_distance_m = self._d1_line_distance_m
            else:
                self._d2_line_distance_m = distance_mm / 1000.0
                line_distance_m = self._d2_line_distance_m
            self._transport_forward_base_distance_m = cumulative_distance_m
            self._transport_forward_distance_m = line_distance_m
            self._safe_zone_stop_since_ns = None
            self._safe_zone_phase = forward_phase
            self.state = MatchState.TRANSPORT_FORWARD
            self._begin_action_settle(
                timestamp_ns,
                settle_phase,
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                start_reason,
                posture=GripperPosture.CLOSED,
            )

        if self._safe_zone_phase == "align_y_at_d2":
            heading_error = normalize_angle(
                self._safe_zone_forward_heading_rad() - heading_rad
            )
            if abs(heading_error) > self.config.safe_zone_fallback_heading_tolerance_rad:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    _clamp(
                        self.config.safe_zone_fallback_heading_kp_rad_s * heading_error,
                        -self.config.safe_zone_fallback_max_angular_velocity_rad_s,
                        self.config.safe_zone_fallback_max_angular_velocity_rad_s,
                    ),
                    "safe_zone_d2_turn_to_"
                    f"{self._safe_zone_heading_label()}",
                    posture=GripperPosture.OPEN,
                )
            self._safe_zone_phase = "stopping_after_d2_heading"
            self._safe_zone_stop_since_ns = None
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_d2_heading_"
                f"{self._safe_zone_heading_label()}_reached_stop_before_forward",
                posture=GripperPosture.OPEN,
            )

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
                f"safe_zone_keypoints={keypoint_count}/3,"
                f"safe_zone_bbox_center={bbox_center_text},"
                f"safe_zone_bbox_turn_direction={bbox_turn_direction_text},"
                f"transport_endpoint=({transport_endpoint.x:.1f},{transport_endpoint.y:.1f}),"
                f"d1_target=({d1_target.x:.1f},{d1_target.y:.1f}),"
                f"d2_target=({d2_target.x:.1f},{d2_target.y:.1f}),"
                f"safe_zone_braking_overrun_mm=({self.config.safe_zone_d2_braking_overrun_x_mm:.1f},"
                f"{self.config.safe_zone_d2_braking_overrun_y_mm:.1f}),"
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
            f"safe_zone_keypoints={keypoint_count}/3,"
            f"safe_zone_bbox_center={bbox_center_text},"
            f"safe_zone_bbox_turn_direction={bbox_turn_direction_text},"
            f"calibration_y={transport_endpoint.y:.1f}{calibration_y_offset_text},"
            f"d1_line_heading_deg={d1_line_heading_text},"
            f"d1_line_distance_m={d1_line_distance_text},"
            f"d2_line_heading_deg={d2_line_heading_text},"
            f"d2_line_distance_m={d2_line_distance_text},"
            f"safe_zone_braking_overrun_mm=({self.config.safe_zone_d2_braking_overrun_x_mm:.1f},"
            f"{self.config.safe_zone_d2_braking_overrun_y_mm:.1f}),"
            f"final_nominal_y={transport_endpoint.y:.0f},"
            f"final_target_y={self._safe_zone_final_target_y_mm():.0f},"
            f"final_speed_m_s={final_speed:.3f},"
            f"final_braking_overrun_mm={final_braking_overrun:.1f},"
            f"d1={self.config.safe_zone_calibration_start_offset_mm:.1f},"
            f"d2={self.config.safe_zone_open_offset_mm:.1f},"
            f"key_samples={len(self._safe_zone_key_samples)}"
        )

    def _step_transport_forward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """执行夹取后 d1 直线段、视觉校准后的 d2 直线段和末段前进。"""

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

        if self._safe_zone_phase == "forward_d1_line":
            if self._d1_line_coordinate_threshold_reached():
                self._transport_forward_base_distance_m = None
                self._transport_forward_distance_m = None
                self._safe_zone_phase = "stopping_before_calibration"
                self._safe_zone_key_samples = []
                self._safe_zone_key_last_frame = None
                self._safe_zone_stop_since_ns = None
                self.state = MatchState.TRANSPORT_RELEASE
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_d1_coordinate_threshold_reached_stop_before_calibration",
                    posture=GripperPosture.CLOSED,
                )
            angular = self._heading_hold_angular_velocity(
                self._d1_line_heading_rad,
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
                self.config.safe_zone_grab_to_d1_speed_m_s,
                angular,
                "safe_zone_forward_along_gripper_to_d1_line",
                posture=GripperPosture.CLOSED,
            )

        if self._safe_zone_phase == "forward_d2_line":
            coordinate_threshold_reached = self._d2_line_coordinate_threshold_reached()
            if coordinate_threshold_reached:
                self._transport_forward_base_distance_m = None
                self._transport_forward_distance_m = None
                self.state = MatchState.TRANSPORT_RELEASE
                self._safe_zone_phase = "stopping_before_d2_opening"
                self._safe_zone_stop_since_ns = None
                if self._begin_action_settle(
                    timestamp_ns,
                    "safe_before_d2_opening",
                ):
                    return self._decision(
                        timestamp_ns,
                        0.0,
                        0.0,
                        "safe_zone_d2_reached_wait_before_opening",
                        posture=GripperPosture.CLOSED,
                    )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_d2_coordinate_threshold_reached_wait_before_opening",
                    posture=GripperPosture.CLOSED,
                )
            angular = self._heading_hold_angular_velocity(
                self._d2_line_heading_rad,
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
                self.config.safe_zone_d1_to_d2_speed_m_s,
                angular,
                "safe_zone_forward_along_d1_d2_line",
                posture=GripperPosture.CLOSED,
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
        """倒车退出、视觉纠偏并取得新帧后才进入下一轮搜索。"""

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
            base_distance = self._safe_zone_exit_base_distance_m
            if cumulative_distance_m is None or base_distance is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_exit_waiting_for_odometry",
                    posture=GripperPosture.OPEN,
                )
            travelled = base_distance - cumulative_distance_m
            if travelled >= self.config.safe_zone_exit_distance_m - 1e-9:
                self._return_phase = "stopping_after_exit"
                self._safe_zone_stop_since_ns = None
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_exit_distance_reached_wait_for_stop",
                    posture=GripperPosture.OPEN,
                )
            return self._decision(
                timestamp_ns,
                -self.config.return_backup_speed_m_s,
                0.0,
                "safe_zone_exit_reverse_to_field",
                posture=GripperPosture.OPEN,
            )

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
            self._safe_zone_stop_since_ns = None
            self._safe_zone_calibration_after_exit = True
            self._safe_zone_phase = "searching_safe_zone_keypoints"
            self._safe_zone_key_samples = []
            self._safe_zone_key_last_frame = None
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
            self._safe_zone_calibration_last_failure = None
            self._safe_zone_bbox_turn_direction = None
            self._safe_zone_reacquire_frame_floor = None
            self._return_phase = "calibrating_after_exit"
            self.state = MatchState.TRANSPORT_RELEASE
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_exit_stopped_start_visual_calibration",
                posture=GripperPosture.OPEN,
            )

        if phase == "waiting_for_new_search_frame":
            perception = self._latest_perception
            frame_floor = self._search_frame_floor
            if perception is None or not self._fresh_perception(
                perception, timestamp_ns
            ) or (
                frame_floor is not None and perception.frame_sequence <= frame_floor
            ):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_exit_waiting_for_new_perception",
                    posture=GripperPosture.OPEN,
                )
            self._return_phase = "idle"
            self._search_frame_floor = None
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_exit_complete_start_search",
                posture=GripperPosture.OPEN,
            )

        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "safe_zone_exit_unknown_phase_hold",
            posture=GripperPosture.OPEN,
        )

    def _finish_post_exit_visual_calibration(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """纠偏成功后清空交付期目标，并等待一帧新的搜索输入。"""

        self._safe_zone_calibration_after_exit = False
        self._safe_zone_phase = "idle"
        self._reset_tracker_for_new_preview_epoch()
        self._selected_track_id = None
        self._transport_target_classes = ()
        self._return_backup_base_distance_m = None
        self._transport_forward_distance_m = None
        frame_floor = self._last_tracker_frame_sequence
        perception = self._latest_perception
        if perception is not None:
            frame_floor = max(
                frame_floor if frame_floor is not None else perception.frame_sequence,
                perception.frame_sequence,
            )
        self._search_frame_floor = frame_floor
        self._return_phase = "waiting_for_new_search_frame"
        self.state = MatchState.RETURN_BACKUP
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "safe_zone_exit_visual_calibrated_waiting_for_new_perception",
            posture=GripperPosture.OPEN,
        )

    def _step_transport_release(self, timestamp_ns: int) -> MatchDecision:
        """处理 d2 张爪、转向、闭爪推进，以及推进停稳后的张爪释放。"""

        calibration_posture = (
            GripperPosture.OPEN
            if self._safe_zone_calibration_after_exit
            else GripperPosture.CLOSED
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
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
            self._safe_zone_calibration_last_failure = None
            self._safe_zone_bbox_turn_direction = None
            self._safe_zone_reacquire_frame_floor = None
            return self._step_safe_zone_bbox_key_search(timestamp_ns)

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
            self._safe_zone_key_samples = []
            self._safe_zone_key_last_frame = None
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
            self._safe_zone_calibration_last_failure = None
            self._safe_zone_reacquire_frame_floor = None
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_stopped_start_calibration",
                posture=calibration_posture,
            )

        if self._safe_zone_phase == "collecting_safe_zone_keys_closed":
            zone = self._own_safe_zone_observation()
            if (
                zone is None
                or not self._safe_zone_bbox_fully_visible(zone)
                or not self._safe_zone_has_complete_ground_keypoints(zone)
            ):
                self._safe_zone_phase = "searching_safe_zone_keypoints"
                self._safe_zone_bbox_turn_direction = None
                self._safe_zone_key_samples = []
                self._safe_zone_key_last_frame = None
                self._safe_zone_calibration_snapshot = None
                self._safe_zone_calibration_zone = None
                return self._step_safe_zone_bbox_key_search(timestamp_ns)
            collected = self._collect_safe_zone_key_sample()
            if collected and len(self._safe_zone_key_samples) >= 5:
                if not self._lock_safe_zone_calibration_plan(timestamp_ns):
                    self._safe_zone_key_samples = []
                    self._safe_zone_key_last_frame = None
                    self._safe_zone_calibration_snapshot = None
                    self._safe_zone_calibration_zone = None
                    self._safe_zone_phase = "searching_safe_zone_keypoints"
                    self._safe_zone_bbox_turn_direction = None
                    perception = self._latest_perception
                    self._safe_zone_reacquire_frame_floor = (
                        perception.frame_sequence if perception is not None else None
                    )
                    return self._step_safe_zone_bbox_key_search(timestamp_ns)
                if self._safe_zone_calibration_after_exit:
                    return self._finish_post_exit_visual_calibration(timestamp_ns)
                self._gripper_phase_started_ns = None
                self.state = MatchState.TRANSPORT_ALIGN_RED_ZONE
                self._safe_zone_phase = "align_d2_line"
                self._safe_zone_reacquire_frame_floor = None
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_visual_calibrated_start_d2_line",
                    posture=calibration_posture,
                )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_collecting_5_point_reference_closed",
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

    def _safe_zone_vehicle_stopped(self, timestamp_ns: int) -> bool:
        """要求左右轮速反馈持续低于阈值，才允许进入下一步。"""

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
