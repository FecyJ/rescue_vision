"""正式比赛流程：基于当前正式动作编排，纯逻辑不打开硬件。"""
from __future__ import annotations

import argparse
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING

from rescue_vision.app.cluster_breakup import GripperPosture
from rescue_vision.app.breakup_planner import (
    BreakupPlan, BreakupTarget, physical_radii, plan_breakup,
    safe_zone_intersection, segment_clear,
)
from rescue_vision.perception.target_ground_geometry import TargetGroundGeometryConfig

from rescue_vision.app.gripper_width_sequence import (
    GraspPreparation,
    GripperWidthPickupResult,
    GripperWidthPickupSequence,
)
from rescue_vision.app.near_field_grasp import (
    GraspScore,
    GraspSelection,
    GraspTarget,
    NearFieldGraspPlan,
    NearFieldHandoffPrior,
    NearFieldGraspPolicy,
    NearFieldGraspSelector,
)
from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.geometry.ground_projector import GroundProjector
from rescue_vision.localization import (
    FieldPose2D,
    SafeZoneCornerLocalizer,
    normalize_angle,
)
from rescue_vision.perception import (
    ObservationQuality,
    FieldFeatureDetectionResult,
    PerceptionSnapshot,
    SafeZoneColor,
    SafeZoneObservation,
    TargetClass,
    UndistortedBoundingBox,
)
from rescue_vision.perception.gripper_width import measure_target_envelope
from rescue_vision.tracking import MultiTargetTracker, TrackStatus, TrackedTarget
from rescue_vision.mission import SafetySignals
from rescue_vision.motion.gripper_kinematics import GripperKinematics
from rescue_vision.motion.protocol import OdometryImu
from rescue_vision.world.static_map import PhysicalRegionKind, StaticFieldMap, TeamColor

# ``match_runtime`` 是共用硬件循环，并按身份比较这些流程类型。策略入口不再
# 保留正式流程类型的实现副本，直接引用正式入口的共享类型，确保状态比较、
# 预检和共享观察器在独立启动策略时仍然有效。``MatchSequence`` 需要保持别名，
# 以便阶段分派显式委托回正式实现。
from rescue_vision.app.match import (
    GraspRoute,
    MatchDecision,
    MatchPreflight,
    MatchSequence as _SharedMatchSequence,
    MatchStartArea,
    MatchState,
    configure_match_start_area,
)

if TYPE_CHECKING:
    from rescue_vision.config import AppConfig, MatchRuntimeConfig


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


@dataclass(frozen=True, slots=True)
class _GroundClusterMeasurement:
    """当前目标团的中心和最前方有效地面锚点距离，单位 mm。"""

    center: GroundPoint
    nearest_forward_x_mm: float

class MatchSequence(_SharedMatchSequence):
    """可重放的正式动作流程；step() 只消费观测、里程和航向并输出意图。"""

    # 独立的抓取—运输联调入口覆盖为只搜索，不执行正式解团路由。
    _first_green_blocked_routes_to_breakup = True
    _dynamic_breakup_enabled = True

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
    ) -> None:
        """共用正式流程的全部状态，只追加策略专用字段。

        正式流程的状态集合（位姿历史、物理失败记忆、旋转预算、绿色对准
        预算等）由基类建立；本变体不再复制一份，避免再次出现"变体缺少
        基类新增状态"的静默 AttributeError。
        """

        super().__init__(
            config,
            tracker=tracker,
            gripper_full_travel_time_s=gripper_full_travel_time_s,
            team_color=team_color,
            initial_field_position=initial_field_position,
            safe_zone_corner_localizer=safe_zone_corner_localizer,
            static_map=static_map,
            breakup_clearance_mm=breakup_clearance_mm,
            near_field_pickup=near_field_pickup,
            near_field_grasp_config=near_field_grasp_config,
            transport_corridor_half_width_mm=transport_corridor_half_width_mm,
            breakup_target_geometry=breakup_target_geometry,
        )
        # 策略专用启动冲刺：第一段为右转/短冲，第二段为左转/长冲。
        self._strategy_startup_leg = 1
        self._strategy_blue_transport = False
        self._strategy_delivery_slot = 0
        self._strategy_safe_zone_color: SafeZoneColor | None = None
        self._strategy_blue_grasp_phase = "idle"
        self._strategy_saved_near_field_pickup: GripperWidthPickupSequence | None = None
        self._strategy_blue_projector: GroundProjector | None = None
        self._strategy_blue_selector: NearFieldGraspSelector | None = None
        self._strategy_blue_last_preparation: GraspPreparation | None = None
        self._strategy_blue_preparation_key: tuple[int, int, int] | None = None
        self._strategy_blue_confirmation_count = 0
        self._strategy_blue_confirmation_last_frame: int | None = None
        self._strategy_blue_confirmation_ids: tuple[int, ...] | None = None
        self._strategy_formal_phase = False

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
        if cls._dynamic_breakup_enabled and runtime.breakup_max_wheel_acceleration_m_s2 is None:
            runtime = replace(runtime, breakup_max_wheel_acceleration_m_s2=config.motion.max_wheel_acceleration_m_s2)
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
        strategy_geometry = config.build_geometry()
        if strategy_geometry is None or not isinstance(
            strategy_geometry.ground_projector,
            GroundProjector,
        ):
            raise RuntimeError(
                "Strategy blue grasp requires a GroundProjector."
            )
        strategy_selector = NearFieldGraspSelector(
            config.near_field_grasp,
            strategy_geometry.ground_projector,
            GripperKinematics(),
            target_geometry=config.perception.target_ground_geometry,
            open_servo_angles_deg=(
                gripper.open_left_angle_deg,
                gripper.open_right_angle_deg,
            ),
            closed_servo_angles_deg=(
                gripper.closed_left_angle_deg,
                gripper.closed_right_angle_deg,
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
            breakup_target_geometry=config.perception.target_ground_geometry,
        )
        # 该入口只执行单个蓝色危险物块；近场动作复用正式 match 的
        # GripperWidthPickupSequence，唯一差别是蓝色目标由本文件生成同构计划。
        sequence._strategy_saved_near_field_pickup = near_field_pickup
        sequence._strategy_blue_projector = strategy_geometry.ground_projector
        sequence._strategy_blue_selector = strategy_selector
        sequence._near_field_pickup = None
        sequence._strategy_safe_zone_color = (
            SafeZoneColor.BLUE
            if config.world.team_color is TeamColor.RED
            else SafeZoneColor.RED
        )
        endpoint = runtime.safe_zone_fallback_target_field
        if math.isclose(endpoint.y, initial.position.y, abs_tol=1e-6):
            sequence._safe_zone_forward_y_sign = (
                -1.0 if config.world.team_color is TeamColor.BLUE else 1.0
            )
        else:
            sequence._safe_zone_forward_y_sign = (
                1.0 if endpoint.y > initial.position.y else -1.0
            )
        sequence._strategy_startup_leg = 1
        sequence._strategy_blue_transport = False
        sequence._strategy_delivery_slot = 0
        sequence._strategy_formal_phase = False
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
        self._stationary_motion.observe(message)

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
        return is_open

    @property
    def near_field_policy(self) -> NearFieldGraspPolicy:
        if self._transport_count == 0:
            return NearFieldGraspPolicy(frozenset((TargetClass.GREEN_SUPPLY,)), 1)
        if self._greedy_active:
            return NearFieldGraspPolicy(
                frozenset((TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE)),
                max(1, self._supply_capacity() - len(self._transport_target_classes)),
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
        """重置策略专用状态，再复用正式流程的启动复位。"""

        self._strategy_startup_leg = 1
        self._strategy_blue_transport = False
        self._strategy_delivery_slot = self._transport_count
        self._strategy_blue_grasp_phase = "idle"
        self._strategy_blue_preparation_key = None
        self._strategy_formal_phase = False
        return super().start(timestamp_ns)

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
        self._breakup_grasp_preparation = near_field_preparation
        self._observe_breakup_feedback(timestamp_ns)

        if self._dynamic_breakup_enabled:
            dynamic = self._step_dynamic_breakup(timestamp_ns, cumulative_distance_m)
            if dynamic is not None:
                return dynamic

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
            return self._step_strategy_startup_forward_settle(timestamp_ns)
        if self.state is MatchState.SEARCH_CLUSTER:
            if self._strategy_formal_phase:
                return _SharedMatchSequence._step_search_cluster(
                    self,
                    timestamp_ns,
                    heading_rad,
                )
            return self._step_strategy_search_cluster(timestamp_ns, heading_rad)
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
            targets.append(BreakupTarget(target.track_id, target.last_seen_timestamp_ns,
                                         target.target_class, target.ground_point, contact, safety))
        return tuple(targets)

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

    def _dynamic_segment_speed(self, remaining_mm: float, maximum_m_s: float) -> float:
        acceleration = self.config.breakup_deceleration_m_s2
        if self.config.breakup_max_wheel_acceleration_m_s2 is not None:
            acceleration = min(acceleration, self.config.breakup_max_wheel_acceleration_m_s2)
        distance_m = max(0.0, remaining_mm-self.config.breakup_braking_margin_mm)/1000.0
        return min(maximum_m_s, math.sqrt(2*acceleration*distance_m))

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
        if not segment_clear(self._breakup_static_map, origin, end, self._breakup_allowed_field_bounds(), self._breakup_clearance_mm):
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
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if target.ground_point is not None
            and target.ever_confirmed
            and self._target_is_fresh(target, timestamp_ns)
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
        end = FieldPoint(position.x + distance_m * 1000.0 * math.cos(heading),
                         position.y + distance_m * 1000.0 * math.sin(heading))
        return safe_zone_intersection(self._breakup_static_map, position, end, self._breakup_clearance_mm)

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

    _greedy_pickup_enabled = True

    def _supply_capacity(self) -> int:
        config = self._near_field_grasp_config
        return 3 if config is None else config.max_targets

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
            else config.target_final_x_mm + config.range_hysteresis_mm
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
        self._greedy_active = False
        self._near_field_handoff_prior = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id += 1
        return self._start_safe_zone_transport(
            timestamp_ns, transport_opened=False, posture=GripperPosture.CLOSED,
            reason=f"greedy_return:{reason}",
        )

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
        self._safe_zone_key_reobserve_until_ns = None
        self._safe_zone_key_reobserve_frame_floor = None
        self._safe_zone_keypoint_scan_until_ns = None
        self._safe_zone_keypoint_scan_frame_floor = None
        self._safe_zone_keypoint_scan_attempted = False
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
        self._reset_safe_zone_turn_gate()
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

    def _reset_safe_zone_turn_gate(self) -> None:
        """清理 bbox 追框的滞回和换向停车门禁。"""

        self._safe_zone_bbox_centered = False
        self._safe_zone_turn_reversal_requested_ns = None
        self._safe_zone_turn_reversal_frame_floor = None
        self._safe_zone_turn_reversal_stop_confirmed_ns = None
        self._safe_zone_turn_reversal_stop_frame_floor = None

    def _safe_zone_request_turn_reversal(self, timestamp_ns: int) -> None:
        """记录一次换向请求；实际换向必须等待停稳及停车后新帧。"""

        self._safe_zone_turn_reversal_requested_ns = timestamp_ns
        perception = self._latest_perception
        self._safe_zone_turn_reversal_frame_floor = (
            None if perception is None else perception.frame_sequence
        )
        self._safe_zone_turn_reversal_stop_confirmed_ns = None
        self._safe_zone_turn_reversal_stop_frame_floor = None
        # 重新使用已有的轮速停稳确认，不能沿用转向前的静止区间。
        self._safe_zone_stop_since_ns = None

    def _safe_zone_reversal_waiting_decision(
        self,
        timestamp_ns: int,
        *,
        posture: GripperPosture,
    ) -> MatchDecision | None:
        """返回换向门禁意图；满足停稳和新帧后返回 None。"""

        requested = self._safe_zone_turn_reversal_requested_ns
        if requested is None:
            return None
        if not self._safe_zone_vehicle_stopped(timestamp_ns):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_waiting_for_vehicle_stop_before_reversal",
                posture=posture,
            )
        if self._safe_zone_turn_reversal_stop_confirmed_ns is None:
            self._safe_zone_turn_reversal_stop_confirmed_ns = timestamp_ns
            perception = self._latest_perception
            self._safe_zone_turn_reversal_stop_frame_floor = (
                None if perception is None else perception.frame_sequence
            )
        perception = self._latest_perception
        frame_floor = self._safe_zone_turn_reversal_stop_frame_floor
        stop_confirmed = self._safe_zone_turn_reversal_stop_confirmed_ns
        if (
            perception is None
            or frame_floor is not None
            and perception.frame_sequence <= frame_floor
            or stop_confirmed is not None
            and perception.capture_timestamp_ns <= stop_confirmed
        ):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_waiting_for_new_frame_before_reversal",
                posture=posture,
            )
        # 允许当前新帧重新决定方向；清空旧方向以免再次被识别成同一次换向。
        self._safe_zone_turn_reversal_requested_ns = None
        self._safe_zone_turn_reversal_frame_floor = None
        self._safe_zone_turn_reversal_stop_confirmed_ns = None
        self._safe_zone_turn_reversal_stop_frame_floor = None
        self._safe_zone_bbox_turn_direction = None
        return None

    def _safe_zone_bbox_turn_command(
        self,
        zone: SafeZoneObservation,
        timestamp_ns: int,
    ) -> float:
        """按 bbox 归一化水平误差比例转向，正角速度为左转。"""

        perception = self._latest_perception
        if perception is None or perception.field_features is None:
            return 0.0
        image_width, _ = perception.field_features.image_size
        bbox_center_u = 0.5 * (zone.box.x_min + zone.box.x_max)
        image_center_u = image_width / 2.0
        half_width = image_width / 2.0
        if half_width <= 0.0:
            return 0.0
        normalized_error = (bbox_center_u - image_center_u) / half_width
        magnitude = abs(normalized_error)
        deadband = self.config.safe_zone_bbox_turn_deadband_ratio
        resume = self.config.safe_zone_bbox_turn_resume_ratio
        if magnitude <= deadband:
            self._safe_zone_bbox_centered = True
            return 0.0
        if self._safe_zone_bbox_centered and magnitude <= resume:
            return 0.0
        self._safe_zone_bbox_centered = False
        angular = -self.config.safe_zone_bbox_turn_kp_rad_s * normalized_error
        angular = max(
            -self.config.safe_zone_bbox_turn_max_angular_velocity_rad_s,
            min(self.config.safe_zone_bbox_turn_max_angular_velocity_rad_s, angular),
        )
        if abs(angular) < 1e-12:
            return 0.0
        direction = 1.0 if angular > 0.0 else -1.0
        previous_direction = self._safe_zone_bbox_turn_direction
        if (
            previous_direction is not None
            and previous_direction * direction < 0.0
        ):
            self._safe_zone_request_turn_reversal(timestamp_ns)
            return 0.0
        self._safe_zone_bbox_turn_direction = direction
        return angular

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
        """先把安全区 bbox 居中，再以有界扫视寻找完整 K0/K1/K2。"""

        posture = (
            GripperPosture.OPEN
            if self._safe_zone_calibration_after_exit
            else GripperPosture.CLOSED
        )
        perception = self._latest_perception
        reversal_wait = self._safe_zone_reversal_waiting_decision(
            timestamp_ns,
            posture=posture,
        )
        if reversal_wait is not None:
            return reversal_wait
        zone = self._own_safe_zone_observation()
        reacquire_floor = self._safe_zone_reacquire_frame_floor
        if reacquire_floor is not None:
            if perception is None or perception.frame_sequence <= reacquire_floor:
                if zone is None:
                    angular_velocity = self._safe_zone_no_bbox_turn_command()
                else:
                    angular_velocity = self._safe_zone_bbox_turn_command(
                        zone, timestamp_ns
                    )
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
        # 关键点即使已经出现，也必须先把安全区 bbox 带到图像水平中心；
        # 这样取样时保持的航向与后续安全区路线一致。
        if not self._safe_zone_bbox_is_centered(zone):
            return self._decision(
                timestamp_ns,
                0.0,
                self._safe_zone_bbox_turn_command(zone, timestamp_ns),
                "safe_zone_center_full_bbox_before_keypoints",
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
        if self._safe_zone_keypoint_reverse_attempted:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoint_reverse_limit_reached",
                posture=posture,
            )
        # 关键点缺失时先固定不动等待新的感知帧；超时后由
        # ``_step_safe_zone_keypoint_scan`` 执行一次低速有界扫视。
        if not self._safe_zone_keypoint_scan_attempted:
            self._begin_safe_zone_keypoint_reobserve(timestamp_ns)
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_missing_start_reobserve",
                posture=posture,
            )
        return self._decision(
            timestamp_ns,
            0.0,
            self._safe_zone_bbox_turn_command(zone, timestamp_ns),
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

    def _begin_safe_zone_keypoint_reverse(self) -> None:
        """在 bbox 居中但关键点不足时开始一次有界的小距离倒车。"""

        self._safe_zone_phase = "reversing_for_safe_zone_keypoints"
        self._safe_zone_keypoint_reverse_base_distance_m = (
            self._latest_cumulative_distance_m
        )
        self._safe_zone_keypoint_reverse_attempted = True
        self._safe_zone_stop_since_ns = None

    def _step_safe_zone_keypoint_reverse(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """倒车观察安全区，关键点完整出现后立即停车。"""

        posture = (
            GripperPosture.OPEN
            if self._safe_zone_calibration_after_exit
            else GripperPosture.CLOSED
        )
        zone = self._own_safe_zone_observation()
        if (
            zone is not None
            and self._safe_zone_bbox_fully_visible(zone)
            and self._safe_zone_has_complete_ground_keypoints(zone)
        ):
            self._safe_zone_phase = "stopping_after_bbox_keypoints"
            self._safe_zone_keypoint_reverse_base_distance_m = None
            self._safe_zone_stop_since_ns = None
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_seen_stop_after_reverse",
                posture=posture,
            )

        current_distance = self._latest_cumulative_distance_m
        base_distance = self._safe_zone_keypoint_reverse_base_distance_m
        if current_distance is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoint_reverse_waiting_for_odometry",
                posture=posture,
            )
        if base_distance is None:
            self._safe_zone_keypoint_reverse_base_distance_m = current_distance
            base_distance = current_distance
        travelled_m = max(0.0, base_distance - current_distance)
        if (
            travelled_m
            >= self.config.safe_zone_keypoint_reverse_max_distance_m - 1e-9
        ):
            self._safe_zone_phase = "searching_safe_zone_keypoints"
            self._safe_zone_keypoint_reverse_base_distance_m = None
            self._reset_safe_zone_turn_gate()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoint_reverse_limit_reached",
                posture=posture,
            )
        return self._decision(
            timestamp_ns,
            -self.config.safe_zone_keypoint_reverse_speed_m_s,
            0.0,
            "safe_zone_reversing_for_keypoints",
            posture=posture,
        )

    def _begin_safe_zone_keypoint_reobserve(self, timestamp_ns: int) -> None:
        """在关键点短缺后开启一次固定时限的静止新帧等待。"""

        self._safe_zone_phase = "reobserving_safe_zone_keypoints"
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

    def _safe_zone_bbox_is_centered(
        self,
        zone: SafeZoneObservation | None,
    ) -> bool:
        """判断 bbox 中心是否落在追框死区内。"""

        perception = self._latest_perception
        if zone is None or perception is None or perception.field_features is None:
            return False
        image_width, _ = perception.field_features.image_size
        if image_width <= 0:
            return False
        center = 0.5 * (zone.box.x_min + zone.box.x_max)
        normalized = abs(center - image_width / 2.0) / (image_width / 2.0)
        threshold = (
            self.config.safe_zone_bbox_turn_resume_ratio
            if self._safe_zone_bbox_centered
            else self.config.safe_zone_bbox_turn_deadband_ratio
        )
        return normalized <= threshold

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
        """固定不动等待新帧；超时后最多进行一次低速扫描。"""

        deadline = self._safe_zone_key_reobserve_until_ns
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
            if (
                self._safe_zone_bbox_fully_visible(zone)
                and self._safe_zone_has_complete_ground_keypoints(zone)
            ):
                self._safe_zone_key_reobserve_until_ns = None
                self._safe_zone_key_reobserve_frame_floor = None
                self._safe_zone_phase = "stopping_after_bbox_keypoints"
                self._safe_zone_stop_since_ns = None
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_keypoints_reobserved_stop_before_calibration",
                    posture=(
                        GripperPosture.OPEN
                        if self._safe_zone_calibration_after_exit
                        else GripperPosture.CLOSED
                    ),
                )
        if timestamp_ns < deadline:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_reobserve_waiting_for_new_frame",
                posture=(
                    GripperPosture.OPEN
                    if self._safe_zone_calibration_after_exit
                    else GripperPosture.CLOSED
                ),
            )

        zone = self._own_safe_zone_observation()
        self._safe_zone_key_reobserve_until_ns = None
        self._safe_zone_key_reobserve_frame_floor = None
        self._safe_zone_key_samples = []
        self._safe_zone_key_last_frame = None
        self._safe_zone_calibration_snapshot = None
        self._safe_zone_calibration_zone = None
        if (
            self._safe_zone_bbox_is_centered(zone)
            and not self._safe_zone_keypoint_scan_attempted
        ):
            self._safe_zone_keypoint_scan_attempted = True
            self._safe_zone_phase = "scanning_safe_zone_keypoints"
            self._safe_zone_keypoint_scan_until_ns = timestamp_ns + round(
                self.config.safe_zone_keypoint_reobserve_timeout_s
                * 1_000_000_000
            )
            self._safe_zone_keypoint_scan_frame_floor = (
                None
                if self._latest_perception is None
                else self._latest_perception.frame_sequence
            )
            self._safe_zone_bbox_centered = False
            return self._step_safe_zone_keypoint_scan(timestamp_ns)
        self._safe_zone_phase = "searching_safe_zone_keypoints"
        self._reset_safe_zone_turn_gate()
        return self._step_safe_zone_bbox_key_search(timestamp_ns)

    def _step_safe_zone_keypoint_scan(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """沿固定方向低速扫视关键点，达到时限后重新静止观察。"""

        deadline = self._safe_zone_keypoint_scan_until_ns
        posture = (
            GripperPosture.OPEN
            if self._safe_zone_calibration_after_exit
            else GripperPosture.CLOSED
        )
        zone = self._own_safe_zone_observation()
        frame_floor = self._safe_zone_keypoint_scan_frame_floor
        is_new_frame = (
            self._latest_perception is not None
            and (
                frame_floor is None
                or self._latest_perception.frame_sequence > frame_floor
            )
        )
        if (
            is_new_frame
            and
            zone is not None
            and self._safe_zone_bbox_fully_visible(zone)
            and self._safe_zone_has_complete_ground_keypoints(zone)
        ):
            self._safe_zone_keypoint_scan_until_ns = None
            self._safe_zone_keypoint_scan_frame_floor = None
            self._safe_zone_phase = "stopping_after_bbox_keypoints"
            self._safe_zone_stop_since_ns = None
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoints_seen_stop_after_reobserve_scan",
                posture=posture,
            )
        if deadline is not None and timestamp_ns >= deadline:
            self._safe_zone_keypoint_scan_until_ns = None
            self._safe_zone_keypoint_scan_frame_floor = None
            self._begin_safe_zone_keypoint_reobserve(timestamp_ns)
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "safe_zone_keypoint_scan_complete_reobserve",
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
        localization = localizer.localize(
            averaged_features,
            prior_pose=FieldPose2D(position, heading),
            current_timestamp_ns=timestamp_ns,
        )
        calibration = localization.observation
        if calibration is None:
            # 直接落定位器的拒绝原因：观测年龄门和三点几何门必须可区分。
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
                self._safe_zone_key_reobserve_until_ns = None
                self._safe_zone_key_reobserve_frame_floor = None
                self._safe_zone_keypoint_scan_until_ns = None
                self._safe_zone_keypoint_scan_attempted = False
                self._reset_safe_zone_turn_gate()
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

    def _finish_post_exit_visual_calibration(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """纠偏成功后清空交付期目标，并等待一帧新的搜索输入。"""

        self._safe_zone_calibration_after_exit = False
        self._safe_zone_phase = "idle"
        self._safe_zone_key_reobserve_until_ns = None
        self._safe_zone_key_reobserve_frame_floor = None
        self._safe_zone_keypoint_scan_until_ns = None
        self._safe_zone_keypoint_scan_attempted = False
        self._safe_zone_keypoint_reverse_base_distance_m = None
        self._safe_zone_keypoint_reverse_attempted = False
        self._reset_safe_zone_turn_gate()
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
            self._safe_zone_key_reobserve_until_ns = None
            self._safe_zone_key_reobserve_frame_floor = None
            self._safe_zone_keypoint_scan_until_ns = None
            self._safe_zone_keypoint_scan_frame_floor = None
            self._safe_zone_keypoint_scan_attempted = False
            self._safe_zone_calibration_snapshot = None
            self._safe_zone_calibration_zone = None
            self._safe_zone_calibration_last_failure = None
            self._safe_zone_bbox_turn_direction = None
            self._reset_safe_zone_turn_gate()
            self._safe_zone_reacquire_frame_floor = None
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
            self._safe_zone_key_samples = []
            self._safe_zone_key_last_frame = None
            self._safe_zone_key_reobserve_until_ns = None
            self._safe_zone_key_reobserve_frame_floor = None
            self._safe_zone_keypoint_scan_until_ns = None
            self._safe_zone_keypoint_scan_frame_floor = None
            self._safe_zone_keypoint_reverse_base_distance_m = None
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
                self._begin_safe_zone_keypoint_reobserve(timestamp_ns)
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "safe_zone_keypoints_missing_start_reobserve",
                    posture=calibration_posture,
                )
            collected = self._collect_safe_zone_key_sample()
            if collected and len(self._safe_zone_key_samples) >= 5:
                if not self._lock_safe_zone_calibration_plan(timestamp_ns):
                    self._safe_zone_key_samples = []
                    self._safe_zone_key_last_frame = None
                    self._safe_zone_key_reobserve_until_ns = None
                    self._safe_zone_key_reobserve_frame_floor = None
                    self._safe_zone_keypoint_scan_until_ns = None
                    self._safe_zone_keypoint_scan_frame_floor = None
                    self._safe_zone_keypoint_scan_attempted = False
                    self._safe_zone_keypoint_reverse_base_distance_m = None
                    self._safe_zone_keypoint_reverse_attempted = False
                    self._safe_zone_calibration_snapshot = None
                    self._safe_zone_calibration_zone = None
                    self._safe_zone_phase = "searching_safe_zone_keypoints"
                    self._safe_zone_bbox_turn_direction = None
                    self._reset_safe_zone_turn_gate()
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

    # ------------------------------------------------------------------
    # strategy 专用流程
    # ------------------------------------------------------------------

    def _strategy_startup_turn_velocity(self) -> float:
        """返回当前冲刺腿的转向速度；第一腿右转，第二腿左转。"""

        magnitude = abs(self.config.startup_turn_angular_velocity_rad_s)
        return -magnitude if self._strategy_startup_leg == 1 else magnitude

    def _step_startup_turn(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
    ) -> MatchDecision:
        """执行 strategy 的两次等角度启动转向。"""

        angular_velocity = self._strategy_startup_turn_velocity()
        side = "right" if angular_velocity < 0.0 else "left"
        if heading_rad is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"strategy_startup_{side}_turn_waiting_for_heading",
            )
        previous = self._startup_turn_last_heading
        self._startup_turn_last_heading = heading_rad
        if previous is None:
            return self._decision(
                timestamp_ns,
                0.0,
                angular_velocity,
                f"strategy_startup_{side}_turn",
            )
        self._startup_turn_progress_rad += self._directional_delta(
            previous,
            heading_rad,
            angular_velocity,
        )
        if self._startup_turn_progress_rad >= self.config.startup_turn_angle_rad - 1e-9:
            self.state = MatchState.STARTUP_TURN_SETTLE
            self._settle_until_ns = timestamp_ns + self._seconds_to_ns(
                self.config.startup_turn_settle_time_s
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"strategy_startup_{side}_turn_complete_wait",
            )
        return self._decision(
            timestamp_ns,
            0.0,
            angular_velocity,
            f"strategy_startup_{side}_turn",
        )

    def _step_startup_forward(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None,
        right_speed_feedback_m_s: float | None,
    ) -> MatchDecision:
        """执行第一腿 150 cm 短冲或第二腿 1 m 前冲。"""

        leg = self._strategy_startup_leg
        distance_m = (
            self.config.startup_forward_distance_m
            if leg == 1
            else self.config.cluster_relocate_distance_m
        )
        speed_m_s = (
            self.config.startup_forward_speed_m_s
            if leg == 1
            else self.config.cluster_relocate_speed_m_s
        )
        if cumulative_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"strategy_startup_forward_leg_{leg}_waiting_for_odometry",
            )
        if self._startup_forward_base_distance_m is None:
            self._startup_forward_base_distance_m = cumulative_distance_m
        travelled = cumulative_distance_m - self._startup_forward_base_distance_m
        if travelled >= distance_m - 1e-9:
            self.state = MatchState.STARTUP_FORWARD_SETTLE
            self._settle_until_ns = timestamp_ns + self._seconds_to_ns(
                self.config.startup_forward_settle_time_s
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"strategy_startup_forward_leg_{leg}_complete_wait",
            )
        return self._decision(
            timestamp_ns,
            speed_m_s,
            self._straight_pid_output(
                timestamp_ns,
                left_speed_feedback_m_s,
                right_speed_feedback_m_s,
            ),
            f"strategy_startup_forward_leg_{leg}",
        )

    def _step_strategy_startup_forward_settle(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """在两腿启动冲刺之间切换，第二腿结束后进入蓝色目标搜索。"""

        if timestamp_ns < self._settle_until_ns:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "strategy_startup_waiting_after_forward",
            )
        if self._strategy_startup_leg == 1:
            self._strategy_startup_leg = 2
            self._startup_turn_last_heading = None
            self._startup_turn_progress_rad = 0.0
            self._startup_forward_base_distance_m = None
            self._reset_straight_pid()
            self.state = MatchState.STARTUP_TURN_RIGHT
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "strategy_startup_short_forward_settled_start_left_turn",
            )
        self._begin_cluster_search()
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(
            timestamp_ns,
            0.0,
            self._cluster_search_angular_velocity_rad_s,
            "strategy_startup_long_forward_settled_start_blue_search",
        )

    @staticmethod
    def _strategy_blue_target_is_usable(target: TrackedTarget) -> bool:
        """蓝色危险物块的直接夹取/解团接触证据门禁。"""

        if (
            target.target_class is not TargetClass.BLUE_DANGER
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

    def _strategy_blue_target(self, timestamp_ns: int) -> TrackedTarget | None:
        """按当前机器人坐标返回最近的可靠蓝色物块。"""

        candidates = tuple(
            target
            for target in self._tracker.tracks
            if self._target_is_fresh(target, timestamp_ns)
            and self._strategy_blue_target_is_usable(target)
            and target.ground_point is not None
            and target.ground_point.x > 0.0
        )
        return min(
            candidates,
            key=lambda target: (
                math.hypot(target.ground_point.x, target.ground_point.y),
                abs(target.ground_point.y),
                target.track_id,
            ),
            default=None,
        )

    def _strategy_blue_current_target(
        self,
        timestamp_ns: int,
        *,
        expected_range_mm: float | None = None,
    ) -> TrackedTarget | None:
        """优先保留锁定 ID；跟踪器换 ID 时用同帧蓝块恢复身份。"""

        selected = self._selected_target()
        if (
            selected is not None
            and self._target_is_fresh(selected, timestamp_ns)
            and self._strategy_blue_target_is_usable(selected)
        ):
            return selected
        candidates = tuple(
            target
            for target in self._tracker.tracks
            if self._target_is_fresh(target, timestamp_ns)
            and self._strategy_blue_target_is_usable(target)
            and target.ground_point is not None
            and target.ground_point.x > 0.0
        )
        return min(
            candidates,
            key=lambda target: (
                (
                    abs(
                        math.hypot(
                            target.ground_point.x,
                            target.ground_point.y,
                        )
                        - expected_range_mm
                    )
                    if expected_range_mm is not None
                    else 0.0
                ),
                abs(target.ground_point.y),
                math.hypot(target.ground_point.x, target.ground_point.y),
                target.track_id,
            ),
            default=None,
        )

    def _strategy_blue_path_clear(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        reference: GroundPoint | None = None,
        ignored_track_ids: frozenset[int] = frozenset(),
        tracks: tuple[TrackedTarget, ...] | None = None,
    ) -> bool:
        """检查蓝色物块的直接接近走廊和同团门禁。"""

        point = target.ground_point if reference is None else reference
        if (
            not self._strategy_blue_target_is_usable(target)
            or point is None
            or point.x <= 0.0
            or math.hypot(point.x, point.y) <= 1e-6
            or self.estimated_field_position is None
            or self._latest_heading_rad is None
            or self._candidate_path_blocked(point, breakup=False)
        ):
            return False
        target_range_mm = math.hypot(point.x, point.y)
        lateral_half_width_mm = self._transport_corridor_half_width_mm
        source_tracks = self._tracker.tracks if tracks is None else tuple(tracks)
        for other in source_tracks:
            if (
                other.track_id == target.track_id
                or other.track_id in ignored_track_ids
                or other.track_id in self._green_preclose_consumed_track_ids
                or not self._target_is_fresh(other, timestamp_ns)
            ):
                continue
            if other.ground_point is None:
                # 当前仍有效但没有 K0 的目标不能证明直接夹取走廊为空。
                return False
            if (
                other.target_class is TargetClass.BLUE_DANGER
                and math.hypot(
                    other.ground_point.x - point.x,
                    other.ground_point.y - point.y,
                )
                <= self.config.cluster_group_ground_mm
            ):
                return False
            relative = self._target_aligned_coordinates(point, other.ground_point)
            if relative is None:
                return False
            forward_mm, lateral_mm = relative
            if (
                0.0 <= forward_mm < target_range_mm
                and abs(lateral_mm) <= lateral_half_width_mm
            ):
                return False
        return True

    def _step_strategy_search_cluster(
        self,
        timestamp_ns: int,
        heading_rad: float | None,
    ) -> MatchDecision:
        """只搜索最近蓝色物块：可直夹则直夹，否则进入动态解团。"""

        if self._path_recovery:
            if not self._safe_zone_vehicle_stopped(timestamp_ns):
                self._path_recovery_stopped_ns = None
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "strategy_search_waiting_for_recovery_stop",
                    posture=self._path_recovery_posture,
                )
            if self._path_recovery_stopped_ns is None:
                self._path_recovery_stopped_ns = timestamp_ns
                self._reset_tracker_for_new_preview_epoch()
                self._last_tracker_frame_sequence = None
            perception = self._latest_perception
            if (
                perception is None
                or not self._fresh_perception(perception, timestamp_ns)
                or perception.capture_timestamp_ns <= self._path_recovery_stopped_ns
            ):
                self._reset_tracker_for_new_preview_epoch()
                self._last_tracker_frame_sequence = None
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "strategy_search_waiting_for_recovery_new_frame",
                    posture=self._path_recovery_posture,
                )
            self._path_recovery = False
            self._path_recovery_stopped_ns = None
            self._safe_zone_stop_since_ns = None

        self._cluster_search_angular_velocity_rad_s = math.copysign(
            abs(self.config.cluster_search_empty_angular_velocity_rad_s),
            self._cluster_search_angular_velocity_rad_s,
        )
        target = self._strategy_blue_target(timestamp_ns)
        if target is not None and self._strategy_blue_path_clear(target, timestamp_ns):
            return self._begin_strategy_blue_transport(timestamp_ns, target)
        if target is not None:
            plan = self._choose_breakup_plan(
                timestamp_ns,
                approach=False,
                required_aim_id=target.track_id,
            )
            if plan is not None:
                self._start_breakup_attempt(timestamp_ns, plan)
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    f"strategy_blue_group_seen_start_breakup:{plan.aim_id}",
                    soft_brake=True,
                )

        committed = _SharedMatchSequence._advance_cluster_search_sweep(
            self,
            timestamp_ns,
            heading_rad,
        )
        if committed is not None:
            return committed
        direction = "right" if self._cluster_search_angular_velocity_rad_s < 0.0 else "left"
        return self._decision(
            timestamp_ns,
            0.0,
            self._cluster_search_angular_velocity_rad_s,
            f"strategy_search_blue_{direction}",
        )

    def _begin_strategy_blue_transport(
        self,
        timestamp_ns: int,
        target: TrackedTarget,
    ) -> MatchDecision:
        """锁定一个蓝色物块，复用现有相对对准和安全区运输状态。"""

        if not self._strategy_blue_target_is_usable(target):
            raise ValueError("strategy blue transport requires a usable blue target.")
        pickup = self._strategy_saved_near_field_pickup
        if pickup is None:
            raise RuntimeError(
                "Strategy blue grasp requires the match pickup sequence."
            )
        self._near_field_pickup = pickup
        pickup.reset()
        self._strategy_blue_last_preparation = None
        self._strategy_blue_preparation_key = None
        self._strategy_blue_confirmation_count = 0
        self._strategy_blue_confirmation_last_frame = None
        self._strategy_blue_confirmation_ids = None
        self._strategy_blue_transport = True
        self._strategy_blue_grasp_phase = "approach_handoff"
        self._strategy_delivery_slot = self._transport_count
        self._selected_track_id = target.track_id
        self._cluster_selected_track_ids = ()
        self._breakup_only = False
        self._selected_green_ground = target.ground_point
        self._opportunistic_single_green = False
        self._near_field_group_preview = False
        self._green_realign_pending = False
        self._green_realign_done = False
        self._green_preclose_consumed_track_ids.clear()
        self._green_preclose_carried_count = 0
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
            f"strategy_blue_direct_grasp_selected:{target.track_id}",
            posture=GripperPosture.TRANSPORT,
        )

    def _strategy_blue_current_observation(
        self,
        target: TrackedTarget,
    ):
        """从当前 perception 中取出与 tracker 目标对应的原始观测。"""

        perception = self._latest_perception
        if (
            perception is None
            or target.last_seen_timestamp_ns != perception.capture_timestamp_ns
            or target.frame_sequence != perception.frame_sequence
        ):
            return None
        return next(
            (
                observation
                for observation in perception.observations
                if observation.target_class is TargetClass.BLUE_DANGER
                and observation.box == target.box
                and observation.k0 == target.k0
            ),
            None,
        )

    def _strategy_blue_grasp_preparation(
        self,
        timestamp_ns: int,
    ) -> GraspPreparation | None:
        """用绿色近场几何/状态机为一个蓝块生成同构执行计划。"""

        selector = self._strategy_blue_selector
        projector = self._strategy_blue_projector
        pickup = self._near_field_pickup
        if selector is None or projector is None or pickup is None:
            return None
        target = self._strategy_blue_current_target(
            timestamp_ns,
            expected_range_mm=self._near_field_handoff_range_mm(),
        )
        if target is None or target.ground_point is None:
            return self._strategy_blue_last_preparation
        observation = self._strategy_blue_current_observation(target)
        if observation is None:
            return self._strategy_blue_last_preparation
        preparation_key = (
            target.track_id,
            observation.frame_sequence,
            observation.capture_timestamp_ns,
        )
        if preparation_key == self._strategy_blue_preparation_key:
            return self._strategy_blue_last_preparation
        self._strategy_blue_preparation_key = preparation_key
        try:
            envelope = measure_target_envelope(
                observation,
                projector,
                min_mask_pixels=selector.config.min_mask_pixels,
            )
            if envelope is None:
                return self._strategy_blue_last_preparation
            member = GraspTarget(
                target.track_id,
                observation,
                envelope,
                True,
                True,
                observed=True,
            )
            geometry = selector._group_geometry(
                (member,),
                alignment_tolerance_mm=selector.config.center_tolerance_mm,
                range_limit_mm=selector.config.max_range_mm,
            )
        except (TypeError, ValueError):
            return self._strategy_blue_last_preparation
        if geometry.reasons:
            return self._strategy_blue_last_preparation

        bounds = (
            GroundPoint(geometry.x0_mm, geometry.y0_mm),
            GroundPoint(geometry.x1_mm, geometry.y0_mm),
            GroundPoint(geometry.x1_mm, geometry.y1_mm),
            GroundPoint(geometry.x0_mm, geometry.y1_mm),
        )
        plan = NearFieldGraspPlan(
            observation.frame_sequence,
            observation.capture_timestamp_ns,
            (member,),
            geometry.center,
            geometry.angle_rad,
            bounds,
            geometry.opening_width_mm,
            selector.maximum_opening_mm,
            geometry.opening_servo_angles_deg,
            geometry.forward_distance_mm,
            0.0,
            GraspScore(1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            selector._regions(
                geometry.angle_rad,
                geometry.forward_distance_mm,
                geometry.left_tip_y_mm,
                geometry.right_tip_y_mm,
            ),
        )
        member_ids = plan.member_ids
        if self._strategy_blue_confirmation_ids != member_ids:
            self._strategy_blue_confirmation_ids = member_ids
            self._strategy_blue_confirmation_count = 0
            self._strategy_blue_confirmation_last_frame = None
        if geometry.angle_rad != 0.0:
            self._strategy_blue_confirmation_count = 0
            self._strategy_blue_confirmation_last_frame = None
        elif (
            self._strategy_blue_confirmation_last_frame
            != observation.frame_sequence
        ):
            self._strategy_blue_confirmation_last_frame = (
                observation.frame_sequence
            )
            self._strategy_blue_confirmation_count = min(
                selector.config.confirmation_frames,
                self._strategy_blue_confirmation_count + 1,
            )
        required = selector.config.confirmation_frames
        preparation = GraspPreparation(
            observation.capture_timestamp_ns,
            GraspSelection(plan, ()),
            (member,),
            ready=self._strategy_blue_confirmation_count >= required,
            checked_member_ids=pickup.locked_ids,
            session_id=self._near_field_session_id,
            confirmation_count=self._strategy_blue_confirmation_count,
            confirmation_required=required,
            prepared_timestamp_ns=timestamp_ns,
            result_timestamp_ns=observation.result_timestamp_ns,
        )
        self._strategy_blue_last_preparation = preparation
        return preparation

    def _begin_near_field_grasp(
        self,
        timestamp_ns: int,
        *,
        handoff_prior: NearFieldHandoffPrior | None = None,
    ) -> MatchDecision:
        if self._strategy_formal_phase:
            return _SharedMatchSequence._begin_near_field_grasp(
                self,
                timestamp_ns,
                handoff_prior=handoff_prior,
            )
        del handoff_prior
        if not self._strategy_blue_transport:
            return _SharedMatchSequence._begin_near_field_grasp(
                self,
                timestamp_ns,
                handoff_prior=None,
            )
        pickup = self._near_field_pickup
        if pickup is None:
            raise RuntimeError("Strategy blue grasp pickup sequence is missing.")
        self._near_field_session_id += 1
        pickup.reset()
        self._near_field_handoff_prior = None
        self._near_field_last_failure_diagnostic = None
        self._near_field_route = GraspRoute.DECIDING
        self._near_field_confirmation_started_ns = None
        self._near_field_route_rejections = ()
        self._near_field_route_elapsed_ms = 0.0
        self._near_field_route_candidate_count = 0
        self._strategy_blue_grasp_phase = "near_field"
        self._strategy_blue_last_preparation = None
        self._strategy_blue_preparation_key = None
        self._strategy_blue_confirmation_count = 0
        self._strategy_blue_confirmation_last_frame = None
        self._strategy_blue_confirmation_ids = None
        self.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
        self._begin_action_settle(timestamp_ns, "near_field_grasp")
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "strategy_blue_near_field_handoff_started",
            posture=GripperPosture.CLOSED,
            soft_brake=True,
        )

    def _step_near_field_grasp(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
        preparation: GraspPreparation | None,
        path_clear: bool | None,
    ) -> MatchDecision:
        if self._strategy_formal_phase:
            return _SharedMatchSequence._step_near_field_grasp(
                self,
                timestamp_ns,
                cumulative_distance_m,
                preparation,
                path_clear,
            )
        if not self._strategy_blue_transport:
            return _SharedMatchSequence._step_near_field_grasp(
                self,
                timestamp_ns,
                cumulative_distance_m,
                preparation,
                path_clear,
            )
        del preparation, path_clear
        current_preparation = self._strategy_blue_grasp_preparation(timestamp_ns)
        plan = (
            None
            if current_preparation is None
            else current_preparation.selection.plan
        )
        current_path_clear: bool | None = None
        if plan is not None and plan.alignment_angle_rad == 0.0:
            current_path_clear = self._near_field_path_clear(
                plan,
                self._latest_heading_rad,
            )
        return _SharedMatchSequence._step_near_field_grasp(
            self,
            timestamp_ns,
            cumulative_distance_m,
            current_preparation,
            current_path_clear,
        )

    @property
    def near_field_enabled(self) -> bool:
        # 蓝色近场计划在控制线程内直接生成，避免共用 worker 按正式规则
        # 把 blue_danger 当作不可抓目标；恢复正式 match 后重新启用 worker。
        if not self._strategy_formal_phase:
            return False
        return self._near_field_pickup is not None

    def _graspable_target_is_usable(self, target: TrackedTarget) -> bool:
        """策略入口允许蓝色危险物块作为单目标接触证据。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence._graspable_target_is_usable(target)
        if target.target_class is TargetClass.BLUE_DANGER:
            return MatchSequence._strategy_blue_target_is_usable(target)
        return False

    def _selected_green_is_usable(self, target: TrackedTarget) -> bool:
        if self._strategy_formal_phase:
            return _SharedMatchSequence._selected_green_is_usable(self, target)
        if not self._strategy_blue_transport:
            return False
        return self._strategy_blue_target_is_usable(target)

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
        if self._strategy_formal_phase:
            return _SharedMatchSequence._green_path_is_clear_for_point(
                self,
                target,
                point,
                timestamp_ns,
                ignored_track_ids=ignored_track_ids,
                allow_behind=allow_behind,
                lateral_half_width_mm=lateral_half_width_mm,
                tracks=tracks,
            )
        del allow_behind, lateral_half_width_mm
        if target.target_class is not TargetClass.BLUE_DANGER:
            return False
        return self._strategy_blue_path_clear(
            target,
            timestamp_ns,
            reference=point,
            ignored_track_ids=ignored_track_ids,
            tracks=tracks,
        )

    def _step_align_green(self, timestamp_ns: int) -> MatchDecision:
        """蓝色运输沿用真机验证过的历史对准路径，其余阶段走共享实现。

        共享的 ``_step_align_green`` 在 ``_near_field_pickup`` 不为 None 时会
        改走 ``_step_formal_green_align``，而 ``_begin_strategy_blue_transport``
        恰好会重新暴露该序列；不在这里按相位区分就会静默改变蓝色运输的对准
        行为，那是 2026-09-11 真机跑通过的路径。
        """

        if self._strategy_blue_transport:
            return _SharedMatchSequence._align_green_legacy(self, timestamp_ns)
        return _SharedMatchSequence._step_align_green(self, timestamp_ns)

    def _step_approach_green(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        """按 match 的远场接近逻辑到近场，再交给同一套夹取状态机。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence._step_approach_green(
                self,
                timestamp_ns,
                cumulative_distance_m,
            )
        settling = self._consume_action_settle(
            timestamp_ns,
            "green_before_approach",
            posture=GripperPosture.TRANSPORT,
            reason="strategy_blue_waiting_after_alignment",
        )
        if settling is not None:
            return settling
        if cumulative_distance_m is None or self._green_approach_distance_m is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "strategy_blue_approach_waiting_for_odometry",
                posture=GripperPosture.TRANSPORT,
            )
        target = self._strategy_blue_current_target(
            timestamp_ns,
            expected_range_mm=self._near_field_handoff_range_mm(),
        )
        if target is not None:
            self._selected_track_id = target.track_id
        target_point = None if target is None else target.ground_point
        if target is None or target_point is None:
            if self._green_align_lost_since_ns is None:
                self._green_align_lost_since_ns = timestamp_ns
            if timestamp_ns - self._green_align_lost_since_ns < self._green_align_hold_ns:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "strategy_blue_approach_waiting_for_current_ground_point",
                    posture=GripperPosture.TRANSPORT,
                )
            self._strategy_blue_transport = False
            self._strategy_blue_grasp_phase = "idle"
            self._selected_track_id = None
            self._begin_cluster_search()
            self.state = MatchState.SEARCH_CLUSTER
            return self._decision(
                timestamp_ns,
                0.0,
                self._cluster_search_angular_velocity_rad_s,
                "strategy_blue_target_lost_restart_search",
                posture=GripperPosture.TRANSPORT,
            )
        self._green_align_lost_since_ns = None
        if self._green_approach_base_distance_m is None:
            self._green_approach_base_distance_m = cumulative_distance_m
        travelled = cumulative_distance_m - self._green_approach_base_distance_m
        if travelled >= self._green_approach_distance_m - 1e-9:
            return self._begin_near_field_grasp(timestamp_ns)
        angular = self._heading_hold_angular_velocity(
            self._green_reference_heading_rad,
            kp_rad_s=self.config.green_alignment_kp_rad_s,
            max_angular_velocity_rad_s=(
                self.config.green_alignment_max_angular_velocity_rad_s
            ),
            tolerance_rad=math.atan2(
                self.config.green_alignment_tolerance_mm,
                max(
                    math.hypot(
                        self._green_reference.x,
                        self._green_reference.y,
                    )
                    if self._green_reference is not None
                    else self._near_field_handoff_range_mm(),
                    1e-6,
                ),
            ),
        )
        if angular is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "strategy_blue_approach_waiting_for_heading",
                posture=GripperPosture.TRANSPORT,
            )
        return self._decision(
            timestamp_ns,
            self.config.green_approach_speed_m_s,
            angular,
            "strategy_blue_approach_to_near_field_handoff",
            posture=GripperPosture.TRANSPORT,
        )
    def _step_transport_close(self, timestamp_ns: int) -> MatchDecision:
        if self._strategy_formal_phase:
            return _SharedMatchSequence._step_transport_close(self, timestamp_ns)
        if self._strategy_blue_transport:
            if not self._gripper_action_completed(timestamp_ns):
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "strategy_blue_closing_gripper",
                    posture=GripperPosture.CLOSED,
                )
            self._strategy_blue_transport = False
            self._transport_target_classes = (TargetClass.BLUE_DANGER,)
            self._strategy_delivery_slot = self._transport_count
            return self._start_safe_zone_transport(
                timestamp_ns,
                transport_opened=False,
                posture=GripperPosture.CLOSED,
                reason="strategy_blue_gripper_closed_start_d2_transport",
            )
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "strategy_close_without_blue_target",
            posture=GripperPosture.CLOSED,
        )

    def _choose_breakup_plan(
        self,
        timestamp_ns: int,
        *,
        approach: bool = False,
        required_ids: frozenset[int] | None = None,
        required_aim_id: int | None = None,
    ) -> BreakupPlan | None:
        """为蓝色目标选择可接触的动态解团计划；全蓝团也在此路径。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence._choose_breakup_plan(
                self,
                timestamp_ns,
                approach=approach,
                required_ids=required_ids,
                required_aim_id=required_aim_id,
            )
        origin = self.estimated_field_position
        heading = self._latest_heading_rad
        if (
            origin is None
            or heading is None
            or self._breakup_static_map is None
            or self._breakup_target_geometry is None
        ):
            self._last_cluster_rejection_reason = (
                "strategy_blue_breakup_missing_pose_map_or_geometry"
            )
            return None
        targets = self._breakup_targets(timestamp_ns)
        blue_ids = {
            target.track_id
            for target in targets
            if target.target_class is TargetClass.BLUE_DANGER
        }
        if required_aim_id is not None and required_aim_id not in blue_ids:
            self._last_cluster_rejection_reason = "strategy_blue_aim_not_current"
            return None
        # 已放入安全区的物资不再作为成组或瞄准点候选；它们仍留在 targets 中
        # 参加推移净空检查，不能被碰撞也不能被忽略。
        non_contact_ids = frozenset(
            target.track_id
            for target in targets
            if self._ground_in_safe_zone(target.center, target.capture_timestamp_ns)
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
        breakup_args = dict(
            config=self.config,
            origin=origin,
            heading_rad=heading,
            static_map=self._breakup_static_map,
            field_bounds=self._breakup_allowed_field_bounds(),
            front_mm=GripperKinematics().left_tip_position(0).x,
            allowed_classes=frozenset((TargetClass.BLUE_DANGER,)),
            approach=approach,
            required_ids=required_ids,
            rejection_reasons=self._near_field_route_rejections,
            priority_ids=frozenset(priority_ids),
            required_aim_id=required_aim_id,
            non_contact_ids=non_contact_ids,
        )
        rejections: list[str] = []
        candidates = plan_breakup(targets, rejections=rejections, **breakup_args)
        tried_groups: set[tuple[int, ...]] = set()
        for candidate in candidates:
            if candidate.aim_id not in blue_ids:
                continue
            if self._aim_in_failed_aims(candidate.aim_field):
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
                )
                <= self.config.cluster_group_ground_mm
                and self._breakup_attempt_is_current(index)
            ]
            if len(history) >= self.config.breakup_max_attempts:
                continue
            if not history:
                return candidate
            previous = history[-1]
            retries = plan_breakup(
                targets,
                **{
                    **breakup_args,
                    "required_ids": frozenset(candidate.member_ids),
                },
                attempt=len(history) + 1,
                previous_aim=previous.aim_field,
                previous_penetration_mm=previous.penetration_mm,
                rejections=rejections,
            )
            for retry in retries:
                if retry.aim_id in blue_ids:
                    self._breakup_plan_rejections = tuple(rejections[-8:])
                    return retry
        self._breakup_plan_rejections = tuple(rejections[-8:])
        self._last_cluster_rejection_reason = "strategy_blue_breakup_no_safe_plan"
        return None

    def _breakup_geometric_block(self, timestamp_ns: int) -> bool:
        if self._strategy_formal_phase:
            return _SharedMatchSequence._breakup_geometric_block(
                self,
                timestamp_ns,
            )
        target = self._strategy_blue_target(timestamp_ns)
        return target is not None and not self._strategy_blue_path_clear(
            target,
            timestamp_ns,
        )

    def _try_dynamic_grasp(self, timestamp_ns: int) -> MatchDecision | None:
        if self._strategy_formal_phase:
            return _SharedMatchSequence._try_dynamic_grasp(self, timestamp_ns)
        # 解团合爪后的复核阶段如果已露出可直接夹取的蓝色物块，立即转入
        # 单目标路径；不把绿/黑/橙策略带入本入口。
        if self.state is MatchState.CLOSE_GRIPPER_SPIN:
            target = self._strategy_blue_target(timestamp_ns)
            if target is not None and self._strategy_blue_path_clear(target, timestamp_ns):
                return self._begin_strategy_blue_transport(timestamp_ns, target)
        return None

    def green_target_diagnostic(self, timestamp_ns: int) -> str:
        """记录策略蓝块的当前 K0、距离和夹取阶段，便于复盘实车失败。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence.green_target_diagnostic(
                self,
                timestamp_ns,
            )
        targets = tuple(
            target
            for target in self._tracker.tracks
            if target.target_class is TargetClass.BLUE_DANGER
        )
        if not targets:
            return f"blue=none,grasp_phase={self._strategy_blue_grasp_phase}"
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
        target_final_x_mm = (
            135.0
            if self._near_field_grasp_config is None
            else self._near_field_grasp_config.target_final_x_mm
        )
        active_plan = (
            None
            if self._near_field_pickup is None
            else self._near_field_pickup.active_plan
        )
        forward_distance_mm = (
            None if active_plan is None else active_plan.forward_distance_mm
        )
        forward_text = (
            "none"
            if forward_distance_mm is None
            else f"{forward_distance_mm:.1f}"
        )
        return (
            ";".join(entries)
            + f",grasp_phase={self._strategy_blue_grasp_phase}"
            + f",handoff_range_mm={self._near_field_handoff_range_mm():.1f}"
            + f",target_final_x_mm={target_final_x_mm:.1f}"
            + f",active_plan_opening_angles={None if active_plan is None else active_plan.opening_servo_angles_deg}"
            + f",planned_forward_mm={forward_text}"
        )

    def _own_safe_zone_observation(self):
        """策略将目标安全区设为启动侧的对面安全区。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence._own_safe_zone_observation(self)
        perception = self._latest_perception
        if (
            perception is None
            or perception.dropped_stale_age_ms is not None
            or perception.field_features is None
        ):
            return None
        expected = self._strategy_safe_zone_color
        if expected is None:
            expected = (
                SafeZoneColor.BLUE
                if self._team_color is TeamColor.RED
                else SafeZoneColor.RED
            )
        candidates = tuple(
            zone
            for zone in perception.field_features.safe_zones
            if zone.physical_color is expected
        )
        return max(candidates, key=lambda zone: zone.confidence, default=None)

    def _safe_zone_transport_endpoint(self) -> FieldPoint:
        """第一趟使用对面安全区左侧 D2，第二趟使用右侧 D2。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence._safe_zone_transport_endpoint(self)
        if self._strategy_delivery_slot == 0:
            return self.config.safe_zone_fallback_target_field
        return self.config.safe_zone_injured_target_field

    def _safe_zone_d2_target(self) -> FieldPoint:
        """策略配置的左右航点就是最终 D2 对齐点，不再向内偏移。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence._safe_zone_d2_target(self)
        return self._safe_zone_braking_compensated_target(
            self._safe_zone_transport_endpoint()
        )

    def _safe_zone_final_target_y_mm(self) -> float:
        """策略在 D2 完成释放，不再向安全区深处二次推进。"""

        if self._strategy_formal_phase:
            return _SharedMatchSequence._safe_zone_final_target_y_mm(self)
        return self._safe_zone_d2_target().y

    def _begin_formal_match_after_strategy(
        self,
        timestamp_ns: int,
    ) -> MatchDecision:
        """蓝色两趟完成后恢复 match 的绿块首趟流程。"""

        own_y_sign = -1.0 if self._team_color is TeamColor.BLUE else 1.0
        fallback = self.config.safe_zone_fallback_target_field
        injured = self.config.safe_zone_injured_target_field
        self.config = replace(
            self.config,
            # strategy 配置的两个航点位于对面；后续 match 流程改回己方
            # 安全区，并重新打开正式流程的机会抓取入口。
            safe_zone_fallback_target_field=FieldPoint(
                fallback.x,
                own_y_sign * abs(fallback.y),
            ),
            safe_zone_injured_target_field=FieldPoint(
                injured.x,
                own_y_sign * abs(injured.y),
            ),
            opportunistic_single_green_enabled=True,
        )
        self._strategy_formal_phase = True
        self._near_field_pickup = self._strategy_saved_near_field_pickup
        self._strategy_safe_zone_color = None
        self._safe_zone_forward_y_sign = own_y_sign
        self._strategy_blue_transport = False
        self._strategy_blue_grasp_phase = "idle"
        self._strategy_delivery_slot = 0
        self._transport_count = 0
        self._transport_target_classes = ()
        self._breakup_plan = None
        self._breakup_proposal = None
        self._breakup_failed_aims.clear()
        self._breakup_failed_aim_positions.clear()
        self._breakup_attempts.clear()
        self._breakup_attempt_positions.clear()
        self._breakup_rejected_grasp_ids.clear()
        self._breakup_only = False
        self._selected_track_id = None
        self._selected_green_ground = None
        # 阶段翻转是真正的会话边界：策略阶段累积的转过角度不能带进正式阶段，
        # 否则正式搜索一开场就接近整圈预算。
        self._reset_rotation_budget()
        # 蓝色阶段的选中目标不能残留成正式绿块对准的场地锚点。
        self._green_target_field_point = None
        self._green_reference_samples = []
        self._green_reference_last_seen_ns = None
        self._green_reference = None
        self._green_reference_heading_rad = None
        self._green_reference_distance_m = None
        self._green_approach_base_distance_m = None
        self._green_approach_distance_m = None
        self._reset_green_alignment_gate()

        # 复用正式 match 的退出后视觉纠偏，再用纠偏后的新帧进入绿色搜索。
        self._safe_zone_calibration_after_exit = True
        self._safe_zone_phase = "searching_safe_zone_keypoints"
        self._safe_zone_key_samples = []
        self._safe_zone_key_last_frame = None
        self._safe_zone_key_reobserve_until_ns = None
        self._safe_zone_key_reobserve_frame_floor = None
        self._safe_zone_keypoint_scan_until_ns = None
        self._safe_zone_keypoint_scan_frame_floor = None
        self._safe_zone_keypoint_scan_attempted = False
        self._safe_zone_keypoint_reverse_base_distance_m = None
        self._safe_zone_keypoint_reverse_attempted = False
        self._safe_zone_calibration_snapshot = None
        self._safe_zone_calibration_zone = None
        self._safe_zone_calibration_last_failure = None
        self._safe_zone_bbox_turn_direction = None
        self._safe_zone_reacquire_frame_floor = None
        self._reset_safe_zone_turn_gate()
        self._safe_zone_stop_since_ns = None
        self._return_phase = "calibrating_after_exit"
        self.state = MatchState.TRANSPORT_RELEASE
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "strategy_blue_tasks_complete_start_match_green_search",
            posture=GripperPosture.OPEN,
        )

    def _step_return_backup(
        self,
        timestamp_ns: int,
        cumulative_distance_m: float | None,
    ) -> MatchDecision:
        if self._strategy_formal_phase:
            return _SharedMatchSequence._step_return_backup(
                self,
                timestamp_ns,
                cumulative_distance_m,
            )
        decision = _SharedMatchSequence._step_return_backup(
            self,
            timestamp_ns,
            cumulative_distance_m,
        )
        if decision.state is MatchState.FINISH_STOP:
            return self._begin_formal_match_after_strategy(timestamp_ns)
        return decision


def main() -> None:
    from rescue_vision.app.match_runtime import _run_hardware

    parser = argparse.ArgumentParser(
        description="Run the blue-danger transport strategy."
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
        sequence_factory=MatchSequence.from_app_config,
        mode_name="match_strategy",
        log_file_prefix="strategy_",
        preview_title="Match strategy perception",
    )


if __name__ == "__main__":
    main()
