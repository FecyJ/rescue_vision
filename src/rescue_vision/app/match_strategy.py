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
    FieldFeatureDetectionResult,
    PerceptionSnapshot,
    SafeZoneColor,
    SafeZoneObservation,
    TargetClass,
    UndistortedBoundingBox,
)
from rescue_vision.perception.gripper_width import measure_target_envelope
from rescue_vision.perception.gripper_color import GripperColorConfig
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
        gripper_color_config: GripperColorConfig | None = None,
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
            gripper_color_config=gripper_color_config,
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
            terminal_speed_gain_s_inv=runtime.pickup_terminal_speed_gain_s_inv,
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
            gripper_color_config=config.perception.gripper_color,
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

        if self._strategy_formal_phase:
            return _SharedMatchSequence._step_actions(
                self, timestamp_ns, perception=perception, heading_rad=heading_rad,
                cumulative_distance_m=cumulative_distance_m,
                left_speed_feedback_m_s=left_speed_feedback_m_s,
                right_speed_feedback_m_s=right_speed_feedback_m_s, safety=safety,
                near_field_preparation=near_field_preparation,
                near_field_path_clear=near_field_path_clear,
            )
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


    def _dynamic_segment_speed(self, remaining_mm: float, maximum_m_s: float) -> float:
        acceleration = self.config.breakup_deceleration_m_s2
        if self.config.breakup_max_linear_deceleration_m_s2 is not None:
            acceleration = min(
                acceleration,
                self.config.breakup_max_linear_deceleration_m_s2,
            )
        distance_m = max(0.0, remaining_mm-self.config.breakup_braking_margin_mm)/1000.0
        return min(maximum_m_s, math.sqrt(2*acceleration*distance_m))


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


    def _target_is_fresh(self, target: TrackedTarget, timestamp_ns: int) -> bool:
        return (
            timestamp_ns >= target.last_seen_timestamp_ns
            and (timestamp_ns - target.last_seen_timestamp_ns) / 1_000_000.0
            <= self.config.green_max_age_ms
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
        if self._strategy_formal_phase:
            return _SharedMatchSequence.step(
                self, timestamp_ns, perception=perception, heading_rad=heading_rad,
                cumulative_distance_m=cumulative_distance_m,
                left_speed_feedback_m_s=left_speed_feedback_m_s,
                right_speed_feedback_m_s=right_speed_feedback_m_s, safety=safety,
                near_field_preparation=near_field_preparation,
                near_field_path_clear=near_field_path_clear,
            )
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
                self._safe_zone_observation_deadline_ns = None
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
        return True

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
            field_bounds=self._physical_field_bounds(),
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
            else (
                self._near_field_grasp_config.greedy_target_final_x_mm
                if self._greedy_active
                else self._near_field_grasp_config.target_final_x_mm
            )
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
        return self._safe_zone_transport_endpoint()

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
        self._breakup_frozen_plan = None
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

        return replace(self._finish_safe_zone_exit(timestamp_ns),
                       reason="strategy_blue_tasks_complete_start_match_green_search")

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
