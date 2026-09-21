"""正式 ``match`` 的相对动作序列开场变体。

除开场外全部复用当前 :class:`MatchSequence`。开场从当前位姿出发，按配置中的
``turn`` / ``straight`` 动作顺序执行；转向使用 IMU 航向进度，直行使用编码器
累计路程。动作之间只保留一个短暂零速切换，不根据场地点反复重算目标。
"""

from __future__ import annotations

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import TYPE_CHECKING

from rescue_vision.app.match import (
    GripperPosture,
    MatchDecision,
    MatchPreflight,
    MatchSequence,
    MatchStartArea,
    MatchState,
)
from rescue_vision.app.match_runtime import _run_hardware
from rescue_vision.config import (
    NBOpeningAction,
    NBOpeningStraight,
    NBOpeningTurn,
    NBOpeningWheelTurn,
)
from rescue_vision.localization import normalize_angle
from rescue_vision.motion import (
    MotionAccelerationOverrides,
    WheelAccelerationOverrides,
    RelativeActionCommand,
    RelativeActionController,
    RelativeActionFeedback,
    RelativeActionKind,
    RelativeActionProfile,
    WheelActionCommand,
    WheelActionFeedback,
    WheelActionPhase,
    WheelActionProfile,
    WheelTurnAndAdvanceController,
    WHEEL_COMMAND_REFRESH_S,
)

if TYPE_CHECKING:
    from rescue_vision.app.gripper_width_sequence import GraspPreparation
    from rescue_vision.config import AppConfig
    from rescue_vision.perception import PerceptionSnapshot


class MatchNBSequence(MatchSequence):
    """只替换开场动作，其余行为始终委托给当前正式流程。"""

    INITIAL_HEADING_RAD = -3.0 * math.pi / 4.0

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
        *,
        start_area: MatchStartArea | str | int = MatchStartArea.AREA_2,
    ) -> MatchNBSequence:
        sequence = super().from_app_config(config, start_area=start_area)
        if not isinstance(sequence, cls):
            raise TypeError("MatchNBSequence factory returned an unexpected type.")
        sequence._nb_motion_profile = replace(
            sequence._motion_profile,
            action_timeout_s=config.match.nb_opening_turn_timeout_s,
            stationary_confirm_time_s=config.match.nb_opening_settle_time_s,
        )
        sequence._nb_wheel_track_m = config.motion.wheel_track_m
        sequence._nb_initial_heading_rad = config.localization.fusion.initial_pose.heading_rad
        return sequence

    @property
    def motion_acceleration_limits(self) -> MotionAccelerationOverrides | None:
        """让 NB 规划使用与底层控制器相同的有效减速度。"""

        if self.state is not MatchState.NB_OPENING_SEQUENCE:
            return super().motion_acceleration_limits
        profile = getattr(self, "_nb_motion_profile", None)
        if profile is None:
            return super().motion_acceleration_limits
        return MotionAccelerationOverrides(
            linear_deceleration_m_s2=profile.linear_deceleration_m_s2,
            angular_deceleration_rad_s2=profile.angular_deceleration_rad_s2,
        )

    @property
    def wheel_acceleration_limits(self) -> WheelAccelerationOverrides | None:
        """轮级开场动作只限制左轮过渡，刹车阶段恢复车体默认限幅。"""

        if self.state is not MatchState.NB_OPENING_SEQUENCE:
            return None
        action = self._nb_active_action()
        controller = getattr(self, "_nb_wheel_action_controller", None)
        if not isinstance(action, NBOpeningWheelTurn) or controller is None:
            return None
        if controller.phase in {
            WheelActionPhase.STOPPING,
            WheelActionPhase.COMPLETE,
            WheelActionPhase.TIMEOUT,
            WheelActionPhase.WAITING_FEEDBACK,
        }:
            return None
        acceleration = action.left_transition_acceleration_m_s2
        return WheelAccelerationOverrides(
            left_acceleration_m_s2=acceleration,
            left_deceleration_m_s2=acceleration,
        )

    def start(self, timestamp_ns: int) -> MatchDecision:
        super().start(timestamp_ns)
        if not hasattr(self, "_nb_motion_profile"):
            self._nb_motion_profile = self._profile_from_runtime_config()
        if not hasattr(self, "_nb_wheel_track_m"):
            self._nb_wheel_track_m = self._motion_wheel_track_m
        self._nb_action_index = 0
        self._nb_action_started_ns: int | None = None
        self._nb_action_start_heading_rad: float | None = None
        self._nb_turn_last_heading_rad: float | None = None
        self._nb_turn_progress_rad = 0.0
        self._nb_action_start_distance_m: float | None = None
        self._nb_route_heading_rad: float | None = getattr(self, "_nb_initial_heading_rad", None)
        self._nb_action_controller: RelativeActionController | None = None
        self._nb_last_action_command: RelativeActionCommand | None = None
        self._nb_wheel_action_controller: WheelTurnAndAdvanceController | None = None
        self._nb_last_wheel_action_command: WheelActionCommand | None = None
        self._nb_gripper_opened = False
        self.state = MatchState.NB_OPENING_SEQUENCE
        return self._decision(timestamp_ns, 0.0, 0.0, "nb_opening_started")

    def _profile_from_runtime_config(self) -> RelativeActionProfile:
        """Build the testable default when no full AppConfig was provided."""

        return replace(
            self._motion_profile,
            action_timeout_s=self.config.nb_opening_turn_timeout_s,
            stationary_confirm_time_s=self.config.nb_opening_settle_time_s,
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
        if self.state is MatchState.NB_OPENING_SEQUENCE:
            return self._step_nb_sequence(
                timestamp_ns,
                heading_rad=heading_rad,
                cumulative_distance_m=cumulative_distance_m,
                left_speed_feedback_m_s=left_speed_feedback_m_s,
                right_speed_feedback_m_s=right_speed_feedback_m_s,
            )
        if self.state is MatchState.NB_OPENING_GRIPPER_OPEN:
            return self._step_nb_gripper_open(timestamp_ns)
        return super()._dispatch_state(
            timestamp_ns,
            perception=perception,
            heading_rad=heading_rad,
            cumulative_distance_m=cumulative_distance_m,
            left_speed_feedback_m_s=left_speed_feedback_m_s,
            right_speed_feedback_m_s=right_speed_feedback_m_s,
            near_field_preparation=near_field_preparation,
            near_field_path_clear=near_field_path_clear,
        )

    def _nb_active_action(self) -> NBOpeningAction | None:
        if self._nb_action_index >= len(self.config.nb_opening_actions):
            return None
        return self.config.nb_opening_actions[self._nb_action_index]

    @property
    def nb_opening_route_phase(self) -> str | None:
        if self.state is MatchState.NB_OPENING_GRIPPER_OPEN:
            return "open_gripper"
        if self.state is not MatchState.NB_OPENING_SEQUENCE:
            return None
        action = self._nb_active_action()
        if action is None:
            return None
        kind = (
            "wheel_turn" if isinstance(action, NBOpeningWheelTurn)
            else "turn" if isinstance(action, NBOpeningTurn)
            else "straight"
        )
        return (
            f"action_{self._nb_action_index + 1}_"
            f"{len(self.config.nb_opening_actions)}_{kind}"
        )

    @property
    def nb_opening_diagnostic(self) -> str | None:
        if self.state is MatchState.NB_OPENING_GRIPPER_OPEN:
            return "action=gripper_open"
        if self.state is not MatchState.NB_OPENING_SEQUENCE:
            return None
        action = self._nb_active_action()
        if action is None:
            return None
        heading = self._latest_heading_rad
        heading_text = (
            "unavailable"
            if heading is None
            else f"{math.degrees(normalize_angle(heading)):+.1f}deg"
        )
        wheel_command = self._nb_last_wheel_action_command
        if isinstance(action, NBOpeningWheelTurn):
            if wheel_command is None:
                detail = (
                    f"wheel_turn={math.degrees(action.angle_rad):+.1f}deg "
                    f"post_distance={action.post_turn_distance_m:.3f}m "
                    f"left={action.left_wheel_hold_speed_m_s:.3f}->"
                    f"{action.left_wheel_final_speed_m_s:.3f}m/s "
                    f"right={action.right_wheel_speed_m_s:.3f}m/s"
                )
            else:
                detail = (
                    f"wheel_turn={math.degrees(action.angle_rad):+.1f}deg "
                    f"angle_progress={math.degrees(wheel_command.angle_progress_rad):.1f}deg "
                    f"post_distance={action.post_turn_distance_m:.3f}m "
                    f"distance_progress={wheel_command.distance_progress_m:.3f}m "
                    f"left={wheel_command.left_wheel_velocity_m_s:.3f}m/s "
                    f"right={wheel_command.right_wheel_velocity_m_s:.3f}m/s"
                )
        elif isinstance(action, NBOpeningTurn):
            direction = 1.0 if action.angle_rad > 0.0 else -1.0
            progress = direction * self._nb_turn_progress_rad
            detail = (
                f"angle={action.angle_rad:+.3f}rad "
                f"progress={progress:.3f}/{abs(action.angle_rad):.3f}rad "
                f"speed={action.angular_velocity_rad_s:.3f}rad/s"
            )
        else:
            progress = None
            if (
                self._nb_action_start_distance_m is not None
                and self._latest_cumulative_distance_m is not None
            ):
                direction = 1.0 if action.distance_m > 0.0 else -1.0
                progress = direction * (
                    self._latest_cumulative_distance_m
                    - self._nb_action_start_distance_m
                )
            progress_text = "unavailable" if progress is None else f"{progress:.3f}"
            detail = (
                f"distance={action.distance_m:+.3f}m "
                f"progress={progress_text}/{abs(action.distance_m):.3f}m "
                f"speed={action.speed_m_s:.3f}m/s"
            )
        command = self._nb_last_action_command
        if isinstance(action, NBOpeningWheelTurn):
            command_text = (
                "command=none"
                if wheel_command is None
                else f"phase={wheel_command.phase.value} reason={wheel_command.reason}"
            )
        else:
            command_text = (
                "command=none"
                if command is None
                else (
                    f"phase={command.phase.value} "
                    f"error={command.position_error:+.4f} "
                    f"brake={command.braking_distance:.4f} "
                    f"delay={command.telemetry_delay_s:.3f}"
                )
            )
        timing_text = self._nb_action_diagnostic_context(
            self._latest_motion_timestamp_ns(),
            confirmation_progress=(
                "geometry_pending" if command is None else "geometry_checked"
            ),
        )
        return (
            f"action={self._nb_action_index + 1}/"
            f"{len(self.config.nb_opening_actions)} {detail} "
            f"heading={heading_text} "
            f"cumulative_distance_m={self._latest_cumulative_distance_m} "
            f"{command_text} {timing_text}"
        )

    def _latest_motion_timestamp_ns(self) -> int:
        latest = self._stationary_motion.latest
        if latest is None:
            return self._last_timestamp_ns or 0
        return max(self._last_timestamp_ns or latest.received_timestamp_ns,
                   latest.received_timestamp_ns)

    def _nb_action_diagnostic_context(
        self,
        timestamp_ns: int,
        *,
        confirmation_progress: str,
    ) -> str:
        started = self._nb_action_started_ns
        if started is None:
            elapsed_ms = "none"
            deadline = "none"
        else:
            elapsed_ms = f"{max(0, timestamp_ns - started) / 1e6:.1f}"
            deadline = str(
                started + self._seconds_to_ns(self.config.nb_opening_turn_timeout_s)
            )
        return (
            f"elapsed_ms={elapsed_ms} action_started_ns={started} "
            f"action_deadline_ns={deadline} "
            f"confirmation_progress={confirmation_progress} "
            f"preparation_age_ms=none "
            f"{self._stationary_motion.diagnostic(timestamp_ns)}"
        )

    def _nb_waiting_reason(
        self,
        base_reason: str,
        timestamp_ns: int,
        *,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
    ) -> str:
        return (
            f"{base_reason}:heading={'available' if heading_rad is not None else 'missing'} "
            f"distance={'available' if cumulative_distance_m is not None else 'missing'} "
            f"{self._nb_action_diagnostic_context(
                timestamp_ns, confirmation_progress='not_started'
            )}"
        )

    def _nb_gripper_angles_deg(self) -> tuple[float, float]:
        return (
            self.config.nb_opening_gripper_left_deg,
            self.config.nb_opening_gripper_right_deg,
        )

    def _nb_posture(
        self,
    ) -> tuple[GripperPosture, tuple[float, float] | None]:
        if self._nb_gripper_opened:
            return GripperPosture.OPEN, self._nb_gripper_angles_deg()
        return GripperPosture.CLOSED, None

    def _nb_begin_action(
        self,
        timestamp_ns: int,
        *,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
    ) -> bool:
        action = self._nb_active_action()
        assert action is not None
        if self._nb_action_started_ns is None:
            self._nb_action_started_ns = timestamp_ns
        if isinstance(action, NBOpeningWheelTurn):
            if heading_rad is None or cumulative_distance_m is None:
                return False
            self._nb_action_start_heading_rad = heading_rad
            self._nb_action_start_distance_m = cumulative_distance_m
            profile = WheelActionProfile(
                wheel_track_m=self._nb_wheel_track_m,
                right_wheel_speed_m_s=action.right_wheel_speed_m_s,
                left_wheel_hold_speed_m_s=action.left_wheel_hold_speed_m_s,
                left_wheel_final_speed_m_s=action.left_wheel_final_speed_m_s,
                left_transition_acceleration_m_s2=(
                    action.left_transition_acceleration_m_s2
                ),
                target_angle_rad=action.angle_rad,
                post_turn_distance_m=action.post_turn_distance_m,
                angle_tolerance_rad=self.config.nb_opening_heading_tolerance_rad,
                distance_tolerance_m=self.config.nb_opening_distance_tolerance_m,
                linear_deceleration_m_s2=self._nb_motion_profile.linear_deceleration_m_s2,
                telemetry_delay_s=(
                    self._nb_motion_profile.command_wait_s
                    + self._nb_motion_profile.execution_response_s
                ),
                stop_wheel_speed_m_s=self._nb_motion_profile.stop_wheel_speed_m_s,
                stop_angular_velocity_rad_s=self._nb_motion_profile.stop_angular_velocity_rad_s,
                stationary_confirm_time_s=self._nb_motion_profile.stationary_confirm_time_s,
                max_telemetry_age_s=self._nb_motion_profile.max_telemetry_age_s,
                action_timeout_s=self.config.nb_opening_turn_timeout_s,
            )
            self._nb_wheel_action_controller = WheelTurnAndAdvanceController(profile)
            self._nb_wheel_action_controller.begin(
                timestamp_ns=timestamp_ns,
                heading_rad=heading_rad,
                distance_m=cumulative_distance_m,
            )
            return True
        if isinstance(action, NBOpeningTurn):
            if heading_rad is None:
                return False
            self._nb_action_start_heading_rad = heading_rad
            self._nb_turn_last_heading_rad = heading_rad
            self._nb_turn_progress_rad = 0.0
            if self._nb_route_heading_rad is None:
                self._nb_route_heading_rad = heading_rad
            self._nb_route_heading_rad = normalize_angle(
                self._nb_route_heading_rad + action.angle_rad
            )
            self._nb_action_controller = RelativeActionController(
                self._nb_motion_profile
            )
            self._nb_action_controller.set_tolerances(
                position_tolerance=self.config.nb_opening_turn_tolerance_rad,
                heading_tolerance=self.config.nb_opening_heading_tolerance_rad,
            )
            self._nb_action_controller.begin(
                RelativeActionKind.TURN,
                action.angle_rad,
                start_heading_rad=heading_rad,
                timestamp_ns=timestamp_ns,
                cruise_speed=action.angular_velocity_rad_s,
                target_heading_rad=self._nb_route_heading_rad,
                pivot_track_m=(self._nb_wheel_track_m if action.pivot_wheel else None),
                exit_speed_m_s=action.exit_speed_m_s,
            )
        else:
            if cumulative_distance_m is None or heading_rad is None:
                return False
            self._nb_action_start_distance_m = cumulative_distance_m
            if self._nb_route_heading_rad is None:
                # The first straight action has no previous route heading.
                # Later straight actions retain the intended heading from the
                # preceding turn rather than accepting a turn's measured error
                # as a new reference.
                self._nb_route_heading_rad = heading_rad
            self._nb_action_controller = RelativeActionController(
                self._nb_motion_profile if action.settle_time_s is None else replace(
                    self._nb_motion_profile, stationary_confirm_time_s=action.settle_time_s,
                )
            )
            self._nb_action_controller.set_tolerances(
                position_tolerance=self.config.nb_opening_distance_tolerance_m,
                heading_tolerance=self.config.nb_opening_heading_tolerance_rad,
            )
            self._nb_action_controller.begin(
                RelativeActionKind.STRAIGHT,
                action.distance_m,
                start_heading_rad=heading_rad,
                timestamp_ns=timestamp_ns,
                cruise_speed=action.speed_m_s,
                target_heading_rad=self._nb_route_heading_rad,
            )
        return True

    def _nb_motion_feedback(
        self,
        timestamp_ns: int,
        *,
        progress: float,
        heading_rad: float | None,
        left_speed_feedback_m_s: float | None,
        right_speed_feedback_m_s: float | None,
    ) -> RelativeActionFeedback:
        latest = self._stationary_motion.latest
        telemetry_age_s: float | None = None
        stationary_latest_ns: int | None = None
        angular_velocity_rad_s: float | None = None
        if latest is not None:
            stationary_latest_ns = latest.received_timestamp_ns
            if latest.received_timestamp_ns <= timestamp_ns:
                telemetry_age_s = (
                    timestamp_ns - latest.received_timestamp_ns
                ) / 1_000_000_000.0
            if math.isfinite(latest.gyro_z_rad_s):
                # Only the magnitude is used for the stopping/braking gate;
                # turn direction comes from the unwrapped IMU heading.  This
                # keeps the calibrated gyro sign in the existing heading path.
                angular_velocity_rad_s = latest.gyro_z_rad_s
        return RelativeActionFeedback(
            timestamp_ns=timestamp_ns,
            progress=progress,
            heading_rad=heading_rad,
            left_wheel_velocity_m_s=left_speed_feedback_m_s,
            right_wheel_velocity_m_s=right_speed_feedback_m_s,
            angular_velocity_rad_s=angular_velocity_rad_s,
            telemetry_age_s=telemetry_age_s,
            stationary_since_ns=self._stationary_motion.stationary_since(
                timestamp_ns
            ),
            stationary_latest_ns=stationary_latest_ns,
        )

    def _nb_wheel_feedback(
        self,
        timestamp_ns: int,
        *,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None,
        right_speed_feedback_m_s: float | None,
    ) -> WheelActionFeedback:
        """组装一帧轮级动作反馈；缺失关键遥测时让控制器进入有界等待。"""

        latest = self._stationary_motion.latest
        stationary_latest_ns = None if latest is None else latest.received_timestamp_ns
        feedback_missing = any(
            value is None
            for value in (
                heading_rad,
                cumulative_distance_m,
                left_speed_feedback_m_s,
                right_speed_feedback_m_s,
            )
        )
        gyro_available = latest is not None and math.isfinite(latest.gyro_z_rad_s)
        if (
            feedback_missing
            or not gyro_available
            or stationary_latest_ns is None
            or stationary_latest_ns > timestamp_ns
        ):
            telemetry_age_s = self._nb_motion_profile.max_telemetry_age_s * 2.0
        else:
            telemetry_age_s = (timestamp_ns - stationary_latest_ns) / 1e9
        start_heading = self._nb_action_start_heading_rad or 0.0
        start_distance = self._nb_action_start_distance_m or 0.0
        gyro = 0.0 if latest is None or not math.isfinite(latest.gyro_z_rad_s) else latest.gyro_z_rad_s
        return WheelActionFeedback(
            timestamp_ns=timestamp_ns,
            heading_rad=start_heading if heading_rad is None else heading_rad,
            distance_m=(start_distance if cumulative_distance_m is None else cumulative_distance_m),
            left_wheel_velocity_m_s=(0.0 if left_speed_feedback_m_s is None else left_speed_feedback_m_s),
            right_wheel_velocity_m_s=(0.0 if right_speed_feedback_m_s is None else right_speed_feedback_m_s),
            angular_velocity_rad_s=gyro,
            telemetry_age_s=max(0.0, telemetry_age_s),
            stationary_since_ns=self._stationary_motion.stationary_since(timestamp_ns),
            stationary_latest_ns=stationary_latest_ns,
        )

    def _nb_finish_action(
        self,
        timestamp_ns: int,
        *,
        reason: str,
    ) -> MatchDecision:
        posture, angles = self._nb_posture()
        completed_index = self._nb_action_index + 1
        self._nb_action_index += 1
        self._nb_action_started_ns = None
        self._nb_action_start_heading_rad = None
        self._nb_turn_last_heading_rad = None
        self._nb_turn_progress_rad = 0.0
        self._nb_action_start_distance_m = None
        self._nb_action_controller = None
        self._nb_wheel_action_controller = None
        self._nb_last_action_command = None
        self._nb_last_wheel_action_command = None
        if self.config.nb_opening_gripper_after_action == completed_index:
            self._nb_gripper_opened = True
            self.state = MatchState.NB_OPENING_GRIPPER_OPEN
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "nb_opening_gripper_opened",
                posture=GripperPosture.OPEN,
                gripper_angles_deg=self._nb_gripper_angles_deg(),
            )
        if self._nb_action_index >= len(self.config.nb_opening_actions):
            self.state = MatchState.SEARCH_CLUSTER
            self._begin_cluster_search()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "nb_opening_sequence_complete",
                posture=posture,
                gripper_angles_deg=angles,
            )
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "nb_opening_action_advanced",
            posture=posture,
            gripper_angles_deg=angles,
        )

    def _step_nb_gripper_open(self, timestamp_ns: int) -> MatchDecision:
        self.state = MatchState.NB_OPENING_SEQUENCE
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "nb_opening_gripper_opened",
            posture=GripperPosture.OPEN,
            gripper_angles_deg=self._nb_gripper_angles_deg(),
        )

    def _step_nb_sequence(
        self,
        timestamp_ns: int,
        *,
        heading_rad: float | None,
        cumulative_distance_m: float | None,
        left_speed_feedback_m_s: float | None,
        right_speed_feedback_m_s: float | None,
    ) -> MatchDecision:
        action = self._nb_active_action()
        posture, angles = self._nb_posture()
        if action is None:
            self.state = MatchState.SEARCH_CLUSTER
            self._begin_cluster_search()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "nb_opening_sequence_complete",
                posture=posture,
                gripper_angles_deg=angles,
            )
        if self._nb_action_controller is None and self._nb_wheel_action_controller is None:
            if not self._nb_begin_action(
                timestamp_ns,
                heading_rad=heading_rad,
                cumulative_distance_m=cumulative_distance_m,
            ):
                assert self._nb_action_started_ns is not None
                kind = (
                    "wheel_turn" if isinstance(action, NBOpeningWheelTurn)
                    else "turn" if isinstance(action, NBOpeningTurn)
                    else "straight"
                )
                if timestamp_ns - self._nb_action_started_ns >= self._seconds_to_ns(
                    self.config.nb_opening_turn_timeout_s
                ):
                    self.state = MatchState.TERMINAL_STOP
                    return self._decision(
                        timestamp_ns,
                        0.0,
                        0.0,
                        f"nb_opening_{kind}_timeout_stop_waiting_feedback",
                        posture=posture,
                        gripper_angles_deg=angles,
                    )
                waiting_reason = self._nb_waiting_reason(
                    (
                        (
                            "nb_opening_wheel_turn_waiting_heading_or_distance"
                            if isinstance(action, NBOpeningWheelTurn)
                            else "nb_opening_turn_waiting_heading"
                            if isinstance(action, NBOpeningTurn)
                            else "nb_opening_straight_waiting_heading_or_distance"
                        )
                    ),
                    timestamp_ns,
                    heading_rad=heading_rad,
                    cumulative_distance_m=cumulative_distance_m,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    waiting_reason,
                    posture=posture,
                    gripper_angles_deg=angles,
                )

        if isinstance(action, NBOpeningWheelTurn):
            controller = self._nb_wheel_action_controller
            assert controller is not None
            command = controller.update(
                self._nb_wheel_feedback(
                    timestamp_ns,
                    heading_rad=heading_rad,
                    cumulative_distance_m=cumulative_distance_m,
                    left_speed_feedback_m_s=left_speed_feedback_m_s,
                    right_speed_feedback_m_s=right_speed_feedback_m_s,
                )
            )
            self._nb_last_wheel_action_command = command
            action_number = self._nb_action_index + 1
            if command.timed_out:
                self.state = MatchState.TERMINAL_STOP
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    f"nb_opening_wheel_turn_{action_number}_timeout:{command.reason}:"
                    f"{self._nb_action_diagnostic_context(
                        timestamp_ns, confirmation_progress='timeout'
                    )}",
                    posture=posture,
                    gripper_angles_deg=angles,
                )
            if command.complete:
                return self._nb_finish_action(
                    timestamp_ns,
                    reason=(
                        f"nb_opening_action_{action_number}_complete"
                        f":{command.reason}"
                    ),
                )
            left = command.left_wheel_velocity_m_s
            right = command.right_wheel_velocity_m_s
            left_body = left / self._motion_wheel_weights[0]
            right_body = right / self._motion_wheel_weights[1]
            linear = (left_body + right_body) / 2.0
            angular = (right_body - left_body) / self._nb_wheel_track_m
            return self._decision(
                timestamp_ns,
                linear,
                angular,
                f"nb_opening_wheel_turn_{action_number}_{command.phase.value}"
                f":{command.reason}:"
                f"{self._nb_action_diagnostic_context(
                    timestamp_ns,
                    confirmation_progress=command.phase.value,
                )}",
                posture=posture,
                gripper_angles_deg=angles,
                min_wheel_velocity_m_s=0.0,
                wheel_speeds_m_s=(left, right),
            )

        if isinstance(action, NBOpeningTurn):
            if heading_rad is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    self._nb_waiting_reason(
                        "nb_opening_turn_waiting_heading",
                        timestamp_ns,
                        heading_rad=heading_rad,
                        cumulative_distance_m=cumulative_distance_m,
                    ),
                    posture=posture,
                    gripper_angles_deg=angles,
                )
            assert self._nb_turn_last_heading_rad is not None
            self._nb_turn_progress_rad += normalize_angle(
                heading_rad - self._nb_turn_last_heading_rad
            )
            self._nb_turn_last_heading_rad = heading_rad
            direction = 1.0 if action.angle_rad > 0.0 else -1.0
            progress = direction * self._nb_turn_progress_rad
        else:
            if cumulative_distance_m is None:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    self._nb_waiting_reason(
                        "nb_opening_straight_waiting_distance",
                        timestamp_ns,
                        heading_rad=heading_rad,
                        cumulative_distance_m=cumulative_distance_m,
                    ),
                    posture=posture,
                    gripper_angles_deg=angles,
                )
            assert self._nb_action_start_distance_m is not None
            direction = 1.0 if action.distance_m > 0.0 else -1.0
            progress = direction * (
                cumulative_distance_m - self._nb_action_start_distance_m
            )
        if (
            isinstance(action, NBOpeningStraight)
            and self._nb_action_index + 1 < len(self.config.nb_opening_actions)
            and isinstance(
                self.config.nb_opening_actions[self._nb_action_index + 1],
                NBOpeningWheelTurn,
            )
            and progress >= abs(action.distance_m)
        ):
            # 轮级动作明确要求 1.6 m 处右轮不回零：把当前直行的
            # 路程样本直接作为 wheel_turn 的起点，跳过旧动作的全车刹停。
            self._nb_action_index += 1
            self._nb_action_started_ns = None
            self._nb_action_start_heading_rad = None
            self._nb_turn_last_heading_rad = None
            self._nb_turn_progress_rad = 0.0
            self._nb_action_start_distance_m = None
            self._nb_action_controller = None
            self._nb_wheel_action_controller = None
            self._nb_last_action_command = None
            self._nb_last_wheel_action_command = None
            return self._step_nb_sequence(
                timestamp_ns,
                heading_rad=heading_rad,
                cumulative_distance_m=cumulative_distance_m,
                left_speed_feedback_m_s=left_speed_feedback_m_s,
                right_speed_feedback_m_s=right_speed_feedback_m_s,
            )
        controller = self._nb_action_controller
        assert controller is not None
        feedback = self._nb_motion_feedback(
            timestamp_ns,
            progress=progress,
            heading_rad=heading_rad,
            left_speed_feedback_m_s=left_speed_feedback_m_s,
            right_speed_feedback_m_s=right_speed_feedback_m_s,
        )
        command = controller.update(feedback)
        self._nb_last_action_command = command
        action_number = self._nb_action_index + 1
        kind = "turn" if isinstance(action, NBOpeningTurn) else "straight"
        if command.timed_out:
            self.state = MatchState.TERMINAL_STOP
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"nb_opening_{kind}_{action_number}_timeout:{command.reason}:"
                f"{self._nb_action_diagnostic_context(
                    timestamp_ns, confirmation_progress='timeout'
                )}",
                posture=posture,
                gripper_angles_deg=angles,
            )
        if command.complete:
            advanced = self._nb_finish_action(
                timestamp_ns,
                reason=(
                    f"nb_opening_action_{action_number}_complete"
                    f":{command.reason}"
                ),
            )
            if isinstance(action, NBOpeningTurn) and action.exit_speed_m_s > 0.0:
                # Freeze the straight origin at this feedback sample. Its
                # first target lets the left catch the still-moving right;
                # do not inject the usual zero-speed action-switch command.
                if cumulative_distance_m is None or not self._nb_begin_action(
                    timestamp_ns, heading_rad=heading_rad,
                    cumulative_distance_m=cumulative_distance_m,
                ):
                    self.state = MatchState.TERMINAL_STOP
                    return self._decision(timestamp_ns, 0.0, 0.0, "nb_pivot_handoff_missing_distance")
                return replace(
                    advanced, linear_velocity_m_s=command.linear_velocity_m_s,
                    angular_velocity_rad_s=0.0, min_wheel_velocity_m_s=0.0,
                    reason="nb_opening_pivot_moving_handoff",
                )
            return advanced
        return self._decision(
            timestamp_ns,
            command.linear_velocity_m_s,
            command.angular_velocity_rad_s,
            f"nb_opening_{kind}_{action_number}_{command.phase.value}"
            f":{command.reason}:"
            f"{self._nb_action_diagnostic_context(
                timestamp_ns,
                confirmation_progress=(
                    "stationary_wait" if command.phase.value == "settle"
                    else "geometry_checked"
                ),
            )}",
            posture=posture,
            gripper_angles_deg=angles,
            min_wheel_velocity_m_s=(
                0.0 if command.use_zero_min_wheel_velocity else None
            ),
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run current match with a configurable relative action opening."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--start-area",
        choices=(MatchStartArea.AREA_2.value,),
        default=MatchStartArea.AREA_2.value,
        help="本入口只支持区域 2；开场动作从当前起点按相对配置执行。",
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help="Confirm a physical emergency stop and continuous supervision.",
    )
    parser.add_argument("--local-preview", action="store_true")
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
        sequence_factory=MatchNBSequence.from_app_config,
        mode_name="match_nb",
        log_file_prefix="match_nb_",
        preview_title="Match-NB perception",
    )


__all__ = ["MatchDecision", "MatchNBSequence", "MatchState", "MatchPreflight"]


if __name__ == "__main__":
    main()
