"""正式 ``match`` 的简单开场变体。

除开场外全部复用当前 :class:`MatchSequence`。区域 2 开场固定为：转向并前进到
``FieldPoint(100, 800)``，张爪，转向并前进到 ``FieldPoint(100, -900)``，
保持朝向倒车到 ``FieldPoint(100, 0)``，随后进入正式目标团搜索。

每段只在起步前确定一次直线航向，行驶中只做普通 IMU 航向保持；不做横向
前视、制动提前量或途中重复停车校准。
"""

from __future__ import annotations

import argparse
import math
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
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import normalize_angle

if TYPE_CHECKING:
    from rescue_vision.app.gripper_width_sequence import GraspPreparation
    from rescue_vision.config import AppConfig
    from rescue_vision.perception import PerceptionSnapshot


class MatchNBSequence(MatchSequence):
    """只替换开场动作，其余行为始终委托给当前正式流程。"""

    def start(self, timestamp_ns: int) -> MatchDecision:
        super().start(timestamp_ns)
        self._nb_reset_leg()
        self._nb_leg_arrived = False
        self._nb_settle_until_ns: int | None = None
        self._nb_reverse_heading_rad: float | None = None
        self._nb_turn_settle_until_ns: int | None = None
        self.state = MatchState.NB_OPENING_TO_FIRST
        return self._decision(timestamp_ns, 0.0, 0.0, "nb_opening_started")

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
        if self.state is MatchState.NB_OPENING_TO_FIRST:
            return self._step_nb_to_first(timestamp_ns)
        if self.state is MatchState.NB_OPENING_GRIPPER_OPEN:
            return self._step_nb_gripper_open(timestamp_ns)
        if self.state is MatchState.NB_OPENING_TO_SECOND:
            return self._step_nb_to_second(timestamp_ns)
        if self.state is MatchState.NB_OPENING_REVERSE:
            return self._step_nb_reverse(timestamp_ns)
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

    def _nb_active_leg(self) -> tuple[str, FieldPoint] | None:
        if self.state is MatchState.NB_OPENING_TO_FIRST:
            return "turn_and_move_to_first", self.config.nb_opening_first_target_field
        if self.state is MatchState.NB_OPENING_GRIPPER_OPEN:
            return "open_gripper", self.config.nb_opening_first_target_field
        if self.state is MatchState.NB_OPENING_TO_SECOND:
            return "turn_and_move_to_second", self.config.nb_opening_second_target_field
        if self.state is MatchState.NB_OPENING_REVERSE:
            return "reverse", self.config.nb_opening_reverse_target_field
        return None

    @property
    def nb_opening_route_phase(self) -> str | None:
        leg = self._nb_active_leg()
        return None if leg is None else leg[0]

    @property
    def nb_opening_diagnostic(self) -> str | None:
        leg = self._nb_active_leg()
        if leg is None:
            return None
        phase, target = leg
        position = self._fallback_field_position
        heading = self._latest_heading_rad
        heading_text = (
            "unavailable"
            if heading is None
            else f"{math.degrees(normalize_angle(heading)):+.1f}deg"
        )
        if position is None:
            return (
                f"phase={phase} target=({target.x:+.0f},{target.y:+.0f})mm "
                f"position=unavailable heading={heading_text}"
            )
        return (
            f"phase={phase} target=({target.x:+.0f},{target.y:+.0f})mm "
            f"position=({position.x:+.0f},{position.y:+.0f})mm "
            f"error=({target.x - position.x:+.0f},{target.y - position.y:+.0f})mm "
            f"heading={heading_text}"
        )

    def _nb_gripper_angles_deg(self) -> tuple[float, float]:
        return (
            self.config.nb_opening_gripper_left_deg,
            self.config.nb_opening_gripper_right_deg,
        )

    def _nb_reset_leg(self) -> None:
        self._nb_leg_start_position: FieldPoint | None = None
        self._nb_leg_heading_rad: float | None = None
        self._nb_turn_complete = False
        self._nb_turn_started_ns: int | None = None

    def _nb_start_leg(
        self,
        timestamp_ns: int,
        target: FieldPoint,
        *,
        reverse: bool,
    ) -> bool:
        position = self._fallback_field_position
        if position is None:
            return False
        self._nb_leg_start_position = position
        if reverse and self._nb_reverse_heading_rad is not None:
            self._nb_leg_heading_rad = self._nb_reverse_heading_rad
        elif reverse:
            self._nb_leg_heading_rad = math.atan2(
                position.y - target.y,
                position.x - target.x,
            )
        else:
            self._nb_leg_heading_rad = math.atan2(
                target.y - position.y,
                target.x - position.x,
            )
        self._nb_turn_started_ns = timestamp_ns
        return True

    def _nb_posture(
        self,
        gripper_open: bool,
    ) -> tuple[GripperPosture, tuple[float, float] | None]:
        if gripper_open:
            return GripperPosture.OPEN, self._nb_gripper_angles_deg()
        return GripperPosture.CLOSED, None

    def _nb_finish_leg(
        self,
        timestamp_ns: int,
        *,
        next_state: MatchState,
        arrive_reason: str,
        posture: GripperPosture,
        angles: tuple[float, float] | None,
    ) -> MatchDecision:
        if not self._nb_leg_arrived:
            self._nb_leg_arrived = True
            self._nb_settle_until_ns = timestamp_ns + self._seconds_to_ns(
                self.config.nb_opening_settle_time_s
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"{arrive_reason}_wait",
                posture=posture,
                gripper_angles_deg=angles,
            )
        assert self._nb_settle_until_ns is not None
        if timestamp_ns < self._nb_settle_until_ns:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "nb_opening_waypoint_settle",
                posture=posture,
                gripper_angles_deg=angles,
            )
        if next_state is MatchState.NB_OPENING_REVERSE:
            # 倒车严格沿用第二段前进航向，不在第二航点重新计算或转向。
            self._nb_reverse_heading_rad = self._nb_leg_heading_rad
        self._nb_leg_arrived = False
        self._nb_settle_until_ns = None
        self._nb_reset_leg()
        self.state = next_state
        if next_state is MatchState.SEARCH_CLUSTER:
            self._begin_cluster_search()
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            arrive_reason,
            posture=posture,
            gripper_angles_deg=angles,
        )

    def _nb_drive_leg(
        self,
        timestamp_ns: int,
        *,
        target: FieldPoint,
        speed_m_s: float,
        reverse: bool,
        turn_before_drive: bool,
        gripper_open: bool,
        next_state: MatchState,
        turn_reason: str,
        drive_reason: str,
        arrive_reason: str,
    ) -> MatchDecision:
        posture, angles = self._nb_posture(gripper_open)
        if self._nb_leg_start_position is None and not self._nb_start_leg(
            timestamp_ns,
            target,
            reverse=reverse,
        ):
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"{drive_reason}_waiting_position",
                posture=posture,
                gripper_angles_deg=angles,
            )

        start = self._nb_leg_start_position
        desired_heading = self._nb_leg_heading_rad
        current_heading = self._latest_heading_rad
        assert start is not None
        assert desired_heading is not None
        if current_heading is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"{drive_reason}_waiting_heading",
                posture=posture,
                gripper_angles_deg=angles,
            )

        if self._safe_zone_line_coordinate_threshold_reached(
            target,
            start,
            tolerance_mm=self.config.nb_opening_align_tolerance_mm,
        ):
            return self._nb_finish_leg(
                timestamp_ns,
                next_state=next_state,
                arrive_reason=arrive_reason,
                posture=posture,
                angles=angles,
            )

        if turn_before_drive and not self._nb_turn_complete:
            heading_error = normalize_angle(desired_heading - current_heading)
            if abs(heading_error) <= self.config.nb_opening_heading_tolerance_rad:
                self._nb_turn_complete = True
                self._nb_turn_settle_until_ns = timestamp_ns + self._seconds_to_ns(
                    self.config.nb_opening_settle_time_s
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    f"{turn_reason}_settle",
                    posture=posture,
                    gripper_angles_deg=angles,
                )
            else:
                assert self._nb_turn_started_ns is not None
                if timestamp_ns - self._nb_turn_started_ns >= self._seconds_to_ns(
                    self.config.nb_opening_align_timeout_s
                ):
                    self.state = MatchState.TERMINAL_STOP
                    return self._decision(
                        timestamp_ns,
                        0.0,
                        0.0,
                        "nb_opening_turn_timeout_stop",
                        posture=posture,
                        gripper_angles_deg=angles,
                    )
                angular = self._heading_hold_angular_velocity(
                    desired_heading,
                    kp_rad_s=self.config.nb_opening_heading_kp_rad_s,
                    max_angular_velocity_rad_s=(
                        self.config.nb_opening_align_angular_velocity_rad_s
                    ),
                    tolerance_rad=self.config.nb_opening_heading_tolerance_rad,
                )
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0 if angular is None else angular,
                    turn_reason,
                    posture=posture,
                    gripper_angles_deg=angles,
                )

        if turn_before_drive and self._nb_turn_settle_until_ns is not None:
            if timestamp_ns < self._nb_turn_settle_until_ns:
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    f"{turn_reason}_settle",
                    posture=posture,
                    gripper_angles_deg=angles,
                )
            self._nb_turn_settle_until_ns = None
            # 释放转向惯性后才允许前进；只有此时仍明显偏离才重开一次转向。
            if (
                abs(normalize_angle(desired_heading - current_heading))
                > self.config.nb_opening_heading_tolerance_rad
            ):
                self._nb_turn_complete = False
                self._nb_turn_started_ns = timestamp_ns
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    f"{turn_reason}_settle_realign",
                    posture=posture,
                    gripper_angles_deg=angles,
                )

        angular = self._heading_hold_angular_velocity(
            desired_heading,
            kp_rad_s=self.config.nb_opening_heading_kp_rad_s,
            max_angular_velocity_rad_s=(
                self.config.nb_opening_heading_max_angular_velocity_rad_s
            ),
            # 起步后直接保持本段固定航向，不再引入第二套行驶校准门限。
            tolerance_rad=0.0,
        )
        return self._decision(
            timestamp_ns,
            -speed_m_s if reverse else speed_m_s,
            0.0 if angular is None else angular,
            drive_reason,
            posture=posture,
            gripper_angles_deg=angles,
        )

    def _step_nb_to_first(self, timestamp_ns: int) -> MatchDecision:
        return self._nb_drive_leg(
            timestamp_ns,
            target=self.config.nb_opening_first_target_field,
            speed_m_s=self.config.nb_opening_first_speed_m_s,
            reverse=False,
            turn_before_drive=True,
            gripper_open=False,
            next_state=MatchState.NB_OPENING_GRIPPER_OPEN,
            turn_reason="nb_opening_turn_to_first",
            drive_reason="nb_opening_to_first",
            arrive_reason="nb_opening_first_reached_open_gripper",
        )

    def _step_nb_gripper_open(self, timestamp_ns: int) -> MatchDecision:
        self.state = MatchState.NB_OPENING_TO_SECOND
        self._nb_reset_leg()
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "nb_opening_gripper_opened",
            posture=GripperPosture.OPEN,
            gripper_angles_deg=self._nb_gripper_angles_deg(),
        )

    def _step_nb_to_second(self, timestamp_ns: int) -> MatchDecision:
        return self._nb_drive_leg(
            timestamp_ns,
            target=self.config.nb_opening_second_target_field,
            speed_m_s=self.config.nb_opening_second_speed_m_s,
            reverse=False,
            turn_before_drive=True,
            gripper_open=True,
            next_state=MatchState.NB_OPENING_REVERSE,
            turn_reason="nb_opening_turn_to_second",
            drive_reason="nb_opening_to_second",
            arrive_reason="nb_opening_second_reached_start_reverse",
        )

    def _step_nb_reverse(self, timestamp_ns: int) -> MatchDecision:
        return self._nb_drive_leg(
            timestamp_ns,
            target=self.config.nb_opening_reverse_target_field,
            speed_m_s=self.config.nb_opening_reverse_speed_m_s,
            reverse=True,
            turn_before_drive=False,
            gripper_open=True,
            next_state=MatchState.SEARCH_CLUSTER,
            turn_reason="nb_opening_reverse_turn_unused",
            drive_reason="nb_opening_reverse",
            arrive_reason="nb_opening_reverse_reached_start_search",
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run current match with the simple area-2 waypoint opening."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--start-area",
        choices=(MatchStartArea.AREA_2.value,),
        default=MatchStartArea.AREA_2.value,
        help="本入口只支持区域 2；开场航点按区域 2 场地坐标配置。",
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
