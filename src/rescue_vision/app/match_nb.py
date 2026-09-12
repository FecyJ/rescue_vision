"""无解团（no-breakup）正式流程变体：只替换开场动作。

该入口复用 ``MatchSequence`` 的解团、绿色物资转运、安全区运输和退出逻辑，
唯一区别是开场：删除固定启动转向和定距直行，改为按绝对场地坐标依次
直线航点推进——先直线移动到第一航点，夹爪打开到指定左右角度，再直线向前
到第二航点，最后倒车回退航点，随后进入正常解团流程。

区域 2 的默认示例（初始场地位姿 ``(1350, 1350)``、航向 ``-90°``）：
``(130, 800) -> 张爪(左 50°/右 130°) -> (130, -900) -> 倒车(130, 0)``。
每个航点坐标、每段速度、夹爪左右角度、到位停顿和航向对准参数都通过
``match`` 配置节的 ``nb_opening_*`` 字段单独配置。开场航点按区域 2 的坐标
基准书写；本入口不提供区域 3 的中心对称，区域 3 需单独配置对应坐标。

每段直线先原地对准航向再平移：若一边转向一边前进，起始航向误差会行程一条
弧线，终点会横向偏出数百 mm。到位后按 ``nb_opening_settle_time_s`` 零速停稳，
再从停稳后的位姿重新起算下一段，排除刹车残余位移。
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
    """开场改为绝对坐标直线航点、其余复用正式流程的无解团变体。"""

    def start(self, timestamp_ns: int) -> MatchDecision:
        """预检通过后进入开场第一航点，跳过固定启动转向+直行。"""

        super().start(timestamp_ns)
        self._nb_reset_leg()
        self._nb_leg_arrived = False
        self._nb_settle_until_ns = None
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
        """先处理开场航点状态，其余委托父类正式流程。"""

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

    # ------------------------------------------------------------------
    # 开场诊断
    # ------------------------------------------------------------------

    def _nb_active_leg(self) -> tuple[str, FieldPoint] | None:
        """返回当前开场阶段名和目标航点；不在开场时为 None。"""

        if self.state is MatchState.NB_OPENING_TO_FIRST:
            return "to_first", self.config.nb_opening_first_target_field
        if self.state is MatchState.NB_OPENING_GRIPPER_OPEN:
            return (
                "first_arrived_open_gripper",
                self.config.nb_opening_first_target_field,
            )
        if self.state is MatchState.NB_OPENING_TO_SECOND:
            return "to_second", self.config.nb_opening_second_target_field
        if self.state is MatchState.NB_OPENING_REVERSE:
            return "reverse", self.config.nb_opening_reverse_target_field
        return None

    @property
    def nb_opening_route_phase(self) -> str | None:
        """返回开场航点细分阶段，供车端按阶段变化记录诊断。"""

        leg = self._nb_active_leg()
        return None if leg is None else leg[0]

    @property
    def nb_opening_diagnostic(self) -> str | None:
        """返回一行开场航点诊断：阶段、目标、估计位置、剩余误差和航向。

        这行是定位“车辆实际落点偏离航点”的唯一车端读数：仅凭 state/reason
        横幅无法区分是策略没有到位，还是航位推算本身已经漂移。
        """

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
                f"phase={phase} "
                f"target=({target.x:+.0f},{target.y:+.0f})mm "
                f"position=unavailable heading={heading_text}"
            )
        return (
            f"phase={phase} "
            f"target=({target.x:+.0f},{target.y:+.0f})mm "
            f"position=({position.x:+.0f},{position.y:+.0f})mm "
            f"error=({target.x - position.x:+.0f},"
            f"{target.y - position.y:+.0f})mm "
            f"heading={heading_text}"
        )

    # ------------------------------------------------------------------
    # 开场航点共用的位姿、对准与停稳辅助
    # ------------------------------------------------------------------

    def _nb_gripper_angles_deg(self) -> tuple[float, float]:
        """返回开场的固定张爪左右角度。"""

        return (
            self.config.nb_opening_gripper_left_deg,
            self.config.nb_opening_gripper_right_deg,
        )

    def _nb_gripper_state(
        self,
        gripper_open: bool,
    ) -> tuple[GripperPosture, tuple[float, float] | None]:
        """返回该段直线应保持的夹爪姿态和显式角度。"""

        if not gripper_open:
            return GripperPosture.CLOSED, None
        return GripperPosture.OPEN, self._nb_gripper_angles_deg()

    def _nb_reset_leg(self) -> None:
        """清空当前直线段的起算位姿、航向和对准计时。"""

        self._nb_leg_start_position = None
        self._nb_leg_heading_rad = None
        self._nb_align_started_ns = None

    def _nb_compute_leg_heading_rad(
        self,
        target: FieldPoint,
        *,
        reverse: bool,
    ) -> None:
        """从当前估计场地位置起算本段直线航向；倒车取背向目标航向。"""

        position = self._fallback_field_position
        if position is None:
            self._nb_leg_start_position = None
            self._nb_leg_heading_rad = None
            return
        self._nb_leg_start_position = position
        if reverse:
            self._nb_leg_heading_rad = math.atan2(
                position.y - target.y,
                position.x - target.x,
            )
        else:
            self._nb_leg_heading_rad = math.atan2(
                target.y - position.y,
                target.x - position.x,
            )

    def _nb_begin_settle(self, timestamp_ns: int) -> None:
        """按 ``nb_opening_settle_time_s`` 登记一次到位零速停稳。"""

        duration = self.config.nb_opening_settle_time_s
        self._nb_settle_until_ns = (
            timestamp_ns + self._seconds_to_ns(duration) if duration > 0.0 else None
        )

    def _nb_consume_settle(
        self,
        timestamp_ns: int,
        *,
        reason: str,
        posture: GripperPosture,
        angles: tuple[float, float] | None,
    ) -> MatchDecision | None:
        """到位停稳未到期时返回零速决策，到期后返回 None 放行。"""

        if self._nb_settle_until_ns is None:
            return None
        if timestamp_ns < self._nb_settle_until_ns:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                reason,
                posture=posture,
                gripper_angles_deg=angles,
            )
        self._nb_settle_until_ns = None
        return None

    # ------------------------------------------------------------------
    # 单段直线推进
    # ------------------------------------------------------------------

    def _nb_drive_leg(
        self,
        timestamp_ns: int,
        *,
        target: FieldPoint,
        speed_m_s: float,
        reverse: bool,
        gripper_open: bool,
        next_state: MatchState,
        drive_reason: str,
        align_reason: str,
        arrive_reason: str,
    ) -> MatchDecision:
        """执行一段“先原地对准航向、再直线推进/倒车”的绝对坐标航点段。

        到位后先按 ``nb_opening_settle_time_s`` 零速停稳，停稳结束后才切换到
        下一个状态，因此每段都从停稳后的位姿重新起算，排除刹车残余位移。
        """

        posture, angles = self._nb_gripper_state(gripper_open)

        if self._nb_leg_arrived:
            settling = self._nb_consume_settle(
                timestamp_ns,
                reason=f"{drive_reason}_arrival_settle",
                posture=posture,
                angles=angles,
            )
            if settling is not None:
                return settling
            self._nb_leg_arrived = False
            self._nb_reset_leg()
            self.state = next_state
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                arrive_reason,
                posture=posture,
                gripper_angles_deg=angles,
            )

        start = self._nb_leg_start_position
        if start is None:
            self._nb_compute_leg_heading_rad(target, reverse=reverse)
            start = self._nb_leg_start_position
            self._nb_align_started_ns = timestamp_ns
        if start is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"{drive_reason}_waiting_position",
                posture=posture,
                gripper_angles_deg=angles,
            )

        desired_heading = self._nb_leg_heading_rad
        current_heading = self._latest_heading_rad
        if desired_heading is None or current_heading is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"{drive_reason}_waiting_heading",
                posture=posture,
                gripper_angles_deg=angles,
            )

        tolerance_rad = self.config.nb_opening_heading_tolerance_rad
        aligned = (
            abs(normalize_angle(desired_heading - current_heading)) <= tolerance_rad
        )
        # 未对准时只原地旋转、不前进：否则起始大角度误差会行程弧线并把终点
        # 横向带偏。对准后改用较小的直线保持角速度上限做小幅修正。
        angular = self._heading_hold_angular_velocity(
            desired_heading,
            kp_rad_s=self.config.nb_opening_heading_kp_rad_s,
            max_angular_velocity_rad_s=(
                self.config.nb_opening_heading_max_angular_velocity_rad_s
                if aligned
                else self.config.nb_opening_align_angular_velocity_rad_s
            ),
            tolerance_rad=tolerance_rad,
        )
        if angular is None:
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"{drive_reason}_waiting_heading",
                posture=posture,
                gripper_angles_deg=angles,
            )
        if aligned:
            # 计时器表示“连续未对准”的时长；对准后立即重新起算，这样直线段中途
            # 一次瞬时偏航不会累加到超时上而误触发保守停车。
            self._nb_align_started_ns = timestamp_ns
        else:
            if self._nb_align_timed_out(timestamp_ns):
                self.state = MatchState.TERMINAL_STOP
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    "nb_opening_align_timeout_stop",
                    posture=posture,
                    gripper_angles_deg=angles,
                )
            return self._decision(
                timestamp_ns,
                0.0,
                angular,
                align_reason,
                posture=posture,
                gripper_angles_deg=angles,
            )

        if self._safe_zone_line_coordinate_threshold_reached(
            target,
            start,
            tolerance_mm=self.config.nb_opening_align_tolerance_mm,
        ):
            self._nb_leg_arrived = True
            self._nb_begin_settle(timestamp_ns)
            # 到位即作废本段起算数据，保证下一段只能从停稳后的新位姿重算。
            self._nb_reset_leg()
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                f"{drive_reason}_arrived",
                posture=posture,
                gripper_angles_deg=angles,
            )

        return self._decision(
            timestamp_ns,
            -speed_m_s if reverse else speed_m_s,
            angular,
            drive_reason,
            posture=posture,
            gripper_angles_deg=angles,
        )

    def _nb_align_timed_out(self, timestamp_ns: int) -> bool:
        """航向连续对不准超过配置时长时保守停车，避免持续旋转。"""

        started = self._nb_align_started_ns
        if started is None:
            return False
        timeout_ns = self._seconds_to_ns(self.config.nb_opening_align_timeout_s)
        return timestamp_ns - started >= timeout_ns

    def _step_nb_to_first(self, timestamp_ns: int) -> MatchDecision:
        """开场第一段：直线移动到第一航点（夹爪保持闭合）。"""

        return self._nb_drive_leg(
            timestamp_ns,
            target=self.config.nb_opening_first_target_field,
            speed_m_s=self.config.nb_opening_first_speed_m_s,
            reverse=False,
            gripper_open=False,
            next_state=MatchState.NB_OPENING_GRIPPER_OPEN,
            drive_reason="nb_opening_to_first",
            align_reason="nb_opening_to_first_align",
            arrive_reason="nb_opening_first_reached_open_gripper",
        )

    def _step_nb_gripper_open(self, timestamp_ns: int) -> MatchDecision:
        """开场张爪到指定左右角度，随后进入第二段直线。"""

        decision = self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "nb_opening_gripper_opened",
            posture=GripperPosture.OPEN,
            gripper_angles_deg=self._nb_gripper_angles_deg(),
        )
        self._nb_reset_leg()
        self.state = MatchState.NB_OPENING_TO_SECOND
        return decision

    def _step_nb_to_second(self, timestamp_ns: int) -> MatchDecision:
        """开场第二段：直线向前移动到第二航点。"""

        return self._nb_drive_leg(
            timestamp_ns,
            target=self.config.nb_opening_second_target_field,
            speed_m_s=self.config.nb_opening_second_speed_m_s,
            reverse=False,
            gripper_open=True,
            next_state=MatchState.NB_OPENING_REVERSE,
            drive_reason="nb_opening_to_second",
            align_reason="nb_opening_to_second_align",
            arrive_reason="nb_opening_second_reached_start_reverse",
        )

    def _step_nb_reverse(self, timestamp_ns: int) -> MatchDecision:
        """开场第三段：倒车回退到回退航点，停稳后进入正常解团流程。"""

        decision = self._nb_drive_leg(
            timestamp_ns,
            target=self.config.nb_opening_reverse_target_field,
            speed_m_s=self.config.nb_opening_reverse_speed_m_s,
            reverse=True,
            gripper_open=True,
            next_state=MatchState.SEARCH_CLUSTER,
            drive_reason="nb_opening_reverse",
            align_reason="nb_opening_reverse_align",
            arrive_reason="nb_opening_reverse_reached_start_search",
        )
        if decision.state is MatchState.SEARCH_CLUSTER:
            self._begin_cluster_search()
        return decision


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the no-breakup match variant with a configurable straight-line "
            "waypoint opening instead of the fixed startup turn+forward."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--start-area",
        choices=(MatchStartArea.AREA_2.value,),
        default=MatchStartArea.AREA_2.value,
        help="本入口仅支持启动区域 2（地图右上角/红方）；开场航点按区域 2 坐标书写。",
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
        sequence_factory=MatchNBSequence.from_app_config,
        mode_name="match_nb",
        log_file_prefix="match_nb_",
        preview_title="Match-NB perception",
    )


__all__ = [
    "MatchDecision",
    "MatchNBSequence",
    "MatchState",
    "MatchPreflight",
]


if __name__ == "__main__":
    main()
