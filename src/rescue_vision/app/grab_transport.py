"""正式流程的独立绿色物块推送—运输测试流程。

该入口复用 ``MatchSequence`` 的绿色目标确认、对准、夹取、安全区运输和退出逻辑，
只在安全区末端保持夹爪张开推入物块；不会执行目标团解团。车辆从场地
``FieldPoint(0, 0)``、场地航向 ``+90°`` 开始，搜索到满足正式流程单物块门禁的
绿色普通物资后直接进入运输流程；进入 450 mm 近场后复用近场宽度抓取的选组、动态开度
和定距动作，适合在已把物资摆散的场地上单独联调末端推送和运输。
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
    MatchState,
)
from rescue_vision.app.match_runtime import _run_hardware
from rescue_vision.app.near_field_grasp import NearFieldGraspPolicy
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass

if TYPE_CHECKING:
    from rescue_vision.config import AppConfig
    from rescue_vision.tracking import TrackedTarget


class GrabTransportSequence(
    MatchSequence
):
    """只搜索绿色单物块并在运输末端张爪推送的纯逻辑流程。

    前段夹取和运输沿用父类；仅覆盖安全区末端的闭爪推进姿态。
    """

    _greedy_pickup_enabled = False

    _first_green_blocked_routes_to_breakup = False
    # This entry is a single-green transport bench test; it must never enter
    # the formal match's dynamic breakup planner.
    _dynamic_breakup_enabled = False

    INITIAL_FIELD_POSITION: FieldPoint = FieldPoint(0.0, 0.0)
    INITIAL_HEADING_RAD: float = math.pi / 2.0

    _FINAL_PUSH_OPEN_PHASES = frozenset(
        {
            "closing_before_final_forward",
            "forward_final_open",
            "stopping_before_exit_opening",
            "opening_after_transport",
        }
    )

    @property
    def near_field_policy(self) -> NearFieldGraspPolicy:
        """联调入口始终只允许单个绿色普通物资。"""

        return NearFieldGraspPolicy(frozenset((TargetClass.GREEN_SUPPLY,)), 1)

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
    ) -> MatchDecision:
        """只在安全区末端推入和推入后的退出阶段保持张开。"""

        if self._safe_zone_phase in self._FINAL_PUSH_OPEN_PHASES:
            posture = GripperPosture.OPEN
        return super()._decision(
            timestamp_ns,
            linear,
            angular,
            reason,
            posture=posture,
            gripper_angles_deg=gripper_angles_deg,
            soft_brake=soft_brake,
        )

    def _step_transport_release(self, timestamp_ns: int) -> MatchDecision:
        """末端推入不再重复闭爪，直接保持张开进入最终直线段。"""

        if self._safe_zone_phase == "closing_before_final_forward":
            self._gripper_phase_started_ns = None
            self._safe_zone_phase = "forward_final_open"
            self.state = MatchState.TRANSPORT_FORWARD
            self._begin_action_settle(
                timestamp_ns,
                "safe_before_final_forward",
            )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                "gripper_open_start_final_push",
                posture=GripperPosture.OPEN,
            )
        return super()._step_transport_release(timestamp_ns)

    def _gripper_action_completed(self, timestamp_ns: int) -> bool:
        """联调流程发送舵机指令后立即进入下一步，不等待机械行程。"""

        del timestamp_ns
        return True

    @classmethod
    def from_app_config(
        cls,
        config: AppConfig,
    ) -> GrabTransportSequence:
        """从正式流程配置装配测试流程，并固定覆盖测试初始场地位置。"""

        sequence = super().from_app_config(config)
        if not sequence.config.opportunistic_single_green_enabled:
            raise RuntimeError(
                "grab-transport test requires "
                "match.opportunistic_single_green_enabled=true."
            )
        # 这里故意只覆盖该测试流程的受限航位初值；相机、运动和标定仍完全
        # 来自同一份 runtime 配置，避免复制第二套机械/几何参数。
        sequence._initial_field_position = cls.INITIAL_FIELD_POSITION
        sequence._fallback_field_position = cls.INITIAL_FIELD_POSITION
        return sequence

    def start(self, timestamp_ns: int) -> MatchDecision:
        """跳过固定启动动作，预检通过后立即进入绿色目标搜索。"""

        self._validate_timestamp(timestamp_ns)
        if self.state is not MatchState.PREFLIGHT:
            raise RuntimeError("start() requires a successful PREFLIGHT.")
        self._started = True
        self._reset_straight_pid()
        self._fallback_field_position = self._initial_field_position
        self._fallback_last_distance_m = None
        if self._near_field_pickup is not None:
            self._near_field_pickup.reset()
            self._near_field_session_id = 0
        self._begin_safe_zone_scan()
        self._begin_cluster_search()
        self.state = MatchState.SEARCH_CLUSTER
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            "direct_search_transport_started",
        )

    def _cluster_ground_measurement(self, timestamp_ns: int) -> None:
        """禁用解团候选；``SEARCH_CLUSTER`` 仅保留绿色目标搜索语义。"""

        del timestamp_ns
        return None

    def _transport_group_size(
        self,
        target: TrackedTarget,
        timestamp_ns: int,
        *,
        reference: GroundPoint | None = None,
    ) -> int | None:
        """独立联调入口继续要求整条固定走廊内只有一个绿色物块。"""

        size = super()._transport_group_size(
            target,
            timestamp_ns,
            reference=reference,
        )
        if size is None:
            return None
        for other in self._tracker.tracks:
            if (
                other.track_id == target.track_id
                or not self._target_is_fresh(other, timestamp_ns)
                or other.target_class is not TargetClass.GREEN_SUPPLY
                or other.ground_point is None
            ):
                continue
            if self._point_in_transport_corridor(other.ground_point):
                return None
        return 1


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the standalone formal-flow green grab-and-transport test "
            "from FieldPoint(0, 0) at +90 degrees."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
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
        sequence_factory=(
            GrabTransportSequence.from_app_config
        ),
        initial_field_position=(
            GrabTransportSequence.INITIAL_FIELD_POSITION
        ),
        initial_heading_rad=(
            GrabTransportSequence.INITIAL_HEADING_RAD
        ),
        mode_name="grab_transport",
        log_file_prefix="match_grab_transport_",
        preview_title="Grab-transport perception",
    )


__all__ = [
    "MatchDecision",
    "GrabTransportSequence",
    "MatchState",
    "MatchPreflight",
]


if __name__ == "__main__":
    main()
