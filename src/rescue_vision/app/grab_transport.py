"""正式流程的独立绿色夹取—运输测试流程。

该入口复用 ``MatchSequence`` 的绿色目标确认、对准、夹取、
安全区交付和退出逻辑，但不会执行目标团解团。车辆从场地
``FieldPoint(0, 0)``、场地航向 ``+90°`` 开始，搜索到满足正式流程单物块门禁的
绿色普通物资后直接进入运输流程；到达抓取偏移后仍按正式流程配置停车复核相邻绿块、
重复对准纳入和最多携带数量，适合在已把物资摆散的场地上单独联调夹爪和运输。
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import TYPE_CHECKING

from rescue_vision.app.match import (
    MatchDecision,
    MatchPreflight,
    MatchSequence,
    MatchState,
)
from rescue_vision.app.match_runtime import _run_hardware
from rescue_vision.geometry.types import FieldPoint

if TYPE_CHECKING:
    from rescue_vision.config import AppConfig


class GrabTransportSequence(
    MatchSequence
):
    """只搜索绿色单物块并执行正式流程夹取—运输的纯逻辑流程。"""

    INITIAL_FIELD_POSITION: FieldPoint = FieldPoint(0.0, 0.0)
    INITIAL_HEADING_RAD: float = math.pi / 2.0

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
