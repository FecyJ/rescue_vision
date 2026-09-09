"""识别单个绿色物资、开夹爪接近并合爪抓取的简化真车入口。

本入口面向“只有内参、尚无地面标定”的早期联调：绿色目标由现有
``TargetPoseDetector``（Hailo YOLO Pose v3）识别，导航退化为去畸变像素空间的
“居中 + 前进”，不把像素伪造成 ``GroundPoint``。绿色确认后打开夹爪，框下边进入
图像底部比例阈值后停车合爪。这不是比赛程序，仍需物理急停与全程监督。
"""

from __future__ import annotations

import argparse
import math
import signal
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from threading import Thread

from rescue_vision.app.cluster_breakup import CameraPerceptionPump, GripperPosture
from rescue_vision.config import GreenGrabRuntimeConfig, load_runtime_config
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    MotionController,
    MotionSynchronizationError,
)
from rescue_vision.perception import (
    PerceptionFrameRenderer,
    PerceptionSnapshot,
    TargetClass,
    TargetObservation,
)

# 启动期间 SOFT_BRAKE 同步的短暂重试窗口；急停锁存从不重试。
_PREFLIGHT_RETRY_WINDOW_NS = 5_000_000_000


class GreenGrabState(str, Enum):
    """绿色抓取的简化流程状态；不替代 mission 包的规则阶段。"""

    SEARCH = "search"
    ALIGN = "align"
    APPROACH = "approach"
    GRAB = "grab"
    DONE = "done"


@dataclass(frozen=True, slots=True)
class GreenPixelTarget:
    """绿色物资的去畸变像素级测量；无地面标定时用于居中接近。"""

    center_u: float
    bottom_v: float
    image_width: int
    image_height: int

    def __post_init__(self) -> None:
        for name in ("center_u", "bottom_v"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite, got {value!r}.")
        if (
            isinstance(self.image_width, bool)
            or not isinstance(self.image_width, int)
            or self.image_width <= 0
            or isinstance(self.image_height, bool)
            or not isinstance(self.image_height, int)
            or self.image_height <= 0
        ):
            raise ValueError(
                "image_width and image_height must be positive integers."
            )
        if not 0.0 <= self.center_u <= self.image_width:
            raise ValueError(
                f"center_u must be within [0, {self.image_width}], "
                f"got {self.center_u!r}."
            )
        if not 0.0 <= self.bottom_v <= self.image_height:
            raise ValueError(
                f"bottom_v must be within [0, {self.image_height}], "
                f"got {self.bottom_v!r}."
            )


@dataclass(frozen=True, slots=True)
class GreenGrabDecision:
    """单个控制周期的轻量意图：差速 twist 与夹爪姿态。"""

    timestamp_ns: int
    state: GreenGrabState
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    gripper_posture: GripperPosture
    reason: str

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if not isinstance(self.state, GreenGrabState):
            raise ValueError("state must be a GreenGrabState.")
        for name in ("linear_velocity_m_s", "angular_velocity_rad_s"):
            value = float(getattr(self, name))
            if not math.isfinite(value):
                raise ValueError(f"{name} must be finite.")
        if not isinstance(self.gripper_posture, GripperPosture):
            raise ValueError("gripper_posture must be a GripperPosture.")
        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string.")


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return min(max(value, minimum), maximum)


def _horizontal_error_ratio(target: GreenPixelTarget) -> float:
    """目标水平中心相对半幅图像的比例误差；正值为目标在图像右侧。"""

    half = target.image_width * 0.5
    return (target.center_u - half) / half


def _green_pixel_target(observation: TargetObservation) -> GreenPixelTarget:
    """从任务观测的检测框提取像素级中心与底边；不构造 ``GroundPoint``。"""

    return GreenPixelTarget(
        center_u=0.5 * (observation.box.x_min + observation.box.x_max),
        bottom_v=observation.box.y_max,
        image_width=observation.image_size[0],
        image_height=observation.image_size[1],
    )


def find_green_target(
    snapshot: PerceptionSnapshot | None,
) -> GreenPixelTarget | None:
    """从最新感知快照提取第一个绿色物资；无快照、过期或无色返回 ``None``。"""

    if snapshot is None or snapshot.dropped_stale_age_ms is not None:
        return None
    for observation in snapshot.observations:
        if observation.target_class is TargetClass.GREEN_SUPPLY:
            return _green_pixel_target(observation)
    return None


class GreenGrabSequence:
    """可重放的绿色抓取简化状态机，只读取已完成快照并返回轻量意图。

    ``step()`` 不等待相机、推理、UART 或网络；车端入口负责把这些意图交给
    同一个 ``MotionController``，并在旁路故障时走停车路径。角速度左转为正。
    """

    def __init__(
        self,
        config: GreenGrabRuntimeConfig,
        *,
        gripper_full_travel_time_s: float,
    ) -> None:
        if not isinstance(config, GreenGrabRuntimeConfig):
            raise TypeError("config must be a GreenGrabRuntimeConfig.")
        if not config.enabled:
            raise ValueError("GreenGrabSequence requires enabled config.")
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
        self._gripper_full_travel_time_ns = round(
            float(gripper_full_travel_time_s) * 1_000_000_000
        )
        self.state = GreenGrabState.SEARCH
        self._last_timestamp_ns: int | None = None
        self._green_confirm_count = 0
        self._last_green_seen_ns: int | None = None
        self._grab_started_ns: int | None = None

    def step(
        self,
        timestamp_ns: int,
        green: GreenPixelTarget | None,
    ) -> GreenGrabDecision:
        """消费一个已完成快照对应的绿色目标并返回本周期控制意图。"""

        self._validate_timestamp(timestamp_ns)
        if green is not None:
            if not isinstance(green, GreenPixelTarget):
                raise TypeError("green must be a GreenPixelTarget or None.")
            self._last_green_seen_ns = timestamp_ns

        if self.state is GreenGrabState.DONE:
            return self._decision(timestamp_ns, 0.0, 0.0, GripperPosture.CLOSED, "done")
        if self.state is GreenGrabState.SEARCH:
            return self._step_search(timestamp_ns, green)
        if self.state is GreenGrabState.ALIGN:
            return self._step_align(timestamp_ns, green)
        if self.state is GreenGrabState.APPROACH:
            return self._step_approach(timestamp_ns, green)
        if self.state is GreenGrabState.GRAB:
            return self._step_grab(timestamp_ns)
        return self._decision(
            timestamp_ns,
            0.0,
            0.0,
            GripperPosture.CLOSED,
            f"unhandled_state:{self.state.value}",
        )

    def _step_search(
        self,
        timestamp_ns: int,
        green: GreenPixelTarget | None,
    ) -> GreenGrabDecision:
        if green is not None:
            self._green_confirm_count += 1
            if self._green_confirm_count >= self.config.confirm_frames:
                self.state = GreenGrabState.ALIGN
                return self._decision(
                    timestamp_ns,
                    0.0,
                    0.0,
                    GripperPosture.OPEN,
                    "green_confirmed_open_and_align",
                )
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                GripperPosture.CLOSED,
                "confirming_green",
            )
        self._green_confirm_count = 0
        return self._decision(
            timestamp_ns,
            0.0,
            self.config.search_angular_velocity_rad_s,
            GripperPosture.CLOSED,
            "search_green",
        )

    def _step_align(
        self,
        timestamp_ns: int,
        green: GreenPixelTarget | None,
    ) -> GreenGrabDecision:
        if green is None:
            return self._lost_target(timestamp_ns)
        self._green_confirm_count = 0
        ratio = _horizontal_error_ratio(green)
        if abs(ratio) <= self.config.align_tolerance_ratio:
            self.state = GreenGrabState.APPROACH
            return self._decision(
                timestamp_ns, 0.0, 0.0, GripperPosture.OPEN, "aligned_approach"
            )
        return self._decision(
            timestamp_ns,
            0.0,
            self._align_angular(ratio),
            GripperPosture.OPEN,
            "align_green",
        )

    def _step_approach(
        self,
        timestamp_ns: int,
        green: GreenPixelTarget | None,
    ) -> GreenGrabDecision:
        if green is None:
            return self._lost_target(timestamp_ns)
        self._green_confirm_count = 0
        if green.bottom_v >= self.config.engage_bottom_fraction * green.image_height:
            self.state = GreenGrabState.GRAB
            self._grab_started_ns = timestamp_ns
            return self._decision(
                timestamp_ns,
                0.0,
                0.0,
                GripperPosture.CLOSED,
                "engage_close_gripper",
            )
        return self._decision(
            timestamp_ns,
            self.config.approach_speed_m_s,
            self._align_angular(_horizontal_error_ratio(green)),
            GripperPosture.OPEN,
            "approach_green",
        )

    def _step_grab(self, timestamp_ns: int) -> GreenGrabDecision:
        assert self._grab_started_ns is not None
        if timestamp_ns - self._grab_started_ns < self._gripper_full_travel_time_ns:
            return self._decision(
                timestamp_ns, 0.0, 0.0, GripperPosture.CLOSED, "holding_grab"
            )
        self.state = GreenGrabState.DONE
        return self._decision(
            timestamp_ns, 0.0, 0.0, GripperPosture.CLOSED, "grab_complete"
        )

    def _lost_target(self, timestamp_ns: int) -> GreenGrabDecision:
        """对齐/接近中短暂丢失目标先保持，超时后回到搜索并合爪。"""

        if self._last_green_seen_ns is None:
            self._reset_to_search()
            return self._decision(
                timestamp_ns,
                0.0,
                self.config.search_angular_velocity_rad_s,
                GripperPosture.CLOSED,
                "search_no_green",
            )
        lost_ms = (timestamp_ns - self._last_green_seen_ns) / 1_000_000.0
        if lost_ms >= self.config.target_loss_timeout_ms:
            self._reset_to_search()
            return self._decision(
                timestamp_ns,
                0.0,
                self.config.search_angular_velocity_rad_s,
                GripperPosture.CLOSED,
                "target_lost_resume_search",
            )
        return self._decision(
            timestamp_ns, 0.0, 0.0, GripperPosture.OPEN, "target_temporarily_lost"
        )

    def _reset_to_search(self) -> None:
        self.state = GreenGrabState.SEARCH
        self._green_confirm_count = 0

    def _align_angular(self, horizontal_error_ratio: float) -> float:
        # 图像右侧为正误差；机器人需右转，项目角速度右转为负。
        requested = -self.config.align_kp_rad_s * horizontal_error_ratio
        maximum = self.config.align_max_angular_velocity_rad_s
        return _clamp(requested, -maximum, maximum)

    def _validate_timestamp(self, timestamp_ns: int) -> None:
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if (
            self._last_timestamp_ns is not None
            and timestamp_ns < self._last_timestamp_ns
        ):
            raise ValueError(
                f"timestamp_ns moved backwards from {self._last_timestamp_ns} "
                f"to {timestamp_ns}."
            )
        self._last_timestamp_ns = timestamp_ns

    def _decision(
        self,
        timestamp_ns: int,
        linear: float,
        angular: float,
        posture: GripperPosture,
        reason: str,
    ) -> GreenGrabDecision:
        return GreenGrabDecision(
            timestamp_ns=timestamp_ns,
            state=self.state,
            linear_velocity_m_s=float(linear),
            angular_velocity_rad_s=float(angular),
            gripper_posture=posture,
            reason=reason,
        )


def _run_hardware(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
) -> None:
    # 本地导入，避免测试收集阶段访问相机、Hailo 或串口。
    from rescue_vision.app.manual_capture import build_camera_pipeline

    config = load_runtime_config(config_path)
    if not config.green_grab.enabled:
        raise RuntimeError("green_grab.enabled must be true.")
    if not supervised_stop_ready:
        raise RuntimeError(
            "A physical emergency stop and continuous supervision are required "
            "until the STM32 watchdog has been verified."
        )
    if not config.hailo.enabled:
        raise RuntimeError("green_grab requires hailo.enabled with a deployed v3 model.")
    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("green_grab requires enabled uart.")
    controller = config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("green_grab requires enabled motion.")
    gripper = config.motion.gripper.build_calibration()
    if gripper is None:
        raise RuntimeError("green_grab requires enabled gripper calibration.")

    pipeline = build_camera_pipeline(config)
    renderer = PerceptionFrameRenderer(
        lambda: config.build_target_pose_detector(
            ground_projector=pipeline.ground_projector
        ),
        render_enabled=False,
    )
    camera_pump = CameraPerceptionPump(pipeline.source, pipeline.prepare, renderer)
    sequence = GreenGrabSequence(
        config.green_grab,
        gripper_full_travel_time_s=gripper.full_travel_time_s,
    )

    stop_requested = False
    latest_status: CarSystemStatus | None = None
    latest_snapshot: PerceptionSnapshot | None = None
    camera_start_thread: Thread | None = None
    camera_pump_started = False
    last_posture: GripperPosture | None = None
    last_state: GreenGrabState | None = None
    next_progress_ns = 0

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal stop_requested
        stop_requested = True

    def consume(message: object) -> None:
        nonlocal latest_status
        if isinstance(message, CarSystemStatus):
            latest_status = message

    def service_uart_during_camera_startup() -> None:
        nonlocal latest_status
        controller.update(now_ns=time.monotonic_ns())
        for message in controller.drain_messages():
            if isinstance(message, CarSystemStatus):
                latest_status = message
                if message.emergency_stop_latched:
                    raise RuntimeError(
                        "STM32 emergency stop is latched during camera startup."
                    )

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, request_stop)
    try:
        with channel:
            try:
                camera_start_thread = camera_pump.start_in_background()
                sync_deadline_ns = time.monotonic_ns() + _PREFLIGHT_RETRY_WINDOW_NS
                while True:
                    try:
                        controller.synchronize(
                            timeout_s=config.motion.synchronization_timeout_s,
                            on_message=consume,
                        )
                        break
                    except MotionSynchronizationError:
                        if (
                            latest_status is not None
                            and latest_status.emergency_stop_latched
                        ):
                            raise RuntimeError(
                                "STM32 emergency stop is latched during camera startup."
                            )
                        if time.monotonic_ns() >= sync_deadline_ns:
                            raise
                        time.sleep(0.02)
                camera_pump.wait_until_started(
                    camera_start_thread,
                    on_wait=service_uart_during_camera_startup,
                )
                camera_pump_started = True
                controller.query_state()
                while latest_snapshot is None and not stop_requested:
                    controller.update(now_ns=time.monotonic_ns())
                    for message in controller.drain_messages():
                        consume(message)
                    camera_pump.check_health()
                    latest_snapshot = renderer.latest_fresh_snapshot(
                        time.monotonic_ns(),
                        config.processing.max_observation_age_ms,
                    )
                    time.sleep(0.005)
                if latest_snapshot is None:
                    raise RuntimeError("No fresh perception snapshot before start.")

                while not stop_requested:
                    now_ns = time.monotonic_ns()
                    if controller.needs_synchronization:
                        controller.synchronize(
                            timeout_s=config.motion.synchronization_timeout_s,
                            on_message=consume,
                        )
                    controller.update(now_ns=now_ns)
                    for message in controller.drain_messages():
                        if isinstance(message, CarCommandReply):
                            if message.result is CommandResult.SEQUENCE_OLD:
                                continue
                            if message.result is not CommandResult.ACCEPTED:
                                raise RuntimeError(
                                    "STM32 rejected command "
                                    f"{message.command_type.name.lower()}: "
                                    f"{message.result.name.lower()}."
                                )
                        elif isinstance(message, CarSystemStatus):
                            latest_status = message
                            if message.emergency_stop_latched:
                                raise RuntimeError(
                                    "STM32 emergency stop is latched."
                                )
                            if (
                                not message.protocol_ready
                                or message.reply_queue_full
                                or message.tx_degraded
                            ):
                                raise RuntimeError(
                                    "STM32 UART health is degraded; "
                                    f"protocol_ready={message.protocol_ready}, "
                                    f"reply_queue_full={message.reply_queue_full}, "
                                    f"tx_degraded={message.tx_degraded}."
                                )
                    camera_pump.check_health()
                    latest_snapshot = renderer.latest_fresh_snapshot(
                        now_ns,
                        config.processing.max_observation_age_ms,
                    )
                    green = find_green_target(latest_snapshot)
                    decision = sequence.step(now_ns, green)
                    if decision.gripper_posture is not last_posture:
                        if decision.gripper_posture is GripperPosture.OPEN:
                            angles = (
                                gripper.open_left_angle_deg,
                                gripper.open_right_angle_deg,
                            )
                        else:
                            angles = (
                                gripper.closed_left_angle_deg,
                                gripper.closed_right_angle_deg,
                            )
                        controller.set_gripper_angles(*angles)
                        last_posture = decision.gripper_posture
                    controller.drive_wheel_limited(
                        decision.linear_velocity_m_s,
                        decision.angular_velocity_rad_s,
                    )
                    if decision.state is not last_state or now_ns >= next_progress_ns:
                        status_text = (
                            "status=none"
                            if latest_status is None
                            else (
                                "status=("
                                f"motor_output={latest_status.motor_output_enabled},"
                                f"watchdog={latest_status.watchdog_armed},"
                                f"estop={latest_status.emergency_stop_latched},"
                                f"stop_reason={latest_status.stop_reason.name.lower()})"
                            )
                        )
                        target_text = (
                            "green=none"
                            if green is None
                            else (
                                f"green=(center_u={green.center_u:.1f},"
                                f"bottom_v={green.bottom_v:.1f},"
                                f"ratio={_horizontal_error_ratio(green):+.3f})"
                            )
                        )
                        print(
                            f"state={decision.state.value} reason={decision.reason} "
                            f"{target_text} "
                            f"linear={decision.linear_velocity_m_s:.3f} "
                            f"angular={decision.angular_velocity_rad_s:.3f} "
                            f"posture={decision.gripper_posture.value} "
                            f"target_wheel_m_s={controller.target_wheel_speeds_m_s} "
                            f"commanded_wheel_m_s={controller.commanded_wheel_speeds_m_s} "
                            f"{status_text}",
                            flush=True,
                        )
                        last_state = decision.state
                        next_progress_ns = now_ns + 1_000_000_000
                    if decision.state is GreenGrabState.DONE:
                        break
                    time.sleep(0.005)
                controller.soft_brake()
            finally:
                try:
                    controller.soft_brake()
                finally:
                    if camera_start_thread is not None or camera_pump_started:
                        camera_pump.stop()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Detect a green supply block, open the gripper, approach and close."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help=(
            "Confirm a physical emergency stop and continuous test supervision "
            "while the firmware watchdog is unavailable."
        ),
    )
    args = parser.parse_args()
    _run_hardware(
        args.config.expanduser().resolve(),
        supervised_stop_ready=args.supervised_physical_stop_ready,
    )


if __name__ == "__main__":
    main()
