"""键盘操控车辆：前进/后退/转弯 + 实时相机与模型结果回传 + Esc 退出。

本脚本是赛外受监督手动驾驶的人工验收入口，不是正式比赛应用。它把
``TargetPoseDetector``（Hailo YOLO Pose v3）的取帧、去畸变和推理放在独立
旁路线程（``CameraPerceptionPump`` + ``PerceptionFrameRenderer``），主线程
只做三件轻量工作：非阻塞读取键盘、刷新轮速目标并排空 UART 回传、读取最新
推理快照并打印/显示。因此慢取帧、慢去畸变或慢推理不会拖慢运动安全循环。

键盘从终端读取（通过 SSH 也有效）；``--display`` 可选地在本地监视器上弹出
OpenCV 叠加画面。方向键和 WASD 等价：

    W / ↑      前进          S / ↓      后退
    A / ←      原地左转       D / →      原地右转
    Z          夹爪打开       C          夹爪关闭
    X          夹爪运输姿态   空格        松手停车
    Esc / q    退出

按住前进与转向可组合成弧线（例如 ``W`` + ``D`` 为右前弧）。所有轮速仍受
``motion`` 配置的线速度、角速度和单轮加速度上限约束，超限请求会被拒绝。
夹爪键（``Z``/``X``/``C``）只发送一次已标定角度，不进入持续推进状态；
``X`` 的运输姿态还需要配置 ``motion.gripper.transport_*_angle_deg``。

运行前必须在 ``runtime.yaml`` 中启用 ``uart``、``motion`` 和 ``hailo``，并
确认物理急停可立即触发、操作员全程监督。夹爪控制需要 ``motion.gripper``
已标定并启用。
"""

from __future__ import annotations

import argparse
import math
import os
import select
import signal
import sys
import termios
import time
import tty
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

from rescue_vision.config import load_runtime_config
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    GripperCalibration,
    MotionSynchronizationError,
    OdometryImu,
    SensorFlags,
)

if TYPE_CHECKING:
    from rescue_vision.localization import FusedPoseEstimate
    from rescue_vision.perception import PerceptionSnapshot

# 启动期间 SOFT_BRAKE 同步的短暂重试窗口；急停锁存从不重试。
_PREFLIGHT_RETRY_WINDOW_NS = 5_000_000_000
# 运动循环周期；足够小以稳定捕获按键自动重复，又不额外占用 CPU。
_LOOP_PERIOD_S = 0.005
# 终端状态打印周期（5 Hz）。
_STATUS_PERIOD_NS = 200_000_000
# 按键自动重复时，驱动按键保持“按住”的最近事件超时。
_HOLD_TIMEOUT_NS = 200_000_000
# 判断孤立 Esc 是否后跟 CSI 方向键序列的短暂等待。
_ESC_LOOKAHEAD_S = 0.03

# 驱动键（水平触发，依赖按键自动重复维持“按住”）。
_DRIVE_KEYS = ("forward", "backward", "turn_left", "turn_right")

# 夹爪键（边沿触发，每次按下发送一次已标定角度）。
_GRIPPER_ACTIONS = {
    "gripper_open": "open",
    "gripper_transport": "transport",
    "gripper_close": "close",
}

# 单字节按键映射。WASD 为项目默认，方向键通过 CSI 序列识别。
_SINGLE_KEYS = {
    ord("w"): "forward",
    ord("W"): "forward",
    ord("s"): "backward",
    ord("S"): "backward",
    ord("a"): "turn_left",
    ord("A"): "turn_left",
    ord("d"): "turn_right",
    ord("D"): "turn_right",
    ord("z"): "gripper_open",
    ord("Z"): "gripper_open",
    ord("x"): "gripper_transport",
    ord("X"): "gripper_transport",
    ord("c"): "gripper_close",
    ord("C"): "gripper_close",
    ord(" "): "stop",
    ord("q"): "quit",
    ord("Q"): "quit",
    0x03: "quit",  # Ctrl+C，防御性处理（cbreak 通常已转成 SIGINT）。
}

# CSI 方向键序列的终结字节：ESC [ A/B/C/D。
_CSI_KEYS = {
    ord("A"): "forward",  # Up
    ord("B"): "backward",  # Down
    ord("C"): "turn_right",  # Right
    ord("D"): "turn_left",  # Left
}


def decode_keys(buffer: bytes) -> tuple[list[str], bytes]:
    """把原始终端字节解析成驱动键事件。

    返回 ``(events, leftover)``：``leftover`` 保留末尾不完整的转义序列，
    调用方应把它拼接到下一次轮询读到的字节前。孤立 ``Esc``（未跟 ``[``）
    作为 ``"quit"``；``Esc [ X`` 中未知的 X 会被跳过。
    """

    if not isinstance(buffer, (bytes, bytearray)):
        raise TypeError(f"buffer must be bytes, got {type(buffer).__name__}.")
    events: list[str] = []
    index = 0
    length = len(buffer)
    while index < length:
        byte = buffer[index]
        if byte == 0x1B:
            if index + 1 >= length:
                # 末尾孤立 Esc，可能是 CSI 前缀，留给下次轮询判定。
                return events, buffer[index:]
            if buffer[index + 1] == ord("["):
                if index + 2 >= length:
                    return events, buffer[index:]  # 不完整的 CSI 序列
                key = _CSI_KEYS.get(buffer[index + 2])
                if key is not None:
                    events.append(key)
                index += 3
                continue
            # Esc 后不是 '['，视为单独 Esc（退出）。
            events.append("quit")
            index += 1
            continue
        key = _SINGLE_KEYS.get(byte)
        if key is not None:
            events.append(key)
        index += 1
    return events, b""


def compute_twist(
    active_keys: set[str] | frozenset[str],
    *,
    linear_speed_m_s: float,
    angular_speed_rad_s: float,
) -> tuple[float, float]:
    """把当前按住的驱动键换算成车体 ``(线速度, 角速度)``。

    前进/后退互斥，左/右转互斥；同时按下相对的两个键时该轴归零。角速度
    左转为正，符合 ``MotionController.drive`` 的项目约定。
    """

    forward = "forward" in active_keys
    backward = "backward" in active_keys
    turn_left = "turn_left" in active_keys
    turn_right = "turn_right" in active_keys
    linear = 0.0
    angular = 0.0
    if forward and not backward:
        linear = float(linear_speed_m_s)
    elif backward and not forward:
        linear = -float(linear_speed_m_s)
    if turn_left and not turn_right:
        angular = float(angular_speed_rad_s)
    elif turn_right and not turn_left:
        angular = -float(angular_speed_rad_s)
    return linear, angular


def _command_label(active_keys: set[str] | frozenset[str]) -> str:
    """把当前按住的驱动键概括成一行命令名，便于终端显示。"""

    forward = "forward" in active_keys
    backward = "backward" in active_keys
    turn_left = "turn_left" in active_keys
    turn_right = "turn_right" in active_keys
    parts: list[str] = []
    if forward and not backward:
        parts.append("forward")
    elif backward and not forward:
        parts.append("backward")
    if turn_left and not turn_right:
        parts.append("left")
    elif turn_right and not turn_left:
        parts.append("right")
    return "-".join(parts) if parts else "stop"


def gripper_angles_for(
    action: str,
    calibration: GripperCalibration,
) -> tuple[float, float]:
    """把夹爪动作映射为标定后的左右舵机角度（degree）。

    ``action`` 取 ``"open"``、``"transport"`` 或 ``"close"``；运输姿态未配置
    时对 ``"transport"`` 抛出 ``ValueError``。
    """

    if not isinstance(calibration, GripperCalibration):
        raise TypeError("calibration must be a GripperCalibration.")
    if action == "open":
        return (
            calibration.open_left_angle_deg,
            calibration.open_right_angle_deg,
        )
    if action == "close":
        return (
            calibration.closed_left_angle_deg,
            calibration.closed_right_angle_deg,
        )
    if action == "transport":
        angles = calibration.transport_angles_deg
        if angles is None:
            raise ValueError(
                "transport gripper posture is not configured; set "
                "motion.gripper.transport_left_angle_deg and "
                "transport_right_angle_deg."
            )
        return angles
    raise ValueError(
        f"unknown gripper action {action!r}; expected open, transport or close."
    )


class KeyDriveState:
    """按键事件到“按住状态”的有状态转换。

    驱动键依赖终端按键自动重复维持按住：只要事件在 ``hold_timeout_ns`` 内
    持续到达，就认为仍在按住；停止重复（松键）后自动失效。``stop``、
    ``quit`` 和夹爪键是边沿触发的单次标志，通过 ``consume_*`` 消费。
    """

    def __init__(self, *, hold_timeout_ns: int = _HOLD_TIMEOUT_NS) -> None:
        if (
            isinstance(hold_timeout_ns, bool)
            or not isinstance(hold_timeout_ns, int)
            or hold_timeout_ns < 0
        ):
            raise ValueError(
                f"hold_timeout_ns must be a non-negative integer, "
                f"got {hold_timeout_ns!r}."
            )
        self._hold_timeout_ns = hold_timeout_ns
        self._last_seen_ns: dict[str, int] = {}
        self._stop_pending = False
        self._quit_pending = False
        self._gripper_pending: str | None = None

    def apply(self, events: Sequence[str], now_ns: int) -> None:
        """记录本轮解码出的键事件；``now_ns`` 必须单调不减。"""

        for event in events:
            if event == "stop":
                self._stop_pending = True
            elif event == "quit":
                self._quit_pending = True
            elif event in _DRIVE_KEYS:
                self._last_seen_ns[event] = now_ns
            elif event in _GRIPPER_ACTIONS:
                self._gripper_pending = _GRIPPER_ACTIONS[event]

    def active_keys(self, now_ns: int) -> frozenset[str]:
        """返回仍在按住超时窗口内的驱动键。"""

        return frozenset(
            key
            for key, seen_ns in self._last_seen_ns.items()
            if now_ns - seen_ns <= self._hold_timeout_ns
        )

    def consume_stop(self) -> bool:
        """消费一次停车请求，并清除当前所有驱动键。"""

        if not self._stop_pending:
            return False
        self._stop_pending = False
        self._last_seen_ns.clear()
        return True

    def consume_quit(self) -> bool:
        """消费一次退出请求。"""

        if not self._quit_pending:
            return False
        self._quit_pending = False
        return True

    def consume_gripper(self) -> str | None:
        """消费一次夹爪请求，返回 ``"open"``/``"transport"``/``"close"``。"""

        action = self._gripper_pending
        self._gripper_pending = None
        return action


class _RawKeyboard:
    """把终端置为 cbreak + 非阻塞，供主循环轮询驱动键。

    依赖按键自动重复来维持“按住”；``poll()`` 返回自上次调用以来新解码出的
    键事件，并在孤立 ``Esc`` 后短暂等待，以区分退出键与 CSI 方向键前缀。
    """

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._saved_attrs: object | None = None
        self._buffer = b""

    def __enter__(self) -> _RawKeyboard:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "Keyboard driving requires an interactive terminal (a TTY)."
            )
        self._saved_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        os.set_blocking(self._fd, False)
        return self

    def __exit__(self, *exc_info: object) -> None:
        if self._saved_attrs is not None:
            try:
                os.set_blocking(self._fd, True)
            finally:
                termios.tcsetattr(self._fd, termios.TCSADRAIN, self._saved_attrs)

    def poll(self) -> list[str]:
        self._drain()
        if self._buffer.endswith(b"\x1b"):
            ready, _, _ = select.select([self._fd], [], [], _ESC_LOOKAHEAD_S)
            if ready:
                self._drain()
            else:
                events, self._buffer = decode_keys(self._buffer[:-1])
                events.append("quit")
                return events
        events, self._buffer = decode_keys(self._buffer)
        return events

    def _drain(self) -> None:
        while True:
            try:
                chunk = os.read(self._fd, 4096)
            except (BlockingIOError, OSError):
                return
            if not chunk:
                return
            self._buffer += chunk


def _describe_snapshot(snapshot: PerceptionSnapshot | None) -> str:
    """把最新感知快照概括成一行目标与场地特征文本。"""

    if snapshot is None:
        return "targets=none"
    if snapshot.dropped_stale_age_ms is not None:
        return f"targets=STALE({snapshot.dropped_stale_age_ms:.0f}ms)"
    parts: list[str] = []
    for observation in snapshot.observations:
        ground = (
            f"g=({observation.ground_point.x:.0f},{observation.ground_point.y:.0f})mm"
            if observation.ground_point is not None
            else "g=-"
        )
        parts.append(
            f"{observation.target_class.value}="
            f"{observation.detection_confidence:.2f} {ground}"
        )
    feature_text = ""
    if snapshot.field_features is not None:
        cross = snapshot.field_features.center_cross is not None
        safe_zones = len(snapshot.field_features.safe_zones)
        feature_text = f" cross={'Y' if cross else 'n'} safe_zones={safe_zones}"
    return f"targets=[{','.join(parts) or '-'}]{feature_text}"


def _describe_status(status: CarSystemStatus | None) -> str:
    if status is None:
        return "stm32=none"
    return (
        f"stm32=(motor={status.motor_output_enabled},"
        f"watchdog={status.watchdog_armed},"
        f"estop={status.emergency_stop_latched},"
        f"stop={status.stop_reason.name.lower()})"
    )


def _draw_gyro_overlay(
    image_bgr: object,
    odometry: OdometryImu | None,
    now_ns: int,
) -> object:
    """在显示副本上叠加最新 STM32 原始陀螺仪数据。"""

    if odometry is None:
        text = "gyro=none"
    else:
        age_ms = max(0.0, (now_ns - odometry.received_timestamp_ns) / 1_000_000.0)
        valid = bool(odometry.sensor_flags & SensorFlags.IMU_VALID)
        saturated = bool(odometry.sensor_flags & SensorFlags.GYRO_SATURATED)
        text = (
            f"gyro_xyz=({odometry.gyro_x_urad_s / 1_000_000:+.3f},"
            f"{odometry.gyro_y_urad_s / 1_000_000:+.3f},"
            f"{odometry.gyro_z_urad_s / 1_000_000:+.3f}) rad/s "
            f"sample_us={odometry.sample_timestamp_us} age={age_ms:.0f}ms "
            f"{'VALID' if valid else 'INVALID'}"
            f"{' SATURATED' if saturated else ''}"
        )
    _draw_overlay_text(image_bgr, text, (12, 28))
    return image_bgr


def _draw_pose_overlay(
    image_bgr: object,
    estimate: FusedPoseEstimate | None,
    now_ns: int,
) -> object:
    """在显示副本上叠加融合后的场地绝对位姿。"""

    pose = None if estimate is None else estimate.pose
    if pose is None:
        if estimate is None:
            text = "pose=unavailable (no fusion estimate)"
        else:
            quality = ",".join(sorted(item.value for item in estimate.quality))
            text = f"pose=unavailable ({quality or 'no quality'})"
    else:
        estimate_timestamp_ns = estimate.estimate_timestamp_ns
        age_text = "age=n/a"
        if estimate_timestamp_ns is not None:
            age_ms = max(0.0, (now_ns - estimate_timestamp_ns) / 1_000_000.0)
            age_text = f"age={age_ms:.0f}ms"
        text = (
            f"field_xy=({pose.position.x:+.0f},{pose.position.y:+.0f})mm "
            f"heading={math.degrees(pose.heading_rad):+.1f}deg {age_text}"
        )
    _draw_overlay_text(image_bgr, text, (12, 62))
    return image_bgr


def _draw_overlay_text(image_bgr: object, text: str, origin: tuple[int, int]) -> None:
    """绘制大号黑字，并加白底保证在相机画面上可读。"""

    import cv2

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thickness = 2
    (width, height), baseline = cv2.getTextSize(text, font, scale, thickness)
    x, y = origin
    cv2.rectangle(
        image_bgr,
        (x - 5, y - height - baseline - 5),
        (x + width + 5, y + baseline + 5),
        (255, 255, 255),
        cv2.FILLED,
    )
    cv2.putText(
        image_bgr,
        text,
        origin,
        font,
        scale,
        (0, 0, 0),
        thickness,
        cv2.LINE_AA,
    )


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite value greater than zero, got {value!r}"
        )
    return parsed


def _run_hardware(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    linear_speed_m_s: float | None,
    angular_speed_rad_s: float | None,
    display: bool,
    enable_localization: bool,
) -> None:
    # 本地导入，避免测试收集阶段访问相机、Hailo、串口或 OpenCV 窗口。
    from rescue_vision.app.cluster_breakup import CameraPerceptionPump
    from rescue_vision.app.manual_capture import build_camera_pipeline
    from rescue_vision.perception import (
        PerceptionFrameRenderer,
        PerceptionSnapshot,
    )

    config = load_runtime_config(config_path)
    if not supervised_stop_ready:
        raise RuntimeError(
            "A physical emergency stop and continuous supervision are required "
            "until the STM32 watchdog has been verified."
        )
    if not config.uart.enabled:
        raise RuntimeError("keyboard_drive requires uart.enabled=true.")
    if not config.motion.enabled:
        raise RuntimeError("keyboard_drive requires motion.enabled=true.")
    if not config.hailo.enabled:
        raise RuntimeError(
            "keyboard_drive requires hailo.enabled with a deployed v3 model "
            "for live model feedback."
        )
    localization_fusion = (
        config.build_odometry_imu_fusion()
        if enable_localization or config.localization.fusion.enabled
        else None
    )
    if enable_localization and localization_fusion is None:
        raise RuntimeError(
            "--enable-localization requires localization.fusion.enabled=true "
            "and complete motion.odometry calibration."
        )

    linear_speed = (
        min(0.15, config.motion.max_linear_velocity_m_s)
        if linear_speed_m_s is None
        else linear_speed_m_s
    )
    angular_speed = (
        min(0.60, config.motion.max_angular_velocity_rad_s)
        if angular_speed_rad_s is None
        else angular_speed_rad_s
    )
    if not 0.0 < linear_speed <= config.motion.max_linear_velocity_m_s:
        raise RuntimeError(
            f"--speed-m-s {linear_speed} exceeds motion.max_linear_velocity_m_s "
            f"{config.motion.max_linear_velocity_m_s}."
        )
    if not 0.0 < angular_speed <= config.motion.max_angular_velocity_rad_s:
        raise RuntimeError(
            f"--turn-rad-s {angular_speed} exceeds motion.max_angular_velocity_rad_s "
            f"{config.motion.max_angular_velocity_rad_s}."
        )

    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("keyboard_drive requires enabled uart.")
    controller = config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("keyboard_drive requires enabled motion.")
    gripper = config.motion.gripper.build_calibration()
    if gripper is None:
        print(
            "keyboard_drive: motion.gripper not enabled; Z/X/C are unavailable.",
            flush=True,
        )

    pipeline = build_camera_pipeline(config)
    renderer = PerceptionFrameRenderer(
        lambda: config.build_target_pose_detector(
            ground_projector=pipeline.ground_projector
        ),
        render_enabled=display,
    )
    camera_pump = CameraPerceptionPump(pipeline.source, pipeline.prepare, renderer)

    stop_requested = False
    latest_status: CarSystemStatus | None = None
    latest_odometry: OdometryImu | None = None
    latest_pose_estimate: FusedPoseEstimate | None = None
    latest_snapshot: PerceptionSnapshot | None = None
    camera_start_thread = None
    camera_pump_started = False
    next_status_ns = 0
    drive_state = KeyDriveState()

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

    viewer = None
    if display:
        from rescue_vision.camera.viewer import OpenCvFrameViewer

        viewer = OpenCvFrameViewer("keyboard drive — Esc/q to quit")

    try:
        with _RawKeyboard() as keyboard, channel:
            try:
                # Hailo/相机预热与运动通道同步并行；等待期间主线程仍排空 UART。
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
                print(
                    "keyboard_drive: WASD/arrows drive, space brake, X transport, "
                    "Esc/q quit; "
                    f"speed={linear_speed:.3f} m/s turn={angular_speed:.3f} rad/s",
                    flush=True,
                )

                while not stop_requested:
                    now_ns = time.monotonic_ns()
                    if controller.needs_synchronization:
                        controller.synchronize(
                            timeout_s=config.motion.synchronization_timeout_s,
                            on_message=consume,
                        )
                    drive_state.apply(keyboard.poll(), now_ns)
                    if drive_state.consume_quit():
                        break
                    drive_state.consume_stop()
                    gripper_action = drive_state.consume_gripper()
                    if gripper_action is not None and gripper is not None:
                        try:
                            angles = gripper_angles_for(gripper_action, gripper)
                        except ValueError as exc:
                            print(
                                f"gripper={gripper_action} skipped: {exc}",
                                flush=True,
                            )
                        else:
                            controller.set_gripper_angles(*angles)
                            print(
                                f"gripper={gripper_action} "
                                f"angles=({angles[0]:.1f},{angles[1]:.1f})deg",
                                flush=True,
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
                        elif isinstance(message, OdometryImu):
                            # 键盘驾驶不消费定位；保留接收仅用于排空 100 Hz 遥测。
                            latest_odometry = message
                            if localization_fusion is not None:
                                latest_pose_estimate = (
                                    localization_fusion.submit_odometry(message)
                                )
                    camera_pump.check_health()

                    latest_snapshot = renderer.latest_fresh_snapshot(
                        now_ns,
                        config.processing.max_observation_age_ms,
                    )

                    active_keys = drive_state.active_keys(now_ns)
                    linear, angular = compute_twist(
                        active_keys,
                        linear_speed_m_s=linear_speed,
                        angular_speed_rad_s=angular_speed,
                    )
                    controller.drive_wheel_limited(linear, angular)

                    if viewer is not None:
                        rendered = renderer.latest()
                        if rendered is not None:
                            display_image = rendered.image_bgr.copy()
                            _draw_gyro_overlay(
                                display_image,
                                latest_odometry,
                                now_ns,
                            )
                            _draw_pose_overlay(
                                display_image,
                                latest_pose_estimate,
                                now_ns,
                            )
                            if not viewer.show(display_image):
                                break

                    if now_ns >= next_status_ns:
                        gripper_target = controller.gripper_target_angles_deg
                        gripper_text = (
                            "grip=none"
                            if gripper_target is None
                            else (
                                f"grip=({gripper_target[0]:.0f},"
                                f"{gripper_target[1]:.0f})deg"
                            )
                        )
                        print(
                            f"cmd={_command_label(active_keys)} "
                            f"linear={linear:+.3f} angular={angular:+.3f} "
                            "wheel="
                            f"({controller.target_wheel_speeds_m_s[0]:+.3f},"
                            f"{controller.target_wheel_speeds_m_s[1]:+.3f}) "
                            f"{gripper_text} "
                            f"{_describe_snapshot(latest_snapshot)} "
                            f"{_describe_status(latest_status)}",
                            flush=True,
                        )
                        next_status_ns = now_ns + _STATUS_PERIOD_NS
                    time.sleep(_LOOP_PERIOD_S)
            finally:
                try:
                    controller.soft_brake()
                finally:
                    if camera_start_thread is not None or camera_pump_started:
                        camera_pump.stop()
    finally:
        if viewer is not None:
            viewer.close()
        signal.signal(signal.SIGTERM, previous_sigterm)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Keyboard-controlled driving with live camera and perception "
            "feedback. Drive keys come from the terminal (works over SSH); "
            "Esc/q quit, space brake, X gripper transport, --display opens an optional window."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.yaml"),
        help="Runtime YAML path (default: configs/runtime.yaml).",
    )
    parser.add_argument(
        "--speed-m-s",
        type=_positive_float,
        default=None,
        help="Forward/backward speed in m/s (default: min(0.15, config limit)).",
    )
    parser.add_argument(
        "--turn-rad-s",
        type=_positive_float,
        default=None,
        help="Turn angular speed in rad/s (default: min(0.60, config limit)).",
    )
    parser.add_argument(
        "--display",
        action="store_true",
        help="Open an OpenCV window with the annotated perception frame.",
    )
    parser.add_argument(
        "--enable-localization",
        action="store_true",
        help=(
            "Enable encoder+IMU field-pose overlay; requires enabled and "
            "calibrated localization.fusion config."
        ),
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help=(
            "Confirm that a physical emergency stop is ready and an operator "
            "will supervise the whole test."
        ),
    )
    args = parser.parse_args()
    if not args.supervised_physical_stop_ready:
        parser.error(
            "--supervised-physical-stop-ready is required for a live motion test"
        )
    try:
        _run_hardware(
            args.config.expanduser().resolve(),
            supervised_stop_ready=args.supervised_physical_stop_ready,
            linear_speed_m_s=args.speed_m_s,
            angular_speed_rad_s=args.turn_rad_s,
            display=args.display,
            enable_localization=args.enable_localization,
        )
    except KeyboardInterrupt:
        print("stopped_by_user=true", flush=True)


if __name__ == "__main__":
    main()
