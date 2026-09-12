"""键盘操控车辆：前进/后退/转弯 + 实时相机与模型结果回传 + Esc 退出。

本脚本是赛外受监督手动驾驶的人工验收入口，不是正式比赛应用。它把
``TargetPoseDetector``（Hailo YOLO Pose v3）的取帧、去畸变和推理放在独立
旁路线程（``CameraPerceptionPump`` + ``PerceptionFrameRenderer``），主线程
只做三件轻量工作：非阻塞读取键盘、刷新轮速目标并排空 UART 回传、读取最新
推理快照并打印/显示。因此慢取帧、慢去畸变或慢推理不会拖慢运动安全循环。

键盘从终端读取（通过 SSH 也有效）；``--display`` 可选地在本地监视器上弹出
OpenCV 叠加画面。方向键和 WASD 等价：

    W / ↑      锁存前进       S / ↓      锁存后退
    A / ←      按住左转       D / →      按住右转
    Z          夹爪打开       C          夹爪关闭
    X          夹爪运输姿态   空格        松手停车
    Esc / q    退出

先按 W/S 锁存纵向运动，再按住 A/D 即可组合成弧线（例如 W 后按住 D 为右前弧）；
松开 A/D 后转向自动归中，空格清除全部运动。所有轮速仍受
``motion`` 配置的线速度、角速度和单轮加速度上限约束，超限请求会被拒绝。
脚本把每次变化后的实际底盘目标实时打印，并把目标及 100 Hz 编码器/IMU 遥测
写入 JSONL；``--replay`` 使用轮累计行程和相对航向闭环回放。日志不包含夹爪
动作；编码器闭环不能观测整车横向打滑，因此仍不是场地绝对轨迹控制。
夹爪键（``Z``/``X``/``C``）只发送一次已标定角度，不进入持续推进状态；
``X`` 的运输姿态还需要配置 ``motion.gripper.transport_*_angle_deg``。

运行前必须在 ``runtime.yaml`` 中启用 ``uart``、``motion`` 和 ``hailo``，并
确认物理急停可立即触发、操作员全程监督。夹爪控制需要 ``motion.gripper``
已标定并启用。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import select
import signal
import sys
import termios
import threading
import time
import tty
from collections.abc import Sequence
from bisect import bisect_right
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, TextIO

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

_LOG_FORMAT = "rescue_vision.keyboard_drive"
_LOG_VERSION = 2
_REQUIRED_ENCODER_FLAGS = (
    SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
)
_REPLAY_POSITION_KP_S_INV = 2.0
_REPLAY_HEADING_KP_S_INV = 1.5
_REPLAY_MAX_HEADING_CORRECTION_RAD_S = 0.5
_REPLAY_TELEMETRY_TIMEOUT_NS = 200_000_000
_REPLAY_INITIAL_TELEMETRY_TIMEOUT_S = 1.0
_REPLAY_SETTLE_TIMEOUT_S = 3.0
_REPLAY_POSITION_TOLERANCE_M = 0.005
_REPLAY_HEADING_TOLERANCE_RAD = math.radians(3.0)

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
    """把终端按键事件转换为可组合的纵向锁存与瞬时转向。

    普通终端没有 key-up 事件，而且通常只自动重复最后按下的键。为可靠支持
    ``W/S + A/D``，纵向 W/S 锁存到空格停车或相反方向键，转向 A/D 仍依赖
    自动重复并在松键后超时归中。``stop``、``quit`` 和夹爪键是边沿事件。
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
        self._linear_key: str | None = None
        self._turn_last_seen_ns: dict[str, int] = {}
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
            elif event in {"forward", "backward"}:
                self._linear_key = event
            elif event in {"turn_left", "turn_right"}:
                self._turn_last_seen_ns[event] = now_ns
                opposite = (
                    "turn_right" if event == "turn_left" else "turn_left"
                )
                self._turn_last_seen_ns.pop(opposite, None)
            elif event in _GRIPPER_ACTIONS:
                self._gripper_pending = _GRIPPER_ACTIONS[event]

    def active_keys(self, now_ns: int) -> frozenset[str]:
        """返回仍在按住超时窗口内的驱动键。"""

        turn_keys = {
            key
            for key, seen_ns in self._turn_last_seen_ns.items()
            if now_ns - seen_ns <= self._hold_timeout_ns
        }
        if self._linear_key is not None:
            turn_keys.add(self._linear_key)
        return frozenset(turn_keys)

    def consume_stop(self) -> bool:
        """消费一次停车请求，并清除当前所有驱动键。"""

        if not self._stop_pending:
            return False
        self._stop_pending = False
        self._linear_key = None
        self._turn_last_seen_ns.clear()
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


@dataclass(frozen=True, slots=True)
class KeyboardDriveCommand:
    """一条可按相对时间回放的底盘目标指令。"""

    elapsed_ns: int
    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    left_target_m_s: float
    right_target_m_s: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.elapsed_ns, bool)
            or not isinstance(self.elapsed_ns, int)
            or self.elapsed_ns < 0
        ):
            raise ValueError(
                "elapsed_ns must be a non-negative integer, "
                f"got {self.elapsed_ns!r}."
            )
        for name in (
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
            "left_target_m_s",
            "right_target_m_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite, got {value!r}.")

    @property
    def stopped(self) -> bool:
        return self.left_target_m_s == 0.0 and self.right_target_m_s == 0.0


@dataclass(frozen=True, slots=True)
class KeyboardDriveTelemetry:
    """日志中的一条累计编码器与 Z 轴陀螺仪反馈。"""

    elapsed_ns: int
    telemetry_sequence: int
    sample_timestamp_us: int
    left_encoder_count: int
    right_encoder_count: int
    gyro_z_urad_s: int
    sensor_flags: int

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"{name} must be an integer, got {value!r}.")
        if self.elapsed_ns < 0 or self.sample_timestamp_us < 0:
            raise ValueError("Telemetry timestamps must be non-negative.")
        if not 0 <= self.telemetry_sequence <= 0xFFFF:
            raise ValueError("telemetry_sequence must be in [0, 65535].")
        known_flags = 0
        for flag in SensorFlags:
            known_flags |= int(flag)
        if self.sensor_flags < 0 or self.sensor_flags & ~known_flags:
            raise ValueError(
                f"sensor_flags contains unknown bits: {self.sensor_flags:#x}."
            )

    @classmethod
    def from_message(
        cls,
        message: OdometryImu,
        *,
        started_ns: int,
    ) -> KeyboardDriveTelemetry:
        return cls(
            elapsed_ns=message.received_timestamp_ns - started_ns,
            telemetry_sequence=message.telemetry_sequence,
            sample_timestamp_us=message.sample_timestamp_us,
            left_encoder_count=message.left_encoder_count,
            right_encoder_count=message.right_encoder_count,
            gyro_z_urad_s=message.gyro_z_urad_s,
            sensor_flags=int(message.sensor_flags),
        )


@dataclass(frozen=True, slots=True)
class KeyboardDriveRecording:
    commands: tuple[KeyboardDriveCommand, ...]
    telemetry: tuple[KeyboardDriveTelemetry, ...]


def _motion_fingerprint(controller: object) -> dict[str, float]:
    limits = controller.limits
    return {
        "wheel_track_m": limits.wheel_track_m,
        "max_linear_velocity_m_s": limits.max_linear_velocity_m_s,
        "max_angular_velocity_rad_s": limits.max_angular_velocity_rad_s,
        "max_wheel_velocity_m_s": limits.max_wheel_velocity_m_s,
        "min_wheel_velocity_m_s": limits.min_wheel_velocity_m_s,
        "max_wheel_acceleration_m_s2": limits.max_wheel_acceleration_m_s2,
        "left_wheel_speed_weight": limits.left_wheel_speed_weight,
        "right_wheel_speed_weight": limits.right_wheel_speed_weight,
    }


def _odometry_fingerprint(calibration: object) -> dict[str, object]:
    return {
        "encoder_counts_per_revolution": (
            calibration.encoder_counts_per_revolution
        ),
        "left_wheel_radius_mm": calibration.left_wheel_radius_mm,
        "right_wheel_radius_mm": calibration.right_wheel_radius_mm,
        "gyro_z_sign": calibration.gyro_z_sign,
    }


class KeyboardDriveLogWriter:
    """通过有界旁路线程实时写入并打印键盘底盘指令 JSONL。"""

    def __init__(
        self,
        path: Path,
        controller: object,
        calibration: object,
        *,
        started_ns: int,
    ) -> None:
        if started_ns < 0:
            raise ValueError("started_ns must be non-negative.")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path.resolve()
        self._started_ns = started_ns
        self._stream: TextIO = self.path.open("x", encoding="utf-8")
        self._last: KeyboardDriveCommand | None = None
        self._queue: queue.Queue[dict[str, object] | None] = queue.Queue(
            maxsize=4096
        )
        self._worker_error: BaseException | None = None
        header = {
            "record_type": "header",
            "format": _LOG_FORMAT,
            "version": _LOG_VERSION,
            "created_at": datetime.now().astimezone().isoformat(),
            "motion": _motion_fingerprint(controller),
            "odometry": _odometry_fingerprint(calibration),
        }
        self._write(header)
        self._worker = threading.Thread(
            target=self._run_writer,
            name="keyboard-drive-log",
            daemon=False,
        )
        self._worker.start()

    def __enter__(self) -> KeyboardDriveLogWriter:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._queue.put(None)
        self._worker.join()
        self._stream.close()
        if self._worker_error is not None and exc_info[0] is None:
            raise RuntimeError(
                "Keyboard drive log writer failed."
            ) from self._worker_error

    @property
    def last_stopped(self) -> bool:
        return self._last is not None and self._last.stopped

    def record(
        self,
        now_ns: int,
        *,
        linear_velocity_m_s: float,
        angular_velocity_rad_s: float,
        target_wheel_speeds_m_s: tuple[float, float],
        force: bool = False,
    ) -> bool:
        elapsed_ns = now_ns - self._started_ns
        if force and self._last is not None and elapsed_ns <= self._last.elapsed_ns:
            elapsed_ns = self._last.elapsed_ns + 1
        command = KeyboardDriveCommand(
            elapsed_ns=elapsed_ns,
            linear_velocity_m_s=linear_velocity_m_s,
            angular_velocity_rad_s=angular_velocity_rad_s,
            left_target_m_s=target_wheel_speeds_m_s[0],
            right_target_m_s=target_wheel_speeds_m_s[1],
        )
        if not force and self._last is not None and (
            command.linear_velocity_m_s == self._last.linear_velocity_m_s
            and command.angular_velocity_rad_s == self._last.angular_velocity_rad_s
            and command.left_target_m_s == self._last.left_target_m_s
            and command.right_target_m_s == self._last.right_target_m_s
        ):
            return False
        if self._worker_error is not None:
            raise RuntimeError(
                "Keyboard drive log writer failed."
            ) from self._worker_error
        self._last = command
        try:
            self._queue.put_nowait({"record_type": "command", **asdict(command)})
        except queue.Full as exc:
            raise RuntimeError(
                "Keyboard drive log queue is full; stopping motion."
            ) from exc
        return True

    def record_telemetry(self, message: OdometryImu) -> bool:
        if message.received_timestamp_ns < self._started_ns:
            return False
        if self._worker_error is not None:
            raise RuntimeError(
                "Keyboard drive log writer failed."
            ) from self._worker_error
        sample = KeyboardDriveTelemetry.from_message(
            message,
            started_ns=self._started_ns,
        )
        try:
            self._queue.put_nowait({"record_type": "telemetry", **asdict(sample)})
        except queue.Full as exc:
            raise RuntimeError(
                "Keyboard drive log queue is full; stopping motion."
            ) from exc
        return True

    def _run_writer(self) -> None:
        try:
            while True:
                record = self._queue.get()
                if record is None:
                    return
                self._write(record)
        except BaseException as exc:  # pragma: no cover - depends on filesystem failure
            self._worker_error = exc

    def _write(self, record: dict[str, object]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        self._stream.write(line + "\n")
        self._stream.flush()
        if record.get("record_type") != "telemetry":
            print(f"control_log={line}", flush=True)


def apply_keyboard_drive_target(
    controller: object,
    command_log: KeyboardDriveLogWriter,
    *,
    requested_linear_m_s: float,
    requested_angular_rad_s: float,
) -> tuple[float, float]:
    """设置、记录并刷新一次目标，保证控制器使用单调递增的内部时钟。"""

    linear, angular = controller.drive_wheel_limited(
        requested_linear_m_s,
        requested_angular_rad_s,
    )
    command_log.record(
        time.monotonic_ns(),
        linear_velocity_m_s=linear,
        angular_velocity_rad_s=angular,
        target_wheel_speeds_m_s=controller.target_wheel_speeds_m_s,
    )
    # drive_wheel_limited() -> set_wheel_speeds() first settles the preceding
    # target using a fresh internal timestamp. A timestamp captured before that
    # call would be rejected as clock rollback, so let update() sample its clock.
    controller.update()
    return linear, angular


def load_keyboard_drive_log(
    path: Path,
    controller: object,
    calibration: object,
) -> KeyboardDriveRecording:
    """严格读取日志并校验当前运动配置和安全的起止零速。"""

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise ValueError(f"Cannot read keyboard drive log {path}: {exc}.") from exc
    if len(lines) < 4:
        raise ValueError(
            "Keyboard drive log must contain a header, commands and telemetry."
        )
    try:
        header = json.loads(lines[0])
    except json.JSONDecodeError as exc:
        raise ValueError("Keyboard drive log header is not valid JSON.") from exc
    expected_header = {
        "record_type",
        "format",
        "version",
        "created_at",
        "motion",
        "odometry",
    }
    if not isinstance(header, dict) or set(header) != expected_header:
        raise ValueError(
            "Keyboard drive log header fields do not match the current schema."
        )
    if (
        header["record_type"] != "header"
        or header["format"] != _LOG_FORMAT
        or header["version"] != _LOG_VERSION
    ):
        raise ValueError("Keyboard drive log format or version is unsupported.")
    if header["motion"] != _motion_fingerprint(controller):
        raise ValueError(
            "Keyboard drive log motion configuration does not match the current runtime config."
        )
    if header["odometry"] != _odometry_fingerprint(calibration):
        raise ValueError(
            "Keyboard drive log odometry calibration does not match the "
            "current runtime config."
        )

    commands: list[KeyboardDriveCommand] = []
    telemetry: list[KeyboardDriveTelemetry] = []
    command_fields = {"record_type", *KeyboardDriveCommand.__dataclass_fields__}
    telemetry_fields = {
        "record_type",
        *KeyboardDriveTelemetry.__dataclass_fields__,
    }
    for line_number, line in enumerate(lines[1:], start=2):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Keyboard drive log line {line_number} is not valid JSON.") from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Keyboard drive log line {line_number} is not an object."
            )
        record_type = record.get("record_type")
        values = {
            key: value for key, value in record.items() if key != "record_type"
        }
        if record_type == "command" and set(record) == command_fields:
            try:
                command = KeyboardDriveCommand(**values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid command on keyboard drive log line {line_number}."
                ) from exc
            if commands and command.elapsed_ns <= commands[-1].elapsed_ns:
                raise ValueError(
                    "Keyboard drive command elapsed_ns values must strictly increase."
                )
            commands.append(command)
        elif record_type == "telemetry" and set(record) == telemetry_fields:
            try:
                sample = KeyboardDriveTelemetry(**values)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid telemetry on keyboard drive log line {line_number}."
                ) from exc
            if telemetry and sample.sample_timestamp_us <= telemetry[-1].sample_timestamp_us:
                raise ValueError(
                    "Keyboard drive telemetry device timestamps must strictly increase."
                )
            telemetry.append(sample)
        else:
            raise ValueError(
                f"Keyboard drive log line {line_number} has an invalid record schema."
            )
    if not commands:
        raise ValueError("Keyboard drive log contains no commands.")
    if commands[0].elapsed_ns != 0 or not commands[0].stopped:
        raise ValueError(
            "Keyboard drive log must start with a zero-speed command "
            "at elapsed_ns=0."
        )
    if not commands[-1].stopped:
        raise ValueError("Keyboard drive log must end with a zero-speed command.")
    if len(telemetry) < 2:
        raise ValueError("Keyboard drive log requires at least two telemetry samples.")
    for index, sample in enumerate(telemetry):
        flags = SensorFlags(sample.sensor_flags)
        if flags & _REQUIRED_ENCODER_FLAGS != _REQUIRED_ENCODER_FLAGS:
            raise ValueError(f"Encoder telemetry is invalid at sample {index}.")
    maximum = controller.limits.max_wheel_velocity_m_s
    for index, command in enumerate(commands):
        if max(abs(command.left_target_m_s), abs(command.right_target_m_s)) > maximum:
            raise ValueError(
                f"Keyboard drive command {index} exceeds the configured "
                "wheel-speed limit."
            )
    return KeyboardDriveRecording(tuple(commands), tuple(telemetry))


class _RawKeyboard:
    """把终端置为 cbreak + 非阻塞，供主循环轮询驱动键。

    ``poll()`` 返回自上次调用以来新解码出的键事件，并在孤立 ``Esc`` 后短暂
    等待，以区分退出键与 CSI 方向键前缀。转向键依赖自动重复表达“按住”。
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


def _default_keyboard_log_path(log_dir: Path) -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    return (log_dir / f"keyboard_drive_{stamp}.jsonl").resolve()


def _confirm_replay(path: Path, duration_s: float) -> bool:
    try:
        answer = input(
            f"回放日志：{path}\n"
            f"记录时长：{duration_s:.3f} s。把车辆放回起点并清空运动区域，"
            "输入 REPLAY 后按 Enter 开始（其它输入退出）： "
        )
    except EOFError:
        return False
    return answer.strip() == "REPLAY"


@dataclass(frozen=True, slots=True)
class ReplayReferencePoint:
    elapsed_s: float
    left_distance_m: float
    right_distance_m: float
    heading_rad: float | None


def _metres_per_count(calibration: object, *, left: bool) -> float:
    radius_mm = (
        calibration.left_wheel_radius_mm
        if left
        else calibration.right_wheel_radius_mm
    )
    return (
        2.0
        * math.pi
        * radius_mm
        / calibration.encoder_counts_per_revolution
        / 1000.0
    )


def build_replay_reference(
    telemetry: Sequence[KeyboardDriveTelemetry],
    calibration: object,
) -> tuple[ReplayReferencePoint, ...]:
    """把原始累计计数和陀螺仪转换为相对轮行程/航向参考。"""

    if len(telemetry) < 2:
        raise ValueError("Replay reference requires at least two telemetry samples.")
    first = telemetry[0]
    left_scale = _metres_per_count(calibration, left=True)
    right_scale = _metres_per_count(calibration, left=False)
    heading_rad = 0.0
    previous = first
    previous_gyro_valid = False
    points: list[ReplayReferencePoint] = []
    for index, sample in enumerate(telemetry):
        flags = SensorFlags(sample.sensor_flags)
        encoder_valid = (
            flags & _REQUIRED_ENCODER_FLAGS == _REQUIRED_ENCODER_FLAGS
        )
        if not encoder_valid:
            raise ValueError(f"Encoder telemetry is invalid at sample {index}.")
        if index and sample.sample_timestamp_us <= previous.sample_timestamp_us:
            raise ValueError("Telemetry sample timestamps must strictly increase.")
        gyro_valid = bool(flags & SensorFlags.IMU_VALID) and not bool(
            flags & (SensorFlags.GYRO_SATURATED | SensorFlags.SAMPLE_OVERRUN)
        )
        if index and gyro_valid and previous_gyro_valid:
            dt_s = (
                sample.sample_timestamp_us - previous.sample_timestamp_us
            ) / 1_000_000.0
            previous_rate = (
                previous.gyro_z_urad_s
                / 1_000_000.0
                * calibration.gyro_z_sign
            )
            current_rate = (
                sample.gyro_z_urad_s
                / 1_000_000.0
                * calibration.gyro_z_sign
            )
            heading_rad += 0.5 * (previous_rate + current_rate) * dt_s
        points.append(
            ReplayReferencePoint(
                elapsed_s=(
                    sample.sample_timestamp_us - first.sample_timestamp_us
                )
                / 1_000_000.0,
                left_distance_m=(
                    sample.left_encoder_count - first.left_encoder_count
                )
                * left_scale,
                right_distance_m=(
                    sample.right_encoder_count - first.right_encoder_count
                )
                * right_scale,
                heading_rad=heading_rad if gyro_valid else None,
            )
        )
        previous = sample
        previous_gyro_valid = gyro_valid
    if points[-1].elapsed_s <= 0.0:
        raise ValueError("Replay telemetry duration must be positive.")
    return tuple(points)


def interpolate_replay_reference(
    points: Sequence[ReplayReferencePoint],
    elapsed_s: float,
) -> ReplayReferencePoint:
    if not points:
        raise ValueError("Replay reference is empty.")
    timestamps = [point.elapsed_s for point in points]
    index = bisect_right(timestamps, elapsed_s)
    if index <= 0:
        return points[0]
    if index >= len(points):
        return points[-1]
    before = points[index - 1]
    after = points[index]
    ratio = (elapsed_s - before.elapsed_s) / (after.elapsed_s - before.elapsed_s)
    heading = None
    if before.heading_rad is not None and after.heading_rad is not None:
        heading = before.heading_rad + (after.heading_rad - before.heading_rad) * ratio
    return ReplayReferencePoint(
        elapsed_s=elapsed_s,
        left_distance_m=before.left_distance_m
        + (after.left_distance_m - before.left_distance_m) * ratio,
        right_distance_m=before.right_distance_m
        + (after.right_distance_m - before.right_distance_m) * ratio,
        heading_rad=heading,
    )


class LiveReplayFeedback:
    """把回放现场遥测转换为相对编码器行程和相对 IMU 航向。"""

    def __init__(self, calibration: object) -> None:
        self._calibration = calibration
        self._first: OdometryImu | None = None
        self._previous: OdometryImu | None = None
        self.left_distance_m = 0.0
        self.right_distance_m = 0.0
        self.heading_rad = 0.0
        self.heading_available = False
        self.last_received_ns: int | None = None

    @property
    def ready(self) -> bool:
        return self._first is not None

    def observe(self, message: OdometryImu) -> None:
        flags = message.sensor_flags
        if flags & _REQUIRED_ENCODER_FLAGS != _REQUIRED_ENCODER_FLAGS:
            raise RuntimeError("Replay received invalid encoder telemetry.")
        previous = self._previous
        if previous is not None and message.sample_timestamp_us <= previous.sample_timestamp_us:
            raise RuntimeError("Replay telemetry device timestamp did not increase.")
        if self._first is None:
            self._first = message
        first = self._first
        self.left_distance_m = (
            message.left_encoder_count - first.left_encoder_count
        ) * _metres_per_count(self._calibration, left=True)
        self.right_distance_m = (
            message.right_encoder_count - first.right_encoder_count
        ) * _metres_per_count(self._calibration, left=False)
        gyro_valid = bool(flags & SensorFlags.IMU_VALID) and not bool(
            flags & (SensorFlags.GYRO_SATURATED | SensorFlags.SAMPLE_OVERRUN)
        )
        if previous is not None and gyro_valid:
            previous_flags = previous.sensor_flags
            previous_valid = bool(previous_flags & SensorFlags.IMU_VALID) and not bool(
                previous_flags
                & (SensorFlags.GYRO_SATURATED | SensorFlags.SAMPLE_OVERRUN)
            )
            if previous_valid:
                dt_s = (
                    message.sample_timestamp_us - previous.sample_timestamp_us
                ) / 1_000_000.0
                previous_rate = (
                    previous.gyro_z_rad_s * self._calibration.gyro_z_sign
                )
                current_rate = (
                    message.gyro_z_rad_s * self._calibration.gyro_z_sign
                )
                self.heading_rad += 0.5 * (previous_rate + current_rate) * dt_s
                self.heading_available = True
        self._previous = message
        self.last_received_ns = message.received_timestamp_ns

    def require_recent(self, now_ns: int) -> None:
        if self.last_received_ns is None:
            raise RuntimeError("Replay has not received encoder telemetry.")
        age_ns = now_ns - self.last_received_ns
        if age_ns < 0 or age_ns > _REPLAY_TELEMETRY_TIMEOUT_NS:
            raise RuntimeError(
                "Replay encoder/IMU telemetry is stale; soft braking."
            )


def closed_loop_wheel_targets(
    controller: object,
    reference: ReplayReferencePoint,
    feedback: LiveReplayFeedback,
    feedforward: KeyboardDriveCommand,
) -> tuple[float, float, float, float]:
    """计算带左右轮位置反馈和 IMU 航向反馈的有界轮速目标。"""

    left_error = reference.left_distance_m - feedback.left_distance_m
    right_error = reference.right_distance_m - feedback.right_distance_m
    left = feedforward.left_target_m_s + _REPLAY_POSITION_KP_S_INV * left_error
    right = feedforward.right_target_m_s + _REPLAY_POSITION_KP_S_INV * right_error
    heading_error = 0.0
    if reference.heading_rad is not None and feedback.heading_available:
        heading_error = math.atan2(
            math.sin(reference.heading_rad - feedback.heading_rad),
            math.cos(reference.heading_rad - feedback.heading_rad),
        )
        correction = max(
            -_REPLAY_MAX_HEADING_CORRECTION_RAD_S,
            min(
                _REPLAY_MAX_HEADING_CORRECTION_RAD_S,
                _REPLAY_HEADING_KP_S_INV * heading_error,
            ),
        )
        half_track = controller.limits.wheel_track_m / 2.0
        left -= correction * half_track
        right += correction * half_track
    peak = max(abs(left), abs(right))
    maximum = controller.limits.max_wheel_velocity_m_s
    if peak > maximum:
        scale = math.nextafter(maximum / peak, 0.0)
        left *= scale
        right *= scale
    return left, right, left_error, right_error


def _command_at_elapsed_ns(
    commands: Sequence[KeyboardDriveCommand],
    elapsed_ns: int,
) -> KeyboardDriveCommand:
    timestamps = [command.elapsed_ns for command in commands]
    index = max(0, bisect_right(timestamps, elapsed_ns) - 1)
    return commands[index]


def replay_keyboard_recording(
    controller: object,
    recording: KeyboardDriveRecording,
    calibration: object,
    keyboard: _RawKeyboard,
    *,
    stop_event: threading.Event | None = None,
) -> bool:
    """按时间推进参考，以累计轮行程和相对航向闭环复现运动。"""

    points = build_replay_reference(recording.telemetry, calibration)
    feedback = LiveReplayFeedback(calibration)
    initial_deadline = time.monotonic() + _REPLAY_INITIAL_TELEMETRY_TIMEOUT_S
    while not feedback.ready:
        if time.monotonic() >= initial_deadline:
            raise RuntimeError("Timed out waiting for initial replay telemetry.")
        try:
            message = controller.receive_message(timeout=0.05)
        except TimeoutError:
            continue
        if isinstance(message, OdometryImu):
            feedback.observe(message)

    started_ns = time.monotonic_ns()
    reference_duration_s = points[-1].elapsed_s
    recorded_origin_ns = recording.telemetry[0].elapsed_ns
    settle_deadline_ns = started_ns + round(
        (reference_duration_s + _REPLAY_SETTLE_TIMEOUT_S) * 1_000_000_000
    )
    settled_samples = 0
    next_print_ns = started_ns
    try:
        while True:
            events = keyboard.poll()
            now_ns = time.monotonic_ns()
            if (
                (stop_event is not None and stop_event.is_set())
                or "stop" in events
                or "quit" in events
            ):
                return False
            elapsed_s = (now_ns - started_ns) / 1_000_000_000.0
            reference_elapsed_s = min(elapsed_s, reference_duration_s)
            reference = interpolate_replay_reference(
                points,
                reference_elapsed_s,
            )
            recorded_elapsed_ns = recorded_origin_ns + round(
                reference_elapsed_s * 1_000_000_000
            )
            feedforward = _command_at_elapsed_ns(
                recording.commands,
                recorded_elapsed_ns,
            )
            if elapsed_s >= reference_duration_s:
                feedforward = KeyboardDriveCommand(
                    recorded_elapsed_ns,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                )
            left, right, left_error, right_error = closed_loop_wheel_targets(
                controller,
                reference,
                feedback,
                feedforward,
            )
            controller.set_wheel_speeds(
                left,
                right,
                min_wheel_velocity_m_s=0.0,
            )
            controller.update()
            for message in controller.drain_messages():
                if isinstance(message, OdometryImu):
                    feedback.observe(message)
                elif isinstance(message, CarCommandReply):
                    if message.result not in {
                        CommandResult.ACCEPTED,
                        CommandResult.SEQUENCE_OLD,
                    }:
                        raise RuntimeError("STM32 rejected a replay wheel command.")
                elif isinstance(message, CarSystemStatus):
                    if message.emergency_stop_latched:
                        raise RuntimeError("STM32 emergency stop is latched during replay.")
                    if (
                        not message.protocol_ready
                        or message.reply_queue_full
                        or message.tx_degraded
                    ):
                        raise RuntimeError("STM32 UART health is degraded during replay.")
            feedback.require_recent(time.monotonic_ns())
            if controller.needs_synchronization or controller.link_degraded:
                raise RuntimeError("STM32 link lost synchronization during replay.")

            heading_error = 0.0
            if reference.heading_rad is not None and feedback.heading_available:
                heading_error = math.atan2(
                    math.sin(reference.heading_rad - feedback.heading_rad),
                    math.cos(reference.heading_rad - feedback.heading_rad),
                )
            within_tolerance = (
                elapsed_s >= reference_duration_s
                and abs(left_error) <= _REPLAY_POSITION_TOLERANCE_M
                and abs(right_error) <= _REPLAY_POSITION_TOLERANCE_M
                and (
                    reference.heading_rad is None
                    or not feedback.heading_available
                    or abs(heading_error) <= _REPLAY_HEADING_TOLERANCE_RAD
                )
            )
            settled_samples = settled_samples + 1 if within_tolerance else 0
            if settled_samples >= 3:
                return True
            if now_ns >= settle_deadline_ns:
                raise RuntimeError(
                    "Closed-loop replay did not reach the recorded endpoint "
                    "within the settle timeout."
                )
            if now_ns >= next_print_ns:
                print(
                    f"replay={reference_elapsed_s:.2f}/{reference_duration_s:.2f}s "
                    f"wheel_error=({left_error * 1000:+.1f},"
                    f"{right_error * 1000:+.1f})mm "
                    f"heading_error={math.degrees(heading_error):+.1f}deg "
                    f"target=({left:+.3f},{right:+.3f})m/s",
                    flush=True,
                )
                next_print_ns = now_ns + _STATUS_PERIOD_NS
            time.sleep(_LOOP_PERIOD_S)
    finally:
        controller.soft_brake()


def _run_replay_hardware(
    config_path: Path,
    replay_path: Path,
    *,
    supervised_stop_ready: bool,
) -> None:
    config = load_runtime_config(config_path)
    if not supervised_stop_ready:
        raise RuntimeError(
            "A physical emergency stop and continuous supervision are required."
        )
    if not config.uart.enabled or not config.motion.enabled:
        raise RuntimeError("keyboard_drive replay requires enabled uart and motion.")
    calibration = config.motion.odometry.build_calibration()
    if calibration is None:
        raise RuntimeError(
            "keyboard_drive replay requires motion.odometry.enabled=true "
            "with complete encoder/IMU calibration."
        )
    channel = config.uart.build_channel()
    controller = config.motion.build_controller(channel)
    if channel is None or controller is None:
        raise RuntimeError("Runtime config did not build a UART motion controller.")
    recording = load_keyboard_drive_log(replay_path, controller, calibration)
    reference = build_replay_reference(recording.telemetry, calibration)
    duration_s = reference[-1].elapsed_s
    if not _confirm_replay(replay_path, duration_s):
        print("未执行回放。", flush=True)
        return
    stop_event = threading.Event()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, lambda _signum, _frame: stop_event.set())
    try:
        with _RawKeyboard() as keyboard, channel:
            try:
                controller.synchronize(
                    timeout_s=config.motion.synchronization_timeout_s
                )
                if controller.emergency_stop_latched:
                    raise RuntimeError(
                        "STM32 emergency stop is latched; reset it first."
                    )
                print(
                    "开始回放；空格、q、Esc、Ctrl+C 或 SIGTERM 可随时软刹车。",
                    flush=True,
                )
                completed = replay_keyboard_recording(
                    controller,
                    recording,
                    calibration,
                    keyboard,
                    stop_event=stop_event,
                )
                print(
                    "回放完成。" if completed else "回放已由操作员停止。",
                    flush=True,
                )
            finally:
                controller.soft_brake()
    finally:
        signal.signal(signal.SIGTERM, previous_sigterm)


def _record_braking_tail(
    controller: object,
    command_log: KeyboardDriveLogWriter,
) -> None:
    """软刹车后继续记录实际停车行程，直到编码器稳定或有界超时。"""

    controller.soft_brake()
    started_ns = time.monotonic_ns()
    minimum_end_ns = started_ns + 250_000_000
    deadline_ns = started_ns + 1_500_000_000
    previous_counts: tuple[int, int] | None = None
    stable_samples = 0
    while time.monotonic_ns() < deadline_ns:
        for message in controller.drain_messages():
            if isinstance(message, OdometryImu):
                command_log.record_telemetry(message)
                counts = (
                    message.left_encoder_count,
                    message.right_encoder_count,
                )
                stable_samples = stable_samples + 1 if counts == previous_counts else 0
                previous_counts = counts
            elif isinstance(message, CarSystemStatus):
                if message.emergency_stop_latched:
                    return
        now_ns = time.monotonic_ns()
        if now_ns >= minimum_end_ns and stable_samples >= 5:
            return
        time.sleep(0.01)


def _run_hardware(
    config_path: Path,
    *,
    supervised_stop_ready: bool,
    linear_speed_m_s: float | None,
    angular_speed_rad_s: float | None,
    display: bool,
    enable_localization: bool,
    log_dir: Path,
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
    odometry_calibration = config.motion.odometry.build_calibration()
    if odometry_calibration is None:
        raise RuntimeError(
            "keyboard_drive recording requires motion.odometry.enabled=true "
            "with complete encoder/IMU calibration."
        )
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
    log_path = _default_keyboard_log_path(log_dir)

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
                log_started_ns = time.monotonic_ns()
                print(
                    "keyboard_drive: W/S latch forward/backward, hold A/D to steer, "
                    "space brake, X transport, "
                    "Esc/q quit; "
                    f"speed={linear_speed:.3f} m/s turn={angular_speed:.3f} rad/s; "
                    f"log={log_path}",
                    flush=True,
                )
                with KeyboardDriveLogWriter(
                    log_path,
                    controller,
                    odometry_calibration,
                    started_ns=log_started_ns,
                ) as command_log:
                    command_log.record(
                        log_started_ns,
                        linear_velocity_m_s=0.0,
                        angular_velocity_rad_s=0.0,
                        target_wheel_speeds_m_s=(0.0, 0.0),
                    )
                    try:
                        while not stop_requested:
                            if controller.needs_synchronization:
                                controller.synchronize(
                                    timeout_s=config.motion.synchronization_timeout_s,
                                    on_message=consume,
                                )
                            events = keyboard.poll()
                            now_ns = time.monotonic_ns()
                            drive_state.apply(events, now_ns)
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

                            active_keys = drive_state.active_keys(now_ns)
                            requested_linear, requested_angular = compute_twist(
                                active_keys,
                                linear_speed_m_s=linear_speed,
                                angular_speed_rad_s=angular_speed,
                            )
                            linear, angular = apply_keyboard_drive_target(
                                controller,
                                command_log,
                                requested_linear_m_s=requested_linear,
                                requested_angular_rad_s=requested_angular,
                            )
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
                                    # 只消费最新值，并持续排空 100 Hz 遥测。
                                    latest_odometry = message
                                    command_log.record_telemetry(message)
                                    if localization_fusion is not None:
                                        latest_pose_estimate = (
                                            localization_fusion.submit_odometry(message)
                                        )
                            camera_pump.check_health()
                            now_ns = time.monotonic_ns()

                            latest_snapshot = renderer.latest_fresh_snapshot(
                                now_ns,
                                config.processing.max_observation_age_ms,
                            )

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
                        controller.drive_wheel_limited(0.0, 0.0)
                        if not command_log.last_stopped:
                            command_log.record(
                                time.monotonic_ns(),
                                linear_velocity_m_s=0.0,
                                angular_velocity_rad_s=0.0,
                                target_wheel_speeds_m_s=(0.0, 0.0),
                                force=True,
                            )
                        _record_braking_tail(controller, command_log)
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
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory for timestamped keyboard command JSONL (default: logs).",
    )
    parser.add_argument(
        "--replay",
        type=Path,
        help=(
            "Replay a v2 keyboard_drive JSONL with encoder/IMU feedback "
            "instead of opening camera/Hailo."
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
        config_path = args.config.expanduser().resolve()
        if args.replay is not None:
            if args.display or args.enable_localization or args.speed_m_s or args.turn_rad_s:
                parser.error(
                    "--replay cannot be combined with --display, --enable-localization, "
                    "--speed-m-s or --turn-rad-s"
                )
            _run_replay_hardware(
                config_path,
                args.replay.expanduser().resolve(),
                supervised_stop_ready=args.supervised_physical_stop_ready,
            )
        else:
            _run_hardware(
                config_path,
                supervised_stop_ready=args.supervised_physical_stop_ready,
                linear_speed_m_s=args.speed_m_s,
                angular_speed_rad_s=args.turn_rad_s,
                display=args.display,
                enable_localization=args.enable_localization,
                log_dir=args.log_dir.expanduser().resolve(),
            )
    except KeyboardInterrupt:
        print("stopped_by_user=true", flush=True)


if __name__ == "__main__":
    main()
