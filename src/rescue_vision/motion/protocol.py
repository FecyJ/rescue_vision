"""Rescue Car v2.0 UART 命令编码与回传解析。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from rescue_vision.communication import ReceivedUartLine


class CarLineChannel(Protocol):
    """运动层所需的最小行通道接口。"""

    def send_line(self, payload: bytes) -> None: ...

    def receive_line(self, timeout: float | None = None) -> ReceivedUartLine: ...


def _finite_float(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return float(value)


def _non_negative_int(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{location} must be a non-negative integer, got {value!r}."
        )
    return value


def _format_float(value: object, location: str) -> str:
    converted = _finite_float(value, location)
    if converted == 0.0:
        converted = 0.0
    return format(converted, ".9g")


def encode_wheel_speed_command(left_m_s: float, right_m_s: float) -> bytes:
    """编码立即设置左右轮目标速度的 ``m`` 指令。"""

    left = _format_float(left_m_s, "left_m_s")
    right = _format_float(right_m_s, "right_m_s")
    return f"m{left},{right}".encode("ascii")


def encode_soft_brake_command(
    left_m_s: float = 0.0,
    right_m_s: float = 0.0,
) -> bytes:
    """编码斜坡制动到指定左右轮速度的 ``b`` 指令。"""

    left = _format_float(left_m_s, "left_m_s")
    right = _format_float(right_m_s, "right_m_s")
    return f"b{left},{right}".encode("ascii")


def encode_emergency_stop_command() -> bytes:
    """编码固件急停指令。"""

    return b"e"


def encode_state_query_command() -> bytes:
    """编码当前状态查询指令。"""

    return b"v"


@dataclass(frozen=True, slots=True)
class CarTelemetry:
    """STM32 的 10 Hz 遥测；速度为 m/s，舵机角度为 degree。"""

    uart_sequence: int
    received_timestamp_ns: int
    controller_timestamp_ms: int
    actual_left_m_s: float
    actual_right_m_s: float
    target_left_m_s: float
    target_right_m_s: float
    servo_left_deg: float
    servo_right_deg: float

    def __post_init__(self) -> None:
        for name in (
            "uart_sequence",
            "received_timestamp_ns",
            "controller_timestamp_ms",
        ):
            _non_negative_int(getattr(self, name), name)
        for name in (
            "actual_left_m_s",
            "actual_right_m_s",
            "target_left_m_s",
            "target_right_m_s",
            "servo_left_deg",
            "servo_right_deg",
        ):
            object.__setattr__(
                self,
                name,
                _finite_float(getattr(self, name), name),
            )
        for name in ("servo_left_deg", "servo_right_deg"):
            angle = getattr(self, name)
            if not 0.0 <= angle <= 180.0:
                raise ValueError(f"{name} must be in [0, 180], got {angle!r}.")


class CarStopReason(str, Enum):
    """STM32 安全状态报告的当前停车/运行原因。"""

    STARTUP = "startup"
    RUNNING = "running"
    SOFT_BRAKE = "soft_brake"
    WATCHDOG_TIMEOUT = "watchdog_timeout"
    EMERGENCY_STOP = "emergency_stop"


@dataclass(frozen=True, slots=True)
class CarSafetyStatus:
    """版本化 ``s1`` 安全状态；所有时间均来自 STM32 单调时钟。"""

    uart_sequence: int
    received_timestamp_ns: int
    controller_timestamp_ms: int
    watchdog_timeout_ms: int
    watchdog_armed: bool
    emergency_stop_latched: bool
    last_motion_command_age_ms: int | None
    stop_reason: CarStopReason

    def __post_init__(self) -> None:
        for name in (
            "uart_sequence",
            "received_timestamp_ns",
            "controller_timestamp_ms",
        ):
            _non_negative_int(getattr(self, name), name)
        if (
            isinstance(self.watchdog_timeout_ms, bool)
            or not isinstance(self.watchdog_timeout_ms, int)
            or self.watchdog_timeout_ms <= 0
        ):
            raise ValueError(
                "watchdog_timeout_ms must be a positive integer, "
                f"got {self.watchdog_timeout_ms!r}."
            )
        for name in ("watchdog_armed", "emergency_stop_latched"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean.")
        if self.last_motion_command_age_ms is not None:
            _non_negative_int(
                self.last_motion_command_age_ms,
                "last_motion_command_age_ms",
            )
        if not isinstance(self.stop_reason, CarStopReason):
            raise ValueError(
                "stop_reason must be a CarStopReason, "
                f"got {self.stop_reason!r}."
            )


@dataclass(frozen=True, slots=True)
class CarCommandReply:
    """STM32 对一条命令的成功或错误回复。"""

    uart_sequence: int
    received_timestamp_ns: int
    succeeded: bool
    detail: str

    def __post_init__(self) -> None:
        _non_negative_int(self.uart_sequence, "uart_sequence")
        _non_negative_int(self.received_timestamp_ns, "received_timestamp_ns")
        if not isinstance(self.succeeded, bool):
            raise ValueError("succeeded must be a boolean.")
        if not isinstance(self.detail, str):
            raise ValueError("detail must be a string.")


@dataclass(frozen=True, slots=True)
class UnknownCarMessage:
    """保留未知前缀，供同一 UART 后续扩展 IMU 等消息。"""

    uart_sequence: int
    received_timestamp_ns: int
    payload: bytes

    def __post_init__(self) -> None:
        _non_negative_int(self.uart_sequence, "uart_sequence")
        _non_negative_int(self.received_timestamp_ns, "received_timestamp_ns")
        if not isinstance(self.payload, bytes) or not self.payload:
            raise ValueError("payload must be non-empty bytes.")


ParsedCarMessage = (
    CarTelemetry | CarSafetyStatus | CarCommandReply | UnknownCarMessage
)


def parse_car_line(line: ReceivedUartLine) -> ParsedCarMessage:
    """解析一条 Rescue Car 回传，同时保留树莓派接收时间。"""

    if not isinstance(line, ReceivedUartLine):
        raise TypeError(
            f"line must be ReceivedUartLine, got {type(line).__name__}."
        )
    try:
        text = line.payload.decode("ascii")
    except UnicodeDecodeError:
        # 串口噪声、调试输出或尚未支持的二进制扩展不能使受监督控制循环退出。
        # 原始 bytes 会由运动日志以十六进制保存，供现场追查。
        return UnknownCarMessage(
            line.sequence,
            line.received_timestamp_ns,
            line.payload,
        )

    if text == "OK" or text.startswith("OK "):
        return CarCommandReply(
            line.sequence,
            line.received_timestamp_ns,
            True,
            text[2:].strip(),
        )
    if text.startswith("ERR:"):
        return CarCommandReply(
            line.sequence,
            line.received_timestamp_ns,
            False,
            text[3:].lstrip(": "),
        )
    if text.startswith("s1,"):
        return _parse_safety_status(line, text)
    if not text.startswith("t"):
        return UnknownCarMessage(
            line.sequence,
            line.received_timestamp_ns,
            line.payload,
        )

    fields = text[1:].split(",")
    if len(fields) != 7:
        raise ValueError(
            "Car telemetry must contain controller timestamp and six values, "
            f"got {len(fields)} fields in {text!r}."
        )
    try:
        controller_timestamp_ms = int(fields[0])
    except ValueError as exc:
        raise ValueError(
            f"Invalid controller timestamp in telemetry {text!r}."
        ) from exc
    if controller_timestamp_ms < 0:
        raise ValueError(
            "controller_timestamp_ms must be non-negative, "
            f"got {controller_timestamp_ms!r}."
        )
    values = tuple(
        _finite_float(
            float(value),
            f"telemetry[{index + 1}]",
        )
        for index, value in enumerate(fields[1:])
    )
    return CarTelemetry(
        uart_sequence=line.sequence,
        received_timestamp_ns=line.received_timestamp_ns,
        controller_timestamp_ms=controller_timestamp_ms,
        actual_left_m_s=values[0],
        actual_right_m_s=values[1],
        target_left_m_s=values[2],
        target_right_m_s=values[3],
        servo_left_deg=values[4],
        servo_right_deg=values[5],
    )


def _parse_safety_status(
    line: ReceivedUartLine,
    text: str,
) -> CarSafetyStatus:
    fields = text.split(",")
    if len(fields) != 7:
        raise ValueError(
            "Car safety status s1 must contain six values, "
            f"got {len(fields) - 1} in {text!r}."
        )
    integer_names = (
        "controller_timestamp_ms",
        "watchdog_timeout_ms",
        "watchdog_armed",
        "emergency_stop_latched",
        "last_motion_command_age_ms",
    )
    parsed: list[int] = []
    for name, value in zip(integer_names, fields[1:6], strict=True):
        try:
            parsed.append(int(value))
        except ValueError as exc:
            raise ValueError(
                f"Invalid {name} in car safety status {text!r}."
            ) from exc
    controller_timestamp_ms, watchdog_timeout_ms, armed, latched, age = parsed
    if controller_timestamp_ms < 0:
        raise ValueError("controller_timestamp_ms must be non-negative.")
    if watchdog_timeout_ms <= 0:
        raise ValueError("watchdog_timeout_ms must be positive.")
    if armed not in (0, 1):
        raise ValueError("watchdog_armed must be encoded as 0 or 1.")
    if latched not in (0, 1):
        raise ValueError(
            "emergency_stop_latched must be encoded as 0 or 1."
        )
    if age < -1:
        raise ValueError(
            "last_motion_command_age_ms must be -1 or non-negative."
        )
    try:
        stop_reason = CarStopReason(fields[6])
    except ValueError as exc:
        raise ValueError(
            f"Unsupported stop_reason in car safety status {text!r}."
        ) from exc
    return CarSafetyStatus(
        uart_sequence=line.sequence,
        received_timestamp_ns=line.received_timestamp_ns,
        controller_timestamp_ms=controller_timestamp_ms,
        watchdog_timeout_ms=watchdog_timeout_ms,
        watchdog_armed=bool(armed),
        emergency_stop_latched=bool(latched),
        last_motion_command_age_ms=None if age == -1 else age,
        stop_reason=stop_reason,
    )
