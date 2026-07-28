"""Rescue Car v2.0 UART 命令编码与回传解析。"""

from __future__ import annotations

import math
from dataclasses import dataclass
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


ParsedCarMessage = CarTelemetry | CarCommandReply | UnknownCarMessage


def parse_car_line(line: ReceivedUartLine) -> ParsedCarMessage:
    """解析一条 Rescue Car 回传，同时保留树莓派接收时间。"""

    if not isinstance(line, ReceivedUartLine):
        raise TypeError(
            f"line must be ReceivedUartLine, got {type(line).__name__}."
        )
    try:
        text = line.payload.decode("ascii")
    except UnicodeDecodeError as exc:
        raise ValueError("Car UART line must contain ASCII bytes.") from exc

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
