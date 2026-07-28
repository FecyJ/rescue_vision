"""双轮差速小车运动函数。"""

from __future__ import annotations

import math
from dataclasses import dataclass

from rescue_vision.motion.protocol import (
    CarLineChannel,
    ParsedCarMessage,
    UnknownCarMessage,
    encode_emergency_stop_command,
    encode_soft_brake_command,
    encode_state_query_command,
    encode_wheel_speed_command,
    parse_car_line,
)


def _positive_finite(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not 0.0 < float(value) < math.inf
    ):
        raise ValueError(f"{location} must be finite and > 0, got {value!r}.")
    return float(value)


def _finite(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return float(value)


@dataclass(frozen=True, slots=True)
class MotionLimits:
    """小车运动学参数和赛外调试限速。"""

    wheel_track_m: float
    max_linear_velocity_m_s: float
    max_angular_velocity_rad_s: float
    max_wheel_velocity_m_s: float
    max_remote_command_valid_for_ms: int

    def __post_init__(self) -> None:
        for name in (
            "wheel_track_m",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_wheel_velocity_m_s",
        ):
            object.__setattr__(
                self,
                name,
                _positive_finite(getattr(self, name), name),
            )
        if (
            isinstance(self.max_remote_command_valid_for_ms, bool)
            or not isinstance(self.max_remote_command_valid_for_ms, int)
            or not 1 <= self.max_remote_command_valid_for_ms <= 5_000
        ):
            raise ValueError(
                "max_remote_command_valid_for_ms must be an integer in "
                f"[1, 5000], got {self.max_remote_command_valid_for_ms!r}."
            )


class MotionController:
    """通过行通道驱动 Rescue Car 双轮差速底盘。"""

    def __init__(
        self,
        channel: CarLineChannel,
        limits: MotionLimits,
    ) -> None:
        if not isinstance(limits, MotionLimits):
            raise TypeError("limits must be MotionLimits.")
        self._channel = channel
        self.limits = limits

    def set_wheel_speeds(
        self,
        left_m_s: float,
        right_m_s: float,
    ) -> None:
        """立即设置左右轮速度。正值前进，单位 m/s。"""

        left = _finite(left_m_s, "left_m_s")
        right = _finite(right_m_s, "right_m_s")
        maximum = self.limits.max_wheel_velocity_m_s
        if abs(left) > maximum or abs(right) > maximum:
            raise ValueError(
                "Wheel velocity exceeds configured limit "
                f"{maximum} m/s: left={left}, right={right}."
            )
        self._channel.send_line(encode_wheel_speed_command(left, right))

    def drive(
        self,
        linear_velocity_m_s: float,
        angular_velocity_rad_s: float,
    ) -> None:
        """执行车体 twist；角速度逆时针（左转）为正。"""

        linear = _finite(linear_velocity_m_s, "linear_velocity_m_s")
        angular = _finite(angular_velocity_rad_s, "angular_velocity_rad_s")
        if abs(linear) > self.limits.max_linear_velocity_m_s:
            raise ValueError(
                "linear_velocity_m_s exceeds configured limit "
                f"{self.limits.max_linear_velocity_m_s}: {linear}."
            )
        if abs(angular) > self.limits.max_angular_velocity_rad_s:
            raise ValueError(
                "angular_velocity_rad_s exceeds configured limit "
                f"{self.limits.max_angular_velocity_rad_s}: {angular}."
            )
        half_track = self.limits.wheel_track_m / 2.0
        self.set_wheel_speeds(
            linear - angular * half_track,
            linear + angular * half_track,
        )

    def forward(self, speed_m_s: float) -> None:
        """以非负速度直行前进。"""

        self.drive(_non_negative(speed_m_s, "speed_m_s"), 0.0)

    def backward(self, speed_m_s: float) -> None:
        """以非负速度直行后退。"""

        self.drive(-_non_negative(speed_m_s, "speed_m_s"), 0.0)

    def turn_left(self, angular_velocity_rad_s: float) -> None:
        """以非负角速度原地左转。"""

        self.drive(0.0, _non_negative(angular_velocity_rad_s, "angular_velocity_rad_s"))

    def turn_right(self, angular_velocity_rad_s: float) -> None:
        """以非负角速度原地右转。"""

        angular = _non_negative(
            angular_velocity_rad_s,
            "angular_velocity_rad_s",
        )
        self.drive(0.0, -angular)

    def soft_brake(self) -> None:
        """按固件减速度斜坡制动到静止。"""

        self._channel.send_line(encode_soft_brake_command())

    def emergency_stop(self) -> None:
        """触发固件急停。"""

        self._channel.send_line(encode_emergency_stop_command())

    def query_state(self) -> None:
        """请求固件立即返回当前状态。"""

        self._channel.send_line(encode_state_query_command())

    def receive_message(self, timeout: float | None = None) -> ParsedCarMessage:
        """接收回传；损坏的单行隔离为未知消息，不中断实时循环。"""

        line = self._channel.receive_line(timeout=timeout)
        try:
            return parse_car_line(line)
        except ValueError:
            return UnknownCarMessage(
                uart_sequence=line.sequence,
                received_timestamp_ns=line.received_timestamp_ns,
                payload=line.payload,
            )

    def drain_messages(self) -> tuple[ParsedCarMessage, ...]:
        """非阻塞排空当前回传，防止 10 Hz 遥测挤满 UART 队列。"""

        messages: list[ParsedCarMessage] = []
        while True:
            try:
                messages.append(self.receive_message(timeout=0))
            except TimeoutError:
                return tuple(messages)


def _non_negative(value: object, location: str) -> float:
    converted = _finite(value, location)
    if converted < 0.0:
        raise ValueError(f"{location} must be >= 0, got {converted!r}.")
    return converted
