"""Rescue Car 双轮差速与夹爪舵机控制函数。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from rescue_vision.motion.protocol import (
    CarFrameChannel,
    CarSystemStatus,
    ControllerProtocolError,
    ParsedCarMessage,
    encode_emergency_stop_command,
    encode_gripper_command,
    encode_soft_brake_command,
    encode_state_query_command,
    encode_wheel_speed_command,
    parse_controller_frame,
)


_WHEEL_COMMAND_REFRESH_NS = 40_000_000
_MAX_ACTIVE_UPDATE_GAP_NS = 100_000_000


class MotionControlTimingError(RuntimeError):
    """活动运动控制循环停顿过久，已先发送柔和停车。"""


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
    max_wheel_acceleration_m_s2: float
    max_remote_command_valid_for_ms: int

    def __post_init__(self) -> None:
        for name in (
            "wheel_track_m",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_wheel_velocity_m_s",
            "max_wheel_acceleration_m_s2",
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
    """通过同一 COBS 帧通道驱动 STM32 底盘和夹爪。

    `drive()` 的车体 twist 参考点是两驱动轮接地点连线的中点，与机器人
    地面坐标系原点一致。
    """

    def __init__(
        self,
        channel: CarFrameChannel,
        limits: MotionLimits,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(limits, MotionLimits):
            raise TypeError("limits must be MotionLimits.")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable.")
        self._channel = channel
        self.limits = limits
        self._monotonic_ns = monotonic_ns
        self._target_wheel_speeds_m_s = (0.0, 0.0)
        self._commanded_wheel_speeds_m_s = (0.0, 0.0)
        self._last_acceleration_update_ns = self._now()
        self._last_wheel_command_ns = self._last_acceleration_update_ns
        self._command_sequence = 0
        self._gripper_target_angles_deg: tuple[float, float] | None = None
        self._has_gripper_command = False
        self.invalid_received_frames = 0

    @property
    def target_wheel_speeds_m_s(self) -> tuple[float, float]:
        """返回最近请求的左右轮目标速度。"""

        return self._target_wheel_speeds_m_s

    @property
    def commanded_wheel_speeds_m_s(self) -> tuple[float, float]:
        """返回加速度限制后最近下发的左右轮速度。"""

        return self._commanded_wheel_speeds_m_s

    @property
    def gripper_target_angles_deg(self) -> tuple[float, float] | None:
        """返回启动遥测或本进程最近下发的左右舵机目标角度。"""

        return self._gripper_target_angles_deg

    def set_wheel_speeds(
        self,
        left_m_s: float,
        right_m_s: float,
    ) -> None:
        """设置左右轮目标速度；实际下发由 :meth:`update` 渐进逼近。"""

        left = _finite(left_m_s, "left_m_s")
        right = _finite(right_m_s, "right_m_s")
        maximum = self.limits.max_wheel_velocity_m_s
        if abs(left) > maximum or abs(right) > maximum:
            raise ValueError(
                "Wheel velocity exceeds configured limit "
                f"{maximum} m/s: left={left}, right={right}."
            )
        # 先按旧目标结算到“新目标生效”的时刻，避免把此前的静止空闲时间
        # 错算成新目标可用的加速时间。
        self.update()
        self._target_wheel_speeds_m_s = (left, right)

    def update(self, *, now_ns: int | None = None) -> bool:
        """按单轮最大加速度推进目标并下发；有新命令时返回 ``True``。"""

        current_ns = self._now(now_ns)
        if current_ns < self._last_acceleration_update_ns:
            raise ValueError(
                "now_ns must not precede the previous acceleration update: "
                f"{current_ns} < {self._last_acceleration_update_ns}."
            )
        elapsed_s = (
            current_ns - self._last_acceleration_update_ns
        ) / 1_000_000_000.0
        elapsed_ns = current_ns - self._last_acceleration_update_ns
        if elapsed_ns > _MAX_ACTIVE_UPDATE_GAP_NS and any(
            value != 0.0
            for value in (
                *self._target_wheel_speeds_m_s,
                *self._commanded_wheel_speeds_m_s,
            )
        ):
            gap_ms = elapsed_ns / 1_000_000.0
            self.soft_brake()
            raise MotionControlTimingError(
                "Active motion update gap exceeded 100 ms; "
                f"soft brake was sent after {gap_ms:.3f} ms."
            )
        maximum_delta = self.limits.max_wheel_acceleration_m_s2 * elapsed_s
        previous_left, previous_right = self._commanded_wheel_speeds_m_s
        target_left, target_right = self._target_wheel_speeds_m_s
        next_left = _move_toward(previous_left, target_left, maximum_delta)
        next_right = _move_toward(previous_right, target_right, maximum_delta)
        self._last_acceleration_update_ns = current_ns
        changed = next_left != previous_left or next_right != previous_right
        refresh_due = (
            current_ns - self._last_wheel_command_ns
            >= _WHEEL_COMMAND_REFRESH_NS
        )
        if not changed and not refresh_due:
            return False
        sequence = self._next_command_sequence()
        self._channel.send_frame(
            encode_wheel_speed_command(sequence, next_left, next_right)
        )
        self._commanded_wheel_speeds_m_s = (next_left, next_right)
        self._last_wheel_command_ns = current_ns
        return True

    def drive(
        self,
        linear_velocity_m_s: float,
        angular_velocity_rad_s: float,
    ) -> None:
        """执行车体 twist；角速度逆时针（左转）为正。"""

        linear, angular = self._validate_twist(
            linear_velocity_m_s,
            angular_velocity_rad_s,
        )
        half_track = self.limits.wheel_track_m / 2.0
        self.set_wheel_speeds(
            linear - angular * half_track,
            linear + angular * half_track,
        )

    def drive_wheel_limited(
        self,
        linear_velocity_m_s: float,
        angular_velocity_rad_s: float,
    ) -> tuple[float, float]:
        """按比例缩放合法 twist，使差速合成不超过单轮硬上限。

        返回实际采用的 ``(linear_velocity_m_s, angular_velocity_rad_s)``。
        线速度或角速度自身超限时仍拒绝，不以缩放掩盖非法请求。
        """

        linear, angular = self._validate_twist(
            linear_velocity_m_s,
            angular_velocity_rad_s,
        )
        half_track = self.limits.wheel_track_m / 2.0
        left = linear - angular * half_track
        right = linear + angular * half_track
        peak_wheel_speed = max(abs(left), abs(right))
        maximum = self.limits.max_wheel_velocity_m_s
        if peak_wheel_speed > maximum:
            # 朝零取下一浮点数，避免除乘舍入让理论边界重新高出硬上限。
            scale = math.nextafter(maximum / peak_wheel_speed, 0.0)
            linear *= scale
            angular *= scale
            left *= scale
            right *= scale
        self.set_wheel_speeds(left, right)
        return linear, angular

    def _validate_twist(
        self,
        linear_velocity_m_s: float,
        angular_velocity_rad_s: float,
    ) -> tuple[float, float]:
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
        return linear, angular

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

    def set_gripper_angles(
        self,
        left_angle_deg: float,
        right_angle_deg: float,
    ) -> None:
        """同时设置左右夹爪舵机角度，单位 degree。"""

        payload = encode_gripper_command(
            self._next_command_sequence(),
            left_angle_deg,
            right_angle_deg,
        )
        self._channel.send_frame(payload)
        self._gripper_target_angles_deg = (
            float(left_angle_deg),
            float(right_angle_deg),
        )
        self._has_gripper_command = True

    def soft_brake(self) -> None:
        """按固件减速度斜坡制动到静止。"""

        self._channel.send_frame(
            encode_soft_brake_command(self._next_command_sequence())
        )
        self._reset_acceleration_state()

    def emergency_stop(self) -> None:
        """触发固件急停。"""

        self._channel.send_frame(
            encode_emergency_stop_command(self._next_command_sequence())
        )
        self._reset_acceleration_state()

    def query_state(self) -> None:
        """请求固件立即返回当前状态。"""

        self._channel.send_frame(
            encode_state_query_command(self._next_command_sequence())
        )

    def receive_message(self, timeout: float | None = None) -> ParsedCarMessage:
        """接收一条合法 STM32 消息；损坏、未知和方向错误的帧直接丢弃。"""

        deadline_ns: int | None = None
        wait_s = timeout
        if timeout is not None:
            wait_s = _non_negative(timeout, "timeout")
            deadline_ns = self._now() + round(wait_s * 1_000_000_000)
        while True:
            frame = self._channel.receive_frame(timeout=wait_s)
            try:
                message = parse_controller_frame(frame)
            except ControllerProtocolError:
                self.invalid_received_frames += 1
                if deadline_ns is not None:
                    wait_s = max(
                        0.0,
                        (deadline_ns - self._now()) / 1_000_000_000.0,
                    )
                continue
            if (
                isinstance(message, CarSystemStatus)
                and not self._has_gripper_command
            ):
                self._gripper_target_angles_deg = (
                    message.servo_left_deg,
                    message.servo_right_deg,
                )
            return message

    def drain_messages(self) -> tuple[ParsedCarMessage, ...]:
        """非阻塞排空当前回传，防止 100 Hz 定位遥测挤满 UART 队列。"""

        messages: list[ParsedCarMessage] = []
        while True:
            try:
                messages.append(self.receive_message(timeout=0))
            except TimeoutError:
                return tuple(messages)

    def _reset_acceleration_state(self) -> None:
        self._target_wheel_speeds_m_s = (0.0, 0.0)
        self._commanded_wheel_speeds_m_s = (0.0, 0.0)
        current_ns = self._now()
        self._last_acceleration_update_ns = current_ns
        self._last_wheel_command_ns = current_ns

    def _next_command_sequence(self) -> int:
        sequence = self._command_sequence
        self._command_sequence = (sequence + 1) & 0xFFFF
        return sequence

    def _now(self, value: int | None = None) -> int:
        current = self._monotonic_ns() if value is None else value
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            raise ValueError(
                f"now_ns must be a non-negative integer, got {current!r}."
            )
        return current


def _non_negative(value: object, location: str) -> float:
    converted = _finite(value, location)
    if converted < 0.0:
        raise ValueError(f"{location} must be >= 0, got {converted!r}.")
    return converted


def _move_toward(current: float, target: float, maximum_delta: float) -> float:
    delta = target - current
    distance = abs(delta)
    if distance <= maximum_delta or math.isclose(
        distance,
        maximum_delta,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return target
    if delta > 0.0:
        return current + maximum_delta
    return current - maximum_delta
