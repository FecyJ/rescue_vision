"""Rescue Car 双轮差速与夹爪舵机控制函数。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass

from rescue_vision.motion.protocol import (
    CarCommandReply,
    CarFrameChannel,
    CarSystemStatus,
    CommandResult,
    ControllerProtocolError,
    MessageType,
    OdometryImu,
    ParsedCarMessage,
    SensorFlags,
    encode_emergency_stop_command,
    encode_gripper_command,
    encode_soft_brake_command,
    encode_state_query_command,
    encode_wheel_speed_command,
    parse_controller_frame,
)


_WHEEL_COMMAND_REFRESH_NS = 40_000_000
_MAX_ACTIVE_UPDATE_GAP_NS = 200_000_000
_WHEEL_REPLY_TIMEOUT_NS = 100_000_000
_MAX_PENDING_WHEEL_COMMANDS = 4


class MotionControlTimingError(RuntimeError):
    """活动运动控制循环停顿过久，已先发送柔和停车。"""


class MotionStallError(RuntimeError):
    """编码器在持续轮速命令下没有变化，已先发送柔和停车。"""


class MotionSynchronizationError(RuntimeError):
    """STM32 运动序号安全同步未在规定时间内完成。"""


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
    min_wheel_velocity_m_s: float = 0.02
    left_wheel_speed_weight: float = 1.0
    right_wheel_speed_weight: float = 1.0
    stall_guard_enabled: bool = True
    stall_guard_timeout_ms: int = 300
    stall_guard_min_command_speed_m_s: float = 0.02
    stall_guard_stationary_encoder_delta_count: int = 0

    def __post_init__(self) -> None:
        for name in (
            "wheel_track_m",
            "max_linear_velocity_m_s",
            "max_angular_velocity_rad_s",
            "max_wheel_velocity_m_s",
            "max_wheel_acceleration_m_s2",
            "min_wheel_velocity_m_s",
            "left_wheel_speed_weight",
            "right_wheel_speed_weight",
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
        if not isinstance(self.stall_guard_enabled, bool):
            raise ValueError("stall_guard_enabled must be a boolean.")
        if (
            isinstance(self.stall_guard_timeout_ms, bool)
            or not isinstance(self.stall_guard_timeout_ms, int)
            or not 1 <= self.stall_guard_timeout_ms <= 5_000
        ):
            raise ValueError(
                "stall_guard_timeout_ms must be an integer in [1, 5000], "
                f"got {self.stall_guard_timeout_ms!r}."
            )
        object.__setattr__(
            self,
            "stall_guard_min_command_speed_m_s",
            _positive_finite(
                self.stall_guard_min_command_speed_m_s,
                "stall_guard_min_command_speed_m_s",
            ),
        )
        if self.stall_guard_min_command_speed_m_s > self.max_wheel_velocity_m_s:
            raise ValueError(
                "stall_guard_min_command_speed_m_s must not exceed "
                "max_wheel_velocity_m_s."
            )
        if self.min_wheel_velocity_m_s > self.max_wheel_velocity_m_s:
            raise ValueError(
                "min_wheel_velocity_m_s must not exceed max_wheel_velocity_m_s, "
                f"got {self.min_wheel_velocity_m_s!r} > "
                f"{self.max_wheel_velocity_m_s!r}."
            )
        if (
            isinstance(self.stall_guard_stationary_encoder_delta_count, bool)
            or not isinstance(
                self.stall_guard_stationary_encoder_delta_count,
                int,
            )
            or self.stall_guard_stationary_encoder_delta_count < 0
        ):
            raise ValueError(
                "stall_guard_stationary_encoder_delta_count must be a "
                "non-negative integer, got "
                f"{self.stall_guard_stationary_encoder_delta_count!r}."
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
        self._wheel_acceleration_limit_m_s2 = (
            limits.max_wheel_acceleration_m_s2
        )
        self._last_sent_wheel_speeds_mm_s = (0, 0)
        self._last_acceleration_update_ns = self._now()
        self._last_wheel_command_ns = self._last_acceleration_update_ns
        self._command_sequence = 0
        self._gripper_target_angles_deg: tuple[float, float] | None = None
        self._has_gripper_command = False
        self._pending_wheel_commands: dict[int, int] = {}
        self._last_wheel_ack_ns: int | None = None
        self._synchronization_enforced = False
        self._synchronization_sequence: int | None = None
        self._synchronization_acknowledged = False
        self._emergency_stop_latched = False
        self._link_degraded = False
        self._latest_protocol_healthy = True
        self.invalid_received_frames = 0
        self._last_encoder_counts: tuple[int, int] | None = None
        self._stall_started_ns: list[int | None] = [None, None]

    @property
    def target_wheel_speeds_m_s(self) -> tuple[float, float]:
        """返回最近请求的左右轮目标速度。"""

        return self._target_wheel_speeds_m_s

    @property
    def commanded_wheel_speeds_m_s(self) -> tuple[float, float]:
        """返回加速度限制后最近下发的左右轮速度。"""

        return self._commanded_wheel_speeds_m_s

    @property
    def wheel_acceleration_limit_m_s2(self) -> float:
        """返回当前生效的单轮最大加速度，单位 m/s²。"""

        return self._wheel_acceleration_limit_m_s2

    def set_wheel_acceleration_limit_m_s2(
        self,
        acceleration_m_s2: float | None,
    ) -> None:
        """设置临时单轮加速度上限；传入 ``None`` 恢复全局配置值。

        临时值可以独立于 ``MotionLimits`` 的全局基准。该方法只改变后续
        ``update()`` 的速度斜坡，不会立即发送轮速或改变当前目标速度。
        """

        if acceleration_m_s2 is None:
            self._wheel_acceleration_limit_m_s2 = (
                self.limits.max_wheel_acceleration_m_s2
            )
            return
        acceleration = _positive_finite(
            acceleration_m_s2,
            "acceleration_m_s2",
        )
        self._wheel_acceleration_limit_m_s2 = acceleration

    @property
    def gripper_target_angles_deg(self) -> tuple[float, float] | None:
        """返回启动遥测或本进程最近下发的左右舵机目标角度。"""

        return self._gripper_target_angles_deg

    @property
    def pending_wheel_command_count(self) -> int:
        """返回尚未收到 `COMMAND_REPLY` 的轮速命令数量。"""

        return len(self._pending_wheel_commands)

    @property
    def last_wheel_ack_timestamp_ns(self) -> int | None:
        """返回最近一条 `SET_WHEEL_SPEED accepted` 的本机时间。"""

        return self._last_wheel_ack_ns

    @property
    def motion_synchronized(self) -> bool:
        """返回当前运动序号是否已通过 `SOFT_BRAKE` 安全同步。"""

        return not self._synchronization_enforced or (
            self._synchronization_acknowledged
            and self._synchronization_sequence is None
        )

    @property
    def needs_synchronization(self) -> bool:
        """返回生产控制循环是否必须等待 `SOFT_BRAKE` 回复。"""

        return self._synchronization_enforced and not self.motion_synchronized

    @property
    def emergency_stop_latched(self) -> bool:
        """返回是否已经观察到急停锁存；协议没有远程解除动作。"""

        return self._emergency_stop_latched

    @property
    def link_degraded(self) -> bool:
        """返回是否观察到仍会阻止运动的 STM32 链路告警。

        ``rx_degraded`` 是 STM32 本次启动期间的历史接收告警，只保留在
        ``CarSystemStatus`` 中供记录和诊断，不单独阻止树莓派继续控制。
        """

        return self._link_degraded

    def synchronize(
        self,
        *,
        timeout_s: float = 1.0,
        on_message: Callable[[ParsedCarMessage], None] | None = None,
    ) -> None:
        """发送 `SOFT_BRAKE` 并等待同序号的 `accepted` 回复。

        该方法应在每次打开 UART、重连或检测到运动序号/链路异常后调用。
        运动遥测可能在等待期间到达；若提供 ``on_message``，这些消息会按
        到达顺序交给调用方，避免为等待回复而丢弃定位数据。
        超时或明确拒绝会清理本次待同步序号，调用方可再次调用以发送新的
        ``SOFT_BRAKE`` 尝试。
        """

        if (
            isinstance(timeout_s, bool)
            or not isinstance(timeout_s, (int, float))
            or not 0.0 < float(timeout_s) < float("inf")
        ):
            raise ValueError(
                f"timeout_s must be finite and > 0, got {timeout_s!r}."
            )
        if on_message is not None and not callable(on_message):
            raise TypeError("on_message must be callable or None.")

        self._synchronization_enforced = True
        if self._synchronization_sequence is None:
            self._send_soft_brake()
        sequence = self._synchronization_sequence
        assert sequence is not None
        deadline_ns = self._now() + round(float(timeout_s) * 1_000_000_000)
        while not self.motion_synchronized:
            remaining_ns = deadline_ns - self._now()
            if remaining_ns <= 0:
                self._clear_synchronization_attempt()
                raise MotionSynchronizationError(
                    "Timed out waiting for SOFT_BRAKE accepted reply "
                    f"for command_sequence={sequence}."
                )
            try:
                message = self.receive_message(
                    timeout=remaining_ns / 1_000_000_000.0
                )
            except TimeoutError as exc:
                self._clear_synchronization_attempt()
                raise MotionSynchronizationError(
                    "Timed out waiting for SOFT_BRAKE accepted reply "
                    f"for command_sequence={sequence}."
                ) from exc
            if on_message is not None:
                on_message(message)

        if self._emergency_stop_latched:
            # SOFT_BRAKE is allowed to acknowledge while the emergency stop is
            # latched, but it must never make motion available again.
            return
        if self._latest_protocol_healthy:
            self._link_degraded = False

    def set_wheel_speeds(
        self,
        left_m_s: float,
        right_m_s: float,
        *,
        min_wheel_velocity_m_s: float | None = None,
    ) -> None:
        """设置左右轮目标速度；实际下发由 :meth:`update` 渐进逼近。"""

        minimum = self._resolve_minimum_wheel_velocity(
            min_wheel_velocity_m_s
        )
        left = self._apply_minimum_wheel_velocity(
            _finite(left_m_s, "left_m_s"),
            minimum,
        )
        right = self._apply_minimum_wheel_velocity(
            _finite(right_m_s, "right_m_s"),
            minimum,
        )
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
        if self._emergency_stop_latched:
            self._reset_acceleration_state_at(current_ns)
            return False
        if self._link_degraded:
            self._reset_acceleration_state_at(current_ns)
            return False
        if self.needs_synchronization:
            self._reset_acceleration_state_at(current_ns)
            return False
        if self._synchronization_enforced and self._wheel_ack_health_failed(
            current_ns
        ):
            self._request_synchronization()
            return True
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
                "Active motion update gap exceeded 200 ms; "
                f"soft brake was sent after {gap_ms:.3f} ms."
            )
        maximum_delta = self._wheel_acceleration_limit_m_s2 * elapsed_s
        previous_left, previous_right = self._commanded_wheel_speeds_m_s
        target_left, target_right = self._target_wheel_speeds_m_s
        next_left = _move_toward(previous_left, target_left, maximum_delta)
        next_right = _move_toward(previous_right, target_right, maximum_delta)
        self._last_acceleration_update_ns = current_ns
        next_wire_speeds_mm_s = (
            round(next_left * 1000.0),
            round(next_right * 1000.0),
        )
        refresh_due = (
            current_ns - self._last_wheel_command_ns
            >= _WHEEL_COMMAND_REFRESH_NS
        )
        self._commanded_wheel_speeds_m_s = (next_left, next_right)
        # The application safety loop may run every 5 ms, but the STM32
        # command/reply path is intentionally refreshed at 25 Hz.  Sending
        # every acceleration quantization step creates more outstanding
        # replies than the bounded UART path can acknowledge and causes the
        # safety resynchronization to soft-brake an otherwise healthy drive.
        if not refresh_due:
            return False
        if self._synchronization_enforced and self._pending_wheel_commands:
            # Keep advancing the local slew-limited value, then send the
            # newest value after the previous command is acknowledged.  A
            # missing reply still reaches _wheel_ack_health_failed above.
            return False
        sequence = self._next_command_sequence()
        self._channel.send_frame(
            encode_wheel_speed_command(sequence, next_left, next_right)
        )
        if self._synchronization_enforced:
            self._pending_wheel_commands[sequence] = current_ns
        self._last_sent_wheel_speeds_mm_s = next_wire_speeds_mm_s
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
        left, right = self._twist_to_wheel_speeds(linear, angular)
        self.set_wheel_speeds(left, right)

    def drive_wheel_limited(
        self,
        linear_velocity_m_s: float,
        angular_velocity_rad_s: float,
        *,
        min_wheel_velocity_m_s: float | None = None,
    ) -> tuple[float, float]:
        """按比例缩放合法 twist，使差速合成不超过单轮硬上限。

        返回实际采用的 ``(linear_velocity_m_s, angular_velocity_rad_s)``。
        线速度或角速度自身超限时仍拒绝，不以缩放掩盖非法请求。
        """

        linear, angular = self._validate_twist(
            linear_velocity_m_s,
            angular_velocity_rad_s,
        )
        left, right = self._twist_to_wheel_speeds(linear, angular)
        peak_wheel_speed = max(abs(left), abs(right))
        maximum = self.limits.max_wheel_velocity_m_s
        if peak_wheel_speed > maximum:
            # 朝零取下一浮点数，避免除乘舍入让理论边界重新高出硬上限。
            scale = math.nextafter(maximum / peak_wheel_speed, 0.0)
            linear *= scale
            angular *= scale
            left *= scale
            right *= scale
        self.set_wheel_speeds(
            left,
            right,
            min_wheel_velocity_m_s=min_wheel_velocity_m_s,
        )
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

    def _twist_to_wheel_speeds(
        self,
        linear: float,
        angular: float,
    ) -> tuple[float, float]:
        """差速换算左右轮速度，并施加单轮权重补偿机械重心偏移。

        默认权重均为 ``1.0``，等价于标准差速。车辆重心偏离两轮连线的
        中点时，同一指令下左右轮的接地载荷与摩擦不同，直线行驶会朝一侧
        偏转；通过把左右轮速度乘以不同的权重（如偏左时左轮略快），可
        在开环指令上抵消该偏差。
        """

        half_track = self.limits.wheel_track_m / 2.0
        left = (
            linear - angular * half_track
        ) * self.limits.left_wheel_speed_weight
        right = (
            linear + angular * half_track
        ) * self.limits.right_wheel_speed_weight
        return left, right

    def _resolve_minimum_wheel_velocity(
        self,
        minimum_m_s: float | None,
    ) -> float:
        if minimum_m_s is None:
            return self.limits.min_wheel_velocity_m_s
        if (
            isinstance(minimum_m_s, bool)
            or not isinstance(minimum_m_s, (int, float))
            or not math.isfinite(float(minimum_m_s))
            or float(minimum_m_s) < 0.0
        ):
            raise ValueError(
                "min_wheel_velocity_m_s must be finite and non-negative, "
                f"got {minimum_m_s!r}."
            )
        minimum = float(minimum_m_s)
        if minimum > self.limits.max_wheel_velocity_m_s:
            raise ValueError(
                "min_wheel_velocity_m_s must not exceed "
                "max_wheel_velocity_m_s, got "
                f"{minimum!r} > {self.limits.max_wheel_velocity_m_s!r}."
            )
        return minimum

    @staticmethod
    def _apply_minimum_wheel_velocity(
        speed_m_s: float,
        minimum_m_s: float,
    ) -> float:
        """将非零轮速目标抬到减速电机可持续工作的最低速度。"""

        if speed_m_s == 0.0:
            return 0.0
        return math.copysign(
            max(abs(speed_m_s), minimum_m_s),
            speed_m_s,
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

        self._send_soft_brake()

    def emergency_stop(self) -> None:
        """触发固件急停。"""

        self._emergency_stop_latched = True
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
            self._observe_message(message)
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
        self._reset_acceleration_state_at(self._now())

    def _reset_acceleration_state_at(self, current_ns: int) -> None:
        self._target_wheel_speeds_m_s = (0.0, 0.0)
        self._commanded_wheel_speeds_m_s = (0.0, 0.0)
        self._last_sent_wheel_speeds_mm_s = (0, 0)
        self._last_acceleration_update_ns = current_ns
        self._last_wheel_command_ns = current_ns
        self._stall_started_ns = [None, None]

    def _send_soft_brake(self) -> None:
        sequence = self._next_command_sequence()
        self._channel.send_frame(encode_soft_brake_command(sequence))
        self._pending_wheel_commands.clear()
        self._last_wheel_ack_ns = None
        self._synchronization_sequence = (
            sequence if self._synchronization_enforced else None
        )
        if self._synchronization_enforced:
            self._synchronization_acknowledged = False
        self._reset_acceleration_state()

    def _clear_synchronization_attempt(self) -> None:
        """使失败的同步尝试可由上层重新发送新的停车序号。"""

        self._synchronization_sequence = None
        self._synchronization_acknowledged = False

    def _request_synchronization(self) -> None:
        self._synchronization_enforced = True
        if self._synchronization_sequence is None:
            self._send_soft_brake()

    def _wheel_ack_health_failed(self, current_ns: int) -> bool:
        if len(self._pending_wheel_commands) >= _MAX_PENDING_WHEEL_COMMANDS:
            return True
        if not self._pending_wheel_commands:
            return False
        oldest_pending_ns = min(self._pending_wheel_commands.values())
        reference_ns = oldest_pending_ns
        if self._last_wheel_ack_ns is not None:
            reference_ns = max(reference_ns, self._last_wheel_ack_ns)
        return current_ns - reference_ns >= _WHEEL_REPLY_TIMEOUT_NS

    def _observe_message(self, message: ParsedCarMessage) -> None:
        if isinstance(message, CarCommandReply):
            if message.command_type is MessageType.SET_WHEEL_SPEED:
                self._pending_wheel_commands.pop(message.command_sequence, None)
                if message.result is CommandResult.ACCEPTED:
                    self._last_wheel_ack_ns = message.received_timestamp_ns
                elif message.result is CommandResult.SEQUENCE_OLD:
                    self._request_synchronization()
                elif message.result is CommandResult.EMERGENCY_STOP_LATCHED:
                    self._emergency_stop_latched = True
                    self._reset_acceleration_state()
                else:
                    self._request_synchronization()
            if (
                message.command_type is MessageType.SOFT_BRAKE
                and self._synchronization_sequence == message.command_sequence
            ):
                if message.result is CommandResult.ACCEPTED:
                    self._synchronization_sequence = None
                    self._synchronization_acknowledged = True
                elif message.result is CommandResult.EMERGENCY_STOP_LATCHED:
                    self._emergency_stop_latched = True
                    self._synchronization_sequence = None
                    self._synchronization_acknowledged = True
                else:
                    self._clear_synchronization_attempt()
                    raise MotionSynchronizationError(
                        "STM32 rejected SOFT_BRAKE synchronization with "
                        f"{message.result.name.lower()}."
                    )
        elif isinstance(message, OdometryImu):
            self._observe_encoder_stall(message)
        elif isinstance(message, CarSystemStatus):
            # rx_degraded is a sticky STM32-side historical diagnostic.  It is
            # intentionally observable through CarSystemStatus, but it is not
            # a host-side motion gate: a transient bad RX frame must not make
            # later valid ACKs and wheel commands unusable for this boot.
            protocol_healthy = (
                message.protocol_ready
                and not message.reply_queue_full
                and not message.tx_degraded
            )
            self._latest_protocol_healthy = protocol_healthy
            if message.emergency_stop_latched:
                self._emergency_stop_latched = True
                self._reset_acceleration_state()
            if (
                message.reply_queue_full
                or message.tx_degraded
            ):
                self._link_degraded = True
            if (
                not message.protocol_ready
                or message.reply_queue_full
                or message.tx_degraded
            ):
                self._request_synchronization()
            elif self.motion_synchronized:
                self._link_degraded = False

    def _observe_encoder_stall(self, message: OdometryImu) -> None:
        """检查有效编码器是否在持续轮速命令下保持不动。"""

        current_counts = (
            message.left_encoder_count,
            message.right_encoder_count,
        )
        previous_counts = self._last_encoder_counts
        self._last_encoder_counts = current_counts
        if not self.limits.stall_guard_enabled or previous_counts is None:
            self._stall_started_ns = [None, None]
            return

        encoder_valid = (
            bool(message.sensor_flags & SensorFlags.LEFT_ENCODER_VALID),
            bool(message.sensor_flags & SensorFlags.RIGHT_ENCODER_VALID),
        )
        timeout_ns = self.limits.stall_guard_timeout_ms * 1_000_000
        max_encoder_delta = (
            self.limits.stall_guard_stationary_encoder_delta_count
        )
        for index, (is_valid, command_speed) in enumerate(
            zip(encoder_valid, self._commanded_wheel_speeds_m_s)
        ):
            if (
                not is_valid
                or abs(command_speed)
                < self.limits.stall_guard_min_command_speed_m_s
            ):
                self._stall_started_ns[index] = None
                continue
            encoder_delta = abs(current_counts[index] - previous_counts[index])
            if encoder_delta > max_encoder_delta:
                self._stall_started_ns[index] = None
                continue
            started_ns = self._stall_started_ns[index]
            if started_ns is None:
                self._stall_started_ns[index] = message.received_timestamp_ns
                continue
            if message.received_timestamp_ns - started_ns < timeout_ns:
                continue
            side = "left" if index == 0 else "right"
            self.soft_brake()
            raise MotionStallError(
                f"{side} wheel encoder did not change by more than "
                f"{max_encoder_delta} count for at least "
                f"{self.limits.stall_guard_timeout_ms} ms while commanded "
                f"speed was {command_speed:.3f} m/s."
            )

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
