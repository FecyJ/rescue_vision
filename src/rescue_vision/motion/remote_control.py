"""远程调试运动与夹爪指令到 Rescue Car 控制函数的适配。"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from rescue_vision.communication import (
    DebugGripperCommand,
    DebugMotionCommand,
    MotionControlMode,
    ReceivedRemoteMessage,
    RemoteStream,
    RemoteTopic,
    UartError,
)
from rescue_vision.exception_notes import add_exception_note
from rescue_vision.motion.controller import MotionController
from rescue_vision.motion.protocol import ParsedCarMessage


def _finite_float(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be finite, got {value!r}.")
    return float(value)


def _move_toward(current: float, target: float, maximum_delta: float) -> float:
    distance = abs(target - current)
    if distance <= maximum_delta or math.isclose(
        distance,
        maximum_delta,
        rel_tol=1e-12,
        abs_tol=1e-15,
    ):
        return target
    return (
        current + maximum_delta
        if target > current
        else current - maximum_delta
    )


class RemoteControlReceiver(Protocol):
    def receive_control(
        self,
        timeout: float | None = None,
    ) -> ReceivedRemoteMessage: ...


class RemoteMotionError(RuntimeError):
    """远程运动消息无法安全执行。"""


class RemoteGripperError(RuntimeError):
    """远程夹爪消息无法执行。"""


class RemoteGripperResult(str, Enum):
    APPLIED = "applied"
    STOPPED = "stopped"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class GripperCalibration:
    """远程连续控制所需的双舵机机械端点和速度标定。"""

    open_left_angle_deg: float
    open_right_angle_deg: float
    closed_left_angle_deg: float
    closed_right_angle_deg: float
    full_travel_time_s: float
    angle_sum_deg: float

    def __post_init__(self) -> None:
        for name in (
            "open_left_angle_deg",
            "open_right_angle_deg",
            "closed_left_angle_deg",
            "closed_right_angle_deg",
        ):
            value = _finite_float(getattr(self, name), name)
            if not 0.0 <= value <= 180.0:
                raise ValueError(f"{name} must be in [0, 180], got {value!r}.")
            object.__setattr__(self, name, value)
        travel_time = _finite_float(
            self.full_travel_time_s,
            "full_travel_time_s",
        )
        if travel_time <= 0.0:
            raise ValueError(
                "full_travel_time_s must be > 0, "
                f"got {travel_time!r}."
            )
        object.__setattr__(self, "full_travel_time_s", travel_time)
        angle_sum = _finite_float(self.angle_sum_deg, "angle_sum_deg")
        if not 0.0 < angle_sum <= 360.0:
            raise ValueError(
                "angle_sum_deg must be in (0, 360], "
                f"got {angle_sum!r}."
            )
        object.__setattr__(self, "angle_sum_deg", angle_sum)
        if self.open_left_angle_deg == self.closed_left_angle_deg:
            raise ValueError("Left gripper open and closed angles must differ.")
        if self.open_right_angle_deg == self.closed_right_angle_deg:
            raise ValueError("Right gripper open and closed angles must differ.")
        for state, left, right in (
            (
                "open",
                self.open_left_angle_deg,
                self.open_right_angle_deg,
            ),
            (
                "closed",
                self.closed_left_angle_deg,
                self.closed_right_angle_deg,
            ),
        ):
            angle_sum = left + right
            if not math.isclose(
                angle_sum,
                self.angle_sum_deg,
                rel_tol=1e-12,
                abs_tol=1e-9,
            ):
                raise ValueError(
                    "Gripper "
                    f"{state} angles must sum to "
                    f"{self.angle_sum_deg:g} degrees, "
                    f"got {left} + {right} = {angle_sum}."
                )


@dataclass(frozen=True, slots=True)
class ExecutedRemoteGripper:
    command_id: str
    result: RemoteGripperResult
    received_timestamp_ns: int
    deadline_timestamp_ns: int
    open_pressed: bool
    close_pressed: bool


class RemoteMotionResult(str, Enum):
    APPLIED = "applied"
    STOPPED_DEADMAN = "stopped_deadman"
    EXPIRED = "expired"


@dataclass(frozen=True, slots=True)
class ExecutedRemoteMotion:
    command_id: str
    result: RemoteMotionResult
    received_timestamp_ns: int
    deadline_timestamp_ns: int
    deadman_enabled: bool
    linear_velocity_m_s: float
    angular_velocity_rad_s: float


class RemoteGripperExecutor:
    """把持续刷新的扳机状态渐进转换为标定后的双舵机角度。"""

    def __init__(
        self,
        controller: MotionController,
        calibration: GripperCalibration,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        if not isinstance(controller, MotionController):
            raise TypeError("controller must be MotionController.")
        if not isinstance(calibration, GripperCalibration):
            raise TypeError("calibration must be GripperCalibration.")
        if not callable(monotonic_ns):
            raise TypeError("monotonic_ns must be callable.")
        self.controller = controller
        self.calibration = calibration
        self._monotonic_ns = monotonic_ns
        self._active_deadline_ns: int | None = None
        self._active_command_id: str | None = None
        self._open_pressed = False
        self._close_pressed = False
        self._last_update_ns: int | None = None

    @property
    def active_deadline_ns(self) -> int | None:
        return self._active_deadline_ns

    @property
    def active_command_id(self) -> str | None:
        return self._active_command_id

    def execute(
        self,
        message: ReceivedRemoteMessage,
        *,
        now_ns: int | None = None,
    ) -> ExecutedRemoteGripper:
        """接受扳机状态；释放或两个方向同时按下时停止继续改变角度。"""

        current_ns = self._now(now_ns)
        try:
            self._validate_envelope(message)
            command = DebugGripperCommand.from_payload(message.payload)
            deadline_ns = message.received_timestamp_ns + (
                command.valid_for_ms * 1_000_000
            )
            if current_ns < message.received_timestamp_ns:
                raise RemoteGripperError(
                    "now_ns precedes the Raspberry Pi receive timestamp: "
                    f"{current_ns} < {message.received_timestamp_ns}."
                )
            if (
                command.valid_for_ms
                > self.controller.limits.max_remote_command_valid_for_ms
            ):
                raise RemoteGripperError(
                    "Remote command valid_for_ms exceeds configured limit "
                    f"{self.controller.limits.max_remote_command_valid_for_ms}: "
                    f"{command.valid_for_ms}."
                )
            self.update(now_ns=current_ns)
            if current_ns >= deadline_ns:
                return ExecutedRemoteGripper(
                    command.command_id,
                    RemoteGripperResult.EXPIRED,
                    message.received_timestamp_ns,
                    deadline_ns,
                    command.open_pressed,
                    command.close_pressed,
                )
            self._last_update_ns = current_ns
            if command.open_pressed == command.close_pressed:
                self._clear_active()
                result = RemoteGripperResult.STOPPED
            else:
                self._active_deadline_ns = deadline_ns
                self._active_command_id = command.command_id
                self._open_pressed = command.open_pressed
                self._close_pressed = command.close_pressed
                result = RemoteGripperResult.APPLIED
        except Exception as exc:
            self.stop(now_ns=current_ns)
            if isinstance(exc, (RemoteGripperError, UartError)):
                raise
            raise RemoteGripperError("Invalid remote gripper command.") from exc

        return ExecutedRemoteGripper(
            command.command_id,
            result,
            message.received_timestamp_ns,
            deadline_ns,
            command.open_pressed,
            command.close_pressed,
        )

    def update(self, *, now_ns: int | None = None) -> bool:
        """按按压方向和固定全行程时间推进舵机目标。"""

        current_ns = self._now(now_ns)
        if (
            self._last_update_ns is not None
            and current_ns < self._last_update_ns
        ):
            raise ValueError(
                "now_ns must not precede the previous gripper update: "
                f"{current_ns} < {self._last_update_ns}."
            )
        elapsed_s = (
            0.0
            if self._last_update_ns is None
            else (current_ns - self._last_update_ns) / 1_000_000_000.0
        )
        self._last_update_ns = current_ns
        if self._active_deadline_ns is None:
            return False
        if current_ns >= self._active_deadline_ns:
            self._clear_active()
            return False
        current_angles = self.controller.gripper_target_angles_deg
        if current_angles is None:
            return False
        if self._close_pressed:
            target_left_angle = self.calibration.closed_left_angle_deg
        else:
            target_left_angle = self.calibration.open_left_angle_deg
        left_rate = (
            abs(
                self.calibration.closed_left_angle_deg
                - self.calibration.open_left_angle_deg
            )
            / self.calibration.full_travel_time_s
        )
        # 把可能来自旧命令且不满足角度和约束的目标投影到有效线段，
        # 再只推进一个自由度；所有远程 UART 下发都重新构造右角。
        minimum_left_angle = max(0.0, self.calibration.angle_sum_deg - 180.0)
        maximum_left_angle = min(180.0, self.calibration.angle_sum_deg)
        projected_left_angle = (
            current_angles[0]
            + (self.calibration.angle_sum_deg - current_angles[1])
        ) / 2.0
        current_left_angle = min(
            maximum_left_angle,
            max(minimum_left_angle, projected_left_angle),
        )
        next_left_angle = _move_toward(
            current_left_angle,
            target_left_angle,
            left_rate * elapsed_s,
        )
        next_angles = (
            next_left_angle,
            self.calibration.angle_sum_deg - next_left_angle,
        )
        if next_angles == current_angles:
            return False
        self.controller.set_gripper_angles(*next_angles)
        return True

    def check_timeout(self, *, now_ns: int | None = None) -> str | None:
        """到期时停止推进，并返回刚超时的命令 ID。"""

        if self._active_deadline_ns is None:
            return None
        current_ns = self._now(now_ns)
        if current_ns < self._active_deadline_ns:
            return None
        command_id = self._active_command_id
        assert command_id is not None
        self._last_update_ns = current_ns
        self._clear_active()
        return command_id

    def stop(self, *, now_ns: int | None = None) -> None:
        """清除持续控制状态；保留当前舵机目标角度。"""

        self._last_update_ns = self._now(now_ns)
        self._clear_active()

    def _clear_active(self) -> None:
        self._active_deadline_ns = None
        self._active_command_id = None
        self._open_pressed = False
        self._close_pressed = False

    def _now(self, value: int | None) -> int:
        current = self._monotonic_ns() if value is None else value
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            raise ValueError(
                f"now_ns must be a non-negative integer, got {current!r}."
            )
        return current

    @staticmethod
    def _validate_envelope(message: ReceivedRemoteMessage) -> None:
        if not isinstance(message, ReceivedRemoteMessage):
            raise TypeError("message must be ReceivedRemoteMessage.")
        if message.stream is not RemoteStream.CONTROL:
            raise RemoteGripperError(
                "Remote gripper message must use control stream."
            )
        if message.topic != RemoteTopic.DEBUG_GRIPPER.value:
            raise RemoteGripperError(
                f"Unexpected remote gripper topic {message.topic!r}."
            )
        if message.content_type != "application/json":
            raise RemoteGripperError(
                "Remote gripper content_type must be 'application/json'."
            )
        if message.attributes:
            raise RemoteGripperError(
                "Remote gripper attributes must be empty."
            )


class RemoteMotionExecutor:
    """执行已经通过 TCP 帧校验的调试运动消息。"""

    def __init__(
        self,
        controller: MotionController,
        *,
        monotonic_ns: Callable[[], int] = time.monotonic_ns,
    ) -> None:
        self.controller = controller
        self._monotonic_ns = monotonic_ns
        self._active_deadline_ns: int | None = None

    @property
    def active_deadline_ns(self) -> int | None:
        return self._active_deadline_ns

    def execute(
        self,
        message: ReceivedRemoteMessage,
        *,
        now_ns: int | None = None,
    ) -> ExecutedRemoteMotion:
        """执行运动消息；非法消息先停车再报错。"""

        current_ns = self._now(now_ns)
        try:
            self._validate_envelope(message)
            command = DebugMotionCommand.from_payload(message.payload)
            deadline_ns = message.received_timestamp_ns + (
                command.valid_for_ms * 1_000_000
            )
            if current_ns < message.received_timestamp_ns:
                raise RemoteMotionError(
                    "now_ns precedes the Raspberry Pi receive timestamp: "
                    f"{current_ns} < {message.received_timestamp_ns}."
                )
            if (
                command.valid_for_ms
                > self.controller.limits.max_remote_command_valid_for_ms
            ):
                raise RemoteMotionError(
                    "Remote command valid_for_ms exceeds configured limit "
                    f"{self.controller.limits.max_remote_command_valid_for_ms}: "
                    f"{command.valid_for_ms}."
                )
            if current_ns >= deadline_ns:
                self.stop()
                return ExecutedRemoteMotion(
                    command.command_id,
                    RemoteMotionResult.EXPIRED,
                    message.received_timestamp_ns,
                    deadline_ns,
                    command.deadman_enabled,
                    command.linear_velocity_m_s,
                    command.angular_velocity_rad_s,
                )
            if not command.deadman_enabled:
                self.stop()
                return ExecutedRemoteMotion(
                    command.command_id,
                    RemoteMotionResult.STOPPED_DEADMAN,
                    message.received_timestamp_ns,
                    deadline_ns,
                    command.deadman_enabled,
                    command.linear_velocity_m_s,
                    command.angular_velocity_rad_s,
                )
            if command.control_mode is not MotionControlMode.TWIST:
                raise RemoteMotionError(
                    "target_heading is unavailable until localization or IMU "
                    "provides its declared heading reference."
                )
            zero_twist = (
                command.linear_velocity_m_s == 0.0
                and command.angular_velocity_rad_s == 0.0
            )
            if zero_twist:
                # 死手仍开启时，手柄回中是普通目标变化，必须与其他 twist
                # 共用树莓派单轮加速度限制。真机固件的 b0,0 会让实测轮速
                # 直接归零，只保留给死手关闭、超时和异常等安全停车路径。
                self.controller.drive(0.0, 0.0)
                applied_linear = 0.0
                applied_angular = 0.0
            else:
                # 手柄两个轴分别合法时，差速合成仍可能让外侧轮超限。
                # 同比缩放保留曲率，底层单轮硬上限仍由 controller 校验。
                applied_linear, applied_angular = (
                    self.controller.drive_wheel_limited(
                        command.linear_velocity_m_s,
                        command.angular_velocity_rad_s,
                    )
                )
        except BaseException as exc:
            try:
                self.stop()
            except BaseException as stop_error:
                add_exception_note(
                    exc,
                    f"Remote motion safety stop also failed: {stop_error!r}",
                )
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, UartError):
                raise
            if isinstance(exc, RemoteMotionError):
                raise
            raise RemoteMotionError("Invalid remote motion command.") from exc

        self._active_deadline_ns = None if zero_twist else deadline_ns
        return ExecutedRemoteMotion(
            command.command_id,
            RemoteMotionResult.APPLIED,
            message.received_timestamp_ns,
            deadline_ns,
            command.deadman_enabled,
            applied_linear,
            applied_angular,
        )

    def check_timeout(self, *, now_ns: int | None = None) -> bool:
        """到达当前命令有效期时停车。"""

        if self._active_deadline_ns is None:
            return False
        if self._now(now_ns) < self._active_deadline_ns:
            return False
        self.stop()
        return True

    def next_wait_s(
        self,
        maximum_wait_s: float,
        *,
        now_ns: int | None = None,
    ) -> float:
        """返回不越过当前命令期限的下一次接收等待时间。"""

        if not 0.0 < maximum_wait_s < float("inf"):
            raise ValueError("maximum_wait_s must be finite and > 0.")
        if self._active_deadline_ns is None:
            return float(maximum_wait_s)
        remaining_s = (
            self._active_deadline_ns - self._now(now_ns)
        ) / 1_000_000_000.0
        return max(0.0, min(float(maximum_wait_s), remaining_s))

    def stop(self) -> None:
        """清除远程使能并统一走柔和停车路径。"""

        self._active_deadline_ns = None
        self.controller.soft_brake()

    def _now(self, value: int | None) -> int:
        current = self._monotonic_ns() if value is None else value
        if isinstance(current, bool) or not isinstance(current, int) or current < 0:
            raise ValueError(f"now_ns must be a non-negative integer, got {current!r}.")
        return current

    @staticmethod
    def _validate_envelope(message: ReceivedRemoteMessage) -> None:
        if not isinstance(message, ReceivedRemoteMessage):
            raise TypeError("message must be ReceivedRemoteMessage.")
        if message.stream is not RemoteStream.CONTROL:
            raise RemoteMotionError("Remote motion message must use control stream.")
        if message.topic != RemoteTopic.DEBUG_MOTION.value:
            raise RemoteMotionError(
                f"Unexpected remote motion topic {message.topic!r}."
            )
        if message.content_type != "application/json":
            raise RemoteMotionError(
                "Remote motion content_type must be 'application/json'."
            )
        if message.attributes:
            raise RemoteMotionError("Remote motion attributes must be empty.")


def run_remote_motion(
    receiver: RemoteControlReceiver,
    executor: RemoteMotionExecutor,
    *,
    stop_requested: Callable[[], bool],
    on_car_message: Callable[[ParsedCarMessage], None] | None = None,
    on_motion_executed: Callable[[ExecutedRemoteMotion], None] | None = None,
    on_motion_timeout: Callable[[], None] | None = None,
    on_other_control: Callable[[ReceivedRemoteMessage], None] | None = None,
    on_cycle: Callable[[], None] | None = None,
    poll_interval_s: float = 0.05,
    synchronize_on_start: bool = False,
    synchronization_timeout_s: float = 0.5,
) -> None:
    """循环执行远程运动，并为应用装配层提供有界的同线程钩子。"""

    if not 0.0 < poll_interval_s <= 0.1:
        raise ValueError("poll_interval_s must be in (0, 0.1].")
    if (
        isinstance(synchronization_timeout_s, bool)
        or not isinstance(synchronization_timeout_s, (int, float))
        or not 0.0 < float(synchronization_timeout_s) < float("inf")
    ):
        raise ValueError(
            "synchronization_timeout_s must be finite and > 0."
        )
    try:
        if synchronize_on_start:
            executor.controller.synchronize(
                timeout_s=float(synchronization_timeout_s),
                on_message=on_car_message,
            )
        while not stop_requested():
            if executor.controller.needs_synchronization:
                executor.controller.synchronize(
                    timeout_s=float(synchronization_timeout_s),
                    on_message=on_car_message,
                )
            timed_out = executor.check_timeout()
            if not timed_out:
                executor.controller.update()
            if timed_out and on_motion_timeout is not None:
                on_motion_timeout()
            for car_message in executor.controller.drain_messages():
                if on_car_message is not None:
                    on_car_message(car_message)
            if on_cycle is not None:
                on_cycle()
            try:
                message = receiver.receive_control(
                    timeout=executor.next_wait_s(poll_interval_s)
                )
            except TimeoutError:
                continue
            if message.topic == RemoteTopic.DEBUG_MOTION.value:
                outcome = executor.execute(message)
                if on_motion_executed is not None:
                    on_motion_executed(outcome)
            elif on_other_control is not None:
                on_other_control(message)
            else:
                raise RemoteMotionError(
                    f"Unexpected remote control topic {message.topic!r}."
                )
    except BaseException as exc:
        try:
            executor.stop()
        except BaseException as stop_error:
            add_exception_note(
                exc,
                f"Remote motion shutdown stop also failed: {stop_error!r}",
            )
        raise
    else:
        executor.stop()
