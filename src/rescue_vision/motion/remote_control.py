"""远程调试运动指令到小车运动函数的安全适配。"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from rescue_vision.communication import (
    DebugMotionCommand,
    MotionControlMode,
    ReceivedRemoteMessage,
    RemoteStream,
    RemoteTopic,
)
from rescue_vision.motion.controller import MotionController
from rescue_vision.motion.protocol import ParsedCarMessage


class RemoteControlReceiver(Protocol):
    def receive_control(
        self,
        timeout: float | None = None,
    ) -> ReceivedRemoteMessage: ...


class RemoteMotionError(RuntimeError):
    """远程运动消息无法安全执行。"""


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
                )
            if not command.deadman_enabled:
                self.stop()
                return ExecutedRemoteMotion(
                    command.command_id,
                    RemoteMotionResult.STOPPED_DEADMAN,
                    message.received_timestamp_ns,
                    deadline_ns,
                )
            if command.control_mode is not MotionControlMode.TWIST:
                raise RemoteMotionError(
                    "target_heading is unavailable until localization or IMU "
                    "provides its declared heading reference."
                )
            self.controller.drive(
                command.linear_velocity_m_s,
                command.angular_velocity_rad_s,
            )
        except BaseException as exc:
            try:
                self.stop()
            except BaseException as stop_error:
                exc.add_note(f"Remote motion safety stop also failed: {stop_error!r}")
            if not isinstance(exc, Exception):
                raise
            if isinstance(exc, RemoteMotionError):
                raise
            raise RemoteMotionError("Invalid remote motion command.") from exc

        self._active_deadline_ns = deadline_ns
        return ExecutedRemoteMotion(
            command.command_id,
            RemoteMotionResult.APPLIED,
            message.received_timestamp_ns,
            deadline_ns,
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
) -> None:
    """循环执行远程运动，并为应用装配层提供有界的同线程钩子。"""

    if not 0.0 < poll_interval_s <= 0.1:
        raise ValueError("poll_interval_s must be in (0, 0.1].")
    try:
        while not stop_requested():
            if executor.check_timeout() and on_motion_timeout is not None:
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
            exc.add_note(f"Remote motion shutdown stop also failed: {stop_error!r}")
        raise
    else:
        executor.stop()
