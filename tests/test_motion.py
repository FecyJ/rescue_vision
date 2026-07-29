from __future__ import annotations

from collections.abc import Iterator

import pytest

from rescue_vision.communication import (
    DebugMotionCommand,
    HeadingReference,
    MotionControlMode,
    ReceivedRemoteMessage,
    ReceivedUartLine,
    RemoteStream,
    RemoteTopic,
)
from rescue_vision.motion import (
    CarCommandReply,
    CarSafetyStatus,
    CarStopReason,
    CarTelemetry,
    MotionController,
    MotionLimits,
    RemoteMotionError,
    RemoteMotionExecutor,
    RemoteMotionResult,
    UnknownCarMessage,
    parse_car_line,
    run_remote_motion,
)


class FakeCarChannel:
    def __init__(self, received: list[ReceivedUartLine] | None = None) -> None:
        self.sent: list[bytes] = []
        self.received = list(received or [])

    def send_line(self, payload: bytes) -> None:
        self.sent.append(payload)

    def receive_line(self, timeout: float | None = None) -> ReceivedUartLine:
        del timeout
        if not self.received:
            raise TimeoutError
        return self.received.pop(0)


class FakeRemoteReceiver:
    def __init__(self, messages: Iterator[ReceivedRemoteMessage]) -> None:
        self._messages = messages

    def receive_control(
        self,
        timeout: float | None = None,
    ) -> ReceivedRemoteMessage:
        del timeout
        try:
            return next(self._messages)
        except StopIteration as exc:
            raise TimeoutError from exc


class FakeClock:
    def __init__(self, timestamp_ns: int = 0) -> None:
        self.timestamp_ns = timestamp_ns

    def __call__(self) -> int:
        return self.timestamp_ns

    def advance(self, seconds: float) -> None:
        self.timestamp_ns += round(seconds * 1_000_000_000)


def limits() -> MotionLimits:
    return MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.30,
        max_angular_velocity_rad_s=2.0,
        max_wheel_velocity_m_s=0.40,
        max_wheel_acceleration_m_s2=0.50,
        max_remote_command_valid_for_ms=500,
    )


def remote_message(
    command: DebugMotionCommand,
    *,
    received_timestamp_ns: int = 1_000_000_000,
    topic: str = RemoteTopic.DEBUG_MOTION.value,
) -> ReceivedRemoteMessage:
    return ReceivedRemoteMessage(
        stream=RemoteStream.CONTROL,
        topic=topic,
        content_type="application/json",
        sequence=0,
        sender_timestamp_ns=123,
        received_timestamp_ns=received_timestamp_ns,
        attributes={},
        payload=command.to_payload(),
    )


def twist_command(
    *,
    deadman_enabled: bool = True,
    linear_velocity_m_s: float = 0.2,
    angular_velocity_rad_s: float = 1.0,
    valid_for_ms: int = 200,
) -> DebugMotionCommand:
    return DebugMotionCommand(
        command_id="drive-1",
        issued_timestamp_ns=123,
        valid_for_ms=valid_for_ms,
        deadman_enabled=deadman_enabled,
        control_mode=MotionControlMode.TWIST,
        linear_velocity_m_s=linear_velocity_m_s,
        angular_velocity_rad_s=angular_velocity_rad_s,
    )


def test_motion_functions_encode_differential_drive_and_stops() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    controller.drive(0.2, 1.0)
    clock.advance(1.0)
    assert controller.update()
    controller.forward(0.1)
    clock.advance(1.0)
    assert controller.update()
    controller.backward(0.1)
    clock.advance(1.0)
    assert controller.update()
    controller.turn_left(1.0)
    clock.advance(1.0)
    assert controller.update()
    controller.turn_right(1.0)
    clock.advance(1.0)
    assert controller.update()
    controller.soft_brake()
    controller.emergency_stop()
    controller.query_state()

    assert channel.sent == [
        b"m0.1,0.3",
        b"m0.1,0.1",
        b"m-0.1,-0.1",
        b"m-0.1,0.1",
        b"m0.1,-0.1",
        b"b0,0",
        b"e",
        b"v",
    ]


def test_wheel_targets_are_slew_limited_across_acceleration_and_reversal() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    clock.advance(10.0)
    controller.set_wheel_speeds(0.30, -0.30)
    assert controller.target_wheel_speeds_m_s == pytest.approx((0.30, -0.30))
    assert not controller.update()

    clock.advance(0.1)
    assert controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.05, -0.05))

    controller.set_wheel_speeds(-0.30, 0.30)
    clock.advance(0.1)
    assert controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.0, 0.0))

    clock.advance(0.1)
    assert controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((-0.05, 0.05))
    assert channel.sent == [b"m0.05,-0.05", b"m0,0", b"m-0.05,0.05"]


def test_soft_brake_clears_pending_acceleration_target() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.forward(0.30)
    clock.advance(0.1)
    controller.update()

    controller.soft_brake()
    clock.advance(1.0)

    assert not controller.update()
    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
    assert controller.commanded_wheel_speeds_m_s == (0.0, 0.0)
    assert channel.sent == [b"m0.05,0.05", b"b0,0"]


def test_motion_limits_reject_instead_of_clamping() -> None:
    channel = FakeCarChannel()
    controller = MotionController(channel, limits())

    with pytest.raises(ValueError, match="linear_velocity"):
        controller.drive(0.31, 0.0)
    with pytest.raises(ValueError, match="angular_velocity"):
        controller.drive(0.0, 2.1)
    with pytest.raises(ValueError, match="Wheel velocity"):
        controller.drive(0.30, 2.0)
    with pytest.raises(ValueError, match="must be >= 0"):
        controller.forward(-0.1)

    assert channel.sent == []


def test_parse_car_replies_telemetry_and_unknown_prefix() -> None:
    telemetry = parse_car_line(
        ReceivedUartLine(
            3,
            5_000,
            b"t12345,0.19,0.20,0.20,0.20,90,45",
        )
    )
    ok = parse_car_line(ReceivedUartLine(4, 6_000, b"OK m=0.20,0.20"))
    error = parse_car_line(ReceivedUartLine(5, 7_000, b"ERR: unknown cmd 'z'"))
    unknown = parse_car_line(ReceivedUartLine(6, 8_000, b"imu,1,2,3"))
    binary = parse_car_line(ReceivedUartLine(7, 9_000, b"\xff\x00\x80"))

    assert isinstance(telemetry, CarTelemetry)
    assert telemetry.controller_timestamp_ms == 12_345
    assert telemetry.actual_left_m_s == pytest.approx(0.19)
    assert telemetry.servo_right_deg == pytest.approx(45.0)
    assert ok == CarCommandReply(4, 6_000, True, "m=0.20,0.20")
    assert error == CarCommandReply(5, 7_000, False, "unknown cmd 'z'")
    assert unknown == UnknownCarMessage(6, 8_000, b"imu,1,2,3")
    assert binary == UnknownCarMessage(7, 9_000, b"\xff\x00\x80")


def test_controller_ignores_empty_uart_lines_before_valid_message() -> None:
    channel = FakeCarChannel(
        [
            ReceivedUartLine(0, 1_000, b""),
            ReceivedUartLine(1, 2_000, b""),
            ReceivedUartLine(2, 3_000, b"OK"),
        ]
    )
    controller = MotionController(channel, limits())

    assert controller.receive_message(timeout=0) == CarCommandReply(
        uart_sequence=2,
        received_timestamp_ns=3_000,
        succeeded=True,
        detail="",
    )


def test_parse_versioned_car_safety_status() -> None:
    status = parse_car_line(
        ReceivedUartLine(
            7,
            9_000,
            b"s1,12350,300,1,0,25,running",
        )
    )
    startup = parse_car_line(
        ReceivedUartLine(
            8,
            10_000,
            b"s1,10,300,1,0,-1,startup",
        )
    )

    assert status == CarSafetyStatus(
        uart_sequence=7,
        received_timestamp_ns=9_000,
        controller_timestamp_ms=12_350,
        watchdog_timeout_ms=300,
        watchdog_armed=True,
        emergency_stop_latched=False,
        last_motion_command_age_ms=25,
        stop_reason=CarStopReason.RUNNING,
    )
    assert isinstance(startup, CarSafetyStatus)
    assert startup.last_motion_command_age_ms is None


@pytest.mark.parametrize(
    "payload",
    [
        b"t1,0,0",
        b"t-1,0,0,0,0,90,90",
        b"t1,nan,0,0,0,90,90",
        b"t1,0,0,0,0,181,90",
        b"s1,1,300,2,0,10,running",
        b"s1,1,300,1,0,-2,running",
        b"s1,1,0,1,0,10,running",
        b"s1,1,300,1,0,10,future_reason",
        b"s1,1,300,1,0,10",
    ],
)
def test_parse_car_line_rejects_malformed_known_messages(payload: bytes) -> None:
    with pytest.raises(ValueError):
        parse_car_line(ReceivedUartLine(0, 0, payload))


def test_remote_twist_executes_and_expiry_uses_receive_clock() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    executor = RemoteMotionExecutor(
        controller,
        monotonic_ns=lambda: 1_050_000_000,
    )

    result = executor.execute(remote_message(twist_command()))

    assert result.result is RemoteMotionResult.APPLIED
    assert result.deadline_timestamp_ns == 1_200_000_000
    assert executor.active_deadline_ns == 1_200_000_000
    assert channel.sent == []
    clock.advance(0.2)
    assert controller.update()
    assert channel.sent == [b"m0.1,0.1"]
    assert not executor.check_timeout(now_ns=1_199_999_999)
    assert executor.next_wait_s(
        0.05,
        now_ns=1_190_000_000,
    ) == pytest.approx(0.01)
    assert executor.check_timeout(now_ns=1_200_000_000)
    assert channel.sent == [b"m0.1,0.1", b"b0,0"]


def test_remote_zero_twist_slew_limits_to_zero_and_clears_deadline() -> None:
    channel = FakeCarChannel()
    clock = FakeClock(1_050_000_000)
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    executor = RemoteMotionExecutor(
        controller,
        monotonic_ns=clock,
    )
    executor.execute(
        remote_message(
            twist_command(
                linear_velocity_m_s=0.2,
                angular_velocity_rad_s=0.0,
                valid_for_ms=500,
            )
        )
    )
    clock.advance(0.4)
    assert controller.update()

    stopped = executor.execute(
        remote_message(
            twist_command(
                linear_velocity_m_s=0.0,
                angular_velocity_rad_s=0.0,
                valid_for_ms=500,
            ),
            received_timestamp_ns=1_440_000_000,
        ),
    )

    assert stopped.result is RemoteMotionResult.APPLIED
    assert stopped.linear_velocity_m_s == 0.0
    assert stopped.angular_velocity_rad_s == 0.0
    assert executor.active_deadline_ns is None
    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.2, 0.2))
    assert channel.sent == [b"m0.2,0.2"]

    for _ in range(4):
        clock.advance(0.1)
        assert controller.update()

    assert controller.commanded_wheel_speeds_m_s == (0.0, 0.0)
    assert channel.sent == [
        b"m0.2,0.2",
        b"m0.15,0.15",
        b"m0.1,0.1",
        b"m0.05,0.05",
        b"m0,0",
    ]


def test_remote_deadman_off_and_already_expired_commands_stop() -> None:
    channel = FakeCarChannel()
    executor = RemoteMotionExecutor(MotionController(channel, limits()))
    stopped = executor.execute(
        remote_message(
            twist_command(
                deadman_enabled=False,
                linear_velocity_m_s=0.0,
                angular_velocity_rad_s=0.0,
            )
        ),
        now_ns=1_050_000_000,
    )
    expired = executor.execute(
        remote_message(twist_command()),
        now_ns=1_200_000_000,
    )

    assert stopped.result is RemoteMotionResult.STOPPED_DEADMAN
    assert expired.result is RemoteMotionResult.EXPIRED
    assert channel.sent == [b"b0,0", b"b0,0"]


def test_invalid_remote_commands_stop_before_reporting_error() -> None:
    channel = FakeCarChannel()
    executor = RemoteMotionExecutor(MotionController(channel, limits()))
    target_heading = DebugMotionCommand(
        command_id="heading",
        issued_timestamp_ns=0,
        valid_for_ms=200,
        deadman_enabled=True,
        control_mode=MotionControlMode.TARGET_HEADING,
        linear_velocity_m_s=0.0,
        angular_velocity_rad_s=0.0,
        target_heading_rad=1.0,
        heading_reference=HeadingReference.SESSION_START,
    )

    with pytest.raises(RemoteMotionError, match="target_heading"):
        executor.execute(
            remote_message(target_heading),
            now_ns=1_050_000_000,
        )
    with pytest.raises(RemoteMotionError, match="Invalid"):
        executor.execute(
            remote_message(twist_command(linear_velocity_m_s=0.31)),
            now_ns=1_050_000_000,
        )
    with pytest.raises(RemoteMotionError, match="topic"):
        executor.execute(
            remote_message(twist_command(), topic="control/debug/capture"),
            now_ns=1_050_000_000,
        )

    assert channel.sent == [b"b0,0", b"b0,0", b"b0,0"]


def test_remote_validity_limit_is_enforced_without_using_sender_clock() -> None:
    channel = FakeCarChannel()
    executor = RemoteMotionExecutor(MotionController(channel, limits()))

    with pytest.raises(RemoteMotionError, match="valid_for_ms"):
        executor.execute(
            remote_message(twist_command(valid_for_ms=501)),
            now_ns=1_000_000_001,
        )

    assert channel.sent == [b"b0,0"]


def test_remote_loop_drains_uart_and_stops_on_exit() -> None:
    channel = FakeCarChannel(
        [
            ReceivedUartLine(0, 8, b""),
            ReceivedUartLine(1, 9, b"\xff\x00\x80"),
            ReceivedUartLine(
                2,
                10,
                b"t2,0.1,OK m=-0.05,0.1,0.1,90,90",
            ),
            ReceivedUartLine(
                3,
                11,
                b"t1,0.1,0.1,0.1,0.1,90,90",
            )
        ]
    )
    clock = FakeClock()
    executor = RemoteMotionExecutor(
        MotionController(channel, limits(), monotonic_ns=clock),
        monotonic_ns=lambda: 1_000_000_001,
    )

    polls = 0
    car_messages: list[object] = []

    def stop_requested() -> bool:
        nonlocal polls
        polls += 1
        clock.advance(0.1)
        return polls > 2

    run_remote_motion(
        FakeRemoteReceiver(iter([remote_message(twist_command())])),
        executor,
        stop_requested=stop_requested,
        on_car_message=car_messages.append,
        poll_interval_s=0.01,
    )

    assert car_messages == [
        UnknownCarMessage(1, 9, b"\xff\x00\x80"),
        UnknownCarMessage(
            2,
            10,
            b"t2,0.1,OK m=-0.05,0.1,0.1,90,90",
        ),
        CarTelemetry(
            uart_sequence=3,
            received_timestamp_ns=11,
            controller_timestamp_ms=1,
            actual_left_m_s=0.1,
            actual_right_m_s=0.1,
            target_left_m_s=0.1,
            target_right_m_s=0.1,
            servo_left_deg=90,
            servo_right_deg=90,
        ),
    ]
    assert channel.sent == [b"m0.05,0.05", b"b0,0"]


def test_remote_loop_routes_other_controls_without_bypassing_stop() -> None:
    channel = FakeCarChannel()
    executor = RemoteMotionExecutor(MotionController(channel, limits()))
    capture_message = remote_message(
        twist_command(),
        topic=RemoteTopic.DEBUG_CAPTURE.value,
    )
    routed: list[ReceivedRemoteMessage] = []
    polls = 0

    def stop_requested() -> bool:
        nonlocal polls
        polls += 1
        return polls > 1

    run_remote_motion(
        FakeRemoteReceiver(iter([capture_message])),
        executor,
        stop_requested=stop_requested,
        on_other_control=routed.append,
    )

    assert routed == [capture_message]
    assert channel.sent == [b"b0,0"]


def test_remote_loop_uart_receive_fault_stops_before_propagating() -> None:
    class FaultingCarChannel(FakeCarChannel):
        def receive_line(self, timeout: float | None = None) -> ReceivedUartLine:
            del timeout
            raise RuntimeError("UART receive failed")

    channel = FaultingCarChannel()
    executor = RemoteMotionExecutor(MotionController(channel, limits()))

    with pytest.raises(RuntimeError, match="UART receive failed"):
        run_remote_motion(
            FakeRemoteReceiver(iter(())),
            executor,
            stop_requested=lambda: False,
        )

    assert channel.sent == [b"b0,0"]
