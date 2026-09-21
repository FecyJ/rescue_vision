from __future__ import annotations

from collections.abc import Iterator
import random
import struct

import pytest

from rescue_vision.communication import (
    DebugGripperCommand,
    DebugMotionCommand,
    HeadingReference,
    MotionControlMode,
    ReceivedRemoteMessage,
    ReceivedUartFrame,
    RemoteStream,
    RemoteTopic,
)
from rescue_vision.motion import (
    CarCommandReply,
    CarStopReason,
    CarSystemStatus,
    CommandResult,
    GripperCalibration,
    MotionAccelerationLimits,
    WheelAccelerationOverrides,
    MotionController,
    MotionControlTimingError,
    MotionStallError,
    MotionSynchronizationError,
    MotionLimits,
    RemoteGripperError,
    RemoteGripperExecutor,
    RemoteGripperResult,
    RemoteMotionError,
    RemoteMotionExecutor,
    RemoteMotionResult,
    MessageType,
    OdometryImu,
    SensorFlags,
    SystemFlags,
    crc16_ccitt_false,
    encode_emergency_stop_command,
    encode_gripper_command,
    encode_soft_brake_command,
    encode_state_query_command,
    encode_wheel_speed_command,
    pack_protocol_frame,
    parse_controller_frame,
    run_remote_motion,
)


class FakeCarChannel:
    def __init__(self, received: list[ReceivedUartFrame] | None = None) -> None:
        self.sent: list[bytes] = []
        self.received = list(received or [])

    def send_frame(self, payload: bytes) -> None:
        self.sent.append(payload)

    def receive_frame(self, timeout: float | None = None) -> ReceivedUartFrame:
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
        max_linear_acceleration_m_s2=0.50,
        max_linear_deceleration_m_s2=0.50,
        max_angular_acceleration_rad_s2=5,
        max_angular_deceleration_rad_s2=5,
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


def gripper_message(
    command: DebugGripperCommand,
    *,
    received_timestamp_ns: int = 1_000_000_000,
    topic: str = RemoteTopic.DEBUG_GRIPPER.value,
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


def gripper_command(
    *,
    valid_for_ms: int = 200,
    open_pressed: bool = False,
    close_pressed: bool = True,
) -> DebugGripperCommand:
    return DebugGripperCommand(
        command_id="grip-1",
        issued_timestamp_ns=123,
        valid_for_ms=valid_for_ms,
        open_pressed=open_pressed,
        close_pressed=close_pressed,
    )


def gripper_calibration() -> GripperCalibration:
    return GripperCalibration(
        open_left_angle_deg=20.0,
        open_right_angle_deg=174.0,
        closed_left_angle_deg=80.0,
        closed_right_angle_deg=114.0,
        full_travel_time_s=1.0,
        angle_sum_deg=194.0,
        transport_left_angle_deg=50.0,
        transport_right_angle_deg=144.0,
    )


def system_status_frame(
    uart_sequence: int,
    received_timestamp_ns: int,
    *,
    status_sequence: int = 1,
    controller_timestamp_us: int = 10_000,
    system_flags: SystemFlags = SystemFlags.PROTOCOL_READY,
    stop_reason: CarStopReason = CarStopReason.RUNNING,
    left_cdeg: int = 9000,
    right_cdeg: int = 9000,
) -> ReceivedUartFrame:
    payload = struct.pack(
        "<HQHIHBHH",
        status_sequence,
        controller_timestamp_us,
        300,
        25,
        int(system_flags),
        int(stop_reason),
        left_cdeg,
        right_cdeg,
    )
    return ReceivedUartFrame(
        uart_sequence,
        received_timestamp_ns,
        pack_protocol_frame(MessageType.SYSTEM_STATUS, payload),
    )


def odometry_frame(
    uart_sequence: int,
    received_timestamp_ns: int,
    *,
    telemetry_sequence: int = 0,
    sample_timestamp_us: int = 0,
    left_encoder_count: int = 0,
    right_encoder_count: int = 0,
    sensor_flags: SensorFlags = (
        SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
    ),
) -> ReceivedUartFrame:
    payload = struct.pack(
        "<HQqqiiiiiihH",
        telemetry_sequence,
        sample_timestamp_us,
        left_encoder_count,
        right_encoder_count,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
        int(sensor_flags),
    )
    return ReceivedUartFrame(
        uart_sequence,
        received_timestamp_ns,
        pack_protocol_frame(MessageType.ODOMETRY_IMU, payload),
    )


def command_reply_frame(
    uart_sequence: int,
    received_timestamp_ns: int,
    *,
    command_sequence: int,
    command_type: MessageType = MessageType.SET_WHEEL_SPEED,
    result: CommandResult = CommandResult.ACCEPTED,
) -> ReceivedUartFrame:
    return ReceivedUartFrame(
        uart_sequence,
        received_timestamp_ns,
        pack_protocol_frame(
            MessageType.COMMAND_REPLY,
            struct.pack(
                "<HBB",
                command_sequence,
                int(command_type),
                int(result),
            ),
        ),
    )


def test_gripper_calibration_rejects_unsafe_endpoints_and_travel_time() -> None:
    with pytest.raises(ValueError, match="Left gripper"):
        GripperCalibration(20.0, 174.0, 20.0, 174.0, 1.0, 194.0)
    with pytest.raises(ValueError, match="Right gripper"):
        GripperCalibration(20.0, 174.0, 80.0, 174.0, 1.0, 194.0)
    with pytest.raises(ValueError, match="full_travel_time_s"):
        GripperCalibration(20.0, 174.0, 80.0, 114.0, 0.0, 194.0)
    with pytest.raises(ValueError, match="sum to 194"):
        GripperCalibration(20.0, 160.0, 80.0, 114.0, 1.0, 194.0)
    with pytest.raises(ValueError, match="angle_sum_deg"):
        GripperCalibration(20.0, 174.0, 80.0, 114.0, 1.0, 0.0)


def test_gripper_calibration_exposes_single_object_transport_posture() -> None:
    calibration = gripper_calibration()
    assert calibration.transport_angles_deg == pytest.approx((50.0, 144.0))

    with pytest.raises(ValueError, match="transport angles must sum"):
        GripperCalibration(
            20.0,
            174.0,
            80.0,
            114.0,
            1.0,
            194.0,
            50.0,
            145.0,
        )
    with pytest.raises(ValueError, match="strictly between"):
        GripperCalibration(
            20.0,
            174.0,
            80.0,
            114.0,
            1.0,
            194.0,
            20.0,
            174.0,
        )


def test_motion_functions_encode_differential_drive_and_stops() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    fast_limits = MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.30,
        max_angular_velocity_rad_s=2.0,
        max_wheel_velocity_m_s=0.40,
        max_linear_acceleration_m_s2=10.0,
        max_linear_deceleration_m_s2=10.0,
        max_angular_acceleration_rad_s2=100,
        max_angular_deceleration_rad_s2=100,
        max_remote_command_valid_for_ms=500,
    )
    controller = MotionController(channel, fast_limits, monotonic_ns=clock)

    controller.drive(0.2, 1.0)
    clock.advance(0.05)
    assert controller.update()
    controller.forward(0.1)
    clock.advance(0.05)
    assert controller.update()
    controller.backward(0.1)
    clock.advance(0.05)
    assert controller.update()
    controller.turn_left(1.0)
    clock.advance(0.05)
    assert controller.update()
    controller.turn_right(1.0)
    clock.advance(0.05)
    assert controller.update()
    controller.soft_brake()
    controller.emergency_stop()
    controller.query_state()

    assert channel.sent == [
        encode_wheel_speed_command(0, 0.1, 0.3),
        encode_wheel_speed_command(1, 0.1, 0.1),
        encode_wheel_speed_command(2, -0.1, -0.1),
        encode_wheel_speed_command(3, -0.1, 0.1),
        encode_wheel_speed_command(4, 0.1, -0.1),
        encode_soft_brake_command(5),
        encode_emergency_stop_command(6),
        encode_state_query_command(7),
    ]


def test_nonzero_wheel_targets_are_raised_to_motor_minimum() -> None:
    channel = FakeCarChannel()
    controller = MotionController(channel, limits())

    controller.set_wheel_speeds(0.01, -0.019)

    assert controller.target_wheel_speeds_m_s == pytest.approx((0.02, -0.02))

    controller.set_wheel_speeds(0.0, 0.0)
    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)


def test_per_command_minimum_wheel_velocity_overrides_global_floor() -> None:
    channel = FakeCarChannel()
    controller = MotionController(channel, limits())

    controller.drive_wheel_limited(
        0.0,
        0.05,
        min_wheel_velocity_m_s=0.01,
    )
    assert controller.target_wheel_speeds_m_s == pytest.approx(
        (-0.01, 0.01)
    )

    controller.drive_wheel_limited(0.0, 0.05)
    assert controller.target_wheel_speeds_m_s == pytest.approx(
        (-0.02, 0.02)
    )

    controller.drive_wheel_limited(
        0.0,
        0.05,
        min_wheel_velocity_m_s=0.0,
    )
    assert controller.target_wheel_speeds_m_s == pytest.approx(
        (-0.005, 0.005)
    )


def test_drive_applies_per_wheel_speed_weights() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    weighted_limits = MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.30,
        max_angular_velocity_rad_s=2.0,
        max_wheel_velocity_m_s=0.40,
        max_linear_acceleration_m_s2=10.0,
        max_linear_deceleration_m_s2=10.0,
        max_angular_acceleration_rad_s2=100,
        max_angular_deceleration_rad_s2=100,
        max_remote_command_valid_for_ms=500,
        left_wheel_speed_weight=1.05,
        right_wheel_speed_weight=0.95,
    )
    controller = MotionController(channel, weighted_limits, monotonic_ns=clock)

    controller.forward(0.1)
    clock.advance(0.05)
    assert controller.update()

    assert controller.commanded_wheel_speeds_m_s == pytest.approx(
        (0.105, 0.095)
    )


def test_wheel_limited_drive_scales_weighted_wheels_to_limit() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    weighted_limits = MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.30,
        max_angular_velocity_rad_s=2.0,
        max_wheel_velocity_m_s=0.30,
        max_linear_acceleration_m_s2=10.0,
        max_linear_deceleration_m_s2=10.0,
        max_angular_acceleration_rad_s2=100,
        max_angular_deceleration_rad_s2=100,
        max_remote_command_valid_for_ms=500,
        left_wheel_speed_weight=1.2,
        right_wheel_speed_weight=1.0,
    )
    controller = MotionController(channel, weighted_limits, monotonic_ns=clock)

    applied_linear, applied_angular = controller.drive_wheel_limited(0.30, 0.0)

    # left = 0.30 * 1.2 = 0.36 超过单轮上限，需按比例缩到 0.30。
    assert applied_linear == pytest.approx(0.30 / 1.2)
    assert applied_angular == pytest.approx(0.0)


def test_motion_limits_reject_non_positive_wheel_weight() -> None:
    with pytest.raises(ValueError, match="left_wheel_speed_weight"):
        MotionLimits(
            wheel_track_m=0.20,
            max_linear_velocity_m_s=0.30,
            max_angular_velocity_rad_s=2.0,
            max_wheel_velocity_m_s=0.40,
            max_linear_acceleration_m_s2=0.50,
            max_linear_deceleration_m_s2=0.50,
            max_angular_acceleration_rad_s2=5,
            max_angular_deceleration_rad_s2=5,
            max_remote_command_valid_for_ms=500,
            left_wheel_speed_weight=0.0,
        )


def test_motion_limits_reject_minimum_above_maximum_wheel_velocity() -> None:
    with pytest.raises(ValueError, match="min_wheel_velocity_m_s"):
        MotionLimits(
            wheel_track_m=0.20,
            max_linear_velocity_m_s=0.30,
            max_angular_velocity_rad_s=2.0,
            max_wheel_velocity_m_s=0.04,
            max_linear_acceleration_m_s2=0.50,
            max_linear_deceleration_m_s2=0.50,
            max_angular_acceleration_rad_s2=5,
            max_angular_deceleration_rad_s2=5,
            max_remote_command_valid_for_ms=500,
            min_wheel_velocity_m_s=0.05,
            stall_guard_min_command_speed_m_s=0.01,
        )


def test_gripper_angles_encode_left_then_right_and_reject_invalid_values() -> None:
    channel = FakeCarChannel()
    controller = MotionController(channel, limits())

    controller.set_gripper_angles(27.0, 167.0)
    controller.set_gripper_angles(0.0, 180.0)
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        controller.set_gripper_angles(-1.0, 90.0)
    with pytest.raises(ValueError, match="finite"):
        controller.set_gripper_angles(float("nan"), 90.0)

    assert channel.sent == [
        encode_gripper_command(0, 27.0, 167.0),
        encode_gripper_command(1, 0.0, 180.0),
    ]


def test_gripper_target_uses_initial_telemetry_then_local_commands() -> None:
    clock = FakeClock(100)
    channel = FakeCarChannel(
        [
            system_status_frame(0, 99),
            system_status_frame(1, 101, left_cdeg=3000, right_cdeg=15000),
        ]
    )
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    controller.receive_message(timeout=0)
    assert controller.gripper_target_angles_deg == pytest.approx((90.0, 90.0))
    controller.set_gripper_angles(20.0, 160.0)
    assert controller.gripper_target_angles_deg == pytest.approx((20.0, 160.0))
    controller.receive_message(timeout=0)
    assert controller.gripper_target_angles_deg == pytest.approx((20.0, 160.0))


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
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.0, 0.0),
        encode_wheel_speed_command(1, 0.05, -0.05),
        encode_wheel_speed_command(2, 0.0, 0.0),
        encode_wheel_speed_command(3, -0.05, 0.05),
    ]


def test_wheel_acceleration_override_slows_only_left_wheel_transition() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.set_wheel_acceleration_limits(
        WheelAccelerationOverrides(
            left_acceleration_m_s2=0.5,
            left_deceleration_m_s2=0.5,
        )
    )

    controller.set_wheel_speeds(0.30, 0.30)
    for _ in range(10):
        clock.advance(0.1)
        controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.30, 0.30))

    controller.set_wheel_speeds(0.10, 0.30)
    clock.advance(0.1)
    controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.25, 0.30))


def test_linear_acceleration_and_deceleration_are_independent() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    custom_limits = MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.30,
        max_angular_velocity_rad_s=2.0,
        max_wheel_velocity_m_s=0.40,
        max_linear_acceleration_m_s2=0.20,
        max_linear_deceleration_m_s2=0.50,
        max_angular_acceleration_rad_s2=1.0,
        max_angular_deceleration_rad_s2=3.0,
        max_remote_command_valid_for_ms=500,
    )
    controller = MotionController(channel, custom_limits, monotonic_ns=clock)

    controller.forward(0.30)
    for _ in range(4):
        clock.advance(0.10)
        controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.08, 0.08))

    controller.forward(0.0)
    clock.advance(0.10)
    controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.03, 0.03))


def test_angular_acceleration_and_deceleration_are_independent() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    custom_limits = MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.30,
        max_angular_velocity_rad_s=2.0,
        max_wheel_velocity_m_s=0.40,
        max_linear_acceleration_m_s2=0.20,
        max_linear_deceleration_m_s2=0.50,
        max_angular_acceleration_rad_s2=1.0,
        max_angular_deceleration_rad_s2=3.0,
        max_remote_command_valid_for_ms=500,
    )
    controller = MotionController(channel, custom_limits, monotonic_ns=clock)

    controller.turn_left(2.0)
    for _ in range(4):
        clock.advance(0.10)
        controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((-0.04, 0.04))

    controller.turn_left(0.0)
    clock.advance(0.10)
    controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((-0.01, 0.01))


def test_direction_reversal_decelerates_to_zero_before_accelerating() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    custom_limits = MotionLimits(
        wheel_track_m=0.20,
        max_linear_velocity_m_s=0.30,
        max_angular_velocity_rad_s=2.0,
        max_wheel_velocity_m_s=0.40,
        max_linear_acceleration_m_s2=0.20,
        max_linear_deceleration_m_s2=0.50,
        max_angular_acceleration_rad_s2=1.0,
        max_angular_deceleration_rad_s2=3.0,
        max_remote_command_valid_for_ms=500,
    )
    controller = MotionController(channel, custom_limits, monotonic_ns=clock)

    controller.forward(0.30)
    for _ in range(4):
        clock.advance(0.10)
        controller.update()
    controller.backward(0.30)
    clock.advance(0.20)
    controller.update()

    # 0.16 s is consumed braking 0.08 m/s to zero; only the remaining
    # 0.04 s accelerates backward at 0.20 m/s².
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((-0.008, -0.008))


def test_unchanged_wheel_target_is_refreshed_for_firmware_watchdog() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    clock.advance(0.039)
    assert not controller.update()
    clock.advance(0.001)
    assert controller.update()
    clock.advance(0.040)
    assert controller.update()

    assert channel.sent == [
        encode_wheel_speed_command(0, 0.0, 0.0),
        encode_wheel_speed_command(1, 0.0, 0.0),
    ]


def test_changing_wheel_target_is_throttled_to_watchdog_refresh_period() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.forward(0.20)

    clock.advance(0.005)
    assert not controller.update()
    clock.advance(0.035)
    assert controller.update()
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.02, 0.02))

    clock.advance(0.005)
    assert not controller.update()
    assert len(channel.sent) == 1


def test_active_control_loop_gap_soft_brakes_instead_of_resuming_target() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.forward(0.20)
    clock.advance(0.05)
    assert controller.update()

    clock.advance(0.201)
    with pytest.raises(MotionControlTimingError, match="soft brake"):
        controller.update()

    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
    assert controller.commanded_wheel_speeds_m_s == (0.0, 0.0)
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.025, 0.025),
        encode_soft_brake_command(1),
    ]


def test_encoder_stall_guard_soft_brakes_and_reports_fault() -> None:
    channel = FakeCarChannel(
        [
            odometry_frame(0, 0, left_encoder_count=100, right_encoder_count=100),
        ]
    )
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    controller.receive_message(timeout=0)
    controller.forward(0.10)
    clock.advance(0.10)
    assert controller.update()
    channel.received.extend(
        [
            odometry_frame(
                1,
                100_000_000,
                telemetry_sequence=1,
                left_encoder_count=100,
                right_encoder_count=100,
            ),
            odometry_frame(
                2,
                200_000_000,
                telemetry_sequence=2,
                left_encoder_count=100,
                right_encoder_count=100,
            ),
            odometry_frame(
                3,
                300_000_000,
                telemetry_sequence=3,
                left_encoder_count=100,
                right_encoder_count=100,
            ),
            odometry_frame(
                4,
                400_000_000,
                telemetry_sequence=4,
                left_encoder_count=100,
                right_encoder_count=100,
            ),
        ]
    )

    with pytest.raises(MotionStallError, match="left wheel encoder"):
        controller.drain_messages()

    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
    assert controller.commanded_wheel_speeds_m_s == (0.0, 0.0)
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.05, 0.05),
        encode_soft_brake_command(1),
    ]


def test_encoder_stall_guard_does_not_trigger_when_encoder_moves() -> None:
    channel = FakeCarChannel(
        [
            odometry_frame(0, 0, left_encoder_count=100, right_encoder_count=100),
        ]
    )
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    controller.receive_message(timeout=0)
    controller.forward(0.10)
    clock.advance(0.04)
    assert controller.update()
    channel.received.extend(
        [
            odometry_frame(
                1,
                100_000_000,
                telemetry_sequence=1,
                left_encoder_count=101,
                right_encoder_count=101,
            ),
            odometry_frame(
                2,
                200_000_000,
                telemetry_sequence=2,
                left_encoder_count=102,
                right_encoder_count=102,
            ),
            odometry_frame(
                3,
                300_000_000,
                telemetry_sequence=3,
                left_encoder_count=103,
                right_encoder_count=103,
            ),
            odometry_frame(
                4,
                400_000_000,
                telemetry_sequence=4,
                left_encoder_count=104,
                right_encoder_count=104,
            ),
        ]
    )

    controller.drain_messages()

    assert controller.target_wheel_speeds_m_s == pytest.approx((0.10, 0.10))
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.02, 0.02))
    assert channel.sent == [encode_wheel_speed_command(0, 0.02, 0.02)]


def test_idle_control_loop_gap_only_refreshes_zero_watchdog_command() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    clock.advance(1.0)
    assert controller.update()

    assert channel.sent == [encode_wheel_speed_command(0, 0.0, 0.0)]


def test_random_joystick_stress_never_emits_unsigned_or_overlimit_wheels() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    generator = random.Random(20260824)

    for _ in range(2_000):
        controller.drive_wheel_limited(
            generator.uniform(-0.25, 0.25),
            generator.uniform(-1.0, 1.0),
        )
        clock.advance(0.02)
        controller.update()

    signed_wheels: list[tuple[int, int]] = []
    for frame in channel.sent:
        assert frame[0] == MessageType.SET_WHEEL_SPEED
        assert struct.unpack("<H", frame[-2:])[0] == crc16_ccitt_false(frame[:-2])
        _, left_mm_s, right_mm_s = struct.unpack("<Hhh", frame[1:-2])
        signed_wheels.append((left_mm_s, right_mm_s))

    assert signed_wheels
    assert any(left < 0 or right < 0 for left, right in signed_wheels)
    assert all(
        abs(left) <= 300 and abs(right) <= 300
        for left, right in signed_wheels
    )


def test_soft_brake_clears_pending_acceleration_target() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.forward(0.30)
    clock.advance(0.1)
    controller.update()

    controller.soft_brake()
    clock.advance(1.0)

    assert controller.update()
    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
    assert controller.commanded_wheel_speeds_m_s == (0.0, 0.0)
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.05, 0.05),
        encode_soft_brake_command(1),
        encode_wheel_speed_command(2, 0.0, 0.0),
    ]


def test_temporary_acceleration_limits_are_applied_and_restored() -> None:
    channel = FakeCarChannel()
    clock = FakeClock()
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    controller.set_acceleration_limits(
        linear_acceleration_m_s2=0.10,
        linear_deceleration_m_s2=0.20,
        angular_acceleration_rad_s2=1.0,
        angular_deceleration_rad_s2=2.0,
    )
    controller.forward(0.30)
    clock.advance(0.1)
    controller.update()

    assert controller.acceleration_limits == MotionAccelerationLimits(
        linear_acceleration_m_s2=0.10,
        linear_deceleration_m_s2=0.20,
        angular_acceleration_rad_s2=1.0,
        angular_deceleration_rad_s2=2.0,
    )
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.01, 0.01))

    controller.set_acceleration_limits()
    clock.advance(0.1)
    controller.update()

    assert controller.acceleration_limits == MotionAccelerationLimits(
        linear_acceleration_m_s2=0.50,
        linear_deceleration_m_s2=0.50,
        angular_acceleration_rad_s2=5.0,
        angular_deceleration_rad_s2=5.0,
    )
    assert controller.commanded_wheel_speeds_m_s == pytest.approx((0.06, 0.06))


def test_temporary_acceleration_limit_can_exceed_global_limit() -> None:
    controller = MotionController(FakeCarChannel(), limits())

    controller.set_acceleration_limits(linear_acceleration_m_s2=0.51)
    assert (
        controller.acceleration_limits.linear_acceleration_m_s2
        == pytest.approx(0.51)
    )

    with pytest.raises(ValueError, match="linear_acceleration_m_s2"):
        controller.set_acceleration_limits(linear_acceleration_m_s2=0.0)


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


@pytest.mark.parametrize(
    ("linear", "angular"),
    [
        (0.2, -0.8),
        (0.2, 0.8),
        (-0.2, -0.8),
        (-0.2, 0.8),
    ],
)
def test_wheel_limited_drive_preserves_curvature_at_joystick_diagonal(
    linear: float,
    angular: float,
) -> None:
    channel = FakeCarChannel()
    controller = MotionController(
        channel,
        MotionLimits(
            wheel_track_m=0.275,
            max_linear_velocity_m_s=0.25,
            max_angular_velocity_rad_s=1.0,
            max_wheel_velocity_m_s=0.30,
            max_linear_acceleration_m_s2=0.50,
            max_linear_deceleration_m_s2=0.50,
            max_angular_acceleration_rad_s2=5,
            max_angular_deceleration_rad_s2=5,
            max_remote_command_valid_for_ms=500,
        ),
    )

    applied_linear, applied_angular = controller.drive_wheel_limited(
        linear,
        angular,
    )

    scale = 0.30 / 0.31
    original_wheels = (
        linear - angular * 0.275 / 2.0,
        linear + angular * 0.275 / 2.0,
    )
    assert applied_linear == pytest.approx(linear * scale)
    assert applied_angular == pytest.approx(angular * scale)
    assert controller.target_wheel_speeds_m_s == pytest.approx(
        tuple(wheel * scale for wheel in original_wheels)
    )
    assert max(map(abs, controller.target_wheel_speeds_m_s)) == pytest.approx(
        0.30
    )
    assert applied_angular / applied_linear == pytest.approx(angular / linear)
    assert channel.sent == []


def test_wheel_limited_drive_still_rejects_body_velocity_limit() -> None:
    channel = FakeCarChannel()
    controller = MotionController(channel, limits())

    with pytest.raises(ValueError, match="linear_velocity"):
        controller.drive_wheel_limited(0.31, 0.0)
    with pytest.raises(ValueError, match="angular_velocity"):
        controller.drive_wheel_limited(0.0, 2.1)

    assert channel.sent == []


def test_controller_discards_invalid_protocol_frames_before_valid_message() -> None:
    invalid_crc = bytearray(
        command_reply_frame(0, 1_000, command_sequence=3).payload
    )
    invalid_crc[-1] ^= 0xFF
    channel = FakeCarChannel(
        [
            ReceivedUartFrame(0, 1_000, bytes(invalid_crc)),
            command_reply_frame(1, 3_000, command_sequence=4),
        ]
    )
    controller = MotionController(channel, limits())

    assert controller.receive_message(timeout=0) == CarCommandReply(
        uart_sequence=1,
        received_timestamp_ns=3_000,
        command_sequence=4,
        command_type=MessageType.SET_WHEEL_SPEED,
        result=CommandResult.ACCEPTED,
    )
    assert controller.invalid_received_frames == 1


def test_motion_synchronization_waits_for_matching_soft_brake_reply() -> None:
    channel = FakeCarChannel(
        [
            system_status_frame(0, 1_000),
            command_reply_frame(
                1,
                2_000,
                command_sequence=0,
                command_type=MessageType.SOFT_BRAKE,
            ),
        ]
    )
    controller = MotionController(channel, limits())
    received: list[object] = []

    controller.synchronize(on_message=received.append)

    assert controller.motion_synchronized
    assert not controller.needs_synchronization
    assert channel.sent == [encode_soft_brake_command(0)]
    assert isinstance(received[0], CarSystemStatus)
    assert isinstance(received[1], CarCommandReply)


def test_timed_out_motion_synchronization_can_send_a_new_attempt() -> None:
    channel = FakeCarChannel()
    controller = MotionController(channel, limits())

    with pytest.raises(MotionSynchronizationError):
        controller.synchronize(timeout_s=0.001)

    assert controller.needs_synchronization
    channel.received.append(
        command_reply_frame(
            1,
            2_000,
            command_sequence=1,
            command_type=MessageType.SOFT_BRAKE,
        )
    )

    controller.synchronize(timeout_s=0.001)

    assert channel.sent == [
        encode_soft_brake_command(0),
        encode_soft_brake_command(1),
    ]
    assert controller.motion_synchronized


def test_sticky_rx_degraded_status_does_not_gate_wheel_motion() -> None:
    clock = FakeClock()
    channel = FakeCarChannel(
        [
            system_status_frame(
                0,
                1,
                system_flags=SystemFlags.PROTOCOL_READY | SystemFlags.RX_DEGRADED,
            )
        ]
    )
    controller = MotionController(channel, limits(), monotonic_ns=clock)

    message = controller.receive_message(timeout=0)
    assert isinstance(message, CarSystemStatus)
    assert message.rx_degraded
    assert not controller.link_degraded

    controller.forward(0.1)
    clock.advance(0.04)
    assert controller.update()
    assert channel.sent == [encode_wheel_speed_command(0, 0.02, 0.02)]


def test_sequence_old_requests_soft_brake_resynchronization() -> None:
    clock = FakeClock()
    channel = FakeCarChannel(
        [
            command_reply_frame(
                0,
                1,
                command_sequence=0,
                command_type=MessageType.SOFT_BRAKE,
            )
        ]
    )
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.synchronize()

    controller.forward(0.1)
    clock.advance(0.05)
    assert controller.update()
    channel.received.append(
        command_reply_frame(
            1,
            2,
            command_sequence=1,
            command_type=MessageType.SET_WHEEL_SPEED,
            result=CommandResult.SEQUENCE_OLD,
        )
    )
    assert isinstance(controller.receive_message(timeout=0), CarCommandReply)

    assert controller.needs_synchronization
    assert controller.target_wheel_speeds_m_s == (0.0, 0.0)
    assert channel.sent == [
        encode_soft_brake_command(0),
        encode_wheel_speed_command(1, 0.025, 0.025),
        encode_soft_brake_command(2),
    ]

    channel.received.append(
        command_reply_frame(
            2,
            3,
            command_sequence=2,
            command_type=MessageType.SOFT_BRAKE,
        )
    )
    controller.synchronize()
    assert controller.motion_synchronized


def test_unacknowledged_wheel_stream_enters_soft_brake_resynchronization() -> None:
    clock = FakeClock()
    channel = FakeCarChannel(
        [
            command_reply_frame(
                0,
                1,
                command_sequence=0,
                command_type=MessageType.SOFT_BRAKE,
            )
        ]
    )
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.synchronize()
    controller.forward(0.1)

    clock.advance(0.04)
    assert controller.update()
    clock.advance(0.04)
    assert not controller.update()
    clock.advance(0.04)
    assert not controller.update()
    clock.advance(0.04)
    assert controller.update()

    assert controller.needs_synchronization
    assert channel.sent[-1] == encode_soft_brake_command(2)


def test_parse_system_status_and_initial_gripper_target() -> None:
    status = parse_controller_frame(system_status_frame(7, 9_000))

    assert status == CarSystemStatus(
        uart_sequence=7,
        received_timestamp_ns=9_000,
        status_sequence=1,
        controller_timestamp_us=10_000,
        watchdog_timeout_ms=300,
        last_motion_command_age_ms=25,
        system_flags=SystemFlags.PROTOCOL_READY,
        stop_reason=CarStopReason.RUNNING,
        servo_left_target_cdeg=9000,
        servo_right_target_cdeg=9000,
    )


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
    clock.advance(0.1)
    assert controller.update()
    assert channel.sent == [encode_wheel_speed_command(0, 0.025, 0.075)]
    assert not executor.check_timeout(now_ns=1_199_999_999)
    assert executor.next_wait_s(
        0.05,
        now_ns=1_190_000_000,
    ) == pytest.approx(0.01)
    assert executor.check_timeout(now_ns=1_200_000_000)
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.025, 0.075),
        encode_soft_brake_command(1),
    ]


def test_remote_twist_scales_coupled_wheel_limit_instead_of_stopping() -> None:
    channel = FakeCarChannel()
    controller = MotionController(
        channel,
        MotionLimits(
            wheel_track_m=0.275,
            max_linear_velocity_m_s=0.25,
            max_angular_velocity_rad_s=1.0,
            max_wheel_velocity_m_s=0.30,
            max_linear_acceleration_m_s2=0.50,
            max_linear_deceleration_m_s2=0.50,
            max_angular_acceleration_rad_s2=5,
            max_angular_deceleration_rad_s2=5,
            max_remote_command_valid_for_ms=500,
        ),
    )
    executor = RemoteMotionExecutor(
        controller,
        monotonic_ns=lambda: 1_050_000_000,
    )

    result = executor.execute(
        remote_message(
            twist_command(
                linear_velocity_m_s=0.2,
                angular_velocity_rad_s=-0.8,
            )
        )
    )

    scale = 0.30 / 0.31
    assert result.result is RemoteMotionResult.APPLIED
    assert result.linear_velocity_m_s == pytest.approx(0.2 * scale)
    assert result.angular_velocity_rad_s == pytest.approx(-0.8 * scale)
    assert controller.target_wheel_speeds_m_s == pytest.approx(
        (0.30, 0.09 * scale)
    )
    assert executor.active_deadline_ns == 1_200_000_000
    assert channel.sent == []


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
    for _ in range(4):
        clock.advance(0.1)
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
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.05, 0.05),
        encode_wheel_speed_command(1, 0.1, 0.1),
        encode_wheel_speed_command(2, 0.15, 0.15),
        encode_wheel_speed_command(3, 0.2, 0.2),
    ]

    for _ in range(4):
        clock.advance(0.1)
        assert controller.update()

    assert controller.commanded_wheel_speeds_m_s == (0.0, 0.0)
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.05, 0.05),
        encode_wheel_speed_command(1, 0.1, 0.1),
        encode_wheel_speed_command(2, 0.15, 0.15),
        encode_wheel_speed_command(3, 0.2, 0.2),
        encode_wheel_speed_command(4, 0.15, 0.15),
        encode_wheel_speed_command(5, 0.1, 0.1),
        encode_wheel_speed_command(6, 0.05, 0.05),
        encode_wheel_speed_command(7, 0.0, 0.0),
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
    assert channel.sent == [
        encode_soft_brake_command(0),
        encode_soft_brake_command(1),
    ]


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

    assert channel.sent == [
        encode_soft_brake_command(0),
        encode_soft_brake_command(1),
        encode_soft_brake_command(2),
    ]


def test_remote_validity_limit_is_enforced_without_using_sender_clock() -> None:
    channel = FakeCarChannel()
    executor = RemoteMotionExecutor(MotionController(channel, limits()))

    with pytest.raises(RemoteMotionError, match="valid_for_ms"):
        executor.execute(
            remote_message(twist_command(valid_for_ms=501)),
            now_ns=1_000_000_001,
        )

    assert channel.sent == [encode_soft_brake_command(0)]


def test_remote_gripper_advances_while_refreshed_and_stops_on_release() -> None:
    clock = FakeClock(1_050_000_000)
    channel = FakeCarChannel(
        [
            system_status_frame(
                0,
                1_000_000_000,
                left_cdeg=2000,
                right_cdeg=17400,
            )
        ]
    )
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.receive_message(timeout=0)
    executor = RemoteGripperExecutor(
        controller,
        gripper_calibration(),
        monotonic_ns=clock,
    )

    applied = executor.execute(gripper_message(gripper_command()))
    clock.advance(0.1)
    assert executor.update()
    stopped = executor.execute(
        gripper_message(
            gripper_command(open_pressed=False, close_pressed=False),
            received_timestamp_ns=clock.timestamp_ns,
        )
    )
    clock.advance(0.1)
    assert not executor.update()
    expired = executor.execute(
        gripper_message(gripper_command(), received_timestamp_ns=800_000_000)
    )

    assert applied.result is RemoteGripperResult.APPLIED
    assert applied.deadline_timestamp_ns == 1_200_000_000
    assert stopped.result is RemoteGripperResult.STOPPED
    assert expired.result is RemoteGripperResult.EXPIRED
    assert channel.sent == [encode_gripper_command(0, 26.0, 168.0)]


def test_remote_gripper_uses_fixed_speed_and_stops_on_conflicting_triggers() -> None:
    clock = FakeClock(1_000_000_000)
    channel = FakeCarChannel()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.set_gripper_angles(50.0, 144.0)
    channel.sent.clear()
    executor = RemoteGripperExecutor(
        controller,
        gripper_calibration(),
        monotonic_ns=clock,
    )

    executor.execute(
        gripper_message(
            gripper_command(
                valid_for_ms=500,
                open_pressed=True,
                close_pressed=False,
            )
        ),
        now_ns=clock.timestamp_ns,
    )
    clock.advance(0.2)
    assert executor.update()
    cancelled = executor.execute(
        gripper_message(
            gripper_command(open_pressed=True, close_pressed=True),
            received_timestamp_ns=clock.timestamp_ns,
        ),
        now_ns=clock.timestamp_ns,
    )

    assert cancelled.result is RemoteGripperResult.STOPPED
    assert channel.sent == [encode_gripper_command(1, 38.0, 156.0)]


def test_remote_gripper_timeout_stops_at_last_target() -> None:
    clock = FakeClock(1_000_000_000)
    channel = FakeCarChannel()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.set_gripper_angles(20.0, 174.0)
    channel.sent.clear()
    executor = RemoteGripperExecutor(
        controller,
        gripper_calibration(),
        monotonic_ns=clock,
    )
    executor.execute(
        gripper_message(gripper_command(valid_for_ms=100)),
        now_ns=clock.timestamp_ns,
    )

    clock.advance(0.05)
    assert executor.update()
    clock.advance(0.05)
    assert executor.check_timeout() == "grip-1"
    clock.advance(0.1)
    assert not executor.update()
    assert controller.gripper_target_angles_deg == pytest.approx((23.0, 171.0))
    assert channel.sent == [encode_gripper_command(1, 23.0, 171.0)]


def test_remote_gripper_projects_state_to_194_degree_sum_before_motion() -> None:
    clock = FakeClock(1_000_000_000)
    channel = FakeCarChannel()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.set_gripper_angles(20.0, 20.0)
    channel.sent.clear()
    executor = RemoteGripperExecutor(
        controller,
        gripper_calibration(),
        monotonic_ns=clock,
    )
    executor.execute(
        gripper_message(gripper_command(valid_for_ms=500)),
        now_ns=clock.timestamp_ns,
    )

    clock.advance(0.1)
    assert executor.update()

    left, right = controller.gripper_target_angles_deg or (0.0, 0.0)
    assert left == pytest.approx(91.0)
    assert right == pytest.approx(103.0)
    assert left + right == pytest.approx(194.0)
    assert channel.sent == [encode_gripper_command(1, 91.0, 103.0)]


def test_remote_gripper_uses_configured_angle_sum() -> None:
    clock = FakeClock(1_000_000_000)
    channel = FakeCarChannel()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.set_gripper_angles(20.0, 20.0)
    channel.sent.clear()
    calibration = GripperCalibration(
        open_left_angle_deg=30.0,
        open_right_angle_deg=170.0,
        closed_left_angle_deg=90.0,
        closed_right_angle_deg=110.0,
        full_travel_time_s=1.0,
        angle_sum_deg=200.0,
    )
    executor = RemoteGripperExecutor(
        controller,
        calibration,
        monotonic_ns=clock,
    )
    executor.execute(
        gripper_message(gripper_command(valid_for_ms=500)),
        now_ns=clock.timestamp_ns,
    )

    clock.advance(0.1)
    assert executor.update()

    left, right = controller.gripper_target_angles_deg or (0.0, 0.0)
    assert left + right == pytest.approx(200.0)
    assert channel.sent == [encode_gripper_command(1, 94.0, 106.0)]


def test_remote_gripper_projects_custom_sum_inside_servo_range() -> None:
    clock = FakeClock(1_000_000_000)
    channel = FakeCarChannel()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.set_gripper_angles(0.0, 180.0)
    channel.sent.clear()
    calibration = GripperCalibration(
        open_left_angle_deg=20.0,
        open_right_angle_deg=80.0,
        closed_left_angle_deg=80.0,
        closed_right_angle_deg=20.0,
        full_travel_time_s=1.0,
        angle_sum_deg=100.0,
    )
    executor = RemoteGripperExecutor(
        controller,
        calibration,
        monotonic_ns=clock,
    )
    executor.execute(
        gripper_message(gripper_command(valid_for_ms=500)),
        now_ns=clock.timestamp_ns,
    )

    clock.advance(0.1)
    assert executor.update()

    left, right = controller.gripper_target_angles_deg or (0.0, 0.0)
    assert left == pytest.approx(6.0)
    assert right == pytest.approx(94.0)
    assert 0.0 <= left <= 180.0
    assert 0.0 <= right <= 180.0


def test_remote_gripper_clamps_projected_state_to_valid_servo_range() -> None:
    clock = FakeClock(1_000_000_000)
    channel = FakeCarChannel()
    controller = MotionController(channel, limits(), monotonic_ns=clock)
    controller.set_gripper_angles(0.0, 180.0)
    channel.sent.clear()
    executor = RemoteGripperExecutor(
        controller,
        gripper_calibration(),
        monotonic_ns=clock,
    )
    executor.execute(
        gripper_message(gripper_command(valid_for_ms=500)),
        now_ns=clock.timestamp_ns,
    )

    clock.advance(0.1)
    assert executor.update()

    assert controller.gripper_target_angles_deg == pytest.approx((20.0, 174.0))
    assert channel.sent == [encode_gripper_command(1, 20.0, 174.0)]


def test_invalid_remote_gripper_command_does_not_actuate() -> None:
    channel = FakeCarChannel()
    executor = RemoteGripperExecutor(
        MotionController(channel, limits()),
        gripper_calibration(),
    )

    executor.execute(
        gripper_message(gripper_command()),
        now_ns=1_000_000_001,
    )
    with pytest.raises(RemoteGripperError, match="valid_for_ms"):
        executor.execute(
            gripper_message(gripper_command(valid_for_ms=501)),
            now_ns=1_000_000_002,
        )
    assert executor.active_command_id is None
    with pytest.raises(RemoteGripperError, match="topic"):
        executor.execute(
            gripper_message(
                gripper_command(),
                topic=RemoteTopic.DEBUG_CAPTURE.value,
            ),
            now_ns=1_000_000_003,
        )

    assert channel.sent == []


def test_remote_loop_drains_uart_and_stops_on_exit() -> None:
    invalid_crc = bytearray(
        command_reply_frame(0, 8, command_sequence=5).payload
    )
    invalid_crc[-1] ^= 0x01
    channel = FakeCarChannel(
        [
            ReceivedUartFrame(0, 8, bytes(invalid_crc)),
            system_status_frame(1, 11),
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

    assert car_messages == [parse_controller_frame(system_status_frame(1, 11))]
    assert executor.controller.invalid_received_frames == 1
    assert channel.sent == [
        encode_wheel_speed_command(0, 0.0, 0.0),
        encode_wheel_speed_command(1, 0.025, 0.075),
        encode_soft_brake_command(2),
    ]


def test_remote_loop_retries_timed_out_start_synchronization() -> None:
    class RetrySynchronizationChannel(FakeCarChannel):
        def send_frame(self, payload: bytes) -> None:
            super().send_frame(payload)
            if payload == encode_soft_brake_command(1):
                self.received.append(
                    command_reply_frame(
                        0,
                        1,
                        command_sequence=1,
                        command_type=MessageType.SOFT_BRAKE,
                    )
                )

    channel = RetrySynchronizationChannel()
    executor = RemoteMotionExecutor(MotionController(channel, limits()))
    stop_checks = 0

    def stop_requested() -> bool:
        nonlocal stop_checks
        stop_checks += 1
        return stop_checks >= 3

    run_remote_motion(
        FakeRemoteReceiver(iter(())),
        executor,
        stop_requested=stop_requested,
        synchronize_on_start=True,
        synchronization_timeout_s=0.001,
    )

    assert channel.sent == [
        encode_soft_brake_command(0),
        encode_soft_brake_command(1),
        encode_soft_brake_command(2),
    ]


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
    assert channel.sent == [encode_soft_brake_command(0)]


def test_remote_loop_uart_receive_fault_stops_before_propagating() -> None:
    class FaultingCarChannel(FakeCarChannel):
        def receive_frame(self, timeout: float | None = None) -> ReceivedUartFrame:
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

    assert channel.sent == [encode_soft_brake_command(0)]
