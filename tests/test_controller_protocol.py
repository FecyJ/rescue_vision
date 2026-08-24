from __future__ import annotations

import struct

import pytest

from rescue_vision.communication import ReceivedUartFrame
from rescue_vision.motion import (
    CarCommandReply,
    CarStopReason,
    CarSystemStatus,
    CommandResult,
    ControllerProtocolError,
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
)


def received(message_type: MessageType, payload: bytes) -> ReceivedUartFrame:
    return ReceivedUartFrame(
        sequence=7,
        received_timestamp_ns=9_000,
        payload=pack_protocol_frame(message_type, payload),
    )


def test_crc16_matches_frozen_check_value() -> None:
    assert crc16_ccitt_false(b"123456789") == 0x29B1


def test_commands_match_fixed_little_endian_layouts() -> None:
    wheel = encode_wheel_speed_command(0x1234, -0.125, 0.25)
    gripper = encode_gripper_command(9, 27.25, 167.5)

    assert len(wheel) == 9
    assert wheel[:-2] == b"\x10" + struct.pack("<Hhh", 0x1234, -125, 250)
    assert len(gripper) == 9
    assert gripper[:-2] == b"\x13" + struct.pack("<HHH", 9, 2725, 16750)
    assert encode_soft_brake_command(10)[:-2] == b"\x11\x0a\x00"
    assert encode_emergency_stop_command(11)[:-2] == b"\x12\x0b\x00"
    assert encode_state_query_command(12)[:-2] == b"\x14\x0c\x00"
    for frame in (wheel, gripper):
        assert struct.unpack("<H", frame[-2:])[0] == crc16_ccitt_false(frame[:-2])


def test_command_encoder_rejects_invalid_units_and_ranges() -> None:
    with pytest.raises(ValueError, match="command_sequence"):
        encode_soft_brake_command(65536)
    with pytest.raises(ValueError, match="finite"):
        encode_wheel_speed_command(0, float("nan"), 0.0)
    with pytest.raises(ValueError, match="int16"):
        encode_wheel_speed_command(0, 40.0, 0.0)
    with pytest.raises(ValueError, match=r"\[0, 180\]"):
        encode_gripper_command(0, 181.0, 0.0)


def test_parse_command_reply() -> None:
    message = parse_controller_frame(
        received(
            MessageType.COMMAND_REPLY,
            struct.pack(
                "<HBB",
                42,
                MessageType.SET_WHEEL_SPEED,
                CommandResult.ACCEPTED,
            ),
        )
    )

    assert message == CarCommandReply(
        uart_sequence=7,
        received_timestamp_ns=9_000,
        command_sequence=42,
        command_type=MessageType.SET_WHEEL_SPEED,
        result=CommandResult.ACCEPTED,
    )


def test_parse_odometry_imu_preserves_wire_units_and_flags() -> None:
    flags = (
        SensorFlags.IMU_VALID
        | SensorFlags.IMU_CALIBRATED
        | SensorFlags.LEFT_ENCODER_VALID
        | SensorFlags.RIGHT_ENCODER_VALID
    )
    payload = struct.pack(
        "<HQqqiiiiiihH",
        65535,
        123_456_789,
        -100,
        200,
        1,
        -2,
        300_000,
        10,
        -20,
        9_807,
        3642,
        int(flags),
    )

    message = parse_controller_frame(received(MessageType.ODOMETRY_IMU, payload))

    assert message == OdometryImu(
        uart_sequence=7,
        received_timestamp_ns=9_000,
        telemetry_sequence=65535,
        sample_timestamp_us=123_456_789,
        left_encoder_count=-100,
        right_encoder_count=200,
        gyro_x_urad_s=1,
        gyro_y_urad_s=-2,
        gyro_z_urad_s=300_000,
        accel_x_mm_s2=10,
        accel_y_mm_s2=-20,
        accel_z_mm_s2=9_807,
        imu_temperature_cdeg=3642,
        sensor_flags=flags,
    )
    assert message.gyro_z_rad_s == pytest.approx(0.3)


def test_parse_system_status_decodes_sentinel_flags_and_angles() -> None:
    flags = SystemFlags.WATCHDOG_ARMED | SystemFlags.GRIPPER_OUTPUT_AVAILABLE
    payload = struct.pack(
        "<HQHIHBHH",
        3,
        9_876_543,
        300,
        0xFFFFFFFF,
        int(flags),
        CarStopReason.STARTUP,
        2725,
        16750,
    )

    message = parse_controller_frame(received(MessageType.SYSTEM_STATUS, payload))

    assert message == CarSystemStatus(
        uart_sequence=7,
        received_timestamp_ns=9_000,
        status_sequence=3,
        controller_timestamp_us=9_876_543,
        watchdog_timeout_ms=300,
        last_motion_command_age_ms=None,
        system_flags=flags,
        stop_reason=CarStopReason.STARTUP,
        servo_left_target_cdeg=2725,
        servo_right_target_cdeg=16750,
    )
    assert message.watchdog_armed
    assert not message.emergency_stop_latched
    assert message.servo_left_deg == pytest.approx(27.25)


def test_parser_rejects_crc_length_type_direction_enum_and_undefined_flags() -> None:
    valid = bytearray(
        received(
            MessageType.COMMAND_REPLY,
            struct.pack("<HBB", 1, MessageType.SOFT_BRAKE, CommandResult.ACCEPTED),
        ).payload
    )
    valid[-1] ^= 1
    unknown_body = b"\x99"
    unknown_type = ReceivedUartFrame(
        0,
        0,
        unknown_body + struct.pack("<H", crc16_ccitt_false(unknown_body)),
    )
    cases = [
        ReceivedUartFrame(0, 0, bytes(valid)),
        received(MessageType.COMMAND_REPLY, b"\x00"),
        unknown_type,
        received(MessageType.SET_WHEEL_SPEED, struct.pack("<Hhh", 0, 0, 0)),
        received(MessageType.COMMAND_REPLY, struct.pack("<HBB", 1, 0x10, 0xFF)),
        received(
            MessageType.ODOMETRY_IMU,
            struct.pack("<HQqqiiiiiihH", 0, 1, 0, 0, 0, 0, 0, 0, 0, 9807, 2500, 1 << 15),
        ),
    ]

    for frame in cases:
        with pytest.raises(ControllerProtocolError):
            parse_controller_frame(frame)
