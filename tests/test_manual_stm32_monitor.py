from __future__ import annotations

from manual_tests.stm32_monitor import (
    MonitorStats,
    SequenceHealth,
    format_message,
    format_summary,
)
from rescue_vision.motion import (
    CarCommandReply,
    CarStopReason,
    CarSystemStatus,
    CommandResult,
    MessageType,
    OdometryImu,
    SensorFlags,
    SystemFlags,
)


def odometry(sequence: int) -> OdometryImu:
    return OdometryImu(
        uart_sequence=sequence,
        received_timestamp_ns=1_000_000_000 + sequence,
        telemetry_sequence=sequence,
        sample_timestamp_us=10_000 * sequence,
        left_encoder_count=-12,
        right_encoder_count=34,
        gyro_x_urad_s=1_000,
        gyro_y_urad_s=-2_000,
        gyro_z_urad_s=300_000,
        accel_x_mm_s2=10,
        accel_y_mm_s2=-20,
        accel_z_mm_s2=9_807,
        imu_temperature_cdeg=3_625,
        sensor_flags=(
            SensorFlags.IMU_VALID
            | SensorFlags.LEFT_ENCODER_VALID
            | SensorFlags.RIGHT_ENCODER_VALID
        ),
    )


def system_status(sequence: int) -> CarSystemStatus:
    return CarSystemStatus(
        uart_sequence=sequence,
        received_timestamp_ns=1_000_000_000 + sequence,
        status_sequence=sequence,
        controller_timestamp_us=100_000 * sequence,
        watchdog_timeout_ms=300,
        last_motion_command_age_ms=None,
        system_flags=(
            SystemFlags.WATCHDOG_ARMED
            | SystemFlags.GRIPPER_OUTPUT_AVAILABLE
        ),
        stop_reason=CarStopReason.STARTUP,
        servo_left_target_cdeg=9_000,
        servo_right_target_cdeg=9_000,
    )


def test_sequence_health_handles_wrap_missing_duplicate_and_regression() -> None:
    health = SequenceHealth()
    for sequence in (65_535, 0, 2, 2, 1):
        health.observe(sequence)

    assert health.missing == 1
    assert health.duplicates == 1
    assert health.regressions == 1


def test_message_formatting_exposes_units_flags_and_reply_result() -> None:
    odom_line = format_message(odometry(7))
    status_line = format_message(system_status(8))
    reply_line = format_message(
        CarCommandReply(
            uart_sequence=9,
            received_timestamp_ns=1_000_000_009,
            command_sequence=4,
            command_type=MessageType.QUERY_STATUS,
            result=CommandResult.ACCEPTED,
        )
    )

    assert "enc=(-12,34)" in odom_line
    assert "gyro_rad_s=(0.001000,-0.002000,0.300000)" in odom_line
    assert "temp_c=36.25" in odom_line
    assert "imu_valid" in odom_line
    assert "reason=startup" in status_line
    assert "watchdog_armed" in status_line
    assert "command=query_status result=accepted" in reply_line


def test_monitor_stats_counts_types_errors_rates_and_sequence_gaps() -> None:
    stats = MonitorStats(started_timestamp_ns=1_000_000_000)
    stats.observe(odometry(10))
    stats.observe(odometry(12))
    stats.observe(system_status(20))
    stats.observe(
        CarCommandReply(
            uart_sequence=4,
            received_timestamp_ns=1_500_000_000,
            command_sequence=0,
            command_type=MessageType.QUERY_STATUS,
            result=CommandResult.ACCEPTED,
        )
    )

    summary = format_summary(
        stats,
        now_ns=3_000_000_000,
        received_frames=4,
        discarded_cobs_frames=1,
    )

    assert "odom=2(1.0Hz)" in summary
    assert "status=1(0.5Hz)" in summary
    assert "reply=1" in summary
    assert "cobs_drop=1" in summary
    assert "odom_missing=1" in summary
