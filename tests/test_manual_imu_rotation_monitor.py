from __future__ import annotations

import pytest

from manual_tests.imu_rotation_monitor import (
    _format_odometry,
    _requested_angular_velocity,
)
from rescue_vision.motion import OdometryImu, SensorFlags


def test_requested_angular_velocity_uses_left_positive_convention() -> None:
    assert _requested_angular_velocity("left", 0.15) == pytest.approx(0.15)
    assert _requested_angular_velocity("right", 0.15) == pytest.approx(-0.15)


def test_requested_angular_velocity_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="direction"):
        _requested_angular_velocity("forward", 0.15)
    with pytest.raises(ValueError, match="positive"):
        _requested_angular_velocity("left", 0.0)


def test_format_odometry_prints_raw_gyro_z_and_flags() -> None:
    message = OdometryImu(
        uart_sequence=1,
        received_timestamp_ns=2,
        telemetry_sequence=3,
        sample_timestamp_us=4,
        left_encoder_count=5,
        right_encoder_count=6,
        gyro_x_urad_s=1_000,
        gyro_y_urad_s=-2_000,
        gyro_z_urad_s=300_000,
        accel_x_mm_s2=0,
        accel_y_mm_s2=0,
        accel_z_mm_s2=9_807,
        imu_temperature_cdeg=2_500,
        sensor_flags=SensorFlags.IMU_VALID,
    )

    output = _format_odometry(message)
    assert "sample_us=4" in output
    assert "gyro_z=+0.300000 rad/s" in output
    assert "gyro_xyz=(+0.001000,-0.002000,+0.300000) rad/s" in output
    assert "flags=imu_valid" in output
