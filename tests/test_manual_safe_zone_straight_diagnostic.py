from __future__ import annotations

import pytest

from manual_tests.safe_zone_straight_diagnostic import (
    _format_odometry_diagnostic,
)
from rescue_vision.motion import OdometryImu, SensorFlags


def _odometry(sequence: int, sample_us: int, left: int, right: int) -> OdometryImu:
    return OdometryImu(
        uart_sequence=sequence,
        received_timestamp_ns=sample_us * 1000,
        telemetry_sequence=sequence,
        sample_timestamp_us=sample_us,
        left_encoder_count=left,
        right_encoder_count=right,
        gyro_x_urad_s=1_000,
        gyro_y_urad_s=-2_000,
        gyro_z_urad_s=300_000,
        accel_x_mm_s2=10,
        accel_y_mm_s2=-20,
        accel_z_mm_s2=9_807,
        imu_temperature_cdeg=2_500,
        sensor_flags=(
            SensorFlags.IMU_VALID
            | SensorFlags.LEFT_ENCODER_VALID
            | SensorFlags.RIGHT_ENCODER_VALID
        ),
    )


def test_format_odometry_diagnostic_includes_sample_and_host_timing() -> None:
    previous = _odometry(10, 1_000_000, 100, 200)
    current = _odometry(11, 1_010_000, 103, 204)

    output = _format_odometry_diagnostic(current, previous=previous)

    assert "seq=11" in output
    assert "seq_delta=1" in output
    assert "sample_dt_ms=10.000" in output
    assert "host_dt_ms=10.000" in output
    assert "enc_delta=(3,4)" in output
    assert "gyro_xyz_rad_s=(+0.001000,-0.002000,+0.300000)" in output
    assert "flags=imu_valid|left_encoder_valid|right_encoder_valid" in output


def test_format_odometry_diagnostic_handles_baseline_without_previous() -> None:
    output = _format_odometry_diagnostic(
        _odometry(1, 100, 0, 0),
        previous=None,
    )

    assert "seq_delta=none" in output
    assert "sample_dt_ms=none" in output
    assert "host_dt_ms=none" in output
    assert "enc_delta=none" in output
