from __future__ import annotations

from manual_tests.motion_sample_overrun import OverrunProbeStats
from rescue_vision.motion import OdometryImu, SensorFlags


def odometry(*, sequence: int, flags: SensorFlags) -> OdometryImu:
    return OdometryImu(
        uart_sequence=sequence,
        received_timestamp_ns=sequence * 1_000_000,
        telemetry_sequence=sequence,
        sample_timestamp_us=sequence * 1_000,
        left_encoder_count=sequence * 10,
        right_encoder_count=sequence * 9,
        gyro_x_urad_s=0,
        gyro_y_urad_s=0,
        gyro_z_urad_s=0,
        accel_x_mm_s2=0,
        accel_y_mm_s2=0,
        accel_z_mm_s2=9807,
        imu_temperature_cdeg=2500,
        sensor_flags=flags,
    )


def test_overrun_probe_records_first_sample_overrun() -> None:
    valid = (
        SensorFlags.IMU_VALID
        | SensorFlags.LEFT_ENCODER_VALID
        | SensorFlags.RIGHT_ENCODER_VALID
    )
    stats = OverrunProbeStats(started_timestamp_ns=1)

    assert not stats.observe(odometry(sequence=1, flags=valid))
    assert stats.observe(
        odometry(sequence=2, flags=valid | SensorFlags.SAMPLE_OVERRUN)
    )
    assert stats.observe(
        odometry(sequence=3, flags=valid | SensorFlags.SAMPLE_OVERRUN)
    )

    assert stats.odometry_count == 3
    assert stats.sample_overrun_count == 2
    assert stats.first_sample_overrun is not None
    assert stats.first_sample_overrun.telemetry_sequence == 2
    assert "sample_overrun=2" in stats.summary(duration_s=0.1)
