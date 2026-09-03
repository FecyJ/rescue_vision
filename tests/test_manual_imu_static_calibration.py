from __future__ import annotations

import pytest

from manual_tests.imu_static_calibration import StaticImuCalibration
from rescue_vision.motion import OdometryImu, SensorFlags


def odometry(
    sequence: int,
    *,
    gyro_z_urad_s: int = 0,
    left_encoder_count: int = 10,
    right_encoder_count: int = 20,
    flags: SensorFlags = (
        SensorFlags.IMU_VALID
        | SensorFlags.LEFT_ENCODER_VALID
        | SensorFlags.RIGHT_ENCODER_VALID
    ),
) -> OdometryImu:
    return OdometryImu(
        uart_sequence=sequence,
        received_timestamp_ns=sequence,
        telemetry_sequence=sequence,
        sample_timestamp_us=10_000 * sequence,
        left_encoder_count=left_encoder_count,
        right_encoder_count=right_encoder_count,
        gyro_x_urad_s=1_000,
        gyro_y_urad_s=-2_000,
        gyro_z_urad_s=gyro_z_urad_s,
        accel_x_mm_s2=10,
        accel_y_mm_s2=-20,
        accel_z_mm_s2=9_807,
        imu_temperature_cdeg=2_500,
        sensor_flags=flags,
    )


def test_static_calibration_accumulates_bias_noise_and_recommendation() -> None:
    calibration = StaticImuCalibration()

    assert calibration.observe(odometry(1, gyro_z_urad_s=100_000))
    assert calibration.observe(odometry(2, gyro_z_urad_s=300_000))

    report = calibration.report()

    assert report["sample_count"] == 2
    assert report["gyro_bias_rad_s_sensor_frame"] == pytest.approx(
        [0.001, -0.002, 0.2]
    )
    assert report["gyro_noise_std_rad_s_sensor_frame"] == pytest.approx(
        [0.0, 0.0, 0.1]
    )
    recommended = report["recommended_config"]
    assert recommended["localization"]["fusion"]["imu_calibration"][
        "reference_temperature_c"
    ] == pytest.approx(25.0)


def test_static_calibration_rejects_bad_quality_and_motion() -> None:
    calibration = StaticImuCalibration()

    assert not calibration.observe(
        odometry(1, flags=SensorFlags.IMU_VALID)
    )
    assert not calibration.observe(
        odometry(
            2,
            left_encoder_count=11,
            flags=(
                SensorFlags.IMU_VALID
                | SensorFlags.LEFT_ENCODER_VALID
                | SensorFlags.RIGHT_ENCODER_VALID
                | SensorFlags.SAMPLE_OVERRUN
            ),
        )
    )
    assert not calibration.observe(odometry(3, left_encoder_count=12))

    assert calibration.accepted_sample_count == 0
    assert calibration.rejected_sample_count == 3
    assert calibration.rejected_reasons == {
        "encoder_invalid": 1,
        "sample_overrun": 1,
        "vehicle_moved": 1,
    }


def test_static_calibration_requires_enough_samples() -> None:
    with pytest.raises(ValueError, match="at least two"):
        StaticImuCalibration().report()

    with pytest.raises(ValueError, match="non-negative"):
        StaticImuCalibration(max_encoder_delta_count=-1)
