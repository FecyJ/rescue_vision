from __future__ import annotations

import pytest

from manual_tests.imu_static_calibration import (
    RotationObservation,
    RotationPhase,
    StaticImuCalibration,
    analyze_rotation_phase,
    combine_rotation_reports,
    _matrix_vector,
    sensor_to_robot_rotation_from_gravity,
)
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
    rotation = recommended["localization"]["fusion"]["imu_calibration"][
        "sensor_to_robot_rotation"
    ]
    mapped = _matrix_vector(
        tuple(tuple(float(value) for value in row) for row in rotation),
        (10.0, -20.0, 9_807.0),
    )
    assert mapped[0] == pytest.approx(0.0, abs=1e-5)
    assert mapped[1] == pytest.approx(0.0, abs=1e-5)
    assert mapped[2] == pytest.approx(
        (10.0**2 + 20.0**2 + 9_807.0**2) ** 0.5,
        rel=1e-9,
    )


def test_gravity_rotation_rejects_zero_vector_and_aligns_tilt() -> None:
    with pytest.raises(ValueError, match="norm"):
        sensor_to_robot_rotation_from_gravity((0.0, 0.0, 0.0))

    rotation = sensor_to_robot_rotation_from_gravity((0.0, -1.0, 1.0))
    mapped = _matrix_vector(rotation, (0.0, -1.0, 1.0))
    assert mapped[0] == pytest.approx(0.0, abs=1e-12)
    assert mapped[1] == pytest.approx(0.0, abs=1e-12)
    assert mapped[2] == pytest.approx(2.0**0.5)


def test_active_rotation_compares_integrated_gyro_with_encoder_angle() -> None:
    phase = RotationPhase(
        direction=1,
        baseline_left_count=0,
        baseline_right_count=0,
        baseline_sample_timestamp_us=0,
        encoder_counts_per_revolution=100,
        left_wheel_radius_m=0.1,
        right_wheel_radius_m=0.1,
        wheel_track_m=0.5,
        observations=[
            RotationObservation(100_000, 0.0, (0.0, 0.0, 0.5)),
            RotationObservation(200_000, 0.05, (0.0, 0.0, 0.5)),
        ],
    )

    result = analyze_rotation_phase(
        phase,
        sensor_to_robot_rotation=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        ),
        gyro_bias_rad_s_sensor_frame=(0.0, 0.0, 0.0),
    )

    assert result["gyro_integrated_yaw_rad"] == pytest.approx(0.05)
    assert result["encoder_yaw_rad"] == pytest.approx(0.05)
    assert result["gyro_to_encoder_scale"] == pytest.approx(1.0)
    assert result["gyro_z_sign"] == 1
    combined = combine_rotation_reports([result])
    assert combined["gyro_z_sign"] == 1
    assert combined["sign_consistent"] is True


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
