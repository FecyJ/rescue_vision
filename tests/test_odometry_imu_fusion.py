from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization import (
    CenterCrossPoseCandidate,
    CenterCrossPoseObservation,
    CenterCrossSelectionSource,
    FieldPose2D,
    FusionConfig,
    FusionQuality,
    ImuFrameCalibration,
    OdometryCalibration,
    OdometryImuFusion,
)
from rescue_vision.motion import OdometryImu, SensorFlags


FLAGS = (
    SensorFlags.IMU_VALID
    | SensorFlags.LEFT_ENCODER_VALID
    | SensorFlags.RIGHT_ENCODER_VALID
)


def config(**changes: object) -> FusionConfig:
    values: dict[str, object] = {
        "enabled": True,
        "initial_pose": FieldPose2D(FieldPoint(100.0, 200.0), 0.0),
        "initial_position_uncertainty_mm": 20.0,
        "initial_heading_uncertainty_rad": 0.1,
        "initial_confidence": 0.8,
        "encoder_distance_noise_fraction": 0.01,
        "encoder_heading_noise_std_rad": 0.02,
        "gyro_noise_std_rad_s": 0.03,
        "gyro_bias_random_walk_std_rad_s_per_sqrt_s": 0.001,
        "stationary_gyro_noise_std_rad_s": 0.01,
        "stationary_encoder_delta_count": 0,
        "allow_wheel_only": True,
        "wheel_only_covariance_scale": 4.0,
        "dropped_sample_covariance_scale": 3.0,
        "max_sample_interval_ms": 100.0,
        "max_telemetry_age_ms": 200.0,
        "max_encoder_speed_mm_s": 50_000.0,
        "max_visual_alignment_error_ms": 30.0,
        "visual_innovation_gate": 25.0,
        "history_duration_ms": 1000.0,
        "max_tilt_deg": 25.0,
        "impact_accel_threshold_mm_s2": 4000.0,
    }
    values.update(changes)
    return FusionConfig(**values)  # type: ignore[arg-type]


def fusion(**changes: object) -> OdometryImuFusion:
    return OdometryImuFusion(
        config(**changes),
        OdometryCalibration(
            1000,
            100.0 / (2.0 * math.pi),
            100.0 / (2.0 * math.pi),
        ),
        wheel_track_m=0.2,
    )


@pytest.mark.parametrize("value", [-1, 2, True])
def test_interpolated_overrun_budget_is_limited_to_one_sample(value: object) -> None:
    with pytest.raises(ValueError, match="max_interpolated_overrun_samples"):
        fusion(max_interpolated_overrun_samples=value)


def odom(
    sequence: int,
    timestamp_us: int,
    left: int,
    right: int,
    *,
    gyro_x_rad_s: float = 0.0,
    gyro_y_rad_s: float = 0.0,
    gyro_z_rad_s: float = 0.0,
    accel_x_mm_s2: int = 0,
    accel_y_mm_s2: int = 0,
    accel_z_mm_s2: int = 9807,
    flags: SensorFlags = FLAGS,
    received_offset_ns: int = 1_000_000_000,
) -> OdometryImu:
    return OdometryImu(
        uart_sequence=sequence,
        received_timestamp_ns=timestamp_us * 1000 + received_offset_ns,
        telemetry_sequence=sequence & 0xFFFF,
        sample_timestamp_us=timestamp_us,
        left_encoder_count=left,
        right_encoder_count=right,
        gyro_x_urad_s=round(gyro_x_rad_s * 1_000_000),
        gyro_y_urad_s=round(gyro_y_rad_s * 1_000_000),
        gyro_z_urad_s=round(gyro_z_rad_s * 1_000_000),
        accel_x_mm_s2=accel_x_mm_s2,
        accel_y_mm_s2=accel_y_mm_s2,
        accel_z_mm_s2=accel_z_mm_s2,
        imu_temperature_cdeg=2500,
        sensor_flags=flags,
    )


def visual(timestamp_ns: int, pose: FieldPose2D, confidence: float = 0.9):
    candidates = tuple(
        CenterCrossPoseCandidate(
            pose if index == 0 else FieldPose2D(pose.position, index * math.pi / 2),
            index,
            10.0,
            0.04,
        )
        for index in range(4)
    )
    return CenterCrossPoseObservation(
        frame_sequence=1,
        capture_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns + 1,
        candidates=candidates,
        terminals=(),
        selected_pose=pose,
        selection_source=CenterCrossSelectionSource.RED_SAFE_ZONE,
        confidence=confidence,
        quality=frozenset(),
    )


def test_first_sample_initializes_and_straight_counts_propagate() -> None:
    estimator = fusion()
    first = odom(0, 10_000, 0, 0)
    second = odom(1, 20_000, 1000, 1000)

    initial = estimator.submit_odometry(first)
    moved = estimator.submit_odometry(second)

    assert initial.pose == FieldPose2D(FieldPoint(100.0, 200.0), 0.0)
    assert moved.pose is not None
    assert moved.pose.position.x == pytest.approx(200.0, abs=0.1)
    assert moved.pose.position.y == pytest.approx(200.0, abs=0.1)


def test_wheel_only_turn_and_sequence_wrap_are_supported() -> None:
    estimator = fusion()
    wheel_flags = FLAGS & ~SensorFlags.IMU_VALID
    estimator.submit_odometry(odom(65535, 10_000, 0, 0, flags=wheel_flags))
    result = estimator.submit_odometry(
        odom(0, 20_000, -1000, 1000, flags=wheel_flags)
    )

    assert result.pose is not None
    assert result.pose.heading_rad == pytest.approx(1.0, abs=1e-6)
    assert FusionQuality.WHEEL_ONLY in result.quality


def test_arc_motion_and_heading_wrap_follow_field_axes() -> None:
    estimator = fusion(
        initial_pose=FieldPose2D(FieldPoint(0.0, 0.0), 3.1),
    )
    wheel_flags = FLAGS & ~SensorFlags.IMU_VALID
    estimator.submit_odometry(odom(0, 10_000, 0, 0, flags=wheel_flags))
    result = estimator.submit_odometry(
        odom(1, 20_000, 500, 1000, flags=wheel_flags)
    )

    assert result.pose is not None
    assert result.pose.heading_rad == pytest.approx(
        (3.1 + 0.25 + math.pi) % (2 * math.pi) - math.pi,
        abs=1e-6,
    )
    assert result.pose.position.x < 0.0
    assert result.pose.position.y < 0.0


def test_wheel_radius_difference_and_gyro_bias_are_applied() -> None:
    estimator = OdometryImuFusion(
        config(
            initial_pose=FieldPose2D(FieldPoint(0.0, 0.0), 0.0),
            imu_frame_calibration=ImuFrameCalibration(
                gyro_bias_rad_s=(0.0, 0.0, 0.1)
            ),
        ),
        OdometryCalibration(
            1000,
            100.0 / (2.0 * math.pi),
            110.0 / (2.0 * math.pi),
        ),
        wheel_track_m=0.2,
    )
    estimator.submit_odometry(odom(0, 10_000, 0, 0, gyro_z_rad_s=0.1))
    result = estimator.submit_odometry(
        odom(1, 20_000, 1000, 1000, gyro_z_rad_s=0.1)
    )

    assert result.pose is not None
    # The calibrated gyro bias removes stationary yaw; unequal effective
    # radii still contribute a small encoder-weighted turn.
    assert 0.0 < result.pose.heading_rad < 0.05
    assert result.position_uncertainty_mm is not None
    assert result.position_uncertainty_mm > 20.0


def test_negative_raw_left_turn_is_converted_to_positive_canonical_yaw() -> None:
    estimator = OdometryImuFusion(
        config(
            initial_pose=FieldPose2D(FieldPoint(0.0, 0.0), 0.0),
            imu_frame_calibration=ImuFrameCalibration(
                gyro_bias_rad_s=(0.0, 0.0, 0.1)
            ),
        ),
        OdometryCalibration(
            1000,
            100.0 / (2.0 * math.pi),
            100.0 / (2.0 * math.pi),
            -1,
        ),
        wheel_track_m=0.2,
    )
    estimator.submit_odometry(odom(0, 10_000, 0, 0, gyro_z_rad_s=0.1))
    result = estimator.submit_odometry(
        odom(1, 20_000, 1, 1, gyro_z_rad_s=-0.9)
    )

    assert result.pose is not None
    assert result.pose.heading_rad > 0.0


def test_sensor_frame_rotation_is_applied_before_heading_and_tilt_checks() -> None:
    rotation = ImuFrameCalibration(
        sensor_to_robot_rotation=(
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        )
    )
    estimator = fusion(
        imu_frame_calibration=rotation,
        max_tilt_deg=5.0,
    )
    estimator.submit_odometry(
        odom(
            0,
            10_000,
            0,
            0,
            gyro_x_rad_s=1.0,
            accel_x_mm_s2=9807,
            accel_z_mm_s2=0,
        )
    )
    result = estimator.submit_odometry(
        odom(
            1,
            20_000,
            0,
            0,
            gyro_x_rad_s=1.0,
            accel_x_mm_s2=9807,
            accel_z_mm_s2=0,
        )
    )

    assert result.pose is not None
    assert result.pose.heading_rad > 0.0
    assert FusionQuality.TILT_DETECTED not in result.quality
    assert FusionQuality.IMPACT_DETECTED not in result.quality


def test_temperature_bias_and_cross_axis_scale_are_applied_before_rotation() -> None:
    calibration = ImuFrameCalibration(
        reference_temperature_c=25.0,
        gyro_bias_rad_s=(1.0, 0.0, 0.0),
        gyro_bias_temperature_coefficient_rad_s_per_c=(0.1, 0.0, 0.0),
        gyro_cross_axis_scale=(
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.5, 0.25, 1.0),
        ),
        accel_bias_mm_s2=(100.0, 0.0, 0.0),
        accel_bias_temperature_coefficient_mm_s2_per_c=(10.0, 0.0, 0.0),
    )
    estimator = fusion(
        imu_frame_calibration=calibration,
        max_tilt_deg=5.0,
    )
    first = odom(
        0,
        10_000,
        0,
        0,
        gyro_x_rad_s=3.0,
        gyro_y_rad_s=2.0,
        accel_x_mm_s2=200,
        accel_z_mm_s2=9807,
    )
    second = odom(
        1,
        20_000,
        0,
        0,
        gyro_x_rad_s=3.0,
        gyro_y_rad_s=2.0,
        accel_x_mm_s2=200,
        accel_z_mm_s2=9807,
    )
    first = replace(first, imu_temperature_cdeg=3500)
    second = replace(second, imu_temperature_cdeg=3500)

    estimator.submit_odometry(first)
    result = estimator.submit_odometry(second)

    assert result.pose is not None
    assert result.pose.heading_rad > 0.0
    assert FusionQuality.TILT_DETECTED not in result.quality
    assert FusionQuality.IMPACT_DETECTED not in result.quality


def test_forward_drop_is_accepted_but_reverse_sequence_clears_pose() -> None:
    estimator = fusion()
    estimator.submit_odometry(odom(10, 10_000, 0, 0))
    dropped = estimator.submit_odometry(odom(12, 30_000, 10, 10))
    assert dropped.pose is not None
    assert FusionQuality.DROPPED_TELEMETRY in dropped.quality

    reset = estimator.submit_odometry(odom(11, 40_000, 20, 20))
    assert reset.pose is None
    assert FusionQuality.CONTINUITY_LOST in reset.quality


@pytest.mark.parametrize(
    "message",
    [
        odom(1, 10_000, 1, 1),
        odom(2, 300_000, 2, 2),
        odom(2, 20_000, 100_000, 100_000),
        odom(2, 20_000, 2, 2, flags=FLAGS & ~SensorFlags.LEFT_ENCODER_VALID),
    ],
)
def test_invalid_continuity_clears_global_pose(message: OdometryImu) -> None:
    estimator = fusion()
    estimator.submit_odometry(odom(1, 10_000, 0, 0))
    result = estimator.submit_odometry(message)
    assert result.pose is None
    assert FusionQuality.CONTINUITY_LOST in result.quality


def test_single_sample_overrun_is_recovered_with_interpolated_imu() -> None:
    estimator = fusion()
    first = odom(0, 10_000, 0, 0, gyro_z_rad_s=0.0)
    estimator.submit_odometry(first)

    overrun = replace(
        odom(
            1,
            10_000,
            5,
            5,
            gyro_z_rad_s=50.0,
            flags=FLAGS | SensorFlags.SAMPLE_OVERRUN,
        ),
        received_timestamp_ns=first.received_timestamp_ns + 10_000_000,
    )
    pending = estimator.submit_odometry(overrun)
    assert pending.pose is not None
    assert FusionQuality.INTERPOLATED_IMU not in pending.quality

    recovered = estimator.submit_odometry(
        odom(2, 30_000, 10, 10, gyro_z_rad_s=1.0)
    )

    assert recovered.pose is not None
    assert recovered.pose.position.x == pytest.approx(101.0, abs=0.01)
    assert recovered.pose.heading_rad > 0.0
    assert FusionQuality.INTERPOLATED_IMU in recovered.quality


def test_consecutive_overruns_clear_continuity_after_one_recovery_budget() -> None:
    estimator = fusion()
    first = odom(0, 10_000, 0, 0)
    estimator.submit_odometry(first)
    first_overrun = replace(
        odom(
            1,
            10_000,
            5,
            5,
            flags=FLAGS | SensorFlags.SAMPLE_OVERRUN,
        ),
        received_timestamp_ns=first.received_timestamp_ns + 10_000_000,
    )
    estimator.submit_odometry(first_overrun)
    second_overrun = replace(
        odom(
            2,
            10_000,
            10,
            10,
            flags=FLAGS | SensorFlags.SAMPLE_OVERRUN,
        ),
        received_timestamp_ns=first_overrun.received_timestamp_ns + 10_000_000,
    )

    result = estimator.submit_odometry(second_overrun)

    assert result.pose is None
    assert FusionQuality.CONTINUITY_LOST in result.quality


def test_delayed_visual_update_replays_later_motion() -> None:
    estimator = fusion(visual_innovation_gate=10_000.0)
    estimator.submit_odometry(odom(0, 10_000, 0, 0))
    estimator.submit_odometry(odom(1, 20_000, 100, 100))
    estimator.submit_odometry(odom(2, 30_000, 200, 200))
    capture_ns = 1_020_000_000

    result = estimator.submit_visual(
        visual(capture_ns, FieldPose2D(FieldPoint(500.0, 600.0), 0.0))
    )
    latest = estimator.latest_estimate(1_030_000_000)

    assert result.accepted
    assert latest.pose is not None
    assert latest.pose.position.x > 120.0
    assert latest.anchor_source == "red_safe_zone"


def test_visual_outlier_rejected_and_visual_can_reinitialize_after_reset() -> None:
    estimator = fusion(visual_innovation_gate=1.0)
    estimator.submit_odometry(odom(0, 10_000, 0, 0))
    rejected = estimator.submit_visual(
        visual(1_010_000_000, FieldPose2D(FieldPoint(2000.0, 2000.0), 1.0))
    )
    assert not rejected.accepted

    estimator.reset("controller restart")
    anchored = estimator.submit_visual(
        visual(1_020_000_000, FieldPose2D(FieldPoint(10.0, 20.0), 0.3))
    )
    assert anchored.accepted
    assert estimator.latest_estimate(1_020_000_000).pose == FieldPose2D(
        FieldPoint(10.0, 20.0), 0.3
    )


def test_stale_estimate_is_cleared_and_absolute_visual_reinitializes() -> None:
    estimator = fusion(max_telemetry_age_ms=10.0, max_visual_alignment_error_ms=5.0)
    estimator.submit_odometry(odom(0, 10_000, 0, 0))
    stale = estimator.latest_estimate(1_030_000_000)
    assert stale.pose is None
    assert FusionQuality.STALE in stale.quality

    result = estimator.submit_visual(
        visual(1_030_000_000, FieldPose2D(FieldPoint(100.0, 200.0), 0.0))
    )
    assert result.accepted
    assert estimator.latest_estimate(1_030_000_000).pose is not None


def test_saturated_imu_and_abnormal_acceleration_mark_degraded_quality() -> None:
    estimator = fusion()
    estimator.submit_odometry(odom(0, 10_000, 0, 0))
    message = odom(
        1,
        20_000,
        1,
        1,
        flags=FLAGS | SensorFlags.GYRO_SATURATED | SensorFlags.ACCEL_SATURATED,
    )
    result = estimator.submit_odometry(message)

    assert result.pose is not None
    assert FusionQuality.WHEEL_ONLY in result.quality
    assert FusionQuality.IMPACT_DETECTED in result.quality


def test_visual_alignment_rejects_old_measurement_while_telemetry_is_fresh() -> None:
    estimator = fusion(max_telemetry_age_ms=100.0, max_visual_alignment_error_ms=5.0)
    estimator.submit_odometry(odom(0, 10_000, 0, 0))
    result = estimator.submit_visual(
        visual(1_030_000_000, FieldPose2D(FieldPoint(100.0, 200.0), 0.0))
    )
    assert not result.accepted


def test_configuration_rejects_invalid_values() -> None:
    with pytest.raises(ValueError, match="history_duration"):
        config(history_duration_ms=10.0, max_visual_alignment_error_ms=20.0)
    with pytest.raises(ValueError, match="encoder_counts"):
        OdometryCalibration(0, 10.0, 10.0)
    with pytest.raises(ValueError, match="gyro_z_sign"):
        OdometryCalibration(1000, 10.0, 10.0, 0)
    with pytest.raises(ValueError, match="orthonormal"):
        ImuFrameCalibration(
            sensor_to_robot_rotation=(
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 2.0),
            )
        )
    with pytest.raises(ValueError, match="invertible"):
        ImuFrameCalibration(
            gyro_cross_axis_scale=(
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
