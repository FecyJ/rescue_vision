from __future__ import annotations

import json

import pytest

from rescue_vision.localization import OdometryCalibration
from rescue_vision.motion import (
    OdometryImu,
    D2TelemetryLogger,
    SensorFlags,
)


_ENCODERS_VALID = (
    SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
)


def odometry(
    *,
    uart_sequence: int,
    telemetry_sequence: int,
    sample_timestamp_us: int,
    left_encoder_count: int,
    right_encoder_count: int,
    sensor_flags: SensorFlags = _ENCODERS_VALID,
) -> OdometryImu:
    return OdometryImu(
        uart_sequence=uart_sequence,
        received_timestamp_ns=sample_timestamp_us * 1_000,
        telemetry_sequence=telemetry_sequence,
        sample_timestamp_us=sample_timestamp_us,
        left_encoder_count=left_encoder_count,
        right_encoder_count=right_encoder_count,
        gyro_x_urad_s=1_000,
        gyro_y_urad_s=-2_000,
        gyro_z_urad_s=3_000,
        accel_x_mm_s2=10,
        accel_y_mm_s2=20,
        accel_z_mm_s2=9_807,
        imu_temperature_cdeg=3_600,
        sensor_flags=sensor_flags,
    )


def read_events(path) -> list[dict[str, object]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]


def test_d2_logger_records_every_active_sample_and_encoder_speed(
    tmp_path,
) -> None:
    path = tmp_path / "d2.jsonl"
    logger = D2TelemetryLogger(
        path,
        OdometryCalibration(1_000, 100.0, 200.0),
    )
    logger.start(timestamp_ns=1)
    logger.set_process_start_timestamp_ns(10_000_000)

    logger.record_odometry(
        odometry(
            uart_sequence=1,
            telemetry_sequence=10,
            sample_timestamp_us=1_000,
            left_encoder_count=0,
            right_encoder_count=0,
        ),
        active=False,
        state="transport_forward",
        route_phase="forward_d2_line",
    )
    logger.begin_phase(
        timestamp_ns=2,
        state="transport_release",
        route_phase="stopping_before_d2_opening",
        reason="safe_zone_d2_coordinate_threshold_reached_wait_before_opening",
    )
    logger.record_odometry(
        odometry(
            uart_sequence=2,
            telemetry_sequence=11,
            sample_timestamp_us=11_000,
            left_encoder_count=10,
            right_encoder_count=20,
        ),
        active=True,
        state="transport_align_red_zone",
        route_phase="align_y_at_d2",
        target_wheel_speeds_m_s=(0.1, 0.2),
        commanded_wheel_speeds_m_s=(0.08, 0.18),
    )
    logger.end_phase(
        timestamp_ns=3,
        state="return_backup",
        route_phase="stopping_before_exit",
        reason="safe_zone_reached_transport_endpoint_wait_before_opening",
    )
    logger.stop(timestamp_ns=4)

    events = read_events(path)
    assert [event["event_type"] for event in events] == [
        "stream_started",
        "process_started",
        "phase_started",
        "odometry_imu",
        "phase_finished",
        "stream_finished",
    ]
    assert events[1]["process_timestamp_ms"] == pytest.approx(0.0)
    sample = events[3]
    assert sample["process_timestamp_ms"] == pytest.approx(1.0)
    assert sample["sample_dt_us"] == 10_000
    assert sample["left_encoder_delta_count"] == 10
    assert sample["right_encoder_delta_count"] == 20
    assert sample["encoder_speed_valid"] is True
    assert sample["left_encoder_speed_m_s"] == pytest.approx(
        2.0 * 3.141592653589793 * 100.0 / 1_000.0
    )
    assert sample["right_encoder_speed_m_s"] == pytest.approx(
        20.0 * 2.0 * 3.141592653589793 * 200.0 / 1_000.0 / 10.0
    )
    assert sample["gyro_z_urad_s"] == 3_000
    assert sample["accel_z_mm_s2"] == 9_807
    assert sample["target_left_wheel_speed_m_s"] == pytest.approx(0.1)
    assert events[4]["reason"] == "safe_zone_reached_transport_endpoint_wait_before_opening"


def test_d2_logger_does_not_derive_speed_from_overrun_sample(tmp_path) -> None:
    path = tmp_path / "d2_overrun.jsonl"
    logger = D2TelemetryLogger(
        path,
        OdometryCalibration(1_000, 33.0, 33.0),
    )
    logger.start(timestamp_ns=1)
    logger.set_process_start_timestamp_ns(10_000_000)
    logger.begin_phase(
        timestamp_ns=2,
        state="transport_release",
        route_phase="stopping_before_d2_opening",
        reason="safe_zone_d2_reached_wait_before_opening",
    )
    logger.record_odometry(
        odometry(
            uart_sequence=1,
            telemetry_sequence=1,
            sample_timestamp_us=1_000,
            left_encoder_count=0,
            right_encoder_count=0,
        ),
        active=True,
        state="transport_release",
        route_phase="stopping_before_d2_opening",
    )
    logger.record_odometry(
        odometry(
            uart_sequence=2,
            telemetry_sequence=2,
            sample_timestamp_us=11_000,
            left_encoder_count=10,
            right_encoder_count=10,
            sensor_flags=_ENCODERS_VALID | SensorFlags.SAMPLE_OVERRUN,
        ),
        active=True,
        state="transport_align_red_zone",
        route_phase="align_y_at_d2",
    )
    logger.stop(timestamp_ns=3)

    sample = [
        event for event in read_events(path) if event["event_type"] == "odometry_imu"
    ][1]
    assert sample["encoder_speed_valid"] is False
    assert sample["encoder_speed_invalid_reason"] == "sample_overrun"
    assert sample["left_encoder_speed_m_s"] is None
