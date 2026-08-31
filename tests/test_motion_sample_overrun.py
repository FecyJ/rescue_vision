from __future__ import annotations

import pytest

from manual_tests.motion_sample_overrun import (
    OverrunProbeStats,
    require_motion_status_healthy,
)
from rescue_vision.motion import (
    CarStopReason,
    CarSystemStatus,
    OdometryImu,
    SensorFlags,
    SystemFlags,
)


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


def system_status(*, flags: SystemFlags) -> CarSystemStatus:
    return CarSystemStatus(
        uart_sequence=1,
        received_timestamp_ns=1_000_000,
        status_sequence=1,
        controller_timestamp_us=1_000,
        watchdog_timeout_ms=300,
        last_motion_command_age_ms=10,
        system_flags=flags,
        stop_reason=CarStopReason.RUNNING,
        servo_left_target_cdeg=9000,
        servo_right_target_cdeg=9000,
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


def test_overrun_probe_accepts_healthy_motion_status() -> None:
    require_motion_status_healthy(
        system_status(flags=SystemFlags.PROTOCOL_READY)
    )


@pytest.mark.parametrize(
    "unhealthy_flag,expected",
    [
        (SystemFlags.REPLY_QUEUE_FULL, "reply_queue_full=True"),
        (SystemFlags.TX_DEGRADED, "tx_degraded=True"),
    ],
)
def test_overrun_probe_rejects_sticky_uart_health_flags(
    unhealthy_flag: SystemFlags,
    expected: str,
) -> None:
    status = system_status(
        flags=SystemFlags.PROTOCOL_READY | unhealthy_flag
    )

    with pytest.raises(RuntimeError, match=expected) as caught:
        require_motion_status_healthy(status)

    assert "cannot clear the sticky queue/TX flags" in str(caught.value)
    assert "reset or power-cycle the STM32" in str(caught.value)


def test_overrun_probe_does_not_gate_sticky_rx_health_flag() -> None:
    require_motion_status_healthy(
        system_status(
            flags=SystemFlags.PROTOCOL_READY | SystemFlags.RX_DEGRADED
        )
    )


def test_overrun_probe_rejects_protocol_not_ready() -> None:
    with pytest.raises(RuntimeError, match="protocol_ready=False"):
        require_motion_status_healthy(system_status(flags=SystemFlags(0)))
