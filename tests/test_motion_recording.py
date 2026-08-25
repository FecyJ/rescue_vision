from __future__ import annotations

import json

import pytest

from rescue_vision.motion import (
    CarCommandReply,
    CarStopReason,
    CarSystemStatus,
    CommandResult,
    ExecutedRemoteGripper,
    ExecutedRemoteMotion,
    ManualMotionLogWriter,
    RemoteGripperResult,
    RemoteMotionResult,
    MessageType,
    OdometryImu,
    SensorFlags,
    SystemFlags,
    inspect_manual_motion_log,
)


def test_manual_motion_log_round_trip_preserves_monotonic_time_sources(
    tmp_path,
) -> None:
    path = tmp_path / "motion.jsonl"
    writer = ManualMotionLogWriter(path)
    writer.start(timestamp_ns=100)
    writer.record_motion(
        ExecutedRemoteMotion(
            command_id="drive-1",
            result=RemoteMotionResult.APPLIED,
            received_timestamp_ns=110,
            deadline_timestamp_ns=210,
            deadman_enabled=True,
            linear_velocity_m_s=0.1,
            angular_velocity_rad_s=-0.2,
        )
    )
    writer.record_gripper(
        ExecutedRemoteGripper(
            command_id="grip-1",
            result=RemoteGripperResult.APPLIED,
            received_timestamp_ns=115,
            deadline_timestamp_ns=215,
            open_pressed=False,
            close_pressed=True,
        )
    )
    writer.record_car_message(
        OdometryImu(
            uart_sequence=3,
            received_timestamp_ns=120,
            telemetry_sequence=7,
            sample_timestamp_us=50_000,
            left_encoder_count=100,
            right_encoder_count=101,
            gyro_x_urad_s=1,
            gyro_y_urad_s=2,
            gyro_z_urad_s=3,
            accel_x_mm_s2=4,
            accel_y_mm_s2=5,
            accel_z_mm_s2=9807,
            imu_temperature_cdeg=3600,
            sensor_flags=(
                SensorFlags.IMU_VALID
                | SensorFlags.LEFT_ENCODER_VALID
                | SensorFlags.RIGHT_ENCODER_VALID
            ),
        )
    )
    writer.record_car_message(
        CarCommandReply(
            4,
            130,
            9,
            MessageType.SET_WHEEL_SPEED,
            CommandResult.ACCEPTED,
        )
    )
    writer.record_car_message(
        CarSystemStatus(
            uart_sequence=5,
            received_timestamp_ns=135,
            status_sequence=8,
            controller_timestamp_us=65_000,
            watchdog_timeout_ms=300,
            last_motion_command_age_ms=15,
            system_flags=SystemFlags.PROTOCOL_READY,
            stop_reason=CarStopReason.RUNNING,
            servo_left_target_cdeg=9000,
            servo_right_target_cdeg=9000,
        )
    )
    writer.record_timeout(command_id="drive-1", timestamp_ns=210)
    writer.record_gripper_timeout(
        command_id="grip-1",
        timestamp_ns=216,
    )
    writer.record_safety_stop(
        reason="application_shutdown",
        timestamp_ns=220,
    )
    writer.stop(timestamp_ns=230)

    report = inspect_manual_motion_log(path)

    assert report["event_count"] == 10
    assert report["first_timestamp_ns"] == 100
    assert report["last_timestamp_ns"] == 230
    assert report["event_counts"] == {
        "command_reply": 1,
        "gripper_command": 1,
        "gripper_timeout": 1,
        "motion_command": 1,
        "motion_timeout": 1,
        "safety_stop": 1,
        "system_status": 1,
        "stream_finished": 1,
        "stream_started": 1,
        "odometry_imu": 1,
    }
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[1]["timestamp_ns"] == 110
    assert events[2]["timestamp_ns"] == 115
    assert events[3]["timestamp_ns"] == 120
    assert events[5]["stop_reason"] == "running"
    assert events[3]["sample_timestamp_us"] == 50_000
    assert events[7]["timestamp_ns"] == 216


def test_manual_motion_log_rejects_sequence_and_truncation(tmp_path) -> None:
    path = tmp_path / "motion.jsonl"
    writer = ManualMotionLogWriter(path)
    writer.start(timestamp_ns=1)
    writer.stop(timestamp_ns=2)
    events = path.read_text(encoding="utf-8").splitlines()

    first = json.loads(events[0])
    first["log_sequence"] = 4
    path.write_text(
        json.dumps(first) + "\n" + events[1] + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="log_sequence"):
        inspect_manual_motion_log(path)

    path.write_text(events[0] + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="stream_finished"):
        inspect_manual_motion_log(path)
