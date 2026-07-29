from __future__ import annotations

import json

import pytest

from rescue_vision.motion import (
    CarCommandReply,
    CarSafetyStatus,
    CarStopReason,
    CarTelemetry,
    ExecutedRemoteGripper,
    ExecutedRemoteMotion,
    ManualMotionLogWriter,
    RemoteGripperResult,
    RemoteMotionResult,
    UnknownCarMessage,
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
            left_angle_deg=27.0,
            right_angle_deg=167.0,
        )
    )
    writer.record_car_message(
        CarTelemetry(
            uart_sequence=3,
            received_timestamp_ns=120,
            controller_timestamp_ms=50,
            actual_left_m_s=0.11,
            actual_right_m_s=0.09,
            target_left_m_s=0.12,
            target_right_m_s=0.08,
            servo_left_deg=90,
            servo_right_deg=90,
        )
    )
    writer.record_car_message(
        CarCommandReply(4, 130, True, "m=0.12,0.08")
    )
    writer.record_car_message(
        CarSafetyStatus(
            uart_sequence=5,
            received_timestamp_ns=135,
            controller_timestamp_ms=65,
            watchdog_timeout_ms=300,
            watchdog_armed=True,
            emergency_stop_latched=False,
            last_motion_command_age_ms=15,
            stop_reason=CarStopReason.RUNNING,
        )
    )
    writer.record_car_message(UnknownCarMessage(6, 140, b"imu,1,2,3"))
    writer.record_timeout(command_id="drive-1", timestamp_ns=210)
    writer.record_safety_stop(
        reason="application_shutdown",
        timestamp_ns=220,
    )
    writer.stop(timestamp_ns=230)

    report = inspect_manual_motion_log(path)

    assert report["schema_version"] == 3
    assert report["event_count"] == 10
    assert report["first_timestamp_ns"] == 100
    assert report["last_timestamp_ns"] == 230
    assert report["event_counts"] == {
        "command_reply": 1,
        "gripper_command": 1,
        "motion_command": 1,
        "motion_timeout": 1,
        "safety_stop": 1,
        "safety_status": 1,
        "stream_finished": 1,
        "stream_started": 1,
        "unknown_uart": 1,
        "wheel_telemetry": 1,
    }
    events = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert events[1]["timestamp_ns"] == 110
    assert events[2]["timestamp_ns"] == 115
    assert events[3]["timestamp_ns"] == 120
    assert events[5]["stop_reason"] == "running"
    assert events[6]["payload_hex"] == b"imu,1,2,3".hex()


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
