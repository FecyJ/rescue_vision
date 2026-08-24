"""手动运动/夹爪命令、UART 遥测与停车原因记录流。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, TextIO

from rescue_vision.motion.protocol import (
    CarCommandReply,
    CarSystemStatus,
    OdometryImu,
    ParsedCarMessage,
)
from rescue_vision.motion.remote_control import (
    ExecutedRemoteGripper,
    ExecutedRemoteMotion,
)


MANUAL_MOTION_STREAM_NAME = "manual_motion"
MANUAL_MOTION_LOG_FILENAME = "motion.jsonl"

_EVENT_KEYS = {
    "stream_started": set(),
    "stream_finished": set(),
    "motion_command": {
        "command_id",
        "result",
        "deadline_timestamp_ns",
        "deadman_enabled",
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
    },
    "motion_timeout": {"command_id"},
    "gripper_command": {
        "command_id",
        "result",
        "deadline_timestamp_ns",
        "open_pressed",
        "close_pressed",
    },
    "gripper_timeout": {"command_id"},
    "odometry_imu": {
        "uart_sequence",
        "telemetry_sequence",
        "sample_timestamp_us",
        "left_encoder_count",
        "right_encoder_count",
        "gyro_x_urad_s",
        "gyro_y_urad_s",
        "gyro_z_urad_s",
        "accel_x_mm_s2",
        "accel_y_mm_s2",
        "accel_z_mm_s2",
        "imu_temperature_cdeg",
        "sensor_flags",
    },
    "system_status": {
        "uart_sequence",
        "status_sequence",
        "controller_timestamp_us",
        "watchdog_timeout_ms",
        "last_motion_command_age_ms",
        "system_flags",
        "stop_reason",
        "servo_left_target_cdeg",
        "servo_right_target_cdeg",
    },
    "command_reply": {
        "uart_sequence",
        "command_sequence",
        "command_type",
        "result",
    },
    "safety_stop": {"reason"},
}
_COMMON_KEYS = {
    "log_sequence",
    "event_type",
    "timestamp_ns",
}
_MOTION_RESULTS = {"applied", "stopped_deadman", "expired"}
_STOP_REASONS = {
    "deadman_release",
    "command_expired",
    "remote_disconnected",
    "uart_fault",
    "controller_watchdog",
    "emergency_stop",
    "camera_fault",
    "application_shutdown",
    "unknown",
}


class ManualMotionLogWriter:
    """同步写入运动事件与 100 Hz 定位遥测；每行刷新以保留故障前证据。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()
        self._stream: TextIO | None = None
        self._sequence = 0

    def start(self, *, timestamp_ns: int) -> None:
        if self._stream is not None:
            raise RuntimeError("Manual motion log is already started.")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._stream = self.path.open("x", encoding="utf-8", buffering=1)
        try:
            self._write("stream_started", timestamp_ns)
        except BaseException:
            self._stream.close()
            self._stream = None
            raise

    def record_motion(self, outcome: ExecutedRemoteMotion) -> None:
        self._write(
            "motion_command",
            outcome.received_timestamp_ns,
            command_id=outcome.command_id,
            result=outcome.result.value,
            deadline_timestamp_ns=outcome.deadline_timestamp_ns,
            deadman_enabled=outcome.deadman_enabled,
            linear_velocity_m_s=outcome.linear_velocity_m_s,
            angular_velocity_rad_s=outcome.angular_velocity_rad_s,
        )

    def record_timeout(self, *, command_id: str, timestamp_ns: int) -> None:
        self._write(
            "motion_timeout",
            timestamp_ns,
            command_id=command_id,
        )

    def record_gripper(self, outcome: ExecutedRemoteGripper) -> None:
        self._write(
            "gripper_command",
            outcome.received_timestamp_ns,
            command_id=outcome.command_id,
            result=outcome.result.value,
            deadline_timestamp_ns=outcome.deadline_timestamp_ns,
            open_pressed=outcome.open_pressed,
            close_pressed=outcome.close_pressed,
        )

    def record_gripper_timeout(
        self,
        *,
        command_id: str,
        timestamp_ns: int,
    ) -> None:
        self._write(
            "gripper_timeout",
            timestamp_ns,
            command_id=command_id,
        )

    def record_car_message(self, message: ParsedCarMessage) -> None:
        if isinstance(message, OdometryImu):
            self._write(
                "odometry_imu",
                message.received_timestamp_ns,
                uart_sequence=message.uart_sequence,
                telemetry_sequence=message.telemetry_sequence,
                sample_timestamp_us=message.sample_timestamp_us,
                left_encoder_count=message.left_encoder_count,
                right_encoder_count=message.right_encoder_count,
                gyro_x_urad_s=message.gyro_x_urad_s,
                gyro_y_urad_s=message.gyro_y_urad_s,
                gyro_z_urad_s=message.gyro_z_urad_s,
                accel_x_mm_s2=message.accel_x_mm_s2,
                accel_y_mm_s2=message.accel_y_mm_s2,
                accel_z_mm_s2=message.accel_z_mm_s2,
                imu_temperature_cdeg=message.imu_temperature_cdeg,
                sensor_flags=int(message.sensor_flags),
            )
        elif isinstance(message, CarSystemStatus):
            self._write(
                "system_status",
                message.received_timestamp_ns,
                uart_sequence=message.uart_sequence,
                status_sequence=message.status_sequence,
                controller_timestamp_us=message.controller_timestamp_us,
                watchdog_timeout_ms=message.watchdog_timeout_ms,
                last_motion_command_age_ms=message.last_motion_command_age_ms,
                system_flags=int(message.system_flags),
                stop_reason=message.stop_reason.name.lower(),
                servo_left_target_cdeg=message.servo_left_target_cdeg,
                servo_right_target_cdeg=message.servo_right_target_cdeg,
            )
        elif isinstance(message, CarCommandReply):
            self._write(
                "command_reply",
                message.received_timestamp_ns,
                uart_sequence=message.uart_sequence,
                command_sequence=message.command_sequence,
                command_type=message.command_type.name.lower(),
                result=message.result.name.lower(),
            )
        else:
            raise TypeError(
                f"Unsupported car message {type(message).__name__}."
            )

    def record_safety_stop(self, *, reason: str, timestamp_ns: int) -> None:
        self._write("safety_stop", timestamp_ns, reason=reason)

    def stop(self, *, timestamp_ns: int) -> None:
        stream = self._stream
        if stream is None:
            return
        try:
            self._write("stream_finished", timestamp_ns)
        finally:
            self._stream = None
            stream.close()

    def _write(
        self,
        event_type: str,
        timestamp_ns: int,
        **fields: object,
    ) -> None:
        stream = self._stream
        if stream is None:
            raise RuntimeError("Manual motion log is not started.")
        _non_negative_integer(timestamp_ns, "timestamp_ns")
        record = {
            "log_sequence": self._sequence,
            "event_type": event_type,
            "timestamp_ns": timestamp_ns,
            **fields,
        }
        if event_type not in _EVENT_KEYS or set(record) != (
            _COMMON_KEYS | _EVENT_KEYS[event_type]
        ):
            raise ValueError(f"Invalid manual motion event {event_type!r}.")
        _validate_event_fields(event_type, record, location=event_type)
        stream.write(
            json.dumps(
                record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
            )
            + "\n"
        )
        stream.flush()
        self._sequence += 1


def inspect_manual_motion_log(path: str | Path) -> dict[str, object]:
    """严格校验手动运动 JSONL，并返回事件覆盖与单调时间范围。"""

    resolved = Path(path).expanduser().resolve()
    lines = resolved.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise ValueError("Manual motion log must contain at least one event.")
    counts: dict[str, int] = {}
    timestamps_ns: list[int] = []
    for line_number, line in enumerate(lines, start=1):
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{resolved}:{line_number}: invalid JSON."
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"{resolved}:{line_number}: event must be an object."
            )
        event_type = record.get("event_type")
        if event_type not in _EVENT_KEYS:
            raise ValueError(
                f"{resolved}:{line_number}: unknown event_type "
                f"{event_type!r}."
            )
        expected = _COMMON_KEYS | _EVENT_KEYS[str(event_type)]
        if set(record) != expected:
            raise ValueError(
                f"{resolved}:{line_number}: keys must be exactly "
                f"{sorted(expected)}."
            )
        if record["log_sequence"] != line_number - 1:
            raise ValueError(
                f"{resolved}:{line_number}: log_sequence must be "
                f"{line_number - 1}."
            )
        timestamp_ns = _non_negative_integer(
            record["timestamp_ns"],
            f"{resolved}:{line_number}.timestamp_ns",
        )
        _validate_event_fields(
            str(event_type),
            record,
            location=f"{resolved}:{line_number}",
        )
        timestamps_ns.append(timestamp_ns)
        counts[str(event_type)] = counts.get(str(event_type), 0) + 1
    if lines and json.loads(lines[0])["event_type"] != "stream_started":
        raise ValueError("Manual motion log must start with stream_started.")
    if json.loads(lines[-1])["event_type"] != "stream_finished":
        raise ValueError("Manual motion log must end with stream_finished.")
    return {
        "event_count": len(lines),
        "event_counts": dict(sorted(counts.items())),
        "first_timestamp_ns": min(timestamps_ns),
        "last_timestamp_ns": max(timestamps_ns),
    }


def _validate_event_fields(
    event_type: str,
    record: dict[str, Any],
    *,
    location: str,
) -> None:
    for name in (
        "deadline_timestamp_ns",
        "uart_sequence",
        "telemetry_sequence",
        "sample_timestamp_us",
        "status_sequence",
        "controller_timestamp_us",
        "watchdog_timeout_ms",
        "sensor_flags",
        "system_flags",
        "servo_left_target_cdeg",
        "servo_right_target_cdeg",
        "command_sequence",
    ):
        if name in record:
            _non_negative_integer(record[name], f"{location}.{name}")
    for name in (
        "linear_velocity_m_s",
        "angular_velocity_rad_s",
    ):
        if name in record:
            _finite_number(record[name], f"{location}.{name}")
    for name in (
        "command_id",
        "command_type",
        "result",
        "reason",
        "stop_reason",
    ):
        if name in record and (
            not isinstance(record[name], str) or not record[name]
        ):
            raise ValueError(f"{location}.{name} must be a non-empty string.")
    if "deadman_enabled" in record and not isinstance(
        record["deadman_enabled"], bool
    ):
        raise ValueError(f"{location}.deadman_enabled must be a boolean.")
    for name in ("open_pressed", "close_pressed"):
        if name in record and not isinstance(record[name], bool):
            raise ValueError(f"{location}.{name} must be a boolean.")
    if "last_motion_command_age_ms" in record:
        age = record["last_motion_command_age_ms"]
        if age is not None:
            _non_negative_integer(age, f"{location}.last_motion_command_age_ms")
    if (
        event_type == "motion_command"
        and record["result"] not in _MOTION_RESULTS
    ):
        raise ValueError(f"{location}.result is not supported.")
    if event_type == "gripper_command" and record["result"] not in {
        "applied",
        "stopped",
        "expired",
    }:
        raise ValueError(f"{location}.result is not supported.")
    if event_type in {"motion_command", "gripper_command"} and int(
        record["deadline_timestamp_ns"]
    ) <= int(record["timestamp_ns"]):
        raise ValueError(
            f"{location}.deadline_timestamp_ns must follow timestamp_ns."
        )
    if (
        event_type == "safety_stop"
        and record["reason"] not in _STOP_REASONS
    ):
        raise ValueError(f"{location}.reason is not supported.")
    if event_type == "odometry_imu":
        for name in (
            "left_encoder_count",
            "right_encoder_count",
            "gyro_x_urad_s",
            "gyro_y_urad_s",
            "gyro_z_urad_s",
            "accel_x_mm_s2",
            "accel_y_mm_s2",
            "accel_z_mm_s2",
            "imu_temperature_cdeg",
        ):
            if isinstance(record[name], bool) or not isinstance(record[name], int):
                raise ValueError(f"{location}.{name} must be an integer.")
    if event_type == "system_status":
        if int(record["watchdog_timeout_ms"]) <= 0:
            raise ValueError(
                f"{location}.watchdog_timeout_ms must be positive."
            )
        if record["stop_reason"] not in {
            "startup",
            "running",
            "soft_brake",
            "watchdog_timeout",
            "emergency_stop",
        }:
            raise ValueError(f"{location}.stop_reason is not supported.")
        for name in ("servo_left_target_cdeg", "servo_right_target_cdeg"):
            if not 0 <= int(record[name]) <= 18000:
                raise ValueError(f"{location}.{name} must be in [0, 18000].")


def _non_negative_integer(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{location} must be a non-negative integer.")
    return value


def _finite_number(value: object, location: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{location} must be a finite number.")
    return float(value)
