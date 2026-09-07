"""D2 到安全区末端的高频编码器/IMU 记录。"""

from __future__ import annotations

import json
import math
from pathlib import Path
from queue import Empty, Full, Queue
from threading import Event, Lock, Thread
from typing import TYPE_CHECKING, TextIO

from rescue_vision.motion.protocol import OdometryImu, SensorFlags

if TYPE_CHECKING:
    from rescue_vision.localization import OdometryCalibration


MATCH_D2_TELEMETRY_STAGE = "d2_to_safe_zone"
MATCH_D2_TELEMETRY_DEFAULT_QUEUE_CAPACITY = 512


def _timestamp_ns(value: object, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            f"{location} must be a non-negative integer, got {value!r}."
        )
    return value


def _finite_or_none(value: object, location: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location} must be a finite number or None, got {value!r}.")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{location} must be a finite number or None, got {value!r}.")
    return converted


def _non_empty_text(value: object, location: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{location} must be a non-empty string, got {value!r}.")
    return value


def _speed_pair(
    value: tuple[float, float] | None,
    location: str,
) -> tuple[float | None, float | None]:
    if value is None:
        return None, None
    if not isinstance(value, tuple) or len(value) != 2:
        raise ValueError(f"{location} must be a 2-tuple or None, got {value!r}.")
    return (
        _finite_or_none(value[0], f"{location}[0]"),
        _finite_or_none(value[1], f"{location}[1]"),
    )


def _calibration_values(
    calibration: OdometryCalibration,
) -> tuple[int, float, float]:
    """校验并提取编码器换算所需的机械标定值。"""

    counts = getattr(calibration, "encoder_counts_per_revolution", None)
    if isinstance(counts, bool) or not isinstance(counts, int) or counts <= 0:
        raise TypeError(
            "calibration must provide a positive integer "
            "encoder_counts_per_revolution."
        )
    radii: list[float] = []
    for name in ("left_wheel_radius_mm", "right_wheel_radius_mm"):
        value = getattr(calibration, name, None)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"calibration must provide finite {name}.")
        radius = float(value)
        if not math.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"calibration.{name} must be finite and positive.")
        radii.append(radius)
    return counts, radii[0], radii[1]


class D2TelemetryLogger:
    """异步记录 正式流程 的 D2→安全区末端遥测。

    ``record_odometry()`` 应在 UART 消费路径中对每个 ``ODOMETRY_IMU`` 调用，
    不做时间抽样。编码器轮速使用 STM32 的 ``sample_timestamp_us`` 和相邻累计
    计数计算，而不是使用可能受 UART 批量接收影响的主机打印间隔。JSONL 写盘
    在独立线程中执行；队列有界且满时丢弃最旧待写记录，控制循环不会等待磁盘。
    """

    def __init__(
        self,
        path: str | Path,
        calibration: OdometryCalibration,
        *,
        queue_capacity: int = MATCH_D2_TELEMETRY_DEFAULT_QUEUE_CAPACITY,
    ) -> None:
        counts, left_radius_mm, right_radius_mm = _calibration_values(calibration)
        if (
            isinstance(queue_capacity, bool)
            or not isinstance(queue_capacity, int)
            or queue_capacity <= 0
        ):
            raise ValueError(
                "queue_capacity must be a positive integer, "
                f"got {queue_capacity!r}."
            )
        self.path = Path(path).expanduser().resolve()
        self._encoder_counts_per_revolution = counts
        self._left_wheel_radius_mm = left_radius_mm
        self._right_wheel_radius_mm = right_radius_mm
        self._queue_capacity = queue_capacity
        self._queue: Queue[dict[str, object]] = Queue(maxsize=queue_capacity)
        self._stop_event = Event()
        self._error_lock = Lock()
        self._worker_error: BaseException | None = None
        self._stream: TextIO | None = None
        self._thread: Thread | None = None
        self._started = False
        self._sequence = 0
        self._dropped_records = 0
        self._previous: OdometryImu | None = None
        self._process_started_timestamp_ns: int | None = None
        self._phase_started_timestamp_ns: int | None = None

    @property
    def started(self) -> bool:
        return self._started

    @property
    def dropped_records(self) -> int:
        return self._dropped_records

    @property
    def worker_error(self) -> BaseException | None:
        with self._error_lock:
            return self._worker_error

    def start(self, *, timestamp_ns: int) -> None:
        """启动后台写线程并写入流头。"""

        if self._started:
            raise RuntimeError("正式流程 D2 telemetry logger is already started.")
        timestamp = _timestamp_ns(timestamp_ns, "timestamp_ns")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        stream = self.path.open("x", encoding="utf-8", buffering=1)
        self._stream = stream
        self._queue = Queue(maxsize=self._queue_capacity)
        self._stop_event.clear()
        with self._error_lock:
            self._worker_error = None
        self._sequence = 0
        self._dropped_records = 0
        self._previous = None
        self._process_started_timestamp_ns = None
        self._phase_started_timestamp_ns = None
        self._thread = Thread(
            target=self._worker_loop,
            name="rescue-match-d2-telemetry-log",
            daemon=True,
        )
        self._started = True
        try:
            self._thread.start()
            self._enqueue(
                {
                    "event_type": "stream_started",
                    "timestamp_ns": timestamp,
                    "process_timestamp_ms": None,
                    "stage": MATCH_D2_TELEMETRY_STAGE,
                    "sample_period_target_ms": 10,
                    "sampling": "every_odometry_imu",
                    "queue_capacity": self._queue_capacity,
                }
            )
        except BaseException:
            self._started = False
            self._thread = None
            stream.close()
            self._stream = None
            raise

    def set_process_start_timestamp_ns(self, timestamp_ns: int) -> None:
        """设置正式流程起点，并写入流程时间 ``t=0`` 标记。"""

        self._require_started()
        timestamp = _timestamp_ns(timestamp_ns, "timestamp_ns")
        if self._process_started_timestamp_ns is not None:
            raise RuntimeError("Process start timestamp is already set.")
        self._process_started_timestamp_ns = timestamp
        self._enqueue(
            {
                "event_type": "process_started",
                "timestamp_ns": timestamp,
                "process_timestamp_ms": 0.0,
                "stage": MATCH_D2_TELEMETRY_STAGE,
            }
        )

    def begin_phase(
        self,
        *,
        timestamp_ns: int,
        state: str,
        route_phase: str,
        reason: str,
    ) -> None:
        """记录 D2 到达并开始接收逐帧遥测。"""

        self._require_started()
        timestamp = _timestamp_ns(timestamp_ns, "timestamp_ns")
        state_text = _non_empty_text(state, "state")
        phase_text = _non_empty_text(route_phase, "route_phase")
        reason_text = _non_empty_text(reason, "reason")
        if self._phase_started_timestamp_ns is not None:
            raise RuntimeError("正式流程 D2 telemetry phase is already active.")
        self._phase_started_timestamp_ns = timestamp
        self._enqueue(
            {
                "event_type": "phase_started",
                "timestamp_ns": timestamp,
                "process_timestamp_ms": self._process_timestamp_ms(timestamp),
                "stage": MATCH_D2_TELEMETRY_STAGE,
                "state": state_text,
                "route_phase": phase_text,
                "reason": reason_text,
            }
        )

    def end_phase(
        self,
        *,
        timestamp_ns: int,
        state: str,
        route_phase: str,
        reason: str,
    ) -> None:
        """记录物块到达安全区末端，停止该阶段采样。"""

        self._require_started()
        timestamp = _timestamp_ns(timestamp_ns, "timestamp_ns")
        state_text = _non_empty_text(state, "state")
        phase_text = _non_empty_text(route_phase, "route_phase")
        reason_text = _non_empty_text(reason, "reason")
        if self._phase_started_timestamp_ns is None:
            return
        self._enqueue(
            {
                "event_type": "phase_finished",
                "timestamp_ns": timestamp,
                "process_timestamp_ms": self._process_timestamp_ms(timestamp),
                "stage": MATCH_D2_TELEMETRY_STAGE,
                "state": state_text,
                "route_phase": phase_text,
                "reason": reason_text,
                "dropped_records": self._dropped_records,
            }
        )
        self._phase_started_timestamp_ns = None

    def record_odometry(
        self,
        message: OdometryImu,
        *,
        active: bool,
        state: str,
        route_phase: str,
        target_wheel_speeds_m_s: tuple[float, float] | None = None,
        commanded_wheel_speeds_m_s: tuple[float, float] | None = None,
    ) -> None:
        """观察一帧遥测；阶段激活时将该帧加入后台日志队列。"""

        self._require_started()
        if not isinstance(message, OdometryImu):
            raise TypeError("message must be an OdometryImu.")
        if not isinstance(active, bool):
            raise TypeError("active must be a boolean.")
        state_text = _non_empty_text(state, "state")
        phase_text = _non_empty_text(route_phase, "route_phase")
        target_pair = _speed_pair(
            target_wheel_speeds_m_s,
            "target_wheel_speeds_m_s",
        )
        commanded_pair = _speed_pair(
            commanded_wheel_speeds_m_s,
            "commanded_wheel_speeds_m_s",
        )
        previous = self._previous
        values = self._encoder_values(message, previous)
        self._previous = message
        if not active:
            return
        if self._phase_started_timestamp_ns is None:
            raise RuntimeError("Active D2 telemetry requires begin_phase().")
        phase_elapsed_ms = max(
            0.0,
            (message.received_timestamp_ns - self._phase_started_timestamp_ns)
            / 1_000_000.0,
        )
        record: dict[str, object] = {
            "event_type": "odometry_imu",
            "timestamp_ns": message.received_timestamp_ns,
            "process_timestamp_ms": self._process_timestamp_ms(
                message.received_timestamp_ns
            ),
            "stage": MATCH_D2_TELEMETRY_STAGE,
            "phase_elapsed_ms": phase_elapsed_ms,
            "state": state_text,
            "route_phase": phase_text,
            "uart_sequence": message.uart_sequence,
            "telemetry_sequence": message.telemetry_sequence,
            "telemetry_sequence_delta": values["telemetry_sequence_delta"],
            "sample_timestamp_us": message.sample_timestamp_us,
            "sample_dt_us": values["sample_dt_us"],
            "host_dt_ns": values["host_dt_ns"],
            "left_encoder_count": message.left_encoder_count,
            "right_encoder_count": message.right_encoder_count,
            "left_encoder_delta_count": values["left_encoder_delta_count"],
            "right_encoder_delta_count": values["right_encoder_delta_count"],
            "encoder_speed_valid": values["encoder_speed_valid"],
            "encoder_speed_invalid_reason": values["encoder_speed_invalid_reason"],
            "left_encoder_speed_m_s": values["left_encoder_speed_m_s"],
            "right_encoder_speed_m_s": values["right_encoder_speed_m_s"],
            "center_encoder_speed_m_s": values["center_encoder_speed_m_s"],
            "target_left_wheel_speed_m_s": target_pair[0],
            "target_right_wheel_speed_m_s": target_pair[1],
            "commanded_left_wheel_speed_m_s": commanded_pair[0],
            "commanded_right_wheel_speed_m_s": commanded_pair[1],
            "gyro_x_urad_s": message.gyro_x_urad_s,
            "gyro_y_urad_s": message.gyro_y_urad_s,
            "gyro_z_urad_s": message.gyro_z_urad_s,
            "gyro_x_rad_s": message.gyro_x_urad_s / 1_000_000.0,
            "gyro_y_rad_s": message.gyro_y_urad_s / 1_000_000.0,
            "gyro_z_rad_s": message.gyro_z_urad_s / 1_000_000.0,
            "accel_x_mm_s2": message.accel_x_mm_s2,
            "accel_y_mm_s2": message.accel_y_mm_s2,
            "accel_z_mm_s2": message.accel_z_mm_s2,
            "imu_temperature_cdeg": message.imu_temperature_cdeg,
            "imu_temperature_c": message.imu_temperature_cdeg / 100.0,
            "sensor_flags": int(message.sensor_flags),
            "dropped_records": self._dropped_records,
        }
        self._enqueue(record)

    def stop(self, *, timestamp_ns: int) -> None:
        """停止后台线程并尽量排空已接收的日志。"""

        if not self._started:
            return
        timestamp = _timestamp_ns(timestamp_ns, "timestamp_ns")
        try:
            if self._phase_started_timestamp_ns is not None:
                self.end_phase(
                    timestamp_ns=timestamp,
                    state="unknown",
                    route_phase="shutdown",
                    reason="logger_stopped_before_phase_end",
                )
            self._enqueue(
                {
                    "event_type": "stream_finished",
                    "timestamp_ns": timestamp,
                    "process_timestamp_ms": self._process_timestamp_ms(timestamp),
                    "stage": MATCH_D2_TELEMETRY_STAGE,
                    "dropped_records": self._dropped_records,
                }
            )
            self._stop_event.set()
            thread = self._thread
            if thread is not None:
                thread.join(timeout=2.0)
            if thread is not None and thread.is_alive():
                raise RuntimeError("正式流程 D2 telemetry logger worker did not stop.")
        finally:
            stream = self._stream
            self._stream = None
            self._thread = None
            self._started = False
            if stream is not None:
                stream.close()

    def _encoder_values(
        self,
        message: OdometryImu,
        previous: OdometryImu | None,
    ) -> dict[str, object]:
        if previous is None:
            return {
                "telemetry_sequence_delta": None,
                "sample_dt_us": None,
                "host_dt_ns": None,
                "left_encoder_delta_count": None,
                "right_encoder_delta_count": None,
                "encoder_speed_valid": False,
                "encoder_speed_invalid_reason": "no_previous_sample",
                "left_encoder_speed_m_s": None,
                "right_encoder_speed_m_s": None,
                "center_encoder_speed_m_s": None,
            }
        sample_dt_us = message.sample_timestamp_us - previous.sample_timestamp_us
        host_dt_ns = message.received_timestamp_ns - previous.received_timestamp_ns
        left_delta = message.left_encoder_count - previous.left_encoder_count
        right_delta = message.right_encoder_count - previous.right_encoder_count
        sequence_delta = (
            message.telemetry_sequence - previous.telemetry_sequence
        ) & 0xFFFF
        required_flags = (
            SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
        )
        overrun = bool(
            message.sensor_flags & SensorFlags.SAMPLE_OVERRUN
            or previous.sensor_flags & SensorFlags.SAMPLE_OVERRUN
        )
        if overrun:
            invalid_reason = "sample_overrun"
        elif message.sensor_flags & required_flags != required_flags:
            invalid_reason = "encoder_invalid"
        elif previous.sensor_flags & required_flags != required_flags:
            invalid_reason = "previous_encoder_invalid"
        elif sample_dt_us <= 0:
            invalid_reason = "nonpositive_sample_dt"
        else:
            invalid_reason = None
        if invalid_reason is not None:
            left_speed = None
            right_speed = None
            center_speed = None
            valid = False
        else:
            dt_s = sample_dt_us / 1_000_000.0
            left_speed = (
                left_delta
                * 2.0
                * math.pi
                * self._left_wheel_radius_mm
                / self._encoder_counts_per_revolution
                / 1000.0
                / dt_s
            )
            right_speed = (
                right_delta
                * 2.0
                * math.pi
                * self._right_wheel_radius_mm
                / self._encoder_counts_per_revolution
                / 1000.0
                / dt_s
            )
            center_speed = 0.5 * (left_speed + right_speed)
            valid = True
        return {
            "telemetry_sequence_delta": sequence_delta,
            "sample_dt_us": sample_dt_us,
            "host_dt_ns": host_dt_ns if host_dt_ns >= 0 else None,
            "left_encoder_delta_count": left_delta,
            "right_encoder_delta_count": right_delta,
            "encoder_speed_valid": valid,
            "encoder_speed_invalid_reason": invalid_reason,
            "left_encoder_speed_m_s": left_speed,
            "right_encoder_speed_m_s": right_speed,
            "center_encoder_speed_m_s": center_speed,
        }

    def _enqueue(self, fields: dict[str, object]) -> None:
        record = {
            "log_sequence": self._sequence,
            **fields,
        }
        self._sequence += 1
        while True:
            try:
                self._queue.put_nowait(record)
                return
            except Full:
                try:
                    self._queue.get_nowait()
                except Empty:
                    continue
                self._queue.task_done()
                self._dropped_records += 1

    def _process_timestamp_ms(self, timestamp_ns: int) -> float | None:
        started = self._process_started_timestamp_ns
        if started is None:
            return None
        return max(0.0, (timestamp_ns - started) / 1_000_000.0)

    def _worker_loop(self) -> None:
        stream = self._stream
        if stream is None:
            return
        while True:
            if self._stop_event.is_set() and self._queue.empty():
                return
            try:
                record = self._queue.get(timeout=0.05)
            except Empty:
                continue
            try:
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
            except BaseException as exc:
                with self._error_lock:
                    self._worker_error = exc
                return
            finally:
                self._queue.task_done()

    def _require_started(self) -> None:
        if not self._started:
            raise RuntimeError("正式流程 D2 telemetry logger is not started.")


__all__ = [
    "MATCH_D2_TELEMETRY_DEFAULT_QUEUE_CAPACITY",
    "MATCH_D2_TELEMETRY_STAGE",
    "D2TelemetryLogger",
]
