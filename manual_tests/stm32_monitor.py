"""被动监测 STM32 COBS/CRC16 串口协议；默认不发送任何命令。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from enum import IntFlag
from pathlib import Path
import time

from rescue_vision.config import load_runtime_config
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    ControllerProtocolError,
    OdometryImu,
    ParsedCarMessage,
    encode_state_query_command,
    parse_controller_frame,
)


@dataclass(slots=True)
class SequenceHealth:
    previous: int | None = None
    missing: int = 0
    duplicates: int = 0
    regressions: int = 0

    def observe(self, sequence: int) -> None:
        if self.previous is None:
            self.previous = sequence
            return
        delta = (sequence - self.previous) & 0xFFFF
        self.previous = sequence
        if delta == 1:
            return
        if delta == 0:
            self.duplicates += 1
        elif delta < 0x8000:
            self.missing += delta - 1
        else:
            self.regressions += 1


@dataclass(slots=True)
class MonitorStats:
    started_timestamp_ns: int
    odometry_count: int = 0
    status_count: int = 0
    reply_count: int = 0
    protocol_error_count: int = 0
    odometry_sequence: SequenceHealth | None = None
    status_sequence: SequenceHealth | None = None
    latest_odometry: OdometryImu | None = None
    latest_status: CarSystemStatus | None = None
    latest_protocol_error: str | None = None

    def __post_init__(self) -> None:
        if self.odometry_sequence is None:
            self.odometry_sequence = SequenceHealth()
        if self.status_sequence is None:
            self.status_sequence = SequenceHealth()

    def observe(self, message: ParsedCarMessage) -> None:
        if isinstance(message, OdometryImu):
            self.odometry_count += 1
            assert self.odometry_sequence is not None
            self.odometry_sequence.observe(message.telemetry_sequence)
            self.latest_odometry = message
        elif isinstance(message, CarSystemStatus):
            self.status_count += 1
            assert self.status_sequence is not None
            self.status_sequence.observe(message.status_sequence)
            self.latest_status = message
        elif isinstance(message, CarCommandReply):
            self.reply_count += 1
        else:
            raise TypeError(f"Unsupported message {type(message).__name__}.")

    def observe_protocol_error(self, error: ControllerProtocolError) -> None:
        self.protocol_error_count += 1
        self.latest_protocol_error = str(error)


def _flag_names(flags: IntFlag) -> str:
    names = [flag.name.lower() for flag in type(flags) if flags & flag]
    return "|".join(names) if names else "none"


def format_message(message: ParsedCarMessage) -> str:
    if isinstance(message, OdometryImu):
        return (
            "ODOM "
            f"seq={message.telemetry_sequence} "
            f"sample_us={message.sample_timestamp_us} "
            f"enc=({message.left_encoder_count},{message.right_encoder_count}) "
            "gyro_rad_s=("
            f"{message.gyro_x_urad_s / 1_000_000:.6f},"
            f"{message.gyro_y_urad_s / 1_000_000:.6f},"
            f"{message.gyro_z_urad_s / 1_000_000:.6f}) "
            "accel_m_s2=("
            f"{message.accel_x_mm_s2 / 1000:.3f},"
            f"{message.accel_y_mm_s2 / 1000:.3f},"
            f"{message.accel_z_mm_s2 / 1000:.3f}) "
            f"temp_c={message.imu_temperature_cdeg / 100:.2f} "
            f"flags={_flag_names(message.sensor_flags)}"
        )
    if isinstance(message, CarSystemStatus):
        age = (
            "none"
            if message.last_motion_command_age_ms is None
            else str(message.last_motion_command_age_ms)
        )
        return (
            "STATUS "
            f"seq={message.status_sequence} "
            f"controller_us={message.controller_timestamp_us} "
            f"watchdog_ms={message.watchdog_timeout_ms} "
            f"motion_age_ms={age} "
            f"reason={message.stop_reason.name.lower()} "
            f"servo_deg=({message.servo_left_deg:.2f},"
            f"{message.servo_right_deg:.2f}) "
            f"flags={_flag_names(message.system_flags)}"
        )
    if isinstance(message, CarCommandReply):
        return (
            "REPLY "
            f"command_seq={message.command_sequence} "
            f"command={message.command_type.name.lower()} "
            f"result={message.result.name.lower()}"
        )
    raise TypeError(f"Unsupported message {type(message).__name__}.")


def format_summary(
    stats: MonitorStats,
    *,
    now_ns: int,
    received_frames: int,
    discarded_cobs_frames: int,
) -> str:
    elapsed_s = max((now_ns - stats.started_timestamp_ns) / 1e9, 1e-9)
    assert stats.odometry_sequence is not None
    assert stats.status_sequence is not None
    return (
        "SUMMARY "
        f"elapsed_s={elapsed_s:.1f} "
        f"rx={received_frames} "
        f"odom={stats.odometry_count}({stats.odometry_count / elapsed_s:.1f}Hz) "
        f"status={stats.status_count}({stats.status_count / elapsed_s:.1f}Hz) "
        f"reply={stats.reply_count} "
        f"cobs_drop={discarded_cobs_frames} "
        f"protocol_error={stats.protocol_error_count} "
        f"odom_missing={stats.odometry_sequence.missing} "
        f"odom_duplicate={stats.odometry_sequence.duplicates} "
        f"odom_regression={stats.odometry_sequence.regressions} "
        f"status_missing={stats.status_sequence.missing} "
        f"status_duplicate={stats.status_sequence.duplicates} "
        f"status_regression={stats.status_sequence.regressions}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Monitor the configured STM32 COBS/CRC16 UART. Passive by default; "
            "--query-status sends one non-motion QUERY_STATUS frame."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.yaml"),
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=0.0,
        help="0 monitors until Ctrl+C; a positive value stops automatically.",
    )
    parser.add_argument(
        "--print-interval",
        type=float,
        default=1.0,
        help="Seconds between compact summaries and latest values.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print every valid frame (about 110 lines/s).",
    )
    parser.add_argument(
        "--query-status",
        action="store_true",
        help="Send one QUERY_STATUS after opening; never sends motion commands.",
    )
    args = parser.parse_args()
    if args.duration_seconds < 0:
        parser.error("--duration-seconds must be zero or positive")
    if args.print_interval <= 0:
        parser.error("--print-interval must be positive")
    return args


def main() -> None:
    args = _parse_args()
    config = load_runtime_config(args.config)
    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("uart.enabled must be true")

    started_ns = time.monotonic_ns()
    deadline_ns = (
        None
        if args.duration_seconds == 0
        else started_ns + round(args.duration_seconds * 1e9)
    )
    next_print_ns = started_ns + round(args.print_interval * 1e9)
    stats = MonitorStats(started_timestamp_ns=started_ns)

    print(
        f"device={channel.device} baudrate={channel.baudrate} "
        f"mode={'query_status_once' if args.query_status else 'passive'}",
        flush=True,
    )
    try:
        with channel:
            if args.query_status:
                channel.send_frame(encode_state_query_command(0))
            while deadline_ns is None or time.monotonic_ns() < deadline_ns:
                try:
                    frame = channel.receive_frame(timeout=0.1)
                except TimeoutError:
                    frame = None
                if frame is not None:
                    try:
                        message = parse_controller_frame(frame)
                    except ControllerProtocolError as error:
                        stats.observe_protocol_error(error)
                        if args.verbose:
                            print(f"PROTOCOL_ERROR {error}", flush=True)
                    else:
                        stats.observe(message)
                        if args.verbose or isinstance(message, CarCommandReply):
                            print(format_message(message), flush=True)

                now_ns = time.monotonic_ns()
                if now_ns >= next_print_ns:
                    print(
                        format_summary(
                            stats,
                            now_ns=now_ns,
                            received_frames=channel.received_frames,
                            discarded_cobs_frames=channel.discarded_frames,
                        ),
                        flush=True,
                    )
                    if stats.latest_odometry is not None:
                        print(format_message(stats.latest_odometry), flush=True)
                    if stats.latest_status is not None:
                        print(format_message(stats.latest_status), flush=True)
                    if stats.latest_protocol_error is not None:
                        print(
                            f"LATEST_PROTOCOL_ERROR {stats.latest_protocol_error}",
                            flush=True,
                        )
                    next_print_ns = now_ns + round(args.print_interval * 1e9)
    except KeyboardInterrupt:
        print("stopped_by_user=true", flush=True)
    finally:
        now_ns = time.monotonic_ns()
        print(
            format_summary(
                stats,
                now_ns=now_ns,
                received_frames=channel.received_frames,
                discarded_cobs_frames=channel.discarded_frames,
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
