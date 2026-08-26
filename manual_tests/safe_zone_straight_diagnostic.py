"""按配置执行 LEAVE_START 直行段并诊断 ODOMETRY_IMU sample_overrun。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from rescue_vision.app.cluster_breakup import EncoderTravelTracker
from rescue_vision.config import load_runtime_config
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    OdometryImu,
    SensorFlags,
)

if __package__:
    from .motion_sample_overrun import (
        OverrunProbeStats,
        _sensor_flag_text,
        format_status,
        require_motion_status_healthy,
    )
else:
    from motion_sample_overrun import (
        OverrunProbeStats,
        _sensor_flag_text,
        format_status,
        require_motion_status_healthy,
    )


_UPDATE_PERIOD_S = 0.01
_DEFAULT_PRINT_INTERVAL_S = 0.1
_BASELINE_TIMEOUT_S = 1.0


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite value greater than zero, got {value!r}"
        )
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run only the configured LEAVE_START straight segment and print "
            "detailed UART/ODOMETRY_IMU diagnostics. The test stops at the "
            "configured departure distance or on a second consecutive overrun."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.simulation-20min.yaml"),
        help="Runtime YAML path (default: configs/runtime.simulation-20min.yaml).",
    )
    parser.add_argument(
        "--speed-m-s",
        type=_positive_float,
        default=None,
        help="Override cluster_breakup.departure_speed_m_s.",
    )
    parser.add_argument(
        "--distance-m",
        type=_positive_float,
        default=None,
        help="Override cluster_breakup.departure_distance_m.",
    )
    parser.add_argument(
        "--print-interval-seconds",
        type=_positive_float,
        default=_DEFAULT_PRINT_INTERVAL_S,
        help=f"Progress output interval (default: {_DEFAULT_PRINT_INTERVAL_S}).",
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help=(
            "Confirm that a physical emergency stop is ready and an operator "
            "will supervise the whole test."
        ),
    )
    args = parser.parse_args()
    if not args.supervised_physical_stop_ready:
        parser.error(
            "--supervised-physical-stop-ready is required for a live motion test"
        )
    return args


def _format_odometry_diagnostic(
    message: OdometryImu,
    *,
    previous: OdometryImu | None,
) -> str:
    if previous is None:
        sequence_delta = "none"
        sample_dt_ms = "none"
        host_dt_ms = "none"
        encoder_delta = "none"
    else:
        sequence_delta = str(
            (message.telemetry_sequence - previous.telemetry_sequence) & 0xFFFF
        )
        sample_dt_ms = (
            f"{(message.sample_timestamp_us - previous.sample_timestamp_us) / 1000.0:.3f}"
        )
        host_dt_ms = (
            f"{(message.received_timestamp_ns - previous.received_timestamp_ns) / 1_000_000.0:.3f}"
        )
        encoder_delta = (
            f"({message.left_encoder_count - previous.left_encoder_count},"
            f"{message.right_encoder_count - previous.right_encoder_count})"
        )
    return (
        f"ODOM seq={message.telemetry_sequence} seq_delta={sequence_delta} "
        f"sample_us={message.sample_timestamp_us} sample_dt_ms={sample_dt_ms} "
        f"host_dt_ms={host_dt_ms} enc=({message.left_encoder_count},"
        f"{message.right_encoder_count}) enc_delta={encoder_delta} "
        f"gyro_xyz_rad_s=({message.gyro_x_urad_s / 1_000_000.0:+.6f},"
        f"{message.gyro_y_urad_s / 1_000_000.0:+.6f},"
        f"{message.gyro_z_urad_s / 1_000_000.0:+.6f}) "
        f"accel_mm_s2=({message.accel_x_mm_s2},"
        f"{message.accel_y_mm_s2},{message.accel_z_mm_s2}) "
        f"temp_c={message.imu_temperature_cdeg / 100.0:.2f} "
        f"flags={_sensor_flag_text(message.sensor_flags)}"
    )


def _validate_runtime_config(config: object) -> None:
    if not config.uart.enabled:
        raise RuntimeError("uart.enabled must be true for the straight diagnostic.")
    if not config.motion.enabled:
        raise RuntimeError("motion.enabled must be true for the straight diagnostic.")
    if not config.motion.odometry.enabled:
        raise RuntimeError(
            "motion.odometry.enabled must be true for the straight diagnostic."
        )
    if not config.motion.cluster_breakup.enabled:
        raise RuntimeError(
            "motion.cluster_breakup.enabled must be true for the straight diagnostic."
        )


def main() -> int:
    args = _parse_args()
    config = load_runtime_config(args.config.expanduser().resolve())
    _validate_runtime_config(config)

    speed_m_s = (
        config.motion.cluster_breakup.departure_speed_m_s
        if args.speed_m_s is None
        else args.speed_m_s
    )
    distance_m = (
        config.motion.cluster_breakup.departure_distance_m
        if args.distance_m is None
        else args.distance_m
    )
    if speed_m_s > config.motion.max_linear_velocity_m_s:
        raise ValueError(
            "Straight speed exceeds configured linear limit: "
            f"{speed_m_s} > {config.motion.max_linear_velocity_m_s} m/s."
        )

    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("The configured UART channel could not be created.")
    controller = config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("The configured motion controller could not be created.")
    calibration = config.motion.odometry.build_calibration()
    if calibration is None:
        raise RuntimeError("The configured odometry calibration is unavailable.")

    tracker = EncoderTravelTracker(
        calibration,
        max_wheel_velocity_m_s=config.motion.max_wheel_velocity_m_s,
        max_consecutive_overrun_samples=(
            config.motion.odometry.max_consecutive_overrun_samples
        ),
    )
    stats = OverrunProbeStats(started_timestamp_ns=time.monotonic_ns())
    latest_status: CarSystemStatus | None = None
    latest_odometry: OdometryImu | None = None
    previous_odometry: OdometryImu | None = None
    latest_odometry_diagnostic: str | None = None
    baseline_odometry: OdometryImu | None = None
    last_progress_ns = 0
    motion_started_ns: int | None = None
    stopped_for_overrun = False
    reached_distance = False

    def consume(message: object, *, print_overrun: bool = True) -> None:
        nonlocal latest_status, latest_odometry, previous_odometry
        nonlocal latest_odometry_diagnostic, baseline_odometry
        if isinstance(message, OdometryImu):
            is_overrun = stats.observe(message)
            diagnostic = _format_odometry_diagnostic(
                message,
                previous=previous_odometry,
            )
            latest_odometry = message
            latest_odometry_diagnostic = diagnostic
            if print_overrun and is_overrun:
                print(
                    "SAMPLE_OVERRUN "
                    f"{diagnostic} "
                    f"tracker=({tracker.diagnostic()})",
                    flush=True,
                )
            if (
                not is_overrun
                and message.sensor_flags
                & (SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID)
                == (
                    SensorFlags.LEFT_ENCODER_VALID
                    | SensorFlags.RIGHT_ENCODER_VALID
                )
            ):
                baseline_odometry = message
            previous_odometry = message
            return
        if isinstance(message, CarSystemStatus):
            stats.observe(message)
            latest_status = message
            require_motion_status_healthy(message)
            if message.emergency_stop_latched:
                raise RuntimeError(
                    "STM32 emergency stop is latched; reset it before motion."
                )
            return
        if isinstance(message, CarCommandReply):
            stats.observe(message)
            if message.result is CommandResult.SEQUENCE_OLD:
                print(
                    "COMMAND_REPLY sequence_old; controller resynchronization "
                    f"pending command={message.command_type.name.lower()}",
                    flush=True,
                )
                return
            if message.result is not CommandResult.ACCEPTED:
                raise RuntimeError(
                    "STM32 rejected command: "
                    f"{message.command_type.name.lower()}="
                    f"{message.result.name.lower()}"
                )
            return
        raise TypeError(f"Unsupported controller message {type(message).__name__}.")

    def synchronize_if_needed() -> None:
        if not controller.needs_synchronization:
            return
        stats.resynchronization_count += 1
        print(
            "RESYNC_REQUESTED "
            f"pending_wheel_commands={controller.pending_wheel_command_count} "
            f"link_degraded={controller.link_degraded} "
            f"status=({format_status(latest_status)})",
            flush=True,
        )
        controller.synchronize(on_message=consume)
        print("RESYNC_SYNC=accepted", flush=True)
        if controller.link_degraded:
            raise RuntimeError(
                "STM32 reported a sticky degraded UART state; reset or "
                "power-cycle it before retrying."
            )
        if controller.emergency_stop_latched:
            raise RuntimeError("STM32 emergency stop became latched during resync.")

    def drain() -> None:
        for message in controller.drain_messages():
            consume(message)

    def service(seconds: float) -> None:
        deadline_ns = time.monotonic_ns() + round(seconds * 1_000_000_000)
        while time.monotonic_ns() < deadline_ns:
            synchronize_if_needed()
            controller.update(now_ns=time.monotonic_ns())
            drain()
            time.sleep(_UPDATE_PERIOD_S)

    print(
        "TEST=safe_zone_straight_diagnostic "
        f"device={channel.device} speed_m_s={speed_m_s:.3f} "
        f"distance_m={distance_m:.3f} "
        f"max_duration_s={config.motion.cluster_breakup.motion_phase_timeout_s:.2f} "
        f"max_consecutive_overrun_samples="
        f"{config.motion.odometry.max_consecutive_overrun_samples}",
        flush=True,
    )

    try:
        with channel:
            try:
                controller.synchronize(on_message=consume)
                print("SOFT_BRAKE_SYNC=accepted", flush=True)
                service(_BASELINE_TIMEOUT_S)
                if baseline_odometry is None:
                    raise RuntimeError("No ODOMETRY_IMU sample before motion.")
                tracker.submit(baseline_odometry)
                previous_odometry = baseline_odometry
                synchronize_if_needed()
                controller.forward(speed_m_s)
                motion_started_ns = time.monotonic_ns()
                deadline_ns = motion_started_ns + round(
                    config.motion.cluster_breakup.motion_phase_timeout_s
                    * 1_000_000_000
                )
                while True:
                    synchronize_if_needed()
                    now_ns = time.monotonic_ns()
                    if now_ns >= deadline_ns:
                        raise RuntimeError(
                            "LEAVE_START straight segment timed out before "
                            f"reaching {distance_m:.3f} m; {tracker.diagnostic()}"
                        )
                    controller.update(now_ns=now_ns)
                    for message in controller.drain_messages():
                        if isinstance(message, OdometryImu):
                            is_overrun = stats.observe(message)
                            diagnostic = _format_odometry_diagnostic(
                                message,
                                previous=previous_odometry,
                            )
                            if is_overrun:
                                print(
                                    "SAMPLE_OVERRUN "
                                    f"{diagnostic} "
                                    f"tracker_before=({tracker.diagnostic()})",
                                    flush=True,
                                )
                            try:
                                distance = tracker.submit(message)
                            except RuntimeError as error:
                                stopped_for_overrun = is_overrun
                                raise RuntimeError(
                                    f"Encoder tracker rejected telemetry: {error}; "
                                    f"{_format_odometry_diagnostic(message, previous=previous_odometry)}"
                                ) from error
                            if tracker.forward_sign_mismatch():
                                raise RuntimeError(
                                    "Forward encoder signs disagree with the "
                                    f"protocol; {tracker.diagnostic()}"
                                )
                            previous_odometry = message
                            latest_odometry = message
                            latest_odometry_diagnostic = diagnostic
                            if distance >= distance_m:
                                reached_distance = True
                                break
                        else:
                            consume(message)
                    if reached_distance:
                        break
                    if now_ns >= last_progress_ns:
                        status_text = format_status(latest_status)
                        odom_text = latest_odometry_diagnostic or "odom=none"
                        print(
                            "PROGRESS "
                            f"elapsed_s={(now_ns - motion_started_ns) / 1e9:.3f} "
                            f"{tracker.diagnostic()} "
                            f"target_wheel_m_s={controller.target_wheel_speeds_m_s} "
                            f"commanded_wheel_m_s={controller.commanded_wheel_speeds_m_s} "
                            f"{odom_text} "
                            f"status=({status_text}) "
                            f"uart_rx={channel.received_frames} "
                            f"uart_cobs_drop={channel.discarded_frames} "
                            f"controller_invalid_rx={controller.invalid_received_frames}",
                            flush=True,
                        )
                        last_progress_ns = now_ns + round(
                            args.print_interval_seconds * 1_000_000_000
                        )
                    time.sleep(_UPDATE_PERIOD_S)
                controller.soft_brake()
            finally:
                controller.soft_brake()
    except KeyboardInterrupt:
        print("INTERRUPTED; soft brake sent.", flush=True)
        return 130
    except BaseException as error:
        print(f"TEST_ERROR={type(error).__name__}: {error}", flush=True)
        print(
            stats.summary(
                duration_s=(
                    0.0
                    if motion_started_ns is None
                    else (time.monotonic_ns() - motion_started_ns) / 1e9
                )
            ),
            flush=True,
        )
        return 2 if stopped_for_overrun or stats.sample_overrun_count else 1

    duration_s = (
        0.0
        if motion_started_ns is None
        else (time.monotonic_ns() - motion_started_ns) / 1e9
    )
    print(stats.summary(duration_s=duration_s), flush=True)
    print(
        "RESULT=departure_distance_reached "
        f"reached={reached_distance} "
        f"{tracker.diagnostic()} "
        f"target_distance_m={distance_m:.3f}",
        flush=True,
    )
    return 2 if stats.sample_overrun_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
