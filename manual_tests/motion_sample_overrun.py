"""低速架空轮检查 STM32 轮子启动后是否产生 sample_overrun。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time

from rescue_vision.config import load_runtime_config
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    MotionController,
    OdometryImu,
    SensorFlags,
)


_UPDATE_PERIOD_S = 0.01


@dataclass(frozen=True, slots=True)
class SampleOverrunEvent:
    telemetry_sequence: int
    sample_timestamp_us: int
    received_timestamp_ns: int
    left_encoder_count: int
    right_encoder_count: int
    sensor_flags: SensorFlags


@dataclass(slots=True)
class OverrunProbeStats:
    """保存一次架空轮检查的无硬件统计结果。"""

    started_timestamp_ns: int
    odometry_count: int = 0
    status_count: int = 0
    reply_count: int = 0
    rejected_reply_count: int = 0
    sample_overrun_count: int = 0
    resynchronization_count: int = 0
    first_sample_overrun: SampleOverrunEvent | None = None
    latest_status: CarSystemStatus | None = None

    def observe(self, message: object) -> bool:
        """记录一条已解析回传，发现 sample_overrun 时返回 ``True``。"""

        if isinstance(message, OdometryImu):
            self.odometry_count += 1
            if not message.sensor_flags & SensorFlags.SAMPLE_OVERRUN:
                return False
            self.sample_overrun_count += 1
            if self.first_sample_overrun is None:
                self.first_sample_overrun = SampleOverrunEvent(
                    telemetry_sequence=message.telemetry_sequence,
                    sample_timestamp_us=message.sample_timestamp_us,
                    received_timestamp_ns=message.received_timestamp_ns,
                    left_encoder_count=message.left_encoder_count,
                    right_encoder_count=message.right_encoder_count,
                    sensor_flags=message.sensor_flags,
                )
            return True
        if isinstance(message, CarSystemStatus):
            self.status_count += 1
            self.latest_status = message
            return False
        if isinstance(message, CarCommandReply):
            self.reply_count += 1
            if message.result is not CommandResult.ACCEPTED:
                self.rejected_reply_count += 1
            return False
        raise TypeError(f"Unsupported controller message {type(message).__name__}.")

    def summary(self, *, duration_s: float) -> str:
        event = self.first_sample_overrun
        first = "none"
        if event is not None:
            first = (
                f"telemetry_seq={event.telemetry_sequence},"
                f"sample_us={event.sample_timestamp_us},"
                f"enc=({event.left_encoder_count},{event.right_encoder_count}),"
                f"flags={_sensor_flag_text(event.sensor_flags)}"
            )
        return (
            f"SUMMARY duration_s={duration_s:.3f} "
            f"odom={self.odometry_count} "
            f"status={self.status_count} "
            f"reply={self.reply_count} "
            f"rejected_reply={self.rejected_reply_count} "
            f"sample_overrun={self.sample_overrun_count} "
            f"resynchronization={self.resynchronization_count} "
            f"first_overrun=({first}) "
            f"last_status=({format_status(self.latest_status)})"
        )


def _sensor_flag_text(flags: SensorFlags) -> str:
    names = [flag.name.lower() for flag in SensorFlags if flags & flag]
    return "|".join(names) if names else "none"


def format_status(status: CarSystemStatus | None) -> str:
    if status is None:
        return "none"
    return (
        f"protocol_ready={status.protocol_ready},"
        f"rx_degraded={status.rx_degraded},"
        f"tx_degraded={status.tx_degraded},"
        f"reply_queue_full={status.reply_queue_full},"
        f"estop={status.emergency_stop_latched},"
        f"motor_output={status.motor_output_enabled},"
        f"stop_reason={status.stop_reason.name.lower()},"
        f"motion_age_ms={status.last_motion_command_age_ms}"
    )


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
            "Run a low-speed suspended-wheel test and stop on the first "
            "STM32 ODOMETRY_IMU sample_overrun."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.simulation-20min.yaml"),
        help="Runtime YAML path.",
    )
    parser.add_argument(
        "--speed-m-s",
        type=_positive_float,
        default=0.05,
        help="Low suspended-wheel speed in m/s (default: 0.05).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=_positive_float,
        default=5.0,
        help="Maximum wheel-start test duration (default: 5).",
    )
    parser.add_argument(
        "--fail-on-overrun",
        action="store_true",
        help="Return exit code 2 when sample_overrun is detected.",
    )
    parser.add_argument(
        "--supervised-physical-stop-ready",
        action="store_true",
        help="Confirm suspended wheels/physical stop and continuous supervision.",
    )
    args = parser.parse_args()
    if not args.supervised_physical_stop_ready:
        parser.error(
            "--supervised-physical-stop-ready is required for a live wheel test"
        )
    return args


def _validate_runtime_config(config: object) -> None:
    if not config.uart.enabled:
        raise RuntimeError("uart.enabled must be true for the overrun test.")
    if not config.motion.enabled:
        raise RuntimeError("motion.enabled must be true for the overrun test.")
    if not config.motion.odometry.enabled:
        raise RuntimeError(
            "motion.odometry.enabled must be true for the overrun test."
        )


def main() -> int:
    args = _parse_args()
    config = load_runtime_config(args.config.expanduser().resolve())
    _validate_runtime_config(config)

    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("The configured UART channel could not be created.")
    controller = config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("The configured motion controller could not be created.")

    print(
        f"TEST=motion_sample_overrun device={channel.device} "
        f"speed_m_s={args.speed_m_s:.3f} duration_s={args.duration_seconds:.2f} "
        "wheels_must_be_suspended",
        flush=True,
    )
    stats = OverrunProbeStats(started_timestamp_ns=time.monotonic_ns())
    detected = False
    elapsed_s = 0.0
    try:
        with channel:
            try:
                controller.synchronize()
                print("SOFT_BRAKE_SYNC=accepted", flush=True)
                stats = OverrunProbeStats(started_timestamp_ns=time.monotonic_ns())

                def observe_resync_message(message: object) -> None:
                    if stats.observe(message):
                        raise RuntimeError(
                            "SAMPLE_OVERRUN arrived during motion resynchronization."
                        )

                controller.forward(args.speed_m_s)
                deadline_ns = time.monotonic_ns() + round(
                    args.duration_seconds * 1_000_000_000
                )
                motion_started_ns = time.monotonic_ns()
                while True:
                    now_ns = time.monotonic_ns()
                    if now_ns >= deadline_ns:
                        break
                    controller.update(now_ns=now_ns)
                    for message in controller.drain_messages():
                        if isinstance(message, CarCommandReply) and (
                            message.result is not CommandResult.ACCEPTED
                        ):
                            raise RuntimeError(
                                "STM32 rejected a motion-test command: "
                                f"{message.command_type.name.lower()}="
                                f"{message.result.name.lower()}."
                            )
                        if isinstance(message, CarSystemStatus) and (
                            message.emergency_stop_latched
                        ):
                            raise RuntimeError(
                                "STM32 emergency stop became latched during the test."
                            )
                        if stats.observe(message):
                            event = stats.first_sample_overrun
                            assert event is not None
                            print(
                                "SAMPLE_OVERRUN detected "
                                f"telemetry_seq={event.telemetry_sequence} "
                                f"sample_us={event.sample_timestamp_us} "
                                f"enc=({event.left_encoder_count},"
                                f"{event.right_encoder_count}) "
                                f"flags={_sensor_flag_text(event.sensor_flags)}",
                                flush=True,
                            )
                            detected = True
                            break
                    if detected:
                        break
                    if controller.needs_synchronization:
                        stats.resynchronization_count += 1
                        print(
                            "RESYNC_REQUESTED "
                            f"pending_wheel_commands={controller.pending_wheel_command_count} "
                            f"link_degraded={controller.link_degraded} "
                            f"status={format_status(stats.latest_status)}",
                            flush=True,
                        )
                        controller.synchronize(on_message=observe_resync_message)
                        print("RESYNC_SYNC=accepted", flush=True)
                        if controller.link_degraded:
                            raise RuntimeError(
                                "Motion controller link remained degraded after "
                                "resynchronization."
                            )
                    time.sleep(min(_UPDATE_PERIOD_S, max(
                        0.0,
                        (deadline_ns - time.monotonic_ns()) / 1_000_000_000.0,
                    )))
                elapsed_s = (time.monotonic_ns() - motion_started_ns) / 1e9
            finally:
                controller.soft_brake()
    except KeyboardInterrupt:
        elapsed_s = (time.monotonic_ns() - stats.started_timestamp_ns) / 1e9
        print("INTERRUPTED; soft brake sent.", flush=True)
        return 130
    except BaseException as error:
        elapsed_s = max(
            elapsed_s,
            (time.monotonic_ns() - stats.started_timestamp_ns) / 1e9,
        )
        print(f"TEST_ERROR={type(error).__name__}: {error}", flush=True)
        print(stats.summary(duration_s=elapsed_s), flush=True)
        return 1

    print(stats.summary(duration_s=elapsed_s), flush=True)
    if detected:
        print("RESULT=sample_overrun_detected; motion_stopped", flush=True)
        return 2 if args.fail_on_overrun else 0
    print("RESULT=no_sample_overrun_observed", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
