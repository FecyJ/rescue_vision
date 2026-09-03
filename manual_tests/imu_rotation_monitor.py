"""保持低速原地旋转并实时显示 STM32 原始 gyro_z。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from rescue_vision.config import load_runtime_config
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    OdometryImu,
)


_DEFAULT_ANGULAR_VELOCITY_RAD_S = 0.15
_DEFAULT_PRINT_INTERVAL_S = 0.1
_UPDATE_PERIOD_S = 0.01


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
            "Keep the car turning slowly in place and print the latest raw "
            "STM32 gyro_z until Ctrl+C. The controller softly brakes on exit."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.yaml"),
        help="Runtime YAML path (default: configs/runtime.yaml).",
    )
    parser.add_argument(
        "--angular-velocity-rad-s",
        type=_positive_float,
        default=_DEFAULT_ANGULAR_VELOCITY_RAD_S,
        help=(
            "Positive in-place angular speed magnitude in rad/s "
            f"(default: {_DEFAULT_ANGULAR_VELOCITY_RAD_S})."
        ),
    )
    parser.add_argument(
        "--direction",
        choices=("left", "right"),
        default="left",
        help="Turn direction (default: left).",
    )
    parser.add_argument(
        "--print-interval-seconds",
        type=_positive_float,
        default=_DEFAULT_PRINT_INTERVAL_S,
        help=(
            "Maximum interval between terminal lines; each line contains the "
            f"latest ODOM sample (default: {_DEFAULT_PRINT_INTERVAL_S})."
        ),
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


def _requested_angular_velocity(direction: str, magnitude: float) -> float:
    if direction not in {"left", "right"}:
        raise ValueError(f"direction must be left or right, got {direction!r}.")
    if not math.isfinite(float(magnitude)) or float(magnitude) <= 0.0:
        raise ValueError("magnitude must be finite and positive.")
    return float(magnitude) if direction == "left" else -float(magnitude)


def _format_odometry(message: OdometryImu) -> str:
    flags = "|".join(
        flag.name.lower()
        for flag in type(message.sensor_flags)
        if message.sensor_flags & flag
    ) or "none"
    return (
        f"sample_us={message.sample_timestamp_us} "
        f"gyro_z={message.gyro_z_rad_s:+.6f} rad/s "
        f"gyro_xyz=({message.gyro_x_urad_s / 1_000_000.0:+.6f},"
        f"{message.gyro_y_urad_s / 1_000_000.0:+.6f},"
        f"{message.gyro_z_rad_s:+.6f}) rad/s "
        f"enc=({message.left_encoder_count},{message.right_encoder_count}) "
        f"flags={flags}"
    )


def main() -> None:
    args = _parse_args()
    config = load_runtime_config(args.config)
    if not config.uart.enabled:
        raise RuntimeError("uart.enabled must be true for a motion test.")
    if not config.motion.enabled:
        raise RuntimeError("motion.enabled must be true for a motion test.")
    if args.angular_velocity_rad_s > config.motion.max_angular_velocity_rad_s:
        raise ValueError(
            "Requested angular velocity exceeds configured limit: "
            f"{args.angular_velocity_rad_s} > "
            f"{config.motion.max_angular_velocity_rad_s} rad/s."
        )

    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("The configured UART channel could not be created.")
    controller = config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("The configured motion controller could not be created.")

    angular_velocity = _requested_angular_velocity(
        args.direction,
        args.angular_velocity_rad_s,
    )
    latest_odometry: OdometryImu | None = None
    odometry_count = 0
    next_print_ns = 0

    def consume(message: object) -> None:
        nonlocal latest_odometry, odometry_count
        if isinstance(message, OdometryImu):
            latest_odometry = message
            odometry_count += 1
            return
        if isinstance(message, CarCommandReply):
            if message.result is not CommandResult.ACCEPTED:
                raise RuntimeError(
                    "STM32 rejected command: "
                    f"{message.command_type.name.lower()}="
                    f"{message.result.name.lower()}"
                )
            return
        if isinstance(message, CarSystemStatus) and message.emergency_stop_latched:
            raise RuntimeError("STM32 emergency stop is latched.")

    print(
        f"direction={args.direction} "
        f"angular_velocity_rad_s={angular_velocity:+.3f} "
        f"print_interval_s={args.print_interval_seconds:.3f} "
        f"device={channel.device}; press Ctrl+C to stop",
        flush=True,
    )

    with channel:
        try:
            controller.synchronize(
                timeout_s=config.motion.synchronization_timeout_s,
                on_message=consume,
            )
            while True:
                now_ns = time.monotonic_ns()
                controller.update(now_ns=now_ns)
                for message in controller.drain_messages():
                    consume(message)

                if latest_odometry is not None and now_ns >= next_print_ns:
                    print(
                        f"odom_count={odometry_count} "
                        f"{_format_odometry(latest_odometry)}",
                        flush=True,
                    )
                    next_print_ns = now_ns + round(
                        args.print_interval_seconds * 1_000_000_000
                    )

                controller.drive_wheel_limited(0.0, angular_velocity)
                time.sleep(_UPDATE_PERIOD_S)
        except KeyboardInterrupt:
            print("stopped_by_user=true", flush=True)
        finally:
            controller.soft_brake()


if __name__ == "__main__":
    main()
