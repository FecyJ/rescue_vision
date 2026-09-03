"""执行一次配置化的低速直行检查；仅用于人工/真机验收。"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import time

from rescue_vision.config import load_runtime_config


_DEFAULT_SPEED_M_S = 0.05
_DEFAULT_DURATION_S = 1.0
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
            "Run one low-speed forward motion check with the configured UART. "
            "The controller is updated periodically and softly braked on exit."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.yaml"),
        help="Runtime YAML path (default: configs/runtime.yaml).",
    )
    parser.add_argument(
        "--speed-m-s",
        type=_positive_float,
        default=_DEFAULT_SPEED_M_S,
        help=f"Forward speed in m/s (default: {_DEFAULT_SPEED_M_S}).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=_positive_float,
        default=_DEFAULT_DURATION_S,
        help=f"Motion duration in seconds (default: {_DEFAULT_DURATION_S}).",
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


def main() -> None:
    args = _parse_args()
    config = load_runtime_config(args.config)
    if not config.uart.enabled:
        raise RuntimeError("uart.enabled must be true for a motion test.")
    if not config.motion.enabled:
        raise RuntimeError("motion.enabled must be true for a motion test.")

    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("The configured UART channel could not be created.")
    controller = config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("The configured motion controller could not be created.")

    print(
        f"forward speed={args.speed_m_s:.3f} m/s "
        f"duration={args.duration_seconds:.2f} s device={channel.device}; "
        "physical emergency stop must remain ready",
        flush=True,
    )

    with channel:
        try:
            controller.synchronize(
                timeout_s=config.motion.synchronization_timeout_s,
            )
            controller.forward(args.speed_m_s)
            deadline_ns = time.monotonic_ns() + round(
                args.duration_seconds * 1_000_000_000
            )
            while True:
                now_ns = time.monotonic_ns()
                if now_ns >= deadline_ns:
                    break
                controller.update(now_ns=now_ns)
                # STM32 may publish telemetry continuously; keep the bounded
                # receive queue drained without adding a blocking read.
                controller.drain_messages()
                time.sleep(_UPDATE_PERIOD_S)
        finally:
            controller.soft_brake()


if __name__ == "__main__":
    main()
