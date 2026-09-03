"""收集静止 STM32 IMU 样本并生成陀螺零偏标定报告。"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import time

from rescue_vision.config import load_runtime_config
from rescue_vision.motion import (
    ControllerProtocolError,
    OdometryImu,
    SensorFlags,
    parse_controller_frame,
)


_DEFAULT_DURATION_SECONDS = 30.0
_DEFAULT_WARMUP_SECONDS = 5.0
_DEFAULT_MIN_SAMPLES = 100
_POLL_TIMEOUT_SECONDS = 0.1


def _vector_mean(values: list[tuple[float, float, float]]) -> list[float]:
    if not values:
        raise ValueError("at least one vector sample is required")
    return [
        statistics.fmean(sample[index] for sample in values)
        for index in range(3)
    ]


def _vector_std(values: list[tuple[float, float, float]]) -> list[float]:
    if not values:
        raise ValueError("at least one vector sample is required")
    return [
        statistics.pstdev(sample[index] for sample in values)
        for index in range(3)
    ]


@dataclass(slots=True)
class StaticImuCalibration:
    """按质量门禁收集静止的传感器坐标系 IMU 样本。"""

    max_encoder_delta_count: int = 0
    reference_temperature_c: float = 25.0
    gyro_samples_rad_s: list[tuple[float, float, float]] = field(
        default_factory=list
    )
    accel_samples_mm_s2: list[tuple[float, float, float]] = field(
        default_factory=list
    )
    temperature_samples_c: list[float] = field(default_factory=list)
    sample_timestamps_us: list[int] = field(default_factory=list)
    rejected_reasons: dict[str, int] = field(default_factory=dict)
    _previous_encoder_counts: tuple[int, int] | None = field(
        default=None,
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.reference_temperature_c)):
            raise ValueError("reference_temperature_c must be finite.")
        self.reference_temperature_c = float(self.reference_temperature_c)
        if (
            isinstance(self.max_encoder_delta_count, bool)
            or not isinstance(self.max_encoder_delta_count, int)
            or self.max_encoder_delta_count < 0
        ):
            raise ValueError("max_encoder_delta_count must be non-negative.")

    @property
    def accepted_sample_count(self) -> int:
        return len(self.gyro_samples_rad_s)

    @property
    def rejected_sample_count(self) -> int:
        return sum(self.rejected_reasons.values())

    def _reject(self, reason: str) -> bool:
        self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1
        return False

    def observe(self, message: OdometryImu) -> bool:
        """消费一帧遥测；返回值表示是否纳入静止标定统计。"""

        if not isinstance(message, OdometryImu):
            raise TypeError(
                f"message must be OdometryImu, got {type(message).__name__}."
            )

        current_encoder_counts = (
            message.left_encoder_count,
            message.right_encoder_count,
        )
        previous_encoder_counts = self._previous_encoder_counts
        self._previous_encoder_counts = current_encoder_counts

        flags = message.sensor_flags
        if not flags & SensorFlags.IMU_VALID:
            return self._reject("imu_invalid")
        if flags & SensorFlags.SAMPLE_OVERRUN:
            return self._reject("sample_overrun")
        if flags & SensorFlags.GYRO_SATURATED:
            return self._reject("gyro_saturated")
        if flags & SensorFlags.ACCEL_SATURATED:
            return self._reject("accel_saturated")
        if not (
            flags & SensorFlags.LEFT_ENCODER_VALID
            and flags & SensorFlags.RIGHT_ENCODER_VALID
        ):
            return self._reject("encoder_invalid")
        if previous_encoder_counts is not None:
            if any(
                abs(current - previous) > self.max_encoder_delta_count
                for current, previous in zip(
                    current_encoder_counts,
                    previous_encoder_counts,
                )
            ):
                return self._reject("vehicle_moved")

        self.gyro_samples_rad_s.append(
            (
                message.gyro_x_urad_s / 1_000_000.0,
                message.gyro_y_urad_s / 1_000_000.0,
                message.gyro_z_urad_s / 1_000_000.0,
            )
        )
        self.accel_samples_mm_s2.append(
            (
                float(message.accel_x_mm_s2),
                float(message.accel_y_mm_s2),
                float(message.accel_z_mm_s2),
            )
        )
        self.temperature_samples_c.append(message.imu_temperature_cdeg / 100.0)
        self.sample_timestamps_us.append(message.sample_timestamp_us)
        return True

    def report(self) -> dict[str, object]:
        """返回可审查的报告及供人工复制的配置片段。"""

        if self.accepted_sample_count < 2:
            raise ValueError(
                "at least two valid stationary IMU samples are required; "
                f"got {self.accepted_sample_count}."
            )
        gyro_bias = _vector_mean(self.gyro_samples_rad_s)
        gyro_noise = _vector_std(self.gyro_samples_rad_s)
        accel_mean = _vector_mean(self.accel_samples_mm_s2)
        temperature_mean = statistics.fmean(self.temperature_samples_c)
        temperature_is_usable = any(
            abs(value) > 1e-9 for value in self.temperature_samples_c
        )
        recommended_temperature = (
            temperature_mean
            if temperature_is_usable
            else self.reference_temperature_c
        )
        timestamp_span_s = (
            max(self.sample_timestamps_us) - min(self.sample_timestamps_us)
        ) / 1_000_000.0
        return {
            "sample_count": self.accepted_sample_count,
            "rejected_sample_count": self.rejected_sample_count,
            "rejected_reasons": dict(sorted(self.rejected_reasons.items())),
            "sample_timestamp_span_s": timestamp_span_s,
            "temperature_mean_c": temperature_mean,
            "temperature_min_c": min(self.temperature_samples_c),
            "temperature_max_c": max(self.temperature_samples_c),
            "temperature_status": (
                "sample_mean"
                if temperature_is_usable
                else "zero_samples_use_configured_reference"
            ),
            "gyro_bias_rad_s_sensor_frame": gyro_bias,
            "gyro_noise_std_rad_s_sensor_frame": gyro_noise,
            "accel_mean_mm_s2_sensor_frame": accel_mean,
            "recommended_config": {
                "localization": {
                    "fusion": {
                        "imu_calibration": {
                            "reference_temperature_c": recommended_temperature,
                            "gyro_bias_rad_s": gyro_bias,
                            "gyro_bias_temperature_coefficient_rad_s_per_c": [
                                0.0,
                                0.0,
                                0.0,
                            ],
                        },
                        # Runtime config rejects zero; keep the physical result
                        # above and use only the schema minimum in the snippet.
                        "gyro_noise_std_rad_s": max(gyro_noise[2], 0.001),
                    }
                }
            },
        }


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite value greater than zero, got {value!r}"
        )
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer, got {value!r}"
        )
    return parsed


def _minimum_samples(value: str) -> int:
    parsed = _non_negative_int(value)
    if parsed < 2:
        raise argparse.ArgumentTypeError(
            f"expected an integer of at least two, got {value!r}"
        )
    return parsed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect stationary raw STM32 IMU samples. This tool is passive "
            "and never sends motion commands."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/runtime.yaml"),
        help="Runtime YAML path (default: configs/runtime.yaml).",
    )
    parser.add_argument(
        "--duration-seconds",
        type=_positive_float,
        default=_DEFAULT_DURATION_SECONDS,
        help=(
            "Stationary capture duration after warmup "
            f"(default: {_DEFAULT_DURATION_SECONDS})."
        ),
    )
    parser.add_argument(
        "--warmup-seconds",
        type=_non_negative_int,
        default=int(_DEFAULT_WARMUP_SECONDS),
        help=(
            "Discard samples during this warmup period "
            f"(default: {_DEFAULT_WARMUP_SECONDS:.0f})."
        ),
    )
    parser.add_argument(
        "--min-samples",
        type=_minimum_samples,
        default=_DEFAULT_MIN_SAMPLES,
        help=f"Minimum accepted samples (default: {_DEFAULT_MIN_SAMPLES}).",
    )
    parser.add_argument(
        "--max-encoder-delta-count",
        type=_non_negative_int,
        default=0,
        help=(
            "Maximum absolute encoder change between samples accepted as "
            "stationary (default: 0)."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("imu_static_calibration.json"),
        help="Calibration report path (default: imu_static_calibration.json).",
    )
    parser.add_argument(
        "--stationary-vehicle-confirmed",
        action="store_true",
        help="Confirm the robot is physically stationary for the whole capture.",
    )
    args = parser.parse_args()
    if not args.stationary_vehicle_confirmed:
        parser.error(
            "--stationary-vehicle-confirmed is required for IMU calibration"
        )
    return args


def main() -> None:
    args = _parse_args()
    config = load_runtime_config(args.config)
    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("uart.enabled must be true for IMU calibration.")

    accumulator = StaticImuCalibration(
        max_encoder_delta_count=args.max_encoder_delta_count,
        reference_temperature_c=(
            config.localization.fusion.imu_frame_calibration.reference_temperature_c
        ),
    )
    opened_ns = time.monotonic_ns()
    capture_start_ns = opened_ns + round(args.warmup_seconds * 1_000_000_000)
    deadline_ns = capture_start_ns + round(
        args.duration_seconds * 1_000_000_000
    )
    protocol_error_count = 0

    print(
        f"device={channel.device} warmup_s={args.warmup_seconds} "
        f"duration_s={args.duration_seconds:.1f}; "
        "do not touch or move the robot",
        flush=True,
    )
    try:
        with channel:
            while time.monotonic_ns() < deadline_ns:
                try:
                    frame = channel.receive_frame(timeout=_POLL_TIMEOUT_SECONDS)
                except TimeoutError:
                    continue
                try:
                    message = parse_controller_frame(frame)
                except ControllerProtocolError:
                    protocol_error_count += 1
                    continue
                if not isinstance(message, OdometryImu):
                    continue
                if time.monotonic_ns() >= capture_start_ns:
                    accumulator.observe(message)
    except KeyboardInterrupt:
        raise RuntimeError("calibration interrupted before report was written.")

    if accumulator.accepted_sample_count < args.min_samples:
        raise RuntimeError(
            "Not enough valid stationary samples: "
            f"{accumulator.accepted_sample_count} < {args.min_samples}; "
            f"rejected={accumulator.rejected_sample_count}, "
            f"protocol_errors={protocol_error_count}, "
            f"reasons={dict(sorted(accumulator.rejected_reasons.items()))}."
        )

    report = accumulator.report()
    report["protocol_error_count"] = protocol_error_count
    report["config_path"] = str(args.config)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
