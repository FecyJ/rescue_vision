"""标定 STM32 IMU，并可直接执行受监督的原地旋转复核。"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import statistics
import time

from rescue_vision.config import load_runtime_config
from rescue_vision.motion import (
    CarCommandReply,
    CarSystemStatus,
    CommandResult,
    OdometryImu,
    SensorFlags,
)


_DEFAULT_DURATION_SECONDS = 30.0
_DEFAULT_WARMUP_SECONDS = 5.0
_DEFAULT_MIN_SAMPLES = 100
_DEFAULT_ROTATION_ANGULAR_VELOCITY_RAD_S = 0.20
_DEFAULT_ROTATION_TIMEOUT_SECONDS = 60.0
_DEFAULT_ROTATION_SETTLE_SECONDS = 1.0
_CONTROL_PERIOD_SECONDS = 0.01

Vector3 = tuple[float, float, float]
Matrix3 = tuple[Vector3, Vector3, Vector3]


def _matrix_vector(matrix: Matrix3, vector: Vector3) -> Vector3:
    return tuple(
        sum(matrix[row][column] * vector[column] for column in range(3))
        for row in range(3)
    )  # type: ignore[return-value]


def _matrix_multiply(left: Matrix3, right: Matrix3) -> Matrix3:
    return tuple(
        tuple(
            sum(left[row][index] * right[index][column] for index in range(3))
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def sensor_to_robot_rotation_from_gravity(accel_mean: Vector3) -> Matrix3:
    """求把静止重力方向对齐到机器人 ``+z`` 的最小旋转。

    该计算只确定横滚/俯仰。绕重力轴的偏航在单一静止姿态中不可观测，
    因此这里选择与传感器水平轴最接近的最小旋转；若 IMU 绕 ``z`` 轴也
    发生了安装旋转，仍需提供已知航向基准后再补充偏航角。
    """

    norm = math.sqrt(sum(value * value for value in accel_mean))
    if not math.isfinite(norm) or norm <= 1e-9:
        raise ValueError(
            "stationary acceleration norm must be finite and greater than zero, "
            f"got {accel_mean!r}."
        )
    source = tuple(value / norm for value in accel_mean)
    target = (0.0, 0.0, 1.0)
    cross = (
        source[1] * target[2] - source[2] * target[1],
        source[2] * target[0] - source[0] * target[2],
        source[0] * target[1] - source[1] * target[0],
    )
    sine = math.sqrt(sum(value * value for value in cross))
    cosine = sum(source[index] * target[index] for index in range(3))
    if cosine < -1.0 + 1e-9:
        raise ValueError(
            "stationary acceleration points opposite to robot +z; "
            "check the IMU acceleration sign and robot attitude before calibration."
        )
    if sine <= 1e-9:
        return (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )

    skew: Matrix3 = (
        (0.0, -cross[2], cross[1]),
        (cross[2], 0.0, -cross[0]),
        (-cross[1], cross[0], 0.0),
    )
    skew_squared = _matrix_multiply(skew, skew)
    factor = (1.0 - cosine) / (sine * sine)
    return tuple(
        tuple(
            (1.0 if row == column else 0.0)
            + skew[row][column]
            + factor * skew_squared[row][column]
            for column in range(3)
        )
        for row in range(3)
    )  # type: ignore[return-value]


def _matrix_as_lists(matrix: Matrix3, *, digits: int = 9) -> list[list[float]]:
    return [
        [round(float(value), digits) for value in row]
        for row in matrix
    ]


def _vector_as_list(vector: Vector3) -> list[float]:
    return [float(value) for value in vector]


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
        accel_mean_tuple = tuple(accel_mean)  # type: ignore[assignment]
        sensor_to_robot_rotation = sensor_to_robot_rotation_from_gravity(
            accel_mean_tuple
        )
        rotated_gravity = _matrix_vector(
            sensor_to_robot_rotation,
            accel_mean_tuple,
        )
        gravity_norm = math.sqrt(
            sum(value * value for value in accel_mean_tuple)
        )
        tilt_angle_deg = math.degrees(
            math.acos(
                max(
                    -1.0,
                    min(1.0, accel_mean_tuple[2] / gravity_norm),
                )
            )
        )
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
            "gravity_norm_mm_s2": gravity_norm,
            "gravity_after_rotation_mm_s2_robot_frame": _vector_as_list(
                rotated_gravity
            ),
            "mount_tilt_angle_deg": tilt_angle_deg,
            "sensor_to_robot_rotation": _matrix_as_lists(
                sensor_to_robot_rotation
            ),
            "rotation_yaw_status": (
                "unobservable_from_single_static_attitude; "
                "minimum_rotation_used"
            ),
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
                            "sensor_to_robot_rotation": _matrix_as_lists(
                                sensor_to_robot_rotation
                            ),
                        },
                        # Runtime config rejects zero; keep the physical result
                        # above and use only the schema minimum in the snippet.
                        "gyro_noise_std_rad_s": max(gyro_noise[2], 0.001),
                    }
                }
            },
        }


@dataclass(frozen=True, slots=True)
class RotationObservation:
    """一次有效自转遥测样本。"""

    sample_timestamp_us: int
    heading_delta_rad: float
    gyro_rad_s_sensor_frame: Vector3


@dataclass(slots=True)
class RotationPhase:
    """保存一个方向的受监督原地旋转数据。"""

    direction: int
    baseline_left_count: int
    baseline_right_count: int
    baseline_sample_timestamp_us: int
    encoder_counts_per_revolution: int
    left_wheel_radius_m: float
    right_wheel_radius_m: float
    wheel_track_m: float
    observations: list[RotationObservation] = field(default_factory=list)
    rejected_reasons: dict[str, int] = field(default_factory=dict)

    def _reject(self, reason: str) -> None:
        self.rejected_reasons[reason] = self.rejected_reasons.get(reason, 0) + 1

    def _heading_delta(self, message: OdometryImu) -> float:
        left_distance_m = (
            (message.left_encoder_count - self.baseline_left_count)
            / self.encoder_counts_per_revolution
            * (2.0 * math.pi)
            * self.left_wheel_radius_m
        )
        right_distance_m = (
            (message.right_encoder_count - self.baseline_right_count)
            / self.encoder_counts_per_revolution
            * (2.0 * math.pi)
            * self.right_wheel_radius_m
        )
        # x 向前、y 向左；右轮路程减左轮路程对应逆时针正角速度。
        return (right_distance_m - left_distance_m) / self.wheel_track_m

    def observe(self, message: OdometryImu) -> bool:
        flags = message.sensor_flags
        if not flags & SensorFlags.IMU_VALID:
            self._reject("imu_invalid")
            return False
        if flags & SensorFlags.SAMPLE_OVERRUN:
            self._reject("sample_overrun")
            return False
        if flags & SensorFlags.GYRO_SATURATED:
            self._reject("gyro_saturated")
            return False
        if not (
            flags & SensorFlags.LEFT_ENCODER_VALID
            and flags & SensorFlags.RIGHT_ENCODER_VALID
        ):
            self._reject("encoder_invalid")
            return False
        if message.sample_timestamp_us <= self.baseline_sample_timestamp_us:
            self._reject("before_baseline")
            return False
        heading_delta_rad = self._heading_delta(message)
        if self.observations and message.sample_timestamp_us <= (
            self.observations[-1].sample_timestamp_us
        ):
            self._reject("timestamp_not_increasing")
            return False
        self.observations.append(
            RotationObservation(
                sample_timestamp_us=message.sample_timestamp_us,
                heading_delta_rad=heading_delta_rad,
                gyro_rad_s_sensor_frame=(
                    message.gyro_x_urad_s / 1_000_000.0,
                    message.gyro_y_urad_s / 1_000_000.0,
                    message.gyro_z_urad_s / 1_000_000.0,
                ),
            )
        )
        return True

    @property
    def progress_rad(self) -> float:
        if not self.observations:
            return 0.0
        return self.direction * self.observations[-1].heading_delta_rad


def analyze_rotation_phase(
    phase: RotationPhase,
    *,
    sensor_to_robot_rotation: Matrix3,
    gyro_bias_rad_s_sensor_frame: Vector3,
) -> dict[str, object]:
    """比较自转编码器角度与校正后三轴陀螺积分角度。"""

    if len(phase.observations) < 2:
        raise ValueError(
            "at least two valid rotation samples are required; "
            f"got {len(phase.observations)}."
        )
    integrated_yaw_rad = 0.0
    previous_timestamp_us: int | None = None
    previous_gyro_z_rad_s: float | None = None
    for observation in phase.observations:
        corrected = tuple(
            observation.gyro_rad_s_sensor_frame[index]
            - gyro_bias_rad_s_sensor_frame[index]
            for index in range(3)
        )  # type: ignore[assignment]
        gyro_robot = _matrix_vector(sensor_to_robot_rotation, corrected)
        if (
            previous_timestamp_us is not None
            and previous_gyro_z_rad_s is not None
        ):
            dt_s = (
                observation.sample_timestamp_us - previous_timestamp_us
            ) / 1_000_000.0
            if 0.0 < dt_s <= 0.2:
                integrated_yaw_rad += 0.5 * (
                    previous_gyro_z_rad_s + gyro_robot[2]
                ) * dt_s
        previous_timestamp_us = observation.sample_timestamp_us
        previous_gyro_z_rad_s = gyro_robot[2]

    encoder_yaw_rad = phase.observations[-1].heading_delta_rad
    if abs(encoder_yaw_rad) <= 1e-6:
        raise ValueError("rotation did not produce a measurable encoder angle.")
    gyro_to_encoder_scale = integrated_yaw_rad / encoder_yaw_rad
    return {
        "direction": "left" if phase.direction > 0 else "right",
        "sample_count": len(phase.observations),
        "rejected_reasons": dict(sorted(phase.rejected_reasons.items())),
        "encoder_yaw_rad": encoder_yaw_rad,
        "gyro_integrated_yaw_rad": integrated_yaw_rad,
        "gyro_to_encoder_scale": gyro_to_encoder_scale,
        "gyro_z_sign": 1 if gyro_to_encoder_scale >= 0.0 else -1,
    }


def combine_rotation_reports(
    reports: list[dict[str, object]],
) -> dict[str, object]:
    """合并正反转结果，给出极性和比例诊断。"""

    if not reports:
        raise ValueError("at least one rotation report is required.")
    scales = [float(report["gyro_to_encoder_scale"]) for report in reports]
    weights = [abs(float(report["encoder_yaw_rad"])) for report in reports]
    total_weight = sum(weights)
    weighted_scale = sum(scale * weight for scale, weight in zip(scales, weights)) / (
        total_weight if total_weight > 0.0 else len(scales)
    )
    sign_consistent = all(
        (scale >= 0.0) == (weighted_scale >= 0.0) for scale in scales
    )
    result = {
        "phase_reports": reports,
        "gyro_z_sign": 1 if weighted_scale >= 0.0 else -1,
        "gyro_to_encoder_scale": weighted_scale,
        "gyro_scale_correction_if_needed": (
            1.0 / abs(weighted_scale)
            if abs(weighted_scale) > 1e-9
            else None
        ),
        "sign_consistent": sign_consistent,
    }
    if not sign_consistent:
        result["warning"] = "forward and reverse rotation polarity disagrees."
    return result


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


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a finite non-negative value, got {value!r}"
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
            "Calibrate stationary raw STM32 IMU samples and optionally run "
            "supervised in-place rotations to validate gyro polarity."
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
        type=_non_negative_float,
        default=_DEFAULT_WARMUP_SECONDS,
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
        "--rotation-angle-deg",
        type=_positive_float,
        default=None,
        help=(
            "Run an active in-place calibration rotation in each selected "
            "direction until this encoder angle is reached; omit for a "
            "passive stationary-only run."
        ),
    )
    parser.add_argument(
        "--rotation-direction",
        choices=("left", "right", "both"),
        default="both",
        help="Active rotation direction (default: both).",
    )
    parser.add_argument(
        "--rotation-angular-velocity-rad-s",
        type=_positive_float,
        default=_DEFAULT_ROTATION_ANGULAR_VELOCITY_RAD_S,
        help=(
            "Positive magnitude for active in-place rotation, in rad/s "
            f"(default: {_DEFAULT_ROTATION_ANGULAR_VELOCITY_RAD_S})."
        ),
    )
    parser.add_argument(
        "--rotation-timeout-seconds",
        type=_positive_float,
        default=_DEFAULT_ROTATION_TIMEOUT_SECONDS,
        help=(
            "Maximum time for each active rotation, in seconds "
            f"(default: {_DEFAULT_ROTATION_TIMEOUT_SECONDS})."
        ),
    )
    parser.add_argument(
        "--rotation-settle-seconds",
        type=_non_negative_float,
        default=_DEFAULT_ROTATION_SETTLE_SECONDS,
        help=(
            "Stationary settling time after each active rotation, in seconds "
            f"(default: {_DEFAULT_ROTATION_SETTLE_SECONDS})."
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
        help=(
            "Confirm the robot is stationary during static capture and that "
            "any active rotation is fully supervised."
        ),
    )
    args = parser.parse_args()
    if not args.stationary_vehicle_confirmed:
        parser.error(
            "--stationary-vehicle-confirmed is required for IMU calibration"
        )
    return args


def _sleep_control_period() -> None:
    time.sleep(_CONTROL_PERIOD_SECONDS)


def _rotation_directions(value: str) -> tuple[int, ...]:
    if value == "left":
        return (1,)
    if value == "right":
        return (-1,)
    if value == "both":
        return (1, -1)
    raise ValueError(f"unsupported rotation direction {value!r}.")


def _print_matrix(matrix: object) -> None:
    if not isinstance(matrix, list):
        return
    print("sensor_to_robot_rotation (sensor frame -> robot frame):")
    for row in matrix:
        print("  [" + ", ".join(f"{float(value): .9f}" for value in row) + "]")


def _print_report_summary(report: dict[str, object]) -> None:
    _print_matrix(report.get("sensor_to_robot_rotation"))
    print(
        "gravity_after_rotation_mm_s2_robot_frame="
        f"{report['gravity_after_rotation_mm_s2_robot_frame']} "
        f"tilt_deg={float(report['mount_tilt_angle_deg']):.3f} "
        f"yaw_status={report['rotation_yaw_status']}",
        flush=True,
    )
    active_rotation = report.get("active_rotation")
    if isinstance(active_rotation, dict):
        print(
            "active_rotation="
            f"gyro_z_sign={active_rotation['gyro_z_sign']} "
            f"gyro_to_encoder_scale={float(active_rotation['gyro_to_encoder_scale']):.6f} "
            f"sign_consistent={active_rotation['sign_consistent']}",
            flush=True,
        )
    try:
        import yaml

        print("recommended_config:")
        print(
            yaml.safe_dump(
                report["recommended_config"],
                allow_unicode=True,
                sort_keys=False,
            ),
            end="",
            flush=True,
        )
    except ImportError:
        print(
            "recommended_config_json="
            + json.dumps(
                report["recommended_config"],
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


def main() -> None:
    args = _parse_args()
    config = load_runtime_config(args.config)
    channel = config.uart.build_channel()
    if channel is None:
        raise RuntimeError("uart.enabled must be true for IMU calibration.")
    if not config.motion.enabled:
        raise RuntimeError("motion.enabled must be true for IMU calibration.")
    controller = config.motion.build_controller(channel)
    if controller is None:
        raise RuntimeError("the configured motion controller could not be created.")

    odometry_config = config.motion.odometry
    if args.rotation_angle_deg is not None:
        if not odometry_config.enabled:
            raise RuntimeError(
                "motion.odometry.enabled must be true for active rotation."
            )
        if args.rotation_angular_velocity_rad_s > (
            config.motion.max_angular_velocity_rad_s
        ):
            raise ValueError(
                "rotation angular velocity exceeds configured limit: "
                f"{args.rotation_angular_velocity_rad_s} > "
                f"{config.motion.max_angular_velocity_rad_s} rad/s."
            )
        if config.motion.wheel_track_m is None:
            raise RuntimeError("motion.wheel_track_m is required for active rotation.")
        assert odometry_config.encoder_counts_per_revolution is not None
        assert odometry_config.left_wheel_radius_mm is not None
        assert odometry_config.right_wheel_radius_mm is not None

    accumulator = StaticImuCalibration(
        max_encoder_delta_count=args.max_encoder_delta_count,
        reference_temperature_c=(
            config.localization.fusion.imu_frame_calibration.reference_temperature_c
        ),
    )
    active_static = False
    active_rotation: RotationPhase | None = None
    latest_odometry: OdometryImu | None = None

    def consume(message: object) -> None:
        nonlocal latest_odometry
        if isinstance(message, CarCommandReply):
            if message.result is not CommandResult.ACCEPTED:
                raise RuntimeError(
                    "STM32 rejected command: "
                    f"{message.command_type.name.lower()}="
                    f"{message.result.name.lower()}"
                )
            return
        if isinstance(message, CarSystemStatus):
            if message.emergency_stop_latched:
                raise RuntimeError("STM32 emergency stop is latched.")
            return
        if not isinstance(message, OdometryImu):
            return
        latest_odometry = message
        if active_static:
            accumulator.observe(message)
        if active_rotation is not None:
            active_rotation.observe(message)

    def pump_controller() -> None:
        controller.update(now_ns=time.monotonic_ns())
        for message in controller.drain_messages():
            consume(message)

    print(
        f"device={channel.device} warmup_s={args.warmup_seconds} "
        f"stationary_duration_s={args.duration_seconds:.1f}; "
        "keep the robot completely still during this phase",
        flush=True,
    )
    try:
        with ExitStack() as stack:
            stack.enter_context(channel)
            # ExitStack callbacks run before the channel context is closed.
            stack.callback(controller.soft_brake)
            controller.synchronize(
                timeout_s=config.motion.synchronization_timeout_s,
                on_message=consume,
            )
            warmup_deadline_ns = time.monotonic_ns() + round(
                args.warmup_seconds * 1_000_000_000
            )
            while time.monotonic_ns() < warmup_deadline_ns:
                pump_controller()
                _sleep_control_period()

            active_static = True
            stationary_deadline_ns = time.monotonic_ns() + round(
                args.duration_seconds * 1_000_000_000
            )
            while time.monotonic_ns() < stationary_deadline_ns:
                pump_controller()
                _sleep_control_period()
            active_static = False

            if accumulator.accepted_sample_count < args.min_samples:
                raise RuntimeError(
                    "Not enough valid stationary samples: "
                    f"{accumulator.accepted_sample_count} < {args.min_samples}; "
                    f"rejected={accumulator.rejected_sample_count}; "
                    f"reasons={dict(sorted(accumulator.rejected_reasons.items()))}."
                )

            report = accumulator.report()
            rotation_reports: list[dict[str, object]] = []
            if args.rotation_angle_deg is not None:
                assert config.motion.wheel_track_m is not None
                assert odometry_config.encoder_counts_per_revolution is not None
                assert odometry_config.left_wheel_radius_mm is not None
                assert odometry_config.right_wheel_radius_mm is not None
                rotation_matrix = tuple(
                    tuple(float(value) for value in row)
                    for row in report["sensor_to_robot_rotation"]  # type: ignore[index]
                )  # type: ignore[assignment]
                gyro_bias = tuple(
                    float(value)
                    for value in report["gyro_bias_rad_s_sensor_frame"]  # type: ignore[index]
                )  # type: ignore[assignment]
                target_angle_rad = math.radians(args.rotation_angle_deg)
                for direction in _rotation_directions(args.rotation_direction):
                    if latest_odometry is None:
                        raise RuntimeError(
                            "no valid odometry sample was received before rotation."
                        )
                    phase = RotationPhase(
                        direction=direction,
                        baseline_left_count=latest_odometry.left_encoder_count,
                        baseline_right_count=latest_odometry.right_encoder_count,
                        baseline_sample_timestamp_us=(
                            latest_odometry.sample_timestamp_us
                        ),
                        encoder_counts_per_revolution=(
                            odometry_config.encoder_counts_per_revolution
                        ),
                        left_wheel_radius_m=(
                            odometry_config.left_wheel_radius_mm / 1000.0
                        ),
                        right_wheel_radius_m=(
                            odometry_config.right_wheel_radius_mm / 1000.0
                        ),
                        wheel_track_m=config.motion.wheel_track_m,
                    )
                    active_rotation = phase
                    direction_name = "left" if direction > 0 else "right"
                    print(
                        f"rotation_start=direction:{direction_name} "
                        f"target_deg:{args.rotation_angle_deg:.1f} "
                        f"angular_velocity_rad_s:{direction * args.rotation_angular_velocity_rad_s:+.3f}; "
                        "keep the physical emergency stop ready",
                        flush=True,
                    )
                    phase_start_ns = time.monotonic_ns()
                    phase_deadline_ns = phase_start_ns + round(
                        args.rotation_timeout_seconds * 1_000_000_000
                    )
                    while time.monotonic_ns() < phase_deadline_ns:
                        pump_controller()
                        if phase.progress_rad >= target_angle_rad:
                            break
                        if (
                            len(phase.observations) >= 20
                            and phase.progress_rad < -0.1
                        ):
                            raise RuntimeError(
                                "encoder yaw sign is opposite to the requested "
                                f"{direction_name} rotation; check encoder polarity."
                            )
                        controller.drive_wheel_limited(
                            0.0,
                            direction * args.rotation_angular_velocity_rad_s,
                        )
                        _sleep_control_period()
                    else:
                        raise RuntimeError(
                            f"{direction_name} rotation timed out at "
                            f"{math.degrees(phase.progress_rad):.1f} degrees."
                        )

                    active_rotation = None
                    controller.drive_wheel_limited(0.0, 0.0)
                    settle_deadline_ns = time.monotonic_ns() + round(
                        args.rotation_settle_seconds * 1_000_000_000
                    )
                    while time.monotonic_ns() < settle_deadline_ns:
                        pump_controller()
                        _sleep_control_period()
                    phase_report = analyze_rotation_phase(
                        phase,
                        sensor_to_robot_rotation=rotation_matrix,
                        gyro_bias_rad_s_sensor_frame=gyro_bias,
                    )
                    rotation_reports.append(phase_report)
                    print(
                        f"rotation_done=direction:{direction_name} "
                        f"encoder_deg:{math.degrees(float(phase_report['encoder_yaw_rad'])):.2f} "
                        f"gyro_deg:{math.degrees(float(phase_report['gyro_integrated_yaw_rad'])):.2f} "
                        f"scale:{float(phase_report['gyro_to_encoder_scale']):.6f}",
                        flush=True,
                    )

            if rotation_reports:
                active_rotation_report = combine_rotation_reports(rotation_reports)
                active_rotation_report["requested_angle_deg"] = (
                    args.rotation_angle_deg
                )
                active_rotation_report["requested_direction"] = (
                    args.rotation_direction
                )
                report["active_rotation"] = active_rotation_report
                report["recommended_config"]["motion"] = {  # type: ignore[index]
                    "odometry": {
                        "gyro_z_sign": active_rotation_report["gyro_z_sign"]
                    }
                }
    except KeyboardInterrupt as exc:
        raise RuntimeError(
            "calibration interrupted before report was written; the controller "
            "will be softly braked."
        ) from exc
    finally:
        active_static = False
        active_rotation = None

    report["config_path"] = str(args.config)
    report["active_motion_control"] = args.rotation_angle_deg is not None
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _print_report_summary(report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"report={args.output}", flush=True)


if __name__ == "__main__":
    main()
