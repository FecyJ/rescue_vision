"""Continuous differential-drive, IMU and delayed visual pose fusion."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from enum import Enum
import math
import threading

import numpy as np

from rescue_vision.geometry.types import FieldPoint
from rescue_vision.localization.types import (
    CenterCrossPoseObservation,
    CenterCrossSelectionSource,
    FieldPose2D,
    normalize_angle,
)
from rescue_vision.motion.protocol import OdometryImu, SensorFlags


class FusionQuality(str, Enum):
    INITIALIZING = "initializing"
    FUSED = "fused"
    WHEEL_ONLY = "wheel_only"
    DROPPED_TELEMETRY = "dropped_telemetry"
    TILT_DETECTED = "tilt_detected"
    IMPACT_DETECTED = "impact_detected"
    VISUAL_REJECTED = "visual_rejected"
    CONTINUITY_LOST = "continuity_lost"
    STALE = "stale"
    INTERPOLATED_IMU = "interpolated_imu"


def _finite_vector3(value: object, name: str) -> tuple[float, float, float]:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite three-element vector.") from exc
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must be a finite three-element vector.")
    return tuple(float(item) for item in vector)  # type: ignore[return-value]


def _finite_matrix3(value: object, name: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite 3x3 matrix.") from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must be a finite 3x3 matrix.")
    return matrix


def _invertible_matrix3(
    value: object, name: str
) -> tuple[tuple[float, float, float], ...]:
    matrix = _finite_matrix3(value, name)
    if math.isclose(float(np.linalg.det(matrix)), 0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be invertible.")
    return tuple(tuple(float(item) for item in row) for row in matrix)


@dataclass(frozen=True, slots=True)
class ImuFrameCalibration:
    """Complete sensor-frame IMU calibration applied once on the Pi."""

    reference_temperature_c: float = 25.0
    gyro_bias_rad_s: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gyro_bias_temperature_coefficient_rad_s_per_c: tuple[float, float, float] = (
        0.0,
        0.0,
        0.0,
    )
    gyro_cross_axis_scale: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    accel_bias_mm_s2: tuple[float, float, float] = (0.0, 0.0, 0.0)
    accel_bias_temperature_coefficient_mm_s2_per_c: tuple[
        float, float, float
    ] = (0.0, 0.0, 0.0)
    accel_cross_axis_scale: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    sensor_to_robot_rotation: tuple[
        tuple[float, float, float],
        tuple[float, float, float],
        tuple[float, float, float],
    ] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    def __post_init__(self) -> None:
        reference_temperature_c = float(self.reference_temperature_c)
        if not math.isfinite(reference_temperature_c):
            raise ValueError("reference_temperature_c must be finite.")
        object.__setattr__(self, "reference_temperature_c", reference_temperature_c)
        for name in (
            "gyro_bias_rad_s",
            "gyro_bias_temperature_coefficient_rad_s_per_c",
            "accel_bias_mm_s2",
            "accel_bias_temperature_coefficient_mm_s2_per_c",
        ):
            object.__setattr__(self, name, _finite_vector3(getattr(self, name), name))

        gyro_scale = _invertible_matrix3(
            self.gyro_cross_axis_scale, "gyro_cross_axis_scale"
        )
        accel_scale = _invertible_matrix3(
            self.accel_cross_axis_scale, "accel_cross_axis_scale"
        )
        matrix = _finite_matrix3(
            self.sensor_to_robot_rotation, "sensor_to_robot_rotation"
        )
        if not np.allclose(
            matrix @ matrix.T,
            np.eye(3, dtype=np.float64),
            atol=1e-6,
            rtol=1e-6,
        ):
            raise ValueError(
                "sensor_to_robot_rotation must be orthonormal."
            )
        determinant = float(np.linalg.det(matrix))
        if not math.isclose(determinant, 1.0, abs_tol=1e-6):
            raise ValueError(
                "sensor_to_robot_rotation must have determinant +1, "
                f"got {determinant!r}."
            )
        object.__setattr__(
            self,
            "sensor_to_robot_rotation",
            tuple(tuple(float(value) for value in row) for row in matrix),
        )
        object.__setattr__(self, "gyro_cross_axis_scale", gyro_scale)
        object.__setattr__(self, "accel_cross_axis_scale", accel_scale)


@dataclass(frozen=True, slots=True)
class OdometryCalibration:
    encoder_counts_per_revolution: int
    left_wheel_radius_mm: float
    right_wheel_radius_mm: float
    gyro_z_sign: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.encoder_counts_per_revolution, bool)
            or not isinstance(self.encoder_counts_per_revolution, int)
            or self.encoder_counts_per_revolution <= 0
        ):
            raise ValueError("encoder_counts_per_revolution must be a positive integer.")
        for name in ("left_wheel_radius_mm", "right_wheel_radius_mm"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        if (
            isinstance(self.gyro_z_sign, bool)
            or not isinstance(self.gyro_z_sign, int)
            or self.gyro_z_sign not in {-1, 1}
        ):
            raise ValueError("gyro_z_sign must be exactly -1 or 1.")


@dataclass(frozen=True, slots=True)
class FusionConfig:
    enabled: bool
    initial_pose: FieldPose2D
    initial_position_uncertainty_mm: float
    initial_heading_uncertainty_rad: float
    initial_confidence: float
    encoder_distance_noise_fraction: float
    encoder_heading_noise_std_rad: float
    gyro_noise_std_rad_s: float
    gyro_bias_random_walk_std_rad_s_per_sqrt_s: float
    stationary_gyro_noise_std_rad_s: float
    stationary_encoder_delta_count: int
    allow_wheel_only: bool
    wheel_only_covariance_scale: float
    dropped_sample_covariance_scale: float
    max_sample_interval_ms: float
    max_telemetry_age_ms: float
    max_encoder_speed_mm_s: float
    max_visual_alignment_error_ms: float
    visual_innovation_gate: float
    history_duration_ms: float
    max_tilt_deg: float
    impact_accel_threshold_mm_s2: float
    imu_frame_calibration: ImuFrameCalibration = field(
        default_factory=ImuFrameCalibration
    )
    max_interpolated_overrun_samples: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool) or not isinstance(
            self.allow_wheel_only, bool
        ):
            raise ValueError("enabled and allow_wheel_only must be booleans.")
        if not isinstance(self.initial_pose, FieldPose2D):
            raise ValueError("initial_pose must be a FieldPose2D.")
        if not isinstance(self.imu_frame_calibration, ImuFrameCalibration):
            raise ValueError(
                "imu_frame_calibration must be an ImuFrameCalibration."
            )
        positive = (
            "initial_position_uncertainty_mm",
            "initial_heading_uncertainty_rad",
            "encoder_heading_noise_std_rad",
            "gyro_noise_std_rad_s",
            "gyro_bias_random_walk_std_rad_s_per_sqrt_s",
            "stationary_gyro_noise_std_rad_s",
            "wheel_only_covariance_scale",
            "dropped_sample_covariance_scale",
            "max_sample_interval_ms",
            "max_telemetry_age_ms",
            "max_encoder_speed_mm_s",
            "max_visual_alignment_error_ms",
            "visual_innovation_gate",
            "history_duration_ms",
            "max_tilt_deg",
            "impact_accel_threshold_mm_s2",
        )
        for name in positive:
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive.")
        fraction = float(self.encoder_distance_noise_fraction)
        if not math.isfinite(fraction) or fraction < 0.0:
            raise ValueError("encoder_distance_noise_fraction must be non-negative.")
        if not 0.0 <= float(self.initial_confidence) <= 1.0:
            raise ValueError("initial_confidence must be in [0, 1].")
        if (
            isinstance(self.stationary_encoder_delta_count, bool)
            or not isinstance(self.stationary_encoder_delta_count, int)
            or self.stationary_encoder_delta_count < 0
        ):
            raise ValueError("stationary_encoder_delta_count must be non-negative.")
        if self.history_duration_ms < self.max_visual_alignment_error_ms:
            raise ValueError(
                "history_duration_ms must be at least max_visual_alignment_error_ms."
            )
        if (
            isinstance(self.max_interpolated_overrun_samples, bool)
            or not isinstance(self.max_interpolated_overrun_samples, int)
            or not 0 <= self.max_interpolated_overrun_samples <= 1
        ):
            raise ValueError(
                "max_interpolated_overrun_samples must be 0 or 1, got "
                f"{self.max_interpolated_overrun_samples!r}."
            )
        if self.max_tilt_deg >= 180.0:
            raise ValueError("max_tilt_deg must be less than 180 degrees.")


@dataclass(frozen=True, slots=True)
class FusedPoseEstimate:
    pose: FieldPose2D | None
    estimate_timestamp_ns: int | None
    position_uncertainty_mm: float | None
    heading_uncertainty_rad: float | None
    confidence: float
    anchor_source: str | None
    quality: frozenset[FusionQuality]


@dataclass(frozen=True, slots=True)
class VisualFusionResult:
    accepted: bool
    alignment_error_ns: int | None
    innovation_mahalanobis: float | None


@dataclass(frozen=True, slots=True)
class _Prediction:
    distance_mm: float
    encoder_heading_rad: float
    gyro_z_rad_s: float | None
    dt_s: float
    covariance_scale: float
    qualities: frozenset[FusionQuality]
    stationary: bool


@dataclass(slots=True)
class _HistoryEntry:
    timestamp_ns: int
    state: np.ndarray
    covariance: np.ndarray
    prediction: _Prediction | None
    qualities: frozenset[FusionQuality]


class OdometryImuFusion:
    """Thread-safe continuous field-pose estimator with bounded replay history."""

    def __init__(
        self,
        config: FusionConfig,
        calibration: OdometryCalibration,
        *,
        wheel_track_m: float,
    ) -> None:
        if not isinstance(config, FusionConfig):
            raise TypeError("config must be a FusionConfig.")
        if not config.enabled:
            raise ValueError("OdometryImuFusion requires config.enabled=true.")
        if not isinstance(calibration, OdometryCalibration):
            raise TypeError("calibration must be an OdometryCalibration.")
        track_mm = float(wheel_track_m) * 1000.0
        if not math.isfinite(track_mm) or track_mm <= 0.0:
            raise ValueError("wheel_track_m must be finite and positive.")
        self.config = config
        self.calibration = calibration
        self._track_mm = track_mm
        self._imu_rotation = np.asarray(
            config.imu_frame_calibration.sensor_to_robot_rotation,
            dtype=np.float64,
        )
        self._gyro_cross_axis_scale = np.asarray(
            config.imu_frame_calibration.gyro_cross_axis_scale,
            dtype=np.float64,
        )
        self._accel_cross_axis_scale = np.asarray(
            config.imu_frame_calibration.accel_cross_axis_scale,
            dtype=np.float64,
        )
        self._gyro_bias = np.asarray(
            config.imu_frame_calibration.gyro_bias_rad_s,
            dtype=np.float64,
        )
        self._gyro_bias_temperature_coefficient = np.asarray(
            config.imu_frame_calibration.gyro_bias_temperature_coefficient_rad_s_per_c,
            dtype=np.float64,
        )
        self._accel_bias = np.asarray(
            config.imu_frame_calibration.accel_bias_mm_s2,
            dtype=np.float64,
        )
        self._accel_bias_temperature_coefficient = np.asarray(
            config.imu_frame_calibration.accel_bias_temperature_coefficient_mm_s2_per_c,
            dtype=np.float64,
        )
        self._lock = threading.RLock()
        self._history: deque[_HistoryEntry] = deque()
        self._last_sample: OdometryImu | None = None
        self._pending_overrun: OdometryImu | None = None
        self._last_host_timestamp_ns: int | None = None
        self._clock_offsets_ns: deque[int] = deque(maxlen=128)
        self._initial_pose_available = True
        self._anchor_source: str | None = None
        self._anchor_confidence = 0.0
        self._status = frozenset({FusionQuality.INITIALIZING})

    def reset(self, reason: str) -> None:
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be a non-empty string.")
        with self._lock:
            self._lose_continuity()

    def submit_odometry(self, message: OdometryImu) -> FusedPoseEstimate:
        if not isinstance(message, OdometryImu):
            raise TypeError("message must be an OdometryImu.")
        with self._lock:
            if message.sensor_flags & SensorFlags.SAMPLE_OVERRUN:
                return self._submit_overrun(message)
            host_timestamp_ns = self._map_controller_time(message)
            previous = self._last_sample
            if self._pending_overrun is not None:
                if previous is None:
                    self._lose_continuity()
                    return self._estimate(message.received_timestamp_ns)
                return self._recover_overrun(
                    previous,
                    self._pending_overrun,
                    message,
                    host_timestamp_ns,
                )
            if previous is None:
                required = (
                    SensorFlags.LEFT_ENCODER_VALID
                    | SensorFlags.RIGHT_ENCODER_VALID
                )
                if message.sensor_flags & required != required:
                    self._clock_offsets_ns.clear()
                    self._last_host_timestamp_ns = None
                    return self._estimate(message.received_timestamp_ns)
                current_entry = self._state_entry()
                if (
                    current_entry is not None
                    and host_timestamp_ns < current_entry.timestamp_ns
                ):
                    self._clock_offsets_ns.clear()
                    self._last_host_timestamp_ns = None
                    return self._estimate(message.received_timestamp_ns)
                self._last_sample = message
                self._last_host_timestamp_ns = host_timestamp_ns
                if self._initial_pose_available and self._state_entry() is None:
                    self._initialize(
                        self.config.initial_pose,
                        host_timestamp_ns,
                        self.config.initial_position_uncertainty_mm,
                        self.config.initial_heading_uncertainty_rad,
                        "configured_start",
                        self.config.initial_confidence,
                    )
                    self._initial_pose_available = False
                elif current_entry is not None:
                    self._append_baseline(host_timestamp_ns)
                return self._estimate(message.received_timestamp_ns)

            discontinuity = self._validate_continuity(previous, message)
            if discontinuity is not None:
                self._lose_continuity()
                self._last_sample = message
                self._last_host_timestamp_ns = self._map_controller_time(message)
                return self._estimate(message.received_timestamp_ns)

            dt_s = (message.sample_timestamp_us - previous.sample_timestamp_us) / 1e6
            left_delta = message.left_encoder_count - previous.left_encoder_count
            right_delta = message.right_encoder_count - previous.right_encoder_count
            left_mm = self._count_distance(left_delta, left=True)
            right_mm = self._count_distance(right_delta, left=False)
            if max(abs(left_mm), abs(right_mm)) / dt_s > self.config.max_encoder_speed_mm_s:
                self._lose_continuity()
                self._last_sample = message
                self._last_host_timestamp_ns = self._map_controller_time(message)
                return self._estimate(message.received_timestamp_ns)

            sequence_delta = (message.telemetry_sequence - previous.telemetry_sequence) & 0xFFFF
            covariance_scale = 1.0
            qualities: set[FusionQuality] = set()
            if sequence_delta > 1:
                covariance_scale *= self.config.dropped_sample_covariance_scale
                qualities.add(FusionQuality.DROPPED_TELEMETRY)
            flags = message.sensor_flags
            imu_usable = bool(flags & SensorFlags.IMU_VALID) and not bool(
                flags & SensorFlags.GYRO_SATURATED
            )
            if not imu_usable:
                if not self.config.allow_wheel_only:
                    self._lose_continuity()
                    self._last_sample = message
                    self._last_host_timestamp_ns = host_timestamp_ns
                    return self._estimate(message.received_timestamp_ns)
                covariance_scale *= self.config.wheel_only_covariance_scale
                qualities.add(FusionQuality.WHEEL_ONLY)
            gyro_robot: np.ndarray | None = None
            if flags & SensorFlags.IMU_VALID:
                gyro_robot, accel_robot = self._calibrated_imu_vectors(message)
                qualities.update(
                    self._acceleration_quality(
                        accel_robot,
                        saturated=bool(flags & SensorFlags.ACCEL_SATURATED),
                    )
                )
            elif flags & SensorFlags.ACCEL_SATURATED:
                qualities.add(FusionQuality.IMPACT_DETECTED)
            if (
                FusionQuality.TILT_DETECTED in qualities
                or FusionQuality.IMPACT_DETECTED in qualities
            ):
                covariance_scale *= self.config.wheel_only_covariance_scale

            prediction = _Prediction(
                distance_mm=(left_mm + right_mm) / 2.0,
                encoder_heading_rad=(right_mm - left_mm) / self._track_mm,
                gyro_z_rad_s=(
                    float(gyro_robot[2])
                    if imu_usable and gyro_robot is not None
                    else None
                ),
                dt_s=dt_s,
                covariance_scale=covariance_scale,
                qualities=frozenset(qualities),
                stationary=(
                    abs(left_delta) <= self.config.stationary_encoder_delta_count
                    and abs(right_delta) <= self.config.stationary_encoder_delta_count
                ),
            )
            current = self._state_entry()
            if current is not None:
                self._append_prediction(host_timestamp_ns, prediction)
                self._status = self._history[-1].qualities
            self._last_sample = message
            self._last_host_timestamp_ns = host_timestamp_ns
            return self._estimate(message.received_timestamp_ns)

    def _submit_overrun(self, message: OdometryImu) -> FusedPoseEstimate:
        """Buffer one bad IMU period until the following valid sample arrives."""

        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
        previous = self._last_sample
        if (
            self.config.max_interpolated_overrun_samples == 0
            or message.sensor_flags & required != required
            or self._pending_overrun is not None
        ):
            self._lose_continuity()
            return self._estimate(message.received_timestamp_ns)
        if previous is None:
            # There is no valid IMU endpoint from which to interpolate.  Keep the
            # configured initial pose available and use the next clean sample as
            # the first baseline.
            self._clock_offsets_ns.clear()
            self._last_host_timestamp_ns = None
            return self._estimate(message.received_timestamp_ns)
        sequence_delta = (
            message.telemetry_sequence - previous.telemetry_sequence
        ) & 0xFFFF
        if (
            sequence_delta != 1
            or message.received_timestamp_ns <= previous.received_timestamp_ns
        ):
            self._lose_continuity()
            return self._estimate(message.received_timestamp_ns)
        dt_s = (
            message.received_timestamp_ns - previous.received_timestamp_ns
        ) / 1_000_000_000.0
        left_mm = self._count_distance(
            message.left_encoder_count - previous.left_encoder_count,
            left=True,
        )
        right_mm = self._count_distance(
            message.right_encoder_count - previous.right_encoder_count,
            left=False,
        )
        if (
            dt_s <= 0.0
            or max(abs(left_mm), abs(right_mm)) / dt_s
            > self.config.max_encoder_speed_mm_s
        ):
            self._lose_continuity()
            return self._estimate(message.received_timestamp_ns)
        self._pending_overrun = message
        self._status = frozenset({FusionQuality.INTERPOLATED_IMU})
        return self._estimate(message.received_timestamp_ns)

    def _recover_overrun(
        self,
        previous: OdometryImu,
        overrun: OdometryImu,
        current: OdometryImu,
        host_timestamp_ns: int,
    ) -> FusedPoseEstimate:
        """Replay the two half-periods around one overrun with linear IMU data."""

        discontinuity = self._validate_continuity(previous, current)
        if discontinuity is not None:
            self._lose_continuity()
            return self._estimate(current.received_timestamp_ns)
        previous_to_overrun = (
            overrun.telemetry_sequence - previous.telemetry_sequence
        ) & 0xFFFF
        overrun_to_current = (
            current.telemetry_sequence - overrun.telemetry_sequence
        ) & 0xFFFF
        if previous_to_overrun != 1 or overrun_to_current != 1:
            self._lose_continuity()
            return self._estimate(current.received_timestamp_ns)
        if (
            overrun.received_timestamp_ns <= previous.received_timestamp_ns
            or current.received_timestamp_ns <= overrun.received_timestamp_ns
        ):
            self._lose_continuity()
            return self._estimate(current.received_timestamp_ns)
        previous_host_timestamp_ns = self._last_host_timestamp_ns
        if (
            previous_host_timestamp_ns is None
            or host_timestamp_ns <= previous_host_timestamp_ns
        ):
            self._lose_continuity()
            return self._estimate(current.received_timestamp_ns)

        total_dt_s = (
            current.sample_timestamp_us - previous.sample_timestamp_us
        ) / 1_000_000.0
        if total_dt_s <= 0.0:
            self._lose_continuity()
            return self._estimate(current.received_timestamp_ns)
        segment_dt_s = total_dt_s / 2.0
        first_left_mm = self._count_distance(
            overrun.left_encoder_count - previous.left_encoder_count,
            left=True,
        )
        first_right_mm = self._count_distance(
            overrun.right_encoder_count - previous.right_encoder_count,
            left=False,
        )
        second_left_mm = self._count_distance(
            current.left_encoder_count - overrun.left_encoder_count,
            left=True,
        )
        second_right_mm = self._count_distance(
            current.right_encoder_count - overrun.right_encoder_count,
            left=False,
        )
        if (
            segment_dt_s <= 0.0
            or max(abs(first_left_mm), abs(first_right_mm)) / segment_dt_s
            > self.config.max_encoder_speed_mm_s
            or max(abs(second_left_mm), abs(second_right_mm)) / segment_dt_s
            > self.config.max_encoder_speed_mm_s
        ):
            self._lose_continuity()
            return self._estimate(current.received_timestamp_ns)

        previous_gyro = self._usable_gyro(previous)
        current_gyro = self._usable_gyro(current)
        qualities: set[FusionQuality] = {FusionQuality.INTERPOLATED_IMU}
        if previous_gyro is None or current_gyro is None:
            if not self.config.allow_wheel_only:
                self._lose_continuity()
                return self._estimate(current.received_timestamp_ns)
            qualities.add(FusionQuality.WHEEL_ONLY)
            first_gyro_z = None
            second_gyro_z = None
        else:
            first_gyro = previous_gyro + (current_gyro - previous_gyro) * 0.25
            second_gyro = previous_gyro + (current_gyro - previous_gyro) * 0.75
            first_gyro_z = float(first_gyro[2])
            second_gyro_z = float(second_gyro[2])

        current_acceleration_quality: set[FusionQuality] = set()
        if current.sensor_flags & SensorFlags.IMU_VALID:
            _, current_accel = self._calibrated_imu_vectors(current)
            current_acceleration_quality = self._acceleration_quality(
                current_accel,
                saturated=bool(current.sensor_flags & SensorFlags.ACCEL_SATURATED),
            )
        qualities.update(current_acceleration_quality)
        covariance_scale = self.config.dropped_sample_covariance_scale
        if FusionQuality.WHEEL_ONLY in qualities or {
            FusionQuality.TILT_DETECTED,
            FusionQuality.IMPACT_DETECTED,
        } & qualities:
            covariance_scale *= self.config.wheel_only_covariance_scale
        first_qualities = frozenset(qualities - current_acceleration_quality)
        second_qualities = frozenset(qualities)
        first_prediction = _Prediction(
            distance_mm=(first_left_mm + first_right_mm) / 2.0,
            encoder_heading_rad=(first_right_mm - first_left_mm) / self._track_mm,
            gyro_z_rad_s=first_gyro_z,
            dt_s=segment_dt_s,
            covariance_scale=covariance_scale,
            qualities=first_qualities,
            stationary=(
                abs(overrun.left_encoder_count - previous.left_encoder_count)
                <= self.config.stationary_encoder_delta_count
                and abs(overrun.right_encoder_count - previous.right_encoder_count)
                <= self.config.stationary_encoder_delta_count
            ),
        )
        second_prediction = _Prediction(
            distance_mm=(second_left_mm + second_right_mm) / 2.0,
            encoder_heading_rad=(second_right_mm - second_left_mm) / self._track_mm,
            gyro_z_rad_s=second_gyro_z,
            dt_s=segment_dt_s,
            covariance_scale=covariance_scale,
            qualities=second_qualities,
            stationary=(
                abs(current.left_encoder_count - overrun.left_encoder_count)
                <= self.config.stationary_encoder_delta_count
                and abs(current.right_encoder_count - overrun.right_encoder_count)
                <= self.config.stationary_encoder_delta_count
            ),
        )
        midpoint_host_timestamp_ns = previous_host_timestamp_ns + round(
            (host_timestamp_ns - previous_host_timestamp_ns) / 2.0
        )
        self._append_prediction(midpoint_host_timestamp_ns, first_prediction)
        self._append_prediction(host_timestamp_ns, second_prediction)
        if self._history:
            self._status = self._history[-1].qualities
        self._pending_overrun = None
        self._last_sample = current
        self._last_host_timestamp_ns = host_timestamp_ns
        return self._estimate(current.received_timestamp_ns)

    def pose_at(self, timestamp_ns: int) -> FusedPoseEstimate:
        self._validate_timestamp(timestamp_ns)
        with self._lock:
            entry, _ = self._nearest_entry(timestamp_ns)
            return self._entry_estimate(
                entry,
                timestamp_ns if entry is None else max(timestamp_ns, entry.timestamp_ns),
            )

    def submit_visual(
        self,
        observation: CenterCrossPoseObservation,
    ) -> VisualFusionResult:
        if not isinstance(observation, CenterCrossPoseObservation):
            raise TypeError("observation must be a CenterCrossPoseObservation.")
        if observation.selected_pose is None:
            return VisualFusionResult(False, None, None)
        with self._lock:
            entry, index = self._nearest_entry(observation.capture_timestamp_ns)
            if entry is None or index is None:
                if self._is_absolute_visual(observation):
                    self._initialize_from_visual(observation)
                    return VisualFusionResult(True, None, 0.0)
                return VisualFusionResult(False, None, None)
            alignment_error_ns = abs(
                entry.timestamp_ns - observation.capture_timestamp_ns
            )
            if alignment_error_ns > round(
                self.config.max_visual_alignment_error_ms * 1_000_000
            ):
                if (
                    self._is_absolute_visual(observation)
                    and observation.capture_timestamp_ns - entry.timestamp_ns
                    > round(self.config.max_telemetry_age_ms * 1_000_000)
                ):
                    self._lose_continuity()
                    self._initialize_from_visual(observation)
                    return VisualFusionResult(True, alignment_error_ns, 0.0)
                self._status = frozenset({FusionQuality.VISUAL_REJECTED})
                return VisualFusionResult(False, alignment_error_ns, None)
            candidate = min(
                observation.candidates,
                key=lambda item: (
                    (item.pose.position.x - observation.selected_pose.position.x) ** 2
                    + (item.pose.position.y - observation.selected_pose.position.y) ** 2
                    + normalize_angle(
                        item.pose.heading_rad - observation.selected_pose.heading_rad
                    ) ** 2
                ),
            )
            measurement = np.array(
                [
                    observation.selected_pose.position.x,
                    observation.selected_pose.position.y,
                    observation.selected_pose.heading_rad,
                ],
                dtype=np.float64,
            )
            confidence_scale = 1.0 / max(observation.confidence, 0.05)
            measurement_covariance = np.diag(
                [
                    candidate.position_uncertainty_mm**2 * confidence_scale,
                    candidate.position_uncertainty_mm**2 * confidence_scale,
                    candidate.heading_uncertainty_rad**2 * confidence_scale,
                ]
            )
            innovation = measurement - entry.state[:3]
            innovation[2] = normalize_angle(float(innovation[2]))
            h = np.zeros((3, 4), dtype=np.float64)
            h[:, :3] = np.eye(3)
            innovation_covariance = h @ entry.covariance @ h.T + measurement_covariance
            mahalanobis = float(
                innovation.T @ np.linalg.solve(innovation_covariance, innovation)
            )
            if mahalanobis > self.config.visual_innovation_gate:
                self._status = frozenset({FusionQuality.VISUAL_REJECTED})
                return VisualFusionResult(False, alignment_error_ns, mahalanobis)
            gain = entry.covariance @ h.T @ np.linalg.inv(innovation_covariance)
            corrected_state = entry.state + gain @ innovation
            corrected_state[2] = normalize_angle(float(corrected_state[2]))
            identity = np.eye(4)
            residual = identity - gain @ h
            corrected_covariance = (
                residual @ entry.covariance @ residual.T
                + gain @ measurement_covariance @ gain.T
            )
            entries = list(self._history)
            entries[index].state = corrected_state
            entries[index].covariance = corrected_covariance
            entries[index].qualities = frozenset({FusionQuality.FUSED})
            for replay_index in range(index + 1, len(entries)):
                prediction = entries[replay_index].prediction
                assert prediction is not None
                state, covariance = self._predict(
                    entries[replay_index - 1].state,
                    entries[replay_index - 1].covariance,
                    prediction,
                )
                entries[replay_index].state = state
                entries[replay_index].covariance = covariance
            self._history = deque(entries)
            assert observation.selection_source is not None
            self._anchor_source = observation.selection_source.value
            self._anchor_confidence = observation.confidence
            self._status = frozenset({FusionQuality.FUSED})
            return VisualFusionResult(True, alignment_error_ns, mahalanobis)

    def latest_estimate(self, current_timestamp_ns: int) -> FusedPoseEstimate:
        self._validate_timestamp(current_timestamp_ns)
        with self._lock:
            return self._estimate(current_timestamp_ns)

    def _validate_continuity(
        self, previous: OdometryImu, current: OdometryImu
    ) -> str | None:
        sequence_delta = (current.telemetry_sequence - previous.telemetry_sequence) & 0xFFFF
        if sequence_delta == 0 or sequence_delta > 0x7FFF:
            return "telemetry sequence did not advance"
        dt_us = current.sample_timestamp_us - previous.sample_timestamp_us
        if dt_us <= 0:
            return "controller sample time did not advance"
        if dt_us > round(self.config.max_sample_interval_ms * 1000):
            return "controller sample interval exceeded limit"
        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID
        if current.sensor_flags & required != required:
            return "both encoders are not valid"
        if current.sensor_flags & SensorFlags.SAMPLE_OVERRUN:
            return "controller reported sample overrun"
        return None

    def _map_controller_time(self, message: OdometryImu) -> int:
        controller_ns = message.sample_timestamp_us * 1000
        observed_offset = message.received_timestamp_ns - controller_ns
        self._clock_offsets_ns.append(observed_offset)
        mapped = controller_ns + min(self._clock_offsets_ns)
        mapped = min(mapped, message.received_timestamp_ns)
        if self._last_host_timestamp_ns is not None:
            mapped = max(mapped, self._last_host_timestamp_ns + 1)
        return mapped

    def _count_distance(self, count: int, *, left: bool) -> float:
        radius = (
            self.calibration.left_wheel_radius_mm
            if left
            else self.calibration.right_wheel_radius_mm
        )
        return (
            count
            * 2.0
            * math.pi
            * radius
            / self.calibration.encoder_counts_per_revolution
        )

    def _predict(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        prediction: _Prediction,
    ) -> tuple[np.ndarray, np.ndarray]:
        encoder_variance = self.config.encoder_heading_noise_std_rad**2
        if prediction.gyro_z_rad_s is None:
            gyro_weight = 0.0
            heading_delta = prediction.encoder_heading_rad
            heading_variance = encoder_variance
        else:
            gyro_variance = (self.config.gyro_noise_std_rad_s * prediction.dt_s) ** 2
            gyro_weight = encoder_variance / (encoder_variance + gyro_variance)
            gyro_delta = (
                prediction.gyro_z_rad_s - float(state[3])
            ) * prediction.dt_s
            heading_delta = (
                gyro_weight * gyro_delta
                + (1.0 - gyro_weight) * prediction.encoder_heading_rad
            )
            heading_variance = (
                encoder_variance * gyro_variance
                / (encoder_variance + gyro_variance)
            )
        midpoint = float(state[2]) + heading_delta / 2.0
        distance = prediction.distance_mm
        next_state = state.copy()
        next_state[0] += distance * math.cos(midpoint)
        next_state[1] += distance * math.sin(midpoint)
        next_state[2] = normalize_angle(float(state[2]) + heading_delta)

        bias_derivative = -gyro_weight * prediction.dt_s
        f = np.eye(4)
        f[0, 2] = -distance * math.sin(midpoint)
        f[1, 2] = distance * math.cos(midpoint)
        f[0, 3] = -distance * math.sin(midpoint) * bias_derivative / 2.0
        f[1, 3] = distance * math.cos(midpoint) * bias_derivative / 2.0
        f[2, 3] = bias_derivative
        distance_std = max(
            0.01,
            abs(distance) * self.config.encoder_distance_noise_fraction,
        )
        g = np.array(
            [
                [math.cos(midpoint), -distance * math.sin(midpoint) / 2.0, 0.0],
                [math.sin(midpoint), distance * math.cos(midpoint) / 2.0, 0.0],
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        q = np.diag(
            [
                distance_std**2,
                heading_variance,
                (
                    self.config.gyro_bias_random_walk_std_rad_s_per_sqrt_s
                    * math.sqrt(prediction.dt_s)
                )
                ** 2,
            ]
        ) * prediction.covariance_scale
        next_covariance = f @ covariance @ f.T + g @ q @ g.T
        if prediction.stationary and prediction.gyro_z_rad_s is not None:
            h = np.array([[0.0, 0.0, 0.0, 1.0]])
            residual = prediction.gyro_z_rad_s - float(next_state[3])
            r = self.config.stationary_gyro_noise_std_rad_s**2
            innovation_variance = float((h @ next_covariance @ h.T)[0, 0]) + r
            gain = next_covariance @ h.T / innovation_variance
            next_state = next_state + gain[:, 0] * residual
            next_covariance = (np.eye(4) - gain @ h) @ next_covariance
        return next_state, (next_covariance + next_covariance.T) / 2.0

    def _append_prediction(
        self,
        timestamp_ns: int,
        prediction: _Prediction,
    ) -> None:
        current = self._state_entry()
        if current is None:
            return
        state, covariance = self._predict(
            current.state,
            current.covariance,
            prediction,
        )
        self._history.append(
            _HistoryEntry(
                timestamp_ns,
                state,
                covariance,
                prediction,
                prediction.qualities or frozenset({FusionQuality.FUSED}),
            )
        )
        self._trim_history(timestamp_ns)

    def _usable_gyro(self, message: OdometryImu) -> np.ndarray | None:
        flags = message.sensor_flags
        if not flags & SensorFlags.IMU_VALID:
            return None
        if flags & SensorFlags.GYRO_SATURATED:
            return None
        gyro, _ = self._calibrated_imu_vectors(message)
        return gyro

    def _calibrated_imu_vectors(
        self, message: OdometryImu
    ) -> tuple[np.ndarray, np.ndarray]:
        gyro_sensor = np.array(
            [
                message.gyro_x_urad_s / 1_000_000.0,
                message.gyro_y_urad_s / 1_000_000.0,
                message.gyro_z_urad_s / 1_000_000.0,
            ],
            dtype=np.float64,
        )
        temperature_delta_c = (
            message.imu_temperature_cdeg / 100.0
            - self.config.imu_frame_calibration.reference_temperature_c
        )
        gyro_bias = (
            self._gyro_bias
            + self._gyro_bias_temperature_coefficient * temperature_delta_c
        )
        gyro_corrected = self._gyro_cross_axis_scale @ (gyro_sensor - gyro_bias)
        gyro_robot = self._imu_rotation @ gyro_corrected
        gyro_robot[2] *= self.calibration.gyro_z_sign

        accel_sensor = np.array(
            [
                float(message.accel_x_mm_s2),
                float(message.accel_y_mm_s2),
                float(message.accel_z_mm_s2),
            ],
            dtype=np.float64,
        )
        accel_bias = (
            self._accel_bias
            + self._accel_bias_temperature_coefficient * temperature_delta_c
        )
        accel_corrected = self._accel_cross_axis_scale @ (accel_sensor - accel_bias)
        accel_robot = self._imu_rotation @ accel_corrected
        return gyro_robot, accel_robot

    def _acceleration_quality(
        self,
        acceleration_robot_mm_s2: np.ndarray,
        *,
        saturated: bool,
    ) -> set[FusionQuality]:
        qualities: set[FusionQuality] = set()
        if saturated:
            qualities.add(FusionQuality.IMPACT_DETECTED)
            return qualities
        acceleration = math.sqrt(
            float(acceleration_robot_mm_s2[0]) ** 2
            + float(acceleration_robot_mm_s2[1]) ** 2
            + float(acceleration_robot_mm_s2[2]) ** 2
        )
        if abs(acceleration - 9807.0) > self.config.impact_accel_threshold_mm_s2:
            qualities.add(FusionQuality.IMPACT_DETECTED)
        if acceleration > 1e-9:
            cosine = max(
                -1.0,
                min(1.0, float(acceleration_robot_mm_s2[2]) / acceleration),
            )
            if math.degrees(math.acos(cosine)) > self.config.max_tilt_deg:
                qualities.add(FusionQuality.TILT_DETECTED)
        return qualities

    def _initialize(
        self,
        pose: FieldPose2D,
        timestamp_ns: int,
        position_uncertainty_mm: float,
        heading_uncertainty_rad: float,
        source: str,
        confidence: float,
    ) -> None:
        state = np.array(
            [
                pose.position.x,
                pose.position.y,
                pose.heading_rad,
                0.0,
            ],
            dtype=np.float64,
        )
        covariance = np.diag(
            [
                position_uncertainty_mm**2,
                position_uncertainty_mm**2,
                heading_uncertainty_rad**2,
                self.config.stationary_gyro_noise_std_rad_s**2,
            ]
        )
        self._history = deque(
            [_HistoryEntry(timestamp_ns, state, covariance, None, frozenset({FusionQuality.FUSED}))]
        )
        self._anchor_source = source
        self._anchor_confidence = confidence
        self._status = frozenset({FusionQuality.FUSED})

    def _initialize_from_visual(self, observation: CenterCrossPoseObservation) -> None:
        assert observation.selected_pose is not None
        assert observation.selection_source is not None
        candidate = min(
            observation.candidates,
            key=lambda item: abs(
                normalize_angle(
                    item.pose.heading_rad - observation.selected_pose.heading_rad
                )
            ),
        )
        self._initialize(
            observation.selected_pose,
            observation.capture_timestamp_ns,
            candidate.position_uncertainty_mm,
            candidate.heading_uncertainty_rad,
            observation.selection_source.value,
            observation.confidence,
        )
        # A visual re-anchor is not guaranteed to coincide with the previous
        # controller sample.  Establish a fresh encoder baseline on the next
        # telemetry frame instead of replaying a delta that began before it.
        self._pending_overrun = None
        self._last_sample = None
        self._last_host_timestamp_ns = None
        self._clock_offsets_ns.clear()
        self._initial_pose_available = False

    @staticmethod
    def _is_absolute_visual(observation: CenterCrossPoseObservation) -> bool:
        return observation.selection_source in {
            CenterCrossSelectionSource.RED_SAFE_ZONE,
            CenterCrossSelectionSource.BLUE_SAFE_ZONE,
            CenterCrossSelectionSource.RED_BLUE_SAFE_ZONES,
            CenterCrossSelectionSource.STATIC_MAP_TERMINAL,
        }

    def _append_baseline(self, timestamp_ns: int) -> None:
        current = self._state_entry()
        assert current is not None
        if timestamp_ns > current.timestamp_ns:
            self._history.append(
                _HistoryEntry(
                    timestamp_ns,
                    current.state.copy(),
                    current.covariance.copy(),
                    None,
                    current.qualities,
                )
            )

    def _lose_continuity(self) -> None:
        self._history.clear()
        self._pending_overrun = None
        self._last_sample = None
        self._last_host_timestamp_ns = None
        self._clock_offsets_ns.clear()
        self._initial_pose_available = False
        self._anchor_source = None
        self._anchor_confidence = 0.0
        self._status = frozenset({FusionQuality.CONTINUITY_LOST})

    def _state_entry(self) -> _HistoryEntry | None:
        return self._history[-1] if self._history else None

    def _nearest_entry(
        self, timestamp_ns: int
    ) -> tuple[_HistoryEntry | None, int | None]:
        if not self._history:
            return None, None
        entries = list(self._history)
        index = min(
            range(len(entries)),
            key=lambda item: abs(entries[item].timestamp_ns - timestamp_ns),
        )
        return entries[index], index

    def _trim_history(self, current_timestamp_ns: int) -> None:
        minimum = current_timestamp_ns - round(
            self.config.history_duration_ms * 1_000_000
        )
        while len(self._history) > 1 and self._history[1].timestamp_ns < minimum:
            self._history.popleft()
            self._history[0].prediction = None

    def _estimate(self, current_timestamp_ns: int) -> FusedPoseEstimate:
        return self._entry_estimate(self._state_entry(), current_timestamp_ns)

    def _entry_estimate(
        self, entry: _HistoryEntry | None, current_timestamp_ns: int
    ) -> FusedPoseEstimate:
        if entry is None:
            return FusedPoseEstimate(
                None, None, None, None, 0.0, None, self._status
            )
        qualities = set(entry.qualities)
        if (
            current_timestamp_ns < entry.timestamp_ns
            or current_timestamp_ns - entry.timestamp_ns
            > round(self.config.max_telemetry_age_ms * 1_000_000)
        ):
            qualities.add(FusionQuality.STALE)
            return FusedPoseEstimate(
                None,
                entry.timestamp_ns,
                None,
                None,
                0.0,
                self._anchor_source,
                frozenset(qualities),
            )
        xy_covariance = entry.covariance[:2, :2]
        position_uncertainty = math.sqrt(
            max(0.0, float(np.linalg.eigvalsh(xy_covariance)[-1]))
        )
        heading_uncertainty = math.sqrt(max(0.0, float(entry.covariance[2, 2])))
        return FusedPoseEstimate(
            FieldPose2D(
                FieldPoint(float(entry.state[0]), float(entry.state[1])),
                float(entry.state[2]),
            ),
            entry.timestamp_ns,
            max(position_uncertainty, 1e-6),
            max(heading_uncertainty, 1e-9),
            self._anchor_confidence,
            self._anchor_source,
            frozenset(qualities),
        )

    @staticmethod
    def _validate_timestamp(timestamp_ns: int) -> None:
        if isinstance(timestamp_ns, bool) or not isinstance(timestamp_ns, int) or timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer.")
