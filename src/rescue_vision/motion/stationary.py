"""Continuous encoder/IMU evidence, independent of application state changes."""
from __future__ import annotations

import math

from rescue_vision.motion.protocol import OdometryImu, SensorFlags


class StationaryMotionEvidence:
    """Host monotonic ns are compared to camera capture ns; device time checks continuity."""

    def __init__(self, *, max_gap_ns: int, max_gyro_rad_s: float):
        if isinstance(max_gap_ns, bool) or not isinstance(max_gap_ns, int) or max_gap_ns <= 0:
            raise ValueError(f"Invalid max_gap_ns={max_gap_ns!r}")
        if (isinstance(max_gyro_rad_s, bool) or not isinstance(max_gyro_rad_s, (int, float))
                or not math.isfinite(max_gyro_rad_s) or max_gyro_rad_s <= 0):
            raise ValueError(f"Invalid max_gyro_rad_s={max_gyro_rad_s!r}")
        self.max_gap_ns = max_gap_ns
        self.max_gyro_rad_s = max_gyro_rad_s
        self.latest: OdometryImu | None = None
        self.since_ns: int | None = None
        self.invalid_reason = "no_sample"

    def _valid(self, sample: OdometryImu) -> bool:
        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID
        return (sample.sensor_flags & required == required
                and not sample.sensor_flags & (SensorFlags.SAMPLE_OVERRUN | SensorFlags.GYRO_SATURATED)
                and abs(sample.gyro_z_rad_s) <= self.max_gyro_rad_s)

    def observe(self, sample: OdometryImu) -> None:
        if not isinstance(sample, OdometryImu):
            raise TypeError("sample must be OdometryImu")
        previous = self.latest
        self.latest = sample
        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID
        if sample.sensor_flags & required != required:
            reason = "sensor_invalid"
        elif sample.sensor_flags & (SensorFlags.SAMPLE_OVERRUN | SensorFlags.GYRO_SATURATED):
            reason = "sample_overrun_or_gyro_saturated"
        elif not math.isfinite(sample.gyro_z_rad_s) or abs(sample.gyro_z_rad_s) > self.max_gyro_rad_s:
            reason = "rotation"
        elif previous is None:
            reason = "first_sample"
        elif not self._valid(previous):
            reason = "previous_sample_invalid"
        elif not 0 <= sample.received_timestamp_ns - previous.received_timestamp_ns <= self.max_gap_ns:
            reason = "host_sample_gap"
        elif sample.sample_timestamp_us <= previous.sample_timestamp_us:
            reason = "duplicate_or_reversed_device_sample"
        elif (sample.sample_timestamp_us - previous.sample_timestamp_us) * 1000 > self.max_gap_ns:
            reason = "device_sample_gap"
        elif (sample.left_encoder_count != previous.left_encoder_count
              or sample.right_encoder_count != previous.right_encoder_count):
            reason = "encoder_motion"
        else:
            reason = "continuous_stationary"
        self.invalid_reason = reason
        if reason != "continuous_stationary":
            self.since_ns = None
        elif self.since_ns is None:
            self.since_ns = sample.received_timestamp_ns

    def stationary_since(self, now_ns: int) -> int | None:
        if self.latest is None or not 0 <= now_ns-self.latest.received_timestamp_ns <= self.max_gap_ns:
            return None
        return self.since_ns

    def capture_valid(self, capture_ns: int, now_ns: int, *, max_age_ns: int) -> bool:
        since = self.stationary_since(now_ns)
        return (since is not None and self.latest is not None
                and since <= capture_ns <= self.latest.received_timestamp_ns
                and 0 <= now_ns-capture_ns <= max_age_ns)

    def diagnostic(self, now_ns: int) -> str:
        since = "none" if self.since_ns is None else f"{self.since_ns/1e6:.1f}"
        age = "none" if self.latest is None else f"{(now_ns-self.latest.received_timestamp_ns)/1e6:.1f}"
        gyro = "none" if self.latest is None else f"{self.latest.gyro_z_rad_s:.4f}"
        reason = self.invalid_reason
        if self.latest is not None and not 0 <= now_ns-self.latest.received_timestamp_ns <= self.max_gap_ns:
            reason = "telemetry_age_invalid"
        return (f"stationary_since_ms={since} motion_age_ms={age} gyro_z_rad_s={gyro} "
                f"stationary_reason={reason}")
