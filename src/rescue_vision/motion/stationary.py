"""Continuous encoder/IMU evidence, independent of application state changes."""
from __future__ import annotations

import math

from rescue_vision.motion.protocol import OdometryImu, SensorFlags


class StationaryMotionEvidence:
    """Host monotonic ns are compared to camera capture ns; device time checks continuity."""

    def __init__(
        self,
        *,
        max_gap_ns: int,
        max_gyro_rad_s: float,
        exit_gyro_rad_s: float | None = None,
        motion_confirm_ns: int = 80_000_000,
        encoder_tolerance_counts: int = 0,
    ):
        if isinstance(max_gap_ns, bool) or not isinstance(max_gap_ns, int) or max_gap_ns <= 0:
            raise ValueError(f"Invalid max_gap_ns={max_gap_ns!r}")
        if (isinstance(max_gyro_rad_s, bool) or not isinstance(max_gyro_rad_s, (int, float))
                or not math.isfinite(max_gyro_rad_s) or max_gyro_rad_s <= 0):
            raise ValueError(f"Invalid max_gyro_rad_s={max_gyro_rad_s!r}")
        if exit_gyro_rad_s is None:
            exit_gyro_rad_s = max(max_gyro_rad_s * 2.0, max_gyro_rad_s + 0.01)
        if (isinstance(exit_gyro_rad_s, bool) or not isinstance(exit_gyro_rad_s, (int, float))
                or not math.isfinite(exit_gyro_rad_s) or exit_gyro_rad_s < max_gyro_rad_s):
            raise ValueError(
                "Invalid exit_gyro_rad_s="
                f"{exit_gyro_rad_s!r}; it must be finite and >= max_gyro_rad_s."
            )
        if (isinstance(motion_confirm_ns, bool) or not isinstance(motion_confirm_ns, int)
                or motion_confirm_ns <= 0):
            raise ValueError(f"Invalid motion_confirm_ns={motion_confirm_ns!r}")
        if (isinstance(encoder_tolerance_counts, bool)
                or not isinstance(encoder_tolerance_counts, int)
                or encoder_tolerance_counts < 0):
            raise ValueError(
                f"Invalid encoder_tolerance_counts={encoder_tolerance_counts!r}"
            )
        self.max_gap_ns = max_gap_ns
        self.max_gyro_rad_s = max_gyro_rad_s
        self.exit_gyro_rad_s = exit_gyro_rad_s
        self.motion_confirm_ns = motion_confirm_ns
        self.encoder_tolerance_counts = encoder_tolerance_counts
        self.latest: OdometryImu | None = None
        self.since_ns: int | None = None
        self.invalid_reason = "no_sample"
        self._stationary_reference_counts: tuple[int, int] | None = None
        self._motion_pending_since_ns: int | None = None
        self._motion_pending_reason: str | None = None
        self._motion_pending_samples = 0
        self._needs_continuous_sample = False

    def _data_valid(self, sample: OdometryImu) -> bool:
        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID
        return (sample.sensor_flags & required == required
                and not sample.sensor_flags & (SensorFlags.SAMPLE_OVERRUN | SensorFlags.GYRO_SATURATED)
                and math.isfinite(sample.gyro_z_rad_s))

    @staticmethod
    def _encoder_counts(sample: OdometryImu) -> tuple[int, int]:
        return sample.left_encoder_count, sample.right_encoder_count

    def _encoder_delta(self, sample: OdometryImu) -> tuple[int, int]:
        if self._stationary_reference_counts is None:
            return (0, 0)
        return tuple(
            current - reference
            for current, reference in zip(
                self._encoder_counts(sample), self._stationary_reference_counts
            )
        )

    def _clear_motion_pending(self) -> None:
        self._motion_pending_since_ns = None
        self._motion_pending_reason = None
        self._motion_pending_samples = 0

    def _invalidate(self, reason: str) -> None:
        self.invalid_reason = reason
        self.since_ns = None
        self._stationary_reference_counts = None
        self._clear_motion_pending()
        self._needs_continuous_sample = True

    def observe(self, sample: OdometryImu) -> None:
        if not isinstance(sample, OdometryImu):
            raise TypeError("sample must be OdometryImu")
        previous = self.latest
        self.latest = sample
        required = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID
        if sample.sensor_flags & required != required:
            self._invalidate("sensor_invalid")
            return
        elif sample.sensor_flags & (SensorFlags.SAMPLE_OVERRUN | SensorFlags.GYRO_SATURATED):
            self._invalidate("sample_overrun_or_gyro_saturated")
            return
        elif not math.isfinite(sample.gyro_z_rad_s):
            self._invalidate("rotation")
            return
        elif previous is None:
            self.invalid_reason = "first_sample"
            self._clear_motion_pending()
            self._stationary_reference_counts = None
            self.since_ns = None
            self._needs_continuous_sample = False
            return
        elif not self._data_valid(previous):
            self._invalidate("previous_sample_invalid")
            return
        elif not 0 <= sample.received_timestamp_ns - previous.received_timestamp_ns <= self.max_gap_ns:
            self._invalidate("host_sample_gap")
            return
        elif sample.sample_timestamp_us <= previous.sample_timestamp_us:
            self._invalidate("duplicate_or_reversed_device_sample")
            return
        elif (sample.sample_timestamp_us - previous.sample_timestamp_us) * 1000 > self.max_gap_ns:
            self._invalidate("device_sample_gap")
            return

        if self._needs_continuous_sample:
            # The first valid packet after a gap/duplicate/invalid packet is
            # only a continuity recovery point.  Do not use it to manufacture
            # a new stationary interval before another consecutive sample.
            self._needs_continuous_sample = False
            self._clear_motion_pending()
            self.since_ns = None
            self._stationary_reference_counts = self._encoder_counts(sample)
            self.invalid_reason = "continuity_recovered"
            return

        encoder_delta = self._encoder_delta(sample)
        gyro_abs = abs(sample.gyro_z_rad_s)
        motion_reason: str | None = None
        if gyro_abs >= self.exit_gyro_rad_s:
            motion_reason = "rotation"
        elif max(abs(value) for value in encoder_delta) > self.encoder_tolerance_counts:
            motion_reason = "encoder_motion"

        if motion_reason is not None:
            if self._motion_pending_since_ns is None:
                self._motion_pending_since_ns = sample.received_timestamp_ns
                self._motion_pending_reason = motion_reason
                self._motion_pending_samples = 0
            self._motion_pending_samples += 1
            pending_elapsed_ns = (
                sample.received_timestamp_ns - self._motion_pending_since_ns
            )
            # A clearly impossible-to-ignore angular rate remains an immediate
            # invalidation.  Smaller excursions must persist before discarding
            # the complete stationary interval.
            hard_rotation = gyro_abs >= max(
                self.exit_gyro_rad_s * 1.5,
                self.max_gyro_rad_s + 0.05,
            )
            if hard_rotation or pending_elapsed_ns >= self.motion_confirm_ns:
                self._invalidate(motion_reason)
            else:
                self.invalid_reason = motion_reason
            return

        # The excursion ended before the debounce interval.  Keep the original
        # stationary start; capture_valid() remains closed while pending via
        # stationary_since(), but a one-sample jitter does not erase evidence.
        self._clear_motion_pending()
        if self.since_ns is None:
            if gyro_abs <= self.max_gyro_rad_s:
                self.since_ns = sample.received_timestamp_ns
                self._stationary_reference_counts = self._encoder_counts(sample)
                self.invalid_reason = "continuous_stationary"
            else:
                self.invalid_reason = "waiting_stationary_threshold"
        else:
            self.invalid_reason = "continuous_stationary"

    def stationary_since(self, now_ns: int) -> int | None:
        if self.latest is None or not 0 <= now_ns-self.latest.received_timestamp_ns <= self.max_gap_ns:
            return None
        if self._motion_pending_since_ns is not None:
            return None
        return self.since_ns

    def capture_valid(self, capture_ns: int, now_ns: int, *, max_age_ns: int) -> bool:
        since = self.stationary_since(now_ns)
        return (since is not None and self.latest is not None
                and since <= capture_ns <= self.latest.received_timestamp_ns
                and 0 <= now_ns-capture_ns <= max_age_ns)

    def diagnostic(self, now_ns: int) -> str:
        effective_since = self.stationary_since(now_ns)
        since = "none" if effective_since is None else f"{effective_since/1e6:.1f}"
        age = "none" if self.latest is None else f"{(now_ns-self.latest.received_timestamp_ns)/1e6:.1f}"
        gyro = "none" if self.latest is None else f"{self.latest.gyro_z_rad_s:.4f}"
        reason = self.invalid_reason
        if self.latest is not None and not 0 <= now_ns-self.latest.received_timestamp_ns <= self.max_gap_ns:
            reason = "telemetry_age_invalid"
        pending = self._motion_pending_since_ns is not None
        pending_age = (
            "none"
            if self._motion_pending_since_ns is None
            else f"{(now_ns-self._motion_pending_since_ns)/1e6:.1f}"
        )
        delta = self._encoder_delta(self.latest) if self.latest is not None else (0, 0)
        return (
            f"stationary_since_ms={since} motion_age_ms={age} gyro_z_rad_s={gyro} "
            f"stationary_reason={reason} motion_pending={int(pending)} "
            f"motion_pending_ms={pending_age} motion_pending_samples={self._motion_pending_samples} "
            f"encoder_delta_counts={delta[0]},{delta[1]}"
        )
