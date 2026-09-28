"""Feedback-controlled relative distance and angle actions.

The controller in this module is deliberately independent of UART, cameras and
the match state machine.  It produces a body-twist target from a relative goal
and measured motion.  ``MotionController`` remains responsible for the final
wheel conversion and acceleration limiting.

The profile is intentionally conservative about the time between a measurement
and a new command.  A stale measurement therefore moves the brake point earlier
instead of making the action look more precise than it is.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from rescue_vision.localization.types import normalize_angle


class RelativeActionKind(str, Enum):
    """The two relative actions used by ``match_nb``."""

    TURN = "turn"
    STRAIGHT = "straight"


class RelativeActionPhase(str, Enum):
    """Progress phase reported to the caller for diagnostics and tests."""

    WAITING_FEEDBACK = "waiting_feedback"
    CRUISE = "cruise"
    BRAKE = "brake"
    FINE = "fine"
    SETTLE = "settle"
    COMPLETE = "complete"
    TIMEOUT = "timeout"


def _finite(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
    ):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    return float(value)


def _positive(value: object, name: str) -> float:
    converted = _finite(value, name)
    if converted <= 0.0:
        raise ValueError(f"{name} must be positive, got {value!r}.")
    return converted


def _nonnegative(value: object, name: str) -> float:
    converted = _finite(value, name)
    if converted < 0.0:
        raise ValueError(f"{name} must be non-negative, got {value!r}.")
    return converted


@dataclass(frozen=True, slots=True)
class RelativeActionProfile:
    """Limits and timing assumptions for a profiled relative action.

    ``command_wait_s`` is the lower-layer wheel-command refresh period.  The
    separate ``execution_response_s`` accounts for motor/firmware response and
    is expected to be calibrated on the vehicle.  Neither value claims to make
    the final field error zero; they only move the first braking decision to a
    safer side of the goal.
    """

    linear_deceleration_m_s2: float = 1.0
    angular_deceleration_rad_s2: float = 2.0
    command_wait_s: float = 0.04
    execution_response_s: float = 0.04
    max_telemetry_age_s: float = 0.16
    fine_linear_speed_m_s: float = 0.05
    fine_angular_velocity_rad_s: float = 0.12
    stop_wheel_speed_m_s: float = 0.015
    stop_angular_velocity_rad_s: float = 0.06
    heading_kp_rad_s: float = 2.0
    heading_max_angular_velocity_rad_s: float = 0.25
    correction_max_distance_m: float = 0.08
    correction_max_angle_rad: float = 0.12
    correction_timeout_s: float = 0.50
    action_timeout_s: float = 8.0
    stationary_confirm_time_s: float = 0.15

    def __post_init__(self) -> None:
        for name in (
            "linear_deceleration_m_s2",
            "angular_deceleration_rad_s2",
            "command_wait_s",
            "execution_response_s",
            "max_telemetry_age_s",
            "fine_linear_speed_m_s",
            "fine_angular_velocity_rad_s",
            "stop_wheel_speed_m_s",
            "stop_angular_velocity_rad_s",
            "heading_kp_rad_s",
            "heading_max_angular_velocity_rad_s",
            "correction_max_distance_m",
            "correction_max_angle_rad",
            "correction_timeout_s",
            "action_timeout_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        object.__setattr__(
            self,
            "stationary_confirm_time_s",
            _nonnegative(self.stationary_confirm_time_s, "stationary_confirm_time_s"),
        )


@dataclass(frozen=True, slots=True)
class RelativeActionFeedback:
    """One control-cycle observation for :class:`RelativeActionController`.

    ``progress`` is signed in the action's forward direction: positive means
    progress toward the configured target, regardless of whether the physical
    action is forward/behind or left/right.  The wheel speeds and gyro speed
    are measured values, not commands.
    """

    timestamp_ns: int
    progress: float
    heading_rad: float | None
    left_wheel_velocity_m_s: float | None
    right_wheel_velocity_m_s: float | None
    angular_velocity_rad_s: float | None
    telemetry_age_s: float | None
    stationary_since_ns: int | None
    stationary_latest_ns: int | None

    def __post_init__(self) -> None:
        if (
            isinstance(self.timestamp_ns, bool)
            or not isinstance(self.timestamp_ns, int)
            or self.timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        object.__setattr__(self, "progress", _finite(self.progress, "progress"))
        if self.heading_rad is not None:
            object.__setattr__(
                self, "heading_rad", _finite(self.heading_rad, "heading_rad")
            )
        for name in (
            "left_wheel_velocity_m_s",
            "right_wheel_velocity_m_s",
            "angular_velocity_rad_s",
            "telemetry_age_s",
        ):
            value = getattr(self, name)
            if value is not None:
                converted = _finite(value, name)
                if name == "telemetry_age_s" and converted < 0.0:
                    raise ValueError(f"{name} must be non-negative, got {value!r}.")
                object.__setattr__(self, name, converted)
        for name in ("stationary_since_ns", "stationary_latest_ns"):
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None.")


@dataclass(frozen=True, slots=True)
class RelativeActionCommand:
    """Controller output and diagnostics for one action step."""

    linear_velocity_m_s: float
    angular_velocity_rad_s: float
    phase: RelativeActionPhase
    complete: bool
    timed_out: bool
    reason: str
    position_error: float
    heading_error_rad: float | None
    braking_distance: float
    telemetry_delay_s: float
    use_zero_min_wheel_velocity: bool

    def __post_init__(self) -> None:
        for name in (
            "linear_velocity_m_s",
            "angular_velocity_rad_s",
            "position_error",
            "braking_distance",
            "telemetry_delay_s",
        ):
            object.__setattr__(self, name, _finite(getattr(self, name), name))
        if self.heading_error_rad is not None:
            object.__setattr__(
                self,
                "heading_error_rad",
                _finite(self.heading_error_rad, "heading_error_rad"),
            )
        if not isinstance(self.phase, RelativeActionPhase):
            raise ValueError("phase must be a RelativeActionPhase.")
        if not isinstance(self.complete, bool) or not isinstance(self.timed_out, bool):
            raise ValueError("complete and timed_out must be booleans.")
        if not isinstance(self.reason, str) or not self.reason:
            raise ValueError("reason must be a non-empty string.")


class RelativeActionController:
    """Generate a trapezoid-like relative action target with feedback closure.

    The profile is velocity-command based rather than a motor-position PID:
    the encoder/IMU measurements close the position loop, while
    ``MotionController`` supplies the final body-to-wheel acceleration ramp.
    """

    def __init__(self, profile: RelativeActionProfile | None = None) -> None:
        self.profile = profile or RelativeActionProfile()
        self._kind: RelativeActionKind | None = None
        self._target_signed = 0.0
        self._target_abs = 0.0
        self._cruise_speed = 0.0
        self._start_heading_rad: float | None = None
        self._target_heading_rad: float | None = None
        self._started_ns: int | None = None
        self._last_linear_command_m_s = 0.0
        self._last_angular_command_rad_s = 0.0
        self._pivot_track_m: float | None = None
        self._exit_speed_m_s = 0.0

    @property
    def active(self) -> bool:
        return self._kind is not None

    @property
    def kind(self) -> RelativeActionKind | None:
        return self._kind

    @property
    def target_heading_rad(self) -> float | None:
        return self._target_heading_rad

    def begin(
        self,
        kind: RelativeActionKind,
        target_signed: float,
        *,
        start_heading_rad: float,
        timestamp_ns: int,
        cruise_speed: float,
        target_heading_rad: float | None = None,
        pivot_track_m: float | None = None,
        exit_speed_m_s: float = 0.0,
    ) -> None:
        if not isinstance(kind, RelativeActionKind):
            raise TypeError("kind must be a RelativeActionKind.")
        target = _finite(target_signed, "target_signed")
        if abs(target) <= 0.0:
            raise ValueError("target_signed must be non-zero.")
        heading = _finite(start_heading_rad, "start_heading_rad")
        cruise = _positive(cruise_speed, "cruise_speed")
        exit_speed = _nonnegative(exit_speed_m_s, "exit_speed_m_s")
        if pivot_track_m is not None:
            pivot_track_m = _positive(pivot_track_m, "pivot_track_m")
            if kind is not RelativeActionKind.TURN or target <= 0.0:
                raise ValueError("Left pivot requires a positive TURN target.")
            if exit_speed > cruise * pivot_track_m:
                raise ValueError("exit_speed_m_s exceeds pivot wheel cruise speed.")
        elif exit_speed:
            raise ValueError("exit_speed_m_s requires pivot_track_m.")
        if (
            isinstance(timestamp_ns, bool)
            or not isinstance(timestamp_ns, int)
            or timestamp_ns < 0
        ):
            raise ValueError("timestamp_ns must be a non-negative integer.")
        if target_heading_rad is None:
            target_heading = normalize_angle(heading + target)
        else:
            target_heading = normalize_angle(
                _finite(target_heading_rad, "target_heading_rad")
            )
        self._kind = kind
        self._pivot_track_m = pivot_track_m
        self._exit_speed_m_s = exit_speed
        self._target_signed = target
        self._target_abs = abs(target)
        self._cruise_speed = cruise
        self._start_heading_rad = heading
        self._target_heading_rad = target_heading
        self._started_ns = timestamp_ns
        self._last_linear_command_m_s = 0.0
        self._last_angular_command_rad_s = 0.0

    def reset(self) -> None:
        self._kind = None
        self._pivot_track_m = None
        self._exit_speed_m_s = 0.0
        self._target_signed = 0.0
        self._target_abs = 0.0
        self._cruise_speed = 0.0
        self._start_heading_rad = None
        self._target_heading_rad = None
        self._started_ns = None
        self._last_linear_command_m_s = 0.0
        self._last_angular_command_rad_s = 0.0

    def update(self, feedback: RelativeActionFeedback) -> RelativeActionCommand:
        if not self.active:
            raise RuntimeError("begin() must be called before update().")
        assert self._kind is not None
        assert self._started_ns is not None
        elapsed_s = max(0.0, (feedback.timestamp_ns - self._started_ns) / 1e9)
        heading_error = self._heading_error(feedback.heading_rad)
        position_error = self._target_abs - feedback.progress
        measured_speed = self._measured_speed(feedback)
        telemetry_delay_s = self._telemetry_delay(feedback)
        braking_distance = self._braking_distance(measured_speed, telemetry_delay_s)
        position_tolerance = self._position_tolerance()
        heading_tolerance = self._heading_tolerance()
        stopped = self._stopped(feedback)
        stationary = self._stationary_confirmed(feedback)

        if elapsed_s >= self.profile.action_timeout_s:
            return self._command(
                0.0,
                0.0,
                RelativeActionPhase.TIMEOUT,
                reason="action_timeout",
                position_error=position_error,
                heading_error=heading_error,
                braking_distance=braking_distance,
                telemetry_delay_s=telemetry_delay_s,
                timed_out=True,
            )

        if (feedback.telemetry_age_s is None
                or feedback.telemetry_age_s > self.profile.max_telemetry_age_s
                or feedback.heading_rad is None
                or feedback.angular_velocity_rad_s is None
                or feedback.left_wheel_velocity_m_s is None
                or feedback.right_wheel_velocity_m_s is None):
            return self._command(
                0.0, 0.0, RelativeActionPhase.WAITING_FEEDBACK,
                reason="critical_motion_feedback_unavailable",
                position_error=position_error, heading_error=heading_error,
                braking_distance=braking_distance, telemetry_delay_s=telemetry_delay_s,
            )

        if self._pivot_track_m is not None and self._exit_speed_m_s > 0.0:
            return self._rolling_pivot_command(
                feedback, position_error, heading_error, measured_speed,
                braking_distance, telemetry_delay_s,
            )

        if position_error < -position_tolerance:
            # Correct through zero without an intermediate stop gate. The
            # lower motion controller still applies its deceleration and
            # direction-change ramp, so this requests progress immediately
            # without commanding an abrupt wheel reversal.
            return self._fine_command(
                feedback,
                position_error=position_error,
                heading_error=heading_error,
                braking_distance=braking_distance,
                telemetry_delay_s=telemetry_delay_s,
                correction=True,
            )

        position_ok = abs(position_error) <= position_tolerance
        heading_ok = (
            heading_error is not None and abs(heading_error) <= heading_tolerance
        )
        if position_ok and heading_ok:
            if stopped and stationary:
                return self._command(
                    0.0,
                    0.0,
                    RelativeActionPhase.COMPLETE,
                    reason="position_heading_and_stationary_ok",
                    position_error=position_error,
                    heading_error=heading_error,
                    braking_distance=braking_distance,
                    telemetry_delay_s=telemetry_delay_s,
                    complete=True,
                )
            return self._command(
                0.0,
                0.0,
                RelativeActionPhase.SETTLE,
                reason=self._settle_reason(feedback, stopped, stationary),
                position_error=position_error,
                heading_error=heading_error,
                braking_distance=braking_distance,
                telemetry_delay_s=telemetry_delay_s,
            )

        if position_ok and self._kind is RelativeActionKind.TURN:
            # A turn has two independent completion gates: accumulated angle
            # and final heading.  The angle can be inside its wider tolerance
            # while the heading is still outside the route tolerance (this is
            # exactly the small overshoot case).  Request the bounded trim
            # immediately; MotionController decelerates the existing motion
            # and ramps through the direction change without a zero-command
            # stop gate.
            assert heading_error is not None
            return self._command(
                0.0,
                self._heading_trim_command(heading_error),
                RelativeActionPhase.FINE,
                reason="angle_ok_heading_trim",
                position_error=position_error,
                heading_error=heading_error,
                braking_distance=braking_distance,
                telemetry_delay_s=telemetry_delay_s,
            )

        if position_ok and self._kind is RelativeActionKind.STRAIGHT:
            # Keep progressing toward the requested heading instead of
            # waiting at zero for a separate stopped state.  The lower layer
            # remains the sole authority for the physical acceleration ramp.
            assert heading_error is not None
            return self._command(
                0.0,
                self._heading_trim_command(heading_error),
                RelativeActionPhase.FINE,
                reason="distance_ok_heading_trim",
                position_error=position_error,
                heading_error=heading_error,
                braking_distance=braking_distance,
                telemetry_delay_s=telemetry_delay_s,
            )

        if self._kind is RelativeActionKind.STRAIGHT:
            linear = self._profiled_linear_command(
                position_error, measured_speed, telemetry_delay_s
            )
            angular = self._heading_command(heading_error)
            phase = (
                RelativeActionPhase.FINE
                if linear <= self.profile.fine_linear_speed_m_s + 1e-12
                else (
                    RelativeActionPhase.BRAKE
                    if position_error <= braking_distance
                    else RelativeActionPhase.CRUISE
                )
            )
            return self._command(
                math.copysign(linear, self._target_signed),
                angular,
                phase,
                reason=(
                    "fine_distance_feedback"
                    if phase is RelativeActionPhase.FINE
                    else (
                        "braking_distance_feedback"
                        if phase is RelativeActionPhase.BRAKE
                        else "cruise"
                    )
                ),
                position_error=position_error,
                heading_error=heading_error,
                braking_distance=braking_distance,
                telemetry_delay_s=telemetry_delay_s,
            )

        angular = self._profiled_angular_command(
            position_error, measured_speed, telemetry_delay_s
        )
        phase = (
            RelativeActionPhase.FINE
            if angular <= self.profile.fine_angular_velocity_rad_s + 1e-12
            else (
                RelativeActionPhase.BRAKE
                if position_error <= braking_distance
                else RelativeActionPhase.CRUISE
            )
        )
        return self._command(
            0.0,
            math.copysign(angular, self._target_signed),
            phase,
            reason=(
                "fine_angle_feedback"
                if phase is RelativeActionPhase.FINE
                else (
                    "braking_distance_feedback"
                    if phase is RelativeActionPhase.BRAKE
                    else "cruise"
                )
            ),
            position_error=position_error,
            heading_error=heading_error,
            braking_distance=braking_distance,
            telemetry_delay_s=telemetry_delay_s,
        )

    def _position_tolerance(self) -> float:
        # The caller encodes the action tolerance in the profile's target
        # domain by setting it immediately before begin().  The two defaults
        # below are deliberately small and are replaced by the NB adapter.
        return self._configured_position_tolerance

    def _heading_tolerance(self) -> float:
        return self._configured_heading_tolerance

    @property
    def _configured_position_tolerance(self) -> float:
        return getattr(self, "_position_tolerance_value", 0.01)

    @property
    def _configured_heading_tolerance(self) -> float:
        return getattr(self, "_heading_tolerance_value", 0.03)

    def set_tolerances(
        self,
        *,
        position_tolerance: float,
        heading_tolerance: float,
    ) -> None:
        """Set tolerances for the next/current action in its native units."""

        self._position_tolerance_value = _positive(
            position_tolerance, "position_tolerance"
        )
        self._heading_tolerance_value = _positive(
            heading_tolerance, "heading_tolerance"
        )

    def _heading_error(self, heading_rad: float | None) -> float | None:
        if heading_rad is None or self._target_heading_rad is None:
            return None
        return normalize_angle(self._target_heading_rad - heading_rad)

    def _measured_speed(self, feedback: RelativeActionFeedback) -> float:
        if self._kind is RelativeActionKind.TURN:
            measured = feedback.angular_velocity_rad_s
            fallback = self._last_angular_command_rad_s
        else:
            if (
                feedback.left_wheel_velocity_m_s is None
                or feedback.right_wheel_velocity_m_s is None
            ):
                measured = None
            else:
                measured = 0.5 * (
                    feedback.left_wheel_velocity_m_s
                    + feedback.right_wheel_velocity_m_s
                )
            fallback = self._last_linear_command_m_s
        if measured is None:
            return abs(fallback)
        # A fresh encoder/gyro value can still lag the wheel command by the
        # UART/firmware response time.  Use the larger of measured and the
        # last command for braking, so a low instantaneous sample cannot move
        # the brake point dangerously close to the goal.  The telemetry age is
        # still included separately in ``_telemetry_delay``.
        return max(abs(measured), abs(fallback))

    def _telemetry_delay(self, feedback: RelativeActionFeedback) -> float:
        age = feedback.telemetry_age_s
        measured_age = (
            self.profile.max_telemetry_age_s
            if age is None
            else max(age, 0.0)
        )
        return measured_age + self.profile.command_wait_s + self.profile.execution_response_s

    def _braking_distance(self, speed: float, telemetry_delay_s: float) -> float:
        deceleration = (
            self.profile.angular_deceleration_rad_s2
            if self._kind is RelativeActionKind.TURN
            else self.profile.linear_deceleration_m_s2
        )
        return speed * telemetry_delay_s + speed * speed / (2.0 * deceleration)

    def _profiled_linear_command(
        self, position_error: float, speed: float, telemetry_delay_s: float
    ) -> float:
        if position_error <= self._position_tolerance():
            return 0.0
        safe_distance = max(0.0, position_error - speed * telemetry_delay_s)
        safe_speed = math.sqrt(2.0 * self.profile.linear_deceleration_m_s2 * safe_distance)
        if safe_speed <= self.profile.fine_linear_speed_m_s:
            return self.profile.fine_linear_speed_m_s
        return min(self._cruise_speed, safe_speed)

    def _profiled_angular_command(
        self, position_error: float, speed: float, telemetry_delay_s: float
    ) -> float:
        if position_error <= self._position_tolerance():
            return 0.0
        safe_distance = max(0.0, position_error - speed * telemetry_delay_s)
        safe_speed = math.sqrt(
            2.0 * self.profile.angular_deceleration_rad_s2 * safe_distance
        )
        if safe_speed <= self.profile.fine_angular_velocity_rad_s:
            return self.profile.fine_angular_velocity_rad_s
        return min(self._cruise_speed, safe_speed)

    def _rolling_pivot_command(
        self,
        feedback: RelativeActionFeedback,
        position_error: float,
        heading_error: float | None,
        speed: float,
        braking_distance: float,
        delay_s: float,
    ) -> RelativeActionCommand:
        """Keep the left wheel zero until a bounded moving handoff is possible.

        The right wheel slows to exit speed. Completion predicts the residual
        yaw while the left catches up; the next straight closes heading error.
        This is a moving handoff, never a claim of stationary completion.
        """
        assert self._pivot_track_m is not None
        assert heading_error is not None
        assert feedback.angular_velocity_rad_s is not None
        assert feedback.left_wheel_velocity_m_s is not None
        assert feedback.right_wheel_velocity_m_s is not None
        tolerance = min(self._position_tolerance(), self._heading_tolerance())
        angular = feedback.angular_velocity_rad_s
        residual = (max(0.0, angular) * delay_s
                    + max(0.0, angular) ** 2 / (2.0 * self.profile.angular_deceleration_rad_s2))
        exit_angular = self._exit_speed_m_s / self._pivot_track_m
        wheel_tolerance = self.profile.stop_wheel_speed_m_s
        complete = (
            abs(position_error) <= self._position_tolerance()
            and abs(heading_error) <= self._heading_tolerance()
            and abs(position_error - residual) <= self._position_tolerance()
            and abs(heading_error - residual) <= self._heading_tolerance()
            and 0.0 <= angular <= exit_angular * 1.2
            and abs(feedback.left_wheel_velocity_m_s) <= wheel_tolerance
            and abs(feedback.right_wheel_velocity_m_s - self._exit_speed_m_s) <= wheel_tolerance
        )
        if complete:
            return self._command(
                self._exit_speed_m_s, 0.0, RelativeActionPhase.COMPLETE,
                reason="pivot_moving_handoff", complete=True,
                position_error=position_error, heading_error=heading_error,
                braking_distance=braking_distance, telemetry_delay_s=delay_s,
            )
        if min(position_error, heading_error) < -tolerance:
            return self._command(
                0.0, 0.0, RelativeActionPhase.TIMEOUT,
                reason="pivot_handoff_missed", timed_out=True,
                position_error=position_error, heading_error=heading_error,
                braking_distance=braking_distance, telemetry_delay_s=delay_s,
            )
        remaining = max(0.0, min(position_error, heading_error))
        # Use the terminal wheel speed as a floor; do not stop and restart
        # the right wheel between the pivot and the following straight.
        safe = math.sqrt(2.0 * self.profile.angular_deceleration_rad_s2
                         * max(0.0, remaining - speed * delay_s))
        target_angular = min(self._cruise_speed, max(exit_angular, safe))
        phase = (RelativeActionPhase.FINE if target_angular <= exit_angular
                 else RelativeActionPhase.BRAKE if target_angular < self._cruise_speed
                 else RelativeActionPhase.CRUISE)
        return self._command(
            0.0, target_angular, phase, reason="left_wheel_pivot",
            position_error=position_error, heading_error=heading_error,
            braking_distance=braking_distance, telemetry_delay_s=delay_s,
        )

    def _heading_command(self, heading_error: float | None) -> float:
        if heading_error is None:
            return 0.0
        if abs(heading_error) < 1e-12:
            return 0.0
        return max(
            -self.profile.heading_max_angular_velocity_rad_s,
            min(
                self.profile.heading_max_angular_velocity_rad_s,
                self.profile.heading_kp_rad_s * heading_error,
            ),
        )

    def _heading_trim_command(self, heading_error: float) -> float:
        """Return a bounded low-speed heading correction."""

        magnitude = min(
            self.profile.fine_angular_velocity_rad_s,
            abs(self.profile.heading_kp_rad_s * heading_error),
        )
        if magnitude <= 0.0:
            return 0.0
        return math.copysign(magnitude, heading_error)

    def _stopped(self, feedback: RelativeActionFeedback) -> bool:
        left = feedback.left_wheel_velocity_m_s
        right = feedback.right_wheel_velocity_m_s
        if left is None or right is None:
            return False
        if max(abs(left), abs(right)) > self.profile.stop_wheel_speed_m_s:
            return False
        angular = feedback.angular_velocity_rad_s
        return angular is not None and abs(angular) <= self.profile.stop_angular_velocity_rad_s

    def _stationary_confirmed(self, feedback: RelativeActionFeedback) -> bool:
        since = feedback.stationary_since_ns
        latest = feedback.stationary_latest_ns
        if since is None or latest is None or self._started_ns is None:
            return False
        # A continuously stationary vehicle remains stationary across actions.
        # Require a fresh sample, but never move the physical stop origin.
        if latest < self._started_ns or latest > feedback.timestamp_ns or since > latest:
            return False
        if feedback.telemetry_age_s is None or feedback.telemetry_age_s > self.profile.max_telemetry_age_s:
            return False
        observed_duration_s = max(0.0, (latest - since) / 1e9)
        return observed_duration_s >= self.profile.stationary_confirm_time_s

    def _settle_reason(
        self,
        feedback: RelativeActionFeedback,
        stopped: bool,
        stationary: bool,
    ) -> str:
        return (
            "settle_waiting_"
            f"{'speed_ok' if stopped else 'speed'}_"
            f"{'stationary_ok' if stationary else 'new_telemetry'}"
        )

    def _fine_command(
        self,
        feedback: RelativeActionFeedback,
        *,
        position_error: float,
        heading_error: float | None,
        braking_distance: float,
        telemetry_delay_s: float,
        correction: bool,
    ) -> RelativeActionCommand:
        if self._kind is RelativeActionKind.TURN:
            correction_direction = 1.0 if position_error >= 0.0 else -1.0
            angular = math.copysign(
                self.profile.fine_angular_velocity_rad_s,
                self._target_signed * correction_direction,
            )
            return self._command(
                0.0,
                angular,
                RelativeActionPhase.FINE,
                reason="overshoot_correction" if correction else "fine_turn",
                position_error=position_error,
                heading_error=heading_error,
                braking_distance=braking_distance,
                telemetry_delay_s=telemetry_delay_s,
            )
        linear = self.profile.fine_linear_speed_m_s
        correction_direction = 1.0 if position_error >= 0.0 else -1.0
        linear = math.copysign(
            linear,
            self._target_signed * correction_direction,
        )
        return self._command(
            linear,
            self._heading_command(heading_error),
            RelativeActionPhase.FINE,
            reason="overshoot_correction" if correction else "fine_straight",
            position_error=position_error,
            heading_error=heading_error,
            braking_distance=braking_distance,
            telemetry_delay_s=telemetry_delay_s,
        )

    def _command(
        self,
        linear: float,
        angular: float,
        phase: RelativeActionPhase,
        *,
        reason: str,
        position_error: float,
        heading_error: float | None,
        braking_distance: float,
        telemetry_delay_s: float,
        complete: bool = False,
        timed_out: bool = False,
    ) -> RelativeActionCommand:
        if self._pivot_track_m is not None and angular != 0.0:
            linear = angular * self._pivot_track_m / 2.0
        self._last_linear_command_m_s = linear
        self._last_angular_command_rad_s = angular
        heading_text = (
            "none" if heading_error is None else f"{heading_error:.4f}"
        )
        return RelativeActionCommand(
            linear_velocity_m_s=linear,
            angular_velocity_rad_s=angular,
            phase=phase,
            complete=complete,
            timed_out=timed_out,
            reason=(
                f"{reason}:error={position_error:.4f},"
                f"heading_error={heading_text},"
                f"brake={braking_distance:.4f},delay={telemetry_delay_s:.4f}"
            ),
            position_error=position_error,
            heading_error_rad=heading_error,
            braking_distance=braking_distance,
            telemetry_delay_s=telemetry_delay_s,
            use_zero_min_wheel_velocity=phase
            in {
                RelativeActionPhase.FINE,
                RelativeActionPhase.SETTLE,
                RelativeActionPhase.COMPLETE,
                RelativeActionPhase.TIMEOUT,
            },
        )


__all__ = [
    "RelativeActionCommand",
    "RelativeActionController",
    "RelativeActionFeedback",
    "RelativeActionKind",
    "RelativeActionPhase",
    "RelativeActionProfile",
]
