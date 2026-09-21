"""Closed-loop left/right wheel actions used by the NB opening route."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from rescue_vision.localization.types import normalize_angle
from rescue_vision.motion.relative_action import _finite, _positive


class WheelActionPhase(str, Enum):
    LOWER_LEFT = "lower_left"
    HOLD_ANGLE = "hold_angle"
    RAISE_LEFT = "raise_left"
    DRIVE_DISTANCE = "drive_distance"
    STOPPING = "stopping"
    COMPLETE = "complete"
    WAITING_FEEDBACK = "waiting_feedback"
    TIMEOUT = "timeout"


@dataclass(frozen=True, slots=True)
class WheelActionProfile:
    wheel_track_m: float
    right_wheel_speed_m_s: float = 1.5
    left_wheel_hold_speed_m_s: float = 1.0
    left_wheel_final_speed_m_s: float = 1.5
    left_transition_acceleration_m_s2: float = 0.5
    target_angle_rad: float = math.pi / 3.0
    post_turn_distance_m: float = 1.0
    angle_tolerance_rad: float = 0.03
    distance_tolerance_m: float = 0.02
    speed_tolerance_m_s: float = 0.03
    linear_deceleration_m_s2: float = 2.0
    telemetry_delay_s: float = 0.10
    stop_wheel_speed_m_s: float = 0.015
    stop_angular_velocity_rad_s: float = 0.06
    stationary_confirm_time_s: float = 0.10
    max_telemetry_age_s: float = 0.16
    action_timeout_s: float = 8.0

    def __post_init__(self) -> None:
        for name in (
            "wheel_track_m",
            "right_wheel_speed_m_s",
            "left_wheel_hold_speed_m_s",
            "left_wheel_final_speed_m_s",
            "left_transition_acceleration_m_s2",
            "target_angle_rad",
            "post_turn_distance_m",
            "angle_tolerance_rad",
            "distance_tolerance_m",
            "speed_tolerance_m_s",
            "linear_deceleration_m_s2",
            "telemetry_delay_s",
            "stop_wheel_speed_m_s",
            "stop_angular_velocity_rad_s",
            "max_telemetry_age_s",
            "action_timeout_s",
        ):
            object.__setattr__(self, name, _positive(getattr(self, name), name))
        if self.left_wheel_hold_speed_m_s >= self.right_wheel_speed_m_s:
            raise ValueError("left_wheel_hold_speed_m_s must be below right_wheel_speed_m_s.")
        if self.left_wheel_final_speed_m_s < self.right_wheel_speed_m_s:
            raise ValueError("left_wheel_final_speed_m_s must reach right_wheel_speed_m_s.")
        if self.stationary_confirm_time_s < 0.0 or not math.isfinite(self.stationary_confirm_time_s):
            raise ValueError("stationary_confirm_time_s must be finite and non-negative.")


@dataclass(frozen=True, slots=True)
class WheelActionFeedback:
    timestamp_ns: int
    heading_rad: float
    distance_m: float
    left_wheel_velocity_m_s: float
    right_wheel_velocity_m_s: float
    angular_velocity_rad_s: float
    telemetry_age_s: float
    stationary_since_ns: int | None
    stationary_latest_ns: int | None

    def __post_init__(self) -> None:
        if isinstance(self.timestamp_ns, bool) or not isinstance(self.timestamp_ns, int) or self.timestamp_ns < 0:
            raise ValueError("timestamp_ns must be a non-negative integer.")
        for name in (
            "heading_rad", "distance_m", "left_wheel_velocity_m_s",
            "right_wheel_velocity_m_s", "angular_velocity_rad_s", "telemetry_age_s",
        ):
            value = _finite(getattr(self, name), name)
            if name == "telemetry_age_s" and value < 0.0:
                raise ValueError("telemetry_age_s must be non-negative.")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class WheelActionCommand:
    left_wheel_velocity_m_s: float
    right_wheel_velocity_m_s: float
    phase: WheelActionPhase
    complete: bool
    timed_out: bool
    reason: str
    angle_progress_rad: float
    distance_progress_m: float
    use_zero_min_wheel_velocity: bool = True


class WheelTurnAndAdvanceController:
    """Ramp the left wheel, hold a right-wheel speed, turn by IMU, then advance."""

    def __init__(self, profile: WheelActionProfile) -> None:
        self.profile = profile
        self._started_ns: int | None = None
        self._start_heading_rad = 0.0
        self._distance_origin_m: float | None = None
        self._phase = WheelActionPhase.LOWER_LEFT

    @property
    def phase(self) -> WheelActionPhase:
        return self._phase

    @property
    def transition_acceleration_m_s2(self) -> float:
        return self.profile.left_transition_acceleration_m_s2

    def begin(self, *, timestamp_ns: int, heading_rad: float, distance_m: float) -> None:
        if timestamp_ns < 0:
            raise ValueError("timestamp_ns must be non-negative.")
        heading = _finite(heading_rad, "heading_rad")
        distance = _finite(distance_m, "distance_m")
        self._started_ns = timestamp_ns
        self._start_heading_rad = heading
        self._distance_origin_m = None
        self._phase = WheelActionPhase.LOWER_LEFT
        self._initial_distance_m = distance

    def update(self, feedback: WheelActionFeedback) -> WheelActionCommand:
        if self._started_ns is None:
            raise RuntimeError("begin() must precede update().")
        elapsed_s = max(0.0, (feedback.timestamp_ns - self._started_ns) / 1e9)
        angle_progress = normalize_angle(feedback.heading_rad - self._start_heading_rad)
        distance_progress = feedback.distance_m - self._initial_distance_m
        if elapsed_s >= self.profile.action_timeout_s:
            self._phase = WheelActionPhase.TIMEOUT
            return self._command(0.0, 0.0, angle_progress, distance_progress, timed_out=True,
                                 reason="wheel_action_timeout")
        if feedback.telemetry_age_s > self.profile.max_telemetry_age_s:
            self._phase = WheelActionPhase.WAITING_FEEDBACK
            return self._command(0.0, 0.0, angle_progress, distance_progress,
                                 reason="wheel_action_feedback_stale")

        if self._phase in {WheelActionPhase.LOWER_LEFT, WheelActionPhase.WAITING_FEEDBACK}:
            self._phase = WheelActionPhase.LOWER_LEFT
            if (feedback.left_wheel_velocity_m_s <= self.profile.left_wheel_hold_speed_m_s + self.profile.speed_tolerance_m_s
                    and abs(feedback.right_wheel_velocity_m_s - self.profile.right_wheel_speed_m_s)
                    <= self.profile.speed_tolerance_m_s):
                self._phase = WheelActionPhase.HOLD_ANGLE
            else:
                return self._command(
                    self.profile.left_wheel_hold_speed_m_s,
                    self.profile.right_wheel_speed_m_s,
                    angle_progress,
                    distance_progress,
                    reason="left_wheel_decelerating",
                )

        if self._phase is WheelActionPhase.HOLD_ANGLE:
            if angle_progress + 1e-9 < self.profile.target_angle_rad:
                return self._command(
                    self.profile.left_wheel_hold_speed_m_s,
                    self.profile.right_wheel_speed_m_s,
                    angle_progress,
                    distance_progress,
                    reason="right_wheel_held_angle_closing",
                )
            self._phase = WheelActionPhase.RAISE_LEFT

        if self._phase is WheelActionPhase.RAISE_LEFT:
            if (feedback.left_wheel_velocity_m_s
                    >= self.profile.left_wheel_final_speed_m_s - self.profile.speed_tolerance_m_s
                    and abs(feedback.right_wheel_velocity_m_s - self.profile.right_wheel_speed_m_s)
                    <= self.profile.speed_tolerance_m_s):
                self._distance_origin_m = feedback.distance_m
                self._phase = WheelActionPhase.DRIVE_DISTANCE
            else:
                return self._command(
                    self.profile.left_wheel_final_speed_m_s,
                    self.profile.right_wheel_speed_m_s,
                    angle_progress,
                    distance_progress,
                    reason="left_wheel_accelerating",
                )

        if self._phase is WheelActionPhase.DRIVE_DISTANCE:
            assert self._distance_origin_m is not None
            post_distance = feedback.distance_m - self._distance_origin_m
            remaining = self.profile.post_turn_distance_m - post_distance
            speed = max(
                abs(feedback.left_wheel_velocity_m_s),
                abs(feedback.right_wheel_velocity_m_s),
                self.profile.right_wheel_speed_m_s,
            )
            telemetry_delay_s = (
                max(0.0, feedback.telemetry_age_s)
                + self.profile.telemetry_delay_s
            )
            braking_distance = speed * telemetry_delay_s + speed * speed / (
                2.0 * self.profile.linear_deceleration_m_s2
            )
            if remaining > braking_distance:
                return self._command(
                    self.profile.left_wheel_final_speed_m_s,
                    self.profile.right_wheel_speed_m_s,
                    angle_progress,
                    post_distance,
                    reason="both_wheels_fixed_distance_cruise",
                )
            self._phase = WheelActionPhase.STOPPING

        if self._phase is WheelActionPhase.STOPPING:
            assert self._distance_origin_m is not None
            post_distance = feedback.distance_m - self._distance_origin_m
            stationary_duration = (
                0.0 if feedback.stationary_since_ns is None or feedback.stationary_latest_ns is None
                else max(0.0, (feedback.stationary_latest_ns - feedback.stationary_since_ns) / 1e9)
            )
            stopped = (
                max(abs(feedback.left_wheel_velocity_m_s), abs(feedback.right_wheel_velocity_m_s))
                <= self.profile.stop_wheel_speed_m_s
                and abs(feedback.angular_velocity_rad_s) <= self.profile.stop_angular_velocity_rad_s
                and feedback.telemetry_age_s <= self.profile.max_telemetry_age_s
            )
            if stopped and stationary_duration >= self.profile.stationary_confirm_time_s:
                angle_error = self.profile.target_angle_rad - angle_progress
                distance_ok = (
                    abs(post_distance - self.profile.post_turn_distance_m)
                    <= self.profile.distance_tolerance_m
                )
                angle_ok = abs(angle_error) <= self.profile.angle_tolerance_rad
                if distance_ok and angle_ok:
                    self._phase = WheelActionPhase.COMPLETE
                    return self._command(0.0, 0.0, angle_progress, post_distance,
                                         reason="wheel_angle_and_distance_stationary_ok", complete=True)
                if not angle_ok:
                    reason = "post_turn_angle_error"
                else:
                    reason = "post_turn_distance_error"
                self._phase = WheelActionPhase.TIMEOUT
                return self._command(0.0, 0.0, angle_progress, post_distance,
                                     reason=reason, timed_out=True)
            return self._command(0.0, 0.0, angle_progress, post_distance,
                                 reason="braking_after_fixed_distance")

        return self._command(0.0, 0.0, angle_progress, distance_progress,
                             reason=self._phase.value)

    def _command(self, left: float, right: float, angle: float, distance: float,
                 *, reason: str, complete: bool = False, timed_out: bool = False) -> WheelActionCommand:
        return WheelActionCommand(
            left_wheel_velocity_m_s=left,
            right_wheel_velocity_m_s=right,
            phase=self._phase,
            complete=complete,
            timed_out=timed_out,
            reason=(f"{reason}:angle={angle:.4f},distance={distance:.4f}"),
            angle_progress_rad=angle,
            distance_progress_m=distance,
        )


__all__ = [
    "WheelActionCommand",
    "WheelActionFeedback",
    "WheelActionPhase",
    "WheelActionProfile",
    "WheelTurnAndAdvanceController",
]
