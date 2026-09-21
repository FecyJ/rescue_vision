"""Bounded planar goal feedback, sharing relative-action braking and settling.

Coordinates are supplied by the caller's pose authority. No sensors, threads,
map, or independent odometry live here. The caller must validate the predicted
sweep before executing a command (including turns and reverse corrections).
"""
from __future__ import annotations

import math
from dataclasses import replace

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization.types import FieldPose2D, normalize_angle
from rescue_vision.motion.relative_action import (
    RelativeActionCommand, RelativeActionController, RelativeActionFeedback,
    RelativeActionKind, RelativeActionPhase, RelativeActionProfile, _positive, _finite,
)


class PointActionController(RelativeActionController):
    """Polar point follower with curvature-limited speed and bounded correction.

    The steering law is the waypoint (k_phi=0) form of smooth polar feedback:
    kappa = (k_delta * alpha + sin(alpha)) / r. Large heading errors are
    resolved at zero translation. Terminal orientation is handled only after
    position convergence. RelativeActionController owns braking/stop evidence.
    """

    def __init__(self, profile: RelativeActionProfile, *, wheel_track_m: float,
                 max_wheel_speed_m_s: float, max_angular_velocity_rad_s: float,
                 left_wheel_weight: float = 1.0, right_wheel_weight: float = 1.0) -> None:
        super().__init__(profile)
        self.wheel_track_m = _positive(wheel_track_m, "wheel_track_m")
        self.max_wheel_speed_m_s = _positive(max_wheel_speed_m_s, "max_wheel_speed_m_s")
        self.max_angular_velocity_rad_s = _positive(max_angular_velocity_rad_s, "max_angular_velocity_rad_s")
        self.left_wheel_weight = _positive(left_wheel_weight, "left_wheel_weight")
        self.right_wheel_weight = _positive(right_wheel_weight, "right_wheel_weight")
        self.goal: FieldPoint | None = None
        self.final_heading_rad: float | None = None
        self._point_origin: FieldPoint | None = None
        self._point_correction_ns: int | None = None
        self._point_reverse = False
        self.straight_approach = False

    def begin_point(self, target: FieldPoint, *, pose: FieldPose2D, timestamp_ns: int,
                    cruise_speed_m_s: float, final_heading_rad: float | None = None) -> None:
        if not isinstance(target, FieldPoint) or not isinstance(pose, FieldPose2D):
            raise TypeError(f"Expected FieldPoint and FieldPose2D, got {target!r}, {pose!r}")
        for value in (target.x, target.y, pose.position.x, pose.position.y, pose.heading_rad):
            _finite(value, "point goal/pose")
        self.goal = target
        self.final_heading_rad = (None if final_heading_rad is None else
                                  normalize_angle(_finite(final_heading_rad, "final_heading_rad")))
        self._point_origin = pose.position
        self._point_correction_ns = None
        self._point_reverse = False
        self.straight_approach = False
        distance = math.hypot(target.x - pose.position.x, target.y - pose.position.y) / 1000
        self.begin(RelativeActionKind.STRAIGHT, max(distance, 1e-9),
                   cruise_speed=cruise_speed_m_s, timestamp_ns=timestamp_ns,
                   start_heading_rad=pose.heading_rad, target_heading_rad=pose.heading_rad)

    def begin_relative_point(self, target: GroundPoint, *, pose: FieldPose2D,
                             timestamp_ns: int, cruise_speed_m_s: float,
                             final_heading_rad: float | None = None) -> None:
        """Freeze the robot-relative point exactly once in the supplied frame."""
        if not isinstance(target, GroundPoint):
            raise TypeError(f"Expected GroundPoint, got {target!r}")
        c, s = math.cos(pose.heading_rad), math.sin(pose.heading_rad)
        self.begin_point(FieldPoint(pose.position.x + c * target.x - s * target.y,
                                    pose.position.y + s * target.x + c * target.y),
                         pose=pose, timestamp_ns=timestamp_ns,
                         cruise_speed_m_s=cruise_speed_m_s, final_heading_rad=final_heading_rad)

    def _geometry(self, pose: FieldPose2D) -> tuple[float, float]:
        assert self.goal is not None
        dx, dy = self.goal.x - pose.position.x, self.goal.y - pose.position.y
        alpha = normalize_angle(math.atan2(dy, dx) - pose.heading_rad
                                - (math.pi if self._point_reverse else 0.0))
        return math.hypot(dx, dy) / 1000, alpha

    def _steer(self, pose: FieldPose2D, speed: float) -> tuple[float, float]:
        distance, alpha = self._geometry(pose)
        turn_threshold = self._heading_tolerance() if self.straight_approach else math.pi / 3
        if abs(alpha) > turn_threshold:
            angular = math.copysign(min(self.max_angular_velocity_rad_s,
                self.profile.heading_kp_rad_s * abs(alpha),
                math.sqrt(2 * self.profile.angular_deceleration_rad_s2 * abs(alpha))), alpha)
            return self._limit_wheels(0.0, angular)
        curvature = (self.profile.heading_kp_rad_s * alpha + math.sin(alpha)) / max(distance, 0.02)
        speed = min(speed, self.max_angular_velocity_rad_s / max(abs(curvature), 1e-9))
        return self._limit_wheels(-speed if self._point_reverse else speed, speed * curvature)

    def _limit_wheels(self, linear: float, angular: float) -> tuple[float, float]:
        peak = max(abs((linear - angular * self.wheel_track_m / 2) * self.left_wheel_weight),
                   abs((linear + angular * self.wheel_track_m / 2) * self.right_wheel_weight))
        scale = min(1.0, self.max_wheel_speed_m_s / max(peak, 1e-9))
        return linear * scale, angular * scale

    def update_point(self, feedback: RelativeActionFeedback, pose: FieldPose2D) -> RelativeActionCommand:
        if not self.active or self.goal is None or self._point_origin is None:
            raise RuntimeError("begin_point() must precede update_point().")
        for value in (pose.position.x, pose.position.y, pose.heading_rad):
            _finite(value, "point feedback pose")
        distance, alpha = self._geometry(pose)
        inside = distance <= self._position_tolerance()
        self._target_heading_rad = (self.final_heading_rad if inside and self.final_heading_rad is not None
                                   else pose.heading_rad)
        # Sharing the 1-D executor means telemetry validity, total deadline,
        # braking and real stop confirmation have exactly one implementation.
        command = super().update(replace(feedback, progress=self._target_abs - distance,
                                         heading_rad=pose.heading_rad))
        if command.timed_out or command.phase is RelativeActionPhase.WAITING_FEEDBACK:
            return command
        gx, gy = self.goal.x - self._point_origin.x, self.goal.y - self._point_origin.y
        crossed = ((pose.position.x - self.goal.x) * gx +
                   (pose.position.y - self.goal.y) * gy) > 0
        if not inside and crossed and self._point_correction_ns is None:
            self._point_correction_ns = feedback.timestamp_ns
        if self._point_correction_ns is not None and not inside:
            if not self._point_reverse:
                # Keep the point follower live across the goal plane. Wheel
                # reversal remains acceleration-limited by MotionController.
                self._point_reverse = abs(alpha) > math.pi / 2
        if inside:
            return command
        linear, angular = self._steer(pose, abs(command.linear_velocity_m_s))
        if self._point_correction_ns is not None:
            linear = math.copysign(min(abs(linear), self.profile.fine_linear_speed_m_s), linear)
            command = replace(command, phase=RelativeActionPhase.FINE)
        self._last_linear_command_m_s, self._last_angular_command_rad_s = linear, angular
        return replace(command, linear_velocity_m_s=linear, angular_velocity_rad_s=angular,
                       heading_error_rad=alpha, use_zero_min_wheel_velocity=True,
                       reason=("point_correction:" if self._point_correction_ns is not None
                               else "point_tracking:") + command.reason)

    def predicted_path(self, pose: FieldPose2D) -> tuple[FieldPose2D, ...]:
        """Bounded full geometric route, including rotation; empty if unresolved.

        Each step moves at most 25 mm or turns at most 0.08 rad. Callers inflate
        the sweep for this discretization and the actual braking distance.
        This is a geometric prediction, not another acceleration executor.
        """
        result = [pose]
        for _ in range(320):
            distance, _ = self._geometry(pose)
            if distance <= self._position_tolerance():
                return tuple(result)
            linear, angular = self._steer(pose, min(self._cruise_speed, max(0.03, distance)))
            dt = min(0.1, 0.025 / max(abs(linear), 1e-9), 0.08 / max(abs(angular), 1e-9))
            mid = pose.heading_rad + angular * dt / 2
            pose = FieldPose2D(FieldPoint(pose.position.x + 1000 * linear * dt * math.cos(mid),
                                          pose.position.y + 1000 * linear * dt * math.sin(mid)),
                               normalize_angle(pose.heading_rad + angular * dt))
            result.append(pose)
        return ()
