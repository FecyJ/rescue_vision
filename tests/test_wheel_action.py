"""Closed-loop wheel-turn action regression tests."""

from __future__ import annotations

import math

from rescue_vision.motion import (
    WheelActionFeedback,
    WheelActionPhase,
    WheelActionProfile,
    WheelTurnAndAdvanceController,
)


def feedback(
    timestamp_ns: int,
    *,
    heading: float,
    distance: float,
    left: float,
    right: float,
    gyro: float = 0.0,
    stationary_since_ns: int | None = None,
) -> WheelActionFeedback:
    return WheelActionFeedback(
        timestamp_ns=timestamp_ns,
        heading_rad=heading,
        distance_m=distance,
        left_wheel_velocity_m_s=left,
        right_wheel_velocity_m_s=right,
        angular_velocity_rad_s=gyro,
        telemetry_age_s=0.0,
        stationary_since_ns=stationary_since_ns,
        stationary_latest_ns=(
            None if stationary_since_ns is None else timestamp_ns
        ),
    )


def test_wheel_turn_ramps_left_holds_angle_then_drives_and_stops() -> None:
    controller = WheelTurnAndAdvanceController(
        WheelActionProfile(
            wheel_track_m=0.235,
            target_angle_rad=math.pi / 3.0,
            post_turn_distance_m=1.0,
            stationary_confirm_time_s=0.1,
        )
    )
    controller.begin(timestamp_ns=0, heading_rad=0.0, distance_m=0.0)

    lowering = controller.update(
        feedback(0, heading=0.0, distance=0.0, left=1.5, right=1.5)
    )
    assert lowering.phase is WheelActionPhase.LOWER_LEFT
    assert (lowering.left_wheel_velocity_m_s, lowering.right_wheel_velocity_m_s) == (1.0, 1.5)

    holding = controller.update(
        feedback(1_000_000_000, heading=0.0, distance=0.0, left=1.0, right=1.5)
    )
    assert holding.phase is WheelActionPhase.HOLD_ANGLE
    assert holding.left_wheel_velocity_m_s == 1.0

    raising = controller.update(
        feedback(2_000_000_000, heading=math.pi / 3.0, distance=0.0, left=1.0, right=1.5)
    )
    assert raising.phase is WheelActionPhase.RAISE_LEFT
    assert raising.left_wheel_velocity_m_s == 1.5

    cruising = controller.update(
        feedback(3_000_000_000, heading=math.pi / 3.0, distance=0.0, left=1.5, right=1.5)
    )
    assert cruising.phase is WheelActionPhase.DRIVE_DISTANCE
    assert cruising.left_wheel_velocity_m_s == 1.5

    braking = controller.update(
        feedback(4_000_000_000, heading=math.pi / 3.0, distance=0.30, left=1.5, right=1.5)
    )
    assert braking.phase is WheelActionPhase.STOPPING
    assert (braking.left_wheel_velocity_m_s, braking.right_wheel_velocity_m_s) == (0.0, 0.0)

    complete = controller.update(
        feedback(
            5_000_000_000,
            heading=math.pi / 3.0,
            distance=1.0,
            left=0.0,
            right=0.0,
            stationary_since_ns=4_800_000_000,
        )
    )
    assert complete.phase is WheelActionPhase.COMPLETE
    assert complete.complete
    assert not complete.timed_out


def test_wheel_turn_does_not_finish_from_angle_before_stationary_distance_check() -> None:
    controller = WheelTurnAndAdvanceController(
        WheelActionProfile(
            wheel_track_m=0.235,
            target_angle_rad=1.0,
            post_turn_distance_m=1.0,
        )
    )
    controller.begin(timestamp_ns=0, heading_rad=0.0, distance_m=0.0)
    controller.update(feedback(0, heading=0.0, distance=0.0, left=1.0, right=1.5))
    command = controller.update(
        feedback(1_000_000_000, heading=1.0, distance=0.0, left=1.0, right=1.5)
    )
    assert command.phase is WheelActionPhase.RAISE_LEFT
    assert not command.complete
