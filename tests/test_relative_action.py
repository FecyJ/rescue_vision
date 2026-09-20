"""Regression tests for NB's profiled relative-action controller."""

from __future__ import annotations

import pytest

from rescue_vision.motion import (
    RelativeActionController,
    RelativeActionFeedback,
    RelativeActionKind,
    RelativeActionPhase,
    RelativeActionProfile,
)


def feedback(
    timestamp_ns: int,
    *,
    progress: float,
    heading_rad: float = 0.0,
    left: float = 0.0,
    right: float = 0.0,
    angular: float = 0.0,
    age_s: float = 0.0,
    stationary_since_ns: int | None = None,
    stationary_latest_ns: int | None = None,
) -> RelativeActionFeedback:
    return RelativeActionFeedback(
        timestamp_ns=timestamp_ns,
        progress=progress,
        heading_rad=heading_rad,
        left_wheel_velocity_m_s=left,
        right_wheel_velocity_m_s=right,
        angular_velocity_rad_s=angular,
        telemetry_age_s=age_s,
        stationary_since_ns=stationary_since_ns,
        stationary_latest_ns=stationary_latest_ns,
    )


def test_braking_distance_uses_measured_speed_and_all_delay_terms() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(
            linear_deceleration_m_s2=5.0,
            command_wait_s=0.04,
            execution_response_s=0.05,
        )
    )
    controller.set_tolerances(position_tolerance=0.01, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.STRAIGHT,
        1.0,
        start_heading_rad=0.0,
        target_heading_rad=0.0,
        timestamp_ns=0,
        cruise_speed=1.5,
    )

    command = controller.update(
        feedback(
            100_000_000,
            progress=0.70,
            left=1.5,
            right=1.5,
            age_s=0.10,
        )
    )

    assert command.phase is RelativeActionPhase.BRAKE
    assert command.braking_distance == pytest.approx(
        1.5 * (0.10 + 0.04 + 0.05) + 1.5**2 / (2.0 * 5.0)
    )
    assert 0.0 < command.linear_velocity_m_s < 1.5


def test_braking_uses_command_speed_when_fresh_feedback_lags() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(linear_deceleration_m_s2=2.0)
    )
    controller.set_tolerances(position_tolerance=0.03, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.STRAIGHT,
        1.0,
        start_heading_rad=0.0,
        target_heading_rad=0.0,
        timestamp_ns=0,
        cruise_speed=1.5,
    )

    cruise = controller.update(
        feedback(
            100_000_000,
            progress=0.10,
            left=1.5,
            right=1.5,
        )
    )
    assert cruise.linear_velocity_m_s == pytest.approx(1.5)

    command = controller.update(
        feedback(
            200_000_000,
            progress=0.6915,
            left=1.0,
            right=1.0,
            age_s=0.01,
        )
    )

    # The fresh encoder sample is below the previous 1.5 m/s command.  Using
    # only that sample would leave this point in cruise; the conservative
    # command-speed floor moves it into braking before the target.
    assert command.phase is RelativeActionPhase.BRAKE
    assert command.braking_distance == pytest.approx(
        1.5 * (0.01 + 0.04 + 0.04) + 1.5**2 / (2.0 * 2.0)
    )
    assert command.linear_velocity_m_s < 1.5


def test_geometry_alone_never_finishes_and_duplicate_stationary_sample_does_not_count() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(stationary_confirm_time_s=0.10)
    )
    controller.set_tolerances(position_tolerance=0.02, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.TURN,
        0.20,
        start_heading_rad=0.0,
        timestamp_ns=1_000_000_000,
        cruise_speed=0.8,
    )

    inside = controller.update(
        feedback(
            1_100_000_000,
            progress=0.19,
            heading_rad=0.19,
            left=0.0,
            right=0.0,
            angular=0.0,
            stationary_since_ns=1_000_000_000,
            stationary_latest_ns=1_050_000_000,
        )
    )
    assert inside.phase is RelativeActionPhase.SETTLE
    assert not inside.complete

    duplicate = controller.update(
        feedback(
            1_300_000_000,
            progress=0.19,
            heading_rad=0.19,
            stationary_since_ns=1_000_000_000,
            stationary_latest_ns=1_050_000_000,
        )
    )
    assert duplicate.phase is RelativeActionPhase.SETTLE
    assert not duplicate.complete

    complete = controller.update(
        feedback(
            1_300_000_000,
            progress=0.19,
            heading_rad=0.19,
            stationary_since_ns=1_000_000_000,
            stationary_latest_ns=1_200_000_000,
        )
    )
    assert complete.complete
    assert complete.phase is RelativeActionPhase.COMPLETE


def test_turn_inside_angle_tolerance_still_trims_heading_error() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(
            fine_angular_velocity_rad_s=0.12,
            heading_kp_rad_s=2.0,
        )
    )
    controller.set_tolerances(position_tolerance=0.08, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.TURN,
        -0.80,
        start_heading_rad=-1.5707963267948966,
        target_heading_rad=-2.3707963267948966,
        timestamp_ns=0,
        cruise_speed=2.0,
    )

    command = controller.update(
        feedback(
            1_000_000_000,
            progress=0.8563,
            heading_rad=-2.4270820367948973,
            left=0.0,
            right=0.0,
            angular=0.0,
        )
    )

    assert command.phase is RelativeActionPhase.FINE
    assert command.reason.startswith("angle_ok_heading_trim")
    assert command.angular_velocity_rad_s == pytest.approx(0.1125714)
    assert not command.complete


def test_negative_turn_overshoot_correction_reverses_direction() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(fine_angular_velocity_rad_s=0.12)
    )
    controller.set_tolerances(position_tolerance=0.08, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.TURN,
        -0.80,
        start_heading_rad=0.0,
        target_heading_rad=-0.80,
        timestamp_ns=0,
        cruise_speed=2.0,
    )

    command = controller.update(
        feedback(
            100_000_000,
            progress=0.90,
            heading_rad=-0.90,
            left=0.0,
            right=0.0,
            angular=0.0,
        )
    )

    assert command.phase is RelativeActionPhase.FINE
    assert command.reason.startswith("overshoot_correction")
    assert command.angular_velocity_rad_s == pytest.approx(0.12)


def test_negative_straight_overshoot_correction_reverses_direction() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(fine_linear_speed_m_s=0.05)
    )
    controller.set_tolerances(position_tolerance=0.02, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.STRAIGHT,
        -0.50,
        start_heading_rad=0.0,
        target_heading_rad=0.0,
        timestamp_ns=0,
        cruise_speed=1.0,
    )

    command = controller.update(
        feedback(
            100_000_000,
            progress=0.55,
            heading_rad=0.0,
            left=0.0,
            right=0.0,
            angular=0.0,
        )
    )

    assert command.phase is RelativeActionPhase.FINE
    assert command.reason.startswith("overshoot_correction")
    assert command.linear_velocity_m_s == pytest.approx(0.05)


def test_overshoot_waits_for_stop_before_correction_or_limit_failure() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(correction_max_distance_m=0.12)
    )
    controller.set_tolerances(position_tolerance=0.02, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.STRAIGHT,
        1.0,
        start_heading_rad=0.0,
        target_heading_rad=0.0,
        timestamp_ns=0,
        cruise_speed=1.0,
    )

    moving = controller.update(
        feedback(
            100_000_000,
            progress=1.05,
            left=0.04,
            right=0.04,
        )
    )
    assert moving.phase is RelativeActionPhase.SETTLE
    assert moving.reason.startswith("overshoot_waiting_for_stop")
    assert moving.linear_velocity_m_s == 0.0

    stopped = controller.update(
        feedback(
            200_000_000,
            progress=1.05,
            left=0.0,
            right=0.0,
            angular=0.0,
        )
    )
    assert stopped.phase is RelativeActionPhase.FINE
    assert stopped.reason.startswith("overshoot_correction")
    assert stopped.linear_velocity_m_s == pytest.approx(-0.05)


def test_large_overshoot_is_rejected_after_vehicle_stops() -> None:
    controller = RelativeActionController()
    controller.set_tolerances(position_tolerance=0.02, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.STRAIGHT,
        1.0,
        start_heading_rad=0.0,
        target_heading_rad=0.0,
        timestamp_ns=0,
        cruise_speed=1.0,
    )

    command = controller.update(
        feedback(
            100_000_000,
            progress=1.09,
            left=0.0,
            right=0.0,
            angular=0.0,
        )
    )

    assert command.phase is RelativeActionPhase.TIMEOUT
    assert command.reason.startswith("overshoot_correction_limit")
    assert command.timed_out


def test_overshoot_stop_wait_has_a_bounded_deadline() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(correction_max_distance_m=0.12)
    )
    controller.set_tolerances(position_tolerance=0.02, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.STRAIGHT,
        1.0,
        start_heading_rad=0.0,
        target_heading_rad=0.0,
        timestamp_ns=0,
        cruise_speed=1.0,
    )

    waiting = controller.update(
        feedback(
            100_000_000,
            progress=1.05,
            left=0.04,
            right=0.04,
        )
    )
    assert waiting.phase is RelativeActionPhase.SETTLE

    command = controller.update(
        feedback(
            700_000_000,
            progress=1.05,
            left=0.04,
            right=0.04,
        )
    )

    assert command.phase is RelativeActionPhase.TIMEOUT
    assert command.reason.startswith("overshoot_settle_timeout")
    assert command.timed_out


def test_straight_action_holds_route_heading_and_allows_fine_speed_without_wheel_lift() -> None:
    controller = RelativeActionController(
        RelativeActionProfile(
            fine_linear_speed_m_s=0.04,
            heading_kp_rad_s=2.0,
            heading_max_angular_velocity_rad_s=0.20,
        )
    )
    controller.set_tolerances(position_tolerance=0.0001, heading_tolerance=0.03)
    controller.begin(
        RelativeActionKind.STRAIGHT,
        0.10,
        start_heading_rad=0.0,
        target_heading_rad=0.0,
        timestamp_ns=0,
        cruise_speed=1.0,
    )

    command = controller.update(
        feedback(100_000_000, progress=0.0995, heading_rad=0.10)
    )

    assert command.phase is RelativeActionPhase.FINE
    assert command.linear_velocity_m_s == pytest.approx(0.04)
    assert command.angular_velocity_rad_s == pytest.approx(-0.20)
    assert command.use_zero_min_wheel_velocity
