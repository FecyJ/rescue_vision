from __future__ import annotations

import pytest

from rescue_vision.geometry.types import GroundPoint
from rescue_vision.motion import GripperKinematics


def test_closed_tip_positions_match_the_given_geometry() -> None:
    kinematics = GripperKinematics()

    assert kinematics.left_tip_position(0.0) == GroundPoint(157.5, 0.0)
    assert kinematics.right_tip_position(0.0) == GroundPoint(157.5, 0.0)
    assert kinematics.symmetric_opening_width_mm(0.0) == pytest.approx(0.0)


def test_symmetric_opening_matches_the_rotation_matrix_geometry() -> None:
    kinematics = GripperKinematics()

    left = kinematics.left_tip_position(20.0)
    right = kinematics.right_tip_position(20.0)

    assert left.x == pytest.approx(176.8192359)
    assert left.y == pytest.approx(40.4351685)
    assert right.x == pytest.approx(176.8192359)
    assert right.y == pytest.approx(-40.4351685)
    assert kinematics.symmetric_opening_width_mm(20.0) == pytest.approx(80.8703370)


def test_opening_width_round_trips_to_symmetric_servo_angles() -> None:
    kinematics = GripperKinematics()
    opening = kinematics.symmetric_opening_width_mm(20.0)

    left, right = kinematics.servo_angles_for_opening(
        opening,
        open_left_angle_deg=0.0,
        open_right_angle_deg=180.0,
        closed_left_angle_deg=80.0,
        closed_right_angle_deg=100.0,
    )

    assert left == pytest.approx(60.0)
    assert right == pytest.approx(120.0)


def test_servo_angle_conversion_rejects_an_overwide_target() -> None:
    with pytest.raises(ValueError, match="exceeds"):
        GripperKinematics().servo_angles_for_opening(
            1_000.0,
            open_left_angle_deg=0.0,
            open_right_angle_deg=180.0,
            closed_left_angle_deg=80.0,
            closed_right_angle_deg=100.0,
        )


def test_servo_angle_conversion_rejects_wrong_servo_direction() -> None:
    with pytest.raises(ValueError, match="left servo direction"):
        GripperKinematics().servo_angles_for_opening(
            20.0,
            open_left_angle_deg=100.0,
            open_right_angle_deg=180.0,
            closed_left_angle_deg=80.0,
            closed_right_angle_deg=100.0,
        )
