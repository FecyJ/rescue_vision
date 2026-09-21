from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import FieldPose2D, normalize_angle
from rescue_vision.motion.point_action import PointActionController
from rescue_vision.motion.relative_action import RelativeActionProfile, RelativeActionPhase
from test_relative_action import feedback


def controller() -> PointActionController:
    result = PointActionController(RelativeActionProfile(action_timeout_s=15),
        wheel_track_m=0.235, max_wheel_speed_m_s=1.5, max_angular_velocity_rad_s=1.2)
    result.set_tolerances(position_tolerance=0.02, heading_tolerance=0.02)
    return result


@pytest.mark.parametrize("heading,target", [
    (0.0, FieldPoint(1800, 450)), (2.8, FieldPoint(1000, 0)),
    (-3.12, FieldPoint(-1200, -50)), (0.0, FieldPoint(40, 20)),
])
@pytest.mark.parametrize("dt", [0.005, 0.01])
def test_point_reaches_goal_with_bounded_wheels(heading, target, dt):
    action = controller()
    pose = FieldPose2D(FieldPoint(0, 0), heading)
    action.begin_point(target, pose=pose, timestamp_ns=0, cruise_speed_m_s=1.2)
    since = None
    linear = angular = 0.0
    for i in range(round(15 / dt)):
        now = round(i * dt * 1e9)
        if abs(linear) < 0.015 and abs(angular) < 0.06:
            since = now if since is None else since
        else:
            since = None
        command = action.update_point(feedback(now, progress=0, heading_rad=pose.heading_rad,
            left=linear-angular*0.235/2, right=linear+angular*0.235/2, angular=angular,
            stationary_since_ns=since, stationary_latest_ns=now), pose)
        assert not command.timed_out, command.reason
        assert max(abs(command.linear_velocity_m_s-command.angular_velocity_rad_s*0.235/2),
                   abs(command.linear_velocity_m_s+command.angular_velocity_rad_s*0.235/2)) <= 1.5 + 1e-9
        if command.complete:
            assert math.hypot(target.x-pose.position.x, target.y-pose.position.y) <= 20
            break
        # The actuator slew belongs to the plant, not to the position controller.
        linear += max(-dt, min(dt, command.linear_velocity_m_s-linear))
        angular += max(-2*dt, min(2*dt, command.angular_velocity_rad_s-angular))
        mid = pose.heading_rad + angular*dt/2
        pose = FieldPose2D(FieldPoint(pose.position.x+1000*linear*dt*math.cos(mid),
                                     pose.position.y+1000*linear*dt*math.sin(mid)),
                           normalize_angle(pose.heading_rad+angular*dt))
    else:
        pytest.fail(f"No completion: {pose}, {command}")


def test_crossing_goal_plane_with_lateral_error_keeps_correcting():
    action = controller()
    action.begin_point(FieldPoint(1000, 0), pose=FieldPose2D(FieldPoint(0, 0), 0),
                       timestamp_ns=0, cruise_speed_m_s=1)
    result = action.update_point(feedback(1_000_000_000, progress=0,
        stationary_since_ns=0, stationary_latest_ns=1_000_000_000),
        FieldPose2D(FieldPoint(1010, 150), 0))
    assert not result.complete
    assert not result.timed_out
    assert result.phase is RelativeActionPhase.FINE
    assert result.linear_velocity_m_s != 0 or result.angular_velocity_rad_s != 0


def test_small_goal_plane_crossing_corrects_without_stop_gate():
    action = controller()
    action.begin_point(FieldPoint(1000, 0), pose=FieldPose2D(FieldPoint(0, 0), 0),
                       timestamp_ns=0, cruise_speed_m_s=1)
    result = action.update_point(feedback(100_000_000, progress=0,
        left=0.04, right=0.04, angular=0.0),
        FieldPose2D(FieldPoint(1020, 10), 0))
    assert not result.complete
    assert not result.timed_out
    assert result.phase is RelativeActionPhase.FINE
    assert result.linear_velocity_m_s < 0.0
    assert "waiting_stop" not in result.reason


def test_relative_point_is_frozen_and_missing_telemetry_never_completes():
    action = controller()
    action.begin_relative_point(GroundPoint(100, 0),
        pose=FieldPose2D(FieldPoint(200, 300), math.pi/2), timestamp_ns=0, cruise_speed_m_s=0.3)
    assert action.goal.x == pytest.approx(200)
    assert action.goal.y == pytest.approx(400)
    result = action.update_point(replace(feedback(1_000_000_000, progress=0), telemetry_age_s=None),
                                 FieldPose2D(action.goal, math.pi/2))
    assert not result.complete
    assert result.phase is RelativeActionPhase.WAITING_FEEDBACK


def test_existing_stationary_origin_is_preserved_and_final_heading_is_required():
    action = controller()
    goal = FieldPoint(0, 0)
    action.begin_point(goal, pose=FieldPose2D(goal, 0), timestamp_ns=1_000_000_000,
                       cruise_speed_m_s=0.3, final_heading_rad=0.1)
    sample = feedback(1_100_000_000, progress=0, stationary_since_ns=0,
                      stationary_latest_ns=1_100_000_000)
    turning = action.update_point(sample, FieldPose2D(goal, 0))
    assert not turning.complete
    assert turning.angular_velocity_rad_s > 0
    done = action.update_point(sample, FieldPose2D(goal, 0.1))
    assert done.complete


def test_full_route_prediction_is_bounded_and_reaches_target():
    action = controller()
    pose = FieldPose2D(FieldPoint(0, 0), math.pi)
    action.begin_point(FieldPoint(1200, 300), pose=pose, timestamp_ns=0, cruise_speed_m_s=1)
    route = action.predicted_path(pose)
    assert 2 < len(route) <= 321
    assert math.hypot(route[-1].position.x-1200, route[-1].position.y-300) <= 20
