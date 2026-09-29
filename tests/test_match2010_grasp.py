"""2010: keep the selected black core after a delayed physical alignment."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.app.gripper_width_sequence import GraspPreparationSession, GripperWidthPickupState
from rescue_vision.app.near_field_grasp import GraspTargetTracker, NearFieldGraspPolicy
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.localization import FieldPose2D
from rescue_vision.perception import TargetClass
from rescue_vision.tracking import TrackingConfig
from test_gripper_width_sequence import sequence, motion_sample
from test_match import snapshot
from test_match_near_field import _sequence
from test_near_field_grasp import selector, projector, target

B = TargetClass.BLACK_CORE
G = TargetClass.GREEN_SUPPLY
D = TargetClass.BLUE_DANGER


def session():
    planner = selector(confirmation_frames=1)
    return GraspPreparationSession(GraspTargetTracker(
        TrackingConfig(1, 80, .1, 500, 1, .1).build_tracker(), projector(), planner.config), planner)


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250), (10,600,400)])
@pytest.mark.parametrize('intruder', [False, True])
def test_turned_black_core_reassociates_and_advances_without_another_frame(poll_ms, delay_ms, period_ms, intruder):
    planner = session()
    policy = NearFieldGraspPolicy(frozenset({B,G}), max_targets=1)
    angle = .61
    first = target(x=300*math.cos(angle), y=300*math.sin(angle), cls=B,
                   frame=1, timestamp=0).observation
    before = planner.update(snapshot(1, 0, first), locked_ids=None, policy=policy,
                            capture_pose=FieldPose2D(FieldPoint(0,0),0), stationary_since_ns=0)
    assert before.selection.plan is not None
    identity = before.selection.plan.member_ids
    # Perception after a turn contains a new low-value neighbour first, with
    # the tracker ID previously owned by the black block.
    capture = 2_000_000_000
    points = [target(x=300,y=180,cls=G,frame=2,timestamp=capture).observation,
              target(x=300,y=0,cls=B,frame=2,timestamp=capture).observation]
    if intruder:
        points.append(target(x=240,y=0,cls=D,frame=2,timestamp=capture).observation)
    after = planner.update(snapshot(2,capture,*points), locked_ids=identity, policy=policy,
        capture_pose=FieldPose2D(FieldPoint(0,0),angle), stationary_since_ns=capture-100_000_000)
    if intruder:
        assert after.selection.plan is None
        return
    assert after.selection.plan.member_ids == identity
    assert after.selection.plan.members[0].observation.target_class is B
    assert after.selection.plan.alignment_angle_rad == 0
    pickup = sequence(alignment_timeout_ms=1200)
    prepared_at = capture + delay_ms*1_000_000
    after = replace(after, prepared_timestamp_ns=prepared_at, result_timestamp_ns=prepared_at)
    # Control polls much faster than perception; the result arrives between
    # frame publications, and must open/advance using that one valid result.
    for ms in range(1900, 3001, poll_ms):
        now = ms*1_000_000
        pickup.observe_motion(motion_sample(now))
        decision = pickup.step(now, after if now >= prepared_at else None,
                               cumulative_distance_m=0, heading_rad=angle)
        if decision.state is GripperWidthPickupState.FORWARD:
            assert decision.linear_velocity_m_s > 0
            assert now < prepared_at + period_ms*1_000_000
            assert pickup.active_plan.members[0].observation.target_class is B
            break
    else:
        pytest.fail(f'No actual grasp progress: {decision}')


def test_new_scene_core_comparison_uses_capture_pose_not_raw_robot_coordinates():
    seq = _sequence(transports=1)
    seq._started = True
    seq._fallback_field_position = FieldPoint(0,0)
    seq._record_pose_history(0,0,0)
    angle = .61
    seq._record_pose_history(1_000_000_000,angle,0)
    member = target(x=300*math.cos(angle),y=300*math.sin(angle),cls=B)
    plan = selector().select((member,)).plan
    assert plan is not None
    seq._latest_perception = snapshot(2,1_000_000_000,
        target(x=300,y=0,cls=B,frame=2,timestamp=1_000_000_000).observation)
    assert seq._latest_scene_preserves_grasp(plan,1_000_000_000)


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250),(10,600,400)])
def test_observation_budget_starts_after_measured_alignment_stop(poll_ms, delay_ms, period_ms):
    planner = selector(confirmation_frames=1)
    pickup = sequence(alignment_timeout_ms=1200)
    initial = planner.select((target(y=150,cls=B,frame=1,timestamp=50_000_000),)).plan
    from test_gripper_width_sequence import prep
    first = prep(initial,50_000_000)
    for ms in range(0, 101, poll_ms):
        pickup.observe_motion(motion_sample(ms*1_000_000))
    decision = pickup.step(100_000_000,first,cumulative_distance_m=0,heading_rad=0)
    assert decision.state is GripperWidthPickupState.ALIGNING
    ready = None
    for ms in range(100+poll_ms, 3601, poll_ms):
        now = ms*1_000_000
        # Turn + mechanical braking take 1.4 s. A new stationary frame then
        # arrives after both camera interval and processing latency.
        pickup.observe_motion(motion_sample(now,gyro=200_000 if ms<1500 else 0))
        heading = initial.alignment_angle_rad if ms>=1200 else initial.alignment_angle_rad*(ms-100)/1100
        capture_ms = 1500+period_ms
        if ready is None and ms >= capture_ms+delay_ms:
            plan = planner.select((target(y=0,cls=B,frame=2,timestamp=capture_ms*1_000_000),)).plan
            ready = replace(prep(plan,capture_ms*1_000_000),prepared_timestamp_ns=now,result_timestamp_ns=now)
        decision = pickup.step(now,ready,cumulative_distance_m=0,heading_rad=heading)
        assert decision.reason != 'alignment_timeout'
        if decision.state is GripperWidthPickupState.FORWARD:
            assert decision.linear_velocity_m_s > 0
            break
    else:
        pytest.fail(f'Late stopped frame never committed: {decision}')
