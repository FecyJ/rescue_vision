"""Regressions from the 1818 replay: delayed scenes and real motion progress."""
from __future__ import annotations

from dataclasses import replace

import pytest

from rescue_vision.app.gripper_width_sequence import GraspPreparationSession, GripperWidthPickupState
from rescue_vision.app.match import MatchState
from rescue_vision.app.near_field_grasp import GraspTargetTracker, NearFieldHandoffPrior
from rescue_vision.geometry.types import GroundPoint
from rescue_vision.tracking import TrackingConfig
from test_gripper_width_sequence import motion_sample, prep, sequence, snapshot_at
from test_match_near_field import _sequence
from test_near_field_grasp import GREEN, projector, selector, target


@pytest.mark.parametrize('poll_ms,delay_ms', [(5,300),(10,600)])
def test_turn_completion_does_not_follow_poll_clock(poll_ms, delay_ms):
    pickup = sequence(max_observation_age_ms=1000)
    initial = selector().select((target(y=200),)).plan
    pickup.step(0, prep(initial), cumulative_distance_m=0, heading_rad=0)
    pickup.step(1_600_000_000, None, cumulative_distance_m=0,
                heading_rad=initial.alignment_angle_rad)
    completed = pickup._alignment_completed_ns
    for ms in range(1600+poll_ms, 1700+delay_ms, poll_ms):
        decision = pickup.step(ms*1_000_000, None, cumulative_distance_m=0,
                               heading_rad=initial.alignment_angle_rad-.002)
        assert decision.angular_velocity_rad_s == 0
        assert pickup._alignment_completed_ns == completed
    newer = selector().select((target(y=180, timestamp=1_700_000_000, frame=2),)).plan
    result = replace(prep(newer,1_700_000_000), prepared_timestamp_ns=(1700+delay_ms)*1_000_000)
    decision = pickup.step((1700+delay_ms)*1_000_000, result, cumulative_distance_m=0,
                           heading_rad=initial.alignment_angle_rad)
    assert decision.angular_velocity_rad_s > 0
    assert pickup._alignment_attempts == 2


@pytest.mark.parametrize('poll_ms,delay_ms', [(5,300),(10,600)])
def test_real_single_scene_opens_then_corrects_startup_yaw_without_new_frame(poll_ms, delay_ms):
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(TrackingConfig(1,80,.1,500,1,.1).build_tracker(),
                           projector(),planner.config),planner)
    pickup = sequence(gripper_full_travel_time_s=1, max_observation_age_ms=1000)
    capture = 100_000_000
    published = capture+delay_ms*1_000_000
    scene = snapshot_at(1,capture,(target(),))
    for now in range(0,published+1,poll_ms*1_000_000):
        pickup.observe_motion(motion_sample(now))
    result = replace(session.update(scene,locked_ids=None),prepared_timestamp_ns=published)
    opening = pickup.step(published,result,cumulative_distance_m=0,heading_rad=-1.70)
    assert opening.state is GripperWidthPickupState.OPENING
    moving = pickup.step(published+pickup.travel_ns,None,cumulative_distance_m=0,heading_rad=-1.75)
    assert moving.linear_velocity_m_s > 0
    assert moving.angular_velocity_rad_s > 0  # right drift must command left correction
    same = pickup.step(published+pickup.travel_ns+10_000_000,None,cumulative_distance_m=.01,heading_rad=-1.70)
    assert same.angular_velocity_rad_s == pytest.approx(0)
    assert pickup.active_plan.member_ids == result.selection.plan.member_ids


def test_far_handoff_continues_checked_approach_without_new_session():
    flow = _sequence()
    flow._started = True
    flow._latest_heading_rad = 0
    flow._latest_cumulative_distance_m = 0
    member = target(x=758)
    flow._latest_perception = snapshot_at(1,0,(member,))
    tracked, = flow._tracker.update(0,flow._latest_perception.observations)
    prior = NearFieldHandoffPrior(GREEN,member.observation.ground_point,tracked.track_id)
    session = flow.near_field_session_id
    decision = flow._begin_near_field_grasp(0,handoff_prior=prior)
    assert decision.state is MatchState.TRANSPORT_APPROACH_GREEN
    assert flow.near_field_session_id == session
    assert flow.selected_track_id == tracked.track_id


def test_failed_local_recovery_selects_other_visible_safe_target_before_search():
    flow = _sequence(transports=1)
    flow._started = True
    flow._latest_heading_rad = 0
    flow._latest_cumulative_distance_m = 0
    members = (target(x=260,y=-150),target(2,x=650,y=200))
    flow._latest_perception = snapshot_at(1,0,members)
    first, other = flow._tracker.update(0,flow._latest_perception.observations)
    flow._begin_near_field_grasp(0,handoff_prior=NearFieldHandoffPrior(GREEN,first.ground_point,first.track_id))
    decision = flow._return_to_near_field_search(0,'no_safe_grasp_or_recovery')
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert flow.selected_track_id == other.track_id
    assert flow._target_attempt_blocked(first,0)
    assert not flow._target_attempt_blocked(other,0)


def test_formal_turn_keeps_imu_progress_when_no_preparation_returns():
    flow = _sequence(max_observation_age_ms=1000)
    flow._latest_heading_rad = 0
    initial = selector().select((target(y=200),)).plan
    started = flow._step_near_field_grasp(0,0,prep(initial,session_id=1),True)
    assert started.angular_velocity_rad_s > 0
    flow._latest_heading_rad = .2
    waiting = flow._step_near_field_grasp(600_000_000,0,None,True)
    # Missing preparation used to hide the available IMU and expire at 400 ms.
    assert waiting.angular_velocity_rad_s > 0
    assert not flow._near_field_pickup._alignment_finished
    flow._latest_heading_rad = initial.alignment_angle_rad
    done = flow._step_near_field_grasp(1_600_000_000,0,None,True)
    assert done.angular_velocity_rad_s == 0
    assert flow._near_field_pickup._alignment_finished
