"""Far breakup must align delayed geometry and actually penetrate the contact core."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app.match import MatchState
from rescue_vision.geometry.types import GroundPoint
from rescue_vision.perception import TargetClass
from test_gripper_width_sequence import motion_sample
from test_match import observation, snapshot
from test_match_breakup import sequence, frozen_plan, tick
from test_match_breakup_latency import scene


@pytest.mark.parametrize('delay_ms,period_ms,poll_ms', [(300,250,5),(600,400,10)])
@pytest.mark.parametrize('confirmation_frames', [1, 3])
def test_delayed_rotating_search_aligns_small_far_core_then_pushes(delay_ms, period_ms, poll_ms, confirmation_frames):
    seq = sequence()
    # Exercise delayed far-ray alignment with an explicitly longer test action;
    # the production 0.5 m action must reject a core beyond its contact reach.
    seq.config = replace(seq.config, breakup_confirmation_frames=confirmation_frames,
                         breakup_forward_distance_m=0.8)
    specs = ((700., 130., TargetClass.BLACK_CORE, 10.),
             (750., 160., TargetClass.GREEN_SUPPLY, 40.))
    heading = -0.15
    distance = 0.
    omega = speed = 0.
    history = {}
    latest = None
    frozen = None
    start_push = None
    for ms in range(0, 12000, poll_ms):
        now = ms*1_000_000
        history[ms] = heading
        seq.observe_grasp_motion(motion_sample(now, count=round(distance*10000), gyro=round(omega*1e6)))
        if ms >= delay_ms and (ms-delay_ms) % period_ms == 0:
            capture_ms = ms-delay_ms
            frame = capture_ms//period_ms+1
            members = scene(frame,capture_ms*1_000_000,specs,history[capture_ms])
            if frame % 2:
                members += scene(frame,capture_ms*1_000_000,
                    ((730.,220.,TargetClass.GREEN_SUPPLY,70.),),history[capture_ms])
            latest = snapshot(frame,capture_ms*1_000_000,*members)
        decision = seq.step(now, perception=latest, heading_rad=heading,
            cumulative_distance_m=distance, left_speed_feedback_m_s=speed,
            right_speed_feedback_m_s=speed)
        if decision.reason == 'breakup_plan_frozen':
            frozen = seq._breakup_plan
            start_push = distance
            assert abs(frozen.heading_rad-heading) <= seq._breakup_alignment_tolerance(frozen)
            # Compensation keeps the physical field aim fixed through the turn.
            assert any(math.hypot(frozen.aim_field.x-x,frozen.aim_field.y-y)<1.
                       for x,y,_,_ in specs)
        if decision.state is MatchState.BREAKUP_BACKWARD:
            assert frozen is not None
            assert (distance-start_push)*1000 >= frozen.forward_distance_mm-1.
            break
        omega, speed = decision.angular_velocity_rad_s, decision.linear_velocity_m_s
        heading += omega*poll_ms/1000
        distance += speed*poll_ms/1000
    else:
        pytest.fail(f'No completed push: {decision.reason}, state={seq.state}, ms={ms}')


def test_forward_keeps_moving_inside_old_braking_margin():
    seq = sequence()
    plan = frozen_plan(seq, forward=200.)
    seq.state = MatchState.BREAKUP_FORWARD
    for travelled in (181., 190., 198.):
        decision = tick(seq,int(travelled),travelled,distance=travelled/1000,observations=())
        assert decision.linear_velocity_m_s > 0
        assert decision.state is MatchState.BREAKUP_FORWARD
    assert plan.forward_distance_mm == 200.


def test_missing_capture_pose_cannot_create_current_breakup_ray():
    seq = sequence()
    for frame in (1,2):
        capture = frame*10_000_000
        seq._tracker.update(capture,(observation(frame,capture,GroundPoint(700,130)),
            observation(frame,capture,GroundPoint(750,160),box_x=40)))
        seq._latest_perception = snapshot(frame,capture)
    seq._latest_heading_rad = .5
    assert seq._breakup_targets(600_000_000) == ()
