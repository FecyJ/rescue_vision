"""2032 match failure: one physical boundary, one allowance per moving entity."""
from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app.breakup_planner import plan_breakup
from rescue_vision.app.match import MatchState
from rescue_vision.geometry.types import FieldPoint
from rescue_vision.perception import TargetClass as C
from test_gripper_width_sequence import motion_sample
from test_match import snapshot
from test_match_breakup import sequence
from test_match_breakup_latency import feed, scene


HEADING = -1.2095
# Approximate K0 scene around screenshot t=637 s, not a sensor replay.
LOCAL = ((412., -67., C.GREEN_SUPPLY, 10.),
         (450., 0., C.BLUE_DANGER, 40.), (480., 50., C.BLUE_DANGER, 70.))
OFFSETS = tuple((math.cos(HEADING)*x-math.sin(HEADING)*y,
                 math.sin(HEADING)*x+math.cos(HEADING)*y, cls, box)
                for x,y,cls,box in LOCAL)


def test_screenshot_region_has_safe_contact_plans_without_double_inset():
    seq = sequence(initial_field_position=FieldPoint(979., -158.))
    seq._latest_heading_rad = HEADING
    for frame in (1,2):
        capture, members = feed(seq, frame, frame*10, OFFSETS, HEADING)
        seq._latest_perception = snapshot(frame,capture,*members)
    args = dict(config=seq.config, origin=seq.estimated_field_position,
        heading_rad=HEADING, static_map=seq._breakup_static_map,
        front_mm=157.5, allowed_classes=seq.near_field_policy.allowed_classes, approach=False)
    # The old center-only inset was incorrectly fed to another footprint check.
    assert not plan_breakup(seq._breakup_targets(capture),
        field_bounds=seq._breakup_allowed_field_bounds(), **args)
    assert seq._choose_breakup_plan(capture) is not None


@pytest.mark.parametrize('poll_ms,delay_ms,period_ms', [(5,300,250),(10,600,400)])
def test_boundary_cluster_progresses_with_delayed_frames_and_real_settle(poll_ms,delay_ms,period_ms):
    seq = sequence(initial_field_position=FieldPoint(979., -158.))
    seq.config = replace(seq.config, breakup_confirmation_frames=3)
    latest = None
    heading = HEADING
    omega = 0.
    history = {}
    for ms in range(0, 9000, poll_ms):
        now = ms*1_000_000
        history[ms] = heading
        seq.observe_grasp_motion(motion_sample(now,gyro=round(omega*1e6)))
        if ms >= delay_ms and (ms-delay_ms)%period_ms == 0:
            capture_ms = ms-delay_ms
            frame = capture_ms//period_ms+1
            specs = OFFSETS if frame%2 else OFFSETS[:2]
            latest = snapshot(frame,capture_ms*1_000_000,
                              *scene(frame,capture_ms*1_000_000,specs,history[capture_ms]))
        decision = seq.step(now,perception=latest,heading_rad=heading,cumulative_distance_m=0.,
                           left_speed_feedback_m_s=0.,right_speed_feedback_m_s=0.)
        if decision.linear_velocity_m_s > 0:
            assert decision.state is MatchState.BREAKUP_FORWARD
            assert seq._breakup_plan is not None
            assert len(seq._breakup_reference_frames) == 3
            return
        omega = decision.angular_velocity_rad_s
        heading += omega*poll_ms/1000
    pytest.fail(f'No action: {decision.reason}; {seq.cluster_diagnostic(now)}')


@pytest.mark.parametrize('origin', [FieldPoint(1400,0), FieldPoint(0,1400), FieldPoint(-1400,0)])
def test_body_or_jaw_outside_physical_field_still_rejected(origin):
    seq = sequence(initial_field_position=origin)
    assert seq._near_field_segment_clear(0.,.05) is False


def test_legal_translation_near_edge_does_not_double_count_jaw_and_body():
    seq = sequence(initial_field_position=FieldPoint(1180,0))
    assert seq._near_field_segment_clear(-math.pi/2,.2) is True
    assert seq._near_field_segment_clear(0.,.2) is False


def test_rejection_names_danger_entity_and_physical_boundary():
    from test_breakup_planner import plans, target
    rejected = []
    assert not plans([target(1,450,0),target(2,480,0,C.BLUE_DANGER)],
        origin=FieldPoint(1000,0), approach=False, rejections=rejected)
    assert any('blocked_path=' in reason and 'field_boundary' in reason for reason in rejected)


def test_blue_member_near_real_boundary_still_blocks_the_whole_group():
    from test_breakup_planner import plans, target
    rejected = []
    assert not plans([target(1,450,0), target(2,450,-90),target(3,450,-180,C.BLUE_DANGER)],
        origin=FieldPoint(1250,0),heading_rad=math.pi/2,approach=False,rejections=rejected)
    assert any('target_3_blue_danger:field_boundary' in reason for reason in rejected)
