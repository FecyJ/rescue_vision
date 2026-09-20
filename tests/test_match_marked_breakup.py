"""Closed-jaw recovery preserves measured scoring objectives across motion."""
from __future__ import annotations

from dataclasses import replace
import math

import pytest

from rescue_vision.app.match import BreakupMarkedTarget, GripperPosture, MatchState
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from test_match import observation, snapshot
from test_match_breakup import frozen_plan
from test_match_near_field import _sequence


def marked_flow():
    flow = _sequence()
    flow.config = replace(flow.config, green_max_age_ms=800)
    flow._started = True
    flow._latest_heading_rad = 0.0
    flow._record_pose_history(0, 0.0, 0.0)
    flow._adopt_grasp_task(0, track_id=52, target_class=TargetClass.GREEN_SUPPLY,
                          point=GroundPoint(350, 0), field=FieldPoint(350, 0))
    flow.grasp_task.marked_targets = (
        BreakupMarkedTarget(TargetClass.GREEN_SUPPLY, FieldPoint(350, 0), 0, 0, 52),
        BreakupMarkedTarget(TargetClass.BLACK_CORE, FieldPoint(370, 80), 0, 0, 53),
    )
    frozen_plan(flow, forward=500, backward=300)
    flow.state = MatchState.BREAKUP_FORWARD
    return flow


def update_marks(flow, now, capture, distance, observations, *, heading=0.0):
    flow._latest_heading_rad = heading
    flow._fallback_field_position = FieldPoint(distance*1000, 0)
    flow._record_pose_history(capture, heading, distance)
    if now != capture:
        flow._record_pose_history(now, heading, distance)
    flow._latest_perception = replace(snapshot(capture, capture, *observations),
                                      result_timestamp_ns=now, timing=None)
    flow._update_tracker(now, flow._latest_perception)
    flow._update_breakup_marks(now)


def test_startup_enters_normal_search_and_can_grab_first_green_without_breakup():
    flow = _sequence()
    flow._started = True
    flow.state = MatchState.STARTUP_FORWARD_SETTLE
    flow._settle_until_ns = 0
    for frame, now in enumerate((0, 10_000_000, 20_000_000)):
        decision = flow.step(now, perception=snapshot(frame, now,
                             observation(frame, now, GroundPoint(300, 0))),
                             heading_rad=0, cumulative_distance_m=0)
        assert decision.state not in {MatchState.BREAKUP_SETTLE, MatchState.BREAKUP_FORWARD}
    assert flow.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert flow.grasp_task.entry_class is TargetClass.GREEN_SUPPLY
    assert flow.grasp_task.recovery_count == 0


@pytest.mark.parametrize('delay_ms', [300, 600])
def test_marks_follow_measured_push_after_tracker_reset_and_peripheral_loss(delay_ms):
    flow = marked_flow()
    identity, deadline = flow.grasp_task.task_id, flow.grasp_task.deadline_ns
    flow._tracker.reset()
    capture = 400_000_000
    # Green slid 140 mm in the field while the chassis travelled 300 mm.
    update_marks(flow, capture+delay_ms*1_000_000, capture, .3,
                 (observation(capture, capture, GroundPoint(190, 0)),))
    green, black = flow.grasp_task.marked_targets
    assert green.field_point == FieldPoint(490, 0)
    assert green.capture_timestamp_ns == capture
    assert green.track_id != 52
    assert black.field_point == FieldPoint(370, 80)
    assert black.capture_timestamp_ns == 0
    assert (flow.grasp_task.task_id, flow.grasp_task.deadline_ns) == (identity, deadline)
    # Re-reading the same frame does not change the capture time or position.
    flow._update_breakup_marks(capture+delay_ms*1_000_000+5_000_000)
    assert green.capture_timestamp_ns == capture


def test_rotation_compensation_preserves_field_location():
    flow = marked_flow()
    capture = 100_000_000
    angle = .3
    update_marks(flow, capture, capture, 0,
                 (observation(capture, capture,
                              GroundPoint(350*math.cos(angle), -350*math.sin(angle))),),
                 heading=angle)
    mark = flow.grasp_task.marked_targets[0]
    assert mark.field_point.x == pytest.approx(350)
    assert mark.field_point.y == pytest.approx(0)


def test_ambiguous_or_dangerous_observation_does_not_overwrite_mark():
    flow = marked_flow()
    capture = 100_000_000
    update_marks(flow, capture, capture, 0,
                 (observation(capture, capture, GroundPoint(350, 10)),
                  observation(capture, capture, GroundPoint(350, -10), box_x=80),
                  observation(capture, capture, GroundPoint(350, 0),
                              target_class=TargetClass.BLUE_DANGER, box_x=40)))
    mark = flow.grasp_task.marked_targets[0]
    assert mark.field_point == FieldPoint(350, 0)
    assert mark.capture_timestamp_ns == 0


def test_retreat_requires_current_marked_geometry_and_keeps_handoff():
    flow = marked_flow()
    task = flow.grasp_task
    task.scene_floor_ns = 500_000_000
    task.recovery_reobserve_started_ns = 500_000_000
    flow.state = MatchState.CHECK_ISOLATED_GREEN
    flow._breakup_actual_forward_mm = 500
    update_marks(flow, 700_000_000, 400_000_000, .2,
                 (observation(400_000_000, 400_000_000, GroundPoint(290, 20)),))
    old = flow._reacquire_breakup_target(700_000_000)
    assert old.state is MatchState.CHECK_ISOLATED_GREEN
    assert old.gripper_posture is GripperPosture.CLOSED
    update_marks(flow, 1_000_000_000, 800_000_000, .2,
                 (observation(800_000_000, 800_000_000, GroundPoint(290, 20)),))
    decision = flow._reacquire_breakup_target(1_000_000_000)
    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert flow.grasp_task is task
    assert flow.near_field_handoff_prior.ground_point == GroundPoint(290, 20)
    assert flow.near_field_handoff_required
    assert task.entry_field == FieldPoint(490, 20)


@pytest.mark.parametrize('poll_ms', [5, 10])
def test_missing_mark_exits_once_and_does_not_extend_reobserve_deadline(poll_ms):
    flow = marked_flow()
    task = flow.grasp_task
    task.scene_floor_ns = 0
    task.recovery_reobserve_started_ns = 0
    flow.state = MatchState.CHECK_ISOLATED_GREEN
    deadline = None
    for ms in range(0, 2001, poll_ms):
        now = ms*1_000_000
        flow._record_pose_history(now, 0.0, 0.0)
        decision = flow._reacquire_breakup_target(now)
        if flow.grasp_task is None:
            break
        if deadline is None:
            deadline = task.recovery_reobserve_deadline_ns
        assert task.recovery_reobserve_deadline_ns == deadline
    assert flow.grasp_task is None
    assert decision.state is MatchState.SEARCH_CLUSTER
    assert 'breakup_marked_target_unobserved' in decision.reason
    assert now <= 1_010_000_000


def test_full_fixed_distance_is_rejected_instead_of_shortened_at_boundary():
    from test_breakup_planner import plans, target
    reasons = []
    result = plans((target(1, 450, 0), target(2, 480, 0, TargetClass.BLUE_DANGER)),
                   origin=FieldPoint(700, 0), approach=False, rejections=reasons)
    assert not result
    assert any('fixed_push_path_blocked' in reason for reason in reasons)


def test_marks_include_all_scoring_members_before_turn_and_exclude_danger():
    from rescue_vision.app.gripper_width_sequence import GraspPreparation, GraspSelection
    from test_breakup_planner import plans, target as contact
    from test_near_field_grasp import target
    classes = (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE,
               TargetClass.ORANGE_INJURED, TargetClass.BLUE_DANGER)
    items = tuple(contact(i+1, 350+i*20, 0, cls) for i, cls in enumerate(classes))
    plan = plans(items, approach=False)[0]
    preparation = GraspPreparation(0, GraspSelection(None, ('blocked_target:4:blue_danger',)),
        tuple(target(i+1, x=item.center.x, cls=item.target_class)
              for i, item in enumerate(items)), ready=True, recovery_plan=plan)
    flow = _sequence()
    flow._record_pose_history(0, 0, 0)
    assert flow._mark_breakup_targets(preparation)
    assert {mark.target_class for mark in flow.grasp_task.marked_targets} == set(classes[:3])
    assert flow.grasp_task.entry_class is TargetClass.GREEN_SUPPLY
    for mark in flow.grasp_task.marked_targets:
        assert mark.capture_timestamp_ns == 0


def test_no_scoring_geometry_cannot_start_recovery():
    from rescue_vision.app.gripper_width_sequence import GraspPreparation, GraspSelection
    flow = marked_flow()
    preparation = GraspPreparation(0, GraspSelection(None, ()), (),
                                   ready=True, recovery_plan=flow._breakup_plan)
    assert not flow._mark_breakup_targets(preparation)


def test_invalid_telemetry_cannot_finish_a_segment_from_old_zero_speed():
    from test_gripper_width_sequence import motion_sample
    flow = marked_flow()
    flow.observe_grasp_motion(motion_sample(0))
    decision = flow.step(1_000_000_000, perception=None, heading_rad=0,
                         cumulative_distance_m=.5,
                         left_speed_feedback_m_s=0, right_speed_feedback_m_s=0)
    assert decision.state is MatchState.BREAKUP_FORWARD
    assert decision.soft_brake
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert 'breakup_segment_stop:' in decision.reason
