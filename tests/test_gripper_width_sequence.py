from __future__ import annotations

from dataclasses import replace

import pytest

from rescue_vision.app.gripper_width_sequence import (
    GraspPreparation, GraspPreparationSession, GripperWidthPickupSequence, GripperWidthPickupState as State,
)
from rescue_vision.app.near_field_grasp import (
    GraspSelection,
    GraspTargetTracker,
    NearFieldGraspPolicy,
    NearFieldHandoffPrior,
    NearFieldGraspSelector,
)
from rescue_vision.geometry.types import GroundPoint
from rescue_vision.motion import GripperKinematics
from rescue_vision.perception import PerceptionSnapshot
from rescue_vision.perception.types import TargetClass
from rescue_vision.tracking import TrackingConfig
from rescue_vision.config import load_runtime_config
from test_near_field_grasp import target, selector, projector, BLUE, BLACK


def sequence(**overrides):
    values = dict(
        gripper_full_travel_time_s=.1,
        forward_speed_m_s=.1,
        closed_servo_angles_deg=(90,90),
        max_observation_age_ms=500,
        alignment_min_wheel_velocity_m_s=.01,
    )
    values.update(overrides)
    return GripperWidthPickupSequence(**values)


def prep(plan, timestamp=0, ready=True, **kw):
    return GraspPreparation(
        timestamp,
        GraspSelection(plan,()),
        plan.members if plan else (),
        ready,
        checked_member_ids=plan.member_ids if plan else None,
        confirmation_count=3 if ready else 0,
        confirmation_required=3,
        prepared_timestamp_ns=timestamp,
        result_timestamp_ns=timestamp,
        **kw,
    )


def start(seq, plan):
    first=replace(prep(plan,ready=False),checked_member_ids=None)
    assert seq.step(0,first,cumulative_distance_m=0).state is State.VERIFYING
    return seq.step(1,prep(plan),cumulative_distance_m=0)


def test_pickup_completes_without_turn_or_automatic_rearm():
    seq=sequence(); plan=selector().select((target(1,y=-30),target(2,y=30,cls=BLACK))).plan
    opening=start(seq,plan)
    assert opening.state is State.OPENING and opening.gripper_angles_deg==plan.opening_servo_angles_deg
    forward=seq.step(100_000_001,prep(plan,100_000_001),cumulative_distance_m=0)
    assert forward.state is State.FORWARD and forward.linear_velocity_m_s>0
    closing=seq.step(200_000_001,prep(plan,200_000_001),cumulative_distance_m=plan.forward_distance_mm/1000)
    assert closing.state is State.CLOSING and closing.gripper_angles_deg==(90,90)
    complete=seq.step(300_000_001,prep(plan,300_000_001),cumulative_distance_m=plan.forward_distance_mm/1000)
    assert complete.state is State.COMPLETE and complete.angular_velocity_rad_s==0
    assert seq.result.member_ids==(1,2) and not seq.result.capture_confirmed
    assert seq.result.completed_timestamp_ns==300_000_001
    assert seq.result.final_servo_angles_deg==(90,90)
    again=seq.step(10_000_000_000,None,cumulative_distance_m=None)
    assert again.state is State.COMPLETE and again.gripper_angles_deg is None and again.soft_brake


def test_opening_uses_submitted_plan_without_later_visual_recheck():
    seq=sequence(); plan=selector().select((target(),)).plan
    opening=start(seq,plan)
    assert opening.state is State.OPENING
    waiting=seq.step(100_000_001,None,cumulative_distance_m=0)
    assert waiting.state is State.FORWARD
    assert waiting.reason=='forward_open_loop'
    assert waiting.linear_velocity_m_s>0


@pytest.mark.parametrize('y,sign',[(100,1),(-100,-1),(25,1)])
def test_aligns_group_and_locks_identity(y,sign):
    seq=sequence(); plan=selector().select((target(y=y),)).plan
    decision=seq.step(0,replace(prep(plan,ready=False),checked_member_ids=None),cumulative_distance_m=0)
    assert decision.state is State.ALIGNING and decision.angular_velocity_rad_s*sign>0
    assert decision.min_wheel_velocity_m_s == pytest.approx(.01)
    assert seq.locked_ids==(1,)
    # 单帧丢失，仍在已检查旋转时域内，低速继续；不是立即停车。
    missing=GraspPreparation(100,GraspSelection(None,('locked_members_missing',)),(),checked_member_ids=(1,))
    coast=seq.step(100,missing,cumulative_distance_m=0)
    assert coast.state is State.ALIGNING and coast.angular_velocity_rad_s*sign>0
    expired=seq.step(500_000_001,None,cumulative_distance_m=0)
    assert expired.state is State.ALIGNING and expired.soft_brake
    assert expired.reason == 'waiting_locked_target_observation'


def test_alignment_direction_does_not_reverse_after_target_crosses_zero() -> None:
    seq = sequence()
    positive = selector().select((target(y=100),)).plan
    negative = selector().select((target(y=-100, timestamp=100, frame=1),)).plan
    assert positive is not None and negative is not None

    first = seq.step(
        0,
        replace(prep(positive, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    assert first.state is State.ALIGNING
    assert first.angular_velocity_rad_s > 0

    crossed = seq.step(100, prep(negative, 100, ready=False), cumulative_distance_m=0)
    assert crossed.state is State.VERIFYING
    assert crossed.reason == "alignment_crossed_zero_verify"
    assert crossed.angular_velocity_rad_s == 0
    assert crossed.soft_brake


def test_alignment_verify_loss_returns_to_aligning_without_resetting_timeout() -> None:
    seq = sequence()
    positive = selector().select((target(y=100),)).plan
    zero = selector().select((target(timestamp=100, frame=1),)).plan
    positive_later = selector().select(
        (target(y=100, timestamp=200, frame=2),)
    ).plan
    zero_later = selector().select(
        (target(timestamp=300, frame=3),)
    ).plan
    assert (
        positive is not None
        and zero is not None
        and positive_later is not None
        and zero_later is not None
    )

    first = seq.step(
        0,
        replace(prep(positive, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    assert first.state is State.ALIGNING

    verifying = seq.step(100, prep(zero, 100, ready=False), cumulative_distance_m=0)
    assert verifying.state is State.VERIFYING

    realigning = seq.step(
        200,
        prep(positive_later, 200, ready=False),
        cumulative_distance_m=0,
    )
    assert realigning.state is State.ALIGNING
    assert realigning.reason == "alignment_verify_lost_realign"
    assert realigning.angular_velocity_rad_s > 0
    assert seq._alignment_started_ns == 0

    opening = seq.step(
        300,
        prep(zero_later, 300, ready=True),
        cumulative_distance_m=0,
    )
    assert opening.state is State.OPENING


def test_confirmation_window_is_the_only_preopening_frame_gate() -> None:
    seq = sequence(grasp_commit_max_observation_age_ms=150.0)
    plan = selector().select((target(timestamp=0, frame=0),)).plan
    assert plan is not None

    prepared = replace(
        prep(plan, ready=True),
        checked_member_ids=None,
        confirmation_count=3,
        confirmation_required=3,
    )
    opening = seq.step(0, prepared, cumulative_distance_m=0)

    assert opening.state is State.OPENING
    assert opening.reason == "open_group_width"


def test_confirmation_rejects_old_zero_angle_plan_before_opening() -> None:
    seq = sequence(grasp_commit_max_observation_age_ms=150.0)
    plan = selector().select((target(timestamp=0, frame=0),)).plan
    assert plan is not None

    first = seq.step(
        0,
        replace(prep(plan, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    assert first.state is State.VERIFYING

    stale = seq.step(
        200_000_000,
        prep(plan, 0, ready=False),
        cumulative_distance_m=0,
    )
    assert stale.state is State.VERIFYING
    assert stale.reason == "confirmation_waiting_for_fresh_plan"


def test_production_confirmation_does_not_require_a_second_frame_window() -> None:
    """回归：准备器已完成生产确认时，控制周期不得再等三帧。"""

    runtime = load_runtime_config("configs/runtime.match.yaml")
    near = runtime.near_field_grasp
    gripper = runtime.motion.gripper.build_calibration()
    assert gripper is not None
    seq = GripperWidthPickupSequence(
        gripper_full_travel_time_s=gripper.full_travel_time_s,
        forward_speed_m_s=runtime.match.green_approach_speed_m_s,
        closed_servo_angles_deg=(
            gripper.closed_left_angle_deg,
            gripper.closed_right_angle_deg,
        ),
        max_observation_age_ms=runtime.processing.max_observation_age_ms,
        alignment_kp_rad_s=runtime.match.green_alignment_kp_rad_s,
        alignment_max_angular_velocity_rad_s=(
            runtime.match.green_alignment_max_angular_velocity_rad_s
        ),
        alignment_min_wheel_velocity_m_s=(
            runtime.match.green_alignment_min_wheel_velocity_m_s
        ),
        alignment_timeout_ms=near.alignment_timeout_ms,
        grasp_commit_max_observation_age_ms=(
            near.grasp_commit_max_observation_age_ms
        ),
        fine_alignment_zone_rad=near.fine_alignment_zone_rad,
        fine_alignment_min_wheel_velocity_m_s=(
            near.fine_alignment_min_wheel_velocity_m_s
        ),
    )
    production_selector = NearFieldGraspSelector(
        near,
        projector(),
        GripperKinematics(),
        open_servo_angles_deg=(
            gripper.open_left_angle_deg,
            gripper.open_right_angle_deg,
        ),
        closed_servo_angles_deg=(
            gripper.closed_left_angle_deg,
            gripper.closed_right_angle_deg,
        ),
    )
    plan = production_selector.select((target(timestamp=0, frame=0),)).plan
    assert plan is not None
    preparation = replace(
        prep(plan, timestamp=0, ready=True),
        checked_member_ids=None,
        confirmation_count=near.confirmation_frames,
        confirmation_required=near.confirmation_frames,
    )

    opening = seq.step(0, preparation, cumulative_distance_m=0.0)

    assert opening.state is State.OPENING
    assert opening.gripper_angles_deg == plan.opening_servo_angles_deg

    forward = seq.step(
        1_000_000_001,
        None,
        cumulative_distance_m=0.0,
    )
    assert forward.state is State.FORWARD
    assert forward.linear_velocity_m_s > 0.0
    closing = seq.step(
        1_100_000_001,
        None,
        cumulative_distance_m=plan.forward_distance_mm / 1000.0,
    )
    assert closing.state is State.CLOSING
    complete = seq.step(
        2_100_000_001,
        None,
        cumulative_distance_m=plan.forward_distance_mm / 1000.0,
    )
    assert complete.state is State.COMPLETE
    assert seq.result is not None
    assert seq.result.member_ids == plan.member_ids


def test_fine_alignment_zone_allows_zero_minimum_wheel_velocity() -> None:
    seq = sequence()
    plan = selector().select((target(x=400.0, y=25.0),)).plan
    assert plan is not None

    decision = seq.step(
        0,
        replace(prep(plan, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    assert decision.state is State.ALIGNING
    assert decision.min_wheel_velocity_m_s == 0.0


def test_unmatched_worker_result_cannot_open_new_group():
    seq=sequence(); plan=selector().select((target(),)).plan
    seq.step(0,replace(prep(plan,ready=False),checked_member_ids=None),cumulative_distance_m=0)
    other=selector().select((target(2),)).plan
    decision=seq.step(1,prep(other),cumulative_distance_m=0)
    assert decision.gripper_angles_deg is None and seq.locked_ids==(1,)


def test_explicit_blue_before_opening_invalidates_confirmation():
    seq=sequence(); plan=selector().select((target(),)).plan
    seq.step(0,replace(prep(plan,ready=False),checked_member_ids=None),cumulative_distance_m=0)
    blocked=GraspPreparation(1,GraspSelection(None,('blocked_target:2:blue_danger',)),(),checked_member_ids=(1,))
    decision=seq.step(1,blocked,cumulative_distance_m=0)
    assert decision.state is State.SEARCH and decision.gripper_angles_deg is None
    assert decision.reason == 'candidate_replan:candidate_invalid:blocked_target:2:blue_danger'


def test_preopening_replan_has_bounded_wait() -> None:
    seq=sequence(); plan=selector().select((target(),)).plan
    seq.step(0,replace(prep(plan,ready=False),checked_member_ids=None),cumulative_distance_m=0)
    blocked=GraspPreparation(1,GraspSelection(None,('blocked_target:2:blue_danger',)),(),checked_member_ids=(1,))
    seq.step(1,blocked,cumulative_distance_m=0)
    exhausted=seq.step(1_000_000_001,None,cumulative_distance_m=0)
    assert exhausted.state is State.SEARCH
    assert exhausted.reason == 'waiting_eligible_group'


def test_replan_does_not_accept_late_result_from_old_locked_group() -> None:
    seq=sequence(); plan=selector().select((target(),)).plan
    seq.step(0,replace(prep(plan,ready=False),checked_member_ids=None),cumulative_distance_m=0)
    blocked=GraspPreparation(1,GraspSelection(None,('blocked_target:2:blue_danger',)),(),checked_member_ids=(1,))
    seq.step(1,blocked,cumulative_distance_m=0)
    late=replace(prep(plan,2),checked_member_ids=(1,))
    decision=seq.step(2,late,cumulative_distance_m=0)
    assert decision.state is State.SEARCH
    assert seq.active_plan is None
    assert decision.reason == 'waiting_eligible_group'


def test_stale_or_degraded_observation_does_not_gate_open_loop_forward():
    seq=sequence(); plan=selector().select((target(),)).plan
    start(seq,plan)
    missing=prep(plan,100_000_001)
    coast=seq.step(100_000_001,missing,cumulative_distance_m=.01)
    assert coast.state is State.FORWARD and coast.reason=='forward_open_loop'
    assert coast.linear_velocity_m_s==pytest.approx(.1)
    stale=seq.step(2_000_000_001,None,cumulative_distance_m=.02)
    assert stale.state is State.FORWARD and stale.reason=='forward_open_loop'


def test_new_obstacle_after_plan_does_not_exit_open_loop_action():
    seq=sequence(); plan=selector().select((target(),)).plan
    start(seq,plan)
    decision=seq.step(100_000_001,prep(plan,100_000_001),cumulative_distance_m=0)
    assert decision.state is State.FORWARD and decision.reason=='forward_open_loop'
    assert not decision.soft_brake


def test_critical_missing_odometry_exits_active_action():
    seq=sequence(); plan=selector().select((target(),)).plan
    start(seq,plan)
    decision=seq.step(100_000_001,prep(plan,100_000_001),cumulative_distance_m=None)
    assert decision.state is State.ABORTED and decision.reason=='critical_odometry_unavailable'


def session():
    s=selector()
    tracker=GraspTargetTracker(TrackingConfig(1,80,.1,500,1,.1).build_tracker(),projector(),s.config)
    return GraspPreparationSession(tracker,s)


def snapshot(frame,targets):
    ts=frame*10_000_000
    observations=tuple(replace(t.observation,frame_sequence=frame,capture_timestamp_ns=ts,result_timestamp_ns=ts) for t in targets)
    return PerceptionSnapshot(frame,ts,ts,observations,None)


def snapshot_at(frame, timestamp, targets):
    observations=tuple(
        replace(
            t.observation,
            frame_sequence=frame,
            capture_timestamp_ns=timestamp,
            result_timestamp_ns=timestamp,
        )
        for t in targets
    )
    return PerceptionSnapshot(frame,timestamp,timestamp,observations,None)


def test_confirmation_does_not_count_the_same_frame_twice():
    worker = session()
    first = worker.update(snapshot(0, (target(),)), locked_ids=None)
    ids = first.selection.plan.member_ids
    first_confirmed = worker.update(
        snapshot_at(1, 10_000_000, (target(),)),
        locked_ids=ids,
    )
    duplicate = worker.update(
        snapshot_at(1, 11_000_000, (target(),)),
        locked_ids=ids,
    )

    assert first_confirmed.confirmation_progress == (1, 3)
    assert duplicate.confirmation_progress == (1, 3)


def test_plan_and_preparation_ages_use_their_real_timestamps():
    plan = selector().select((target(timestamp=100, frame=10),)).plan
    assert plan is not None
    prepared = GraspPreparation(
        100,
        GraspSelection(plan, ()),
        plan.members,
        ready=True,
        confirmation_count=3,
        confirmation_required=3,
        prepared_timestamp_ns=130,
        result_timestamp_ns=120,
    )

    assert prepared.plan_age_ns(150) == 50
    assert prepared.preparation_age_ns(150) == 20


def test_confirmation_window_requires_distinct_valid_frames():
    worker=session()
    first=worker.update(snapshot(0,(target(),)),locked_ids=None)
    ids=first.selection.plan.member_ids
    result = worker.update(snapshot(1,()),locked_ids=ids)
    assert result.confirmation_count == 0
    for i in (2, 3, 4):
        result=worker.update(snapshot(i,(target(),)),locked_ids=ids)
    assert result.ready
    assert result.confirmation_progress == (3, 3)
    assert result.selection.plan.member_ids==ids


def test_preparation_session_uses_handoff_prior_for_first_tentative_frame():
    s = selector()
    worker = GraspPreparationSession(
        GraspTargetTracker(
            TrackingConfig(2, 80, .1, 500, 1, .1).build_tracker(),
            projector(),
            s.config,
        ),
        s,
    )
    result = worker.update(
        snapshot(0, (target(x=430, y=-25),)),
        locked_ids=None,
        handoff_prior=NearFieldHandoffPrior(
            target_class=target().observation.target_class,
            ground_point=target(x=425, y=-29).observation.ground_point,
            source_track_id=74,
        ),
    )
    assert result.targets[0].handoff_matched
    assert not result.targets[0].confirmed
    assert result.selection.plan is not None


def test_first_green_preparation_waits_for_handoff_instead_of_selecting_neighbor():
    s = selector()
    worker = GraspPreparationSession(
        GraspTargetTracker(
            TrackingConfig(2, 80, .1, 500, 1, .1).build_tracker(),
            projector(),
            s.config,
        ),
        s,
    )
    result = worker.update(
        snapshot(0, (target(2, x=180, y=-120),)),
        locked_ids=None,
        policy=NearFieldGraspPolicy(frozenset((TargetClass.GREEN_SUPPLY,)), 1),
        handoff_prior=NearFieldHandoffPrior(
            target_class=TargetClass.GREEN_SUPPLY,
            ground_point=GroundPoint(430, 0),
            source_track_id=74,
        ),
    )

    assert result.selection.plan is None


def test_confirmation_latch_survives_a_short_target_leak():
    worker=session()
    first=worker.update(snapshot(0,(target(),)),locked_ids=None)
    ids = first.selection.plan.member_ids
    for i in (1, 2, 3):
        result = worker.update(snapshot(i, (target(),)), locked_ids=ids)
    assert result.ready
    leaked = worker.update(snapshot(4, ()), locked_ids=ids)
    assert leaked.ready
    assert leaked.selection.plan is not None
    assert leaked.confirmation_progress == (3, 3)


def test_general_handoff_waits_for_local_companion_then_confirms_both():
    planner = selector()
    worker = GraspPreparationSession(
        GraspTargetTracker(
            TrackingConfig(2, 80, .1, 500, 1, .1).build_tracker(),
            projector(), planner.config,
        ), planner,
    )
    targets = (target(x=334, y=5), target(2, x=332, y=-39, cls=BLACK))
    prior = NearFieldHandoffPrior(
        target_class=TargetClass.GREEN_SUPPLY,
        ground_point=GroundPoint(334, 5), source_track_id=6,
    )
    first = worker.update(snapshot(0, targets), locked_ids=None, handoff_prior=prior)
    assert first.selection.plan is None
    second = worker.update(snapshot(1, targets), locked_ids=None, handoff_prior=prior)
    assert second.selection.plan is not None
    ids = second.selection.plan.member_ids
    assert len(ids) == 2
    for frame in (2, 3, 4):
        confirmed = worker.update(snapshot(frame, targets), locked_ids=ids)
    assert confirmed.ready
    assert confirmed.selection.plan.member_ids == ids


def test_rejected_orange_alternative_does_not_starve_locked_green_confirmation():
    worker = session()
    seq = sequence()
    targets = (
        target(),
        target(2, x=400, y=180, cls=TargetClass.ORANGE_INJURED),
        target(3, x=400, y=220, cls=BLUE),
    )
    first = worker.update(snapshot(0, targets), locked_ids=None)
    assert first.selection.plan is not None
    assert any(r.startswith('orange_not_isolated_track:') for r in first.selection.rejections)
    ids = first.selection.plan.member_ids
    assert seq.step(0, first, cumulative_distance_m=0).state is State.VERIFYING
    for frame in (1, 2, 3):
        result = worker.update(snapshot(frame, targets), locked_ids=ids)
        assert result.confirmation_progress == (frame, 3)
        assert result.selection.rejections == ()
        decision = seq.step(frame * 10_000_000, result, cumulative_distance_m=0)
    assert decision.state is State.OPENING


@pytest.mark.parametrize('reason', [
    'orange_not_isolated_track:2:distance_mm=85.3',
    'outside_near_field',
    'maximum_opening_exceeded',
    'blocked_target:3:blue_danger',
])
def test_valid_selection_rejections_are_alternative_diagnostics(reason):
    plan = selector().select((target(),)).plan
    assert plan is not None
    assert not GraspPreparationSession._is_explicit_invalidation(
        GraspSelection(plan, (reason,)), plan.members, plan.member_ids,
    )
    assert GraspPreparationSession._is_explicit_invalidation(
        GraspSelection(None, (reason,)), plan.members, plan.member_ids,
    )


def test_locked_target_identity_is_stable_across_class_order_changes():
    worker = session()
    first = worker.update(
        snapshot(0, (target(1, x=300, y=-30), target(2, x=320, y=30))),
        locked_ids=None,
    )
    ids = first.selection.plan.member_ids
    locked = worker.update(
        snapshot(1, (target(1, x=300, y=-30), target(2, x=320, y=30))),
        locked_ids=ids,
    )
    assert locked.checked_member_ids == ids
    later = worker.update(
        snapshot(2, (target(9, x=320, y=30, cls=BLACK), target(8, x=300, y=-30))),
        locked_ids=ids,
    )
    assert later.checked_member_ids == ids
    assert later.selection.plan is None
    assert not later.ready


def test_confirmation_uses_latest_geometry_for_the_locked_identity():
    worker = session()
    first = worker.update(
        snapshot(0, (target(1, x=300, y=0),)),
        locked_ids=None,
    )
    first_plan = first.selection.plan
    assert first_plan is not None

    ids = first_plan.member_ids
    worker.update(snapshot(1, (target(1, x=320, y=0),)), locked_ids=ids)
    worker.update(snapshot(2, (target(1, x=340, y=0),)), locked_ids=ids)
    latest = worker.update(snapshot(3, (target(1, x=360, y=0),)), locked_ids=ids)

    assert latest.ready
    assert latest.selection.plan is not None
    assert latest.selection.plan.capture_timestamp_ns == 30_000_000
    assert latest.selection.plan.alignment_point.x == pytest.approx(360.0)
    assert latest.selection.plan.alignment_point != first_plan.alignment_point


def test_alignment_hysteresis_keeps_small_post_lock_jitter_stable():
    s = selector(center_tolerance_mm=5.0, alignment_hysteresis_mm=10.0)
    tracker = GraspTargetTracker(
        TrackingConfig(1, 80, .1, 500, 1, .1).build_tracker(),
        projector(),
        s.config,
    )
    worker = GraspPreparationSession(tracker, s)
    first = worker.update(
        snapshot(0, (target(),)),
        locked_ids=None,
    )
    ids = first.selection.plan.member_ids

    ready_result = None
    for frame in range(1, 5):
        y = 0.0 if frame <= 3 else 8.0
        result = worker.update(
            snapshot(frame, (target(y=y),)),
            locked_ids=ids,
        )

        if result.ready:
            ready_result = result

    assert ready_result is not None
    assert ready_result.selection.plan is not None
    assert ready_result.selection.plan.alignment_angle_rad == 0.0


def test_locked_group_cannot_collect_different_nearest_objects():
    worker=session()
    first=worker.update(snapshot(0,(target(),)),locked_ids=None)
    for i in range(1,6):
        result=worker.update(snapshot(i,(target(x=300,y=200),)),locked_ids=first.selection.plan.member_ids)
    assert not result.ready and result.selection.plan is None


def test_confirmation_window_does_not_ignore_later_corridor_obstacle_evidence():
    worker=session()
    first=worker.update(snapshot(0,(target(),)),locked_ids=None)
    plan=first.selection.plan
    worker.update(snapshot(1,(target(),)),locked_ids=plan.member_ids)
    worker.update(snapshot(2,(target(),)),locked_ids=plan.member_ids)
    blocked=worker.update(snapshot(3,(target(),target(2,x=200,y=25,cls=BLUE))),locked_ids=plan.member_ids)
    assert not blocked.ready
    assert blocked.selection.plan is None


def test_closing_occlusion_does_not_rearm_an_already_submitted_action():
    seq=sequence(); plan=selector().select((target(),)).plan
    start(seq,plan)
    closed=seq.step(100_000_001,prep(plan,100_000_001),cumulative_distance_m=plan.forward_distance_mm/1000)
    assert closed.state is State.CLOSING
    complete=seq.step(700_000_000,None,cumulative_distance_m=None)
    assert complete.state is State.COMPLETE and seq.result.capture_confirmed is False


def test_alignment_timeout_restarts_search_without_opening_gripper():
    seq = GripperWidthPickupSequence(
        gripper_full_travel_time_s=.1,
        forward_speed_m_s=.1,
        closed_servo_angles_deg=(90, 90),
        max_observation_age_ms=500,
        alignment_min_wheel_velocity_m_s=.01,
        alignment_timeout_ms=.1,
    )
    plan = selector().select((target(y=100),)).plan
    assert plan is not None

    first = seq.step(
        0,
        replace(prep(plan, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    assert first.state is State.ALIGNING

    timed_out = seq.step(100_001, None, cumulative_distance_m=0)
    assert timed_out.state is State.SEARCH
    assert timed_out.reason == "alignment_timeout"
    assert timed_out.gripper_angles_deg is None
    assert timed_out.soft_brake


def test_verification_timeout_restarts_search_when_stability_never_completes():
    seq = GripperWidthPickupSequence(
        gripper_full_travel_time_s=.1,
        forward_speed_m_s=.1,
        closed_servo_angles_deg=(90, 90),
        max_observation_age_ms=500,
        alignment_min_wheel_velocity_m_s=.01,
        alignment_timeout_ms=.1,
    )
    plan = selector().select((target(),)).plan
    assert plan is not None

    first = seq.step(
        0,
        replace(prep(plan, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    timed_out = seq.step(100_001, None, cumulative_distance_m=0)

    assert first.state is State.VERIFYING
    assert timed_out.state is State.SEARCH
    assert timed_out.reason == "alignment_timeout"


def test_candidate_replan_starts_a_fresh_candidate_timeout():
    seq = GripperWidthPickupSequence(
        gripper_full_travel_time_s=.1,
        forward_speed_m_s=.1,
        closed_servo_angles_deg=(90, 90),
        max_observation_age_ms=500,
        alignment_min_wheel_velocity_m_s=.01,
        alignment_timeout_ms=.1,
    )
    first_plan = selector().select((target(y=100),)).plan
    assert first_plan is not None
    seq.step(
        0,
        replace(prep(first_plan, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    invalid = GraspPreparation(
        40_000,
        GraspSelection(None, ('blocked_target:2:unknown',)),
        (),
        checked_member_ids=first_plan.member_ids,
    )
    replanning = seq.step(40_000, invalid, cumulative_distance_m=0)
    assert replanning.reason.startswith('candidate_replan:')

    replacement = selector().select((target(2, y=-100),)).plan
    assert replacement is not None
    seq.step(
        50_000,
        replace(prep(replacement, 50_000, ready=False), checked_member_ids=None),
        cumulative_distance_m=0,
    )
    still_aligning = seq.step(100_001, None, cumulative_distance_m=0)

    assert still_aligning.state is State.ALIGNING

    timed_out = seq.step(200_001, None, cumulative_distance_m=0)

    assert timed_out.state is State.SEARCH
    assert timed_out.reason == 'alignment_timeout'
    assert timed_out.soft_brake


def motion_sample(timestamp_ns, *, count=0, gyro=0, flags=None):
    from rescue_vision.motion.protocol import SensorFlags
    from test_cluster_breakup import odometry
    if flags is None:
        flags = SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID | SensorFlags.IMU_VALID
    return replace(odometry(timestamp_ns, count, sensor_flags=flags), gyro_z_urad_s=gyro)


@pytest.mark.parametrize('delay_ms', [300, 350, 450])
def test_delayed_stationary_capture_opens_and_completes(delay_ms):
    """1855日志：确认3/3，几何300～450ms，发布年龄小于150ms。"""
    runtime = load_runtime_config('configs/runtime.match.yaml')
    seq = sequence(max_observation_age_ms=runtime.processing.max_observation_age_ms)
    capture = 100_000_000
    now = capture + delay_ms * 1_000_000
    plan = selector().select((target(timestamp=capture, frame=1),)).plan
    for stamp in range(0, now + 1, 10_000_000):
        seq.observe_motion(motion_sample(stamp))
    prepared = replace(prep(plan, capture), checked_member_ids=None,
                       prepared_timestamp_ns=now - 20_000_000)
    decision = seq.step(now, prepared, cumulative_distance_m=0)
    assert decision.state is State.OPENING
    assert decision.gripper_angles_deg == plan.opening_servo_angles_deg
    assert seq.step(now + seq.travel_ns, None, cumulative_distance_m=0).state is State.FORWARD
    assert seq.step(now + seq.travel_ns + 1, None,
                    cumulative_distance_m=plan.forward_distance_mm / 1000).state is State.CLOSING
    assert seq.step(now + 2 * seq.travel_ns + 1, None,
                    cumulative_distance_m=plan.forward_distance_mm / 1000).state is State.COMPLETE


@pytest.mark.parametrize('failure', ['wheel', 'rotation', 'gap', 'duplicate', 'invalid', 'old_geometry', 'expired', 'before_stop'])
def test_delayed_plan_requires_continuous_real_motion_evidence(failure):
    from rescue_vision.motion.protocol import SensorFlags
    seq = sequence(max_observation_age_ms=800)
    capture, now = 100_000_000, 450_000_000
    for stamp in range(0, now + 1, 10_000_000):
        sample = motion_sample(stamp)
        if failure == 'wheel' and stamp >= 200_000_000:
            sample = motion_sample(stamp, count=1)
        if failure == 'rotation' and stamp == 300_000_000:
            sample = motion_sample(stamp, gyro=100_000)
        if failure == 'gap' and 150_000_000 <= stamp < 400_000_000:
            continue
        if failure == 'duplicate' and stamp >= 300_000_000:
            sample = motion_sample(290_000_000)
        if failure == 'invalid':
            sample = motion_sample(stamp, flags=SensorFlags.LEFT_ENCODER_VALID | SensorFlags.RIGHT_ENCODER_VALID)
        seq.observe_motion(sample)
    if failure == 'before_stop':
        capture = 0
    plan = selector().select((target(timestamp=capture, frame=1),)).plan
    prep_capture = now if failure == 'old_geometry' else capture
    if failure == 'expired':
        now = 2_000_000_000
    prepared = replace(prep(plan, prep_capture), checked_member_ids=None,
                       prepared_timestamp_ns=now - 10_000_000)
    decision = seq.step(now, prepared, cumulative_distance_m=0)
    assert decision.gripper_angles_deg is None
    assert decision.state is not State.OPENING


def test_slow_diagnostic_callback_does_not_block_next_preparation():
    from threading import Event
    from rescue_vision.app.gripper_width_sequence import GraspPreparationWorker
    entered, release, processed = Event(), Event(), Event()
    planner = session()
    original_update = planner.update

    def update(*args, **kwargs):
        result = original_update(*args, **kwargs)
        if result.capture_timestamp_ns == 10_000_000:
            processed.set()
        return result

    planner.update = update

    def diagnostic(_items):
        entered.set()
        assert release.wait(3)

    with GraspPreparationWorker(planner, planner.selector, diagnostics_callback=diagnostic) as worker:
        worker.begin(1)
        try:
            worker.submit(snapshot(0, (target(),)), session_id=1,
                          policy=planner.selector.default_policy, locked_ids=None)
            assert entered.wait(3)
            worker.submit(snapshot(1, (target(),)), session_id=1,
                          policy=planner.selector.default_policy, locked_ids=None)
            assert processed.wait(3), 'diagnostic blocked the control preparation thread'
        finally:
            release.set()


def test_even_fresh_plan_cannot_commit_after_observed_motion():
    seq = sequence()
    seq.observe_motion(motion_sample(0))
    seq.observe_motion(motion_sample(10_000_000))
    seq.observe_motion(motion_sample(30_000_000, count=1))
    plan = selector().select((target(timestamp=20_000_000, frame=1),)).plan
    decision = seq.step(40_000_000, replace(prep(plan, 20_000_000), checked_member_ids=None),
                        cumulative_distance_m=0)
    assert decision.reason == 'confirmation_waiting_for_stationary_capture'
    assert decision.gripper_angles_deg is None


def test_uart_batch_with_equal_receive_time_preserves_distinct_samples():
    seq = sequence()
    for stamp in range(0, 460_000_000, 10_000_000):
        # 同次UART读取可能给多个有效递增设备样本标注同一主机接收时刻。
        seq.observe_motion(replace(motion_sample(stamp), sample_timestamp_us=stamp//1000))
        seq.observe_motion(replace(motion_sample(stamp), sample_timestamp_us=stamp//1000+1000))
    plan = selector().select((target(timestamp=100_000_000, frame=1),)).plan
    prepared = replace(prep(plan, 100_000_000), checked_member_ids=None, prepared_timestamp_ns=440_000_000)
    assert seq.step(450_000_000, prepared, cumulative_distance_m=0).state is State.OPENING
