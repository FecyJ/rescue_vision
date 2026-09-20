from __future__ import annotations

import math
from dataclasses import replace

import pytest

from rescue_vision.app.gripper_width_sequence import (
    GraspPreparationSession,
    GripperWidthPickupSequence,
)
from rescue_vision.app.match import GraspRoute, MatchState
from rescue_vision.app.near_field_grasp import GraspTargetTracker
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.perception import TargetClass
from rescue_vision.tracking import TrackingConfig

from test_gripper_width_sequence import prep
from test_match import observation, snapshot
from test_match_near_field import _sequence
from test_near_field_grasp import projector, selector, target


def test_current_heading_commits_a_physically_reachable_off_center_target() -> None:
    planner = selector(center_tolerance_mm=5.0)
    result = planner.select((target(x=300.0, y=20.0),))

    assert result.plan is not None
    assert result.plan.alignment_angle_rad == 0.0
    assert result.plan.opening_servo_angles_deg[0] != result.plan.opening_servo_angles_deg[1]


def test_unreachable_current_heading_uses_the_minimum_mechanical_turn() -> None:
    planner = selector()
    result = planner.select((target(x=300.0, y=100.0),))

    assert result.plan is not None
    assert 0.0 < result.plan.alignment_angle_rad < math.atan2(100.0, 300.0)


def test_formal_green_alignment_updates_delayed_geometry_from_pose_history() -> None:
    sequence = _sequence()
    sequence._started = True
    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN
    sequence._selected_track_id = 1

    first = sequence.step(
        0,
        perception=snapshot(0, 0, observation(0, 0, GroundPoint(700.0, 100.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert first.reason == "green_coarse_align_imu_closed_loop"

    delayed = sequence.step(
        500_000_000,
        perception=None,
        heading_rad=0.0,
        cumulative_distance_m=0.1,
    )

    assert delayed.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert delayed.angular_velocity_rad_s == pytest.approx(math.atan2(100.0, 600.0))
    assert sequence._selected_green_ground == GroundPoint(600.0, 100.0)


def test_near_field_old_angle_stops_after_one_bounded_turn_without_new_frame() -> None:
    pickup = GripperWidthPickupSequence(
        gripper_full_travel_time_s=0.1,
        forward_speed_m_s=0.1,
        closed_servo_angles_deg=(90.0, 90.0),
        max_observation_age_ms=800.0,
        alignment_continue_max_age_ms=300.0,
    )
    plan = selector().select((target(x=300.0, y=100.0),)).plan
    assert plan is not None

    first = pickup.step(
        0,
        replace(prep(plan, ready=False), checked_member_ids=None),
        cumulative_distance_m=0.0,
        heading_rad=0.0,
    )
    stopped = pickup.step(
        400_000_000,
        None,
        cumulative_distance_m=0.0,
        heading_rad=plan.alignment_angle_rad,
    )

    assert first.angular_velocity_rad_s > 0.0
    assert stopped.angular_velocity_rad_s == 0.0
    assert stopped.soft_brake
    assert stopped.state.value == "aligning"


def test_near_field_allows_only_one_correction_after_the_coarse_turn() -> None:
    pickup = GripperWidthPickupSequence(
        gripper_full_travel_time_s=0.1,
        forward_speed_m_s=0.1,
        closed_servo_angles_deg=(90.0, 90.0),
        max_observation_age_ms=800.0,
        alignment_continue_max_age_ms=300.0,
    )
    first_plan = selector().select(
        (target(x=300.0, y=100.0, frame=0, timestamp=0),)
    ).plan
    correction_plan = selector().select(
        (target(x=300.0, y=100.0, frame=1, timestamp=500_000_000),)
    ).plan
    assert first_plan is not None and correction_plan is not None

    pickup.step(
        0,
        replace(prep(first_plan, 0, ready=False), checked_member_ids=None),
        cumulative_distance_m=0.0,
        heading_rad=0.0,
    )
    pickup.step(400_000_000, None, cumulative_distance_m=0.0, heading_rad=first_plan.alignment_angle_rad)
    correction = pickup.step(
        500_000_000,
        replace(
            prep(correction_plan, 500_000_000, ready=False),
            checked_member_ids=(1,),
        ),
        cumulative_distance_m=0.0,
        heading_rad=first_plan.alignment_angle_rad,
    )
    pickup.step(900_000_000, None, cumulative_distance_m=0.0, heading_rad=first_plan.alignment_angle_rad + correction_plan.alignment_angle_rad)
    third = pickup.step(
        1_000_000_000,
        replace(
            prep(
                selector().select(
                    (target(x=300.0, y=100.0, frame=2, timestamp=1_000_000_000),)
                ).plan,
                1_000_000_000,
                ready=False,
            ),
            checked_member_ids=(1,),
        ),
        cumulative_distance_m=0.0,
        heading_rad=0.0,
    )

    assert correction.reason == "align_group_envelope_correction"
    assert pickup._alignment_attempts == 2
    assert third.reason.startswith("candidate_replan:alignment_attempt_budget")


def test_formal_match_opens_and_advances_for_current_heading_reachable_target() -> None:
    sequence = _sequence()
    sequence._started = True
    sequence.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
    sequence._near_field_session_id = 1
    sequence._near_field_route = GraspRoute.DIRECT_NEAR
    plan = selector().select((target(x=300.0, y=20.0),)).plan
    assert plan is not None and plan.alignment_angle_rad == 0.0

    opening = sequence._step_near_field_grasp(
        0,
        cumulative_distance_m=0.0,
        preparation=replace(
            prep(plan, 0, ready=True, session_id=1),
            checked_member_ids=None,
        ),
        path_clear=True,
    )
    forward = sequence._step_near_field_grasp(
        sequence._near_field_pickup.travel_ns + 1,
        cumulative_distance_m=0.0,
        preparation=None,
        path_clear=True,
    )

    assert opening.reason == "near_field_opening:open_group_width"
    assert forward.reason == "near_field_forward:forward_encoder_heading_hold"
    assert forward.linear_velocity_m_s > 0.0


def test_locked_unknown_preserves_confirmation_progress_but_not_commit_evidence() -> None:
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(
            TrackingConfig(1, 80.0, 0.1, 500.0, 1.0, 0.1).build_tracker(),
            projector(),
            planner.config,
        ),
        planner,
    )
    first = session.update(snapshot(0, 0, target().observation), locked_ids=None)
    ids = first.selection.plan.member_ids
    clean = session.update(snapshot(1, 10, target(frame=1, timestamp=10).observation), locked_ids=ids)
    unknown = session.update(
        snapshot(2, 20, target(cls=TargetClass.GREEN_SUPPLY, frame=2, timestamp=20).observation),
        locked_ids=ids,
    )

    assert clean.confirmation_progress == (1, 1)
    assert unknown.confirmation_progress == (1, 1)
    assert unknown.selection.plan is not None
    assert "locked_member_class_changed" not in unknown.selection.rejections


def test_peripheral_unknown_without_geometry_does_not_clear_locked_core_progress() -> None:
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(
            TrackingConfig(1, 80.0, 0.1, 500.0, 1.0, 0.1).build_tracker(),
            projector(),
            planner.config,
        ),
        planner,
    )
    first = session.update(snapshot(0, 0, target().observation), locked_ids=None)
    ids = first.selection.plan.member_ids
    session.update(snapshot(1, 10, target(frame=1, timestamp=10).observation), locked_ids=ids)
    peripheral = target(i=2, x=700.0, cls=TargetClass.GREEN_SUPPLY, frame=2, timestamp=20)
    peripheral_observation = replace(peripheral.observation, ground_point=None, k0=None)
    result = session.update(
        snapshot(2, 20, target(frame=2, timestamp=20).observation, peripheral_observation),
        locked_ids=ids,
    )

    assert result.confirmation_progress == (1, 1)
    assert result.selection.plan is not None
    assert "locked_member_class_changed" not in result.selection.rejections


def test_locked_member_keeps_canonical_identity_when_tracker_id_changes() -> None:
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(
            TrackingConfig(1, 80.0, 0.1, 500.0, 1.0, 0.1).build_tracker(),
            projector(),
            planner.config,
        ),
        planner,
    )
    first = session.update(snapshot(0, 0, target().observation), locked_ids=None)
    ids = first.selection.plan.member_ids
    session.update(snapshot(1, 10, target(frame=1, timestamp=10).observation), locked_ids=ids)

    session.tracker.tracker.reset()
    changed = session.update(
        snapshot(
            2,
            20,
            observation(
                2,
                20,
                GroundPoint(100.0, 300.0),
                target_class=TargetClass.BLUE_DANGER,
                box_x=1.0,
            ),
            target(i=2, frame=2, timestamp=20).observation,
        ),
        locked_ids=ids,
    )

    assert changed.selection.plan is not None
    assert changed.selection.plan.member_ids == ids
    assert changed.checked_member_ids == ids


def test_failure_record_binds_to_the_plan_member_not_a_tracker_id_lookup() -> None:
    """主 tracker 与近场会话编号不同，失败记录必须绑定计划成员的实际几何。"""

    sequence = _sequence()
    sequence._started = True
    sequence._latest_heading_rad = 0.0
    sequence.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP

    plan = selector().select((target(1, x=400.0, y=0.0),)).plan
    assert plan is not None
    preparation = prep(plan, 0, ready=False)

    # 主 tracker 里同 ID 的目标在完全不同的位置：按 ID 回查会记错区域。
    sequence._selected_track_id = 1
    sequence._tracker.update(
        0, (observation(0, 0, GroundPoint(1500.0, 0.0)),) * 2,
    )
    selected = sequence._selected_target()
    assert selected is not None
    assert sequence._current_ground_point_for_track(selected, 0) == GroundPoint(
        1500.0, 0.0
    )

    sequence._remember_near_field_failure(0, "confirmation_timeout", preparation)
    record = sequence._near_field_failures[-1]

    assert record.target_class is TargetClass.GREEN_SUPPLY
    assert record.region_field_points == (FieldPoint(400.0, 0.0),)
    assert record.field_point == FieldPoint(400.0, 0.0)


def test_failed_physical_region_is_not_reentered_until_geometry_changes() -> None:
    sequence = _sequence()
    sequence._started = True
    sequence.state = MatchState.SEARCH_CLUSTER
    first = sequence.step(
        0,
        perception=snapshot(0, 0, observation(0, 0, GroundPoint(500.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    assert first.state is MatchState.TRANSPORT_ALIGN_GREEN
    sequence._remember_near_field_failure(0, "confirmation_timeout")
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence._selected_track_id = None

    same = sequence.step(
        10,
        perception=snapshot(1, 10, observation(1, 10, GroundPoint(500.0, 0.0))),
        heading_rad=0.0,
        cumulative_distance_m=0.0,
    )
    moved = sequence.step(
        20,
        perception=snapshot(
            2,
            20,
            observation(2, 20, GroundPoint(400.0, 0.0)),
        ),
        heading_rad=0.0,
        cumulative_distance_m=0.1,
    )

    assert same.state is MatchState.SEARCH_CLUSTER
    assert moved.state is MatchState.TRANSPORT_ALIGN_GREEN


@pytest.mark.parametrize('poll_ms,delay_ms', [(5, 300), (10, 600)])
def test_async_preparation_without_new_camera_frame_opens_and_moves(poll_ms, delay_ms):
    from test_gripper_width_sequence import motion_sample
    sequence = _sequence(max_observation_age_ms=800.0)
    sequence._started = True
    sequence.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
    sequence._near_field_session_id = 1
    sequence._near_field_route = GraspRoute.DIRECT_NEAR
    capture_ns = 100_000_000
    plan = selector().select((target(timestamp=capture_ns, frame=1),)).plan
    assert plan is not None
    latest = snapshot(1, capture_ns, plan.members[0].observation)
    ready_ms = 100 + delay_ms
    prepared = replace(prep(plan, capture_ns, session_id=1), checked_member_ids=None,
                       prepared_timestamp_ns=ready_ms*1_000_000)
    opening_ms = None
    moved = False
    for ms in range(0, ready_ms + 2000, poll_ms):
        now = ms*1_000_000
        if ms%10 == 0:
            sequence.observe_grasp_motion(motion_sample(now))
        decision = sequence.step(now, perception=latest if ms >= 100 else None,
                                 heading_rad=0., cumulative_distance_m=0.,
                                 near_field_preparation=prepared if ms >= ready_ms else None,
                                 near_field_path_clear=True)
        if decision.reason.startswith('near_field_opening:') and opening_ms is None:
            opening_ms = ms
        if decision.linear_velocity_m_s > 0:
            moved = True
            break
    assert opening_ms == ready_ms
    assert moved


@pytest.mark.parametrize('delay_ms,period_ms,poll_ms', [(300,250,5), (600,400,10)])
def test_rejected_handoff_selects_another_physical_target_without_session_loop(delay_ms, period_ms, poll_ms):
    from rescue_vision.app.near_field_grasp import NearFieldHandoffPrior
    from test_gripper_width_sequence import motion_sample, snapshot_at

    seq = _sequence(max_observation_age_ms=1000.0)
    seq.config = replace(seq.config, green_max_age_ms=1000.0)
    seq._started = True
    seq._latest_heading_rad = 0.0
    seq._latest_cumulative_distance_m = 0.0
    points = (GroundPoint(265, -176), GroundPoint(427, -2))
    members = tuple(target(i+1, x=p.x, y=p.y, timestamp=100_000_000)
                    for i, p in enumerate(points))
    latest = snapshot_at(1, 100_000_000, members)
    first, other = seq._tracker.update(100_000_000, latest.observations)
    prior = NearFieldHandoffPrior(TargetClass.GREEN_SUPPLY, points[0], first.track_id)
    seq._begin_near_field_grasp(0, handoff_prior=prior)
    initial_session = seq.near_field_session_id
    initial_task = seq.grasp_task
    deadline = initial_task.deadline_ns
    planner = selector(confirmation_frames=1)
    session = GraspPreparationSession(
        GraspTargetTracker(TrackingConfig(1,80,.1,500,1,.1).build_tracker(),
                           projector(), planner.config), planner)
    prepared = routed = None
    for ms in range(0, 100+delay_ms+period_ms+100, poll_ms):
        now = ms * 1_000_000
        seq.observe_grasp_motion(motion_sample(now))
        if ms == 100+delay_ms:
            result = session.update(latest, locked_ids=None, handoff_prior=prior,
                                    session_id=initial_session, policy=seq.near_field_policy)
            prepared = replace(result, prepared_timestamp_ns=now, result_timestamp_ns=now)
            assert prepared.ready
            assert prepared.selection.plan.alignment_angle_rad == 0
            assert prepared.selection.plan.members[0].observation.ground_point == points[1]
        decision = seq.step(now, perception=latest if prepared else None,
                            heading_rad=0., cumulative_distance_m=0.,
                            near_field_preparation=prepared, near_field_path_clear=True)
        if decision.gripper_angles_deg is not None and prepared is not None:
            routed = decision
            break
    assert routed is not None
    assert seq.selected_track_id == other.track_id
    assert seq._target_attempt_blocked(first, now)
    assert not seq._target_attempt_blocked(other, now)
    assert seq._target_attempt_blocked(replace(first, track_id=100), now)
    assert seq.near_field_session_id == initial_session
    assert seq.grasp_task is initial_task and seq.grasp_task.deadline_ns == deadline


@pytest.mark.parametrize('classes', [
    (TargetClass.GREEN_SUPPLY, TargetClass.GREEN_SUPPLY),
    (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE),
    (TargetClass.BLACK_CORE, TargetClass.BLACK_CORE),
    (TargetClass.GREEN_SUPPLY, TargetClass.BLACK_CORE, TargetClass.GREEN_SUPPLY),
])
def test_approach_reacquires_renumbered_near_supplies_and_opens_group(classes):
    from test_match_near_field import _preparation
    seq = _sequence(transports=1)
    seq._started = True
    seq.state = MatchState.TRANSPORT_APPROACH_GREEN
    seq._selected_track_id = 10  # 1258 日志：旧目标消失，新目标为 15/17。
    seq._green_reference = GroundPoint(553, 0)
    seq._green_approach_base_distance_m = 0.
    seq._green_approach_distance_m = .103
    latest = snapshot(1, 0, *(observation(1,0,GroundPoint(260, (i-1)*42),
        target_class=cls, box_x=i*40) for i,cls in enumerate(classes)))
    decision = seq.step(0, perception=latest, heading_rad=0., cumulative_distance_m=.35)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert seq.selected_track_id != 10
    decision = seq.step(5_000_000, perception=latest, heading_rad=0., cumulative_distance_m=.35)
    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    members = tuple(target(i+1,x=260,y=(i-1)*42,cls=cls,frame=2,timestamp=10_000_000)
                    for i,cls in enumerate(classes))
    plan = selector().select(members, policy=seq.near_field_policy).plan
    assert plan is not None and len(plan.members) == len(classes)
    decision = seq._step_near_field_grasp(10_000_000, cumulative_distance_m=.35,
        preparation=_preparation(plan, session_id=seq.near_field_session_id,
                                 timestamp_ns=10_000_000), path_clear=True)
    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert seq._near_field_pickup.state.value == 'opening'
    assert decision.gripper_angles_deg is not None


def test_approach_reference_without_tracker_has_fixed_encoder_endpoint():
    seq = _sequence(transports=1)
    seq._started = True
    seq.state = MatchState.TRANSPORT_APPROACH_GREEN
    seq._green_reference = GroundPoint(550, 0)
    seq._green_approach_base_distance_m = 0.
    seq._green_approach_distance_m = .1
    for ms in range(0, 1200, 5):
        decision = seq.step(ms*1_000_000, perception=None, heading_rad=0.,
                            cumulative_distance_m=ms*.0001)
        if decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP:
            break
    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert ms <= 1005


@pytest.mark.parametrize('other_class,expected', [(TargetClass.GREEN_SUPPLY,True),
    (TargetClass.BLACK_CORE,True), (TargetClass.BLUE_DANGER,False), (TargetClass.ORANGE_INJURED,False)])
def test_approach_path_keeps_collectible_group_members_but_rejects_intruders(other_class,expected):
    seq = _sequence(transports=1)
    seq._latest_heading_rad = 0.
    seq._record_pose_history(0,0.,0.)
    tracks = seq._tracker.update(0, (
        observation(1,0,GroundPoint(600,0)),
        observation(1,0,GroundPoint(560,20),target_class=other_class,box_x=40)))
    seq._selected_track_id = tracks[0].track_id
    assert seq._green_path_is_clear_at_now(tracks[0],GroundPoint(600,0),0) is expected


def _greedy_sequence_for_test():
    sequence = _sequence(transports=1)
    sequence._started = True
    sequence._latest_heading_rad = 0.0
    sequence._transport_target_classes = (TargetClass.GREEN_SUPPLY,)
    sequence._greedy_active = True
    sequence._greedy_started_ns = 0
    sequence._greedy_last_heading = 0.0
    sequence.state = MatchState.TRANSPORT_GREEDY_SCAN
    return sequence


def test_supplementary_scan_accepts_far_field_and_prefers_nearest_supply():
    sequence = _greedy_sequence_for_test()
    nearer = target(
        1,
        x=650.0,
        cls=TargetClass.GREEN_SUPPLY,
        timestamp=100_000_000,
        frame=1,
    ).observation
    farther = target(
        2,
        x=700.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=100_000_000,
        frame=1,
    ).observation
    tracks = sequence._tracker.update(100_000_000, (nearer, farther))

    decision = sequence._step_greedy_scan(100_000_000, heading_rad=0.0)

    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.reason == f"green_path_clear_opportunistic_single:{tracks[0].track_id}"
    assert sequence.selected_track_id == tracks[0].track_id
    assert tracks[0].target_class is TargetClass.GREEN_SUPPLY
    assert math.hypot(nearer.ground_point.x, nearer.ground_point.y) > 450.0


def test_supplementary_selected_target_does_not_change_for_a_new_closer_supply():
    sequence = _greedy_sequence_for_test()
    original = target(
        1,
        x=700.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=100_000_000,
        frame=1,
    ).observation
    sequence._tracker.update(100_000_000, (original,))
    sequence._step_greedy_scan(100_000_000, heading_rad=0.0)
    selected_id = sequence.selected_track_id
    assert selected_id is not None

    closer = target(
        2,
        x=300.0,
        cls=TargetClass.GREEN_SUPPLY,
        timestamp=200_000_000,
        frame=2,
    ).observation
    refreshed = target(
        1,
        x=700.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=200_000_000,
        frame=2,
    ).observation
    sequence._tracker.update(200_000_000, (refreshed, closer))

    sequence._step_formal_green_align(200_000_000)

    assert sequence.selected_track_id == selected_id


def test_supplementary_disappeared_target_can_handoff_to_next_supply():
    sequence = _greedy_sequence_for_test()
    original = target(
        1,
        x=700.0,
        cls=TargetClass.GREEN_SUPPLY,
        timestamp=0,
        frame=0,
    ).observation
    sequence._tracker.update(0, (original,))
    sequence._selected_track_id = sequence._tracker.tracks[0].track_id
    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._green_reference = GroundPoint(700.0, 0.0)
    sequence._action_settle_phase = None
    sequence._action_settle_until_ns = None

    next_observation = target(
        2,
        x=650.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=600_000_000,
        frame=1,
    ).observation
    next_track = sequence._tracker.update(600_000_000, (next_observation,))[0]

    decision = sequence._step_formal_green_approach(
        600_000_000,
        cumulative_distance_m=0.0,
    )

    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.reason == f"near_field_group_preview:{next_track.track_id}"
    assert sequence.selected_track_id == next_track.track_id


def test_supplementary_near_field_loss_restarts_normal_chain_for_next_supply():
    from rescue_vision.app.gripper_width_sequence import GraspPreparation, GraspSelection
    from rescue_vision.app.near_field_grasp import NearFieldHandoffPrior

    sequence = _greedy_sequence_for_test()
    sequence.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_session_id = 1
    sequence._near_field_confirmation_started_ns = 0
    sequence._near_field_handoff_prior = NearFieldHandoffPrior(
        TargetClass.GREEN_SUPPLY,
        GroundPoint(300.0, 0.0),
        source_track_id=1,
    )
    next_observation = target(
        2,
        x=650.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=100_000_000,
        frame=1,
    ).observation
    next_track = sequence._tracker.update(100_000_000, (next_observation,))[0]
    preparation = GraspPreparation(
        100_000_000,
        GraspSelection(None, ("handoff_target_missing",)),
        (),
        session_id=1,
        prepared_timestamp_ns=100_000_000,
        result_timestamp_ns=100_000_000,
    )

    decision = sequence._step_near_field_grasp(
        100_000_000,
        cumulative_distance_m=0.0,
        preparation=preparation,
        path_clear=True,
    )

    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.reason == f"greedy_target_disappeared_next:{next_track.track_id}"
    assert sequence.selected_track_id == next_track.track_id
