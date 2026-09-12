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
    assert forward.reason == "near_field_forward:forward_open_loop"
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
