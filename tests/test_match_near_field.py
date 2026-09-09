from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from rescue_vision.app.match import GripperPosture, MatchSequence, MatchState
from rescue_vision.app.match import GraspRoute
from rescue_vision.app.match_runtime import (
    _overlay_match_selected_targets,
    _overlay_near_field_corridor,
)
from rescue_vision.app.gripper_width_sequence import (
    GraspPreparation,
    GraspSelection,
    GripperWidthPickupDecision,
    GripperWidthPickupSequence,
    GripperWidthPickupState,
)
from rescue_vision.app.near_field_grasp import (
    GraspSelection,
    NearFieldGraspPolicy,
)
from rescue_vision.config.near_field_grasp import NearFieldGraspConfig
from rescue_vision.geometry.types import FieldPoint, GroundPoint
from rescue_vision.camera.frame import CameraFrame
from rescue_vision.perception import TargetClass
from rescue_vision.world import TeamColor
from rescue_vision.tracking import TrackingConfig

from test_match import runtime_config
from test_near_field_grasp import selector, target


def _sequence(*, transports: int = 0, config=None, transport_half_width_mm: float = 34.0) -> MatchSequence:
    near_config = NearFieldGraspConfig()
    pickup = GripperWidthPickupSequence(
        gripper_full_travel_time_s=0.1,
        forward_speed_m_s=0.1,
        closed_servo_angles_deg=(90.0, 90.0),
        max_observation_age_ms=500.0,
        alignment_min_wheel_velocity_m_s=0.01,
    )
    sequence = MatchSequence(
        config or runtime_config(action_settle_time_s=0.0),
        tracker=TrackingConfig(1, 80.0, 0.1, 500.0, 1.0, 0.1).build_tracker(),
        gripper_full_travel_time_s=0.1,
        team_color=TeamColor.RED,
        initial_field_position=FieldPoint(0.0, 0.0),
        near_field_pickup=pickup,
        near_field_grasp_config=near_config,
        transport_corridor_half_width_mm=transport_half_width_mm,
    )
    sequence._transport_count = transports
    sequence.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
    sequence._near_field_session_id = 1
    # These unit tests exercise the pickup state machine directly; route
    # dispatch is covered by dedicated tests below.
    sequence._near_field_route = GraspRoute.DIRECT_NEAR
    return sequence


def _preparation(plan, *, session_id: int = 1, timestamp_ns: int = 0, ready: bool = True):
    return GraspPreparation(
        timestamp_ns,
        GraspSelection(plan, ()),
        plan.members if plan is not None else (),
        ready,
        checked_member_ids=None,
        session_id=session_id,
        confirmation_count=3 if ready else 0,
        confirmation_required=3,
        prepared_timestamp_ns=timestamp_ns,
        result_timestamp_ns=timestamp_ns,
    )


def test_formal_policy_is_single_green_then_supplies_or_single_injured() -> None:
    sequence = _sequence()
    first = sequence.near_field_policy
    assert first.allowed_classes == frozenset((TargetClass.GREEN_SUPPLY,))
    assert first.max_targets == 1

    sequence._transport_count = 1
    later = sequence.near_field_policy
    assert later.allowed_classes == frozenset(
        (
            TargetClass.GREEN_SUPPLY,
            TargetClass.BLACK_CORE,
            TargetClass.ORANGE_INJURED,
        )
    )
    assert later.max_targets == 3


def test_general_stage_accepts_only_isolated_orange_in_transport_corridor() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
    )
    sequence._latest_heading_rad = 0.0
    orange = target(
        i=1,
        x=300.0,
        y=0.0,
        cls=TargetClass.ORANGE_INJURED,
        timestamp=10,
        frame=1,
    ).observation
    sequence._tracker.update(10, [orange])
    found = sequence._find_opportunistic_single_green(10)
    assert found is not None and found.target_class is TargetClass.ORANGE_INJURED

    green = target(i=2, x=250.0, y=0.0, timestamp=20, frame=2).observation
    refreshed_orange = target(
        i=1,
        x=300.0,
        y=0.0,
        cls=TargetClass.ORANGE_INJURED,
        timestamp=20,
        frame=2,
    ).observation
    sequence._tracker.update(20, [refreshed_orange, green])
    orange_track = next(
        item
        for item in sequence._tracker.tracks
        if item.target_class is TargetClass.ORANGE_INJURED
    )
    assert sequence._transport_group_size(orange_track, 20) is None


def test_general_stage_orange_rejects_any_target_inside_fifty_mm_radius() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
    )
    orange = target(
        i=1,
        x=300.0,
        y=0.0,
        cls=TargetClass.ORANGE_INJURED,
        timestamp=10,
        frame=1,
    ).observation
    neighbor = target(
        i=2,
        x=350.0,
        y=0.0,
        cls=TargetClass.GREEN_SUPPLY,
        timestamp=20,
        frame=2,
    ).observation
    sequence._tracker.update(10, [orange])
    sequence._tracker.update(20, [
        target(
            i=1,
            x=300.0,
            y=0.0,
            cls=TargetClass.ORANGE_INJURED,
            timestamp=20,
            frame=2,
        ).observation,
        neighbor,
    ])
    orange_track = next(
        item
        for item in sequence._tracker.tracks
        if item.target_class is TargetClass.ORANGE_INJURED
    )

    assert sequence._transport_group_size(orange_track, 20) is None


def test_general_stage_orange_ignores_unknown_neighbor_ground_position() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
    )
    orange_10 = target(
        i=1,
        x=300.0,
        y=0.0,
        cls=TargetClass.ORANGE_INJURED,
        timestamp=10,
        frame=1,
    ).observation
    orange_20 = target(
        i=1,
        x=300.0,
        y=0.0,
        cls=TargetClass.ORANGE_INJURED,
        timestamp=20,
        frame=2,
    ).observation
    unknown_neighbor = replace(
        target(
            i=2,
            x=500.0,
            y=0.0,
            cls=TargetClass.GREEN_SUPPLY,
            timestamp=20,
            frame=2,
        ).observation,
        ground_point=None,
    )
    sequence._tracker.update(10, [orange_10])
    sequence._tracker.update(20, [orange_20, unknown_neighbor])
    orange_track = next(
        item
        for item in sequence._tracker.tracks
        if item.target_class is TargetClass.ORANGE_INJURED
    )

    assert sequence._transport_group_size(orange_track, 20) == 1


def test_near_field_handoff_uses_450mm_authoritative_range() -> None:
    sequence = _sequence()
    sequence._latest_heading_rad = 0.0
    sequence._green_reference = GroundPoint(300.0, 0.0)
    decision = sequence._step_align_green(0)
    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert decision.reason == "green_near_field_handoff_started"
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert sequence.near_field_session_id == 2


def test_already_near_target_skips_the_far_reference_collection() -> None:
    sequence = _sequence()
    sequence._latest_heading_rad = 0.0
    sequence._selected_track_id = 1
    sequence._tracker.update(
        0,
        (target(i=1, x=300.0, y=0.0, timestamp=0, frame=0).observation,),
    )

    decision = sequence._step_align_green(0)

    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert sequence._green_reference_samples == []
    assert sequence.near_field_session_id == 2


def test_approach_handoff_does_not_fall_into_legacy_preclose_close() -> None:
    sequence = _sequence()
    sequence._latest_heading_rad = 0.0
    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._green_approach_base_distance_m = 0.0
    sequence._green_approach_distance_m = 0.01
    sequence._green_reference = GroundPoint(460.0, 0.0)
    sequence._green_reference_heading_rad = 0.0

    decision = sequence._step_approach_green(10, 0.01)

    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert decision.reason == "green_near_field_handoff_started"
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert sequence.near_field_session_id == 2


def test_far_field_alignment_and_approach_keep_gripper_closed() -> None:
    sequence = _sequence()
    sequence._latest_heading_rad = 0.0
    sequence._green_reference = GroundPoint(700.0, 0.0)
    sequence._green_reference_heading_rad = 0.0
    sequence._green_reference_distance_m = 0.5

    sequence.state = MatchState.TRANSPORT_ALIGN_GREEN
    aligning = sequence._step_align_green(0)
    assert aligning.gripper_posture is GripperPosture.CLOSED
    assert aligning.gripper_angles_deg is None

    sequence.state = MatchState.TRANSPORT_APPROACH_GREEN
    sequence._green_approach_base_distance_m = 0.0
    approaching = sequence._step_approach_green(1, 0.0)
    assert approaching.gripper_posture is GripperPosture.CLOSED
    assert approaching.gripper_angles_deg is None


def test_near_field_confirmation_budget_starts_when_observation_window_opens() -> None:
    sequence = _sequence()
    sequence.state = MatchState.TRANSPORT_NEAR_FIELD_GRASP
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_confirmation_started_ns = None
    sequence._action_settle_phase = "near_field_grasp"
    sequence._action_settle_until_ns = 1_000_000_000

    assert not sequence.near_field_observation_window_open(600_000_000)
    assert sequence._near_field_confirmation_started_ns is None

    sequence._action_settle_phase = None
    sequence._action_settle_until_ns = None
    assert sequence.near_field_observation_window_open(1_000_000_000)
    assert sequence._near_field_confirmation_started_ns == 1_000_000_000
    assert sequence._near_field_route_decision(6_799_999_999, None) is None

    route = sequence._near_field_route_decision(7_000_000_000, None)
    assert route is not None and route.route is GraspRoute.RESELECT
    assert sequence.near_field_last_failure_diagnostic is not None
    assert "kind=confirmation_timeout_reselect" in (
        sequence.near_field_last_failure_diagnostic or ""
    )


def test_confirmation_timeout_reselects_instead_of_falling_into_breakup() -> None:
    sequence = _sequence()
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_confirmation_started_ns = 0
    preparation = GraspPreparation(
        0,
        GraspSelection(None, ("locked_members_missing",)),
        (),
        session_id=1,
    )

    decision = sequence._step_near_field_grasp(
        6_000_000_000,
        cumulative_distance_m=0.0,
        preparation=preparation,
        path_clear=True,
    )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "near_field_route:reselect:confirmation_timeout"
    assert sequence.near_field_route is GraspRoute.RESELECT


def test_stale_confirmed_plan_is_reselected_without_a_fake_fresh_timestamp() -> None:
    sequence = _sequence()
    plan = selector().select((target(),)).plan
    assert plan is not None
    sequence._near_field_confirmation_started_ns = 0
    stale = _preparation(plan, timestamp_ns=0, ready=True)

    decision = sequence._step_near_field_grasp(
        6_000_000_000,
        cumulative_distance_m=0.0,
        preparation=stale,
        path_clear=True,
    )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "near_field_route:reselect:confirmation_timeout"
    assert sequence.near_field_route is GraspRoute.RESELECT


def test_near_field_route_prefers_far_clear_target_after_blocked_near_target() -> None:
    sequence = _sequence(transports=1)
    decision_timestamp_ns = 800_000_000
    sequence._latest_heading_rad = 0.0
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_confirmation_started_ns = 0
    sequence._tracker.update(
        decision_timestamp_ns,
        (
            target(i=1, x=300.0, y=0.0, timestamp=decision_timestamp_ns, frame=1).observation,
            target(i=2, x=200.0, y=0.0, cls=TargetClass.BLUE_DANGER, timestamp=decision_timestamp_ns, frame=1).observation,
            target(i=3, x=700.0, y=300.0, timestamp=decision_timestamp_ns, frame=1).observation,
        ),
    )
    empty = GraspPreparation(
        decision_timestamp_ns,
        GraspSelection(None, ()),
        (),
        session_id=1,
    )
    late_timestamp_ns = 6_000_000_000
    sequence._tracker.update(
        late_timestamp_ns,
        (
            target(i=1, x=300.0, y=0.0, timestamp=late_timestamp_ns, frame=2).observation,
            target(i=2, x=200.0, y=0.0, cls=TargetClass.BLUE_DANGER, timestamp=late_timestamp_ns, frame=2).observation,
            target(i=3, x=700.0, y=300.0, timestamp=late_timestamp_ns, frame=2).observation,
        ),
    )
    route = sequence._near_field_route_decision(late_timestamp_ns, empty)
    assert route is not None
    assert route.route is GraspRoute.FAR_REAPPROACH
    assert route.target is not None
    assert route.target.ground_point is not None
    assert route.target.ground_point.x == pytest.approx(700.0)
    assert sequence._near_field_far_reapproach_used


def test_near_field_does_not_repeat_far_reapproach_after_one_failed_cycle() -> None:
    sequence = _sequence(transports=1)
    sequence._near_field_far_reapproach_used = True
    sequence._near_field_route = GraspRoute.DIRECT_NEAR

    decision = sequence._route_after_near_field_failure(800_000_000)

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason.startswith("near_field_route:reselect:")
    assert sequence.near_field_route is GraspRoute.RESELECT


def test_near_field_route_does_not_queue_unconfirmed_alternatives() -> None:
    sequence = _sequence(transports=1)
    first = selector().select((target(1, x=300, y=-100),)).plan
    second = selector().select((target(2, x=300, y=100),)).plan
    assert first is not None and second is not None
    sequence._near_field_route = GraspRoute.DECIDING
    preparation = GraspPreparation(
        0,
        GraspSelection(first, (), first),
        first.members + second.members,
        session_id=1,
    )

    route = sequence._near_field_route_decision(0, preparation)

    assert route is not None and route.route is GraspRoute.DIRECT_NEAR
    assert route.plan is first


def test_first_green_blocked_corridor_routes_directly_to_breakup() -> None:
    sequence = _sequence(transports=0)
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_confirmation_started_ns = 0
    nearest = target(1, x=300.0, y=0.0)
    far_green = target(2, x=700.0, y=0.0)
    blocker = target(3, x=200.0, y=0.0, cls=TargetClass.BLACK_CORE)
    preparation = GraspPreparation(
        800_000_000,
        GraspSelection(None, ("blocked_target:3:black_core",)),
        (),
        session_id=1,
    )

    decision = sequence._step_near_field_grasp(
        800_000_000,
        cumulative_distance_m=0.0,
        preparation=preparation,
        path_clear=True,
    )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "near_field_route:breakup"
    assert sequence.near_field_route is GraspRoute.BREAKUP
    assert sequence.selected_track_id is None


def test_first_green_blocked_corridor_does_not_select_a_farther_green() -> None:
    sequence = _sequence(
        config=runtime_config(opportunistic_single_green_enabled=True),
    )
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence._latest_heading_rad = 0.0
    sequence._tracker.update(
        10,
        (
            target(1, x=300.0, y=0.0, timestamp=10, frame=1).observation,
            target(2, x=700.0, y=0.0, timestamp=10, frame=1).observation,
            target(3, x=200.0, y=0.0, cls=TargetClass.BLACK_CORE, timestamp=10, frame=1).observation,
        ),
    )

    decision = sequence._step_search_cluster(10, 0.0)

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "near_field_route:breakup"
    assert sequence.near_field_route is GraspRoute.BREAKUP
    assert sequence.selected_track_id is None


def test_side_adjacent_incompatible_target_routes_to_breakup_without_reapproach() -> None:
    sequence = _sequence(transports=1)
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_confirmation_started_ns = 0
    rejection = (
        "side_adjacent_incompatible_single_green:"
        "track=2:class=blue_danger:dx_mm=12.0:dy_mm=55.0"
    )
    preparation = GraspPreparation(
        800_000_000,
        GraspSelection(None, (rejection,)),
        (),
        session_id=1,
    )

    decision = sequence._step_near_field_grasp(
        800_000_000,
        cumulative_distance_m=0.0,
        preparation=preparation,
        path_clear=True,
    )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "near_field_route:breakup"
    assert not sequence._near_field_far_reapproach_used


def test_near_field_route_prefers_safe_multi_target_plan() -> None:
    sequence = _sequence(transports=1)
    plan = selector(max_candidates=12).select(
        (target(1, x=300.0, y=-25.0), target(2, x=320.0, y=25.0)),
        policy=sequence.near_field_policy,
    ).plan
    assert plan is not None and len(plan.member_ids) == 2
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_confirmation_started_ns = 0
    preparation = GraspPreparation(
        0,
        GraspSelection(plan, ()),
        plan.members,
        session_id=1,
    )
    route = sequence._near_field_route_decision(0, preparation)
    assert route is not None
    assert route.route is GraspRoute.DIRECT_NEAR
    assert route.plan is not None and route.plan.member_ids == (1, 2)


def test_near_field_route_locks_the_selected_plan() -> None:
    sequence = _sequence(transports=1)
    sequence._near_field_route = GraspRoute.DECIDING
    selected_plan = selector().select((target(1, x=300.0, y=0.0),)).plan
    later_plan = selector().select((target(2, x=300.0, y=0.0),)).plan
    assert selected_plan is not None and later_plan is not None
    preparation = GraspPreparation(
        10,
        GraspSelection(later_plan, ()),
        later_plan.members,
        ready=False,
        checked_member_ids=None,
        session_id=1,
    )

    decision = sequence._step_near_field_grasp(
        10,
        cumulative_distance_m=0.0,
        preparation=preparation,
        path_clear=True,
    )

    assert decision.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert sequence.near_field_locked_ids == later_plan.member_ids


def test_near_field_route_without_near_or_far_target_reselects() -> None:
    sequence = _sequence(transports=1)
    sequence._near_field_route = GraspRoute.DECIDING
    sequence._near_field_confirmation_started_ns = 0
    empty = GraspPreparation(
        0,
        GraspSelection(None, ()),
        (),
        session_id=1,
    )
    route = sequence._near_field_route_decision(6_000_000_000, empty)
    assert route is not None and route.route is GraspRoute.RESELECT


def test_match_near_field_timeout_is_not_bypassed_by_missing_static_path() -> None:
    sequence = _sequence()
    plan = selector().select((target(),)).plan
    assert plan is not None
    sequence._near_field_pickup.state = GripperWidthPickupState.ALIGNING
    sequence._near_field_pickup._alignment_started_ns = 0

    decision = sequence._step_near_field_grasp(
        8_000_000_000,
        cumulative_distance_m=0.0,
        preparation=_preparation(plan),
        path_clear=None,
    )

    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason.startswith("near_field_route:reselect:")


def test_missing_static_path_wait_starts_near_field_alignment_deadline() -> None:
    sequence = _sequence()
    plan = selector().select((target(),)).plan
    assert plan is not None

    waiting = sequence._step_near_field_grasp(
        0,
        cumulative_distance_m=0.0,
        preparation=_preparation(plan),
        path_clear=None,
    )
    timed_out = sequence._step_near_field_grasp(
        8_000_000_000,
        cumulative_distance_m=0.0,
        preparation=_preparation(plan),
        path_clear=None,
    )

    assert waiting.reason == "near_field_waiting_for_static_path_validation"
    assert timed_out.state is MatchState.SEARCH_CLUSTER
    assert timed_out.reason.startswith("near_field_route:reselect:")


def test_opportunistic_green_uses_fixed_transport_corridor_k0() -> None:
    sequence = _sequence(
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    sequence._latest_heading_rad = 0.0
    green = target(i=1, x=500.0, y=0.0, timestamp=10, frame=1).observation
    # A nearby non-green target outside the fixed TRANSPORT corridor does not
    # invalidate the single-green opportunity; only K0 occupancy in the
    # forward corridor matters at this stage.
    black = target(
        i=2,
        x=500.0,
        y=80.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=10,
        frame=1,
    ).observation
    sequence._tracker.update(10, [green, black])
    found = sequence._find_opportunistic_single_green(10)
    assert found is not None and found.track_id == 1
    assert sequence.transport_corridor_half_width_mm == 34.0
    assert sequence.transport_corridor_effective_half_width_mm == 44.0


def test_single_green_side_adjacent_blue_is_rejected_before_near_field_approach() -> None:
    sequence = _sequence(
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    green = target(i=1, x=400.0, y=0.0, timestamp=10, frame=1).observation
    blue = target(
        i=2,
        x=400.0,
        y=70.0,
        cls=TargetClass.BLUE_DANGER,
        timestamp=10,
        frame=1,
    ).observation
    sequence._tracker.update(10, [green, blue])

    assert sequence._single_green_side_neighbor_requires_breakup(10)
    assert sequence._find_opportunistic_single_green(10) is None
    decision = sequence._step_search_cluster(10, 0.0)
    assert decision.state is MatchState.SEARCH_CLUSTER
    assert decision.reason == "near_field_route:breakup"


def test_single_green_side_adjacent_orange_is_not_rejected_before_approach() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    green = target(i=1, x=400.0, y=0.0, timestamp=10, frame=1).observation
    orange = target(
        i=2,
        x=400.0,
        y=70.0,
        cls=TargetClass.ORANGE_INJURED,
        timestamp=10,
        frame=1,
    ).observation
    sequence._tracker.update(10, [green, orange])

    assert not sequence._single_green_side_neighbor_requires_breakup(10)
    assert sequence._find_opportunistic_single_green(10) is not None
    decision = sequence._step_search_cluster(10, 0.0)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    # 绿块仍合法，但新的一般阶段优先级应选择同为近场的15分单橙。
    assert decision.reason == "near_field_group_preview:2"


def test_target_without_ground_point_does_not_block_fixed_transport_corridor() -> None:
    sequence = _sequence(
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    sequence._latest_heading_rad = 0.0
    green = target(i=1, x=500.0, y=0.0, timestamp=10, frame=1).observation
    unknown_position = replace(
        target(
            i=2,
            x=200.0,
            y=0.0,
            cls=TargetClass.BLUE_DANGER,
            timestamp=10,
            frame=1,
        ).observation,
        ground_point=None,
    )
    sequence._tracker.update(10, [green, unknown_position])

    found = sequence._find_opportunistic_single_green(10)

    assert found is not None and found.track_id == 1
    assert not sequence._first_green_forward_corridor_blocked(10)


def test_opportunistic_green_outside_transport_corridor_continues_cluster_logic() -> None:
    sequence = _sequence(
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    sequence._latest_heading_rad = 0.0
    observations = [
        target(i=1, x=500.0, y=-80.0, timestamp=10, frame=1).observation,
        target(i=2, x=500.0, y=80.0, timestamp=10, frame=1).observation,
    ]
    sequence._tracker.update(10, observations)
    decision = sequence._step_search_cluster(10, 0.0)
    assert decision.state is MatchState.ALIGN_CLUSTER_ONCE
    assert decision.reason == "cluster_seen_stop_collect_reference"


def test_second_green_in_fixed_transport_corridor_blocks_single_opportunity() -> None:
    sequence = _sequence(
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    sequence._latest_heading_rad = 0.0
    observations = [
        target(i=1, x=400.0, y=-20.0, timestamp=10, frame=1).observation,
        target(i=2, x=400.0, y=20.0, timestamp=10, frame=1).observation,
    ]
    sequence._tracker.update(10, observations)
    assert sequence._find_opportunistic_single_green(10) is None


def test_general_phase_prefers_adjacent_green_group_before_breakup() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence._latest_heading_rad = 0.0
    observations = [
        target(i=1, x=400.0, y=-20.0, timestamp=10, frame=1).observation,
        target(i=2, x=400.0, y=20.0, timestamp=10, frame=1).observation,
    ]
    sequence._tracker.update(10, observations)
    chosen = sequence._find_opportunistic_single_green(10)
    assert chosen is not None
    decision = sequence._step_search_cluster(10, 0.0)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.reason.startswith('green_path_clear_opportunistic_single:')


def test_general_phase_tries_shorter_x_anchor_when_fourth_supply_is_ahead() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    observations = [
        target(i=i, x=x, y=0.0, timestamp=10, frame=1).observation
        for i, x in enumerate((180.0, 240.0, 300.0, 360.0), start=1)
    ]
    sequence._tracker.update(10, observations)
    chosen = sequence._find_opportunistic_single_green(10)
    assert chosen is not None
    assert chosen.ground_point == GroundPoint(300.0, 0.0)


def test_general_phase_group_preview_prevents_off_corridor_pair_from_breakup() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
        transport_half_width_mm=34.0,
    )
    sequence.state = MatchState.SEARCH_CLUSTER
    sequence._latest_heading_rad = 0.0
    observations = [
        target(i=1, x=420.0, y=120.0, timestamp=10, frame=1).observation,
        target(i=2, x=430.0, y=180.0, timestamp=10, frame=1).observation,
    ]
    sequence._tracker.update(10, observations)
    assert sequence._find_opportunistic_single_green(10) is None
    decision = sequence._step_search_cluster(10, 0.0)
    assert decision.state is MatchState.TRANSPORT_ALIGN_GREEN
    assert decision.reason == "near_field_group_preview:1"
    assert sequence._near_field_group_preview


def test_general_phase_fast_entry_accepts_single_black_core() -> None:
    sequence = _sequence(
        transports=1,
        config=runtime_config(opportunistic_single_green_enabled=True),
    )
    sequence._latest_heading_rad = 0.0
    black = target(
        i=1,
        x=400.0,
        y=0.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=10,
        frame=1,
    ).observation
    sequence._tracker.update(10, (black,))
    chosen = sequence._find_opportunistic_single_green(10)
    assert chosen is not None
    assert chosen.target_class is TargetClass.BLACK_CORE


def test_first_phase_fast_entry_rejects_single_black_core() -> None:
    sequence = _sequence(
        transports=0,
        config=runtime_config(opportunistic_single_green_enabled=True),
    )
    sequence._latest_heading_rad = 0.0
    black = target(
        i=1,
        x=400.0,
        y=0.0,
        cls=TargetClass.BLACK_CORE,
        timestamp=10,
        frame=1,
    ).observation
    sequence._tracker.update(10, (black,))
    assert sequence._find_opportunistic_single_green(10) is None


def test_near_field_session_ignores_old_preparation_and_emits_dynamic_angle() -> None:
    sequence = _sequence()
    plan = selector().select((target(),), policy=sequence.near_field_policy).plan
    assert plan is not None
    stale = _preparation(plan, session_id=0)
    ignored = sequence._step_near_field_grasp(0, 0.0, stale, True)
    assert ignored.reason == "near_field_search:waiting_eligible_group"

    opening = sequence._step_near_field_grasp(
        1, 0.0, _preparation(plan), True
    )
    assert opening.state is MatchState.TRANSPORT_NEAR_FIELD_GRASP
    assert opening.gripper_angles_deg == plan.opening_servo_angles_deg
    assert opening.gripper_posture.value == "transport"
    assert opening.soft_brake
    assert sequence.near_field_active_plan is not None


def test_near_field_alignment_keeps_transport_posture_without_full_open() -> None:
    sequence = _sequence()
    decision = sequence._near_field_decision(
        0,
        GripperWidthPickupDecision(
            0,
            GripperWidthPickupState.ALIGNING,
            0.0,
            0.1,
            None,
            False,
            "align_group_envelope",
            min_wheel_velocity_m_s=0.01,
        ),
    )
    assert decision.gripper_posture is GripperPosture.CLOSED
    assert decision.gripper_angles_deg is None
    assert decision.min_wheel_velocity_m_s == 0.01


def test_near_field_planning_corridor_is_drawn_on_matching_frame() -> None:
    grasp_selector = selector()
    plan = grasp_selector.select((target(),)).plan
    assert plan is not None
    preparation = _preparation(plan)
    frame = CameraFrame(0, 0, np.zeros((1000, 1000, 3), dtype=np.uint8))
    rendered = _overlay_near_field_corridor(
        frame,
        preparation,
        grasp_selector,
    )
    assert rendered is not None
    assert rendered.metadata["near_field_corridor"] == "CORRIDOR READY"
    assert rendered.metadata["near_field_selected_ids"] == "1"
    assert np.any(rendered.image_bgr != frame.image_bgr)


def test_match_preview_highlights_selected_far_field_candidate_only() -> None:
    sequence = _sequence()
    first = target(i=1, x=300.0, y=0.0, timestamp=0, frame=0).observation
    second = target(i=2, x=600.0, y=200.0, timestamp=0, frame=0).observation
    sequence._tracker.update(0, (first, second))
    sequence._selected_track_id = 1
    frame = CameraFrame(0, 0, np.zeros((1000, 1000, 3), dtype=np.uint8))

    rendered = _overlay_match_selected_targets(frame, sequence)

    assert rendered is not None
    assert rendered.metadata["match_selected_ids"] == "1"
    assert np.any(rendered.image_bgr[475:525, 475:525] == (0, 165, 255))
    assert not np.any(rendered.image_bgr[175:225, 275:325] != 0)


def test_match_preview_highlights_all_members_of_near_field_selected_group() -> None:
    grasp_selector = selector()
    plan = grasp_selector.select((target(i=1), target(i=2, y=40.0))).plan
    assert plan is not None
    preparation = _preparation(plan)
    sequence = _sequence()
    frame = CameraFrame(0, 0, np.zeros((1000, 1000, 3), dtype=np.uint8))

    rendered = _overlay_match_selected_targets(frame, sequence, preparation)

    assert rendered is not None
    assert rendered.metadata["match_selected_ids"] == "1,2"
    assert np.any(rendered.image_bgr[475:525, 475:525] == (0, 165, 255))
    assert np.any(rendered.image_bgr[435:485, 435:485] == (0, 165, 255))


def test_match_preview_highlights_the_selected_far_field_cluster() -> None:
    sequence = _sequence()
    sequence._latest_heading_rad = 0.0
    first = target(i=1, x=400.0, y=-20.0, timestamp=0, frame=0).observation
    second = target(i=2, x=420.0, y=20.0, timestamp=0, frame=0).observation
    sequence._tracker.update(0, (first, second))

    measurement = sequence._cluster_ground_measurement(0)
    assert measurement is not None
    assert sequence.preview_selected_track_ids == (1, 2)
    frame = CameraFrame(0, 0, np.zeros((1000, 1000, 3), dtype=np.uint8))

    rendered = _overlay_match_selected_targets(frame, sequence)

    assert rendered is not None
    assert rendered.metadata["match_selected_ids"] == "1,2"


def test_match_preview_uses_near_field_worker_ids_for_current_frames() -> None:
    grasp_selector = selector()
    plan = grasp_selector.select((target(i=7, timestamp=0, frame=0),)).plan
    assert plan is not None
    current_target = target(i=7, x=250.0, timestamp=1, frame=1)
    preparation = GraspPreparation(
        1,
        GraspSelection(plan, ()),
        (current_target,),
        True,
        session_id=1,
    )
    sequence = _sequence()
    assert sequence._near_field_pickup is not None
    sequence._near_field_pickup.active_plan = plan
    frame = CameraFrame(1, 1, np.zeros((1000, 1000, 3), dtype=np.uint8))

    rendered = _overlay_match_selected_targets(frame, sequence, preparation)

    assert rendered is not None
    assert rendered.metadata["match_selected_ids"] == "7"
    assert np.any(rendered.image_bgr[525:575, 475:525] == (0, 165, 255))


def test_near_field_path_block_returns_to_formal_search_before_opening() -> None:
    sequence = _sequence()
    plan = selector().select((target(),), policy=sequence.near_field_policy).plan
    assert plan is not None
    blocked = sequence._step_near_field_grasp(
        0, 0.0, _preparation(plan), False
    )
    assert blocked.state is MatchState.SEARCH_CLUSTER
    assert blocked.reason == "near_field_route:reselect:static_path_blocked"
    assert not sequence._breakup_only
    assert "field_position=" in sequence._near_field_last_failure_diagnostic
    assert sequence.near_field_active_plan is None


def test_approach_seed_checks_same_field_boundary_as_near_field(monkeypatch):
    sequence = _sequence(transports=1)
    sequence._latest_heading_rad = 0.0
    monkeypatch.setattr(sequence, '_safe_zone_path_blocked', lambda *args: False)
    assert not sequence._candidate_path_blocked(GroundPoint(300, 0), breakup=False)
    assert sequence._candidate_path_blocked(GroundPoint(1800, 0), breakup=False)


def test_near_field_complete_enters_existing_d1_transport() -> None:
    sequence = _sequence()
    plan = selector().select((target(),), policy=sequence.near_field_policy).plan
    assert plan is not None
    prep = _preparation(plan)
    opening = sequence._step_near_field_grasp(0, 0.0, prep, True)
    assert opening.gripper_angles_deg == plan.opening_servo_angles_deg
    forward = sequence._step_near_field_grasp(100_000_001, 0.0, prep, True)
    assert sequence._near_field_pickup is not None
    assert sequence._near_field_pickup.state is GripperWidthPickupState.FORWARD
    assert forward.linear_velocity_m_s > 0.0
    assert forward.gripper_angles_deg == plan.opening_servo_angles_deg
    closing = sequence._step_near_field_grasp(
        200_000_001,
        plan.forward_distance_mm / 1000.0,
        replace(
            prep,
            capture_timestamp_ns=200_000_001,
            prepared_timestamp_ns=200_000_001,
            result_timestamp_ns=200_000_001,
        ),
        True,
    )
    assert closing.gripper_angles_deg == (90.0, 90.0)
    complete = sequence._step_near_field_grasp(
        300_000_001,
        plan.forward_distance_mm / 1000.0,
        replace(
            prep,
            capture_timestamp_ns=300_000_001,
            prepared_timestamp_ns=300_000_001,
            result_timestamp_ns=300_000_001,
        ),
        True,
    )
    assert complete.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert complete.reason == "near_field_grasp_complete_start_safe_zone_d1_line"


def test_single_orange_completion_routes_to_injured_zone_endpoint() -> None:
    injured_endpoint = FieldPoint(175.0, 1137.0)
    sequence = _sequence(
        transports=1,
        config=runtime_config(safe_zone_injured_target_field=injured_endpoint),
    )
    plan = selector().select(
        (target(cls=TargetClass.ORANGE_INJURED),),
        policy=sequence.near_field_policy,
    ).plan
    assert plan is not None
    prep = _preparation(plan)

    sequence._step_near_field_grasp(0, 0.0, prep, True)
    sequence._step_near_field_grasp(100_000_001, 0.0, prep, True)
    sequence._step_near_field_grasp(
        200_000_001,
        plan.forward_distance_mm / 1000.0,
        replace(
            prep,
            capture_timestamp_ns=200_000_001,
            prepared_timestamp_ns=200_000_001,
            result_timestamp_ns=200_000_001,
        ),
        True,
    )
    complete = sequence._step_near_field_grasp(
        300_000_001,
        plan.forward_distance_mm / 1000.0,
        replace(
            prep,
            capture_timestamp_ns=300_000_001,
            prepared_timestamp_ns=300_000_001,
            result_timestamp_ns=300_000_001,
        ),
        True,
    )

    assert complete.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert sequence._safe_zone_transport_endpoint() == injured_endpoint


def test_delayed_stationary_plan_hands_off_to_unchanged_transport():
    from test_gripper_width_sequence import motion_sample
    sequence = _sequence()
    capture, now = 100_000_000, 450_000_000
    for stamp in range(0, now + 1, 10_000_000):
        sequence.observe_grasp_motion(motion_sample(stamp))
    plan = selector().select((target(timestamp=capture, frame=1),), policy=sequence.near_field_policy).plan
    prepared = replace(_preparation(plan, timestamp_ns=capture), prepared_timestamp_ns=now-20_000_000)
    opening = sequence._step_near_field_grasp(now, 0.0, prepared, True)
    assert opening.gripper_angles_deg == plan.opening_servo_angles_deg
    forward = sequence._step_near_field_grasp(now+100_000_001, 0.0, None, True)
    assert forward.linear_velocity_m_s > 0
    closing = sequence._step_near_field_grasp(now+200_000_001, plan.forward_distance_mm/1000, None, True)
    assert closing.gripper_angles_deg == (90.0, 90.0)
    complete = sequence._step_near_field_grasp(now+300_000_001, plan.forward_distance_mm/1000, None, True)
    assert complete.state is MatchState.TRANSPORT_ALIGN_RED_ZONE
    assert complete.reason == 'near_field_grasp_complete_start_safe_zone_d1_line'


def test_1930_search_prefers_near_orange_and_rejects_blocked_black_seed():
    sequence = _sequence(transports=1, config=runtime_config(opportunistic_single_green_enabled=True))
    sequence._near_field_grasp_config = replace(sequence._near_field_grasp_config, orange_isolation_radius_mm=100)
    sequence._latest_heading_rad = 0.0
    # 日志183066ms K0布局：近橙12、侧后绿13、远黑3及其路径中的蓝1。
    objects = [target(12,x=312.6,y=-144.5,cls=TargetClass.ORANGE_INJURED),
               target(13,x=378.4,y=-94.2), target(3,x=700,y=311.2,cls=TargetClass.BLACK_CORE),
               target(1,x=434.1,y=205.8,cls=TargetClass.BLUE_DANGER)]
    sequence._tracker.update(0, [replace(t.observation, ground_point=GroundPoint(801.1,311.2))
                                 if t.track_id == 3 else t.observation for t in objects])
    found = sequence._find_approach_seed(0)
    assert found is not None and found.target_class is TargetClass.ORANGE_INJURED
    sequence.state = MatchState.SEARCH_CLUSTER
    decision = sequence._step_search_cluster(0, 0.0)
    assert decision.selected_track_id == found.track_id
    assert sequence.state is MatchState.TRANSPORT_ALIGN_GREEN


def test_approach_seed_does_not_repeat_known_blocked_preview():
    sequence = _sequence(transports=1)
    sequence._latest_heading_rad = 0.0
    for frame in range(12):
        stamp = frame * 100_000_000
        sequence._tracker.update(stamp, [target(1,x=700,y=311*700/801,cls=TargetClass.BLACK_CORE,
            timestamp=stamp,frame=frame).observation, target(2,x=434,y=205,cls=TargetClass.BLUE_DANGER,
            timestamp=stamp,frame=frame).observation])
        assert sequence._find_approach_seed(stamp) is None


def test_formal_far_alignment_uses_confirmed_track_without_ten_frame_wait():
    sequence = _sequence(transports=1, config=runtime_config(action_settle_time_s=0))
    sequence._latest_heading_rad = 0.0
    sequence._tracker.update(10, [target(x=650,y=100,timestamp=10,frame=1).observation])
    seed = sequence._find_approach_seed(10)
    assert seed is not None
    sequence._begin_green_transport(10, seed, group_preview=True)
    decision = sequence._step_align_green(10)
    assert sequence._green_reference is not None
    assert decision.reason != 'green_collecting_10_point_reference'
    assert decision.angular_velocity_rad_s > 0


def test_reapproach_rejects_same_danger_path_as_search_seed():
    sequence = _sequence(transports=1)
    sequence._latest_heading_rad = 0.0
    sequence._tracker.update(10, [target(x=650,y=100,timestamp=10,frame=1).observation,
                                target(2,x=350,y=50,cls=TargetClass.BLUE_DANGER,timestamp=10,frame=1).observation])
    assert sequence._find_approach_seed(10) is None
    assert sequence._find_far_reapproach_target(10) is None
